# Laya MLX performance research

Research date: 2026-09-19. Benchmark snapshot: commit `49daed6` from that date. Target: Apple M3 Max, 40 GPU cores, 128 GiB unified memory, MLX/MLX Metal 0.32.2. This is a static review of the native runtime, installed MLX implementation, official documentation, and existing benchmark JSON. **No GPU benchmark or model inference was run for this research. None of the proposed optimizations below has a measured speedup in this report.**

The first experiments should be whole-model compilation and representative batch scheduling, followed by selective quantized matrix multiplication. These address the dominant repeated work. A specialized local-attention kernel is a credible longer-term project for long inputs. Exact pruning of the last decision-head layer is feasible, but its whole-model arithmetic saving is only a few percent. Large improvements without changing the checkpoint will require improving the dense backbone, eliminating genuinely redundant requests, or finding a measured implementation bottleneck; simply replacing an activation or enabling another attention flag is unlikely to suffice.

## What the existing measurements establish

The following are existing **end-to-end median latencies**, including prompt preparation and result formatting, with synchronized GPU completion, five warmups, and 50 measured iterations. Model loading and downloads are excluded. The benchmark allows a batch of 64 questions; the public runtime default is 16, so the 50-question result is not the default API configuration.

| Checkpoint / precision | Short 1 question | Short 10 questions | Short 50 questions | Long 1 question | Long 10 questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya MLX FP16 | 13.421 ms | 71.068 ms | 336.030 ms | 44.927 ms | 420.987 ms |
| Laya MLX FP32 | 15.954 ms | 98.820 ms | 450.712 ms | 61.331 ms | 534.242 ms |
| Laya stock Torch MPS FP32 | 24.918 ms | 95.265 ms | 497.856 ms | 65.581 ms | 586.594 ms |
| Multilingual MLX FP16 | 7.390 ms | 27.386 ms | 127.565 ms | 37.635 ms | 389.487 ms |
| Multilingual MLX FP32 | 7.988 ms | 32.337 ms | 151.387 ms | 47.208 ms | 451.331 ms |
| Multilingual stock Torch MPS FP32 | 19.349 ms | 43.158 ms | 194.171 ms | 52.939 ms | 534.492 ms |

Sources at the research snapshot: [Laya FP16](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-mlx-float16.json), [Laya FP32](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-mlx-float32.json), [Laya MPS](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-torch-mps-float32.json), [multilingual FP16](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-multilingual-mlx-float16.json), [multilingual FP32](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-multilingual-mlx-float32.json), and [multilingual MPS](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-multilingual-torch-mps-float32.json). Long inputs contain 512 tokens for Laya and 1024 for multilingual; comparing their long-input latencies is therefore not a comparison at equal sequence length. Short padded lengths are 93 and 91, respectively. Torch comparisons must retain the FP32 label: they combine a backend change with a precision change when compared against MLX FP16.

There is meaningful run variability. For example, the multilingual FP16 long 10-question run has p50 389.487 ms, p95 462.319 ms, and maximum 619.663 ms. Its short single-question forward median is 8.023 ms while the independently measured end-to-end median is 7.390 ms. Subtracting those medians would produce a nonsensical negative preprocessing time. The current files do **not** isolate tokenizer, Python dispatch, individual GPU kernels, or synchronization costs. They establish useful baselines, not a kernel-level bottleneck diagnosis.

The [validation reports](../benchmarks/results/) record 63/63 argmax agreement for each of three checkpoints in FP32 and FP16, and 100 finite, deterministic repeated calls per variant: 378/378 answer agreements and 600 repeated calls altogether. These are regression checks over a small fixture corpus, including repeated questions. They are not evidence that future quantization or architecture changes preserve general task accuracy.

## Why dense matrix multiplication deserves priority

The [model implementation](../laya_mlx/model.py) applies QKV and output projections, a gated encoder MLP, and two conventional transformer decision-head layers to every padded token. Let `D` be hidden size, `I` encoder intermediate size, `N` encoder layer count, and `H` decision-head layer count. The number of matrix weights used per token in these blocks is:

```text
A = N * (4 * D^2 + 3 * D * I) + H * 12 * D^2
dense FLOPs per batch ~= 2 * B * L * A
dense attention FLOPs ~= 4 * B * (N + H) * L^2 * D
```

These estimates count multiply and add separately and exclude normalization, activations, embeddings, scoring, masking, memory movement, and kernel overhead. They are an arithmetic model, not a runtime profile.

| Checkpoint family | D / I / encoder layers | Token embedding weights | Main token-wise matrix weights A | Encoder global / local layers |
| --- | --- | ---: | ---: | --- |
| Laya / typed decisions | 1024 / 2624 / 28 | 51,576,832 | 368,312,320 | 10 / 18 |
| Multilingual | 768 / 1152 / 22 | 196,608,000 | 124,452,864 | 8 / 14 |

The multilingual checkpoint's approximately 322 million total parameters include 196.6 million embedding parameters. Only selected embedding rows are gathered for inference; this is not a full vocabulary projection. Its main per-token matrix workload is approximately one third of the English model's, despite their total parameter counts appearing much closer. This is consistent with the measured short-batch latency gap, although it does not prove a particular hardware bottleneck. Embedding-only quantization would predominantly reduce resident weight size, especially for multilingual; it need not improve inference latency.

With the current dense attention path, attention products account for about 1.5% of modeled FLOPs at L=93 for Laya, 7.9% at L=512 for Laya, and 23.3% at L=1024 for multilingual. Their fraction of **wall time** can differ substantially. A profiler should distinguish matrix kernels, attention kernels, elementwise kernels, CPU graph construction, and idle gaps before committing to custom Metal work.

## Prioritized experiments

| Priority | Experiment | Best target | Main tradeoff / acceptance condition |
| --- | --- | --- | --- |
| P0 | Profile one short and one long shape; compile the model or encoder blocks | Single-request latency and Python dispatch | Keep identical outputs within the existing precision tolerance; measure first-use compilation separately |
| P0 | Tune batching by actual token budget and length distribution | Many distinct questions and mixed-length traffic | Optimize throughput subject to a p95 latency and memory budget; account for queueing |
| P1 | Quantize selected backbone linear layers, starting at 8-bit then 4-bit | Weight traffic and potentially dense inference | Quality and calibration gates; actual M3 Max speed can regress |
| P1 | Cache shared state tokenization and stable question templates | Many questions sharing a state or repeated rubrics | Exact token identity; bounded cache; separate cached and uncached results |
| P1 | Compute only required outputs of the final head layer | Every workload, especially longer sequences | Exact dependency pruning; modest whole-model arithmetic saving |
| P2 | Genuine local-window attention with tile bounds | 512/1024-token workloads | New kernel complexity; preserve bidirectional window and padding semantics |
| P2 | Fuse residual/norm or GELU/gate only where the profile warrants it | Small-kernel overhead or activation traffic | Existing fast kernels already cover much of this; keep exact GELU semantics |
| Separate product feature | Deduplicate identical forward inputs | Workloads that actually repeat questions | Report unique inference count and cache hits; do not present it as a general kernel speedup |

### Compile the complete evaluated inference path

`Agent.forward()` currently constructs arrays, calls `self.model`, and evaluates the result. There is no enclosing `mx.compile` at model or block level. Fixed-shape compilation can reduce Python graph construction and fuse supported operations. MLX documents shape specialization and explicit state capture; it also warns that shapeless compilation cannot safely preserve arbitrary shape-dependent Python operations. See the [official compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

Start with a compiled callable created **after loading, casting, and evaluating weights**, using normal shape specialization. Keep the existing uncompiled callable for parity comparison and CPU compatibility. For a frozen inference model, weights can remain captured for that model instance; if weights or module structure change, rebuild the callable or explicitly capture the relevant state. Do not reuse a compiled closure across checkpoint replacements.

Test a whole-model wrapper and, if tracing limitations or compilation cost make that unattractive, compile encoder blocks and the head separately. The current code reads `x.shape` into Python integers, reshapes with explicit batch/length values, creates `arange(length)` masks, and indexes markers using a shape-derived row range. Applying `shapeless=True` to this complete graph without redesign is unsafe. Moving dynamic mask construction outside a compiled block and using shape-independent flatten/unflatten operations may enable a later shapeless variant; verify it across changed B, L, and marker counts.

Shape buckets can limit retracing, but padding has a compute cost. Padding L=93 to 96 adds approximately 3.2% token-wise work; padding it to 128 adds approximately 37.6%. Compare exact-shape compilation with small length multiples and a bounded set of workload-informed buckets. Include `(batch size, padded length, marker slots, dtype, device/model instance)` in cache policy decisions, and measure cold compilation latency and retained memory under shape churn.

The standalone `forward` benchmark in [worker.py](../benchmarks/worker.py) calls `agent.model` directly. If compilation is added only in `Agent.forward`, the current forward benchmark would bypass it while the end-to-end benchmark would use it. Both paths must explicitly select the same candidate implementation for a meaningful comparison. Keep `mx.eval` and GPU synchronization in the timing procedure: measuring graph construction alone would not measure inference.

### Batch by useful tokens, then inspect matrix scheduling

[collate_items](../laya_mlx/agent.py) right-pads each chunk to its longest sequence; the runtime groups questions in insertion order. For heterogeneous traffic, sort or bucket by prepared length, use a token budget in addition to a question-count ceiling, and restore original question IDs and output order. Compare batches of 1, 2, 4, 8, 16, 32, and 64 only where representative of the service. For online requests, include time waiting for a batch; offline questions/second alone can hide unacceptable latency.

The current short 50-question workload wastes approximately 8.9% of padded tokens for Laya and 5.8% for multilingual. Long benchmark rows have no padding waste. Consequently, unpadding or sorting alone has limited arithmetic upside on **these** fixtures. A mixed-length distribution, including one long question among many short ones, is necessary to reveal the production benefit. Padding removal must preserve per-example RoPE positions, marker positions, and attention boundaries; concatenating examples into one sequence without an isolation mask changes the model.

For dense kernels, inspect actual shapes and strides. QKV is already a single projection, and the two encoder MLP input branches already share a projection. Splitting those indiscriminately would add launches. Compare explicit flattening of contiguous `[B,L,D]` input into `[B*L,D]` only if the profiler or MLX dispatch trace shows undesirable batched GEMMs; the framework may already flatten efficiently. Source inspection alone does not justify claiming a missed GEMM optimization.

Do not insert a synchronization after every layer in the production implementation. The present runtime evaluates once per chunk. Extra waits could remove CPU/GPU overlap and obscure a scheduling improvement; layer-level profiling should be a separate diagnostic run.

### Quantization: target the backbone and implement its storage contract

The installed MLX 0.32.2 [quantized layer implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py) provides `nn.quantize(..., class_predicate=...)` and `QuantizedLinear`, with weight-only matrix multiplication through `mx.quantized_matmul`. Grouped affine quantization supports 8-bit and 4-bit experiments. Start with encoder linear layers at group size 64, retaining activations, norms, type embeddings, scorer, and action head in FP16. Then independently add decision-head linear layers and optionally embedding quantization. Measure each variant; weight-only kernels can lose to FP16 GEMMs at high token counts.

There are concrete integration hazards in the present loader and model:

1. `Agent.__init__` casts **every** stored weight to a floating dtype and instantiates only dense modules before strict loading. A quantized checkpoint requires metadata describing the selected modules, group size, bit width, and mode; instantiate matching quantized modules before loading and preserve packed integer weights. A floating cast of packed weights is not valid loading.
2. The first action-head linear layer has input width `D+4`, namely 1028 or 772, which is not divisible by an affine group size of 32, 64, or 128. A blanket quantization call is therefore unsuitable. Check every selected layer's input width before conversion.
3. `DecisionModel.__call__` chooses the action-head input dtype from `self.act_head.layers[0].weight.dtype`. For a quantized layer that weight would be packed integer storage, not the desired activation dtype. Excluding the action head avoids this path initially; supporting it later requires an explicit activation-dtype contract.
4. Quantization changes logits and calibrated probabilities. The existing small FP16 fixture agreement is insufficient evidence for 4-bit quality. Use held-out labeled choice, score, and noul tasks, multilingual inputs, close decisions, different option counts, and escalation examples. Track argmax agreement, task accuracy, score error, probability drift, calibration, and action probabilities. Saturated action outputs can hide large action-logit changes.

For FP16 affine scales and offsets with group size 64, the approximate matrix storage is `bits/8 + 4/64` bytes per parameter: 1.0625 bytes at 8-bit and 0.5625 bytes at 4-bit, compared with 2 bytes at FP16. These are storage estimates for quantized matrices, excluding other tensors and packaging overhead; they are not speedup estimates. The [official quantize API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) describes group divisibility and formats.

Newer low-bit formats should be evaluated against the actual M3 Max backend, not assumed to use hardware from later Apple chips. MLX's NAX availability check requires a newer architecture generation than the recorded `applegpu_g15s` device. See the [MLX 0.32.2 device check](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L947).

### Reuse CPU preparation where the inputs are actually identical

[build_sequence](../laya_mlx/common.py) serializes, sanitizes, and tokenizes the same state separately for every question. It also separately tokenizes each instruction and option. The Rust tokenizer is already used directly; replacing Transformers tokenization is not a remaining optimization.

Serialize and sanitize the state once per `prepare` call, encode it once, and slice its token IDs to each question's available room. Cache immutable prepared question prefixes when the same rubric is used across states, with keys that include tokenizer identity/revision, question type, ordered criteria, instruction serialization, special-token sanitation, and token budgets. Bounded caches must not reuse results after tokenizer or configuration changes. Batched tokenizer encoding is another experiment, provided its output exactly matches the current sequence of independent encodes.

Do not tokenize a newly concatenated prompt as a substitute for concatenating independently encoded pieces: subword boundaries can change. Verify byte-for-byte input IDs, attention masks, marker positions, qtypes, truncation behavior, structured criteria, mask literals, empty inputs, and output mapping.

For the long benchmark, the state repeats a sentence 200 times before truncation. Avoiding N repeated state encodes could help CPU preparation, but the existing end-to-end/forward timing difference does not measure that saving. Benchmark `prepare`, collate/array construction, forward, and postprocessing independently, then confirm the end-to-end result with a nonrepeating held-out corpus.

**A decoder KV cache does not apply to this encoder.** Its first layer is global bidirectional attention; state-token representations depend on the question, options, and their positions. Reusing state hidden states or K/V across different questions changes results. Tokenization and identical whole-input results can be cached; arbitrary contextual encoder state cannot.

### Exactly prune the final decision-head outputs

After the last `HeadLayer`, only the `[CLS]` token and option-marker tokens are consumed. Earlier head layers must still produce all tokens, because the final layer reads their K/V. In the final layer alone:

1. Normalize all input tokens and compute all K/V.
2. Gather Q at `[CLS]` and valid option positions and run those queries against the complete masked K/V sequence.
3. Apply the output projection, residual, second norm, and feed-forward network only to those selected positions.
4. Use the selected `[CLS]` output for the action head and selected marker outputs for scoring; preserve marker padding and original order.

A first implementation can retain the fused full QKV projection and gather Q afterward. A more aggressive variant splits its weights into a full-length KV projection and selected-token Q projection. That saves more arithmetic but may make GEMM scheduling less efficient. Duplicate padded marker indices are harmless only if masked results remain unobservable. The 1-option, many-option, and variable-marker cases need explicit parity checks. Since fewer queries may select a different SDPA kernel, mathematical equivalence does not imply bitwise identical floating-point results.

With `R = 1 + number of option slots`, retaining full QKV removes approximately `18 * B * (L-R) * D^2` dense FLOPs and `4 * B * L * (L-R) * D` attention FLOPs from the last head layer. Splitting Q/KV changes the dense coefficient from 18 to 20. Relative to the whole-model arithmetic estimate above, using R=5 gives:

| Checkpoint / length | Keep fused full QKV | Also compute only selected Q |
| --- | ---: | ---: |
| Laya, L=93 | 2.44% | 2.70% |
| Laya, L=512 | 2.60% | 2.86% |
| Typed decisions, L=1024 | 2.66% | 2.90% |
| Multilingual, L=93 | 4.03% | 4.47% |
| Multilingual, L=1024 | 4.22% | 4.58% |

These are static FLOP reductions, not predicted latency reductions. The technique removes most work from one head layer, not most work from the model. It is useful because it preserves dependencies and is implementable without retraining, not because it promises a multiple of whole-model speed.

### Build genuinely local attention only after measuring its contribution

All encoder attention calls already use `mx.fast.scaled_dot_product_attention`; RoPE is already `mx.fast.rope`; `nn.LayerNorm` calls the fast normalization primitive. MLX's [attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) accepts boolean masks and performs softmax in FP32. The current head dimension is 64. The [MLX 0.32.2 Metal dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp#L633) supports this shape with array masks and does not select the unfused fallback for it during inference. **There is no evidence that Laya's boolean mask disables fused attention.** `force_fused=True`, available in the installed version, is useful as a diagnostic assertion but should not be advertised as a new fast path here.

The remaining limitation is structured sparsity. The implementation builds a dense local boolean mask of shape `[B,1,L,L]`. In the conventional [Metal attention kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L236), the noncausal loop traverses the full KV tile range; the array mask is applied to scores after QK multiplication. It preserves local-attention semantics without exploiting a local tile range.

An exact specialized kernel can limit each query tile to the overlapping K/V window, keep FP32 softmax accumulation, and avoid a dense L-by-L mask. The correct window is **bidirectional and inclusive**: `abs(query_position - key_position) <= 64`. Interior queries can see 129 positions, despite the configuration name `local_attention=128`. Full-attention layers and both decision-head layers must remain global. Padded keys must remain excluded, and unused padded queries need defined finite behavior.

A lower-effort prototype can group query blocks with overlapping K/V slices and call the existing SDPA with a smaller exact mask. Use already-positioned RoPE Q/K, or explicitly retain absolute offsets. Prefer batched blocks over many Python calls, and account for duplicated K/V materialization. This prototype may lose to the current kernel at short lengths; it is a correctness and break-even experiment before maintaining custom Metal.

The maximum reduction in modeled **whole-model** FLOPs from removing all forbidden local attention pairs is small for short inputs and more promising for long ones:

| Checkpoint / length | Ideal total FLOP reduction from exact local sparsity |
| --- | ---: |
| Laya, L=93 | 0.086% |
| Laya, L=512 | 3.61% |
| Typed decisions, L=1024 | 7.69% |
| Multilingual, L=93 | 0.147% |
| Multilingual, L=1024 | 11.92% |

The estimates use `local_pairs = L*(2*r+1) - r*(r+1)` for L>r, with r=64. They include all dense projections and both full-attention decision-head layers. They exclude mask generation and memory traffic. Runtime benefit can exceed or fall below the FLOP fraction because attention and GEMMs have different efficiency; only profiling can establish it. At 8192 tokens the tradeoff would be different, but the shipped agents limit inputs to 512 or 1024, so an 8192-token claim would require a separately supported workload.

### Fusion beyond compilation

The installed exact `nn.gelu` is already decorated with shapeless compilation, and `nn.Linear` already uses a bias-aware `addmm` when appropriate. Whole-block compilation may still fuse the GELU with its gate multiplication, residual additions, casts, masks, and small score-feature operations. Inspect the compiled kernel graph before implementing an equivalent custom kernel.

If activation traffic remains significant, prototype exact GELU-and-gate or residual-and-LayerNorm fusion. Preserve the current exact erf-based GELU; a tanh or sigmoid approximation changes the model and needs separate quality measurements. Inspect actual Q/K/V strides and copies before adding layout conversions: MLX's full attention implementation accepts a contiguous head dimension with other striding and writes an output layout convenient for merging heads. An unconditional contiguous copy may add work.

The marker softmax, top-two sort, entropy, small action head, and NumPy result formatting are legitimate later targets only if measured. There are few option positions compared with hundreds of full-width transformer operations, so optimizing them first is unlikely to address the dominant path.

## Protect benchmark meaning while pursuing aggressive gains

The current [workload generator](../benchmarks/common.py) cycles **three question definitions** to construct 5, 10, or 50 questions. There are at most three unique model inputs in those batches. Exact per-call deduplication can avoid redundant inference in a real application, but it would disproportionately improve these fixtures. Keep this feature separate from kernel optimization and report `questions`, `unique_forward_inputs`, cache hits, and actual evaluated tokens. Reconstruct each original answer using its own labels, ordered criteria, and calibration metadata. Retain a suite with 50 genuinely different questions and a suite with deliberately duplicated inputs.

Do not use cross-call result caching for the primary inference benchmark: it repeatedly calls exactly the same request. Report any cache experiment as such. Tokenizer-cache benchmarks should include both a repeated-rubric scenario and a fresh-input scenario.

For each candidate, use this experiment design:

1. **Hold the target constant.** Record source revision/hash, model revision, dtype, compilation flags, selected quantized modules, token IDs or their hash, shape, marker count, batching policy, warmup, synchronization, and device. The existing `input_sha256` hashes state/questions rather than actual token tensors; it does not establish cross-tokenizer tensor identity. Re-run the baseline from the same final source revision because historical JSON source hashes differ.
2. **Separate workload classes.** Use fixed shapes for controlled kernel comparisons; real variable lengths for scheduling and compilation; distinct questions for throughput; repeated rubrics for legitimate CPU caching; and deliberately duplicated questions for deduplication. Include long-length tails and different option counts. Keep the source text and truncation identical within each backend comparison.
3. **Measure cold and warm behavior.** Record model load and first compilation separately. Measure preparation, graph construction, synchronized evaluated forward, output conversion, and end-to-end latency without treating independent medians as additive. Profile kernels in a separate run, because tracing can perturb latency.
4. **Change one optimization at a time.** Screen with the existing iteration count, then repeat finalists in alternating baseline/candidate blocks and collect enough samples for a credible p95, for example at least 200 timed requests per workload. Use a single active GPU benchmark, consistent power/thermal conditions, and preserve raw samples. Require an improvement larger than measured run variability.
5. **Check correctness and stability.** Compare against MLX at the same dtype and the existing FP32 reference; enforce token identity for scheduling and CPU preparation changes. Exercise lengths around 64, 128, and bucket boundaries; batches around chunk limits; one and many options; multilingual inputs; padding; and shape changes after compilation. Repeat changing-shape requests, watch both active and cache memory after warmup, and verify finite deterministic results within each configuration.
6. **Apply stronger gates to approximate changes.** Quantization, activation approximation, token pruning, early exits, and distillation require held-out task and calibration results beyond the small regression fixture. Preserve a separate model identity and benchmark label when changing trained behavior. A faster multilingual checkpoint or distilled model is a different model, not a speedup of the identical Laya checkpoint.

For any measured hotspot occupying fraction `f` of end-to-end time and accelerated by factor `s`, use Amdahl's bound `1 / (1 - f + f/s)` to assess expected total impact. Use a measured time fraction for `f`; the arithmetic fractions above are not substitutes. The next concrete engineering decision should follow the compile/batching ablation and a short-versus-long kernel profile, not an unverified multiplier.

## Documentation and reproducibility notes

The documentation lookup used the required Context7 workflow: one `library MLX` resolution, then separate official-documentation queries for compilation and fast attention, using `/websites/ml-explore_github_io_mlx_build_html` (three commands total). Installed `.pyi` and Python sources were inspected to verify MLX 0.32.2 behavior, including `force_fused`, quantization APIs, compiled GELU, and fast LayerNorm. Version-pinned upstream C++/Metal sources were read for dispatch and tile-loop details. No library was upgraded for this research.

The two FP16 baseline files used for the detailed observations had these SHA-256 digests at inspection time:

```text
laya-mlx-float16.json
63146dd664d039dde1a728b17aad896e491bd01ea36ea0786953691180e55b09

laya-multilingual-mlx-float16.json
9af74bd5a11e4edc15e6a8c9dc929a7b9fd2d19cb06f348cd0e04a076f912473
```

The main benchmark process was still producing additional artifacts during this research. The table deliberately cites complete baseline files already available when reviewed and makes no claims about unmeasured candidate implementations.
