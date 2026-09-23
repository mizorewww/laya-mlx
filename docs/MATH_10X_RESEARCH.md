# Can Laya MLX become ten times faster? A mathematical investigation

Research date: 2026-09-19. Baseline: the committed FP16 MLX results at `49daed6` on the Apple M3 Max with 40 GPU cores and 128 GiB unified memory. This report separates **algebraic facts**, **static cost estimates**, **CPU measurements of selected checkpoint matrices**, and **hypotheses requiring inference experiments**. No GPU inference or new latency measurement was performed for this mathematical investigation. The companion [engineering investigation](ENGINEERING_10X_RESEARCH.md) contains candidate timings when available. Here, “exact” refers to preserving mathematical dependencies and the real-arithmetic function; a different GPU reduction order or kernel can still change floating-point results, so the existing numerical tolerances and exposed-output contract remain acceptance gates.

**Decision:** do not budget for a universal 10× end-to-end improvement from hand-written kernels while retaining these checkpoints and their full outputs. Exact local attention and output pruning are worthwhile, bounded improvements. Direct low-rank decomposition is not close to lossless in the four sampled weight matrices. A 10× product improvement is credible for workloads with substantial exact repetition, or as the goal of a substantially smaller distilled/restructured model. Those are different promises and must have different benchmarks.

## 1. What ten times faster actually requires

These are the existing synchronized, warm end-to-end medians, including preparation and formatting; they are **not** new measurements. Each baseline used five warmups, 50 timed iterations, and a question batch limit of 64. A short 50-question fixture repeats three question definitions; its original runtime nevertheless evaluates all 50. Download, loading, and compilation time are outside these medians.

| Model | Short 1: baseline → 10× target | Short 10 | Short 50 | Long 1 | Long 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya | 13.421 → **1.342 ms** | 71.068 → 7.107 ms | 336.030 → 33.603 ms | 44.927 → **4.493 ms** | 420.987 → 42.099 ms |
| Multilingual | 7.390 → **0.739 ms** | 27.386 → 2.739 ms | 127.565 → 12.756 ms | 37.635 → **3.763 ms** | 389.487 → 38.949 ms |
| Typed decisions | 13.712 → **1.371 ms** | 75.618 → 7.562 ms | 380.560 → 38.056 ms | 99.233 → **9.923 ms** | 1000.294 → 100.029 ms |

Sources at the research snapshot: [Laya FP16](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-mlx-float16.json), [multilingual FP16](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-multilingual-mlx-float16.json), [typed-decisions FP16](https://github.com/mizorewww/laya-mlx/blob/49daed6609cb3da142d1a0c88e538dc07f00d974/benchmarks/results/laya-typed-decisions-mlx-float16.json). Short padded lengths are 93/91/93; long lengths are 512/1024/1024. Comparing their long rows does not hold token length constant. The target is a further improvement over native MLX FP16, not over PyTorch MPS FP32.

For any proposed optimization, let `f` be its **measured fraction of end-to-end wall time** and `s` its own acceleration. Amdahl's law gives:

```text
whole-request speedup = 1 / (1 - f + f/s)
10× requires f > 0.9 and s >= f / (f - 0.9)
```

Even accelerating a hotspot containing 95% of request time requires a **19×** hotspot improvement. At 98% it still requires 12.25×; at 99%, 11×. Any untouched preparation, output conversion, or synchronization consuming at least 10% prevents a finite 10× gain from the remaining portion alone. Independent forward and end-to-end medians cannot be subtracted to estimate that fraction; use a segmented timing experiment or profile.

## 2. Dense work and conditional lower bounds

From [model.py](../laya_mlx/model.py), define hidden width `D`, encoder MLP width `I`, encoder depth `N`, head depth `H`, batch size `B`, and padded token length `L`. Counting a multiply and an add as two FLOPs:

```text
A = N * (4 D² + 3 D I) + H * 12 D²
main dense FLOPs = 2 B L A
current dense attention products = 4 B (N + H) L² D
```

`3DI` includes both branches of the gated encoder MLP and its output projection. Head MLPs use a different, conventional `4D` expansion. These formulas exclude norms, activations, scoring, masks, transfers, and scheduling; they are a conventional dense implementation cost model, **not an unconditional arithmetic lower bound over all possible algorithms**.

| Family | D / I / N / H | Main dense weights A | Encoder MLP share of A | FP16 bytes for A |
| --- | --- | ---: | ---: | ---: |
| Laya / typed decisions | 1024 / 2624 / 28 / 2 | 368,312,320 | 61.28% | 736,624,640 |
| Multilingual | 768 / 1152 / 22 / 2 | 124,452,864 | 46.92% | 248,905,728 |

The multilingual embedding has 196,608,000 weights, but inference gathers selected rows rather than multiplying by the entire vocabulary. Total checkpoint parameters therefore exaggerate its per-token work relative to English. A smaller embedding file is not automatically faster inference.

| Model / shape | Dense + dense-attention work | Required effective throughput at 10× target |
| --- | ---: | ---: |
| Laya, B=1 L=93 | 69.57 GFLOPs | **51.84 TFLOP/s** |
| Laya, B=50 L=93 | 3478.44 GFLOPs | **103.52 TFLOP/s** |
| Laya, B=1 L=512 | 409.36 GFLOPs | **91.12 TFLOP/s** |
| Multilingual, B=1 L=91 | 23.26 GFLOPs | **31.48 TFLOP/s** |
| Multilingual, B=50 L=91 | 1163.05 GFLOPs | **91.17 TFLOP/s** |
| Multilingual, B=1 L=1024 | 332.19 GFLOPs | **88.27 TFLOP/s** |
| Typed decisions, B=1 L=1024 | 883.15 GFLOPs | **89.00 TFLOP/s** |

These are **requirements**, not claimed Apple GPU peaks. A measured dense GEMM ceiling at these shapes is the useful engineering comparison. A measured ceiling can make a project implausible; it is still not proof of a hardware upper bound. Since CPU work also fits inside the target, actual GPU execution must finish sooner than this table allows if CPU work does not overlap it.

Apple specifies 400 GB/s unified-memory bandwidth for this 40-core M3 Max configuration. Under the explicit assumption that the main FP16 matrix weights are read from unified memory once per request and are not already retained in on-chip cache, the ideal streaming floors are **1.842 ms** for English/typed and **0.622 ms** for multilingual. These omit activations, embeddings, scorer weights, and all compute. They also assume the advertised aggregate bandwidth is fully available to this workload. [Apple technical specifications](https://support.apple.com/en-us/117737).

The English short-single targets of 1.342/1.371 ms are already below that conventional FP16 streaming floor. To satisfy those targets at 400 GB/s without changing weight storage would require approximately **200/188 MB of the counted weights to avoid the memory read**, or another change to the execution assumptions. This report does not assume or invent an on-chip cache capacity. Weight reuse across a batch, exact compression, and alternative algorithms change the bound; the observation is not a universal impossibility theorem.

For the dense part alone, ideal weight-only arithmetic intensity is `BL FLOPs/byte` in FP16: 93 at B=1 L=93 and 4650 at B=50. Activation traffic lowers those numbers. This explains why weight compression becomes a less convincing throughput strategy as the number of tokens sharing each matrix grows.

## 3. How much exact work can actually be removed?

The first global encoder layer makes every valid token potentially relevant, and both decision-head layers are global. A token being absent from the final output does not make its intermediate representation dispensable: later queries still use it as a key/value.

The exact exception is the **last head layer**. Compute its K/V for all valid tokens, Q only for CLS and option markers, and its output projection/MLP only at those selected positions. The previous head and full encoder must still produce all valid token states. This is dependency pruning, not attention-head removal. With `R=5` selected positions including CLS, its maximum dense saving is `20 B (L-R) D²` FLOPs when Q is split out of the fused QKV projection, plus `4 B L (L-R) D` attention FLOPs. Preserving the fused QKV projection is simpler but saves less.

Local encoder attention permits `abs(q_position-k_position) <= 64`, an **inclusive 129-position** interior window. The number of valid local pairs for `L>64` is `129L - 64*65`. Skipping forbidden K/V tiles is mathematically exact if padding and positions are preserved. A mask applied after dense QK multiplication does not realize that arithmetic saving.

| Model / length | Attention-products share of total modeled FLOPs | Exact local-window saving | Final-head selection saving, R=5 | Combined saving |
| --- | ---: | ---: | ---: | ---: |
| Laya, 93 | 1.53% | 0.086% | 2.701% | **2.787%** |
| Laya, 512 | 7.87% | 3.607% | 2.857% | **6.464%** |
| Typed decisions, 1024 | 14.59% | 7.686% | 2.904% | **10.589%** |
| Multilingual, 91 | 2.62% | 0.130% | 4.465% | **4.595%** |
| Multilingual, 1024 | 23.27% | 11.919% | 4.584% | **16.503%** |

These are reductions in modeled work, not latency predictions. They leave 83.5–97.2% of the modeled work intact, far from the 10% budget. Even making **all attention products free** removes only 1.53–23.27% here. Conversely, a hypothetical 10× acceleration of every dense projection while leaving attention unchanged gives only 8.79× for English short inputs and 3.23× for multilingual long inputs under equal FLOP efficiency. A real profile must replace these arithmetic fractions with measured time fractions before applying Amdahl.

The runtime already calls MLX fast SDPA. FlashAttention computes the same dense softmax-attention function with less intermediate memory traffic; its tiling does not make arbitrary global attention linear in token count. Exploiting the checkpoint's existing local mask is exact, while imposing new sparsity on its global layers changes the model. The low rank of `QKᵀ` does not imply low rank after elementwise exponentiation and row normalization. [FlashAttention paper](https://arxiv.org/abs/2205.14135), [MLX attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).

Unpadding is also exact with independent sequences, original positions, and output order maintained. The current short 50-question fixtures waste only 8.9% of English/typed and 5.8% of multilingual padded tokens. Those fixtures cannot obtain 10× from padding removal. A different workload containing one 1024-token item and 49 64-token items would waste enough padding for a 12.3× token-work ratio; that would be a scheduling result specific to that distribution.

## 4. Does low-rank factorization reveal a hidden 10× shortcut?

For a frozen matrix `W` of shape `m × n`, replacing it with two factors `U(m × r)` and `V(r × n)` changes per-token multiplication cost from `mn` to `r(m+n)`. Breaking even requires `r < mn/(m+n)`; a 10× matrix-work reduction requires:

```text
r <= mn / (10(m+n))
best squared Frobenius residual at rank r = sum_{j>r} sigma_j²
```

The second expression is the truncated-SVD optimum. A square 1024-wide projection needs rank at most 51; its exact full-rank factorization instead doubles the multiplication count. GELU, gating, softmax, and input-dependent LayerNorm prevent simply multiplying neighboring layer weights into one constant matrix.

I inspected **four explicitly selected middle-layer matrices** using a CPU float64 Gram eigensolver, one attention output matrix and one fused MLP input matrix from each model family. The largest eigensystem was 1024×1024; all requested BLAS thread limits were one, no GPU libraries were imported, and no model forward pass was run. These are full spectra of the selected matrices, not a random projection estimate and not a survey of every layer.

| Sample matrix | Shape | Maximum rank for 10× matrix-work reduction | Squared Frobenius energy retained at that rank | Best relative Frobenius error | Rank retaining 99% energy |
| --- | --- | ---: | ---: | ---: | ---: |
| Laya layer 14 attention Wo | 1024×1024 | 51 | 28.66% | **84.46%** | 690 |
| Laya layer 14 MLP Wi | 5248×1024 | 85 | 29.29% | **84.09%** | 956 |
| Multilingual layer 11 attention Wo | 768×768 | 38 | 24.97% | **86.62%** | 516 |
| Multilingual layer 11 MLP Wi | 2304×768 | 57 | 32.88% | **81.93%** | 697 |

Raw measurements, selected-matrix SHA-256 values, singular-value extremes, stable ranks, and additional candidate ranks are in [math_spectrum.json](../experiments/math_spectrum.json). Each sampled matrix is numerically full rank. In all four, retaining 99% squared Frobenius energy requires a rank **above the two-factor arithmetic break-even point**. The multilingual MLP Wi has stable rank only 13.29, yet rank 57 retains merely 32.88% of total energy: stable rank is not the dimension needed for a small reconstruction error.

This is strong evidence against **plain weight-only SVD as a near-lossless shortcut**. It does not prove poor task accuracy for every low-rank model: token activations can occupy a restricted distribution, and retraining can move useful computation into a smaller representation. Activation-aware compression should minimize error weighted by actual input covariance, approximately `||(W-Wr) Sigma_x^(1/2)||F`, and then test end-to-end quality. The four-matrix analysis does not estimate those covariances, a global logit-error bound, or a full-model attainable speedup. A low-rank fine-tuning delta also does not imply that the frozen pretrained matrix itself can be discarded.

Hand-written fast matrix multiplication does not remove this evidence. As an arithmetic illustration, a seven-product block recursion instead of eight saves only 12.5% of multiplication work per level, before additional matrix additions and traffic; even ten ideal levels produce about 3.8× fewer multiplications. Applying deep recursion to 768–1024-wide projections is not a credible 10× latency plan, particularly against already tiled GPU GEMMs. This is not a claim that all possible exact algorithms have been ruled out.

## 5. Quantization, pruning, and early exit change the contract

**Weight-only quantization.** For affine groups of 64 with FP16 scale and offset, stored bytes per matrix parameter are approximately `bits/8 + 4/64`. That yields ideal matrix-storage reductions of **1.88× at 8-bit**, **3.56× at 4-bit**, and **6.40× at 2-bit** relative to FP16. These are not compute reductions or wall-time gains. Ten times less weight traffic by this mechanism alone would require roughly one bit per weight plus metadata, a radically different approximation. Existing norm/embedding/head tensors and decode overhead reduce the whole-request benefit further. [MLX quantize documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html), [quantized matmul documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html).

Quantization can be useful at B=1 if weight traffic dominates, but it need not accelerate a large token GEMM. It changes logits, score expectations, entropy confidence, and action probabilities. The public API exposes all of these, so argmax agreement alone is insufficient. If each final logit changes by at most `epsilon`, a top-two logit margin greater than `2*epsilon` certifies the winning label, but does **not** certify probability, score, or action agreement. Temperature calibration divides logit errors by its temperature; a small temperature can magnify a seemingly small raw error. The existing checkpoints contain option-count temperature buckets near 0.1006, making this relevant.

**Token pruning.** At an unchanged width/depth and dense-compute efficiency, an approximate 10× dense-work goal requires retaining about 10% of token processing across layers, not removing a few punctuation tokens. Pruning after processing fraction `a` of the original depth has dense-work ratio `a + (1-a)rho`, where `rho` is the retained-token fraction for subsequent layers. If `a >= 0.1`, even discarding every remaining token cannot yield more than 10× in that simplified model. In this model, the first global layer already connects each state token to all unmasked question/option tokens. Learned token importance and dynamic pruning can be studied, but their correctness is a task-quality claim requiring training/calibration. Dropped tokens may contain negations, rare entities, or the fact deciding a close option; low early attention is not a proof of irrelevance at later layers.

**Early exit.** Simply feeding layer-3 hidden states to a head trained after layer 28 does not preserve its input distribution. Intermediate heads and a validated confidence rule must be trained. A 10× uniform-cost depth budget is about 2.8 encoder layers for English or 2.2 for multilingual, before head and CPU overhead. Keeping the full current head makes the short-sequence dense budget stricter: its two layers alone are 6.83%/11.37% of English/multilingual `A`. For multilingual, that head alone exceeds the entire 10% dense-work budget. Batch divergence also matters: exiting individual items saves nothing if they remain in an unshrunk dense batch. [FastBERT](https://arxiv.org/abs/2004.02178) and [DeeBERT](https://arxiv.org/abs/2004.12993) establish trained adaptive-inference approaches with accuracy/speed tradeoffs; their reported gains are not measurements of Laya or of this Mac.

An approximate student plus teacher fallback has expected normalized cost approximately `c + q`, where `c` is student cost divided by teacher cost and `q` is the teacher fallback rate, assuming serial execution. To meet 10×, `c + q <= 0.1`: a student costing 5% of the teacher leaves at most 5% of requests for fallback. That can improve mean latency while the difficult-request p95 remains near teacher latency. Confidence-based routing is not an exact equivalence certificate.

## 6. Precisely what can be reused across questions?

The prompt is `[CLS] question/options [SEP] state [SEP]`; different questions can change both the state offset and the state truncation. For first-layer query `i`, the attention output is:

```text
o_i = sum_j exp(q_i dot k_j / sqrt(d)) v_j
      / sum_j exp(q_i dot k_j / sqrt(d))
```

Changing any unmasked question key/value can change both numerator and denominator for every state query. For finite logits, the unmasked softmax weights are positive in real arithmetic. Consequently, the contextual state after the first global layer depends on the question. All subsequent K/V depend on those changed states. Reusing a complete state encoding or a decoder-style K/V cache across different questions therefore changes the function. Shared raw text is insufficient; offset, truncation, masks, and marker metadata also matter.

There is a small exact exception worth distinguishing: before that first attention operation, normalized token embeddings and their **first-layer, pre-RoPE Q/K/V projections** depend only on token identity. They can be cached or precomputed, with absolute RoPE positions applied afterward. Eliminating the entire first QKV projection removes only **0.85%/1.42% of main dense work** in English/multilingual, before lookup traffic. A full-vocabulary QKV table adds roughly 309 MB/1.18 GB of FP16 storage if retained alongside the ordinary embeddings. The first layer's state-to-state softmax sufficient statistics can also be reused under identical state token/truncation and relative-position conditions, then combined with question contributions by stable softmax merging. This only saves a portion of one attention layer; it does not make later contextual states reusable.

**Exact whole-input deduplication** has much larger potential. If `N` requested questions contain `U` identical prepared forward inputs, evaluate `U` and map their raw outputs back to all original questions with the correct ordered labels, calibration, IDs, and usage accounting. Equality and cache keys must cover all prepared tensors, including masks, marker positions, and question types; cross-call keys must also identify checkpoint revision, dtype, and execution configuration. In the existing 50-question fixture, `U <= 3`, so the ideal linear-work ratio is **50/3 = 16.67×**. Actual latency is less predictable because small batches have different efficiency and preparation/output mapping remain. For 10 questions and three unique inputs, the ratio is only 3.33×. A 50-distinct-question benchmark must accompany either result.

With cross-call result caching, average normalized latency is `1-h+h*epsilon`, where `h` is the hit rate and `epsilon` the cache-hit cost divided by uncached inference cost. A 10× average improvement requires `h >= 0.9/(1-epsilon)`; if a hit costs 1% of inference, the required hit rate is **90.91%**. Report hit rates, misses, cold-cache latency, and the miss path separately. The standard benchmark repeats exactly the same request, so an unlabelled cross-call cache would largely stop measuring model execution.

**A shared-state encoder plus question-specific cross-attention** is a promising redesigned product, but requires retraining or distillation because it removes the original early question/state interaction. If the reusable state pass costs roughly one old per-question pass, then at `Q=50` the shared pass consumes 2% of the old total budget; question-specific work may consume at most another 8% for a 10× target. At `Q=10`, that single shared pass already uses 10% before question-specific work. State length, option complexity, and the chosen smaller state encoder change this estimate.

## 7. A credible route to an order-of-magnitude model improvement

Reducing both depth and width provides enough arithmetic room to absorb overhead. The following are **student design budgets**, not implemented models, quality claims, or measured speedups. They retain the same token lengths, use a gated encoder MLP and one conventional decision-head layer, and count the same `A` formula.

| Teacher | Candidate N / D / I / H | Main dense weights | Teacher/student dense-work ratio |
| --- | --- | ---: | ---: |
| Laya / typed | 6 / 512 / 1344 / 1 | 21.82 M | **16.88×** |
| Laya / typed | 4 / 512 / 1344 / 1 | 15.60 M | **23.61×** |
| Multilingual | 6 / 384 / 576 / 1 | 9.29 M | **13.40×** |
| Multilingual | 4 / 384 / 576 / 1 | 6.78 M | **18.35×** |

The embedding can remain relatively large and still be cheap to gather. Distill teacher option distributions and action outputs, mix in labeled tasks, cover score/noul behavior and varying option counts, then refit output calibration on held-out data. Evaluate full language coverage and distinct questions. [TinyBERT](https://arxiv.org/abs/1909.10351) is evidence that jointly reducing encoder depth/width through distillation can yield a major speed/quality tradeoff in another BERT setting; its reported 9.4× inference gain does not transfer numerically to Laya.

For hand-written engineering, prioritize the following decisions:

1. **Establish the limit before writing a new GEMM.** Measure compiled model time and representative dense primitives at B=1 and a throughput batch. Compare actual sustained throughput with the 31–104 TFLOP/s requirements above. If launch removal leaves dense execution dominant, new elementwise kernels cannot supply the missing order of magnitude.
2. **Implement exact output selection and exact local attention as bounded projects.** The last head has a clear dependency proof; the long multilingual local path has the largest exact arithmetic opportunity. Inspect measured wall-time fractions before maintaining custom Metal. Preserve existing fast attention's numerical contract and test boundary lengths/padding.
3. **Ship exact deduplication only with workload accounting.** This is the fastest path to a possible 10× result for a sufficiently repetitive application. It must coexist with uncached, unique-input benchmarks so users can predict their own results.
4. **Treat 4/8-bit and activation-aware low rank as measured approximations.** Require a speed improvement and calibrated quality gates together. Neither fewer stored bytes nor a low stable-rank number is sufficient.
5. **If 10× is required on fresh, diverse requests, develop and validate the smaller student or shared-state architecture.** The student budget deliberately targets more than 10× dense-work reduction because attention, CPU preparation, and small-kernel overhead remain. A quality budget and suitable training/evaluation data are prerequisites; the existing parity fixture cannot validate this claim.

For approximate variants, acceptance should record choice agreement and labeled accuracy, score error, probability drift, confidence calibration, action probabilities, and close-margin cases. Existing 378/378 argmax checks, 600 finite repeat calls, and AG News regression alignment establish the current port's behavior on those tests; they do not validate a new compressed model. Keep a distinct model identity and report p50/p95, cold setup, memory, unique input count, and quality together.

## Reproduction and scope

- [math_costs.py](../experiments/math_costs.py) reproduces every architecture-derived table and latency target from checkpoint configs and the existing benchmark JSON; [math_costs.json](../experiments/math_costs.json) records input file hashes and checkpoint revisions.
- [math_spectrum.py](../experiments/math_spectrum.py) reproduces the four selected CPU matrix spectra; [math_spectrum.json](../experiments/math_spectrum.json) includes exact matrix hashes. It uses NumPy and safetensors only and does not load the complete model.
- Run `.venv/bin/python experiments/math_costs.py` and `.venv/bin/python experiments/math_spectrum.py` from the repository root after downloading the pinned source checkpoints. CPU sampling and engineering GPU timings were coordinated to avoid overlap.
- Current MLX quantization semantics were checked with the `find-docs` skill using the required Context7 library resolution followed by a separate quantization documentation query. Official MLX documentation, the ModernBERT/FlashAttention and distillation/early-exit papers, Apple specifications, and the actual local model were used as sources. No performance number from a paper is presented as a measurement on this machine.

**中文结论：** 相同 checkpoint、相同完整输出语义下，暂时没有可信的“手写几个 kernel 就再快 10×”路径。现有模型的大头是 dense 计算；局部 attention 加最后 head 精确裁剪只减少约 2.8%–16.5% 的建模 FLOPs。真实权重抽样显示，把矩阵分解压到十分之一工作量会产生约 82%–87% 的最佳相对 Frobenius 重建误差，不能当成近似无损捷径。10× 更有希望来自高重复输入的精确去重/缓存，或通过蒸馏把层数与宽度一起缩小、重新设计共享 state 的编码方式。前者需要公布命中率与独立输入性能，后者需要训练和重新验证准确率、分数、概率及 action 行为。
