# v0.4.12 qualification evidence

Core ML release follow-up: the 0.5.1 compiled-instance fix has separate [correctness, performance and tail-latency evidence](https://github.com/mizorewww/laya-coreml/blob/main/docs/QUALIFICATION_0_5_1.md). The Core ML figures below describe the earlier loader; MLX runtime code is unchanged.

Upstream: `ffc142012af63d015fa19f05f29f9c5ff329a2cc`. These development qualification results used package version 0.4.0. The 0.5.0 release changes only runtime version metadata; final package and publication evidence is recorded separately in `benchmarks/results/release-v050.json`.

## Correctness and scope

All applicable host interfaces are ported: batch/long prediction, structured decisions, hooks, calibration, evaluation, router, HTTP/MCP and integrations. The 341-entry API audit has no unclassified differences. Native tensor, device, artifact and resource boundaries are explicit; PyTorch/CUDA/TileLang/ONNX execution and training are not emulated.

Both backends pass 63 questions on each of three checkpoints in FP32 and FP16, with 100 stable repeated calls per combination. FP32 public/probability tolerance is 0.0001, FP16 tolerance is 0.02; structures, categories, integers and booleans match exactly. These are finite parity checks, not a proof of general model accuracy. Calibration tests check exact official fitting behavior on identical real native logits, with repeated synthetic labels; they do not measure deployment calibration quality.

## FP32 sequential feature matrix

Apple M3 Max, 128 GiB unified memory. Upstream PyTorch 2.14, native MLX and native Core ML CPU/GPU. Same checkpoints and input hashes; 5 warmups and 30 timed calls per workload. End-to-end completed calls include host processing and inference; model loading and warmup are excluded. Core ML uses dense B=1/2/4/16 adaptive FP32 bundles and the final bounded shape cache.

All 54 output rows (including the 18 upstream reference rows) pass full nested comparison. MLX wins 18/18 P50 comparisons; Core ML wins 14/18. The four Core ML regressions remain visible below; the alternating-order repeats below resolve their qualification with all measured samples retained. Shared desktop load was variable, so one sequential matrix is insufficient to establish stable marginal speedups.

P50 milliseconds; speedup is upstream P50 divided by native P50.

| Model | Workload | Upstream | MLX | MLX speedup | Core ML | Core ML speedup |
|---|---|---:|---:|---:|---:|---:|
| laya | single | 23.37 | 15.64 | 1.49× | 16.85 | 1.39× |
| laya | batch8 | 114.33 | 97.75 | 1.17× | 102.08 | 1.12× |
| laya | long | 132.00 | 96.99 | 1.36× | 137.75 | 0.96× |
| laya | structured | 16.44 | 11.54 | 1.42× | 10.20 | 1.61× |
| laya | stability | 30.66 | 23.17 | 1.32× | 32.27 | 0.95× |
| laya | mixed_shapes | 103.79 | 86.74 | 1.20× | 115.45 | 0.90× |
| laya-multilingual | single | 14.34 | 9.93 | 1.44× | 11.87 | 1.21× |
| laya-multilingual | batch8 | 74.34 | 65.95 | 1.13× | 99.48 | 0.75× |
| laya-multilingual | long | 88.62 | 67.60 | 1.31× | 85.91 | 1.03× |
| laya-multilingual | structured | 11.43 | 7.84 | 1.46× | 5.89 | 1.94× |
| laya-multilingual | stability | 20.73 | 14.19 | 1.46× | 19.40 | 1.07× |
| laya-multilingual | mixed_shapes | 72.18 | 55.45 | 1.30× | 58.23 | 1.24× |
| laya-typed-decisions | single | 31.89 | 16.57 | 1.92× | 24.74 | 1.29× |
| laya-typed-decisions | batch8 | 213.43 | 165.09 | 1.29× | 177.44 | 1.20× |
| laya-typed-decisions | long | 196.19 | 185.00 | 1.06× | 161.13 | 1.22× |
| laya-typed-decisions | structured | 19.28 | 15.13 | 1.27× | 9.84 | 1.96× |
| laya-typed-decisions | stability | 42.42 | 36.51 | 1.16× | 35.39 | 1.20× |
| laya-typed-decisions | mixed_shapes | 146.86 | 116.35 | 1.26× | 129.20 | 1.14× |

The raw [matrix summary](../benchmarks/results/v0412-final/summary.json) includes P95, mean-based throughput ratios, process peak RSS, tolerance and baseline version. Per-run JSON includes every sample and public output. `single` asks two questions; `batch8` asks two questions for eight heterogeneous states; `mixed_shapes` makes four calls/eight decisions per iteration.

## Alternating-order Core ML confirmation

Each checkpoint runs upstream → Core ML → Core ML → upstream in fresh sequential processes, with the same six-workload prefix, 5 warmups and 30 timed calls per phase. Pooled statistics use all 60 samples per backend/workload. All 18 workloads pass full output comparison and improve P50, P95 and mean-based throughput in this confirmation campaign. The earlier sequential matrix is retained above; this does not guarantee that every individual run under changing desktop load will be faster.

| Model | Workload | Upstream P50 ms | Core ML P50 ms | P50 speedup | Throughput ratio |
|---|---|---:|---:|---:|---:|
| laya | single | 22.96 | 15.92 | 1.44× | 1.46× |
| laya | batch8 | 129.50 | 122.53 | 1.06× | 1.06× |
| laya | long | 131.69 | 106.91 | 1.23× | 1.23× |
| laya | structured | 16.39 | 9.82 | 1.67× | 1.65× |
| laya | stability | 30.31 | 23.59 | 1.28× | 1.29× |
| laya | mixed_shapes | 102.72 | 78.73 | 1.30× | 1.29× |
| laya-multilingual | single | 14.23 | 8.74 | 1.63× | 1.56× |
| laya-multilingual | batch8 | 52.73 | 46.32 | 1.14× | 1.15× |
| laya-multilingual | long | 78.43 | 49.10 | 1.60× | 1.59× |
| laya-multilingual | structured | 11.04 | 5.58 | 1.98× | 1.97× |
| laya-multilingual | stability | 16.81 | 15.43 | 1.09× | 1.10× |
| laya-multilingual | mixed_shapes | 64.01 | 45.46 | 1.41× | 1.40× |
| laya-typed-decisions | single | 24.14 | 16.88 | 1.43× | 1.46× |
| laya-typed-decisions | batch8 | 135.80 | 121.61 | 1.12× | 1.11× |
| laya-typed-decisions | long | 139.46 | 108.82 | 1.28× | 1.31× |
| laya-typed-decisions | structured | 16.45 | 9.77 | 1.68× | 1.67× |
| laya-typed-decisions | stability | 30.96 | 24.87 | 1.24× | 1.24× |
| laya-typed-decisions | mixed_shapes | 106.42 | 82.34 | 1.29× | 1.30× |

See the [paired summary](../benchmarks/results/v0412-final-paired/paired-summary.json) for both individual-run medians, pooled P95/means and numeric error; adjacent raw reports contain every sample, environment and checkpoint/source hashes. MLX’s sequential FP32 matrix wins 18/18 P50 and mean-throughput comparisons, with 1.06–1.93× P50 speedup. Core ML’s paired P50 range is 1.06–1.98×.

## Final local package gates

MLX: 319 tests passed; Core ML: 334 passed. Each base environment skips the real LlamaIndex module; its two actual-framework tests separately pass for each backend. Both installed wheels pass all eight shared real-framework integration checks. The unchanged upstream standalone HTTP SDK assertions pass extended/tournament and strict modes, including authentication and limits, for both backends.

Both sdist/wheel pairs pass strict Twine checks. Clean inference-only installed wheels pass all three real models with network blocked, and their native source hashes equal the checkout hashes used by the final performance reports. Core ML’s wheel needs no Torch, MLX or Transformers; MLX’s wheel needs no Torch or Transformers. Package evidence and artifact hashes are in `benchmarks/results/v0412-package-gates.json` in each repository. This section records the local qualification stage. Version 0.5.0 subsequently passed remote release CI and was published to GitHub and PyPI; see the [public release receipt](../benchmarks/results/release-v050.json) for artifact and installed-wheel verification.

Local migration qualification is complete for the pinned upstream and tested workloads. This is not a universal accuracy/latency guarantee, a published release, or a claim of lower Core ML memory consumption.

## Resource and precision limits

Core ML holds up to four additional execution instances plus the physical graphs lazily loaded from the finite bundle manifest. Its cache contains execution instances, never answers. Repeated calls predict fresh input tensors; close releases cache references. Extra instances reduce reshape costs but increase resident memory. The final FP32 matrix recorded process peak RSS ranges of 2.54–2.73 GiB for upstream, 0.76–1.09 GiB for MLX and 5.10–13.51 GiB for Core ML across phases/models. Process peak RSS is not a complete device/unified-allocation metric, so it cannot establish a cross-backend memory-efficiency advantage.

Unseen/cold shapes and model loading can be much slower than steady calls. Existing single-graph Hub artifacts do not acquire adaptive capacity automatically. FP16 exports are separately validated but are not used to claim a same-precision FP32 speedup. ANE has its original fixed-shape limits and is outside this full-context CPU/GPU feature matrix.

## Reproduction and historical experiments

Run `python -m benchmarks.qualification` in this repository after creating the dense Core ML bundles documented in the sibling conversion guide. Upstream is launched with this repository’s PyTorch 2.14 reference environment; Core ML uses its own environment. Do not benchmark upstream in the Core ML conversion environment pinned to PyTorch 2.7. Run accelerator commands sequentially.

`python -m benchmarks.paired_features --directory benchmarks/results/v0412-final-paired` performs the alternating-order repeats.

The earlier `v0412-torch27-matrix`, `v0412-abba-batch-torch27` and `v0412-phase-chain` used PyTorch 2.7 as the performance baseline and are not final qualification evidence. The explicit PyTorch 2.14 golden is identical to the previous pure-FP32 correctness reference. Rejected LRU and earlier cache-policy experiments remain recorded rather than removed.
