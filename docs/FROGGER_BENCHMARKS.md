# Frogger speed and stability: M1

The **truecolor** campaign completed **8,160 moves with zero deaths and 9 safety interventions**. The default model was `aac6fef/laya-multilingual-mlx` in FP16, with compact planner descriptions and a 24 × 16 board. These measurements use the default eager runtime on an Apple M1, not the M3 Max used for the [Snake report](SNAKE_BENCHMARKS.md), so the two reports are not a speed comparison between games.

**Sustained uncapped throughput: 18.94 moves/second overall across 2,400 steps.** Each of four seeds completed 600 moves. Their rates ranged from 18.31 to 19.42 moves/second. Every game survived and kept crossing, with 14–17 crossings per seed (level 5–6). The longest interval between two crossings was 69 moves. The safety layer corrected 4 raw model proposals in these episodes.

**Highest passing tested paced computation budget: 10 FPS.** All four seeds met the requirement that at least 99% of active ticks fit within 100 ms. The actual wall-clock rate, including OS sleep overshoot, was 9.58–9.59 moves/second. This is a budget setting, not a claim of exactly 10 rendered frames every second.

## Sustained maximum speed

| Seed | Steps | Actual steps/s | Active tick p50 / p99 (ms) | Crossings / level | Interventions |
| --- | --- | --- | --- | --- | --- |
| 101 | 600 | 18.70 | 54.38 / 66.64 | 14 / 5 | 1 |
| 102 | 600 | 18.31 | 54.52 / 77.00 | 15 / 6 | 0 |
| 103 | 600 | 19.36 | 51.88 / 63.80 | 17 / 6 | 3 |
| 104 | 600 | 19.42 | 52.13 / 63.71 | 16 / 6 | 0 |

Combined uncapped model-inference p50 / p95 / p99 was **38.59 / 41.42 / 43.89 ms**. Complete active-tick p50 / p99 was **52.70 / 67.29 ms**. An active tick includes the lookahead planner (about 6 ms), synchronized inference, Rich composition, truecolor ANSI serialization and the game update. No pacing delay is inserted. Inference takes longer than in the two-option Flappy demo because each move describes five options.

## Fixed-budget stability

| Seed | Target FPS | Steps | Actual steps/s | Active tick p99 (ms) | Budget misses |
| --- | --- | --- | --- | --- | --- |
| 101 | 10.0 | 600 | 9.58 | 92.01 | 1 (0.17%) |
| 102 | 10.0 | 600 | 9.59 | 84.37 | 1 (0.17%) |
| 103 | 10.0 | 600 | 9.58 | 81.34 | 0 (0.00%) |
| 104 | 10.0 | 600 | 9.59 | 86.83 | 1 (0.17%) |

The passing test had **3 late active ticks out of 2,400 (99.88% within budget overall)**. The longer 12 FPS test passed its first seed (4/600 late, 0.67%) and failed its second (7/600 late, 1.17%), so it was stopped, as the method specifies. All moves wait for a fresh model result, so slow inference reduces the game rate instead of executing stale actions.

## Short sweep

Each candidate below ran 120 moves with seed 7. A short passing probe selects a candidate for the longer four-seed test; it is not sufficient by itself to claim stability. All games survived, including timing failures.

| Target FPS | Actual steps/s | Active tick p99 (ms) | Budget misses | Short sweep |
| --- | --- | --- | --- | --- |
| 10.0 | 9.56 | 86.40 | 0.00% | pass |
| 12.0 | 11.43 | 78.95 | 0.00% | pass |
| 15.0 | 14.50 | 70.65 | 2.50% | fail |
| 18.0 | 17.71 | 63.56 | 23.33% | fail |
| 20.0 | 18.55 | 65.59 | 96.67% | fail |
| 25.0 | 18.66 | 66.62 | 100.00% | fail |
| 30.0 | 18.62 | 64.38 | 100.00% | fail |
| 35.0 | 18.76 | 62.57 | 100.00% | fail |
| 40.0 | 18.85 | 62.55 | 100.00% | fail |
| 45.0 | 18.74 | 62.98 | 100.00% | fail |
| 50.0 | 17.86 | 73.27 | 100.00% | fail |
| 60.0 | 18.12 | 69.18 | 100.00% | fail |

The default 12 FPS presentation target passed this short probe but not the longer test, so it is not claimed as stable on this machine. Normal desktop activity, scheduling and clock behavior were not isolated experimentally. We report the achieved rate, all samples and the highest passing **tested** setting.

## What the stability result includes

Laya receives planner features, including which moves collide, which leave no path to a safe row and which safe move reaches home fastest. It computes real probabilities. The lookahead safety layer may correct execution while retaining the raw probability bars and counting every intervention. The checkpoint was not trained on Frogger here. [Exact UI metric meanings and policy behavior](FROGGER_DEMO.md#what-the-ai-does).

A separate raw top-1 control ran seeds 101–103 for 200 moves each with the execution shield disabled. All three survived, with 6, 5 and 7 crossings. The raw control still supplies planner features. Its survival over this horizon does not establish indefinite unshielded survival or unaided board reasoning.

The published showcase is a separate **100.09-second actual TTY run** (104 × 35 pseudo-terminal, true color), at a 12 FPS target: **1,144 moves, 21 crossings, level 8, zero deaths and 1 intervention**. Achieved throughput was 11.43 moves/second, with mean inference of 37.41 ms. Every recorded board and action is checked by deterministic replay in the test suite. The [30-second video](assets/frogger-demo.mp4), [GIF](assets/frogger-demo.gif) and [poster](assets/frogger-preview.png) are rendered from it.

A separate optimized maximum-speed TTY recording completed **356 moves in 20.04 seconds (17.76 moves/second)** with `--optimize --max-speed`: 10 crossings, level 4, zero deaths and 1 intervention. It includes the actual terminal stream writes and was made before a planner optimization that cut planning from about 9 to 6 ms without changing any decision (verified on 2,000 moves). Its [15-second original-speed video](assets/frogger-fast.mp4) and [complete recording](../benchmarks/results/frogger-fast.jsonl) are included alongside the slower presentation clip. The paired eager/optimized study in [Snake optimization](SNAKE_OPTIMIZATION.md) was not repeated for Frogger.

## Measurement boundaries and provenance

- Apple M1; macOS 26.6.2, Python 3.12.14.
- MLX 0.32.2, NumPy 2.5.3, Rich 15.0.0, tokenizers 0.23.2, huggingface-hub 1.32.0.
- Original FP16 weights, with no quantization or custom kernel. Upstream model revision: `052592a15d198d9ad47da779604259b10b47b7aa`.
- Weight SHA-256 from the previously verified published manifest: `7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1`.
- Each move calls `Agent.predict` once with three batched questions. Inference timing includes tokenization, synchronized MLX evaluation and output conversion.
- The benchmark explicitly enables true color, independent of `NO_COLOR`, and serializes Rich output into an in-memory stream. The terminal emulator's screen painting, model loading and warmup are excluded.
- The [report](../benchmarks/results/frogger.json) stores configuration, environment, source fingerprint and every per-step probability, execution decision and timing.
