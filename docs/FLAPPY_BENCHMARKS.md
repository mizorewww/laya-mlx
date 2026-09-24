# Flappy speed and stability: M1

The **truecolor** campaign completed **8,160 ticks with zero deaths and 0 safety interventions**. The default model was `aac6fef/laya-multilingual-mlx` in FP16, with compact planner descriptions, a 24 × 16 board and a first gap of 5 rows. These measurements use the default eager runtime on an Apple M1, not the M3 Max used for the [Snake report](SNAKE_BENCHMARKS.md), so the two reports are not a speed comparison between games.

**Sustained uncapped throughput: 25.79 ticks/second overall across 2,400 steps.** Each of four seeds completed 600 ticks. Their rates ranged from 24.92 to 26.75 ticks/second. Every game survived and kept passing pipes, reaching level 7 (3-row gaps, pipes 8 columns apart). The safety layer corrected no raw model proposals in these episodes.

**Highest passing tested paced computation budget: 12 FPS.** All four seeds met the requirement that at least 99% of active ticks fit within 83.3 ms. The actual wall-clock rate, including OS sleep overshoot, was 11.50–11.52 ticks/second. This is a budget setting, not a claim of exactly 12 rendered frames every second.

## Sustained maximum speed

| Seed | Steps | Actual steps/s | Active tick p50 / p99 (ms) | Score / level | Interventions |
| --- | --- | --- | --- | --- | --- |
| 101 | 600 | 24.94 | 39.67 / 54.64 | 66 / 7 | 0 |
| 102 | 600 | 24.92 | 39.50 / 72.03 | 66 / 7 | 0 |
| 103 | 600 | 26.66 | 37.23 / 52.52 | 66 / 7 | 0 |
| 104 | 600 | 26.75 | 37.26 / 46.10 | 66 / 7 | 0 |

Combined uncapped model-inference p50 / p95 / p99 was **30.30 / 33.50 / 36.44 ms**. Complete active-tick p50 / p99 was **37.97 / 54.20 ms**. An active tick includes planning, synchronized inference, Rich composition, truecolor ANSI serialization and the game update. No pacing delay is inserted.

Scores are equal across seeds because a surviving bird passes pipes on a schedule fixed by pipe spacing; seeds change where each gap is, not when it arrives.

## Fixed-budget stability

| Seed | Target FPS | Steps | Actual steps/s | Active tick p99 (ms) | Budget misses |
| --- | --- | --- | --- | --- | --- |
| 101 | 12.0 | 600 | 11.51 | 67.05 | 0 (0.00%) |
| 102 | 12.0 | 600 | 11.50 | 66.93 | 0 (0.00%) |
| 103 | 12.0 | 600 | 11.52 | 62.46 | 0 (0.00%) |
| 104 | 12.0 | 600 | 11.50 | 60.75 | 0 (0.00%) |

The passing test had **no late active ticks out of 2,400**. The longer 15 FPS test passed its first seed (5/600 late, 0.83%) and failed its second (7/600 late, 1.17%), so it was stopped, as the method specifies. All ticks wait for a fresh model result, so slow inference reduces the game rate instead of executing stale actions.

## Short sweep

Each candidate below ran 120 ticks with seed 7. A short passing probe selects a candidate for the longer four-seed test; it is not sufficient by itself to claim stability. All games survived, including timing failures.

| Target FPS | Actual steps/s | Active tick p99 (ms) | Budget misses | Short sweep |
| --- | --- | --- | --- | --- |
| 10.0 | 9.57 | 63.11 | 0.00% | pass |
| 12.0 | 11.06 | 71.93 | 0.00% | pass |
| 15.0 | 14.20 | 62.79 | 0.83% | pass |
| 18.0 | 17.18 | 58.81 | 1.67% | fail |
| 20.0 | 19.24 | 58.71 | 1.67% | fail |
| 25.0 | 24.57 | 42.33 | 42.50% | fail |
| 30.0 | 25.43 | 77.79 | 97.50% | fail |
| 35.0 | 24.74 | 57.38 | 100.00% | fail |
| 40.0 | 27.06 | 47.66 | 100.00% | fail |
| 45.0 | 26.28 | 58.83 | 100.00% | fail |
| 50.0 | 25.35 | 57.71 | 100.00% | fail |
| 60.0 | 25.69 | 49.84 | 100.00% | fail |

As in the Snake sweep, measured latency varies over time, and a short passing probe (15 FPS here) can fail the longer test. Normal desktop activity, scheduling and clock behavior were not isolated experimentally. We report the achieved rate, all samples and the highest passing **tested** setting.

## What the stability result includes

Laya receives planner features, including which actions collide, which lead to an unavoidable crash and which safe action keeps the bird closest to the next gap center. It computes real probabilities. The lookahead safety layer may correct execution while retaining the raw probability bars and counting every intervention. The checkpoint was not trained on Flappy here. [Exact UI metric meanings and policy behavior](FLAPPY_DEMO.md#what-the-ai-does).

A separate raw top-1 control ran seeds 101–103 for 200 ticks each with the execution shield disabled. All three survived, each with score 17. The raw control still supplies planner features. Its survival over this horizon does not establish indefinite unshielded survival or unaided board reasoning.

The published showcase is a separate **100.07-second actual TTY run** (104 × 35 pseudo-terminal, true color), at a 12 FPS target: **1,143 ticks, score 134, level 14, zero deaths and zero interventions**. Achieved throughput was 11.42 ticks/second, with mean inference of 38.02 ms. Every recorded board and action is checked by deterministic replay in the test suite. The [30-second video](assets/flappy-demo.mp4), [GIF](assets/flappy-demo.gif) and [poster](assets/flappy-preview.png) are rendered from it.

A separate optimized maximum-speed TTY recording completed **515 ticks in 20.03 seconds (25.71 ticks/second)** with `--optimize --max-speed`, score 56, level 6, zero deaths and zero interventions. This includes the actual terminal stream writes. Its [15-second original-speed video](assets/flappy-fast.mp4) and [complete recording](../benchmarks/results/flappy-fast.jsonl) are included alongside the slower presentation clip. The paired eager/optimized study in [Snake optimization](SNAKE_OPTIMIZATION.md) was not repeated for Flappy.

## Measurement boundaries and provenance

- Apple M1; macOS 26.6.2, Python 3.12.14.
- MLX 0.32.2, NumPy 2.5.3, Rich 15.0.0, tokenizers 0.23.2, huggingface-hub 1.32.0.
- Original FP16 weights, with no quantization or custom kernel. Upstream model revision: `052592a15d198d9ad47da779604259b10b47b7aa`.
- Weight SHA-256 from the previously verified published manifest: `7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1`.
- Each tick calls `Agent.predict` once with three batched questions. Inference timing includes tokenization, synchronized MLX evaluation and output conversion.
- The benchmark explicitly enables true color, independent of `NO_COLOR`, and serializes Rich output into an in-memory stream. The terminal emulator's screen painting, model loading and warmup are excluded.
- The [report](../benchmarks/results/flappy.json) stores configuration, environment, source fingerprint and every per-step probability, execution decision and timing.
