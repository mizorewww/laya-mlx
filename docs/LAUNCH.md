# Release assets and suggested copy

These are ready-to-share files and draft text. No social-media post has been submitted.

| Asset | Format | Timing |
|---|---|---|
| [Demo video](assets/snake-demo.mp4) | 1920 × 1080, H.264, 30 video FPS | 30 seconds, original wall-clock speed |
| [README GIF](assets/snake-demo.gif) | 1040-pixel-wide animated GIF | 15 seconds, original wall-clock speed |
| [Maximum-speed video](assets/snake-fast.mp4) | 1920 × 1080, H.264, 30 video FPS | 15 seconds, original speed, optimized live run |
| [Poster](assets/snake-preview.png) | 1920 × 1080 PNG | Actual recorded state at approximately 85 seconds |
| [Source recording](../benchmarks/results/snake-showcase.jsonl) | JSONL | Entire 100-second actual TTY run |
| [Provenance](assets/snake-demo.json) | JSON | Source hash, model, timestamps and renderer hash |

The source run reached score 40 and length 46, with 1,144 real inference-driven moves, zero deaths and zero safety interventions. Its target was 12 decisions/second for legibility. The video is a render of the recorded terminal cells, identified on-screen as `RECORDED RUN · 1×`.

The extra maximum-speed clip comes from a separate 20.01-second truecolor TTY run with `--optimize --max-speed`: **1,296 moves at 64.77 moves/second**, score 44, length 50, zero deaths and zero interventions. It includes writing the live terminal stream; it is not time-compressed. [Source](../benchmarks/results/snake-fast.jsonl) · [Provenance](assets/snake-fast.json).

## Short English draft

> A local AI that returns probabilities, not generated text.
>
> Laya-MLX runs open-weight typed decision models on Apple Silicon. Watch a 322M model play Snake with a visible cycle safety layer: real probabilities, measured latency, 0 output tokens, no inference API.
>
> One-question API benchmark: 7.39 ms p50 on M3 Max.
>
> `pip install laya-mlx`
>
> Code, weights and reproducible measurements: https://github.com/mizorewww/laya-mlx

## 中文草稿

> 让模型直接选方向，而不是先生成一段文字。
>
> Laya-MLX：在 Mac 上本地运行的开放权重决策模型。这个 3.22 亿参数的贪吃蛇 demo，每一步都显示真实方向概率、推理耗时和安全层接管次数。
>
> 0 个输出 token，无推理 API。M3 Max 单问题基准 P50 为 7.39 ms。
>
> `pip install laya-mlx`
>
> 代码、权重和原始 benchmark：https://github.com/mizorewww/laya-mlx

## Separate performance follow-up

The optimized complete Snake loop measured **75.40 moves/second over 2,400 moves**, with zero deaths, 2 safety interventions and 2,400/2,400 executed-action agreement with the paired eager control. It was about **6.5% faster in that run**. This includes planning, inference, Rich composition, ANSI serialization and game updates, but excludes the terminal emulator's painting.

Use the [optimization report](SNAKE_OPTIMIZATION.md) when sharing that number. The 7.39 ms headline describes the separate one-question API fixture; it is not the frame time of this three-question Snake demonstration. Neither result is a cloud-API comparison or evidence of unaided Snake reasoning.

## Frogger assets

| Asset | Format | Content |
|---|---|---|
| [Demo video](assets/frogger-demo.mp4) | 1920 × 1080, H.264, 30 video FPS | 30 seconds, original wall-clock speed |
| [GIF](assets/frogger-demo.gif) | 1040-pixel-wide animated GIF | 15 seconds, original wall-clock speed |
| [Maximum-speed video](assets/frogger-fast.mp4) | 1920 × 1080, H.264, 30 video FPS | 15 seconds, original speed, optimized live run |
| [Poster](assets/frogger-preview.png) | 1920 × 1080 PNG | Actual recorded state at approximately 75 seconds |
| [Source recording](../benchmarks/results/frogger-showcase.jsonl) | JSONL | Entire 100-second actual TTY run |
| [Provenance](assets/frogger-demo.json) | JSON | Source hash, model, timestamps and renderer hash |

Both Frogger recordings were made on an Apple M1: the showcase at a 12 FPS target (1,144 moves, 21 crossings, level 8, 1 shield intervention) and the maximum-speed clip with `--optimize --max-speed` (**356 moves at 17.76 moves/second**, 10 crossings, 1 intervention). Neither had a death. [Frogger speed and stability](FROGGER_BENCHMARKS.md).
