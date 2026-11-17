# Demo media and provenance

Shareable recordings of the Snake demo and where each one came from.

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

## Optimized run

The optimized complete Snake loop measured **75.40 moves/second over 2,400 moves**, with zero deaths, 2 safety interventions and 2,400/2,400 executed-action agreement with the paired eager control. It was about **6.5% faster in that run**. This includes planning, inference, Rich composition, ANSI serialization and game updates, but excludes the terminal emulator's painting.

See the [optimization report](SNAKE_OPTIMIZATION.md). The 7.39 ms headline describes the separate one-question API fixture; it is not the frame time of this three-question Snake demonstration. Neither result is a cloud-API comparison or evidence of unaided Snake reasoning.
