# Laya-MLX

**Open-weight typed decisions, running natively on Apple Silicon.**

[中文](https://github.com/mizorewww/laya-mlx/blob/main/README.zh-CN.md) · [Usage reference](https://github.com/mizorewww/laya-mlx/blob/main/docs/USAGE.md) · [Benchmarks](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) · [Hugging Face weights](https://huggingface.co/aac6fef/laya-mlx) · [Changelog](https://github.com/mizorewww/laya-mlx/blob/main/CHANGELOG.md)

Laya answers constrained questions (pick an option, score against a rubric, judge true or
false) with calibrated probabilities from one bidirectional forward pass. It generates no
tokens and returns no JSON to parse. This package runs the original
[Laya](https://github.com/NandhaKishorM/laya) weights in [MLX](https://github.com/ml-explore/mlx),
without PyTorch, Transformers or a cloud API.

**13.4 ms** median for a short English decision, **7.4 ms** with the multilingual checkpoint (M3 Max, FP16).

![Laya MLX playing Snake: actual decisions at original speed](https://raw.githubusercontent.com/mizorewww/laya-mlx/main/docs/assets/snake-demo.gif)

## Quick start

Requires Apple Silicon, macOS 14+ and Python 3.11+.

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "Payments, invoices, refunds, and duplicate charges.",
                "technical": "Broken features, errors, and product troubleshooting.",
                "sales": "Pricing, upgrades, and new purchases.",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
print(result["answers"]["department"]["choice"])  # billing
```

Describing each option, as above, works much better than bare labels. The first load
downloads the checkpoint; later inference is fully local.

## Question types

| Type | You provide | You get |
|---|---|---|
| `choice` | Named options, as a list or a label → description dict | Probability per option |
| `score` | Ordered rubric levels | Probability per level and the expected level |
| `noul` | A yes/no proposition | P(true) |

States can be text, JSON-like dicts or conversation lists. Each answer carries calibrated
probabilities and confidence fields; confidence does not guarantee accuracy on a new task.

## Checkpoints

| Pre-converted MLX (FP16) | Original | Parameters | Context | Use |
|---|---|---:|---:|---|
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) | `convaiinnovations/laya` | 421M | 512 | English |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | `convaiinnovations/laya-multilingual` | 322M | 1,024 | Multilingual input |
| [`aac6fef/laya-typed-decisions-mlx`](https://huggingface.co/aac6fef/laya-typed-decisions-mlx) | `convaiinnovations/laya-typed-decisions` | 421M | 1,024 | Upstream typed-decisions workflows |

`laya.load` accepts either column. Context counts instructions, options and state together.
Each published checkpoint includes its model card, validation results, provenance and
checksums ([pinned revisions and hashes](https://github.com/mizorewww/laya-mlx/blob/main/benchmarks/results/hub-publication.json)).

## What else it does

All of these are covered in the [usage reference](https://github.com/mizorewww/laya-mlx/blob/main/docs/USAGE.md):

- **Language routing:** `Router()` detects the language, picks a checkpoint and keeps up to two resident. You can register your own checkpoints.
- **Batches and long documents:** `predict_batch` shares forward passes across identical question schemas, and long-text scanning keeps answer attribution.
- **Confidence gate:** `min_confidence=` marks low-confidence answers for review without changing them.
- **Large choice sets:** `predict_shortlist` (embedding pre-filter) and `predict_tournament` (no embedder).
- **Throughput options:** `compile=True`, `pad_to_multiple=16` and `cache_prompts=True` for repeated workloads.
- **Tools and services:** CLI (`laya-mlx predict`), HTTP (`laya-mlx-serve`), MCP (`laya-mlx-mcp-server`), evaluation and calibration, and LangChain/LlamaIndex/CrewAI/Agno integrations, installed as extras.
- **Export:** `laya-mlx convert` writes a standalone MLX checkpoint.

## Optional hosted decisions through Monid

For hosted TypeSafe typed decisions, Monid offers one API key, pay per call.
Set `MONID_API_KEY` and explicitly select the `monid` integration runner:

```python
from laya_mlx.integrations.monid import MonidRunner

result = MonidRunner().predict(
    "I was billed twice. Please refund the duplicate.",
    {"refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}},
)
```

You can also pass `MonidRunner()` as `agent=` to framework integrations. This sends
state and questions to hosted TypeSafe through Monid; answers may differ from local
Laya. Local inference remains the default, including when the key is set. The runner
accepts TypeSafe question schemas and an optional TypeSafe `model`; local model
controls and hooks are unsupported. HTTP and provider errors raise `RuntimeError`,
with provider errors using the existing `Laya server error` message format.
[Monid docs](https://docs.monid.ai?utm_source=github&utm_medium=oss-pr&utm_campaign=mizorewww-laya-mlx) ·
[Get a key](https://monid.ai?utm_source=github&utm_medium=oss-pr&utm_campaign=mizorewww-laya-mlx&aid=OSS-PR).

## Performance

M3 Max (40-core GPU, 128 GiB), FP16, end to end:

| | Laya 421M | Multilingual 322M |
|---|---:|---:|
| One short question, P50 / P95 | **13.42 / 13.92 ms** | **7.39 / 7.79 ms** |
| 50-question throughput | **146.8 q/s** | **395.0 q/s** |
| Peak MLX allocation, one question | 943.6 MiB | 687.6 MiB |

Timings include prompt preparation, tokenization, inference, calibration and formatting;
model loading is excluded. The throughput row uses `batch_size=64` (the default is 16).
Latency varies with input length, question count and system load.

All three checkpoints matched upstream's selected answer on **63/63 validation questions in
FP32 and FP16** (378/378 comparisons), with deterministic repeated calls and no measured memory
growth. That measures port fidelity on those fixtures, not task accuracy.
[Method, every timing sample and probability errors](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md).

## Snake demo

The GIF above is a real local run at original speed: every move asks Laya three questions, and a
visible safety layer can override unsafe moves.

```bash
pip install 'laya-mlx[demo]'
hf download aac6fef/laya-multilingual-mlx
laya-snake                      # paced; add --max-speed to remove pacing
laya-snake --optimize --max-speed
```

The terminal must be at least 104 × 35. Space pauses, ↑/↓ changes speed, R resets, Q quits.
With `--optimize`, the measured M3 Max run reached **75.4 moves/s over 2,400 moves** with zero
deaths. [Demo guide](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) ·
[speed and stability](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_BENCHMARKS.md) ·
[optimization](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_OPTIMIZATION.md).

## Relationship to upstream

This is an independent port, not an official Convai Innovations release. Prompt formatting,
calibration, routing, output schema and host APIs follow
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) **v0.4.17**; the encoder and decision
heads are reimplemented in MLX. CUDA, ONNX, PyTorch backend controls and training are out of scope.
RLCD training and fine-tuning stay upstream.
[Native boundaries](https://github.com/mizorewww/laya-mlx/blob/main/docs/USAGE.md#upstream-compatibility) ·
[v0.4.17 migration record](https://github.com/mizorewww/laya-mlx/blob/main/docs/MIGRATION_0_4_17.md) ·
[all documentation](https://github.com/mizorewww/laya-mlx/blob/main/docs/README.md).

## Contributing

Bug reports and pull requests are welcome. Upstream-compatible fixes take priority over new
features. See [CONTRIBUTING.md](https://github.com/mizorewww/laya-mlx/blob/main/CONTRIBUTING.md)
for the development setup, tests, benchmarks and release process.

## License

Apache-2.0; see [LICENSE](https://github.com/mizorewww/laya-mlx/blob/main/LICENSE) and
[NOTICE](https://github.com/mizorewww/laya-mlx/blob/main/NOTICE). Laya and its pretrained weights
are by Convai Innovations and upstream contributors. The neural architecture follows Laya and
Hugging Face ModernBERT.
