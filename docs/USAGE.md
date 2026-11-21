# Laya-MLX usage reference

The [README](../README.md) covers installation and a first prediction. This page is the
complete reference for the Python API, routing, command-line tools and checkpoint export.

- [Load a checkpoint](#load-a-checkpoint)
- [Ask questions](#ask-questions)
- [Read the result](#read-the-result)
- [Throughput controls](#throughput-controls)
- [Confidence gate](#confidence-gate)
- [Language routing](#language-routing)
- [Large choice sets](#large-choice-sets)
- [Hooks](#hooks)
- [Parallel option layout](#parallel-option-layout)
- [Command line, HTTP and MCP](#command-line-http-and-mcp)
- [Export an MLX checkpoint](#export-an-mlx-checkpoint)
- [Upstream compatibility](#upstream-compatibility)

## Load a checkpoint

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")  # pre-converted FP16
agent = laya.load("convaiinnovations/laya")  # original checkpoint
agent = laya.load("./models/laya", dtype="float32", batch_size=32)
multi = laya.load("convaiinnovations/laya", subfolder="multilingual")
pinned = laya.load(
    "convaiinnovations/laya",
    revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf",
)
```

| Argument | Default | Meaning |
|---|---|---|
| `dtype` | `"float16"` | `"float32"` gives closer numerical agreement with upstream. `"bfloat16"` works but is not in the validation matrix. |
| `device` | MLX default | `"gpu"` (aliases `"mps"`, `"metal"`) or `"cpu"`. |
| `batch_size` | `16` | Maximum question rows per native forward pass; larger requests run in chunks. |
| `revision`, `subfolder` | — | Hugging Face revision pin and subdirectory. |
| `local_files_only` | `False` | Require an existing local cache; never touch the network. |
| `compile`, `pad_to_multiple`, `cache_prompts` | off | See [throughput controls](#throughput-controls). |

The first load downloads the checkpoint; inference afterwards is local. Loading validates
every parameter name and shape. Unsupported encoders and non-default RoPE scaling fail
explicitly. ModernBERT's global/local attention pattern, inclusive sliding-window boundary,
separate local/global RoPE bases and first-layer normalization are preserved.

Probabilities can differ slightly between FP16 and FP32 even when the selected label agrees.
The measured differences are in [BENCHMARKS.md](../BENCHMARKS.md#numerical-parity-and-stability).

**Calibration clamp.** Following upstream v0.3.5, fitted temperatures are clamped to
`[0.5, 5.0]`. The shipped `choice:11+` bucket is 0.1006, which would otherwise sharpen logits
about 10× and report a coin flip as near-certainty. Raw values stay available as
`agent.temperature_raw` and `agent.temperature_by_options_raw`, and a `RuntimeWarning` names
each clamped bucket at load. Fitting new temperatures needs the optional `calibration` extra;
loading a JSON calibration file does not.

## Ask questions

`agent.predict(state, questions)` (alias `system_one`) answers every question in one call.

```python
result = agent.predict(
    "I was billed twice. Please refund the duplicate today.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs and outages",
                "sales": "new purchases",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
```

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | Dict of label → description, or a list of unique labels (scalars allowed) | Probability per label and the selected `choice` |
| `score` | Ordered rubric levels | Probability per level and the expected zero-based level |
| `noul` | Optional; only `false`/`true` keys (Python booleans allowed) | P(true) |

- **States** may be text, a JSON-like dict, or a conversation list. When context fills,
  conversation lists keep their newest tokens; strings and dicts keep their beginning.
- **`noul` labels:** `labels={"false": "no", "true": "yes"}` changes the words the model sees;
  the answer is still P(true). Any other criteria key raises.
- **Validation:** empty instructions, null score levels and a `None` state raise, and the error
  names the question. Non-string instructions keep their Unicode.
- **Batches and long text:** `predict_batch` groups requests with identical question schemas
  into shared forward passes. Grouping keeps Python key types and option order, so integer `1`
  and string `"1"` stay distinct. Long-document scanning follows upstream windowing and
  attribution rules.
- **Structured decisions:** the `structured` extra adds Pydantic schemas for typed answers.

## Read the result

Results keep upstream's schema and four-decimal rounding.

| Field | Meaning |
|---|---|
| `answers[qid]` | `type` and its value: `choice` (label), `score` (expected level) or `noul` (P(true)) |
| `answers[qid].probabilities` | Per option (`choice`) or per level (`score`) |
| `answers[qid].confidence` | Entropy-based for `choice`/`score`; maximum probability for `noul` |
| `answers[qid].answer_confidence` | Maximum calibrated option probability |
| `answers[qid].action.act_probability` | Probability from the action head |
| `usage.input_tokens`, `output_tokens` | Token accounting; `output_tokens` is always 0 |
| `usage.state_tokens`, `state_tokens_dropped` | State length and the largest truncation across questions |
| `usage.truncated`, `truncated_questions` | Which questions lost state context |
| `usage.options` | Present only when option token spans collide: `total`, `distinct`, `tokens_per_option` |

Neither confidence field guarantees accuracy on a new task. `usage.options` reports lost
distinctions between options; it does not recover them.

## Throughput controls

For repeated workloads, opt in when loading:

```python
agent = laya.load("aac6fef/laya-mlx", compile=True, pad_to_multiple=16, cache_prompts=True)
```

- `compile=True` compiles the native forward pass. It has a first-use cost and specializes on
  shape.
- `pad_to_multiple=16` rounds sequence lengths so fewer shapes are compiled; extra padding can
  make some workloads slower.
- `cache_prompts=True` keeps a bounded 128-entry cache of tokenized question prefixes. Every
  question still gets its own encoder computation, so no encoder states or answers are reused.

Constructor `batch_size` bounds tensor rows per forward pass. `predict_batch(batch_size=...)`
groups logical requests. When a native batch needs multiple forward passes, question rows are
grouped by encoded length if this reduces padding, then restored to their original order.
As with other batch-shape changes, floating-point results can vary slightly. Single-chunk and
equal-length batches keep their order. The measured effect on the Snake workload is in
[SNAKE_OPTIMIZATION.md](SNAKE_OPTIMIZATION.md).

## Confidence gate

`Agent.predict` and `Router.predict` accept `min_confidence` as a scalar or per-bucket map:

```python
result = agent.predict(
    state,
    questions,
    min_confidence={"choice:2": 0.8, "choice:11+": 0.95, "default": 0.7},
)
for answer in result["answers"].values():
    if answer["abstention"] != "passed":
        print("Needs review", answer)
```

Buckets are `2`, `3-5`, `6-10` and `11+`, prefixed with `choice:`, `score:` or `noul:`. A missing
bucket falls back to `default`, then zero. Answers gain `abstention` (`passed`, `abstained` or
`unevaluated`), `abstention_threshold`, and `low_confidence=True` when below threshold. The gate
compares `answer_confidence`; it never replaces the answer or calls a fallback. Invalid bucket
names and non-finite thresholds fail before inference.

## Language routing

`Router` picks among the three checkpoints and loads them on first use.

```python
from laya_mlx import Router, triage_questions

router = Router(dtype="float16")
result = router.predict({"message": "发票被重复扣款，请退款。"}, triage_questions())
print(result["routing"])  # multilingual

result = router.predict(state, questions, task="typed_decisions")
result = router.predict(state, questions, model="english")
```

| Option | Default | Meaning |
|---|---|---|
| `default` | `"multilingual"` | Fallback for undecided text, including very short inputs. Use `"english"` for mostly-English traffic. |
| `max_loaded` | `2` | Resident checkpoints; `1` for constrained hosts, `3` or `preload=True` to keep all. |
| `auto_task_detection` | `False` | Detect typed-decisions workflows automatically. |
| `local_files_only` | `False` | Resolve checkpoints from the local cache only. |

`route()` inspects a request without loading weights. Explicit `model=`, `task=` and `lang=`
override detection; blank or language-neutral hints fall through to detection.

Language detection is heuristic. Unidentified Latin-script languages (Romanian, Polish, Czech,
Turkish, …) route to the multilingual checkpoint on their non-English letters instead of being
assumed English. `detect_language(state)` returns `language`, `is_english`,
`language_undecided` and `diacritic_rate`. `detect_language(state, explain=True)` adds a
`diagnostic` describing which part of the state decided the result.
All-capital acronyms are not mistaken for French or Spanish. Detection examines nested string
values and mixed scripts.

**Custom checkpoints** register beside the built-ins without loading immediately:

```python
router = Router(models={"billing": "./models/billing-mlx"})
router.register("support", "./models/support-mlx", description="Support classifier")
result = router.predict(state, questions, model="support")
router.unregister("support")
```

Names are case-insensitive and may contain letters, digits, `.`, `_` and `-`. Built-ins and the
current default cannot be unregistered. Use `register` rather than editing `router.models`.
`attach(name, agent)` adds an already-loaded Agent.

**Lifecycle.** A re-entrant lock guards the model registry, so concurrent threads share one build
of a checkpoint; inference itself is not serialized. Cold construction runs outside the lock, so
resident models stay usable while another loads. `unload(name)` waits for that checkpoint's build;
`unload()` waits for all. Replacing a source invalidates the old resident model. Eviction releases
MLX's process-wide free-buffer cache; agents still held by callers stay valid.

**Email helpers** (`laya_mlx.email`) clean multilingual mail footers and quoted replies. They are
applied only when you call them.

## Large choice sets

Choice options share one `head_max_len` token budget, so hundreds of labels leave only a few
tokens each. Two opt-in strategies reduce the set first.

```python
# Embedding shortlist: keep the top k labels by cosine similarity, then predict once.
embed_fn = laya.embed_fn_from_agent(agent)  # mean-pools the loaded encoder
result = laya.predict_shortlist(agent, state, questions, embed_fn, k=20)
print(result["shortlist"])

# Tournament: no embedding model; winners of each group advance.
result = laya.predict_tournament(router, state, questions, group_size=16)
print(result["tournament"])
```

A dedicated bi-encoder passed as `embed_fn` usually shortlists better than the decision encoder.
A tournament costs extra passes, follows criteria order when grouping, and can eliminate a good
candidate early. In both cases the final probabilities, confidence and `usage` describe the
**final pass only**.

## Hooks

`Agent` and `Router` accept `hooks=[...]` objects implementing any of `on_route`,
`on_load`, `on_evict`, `on_predict_start`, `on_predict_end` and `on_error`. Async hook methods run
to completion.

- `hooks_timeout` bounds each hook call. Python threads cannot be interrupted, so a hook that
  overruns keeps running in the background and may still change its context.
- With `hooks_concurrent=False`, hooks are serialized. A timed-out hook keeps the lock until it
  exits, and the timeout includes waiting for the lock. A hook whose deadline passes while waiting
  is never called.
- `Router(hooks_raise=False)` turns hook errors and timeouts into warnings.

## Parallel option layout

Checkpoints trained with `"option_layout": "parallel"` work end to end: shared option positions,
option-isolating attention masks, compiled inference, prefix caching and conversion. A missing
setting means `sequential`, which all published checkpoints use; unknown layouts raise.

**Do not edit an existing checkpoint's config to enable parallel layout.** Its weights were trained
with the sequential layout. Permutation invariance is verified within floating-point tolerance on
tiny random models; that is not an accuracy claim. On CPU, parallel-layout inference uses FP32
intermediates to avoid native half-precision kernel failures on macOS 26.

## Command line, HTTP and MCP

```bash
laya-mlx predict --model aac6fef/laya-mlx \
  --state-file examples/state.json --questions examples/questions.json

laya-mlx predict --model aac6fef/laya-multilingual-mlx \
  --state '发票被重复扣款，请退款。' --questions examples/questions.json
```

Input files are read with BOM-aware decoding, and read, encoding and JSON errors are reported before
a model is loaded.

| Command | Extra | Purpose |
|---|---|---|
| `laya-mlx predict` / `convert` | — | One prediction; checkpoint export |
| `laya-mlx-router` | — | Routed predictions from the shell |
| `laya-mlx-serve` | `serve` | Upstream-compatible HTTP service with request-body deadlines and batch token limits |
| `laya-mlx-mcp-server` | `mcp` | Model Context Protocol server |
| `laya-mlx-evals` | — | Evaluation and reliability calibration |
| `laya-snake` | `demo` | Terminal Snake demo ([guide](SNAKE_DEMO.md)) |

`POST /v1/route` on the HTTP service returns the routing decision (`model`, `repo`, `reason`)
without running a model; `"explain": true` adds the language diagnostic. The MCP `laya_route` and
`laya_route_batch` tools take the same `explain` flag. The service logs one warning per request
when a state was truncated.

Framework integrations are available as the `langchain`, `langgraph`, `llamaindex`, `crewai` and
`agno` extras. LangChain's `LayaRouter(decision_policy="native")` follows the `route` field of an
answer (`auto` or `review`) instead of a confidence threshold, for backends that report one. Upstream's standalone HTTP SDK is a separate package and can target `laya-mlx-serve`.

## Export an MLX checkpoint

```bash
laya-mlx convert --model convaiinnovations/laya --dtype float16 --output models/laya-mlx-fp16
laya-mlx predict --model models/laya-mlx-fp16 \
  --state-file examples/state.json --questions examples/questions.json
```

The export contains `model.safetensors`, encoder and agent configs, tokenizer files and
`mlx_config.json`. It never overwrites an existing directory. Conversion renames parameters and
casts dtype only; it does not quantize or retrain. The source checkpoints already store FP16
weights, so FP32 raises arithmetic precision, not weight precision.

## Upstream compatibility

Host behavior (prompts, validation, routing, calibration, evaluation, HTTP/MCP and integrations)
follows [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) v0.5.2
(`e150aa92a0d4059ac621875b39b620847cf1b88a`). The 353-entry public API audit has no
unclassified differences ([audit](UPSTREAM_API_AUDIT.json)).

Native boundaries:

- `backend` and `dtype_for()` report MLX. `set_backend()` is unsupported.
- TileLang `accelerate()` returns `False` (raises with `strict=True`); `deaccelerate()` does not
  disable MLX compilation.
- CUDA, PyTorch compile controls, ONNX execution and training are not provided.
- The vegaml engine is not included: `Agent(vegaml=True)` and `load(vegaml=True)` raise. A Router
  routes images, audio, video, states past `vega_over_tokens` and questions with fitted heads to a
  vega lane only when given a built agent, for example
  `Router(vega=laya.Agent(vegaml=True))` with official laya installed. Otherwise the lane is off.
- Core ML's `compute_units` is not a Router option here.

Deliberate divergences from upstream are listed with their tests in
[UPSTREAM.md](UPSTREAM.md), together with everything that is not included and the file inventory.
