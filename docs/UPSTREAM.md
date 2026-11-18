# Upstream compatibility

Host behavior follows [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) **v0.4.35**
(`b1d656952f0f9581ea691a70cb6f38abf3eb13a9`). CI checks that commit out into `.upstream`, runs its
unchanged host suites against this package, and audits the public API against it. This page is
the current record; what each release changed is in the [changelog](../CHANGELOG.md).

## What is native

MLX owns model construction, weight conversion, tensor execution, streams, `compile`, dtypes, row
batching and prefix caching. Everything else (prompts, validation, routing, hooks, calibration,
evaluation, CLI, HTTP, MCP and the framework integrations) is upstream's host code under the
`laya_mlx` namespace, with upstream `cli.py` as `router_cli.py` and the host half of `agent.py` in
`runtime.py`.

- `Agent` defers MLX, model and tokenizer imports until construction or execution, so host
  validation and signature inspection run on Linux. Construction and forward require MLX; no fake
  MLX module or numerical emulation is installed for tests.
- `backend` and `dtype_for()` report MLX. `set_backend()` raises; TileLang `accelerate()` returns
  `False` (raises with `strict=True`).
- `embed_fn_from_agent` validates `tok`/`model.encoder` before importing MLX and keeps native mean
  pooling. Like [upstream's tokenizer call](https://github.com/NandhaKishorM/laya/blob/b1d656952f0f9581ea691a70cb6f38abf3eb13a9/laya/shortlist.py#L271-L278),
  it truncates through the tokenizer, preserving special tokens, on a private copy of it.
  The native keyword `bucket_by_length=True` groups similar lengths within `batch_size` and
  restores input order, reducing padding for long queries mixed with short labels. It defaults
  to `False`: batch-shape rounding can reorder near-tied candidates, particularly in FP16.
- The final decision-head layer keeps full QKV projections and attention, but computes its output
  projection and FFN only for CLS and option-marker positions, which are the outputs inference
  reads. This preserves the mathematical dependencies; changed projection shapes can affect
  floating-point rounding. `tests/test_selected_head.py` checks selected outputs against the full
  head, including zero head layers, padding, FP32/FP16 and compilation.

## Not included

These exist upstream and are refused by name here rather than emulated:

| Upstream feature | Here |
|---|---|
| PyTorch/CUDA/TileLang backends, `torch.compile` controls, ONNX (`onnx_path`, `ONNXAgent`) | `ValueError` naming official laya |
| Training and fine-tuning CLI | Not shipped |
| vegaml engine: `Agent(vegaml=...)`, `load(vegaml=...)`, `VegaAgent`, `laya.vega`, `laya.video`, `laya.policy` | `vegaml=None`/`False` are accepted; any other value raises `ValueError`. `VegaAgent` is not exported and `vegaml`/`cryptography` are not dependencies |
| Router vega lane built by the Router (`Router(vega=True)`, a reader name, or `"auto"` with vegaml installed) | `"auto"` never turns the lane on. `vega=True` raises `ImportError`. A reader name routes like upstream but raises `ImportError` when loaded |
| CPU sliding-window and packed-head fast paths in `DecisionModel`, `LAYA_CUDA_AMP` parsing | PyTorch execution details; the MLX model is unchanged |
| `agno<4` | The extra still pins `agno<2`; newer Agno majors were not tested here |

The lane itself is ported: `Router(vega=agent)` accepts an already built agent, for example
official `laya.Agent(vegaml=True)`, and then routes `image=`/`audio=`/`video=`, states past
`vega_over_tokens` and questions with fitted heads to it exactly as upstream does. Without a lane,
media arguments raise `ValueError`, and `laya-mlx-serve` and `RemoteRouter` refuse media as upstream.

## Deliberate divergences

These fix defects that upstream v0.4.35 still has. Each keeps upstream's public API and is pinned
by its own regression tests.

| Behavior | Upstream v0.4.35 | MLX | Tests |
|---|---|---|---|
| Long-text window tokens | Decodes and re-encodes each window, which can add tokens and truncate its tail | Window objects reuse their original tokens; plain strings returned by hooks are re-encoded. Attribution still follows upstream's text-equality check | `tests/test_window_embedding_integrity.py` |
| Embedding cache invalidation | In-flight embeddings can repopulate a cleared cache | A generation counter prevents old calls from writing into the new cache | `tests/test_window_embedding_integrity.py` |
| Parallel checkpoint warmup | Synthetic batches omit option layout tensors | Include shared position IDs and isolated option IDs so native parallel models can warm up, including automatic compiled warmup | `tests/test_warmup.py` |
| `Router.predict_batch` grouping | Schemas serialized with `json.dumps`, so key `1` and `"1"` (or `True` and `"true"`) share a group and later requests receive the first request's key types | Grouping signature keeps Python key types and insertion order; equal schemas still share a batch | `tests/test_batch_key_types.py` |
| Serial hook timeout (`hooks_concurrent=False`) | The caller releases the lock when a hook times out while the hook is still running, so later serial hooks can overlap it | The worker holds the lock until the hook exits; `hooks_timeout` includes the lock wait, and a hook whose deadline passes while waiting is never invoked | `tests/test_hook_serialization.py` |

When an upstream release fixes one of these, adopt upstream's version and remove the row.

## v0.4.34 → v0.4.35 file inventory

`git diff v0.4.34 v0.4.35 -- laya tests` touches 3 files (96 additions, 1 deletion). Nothing in
the host code this package mirrors changed; the release adds `Laya.route()` to the TypeScript
HTTP SDK, which calls the `POST /v1/route` endpoint that `laya-mlx-serve` already serves.

| Upstream file | Port | Scope |
|---|---|---|
| `laya/__init__.py` | — | Version bump; not ported. |
| `tests/test_hooks_api.py` | runs unchanged | Two new checks that the TypeScript `RouteSnapshot` and `RouteOptions` types stay outside the root `__all__`. |
| `tests/sdk_route_server_fixture.py` | — | Model-free HTTP fixture for the TypeScript SDK's integration tests; it imports `laya.serve` directly and is not a Python suite. |

The host files were last merged three-way at v0.4.34. Changes to upstream's other SDKs (here the
TypeScript client, its tests and docs), training and research tools, notebooks and deployment
files are outside this package.

## How upstream assertions run

`tests/test_official_host.py` runs 34 pinned upstream suites through
`scripts/run_official_test.py`, which rewrites the `laya` namespace and mock targets and nothing
else: assertions, expected values, public model names and timing thresholds are upstream's. This pin
adds no suites; `hooks_api` gained two export checks.
`tests/test_official_contract.py` adds AST equality for host methods, differential runs against
upstream on identical inputs, and the native MCP suite.
The AST checks normalize only the two token-preserving expressions in `_encode_state` and
`predict_long`; the rest of both methods remains pinned. The MCP cache fixture translates the
Torch tokenizer to the native interface, including cloning and truncation, with shared call
records so all upstream cache assertions remain unchanged.

Excluded as backend-specific:

- ONNX blocks in the hooks, hooks API, evals API, structured and confidence suites; `VegaAgent` in
  the hooks API signature loop.
- Six TileLang/PyTorch padding checks in the confidence suite.
- Four serve tests and one MCP test asserting Torch device, thread or status behavior.
- The training CLI help assertion.
- One `router_vega` test that needs vegaml installed to turn the lane on by itself, and one
  `langchain_native_policy` test that imports `laya.vega`.

Upstream's full `test_calibrate.py`, `test_revision_pinning.py`, `test_predict_long.py`,
`test_predict_batch.py`, `test_temperature_loading.py` and `test_shortlist.py` mix host assertions
with Torch or ONNX fixtures and are not run whole. Their changed host behavior is covered by
`tests/test_upstream_0417.py`, `tests/test_upstream_0434.py` and the native tests.

Some upstream assertions read upstream's docs and examples, so the pinned checkout must be
complete, not only `laya/` and `tests/`.

### Idle-window timing

Upstream's `test_idle_window_starts_after_worker_finishes_even_on_cancel_or_failure` observes a
100 ms no-unload window against a 200 ms idle threshold and can fail on a busy hosted runner. The
same assertion failed against unmodified upstream on the same runner
([run](https://github.com/mizorewww/laya-mlx/actions/runs/39355649931)). The assertion and timeout
are unchanged here; CI runs host contracts in their own process first and, on failure, reruns the
original upstream test for comparison.

## Validation for 0.7.1

Automated port run (macOS arm64 runner, `LAYA_MLX_TEST_DEVICE=cpu`) on the release source:

| Check | Result |
|---|---|
| `pytest -q tests/test_official_host.py tests/test_official_contract.py` | 77 passed (34 upstream suite wrappers plus contract tests) |
| `pytest -q tests/test_parallel_layout.py` | 11 passed |
| `pytest -q tests/test_runtime.py` | 25 passed |
| Remaining native and integration tests | 251 passed, 1 skipped |
| `scripts/audit_public_api.py --package laya_mlx` | 351 surface entries, 0 unclassified differences |
| `ruff check .`, `ruff format --check .`, `compileall` | Passed |
| `python -m build` | Built wheel and sdist |

Real-checkpoint validation (`python -m benchmarks.validate --repeats 100`) and the wheel smoke
test from a clean inference-only environment were not run for this release; they run locally on
Apple silicon. The last real-checkpoint run was for 0.7.0 against the v0.4.34 reference: 63/63
argmax agreement for all three checkpoints in FP32 and FP16, 100 stable calls each. Neither the
model code nor the upstream engine changed between v0.4.34 and v0.4.35.

Performance and memory were not re-measured; [BENCHMARKS.md](../BENCHMARKS.md) describes the
revisions it names. Remote CI runs the same suites on Linux (host only) and macOS 26.

## Reproduce

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev,serve,mcp,structured]'
git clone https://github.com/NandhaKishorM/laya .upstream
git -C .upstream checkout b1d656952f0f9581ea691a70cb6f38abf3eb13a9
.venv/bin/pytest -q tests/test_official_host.py tests/test_upstream_0417.py tests/test_upstream_0434.py
.venv/bin/python scripts/audit_public_api.py --package laya_mlx
```

That is the Linux host check. Native tests, real-checkpoint validation and benchmarks need Apple
silicon; see [CONTRIBUTING.md](../CONTRIBUTING.md).
