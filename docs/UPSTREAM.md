# Upstream compatibility

Host behavior follows [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) **v0.4.34**
(`80a1da1fa5448ea12163f4d977935625a0aea670`). CI checks that commit out into `.upstream`, runs its
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
  pooling.

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

These fix defects that upstream v0.4.34 still has. Each keeps upstream's public API and is pinned
by its own regression tests.

| Behavior | Upstream v0.4.34 | MLX | Tests |
|---|---|---|---|
| `Router.predict_batch` grouping | Schemas serialized with `json.dumps`, so key `1` and `"1"` (or `True` and `"true"`) share a group and later requests receive the first request's key types | Grouping signature keeps Python key types and insertion order; equal schemas still share a batch | `tests/test_batch_key_types.py` |
| Serial hook timeout (`hooks_concurrent=False`) | The caller releases the lock when a hook times out while the hook is still running, so later serial hooks can overlap it | The worker holds the lock until the hook exits; `hooks_timeout` includes the lock wait, and a hook whose deadline passes while waiting is never invoked | `tests/test_hook_serialization.py` |

When an upstream release fixes one of these, adopt upstream's version and remove the row.

## v0.4.17 → v0.4.34 file inventory

`git diff v0.4.17 v0.4.34 -- laya` touches 27 files (2,779 additions, 185 deletions). Host files
were merged three-way from upstream's diff; the port differs from upstream in the same places as
before, plus the vega boundary in `router.py`.

| Upstream file | Port | Scope |
|---|---|---|
| `laya/__init__.py` | — | Version bump and `VegaAgent` export; not ported. |
| `laya/agent.py` | `runtime.py`, `agent.py` | `predict_batch` validates `batch_size` first. `vegaml` parameter accepted and refused. CUDA AMP and TileLang changes not applicable. |
| `laya/calibrate.py` | `calibrate.py` | Vectorized record validation, threshold selection and binning fit; wide integers in `binning_map` values are a named `ValueError`. |
| `laya/cli.py` | `router_cli.py` | `--image`, `--audio`, `--video` forward to `Router.predict`. |
| `laya/common.py` | `common.py` | `check_batch_size`; `window_batch_cap` uses it; a temperature wider than a double falls back to 1.0 and is reported. CPU fast paths not ported. |
| `laya/confidence.py` | `confidence.py` | Wide integers are not confidences; Jev weights whose sum overflows are rescaled. |
| `laya/evals.py`, `laya/evals_cli.py` | same | Datasets, reports and agreement runs saved with a BOM are read. |
| `laya/hooks.py` | `hooks.py` | `OptionStabilityHook` drops a rename probe that renders the original question. |
| `laya/integrations/_guard.py` | same | Non-finite and wide numeric guard values fall back or are refused. |
| `laya/integrations/agno.py`, `crewai.py`, `llamaindex.py` | same | A decision naming an agent or choice outside the list is refused, not indexed. |
| `laya/integrations/langchain.py` | same | Opt-in `decision_policy="native"` follows the answer's `route`; guardrail and triage read nested answers through the shared guard. |
| `laya/integrations/tools.py` | same | `LayaToolSelector` resolves a per-bucket `min_confidence` map. |
| `laya/lang.py` | `lang.py` | Lazy state traversal and a string fast path; results unchanged. |
| `laya/mcp/remote.py` | same | Media arguments are refused in remote mode. |
| `laya/mcp/server.py`, `laya/mcp/tools.py` | same | `explain` on `laya_route` and `laya_route_batch`; `laya_predict_long` refused in remote mode; unknown batch task is `invalid_task`; non-string question type is `invalid_questions`; `laya_tournament` reports router construction errors. |
| `laya/onnx_agent.py` | — | ONNX duplicate of the `batch_size` check; not ported. |
| `laya/policy.py`, `laya/vega.py`, `laya/video.py` | — | vegaml engine, media readers and signed policies; not ported. |
| `laya/revisions.py` | `revisions.py` | A `LAYA_SHA256_DIGESTS` nested too deeply to parse is a `ValueError`. |
| `laya/router.py` | `router.py` | Vega lane (see above); `on_route` dispatch skipped when no hook implements it; `route_batch` fast path; checkpoint source pairs validated at registration; legacy batch keyword fallbacks compose. |
| `laya/serve.py` | `serve.py` | `POST /v1/route`; one warning per request when state tokens were dropped; media keys refused with 422; `LAYA_EXTRA_MODELS` source pairs validated at startup. |
| `laya/structured.py` | `structured.py` | A non-array `anyOf`/`oneOf` is a `SchemaError`. |

Changes to upstream's other SDKs, training and research tools, notebooks and deployment files are
outside this package.

## How upstream assertions run

`tests/test_official_host.py` runs 34 pinned upstream suites through
`scripts/run_official_test.py`, which rewrites the `laya` namespace and mock targets and nothing
else: assertions, expected values, public model names and timing thresholds are upstream's. New
for this pin are `router_vega`, `langchain_native_policy` and `mcp_diagnostics`.
`tests/test_official_contract.py` adds AST equality for host methods, differential runs against
upstream on identical inputs, and the native MCP suite.

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

## Validation for 0.7.0

Local, Apple M-series, on the release source:

| Check | Result |
|---|---|
| `pytest -q tests/test_official_host.py tests/test_official_contract.py` | 77 passed (34 upstream suite wrappers plus contract tests) |
| `pytest -q tests/test_parallel_layout.py` | 11 passed |
| Remaining native and integration tests | 231 passed, 2 skipped |
| `scripts/audit_public_api.py --package laya_mlx` | 351 surface entries, 0 unclassified differences |
| `ruff check .`, `ruff format --check .`, `compileall` | Passed |
| `python -m benchmarks.validate --repeats 100` against the v0.4.34 reference | 63/63 argmax agreement for all three checkpoints in FP32 and FP16; 100 stable calls each. Largest probability error 5.2e-06 (FP32) and 0.0054 (FP16) |
| `python -m build`, `twine check --strict` | Passed for wheel and sdist |
| `scripts/wheel_smoke.py` from a clean inference-only environment | Passed for all three checkpoints, without PyTorch or Transformers installed |

Performance and memory were not re-measured; [BENCHMARKS.md](../BENCHMARKS.md) describes the
revisions it names. Remote CI runs the same suites on Linux (host only) and macOS 26.

## Reproduce

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev,serve,mcp,structured]'
git clone https://github.com/NandhaKishorM/laya .upstream
git -C .upstream checkout 80a1da1fa5448ea12163f4d977935625a0aea670
.venv/bin/pytest -q tests/test_official_host.py tests/test_upstream_0417.py tests/test_upstream_0434.py
.venv/bin/python scripts/audit_public_api.py --package laya_mlx
```

That is the Linux host check. Native tests, real-checkpoint validation and benchmarks need Apple
silicon; see [CONTRIBUTING.md](../CONTRIBUTING.md).
