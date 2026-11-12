# Upstream v0.4.17 source migration

The source target moves from upstream v0.4.12 (`ffc142012af63d015fa19f05f29f9c5ff329a2cc`)
to v0.4.17 (`fe9a6c35023a7d94102e7feaaeaae8857d9d1b07`). The migration base is
`ddaf014`. It ships in **0.6.0** ([changelog](../CHANGELOG.md)); the evidence below was
recorded on the 0.5.0 source tree before the version bump. The 0.6.0 release re-ran the
real-checkpoint validation: 63/63 fixtures for all three checkpoints in FP32 and FP16 against the
v0.4.17 reference, with 100 stable calls each.

All applicable Python changes in `git diff ffc1420..v0.4.17 -- laya` are accounted
for below: 26 upstream files, 1,038 additions and 166 deletions. Twenty-four files
have applicable changes; the other two contain the upstream version bump and the
duplicate ONNX implementation. This is a source migration with checked host
behavior. Linux host and macOS native CPU CI passed for source commit
`0e8c0e51f8914908cc99591576ff1deeed205c82`; full-checkpoint GPU qualification was not rerun.

The current machine-readable inventory is [UPSTREAM_PORT.json](UPSTREAM_PORT.json).
The previous manifest is retained as [UPSTREAM_0412_PORT.json](UPSTREAM_0412_PORT.json).
The v0.4.12 qualification, progress, API audit and benchmark artifacts remain
historical evidence, unchanged; they do not qualify this revision.

## Behavior and native boundaries

The optional `monid` integration exports `MonidRunner`, a hosted TypeSafe prediction
runner for explicit `agent=` selection. This extension is outside upstream parity:
it uses TypeSafe question schemas and results rather than local Laya inference.
It requires `MONID_API_KEY` and never changes default runner selection. Local model
controls and hooks are unsupported; no inference or routing behavior is replaced.

`detect_language(..., explain=True)`, `Router.route(..., explain=True)`,
`Router.route_batch(..., explain=True)` and `laya-mlx-router --explain` expose
heuristic language reasons and bounded source paths. Routing overrides still
bypass detection. Diagnostic fields contain no additional input text. Unicode
letter handling, English evidence and Vietnamese diacritics follow upstream.

Long-document answer attribution now records the answer object and its prediction
fields before any end hook. Reordered answers retain their original window;
cloned, edited or shared answers omit window attribution. Reliability annotations
remain permitted. Missing caller answers raise a named error; a question added by
a start hook can be removed consistently by an end hook. These host methods live
in `runtime.py`; the shared upstream attribution helper runs 49 unchanged checks.

Reliability calibration accepts NumPy real/boolean scalars, validates equal lengths
and finite scores, returns plain float cutoffs, and preserves the precision used
by the hook's decision. Renaming options with empty descriptions retains their
semantic label. Integration wrappers now reject malformed response shapes and
numeric fields through their existing error/fallback paths.

The HTTP server uses `LAYA_BODY_READ_TIMEOUT` (10 seconds by default), releases
admission on a 408 body-read timeout, disables FastAPI documentation/schema routes
when `LAYA_API_KEY` is set, and enforces its forward-token budget even with an
explicit caller batch size. `LAYA_USER_AGENT` or `RemoteRouter(user_agent=...)`
sets a validated Latin-1 remote header; the default uses the local package version,
for example `laya/0.6.0`. Evaluation and CLI readers report the input artifact on malformed,
unreadable or excessively nested data. The native CLI accepts UTF-8 BOMs in both
state and questions files, and reports parse/read errors before loading a model.

MLX model construction, weight conversion, tensor execution, streams, compile,
dtypes, row batching and prefix caching retain their native implementations.
`Agent` defers MLX/model/tokenizer imports until construction or execution, and
its exported `DTYPES` still resolves to real MLX dtypes. Host validation and
signature inspection can therefore run on Linux. Clearing an unused router cache
does not import a backend; an already imported MLX cache is still cleared normally.
`embed_fn_from_agent` validates `tok`/`model.encoder` before importing MLX and keeps
its native mean pooling implementation. Real construction and forward require
MLX; no fake MLX module or numerical emulation is installed for tests.

## Deliberate divergences from upstream

These fix defects that upstream v0.4.17 still has. Each keeps upstream's public API and is pinned
by its own regression tests.

| Behavior | Upstream v0.4.17 | MLX | Tests |
|---|---|---|---|
| `Router.predict_batch` grouping | Schemas serialized with `json.dumps`, so key `1` and `"1"` (or `True` and `"true"`) share a group and later requests receive the first request's key types | Grouping signature keeps Python key types and insertion order; equal schemas still share a batch | `tests/test_batch_key_types.py` |
| Serial hook timeout (`hooks_concurrent=False`) | The caller releases the lock when a hook times out while the hook is still running, so later serial hooks can overlap it | The worker holds the lock until the hook exits; `hooks_timeout` includes the lock wait, and a hook whose deadline passes while waiting is never invoked | `tests/test_hook_serialization.py` |

## Complete upstream file inventory

| Upstream file | Port location | Scope |
|---|---|---|
| `laya/__init__.py` | `laya_mlx/__init__.py` | Upstream package version only; retain laya-mlx 0.5.0. |
| `laya/_eval_policy.py` | `laya_mlx/_eval_policy.py` | Unreadable or excessively nested policy files report EvalError. |
| `laya/agent.py` | `laya_mlx/runtime.py` | Track answer identity and prediction snapshots before end hooks; preserve original scan spans after reordering, omit edited/cloned/shared attribution, detect missing answers, normalize probabilities in place. |
| `laya/calibrate.py` | `laya_mlx/calibrate.py` | Name invalid target_error values while preserving comparable numeric types. |
| `laya/cli.py` | `laya_mlx/router_cli.py` | Routing --explain; BOM-aware questions/batch inputs; path-specific JSON, encoding, depth and OS errors. |
| `laya/common.py` | `laya_mlx/common.py` | Scalar confidence clipping avoids NumPy dispatch while retaining vector entropy order. |
| `laya/confidence.py` | `laya_mlx/confidence.py` | Jev probability coercion handles OverflowError. |
| `laya/evals.py` | `laya_mlx/evals.py` | Validate dataset row shape/read errors, reject bool confidence, require one batch result per row, report incomparable regression metrics. |
| `laya/evals_cli.py` | `laya_mlx/evals_cli.py` | Explicit accuracy thresholds require the exact metric named. |
| `laya/evidence.py` | `laya_mlx/evidence.py` | BOM-aware evidence artifacts and path-specific nesting failures. |
| `laya/hooks.py` | `laya_mlx/hooks.py` | Validate timeout coercion and reliability data; preserve full soft stability and NumPy calibration cutoffs; keep semantic labels when renaming empty descriptions. |
| `laya/integrations/_guard.py` | `laya_mlx/integrations/_guard.py` | Validate response/answer/numeric/probability shapes with named malformed-response errors. |
| `laya/integrations/agno.py` | `laya_mlx/integrations/agno.py` | Fail closed on malformed remote and nested decisions. |
| `laya/integrations/crewai.py` | `laya_mlx/integrations/crewai.py` | Fail closed on malformed remote and nested decisions. |
| `laya/integrations/langchain.py` | `laya_mlx/integrations/langchain.py` | Reject non-object remote response and answers. |
| `laya/integrations/llamaindex.py` | `laya_mlx/integrations/llamaindex.py` | Validate response, choice and ranked probabilities while preserving native port framework compatibility. |
| `laya/integrations/tools.py` | `laya_mlx/integrations/tools.py` | Malformed tool decisions/probabilities take the existing error or fallback path. |
| `laya/lang.py` | `laya_mlx/lang.py` | Unicode letter compatibility table, bisected script ranges, English/Vietnamese heuristics and opt-in diagnostic stage/reason/source paths. |
| `laya/mcp/remote.py` | `laya_mlx/mcp/remote.py` | Validate explicit/environment User-Agent and send it on HTTP calls; default uses local package version. |
| `laya/mcp/server.py` | `laya_mlx/mcp/server.py` | Document LAYA_USER_AGENT. |
| `laya/onnx_agent.py` | `laya_mlx/runtime.py (shared host behavior only)` | ONNX implementation excluded; shared long-attribution and probability changes are implemented once in runtime.py. |
| `laya/revisions.py` | `laya_mlx/revisions.py` | Reject nested Windows drive-anchored digest paths. |
| `laya/router.py` | `laya_mlx/router.py` | Forward opt-in explain on route/route_batch without changing explicit override routing. |
| `laya/serve.py` | `laya_mlx/serve.py` | Bound body read time and release admission on 408; hide schema/docs when authenticated; enforce batch token budget even with caller batch size; report invalid tournament labels as 422. |
| `laya/shortlist.py` | `laya_mlx/shortlist.py` | Validate tok/model.encoder before importing the native backend; preserve native MLX embedding. |
| `laya/structured.py` | `laya_mlx/structured.py` | Reject boolean numeric bounds; ignore non-mapping answer entries during projection/details. |

## Assertion adapter and exclusions

`tests/test_official_host.py` runs 31 pinned upstream suite wrappers through
`scripts/run_official_test.py`. Existing host suites moved out of the mixed native
contract file so Linux can collect them. Coverage now includes language diagnostics,
CLI language guesses, confidence, CrewAI, LangChain, LlamaIndex and the shared
long-attribution helper, in addition to the previously selected suites. Assertions,
expected host values, public model names and timing thresholds are retained.

The adapter changes Python namespaces and mock import targets. Router test Agent
stubs are also dispatched through the native load factory. The long-attribution
helper receives the original upstream `_Tok`/`make_real_agent` fixture definitions;
only inference helpers are stubbed by that fixture. It does not execute ONNX or
substitute the full native model.

The explicit backend exclusions are:

- Existing ONNX blocks/imports in hooks, hooks API, evals API and structured suites;
  confidence now also runs its decoder loop for the real MLX Agent host method,
  excluding the ONNX alternative. Upstream backend-switch implementation checks
  remain excluded; local MLX refusal tests cover that boundary.
- Six TileLang/PyTorch padding implementation/docstring checks embedded in the
  newly selected confidence suite. These refer to `laya.backends`/`tl_kernels`,
  which do not exist in this distribution. All confidence host checks remain.
- Four existing serve tests asserting Torch-specific device/thread/status behavior
  and one MCP Torch device test. Native device behavior stays in local tests.
- The upstream training CLI help assertion, because this package is inference-only.
- The native MCP cache suite remains in `test_official_contract.py`: its fake Torch
  encoder/tokenizer fixture is adapted to real MLX arrays. It cannot run on Linux.

The full upstream `test_calibrate.py`, `test_revision_pinning.py`,
`test_predict_long.py` and `test_shortlist.py` mix host assertions with loader,
Torch/ONNX or tensor-specific fixtures and are not newly advertised as passing
whole suites. New local tests cover changed target-error validation, nested drive
paths, missing scan answers and pre-import embedding validation. Shared attribution
checks are run from `_long_attribution.py` unchanged. Native calibration, loader,
shortlist and differential checks remain in the macOS suite.

Some upstream assertions read the upstream documentation or source files. They
verify upstream contracts/examples and their consistency with the port's API;
they are not evidence that this repository copied all upstream documentation.
The pinned checkout must include `docs/` and `examples/`, not only `laya/` and
`tests/`.

Changes to upstream's other SDKs, training/research tools, notebooks and deployment
artifacts are outside this Python inference migration. No PyTorch, ONNX, TileLang
or other SDK execution path was copied into the package.

## Validation evidence

[Machine-readable results, environment and source/artifact hashes](UPSTREAM_0417_VALIDATION.json)
record this local run.

Local validation used Linux x86_64, CPython 3.13.15 and a project `.venv` created
with `uv venv .venv`. Dependencies were installed with `uv pip --python
.venv/bin/python`; no system packages were changed. Initial attempts to download
all development/reference/framework extras stalled. A smaller host environment
used NumPy 2.5.4, pytest 9.2.0, Ruff 0.15.8, FastAPI 0.162.0, Pydantic 2.12.5,
HTTPX 0.28.1, uvicorn 0.54.0 and MCP 2.3.0. Base tokenizer/Hub dependencies
were subsequently installed; `uv pip check --python .venv/bin/python` passes. Optional framework suites use their
upstream stubs when real frameworks are absent; real LlamaIndex testing remains
in macOS CI.

| Command | Result |
|---|---|
| `.venv/bin/pytest -q tests/test_official_host.py tests/test_upstream_0417.py` | **73 passed**: 31 suite wrappers and 42 local boundary regressions. A wrapper contains multiple upstream assertions/tests; this is not a count of individual upstream assertions. |
| `PYTHONPATH=. .venv/bin/python scripts/run_official_test.py .upstream/tests/_long_attribution.py` | **49 unchanged attribution checks passed**, also included in the host suite wrappers. |
| `.venv/bin/python scripts/audit_public_api.py --package laya_mlx` | **345 surface entries, zero unclassified differences**; [current audit](UPSTREAM_0417_API_AUDIT.json). |
| `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | Passed. |
| `.venv/bin/ruff check laya_mlx --select=E9,F63,F7,F82,F401,F811 --line-length=120` | Passed upstream-style source gate. |
| `.venv/bin/python -m compileall -q laya_mlx tests scripts` | Passed. |
| `uv pip check --python .venv/bin/python` | All 50 installed packages compatible. |
| `.venv/bin/python -m build` | Built wheel and sdist at version **0.5.0**. |

A broader exploratory run of `test_email.py`, `test_router.py`,
`test_decision_helpers.py` and `test_upstream_sync.py` produced **128 passed,
2 failures and 2 errors**: four cases require MLX (tiny checkpoints or the native
Snake experiment import). Those cases were not weakened or skipped to produce a
Linux full-suite claim. The dedicated host command above is the supported Linux
validation entry point.

The local regressions cover lazy host imports, real native-import failure on Linux,
BOM-aware native CLI inputs and named parse/read errors, pre-import shortlist
validation, nested drive paths, timeout coercion, reliability precision, batch
result cardinality, exact metric thresholds, missing scan answers, and AST equality
for the five changed host helper/method definitions. The macOS runtime suite adds
three real dtype construction/forward cases for the lazy `DTYPES` boundary.

## Remote CI

[Run 40822499425](https://github.com/Omicron0314/laya-mlx/actions/runs/40822499425)
passed on source commit `0e8c0e51f8914908cc99591576ff1deeed205c82`:

- Linux host job: 73 passed; API audit 345 entries, zero unclassified differences.
- macOS 26 / Python 3.12 host and native contract phase: 74 passed.
- macOS native and integration phase: 298 passed, 1 skipped, 3 warnings.
  The skip is the Linux-only missing-MLX boundary check. Real dtype/forward,
  sequential/parallel small-model comparisons and installed LlamaIndex tests passed.
- macOS lint, formatting, wheel and sdist build passed.

The parent review also retained the module-level `snapshot_download` replacement
point used by the existing offline-resolution regression. Source hashes in the
validation JSON identify the tested implementation; this CI record is a subsequent
documentation update.

## Reproduce and validation limits

On a host-only Linux environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev,serve,mcp,structured]'
git clone https://github.com/NandhaKishorM/laya .upstream
git -C .upstream checkout fe9a6c35023a7d94102e7feaaeaae8857d9d1b07
.venv/bin/pytest -q tests/test_official_host.py tests/test_upstream_0417.py
.venv/bin/python scripts/audit_public_api.py --package laya_mlx
```

CI pins both Linux host and macOS native jobs to that commit. macOS runs upstream
host assertions before native tests in a separate process and retains the original
idle-window failure diagnostics. No timing threshold or retry policy was changed;
[CI_TIMING.md](CI_TIMING.md) documents the pre-existing upstream scheduling flake.

Native CPU tests and small-model numerical comparisons passed in macOS CI as
recorded above. Real full-checkpoint validation, GPU timing and memory growth
were not rerun. Framework suites using stubs remain distinct from the installed
LlamaIndex tests. Historical 0.5.0/v0.4.12 model and performance reports remain
valid only for their recorded sources. The package version is unchanged and no
release has been requested.
