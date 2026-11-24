# Changelog

Each release names the upstream [Laya](https://github.com/NandhaKishorM/laya) revision whose host
behavior it follows. Detailed records live in [docs/](docs/README.md).

## 0.9.0 — 2026-11-23

Follows upstream **v0.5.3** (`abefd7d9`). [Compatibility record](docs/UPSTREAM.md).

- **Short unaccented Spanish and Portuguese route multilingual:** language detection gains
  upstream's unaccented and closed-class Spanish and Portuguese stopwords (`puedo`, `estoy`,
  `despues`, `faco`, `qual`, ...), so accent-stripped tickets such as `no puedo entrar a mi cuenta`
  are no longer read as English, alone or beside English boilerplate in a structured state.
  Italian `qual` stays Italian, and English words such as `mis`, `dime` and `solo` stay English.
- Single-sequence encoder work is submitted to Metal while the graph is built, single-sequence
  CPU batches score in FP32 to avoid an MLX FP16 crash, and long local attention bounds its
  key/value work to each block's window.
- Upstream's vega choice-label folding and ONNX export verification apply to modules not
  shipped here.
- Validation: host, contract and native tests and the public API audit on the release source.
  Upstream's idle-window timing test failed on the runner, as it does on the pre-port commit.
  Real-checkpoint validation was not re-run.

## 0.8.1 — 2026-11-21

Follows upstream **v0.5.2** (`e150aa92`). [Compatibility record](docs/UPSTREAM.md).

- No change in behavior. Upstream v0.5.2 adds `laya.Escalator` and `laya.vllm_teacher`,
  re-exports of vegaml 0.18's teacher for its continuous learner; like `laya.Learner`, they read
  with vegaml's reader and are not included.
- Upstream's signed-policy tests nest their JSON deeper; `laya.policy` is not shipped.
- Validation: host, contract and native tests and the public API audit on the release source.
  Real-checkpoint validation was not re-run; no model or engine code changed.

## 0.8.0 — 2026-11-21

Follows upstream **v0.5.1** (`090c540e`). [Compatibility record](docs/UPSTREAM.md).

- **`load()` takes no checkpoint named:** `model_id_or_path` now defaults to `None`, as upstream,
  and still loads `convaiinnovations/laya`. Upstream 0.5.0 makes vega the default engine when no
  checkpoint is named; laya-mlx does not include the vegaml engine, so `Agent()` and `load()` keep
  the classic checkpoint (upstream's own behavior without vegaml) and do not warn.
- `laya_mlx.agent.wants_vega()` is ported: it answers `True` only for an explicit `vegaml=`, which
  `Agent` and `load` still refuse.
- Upstream's continuous learner (`laya.Learner`, a vegaml re-export) is not included.
- Upstream's `serve` suite now follows the installed uvicorn's log-level casing; it runs unchanged.
  Upstream's signed-policy hardening applies to `laya.policy`, which is not shipped.
- Validation: host, contract and native tests and the public API audit on the release source.
  Upstream's idle-window timing test failed on the runner, as it does against unmodified upstream.
  Real-checkpoint validation was not re-run; no model or engine code changed.

## 0.7.2 — 2026-11-19

Follows upstream **v0.4.37** (`d1282e79`). [Compatibility record](docs/UPSTREAM.md).

- No change in behavior. Upstream v0.4.36 and v0.4.37 change only upstream's README, package
  versions and Nix package; no host code or test suite this package mirrors changed.
- Validation: host, contract and native tests and the public API audit on the release source.
  Real-checkpoint validation was not re-run; no model or engine code changed.

## 0.7.1 — 2026-11-18

Follows upstream **v0.4.35** (`b1d65695`). [Compatibility record](docs/UPSTREAM.md).

- No change in behavior. Upstream v0.4.35 adds `route()` to its TypeScript HTTP client, which
  calls `POST /v1/route`; `laya-mlx-serve` has served that endpoint since 0.7.0, so the client
  works against it.
- Upstream's `hooks_api` suite now also checks that the TypeScript routing types stay outside the
  root exports; it runs unchanged here.
- Validation: host, contract and native tests and the public API audit on the release source.
  Real-checkpoint validation was not re-run; no model or engine code changed.

## 0.7.0 — 2026-11-16

Follows upstream **v0.4.34** (`80a1da1f`). [Compatibility record](docs/UPSTREAM.md).

- **Routing without inference:** `POST /v1/route` on `laya-mlx-serve`, and `explain` on the MCP
  `laya_route` and `laya_route_batch` tools, return the decision and its language diagnostic.
- **Router vega lane:** `Router(vega=agent)` sends images, audio, video, over-long states and
  questions with fitted heads to an already built vega agent. laya-mlx does not include the vegaml
  engine: `Agent(vegaml=True)`, `load(vegaml=True)` and `Router(vega=True)` raise, and the lane is
  off by default.
- **LangChain:** `LayaRouter(decision_policy="native")` follows an answer's `auto`/`review` route.
  `LayaToolSelector` accepts a per-bucket `min_confidence` map.
- **Stricter validation (may raise where 0.6.0 accepted input):** a non-integer `batch_size`,
  malformed checkpoint source pairs, decisions naming an agent or choice outside the list,
  non-array `anyOf`/`oneOf`, wide integers in calibration bins, and media keys in HTTP bodies.
- **Fixes:** temperatures, confidences and guard values written as integers wider than a double no
  longer crash a load or a decision; BOM-prefixed eval datasets and reports are read;
  `OptionStabilityHook` no longer counts a rename probe identical to the original question;
  an over-nested `LAYA_SHA256_DIGESTS` is a configuration error.
- **Faster:** calibration fitting is vectorized, and routing skips hook dispatch when no hook
  implements `on_route`.
- The server logs one warning per request when a state was truncated.
- Docs and tests: one evergreen `docs/UPSTREAM.md` replaces the per-version migration records;
  local tests that duplicated the pinned upstream suites were removed.
- Validation: 63/63 fixtures for all three checkpoints in FP32 and FP16 against the v0.4.34
  reference, 100 stable calls each. Performance figures were not re-measured for this release.

## 0.6.0 — 2026-10-29

Follows upstream **v0.4.17** (`fe9a6c35`).

- **Language routing:** `detect_language(..., explain=True)` reports why a language was chosen;
  updated Unicode script lookup and English/Vietnamese detection fixes.
- **Long documents:** answer attribution survives end hooks that reorder results; missing
  answers raise, and replaced or edited answers carry no misleading window attribution.
- **Stricter validation (may raise where 0.5.0 accepted input):** reliability calibration
  precision, malformed integration responses, evaluation cardinality and metric checks, confidence
  overflow and digest paths.
- **HTTP and CLI:** request-body deadlines, batch token budgets, configurable remote User-Agent,
  BOM-aware input files; the CLI reports read/encoding/JSON errors before loading a model.
- **Fix:** `predict_batch` no longer groups integer or boolean keys with their string equivalents
  (`1` vs `"1"`), which returned the wrong key types for later requests.
- **Fix:** with `hooks_concurrent=False`, a timed-out hook keeps the serialization lock until it
  exits, and a hook whose deadline passes while waiting is never called.
- Linux host-only CI job; the real-checkpoint validation harness targets v0.4.17.
- Validation: 63/63 fixtures for all three checkpoints in FP32 and FP16 against the v0.4.17
  reference, 100 stable calls each. Performance figures were not re-measured for this release.

## 0.5.0 — 2026-10-20

Follows upstream **v0.4.12** (`ffc14201`). [Qualification](docs/QUALIFICATION_0_4_12.md).

- Batch and long-text inference, structured decisions, hooks, calibration, evaluation, HTTP/MCP
  serving and LangChain/LlamaIndex/CrewAI/Agno integrations.
- `Router()` keeps two models resident by default (`max_loaded=1` restores the old limit).
- Scalar choice labels are accepted; invalid requests use upstream error types and messages.

## 0.4.0 — 2026-10-10

Selectively follows upstream **v0.4.1** (`1adc59f`).

- **Behavior change:** `Router()` defaults to `multilingual` for undecided text, including short
  inputs such as `"refund me"`. Use `Router(default="english")` for the previous fallback.
- Custom checkpoint registration (`register`, `unregister`, `models=`), concurrent-safe lifecycle,
  and MLX buffer-cache release on eviction.
- `min_confidence` gate with per-bucket thresholds; `predict_tournament` for large choice sets.
- Checkpoints with `"option_layout": "parallel"` are supported end to end.
- Language detection no longer treats all-capital acronyms as French or Spanish.
- [Validation report](benchmarks/results/upstream-v041-validation.json): 378/378 argmax agreements.

## 0.3.0 — 2026-10-02

Selectively follows upstream `4aa6761` (v0.3.23 source tree).

- Conversation-list states keep their newest tokens when context fills.
- `noul` criteria accept only `false`/`true` keys; optional `labels=` changes the displayed words.
- `answer_confidence` on every answer; `usage` reports state truncation and colliding option spans.
- Calibration temperatures are clamped to `[0.5, 5.0]` (upstream v0.3.5).
- Email cleaning keeps ordinary requests and recognizes multilingual footers.

## 0.1.0 – 0.2.0 — 2026-09-19 to 2026-09-22

Initial MLX port: native encoder and decision heads, pre-converted checkpoints and benchmarks.
These releases predate the repository's version tags.
