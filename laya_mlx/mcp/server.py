"""MCP stdio server exposing the Laya typed-decision tools.

Part of the optional ``laya[mcp]`` extra. Speaks MCP over stdio, so it can be
wired to any MCP client (OpenClaw, Claude Desktop, cursor, ...):

    laya-mcp-server
    python -m laya.mcp.server

Environment (same meaning as laya.serve where it exists):
  LAYA_DEVICE   the device the checkpoints are asked to build on. Not the value verbatim:
                laya.mcp.device.env_device strips and lower-cases it (`CUDA:0` -> `cuda:0`)
                and refuses with a ValueError naming this variable a string torch cannot parse
                (`gpu`, `cuda:`, `cuda: 0`) or parses into some other device (`cuda:258` asks for
                GPU 2), so the failure reads as a bad environment value
                rather than a torch internal one -- here as a `router construction failed` tool
                error, because a stdio server has no startup to refuse. A preference, not a
                guarantee: as in laya.serve an unavailable GPU falls back to CPU silently, and
                laya_status then answers the loaded agent's real device (before any load it
                answers this preference, flagged with `device_is_preference`). It answers one
                device, though: with two checkpoints resident on different devices the top-level
                `device` is the first entry of `loaded` (the least recently used), so
                `checkpoint_devices` is the field that says where each one actually runs.
  LAYA_PRELOAD  "0" to build the checkpoints lazily instead of at startup (default on)
  LAYA_MODELS   comma list to preload; MCP default is "english,multilingual" so
                typed-decisions stays lazy (in laya.serve an empty value preloads all)
  LAYA_THREADS  cap torch intra-op threads (CPU inference); keep <= physical cores
  LAYA_AUTO_TASK  "1" lets a request auto-route to the typed-decisions checkpoint
                  (same as laya.serve). It does not preload it: LAYA_MODELS still
                  decides what is built at startup.
  LAYA_DEFAULT_MODEL  the checkpoint a state with no language evidence falls back to
                  (same as laya.serve, including its aliases); an unset value leaves
                  it to Router, an unresolvable one is a tool error.
  LAYA_BASE_URL  answer from a running laya-serve over HTTP; unset keeps models local
  LAYA_REMOTE_TIMEOUT  HTTP timeout in seconds for remote mode (default 300)
  LAYA_USER_AGENT  remote HTTP client identity (default laya/<version>)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from importlib import metadata as _metadata
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError
    from pydantic import StrictBool
except ImportError as exc:  # mcp extra not installed
    raise ImportError(
        "the laya[mcp] extra (mcp>=2.2.0) is required to run the MCP server: "
        "pip install 'laya[mcp]'"
    ) from exc

# laya.serve only imports os/typing at module level, so reusing its helpers
# keeps one meaning for LAYA_PRELOAD / LAYA_THREADS / LAYA_DEFAULT_MODEL across the package.
from laya_mlx.serve import _apply_thread_limit, _default_model_option, _env_bool

from .device import env_device
from .tools import (
    ToolError,
    batch_item_key_doc,
    get_available_presets,
    laya_decide,
    laya_predict,
    laya_predict_batch,
    laya_predict_long,
    laya_preset,
    laya_route,
    laya_route_batch,
    laya_shortlist,
    laya_status,
    laya_tournament,
)

try:
    _LAYA_VERSION = _metadata.version("laya-mlx")
except Exception:  # running from a source checkout without install metadata
    import laya_mlx as _laya

    _LAYA_VERSION = getattr(_laya, "__version__", "")

server = MCPServer("laya", version=_LAYA_VERSION)

_ROUTER: Any = None
_ROUTER_LOCK = threading.Lock()

# MCP default preload list. laya.serve preloads every checkpoint when LAYA_MODELS
# is empty; MCP keeps typed-decisions lazy on purpose (it is ~as big as the other
# two, and auto-routing never selects it).
_DEFAULT_MODELS = ("english", "multilingual")

# Shared usage guardrails for the decision tools. laya_status reports instead
# of deciding, so it does not carry them.
_GUARDRAILS = (
    "Use for structured decisions only: choice (finite labels), score (ordinal rubric), "
    "noul (calibrated P(true)). One forward pass ~33ms (GPU) / ~200ms (CPU). "
    "No text generation, so no hallucination. Do NOT use for open Q&A, summarization, "
    "rewriting, code, or multi-hop reasoning. Do NOT use for >20-option choice "
    "questions without narrowing (use laya_shortlist or laya_tournament)."
)

# The `model` argument's contract, spelled out for clients because the tools take a free string.
# The accepted names are deliberately not listed: that registry belongs to laya.router, it grows
# with the checkpoints, and a copy here would be a second thing to keep current. An invalid name
# comes back as an error that lists every accepted one.
_MODEL_DOC = (
    " model: null (default), 'auto', or omitted all let the router pick the checkpoint; "
    "anything else names one, and "
    "core's aliases and casing are honoured ('en', 'ML', 'laya-multilingual'). The answer's "
    "routing.model always reports the canonical name."
)

# The routing overrides and the token budget, spelled out once because the tools take them as free
# strings and integers. Precedence matters to say plainly: a client that pins a model and also sets
# a task would otherwise see the task silently ignored (the tool rejects it instead).
_CONTROLS_DOC = (
    " task: name a checkpoint by what the work is ('typed_decisions', or any checkpoint name) -- "
    "refused on a call that also pins model, since the pin would win and the task would be ignored. "
    "lang: a language code ('de', 'en-US') -- routes non-English text to the multilingual "
    "checkpoint and selects that checkpoint's per-language calibration. "
    "lang_guess: a soft language code -- a probable-but-uncertain language that participates in "
    "routing (checked after lang, before built-in detection) rather than forcing the checkpoint the "
    "way lang does; like task it only routes, so it is refused on a call that pins model. "
    "max_len / head_max_len: positive integers overriding the answering token budget for this call "
    "only -- head_max_len is the option-and-instructions budget, so raise it when a choice question "
    "has many options and the answers look like the labels blur together. "
    "min_confidence: a number in [0, 1] -- an answer whose calibrated confidence falls below it is "
    "flagged 'low_confidence' (and, on laya_decide, its value comes back null), so a caller that "
    "must not act on a guess can set it. Leave any of them unset to keep the checkpoint's own "
    "default."
)


def _models_from_env() -> list[str]:
    """Preload list from LAYA_MODELS (laya.serve contract: comma list of names).

    Unlike laya.serve, an empty value falls back to english+multilingual instead
    of "every checkpoint"; see the module docstring.
    """
    raw = os.environ.get("LAYA_MODELS", "").strip()
    names = [m.strip() for m in raw.split(",") if m.strip()]
    return names or list(_DEFAULT_MODELS)


def _remote_base_url() -> str | None:
    """``LAYA_BASE_URL``: a running laya-serve to answer from instead of loading checkpoints here.

    Set it to the server origin (``http://127.0.0.1:8000``). Blank or unset means the default,
    in-process router. Scheme-less values are accepted as http so a bare ``host:port`` works.
    """
    raw = os.environ.get("LAYA_BASE_URL", "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def _ensure_router() -> Any:
    """Build the Router from the environment, following the laya.serve contract.

    LAYA_DEVICE / LAYA_PRELOAD / LAYA_THREADS / LAYA_AUTO_TASK / LAYA_DEFAULT_MODEL keep the
    same meaning as in laya.serve (the helpers are reused, not duplicated). The last one
    differs only in what an unresolvable value costs: serve refuses to start, while a stdio
    server has no startup to refuse, so the ValueError rides on into the ToolError below.
    LAYA_MODELS follows the serve comma-list but defaults to english+multilingual here, so
    typed-decisions stays lazy: LAYA_AUTO_TASK=1 only lets a matching question schema route
    to it, and it is then loaded on demand. The global is only set once the router is fully
    built, so a failed preload stays retriable on the next tool call, and construction errors
    surface as ToolError payloads instead of being swallowed.
    """
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is not None:
            return _ROUTER
        base_url = _remote_base_url()
        if base_url:
            # Remote mode: no checkpoint is built here, no torch is imported. The RemoteRouter
            # routes locally (pure Python) and answers over HTTP from a running laya-serve.
            try:
                from .remote import RemoteRouter

                router = RemoteRouter(
                    base_url,
                    api_key=os.environ.get("LAYA_API_KEY") or None,
                    auto_task_detection=_env_bool("LAYA_AUTO_TASK", False),
                    **_default_model_option(),
                )
            except Exception as exc:
                raise ToolError(
                    "internal_error", f"remote router construction failed: {exc}"
                ) from exc
            _ROUTER = router
            return router
        try:
            from laya_mlx import Router
        except Exception as exc:
            raise ToolError("internal_error", f"cannot import laya: {exc}") from exc
        try:
            _apply_thread_limit()
            router = Router(
                device=env_device(),
                auto_task_detection=_env_bool("LAYA_AUTO_TASK", False),
                **_default_model_option(),
            )
            if _env_bool("LAYA_PRELOAD", True):
                router.preload(_models_from_env())
        except Exception as exc:
            raise ToolError("internal_error", f"router construction failed: {exc}") from exc
        _ROUTER = router
        return router


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _router_or_error() -> Any:
    """The resident Router, or an McpToolError carrying the ToolError payload."""
    try:
        return _ensure_router()
    except ToolError as exc:
        raise McpToolError(_dump({"error": exc.code, "message": exc.message})) from exc


def _preset_builder(attr_name: str) -> dict:
    import laya_mlx as laya

    builder = getattr(laya, attr_name, None)
    if builder is None:
        raise ToolError("internal_error", f"laya.{attr_name} is not available in this version")
    return builder()


def _wrap(fn, **kwargs) -> str:
    """Run fn; raise McpToolError so isError=true AND the JSON payload is preserved.

    mcp 2.x wraps any non-ToolError Exception as UnexpectedToolError("Error executing
    tool <name>") and redacts the original message. Only
    mcp.server.mcpserver.exceptions.ToolError keeps our JSON on the wire.
    """
    try:
        return _dump(fn(**kwargs))
    except ToolError as exc:
        raise McpToolError(_dump({"error": exc.code, "message": exc.message})) from exc
    except McpToolError:
        raise
    except Exception as exc:
        raise McpToolError(
            _dump({"error": "internal_error", "message": f"{type(exc).__name__}: {exc}"})
        ) from exc


@server.tool(name="laya_status")
def laya_status_tool() -> str:
    """Report the device actually in use per loaded checkpoint (or the configured preference, flagged as such, when nothing is loaded), torch CUDA availability, loaded checkpoints, and package versions."""
    # In remote mode the router is cheap (no checkpoint, no torch), so build it here if a status
    # call comes first: the remote report then describes the server instead of a torch probe of
    # this process, and `device_report()`'s torch import never happens in a remote MCP process.
    router = _ROUTER if (_ROUTER is not None or not _remote_base_url()) else _router_or_error()
    return _wrap(laya_status, router=router, preload=_env_bool("LAYA_PRELOAD", True))


@server.tool(
    name="laya_route",
    description=(
        "Decide which Laya checkpoint would answer, without running a forward pass. "
        "Use this to explain routing (english vs multilingual vs typed-decisions) to the user. "
        "Passing a laya_predict call's model/task/lang/lang_guess here reproduces the routing block it "
        "reported, without paying for the forward pass; leaving model unset (or 'auto') routes as "
        "normal. explain=true adds the final routing decision's language diagnostic, or null; "
        "it describes this MCP process's routing even in remote mode. "
        + _GUARDRAILS
        + _CONTROLS_DOC
    ),
)
def laya_route_tool(
    state: dict,
    questions: dict,
    model: str | None = None,
    task: str | None = None,
    lang: str | None = None,
    lang_guess: str | None = None,
    explain: StrictBool = False,
) -> str:
    """Decide which Laya checkpoint would answer, without running a forward pass."""
    router = _router_or_error()
    return _wrap(
        laya_route,
        state=state,
        questions=questions,
        model=model,
        task=task,
        lang=lang,
        lang_guess=lang_guess,
        explain=explain,
        router=router,
    )


@server.tool(
    name="laya_predict",
    description=(
        "Answer typed questions (choice/score/noul) over any state in one forward pass. "
        "questions: {name: {type: 'choice'|'score'|'noul', instructions: str, criteria?: object|array}}. "
        "For noul, optional labels: {false: str, true: str} changes the model-facing option text. "
        "Any type takes an optional option_order: a permutation of the option indices, slot s showing "
        "option option_order[s]; answers stay keyed in the caller's own order. "
        "Returns answers with confidence, routing metadata and, when it can be read, the real "
        "device of the checkpoint that answered. " + _GUARDRAILS + _MODEL_DOC + _CONTROLS_DOC
    ),
)
def laya_predict_tool(
    state: dict,
    questions: dict,
    model: str | None = None,
    task: str | None = None,
    lang: str | None = None,
    lang_guess: str | None = None,
    max_len: int | None = None,
    head_max_len: int | None = None,
    min_confidence: float | None = None,
) -> str:
    """Answer typed questions (choice/score/noul) over any state in one forward pass."""
    router = _router_or_error()
    return _wrap(
        laya_predict,
        state=state,
        questions=questions,
        model=model,
        task=task,
        lang=lang,
        lang_guess=lang_guess,
        max_len=max_len,
        head_max_len=head_max_len,
        min_confidence=min_confidence,
        router=router,
    )


@server.tool(
    name="laya_predict_batch",
    description=(
        "Answer many typed-question requests in one call: requests is a non-empty array of "
        + batch_item_key_doc()
        + ", each with the same "
        "questions schema as laya_predict. Requests are routed first, grouped by checkpoint, "
        "and share forward passes when their question schemas match, so scoring many "
        "requests costs one round trip instead of N. max_len / head_max_len are per request "
        "here, not per call: a wide question can raise its own budget without shrinking the "
        "batch's other requests to it, and requests that ask for different budgets are split "
        "into separate forward passes. Returns answers in input order with "
        "per-request routing and device, plus model_counts and batch latency. " + _GUARDRAILS
    ),
)
def laya_predict_batch_tool(
    requests: list,
    batch_size: int = 0,
    hooks_timeout: float | None = 0,
    min_confidence: float | None = None,
    sort_by_length: bool = False,
) -> str:
    """Answer many typed-question requests in one batched call."""
    router = _router_or_error()
    # batch_size=0 means "unset": MCP clients send defaults eagerly, and
    # None is what Router.predict_batch takes as "no forward-pass cap". hooks_timeout reads
    # a numeric 0 the same way -- and deliberately only a numeric 0, because `False == 0`
    # would otherwise fold a boolean, which `_validate_hooks_timeout` refuses by contract
    # ("would silently turn True into a one-second deadline").
    # Everything else -- negatives and booleans included -- goes to the tools layer, which
    # owns validation and answers with a ToolError naming the value. Collapsing those here
    # instead turned an invalid deadline into "no deadline" and an invalid gate into
    # "no gate": the caller got a clean response and no way to tell the control it asked
    # for had been dropped.
    if not isinstance(hooks_timeout, bool) and hooks_timeout == 0:
        hooks_timeout = None
    # `batch_size` gets the same boolean-aware collapse for the same reason: `False == 0`, so a plain
    # `or None` fold turned a caller's explicit `batch_size=false` into "no cap" and hid the
    # `_validate_batch_size` refusal that a `batch_size=true` already reaches.
    if not isinstance(batch_size, bool) and batch_size == 0:
        batch_size = None
    return _wrap(
        laya_predict_batch,
        requests=requests,
        batch_size=batch_size,
        hooks_timeout=hooks_timeout,
        min_confidence=min_confidence,
        sort_by_length=sort_by_length,
        router=router,
    )


@server.tool(
    name="laya_route_batch",
    description=(
        "Decide which Laya checkpoint would answer each request, without running any "
        "forward pass or loading a checkpoint: requests is a non-empty array of "
        + batch_item_key_doc(omit=("max_len", "head_max_len"))
        + ". Use this to "
        "inspect or aggregate the routing of a workload before paying model-load cost. "
        "Returns one {model, repo, reason} decision per request in input order, plus "
        "model_counts. explain=true adds each final decision's language diagnostic, or null; "
        "defaults to false. In shared-server mode this is local routing evidence, not the HTTP "
        "server's prediction. hooks_timeout: positive number or unset -- override the "
        "Router's own hook deadline for this sweep's on_route dispatch, so a slow "
        "or hung operator hook cannot stall a routing pass the caller only wants "
        "the checkpoint labels for. " + _GUARDRAILS
    ),
)
def laya_route_batch_tool(
    requests: list, hooks_timeout: float = 0, explain: StrictBool = False
) -> str:
    """Route many requests to checkpoints without a forward pass."""
    router = _router_or_error()
    # hooks_timeout=0 means "unset": MCP clients send defaults eagerly, and
    # None is what Router.route_batch takes as "use my own timeout". Only a numeric 0 collapses:
    # `False == 0`, and `laya_route_batch` refuses a boolean timeout the way `laya_predict_batch`
    # does, so folding `false` into unset here would answer the caller's request with silence.
    if not isinstance(hooks_timeout, bool) and hooks_timeout == 0:
        hooks_timeout = None
    return _wrap(
        laya_route_batch,
        requests=requests,
        hooks_timeout=hooks_timeout,
        explain=explain,
        router=router,
    )


@server.tool(
    name="laya_shortlist",
    description=(
        "Shortlist a many-option choice question to its k most likely labels by embedding "
        "similarity (mean-pooled from the answering checkpoint's own encoder, so no extra "
        "model is downloaded), then answer in one forward pass. Use this instead of "
        "laya_predict when embedding similarity is appropriate for narrowing a large choice. "
        "Returns the answers plus per-question shortlist metadata (kept labels, cosine "
        "scores, k, option count, and whether the question passed through unshortlisted). "
        "Shortlisting narrows the label set; head_max_len decides how many tokens each kept label "
        "is read with, so the two together are the fix for a large-criteria question. "
        "A question that narrows cannot keep a caller-supplied option_order: the order covers "
        "every option while the answer covers the kept labels only, so it is dropped rather "
        "than remapped, and each such question is named under option_order_dropped with its "
        "old and kept option counts. " + _GUARDRAILS + _MODEL_DOC + _CONTROLS_DOC
    ),
)
def laya_shortlist_tool(
    state: dict,
    questions: dict,
    model: str | None = None,
    k: int = 20,
    task: str | None = None,
    lang: str | None = None,
    lang_guess: str | None = None,
    max_len: int | None = None,
    head_max_len: int | None = None,
    min_confidence: float | None = None,
) -> str:
    """Shortlist many-option choice questions, then answer."""
    # k's default mirrors laya.shortlist.DEFAULT_SHORTLIST_K; it is a literal
    # here so the MCP schema carries the default without importing numpy at
    # server start. tests/test_mcp.py reads it out of the runtime and fails if
    # this drifts from it.
    router = _router_or_error()
    return _wrap(
        laya_shortlist,
        state=state,
        questions=questions,
        model=model,
        k=k,
        task=task,
        lang=lang,
        lang_guess=lang_guess,
        max_len=max_len,
        head_max_len=head_max_len,
        min_confidence=min_confidence,
        router=router,
    )


@server.tool(
    name="laya_tournament",
    description=(
        "Answer large choice questions by balanced elimination with the decision model, without "
        "embeddings. group_size: integer at least 2, default 16. All groups share one forward "
        "pass per round; their winners advance until they fit, then one final pass answers the "
        "complete request. Small choices and non-choice questions pass through. "
        "A missing or out-of-group winner stops the call as internal_error before another pass. "
        "Returns answers, routing, total latency_ms, the answering device when readable, and tournament metadata "
        "per choice: labels (finalists in criteria order), n (original option count), rounds "
        "(elimination rounds, zero on passthrough). Probabilities and confidence are over the "
        "finalists only; thresholds fitted on a full option set do not transfer. A narrowed "
        "question's option_order is dropped and reported under option_order_dropped with n and "
        "kept counts. In-process only: with LAYA_BASE_URL set, use laya_predict with the HTTP "
        "server's LAYA_TOURNAMENT_GROUP enabled instead. "
        + _GUARDRAILS
        + _MODEL_DOC
        + _CONTROLS_DOC
    ),
)
def laya_tournament_tool(
    state: dict,
    questions: dict,
    model: str | None = None,
    group_size: int = 16,
    task: str | None = None,
    lang: str | None = None,
    lang_guess: str | None = None,
    max_len: int | None = None,
    head_max_len: int | None = None,
    min_confidence: float | None = None,
) -> str:
    """Answer many-option choices with balanced tournament rounds."""
    router = _router_or_error()
    return _wrap(
        laya_tournament,
        state=state,
        questions=questions,
        model=model,
        group_size=group_size,
        task=task,
        lang=lang,
        lang_guess=lang_guess,
        max_len=max_len,
        head_max_len=head_max_len,
        min_confidence=min_confidence,
        router=router,
    )


_PRESET_INFO = get_available_presets()

# Built from tools.get_available_presets() rather than typed out, because the names in a
# tools/list description are what a client sends back: a hand-written list here could advertise a
# preset the tool rejects, or miss one it accepts. The field each preset reads is the field its own
# instructions name, so this line cannot go stale against laya/presets.py either.
_PRESET_DOC = (
    (
        "Run a built-in workflow: %s. "
        "Use when the task matches one of those presets instead of hand-writing questions. "
        "state: each preset reads one field (%s); a state holding a single string is placed under it "
        "for you, and a state with more keys is passed through as given, so put your text under the "
        "field that applies. Aliases: %s. "
        % (
            " | ".join("'%s'" % name for name in _PRESET_INFO),
            ", ".join(
                "'%s' reads `%s`" % (name, info["state_field"])
                for name, info in _PRESET_INFO.items()
                if info.get("state_field")
            ),
            ", ".join(
                "'%s' is '%s'" % (alias, name)
                for name, info in _PRESET_INFO.items()
                for alias in info.get("aliases", [])
            ),
        )
    )
    + _GUARDRAILS
    + _CONTROLS_DOC
)


@server.tool(
    name="laya_preset",
    description=_PRESET_DOC,
)
def laya_preset_tool(
    preset: str,
    state: dict,
    task: str | None = None,
    lang: str | None = None,
    lang_guess: str | None = None,
    max_len: int | None = None,
    head_max_len: int | None = None,
    min_confidence: float | None = None,
) -> str:
    """Run a built-in workflow preset; the tool description carries the names and state fields."""
    router = _router_or_error()
    return _wrap(
        laya_preset,
        preset=preset,
        state=state,
        task=task,
        lang=lang,
        lang_guess=lang_guess,
        max_len=max_len,
        head_max_len=head_max_len,
        min_confidence=min_confidence,
        router=router,
        preset_builder=_preset_builder,
    )


@server.tool(
    name="laya_decide",
    description=(
        "Answer a JSON-schema-shaped decision in one forward pass and return the decided "
        "values: schema is a JSON schema object whose properties are enum choices, booleans "
        "(noul), or integers with integer minimum/maximum (ordinal score); free strings, "
        "arrays and nested objects are rejected by path. Use this instead of laya_predict "
        "when the caller already knows the answer shape and wants values projected onto the "
        "schema (enum member, integer level, boolean) plus per-field confidence, instead of "
        "an answer map to parse by hand. "
        "min_confidence: a number in [0, 1] -- a field whose answer falls below it comes back as "
        "null in values (its confidence is still reported), so a caller that must not act on a "
        "guess can abstain per field. " + _GUARDRAILS
    ),
)
def laya_decide_tool(
    state: dict, schema: dict, model: str | None = None, min_confidence: float | None = None
) -> str:
    """Answer a JSON-schema-shaped decision and return the decided values."""
    router = _router_or_error()
    return _wrap(
        laya_decide,
        state=state,
        schema=schema,
        model=model,
        min_confidence=min_confidence,
        router=router,
    )


@server.tool(
    name="laya_predict_long",
    description=(
        "Answer typed questions over a state longer than the context window: the state is "
        "scanned in overlapping windows in shared forward passes and aggregated per question "
        "(strongest window for noul, most-confident window for choice/score). Use this instead "
        "of laya_predict whenever the state does not fit one window -- laya_predict answers "
        "from its first window and never reads the rest. Returns the answers, each with the "
        "deciding window's attribution, plus the window count. "
        "window / stride: positive integers sizing the scan (a smaller window isolates a "
        "localized span); unset means the checkpoint budget and 50% overlap. "
        "task: name a checkpoint by what the work is -- refused on a call that also pins model. "
        "lang: a language code -- routes non-English text and selects that checkpoint's "
        "per-language calibration. " + _GUARDRAILS + _MODEL_DOC
    ),
)
def laya_predict_long_tool(
    state: dict,
    questions: dict,
    model: str = "auto",
    window: int | None = None,
    stride: int | None = None,
    task: str | None = None,
    lang: str | None = None,
) -> str:
    """Scan a long state in windows, then answer."""
    router = _router_or_error()
    return _wrap(
        laya_predict_long,
        state=state,
        questions=questions,
        model=model,
        window=window,
        stride=stride,
        task=task,
        lang=lang,
        router=router,
    )


def main() -> None:
    # Windows without Developer Mode cannot create HF cache symlinks (WinError
    # 1314). On POSIX the cache uses symlinks by default and disabling them
    # would copy the files and double disk use, so the override is Windows-only.
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if _remote_base_url():
        # Nothing to preload: the server at LAYA_BASE_URL owns the checkpoints. Say so once, so a
        # reader of the client's stderr knows which process is answering.
        print(
            f"[laya-mcp] remote mode: answering from laya-mlx-serve at {_remote_base_url()}",
            file=sys.stderr,
        )
    elif _env_bool("LAYA_PRELOAD", True):
        try:
            _ensure_router()
        except Exception as exc:
            print(f"[laya-mcp] preload failed (will retry on demand): {exc}", file=sys.stderr)
    server.run()


if __name__ == "__main__":
    main()
