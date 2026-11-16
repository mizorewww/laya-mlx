"""Thin tool layer over laya. Lay-a calls go through injectable hooks for tests."""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol, Sequence

from ..confidence import check_min_confidence
from ..hooks import validate_timeout
from ..presets import state_field
from .device import agent_device, device_report, router_agent


class ToolError(Exception):
    """Raised for user-facing tool failures. Message is safe to return to the LLM."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AgentLike(Protocol):
    def predict(self, state: dict, questions: dict) -> dict: ...


PRESETS: dict[str, str] = {
    "guard": "guard_questions",
    "moderation": "moderation_questions",
    "triage": "triage_questions",
    "model_router": "router_questions",
    "email": "email_questions",
}

# `router` is what the CLI calls this preset, and it is the word a caller reaches for first;
# `model_router` stayed because clients already have it in their prompts. Both name one preset,
# and the canonical key is what comes back in the result.
PRESET_ALIASES: dict[str, str] = {"router": "model_router"}


def get_available_presets() -> dict[str, dict[str, Any]]:
    """The built-in presets a ``laya_preset`` call can name, with what each one reads.

    The state field comes from the questions themselves rather than from a table beside them, so
    this cannot drift from the presets: a caller that has to hand-build a state (and the MCP
    ``tools/list`` description, which is built from here) sees the field each preset actually asks
    about. Builders are resolved through ``laya`` at call time, the way ``laya.mcp.server`` resolves
    them, so a preset added in a newer version shows up here as soon as it is importable.
    """
    import laya_mlx as laya

    aliases: dict[str, list[str]] = {}
    for alias, target in PRESET_ALIASES.items():
        aliases.setdefault(target, []).append(alias)

    out: dict[str, dict[str, Any]] = {}
    for name, attr in sorted(PRESETS.items()):
        entry: dict[str, Any] = {"questions": attr}
        if name in aliases:
            entry["aliases"] = sorted(aliases[name])
        builder = getattr(laya, attr, None)
        if builder is not None:
            questions = builder()
            entry["n_questions"] = len(questions)
            field = state_field(questions)
            if field is not None:
                entry["state_field"] = field
        out[name] = entry
    return out


VALID_TYPES = {"choice", "score", "noul"}
# `auto` is this layer's own sentinel -- "route it, do not pin a checkpoint". Every other value is a
# checkpoint name, and the registry of those (names, aliases, casing) is core's: `laya.router` runs
# every model argument through `normalise_name` before it loads anything, which is the same call the
# tools below end up making through `router.predict(model=...)`. A second list here could only ever
# be narrower than that one, so it is not repeated.
AUTO = "auto"


def _check_noul_labels(name: str, labels: Any) -> None:
    """`labels` renames a noul answer's two model-facing texts; mirror the agent's rule.

    `laya.common._resolve_noul_labels` is the contract: exactly `false` and `true`, each a
    distinct non-empty string. The agent raises `ValueError` on anything else, and the tool
    wrapper turns a `ValueError` into `internal_error` -- the code this layer uses for a tool
    that broke. So the check is repeated here rather than imported, both to keep the message
    in this layer's `questions[...]` voice and to keep `laya.mcp.tools` free of a torch import.
    """
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].labels must map exactly 'false' and 'true' to distinct "
            f"non-empty strings",
        )
    values = [labels["false"], labels["true"]]
    # The agent requires a str here, and it is checked before the text is used: a number or a
    # bool is not a label, it is a type mistake. Stringifying first would quietly accept what
    # the agent rejects, and the rejection would then arrive too late to be reported as a
    # caller error.
    if not all(isinstance(value, str) for value in values):
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].labels must give 'false' and 'true' string values, got "
            f"{[type(v).__name__ for v in values]}",
        )
    texts = [value.strip() for value in values]
    if not texts[0] or not texts[1] or texts[0] == texts[1]:
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].labels must give 'false' and 'true' distinct non-empty strings",
        )


def _check_option_order(name: str, order: Any, n: int) -> list:
    """The agent's `option_order` rule: slot s shows option `order[s]`.

    Anything but a permutation of the option indices would drop an option or show one twice, and
    the agent rejects it with a ValueError -- which, past this layer, reaches the client as
    internal_error. A bool, a float or a numeric string is not an index; the agent does not
    coerce them, so neither does this.
    """
    if (
        not isinstance(order, (list, tuple))
        or len(order) != n
        or not all(isinstance(i, int) and not isinstance(i, bool) for i in order)
        or sorted(order) != list(range(n))
    ):
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].option_order must be a permutation of range({n}) -- one slot per "
            f"option, each option once -- got {order!r}",
        )
    return list(order)


def validate_questions(questions: Any) -> dict:
    if not isinstance(questions, dict) or not questions:
        raise ToolError(
            "invalid_questions",
            "questions must be a non-empty JSON object keyed by question name",
        )
    cleaned: dict[str, Any] = {}
    for name, spec in questions.items():
        if not isinstance(name, str) or not name:
            raise ToolError(
                "invalid_questions", f"question name must be a non-empty string: {name!r}"
            )
        if not isinstance(spec, dict):
            raise ToolError("invalid_questions", f"questions[{name}] must be an object")
        qtype = spec.get("type")
        if not isinstance(qtype, str) or qtype not in VALID_TYPES:
            raise ToolError(
                "invalid_questions",
                f"questions[{name}].type must be one of {sorted(VALID_TYPES)}, got {qtype!r}",
            )
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ToolError(
                "invalid_questions",
                f"questions[{name}].instructions must be a non-empty string",
            )
        entry: dict[str, Any] = {"type": qtype, "instructions": instructions}
        criteria = spec.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].criteria must be a non-empty object of label -> description",
                )
            entry["criteria"] = {str(k): v for k, v in criteria.items()}
        elif qtype == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].criteria must be a non-empty list of rubric levels",
                )
            # A null level is a hole in the rubric. The agent rejects it before encoding, so
            # accepting it here only moved the failure past the point where the tool could
            # still name the question.
            if None in criteria:
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].criteria has a null level at index "
                    f"{list(criteria).index(None)}; give every level a description",
                )
            entry["criteria"] = list(criteria)
        else:  # noul
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise ToolError(
                        "invalid_questions",
                        f"questions[{name}].criteria must be an object when present (noul)",
                    )
                # `render_options` reads these descriptions by name, so a dict keyed any other
                # way is not a noul description at all -- the agent rejects it rather than
                # silently falling back to the default pair.
                keys = {str(k).lower() for k in criteria}
                if not keys <= {"true", "false"}:
                    raise ToolError(
                        "invalid_questions",
                        f"questions[{name}].criteria must be keyed only 'true'/'false' "
                        f"(either or both, omitting it is fine), got {sorted(keys)}. To word "
                        f"the answer differently set 'labels' instead.",
                    )
                entry["criteria"] = {str(k): v for k, v in criteria.items()}
        if "labels" in spec:
            # noul-only, like the agent: on any other type it is a caller mistake.
            if qtype != "noul":
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].labels is only supported for noul questions",
                )
            _check_noul_labels(name, spec["labels"])
            entry["labels"] = spec["labels"]
        if "option_order" in spec:
            # Documented for every type; rebuilding the entry from the keys above dropped it, so
            # the order a caller asked for never reached the model.
            n_options = 2 if qtype == "noul" else len(entry["criteria"])
            entry["option_order"] = _check_option_order(name, spec["option_order"], n_options)
        cleaned[name] = entry
    return cleaned


def validate_state(state: Any) -> dict:
    if not isinstance(state, dict) or not state:
        raise ToolError("invalid_state", "state must be a non-empty JSON object")
    return dict(state)


def validate_preset(preset: Any) -> str:
    """Canonical :data:`PRESETS` key for a preset argument.

    The one spelling a caller reaches for that is not a table key is ``router`` -- what the CLI
    calls ``model_router``. Aliases resolve here so both names work, and the canonical key is what
    comes back, so the preset a caller reads is not the one it happened to type.
    """
    name = PRESET_ALIASES.get(preset, preset) if isinstance(preset, str) else preset
    if not isinstance(name, str) or name not in PRESETS:
        # The arguments arrive from a model, so a JSON list or object is a real possibility and
        # belongs in the same invalid_preset as an unknown name.
        raise ToolError(
            "invalid_preset",
            f"preset must be one of {sorted(PRESETS)}, got {preset!r}",
        )
    return name


def validate_model(model: Any) -> str:
    """Canonical checkpoint name for a tool argument, or ``"auto"``.

    Core's ``normalise_name`` is what decides whether something names a checkpoint: it trims,
    lowercases and resolves ``laya.router._ALIASES``. Running the argument through it here means
    this layer cannot reject a name that ``router.predict(model=...)`` would have accepted a few
    lines later, and an alias comes back canonical so the ``routing.model`` a caller reads does not
    depend on how the checkpoint was spelled. Deferred import: nothing else in this module pulls
    torch in, and importing this file is how an MCP client starts the server.
    """
    if model is None:
        return AUTO
    if isinstance(model, str) and model.strip().lower() == AUTO:
        return AUTO
    from laya_mlx.router import normalise_name

    try:
        return normalise_name(model)
    except ValueError as error:
        raise ToolError("invalid_model", "%s, or %r" % (error, AUTO)) from None


def validate_task(task: Any) -> str | None:
    """Core's ``task`` override, checked the way core checks it, or ``None`` for "not set".

    ``Router._route`` sends ``task`` through the same ``normalise_name`` as ``model`` and then looks
    the result up in its registry, so a task that names nothing raises ``KeyError`` from inside the
    router -- and over MCP a ``KeyError`` is reported as ``internal_error``, which tells a caller
    nothing about what it typed. Validating here turns that into ``invalid_task`` with the list of
    names that do resolve. The caller's own spelling is what gets forwarded, so ``routing.reason``
    still reads as the request that was made.
    """
    if task is None:
        return None
    from laya_mlx.router import normalise_name

    # The remap is core's own expression, copied rather than relied upon through the alias table:
    # `Router._route` turns task="typed_decisions" (the underscore form the CLI and the question ids
    # use) into the hyphenated checkpoint name before normalising it. `_ALIASES` happens to hold the
    # same mapping today, so this costs nothing and keeps the two paths agreeing if it ever goes.
    name = "typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task
    try:
        normalise_name(name)
    except ValueError as error:
        raise ToolError("invalid_task", "%s, or %r" % (error, "typed_decisions")) from None
    return task


def validate_lang(lang: Any) -> str | None:
    """A language code for core's ``lang`` override, or ``None`` for "not set".

    Passed through verbatim: what a code means is core's business -- on the router it decides which
    checkpoint can read the state, on a checkpoint it selects the per-language temperature table --
    and a blank or unknown code falls through to the built-in detection rather than failing. Only
    the type is checked, because a truthy list or object would reach the temperature lookup and
    raise ``AttributeError`` on ``.split`` deep inside a forward pass.
    """
    if lang is None:
        return None
    if not isinstance(lang, str):
        raise ToolError(
            "invalid_lang", "lang must be a language code like 'en' or 'de', got %r" % (lang,)
        )
    return lang


def validate_lang_guess(lang_guess: Any) -> str | None:
    """A soft routing hint for core's ``lang_guess`` override, or ``None`` for "not set".

    ``lang_guess`` sits between an explicit ``lang`` and the built-in detector: it names a probable
    language so routing can prefer the checkpoint that reads it, without forcing the answer the way
    ``lang`` does. Like ``lang`` only the type is checked here -- a code's meaning is core's business,
    and a blank or unknown code falls through to detection rather than failing. Core also accepts a
    callable for ``lang_guess``, but an MCP client carries JSON, not a function, so a non-string is
    refused rather than silently dropped. The batch surface already advertises this key
    (:data:`BATCH_ITEM_OVERRIDES`); the single-request tools forward it through here.
    """
    if lang_guess is None:
        return None
    if not isinstance(lang_guess, str):
        raise ToolError(
            "invalid_lang_guess",
            "lang_guess must be a language code like 'en' or 'de', got %r" % (lang_guess,),
        )
    return lang_guess


def validate_budget(value: Any, name: str) -> int | None:
    """One of the two per-call token budgets (``max_len``, ``head_max_len``), or ``None``.

    Core slices and compares against these, so a float, a numeric string or a zero does not fail
    cleanly -- it truncates the sequence to nothing or raises ``TypeError`` mid-forward. ``None``
    means "use what the checkpoint was trained with", which is the same thing the CLI's flags mean.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolError(
            "invalid_%s" % name, "%s must be a positive integer, got %r" % (name, value)
        )
    return value


def validate_min_confidence(value: Any) -> float | None:
    """The per-call abstention threshold (``min_confidence``), or ``None`` when unset.

    Range, booleans and non-finite values are core's rule, not this layer's, so it defers to
    ``laya.confidence.check_min_confidence`` and only wraps its ``ValueError`` as a clean
    ``invalid_min_confidence`` the client can read -- the way ``validate_task`` carries core's
    task list instead of restating it. ``None`` means "answer everything", which is the flag's
    meaning on the CLI and the checkpoint default on every surface.
    """
    if value is None:
        return None
    try:
        return check_min_confidence(value)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_min_confidence", str(exc)) from None


def _overrides(task: Any, lang: Any, max_len: Any, head_max_len: Any) -> dict:
    """The per-call controls as keyword arguments, with the unset ones left out.

    Only what the caller actually set is forwarded: an injectable router or agent is not required to
    accept a keyword it was never asked about, and dropping the unset ones keeps every call that
    passes no controls byte-identical to the call it made before they existed.
    """
    values = {"task": task, "lang": lang, "max_len": max_len, "head_max_len": head_max_len}
    return {name: value for name, value in values.items() if value is not None}


def _raise_if_remote_error(exc: BaseException) -> None:
    """Turn transport failures into the ToolError the client can read."""
    from .remote import RemoteError

    if isinstance(exc, RemoteError):
        raise ToolError(exc.code, exc.message) from exc


def _normalize_answers(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ToolError("internal_error", "predict returned non-object answers")
    return raw


def _normalize_result(result: Any) -> dict:
    if not isinstance(result, dict):
        raise ToolError("internal_error", "predict returned non-object")
    # Router.predict and Agent.system_one both return the system_one payload,
    # which always carries an "answers" object (empty for empty questions).
    try:
        answers = result["answers"]
    except KeyError:
        raise ToolError("internal_error", "predict result has no 'answers' object") from None
    return {"answers": _normalize_answers(answers), "routing": result.get("routing")}


def _reading_device(router: Any, agent: Any, model_name: str, routing: dict | None) -> str | None:
    """Real device of the checkpoint that answered: Agent.device reflects a
    silent GPU -> CPU fallback. Omitted when it cannot be read, rather than
    guessed.
    """
    if model_name != "auto" and agent is not None:
        return agent_device(agent)
    model_used = (routing or {}).get("model")
    if isinstance(model_used, str) and model_used:
        return agent_device(router_agent(router, model_used))
    return None


def laya_predict(
    state: Any,
    questions: Any,
    model: Any = "auto",
    *,
    task: Any = None,
    lang: Any = None,
    lang_guess: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    min_confidence: Any = None,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Typed questions, one forward pass.

    ``router`` is used when model == "auto"; ``agent`` for a direct checkpoint.

    ``task``/``lang`` are the router's own overrides (an explicit ``model`` outranks an explicit
    ``task``, which outranks an explicit ``lang``); ``lang_guess`` is the router's soft hint that sits
    below ``lang`` and above built-in detection -- it only participates in routing, so like ``task``
    it is refused on a pinned model and never reaches a direct agent; ``max_len``/``head_max_len``
    override the token budget the answering checkpoint was configured with; ``min_confidence`` flags
    any answer whose calibrated confidence falls below it with ``low_confidence: true``. None of them
    is forwarded unless set, so a router or agent that predates those keywords keeps working.
    """
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
    auto_without_router = False
    budget = _overrides(
        validate_task(task),
        validate_lang(lang),
        validate_budget(max_len, "max_len"),
        validate_budget(head_max_len, "head_max_len"),
    )
    lang_guess_code = validate_lang_guess(lang_guess)
    if lang_guess_code is not None:
        # Only when set, so a call that passes no hint stays byte-identical to the one that made
        # no ``lang_guess`` keyword exist, and a deployment's ``Router(lang_guess=...)`` still answers.
        budget["lang_guess"] = lang_guess_code
    min_conf = validate_min_confidence(min_confidence)
    if min_conf is not None:
        # Only when set, so a call that abstains over nothing stays byte-identical to the one
        # that made no ``min_confidence`` keyword exist.
        budget["min_confidence"] = min_conf
    # `task` picks a checkpoint by saying what the work is, so it means nothing once one is pinned:
    # `Router._route` checks an explicit `model` first and never reaches the task, and
    # `Agent.system_one` does not accept the keyword at all. Both would answer as if it were unset,
    # so it is refused here instead.
    if model_name != AUTO and "task" in budget:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )
    # `lang_guess` is routing-only in the same way, and `Agent.predict` does not accept it, so a
    # pinned call that set it would either be ignored (router) or crash (direct agent).
    if model_name != AUTO and "lang_guess" in budget:
        raise ToolError(
            "invalid_lang_guess",
            "lang_guess routes between checkpoints; with a pinned model there is nothing to route",
        )

    def _run() -> Any:
        if model_name == "auto":
            if router is None:
                if agent is None:
                    raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
                # An Agent answers but does not route: run it directly and report
                # a null model below, instead of echoing 'auto' (#444). `task` and `lang_guess` only
                # choose between checkpoints, so they have nothing to do here.
                nonlocal auto_without_router
                auto_without_router = True
                return agent.predict(
                    state_d,
                    questions_d,
                    **{k: v for k, v in budget.items() if k not in ("task", "lang_guess")},
                )
            return router.predict(state_d, questions_d, **budget)
        if agent is not None:
            return agent.predict(state_d, questions_d, **budget)
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        return router.predict(state_d, questions_d, model=model_name, **budget)

    started = time.perf_counter()
    try:
        result = _run()
    except Exception as exc:
        _raise_if_remote_error(exc)
        raise
    latency_ms = (time.perf_counter() - started) * 1000.0

    norm = _normalize_result(result)
    answers = norm["answers"]
    if auto_without_router:
        routing = {"model": None, "repo": None, "reason": "auto routing without router"}
    elif norm["routing"]:
        routing = norm["routing"]
    elif model_name == "auto":
        # Agent.predict() returns the system_one payload, which carries no 'routing' key
        # because the Agent itself does no routing. 'auto' is a routing directive, not a
        # checkpoint name, so it must never be echoed back as the answering model (#444).
        routing = {"model": None, "repo": None, "reason": "auto routing without router"}
    else:
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
    # With no routing decision there is no checkpoint to read the device from.
    device = (
        None if auto_without_router else _reading_device(router, agent, model_name, norm["routing"])
    )
    out: dict[str, Any] = {
        "answers": answers,
        "routing": routing,
        "latency_ms": round(latency_ms, 3),
    }
    if device:
        out["device"] = device
    return out


def _decision_to_dict(decision: Any) -> dict:
    if isinstance(decision, dict):
        return {
            "model": decision.get("model"),
            "repo": decision.get("repo"),
            "reason": decision.get("reason"),
        }
    return {
        "model": getattr(decision, "model", None),
        "repo": getattr(decision, "repo", None),
        "reason": getattr(decision, "reason", None),
    }


def _validate_explain(explain: Any) -> bool:
    if not isinstance(explain, bool):
        raise ToolError("invalid_explain", "explain must be a boolean")
    return explain


def _call_route(route: Callable, *args, explain: bool, **kwargs) -> Any:
    if not explain:
        return route(*args, **kwargs)
    try:
        return route(*args, explain=True, **kwargs)
    except TypeError as exc:
        # Only a keyword binding refusal, not a TypeError raised inside the router.
        # Never retry without the explanation the caller explicitly requested.
        if exc.__traceback__.tb_next is None and "unexpected keyword argument 'explain'" in str(
            exc
        ):
            raise ToolError("unsupported_explain", "router does not support explain=True") from exc
        raise


def _route_to_dict(decision: Any, *, explain: bool) -> dict:
    # laya_shortlist and laya_tournament share _decision_to_dict; keep their routing output unchanged.
    out = _decision_to_dict(decision)
    if explain:
        detection = (
            decision.get("detection")
            if isinstance(decision, dict)
            else getattr(decision, "detection", None)
        )
        out["diagnostic"] = detection.get("diagnostic") if isinstance(detection, dict) else None
    return out


def laya_route(
    state: Any,
    questions: Any,
    *,
    model: Any = None,
    task: Any = None,
    lang: Any = None,
    lang_guess: Any = None,
    explain: Any = False,
    router: Any = None,
) -> dict:
    """Routing decision only: no forward pass.

    Takes the routing overrides :meth:`laya_predict` takes -- ``model``, ``task``, ``lang``,
    ``lang_guess`` -- so "which checkpoint would this go to?" can be asked under a pin without
    running anything.
    Passing a predict call's controls here reproduces the ``routing`` block it returned, which is
    what makes a route a cheap explanation of a decision rather than a different decision.

    ``explain=True`` adds the final decision's built-in detection diagnostic, or ``None`` when
    none exists. It never re-runs detection or exposes the rest of the detection payload.
    """
    explain = _validate_explain(explain)
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    # `auto` is this layer's word for "do not pin"; core has no such name, so it becomes an absent
    # override rather than a ValueError from normalise_name.
    model_name = validate_model(model)
    task_name = validate_task(task)
    lang_guess_code = validate_lang_guess(lang_guess)
    # Same rule as the decision tools: `_route` checks an explicit model first and never reaches
    # the task, so a call that sets both is asking a question with two answers.
    if model_name != AUTO and task_name is not None:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )
    # `lang_guess` routes in the same place, so it is equally meaningless once a checkpoint is pinned.
    if model_name != AUTO and lang_guess_code is not None:
        raise ToolError(
            "invalid_lang_guess",
            "lang_guess routes between checkpoints; with a pinned model there is nothing to route",
        )
    overrides: dict[str, Any] = {}
    if model_name != AUTO:
        overrides["model"] = model_name
    if task_name is not None:
        overrides["task"] = task_name
    lang_code = validate_lang(lang)
    if lang_code is not None:
        overrides["lang"] = lang_code
    if lang_guess_code is not None:
        overrides["lang_guess"] = lang_guess_code
    # Everything the caller typed is checked before the server is asked for a checkpoint.
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "route"):
        raise ToolError("internal_error", "router has no route() method")
    decision = _call_route(router.route, state_d, questions_d, explain=explain, **overrides)
    return _route_to_dict(decision, explain=explain)


# Where the reusable shortlist embedder lives. Kept private and on the agent, so the
# cache's lifetime is the answering checkpoint's rather than this process's.
_EMBED_CACHE_ATTR = "_laya_mcp_shortlist_embed_fn"


def _detected_language(detection: Any) -> str | None:
    """The language a route decision detected, or None when it reported none.

    Accepts either shape ``Router.route`` returns -- a ``RouteDecision`` (a dict)
    or a plain object -- and tolerates a decision that carries no detection at all,
    an explicit-model decision being the normal case.
    """
    if detection is None:
        return None
    if isinstance(detection, dict):
        lang = detection.get("language")
    else:
        lang = getattr(detection, "language", None)
    if isinstance(lang, str) and lang:
        return lang
    return None


def _resident_or_load(router: Any, name: str) -> Any:
    """The resident agent for checkpoint ``name``, loading on demand when possible.

    Read-only lookup first: ``router_agent`` never calls ``load()``. A miss
    falls back to ``Router.load``, the same on-demand build ``Router.predict``
    performs after routing, so a lazily preloaded server (LAYA_PRELOAD=0, or a
    checkpoint outside LAYA_MODELS) behaves exactly like ``laya_predict``.
    """
    resident = router_agent(router, name)
    if resident is not None:
        return resident
    if hasattr(router, "base_url") and hasattr(router, "health"):
        raise ToolError(
            "unsupported_remote",
            "laya_shortlist embeds options with the answering checkpoint in-process; it is not "
            "available when LAYA_BASE_URL points the MCP server at a remote laya-serve. Use "
            "laya_predict with head_max_len raised, or run the MCP server without LAYA_BASE_URL.",
        )
    load = getattr(router, "load", None)
    if load is None:
        raise ToolError("models_not_ready", f"checkpoint {name!r} is not loaded")
    return load(name)


def _embed_fn_with_cache(agent: Any, embed_fn_from_agent: Any, cached_embed_fn: Any) -> Any:
    """An embedder for ``agent`` whose rows survive between shortlist calls.

    ``predict_shortlist`` embeds the query plus every option text on each call, so a
    fixed option list is re-encoded on every request even though it never changes.
    The ``cached_embed_fn`` wrapper from #405 fixes that, but only if the same
    wrapper is reused across calls.

    The wrapper is stored on the answering agent itself, so its lifetime is exactly
    that checkpoint's: another checkpoint, or the same checkpoint reloaded into a
    fresh ``Agent``, starts with an empty cache and can never reuse another
    checkpoint's rows. No module-level state is involved, and the cache stays bounded
    by ``cached_embed_fn``'s own LRU limit.
    """
    cached = getattr(agent, _EMBED_CACHE_ATTR, None)
    if cached is not None:
        return cached
    cached = cached_embed_fn(embed_fn_from_agent(agent))
    try:
        setattr(agent, _EMBED_CACHE_ATTR, cached)
    except (AttributeError, TypeError):
        # An agent that will not take attributes simply gets no reuse between calls,
        # which is the behaviour before this cache existed.
        pass
    return cached


def laya_shortlist(
    state: Any,
    questions: Any,
    model: Any = "auto",
    k: Any = None,
    *,
    task: Any = None,
    lang: Any = None,
    lang_guess: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    min_confidence: Any = None,
    router: Any = None,
    agent: Any = None,
    embed_fn: Callable[[Sequence[str]], Any] | None = None,
) -> dict:
    """Shortlist many-option choice questions to ``k`` labels, then one predict.

    The shared guardrails tell clients not to run >20-option choice questions
    without shortlisting; this tool is that shortlisting (the in-process
    ``laya.shortlist.predict_shortlist`` pattern over MCP). Embeddings come
    from the answering checkpoint's own encoder (``embed_fn_from_agent``), so
    no extra model is downloaded; ``embed_fn`` is injectable for tests or for
    a dedicated bi-encoder.

    ``routing`` reports the real route decision in auto mode (the forward
    pass then runs with an explicit ``model=``, so routing happens once).

    ``head_max_len`` matters more here than anywhere else: shortlisting exists
    because a large label set shares that budget, and narrowing to ``k`` is only
    half of the fix. The budget override reaches the answering forward pass;
    ``task``/``lang``/``lang_guess`` reach the route that chose the checkpoint
    (``lang_guess`` only routes, so like ``task`` it is stripped before the answering pass);
    ``min_confidence`` flags a kept-label answer the checkpoint is unsure of.

    A question that actually narrows cannot keep the caller's ``option_order``: the
    order covers every option, the answer covers the kept labels only, and remapping
    it onto them would silently move options between slots, so core drops it. An MCP
    client never sees core's ``RuntimeWarning`` about the drop, so the response
    reports it itself as ``option_order_dropped``, per question, with the old and kept
    option counts. A question answered as given (``n <= k``), a non-choice question
    and a call without an order keep the response they had before.
    """
    # Before the heavy imports below: a remote router has no in-process encoder to embed with,
    # and the refusal should not cost the MCP process a torch import on the way to saying so.
    if (
        router is not None
        and agent is None
        and hasattr(router, "base_url")
        and hasattr(router, "health")
    ):
        raise ToolError(
            "unsupported_remote",
            "laya_shortlist embeds options with the answering checkpoint in-process; it is not "
            "available when LAYA_BASE_URL points the MCP server at a remote laya-serve. Use "
            "laya_predict with head_max_len raised, or run the MCP server without LAYA_BASE_URL.",
        )
    # Lazy: keeps numpy/shortlist out of module import for laya.mcp.tools.
    from laya_mlx.shortlist import (
        DEFAULT_SHORTLIST_K,
        cached_embed_fn,
        embed_fn_from_agent,
        predict_shortlist,
    )

    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
    routing_overrides = _overrides(validate_task(task), validate_lang(lang), None, None)
    lang_guess_code = validate_lang_guess(lang_guess)
    if lang_guess_code is not None:
        routing_overrides["lang_guess"] = lang_guess_code
    budget = _overrides(
        None,
        None,
        validate_budget(max_len, "max_len"),
        validate_budget(head_max_len, "head_max_len"),
    )
    min_conf = validate_min_confidence(min_confidence)
    if min_conf is not None:
        # Forwarded to the answering predict the same way the budget is, so a shortlisted field the
        # model reads with low confidence comes back flagged rather than as a confident-looking pick.
        budget["min_confidence"] = min_conf
    # Same rule as `laya_predict`: a checkpoint is already pinned, so `task` has nothing left to
    # decide -- and it is a routing keyword, which the answering `system_one` does not accept.
    if model_name != AUTO and "task" in routing_overrides:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )
    # `lang_guess` is routing-only in the same way, so a pinned call has nothing for it to choose.
    if model_name != AUTO and "lang_guess" in routing_overrides:
        raise ToolError(
            "invalid_lang_guess",
            "lang_guess routes between checkpoints; with a pinned model there is nothing to route",
        )
    # `lang` survives pinning because it means two things: it can route, and on the answering
    # checkpoint it selects the per-language temperature table.
    forward_overrides = {
        name: value
        for name, value in routing_overrides.items()
        if name not in ("task", "lang_guess")
    }
    if k is None:
        k = DEFAULT_SHORTLIST_K
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ToolError("invalid_k", f"k must be a positive integer, got {k!r}")

    routing: dict[str, Any]
    if model_name == "auto":
        if router is None:
            raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
        if not hasattr(router, "route"):
            raise ToolError("internal_error", "router has no route() method")
        decision = router.route(state_d, questions_d, **routing_overrides)
        routing = _decision_to_dict(decision)
        routed = routing["model"]
        detection = (
            decision.get("detection")
            if isinstance(decision, dict)
            else getattr(decision, "detection", None)
        )
        if not isinstance(routed, str) or not routed:
            raise ToolError("internal_error", "router.route returned no model")
        predict_target = router
        # The route already happened here, so the forward pass pins that checkpoint instead of
        # routing a second time.
        predict_kwargs: dict[str, Any] = {"model": routed, **forward_overrides, **budget}
        # Answering with an explicit model= makes predict re-route, and an
        # explicit-model decision carries no detection, so the language detected
        # here would otherwise be lost and the checkpoint's per-language
        # temperatures would not apply the way they do in laya_predict. Forward
        # it when the route reported one and the caller did not name one.
        detected = _detected_language(detection)
        if detected is not None and "lang" not in predict_kwargs:
            predict_kwargs["lang"] = detected
        embed_agent = _resident_or_load(router, routed)
    else:
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
        if agent is not None:
            predict_target = agent
            predict_kwargs = dict(forward_overrides)
            embed_agent = agent
        else:
            if router is None:
                raise ToolError("models_not_ready", "no agent/router loaded")
            predict_target = router
            predict_kwargs = {"model": model_name, **forward_overrides}
            embed_agent = _resident_or_load(router, model_name)
        predict_kwargs.update(budget)

    if embed_fn is None:
        try:
            embed_fn = _embed_fn_with_cache(embed_agent, embed_fn_from_agent, cached_embed_fn)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ToolError(
                "models_not_ready",
                f"cannot build shortlist embeddings from checkpoint {routing['model']!r}: {exc}",
            ) from exc

    started = time.perf_counter()
    result = predict_shortlist(
        predict_target, state_d, questions_d, embed_fn, k=k, **predict_kwargs
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not isinstance(result, dict):
        raise ToolError("internal_error", "predict returned non-object")
    answers = _normalize_answers(result["answers"])
    # A narrowed choice no longer fits the caller's full-length order -- core drops the order
    # rather than remapping it onto the kept labels, which would silently move options between
    # slots (#1063), and says so with a RuntimeWarning that stays inside the Python process.
    # An MCP client reads JSON, so the drop is reported here instead, per question, derived
    # from the original question and the returned shortlist meta. Capturing core's warning
    # would be the other route, but warnings are process-global and a concurrent tool call
    # could supply the drop this call reports. Only a valid order reaches this point:
    # validate_questions refuses anything that is not a permutation up front.
    shortlist_meta = result.get("shortlist") or {}
    dropped_orders: dict[str, Any] = {}
    for name, spec in questions_d.items():
        if spec["type"] != "choice" or "option_order" not in spec:
            continue
        entry = shortlist_meta.get(name)
        if not entry or entry.get("passthrough"):
            continue
        dropped_orders[name] = {"n": len(spec["criteria"]), "kept": len(entry["labels"])}
    # The answering checkpoint is the embedding checkpoint in every branch.
    device = agent_device(embed_agent)
    out: dict[str, Any] = {
        "answers": answers,
        "routing": routing,
        "shortlist": shortlist_meta,
        "latency_ms": round(latency_ms, 3),
    }
    # Present only when an order was actually dropped: every other response is unchanged.
    if dropped_orders:
        out["option_order_dropped"] = dropped_orders
    if device:
        out["device"] = device
    return out


def laya_tournament(
    state: Any,
    questions: Any,
    model: Any = "auto",
    group_size: Any = 16,
    *,
    task: Any = None,
    lang: Any = None,
    lang_guess: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    min_confidence: Any = None,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Answer large choices with core's balanced tournament in-process.

    Select the checkpoint from the original request, then pin it for every round.
    Each round uses ``laya_predict``'s validation, answer shape and device reporting.
    Probabilities and confidence describe the final call's finalists; ``tournament``
    adds their ``labels``, the original option count ``n`` and elimination ``rounds``.
    Small choices report zero rounds; non-choice questions reach the final call.
    ``latency_ms`` covers routing and all rounds. A narrowed question's discarded
    ``option_order`` is reported with its original and kept option counts.

    Remote callers use ``laya_predict`` with the HTTP server's tournament enabled;
    its probability and usage contract differs from this core helper's contract.
    """
    if (
        router is not None
        and agent is None
        and hasattr(router, "base_url")
        and hasattr(router, "health")
    ):
        raise ToolError(
            "unsupported_remote",
            "laya_tournament runs core tournament rounds in-process; it is not available when "
            "LAYA_BASE_URL points at laya-serve. Use laya_predict with LAYA_TOURNAMENT_GROUP "
            "enabled on that server, or run the MCP server without LAYA_BASE_URL.",
        )
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ToolError(
            "invalid_group_size", f"group_size must be an integer of at least 2, got {group_size!r}"
        )
    routing_overrides = _overrides(validate_task(task), validate_lang(lang), None, None)
    hint = validate_lang_guess(lang_guess)
    if hint is not None:
        routing_overrides["lang_guess"] = hint
    forward = _overrides(
        None,
        None,
        validate_budget(max_len, "max_len"),
        validate_budget(head_max_len, "head_max_len"),
    )
    min_conf = validate_min_confidence(min_confidence)
    if min_conf is not None:
        forward["min_confidence"] = min_conf
    for name in ("task", "lang_guess"):
        if model_name != AUTO and name in routing_overrides:
            raise ToolError(
                f"invalid_{name}",
                f"{name} routes between checkpoints; with a pinned model there is nothing to route",
            )
    if "lang" in routing_overrides:
        forward["lang"] = routing_overrides["lang"]

    from laya_mlx.shortlist import predict_tournament

    started = time.perf_counter()
    routing = None
    predict_agent = agent
    if model_name == AUTO and router is not None:
        if not hasattr(router, "route"):
            raise ToolError("internal_error", "router has no route() method")
        decision = router.route(state_d, questions_d, **routing_overrides)
        routing = _decision_to_dict(decision)
        model_name = routing["model"]
        if not isinstance(model_name, str) or not model_name:
            raise ToolError("internal_error", "router.route returned no model")
        detection = (
            decision.get("detection")
            if isinstance(decision, dict)
            else getattr(decision, "detection", None)
        )
        detected = _detected_language(detection)
        if detected is not None and "lang" not in forward:
            forward["lang"] = detected
        predict_agent = None

    class Predictor:
        def predict(self, round_state, round_questions):
            result = laya_predict(
                round_state,
                round_questions,
                model=model_name,
                router=router,
                agent=predict_agent,
                **forward,
            )
            for name, spec in round_questions.items():
                if spec["type"] != "choice":
                    continue
                answer = result["answers"].get(name)
                if (
                    not isinstance(answer, dict)
                    or not isinstance(answer.get("choice"), str)
                    or answer["choice"] not in spec["criteria"]
                ):
                    raise ToolError(
                        "internal_error",
                        f"tournament prediction for question {name!r} must return a choice from its criteria",
                    )
            return result

    out = predict_tournament(Predictor(), state_d, questions_d, group_size=group_size)
    if routing is not None:
        out["routing"] = routing
    out["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    dropped_orders = {
        name: {"n": entry["n"], "kept": len(entry["labels"])}
        for name, entry in out["tournament"].items()
        if entry["rounds"] and "option_order" in questions_d[name]
    }
    if dropped_orders:
        out["option_order_dropped"] = dropped_orders
    return out


def laya_predict_long(
    state: Any,
    questions: Any,
    model: Any = "auto",
    *,
    window: Any = None,
    stride: Any = None,
    task: Any = None,
    lang: Any = None,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Scan a state longer than the context window, then answer.

    The in-process ``predict_long`` pattern over MCP: the state is tokenized once, scored in
    overlapping windows in shared forward passes, and aggregated per question (strongest window
    for ``noul``, most-confident window for ``choice``/``score``). Use this instead of
    ``laya_predict`` when the state does not fit one window -- ``laya_predict`` answers from
    its first window and never reads the rest.

    ``router`` is used when model == "auto"; ``agent`` for a direct checkpoint. ``task``/``lang``
    are the router's own overrides (an explicit ``model`` outranks an explicit ``task``, which
    outranks an explicit ``lang``); ``window``/``stride`` size the scan the way core's
    ``predict_long`` sizes it, and only set values are forwarded, so a checkpoint that predates
    those keywords keeps working. ``max_len``/``head_max_len``/``min_confidence`` are not
    accepted here because core's ``predict_long`` takes none of them: a window is sized by
    ``window`` or the checkpoint budget, and there is no abstention gate on the scan.

    Each answer carries the deciding window's ``window`` attribution (index and token offsets
    into the tokenized state), and ``windows`` counts the windows the model scored.
    """
    if (
        router is not None
        and agent is None
        and hasattr(router, "base_url")
        and hasattr(router, "health")
    ):
        raise ToolError(
            "unsupported_remote",
            "laya_predict_long scans the state window by window in-process; laya-serve has no "
            "long-state endpoint, so it is not available when LAYA_BASE_URL points at laya-serve. "
            "Run the MCP server without LAYA_BASE_URL to scan locally, or use laya_predict, which "
            "answers from the first window only.",
        )
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
    # Only what the caller actually set is forwarded: `_overrides` is not reused because its keys
    # are the single-window budgets, and `predict_long` rejects those -- the scan is sized by
    # `window`/`stride` or the checkpoint budget instead.
    scan = {
        name: value
        for name, value in (
            ("task", validate_task(task)),
            ("lang", validate_lang(lang)),
            ("window", validate_budget(window, "window")),
            ("stride", validate_budget(stride, "stride")),
        )
        if value is not None
    }
    # `task` picks a checkpoint by saying what the work is, so it means nothing once one is pinned:
    # same rule as `laya_predict` -- the router checks an explicit `model` first and never reaches
    # the task, and a checkpoint's `predict_long` does not accept the keyword at all.
    if model_name != AUTO and "task" in scan:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )
    auto_without_router = False

    def _run() -> Any:
        if model_name == "auto":
            if router is None:
                if agent is None:
                    raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
                # An Agent answers but does not route: run it directly and report
                # a null model below, instead of echoing 'auto' (#444). `Agent.predict_long`
                # has no `task` parameter -- only a Router routes between checkpoints -- so the
                # keyword is dropped here exactly as `laya_predict` drops it (`tools.py:415`);
                # forwarding it would surface as a raw TypeError, not an actionable refusal.
                nonlocal auto_without_router
                auto_without_router = True
                return _scan(
                    agent, state_d, questions_d, **{k: v for k, v in scan.items() if k != "task"}
                )
            return _scan(router, state_d, questions_d, **scan)
        if agent is not None:
            return _scan(agent, state_d, questions_d, **scan)
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        return _scan(router, state_d, questions_d, model=model_name, **scan)

    started = time.perf_counter()
    result = _run()
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not isinstance(result, dict):
        raise ToolError("internal_error", "predict_long returned non-object")
    answers = _normalize_answers(result.get("answers"))
    usage = result.get("usage") or {}
    windows = usage.get("windows", 0) if isinstance(usage, dict) else 0
    if auto_without_router:
        routing = {"model": None, "repo": None, "reason": "auto routing without router"}
    elif model_name == "auto":
        # A bare agent's scan carries no 'routing' key because the Agent itself does no
        # routing; 'auto' is a routing directive, not a checkpoint name (#444). A Router's
        # scan carries the decision it routed with.
        routing = result.get("routing") or {
            "model": None,
            "repo": None,
            "reason": "auto routing without router",
        }
    else:
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
    device = (
        None
        if auto_without_router
        else _reading_device(router, agent, model_name, result.get("routing"))
    )
    out: dict[str, Any] = {
        "answers": answers,
        "routing": routing,
        "windows": windows,
        "latency_ms": round(latency_ms, 3),
    }
    if device:
        out["device"] = device
    return out


def _scan(target: Any, state: Any, questions: Any, **kwargs: Any) -> Any:
    """Call ``predict_long`` on a router or agent, refusing clearly without it.

    A checkpoint old enough to predate the scan (or a hand-rolled stand-in) has no
    ``predict_long``; falling back to ``predict`` would silently answer from the first
    window, which is exactly what this tool exists to avoid.
    """
    fn = getattr(target, "predict_long", None)
    if not callable(fn):
        raise ToolError("internal_error", "%s has no predict_long() method" % type(target).__name__)
    return fn(state, questions, **kwargs)


def laya_preset(
    preset: Any,
    state: Any,
    *,
    task: Any = None,
    lang: Any = None,
    lang_guess: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    min_confidence: Any = None,
    router: Any = None,
    agent: Any = None,
    preset_builder: Callable[[str], dict] | None = None,
) -> dict:
    """Run a built-in workflow preset (guard / moderation / triage / model_router / email).

    A preset's questions read one named field of the state -- ``guard`` asks about `` `prompt` ``,
    ``triage`` about `` `message` `` -- and a caller that hands over its text under any other key is
    asked to trust an answer about a field that is not there. So a state that is one string gets
    placed under the field the questions actually name; anything richer than that is the caller's
    shape and is passed through untouched.

    The preset fixes the questions, not the route or the budget, so the per-call controls a
    hand-written :func:`laya_predict` takes -- ``task``/``lang``/``lang_guess``/``max_len``/
    ``head_max_len`` and ``min_confidence`` -- are available here too; most usefully ``lang``, since a
    preset's instructions are English text whatever state they read.
    """
    preset_name = validate_preset(preset)
    state_d = validate_state(state)
    if preset_builder is None:
        raise ToolError("internal_error", "preset_builder is not configured")
    questions = preset_builder(PRESETS[preset_name])
    field = state_field(questions)
    if field is not None and field not in state_d and len(state_d) == 1:
        ((key, value),) = state_d.items()
        if isinstance(value, str):
            state_d = {field: value}
    return laya_predict(
        state_d,
        questions,
        model="auto",
        task=task,
        lang=lang,
        lang_guess=lang_guess,
        max_len=max_len,
        head_max_len=head_max_len,
        min_confidence=min_confidence,
        router=router,
        agent=agent,
    )


def _remote_status(router: Any, preload: bool) -> dict:
    """``laya_status`` for a remote router: the server's own /health, no torch import here."""
    versions: dict[str, str | None] = {"laya": None}
    try:
        versions["laya"] = getattr(__import__("laya_mlx"), "__version__", "unknown")
    except Exception:
        pass
    out: dict[str, Any] = {
        "mode": "remote",
        "base_url": router.base_url,
        "router_preload": False,
        "router_ready": True,
        "package_versions": versions,
    }
    try:
        health = router.health()
    except Exception as exc:  # noqa: BLE001 -- the status tool reports, it never raises
        out["server"] = None
        out["server_error"] = f"{type(exc).__name__}: {exc}"
        out["loaded"] = []
        return out
    out["server"] = health
    loaded = health.get("loaded")
    out["loaded"] = [v for v in loaded if isinstance(v, str)] if isinstance(loaded, list) else []
    for key in ("device", "device_is_preference", "checkpoint_devices"):
        if key in health:
            out[key] = health[key]
    return out


def laya_status(
    *, router: Any = None, loaded: list[str] | None = None, preload: bool = True
) -> dict:
    if router is not None and hasattr(router, "base_url") and hasattr(router, "health"):
        return _remote_status(router, preload)
    report = device_report()
    versions: dict[str, str | None] = {
        "laya": None,
        "torch": report.get("torch_version"),
        "transformers": None,
    }
    for pkg in ("laya", "transformers"):
        try:
            if pkg == "laya":
                from laya_mlx import __version__

                versions[pkg] = __version__
            else:
                from importlib.metadata import version

                versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = None

    if loaded is None and router is not None:
        try:
            loaded = list(getattr(router, "loaded", []) or [])
        except Exception:
            loaded = []

    # Real device of every loaded checkpoint (Agent.device reflects a silent
    # GPU -> CPU fallback). The top-level "device" is that fact when something
    # is loaded; before any load it is the configured preference (LAYA_DEVICE
    # or auto), which "device_is_preference" flags as such.
    checkpoint_devices: dict[str, str] = {}
    for name in loaded or []:
        device = agent_device(router_agent(router, name))
        if device:
            checkpoint_devices[name] = device
    actual = next(iter(checkpoint_devices.values()), None)

    return {
        **report,
        "device": actual or report["device"],
        "device_is_preference": actual is None,
        "checkpoint_devices": checkpoint_devices,
        "loaded": list(loaded or []),
        "router_preload": bool(preload),
        "router_ready": router is not None,
        "package_versions": versions,
    }


# --- batch tools ----------------------------------------------------------


def _validate_batch_model(value: Any, where: str) -> str | None:
    """``model`` as a Router routing override: "auto" (or absent) means None so
    Router.route resolves the checkpoint per request; a name pins it. Kept
    separate from ``validate_model``, whose "auto" means "answer via
    router.predict" on the single-request path.
    """
    if value is None:
        return None
    try:
        name = validate_model(value)
    except ToolError as error:
        raise ToolError("invalid_model", "%s['model']: %s" % (where, error)) from None
    return None if name == AUTO else name


def _validate_batch_str(value: Any, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ToolError("invalid_request", f"{where} must be a non-empty string, got {value!r}")
    return value


def _validate_batch_budget(value: Any, name: str, where: str) -> int | None:
    """One of the two per-request token budgets, with the batch item named in the message.

    Same guard and same error code as the single-request tools take, so `invalid_max_len` means
    one thing whichever tool a client called.
    """
    if value is None:
        return None
    try:
        return validate_budget(value, name)
    except ToolError as error:
        raise ToolError("invalid_%s" % name, "%s[%r]: %s" % (where, name, error)) from None


def _validate_batch_item(request: Any, i: int) -> dict:
    where = "requests[%d]" % i
    if not isinstance(request, dict):
        raise ToolError(
            "invalid_request", "%s must be an object with 'state' and 'questions'" % where
        )
    item: dict[str, Any] = {
        "state": validate_state(request.get("state")),
        "questions": validate_questions(request.get("questions")),
    }
    if "model" in request:
        model = _validate_batch_model(request["model"], where)
        if model is not None:
            item["model"] = model
    for key in ("task", "lang"):
        if key in request:
            item[key] = _validate_batch_str(request[key], "%s[%r]" % (where, key))
    # The router reads `task` only when no model is pinned, so that is when a name nothing answers
    # to must be refused here: past this point it is a ValueError and the client sees internal_error.
    if "task" in item and "model" not in item:
        try:
            validate_task(item["task"])
        except ToolError as error:
            raise ToolError("invalid_task", "%s['task']: %s" % (where, error)) from None
    if "lang_guess" in request:
        # Type-check it the way the single-request path does (`validate_lang_guess`), instead of
        # passing it through raw: a non-string (e.g. a JSON object) otherwise reached
        # `_english_from_code(dict)`, stringified to something that matches no English subtag, and
        # silently routed that item to the multilingual checkpoint instead of raising cleanly.
        item["lang_guess"] = validate_lang_guess(request["lang_guess"])
    # `Router.predict_batch` reads both off the request dict and splits requests that ask for
    # different budgets into separate forward passes, so an item that names one must keep it.
    for key in ("max_len", "head_max_len"):
        if key in request:
            budget = _validate_batch_budget(request[key], key, where)
            if budget is not None:
                item[key] = budget
    return item


# The per-request overrides a batch item may carry. Written out from here rather than retyped,
# because the key list a client sees is the key list `_validate_batch_item` keeps: an enumeration
# that drops one advertises a control that would be ignored, and an enumeration that gains one
# rejects a request the tool would have answered.
BATCH_ITEM_OVERRIDES = ("model", "task", "lang", "lang_guess", "max_len", "head_max_len")


def batch_item_key_doc(omit: tuple = ()) -> str:
    """The batch item shape as one phrase: ``{state, questions, model?, task?, ...}``.

    `omit` drops the overrides a given tool has no use for -- the token budgets are kept by the
    shared validator but mean nothing to `route_batch`, which never runs a forward pass.
    """
    keys = [key for key in BATCH_ITEM_OVERRIDES if key not in omit]
    return "{state, questions, %s} objects" % ", ".join("%s?" % key for key in keys)


def validate_batch_requests(requests: Any) -> list[dict]:
    """Validate a tool payload of many requests, preserving order.

    Same per-item validation as ``laya_predict``/``laya_route`` (state, questions,
    optional model/task/lang/lang_guess/max_len/head_max_len overrides), run before any model
    loads so one malformed item fails the whole call instead of a partial batch.
    """
    if not isinstance(requests, list) or not requests:
        raise ToolError(
            "invalid_request",
            "requests must be a non-empty array of %s" % batch_item_key_doc(),
        )
    return [_validate_batch_item(request, i) for i, request in enumerate(requests)]


def _validate_batch_size(batch_size: Any) -> int | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ToolError(
            "invalid_batch_size", f"batch_size must be a positive integer, got {batch_size!r}"
        )
    return batch_size


def _validate_hooks_timeout(value: Any) -> float | None:
    # Mirrors laya_predict_batch's forwarding shape: unset stays unset so a
    # Router with its own default is not shadowed by 0, and core's
    # ``validate_timeout`` decides what counts as a real deadline. Wrapping
    # its ValueError as a ToolError keeps a bad arg from surfacing as
    # ``internal_error: ValueError`` at the MCP boundary. Bools are refused
    # here the same way _validate_batch_size refuses them, so a wrong type
    # reaches the caller as `invalid_hooks_timeout` instead of whichever of
    # core's two refusals happens to fire. Core refuses a bool as well now:
    # float(True) == 1.0 is no longer silently read as a one-second deadline.
    if value is None:
        return None
    if isinstance(value, bool):
        raise ToolError(
            "invalid_hooks_timeout",
            "hooks_timeout must be a positive number or None, got %r" % (value,),
        )
    try:
        return validate_timeout(value)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_hooks_timeout", str(exc)) from exc


def laya_predict_batch(
    requests: Any,
    batch_size: Any = None,
    hooks_timeout: Any = None,
    min_confidence: Any = None,
    sort_by_length: bool = False,
    *,
    router: Any = None,
) -> dict:
    """Answer many requests in one call: ``Router.predict_batch`` over MCP.

    ``laya_predict`` scores one request per round trip; an agent with many
    tickets to triage pays a routing pass and a separate forward pass every
    time. This tool hands the whole list to ``Router.predict_batch``, which
    groups requests by routed checkpoint and shares forward passes between
    requests with the same question schema, and returns the answers in input
    order. Every item carries the same ``answers``/``routing``/``device``
    fields as ``laya_predict``. Requests are validated up front, so one
    malformed item errors before any model loads.
    """
    items = validate_batch_requests(requests)
    size = _validate_batch_size(batch_size)
    # Validate the call-level controls up front, the way every sibling tool does, so a bad value is
    # a clean ToolError the client can read -- not an `internal_error: ValueError` that escapes the
    # TypeError-only handler below (min_confidence), and not a silent 1-second hook deadline from
    # core's `float(True) == 1.0` (hooks_timeout).
    mc = validate_min_confidence(min_confidence)
    ht = _validate_hooks_timeout(hooks_timeout)
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "predict_batch"):
        raise ToolError("internal_error", "router has no predict_batch() method")

    started = time.perf_counter()
    try:
        kwargs = {}
        if size is not None:
            kwargs["batch_size"] = size
        if ht is not None:
            kwargs["hooks_timeout"] = ht
        if mc is not None:
            kwargs["min_confidence"] = mc
        if sort_by_length:
            kwargs["sort_by_length"] = True
        results = router.predict_batch(items, **kwargs)
    except Exception as exc:
        _raise_if_remote_error(exc)
        if not isinstance(exc, TypeError):
            raise

        # A Router without batch support raises at the call itself; anything
        # else is a real bug and must surface unchanged.
        raise ToolError(
            "internal_error",
            "router.predict_batch failed: %s: %s" % (type(exc).__name__, exc),
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not isinstance(results, list) or len(results) != len(items):
        raise ToolError(
            "internal_error",
            "predict_batch returned %r results for %d requests"
            % (len(results) if isinstance(results, list) else type(results).__name__, len(items)),
        )

    out_requests: list[dict[str, Any]] = []
    for result in results:
        norm = _normalize_result(result)
        entry: dict[str, Any] = {"answers": norm["answers"], "routing": norm["routing"] or {}}
        device = _reading_device(router, None, "auto", norm["routing"])
        if device:
            entry["device"] = device
        out_requests.append(entry)

    model_counts: dict[str, int] = {}
    for entry in out_requests:
        name = entry["routing"].get("model")
        key = name if isinstance(name, str) and name else "unknown"
        model_counts[key] = model_counts.get(key, 0) + 1

    return {
        "requests": out_requests,
        "model_counts": model_counts,
        "total_latency_ms": round(latency_ms, 3),
        "per_request_latency_ms": round(latency_ms / len(items), 3),
    }


def laya_route_batch(
    requests: Any, hooks_timeout: Any = None, *, explain: Any = False, router: Any = None
) -> dict:
    """Routing decisions for many requests: no forward pass, no checkpoint loads.

    The batch form of ``laya_route``, mirroring ``Router.route_batch``: it
    reports which checkpoint each request *would* answer from so clients can
    inspect or aggregate a workload's routing before paying any load cost.
    ``hooks_timeout`` overrides the Router's own value for this call's
    ``on_route`` dispatch, exactly as it does for ``laya_predict_batch``: an
    operator-installed hook that hangs should not stall a whole routing sweep.
    ``explain=True`` adds a diagnostic (or ``None``) to each final decision,
    using the same contract as ``laya_route`` and preserving input order.
    """
    explain = _validate_explain(explain)
    items = validate_batch_requests(requests)
    timeout = _validate_hooks_timeout(hooks_timeout)
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "route_batch"):
        raise ToolError("internal_error", "router has no route_batch() method")
    overrides = {"hooks_timeout": timeout} if timeout is not None else {}
    decisions = _call_route(router.route_batch, items, explain=explain, **overrides)
    if not isinstance(decisions, list) or len(decisions) != len(items):
        raise ToolError(
            "internal_error",
            "route_batch returned %r decisions for %d requests"
            % (
                len(decisions) if isinstance(decisions, list) else type(decisions).__name__,
                len(items),
            ),
        )
    out = [_route_to_dict(decision, explain=explain) for decision in decisions]
    model_counts: dict[str, int] = {}
    for entry in out:
        name = entry["model"]
        key = name if isinstance(name, str) and name else "unknown"
        model_counts[key] = model_counts.get(key, 0) + 1
    return {"decisions": out, "model_counts": model_counts}


def laya_decide(
    state: Any,
    schema: Any,
    model: Any = "auto",
    *,
    min_confidence: Any = None,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Answer a JSON-schema-shaped decision and return the decided values.

    The MCP form of ``laya.decide`` (the schema-driven API: an object of enum /
    bounded-integer / boolean properties is turned into Laya questions, answered
    in one forward pass, and projected back onto the schema). MCP clients carry
    JSON, not pydantic classes, so ``schema`` is a JSON schema dictionary. The
    answer is ``values`` -- enum members, integer levels, booleans -- beside the
    per-field ``confidence``/``probabilities`` and the usual routing/device/
    latency metadata, so a client never parses an answer map by hand.

    ``min_confidence`` is ``laya.decide``'s own abstention control, not a generic
    predict keyword: a field whose answer falls below it comes back as ``null`` in
    ``values`` -- the raw answer stays in ``confidence``/``probabilities`` -- so a
    client that trusts the values can tell "not confident" from a confident pick.
    """
    # Lazy: keeps laya.structured (pure Python, but a module import is still a
    # module import) out of this module's import-time surface.
    from laya_mlx.structured import SchemaError, decide

    state_d = validate_state(state)
    model_name = validate_model(model)
    min_conf = validate_min_confidence(min_confidence)
    try:
        # `decide` validates the schema itself (SchemaError names the offending
        # path) and projects the answers; calling it with return_details keeps
        # this tool on the core's exact semantics instead of a copied projection.
        started = time.perf_counter()
        if model_name == "auto":
            if router is None:
                raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
            details = decide(
                router, state_d, schema=schema, return_details=True, min_confidence=min_conf
            )
        elif agent is not None:
            details = decide(
                agent, state_d, schema=schema, return_details=True, min_confidence=min_conf
            )
        else:
            if router is None:
                raise ToolError("models_not_ready", "no agent/router loaded")
            details = decide(
                router,
                state_d,
                schema=schema,
                return_details=True,
                model=model_name,
                min_confidence=min_conf,
            )
    except SchemaError as exc:
        raise ToolError("invalid_schema", str(exc)) from exc
    except Exception as exc:
        _raise_if_remote_error(exc)
        raise
    latency_ms = (time.perf_counter() - started) * 1000.0

    answers = _normalize_answers(details.answers)  # noqa: F841
    routing = details.routing or {"model": model_name, "repo": None, "reason": "explicit model"}
    device = _reading_device(router, agent, model_name, details.routing)
    out: dict[str, Any] = {
        "values": details.values,
        "confidence": details.confidence,
        "probabilities": details.probabilities,
        "routing": routing,
        "latency_ms": round(latency_ms, 3),
    }
    if details.usage:
        out["usage"] = details.usage
    if device:
        out["device"] = device
    return out
