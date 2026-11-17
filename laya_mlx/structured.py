"""Schema-driven decisions: turn a JSON schema (or a pydantic model) into Laya questions.

Pure Python and torch-free, so `import laya` stays light. The mapping is a documented subset:
an object of properties, each an enum choice, a boolean, or a bounded integer scale. Anything
that cannot be answered from a fixed option set (free strings, arrays, nested objects) is
rejected with an error that names the path.

    import laya_mlx as laya

    class Ticket(BaseModel):
        department: Literal["billing", "support", "sales"]
        urgency: Literal[0, 1, 2]
        needs_human: bool

    values = laya.decide(agent, state, schema=Ticket)
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .confidence import answer_confidence_value, apply_confidence_gate, check_min_confidence

MAX_PROPERTIES = 32
MAX_OPTIONS = 32
MAX_SCORE_LEVELS = 10


class SchemaError(ValueError):
    """A schema cannot be expressed as Laya questions; the message names the path."""


@dataclass
class DecisionResult:
    """The detailed result of `decide(..., return_details=True)`.

    `values` is the schema-shaped output. `confidence` and `probabilities` are keyed by field,
    and `answers` is Laya's raw answer per field.

    `answer_confidence` is `max(p)` per field -- the probability mass on the answer being
    reported, under its own name. That makes it the same decision quantity `min_confidence`
    compares against and the calibration and eval stack measures, which is the reason to report
    it: a caller filtering this artifact to decide what to automate has to be filtering on the
    number the gate actually used.

    It is not a claim that the number is right. Reading it as "about c of the answers returned at
    c are correct" holds only after temperatures are fitted and validated on held-out data for
    that checkpoint and question shape; the shipped checkpoints are over-confident as shipped and
    `laya-multilingual` ships with no fitted temperatures at all. See `common.answer_confidence`
    and the README's Calibration section.

    `confidence` keeps the value each field's type has always produced: normalized entropy
    `1 - H(p) / log(k)` for `choice` and `score`, whose scale depends on the label count,
    and `max(p_true, 1 - p_true)` for `noul` -- a probability of the reported side, not an
    entropy, and not label-count dependent. `structured._details` reads the answer verbatim,
    so the formula is per-type; a single sentence cannot describe every field in this dict.
    A field that reported no usable `answer_confidence` maps to `None`, which is not the same
    as a reported `0.0`.
    """

    values: Dict[str, Any]
    confidence: Dict[str, float]
    probabilities: Dict[str, Dict[str, Any]]
    answers: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    routing: Optional[Dict[str, Any]] = None
    # Appended with a default so every existing construction of this dataclass keeps working, and
    # the first six fields keep their positions.
    answer_confidence: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class _Field:
    name: str
    kind: str  # "choice" | "score" | "noul"
    question: Dict[str, Any]
    options: List[Tuple[str, Any]] = field(default_factory=list)  # choice: (label, value)
    minimum: Optional[int] = None  # score: level 0 value


def _require_pydantic():
    try:
        import pydantic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pydantic is required for the typed-model helpers; install 'laya[structured]'"
        ) from exc


def _schema_of(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):  # pydantic v2
        return model.model_json_schema()
    if hasattr(model, "schema"):  # pydantic v1
        return model.schema()
    if isinstance(model, dict):
        return model
    raise SchemaError(
        "expected a JSON schema dict or a pydantic model, got %s" % type(model).__name__
    )


def _enum_field(path: str, name: str, values: Sequence[Any], description: Optional[str]) -> _Field:
    # A JSON 'enum' is an array. Reject a bare scalar before len(): a str is iterable, so without
    # this check it silently splits into per-character options instead of erroring (the module
    # promises malformed schemas raise a SchemaError that names the path).
    if not isinstance(values, (list, tuple)):
        raise SchemaError(
            "%s: 'enum' must be an array of values, got %s" % (path, type(values).__name__)
        )
    if len(values) > MAX_OPTIONS:
        raise SchemaError(
            "%s: %d options exceeds MAX_OPTIONS=%d" % (path, len(values), MAX_OPTIONS)
        )
    if not values:
        raise SchemaError("%s: 'enum' must not be empty" % path)
    if all(isinstance(v, bool) for v in values) and len(set(values)) == 2:
        return _noul_field(path, name, description)
    options = [(("null" if v is None else str(v)), v) for v in values]
    if len({label for label, _ in options}) != len(options):
        raise SchemaError("%s: enum values produce duplicate choice labels" % path)
    criteria = {label: None for label, _ in options}
    question = {
        "type": "choice",
        "instructions": description or ("What is `%s`?" % name),
        "criteria": criteria,
    }
    return _Field(name=name, kind="choice", question=question, options=options)


def _noul_field(path: str, name: str, description: Optional[str]) -> _Field:
    question = {
        "type": "noul",
        "instructions": description or ("Is `%s` true?" % name),
    }
    return _Field(name=name, kind="noul", question=question)


def _score_field(path: str, name: str, prop: Dict[str, Any], description: Optional[str]) -> _Field:
    lo, hi = prop.get("minimum"), prop.get("maximum")
    # `bool` is a subclass of `int`, so `isinstance(True, int)` is true: a JSON `"minimum": true`
    # would otherwise slip through and silently become a bound of 1 (`"maximum": false` a bound of
    # 0), even though the same message rejects `"0"` and `1.5`. `_enum_field` below refuses a
    # `[true, false]` enum on the same principle -- a boolean is a yes/no, not a number -- so the
    # exclusion belongs here too, routed into the existing "needs integer" error the string and
    # float bounds already raise.
    if (
        isinstance(lo, bool)
        or isinstance(hi, bool)
        or not isinstance(lo, int)
        or not isinstance(hi, int)
    ):
        raise SchemaError(
            "%s: a numeric field needs integer 'minimum' and 'maximum' to become a score" % path
        )
    if hi < lo:
        raise SchemaError("%s: 'maximum' %d is below 'minimum' %d" % (path, hi, lo))
    span = hi - lo + 1
    if span > MAX_SCORE_LEVELS:
        raise SchemaError(
            "%s: %d levels exceeds MAX_SCORE_LEVELS=%d; narrow the range or use an enum"
            % (path, span, MAX_SCORE_LEVELS)
        )
    question = {
        "type": "score",
        "instructions": description or ("Score `%s` from %d to %d" % (name, lo, hi)),
        "criteria": [str(v) for v in range(lo, hi + 1)],
    }
    return _Field(name=name, kind="score", question=question, minimum=lo)


_LOCAL_REFS = (("#/$defs/", "$defs"), ("#/definitions/", "definitions"))


def _inline_ref(
    path: str, prop: Dict[str, Any], root: Dict[str, Any], seen: Tuple[str, ...]
) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Replace a local `$ref` with the definition it names, the property's own keys on top.

    Pydantic renders every `Enum` field as `{"$ref": "#/$defs/<Name>"}` (draft-07 and pydantic v1
    spell it `#/definitions/`), with a sibling `description` when the field has one. The
    definition's own `description` is dropped: pydantic fills it from the enum's docstring (v1 with
    "An enumeration." when there is none), which describes the type rather than asking about this
    field, so the wording stays the field's and an `Enum` asks what the same `Literal` would. Only
    this schema's own definitions are looked up; anything else is rejected rather than fetched, and
    a ref met twice on one path is a cycle.
    """
    ref = prop["$ref"]
    if ref in seen:
        raise SchemaError("%s: $ref %r is recursive; flatten the schema" % (path, ref))
    target = None
    for prefix, key in _LOCAL_REFS:
        defs = root.get(key)
        if isinstance(ref, str) and ref.startswith(prefix) and isinstance(defs, dict):
            target = defs.get(ref[len(prefix) :])
    if not isinstance(target, dict):
        raise SchemaError(
            "%s: $ref %r does not resolve to an entry of this schema's '$defs' or "
            "'definitions'" % (path, ref)
        )
    # a `$ref` the definition carries is an alias, followed by the caller's loop
    merged = {k: v for k, v in target.items() if k != "description"}
    merged.update((k, v) for k, v in prop.items() if k != "$ref")
    return merged, seen + (ref,)


def _field(
    path: str,
    name: str,
    prop: Dict[str, Any],
    root: Optional[Dict[str, Any]] = None,
    seen: Tuple[str, ...] = (),
) -> _Field:
    if not isinstance(prop, dict):
        raise SchemaError("%s: property must be an object, got %s" % (path, type(prop).__name__))
    while "$ref" in prop:
        prop, seen = _inline_ref(path, prop, root or {}, seen)
    description = prop.get("description")
    # Pydantic v2 renders `Optional[X]` as `{"anyOf": [<X>, {"type": "null"}]}` with no
    # top-level type/enum/const, the same nullable shape the list form `type: ["string", "null"]`
    # already handles below. Unwrap the single non-null branch (carrying the outer description)
    # so `Optional[Literal[...]]`, `Optional[int]` and friends map instead of raising. A union of
    # two real types is genuinely ambiguous and still rejected.
    if not ({"const", "enum", "type"} & set(prop)):
        # pydantic v1 wraps a described `$ref` as `{"allOf": [{"$ref": ...}], "description": ...}`;
        # a one-item `allOf` is that schema, so it is unwrapped too, the outer keys on top.
        wrapped = prop.get("allOf")
        if isinstance(wrapped, list) and len(wrapped) == 1 and isinstance(wrapped[0], dict):
            branch = dict(wrapped[0])
            branch.update((k, v) for k, v in prop.items() if k != "allOf")
            return _field(path, name, branch, root, seen)
        union = prop.get("anyOf") or prop.get("oneOf")
        # The `allOf` branch above guards its wrapper's shape before iterating it; an `anyOf`/`oneOf`
        # that is not an array (a JSON number, a boolean -- `bool` subclasses `int` and so is neither
        # `None` nor iterable the way a list is) used to fall straight into the comprehension below
        # and leak `TypeError: 'int' object is not iterable`. A non-array union is not a recognised
        # optional wrapper, so it is left to the terminal "unsupported schema" rejection, which names
        # the path exactly like the non-array `enum` case does.
        if isinstance(union, list):
            branches = [b for b in union if isinstance(b, dict) and b.get("type") != "null"]
            if len(branches) != 1:
                raise SchemaError(
                    "%s: only 'Optional[...]' unions (one non-null branch) are supported, got %d"
                    % (path, len(branches))
                )
            branch = dict(branches[0])
            branch.setdefault("description", description)
            return _field(path, name, branch, root, seen)
    if "const" in prop:
        return _enum_field(path, name, [prop["const"]], description)
    if "enum" in prop:
        return _enum_field(path, name, prop["enum"], description)

    jtype = prop.get("type")
    if isinstance(jtype, list):  # nullable: ["string", "null"]
        non_null_types = [t for t in jtype if t != "null"]
        if len(non_null_types) > 1:
            raise SchemaError(
                "%s: 'type' has multiple non-null types; unions are not supported" % path
            )
        jtype = non_null_types[0] if non_null_types else None
    if jtype == "boolean":
        return _noul_field(path, name, description)
    if jtype == "string":
        raise SchemaError(
            "%s: a free string cannot be a fixed option set; use 'enum' or a boolean" % path
        )
    if jtype in ("integer", "number"):
        return _score_field(path, name, prop, description)
    if jtype == "array":
        raise SchemaError("%s: arrays are not supported; ask one field per element" % path)
    if jtype == "object":
        raise SchemaError("%s: nested objects are not supported; flatten the schema" % path)
    raise SchemaError("%s: unsupported schema %r" % (path, prop))


def plan_from_json_schema(schema: Dict[str, Any]) -> List[_Field]:
    """Validate a JSON schema and return one planned field per property."""
    if not isinstance(schema, dict):
        raise SchemaError("expected a JSON schema object, got %s" % type(schema).__name__)
    if schema.get("type") not in (None, "object") or "properties" not in schema:
        raise SchemaError("the top level must be an object with 'properties'")
    properties = schema["properties"]
    if not isinstance(properties, dict) or not properties:
        raise SchemaError("'properties' must be a non-empty object")
    if len(properties) > MAX_PROPERTIES:
        raise SchemaError(
            "%d properties exceeds MAX_PROPERTIES=%d" % (len(properties), MAX_PROPERTIES)
        )
    return [_field("properties.%s" % name, name, prop, schema) for name, prop in properties.items()]


def questions_from_json_schema(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Turn a JSON schema into Laya questions (a documented subset; see the module docstring)."""
    return {f.name: f.question for f in plan_from_json_schema(schema)}


def questions_from_pydantic(model: Any) -> Dict[str, Dict[str, Any]]:
    """Turn a pydantic model into Laya questions. Requires pydantic (the `structured` extra)."""
    _require_pydantic()
    return questions_from_json_schema(_schema_of(model))


def _project(answers: Dict[str, Any], fields: Sequence[_Field]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for f in fields:
        answer = answers.get(f.name)
        # A field whose answer is not an answer object carries nothing to project -- exactly the
        # absent field the `None` branch already omits. `apply_confidence_gate` makes the same call
        # and skips a non-dict answer (confidence.py:298); reading `.get` off one instead raised a
        # raw AttributeError here and crashed the whole projection.
        if not isinstance(answer, dict):
            continue
        if answer.get("low_confidence"):
            values[f.name] = None
            continue
        if f.kind == "noul":
            values[f.name] = bool(float(answer.get("noul", 0.0)) >= 0.5)
        elif f.kind == "score":
            probs = answer.get("probabilities") or {}
            if probs:
                idx = max(
                    range(len(probs)), key=lambda i: float(probs.get(str(i), probs.get(i, 0.0)))
                )
            else:
                # `score` is always the probability-weighted 0-based level index (see
                # `DecisionModel._decode_answers`), the same space `idx` is in above -- not an
                # absolute field value -- so it is rounded on its own, with no `minimum`
                # subtracted first. Subtracting it here and adding it back below used to cancel
                # out, silently dropping `minimum` from every field whose schema does not start
                # at 0.
                idx = int(round(float(answer.get("score", 0.0))))
            values[f.name] = int(f.minimum or 0) + idx
        else:  # choice
            label = str(answer.get("choice"))
            values[f.name] = next((value for lbl, value in f.options if lbl == label), label)
    return values


def answers_to_json(answers: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Project Laya answers onto the schema values (choice value, integer level, boolean)."""
    return _project(answers, plan_from_json_schema(schema))


def answer_to_pydantic(model: Any, answers: Dict[str, Any]) -> Any:
    """Project Laya answers into a pydantic model instance. Requires pydantic."""
    _require_pydantic()
    return model(**answers_to_json(answers, _schema_of(model)))


def _details(
    values: Dict[str, Any], answers: Dict[str, Any], result: Dict[str, Any]
) -> DecisionResult:
    confidence: Dict[str, float] = {}
    answer_confidence: Dict[str, Optional[float]] = {}
    probabilities: Dict[str, Dict[str, Any]] = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict):
            # Same rule as `_project` and `apply_confidence_gate`: a non-dict answer is no answer,
            # so it reports no confidence rather than raising AttributeError on `.get`.
            continue
        confidence[name] = float(answer.get("confidence", 0.0))
        # Read through the module that owns the definition, so this and the `min_confidence` gate
        # cannot disagree about which quantity is being reported.
        answer_confidence[name] = answer_confidence_value(answer)
        if answer.get("type") == "noul":
            p = float(answer.get("noul", 0.0))
            probabilities[name] = {"false": round(1.0 - p, 4), "true": round(p, 4)}
        else:
            probabilities[name] = dict(answer.get("probabilities") or {})
    return DecisionResult(
        values=values,
        confidence=confidence,
        probabilities=probabilities,
        answers=dict(answers),
        usage=result.get("usage"),
        routing=result.get("routing"),
        answer_confidence=answer_confidence,
    )


def decide(
    runner,
    state: Any,
    schema: Any = None,
    *,
    questions: Optional[Dict[str, Any]] = None,
    return_details: bool = False,
    min_confidence: Optional[float] = None,
    **predict_kwargs,
) -> Any:
    """Answer `state` against a schema (or explicit questions) and return the decided values.

    Pass exactly one of `schema` or `questions`. With `schema`, the values follow the schema
    (choice values, integer levels, booleans). With `questions`, the raw answers are returned.
    `return_details=True` returns a `DecisionResult` instead of the plain values. Extra keyword
    arguments are forwarded to `runner.predict`.
    """
    if (schema is None) == (questions is None):
        raise ValueError("pass exactly one of schema= or questions=")

    mc = check_min_confidence(min_confidence) if min_confidence is not None else None
    if mc is not None:
        predict_kwargs["min_confidence"] = mc

    fields: Optional[List[_Field]] = None
    if schema is not None:
        fields = plan_from_json_schema(_schema_of(schema))
        questions = {f.name: f.question for f in fields}

    try:
        result = runner.predict(state, questions, **predict_kwargs)
    except TypeError as e:
        if mc is not None and "unexpected keyword argument 'min_confidence'" in str(e):
            predict_kwargs.pop("min_confidence", None)
            result = runner.predict(state, questions, **predict_kwargs)
        else:
            raise

    if isinstance(result, dict):
        apply_confidence_gate([result], mc)

    answers = result.get("answers", {}) or {}
    values = _project(answers, fields) if fields is not None else dict(answers)
    if return_details:
        return _details(values, answers, result)
    return values


# The keys `Router.predict_batch` reads off each request (`route_batch` and the token-budget
# overrides) rather than accepting as keywords; laya-serve's BATCH_BODY_ITEM_CONTROLS plus `model`.
_ROUTER_REQUEST_KEYS = ("model", "task", "lang", "lang_guess", "max_len", "head_max_len")


def decide_batch(
    runner,
    states: Sequence[Any],
    schema: Any = None,
    *,
    questions: Optional[Dict[str, Any]] = None,
    return_details: bool = False,
    min_confidence: Optional[float] = None,
    **predict_kwargs,
) -> List[Any]:
    """Answer many states against one schema in one batched call, in input order.

    The throughput form of :meth:`decide`: the schema is planned once and its questions
    are evaluated over every state through ``runner.predict_batch`` (the same
    shared-forward-pass path as :meth:`Agent.predict_batch` /
    :meth:`Router.predict_batch`), then each state's answers are projected exactly as
    ``decide`` does. Pass exactly one of ``schema`` or ``questions``. The extra keywords you leave
    here (``batch_size=``, ``hooks=``, ``sort_by_length=``, ...) go to ``runner.predict_batch``,
    but the routing pins and the token budget (``model=``, ``task=``, ``lang=``, ``lang_guess=``,
    ``max_len=``, ``head_max_len=``) are read off every *request* by a ``Router``, whose
    ``predict_batch`` declares none of them: ``decide_batch`` copies them onto each request
    instead, so ``model=`` pins a registered checkpoint there and raises ``TypeError`` on an
    ``Agent``-like runner, whose ``predict_batch`` declares ``lang=`` but no ``model=``.
    With ``return_details=True`` each item is a
    ``DecisionResult``. ``min_confidence`` works as in ``decide``: a field whose answer falls
    below it comes back as ``None``, with the answer kept in the details.

    Both batch calling conventions are handled: an ``Agent``-like runner receives
    ``(states, questions)``, while a ``Router``-like runner (one exposing
    ``route_batch``) receives one ``{"state": ..., "questions": ...}`` request per
    state, so states may route to different checkpoints.

    ``Agent``, ``ONNXAgent`` and ``Router`` all batch. A runner with no ``predict_batch``
    raises ``TypeError`` here rather than silently degrading to N sequential ``decide``
    calls -- loop ``decide`` yourself when the runner cannot batch.
    """
    if (schema is None) == (questions is None):
        raise ValueError("pass exactly one of schema= or questions=")
    if isinstance(states, (str, bytes)) or not isinstance(states, SequenceABC):
        raise TypeError("states must be a sequence of states, not %s" % type(states).__name__)

    mc = check_min_confidence(min_confidence) if min_confidence is not None else None

    fields: Optional[List[_Field]] = None
    if schema is not None:
        fields = plan_from_json_schema(_schema_of(schema))
        questions = {f.name: f.question for f in fields}

    predict_batch = getattr(runner, "predict_batch", None)
    if predict_batch is None:
        raise TypeError(
            "%s has no predict_batch; loop decide() over the states instead" % type(runner).__name__
        )

    if hasattr(runner, "route_batch"):
        # Router convention: one request dict per state, each carrying the shared
        # questions, so it routes, groups by checkpoint and restores input order. The routing
        # pins and the token budget are read off each request there, not taken as keywords of
        # predict_batch, so they ride on every request; the rest (batch_size, hooks, ...) is
        # call-level.
        per_request = {
            key: predict_kwargs.pop(key) for key in _ROUTER_REQUEST_KEYS if key in predict_kwargs
        }
        results = predict_batch(
            [dict({"state": s, "questions": questions}, **per_request) for s in states],
            **predict_kwargs,
        )
    else:
        # Agent convention: a list of states evaluated against one question set.
        results = predict_batch(list(states), questions, **predict_kwargs)

    # Applied here rather than passed down, so a runner whose predict_batch predates the
    # keyword still gets the same projection and the same reported state.
    apply_confidence_gate([r for r in results if isinstance(r, dict)], mc)

    def _one(r: Dict[str, Any]) -> Any:
        answers = r.get("answers", {}) or {}
        values = _project(answers, fields) if fields is not None else dict(answers)
        if return_details:
            return _details(values, answers, r)
        return values

    return [_one(r) for r in results]


__all__ = [
    "DecisionResult",
    "SchemaError",
    "answers_to_json",
    "answer_to_pydantic",
    "decide",
    "decide_batch",
    "plan_from_json_schema",
    "questions_from_json_schema",
    "questions_from_pydantic",
]
