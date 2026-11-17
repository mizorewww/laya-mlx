"""Route a request to the Laya checkpoint best suited to it.

Three checkpoints, measured on a shared benchmark (17,416 questions, one T4, identical questions
per model -- see the repository's benchmark notebook):

  english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
  multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
  typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
                                                                fine-tuned on the typed-decisions
                                                                workflows

Why routing is worth it -- accuracy by language family:

                      english   multilingual
  MASSIVE intent  en    0.783       0.657        <- English checkpoint wins
  MASSIVE intent  non-en 0.306      0.451
  XNLI            en    0.860       0.843
  XNLI            non-en 0.521      0.731        <- +21 points for multilingual
  English suites        0.684       0.619

The English checkpoint does not gently degrade off English, it collapses: on 20-option MASSIVE
intent it scores 0.100 on Hindi and 0.103 on Korean, against 0.050 for random guessing -- and it
reports high confidence while doing so (ECE 0.855 on Hindi). Script detection is therefore the
primary routing signal.

`typed-decisions` is never selected automatically unless you opt in with
`auto_task_detection=True` or pass `task="typed_decisions"`: it is fine-tuned on four specific
synthetic workflows and should not be a silent default.
"""

import contextvars
import gc
import inspect
import json
import os
import re
import threading
import time
import warnings
from collections.abc import Sequence as SequenceABC
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .confidence import apply_confidence_gate, check_min_confidence
from .hooks import (
    _SKIP_DEFAULTS,
    AsyncHook,
    BaseHook,
    HookRegistry,
    PredictContext,
    aggregate_usage,
    compose_hooks,
    dispatch,
    normalise_hooks,
    validate_timeout,
)
from .lang import analyse

# The hub repo bundles all three checkpoints; only the requested subfolder is downloaded.
BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}

#: The lane for what the encoders cannot read: images, speech and video, states past the encoders'
#: 8,192 tokens, and questions with fitted heads. Upstream serves it with vegaml's universal reader
#: through `laya.Agent(vegaml=True)`. This distribution has no vegaml engine, so the lane serves
#: requests only when the Router is given an already built official agent (`Router(vega=agent)`).
#: Text the encoders can read keeps going to the encoders exactly as before.
VEGA_MODEL = "vega"
VEGA_REPO = "nandakishorm/vega-U"
_NO_VEGA_ENGINE = (
    "%s needs vegaml's engine, which ships with official laya and not with this MLX "
    "distribution: pip install -U laya, then pass a built laya.Agent(vegaml=...) as Router(vega=agent)"
)

# The same checkpoints also live in their own repos, for anyone who prefers them.
STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}


def _estimate_tokens(state: Any) -> int:
    """A cheap token estimate for routing: characters over four, over the flattened state.
    Only the decision "can an encoder read this at all" rests on it, so a rough count is enough."""
    if state is None:
        return 0
    try:
        from .lang import state_text

        text = state_text(state, max_chars=1 << 30)
    except Exception:
        text = str(state)
    return len(text) // 4


def _repo_str(spec):
    """Human-readable id for a model spec: 'repo' or 'repo/subfolder'."""
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


def _split(spec):
    """Normalise a model spec to (repo_or_path, subfolder)."""
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None


# Aliases people are likely to type.
_ALIASES = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {
        "discrepancy_severity",
        "disposition",
        "duplicate",
        "matches_order",
        "urgency",
    },
    "security_incidents": {
        "credential_compromise",
        "disposition",
        "severity",
        "true_positive",
        "urgency",
    },
}

# What a caller may name a checkpoint of their own, once `canonical_name` has trimmed and
# lowercased it (so `Papers` is registered as `papers`): letters, digits, `.`, `_` and `-`, starting
# with a letter or digit. No `/`, because a Hub id or a path is a source, not a name.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# An adjacent transposition counts as one edit, so `enlgish` is one away from
# `english`. Set this false for plain Levenshtein, where that pair costs two.
_COUNT_ADJACENT_SWAP = True


def _edit_distance(left: str, right: str) -> int:
    """Edits that turn `left` into `right`: insert, delete, substitute, adjacent swap.

    The swap is the only step that is not Levenshtein. It is gated by
    `_COUNT_ADJACENT_SWAP`, so dropping that flag is the whole change.
    """
    if left == right:
        return 0
    n, m = len(left), len(right)
    # `prev2` is the row two steps back, which is what an adjacent swap reads.
    prev2: Optional[List[int]] = None
    prev: List[int] = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i]
        lc = left[i - 1]
        for j in range(1, m + 1):
            rc = right[j - 1]
            cost = 0 if lc == rc else 1
            best = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if (
                _COUNT_ADJACENT_SWAP
                and prev2 is not None
                and j > 1
                and lc == right[j - 2]
                and left[i - 2] == rc
            ):
                best = min(best, prev2[j - 2] + 1)
            cur.append(best)
        prev2 = prev
        prev = cur
    return prev[m]


def _near_builtin(key: str) -> Optional[str]:
    """Built-in name within one edit of `key`, or None.

    Compared with the built-in names only, not their aliases. `en` and `ml` are
    short enough that ordinary names sit one edit away from them.
    """
    for name in DEFAULT_MODELS:
        if abs(len(key) - len(name)) > 1:
            continue
        if _edit_distance(key, name) == 1:
            return name
    return None


def canonical_name(name: Any) -> str:
    """Trim, lowercase and resolve an alias, without asking whether the result names a checkpoint.

    The spelling half of :func:`normalise_name`. `Router.resolve` applies it and then checks the
    Router's own registry, which holds the built-in checkpoints plus any the caller registered;
    `normalise_name` applies it and checks the built-in table alone.
    """
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a dict so it serialises straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    """Canonical name of a built-in checkpoint, or ValueError.

    This is the registry of the checkpoints the package ships. A Router may know more -- the
    checkpoints registered on it with `Router(models=...)` or `Router.register` -- and resolves
    names through `Router.resolve`, which accepts those too. Code with no Router at hand (the CLI's
    `--model`, `laya.load`) keeps using this one.
    """
    key = canonical_name(name)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES))
        )
    return key


def resolve_model_spec(name: str) -> Optional[Tuple[str, Optional[str]]]:
    """Registry spec for a checkpoint name or alias, or None when it is not one.

    The non-raising sibling of :func:`normalise_name`, for callers that also accept
    things the registry knows nothing about -- a Hub repo id, a local directory, an
    ONNX export. Those pass through untouched; a name or alias the registry does know
    resolves to its ``(repo, subfolder)`` pair, so ``load("typed-decisions")`` and
    ``Router(model="typed-decisions")`` name the same checkpoint from one table.
    """
    try:
        key = normalise_name(name)
    except ValueError:
        return None
    return tuple(_split(DEFAULT_MODELS[key]))


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """Name of the typed-decisions workflow whose question ids these are, else None.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


def _question_schema(questions: Dict[str, Any]) -> str:
    """Order- and type-sensitive signature for sharing forward passes.

    JSON object keys lose their Python types: 1 and "1" both become "1". Those types
    survive in public question IDs and choice answers, so encode dictionaries as ordered
    pairs with typed keys before serializing. Tag containers too, so a caller's list cannot
    collide with an encoded dictionary. Keep insertion order at every nesting level because
    option order is positional in render_options.
    """

    def typed(value):
        if isinstance(value, dict):
            return ("dict", [(typed(key), typed(item)) for key, item in value.items()])
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, [typed(item) for item in value])
        return (type(value).__name__, value)

    return json.dumps(typed(questions), ensure_ascii=False, default=str)


# Subtags that mean "the English checkpoint can read this". Routing needs one bit -- is this
# English Latin text, or something the English checkpoint cannot read -- not a language id, so
# every other code that names a language resolves to the multilingual checkpoint.
_ENGLISH_SUBTAGS = ("en", "eng", "english")

# Codes that are valid `$LANG` values but name no language, so they answer nothing about the
# state. `C`, `POSIX` and `C.UTF-8` are what minimal images ship -- `C.UTF-8` is the default
# `LANG` in the official Python image, which is where `laya-serve` runs -- and the ISO 639-2
# special codes say the same thing in the standard's own vocabulary: `und` undetermined,
# `zxx` no linguistic content, `mul` multiple languages. They abstain, which is what the blank
# case below already does, rather than forcing the multilingual checkpoint on English text.
_LANGUAGE_AGNOSTIC_CODES = ("c", "posix", "und", "zxx", "mul")


def _english_from_code(value: Any) -> Optional[bool]:
    """True/False for a language code, or None when the code identifies nothing.

    Accepts the forms a caller is likely to have to hand: `"en"`, `"EN"`, `"en-US"`, the
    POSIX `"en_US"` (which `$LANG` holds), and `"en_US.UTF-8"`. `None` here means "no usable
    hint", which is what lets a language-identification model abstain -- and it is also what a
    code that names no language returns, so `LANG=C` falls through to detection instead of
    pinning every request to one checkpoint.
    """
    if value is None:
        return None
    code = str(value).strip().lower()
    if not code:
        return None
    code = code.split(".", 1)[0]  # en_US.UTF-8 -> en_US
    primary = code.replace("_", "-").split("-", 1)[0]  # en_US -> en
    if not primary or primary in _LANGUAGE_AGNOSTIC_CODES:
        return None
    return primary in _ENGLISH_SUBTAGS


def _takes_lang(fn) -> bool:
    """Whether `fn` can be given `lang=` as a keyword, read from its signature.

    The same predicate `laya.evals` uses for the optional arguments it forwards (`_takes_lang`'s
    `sort_by_length` and `min_confidence` siblings), with two differences worth naming:

    * The kind is checked, not just the name: a positional-only `lang` (`..., lang, /`) or a
      `*lang` cannot be passed as a keyword, so neither counts. A `**kwargs` forwarder does count,
      because whatever it forwards to is the real entry point, and a decorator built with
      `functools.wraps` is followed through to that entry point (`inspect.signature`'s default).
    * A callable whose signature `inspect` cannot produce is given the argument, where
      `laya.evals` withholds it. Opposite defaults for opposite stakes: `sort_by_length` is an
      optimisation whose absence changes no answer, while dropping a `lang` silently changes which
      language the document is read as. Passing it is also exactly what the Router did before, so
      it is the no-change branch. `Exception` rather than `(TypeError, ValueError)` for the same
      reason: `inspect.signature` reads `__signature__`, which is arbitrary code that can raise
      anything, and a deployer's agent must not become uncallable because its signature is
      awkward to introspect. (A `__call__` object is not this case: `inspect` reads its `__call__`
      and it is handled like any other signature.)
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except Exception:  # see above: any failure to introspect means "unknown"
        return True
    return any(
        (p.name == "lang" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
        or p.kind is p.VAR_KEYWORD
        for p in params
    )


class _ScanLong:
    """The scan `Router.predict_long` asks `predict` to run instead of `system_one`.

    This was a start hook, appended after every other one, and being a hook is what made the scan
    inherit the caller's hook machinery: `hooks_timeout` bounded laya's own inference and aborted
    it, so any timeout shorter than the scan failed the call (measured: a ~4 s scan failed at both
    `hooks_timeout=0.5` and `2.0`), `hooks_concurrent=False` held the hooks lock across the whole
    forward pass instead of across each hook, `_SKIP_DEFAULTS` had not been entered yet so every
    process-wide default hook fired a second time, and an error from the scan was swallowed under
    `hooks_raise=False`, answering a single window where a scan was asked for. It is a plain
    command object now, and `predict` runs it after the start chain returns.

    Everything it reads, it reads from `ctx` at the moment it runs, which is what keeps this
    equivalent to the hook it replaces: `ctx.agent` rather than the agent `predict` resolved, and
    `ctx.decision` for the detected language, both of which a start hook can still have rewritten.
    `predict` runs it last and only when nothing else has answered, so a hook that called
    `ctx.skip(...)` or rewrote the state/questions wins exactly as before.
    """

    def __init__(self, window, stride, aggregate, batch_size, lang):
        self.window = window
        self.stride = stride
        self.aggregate = aggregate
        self.batch_size = batch_size
        self.lang = lang

    def run(self, ctx):
        """Scan `ctx.states[0]` on `ctx.agent`, or refuse plainly when that agent cannot."""
        agent = ctx.agent
        scan = getattr(agent, "predict_long", None)
        if scan is None:
            raise TypeError(
                "%s has no predict_long, so a state longer than its window cannot be scanned; "
                "the PyTorch Agent and ONNXAgent implement it, and an agent attached by hand needs "
                "it too" % type(agent).__name__
            )
        lang = self.lang
        if lang is None:
            lang = ((ctx.decision or {}).get("detection") or {}).get("language")
        kwargs = {
            "window": self.window,
            "stride": self.stride,
            "aggregate": self.aggregate,
            "batch_size": self.batch_size,
        }
        # `lang` is passed only to an entry point whose signature takes it -- the check
        # `laya.evals` already makes for the arguments it forwards. Deciding this from the
        # signature rather than from a `TypeError` matters here: catching the error cannot tell
        # `predict_long` refusing `lang` from something deeper inside it refusing `lang`, so a
        # retry would re-run the whole scan, drop the caller's language, and answer as if nothing
        # had happened. Nothing is caught, so every error the scan raises is the caller's to see.
        if _takes_lang(scan):
            kwargs["lang"] = lang
        elif lang is not None:
            # What reaches the log line is bounded, and so is the work of building it, because
            # `lang` is caller data: `serve.py` type-checks it but does not cap its length, and the
            # library entry point does not even do that. So an exact `str` is sliced to 32
            # characters BEFORE it is rendered -- slicing the value rather than the rendered text,
            # because a `str` subclass can report `len() == 2` with a megabyte of `__repr__`, and
            # because slicing a subclass would run the subclass's `__getitem__` -- and anything
            # else is a constant. Not even its type name: `__name__` is a writable slot on any heap
            # type and a metaclass can make it a property that returns a megabyte, returns an
            # object whose `__str__` then runs, or simply raises. A constant has no such reach.
            # `repr` on the `str` branch is not decoration -- it is what escapes NUL and
            # ANSI escapes out of the value, so a terminal-injection payload in `lang` reaches a
            # log inert. Simplifying it to `%s` would be a regression, not a tidy-up.
            if type(lang) is str:
                shown = repr(lang[:32]) + ("..." if len(lang) > 32 else "")
            else:
                shown = "<not a string>"
            warnings.warn(
                "laya: Router.predict_long: %s.predict_long has no `lang` parameter, so the %s "
                "language %s is not reaching it and the scan runs without it"
                % (
                    type(agent).__name__,
                    "requested" if self.lang is not None else "detected",
                    shown,
                ),
                # 4, not 2: the frames from here are `run` -> `predict` -> `predict_long` -> the
                # caller, so 2 points inside this file. That is not only the wrong attribution --
                # the default "once per location" filter keys its registry on the frame this picks,
                # so at 2 a process that drops the same language at twenty call sites would warn
                # once and stay silent for the rest of its life.
                RuntimeWarning,
                stacklevel=4,
            )
        return scan(ctx.states[0], ctx.questions, **kwargs)


# How `predict_long` reaches `predict` without widening `predict`'s signature: a parameter there is
# public API -- `tests/test_serve.py` asserts `predict`'s parameter list equals serve's
# `BODY_CONTROLS | BODY_REFUSALS`, so a new one has to be forwarded from or refused on the HTTP body
# -- and this is internal plumbing with no wire meaning. It travels the way `_SKIP_DEFAULTS` does.
# `predict` clears it on read, so a hook that calls back into `predict` cannot inherit a scan that
# was not meant for it.
_SCAN: "contextvars.ContextVar[Optional[_ScanLong]]" = contextvars.ContextVar(
    "laya_router_scan", default=None
)


# Checkpoint options a Router will not accept through `agent_kwargs`: the ones it sets for itself
# on every `Agent(...)` it builds, plus the hook family, which has its own home in `Router(hooks=)`
# and would otherwise fire from two registries at once.
_ROUTER_OWNED_AGENT_ARGS = frozenset(
    {
        "model_id_or_path",
        "device",
        "token",
        "subfolder",
        "revision",
        "hooks",
        "on_predict_start",
        "on_predict_end",
        "hooks_raise",
        "hooks_concurrent",
        "hooks_timeout",
    }
)


def _has_on_route(hooks: Sequence[Any]) -> bool:
    """True if at least one hook provides a non-default on_route implementation."""
    for hook in hooks:
        while isinstance(hook, AsyncHook):
            hook = hook.hook
        method = getattr(hook, "on_route", None)
        if method is not None and getattr(method, "__func__", method) is not BaseHook.on_route:
            return True
    return False


def check_agent_kwargs(agent_kwargs: Dict[str, Any]) -> None:
    """Reject `agent_kwargs` names a Router cannot honour, before any checkpoint is loaded.

    The accepted names are read out of `Agent.__init__`'s signature rather than written down here:
    the whole point of `agent_kwargs` is that this file stops keeping its own list of checkpoint
    options, so a hard-coded copy of them would go stale in exactly the way this argument removes.
    The import is only reached when a caller passes the argument, which is also the moment torch is
    about to be loaded anyway.
    """
    from .agent import Agent

    accepted = set(inspect.signature(Agent.__init__).parameters) - {"self"}
    reserved = sorted(set(agent_kwargs) & _ROUTER_OWNED_AGENT_ARGS)
    if reserved:
        raise ValueError(
            "Router sets %s itself; pass them to Router(...) instead of in agent_kwargs"
            % (", ".join(repr(n) for n in reserved),)
        )
    unknown = sorted(set(agent_kwargs) - accepted)
    if unknown:
        raise ValueError(
            "Agent accepts none of %s; it accepts %s"
            % (", ".join(repr(n) for n in unknown), sorted(accepted - _ROUTER_OWNED_AGENT_ARGS))
        )


def _digests_from_env(
    models: Dict[str, Any], resolve=normalise_name
) -> Dict[str, Optional[Dict[str, str]]]:
    """Turn `LAYA_SHA256_DIGESTS` into per-checkpoint digest maps when it names models.

    The variable has two shapes, and the value types say which. A flat
    `{artifact: digest}` map is `laya.revisions.verify_digests`'s own reading -- the same
    files checked on every checkpoint the process loads -- and is returned as `{}` here so
    that path stays untouched. A nested `{model: {artifact: digest}}` map pins each
    checkpoint with its own files, which is what a server holding several resident needs:
    the bundled repository ships a separate `model.safetensors` per checkpoint, so one flat
    map can only ever match one of them and refuses the rest at startup.

    A checkpoint the nested map does not name is returned as `{}`, meaning "not covered by this
    variable, and do not let `verify_digests` read this nested map as an artifact map either" -- it
    would fail with `cannot verify 'english': no such file` on a checkpoint the variable never
    named. It needs no distinguishing from a `{}` somebody wrote: both name no files, and
    `_merge_expected_digests` merges them identically.
    Keys are normalised exactly as `Router(sha256_digests=...)` normalises them, so `en`
    names `english` and a name core does not know raises here rather than quietly leaving
    that checkpoint unverified. Unset, empty or unparseable input names nothing:
    `laya.revisions` reports a malformed value in its own words.
    """
    raw = os.environ.get("LAYA_SHA256_DIGESTS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        # `RecursionError` is a `RuntimeError`, so `except ValueError` did not cover the one shape
        # of unparseable this variable can have that is not a decode error: `json` answers a
        # nesting depth it will not walk with a stack overflow. It names nothing, so it names no
        # checkpoint here either -- `laya.revisions` reports it, in the words this docstring
        # already promises are the only ones this layer uses.
        return {}
    if not isinstance(data, dict) or not data:
        return {}
    values = list(data.values())
    if all(isinstance(v, str) for v in values):
        return {}  # flat: laya.revisions already applies it
    if not all(isinstance(v, dict) for v in values):
        raise ValueError(
            "LAYA_SHA256_DIGESTS must be either {artifact: digest} for every "
            "checkpoint or {model: {artifact: digest}} per checkpoint; %s mixes "
            "the two or holds a value that is neither" % sorted(data)
        )
    per_model = {resolve(k): dict(v) for k, v in data.items()}
    for name in models:
        per_model.setdefault(resolve(name), {})
    return per_model


def _revision_pin(value: Optional[str]) -> Optional[str]:
    """The revision `value` asks for, or None when it asks for nothing.

    Unset and blank are the same answer, because both are what a configuration line that did not
    get filled in leaves behind, and neither is a commit anything can be pinned to.
    """
    return value if value is not None and str(value).strip() else None


def _digest_entry(
    digests: Dict[str, Optional[Dict[str, str]]],
    key: str,
) -> Optional[Dict[str, str]]:
    """One checkpoint's own digest entry, as `Router.sha256_digests` holds it right now.

    None when the checkpoint has no entry at all, which is what leaves `verify_digests` its own
    environment fallback; `{}` for an entry that names no files, whether that is the placeholder a
    nested `LAYA_SHA256_DIGESTS` left or a `None` a caller wrote.

    Read here rather than decided once at construction, because `sha256_digests` is a public mutable
    attribute: a per-checkpoint pin assigned to it -- or added to an existing entry in place -- after
    `Router(...)` has to count, and deciding from a key set frozen at construction dropped exactly
    that.
    """
    if key not in digests:
        return None
    entry = digests[key]
    return {} if entry is None else entry


def _merge_expected_digests(
    shared: Optional[Dict[str, str]],
    per_model: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """The `expected_sha256` one checkpoint is built with, out of the two channels that pin it.

    `shared` is `agent_kwargs["expected_sha256"]`, which the class docstring advertises and which
    reaches every checkpoint the Router builds. `per_model` is this checkpoint's own entry: None when
    it has none at all, and `{}` for the placeholder a nested `LAYA_SHA256_DIGESTS` leaves for a
    checkpoint it does not name.

    None -- both channels silent -- means "pass no `expected_sha256`", which leaves `verify_digests`
    its own `LAYA_SHA256_DIGESTS` fallback and keeps an unconfigured load byte for byte what it was.
    An empty `{}` is **not** the same thing and must still reach `Agent`: it masks that fallback,
    which for a *nested* variable would otherwise be read as an artifact map and raise
    `cannot verify 'english': no such file` on a checkpoint the variable never named.

    A value that is not a mapping is handed on untouched, so `verify_digests` keeps ownership of the
    "expected_sha256 must be a mapping" message rather than this layer growing a second copy.

    The two channels are merged file by file, because each names files and a digest is a claim about
    one file -- dropping either half would verify less than the caller asked for. **For a file both
    name, the per-checkpoint entry wins.** They are not equally specific: `shared` reaches every
    checkpoint, and `model.safetensors` is the one name every checkpoint uses for a *different* file,
    so a shared entry for it cannot be a correct claim about all of them at once. Refusing that
    overlap as a contradiction broke the ordinary shape it appears in -- a shared pin plus a
    per-checkpoint override, which loaded correctly before this function existed.

    An empty placeholder therefore needs no special case: merging it changes nothing.
    """
    if per_model is not None and not isinstance(per_model, dict):
        return per_model
    if shared is not None and not isinstance(shared, dict):
        return shared
    if shared is None and per_model is None:
        return None
    merged = dict(shared or {})
    merged.update(per_model or {})
    return merged


class _InFlightBuild:
    """Private synchronization descriptor for an in-flight checkpoint build.

    `done` is signalled once the build finishes (or fails), releasing callers waiting
    for this specific checkpoint. `error` records any exception raised during the build
    so concurrent waiting callers unblock and receive the failure instead of deadlocking.
    """

    __slots__ = ("done", "error")

    def __init__(self):
        self.done = threading.Event()
        self.error: Optional[BaseException] = None


class Router(HookRegistry):
    """Lazily loads Laya checkpoints and sends each request to the right one.

        from laya_mlx import Router

        r = Router()
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)   # -> multilingual
        r.predict({"message": "I was charged twice"}, questions)                 # -> english
        r.predict(state, questions, model="typed-decisions")                     # explicit

    Models are downloaded and built on first use. `max_loaded` caps how many stay resident
    (least-recently-used is evicted), because all three together are ~1.16B parameters.

    The default is 2, because automatic routing only ever chooses between `english` and
    `multilingual`: a cap of one rebuilds the checkpoint it just evicted on every script switch,
    which is seconds per request on exactly the traffic the Router exists for. Traffic that only
    ever sees one language never builds the second checkpoint, so the default costs it nothing.
    Lower it to 1 for a memory-constrained host, and raise it to 3 (or preload) when
    `auto_task_detection`, an explicit `model=` or an explicit `task=` can reach
    `typed-decisions` as well.

    For a server or a demo, preload instead: a cold load costs seconds, while detection costs
    microseconds, so even the default still pays a load the first time a language appears.

        r = Router(preload=True)                    # all three resident, routing is free
        r = Router(preload=True, device="cuda")
        r.preload(["english", "multilingual"])      # or just the two you serve

    Hub revisions are opt-in. `revision` applies one commit to every model;
    `revisions={"english": "...", "multilingual": "..."}` overrides that per model,
    which is useful when standalone repositories were reviewed at different commits.
    Without either, huggingface_hub's normal default and existing offline cache are used.

    A `revisions` entry that is `None` or blank is "no override for this model", so the model
    inherits `revision` -- the shape `{"english": os.environ.get("EN_SHA")}` writes when the
    variable is unset, which must not cost the caller the pin it did ask for. Nothing in
    `revisions` can unpin one model while `revision` pins the rest; leave `revision` unset and
    name the models you want pinned instead.

    A blank `revision` is read the same way, which is a deliberate change in what the Router
    passes on: `Router(revision="   ")` used to reach `resolve_revision`, where a truthy but blank
    string suppressed the `LAYA_REVISION` fallback and left huggingface_hub's default, and is now
    dropped before it gets there, so a Router configured with whitespace behaves like one
    configured with nothing and `$LAYA_REVISION` applies. That is what "this configuration line was
    never filled in" has to mean if `revision` and a `revisions` entry are to be read the same way.

    It is one of two places a weaker source ends up ahead of an explicit argument. The other is a
    per-checkpoint digest entry from `LAYA_SHA256_DIGESTS`, which wins over an `expected_sha256`
    passed through `agent_kwargs`; see the class docstring.

    Anything else `laya.Agent` accepts is reachable through `agent_kwargs`, which is merged into
    every checkpoint the Router builds:

        Router(agent_kwargs={"lang_temperatures": {"de": {"temperature": [1.0, 1.4, 2.0]}}})
        Router(agent_kwargs={"expected_sha256": {"model.safetensors": "a3f1..."}})
        Router(agent_kwargs={"fast": True})

    `expected_sha256` there pins the same files on every checkpoint, which is what a single
    resident checkpoint or a shared `tokenizer.json` wants. It is never discarded wholesale by a
    checkpoint's own digests: where one also has an entry, from `sha256_digests` or from a
    **per-checkpoint** `LAYA_SHA256_DIGESTS`, the two maps are merged **file by file**, so a file
    only one of them names is still verified.

    Two exceptions, both deliberate and both tested, because "merged file by file" is not the whole
    story and the difference is a supply-chain control:

    * A **flat** `LAYA_SHA256_DIGESTS` -- `{artifact: digest}` rather than `{model: {...}}` -- is
      not a layer here at all. `verify_digests` applies it itself, but only when nothing else pins
      (`if expected is None`), so ANY `expected_sha256` reaching `Agent`, from here or from a
      checkpoint entry, means the flat variable is not consulted for that load. Verified on real
      files: a flat variable pinning `model.safetensors` plus an `agent_kwargs` map pinning
      `tokenizer.json` loads a tampered `model.safetensors`. Use the per-checkpoint shape, or name
      every file you care about in one map, if you need both. This is unchanged from `main`.
    * An explicit `{}` or `None` entry MASKS what would otherwise apply -- that is what "load this
      one unverified" has to mean, and `test_an_explicit_none_entry_masks_a_flat_environment_map`
      pins it.

    And the precedence is per-checkpoint over shared regardless of where each came from, so a
    per-checkpoint entry synthesised from `LAYA_SHA256_DIGESTS` wins over an `expected_sha256`
    passed here in code. An environment variable beating an explicit argument is worth stating
    plainly on a control like this; `test_an_environment_pin_overrides_the_shared_one_per_checkpoint`
    is where that is pinned.

    For a file both name, the per-checkpoint entry wins. The two are not equally specific: the
    `agent_kwargs` map reaches every checkpoint the Router builds, and `model.safetensors` is the one
    name every checkpoint uses for a *different* file, so a shared entry for it cannot be a correct
    claim about all of them at once. **Nothing raises over that overlap** -- refusing it would reject
    a shared pin plus a per-checkpoint override, which is the ordinary shape and which loaded
    correctly before any of this existed. If you need to know which digest a checkpoint was verified
    against, read it back: the map handed to each `Agent` is the merge described above.

    Both `agent_kwargs` and `sha256_digests` are public and mutable, and a checkpoint's entry is read
    on the load rather than at construction, so a pin assigned afterwards -- or added to an
    existing entry in place -- counts.

    The names the Router sets for itself -- `model_id_or_path`, `device`, `token`, `subfolder`,
    `revision` and the hook arguments -- are refused here rather than silently shadowed, and the
    remaining names are checked against `Agent.__init__` at construction, so a misspelled option
    fails on the `Router(...)` line instead of on the first request.

    Artifact digests are opt-in and always per model: `sha256_digests={"english": {...}}`
    passes that `{path relative to the checkpoint dir: hexdigest}` map to the `Agent` that
    loads it, so a tampered or substituted weight file is refused before it is parsed. There
    is no Router-wide equivalent of `revision` because digests, unlike a commit SHA, are not
    shareable: the bundled repository ships a separate `model.safetensors` for each of
    `english`, `multilingual` and `typed-decisions`, so one flat map can only ever match one
    of them. A model listed with `None` or `{}` adds no files of its own, which loads it
    unverified unless `agent_kwargs["expected_sha256"]` pins it; it still masks a flat
    `LAYA_SHA256_DIGESTS`, which is what listing it that way is for.

    The same split is available to a process configured only by environment: when
    `LAYA_SHA256_DIGESTS` holds a model-keyed map (`{"english": {...}, "multilingual": {...}}`)
    this seeds it per checkpoint, so a server that keeps several resident can pin each with its
    own digests instead of refusing to start on the second one. A flat `LAYA_SHA256_DIGESTS`
    keeps its existing meaning, applied by `laya.revisions` to every checkpoint the process
    loads, which is right for a single-checkpoint one. An argument entry wins over the
    environment for the model it names. A nested variable that names some checkpoints and not
    others says nothing about the others: they keep whatever `agent_kwargs["expected_sha256"]`
    pins them with, because pinning one checkpoint from the environment is not a request to stop
    verifying the rest.

    Hooks are opt-in and run at the Router level: `on_route` sees the routing decision,
    `on_load` / `on_evict` see model lifecycle, and `on_predict_start` / `on_predict_end`
    wrap the whole route+infer call. See `laya.hooks`.
    """

    # Opt-in defaults so a hand-built instance (`Router.__new__` in tests) works unset.
    # `hooks`/`_hooks_mutex` come from HookRegistry.
    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None
    agent_kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        revisions: Optional[Dict[str, Optional[str]]] = None,
        max_loaded: int = 2,
        # The fallback for text whose language detection abstains, not for all traffic:
        # detected non-Latin script already routes to `multilingual` whatever this says.
        # `multilingual` since 0.4.0, on the refreshed 51-language sweep: it leads on 50 of
        # the 51, the one exception being English itself (0.820 against 0.710), and by 0.180
        # macro accuracy excluding English. The break-even is about 62% English traffic, so a
        # mostly-English deployment should set `default="english"` back. The asymmetry that
        # settles it for undecided text is the failure mode rather than the mean: off English
        # the English checkpoint collapses while staying confident (Khmer 0.000 accuracy at
        # 0.952 mean confidence), so no `min_confidence` gate downstream can catch it, while
        # the multilingual checkpoint gives up 0.110 on English and stays gateable.
        default: str = "multilingual",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
        lang_guess: Optional[Any] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
        hooks_timeout: Optional[float] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        sha256_digests: Optional[Dict[str, Optional[Dict[str, str]]]] = None,
        # The vega lane. This distribution has no vegaml engine of its own, so "auto" (and False
        # or None) leaves the lane off and the Router behaves exactly as it did before; an
        # already built official `laya.Agent(vegaml=...)` is used as is, and its fitted heads
        # route the questions they were fitted for; a reader name routes but cannot be loaded.
        vega: Any = "auto",
        # A state estimated past this many tokens goes to vega, because no encoder checkpoint can
        # read it whole; 8,192 is `laya-multilingual` at `max_len=8192`, the most any encoder takes.
        vega_over_tokens: int = 8192,
        *,
        local_files_only=False,
    ):
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        # The registry: built-in names first, then whatever the caller adds. A key that is a
        # built-in name re-points that checkpoint (a fine-tune standing in for `english`); any
        # other key registers a checkpoint of the caller's own, served beside the built-ins and
        # named in `model=` like one of them.
        self.models: Dict[str, Any] = dict(
            STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS
        )
        # Free text per registered checkpoint, reported by `registered`; a public mutable attribute
        # like `sha256_digests`, filled by `register`.
        self.descriptions: Dict[str, str] = {}
        for k, v in (models or {}).items():
            self._add(k, v)
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        # Optional Hub revision (commit SHA/branch/tag) applied to every checkpoint load.
        # `revisions` overrides it per normalized model name, for standalone repos whose
        # reviewed commits differ.
        self.revision = revision
        self.revisions: Dict[str, Optional[str]] = {
            self.resolve(k): v for k, v in (revisions or {}).items()
        }
        # Options forwarded to every `Agent` this Router builds, checked now so a bad name fails on
        # this line rather than the first request that happens to load a checkpoint.
        self.agent_kwargs: Dict[str, Any] = dict(agent_kwargs or {})
        if local_files_only:
            self.agent_kwargs["local_files_only"] = True
        if self.agent_kwargs:
            check_agent_kwargs(self.agent_kwargs)
        # Per checkpoint SHA-256 map: seeded from a model-named `LAYA_SHA256_DIGESTS`, then
        # overridden checkpoint by checkpoint by the argument. Keyed and normalised exactly like
        # `revisions`, so a misspelled model name fails here rather than leaving that checkpoint
        # unverified.
        self.sha256_digests: Dict[str, Optional[Dict[str, str]]] = _digests_from_env(
            self.models, self.resolve
        )
        argument = {self.resolve(k): v for k, v in (sha256_digests or {}).items()}
        self.sha256_digests.update(argument)
        self.max_loaded = max(1, int(max_loaded))
        self.default = self.resolve(default)
        self.auto_task_detection = bool(auto_task_detection)
        # An opt-in language hint installed for every request: a code, or a callable taking the
        # state and returning one (or None to abstain). Checked before the built-in detection,
        # never before an explicit `model`, `task` or `lang`. The default path is unchanged, so
        # the heuristic stays dependency-free; this is the seam for a real LID model.
        self.lang_guess = lang_guess
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []  # least-recently-used first
        # Synchronization invariants:
        # - `_lock`: Re-entrant lock guarding Router shared state: `_agents`, `_order`,
        #   the `_loading` registry, and lifecycle bookkeeping. Never held during expensive
        #   checkpoint construction (Agent download/init), nor while waiting on in-flight events.
        # - `_build_lock`: Non-reentrant lock serializing expensive Agent construction globally
        #   (one build at a time across all checkpoints to preserve peak memory bounds).
        #   Acquired outside `_lock` (never inside) to prevent deadlocks, and released before
        #   dispatching lifecycle hooks.
        # - `_loading[name]`: Private per-checkpoint in-flight build registry. Maps normalized
        #   model names to their `_InFlightBuild` descriptor so concurrent loads deduplicate (#95)
        #   and `unload(name)` waits only for builds of the SAME checkpoint without stalling
        #   on unrelated builds holding `_build_lock`.
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._loading: Dict[str, _InFlightBuild] = {}
        self.vega_over_tokens = int(vega_over_tokens)
        self._vega_spec: Any = None
        self._install_vega(vega)
        if preload:
            self.preload()

    def _install_vega(self, vega: Any) -> None:
        """Register the vega lane according to `vega`. The vegaml engine belongs to official
        laya, so this distribution never builds one: a built agent is served as is, a reader name
        routes like upstream but is refused when loaded, and "auto" leaves the lane off."""
        # The lane lives beside `models`, not in it: `models` is the checkpoint registry that
        # `registered`, `preload` and the tests pin, and the lane is not a checkpoint anyone
        # registered. `resolve` knows the name, so `model="vega"` and `load("vega")` still work.
        if vega is None or vega is False:
            return
        if hasattr(vega, "predict") and not isinstance(vega, str):
            self._agents[VEGA_MODEL] = vega
            self._order.append(VEGA_MODEL)
            self.max_loaded = max(self.max_loaded, len(self._agents))
            self._vega_spec = vega
            return
        if isinstance(vega, str) and vega != "auto":
            self._vega_spec = vega  # routes like upstream; loading it is refused in `_build`
            return
        if vega is True:
            raise ImportError(_NO_VEGA_ENGINE % "Router(vega=True)")
        # "auto" is upstream's "on when the engine is available", which it never is here.

    @property
    def vega_enabled(self) -> bool:
        """Whether this Router has a vega lane to route media, long states and fitted heads to."""
        return self._vega_spec is not None

    def _vega_heads(self) -> set:
        """The question ids a resident vega agent has fitted heads for, or nothing."""
        agent = self._agents.get(VEGA_MODEL)
        engine = getattr(agent, "vega", None)
        heads = getattr(engine, "heads", None)
        return set(heads) if isinstance(heads, dict) else set()

    # ------------------------------------------------------------------ registry
    def resolve(self, name: Any) -> str:
        """Canonical key of a checkpoint this Router knows: built-in, aliased, or registered.

        The instance counterpart of :func:`normalise_name`. Everything on the Router that takes a
        checkpoint name goes through here, so a checkpoint registered under `papers` is accepted
        wherever `english` is: `model=`, `task=`, `load`, `unload`, `preload`, `revisions`.
        """
        key = canonical_name(name)
        if key in self.models:
            return key
        if key == VEGA_MODEL and getattr(self, "_vega_spec", None) is not None:
            return key
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (
                name,
                sorted(self.models)
                + ([VEGA_MODEL] if getattr(self, "_vega_spec", None) is not None else []),
                sorted(_ALIASES),
            )
        )

    def register(self, name: str, source: Any, description: Optional[str] = None) -> str:
        """Add a checkpoint under `name`, to be served beside the built-in ones.

        `source` is what `laya.load` accepts: a Hub repo id, a `(repo, subfolder)` pair, or a local
        directory holding `model.safetensors` and `rl_agent_config.json` -- a fine-tune exported by
        the training recipe, for instance. Nothing is downloaded or built here; the checkpoint loads
        on first use like the built-ins, is evicted and unloaded like them, and counts toward
        `max_loaded` like them.

        `description` is free text about the checkpoint for whoever reads `registered`.

        A name within one edit of a built-in is still registered, with a warning that names both
        and leaves the built-in unchanged.

        A nested `LAYA_SHA256_DIGESTS` entry for this name is picked up here; an entry for a name the
        constructor did not know still fails at construction, as before.

        Returns the canonical name. Registering an existing name replaces its source and unloads the
        resident Agent, so the next load builds from the new source.
        """
        with self._lock:
            key = canonical_name(name)
            missing = object()  # a present None (an opted-out digest entry) is not an absent key
            before = (
                self.models.get(key, missing),
                self.sha256_digests.get(key, missing),
                self.descriptions.get(key, missing),
            )
            try:
                key = self._add(name, source)
                if key not in self.sha256_digests:
                    # A nested LAYA_SHA256_DIGESTS names no file of this checkpoint: the `{}` placeholder
                    # the constructor gives every model keeps `verify_digests` off the nested map.
                    from_env = _digests_from_env([key], canonical_name).get(key)
                    if from_env is not None:
                        self.sha256_digests[key] = from_env
                if description is not None:
                    self.descriptions[key] = str(description)
            except BaseException:  # a refused register() leaves the router as it was
                for table, old in (
                    (self.models, before[0]),
                    (self.sha256_digests, before[1]),
                    (self.descriptions, before[2]),
                ):
                    if old is missing:
                        table.pop(key, None)
                    else:
                        table[key] = old
                raise
            replaced = before[0] is not missing and before[0] != self.models[key]
        if replaced:
            freed = self._unload_key(key)  # outside the lock, waiting for a build of the old source
            if freed:
                self._dispatch_lifecycle("on_evict", freed)
        return key

    def unregister(self, name: str) -> None:
        """Remove a registered checkpoint: its source, description, revision and digest entries, and resident Agent.

        A built-in name is refused (re-point it with `register`, it cannot be removed), and so is a
        name the router does not know. A resident Agent is unloaded first, waiting for an in-flight
        build of it, so a request that already holds it finishes; a request that names it afterwards
        gets the usual unknown-model error.
        """
        key = self.resolve(name)
        if key in DEFAULT_MODELS:
            raise ValueError("%r is a built-in checkpoint and cannot be unregistered" % key)
        if key == self.default:
            raise ValueError(
                "%r is this Router's default checkpoint; set another default before "
                "unregistering it" % key
            )
        with self._lock:
            self.models.pop(key, None)  # first, so no new load can start a build of it
        freed = self._unload_key(key)  # waits for a build that had already read its source
        with self._lock:
            for table in (self.descriptions, self.sha256_digests, self.revisions):
                table.pop(key, None)
            self._agents.pop(key, None)
            if key in self._order:
                self._order.remove(key)
        self._dispatch_lifecycle("on_evict", freed)

    @property
    def registered(self) -> Dict[str, Dict[str, Any]]:
        """The checkpoints registered beside the built-ins: name -> source and description.

        Serialisable as it is.
        """
        with self._lock:
            return {
                name: {
                    "source": _repo_str(spec),
                    "description": self.descriptions.get(name),
                }
                # the vega lane is a model name (`model="vega"`, `load("vega")`) but not a checkpoint
                # anyone registered, so it is reported by `vega_enabled`, not here
                for name, spec in self.models.items()
                if name not in DEFAULT_MODELS
            }

    def _add(self, name: Any, source: Any) -> str:
        """`register` without the lock, for the constructor."""
        key = canonical_name(name)
        if key not in DEFAULT_MODELS:
            if not isinstance(name, str) or not _NAME_RE.match(key):
                raise ValueError(
                    "invalid checkpoint name %r: use lowercase letters, digits, '.', '_' or '-', "
                    "starting with a letter or digit" % (name,)
                )
            if key == "auto":
                raise ValueError(
                    "invalid checkpoint name %r: 'auto' means automatic routing in every "
                    "model= argument" % (name,)
                )
            if key == VEGA_MODEL:
                raise ValueError(
                    "invalid checkpoint name %r: it names the vega lane; pass Router(vega=...) "
                    "to configure it" % (name,)
                )
        if source is not None and not isinstance(source, (str, tuple, list)):
            raise TypeError(
                "checkpoint source for %r must be a repo id, a local path or a (repo, subfolder) "
                "pair, got %s" % (name, type(source).__name__)
            )
        # The sequence half of that sentence was never opened. `_split` pads with one `None`, so an
        # empty pair unpacks to one value and `registered`, `route(model=...)` and `load()` all fail
        # later with `ValueError: not enough values to unpack (expected 2, got 1)` -- after the
        # registration reported success -- while a third element is silently truncated and a number
        # part is stringified into a repo id like `acme/5`. Refuse the shape here, where the name is
        # known, so nothing is stored and no later call has to guess what the pair meant.
        if isinstance(source, (tuple, list)) and (
            not 1 <= len(source) <= 2
            or not all(part is None or isinstance(part, str) for part in source)
            or not any(isinstance(part, str) for part in source)
        ):
            raise TypeError(
                "checkpoint source for %r must be a repo id, a local path or a (repo, subfolder) "
                "pair of strings, got %r" % (name, source)
            )
        if isinstance(source, str) and source.startswith("~"):
            source = os.path.expanduser(source)  # a local path, never a Hub repo id
        # After every check that can refuse the registration, and before the write. A filter
        # that turns warnings into errors then raises with nothing stored, and `register`'s
        # rollback still runs because the warning is inside its try. Built-in names and their
        # aliases re-point that checkpoint, so they are not typos of themselves.
        # 3, not 2: the frames from here are `_add` -> `register` / `Router.__init__` / `attach`
        # -> the caller. 2 points inside this file, and the default once-per-location filter
        # would then warn once for the process instead of once per call site.
        if key not in DEFAULT_MODELS:
            near = _near_builtin(key)
            if near is not None:
                warnings.warn(
                    "laya: checkpoint name %r is one edit from the built-in %r; it is registered "
                    "as a separate checkpoint and %r is unchanged." % (key, near, near),
                    RuntimeWarning,
                    stacklevel=3,
                )
        self.models[key] = source
        return key

    # ------------------------------------------------------------------ loading
    def load(self, name: str):
        """Return the Agent for `name`, downloading and building it on first use.

        Concurrent callers share a single Agent instead of building duplicates.
        """
        key = self.resolve(name)
        while True:
            with self._lock:
                if key in self._agents:
                    self._touch(key)
                    return self._agents[key]
                inflight = self._loading.get(key)
                if inflight is None:
                    inflight = _InFlightBuild()
                    self._loading[key] = inflight
                    break
            inflight.done.wait()
            if inflight.error is not None:
                raise inflight.error

        agent = None
        evicted = []
        try:
            with self._build_lock:
                with self._lock:
                    # Built (or attached) while this caller waited for the build lock.
                    if key in self._agents:
                        self._touch(key)
                        agent = self._agents[key]
                        if self._loading.get(key) is inflight:
                            self._loading.pop(key, None)
                        inflight.done.set()
                        return agent
                built_agent = self._build(key)
                with self._lock:
                    if key in self._agents:  # attached while it was building: keep that one
                        self._touch(key)
                        agent = self._agents[key]
                    else:
                        agent = built_agent
                        self._agents[key] = agent
                        self._order.append(key)
                        evicted = self._evict_locked()
                    if self._loading.get(key) is inflight:
                        self._loading.pop(key, None)
                    inflight.done.set()
        except BaseException as e:
            with self._lock:
                if self._loading.get(key) is inflight:
                    self._loading.pop(key, None)
                inflight.error = e
                inflight.done.set()
            raise

        # Lifecycle hooks fire after the lock is released, so a hook can safely call the Router.
        self._dispatch_lifecycle("on_evict", evicted)
        dispatch(
            compose_hooks(self.hooks),
            "on_load",
            PredictContext(states=[], questions={}, model=key, agent=agent, router=self),
            raise_errors=self.hooks_raise,
            lock=self._hooks_lock,
            timeout=self.hooks_timeout,
        )
        return agent

    def _build(self, key: str):
        """Construct the Agent for `key`: config is read under `_lock`, the build runs outside it."""
        from .agent import load as Agent

        if key == VEGA_MODEL and self._vega_spec is not None:
            if not isinstance(self._vega_spec, (bool, str)):
                return self._vega_spec  # an attached agent, back after an eviction
            raise ImportError(_NO_VEGA_ENGINE % ("Router(vega=%r)" % (self._vega_spec,)))
        with self._lock:
            if self.models.get(key) is None:
                raise ValueError(
                    "checkpoint %r has no source to load from: it was attached as a built Agent "
                    "and is no longer resident; attach it again, or register a path or repo" % key
                )
            repo, sub = _split(self.models[key])
            kwargs = {"device": self.device, "token": self.token, "subfolder": sub}
            # A None or blank entry in `revisions` is "no pin of its own", so the checkpoint
            # inherits `revision`. Reading it as a pin of nothing dropped the caller's `revision`
            # and handed that one checkpoint to `$LAYA_REVISION` instead, which is the opposite of
            # what asking for a pin means.
            model_revision = _revision_pin(self.revisions.get(key)) or _revision_pin(self.revision)
            if model_revision is not None:
                kwargs["revision"] = model_revision
            # Safe to merge last for the names above, which `agent_kwargs` refuses -- but not for
            # `expected_sha256`, which is deliberately reachable through `agent_kwargs` and is
            # therefore merged with this checkpoint's entry rather than replaced by it.
            kwargs.update(self.agent_kwargs)
            expected = _merge_expected_digests(
                kwargs.pop("expected_sha256", None), _digest_entry(self.sha256_digests, key)
            )
            if expected is not None:
                kwargs["expected_sha256"] = expected
        return Agent(repo, **kwargs)

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict_locked(self) -> List[str]:
        """Drop least-recently-used agents until `max_loaded` holds. Returns evicted names."""
        evicted: List[str] = []
        while len(self._order) > self.max_loaded:
            victim = self._order.pop(0)
            agent = self._agents.pop(victim, None)
            if agent is not None:
                evicted.append(victim)
                del agent
        if len(self._order) < len(self._agents):  # keep the two views consistent
            for k in list(self._agents):
                if k not in self._order:
                    agent = self._agents.pop(k, None)
                    if agent is not None:
                        evicted.append(k)
                        del agent
        if evicted:
            gc.collect()
            from .agent import clear_cache

            clear_cache()
        return evicted

    def _evict(self) -> List[str]:
        with self._lock:
            return self._evict_locked()

    def _dispatch_lifecycle(self, event: str, names: List[str]) -> None:
        for name in names:
            dispatch(
                compose_hooks(self.hooks),
                event,
                PredictContext(states=[], questions={}, model=name, router=self),
                raise_errors=self.hooks_raise,
                lock=self._hooks_lock,
                timeout=self.hooks_timeout,
            )

    def attach(self, name: str, agent: Any):
        """Register an already-built Agent under `name` instead of loading a second copy.

        Useful when the process has a checkpoint loaded for other reasons: a demo that already
        built `convaiinnovations/laya` can hand it to the router rather than pay for -- and hold
        in memory -- a duplicate 421M parameters.

        A name the Router does not know yet is registered on the spot, with no source: the agent
        serves under that name while resident, and loading it again after an unload needs
        `register(name, source)` first.
        """
        with self._lock:
            try:
                key = self.resolve(name)
            except ValueError:
                key = self._add(name, None)
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: Optional[List[str]] = None):
        """Download and build checkpoints up front so no request ever pays a model load.

        A cold load costs seconds; language detection costs microseconds. With every
        checkpoint resident, routing is effectively free -- which is what you want in a
        server or a demo. `max_loaded` is raised to fit both the requested checkpoints and
        all already-resident agents, so incremental preloading does not evict either.
        """
        if names is None:
            # Every checkpoint that has a source to build from; an `attach`ed agent registered
            # without one is resident already and would raise here once evicted.
            with self._lock:
                names = [n for n, source in self.models.items() if source is not None]
        names = [self.resolve(n) for n in names]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(set(names) | set(self._agents)))
        for n in names:
            with self._lock:
                already = n in self._agents  # an attached agent is already built
            if not already:
                # load() dispatches on_load outside the lock; do not hold it across the call.
                self.load(n)
        return self

    def _unload_key(self, key: str) -> List[str]:
        """`unload` for a canonical key that may already be gone from `models`; returns what it freed."""
        freed: List[str] = []
        while True:
            with self._lock:
                inflight = self._loading.get(key)
                if inflight is None:
                    agent = self._agents.pop(key, None)
                    if key in self._order:
                        self._order.remove(key)
                    freed = [key] if agent is not None else []
                    del agent
                    gc.collect()
                    from .agent import clear_cache

                    clear_cache()
                    break
            inflight.done.wait()
        return freed

    def unload(self, name: Optional[str] = None):
        """Free one model, or all of them."""
        # When unloading a specific checkpoint, wait only for an in-flight build of THAT
        # checkpoint so unrelated builds holding `_build_lock` do not stall this unload.
        # When unloading all checkpoints (`name is None`), wait for all in-flight builds.
        freed: List[str] = []
        if name is not None:
            freed = self._unload_key(self.resolve(name))
        else:
            while True:
                with self._lock:
                    inflights = list(self._loading.values())
                    if not inflights:
                        freed = list(self._order)
                        self._agents.clear()
                        self._order.clear()
                        gc.collect()
                        from .agent import clear_cache

                        clear_cache()
                        break
                for inflight in inflights:
                    inflight.done.wait()

        self._dispatch_lifecycle("on_evict", freed)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    @property
    def loaded_revisions(self) -> Dict[str, Optional[str]]:
        """Commit SHA each resident agent was loaded from (None for local paths)."""
        with self._lock:
            return {name: getattr(agent, "revision", None) for name, agent in self._agents.items()}

    def _resolve_hint(self, hint: Any, state: Union[str, dict, list, None]) -> Optional[bool]:
        """True/False for a hint about whether the English checkpoint can read `state`.

        `hint` is either a language code or a callable taking the state. Anything the hint
        cannot answer returns None, which makes `route` fall through to detection rather than
        picking a checkpoint on no evidence.
        """
        if hint is None:
            return None
        if callable(hint):
            hint = hint(state)
        return _english_from_code(hint)

    def _vega_repo(self) -> str:
        spec = self._vega_spec
        if isinstance(spec, str):
            return spec
        model_id = getattr(spec, "model_id", None)
        return model_id if isinstance(model_id, str) else VEGA_REPO

    # ------------------------------------------------------------------ routing
    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        *,
        explain: bool = False,
        media: Optional[str] = None,
        allow_vega: bool = True,
    ) -> RouteDecision:
        """Decide which checkpoint to use, then let `on_route` hooks observe or replace it.

        `media` is "image", "audio" or "video" when the caller is passing one of those instead of
        a text state; see `predict`. `allow_vega=False` keeps the decision on the encoders.

        `ctx.decision` is the `RouteDecision`; a hook may replace it (for example to pin a
        checkpoint) and the replacement is what gets returned and used. `hooks` are per-call
        hooks, appended after any installed on the Router.

        `explain=True` adds language diagnostics when built-in detection runs. Explicit routing
        overrides still skip it. A hook can replace the decision, including its diagnostics.
        """
        if questions is not None and not isinstance(questions, dict):
            raise TypeError(
                "questions must be a dict of question id -> definition, got %s"
                % type(questions).__name__
            )
        decision = self._route(
            state,
            questions,
            model=model,
            task=task,
            lang=lang,
            lang_guess=lang_guess,
            media=media,
            allow_vega=allow_vega,
            **({"explain": True} if explain else {}),
        )
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        active = compose_hooks(self.hooks, hooks)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        if not active or not _has_on_route(active):
            return decision
        ctx = PredictContext(
            states=[state], questions=questions or {}, decision=decision, router=self
        )
        dispatch(
            active,
            "on_route",
            ctx,
            raise_errors=raise_errors,
            lock=self._hooks_lock,
            timeout=timeout,
        )
        return ctx.decision

    def _route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        *,
        explain: bool = False,
        media: Optional[str] = None,
        allow_vega: bool = True,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything.

        Precedence: explicit `model` > explicit `task` > the vega lane (an image, a recording or a
        clip; a question with a fitted head; a state past `vega_over_tokens`) > detected workflow
        (opt-in) > explicit `lang` > `lang_guess` > detected script/language > default.

        `media` names the kind of non-text input the caller passed ("image", "audio" or "video");
        `allow_vega=False` keeps a decision on the encoders, which is what `predict_batch` does,
        because a batch shares one forward pass on one checkpoint.

        `lang_guess` is an opt-in hint -- a language code or a callable taking the state --
        checked after an explicit `lang` and before the built-in detection. It only answers
        "can the English checkpoint read this?", so any non-English code routes to the
        multilingual checkpoint. A hint that resolves to nothing falls through to detection,
        which lets a language-identification model abstain. Pass one here, or set
        `Router(lang_guess=...)` to apply it to every request.
        """
        # `.get`, not `[]`: a concurrent `unregister` can remove the name between `resolve` and this
        # lookup, and the decision then reports no repo rather than raising.
        if model is not None:
            key = self.resolve(model)
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models.get(key)),
                reason="explicit model=%r" % model,
                detection=None,
                workflow=None,
            )

        if task is not None:
            key = self.resolve(
                "typed-decisions"
                if str(task).lower().replace("-", "_") == "typed_decisions"
                else task
            )
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models.get(key)),
                reason="explicit task=%r" % task,
                detection=None,
                workflow=None,
            )

        vega_on = allow_vega and self.vega_enabled
        if media is not None:
            if not vega_on:
                raise ValueError(
                    "an %s needs the vega lane, which this Router does not have: build an "
                    "official laya.Agent(vegaml=True) and pass it as Router(vega=agent), or pass "
                    "media to that agent directly" % media
                )
            return RouteDecision(
                model=VEGA_MODEL,
                repo=self._vega_repo(),
                reason="%s input: the encoders have no vision or speech path" % media,
                detection=None,
                workflow=None,
            )
        if vega_on:
            fitted = sorted(q for q in (questions or {}) if q in self._vega_heads())
            if fitted:
                return RouteDecision(
                    model=VEGA_MODEL,
                    repo=self._vega_repo(),
                    reason="vega has fitted heads for %s" % ", ".join(fitted),
                    detection=None,
                    workflow=None,
                )
            n_tokens = _estimate_tokens(state)
            if n_tokens > self.vega_over_tokens:
                return RouteDecision(
                    model=VEGA_MODEL,
                    repo=self._vega_repo(),
                    reason="state is about %d tokens, past the %d the encoders can read; "
                    "vega reads it whole" % (n_tokens, self.vega_over_tokens),
                    detection=None,
                    workflow=None,
                )

        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(
                model="typed-decisions",
                repo=_repo_str(self.models["typed-decisions"]),
                reason="question ids match the %r typed-decisions workflow" % workflow,
                detection=None,
                workflow=workflow,
            )

        if lang is not None:
            # An explicit `lang` is decisive only when the code names a language. Blank or
            # whitespace resolves to no usable hint, so it falls through to lang_guess/detection
            # exactly as an abstaining hint does; real English/non-English codes still route now.
            resolved = _english_from_code(lang)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                return RouteDecision(
                    model=key,
                    repo=_repo_str(self.models[key]),
                    reason="explicit lang=%r" % lang,
                    detection=None,
                    workflow=workflow,
                )

        # Caller-supplied hint, per-call first then the one installed on the Router. Only a hint
        # that actually answers the question routes here; anything else falls through.
        for source, hint in (
            ("lang_guess", lang_guess),
            ("Router(lang_guess=...)", self.lang_guess),
        ):
            resolved = self._resolve_hint(hint, state)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                return RouteDecision(
                    model=key,
                    repo=_repo_str(self.models[key]),
                    reason="%s: the caller identified this as %s text"
                    % (source, "English" if resolved else "non-English"),
                    detection=None,
                    workflow=workflow,
                )

        det = analyse(state, explain=True) if explain else analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = (
                "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                % (det["script"], 100 * float(det["non_latin_fraction"]))
            )
        elif not det["is_english"]:
            key = "multilingual"
            if det.get("mixed_segment"):
                reason = (
                    "Latin script, mostly English, but a line or field reads as %r (%r); "
                    "the English checkpoint cannot read it"
                    % (det["language"], det["mixed_segment"][:60])
                )
            elif det["language"]:
                reason = "Latin script but language looks like %r, not English" % det["language"]
            else:
                # Unidentified Latin-script language: routed on the non-English letters alone,
                # because no stopword list here covers it.
                reason = (
                    "Latin script, language not identified but %.0f%% non-English letters; "
                    "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"]))
                )
        elif det["language_undecided"]:
            # Nothing identifies the language: too short, or only content words ("Quero cancelar",
            # "Esqueci minha senha"). That is no evidence of English either, so it takes the same
            # `default` as a state with no letters, which is `multilingual` since 0.4.0.
            # #54 measured the cost of the old English default here: 128 of 200 German
            # utterances reached this branch and lost 20 accuracy points against
            # `model="multilingual"` on the same rows. A mostly-English deployment sets
            # `Router(default="english")`.
            key = self.default
            reason = (
                "Latin script, language not identified and no non-English letters; "
                "using default (%s)" % key
            )
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(
            model=key,
            repo=_repo_str(self.models[key]),
            reason=reason,
            detection=det,
            workflow=workflow,
        )

    # ------------------------------------------------------------------ running
    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        min_confidence: Optional[float] = None,
        image: Any = None,
        audio: Any = None,
        video: Any = None,
    ) -> Dict[str, Any]:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        `image`, `audio` or `video` in place of `state` route to the vega lane, which is the only
        checkpoint that can read them; so does a state past `vega_over_tokens` and a question the
        vega agent has a fitted head for. Everything else routes exactly as before.

        The result is the usual `system_one` payload plus a `routing` key recording the decision.
        Router-level `on_predict_start` / `on_predict_end` hooks wrap the whole route+infer call
        and see `ctx.decision`; see `laya.hooks`. `max_len` / `head_max_len` override the agent
        token budget for this call (a start hook may set `ctx.max_len` / `ctx.head_max_len`).
        """
        media_given = [
            k for k, v in (("image", image), ("audio", audio), ("video", video)) if v is not None
        ]
        if len(media_given) > 1:
            raise ValueError("pass one of image=, audio=, video=; got %s" % ", ".join(media_given))
        media = media_given[0] if media_given else None
        if state is None and media is None:
            raise TypeError(
                "state must not be None; pass a string, dict, or list (or image=, audio=, video=)"
            )
        if state is not None and media is not None:
            raise ValueError("pass either a state or %s=, not both" % media)
        if not isinstance(questions, dict):
            raise TypeError(
                "questions must be a dict of question id -> definition, got %s"
                % type(questions).__name__
            )
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)

        ctx = PredictContext(
            states=[state],
            questions=questions,
            decision=None,
            model=None,
            agent=None,
            router=self,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        # Read and clear: this call owns the scan, and a hook that calls back into `predict` must
        # not inherit it. `predict_long` restores the previous value when it returns.
        scan = _SCAN.get()
        if scan is not None:
            _SCAN.set(None)
        try:
            # Per-call hooks apply to the whole call, including on_route inside route().
            decision = self.route(
                state,
                questions,
                model=model,
                task=task,
                lang=lang,
                lang_guess=lang_guess,
                hooks=hooks,
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
                media=media,
            )
            ctx.decision = dict(decision)
            ctx.model = decision["model"]
            agent = self.load(decision["model"])
            ctx.agent = agent
            effective_lang = lang
            if (
                effective_lang is None
                and decision.get("detection")
                and decision["detection"].get("language")
            ):
                effective_lang = decision["detection"]["language"]

            # Keep the existing success-path timing contract: elapsed_ms starts here, after
            # route/load and immediately before the prediction start hook.
            ctx.started_at = time.perf_counter()
            dispatch(
                active,
                "on_predict_start",
                ctx,
                raise_errors=raise_errors,
                lock=self._hooks_lock,
                timeout=timeout,
            )
            if ctx.results is None:
                skip = _SKIP_DEFAULTS.set(True)
                try:
                    if scan is not None:
                        # After the start chain, not inside it: the scan is laya's own forward pass,
                        # so it is not bounded by `hooks_timeout`, does not hold the hooks lock, and
                        # runs under `_SKIP_DEFAULTS` like every other inference this method does.
                        # It sizes its windows from `window`/`stride` and the agent's own config, so
                        # `ctx.max_len`/`ctx.head_max_len` do not reach it -- as has always been the
                        # case, and as `predict_long` documents by not accepting them.
                        result = scan.run(ctx)
                    else:
                        # Pass token-budget overrides only when set, so any Agent-like object that
                        # does not accept them still works on the default path.
                        overrides = {}
                        if ctx.max_len is not None:
                            overrides["max_len"] = ctx.max_len
                        if ctx.head_max_len is not None:
                            overrides["head_max_len"] = ctx.head_max_len
                        if decision["model"] == VEGA_MODEL:
                            # vega's predict takes laya's shape plus the media; `max_len` and
                            # `head_max_len` are recorded there and not applied
                            result = agent.predict(
                                ctx.states[0],
                                ctx.questions,
                                lang=effective_lang,
                                image=image,
                                audio=audio,
                                video=video,
                                **overrides,
                            )
                        else:
                            try:
                                result = agent.system_one(
                                    ctx.states[0], ctx.questions, lang=effective_lang, **overrides
                                )
                            except TypeError as e:
                                if "unexpected keyword argument 'lang'" in str(e):
                                    result = agent.system_one(
                                        ctx.states[0], ctx.questions, **overrides
                                    )
                                else:
                                    raise
                finally:
                    _SKIP_DEFAULTS.reset(skip)
                if scan is not None:
                    # `setdefault`, because the scan used to set `ctx.results` from inside the
                    # hook chain and reach the branch below: an agent whose `predict_long` returns
                    # its own `routing` kept it, and that stays true.
                    result.setdefault("routing", dict(decision))
                else:
                    result["routing"] = dict(decision)
                ctx.results = [result]
            else:
                # A cache hit short-circuits inference, but Router.predict still promises a
                # `routing` key. Add it without overwriting a routing the cached payload has.
                for result in ctx.results:
                    if isinstance(result, dict):
                        result.setdefault("routing", dict(decision))
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(
                    active,
                    "on_error",
                    ctx,
                    raise_errors=raise_errors,
                    lock=self._hooks_lock,
                    timeout=timeout,
                )
            except BaseException as hook_exc:
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
                apply_confidence_gate(ctx.results, mc)
            try:
                dispatch(
                    active,
                    "on_predict_end",
                    ctx,
                    raise_errors=raise_errors,
                    lock=self._hooks_lock,
                    timeout=timeout,
                )
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results[0]

    def predict_long(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        aggregate: str = "auto",
        batch_size: Optional[int] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Route, then scan every window of the state instead of only its first one.

        `predict` scores a state from a single window: anything past `max_len` is cut off (the
        first window, or for a conversation list the last) and never reaches the model. This
        routes exactly as `predict` does -- the same `model`/`task`/`lang` hints, the same
        router-level hooks, the same `routing` key and `usage` -- and scores the routed state with
        that agent's `predict_long`, which splits it into overlapping windows and aggregates per
        question. The aggregation rules are `laya.agent.Agent.predict_long`'s: `noul` takes the
        strongest window, `choice`/`score` the most confident one.

        Per-call hooks (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`,
        `hooks_timeout`) wrap the whole route+scan exactly as they wrap `predict`: the scan runs
        last, so a start hook that answers (`ctx.skip(...)`) or rewrites the state wins.
        `max_len` / `head_max_len` are not accepted here -- a window is sized by `window` or the
        checkpoint budget, and overriding the single-window truncation is what `predict_long` is for.

        Args:
            window: state tokens per window. Defaults to the routed checkpoint's budget
                    (`max(64, max_len - head_max_len - 8)`; the 64 is a floor), capped at the room
                    the questions leave for the state so no window is truncated again on the way
                    in; a smaller window isolates a localized span.
            stride: token step between windows; defaults to half the effective window (50%
                    overlap). A stride past that window is a `ValueError`, since the tokens between
                    windows would reach no model.
            aggregate: "auto" (the per-type rules above) is the only mode.
            batch_size: cap on windows per forward pass, to bound memory on very long states.

        Returns:
            The usual `predict` payload, with `usage["windows"]` counting the windows scored.

        Raises:
            TypeError: the routed agent has no `predict_long` -- one attached by hand, since both
                    `Agent` and `ONNXAgent` implement it -- so there is nothing to scan with.
                    Always raised, whatever `hooks_raise` is, and so is every other error the scan
                    itself raises -- none are caught: the scan is this method's own work, not a
                    caller's hook, so the hook error policy does not decide whether it may be
                    skipped. It previously ran as a start hook, where `hooks_raise=False` swallowed
                    this and returned one window scored by `system_one` -- a different question than
                    the one asked. An agent whose `predict_long` has no `lang` parameter is scanned
                    without one and warned about, not failed; that is a signature check, not a
                    swallowed error.
        """
        token = _SCAN.set(
            _ScanLong(
                window=window, stride=stride, aggregate=aggregate, batch_size=batch_size, lang=lang
            )
        )
        try:
            return self.predict(
                state,
                questions,
                model=model,
                task=task,
                lang=lang,
                lang_guess=lang_guess,
                hooks=normalise_hooks(hooks, on_predict_start, on_predict_end),
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
            )
        finally:
            _SCAN.reset(token)

    def decide(
        self,
        state: Union[str, dict, list],
        schema: Any = None,
        *,
        questions: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
        min_confidence: Optional[float] = None,
        **predict_kwargs,
    ) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        (for example `model=`, `task=`, `hooks=`) are forwarded to `predict`.
        """
        from .structured import decide as _decide

        return _decide(
            self,
            state,
            schema,
            questions=questions,
            return_details=return_details,
            min_confidence=min_confidence,
            **predict_kwargs,
        )

    def decide_batch(
        self,
        states: Sequence[Any],
        schema: Any = None,
        *,
        questions: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
        min_confidence: Optional[float] = None,
        **predict_kwargs,
    ) -> List[Any]:
        """Answer many states against one schema (JSON schema or pydantic model) in one batched call.

        The throughput form of :meth:`decide`: the schema is planned once and its questions
        run over every state through :meth:`predict_batch` (grouped forward passes, results in
        input order), then each state's answers are projected as ``decide`` does. Extra keyword
        arguments (``batch_size=``, ``model=``, ``hooks=``, ...) are forwarded to
        ``predict_batch``. See `laya.structured`.
        """
        from .structured import decide_batch as _decide_batch

        return _decide_batch(
            self,
            states,
            schema,
            questions=questions,
            return_details=return_details,
            min_confidence=min_confidence,
            **predict_kwargs,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    system_one = predict

    def route_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        hooks_timeout: Optional[float] = None,
        *,
        hooks=None,
        hooks_raise: Optional[bool] = None,
        explain: bool = False,
    ) -> List[RouteDecision]:
        """Route a heterogeneous request batch without loading any checkpoints.

        Each request is a mapping with ``state`` and ``questions`` plus the same optional
        routing overrides accepted by :meth:`route`: ``model``, ``task``, ``lang`` and
        ``lang_guess``. The returned decisions preserve input order.

        This is intentionally separate from inference so callers can inspect or aggregate
        routing decisions before paying model-load cost.

        Args:
            requests: Sequence of request dictionaries, each requiring ``state`` and
                ``questions``.
            hooks_timeout: Override the Router's ``hooks_timeout`` for this call, applied to
                every request's ``on_route`` dispatch, as on :meth:`route`.
            hooks (HookArg): Per-call hook or sequence of hooks for this call.
            hooks_raise: Override the Router's ``hooks_raise`` policy for this call.
            explain: Include built-in language diagnostics for each request that uses detection.
        """
        if not isinstance(requests, SequenceABC) or isinstance(requests, (str, bytes)):
            raise TypeError("requests must be a sequence of request dictionaries")
        if not requests:
            return []

        use_fast = type(self).route is Router.route
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        active = compose_hooks(self.hooks, hooks)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        has_on_route = _has_on_route(active)

        decisions: List[RouteDecision] = []
        for i, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError("request %d must be a dict, got %s" % (i, type(request).__name__))
            if "state" not in request:
                raise ValueError("request %d is missing required key 'state'" % i)
            if request["state"] is None:
                raise TypeError(
                    "request %d 'state' must not be None; pass a string, dict, or list" % i
                )
            if "questions" not in request:
                raise ValueError("request %d is missing required key 'questions'" % i)

            questions = request["questions"]
            if not isinstance(questions, dict):
                raise TypeError(
                    "request %d 'questions' must be a dict, got %s" % (i, type(questions).__name__)
                )

            if not use_fast:
                decisions.append(
                    self.route(
                        request["state"],
                        questions,
                        model=request.get("model"),
                        task=request.get("task"),
                        lang=request.get("lang"),
                        lang_guess=request.get("lang_guess"),
                        hooks=hooks,
                        hooks_raise=hooks_raise,
                        hooks_timeout=hooks_timeout,
                        allow_vega=False,
                        **({"explain": True} if explain else {}),
                    )
                )
            elif not has_on_route:
                decisions.append(
                    self._route(
                        request["state"],
                        questions,
                        model=request.get("model"),
                        task=request.get("task"),
                        lang=request.get("lang"),
                        lang_guess=request.get("lang_guess"),
                        allow_vega=False,
                        **({"explain": True} if explain else {}),
                    )
                )
            else:
                decision = self._route(
                    request["state"],
                    questions,
                    model=request.get("model"),
                    task=request.get("task"),
                    lang=request.get("lang"),
                    lang_guess=request.get("lang_guess"),
                    allow_vega=False,
                    **({"explain": True} if explain else {}),
                )
                ctx = PredictContext(
                    states=[request["state"]],
                    questions=questions,
                    decision=decision,
                    router=self,
                )
                dispatch(
                    active,
                    "on_route",
                    ctx,
                    raise_errors=raise_errors,
                    lock=self._hooks_lock,
                    timeout=timeout,
                )
                decisions.append(ctx.decision)

        return decisions

    def predict_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        batch_size: Optional[int] = None,
        hooks_timeout: Optional[float] = None,
        min_confidence: Optional[float] = None,
        sort_by_length: bool = False,
        *,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Route and execute a heterogeneous request batch with minimal model churn.

        Requests are routed first and grouped by checkpoint. Within each checkpoint,
        requests that share the same question schema are passed to
        ``Agent.predict_batch`` so their states can share forward passes. Results are
        then restored to the original request order.

        Requests may independently specify ``model``, ``task``, ``lang``,
        ``lang_guess``, ``max_len`` or ``head_max_len`` and may use different question schemas.
        ``max_len`` / ``head_max_len`` are the per-request form of the token-budget override
        ``predict`` takes as call arguments: they set the checkpoint's state and question-head
        budgets for that one request, so a wide question can be asked without shrinking the
        batch's other requests to the same window. Requests that ask for different budgets are
        split into separate forward passes, since one ``Agent.predict_batch`` call carries one
        budget for all its states. A start hook may still replace either value on ``ctx``.

        Router-level predict hooks run per request, as ``predict`` runs them: each request
        gets its own ``PredictContext``, so ``on_predict_start`` can replace that request's
        state, questions or token budget, or ``ctx.skip(...)`` it, and ``on_predict_end``
        sees and may replace its result. Requests are grouped for the forward pass after
        their start hooks have run, and a checkpoint group's requests end in reverse of the
        order they started. If a checkpoint group fails, every request of it whose start hook
        ran fails with the exception, a cache hit included: each gets ``on_error`` and then
        ``on_predict_end`` before the exception propagates.

        Args:
            requests: Sequence of request dictionaries. Every item requires ``state`` and
                ``questions`` and may include ``model``, ``task``, ``lang`` or
                ``lang_guess`` routing overrides and ``max_len`` / ``head_max_len`` token-budget
                overrides.
            batch_size: Optional maximum number of states per Agent forward-pass batch.
            hooks_timeout: Override the Router's ``hooks_timeout`` for this call.
            min_confidence: Optional float or per-bucket mapping for confidence gating.
            sort_by_length: Forwarded to every ``Agent.predict_batch`` call, so each question
                group pads to a shorter maximum; see ``Agent.predict_batch``. Results retain the
                input order either way. Silently dropped for an attached agent whose
                ``predict_batch`` predates the knob (#294).
            hooks (HookArg): Per-call hook or sequence of hooks for this call.
            on_predict_start (PredictHookArg): Plain callable or sequence of callables for start events.
            on_predict_end (PredictHookArg): Plain callable or sequence of callables for end events.
            hooks_raise: Override the Router's ``hooks_raise`` policy for this call.
            diagnostics: Optional dict that receives the per-group sort outcome under the key
                ``"sort_by_length"`` once the batch has run: how many question groups actually
                reached an agent, how many of those could reorder, and the smallest / largest
                group size. The reorder switches per group inside this call, so a caller that
                wants to say whether the knob did anything (#1111) reads this instead of
                guessing from the request count beforehand. Untouched when ``sort_by_length``
                is false or no group reached an agent.

        Returns:
            One normal Router prediction result per request, in the same order as the input.
        """
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        decisions = self.route_batch(
            requests, hooks=hooks, hooks_raise=hooks_raise, hooks_timeout=hooks_timeout
        )
        if not decisions:
            return []

        # Dict insertion order preserves the order in which model groups first appear. This
        # keeps cache effects deterministic while collapsing an arbitrarily interleaved
        # workload to at most one load per routed checkpoint for this call.
        groups: Dict[str, List[int]] = {}
        for i, decision in enumerate(decisions):
            # Indexed, not `.model`: an on_route hook may replace the decision with a plain dict,
            # which `predict` accepts too. Normalised, because such a hook may name the checkpoint
            # by an alias ("ml"), and that request must share its checkpoint's forward pass.
            groups.setdefault(self.resolve(decision["model"]), []).append(i)

        # Counted rather than marked with None: an end hook may leave a None result, which
        # `predict` returns as it is.
        results: List[Any] = [None] * len(requests)
        answered = 0
        # The per-group sort outcome, kept only for a caller that asked for it (#1111): the
        # reorder switches per question group inside the loop below, and no caller outside
        # this method can see the group sizes it forms.
        sort_report = (
            {"groups": 0, "reordered": 0, "smallest": None, "largest": None}
            if sort_by_length and diagnostics is not None
            else None
        )
        # `compose_hooks`, not `list(self.hooks)`: the composition `predict` uses. It merges
        # `set_default_hooks` defaults, then installed hooks, then this call's `hooks`,
        # `on_predict_start` and `on_predict_end`. Reading the instance list alone silently
        # dropped every `set_default_hooks` default from the batched path while `predict`
        # kept them, which was the bug behind #909: a default audit or metrics hook saw no
        # Router-level event for a request that arrived through `predict_batch`.
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)

        for model_name, indices in groups.items():
            agent = self.load(model_name)
            started: List[PredictContext] = []
            try:
                # One context per request, built and started the way `predict` does it, so a
                # start hook sees -- and can redact, rewrite or skip -- each request before it
                # joins a shared forward pass. A request's own `max_len` / `head_max_len` seed the
                # context here; the grouping below already splits on those fields, so this is the
                # only place the per-request budget can enter.
                for i in indices:
                    ctx = PredictContext(
                        states=[requests[i]["state"]],
                        questions=requests[i]["questions"],
                        decision=dict(decisions[i]),
                        model=model_name,
                        agent=agent,
                        router=self,
                        max_len=requests[i].get("max_len"),
                        head_max_len=requests[i].get("head_max_len"),
                    )
                    started.append(ctx)
                    dispatch(
                        active,
                        "on_predict_start",
                        ctx,
                        raise_errors=raise_errors,
                        lock=self._hooks_lock,
                        timeout=timeout,
                    )

                # Agent.predict_batch evaluates one shared question schema and token budget over
                # many states. Preserve Router's heterogeneous-request API by splitting each
                # checkpoint group again on what the start hooks left, so a rewritten question
                # set or `ctx.max_len` only applies to its own request.
                question_groups: List[Dict[str, Any]] = []
                last_questions = None
                last_schema = None
                for i, ctx in zip(indices, started):
                    if ctx.results is not None:
                        # A cache hit short-circuits inference; keep the `routing` key predict adds.
                        for result in ctx.results:
                            if isinstance(result, dict):
                                result.setdefault("routing", dict(decisions[i]))
                        continue
                    # Pass token-budget overrides only when set, as `predict` does, so an
                    # Agent-like object that does not accept them still works.
                    if ctx.max_len is None and ctx.head_max_len is None:
                        overrides = {}
                    else:
                        overrides = {
                            key: value
                            for key, value in (
                                ("max_len", ctx.max_len),
                                ("head_max_len", ctx.head_max_len),
                            )
                            if value is not None
                        }
                    # `predict` forwards the language of the request so the agent can apply its
                    # per-language temperatures; the batched path forwarded only the token
                    # budgets, so the same request scored differently depending on the entry
                    # point. Only computed for an agent that actually carries them: `lang` is
                    # otherwise unused, and adding it to the group key would split a group that
                    # shares one forward pass today.
                    lang_key = None
                    if getattr(agent, "lang_temperatures", None):
                        lang_key = requests[i].get("lang")
                        if lang_key is None:
                            detection = decisions[i].get("detection") or {}
                            lang_key = detection.get("language")
                    # Order-sensitive at every nesting level (#166): options are positional, so two
                    # equal schemas with different key orders must not share a group.
                    if ctx.questions is last_questions:
                        schema = last_schema
                    else:
                        schema = _question_schema(ctx.questions)
                        last_questions = ctx.questions
                        last_schema = schema
                    for group in question_groups:
                        if (
                            group["schema"] == schema
                            and group["overrides"] == overrides
                            and group["lang"] == lang_key
                        ):
                            group["items"].append((i, ctx))
                            break
                    else:
                        question_groups.append(
                            {
                                "questions": ctx.questions,
                                "schema": schema,
                                "overrides": overrides,
                                "lang": lang_key,
                                "items": [(i, ctx)],
                            }
                        )

                for group in question_groups:
                    items = group["items"]
                    batch_kwargs = dict(group["overrides"])
                    if group["lang"] is not None:
                        batch_kwargs["lang"] = group["lang"]
                    if sort_by_length:
                        # Passed only when on, as the token-budget overrides are: agents attached
                        # via `attach()` may predate the knob (#294).
                        batch_kwargs["sort_by_length"] = True
                    skip = _SKIP_DEFAULTS.set(True)
                    try:
                        while True:
                            try:
                                batch_results = agent.predict_batch(
                                    [ctx.states[0] for _, ctx in items],
                                    group["questions"],
                                    batch_size=batch_size,
                                    **batch_kwargs,
                                )
                                break
                            except TypeError as e:
                                # An older agent may predate both optional arguments. Remove
                                # only the one it refused, preserving any supported argument;
                                # each removal bounds the retry to at most two extra calls.
                                for kwarg in ("lang", "sort_by_length"):
                                    if (
                                        kwarg in batch_kwargs
                                        and "unexpected keyword argument '%s'" % kwarg in str(e)
                                    ):
                                        batch_kwargs.pop(kwarg)
                                        break
                                else:
                                    raise
                    finally:
                        _SKIP_DEFAULTS.reset(skip)

                    if len(batch_results) != len(items):
                        raise RuntimeError(
                            "internal error: Agent.predict_batch returned %d results for %d states"
                            % (len(batch_results), len(items))
                        )

                    if sort_report is not None:
                        # What this group actually did with the knob: the agent reorders only
                        # when `batch_size` clears its own `1 < chunk < len(states)` gate
                        # (agent.py:1519, onnx_agent.py:761) over this group's states. A group
                        # whose agent predates the knob (#294) reports as unsorted, not as an
                        # error -- the run answered, just at one state's width.
                        n = len(items)
                        reorders = bool(
                            "sort_by_length" in batch_kwargs
                            and batch_size
                            and batch_size > 1
                            and batch_size < n
                        )
                        sort_report["groups"] += 1
                        sort_report["reordered"] += int(reorders)
                        if sort_report["smallest"] is None or n < sort_report["smallest"]:
                            sort_report["smallest"] = n
                        if sort_report["largest"] is None or n > sort_report["largest"]:
                            sort_report["largest"] = n

                    for (i, ctx), result in zip(items, batch_results):
                        result["routing"] = dict(decisions[i])
                        ctx.results = [result]

                # Usage is summed here, not while ending, so a malformed usage block (a cached
                # payload's, say) fails the group like any other error instead of escaping from
                # the end loop before a single end hook has run.
                usages = [aggregate_usage(ctx.results) for ctx in started]
                for ctx, usage in zip(started, usages):
                    ctx.usage = usage
            except BaseException as exc:
                # The caller gets this exception and no result, so every started request of the
                # group failed with it, including a cache hit or a request whose question group
                # had already run. Each is still ended, so a hook that opens something in start
                # (a span, an in-flight count) always sees the matching end.
                for ctx in started:
                    if ctx.results:
                        apply_confidence_gate(ctx.results, mc)
                self._end_contexts(active, started, raise_errors, timeout, error=exc)
                raise

            for ctx in started:
                if ctx.results:
                    apply_confidence_gate(ctx.results, mc)
            self._end_contexts(active, started, raise_errors, timeout)
            for i, ctx in zip(indices, started):
                results[i] = ctx.results[0]
            answered += len(indices)
            # Drop this group's references before the next load() can evict its checkpoint:
            # eviction's gc.collect() and empty_cache() only free an Agent nothing still holds,
            # and every context of the group holds it as `ctx.agent`.
            agent = ctx = started = question_groups = group = items = None

        # Every input index is assigned exactly once by construction. Keep this assertion local
        # so a future refactor cannot silently return a partially-filled batch.
        if answered != len(requests):
            raise RuntimeError("internal error: batch execution did not produce every result")

        if sort_report is not None:
            diagnostics["sort_by_length"] = sort_report
        return results

    predict_many = predict_batch

    def _end_contexts(
        self,
        active: List[Any],
        contexts: List[PredictContext],
        raise_errors: bool,
        timeout: Optional[float] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Finish each request of a batch the way `predict` finishes one.

        `error` is the exception that failed the batch, if one did: every context then fails
        with it and gets `on_error` before its `on_predict_end`, as in `predict`. Contexts end
        in reverse of the order they started, because all of them started before any ends: a
        hook that sets something in start and resets it in end (a contextvar, a tracing
        context) must unwind the last one first. Every context gets its `on_predict_end` even if
        another's hooks raise; the first such failure is raised afterwards. On a failed context,
        a raising hook is chained onto its error instead, as in `predict`. `elapsed_ms` is set
        on all of them first, so one request's never includes another's error or end hooks.
        """
        now = time.perf_counter()
        for ctx in contexts:
            ctx.elapsed_ms = (now - ctx.started_at) * 1000.0
            if error is not None:
                ctx.error = error
        first_error: Optional[BaseException] = None
        for ctx in reversed(contexts):
            events = (
                ("on_error", "on_predict_end") if ctx.error is not None else ("on_predict_end",)
            )
            for event in events:
                try:
                    dispatch(
                        active,
                        event,
                        ctx,
                        raise_errors=raise_errors,
                        lock=self._hooks_lock,
                        timeout=timeout,
                    )
                except BaseException as hook_exc:
                    if ctx.error is not None:
                        ctx.error.__context__ = hook_exc
                    elif first_error is None:
                        first_error = hook_exc
        if first_error is not None:
            raise first_error

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (
            self.loaded,
            self.max_loaded,
            self.default,
        )
