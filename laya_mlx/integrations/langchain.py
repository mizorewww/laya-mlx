"""LangChain and LangGraph integration for Laya System 1 decision engine.

Provides fast (~33 ms), non-autoregressive routing, real-time guardrails, schema-driven
decisions and state evaluation nodes for LangChain Expression Language (LCEL) and LangGraph.
Each runnable also batches: `batch()` and `abatch()` evaluate a list of inputs on
Laya's shared forward passes instead of one call per input.

Supports both local in-process models (`Agent` / `Router`) and remote HTTP
deployments (your own `laya-serve`) without requiring PyTorch on edge clients.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Optional LangChain base class integration
try:
    from langchain_core.runnables import RunnableConfig, RunnableSerializable

    _RUNNABLE_AVAILABLE = True
except ImportError:
    _RUNNABLE_AVAILABLE = False
    RunnableSerializable = object  # type: ignore
    RunnableConfig = Any  # type: ignore

# The per-call control rules live in `._controls` because the CrewAI and LlamaIndex wrappers end
# at the same `predict` call and the same laya-serve body; keeping the rule in three places is how
# two of them ended up forwarding only `model`.
from ..confidence import _gate_confidence
from ._controls import budget_kwargs as _budget_kwargs
from ._controls import decision_kwargs as _decision_kwargs
from ._controls import hook_kwargs as _hook_kwargs
from ._controls import predict_kwargs as _predict_kwargs
from ._controls import reject_remote_hooks as _reject_remote_hooks
from ._controls import reject_remote_lang_guess as _reject_remote_lang_guess
from ._controls import require_router_controls as _require_router_controls
from ._controls import router_kwargs as _router_kwargs
from ._guard import answer_number as _answer_number
from ._guard import answers_map as _answers_map
from ._guard import malformed as _malformed
from ._guard import score_violation_probability as _score_violation_probability


class LayaGuardrailError(ValueError):
    """Raised when an input violates a Laya guardrail policy."""

    def __init__(self, message: str, violations: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.violations = violations
        self.raw_decision = raw_decision


def _extract_text(
    input_val: Any, state_key: Optional[Union[str, Callable[[Any], Any]]] = None
) -> Union[str, dict, list]:
    """Extract evaluatable text from arbitrary LangChain/LangGraph states or messages."""
    if state_key is not None:
        if callable(state_key):
            return state_key(input_val)
        if isinstance(input_val, dict) and state_key in input_val:
            return _extract_from_message_or_value(input_val[state_key])

    if isinstance(input_val, str):
        return input_val

    if isinstance(input_val, dict):
        for candidate in ("input", "text", "query", "prompt", "message", "body", "content"):
            if candidate in input_val:
                return _extract_from_message_or_value(input_val[candidate])
        if "messages" in input_val and isinstance(input_val["messages"], list):
            return _extract_from_messages_list(input_val["messages"])
        return input_val

    if isinstance(input_val, list):
        text = _content_text(input_val)
        if text is not input_val:
            return text
        return _extract_from_messages_list(input_val)

    return str(input_val)


def _content_text(val: Any) -> Any:
    """The text of a message `content` value, unwrapped from a content-block list.

    `content` is not only a string. A `HumanMessage(content=[{"type": "text", "text": ...}])`
    is accepted by langchain-core and keeps `.content` as a list, so `str(content)` produced a
    Python repr -- braces, quotes, and the literal field names `type` and `text` -- and that
    repr is what Laya scored. Nothing raised; the guardrail simply answered about a string the
    caller never wrote.

    Text-bearing blocks are concatenated in the order they appear, which is how such a list is
    meant to be read. A list carrying no text block is returned **unchanged**, so a genuinely
    structured state still reaches the caller as the caller shaped it instead of being
    flattened into invented prose; callers distinguish "extracted" from "unchanged" by identity.
    A plain string is returned untouched.
    """
    if not isinstance(val, list):
        return val
    parts = []
    for block in val:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        return val
    return "".join(parts) if len(parts) == 1 else "\n".join(parts)


def _extract_from_message_or_value(val: Any) -> Any:
    if hasattr(val, "content"):
        return _content_text(val.content)
    if isinstance(val, list):
        # a content-block list is already text; a list of messages is not
        text = _content_text(val)
        if text is not val:
            return text
        return _extract_from_messages_list(val)
    return val


def _extract_from_messages_list(msgs: Sequence[Any]) -> Union[str, dict, list]:
    # Not `-> str`. `_content_text` hands a content value back untouched when it holds no
    # text block, so a message carrying only structured content returns that list, and a list
    # entry with no `.content` returns the entry itself. Both are states `Agent._encode_state`
    # already accepts (`state: Union[str, dict, list]`, documented as a conversation turn list),
    # and the public `_extract_text` above already declares the same three types -- so this
    # widens the annotation to match the behaviour, not the behaviour to match the annotation.
    if not msgs:
        return ""
    # Search backwards for the most recent human/user message
    for m in reversed(msgs):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return _content_text(getattr(m, "content", m))
    last = msgs[-1]
    return _content_text(getattr(last, "content", last))


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward bearer credentials across an origin or HTTPS downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        old_port = old.port or (443 if old.scheme.lower() == "https" else 80)
        new_port = new.port or (443 if new.scheme.lower() == "https" else 80)
        if (
            old.scheme.lower() != new.scheme.lower()
            or (old.hostname or "").lower() != (new.hostname or "").lower()
            or old_port != new_port
        ):
            raise urllib.error.URLError("refusing cross-origin or HTTPS-downgrade redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _call_remote(
    base_url: str,
    state: Any,
    questions: Dict[str, Any],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 10.0,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    lang: Optional[str] = None,
    min_confidence: Optional[float] = None,
    task: Optional[str] = None,
    lang_guess: Optional[str] = None,
) -> Dict[str, Any]:
    """Send decision request to a remote laya-serve HTTP instance using standard library urllib.

    `max_len` / `head_max_len` travel in the body; laya-serve applies them up to its
    `LAYA_MAX_TOKEN_BUDGET` ceiling and answers a larger value with 422. `lang` / `min_confidence`
    ride in the same body (they are laya-serve `BODY_CONTROLS` too): the language codes the state
    for routing and calibration, and the abstention gate flags a low-confidence answer. `task` names
    the checkpoint that answers and `lang_guess` is the language code routing reads before its own
    detection -- both are laya-serve body controls too, and only `lang_guess`'s string form has a
    wire representation (`reject_remote_lang_guess` refuses the callable before the request).
    """
    url = base_url.rstrip("/")
    if not url.endswith("/v1/systemone"):
        url = f"{url}/v1/systemone"

    payload: Dict[str, Any] = {"state": state, "questions": questions}
    if model:
        payload["model"] = model
    if max_len is not None:
        payload["max_len"] = max_len
    if head_max_len is not None:
        payload["head_max_len"] = head_max_len
    if lang is not None:
        payload["lang"] = lang
    if min_confidence is not None:
        payload["min_confidence"] = min_confidence
    if task is not None:
        payload["task"] = task
    if lang_guess is not None:
        payload["lang_guess"] = lang_guess

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(_SameOriginRedirectHandler())
    try:
        with opener.open(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Laya server error {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Laya server at {url}: {e}") from e
    # A remote laya-serve answers with a JSON object, but `json.loads` accepts any top-level value:
    # a gateway, a mis-pointed `base_url`, or a proxy error page can reply with a bare array, string,
    # number or `true`/`null`. Consumers read the result with `result.get("answers", …)`, so a
    # non-object leaks ``AttributeError: 'list' object has no attribute 'get'`` instead of the
    # ``RuntimeError`` every other remote failure here already raises. The check sits after the
    # ``try`` so it is not swallowed by the catch-all connect handler above.
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Malformed decision response from Laya server at {url}: "
            f"expected a JSON object, got {type(parsed).__name__}"
        )
    # `parsed` is an object now, but its `answers` field is the same arbitrary JSON: a gateway can
    # answer `{"answers": []}`. The consumers read it four ways -- `result["answers"][question_id]`,
    # `.get("answers", {}).get("intent", …)`, `.items()` and one that hands it back to the caller --
    # so a non-object leaks TypeError/AttributeError, or worse a non-mapping into user code, past the
    # fail-closed paths. This is the guard above one level deeper. A missing `answers` is left alone:
    # `LayaRouter` already reports that as its own KeyError, and the guardrail as no violations.
    if "answers" in parsed and not isinstance(parsed["answers"], dict):
        raise RuntimeError(
            f"Malformed decision response from Laya server at {url}: "
            f"'answers' must be a JSON object, got {type(parsed['answers']).__name__}"
        )
    return parsed


_DEFAULT_ROUTER = None
_DEFAULT_ROUTER_LOCK = threading.Lock()


def _get_default_router():
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        with _DEFAULT_ROUTER_LOCK:
            if _DEFAULT_ROUTER is None:
                from ..router import Router

                _DEFAULT_ROUTER = Router()
    return _DEFAULT_ROUTER


def _execute_decision(
    state: Any,
    questions: Dict[str, Any],
    agent: Optional[Any] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    lang: Optional[str] = None,
    min_confidence: Optional[float] = None,
    task: Optional[str] = None,
    lang_guess: Optional[Any] = None,
    hooks: Optional[Any] = None,
    on_predict_start: Optional[Any] = None,
    on_predict_end: Optional[Any] = None,
    hooks_raise: Optional[bool] = None,
    hooks_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    hook_kwargs = _hook_kwargs(hooks, on_predict_start, on_predict_end, hooks_raise, hooks_timeout)
    router_kwargs = _router_kwargs(task, lang_guess)
    if base_url:
        _reject_remote_hooks(hook_kwargs, base_url)
        _reject_remote_lang_guess(router_kwargs, base_url)
        budget = _budget_kwargs(max_len, head_max_len)
        decision = _decision_kwargs(lang, min_confidence)
        return _call_remote(
            base_url,
            state,
            questions,
            api_key=api_key,
            model=model,
            **budget,
            **decision,
            **router_kwargs,
        )
    runner = agent if agent is not None else _get_default_router()
    kwargs = _predict_kwargs(model, max_len, head_max_len, lang, min_confidence)
    kwargs.update(_require_router_controls(runner, "predict", router_kwargs))
    kwargs.update(hook_kwargs)
    return runner.predict(state, questions, **kwargs)


def _can_batch(
    agent: Optional[Any] = None, base_url: Optional[str] = None, hooked: bool = False
) -> bool:
    """Whether the path this runnable would take supports one batched forward call.

    False for a remote deployment (`laya-serve` answers one request per POST), for a
    caller-supplied runner that only implements `predict`, and for a `Router` when per-call
    hooks are configured: `Router.predict_batch` takes no per-call hooks, so the per-input
    loop is what keeps them running.
    """
    if base_url:
        return False
    runner = agent if agent is not None else _get_default_router()
    if hooked and hasattr(runner, "route_batch"):
        return False
    return getattr(runner, "predict_batch", None) is not None


def _execute_batch(
    states: Sequence[Any],
    questions: Dict[str, Any],
    agent: Optional[Any] = None,
    model: Optional[str] = None,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    lang: Optional[str] = None,
    min_confidence: Optional[float] = None,
    hook_kwargs: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
    lang_guess: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Evaluate one question set over many states, packing them into shared forward passes.

    The local sibling of `_execute_decision`; results come back in input order. `Agent`
    and `Router` disagree about how `predict_batch` is called (states plus one question
    set, versus one request dict each), so both forms are built here. The token budget,
    `lang` and `min_confidence` ride the same way they do on `_execute_decision`, so
    `.batch()` answers identically to `.invoke()` instead of silently dropping them --
    and so do the two routing hints, which a Router lifts off each request dict.
    """
    runner = agent if agent is not None else _get_default_router()
    router_kwargs = _router_kwargs(task, lang_guess)
    if hasattr(runner, "route_batch"):
        # Router: `lang` is a per-request control it lifts off each item, `min_confidence` is a
        # call-level argument -- the same split laya-serve's batch endpoint makes between
        # BATCH_BODY_ITEM_CONTROLS and BATCH_BODY_CALL_CONTROLS. `task` and `lang_guess` are
        # per-request as well, which is why they are checked against `route`, the signature
        # that reads them, rather than against `predict_batch`.
        item_overrides = _predict_kwargs(model, max_len, head_max_len, lang)
        item_overrides.update(_require_router_controls(runner, "route", router_kwargs))
        requests = [
            dict({"state": state, "questions": questions}, **item_overrides) for state in states
        ]
        return runner.predict_batch(requests, **_decision_kwargs(None, min_confidence))
    # Agent: `lang` and `min_confidence` are both call-level keyword arguments of `predict_batch`.
    overrides = _predict_kwargs(model, max_len, head_max_len, lang, min_confidence)
    overrides.update(_require_router_controls(runner, "predict_batch", router_kwargs))
    return runner.predict_batch(list(states), questions, **overrides, **(hook_kwargs or {}))


def _per_input_config(config: Any, n: int) -> List[Any]:
    """One config per input: LangChain hands `batch` a single config, a list of them, or None.

    `RunnableSequence` and `RunnableParallel` pass the list form, so anything that loops
    `invoke` itself has to expand it -- `invoke` expects one config, not a list of them.
    """
    if config is None:
        return [None] * n
    if isinstance(config, list):
        return list(config)
    return [config] * n


class _BatchedRunnable:
    """Gives a Laya runnable a real `batch()`, on Laya's shared forward passes.

    LangChain's default `batch` runs `invoke` once per input on a thread pool, which for
    a local Laya runner means N independent passes over the same questions -- and N
    threads contending for the same torch interpreter. `predict_batch` exists precisely
    to avoid that, so `chain.batch(...)`, `RunnableParallel` and LangGraph map-reduce
    nodes get it here. Outputs are identical to calling `invoke` per input, in order.
    """

    def _questions(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _finish(self, result: Dict[str, Any], input: Any) -> Any:
        raise NotImplementedError

    def batch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """Answer every input in one batched call, returning outputs in input order."""
        inputs = list(inputs)
        if not inputs:
            return []
        if return_exceptions:
            # A shared forward pass fails as a unit, so there is no per-input exception to
            # collect: honour the flag the way the per-input loop does.
            outcomes: List[Any] = []
            for item, one_config in zip(inputs, _per_input_config(config, len(inputs))):
                try:
                    outcomes.append(self.invoke(item, one_config, **kwargs))
                except Exception as exc:  # noqa: BLE001 - returned, per the Runnable contract
                    outcomes.append(exc)
            return outcomes
        hook_kwargs = _hook_kwargs(
            self.hooks,
            self.on_predict_start,
            self.on_predict_end,
            self.hooks_raise,
            self.hooks_timeout,
        )
        if not _can_batch(self.agent, self.base_url, hooked=bool(hook_kwargs)):
            if _RUNNABLE_AVAILABLE:
                # LangChain's own loop already understands both config shapes.
                return super().batch(inputs, config, **kwargs)  # type: ignore[misc]
            return [
                self.invoke(item, one_config, **kwargs)
                for item, one_config in zip(inputs, _per_input_config(config, len(inputs)))
            ]
        states = [_extract_text(item, self.state_key) for item in inputs]
        results = _execute_batch(
            states,
            self._questions(),
            agent=self.agent,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=getattr(self, "lang", None),
            min_confidence=getattr(self, "min_confidence", None),
            task=self.task,
            lang_guess=self.lang_guess,
            hook_kwargs=hook_kwargs,
        )
        return [self._finish(result, item) for result, item in zip(results, inputs)]

    async def abatch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """The async entry point, on the same batched call (`batch` is synchronous torch work)."""
        from langchain_core.runnables.config import run_in_executor

        # `batch` is blocking torch work, so it goes to the executor like any other sync
        # Runnable; the first config carries the executor settings (`RunnableSequence`
        # hands a step a list of per-input configs, not one).
        executor_config = _per_input_config(config, max(len(inputs), 1))[0]
        return await run_in_executor(
            executor_config,
            self.batch,
            inputs,
            config,
            return_exceptions=return_exceptions,
            **kwargs,
        )


class LayaRouter(_BatchedRunnable, RunnableSerializable):
    """Zero-latency LangGraph conditional edge and LangChain LCEL routing runnable.

    Evaluates user input against typed criteria in ~33 ms without token generation.
    Supports confidence threshold gating or an opt-in native auto/review policy,
    and answers a list of inputs in one shared forward pass through `batch`.
    """

    criteria: Dict[str, str]
    instructions: str = "Which route should handle this request?"
    confidence_threshold: float = 0.0
    decision_policy: str = "confidence"
    fallback: Optional[str] = None
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    lang: Optional[str] = None
    min_confidence: Optional[float] = None
    task: Optional[str] = None
    lang_guess: Optional[Any] = None
    hooks: Optional[Any] = None
    on_predict_start: Optional[Any] = None
    on_predict_end: Optional[Any] = None
    hooks_raise: Optional[bool] = None
    hooks_timeout: Optional[float] = None
    question_id: str = "route"
    last_decision: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        criteria: Dict[str, str],
        instructions: str = "Which route should handle this request?",
        confidence_threshold: float = 0.0,
        fallback: Optional[str] = None,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        task: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        *,
        decision_policy: str = "confidence",
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                criteria=criteria,
                instructions=instructions,
                confidence_threshold=confidence_threshold,
                decision_policy=decision_policy,
                fallback=fallback,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_len=max_len,
                head_max_len=head_max_len,
                lang=lang,
                min_confidence=min_confidence,
                task=task,
                lang_guess=lang_guess,
                hooks=hooks,
                on_predict_start=on_predict_start,
                on_predict_end=on_predict_end,
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
                **kwargs,
            )
        else:
            self.criteria = criteria
            self.instructions = instructions
            self.confidence_threshold = confidence_threshold
            self.decision_policy = decision_policy
            self.fallback = fallback
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.max_len = max_len
            self.head_max_len = head_max_len
            self.lang = lang
            self.min_confidence = min_confidence
            self.task = task
            self.lang_guess = lang_guess
            self.hooks = hooks
            self.on_predict_start = on_predict_start
            self.on_predict_end = on_predict_end
            self.hooks_raise = hooks_raise
            self.hooks_timeout = hooks_timeout
        self.question_id = "route"
        self.last_decision: Optional[Dict[str, Any]] = None
        self._validate_decision_policy()

    def _validate_decision_policy(self) -> None:
        if self.decision_policy not in ("confidence", "native"):
            raise ValueError("decision_policy must be 'confidence' or 'native'")
        if self.decision_policy == "native":
            if not isinstance(self.fallback, str) or not self.fallback.strip():
                raise ValueError("native decision_policy requires a non-empty fallback")
            if self.confidence_threshold != 0.0:
                raise ValueError(
                    "native decision_policy requires confidence_threshold=0; "
                    "configure the backend's min_confidence instead"
                )

    def _questions(self) -> Dict[str, Any]:
        # Fields are mutable (including via Pydantic copies); refuse invalid policy before inference.
        self._validate_decision_policy()
        return {
            self.question_id: {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": self.criteria,
            }
        }

    def _finish(self, result: Dict[str, Any], input: Any) -> str:
        self.last_decision = result
        self._validate_decision_policy()
        if self.decision_policy == "native":
            ans = _answers_map(result, "LayaRouter").get(self.question_id)
            if ans is None:
                raise _malformed("LayaRouter", "missing answer for %r" % self.question_id)
            route = ans.get("route")
            # Review need not carry a usable business choice. The backend has already
            # combined its signals; do not reconstruct them or consult shared last_decision.
            if route == "review":
                return self.fallback
            if route != "auto":
                raise _malformed("LayaRouter", "native route must be 'auto' or 'review'")
            choice = ans.get("choice")
            if not isinstance(choice, str) or choice not in self.criteria:
                raise _malformed(
                    "LayaRouter", "native auto choice must name a configured criterion"
                )
            return choice
        ans = result["answers"][self.question_id]
        choice = ans["choice"]
        # Gate on the same number core's `flag_low_confidence` gates on: `answer_confidence`
        # first, falling back to the entropy `confidence` so an answer that carries only the
        # older field is still gated rather than silently passed (fail-closed). An answer with
        # no usable number at all is treated as fully confident.
        confidence = _gate_confidence(ans)
        if confidence is None:
            confidence = 1.0

        if self.confidence_threshold > 0.0 and confidence < self.confidence_threshold:
            if self.fallback is not None:
                return self.fallback

        return choice

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        """Route input to a destination branch label."""
        text = _extract_text(input, self.state_key)
        res = _execute_decision(
            text,
            self._questions(),
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        return self._finish(res, input)

    def __call__(self, state: Any) -> str:
        """Callable protocol for direct use as a LangGraph conditional edge."""
        return self.invoke(state)


class LayaGuardrail(_BatchedRunnable, RunnableSerializable):
    """Sub-40ms inline guardrail for LangChain chains and LangGraph nodes.

    Screens for prompt injections, jailbreaks, sensitive data, or custom harm
    criteria before passing inputs downstream. A list of inputs is screened in one
    shared forward pass through `batch`.

    `threshold` is a violation probability in [0, 1] for every `noul` and `score`
    question: for a `noul` question it applies to `noul`, and for a `score`
    question to the probability that the level is at or above the middle of the
    scale (`serious` or `severe` for the default `harm_severity`; the middle level
    counts on an odd scale). Levels of a `score` question must run from harmless
    to worst.
    """

    questions: Optional[Dict[str, Any]] = None
    action: str = "raise"  # "raise", "filter", or "annotate"
    rejection_message: str = "I cannot fulfill this request because it violates safety guidelines."
    threshold: float = 0.5
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    lang: Optional[str] = None
    min_confidence: Optional[float] = None
    task: Optional[str] = None
    lang_guess: Optional[Any] = None
    hooks: Optional[Any] = None
    on_predict_start: Optional[Any] = None
    on_predict_end: Optional[Any] = None
    hooks_raise: Optional[bool] = None
    hooks_timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "raise",
        rejection_message: str = "I cannot fulfill this request because it violates safety guidelines.",
        threshold: float = 0.5,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        task: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a probability in [0, 1]; got %r" % (threshold,))
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                questions=questions,
                action=action,
                rejection_message=rejection_message,
                threshold=threshold,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_len=max_len,
                head_max_len=head_max_len,
                lang=lang,
                min_confidence=min_confidence,
                task=task,
                lang_guess=lang_guess,
                hooks=hooks,
                on_predict_start=on_predict_start,
                on_predict_end=on_predict_end,
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
                **kwargs,
            )
        else:
            self.questions = questions
            self.action = action
            self.rejection_message = rejection_message
            self.threshold = threshold
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.max_len = max_len
            self.head_max_len = head_max_len
            self.lang = lang
            self.min_confidence = min_confidence
            self.task = task
            self.lang_guess = lang_guess
            self.hooks = hooks
            self.on_predict_start = on_predict_start
            self.on_predict_end = on_predict_end
            self.hooks_raise = hooks_raise
            self.hooks_timeout = hooks_timeout

    def _default_questions(self) -> Dict[str, Any]:
        from ..presets import guard_questions

        return guard_questions()

    def _questions(self) -> Dict[str, Any]:
        return self.questions if self.questions is not None else self._default_questions()

    def _finish(self, result: Dict[str, Any], input: Any) -> Any:
        """Apply the violation scan and the configured action to one decision."""
        # A remote body is arbitrary JSON, so `answers` and each answer are checked before anything
        # is read out of them -- the same two levels and the same `noul` check `LayaTaskGuard.screen`
        # and `LayaAgentRouter.screen` already delegate to `_guard`. `result.get("answers", {})` only
        # defaults when the key is ABSENT: a body whose answer is a string (`{"jailbreak": "nope"}`)
        # raised AttributeError at `ans.get("type")`, a `noul` of `"high"` or `null` raised TypeError
        # from `>=`, and a `{"type": "noul"}` with no `noul` at all raised KeyError at `ans["noul"]`
        # whenever the threshold let the comparison through.
        where = f"Laya server at {self.base_url}" if self.base_url else "Laya"
        answers = _answers_map(result, where)

        violations: Dict[str, Any] = {}
        for qid, ans in answers.items():
            t = ans.get("type")
            if t == "noul":
                probability = _answer_number(ans, "noul", where)
                if probability >= self.threshold:
                    violations[qid] = {
                        "probability": probability,
                        "confidence": ans.get("confidence", 0.0),
                    }
            elif t == "score":
                # `score` is the expected level (0..k-1), not a probability: gate on the
                # probability that the level is at or above the middle of the scale.
                levels = len(self._questions().get(qid, {}).get("criteria") or [])
                p_violation = _score_violation_probability(ans, levels)
                if p_violation >= self.threshold:
                    violations[qid] = {
                        "score": ans.get("score", 0.0),
                        "probability": round(p_violation, 4),
                        "confidence": ans.get("confidence", 0.0),
                    }

        is_safe = len(violations) == 0

        if not is_safe and self.action == "raise":
            raise LayaGuardrailError(
                f"Laya guardrail policy violation detected: {list(violations.keys())}",
                violations=violations,
                raw_decision=result,
            )

        if not is_safe and self.action == "filter":
            if isinstance(input, dict):
                filtered = dict(input)
                filtered["output"] = self.rejection_message
                return filtered
            return self.rejection_message

        if self.action == "annotate":
            if isinstance(input, dict):
                annotated = dict(input)
                annotated["guardrails"] = {
                    "passed": is_safe,
                    "violations": violations,
                    "answers": answers,
                }
                return annotated
            return {
                "input": input,
                "guardrails": {
                    "passed": is_safe,
                    "violations": violations,
                    "answers": answers,
                },
            }

        return input

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Screen input against guardrail questions."""
        text = _extract_text(input, self.state_key)

        res = _execute_decision(
            text,
            self._questions(),
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        return self._finish(res, input)

    def __call__(self, state: Any) -> Any:
        return self.invoke(state)


class LayaTriage(_BatchedRunnable, RunnableSerializable):
    """Customer support ticket and incoming message triage node for LangGraph.

    Analyzes intent, urgency, customer frustration, and churn risk in one single
    forward pass and enriches the graph state dictionary. A backlog is triaged in one
    shared forward pass through `batch`.
    """

    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    lang: Optional[str] = None
    min_confidence: Optional[float] = None
    task: Optional[str] = None
    lang_guess: Optional[Any] = None
    hooks: Optional[Any] = None
    on_predict_start: Optional[Any] = None
    on_predict_end: Optional[Any] = None
    hooks_raise: Optional[bool] = None
    hooks_timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        task: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_len=max_len,
                head_max_len=head_max_len,
                lang=lang,
                min_confidence=min_confidence,
                task=task,
                lang_guess=lang_guess,
                hooks=hooks,
                on_predict_start=on_predict_start,
                on_predict_end=on_predict_end,
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
                **kwargs,
            )
        else:
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.max_len = max_len
            self.head_max_len = head_max_len
            self.lang = lang
            self.min_confidence = min_confidence
            self.task = task
            self.lang_guess = lang_guess
            self.hooks = hooks
            self.on_predict_start = on_predict_start
            self.on_predict_end = on_predict_end
            self.hooks_raise = hooks_raise
            self.hooks_timeout = hooks_timeout

    def _questions(self) -> Dict[str, Any]:
        from ..presets import triage_questions

        return triage_questions()

    def _finish(self, result: Dict[str, Any], state: Any) -> Dict[str, Any]:
        """Project one triage decision onto the graph state."""
        # Three levels of the same body are read here, and only the first was checked: `answers`
        # itself (a list raised AttributeError at `.get`), the answer under each question id (a
        # string raised AttributeError at the nested `.get`), and the `noul` inside it (a `"0.9"` or
        # `null` raised TypeError from `>=`, a `true` read as 1.0 because bool subclasses int, and a
        # NaN compared False -- i.e. silently "not urgent"). `_answers_map` validates the first two
        # and `_answer_number` the third, the way the guardrail screens two lines above do.
        where = f"Laya server at {self.base_url}" if self.base_url else "Laya"
        ans = _answers_map(result, where)

        def _noul(qid: str) -> float:
            return _answer_number(ans.get(qid, {}), "noul", where)

        triage_info = {
            "intent": ans.get("intent", {}).get("choice"),
            "intent_confidence": ans.get("intent", {}).get("confidence"),
            "is_urgent": _noul("is_urgent") >= 0.5,
            "frustration_score": ans.get("frustration", {}).get("score"),
            "churn_risk": _noul("churn_risk") >= 0.5,
            "refund_requested": _noul("refund_requested") >= 0.5,
        }

        if isinstance(state, dict):
            updated = dict(state)
            updated["triage"] = triage_info
            return updated

        return {"input": state, "triage": triage_info}

    def invoke(self, state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Triage the state and return enriched fields."""
        text = _extract_text(state, self.state_key)
        res = _execute_decision(
            text,
            self._questions(),
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        return self._finish(res, state)

    def __call__(self, state: Any) -> Dict[str, Any]:
        return self.invoke(state)


class LayaEvaluator(_BatchedRunnable, RunnableSerializable):
    """Rubric-based output grading and hallucination evaluation for LangChain.

    Evaluates LLM responses against criteria without generating text. A list of
    responses is graded in one shared forward pass through `batch`.
    """

    questions: Dict[str, Any]
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    lang: Optional[str] = None
    min_confidence: Optional[float] = None
    task: Optional[str] = None
    lang_guess: Optional[Any] = None
    hooks: Optional[Any] = None
    on_predict_start: Optional[Any] = None
    on_predict_end: Optional[Any] = None
    hooks_raise: Optional[bool] = None
    hooks_timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        questions: Dict[str, Any],
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        task: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                questions=questions,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_len=max_len,
                head_max_len=head_max_len,
                lang=lang,
                min_confidence=min_confidence,
                task=task,
                lang_guess=lang_guess,
                hooks=hooks,
                on_predict_start=on_predict_start,
                on_predict_end=on_predict_end,
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
                **kwargs,
            )
        else:
            self.questions = questions
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.max_len = max_len
            self.head_max_len = head_max_len
            self.lang = lang
            self.min_confidence = min_confidence
            self.task = task
            self.lang_guess = lang_guess
            self.hooks = hooks
            self.on_predict_start = on_predict_start
            self.on_predict_end = on_predict_end
            self.hooks_raise = hooks_raise
            self.hooks_timeout = hooks_timeout

    def evaluate_strings(
        self, *, prediction: str, input: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """LangChain standard string evaluation interface."""
        state = {"input": input, "prediction": prediction} if input else prediction
        res = _execute_decision(
            state,
            self.questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        return res.get("answers", {})

    def _questions(self) -> Dict[str, Any]:
        return self.questions

    def _finish(self, result: Dict[str, Any], input: Any) -> Dict[str, Any]:
        return result.get("answers", {})

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        text = _extract_text(input, self.state_key)
        res = _execute_decision(
            text,
            self._questions(),
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        return self._finish(res, input)

    def __call__(self, state: Any) -> Dict[str, Any]:
        return self.invoke(state)


def _validate_decision_schema(schema: Any) -> None:
    """Plan ``schema`` into Laya questions and keep only the errors it raises."""
    from ..structured import questions_from_json_schema, questions_from_pydantic

    if hasattr(schema, "model_json_schema") or hasattr(schema, "schema"):
        questions_from_pydantic(schema)
    else:
        questions_from_json_schema(schema)


class _RemoteDecisionRunner:
    """Present a remote ``laya-serve`` endpoint with the ``predict`` API ``laya.decide`` drives.

    The HTTP endpoint answers a question set exactly as a local runner does, so wrapping it lets
    remote mode share core's schema projection instead of reimplementing it here.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    def predict(self, state: Any, questions: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
        # `overrides` carries only the controls that were actually set -- `LayaDecision.invoke`
        # builds it with the shared omit-unset helpers -- so an unset budget never reaches
        # `_call_remote` and never shadows what the endpoint was started with.
        return _call_remote(self.base_url, state, questions, api_key=self.api_key, **overrides)


class LayaDecision(RunnableSerializable):
    """Schema-driven decision node: a JSON schema or pydantic model in, schema-shaped values out.

    This is the LCEL form of `laya.decide`, so a chain or graph node gets a typed decision -- an
    enum choice, an integer level, a boolean -- from one forward pass, without writing the Laya
    questions by hand and without a structured-output parser downstream.

    The schema is planned at construction: a property Laya cannot answer from a fixed option set
    (a free string, an array, a nested object) raises `SchemaError` here rather than on the first
    request, after the chain has already paid for every earlier step.
    """

    decision_schema: Any
    return_details: bool = False
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    task: Optional[str] = None
    lang_guess: Optional[Any] = None
    hooks: Optional[Any] = None
    on_predict_start: Optional[Any] = None
    on_predict_end: Optional[Any] = None
    hooks_raise: Optional[bool] = None
    hooks_timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        decision_schema: Any,
        return_details: bool = False,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        task: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        _validate_decision_schema(decision_schema)
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                decision_schema=decision_schema,
                return_details=return_details,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_len=max_len,
                head_max_len=head_max_len,
                task=task,
                lang_guess=lang_guess,
                hooks=hooks,
                on_predict_start=on_predict_start,
                on_predict_end=on_predict_end,
                hooks_raise=hooks_raise,
                hooks_timeout=hooks_timeout,
                **kwargs,
            )
        else:
            self.decision_schema = decision_schema
            self.return_details = return_details
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.max_len = max_len
            self.head_max_len = head_max_len
            self.task = task
            self.lang_guess = lang_guess
            self.hooks = hooks
            self.on_predict_start = on_predict_start
            self.on_predict_end = on_predict_end
            self.hooks_raise = hooks_raise
            self.hooks_timeout = hooks_timeout

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Decide ``input`` against the schema and return its values, or a ``DecisionResult``."""
        from ..structured import decide

        text = _extract_text(input, self.state_key)
        hook_kwargs = _hook_kwargs(
            self.hooks,
            self.on_predict_start,
            self.on_predict_end,
            self.hooks_raise,
            self.hooks_timeout,
        )
        overrides = _predict_kwargs(self.model, self.max_len, self.head_max_len)
        router_kwargs = _router_kwargs(self.task, self.lang_guess)
        if self.base_url:
            _reject_remote_hooks(hook_kwargs, self.base_url)
            _reject_remote_lang_guess(router_kwargs, self.base_url)
            runner: Any = _RemoteDecisionRunner(self.base_url, self.api_key)
            overrides.update(router_kwargs)
        else:
            runner = self.agent if self.agent is not None else _get_default_router()
            overrides.update(_require_router_controls(runner, "predict", router_kwargs))
            overrides.update(hook_kwargs)
        return decide(
            runner,
            text,
            schema=self.decision_schema,
            return_details=self.return_details,
            **overrides,
        )

    def __call__(self, state: Any) -> Any:
        return self.invoke(state)
