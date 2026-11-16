# ruff: noqa: E402
"""CrewAI integration for Laya System 1 decision engine.

Provides sub-35ms, non-autoregressive task delegation routing and real-time task
guardrails for CrewAI multi-agent crews.

Replaces slow, token-generating LLM managers with calibrated, typed decisions
executed in a single forward pass without token generation costs.

Supports both local in-process models (`Agent` / `Router`) and remote HTTP
deployments (your own `laya-serve`) without requiring PyTorch on edge clients.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

# Optional CrewAI base classes
try:
    from crewai import Agent as CrewAgent
    from crewai import Task as CrewTask

    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

    @dataclass
    class CrewAgent:  # type: ignore
        """Lightweight shim when crewai is not installed."""

        role: str
        goal: str = ""
        backstory: str = ""

    @dataclass
    class CrewTask:  # type: ignore
        """Lightweight shim when crewai is not installed."""

        description: str
        expected_output: str = ""
        agent: Optional[Any] = None


class LayaTaskGuardError(ValueError):
    """Raised when a task or instruction violates a Laya task guard policy."""

    def __init__(self, message: str, violations: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.violations = violations
        self.raw_decision = raw_decision


# The per-call control rules are shared with the other two integrations: they end at the same
# `predict` call and the same laya-serve request body, and three copies of the rule is how two of
# them came to forward only `model`.
from ..confidence import _gate_confidence  # noqa: E402
from ._controls import budget_kwargs as _budget_kwargs  # noqa: E402
from ._controls import decision_kwargs as _decision_kwargs  # noqa: E402
from ._controls import hook_kwargs as _hook_kwargs
from ._controls import predict_kwargs as _predict_kwargs  # noqa: E402
from ._controls import reject_remote_hooks as _reject_remote_hooks
from ._controls import (
    reject_remote_lang_guess as _reject_remote_lang_guess,  # noqa: E402
)
from ._controls import require_router_controls as _require_router_controls  # noqa: E402
from ._controls import router_kwargs as _router_kwargs  # noqa: E402

# One class for every integration, so `except LayaLowConfidenceError` catches all of them.
from ._errors import LayaLowConfidenceError  # noqa: E402
from ._guard import answer_number as _answer_number  # noqa: E402
from ._guard import answers_map as _answers_map  # noqa: E402
from ._guard import malformed as _malformed  # noqa: E402
from ._guard import score_violation_probability as _score_violation_probability  # noqa: E402


@dataclass
class CrewRouteDecision:
    """Result of a Laya sub-35ms task delegation decision."""

    agent: Any
    agent_index: int
    role: str
    confidence: float
    reason: str
    raw_decision: Dict[str, Any]


def _extract_task_str(task: Union[str, CrewTask, Dict[str, Any], Any]) -> str:
    """Extract descriptive text from a CrewAI task or string."""
    if isinstance(task, str):
        return task
    if hasattr(task, "description"):
        desc = str(task.description)
        expected = getattr(task, "expected_output", "")
        if expected:
            return f"{desc} (Expected output: {expected})"
        return desc
    if isinstance(task, dict):
        desc = task.get("description", task.get("task", task.get("prompt", str(task))))
        expected = task.get("expected_output", "")
        if expected:
            return f"{desc} (Expected output: {expected})"
        return str(desc)
    return str(task)


def _get_agent_role(agent_obj: Any, default_idx: int) -> str:
    """Extract role string from agent object, dict, or string."""
    if isinstance(agent_obj, str):
        return agent_obj
    if isinstance(agent_obj, dict):
        return str(agent_obj.get("role") or f"Agent {default_idx}")
    return str(getattr(agent_obj, "role", None) or f"Agent {default_idx}")


def _format_agent_criteria(
    agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
) -> Dict[str, str]:
    """Format candidate CrewAI agents into Laya choice criteria."""
    criteria: Dict[str, str] = {}
    for i, agent in enumerate(agents):
        key = f"agent_{i}"
        role = _get_agent_role(agent, i)
        goal = getattr(agent, "goal", None)

        if isinstance(agent, dict):
            goal = agent.get("goal", goal)

        if goal:
            criteria[key] = f"{role}: {goal}"
        else:
            criteria[key] = str(role)

    return criteria


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
    is routed and answered in, and core's abstention gate. `task` names the checkpoint that answers
    and `lang_guess` is the language code routing reads before its own detection -- both are
    laya-serve body controls too, and only `lang_guess`'s string form has a wire representation
    (`reject_remote_lang_guess` refuses the callable before the request).
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
    # number or `true`/`null`. Consumers read the result with `res.get("answers", …)` (the delegation
    # lookup and the guardrail screen), so a non-object leaks ``AttributeError: 'list' object has no
    # attribute 'get'`` instead of the ``RuntimeError`` every other remote failure here already
    # raises. The check sits after the ``try`` so it is not swallowed by the catch-all above.
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Malformed decision response from Laya server at {url}: "
            f"expected a JSON object, got {type(parsed).__name__}"
        )
    # `parsed` is an object now, but its `answers` field is the same arbitrary JSON: a gateway can
    # answer `{"answers": []}`. Both consumers read it as `res.get("answers", {})` and then call
    # `.get("delegation", …)` / `.items()` on it, so a non-object there leaks AttributeError past the
    # same fail-closed paths -- this is the guard above one level deeper. A missing `answers` is left
    # alone: that is the no-decision case the consumers already handle, not a malformed one.
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
    """One decision on the local runner or a laya-serve node, with the per-call controls kept.

    Which controls either path accepts is `._controls`' rule, shared with the LangChain and
    LlamaIndex wrappers -- they end at the same `predict` call and the same request body.
    """
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


class LayaCrewRouter:
    """Sub-35ms task delegation router for CrewAI multi-agent crews.

    Evaluates tasks against crew agent roles and goals in a single non-autoregressive
    forward pass, eliminating 2-4 second LLM manager delegation latency.
    """

    def __init__(
        self,
        instructions: str = "Which agent in the crew is best qualified to execute this task?",
        confidence_threshold: float = 0.0,
        fallback_agent_index: Optional[int] = None,
        raise_on_low_confidence: bool = False,
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
    ):
        self.instructions = instructions
        self.confidence_threshold = confidence_threshold
        self.fallback_agent_index = fallback_agent_index
        self.raise_on_low_confidence = raise_on_low_confidence
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
        self.last_decision: Optional[Dict[str, Any]] = None

    def route(
        self,
        task: Union[str, CrewTask, Dict[str, Any], Any],
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> CrewRouteDecision:
        """Evaluate task and route to the best matching crew agent in ~33ms."""
        if not agents:
            raise ValueError("No agents provided to route task to.")

        task_str = _extract_task_str(task)
        criteria = _format_agent_criteria(agents)

        questions = {
            "delegation": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        }

        res = _execute_decision(
            task_str,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            # `task` here is core's checkpoint selector, not the CrewAI task being decided on --
            # `task_str` above is that one. Both are on this call, and only one routes.
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        self.last_decision = res

        # The same nested-shape checks as the guardrail screen: `answers` must be an object,
        # the answer must be an object, and `choice` is used with `.startswith` below -- a list
        # there raised AttributeError out of a routing decision.
        where = f"Laya server at {self.base_url}" if self.base_url else "Laya"
        ans = _answers_map(res, where).get("delegation", {})
        chosen_key = ans.get("choice")
        if chosen_key is not None and not isinstance(chosen_key, str):
            raise _malformed(where, "'choice' must be a string, got %s" % type(chosen_key).__name__)
        # Gate on the same number core's `flag_low_confidence` gates on: `answer_confidence`
        # first, falling back to the entropy `confidence` so an answer that carries only the
        # older field is still gated rather than silently passed (fail-closed). An answer with
        # no usable number at all is treated as fully confident.
        conf = _gate_confidence(ans)
        if conf is None:
            conf = 1.0

        # Map agent_i back to integer index i
        chosen_idx: int = 0
        if chosen_key and chosen_key.startswith("agent_"):
            try:
                chosen_idx = int(chosen_key.split("_")[1])
            except (ValueError, IndexError):
                chosen_idx = 0
        elif chosen_key in criteria:
            chosen_idx = list(criteria.keys()).index(chosen_key)

        # `chosen_key` is remote JSON and `agents` is the caller's list, so a two-agent crew can be
        # told to delegate to `agent_9`: the parsed number indexes no agent and `agents[chosen_idx]`
        # below raised a bare IndexError out of the routing decision, while `agent_-1` silently
        # wrapped to the LAST agent. `criteria` is built from exactly these agents, so an index it
        # does not name is a candidate that does not exist -- the same "unknown decision" that
        # `tools.py:493` refuses by name rather than raising.
        if not 0 <= chosen_idx < len(agents):
            raise _malformed(
                where,
                "decision %r names agent %d, which is not in the %d-agent crew"
                % (chosen_key, chosen_idx, len(agents)),
            )

        # Confidence gating
        if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
            if self.fallback_agent_index is not None:
                fallback_obj = agents[self.fallback_agent_index]
                fallback_role = _get_agent_role(fallback_obj, self.fallback_agent_index)
                reason = (
                    f"Delegated to fallback agent '{fallback_role}' because routing confidence "
                    f"({conf:.3f}) was below threshold ({self.confidence_threshold:.3f})."
                )
                return CrewRouteDecision(
                    agent=fallback_obj,
                    agent_index=self.fallback_agent_index,
                    role=fallback_role,
                    confidence=conf,
                    reason=reason,
                    raw_decision=res,
                )
            if self.raise_on_low_confidence:
                raise LayaLowConfidenceError(
                    f"Routing confidence {conf:.3f} below threshold {self.confidence_threshold:.3f} "
                    f"for task: {task_str!r}",
                    confidence=conf,
                    threshold=self.confidence_threshold,
                    raw_decision=res,
                )

        selected_agent = agents[chosen_idx]
        role_name = _get_agent_role(selected_agent, chosen_idx)
        reason = f"Delegated to '{role_name}' via Laya System 1 decision (confidence: {conf:.3f})."

        return CrewRouteDecision(
            agent=selected_agent,
            agent_index=chosen_idx,
            role=role_name,
            confidence=conf,
            reason=reason,
            raw_decision=res,
        )

    def delegate(
        self,
        task: Any,
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> Any:
        """Assign task.agent to the winning agent and return the selected agent."""
        decision = self.route(task, agents)
        if hasattr(task, "agent"):
            try:
                task.agent = decision.agent
            except Exception:
                pass
        return decision.agent

    async def aroute(
        self,
        task: Union[str, CrewTask, Dict[str, Any], Any],
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> CrewRouteDecision:
        """Asynchronously route task without blocking the asyncio event loop."""
        return await asyncio.to_thread(self.route, task, agents)

    async def adelegate(
        self,
        task: Any,
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> Any:
        """Asynchronously delegate task to the winning agent."""
        decision = await self.aroute(task, agents)
        if hasattr(task, "agent"):
            try:
                task.agent = decision.agent
            except Exception:
                pass
        return decision.agent


class LayaTaskGuard:
    """Sub-40ms inline guardrail for CrewAI tasks and agent inputs.

    Screens incoming user instructions and task specifications for jailbreaks,
    prompt injections, and policy violations before agents execute tools.
    """

    def __init__(
        self,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "raise",  # "raise", "filter", or "annotate"
        rejection_message: str = "This task cannot be executed because it violates safety guidelines.",
        threshold: float = 0.5,
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
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a probability in [0, 1]; got %r" % (threshold,))
        self.questions = questions
        self.action = action
        self.rejection_message = rejection_message
        self.threshold = threshold
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
        self.last_decision: Optional[Dict[str, Any]] = None

    def _default_questions(self) -> Dict[str, Any]:
        from ..presets import guard_questions

        return guard_questions()

    def screen(self, task: Union[str, CrewTask, Dict[str, Any], Any]) -> Any:
        """Screen task against safety policies."""
        task_str = _extract_task_str(task)
        qdefs = self.questions if self.questions is not None else self._default_questions()

        res = _execute_decision(
            task_str,
            qdefs,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            # `task` here is core's checkpoint selector, not the CrewAI task being decided on --
            # `task_str` above is that one. Both are on this call, and only one routes.
            task=self.task,
            lang_guess=self.lang_guess,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        self.last_decision = res
        # A remote body is arbitrary JSON, so `answers` and each answer are checked before anything
        # is read out of them: `res.get("answers", {})` only defaults when the key is absent, and a
        # present-but-non-object value used to raise AttributeError/TypeError out of the one path
        # whose job is to fail closed. A `noul` value is checked too (`>=` against a string or null
        # is a TypeError, a bool reads as 1.0/0.0, NaN compares False i.e. "safe").
        where = f"Laya server at {self.base_url}" if self.base_url else "Laya"
        answers = _answers_map(res, where)

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
                levels = len(qdefs.get(qid, {}).get("criteria") or [])
                p_violation = _score_violation_probability(ans, levels)
                if p_violation >= self.threshold:
                    violations[qid] = {
                        "score": ans.get("score", 0.0),
                        "probability": round(p_violation, 4),
                        "confidence": ans.get("confidence", 0.0),
                    }

        is_safe = len(violations) == 0

        if not is_safe and self.action == "raise":
            raise LayaTaskGuardError(
                f"Laya task guardrail policy violation detected: {list(violations.keys())}",
                violations=violations,
                raw_decision=res,
            )

        if not is_safe and self.action == "filter":
            if hasattr(task, "description"):
                task.description = self.rejection_message
                return task
            if isinstance(task, dict):
                filtered = dict(task)
                filtered["description"] = self.rejection_message
                return filtered
            return self.rejection_message

        if self.action == "annotate":
            guard_meta = {
                "passed": is_safe,
                "violations": violations,
                "answers": answers,
            }
            if hasattr(task, "description"):
                setattr(task, "guardrail", guard_meta)
                return task
            if isinstance(task, dict):
                annotated = dict(task)
                annotated["guardrail"] = guard_meta
                return annotated
            return {
                "task": task,
                "guardrail": guard_meta,
            }

        return task

    def __call__(self, task: Any) -> Any:
        return self.screen(task)

    async def ascreen(self, task: Union[str, CrewTask, Dict[str, Any], Any]) -> Any:
        """Asynchronously screen task without blocking the event loop."""
        return await asyncio.to_thread(self.screen, task)
