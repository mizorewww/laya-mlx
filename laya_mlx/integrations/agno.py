"""Agno integration for Laya System 1 decision engine.

Provides sub-35ms, non-autoregressive agent routing, real-time message guardrails,
and workflow step selection for Agno agent teams.

Replaces slow, token-generating Team leader routing (``mode="route"``) with
calibrated, typed decisions executed in a single forward pass without token cost.

Supports both local in-process models (``Agent`` / ``Router``) and remote HTTP
deployments (your own ``laya-serve``) without requiring PyTorch on edge clients.

Typical usage::

    from agno.agent import Agent as AgnoAgent
    from agno.team import Team
    from laya_mlx.integrations.agno import LayaAgnoRouter

    researcher = AgnoAgent(name="Researcher", role="Deep research and fact-finding")
    writer = AgnoAgent(name="Writer", role="Drafting clear prose")
    coder = AgnoAgent(name="Coder", role="Writing and reviewing code")

    router = LayaAgnoRouter()
    decision = router.route("Write a Python function to reverse a string", [researcher, writer, coder])
    print(decision.name, decision.confidence)
    # → Coder  0.94

    # Or pass a Team directly:
    team = Team(members=[researcher, writer, coder], mode="route")
    decision = router.route_team("Summarise the latest IPCC report", team)
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Optional Agno base classes – graceful shims when the package is not present.
# This mirrors the try/except pattern used in crewai.py and llamaindex.py so
# that `import laya_mlx.integrations.agno` never fails at import time; callers
# learn about the missing dependency only when they construct a live object.
# ---------------------------------------------------------------------------
try:
    from agno.agent import Agent as AgnoAgent
    from agno.team import Team as AgnoTeam

    _AGNO_AVAILABLE = True
except ImportError:
    _AGNO_AVAILABLE = False

    @dataclass
    class AgnoAgent:  # type: ignore[no-redef]
        """Lightweight shim when ``agno`` is not installed."""

        name: str
        role: str = ""
        description: str = ""
        instructions: str = ""

    @dataclass
    class AgnoTeam:  # type: ignore[no-redef]
        """Lightweight shim when ``agno`` is not installed.

        Uses ``members`` to match the Agno ``Team`` contract
        (``Team(members=[...], mode="route")``).  The legacy field
        name ``agents`` is accepted in :meth:`route_team` for compatibility
        with older hand-crafted dicts or mocked objects.
        """

        members: List[Any] = None  # type: ignore[assignment]
        mode: str = "route"

        def __post_init__(self):
            if self.members is None:
                self.members = []


# ---------------------------------------------------------------------------
# Shared per-call control infrastructure (same as crewai.py / langchain.py /
# llamaindex.py).  The rule for which controls exist lives in one place so
# that adding a control in `._controls` without updating a wrapper fails in
# that wrapper's test suite rather than silently disappearing.
# ---------------------------------------------------------------------------
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

# One exception class per integration so `except LayaLowConfidenceError` works
# even when only one integration is imported.
from ._errors import LayaLowConfidenceError  # noqa: E402
from ._guard import answer_number as _answer_number  # noqa: E402
from ._guard import answers_map as _answers_map  # noqa: E402
from ._guard import malformed as _malformed  # noqa: E402
from ._guard import score_violation_probability as _score_violation_probability  # noqa: E402

# ---------------------------------------------------------------------------
# Public error
# ---------------------------------------------------------------------------


class LayaAgnoGuardrailError(ValueError):
    """Raised when an Agno agent message violates a Laya guardrail policy."""

    def __init__(self, message: str, violations: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.violations = violations
        self.raw_decision = raw_decision


# ---------------------------------------------------------------------------
# Typed result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgnoRouteDecision:
    """Result of a Laya sub-35ms Agno agent routing decision.

    Attributes:
        agent: The winning ``AgnoAgent`` object (or dict / string as supplied).
        agent_index: Zero-based position in the ``agents`` sequence.
        name: Display name extracted from the agent (``agent.name``).
        role: Role string extracted from the agent (``agent.role``).
        confidence: Calibrated probability of the chosen option being correct.
        reason: Human-readable one-line explanation appended by this wrapper.
        raw_decision: Unmodified dict returned by ``runner.predict`` or the
            laya-serve HTTP response.
    """

    agent: Any
    agent_index: int
    name: str
    role: str
    confidence: float
    reason: str
    raw_decision: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _extract_message_str(message: Any) -> str:
    """Return a plain string from an Agno message, RunResponse, str, or dict.

    Agno's ``Team.run()`` / ``Agent.run()`` can return a ``RunResponse`` whose
    string content sits in ``.content``.  When a caller passes a raw string or a
    dict with ``"content"`` / ``"message"`` keys we extract those too, so the
    router can be dropped into any point in an Agno pipeline.
    """
    if isinstance(message, str):
        return message
    # agno.models.response.RunResponse and similar objects carry .content
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        # content may itself be a list of content blocks (multi-modal)
        if isinstance(content, list):
            parts = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ]
            joined = " ".join(p for p in parts if p).strip()
            # Return the joined text even when empty; only fall through to str(message)
            # when content is not a list at all.
            return joined
        return str(content)
    if isinstance(message, dict):
        for key in ("content", "message", "text", "input", "query"):
            if key in message:
                return str(message[key])
        return str(message)
    return str(message)


def _get_agent_name(agent: Any, idx: int) -> str:
    """Extract a display name from an AgnoAgent, dict, or string."""
    if isinstance(agent, str):
        return agent
    if isinstance(agent, dict):
        return str(agent.get("name") or agent.get("role") or f"Agent {idx}")
    return str(getattr(agent, "name", None) or getattr(agent, "role", None) or f"Agent {idx}")


def _get_agent_role(agent: Any, idx: int) -> str:
    """Extract a role / description string from an AgnoAgent, dict, or string."""
    if isinstance(agent, str):
        return agent
    if isinstance(agent, dict):
        return str(agent.get("role") or agent.get("description") or f"Agent {idx}")
    role = getattr(agent, "role", None)
    if role:
        return str(role)
    desc = getattr(agent, "description", None)
    if desc:
        return str(desc)
    return f"Agent {idx}"


def _format_agno_criteria(
    agents: Sequence[Union[AgnoAgent, Dict[str, Any], str, Any]],
) -> Dict[str, str]:
    """Build the Laya ``criteria`` dict from a sequence of Agno agents.

    Each agent becomes one choice in the decision question:
    ``{"agent_0": "Name (Role): description", "agent_1": ...}``.

    This mirrors ``_format_agent_criteria`` in ``crewai.py``.
    """
    criteria: Dict[str, str] = {}
    for i, agent in enumerate(agents):
        key = f"agent_{i}"
        name = _get_agent_name(agent, i)
        role = _get_agent_role(agent, i)

        # Pull the richer description / instructions fields when available.
        description = ""
        if isinstance(agent, dict):
            description = str(agent.get("description") or agent.get("instructions") or "")
        else:
            description = str(
                getattr(agent, "description", None) or getattr(agent, "instructions", None) or ""
            )

        # Avoid duplicating text that is already captured in role.
        # When _get_agent_role returned description (because role was empty),
        # role == description, so we should not append description again.
        role_is_description = role == description

        # Build a concise but informative criteria string.
        if description and not role_is_description and role and role != name:
            criteria[key] = f"{name} ({role}): {description}"
        elif description and not role_is_description:
            criteria[key] = f"{name}: {description}"
        elif description and role_is_description and role != name:
            # role came from description; use it as the parenthetical (no redundant colon+desc)
            criteria[key] = f"{name} ({role})"
        elif description and role_is_description:
            # name == role == description edge case
            criteria[key] = name
        elif role and role != name:
            criteria[key] = f"{name} ({role})"
        else:
            criteria[key] = name

    return criteria


# ---------------------------------------------------------------------------
# HTTP helper (cross-origin redirect guard + urllib POST)
# Identical in structure to the one in crewai.py – they call the same endpoint.
# ---------------------------------------------------------------------------


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
) -> Dict[str, Any]:
    """Send a decision request to a remote laya-serve HTTP instance via urllib.

    ``max_len`` / ``head_max_len`` travel in the body; laya-serve applies them
    up to its ``LAYA_MAX_TOKEN_BUDGET`` ceiling.  ``lang`` / ``min_confidence``
    are laya-serve ``BODY_CONTROLS`` too and ride in the same body.
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
    # number or `true`/`null`. Both consumers read the result with `res.get("answers", …)` (route's
    # delegation lookup and the guardrail screen), so a non-object leaks ``AttributeError: 'list'
    # object has no attribute 'get'`` past the fail-closed path instead of the ``RuntimeError`` every
    # other remote failure here already raises. Reject the shape where it is parsed. The check sits
    # after the ``try`` so it is not swallowed by the catch-all connect handler above.
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


# ---------------------------------------------------------------------------
# Default local router (lazily loaded, thread-safe)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared decision executor
# ---------------------------------------------------------------------------


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
    """Run one decision on the local runner or a laya-serve node.

    Which controls either path accepts is ``._controls``' rule, shared with the
    LangChain, CrewAI and LlamaIndex wrappers – they all end at the same
    ``predict`` call and the same request body.
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


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


class LayaAgnoRouter:
    """Sub-35ms agent router for Agno ``Team`` (``mode="route"``) workflows.

    Evaluates messages against Agno agent descriptions in a single
    non-autoregressive forward pass, eliminating the 2–4 second LLM manager
    delegation latency that Agno's built-in ``Team`` routing incurs.

    Supports both a list of agents passed at call time (stateless) and a fixed
    list provided at construction (stateful), as well as the convenience method
    :meth:`route_team` which accepts an ``AgnoTeam`` directly.

    Args:
        instructions: Question sent to the model for delegation.
        confidence_threshold: Minimum acceptable confidence.  When the decision
            falls below this value and *fallback_agent_index* is set the
            fallback is returned instead; when *raise_on_low_confidence* is
            ``True`` a :exc:`LayaLowConfidenceError` is raised.
        fallback_agent_index: Zero-based index into the agents list to use when
            confidence is below *confidence_threshold*.
        raise_on_low_confidence: Raise :exc:`LayaLowConfidenceError` instead of
            silently accepting a low-confidence pick (no fallback must be set).
        agent: Optional local Laya ``Agent`` or ``Router`` instance.  When
            omitted the module-level default ``Router`` is used.
        base_url: Optional laya-serve endpoint (e.g. ``"http://localhost:8000"``).
            When set the router posts to that node instead of running locally.
        api_key: Bearer token for a protected laya-serve endpoint.
        model: Model name override forwarded to ``runner.predict``.
        max_len: Maximum input token budget for the state field.
        head_max_len: Maximum token budget for each question head.
        lang: Force a specific language code (e.g. ``"es"``, ``"fr"``).
        min_confidence: Core abstention gate – decisions below this value are
            abstained from by the runner itself, before the wrapper gates them.
        hooks / on_predict_start / on_predict_end / hooks_raise / hooks_timeout:
            Per-call hook controls forwarded to ``runner.predict``.  Only valid
            on the local path; a ``base_url`` with hooks configured raises
            :exc:`ValueError` instead of silently dropping them.
    """

    def __init__(
        self,
        instructions: str = ("Which agent is best qualified to handle this task or message?"),
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

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route(
        self,
        message: Any,
        agents: Sequence[Union[AgnoAgent, Dict[str, Any], str, Any]],
    ) -> AgnoRouteDecision:
        """Route *message* to the best-matching agent in ~33 ms.

        Args:
            message: A plain string, an Agno ``RunResponse``, or any dict that
                carries the text under a ``"content"`` / ``"message"`` key.
            agents: The candidate agents.  Each item may be an ``AgnoAgent``
                instance, a dict with ``"name"`` / ``"role"`` / ``"description"``
                keys, or a plain string used as both name and role.

        Returns:
            :class:`AgnoRouteDecision` with the selected agent and confidence.

        Raises:
            ValueError: If *agents* is empty.
            LayaLowConfidenceError: If confidence < *confidence_threshold* and
                no fallback is configured and *raise_on_low_confidence* is ``True``.
        """
        if not agents:
            raise ValueError("No agents provided to route the message to.")

        message_str = _extract_message_str(message)
        criteria = _format_agno_criteria(agents)

        questions = {
            "delegation": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        }

        res = _execute_decision(
            message_str,
            questions,
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
        self.last_decision = res

        # The same nested-shape checks as the guardrail screen: `answers` must be an object,
        # the answer must be an object, and `choice` is used with `.startswith` below -- a list
        # there raised AttributeError out of a routing decision.
        where = f"Laya server at {self.base_url}" if self.base_url else "Laya"
        ans = _answers_map(res, where).get("delegation", {})
        chosen_key = ans.get("choice")
        if chosen_key is not None and not isinstance(chosen_key, str):
            raise _malformed(where, "'choice' must be a string, got %s" % type(chosen_key).__name__)

        # Gate on the same number core's `flag_low_confidence` gates on:
        # `answer_confidence` first, then the entropy `confidence` field, so an
        # answer carrying only the older field is still gated (fail-closed).
        conf = _gate_confidence(ans)
        if conf is None:
            conf = 1.0

        # Map "agent_i" → integer index i.
        chosen_idx: int = 0
        if chosen_key and chosen_key.startswith("agent_"):
            try:
                chosen_idx = int(chosen_key.split("_")[1])
            except (ValueError, IndexError):
                chosen_idx = 0
        elif chosen_key in criteria:
            chosen_idx = list(criteria.keys()).index(chosen_key)

        # `chosen_key` is remote JSON and `agents` is the caller's list, so a two-agent team can be
        # told to route to `agent_9`: the parsed number indexes no agent and `agents[chosen_idx]`
        # below raised a bare IndexError out of the routing decision, while `agent_-1` silently
        # wrapped to the LAST agent. `criteria` is built from exactly these agents, so an index it
        # does not name is a candidate that does not exist -- the same "unknown decision" that
        # `tools.py:493` refuses by name rather than raising.
        if not 0 <= chosen_idx < len(agents):
            raise _malformed(
                where,
                "decision %r names agent %d, which is not in the %d-agent team"
                % (chosen_key, chosen_idx, len(agents)),
            )

        # Confidence gating (same logic as LayaCrewRouter).
        if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
            if self.fallback_agent_index is not None:
                fallback_obj = agents[self.fallback_agent_index]
                fallback_name = _get_agent_name(fallback_obj, self.fallback_agent_index)
                fallback_role = _get_agent_role(fallback_obj, self.fallback_agent_index)
                reason = (
                    f"Routed to fallback agent '{fallback_name}' because routing confidence "
                    f"({conf:.3f}) was below threshold ({self.confidence_threshold:.3f})."
                )
                return AgnoRouteDecision(
                    agent=fallback_obj,
                    agent_index=self.fallback_agent_index,
                    name=fallback_name,
                    role=fallback_role,
                    confidence=conf,
                    reason=reason,
                    raw_decision=res,
                )
            if self.raise_on_low_confidence:
                raise LayaLowConfidenceError(
                    f"Agno routing confidence {conf:.3f} below threshold "
                    f"{self.confidence_threshold:.3f} for message: {message_str!r}",
                    confidence=conf,
                    threshold=self.confidence_threshold,
                    raw_decision=res,
                )

        selected_agent = agents[chosen_idx]
        selected_name = _get_agent_name(selected_agent, chosen_idx)
        selected_role = _get_agent_role(selected_agent, chosen_idx)
        reason = f"Routed to '{selected_name}' via Laya System 1 decision (confidence: {conf:.3f})."

        return AgnoRouteDecision(
            agent=selected_agent,
            agent_index=chosen_idx,
            name=selected_name,
            role=selected_role,
            confidence=conf,
            reason=reason,
            raw_decision=res,
        )

    def route_team(
        self,
        message: Any,
        team: AgnoTeam,
    ) -> AgnoRouteDecision:
        """Convenience wrapper: route *message* using members from an ``AgnoTeam``.

        Reads ``team.members`` (the Agno ``Team`` contract, ``agno>=1.1.14,<2``)
        with a compat fallback to ``team.agents`` for hand-crafted mocks or test
        objects using the old attribute name.  Members may also be a callable factory
        (Agno supports dynamic member lists); it is called with no arguments.

        Args:
            message: The message to route.
            team: An ``AgnoTeam`` instance (``Team(members=[...], mode="route")``).

        Returns:
            :class:`AgnoRouteDecision` with the selected agent and confidence.
        """
        # Primary: Agno Team API (>=1.1.14,<2) uses .members
        members = getattr(team, "members", None)
        if not members:
            # Compat fallback for hand-crafted dicts / mocks using the old .agents attribute
            members = getattr(team, "agents", None) or []
        # Agno supports a callable factory for dynamic member lists
        if callable(members):
            members = members()
        return self.route(message, members)

    def run(
        self,
        message: Any,
        agents: Sequence[Union[AgnoAgent, Dict[str, Any], str, Any]],
    ) -> AgnoRouteDecision:
        """Route *message* and return the selected :class:`AgnoRouteDecision`.

        Alias for :meth:`route`; provided for API convention consistency with
        Agno without executing the selected agent directly.
        """
        return self.route(message, agents)

    def __call__(
        self,
        message: Any,
        agents: Sequence[Union[AgnoAgent, Dict[str, Any], str, Any]],
    ) -> AgnoRouteDecision:
        return self.route(message, agents)

    async def aroute(
        self,
        message: Any,
        agents: Sequence[Union[AgnoAgent, Dict[str, Any], str, Any]],
    ) -> AgnoRouteDecision:
        """Asynchronously route *message* without blocking the asyncio event loop."""
        return await asyncio.to_thread(self.route, message, agents)

    async def aroute_team(
        self,
        message: Any,
        team: AgnoTeam,
    ) -> AgnoRouteDecision:
        """Asynchronously route *message* using members from an ``AgnoTeam``."""
        return await asyncio.to_thread(self.route_team, message, team)

    async def arun(
        self,
        message: Any,
        agents: Sequence[Union[AgnoAgent, Dict[str, Any], str, Any]],
    ) -> AgnoRouteDecision:
        """Asynchronously route *message* without blocking the event loop.

        Alias for :meth:`aroute`; returns the selected :class:`AgnoRouteDecision`
        without executing the selected agent directly.
        """
        return await asyncio.to_thread(self.route, message, agents)


class LayaAgnoGuardrail:
    """Sub-40ms inline content guardrail for Agno agent messages.

    Screens messages before they reach an Agno ``Agent`` or ``Team``, detecting
    jailbreaks, prompt injections, and policy violations in a single
    non-autoregressive Laya forward pass – without calling an LLM.

    Args:
        questions: Custom guardrail question definitions.  When ``None`` the
            built-in ``guard_questions()`` preset is used.
        action: One of ``"raise"`` (default), ``"filter"``, or ``"annotate"``.

            - ``"raise"``: raise :exc:`LayaAgnoGuardrailError` when a violation
              is detected.
            - ``"filter"``: return *rejection_message* instead of the original
              message.
            - ``"annotate"``: return a dict ``{"message": ..., "guardrail": {...}}``
              with violation metadata attached.

        rejection_message: Replacement string returned in ``"filter"`` mode.
        threshold: Violation probability above which a question is considered a
            violation.  Must be in ``[0, 1]``.
        agent / base_url / api_key / model / max_len / head_max_len / lang /
        min_confidence / hooks / …: Standard Laya per-call controls (see
            :class:`LayaAgnoRouter`).
    """

    def __init__(
        self,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "raise",
        rejection_message: str = "This message cannot be processed because it violates safety guidelines.",
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

    def screen(self, message: Any) -> Any:
        """Screen *message* against safety policies in ~33 ms.

        Args:
            message: A plain string, Agno ``RunResponse``, or dict.

        Returns:
            The original *message* when it passes (``"raise"`` / ``"annotate"``
            mode), the *rejection_message* string or annotated dict otherwise.

        Raises:
            LayaAgnoGuardrailError: In ``"raise"`` mode when a violation is
                detected.
        """
        message_str = _extract_message_str(message)
        qdefs = self.questions if self.questions is not None else self._default_questions()

        res = _execute_decision(
            message_str,
            qdefs,
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
                # Gate on the probability that the level is at or above the
                # middle of the scale – same rule as LayaGuardrail / LayaTaskGuard.
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
            raise LayaAgnoGuardrailError(
                f"Laya Agno guardrail policy violation detected: {list(violations.keys())}",
                violations=violations,
                raw_decision=res,
            )

        if not is_safe and self.action == "filter":
            return self.rejection_message

        if self.action == "annotate":
            guard_meta = {
                "passed": is_safe,
                "violations": violations,
                "answers": answers,
            }
            if isinstance(message, dict):
                annotated = dict(message)
                annotated["guardrail"] = guard_meta
                return annotated
            return {"message": message, "guardrail": guard_meta}

        return message

    def __call__(self, message: Any) -> Any:
        return self.screen(message)

    async def ascreen(self, message: Any) -> Any:
        """Asynchronously screen *message* without blocking the event loop."""
        return await asyncio.to_thread(self.screen, message)
