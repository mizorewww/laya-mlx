"""Tool and function selection for agent frameworks using Laya System 1 decision engine.

Provides non-autoregressive tool routing and function selection for AI agent
workflows (LangChain, CrewAI, LlamaIndex, AutoGen, and custom agent loops).

Replaces slow, token-generating LLM tool calling (1,000-2,500ms) with calibrated, typed
decisions executed in a single forward pass without token generation cost.

Supports Python functions, LangChain/CrewAI/LlamaIndex tools, and JSON Schema specifications.
Operates seamlessly with local in-process models (`Agent` / `Router`) and remote HTTP
deployments (self-hosted `laya-serve`) without heavy client dependencies.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ..confidence import _gate_confidence, check_min_confidence, resolve_min_confidence

# Shared per-call control rules across integrations
from ._controls import budget_kwargs as _budget_kwargs
from ._controls import decision_kwargs as _decision_kwargs
from ._controls import hook_kwargs as _hook_kwargs
from ._controls import predict_kwargs as _predict_kwargs
from ._controls import reject_remote_hooks as _reject_remote_hooks
from ._errors import LayaLowConfidenceError


def _gated_confidence(answer: Dict[str, Any]) -> Optional[float]:
    """Extract calibrated confidence score from a Laya decision answer, or None if unusable."""
    return _gate_confidence(answer or {})


@dataclass
class ToolRouteDecision:
    """Result of a Laya tool routing decision."""

    tool: Any
    tool_name: str
    tool_index: int
    confidence: float
    probabilities: Dict[str, float]
    reason: str
    is_direct_answer: bool
    is_fallback: bool
    raw_decision: Dict[str, Any]


__all__ = [
    "LayaToolSelector",
    "ToolRouteDecision",
    "LayaLowConfidenceError",
]


def _extract_tool_metadata(tool: Any, default_index: int) -> Tuple[str, str]:
    """Extract (name, description) from any tool representation."""
    if isinstance(tool, str):
        if ":" in tool:
            parts = tool.split(":", 1)
            return parts[0].strip(), parts[1].strip()
        return tool.strip(), tool.strip()

    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            fn = tool["function"]
            name = fn.get("name") or f"tool_{default_index}"
            desc = fn.get("description") or name
            return str(name), str(desc)
        name = tool.get("name") or f"tool_{default_index}"
        desc = tool.get("description") or tool.get("doc") or name
        return str(name), str(desc)

    # Check for LlamaIndex tool.metadata
    if hasattr(tool, "metadata"):
        meta = tool.metadata
        name = getattr(meta, "name", None) or f"tool_{default_index}"
        desc = getattr(meta, "description", None) or name
        return str(name), str(desc)

    # Check for .name and .description attributes (LangChain, CrewAI, custom classes)
    name = getattr(tool, "name", None)
    desc = getattr(tool, "description", None)
    if name is not None:
        return str(name), str(desc or name)

    # Check callable / function
    if callable(tool):
        fn_name = getattr(tool, "__name__", None) or f"tool_{default_index}"
        doc = getattr(tool, "__doc__", None)
        desc = doc.strip().split("\n")[0] if doc else fn_name
        return str(fn_name), str(desc)

    return f"tool_{default_index}", f"Tool {default_index}"


def _format_tools_criteria(
    tools: Sequence[Any],
    allow_direct_answer: bool = False,
    direct_answer_key: str = "__direct_answer__",
    direct_answer_description: str = "Answer the user request directly without calling any tools.",
) -> Tuple[Dict[str, str], Dict[str, Tuple[int, Any, str]]]:
    """Format candidate tools into Laya choice criteria and reverse lookup map."""
    criteria: Dict[str, str] = {}
    lookup: Dict[str, Tuple[int, Any, str]] = {}

    for i, tool in enumerate(tools):
        name, desc = _extract_tool_metadata(tool, i)
        key = f"tool_{i}"
        criteria[key] = f"{name}: {desc}" if desc and desc != name else name
        lookup[key] = (i, tool, name)

    if allow_direct_answer:
        direct_key = "direct_answer"
        criteria[direct_key] = f"{direct_answer_key}: {direct_answer_description}"
        lookup[direct_key] = (-1, None, direct_answer_key)

    return criteria, lookup


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that switch origins or downgrade from HTTPS to HTTP."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlparse(req.full_url)
        new = urllib.parse.urlparse(newurl)
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
    """Send decision request to a remote laya-serve HTTP instance using standard library urllib."""
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
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Laya server error {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Laya server at {url}: {e}") from e


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
    hooks: Optional[Any] = None,
    on_predict_start: Optional[Any] = None,
    on_predict_end: Optional[Any] = None,
    hooks_raise: Optional[bool] = None,
    hooks_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute decision on local runner or remote laya-serve instance."""
    hook_kwargs = _hook_kwargs(hooks, on_predict_start, on_predict_end, hooks_raise, hooks_timeout)
    if base_url:
        _reject_remote_hooks(hook_kwargs, base_url)
        budget = _budget_kwargs(max_len, head_max_len)
        decision = _decision_kwargs(lang, min_confidence)
        return _call_remote(
            base_url, state, questions, api_key=api_key, model=model, **budget, **decision
        )
    runner = agent if agent is not None else _get_default_router()
    kwargs = _predict_kwargs(model, max_len, head_max_len, lang, min_confidence)
    kwargs.update(hook_kwargs)
    return runner.predict(state, questions, **kwargs)


class LayaToolSelector:
    """Non-autoregressive tool and function router for AI agent workflows.

    Evaluates candidate tools against a user prompt or agent state in a single
    forward pass (measured at 32.8 ms with `laya-multilingual` and 39.5 ms with
    `laya` on a Tesla T4 GPU; ~88–126 ms on Apple Silicon; 193–464 ms on CPU).

    Supports Python functions, LangChain/CrewAI/LlamaIndex tools, and JSON Schema specs.
    Works seamlessly with local in-process models (`Agent` / `Router`) and remote
    HTTP `laya-serve` instances.
    """

    def __init__(
        self,
        tools: Optional[Sequence[Any]] = None,
        instructions: str = "Which tool should be called to handle the user request?",
        allow_direct_answer: bool = False,
        direct_answer_description: str = "Answer the user request directly without calling any tools.",
        confidence_threshold: Optional[float] = None,
        fallback_tool: Optional[Any] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ):
        self.tools = list(tools) if tools is not None else None
        self.instructions = instructions
        self.allow_direct_answer = bool(allow_direct_answer)
        self.direct_answer_description = direct_answer_description
        if confidence_threshold is not None:
            if (
                isinstance(confidence_threshold, bool)
                or not isinstance(confidence_threshold, (int, float))
                or not math.isfinite(confidence_threshold)
                or confidence_threshold < 0.0
                or confidence_threshold > 1.0
            ):
                raise ValueError(
                    f"confidence_threshold must be a float in [0.0, 1.0], got {confidence_threshold!r}"
                )
            self.confidence_threshold = float(confidence_threshold)
        else:
            self.confidence_threshold = None
        self.fallback_tool = fallback_tool
        self.agent = agent
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.lang = lang
        if min_confidence is not None:
            check_min_confidence(min_confidence)
        self.min_confidence = min_confidence
        self.hooks = hooks
        self.on_predict_start = on_predict_start
        self.on_predict_end = on_predict_end
        self.hooks_raise = hooks_raise
        self.hooks_timeout = hooks_timeout

    def select(
        self,
        state: Union[str, dict, list, Any],
        tools: Optional[Sequence[Any]] = None,
        instructions: Optional[str] = None,
    ) -> ToolRouteDecision:
        """Select the best candidate tool for the given query/state in a single forward pass."""
        candidate_tools = tools if tools is not None else self.tools
        if not candidate_tools and not self.allow_direct_answer:
            raise ValueError("No tools provided to LayaToolSelector.")

        candidate_tools = candidate_tools or []
        instruct = instructions or self.instructions

        criteria, lookup = _format_tools_criteria(
            candidate_tools,
            allow_direct_answer=self.allow_direct_answer,
            direct_answer_description=self.direct_answer_description,
        )

        question = {
            "type": "choice",
            "instructions": instruct,
            "criteria": criteria,
        }
        questions = {"tool_select": question}

        state_payload = state if isinstance(state, (str, dict, list)) else str(state)
        result = _execute_decision(
            state_payload,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )

        # A remote laya-serve body is arbitrary JSON: a gateway or mis-pointed URL can answer
        # with a top-level array, string or number, where `.get` would raise AttributeError and
        # defeat the fail-closed branch below -- including the fallback_tool path meant to absorb
        # exactly this malformed response. Read `answers` only from an object; anything else is
        # the same "missing or malformed decision response" the guard already handles.
        answers_dict = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers_dict, dict) or "tool_select" not in answers_dict:
            if self.fallback_tool is not None:
                fb_name, _ = _extract_tool_metadata(self.fallback_tool, -1)
                return ToolRouteDecision(
                    tool=self.fallback_tool,
                    tool_name=fb_name,
                    tool_index=-1,
                    confidence=0.0,
                    probabilities={},
                    reason="Missing or malformed decision response; routed to fallback tool",
                    is_direct_answer=False,
                    is_fallback=True,
                    raw_decision=result,
                )
            raise RuntimeError(f"Malformed or missing decision response from Laya: {result!r}")

        answer = answers_dict["tool_select"]
        if not isinstance(answer, dict):
            if self.fallback_tool is not None:
                fb_name, _ = _extract_tool_metadata(self.fallback_tool, -1)
                return ToolRouteDecision(
                    tool=self.fallback_tool,
                    tool_name=fb_name,
                    tool_index=-1,
                    confidence=0.0,
                    probabilities={},
                    reason="Malformed decision answer; routed to fallback tool",
                    is_direct_answer=False,
                    is_fallback=True,
                    raw_decision=result,
                )
            raise RuntimeError(f"Malformed decision answer from Laya: {answer!r}")

        chosen_key = answer.get("choice")
        conf = _gated_confidence(answer)
        raw_probs = answer.get("probabilities", {})

        # Map internal keys back to tool names in probabilities. A remote body can carry a
        # probability that is not a number -- a bool, None, or a string like "high" -- and float()
        # would then raise a raw ValueError/TypeError here, after the malformed-response guards
        # above, defeating the same fallback_tool path that absorbs a malformed `answers` block.
        # A boolean is refused explicitly because bool is a subclass of int, so float(True) would
        # silently read as 1.0. Anything unparsable fails closed like those guards. Numeric strings
        # keep their existing acceptance.
        probabilities: Dict[str, float] = {}
        bad_prob = False
        if isinstance(raw_probs, dict):
            for k, prob in raw_probs.items():
                if isinstance(prob, bool):
                    bad_prob = True
                    break
                try:
                    value = float(prob)
                except (TypeError, ValueError):
                    bad_prob = True
                    break
                if k in lookup:
                    tool_info = lookup[k]
                    probabilities[tool_info[2]] = value
                else:
                    probabilities[k] = value

        if bad_prob:
            if self.fallback_tool is not None:
                fb_name, _ = _extract_tool_metadata(self.fallback_tool, -1)
                return ToolRouteDecision(
                    tool=self.fallback_tool,
                    tool_name=fb_name,
                    tool_index=-1,
                    confidence=0.0,
                    probabilities={},
                    reason="Malformed probability in decision answer; routed to fallback tool",
                    is_direct_answer=False,
                    is_fallback=True,
                    raw_decision=result,
                )
            raise RuntimeError(f"Malformed probability in decision answer from Laya: {raw_probs!r}")

        # Check confidence gating: unusable, missing, or NaN confidence fails closed
        threshold = (
            self.confidence_threshold
            if self.confidence_threshold is not None
            else self.min_confidence
        )
        # `min_confidence` may be a per-bucket map (#394), which `check_min_confidence` accepts in
        # `__init__`. Compare against the threshold this answer's bucket is gated at, the way
        # `flag_low_confidence` does, rather than against the dict itself.
        if isinstance(threshold, dict):
            threshold = resolve_min_confidence(answer, threshold)
        is_low_confidence = (
            bool(answer.get("low_confidence"))
            or conf is None
            or (threshold is not None and conf < threshold)
        )

        if is_low_confidence:
            effective_conf = conf if conf is not None else 0.0
            if self.fallback_tool is not None:
                fb_name, _ = _extract_tool_metadata(self.fallback_tool, -1)
                return ToolRouteDecision(
                    tool=self.fallback_tool,
                    tool_name=fb_name,
                    tool_index=-1,
                    confidence=effective_conf,
                    probabilities=probabilities,
                    reason=(
                        f"Confidence {effective_conf:.3f} below threshold {threshold}; routed to fallback tool"
                        if threshold is not None
                        else "Unusable or low confidence decision; routed to fallback tool"
                    ),
                    is_direct_answer=False,
                    is_fallback=True,
                    raw_decision=result,
                )
            raise LayaLowConfidenceError(
                (
                    f"Laya tool selection confidence {effective_conf:.3f} is below threshold {threshold}"
                    if threshold is not None
                    else "Laya tool selection returned unusable or low confidence."
                ),
                confidence=effective_conf,
                threshold=float(threshold) if threshold is not None else 0.0,
                raw_decision=result,
            )

        # A remote body can answer with a JSON array or object for `choice`, and `chosen_key not in
        # lookup` then raises a bare interpreter TypeError instead of taking the fallback path below
        # or naming the decision. `agno`/`crewai`/`llamaindex` refuse a non-string `choice` the same
        # way; here it belongs in the existing "unknown decision" branch so fallback still applies.
        if chosen_key is None or not isinstance(chosen_key, str) or chosen_key not in lookup:
            if self.fallback_tool is not None:
                fb_name, _ = _extract_tool_metadata(self.fallback_tool, -1)
                return ToolRouteDecision(
                    tool=self.fallback_tool,
                    tool_name=fb_name,
                    tool_index=-1,
                    confidence=conf if conf is not None else 0.0,
                    probabilities=probabilities,
                    reason=f"Unknown or missing tool decision {chosen_key!r}; routed to fallback tool",
                    is_direct_answer=False,
                    is_fallback=True,
                    raw_decision=result,
                )
            raise RuntimeError(
                f"Unknown or missing tool decision from Laya: choice {chosen_key!r} not in candidate tools."
            )

        idx, tool_obj, name = lookup[chosen_key]
        is_direct = idx == -1
        reason = (
            "Direct answer requested; no tool required"
            if is_direct
            else f"Selected tool '{name}' with confidence {conf:.3f}"
        )

        return ToolRouteDecision(
            tool=tool_obj,
            tool_name=name,
            tool_index=idx,
            confidence=conf,
            probabilities=probabilities,
            reason=reason,
            is_direct_answer=is_direct,
            is_fallback=False,
            raw_decision=result,
        )

    async def aselect(
        self,
        state: Union[str, dict, list, Any],
        tools: Optional[Sequence[Any]] = None,
        instructions: Optional[str] = None,
    ) -> ToolRouteDecision:
        """Asynchronously select the best candidate tool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.select, state, tools, instructions)

    def call(
        self,
        state: Union[str, dict, list, Any],
        tools: Optional[Sequence[Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Select the appropriate tool and invoke it directly with the provided arguments."""
        decision = self.select(state, tools=tools)
        if decision.is_direct_answer:
            return None
        tool = decision.tool
        if callable(tool):
            return tool(*args, **kwargs)
        if hasattr(tool, "run") and callable(tool.run):
            return tool.run(*args, **kwargs)
        if hasattr(tool, "invoke") and callable(tool.invoke):
            return tool.invoke(*args, **kwargs)
        raise TypeError(
            f"Selected tool '{decision.tool_name}' ({type(tool).__name__}) is not callable."
        )

    def as_runnable(self) -> Any:
        """Return a LangChain-compatible Runnable wrapper for use in LCEL chains."""
        try:
            from langchain_core.runnables import RunnableLambda

            return RunnableLambda(lambda inp: self.select(inp))
        except ImportError:

            class _SimpleRunnable:
                def __init__(self, selector: LayaToolSelector):
                    self.selector = selector

                def invoke(self, inp: Any) -> ToolRouteDecision:
                    return self.selector.select(inp)

                def __or__(self, other: Any) -> Any:
                    class _ChainedRunnable:
                        def __init__(self, first: Any, second: Any):
                            self.first = first
                            self.second = second

                        def invoke(self, inp: Any) -> Any:
                            res = self.first.invoke(inp)
                            if hasattr(self.second, "invoke"):
                                return self.second.invoke(res)
                            if callable(self.second):
                                return self.second(res)
                            return res

                    return _ChainedRunnable(self, other)

            return _SimpleRunnable(self)
