"""A Router that answers over HTTP from a running ``laya-serve`` instead of loading weights.

``laya-mcp-server`` builds a ``Router`` and loads checkpoints into its own process. That is the
right default for one agent on one machine, but it means every MCP client that opens a session
holds its own copy of the model: two Claude sessions plus a ``laya-serve`` for scripts is three
times 1.3 GB for the same checkpoint. Set ``LAYA_BASE_URL`` and the MCP server uses this class
instead: ``predict`` and ``predict_batch`` become ``POST /v1/systemone`` calls to that server,
and the MCP process itself stays at a few tens of megabytes, with no torch import at all.

Routing stays local and free. ``RemoteRouter`` is a ``Router``, so ``route`` / ``route_batch``
run the same language and script detection the server runs, without a model. The server makes
its own routing decision for the actual answer and reports it in ``routing``; the two agree when
the MCP process and the server see the same ``LAYA_DEFAULT_MODEL`` / ``LAYA_AUTO_TASK``.

Not available remotely: ``load`` (and therefore ``laya_shortlist``, which needs the answering
checkpoint's embeddings in-process), ``preload`` and ``unload`` (no-ops: the server owns its
checkpoints' lifetime, see ``LAYA_IDLE_UNLOAD_SECONDS``), and per-call hooks (a hook is a callable
that has to live where inference runs, exactly as ``laya.serve`` already refuses them).

Standard library only: ``urllib`` carries the requests, so the extra pulls in nothing new.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Union

from .. import __version__
from ..router import Router
from ..serve import BODY_CONTROLS, BODY_REFUSALS

DEFAULT_TIMEOUT_S = 300.0

# Request fields `laya.serve` reads off a /v1/systemone body, in the order the single-request
# handler validates them. Anything not in this list has no wire form and is refused or dropped.
_WIRE_CONTROLS = BODY_CONTROLS


class RemoteError(RuntimeError):
    """An HTTP-level failure talking to laya-serve. ``code`` is a tool-error style slug."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _timeout_from_env() -> float:
    raw = os.environ.get("LAYA_REMOTE_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        seconds = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return seconds if math.isfinite(seconds) and seconds > 0 else DEFAULT_TIMEOUT_S


def _resolve_user_agent(value: Optional[str]) -> str:
    raw = os.environ.get("LAYA_USER_AGENT", "") if value is None else value
    if not isinstance(raw, str):
        raise TypeError("user_agent must be a string or None")
    # Unlike a bad timeout, an invalid header must fail rather than hide control characters
    # behind a default. Check before trimming so trailing CR/LF cannot disappear.
    if any(
        (ord(char) < 32 and char != "\t") or ord(char) == 127 or ord(char) > 255 for char in raw
    ):
        raise ValueError(
            "LAYA_USER_AGENT / user_agent must be a single Latin-1 header value without control characters"
        )
    return raw.strip(" \t") or "laya/" + __version__


class RemoteRouter(Router):
    """Route locally, answer from a ``laya-serve`` at ``base_url``.

    ``base_url`` is the server origin (``http://127.0.0.1:8000``); the ``/v1/systemone`` paths are
    appended here. ``api_key``, when given, is sent as ``Authorization: Bearer`` and must match the
    server's ``LAYA_API_KEY``.

    ``user_agent`` overrides ``LAYA_USER_AGENT``; ``None`` reads the environment. A selected
    empty or space/tab-only value uses ``laya/<version>``. Header values must be Latin-1 and
    contain no ASCII controls other than tab. The value is resolved once at construction.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        *,
        user_agent: Optional[str] = None,
        **router_kwargs: Any,
    ):
        # device/preload make no sense for a router that never builds an Agent.
        router_kwargs.pop("device", None)
        router_kwargs.pop("preload", None)
        super().__init__(**router_kwargs)
        self.base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) server URL without a query or fragment")
        self.api_key = api_key or None
        self.timeout = _timeout_from_env() if timeout is None else float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        self.user_agent = _resolve_user_agent(user_agent)

    # ------------------------------------------------------------------ transport
    def _request(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            if exc.code == 401:
                raise RemoteError(
                    "unauthorized", "laya-serve refused the bearer token: " + detail
                ) from exc
            if exc.code in (400, 413, 422):
                raise RemoteError("invalid_request", detail) from exc
            if exc.code == 503:
                raise RemoteError("busy", "laya-serve is busy: " + detail) from exc
            raise RemoteError(
                "upstream_error", "laya-serve answered HTTP %d: %s" % (exc.code, detail)
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RemoteError(
                    "timeout", "laya-serve did not answer within %.0fs" % self.timeout
                ) from exc
            raise RemoteError(
                "unreachable", "cannot reach laya-serve at %s: %s" % (self.base_url, exc.reason)
            ) from exc
        except TimeoutError as exc:
            raise RemoteError(
                "timeout", "laya-serve did not answer within %.0fs" % self.timeout
            ) from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteError("upstream_error", "laya-serve returned a non-JSON body") from exc
        if not isinstance(payload, dict):
            raise RemoteError("upstream_error", "laya-serve returned a non-object body")
        return payload

    # ------------------------------------------------------------------ server state
    def health(self) -> Dict[str, Any]:
        """The server's ``GET /health`` payload (liveness only when the bearer is wrong)."""
        return self._request("/health")

    @property
    def loaded(self) -> List[str]:
        """Checkpoints resident on the server right now, or [] when it cannot be asked."""
        try:
            value = self.health().get("loaded")
        except RemoteError:
            return []
        return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []

    @property
    def loaded_revisions(self) -> Dict[str, Optional[str]]:
        try:
            value = self.health().get("revisions")
        except RemoteError:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    # ------------------------------------------------------------------ lifecycle (server-owned)
    def load(self, name: str):
        raise RemoteError(
            "unsupported_remote",
            "checkpoint %r lives in laya-serve at %s; a remote router cannot hand out an Agent"
            % (name, self.base_url),
        )

    def preload(self, names: Optional[List[str]] = None):
        return self  # the server decides what is resident

    def unload(self, name: Optional[str] = None):
        return None  # the server decides when to free memory (LAYA_IDLE_UNLOAD_SECONDS)

    def attach(self, name: str, agent: Any):
        raise RemoteError("unsupported_remote", "a remote router holds no agents to attach to")

    # ------------------------------------------------------------------ answering
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
        if any(value is not None for value in (image, audio, video)):
            # `Router.predict` takes these since 0.4.25 for the vega lane; laya-serve has no media
            # encoding, so the remote says so here rather than sending a file name as a state
            raise RemoteError(
                "unsupported_remote",
                "images, recordings and clips cannot be sent to laya-serve; use "
                "laya.Agent(vegaml=True) or a local Router with the vega lane",
            )
        if hooks_timeout is not None:
            raise RemoteError("unsupported_remote", "hooks_timeout belongs to the server's hooks")
        if any(
            value is not None for value in (hooks, on_predict_start, on_predict_end, hooks_raise)
        ):
            raise RemoteError(
                "unsupported_remote",
                "hooks are callables and cannot be sent to laya-serve; install them on the server",
            )
        body: Dict[str, Any] = {"state": state, "questions": questions}
        controls = {
            "model": model,
            "task": task,
            "lang": lang,
            "lang_guess": lang_guess,
            "max_len": max_len,
            "head_max_len": head_max_len,
            "min_confidence": min_confidence,
        }
        for key in _WIRE_CONTROLS:
            value = controls[key]
            if value is not None and not (key == "model" and value == "auto"):
                if key == "lang_guess" and not isinstance(value, str):
                    raise RemoteError(
                        "unsupported_remote", "a callable lang_guess cannot be sent over HTTP"
                    )
                body[key] = value
        return self._request("/v1/systemone", body)

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
        """One ``/v1/systemone`` call per request, answers in input order.

        ``laya-serve``'s ``/v1/systemone/batch`` shares one ``questions`` object across every state,
        while a Router batch lets each request carry its own questions and controls, so the general
        form is the per-request call. ``batch_size`` and ``sort_by_length`` shape forward passes and
        have no meaning here; they are accepted and ignored so callers written against ``Router``
        keep working, and ``diagnostics`` is the same kind of tolerance: a remote run has no local
        checkpoint grouping to report (#1111), so the dict is accepted and left untouched.
        The server applies its own batching inside each call.
        """
        results: List[Dict[str, Any]] = []
        if hooks_timeout is not None:
            raise RemoteError("unsupported_remote", "hooks_timeout belongs to the server's hooks")
        if any(
            value is not None for value in (hooks, on_predict_start, on_predict_end, hooks_raise)
        ):
            raise RemoteError(
                "unsupported_remote",
                "hooks are callables and cannot be sent to laya-serve; install them on the server",
            )
        for item in requests:
            if not isinstance(item, dict):
                raise TypeError(
                    "predict_batch requests must be dicts, got %s" % type(item).__name__
                )
            if any(item.get(key) is not None for key in BODY_REFUSALS):
                raise RemoteError(
                    "unsupported_remote", "per-request hooks cannot be sent to laya-serve"
                )
            kwargs = {
                k: item[k]
                for k in ("model", "task", "lang", "lang_guess", "max_len", "head_max_len")
                if k in item
            }
            if min_confidence is not None:
                kwargs["min_confidence"] = min_confidence
            results.append(self.predict(item["state"], item["questions"], **kwargs))
        return results

    def __repr__(self) -> str:
        return "RemoteRouter(base_url=%r, timeout=%g)" % (self.base_url, self.timeout)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The server's ``detail`` string when the error body is FastAPI-shaped, else the status text."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 -- any unreadable body falls back to the HTTP reason
        return str(exc.reason)
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail, ensure_ascii=False)
    return str(exc.reason)
