"""Opt-in TypeSafe prediction runner through Monid (standard library HTTP only)."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from .tools import _SameOriginRedirectHandler


class MonidRunner:
    """Pass as ``agent=`` to framework integrations to use hosted typed decisions.

    Requires MONID_API_KEY. Local model controls and hooks are intentionally unsupported.
    """

    def __init__(self, timeout: float = 60.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def predict(self, state, questions, *, model=None):
        """Evaluate TypeSafe questions and return the provider's decision result."""
        api_key = os.environ.get("MONID_API_KEY")
        if not api_key:
            raise ValueError("MONID_API_KEY is required for MonidRunner")
        body = {"state": state, "questions": questions}
        if model is not None:
            body["model"] = model
        payload = {
            "provider": "typesafe",
            "endpoint": "/systemone",
            "input": {"body": body},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        opener = urllib.request.build_opener(_SameOriginRedirectHandler())
        deadline = time.monotonic() + self.timeout

        def request(url, data=None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Monid run TIMED_OUT")
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with opener.open(req, timeout=remaining) as response:
                    return response.status, json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Monid server error {exc.code}: {error}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Failed to connect to Monid server: {exc}") from exc

        status, run = request(
            "https://api.monid.ai/v1/run",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        if status == 202:
            run_id = urllib.parse.quote(run["runId"], safe="")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Monid run TIMED_OUT")
                time.sleep(min(2, remaining))
                _, run = request(f"https://api.monid.ai/v1/runs/{run_id}")
                if run.get("status") in ("COMPLETED", "FAILED", "TIMED_OUT", "STOPPED", "BLOCKED"):
                    break
        if run.get("status") not in (None, "COMPLETED"):
            raise RuntimeError(f"Monid run {run.get('status')}")
        provider = run.get("providerResponse") or {}
        if provider.get("httpStatus", 200) >= 400:
            error = json.dumps(provider.get("error"), ensure_ascii=False)
            raise RuntimeError(f"Laya server error {provider['httpStatus']}: {error}")
        return run["output"]
