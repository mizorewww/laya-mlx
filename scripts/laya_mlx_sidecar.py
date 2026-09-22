#!/usr/bin/env python3
"""A reference MLX sidecar speaking the same HTTP contract as `laya-mcp serve`.

`laya-mcp` drives the upstream PyTorch runtime. This serves the MLX port in
`laya_mlx` behind the same three endpoints, so `dsh-laya` can be pointed at it
with nothing but a `sidecarUrl` change:

    .venv/bin/python scripts/laya_mlx_sidecar.py --port 8787

Contract notes, and where this differs from `laya-mcp`:

* ``POST /ask``   - noul / choice / score, same answer shape.
* ``POST /plan``  - the same budget arithmetic, computed from the tokenizer with
  no forward pass. ``exact`` is false: token counts are estimates, as upstream's
  own are.
* ``GET /health`` - ok / loaded / degraded / calls / failures.

A ``noul`` is re-expressed as a neutral ``A``/``B`` choice before it reaches the
model, exactly as ``laya_mcp.protocol.noul_as_choice`` does, because a noul
renders as ``false: ...`` / ``true: ...`` and that framing is unreliable. Pass
``--raw-noul`` to send nouls through unmodified instead.

This is deliberately small and is not a reimplementation of `laya-mcp`: it has no
calibration store, no language routing, no strict mode and no MCP transport. It
exists so the MLX speedup can be measured against the same contract.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: The neutral labels a noul is re-asked under. See laya_mcp.protocol.
NEUTRAL_LABELS = ("A", "B")

#: Both boundaries belong to "uncertain" - a probability on a threshold is
#: exactly the case the band exists to call unclear.
BAND_LOW, BAND_HIGH = 0.30, 0.70


def band_of(probability: float) -> str:
    if probability < BAND_LOW:
        return "no"
    if probability > BAND_HIGH:
        return "yes"
    return "uncertain"


def noul_as_choice(question: dict) -> dict:
    """Re-express a noul as a two-option choice under neutral labels.

    The caller's own option text is kept; its absence is left absent rather than
    filled with boilerplate, which would itself cause the failure this avoids.
    """
    criteria = question.get("criteria")
    criteria = criteria if isinstance(criteria, dict) else {}
    true_text = criteria.get("true")
    false_text = criteria.get("false")
    return {
        "type": "choice",
        "instructions": question["instructions"],
        "criteria": {
            NEUTRAL_LABELS[0]: true_text if true_text not in (None, "") else "",
            NEUTRAL_LABELS[1]: false_text if false_text not in (None, "") else "",
        },
    }


class Engine:
    """Loads the MLX agent and serialises access to it.

    Laya is not thread-safe - ``Agent`` moves its own device on an OOM - so every
    forward pass takes the lock, matching `laya-mcp`'s default concurrency of 1.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.agent = None
        self.load_seconds = 0.0
        self._lock = threading.Lock()
        self._started = time.time()
        self.calls = 0
        self.failures = 0
        self.last_latency_ms = 0.0

    def load(self) -> None:
        from laya_mlx import Agent

        t0 = time.time()
        self.agent = Agent(
            self.args.model,
            dtype=self.args.dtype,
            device=self.args.device,
            batch_size=self.args.batch_size,
        )
        self.load_seconds = time.time() - t0

    @property
    def max_len(self) -> int:
        return int(self.agent.cfg.get("max_len", 512))

    @property
    def head_max_len(self) -> int:
        return int(self.agent.cfg.get("head_max_len", 192))

    def _run(self, state, questions: dict) -> tuple[dict, dict]:
        converted: dict = {}
        is_noul: dict = {}
        for qid, definition in questions.items():
            if definition.get("type") == "noul" and not self.args.raw_noul:
                converted[qid] = noul_as_choice(definition)
                is_noul[qid] = True
            else:
                converted[qid] = definition
                is_noul[qid] = False

        result = self.agent.predict(state, converted)
        answers: dict = {}
        for qid, answer in result["answers"].items():
            entry = {k: v for k, v in answer.items() if k != "action"}
            entry["question_id"] = qid
            if is_noul.get(qid):
                probabilities = entry.get("probabilities") or {}
                p = float(probabilities.get(NEUTRAL_LABELS[0], 0.0))
                entry = {
                    "question_id": qid,
                    "type": "noul",
                    "noul": round(p, 4),
                    "confidence": round(max(p, 1.0 - p), 4),
                    "band": band_of(p),
                }
            answers[qid] = entry
        return answers, result.get("usage", {})

    def ask(self, payload: dict) -> dict:
        state = payload.get("state")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("`questions` must be a non-empty object keyed by question id")
        with self._lock:
            t0 = time.time()
            answers, usage = self._run(state, questions)
            self.last_latency_ms = (time.time() - t0) * 1000
        self.calls += 1
        return {
            "ok": True,
            "answers": answers,
            "usage": usage,
            "latency_ms": round(self.last_latency_ms, 1),
            "model": "laya_mlx",
            "device": str(self.agent.device),
        }

    def plan(self, payload: dict) -> dict:
        state = payload.get("state")
        questions = payload.get("questions") or {}
        state_text = state if isinstance(state, str) else json.dumps(state)
        tokens = len(self.agent.tok(state_text, add_special_tokens=False)["input_ids"])
        room = max(0, self.max_len - self.head_max_len)
        would_truncate = tokens > room
        warnings = []
        if would_truncate:
            warnings.append(
                "the state is close to or over its budget, so its tail is likely to be "
                "discarded; raise max_len at startup, shorten the state, or set strict=true "
                "to refuse instead"
            )
        return {
            "ok": True,
            "checkpoint": self.args.model,
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
            "exact": False,
            "state_chars": len(state_text),
            "state_tokens_estimated": tokens,
            "state_room_estimated": room,
            "would_truncate_state": would_truncate,
            "fits": not would_truncate,
            "questions": [
                {
                    "question_id": qid,
                    "type": definition.get("type"),
                    "would_truncate_state": would_truncate,
                }
                for qid, definition in questions.items()
            ],
            "warnings": warnings,
        }

    def health(self) -> dict:
        loaded = self.agent is not None
        return {
            "ok": loaded,
            "loaded": loaded,
            "degraded": False,
            "calls": self.calls,
            "failures": self.failures,
            "last_latency_ms": round(self.last_latency_ms, 1),
            "uptime_s": round(time.time() - self._started, 1),
            "load_seconds": round(self.load_seconds, 1),
            "checkpoints": {
                self.args.model: {
                    "device": str(self.agent.device) if loaded else "unloaded",
                    "dtype": self.args.dtype,
                    "max_len": self.max_len if loaded else 0,
                    "head_max_len": self.head_max_len if loaded else 0,
                }
            },
        }


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, self.engine.health())
        else:
            self._send(404, {"ok": False, "error": f"no route for GET {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/")
        if route not in ("/ask", "/plan"):
            self._send(404, {"ok": False, "error": f"no route for POST {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"ok": False, "error": f"invalid JSON: {exc}"})
            return
        try:
            result = self.engine.ask(payload) if route == "/ask" else self.engine.plan(payload)
            self._send(200, result)
        except Exception as exc:  # noqa: BLE001
            self.engine.failures += 1
            self._send(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        if self.engine.args.log_level == "debug":
            print(f"[laya-mlx-sidecar] {fmt % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--device", default="gpu", choices=("gpu", "metal", "cpu"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-noul", action="store_true",
                        help="send nouls unmodified instead of re-asking them as an A/B choice")
    parser.add_argument("--log-level", default="info", choices=("debug", "info", "warning"))
    args = parser.parse_args()

    engine = Engine(args)
    print("[laya-mlx-sidecar] loading the MLX agent ...", flush=True)
    engine.load()
    print(
        f"[laya-mlx-sidecar] ready in {engine.load_seconds:.1f}s · "
        f"device={engine.agent.device} dtype={args.dtype} "
        f"max_len={engine.max_len} head_max_len={engine.head_max_len}",
        flush=True,
    )
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[laya-mlx-sidecar] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
