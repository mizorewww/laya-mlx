"""Lightweight production HTTP and SSE Decision Server for Laya-MLX.

Provides a fast local decision server without heavy external dependencies.
Ideal for edge deployments on low-end devices (e.g. Mac mini, MacBook Air, local microservices).
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

try:
    from .agent import Agent
except ImportError:
    Agent = None  # type: ignore[assignment, misc]

logger = logging.getLogger("laya_mlx.server")


class DecisionHandler(BaseHTTPRequestHandler):
    agent: Optional[object] = None

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        if self.path in ("/health", "/"):
            resp = {
                "status": "ok",
                "service": "laya-mlx",
                "model": getattr(self.agent, "model_id", "unknown"),
                "device": str(getattr(self.agent, "device", "unknown")),
                "quantized": getattr(self.agent, "quantize", None),
                "dtype": str(getattr(self.agent, "dtype", "unknown")),
            }
            body = json.dumps(resp).encode("utf-8")
            self._set_headers(200)
            self.wfile.write(body)
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def do_POST(self):
        if self.path not in ("/predict", "/v1/decisions"):
            self._set_headers(404)
            self.wfile.write(b'{"error": "Endpoint not found"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._set_headers(400)
            self.wfile.write(b'{"error": "Missing or empty request body"}')
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {e}"}).encode("utf-8"))
            return

        state = payload.get("state", "")
        questions = payload.get("questions", {})
        stream = payload.get("stream", False) or "text/event-stream" in self.headers.get(
            "Accept", ""
        )

        if not isinstance(questions, dict):
            self._set_headers(400)
            self.wfile.write(b'{"error": "\'questions\' field must be a dictionary"}')
            return

        if self.agent is None:
            self._set_headers(503)
            self.wfile.write(b'{"error": "Laya agent is not loaded"}')
            return

        try:
            result = self.agent.predict(state, questions)

            if stream:
                # Stream answers individually as SSE events
                self._set_headers(200, content_type="text/event-stream")
                for qid, ans in result.get("answers", {}).items():
                    event = {"question_id": qid, "answer": ans}
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(
                    f"data: {json.dumps({'done': True, 'usage': result.get('usage')})}\n\n".encode(
                        "utf-8"
                    )
                )
                self.wfile.flush()
            else:
                body = json.dumps(result).encode("utf-8")
                self._set_headers(200)
                self.wfile.write(body)
        except Exception as e:
            logger.exception("Prediction failed")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Prediction failed: {e}"}).encode("utf-8"))


def serve(
    agent_or_path="convaiinnovations/laya",
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    dtype: str = "float16",
    quantize: Optional[int] = None,
    batch_size: int = 16,
    low_memory: bool = False,
):
    """Start a lightweight decision server."""
    if hasattr(agent_or_path, "predict"):
        agent = agent_or_path
    else:
        if Agent is None:
            raise RuntimeError("Agent requires MLX which is not installed on this system")
        agent = Agent(
            agent_or_path,
            dtype=dtype,
            quantize=quantize,
            batch_size=batch_size,
            low_memory=low_memory,
        )

    DecisionHandler.agent = agent
    server = ThreadingHTTPServer((host, port), DecisionHandler)
    logger.info(f"Laya-MLX decision server listening on http://{host}:{port}")
    print(f"[Laya-MLX] Decision server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping decision server...")
        server.server_close()
