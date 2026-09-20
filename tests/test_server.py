import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from laya_mlx.server import DecisionHandler


class DummyAgent:
    def __init__(self):
        self.model_id = "test/mock-model"
        self.device = "gpu"
        self.quantize = 4
        self.dtype = "float16"

    def predict(self, state, questions):
        if state == "error":
            raise RuntimeError("Simulated inference failure")
        answers = {}
        for qid, qdef in questions.items():
            answers[qid] = {
                "type": qdef.get("type", "choice"),
                "choice": "mock_choice",
                "score": 0.95,
            }
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}


@pytest.fixture
def mock_server():
    DecisionHandler.agent = DummyAgent()
    server = ThreadingHTTPServer(("127.0.0.1", 0), DecisionHandler)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


def test_health_check(mock_server):
    req = urllib.request.Request(f"{mock_server}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["service"] == "laya-mlx"
        assert data["model"] == "test/mock-model"
        assert data["quantized"] == 4


def test_options_cors(mock_server):
    req = urllib.request.Request(f"{mock_server}/predict", method="OPTIONS")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


def test_predict_success(mock_server):
    payload = {
        "state": "The user would like to cancel their subscription",
        "questions": {
            "intent": {"type": "choice", "instructions": "Classify", "criteria": ["cancel", "keep"]}
        },
    }
    req = urllib.request.Request(
        f"{mock_server}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))
        assert "answers" in result
        assert "intent" in result["answers"]
        assert result["answers"]["intent"]["choice"] == "mock_choice"
        assert result["usage"]["input_tokens"] == 10


def test_predict_sse_streaming(mock_server):
    payload = {
        "state": "test stream state",
        "questions": {
            "q1": {"type": "choice"},
            "q2": {"type": "score"},
        },
        "stream": True,
    }
    req = urllib.request.Request(
        f"{mock_server}/v1/decisions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type")
        lines = resp.read().decode("utf-8").strip().split("\n\n")
        events = [
            json.loads(line.replace("data: ", "")) for line in lines if line.startswith("data: ")
        ]
        assert len(events) == 3  # q1, q2, and done
        assert events[0]["question_id"] in ("q1", "q2")
        assert events[-1].get("done") is True


def test_predict_invalid_inputs(mock_server):
    # Empty body
    req_empty = urllib.request.Request(
        f"{mock_server}/predict",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req_empty)
    assert exc_info.value.code == 400

    # Malformed json
    req_bad_json = urllib.request.Request(
        f"{mock_server}/predict",
        data=b"invalid-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req_bad_json)
    assert exc_info.value.code == 400

    # Questions not a dict
    req_bad_questions = urllib.request.Request(
        f"{mock_server}/predict",
        data=json.dumps({"state": "hello", "questions": ["not-a-dict"]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req_bad_questions)
    assert exc_info.value.code == 400


def test_predict_agent_error(mock_server):
    req = urllib.request.Request(
        f"{mock_server}/predict",
        data=json.dumps({"state": "error", "questions": {"q": {"type": "choice"}}}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 500


def test_not_found(mock_server):
    req = urllib.request.Request(f"{mock_server}/unknown_endpoint")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 404
