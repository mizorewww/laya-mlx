import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya_mlx.serve import _resolve_model, create_app  # noqa: E402


class FakeRouter:
    loaded = ["english"]

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions, model))
        return {
            "model": "laya-rl-agent",
            "answers": {"dept": {"choice": "billing"}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


def test_systemone_matches_jev_shape_and_model_routing(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = FakeRouter()
    client = TestClient(create_app(router))
    response = client.post(
        "/v1/systemone",
        json={
            "model": "convaiinnovations/laya-multilingual",
            "state": {"body": "hello"},
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "team",
                    "criteria": ["billing"],
                }
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["usage"]["output_tokens"] == 0
    assert router.calls[0][2] == "multilingual"


def test_unknown_model_is_auto_routed(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = FakeRouter()
    client = TestClient(create_app(router))
    response = client.post("/v1/systemone", json={"model": "jev-1", "questions": {}})
    assert response.status_code == 200
    assert router.calls[0][2] is None


def test_invalid_request_and_auth(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "secret")
    client = TestClient(create_app(FakeRouter()))
    assert client.post("/v1/systemone", content=b"not json").status_code == 401
    assert (
        client.post(
            "/v1/systemone",
            json={"questions": {}},
            headers={"Authorization": "Bearer secret"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/systemone",
            content=b"not json",
            headers={"Authorization": "Bearer secret"},
        ).status_code
        == 400
    )


def test_health(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    response = TestClient(create_app(FakeRouter())).get("/health")
    assert response.json() == {"status": "ok", "loaded": ["english"], "device": "auto"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("multilingual", "multilingual"),
        ("convaiinnovations/laya-typed-decisions", "typed-decisions"),
        ("convaiinnovations/laya", None),
        ("jev-1", None),
    ],
)
def test_resolve_model(model, expected):
    assert _resolve_model(model) == expected
