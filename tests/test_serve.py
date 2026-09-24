import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya_mlx.serve import _env_positive_int, _resolve_model, build_router, create_app  # noqa: E402


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


def test_max_len_is_forwarded_to_router(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_LEN", "256")
    monkeypatch.setenv("LAYA_PRELOAD", "0")
    router = build_router()
    assert router.max_len == 256


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_max_len_rejects_invalid_environment_values(monkeypatch, value):
    monkeypatch.setenv("LAYA_MAX_LEN", value)
    with pytest.raises(ValueError, match="LAYA_MAX_LEN"):
        _env_positive_int("LAYA_MAX_LEN")


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
