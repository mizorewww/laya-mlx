"""Native MLX HTTP server contracts and replacements for Torch-specific upstream serve checks."""

import inspect
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from laya_mlx.common import serialize_state
from laya_mlx.mcp.tools import laya_status
from laya_mlx.router import Router
from laya_mlx.serve import (
    _LONE_SURROGATE_DETAIL,
    BATCH_BODY_CALL_CONTROLS,
    BATCH_BODY_ITEM_CONTROLS,
    BATCH_BODY_OUT_PARAMETERS,
    BODY_CONTROLS,
    BODY_REFUSALS,
    LIVENESS_ONLY,
    _apply_thread_limit,
    _state_length,
    build_router,
    create_app,
)


class _EmptyRouter:
    loaded = []
    loaded_revisions = {}
    _agents = {}


class _DeviceRouter:
    def __init__(self, **devices):
        self.loaded = list(devices)
        self.loaded_revisions = {name: "0" * 40 for name in devices}
        self._agents = {name: SimpleNamespace(device=dev) for name, dev in devices.items()}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gpu", "gpu"),
        ("mps", "mps"),
        ("metal", "metal"),
        ("cpu", "cpu"),
    ],
)
def test_health_without_a_resident_checkpoint_flags_the_preference(monkeypatch, raw, expected):
    monkeypatch.setenv("LAYA_DEVICE", raw)
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    body = TestClient(create_app(router=_EmptyRouter())).get("/health").json()
    assert body["device_is_preference"] is True, body
    assert body["checkpoint_devices"] == {}, body
    assert body["device"] == expected, body


def test_health_agrees_with_the_mcp_status_tool(monkeypatch):
    """One fact about the native MLX device, reported the same way by both surfaces."""
    monkeypatch.setenv("LAYA_DEVICE", "gpu")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = _DeviceRouter(english="cpu")
    body = TestClient(create_app(router=fake)).get("/health").json()
    status = laya_status(router=fake, loaded=list(fake.loaded))
    assert body["device"] == status["device"] == "cpu"
    assert body["checkpoint_devices"] == status["checkpoint_devices"] == {"english": "cpu"}
    assert body["device_is_preference"] == status["device_is_preference"] is False


def test_build_router_strips_and_validates_mlx_device_preferences(monkeypatch):
    """Whitespace/newlines from .env or Dockerfiles are stripped; CUDA strings are refused."""
    import laya_mlx.router

    seen = {}

    class RecordingRouter:
        def __init__(self, device=None, **kwargs):
            seen["device"] = device

        def preload(self, names=None):
            seen["preloaded"] = names

    monkeypatch.setattr(laya_mlx.router, "Router", RecordingRouter)
    monkeypatch.setenv("LAYA_PRELOAD", "0")
    for raw, want in (
        (" cpu\n", "cpu"),
        ("MPS ", "mps"),
        (" GPU\n", "gpu"),
        ("Metal ", "metal"),
        ("   ", None),
        ("cpu", "cpu"),
    ):
        monkeypatch.setenv("LAYA_DEVICE", raw)
        build_router()
        assert seen["device"] == want, "%r -> %r" % (raw, seen["device"])
    monkeypatch.delenv("LAYA_DEVICE", raising=False)
    build_router()
    assert seen["device"] is None
    for unsupported in ("cuda", "cuda:0", "rocm"):
        monkeypatch.setenv("LAYA_DEVICE", unsupported)
        with pytest.raises(ValueError, match="LAYA_DEVICE"):
            build_router()


def test_thread_limit_refuses_mlx_thread_override(monkeypatch):
    monkeypatch.delenv("LAYA_THREADS", raising=False)
    assert _apply_thread_limit() is None
    for ignored in ("0", "-4", "abc", ""):
        monkeypatch.setenv("LAYA_THREADS", ignored)
        assert _apply_thread_limit() is None
    monkeypatch.setenv("LAYA_THREADS", "8")
    with pytest.raises(ValueError, match="LAYA_THREADS cannot control MLX thread pools"):
        _apply_thread_limit()


def test_body_controls_and_refusals_partition_router_predict_signature():
    params = [
        name
        for name in inspect.signature(Router.predict).parameters
        if name not in ("self", "state", "questions")
    ]
    assert set(params) == set(BODY_CONTROLS) | set(BODY_REFUSALS)
    assert set(BODY_CONTROLS).isdisjoint(BODY_REFUSALS)


def test_batch_body_controls_partition_router_predict_batch_signature():
    params = [
        name
        for name in inspect.signature(Router.predict_batch).parameters
        if name not in ("self", "requests")
    ]
    assert set(params) == (
        set(BATCH_BODY_CALL_CONTROLS) | set(BATCH_BODY_OUT_PARAMETERS) | set(BODY_REFUSALS)
    )
    assert set(BATCH_BODY_ITEM_CONTROLS) <= set(BODY_CONTROLS)


@pytest.mark.parametrize(
    "state",
    [
        "plain string",
        {"body": 'quote " and unicode \u200b'},
        ["turn 1", {"nested": [1, True, None]}],
    ],
)
def test_state_length_matches_serialize_state(state):
    assert _state_length(state) == len(serialize_state(state))


def test_lone_surrogate_rejected_on_single_and_batch_routes(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=_EmptyRouter()))
    single_raw = b'{"state": "\\ud800", "questions": {"q": {"type": "noul", "instructions": "?"}}}'
    batch_raw = (
        b'{"states": ["\\ud800"], "questions": {"q": {"type": "noul", "instructions": "?"}}}'
    )
    for route, payload in (("/v1/systemone", single_raw), ("/v1/systemone/batch", batch_raw)):
        resp = client.post(route, content=payload, headers={"content-type": "application/json"})
        assert resp.status_code == 400
        assert resp.json()["detail"] == _LONE_SURROGATE_DETAIL


def test_health_liveness_only_when_api_key_set(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "secret-token")
    fake = _DeviceRouter(english="gpu")
    client = TestClient(create_app(router=fake))
    unauth = client.get("/health")
    assert unauth.status_code == 200
    assert unauth.json() == LIVENESS_ONLY
    auth = client.get("/health", headers={"Authorization": "Bearer secret-token"})
    assert auth.status_code == 200
    assert auth.json()["checkpoint_devices"] == {"english": "gpu"}
