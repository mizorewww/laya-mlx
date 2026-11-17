"""Native replacements for the four upstream serve tests excluded as Torch-specific.

The rest of upstream's tests/test_serve.py runs unchanged through test_official_host.py.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from laya_mlx.mcp.tools import laya_status
from laya_mlx.serve import _apply_thread_limit, build_router, create_app


class _EmptyRouter:
    loaded = []
    loaded_revisions = {}
    _agents = {}


class _DeviceRouter:
    def __init__(self, **devices):
        self.loaded = list(devices)
        self.loaded_revisions = {name: "0" * 40 for name in devices}
        self._agents = {name: SimpleNamespace(device=dev) for name, dev in devices.items()}


@pytest.mark.parametrize("raw", ["gpu", "mps", "metal", "cpu"])
def test_health_without_a_resident_checkpoint_flags_the_preference(monkeypatch, raw):
    monkeypatch.setenv("LAYA_DEVICE", raw)
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    body = TestClient(create_app(router=_EmptyRouter())).get("/health").json()
    assert body["device_is_preference"] is True, body
    assert body["checkpoint_devices"] == {}, body
    assert body["device"] == raw, body


def test_health_agrees_with_the_mcp_status_tool(monkeypatch):
    monkeypatch.setenv("LAYA_DEVICE", "gpu")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = _DeviceRouter(english="cpu")
    body = TestClient(create_app(router=fake)).get("/health").json()
    status = laya_status(router=fake, loaded=list(fake.loaded))
    assert body["device"] == status["device"] == "cpu"
    assert body["checkpoint_devices"] == status["checkpoint_devices"] == {"english": "cpu"}
    assert body["device_is_preference"] == status["device_is_preference"] is False


def test_build_router_strips_and_validates_mlx_device_preferences(monkeypatch):
    """Whitespace from .env or Dockerfiles is stripped; CUDA strings are refused."""
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
