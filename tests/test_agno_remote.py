"""Agno remote routing hints reach the laya-serve request body."""

import io
import json
import urllib.request

import pytest

from laya_mlx.integrations.agno import LayaAgnoGuardrail, LayaAgnoRouter

URL = "http://localhost:8000/v1/systemone"
HINTS = [
    {},
    {"task": "guard"},
    {"lang_guess": "de"},
    {"task": "guard", "lang_guess": "de"},
    {"lang_guess": ""},
]


class RemoteServer:
    """Stands in for ``urllib.request.build_opener`` and for the opener it returns."""

    def __init__(self):
        self.reply = {}
        self.openers = 0
        self.requests = []

    def __call__(self, *handlers):
        self.openers += 1
        return self

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return io.BytesIO(json.dumps(self.reply).encode("utf-8"))


@pytest.fixture
def remote(monkeypatch):
    server = RemoteServer()
    monkeypatch.setattr(urllib.request, "build_opener", server)
    return server


def route(**hints):
    return LayaAgnoRouter(base_url="http://localhost:8000", **hints).route("hello", ["a", "b"])


def screen(**hints):
    return LayaAgnoGuardrail(base_url="http://localhost:8000", **hints).screen("hello")


ENTRIES = [
    pytest.param(
        route,
        {"answers": {"delegation": {"choice": "agent_1", "confidence": 0.99}}},
        "b",
        id="route",
    ),
    pytest.param(screen, {"answers": {}}, "hello", id="screen"),
]


@pytest.mark.parametrize("hints", HINTS, ids=["none", "task", "lang", "both", "blank-lang"])
@pytest.mark.parametrize("entry, reply, expected", ENTRIES)
def test_agno_remote_routing_hints(remote, entry, reply, expected, hints):
    remote.reply = reply
    decision = entry(**hints)
    assert getattr(decision, "agent", decision) == expected
    ((request, _timeout),) = remote.requests
    assert request.get_method() == "POST"
    assert request.full_url == URL
    body = json.loads(request.data)
    assert {key: body[key] for key in ("task", "lang_guess") if key in body} == hints


@pytest.mark.parametrize("entry, reply, expected", ENTRIES)
def test_agno_remote_rejects_callable_before_http(remote, entry, reply, expected):
    remote.reply = reply
    with pytest.raises(ValueError, match="lang_guess"):
        entry(lang_guess=lambda state: "de")
    assert remote.openers == 0
    assert remote.requests == []
