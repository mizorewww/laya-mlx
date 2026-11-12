"""Monid runner contracts, with no network or model loading."""

import io
import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from laya_mlx.integrations import MonidRunner, monid, tools


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv("MONID_API_KEY", "fake-monid-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    opener = MagicMock()
    monkeypatch.setattr(monid.urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(monid.time, "sleep", lambda seconds: None)
    return opener


def response(data, status=200):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status = status
    result.read.return_value = json.dumps(data).encode("utf-8")
    return result


def test_predict_request_and_result(http):
    output = {"answers": {"yes": {"value": 0.9}}}
    http.open.return_value = response({"output": output})
    questions = {"yes": {"type": "noul", "instructions": "Is this a refund?"}}
    assert MonidRunner().predict("退款", questions) == output
    req = http.open.call_args.args[0]
    assert req.full_url == "https://api.monid.ai/v1/run"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer fake-monid-key"
    assert req.get_header("Content-type") == "application/json; charset=utf-8"
    assert "退款".encode() in req.data
    assert json.loads(req.data) == {
        "provider": "typesafe",
        "endpoint": "/systemone",
        "input": {"body": {"state": "退款", "questions": questions}},
    }


def test_explicit_runner_integration(http):
    output = {"answers": {}}
    http.open.return_value = response({"output": output})
    assert tools._execute_decision("state", {}, agent=MonidRunner()) == output


def test_no_key_keeps_local_default(monkeypatch):
    monkeypatch.delenv("MONID_API_KEY", raising=False)
    runner = MagicMock()
    monkeypatch.setattr(tools, "_get_default_router", lambda: runner)
    assert tools._execute_decision("state", {}) == runner.predict.return_value
    runner.predict.assert_called_once_with("state", {})
    with pytest.raises(ValueError, match="MONID_API_KEY"):
        MonidRunner().predict("state", {})


def test_async_polling_and_model(http):
    http.open.side_effect = [
        response({"runId": "run-one"}, 202),
        response({"status": "RUNNING"}),
        response({"status": "COMPLETED", "output": {"answers": {}}}),
    ]
    assert MonidRunner().predict({}, {}, model="jev-preview") == {"answers": {}}
    first, *polls = http.open.call_args_list
    assert json.loads(first.args[0].data)["input"]["body"]["model"] == "jev-preview"
    for call in polls:
        assert call.args[0].full_url == "https://api.monid.ai/v1/runs/run-one"
        assert call.args[0].get_method() == "GET"
        assert call.args[0].get_header("Authorization") == "Bearer fake-monid-key"


@pytest.mark.parametrize("status", [400, 404, 429])
def test_provider_errors_use_existing_server_error(http, status):
    http.open.return_value = response(
        {"status": "COMPLETED", "providerResponse": {"httpStatus": status, "error": "bad input"}}
    )
    with pytest.raises(RuntimeError, match=f"Laya server error {status}: .*bad input"):
        MonidRunner().predict("state", {})


@pytest.mark.parametrize("status", ["FAILED", "TIMED_OUT", "STOPPED", "BLOCKED"])
def test_terminal_failure(http, status):
    http.open.side_effect = [response({"runId": "one"}, 202), response({"status": status})]
    with pytest.raises(RuntimeError, match=f"Monid run {status}"):
        MonidRunner().predict("state", {})


def test_http_error(http):
    http.open.side_effect = urllib.error.HTTPError(
        "https://api.monid.ai/v1/run", 401, "Unauthorized", {}, io.BytesIO(b'"bad key"')
    )
    with pytest.raises(RuntimeError, match="Monid server error 401: .*bad key"):
        MonidRunner().predict("state", {})


def test_poll_timeout(http, monkeypatch):
    http.open.return_value = response({"runId": "one"}, 202)
    times = iter([0, 0, 61])
    monkeypatch.setattr(monid.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="Monid run TIMED_OUT"):
        MonidRunner().predict("state", {})
