"""v0.4.34 boundary regressions that run without native MLX or model weights."""

import sys
import types
import warnings

import numpy as np
import pytest

from laya_mlx.agent import Agent, load
from laya_mlx.calibrate import apply_calibration_payload
from laya_mlx.common import (
    check_batch_size,
    clamp_temperature,
    temperature_rejected,
    window_batch_cap,
)
from laya_mlx.revisions import verify_digests
from laya_mlx.router import VEGA_MODEL, Router

WIDE = 10**400  # a JSON integer literal that `float()` refuses


@pytest.mark.parametrize("value", [2.0, "2", True, False, [2], object()])
def test_batch_size_is_refused_by_name_before_any_work(value):
    with pytest.raises(ValueError, match="batch_size must be an integer count of states"):
        check_batch_size(value)
    with pytest.raises(ValueError, match="batch_size must be an integer count of states"):
        window_batch_cap(4, 64, 512, value)
    # The native Agent validates before it reads hooks, the tokenizer or the model.
    with pytest.raises(ValueError, match="batch_size must be an integer count of states"):
        Agent.__new__(Agent).predict_batch(["a", "b"], {}, batch_size=value)


def test_batch_size_keeps_the_no_cap_spellings_and_accepts_index_types():
    assert [check_batch_size(v) for v in (None, 0, -1, 8)] == [None, 0, -1, 8]
    assert type(check_batch_size(np.int64(8))) is int
    assert window_batch_cap(4, 64, 512, np.int64(2)) == 2
    for unset in (0, -1):
        assert window_batch_cap(4, 64, 512, unset) == window_batch_cap(4, 64, 512, None)


@pytest.mark.parametrize("value", [WIDE, -WIDE, 2**1024])
def test_temperature_wider_than_a_double_falls_back_and_is_named(value):
    assert clamp_temperature(value) == 1.0
    assert temperature_rejected(value, 1.0)
    agent = Agent.__new__(Agent)
    agent.model_id_or_path, agent.subfolder, agent.cfg = "local", None, {}
    payload = {"temperature": [value, 1.2, 1.1], "temperature_by_options": {"choice:2": value}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_calibration_payload(agent, payload)
    assert agent.temperature == [1.0, 1.2, 1.1]
    assert agent.temperature_by_options == {"choice:2": 1.0}
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "temperature[0]=%d -> 1" % value in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize("value", [WIDE, -WIDE, 2**1024])
def test_binning_value_wider_than_a_double_is_a_named_refusal(value):
    payload = {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [value]}}}
    with pytest.raises(ValueError, match="number"):
        apply_calibration_payload(Agent.__new__(Agent), payload)


def test_digest_variable_too_deep_to_parse_is_a_configuration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_SHA256_DIGESTS", "[" * 100_000)
    with pytest.raises(ValueError, match="LAYA_SHA256_DIGESTS must be a JSON object"):
        verify_digests(str(tmp_path))
    assert Router().route("Please refund the duplicate charge on my account.").model == "english"


@pytest.mark.parametrize("value", [True, "4b", "nandakishorm/vega-U", 1])
def test_vegaml_is_refused_before_loading_anything(value, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)  # any native import would now fail
    with pytest.raises(ValueError, match="requires official laya"):
        Agent(vegaml=value)
    with pytest.raises(ValueError, match="requires official laya"):
        load(vegaml=value)


def test_vega_lane_stays_off_by_default_even_when_vegaml_is_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "vegaml", types.ModuleType("vegaml"))
    router = Router()
    assert not router.vega_enabled
    assert router.route("x" * 80_000).model != VEGA_MODEL


def test_a_reader_name_routes_but_cannot_be_loaded_here():
    router = Router(vega="4b")
    assert router.route(None, {}, media="image").model == VEGA_MODEL
    with pytest.raises(ImportError, match="official laya"):
        router.load(VEGA_MODEL)
    with pytest.raises(ImportError, match="official laya"):
        router.predict(None, {}, image="invoice.png")
