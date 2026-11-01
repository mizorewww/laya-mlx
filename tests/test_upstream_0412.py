"""Regressions ported with upstream v0.4.12; no pretrained weights required."""

import numpy as np
import pytest

from laya_mlx.calibrate import fit_one_temperature, fit_temperature_map
from laya_mlx.common import ece_score, resolve_lang_temperatures
from laya_mlx.shortlist import predict_shortlist, predict_tournament
from laya_mlx.structured import SchemaError, questions_from_json_schema


def spread_pool(width):
    rng = np.random.RandomState(0)
    pairs = []
    for _ in range(24):
        logits = rng.randn(width)
        target = np.zeros(width)
        target[np.argmax(logits)] = 1
        pairs.append((logits, target))
    return pairs


def test_ragged_temperature_fit_preserves_real_option_counts(monkeypatch):
    # Same seeded fixture as upstream tests/test_calibrate.py: padding must not
    # compete in softmax, and a runaway log(T) must not turn the fit into NaN/1.
    pairs = spread_pool(2) + spread_pool(9)
    assert fit_one_temperature(pairs, min_n=10) == 0.5
    report = fit_temperature_map([(0, z, t) for z, t in pairs])
    assert report["temperature"][0] == 0.5
    assert report["temperature_clamped"]["choice"]["bound"] == "min"
    # Verify the pure-NumPy fallback path produces the same result when torch is absent.
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    assert fit_one_temperature(pairs, min_n=10) == 0.5
    rng = np.random.RandomState(42)
    z = rng.randn(64, 4).astype(np.float32)
    shifted = z / 1.8 - (z / 1.8).max(axis=-1, keepdims=True)
    p = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    assert fit_one_temperature(list(zip(z, p)), min_n=10) == pytest.approx(1.8, rel=1e-4)


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("key", ["const", "enum"])
def test_boolean_singletons_keep_schema_constraint(value, key):
    constraint = {key: [value] if key == "enum" else value}
    schema = {"type": "object", "properties": {"flag": {"type": "boolean", **constraint}}}
    q = questions_from_json_schema(schema)["flag"]
    assert q["type"] == "choice"
    assert q["criteria"] == {str(value): None}


def test_enum_string_is_not_split_into_characters():
    with pytest.raises(SchemaError, match="enum.*array"):
        questions_from_json_schema({"type": "object", "properties": {"x": {"enum": "yes"}}})


@pytest.mark.parametrize("bins", [0, -1, True, 2.5])
def test_ece_refuses_invalid_histogram_counts(bins):
    with pytest.raises(ValueError, match="bins"):
        ece_score(np.array([0.9]), np.array([0]), bins=bins)


def test_language_temperature_replacement_is_reported():
    with pytest.warns(RuntimeWarning, match="temperature\\[0\\]=True"):
        value = resolve_lang_temperatures({"de": {"temperature": [True, 2, 3]}}, [1, 1, 1])
    assert value["de"]["temperature"] == [1, 2, 3]


@pytest.mark.parametrize("helper", ["shortlist", "tournament"])
def test_narrowing_drops_only_stale_order_without_mutating_input(helper):
    questions = {
        "q": {
            "type": "choice",
            "instructions": "choose",
            "criteria": {str(i): None for i in range(6)},
            "option_order": list(range(6)),
        }
    }
    seen = []

    class Runner:
        def predict(self, state, qs):
            seen.extend(qs.values())
            return {
                "answers": {qid: {"choice": next(iter(q["criteria"]))} for qid, q in qs.items()}
            }

    with pytest.warns(RuntimeWarning, match="option_order"):
        if helper == "shortlist":
            predict_shortlist(
                Runner(), "state", questions, lambda texts: np.ones((len(texts), 2)), k=2
            )
        else:
            predict_tournament(Runner(), "state", questions, group_size=2)
    assert all("option_order" not in q for q in seen)
    assert questions["q"]["option_order"] == list(range(6))
    assert len(questions["q"]["criteria"]) == 6


@pytest.mark.parametrize("layout", ["sequential", "parallel"])
def test_native_agent_exposes_layout_to_stability_hook(layout):
    from laya_mlx.agent import Agent
    from laya_mlx.hooks import OptionStabilityHook, PredictContext

    agent = Agent.__new__(Agent)
    agent.cfg = {"option_layout": layout}
    ctx = PredictContext(states=["x"], questions={}, agent=agent)
    assert OptionStabilityHook._detect_layout(ctx) == layout


def test_device_report_tracks_native_cpu_and_gpu_without_loading_models():
    import mlx.core as mx

    from laya_mlx.mcp.device import agent_device

    class NativeAgent:
        pass

    agent = NativeAgent()
    current = mx.default_device()
    current_kind = str(current.type).rsplit(".", 1)[-1]
    for value, expected in ((mx.cpu, "cpu"), (mx.gpu, "gpu"), (current, current_kind)):
        agent.device = value
        assert agent_device(agent) == expected
