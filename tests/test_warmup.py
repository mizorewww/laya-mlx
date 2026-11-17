"""The public warmup uses official synthetic shapes through native transport."""

import json
import warnings

import numpy as np
import pytest

from laya_mlx.runtime import RuntimeMixin


def make_agent():
    agent = RuntimeMixin.__new__(RuntimeMixin)
    agent.cfg = {"max_len": 96}
    agent.tok = type("Tokenizer", (), {"cls_token_id": 7})()
    calls = []
    agent._forward = calls.append
    agent.predict = lambda *args, **kwargs: pytest.fail("warmup must bypass hooks and decoding")
    return agent, calls


def test_official_default_shapes_cap_context_without_hooks():
    agent, calls = make_agent()
    assert agent.warmup() >= 0
    assert [batch["input_ids"].shape for batch in calls] == [(1, 64), (4, 96)]
    for batch in calls:
        assert (batch["input_ids"] == 7).all()
        assert batch["attention_mask"].all()
        assert batch["marker_mask"].all()
        assert (batch["qtype"] == 0).all()
        np.testing.assert_array_equal(
            batch["marker_pos"][0], np.arange(1, batch["marker_mask"].shape[1] + 1)
        )


def test_custom_shapes_minimum_room_and_explicit_failure():
    agent, calls = make_agent()
    agent.tok.cls_token_id = None
    agent.warmup([(2, 1, 3)])
    assert calls[0]["input_ids"].shape == (2, 5)
    assert not calls[0]["input_ids"].any()
    agent.warmup([])
    assert len(calls) == 1

    def fail(batch):
        raise RuntimeError("native failure")

    agent._forward = fail
    with pytest.raises(RuntimeError, match="native failure"):
        agent.warmup()


def test_native_backend_introspection_and_explicit_switch_boundary():
    agent, _ = make_agent()
    agent.dtype = np.dtype("float16")
    agent.manifest = {"precision": "float16"}
    assert agent.backend == "mlx"
    assert agent.backend_object is None
    assert agent.dtype_for(1) == agent.dtype_for(32) == np.dtype("float16")
    assert agent.accelerate() is False
    with pytest.raises(RuntimeError, match="CUDA"):
        agent.accelerate(strict=True)
    assert agent.deaccelerate() is None
    assert agent.backend == "mlx"
    for strict in (False, True):
        with pytest.raises(NotImplementedError, match="backend switching"):
            agent.set_backend("auto", strict=strict)


def test_parallel_warmup_supplies_shared_positions_and_isolated_options():
    agent, calls = make_agent()
    agent.cfg["option_layout"] = "parallel"
    agent.warmup([(2, 8, 3)])
    batch = calls[0]
    np.testing.assert_array_equal(batch["position_ids"], [[0, 1, 1, 1, 2, 3, 4, 5]] * 2)
    np.testing.assert_array_equal(batch["option_ids"], [[0, 1, 2, 3, 0, 0, 0, 0]] * 2)


@pytest.mark.parametrize("compiled", [False, True])
def test_parallel_native_warmup(tiny_checkpoint, compiled):
    from laya_mlx import Agent

    path = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(path.read_text())
    cfg["option_layout"] = "parallel"
    path.write_text(json.dumps(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        agent = Agent(tiny_checkpoint, compile=compiled, pad_to_multiple=16, batch_size=2)
        assert agent.warmup() >= 0
        assert agent.warmup([(2, 32, 3)]) >= 0
