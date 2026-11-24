"""Sequential half-precision action heads accumulate in FP32 on the CPU."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from laya_mlx.model import DecisionModel, EncoderConfig


@pytest.fixture
def cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("rows", [1, 3])
def test_sequential_cpu_fp16_action_retains_fp32_precision(cpu_default_device, rows):
    cfg = EncoderConfig.from_dict(
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 3,
            "num_attention_heads": 1,
            "local_attention": 16,
        }
    )
    model = DecisionModel(cfg, {"head_layers": 0, "act_costs": {"escalate": 0.5}})
    assert not model.parallel
    model.set_dtype(mx.float16)
    first, _, last = model.act_head.layers
    first.weight[:] = 0.0
    first.bias[:] = 1.0
    last.weight[:] = 0.0
    last.weight[:, 0] = 1.0
    last.bias[:] = 0.0
    mx.eval(model.parameters())
    markers = mx.array(np.tile([[1, 2]], (rows, 1)), dtype=mx.int32)
    _, action = model._decision(
        mx.zeros((rows, 4, 64), dtype=mx.float16),
        mx.ones((rows, 4), dtype=mx.bool_),
        markers,
        mx.ones((rows, 2), dtype=mx.bool_),
        mx.zeros((rows,), dtype=mx.int32),
    )
    expected = nn.gelu(mx.array([1.0], dtype=mx.float32)).item()
    assert action.shape == (rows, 2)
    assert action.dtype == mx.float32
    assert first.weight.dtype == mx.float16
    assert last.weight.dtype == mx.float16
    np.testing.assert_allclose(np.asarray(action), expected, rtol=0, atol=1e-7)
