import mlx.core as mx
import numpy as np
import pytest

from laya_mlx.model import DecisionModel, EncoderConfig


def small_model(dtype):
    cfg = EncoderConfig.from_dict(
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 7,  # submissions after layers 1, 3 and 7
            "num_attention_heads": 2,
            "local_attention": 8,
        }
    )
    mx.random.seed(0)
    model = DecisionModel(cfg, {"head_layers": 2, "act_costs": {"escalate": 0.5}})
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def batch(rows, length=40):
    rng = np.random.default_rng(rows)
    mask = np.ones((rows, length), np.int32)
    mask[-1, length - 9 :] = 0  # padding in the last row
    return {
        "input_ids": mx.array(rng.integers(1, 128, (rows, length)).astype(np.int32)),
        "attention_mask": mx.array(mask),
        "marker_pos": mx.array(np.tile([[4, 9, 13]], (rows, 1)).astype(np.int32)),
        "marker_mask": mx.array(np.ones((rows, 3), bool)),
        "qtype": mx.array(np.zeros(rows, np.int32)),
    }


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("rows", [1, 3])
def test_pipelined_forward_is_identical_to_eager(dtype, rows):
    model = small_model(dtype)
    inputs = batch(rows)
    expected = model(**inputs)
    actual = model._pipelined(**inputs)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
