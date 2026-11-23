import mlx.core as mx
import numpy as np
import pytest

from laya_mlx.model import DecisionModel, EncoderConfig, _local_attention, attention_masks


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


@pytest.mark.parametrize("length", [384, 417])
@pytest.mark.parametrize("window", [64, 127, 128])
def test_block_local_attention_keeps_keys_at_window_boundaries(length, window):
    rng = np.random.default_rng(7)
    q, k, v = [
        mx.array(rng.standard_normal((1, 2, length, 32), dtype=np.float32)) for _ in range(3)
    ]
    mask = attention_masks(mx.ones((1, length), dtype=mx.bool_), window)["sliding_attention"]
    scale = q.shape[-1] ** -0.5
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    actual = _local_attention(q, k, v, mask, scale, window)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_pipelined_forward_uses_block_local_attention(monkeypatch):
    import laya_mlx.model as model_module

    cfg = EncoderConfig.from_dict(
        {
            "vocab_size": 128,
            "hidden_size": 128,  # head_dim 64 enables the block-local path
            "intermediate_size": 96,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "local_attention": 128,
        }
    )
    mx.random.seed(0)
    model = DecisionModel(cfg, {"head_layers": 2, "act_costs": {"escalate": 0.5}})
    mx.eval(model.parameters())
    length = 768
    rng = np.random.default_rng(3)
    inputs = {
        "input_ids": mx.array(rng.integers(1, 128, (1, length)).astype(np.int32)),
        "attention_mask": mx.ones((1, length), dtype=mx.bool_),
        "marker_pos": mx.array([[4, 9, 13]], dtype=mx.int32),
        "marker_mask": mx.array(np.ones((1, 3), bool)),
        "qtype": mx.array(np.zeros(1, np.int32)),
    }
    expected = model(**inputs)
    calls = []
    local = model_module._local_attention

    def record(*args):
        calls.append(args[-1])
        return local(*args)

    monkeypatch.setattr(model_module, "_local_attention", record)
    actual = model._pipelined(**inputs)
    assert calls == [128, 128]  # the two sliding-window layers
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("tail", [1, 8])
def test_block_local_attention_folds_small_query_tail(tail, monkeypatch):
    length = 256 + tail
    inputs = mx.zeros((1, 1, length, 32))
    mask = attention_masks(mx.ones((1, length), dtype=mx.bool_), 128)["sliding_attention"]
    query_sizes = []
    sdpa = mx.fast.scaled_dot_product_attention

    def record(query, key, value, **kwargs):
        query_sizes.append(query.shape[2])
        return sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", record)
    actual = _local_attention(inputs, inputs, inputs, mask, 32**-0.5, 128)
    assert query_sizes == [128, 128 + tail]
    assert actual.shape == inputs.shape
