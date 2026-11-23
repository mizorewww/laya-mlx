"""Eager Metal dispatch and the block-local sliding-attention boundary."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers

import laya_mlx.model as model_module
from laya_mlx import Agent
from laya_mlx.common import parallel_layout
from laya_mlx.model import DecisionModel, EncoderConfig, _local_attention, attention_masks

MARKERS = [4, 9, 13]


def encoder_model(hidden_size=128, window=128, **agent):
    cfg = EncoderConfig.from_dict(
        {
            "vocab_size": 128,
            "hidden_size": hidden_size,
            "intermediate_size": 192,
            "num_hidden_layers": 3,  # one global layer, then two sliding-window layers
            "num_attention_heads": 2,
            "local_attention": window,
        }
    )
    mx.random.seed(0)
    model = DecisionModel(cfg, {"head_layers": 2, "act_costs": {"escalate": 0.5}, **agent})
    mx.eval(model.parameters())
    return model


def prepared_batch(length, valid_lengths):
    """A sequential host batch whose rows keep only their first `valid_lengths` positions."""
    rng = np.random.default_rng(length)
    mask = np.zeros((len(valid_lengths), length), dtype=np.bool_)
    for row, valid_length in enumerate(valid_lengths):
        mask[row, :valid_length] = True
    return {
        "input_ids": rng.integers(1, 128, (len(valid_lengths), length)).astype(np.int32),
        "attention_mask": mask,
        "marker_pos": np.tile([MARKERS], (len(valid_lengths), 1)).astype(np.int32),
        "marker_mask": np.ones((len(valid_lengths), len(MARKERS)), dtype=np.bool_),
        "qtype": np.zeros(len(valid_lengths), dtype=np.int32),
    }


def locate(full, part):
    """Offset of the sequence slice `part` inside `full`, read from its first row."""
    candidates = full[:, :, : full.shape[2] - part.shape[2] + 1]
    (start,) = np.flatnonzero((candidates == part[:, :, :1]).all(axis=(0, 1, 3)))
    np.testing.assert_array_equal(full[:, :, start : start + part.shape[2]], part)
    return start


def capture_block_attention(monkeypatch, call):
    """Return `call()`'s result, each sliding layer's full mask and the blocks it attended with.

    A block is `(query_start, key_start, mask)`, both offsets absolute positions in the sequence.
    """
    layers, calls, depth = [], [], 0
    local = model_module._local_attention
    sdpa = mx.fast.scaled_dot_product_attention

    def capture_local(q, k, v, mask, scale, window):
        nonlocal depth
        mx.eval(q, k, mask)
        layers.append((q, k, mask))
        depth += 1
        try:
            return local(q, k, v, mask, scale, window)
        finally:
            depth -= 1

    def capture_sdpa(query, key, value, **kwargs):
        if depth:
            calls.append((len(layers) - 1, query, key, kwargs["mask"]))
        return sdpa(query, key, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model_module, "_local_attention", capture_local)
        patch.setattr(mx.fast, "scaled_dot_product_attention", capture_sdpa)
        result = call()
    if calls:
        mx.eval(*[array for _, query, key, mask in calls for array in (query, key, mask)])
    masks = [np.asarray(mask) for _, _, mask in layers]
    blocks = []
    for layer, query, key, mask in calls:
        q, k = (np.asarray(array) for array in layers[layer][:2])
        query, key, mask = (np.asarray(array) for array in (query, key, mask))
        blocks.append((locate(q, query), locate(k, key), mask))
    return result, masks, blocks


@pytest.mark.parametrize(
    "length, valid_lengths",
    [(768, [768, 768]), (769, [769, 17]), (776, [128, 65, 776]), (777, [33, 777]), (768, [17])],
)
def test_block_masks_keep_valid_support(monkeypatch, length, valid_lengths):
    model = encoder_model()
    inputs = {k: mx.array(v) for k, v in prepared_batch(length, valid_lengths).items()}
    _, masks, blocks = capture_block_attention(monkeypatch, lambda: model._pipelined(**inputs))
    assert len(masks) == 2  # the two sliding-window layers
    np.testing.assert_array_equal(masks[0], masks[1])
    valid = np.asarray(inputs["attention_mask"])
    positions = np.arange(length)
    covered = []
    for start, left, mask in blocks:
        rows, width = mask.shape[2:]
        queries, keys = positions[start : start + rows], positions[left : left + width]
        near = np.abs(queries[:, None] - keys[None, :]) <= 64
        self_key = queries[:, None] == keys[None, :]
        for row in range(len(valid_lengths)):
            key_valid, live = valid[row, keys], valid[row, queries]
            np.testing.assert_array_equal(mask[row, 0][live], (near & key_valid)[live])
            assert mask[row, 0].any(axis=-1).all()
            padded = mask[row, 0][~live]
            assert not (padded & ~(key_valid | self_key)[~live]).any()
        assert width < length
        covered.extend(queries)
    np.testing.assert_array_equal(np.sort(covered), np.repeat(positions, len(masks)))

    on_gpu = mx.default_device() == mx.gpu
    dtype = np.float16 if on_gpu else np.float32
    rng = np.random.default_rng(11)
    q, k, v = [
        mx.array(rng.standard_normal((len(valid_lengths), 2, length, 64)).astype(dtype))
        for _ in range(3)
    ]
    dense = attention_masks(inputs["attention_mask"], 128)["sliding_attention"]
    expected = np.asarray(mx.fast.scaled_dot_product_attention(q, k, v, scale=64**-0.5, mask=dense))
    actual = np.asarray(_local_attention(q, k, v, mx.array(masks[0]), 64**-0.5, 128))
    keep = np.broadcast_to(valid[:, None, :, None], actual.shape)
    if on_gpu:
        np.testing.assert_array_equal(actual[keep], expected[keep])
    else:
        np.testing.assert_allclose(actual[keep], expected[keep], rtol=1e-5, atol=1e-6)
    assert np.isfinite(actual).all()


@pytest.fixture
def long_checkpoint(tmp_path):
    """A sequential checkpoint with the 64-dim heads and 128-token window the block path needs."""
    cfg = {
        "model_type": "modernbert",
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 192,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "local_attention": 128,
        "max_position_embeddings": 1024,
    }
    agent_cfg = {
        "encoder": "test/long",
        "head_layers": 1,
        "max_len": 1024,
        "head_max_len": 32,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:2": 1.7},
    }
    path = tmp_path / "checkpoint"
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg))
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            }
        )
    )
    mx.random.seed(7)
    model = encoder_model(head_layers=1)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return path


@pytest.mark.parametrize(
    "length, valid_lengths, expected",
    [
        (767, [767, 17], (0, 0)),
        (767, [767, 767], (0, 0)),
        (768, [768, 17], (2, 0)),
        (768, [768, 768], (2, 0)),
        (769, [769, 17], (2, 0)),
        (769, [769, 769], (2, 0)),
        (768, [17], (2, 2)),
        (768, [768], (2, 2)),
    ],
)
def test_agent_forward_dispatches_long_batches(
    long_checkpoint, monkeypatch, length, valid_lengths, expected
):
    if not mx.metal.is_available():
        pytest.skip("the eager Metal path needs a GPU")
    agent = Agent(long_checkpoint, dtype="float16", device="gpu")
    batch = prepared_batch(length, valid_lengths)
    with mx.stream(mx.gpu):
        dense = agent.model(**{k: mx.array(v) for k, v in batch.items()})
    calls = []
    local, async_eval = model_module._local_attention, mx.async_eval

    def local_spy(*args):
        calls.append("local")
        return local(*args)

    def async_eval_spy(*args, **kwargs):
        calls.append("async_eval")
        return async_eval(*args, **kwargs)

    monkeypatch.setattr(model_module, "_local_attention", local_spy)
    monkeypatch.setattr(mx, "async_eval", async_eval_spy)
    result = agent.forward(batch)
    assert (calls.count("local"), calls.count("async_eval")) == expected
    for actual, reference in zip(result, dense):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


@pytest.mark.parametrize(
    "hidden_size, window", [(64, 128), (128, 130)], ids=["head_dim_32", "window_130"]
)
def test_block_local_fallbacks_keep_dense_attention(monkeypatch, hidden_size, window):
    model = encoder_model(hidden_size, window)
    inputs = {k: mx.array(v) for k, v in prepared_batch(768, [768]).items()}
    dense = model(**inputs)
    actual, masks, blocks = capture_block_attention(monkeypatch, lambda: model._pipelined(**inputs))
    assert masks == [] and blocks == []
    for x, y in zip(actual, dense):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_block_local_fallbacks_keep_parallel_layout(monkeypatch):
    model = encoder_model(option_layout="parallel")
    inputs = {k: mx.array(v) for k, v in prepared_batch(768, [768]).items()}
    inputs.update({k: mx.array([v]) for k, v in parallel_layout(MARKERS, 16, 768).items()})
    dense = model(**inputs)
    option_masks = []
    parallel_option_masks = model_module.parallel_option_masks

    def record(*args):
        option_masks.append(args)
        return parallel_option_masks(*args)

    monkeypatch.setattr(model_module, "parallel_option_masks", record)
    actual, masks, blocks = capture_block_attention(monkeypatch, lambda: model._pipelined(**inputs))
    assert len(option_masks) == 1 and masks == [] and blocks == []
    for x, y in zip(actual, dense):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    missing = {k: v for k, v in inputs.items() if k not in ("position_ids", "option_ids")}
    with pytest.raises(ValueError, match="option layout"):
        model._pipelined(**missing)
    with pytest.raises(ValueError, match="option layout"):
        encoder_model()._pipelined(**inputs)


@pytest.mark.parametrize(
    "options",
    [
        dict(device="cpu", dtype="float32"),
        dict(device="gpu", dtype="float32"),
        dict(device="gpu", dtype="float16", compile=True),
    ],
    ids=["cpu", "float32", "compiled"],
)
def test_block_local_fallbacks_skip_other_agents(long_checkpoint, monkeypatch, options):
    if options["device"] == "gpu" and not mx.metal.is_available():
        pytest.skip("needs a GPU")
    entered = []
    pipelined = DecisionModel._pipelined

    def record(self, *args, **kwargs):
        entered.append(self)
        return pipelined(self, *args, **kwargs)

    monkeypatch.setattr(DecisionModel, "_pipelined", record)
    agent = Agent(long_checkpoint, **options)
    agent.forward(prepared_batch(768, [768]))
    assert entered == []
