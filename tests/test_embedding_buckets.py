"""Length bucketing preserves row identity and bounds padding without changing defaults."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from laya_mlx import Agent
from laya_mlx.shortlist import embed_fn_from_agent


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_bucketed_embeddings_preserve_rows_and_reduce_padding(tiny_checkpoint, batch_size):
    agent = Agent(tiny_checkpoint)
    shapes = []

    class Encoder:
        config = SimpleNamespace(hidden_size=2)

        def __call__(self, input_ids, attention_mask):
            shapes.append(input_ids.shape)
            ids = input_ids.astype(mx.float32)
            return mx.stack([ids, ids * ids], axis=-1)

    agent.model.encoder = Encoder()
    texts = ["hello " * 100, "other", "", "hello other", None, "hello", "hello other"]
    baseline = embed_fn_from_agent(agent, max_length=32, batch_size=batch_size)
    grouped = embed_fn_from_agent(
        agent, max_length=32, batch_size=batch_size, bucket_by_length=True
    )
    expected = baseline(texts)
    default_shapes = list(shapes)
    assert default_shapes[0][1] == 32  # The default still starts with the long input.
    shapes.clear()
    np.testing.assert_array_equal(grouped(texts), expected)
    assert all(rows <= batch_size and width <= 32 for rows, width in shapes)
    old_work = sum(rows * width for rows, width in default_shapes)
    new_work = sum(rows * width for rows, width in shapes)
    assert new_work <= old_work
    if batch_size > 1:
        assert new_work < old_work
    shapes.clear()
    assert grouped([]).shape == (0, 2)
    assert not shapes


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_bucketed_embeddings_match_small_model_with_rounding(tiny_checkpoint, dtype):
    agent = Agent(tiny_checkpoint, dtype=dtype)
    texts = ["hello " * 100, "other", "hello other", "hello", "other " * 12]
    expected = embed_fn_from_agent(agent, max_length=32, batch_size=4)(texts)
    actual = embed_fn_from_agent(agent, max_length=32, batch_size=4, bucket_by_length=True)(texts)
    assert np.isfinite(actual).all()
    if dtype == "float32":
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    else:
        cosine = (actual * expected).sum(1) / (
            np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
        )
        assert np.min(cosine) > 0.9999
