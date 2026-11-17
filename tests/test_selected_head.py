"""Selecting final-head outputs preserves the full sequence computation."""

import mlx.core as mx
import numpy as np
import pytest

from laya_mlx.model import DecisionHead


@pytest.mark.parametrize("layers", [0, 1, 2])
@pytest.mark.parametrize("dtype,tolerance", [(mx.float32, 2e-5), (mx.float16, 4e-3)])
@pytest.mark.parametrize("compiled", [False, True])
def test_selected_head_matches_full_outputs(layers, dtype, tolerance, compiled):
    mx.random.seed(31)
    head = DecisionHead(64, layers)
    head.set_dtype(dtype)
    head.eval()
    mx.eval(head.parameters())
    x = mx.array(np.random.default_rng(31).normal(size=(2, 19, 64)), dtype=dtype)
    mask = np.ones((2, 1, 1, 19), dtype=bool)
    mask[1, :, :, 8:] = False
    mask = mx.array(mask)
    # Includes CLS, repeated positions, and zero-filled padded marker slots.
    positions = mx.array([[0, 5, 9, 9], [0, 6, 0, 0]])
    full = head(x, mask)
    expected = full[mx.arange(2)[:, None], positions]
    fn = mx.compile(head) if compiled else head
    actual = fn(x, mask, positions)
    mx.eval(expected, actual)
    assert actual.shape == (2, 4, 64)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=tolerance, rtol=tolerance
    )
