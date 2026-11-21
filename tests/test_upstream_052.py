"""v0.5.2 boundary regressions that run without native MLX or model weights."""

import pytest

import laya_mlx


@pytest.mark.parametrize("name", ["Escalator", "vllm_teacher"])
def test_vegamls_teacher_is_not_exported_here(name):
    # Upstream 0.5.2 re-exports vegaml's teacher for its learner; both read with vegaml's
    # reader, not the MLX encoder, so neither is part of this distribution.
    assert name not in laya_mlx.__all__
    with pytest.raises(AttributeError):
        getattr(laya_mlx, name)
