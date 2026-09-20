import importlib.util
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from laya_mlx.model import (  # noqa: E402
    DecisionModel,
    EncoderConfig,
    HeadLayer,
    ModernBert,
    attention_masks,
    sanitize_weights,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def torch_config(**kwargs):
    return transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=1,
        local_attention=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        **kwargs,
    )


@pytest.mark.parametrize("local_theta", [10000.0, 160000.0])
def test_encoder_matches_transformers_across_window_and_padding(local_theta):
    torch.manual_seed(0)
    cfg = torch_config(
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": local_theta},
        }
    )
    reference = transformers.ModernBertModel(cfg).eval()
    model = ModernBert(EncoderConfig.from_dict(cfg.to_dict()))
    model.load_weights(
        [(k, mx.array(v.detach().numpy())) for k, v in reference.state_dict().items()]
    )
    ids = np.random.default_rng(1).integers(1, 128, size=(2, 145)).astype(np.int32)
    mask = np.ones((2, 145), np.int32)
    mask[1, 13:] = 0  # padded queries beyond the local window must remain finite
    with torch.inference_mode():
        expected = reference(
            torch.tensor(ids), attention_mask=torch.tensor(mask)
        ).last_hidden_state.numpy()
    actual = np.asarray(model(mx.array(ids), mx.array(mask)))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[mask.astype(bool)], expected[mask.astype(bool)], atol=2e-5, rtol=2e-5
    )


def test_decision_head_uses_pre_norm_relu_and_biases():
    torch.manual_seed(1)
    reference = torch.nn.TransformerEncoderLayer(
        64, 1, 256, batch_first=True, norm_first=True
    ).eval()
    model = HeadLayer(64)
    model.load_weights(
        list(
            sanitize_weights(
                {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
            ).items()
        )
    )
    x = np.random.default_rng(2).normal(size=(2, 19, 64)).astype(np.float32)
    mask = np.ones((2, 19), bool)
    mask[1, 6:] = False
    with torch.inference_mode():
        expected = reference(torch.tensor(x), src_key_padding_mask=torch.tensor(~mask)).numpy()
    actual = np.asarray(model(mx.array(x), mx.array(mask[:, None, None, :])))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_local_attention_includes_both_boundary_tokens():
    mask = np.ones((1, 140), bool)
    local = np.asarray(attention_masks(mx.array(mask), 128)["sliding_attention"])[0, 0]
    assert local[70, 6] and local[70, 134]
    assert not local[70, 5] and not local[70, 135]


@pytest.mark.parametrize("head_layers", [0, 2])
def test_full_model_matches_actual_upstream(head_layers):
    path = Path(__file__).parents[1] / ".upstream/laya/common.py"
    if not path.exists():
        pytest.skip("Clone upstream into .upstream for full reference parity")
    spec = importlib.util.spec_from_file_location("upstream_common", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(3)
    cfg = torch_config()
    reference = upstream.DecisionModel(
        transformers.ModernBertModel(cfg), head_layers=head_layers
    ).eval()
    model = DecisionModel(
        EncoderConfig.from_dict(cfg.to_dict()),
        {
            "head_layers": head_layers,
            "act_costs": {"escalate": 0.5},
        },
    )
    model.load_weights(
        list(
            sanitize_weights(
                {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
            ).items()
        )
    )
    batch = {
        "input_ids": np.random.default_rng(3).integers(1, 128, (3, 75)).astype(np.int32),
        "attention_mask": np.ones((3, 75), np.int32),
        "marker_pos": np.array([[5, 9, 12], [6, 9, 0], [7, 10, 0]], np.int32),
        "marker_mask": np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], bool),
        "qtype": np.array([0, 1, 2], np.int32),
    }
    batch["attention_mask"][1, 20:] = 0
    with torch.inference_mode():
        expected = reference(**{k: torch.tensor(v) for k, v in batch.items()})
    actual = model(**{k: mx.array(v) for k, v in batch.items()})
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(x), y.numpy(), atol=3e-5, rtol=3e-5)


def test_unsupported_rope_scaling_is_rejected():
    cfg = torch_config().to_dict()
    cfg["rope_parameters"] = {"full_attention": {"rope_type": "linear", "factor": 2}}
    with pytest.raises(ValueError, match="unscaled"):
        EncoderConfig.from_dict(cfg)
