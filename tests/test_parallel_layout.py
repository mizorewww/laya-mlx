"""Layout wiring, permutation equivariance, and parity with pinned upstream."""

import importlib.util
import itertools
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from laya_mlx import Agent
from laya_mlx.agent import collate_items
from laya_mlx.common import build_sequence, parallel_layout
from laya_mlx.convert import convert
from laya_mlx.model import DecisionModel, EncoderConfig, parallel_option_masks, sanitize_weights


class WordTokenizer:
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id, pad_token_id = 1, 2, 3, 0

    def __init__(self):
        self.words = {}

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.words.setdefault(w, len(self.words) + 10) for w in text.split()]}


def config():
    return EncoderConfig.from_dict(
        dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=4,
            num_attention_heads=4,
            local_attention=8,
            global_attn_every_n_layers=2,
        )
    )


def sequence_batch(order=None, parallel=True):
    tok = WordTokenizer()
    q = {
        "t": "choice",
        "ins": "Which department handles this request?",
        "crit": {
            "refund": "money back please",
            "tech": "a bug",
            "sales": "pricing",
            "other": "none of these fit",
        },
    }
    # Prepopulate vocab so different permutations have the same token IDs.
    build_sequence(tok, "hello", q)
    ids, markers, *layout = build_sequence(
        tok, "hello customer " * 9, q, option_order=order, return_layout=parallel
    )
    item = {"ids": ids, "markers": markers, "qtype": 0}
    if parallel:
        item["layout"] = layout[0]
    return collate_items([item], 0)


def test_layout_spans_and_clamp():
    layout = parallel_layout([4, 6, 9], 11, 14)
    assert layout == {
        "position_ids": [0, 1, 2, 3, 4, 5, 4, 5, 6, 4, 7, 8, 9, 10],
        "option_ids": [0] * 4 + [1, 1, 2, 2, 2, 3] + [0] * 4,
    }
    tok = WordTokenizer()
    q = {"t": "choice", "ins": "choose", "crit": {str(i): "long text" for i in range(8)}}
    full = build_sequence(tok, "hello", q, return_layout=True)
    cut = build_sequence(tok, "hello", q, max_len=16, return_layout=True)
    assert cut[2] == {key: values[:16] for key, values in full[2].items()}


def test_parallel_masks_isolate_options_and_use_position_distance():
    b = sequence_batch()
    masks = parallel_option_masks(
        *[mx.array(b[k]) for k in ("attention_mask", "position_ids", "option_ids")], 8
    )
    full = np.asarray(masks["full_attention"])[0, 0]
    local = np.asarray(masks["sliding_attention"])[0, 0]
    markers = b["marker_pos"][0]
    assert not full[markers[0], markers[1]]
    assert full[markers[0], 0] and full[0, markers[1]]
    pos = b["position_ids"][0]
    np.testing.assert_array_equal(local, full & (abs(pos[:, None] - pos[None, :]) <= 4))


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 2e-5), (mx.float16, 4e-3)])
def test_all_permutations_preserve_option_logits_and_action(dtype, atol):
    mx.random.seed(17)
    model = DecisionModel(config(), {"head_layers": 2, "option_layout": "parallel"})
    model.set_dtype(dtype)
    model.eval()
    # Match checkpoint loading: finish lazy random initialization before inference.
    mx.eval(model.parameters())

    def run(order, parallel):
        model.parallel = parallel
        return tuple(
            np.asarray(v)
            for v in model(**{k: mx.array(v) for k, v in sequence_batch(order, parallel).items()})
        )

    expected, action = run(None, True)
    sequential, _ = run(None, False)
    sequential_difference = 0
    for order in itertools.permutations(range(4)):
        actual, act = run(list(order), True)
        np.testing.assert_allclose(actual, expected[:, list(order)], atol=atol, rtol=atol)
        np.testing.assert_allclose(act, action, atol=atol, rtol=atol)
        seq, _ = run(list(order), False)
        sequential_difference = max(
            sequential_difference, np.max(abs(seq - sequential[:, list(order)]))
        )
    assert sequential_difference > 1e-4  # proves the permutation test is sensitive


def test_parallel_model_matches_upstream_v0412():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = Path(__file__).parents[1] / ".upstream/laya/common.py"
    if not path.exists():
        pytest.skip("Check out pinned upstream into .upstream for parallel-layout parity")
    spec = importlib.util.spec_from_file_location("upstream_v0412_common", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    cfg = transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        local_attention=8,
        global_attn_every_n_layers=2,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
    )
    torch.manual_seed(17)
    reference = upstream.DecisionModel(transformers.ModernBertModel(cfg), head_layers=2).eval()
    model = DecisionModel(
        EncoderConfig.from_dict(cfg.to_dict()),
        {"head_layers": 2, "act_costs": {"escalate": 0.5}, "option_layout": "parallel"},
    )
    model.load_weights(
        list(
            sanitize_weights(
                {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
            ).items()
        )
    )
    b = sequence_batch()
    # Add a padded row to exercise the local-mask edge cases.
    b = {k: np.repeat(v, 2, axis=0) for k, v in b.items()}
    b["attention_mask"][1, -8:] = False
    b["input_ids"][1, -8:] = 0
    b["option_ids"][1, -8:] = 0
    b["position_ids"][1, -8:] = 0
    with torch.inference_mode():
        expected = reference(**{k: torch.tensor(v) for k, v in b.items()})
    actual = model(**{k: mx.array(v) for k, v in b.items()})
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(x), y.numpy(), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_parallel_cached_preparation_and_export(
    tiny_checkpoint, tmp_path, questions, compiled, dtype
):
    p = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(p.read_text())
    cfg["option_layout"] = "parallel"
    p.write_text(json.dumps(cfg))
    # Distinct options, rather than three [UNK] spans with an exact mathematical tie:
    # tiny floating-point differences in compiled reductions can break ties differently.
    tokenizer_path = tiny_checkpoint / "tokenizer/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    vocab = tokenizer["model"]["vocab"]
    for word in ("a", "b", "c", "low", "high"):
        vocab[word] = len(vocab)
    tokenizer_path.write_text(json.dumps(tokenizer))
    plain = Agent(tiny_checkpoint, dtype=dtype)
    cached = Agent(
        tiny_checkpoint, dtype=dtype, cache_prompts=True, compile=compiled, pad_to_multiple=16
    )
    state = ["old " * 200, "hello"]
    assert plain.prepare(state, questions) == cached.prepare(state, questions)
    expected = plain.predict(state, questions)
    actual = cached.predict(state, questions)
    assert actual["usage"] == expected["usage"]
    for qid, answer in expected["answers"].items():
        got = actual["answers"][qid]
        assert got.keys() == answer.keys()
        for field, value in answer.items():
            # FP16 fusion/reduction rounding need not produce identical fourth decimals.
            tolerance = 1e-3 if dtype == "float16" else 1e-4
            if isinstance(value, float):
                assert got[field] == pytest.approx(value, abs=tolerance)
            elif field in ("action", "probabilities"):
                assert got[field] == pytest.approx(value, abs=tolerance)
            else:
                assert got[field] == value
    exported = convert(tiny_checkpoint, tmp_path / "converted", dtype=dtype)
    assert Agent(exported, dtype=dtype).predict(state, questions) == plain.predict(state, questions)
    assert (
        json.loads((exported / "rl_agent_config.json").read_text())["option_layout"] == "parallel"
    )


def test_invalid_layout_rejected_before_weights(tiny_checkpoint):
    p = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(p.read_text())
    cfg["option_layout"] = "typo"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="option_layout"):
        Agent(tiny_checkpoint)


def test_collate_rejects_mixed_layouts():
    item = {"ids": [2, 1, 3], "markers": [1], "qtype": 0}
    with pytest.raises(ValueError, match="mixed"):
        collate_items([item, dict(item, layout=parallel_layout([1], 3, 3))], 0)
