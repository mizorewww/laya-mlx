"""Native row grouping reduces padding without changing answer attribution."""

import json

import numpy as np
import pytest

from laya_mlx.agent import Agent, collate_items


@pytest.mark.parametrize(
    "lengths,multiple,expected,row_order",
    [
        ([4, 20, 6, 18], None, [6, 20], [1, 3, 4, 2]),
        ([4, 4, 4, 4], None, [4, 4], [1, 2, 3, 4]),
        ([4, 20], None, [20], [1, 2]),
        ([5, 8, 6, 7], 8, [8, 8], [1, 2, 3, 4]),
        ([4, 20, 6, 18, 8], None, [6, 18, 20], [1, 3, 5, 4, 2]),
    ],
)
def test_grouping_restores_logits_actions_and_parallel_rows(lengths, multiple, expected, row_order):
    agent = object.__new__(Agent)
    agent.batch_size = 2
    agent.pad_to_multiple = multiple
    agent.cfg = {"max_len": 32}
    agent.tok = type("Tokenizer", (), {"pad_token_id": 0})()
    items = []
    for row, length in enumerate(lengths):
        items.append(
            {
                "ids": [row + 1] * length,
                "markers": [1, 2],
                "qtype": row % 3,
                "layout": {"position_ids": list(range(length)), "option_ids": [row + 1] * length},
            }
        )
    batch = collate_items(items, 0)
    calls = []
    seen = []

    def forward(arrays):
        calls.append(arrays["input_ids"].shape[1])
        tags = arrays["input_ids"][:, 0]
        seen.extend(tags.tolist())
        for i, tag in enumerate(tags):
            n = lengths[tag - 1]
            assert arrays["qtype"][i] == (tag - 1) % 3
            np.testing.assert_array_equal(arrays["position_ids"][i, :n], np.arange(n))
            np.testing.assert_array_equal(arrays["option_ids"][i, :n], np.full(n, tag))
        return np.column_stack([tags, -tags]), np.column_stack([tags, np.zeros(len(tags))])

    agent.forward = forward
    before = {k: v.copy() for k, v in batch.items()}
    logits, actions = agent._forward(batch)
    tags = np.arange(1, len(lengths) + 1)
    np.testing.assert_array_equal(logits, np.column_stack([tags, -tags]))
    np.testing.assert_allclose(actions[:, 0], 1 / (1 + np.exp(-tags)))
    assert calls == expected
    assert seen == row_order
    for key in batch:
        np.testing.assert_array_equal(batch[key], before[key])


@pytest.mark.parametrize("parallel", [False, True])
def test_grouped_native_outputs_match_single_forward(tiny_checkpoint, parallel):
    # Distinct option tokens: all-[UNK] options tie exactly under parallel
    # attention, and harmless batch-shape rounding can change their argmax.
    tokenizer_path = tiny_checkpoint / "tokenizer/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    vocab = tokenizer["model"]["vocab"]
    for word in ("x", "y", "z", "low", "high"):
        vocab[word] = len(vocab)
    tokenizer_path.write_text(json.dumps(tokenizer))
    if parallel:
        path = tiny_checkpoint / "rl_agent_config.json"
        config = json.loads(path.read_text())
        config["option_layout"] = "parallel"
        path.write_text(json.dumps(config))
    agent = Agent(tiny_checkpoint, dtype="float32", batch_size=4)
    questions = {
        "a": {"type": "noul", "instructions": "hello"},
        "b": {"type": "choice", "instructions": "hello " * 20, "criteria": ["x", "y", "z"]},
        "c": {"type": "score", "instructions": "hello hello", "criteria": ["low", "high"]},
        "d": {"type": "noul", "instructions": "hello " * 18},
    }
    expected = agent.predict("hello", questions)
    agent.batch_size = 2
    actual = agent.predict("hello", questions)
    assert actual["usage"] == expected["usage"]
    assert list(actual["answers"]) == list(expected["answers"])
    for key, answer in expected["answers"].items():
        other = actual["answers"][key]
        for field, value in answer.items():
            if isinstance(value, float) or field in ("probabilities", "action"):
                assert other[field] == pytest.approx(value, abs=1e-4)
            else:
                assert other[field] == value
