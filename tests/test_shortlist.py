"""Offline tests for the embedding shortlist, ported from upstream v0.3.5 (#106).

A fake ``embed_fn`` supplies vectors. ``predict`` / ``system_one`` are mocks, so the
decision model is never constructed. The MLX ``embed_fn_from_agent`` is exercised against
a stub encoder and against a real tiny checkpoint.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import laya_mlx
from laya_mlx import Agent
from laya_mlx.common import render_options
from laya_mlx.shortlist import _embeddings, embed_fn_from_agent, predict_shortlist, shortlist_choice


class TableEmbed:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        missing = [t for t in texts if t not in self.vectors]
        if missing:
            raise AssertionError("unexpected texts %r" % (missing,))
        return [self.vectors[t] for t in texts]


class BoomEmbed:
    def __call__(self, texts):
        raise AssertionError("embed_fn should not run when k >= n")


class Recorder:
    """Stand-in for Agent. Records the questions handed to predict."""

    def __init__(self):
        self.calls = []
        self.system_one_calls = 0

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {"model": "fake", "answers": _answers(questions)}

    def system_one(self, state, questions, **kwargs):
        self.system_one_calls += 1
        self.calls.append((state, questions, kwargs))
        return {"model": "fake", "answers": _answers(questions)}


def _answers(questions):
    answers = {}
    for qid, qdef in questions.items():
        if isinstance(qdef, dict) and qdef.get("type") == "choice":
            crit = qdef["criteria"]
            keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            answers[qid] = {"type": "choice", "choice": keys[0]}
    return answers


CRITERIA = {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"}
# option texts follow render_options: bare key when value is None or ""
OPTION_TEXTS = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "gamma: mid": [0.6, 0.8],
    "delta: same": [1.0, 0.0],
}


def _embed_for(query_text, option_vectors):
    vectors = {query_text: [1.0, 0.0]}
    vectors.update(option_vectors)
    return TableEmbed(vectors)


# --------------------------------------------------------------------- exports


def test_exports():
    assert laya_mlx.shortlist_choice is shortlist_choice
    assert laya_mlx.predict_shortlist is predict_shortlist
    assert laya_mlx.embed_fn_from_agent is embed_fn_from_agent
    for name in ("shortlist_choice", "predict_shortlist", "embed_fn_from_agent"):
        assert name in laya_mlx.__all__
    assert Agent.predict is Agent.system_one


# --------------------------------------------------------------------- deterministic top-k


def test_topk_keeps_cosine_tie_in_input_order():
    embed = _embed_for("pay me", OPTION_TEXTS)
    assert shortlist_choice("pay me", CRITERIA, embed, k=2) == ["alpha", "delta"]
    assert len(embed.calls) == 1
    assert embed.calls[0][0] == "pay me"
    assert embed.calls[0][1:] == render_options({"t": "choice", "ins": "", "crit": CRITERIA})
    assert shortlist_choice("pay me", CRITERIA, embed, k=1) == ["alpha"]
    assert shortlist_choice("pay me", CRITERIA, embed, k=3) == ["alpha", "delta", "gamma"]


def test_topk_zero_query_keeps_original_order():
    zero_q = TableEmbed(
        {
            "pay me": [0.0, 0.0],
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma: mid": [0.6, 0.8],
            "delta: same": [3.0, 4.0],
        }
    )
    assert shortlist_choice("pay me", CRITERIA, zero_q, k=2) == ["alpha", "beta"]


def test_topk_nan_vector_sorts_behind_a_finite_match():
    nan_embed = TableEmbed(
        {"pay me": [1.0, 0.0], "alpha": [float("nan"), float("nan")], "beta": [1.0, 0.0]}
    )
    assert shortlist_choice("pay me", {"alpha": None, "beta": None}, nan_embed, k=1) == ["beta"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_embedding_sanitation_preserves_layout_and_input_storage(dtype):
    raw = np.array([[1.0, 2.0, 3.0, 4.0], [np.nan, np.inf, -np.inf, -0.0]], dtype=dtype, order="F")
    raw.flags.writeable = False
    original = raw.tobytes(order="F")

    result = _embeddings(lambda texts: raw, ["query", "option"])

    expected = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, -0.0]], order="F")
    assert result.dtype == np.float64
    assert result.flags.f_contiguous
    assert result.tobytes(order="F") == expected.tobytes(order="F")
    assert not np.shares_memory(result, raw)
    assert raw.tobytes(order="F") == original


# --------------------------------------------------------------------- list criteria and instructions


def test_list_criteria_and_instructions_change_the_query():
    list_embed = TableEmbed(
        {
            "Classify\npay me": [0.0, 1.0],
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.0, 0.2],
        }
    )
    labels = shortlist_choice(
        "pay me", ["alpha", "beta", "gamma"], list_embed, k=2, instructions="Classify"
    )
    assert labels == ["beta", "gamma"]
    assert list_embed.calls[0][0] == "Classify\npay me"
    assert list_embed.calls[0][1:] == ["alpha", "beta", "gamma"]


def test_dict_state_is_serialized():
    dict_embed = TableEmbed(
        {'Classify\n{"text": "hi"}': [1.0, 0.0], "alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
    )
    labels = shortlist_choice(
        {"text": "hi"}, ["alpha", "beta"], dict_embed, k=1, instructions="Classify"
    )
    assert labels == ["alpha"]
    assert dict_embed.calls[0][0] == 'Classify\n{"text": "hi"}'


def test_zero_and_false_stay_in_the_option_text():
    rich = {"zero": 0, "no": False, "bare": None, "named": {"desc": "payments"}}
    rich_rendered = render_options({"t": "choice", "ins": "", "crit": rich})
    rich_embed = TableEmbed({"pay me": [1.0, 0.0], **{text: [1.0, 0.0] for text in rich_rendered}})
    shortlist_choice("pay me", rich, rich_embed, k=1)
    assert rich_embed.calls[0][1:] == rich_rendered


# --------------------------------------------------------------------- k >= n pass-through


def test_passthrough_returns_every_label_without_embedding():
    boom = BoomEmbed()
    assert shortlist_choice("pay me", CRITERIA, boom, k=4) == list(CRITERIA)
    assert shortlist_choice("pay me", CRITERIA, boom, k=20) == list(CRITERIA)


# --------------------------------------------------------------------- mock predict sees only k criteria

SENTINEL = {"desc": "payments"}
FULL = {"billing": SENTINEL, "tech": "bugs", "sales": None, "other": "misc"}
FULL_VECTORS = {
    "Which desk?\nI was charged twice": [1.0, 0.0],
    'billing: {"desc": "payments"}': [0.0, 1.0],
    "tech: bugs": [1.0, 0.0],
    "sales": [0.2, 0.2],
    "other: misc": [0.0, 1.0],
}
# cosine vs [1, 0]: tech=1, sales=0.707, billing=0, other=0. k=2 -> tech, sales.
SCORE_Q = {
    "type": "score",
    "instructions": "How urgent?",
    "criteria": ["low", "mid", "high", "now"],
}
NOUL_Q = {"type": "noul", "instructions": "Is a refund requested?"}


def test_predict_shortlist_reduces_choice_and_preserves_the_rest():
    agent = Recorder()
    questions = {
        "intent": {"type": "choice", "instructions": "Which desk?", "criteria": FULL},
        "urgency": SCORE_Q,
        "refund": NOUL_Q,
        "note": "leave me alone",
    }
    state = "I was charged twice"
    result = predict_shortlist(
        agent, state, questions, TableEmbed(FULL_VECTORS), k=2, model="english"
    )

    assert len(agent.calls) == 1
    assert agent.system_one_calls == 0
    got_state, got_questions, got_kwargs = agent.calls[0]
    assert got_state is state
    assert got_kwargs == {"model": "english"}
    assert list(got_questions["intent"]["criteria"]) == ["tech", "sales"]
    assert got_questions["intent"]["criteria"]["tech"] == "bugs"
    assert got_questions["urgency"] is SCORE_Q
    assert got_questions["refund"] is NOUL_Q
    assert got_questions["note"] is questions["note"]
    # the caller's questions are not mutated
    assert questions["intent"]["criteria"] is FULL
    assert FULL["billing"] is SENTINEL

    internal = Agent._to_internal(got_questions["intent"])
    assert list(internal["crit"]) == ["tech", "sales"]
    assert result["answers"]["intent"]["choice"] == "tech"
    meta = result["shortlist"]["intent"]
    assert meta["labels"] == ["tech", "sales"]
    assert meta["scores"][0] > meta["scores"][1] > 0
    assert (meta["k"], meta["n"]) == (2, 4)
    assert meta["passthrough"] is False
    assert "urgency" not in result["shortlist"]


def test_shortlist_key_is_attached_to_a_copy():
    held = {}

    class Holding:
        def predict(self, state, questions, **kwargs):
            held["result"] = {"model": "fake", "answers": {}}
            return held["result"]

    out = predict_shortlist(
        Holding(), "pay me", {"intent": {"type": "choice", "criteria": CRITERIA}}, BoomEmbed(), k=4
    )
    assert "shortlist" in out
    assert "shortlist" not in held["result"]


def test_passthrough_reaches_predict_unchanged():
    agent = Recorder()
    original_q = {"type": "choice", "instructions": "Which desk?", "criteria": FULL}
    out = predict_shortlist(agent, "I was charged twice", {"intent": original_q}, BoomEmbed(), k=4)
    assert agent.calls[0][1]["intent"] is original_q
    meta = out["shortlist"]["intent"]
    assert meta["labels"] == list(FULL)
    assert meta["scores"] is None
    assert meta["passthrough"] is True
    out = predict_shortlist(Recorder(), "x", {"intent": original_q}, BoomEmbed(), k=99)
    assert out["shortlist"]["intent"]["passthrough"] is True


def test_list_criteria_stay_a_list_in_rank_order():
    agent = Recorder()
    list_q_embed = TableEmbed(
        {"Which?\nhello": [1.0, 0.0], "alpha": [0.0, 1.0], "beta": [1.0, 0.0], "gamma": [0.0, 0.0]}
    )
    list_questions = {
        "intent": {
            "type": "choice",
            "instructions": "Which?",
            "criteria": ["alpha", "beta", "gamma"],
        }
    }
    predict_shortlist(agent, "hello", list_questions, list_q_embed, k=2)
    received = agent.calls[0][1]["intent"]["criteria"]
    assert received == ["beta", "alpha"]
    assert list_questions["intent"]["criteria"] == ["alpha", "beta", "gamma"]
    assert list(Agent._to_internal(agent.calls[0][1]["intent"])["crit"]) == ["beta", "alpha"]


def test_system_one_used_when_predict_is_absent():
    class SystemOneOnly:
        def system_one(self, state, questions):
            self.questions = questions
            return {"answers": {"intent": {"choice": "beta"}}}

    list_q_embed = TableEmbed({"hello": [1.0, 0.0], "alpha": [0.0, 1.0], "beta": [1.0, 0.0]})
    only = SystemOneOnly()
    out = predict_shortlist(
        only,
        "hello",
        {"intent": {"type": "choice", "criteria": ["alpha", "beta"]}},
        list_q_embed,
        k=1,
    )
    assert out["answers"]["intent"]["choice"] == "beta"
    assert len(only.questions["intent"]["criteria"]) == 1


# --------------------------------------------------------------------- errors


@pytest.mark.parametrize("k", [0, -3, True, 1.5, "2"])
def test_bad_k_rejected(k):
    embed = _embed_for("pay me", OPTION_TEXTS)
    with pytest.raises(ValueError):
        shortlist_choice("pay me", CRITERIA, embed, k=k)


def test_invalid_criteria_rejected():
    embed = _embed_for("pay me", OPTION_TEXTS)
    with pytest.raises(ValueError):
        shortlist_choice("pay me", {}, embed, k=1)
    with pytest.raises(ValueError):
        shortlist_choice("pay me", [], embed, k=1)
    with pytest.raises(TypeError):
        shortlist_choice("pay me", ("alpha", "beta"), embed, k=1)
    with pytest.raises(ValueError):
        shortlist_choice("pay me", ["alpha", "alpha"], embed, k=1)


def test_missing_criteria_and_bad_questions_rejected():
    embed = _embed_for("pay me", OPTION_TEXTS)
    with pytest.raises(ValueError):
        predict_shortlist(Recorder(), "pay me", {"intent": {"type": "choice"}}, embed, k=1)
    with pytest.raises(TypeError):
        predict_shortlist(Recorder(), "pay me", [], embed, k=1)


def test_bad_embed_shape_rejected_before_predict():
    def _bad_shape(texts):
        return np.zeros((1, 4))

    agent = Recorder()
    original_q = {"type": "choice", "instructions": "Which desk?", "criteria": FULL}
    with pytest.raises(ValueError):
        predict_shortlist(agent, "pay me", {"intent": original_q}, _bad_shape, k=2)
    assert len(agent.calls) == 0


# --------------------------------------------------------------------- MLX encoder mean-pool


class TinyEncoder(nn.Module):
    """Channel 0 is the token id; channel 1 is 1. Padding must drop out of the mean."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 2})()

    def __call__(self, input_ids, attention_mask):
        ids = input_ids.astype(mx.float32)
        return mx.stack([ids, mx.ones_like(ids)], axis=-1)


class StubAgent:
    def __init__(self):
        self.encoder = TinyEncoder()
        self.model = type("M", (), {"encoder": self.encoder})()
        self.device = mx.cpu
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("decision forward must not run during embedding")


def test_embed_fn_from_agent_mean_pools_without_padding(tiny_checkpoint):
    agent = StubAgent()
    agent.tok = Agent(tiny_checkpoint).tok  # real tokenizer, fake encoder
    fn = embed_fn_from_agent(agent, max_length=32, batch_size=2)
    first = fn(["hello", "hello hello"])
    second = fn(["hello", "hello hello"])
    assert first.shape == (2, 2)
    assert np.array_equal(first, second)
    assert agent.forward_calls == 0
    # WordLevel vocab: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4 hello=5. The fixture
    # tokenizer has no post-processor, so no special tokens are added: "hello" is [5]
    # and "hello hello" is [5, 5]. In one batch row 0 is padded to length 2; a mean
    # that leaked padding would read (5+0)/2 = 2.5, so 5.0 proves padding is excluded.
    assert first[0].tolist() == [5.0, 1.0]
    assert first[1].tolist() == [5.0, 1.0]
    assert fn([]).shape == (0, 2)
    with pytest.raises(ValueError):
        embed_fn_from_agent(agent, max_length=0)
    with pytest.raises(ValueError):
        embed_fn_from_agent(agent, batch_size=True)


def test_embed_fn_from_agent_end_to_end_on_a_real_checkpoint(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, dtype="float32")
    fn = embed_fn_from_agent(agent)
    vectors = fn(["hello", "hello hello", ""])
    assert vectors.shape == (3, 64)
    assert np.isfinite(vectors).all()
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "Which?",
            "criteria": ["alpha", "beta", "gamma", "delta"],
        }
    }
    out = predict_shortlist(agent, "hello", questions, fn, k=2)
    assert out["shortlist"]["intent"]["n"] == 4
    assert len(out["shortlist"]["intent"]["labels"]) == 2
    assert out["answers"]["intent"]["choice"] in out["shortlist"]["intent"]["labels"]
