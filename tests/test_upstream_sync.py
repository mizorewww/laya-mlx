"""Regression cases for the selective post-v0.3.5 upstream sync."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from laya_mlx.agent import Agent
from laya_mlx.common import build_sequence, render_options
from laya_mlx.email import clean_email_body
from laya_mlx.prepared import PrefixCache
from laya_mlx.router import Router


class WordTokenizer:
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id = 1, 2, 3

    def __init__(self):
        self.vocab = {}
        self.calls = []

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        self.calls.append(text)
        return {
            "input_ids": [self.vocab.setdefault(w, len(self.vocab) + 100) for w in text.split()][
                : max_length if truncation else None
            ]
        }


def preparation_agent(cached):
    agent = object.__new__(Agent)
    agent.tok = WordTokenizer()
    agent.cfg = {"max_len": 48, "head_max_len": 24}
    agent._prefix_cache = PrefixCache(2) if cached else None
    return agent


Q = {"q": {"type": "choice", "instructions": "Choose", "criteria": ["yes", "no"]}}


def test_zero_room_left_truncation_keeps_separator():
    tok = WordTokenizer()
    q = Agent._to_internal(Q["q"])
    size = len(build_sequence(tok, "", q, max_len=1000)[0])
    ids, _ = build_sequence(tok, "old newest", q, max_len=size, truncate_left=True)
    assert ids[-1] == tok.sep_token_id
    assert tok.vocab["old"] not in ids
    assert tok.vocab["newest"] not in ids


@pytest.mark.parametrize("cached", [False, True])
def test_conversation_keeps_tail_and_string_keeps_head(cached):
    a = preparation_agent(cached)
    state = ["OLD " + "middle " * 100 + "NEWEST"]
    items, _ = a.prepare(state, Q)
    assert a.tok.vocab['NEWEST"]'] in items[0]["ids"]
    assert a.tok.vocab['["OLD'] not in items[0]["ids"]
    assert items[0]["state_stats"]["truncated"]
    items, _ = a.prepare(state[0], Q)
    assert a.tok.vocab["OLD"] in items[0]["ids"]
    assert a.tok.vocab["NEWEST"] not in items[0]["ids"]


@pytest.mark.parametrize("cached", [False, True])
def test_noul_labels_unicode_and_named_validation(cached):
    a = preparation_agent(cached)
    q = {
        "type": "noul",
        "instructions": {"text": "这是中文"},
        "criteria": {False: "错误", True: "正确"},
        "labels": {"false": " no ", "true": " yes "},
    }
    items, internal = a.prepare("hello", {"check": q})
    assert render_options(internal[0]) == ["no: 错误", "yes: 正确"]
    assert "这是中文" in internal[0]["ins"]
    assert q["criteria"] == {False: "错误", True: "正确"}
    q["labels"] = {"false": "negative", "true": "positive"}
    assert a.prepare("hello", {"check": q})[0][0]["ids"] != items[0]["ids"]
    for bad in [
        dict(q, criteria={"yes": "wrong"}),
        dict(q, labels={"true": "same", "false": "same"}),
        {"type": [], "instructions": "x"},
        {"type": "score", "instructions": "x", "criteria": [None, "good"]},
        dict(q, instructions=None),
        dict(q, instructions=" "),
    ]:
        with pytest.raises(ValueError, match="question 'check'"):
            a.prepare("state", {"check": bad})
    with pytest.raises(TypeError, match="state"):
        a.prepare(None, Q)


@pytest.mark.parametrize("cached", [False, True])
def test_usage_matches_each_question_budget_and_reports_option_collapse(cached):
    a = preparation_agent(cached)
    a.batch_size = 16
    a.pad_to_multiple = None
    a.tok.pad_token_id = 0
    a.temperature = [1.0] * 3
    a.temperature_by_options = {}
    a.forward = lambda batch: (
        np.zeros(batch["marker_mask"].shape),
        np.zeros((len(batch["qtype"]), 2)),
    )
    qs = {
        "q": Q["q"],
        "many": {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"shared prefix words " + str(i): None for i in range(6)},
        },
    }
    state = "word " * 100
    out = a.predict(state, qs)
    items, _ = a.prepare(state, qs)
    assert out["usage"]["state_tokens"] == 100
    assert out["usage"]["state_tokens_dropped"] == max(
        i["state_stats"]["state_tokens_dropped"] for i in items
    )
    assert out["usage"]["truncated_questions"] == ["q", "many"]
    assert out["usage"]["options"] == {"many": {"total": 6, "distinct": 1, "tokens_per_option": 4}}
    assert out["answers"]["q"]["answer_confidence"] == 0.5
    assert out["answers"]["q"]["confidence"] == 0.0
    assert a.predict("", {})["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_prefix_cache_concurrent_eviction():
    a = preparation_agent(True)
    requests = [{"q": dict(Q["q"], instructions=f"Choose {i}")} for i in range(8)]
    expected = [a.prepare("hello", q) for q in requests]

    def run(i):
        return a.prepare("hello", requests[i % len(requests)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(400)))
    assert results == [expected[i % len(expected)] for i in range(400)]
    assert len(a._prefix_cache.entries) <= 2


def test_incremental_and_empty_preload(monkeypatch):
    monkeypatch.setattr("laya_mlx.agent.Agent", lambda *args, **kwargs: object())
    r = Router()
    first = r.load("english")
    r.preload([])
    assert r.loaded == ["english"]
    r.preload(["multilingual"])
    assert set(r.loaded) == {"english", "multilingual"}
    assert r.load("english") is first


def test_undecided_uses_default_and_nested_cjk_is_seen():
    assert Router(default="multilingual").route("qwerty blorp").model == "multilingual"
    assert (
        Router().route({"body": "Hello", "custom": {"request": "请帮我取消订单"}}).model
        == "multilingual"
    )


@pytest.mark.parametrize(
    "body",
    [
        "Please send the confidential report.",
        "Hello\nThanks for the reply.\nPlease refund the charge.",
        "Hello\nFrom: my perspective this is broken.\nPlease refund the charge.",
    ],
)
def test_email_does_not_delete_requests(body):
    assert clean_email_body(body) == body


@pytest.mark.parametrize("state", ["hello " * 100, ["old " * 100, "latest message"]])
def test_experimental_prefix_ablation_preserves_runtime_contract(state):
    from experiments.snake_runtime import PrefixPreparation

    a = preparation_agent(False)
    assert PrefixPreparation(a)(state, Q) == a.prepare(state, Q)


def test_identifiers_do_not_count_as_non_english_evidence():
    assert Router().route("Please check github.com and contact user@acme.com").model == "english"
