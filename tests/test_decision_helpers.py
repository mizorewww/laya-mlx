"""Public confidence gates and tournament behavior, including edge cases."""

import copy

import pytest

from laya_mlx import Agent, Router, predict_tournament
from laya_mlx.confidence import apply_confidence_gate, check_min_confidence


@pytest.mark.parametrize(
    "bad",
    [
        True,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        "0.5",
        {},
        {"choice:99": 0.8},
        {"choice:2": False},
        {"default": float("nan")},
    ],
)
def test_bad_threshold_fails_before_loading_or_inference(bad, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid thresholds must fail before model work")

    r = Router()
    monkeypatch.setattr(r, "load", unexpected)
    with pytest.raises(ValueError, match="min_confidence"):
        r.predict("hello", {}, min_confidence=bad)
    agent = object.__new__(Agent)
    monkeypatch.setattr(agent, "prepare", unexpected)
    with pytest.raises(ValueError, match="min_confidence"):
        agent.predict("hello", {}, min_confidence=bad)


def test_gate_states_per_bucket_and_re_evaluation():
    result = {
        "answers": {
            "choice": {
                "type": "choice",
                "answer_confidence": 0.7,
                "confidence": 0.01,
                "probabilities": {"a": 0.7, "b": 0.3},
            },
            "score": {
                "type": "score",
                "answer_confidence": 0.8,
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
            },
            "noul": {"type": "noul", "answer_confidence": 0.9},
            "unknown": {"type": "choice", "answer_confidence": float("nan")},
        }
    }
    thresholds = check_min_confidence({"choice:2": 0.75, "score:3-5": 0.8, "default": 0.85})
    apply_confidence_gate([result], thresholds)
    a = result["answers"]
    assert a["choice"]["abstention"] == "abstained"
    assert a["choice"]["low_confidence"] is True
    assert a["choice"]["abstention_threshold"] == 0.75
    assert a["score"]["abstention"] == "passed"
    assert a["noul"]["abstention"] == "passed"
    assert a["unknown"]["abstention"] == "unevaluated"
    apply_confidence_gate([result], 0.0)
    assert "low_confidence" not in a["choice"]
    assert a["choice"]["abstention"] == "passed"
    apply_confidence_gate([result], {"noul:2": 1.0})
    assert a["choice"]["abstention_threshold"] == 0.0
    assert a["noul"]["abstention"] == "abstained"
    a["noul"]["answer_confidence"] = 0.9999
    apply_confidence_gate([result], 0.5)
    assert "low_confidence" not in a["noul"]


def test_agent_router_gates_do_not_change_answers(tiny_checkpoint, questions):
    a = Agent(tiny_checkpoint)
    plain = a.predict("hello", questions)
    gated = a.predict("hello", questions, min_confidence=1.0)
    for key in questions:
        assert "abstention" not in plain["answers"][key]
        assert gated["answers"][key]["abstention"] == "abstained"
        for field, value in plain["answers"][key].items():
            assert gated["answers"][key][field] == value
    r = Router()
    r.attach("mine", a)
    routed = r.predict("hello", questions, model="mine", min_confidence=1.0)
    assert routed["answers"] == gated["answers"]
    assert a.predict("hello", {}, min_confidence=0.5)["answers"] == {}


class HighestLabel:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, copy.deepcopy(questions), kwargs))
        answers = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                labels = list(q["criteria"])
                answers[key] = {
                    "choice": max(labels, key=int),
                    "probabilities": dict.fromkeys(labels, 1 / len(labels)),
                }
            else:
                answers[key] = {"noul": 0.5}
        return {"answers": answers, "usage": {"input_tokens": len(questions)}}


@pytest.mark.parametrize("as_list", [False, True])
def test_tournament_multiple_rounds_preserves_inputs_and_forwards_kwargs(as_list):
    labels = [str(i) for i in range(20)]
    criteria = labels if as_list else {key: "description " + key for key in labels}
    qs = {
        "large": {"type": "choice", "instructions": "choose", "criteria": criteria},
        "small": {"type": "choice", "criteria": ["1", "2"]},
        "flag": {"type": "noul"},
    }
    before = copy.deepcopy(qs)
    a = HighestLabel()
    out = predict_tournament(a, "state", qs, group_size=3, model="custom", min_confidence=0.7)
    assert qs == before
    assert out["answers"]["large"]["choice"] == "19"
    assert out["tournament"]["large"]["rounds"] == 2
    assert out["tournament"]["large"]["n"] == 20  # original label count
    assert len(out["tournament"]["large"]["labels"]) == 3
    assert out["tournament"]["small"]["rounds"] == 0
    assert set(out["answers"]["large"]["probabilities"]) == set(
        out["tournament"]["large"]["labels"]
    )
    assert out["usage"] == {"input_tokens": 3}  # final pass only, per upstream
    assert len(a.calls) == 3
    for state, questions, kwargs in a.calls:
        assert state == "state" and kwargs == {"model": "custom", "min_confidence": 0.7}
        for q in questions.values():
            if q["type"] == "choice":
                assert 1 <= len(q["criteria"]) <= 3


@pytest.mark.parametrize("size", [True, 1, 0, 2.5, "16"])
def test_tournament_invalid_group(size):
    with pytest.raises(ValueError, match="group_size"):
        predict_tournament(HighestLabel(), "hello", {}, group_size=size)


def test_tournament_small_and_empty_requests_are_one_call():
    for qs in ({}, {"x": {"type": "choice", "criteria": ["1"]}}):
        a = HighestLabel()
        out = predict_tournament(a, "hello", qs)
        assert len(a.calls) == 1
        assert out["answers"] == a.predict("hello", qs)["answers"]


@pytest.mark.parametrize("state", ["12345", "qwerty blorp", "refund me", "MON DES EST LA"])
def test_undecided_default_and_legacy_override(state):
    assert Router().route(state).model == "multilingual"
    assert Router(default="english").route(state).model == "english"


def test_capitalized_latin_does_not_outvote_another_script():
    assert Router().route("ПРИВЕТ MON DES EST LA").model == "multilingual"


@pytest.mark.parametrize(
    "footer", ["Envoyé depuis mon iPhone", "Envoyée de ma tablette", "Envoyé de iPad"]
)
def test_french_device_footer(footer):
    from laya_mlx.email import clean_email_body

    # 'tablette' is not a recognized device: keep unrecognized text intact.
    body = "Bonjour, pouvez-vous annuler ma commande ?"
    expected = body + "\n" + footer if "tablette" in footer else body
    assert clean_email_body(body + "\n" + footer) == expected
