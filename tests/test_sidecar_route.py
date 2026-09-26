"""CashCard routing decision in `scripts/laya_mlx_sidecar.py`.

No model weights are loaded: `route_decision` is pure.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "laya_mlx_sidecar", Path(__file__).parents[1] / "scripts" / "laya_mlx_sidecar.py"
)
sidecar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sidecar)

Q = sidecar.CASHCARD_QUEUES


def answers(queue: str, p: float, urgent: float = 0.1) -> dict:
    rest = (1.0 - p) / (len(Q) - 1)
    return {
        "queue": {
            "choice": Q[queue],
            "probabilities": {label: (p if qid == queue else rest) for qid, label in Q.items()},
        },
        "urgent": {"noul": urgent},
    }


def test_route_questions_offer_every_queue():
    assert sidecar.ROUTE_QUESTIONS["queue"]["criteria"] == list(Q.values())
    assert "criteria" not in sidecar.ROUTE_QUESTIONS["urgent"]


def test_confident_answer_is_auto_routed():
    d = sidecar.route_decision(answers("scan", 0.9))
    assert d["queue"] == "scan" and d["auto_routed"]
    assert d["confidence"] == 0.9
    assert set(d["probabilities"]) == set(Q)


def test_low_confidence_goes_to_human_review():
    d = sidecar.route_decision(answers("sales", 0.29))
    assert d["queue"] == sidecar.REVIEW_QUEUE
    assert d["predicted_queue"] == "sales"
    assert not d["auto_routed"]


@pytest.mark.parametrize("p,routed", [(0.59, False), (0.60, True)])
def test_confidence_threshold_is_inclusive(p, routed):
    assert sidecar.route_decision(answers("topup", p))["auto_routed"] is routed


def test_refund_tag_only_when_auto_routed_to_refund():
    assert sidecar.route_decision(answers("refund", 0.8))["refund_requested"]
    assert not sidecar.route_decision(answers("refund", 0.28))["refund_requested"]
    assert not sidecar.route_decision(answers("topup", 0.9))["refund_requested"]


@pytest.mark.parametrize("p,urgent", [(0.79, False), (0.80, True), (0.95, True)])
def test_urgent_threshold(p, urgent):
    assert sidecar.route_decision(answers("scan", 0.9, urgent=p))["urgent"] is urgent


def test_thresholds_are_overridable():
    d = sidecar.route_decision(
        answers("plan", 0.5, urgent=0.5), min_confidence=0.4, urgent_threshold=0.5
    )
    assert d["auto_routed"] and d["urgent"]
