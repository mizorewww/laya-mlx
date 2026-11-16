"""The score-question rule, and the answer-shape checks, shared by the guardrail wrappers."""

import math
from typing import Any, Dict, Iterable, Optional


def score_violation_probability(answer: Dict[str, Any], levels: int = 0) -> float:
    """Probability that a `score` answer is at or above the middle of its scale.

    `score` is the expected level (0..k-1), not a probability, so a guard threshold cannot be
    compared with it directly. `levels` is the number of levels the question defines; it is only
    used when the answer carries no `probabilities`, and then the normalised expected level stands
    in for the distribution.
    """
    probs = answer.get("probabilities")
    # `probabilities` is read straight from a remote or caller decision answer, so both its outer
    # shape and its values are arbitrary JSON: a non-mapping (a list, string or number) raises at
    # `len`/`.get`, and a value that is not a real number raises inside `float` -- while a boolean
    # would read as 0.0/1.0 silently because `bool` subclasses `int`, turning a spurious `true` into
    # a full-probability violation. Normalise anything that is not a non-empty mapping to the empty
    # distribution, and sum a distribution only when every level in range holds a real number;
    # otherwise fall through to the same normalised expected level the empty path already uses, so a
    # malformed body degrades the guardrail screen instead of crashing it. A well-formed numeric
    # distribution -- including numeric strings, which `float()` already accepted -- sums unchanged.
    if not isinstance(probs, dict) or not probs:
        probs = {}
    k = len(probs) or levels
    if k < 2:
        return 0.0
    if probs:
        total = 0.0
        usable = True
        for i in range(k // 2, k):
            value = probs.get(str(i), 0.0)
            if isinstance(value, bool):
                usable = False
                break
            try:
                probability = float(value)
            except (TypeError, ValueError, OverflowError):
                usable = False
                break
            if not math.isfinite(probability):
                usable = False
                break
            total += probability
        if usable and math.isfinite(total):
            return total
    # No usable distribution: stand in with the normalised expected level. `score` rides the same
    # arbitrary-answer path, so a non-number or a bool cannot be divided -- read it as 0.0 instead.
    score = answer.get("score", 0.0)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return 0.0
    try:
        normalized = score / (k - 1)
    except OverflowError:
        return 0.0
    return normalized if math.isfinite(normalized) else 0.0


#: The error every wrapper raises for a decision body it cannot read. Kept in one place so the
#: message a caller sees does not depend on which adapter they wired up.
MALFORMED_PREFIX = "Malformed decision response"


def malformed(where: str, what: str) -> RuntimeError:
    """The error an adapter raises for a body field it cannot read, in one wording."""
    return RuntimeError("%s from %s: %s" % (MALFORMED_PREFIX, where, what))


def answers_map(result: Any, where: str = "Laya") -> Dict[str, Dict[str, Any]]:
    """The `answers` mapping of a decision result, refusing anything that is not one.

    A remote decision body is arbitrary JSON, and every consumer in this package reads it as
    ``res.get("answers", {})`` and then indexes into it. The ``{}`` default only fires when the key
    is ABSENT: a body carrying ``{"answers": []}``, or an answer that is a string or ``null``, is
    handed back verbatim and the next call raises ``AttributeError``/``TypeError`` *out of the path
    whose whole job is to fail closed*. `tools.py` already refuses that shape and routes to its
    fallback tool; the guardrail wrappers had no guard at all.

    Both layers are checked here, once, so the adapters cannot drift from each other.
    """
    answers = result.get("answers", {}) if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        raise RuntimeError(
            "%s from %s: 'answers' must be a JSON object, got %s"
            % (MALFORMED_PREFIX, where, type(answers).__name__)
        )
    for qid, answer in answers.items():
        if not isinstance(answer, dict):
            raise RuntimeError(
                "%s from %s: the answer for %r must be a JSON object, got %s"
                % (MALFORMED_PREFIX, where, qid, type(answer).__name__)
            )
    return answers


def answer_number(
    answer: Dict[str, Any], field: str, where: str = "Laya", default: float = 0.0
) -> float:
    """One numeric field of an answer as a usable float, or `RuntimeError`.

    `noul`, `score` and `confidence` ride the same arbitrary-body path as `probabilities`. A string
    or `None` raises `TypeError` inside the threshold comparison, which is the crash this refuses;
    a `bool` is refused too rather than read as 1.0/0.0 (`bool` subclasses `int`), because a
    spurious `true` would otherwise become a full-probability violation; and a non-finite value is
    refused because every comparison against NaN silently answers False, i.e. "safe".
    A Python `int` may also be wider than a `float`, so `math.isfinite` raises `OverflowError` and
    the field is refused as non-finite.
    """
    value = answer.get(field, default)
    try:
        finite = (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    except OverflowError:
        finite = False
    if not finite:
        raise RuntimeError(
            "%s from %s: %r must be a finite number, got %r"
            % (MALFORMED_PREFIX, where, field, value)
        )
    return float(value)


def probability_map(
    answer: Dict[str, Any], keys: Iterable[str], where: str = "Laya"
) -> Dict[str, float]:
    """`probabilities` restricted to `keys`, each a finite number, missing keys as 0.0.

    The multi-selector sorts and thresholds these values, so a mixed dict of numbers and strings
    raised `TypeError` from `sorted`, and a `true` ranked first as 1.0. Restricting to the caller's
    own keys also means an unknown extra key cannot win the ranking.
    """
    probs: Optional[Any] = answer.get("probabilities")
    if probs is None:
        probs = {}
    if not isinstance(probs, dict):
        raise RuntimeError(
            "%s from %s: 'probabilities' must be a JSON object, got %s"
            % (MALFORMED_PREFIX, where, type(probs).__name__)
        )
    return {key: (answer_number(probs, key, where) if key in probs else 0.0) for key in keys}
