"""Laya prompt construction and calibration, adapted from upstream (see NOTICE)."""

import json
import math
from typing import Dict, List, Optional, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}
_LABELS_ERROR = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"


def resolve_noul_labels(labels=None):
    """The model-facing words for a noul's two slots, as (false, true). Upstream v0.3.20 (#156)."""
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError(_LABELS_ERROR)
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError(_LABELS_ERROR)
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError(_LABELS_ERROR)
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values
        return [
            k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
            for k, v in crit.items()
        ]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label
        + ": "
        + (
            render_criterion(false_crit)
            if false_crit not in (None, "")
            else "no, the statement does not hold"
        ),
        true_label
        + ": "
        + (
            render_criterion(true_crit)
            if true_crit not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def build_prefix(tok, q: Dict, head_max_len: int = 192, option_order=None):
    """Build the question-only prefix, before state tokens and final truncation."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    return ids, markers


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]."""
    ids, markers = build_prefix(tok, q, head_max_len, option_order)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)[
        "input_ids"
    ]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published
# as 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest
# calibration needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(hi, max(lo, t))
