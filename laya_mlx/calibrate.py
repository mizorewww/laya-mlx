"""Fit and persist per-bucket temperatures for Laya logits.

Ports the notebook LBFGS fitter (NLL on softmax(z/T)) and groups records by
`temp_bucket(qtype, k)` so the runtime's `temperature_by_options` lookup has a writer.
Fitting is CPU-side: records are `(qtype, logits, target, k)` and do not load Hub weights.

Two sample floors, on purpose:

- `MIN_BUCKET_N` is the per-bucket floor. Buckets with fewer examples are omitted from
  `temperature_by_options`; the type-level scalar covers them at lookup time.
- `MIN_TYPE_N` is the lower floor used only for that type-level scalar. It stays at the
  previous floor of 10 so a realistic dataset that fills no single bucket still gets a
  temperature instead of remaining at 1.0. Do not raise it to `MIN_BUCKET_N`.

Fitted temperatures go through `clamp_temperature` and therefore `[TEMP_MIN, TEMP_MAX]`
from `laya.common`, the same guard checkpoint load uses. There is no second pair of
bounds in this module. A clamp that moved a fit is reported in `fit_temperature_map`'s
`temperature_clamped`, since the clamped value alone cannot say whether the optimum was
the bound or merely beyond it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

import sys
import warnings
from numbers import Integral
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .common import (
    QTYPE_NAMES,
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    clamp_temperature,
    collate_items,
    ece_score,
    temp_bucket,
    temperature_rejected,
)

# Per-bucket floor. Below this, `temperature_by_options` omits the bucket.
MIN_BUCKET_N = 2000
# Type-level floor only. Lower than MIN_BUCKET_N; see the module docstring.
MIN_TYPE_N = 10
# Fraction of each bucket held out when `compute_ece=True`. Unused otherwise.
ECE_HOLDOUT_FRAC = 0.2
# Log-space search margin, in e-folds, outside the `[TEMP_MIN, TEMP_MAX]` answer band.
# The optimiser runs on log T with no bound of its own; any scale this far below
# `TEMP_MIN` already clamps to `TEMP_MIN`, so the band costs no real minimum, while an
# unbounded walk reaches `exp()` underflow (~-88 in float32), divides the objective by
# zero, and hands `clamp_temperature` a NaN -- which it reports as the neutral 1.0.
LOG_T_MARGIN = 10.0
# Payload schema. Files written before identity was recorded have no version key;
# readers treat that absence as version 1.
CALIBRATION_VERSION = 2
N_QTYPES = len(QTYPES)

# (qtype:int, logits:1d, target:1d, k:int)
Record = Tuple[int, Any, Any, int]

_CONFIG_IDENTITY_SKIP = ("temperature", "temperature_by_options")


def _vec(x) -> np.ndarray:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and isinstance(x, torch_mod.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _validated_pair(logits, target, k=None):
    torch_mod = sys.modules.get("torch")
    vectors = []
    for name, value in (("logits", logits), ("target", target)):
        if torch_mod is not None and isinstance(value, torch_mod.Tensor):
            value = value.detach().cpu().numpy()
        try:
            vector = np.asarray(value)
            if vector.ndim != 1 or vector.dtype.kind not in "iuf":
                raise ValueError("%s must be a one-dimensional numeric vector" % name)
            with np.errstate(over="ignore", invalid="ignore"):
                vector = vector.astype(np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("%s must be a one-dimensional numeric vector" % name) from exc
        if not np.isfinite(vector).all():
            raise ValueError("%s must contain only finite values" % name)
        vectors.append(vector)
    logits, target = vectors
    if len(logits) != len(target):
        raise ValueError("logits and target must have the same length")
    if k is None:
        k = len(logits)
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1 or k > len(logits):
        raise ValueError("k must be an integer between 1 and the vector length")
    logits, target = logits[:k], target[:k]
    if (target < 0).any() or not np.isclose(target.sum(), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("target must be a nonnegative probability vector summing to 1")
    return logits, target, int(k)


def _positive_int(name: str, value: int) -> int:
    """A count parameter: an integer >= 1, normalised to a Python `int`, or a `ValueError` naming it.

    This is the rule the module already applies everywhere a count is checked -- `min_n` in
    :func:`fit_one_temperature`, `k` in :func:`_validated_pair` (which returns `int(k)` for the same
    reason), and the `"bins"` of every `binning_map` entry :func:`apply_calibration_payload` reads --
    applied to the two fitting parameters that were checked with less. The asymmetry was not
    cosmetic: a `bool` is an `Integral`, so `bins` computed as a comparison (`len(bucket) > 15`)
    passed the old `bins < 1` guard and emitted `{"bins": True, ...}`, a map the loader refuses on
    its boolean arm. And a `numpy` integer is an `Integral` but not a Python `int`, so a `bins`
    derived from an array shape emitted a map `Agent.save_calibration`'s own `json.dump` cannot write
    -- a fit that produced an artifact the file format rejects. `int()` keeps that second case
    working. A float, a string or `None` is a mistake, and now names itself instead of raising
    `TypeError` from `range()` (or, for a `min_bucket_n` that compares `nan` to a length, silently
    disabling the sample floor the parameter exists to enforce).
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("%s must be a positive integer, got %r" % (name, value))
    return int(value)


def _numpy_pairs_by_width(pairs: Sequence) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for z, t in pairs:
        groups.setdefault(len(_vec(z)), []).append((_vec(z), _vec(t)))
    return [
        (
            np.stack([z for z, _ in groups[w]]),
            np.stack([t for _t, t in groups[w]]),
        )
        for w in sorted(groups)
    ]


def _pairs_by_width(pairs: Sequence) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """One (logits, target) matrix per distinct option count, narrowest first.

    A ragged pool -- a question type holding 2-option and 12-option decisions -- cannot be
    one matrix. Any value used to square off the short rows is a real column in the single
    softmax every row then goes through, so it competes for the probability mass the fit is
    trying to allocate; at `-1e4` it wins the row it pads once the row's own logits drop
    below it, and it overflows to `inf`/`NaN` in float32 when the scale gets small. Grouping
    keeps each row inside its own option count, which is what the objective means.
    """
    import torch

    return [(torch.from_numpy(z), torch.from_numpy(t)) for z, t in _numpy_pairs_by_width(pairs)]


def _fit_unclamped_temperature_numpy(
    groups: List[Tuple[np.ndarray, np.ndarray]], n_rows: float, log_lo: float, log_hi: float
) -> float:
    log_t = 0.0
    for _ in range(100):
        u = float(np.clip(log_t, log_lo, log_hi))
        scale = np.float32(np.exp(u))
        grad = 0.0
        hess = 0.0
        for z, t in groups:
            s = z / scale
            shifted = s - s.max(axis=-1, keepdims=True)
            exp_s = np.exp(shifted)
            p = exp_s / exp_s.sum(axis=-1, keepdims=True)
            g_row = ((t - p) * s).sum(axis=-1)
            s_mean = (p * s).sum(axis=-1, keepdims=True)
            var_s = (p * (s - s_mean) ** 2).sum(axis=-1)
            grad += float(g_row.sum())
            hess += float((var_s - g_row).sum())
        grad /= n_rows
        hess /= n_rows
        if not np.isfinite(grad) or not np.isfinite(hess):
            return float("nan")
        if abs(grad) < 1e-7:
            break
        step = grad / max(abs(hess), 1e-4)
        log_t = float(np.clip(u - np.clip(step, -1.0, 1.0), log_lo, log_hi))
    return float(np.exp(np.clip(log_t, log_lo, log_hi)))


def _fit_unclamped_temperature(pairs: Sequence, min_n: Optional[int] = None) -> float:
    """The optimised scale before `clamp_temperature`, for callers that report the clamp.

    `fit_one_temperature` is this value clamped, and that is the value anything installs.
    Only the clamp reporting in `_fit_groups` reads the raw scale, so that a bucket whose
    optimum sits outside `[TEMP_MIN, TEMP_MAX]` can be told apart from one whose optimum
    really is a bound. The search band is `LOG_T_MARGIN` e-folds wider than the answer
    band, so a raw value can be far outside it; a NaN from the underflow case described in
    `fit_one_temperature` reaches the caller as a NaN rather than as 1.0.
    """
    if min_n is None:
        min_n = MIN_BUCKET_N
    if isinstance(min_n, bool) or not isinstance(min_n, Integral) or min_n < 1:
        raise ValueError("min_n must be a positive integer")
    sel = []
    for pair in pairs:
        try:
            z, t = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("pair must be (logits, target)") from exc
        z, t, _k = _validated_pair(z, t)
        sel.append((z, t))
    if len(sel) < min_n:
        return 1.0
    n_rows = float(len(sel))
    log_lo = float(np.log(TEMP_MIN)) - LOG_T_MARGIN
    log_hi = float(np.log(TEMP_MAX)) + LOG_T_MARGIN
    try:
        import torch
    except ModuleNotFoundError:
        return _fit_unclamped_temperature_numpy(_numpy_pairs_by_width(sel), n_rows, log_lo, log_hi)
    groups = _pairs_by_width(sel)
    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            scale = log_t.clamp(min=log_lo, max=log_hi).exp()
            nll = sum(-(t * torch.log_softmax(z / scale, -1)).sum(-1).sum() for z, t in groups)
            loss = nll / n_rows
            loss.backward()
            return loss

        opt.step(closure)
    return float(log_t.detach().clamp(min=log_lo, max=log_hi).exp().item())


def fit_one_temperature(pairs: Sequence, min_n: Optional[int] = None) -> float:
    """Fit one scalar T by NLL + LBFGS on log T.

    The result is `clamp_temperature` of the optimised scale, so it lies in
    `[TEMP_MIN, TEMP_MAX]` (or is the neutral 1.0 when the value is not a number).
    The search is bounded to `LOG_T_MARGIN` e-folds outside that band, which covers any
    minimum the answer can use; past it the float32 scale drifts toward `exp()` underflow,
    where the objective divides by zero and every gradient becomes NaN, and
    `clamp_temperature` reads that NaN back as the neutral 1.0.
    Returns 1.0 when fewer than `min_n` pairs are given. `min_n` defaults to
    `MIN_BUCKET_N` (the per-bucket floor). Type-level fits pass `MIN_TYPE_N`, which
    is lower, so a dataset that fills no bucket still gets a scalar instead of
    staying at 1.0.

    A clamped result is indistinguishable from an optimum that really is a bound, because
    both are the bound. `fit_temperature_map`'s `temperature_clamped` is where the
    difference is reported; it carries the unclamped scale beside the value installed.
    """
    return clamp_temperature(_fit_unclamped_temperature(pairs, min_n), TEMP_MIN, TEMP_MAX)


def _iter_records(records: Iterable) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    out = []
    for rec in records:
        if not isinstance(rec, (list, tuple)) or len(rec) not in (3, 4):
            raise ValueError("record must be (qtype, logits, target[, k])")
        if len(rec) == 4:
            qtype, logits, target, k = rec
            if k is None:
                raise ValueError("k must be an integer between 1 and the vector length")
        else:
            qtype, logits, target = rec
            k = None
        if isinstance(qtype, bool) or not isinstance(qtype, Integral) or not 0 <= qtype < N_QTYPES:
            raise ValueError("qtype must be an integer between 0 and %d" % (N_QTYPES - 1))
        logits, target, k = _validated_pair(logits, target, k)
        out.append((int(qtype), logits, target, k))
    return out


def _softmax(z, t_scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t_scale))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _ece_report(recs, temperature, temperature_by_options) -> Dict[str, Any]:
    conf_b, cor_b, conf_a, cor_a = [], [], [], []
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        p0 = _softmax(z[:k], 1.0)
        conf_b.append(float(p0.max()))
        cor_b.append(float(int(p0.argmax()) == y))
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p1 = _softmax(z[:k], t_scale)
        conf_a.append(float(p1.max()))
        cor_a.append(float(int(p1.argmax()) == y))
    return {
        "ece_before": ece_score(np.asarray(conf_b), np.asarray(cor_b)),
        "ece_after": ece_score(np.asarray(conf_a), np.asarray(cor_a)),
        "n": float(len(recs)),
    }


def _group_records(recs):
    by_type = {qt: [] for qt in range(N_QTYPES)}
    by_bucket = {}
    for qt, z, t, k in recs:
        if qt in by_type:
            by_type[qt].append((z, t))
        key = temp_bucket(qt, k)
        by_bucket.setdefault(key, []).append((z, t))
    return by_type, by_bucket


def _ece_split(recs, seed):
    """Split records into fit and eval, stratified by `temp_bucket`.

    Returns `(fit_recs, eval_recs, excluded_buckets)`. The split is a deterministic
    partition (`numpy.random.RandomState(seed)`): every record is in exactly one side.
    A bucket whose fit side would fall below `MIN_BUCKET_N` is not split; all of its
    records stay in `fit_recs` and its key is listed in `excluded_buckets`.
    """
    by_bucket = {}
    for i, rec in enumerate(recs):
        qt, _z, _t, k = rec
        by_bucket.setdefault(temp_bucket(qt, k), []).append(i)

    rng = np.random.RandomState(seed)
    fit_idx = []
    eval_idx = []
    excluded = []
    for key in sorted(by_bucket):
        idxs = by_bucket[key]
        n = len(idxs)
        n_eval = int(round(n * ECE_HOLDOUT_FRAC))
        if n - n_eval < MIN_BUCKET_N:
            fit_idx.extend(idxs)
            excluded.append(key)
            continue
        perm = rng.permutation(n)
        for j in perm[:n_eval]:
            eval_idx.append(idxs[int(j)])
        for j in perm[n_eval:]:
            fit_idx.append(idxs[int(j)])

    fit_idx.sort()
    eval_idx.sort()
    fit_recs = [recs[i] for i in fit_idx]
    eval_recs = [recs[i] for i in eval_idx]
    return fit_recs, eval_recs, excluded


def _clamp_entry(raw: float, applied: float) -> Optional[Dict[str, Any]]:
    """Describe a fit the clamp moved, or None when it left the fit alone.

    A NaN raw scale is the underflow case, not a clamp: `clamp_temperature` answers the
    neutral 1.0 for it, which is a different fact and is not reported here.
    """
    if not np.isfinite(raw) or TEMP_MIN <= raw <= TEMP_MAX:
        return None
    return {"fitted": raw, "applied": applied, "bound": "min" if raw < TEMP_MIN else "max"}


def _fit_groups(by_type, by_bucket):
    temperature = [1.0] * N_QTYPES
    clamped: Dict[str, Dict[str, Any]] = {}
    for qt in range(N_QTYPES):
        pairs = by_type[qt]
        # MIN_TYPE_N, not MIN_BUCKET_N: the scalar has to exist when every bucket is small.
        if len(pairs) >= MIN_TYPE_N:
            raw = _fit_unclamped_temperature(pairs, min_n=MIN_TYPE_N)
            temperature[qt] = clamp_temperature(raw, TEMP_MIN, TEMP_MAX)
            entry = _clamp_entry(raw, temperature[qt])
            if entry is not None:
                clamped[QTYPE_NAMES[qt]] = entry

    temperature_by_options = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < MIN_BUCKET_N:
            continue
        raw = _fit_unclamped_temperature(pairs)
        temperature_by_options[key] = clamp_temperature(raw, TEMP_MIN, TEMP_MAX)
        entry = _clamp_entry(raw, temperature_by_options[key])
        if entry is not None:
            clamped[key] = entry
    return temperature, temperature_by_options, clamped


def fit_temperature_map(
    records: Iterable, compute_ece: bool = False, seed: int = 0
) -> Dict[str, Any]:
    """Fit type-level scalars and per-bucket temperatures.

    `MIN_BUCKET_N` is the per-bucket floor: smaller buckets are omitted from
    `temperature_by_options` and the type-level scalar covers them. `MIN_TYPE_N` is
    the separate, lower floor for that scalar only.

    `compute_ece=False` (the default, and the path `Agent.fit_temperatures` stores)
    fits on every record and returns no `report` key. `seed` is ignored on this path.

    `compute_ece=True` holds out `ECE_HOLDOUT_FRAC` of each bucket, stratified by
    `temp_bucket`, using `seed` so the same records always split the same way. Temperatures
    are fit on the remainder only and ECE is scored only on the held-out records.
    `seed` is this module's own count rule -- an integer in `[0, 2**32-1]`, which is what
    `numpy.random.RandomState` can take -- so a `10.0`, a `"7"`, a `True` or an out-of-range
    value is a `ValueError` naming `seed` on either path, not numpy's cast error on one path
    and silence on the other. `seed=None` stays numpy's entropy seed: accepted, and the split
    is then not reproducible.
    `temperature_clamped` names every fit the clamp moved, on either path: a bucket key as
    `temperature_by_options` spells it, or a question-type name for one of the `temperature`
    scalars. Each entry carries `fitted` (the unclamped optimum), `applied` (the value in
    this result) and `bound`, `"min"` or `"max"`. It is empty when nothing was clamped, and
    it reports a fact a caller cannot otherwise recover: a value of exactly `TEMP_MIN` or
    `TEMP_MAX` is what both a clamped fit and a genuine fit on the bound look like, so
    without this an optimum of 0.387 installs as 0.5 and reads as a finished fit. The
    clamped value is still the one to install, because `load_calibration` applies the same
    bounds at checkpoint load; what the entry adds is that the data wanted to go further.

    `report["n"]` is the number of records passed in; `report["n_eval"]` is the held-out
    count the ECE rests on. A bucket that would drop below `MIN_BUCKET_N` after the
    holdout is fit on all of its records, left out of the eval set, and named in
    `report["buckets_excluded_from_eval"]` instead of being dropped. `n_by_bucket`
    always counts the full input, including when the fit itself used a subset.
    """
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be None or an integer in [0, 2**32-1], got %r" % (seed,))
    recs = _iter_records(records)
    all_by_type, all_buckets = _group_records(recs)
    n_by_bucket = {key: len(pairs) for key, pairs in all_buckets.items()}

    if compute_ece:
        _fit_recs, eval_recs, excluded = _ece_split(recs, seed)
        by_type, by_bucket = _group_records(_fit_recs)
    else:
        eval_recs = None
        excluded = []
        by_type, by_bucket = all_by_type, all_buckets

    temperature, temperature_by_options, clamped = _fit_groups(by_type, by_bucket)
    out: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
        "temperature_clamped": clamped,
    }
    if compute_ece:
        report = _ece_report(eval_recs, temperature, temperature_by_options)
        # `n` stays the size of the input. `n_eval` is the held-out count ECE used.
        report["n"] = float(len(recs))
        report["n_eval"] = float(len(eval_recs))
        report["buckets_excluded_from_eval"] = excluded
        out["report"] = report
    return out


fit_temperatures = fit_temperature_map


# Per-bucket floor for abstention thresholds. Lower than `MIN_BUCKET_N` (which governs temperature
# fitting): a threshold is a single order statistic of the calibrated confidences, so it stabilises
# on far fewer examples than an LBFGS temperature does. Buckets below this are omitted, and only the
# map's own "default" entry covers them afterwards -- `resolve_min_confidence` falls from "default"
# to 0.0, which gates nothing. There is no scalar behind that fall-through, unlike a temperature
# bucket omitted from `temperature_by_options`, whose type-level scalar really is the fallback: one
# call's `min_confidence` is either a float or a map, never both, so no caller's float is in scope
# here. A map produced with this floor should name a "default" if any bucket under it must abstain.
MIN_ABSTAIN_BUCKET_N = 100


def _select_abstention_threshold(
    pairs: Sequence[Tuple[float, int]], target_error: float, conservative: bool
) -> float:
    """Smallest confidence `tau` whose accepted set (`conf >= tau`) keeps error <= `target_error`.

    `pairs` are `(confidence, correct)`. Sweeping from the most confident down maximises coverage
    at the target risk (the selective-classification / split-conformal cut). `conservative` adds one
    pseudo-error so a bucket does not clear the gate on a lucky short run. Returns 1.0 when no cut
    holds the risk -- the bucket is too unreliable to accept anything short of a reported certainty.
    """
    ordered = sorted(pairs, key=lambda cc: cc[0], reverse=True)
    levels = sorted({c for c, _ in ordered}, reverse=True)
    n = len(ordered)
    n_acc = n_err = idx = 0
    best = None
    # Evaluate the error over the WHOLE accepted set {conf >= tau} at each distinct level, not
    # incrementally within a tie: a threshold accepts every answer at its own confidence, so a tie
    # group's errors must all be counted before the level is judged (else the cut sinks into a bad
    # cohort on its first few correct members).
    for tau in levels:
        while idx < n and ordered[idx][0] >= tau:
            n_acc += 1
            n_err += 0 if ordered[idx][1] else 1
            idx += 1
        rate = (n_err + 1.0) / (n_acc + 1.0) if conservative else (n_err / n_acc)
        if n_acc > 0 and rate <= target_error:
            best = tau
    if best is None:
        return 1.0
    return float(min(1.0, max(0.0, best)))


def fit_abstention_thresholds(
    records: Iterable,
    temperature: Sequence[float],
    temperature_by_options: Dict[str, float],
    *,
    binning_map: Optional[Dict[str, Dict[str, Any]]] = None,
    target_error: float = 0.10,
    min_bucket_n: int = MIN_ABSTAIN_BUCKET_N,
    conservative: bool = True,
) -> Dict[str, float]:
    """Fit a per-`temp_bucket` abstention threshold so a gate keeps a target error in each fitted bucket.

    A single `min_confidence` does not transfer across option counts (#394): the calibrated
    confidence of a 2-option and a 12-option answer live on different scales, so one cut over- or
    under-abstains depending on the question. This fits one cut per bucket instead, keyed exactly
    like `temperature_by_options` (`common.temp_bucket`, e.g. ``"choice:3-5"``), and the result is a
    `min_confidence` map that :func:`laya.confidence.check_min_confidence` /
    :func:`laya.confidence.apply_confidence_gate` accept directly -- with one exception to keep:
    every bucket below `min_bucket_n` is omitted, so a calibration set too small to fill a single
    bucket returns ``{}``, and the validator refuses an empty map ("must be a non-empty dict"). Pass
    no gate in that case rather than passing what came back; `laya.train` writes the map into a
    checkpoint only `if abstention_thresholds:`.

    `records` are the same `(qtype, logits, target[, k])` tuples `fit_temperature_map` consumes
    (`records_from_labeled` builds them). Confidence is the **calibrated** `max(p)` -- the logits are
    scaled by the fitted `temperature` / `temperature_by_options` first, so thresholds and the
    numbers the runtime reports are on the same scale. `target_error` is the tolerated error among
    accepted answers; `min_bucket_n` omits buckets too small to fit, and `conservative` adds a
    one-sample margin. An omitted bucket is not caught by a fallback: the map lookup falls to its own
    ``"default"`` entry and then to 0.0, which gates nothing, so a `min_bucket_n` that excludes a
    bucket leaves that bucket ungated unless the map names a ``"default"``. The thresholds are
    empirical cuts on the calibration set, not a formal coverage guarantee -- validate on held-out
    data (`fit_temperature_map(..., compute_ece=True)` gives a held-out split) for a production gate.

    Pass `binning_map` when the agent that will serve these thresholds has one installed -- by
    `Agent.fit_binning`, or by a calibration payload that carries `binning_map` -- because the
    runtime recalibrates `answer_confidence` through that map before anything reads it, so a cut
    fitted without it is a cut on a scale the gate never sees. The thresholds are then on the binned
    scale, and the order the two were fitted in stops mattering. Measured on 1,200 synthetic
    12-option records at `target_error=0.10`: the cut fitted without a map holds 9.8% error over 50%
    coverage on un-binned confidences, and admits 94.5% of answers at 25.6% error once the same
    number is compared against binned ones.
    """
    # A bool is the one value a range test cannot see: `0.0 <= True <= 1.0` holds, so `False` would
    # be read as 0.0 -- the cut that abstains on every answer -- and `True` as 1.0, which gates
    # nothing. Both fit a map, so nothing downstream would mention it. `TrainConfig` refuses the
    # same bool on the way to this call (`train.py`), as does `_iter_records` for a `qtype`.
    # Everything the range test cannot even compare -- a `"0.1"` from a config file, a `None` from a
    # missing key -- left a bare `TypeError: '<=' not supported between instances of 'float' and
    # 'str'`, naming neither the parameter nor the caller's mistake, while `min_bucket_n` one line
    # below has always named itself. The comparison is run rather than replaced by an `isinstance`
    # list so that a value which does compare -- `Decimal`, which `numbers.Real` does not cover --
    # keeps fitting.
    try:
        comparable = 0.0 <= target_error <= 1.0
    except TypeError:
        comparable = False
    if isinstance(target_error, bool) or not comparable:
        raise ValueError("target_error must be in [0.0, 1.0], got %r" % (target_error,))
    min_bucket_n = _positive_int("min_bucket_n", min_bucket_n)
    recs = _iter_records(records)
    by_bucket: Dict[str, List[Tuple[float, int]]] = {}
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        bucket = temp_bucket(qt, k)
        t_scale = temperature_by_options.get(bucket, temperature[qt])
        p = _softmax(z[:k], t_scale)
        conf = float(p.max())
        if binning_map:
            # The runtime bins before anyone reads `answer_confidence` (`Agent._decode_answers`),
            # so the cut has to be chosen on the binned scale or it gates a different quantity.
            conf = apply_binning_map(conf, bucket, binning_map)
        # `_decode_answers` reports `answer_confidence` rounded to 4 decimals, after binning, and
        # the gate compares that. A cut picked at full precision can sit above the rounded value of
        # its own cohort (bin 5/6 is reported as 0.8333 < 0.83333...), which abstains all of it.
        conf = round(conf, 4)
        by_bucket.setdefault(bucket, []).append((conf, int(int(p.argmax()) == y)))
    out: Dict[str, float] = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < min_bucket_n:
            continue
        out[key] = _select_abstention_threshold(pairs, target_error, conservative)
    return out


# Per-bucket floor for histogram-binning recalibration. Higher than the abstention floor because a
# binning map splits each bucket's examples across `bins`, so each bin needs its own sample; lower
# than the temperature floor because binning is a count per bin, not an optimisation.
MIN_BINNING_BUCKET_N = 200


def fit_binning_map(
    records: Iterable,
    temperature: Sequence[float],
    temperature_by_options: Dict[str, float],
    *,
    bins: int = 15,
    min_bucket_n: int = MIN_BINNING_BUCKET_N,
) -> Dict[str, Dict[str, Any]]:
    """Fit a per-`temp_bucket` histogram-binning recalibration map for `answer_confidence`.

    Temperature scaling applies one scalar per bucket; it cannot fix a bucket whose reliability
    curve is not a simple sharpening/softening (the pathological `choice:11+` the shipped English
    checkpoint carries is one). Histogram binning is the non-parametric alternative: split the
    calibrated confidences of a bucket into `bins` equal-width bins over [0, 1], and map every
    confidence that lands in a bin to that bin's empirical accuracy. It needs no monotonicity
    assumption and no extra dependency (NumPy only; isotonic regression would pull in scikit-learn).

    `records` are the same `(qtype, logits, target[, k])` tuples `fit_temperature_map` consumes;
    confidence is the calibrated `max(p)` (logits scaled by the fitted `temperature` /
    `temperature_by_options` first), so a binning map composes on top of a temperature map rather
    than replacing it. Returns `{bucket: {"bins": N, "values": [recalibrated confidence per bin]}}`;
    buckets below `min_bucket_n` are omitted. Apply it with :func:`apply_binning_map`. An empty bin
    (a confidence range the calibration set never produced) maps to that bin's centre, a point
    inside the region the value came from: the value is still rewritten, and moves by at most half a
    bin width. What the centre buys is that an unseen confidence is never recalibrated to a
    fabricated 0, not that it passes through -- only a bucket the map has no entry for does that
    (:func:`apply_binning_map`).
    """
    bins = _positive_int("bins", bins)
    min_bucket_n = _positive_int("min_bucket_n", min_bucket_n)
    recs = _iter_records(records)
    by_bucket: Dict[str, List[Tuple[float, int]]] = {}
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        bucket = temp_bucket(qt, k)
        t_scale = temperature_by_options.get(bucket, temperature[qt])
        p = _softmax(z[:k], t_scale)
        by_bucket.setdefault(bucket, []).append((float(p.max()), int(int(p.argmax()) == y)))
    out: Dict[str, Dict[str, Any]] = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < min_bucket_n:
            continue
        conf = np.asarray([c for c, _ in pairs], dtype=float)
        corr = np.asarray([c for _, c in pairs], dtype=float)
        idx = np.clip((conf * bins).astype(int), 0, bins - 1)
        values = []
        for b in range(bins):
            mask = idx == b
            values.append(float(corr[mask].mean()) if mask.any() else (b + 0.5) / bins)
        out[key] = {"bins": bins, "values": values}
    return out


def apply_binning_map(
    confidence: float, bucket: str, binning_map: Dict[str, Dict[str, Any]]
) -> float:
    """Recalibrate one `answer_confidence` for its option-count `bucket` (`common.temp_bucket`).

    Returns the confidence unchanged when the map has no entry for the bucket, so a bucket the map
    was not fit for passes through rather than being forced to a wrong value.
    """
    entry = binning_map.get(bucket)
    if not entry:
        return float(confidence)
    bins = int(entry["bins"])
    b = min(bins - 1, max(0, int(float(confidence) * bins)))
    return float(entry["values"][b])


def records_from_labeled(agent, pairs: Sequence) -> List[Record]:
    """Collect CPU records from `(state, questions, targets)`.

    Logits are the raw rows `_decode_answers` scales, not the probabilities it returns.
    `_encode_state` builds one state's items, `_forward` runs them, and `_option_logits`
    slices each question to the same marker width `_decode_answers` divides by temperature.
    No Hub download and no `model.safetensors` write. Tests stay weight-free by passing
    synthetic records to `fit_temperature_map` instead of calling this.
    """
    # Imported lazily: `agent` imports this module at load, and `_option_logits` is defined
    # on the way through that import.
    from .runtime import _option_logits

    records: List[Record] = []
    for state, questions, targets in pairs:
        ids = list(questions.keys())
        if not ids:
            continue
        for qid in ids:
            agent._check_question(qid, questions[qid])
        internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
        items = agent._encode_state(state, ids, internal)
        batch = collate_items([items], agent.tok.pad_token_id)
        if batch is None:
            continue
        logits, _act = agent._forward(batch)
        rows = _option_logits(logits, items, 0)
        for j, qid in enumerate(ids):
            k = len(items[j]["markers"])
            qt = int(items[j]["qtype"])
            records.append((qt, _vec(rows[j]), _vec(targets[qid])[:k], k))
    return records


def _config_identity(cfg):
    """Checkpoint config without the temperature fields this file itself stores."""
    if cfg is None:
        return None
    return {str(k): cfg[k] for k in cfg if k not in _CONFIG_IDENTITY_SKIP}


def calibration_payload(
    temperature,
    temperature_by_options,
    model_id_or_path=None,
    subfolder=None,
    config=None,
    binning_map=None,
) -> Dict[str, Any]:
    """JSON body written by `Agent.save_calibration` (no weights).

    `version` is `CALIBRATION_VERSION`. Payloads written before identity was recorded have
    no version key; `apply_calibration_payload` treats that as version 1 and still loads.

    `model_id_or_path`, `subfolder`, and `config` say which checkpoint the map was fitted
    against. `config` is `rl_agent_config.json` without `temperature` /
    `temperature_by_options` (those live at the top of this payload).

    `binning_map` is the optional histogram-binning recalibration map fitted by
    `fit_binning_map`; the key is omitted when no map is installed, which is `None`, not an empty
    map. A map that fitted nothing -- `{}`, the return value when no bucket reached the sample
    floor -- is still written, so the round trip through :func:`apply_calibration_payload` gives
    `{}` back rather than `None`.
    """
    payload = {
        "version": CALIBRATION_VERSION,
        "temperature": [float(x) for x in temperature],
        "temperature_by_options": {
            str(k): float(v) for k, v in dict(temperature_by_options).items()
        },
        "model_id_or_path": model_id_or_path,
        "subfolder": subfolder,
        "config": _config_identity(config),
    }
    if binning_map is not None:
        payload["binning_map"] = binning_map
    return payload


def _warn_if_identity_mismatch(obj, payload) -> None:
    recorded_model = payload.get("model_id_or_path")
    recorded_sub = payload.get("subfolder")
    recorded_cfg = payload.get("config")
    current_model = getattr(obj, "model_id_or_path", None)
    current_sub = getattr(obj, "subfolder", None)
    current_cfg = _config_identity(getattr(obj, "cfg", None))
    if (recorded_model, recorded_sub, recorded_cfg) == (current_model, current_sub, current_cfg):
        return
    cfg_note = ""
    if recorded_cfg != current_cfg:
        cfg_note = " Checkpoint config identity differs."
    warnings.warn(
        "calibration was fitted for model_id_or_path=%r subfolder=%r but this agent is "
        "model_id_or_path=%r subfolder=%r.%s Loading the temperatures anyway."
        % (recorded_model, recorded_sub, current_model, current_sub, cfg_note),
        UserWarning,
        stacklevel=3,
    )


def _rejected_temperatures(entries):
    """Names whose raw value is not the number `clamp_temperature` kept, matching checkpoint load."""
    rejected = []
    for name, raw, applied in entries:
        if not temperature_rejected(raw, applied):
            continue
        rejected.append("%s=%r -> %g" % (name, raw, applied))
    return rejected


def _install_temperatures(obj, temperature, temperature_by_options, warn: bool = True) -> None:
    """Store a map the way checkpoint load does: raw values, and the clamped ones inference reads.

    `warn=False` is the fitter path: those values were already passed through `clamp_temperature`,
    so a second pass must not report them as a bad file.
    """
    raw_t = list(temperature)
    applied_t = [clamp_temperature(t) for t in raw_t]
    raw_b = {str(k): v for k, v in dict(temperature_by_options).items()}
    applied_b = {k: clamp_temperature(v) for k, v in raw_b.items()}
    obj.temperature_raw = raw_t
    obj.temperature = applied_t
    obj.temperature_by_options_raw = raw_b
    obj.temperature_by_options = applied_b
    if not warn:
        return
    entries = [("temperature[%d]" % i, raw, applied_t[i]) for i, raw in enumerate(raw_t)]
    entries += [(key, raw_b[key], applied_b[key]) for key in raw_b]
    rejected = _rejected_temperatures(entries)
    if rejected:
        warnings.warn(
            "laya: calibration temperatures are invalid or outside [%g, %g]; "
            "using %s. Treat confidence from the affected entries as uncalibrated."
            % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
            RuntimeWarning,
            stacklevel=3,
        )


def apply_calibration_payload(obj, payload: Dict[str, Any]) -> None:
    """Copy `temperature`, `temperature_by_options` and `binning_map` onto `obj`.

    Those are the three fields this reads and the three it writes. `binning_map` is installed
    whatever the payload holds, including nothing: a payload with no ``binning_map`` key -- every
    file written before histogram binning existed, and every agent that saved before
    :meth:`Agent.fit_binning` ran -- installs `None`, which clears a map this object already
    fitted. The file is the whole calibration state, not a patch onto the current one.

    A missing `version` is version 1 (temperatures only, no checkpoint identity). Version
    `CALIBRATION_VERSION` records the checkpoint the map was fitted for; a mismatch warns
    and still loads, so an older file never becomes a hard failure. Each temperature is passed
    through `clamp_temperature`, so a non-numeric or out-of-range entry cannot crash a later
    forward the way an unclamped zero used to.

    `temperature` and `binning_map` take opposite value policies, and that is the sentence to read
    before writing a file by hand: a bad *temperature* is forgiven and clamped into
    `[TEMP_MIN, TEMP_MAX]`, while a bad *binning value* is refused. Nothing recalibrates a
    confidence the way an out-of-range binning value would, so there is no defensible fallback
    for it.

    The payload's *shape* is checked before any of that, and refused with a `ValueError` naming
    the field: the values may be junk the clamp forgives, but the payload itself not being an
    object, a `version` that is not an integer, a `temperature` that is not a list of three, a
    `temperature_by_options` that is not an object, a `binning_map` that is not an object, or a
    `binning_map` entry that is not an object with an integer `bins` >= 1 and `values` of exactly
    that length holding finite numbers in [0, 1] is a file this code cannot read -- JSON gives
    `int`, `str`, `list` and `dict` for each of those mistakes just as happily as it gives the
    right shapes, and each used to fail with a raw `TypeError`/`AttributeError` from
    `len()`/`dict()`, or worse, to load: a `{"a": 1, "b": 2, "c": 3}` or `"abc"` has `len` 3 and
    used to pass the length check and install its *keys* as temperatures.
    """
    if not isinstance(payload, dict):
        raise ValueError("calibration JSON must be an object, got %s" % type(payload).__name__)
    version = payload.get("version", 1)
    if version is None:
        version = 1
    if isinstance(version, bool) or not isinstance(version, (int, float, str)):
        raise ValueError("calibration JSON version must be an integer, got %r" % (version,))
    if isinstance(version, float) and not version.is_integer():
        # `int()` truncates, and the truncation is not cosmetic: a file whose `version` is 1.9
        # became 1, which is below CALIBRATION_VERSION, so the identity check at the foot of this
        # function never ran and a map fitted for another checkpoint loaded silently. The string
        # form of the same mistake already raises below; a whole number written as `2.0` is an
        # integer and still loads.
        raise ValueError("calibration JSON version must be an integer, got %r" % (version,))
    try:
        version = int(version)
    except (TypeError, ValueError, OverflowError) as exc:  # Overflow: JSON's 1e999 is inf
        raise ValueError(
            "calibration JSON version must be an integer, got %r" % (version,)
        ) from exc
    temps = payload.get("temperature")
    if not isinstance(temps, (list, tuple)) or len(temps) != N_QTYPES:
        raise ValueError("calibration JSON must contain temperature: [3 floats]")
    by_options = payload.get("temperature_by_options")
    if by_options is None:  # absent, or an explicit null: a file from before bucket temperatures
        by_options = {}
    if not isinstance(by_options, dict):
        raise ValueError(
            "calibration JSON temperature_by_options must be an object of bucket -> float, "
            "got %s" % type(by_options).__name__
        )
    binning = payload.get("binning_map")
    if binning is not None:
        if not isinstance(binning, dict):
            raise ValueError(
                "calibration JSON binning_map must be an object of bucket -> {bins, values}, "
                "got %s" % type(binning).__name__
            )
        for name, entry in binning.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    'calibration JSON binning_map[%r] must be an object with "bins" and "values", '
                    "got %s" % (name, type(entry).__name__)
                )
            n_bins = entry.get("bins")
            values = entry.get("values")
            if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
                raise ValueError(
                    'calibration JSON binning_map[%r] must have an integer "bins" >= 1, got %r'
                    % (name, n_bins)
                )
            if not isinstance(values, (list, tuple)) or len(values) != n_bins:
                raise ValueError(
                    'calibration JSON binning_map[%r] must have "values" of length "bins" (%d)'
                    % (name, n_bins)
                )
            try:
                parsed_values = [float(v) for v in values]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "calibration JSON binning_map[%r] values must be numbers, got %r"
                    % (name, values)
                ) from exc
            for v in values:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(
                        "calibration JSON binning_map[%r] values must be numbers, got %r"
                        % (name, values)
                    )
            for v in parsed_values:
                if not np.isfinite(v):
                    raise ValueError(
                        "calibration JSON binning_map[%r] values must be finite, got %r"
                        % (name, values)
                    )
                if not 0.0 <= v <= 1.0:
                    raise ValueError(
                        "calibration JSON binning_map[%r] values must be in [0, 1], got %r"
                        % (name, values)
                    )
    obj.binning_map = binning
    if version >= CALIBRATION_VERSION:
        _warn_if_identity_mismatch(obj, payload)
    _install_temperatures(obj, temps, by_options)
