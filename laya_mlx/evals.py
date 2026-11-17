"""Labelled evaluation harness: score a runner on a dataset and gate a build on it.

Pure Python plus numpy, and free of torch at import time, so the metric math and the dataset
parsing can be unit tested with no weights. `evaluate` only needs a runner with a
``predict(state, questions, model=...)`` method, plus -- for ``batch_size`` above 1 -- a
``predict_batch`` in either of the two shapes the repo ships: the positional one, or the
per-request-dict one `Router.predict_batch` documents. A fixture runner stands in for a
checkpoint in tests.

The report is deterministic for a fixed runner: the same dataset produces the same numbers, and
``EvalReport.compare`` turns a baseline into a pass/fail with the per-metric deltas, which is what
the CI gate consumes.

`compare` reads ``overall`` and nothing else, so a run that says nothing about *what it measured*
makes a gate that can only compare arithmetic. Every report therefore carries a run identity in
``config`` -- ``schema``, ``questions_sha256``, ``laya_version`` and (from the CLI)
``dataset_sha256`` and the ``thresholds`` actually applied -- and
``EvalReport.comparable_to`` refuses a comparison between two runs that are not the same
measurement. The identity is deliberately free of anything time-bearing, so a report stays
byte-reproducible for a fixed runner.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .confidence import check_min_confidence

#: The shape of the report this module writes, so a consumer can refuse one it cannot read.
#: ``research/evals/act_head_eval.py`` publishes a report under its own tag off this prefix.
REPORT_SCHEMA = "laya-evals-report/1"

#: The keys that decide whether two reports are the same measurement. A key missing
#: on either side is unknown rather than a conflict, so a baseline committed before these existed
#: keeps comparing exactly as it did. ``schema`` is also read at the top level, which is where
#: ``research/evals/act_head_eval.py`` puts its own tag.
_IDENTITY_KEYS = ("schema", "dataset_sha256", "questions_sha256")

_IDENTITY_LABELS = {
    "schema": "report schema",
    "dataset_sha256": "dataset bytes",
    "questions_sha256": "question schema",
}


def _identity_of(document: Any) -> Dict[str, Any]:
    """Pull the identity keys out of a report, from ``config`` or from the top level."""
    if not isinstance(document, dict):
        return {}
    config = document.get("config")
    found = dict(config) if isinstance(config, dict) else {}
    for key in _IDENTITY_KEYS:
        if found.get(key) is None and document.get(key) is not None:
            found[key] = document[key]
    return found


class EvalError(ValueError):
    """A dataset or report is malformed; the message names the row or field."""


def _metric_map(values: Any, label: str, *, allow_empty: bool = False) -> Dict[str, Any]:
    """Return a report's metric map, naming any shape `compare` cannot do arithmetic on.

    Both sides reach `compare` as JSON read from a file the caller named, and the loop subtracts
    one value from the other for every shared metric. A JSON scalar or list, an `overall` that is
    not an object, and a metric whose value is a string, null or dict each used to get that far:
    they raised `AttributeError`/`TypeError` as an unhandled traceback, and `laya-evals` exits 1
    for an unhandled traceback -- the code it reserves for a quality failure, so a broken baseline
    file printed the same verdict as a regressed model. A bool was worse than a crash: `float()`
    turns it into 1.0, so `"choice_accuracy": true` compared as a measurement of 100%.

    An empty map is refused unless `allow_empty`, because it has no metric left that could fail:
    the committed baseline that turns out to be `{}` or `null` would green the gate. `allow_empty`
    is the run's own side, where losing a metric a baseline still has is a result to report
    (`missing: True`) rather than a malformed document.
    """
    if not isinstance(values, dict):
        raise EvalError(
            "%s metrics must be an object of metric name to number, got %s"
            % (label, type(values).__name__)
        )
    if not values and not allow_empty:
        raise EvalError("%s has no metrics to compare, so nothing in it could fail" % label)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvalError(
                "%s metric %r is not a number, got %s" % (label, name, type(value).__name__)
            )
    return values


def _tolerance_map(tolerances: Any) -> Dict[str, Any]:
    """Return a caller's tolerance map, naming any entry `compare` cannot gate on.

    Both metric maps go through ``_metric_map`` for this reason; the tolerances reach the same
    subtraction as a dict the caller built -- a hand-edited `thresholds.json` for
    `research/evals/check_regression.py`, a config module for a test -- and were read with a bare
    ``float(tolerances.get(metric, 0.0))``. A string, null, list or dict there raised ``TypeError``
    from inside the arithmetic and a list or string given as the whole map raised
    ``AttributeError``, so the gate died with exit 1 -- the code `laya-evals` reserves for a
    quality failure -- on what is a broken gate parameter. The falsy shapes (`0`, `0.0`, `""`, `()`)
    were quieter still: ``tolerances or {}`` read them as no tolerances at all. A bool was read as
    1.0, which for a metric that lives in [0, 1] accepts any drift the run can report.

    ``Infinity`` is the one number that cannot fail a comparison: every finite drift is inside it,
    so naming a metric with it turns that metric's gate into a no-op while the run still exits 0
    -- the code that means "the gate ran and the numbers passed". A NaN or a negative tolerance
    decides the opposite way (every comparison with NaN is False, and no absolute difference is
    ``<=`` a negative number), so those stay results rather than refusals; see
    ``test_compare_refuses_a_tolerance_that_cannot_fail``. An empty map is a complete answer here,
    unlike an empty metric map: it means every shared metric must match exactly.
    """
    if tolerances is None:
        return {}
    if not isinstance(tolerances, dict):
        raise EvalError(
            "tolerances must be an object of metric name to number, got %s"
            % type(tolerances).__name__
        )
    import math as _math

    for name, value in tolerances.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvalError(
                "tolerance for %r is not a number, got %s" % (name, type(value).__name__)
            )
        if _math.isinf(value) and value > 0:
            raise EvalError(
                "tolerance for %r is Infinity, so it accepts a drift of any size" % name
            )
    return tolerances


def compare_agreement(
    reference: Dict, repeat: Dict, student: Dict, *, bootstrap_samples: int = 2000, seed: int = 0
) -> Dict[str, Any]:
    """Compare saved teacher reference, teacher repeat, and student choice observations.

    Inputs use ``laya-agreement-run/1``; see the evaluation guide for the schema.
    All runs must contain the same cases and questions. Student nulls count as
    disagreements and reduce coverage. The report includes a paired, case-clustered
    bootstrap interval (100–100000 replicates), provenance, and individual decisions.
    Teacher agreement is not ground-truth accuracy or a ceiling on student performance.
    """
    from ._agreement import compare_agreement as compare

    return compare(reference, repeat, student, bootstrap_samples=bootstrap_samples, seed=seed)


def evaluate_agreement(
    runner: Any,
    reference: Dict,
    repeat: Dict,
    *,
    student_provenance: Dict,
    model: Optional[str] = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate a local student's agreement with two saved teacher observations.

    Validate both teacher runs before calling ``runner.predict(state, questions,
    model=model)`` once per original case. Record the caller's student provenance;
    use a null revision when it is unknown. Runner errors propagate, and missing or
    invalid answers raise ``EvalError`` instead of dropping decisions. An answer's
    ``abstention='abstained'`` becomes a null student observation.
    """
    from ._agreement import evaluate_agreement as evaluate_student

    return evaluate_student(
        runner,
        reference,
        repeat,
        student_provenance=student_provenance,
        model=model,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


@dataclass
class Example:
    """One labelled decision: a state, its questions, and the expected answer per question id."""

    state: Any
    questions: Dict[str, Any]
    expected: Dict[str, Any]
    tags: Tuple[str, ...] = ()
    language: Optional[str] = None
    model: Optional[str] = None  # explicit checkpoint, forwarded to the runner

    @classmethod
    def from_dict(cls, row: Any, where: str = "row") -> "Example":
        if not isinstance(row, dict):
            raise EvalError("%s must be an object, got %s" % (where, type(row).__name__))
        for key in ("state", "questions", "expected"):
            if key not in row:
                raise EvalError("%s is missing %r" % (where, key))
        questions, expected = row["questions"], row["expected"]
        if not isinstance(questions, dict):
            raise EvalError("%s 'questions' must be an object" % where)
        if not isinstance(expected, dict):
            raise EvalError("%s 'expected' must be an object keyed by question id" % where)
        unknown = sorted(set(expected) - set(questions))
        if unknown:
            raise EvalError(
                "%s 'expected' names unknown question(s): %s" % (where, ", ".join(unknown))
            )
        tags = row.get("tags") or ()
        if not isinstance(tags, (list, tuple)):
            raise EvalError("%s 'tags' must be a list" % where)
        return cls(
            state=row["state"],
            questions=questions,
            expected=expected,
            tags=tuple(str(t) for t in tags),
            language=row.get("language"),
            model=row.get("model"),
        )


class Dataset:
    """An ordered collection of labelled examples."""

    def __init__(self, examples: Iterable[Example]):
        self.examples: List[Example] = list(examples)
        # `Example.from_dict` refuses a JSONL row whose `questions` or `expected` is not an object, or
        # whose `expected` names a question `questions` does not carry. A row built in Python with
        # `Example(...)` skipped all three, and every reader of it takes the two fields as mappings: a
        # non-object `expected` died as an unhandled ``AttributeError``, in `evaluate` only after the
        # runner had answered (`laya/evals.py:1205`), and a non-object `questions` leaked through
        # `questions_fingerprint` (`laya/evals.py::questions_fingerprint`). An `expected` key with no
        # question behind it was answered with "runner returned no answer for question 'x'" -- a
        # verdict about the runner, paid for with a whole inference run, for a mistake in the
        # caller's row. The two opt-in doors
        # refused the same rows before calling anything, but as tracebacks
        # (`laya/evals_shortlist.py:38`, `laya/evals_tournament.py:33`). Checking at the door all four
        # readers share names the row and lands before a single call: a leaked traceback exits 1, the
        # code `laya-evals` reserves for a failed threshold (`laya/evals_cli.py:591`).
        for index, example in enumerate(self.examples):
            # Both checks below read the row's own fields, so a row that is not a row at all died on the
            # attribute before either could answer: `Dataset(json.load(handle))` -- the natural mistake
            # when a dataset is a JSON *array* rather than JSONL, and the shape `Example.from_dict` exists
            # to convert -- raised `AttributeError: 'dict' object has no attribute 'questions'` from
            # inside the door. `docs/evals.md` ("Dataset format") promises this route fails as `EvalError`
            # before the first model call, and a traceback names no row. The check is on the two attributes
            # this door reads rather than on the type, so no row that works today starts being refused.
            if not hasattr(example, "questions") or not hasattr(example, "expected"):
                missing = "questions" if not hasattr(example, "questions") else "expected"
                raise EvalError(
                    "example %d must be an Example, got %s with no %r; build each row "
                    "with laya.evals.Example(...) or Example.from_dict(...)"
                    % (index, type(example).__name__, missing)
                )
            if not isinstance(example.questions, dict):
                raise EvalError(
                    "example %d 'questions' must be an object, got %s"
                    % (index, type(example.questions).__name__)
                )
            if not isinstance(example.expected, dict):
                raise EvalError(
                    "example %d 'expected' must be an object keyed by question id, got %s"
                    % (index, type(example.expected).__name__)
                )
            unknown = sorted(set(example.expected) - set(example.questions))
            if unknown:
                raise EvalError(
                    "example %d 'expected' names unknown question(s): %s"
                    % (index, ", ".join(unknown))
                )

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        # `laya-evals` promises exit 2 on a usage error and exit 1 only when a threshold or a
        # tolerance failed (`docs/evals.md`), and a dataset the caller cannot hand over is the
        # caller's mistake. Four shapes of unreadable escaped that as a traceback and exit 1 -- a
        # path that is a directory, a file without read permission, bytes that are not UTF-8 and a
        # line nested deeper than `json` will walk -- while a line that is merely malformed JSON was
        # already reported, three lines below. `file_fingerprint` above and `_load_report` in
        # `laya/evals_cli.py` read a caller's path this way already; this makes the dataset reader
        # agree with them. `UnicodeDecodeError` rather than `ValueError` on purpose: `EvalError`
        # *is* a `ValueError`, so the wider class would re-wrap the row errors named just below.
        examples: List[Example] = []
        try:
            # `utf-8-sig`, not `utf-8`: a dataset exported from Excel or saved by Notepad starts
            # with a BOM, and a plain `utf-8` read leaves it as U+FEFF on line 1, so `json` refuses a
            # file that is valid JSON -- and says so with its own "decode using utf-8-sig" hint.
            # `laya/cli.py`, `laya/train.py` and `laya/evidence.py` open a caller's file this way.
            with open(path, "r", encoding="utf-8-sig") as handle:
                for lineno, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError as exc:
                        raise EvalError("%s:%d is not valid JSON: %s" % (path, lineno, exc))
                    examples.append(Example.from_dict(row, where="%s:%d" % (path, lineno)))
        except (OSError, UnicodeDecodeError, RecursionError) as exc:
            raise EvalError("%s cannot be read: %s" % (path, exc)) from exc
        if not examples:
            raise EvalError("%s contains no examples" % path)
        return cls(examples)


def _canonical(value: Any) -> str:
    # `sort_keys=False` on purpose. `examples/hooks/cache.py:22-25` states the rule this mirrors:
    # "Deliberately not `sort_keys=True`: a choice question's criteria order is positional, so two
    # orders are two questions, and `_question_schema` in `laya/router.py` keeps them apart for the
    # same reason." `sort_keys` reorders *dict* keys and leaves *lists* alone, so with a list-valued
    # `criteria` -- the shape a dataset row actually has -- this was already order-preserving; the
    # flag only ever mattered for a dict-valued one, where folding is the wrong direction.
    return json.dumps(
        value, sort_keys=False, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _as_tokenized_instructions(body: Dict[str, Any]) -> Any:
    """`instructions` exactly as `laya/agent.py::Agent._to_internal` hands it over.

    A non-string becomes ``json.dumps(ins, ensure_ascii=False)``, and that step is not cosmetic:
    ``tests/test_criteria.py:229-251`` pins that the default ``ensure_ascii=True`` escaped
    non-ASCII to literal ``\\uXXXX``, the tokenizer read it as escape text, and one German
    question answered noul=0.1652 as a dict against 0.2650 as the identical plain string. So the
    fingerprint normalizes the same way, or it hashes the input's JSON shape instead of the text
    the model reads.

    Not mirrored: the tokenizer's mask-token strip in `common.build_sequence`, which needs
    `tok.mask_token` and would mean guessing which tokenizer a run loads. Instructions carrying
    a mask token therefore hash apart even though they render alike -- a false refusal, which is
    the safe direction to be wrong in.
    """
    ins = body.get("instructions")
    return ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)


def questions_fingerprint(dataset: "Dataset") -> str:
    """A stable hash of *what was asked*, over every question in `dataset`.

    This is the question-schema identity `docs/staged-adoption.md` asks an operator to record
    with the policy, and the part of the run identity a report can compute for itself: the
    dataset file hash is the CLI's, but the questions are parsed here.

    It covers every field the answer depends on, which is the same rule `examples/hooks/cache.py`
    states for a cache key ("has to cover everything the answer depends on") and the same one
    Laya already applies to its own question identity: `laya/router.py::_question_schema` and this
    module's batch grouping (`laya/evals.py::evaluate`) both hash the whole questions dict.
    `tests/test_router_batch.py:543` pins that rewording `instructions` alone moves a row into its
    own batch group. Those two need nothing when a question grows a field, because a whole-dict hash
    takes whatever the dict holds; this one lists its fields, so a sixth field
    `laya/agent.py::Agent._check_question` starts reading has to be added here as well or the
    identity goes blind to it again.

    It is a function of the decision space, not of the rows, so scoring more states on the same
    questions leaves it unchanged.

    `criteria` is hashed raw, without `Agent._to_internal`'s list-to-dict and lowercase-key
    rewrites. That is deliberate and asymmetric: two `noul` questions that render alike but are
    written differently would then be refused, which is the safe direction to be wrong in.
    Normalizing them would widen the change past what the identity needs.

    `labels` is hashed for the same reason `instructions` is:
    `laya/agent.py::Agent._check_question` validates it through
    `laya/common.py::_resolve_noul_labels`, `laya/agent.py::Agent._to_internal` carries it into
    the internal question, and that resolver is what turns it into the option text the model
    reads. Leaving it out let two question sets with different rendered options hash alike, which
    is the one thing this function exists to prevent.

    `option_order` is the fifth and last field a question body can carry, and it was the one left
    out: `laya/agent.py::Agent._check_question` accepts only a permutation of the option indices,
    `Agent._to_internal` copies it into the internal question, and from there it reaches both the
    tokenized head (`laya/common.py::build_sequence` through `build_head`) and the reported
    probabilities (`laya/common.py::unpermute_probs`, applied in `Agent._decode_answers`). It is
    hashed only when a row names it, so the digest of a question set that does not use the field
    stays the number its committed baseline already carries.
    """
    schemas = set()
    for example in dataset.examples:
        for qid, question in example.questions.items():
            body = question if isinstance(question, dict) else {}
            hashed = {
                "qid": qid,
                "type": body.get("type"),
                "instructions": _as_tokenized_instructions(body),
                "labels": body.get("labels"),
                "criteria": body.get("criteria"),
            }
            if "option_order" in body:
                schemas.add(_canonical(dict(hashed, option_order=body["option_order"])))
            else:
                # Left out rather than hashed as `null`, so a dataset that never names the field
                # keeps the digest its committed baseline already carries.
                schemas.add(_canonical(hashed))
    digest = hashlib.sha256()
    for schema in sorted(schemas):
        digest.update(schema.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_fingerprint(path: str) -> str:
    """The sha256 of a file's bytes -- the dataset identity, where the path is only a name.

    `config.dataset` is the path as typed, and two datasets share a path across a rebase, a CI
    cache or a colleague's checkout. Raises `EvalError` when the file cannot be read, so a report
    never carries a hash of something other than the bytes that were parsed.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
    except OSError as exc:
        raise EvalError("%s cannot be read for a fingerprint: %s" % (path, exc)) from exc
    return digest.hexdigest()


def _library_version() -> Optional[str]:
    """`laya.__version__`, read lazily so this module stays importable on its own.

    `laya/__init__.py` defines the string at import time, so this is a module lookup rather
    than a load of anything torch-backed -- but it is imported inside the call to keep
    `laya.evals` free of an import cycle, and a failure here must not fail a run that has
    already scored every row.
    """
    try:
        import laya_mlx as laya
    except Exception:  # pragma: no cover - identity only
        return None
    return getattr(laya, "__version__", None)


#: Which definition of the coverage metrics (`aurc`, `selective_accuracy@*`) produced a report.
#: Bumped when a cut's meaning changes, which is not a schema change -- the keys and their types
#: are identical, only the arithmetic behind two of them moves. Version 2 cuts on a confidence
#: THRESHOLD, so a cut never splits a group of equal confidences; version 1 (every report written
#: before this existed, and recognised by the key's ABSENCE) cut at a row index and therefore
#: depended on the order the dataset happened to arrive in.
#:
#: A relative gate rule subtracts a baseline's number from this run's. Across definitions that
#: subtraction is meaningless, and it is unsafe in the PASS direction: a baseline recorded at 0.0333
#: by version 1 -- a value version 2 reports as 0.5167 for the same rows -- lets a candidate that
#: truly regressed by 0.2167 clear a `max_drop` of 0.05. `_eval_policy` refuses that comparison
#: rather than performing the arithmetic; see `check_policy`.
COVERAGE_METRIC_DEFINITION = 2


def _run_identity(dataset: "Dataset") -> Dict[str, Any]:
    """The `config` keys this module can fill in on its own, with no knowledge of the CLI."""
    identity: Dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "questions_sha256": questions_fingerprint(dataset),
        "coverage_metric_definition": COVERAGE_METRIC_DEFINITION,
    }
    version = _library_version()
    if version is not None:
        identity["laya_version"] = version
    return identity


# --------------------------------------------------------------------------- evaluators
class Evaluator:
    """Per-answer metric. `score` returns a number, or None when it does not apply."""

    name = "evaluator"

    def score(self, answer: Dict[str, Any], expected: Any) -> Optional[float]:
        raise NotImplementedError


class ChoiceAccuracy(Evaluator):
    name = "choice_accuracy"

    def score(self, answer, expected):
        if answer.get("type") != "choice" or not isinstance(expected, str):
            return None
        return 1.0 if answer.get("choice") == expected else 0.0


class NoulAccuracy(Evaluator):
    name = "noul_accuracy"

    def score(self, answer, expected):
        if answer.get("type") != "noul" or not isinstance(expected, bool):
            return None
        return 1.0 if bool(answer.get("noul", 0.0) >= 0.5) == expected else 0.0


class ScoreMAE(Evaluator):
    name = "score_mae"

    def score(self, answer, expected):
        if (
            answer.get("type") != "score"
            or not isinstance(expected, (int, float))
            or isinstance(expected, bool)
        ):
            return None
        return abs(float(answer.get("score", 0.0)) - float(expected))


class ScoreWithin(Evaluator):
    name = "score_within"

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = float(tolerance)
        self.name = "score_within_%g" % self.tolerance

    def score(self, answer, expected):
        if (
            answer.get("type") != "score"
            or not isinstance(expected, (int, float))
            or isinstance(expected, bool)
        ):
            return None
        return (
            1.0 if abs(float(answer.get("score", 0.0)) - float(expected)) <= self.tolerance else 0.0
        )


class MeanConfidence(Evaluator):
    name = "mean_confidence"

    def score(self, answer, expected):
        return _answer_confidence(answer)


DEFAULT_EVALUATORS = (ChoiceAccuracy, NoulAccuracy, ScoreMAE, MeanConfidence)


def default_evaluators() -> List[Evaluator]:
    return [factory() for factory in DEFAULT_EVALUATORS]


def _answer_confidence(answer: Dict[str, Any]) -> Optional[float]:
    """The column the calibration metrics read: `answer_confidence` when the answer carries one.

    `answer_confidence` is the probability of the answer being reported -- the quantity temperature
    scaling fits and the quantity this repository's calibration figures are computed on. It is not
    *calibrated* as shipped: both base checkpoints are over-confident and `laya-multilingual` ships no
    fitted temperatures at all (README, Calibration), which is why `ece` measures this column instead
    of assuming it.

    An answer carrying no `answer_confidence` falls through three more reads, in order: the
    `confidence` field, then `max(p, 1 - p)` for a `noul`, then `max(probabilities)`. Those are not
    the same quantity, and the first is the one this function otherwise exists to avoid: `confidence`
    is normalized entropy on `choice` and `score`, which moves with the option count (#394) rather
    than with how right the answer is, and `laya.confidence.answer_confidence_value` refuses to fall
    back to it for the abstention gate on exactly that reasoning. `max(probabilities)` is the mass on
    the top option, which is the reported answer's probability only when the answer *is* the argmax.

    The stripped-down shape is not hypothetical. `LAYA_JEV_STRICT` projects the served response onto
    the Jev wire contract, which carries no `answer_confidence` (`laya/serve.py`, and the flag's row
    in docs/http-api.md), so a report run over recorded strict responses calibrates the entropy
    number and is not comparable to one run over full payloads. `tests/test_evals.py` pins both paths
    and the gap between them; `docs/evals.md` says so where an operator reads the metrics.
    """
    confidence = answer.get("answer_confidence")
    # `bool` is a subclass of `int`, so `isinstance(True, (int, float))` is true: a JSON
    # `"answer_confidence": true` would be read as a confidence of 1.0 and mask the numeric
    # `confidence` fallback below. A yes/no flag is not a probability, so it is skipped exactly
    # like an absent or non-numeric field -- the same exclusion `confidence.py` and the schema
    # guards apply to every other caller-supplied probability.
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        return float(confidence)
    confidence = answer.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        return float(confidence)
    if answer.get("type") == "noul":
        p = float(answer.get("noul", 0.0))
        return max(p, 1.0 - p)
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        return max(float(v) for v in probabilities.values())
    return None


def _correct(answer: Dict[str, Any], expected: Any) -> Optional[bool]:
    if answer.get("type") == "choice" and isinstance(expected, str):
        return answer.get("choice") == expected
    if answer.get("type") == "noul" and isinstance(expected, bool):
        return bool(answer.get("noul", 0.0) >= 0.5) == expected
    return None


def ece(confidences: Sequence[float], corrects: Sequence[bool], bins: int = 15) -> Optional[float]:
    """Expected Calibration Error, reusing the repository's own `common.ece_score`."""
    if not confidences:
        return None
    from .common import ece_score  # lazy: keeps `import laya.evals` torch-free

    return float(
        ece_score(np.asarray(confidences, dtype=float), np.asarray(corrects, dtype=bool), bins=bins)
    )


# Selective-classification metrics. The abstention gate (#361/#456) and the per-bucket thresholds
# (#394) decide *what to answer*; these say *how well the confidence ranks right from wrong* and
# *what a coverage/risk trade buys*, which ECE (a calibration number) does not. All read the same
# `(confidence, correct)` pairs ECE does, and are pure NumPy / torch-free.
def brier(confidences: Sequence[float], corrects: Sequence[bool]) -> Optional[float]:
    """Brier score of confidence as P(correct): mean((confidence - correct)**2). Lower is better."""
    if not confidences:
        return None
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(corrects, dtype=float)
    return float(np.mean((c - y) ** 2))


def _accepted_at_cut(sorted_desc, k: int) -> int:
    """How many answers a confidence threshold placed at `sorted_desc[k - 1]` accepts.

    The end of the tie group the cut landed in, which is what makes a coverage number independent
    of the order a dataset happened to arrive in.

    This is the single-cut answer; `_levels` computes the same group boundaries for *every* level,
    which is what `aurc` integrates over. Two routes to one rule, so they are not merely asserted
    to agree -- `test_a_single_cut_agrees_with_the_risk_coverage_curve` sweeps them against each
    other, including over NaN and infinities, so an edit to either side fails a test rather than
    diverging quietly. Computing one cut from the full scan instead cost 19% to 58% on
    `selective_accuracy` for no change in any value.

    NaN is handled explicitly rather than left to the comparison. Every comparison against a NaN
    is false, so `sorted_desc >= nan` accepts nothing and the cut would fall back to "whichever
    rows came first" -- the exact order-dependence this exists to remove, for the input a custom
    runner is most likely to hand over by accident. NaNs sort last and form one group, so a cut
    inside them accepts all of them.

    The NaN branch is kept although `searchsorted` happens to answer the same thing -- numpy ranks
    NaN last for a binary search too, verified over 63,697 NaN cuts without a disagreement. Leaving
    it implicit would make a correct answer depend on an undocumented ordering convention, so the
    branch stays and its mutant is reported as equivalent rather than as a test gap.
    """
    cut = sorted_desc[k - 1]
    if cut != cut:  # NaN
        return len(sorted_desc)
    # `sorted_desc` descends, so its negation ascends and a binary search finds the end of the tie
    # group in O(log n) rather than a full pass per call. NaNs rank above every number in numpy's
    # sort order, so they sit at the end of the negated array and a finite cut never counts them.
    return int(np.searchsorted(-sorted_desc, -cut, side="right"))


def _levels(confidences: Sequence[float], corrects: Sequence[bool]):
    """`(accepted, hits, n)` at each distinct confidence level, most confident first.

    One row per LEVEL, not per answer. A coverage cut is bought with a confidence threshold, and a
    threshold accepts every answer at its own confidence -- `calibrate._select_abstention_threshold`
    already says exactly this about the error it judges ("a tie group's errors must all be counted
    before the level is judged"), and it is the same rule here.

    Cutting at a row index instead reports whichever tie members the dataset happened to list
    first. A stable sort makes that reproducible, not order-independent, and ties are the normal
    case rather than a corner: `answer_confidence`'s own docstring records that the shipped
    `choice:11+` bucket "returns a point mass at 1.0". Permuting a JSONL of 8 rows, all tied, 4
    correct, moved `aurc` from 0.1827 to 0.8173 and `selective_accuracy@50` from 1.000 to 0.000.

    `accepted[i]` is how many answers a threshold at level `i` accepts, which is what lets `aurc`
    weight a level by the answers it spans rather than counting a 1-answer level and a 999-answer
    level alike. `hits[i]` is how many of those are correct.

    Counts, not a risk already divided: `corrects` is 0.0/1.0, so every partial sum is an exact
    integer in float64 and `hits / accepted` reproduces `np.mean` of the accepted slice bit for
    bit. Returning `1 - hits / accepted` and recovering accuracy as `1 - risk` instead costs the
    last bits of 157,289 values in a 1.2-million-value sweep -- harmless to a gate, but it would
    make this change move numbers on datasets with no tie in them, which it is not allowed to do.
    """
    # `atleast_1d`: a 0-dimensional input (a bare scalar) answered before this change, and indexing
    # it with an argsort result raises.
    c = np.atleast_1d(np.asarray(confidences, dtype=float))
    # `corrects` stays boolean here: `cumsum` of a boolean array counts in int64, which is exact,
    # and skipping the float conversion saves a pass over the whole column.
    y = np.atleast_1d(np.asarray(corrects, dtype=bool))
    n = len(y)
    if n == 0:
        return np.empty(0), np.empty(0), 0
    order = np.argsort(-c, kind="mergesort")  # desc, stable; NaN sorts last
    c, y = c[order], y[order]
    # The last answer of each tie group: where the next confidence differs, plus the final one.
    boundary = c[1:] != c[:-1]
    # NaN needs `isnan` on both sides as well as `!=`, because NaN != NaN is true: without it every
    # NaN confidence becomes a level of its own, the running total walks them in dataset order, and
    # the order-dependence this exists to remove survives for exactly the input a custom runner is
    # most likely to hand over by accident. All NaNs are one group, ranked last.
    #
    # Guarded by one scalar test rather than paid for always: NaN sorts to the end, so if the last
    # confidence is not NaN then none are, and two full `isnan` passes over the column can be
    # skipped. That is the common case, and the two passes were a third of this function's cost at
    # a million rows.
    if np.isnan(c[-1]):
        boundary &= ~(np.isnan(c[1:]) & np.isnan(c[:-1]))
    last = np.flatnonzero(np.concatenate((boundary, [True])))
    accepted = last.astype(float) + 1.0
    return accepted, np.cumsum(y)[last], n


def _risk_coverage(confidences: Sequence[float], corrects: Sequence[bool]):
    """`(coverage, risk)` at each distinct confidence level.

    `_levels` with coverage as the x axis, for callers that want the curve rather than its area.
    """
    accepted, hits, n = _levels(confidences, corrects)
    if n == 0:
        return np.empty(0), np.empty(0)
    return accepted / n, 1.0 - hits / accepted


def aurc(confidences: Sequence[float], corrects: Sequence[bool]) -> Optional[float]:
    """Area Under the Risk-Coverage curve: selective risk integrated over coverage. Lower is better.

    A classifier whose confidence perfectly ranks right from wrong drives AURC toward the overall
    error rate's area under the ideal curve; a confidence no better than random leaves it at the
    base error rate. It rewards a confidence that *orders* answers, which calibration (ECE) does not.
    """
    if not confidences:
        return None
    accepted, hits, n = _levels(confidences, corrects)
    if n == 0:
        return None
    risk = 1.0 - hits / accepted
    # Each level's risk weighted by the ANSWERS it spans, which is what keeps this an area rather
    # than an average of unevenly sized points: with 999 answers at one confidence and 1 at another,
    # an unweighted mean of the two risks reports 5e-4 where the area is 1e-6.
    #
    # `sum(risk * rows) / n`, not `sum(risk * width)`: with no ties every level holds one answer, so
    # this reduces to the same pairwise sum over the same denominator and is bit-identical to the
    # per-answer mean it replaces. Dividing inside the sum instead loses that in the last bits.
    rows_per_level = np.diff(np.concatenate(([0.0], accepted)))
    return float(np.sum(risk * rows_per_level) / n)


def selective_accuracy(
    confidences: Sequence[float], corrects: Sequence[bool], coverage: float
) -> Optional[float]:
    """Accuracy over the most-confident `coverage` fraction of answers (0 < coverage <= 1).

    The cut is a confidence threshold, so it cannot land inside a group of equal confidences: when
    `coverage * n` falls within one, every member of that group is included and the reported
    coverage is the group's upper edge rather than the fraction asked for. That is the only
    order-independent reading -- see `_risk_coverage` -- and it is what makes this comparable
    across two permutations of the same dataset. Without ties, `k` is unchanged.

    The group this cut lands in comes from `_accepted_at_cut`, which answers for one cut what
    `_levels` answers for every level. The two are swept against each other by
    `test_a_single_cut_agrees_with_the_risk_coverage_curve`, so they cannot drift apart silently;
    deriving this from the full scan instead was measured at 19% to 58% slower here for no change
    in any value.
    """
    if not confidences or not 0.0 < coverage <= 1.0:
        return None
    import math as _math

    c = np.atleast_1d(np.asarray(confidences, dtype=float))
    order = np.argsort(-c, kind="mergesort")
    c = c[order]
    y = np.atleast_1d(np.asarray(corrects, dtype=bool))[order]
    k = max(1, int(_math.ceil(coverage * len(y))))
    # Extend through the tie group the cut fell in, so the cut is a confidence threshold rather
    # than a row index and cannot report whichever tie members arrived first.
    accepted = _accepted_at_cut(c, k)
    # `count_nonzero / accepted` rather than `mean`: both are exact for a boolean column, and this
    # skips the float conversion of the accepted slice.
    return float(np.count_nonzero(y[:accepted]) / accepted)


#: Coverage points reported as `selective_accuracy@NN`.
SELECTIVE_COVERAGES = (0.5, 0.8)


def is_coverage_metric(name: str) -> bool:
    """True for a metric whose value depends on where a coverage cut falls.

    `ece` and `brier` read the same `(confidence, correct)` pairs but do not cut, so a change to
    what a cut accepts cannot move them and a baseline that predates it stays comparable.
    """
    return name == "aurc" or name.startswith("selective_accuracy@")


def coverage_definition_conflict(candidate_config: Any, baseline: Any) -> Optional[str]:
    """Why a coverage metric must not be compared across these two documents, or None.

    Comparing `aurc` or `selective_accuracy@*` between two reports subtracts one number from the
    other, which is only meaningful if the same definition of a coverage cut produced both.
    Definition 2 cuts on a confidence threshold, so a cut never splits a group of equal
    confidences; definition 1 cut at a row index and depended on the order the dataset arrived in.
    Measured over 400,001 tie-shaped datasets the two disagree by up to 0.35 on `aurc` and 0.50 on
    `selective_accuracy@50`, so the subtraction is not merely noisy -- it is unsafe in the PASS
    direction, letting a real regression through.

    **Both sides are checked, not just the baseline.** A stale CANDIDATE is the more dangerous
    case and the easier one to arrive at by accident: a report produced by a pinned older `laya`
    in a container, or an archived artifact re-checked later, carries definition-1 numbers whose
    row-order artifact can read BETTER than the truth. Gating that against a correctly regenerated
    baseline passes a regression that the correctly scored report fails.

    The absence of the key IS definition 1: every report written since it existed carries it. That
    is why this refuses instead of following `comparable_to`'s "a key missing on either side is
    unknown rather than a conflict" rule, which exists so a patch release cannot invalidate a
    committed baseline. A definition change is not a patch release.

    `ece` and `brier` read the same pairs but do not cut, so they stay comparable; callers gate
    this behind `is_coverage_metric`.
    """
    here = _recorded_coverage_definition(candidate_config)
    there = _recorded_coverage_definition(
        baseline.get("config") if isinstance(baseline, dict) else None
    )
    if here == COVERAGE_METRIC_DEFINITION and there == COVERAGE_METRIC_DEFINITION:
        return None
    stale = []
    if here != COVERAGE_METRIC_DEFINITION:
        stale.append("this report says %s" % _shown_definition(here))
    if there != COVERAGE_METRIC_DEFINITION:
        stale.append("the baseline says %s" % _shown_definition(there))
    return (
        "%s, but a coverage cut here is definition %s -- regenerate it before comparing a "
        "coverage metric" % (" and ".join(stale), COVERAGE_METRIC_DEFINITION)
    )


def _recorded_coverage_definition(config: Any) -> Any:
    return config.get("coverage_metric_definition") if isinstance(config, dict) else None


def _shown_definition(recorded: Any) -> str:
    """A recorded definition, rendered for a message, with attacker-supplied content bounded.

    The value comes from a report file, which on a pull-request gate is content the author
    controls, and the message it lands in is printed to a CI log. A five-megabyte string in that
    field produced a five-megabyte failure line; 40 characters is enough to diagnose a real one.
    """
    if recorded is None:
        return "1 (unrecorded)"
    shown = repr(recorded)
    return shown if len(shown) <= 40 else shown[:40] + "... (truncated)"


def is_confidence_metric(name: str) -> bool:
    """True for a dataset-level metric computed from `(confidence, correct)` pairs, not per-answer
    `scores` -- ECE and the selective-classification metrics. Policy counting treats these alike."""
    return name in ("ece", "brier", "aurc") or name.startswith("selective_accuracy@")


def _answered_model(result: Any) -> Optional[Any]:
    """Which checkpoint produced `result`, read the way each runner records it.

    A `Router` puts its choice under `routing["model"]` and leaves no checkpoint name at the top
    level: `result["model"]` is the payload's family tag, `"laya-rl-agent"`, on every Agent and
    ONNXAgent result too. So a routed run has to read `routing`, or its `by model` slice reports
    one family for every checkpoint that answered.
    """
    result = result or {}
    routing = result.get("routing")
    if isinstance(routing, dict):
        chosen = routing.get("model")
        if isinstance(chosen, str) and chosen:
            return chosen
    return result.get("model")


def _markdown_cell(value: Any) -> str:
    """One slice label, written as a Markdown table cell rather than as table syntax.

    The group key is caller data straight off the dataset: `tags` and question ids are any string
    the author wrote, and `Example.from_dict` coerces them with `str` rather than restricting them.
    A table reader takes an unescaped `|` as the start of the next column and a newline as the end
    of the row, so a group labelled `risk|high` shifts every number in its row one header to the
    left and drops the last, and a label containing a newline takes its number out of the table
    altogether. The label is what a reader of `laya-evals run --markdown` matches a number against,
    so a shifted row is a wrong figure rather than a typo -- and only that group's row moves, which
    is what makes it hard to spot in a rendered report.

    `\\` is escaped before `|` so a backslash at the end of a label cannot swallow the escape that
    follows it. `\\|` is the repository's own convention for a literal pipe inside a cell: the
    parity table in `BENCHMARKS.md` spells its `max |p_fast - p_stock|` columns that way, and
    `tests/test_doc_tables.py` turns the escape back into a pipe to read the header.
    """
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    )


# --------------------------------------------------------------------------- evaluation
@dataclass
class EvalReport:
    """Overall and per-slice metrics, with the per-case records they were derived from."""

    config: Dict[str, Any] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    slices: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "overall": self.overall,
            "slices": self.slices,
            "cases": self.cases,
        }

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "|---|---|"]
        for name in sorted(self.overall):
            lines.append("| %s | %.4f |" % (name, self.overall[name]))
        for dimension in sorted(self.slices):
            lines.append("")
            lines.append("### by %s" % dimension)
            lines.append("")
            metrics = sorted({m for group in self.slices[dimension].values() for m in group})
            lines.append("| %s | %s |" % (dimension, " | ".join(metrics)))
            lines.append("|---|%s" % ("---|" * len(metrics)))
            for value in sorted(self.slices[dimension]):
                cells = [
                    "%.4f" % self.slices[dimension][value].get(m, float("nan"))
                    if m in self.slices[dimension][value]
                    else ""
                    for m in metrics
                ]
                lines.append("| %s | %s |" % (_markdown_cell(value), " | ".join(cells)))
        return "\n".join(lines) + "\n"

    def comparable_to(self, baseline: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Is `baseline` the same measurement as this report? Returns (ok, reasons).

        `compare` reads `overall` and only `overall`, so two reports of different datasets,
        different question schemas or different report shapes produce identical arithmetic and
        an identical pass. That is the failure `docs/staged-adoption.md` runs at: a gate that
        cannot tell which experiment produced a number cannot support a promotion decision.

        A key absent on either side is *unknown*, not a conflict, so every baseline committed
        before the identity existed -- including the scheduled gate's
        `research/results/eval_english_51_languages.json`, which comes from `research/eval/` and
        has no `config.schema` -- keeps comparing exactly as it did. Only a key present on both
        sides with different values refuses the comparison, and each reason names the key and
        both values so the failure is actionable from the console alone.

        `laya_version` and `thresholds` are recorded in `config` but deliberately not compared
        here: a patch release must not invalidate a committed baseline.
        """
        other = _identity_of(baseline)
        reasons: List[str] = []
        for key in _IDENTITY_KEYS:
            here, there = self.config.get(key), other.get(key)
            if here is None or there is None or here == there:
                continue
            reasons.append(
                "%s (%s): baseline is %s, this run is %s"
                % (key, _IDENTITY_LABELS[key], there, here)
            )
        if self.config.get("schema") == other.get("schema") == "laya-tournament-evals-report/1":
            here, there = (
                self.config.get("tournament_group_size"),
                other.get("tournament_group_size"),
            )
            if here is None or there is None or here != there:
                reasons.append(
                    "tournament group size: baseline is %s, this run is %s" % (there, here)
                )
        return (not reasons), reasons

    def compare(
        self,
        baseline: Dict[str, Any],
        tolerances: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """Compare `overall` to a baseline report's `overall`. Returns (ok, deltas).

        With no `tolerances`, every shared metric must match exactly; a tolerance is the maximum
        absolute difference allowed for that metric. A baseline metric the report no longer has
        fails the comparison (its delta carries ``missing: True`` and a NaN value): a run whose
        every example errored under ``on_error="skip"`` has an empty ``overall``, and must not
        pass the gate by having nothing left to compare. A NaN metric, baseline or tolerance
        fails the comparison too.

        Either side that is not a metric name to number map, a baseline with no metrics at all,
        and a tolerance that is not a number or is ``Infinity`` raise ``EvalError`` rather than
        reaching the subtraction: both documents are read from a path the caller names, and a shape
        the gate cannot compare -- or a limit that cannot fail -- is a broken artifact.
        See ``_metric_map`` and ``_tolerance_map``.
        """
        # A baseline may be a full report document or the metric map on its own, which is the
        # shape `research/evals/check_regression.py` builds per language.
        if not isinstance(baseline, dict):
            raise EvalError(
                "baseline report must be a JSON object, got %s" % type(baseline).__name__
            )
        base = _metric_map(baseline.get("overall", baseline), "baseline")
        overall = _metric_map(self.overall, "report", allow_empty=True)
        tolerances = _tolerance_map(tolerances)
        deltas: Dict[str, Dict[str, float]] = {}
        ok = True
        for metric, base_value in base.items():
            # Latency is informational; a re-run differs by timing noise, not quality, so it is
            # compared only when a tolerance explicitly names it.
            if metric.endswith("_ms") and metric not in tolerances:
                continue
            allowed = float(tolerances.get(metric, 0.0))
            # A coverage metric is only comparable when both documents were produced by the same
            # definition of a cut. This path is the one `docs/evals.md` leads with, and
            # `--gate-policy` is opt-in, so without this check the default gate performs exactly
            # the cross-definition subtraction the policy gate refuses.
            if is_coverage_metric(metric):
                conflict = coverage_definition_conflict(self.config, baseline)
                if conflict is not None:
                    deltas[metric] = {
                        "baseline": float(base_value),
                        "value": float(overall.get(metric, float("nan"))),
                        "diff": float("nan"),
                        "tolerance": allowed,
                        "incomparable": conflict,
                    }
                    ok = False
                    continue
            if metric not in overall:
                deltas[metric] = {
                    "baseline": float(base_value),
                    "value": float("nan"),
                    "diff": float("nan"),
                    "tolerance": allowed,
                    "missing": True,
                }
                ok = False
                continue
            value = overall[metric]
            diff = value - float(base_value)
            deltas[metric] = {
                "baseline": float(base_value),
                "value": value,
                "diff": diff,
                "tolerance": allowed,
            }
            # `not ... <=` rather than `>`: every comparison with NaN is False, so a NaN on
            # either side would otherwise pass as "did not move".
            if not abs(diff) <= allowed:
                ok = False
        return ok, deltas


def _group_cases(cases: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        if key == "tag":
            values = case.get("tags") or []
        else:
            value = case.get(key)
            values = [] if value is None else [value]
        for value in values:
            groups.setdefault(str(value), []).append(case)
    return groups


def _percentiles(values: Sequence[float]) -> Tuple[float, float]:
    """(median, 95th) using the nearest-rank rule for latency."""
    ordered = sorted(values)
    n = len(ordered)
    # Nearest-rank 95th percentile: the ceil(0.95 * n)-th smallest value (1-indexed).
    # `int(n * 0.95)` truncated where it needed to round up, returning the
    # (0.95n + 1)-th value whenever n is a multiple of 20. Integer ceil fixes that.
    rank = (95 * n + 99) // 100
    return (float(statistics.median(ordered)), float(ordered[rank - 1]))


def _aggregate(
    cases: Sequence[Dict[str, Any]], evaluators: Sequence[Evaluator]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for evaluator in evaluators:
        values = [
            c["scores"][evaluator.name]
            for c in cases
            if c.get("scores", {}).get(evaluator.name) is not None
        ]
        if values:
            out[evaluator.name] = float(statistics.fmean(values))
    paired = [
        (c["confidence"], c["correct"])
        for c in cases
        if c.get("confidence") is not None and c.get("correct") is not None
    ]
    if paired:
        confs = [p[0] for p in paired]
        corrs = [p[1] for p in paired]
        computed = {
            "ece": ece(confs, corrs),
            "brier": brier(confs, corrs),
            "aurc": aurc(confs, corrs),
        }
        for cov in SELECTIVE_COVERAGES:
            computed["selective_accuracy@%d" % round(cov * 100)] = selective_accuracy(
                confs, corrs, cov
            )
        for name, value in computed.items():
            if value is not None and not np.isnan(value):
                out[name] = float(value)
    return out


# The two batch call shapes the repo ships. `RouterRunner.predict_batch(states, questions,
# model=..., batch_size=...)` is the positional one; `Router.predict_batch(requests, ...)` takes
# one dict per request, the form `route_batch` and the `laya-evals` CLI document.
_BATCH_STATES = "states"
_BATCH_REQUESTS = "requests"


def _batch_form(runner: Any) -> Optional[str]:
    """Return which batch call `runner` accepts, or ``None`` to score it one ``predict`` at a time.

    Deliberately not ``hasattr(runner, "predict_batch")``: having a batch entry point and being
    callable the way this harness calls it are different claims, and the attribute test returned
    true for ``Router`` -- the one real runner whose ``predict(state, questions, model=...)``
    matches the contract `evaluate` states. Every chunk of more than one example then raised
    ``TypeError: ... unexpected keyword argument 'model'``, which under ``on_error="skip"`` became
    a report with zero cases and an empty ``overall``.
    """
    fn = getattr(runner, "predict_batch", None)
    if fn is None:
        return None
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        # Not introspectable (a C-level or hand-rolled ``__call__``): keep the call this harness
        # made before the shape was checked, rather than silently dropping to single predicts.
        return _BATCH_STATES
    if any(p.name == "model" or p.kind is p.VAR_KEYWORD for p in params):
        return _BATCH_STATES
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    # A bound method, so `positional[0]` is the batch argument itself.
    if positional and positional[0].name == "requests":
        return _BATCH_REQUESTS
    return None


def _takes_sort_by_length(runner: Any) -> bool:
    """Whether `runner`'s batch entry point can be given the length-grouping knob.

    The same signature check `_batch_form` makes, for the one optional argument this harness
    forwards: `sort_by_length` is an optimisation, so a runner whose ``predict_batch`` predates it
    (#294) is scored unsorted rather than raising ``TypeError`` at the first chunk of a long run.
    A ``**kwargs`` forwarder counts, because whatever it wraps is a real runner.
    """
    fn = getattr(runner, "predict_batch", None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "sort_by_length" or p.kind is p.VAR_KEYWORD for p in params)


def _takes_min_confidence(runner: Any, fn_name: str = "predict_batch") -> bool:
    """Whether `runner`'s entry point accepts an abstention threshold on the call.

    The same signature check `_takes_sort_by_length` makes, applied to whichever entry point the
    harness is about to call. The gate is a *reporting* control, not a scoring one: it writes
    `low_confidence: True` and `abstention: "abstained"` on answers whose `answer_confidence`
    falls below the threshold, and leaves `answer["choice"] / ["noul"] / ["score"]` as the raw
    argmax -- so every metric in `_aggregate` reads the same numbers at every threshold. What
    changes is `report.config["timing"]["min_confidence"]` and `["min_confidence_sent"]`: those
    keys claim the gate ran. A runner that predates it cannot honour that claim, so when the
    guard is false the harness raises rather than publish a threshold it never applied. The CLI
    catches the raise into a pre-flight message before any checkpoint loads.
    """
    fn = getattr(runner, fn_name, None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "min_confidence" or p.kind is p.VAR_KEYWORD for p in params)


def _takes_model(runner: Any, fn_name: str = "predict") -> bool:
    """Whether `runner`'s entry point accepts a per-example checkpoint on the call.

    The same signature check `_batch_form` makes, applied to the `model` kwarg the
    single-row path forwards from `Example.model` (#1107). The batch half of this exact mistake
    was `_batch_form`'s founding bug -- every chunk raised ``TypeError: ... unexpected keyword
    argument 'model'``, which `on_error="skip"` turned into a zero-case report -- and
    `laya.load()`'s default runner still hit it one row at a time, because `Agent.predict` is
    `system_one`, which takes neither `model` nor any `**kwargs` forwarder. Unlike the threshold,
    an unsent `model` is no lie: a runner without the parameter answers from its own checkpoint.
    So the harness drops the kwarg when no example names one, and the pre-flight refusal in
    `evaluate` fires only when an example does.

    Not introspectable (a C-level or hand-rolled ``__call__``): keep the call this harness makes,
    the fallback `_batch_form` makes for the same kwarg on ``predict_batch`` -- deliberately not
    `_takes_min_confidence`'s ``False``, which had the two guards disagreeing: on a runner whose
    entry points are both C-level, `model` was sent on the batch path and dropped on the single
    one, and a dataset naming an explicit checkpoint was refused even though the batch entry point
    would have carried it (#1112 review). With no example naming a model the value is ``None``
    either way, so the only run the ``True`` fallback changes is the one where sending it is the
    point; a runner whose attribute is missing is the `fn is None` case, and a C-level runner that
    refuses the kwarg fails loudly on the call -- the risk `_batch_form` already accepts.
    """
    fn = getattr(runner, fn_name, None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        # Same fallback as `_batch_form` (its comment records why): keep the call this harness
        # made before the shape was checked, rather than half-dropping the kwarg.
        return True
    return any(p.name == "model" or p.kind is p.VAR_KEYWORD for p in params)


def evaluate(
    runner: Any,
    dataset: Dataset,
    evaluators: Optional[Sequence[Evaluator]] = None,
    batch_size: Optional[int] = None,
    on_error: str = "fail",
    config: Optional[Dict[str, Any]] = None,
    sort_by_length: bool = False,
    min_confidence: Optional[float] = None,
) -> EvalReport:
    """Run `runner` over `dataset`, aggregating per-answer metrics overall and per slice.

    `runner` needs a ``predict(state, questions, model=...)`` method, and for `batch_size` above 1
    a ``predict_batch`` in either shape the repo ships: the positional
    ``predict_batch(states, questions, model=..., batch_size=...)``, or the per-request-dict form
    `Router.predict_batch` takes -- ``predict_batch([{"state": ..., "questions": ...,
    "model": ...}, ...], batch_size=...)``. A runner that offers neither is scored one ``predict``
    at a time, which is slower but not wrong. A runner whose ``predict`` takes no ``model`` --
    ``laya.load()``'s default `Agent` is one, its ``predict`` being an alias of `system_one`
    (#1107) -- is scored on its own checkpoint: the kwarg is dropped unless an example names an
    explicit ``model``, and a dataset that names one on such a runner is refused in pre-flight
    rather than published as a silent zero-case report.

    `sort_by_length` is forwarded to a chunked runner so similarly sized examples share a forward
    pass and pad to a shorter maximum. It reaches the checkpoint's own batching only when the
    runner's ``predict_batch`` takes it, and only when it is on: an unset control is not sent, so a
    runner that predates the knob is unaffected by a run that does not ask. Nothing about the
    scored answers changes -- the results come back in chunk order either way.

    `min_confidence` is the opt-in abstention threshold `Router` and `ONNXAgent` apply to
    `answer_confidence` (#361). `laya.confidence.apply_confidence_gate` marks every answer below
    it with `low_confidence: True` and `abstention: "abstained"`, and the raw argmax stays on
    `answer["choice"] / ["noul"] / ["score"]` -- so `_aggregate` publishes the same accuracy,
    calibration and coverage numbers at every threshold. Like `sort_by_length`, nothing about the
    scored answers changes. Unlike `sort_by_length`, the report itself names the threshold under
    `report.config["timing"]["min_confidence"]` and asserts it was sent under
    `["min_confidence_sent"]`, so a runner whose batch entry point (or whose single ``predict``,
    on the fallback path) predates the gate is refused: the harness will not publish a
    `min_confidence` it did not apply.

    `on_error` is ``"fail"`` (re-raise a runner error) or ``"skip"`` (record it and continue),
    the latter for evaluating a flaky fleet without aborting the whole run.

    A batched run has two honest timing answers, so the report gives both. ``latency_p50_ms`` /
    ``latency_p95_ms`` are per request: every row of a chunk returns from the same call, so each
    one waited that whole call. ``cost_per_decision_p50_ms`` / ``cost_per_decision_p95_ms``
    divide a call by its own chunk size, which is the throughput figure. ``--batch-size`` therefore
    lowers the second and raises the first. What the harness really did -- the runner shape it
    resolved to and how many rows shared a call -- lands in ``report.config["timing"]``, because the
    requested flag alone does not say whether anything was batched.
    """
    if on_error not in ("fail", "skip"):
        raise EvalError("on_error must be 'fail' or 'skip', got %r" % on_error)
    # `min_confidence` is validated here -- and, unlike `sort_by_length`, a run that asks for it
    # on a runner that does not accept it is refused. The gate does not change which answers score
    # as correct: `apply_confidence_gate` writes `low_confidence` and `abstention` state fields
    # while leaving `answer["choice"] / ["noul"] / ["score"]` as the raw argmax, so `_aggregate`
    # publishes the same numbers at every threshold. What would be a lie is
    # `report.config["timing"]["min_confidence"]`: it names the threshold as if the run applied
    # it. A runner that predates the gate (#361) cannot honour that claim, so the guard raises.
    # `laya.confidence.check_min_confidence` is the same validator the Router uses, so the
    # accepted range cannot drift from what the abstention gate itself enforces.
    try:
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
    except ValueError as exc:
        # Core's validator raises a bare ValueError; the CLI's usage-error handler catches EvalError
        # and turns it into exit 2 with a printed message. Re-raising here keeps a mistyped
        # `--min-confidence 1.5` on the same path as a bad tolerance, instead of a traceback.
        raise EvalError(str(exc)) from exc
    if mc is not None:
        # Whichever entry points this run can actually reach all have to take the threshold. A
        # batched run still sends a chunk of one through `predict`, so requiring only
        # ``predict_batch`` would let the runner silently score those rows without the gate --
        # the exact class of lie the guard exists to prevent.
        targets = ["predict"]
        if batch_size is not None and batch_size > 1 and _batch_form(runner):
            targets.append("predict_batch")
        for target in targets:
            if not _takes_min_confidence(runner, target):
                raise EvalError(
                    "evaluate(min_confidence=%r) refused: this runner's %s does not accept the "
                    "abstention threshold. Either use a Router/ONNXAgent that gates on "
                    "answer_confidence (#361), or drop the threshold -- the report would "
                    "otherwise name a `min_confidence` it never applied." % (min_confidence, target)
                )
    # Whether `predict` can be handed `Example.model` at all (#1107). `Agent.predict` is
    # `system_one` -- no `model`, no `**kwargs` -- so the single-row path used to raise `TypeError`
    # on every row, and `on_error="skip"` published a zero-case report. The batch half was already
    # gated by `_batch_form`; this is the single-row guard, computed once and reused at the call.
    takes_model = _takes_model(runner)
    named = next(((i, e) for i, e in enumerate(dataset.examples) if e.model is not None), None)
    if not takes_model and named is not None:
        # Dropping the kwarg here would score each row on whatever checkpoint the runner answers
        # with while the report's `model` slice labels it with the named one -- the same class of
        # lie the `min_confidence` guard exists to prevent. The kwarg is dropped only when no
        # example names a checkpoint, in which case scoring on the runner's own is what was asked.
        # The refusal names the first offender: on a large dataset, "the dataset names a `model`"
        # alone leaves the user grepping for the row (#1112 review).
        first, offender = named
        raise EvalError(
            "evaluate() refused: example %d names model %r, but this runner's predict does not "
            "accept a `model` argument. Either use a Router/ONNXAgent whose predict forwards the "
            "checkpoint, or drop the per-example `model` -- the report would otherwise name a "
            "checkpoint it never sent (#1107)." % (first, offender.model)
        )
    evaluators = list(evaluators) if evaluators is not None else default_evaluators()
    cases: List[Dict[str, Any]] = []
    waits: List[float] = []  # what each request actually waited: its chunk's whole call
    shares: List[float] = []  # that call split across its chunk: throughput per decision
    chunks = rows_grouped = rows_alone = max_chunk = 0
    errors: List[Dict[str, Any]] = []
    examples = dataset.examples
    # Only worth grouping if the runner can be handed the group in one call at all.
    batch_form = _batch_form(runner) if batch_size is not None and batch_size > 1 else None
    # The chunk grouping hashes the questions each row carries, but a `questions` dict is
    # only equal as written: the same choice schema with two `criteria` dicts written in a
    # different key order is a different signature, so rows that would share one forward pass
    # are scored alone. Canonicalizing the key order is only safe where each row reaches the
    # runner with its own `questions` object -- the per-request `_BATCH_REQUESTS` form. The
    # positional `_BATCH_STATES` form hands the whole chunk `chunk[0].questions`, so folding
    # key order there would render a later row with an earlier row's positional `criteria`
    # order and silently move the metric. Gate the fold on the call shape: `sort_keys=True`
    # for per-request runners, `sort_keys=False` (insertion order significant) for positional
    # ones. This is a harness-side grouping relaxation and it is looser than core, which never
    # relaxes: both `Router._question_schema` and `_canonical` above hash `sort_keys=False`
    # unconditionally. On the per-request path two key orderings that share one harness chunk
    # can therefore still split into separate groups inside the Router, which costs throughput
    # and never correctness, because each request carries its own `questions`. A list-valued
    # `criteria` is positional on both paths, and `sort_keys` leaves lists alone.
    folded = batch_form == _BATCH_REQUESTS
    chunk_signatures = [
        json.dumps(dict(sorted((example.questions or {}).items())), sort_keys=folded, default=str)
        for example in examples
    ]
    # The chunk shape travels with the calls that have one: a chunk of a single example is a plain
    # `predict`, which has no batch to reorder, and an off control is not sent at all.
    shape = (
        {"sort_by_length": True}
        if sort_by_length and batch_form and _takes_sort_by_length(runner)
        else {}
    )
    # `min_confidence` joins the batch shape whenever a chunk can go through `predict_batch`, and
    # the single-`predict` path picks it up below. The pre-flight refusal above has already
    # proven every entry point this run can reach accepts it, so no re-check sits here: a run
    # that reaches this line either asked for the gate and will get it on every call, or asked
    # for nothing.
    if mc is not None and batch_form:
        shape = dict(shape, min_confidence=mc)

    index = 0
    while index < len(examples):
        # Batches only when the runner can share a forward pass: consecutive examples with the
        # same checkpoint and identical questions. Otherwise every example is one predict.
        chunk = [examples[index]]
        if batch_form:
            while (
                index + len(chunk) < len(examples)
                and len(chunk) < batch_size
                and (examples[index + len(chunk)].model, chunk_signatures[index + len(chunk)])
                == (examples[index].model, chunk_signatures[index])
            ):
                chunk.append(examples[index + len(chunk)])
        # Why these sit above the call and `waits`/`shares` below: the shape counters answer "what
        # did the harness issue", which is settled the moment the chunk is grouped, while the
        # metric lists answer "what did a request wait", which needs a call that returned. Counting
        # only completed calls made a run that issued one shared forward and lost it to an error
        # under `on_error="skip"` report the same `rows_grouped` and `max_chunk` as a runner with
        # no batching at all (#592 review).
        chunks += 1
        if len(chunk) > 1:
            rows_grouped += len(chunk)
        else:
            rows_alone += 1
        if len(chunk) > max_chunk:
            max_chunk = len(chunk)
        started = time.perf_counter()
        try:
            if len(chunk) > 1:
                if batch_form == _BATCH_REQUESTS:
                    # The chunk already shares one checkpoint and one question schema, so the
                    # per-request dicts carry exactly what the positional call would pass.
                    results = runner.predict_batch(
                        [
                            {"state": e.state, "questions": e.questions, "model": e.model}
                            for e in chunk
                        ],
                        batch_size=batch_size,
                        **shape,
                    )
                else:
                    results = runner.predict_batch(
                        [e.state for e in chunk],
                        chunk[0].questions,
                        model=chunk[0].model,
                        batch_size=batch_size,
                        **shape,
                    )
            else:
                # A chunk of one still sees the gate: the pre-flight refusal proved `predict`
                # accepts it whenever `mc` is set, so nothing here has to re-check the signature.
                # The `model` kwarg is the same honesty (#1107): sent only when `predict` takes
                # it -- kept as-is when the signature cannot be read, `_batch_form`'s fallback on
                # the same kwarg -- and the pre-flight already refused any run where an example
                # needed it sent and the guard said no.
                single_kwargs = {"min_confidence": mc} if mc is not None else {}
                if takes_model:
                    results = [
                        runner.predict(
                            chunk[0].state,
                            chunk[0].questions,
                            model=chunk[0].model,
                            **single_kwargs,
                        )
                    ]
                else:
                    results = [runner.predict(chunk[0].state, chunk[0].questions, **single_kwargs)]
        except Exception as exc:  # noqa: BLE001 -- honoured by on_error
            if on_error == "fail":
                raise
            errors.extend(
                {"index": index + offset, "error": "%s: %s" % (type(exc).__name__, exc)}
                for offset in range(len(chunk))
            )
            index += len(chunk)
            continue
        # One result per row is the batch contract; `zip` below would silently pair the rest of
        # the chunk with nothing, so a runner that drops a row would score a subset while
        # `rows_grouped` still counted every row -- a report that under-scores itself, and a
        # baseline gate that can pass on the subset.
        results = list(results)
        if len(results) != len(chunk):
            raise EvalError(
                "runner.predict_batch returned %d results for a chunk of %d rows; every row "
                "needs its own result" % (len(results), len(chunk))
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        # A chunk is one call, so each of its rows waited all of `elapsed`; the split figure is a
        # throughput share, not a latency, and putting the two in one list made a batched run
        # publish a per-decision cost under the name of the per-request one (#585).
        waits.extend([elapsed] * len(chunk))
        shares.extend([elapsed / len(chunk)] * len(chunk))
        for offset, (example, result) in enumerate(zip(chunk, results)):
            answers = (result or {}).get("answers") or {}
            for qid, expected in example.expected.items():
                answer = answers.get(qid)
                if not isinstance(answer, dict):
                    if on_error == "fail":
                        raise EvalError("runner returned no answer for question %r" % qid)
                    errors.append(
                        {"index": index + offset, "question": qid, "error": "missing answer"}
                    )
                    continue
                case = {
                    "qid": qid,
                    "language": example.language,
                    "model": example.model or _answered_model(result),
                    "tags": list(example.tags),
                    "expected": expected,
                    "answer": answer,
                    "confidence": _answer_confidence(answer),
                    "correct": _correct(answer, expected),
                    "scores": {
                        evaluator.name: evaluator.score(answer, expected)
                        for evaluator in evaluators
                    },
                }
                # Shortlist metadata describes the retrieval stage, not the answer. Keep it
                # beside the case so opt-in evaluations can attribute the source of an error.
                shortlist = (result or {}).get("shortlist")
                if isinstance(shortlist, dict) and qid in shortlist:
                    case["shortlist"] = shortlist[qid]
                tournament = (result or {}).get("tournament")
                if isinstance(tournament, dict) and qid in tournament:
                    case["tournament"] = tournament[qid]
                cases.append(case)
        index += len(chunk)

    report = EvalReport(config=dict(config or {}), cases=cases)
    report.overall = _aggregate(cases, evaluators)
    if waits:
        report.overall["latency_p50_ms"], report.overall["latency_p95_ms"] = _percentiles(waits)
        (report.overall["cost_per_decision_p50_ms"], report.overall["cost_per_decision_p95_ms"]) = (
            _percentiles(shares)
        )
    # The run identity, stamped here rather than left to the caller: everything below is
    # computable from `dataset`, so a programmatic `evaluate` is as identifiable as a CLI run.
    # The CLI adds `dataset_sha256` and `thresholds`, which need the file path and the gate.
    report.config = dict(report.config, **_run_identity(dataset))
    report.config = dict(
        report.config,
        timing={
            "latency_metric": "per request: the wall time of the call that returned it, unsplit",
            "cost_metric": "per decision: that call divided by its own chunk size",
            "batch_size": batch_size,
            "batch_form": batch_form,
            "sort_by_length": sort_by_length,
            # Requested and sent are different claims: `sort_by_length` needs a chunk to reorder, so a
            # run with no batch form asked for something this harness cannot do.
            "sort_by_length_sent": bool(shape.get("sort_by_length")),
            # Same shape for `min_confidence`: a run that asked for a threshold and landed on a
            # chunked path where the runner accepted it gets a different report from one that was
            # refused at the guard, and the report should say which. The `_sent` value reflects
            # whether at least one call actually carried the threshold, not whether it was requested.
            "min_confidence": min_confidence,
            "min_confidence_sent": (
                # The pre-flight refusal proved the entry points this run reaches accept the
                # threshold, so a non-None `mc` on a run that made at least one call is exactly the
                # claim "the threshold was on every call". An empty dataset sends nothing.
                mc is not None and chunks > 0
            ),
            "chunks": chunks,
            "rows_grouped": rows_grouped,
            "rows_alone": rows_alone,
            "max_chunk": max_chunk,
        },
    )
    for dimension in ("language", "model", "qid", "tag"):
        groups = _group_cases(cases, dimension)
        if groups:
            report.slices[dimension] = {
                value: _aggregate(group, evaluators) for value, group in groups.items()
            }
    if errors:
        report.config = dict(report.config, errored=errors)
    return report


def assert_regression(
    report: EvalReport, baseline: Dict[str, Any], tolerances: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, float]]:
    """Raise AssertionError when `report` drifts from `baseline` beyond `tolerances`."""
    ok, deltas = report.compare(baseline, tolerances)
    if not ok:
        # `compare()` fails three shapes, and only the numeric one makes `abs(diff) > tolerance`
        # true: a missing metric and an incomparable coverage definition both store a NaN diff
        # (laya/evals.py:928-936), and every comparison with NaN is False, which is exactly why
        # compare() tests `not abs(diff) <= allowed` (comment at laya/evals.py:946-949). Filtering
        # on `>` alone dropped both, so the gate raised "evaluation regressed: {}" and lost the
        # metric and the reason compare() had already computed.
        failed = {
            m: d
            for m, d in deltas.items()
            if d.get("missing") or d.get("incomparable") or not abs(d["diff"]) <= d["tolerance"]
        }
        raise AssertionError("evaluation regressed: %s" % json.dumps(failed, sort_keys=True))
    return deltas
