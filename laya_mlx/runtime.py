"""Host prediction methods ported from official Laya v0.4.34.

MLX adaptation is restricted to tensor execution and resource cleanup.
"""

import copy
import json
import os
import tempfile
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .calibrate import (
    MIN_BINNING_BUCKET_N,
    _install_temperatures,
    apply_binning_map,
    apply_calibration_payload,
    calibration_payload,
    fit_binning_map,
    fit_temperature_map,
)
from .common import (
    QTYPES,
    _disable_question_token_reuse,
    _resolve_noul_labels,
    _reuse_question_tokens,
    answer_confidence,
    build_sequence,
    check_batch_size,
    collapsed_options,
    collate_items,
    confidence_from_probs,
    encode_text,
    render_criterion,
    resolve_lang_temperatures,
    serialize_state,
    state_room,
    temp_bucket,
    unpermute_probs,
    window_batch_cap,
    window_budget,
)
from .confidence import apply_confidence_gate, check_min_confidence
from .hooks import (
    HookRegistry,
    PredictContext,
    _as_sequence,
    aggregate_usage,
    compose_hooks,
    dispatch,
    normalise_hooks,
    validate_timeout,
)


def _option_count(qdef: Dict) -> int:
    """How many options a validated question definition renders to.

    Mirrors `render_options`: a choice has one option per criterion, a score one per level,
    and a noul is always the pair [false, true].
    """
    if qdef.get("type") == "noul":
        return 2
    crit = qdef.get("criteria")
    return len(crit) if isinstance(crit, (dict, list, tuple)) else 0


def _scan_answer_values(answer):
    # Reliability/action annotations do not change which prediction the model made. Keep the
    # prediction fields independent of in-place edits to probabilities or a score's legend.
    return {
        key: answer[key]
        for key in (
            "type",
            "choice",
            "noul",
            "score",
            "legend",
            "probabilities",
            "confidence",
            "answer_confidence",
        )
        if key in answer
    }


def _scan_answer_index(evidence, qid, answer, fallback):
    recorded = evidence["answers"]
    if recorded is None:
        return fallback  # a replacement batch never dispatched hooks
    record = recorded.get((qid, id(answer)))
    if record is not None and record[0] is answer and record[2] == _scan_answer_values(answer):
        return record[1]
    return None


def _start_evidence(track_answers=False):
    """A recorder for `predict_long`: what the start-hook chain left for inference to run on.

    Why a recorder at all: `ctx.skip()` only assigns `ctx.results` (see `laya.hooks.PredictContext`),
    so outside `predict_batch` the only account of what the hooks did is the context they were
    handed. The result count cannot tell a hook that answered the document from one that replaced
    the window list -- both return fewer results than there were windows, and the two need opposite
    readings: the first scored no window, the second scored the ones it left behind and booked
    tokens for them.

    The probe belongs last in the chain because `compose_hooks` orders defaults, then installed,
    then per-call hooks, and `dispatch` calls them in order, so a probe appended after the caller's
    own start hooks observes exactly what `predict_batch` is about to act on.

    Returns the probe and the dict it fills: `answered` is whether a hook replaced the call before
    inference, `states` is a snapshot of the states that reached it (`None` if the probe never ran,
    which means `predict_batch` was replaced and no hook chain was dispatched).
    """
    evidence = {
        "answered": False,
        "states": None,
        "question_types": None,
        "answers": None,
        "max_len": None,
        "head_max_len": None,
        "questions": None,
    }

    def record_results(ctx):
        if evidence["answered"]:
            return
        recorded = {}
        for index, result in enumerate(ctx.results):
            for qid, answer in result["answers"].items():
                key = (qid, id(answer))
                # Keep the object alive as well as its id. One object emitted for more than
                # one window has no unique origin, even when all prediction fields agree.
                recorded[key] = (
                    None
                    if key in recorded
                    else (answer, index, copy.deepcopy(_scan_answer_values(answer)))
                )
        evidence["answers"] = recorded

    def probe(ctx):
        evidence["answered"] = ctx.results is not None
        evidence["states"] = list(ctx.states)
        # Snapshot only the inference schema; leave malformed questions to predict_batch's
        # validator, and do not inspect questions when a hook has already answered the call.
        if ctx.results is None and isinstance(ctx.questions, dict):
            evidence["question_types"] = {
                qid: qdef.get("type") if isinstance(qdef, dict) else None
                for qid, qdef in ctx.questions.items()
            }
        # Also what the budget and the questions ACTUALLY are once the chain has run, for
        # `_check_scan_budget`. Recorded, never judged here: a hook that raises is governed by the
        # caller's `hooks_raise`/`hooks_timeout`, so a correctness check that raises from inside the
        # chain can be switched off by a policy meant for third-party telemetry. Measured: with
        # `hooks_raise=False` -- what `docs/hooks/tracing.md` recommends -- a refusal became a
        # RuntimeWarning blaming `_StartAdapter`, and the scan proceeded.
        evidence["max_len"] = ctx.max_len
        evidence["head_max_len"] = ctx.head_max_len
        evidence["questions"] = dict(ctx.questions) if ctx.questions else {}
        if track_answers:
            # predict_batch calls this before ANY end hook, including defaults and installed
            # hooks. An end observer appended to the hook list would already see reordered rows.
            ctx._record_scan_results = record_results

    return probe, evidence


def _check_scan_budget(agent, evidence, sized, cfg_max_len, cfg_head_max_len, asked=None):
    """Raise when the budget or questions in force leave less room than the scan was sized for.

    `predict_long` sizes its windows from the agent's config, before any hook has run. A start hook
    may then set `ctx.max_len`/`ctx.head_max_len`, or rewrite `ctx.questions` -- both documented
    powers -- and `build_sequence` uses whatever it finds. When the head widens faster than
    `max_len`, or the questions get more options, the real room SHRINKS: every window is
    re-truncated on the way in, and once the stride exceeds the real room consecutive windows stop
    touching. Measured on the English checkpoint with `widen_for_high_cardinality` from
    `docs/hooks/patterns.md` at 50 options, windows sized 303 against a room of 253; at 100 options
    a reviewer measured 43.4% of a document reaching no model, against the 37.6% this change exists
    to remove. Silently worse than not windowing at all.

    Called by `predict_long` AFTER the chain has run, not raised from inside it. A hook that raises
    is subject to the caller's `hooks_raise` and `hooks_timeout`, so the first version of this check
    could be switched off by `hooks_raise=False` -- which `docs/hooks/tracing.md` recommends -- and
    could blow a `hooks_timeout` the caller set for their own hooks. The cost is that the forward
    pass has already happened when this fires: a misconfigured scan is refused rather than answered
    wrongly, but it is not refused for free. Sizing the windows after the chain instead would change
    when hooks run and what they see, which is a larger change than this one.
    """
    if evidence.get("answered"):
        return  # a hook answered the document; no window was scored
    questions = evidence.get("questions")
    if not questions:
        return  # nothing to fit, as `window_budget` also concludes
    eff_max_len = evidence.get("max_len")
    eff_head = evidence.get("head_max_len")
    eff_max_len = cfg_max_len if eff_max_len is None else eff_max_len
    eff_head = cfg_head_max_len if eff_head is None else eff_head
    if (
        eff_max_len == cfg_max_len
        and eff_head == cfg_head_max_len
        and (asked is None or questions == asked)
    ):
        # Nothing the scan was sized against moved, so there is nothing to recompute -- and
        # recomputing anyway costs a tokenization of every question head, which `window_budget`
        # already paid and `_encode_state` will pay again. That showed up as a third head
        # tokenization in `test_question_token_reuse`, which asserts two.
        return
    try:
        internal = [agent._to_internal(q) for q in questions.values()]
    except Exception:
        return  # malformed questions are the validator's to report
    if not internal:
        return
    room = min(state_room(agent.tok, q, eff_max_len, eff_head) for q in internal)
    if room >= sized:
        return
    raise ValueError(
        "predict_long: after the start hooks ran, the questions and token budget (max_len=%d "
        "head_max_len=%d) leave %d state tokens per window, but the scan was sized for %d from the "
        "agent's config (max_len=%d head_max_len=%d). Every window would be re-truncated and parts "
        "of the document would reach no model. Pass window=/stride= explicitly, or call "
        "predict_batch directly, if you need a start hook to change either."
        % (eff_max_len, eff_head, room, sized, cfg_max_len, cfg_head_max_len)
    )


def _with_start_probe(hook_kwargs, probe):
    """`hook_kwargs` with `probe` appended after the caller's own start hooks.

    `probe` may be one hook or a list of them, appended in order.
    """
    kwargs = dict(hook_kwargs)
    extra = list(probe) if isinstance(probe, (list, tuple)) else [probe]
    kwargs["on_predict_start"] = list(_as_sequence(hook_kwargs.get("on_predict_start"))) + extra
    return kwargs


def _option_logits(logits, items, offset):
    """Raw per-option logits, the rows `_decode_answers` divides by temperature.

    Calibration record collection slices with this same helper, so a fitted map sees the
    option width the decoder scales and not a second tokenization of the state.
    """
    return [logits[offset + j, : len(item["markers"])] for j, item in enumerate(items)]


class RuntimeMixin(HookRegistry):
    @staticmethod
    def _check_question(qid: str, qdef: Any) -> None:
        """Reject a question that cannot be answered, naming it and what to fix.

        `render_options` reads `criteria` in the shape the question's type expects and the decision
        head needs at least one option, so a malformed definition used to surface from three frames
        down as something that names neither the question nor the problem: `AttributeError:
        'NoneType' object has no attribute 'items'`, `KeyError: 'bool'`, or a `selected index k out
        of range` raised inside the model for a question that ended up with no options at all.
        """
        if qid is None:
            raise ValueError("question id must not be None")
        if not isinstance(qid, (str, int)) or (isinstance(qid, str) and not qid.strip()):
            raise ValueError("question id must be a non-empty string, got %r" % (qid,))
        if not isinstance(qdef, dict):
            raise ValueError(
                "question %r: definition must be a dict, got %s" % (qid, type(qdef).__name__)
            )
        t = qdef.get("type")
        if not isinstance(t, str) or t not in QTYPES:
            raise ValueError(
                "question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES))
            )
        if "instructions" not in qdef:
            raise ValueError(
                "question %r: no 'instructions'; add the text the model should answer" % (qid,)
            )
        ins = qdef["instructions"]
        if ins is None:
            raise ValueError(
                "question %r: 'instructions' must not be None; add the text the model should answer"
                % (qid,)
            )
        if isinstance(ins, str) and not ins.strip():
            raise ValueError(
                "question %r: 'instructions' must not be empty; add the text the model should answer"
                % (qid,)
            )
        if isinstance(ins, (list, dict)) and not ins:
            raise ValueError(
                "question %r: 'instructions' must not be empty; add the text the model should answer"
                % (qid,)
            )
        if not isinstance(ins, (str, dict, list, int, float)):
            raise ValueError(
                "question %r: 'instructions' must be a string, dict, or list, got %s"
                % (qid, type(ins).__name__)
            )
        crit = qdef.get("criteria")
        if t == "choice":
            if not isinstance(crit, (dict, list)):
                raise ValueError(
                    "question %r: a choice question takes 'criteria' as a dict of "
                    "label -> description, or a list of labels" % (qid,)
                )
            if not crit:
                raise ValueError(
                    "question %r: a choice question needs at least one criterion" % (qid,)
                )
            # A label is used as a dict key when a list of labels is normalised, so an unhashable
            # label raised `TypeError` from three frames down -- which names neither the question nor
            # the label, and which `serve` cannot classify as a caller error, so over HTTP it became
            # a 500 "inference failed" instead of a 422. Requiring the documented scalars exactly
            # also catches the hashable-but-not-scalar shapes (`tuple`, `frozenset`, `bytes`,
            # `complex`) that the old deny-list let through to the same 500. Labels are rendered as
            # option text, so a nested structure has no meaning here.
            for i, label in enumerate(crit if isinstance(crit, list) else crit.keys()):
                if label is not None and not isinstance(label, (str, int, float, bool)):
                    raise ValueError(
                        "question %r: choice label %d is a %s; a label is rendered as option text "
                        "and used as the answer key, so it must be a scalar (a string, number or "
                        "bool), got %r" % (qid, i, type(label).__name__, label)
                    )
                if label is None:
                    # The score path rejects a null level with the same argument one question type
                    # over: a label is option text AND the answer key, and a null one can be
                    # neither. `_to_internal` normalises a list of labels to `{label: None}`, so a
                    # null label became the option text "None" (`str(None)`) while the answer key
                    # and the `probabilities` key for that same option were the JSON string "null",
                    # because a dict key has to be a str. A client then cannot tell whether that
                    # option was the *string* `"null"` or JSON `null`, and the answer key is
                    # unreachable: `criteria[answer["choice"]]` yields `None`, and
                    # `answer["choice"] == "null"` is False for it. `""` stays accepted, because
                    # unlike a null it does round-trip.
                    raise ValueError(
                        "question %r: choice label %d is null; a label is rendered as option text "
                        "and used as the answer key, so it must be a string, number or bool -- a "
                        'null label renders as the text "None" while its answer key is "null"'
                        % (qid, i)
                    )
            # `_to_internal` normalises the list form to `{label: None}`, so those labels become
            # the answer keys. Two entries that land on one key made the model score fewer options
            # than the caller wrote, and the response carry fewer probabilities than their list,
            # without a word -- the same silent-shape class as the two checks above. Python
            # treats values as one key whenever they compare equal, so `[1, 1.0]` and `[True, 1]`
            # collapse as well as an exact repeat. An unhashable label raised
            # `TypeError: cannot use 'tuple' as a dict key` from `_to_internal`, three frames down,
            # which names neither the question nor the label -- and which `serve` cannot classify
            # as a caller error, so over HTTP it became a 500 "inference failed" instead of a 422.
            if isinstance(crit, list):
                keys: Dict[Any, int] = {}
                for i, label in enumerate(crit):
                    try:
                        first = keys[label]
                    except TypeError as exc:
                        raise ValueError(
                            "question %r: choice label %d (%r) cannot be an answer key because it is "
                            "unhashable; a label is rendered as option text and used as the answer "
                            "key" % (qid, i, label)
                        ) from exc
                    except KeyError:
                        keys[label] = i
                    else:
                        raise ValueError(
                            "question %r: choice label %d (%r) repeats label %d; the labels are the "
                            "answer keys, so every option needs its own (1, 1.0 and True are one "
                            "key)" % (qid, i, label, first)
                        )
        elif t == "score":
            if not isinstance(crit, list):
                raise ValueError(
                    "question %r: a score question takes 'criteria' as a list of level "
                    "descriptions, index 0 first" % (qid,)
                )
            if not crit:
                raise ValueError("question %r: a score question needs at least one level" % (qid,))
            if None in crit:
                raise ValueError(
                    "question %r: score level %d is null; give every level a description, "
                    "index 0 first" % (qid, crit.index(None))
                )
        elif crit is not None and not isinstance(crit, dict):
            raise ValueError(
                "question %r: a noul question takes 'criteria' as a dict with optional "
                "'true'/'false' descriptions, or omits it" % (qid,)
            )
        elif isinstance(crit, dict):
            # `render_options` reads these two descriptions out by name -- `crit.get("false")` and
            # `crit.get("true")` -- so a dict keyed any other way is not a noul description at all.
            # It used to be substituted with the default pair without a word, so a caller saw their
            # descriptions accepted and never reach the model (#156). `labels` just below has
            # rejected the same mistake since #163; this is the same rule on the other parameter,
            # and a noul is a boolean question either way, so those are the only two keys it can have.
            keys = {str(k).lower() for k in crit}
            if not keys <= {"true", "false"}:
                raise ValueError(
                    "question %r: a noul question takes 'criteria' keyed only 'true'/'false' (either "
                    "or both, and omitted is fine), got %s. Those keys are the option texts the model "
                    "reads; any other key was silently dropped and replaced with the defaults. If you "
                    "want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set "
                    "'labels' instead." % (qid, sorted(keys))
                )
        if "option_order" in qdef:
            # Slot s shows option `order[s]`. Anything other than a permutation of the option
            # indices would either drop an option or show one twice, so reject it here rather
            # than let it reach the encoder.
            order = qdef["option_order"]
            n = _option_count(qdef)
            if (
                not isinstance(order, (list, tuple))
                or len(order) != n
                or sorted(int(i) for i in order if isinstance(i, int) and not isinstance(i, bool))
                != list(range(n))
            ):
                raise ValueError(
                    "question %r: 'option_order' must be a permutation of range(%d) -- one slot per "
                    "option, each option once -- got %r" % (qid, n, order)
                )
        if "labels" in qdef:
            if t != "noul":
                raise ValueError(
                    "question %r: 'labels' is only supported for noul questions" % (qid,)
                )
            try:
                _resolve_noul_labels(qdef["labels"])
            except ValueError as e:
                raise ValueError("question %r: %s" % (qid, e)) from e

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        elif t == "noul" and isinstance(crit, dict):
            # Normalize boolean literal keys to string keys ("true"/"false")
            crit = {str(k).lower(): v for k, v in crit.items()}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            # `ensure_ascii=False`, matching `serialize_state` and `render_criterion` in
            # common.py and the instructions path in shortlist.py. The default escaped
            # non-ASCII to literal `\uXXXX`, which the tokenizer then read as escape text:
            # on the English checkpoint one German question answered noul=0.1652 as a dict
            # and noul=0.2650 as the identical plain string.
            ins = json.dumps(ins, ensure_ascii=False)
        q = {"t": t, "ins": ins, "crit": crit}
        if "labels" in qdef:
            q["labels"] = qdef["labels"]
        if "option_order" in qdef:
            q["option_order"] = [int(i) for i in qdef["option_order"]]
        return q

    def _encode_state(
        self,
        state: Union[str, dict, list],
        ids: List[str],
        internal: Dict[str, Dict],
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> List[Dict]:
        """Tokenize one state against every (already validated + normalized) question.

        `max_len` / `head_max_len` override the agent config for this call (a start hook may set
        `ctx.max_len` / `ctx.head_max_len`).
        """
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len
        # A chronological conversation list is serialized newest-last, so the default
        # right-truncation (st[:room]) would silently drop the newest turn. Truncate
        # from the left for lists so the most recent intent is preserved.
        truncate_left = isinstance(state, list)
        # Tokenize the shared state once. The ids are identical for every question, so
        # re-serializing and re-tokenizing it inside build_sequence per question was pure
        # duplicated work. Tokenize in full and let build_sequence slice per question, so
        # left-truncation for conversation lists keeps its meaning.
        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        items = []
        for qid in ids:
            q = internal[qid]
            seq, markers, stats, state_stats, *layout = build_sequence(
                self.tok,
                state,
                q,
                max_len,
                head_max_len,
                option_order=q.get("option_order"),
                truncate_left=truncate_left,
                state_ids=state_ids,
                return_stats=True,
                return_truncation_stats=True,
                return_layout=self.parallel_options,
            )
            n_opts = stats["options"]
            if len(markers) != n_opts:
                # The markers are placed at absolute positions and `build_sequence` then drops the
                # ones past `max_len`, so this is about the question fitting in the sequence --
                # `head_max_len` is how much of it the options were given, and `max_len` is the
                # ceiling that dropped them. Naming only `head_max_len` pointed at the wrong knob
                # in both directions: lowering it shortens the option block and can make the
                # call succeed, while raising it makes the overflow worse.
                #
                # The count reported is the markers that SURVIVED, not `len(seq)`: `build_sequence`
                # truncates to `max_len` first, so `len(seq)` is always exactly `max_len` here and
                # would state the ceiling as though it were the requirement.
                raise ValueError(
                    "question %r: only %d of its %d option markers fit in max_len=%d with "
                    "head_max_len=%d spent on the question; lower head_max_len, raise max_len, "
                    "or use fewer options" % (qid, len(markers), n_opts, max_len, head_max_len)
                )
            item = {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["t"]],
                "options": stats,
                "state_stats": state_stats,
            }
            if layout:
                item["layout"] = layout[0]
            items.append(item)
        return items

    def _decode_answers(
        self,
        logits,
        act,
        items: List[Dict],
        ids: List[str],
        internal: Dict[str, Dict],
        offset: int,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Turn one state's logit rows (starting at `offset`) into typed answers."""
        answers = {}
        raw_rows = _option_logits(logits, items, offset)
        for j, qid in enumerate(ids):
            r = offset + j
            q = internal[qid]
            k = len(items[j]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            if lang and lang.split("-")[0].lower() in self.lang_temperatures:
                l_cfg = self.lang_temperatures[lang.split("-")[0].lower()]
                t_scale = l_cfg["temperature_by_options"].get(
                    temp_bucket(qt, k), l_cfg["temperature"][qt]
                )
            z = raw_rows[j] / t_scale
            p = np.exp(z - z.max())
            p /= p.sum()

            # The row comes back in slot order; everything below indexes by option.
            p = unpermute_probs(p, q.get("option_order"))

            # `confidence` means one thing for `noul` (max(p)) and another for `choice` and
            # `score` (normalized entropy), and only the first is the quantity temperature
            # scaling fits and ECE measures. Rather than change one underneath existing
            # callers, report both: `answer_confidence` is the calibrated one, on every
            # question type, so a caller can gate across types on a single number.
            ans_raw = answer_confidence(p, k)
            lang_override = bool(lang and lang.split("-")[0].lower() in self.lang_temperatures)
            if getattr(self, "binning_map", None) and not lang_override:
                ans_conf = round(
                    apply_binning_map(ans_raw, temp_bucket(qt, k), self.binning_map), 4
                )
            else:
                ans_conf = round(ans_raw, 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    # A legend maps an index to the text of a level, and the keys are already
                    # strings. `render_criterion` rather than `str`: a dict or list level then
                    # comes back as the same JSON text the model was shown, where `str` produced
                    # a Python repr. A numeric scale passed as `[1, 2, 3]` used to come back as
                    # `{"0": 1, "1": 2, "2": 3}`, so the response's JSON types depended on what
                    # the caller happened to pass; `structured` stringifies every level it builds
                    # and `probabilities` stringifies its keys right below, so this was the one
                    # path that did not.
                    "legend": {str(i): render_criterion(c) for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    # identical here: over two options max(p_true, 1 - p_true) is max(p)
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
        return answers

    @_reuse_question_tokens
    def predict_batch(
        self,
        states: List[Union[str, dict, list]],
        questions: Dict[str, Dict[str, Any]],
        batch_size: Optional[int] = None,
        lang: Optional[str] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        sort_by_length: bool = False,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate the same questions over many states, packing them into shared forward passes.

        This is the throughput path. `system_one`/`predict` handle one state per forward pass; on a
        GPU that leaves most of the batch dimension idle. `predict_batch` collates several states'
        question rows into one tensor, so a call that would take N sequential forward passes takes
        one (or `ceil(len(states) / batch_size)`), which is several times faster per decision on GPU.

        Args:
            states: A list of states (each a text string, JSON dict, or conversation turn list).
                    The same `questions` are evaluated against every state.
            questions: Question definitions, exactly as accepted by `system_one`.
            batch_size: Optional cap on states per forward pass. `None` sends them all in one pass;
                        set it to bound peak memory when batching many or long states. A `float`,
                        a `str` or a `bool` is refused by name -- the count is spent as a `range()`
                        step, and `0` or a negative integer means the same "no cap" as `None`.
            hooks (HookArg): Per-call hooks, appended after any installed on the Agent.
                    See `laya.hooks`.
            on_predict_start (PredictHookArg): A per-call start hook. It may rewrite the
                    state/questions or call `ctx.skip(...)` to short-circuit inference.
            on_predict_end (PredictHookArg): A per-call end hook. It may rewrite the results.
            hooks_raise: Override the Agent's `hooks_raise` for this call.
            hooks_timeout: Override the Agent's `hooks_timeout` for this call.
            max_len: Override the agent config's `max_len` for this call. A start hook may also
                    set `ctx.max_len` to shape the token budget.
            head_max_len: Override the agent config's `head_max_len` for this call. A start hook
                    may also set `ctx.head_max_len`.
            sort_by_length: Group similarly sized encoded states within windows of eight batches
                    to reduce padding. Requires an explicit `batch_size` greater than one and
                    smaller than the number of states; otherwise it has no effect. Results retain
                    input order. This buffers up to eight batches of tokenized states instead of
                    one. Changing batch shapes can slightly change floating-point predictions.

        Returns:
            A list of per-state result dicts, each identical in shape to `system_one`'s output and
            aligned with `states` by index.
        """
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        batch_size = check_batch_size(batch_size)
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(
            states=states,
            questions=questions,
            model=self.model_id,
            agent=self,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        try:
            dispatch(
                active,
                "on_predict_start",
                ctx,
                raise_errors=raise_errors,
                lock=self._hooks_lock,
                timeout=timeout,
            )
            states, questions = ctx.states, ctx.questions
            if ctx.results is None:
                # A start hook may have normalised a bare string/dict into a list; only the value
                # that survives the hook is validated.
                if isinstance(states, (str, bytes, dict)):
                    raise TypeError(
                        "predict_batch expects a list of states; pass a single state to predict()/system_one()."
                    )
                if not isinstance(questions, dict):
                    raise TypeError(
                        "questions must be a dict of question id -> definition, got %s"
                        % type(questions).__name__
                    )
                states = list(states)
                if any(state is None for state in states):
                    raise TypeError("state must not be None; pass a string, dict, or list")
                if len(states) == 1:
                    _disable_question_token_reuse()
                if not states:
                    ctx.results = []
                else:
                    ids = list(questions.keys())
                    # Empty questions: empty answers, zero usage, no tokenization or forward.
                    if not ids:
                        ctx.results = [
                            {
                                "model": "laya-rl-agent",
                                "answers": {},
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            }
                            for _ in states
                        ]
                    else:
                        # Validate + normalize each question once (state-independent).
                        for qid in ids:
                            self._check_question(qid, questions[qid])
                        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
                        chunk = batch_size if (batch_size and batch_size > 0) else len(states)

                        # Per-call token-budget overrides (a start hook may have set them).
                        overrides: Dict[str, int] = {}
                        if ctx.max_len is not None:
                            overrides["max_len"] = ctx.max_len
                        if ctx.head_max_len is not None:
                            overrides["head_max_len"] = ctx.head_max_len

                        results: List[Dict[str, Any]] = []
                        reorder = sort_by_length and 1 < chunk < len(states)
                        # Bound tokenized lookahead independently of the full input size. Use
                        # actual post-truncation lengths, with each state's questions kept together.
                        window = chunk * 8 if reorder else chunk
                        for start in range(0, len(states), window):
                            part = states[start : start + window]
                            encoded = [
                                self._encode_state(st, ids, internal, **overrides) for st in part
                            ]
                            order = list(range(len(encoded)))
                            if reorder:
                                order.sort(
                                    key=lambda i: max(len(item["ids"]) for item in encoded[i])
                                )
                            window_results = [None] * len(encoded)
                            for offset in range(0, len(order), chunk):
                                indices = order[offset : offset + chunk]
                                per_state_items = [encoded[i] for i in indices]

                                b = collate_items(per_state_items, self.tok.pad_token_id)
                                logits, act = self._forward(b)
                                att = b["attention_mask"]

                                row = 0
                                for index, items in zip(indices, per_state_items):
                                    nrows = len(items)
                                    n_tokens = int(att[row : row + nrows].sum())
                                    answers = self._decode_answers(
                                        logits,
                                        act,
                                        items,
                                        ids,
                                        internal,
                                        row,
                                        **({"lang": lang} if lang else {}),
                                    )
                                    # Truncation is a token budget that moves with max_len, head_max_len
                                    # and each question's head, so only build_sequence knows it (#174).
                                    stats = [item["state_stats"] for item in items]
                                    dropped = max(s["state_tokens_dropped"] for s in stats)
                                    usage = {
                                        "input_tokens": n_tokens,
                                        "output_tokens": 0,
                                        "state_tokens": stats[0]["state_tokens"],
                                        # worst case: the questions share one state, not one head budget
                                        "state_tokens_dropped": dropped,
                                        "truncated": dropped > 0,
                                        "truncated_questions": [
                                            qid for qid, s in zip(ids, stats) if s["truncated"]
                                        ],
                                    }
                                    # Only when a question actually lost options to the head
                                    # budget: an answer chosen from 42 distinguishable spans of
                                    # 58 has a ceiling the caller cannot otherwise see, and a
                                    # key that is always present would be noise on the
                                    # overwhelming majority of requests that never collapse.
                                    collapsed = collapsed_options(ids, items)
                                    if collapsed:
                                        usage["options"] = collapsed
                                    window_results[index] = {
                                        "model": "laya-rl-agent",
                                        "answers": answers,
                                        "usage": usage,
                                    }
                                    row += nrows
                            results.extend(window_results)
                        ctx.results = results
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(
                    active,
                    "on_error",
                    ctx,
                    raise_errors=raise_errors,
                    lock=self._hooks_lock,
                    timeout=timeout,
                )
            except BaseException as hook_exc:
                # A failing on_error hook must not hide the failure that triggered it.
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
                apply_confidence_gate(ctx.results, mc)
            try:
                record_scan = getattr(ctx, "_record_scan_results", None)
                if ctx.error is None and ctx.results is not None and record_scan is not None:
                    record_scan(ctx)
                dispatch(
                    active,
                    "on_predict_end",
                    ctx,
                    raise_errors=raise_errors,
                    lock=self._hooks_lock,
                    timeout=timeout,
                )
            except BaseException as hook_exc:
                # End hooks run on the failure path too; do not let one mask the real error.
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results

    def predict_long(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Dict[str, Any]],
        window: Optional[int] = None,
        stride: Optional[int] = None,
        aggregate: str = "auto",
        batch_size: Optional[int] = None,
        lang: Optional[str] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate questions over a state longer than the context window, scanning it in
        overlapping windows and aggregating per question.

        `system_one`/`predict` truncate a state that exceeds `max_len` to a single window (the
        first, or for a conversation list the last), silently dropping the rest. `predict_long`
        tokenizes the state once, splits it into overlapping token windows, scores every window in
        shared forward passes (via `predict_batch`), and combines the per-window answers:

          * noul  -> P(true) is the max over windows (the statement holds if any window supports it)
          * choice-> the answer from the single most-confident window, so a localized signal isn't
                     out-voted by the many neutral windows a long document is mostly made of
                     (averaging drowns it -- the neutral majority dominates)
          * score -> the level from the most-confident window, likewise

        The returned probability/confidence is the deciding window's, **not a calibrated number for
        the whole document**: a `noul` max over many windows drifts up with the window count even
        with no signal, and `choice` can land on a confidently-neutral window when nothing in the
        document is decisive. Each answer therefore carries `answer["window"]` — the deciding
        window's `index`, `token_start`/`token_end` into the tokenized state, and the window `count`
        — so a caller can inspect the span the answer came from rather than trust the raw number.
        That span is the one the model read, not merely the one asked for: the window is capped at
        the room the questions leave, so what is handed to `predict_batch` is not cut short again.

        A state the questions leave room for is passed straight to `system_one` (identical output),
        since `system_one` reads it whole; with an explicit `window`, a state that fits that window.

        The hooks wrap the inference that answers the state, which for a document needing several
        windows is the one shared `predict_batch` over them: `on_predict_start` fires once, and
        `ctx.states` holds the decoded window texts in scan order -- not the caller's `state`, which
        was tokenized to produce them. Three outcomes follow from what the chain leaves behind:

          * `ctx.skip([result])` answers the document: the payload comes back with no window
            attribution and `usage["windows"]` at 0, because nothing was scored
          * a scan left as this method built it: every window is scored, each answer carries
            `answer["window"]`, and `usage["windows"]` is the window count
          * a rewritten scan (`ctx.states` replaced, in any way): the answers are aggregated over
            the states that were scored, but no answer carries `answer["window"]` -- the offsets
            above describe this method's windows, not the text the model read

        End hooks may reorder results without changing an answer's original window. Attribution
        follows the original answer object, provided its prediction fields are unchanged. Cloned,
        replaced or edited answers have no window attribution; added reliability metadata is kept.

        Args:
            window: state tokens per window. Defaults to the checkpoint's state budget,
                    `max(64, max_len - head_max_len - 8)` -- the 64 is a floor, so widening
                    `head_max_len` stops shrinking the default once the budget reaches it -- and
                    either way is capped at the room the questions leave for the state inside
                    `max_len` -- the smallest room of them,
                    because one list of windows is scored for every question. A wider window is
                    re-truncated on the way to the model, so it is clamped instead, with a
                    `RuntimeWarning` when the caller is the one who asked for it. Options are what
                    make the room small: on the English checkpoint a 2-option question leaves 483
                    tokens for the state and a 100-option one leaves 100.
                    A smaller window isolates a localized signal better (a short deciding span is a
                    larger fraction of its window, so that window classifies it clearly), at the
                    cost of more windows; the large default favors context and throughput. `noul`
                    is robust to this, `choice`/`score` benefit from a smaller window when the
                    deciding span is a small part of a long, otherwise-neutral document.
            stride: token step between windows. Defaults to half the *effective* window (50%
                    overlap), so a span near a boundary still lands whole inside some window. A
                    stride past the effective window is refused rather than clamped: the tokens
                    between each pair of windows would be read by no window at all, which is the
                    failure this method exists to prevent.
            aggregate: "auto" (the per-type rules above) is the only mode for now.
            batch_size: cap on windows per forward pass, to bound memory on very long states.
            lang: per-language temperature selection, as in `system_one`.
            hooks (HookArg): Per-call hooks, appended after any installed on the Agent.
                    See `laya.hooks`.
            on_predict_start (PredictHookArg): A per-call start hook, as in `system_one`.
            on_predict_end (PredictHookArg): A per-call end hook, as in `system_one`.
            hooks_raise: Override the Agent's `hooks_raise` for this call.
            hooks_timeout: Override the Agent's `hooks_timeout` for this call.

        Raises:
            ValueError: `aggregate` is anything but "auto"; the questions' options fill the whole
                    sequence, leaving no room for the state; or `stride` steps past the effective
                    window, so tokens between two windows would be read by nothing.

        Returns a single result dict, the same shape as `system_one`, with `usage["windows"]` added.
        The key is always present and counts the windows the model scored to produce the answer: `1`
        for a state that fit one window, `N` for a document scanned in `N` overlapping windows (or
        the `N` a start hook rewrote them to), and `0` when a start hook answered the document, or
        left no states to score, before any window was read -- on either path, so a cached answer
        never reads as a window the model read.

        Across several windows the truncation keys are combined like every other `usage` field:
        `truncated`, `state_tokens` and `state_tokens_dropped` are summed (so `truncated` is the
        number of windows that were cut, and the token counts include the overlap), and
        `truncated_questions` is the last window's list. The two can disagree: when only an
        earlier window was cut, `truncated` is above 0 and `truncated_questions` is empty. A
        window is cut when it is larger than the room a question's head leaves. Test
        `usage["truncated"] > 0` here, not `is True`.
        """
        if state is None:
            raise TypeError("state must not be None; pass a string, dict, or list")
        if not isinstance(questions, dict):
            raise TypeError(
                "questions must be a dict of question id -> definition, got %s"
                % type(questions).__name__
            )
        if aggregate != "auto":
            raise ValueError("predict_long: only aggregate='auto' is supported")
        hook_kwargs = {
            "hooks": hooks,
            "on_predict_start": on_predict_start,
            "on_predict_end": on_predict_end,
            "hooks_raise": hooks_raise,
            "hooks_timeout": hooks_timeout,
        }
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        # Size the window against the room these questions actually leave, not the config alone:
        # every window is decoded and scored as a normal state, so a window wider than the room
        # was re-truncated by `build_sequence` on the way in and the tail of it reached no model
        # -- while `answer["window"]` reported the whole span. Measured on the English checkpoint
        # (max_len=512, head_max_len=192, config budget 312): a 2-option question leaves room for
        # 483 tokens, 48 options leave 308, and 100 -- `serve`'s documented maximum -- leave 100.
        # At 87 options the room (152) fell below the 156-token default stride, so the windows
        # stopped overlapping and part of the document was read by no window at all: measured on a
        # 920-token document at 100 options, 420 of its tokens reached no window.
        ids = list(questions.keys())
        for qid in ids:
            self._check_question(qid, questions[qid])
        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
        budget, step, room = window_budget(
            self.tok,
            [internal[qid] for qid in ids],
            max_len,
            head_max_len,
            window=window,
            stride=stride,
        )
        # Snapshot the questions the scan was just sized against, BEFORE any start hook can
        # rewrite them, and compare against this instead of `questions` itself. `==` over the
        # mapping cannot see an in-place rewrite: a hook that adds options to
        # `ctx.questions[q]["criteria"]` -- the `widen_for_high_cardinality` pattern in
        # `docs/hooks/patterns.md`, and the case this guard exists for -- mutates the same nested
        # dict the caller's mapping holds, so both sides change together and compare equal. A deep
        # copy is independent, so `_check_scan_budget` sees the difference; a shallow one would
        # share the nested dicts and miss it exactly as before.
        asked = copy.deepcopy(questions)

        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        # Fits in one window: identical to a plain call, no windowing overhead. `windows` is still
        # written, so the key is total over the three paths this method can take and a caller can
        # ask "how much of the document did the model read?" without handling a KeyError on the
        # shortest, most common inputs. The result is copied first: a start hook that answers with
        # `ctx.skip(...)` hands back its own payload dict, and it may be a cached object.
        # "Fits" is the room the questions leave, not the default window: `system_one` reads a
        # state up to that room whole, so windowing one between the two only re-reads it in pieces
        # and lets the per-window max inflate the answer. An explicit `window` still scans.
        if len(state_ids) <= (budget if window and window > 0 else room):
            probe, evidence = _start_evidence()
            single = dict(
                self.system_one(
                    state, questions, lang=lang, **_with_start_probe(hook_kwargs, probe)
                )
            )
            # Why 0 for a hook answer here: the state did fit one window, but no window was scored,
            # which is the same fact the multi-window path reports as 0. Reading 1 would make a
            # cached answer and a served answer agree on how much of the input the model saw.
            # The same budget check as the multi-window path. Without it a document short enough to
            # fit one window was silently truncated by a re-budgeting hook and still reported
            # `windows: 1`, i.e. "the model read all of it" -- measured, 138 of 240 state tokens
            # never reached the model, while a longer document on the identical input hard-failed.
            # Sized for the whole state when it is longer than the window, so a hook that narrows
            # the room below it is refused rather than cutting its tail.
            _check_scan_budget(
                self, evidence, max(budget, len(state_ids)), max_len, head_max_len, asked
            )
            single["usage"] = {
                **(single.get("usage") or {}),
                "windows": 0 if evidence["answered"] else 1,
            }
            return single

        windows, starts = [], []
        i, n = 0, len(state_ids)
        while i < n:
            # Decode each token window back to text so predict_batch re-tokenizes it as a normal
            # state. For BPE tokenizers the re-tokenized boundaries can shift by a token or two vs
            # this split; harmless for aggregation since the 50% default overlap absorbs it.
            windows.append(self.tok.decode(state_ids[i : i + budget]))
            starts.append(i)
            if i + budget >= n:
                break
            i += step

        probe, evidence = _start_evidence(track_answers=True)
        # `list(windows)`, not `windows`: `ctx.states` is the list the hook receives, so a hook that
        # mutates it in place (`append`, `sort`) would otherwise also grow the split this call
        # attributes answers to, and the counts would agree while `starts` no longer lined up.
        # Bound one forward pass to what the un-capped scan would have used; see
        # `window_batch_cap`. An explicit batch_size is honoured untouched.
        cap = window_batch_cap(
            len(windows), budget, max(64, max_len - head_max_len - 8), batch_size
        )
        results = self.predict_batch(
            list(windows),
            questions,
            batch_size=cap,
            lang=lang,
            **_with_start_probe(hook_kwargs, probe),
        )
        _check_scan_budget(self, evidence, budget, max_len, head_max_len, asked)

        if evidence["answered"]:
            # The hook replaced the call before any window was scored. Aggregating over its payload
            # would pick between answers that were never scored and name a deciding window that
            # decided nothing, so pass the document answer through unattributed and report the
            # count as what it is: none of them.
            if len(results) != 1:
                raise ValueError(
                    "predict_long: a start hook answered this state with %d results; ctx.skip()"
                    " takes one result for the document, not one per window" % len(results)
                )
            warnings.warn(
                "laya: predict_long: a hook answered the state before it was scanned, so "
                "no window decided the result and none is reported",
                RuntimeWarning,
                stacklevel=2,
            )
            document = dict(results[0])
            document["usage"] = dict(document.get("usage") or {})
            document["usage"]["windows"] = 0
            return document

        # The contract on `ctx.states` is that a start hook may replace it (see `docs/hooks/api.md`),
        # so a scan that comes back different from the split above is a supported outcome, not a
        # failure to report: the states that were scored are the hook's, while `starts` describes
        # this method's windows. Aggregate what came back and name nothing.
        rewritten = evidence["states"] is not None and evidence["states"] != windows

        if len(results) != len(windows) and not rewritten:
            # The observer saw the scan leave the hook chain and it is the one computed above, so
            # nothing here explains a count that is not the split. That is `predict_batch`
            # disagreeing with its own input -- a bug, or a replacement that never dispatched hooks.
            seen = (
                "the scan that reached inference was the split made here"
                if evidence["states"] is not None
                else "no start hook chain ran, so nothing rewrote the scan"
            )
            raise ValueError(
                "predict_long: the state was split into %d windows and the call returned %d"
                " results, and %s" % (len(windows), len(results), seen)
            )

        if not results:
            # A hook that left no states scored nothing, which is what 0 already means here; the
            # questions go unanswered rather than being aggregated over an empty list.
            warnings.warn(
                "laya: predict_long: a start hook left no states to score, so the call"
                " aggregated nothing and returns no answers",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "model": "laya-rl-agent",
                "answers": {},
                "usage": {**aggregate_usage(results), "windows": 0},
            }

        question_types = evidence["question_types"]
        if question_types is None:  # a replacement predict_batch may not dispatch hooks
            question_types = {qid: self._to_internal(qdef)["t"] for qid, qdef in questions.items()}
        answers = {}
        for qid, qtype in question_types.items():
            missing = sum(qid not in r["answers"] for r in results)
            if missing:
                # End hooks may remove start-added questions, but must not hide a missing
                # caller answer or an inconsistent set of answers across windows.
                if (
                    missing == len(results)
                    and qid not in asked
                    and evidence["question_types"] is not None
                ):
                    continue
                raise ValueError(
                    "predict_long: answer %r is missing from %d of %d windows after prediction hooks"
                    % (qid, missing, len(results))
                )
            per = [r["answers"][qid] for r in results]
            if qtype == "noul":
                # Evidence anywhere: the strongest window decides. Its own P(true) and confidence
                # (and act) are carried through, so the fields stay mutually consistent.
                best = max(range(len(per)), key=lambda j: float(per[j]["noul"]))
            else:
                # choice / score: the most-confident window wins. Averaging over a long, mostly
                # neutral document lets the neutral majority out-vote the one window that saw the
                # deciding span; the single most-confident window preserves a localized signal.
                best = max(range(len(per)), key=lambda j: float(per[j]["answer_confidence"]))
            original = per[best]
            index = None if rewritten else _scan_answer_index(evidence, qid, original, best)
            ans = dict(original)
            ans.pop("window", None)
            # Name the window that decided, so a caller can check the deciding span itself. The
            # probability here is that window's, NOT a document-level calibrated number.
            if index is not None:
                ans["window"] = {
                    "index": index,
                    "token_start": starts[index],
                    "token_end": min(starts[index] + budget, len(state_ids)),
                    "count": len(results),
                }
            answers[qid] = ans
        # Aggregate usage generically so fields predict_batch may grow later (e.g. the
        # fallback counters from #351) are propagated, not silently dropped: sum numeric
        # fields across windows, merge the per-question records, then record the window
        # count.
        # A per-question field has to be merged rather than replaced. `usage["options"]` is a
        # dict keyed by question id, set only on the windows where option spans actually
        # collapsed, so replacing it left the caller holding whichever collapsing window came
        # last. The deciding window is the most confident one, not the last one, so that could
        # report a collapse for a window that did not decide while the deciding window's own
        # record was gone.
        usage: Dict[str, Any] = {}
        for r in results:
            for key, val in r["usage"].items():
                prev = usage.get(key)
                if isinstance(val, (int, float)):
                    usage[key] = (prev if isinstance(prev, (int, float)) else 0) + val
                elif isinstance(val, dict) and isinstance(prev, dict):
                    usage[key] = {**prev, **val}
                else:
                    usage[key] = val
        usage["output_tokens"] = 0
        usage["windows"] = len(results)
        result = {"model": "laya-rl-agent", "answers": answers, "usage": usage}
        # `predict_long` accepts no `min_confidence` -- the window loop has no single confidence to
        # gate on. The gate is still called, per its contract in laya/confidence.py: with
        # `min_confidence` None it writes nothing, so the payload comes back exactly as the
        # windows built it -- no `abstention`, no `abstention_threshold`, no flag.
        apply_confidence_gate([result], None)
        return result

    def system_one(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Dict[str, Any]],
        lang: Optional[str] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single, parallel forward pass.

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Dictionary mapping question_id -> question definition.
                - choice: {"type": "choice", "instructions": "...", "criteria": {"optA": "...", ...}}
                - score:  {"type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...]}
                - noul:   {"type": "noul", "instructions": "...",
                           "criteria": {"false": "...", "true": "..."},
                           "labels": {"false": "B", "true": "A"}}

                  Noul criteria and labels are optional. Labels only control the text shown to the
                  model; their keys retain false/true semantics, and the returned `noul` value is
                  always P(true). Labels default to false/true for compatibility.

        Returns:
            Dictionary with answers, probabilities, calibrated confidence, and token usage.
            Empty questions return empty answers and zero token usage without tokenization
            or a model forward pass.

            When the head budget leaves two options with the same token span, `usage` carries
            an `options` entry for each question it happened to -- `total`, `distinct` and
            `tokens_per_option` -- because an answer chosen among 42 distinguishable spans of
            58 has a ceiling that is the budget's and not the model's. Questions whose options
            all survive are absent, so a request that collapses nothing is unchanged.

            `usage` also reports whether the state fit: `truncated`, `state_tokens`,
            `state_tokens_dropped`, and `truncated_questions` (the questions whose head left
            too little room). A caller that cares whether the answer saw the whole state should
            read `usage["truncated"]` rather than estimate from the length of what it sent.

        To score many states at once, see `predict_batch`, which shares forward passes across them.
        """
        return self.predict_batch(
            [state],
            questions,
            lang=lang,
            hooks=hooks,
            on_predict_start=on_predict_start,
            on_predict_end=on_predict_end,
            hooks_raise=hooks_raise,
            hooks_timeout=hooks_timeout,
            max_len=max_len,
            head_max_len=head_max_len,
            min_confidence=min_confidence,
        )[0]

    def __enter__(self):
        return self

    def decide(
        self,
        state: Union[str, dict, list],
        schema: Any = None,
        *,
        questions: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
        min_confidence: Optional[float] = None,
        **predict_kwargs,
    ) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        are forwarded to `predict` / `system_one`.
        """
        from .structured import decide as _decide

        return _decide(
            self,
            state,
            schema,
            questions=questions,
            return_details=return_details,
            min_confidence=min_confidence,
            **predict_kwargs,
        )

    def decide_batch(
        self,
        states: List[Union[str, dict, list]],
        schema: Any = None,
        *,
        questions: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
        min_confidence: Optional[float] = None,
        **predict_kwargs,
    ) -> List[Any]:
        """Answer many states against one schema (JSON schema or pydantic model) in one batched call.

        The throughput form of :meth:`decide`: the schema is planned once and its questions
        run over every state through :meth:`predict_batch` (shared forward passes, results
        in input order), then each state's answers are projected as ``decide`` does. Extra
        keyword arguments (``batch_size=``, ``lang=``, ``hooks=``, ...) are forwarded to
        ``predict_batch``. See `laya.structured`.
        """
        from .structured import decide_batch as _decide_batch

        return _decide_batch(
            self,
            states,
            schema,
            questions=questions,
            return_details=return_details,
            min_confidence=min_confidence,
            **predict_kwargs,
        )

    def fit_temperatures(self, records, compute_ece: bool = False, seed: int = 0) -> Dict[str, Any]:
        """Fit per-bucket temperatures from CPU records and store them on this agent.

        `records` are `(qtype, logits, target, k)`. Build them with
        `laya.calibrate.records_from_labeled` when you have labeled forwards; this method
        does not download weights or write `model.safetensors`. `seed` only affects the
        held-out ECE split when `compute_ece` is true. The checkpoint `cfg` is left as loaded.
        """
        result = fit_temperature_map(records, compute_ece=compute_ece, seed=seed)
        # Already clamped inside the fitter; don't report that as a bad calibration file.
        _install_temperatures(
            self, result["temperature"], result["temperature_by_options"], warn=False
        )
        return result

    def fit_binning(self, records, min_bucket_n: int = MIN_BINNING_BUCKET_N) -> Dict[str, Any]:
        """Fit a histogram-binning map on top of this agent's fitted temperatures and store it.

        `records` are the same `(qtype, logits, target[, k])` tuples as `fit_temperatures`
        consumed. The map is keyed exactly like `temperature_by_options`, composes on top of
        the current temperatures, and is written out by `save_calibration` as `binning_map`.
        """
        self.binning_map = fit_binning_map(
            records, self.temperature, self.temperature_by_options, min_bucket_n=min_bucket_n
        )
        return self.binning_map

    def save_calibration(self, path: str) -> None:
        """Write `temperature`, `temperature_by_options`, `binning_map` when the agent has one, and
        the checkpoint they were fitted for. Does not write weights."""
        payload = calibration_payload(
            self.temperature,
            self.temperature_by_options,
            model_id_or_path=getattr(self, "model_id_or_path", None),
            subfolder=getattr(self, "subfolder", None),
            config=getattr(self, "cfg", None),
            binning_map=getattr(self, "binning_map", None),
        )
        destination = os.path.realpath(path)
        fd, temporary = tempfile.mkstemp(
            dir=os.path.dirname(destination), prefix=".calibration.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                mode = os.stat(destination).st_mode & 0o777
            except FileNotFoundError:
                pass
            else:
                os.chmod(temporary, mode)
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def load_calibration(self, path: str) -> None:
        """Read a JSON map written by `save_calibration` onto this agent.

        The file carries three fields and all three are installed: `temperature`,
        `temperature_by_options` and `binning_map`. A file with no `binning_map` key installs
        `None`, so loading one clears a map this agent's `fit_binning` fitted -- the file is the
        whole calibration state, not a patch onto the current one.

        A file with no `version` is treated as version 1 and still loads. A newer file
        whose recorded checkpoint does not match this agent warns and still loads.
        Temperatures that are not numbers, or that sit outside `[TEMP_MIN, TEMP_MAX]`, are
        clamped with `clamp_temperature` the same way checkpoint load is; binning values are
        not clamped, they are refused with a `ValueError` naming the field, because an
        out-of-range binning value would move a confidence with nothing to fall back to.
        """
        with open(path) as f:
            payload = json.load(f)
        apply_calibration_payload(self, payload)

    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None
    model_id = None
    lang_temperatures = {}
    binning_map = None

    @property
    def parallel_options(self):
        from .common import option_layout

        return option_layout(self.cfg) == "parallel"

    def _init_host(
        self,
        *,
        lang_temperatures=None,
        calibration=None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise=True,
        hooks_concurrent=True,
        hooks_timeout=None,
    ):
        self.cfg = dict(self.cfg)
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        self.lang_temperatures = resolve_lang_temperatures(lang_temperatures, self.temperature_raw)
        self.binning_map = None
        if calibration:
            self.load_calibration(calibration)

    def prepare(self, state, questions):
        if state is None:
            raise TypeError("state must not be None; pass a string, dict, or list")
        if not isinstance(questions, dict):
            raise TypeError(
                "questions must be a dict of question id -> definition, got %s"
                % type(questions).__name__
            )
        ids = list(questions)
        for qid in ids:
            self._check_question(qid, questions[qid])
        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
        return (self._encode_state(state, ids, internal) if ids else []), list(internal.values())

    @property
    def backend(self) -> str:
        """Native runtime identity, rather than a PyTorch backend name."""
        return "mlx"

    @property
    def backend_object(self):
        """No official PyTorch Backend object is installed on a native agent."""
        return None

    def set_backend(self, name: str = "auto", strict: bool = False, **options) -> str:
        raise NotImplementedError(
            "Runtime backend switching requires official Laya. Construct a new native agent "
            "to change its execution settings."
        )

    def accelerate(self, use_graphs: bool = True, strict: bool = False):
        """The upstream TileLang fast path is unavailable on this native runtime."""
        if strict:
            raise RuntimeError("laya fast path needs a CUDA device")
        return False

    def deaccelerate(self):
        """No TileLang backend is installed; native execution remains active."""
        return None

    def dtype_for(self, rows: int):
        """Native dtype / export precision; no row-gated PyTorch autocast occurs."""
        return self.dtype

    WARMUP_SHAPES = ((1, 64, 3), (4, 128, 5))

    def warmup(self, shapes=None) -> float:
        """Warm native forwards on upstream's synthetic shapes, without hooks.

        Return elapsed seconds. Tensor transport is native; the shape defaults,
        token cap and fill values follow official Agent.warmup.
        Explicit calls propagate graph or compilation errors.
        """
        max_len = int(self.cfg.get("max_len", 512))
        fill = self.tok.cls_token_id or 0
        started = time.perf_counter()
        for rows, tokens, markers in self.WARMUP_SHAPES if shapes is None else shapes:
            tokens = max(markers + 2, min(int(tokens), max_len))
            self._forward(
                {
                    "input_ids": np.full((rows, tokens), fill, dtype=np.int64),
                    "attention_mask": np.ones((rows, tokens), dtype=np.int64),
                    "marker_pos": np.tile(np.arange(1, markers + 1), (rows, 1)),
                    "marker_mask": np.ones((rows, markers), dtype=bool),
                    "qtype": np.zeros(rows, dtype=np.int64),
                }
            )
        return time.perf_counter() - started

    predict = system_one
