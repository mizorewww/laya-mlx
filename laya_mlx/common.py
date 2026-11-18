"""Official Laya v0.4.35 host semantics; native MLX collation is in agent.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

import json
import math
import threading
import warnings
from contextvars import ContextVar
from functools import wraps
from numbers import Integral
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}
_TOKENIZE_LOCK = threading.RLock()
_QUESTION_TOKEN_CACHE = ContextVar("laya_question_token_cache", default=None)
_WINDOW_CLAMP_WARN_RATIO = 2
_WINDOW_BATCH_BLOWUP = 2
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def encode_text(tok, text, **kwargs):
    """Tokenize `text` while holding the lock a shared fast tokenizer needs."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


def _reuse_question_tokens(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # One cache per prediction call: nested calls get their own scope, and exceptions
        # restore the outer scope. Nothing is retained on an Agent or shared across threads.
        scope = {"thread": threading.get_ident(), "tokens": {}}
        token = _QUESTION_TOKEN_CACHE.set(scope)
        try:
            return fn(*args, **kwargs)
        finally:
            # A timed hook can copy this context to a worker that outlives the call.
            scope["tokens"] = None
            _QUESTION_TOKEN_CACHE.reset(token)

    return wrapped


def _disable_question_token_reuse():
    scope = _QUESTION_TOKEN_CACHE.get()
    if scope is not None:
        scope["tokens"] = None


def _encode_question_text(tok, text, **kwargs):
    scope = _QUESTION_TOKEN_CACHE.get()
    cache = (
        scope["tokens"] if scope is not None and scope["thread"] == threading.get_ident() else None
    )
    if cache is None:
        return encode_text(tok, text, **kwargs)["input_ids"]
    # Key the rendered, mask-sanitized text and encoding settings, not a JSON question:
    # option order, structured criteria and custom noul labels must keep their meaning.
    key = (id(tok), text, tuple(kwargs.items()))
    if key not in cache:
        # Keep the tokenizer alive so its identity cannot be reused within this scope.
        cache[key] = (tok, tuple(encode_text(tok, text, **kwargs)["input_ids"]))
    # Sequence assembly must not mutate token lists retained for later states.
    return list(cache[key][1])


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


class _TokenWindow(str):
    """Decoded text for hooks, with the original tokens for an unchanged scan."""

    def __new__(cls, tokenizer, token_ids):
        obj = super().__new__(cls, tokenizer.decode(token_ids))
        obj.tokenizer = tokenizer
        obj.token_ids = tuple(token_ids)
        return obj

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return str, (str(self),)


def _state_token_ids(tok, state):
    if isinstance(state, _TokenWindow) and state.tokenizer is tok:
        return list(state.token_ids)
    return encode_text(
        tok, serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
    )["input_ids"]


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes a single-line JSON
    document with the default separators -- ``", "`` between members, ``": "`` before a value --
    so a rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [
            str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
            for k, v in crit.items()
        ]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
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


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    state_ids: Optional[List[int]] = None,
    return_stats: bool = False,
    return_truncation_stats: bool = False,
    return_layout: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
    instead of re-serializing and re-tokenizing the same document per question.

    `return_stats` adds a third return value describing what the head budget did to the options:
    `options` (how many the question defines), `options_distinct` (how many still have a token
    span of their own) and `tokens_per_option` (the cap applied to each, or None when none was).

    The state is clamped to whatever room is left after the head, so a long state loses tokens
    here silently. `return_truncation_stats=True` adds one more return value, after the option
    stats when both are asked for, reporting that clamp:

        {"state_tokens": int, "state_tokens_used": int, "state_tokens_dropped": int,
         "truncated": bool}

    Callers cannot reconstruct this from the outside. The budget is in tokens, not characters,
    and the room left for the state depends on `max_len`, `head_max_len`, the instruction and
    the rendered options - so it moves per checkpoint and per question. A caller guessing with a
    fixed character threshold is wrong in both directions: it reports truncation that did not
    happen, and stays silent while evidence is being dropped (issue #174).

    The question half is `build_head`; `state_room` reports how much of `max_len` is left for the
    state after it, which is what a caller must size a window against.

    `return_layout=True` adds one last return value, the parallel option layout from
    `parallel_layout`: `{"position_ids": [...], "option_ids": [...]}`, one entry per token.
    """
    ids, markers, stats = build_head(tok, q, head_max_len, option_order=option_order)
    head_len = len(ids)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_text(
            tok, serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    # Laid out before the clamp: a dropped trailing option must not widen the last surviving span
    layout = parallel_layout(markers, head_len, len(ids)) if return_layout else None
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    extra = ()
    if return_truncation_stats:
        # `room` leaves space for the closing [SEP], so every token in `st` survives the [:max_len] clamp
        extra = (
            {
                "state_tokens": len(state_ids),
                "state_tokens_used": len(st),
                "state_tokens_dropped": len(state_ids) - len(st),
                "truncated": len(st) < len(state_ids),
            },
        )
    if return_layout:
        extra += ({k: v[:max_len] for k, v in layout.items()},)
    if not return_stats:
        return (ids, markers) + extra
    return (ids, markers, stats) + extra


def build_head(tok, q: Dict, head_max_len: int = 192, option_order: Optional[List[int]] = None):
    """The question half of a sequence: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`.

    `build_sequence` appends the state to this and `state_room` measures what is left over for it,
    so neither can disagree with the other about what the head costs.

    Returns `(ids, markers, stats)`: the head token ids, the `[MASK]` position of each option in
    `option_order`, and the stats dict `build_sequence` returns for `return_stats`.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = _encode_question_text(
        tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False
    )
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: `[:48]` still makes the tokenizer process the
        # whole (possibly long) description. truncation=True, max_length=48 keeps the first 48
        # tokens, which is exactly what the previous slice produced.
        opt_tokens = _encode_question_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        per_option = per
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    # Two options that share a prefix can come out of the cut as the same token span: the marker
    # count still matches the option count, so the guard in `Agent._encode_state` passes and
    # nothing downstream can tell that the question lost the ability to name them apart. Counted
    # on the capped option ids, before assembly: re-slicing the finished sequence cannot close
    # the last option's span -- it runs on into the serialized state, which differs per request,
    # so the last option always looks distinguishable however it collided (#538).
    return (
        ids,
        markers,
        {
            "options": len(opt_ids),
            "options_distinct": len({tuple(o) for o in opt_ids}),
            "tokens_per_option": per_option,
        },
    )


def parallel_layout(markers: List[int], head_len: int, length: int) -> Dict[str, List[int]]:
    """Position ids and option ids that make the encoder blind to option order.

    In the sequential layout option `s` sits at positions after option `s-1`, and every option
    attends to every other, so where an option is listed changes its embedding: the slot logits
    of five identical options spread by 3.59 (`tests/test_option_order.py`). Here every option
    starts at the same position, the first one after the instruction's [SEP], and the head's
    closing [SEP] and the state continue after the longest option. Together with
    `parallel_option_masks`, which stops an option from attending to another, reordering the
    options only reorders the marker embeddings.

    `option_ids` is 0 for shared tokens (instruction, closing [SEP], state, padding) and `s + 1`
    for the tokens of slot `s`, which runs from its [MASK] up to the next marker or, for the last
    slot, up to the [SEP] at `head_len - 1`.
    """
    if not markers:
        return {"position_ids": list(range(length)), "option_ids": [0] * length}
    start = markers[0]
    spans = list(zip(markers, markers[1:] + [head_len - 1]))
    position_ids, option_ids = list(range(start)), [0] * start
    for s, (a, b) in enumerate(spans):
        position_ids += range(start, start + b - a)
        option_ids += [s + 1] * (b - a)
    after = start + max(b - a for a, b in spans)
    position_ids += range(after, after + length - len(position_ids))
    option_ids += [0] * (length - len(option_ids))
    return {"position_ids": position_ids, "option_ids": option_ids}


def state_room(tok, q: Dict, max_len: int = 512, head_max_len: int = 192) -> int:
    """How many state tokens `q` leaves inside `max_len`, which is what `build_sequence` keeps.

    The head is the question's own -- its instructions plus one `[MASK]`-prefixed span per option --
    so a question with many options leaves less room for the state than one with two, and two
    questions in the same request do not have to leave the same amount. Anything past the return
    value is cut off (the start is kept, or the end for a conversation turn list).
    """
    head, _, _ = build_head(tok, q, head_max_len)
    return max(0, max_len - len(head) - 1)  # -1 for the [SEP] that closes the state


def window_budget(
    tok,
    questions,
    max_len: int = 512,
    head_max_len: int = 192,
    window: Optional[int] = None,
    stride: Optional[int] = None,
):
    """Window size and stride for scanning a state that is longer than one sequence.

    `predict_long` decodes each token window back to text and scores it as an ordinary state, so a
    window wider than the room the questions leave is re-truncated by `build_sequence` on the way
    in: the tail of every window reaches no model, while the reported span says it did. The window
    is therefore capped at `state_room`, and at the *smallest* room of `questions`, because the
    windows are one list of states scored for every question in shared forward passes -- a window
    sized for the roomiest question would be cut short for the tightest one, and the offsets
    reported on its answers would mean something different per question. With no questions there is
    nothing to fit, so the caller's window (or the checkpoint default) stands. That default is
    `max(64, max_len - head_max_len - 8)`: the state budget the config leaves, with a 64-token
    floor under it, so a widened `head_max_len` stops shrinking it there.

    `questions` are internal question dicts, as `Agent._to_internal` returns them.

    An explicit `window` wider than the room is clamped to it with a `RuntimeWarning`, since a
    scan at the requested size cannot read what it claims to. A `stride` past the effective window
    is refused: the tokens between two windows would be read by neither, which is the failure
    `predict_long` exists to prevent. The default stride keeps its 50% overlap of the *effective*
    window, so a span near a boundary still lands whole inside some window; an explicit stride
    equal to the window still reads every token, with no overlap to catch a span that straddles a
    boundary.

    Returns `(window, stride, room)`.
    """
    rooms = [state_room(tok, q, max_len, head_max_len) for q in questions]
    requested = window if (window and window > 0) else max(64, max_len - head_max_len - 8)
    size = requested
    room = min(rooms) if rooms else size
    if room <= 0:
        raise ValueError(
            "predict_long: the questions' options fill the whole sequence (max_len=%d,"
            " head_max_len=%d), leaving no room for the state; no window can carry any of it."
            " A label set this large is what laya.shortlist.predict_shortlist is for"
            % (max_len, head_max_len)
        )
    if size > room:
        if window and window > 0:
            warnings.warn(
                "laya: predict_long: window=%d is wider than the %d state tokens these questions"
                " leave inside max_len=%d, so every window would be truncated to %d on the way to"
                " the model; scanning with window=%d instead" % (size, room, max_len, room, room),
                RuntimeWarning,
                stacklevel=3,
            )
        elif size >= room * _WINDOW_CLAMP_WARN_RATIO:
            # The DEFAULT window was cut, and cut hard. Capping it is what stops the tail of every
            # window reaching no model, but it is not free and it must not be silent: the scan now
            # needs about `size / room` times as many windows, each one a full forward pass, and
            # nothing in the caller's code says why. Measured on the English checkpoint with 100
            # four-word options: room 102 of max_len 512, so the question head alone is 409 tokens,
            # the window falls 312 -> 102 and the scan goes 11 windows -> 36, a 3.1x wall-clock
            # increase (1902 ms -> 5858 ms) for the same document.
            #
            # The cost is not the capping, it is the shape of the request: 80% of every sequence is
            # the question, and because the encoder is bidirectional the head cannot be computed
            # once and reused -- its representations depend on the state it is paired with. So the
            # warning names the real remedy rather than only reporting the clamp.
            warnings.warn(
                "laya: predict_long: these questions leave only %d of max_len=%d for the state"
                " (their heads take the rest), so the scan window is capped %d -> %d and roughly"
                " %.1fx as many windows -- each a full forward pass -- are needed to read the"
                " document. Fewer or shorter options, a larger max_len, or"
                " laya.shortlist.predict_shortlist for a large label set will all cost less than"
                " scanning at this width" % (room, max_len, size, room, size / room),
                RuntimeWarning,
                stacklevel=3,
            )
        size = room
    step = stride if (stride and stride > 0) else max(1, size // 2)
    if step > size:
        if size < requested and stride and stride <= requested:
            # The window the caller asked for was reduced above, and their stride was valid for the
            # window they asked for -- so this is the library's clamp, not their mistake. Reducing
            # the stride to match keeps the no-gap guarantee without making a self-consistent pair
            # of arguments an error. A stride that overshot the *requested* window is still refused
            # below, because that one really is the caller's.
            warnings.warn(
                "laya: predict_long: stride=%d was a 50%% step for the window=%d you asked for, but"
                " the window was reduced to %d to fit the room these questions leave; scanning with"
                " stride=%d instead" % (step, requested, size, max(1, size // 2)),
                RuntimeWarning,
                stacklevel=3,
            )
            step = max(1, size // 2)
        else:
            raise ValueError(
                "predict_long: stride=%d steps past the %d-token window%s, so %d tokens between"
                " every pair of windows would be read by no window at all; pass stride <= %d"
                % (
                    step,
                    size,
                    " these questions leave room for" if size < requested else "",
                    step - size,
                    size,
                )
            )
    return size, step, room


def check_batch_size(batch_size: Any) -> Optional[int]:
    """Validate the per-call `batch_size` cap before the chunker has to spend it.

    `batch_size` counts states per forward pass, and the batch loop uses it as a `range()` step.
    A `float` -- what a JSON `2.0` parses to, and what `n / 2` yields -- reached `range()` and came
    back as the interpreter's own `'float' object cannot be interpreted as an integer`; a `str` was
    compared to zero first and leaked `'>' not supported between instances of 'str' and 'int'`;
    `True` is an `int` to `range()`, so it silently capped every pass at one state and switched
    `sort_by_length` off, because that knob's own gate is `1 < chunk`. `laya/shortlist.py`,
    `laya/mcp/tools.py` and `laya/serve.py` already refuse these shapes; the two `predict_batch`
    entry points and `window_batch_cap` did not, so a Python caller -- the documented route, and
    the one with no schema in front of it -- met the raw `TypeError` instead of the named refusal.

    `None`, `0` and a negative integer are **not** refused: they mean "no cap", the chunker uses
    `len(states)` and `window_batch_cap` returns `None`. That is pinned, not incidental --
    `laya-java/fixtures/window_plan.json`'s `batch_cap` grid replays `(None, 0, -1, 1, 8)` through
    the Java, Go, C# and C++ ports, and those ports take a typed `int` whose zero and negative
    values are their own "unset", so refusing here would move the oracle for every port while
    helping no caller. Anything with `__index__` is accepted, a `numpy.int64` among them, for the
    same reason `ece_score` in this module gives: a value `range()` can spend is a value worth
    accepting. A `bool` is refused even though it is an `int`, because `True` is not a count that
    anyone meant to pass.
    """
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be an integer count of states, got %r" % (batch_size,))
    return int(batch_size)


def window_batch_cap(
    n_windows: int, window: int, config_budget: int, batch_size: Optional[int] = None
) -> Optional[int]:
    """Keep one forward pass no wider than the un-capped scan's would have been -- but only when
    capping the window has multiplied the window count enough to matter.

    Capping the window at the room the questions leave multiplies the number of windows on exactly
    the inputs it targets: measured, a 4 561-token document at 120 options goes from 29 windows of
    312 tokens to roughly 413 of 23. `predict_batch` with `batch_size=None` puts every state in one
    forward pass, so a caller who passed no `batch_size` would go from a 29-row pass to a 413-row one
    at `max_len` width -- a plausible out-of-memory on an input that used to fit.

    Sending every window in one run is also a deliberate property (`ONNXAgent` asserts it), and a mild
    cap -- a 64-token budget reduced to 43, say -- multiplies the count by well under two. Chunking
    those would trade a real property for no real protection. So the cap only applies past
    `_WINDOW_BATCH_BLOWUP`x, and then bounds the pass at the count the un-capped budget would have
    produced: peak memory stays at parity with the behaviour before the cap, and the scan still reads
    the whole document, just in more passes. An explicit `batch_size` is always honoured --
    `check_batch_size` decides what counts as explicit, and a value it refuses is one no chunker
    downstream could honour either.
    """
    batch_size = check_batch_size(batch_size)
    if batch_size and batch_size > 0:
        return batch_size
    if window >= config_budget:
        return None  # not capped: one pass, exactly as before
    unclamped = max(1, -(-n_windows * max(1, window // 2) // max(1, config_budget // 2)))
    if n_windows <= unclamped * _WINDOW_BATCH_BLOWUP:
        return None  # a mild cap: keep the single shared pass
    return unclamped


def collapsed_options(qids, items) -> Dict[str, Dict[str, Optional[int]]]:
    """The questions whose options no longer have a token span each, from per-item stats.

    `total` is the number of options the question defines, not the number of markers that
    reached the sequence: a report counted from the markers would say "43/58" about a request
    where 28 options never made it into the input at all.
    """
    out = {}
    for qid, item in zip(qids, items):
        stats = item.get("options")
        if stats and stats["options_distinct"] < stats["options"]:
            out[qid] = {
                "total": stats["options"],
                "distinct": stats["options_distinct"],
                "tokens_per_option": stats["tokens_per_option"],
            }
    return out


def unpermute_probs(p: np.ndarray, option_order: Optional[List[int]]) -> np.ndarray:
    """Put a slot-ordered probability row back into the caller's option order.

    `build_sequence` puts option `option_order[s]` in slot `s`, so a model row comes back
    indexed by slot. Everything downstream indexes by option -- `zip(keys, p)` for a choice,
    `arange(k) * p` for a score level, `p[1]` for noul-true -- so the row has to be inverted
    first or the probabilities end up attached to the wrong options, which is silent.

    A missing or mismatched order returns `p` untouched, so the canonical path is unaffected.
    """
    if option_order is None or len(option_order) != len(p):
        return p
    canonical = np.empty_like(p)
    canonical[np.asarray(option_order, dtype=int)] = p
    return canonical


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins.

    `bins` is the number of histogram buckets and must be an integer of at least 1. It is refused
    rather than misread because `np.linspace(0, 1, bins + 1)` yields no interior edge for `0` and
    `-1`: the loop body never runs and the function returns `0.0`, the best score ECE can report,
    for a distribution that is anything but well calibrated. A count that arrives from a config
    file as `0` therefore turns a calibration gate into one that always passes. `True` is refused
    with the same argument -- it would silently histogram into a single bucket. Any other
    `Integral` is accepted, a `numpy.int64` among them, since the arrays here are numpy's own.
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be an integer of at least 1, got %r" % (bins,))
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits, and the quantity every calibration figure in
    this repository is computed on -- both benchmark harnesses take `conf = max(probs)` before
    calling `ece_score`. The README's gating section relies on the property that goes with it:
    of the answers returned at confidence c, about c of them are right. That property is
    conditional, and the condition is not met by default -- it holds only after the temperatures
    have been fitted and validated on held-out data for this checkpoint and this option count.
    The shipped checkpoints are over-confident: `choice:11+` is fitted at 0.1006, a ~10x
    sharpener that would publish a genuine 0.24 top probability as 0.9999. `clamp_temperature`
    refuses it and applies `TEMP_MIN` (0.5) instead, which still sharpens 2x and publishes that
    same 0.24 as 0.4993 -- so a threshold applied to these numbers still selects below model
    accuracy (issue #394). The clamp bounds the damage; it does not calibrate anything.

    `confidence_from_probs` below reports a different quantity on a different scale and carries
    no such guarantee, so the two must not be compared against the same threshold.
    """
    if k < 1:
        return 1.0
    val = float(p[:k].max())
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k).

    How concentrated the whole distribution is. Useful, but not calibrated: it is not what
    temperature scaling fits and not what the reported ECE measures. See `answer_confidence`.

    The formula is reported only for ``k >= 2``: ``log(1)`` and ``log(0)`` are zero, so a
    distribution over fewer than two options has no entropy scale to divide by. Those inputs
    return ``1.0``, the *maximum* score this function can produce, whatever ``p`` holds -- an
    empty or single-option question reads as perfectly concentrated, and a caller comparing the
    reported `confidence` against any threshold sees a pass. That is not the same as the
    abstention gate passing it: `laya.confidence` gates on `answer_confidence` first and reaches
    this number only when that field is absent. The gate does pass a real one-option answer, for
    a different reason -- a softmax over one logit is exactly ``[1.0]``, so both numbers are
    ``1.0``. `laya.structured` builds such an answer from ``{"enum": ["solo"]}``, and the option
    count is bounded only above (`MAX_OPTIONS`).
    """
    if k < 2:
        return 1.0
    # The scalar `np.clip` on the score is the part worth replacing (a native comparison is ~4x
    # the dispatcher); the entropy sum itself stays vectorised. A Python loop over `pk` is faster
    # below roughly k=40 and slower above it -- 0.9 us vs 2.4 us at k=11, but 14.6 us vs 3.0 us at
    # k=256 -- and summing in a different order moves the last bits (1.7e-15 over 136k measured
    # distributions). Both callers `round(..., 4)` this, so that drift is invisible today; it is
    # the SDK goldens in laya-rust, laya-dotnet and laya-ts that have to keep reproducing these
    # numbers, and there is no reason to spend bit-exactness on a microsecond.
    pk = p[:k]
    ent = -float((pk * np.log(np.clip(pk, 1e-12, 1.0))).sum())
    score = 1.0 - ent / math.log(k)
    return 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number.

    A bool is not a number either: `True`/`False` used to float to 1.0/0.0 here and read as
    fitted/sharpening temperatures, the same class of quiet acceptance `check_min_confidence`
    already refuses. A custom `lo`/`hi` still bounds either way.
    """
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError, OverflowError):
        # `OverflowError` is a number this function cannot read, not a bug in the caller: JSON's
        # grammar has no double limit, so a temperature written as a 401-digit integer literal
        # parses as an `int` and `float()` refuses it. (`1e400` in the same file is fine -- JSON
        # reads that as a float, which is `inf`, and the check below already answers 1.0 for it.)
        # `confidence._finite`, `hooks.validate_timeout` and `calibrate.apply_calibration_payload`
        # all read this exception as "does not read as a number" already; here it used to escape,
        # and with it every `Agent`/`ONNXAgent`/`load_calibration` load that had read the
        # checkpoint off disk by then.
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):  # NaN / inf
        return 1.0
    return min(hi, max(lo, t))


def resolve_lang_temperatures(
    raw: Optional[Dict[str, Any]], base_temperature: Sequence[float]
) -> Dict[str, Dict[str, Any]]:
    """Parse the `lang_temperatures` option into `{language: {temperature, temperature_by_options}}`.

    One implementation, because `Agent` and `ONNXAgent` both accept this option and both promise
    the same confidences for it. Reading it with `cfg.get(...)` and `len(...)` before checking the
    shape of either raised `AttributeError` and `TypeError` for exactly the inputs the
    `ValueError` below is written for, after the whole checkpoint had loaded:

        {"de": {"temperature": 2}}       -> TypeError: object of type 'int' has no len()
        {"de": {"temperature": None}}    -> TypeError: object of type 'NoneType' has no len()
        {"de": None}                     -> AttributeError: 'NoneType' object has no attribute 'get'

    A `null` entry or a `null` temperature both mean "inherit the checkpoint's own", which is how
    the `laya-ts` port reads the same option (`agent.ts:322-327`).

    Every value still goes through `clamp_temperature`, and a value the clamp had to replace is named
    in a `RuntimeWarning` for its language: `{"de": {"temperature": [True, 1.5, 2.0]}}` answers
    `[1.0, 1.5, 2.0]` and `[99, 0.01, 1]` answers `[5.0, 0.5, 1.0]`, which is a different confidence
    scale from the one the caller wrote. `temperature_rejected` is what distinguishes a replaced entry
    from an accepted one.
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    for lang, cfg in (raw or {}).items():
        if not isinstance(lang, str):
            raise ValueError("Language override keys must be strings, got %r" % (lang,))
        norm = lang.split("-")[0].lower()
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError(
                "Language override %r must be a mapping, got %s" % (lang, type(cfg).__name__)
            )
        t_raw = cfg.get("temperature")
        inherited = t_raw is None
        if inherited:
            t_raw = base_temperature
        if not isinstance(t_raw, (list, tuple)) or len(t_raw) != 3:
            raise ValueError(
                "Language override %r temperature must be a list of 3 floats, got %r"
                % (lang, t_raw)
            )
        tbo_raw = cfg.get("temperature_by_options") or {}
        if not isinstance(tbo_raw, dict):
            raise ValueError(
                "Language override %r temperature_by_options must be a mapping of "
                "bucket -> float, got %s" % (lang, type(tbo_raw).__name__)
            )
        temperature = [clamp_temperature(t) for t in t_raw]
        by_options = {k: clamp_temperature(v) for k, v in tbo_raw.items()}
        # An override that omits `temperature` inherits the checkpoint's own list, and the loader
        # that got here has already reported the entries it replaced in it. Repeating that line once
        # per language would bury the overrides a reader is actually holding.
        entries = (
            []
            if inherited
            else [
                ("temperature[%d]" % i, raw, applied)
                for i, (raw, applied) in enumerate(zip(t_raw, temperature))
            ]
        )
        entries += [("%s" % k, v, by_options[k]) for k, v in tbo_raw.items()]
        rejected = [
            "%s=%r -> %g" % (name, raw, applied)
            for name, raw, applied in entries
            if temperature_rejected(raw, applied)
        ]
        if rejected:
            warnings.warn(
                "laya: language override %r ships invalid temperatures or values outside "
                "[%g, %g]; using %s. Treat confidence from the affected entries as uncalibrated."
                % (lang, TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning,
                stacklevel=2,
            )
        resolved[norm] = {
            "temperature": temperature,
            "temperature_by_options": by_options,
        }
    return resolved


resolve_noul_labels = _resolve_noul_labels


def option_layout(cfg):
    layout = cfg.get("option_layout", "sequential")
    if layout not in ("sequential", "parallel"):
        raise ValueError(f"Unknown option_layout: {layout!r}")
    return layout


def build_prefix(tok, q, head_max_len=192, option_order=None, *, return_stats=False):
    values = build_head(tok, q, head_max_len, option_order)
    return values if return_stats else values[:2]


def read_temperatures(cfg):
    raw = cfg.get("temperature", [1.0, 1.0, 1.0])
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("temperature must be a list of 3 floats")
    raw_by = cfg.get("temperature_by_options", {})
    temperature = [clamp_temperature(t) for t in raw]
    by = {k: clamp_temperature(v) for k, v in raw_by.items()}
    from .calibrate import _rejected_temperatures

    entries = [(k, v, by[k]) for k, v in raw_by.items()]
    entries += [("temperature[%d]" % i, t, temperature[i]) for i, t in enumerate(raw)]
    rejected = _rejected_temperatures(entries)
    if rejected:
        warnings.warn(
            "laya: this checkpoint ships temperatures outside [%g, %g] which would distort "
            "confidence; clamping %s. Treat confidence from the affected buckets as uncalibrated."
            % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
            RuntimeWarning,
            stacklevel=2,
        )
    return temperature, by, raw, raw_by


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    has_target = any("target" in it for it in items)
    target = np.zeros((n, kmax), dtype=np.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = np.array(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = np.array(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            if len(it["target"]) > k:
                # Otherwise this lands as "The expanded size of the tensor (k) must match the
                # existing size (kmax)" from inside the assignment, which says nothing about the
                # actual mistake: a target with more entries than the item has options. The limit
                # is this item's own marker count, not the batch-wide kmax: in a mixed-width
                # batch a longer sibling row must not legitimise extra entries (#311).
                raise ValueError(
                    "collate_items: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(it["target"]), k)
                )
            target[i, : len(it["target"])] = np.array(it["target"], dtype=np.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": np.array([it["qtype"] for it in items]),
        "label": np.array([it.get("label", -1) for it in items]),
        "meta": [
            {k: it[k] for k in it if k not in ("ids", "markers", "target", "layout")}
            for it in items
        ],
    }
    if target is not None:
        res["target"] = target
    if any("layout" in it for it in items):
        # All or none: a batch that mixes layouts would run half its rows through the wrong masks
        if not all("layout" in it for it in items):
            raise ValueError("collate_items: some items carry a parallel layout and some do not")
        pos = np.zeros((n, L), dtype=np.int64)
        opt = np.zeros((n, L), dtype=np.int64)
        for i, it in enumerate(items):
            pos[i, : len(it["ids"])] = np.array(it["layout"]["position_ids"])
            opt[i, : len(it["ids"])] = np.array(it["layout"]["option_ids"])
        res["position_ids"], res["option_ids"] = pos, opt
    return res


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    import torch

    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    import torch

    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def temperature_rejected(raw, applied: float) -> bool:
    """Whether `raw` is not the number `clamp_temperature` turned into `applied`.

    Loaders use this to name the entries they had to replace. The obvious test, `float(raw) !=
    applied`, misses the bool case: `clamp_temperature(True)` returns the neutral 1.0 and
    `float(True)` *is* 1.0, so a checkpoint or calibration file shipping `true` where a number
    belongs read as an accepted temperature and loaded in silence, while `false` floated to 0.0,
    mismatched and was reported. So ask the question `clamp_temperature` itself asks -- is this a
    number at all -- instead of re-deriving its answer. A value that will not convert is rejected
    the same way: it has a neutral fallback to report, and the diagnostic must never raise for it.
    """
    if isinstance(raw, bool):
        return True
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        # The same "does not read as a number" `clamp_temperature` answers with 1.0, so the entry
        # is named rather than the loader raising while it builds the warning about it. A wide
        # integer literal is that case: it has a neutral fallback to report, and the value that
        # `float()` cannot hold is exactly the one a reader needs to see.
        return True
    return value != applied


resolve_noul_labels = _resolve_noul_labels


def uses_parallel_layout(cfg):
    layout = cfg.get("option_layout", "sequential")
    if layout not in ("sequential", "parallel"):
        raise ValueError(f"Unsupported option_layout: {layout!r}")
    return layout == "parallel"


def finish_sequence(tok, prefix, markers, state_ids, max_len, truncate_left=False):
    """Append the same state slice in cached and uncached preparation."""
    room = max(0, max_len - len(prefix) - 1)
    kept = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]
    ids = (list(prefix) + kept + [tok.sep_token_id])[:max_len]
    return (
        ids,
        [m for m in markers if m < max_len],
        {
            "state_tokens": len(state_ids),
            "state_tokens_used": len(kept),
            "state_tokens_dropped": len(state_ids) - len(kept),
            "truncated": len(kept) < len(state_ids),
        },
    )
