"""Opt-in embedding shortlist and tournament for high-cardinality choice questions.

Choice options share one ``head_max_len`` budget, so a large label set leaves only a few
tokens per label. ``predict_shortlist`` embeds the state and each option with a
caller-supplied ``embed_fn``, keeps the top ``k``, and runs a single ``predict`` (or
``system_one``) on that reduced criteria set. ``predict_tournament`` needs no embedder: the
decision model answers the labels in groups small enough to keep their tokens, and the group
winners meet in one more ``predict``.

``Agent.predict`` and ``Agent.system_one`` are separate: they still score every criterion
they are given. This module does not change ``DecisionModel.forward``.
``predict_shortlist`` adds no second decision-model pass; ``predict_tournament`` adds one
per elimination round.

The coarse-to-fine pattern is the one the README recommends and the one reported in
https://github.com/NandhaKishorM/laya/issues/102. Ranking here is cosine similarity on
whatever vectors ``embed_fn`` returns. Issue #102's BANKING77 figures belong to that
report; this module does not measure them.
"""

import copy
import json
import threading
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .common import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20
DEFAULT_TOURNAMENT_GROUP = 16


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
    return_scores: bool = False,
) -> Any:
    """Return the top-``k`` choice labels for ``state``.

    ``embed_fn`` maps a list of strings to an array of shape ``(len(texts), dim)``.
    It is called once, with the query text first and then one string per option in
    criteria order. Option strings match ``render_options`` for a choice question.

    When ``k`` is at least the number of labels, every label is returned in its
    original order and ``embed_fn`` is not called.

    Ties keep the earlier label. Ranking is a signed cosine, not a similarity floor:
    a label that scores 0 -- no signal at all, or a non-finite vector treated as one
    -- does outrank an earlier label that scored negative, and ``k`` drops the
    negative labels first.

    With ``return_scores=True`` the return is the ``(labels, scores)`` pair, where
    ``scores`` holds the signed cosine per kept label in rank order -- the same
    values ``predict_shortlist`` reports in its ``shortlist`` metadata. ``scores``
    is ``None`` when nothing was dropped, exactly as in that metadata.
    """
    labels, scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    if return_scores:
        return labels, scores
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Shortlist each choice question, then call ``predict`` or ``system_one`` once.

    Non-choice questions are forwarded unchanged. A choice whose label count is
    ``<= k`` is forwarded unchanged and does not call ``embed_fn``. The caller's
    ``questions`` dict is not mutated.

    A narrowed choice that carried an ``option_order`` has it dropped, with one
    ``RuntimeWarning`` naming the question: the order has one slot per option, so an
    order that fitted the caller's label set no longer fits the kept labels and the
    agent refuses the question. Remapping the order onto the kept labels would
    silently move options between slots, so it is dropped instead. A passthrough
    choice and non-choice questions keep theirs.

    The returned dict is the model result plus a ``shortlist`` entry. Probabilities
    on a shortlisted choice are over the kept labels only. ``shortlist`` has an entry
    for each choice question; a non-choice id is absent, because no ranking ran for it.
    Each entry holds ``labels``, ``scores``, ``k``, ``n``, and ``passthrough``.
    ``labels`` is the rank order a shortlist produced, or the criteria order itself when
    ``passthrough`` is set and no ranking ran, and it always holds ``min(k, n)`` labels;
    ``scores`` is the signed cosine of each kept label in that order -- negative
    included, never clamped to 0 -- or ``None`` when nothing was dropped. Neither
    ``k`` nor ``n`` counts what ``labels`` kept: ``k`` is the cap the caller asked
    for, which a question with fewer labels than that does not reach, and ``n`` is
    the caller's label count before any narrowing.

    Extra keyword arguments are forwarded to ``predict`` / ``system_one`` (for
    example ``model=`` on a ``Router``).
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    checked = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    warned: set = set()
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError("question %r is a choice but has no criteria" % (qid,))
        labels, scores, passthrough, n = _rank(
            state, qdef["criteria"], embed_fn, checked, qdef.get("instructions")
        )
        meta[qid] = {
            "labels": list(labels),
            "scores": scores,
            "k": checked,
            "n": n,
            "passthrough": passthrough,
        }
        if passthrough:
            reduced[qid] = qdef
            continue
        updated = dict(qdef)
        updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
        reduced[qid] = _drop_stale_order(qid, updated, len(labels), warned)

    result = _call_predict(agent, state, reduced, **predict_kwargs)
    out = dict(result)
    out["shortlist"] = meta
    return out


def predict_tournament(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    group_size: int = DEFAULT_TOURNAMENT_GROUP,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Narrow each large choice question by elimination, then call ``predict`` once more.

    A choice with more than ``group_size`` labels is cut, in criteria order, into groups of
    near-equal size and at most ``group_size``. One ``predict`` call answers every group of
    every such question -- the groups go in as separate questions, so they share one forward
    pass -- and each group's answer goes through to the next round. Rounds repeat until no
    choice has more than ``group_size`` labels left; at the default 16, up to 256 labels take
    one round. Unlike ``predict_shortlist`` this needs no embedder, and each label is read
    with the option budget of a ``group_size``-label question, not of the whole label set.

    The final call answers every question of the request, with each choice that went through
    a round cut to its finalists. Non-choice questions and choices of at most ``group_size``
    labels go to it unchanged, so when nothing needs a round it is the only call. The
    caller's ``questions`` dict is not mutated.

    A choice that went through a round loses an ``option_order`` it carried, with one
    ``RuntimeWarning`` naming the question: the group and finalist questions hold fewer
    labels than the caller's, and an order no longer fits the kept labels. Remapping it
    onto them would silently move options between slots, so it is dropped instead. A
    choice that needed no round keeps its order.

    The returned dict is the final call's result plus a ``tournament`` entry. It has
    an entry for each choice question -- a non-choice id is absent, since such a
    question plays no round -- holding ``labels`` (the finalists, in criteria order),
    ``n`` and ``rounds``. ``n`` is the caller's label count before the first round,
    the same meaning as in ``predict_shortlist``'s metadata: it says how wide the
    field was, not how many finalists came out of it, so on a choice that played a
    round it is larger than ``len(labels)``. Probabilities, confidences and ``usage``
    come from the final call, so a tournament choice's probabilities are over its
    finalists only.

    Every call's reply is checked against the same shape, the elimination rounds included: a reply
    that is not a dict, and one whose ``answers`` is not a dict, each raise that contract by name
    instead of an indexing error or a bare ``KeyError`` from inside the module. What an answer holds
    is not checked: the final call's answers go to the caller as the agent wrote them.

    Extra keyword arguments are forwarded to every call (for example ``model=`` on a ``Router``).
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer of at least 2, got %r" % (group_size,))
    meta: Dict[str, Dict[str, Any]] = {}
    warned: set = set()
    for qid, qdef in questions.items():
        if isinstance(qdef, dict) and qdef.get("type") == "choice":
            if "criteria" not in qdef:
                raise ValueError("question %r is a choice but has no criteria" % (qid,))
            labels = [key for key, _value in _criteria_items(qdef["criteria"])]
            meta[qid] = {"labels": labels, "n": len(labels), "rounds": 0}

    def cut(qid, labels):
        narrowed = dict(
            questions[qid], criteria=_subset_criteria(questions[qid]["criteria"], labels)
        )
        return _drop_stale_order(qid, narrowed, len(labels), warned)

    while True:
        groups = []
        for qid, entry in meta.items():
            labels = entry["labels"]
            parts = -(-len(labels) // group_size)
            if parts > 1:
                groups += [
                    (qid, labels[i * len(labels) // parts : (i + 1) * len(labels) // parts])
                    for i in range(parts)
                ]
        if not groups:
            break
        round_questions = {str(i): cut(qid, labels) for i, (qid, labels) in enumerate(groups)}
        answers = _round_answers(_call_predict(agent, state, round_questions, **predict_kwargs))
        winners: Dict[str, List[Any]] = {}
        for i, (qid, _labels) in enumerate(groups):
            winners.setdefault(qid, []).append(answers[str(i)]["choice"])
        for qid, labels in winners.items():
            meta[qid]["labels"] = labels
            meta[qid]["rounds"] += 1

    final = dict(questions)
    for qid, entry in meta.items():
        if entry["rounds"]:
            final[qid] = cut(qid, entry["labels"])
    result = _call_predict(agent, state, final, **predict_kwargs)
    out = dict(result)
    out["tournament"] = meta
    return out


def embed_fn_from_agent(
    agent: Any,
    max_length: int = 512,
    batch_size: int = 32,
    *,
    bucket_by_length: bool = False,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Mean-pool the checkpoint encoder already loaded on ``agent``.

    The callable embeds a list of strings with ``agent.tok`` and ``agent.model.encoder``.
    It does not run the decision head and does not download weights. A dedicated
    bi-encoder passed as ``embed_fn`` will usually shortlist better; this helper is
    for callers who only have the Laya checkpoint in memory.

    Padding positions are excluded from the mean. The encoder runs in eval mode on the
    agent's own device (a loaded ``Agent`` is already in eval).

    ``bucket_by_length=True`` groups similar token lengths to reduce padding and
    restores input order. Different batch shapes can change floating-point rounding
    and near-tied shortlist rankings, especially in FP16; the default keeps input batches.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer, got %r" % (max_length,))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer, got %r" % (batch_size,))

    tok = getattr(agent, "tok", None)
    encoder = getattr(getattr(agent, "model", None), "encoder", None)
    if tok is None or encoder is None:
        raise TypeError("agent must provide tok and model.encoder")

    import mlx.core as mx

    device = agent.device
    tokenizer = copy.deepcopy(tok.backend)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.no_padding()

    def embed_fn(texts: Sequence[str]) -> np.ndarray:
        rows = ["" if text is None else str(text) for text in texts]
        hidden = _hidden_size(encoder)
        if not rows:
            return np.zeros((0, hidden), dtype=np.float32)
        parts: List[np.ndarray] = []
        order = []
        for indices, encoded in _embedding_batches(tokenizer, rows, batch_size, bucket_by_length):
            order.extend(indices)
            length = max(1, max(len(ids) for ids in encoded))
            input_ids = np.full((len(encoded), length), tok.pad_token_id, dtype=np.int32)
            attention_mask = np.zeros((len(encoded), length), dtype=np.bool_)
            for i, ids in enumerate(encoded):
                input_ids[i, : len(ids)] = ids
                attention_mask[i, : len(ids)] = True
            with mx.stream(device):
                hidden_states = encoder(mx.array(input_ids), mx.array(attention_mask))
                mask = mx.array(attention_mask.astype(np.float32))[:, :, None]
                pooled = (hidden_states.astype(mx.float32) * mask).sum(axis=1) / mx.maximum(
                    mask.sum(axis=1), 1.0
                )
                mx.eval(pooled)
            parts.append(np.asarray(pooled, dtype=np.float32))
        result = np.concatenate(parts, axis=0)
        return result[np.argsort(order)] if bucket_by_length else result

    return embed_fn


def _embedding_batches(tokenizer, rows, batch_size, bucket_by_length):
    if not bucket_by_length:
        for start in range(0, len(rows), batch_size):
            indices = range(start, min(start + batch_size, len(rows)))
            yield indices, [tokenizer.encode(rows[i], add_special_tokens=True).ids for i in indices]
        return

    encoded = [tokenizer.encode(text, add_special_tokens=True).ids for text in rows]
    lengths = [max(1, len(ids)) for ids in encoded]
    order = sorted(range(len(rows)), key=lengths.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        # Bound padding to twice the shortest row, while respecting the caller's batch cap.
        while (
            end < min(start + batch_size, len(order))
            and lengths[order[end]] <= 2 * lengths[order[start]]
        ):
            end += 1
        indices = order[start:end]
        yield indices, [encoded[i] for i in indices]
        start = end


def cached_embed_fn(
    embed_fn: Callable[[Sequence[str]], Any],
    maxsize: int = 4096,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Cache ``embed_fn`` output per input string, under an LRU bound.

    ``predict_shortlist`` embeds the query plus every option text on each call. When the
    same option set is shortlisted on every request -- a fixed intent or label list, as
    in the README's BANKING77 example -- the option rows do not change between calls,
    yet they are re-embedded every time. Wrapping the embedder once::

        embed_fn = cached_embed_fn(embed_fn_from_agent(agent))

    leaves the first call's embedder work unchanged and reduces each repeat call to
    embedding the new query alone.

    Lookups are exact string matches. Texts missing from the cache are deduplicated and
    embedded in a single ``embed_fn`` call, so a cold cache costs the same number of
    batched calls as the unwrapped function. Rows are stored as float32; the cache holds
    at most ``maxsize`` strings and then evicts the least recently used entry, bounding
    memory at about ``maxsize * dim * 4`` bytes. Nothing is cached when ``embed_fn``
    raises or returns a bad shape. A non-finite value is not a bad shape: ``NaN``,
    ``inf`` and ``-inf`` become ``0.0`` on the way in, the same replacement
    :func:`_embeddings` makes for an unwrapped embedder, and the repaired row is then
    *cached*. One transient ``NaN`` in an option's embedding therefore ranks that option
    as a zero vector on every later request until ``cache_clear()``, while the unwrapped
    embedder would simply re-embed it -- shortlist membership can differ between the two
    on the second call. The float32 storage shows in the scores too: a wrapped first
    call can differ from the unwrapped one in its last digits.

    The wrapper is safe to share between threads: the lock covers only cache reads and
    writes, never the embedding call. The returned callable carries ``cache_info()`` --
    a dict with ``size``, ``maxsize``, ``hits`` and ``misses`` -- and ``cache_clear()``.
    Clear the cache if the model or weights behind ``embed_fn`` change. Calls already in
    flight still return their result, but cannot refill the cleared cache.
    """
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
        raise ValueError("maxsize must be a positive integer, got %r" % (maxsize,))

    rows_by_text: OrderedDict[str, np.ndarray] = OrderedDict()
    lock = threading.Lock()
    counts = {"hits": 0, "misses": 0}
    generation = 0

    def cached(texts: Sequence[str]) -> np.ndarray:
        keys = ["" if text is None else str(text) for text in texts]
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        with lock:
            started_generation = generation
            found: Dict[str, np.ndarray] = {}
            hits = 0
            for key in keys:
                row = rows_by_text.get(key)
                if row is not None:
                    rows_by_text.move_to_end(key)
                    found[key] = row
                    hits += 1
            counts["hits"] += hits
            counts["misses"] += len(keys) - hits
            missing = [key for key in dict.fromkeys(keys) if key not in found]
        if missing:
            raw = embed_fn(list(missing))
            if hasattr(raw, "detach"):
                raw = raw.detach().float().cpu().numpy()
            fresh = np.asarray(raw, dtype=np.float32)
            if fresh.ndim != 2 or fresh.shape[0] != len(missing) or fresh.shape[1] < 1:
                raise ValueError(
                    "embed_fn must return an array of shape (%d, dim), got %s"
                    % (len(missing), tuple(fresh.shape))
                )
            fresh = np.nan_to_num(fresh, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
            found.update(zip(missing, fresh))
            with lock:
                if generation != started_generation:
                    return np.stack([found[key] for key in keys])
                if rows_by_text:
                    # Rows stored under one dimensionality cannot be stacked against rows of
                    # another: a changed embedder (or one whose dimension drifts between calls)
                    # would otherwise surface rows of mixed width, which `_rank` reads as one
                    # matrix. Deciding by identity is what the docstring already asks of the
                    # caller ("clear the cache if the model changes") -- this refuses the call
                    # instead of returning a matrix that silently mixes both. It is inside the
                    # lock because a concurrent insert, eviction or `cache_clear()` mutates the
                    # OrderedDict this reads.
                    want = next(iter(rows_by_text.values())).shape[0]
                    if fresh.shape[1] != want:
                        raise ValueError(
                            "embed_fn returned dim %d, but the cache holds dim %d; "
                            "call cache_clear() if the model behind embed_fn changed"
                            % (fresh.shape[1], want)
                        )
                for key, row in zip(missing, fresh):
                    rows_by_text[key] = row
                    rows_by_text.move_to_end(key)
                    while len(rows_by_text) > maxsize:
                        rows_by_text.popitem(last=False)
        return np.stack([found[key] for key in keys])

    def cache_info() -> Dict[str, int]:
        with lock:
            return {
                "size": len(rows_by_text),
                "maxsize": maxsize,
                "hits": counts["hits"],
                "misses": counts["misses"],
            }

    def cache_clear() -> None:
        nonlocal generation
        with lock:
            generation += 1
            rows_by_text.clear()
            counts["hits"] = 0
            counts["misses"] = 0

    cached.cache_info = cache_info
    cached.cache_clear = cache_clear
    return cached


def _rank(state, criteria, embed_fn, k, instructions):
    checked = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]
    if checked >= n:
        return list(keys), None, True, n
    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = np.argsort(-sims, kind="mergesort")[:checked]
    labels = [keys[int(i)] for i in order]
    scores = [float(sims[int(i)]) for i in order]
    return labels, scores, False, n


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer, got %r" % (k,))
    return k


def _drop_stale_order(
    qid: str, question: Dict[str, Any], n_options: int, warned: set
) -> Dict[str, Any]:
    """Return ``question`` without an ``option_order`` that no longer fits it.

    ``option_order`` holds one slot per option, so an order that fitted the caller's
    label set does not fit a narrowed one and the agent refuses the question.
    Remapping the order onto the kept labels would silently move options between
    slots, so the order is dropped instead, once per question per call, with a
    ``RuntimeWarning``. This is the remedy Nandha asked for on #958.
    """
    if "option_order" not in question:
        return question
    order = question.pop("option_order")
    if qid not in warned:
        warned.add(qid)
        try:
            had = "%d slots" % len(order)
        except TypeError:
            had = repr(order)
        warnings.warn(
            "laya.shortlist: question %r: 'option_order' (%s) was dropped because the label "
            "set was narrowed to %d; the order no longer fits the kept options and remapping "
            "it onto them would move options between slots." % (qid, had, n_options),
            RuntimeWarning,
            stacklevel=2,
        )
    return question


def _criteria_items(criteria):
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise TypeError("choice criteria must be a dict or list, got %s" % type(criteria).__name__)
    if not items:
        raise ValueError("choice criteria must contain at least one option")
    seen = set()
    for key, _value in items:
        if key in seen:
            raise ValueError("choice criteria label %r is duplicated" % (key,))
        seen.add(key)
    return items


def _option_texts(items) -> List[str]:
    crit = {key: value for key, value in items}
    rendered = render_options({"t": "choice", "ins": "", "crit": crit})
    texts = [piece if isinstance(piece, str) else str(piece) for piece in rendered]
    if len(texts) != len(items):
        raise ValueError("could not render every choice option")
    return texts


def _query_text(state, instructions) -> str:
    body = serialize_state(state)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return "%s\n%s" % (instructions, body)


def _subset_criteria(criteria, labels):
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _embeddings(embed_fn, texts: Sequence[str]) -> np.ndarray:
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    raw = embed_fn(list(texts))
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        raise ValueError(
            "embed_fn must return an array of shape (%d, dim), got %s"
            % (len(texts), tuple(arr.shape))
        )
    return np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    qn = float(np.linalg.norm(query))
    if qn == 0.0 or docs.shape[0] == 0:
        return np.zeros(docs.shape[0], dtype=np.float64)
    dn = np.linalg.norm(docs, axis=1)
    denom = dn * qn
    ok = denom > 0.0
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    if np.any(ok):
        sims[ok] = np.clip(np.dot(docs[ok], query) / denom[ok], -1.0, 1.0)
    return sims


def _call_predict(agent, state, questions, **predict_kwargs):
    """One ``predict``/``system_one`` call, checked against the shape every caller here reads.

    Both public functions return that reply to their caller, and ``predict_tournament`` indexes
    into it between rounds, so the check belongs to the call rather than to a call site: an agent
    that answers with a list or ``None`` fails here, naming the contract, instead of raising a
    bare indexing error from inside the module.
    """
    fn = getattr(agent, "predict", None)
    if fn is None:
        fn = getattr(agent, "system_one", None)
    if fn is None:
        raise TypeError("agent must provide predict or system_one")
    result = fn(state, questions, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError("predict/system_one must return a dict, got %s" % type(result).__name__)
    return result


def _round_answers(result):
    """The answers mapping of one elimination round, which this module indexes by group number.

    ``predict_tournament`` numbers its own group questions, so the keys it indexes here are keys the
    caller never wrote. A group a reply leaves out still raises the plain ``KeyError`` for its
    number: ``laya-java/fixtures/tournament.json`` records that as the cross-language contract, for
    the Java and TypeScript ports read the same reply, so naming it here would move three ports
    rather than one module.
    """
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise TypeError(
            "predict/system_one must return an answers dict, got %s" % type(answers).__name__
        )
    return answers


def _hidden_size(encoder) -> int:
    size = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return 0
    return size
