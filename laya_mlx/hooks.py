"""Opt-in prediction hooks: observe or shape every decision without forking.

A hook on `hooks=` is an object implementing any subset of the lifecycle methods on `Hook`;
a plain callable is only accepted on the single-event `on_predict_start=` / `on_predict_end=`
parameters, because `hooks=` reads for method names and a bare function has none. Hooks are
configured on `Agent` / `Router` and can be overridden per call. Everything here is pure
Python: importing `laya` must not start pulling torch.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import math
import numbers
import random
import string
import threading
import time
import uuid
import warnings
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


@dataclass(eq=False)
class PredictContext:
    """Mutable state passed to every hook for one call.

    `states` / `questions` may be rewritten by `on_predict_start`; `results` may be rewritten
    by `on_predict_end`. A start hook can call `skip()` to short-circuit inference with a
    cached result.

    Identity semantics (`eq=False`): two contexts are never equal, and a context is hashable
    by identity, so a hook can keep one in a set without comparing agent/router internals.
    """

    states: List[Any]
    questions: Dict[str, Any]
    run_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )  # shared by every hook of one call
    results: Optional[List[Dict[str, Any]]] = None
    decision: Optional[Dict[str, Any]] = None  # Router: the RouteDecision dict
    model: Optional[str] = None  # resolved checkpoint name
    agent: Any = None
    router: Any = None
    max_len: Optional[int] = None  # per-call overrides; None = agent config
    head_max_len: Optional[int] = None
    usage: Optional[Dict[str, int]] = None  # aggregated input/output tokens
    started_at: float = field(default_factory=time.perf_counter)
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None

    def skip(self, results: List[Dict[str, Any]]) -> None:
        """Set cached results from a start hook; inference is skipped, end hooks still run.

        `results` replaces the whole call, so it carries one entry per state in `ctx.states` --
        the shape `predict_batch` returns -- in that order. A hook fires once per call, and a
        call can carry many states. The one other accepted shape is a single entry for the
        whole call, which is what `predict_long` takes as the document's answer (its scan
        hands the hook every window as a state, and refuses anything but one result).

        The count is checked here, against this contract, so a wrong one fails inside the
        hook under the caller's `hooks_raise` policy instead of downstream: `Router.predict`
        indexed `results[0]` of an empty list (an `IndexError`, which serve maps to 500),
        and `predict_batch` returned a shorter list than it was given states, quietly
        dropping rows the caller was about to zip against.
        """
        states = self.states
        if (
            isinstance(states, (list, tuple))
            and isinstance(results, (list, tuple))
            and len(results) not in (1, len(states))
        ):
            raise ValueError(
                "ctx.skip() takes one result for the whole call or one per state in "
                "ctx.states (%d); got %d" % (len(states), len(results))
            )
        self.results = results


class Hook(Protocol):
    """Optional lifecycle methods. Implement any subset; missing methods are skipped."""

    def on_predict_start(self, ctx: PredictContext) -> None: ...
    def on_predict_end(self, ctx: PredictContext) -> None: ...
    def on_route(self, ctx: PredictContext) -> None: ...
    def on_load(self, ctx: PredictContext) -> None: ...
    def on_evict(self, ctx: PredictContext) -> None: ...
    def on_error(self, ctx: PredictContext) -> None: ...


class BaseHook:
    """No-op base class: subclass it and override only the events you need.

    `Hook` is the structural protocol; `BaseHook` is the concrete convenience when you would
    rather subclass than implement methods by shape. Every method does nothing by default.
    """

    def on_predict_start(self, ctx: PredictContext) -> None:
        pass

    def on_predict_end(self, ctx: PredictContext) -> None:
        pass

    def on_route(self, ctx: PredictContext) -> None:
        pass

    def on_load(self, ctx: PredictContext) -> None:
        pass

    def on_evict(self, ctx: PredictContext) -> None:
        pass

    def on_error(self, ctx: PredictContext) -> None:
        pass


PredictHook = Callable[[PredictContext], None]
HookArg = Union[Hook, Sequence[Hook], None]
PredictHookArg = Union[PredictHook, Sequence[PredictHook], None]

HOOK_EVENTS = (
    "on_predict_start",
    "on_predict_end",
    "on_route",
    "on_load",
    "on_evict",
    "on_error",
)


class _StartAdapter:
    """Wrap a plain `on_predict_start` callable as a `Hook`."""

    __slots__ = ("fn",)

    def __init__(self, fn: PredictHook):
        if not callable(fn):
            raise TypeError("on_predict_start must be callable, got %s" % type(fn).__name__)
        self.fn = fn

    def on_predict_start(self, ctx: PredictContext) -> None:
        return self.fn(ctx)


class _EndAdapter:
    """Wrap a plain `on_predict_end` callable as a `Hook`."""

    __slots__ = ("fn",)

    def __init__(self, fn: PredictHook):
        if not callable(fn):
            raise TypeError("on_predict_end must be callable, got %s" % type(fn).__name__)
        self.fn = fn

    def on_predict_end(self, ctx: PredictContext) -> None:
        return self.fn(ctx)


def _as_sequence(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def normalise_hooks(
    hooks: HookArg = None,
    on_predict_start: PredictHookArg = None,
    on_predict_end: PredictHookArg = None,
) -> List[Any]:
    """Flatten `hooks` and the two convenience callables into one ordered hook list.

    Installed hooks are normalised once at construction; per-call arguments are appended
    after them, so per-call hooks always run last.
    """
    result: List[Any] = []
    for hook in _as_sequence(hooks):
        if isinstance(hook, type):
            raise TypeError(
                "hooks entries must be instances, not classes; got %s. Instantiate it first."
                % hook.__name__
            )
        if not any(hasattr(hook, event) for event in HOOK_EVENTS):
            raise TypeError(
                "hooks entries must implement at least one of %s; got %s. Pass a plain "
                "callable as on_predict_start=/on_predict_end= instead."
                % (", ".join(HOOK_EVENTS), type(hook).__name__)
            )
        for event in HOOK_EVENTS:
            method = getattr(hook, event, None)
            if method is not None and not callable(method):
                raise TypeError(
                    "hooks entry %s.%s must be callable, got %s"
                    % (type(hook).__name__, event, type(method).__name__)
                )
        result.append(hook)
    for fn in _as_sequence(on_predict_start):
        result.append(_StartAdapter(fn))
    for fn in _as_sequence(on_predict_end):
        result.append(_EndAdapter(fn))
    return result


_DEFAULT_HOOKS: List[Any] = []
_DEFAULT_HOOKS_LOCK = threading.Lock()
_SKIP_DEFAULTS = contextvars.ContextVar("laya_skip_default_hooks", default=False)


def default_hooks() -> List[Any]:
    """The process-wide hooks, a copy, in order. Empty unless set via `set_default_hooks`."""
    with _DEFAULT_HOOKS_LOCK:
        return list(_DEFAULT_HOOKS)


def set_default_hooks(hooks=None, on_predict_start=None, on_predict_end=None) -> None:
    """Replace the process-wide default hooks.

    Defaults run before installed and per-call hooks for every `Agent`, `Router` and
    `ONNXAgent` in the process, so a tracer or metrics hook does not have to be threaded
    through every construction. Accepts the same arguments as the `hooks=` parameter.
    """
    normalised = normalise_hooks(hooks, on_predict_start, on_predict_end)
    with _DEFAULT_HOOKS_LOCK:
        _DEFAULT_HOOKS[:] = normalised


def add_default_hook(hook: HookArg) -> None:
    """Append one hook or a sequence to the process-wide defaults."""
    normalised = normalise_hooks(hook)
    with _DEFAULT_HOOKS_LOCK:
        _DEFAULT_HOOKS.extend(normalised)


def clear_default_hooks() -> None:
    """Remove every process-wide default hook."""
    with _DEFAULT_HOOKS_LOCK:
        _DEFAULT_HOOKS.clear()


def compose_hooks(installed, hooks=None, on_predict_start=None, on_predict_end=None) -> List[Any]:
    """Effective hook list for one call: defaults, then installed, then per-call hooks.

    Reads the process-wide defaults at call time, so hooks set after construction still apply.
    """
    defaults = [] if _SKIP_DEFAULTS.get() else default_hooks()
    return defaults + list(installed) + normalise_hooks(hooks, on_predict_start, on_predict_end)


def validate_timeout(value: Optional[float]) -> Optional[float]:
    """Return *value* as a positive float, or ``None`` for no limit.

    A non-positive timeout is rejected here rather than left to
    ``thread.join``: ``join(0)`` and ``join(-1)`` return before the hook has
    started, so the outcome of a fast hook with such a value is a race.
    Non-finite values are rejected too: ``join(nan)`` raises ``ValueError`` and
    ``join(inf)`` raises ``OverflowError`` instead of waiting.
    A bool is rejected as well: ``float(True)`` is ``1.0``, so ``hooks_timeout=True`` would
    otherwise install a one-second deadline on every hook call that the caller never asked for.
    A value ``float()`` cannot read -- a list/object, or a JSON integer too large for a float -- is
    rejected with the same ``ValueError`` rather than leaking ``TypeError``/``OverflowError``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # `False` would be refused below as a non-positive number, which is the wrong reason for a
        # wrong type; `True` was accepted as 1.0. The message names the type so a caller reading it
        # learns what they passed, not just that it was too small. `laya/mcp/tools.py` refuses bools
        # first, so an MCP caller still hears `invalid_hooks_timeout` rather than this ValueError.
        raise ValueError(
            "hooks_timeout must be a positive finite number or None, not a bool; got %r" % (value,)
        )
    try:
        timeout = float(value)
    except (TypeError, OverflowError):
        # A list/object is not a number and a JSON integer larger than a float overflows `float()`;
        # both used to escape as a raw TypeError/OverflowError from the coercion below. `OverflowError`
        # in particular is a valid-JSON number that reached `laya/mcp/tools.py::_validate_hooks_timeout`,
        # which catches only (TypeError, ValueError), so it surfaced as `internal_error`. Refuse both
        # shapes here with the same ValueError as every other bad timeout.
        raise ValueError(
            "hooks_timeout must be a positive finite number or None; got %r" % (value,)
        ) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "hooks_timeout must be a positive finite number or None; got %r" % (value,)
        )
    return timeout


_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_LOOP_LOCK = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop thread, for running coroutines when the caller already has a loop."""
    global _BACKGROUND_LOOP
    with _BACKGROUND_LOOP_LOCK:
        if _BACKGROUND_LOOP is None or not _BACKGROUND_LOOP.is_running():
            _BACKGROUND_LOOP = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_BACKGROUND_LOOP.run_forever, daemon=True, name="laya-async-hooks"
            )
            thread.start()
        return _BACKGROUND_LOOP


def run_coroutine_sync(
    coro: Awaitable[Any], loop: Optional[asyncio.AbstractEventLoop] = None
) -> Any:
    """Run an awaitable to completion from synchronous code.

    Uses `asyncio.run` when the calling thread has no running loop. When it does (a caller
    inside an async function, or a framework that already runs a loop), the coroutine is run on
    a dedicated background loop so the calling thread can block on it without deadlocking. Pass
    `loop` to use a specific loop instead of the background one.

    A supplied `loop` must be running somewhere, and must not be the calling thread's own
    loop. Both are checked: the first would otherwise block forever with no coroutine ever
    scheduled, and the second would block the only thread that could run the coroutine.
    """
    if loop is not None:
        if not loop.is_running():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise ValueError("run_coroutine_sync: the loop passed is not running")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is running:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise ValueError(
                "run_coroutine_sync: the loop passed is running in the calling thread; "
                "blocking on it would deadlock"
            )
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, _background_loop()).result()


class AsyncHook:
    """Wrap an async hook so its coroutine event methods run to completion in the sync core.

    The wrapped object may implement any subset of the events as `async def` methods, or as
    plain callables that return awaitables. Each event is run with `run_coroutine_sync`, so it
    works whether the caller is synchronous or already inside an event loop.

        from laya_mlx import AsyncHook

        class Remote(BaseHook):
            async def on_predict_end(self, ctx):
                await ship(ctx.results)

        agent = laya.load("convaiinnovations/laya", hooks=[AsyncHook(Remote())])

    Pass `loop` to funnel every coroutine onto a specific loop; otherwise a background loop is
    started on demand when the caller already has one.
    """

    __slots__ = ("hook", "loop")

    def __init__(self, hook: Any, loop: Optional[asyncio.AbstractEventLoop] = None):
        if isinstance(hook, type):
            raise TypeError("AsyncHook wraps an instance, not a class; got %s" % hook.__name__)
        if not any(hasattr(hook, event) for event in HOOK_EVENTS):
            raise TypeError(
                "AsyncHook wraps an object implementing at least one of %s; got %s"
                % (", ".join(HOOK_EVENTS), type(hook).__name__)
            )
        self.hook = hook
        self.loop = loop

    def _run(self, event: str, ctx: PredictContext) -> None:
        method = getattr(self.hook, event, None)
        if method is None:
            return
        result = method(ctx)
        if inspect.isawaitable(result):
            run_coroutine_sync(result, loop=self.loop)

    def on_predict_start(self, ctx: PredictContext) -> None:
        self._run("on_predict_start", ctx)

    def on_predict_end(self, ctx: PredictContext) -> None:
        self._run("on_predict_end", ctx)

    def on_route(self, ctx: PredictContext) -> None:
        self._run("on_route", ctx)

    def on_load(self, ctx: PredictContext) -> None:
        self._run("on_load", ctx)

    def on_evict(self, ctx: PredictContext) -> None:
        self._run("on_evict", ctx)

    def on_error(self, ctx: PredictContext) -> None:
        self._run("on_error", ctx)

    def __repr__(self) -> str:
        return "AsyncHook(%r)" % (self.hook,)


class HookRegistry:
    """Mixin giving a runtime-mutable hook list.

    `Agent`, `Router` and `ONNXAgent` use it so hooks can be added, removed or scoped after
    construction. Mutation is guarded by `_hooks_mutex`; a call reads a snapshot of the list,
    so adding or removing a hook never disturbs a call in flight.
    """

    hooks = ()
    _hooks_mutex = None

    def add_hook(self, hook: HookArg) -> "HookRegistry":
        """Install one hook or a sequence of them. Returns self for chaining."""
        self._extend_hooks(normalise_hooks(hook))
        return self

    def remove_hook(self, hook: Any) -> bool:
        """Remove a hook by identity. Returns True if it was installed."""
        return self._remove_hooks(lambda installed: installed is hook) > 0

    @contextmanager
    def hooks_installed(self, *hooks: HookArg):
        """Install hooks for the duration of the `with` block, then remove them.

            with agent.hooks_installed(tracer):
                agent.system_one(state, questions)

        Each argument is one hook or a sequence of them, matching `add_hook` and the
        `hooks=` parameter, so `hooks_installed([tracer, meter])` works too.
        """
        added = normalise_hooks([hook for arg in hooks for hook in _as_sequence(arg)])
        self._extend_hooks(added)
        try:
            yield self
        finally:
            self._remove_instances(added)

    def _remove_instances(self, hooks: Sequence[Any]) -> int:
        """Remove one occurrence of each of `hooks`, the most recent match by identity.

        Removing by identity alone took every copy, so a hook the application had already
        installed was removed along with the block's own and stayed gone -- `docs/hooks/api.md`
        promises the block "restores the previous list on exit". Restoring a snapshot instead is
        wrong in two other ways: with two overlapping blocks the first exit reinstates its
        snapshot and so removes the second block's hook, and a hook added with `add_hook` inside
        the block is discarded because it is not in the snapshot either. Taking one occurrence
        per hook the block added leaves everything else -- including anything added inside the
        block -- in place.

        The most recent match is the one to drop: `_extend_hooks` appends, so a hook that was
        already installed sits earlier in the list than the block's copy.
        """
        if not hooks:
            return 0
        removed = 0
        with self._hooks_mutex_for_registry():
            current = list(self.hooks)
            for hook in hooks:
                for i in range(len(current) - 1, -1, -1):
                    if current[i] is hook:
                        del current[i]
                        removed += 1
                        break
            self.hooks = current
        return removed

    def _extend_hooks(self, hooks: Sequence[Any]) -> None:
        if not hooks:
            return
        with self._hooks_mutex_for_registry():
            self.hooks = list(self.hooks) + list(hooks)

    def _remove_hooks(self, predicate: Callable[[Any], bool]) -> int:
        with self._hooks_mutex_for_registry():
            current = list(self.hooks)
            remaining = [hook for hook in current if not predicate(hook)]
            self.hooks = remaining
            return len(current) - len(remaining)

    def _hooks_mutex_for_registry(self) -> threading.Lock:
        lock = self._hooks_mutex
        if lock is None:
            # Only reachable for an instance built without __init__ (tests); real instances
            # get their mutex in the constructor.
            lock = self._hooks_mutex = threading.Lock()
        return lock


def aggregate_usage(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sum the per-state usage blocks so a hook sees one total for the call."""

    def total(key: str) -> int:
        return sum(int((r.get("usage") or {}).get(key, 0) or 0) for r in results)

    return {"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens")}


def _hook_name(method: Any) -> str:
    return (
        getattr(method, "__qualname__", None) or getattr(method, "__name__", None) or repr(method)
    )


def _call_hook(
    method: Any, ctx: PredictContext, timeout: Optional[float], lock: Any = None
) -> None:
    """Call one hook method, running its result if it is awaitable, under an optional timeout."""
    timeout = validate_timeout(timeout)

    def invoke():
        result = method(ctx)
        if inspect.isawaitable(result):
            run_coroutine_sync(result)

    if timeout is None:
        if lock is None:
            invoke()
        else:
            with lock:
                invoke()
        return

    box: List[BaseException] = []
    deadline = time.monotonic() + timeout

    def timeout_error():
        return TimeoutError("laya: hook %s exceeded %gs" % (_hook_name(method), timeout))

    # Run the hook in a copy of the caller's context, so a `contextvars` value (a request id,
    # a tracing span) set by the caller is visible to the hook even though it runs on another
    # thread. The no-timeout path runs inline and inherits the context already.
    context = contextvars.copy_context()

    def runner():
        acquired = False
        try:
            if lock is not None:
                remaining = deadline - time.monotonic()
                acquired = remaining > 0 and lock.acquire(timeout=remaining)
                if not acquired:
                    raise timeout_error()
            # A timed-out waiter must not run later when the previous hook finally exits.
            if time.monotonic() >= deadline:
                raise timeout_error()
            context.run(invoke)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller thread
            box.append(exc)
        finally:
            if acquired:
                lock.release()

    thread = threading.Thread(target=runner, daemon=True, name="laya-hook-timeout")
    thread.start()
    thread.join(max(0.0, deadline - time.monotonic()))
    if thread.is_alive():
        # The worker owns the lock until the callback actually exits, even after the caller
        # times out. Releasing it on the caller would let subsequent serial hooks overlap.
        raise timeout_error()
    if box:
        raise box[0]


def dispatch(
    hooks: Sequence[Any],
    event: str,
    ctx: PredictContext,
    *,
    raise_errors: bool = True,
    lock: Any = None,
    timeout: Optional[float] = None,
) -> None:
    """Call `event` on every hook that implements it.

    `raise_errors=False` warns and continues, for hooks (telemetry) that must not fail a
    request. `lock` serialises dispatch for hooks that are not safe to run concurrently.
    `timeout` bounds each hook call, including its lock wait; an overrun raises `TimeoutError`, or
    warns when `raise_errors` is False. Python threads cannot be interrupted, so an overrunning
    hook keeps running in the background and retains the serialization lock until it exits.
    A hook that times out waiting for the lock is never invoked. The timeout protects the
    caller's wait, not the process or the mutable context of a hook that already started.
    A hook that returns an awaitable is run to completion before moving on.
    """
    for hook in hooks:
        method = getattr(hook, event, None)
        if method is None:
            continue
        try:
            _call_hook(method, ctx, timeout, lock=lock)
        except Exception as exc:  # noqa: BLE001 -- policy depends on raise_errors
            if raise_errors:
                raise
            warnings.warn(
                "laya: hook %s.%s failed: %s" % (type(hook).__name__, event, exc),
                RuntimeWarning,
                stacklevel=2,
            )


_SEP = "::rel"

ALL_VARIANTS = (
    "reversed",
    "shuffle1",
    "shuffle2",
    "shuffle3",
    "letters",
    "letters_reversed",
    "letters_shuffle",
)

VARIANT_PRESETS = {
    "full": None,
    "fast": ("reversed", "letters_reversed"),
}


def make_variants(
    labels: Sequence[str],
    seed_key: str,
    n_shuffles: int = 3,
    relabel: bool = True,
    include: Optional[Sequence[str]] = None,
    reorder: bool = True,
) -> List[Tuple[str, List[str], bool]]:
    """Meaning-preserving rewrites of choice option order and labels (#635).

    The first variant is always the original question. Shuffles are seeded using a
    zlib CRC32 hash of `seed_key` so variant selection is reproducible across identical
    inputs. Note: this provides reproducible variant generation, not a model-level
    determinism guarantee. Duplicates are dropped because evaluating an identical
    sequence twice would artificially inflate stability.
    """
    labels_list = list(labels)
    rng = random.Random(zlib.crc32(seed_key.encode("utf-8")))
    cands: List[Tuple[str, List[str], bool]] = [
        ("original", labels_list, False),
    ]
    if reorder:
        cands.append(("reversed", labels_list[::-1], False))
        for i in range(n_shuffles):
            order = labels_list[:]
            rng.shuffle(order)
            cands.append((f"shuffle{i + 1}", order, False))
    if relabel and len(labels_list) <= 26:
        cands.append(("letters", labels_list, True))
        cands.append(("letters_reversed", labels_list[::-1], True))
        order = labels_list[:]
        rng.shuffle(order)
        cands.append(("letters_shuffle", order, True))

    if include is not None:
        wanted = set(include)
        cands = [c for c in cands if c[0] == "original" or c[0] in wanted]

    seen = set()
    out = []
    for name, order, rel in cands:
        key = (tuple(order), rel)
        if key not in seen:
            seen.add(key)
            out.append((name, order, rel))
    return out


def decide_reliability(
    soft_stability: float,
    accept: Optional[float],
    escalate: Optional[float] = None,
) -> Optional[str]:
    """Classify soft stability into 'ACCEPT', 'VERIFY', or 'ESCALATE'.

    If `accept` is None, returns None (fail-closed uncalibrated signal).
    """
    if accept is None:
        return None
    if soft_stability >= accept:
        return "ACCEPT"
    if escalate is not None and soft_stability < escalate:
        return "ESCALATE"
    return "VERIFY"


def calibrate_reliability(
    soft_scores: Sequence[float],
    correct: Sequence[bool],
    target_accuracy: float = 0.90,
    max_escalate_accuracy: float = 1 / 3,
) -> Tuple[float, float]:
    """Pick (accept, escalate) thresholds from labelled evaluation examples (#635).

    accept   = lowest cutoff where answers at or above it reach `target_accuracy`.
    escalate = highest cutoff below `accept` where answers below it are at most
               `max_escalate_accuracy` accurate.
    If no cutoff reaches `target_accuracy`, accept is returned as 1.01 (accept nothing).
    Cutoffs use the supplied scores without rounding, matching the hook's reported
    `soft_stability` and runtime comparison. Equal scores are evaluated together.

    Scores and correctness labels must have equal, nonzero lengths. Scores and
    accuracy targets must be finite real numbers in [0, 1], excluding bool;
    invalid lengths or values raise ValueError. Correctness labels must be
    booleans or TypeError is raised. Invalid evidence is never filtered or coerced.

    "Real number" and "boolean" mean the NumPy scalars too, because `correct` is
    normally computed, not written out: `preds == gold` on a NumPy or pandas column
    yields `np.bool_`, which is a boolean by every meaning except `isinstance(x, bool)`.
    `numbers.Real` covers `np.float32` and `np.int64` and still excludes `np.bool_`,
    so the two checks stay disjoint. `soft_scores` and `correct` must be sized; a
    generator raises TypeError, since a length mismatch has to be detectable.
    """
    if len(soft_scores) != len(correct):
        raise ValueError("soft_scores and correct must have the same length")
    if len(soft_scores) == 0:
        raise ValueError("cannot calibrate reliability on empty scores/correct pairs")
    for name, value in (
        ("target_accuracy", target_accuracy),
        ("max_escalate_accuracy", max_escalate_accuracy),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real)
            or not (0.0 <= value <= 1.0)
            or not math.isfinite(value)
        ):
            raise ValueError("%s must be a finite number in [0, 1], got %r" % (name, value))
    for i, score in enumerate(soft_scores):
        if (
            isinstance(score, (bool, np.bool_))
            or not isinstance(score, numbers.Real)
            or not (0.0 <= score <= 1.0)
            or not math.isfinite(score)
        ):
            raise ValueError(
                "soft_scores[%d] must be a finite number in [0, 1], got %r" % (i, score)
            )
    for i, label in enumerate(correct):
        if not isinstance(label, (bool, np.bool_)):
            raise TypeError("correct[%d] must be a bool, got %r" % (i, label))
    pairs = sorted(zip(soft_scores, correct))
    cuts = sorted({s for s, _ in pairs})

    accept = 1.01
    for t in cuts:
        kept = [c for s, c in pairs if s >= t]
        if kept and sum(kept) / len(kept) >= target_accuracy:
            accept = t
            break

    escalate = 0.0
    for t in cuts:
        if t >= accept:
            break
        below = [c for s, c in pairs if s < t]
        if below and sum(below) / len(below) <= max_escalate_accuracy:
            escalate = t
    # The cutoffs are thresholds a caller stores and replays, so hand back plain floats: a
    # `np.float32` score would otherwise travel into a JSON report as a NumPy scalar.
    return float(accept), float(escalate)


def _seed_of(state: Any) -> str:
    return state if isinstance(state, str) else repr(state)


def _has_semantic_description(val: Any) -> bool:
    if val is None or val == "":
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _rendered_options(criteria: Any) -> Any:
    """Order-sensitive identity of the options a choice question renders.

    A dict is compared by its (key, description) sequence because option order is what a
    reorder probe changes; a sequence is compared as given. Two questions with the same
    identity are the same probe -- scoring both would only re-weight one rendering (#1493).
    """
    if isinstance(criteria, dict):
        return tuple(criteria.items())
    if isinstance(criteria, (list, tuple)):
        return tuple(criteria)
    return criteria


class OptionStabilityHook(BaseHook):
    """Opt-in reliability signal via metamorphic option reordering and renaming (#635).

    Generates variant permutations of choice options at `on_predict_start`, scores them
    in the same batch forward pass, and adds a `reliability` dict to each choice answer
    at `on_predict_end`.

    Thresholds `accept` and `escalate` are required keyword-only arguments. A defaulted
    threshold reads as an engine recommendation, but optimal cutoffs depend on domain
    error tolerance, option count, and calibration temperatures (Issue #1245). Pass
    `accept=None, escalate=None` to compute continuous stability metrics without categorical
    triage decisions.

    Layout-aware (#635, #951):
    On parallel option layouts (`option_layout="parallel"`), options cannot attend to each
    other so reordering yields 100% agreement by construction. The hook detects the
    checkpoint's layout from `ctx.agent.config` and runs rename-only probes on parallel
    checkpoints, while running both rename and reorder probes on sequential checkpoints.
    The returned `reliability["probes"]` explicitly names which probes actually ran,
    returning `None` for stability signals when no probe could run.
    """

    def __init__(
        self,
        *,
        accept: Optional[float],
        escalate: Optional[float],
        probes: Optional[Union[str, Sequence[str]]] = None,
        n_shuffles: int = 3,
        relabel: bool = True,
        rename_max_options: int = 20,
        variants: Union[str, Sequence[str], None] = "full",
        seed: Optional[str] = None,
    ):
        if probes is not None:
            if isinstance(probes, str):
                if probes == "auto":
                    self.probes = None
                elif probes in ("reorder", "rename"):
                    self.probes = (probes,)
                elif probes == "all":
                    self.probes = ("reorder", "rename")
                else:
                    raise ValueError(
                        "unknown probe type %r; choose from 'auto', 'reorder', 'rename', 'all'"
                        % (probes,)
                    )
            elif isinstance(probes, (list, tuple, set)):
                probes_list = list(probes)
                if not probes_list:
                    raise ValueError(
                        "probes list must not be empty; choose from 'reorder', 'rename'"
                    )
                bad = [p for p in probes_list if p not in ("reorder", "rename")]
                if bad:
                    raise ValueError(
                        "unknown probe types %r; choose from 'reorder', 'rename'" % (bad,)
                    )
                self.probes = tuple(dict.fromkeys(probes_list))
            else:
                raise TypeError(
                    "probes must be a string or sequence of probe names, got %s"
                    % type(probes).__name__
                )
        else:
            self.probes = None

        if isinstance(n_shuffles, bool) or not isinstance(n_shuffles, int) or n_shuffles < 0:
            raise ValueError("n_shuffles must be a non-negative integer, got %r" % (n_shuffles,))
        if not isinstance(relabel, bool):
            raise TypeError("relabel must be a bool, got %r" % (relabel,))
        if (
            isinstance(rename_max_options, bool)
            or not isinstance(rename_max_options, int)
            or rename_max_options < 2
        ):
            raise ValueError(
                "rename_max_options must be an integer >= 2, got %r" % (rename_max_options,)
            )

        self.n_shuffles = n_shuffles
        self.relabel = relabel
        self.rename_max_options = rename_max_options
        self.variants = variants
        self._include = self._resolve_variants(variants)

        if accept is not None:
            if (
                isinstance(accept, bool)
                or not isinstance(accept, (int, float))
                or not math.isfinite(accept)
                or not (0.0 <= accept <= 1.01)
            ):
                raise ValueError("accept must be a float in [0.0, 1.01], got %r" % (accept,))
            self.accept = float(accept)
        else:
            self.accept = None

        if escalate is not None:
            if (
                isinstance(escalate, bool)
                or not isinstance(escalate, (int, float))
                or not math.isfinite(escalate)
                or not (0.0 <= escalate <= 1.0)
            ):
                raise ValueError("escalate must be a float in [0.0, 1.0], got %r" % (escalate,))
            self.escalate = float(escalate)
        else:
            self.escalate = None

        if self.accept is not None and self.escalate is not None and self.escalate > self.accept:
            raise ValueError(
                "escalate (%r) must not be greater than accept (%r)" % (escalate, accept)
            )

        self.seed = str(seed) if seed is not None else None

        self._plans: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_variants(variants: Union[str, Sequence[str], None]) -> Optional[List[str]]:
        if variants is None:
            return None
        if isinstance(variants, str):
            if variants not in VARIANT_PRESETS:
                raise ValueError(
                    "unknown variants preset %r; use one of %s or a sequence from %s"
                    % (variants, sorted(VARIANT_PRESETS), list(ALL_VARIANTS))
                )
            preset = VARIANT_PRESETS[variants]
            return list(preset) if preset is not None else None
        if isinstance(variants, (list, tuple, set)):
            names = list(variants)
            bad = [n for n in names if n not in ALL_VARIANTS]
            if bad:
                raise ValueError(
                    "unknown variant names %r; choose from %s" % (bad, list(ALL_VARIANTS))
                )
            if not names:
                raise ValueError("variants list is empty: nothing to compare the original against")
            return names
        raise TypeError(
            "variants must be a string preset or sequence of names, got %s"
            % type(variants).__name__
        )

    @staticmethod
    def _detect_layout(ctx: PredictContext) -> str:
        agent = getattr(ctx, "agent", None)
        if agent is not None:
            cfg = getattr(agent, "config", None)
            if isinstance(cfg, dict):
                raw = cfg.get("option_layout")
                if isinstance(raw, str):
                    val = raw.strip().lower()
                    if val in ("parallel", "sequential"):
                        return val
                    warnings.warn(
                        "laya: unknown option_layout %r in agent config; defaulting to 'sequential'"
                        % (raw,),
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return "sequential"

    @staticmethod
    def _variant_question(
        q: Dict[str, Any], order: List[str], relabel: bool
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        shown = list(string.ascii_uppercase[: len(order)]) if relabel else list(order)
        new_q = dict(q)
        crit = q.get("criteria")
        if isinstance(crit, dict):
            if relabel:
                # If an option has no semantic description (None, empty, or whitespace),
                # preserve the original label as the description so the model still receives
                # the semantic text under letter renaming (#1285).
                new_q["criteria"] = {
                    s: (crit[orig] if _has_semantic_description(crit[orig]) else orig)
                    for s, orig in zip(shown, order)
                }
            else:
                new_q["criteria"] = {s: crit[orig] for s, orig in zip(shown, order)}
        elif isinstance(crit, (list, tuple)):
            if relabel:
                new_q["criteria"] = {s: orig for s, orig in zip(shown, order)}
            else:
                new_q["criteria"] = list(order)
        return new_q, dict(zip(shown, order))

    def on_predict_start(self, ctx: PredictContext) -> None:
        if not ctx.questions:
            return

        for qid in ctx.questions:
            if _SEP in qid:
                raise ValueError("question id %r must not contain %r" % (qid, _SEP))

        layout = self._detect_layout(ctx)
        if self.probes is not None:
            configured_probes = set(self.probes)
        else:
            # Layout-aware probe selection (#635, #951):
            # Parallel layout makes reordering structurally invariant (100% agreement by construction).
            # Therefore reorder probes are pure cost and skipped; rename probes remain informative.
            if layout == "parallel":
                configured_probes = {"rename"} if self.relabel else set()
            else:
                configured_probes = {"reorder"}
                if self.relabel:
                    configured_probes.add("rename")

        base_seed = (
            self.seed
            if self.seed is not None
            else (_seed_of(ctx.states[0]) if (ctx.states and len(ctx.states) == 1) else "batch")
        )

        plan: Dict[str, Dict[str, Any]] = {}
        new_questions = dict(ctx.questions)

        for qid, q in ctx.questions.items():
            if not isinstance(q, dict) or q.get("type") != "choice":
                continue
            crit = q.get("criteria")
            if isinstance(crit, dict):
                labels = list(crit.keys())
            elif isinstance(crit, (list, tuple)):
                labels = list(crit)
            else:
                continue
            if len(labels) < 2:
                # Questions with < 2 options cannot be probed
                plan[qid] = {
                    "layout": layout,
                    "variants": [("original", qid, {k: k for k in labels}, False)],
                    "probes_ran": [],
                }
                continue

            can_reorder = "reorder" in configured_probes
            can_rename = ("rename" in configured_probes) and (
                len(labels) <= self.rename_max_options
            )

            if not can_reorder and not can_rename:
                plan[qid] = {
                    "layout": layout,
                    "variants": [("original", qid, {k: k for k in labels}, False)],
                    "probes_ran": [],
                }
                continue

            seed_key = f"{base_seed}|{qid}"
            variants = make_variants(
                labels,
                seed_key,
                self.n_shuffles,
                relabel=can_rename,
                include=self._include,
                reorder=can_reorder,
            )

            q_variants = [("original", qid, {k: k for k in labels}, False)]
            rendered = [_rendered_options(q.get("criteria"))]

            for i, (name, order, rel) in enumerate(variants[1:], 1):
                vq, back = self._variant_question(q, order, rel)
                # A rename probe can render exactly the question the model already saw:
                # when the option keys are already A, B, ... the relabelling maps every
                # label onto itself. Scoring it again duplicates the original's weight in
                # `soft_stability` (#1493), so drop it the way make_variants drops
                # duplicate (order, relabel) candidates.
                options = _rendered_options(vq.get("criteria"))
                if options in rendered:
                    continue
                rendered.append(options)
                key = f"{qid}{_SEP}{i}"
                new_questions[key] = vq
                q_variants.append((name, key, back, rel))

            probes_ran = []
            if any(not rel for _, _, _, rel in q_variants[1:]):
                probes_ran.append("reorder")
            if any(rel for _, _, _, rel in q_variants[1:]):
                probes_ran.append("rename")

            plan[qid] = {
                "layout": layout,
                "variants": q_variants,
                "probes_ran": probes_ran,
            }

        if plan:
            ctx.questions = new_questions
            with self._lock:
                self._plans[ctx.run_id] = plan

    def on_predict_end(self, ctx: PredictContext) -> None:
        with self._lock:
            plan = self._plans.pop(ctx.run_id, None)
        if not plan or not ctx.results:
            return

        for res in ctx.results:
            if not isinstance(res, dict):
                continue
            answers = res.get("answers")
            if not isinstance(answers, dict):
                continue

            all_variant_keys = set()
            for qid, q_plan in plan.items():
                variants = q_plan["variants"]
                layout = q_plan["layout"]
                probes_ran = q_plan["probes_ran"]

                for _, key, _, _ in variants[1:]:
                    all_variant_keys.add(key)

                if qid not in answers:
                    continue
                ans = answers[qid]
                if not isinstance(ans, dict) or ans.get("type") != "choice":
                    continue

                choice = ans.get("choice")
                seen = []
                all_present = all(key in answers for _, key, _, _ in variants)
                if not all_present:
                    continue

                for name, key, back, rel in variants:
                    a = answers[key]
                    raw_probs = a.get("probabilities") or {}
                    probs = {back.get(k, k): v for k, v in raw_probs.items()}
                    v_choice = a.get("choice")
                    mapped_choice = back.get(v_choice, v_choice)
                    seen.append((name, mapped_choice, probs, rel))

                others = seen[1:]
                if not others or not probes_ran:
                    # No probes ran for this question: return None signals so caller
                    # does not gate on an unprobed question (#635).
                    ans["reliability"] = {
                        "probes": [],
                        "option_layout": layout,
                        "decision": None,
                        "soft_stability": None,
                        "stability": None,
                        "rename_stability": None,
                        "reorder_stability": None,
                        "n_variants": len(seen),
                        "distinct_choices": sorted({c for _, c, _, _ in seen if c is not None}),
                        "variant_choices": {name: c for name, c, _, _ in seen},
                        "variant_support": {
                            name: round(p.get(choice, 0.0), 4) for name, _, p, _ in seen
                        },
                    }
                    continue

                reorder_items = [(name, c, p) for name, c, p, rel in others if not rel]
                rename_items = [(name, c, p) for name, c, p, rel in others if rel]

                stability = (
                    (sum(c == choice for _, c, _, _ in others) / len(others)) if others else None
                )
                soft = (
                    (sum(p.get(choice, 0.0) for _, _, p, _ in seen) / len(seen)) if seen else None
                )
                reorder_stab = (
                    (sum(c == choice for _, c, _ in reorder_items) / len(reorder_items))
                    if reorder_items
                    else None
                )
                rename_stab = (
                    (sum(c == choice for _, c, _ in rename_items) / len(rename_items))
                    if rename_items
                    else None
                )

                ans["reliability"] = {
                    "probes": list(probes_ran),
                    "option_layout": layout,
                    "decision": decide_reliability(soft, self.accept, self.escalate)
                    if soft is not None
                    else None,
                    "soft_stability": soft,
                    "stability": round(stability, 4) if stability is not None else None,
                    "rename_stability": round(rename_stab, 4) if rename_stab is not None else None,
                    "reorder_stability": round(reorder_stab, 4)
                    if reorder_stab is not None
                    else None,
                    "n_variants": len(seen),
                    "distinct_choices": sorted({c for _, c, _, _ in seen if c is not None}),
                    "variant_choices": {name: c for name, c, _, _ in seen},
                    "variant_support": {
                        name: round(p.get(choice, 0.0), 4) for name, _, p, _ in seen
                    },
                }

            for key in all_variant_keys:
                answers.pop(key, None)

            usage = res.get("usage")
            if isinstance(usage, dict):
                if "truncated_questions" in usage and isinstance(
                    usage["truncated_questions"], list
                ):
                    usage["truncated_questions"] = [
                        q for q in usage["truncated_questions"] if _SEP not in q
                    ]
                if "options" in usage and isinstance(usage["options"], dict):
                    usage["options"] = {k: v for k, v in usage["options"].items() if _SEP not in k}

    def on_error(self, ctx: PredictContext) -> None:
        with self._lock:
            self._plans.pop(ctx.run_id, None)
