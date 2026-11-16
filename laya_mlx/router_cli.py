"""Command-line interface for testing Laya locally.

    laya "I was charged twice, please refund"           # routing decision only (no model download)
    laya "Refactor this service" --predict              # full answers, loads the checkpoint
    laya                                                # interactive mode
    laya "Mein Konto wurde zweimal belastet" --lang de  # explicit language
    laya "My payment failed twice" --lang-guess en      # a soft hint that still lets routing decide
    laya "My payment failed twice" --model ml           # pin a checkpoint, by name or alias
    laya "My payment failed twice" --preset triage      # a ready-made question preset
    laya --batch tickets.txt --predict                  # a file of requests, one per line
    cat tickets.txt | laya --batch - --predict --json   # stdin, one JSON line of answers each
    laya "Where is my card" --questions intents.json    # your own questions, from a JSON file
    laya "Where is my card" --questions intents.json --head-max-len 384
    laya "Where is my card" --questions intents.json --min-confidence 0.7

Routing (the default) never downloads a checkpoint, so it works offline and
returns in milliseconds. --predict loads the routed checkpoint on first use,
which needs network access to the Hugging Face hub. --preset answers one of the
ready-made question presets from laya_mlx.presets and implies --predict.
--batch scores a whole file (or stdin, with `-`) through Router.predict_batch in one
process, so the states share checkpoint loads and forward passes; --json prints JSONL.
--questions answers the question set in a JSON file instead; --max-len and --head-max-len
raise its token budget for the request, which is what a many-label question needs,
and --min-confidence marks an answer whose confidence falls below the threshold
instead of changing it (it needs an answering mode: routing has no confidence to gate).
"""

import argparse
import inspect
import json
import sys

import laya_mlx as laya

# Ready-made question presets a prediction can run instead of router_questions().
PRESETS = {
    "email": laya.email_questions,
    "guard": laya.guard_questions,
    "moderation": laya.moderation_questions,
    "router": laya.router_questions,
    "triage": laya.triage_questions,
}

# The state field each preset's instructions name, so the CLI puts the text where that question
# set reads it. `router` is also the default for `--predict`, which answers
# `laya.router_questions()`. Kept beside PRESETS so a new preset and its key arrive together.
PRESET_STATE_KEYS = {
    "email": "body",
    "guard": "prompt",
    "moderation": "post",
    "router": "request",
    "triage": "message",
}


def model_name(value):
    """Resolve a `--model` argument the way core resolves it: same names, same aliases, same casing.

    ``argparse``'s ``choices=`` compared strings exactly, so ``--model en`` was rejected with
    "invalid choice" even though the call this flag feeds -- ``router.predict(model=...)`` --
    has always accepted it. ``laya.router.normalise_name`` is the one registry of checkpoints and
    aliases, so delegating here cannot fall behind it, and the name that reaches the router is
    already canonical: what a caller prints is the checkpoint, not its spelling. ``auto`` means
    "do not pin one", which is what omitting the flag means, so it maps to ``None``.
    """
    if value.strip().lower() == "auto":
        return None
    from laya_mlx.router import normalise_name

    try:
        return normalise_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("%s, or 'auto'" % error) from None


def min_confidence_value(value):
    """Resolve a `--min-confidence` argument through core's own validator, so the range is core's.

    ``laya.confidence.check_min_confidence`` is what ``Router.predict`` and ``Router.predict_batch``
    run this argument through, so a threshold this flag accepts is exactly one the call accepts --
    including the boolean and the infinite value that ``float()`` alone waves through
    (``--min-confidence inf`` would ask for a gate no answer can clear).
    """
    from laya_mlx.confidence import check_min_confidence

    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected a number in [0.0, 1.0], got %r" % value
        ) from None
    try:
        return check_min_confidence(number)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def build_parser():
    parser = argparse.ArgumentParser(
        prog="laya",
        description="Test Laya locally: route or answer a request from the command line.",
    )
    parser.add_argument("text", nargs="*", help="the request text (omit for interactive mode)")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="run the full prediction, not just the routing decision (downloads the checkpoint on first use)",
    )
    parser.add_argument(
        "--model",
        type=model_name,
        metavar="NAME",
        help="force a checkpoint instead of auto-routing: a checkpoint name or any of "
        "core's aliases, in any casing, or 'auto' to route it (the default)",
    )
    parser.add_argument("--lang", help="force a language, e.g. en or de, instead of detecting it")
    parser.add_argument(
        "--lang-guess",
        dest="lang_guess",
        metavar="CODE",
        help="a soft language hint that participates in routing instead of skipping "
        "detection: checked after --lang and before the built-in detector, so a "
        "probable-but-uncertain code can nudge the checkpoint without forcing it",
    )
    parser.add_argument("--task", help="force a typed-decisions workflow instead of detecting it")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        metavar="NAME",
        help="answer a ready-made question preset (%s) instead of the router questions; implies --predict"
        % ", ".join(sorted(PRESETS)),
    )
    parser.add_argument(
        "--questions",
        metavar="FILE",
        help="a JSON file of your own typed questions to answer instead of the router questions"
        " or a preset; implies --predict",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        dest="max_len",
        metavar="N",
        help="token budget for the whole request (state plus options); defaults to the"
        " checkpoint's own budget",
    )
    parser.add_argument(
        "--head-max-len",
        type=int,
        dest="head_max_len",
        metavar="N",
        help="token budget the choice options share; raise it when a question has many"
        " labels, so each keeps enough tokens to stay distinct",
    )
    parser.add_argument(
        "--min-confidence",
        type=min_confidence_value,
        dest="min_confidence",
        metavar="P",
        help="flag an answer whose calibrated confidence falls below P (0.0-1.0) as"
        " low-confidence; the answer itself is kept, so this asks for a warning"
        " rather than a refusal. Needs --predict, --preset or --questions.",
    )
    parser.add_argument(
        "--image",
        metavar="FILE",
        help="answer the questions about a picture instead of a text: routes to the vega lane "
        "(needs vegaml, which ships with laya); implies --predict",
    )
    parser.add_argument(
        "--audio",
        metavar="FILE",
        help="answer the questions about a recording (wav, flac, ogg, mp3): transcribed locally, "
        "then decided on the vega lane; implies --predict",
    )
    parser.add_argument(
        "--video",
        metavar="FILE",
        help="answer the questions about a clip: its speech if it has any, else sampled frames, "
        "on the vega lane; implies --predict",
    )
    parser.add_argument("--device", help="MLX device, e.g. cpu, gpu or mps")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="include language-detection reasons and sources in routing mode (no model loading)",
    )
    parser.add_argument(
        "--batch",
        metavar="FILE",
        help="score a file of requests, one per line (use '-' for stdin), instead "
        "of a single text; implies neither --predict nor --preset, but the "
        "same modes apply: routing by default, answers with --predict/--preset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="states per forward pass in --batch mode; the default sends each "
        "routed group in one pass",
    )
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        dest="sort_by_length",
        help="with --batch-size N where 1 < N < the number of requests, group "
        "similarly sized states into the same forward pass so each pads to a "
        "shorter maximum; fewer padded tokens, same answers in input order",
    )
    return parser


def make_router(args):
    return laya.Router(device=args.device, preload=False)


def load_questions(path):
    """Read a ``--questions`` file: returns ``(questions, state_key)``.

    The file is either a bare ``question id -> definition`` mapping, or an object holding one
    under ``"questions"`` plus an optional ``"state_key"`` naming the field the instructions
    refer to. That second field exists because #426 found the CLI sending every request under a
    key no question set named: with a user's own questions the CLI cannot guess the key, so the
    file declares it. Default ``"request"``, matching bare ``--predict``.
    """
    try:
        # `utf-8-sig`, not `utf-8`: a questions file written by Notepad or Excel starts with a
        # UTF-8 BOM, and a plain `utf-8` read leaves it as U+FEFF, so `json.load` refuses a file
        # that is valid JSON -- with its own "decode using utf-8-sig" hint. The other file readers
        # in this repo open the same way for the same reason (docker/entrypoint.py,
        # laya/train.py, scripts/http_inference_smoke.py).
        with open(path, encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise ValueError("no such --questions file: %s" % path)
    except OSError as error:
        # A path that exists but cannot be read as a file -- a directory, or one this process may
        # not open -- is the same class of fault as the three below: it is about *this* file, not
        # about a missing dependency or a failed checkpoint download. Letting the raw OSError reach
        # run()'s `except (ImportError, OSError, RuntimeError)` printed "could not run Laya" and
        # advised checking the network, which is a false diagnosis for a wrong --questions path.
        raise ValueError("--questions file %s cannot be read: %s" % (path, error)) from error
    # The same three refusals `laya/evidence.py::_read_json_object` makes: a bare JSONDecodeError or
    # UnicodeDecodeError says what is wrong with the text but not which file, and a too-deep document
    # surfaces as a RecursionError that the caller's `except (OSError, RuntimeError, ...)` misreads as
    # a dependency/download problem. Prefix each with the path so it is a usage error about this file.
    except json.JSONDecodeError as error:
        raise ValueError("--questions file %s is not valid JSON: %s" % (path, error)) from error
    except UnicodeDecodeError as error:
        raise ValueError("--questions file %s is not UTF-8 text: %s" % (path, error)) from error
    except RecursionError as error:
        raise ValueError(
            "--questions file %s is JSON nested deeper than it can be read: %s" % (path, error)
        ) from error
    if not isinstance(raw, dict):
        raise ValueError("--questions file must be a JSON object of question id -> definition")
    state_key = "request"
    questions = raw
    wrapper_keys = {"questions", "state_key"}
    is_wrapped = False
    if set(raw.keys()) <= wrapper_keys and "questions" in raw:
        inner = raw["questions"]
        if "state_key" in raw or not isinstance(inner, dict) or not inner:
            is_wrapped = True
        elif all(isinstance(v, dict) for v in inner.values()):
            is_wrapped = True
    if is_wrapped:
        if not isinstance(raw["questions"], dict):
            raise ValueError("the 'questions' field of the --questions file must be an object")
        questions = raw["questions"]
        state_key = raw.get("state_key", "request")
        if not isinstance(state_key, str) or not state_key.strip():
            raise ValueError("'state_key' must be a non-empty string, got %r" % (state_key,))
    if not questions:
        raise ValueError("--questions file holds no questions")
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict):
            raise ValueError(
                "question %r must map to an object, got %s" % (qid, type(qdef).__name__)
            )
    return questions, state_key


def show_decision(decision):
    print("Model     :", decision["model"])
    print("Reason    :", decision["reason"])
    detection = decision.get("detection")
    if detection:
        print("Detected  :", json.dumps(detection, ensure_ascii=False))


def show_answers(result):
    routing = result.get("routing")
    if routing:
        show_decision(routing)
        print()
    for qid, answer in result.get("answers", {}).items():
        if "choice" in answer:
            choice = answer["choice"]
            probability = answer.get("probabilities", {}).get(choice, 0.0)
            detail = "%s (p=%.3f)" % (choice, probability)
        elif "score" in answer:
            detail = "%.2f" % answer["score"]
        elif "noul" in answer:
            detail = "%.3f" % answer["noul"]
        else:
            detail = json.dumps(answer, ensure_ascii=False)
        # The gate marks an unsure answer and keeps it; without this line the mark was only ever
        # readable through --json, so a gated run at a terminal looked exactly like an ungated one.
        if answer.get("low_confidence"):
            detail += "  [low-confidence]"
        print("%-12s: %s" % (qid, detail))


def budget_overrides(args):
    """The token-budget flags as `predict` keyword arguments, absent when unset.

    Unset has to mean unsent, so the checkpoint's own defaults stay in charge. A zero is a real
    budget, not an absence, so this tests `is not None` rather than truthiness.
    """
    overrides = {}
    for flag in ("max_len", "head_max_len"):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[flag] = value
    return overrides


def abstention_override(args):
    """`--min-confidence` as a `predict` / `predict_batch` keyword argument, absent when unset.

    Separate from ``budget_overrides`` because the two reach the batch call differently: a token
    budget is per request, so it goes inside each request dict, while ``predict_batch`` validates
    one threshold for the whole batch and takes it as its own argument. Unset has to stay unsent
    for the same reason as the budgets -- a ``None`` here would overwrite a Router built with its
    own ``min_confidence``, and ``0.0`` is a real threshold, not an absence.
    """
    value = getattr(args, "min_confidence", None)
    return {} if value is None else {"min_confidence": value}


def resolve_questions(args):
    """The question set to answer and the state field it reads, from the flags given."""
    if args.questions and args.preset:
        raise ValueError("--questions and --preset choose different question sets; pass one")
    if args.questions:
        return load_questions(args.questions)
    if args.preset:
        return PRESETS[args.preset](), PRESET_STATE_KEYS[args.preset]
    return laya.router_questions(), PRESET_STATE_KEYS["router"]


def media_override(args):
    """The three media arguments of `Router.predict`, as the flags gave them: at most one set,
    the others None, so every control the parser names reaches the call."""
    given = {key: (getattr(args, key, None) or None) for key in ("image", "audio", "video")}
    if sum(1 for value in given.values() if value) > 1:
        raise ValueError(
            "pass one of --image, --audio, --video, not %s"
            % " and ".join("--" + k for k, v in given.items() if v)
        )
    return given


def run(text, args, router=None):
    """Route or predict one request; returns 0 on success, 2 on a handled error."""
    router = router or make_router(args)
    try:
        media = media_override(args)
        if any(media.values()):
            questions, _ = resolve_questions(args)
            result = router.predict(
                None,
                questions,
                model=args.model,
                task=args.task,
                lang=args.lang,
                lang_guess=args.lang_guess,
                **media,
                **budget_overrides(args),
                **abstention_override(args),
            )
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                show_answers(result)
            return 0
        if args.predict or args.preset or args.questions:
            questions, state_key = resolve_questions(args)
            # Each preset's instructions name the field they read -- `` `message` ``,
            # `` `body` ``, `` `prompt` ``, `` `post` ``, `` `request` `` -- and the CLI used to
            # send every request as `{"text": ...}`, a key none of them names, so the model was
            # asked about a field that was not there. The state key therefore follows the
            # question set being answered: the preset's own key, or the one a --questions file
            # declares. Routing is not affected either way: `route` reads the state only for
            # language detection, which is key-invariant, so the default path keeps
            # `{"text": ...}`.
            state = {state_key: text}
            result = router.predict(
                state,
                questions,
                model=args.model,
                task=args.task,
                lang=args.lang,
                lang_guess=args.lang_guess,
                **media,
                **budget_overrides(args),
                **abstention_override(args),
            )
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                show_answers(result)
        else:
            state = {"text": text}
            decision = router.route(
                state,
                model=args.model,
                task=args.task,
                lang=args.lang,
                lang_guess=args.lang_guess,
                **({"explain": True} if args.explain else {}),
            )
            if args.json:
                print(json.dumps(dict(decision), ensure_ascii=False, indent=2, default=str))
            else:
                show_decision(decision)
    except ValueError as error:
        print("laya: %s" % error, file=sys.stderr)
        return 2
    except (ImportError, OSError, RuntimeError) as error:
        print("laya: could not run Laya (%s)." % error, file=sys.stderr)
        print(
            "Check that the dependencies are installed and the checkpoints can be "
            "downloaded from the Hugging Face hub (network access is needed on first use).",
            file=sys.stderr,
        )
        return 2
    return 0


def read_batch_lines(source):
    """One request per non-blank line of FILE, or of stdin for '-'."""
    if source == "-":
        # A pipe is decoded with the locale's codec by default (cp1252 on Windows), not the utf-8 a
        # FILE is read with, so piped non-Latin requests arrived as mojibake.
        if hasattr(sys.stdin, "reconfigure"):
            # `utf-8-sig` for the same reason as FILE below: a producer that wrote the stream with
            # a Windows editor prefixes U+FEFF, which would otherwise become the first request's
            # first character. It reads plain UTF-8 unchanged when there is no BOM.
            sys.stdin.reconfigure(encoding="utf-8-sig")
        lines = sys.stdin.read().splitlines()
    else:
        # `utf-8-sig`: a batch file saved by Notepad or Excel starts with a UTF-8 BOM, which plain
        # `utf-8` would keep as U+FEFF at the front of the first request.
        with open(source, encoding="utf-8-sig") as handle:
            lines = handle.read().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _takes_predict_diagnostics(router) -> bool:
    """Whether `router`'s predict_batch takes the per-group sort report of #1111.

    The same signature check the evals runner makes for `sort_by_length`, but exact-name only:
    the report is a contract with core's own Router, and a `**kwargs` forwarder would swallow
    the dict while staying as blind as any other foreign router. Routers without the parameter
    keep the pre-run note, whose file-wide wording is exact for the single group they predict.
    """
    fn = getattr(router, "predict_batch", None)
    if fn is None:
        return False
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "diagnostics" in parameters


def _note_sort_report(report, total):
    """Say what actually reordered, once core has reported it per checkpoint group."""
    stats = report.get("sort_by_length") or {}
    groups = stats.get("groups") or 0
    if not groups or stats.get("reordered", 0) >= groups:
        return
    smallest, largest = stats.get("smallest"), stats.get("largest")
    if groups == 1:
        # One group saw every line, so the pre-run wording is exact there; keep its copy, with
        # the group's own size rather than the file's, in case a cache hit shrank the batch.
        print(
            "note: --sort-by-length only takes effect with --batch-size N where "
            "1 < N < %d; the states still go in one forward pass" % smallest,
            file=sys.stderr,
        )
        return
    span = "%d" % smallest if smallest == largest else "%d-%d" % (smallest, largest)
    # The post-run counterpart of the pre-run note: instead of guessing from the file before
    # the run, name the groups that could not reorder and the size that would have reached them.
    print(
        "note: --sort-by-length groups within each checkpoint batch, not across the file: "
        "%d of %d batches (%s states) could not reorder; a batch reorders only when "
        "--batch-size is smaller than its state count, and the smallest batch holds %d of "
        "the file's %d lines" % (groups - stats["reordered"], groups, span, smallest, total),
        file=sys.stderr,
    )


def run_batch(lines, args, router=None):
    """Route or predict a whole file in one call, so states share checkpoint loads and
    forward passes; returns 0 on success, 2 on a handled error."""
    router = router or make_router(args)
    overrides = {
        key: value
        for key, value in (
            ("model", args.model),
            ("task", args.task),
            ("lang", args.lang),
            ("lang_guess", args.lang_guess),
        )
        if value is not None
    }
    try:
        if args.predict or args.preset or getattr(args, "questions", None):
            questions, key = resolve_questions(args)
            # Router.predict_batch reads the token budget per request (#583).
            overrides.update(budget_overrides(args))
            requests = [
                {"state": {key: line}, "questions": questions, **overrides} for line in lines
            ]
            kwargs = {"batch_size": args.batch_size, **abstention_override(args)}
            sort_report = None
            if args.sort_by_length:
                # Sent only when asked for, as Router.predict_batch itself forwards it: a stub or
                # an attached agent that predates the knob keeps working for every run that does
                # not request it.
                kwargs["sort_by_length"] = True
                chunk = args.batch_size if args.batch_size and args.batch_size > 0 else len(lines)
                if _takes_predict_diagnostics(router):
                    # The reorder switches per checkpoint/question group inside core, so the
                    # file's line count is only an upper bound on the denominators that decide
                    # it (#1111): a file that routes to several checkpoints can reorder nothing
                    # while `1 < chunk < len(lines)` holds. Ask core for the per-group outcome
                    # and note after the run, when the groups are known.
                    sort_report = {}
                    kwargs["diagnostics"] = sort_report
                elif not 1 < chunk < len(lines):
                    # Core documents this as a no-op rather than an error, so the shell says why
                    # the run was not reordered instead of printing a silent same-speed result.
                    print(
                        "note: --sort-by-length only takes effect with --batch-size N where "
                        "1 < N < %d; the states still go in one forward pass" % len(lines),
                        file=sys.stderr,
                    )
            results = router.predict_batch(requests, **kwargs)
            if sort_report is not None:
                _note_sort_report(sort_report, len(lines))
            for line, result in zip(lines, results):
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, default=str))
                else:
                    print("# %s" % (line if len(line) <= 72 else line[:69] + "..."))
                    show_answers(result)
                    print()
        else:
            decisions = router.route_batch(
                [{"state": {"text": line}, "questions": {}, **overrides} for line in lines],
                **({"explain": True} if args.explain else {}),
            )
            for line, decision in zip(lines, decisions):
                if args.json:
                    print(json.dumps(dict(decision), ensure_ascii=False, default=str))
                elif args.explain:
                    show_decision(decision)
                else:
                    print(
                        "%-14s %s  <- %s"
                        % (
                            decision["model"],
                            decision["reason"],
                            line if len(line) <= 48 else line[:45] + "...",
                        )
                    )
    except ValueError as error:
        print("laya: %s" % error, file=sys.stderr)
        return 2
    except (ImportError, OSError, RuntimeError) as error:
        print("laya: could not run Laya (%s)." % error, file=sys.stderr)
        print(
            "Check that the dependencies are installed and the checkpoints can be "
            "downloaded from the Hugging Face hub (network access is needed on first use).",
            file=sys.stderr,
        )
        return 2
    return 0


def interactive(args):
    print("Laya interactive mode. Type a request and press Enter; Ctrl-D or 'quit' to exit.")
    router = make_router(args)
    code = 0
    while True:
        try:
            text = input("laya> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        # A handled failure stays non-fatal to the session, but the process cannot then report
        # success for a request it knows did not run: `laya < requests.txt` exited 0 with every
        # answer missing and only stderr to say so.
        code = run(text, args, router) or code
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":  # `laya eval ...` mirrors the `laya-evals` script
        from .evals_cli import main as eval_main

        return eval_main(argv[1:])
    if argv and argv[0] == "train":  # `laya train ...` mirrors the `laya-train` script
        print("Training requires official laya; MLX exports support inference.", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    # Redirected output is encoded with the locale's codec by default (cp1252 on Windows), which
    # cannot hold a non-Latin request or answer, so `laya --batch FILE > out.txt` failed with a
    # codec error instead of writing it. utf-8 is what --batch reads back.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    text = " ".join(args.text).strip()
    if args.explain and (args.predict or args.preset or args.questions):
        print(
            "laya: --explain is for routing only; omit --predict, --preset and --questions.",
            file=sys.stderr,
        )
        return 2
    # Routing answers nothing, so there is no confidence to gate: `route` does not take the
    # threshold at all. Accepting the flag there would let `laya --min-confidence 0.9 "..."` exit 0
    # having ignored it, which is the thing this flag exists to avoid.
    if args.min_confidence is not None and not (args.predict or args.preset or args.questions):
        print(
            "laya: --min-confidence gates answers, so it needs --predict, --preset or --questions.",
            file=sys.stderr,
        )
        return 2
    if args.batch:
        if text:
            print("laya: pass either a text or --batch FILE, not both.", file=sys.stderr)
            return 2
        try:
            lines = read_batch_lines(args.batch)
        except (OSError, UnicodeDecodeError) as error:
            print("laya: could not read %s (%s)." % (args.batch, error), file=sys.stderr)
            return 2
        if not lines:
            print("laya: no requests found in %s." % args.batch, file=sys.stderr)
            return 2
        return run_batch(lines, args)
    if any(getattr(args, key, None) for key in ("image", "audio", "video")):
        if text:
            print(
                "laya: pass either a text or one of --image, --audio, --video, not both.",
                file=sys.stderr,
            )
            return 2
        return run("", args)
    if not text:
        return interactive(args)
    return run(text, args)


if __name__ == "__main__":
    sys.exit(main())
