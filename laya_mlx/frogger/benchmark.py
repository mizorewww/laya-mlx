"""Measure live model decisions, game survival, rendering work and paced deadlines."""

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console

from .game import FroggerGame
from .policy import LayaPolicy
from .ui import compose, layout_size


def percentiles(values):
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


def run_episode(policy, *, seed, steps, width, height, fps=None, render=True):
    game = FroggerGame(width, height, seed)
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=layout_size(width, height)[0],
        height=layout_size(width, height)[1],
        highlight=False,
    )
    trace = []
    started = time.perf_counter()
    last_home = 0
    max_home_gap = 0
    interventions = 0
    for index in range(steps):
        tick_start = time.perf_counter()
        decision = policy.decide(game)
        interventions += decision.intervened
        if render:
            canvas = compose(
                game.snapshot(),
                decision.to_dict(),
                {
                    "hardware": policy.metadata["hardware"],
                    "guarded": policy.guarded,
                    "interventions": interventions,
                    "elapsed": tick_start - started,
                    "steps_per_second": fps or 0,
                },
            )
            console.print(canvas.rich_text(), end="")
            buffer.seek(0)
            buffer.truncate(0)
        home = game.step(decision.executed)
        active_ms = (time.perf_counter() - tick_start) * 1000
        if home:
            max_home_gap = max(max_home_gap, game.ticks - last_home)
            last_home = game.ticks
        if policy.guarded and not game.alive:
            raise AssertionError("Guarded game was hit despite the lookahead shield")
        trace.append(
            {
                "tick": game.ticks,
                "frog_after": [game.x, game.y],
                "score": game.score,
                "level": game.level,
                "proposed": decision.proposed,
                "executed": decision.executed,
                "intervened": decision.intervened,
                "probabilities": decision.probabilities,
                "crash_risk": decision.crash_risk,
                "home_reachable": decision.home_reachable,
                "inference_ms": decision.inference_ms,
                "active_tick_ms": active_ms,
                "deadline_miss": bool(fps and active_ms > 1000 / fps),
                "input_tokens": decision.input_tokens,
            }
        )
        if fps:
            remaining = 1 / fps - (time.perf_counter() - tick_start)
            if remaining > 0:
                time.sleep(remaining)
        if not game.alive:
            break
    elapsed = time.perf_counter() - started
    result = {
        "seed": seed,
        "target_fps": fps,
        "requested_steps": steps,
        "steps": len(trace),
        "seconds": elapsed,
        "achieved_steps_per_second": len(trace) / elapsed,
        "score": game.score,
        "level": game.level,
        "alive": game.alive,
        "death_reason": game.death_reason,
        "interventions": interventions,
        "max_completed_home_gap": max_home_gap,
        "inference": percentiles([t["inference_ms"] for t in trace]),
        "active_tick": percentiles([t["active_tick_ms"] for t in trace]),
        "deadline_misses": sum(t["deadline_miss"] for t in trace),
        "trace": trace,
    }
    result["miss_fraction"] = result["deadline_misses"] / len(trace)
    result["passes"] = game.alive and result["miss_fraction"] <= 0.01
    print(
        f"{'shield' if policy.guarded else 'top-1'} seed={seed} target={fps or 'uncapped'} "
        f"steps={len(trace)} score={game.score} alive={game.alive} "
        f"actual={result['achieved_steps_per_second']:.1f}/s "
        f"p99={result['active_tick']['p99_ms']:.1f}ms "
        f"miss={result['miss_fraction']:.2%}",
        flush=True,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--rates", default="10,20,30,40,50,60")
    parser.add_argument("--sweep-steps", type=int, default=120)
    parser.add_argument("--soak-steps", type=int, default=600)
    parser.add_argument("--seeds", default="101,102,103,104")
    parser.add_argument("--raw-steps", type=int, default=200)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume saved episodes using the report's configuration",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/frogger.json"))
    args = parser.parse_args(argv)
    saved = json.loads(args.output.read_text()) if args.resume else None
    if saved:
        if saved["method"].get("color_system") != "truecolor":
            parser.error("This report predates explicit truecolor rendering; start a new report")
        if "prompt" not in saved["settings"]:
            parser.error("This older report has no prompt identifier and cannot be resumed")
        for key in (
            "model",
            "prompt",
            "width",
            "height",
            "rates",
            "sweep_steps",
            "soak_steps",
            "seeds",
            "raw_steps",
        ):
            setattr(args, key, saved["settings"][key])
        args.optimize = saved["settings"].get("optimize", False)
    if min(args.sweep_steps, args.soak_steps, args.raw_steps) < 1:
        parser.error("Step counts must be positive")
    rates = sorted(set(float(x) for x in args.rates.split(",")))
    if any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("Rates must be positive and finite")
    seeds = [int(x) for x in args.seeds.split(",")]
    policy = LayaPolicy(args.model, prompt=args.prompt, optimize=args.optimize)
    warm = FroggerGame(args.width, args.height, 9001)
    for _ in range(20):
        warm.step(policy.decide(warm).executed)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": policy.metadata,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "method": {
            "stable": "Zero deaths and finite outputs; <=1% ticks exceed the computation budget at each paced test rate. Crossings are reported, not required.",
            "timing": "Synchronized real model predict, planner, Rich composition and ANSI serialization, and game step. Paced rates also include real sleeps.",
            "color_system": "truecolor",
            "excluded": "Model loading, warmup, and the terminal emulator's own drawing. Rendering is serialized into an in-memory terminal stream.",
            "safety": "Laya sees planner features. Default shield restricts execution to moves from which a safe row (start or median) stays reachable over the lookahead. Unshielded top-1 is measured separately.",
            "limitation": "Highest passing tested rate on this machine and these seeds, not a universal maximum or proof of unaided game intelligence.",
        },
        "raw_top1": [],
        "uncapped": None,
        "sweep": [],
        "soak_attempts": [],
        "fastest_tested_stable_fps": None,
    }
    if saved:
        for key in ("source_revision", "weight_sha256_from_manifest", "prompt"):
            if saved["model"].get(key) != policy.metadata.get(key):
                parser.error(f"Cannot resume with changed model metadata: {key}")
        report = saved
        report.setdefault("resumed_sessions", []).append(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": policy.metadata,
                "note": "Completed episodes retained; an already failed rate needs no further seeds.",
            }
        )
    report.setdefault("uncapped_soak", [])
    report["method"]["uncapped_stability"] = (
        "Continuous --max-speed equivalent: every move waits for a fresh prediction. "
        "Long episodes test finite outputs, survival and crossings; no fixed frame deadline."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    shared = {"width": args.width, "height": args.height}
    policy.guarded = False
    for seed in seeds[:3]:
        if any(e["seed"] == seed for e in report["raw_top1"]):
            continue
        report["raw_top1"].append(run_episode(policy, seed=seed, steps=args.raw_steps, **shared))
        save()
    policy.guarded = True
    if report["uncapped"] is None:
        report["uncapped"] = run_episode(policy, seed=7, steps=args.sweep_steps, **shared)
        save()
    for rate in rates:
        if any(e["target_fps"] == rate for e in report["sweep"]):
            continue
        report["sweep"].append(
            run_episode(policy, seed=7, steps=args.sweep_steps, fps=rate, **shared)
        )
        save()
    for seed in seeds:
        if any(e["seed"] == seed for e in report["uncapped_soak"]):
            continue
        report["uncapped_soak"].append(
            run_episode(
                policy,
                seed=seed,
                steps=args.soak_steps,
                **shared,
            )
        )
        save()
    candidates = [r["target_fps"] for r in report["sweep"] if r["passes"]]
    for rate in reversed(candidates):
        attempt = next((a for a in report["soak_attempts"] if a["fps"] == rate), None)
        if attempt is None:
            attempt = {"fps": rate, "episodes": []}
            report["soak_attempts"].append(attempt)
        if any(not e["passes"] for e in attempt["episodes"]):
            attempt["passes"] = False
            save()
            continue
        for seed in seeds:
            if any(e["seed"] == seed for e in attempt["episodes"]):
                continue
            attempt["episodes"].append(
                run_episode(
                    policy,
                    seed=seed,
                    steps=args.soak_steps,
                    fps=rate,
                    **shared,
                )
            )
            save()
            if not attempt["episodes"][-1]["passes"]:
                break
        attempt["passes"] = len(attempt["episodes"]) == len(seeds) and all(
            e["passes"] for e in attempt["episodes"]
        )
        if attempt["passes"]:
            report["fastest_tested_stable_fps"] = rate
            save()
            break
        save()
    print(
        json.dumps(
            {
                "report": str(args.output),
                "fastest_tested_stable_fps": report["fastest_tested_stable_fps"],
            },
            indent=2,
        )
    )
    return 0 if report["fastest_tested_stable_fps"] else 1
