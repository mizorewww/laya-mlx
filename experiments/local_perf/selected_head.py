"""Alternate full and selected final-head inference on identical local weights."""

import argparse
import ast
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import laya_mlx.model as model_module
from benchmarks.common import environment, parity_cases
from experiments.engineering.run_variants import distinct_workload
from laya_mlx import Agent
from laya_mlx.agent import collate_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default="8efe86fa172306774ed2f2bddbbfb76124574759")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:laya_mlx/model.py"], text=True
    )
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "DecisionModel"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    namespace = vars(model_module).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline_model", "exec"), namespace)
    agent = Agent(args.model, local_files_only=True)

    def baseline(**kw):
        return namespace["__call__"](agent.model, **kw)

    functions = {"baseline": baseline, "selected": agent.model}
    if args.compiled:
        functions = {k: mx.compile(v) for k, v in functions.items()}
    report = {
        "environment": environment(),
        "baseline": args.baseline,
        "compiled": args.compiled,
        "dtype": "float16",
        "cases": {},
    }
    quality = {
        "questions": 0,
        "exact_results": 0,
        "cases": 0,
        "decision_changes": 0,
        "max_public_numeric_delta": 0.0,
        "max_logit_delta": 0.0,
        "max_action_logit_delta": 0.0,
    }

    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, float):
            assert np.isfinite(b)
            quality["max_public_numeric_delta"] = max(
                quality["max_public_numeric_delta"], abs(a - b)
            )
        else:
            assert a == b, (a, b)

    def check_results(a, b):
        compare(a, b)
        quality["cases"] += 1
        quality["exact_results"] += a == b
        quality["questions"] += len(a["answers"])
        for k, answer in a["answers"].items():
            other = b["answers"][k]
            if answer["type"] == "noul":
                quality["decision_changes"] += (answer["noul"] >= 0.5) != (other["noul"] >= 0.5)
            elif answer["type"] == "score":
                quality["decision_changes"] += max(
                    answer["probabilities"], key=answer["probabilities"].get
                ) != max(other["probabilities"], key=other["probabilities"].get)

    for name, state, questions in parity_cases():
        values = []
        raw = []
        batch = collate_items(agent.prepare(state, questions)[0], agent.tok.pad_token_id)
        for fn in functions.values():
            agent._inference = fn
            values.append(agent.predict(state, questions))
            raw.append(tuple(np.asarray(v) for v in agent.forward(batch)))
        check_results(*values)
        quality["max_logit_delta"] = max(
            quality["max_logit_delta"], float(np.max(abs(raw[0][0] - raw[1][0])))
        )
        quality["max_action_logit_delta"] = max(
            quality["max_action_logit_delta"], float(np.max(abs(raw[0][1] - raw[1][1])))
        )
    print("fixtures", quality, flush=True)
    report["quality"] = quality
    for name, count, long in [
        ("short_1", 1, False),
        ("short_8", 8, False),
        ("long_1", 1, True),
        ("long_4", 4, True),
    ]:
        state, questions = distinct_workload(count, long=long)
        record = {
            "tokens": [len(it["ids"]) for it in agent.prepare(state, questions)[0]],
            "samples_ms": {k: [] for k in functions},
            "rounds": [],
        }
        for fn in functions.values():
            agent._inference = fn
            for _ in range(4):
                agent.predict(state, questions)
        for r in range(args.rounds):
            changed = dict(state)
            changed["ticket"] = str(1000 + r)
            medians, results = {}, {}
            for key in list(functions) if r % 2 == 0 else list(reversed(functions)):
                agent._inference = functions[key]
                agent.predict(changed, questions)
                samples = []
                for _ in range(2):
                    mx.synchronize()
                    t = time.perf_counter_ns()
                    results[key] = agent.predict(changed, questions)
                    mx.synchronize()
                    samples.append((time.perf_counter_ns() - t) / 1e6)
                record["samples_ms"][key].extend(samples)
                medians[key] = statistics.median(samples)
            check_results(results["baseline"], results["selected"])
            record["rounds"].append(medians)
        record["p50_ms"] = {k: statistics.median(v) for k, v in record["samples_ms"].items()}
        ratios = np.array([r["baseline"] / r["selected"] for r in record["rounds"]])
        boot = np.median(
            np.random.default_rng(42).choice(ratios, (2000, len(ratios)), replace=True), axis=1
        )
        record["paired_speedup"] = float(np.median(ratios))
        record["exploratory_95pct_interval"] = np.percentile(boot, [2.5, 97.5]).tolist()
        report["cases"][name] = record
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            record["p50_ms"],
            record["paired_speedup"],
            record["exploratory_95pct_interval"],
            flush=True,
        )
    print("all comparisons", quality, flush=True)


if __name__ == "__main__":
    main()
