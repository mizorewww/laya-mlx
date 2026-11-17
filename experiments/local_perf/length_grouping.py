"""Paired end-to-end benchmark against the unmodified Agent._forward.

Run from repository root:
python -m experiments.local_perf.length_grouping --model /path/to/checkpoint --output result.json
Uses local weights only. Alternate baseline/candidate order; retain all timing samples.
"""

import argparse
import ast
import json
import statistics
import subprocess
import time
import types
from pathlib import Path

import numpy as np

import laya_mlx.agent as module
from benchmarks.common import environment, parity_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default="b82b3e8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:laya_mlx/agent.py"], text=True
    )
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward")
    namespace = vars(module).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline_forward", "exec"), namespace)
    agent = module.Agent(args.model, local_files_only=True)
    methods = {
        "baseline": types.MethodType(namespace["_forward"], agent),
        "grouped": agent._forward,
    }
    instruction = "Does the customer request a refund?"
    policy = (
        "Consider the full support policy. Invoices, payments, duplicate charges and refund "
        "requests belong to billing. Software crashes and login errors belong to technical "
        "support. "
    ) * 14
    mixed = {
        str(i): {
            "type": "noul",
            "instructions": instruction if i % 2 == 0 else policy + instruction,
        }
        for i in range(32)
    }
    uniform = {str(i): {"type": "noul", "instructions": instruction} for i in range(32)}
    cases = {
        "mixed_32": ("I was charged twice. Please refund the duplicate.", mixed),
        "uniform_32": ("Please refund the duplicate.", uniform),
        "single": ("Refund please.", {"q": uniform["0"]}),
    }
    report = {"environment": environment(), "baseline": args.baseline, "cases": {}}
    # Public-output comparisons across changed inputs, types, counts, languages and long text.
    quality = {"questions": 0, "exact_results": 0, "cases": 0, "max_numeric_delta": 0.0}

    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, float):
            quality["max_numeric_delta"] = max(quality["max_numeric_delta"], abs(left - right))
            assert abs(left - right) <= 0.003
        else:
            assert left == right

    for _, state, questions in parity_cases() + [("mixed", *cases["mixed_32"])]:
        results = []
        for method in methods.values():
            agent._forward = method
            results.append(agent.predict(state, questions))
        compare(*results)
        quality["cases"] += 1
        quality["questions"] += len(questions)
        quality["exact_results"] += results[0] == results[1]
    report["quality"] = quality
    print("quality", quality, flush=True)
    for name, (state, questions) in cases.items():
        lengths = [len(item["ids"]) for item in agent.prepare(state, questions)[0]]
        record = {"row_lengths": lengths, "rounds": [], "samples_ms": {key: [] for key in methods}}
        for method in methods.values():
            agent._forward = method
            for _ in range(4):
                agent.predict(state, questions)
        for round_index in range(args.rounds):
            medians = {}
            order = list(methods) if round_index % 2 == 0 else list(reversed(methods))
            for key in order:
                agent._forward = methods[key]
                agent.predict(state, questions)
                samples = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    agent.predict(state, questions)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                record["samples_ms"][key].extend(samples)
                medians[key] = statistics.median(samples)
            record["rounds"].append(medians)
        record["p50_ms"] = {key: statistics.median(v) for key, v in record["samples_ms"].items()}
        ratios = np.array([r["baseline"] / r["grouped"] for r in record["rounds"]])
        rng = np.random.default_rng(42)
        boot = np.median(rng.choice(ratios, (2000, len(ratios)), replace=True), axis=1)
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


if __name__ == "__main__":
    main()
