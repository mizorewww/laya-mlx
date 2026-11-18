"""Compare opt-in embedding buckets with unchanged input-order batches on a local checkpoint.

python -m experiments.local_perf.embedding_buckets MODEL --output results.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from benchmarks.common import environment
from laya_mlx import Agent
from laya_mlx.shortlist import embed_fn_from_agent

LABELS = [
    "Cash withdrawal",
    "Duplicate card charge",
    "Refund missing",
    "Card payment declined",
    "Transfer delayed",
    "Cash deposit",
    "Change personal details",
    "Close account",
    "Exchange rate",
    "Lost card",
    "Card delivery",
    "Account verification",
    "Chargeback request",
    "Subscription cancelled",
    "Pending payment",
    "Payment reversed",
    "Balance incorrect",
    "ATM fee",
    "Transfer fee",
    "Card activation",
    "Incorrect PIN",
    "Reset password",
    "Stolen card",
    "International transfer",
    "Direct debit",
    "Interest rate",
    "Cashback missing",
    "Deposit pending",
    "Recipient details",
    "Credit limit",
    "Fraud report",
]
QUERY = "I was charged twice for a card payment. I need help reversing the duplicate transaction. "


def compare(a, b):
    unit_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    unit_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    ranks = [np.argsort(-(v[1:] @ v[0]), kind="stable") for v in (unit_a, unit_b)]
    return {
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "min_cosine": float(np.min(np.sum(unit_a * unit_b, axis=1))),
        "top_k": {
            str(k): {
                "ordered_equal": bool(np.array_equal(ranks[0][:k], ranks[1][:k])),
                "set_equal": set(ranks[0][:k]) == set(ranks[1][:k]),
            }
            for k in (1, 4, 8)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    results = []
    for dtype in ("float16", "float32"):
        agent = Agent(args.model, dtype=dtype, local_files_only=True)
        cases = [
            ("query_and_labels", [QUERY * 30] + LABELS, 32, 256),
            ("mixed_lengths", [QUERY * 30 if i % 4 == 0 else LABELS[i] for i in range(16)], 4, 256),
            ("short_only", [QUERY] + LABELS, 32, 256),
            ("single_row_batches", [QUERY * 30] + LABELS[:7], 1, 128),
            ("longer_limit", [QUERY * 60] + LABELS[:7], 8, 512),
        ]
        for name, texts, batch_size, limit in cases:
            functions = {
                key: embed_fn_from_agent(
                    agent, max_length=limit, batch_size=batch_size, bucket_by_length=bucket
                )
                for key, bucket in (("baseline", False), ("bucketed", True))
            }
            a, b = [fn(texts) for fn in functions.values()]  # Warm both shapes before timing.
            samples = {key: [] for key in functions}
            for repeat in range(args.repeats):
                for key in list(functions) if repeat % 2 == 0 else list(functions)[::-1]:
                    start = time.perf_counter()
                    functions[key](texts)  # The API materializes the pooled result.
                    samples[key].append((time.perf_counter() - start) * 1000)
            medians = {key: statistics.median(values) for key, values in samples.items()}
            row = {
                "case": name,
                "dtype": dtype,
                "batch_size": batch_size,
                "max_length": limit,
                "median_ms": medians,
                "speedup": medians["baseline"] / medians["bucketed"],
                "numerics": compare(a, b),
                "samples_ms": samples,
            }
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    with open(args.output, "w") as file:
        json.dump(
            {
                "environment": environment(),
                "checkpoint": Path(args.model).name,
                "repeats": args.repeats,
                "cases": results,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
