"""Paired Snake ablation: compilation, fixed length, and bounded question-prefix reuse.

Uses actual recorded game states and the unchanged public Agent prediction path.
These experimental wrappers do not cache model outputs or alter model weights.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import numpy as np

from laya_mlx.prepared import PrefixCache
from laya_mlx.snake.game import SnakeGame


class PrefixPreparation:
    """Use runtime preparation so this ablation shares truncation and usage semantics."""

    def __init__(self, agent):
        self.agent = agent
        self.cache = PrefixCache()

    def __call__(self, state, questions):
        return self.cache.prepare(self.agent, state, questions)


def main():
    from laya_mlx.snake.policy import LayaPolicy
    from laya_mlx.snake.replay import load_record

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/hub/laya-multilingual-mlx")
    parser.add_argument(
        "--recording", type=Path, default=Path("benchmarks/results/snake-showcase.jsonl")
    )
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--bucket", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, frames = load_record(args.recording)
    selected = [frames[i] for i in np.linspace(0, len(frames) - 1, args.states, dtype=int)]
    policy = LayaPolicy(args.model)
    agent = policy.agent
    original_forward = agent.forward
    original_encode = agent._encode_state
    cached_prepare = PrefixPreparation(agent)
    compiled = mx.compile(agent.model)
    mode = "eager"
    shapes = set()
    input_identity_checks = 0

    def encode(state, ids, internal, **kwargs):
        nonlocal input_identity_checks
        agent._prefix_cache = cached_prepare.cache if "prefix" in mode else None
        if mode == "validate":
            agent._prefix_cache = None
            original = original_encode(state, ids, internal, **kwargs)
            agent._prefix_cache = cached_prepare.cache
            cached = original_encode(state, ids, internal, **kwargs)
            assert cached == original
            input_identity_checks += 1
            return original
        return original_encode(state, ids, internal, **kwargs)

    def forward(batch):
        shapes.add(tuple(batch["input_ids"].shape))
        if "96" in mode or "bucket" in mode:
            length = batch["input_ids"].shape[1]
            padded_length = 96 if "96" in mode else args.bucket
            if length > padded_length:
                raise ValueError(f"Actual sequence length {length} exceeds the proposed bucket")
            batch = {
                **batch,
                "input_ids": np.pad(
                    batch["input_ids"],
                    ((0, 0), (0, padded_length - length)),
                    constant_values=agent.tok.pad_token_id,
                ),
                "attention_mask": np.pad(
                    batch["attention_mask"], ((0, 0), (0, padded_length - length))
                ),
            }
        if "compiled" not in mode:
            return original_forward(batch)
        with mx.stream(agent.device):
            result = compiled(**{k: mx.array(v) for k, v in batch.items()})
            mx.eval(result)
        return result

    agent._encode_state, agent.forward = encode, forward
    variants = [
        "eager",
        "prefix",
        "compiled",
        "compiled96",
        "prefix_compiled",
        "compiled_bucket",
        "prefix_compiled_bucket",
    ]
    records, warmup_ms = [], {}

    def restore(frame):
        board = frame["game"]
        game = SnakeGame(board["width"], board["height"], board["seed"])
        game.body = deque(tuple(cell) for cell in board["body"])
        game.food = tuple(board["food"]) if board["food"] else None
        game.score, game.ticks = board["score"], board["ticks"]
        return game

    # Every measured shape is visited before timing; cold work is reported separately.
    for mode in ["validate", *variants]:
        started = time.perf_counter()
        for frame in selected:
            policy.decide(restore(frame))
        warmup_ms[mode] = (time.perf_counter() - started) * 1000
    for index, frame in enumerate(selected):
        record = {"source_tick": frame["game"]["ticks"], "decisions": {}}
        order = variants[index % len(variants) :] + variants[: index % len(variants)]
        record["order"] = order
        game = restore(frame)
        for mode in order:
            record["decisions"][mode] = policy.decide(game).to_dict()
        records.append(record)
    summary = {}
    for variant in variants:
        values = [r["decisions"][variant] for r in records]
        drifts = []
        for record in records:
            baseline, candidate = record["decisions"]["eager"], record["decisions"][variant]
            drifts.extend(
                abs(baseline["probabilities"][d] - candidate["probabilities"][d])
                for d in baseline["probabilities"]
            )
            drifts.extend(
                abs(baseline[k] - candidate[k]) for k in ("dead_end_risk", "food_reachable")
            )
        summary[variant] = {
            "p50_ms": float(np.median([v["inference_ms"] for v in values])),
            "p95_ms": float(np.percentile([v["inference_ms"] for v in values], 95)),
            "max_probability_drift": max(drifts),
            "proposed_agreement": sum(
                r["decisions"][variant]["proposed"] == r["decisions"]["eager"]["proposed"]
                for r in records
            ),
            "executed_agreement": sum(
                r["decisions"][variant]["executed"] == r["decisions"]["eager"]["executed"]
                for r in records
            ),
            "interventions": sum(v["intervened"] for v in values),
        }
    report = {
        "method": "Actual recorded states; cyclic candidate order within each state; every shape warmed before measurement. All candidates perform a fresh model forward. Prefix variant caches at most 128 tokenized question prefixes, and tokenizes shared state once per call.",
        "model": policy.metadata,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_batch_shapes": sorted(shapes),
        "prefix_input_identity_checks": input_identity_checks,
        "warmup_all_shapes_ms_including_any_compilation": warmup_ms,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"shapes": sorted(shapes), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
