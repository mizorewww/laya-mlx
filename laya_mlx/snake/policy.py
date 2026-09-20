"""Real Laya predictions, with an explicit optional deterministic safety shield."""

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from .game import DIRECTIONS

DEFAULT_MODEL = "aac6fef/laya-multilingual-mlx"


def local_checkpoint(value=None):
    """Resolve a directory or an already cached Hub snapshot without network access."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if value is None:
        for path in ("models/hub/laya-multilingual-mlx", "models/laya-multilingual"):
            if Path(path).is_dir():
                return Path(path)
        value = DEFAULT_MODEL
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    if (
        str(value).startswith((".", "/", "~"))
        or Path(value).is_absolute()
        or bool(Path(value).drive)
    ):
        raise FileNotFoundError(f"Local checkpoint does not exist: {value}")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception as error:
        raise FileNotFoundError(
            f"{value} is not cached. Download it before starting the offline demo:\n"
            f"  hf download {value} --local-dir models/snake\n"
            "  laya-snake --model models/snake"
        ) from error


def hardware_name():
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip().removeprefix("Apple ")
    return platform.machine()


def checkpoint_metadata(path):
    meta = path / "mlx_config.json"
    values = json.loads(meta.read_text()) if meta.exists() else {}
    manifest = path / "manifest.json"
    source = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {
        "name": values.get("repository", path.name),
        "source_revision": values.get("source_revision"),
        "weight_sha256_from_manifest": source.get("files", {})
        .get("model.safetensors", {})
        .get("sha256"),
        "hardware": hardware_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("mlx", "numpy", "rich", "tokenizers", "huggingface-hub")
        },
        "network": "offline",
        "policy": "Laya probabilities over planner features; optional cycle safety shield",
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(self, model=None, *, guarded=True, prompt="compact", optimize=False):
        from laya_mlx import Agent

        self.path = local_checkpoint(model)
        self.agent = Agent(
            self.path,
            dtype="float16",
            device="gpu",
            batch_size=3,
            compile=optimize,
            pad_to_multiple=16 if optimize else None,
            cache_prompts=optimize,
        )
        self.guarded = guarded
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.prompt = prompt
        self.metadata = checkpoint_metadata(self.path)
        self.metadata["prompt"] = prompt
        self.metadata["optimization"] = (
            "compile + 16-token buckets + prefix cache" if optimize else "eager"
        )

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move toward food."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the snake."
                    if not m.safe
                    else "Safe. Eat food now. Best."
                    if m.eats
                    else "Safe. Best route to food."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["food"]["instructions"] = "Is food reachable through empty cells?"
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
