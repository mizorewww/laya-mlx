"""Real Laya predictions, with an explicit optional lookahead safety shield."""

import math
import time
from dataclasses import asdict, dataclass

from laya_mlx.snake.policy import checkpoint_metadata, local_checkpoint

from .game import DIRECTIONS

__all__ = ["Decision", "LayaPolicy", "local_checkpoint"]


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    crash_risk: float
    home_reachable: float
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
        self.metadata["policy"] = (
            "Laya probabilities over planner features; optional lookahead shield"
        )
        self.metadata["prompt"] = prompt
        self.metadata["optimization"] = (
            "compile + 16-token buckets + prefix cache" if optimize else "eager"
        )

    def questions(self, game, moves):
        """Planner descriptions in the Snake prompt structure; see FROGGER_DEMO.md for wording."""
        safe = [m for m in moves if m.safe]
        routes = [m for m in safe if m.steps_home >= 0]
        preferred = (
            min(routes, key=lambda m: (m.steps_home, DIRECTIONS.index(m.direction))).direction
            if routes
            else min(safe, key=lambda m: DIRECTIONS.index(m.direction)).direction
            if safe
            else "NONE"
        )
        reachable = bool(routes)
        state = (
            f"Frogger game. {len(safe)} safe directions available. "
            f"Home reachable without being hit or falling in the water: "
            f"{'yes' if reachable else 'no'}. Frog row {game.y} of {game.height - 1}. "
            f"{'There is a safe route forward.' if safe else 'The frog is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward home. Avoid collisions.",
                "criteria": {
                    m.direction: (
                        f"Collision: {m.reason}. Unsafe."
                        if not m.legal
                        else "Unsafe route. Risk of trapping the frog."
                        if not m.safe
                        else "Safe. Reach home immediately. Best move."
                        if m.home
                        else "Safe. Fastest route home. Best move."
                        if m.direction == preferred
                        else "Safe but slower route home."
                    )
                    for m in moves
                },
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the frog?",
            },
            "home": {
                "type": "noul",
                "instructions": "Is home reachable without being hit or falling in the water?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Home reachable: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move toward home."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the frog."
                    if not m.safe
                    else "Safe. Reach home now. Best."
                    if m.home
                    else "Safe. Best move."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["home"]["instructions"] = "Is home reachable?"
        return state, questions, preferred

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m.direction for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Lookahead safety invariant violated: no safe action")
        state, questions, preferred = self.questions(game, moves)
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["home"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        executed = (
            max(safe, key=probabilities.__getitem__)
            if self.guarded and proposed not in safe
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=safe,
            intervened=proposed != executed,
            crash_risk=1 - answers["risk"]["noul"],
            home_reachable=answers["home"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
