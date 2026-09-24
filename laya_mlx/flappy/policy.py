"""Real Laya predictions, with an explicit optional lookahead safety shield."""

import math
import time
from dataclasses import asdict, dataclass

from laya_mlx.snake.policy import checkpoint_metadata, local_checkpoint

from .game import ACTIONS


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_actions: list
    intervened: bool
    crash_risk: float
    clear_pipe: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
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

    def questions(self, game, actions):
        """Planner descriptions; compact wording mirrors the Snake demo's compact prompt."""
        safe = [a for a in actions if a.survivable]
        preferred = min(safe, key=lambda a: (abs(a.offset), a.action)).action if safe else "NONE"
        pipe = game.next_pipe()
        # The lookahead covers every known pipe, so a safe move also reaches the next gap.
        reachable = bool(safe)
        state = (
            f"Flappy game. {len(safe)} safe moves available. "
            f"Next gap: rows {pipe.gap_top}-{pipe.gap_top + pipe.gap - 1}, "
            f"{max(0, pipe.x - game.bird_x)} columns ahead. "
            f"Bird row {game.row}, {'rising' if game.velocity < 0 else 'falling'}. "
            f"{'There is a safe route forward.' if safe else 'The bird is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with the best line through the next gap. Avoid collisions.",
                "criteria": {
                    a.action: (
                        f"Collision: {a.reason}. Unsafe."
                        if a.crashes
                        else "Unsafe route. Risk of trapping the bird."
                        if not a.survivable
                        else "Safe. Best line through the gap. Best move."
                        if a.action == preferred
                        else "Safe but further from the gap center."
                    )
                    for a in actions
                },
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the bird?",
            },
            "clear": {
                "type": "noul",
                "instructions": "Can the bird reach the next gap?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Next gap reachable: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move."
            questions["move"]["criteria"] = {
                a.action: (
                    "Blocked. Collision."
                    if a.crashes
                    else "Unsafe. Traps the bird."
                    if not a.survivable
                    else "Safe. Best route."
                    if a.action == preferred
                    else "Safe. Slower route."
                )
                for a in actions
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["clear"]["instructions"] = "Is the next gap reachable?"
        return state, questions, preferred

    def decide(self, game):
        started = time.perf_counter()
        actions = game.actions()
        safe = [a.action for a in actions if a.survivable]
        if not safe and self.guarded:
            raise RuntimeError("Lookahead safety invariant violated: no survivable action")
        state, questions, preferred = self.questions(game, actions)
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["clear"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no action executed")
        proposed = max(ACTIONS, key=probabilities.__getitem__)
        executed = (
            max(safe, key=probabilities.__getitem__)
            if self.guarded and proposed not in safe
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_actions=safe,
            intervened=proposed != executed,
            crash_risk=1 - answers["risk"]["noul"],
            clear_pipe=answers["clear"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            planner_best=preferred,
        )
