"""Deterministic integer Flappy rules and a separately identified lookahead planner.

The only control is a jump: JUMP gives an upward impulse, WAIT (not jumping) lets gravity
act. Vertical motion uses half-row units so jumps follow a visible arc.
"""

import random
from dataclasses import dataclass
from functools import lru_cache

ACTIONS = ("JUMP", "WAIT")
UNITS = 2  # Vertical positions and speeds are in half-row units.
JUMP_VELOCITY = -3  # 1.5 rows/tick upward, then gravity slows the bird: a 3-row arc.
GRAVITY = 1  # 0.5 rows/tick added every tick without a jump.
MAX_FALL = 4  # 2 rows/tick.
PIPE_WIDTH = 2
LEVEL_PIPES = 10  # Every ten pipes: one row narrower and two columns closer.
MIN_GAP = 3
MIN_SPACING = 8


@dataclass(frozen=True)
class ActionInfo:
    action: str
    y: int
    velocity: int
    crashes: bool
    survivable: bool
    offset: float
    reason: str


@dataclass(frozen=True)
class Pipe:
    x: int
    gap_top: int
    gap: int


class FlappyGame:
    def __init__(self, width=24, height=16, seed=7, gap=5, spacing=12, bird_x=6):
        if gap < MIN_GAP or height < gap + 4 or width < spacing + bird_x + PIPE_WIDTH:
            raise ValueError(
                f"Board too small: need gap >= {MIN_GAP}, height >= gap + 4 "
                "and width >= spacing + bird column + pipe width"
            )
        self.width, self.height, self.seed = width, height, seed
        self.gap, self.spacing, self.bird_x = gap, spacing, bird_x
        # Every known pipe lies inside the lookahead, including one spawned this tick.
        self.horizon = width + spacing
        self.rng = random.Random(seed)
        self.y, self.velocity = height // 2 * UNITS, 0
        self.score = self.ticks = self.spawned = 0
        self.alive, self.death_reason = True, None
        self.pipes = []
        self._next_x = bird_x + spacing
        while self._next_x < width + spacing:
            self._spawn()

    @property
    def row(self):
        return self.y // UNITS

    @property
    def level(self):
        return 1 + self.score // LEVEL_PIPES

    def _spawn(self):
        x, level = self._next_x, self.spawned // LEVEL_PIPES
        gap = max(MIN_GAP, self.gap - level)
        top, bottom = 0, self.height - gap
        zone = self.rng.choice(("top", "bottom", "anywhere"))
        wanted = (
            self.rng.randint(top, top + 1)
            if zone == "top"
            else self.rng.randint(bottom - 1, bottom)
            if zone == "bottom"
            else self.rng.randint(top, bottom)
        )
        # Accept the requested gap only if the bird can still reach it from its current
        # state; otherwise take the nearest reachable row. This keeps the shield complete.
        self.pipes.append(None)
        for gap_top in sorted(range(top, bottom + 1), key=lambda row: (abs(row - wanted), row)):
            self.pipes[-1] = Pipe(x, gap_top, gap)
            if not self.alive or self.survivable(self.y, self.velocity, 0):
                break
        self.spawned += 1
        self._next_x = x + max(MIN_SPACING, self.spacing - 2 * level)

    @staticmethod
    def physics(y, velocity, action):
        velocity = JUMP_VELOCITY if action == "JUMP" else min(MAX_FALL, velocity + GRAVITY)
        return y + velocity, velocity

    def pipes_at(self, dt):
        """Pipe positions dt ticks from now; spawns beyond the right edge are ignored."""
        return [Pipe(p.x - dt, p.gap_top, p.gap) for p in self.pipes if p.x - dt + PIPE_WIDTH > 0]

    def collision(self, y, pipes):
        """Collision for a vertical position in half-row units."""
        if y < 0:
            return "ceiling"
        if y >= self.height * UNITS:
            return "ground"
        y //= UNITS
        for pipe in pipes:
            if pipe.x <= self.bird_x < pipe.x + PIPE_WIDTH and not (
                pipe.gap_top <= y < pipe.gap_top + pipe.gap
            ):
                return "pipe"
        return None

    def next_pipe(self):
        return next(p for p in self.pipes if p.x + PIPE_WIDTH > self.bird_x)

    def survivable(self, y, velocity, dt=1):
        """Whether some action sequence avoids collision for the planning horizon."""
        pipes = tuple(self.pipes)

        @lru_cache(maxsize=None)
        def search(y, velocity, dt):
            if dt >= self.horizon:
                return True
            shifted = [Pipe(p.x - dt - 1, p.gap_top, p.gap) for p in pipes]
            for action in ("WAIT", "JUMP"):
                ny, nv = self.physics(y, velocity, action)
                if self.collision(ny, shifted) is None and search(ny, nv, dt + 1):
                    return True
            return False

        return search(y, velocity, dt)

    def actions(self):
        if not self.alive:
            return []
        pipe = self.next_pipe()
        center = pipe.gap_top + (pipe.gap - 1) / 2
        result = []
        for action in ACTIONS:
            y, velocity = self.physics(self.y, self.velocity, action)
            reason = self.collision(y, self.pipes_at(1))
            crashes = reason is not None
            survivable = not crashes and self.survivable(y, velocity)
            if not crashes and not survivable:
                reason = "no escape within the planning horizon"
            result.append(
                ActionInfo(
                    action, y, velocity, crashes, survivable, y / UNITS - center, reason or "clear"
                )
            )
        return result

    def step(self, action):
        """Advance one tick; returns whether the bird is still alive."""
        if not self.alive:
            raise RuntimeError("Cannot step a finished game")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        self.ticks += 1
        self.y, self.velocity = self.physics(self.y, self.velocity, action)
        self.pipes = self.pipes_at(1)
        self.score += sum(p.x + PIPE_WIDTH == self.bird_x for p in self.pipes)
        self._next_x -= 1
        if self._next_x < self.width + self.spacing:
            self._spawn()
        reason = self.collision(self.y, self.pipes)
        if reason:
            self.alive, self.death_reason = False, reason
        return self.alive

    def snapshot(self):
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "gap": self.gap,
            "bird": [self.bird_x, self.row],
            "y": self.y,
            "velocity": self.velocity,
            "pipes": [[p.x, p.gap_top, p.gap] for p in self.pipes],
            "score": self.score,
            "ticks": self.ticks,
            "alive": self.alive,
            "death_reason": self.death_reason,
        }
