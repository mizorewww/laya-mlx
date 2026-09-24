"""Deterministic Frogger rules and a separately identified lookahead safety planner.

Row 0 is home, rows 1-6 are river lanes with logs, row 7 is a safe median, rows 8-14
are road lanes with cars and row 15 is the safe start. Lanes are cyclic patterns that
shift one cell every `period` ticks, so every future lane state is known exactly.
"""

import random
from dataclasses import dataclass

import numpy as np

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0), "WAIT": (0, 0)}
LEVEL_CROSSINGS = 3  # Every three crossings: faster, denser cars and shorter logs.


@dataclass(frozen=True)
class Lane:
    kind: str  # "river", "road" or "safe"
    direction: int
    period: int
    pattern: tuple  # True marks a log (river) or a car (road).

    def offset(self, t):
        return self.direction * (t // self.period) if self.kind != "safe" else 0

    def moves(self, t):
        """Whether the lane shifts between tick t and t + 1."""
        return self.kind != "safe" and (t + 1) % self.period == 0

    def row(self, t):
        return np.roll(np.array(self.pattern, dtype=bool), self.offset(t))


@dataclass(frozen=True)
class MoveInfo:
    direction: str
    legal: bool
    safe: bool
    steps_home: int  # Ticks to reach home within the horizon; -1 when not reachable.
    reason: str
    home: bool


def _segments(rng, width, lengths, gaps):
    """A cyclic row of objects whose lengths and gaps come from the given ranges."""
    pattern = [False] * width
    x = rng.randrange(width)
    placed = 0
    while placed < width:
        length, gap = rng.randint(*lengths), rng.randint(*gaps)
        if placed + length + gap > width:
            break
        for i in range(length):
            pattern[(x + i) % width] = True
        x, placed = x + length + gap, placed + length + gap
    return tuple(pattern)


class FroggerGame:
    RIVER = range(1, 7)
    MEDIAN = 7
    ROAD = range(8, 15)

    def __init__(self, width=24, height=16, seed=7, horizon=32):
        if width < 8 or height != 16:
            raise ValueError("Board must be 16 rows tall and at least 8 columns wide")
        self.width, self.height, self.seed, self.horizon = width, height, seed, horizon
        self.start = (width // 2, height - 1)
        self.rng = random.Random(seed)
        self.x, self.y = self.start
        self.score = self.ticks = 0
        self.alive, self.death_reason = True, None
        self._set_lanes(self._build_lanes(0))

    @property
    def level(self):
        return 1 + self.score // LEVEL_CROSSINGS

    def _build_lanes(self, level):
        lanes = []
        for y in range(self.height):
            direction = 1 if y % 2 else -1
            if y in self.RIVER:
                period = max(1, self.rng.choice((2, 3, 4)) - level // 2)
                longest = max(2, 4 - level // 3)
                pattern = _segments(self.rng, self.width, (max(2, longest - 1), longest), (3, 6))
                lanes.append(Lane("river", direction, period, pattern))
            elif y in self.ROAD:
                period = max(1, self.rng.choice((2, 3, 4)) - level // 2)
                gaps = (max(2, 4 - level // 2), max(3, 6 - level // 2))
                pattern = _segments(self.rng, self.width, (1, 3), gaps)
                lanes.append(Lane("road", direction, period, pattern))
            else:
                lanes.append(Lane("safe", 0, 1, (False,) * self.width))
        return lanes

    def _set_lanes(self, lanes):
        self.lanes = lanes
        kind = np.array([lane.kind for lane in lanes])
        self._river, self._road = kind == "river", kind == "road"
        self._direction = np.array([lane.direction for lane in lanes])
        self._period = np.array([lane.period for lane in lanes])
        self._grid_cache, self._cache = {}, None

    def _grids(self, t):
        """Car and log occupancy at absolute tick t, as (height, width) arrays."""
        if t in self._grid_cache:
            return self._grid_cache[t]
        if len(self._grid_cache) > 4 * self.horizon:
            self._grid_cache.clear()
        cars = np.zeros((self.height, self.width), dtype=bool)
        logs = np.zeros_like(cars)
        for y, lane in enumerate(self.lanes):
            if lane.kind == "road":
                cars[y] = lane.row(t)
            elif lane.kind == "river":
                logs[y] = lane.row(t)
        self._grid_cache[t] = cars, logs
        return cars, logs

    def _transition(self, t, dx, dy, ys, xs, reasons=False):
        """Vectorized step from tick t: (valid, final_y, final_x[, reason])."""
        cars, _ = self._grids(t)
        cars1, logs1 = self._grids(t + 1)
        ty, tx = ys + dy, xs + dx
        inside = (ty >= 0) & (ty < self.height) & (tx >= 0) & (tx < self.width)
        cy, cx = np.clip(ty, 0, self.height - 1), np.clip(tx, 0, self.width - 1)
        river, road = self._river[cy], self._road[cy]
        carry = self._direction * ((t + 1) % self._period == 0) * self._river
        fx = cx + carry[cy]
        swept = river & ((fx < 0) | (fx >= self.width))
        fx = np.clip(fx, 0, self.width - 1)
        hit = road & (cars[cy, cx] | cars1[cy, fx])
        wet = river & ~logs1[cy, fx]
        valid = inside & ~hit & ~swept & ~wet
        if not reasons:
            return valid, cy, fx
        reason = np.select(
            [~inside, hit, swept, wet], ["wall", "car", "swept away", "water"], "legal"
        )
        return valid, cy, fx, reason

    def _plan(self):
        """Backward induction over the horizon: survival and ticks-to-home per cell."""
        if self._cache and self._cache[0] == (self.ticks, self.score):
            return self._cache[1]
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        safe_row = np.zeros((self.height, self.width), dtype=bool)
        safe_row[[self.MEDIAN, self.height - 1]] = True
        good = safe_row.copy()  # Beyond the horizon only safe rows are trusted.
        dist = np.full((self.height, self.width), np.inf)
        layers = [None] * (self.horizon + 1)
        layers[self.horizon] = (good, dist)
        for k in range(self.horizon - 1, 0, -1):
            t = self.ticks + k
            next_good, next_dist = layers[k + 1]
            good_k = safe_row.copy()
            dist_k = np.full_like(dist, np.inf)
            for dx, dy in VECTORS.values():
                valid, fy, fx = self._transition(t, dx, dy, ys, xs)
                home = valid & (fy == 0)
                good_k |= valid & (home | next_good[fy, fx])
                steps = np.where(home, 1, 1 + next_dist[fy, fx])
                dist_k = np.minimum(dist_k, np.where(valid, steps, np.inf))
            layers[k] = (good_k, dist_k)
        self._cache = ((self.ticks, self.score), layers[1])
        return layers[1]

    def moves(self):
        if not self.alive:
            return []
        good, dist = self._plan()
        result = []
        for direction in DIRECTIONS:
            dx, dy = VECTORS[direction]
            valid, fy, fx, reason = (
                v.item()
                for v in self._transition(
                    self.ticks, dx, dy, np.array(self.y), np.array(self.x), reasons=True
                )
            )
            home = valid and fy == 0
            safe = valid and (home or bool(good[fy, fx]))
            steps = 1 if home else 1 + dist[fy, fx] if valid else np.inf
            if valid and not safe:
                reason = "no escape within the planning horizon"
            result.append(
                MoveInfo(
                    direction,
                    valid,
                    safe,
                    int(steps) if np.isfinite(steps) else -1,
                    reason,
                    home,
                )
            )
        return result

    def home_reachable(self):
        return any(m.safe and m.steps_home >= 0 for m in self.moves())

    def step(self, direction):
        """Advance one tick; returns whether the frog reached home."""
        if not self.alive:
            raise RuntimeError("Cannot step a finished game")
        if direction not in VECTORS:
            raise ValueError(f"Unknown direction: {direction}")
        dx, dy = VECTORS[direction]
        valid, fy, fx, reason = (
            v.item()
            for v in self._transition(
                self.ticks, dx, dy, np.array(self.y), np.array(self.x), reasons=True
            )
        )
        self.ticks += 1
        if not valid:
            self.alive, self.death_reason = False, reason
            return False
        self.x, self.y = fx, fy
        if self.y == 0:
            self.score += 1
            self.x, self.y = self.start
            if self.score % LEVEL_CROSSINGS == 0:
                self._set_lanes(self._build_lanes(self.level - 1))
            return True
        return False

    def snapshot(self):
        cars, logs = self._grids(self.ticks)
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "frog": [self.x, self.y],
            "cars": [[x for x in range(self.width) if cars[y, x]] for y in range(self.height)],
            "logs": [[x for x in range(self.width) if logs[y, x]] for y in range(self.height)],
            "score": self.score,
            "ticks": self.ticks,
            "alive": self.alive,
            "death_reason": self.death_reason,
        }
