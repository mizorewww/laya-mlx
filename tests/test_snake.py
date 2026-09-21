import json
import random
import socket
from collections import deque
from pathlib import Path

import pytest

from laya_mlx.snake.game import DIRECTIONS, SnakeGame, hamiltonian_cycle
from laya_mlx.snake.policy import LayaPolicy, local_checkpoint


@pytest.mark.parametrize("width,height", [(4, 4), (4, 5), (5, 4), (24, 16)])
def test_cycle_visits_every_cell_and_closes(width, height):
    cycle = hamiltonian_cycle(width, height)
    assert len(cycle) == len(set(cycle)) == width * height
    assert all(0 <= x < width and 0 <= y < height for x, y in cycle)
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(cycle, cycle[1:] + cycle[:1])
    )


def test_collision_growth_tail_vacancy_and_board_clear():
    game = SnakeGame(4, 4, initial_length=4)
    game.body = deque([(1, 1), (1, 2), (0, 2), (0, 1)])
    game.food = (3, 3)
    assert game.legal_reason("LEFT") == "legal"
    assert game.legal_reason("DOWN") == "reverse"
    game.step("LEFT")
    assert game.alive and len(game.body) == 4
    game.step("LEFT")
    assert not game.alive and game.death_reason == "wall"

    full = SnakeGame(4, 4, initial_length=15)
    move = next(m for m in full.moves() if m.safe)
    assert full.step(move.direction)
    assert full.won and full.food is None and len(full.body) == 16 and full.score == 1
    assert full.moves() == []


@pytest.mark.parametrize("seed", range(12))
def test_arbitrary_shielded_choices_complete_board_without_starving(seed):
    game = SnakeGame(6, 6, seed=seed)
    rng = random.Random(seed + 100)
    last_food = 0
    # A safe action advances at least one cycle position and never passes food.
    for _ in range(game.capacity * (game.capacity - game.initial_length)):
        allowed = [m.direction for m in game.moves() if m.safe]
        assert allowed
        ate = game.step(rng.choice(allowed))
        assert game.alive and game.cycle_order_valid()
        assert len(game.body) == len(set(game.body))
        assert len(game.body) == game.initial_length + game.score
        assert game.ticks - last_food <= game.capacity
        if ate:
            last_food = game.ticks
        if game.won:
            break
    assert game.won


def test_seed_reproduces_foods_and_actions():
    first, second = SnakeGame(seed=71), SnakeGame(seed=71)
    for _ in range(100):
        direction = max((m for m in first.moves() if m.safe), key=lambda m: m.advance).direction
        first.step(direction)
        second.step(direction)
        assert first.snapshot() == second.snapshot()


def test_guard_preserves_raw_probabilities_and_reports_intervention():
    game = SnakeGame()
    safe = [m.direction for m in game.moves() if m.safe]
    unsafe = next(d for d in DIRECTIONS if d not in safe)
    probabilities = {d: 0.9 if d == unsafe else 0.1 / 3 for d in DIRECTIONS}

    class StubAgent:
        def predict(self, *_):
            return {
                "answers": {
                    "move": {"probabilities": probabilities},
                    "risk": {"noul": 0.97},
                    "food": {"noul": 0.92},
                },
                "usage": {"input_tokens": 100},
            }

    policy = LayaPolicy.__new__(LayaPolicy)
    policy.agent, policy.guarded = StubAgent(), True
    policy.prompt = "compact"
    policy.full_metrics = True
    result = policy.decide(game)
    assert result.proposed == unsafe and result.executed in safe and result.intervened
    assert result.probabilities is probabilities
    assert result.dead_end_risk == pytest.approx(0.03)
    policy.guarded = False
    assert policy.decide(game).executed == unsafe


def test_missing_local_model_fails_without_a_network_attempt(monkeypatch, tmp_path):
    attempts = []

    def forbidden(*_, **__):
        attempts.append(True)
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        local_checkpoint(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="Download it"):
        local_checkpoint("nonexistent-snake-demo-test/no-cache")
    assert attempts == []


def test_small_or_odd_boards_are_rejected():
    for shape in ((3, 4), (4, 3), (5, 5)):
        with pytest.raises(ValueError):
            SnakeGame(*shape)


@pytest.mark.parametrize("times", [(1, 1), (2, 1), (float("nan"),), (-1,)])
def test_recording_rejects_invalid_wall_clock_timestamps(tmp_path, times):
    from laya_mlx.snake.replay import load_record

    path = tmp_path / "record.jsonl"
    events = [{"type": "metadata", "format": "laya-snake-v1"}]
    events.extend({"type": "frame", "at": t} for t in times)
    path.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="strictly increase"):
        load_record(path)


def test_recording_preserves_actual_timestamps_and_probabilities(tmp_path):
    from laya_mlx.snake.replay import load_record

    path = tmp_path / "record.jsonl"
    metadata = {"type": "metadata", "format": "laya-snake-v1"}
    frames = [
        {"type": "frame", "at": 0.019, "decision": {"probabilities": {"UP": 0.8721}}},
        {"type": "frame", "at": 0.119, "decision": {"probabilities": {"UP": 0.0342}}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in [metadata, *frames]))
    assert load_record(path) == (metadata, frames)


@pytest.mark.parametrize("filename", ["snake-showcase.jsonl", "snake-fast.jsonl"])
def test_published_real_showcase_replays_every_board_and_action_exactly(filename):
    from laya_mlx.snake.replay import load_record

    path = Path(__file__).parents[1] / "benchmarks/results" / filename
    metadata, frames = load_record(path)
    settings = metadata["settings"]
    game = SnakeGame(
        settings["width"], settings["height"], settings["seed"], settings["initial_length"]
    )
    interventions = 0
    for frame in frames:
        assert frame["game"] == game.snapshot()
        decision = frame["decision"]
        probabilities = decision["probabilities"]
        assert set(probabilities) == set(DIRECTIONS)
        assert sum(probabilities.values()) == pytest.approx(1, abs=0.00021)
        assert decision["proposed"] == max(DIRECTIONS, key=probabilities.__getitem__)
        safe = [m.direction for m in game.moves() if m.safe]
        assert decision["executed"] in safe
        interventions += decision["intervened"]
        assert frame["stats"]["interventions"] == interventions
        game.step(decision["executed"])
        assert game.alive and game.cycle_order_valid()
    end = json.loads(path.read_text().splitlines()[-1])
    assert end["game"] == game.snapshot()
    assert end["summary"]["steps"] == end["summary"]["inference_calls"] == len(frames)
    assert end["summary"]["interventions"] == interventions
