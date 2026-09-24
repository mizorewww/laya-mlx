import json
import random
import socket
from pathlib import Path

import pytest

from laya_mlx.frogger.game import DIRECTIONS, LEVEL_CROSSINGS, FroggerGame, Lane
from laya_mlx.frogger.policy import LayaPolicy, local_checkpoint
from laya_mlx.frogger.ui import compose, layout_size


def custom_game(**lanes):
    """A board whose lanes are all safe except the ones given as {row: Lane}."""
    game = FroggerGame(seed=0)
    rows = [Lane("safe", 0, 1, (False,) * game.width) for _ in range(game.height)]
    for row, lane in lanes.items():
        rows[int(row[1:])] = lane
    game._set_lanes(rows)
    return game


def test_lanes_are_cyclic_and_shift_on_their_period():
    lane = Lane("road", 1, 3, (True,) + (False,) * 7)
    assert [lane.moves(t) for t in range(6)] == [False, False, True, False, False, True]
    assert list(lane.row(0)).index(True) == 0
    assert list(lane.row(3)).index(True) == 1
    assert list(lane.row(24)).index(True) == 0  # 8 shifts wrap around a width of 8.


def test_car_water_sweep_and_wall_collisions():
    width = 24
    game = custom_game(r14=Lane("road", 1, 1, (False,) * 11 + (True,) + (False,) * 12))
    game.x, game.y = 11, 15
    assert game.step("UP") is False and game.death_reason == "car"  # Car arrives at x=12.

    game = custom_game(r6=Lane("river", 1, 1, (False,) * width))
    game.x, game.y = 5, 7
    game.step("UP")
    assert game.death_reason == "water"

    game = custom_game(r6=Lane("river", 1, 1, (False,) * (width - 1) + (True,)))
    game.x, game.y = width - 1, 6  # On the log's last cell; the log moves right.
    game.step("WAIT")
    assert game.death_reason == "swept away"

    game = custom_game()
    game.x = 0
    game.step("LEFT")
    assert game.death_reason == "wall"


def test_logs_carry_the_frog_and_home_scores_and_resets():
    game = custom_game(r1=Lane("river", 1, 1, (True,) * 4 + (False,) * 20))
    game.x, game.y = 1, 1
    assert game.step("WAIT") is False and (game.x, game.y) == (2, 1)
    assert game.step("UP") is True
    assert game.score == 1 and (game.x, game.y) == game.start and game.alive


def test_level_rises_every_few_crossings_and_rebuilds_lanes():
    game = FroggerGame(seed=3)
    before = game.lanes
    for _ in range(LEVEL_CROSSINGS):
        game.x, game.y = 0, 1
        game._set_lanes([Lane("safe", 0, 1, (False,) * game.width)] * game.height)
        game.step("UP")
    assert game.level == 2 and game.lanes != before


def test_seed_reproduces_lanes_and_actions():
    first, second = FroggerGame(seed=71), FroggerGame(seed=71)
    for _ in range(200):
        direction = next(m.direction for m in first.moves() if m.safe)
        first.step(direction)
        second.step(direction)
        assert first.snapshot() == second.snapshot()


@pytest.mark.parametrize("seed", range(4))
def test_arbitrary_safe_choices_never_die_and_best_routes_reach_home(seed):
    game = FroggerGame(seed=seed)
    rng = random.Random(seed + 100)
    for _ in range(600):
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        assert safe, game.snapshot()
        routes = [m for m in safe if m.steps_home >= 0]
        best = min(routes, key=lambda m: m.steps_home) if routes else None
        move = best if best and rng.random() < 0.8 else rng.choice(safe)
        game.step(move.direction)
        assert game.alive, game.snapshot()
    assert game.score >= 8 and game.level >= 3


def test_guard_preserves_raw_probabilities_and_reports_intervention():
    game = FroggerGame()
    safe = [m.direction for m in game.moves() if m.safe]
    unsafe = next(d for d in DIRECTIONS if d not in safe)
    probabilities = {d: 0.9 if d == unsafe else 0.025 for d in DIRECTIONS}

    class StubAgent:
        def predict(self, *_):
            return {
                "answers": {
                    "move": {"probabilities": probabilities},
                    "risk": {"noul": 0.97},
                    "home": {"noul": 0.92},
                },
                "usage": {"input_tokens": 100},
            }

    policy = LayaPolicy.__new__(LayaPolicy)
    policy.agent, policy.guarded = StubAgent(), True
    policy.prompt = "compact"
    result = policy.decide(game)
    assert result.proposed == unsafe and result.executed in safe and result.intervened
    assert result.probabilities is probabilities
    assert result.crash_risk == pytest.approx(0.03)
    policy.guarded = False
    assert policy.decide(game).executed == unsafe


@pytest.mark.parametrize("prompt", ["compact", "detailed"])
def test_prompts_describe_every_direction_and_mark_the_planner_choice(prompt):
    policy = LayaPolicy.__new__(LayaPolicy)
    policy.prompt = prompt
    game = FroggerGame()
    state, questions, preferred = policy.questions(game, game.moves())
    assert state and preferred in DIRECTIONS
    criteria = questions["move"]["criteria"]
    assert set(criteria) == set(DIRECTIONS)
    assert "Best" in criteria[preferred]


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
        local_checkpoint("nonexistent-frogger-demo-test/no-cache")
    assert attempts == []


def test_unsupported_boards_are_rejected():
    for kwargs in ({"width": 7}, {"height": 15}):
        with pytest.raises(ValueError):
            FroggerGame(**kwargs)


def test_composition_fits_layout_without_covering_the_footer():
    game = FroggerGame(seed=1)
    decision = {"probabilities": dict.fromkeys(DIRECTIONS, 0.2), "proposed": "UP"}
    canvas = compose(game.snapshot(), decision, {"round": 1})
    assert (canvas.width, canvas.height) == layout_size(game.width, game.height)
    assert "".join(canvas.chars[canvas.height - 3][3:-3]) == "─" * (canvas.width - 6)


def test_scores_above_999_use_a_fourth_digit_instead_of_freezing():
    game = FroggerGame(seed=1)
    row = 6 + game.height + 1 + 3  # Top line of the big SCORE digits.

    def fourth_digit(score):
        canvas = compose({**game.snapshot(), "score": score}, {}, {"best": score})
        return "".join(canvas.chars[row][15:18])

    assert fourth_digit(999).strip() == ""
    assert fourth_digit(1234).strip()


@pytest.mark.parametrize("times", [(1, 1), (2, 1), (float("nan"),), (-1,)])
def test_recording_rejects_invalid_wall_clock_timestamps(tmp_path, times):
    from laya_mlx.frogger.replay import load_record

    path = tmp_path / "record.jsonl"
    events = [{"type": "metadata", "format": "laya-frogger-v1"}]
    events.extend({"type": "frame", "at": t} for t in times)
    path.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="strictly increase"):
        load_record(path)


def test_recording_preserves_actual_timestamps_and_probabilities(tmp_path):
    from laya_mlx.frogger.replay import load_record

    path = tmp_path / "record.jsonl"
    metadata = {"type": "metadata", "format": "laya-frogger-v1"}
    frames = [
        {"type": "frame", "at": 0.019, "decision": {"probabilities": {"UP": 0.8721}}},
        {"type": "frame", "at": 0.119, "decision": {"probabilities": {"UP": 0.0342}}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in [metadata, *frames]))
    assert load_record(path) == (metadata, frames)


@pytest.mark.parametrize("filename", ["frogger-showcase.jsonl", "frogger-fast.jsonl"])
def test_published_real_showcase_replays_every_board_and_action_exactly(filename):
    from laya_mlx.frogger.replay import load_record

    path = Path(__file__).parents[1] / "benchmarks/results" / filename
    metadata, frames = load_record(path)
    settings = metadata["settings"]
    game = FroggerGame(settings["width"], settings["height"], settings["seed"])
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
        assert game.alive
    end = json.loads(path.read_text().splitlines()[-1])
    assert end["game"] == game.snapshot()
    assert end["summary"]["steps"] == end["summary"]["inference_calls"] == len(frames)
    assert end["summary"]["interventions"] == interventions
