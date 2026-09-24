import json
import random
import socket
from pathlib import Path

import pytest

from laya_mlx.flappy.game import ACTIONS, MIN_GAP, UNITS, FlappyGame, Pipe
from laya_mlx.flappy.policy import LayaPolicy, local_checkpoint
from laya_mlx.flappy.ui import compose, layout_size


def test_physics_and_boundaries():
    # Positions and speeds are half rows: a jump is an impulse, not a constant climb.
    assert FlappyGame.physics(10, 0, "JUMP") == (7, -3)
    assert FlappyGame.physics(10, -3, "WAIT") == (8, -2)  # Gravity slows the climb.
    assert FlappyGame.physics(10, 4, "WAIT") == (14, 4)  # Fall speed is capped.
    game = FlappyGame(seed=0)
    assert game.collision(-1, []) == "ceiling"
    assert game.collision(game.height * UNITS, []) == "ground"
    pipe = Pipe(game.bird_x, 5, game.gap)
    assert game.collision(4 * UNITS + 1, [pipe]) == "pipe"
    assert game.collision(5 * UNITS, [pipe]) is None
    assert game.collision((5 + game.gap) * UNITS, [pipe]) == "pipe"


def test_jump_follows_an_arc_and_not_jumping_falls():
    y, velocity, path = 20, 0, []
    for action in ("JUMP", "WAIT", "WAIT", "WAIT", "WAIT", "WAIT"):
        y, velocity = FlappyGame.physics(y, velocity, action)
        path.append(y)
    assert path == [17, 15, 14, 14, 15, 17]  # Up fast, slows, apex, then accelerates down.


def test_crash_ends_the_round():
    game = FlappyGame(seed=0)
    while game.alive:
        game.step("JUMP")
    assert game.death_reason in ("ceiling", "pipe")
    with pytest.raises(RuntimeError):
        game.step("WAIT")
    assert game.actions() == []


@pytest.mark.parametrize("seed", range(20))
def test_arbitrary_survivable_choices_never_crash(seed):
    game = FlappyGame(seed=seed)
    rng = random.Random(seed + 100)
    for _ in range(1500):
        allowed = [a.action for a in game.actions() if a.survivable]
        assert allowed, game.snapshot()
        assert game.step(rng.choice(allowed)), game.snapshot()
    assert game.score > 100


def test_pipes_stay_ahead_and_score_counts_passes():
    game = FlappyGame(seed=3)
    first = game.next_pipe()
    for _ in range(first.x - game.bird_x + 2):
        game.step(next(a.action for a in game.actions() if a.survivable))
    assert game.score == 1
    assert max(p.x for p in game.pipes) > game.width


@pytest.mark.parametrize("height", [10, 20, 30])
def test_composition_fits_layout_without_covering_the_footer(height):
    game = FlappyGame(height=height, seed=1)
    game.score = 888
    decision = {"probabilities": dict.fromkeys(ACTIONS, 0.5), "proposed": "JUMP"}
    canvas = compose(game.snapshot(), decision, {"round": 1, "best": 888})
    assert (canvas.width, canvas.height) == layout_size(game.width, game.height)
    assert "".join(canvas.chars[canvas.height - 3][3:-3]) == "─" * (canvas.width - 6)


def test_seed_reproduces_pipes_and_actions():
    first, second = FlappyGame(seed=71), FlappyGame(seed=71)
    for _ in range(300):
        action = next(a.action for a in first.actions() if a.survivable)
        first.step(action)
        second.step(action)
        assert first.snapshot() == second.snapshot()


def test_gaps_reach_the_ceiling_and_the_floor_and_difficulty_rises():
    game = FlappyGame(seed=5)
    rng = random.Random(5)
    seen = {}
    for _ in range(2500):
        for pipe in game.pipes:
            seen[pipe.x + game.ticks] = pipe
        game.step(rng.choice([a.action for a in game.actions() if a.survivable]))
    assert any(p.gap_top == 0 for p in seen.values())
    assert any(p.gap_top + p.gap == game.height for p in seen.values())
    assert game.level > 3
    assert min(p.gap for p in seen.values()) == MIN_GAP
    starts = sorted(seen)
    assert min(b - a for a, b in zip(starts, starts[1:])) < game.spacing


def test_guard_preserves_raw_probabilities_and_reports_intervention():
    game = FlappyGame()
    while all(a.survivable for a in game.actions()):
        game.step("JUMP")
    safe = [a.action for a in game.actions() if a.survivable]
    unsafe = next(a for a in ACTIONS if a not in safe)
    probabilities = {a: 0.9 if a == unsafe else 0.1 for a in ACTIONS}

    class StubAgent:
        def predict(self, *_):
            return {
                "answers": {
                    "move": {"probabilities": probabilities},
                    "risk": {"noul": 0.97},
                    "clear": {"noul": 0.92},
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
def test_prompts_describe_every_action_and_mark_the_planner_choice(prompt):
    policy = LayaPolicy.__new__(LayaPolicy)
    policy.prompt = prompt
    game = FlappyGame()
    state, questions, preferred = policy.questions(game, game.actions())
    assert state and preferred in ACTIONS
    criteria = questions["move"]["criteria"]
    assert set(criteria) == set(ACTIONS)
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
        local_checkpoint("nonexistent-flappy-demo-test/no-cache")
    assert attempts == []


def test_small_boards_are_rejected():
    for kwargs in ({"gap": MIN_GAP - 1}, {"height": 7}, {"width": 12}):
        with pytest.raises(ValueError):
            FlappyGame(**kwargs)


@pytest.mark.parametrize("times", [(1, 1), (2, 1), (float("nan"),), (-1,)])
def test_recording_rejects_invalid_wall_clock_timestamps(tmp_path, times):
    from laya_mlx.flappy.replay import load_record

    path = tmp_path / "record.jsonl"
    events = [{"type": "metadata", "format": "laya-flappy-v1"}]
    events.extend({"type": "frame", "at": t} for t in times)
    path.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="strictly increase"):
        load_record(path)


def test_recording_preserves_actual_timestamps_and_probabilities(tmp_path):
    from laya_mlx.flappy.replay import load_record

    path = tmp_path / "record.jsonl"
    metadata = {"type": "metadata", "format": "laya-flappy-v1"}
    frames = [
        {"type": "frame", "at": 0.019, "decision": {"probabilities": {"JUMP": 0.8721}}},
        {"type": "frame", "at": 0.119, "decision": {"probabilities": {"JUMP": 0.0342}}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in [metadata, *frames]))
    assert load_record(path) == (metadata, frames)


@pytest.mark.parametrize("filename", ["flappy-showcase.jsonl", "flappy-fast.jsonl"])
def test_published_real_showcase_replays_every_board_and_action_exactly(filename):
    from laya_mlx.flappy.replay import load_record

    path = Path(__file__).parents[1] / "benchmarks/results" / filename
    metadata, frames = load_record(path)
    settings = metadata["settings"]
    game = FlappyGame(settings["width"], settings["height"], settings["seed"], settings["gap"])
    interventions = 0
    for frame in frames:
        assert frame["game"] == game.snapshot()
        decision = frame["decision"]
        probabilities = decision["probabilities"]
        assert set(probabilities) == set(ACTIONS)
        assert sum(probabilities.values()) == pytest.approx(1, abs=0.00021)
        assert decision["proposed"] == max(ACTIONS, key=probabilities.__getitem__)
        safe = [a.action for a in game.actions() if a.survivable]
        assert decision["executed"] in safe
        interventions += decision["intervened"]
        assert frame["stats"]["interventions"] == interventions
        assert game.step(decision["executed"])
    end = json.loads(path.read_text().splitlines()[-1])
    assert end["game"] == game.snapshot()
    assert end["summary"]["steps"] == end["summary"]["inference_calls"] == len(frames)
    assert end["summary"]["interventions"] == interventions


def test_scores_above_999_use_a_fourth_digit_instead_of_freezing():
    game = FlappyGame(seed=1)
    row = 6 + game.height + 1 + 3  # Top line of the big SCORE digits.

    def fourth_digit(score):
        canvas = compose({**game.snapshot(), "score": score}, {}, {"best": score})
        return "".join(canvas.chars[row][15:18])

    assert fourth_digit(999).strip() == ""
    assert fourth_digit(1234).strip()  # "4" is drawn; the display does not stop at 999.
