"""Movement-only inference must preserve decisions without inventing metrics."""
import pytest

from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy
from laya_mlx.snake.ui import compose


def test_fast_policy_uses_real_model_and_preserves_movement(tiny_checkpoint):
    full = LayaPolicy(tiny_checkpoint)
    fast = LayaPolicy(tiny_checkpoint, full_metrics=False)
    game = SnakeGame(4, 4, initial_length=4)
    expected, actual = full.decide(game), fast.decide(game)
    assert actual.dead_end_risk is None
    assert actual.food_reachable is None
    assert actual.proposed == expected.proposed
    assert actual.executed == expected.executed
    assert actual.probabilities == pytest.approx(expected.probabilities, abs=0.001)
    assert actual.input_tokens < expected.input_tokens
    assert full.metadata['questions_per_move'] == 3
    assert fast.metadata['questions_per_move'] == 1


def test_uncomputed_metrics_render_as_uncomputed_not_zero():
    text = compose(SnakeGame().snapshot(),
        {'dead_end_risk':None, 'food_reachable':None}, {}).rich_text().plain
    assert text.count('NOT COMPUTED') == 2


def test_full_metrics_keep_their_meaning(tiny_checkpoint):
    result = LayaPolicy(tiny_checkpoint).decide(SnakeGame(4,4,initial_length=4))
    assert 0 <= result.dead_end_risk <= 1
    assert 0 <= result.food_reachable <= 1


def test_round_end_without_a_decision_does_not_invent_zero_metrics():
    game = SnakeGame(4,4,initial_length=15)
    game.step(next(m.direction for m in game.moves() if m.safe))
    assert game.won
    text = compose(game.snapshot(), {}, {}).rich_text().plain
    assert text.count('NOT COMPUTED') == 2
