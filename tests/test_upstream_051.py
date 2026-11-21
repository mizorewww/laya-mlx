"""v0.5.1 boundary regressions that run without native MLX or model weights."""

import inspect
import sys
import types

import pytest

import laya_mlx
from laya_mlx.agent import Agent, load, wants_vega
from laya_mlx.serve import _resolve_log_level


def test_load_takes_no_checkpoint_named_like_upstream():
    assert inspect.signature(load).parameters["model_id_or_path"].default is None
    assert (
        inspect.signature(Agent.__init__).parameters["model_id_or_path"].default
        == "convaiinnovations/laya"
    )


def test_no_checkpoint_named_stays_on_the_mlx_checkpoint_even_with_vegaml_importable(monkeypatch):
    # Upstream 0.5.0 makes vega the default when vegaml is importable; this distribution has no
    # vegaml engine, so the default is the classic checkpoint, as upstream's own fallback.
    monkeypatch.setitem(sys.modules, "vegaml", types.ModuleType("vegaml"))
    assert wants_vega((), {}) is False
    assert wants_vega((), {"device": "cpu", "hooks": []}) is False
    assert wants_vega(("convaiinnovations/laya",), {}) is False
    assert wants_vega((), {"vegaml": False}) is False
    assert wants_vega((), {"vegaml": True}) is True
    assert wants_vega((), {"vegaml": "4b"}) is True
    assert type(Agent.__new__(Agent)) is Agent


def test_load_with_no_checkpoint_named_builds_the_default_checkpoint(monkeypatch):
    import laya_mlx.agent as la

    built = []

    class Recorder:
        def __init__(self, model_id_or_path, **kwargs):
            built.append((model_id_or_path, kwargs.get("subfolder")))

    monkeypatch.setattr(la, "Agent", Recorder)
    load()
    load(None)
    load("convaiinnovations/laya")
    assert len({entry for entry in built}) == 1, built


@pytest.mark.parametrize("value", [True, "4b"])
def test_load_with_no_checkpoint_named_still_refuses_vegaml(value, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    with pytest.raises(ValueError, match="requires official laya"):
        load(vegaml=value)


def test_learner_is_vegamls_and_is_not_exported_here():
    assert "Learner" not in laya_mlx.__all__
    with pytest.raises(AttributeError):
        laya_mlx.Learner  # noqa: B018


def test_log_level_names_are_validated_before_uvicorn(monkeypatch):
    pytest.importorskip("uvicorn")
    monkeypatch.setenv("LAYA_LOG_LEVEL", "INFO")
    assert _resolve_log_level().lower() == "info"
    monkeypatch.setenv("LAYA_LOG_LEVEL", "verbose")
    with pytest.raises(SystemExit):
        _resolve_log_level()
