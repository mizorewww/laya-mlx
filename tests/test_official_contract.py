"""Upstream host contracts are the authority; the backend only transports tensors."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from laya_mlx.agent import Agent
from laya_mlx.runtime import RuntimeMixin

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".upstream"


@pytest.fixture(scope="module")
def official():
    if not (UPSTREAM / "laya/agent.py").exists():
        pytest.skip("checkout pinned upstream into .upstream (CI does this)")
    sys.path.insert(0, str(UPSTREAM))
    return importlib.import_module("laya.agent").Agent


@pytest.mark.parametrize(
    "name",
    [
        "_check_question",
        "_to_internal",
        "_encode_state",
        "_decode_answers",
        "predict_long",
        "system_one",
        "decide",
        "decide_batch",
        "fit_temperatures",
        "fit_binning",
        "save_calibration",
        "load_calibration",
    ],
)
def test_host_methods_match_official_ast(official, name, host_method_ast):
    import inspect
    import textwrap

    def tree(method):
        node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        return host_method_ast(node)

    assert tree(getattr(RuntimeMixin, name)) == tree(getattr(official, name))


class Tokenizer:
    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) + 10 for char in text]
        return {"input_ids": ids[:max_length] if truncation else ids}

    def decode(self, ids):
        return "".join(chr(int(i) - 10) for i in ids)


def make_agent(cls, cached=False):
    a = cls.__new__(cls)
    if cls is Agent:
        from laya_mlx.prepared import PrefixCache

        a._prefix_cache = PrefixCache() if cached else None
    a.cfg = {"max_len": 192, "head_max_len": 96}
    a.tok = Tokenizer()
    a.temperature = a.temperature_raw = [1.3, 1.1, 0.8]
    a.temperature_by_options = {"choice:2": 1.2}
    a.lang_temperatures = {}
    a.binning_map = None
    a.shape = {"batch_size": 2, "max_length": 192, "max_options": 32, "flexible": False}
    a.batch_size = 2

    def forward(batch):
        ids = np.asarray(batch["input_ids"])
        mask = np.asarray(batch["attention_mask"])
        pos = np.asarray(batch["marker_pos"])
        z = np.sin((ids * mask).sum(1)[:, None] * 0.01 + pos * 0.19).astype(np.float32)
        z = np.where(np.asarray(batch["marker_mask"]), z, -1e4)
        act = np.tile(np.array([[0.1, 0.3]], np.float32), (len(ids), 1))
        return z, act

    a.forward = forward
    if cls is not Agent:
        a.parallel_options = False

        def official_forward(batch):
            z, act = forward(batch)
            act = np.exp(act - act.max(-1, keepdims=True))
            return z, act / act.sum(-1, keepdims=True)

        a._forward = official_forward
    return a


QUESTIONS = {
    "label": {
        "type": "choice",
        "instructions": "Pick",
        "criteria": [3, False, "other"],
        "option_order": [2, 0, 1],
    },
    "score": {
        "type": "score",
        "instructions": {"任务": "质量"},
        "criteria": [{"x": 0}, ["好"], True],
    },
    "bool": {"type": "noul", "instructions": "Yes?", "labels": {"true": "Y", "false": "N"}},
}


@pytest.mark.parametrize("states", [["hello", "long " * 70, {"你好": [1, True]}], [""], []])
@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize("cached", [False, True])
def test_batch_result_exactly_matches_official(official, states, batch_size, cached):
    ours, upstream = make_agent(Agent, cached=cached), make_agent(official)
    kw = {"batch_size": batch_size, "sort_by_length": True, "min_confidence": 0.7}
    assert ours.predict_batch(states, QUESTIONS, **kw) == upstream.predict_batch(
        states, QUESTIONS, **kw
    )


def test_long_and_structured_match_official(official):
    ours, upstream = make_agent(Agent), make_agent(official)
    for state in ("abc", "abc def " * 110):
        assert ours.predict_long(state, QUESTIONS) == upstream.predict_long(state, QUESTIONS)
    schema = {
        "type": "object",
        "properties": {"kind": {"enum": [1, 2, 3]}, "ok": {"type": "boolean"}},
    }
    assert ours.decide("hello", schema) == upstream.decide("hello", schema)


# MCP cache assertions use a native MLX encoder fixture.
@pytest.mark.parametrize("script", ["mcp"])
def test_original_upstream_assertions(script):
    path = UPSTREAM / "tests" / f"test_{script}.py"
    if not path.exists():
        pytest.skip("checkout pinned upstream into .upstream")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_official_test.py"), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "choice", "instructions": "x", "criteria": [1, 2, False]},
        {"type": "choice", "instructions": "x", "criteria": [1, True]},
        {"type": "choice", "instructions": "x", "criteria": [None, "x"]},
        {"type": "choice", "instructions": "x", "criteria": ["a", "b"], "option_order": [True, 0]},
        {"type": "score", "instructions": "x", "criteria": [None, "x"]},
        {"type": "noul", "instructions": "x", "criteria": {True: "yes", False: "no"}},
        {"type": "noul", "instructions": "x", "labels": {"true": "same", "false": "same"}},
        {"type": [], "instructions": "x"},
        {},
        None,
    ],
)
def test_validation_and_errors_match_official(official, definition):
    def outcome(cls):
        try:
            cls._check_question("question", definition)
            return cls._to_internal(definition)
        except Exception as exc:
            return type(exc).__name__, str(exc)

    assert outcome(Agent) == outcome(official)


def test_hooks_and_calibration_round_trip_match_official(official, tmp_path):
    from laya_mlx.hooks import BaseHook

    def run(cls, path):
        agent = make_agent(cls)
        events = []

        class Hooks(BaseHook):
            def on_predict_start(self, ctx):
                events.append(("start", list(ctx.states)))
                ctx.max_len = 128

            def on_predict_end(self, ctx):
                events.append(("end", ctx.usage, type(ctx.error).__name__))

            def on_error(self, ctx):
                events.append(("error", type(ctx.error).__name__))

        answer = agent.predict_batch(["hi", "word " * 40], QUESTIONS, hooks=[Hooks()])
        with pytest.raises(TypeError):
            agent.predict(None, QUESTIONS, hooks=[Hooks()])
        agent.fit_binning([(0, [1.0, 0.0], [1.0, 0.0], 2)] * 40)
        agent.save_calibration(str(path))
        agent.load_calibration(str(path))
        return answer, events, path.read_text()

    assert run(Agent, tmp_path / "ours.json") == run(official, tmp_path / "official.json")
