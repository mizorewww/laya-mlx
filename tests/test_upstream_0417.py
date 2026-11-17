"""v0.4.17 boundary regressions that run without native MLX or model weights."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laya_mlx import cli
from laya_mlx.agent import Agent
from laya_mlx.calibrate import fit_abstention_thresholds
from laya_mlx.evals import Dataset, EvalError, Example, evaluate
from laya_mlx.evals_cli import _check_thresholds
from laya_mlx.hooks import calibrate_reliability, validate_timeout
from laya_mlx.revisions import verify_digests
from laya_mlx.shortlist import embed_fn_from_agent

ROOT = Path(__file__).resolve().parents[1]


def test_host_imports_do_not_load_native_or_reference_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from laya_mlx.agent import Agent; "
            "from laya_mlx import cli, shortlist; "
            "Agent._check_question('q', {'type': 'noul', 'instructions': '?'}); "
            "assert not any(n.split('.')[0] in {'mlx', 'torch', 'transformers'} "
            "for n in sys.modules)",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_construction_still_requires_mlx_on_linux():
    if sys.platform != "linux":
        pytest.skip("Linux boundary check; native construction is covered on macOS")
    with pytest.raises(ModuleNotFoundError, match="mlx"):
        Agent("a-checkpoint-that-must-not-be-downloaded")


@pytest.mark.parametrize("option", ["--questions", "--state-file"])
@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_json", "invalid_utf8", "deep"])
def test_native_cli_reports_input_path_before_loading(option, kind, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    if kind == "directory":
        bad.mkdir()
    elif kind != "missing":
        bad.write_bytes({"invalid_json": b"{", "invalid_utf8": b"\xff", "deep": b"[" * 2000}[kind])
    good = tmp_path / "good.json"
    good.write_text("{}")
    args = ["predict", "--questions", str(bad if option == "--questions" else good)]
    args += ["--state", "hello"] if option == "--questions" else ["--state-file", str(bad)]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert option in error and str(bad) in error


def test_native_cli_accepts_bom_in_state_and_questions(tmp_path, monkeypatch, capsys):
    state, questions = {"text": "你好"}, {"q": {"type": "noul", "instructions": "?"}}
    paths = [tmp_path / "state.json", tmp_path / "questions.json"]
    for path, value in zip(paths, [state, questions]):
        path.write_text(json.dumps(value), encoding="utf-8-sig")
    seen = []

    def predict(got_state, got_questions):
        seen.append((got_state, got_questions))
        return {"answers": {}}

    monkeypatch.setattr("laya_mlx.agent.Agent", lambda *a, **kw: SimpleNamespace(predict=predict))
    cli.main(["predict", "--state-file", str(paths[0]), "--questions", str(paths[1])])
    assert seen == [(state, questions)]
    assert json.loads(capsys.readouterr().out) == {"answers": {}}


@pytest.mark.parametrize(
    "agent", [None, object(), SimpleNamespace(tok=object()), SimpleNamespace(model=object())]
)
def test_shortlist_refuses_incomplete_agent_before_importing_mlx(agent):
    with pytest.raises(TypeError, match="agent must provide tok and model.encoder"):
        embed_fn_from_agent(agent)


@pytest.mark.parametrize("name", ["a/b:c", "nested/C:evil", "nested\\C:evil"])
def test_digest_refuses_nested_drive_anchors_before_reading(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe.*path"):
        verify_digests(str(tmp_path), {name: "0" * 64})


@pytest.mark.parametrize("value", ["0.1", None, [], object(), True, False])
def test_abstention_error_names_invalid_target_before_fitting(value):
    with pytest.raises(ValueError, match="target_error"):
        fit_abstention_thresholds([], [1.0, 1.0, 1.0], {}, target_error=value)


@pytest.mark.parametrize("value", [[], {}, object(), 10**400])
def test_hook_timeout_normalizes_non_numeric_and_overflow_errors(value):
    with pytest.raises(ValueError, match="hooks_timeout"):
        validate_timeout(value)


def test_reliability_cutoffs_preserve_numpy_score_precision_and_json_types():
    scores = np.array([0.123456, 0.654321], dtype=np.float32)
    accept, escalate = calibrate_reliability(scores, np.array([False, True]))
    assert accept == float(scores[1])
    assert type(accept) is float and type(escalate) is float
    json.dumps([accept, escalate], allow_nan=False)


@pytest.mark.parametrize("rows", [[], [{"answers": {}}] * 3])
def test_evaluation_refuses_missing_or_extra_batch_results(rows):
    q = {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}
    data = Dataset([Example(state=str(i), questions=q, expected={"q": "a"}) for i in range(2)])
    runner = SimpleNamespace(predict_batch=lambda *args, **kwargs: iter(rows))
    with pytest.raises(EvalError, match="every row needs its own result"):
        evaluate(runner, data, batch_size=2)


def test_explicit_choice_threshold_does_not_accept_noul_accuracy():
    failures = _check_thresholds({"noul_accuracy": 1.0}, {"choice_accuracy": 0.9}, {})
    assert failures and "choice_accuracy" in failures[0]


def _definitions(path):
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            out.update({m.name: m for m in node.body if isinstance(m, ast.FunctionDef)})
    return out


@pytest.mark.parametrize(
    "name",
    [
        "_scan_answer_values",
        "_scan_answer_index",
        "_start_evidence",
        "_decode_answers",
        "predict_long",
    ],
)
def test_changed_host_methods_match_pinned_upstream_ast(name, host_method_ast):
    source = ROOT / ".upstream/laya/agent.py"
    if not source.exists():
        pytest.skip("checkout pinned upstream into .upstream")
    original = _definitions(source)[name]
    native = _definitions(ROOT / "laya_mlx/runtime.py")[name]
    assert host_method_ast(native) == host_method_ast(original)


def _scan_agent():
    """Run the real hook chain with deterministic host-only encoder/forward fixtures."""
    agent = Agent.__new__(Agent)
    agent.cfg = {"max_len": 128, "head_max_len": 20}

    class Tok:
        mask_token = "[MASK]"
        pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    agent.tok = Tok()
    agent.temperature_raw = [1.0, 1.0, 1.0]
    agent._init_host()
    agent._forward = lambda batch: (
        np.zeros((len(batch["input_ids"]), 2)),
        np.full((len(batch["input_ids"]), 2), 0.5),
    )
    agent._decode_answers = lambda logits, act, items, ids, internal, row, **kw: {
        qid: {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9}
        for qid in ids
    }
    return agent


@pytest.mark.parametrize("partial", [False, True])
def test_scan_refuses_missing_caller_answers_after_end_hooks(partial):
    agent = _scan_agent()

    def remove(ctx):
        for row in ctx.results[:1] if partial else ctx.results:
            del row["answers"]["q"]

    with pytest.raises(ValueError, match="answer 'q' is missing from"):
        agent.predict_long(
            "x" * 300,
            {"q": {"type": "noul", "instructions": "?"}},
            window=64,
            stride=32,
            on_predict_end=remove,
        )


def test_scan_allows_end_hook_to_remove_all_start_added_answers():
    agent = _scan_agent()

    def add(ctx):
        ctx.questions = dict(ctx.questions, extra={"type": "noul", "instructions": "?"})

    def remove(ctx):
        for row in ctx.results:
            del row["answers"]["extra"]

    result = agent.predict_long(
        "x" * 300,
        {"q": {"type": "noul", "instructions": "?"}},
        window=64,
        stride=32,
        on_predict_start=add,
        on_predict_end=remove,
    )
    assert set(result["answers"]) == {"q"}
    assert result["answers"]["q"]["window"]["index"] == 0
