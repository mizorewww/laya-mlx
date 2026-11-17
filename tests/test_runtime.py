import json
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from laya_mlx import Agent, Router
from laya_mlx.agent import collate_items, resolve_model
from laya_mlx.common import render_options
from laya_mlx.convert import convert


def test_runtime_does_not_import_torch_or_transformers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import laya_mlx, sys; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_all_primitives_empty_request_and_chunking(tiny_checkpoint, questions):
    agent = Agent(tiny_checkpoint, dtype="float32", batch_size=16)
    together = agent.predict({"text": "hello"}, questions)
    agent.batch_size = 1
    separate = agent.predict({"text": "hello"}, questions)
    assert together == separate
    assert set(together["answers"]) == set(questions)
    assert together["usage"]["output_tokens"] == 0
    assert 0 <= together["answers"]["yes"]["noul"] <= 1
    assert 0 <= together["answers"]["level"]["score"] <= 1
    assert agent.predict("", {})["answers"] == {}
    assert agent.predict("", {})["usage"]["input_tokens"] == 0


def test_single_option_and_long_state(tiny_checkpoint):
    agent = Agent(tiny_checkpoint)
    result = agent.predict(
        "hello " * 1000, {"one": {"type": "choice", "instructions": "choose", "criteria": ["only"]}}
    )
    assert result["answers"]["one"]["probabilities"] == {"only": 1.0}
    assert result["usage"]["input_tokens"] == 128


def test_converted_checkpoint_round_trip_and_no_overwrite(tiny_checkpoint, tmp_path, questions):
    before = Agent(tiny_checkpoint).predict("hello", questions)
    output = convert(tiny_checkpoint, tmp_path / "converted")
    assert json.loads((output / "mlx_config.json").read_text())["dtype"] == "float16"
    assert Agent(output).predict("hello", questions) == before
    with pytest.raises(FileExistsError):
        convert(tiny_checkpoint, output)


def test_missing_and_misshaped_weights_fail_loudly(tiny_checkpoint):
    path = str(tiny_checkpoint / "model.safetensors")
    weights = mx.load(path)
    mx.eval(weights)  # MLX loads lazily; materialize before replacing the source file.
    key = "type_emb.weight"
    weights[key] = mx.zeros((4, 64))
    mx.save_safetensors(path, weights)
    with pytest.raises(ValueError):
        Agent(tiny_checkpoint)
    del weights[key]
    mx.save_safetensors(path, weights)
    with pytest.raises(ValueError):
        Agent(tiny_checkpoint)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "invalid", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": []},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "x", "criteria": {}},
        {"type": "noul", "instructions": "x", "criteria": ["a"]},
        {"type": "noul"},
    ],
)
def test_invalid_questions_rejected(question):
    with pytest.raises(ValueError):
        Agent._check_question("q", question)


def test_structured_criteria_and_mask_injection(tiny_checkpoint):
    q = Agent._to_internal(
        {
            "type": "noul",
            "instructions": {"task": "verify"},
            "criteria": {"false": {"reason": "no"}, "true": {"reason": "yes"}},
        }
    )
    assert render_options(q) == ['false: {"reason": "no"}', 'true: {"reason": "yes"}']
    agent = Agent(tiny_checkpoint)
    items, _ = agent.prepare(
        "[MASK] hello [MASK]", {"x": {"type": "noul", "instructions": "[MASK] true?"}}
    )
    assert items[0]["ids"].count(agent.tok.mask_token_id) == 2
    assert render_options({"t": "choice", "crit": {"zero": 0, "no": False}}) == [
        "zero: 0",
        "no: false",
    ]


def test_collation_never_marks_padding_as_an_option():
    batch = collate_items(
        [
            {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0},
            {"ids": [1, 2], "markers": [1], "qtype": 1},
        ],
        0,
    )
    np.testing.assert_array_equal(batch["marker_mask"], [[True, True], [True, False]])
    np.testing.assert_array_equal(
        batch["attention_mask"], [[True, True, True], [True, True, False]]
    )


def test_cached_prefixes_preserve_inputs_under_mutation_truncation_and_eviction(
    tiny_checkpoint, questions
):
    original = Agent(tiny_checkpoint, dtype="float32")
    cached = Agent(tiny_checkpoint, dtype="float32", cache_prompts=True)
    cached._prefix_cache.capacity = 3
    states = ["", "[MASK] hello", "hello " * 1000, {"text": "你好", "flag": False}]
    for state in states:
        for count in (2, 5, 12):
            questions["topic"]["criteria"] = {str(i): {"value": i} for i in range(count)}
            assert original.prepare(state, questions) == cached.prepare(state, questions)
            assert len(cached._prefix_cache.entries) <= 3
    assert cached.prepare("", {}) == ([], [])


def test_compiled_bucket_path_preserves_predictions_and_handles_shape_changes(
    tiny_checkpoint, questions
):
    original = Agent(tiny_checkpoint, dtype="float32", batch_size=2)
    optimized = Agent(
        tiny_checkpoint,
        dtype="float32",
        batch_size=2,
        compile=True,
        pad_to_multiple=16,
        cache_prompts=True,
    )
    for state in ("hello", "hello " * 70, "hello hello", ""):
        baseline, candidate = (
            original.predict(state, questions),
            optimized.predict(state, questions),
        )
        assert candidate["usage"] == baseline["usage"]
        for qid, answer in baseline["answers"].items():
            other = candidate["answers"][qid]
            if "probabilities" in answer:
                assert other["probabilities"] == pytest.approx(answer["probabilities"], abs=0.0002)
            if "choice" in answer:
                assert other["choice"] == answer["choice"]
            if "noul" in answer:
                assert other["noul"] == pytest.approx(answer["noul"], abs=0.0002)


def test_bucket_padding_is_masked_and_respects_context_limit():
    items = [{"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0}]
    batch = collate_items(items, 9, pad_to_multiple=16, max_length=12)
    assert batch["input_ids"].shape == (1, 12)
    assert batch["input_ids"][0, :3].tolist() == [1, 2, 3]
    assert batch["input_ids"][0, 3:].tolist() == [9] * 9
    assert batch["attention_mask"][0].tolist() == [True] * 3 + [False] * 9


def test_local_path_and_subfolder_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path / "missing")
    with pytest.raises(ValueError):
        resolve_model("test/model", subfolder="../outside")


def test_router_languages_precedence_and_residency():
    router = Router()
    assert router.route("I was charged twice and want a refund").model == "english"
    assert router.route("发票被重复扣款，请退款。").model == "multilingual"
    assert router.route("मुझे धनवापसी चाहिए।").model == "multilingual"
    assert router.route("你好", model="typed").model == "typed-decisions"
    assert router.route("hello", task="typed_decisions").model == "typed-decisions"
    a, b = object(), object()
    router.attach("english", a)
    router.attach("multilingual", b)
    assert router.load("english") is a
    assert router.max_loaded == 2
    router.unload("english")
    assert router.loaded == ["multilingual"]
    router.unload()
    assert router.loaded == []


def test_offline_resolution_reaches_hub_cache_without_coreml_controls(tmp_path, monkeypatch):
    import laya_mlx.agent as module
    from laya_mlx import Router

    folder = tmp_path / "checkpoint"
    (folder / "encoder").mkdir(parents=True)
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json"):
        (folder / name).write_text("{}")
    calls = []

    def cached(repo, **kwargs):
        calls.append((repo, kwargs))
        assert kwargs["local_files_only"] is True
        return str(folder)

    monkeypatch.setattr(module, "snapshot_download", cached)
    assert module.resolve_model("test/offline", local_files_only=True) == folder
    assert calls[0][0] == "test/offline"
    with pytest.raises(TypeError, match="compute_units"):
        Router(compute_units="cpu_gpu")


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_lazy_dtype_export_and_native_construction(dtype, tiny_checkpoint):
    from laya_mlx.agent import DTYPES

    assert DTYPES == {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
    agent = Agent(tiny_checkpoint, dtype=dtype)
    assert agent.dtype == DTYPES[dtype]
    result = agent.predict("hello", {"q": {"type": "noul", "instructions": "?"}})
    assert 0.0 <= result["answers"]["q"]["noul"] <= 1.0


def test_agent_clamps_shipped_temperatures_and_keeps_raw(tiny_checkpoint):
    from laya_mlx.common import TEMP_MIN

    cfg_path = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature_by_options"]["choice:11+"] = 0.10058280825614929
    cfg_path.write_text(json.dumps(cfg))
    with pytest.warns(RuntimeWarning, match="clamping"):
        agent = Agent(tiny_checkpoint)
    assert agent.temperature_by_options["choice:11+"] == TEMP_MIN
    assert agent.temperature_by_options_raw["choice:11+"] == 0.10058280825614929
    assert agent.temperature_by_options["choice:2"] == 1.7  # legitimate value untouched


def test_checkpoint_temperature_wider_than_a_double_loads_with_neutral_fallback(tiny_checkpoint):
    wide = 10**400
    cfg_path = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature"][0] = wide
    cfg["temperature_by_options"]["choice:2"] = wide
    cfg_path.write_text(json.dumps(cfg))
    with pytest.warns(RuntimeWarning, match=str(wide)):
        agent = Agent(tiny_checkpoint, vegaml=False)
    assert agent.temperature == [1.0, 1.1, 2.0]
    assert agent.temperature_by_options == {"choice:2": 1.0}


def test_predict_batch_accepts_index_types_and_no_cap_spellings(tiny_checkpoint, questions):
    # FP32: this checks the chunking spellings, not half-precision padding behavior.
    agent = Agent(tiny_checkpoint, dtype="float32", vegaml=None)
    states = ["hello", "hello hello", "hello hello hello"]
    whole = agent.predict_batch(states, questions)
    for batch_size in (np.int64(2), 0, -1):
        chunked = agent.predict_batch(states, questions, batch_size=batch_size)
        for a, b in zip(whole, chunked):
            for key in questions:
                assert a["answers"][key]["answer_confidence"] == pytest.approx(
                    b["answers"][key]["answer_confidence"], abs=1e-4
                )
