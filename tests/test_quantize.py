import json

import pytest

from laya_mlx.quantize import (
    DEFAULT_GROUP_SIZE,
    SUPPORTED_BITS,
)


def test_quantize_constants():
    assert 4 in SUPPORTED_BITS
    assert 8 in SUPPORTED_BITS
    assert DEFAULT_GROUP_SIZE == 64


def test_invalid_quantize_parameters(tiny_checkpoint):
    pytest.importorskip("mlx.core")
    from laya_mlx.agent import Agent
    from laya_mlx.quantize import quantize_model

    agent = Agent(tiny_checkpoint)
    with pytest.raises(ValueError, match="Unsupported quantization bits"):
        quantize_model(agent.model, bits=3)

    with pytest.raises(ValueError, match="Unsupported group_size"):
        quantize_model(agent.model, bits=8, group_size=17)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_in_memory_and_predict(tiny_checkpoint, questions, bits):
    pytest.importorskip("mlx.core")
    from laya_mlx.agent import Agent

    baseline_agent = Agent(tiny_checkpoint, dtype="float32")
    baseline_agent.predict("hello test state", questions)

    quantized_agent = Agent(tiny_checkpoint, dtype="float32", quantize=bits)
    assert quantized_agent.quantize == bits

    quant_result = quantized_agent.predict("hello test state", questions)
    assert set(quant_result["answers"]) == set(questions)
    assert quant_result["usage"]["input_tokens"] > 0

    # Ensure answers remain valid probabilities [0, 1]
    for qid in questions:
        ans = quant_result["answers"][qid]
        if "noul" in ans:
            assert 0.0 <= ans["noul"] <= 1.0
        if "score" in ans:
            assert 0.0 <= ans["score"] <= 1.0
        if "probabilities" in ans:
            for p in ans["probabilities"].values():
                assert 0.0 <= p <= 1.0


def test_export_quantized_checkpoint(tiny_checkpoint, tmp_path, questions):
    pytest.importorskip("mlx.core")
    from laya_mlx.agent import Agent
    from laya_mlx.quantize import export_quantized

    out_dir = tmp_path / "quantized_q4"
    result_path = export_quantized(
        tiny_checkpoint,
        out_dir,
        bits=4,
        group_size=64,
        quantize_heads=False,
    )
    assert result_path == out_dir
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "mlx_config.json").exists()
    assert (out_dir / "rl_agent_config.json").exists()
    assert (out_dir / "encoder/config.json").exists()
    assert (out_dir / "tokenizer").is_dir()

    meta = json.loads((out_dir / "mlx_config.json").read_text())
    assert meta["quantization"]["bits"] == 4
    assert meta["quantization"]["group_size"] == 64
    assert meta["quantization"]["quantize_heads"] is False

    # Check that exporting again to the same folder raises FileExistsError
    with pytest.raises(FileExistsError):
        export_quantized(tiny_checkpoint, out_dir, bits=4)

    # Load the exported checkpoint with Agent
    loaded_agent = Agent(out_dir)
    pred = loaded_agent.predict("hello world", questions)
    assert set(pred["answers"]) == set(questions)


def test_low_memory_mode_clears_cache(tiny_checkpoint, questions):
    pytest.importorskip("mlx.core")
    from laya_mlx.agent import Agent

    agent = Agent(tiny_checkpoint, dtype="float32", low_memory=True)
    assert agent.low_memory is True
    res = agent.predict("testing low memory metal clear", questions)
    assert set(res["answers"]) == set(questions)
