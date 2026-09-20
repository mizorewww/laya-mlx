import json
import os

import pytest

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from laya_mlx.model import DecisionModel, EncoderConfig

    if os.environ.get("LAYA_MLX_TEST_DEVICE") == "cpu":
        mx.set_default_device(mx.cpu)
    has_mlx = True
except ImportError:
    mx = None
    tree_flatten = None
    DecisionModel = EncoderConfig = None
    has_mlx = False

try:
    from tokenizers import Tokenizer, models, pre_tokenizers
except ImportError:
    Tokenizer = models = pre_tokenizers = None


@pytest.fixture
def tiny_checkpoint(tmp_path):
    if not has_mlx or Tokenizer is None:
        pytest.skip("mlx and tokenizers are required for tiny_checkpoint fixture")
    cfg = {
        "model_type": "modernbert",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 1,
        "local_attention": 16,
        "max_position_embeddings": 256,
    }
    agent_cfg = {
        "encoder": "test/tiny",
        "head_layers": 1,
        "max_len": 128,
        "head_max_len": 32,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:2": 1.7},
    }
    path = tmp_path / "checkpoint"
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg))
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            }
        )
    )
    mx.random.seed(7)
    model = DecisionModel(EncoderConfig.from_dict(cfg), agent_cfg)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return path


@pytest.fixture
def questions():
    return {
        "topic": {"type": "choice", "instructions": "Choose", "criteria": ["a", "b", "c"]},
        "level": {"type": "score", "instructions": "Level", "criteria": ["low", "high"]},
        "yes": {"type": "noul", "instructions": "Is this true?"},
    }
