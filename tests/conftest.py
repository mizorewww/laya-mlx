import ast
import copy
import json
import os

import pytest


@pytest.fixture
def host_method_ast():
    """Normalize only the two token-preserving expressions when pinning host methods."""
    replacements = {
        "_state_token_ids(self.tok, state)": (
            'encode_text(self.tok, serialize_state(state).replace(self.tok.mask_token, " "), '
            'add_special_tokens=False)["input_ids"]'
        ),
        "_TokenWindow(self.tok, state_ids[i:i + budget])": (
            "self.tok.decode(state_ids[i:i + budget])"
        ),
    }
    expressions = {
        ast.dump(ast.parse(native, mode="eval").body): ast.parse(upstream, mode="eval").body
        for native, upstream in replacements.items()
    }

    class WindowTokens(ast.NodeTransformer):
        def visit_Call(self, node):
            replacement = expressions.get(ast.dump(node))
            return (
                copy.deepcopy(replacement) if replacement is not None else self.generic_visit(node)
            )

    def dump(node):
        node = WindowTokens().visit(copy.deepcopy(node))
        node.decorator_list = []
        return ast.dump(node, include_attributes=False)

    return dump


def pytest_sessionstart(session):
    if os.environ.get("LAYA_MLX_TEST_DEVICE") == "cpu":
        import mlx.core as mx

        mx.set_default_device(mx.cpu)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, models, pre_tokenizers

    from laya_mlx.model import DecisionModel, EncoderConfig

    if os.environ.get("LAYA_MLX_TEST_DEVICE") == "cpu":
        mx.set_default_device(mx.cpu)

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
