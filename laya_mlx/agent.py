"""Public MLX inference runtime; prompt and result formats follow upstream Laya."""

import json
import math
from pathlib import Path, PurePosixPath

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from .common import QTYPES, build_sequence, confidence_from_probs, render_options, temp_bucket
from .model import DecisionModel, EncoderConfig, sanitize_weights
from .prepared import PrefixCache
from .tokenizer import Tokenizer

DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def resolve_model(model_id_or_path, *, token=None, subfolder=None, revision=None):
    if subfolder:
        part = PurePosixPath(subfolder)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("subfolder must be a relative path inside the model repository")
    path = Path(model_id_or_path).expanduser()
    if not path.exists():
        value = str(model_id_or_path)
        if value.startswith(("/", "./", "../", "~")) or isinstance(model_id_or_path, Path):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        prefix = subfolder.rstrip("/") + "/" if subfolder else ""
        patterns = [
            prefix + name
            for name in (
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/*",
                "mlx_config.json",
            )
        ]
        path = Path(
            snapshot_download(value, token=token, revision=revision, allow_patterns=patterns)
        )
    if subfolder:
        path /= subfolder
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Not a complete Laya checkpoint: {path / name} is missing")
    return path


def collate_items(items, pad_id, *, pad_to_multiple=None, max_length=None):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    n, length = len(items), max(len(item["ids"]) for item in items)
    if pad_to_multiple:
        length = ((length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if max_length is not None:
            length = min(length, max_length)
    count = max(2, max(len(item["markers"]) for item in items))
    batch = {
        "input_ids": np.full((n, length), pad_id, dtype=np.int32),
        "attention_mask": np.zeros((n, length), dtype=np.bool_),
        "marker_pos": np.zeros((n, count), dtype=np.int32),
        "marker_mask": np.zeros((n, count), dtype=np.bool_),
        "qtype": np.array([item["qtype"] for item in items], dtype=np.int32),
    }
    for i, item in enumerate(items):
        length, count = len(item["ids"]), len(item["markers"])
        batch["input_ids"][i, :length] = item["ids"]
        batch["attention_mask"][i, :length] = True
        batch["marker_pos"][i, :count] = item["markers"]
        batch["marker_mask"][i, :count] = True
    return batch


class Agent:
    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        *,
        dtype="float16",
        revision=None,
        batch_size=16,
        compile=False,
        pad_to_multiple=None,
        cache_prompts=False,
        quantize=None,
        low_memory=False,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")
        if device not in (None, "gpu", "metal", "cpu"):
            raise ValueError("MLX device must be 'gpu', 'metal', or 'cpu'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if quantize is not None and quantize not in (4, 8):
            raise ValueError("quantize must be 4, 8, or None")
        self.device = (
            mx.default_device() if device is None else (mx.cpu if device == "cpu" else mx.gpu)
        )
        self.dtype = DTYPES[dtype]
        self.batch_size = batch_size
        self.quantize = quantize
        self.low_memory = low_memory
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.pad_to_multiple = pad_to_multiple
        self._prefix_cache = PrefixCache() if cache_prompts else None
        self.model_id = str(model_id_or_path)
        self.revision = revision
        self.model_dir = resolve_model(
            model_id_or_path, token=token, subfolder=subfolder, revision=revision
        )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.encoder_cfg = json.loads((self.model_dir / "encoder/config.json").read_text())
        if "encoder" not in self.cfg or "head_layers" not in self.cfg:
            raise ValueError("Laya config must specify encoder and head_layers")
        enc_cfg = EncoderConfig.from_dict(self.encoder_cfg)
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        if not 4 < head_max_len < max_len <= enc_cfg.max_position_embeddings:
            raise ValueError("Expected 4 < head_max_len < max_len <= max_position_embeddings")
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        if len(self.temperature) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature, *self.temperature_by_options.values()]
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        with mx.stream(self.device):
            self.model = DecisionModel(enc_cfg, self.cfg)
            mlx_cfg_path = self.model_dir / "mlx_config.json"
            mlx_cfg = json.loads(mlx_cfg_path.read_text()) if mlx_cfg_path.is_file() else {}
            ckpt_quant = mlx_cfg.get("quantization")

            if ckpt_quant:
                from .quantize import quantize_model

                quantize_model(
                    self.model,
                    bits=ckpt_quant["bits"],
                    group_size=ckpt_quant.get("group_size", 64),
                    quantize_heads=ckpt_quant.get("quantize_heads", False),
                )

            weights = sanitize_weights(mx.load(str(self.model_dir / "model.safetensors")))
            if ckpt_quant:
                weights = {
                    k: (
                        v.astype(self.dtype)
                        if v.dtype in (mx.float16, mx.float32, mx.bfloat16)
                        else v
                    )
                    for k, v in weights.items()
                }
            else:
                weights = {k: v.astype(self.dtype) for k, v in weights.items()}

            self.model.load_weights(list(weights.items()), strict=True)

            if quantize is not None and not ckpt_quant:
                from .quantize import quantize_model

                quantize_model(self.model, bits=quantize)

            self.model.eval()
            mx.eval(self.model.parameters())
        # Frozen inference instance: changing weights or module structure requires a new Agent.
        self._inference = mx.compile(self.model) if compile else self.model

    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(self, state, questions):
        """Construct upstream-compatible CPU inputs, useful for parity and profiling."""
        if self._prefix_cache is not None:
            return self._prefix_cache.prepare(self, state, questions)
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def forward(self, batch):
        """Run one prepared batch and return evaluated MLX logits on this agent's device."""
        with mx.stream(self.device):
            tensors = {k: mx.array(v) for k, v in batch.items()}
            result = self._inference(**tensors)
            mx.eval(result)
        return result

    def system_one(self, state, questions):
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
            )
            logits, act = self.forward(batch)
            logits, act = np.asarray(logits), np.asarray(act)
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            act /= act.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid, q = question_ids[start + row], internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                z = logits[row, :k] / max(1e-3, float(scale))
                p = np.exp(z - z.max())
                p /= p.sum()
                answer = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        if self.low_memory:
            metal = getattr(mx, "metal", None)
            if metal and hasattr(metal, "clear_cache"):
                try:
                    metal.clear_cache()
                except Exception:
                    pass
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }

    predict = system_one


RLAgent = Agent


def load(
    model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None, **kwargs
):
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, **kwargs)
