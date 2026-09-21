"""Native MLX CLIP, used to turn an image into text evidence for a Laya typed decision.

Architecture and parameter names follow Hugging Face `CLIPModel`; see NOTICE. No PyTorch
operations or Transformers model classes are used.
"""

import io
import json
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from .agent import DTYPES

DEFAULT_CLIP = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
PROMPT = "a photo of a %s."
TEXT_CACHE_SIZE = 64
ACTS = {
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "quick_gelu": lambda x: x * mx.sigmoid(1.702 * x),
}

SCENE_LABELS = (
    "person face crowd child animal dog cat bird horse fish insect flower tree plant grass fruit "
    "vegetable food meal drink car bicycle motorcycle bus train airplane boat building house "
    "bridge road city beach mountain forest sky water snow fire book document screenshot chart "
    "computer phone keyboard furniture chair table clothing shoe bag tool machine toy ball sign "
    "map painting logo"
).split()


class Attention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x, mask=None):
        b, n, d = x.shape
        shape = (b, n, self.heads, -1)
        q = self.q_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, dims, hidden, act):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden)
        self.fc2 = nn.Linear(hidden, dims)
        self._act = act

    def __call__(self, x):
        return self.fc2(self._act(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims, eps = cfg["hidden_size"], cfg.get("layer_norm_eps", 1e-5)
        self.layer_norm1 = nn.LayerNorm(dims, eps=eps)
        self.self_attn = Attention(dims, cfg["num_attention_heads"])
        self.layer_norm2 = nn.LayerNorm(dims, eps=eps)
        self.mlp = MLP(dims, cfg["intermediate_size"], _activation(cfg))

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = [Layer(cfg) for _ in range(cfg["num_hidden_layers"])]

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TextEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.position_embedding = nn.Embedding(cfg["max_position_embeddings"], cfg["hidden_size"])

    def __call__(self, ids):
        return self.token_embedding(ids) + self.position_embedding.weight[: ids.shape[1]]


class TextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = TextEmbeddings(cfg)
        self.encoder = Encoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])

    def __call__(self, ids):
        x = self.embeddings(ids)
        mask = mx.triu(mx.full((ids.shape[1], ids.shape[1]), -mx.inf), k=1).astype(x.dtype)
        x = self.final_layer_norm(self.encoder(x, mask))
        # Pooled token is the end-of-text marker, which holds the highest id in the vocabulary.
        return x[mx.arange(ids.shape[0]), ids.argmax(axis=-1)]


class VisionEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims, patch = cfg["hidden_size"], cfg["patch_size"]
        positions = (cfg["image_size"] // patch) ** 2 + 1
        self.class_embedding = mx.zeros((dims,))
        self.patch_embedding = nn.Conv2d(3, dims, patch, stride=patch, bias=False)
        self.position_embedding = nn.Embedding(positions, dims)

    def __call__(self, x):
        patches = self.patch_embedding(x)
        patches = patches.reshape(x.shape[0], -1, patches.shape[-1])
        cls = mx.broadcast_to(self.class_embedding, (x.shape[0], 1, patches.shape[-1]))
        return mx.concatenate([cls, patches], axis=1) + self.position_embedding.weight


class VisionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.get("layer_norm_eps", 1e-5)
        self.embeddings = VisionEmbeddings(cfg)
        self.pre_layrnorm = nn.LayerNorm(cfg["hidden_size"], eps=eps)
        self.encoder = Encoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg["hidden_size"], eps=eps)

    def __call__(self, x):
        x = self.encoder(self.pre_layrnorm(self.embeddings(x)))
        return self.post_layernorm(x[:, 0])


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        text, vision = config["text_config"], config["vision_config"]
        dims = config["projection_dim"]
        self.text_model = TextModel(text)
        self.vision_model = VisionModel(vision)
        self.visual_projection = nn.Linear(vision["hidden_size"], dims, bias=False)
        self.text_projection = nn.Linear(text["hidden_size"], dims, bias=False)
        self.logit_scale = mx.zeros(())


def _activation(cfg):
    name = cfg.get("hidden_act", "quick_gelu")
    if name not in ACTS:
        raise ValueError(f"Unsupported CLIP activation: {name!r}")
    return ACTS[name]


def sanitize_weights(weights):
    """Drop non-parameter buffers and convert patch convolutions to MLX's NHWC layout."""
    out = {}
    for key, value in weights.items():
        if key.endswith("position_ids"):
            continue
        if key.endswith("patch_embedding.weight"):
            value = value.transpose(0, 2, 3, 1)
        out[key] = value
    return out


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Scorer:
    """CLIP zero-shot scoring: one image against named labels, as calibration-free softmax."""

    def __init__(self, model_id=DEFAULT_CLIP, *, dtype="float16", device=None, token=None):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")
        if device not in (None, "gpu", "metal", "cpu"):
            raise ValueError("MLX device must be 'gpu', 'metal', or 'cpu'")
        self.device = (
            mx.default_device() if device is None else (mx.cpu if device == "cpu" else mx.gpu)
        )
        path = Path(model_id).expanduser()
        if not path.exists():
            path = Path(
                snapshot_download(
                    str(model_id), token=token, allow_patterns=["*.json", "model.safetensors"]
                )
            )
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "clip":
            raise ValueError(f"Not a CLIP checkpoint: model_type={config.get('model_type')!r}")
        preproc = json.loads((path / "preprocessor_config.json").read_text())
        self.image_size = config["vision_config"]["image_size"]
        self.context = config["text_config"]["max_position_embeddings"]
        self._text_cache = {}
        self.image_mean = np.array(preproc["image_mean"], dtype=np.float32)
        self.image_std = np.array(preproc["image_std"], dtype=np.float32)
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        with mx.stream(self.device):
            self.model = CLIP(config)
            weights = sanitize_weights(mx.load(str(path / "model.safetensors")))
            weights = {k: v.astype(DTYPES[dtype]) for k, v in weights.items()}
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())

    def preprocess(self, image):
        """Resize the shortest side, center crop and normalize into one NHWC array."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        size = self.image_size
        width, height = image.size
        scale = size / min(width, height)
        image = image.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))), Image.BICUBIC
        )
        width, height = image.size
        left, top = (width - size) // 2, (height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return ((pixels - self.image_mean) / self.image_std)[None]

    def embed_image(self, image):
        with mx.stream(self.device):
            pixels = mx.array(self.preprocess(image)).astype(self.model.logit_scale.dtype)
            out = self.model.visual_projection(self.model.vision_model(pixels))
            mx.eval(out)
        return _normalize(np.asarray(out, dtype=np.float32))[0]

    def embed_text(self, texts):
        """Embed and L2-normalize label texts. Repeated label sets are served from a cache."""
        key = tuple(texts)
        if key in self._text_cache:
            return self._text_cache[key]
        ids = np.zeros((len(key), self.context), dtype=np.int32)
        for row, encoding in enumerate(self.tok.encode_batch(list(key))):
            tokens = encoding.ids[: self.context]
            # Truncation must keep the end-of-text token: it is the pooled position.
            tokens[-1] = encoding.ids[-1]
            ids[row, : len(tokens)] = tokens
        with mx.stream(self.device):
            out = self.model.text_projection(self.model.text_model(mx.array(ids)))
            mx.eval(out)
        embeddings = _normalize(np.asarray(out, dtype=np.float32))
        if len(self._text_cache) >= TEXT_CACHE_SIZE:
            # ponytail: clears wholesale when full; make it an LRU if label sets ever churn.
            self._text_cache.clear()
        self._text_cache[key] = embeddings
        return embeddings

    def rank(self, image_embedding, labels, prompt=PROMPT):
        """Return [(label, probability)] over `labels`, most likely first."""
        labels = list(labels)
        if not labels:
            raise ValueError("Cannot rank an empty label set")
        text = self.embed_text([prompt % label for label in labels])
        with mx.stream(self.device):
            scale = float(mx.exp(self.model.logit_scale.astype(mx.float32)))
        logits = scale * (text @ image_embedding)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return sorted(zip(labels, probs.tolist()), key=lambda pair: -pair[1])


@lru_cache(maxsize=2)
def load_scorer(model_id=DEFAULT_CLIP, dtype="float16", device=None):
    return Scorer(model_id, dtype=dtype, device=device)


def question_labels(qdef):
    """Return the visually rankable option labels of one question, or None."""
    if not isinstance(qdef, dict):
        return None
    criteria = qdef.get("criteria")
    if qdef.get("type") == "choice" and isinstance(criteria, (dict, list)) and criteria:
        return [label for label in criteria if isinstance(label, str)] or None
    if qdef.get("type") == "score" and isinstance(criteria, list) and criteria:
        return [str(level) for level in criteria]
    return None


def _percentages(scores):
    return ", ".join("%s (%d%%)" % (label, round(p * 100)) for label, p in scores)


def describe_image(image, questions=None, *, scorer=None, labels=None, top_k=5, prompt=PROMPT):
    """Render CLIP zero-shot scores for one image as prose Laya can read as state.

    Prose, not a score table: at identical CLIP inputs the sentence form measurably improved
    `noul` answers on the published checkpoints.
    """
    scorer = scorer or load_scorer()
    embedding = scorer.embed_image(image)
    scene = scorer.rank(embedding, labels or SCENE_LABELS, prompt)[:top_k]
    parts = ["This is a photograph. The image shows: %s." % _percentages(scene)]
    for qid, qdef in (questions or {}).items():
        options = question_labels(qdef)
        if options:
            title = qdef.get("instructions") or qid
            parts.append(
                "Visual match for %r: %s."
                % (title, _percentages(scorer.rank(embedding, options, prompt)))
            )
    return " ".join(parts)


def image_state(image, questions=None, state=None, **kwargs):
    """Combine CLIP image evidence with an optional text or structured state.

    Evidence is placed first so it survives Laya's right-truncation of long states.
    """
    evidence = describe_image(image, questions, **kwargs)
    if state is None:
        return evidence
    if isinstance(state, str):
        return evidence + "\n\n" + state
    return {"image": evidence, "state": state}
