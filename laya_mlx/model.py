"""Native MLX ModernBERT and Laya decision heads (inference only).

Architecture follows Laya and Hugging Face ModernBERT; see NOTICE. No PyTorch
operations or Transformers model classes are used by this implementation.
"""

from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None

    @classmethod
    def from_dict(cls, value: dict):
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in value.items() if k in names})
        if cfg.model_type != "modernbert":
            raise ValueError(f"Unsupported encoder: {cfg.model_type!r}; expected modernbert")
        if cfg.hidden_activation != "gelu":
            raise ValueError(f"Unsupported encoder activation: {cfg.hidden_activation!r}")
        if cfg.hidden_size % cfg.num_attention_heads or cfg.head_dim % 2:
            raise ValueError("ModernBERT requires an even, integral attention head dimension")
        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Invalid ModernBERT layer_types")
        for kind in set(cfg.layer_types):
            params = (cfg.rope_parameters or {}).get(kind, {})
            if params.get("rope_type", "default") != "default":
                raise ValueError("Only default (unscaled) ModernBERT RoPE is supported")
        return cfg

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind):
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def __call__(self, ids):
        return self.norm(self.tok_embeddings(ids))


class EncoderAttention(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.base = cfg.rope_base(kind)
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

    def __call__(self, x, mask):
        b, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class EncoderMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def __call__(self, x):
        value, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.attention_type = cfg.layer_types[index]
        self.attn_norm = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        )
        self.attn = EncoderAttention(cfg, self.attention_type)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg)

    def __call__(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


_SLIDING_WINDOW_CACHE = {}


def get_sliding_window_mask(seq_len: int, window: int) -> mx.array:
    """Return precomputed boolean band mask for local sliding-window attention."""
    key = (seq_len, window)
    mask = _SLIDING_WINDOW_CACHE.get(key)
    if mask is None:
        positions = mx.arange(seq_len)
        mask = (mx.abs(positions[:, None] - positions[None, :]) <= window // 2)[None, None]
        if len(_SLIDING_WINDOW_CACHE) < 64:
            _SLIDING_WINDOW_CACHE[key] = mask
    return mask


def attention_masks(attention_mask, window):
    """Boolean key masks, with inclusive local distance <= local_attention // 2.

    Padded queries can see valid keys to avoid all-masked softmax rows. They are
    never used as keys or pooled outputs, so valid-token results are unchanged.
    """
    valid = attention_mask.astype(mx.bool_)
    full = valid[:, None, None, :]
    local = (get_sliding_window_mask(valid.shape[1], window) | ~valid[:, None, :, None]) & full
    return {"full_attention": full, "sliding_attention": local}


class ModernBert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = [EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def __call__(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        masks = attention_masks(attention_mask, self.config.local_attention)
        for layer in self.layers:
            x = layer(x, masks[layer.attention_type])
        return self.final_norm(x)


class HeadAttention(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.num_heads = max(1, dims // 64)
        if dims % self.num_heads:
            raise ValueError("Decision head dimensions must be divisible by its head count")
        self.head_dim = dims // self.num_heads
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x, mask):
        b, length, _ = x.shape
        qkv = self.in_proj(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class HeadLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.self_attn = HeadAttention(dims)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        # PyTorch TransformerEncoderLayer defaults to ReLU, even though the
        # encoder and scoring heads use GELU.
        return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))


class DecisionHead(nn.Module):
    def __init__(self, dims, count):
        super().__init__()
        self.layers = [HeadLayer(dims) for _ in range(count)]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecisionModel(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, agent_config: dict):
        super().__init__()
        dims = encoder_config.hidden_size
        self.encoder = ModernBert(encoder_config)
        self.head = DecisionHead(dims, agent_config.get("head_layers", 2))
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(dims + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(agent_config.get("act_costs", {})) + 1),
        )
        self.temperature = mx.ones((3,))  # checkpoint buffer; calibration uses the JSON config

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype)[:, None, :]
        h = self.head(h, attention_mask[:, None, None, :].astype(mx.bool_))
        markers = h[mx.arange(h.shape[0])[:, None], mx.maximum(marker_pos, 0)]
        logits = self.scorer(markers).squeeze(-1).astype(mx.float32)
        logits = mx.where(marker_mask, logits, -1e4)
        p = mx.softmax(logits, axis=-1)
        k = mx.maximum(marker_mask.sum(axis=-1), 2).astype(mx.float32)
        entropy = -(p * mx.log(mx.maximum(p, 1e-9))).sum(axis=-1) / mx.log(k)
        # The public runtime pads to at least two marker slots for one-option choices.
        top = mx.sort(p, axis=-1)[:, -2:]
        features = mx.stack([top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0], axis=-1)
        pooled = mx.concatenate([h[:, 0].astype(mx.float32), features], axis=-1)
        action = self.act_head(pooled.astype(self.act_head.layers[0].weight.dtype))
        return logits, action.astype(mx.float32)


def sanitize_weights(weights):
    """Map upstream PyTorch parameter names to MLX; Linear layouts already match."""
    result = {}
    for name, value in weights.items():
        name = name.replace(".in_proj_weight", ".in_proj.weight")
        name = name.replace(".in_proj_bias", ".in_proj.bias")
        for prefix in ("scorer", "act_head"):
            if name.startswith(prefix + ".") and not name.startswith(prefix + ".layers."):
                name = prefix + ".layers." + name[len(prefix) + 1 :]
        if name in result:
            raise ValueError(f"Duplicate checkpoint parameter after conversion: {name}")
        result[name] = value
    return result
