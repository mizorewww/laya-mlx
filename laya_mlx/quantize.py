"""Native 4-bit and 8-bit quantization for Laya-MLX.

Enables group-affine weight-only quantization (Q4_0 / Q8_0) using MLX's native
QuantizedLinear primitives. This reduces memory footprint by 2x-4x, slashing memory
bandwidth requirements on memory-constrained devices (such as 8GB/16GB unified memory
Apple Silicon Macs).
"""

import json
import shutil
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from .agent import resolve_model
    from .model import DecisionModel, EncoderConfig, sanitize_weights

    _has_mlx = True
except ImportError:
    mx = nn = tree_flatten = None  # type: ignore[assignment]
    resolve_model = None  # type: ignore[assignment]
    DecisionModel = EncoderConfig = sanitize_weights = None  # type: ignore[assignment, misc]
    _has_mlx = False

SUPPORTED_BITS = (4, 8)
DEFAULT_GROUP_SIZE = 64


def quantize_model(
    model,
    bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
    quantize_heads: bool = False,
) -> None:
    """Quantize linear layers of a DecisionModel in-place.

    By default, only the ModernBERT encoder layers are quantized because they
    comprise >95% of model weights. The small decision heads remain in FP16 to
    preserve strict calibration probabilities.

    Args:
        model: DecisionModel instance.
        bits: Bit-width for quantized weights (4 or 8).
        group_size: Group size for quantization scale factors (typically 64).
        quantize_heads: If True, also quantizes linear layers in the decision heads.
    """
    if not _has_mlx:
        raise RuntimeError("quantize_model requires MLX which is not installed on this system")
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported quantization bits: {bits}. Must be one of {SUPPORTED_BITS}")
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported group_size: {group_size}. Expected 32, 64, or 128")

    def predicate(path: str, module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # ModernBERT layers path check (e.g. layers.0.attn.Wqkv or encoder.layers.0...)
        is_encoder = "layers." in path or "encoder." in path
        if not quantize_heads and not is_encoder:
            return False
        return True

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    mx.eval(model.parameters())


def export_quantized(
    source_path_or_id,
    output_dir,
    *,
    bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
    quantize_heads: bool = False,
    dtype: str = "float16",
    token: str = None,
    subfolder: str = None,
    revision: str = None,
) -> Path:
    """Export an offline quantized MLX checkpoint to disk.

    Creates an optimized model directory containing:
    - model.safetensors (quantized weights)
    - encoder/config.json
    - rl_agent_config.json
    - tokenizer/
    - mlx_config.json (storing quantization metadata)

    Args:
        source_path_or_id: Local path or Hugging Face repository ID.
        output_dir: Destination path for the quantized checkpoint.
        bits: 4 or 8.
        group_size: Quantization group size (default 64).
        quantize_heads: Whether to quantize decision heads.
        dtype: Base unquantized dtype (float16).
        token: Optional HF auth token.
        subfolder: Optional subfolder inside source repository.
        revision: Optional git revision.

    Returns:
        Path to the saved quantized checkpoint.
    """
    if not _has_mlx:
        raise RuntimeError("export_quantized requires MLX which is not installed on this system")
    out = Path(output_dir).expanduser()
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")

    src = resolve_model(source_path_or_id, token=token, subfolder=subfolder, revision=revision)
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    enc_cfg = json.loads((src / "encoder/config.json").read_text())

    model = DecisionModel(EncoderConfig.from_dict(enc_cfg), cfg)
    base_dtype = mx.float16 if dtype == "float16" else mx.float32

    raw_weights = sanitize_weights(mx.load(str(src / "model.safetensors")))
    weights = {k: v.astype(base_dtype) for k, v in raw_weights.items()}
    model.load_weights(list(weights.items()), strict=True)

    # Quantize the model in-place
    quantize_model(model, bits=bits, group_size=group_size, quantize_heads=quantize_heads)

    # Save directory structure
    out.mkdir(parents=True, exist_ok=False)
    (out / "encoder").mkdir()
    (out / "tokenizer").mkdir()

    shutil.copy(src / "rl_agent_config.json", out / "rl_agent_config.json")
    shutil.copy(src / "encoder/config.json", out / "encoder/config.json")
    for item in (src / "tokenizer").iterdir():
        if item.is_file():
            shutil.copy(item, out / "tokenizer" / item.name)

    quantized_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), quantized_weights)

    meta = {
        "model_type": "laya",
        "dtype": dtype,
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "quantize_heads": quantize_heads,
        },
    }
    (out / "mlx_config.json").write_text(json.dumps(meta, indent=2))
    return out
