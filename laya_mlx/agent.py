"""Public MLX inference runtime; prompt and result formats follow upstream Laya."""

import json
import warnings
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union

import numpy as np
from huggingface_hub import snapshot_download

from .common import (
    QTYPES,
    build_sequence,
    encode_text,
    read_temperatures,
    serialize_state,
    uses_parallel_layout,
)
from .prepared import PrefixCache
from .revisions import resolve_revision, snapshot_revision, verify_digests
from .runtime import RuntimeMixin


def __getattr__(name):
    # Importing host methods/signatures must not require a native MLX installation.
    # The public dtype mapping still contains real MLX dtypes, resolved on first use.
    if name == "DTYPES":
        if name in globals():
            return globals()[name]
        import mlx.core as mx

        value = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def resolve_model(
    model_id_or_path, *, token=None, subfolder=None, revision=None, local_files_only=False
):
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
            snapshot_download(
                value,
                token=token,
                revision=revision,
                allow_patterns=patterns,
                local_files_only=local_files_only,
            )
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
    if any("layout" in item for item in items):
        if not all("layout" in item for item in items):
            raise ValueError("Cannot collate mixed option layouts")
        batch["position_ids"] = np.zeros((n, length), dtype=np.int32)
        batch["option_ids"] = np.zeros((n, length), dtype=np.int32)
    for i, item in enumerate(items):
        length, count = len(item["ids"]), len(item["markers"])
        batch["input_ids"][i, :length] = item["ids"]
        batch["attention_mask"][i, :length] = True
        batch["marker_pos"][i, :count] = item["markers"]
        batch["marker_mask"][i, :count] = True
        if "layout" in item:
            for key, values in item["layout"].items():
                batch[key][i, :length] = values
    return batch


def _refuse_vegaml(vegaml):
    """`vegaml=None`/`False` mean the ordinary checkpoint, as upstream; anything else asks for
    vegaml's engine, which only official laya carries."""
    if vegaml is not None and vegaml is not False:
        raise ValueError(
            "vegaml=%r loads vegaml's engine, which requires official laya; this distribution "
            "uses MLX" % (vegaml,)
        )


class Agent(RuntimeMixin):
    @property
    def config(self):
        """Checkpoint metadata for upstream hooks, including option layout."""
        return self.cfg

    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        fast=False,
        compile=False,
        revision=None,
        expected_sha256=None,
        lang_temperatures=None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise=True,
        hooks_concurrent=True,
        hooks_timeout=None,
        calibration=None,
        backend=None,
        compile_warmup=True,
        compile_cache=False,
        compile_mode="default",
        vegaml=None,
        *,
        dtype="float16",
        batch_size=16,
        pad_to_multiple=None,
        cache_prompts=False,
        local_files_only=False,
    ):
        if (
            fast
            or backend not in (None, "auto", "mlx")
            or compile_cache
            or compile_mode != "default"
        ):
            raise ValueError(
                "This distribution uses MLX; PyTorch/CUDA/ONNX backend options require official laya"
            )
        _refuse_vegaml(vegaml)
        import mlx.core as mx

        from .model import DecisionModel, EncoderConfig, sanitize_weights
        from .tokenizer import Tokenizer

        dtypes = __getattr__("DTYPES")
        if device == "mps":
            device = "gpu"
        if dtype not in dtypes:
            raise ValueError(f"dtype must be one of {list(dtypes)}")
        if device not in (None, "gpu", "metal", "cpu"):
            raise ValueError("MLX device must be 'gpu', 'metal', or 'cpu'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = (
            mx.default_device() if device is None else (mx.cpu if device == "cpu" else mx.gpu)
        )
        self.dtype = dtypes[dtype]
        self.batch_size = batch_size
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.pad_to_multiple = pad_to_multiple
        self._prefix_cache = PrefixCache() if cache_prompts else None
        self.model_id = str(model_id_or_path)
        self.model_id_or_path = self.model_id
        self.subfolder = subfolder
        revision = resolve_revision(str(model_id_or_path), revision)
        self.model_dir = resolve_model(
            model_id_or_path,
            token=token,
            subfolder=subfolder,
            revision=revision,
            local_files_only=local_files_only,
        )
        verify_digests(str(self.model_dir), expected_sha256)
        self.revision = snapshot_revision(str(self.model_dir))
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.encoder_cfg = json.loads((self.model_dir / "encoder/config.json").read_text())
        if "encoder" not in self.cfg or "head_layers" not in self.cfg:
            raise ValueError("Laya config must specify encoder and head_layers")
        uses_parallel_layout(self.cfg)
        enc_cfg = EncoderConfig.from_dict(self.encoder_cfg)
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        if not 4 < head_max_len < max_len <= enc_cfg.max_position_embeddings:
            raise ValueError("Expected 4 < head_max_len < max_len <= max_position_embeddings")
        (
            self.temperature,
            self.temperature_by_options,
            self.temperature_raw,
            self.temperature_by_options_raw,
        ) = read_temperatures(self.cfg)
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        with mx.stream(self.device):
            self.model = DecisionModel(enc_cfg, self.cfg)
            weights = sanitize_weights(mx.load(str(self.model_dir / "model.safetensors")))
            weights = {k: v.astype(self.dtype) for k, v in weights.items()}
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())
        # Frozen inference instance: changing weights or module structure requires a new Agent.
        self._inference = mx.compile(self.model) if compile else self.model
        self._init_host(
            lang_temperatures=lang_temperatures,
            calibration=calibration,
            hooks=hooks,
            on_predict_start=on_predict_start,
            on_predict_end=on_predict_end,
            hooks_raise=hooks_raise,
            hooks_concurrent=hooks_concurrent,
            hooks_timeout=hooks_timeout,
        )
        if compile and compile_warmup:
            try:
                self.warmup()
            except Exception as error:
                warnings.warn(
                    "laya-mlx compile warm-up failed (%s: %s); keeping the compiled model. "
                    "Later requests and explicit warmup() calls may still raise."
                    % (type(error).__name__, error),
                    RuntimeWarning,
                )

    def forward(self, batch):
        """Run one prepared batch and return evaluated MLX logits on this agent's device."""
        import mlx.core as mx

        with mx.stream(self.device):
            tensors = {k: mx.array(v) for k, v in batch.items()}
            result = self._inference(**tensors)
            mx.eval(result)
        return result

    def _encode_state(
        self,
        state: Union[str, dict, list],
        ids: List[str],
        internal: Dict[str, Dict],
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> List[Dict]:
        """Tokenize one state against every (already validated + normalized) question.

        `max_len` / `head_max_len` override the agent config for this call (a start hook may set
        `ctx.max_len` / `ctx.head_max_len`).
        """
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len
        # A chronological conversation list is serialized newest-last, so the default
        # right-truncation (st[:room]) would silently drop the newest turn. Truncate
        # from the left for lists so the most recent intent is preserved.
        truncate_left = isinstance(state, list)
        # Tokenize the shared state once. The ids are identical for every question, so
        # re-serializing and re-tokenizing it inside build_sequence per question was pure
        # duplicated work. Tokenize in full and let build_sequence slice per question, so
        # left-truncation for conversation lists keeps its meaning.
        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        items = []
        for qid in ids:
            q = internal[qid]
            cache = getattr(self, "_prefix_cache", None)
            sequence_builder = cache.build_sequence if cache is not None else build_sequence
            seq, markers, stats, state_stats, *layout = sequence_builder(
                self.tok,
                state,
                q,
                max_len,
                head_max_len,
                option_order=q.get("option_order"),
                truncate_left=truncate_left,
                state_ids=state_ids,
                return_stats=True,
                return_truncation_stats=True,
                return_layout=self.parallel_options,
            )
            n_opts = stats["options"]
            if len(markers) != n_opts:
                # The markers are placed at absolute positions and `build_sequence` then drops the
                # ones past `max_len`, so this is about the question fitting in the sequence --
                # `head_max_len` is how much of it the options were given, and `max_len` is the
                # ceiling that dropped them. Naming only `head_max_len` pointed at the wrong knob
                # in both directions: lowering it shortens the option block and can make the
                # call succeed, while raising it makes the overflow worse.
                #
                # The count reported is the markers that SURVIVED, not `len(seq)`: `build_sequence`
                # truncates to `max_len` first, so `len(seq)` is always exactly `max_len` here and
                # would state the ceiling as though it were the requirement.
                raise ValueError(
                    "question %r: only %d of its %d option markers fit in max_len=%d with "
                    "head_max_len=%d spent on the question; lower head_max_len, raise max_len, "
                    "or use fewer options" % (qid, len(markers), n_opts, max_len, head_max_len)
                )
            item = {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["t"]],
                "options": stats,
                "state_stats": state_stats,
            }
            if layout:
                item["layout"] = layout[0]
            items.append(item)
        return items

    @staticmethod
    def _question(qid, definition):
        Agent._check_question(qid, definition)
        return Agent._to_internal(definition)

    def _forward(self, batch):
        width = batch["marker_pos"].shape[1]
        rows = len(batch["input_ids"])
        logits, actions = [], []
        for start in range(0, rows, self.batch_size):
            items = []
            for row in range(start, min(rows, start + self.batch_size)):
                n = int(batch["attention_mask"][row].sum())
                count = int(batch["marker_mask"][row].sum())
                item = {
                    "ids": batch["input_ids"][row, :n],
                    "markers": batch["marker_pos"][row, :count],
                    "qtype": batch["qtype"][row],
                }
                if "option_ids" in batch:
                    item["layout"] = {k: batch[k][row, :n] for k in ("position_ids", "option_ids")}
                items.append(item)
            arrays = collate_items(
                items,
                self.tok.pad_token_id,
                pad_to_multiple=getattr(self, "pad_to_multiple", None),
                max_length=getattr(self, "encoder_cfg", {}).get(
                    "max_position_embeddings", self.cfg.get("max_len", 512)
                ),
            )
            out, act = self.forward(arrays)
            out, act = np.asarray(out), np.asarray(act)
            if not np.isfinite(out).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            # A chunk can have fewer option columns than the logical host batch.
            padded = np.full((len(items), width), -1e4, dtype=out.dtype)
            take = min(width, out.shape[1])
            padded[:, :take] = out[:, :take]
            logits.append(padded)
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            actions.append(act / act.sum(axis=-1, keepdims=True))
        return np.concatenate(logits), np.concatenate(actions)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._inference = None
        self.model = None
        clear_cache()
        return False


RLAgent = Agent


def load(
    model_id_or_path="convaiinnovations/laya",
    device=None,
    token=None,
    subfolder=None,
    fast=False,
    compile=False,
    revision=None,
    expected_sha256=None,
    lang_temperatures=None,
    hooks=None,
    on_predict_start=None,
    on_predict_end=None,
    hooks_raise=True,
    hooks_concurrent=True,
    hooks_timeout=None,
    calibration=None,
    backend=None,
    onnx_path=None,
    compile_warmup=True,
    compile_cache=False,
    compile_mode="default",
    vegaml=None,
    **kwargs,
):
    if onnx_path is not None:
        raise ValueError("ONNX inference requires official laya; this distribution uses MLX")
    _refuse_vegaml(vegaml)
    from .router import DEFAULT_MODELS, canonical_name

    name = canonical_name(str(model_id_or_path))
    if name in DEFAULT_MODELS and not Path(model_id_or_path).expanduser().exists():
        model_id_or_path, bundled_subfolder = DEFAULT_MODELS[name]
        if subfolder is None:
            subfolder = bundled_subfolder
    return Agent(
        model_id_or_path,
        device=device,
        token=token,
        subfolder=subfolder,
        fast=fast,
        compile=compile,
        revision=revision,
        expected_sha256=expected_sha256,
        lang_temperatures=lang_temperatures,
        hooks=hooks,
        on_predict_start=on_predict_start,
        on_predict_end=on_predict_end,
        hooks_raise=hooks_raise,
        hooks_concurrent=hooks_concurrent,
        hooks_timeout=hooks_timeout,
        calibration=calibration,
        backend=backend,
        compile_warmup=compile_warmup,
        compile_cache=compile_cache,
        compile_mode=compile_mode,
        **kwargs,
    )


def clear_cache():
    """Release unused native allocations; live/in-flight arrays remain referenced."""
    import gc
    import sys

    gc.collect()
    # A router used only for host routing or attached test/custom runners has no
    # native allocations. Do not load the backend merely to clear an empty cache.
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        mx.clear_cache()
