"""Token boundaries and embedding cache invalidation preserve inference inputs."""

import copy
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, processors

from laya_mlx import Agent
from laya_mlx.common import _state_token_ids, state_room
from laya_mlx.shortlist import cached_embed_fn, embed_fn_from_agent


@pytest.fixture
def scan_agent(tiny_checkpoint):
    vocab = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "x", "a", "b", " ", "🙂"]
        )
    }
    backend = Tokenizer(models.BPE(vocab, [], unk_token="[UNK]"))
    backend.normalizer = normalizers.Prepend("x")
    backend.decoder = decoders.Fuse()
    backend.save(str(tiny_checkpoint / "tokenizer/tokenizer.json"))
    return Agent(tiny_checkpoint, dtype="float32")


@pytest.mark.parametrize("stride_fraction", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
def test_scan_preserves_original_tokens_and_attribution(scan_agent, stride_fraction, cached):
    agent = scan_agent
    if cached:
        from laya_mlx.prepared import PrefixCache

        agent._prefix_cache = PrefixCache()
    questions = {"q": {"type": "noul", "instructions": "a"}}
    room = state_room(agent.tok, agent._to_internal(questions["q"]), 128, 32)
    step = room // stride_fraction
    text = "a🙂b " * 90
    source = _state_token_ids(agent.tok, text)
    seen = []
    head_length = 128 - room - 1

    def forward(batch):
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            seen.append(ids[head_length : int(mask.sum()) - 1].tolist())
        return np.zeros(batch["marker_pos"].shape), np.zeros((len(batch["input_ids"]), 2))

    agent.forward = forward
    result = agent.predict_long(text, questions, window=room, stride=step)
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + room >= len(source):
            break
        start += step
    assert seen == [source[start : start + room] for start in starts]
    assert result["usage"]["state_tokens_dropped"] == 0
    evidence = result["answers"]["q"]["window"]
    i = evidence["index"]
    assert evidence["token_start"] == starts[i]
    assert evidence["token_end"] == min(starts[i] + room, len(source))


@pytest.mark.parametrize("transform", ["copy", "str", "json", "pickle", "changed"])
def test_scan_hooks_copy_or_replace_window_text(scan_agent, transform):
    seen = []
    convert = {
        "copy": copy.deepcopy,
        "str": str,
        "json": lambda s: json.loads(json.dumps(s)),
        "pickle": lambda s: pickle.loads(pickle.dumps(s)),
        "changed": lambda s: str(s) + "a",
    }[transform]

    def start(ctx):
        original = list(ctx.states)
        ctx.states = [convert(s) for s in original]
        for old, new in zip(original, ctx.states):
            seen.append(_state_token_ids(scan_agent.tok, new))
            if transform != "copy":
                assert seen[-1] == scan_agent.tok(str(new))["input_ids"]
            else:
                assert seen[-1] == list(old.token_ids)

    result = scan_agent.predict_long(
        "a🙂b " * 90,
        {"q": {"type": "noul", "instructions": "a"}},
        window=40,
        stride=40,
        on_predict_start=start,
    )
    assert seen
    assert ("window" not in result["answers"]["q"]) == (transform == "changed")


@pytest.mark.parametrize("limit", [16, 32, 512])
def test_embedding_truncation_preserves_special_tokens_without_changing_shared_tokenizer(
    tiny_checkpoint, limit
):
    agent = Agent(tiny_checkpoint)
    agent.tok.backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    captured = []

    class Encoder:
        config = agent.model.encoder.config

        def __call__(self, input_ids, attention_mask):
            captured.extend(np.asarray(input_ids).tolist())
            return mx.zeros((*input_ids.shape, 1))

    agent.model.encoder = Encoder()
    text = "hello " * 600
    original = agent.tok.backend.encode(text).ids
    embed_fn_from_agent(agent, max_length=limit)([text])
    assert captured == [[2] + [5] * (limit - 2) + [3]]
    assert agent.tok.backend.truncation is None
    assert agent.tok.backend.encode(text).ids == original


@pytest.mark.parametrize("new_dimension", [2, 3])
def test_clearing_cache_rejects_old_inflight_writes(new_dimension):
    entered, release = Event(), Event()
    state = {"version": 1}

    def embed(texts):
        version = state["version"]
        if version == 1:
            entered.set()
            assert release.wait(5)
        width = 2 if version == 1 else new_dimension
        return [[float(version)] * width for _ in texts]

    cached = cached_embed_fn(embed)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(cached, ["option"])
        try:
            assert entered.wait(5)
            state["version"] = 2
            cached.cache_clear()
            assert cached(["option"]).tolist() == [[2.0] * new_dimension]
        finally:
            release.set()
        assert pending.result(5).tolist() == [[1.0, 1.0]]
    assert cached(["option"]).tolist() == [[2.0] * new_dimension]
    assert cached.cache_info()["size"] == 1
