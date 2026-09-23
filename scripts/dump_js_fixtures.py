"""Dump golden fixtures for the laya-js packages (https://github.com/johnhenry/laya-js).

Writes:
  <js>/packages/tensor-backend/fixtures/ops.json      per-op MLX fp32 reference cases
  <out>/tiny/                                          tiny random checkpoint + per-stage activations
  <out>/real/<model>.json                              16 parity cases x 3 checkpoints: ids, logits, results
  <out>/tables/*.json                                  pyjson / lang / tokenizer / email tables

Usage: uv run python scripts/dump_js_fixtures.py --out ../laya-js/packages/laya-fixtures/data
       [--skip-real] [--models english,multilingual,typed-decisions]
"""

import argparse
import base64
import json
import math
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer as RustTokenizer
from tokenizers import models, pre_tokenizers

from benchmarks.common import parity_cases
from laya_mlx import lang
from laya_mlx.agent import Agent, collate_items
from laya_mlx.common import render_options
from laya_mlx.email import clean_email_body
from laya_mlx.model import DecisionModel, EncoderConfig, attention_masks
from laya_mlx.shortlist import embed_fn_from_agent

mx.set_default_device(mx.gpu)  # Metal fp32 reference (MLX CPU JIT needs a working clang toolchain)

REPOS = {
    "english": "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx",
    "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}


def enc(a, dtype=None):
    a = np.asarray(a)
    if dtype is None:
        dtype = "bool" if a.dtype == np.bool_ else "i32" if a.dtype.kind in "iu" else "f32"
    np_dtype = {"bool": np.uint8, "i32": np.int32}.get(dtype, np.float32)
    data = np.ascontiguousarray(a.astype(np_dtype))
    return {"dtype": dtype, "shape": list(a.shape), "b64": base64.b64encode(data.tobytes()).decode()}


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=None, separators=(",", ":")))
    print("wrote", path, "%.1f KiB" % (path.stat().st_size / 1024))


# ----------------------------------------------------------------------------- ops
def op_cases():
    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    cases = []

    def add(name, op, inputs, outputs, args=None, atol=1e-5, rtol=1e-4, f32_only=False):
        outs = outputs if isinstance(outputs, list) else [outputs]
        cases.append(
            {
                "name": name,
                "op": op,
                "args": args or {},
                "inputs": [None if x is None else enc(np.asarray(x)) for x in inputs],
                "outputs": [enc(np.asarray(o)) for o in outs],
                "atol": atol,
                "rtol": rtol,
                **({"f32Only": True} if f32_only else {}),
            }
        )

    A = mx.array
    for op, fn in [("add", mx.add), ("sub", mx.subtract), ("mul", mx.multiply), ("maximum", mx.maximum)]:
        a, b = r(2, 3, 4), r(3, 1)
        add(f"{op}/broadcast", op, [a, b], fn(A(a), A(b)))
        add(f"{op}/same", op, [a, a[::-1].copy()], fn(A(a), A(a[::-1].copy())))
    a, b = r(2, 5), np.abs(r(1, 5)) + 0.5
    add("div/broadcast", "div", [a, b], A(a) / A(b))
    c = rng.random((2, 1, 4)) > 0.5
    x, y = r(2, 3, 4), r(1, 3, 1)
    add("where/broadcast", "where", [c, x, y], mx.where(A(c), A(x), A(y)))
    x = r(4, 7)
    add("scale", "scale", [x], A(x) * 0.125, {"s": 0.125})
    add("exp", "exp", [x], mx.exp(A(x)))
    add("log", "log", [np.abs(x) + 1e-3], mx.log(A(np.abs(x) + 1e-3)))
    add("relu", "relu", [x], nn.relu(A(x)))
    xg = r(3, 64) * 3
    add("gelu/erf", "gelu", [xg], nn.gelu(A(xg)))
    for axis in (0, 1, -1):
        x = r(3, 5, 6)
        add(f"sum/axis{axis}", "sum", [x], mx.sum(A(x), axis=axis), {"axis": axis, "keepDims": False})
        add(f"max/axis{axis}", "max", [x], mx.max(A(x), axis=axis, keepdims=True), {"axis": axis, "keepDims": True})
    x = r(4, 9) * 5
    add("softmax/last", "softmax", [x], mx.softmax(A(x), axis=-1), {"axis": -1})
    add("softmax/axis0", "softmax", [x], mx.softmax(A(x), axis=0), {"axis": 0})
    xm = x.copy()
    xm[:, 5:] = -1e4  # masked logits as in DecisionModel
    add("softmax/masked", "softmax", [xm], mx.softmax(A(xm), axis=-1), {"axis": -1})
    add("sort/last", "sort", [x], mx.sort(A(x), axis=-1), {"axis": -1})
    a, b = r(2, 3, 5, 7), r(2, 3, 7, 4)
    add("matmul/batched", "matmul", [a, b], A(a) @ A(b), atol=1e-4)
    a, b = r(2, 3, 5, 7), r(7, 4)
    add("matmul/broadcast", "matmul", [a, b], A(a) @ A(b), atol=1e-4)
    x, w, bias = r(2, 5, 48), r(40, 48), r(40)
    add("linear/bias", "linear", [x, w, bias], A(x) @ A(w).T + A(bias), atol=1e-4)
    add("linear/nobias", "linear", [x, w, None], A(x) @ A(w).T, atol=1e-4)
    x, w, bias = r(2, 5, 64) * 2 + 1, r(64), r(64)
    for name, ww, bb in [("wb", w, bias), ("w", w, None), ("none", None, None)]:
        ln = mx.fast.layer_norm(A(x), None if ww is None else A(ww), None if bb is None else A(bb), 1e-5)
        add(f"layerNorm/{name}", "layerNorm", [x, ww, bb], ln, {"eps": 1e-5}, atol=1e-4)
    table, ids = r(50, 16), rng.integers(0, 50, (2, 7)).astype(np.int32)
    add("embedding", "embedding", [table, ids], A(table)[A(ids)])
    x, idx = r(3, 10, 8), rng.integers(0, 10, (3, 4)).astype(np.int32)
    add("gatherRows", "gatherRows", [x, idx], A(x)[mx.arange(3)[:, None], A(idx)])
    for base in (10000.0, 160000.0):
        x = r(2, 3, 20, 16)
        out = mx.fast.rope(A(x), 16, traditional=False, base=base, scale=1.0, offset=0)
        add(f"rope/base{int(base)}", "rope", [x], out, {"base": base}, atol=1e-4)
    B, H, L, D = 2, 3, 40, 16
    q, k, v = r(B, H, L, D), r(B, H, L, D), r(B, H, L, D)
    valid = np.ones((B, L), bool)
    valid[1, 29:] = False
    masks = attention_masks(A(valid), 16)
    for kind in ("full_attention", "sliding_attention"):
        m = np.asarray(masks[kind])
        out = mx.fast.scaled_dot_product_attention(A(q), A(k), A(v), scale=D**-0.5, mask=A(m))
        add(f"sdpa/{kind}", "sdpa", [q, k, v, m], out, {"scale": D**-0.5}, atol=1e-4)
    out = mx.fast.scaled_dot_product_attention(A(q), A(k), A(v), scale=D**-0.5, mask=None)
    add("sdpa/nomask", "sdpa", [q, k, v, None], out, {"scale": D**-0.5}, atol=1e-4)
    x = r(2, 3, 4, 5)
    add("reshape", "reshape", [x], x.reshape(6, 20), {"shape": [6, 20]})
    add("transpose/0213", "transpose", [x], x.transpose(0, 2, 1, 3), {"perm": [0, 2, 1, 3]})
    add("slice", "slice", [x], x[:, 1:3, :, 2:5], {"begin": [0, 1, 0, 2], "end": [2, 3, 4, 5]})
    s = r(2, 3, 12)
    add("split/last", "split", [s], list(np.split(s, 2, axis=-1)), {"parts": 2, "axis": 2})
    s3 = r(2, 6, 4)
    add("split/axis1x3", "split", [s3], list(np.split(s3, 3, axis=1)), {"parts": 3, "axis": 1})
    a, b = r(2, 3), r(2, 5)
    add("concat/last", "concat", [a, b], np.concatenate([a, b], axis=-1), {"axis": -1})
    ids = rng.integers(-5, 5, (3, 4)).astype(np.int32)
    add("cast/i32->f32", "cast", [ids], ids.astype(np.float32), {"dtype": "f32"}, f32_only=True)
    add("cast/bool->f32", "cast", [c], c.astype(np.float32), {"dtype": "f32"}, f32_only=True)
    x = r(2, 5, 2 * 24) * 2
    value, gate = mx.split(A(x), 2, axis=-1)
    add("geglu", "geglu", [x], nn.gelu(value) * gate)
    x, m = r(2, 6, 8), np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], bool)
    add("meanPool", "meanPool", [x, m], (x * m[:, :, None]).sum(1) / m.sum(1, keepdims=True))
    return cases


# ----------------------------------------------------------------------------- tiny model
TINY_CFG = {
    "model_type": "modernbert",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,  # layers 0 and 3 full, 1-2 sliding
    "num_attention_heads": 2,
    "local_attention": 16,
    "max_position_embeddings": 256,
}
TINY_AGENT = {
    "encoder": "test/tiny",
    "head_layers": 2,
    "max_len": 128,
    "head_max_len": 32,
    "act_costs": {"escalate": 0.5},
    "temperature": [1.3, 1.1, 2.0],
    "temperature_by_options": {"choice:2": 1.7, "choice:11+": 0.1},
}


def dump_tiny(out: Path):
    path = out / "tiny"
    if path.exists():
        shutil.rmtree(path)
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(TINY_CFG, indent=2))
    (path / "rl_agent_config.json").write_text(json.dumps(TINY_AGENT, indent=2))
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(100)] + [
        "hello", "choice", "score", "noul", "question", ":", "a", "b", "c", "true", "false",
        "level", "0", "1", "yes", "no", "the", "statement", "does", "not", "hold", "holds", ",",
    ]
    vocab = {t: i for i, t in enumerate(words[:128])}
    tok = RustTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps({"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"})
    )
    mx.random.seed(7)
    model = DecisionModel(EncoderConfig.from_dict(TINY_CFG), TINY_AGENT)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))

    # Per-stage activations on a hand-built batch (bypasses the tokenizer).
    rng = np.random.default_rng(1)
    B, L = 3, 40
    lengths = [40, 33, 12]
    ids = np.zeros((B, L), np.int32)
    valid = np.zeros((B, L), bool)
    for i, n in enumerate(lengths):
        ids[i, :n] = rng.integers(5, 128, n)
        valid[i, :n] = True
    marker_pos = np.array([[3, 7, 11], [4, 9, 0], [2, 5, 0]], np.int32)
    marker_mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], bool)
    qtype = np.array([0, 1, 2], np.int32)
    A = mx.array
    enc_ = model.encoder
    stages = {}
    x = enc_.embeddings(A(ids))
    stages["embeddings"] = x
    masks = attention_masks(A(valid), TINY_CFG["local_attention"])
    for i, layer in enumerate(enc_.layers):
        x = layer(x, masks[layer.attention_type])
        stages[f"encoder.layers.{i}"] = x
    h = enc_.final_norm(x)
    stages["encoder.final_norm"] = h
    h = h + model.type_emb(A(qtype))[:, None, :]
    stages["type_emb_added"] = h
    hm = A(valid)[:, None, None, :]
    for j, layer in enumerate(model.head.layers):
        h = layer(h, hm)
        stages[f"head.layers.{j}"] = h
    logits, act = model(A(ids), A(valid), A(marker_pos), A(marker_mask), A(qtype))
    stages["logits"] = logits
    stages["act"] = act
    dump_json(
        path / "activations.json",
        {
            "inputs": {
                "input_ids": enc(ids), "attention_mask": enc(valid), "marker_pos": enc(marker_pos),
                "marker_mask": enc(marker_mask), "qtype": enc(qtype),
            },
            "stages": {k: enc(np.asarray(v.astype(mx.float32))) for k, v in stages.items()},
            "layer_types": EncoderConfig.from_dict(TINY_CFG).layer_types,
        },
    )
    # End-to-end predict through the real Agent (tokenizer included).
    agent = Agent(str(path), dtype="float32", device="gpu", batch_size=2)
    state = {"message": "hello w1 w2 w3 [MASK] w4", "n": 1.5}
    questions = {
        "topic": {"type": "choice", "instructions": "hello choice", "criteria": ["a", "b", "c"]},
        "level": {"type": "score", "instructions": "level", "criteria": ["no", "yes"]},
        "yes": {"type": "noul", "instructions": "the statement"},
        "pair": {"type": "choice", "instructions": "a or b", "criteria": {"a": "w1 w2", "b": None}},
    }
    items, _ = agent.prepare(state, questions)
    dump_json(
        path / "predict.json",
        {"state": state, "questions": questions, "items": items, "result": agent.predict(state, questions)},
    )


# ----------------------------------------------------------------------------- real models
def dump_real(out: Path, names):
    for name in names:
        repo = REPOS[name]
        cases_out = []
        agent32 = Agent(repo, dtype="float32", device="gpu", batch_size=16)
        agent16 = Agent(repo, dtype="float16", device="gpu", batch_size=16)
        max_len = agent32.cfg.get("max_len", 512)
        for case, state, questions in parity_cases():
            items, _ = agent32.prepare(state, questions)
            rows = []
            for start in range(0, len(items), 16):
                chunk = items[start : start + 16]
                batch = collate_items(chunk, agent32.tok.pad_token_id, max_length=max_len)
                logits, act = (np.asarray(t) for t in agent32.forward(batch))
                for i, item in enumerate(chunk):
                    k = len(item["markers"])
                    rows.append({"logits": logits[i, :k].tolist(), "act": act[i].tolist()})
            cases_out.append(
                {
                    "case": case,
                    "state": state,
                    "questions": questions,
                    "items": items,
                    "outputs": rows,
                    "result_fp32": agent32.predict(state, questions),
                    "result_fp16": agent16.predict(state, questions),
                }
            )
        texts = ["I was billed twice.", "发票4411被重复扣款", "", "billing: refunds and invoices"]
        emb = embed_fn_from_agent(agent32)(texts)
        dump_json(
            out / "real" / f"{name}.json",
            {
                "repo": repo,
                "model_dir": str(agent32.model_dir),
                "config": agent32.cfg,
                "cases": cases_out,
                "embeddings": {"texts": texts, "vectors": enc(emb)},
            },
        )
        # Tokenizer table for this checkpoint's tokenizer.
        tok = agent32.tok
        samples = [
            "hello world", "  leading and trailing  ", "[MASK] <mask> [CLS]", "émoji 🙂 and ümlauts",
            "发票4411被重复扣款，请今天退款。", "मुझसे इनवॉइस 4411", "請求書4411で二重に請求されました。",
            "tabs\tand\nnewlines\r\n", "choice question: Who should handle this?", " level 0: low",
            json.dumps({"a": [1, 2.5, None], "b": "x"}, ensure_ascii=False), "a" * 300,
        ]
        dump_json(
            out / "tables" / f"tokenizer-{name}.json",
            {
                "special": {
                    k: [getattr(tok, k), getattr(tok, k + "_id")]
                    for k in ("cls_token", "sep_token", "pad_token", "mask_token")
                },
                "cases": [
                    {"text": s, "ids": tok(s)["input_ids"], "ids_special": tok.backend.encode(s, add_special_tokens=True).ids}
                    for s in samples
                ],
            },
        )
        del agent32, agent16


# ----------------------------------------------------------------------------- tables
def dump_tables(out: Path):
    values = [
        None, True, False, 0, -0.0, 1, -17, 2**53 + 1, 10**30, 1.0, 1.5, 0.1, 1e-7, 1e16, 1.5e300,
        123456789.123, float("nan"), float("inf"), -float("inf"), "", "plain", 'quote " and \\',
        "tab\tnl\n\u0001\u007f", "é ü 中文 🙂", "  ", [], {}, [1, [2, [3]]],
        {"b": 1, "a": [True, None, "x"], "中": {"k": 0.5}}, {"1": 1, "k": {"deep": [1.0, 2.25]}},
    ]
    variants = {
        "default": {},
        "ensure_ascii_false": {"ensure_ascii": False},
        "criterion": {"ensure_ascii": False, "separators": (", ", ": ")},
        "compact": {"separators": (",", ":")},
    }
    pyjson = []
    for v in values:
        row = {"repr": repr(v), "value_json": json.dumps(v, allow_nan=True)}
        for key, kw in variants.items():
            row[key] = json.dumps(v, **kw)
        pyjson.append(row)
    rounds = [0.5, 1.5, 2.5, -0.5, 0.12345, 0.12335, 0.00005, 0.00015, 0.99995, 1 / 3, 2 / 3,
              1e-9, 0.123449999, 0.30000000000000004, 12.34565, -2.00005]
    dump_json(out / "tables" / "pyjson.json", {
        "note": "value_json is the input value encoded with json.dumps(allow_nan=True); "
                "NaN/Infinity tokens must be parsed by the test harness.",
        "variants": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in kw.items()} for k, kw in variants.items()},
        "dumps": pyjson,
        "round4": [{"x": x, "round4": round(x, 4)} for x in rounds],
        "float_repr": [{"x": x, "repr": repr(x)} for x in [1.0, 0.1, 1e16, 1e-7, 123456789.0, 2.5e-5, 1e22, 5e-324]],
    })

    texts = [
        "", "   ", "12345 !!!", "I was charged twice for invoice 4411, please refund it today.",
        "Hello there", "ok", "The quick brown fox jumps over the lazy dog and the cat.",
        "Ich wurde zweimal für Rechnung 4411 belastet, bitte erstatten Sie den Betrag.",
        "J'ai été facturé deux fois pour la facture 4411, remboursez-moi s'il vous plaît.",
        "Me cobraron dos veces la factura 4411, por favor devuélvanme el dinero.",
        "Fui cobrado duas vezes pela fatura, por favor devolvam o dinheiro que não é meu.",
        "Mi hanno addebitato due volte la fattura, per favore rimborsate il denaro.",
        "Ik ben twee keer belast voor de factuur, graag het geld terug van de bank.",
        "Am fost taxat de două ori pentru factură, vă rog să îmi returnați banii.",
        "Am fost taxat de doua ori pentru factura, va rog sa imi returnati banii.",
        "Faturam için iki kez ücret alındı, lütfen paramı iade edin.",
        "Dostałem podwójne obciążenie za fakturę, proszę o zwrot pieniędzy.",
        "发票4411被重复扣款，请今天退款。", "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया",
        "請求書4411で二重に請求されました。", "С меня дважды списали деньги по счёту 4411.",
        "مرحبا، تم خصم المبلغ مرتين", "שלום, חויבתי פעמיים", "안녕하세요, 두 번 청구되었습니다",
        "Բարև, ինձանից երկու անգամ գանձվել է", "สวัสดี ฉันถูกเรียกเก็บเงินสองครั้ง",
        "Mixed English text with some 中文 characters inside it for fun.",
        "café résumé naïve", "Γεια σας, χρεώθηκα δύο φορές",
    ]
    states = texts + [
        {"message": "Ich wurde zweimal belastet, bitte erstatten Sie den Betrag."},
        [{"role": "user", "content": "发票被重复扣款"}], {"Nachricht": "hello there my friend"},
        {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep text de la la"}}}}}}}, None, 42,
    ]
    lang_rows = []
    for s in states:
        text = lang.state_text(s)
        lang_rows.append({
            "state": s, "state_text": text, "detect_script": lang.detect_script(text),
            "script_profile": lang.script_profile(text), "latin_profile": lang.latin_profile(text),
            "guess_latin_language": lang.guess_latin_language(text), "analyse": lang.analyse(s),
            "is_english": lang.is_english(s),
        })
    dump_json(out / "tables" / "lang.json", lang_rows)

    emails = [
        "Hi,\n\nPlease refund invoice 4411.\n\nThanks,\nJane\n--\nJane Doe\nACME Corp",
        "Refund please.\n\nOn Mon, Jan 1, 2024 at 9:00 AM Bob <bob@x.com> wrote:\n> old text\n> more",
        "Body text here.\n\nCONFIDENTIALITY NOTICE: This email is intended only for the recipient.",
        "Sent from my iPhone", "", "Line one\n-----Original Message-----\nFrom: someone",
    ]
    dump_json(out / "tables" / "email.json", [{"body": e, "clean": clean_email_body(e)} for e in emails])

    # render_options table (pure Python, no tokenizer)
    qs = [
        {"t": "choice", "ins": "x", "crit": {"a": None, "b": "", "c": "desc", "d": 0, "e": False, "f": {"k": [1, "é"]}}},
        {"t": "score", "ins": "x", "crit": ["low", {"level": "mid"}, 3, None]},
        {"t": "noul", "ins": "x", "crit": None},
        {"t": "noul", "ins": "x", "crit": {"true": {"reason": "money back"}, "false": ""}},
    ]
    dump_json(out / "tables" / "render_options.json", [{"q": q, "options": render_options(q)} for q in qs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--js", default=None, help="laya-js root (default: <out>/../../..)")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--models", default=",".join(REPOS))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    js = Path(args.js).resolve() if args.js else out.parents[2]
    dump_json(js / "packages/tensor-backend/fixtures/ops.json", op_cases())
    dump_tiny(out)
    dump_tables(out)
    if not args.skip_real:
        mx.set_default_device(mx.gpu)
        dump_real(out, [m for m in args.models.split(",") if m])


if __name__ == "__main__":
    main()
