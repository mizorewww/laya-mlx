import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "60a6ce910b4a2edb2b77e00d8fddeb24349610c8"
MODELS = {
    "laya": "c5d78730f3493e4fe16d61507ef4b78eef7318cf",
    "laya-multilingual": "052592a15d198d9ad47da779604259b10b47b7aa",
    "laya-typed-decisions": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def source_hash():
    h = hashlib.sha256()
    for folder in ("laya_mlx", "benchmarks"):
        for path in sorted((ROOT / folder).glob("*.py")):
            h.update(str(path.relative_to(ROOT)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def environment():
    versions = {}
    for package in (
        "mlx",
        "mlx-metal",
        "coremltools",
        "numpy",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "unified_memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": source_hash(),
        "upstream_revision": UPSTREAM_REVISION,
    }


def load_reference(model_path, device="mps", upstream=ROOT / ".upstream"):
    import sys

    upstream = Path(upstream)
    if not (upstream / "laya/agent.py").exists():
        raise FileNotFoundError(
            "Clone NandhaKishorM/laya into .upstream before running reference checks"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, got {revision}")
    sys.path.insert(0, str(upstream))
    import laya

    agent = laya.load(str(model_path), device=device)
    if agent.device.type != device:
        raise RuntimeError(f"Reference fell back to {agent.device}; benchmark requested {device}")
    return agent


def workload(count=3, long=False):
    state = json.loads((ROOT / "examples/state.json").read_text())
    definitions = list(json.loads((ROOT / "examples/questions.json").read_text()).values())
    if long:
        state["body"] = "The customer reports duplicate billing and requests a refund today. " * 200
    questions = {f"q{i}": definitions[i % len(definitions)] for i in range(count)}
    return state, questions


def parity_cases():
    state, questions = workload()
    cases = [("email", state, questions)]
    messages = {
        "en": "I was charged twice for invoice 4411, please refund it today.",
        "zh": "发票4411被重复扣款，请今天退款。",
        "de": "Ich wurde zweimal für Rechnung 4411 belastet, bitte erstatten Sie den Betrag.",
        "fr": "J'ai été facturé deux fois pour la facture 4411, remboursez-moi s'il vous plaît.",
        "es": "Me cobraron dos veces la factura 4411, por favor devuélvanme el dinero.",
        "hi": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
        "ja": "請求書4411で二重に請求されました。返金してください。",
        "ru": "С меня дважды списали деньги по счёту 4411, верните деньги.",
    }
    for lang, message in messages.items():
        cases.append((lang, {"message": message}, questions))
    cases.extend(
        [
            ("empty_state", "", questions),
            ("long", *workload(3, long=True)),
            ("conversation", [{"role": "user", "content": messages["en"]}], questions),
            ("mask_literals", "[MASK] <mask> hello [MASK] <mask>", questions),
            ("many_questions", *workload(20)),
            (
                "structured",
                state,
                {
                    "choice": {
                        "type": "choice",
                        "instructions": {"task": "choose department"},
                        "criteria": {"billing": {"description": "refunds"}, "other": False},
                    },
                    "score": {
                        "type": "score",
                        "instructions": "Urgency?",
                        "criteria": [{"level": "low"}, "high"],
                    },
                    "noul": {
                        "type": "noul",
                        "instructions": "Refund?",
                        "criteria": {"true": {"reason": "money back"}},
                    },
                },
            ),
            (
                "twenty_options",
                state,
                {
                    "choice": {
                        "type": "choice",
                        "instructions": "Which department handles billing?",
                        "criteria": ["billing"] + [f"department_{i}" for i in range(19)],
                    },
                },
            ),
        ]
    )
    return cases


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def distribution(agent, logits, items):
    from laya_mlx.common import temp_bucket

    values = []
    for row, item in enumerate(items):
        k, qt = len(item["markers"]), item["qtype"]
        scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        values.append(softmax(logits[row, :k] / max(1e-3, float(scale))))
    return values


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
