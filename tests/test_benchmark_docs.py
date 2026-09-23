"""Keep current-facing performance figures aligned with the committed results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
HISTORICAL_REVISION = "49daed6609cb3da142d1a0c88e538dc07f00d974"


def metric(model: str, questions: int, key: str) -> float:
    report = json.loads((RESULTS / f"{model}-mlx-float16.json").read_text())
    row = next(
        result
        for result in report["results"]
        if result["workload"] == "short" and result["questions"] == questions
    )
    return row["end_to_end"][key]


def test_current_readmes_match_committed_benchmarks() -> None:
    english = ROOT.joinpath("README.md").read_text()
    chinese = ROOT.joinpath("README.zh-CN.md").read_text()
    laya = "laya"
    multilingual = "laya-multilingual"
    laya_p50 = f"{metric(laya, 1, 'p50_ms'):.2f}"
    multilingual_p50 = f"{metric(multilingual, 1, 'p50_ms'):.2f}"
    run_date = json.loads((RESULTS / "laya-mlx-float16.json").read_text())["created_at"].split("T")[
        0
    ]

    assert (
        f"**{laya_p50} ms** median end-to-end for a short English typed decision. "
        f"**{multilingual_p50} ms** with the multilingual checkpoint."
        in "\n".join(english.splitlines()[:8])
    )
    assert (
        f"单个短问题端到端中位耗时 **{laya_p50} ms**；multilingual 检查点为 "
        f"**{multilingual_p50} ms**。" in "\n".join(chinese.splitlines()[:8])
    )
    assert f"上面的 {laya_p50} / {multilingual_p50} ms 来自" in chinese
    assert f"committed {run_date} benchmark run" in english
    assert f"{run_date} 的基准测试" in chinese

    for text, labels in (
        (english, ("One short question, P50", "One short question, P95", "50-question throughput")),
        (chinese, ("单个短问题 P50", "单个短问题 P95", "50 问题吞吐量")),
    ):
        for label, questions, key, precision, unit in (
            (labels[0], 1, "p50_ms", 2, "ms"),
            (labels[1], 1, "p95_ms", 2, "ms"),
            (labels[2], 50, "questions_per_second", 1, "q/s"),
        ):
            expected = (
                f"| {label} | **{metric(laya, questions, key):.{precision}f} {unit}**"
                f" | **{metric(multilingual, questions, key):.{precision}f} {unit}** |"
            )
            assert expected in text


def test_release_copy_matches_committed_multilingual_latency() -> None:
    text = ROOT.joinpath("docs", "LAUNCH.md").read_text()
    latency = f"{metric('laya-multilingual', 1, 'p50_ms'):.2f} ms"
    run_date = json.loads((RESULTS / "laya-multilingual-mlx-float16.json").read_text())[
        "created_at"
    ].split("T")[0]
    assert f"One-question API benchmark: {latency} p50" in text
    assert f"单问题基准 P50 为 {latency}" in text
    assert f"The {latency} headline describes" in text
    assert f"committed {run_date} run" in text


def test_historical_research_links_keep_the_original_results() -> None:
    reports = {
        "PERFORMANCE_RESEARCH.md": (
            "laya-mlx-float16",
            "laya-mlx-float32",
            "laya-torch-mps-float32",
            "laya-multilingual-mlx-float16",
            "laya-multilingual-mlx-float32",
            "laya-multilingual-torch-mps-float32",
        ),
        "MATH_10X_RESEARCH.md": (
            "laya-mlx-float16",
            "laya-multilingual-mlx-float16",
            "laya-typed-decisions-mlx-float16",
        ),
    }
    for report, sources in reports.items():
        text = ROOT.joinpath("docs", report).read_text()
        assert "`49daed6`" in text
        for source in sources:
            assert (
                "https://github.com/mizorewww/laya-mlx/blob/"
                f"{HISTORICAL_REVISION}/benchmarks/results/{source}.json"
            ) in text
            assert f"(../benchmarks/results/{source}.json)" not in text
