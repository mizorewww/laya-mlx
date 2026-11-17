"""Unchanged upstream host assertions; no native MLX installation or checkpoints needed.

These suites use their own fake inference fixtures. They qualify Python control
flow, not native tensor execution. Native MCP embedding tests stay in
``test_official_contract.py`` alongside the numerical differential checks.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".upstream"
HOST_SUITES = (
    "option_stability",
    "evals_agreement",
    "evals_tournament",
    "agno",
    "tools",
    "hooks",
    "hooks_api",
    "serve",
    "evals",
    "evals_api",
    "evals_shortlist",
    "evidence",
    "cli",
    "cli_lang_guess",
    "mcp_remote",
    "structured",
    "structured_api",
    "email",
    "lang_stats",
    "lang_guess",
    "lang_diagnostics",
    "blank_lang_routing",
    "router",
    "router_batch",
    "router_vega",
    "criteria_normalization",
    "shortlist_cosine",
    "confidence",
    "crewai",
    "langchain",
    "langchain_native_policy",
    "llamaindex",
    "mcp_diagnostics",
    "_long_attribution",
)


@pytest.mark.parametrize("script", HOST_SUITES)
def test_original_upstream_host_assertions(script):
    name = script if script.startswith("_") else "test_" + script
    path = UPSTREAM / "tests" / (name + ".py")
    if not path.exists():
        pytest.skip("checkout pinned upstream into .upstream (CI does this)")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_official_test.py"), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
