#!/usr/bin/env python3
"""End-to-end verification of a running Laya sidecar and of the dsh-laya contract.

Stdlib only, no dependencies. Checks the HTTP contract the `dsh-laya` plugin
speaks (`POST /ask`, `POST /plan`, `GET /health`) and, with ``--mcp-stdio``, the
other transport `laya-mcp mcp` exposes over stdio.

    python scripts/verify_laya_integration.py
    python scripts/verify_laya_integration.py --base-url http://127.0.0.1:8787
    python scripts/verify_laya_integration.py --mcp-stdio   # cold-loads a 2nd model, ~90s

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUG_Q = {
    "is_defect": {
        "type": "noul",
        "instructions": "Does the user report a product defect or broken behavior?",
        # NOTE: `POST /ask` reads `criteria`. The MCP tool `laya_noul` calls the
        # same thing `boundary` and folds it in at mcp_server.py; sending
        # `boundary` to /ask is silently dropped, which leaves the options with
        # no text at all.
        "criteria": {
            "true": "the user reports an error, crash, or something not working",
            "false": "the user asks about pricing, billing, or purchasing",
        },
    }
}
RUBRIC_Q = {
    "category": {
        "type": "choice",
        "instructions": "Classify the customer message into exactly one category.",
        "criteria": {
            "billing": "About invoices, charges, payment amounts, refunds, or money owed.",
            "technical": "About software behavior, errors, bugs, integration, or how the product works.",
            "sales": "About pricing plans, purchasing, upgrades, demos, or pre-sale contact.",
        },
    }
}
PRIO_Q = {
    "priority": {
        "type": "score",
        "instructions": "Place this ticket on a priority scale.",
        "criteria": [
            "low: cosmetic or nice-to-have",
            "medium: degrades a workflow but has a workaround",
            "high: blocks a paying customer with no workaround",
        ],
    }
}
POS_REVIEW = "This product is fantastic and saved my team hours every week."
NEG_REVIEW = "This product is useless and it broke our workflow completely."


class Verifier:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
        return bool(ok)

    def info(self, name: str, detail: str) -> None:
        print(f"[INFO] {name}\n        {detail}")

    def call(self, path: str, payload: dict, timeout: float = 120):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), (time.time() - t0) * 1000

    def get(self, path: str, timeout: float = 10) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.load(r)


def verify_http(v: Verifier) -> None:
    try:
        h = v.get("/health")
    except Exception as e:
        v.check("1. GET /health", False, f"cannot reach {v.base}: {type(e).__name__}: {e}")
        print("\nIs `laya-mcp serve` running? Start it, or set the dsh-laya plugin's "
              "lifecycle to 'spawn' with a spawnCommand.")
        return
    v.check(
        "1. GET /health",
        h.get("ok") and h.get("loaded") and not h.get("degraded"),
        f"loaded={h.get('loaded')} degraded={h.get('degraded')} "
        f"ckpt={list(h.get('checkpoints', {}))} "
        f"dev={h.get('checkpoints', {}).get('english', {}).get('device')}",
    )

    pos, ms_pos = v.call("/ask", {"state": {"body": "The app crashes every time I upload a CSV."},
                                  "questions": BUG_Q})
    p_pos = pos["answers"]["is_defect"]["noul"]
    v.check("2. noul positive (defect report)", p_pos > 0.5 and pos["answers"]["is_defect"].get("band") == "yes",
            f"P(true)={p_pos:.4f} band={pos['answers']['is_defect'].get('band')}")

    neg, ms_neg = v.call("/ask", {"state": {"body": "What payment methods do you accept?"},
                                  "questions": BUG_Q})
    p_neg = neg["answers"]["is_defect"]["noul"]
    v.check("3. noul negative (pricing question)", p_neg < 0.5 and neg["answers"]["is_defect"].get("band") == "no",
            f"P(true)={p_neg:.4f} band={neg['answers']['is_defect'].get('band')}")

    # The raw checkpoint renders a noul as "false: ..."/"true: ..." and answers
    # false to nearly everything. laya-mcp re-asks it as a neutral A/B choice
    # (protocol.noul_as_choice), so a passing noul above only proves the
    # mitigation is active. Probe the labels directly to show why it exists.
    label_q = {"q": {"type": "choice", "instructions": "Is this review positive or negative?",
                     "criteria": {"A": "the review is positive", "B": "the review is negative"}}}
    ab_pos, _ = v.call("/ask", {"state": {"body": POS_REVIEW}, "questions": label_q})
    ab_neg, _ = v.call("/ask", {"state": {"body": NEG_REVIEW}, "questions": label_q})
    ab_p, ab_n = ab_pos["answers"]["q"]["choice"], ab_neg["answers"]["q"]["choice"]
    v.check("4. neutral A/B labels answer both directions correctly", ab_p == "A" and ab_n == "B",
            f"positive->{ab_p} negative->{ab_n}")

    tf_q = {"q": {"type": "choice", "instructions": "Is this review positive or negative?",
                  "criteria": {"true": "the review is positive", "false": "the review is negative"}}}
    tf, _ = v.call("/ask", {"state": {"body": POS_REVIEW}, "questions": tf_q})
    v.info("same positive review under true/false labels",
           f"-> {tf['answers']['q']['choice']} p={tf['answers']['q']['probabilities']} "
           f"(the collapse noul_as_choice exists to avoid)")

    bare, _ = v.call("/ask", {"state": {"body": "The app crashes every time I upload a CSV."},
                              "questions": {"is_defect": {"type": "noul",
                                                          "instructions": "Does the user report a product defect?"}}})
    bare_answer = bare["answers"]["is_defect"]
    v.check("4b. noul with no option text is still well-formed",
            bare_answer.get("type") == "noul" and bare_answer.get("band") in ("no", "uncertain", "yes"),
            f"returned {bare_answer}; with option text it was {p_pos:.4f}")
    v.info("noul with no option text",
           "legal, but how decisive it is varies by runtime - always supply the text "
           "(criteria for /ask, boundary for the MCP tool) when the answer matters")

    ch, ms_ch = v.call("/ask", {"state": {"body": 'Customer message: "The invoice amount is wrong."'},
                                "questions": RUBRIC_Q})
    a = ch["answers"]["category"]
    v.check("5. choice -> billing", a["choice"] == "billing" and a["probabilities"]["billing"] > 0.5,
            f"choice={a['choice']} p={a['probabilities']}")

    sc, _ = v.call("/ask", {"state": {"body": "Production is down for all customers and there is no workaround."},
                            "questions": PRIO_Q})
    s = sc["answers"]["priority"]
    sprobs = s.get("probabilities") or {}
    top = max(sprobs, key=sprobs.get) if sprobs else None
    legend = s.get("legend") or {}
    v.check("6. score places a Sev-1 at the top level",
            top == "2" and (s.get("score") is None or s["score"] >= 1.5),
            f"expected_level={s.get('score')} top_index={top} p={sprobs} "
            f"top_meaning={legend.get(top)!r}")

    pl, ms_pl = v.call("/plan", {"state": {"body": "hi"}, "questions": RUBRIC_Q})
    v.check("7. POST /plan (no forward pass)", pl.get("ok") and pl.get("fits") and ms_pl < 50,
            f"fits={pl.get('fits')} {ms_pl:.0f} ms (vs {ms_ch:.0f} ms for /ask)")

    plbig, _ = v.call("/plan", {"state": {"body": "lorem ipsum dolor sit amet " * 400}, "questions": RUBRIC_Q})
    q = plbig["questions"][0]
    v.check("8. oversized state is flagged before you ask",
            bool(q.get("would_truncate_state")) or bool(plbig.get("warnings")),
            f"state_chars={plbig['state_chars']} room~{q.get('state_room_estimated')} "
            f"would_truncate={q.get('would_truncate_state')} warnings={plbig.get('warnings')}")

    r1, _ = v.call("/ask", {"state": {"body": "The app crashes every time I upload a CSV."}, "questions": BUG_Q})
    r2, _ = v.call("/ask", {"state": {"body": "The app crashes every time I upload a CSV."}, "questions": BUG_Q})
    v1, v2 = r1["answers"]["is_defect"]["noul"], r2["answers"]["is_defect"]["noul"]
    v.check("9. deterministic under repeat", abs(v1 - v2) < 1e-6, f"{v1:.6f} vs {v2:.6f}")

    def one(i: int) -> float:
        b, _ = v.call("/ask", {"state": {"body": f"Ticket {i}: the app crashes on upload."},
                               "questions": BUG_Q})
        return b["answers"]["is_defect"]["noul"]

    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            vals = list(ex.map(one, range(4)))
        v.check("10. 4 concurrent /ask calls all answered", all(x > 0.5 for x in vals),
                f"nouls={[round(x, 3) for x in vals]}")
    except Exception as e:
        v.check("10. 4 concurrent /ask calls all answered", False, f"raised {type(e).__name__}: {e}")

    # A cold process pays a one-off kernel-compilation cost on the first pass at
    # each padded shape (measured: 2188 ms for the first call, 34 ms for the
    # second at the same shape). Timing that first call measures the compiler,
    # not the model, so warm up and then report steady state - and surface the
    # cold pass separately rather than hiding it.
    if ms_pos > 500:
        v.info("first inference on a cold process",
               f"{ms_pos:.0f} ms for the shape; one-off compilation, not steady-state latency")
    for _ in range(2):
        v.call("/ask", {"state": {"body": "warmup"}, "questions": RUBRIC_Q})
    steady = sorted(v.call("/ask", {"state": {"body": 'Customer message: "The invoice amount is wrong."'},
                                    "questions": RUBRIC_Q})[1] for _ in range(5))
    p50 = steady[len(steady) // 2]
    v.check("11. steady-state latency is sub-200ms", p50 < 200,
            f"p50={p50:.0f}ms over 5 warm calls (min={steady[0]:.0f} max={steady[-1]:.0f}); "
            f"cold first call was {ms_pos:.0f}ms")


def verify_mcp_stdio(v: Verifier) -> None:
    """The other transport: `laya-mcp mcp`, for harnesses that spawn per session."""
    import shutil
    exe = shutil.which("laya-mcp")
    if not exe:
        v.check("MCP stdio: laya-mcp on PATH", False, "laya-mcp not found; activate the venv that has it")
        return
    proc = subprocess.Popen([exe, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    lines: queue.Queue = queue.Queue()

    def _reader():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    counter = {"id": 0}

    def send(method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            counter["id"] += 1
            msg["id"] = counter["id"]
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        return counter["id"]

    def wait_for(want_id, timeout=240):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = lines.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                return None
            if line is None:
                return None
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want_id:
                return msg
        return None

    try:
        t0 = time.time()
        resp = wait_for(send("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                            "clientInfo": {"name": "laya-verify", "version": "1.0"}}), timeout=60)
        if not resp:
            v.check("MCP stdio: initialize", False, "no response within 60s")
            return
        si = resp.get("result", {}).get("serverInfo", {})
        v.check("MCP stdio: initialize", "result" in resp,
                f"server={si.get('name')} version={si.get('version')} in {time.time()-t0:.1f}s")

        send("notifications/initialized", notify=True)
        tl = wait_for(send("tools/list"), timeout=30)
        tools = [t["name"] for t in (tl or {}).get("result", {}).get("tools", [])]
        # Unlike the HTTP/plugin surface, stdio exposes the primitives separately.
        v.check("MCP stdio: tools/list", "laya_ask" in tools and "laya_plan" in tools, f"tools={tools}")

        t1 = time.time()
        call = wait_for(send("tools/call", {"name": "laya_ask", "arguments": {
            "state": {"body": 'Customer message: "The invoice amount is wrong."'},
            "questions": RUBRIC_Q,
        }}), timeout=300)
        if not call:
            v.check("MCP stdio: tools/call laya_ask", False, "no response within 300s")
            return
        text = "".join(c.get("text", "") for c in call.get("result", {}).get("content", []))
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
        inner = parsed.get("answers", {}).get("category", {})
        v.check("MCP stdio: tools/call laya_ask", inner.get("choice") == "billing",
                f"choice={inner.get('choice')} p={inner.get('probabilities')} in {time.time()-t1:.1f}s "
                f"(cold load; the warm sidecar is ~40ms)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8787", help="the laya-mcp serve sidecar")
    p.add_argument("--mcp-stdio", action="store_true",
                   help="also verify `laya-mcp mcp`; cold-loads a second model (~90s)")
    p.add_argument("--skip-http", action="store_true", help="only run the stdio checks")
    args = p.parse_args()

    v = Verifier(args.base_url)
    if not args.skip_http:
        verify_http(v)
    if args.mcp_stdio:
        print()
        verify_mcp_stdio(v)

    passed = sum(1 for _, ok, _ in v.results if ok)
    print("\n" + "=" * 60)
    print(f"laya integration: {passed}/{len(v.results)} passed")
    if passed != len(v.results):
        print("FAILED: " + ", ".join(n for n, ok, _ in v.results if not ok))
    print("=" * 60)
    return 0 if v.results and passed == len(v.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
