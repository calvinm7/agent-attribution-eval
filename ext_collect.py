"""Trace collection for newer Claude versions (REPORT.md section 7).

Runs the authors' agent_runner.ts verbatim (MidScene 1.6.4, element-id
planning mode, live amazon.com) for the Claude version ladder in MODELS,
through the usage-recording proxy (ext_proxy.py). Round-robin: one episode
per model per round until each model reaches its target valid count or its
attempt cap, so a stop at any point leaves near-balanced per-model counts.
Stops early when cumulative spend reaches SPEND_ABORT_USD or DEADLINE
passes. State derives from disk (like the authors' resume logic), so the
script is re-runnable.

Question order: webshop test split, then val, so the bridge control is
question-matched to the original test traces.

Usage: .venv/bin/python ext_collect.py [--workers 3]
Requires: ext_proxy.py running on 127.0.0.1:8399, key in .env.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from common import ROOT, is_valid_trace, load_config, load_trace

HARNESS = ROOT / "data/reference/harness"
OUT_ROOT = ROOT / "data/new_traces"
USAGE_LOG = ROOT / "data/ext_usage.jsonl"
RUN_LOG = ROOT / "data/ext_collection_log.jsonl"
PROXY = "http://127.0.0.1:8399/v1"
PROXY_ALT = "http://localhost:8399/v1"  # string-unequal planning config, same socket

# Demo-scale targets: small n, more Claude versions, framed as a
# demonstration rather than powered inference. Columns:
# (agent_id, model_id, usd/MTok in, usd/MTok out, target valid, attempt cap)
MODELS = [
    ("ext_claude_opus_4_6", "claude-opus-4-6", 5.0, 25.0, 5, 8),    # bridge control
    ("ext_claude_opus_4_7", "claude-opus-4-7", 5.0, 25.0, 3, 5),    # +1 version step
    ("ext_claude_opus_5", "claude-opus-5", 5.0, 25.0, 3, 5),        # +2 version steps
    ("ext_claude_sonnet_4_6", "claude-sonnet-4-6", 3.0, 15.0, 3, 6),  # tier sibling, same gen
    ("ext_claude_sonnet_5", "claude-sonnet-5", 2.0, 10.0, 5, 8),    # tier sibling, newer gen
]
# haiku-4-5 rows come free from the smoke traces (data/new_traces/smoke).
EPISODE_TIMEOUT_S = 300
SPEND_ABORT_USD = 48.0
DEADLINE = datetime.datetime(2026, 8, 16, 0, 0)

log_lock = Lock()


def read_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY=") and len(line.split("=", 1)[1].strip()) > 10:
            return line.split("=", 1)[1].strip()
    sys.exit("no ANTHROPIC_API_KEY in .env")


def questions_in_order():
    qs = json.load(open(ROOT / "data/webshop_questions.json"))
    return qs["test"] + qs["val"]


def spend_usd():
    """Cumulative spend across all models from the proxy's usage log."""
    rates = {m: (i, o) for _, m, i, o, _, _ in MODELS}
    rates["claude-haiku-4-5"] = (1.0, 5.0)  # smoke episodes count too
    total = 0.0
    if USAGE_LOG.exists():
        for line in open(USAGE_LOG):
            try:
                r = json.loads(line)
                ir, orate = rates.get(r.get("model", ""), (0, 0))
                total += r.get("prompt_tokens", 0) / 1e6 * ir + r.get("completion_tokens", 0) / 1e6 * orate
            except Exception:
                pass
    return total


def valid_questions(agent_id, patterns):
    """Questions with a valid trace on disk for this agent (resume state)."""
    done = set()
    for p in (OUT_ROOT / agent_id / "webshop_ext").glob("*.json"):
        try:
            t = load_trace(p)
            if is_valid_trace(t, patterns):
                q = (t.get("meta") or {}).get("question")
                if q:
                    done.add(q)
        except Exception:
            pass
    return done


def run_episode(agent_id, model_id, question, template, key):
    prompt = template.replace("{question}", question).replace("{start_url}", "https://www.amazon.com")
    out_dir = OUT_ROOT / agent_id / "webshop_ext"
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = f"{agent_id}_{uuid.uuid4().hex[:8]}"
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin",
        "HOME": str(Path.home()),
        "MIDSCENE_MODEL_NAME": model_id,
        "MIDSCENE_MODEL_API_KEY": key,
        "MIDSCENE_MODEL_BASE_URL": PROXY,
        "MIDSCENE_PLANNING_MODEL_NAME": model_id,
        "MIDSCENE_PLANNING_MODEL_API_KEY": key,
        "MIDSCENE_PLANNING_MODEL_BASE_URL": PROXY_ALT,
        "MIDSCENE_REPLANNING_CYCLE_LIMIT": "40",
    }
    cmd = ["npx", "tsx", "agent_runner.ts",
           "--question", question, "--task_prompt", prompt,
           "--agent_id", agent_id, "--episode_id", ep,
           "--output_dir", str(out_dir),
           "--start_url", "https://www.amazon.com", "--task_type", "shop"]
    t0 = time.time()
    # start_new_session so a timeout can kill the whole process group; killing
    # npx alone leaves a live tsx child that keeps browsing and spending.
    proc = subprocess.Popen(cmd, cwd=HARNESS, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=EPISODE_TIMEOUT_S)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
    trace = out_dir / f"{ep}.json"
    ok = False
    if trace.exists():
        try:
            ok = is_valid_trace(load_trace(trace), load_config()["invalid_error_patterns"])
        except Exception:
            ok = False
    row = {"ts": time.time(), "agent": agent_id, "episode": ep, "question": question[:60],
           "valid": ok, "timed_out": timed_out, "wall_s": round(time.time() - t0, 1)}
    with log_lock:
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    return ok


def stop_reason():
    s = spend_usd()
    if s >= SPEND_ABORT_USD:
        return f"SPEND_STOP at ${s:.2f}"
    if datetime.datetime.now() >= DEADLINE:
        return "DEADLINE_STOP"
    return None


def attempts_so_far(agent_id):
    if not RUN_LOG.exists():
        return 0
    return sum(1 for l in open(RUN_LOG) if json.loads(l).get("agent") == agent_id)


def collect(workers):
    """One episode per model per round until every model hits its target
    valid count or attempt cap, or a stop rule fires."""
    cfg = load_config()
    patterns = cfg["invalid_error_patterns"]
    key = read_key()
    template = (HARNESS / "task_prompt_templates/shop_amazon.txt").read_text()
    order = questions_in_order()
    done = {a: valid_questions(a, patterns) for a, _, _, _, _, _ in MODELS}
    tries = {a: attempts_so_far(a) for a, _, _, _, _, _ in MODELS}

    def next_question(agent_id):
        for q in order:
            if q not in done[agent_id]:
                return q
        return None

    while True:
        reason = stop_reason()
        if reason:
            break
        jobs = []
        for agent_id, model_id, _, _, target, cap in MODELS:
            if len(done[agent_id]) >= target or tries[agent_id] >= cap:
                continue
            q = next_question(agent_id)
            if q is not None:
                jobs.append((agent_id, model_id, q))
        if not jobs:
            reason = "TARGETS_OR_CAPS_MET"
            break
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_episode, a, m, q, template, key): (a, q) for a, m, q in jobs}
            for fut in as_completed(futs):
                a, q = futs[fut]
                tries[a] += 1
                if fut.result():
                    done[a].add(q)
        print("counts:", {a: len(done[a]) for a in done},
              "attempts:", dict(tries), f"spend ${spend_usd():.2f}", flush=True)
    print(f"{reason}; counts:", {a: len(done[a]) for a in done}, f"spend ${spend_usd():.2f}")
    return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    print("COLLECT_RESULT:", collect(args.workers))


if __name__ == "__main__":
    main()
