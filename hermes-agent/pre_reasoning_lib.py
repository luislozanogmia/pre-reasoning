"""Shared pre-reasoning helpers — local engine only (canonical; copied by hermes_agent_hooks --install)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

MIN_WORDS = 16  # Auto-hook runs only above 15 words; voluntary analyze() anytime (SOUL.md)
STATE_DIR = os.path.expanduser("~/.hermes/hooks/pre_reasoning")
REPROMPT_FLAG = os.path.join(STATE_DIR, "reprompt_needed")
PROBLEM_CACHE = os.path.join(STATE_DIR, "problem.txt")
TURN_META = os.path.join(STATE_DIR, "turn_meta.json")

WEAK_TRACE_SUFFIX_TEMPLATE = (
    "\n\nThe hook trace was weak (<5 blocks; you see blocks={hook_blocks} above). "
    "Enrich the problem with dependencies, blockers, options, stakeholders, constraints, "
    "and numbered pain points, then re-run pre_reasoning.analyze locally before answering.\n\n"
    "REQUIRED 2-LINER when blocks={hook_blocks} (do not skip re-analyze or block counts):\n"
    "**Pre-reasoning:** The hook trace was weak ({hook_blocks} block(s)). I re-ran "
    "pre_reasoning.analyze after enriching the problem and got <M> blocks — "
    "<one concrete insight from the richer trace>. That guided <how you structured the answer>.\n"
    "Replace <M> with the real n_blocks from your second analyze() call. "
    "Do not answer from the weak hook trace alone; do not paraphrase without stating both block counts."
)


def ensure_state_dir() -> None:
    os.makedirs(STATE_DIR, mode=0o700, exist_ok=True)


def analyze(text: str) -> dict | None:
    try:
        from pre_reasoning import analyze as _analyze  # type: ignore

        return _analyze(text)
    except Exception:
        return None


def format_trace(result: dict) -> str | None:
    trace = (result.get("trace") or "").strip()
    if not trace:
        return None
    if trace.startswith("--- STRUCTURAL TRACE ---"):
        parts = trace.split("\n", 1)
        trace = parts[1].strip() if len(parts) > 1 else trace
    grounding = result.get("grounding_level", "unknown")
    enhanced = result.get("l1_enhanced", False)
    tag = f"grounding:{grounding}" + (" +LLM" if enhanced else "")
    n_blocks = result.get("n_blocks", 0)
    header = (
        "--- STRUCTURAL TRACE ---\n"
        "This is a map of the situation - not a perfect map, but a grounding map.\n"
        "It gives you the opportunity to see the whole picture before committing to tokens.\n"
        "Now, create the solution on your own using this map."
    )
    note = (
        "NOTE: You are requested to use the pre-reasoning blocks above to think about the solution. "
        "This is not optional. You MUST show a 2-liner in your response: "
        "what pre-reasoning surfaced and how it guided your answer."
    )
    if n_blocks < 5:
        note += WEAK_TRACE_SUFFIX_TEMPLATE.format(hook_blocks=n_blocks)
    return f"[Mia Pre-Reasoning | {tag} | blocks={n_blocks}]\n{header}\n\n{trace}\n\n{note}"


def save_turn_state(prompt: str, n_blocks: int, session_id: str = "") -> None:
    ensure_state_dir()
    with open(PROBLEM_CACHE, "w", encoding="utf-8") as f:
        f.write(prompt)
    meta = {"n_blocks": n_blocks, "session_id": session_id, "prompt_words": len(prompt.split())}
    with open(TURN_META, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    if n_blocks < 5:
        with open(REPROMPT_FLAG, "w", encoding="utf-8") as f:
            f.write(str(n_blocks))
    elif os.path.exists(REPROMPT_FLAG):
        os.remove(REPROMPT_FLAG)


def clear_turn_state() -> None:
    for p in (REPROMPT_FLAG, PROBLEM_CACHE, TURN_META):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def compliance_markers_present(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if "pre-reasoning" in low or "mia pre-reasoning" in low:
        return True
    if re.search(r"pre[- ]?reasoning", low):
        return True
    markers = ["pre_reasoning", "structural trace", "pre_reasoning.analyze"]
    return any(m in low for m in markers)


def hook_weak_block_count() -> int | None:
    """Blocks from the hook's first analyze() this turn (<5 triggers re-analyze)."""
    if os.path.isfile(REPROMPT_FLAG):
        try:
            with open(REPROMPT_FLAG, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            pass
    if os.path.isfile(TURN_META):
        try:
            with open(TURN_META, encoding="utf-8") as f:
                meta = json.load(f)
            n = int(meta.get("n_blocks") or 0)
            if 0 < n < 5:
                return n
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def weak_trace_compliance_present(text: str, hook_blocks: int) -> bool:
    """Weak-trace turns must cite hook N, re-run, enriched M, and insight."""
    if not compliance_markers_present(text):
        return False
    low = text.lower()
    if not re.search(r"re[\s-]?ran|re[\s-]?prompt", low):
        return False
    block_nums = [int(m) for m in re.findall(r"(\d+)\s*blocks?", low)]
    if len(block_nums) >= 2:
        return hook_blocks in block_nums or any(b >= 5 for b in block_nums)
    if str(hook_blocks) in text and re.search(
        r"got\s+\d+\s*blocks?|now\s+\d+\s*blocks?|\d+\s*blocks?\s*—",
        low,
    ):
        return True
    return False


def turn_compliance_ok(text: str) -> bool:
    hook_n = hook_weak_block_count()
    if hook_n is not None:
        return weak_trace_compliance_present(text, hook_n)
    return compliance_markers_present(text)


def pulse_local(problem: str, response: str) -> dict:
    try:
        from pre_reasoning import pulse  # type: ignore

        return pulse(problem, response)
    except Exception:
        return {"status": "COMPLETE", "gaps": []}