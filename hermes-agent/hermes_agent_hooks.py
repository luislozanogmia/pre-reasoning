#!/usr/bin/env python3
"""
Hermes Agent — Mia Pre-Reasoning integration (local engine only).

Requirements:
  pip install -e /path/to/pre-reasoning   # provides `pre_reasoning` + torch

Install into ~/.hermes:
  python3 hermes_agent_hooks.py --install

Print config snippet for ~/.hermes/config.yaml:
  python3 hermes_agent_hooks.py --print-config

After install: fully quit Hermes desktop (Cmd+Q) and reopen so plugins register.
Verify: tail -1 ~/.hermes/hooks/pre_reasoning/audit.log  (expect session_id + source plugin)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIN_WORDS = 16  # Auto-hook runs only above 15 words; voluntary analyze() anytime (SOUL.md)
STATE_DIR = os.path.expanduser("~/.hermes/hooks/pre_reasoning")
REPROMPT_FLAG = os.path.join(STATE_DIR, "reprompt_needed")
PROBLEM_CACHE = os.path.join(STATE_DIR, "problem.txt")
TURN_META = os.path.join(STATE_DIR, "turn_meta.json")
AUDIT_LOG = os.path.expanduser("~/.hermes/hooks/pre_reasoning/audit.log")

FOOTER_OK = "\n\n---\n✓ **Pre-reasoning:** trace applied this turn."
FOOTER_MISSING = (
    "\n\n---\n⚠ **Pre-reasoning:** compliance not detected — include a 2-liner on what "
    "the structural trace surfaced and how it guided your answer."
)

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

HERMES_CONFIG_SNIPPET = """
# --- Mia Pre-Reasoning (merge into ~/.hermes/config.yaml) ---
hooks_auto_accept: true
plugins:
  enabled:
    - pre-reasoning-hooks
# Optional shell hooks (desktop often uses plugin only; enable if you want parity):
# hooks:
#   pre_llm_call:
#     - command: "python3 ~/.hermes/agent-hooks/hermes_pre_llm.py"
#       timeout: 25
#   post_llm_call:
#     - command: "python3 ~/.hermes/agent-hooks/hermes_post_llm.py"
#       timeout: 25
""".strip()

PLUGIN_YAML = """name: pre-reasoning-hooks
version: \"1.2.1\"
description: \"Inject Mia Pre-Reasoning trace (local engine) and compliance footer.\"
hooks:
  - pre_llm_call
  - post_llm_call
  - transform_llm_output
"""


import importlib.util


def _load_canonical_lib():
    path = Path(__file__).resolve().parent / "pre_reasoning_lib.py"
    spec = importlib.util.spec_from_file_location("pre_reasoning_lib_canon", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_CANON = _load_canonical_lib()
turn_compliance_ok = _CANON.turn_compliance_ok


# ---------------------------------------------------------------------------
# Shared library (local analyze / pulse only — no HTTP API fallback)
# ---------------------------------------------------------------------------

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


def pulse_local(problem: str, response: str) -> dict:
    try:
        from pre_reasoning import pulse  # type: ignore

        return pulse(problem, response)
    except Exception:
        return {"status": "COMPLETE", "gaps": []}


# ---------------------------------------------------------------------------
# Hermes plugin hooks
# ---------------------------------------------------------------------------

def on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    prompt = kwargs.get("user_message") or ""
    if not isinstance(prompt, str) or len(prompt.split()) < MIN_WORDS:
        return None
    session_id = str(kwargs.get("session_id") or "")
    result = analyze(prompt)
    if not result or int(result.get("n_blocks") or 0) == 0:
        clear_turn_state()
        return None
    save_turn_state(prompt, int(result.get("n_blocks") or 0), session_id=session_id)
    trace = format_trace(result)
    return {"context": trace} if trace else None


def on_post_llm_call(**kwargs: Any) -> None:
    assistant = kwargs.get("assistant_response") or ""
    if not isinstance(assistant, str):
        assistant = ""
    problem = ""
    if os.path.exists(PROBLEM_CACHE):
        try:
            with open(PROBLEM_CACHE, encoding="utf-8") as f:
                problem = f.read().strip()
        except OSError:
            pass
    reprompt = os.path.exists(REPROMPT_FLAG)
    compliant = turn_compliance_ok(assistant)
    pulse = {"status": "SKIP"}
    if problem and assistant:
        pulse = pulse_local(problem, assistant)
    ensure_state_dir()
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "session_id": kwargs.get("session_id"),
                    "source": "plugin",
                    "reprompt_expected": reprompt,
                    "compliance_markers": compliant,
                    "pulse_status": pulse.get("status"),
                    "pulse_gaps": pulse.get("gaps", []),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def on_transform_llm_output(**kwargs: Any) -> str | None:
    text = kwargs.get("response_text") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    if not os.path.exists(TURN_META) and not os.path.exists(PROBLEM_CACHE):
        return None
    if turn_compliance_ok(text):
        if FOOTER_OK.strip() in text:
            return None
        return text + FOOTER_OK
    if FOOTER_MISSING.strip() in text:
        return None
    return text + FOOTER_MISSING


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)


# ---------------------------------------------------------------------------
# Install: materialize ~/.hermes/agent-hooks + plugin (from this file)
# ---------------------------------------------------------------------------

def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def install() -> None:
    home = Path(os.path.expanduser("~/.hermes"))
    hooks_dir = home / "agent-hooks"
    plugin_dir = home / "plugins" / "pre-reasoning-hooks"

    lib_src = Path(__file__).resolve().parent / "pre_reasoning_lib.py"
    lib_module = lib_src.read_text(encoding="utf-8")
    plugin_init = '''"""Pre-reasoning hooks for Hermes — installed from pre-reasoning/hermes-agent/hermes_agent_hooks.py."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

_LIB = os.path.expanduser("~/.hermes/agent-hooks")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import pre_reasoning_lib as pr  # noqa: E402

AUDIT_LOG = os.path.expanduser("~/.hermes/hooks/pre_reasoning/audit.log")
FOOTER_OK = "\\n\\n---\\n✓ **Pre-reasoning:** trace applied this turn."
FOOTER_MISSING = (
    "\\n\\n---\\n⚠ **Pre-reasoning:** compliance not detected — include a 2-liner on what "
    "the structural trace surfaced and how it guided your answer."
)


def on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    prompt = kwargs.get("user_message") or ""
    if not isinstance(prompt, str) or len(prompt.split()) < pr.MIN_WORDS:
        return None
    session_id = str(kwargs.get("session_id") or "")
    result = pr.analyze(prompt)
    if not result or int(result.get("n_blocks") or 0) == 0:
        pr.clear_turn_state()
        return None
    pr.save_turn_state(prompt, int(result.get("n_blocks") or 0), session_id=session_id)
    trace = pr.format_trace(result)
    return {"context": trace} if trace else None


def on_post_llm_call(**kwargs: Any) -> None:
    assistant = kwargs.get("assistant_response") or ""
    if not isinstance(assistant, str):
        assistant = ""
    problem = ""
    if os.path.exists(pr.PROBLEM_CACHE):
        try:
            with open(pr.PROBLEM_CACHE, encoding="utf-8") as f:
                problem = f.read().strip()
        except OSError:
            pass
    reprompt = os.path.exists(pr.REPROMPT_FLAG)
    compliant = pr.turn_compliance_ok(assistant)
    pulse = {"status": "SKIP"}
    if problem and assistant:
        pulse = pr.pulse_local(problem, assistant)
    pr.ensure_state_dir()
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "session_id": kwargs.get("session_id"),
                    "source": "plugin",
                    "reprompt_expected": reprompt,
                    "compliance_markers": compliant,
                    "pulse_status": pulse.get("status"),
                    "pulse_gaps": pulse.get("gaps", []),
                },
                ensure_ascii=False,
            )
            + "\\n"
        )


def on_transform_llm_output(**kwargs: Any) -> str | None:
    text = kwargs.get("response_text") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    if not os.path.exists(pr.TURN_META) and not os.path.exists(pr.PROBLEM_CACHE):
        return None
    if pr.turn_compliance_ok(text):
        if FOOTER_OK.strip() in text:
            return None
        return text + FOOTER_OK
    if FOOTER_MISSING.strip() in text:
        return None
    return text + FOOTER_MISSING


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
'''

    pre_llm = '''#!/usr/bin/env python3
"""Hermes pre_llm_call — local Mia Pre-Reasoning."""
from __future__ import annotations
import json, os, sys
HOOKS = os.path.expanduser("~/.hermes/agent-hooks")
if HOOKS not in sys.path:
    sys.path.insert(0, HOOKS)
import pre_reasoning_lib as pr  # noqa: E402

def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print("{}"); return
    extra = payload.get("extra") or {}
    prompt = (extra.get("user_message") or "").strip() if isinstance(extra, dict) else ""
    if len(prompt.split()) < pr.MIN_WORDS:
        print("{}"); return
    result = pr.analyze(prompt)
    if not result or int(result.get("n_blocks") or 0) == 0:
        pr.clear_turn_state(); print("{}"); return
    pr.save_turn_state(prompt, int(result.get("n_blocks") or 0), session_id=str(payload.get("session_id") or ""))
    trace = pr.format_trace(result)
    print(json.dumps({"context": trace}, ensure_ascii=False) if trace else "{}")

if __name__ == "__main__":
    main()
'''

    post_llm = '''#!/usr/bin/env python3
"""Hermes post_llm_call — audit local pre-reasoning usage."""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
HOOKS = os.path.expanduser("~/.hermes/agent-hooks")
if HOOKS not in sys.path:
    sys.path.insert(0, HOOKS)
import pre_reasoning_lib as pr  # noqa: E402
AUDIT_LOG = os.path.expanduser("~/.hermes/hooks/pre_reasoning/audit.log")

def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print("{}"); return
    extra = payload.get("extra") or {}
    assistant = extra.get("assistant_response") if isinstance(extra, dict) else ""
    assistant = assistant if isinstance(assistant, str) else ""
    problem = ""
    if os.path.exists(pr.PROBLEM_CACHE):
        try:
            with open(pr.PROBLEM_CACHE, encoding="utf-8") as f:
                problem = f.read().strip()
        except OSError:
            pass
    pulse = pr.pulse_local(problem, assistant) if problem and assistant else {"status": "SKIP"}
    pr.ensure_state_dir()
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": payload.get("session_id"),
            "reprompt_expected": os.path.exists(pr.REPROMPT_FLAG),
            "compliance_markers": pr.turn_compliance_ok(assistant),
            "pulse_status": pulse.get("status"),
            "pulse_gaps": pulse.get("gaps", []),
        }, ensure_ascii=False) + "\\n")
    print("{}")

if __name__ == "__main__":
    main()
'''

    (hooks_dir / "pre_reasoning_lib.py").write_text(lib_module, encoding="utf-8")
    _write_executable(hooks_dir / "hermes_pre_llm.py", pre_llm)
    _write_executable(hooks_dir / "hermes_post_llm.py", post_llm)
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(plugin_init, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(PLUGIN_YAML, encoding="utf-8")
    print(f"Installed agent-hooks -> {hooks_dir}")
    print(f"Installed plugin -> {plugin_dir}")
    print("Install engine into Hermes Python (required for local-only traces):")
    print("  uv pip install -e /path/to/pre-reasoning --python ~/.hermes/hermes-agent/venv/bin/python3")
    print("Merge config snippet: python3 hermes_agent_hooks.py --print-config")
    print("Then Cmd+Q Hermes and reopen.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes + Mia Pre-Reasoning (local only)")
    parser.add_argument("--install", action="store_true", help="Write ~/.hermes/agent-hooks and plugin")
    parser.add_argument("--print-config", action="store_true", help="Print config.yaml snippet")
    args = parser.parse_args()
    if args.install:
        install()
    elif args.print_config:
        print(HERMES_CONFIG_SNIPPET)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()