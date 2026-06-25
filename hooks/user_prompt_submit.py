#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook -- local pre-reasoning grounding.

Runs the bundled engine (V3 neural + V2 harness) on every substantive
prompt and injects the structural trace as additionalContext. No network,
no remote API.

Enforcer behavior:
  - 0 blocks: conversational prompt, skip silently.
  - 1-4 blocks: inject trace + tell the model to re-run with richer input.
  - 5+ blocks: inject trace as grounding, no reprompt needed.

Install: pip install pre-reasoning
Hook event: UserPromptSubmit
Timeout: 10s recommended

Settings.json snippet:
  {
    "hooks": {
      "UserPromptSubmit": [{
        "matcher": "*",
        "hooks": [{
          "type": "command",
          "command": "python3 /path/to/user_prompt_submit.py",
          "timeout": 10
        }]
      }]
    }
  }
"""

import os
import sys
import json

_self_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _self_dir]

MIN_WORDS = 16  # Auto-hook runs only above 15 words; voluntary analyze() anytime (SOUL.md)
REPROMPT_FLAG = "/tmp/claude_prereasoning_reprompt_needed"
PROBLEM_CACHE = "/tmp/claude_prereasoning_problem.txt"

FORM_TIPS = (
    "The trace was weak (<5 blocks). You MUST build a better prompt and "
    "re-run pre-reasoning yourself using:\n"
    '  python3 -c "from pre_reasoning import analyze; '
    "r = analyze('<your improved prompt>'); print(r['trace'])\"\n\n"
    "Use these signal patterns to enrich your prompt:\n"
    "- Dependencies: 'depends on', 'requires', 'needs', 'blocks'\n"
    "- Blockers: 'is slow', 'fails', 'breaks', 'times out', 'is missing'\n"
    "- Options: 'Option A: ... Option B: ...'\n"
    "- Stakeholders: 'CTO wants X', 'dev warns Y', 'founder insists Z'\n"
    "- Constraints: 'team of N', 'deadline N weeks', 'budget $X'\n"
    "- Pain points: '1) issue 2) issue 3) issue'\n\n"
    "Rewrite the problem with these patterns, call analyze(), THEN answer "
    "using the richer trace.\n\n"
    "REQUIRED 2-LINER when the hook had <5 blocks (include both block counts):\n"
    "**Pre-reasoning:** The hook trace was weak (N block(s)). I re-ran "
    "pre_reasoning.analyze after enriching the problem and got M blocks — "
    "<insight>. That guided <answer>."
)


def format_trace(result):
    trace = (result.get("trace") or "").strip()
    if not trace:
        return None

    grounding = result.get("grounding_level", "unknown")
    enhanced = result.get("l1_enhanced", False)
    tag = "grounding:" + grounding + (" +LLM" if enhanced else "")
    n_blocks = result.get("n_blocks", 0)

    header = (
        "--- STRUCTURAL TRACE ---\n"
        "This is a map of the situation -- not a perfect map, but a "
        "grounding map.\nIt gives you the opportunity to see the whole "
        "picture before committing to tokens.\n"
        "Now, create the solution on your own using this map."
    )

    note = (
        "NOTE: You are requested to use the pre-reasoning blocks above "
        "to think about the solution. This is not optional. You MUST show "
        "a 2-liner in your response: what pre-reasoning surfaced and how "
        "it guided your answer."
    )

    if n_blocks < 5:
        note += "\n\n" + FORM_TIPS

    return "[Mia Pre-Reasoning | {}]\n{}\n\n{}\n\n{}".format(
        tag, header, trace, note
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    prompt = (payload.get("prompt") or "").strip()
    if len(prompt.split()) < MIN_WORDS:
        sys.exit(0)

    try:
        if os.path.exists(REPROMPT_FLAG):
            os.remove(REPROMPT_FLAG)

        from pre_reasoning import analyze

        result = analyze(prompt)
        n_blocks = result.get("n_blocks", 0)

        if n_blocks == 0:
            if os.path.exists(PROBLEM_CACHE):
                os.remove(PROBLEM_CACHE)
            sys.exit(0)

        with open(PROBLEM_CACHE, "w") as f:
            f.write(prompt)

        if n_blocks < 5:
            with open(REPROMPT_FLAG, "w") as f:
                f.write(str(n_blocks))

        trace = format_trace(result)
        if trace:
            sys.stdout.write(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": trace,
                }
            }) + "\n")
            sys.stdout.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
