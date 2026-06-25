#!/usr/bin/env python3
"""Claude Code Stop hook -- enforces pre-reasoning compliance.

The model cannot opt out. Two enforcement checks run before the model
is allowed to stop:

1. Reprompt compliance: if the UserPromptSubmit hook flagged a weak trace
   (<5 blocks) and the model did not re-run analyze() during its turn,
   block the stop and force a reprompt.

2. Pulse check: run pulse(original_problem, model_response) to verify that
   the model's response actually addressed the detected root blockers. If
   pulse returns CONTINUE, block the stop and list the gaps.

Install: pip install pre-reasoning
Hook event: Stop
Timeout: 15s recommended (pulse runs the neural model)

Settings.json snippet:
  {
    "hooks": {
      "Stop": [{
        "matcher": "*",
        "hooks": [{
          "type": "command",
          "command": "python3 /path/to/stop_enforcer.py",
          "timeout": 15
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

REPROMPT_FLAG = "/tmp/claude_prereasoning_reprompt_needed"
PROBLEM_CACHE = "/tmp/claude_prereasoning_problem.txt"
ANALYZE_MARKER = "pre_reasoning"


def extract_last_assistant_text(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    try:
        with open(transcript_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if obj.get("type") == "assistant":
                    return json.dumps(obj)
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception:
        pass
    return ""


def check_reprompt_compliance(assistant_text):
    markers = [ANALYZE_MARKER, "analyze(", "engine.py --analyze"]
    return any(m in assistant_text for m in markers)


def run_pulse(problem, response):
    try:
        from pre_reasoning import pulse
        return pulse(problem, response)
    except Exception:
        return {"status": "COMPLETE", "gaps": []}


def block(reason):
    sys.stdout.write(json.dumps({
        "decision": "block",
        "reason": reason,
    }) + "\n")
    sys.stdout.flush()
    sys.exit(0)


def main():
    stdin_raw = sys.stdin.read()
    flag_exists = os.path.exists(REPROMPT_FLAG)

    try:
        payload = json.loads(stdin_raw)
    except Exception:
        sys.exit(0)

    transcript_path = payload.get("transcript_path", "")
    assistant_text = extract_last_assistant_text(transcript_path)

    if flag_exists:
        try:
            os.remove(REPROMPT_FLAG)
        except OSError:
            pass

        if not check_reprompt_compliance(assistant_text):
            block(
                "Pre-reasoning returned fewer than 5 blocks and you did "
                "not re-run the engine. You MUST rewrite the user's "
                "problem using structural signal patterns (dependencies, "
                "blockers, options, stakeholders, constraints, pain "
                "points) and call: python3 -c \"from pre_reasoning import "
                "analyze; r = analyze('<enriched prompt>'); "
                "print(r['trace'])\" Then use the richer trace to guide "
                "your answer."
            )

    if flag_exists and assistant_text:
        problem = ""
        if os.path.exists(PROBLEM_CACHE):
            try:
                with open(PROBLEM_CACHE, "r") as f:
                    problem = f.read().strip()
            except Exception:
                pass

        if problem:
            result = run_pulse(problem, assistant_text)
            if result.get("status") == "CONTINUE":
                gaps = result.get("gaps", [])
                gap_list = (
                    "; ".join(str(g) for g in gaps)
                    if gaps
                    else "unaddressed root blockers"
                )
                block(
                    "Pre-reasoning pulse check returned CONTINUE. "
                    "Gaps: {}. You MUST address these root blockers "
                    "before finishing your response.".format(gap_list)
                )


if __name__ == "__main__":
    main()
