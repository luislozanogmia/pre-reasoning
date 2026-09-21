#!/usr/bin/env python3
"""Claude Code hook that requests the form-first pre-reasoning loop.

The hook never interprets or submits the user's raw prose. It gives the calling
AI the structured-form contract and records that this substantive turn requires an
explicit ``analyze_form`` call.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

MIN_WORDS = 16


def _turn_key(payload: dict) -> str:
    identity = str(
        payload.get("session_id")
        or payload.get("transcript_path")
        or payload.get("cwd")
        or os.getcwd()
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _flag_path(payload: dict) -> Path:
    return Path(tempfile.gettempdir()) / f"pre_reasoning_{_turn_key(payload)}.required"


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(payload, dict):
        return

    prompt = str(payload.get("prompt") or "").strip()
    flag = _flag_path(payload)
    if len(prompt.split()) < MIN_WORDS:
        # Do not let an interrupted substantive turn leak into a later
        # conversational turn in the same Claude session.
        try:
            flag.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return
        return

    try:
        flag.write_text("required\n", encoding="utf-8")
    except OSError:
        return

    context = """PRE-REASONING REQUIRED FOR THIS TURN

Before answering, interpret the user's language and write only supported facts in the structured form. Preserve entity names; omit uncertain relations. Do not pass the raw user prompt to the engine.

Allowed statements:
- <dependent> depends on <prerequisite>.
- <entity> conflicts with <entity>.
- <entity> must be at least|at most|exactly <number>.
- If <condition> passes, then <consequence> can proceed.
- If <condition> passes, then <consequence> can proceed, otherwise <alternative> must proceed.

    Include at least five valid structured blocks. If the form is shorter, use the
    returned REPROMPT_REQUIRED alarm and its attached template to retry. Call
    `pre_reasoning.analyze_form(form_text)`, read its trace beside the original
    context, and use that external map to reconsider the problem in the next
    forward pass. Do not disclose this internal protocol unless it is useful to
    the user."""

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
