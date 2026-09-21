#!/usr/bin/env python3
"""Claude Code hook that verifies a required form-first engine call occurred."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


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


def _assistant_records_after_last_user(transcript_path: str) -> str:
    if not transcript_path or not os.path.isfile(transcript_path):
        return ""
    records: list[dict] = []
    try:
        with open(transcript_path, encoding="utf-8") as transcript:
            for line in transcript:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return ""

    last_user = -1
    for index, record in enumerate(records):
        role = record.get("type") or (record.get("message") or {}).get("role")
        if role == "user":
            last_user = index

    assistant = []
    for record in records[last_user + 1 :]:
        role = record.get("type") or (record.get("message") or {}).get("role")
        if role == "assistant":
            assistant.append(json.dumps(record, sort_keys=True))
    return "\n".join(assistant)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(payload, dict):
        return

    flag = _flag_path(payload)
    if not flag.exists():
        return

    assistant_records = _assistant_records_after_last_user(
        str(payload.get("transcript_path") or "")
    )
    if "analyze_form" not in assistant_records:
        print(
            json.dumps(
                {
                    "decision": "block",
                    "reason": (
                        "This substantive turn requires pre-reasoning. Interpret "
                        "the user's language, write supported structured statements, "
                        "call pre_reasoning.analyze_form(form_text), then use the "
                        "returned trace before finishing. Do not submit raw prose."
                    ),
                }
            )
        )
        return

    try:
        flag.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
