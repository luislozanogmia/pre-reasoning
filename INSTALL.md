# Install Pre-Reasoning

## Install from this bundle

From the GitHub checkout (run this inside the `1M/` directory):

```bash
python -m pip install .
```

For a local editable install while developing:

```bash
pip install -e .
```

The 1M release is not the currently published PyPI build. Do not use
`pip install pre-reasoning` when you need this exact checkpoint.

Verify the checkpoint and runtime:

```bash
pre-reasoning --info
```

Print the form contract without loading the checkpoint:

```bash
pre-reasoning --form
```

## Core integration

The outer AI must interpret the original language and write the structured form. Do not pass arbitrary user prose directly to the engine.

```python
from pre_reasoning import (
    analyze_form,
    coverage_check_result,
    get_form,
    start_focus_mode,
)

contract = get_form()

form = """DEPENDENCIES
Frontend depends on API.
API depends on Auth.
Auth depends on Key Management.
Key Management depends on HSM Provisioning.
HSM Provisioning depends on Procurement.
"""

analysis = analyze_form(form)
print(analysis["trace"])

# Optional lexical presence check after the AI drafts its second-pass answer.
# This does not verify semantic correctness.
draft = "Resolve Procurement, HSM Provisioning, Key Management, Auth, and API before Frontend."
print(coverage_check_result(analysis, draft))

# Scheduler-backed Focus Mode for longer work sessions.
focus = start_focus_mode()
print(focus.scheduler_request())

# On each scheduled run, build a fresh form from the current work state.
print(focus.pulse(form, force=True)["trace"])
```

The form must contain at least five valid blocks. If it is shorter, the result has
`status: REPROMPT_REQUIRED`, an explicit alarm, and an `attached_form` template
for the AI to use on the next attempt. The important step happens outside this
package: place the returned trace beside the original context and let the AI
continue in a new forward pass.

An automation-capable calling model should immediately consume
`focus.scheduler_request()` and create or update one recurring ten-minute task
in the current chat. The package does not create a sleeper, background process,
or OS crontab entry. `focus.check()` is only the fallback when the host has no
scheduling capability. Every successful pulse resets the local fallback timer.

## Optional agent skill

```bash
mkdir -p ~/.claude/skills/pre-reasoning
cp skill/SKILL.md ~/.claude/skills/pre-reasoning/SKILL.md
```

The skill teaches the AI to construct the form, call `analyze_form()`, and use the returned trace before answering.

## Optional Claude Code hooks

The hooks inject the form-first obligation before substantive turns and verify that the agent used pre-reasoning before stopping. They do not parse the user's prose or compute a trace themselves.

```bash
mkdir -p ~/.claude/hooks/pre-reasoning
cp hooks/user_prompt_submit.py ~/.claude/hooks/pre-reasoning/user_prompt_submit.py
cp hooks/stop_enforcer.py ~/.claude/hooks/pre-reasoning/stop_enforcer.py
chmod +x ~/.claude/hooks/pre-reasoning/*.py
```

Merge into `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/pre-reasoning/user_prompt_submit.py",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/pre-reasoning/stop_enforcer.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

The resulting turn is:

```text
User context
  -> hook reminds AI to build the form
  -> AI calls analyze_form(form_text)
  -> tool result creates a new forward pass
  -> AI answers using original context + external trace
```
