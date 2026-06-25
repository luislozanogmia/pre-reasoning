# Claude Code Enforcer Hooks

Drop-in hooks for Claude Code that enforce pre-reasoning compliance. The model cannot opt out -- these run at the harness level, outside the model's control.

## Requirements

```bash
pip install pre-reasoning
```

## Setup

Copy both hooks and add them to your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/hooks/user_prompt_submit.py",
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
            "command": "python3 /path/to/hooks/stop_enforcer.py",
            "timeout": 15
          }
        ]
      }
    ]
  }
}
```

## How They Work

### `user_prompt_submit.py` (UserPromptSubmit)

Runs `analyze()` on every substantive prompt (8+ words) and injects the structural trace before the model responds. The trace can include derived assumptions from the built-in closure expert.

- **0 blocks**: conversational prompt, skips silently.
- **1-4 blocks**: injects trace + tells the model to re-run with richer input.
- **5+ blocks**: injects trace as grounding.

### `stop_enforcer.py` (Stop)

Runs when the model tries to finish. Two checks:

1. **Reprompt compliance**: if the trace was weak (<5 blocks), verifies the model re-ran `analyze()`. Blocks the stop if it didn't.
2. **Pulse check**: runs `pulse(problem, response)` to verify root blockers were addressed. Blocks the stop if gaps remain.

The hooks do not replace the model's answer. They provide before/after enforcement: trace before drafting, pulse check before completion.

## Why Enforcer Hooks

A skill or system prompt can ask the model to use pre-reasoning. The model can ignore the ask. These hooks enforce compliance at the harness level -- `analyze()` runs before the model sees the prompt, `pulse()` runs before the model is allowed to stop. The model grounds whether it wants to or not.
