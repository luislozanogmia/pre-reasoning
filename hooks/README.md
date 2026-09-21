# Optional Claude Code Hooks

These hooks enforce the form-first Pre-Reasoning loop on substantive turns. They do not parse the user's prompt, run the model on raw prose, require an arbitrary number of blocks, or force the final answer to disclose internal workflow.

`user_prompt_submit.py` injects the structured-form contract and records that the turn requires pre-reasoning. The AI interprets the original context, writes supported statements, and calls `pre_reasoning.analyze_form(form_text)`. `stop_enforcer.py` checks the current turn's assistant-side transcript for that call before allowing completion.

The form should contain at least five valid blocks. A shorter form returns a
`REPROMPT_REQUIRED` alarm with the submitted form and an attached template so
the calling AI can reprompt without silently inventing relationships.

## Setup

```bash
pip install pre-reasoning
chmod +x /path/to/hooks/*.py
```

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
            "timeout": 5
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
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

The stop check is deliberately narrow: it verifies use of the external engine, while semantic form quality remains the responsibility of the calling AI.
