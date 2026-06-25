# Install Pre-Reasoning

This project has two integration levels:

1. Install the Python package so a person or model can run `pre-reasoning`, `analyze()`, and `pulse()`.
2. Optionally install Claude Code hooks so pre-reasoning runs before and after model responses.

The hooks are optional. The Python package is the required part.

## 1. Install The Package

From PyPI:

```bash
pip install pre-reasoning
```

From a local checkout:

```bash
pip install -e .
```

Verify it:

```bash
pre-reasoning --info
```

Expected: the command prints engine info and reports the bundled neural engine availability.

## 2. Run It Manually

CLI:

```bash
pre-reasoning "Frontend depends on API. API depends on Auth."
```

Python:

```python
from pre_reasoning import analyze, pulse

problem = "Frontend depends on API. API depends on Auth."
result = analyze(problem)
print(result["trace"])

draft = "Resolve Auth first, then verify API, then unblock Frontend."
check = pulse(problem, draft)
print(check)
```

Use `analyze()` before answering. The result includes `blocks`, `derived_blocks`, and `derive_meta` in addition to the trace. Use `pulse()` before stopping.

## 3. Optional Skill Install

If the user has Claude skills enabled, copy the skill descriptor:

```bash
mkdir -p ~/.claude/skills/pre-reasoning
cp skill/SKILL.md ~/.claude/skills/pre-reasoning/SKILL.md
```

This tells the model how to call the engine, but it does not force the model to use it.

## 4. Optional Claude Code Hooks

The hooks enforce pre-reasoning outside the model. They are for Claude Code.

Copy the hook files somewhere stable:

```bash
mkdir -p ~/.claude/hooks/pre-reasoning
cp hooks/user_prompt_submit.py ~/.claude/hooks/pre-reasoning/user_prompt_submit.py
cp hooks/stop_enforcer.py ~/.claude/hooks/pre-reasoning/stop_enforcer.py
chmod +x ~/.claude/hooks/pre-reasoning/*.py
```

Add this to `~/.claude/settings.json`, merging with any existing settings:

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
            "timeout": 15
          }
        ]
      }
    ]
  }
}
```

If `~` is not expanded by the host, replace it with the absolute home path.

## 5. How The Hooks Work

`hooks/user_prompt_submit.py` runs before the model answers.

- It reads the user prompt.
- If the prompt is substantive, it calls `pre_reasoning.analyze(prompt)`.
- It injects the structural trace as extra context, including derived assumptions when transitive dependency pressure is detected.
- If the trace is weak, it tells the model to rewrite the problem with stronger structural signals and run `analyze()` again.

`hooks/stop_enforcer.py` runs when the model tries to finish.

- If the first trace was weak, it checks whether the model reran `analyze()`.
- It calls `pre_reasoning.pulse(problem, response)`.
- If root blockers were missed, it blocks the stop and tells the model what gaps to address.

So the flow is:

```text
User prompt
  -> UserPromptSubmit hook
  -> analyze(prompt)
  -> trace injected into model context
  -> model writes answer
  -> Stop hook
  -> pulse(original_problem, model_response)
  -> allow stop or require revision
```

## 6. What To Tell Another AI

If handing this repo to another model, give it this instruction:

```text
Install the package with `pip install pre-reasoning` or `pip install -e .`.
Before answering complex prompts, run `from pre_reasoning import analyze` and use
`analyze(problem)["trace"]` to ground the answer. Before finishing, run
`pulse(problem, draft_response)` and revise if it returns CONTINUE.

If this is Claude Code, optionally install the two hooks in `hooks/`:
`user_prompt_submit.py` runs before the answer and injects the trace;
`stop_enforcer.py` runs after the draft and blocks completion if root blockers
were missed.
```
