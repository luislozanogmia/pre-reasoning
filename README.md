# Pre-Reasoning

> This directory is the self-contained 1M-parameter GitHub staging bundle
> (`v4.0.1`). Install from this directory until this release is published to
> PyPI.

Pre-Reasoning gives an AI a structural checkpoint between forward passes. The AI first interprets the user's context and writes a literally "prompt-themselves" via a form. The bundled 1M model analyzes that form and returns an external trace. The AI then reads the original problem again with the trace in context before it commits to an answer.

This is not a chatbot nor can or intended to understand random word. Think of it as a intelligence but narrow engine it can binds entities to placeholders, runs learned dependency, composition, cycle, conflict, requirement, and conditional operations, restores the original names, and renders the trace but needs someone else to feed the correct prompt, that your agent.

Why we need the pre-reasoning, the models are great at understand and reasoning but without initial grounding they can take too much time to surface the right knowledge or to complete the task, to make LLMs more efficient we build tools around it.

If we map to brain functions:
1. Pre-reasoning: Prefrontal cortex: organizes the problem.
2. LLM: Association cortex: interprets and composes representations.
3. Harness: Motor cortex: converts the result into an action or tool execution

# Highlights
I had a breakthrough when I found out that loss function optimization for the capability and using the cooking recipe for language capability was the winning bet. It helped us to reduce the engine from 13.74M parameters in V3 to 1.02M parameters: **92.6% fewer parameters**. It preserves the five-family structured contract and improves the engine by removing the V3's fixed 32-entity ceiling by processing large graphs through learned operator windows.

## The loop

```text
Original context
  -> AI writes structured form
  -> 1M pre-reasoning model analyzes the form
  -> external structural trace
  -> AI starts its next forward pass with context + trace
  -> grounded answer
```


## Usage

```python
from pre_reasoning import analyze_form, get_form

print(get_form()["template"])

# Written by the calling AI after it interprets the user's context.
form = """DEPENDENCIES
Production Launch depends on Security Review.
Security Review depends on Architecture Approval.

CONFLICTS
Fast Rollout conflicts with Safety Review.

REQUIREMENTS
Test Coverage must be at least 95.
Support Training must be at least 90.
"""

result = analyze_form(form)
print(result["trace"])
```

The model derives `Production Launch depends on Architecture Approval`, identifies `Architecture Approval` as the root blocker, and returns the unlock order. Feed that trace back to the calling AI together with the original context.

Use `analyze_form(form_text)` for the initial analysis pass. It requires the structured form and at least five valid blocks. Fewer than five blocks returns a `REPROMPT_REQUIRED` alarm with the exact submitted form and an attached form template; invalid input raises `FormError` with the exact submitted form in `error.form_text`. Nothing is interpreted silently. Focus Mode uses the same contract for each later pulse.

## Focus Mode pulses

Focus Mode is a scheduler-backed ten-minute reflection loop for long-running work. When the mode starts, it gives the calling model an explicit request to create or update one recurring task attached to the current chat. Every scheduled run asks the AI to pause, re-read the original problem and latest state, write a fresh form, and run the 1M checkpoint before choosing its next move.

```python
from pre_reasoning import start_focus_mode

focus = start_focus_mode()  # ten minutes by default

# Give this request to the host's automation capability. A model using the
# bundled skill is instructed to do this automatically when Focus Mode starts.
print(focus.scheduler_request())

# The recurring task creates a current-state form on each run.
current_form = """DEPENDENCIES
Release depends on Verification.
Verification depends on Test Completion.
Test Completion depends on Implementation.
Implementation depends on Design Approval.
Design Approval depends on Scope Confirmation.
"""
pulse_result = focus.pulse(current_form, force=True)
print(pulse_result["trace"])
```

The Python package cannot create a host automation directly, so `scheduler_request()` provides the exact destination, cadence, instruction, and durable task prompt. An automation-capable model must consume it immediately. In Codex, this should be a recurring task in the current chat so each run retains the active context; a standalone scheduled job is the wrong default. `focus.check()` remains a fallback for hosts without scheduling support. The package never starts a sleeping process or background thread.

A successful pulse resets the local fallback timer. A form with fewer than five blocks returns the same `REPROMPT_REQUIRED` alarm and remains due until corrected. The scheduled task should be paused or removed when the substantive work ends.

For a one-off pulse without a session timer, call `pulse(form_text)`. The previous lexical response checker remains available as `coverage_check(form_text, response)` or `coverage_check_result(analysis, response)`. It only verifies textual presence; it does not prove that a response is correct or semantically resolves the trace.

## Capability families

| Family | Learned operation | Submitted form |
|---|---|---|
| F1 | Dependency direction and cycle status | Dependency statements |
| F2 | Symmetric conflict detection | Conflict statements |
| F3 | Numeric requirement operator | Requirement statements |
| F4 | Conditional role binding | Conditional statements |
| F5 | Transitive composition | Derived from F1 dependency chains |

```text
<dependent> depends on <prerequisite>.
<entity> conflicts with <entity>.
<entity> must be at least <number>.
<entity> must be at most <number>.
<entity> must be exactly <number>.
If <condition> passes, then <consequence> can proceed.
If <condition> passes, then <consequence> can proceed, otherwise <alternative> must proceed.
```

## Release contents

- `pre_reasoning/engine.py` — form validation, neural operator adapter, trace rendering
- `pre_reasoning/inference.py` — exact inference-only architecture
- `pre_reasoning/checkpoints/pre-reasoning-1m.safetensors` — weights-only checkpoint
- `skill/SKILL.md` — agent protocol for form creation, analysis, and reprompting
- `hooks/` — optional Claude Code enforcement of the form-first loop
- `examples/` — minimal integrations

Training code, optimizer state, legacy deterministic reasoning engines, platform-specific sidecars, and previous checkpoints are not part of the release.

See `EVALS.md` for the behavioral scope and `loss_function_optimization.md` for the capability-aware objective.

## License

MIT. See `LICENSE`.
