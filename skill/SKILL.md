---
name: pre-reasoning
description: Externalize a problem as structured form, run the Pre-Reasoning Engine, and use its trace to ground a new forward pass before answering.
---

# Pre-Reasoning Skill

Use Pre-Reasoning when a substantive problem contains dependencies, possible conflicts, numeric constraints, conditional branches, or enough interacting parts that a second structural view would reduce mistakes.

## Required loop

1. Interpret the user's original language yourself.
2. Write only supported relationships in the structured form below. Preserve entity names exactly. Omit uncertain relationships; never invent structure merely to make the form larger. Submit at least five valid blocks; fewer blocks produce a `REPROMPT_REQUIRED` alarm with an attached template.
3. Call `analyze_form(form_text)`.
4. Read the returned trace beside the original context. This tool result creates the opportunity for a new forward pass: reconsider ordering, blockers, cycles, conflicts, requirements, and branches before drafting the answer.
5. For sustained work, start `FocusMode` and immediately consume `focus.scheduler_request()`. Use the host's native scheduling or automation capability to create or update one recurring task every ten minutes in the current chat. In Codex, use a current-thread heartbeat rather than a standalone scheduled project job. Reuse an existing matching Focus Mode task instead of creating a duplicate. Do not launch a sleeping shell process or OS crontab entry.
6. On every scheduled run, re-read the original goal and latest work state, write a fresh form with at least five supported blocks, and call `focus.pulse(form_text, force=True)` before choosing the next move. Keep a routine pulse internal; report only a changed plan, blocker, conflict, completion, or required user input. A short form remains due and must be reprompted with the attached template. Pause or remove the recurring task when substantive work ends.
7. If the host has no scheduling capability, disclose that limitation and use `focus.check()` at natural execution milestones as the fallback.
8. Optionally run `coverage_check_result(analysis, draft)` after drafting. This is a lexical presence check only; never treat `COMPLETE` as proof that the draft is correct or semantically resolves the trace.

Never pass raw user prose to the engine. The calling AI performs semantic interpretation; the bundled 1M model performs learned operations over the form.

## Structured form

```text
DEPENDENCIES
<dependent> depends on <prerequisite>.

CONFLICTS
<entity> conflicts with <entity>.

REQUIREMENTS
<entity> must be at least <number>.
<entity> must be at most <number>.
<entity> must be exactly <number>.

CONDITIONALS
If <condition> passes, then <consequence> can proceed.
If <condition> passes, then <consequence> can proceed, otherwise <alternative> must proceed.
```

Use only the applicable sections. Section headings are optional; use one structured statement per line.

## Call the engine

```python
from pre_reasoning import analyze_form, coverage_check_result, start_focus_mode

form = """DEPENDENCIES
Production Launch depends on Security Review.
Security Review depends on Architecture Approval.
Architecture Approval depends on Threat Model.
Threat Model depends on Risk Register.
Risk Register depends on System Inventory.
"""

analysis = analyze_form(form)
print(analysis["trace"])

draft = "Resolve Architecture Approval before Security Review and Production Launch."
check = coverage_check_result(analysis, draft)

# For a longer work session, immediately create/update the requested recurring
# current-chat task with the host's automation capability.
focus = start_focus_mode()
scheduler_request = focus.scheduler_request()

# On each scheduled run, reinterpret the current work state into a new form.
refreshed = focus.pulse(form, force=True)
```

Use root blockers first, preserve direct versus model-derived relationships, do not sequence parallel windows unnecessarily, and call cycles or conflicts out explicitly. The trace is a structural map, not new domain evidence and not a substitute for the original context.

The bundled 1M-parameter V4 weights-only checkpoint loads automatically. Set `PRE_REASONING_CHECKPOINT=/path/to/checkpoint.safetensors` only to test another architecture-compatible checkpoint.
