---
name: pre-reasoning
description: Use the Pre-Reasoning Engine to ground autonomous work with structural traces, derived assumptions, and root blockers before answering, then pulse-check the response before stopping.
---

# Pre-Reasoning Skill

## Install

```bash
pip install pre-reasoning
```

For a local repo checkout:

```bash
pip install -e .
```

## Ground Before Answering

```python
from pre_reasoning import analyze

problem = """A depends on B. B depends on C.
The CTO wants microservices, but the senior dev warns about complexity."""

trace = analyze(problem)["trace"]
print(trace)
```

The trace includes direct structural blocks plus derived assumptions from the built-in closure expert when transitive dependencies are present. Use it to order the response:

- resolve root blockers first
- follow the unlock sequence
- account for derived assumptions without treating them as direct user claims
- keep parallel work separate
- call out conflicts directly

## Pulse Before Stopping

```python
from pre_reasoning import pulse

check = pulse(problem, draft_response)
print(check)
```

If `status` is `CONTINUE`, revise the response to address the listed `gaps`, then call `pulse()` again. If `status` is `COMPLETE`, the response has addressed the detected root blockers.

Bundled 12M model weights are used automatically. Set `PRE_REASONING_CHECKPOINT=/path/to/weights.safetensors` only when testing another converted weights file.
