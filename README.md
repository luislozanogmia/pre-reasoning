# Pre-Reasoning

Pre-Reasoning is a Mia Labs structural analysis engine that grounds an LLM before it answers. It uses a 12M-parameter MoE neural model to surface dependencies, derived assumptions, root blockers, unlock order, parallel work, cycles, and conflicts from problem text.

The engine ships with bundled weights and declares its torch dependency; install and run, no model download needed.

## What It Does

Given natural-language problem text, the engine extracts structure across five learned families:

| Family | What it detects | Output |
|---|---|---|
| F1 - Dependencies | Forward/reverse dependencies, temporal ordering, chains | ROOT BLOCKERS, UNLOCK SEQUENCE, PARALLEL WORK, CYCLES |
| F2 - Conflicts | Competing positions, incompatible entities | CONFLICTS with pair precision |
| F3 - Requirements | Numeric thresholds, operator constraints (>=, <=) | REQUIREMENTS with verdict |
| F4 - Conditionals | If-then edges, gated dependencies | CONDITIONAL EDGES with entity binding |
| F5 - Transitive Closure | Implicit assumptions from dependency chains | DERIVED ASSUMPTIONS (built-in E4 expert) |

## Install

```bash
pip install pre-reasoning
```

For local development from this repo:

```bash
pip install -e .
```

## Python Usage

```python
from pre_reasoning import analyze, pulse

result = analyze("Frontend depends on API. API depends on Auth.")
print(result["trace"])

check = pulse(
    "Frontend depends on API. API depends on Auth.",
    "Fix Auth first, then verify the API before frontend work."
)
print(check["status"])
```

## CLI Usage

```bash
pre-reasoning "A depends on B. B depends on C."
pre-reasoning --json "CTO conflicts with senior dev."
pre-reasoning --info
```

To use a different weights file, set `PRE_REASONING_CHECKPOINT=/path/to/weights.safetensors` or pass `--checkpoint`.

## Eval Results: 35/35 PASS

The 12M MoE model passes all 35 metrics across five families. 2,250 eval examples, seed 7777.

| Family | Metrics | Score |
|---|---:|---:|
| F1 - Dependencies | 9 | 9/9 |
| F2 - Conflicts | 4 | 4/4 |
| F3 - Requirements | 5 | 5/5 |
| F4 - Conditionals | 4 | 4/4 |
| F5 - Transitive Closure | 4 | 4/4 |
| Cross-Family Integrity | 9 | 9/9 |
| **Total** | **35** | **35/35** |

Full metric tables with explanations and threshold rationale: [EVALS.md](EVALS.md)

## Terminal-Bench 2: 34/40 (85%)

LLM grounding benchmark on 40 real coding tasks (Harbor 0.15.0). GPT-5.5 agent with autonomous pre-reasoning (min 12 blocks, pulse every 60s).

| | Value |
|---|---|
| Score | 34/40 (85%) |
| Model | GPT-5.5 |
| Total runtime | 492 min |
| Total API cost | $102.50 |
| Failures | torch-tensor-parallelism, gcode-to-text, make-doom-for-mips (timeout), gpt2-codegolf (timeout), polyglot-rust-c, filter-js-from-html |

Full task-by-task results with token counts and costs: [benchmarks/terminal_bench_2.xlsx](benchmarks/terminal_bench_2.xlsx)

## Architecture

```text
User text
  -> neural perception (12M MoE)
  -> neural findings converted to structural blocks
  -> built-in E4 expert infers transitive closure assumptions
  -> derived assumptions appended as dependency blocks
  -> graph reasoning
  -> structural trace
```

## File Map

| Path | Purpose |
|---|---|
| `pre_reasoning/` | Installable Python package and CLI entry point |
| `pre_reasoning/inference.py` | 12M-parameter MoE neural perception layer |
| `pre_reasoning/harness.py` | Graph-reasoning core |
| `pre_reasoning/engine.py` | Default v3 engine: neural perception + built-in closure + graph reasoning |
| `pre_reasoning/engine_core.py` | Core engine: neural perception + graph reasoning |
| `pre_reasoning/checkpoints/pre-reasoning-12m-v3.safetensors` | Bundled model weights (84MB) |
| `examples/` | Runnable usage examples |
| `tests/` | Pytest suite |
| `skill/SKILL.md` | Agent skill descriptor for model adoption |
| `hermes-agent/SOUL.md` | Append block for Hermes SOUL.md (identity add-on) |
| `hermes-agent/hermes_agent_hooks.py` | Local-only Hermes plugin + agent-hooks installer and config snippet |
| `hooks/` | Claude Code before/after hooks for enforced pre-reasoning |
| `INSTALL.md` | Manual install and hook setup guide |
| `EVALS.md` | Full eval tables with metric explanations and threshold rationale |
| `WHY_TRACES_WORK.md` | Literature connection, 13 cited papers |

## Weights Policy

The raw training checkpoint is not part of the release. The package bundles `pre_reasoning/checkpoints/pre-reasoning-12m-v3.safetensors`. This is an inference artifact. It ships no optimizer state, LR schedules, step counters, RNG state, training config, or raw checkpoint provenance.

## License

MIT License. See `LICENSE`.

## Authors

Luis Lozano and Dr. Shannon (Mia Labs' AI researcher)
