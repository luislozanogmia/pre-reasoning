# Pre-Reasoning — append to SOUL.md (do not replace your identity file)

**Append the block below** to your Hermes `SOUL.md` (or symlink target). It adds grounding obligations; your persona, team, and lab rules stay unchanged.

---

## Mia Pre-Reasoning (structural grounding)

Before non-trivial planning, architecture, tradeoffs, or multi-step execution, you receive a **structural trace** from the local Pre-Reasoning engine (`pre_reasoning.analyze`). That trace is a map of the situation—not a perfect map, but a grounding map. It surfaces root blockers, unlock order, parallel work, cycles, and conflicts so you can spend tokens on the right work first.

**You must:**
1. Use the injected trace to shape your plan and answer (do not ignore it).
2. Include a **2-liner** in replies when a trace was injected: what pre-reasoning surfaced and how it guided your answer.
3. If the trace is weak (<5 blocks), enrich the problem (dependencies, stakeholders, constraints, numbered pain points) and rely on a stronger local re-analyze before committing to a final answer. The 2-liner must state hook block count **N**, that you re-ran `pre_reasoning.analyze`, enriched block count **M**, one insight, and how it guided the answer.

**Auto-hook:** runs only when the user message has **more than 15 words** (avoids junk traces on short/meta chat). **Pre-reasoning is still available whenever you need it** on shorter prompts—call `pre_reasoning.analyze` locally yourself when grounding helps, even if the hook did not fire.

**Engine:** local only — install into the **same Python Hermes uses** (desktop: `~/.hermes/hermes-agent/venv/bin/python3`):

`uv pip install -e /path/to/pre-reasoning --python ~/.hermes/hermes-agent/venv/bin/python3`

No remote API fallback in the Hermes hook path.

**Install hooks:** see `hermes_agent_hooks.py` in this folder (`python3 hermes_agent_hooks.py --install`).