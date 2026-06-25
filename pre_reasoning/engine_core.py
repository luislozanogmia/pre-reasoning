#!/usr/bin/env python3
"""
engine_core.py -- Combined Reasoning Engine (core)
====================================================

Neural perception + deterministic graph reasoning.

Pipeline:
  1. V3 parses NL text -> extracts entities + relations (learned patterns)
  2. V3 runs neural inference -> cycle detection, requirement validation, conflict verification
  3. V3 findings are converted to V2 blocks
  4. V2 runs structural analysis -> root blockers, critical path, parallel tracks
  5. Output combines V2 structure + V3 neural enrichments

Mode:
  full: V3 neural perception + V2 harness graph analysis (always active)

Author: Dr. Shannon, Mia Labs
Date: 2026-05-19
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "pre-reasoning-12m-v3.safetensors"
CHECKPOINT_ENV = "PRE_REASONING_CHECKPOINT"


class ReasoningEngineV25:
    """Neural perception + deterministic graph reasoning."""

    VERSION = "3.0.0"

    def __init__(self, checkpoint_path=None, device="auto"):
        self._v3 = None
        self._v3_error = None
        env_checkpoint = os.environ.get(CHECKPOINT_ENV)
        self._checkpoint_path = str(checkpoint_path or env_checkpoint or DEFAULT_CHECKPOINT)

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._load_v3()

    def _load_v3(self):
        checkpoint = Path(self._checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Neural weights not found: {checkpoint}. "
                f"Bundled weights are expected at {DEFAULT_CHECKPOINT}; "
                f"use --checkpoint or set {CHECKPOINT_ENV} to override."
            )

        from .inference import ReasoningEngineV3
        self._v3 = ReasoningEngineV3(
            checkpoint_path=self._checkpoint_path,
            device=self._device,
        )

    @property
    def mode(self) -> str:
        return "full"

    def analyze(self, text: str) -> dict:
        t0 = time.perf_counter()

        v3_result = self._v3.analyze(text, mode="ls")
        v2_blocks = self._v3_to_v2_blocks(v3_result)

        v2_result = self._run_v2(v2_blocks)
        merged = self._enrich_with_neural(v2_result, v3_result)

        inference_ms = (time.perf_counter() - t0) * 1000
        merged["inference_ms"] = round(inference_ms, 1)
        merged["version"] = self.VERSION
        merged["mode"] = self.mode
        merged["neural_enriched"] = True

        self._last_result = merged
        return merged

    def analyze_blocks(self, blocks: list) -> dict:
        t0 = time.perf_counter()

        v2_result = self._run_v2(blocks)
        v3_result = self._v3.analyze_blocks(blocks, mode="ls")
        merged = self._enrich_with_neural(v2_result, v3_result)

        inference_ms = (time.perf_counter() - t0) * 1000
        merged["inference_ms"] = round(inference_ms, 1)
        merged["version"] = self.VERSION
        merged["mode"] = self.mode
        merged["neural_enriched"] = True

        self._last_result = merged
        return merged

    def pulse(self, original_problem: str, response: str) -> dict:
        analysis = self.analyze(original_problem)
        root_blockers = analysis.get("root_blockers", [])

        if not root_blockers:
            return {
                "status": "COMPLETE",
                "message": "No root blockers identified.",
                "gaps": [],
            }

        response_lower = response.lower()
        addressed = []
        gaps = []

        for rb in root_blockers:
            name = rb if isinstance(rb, str) else rb.get("name", rb.get("entity", ""))
            terms = [part.strip().lower() for part in name.split(",") if part.strip()]
            addressed_by_terms = bool(terms) and all(term in response_lower for term in terms)
            if name.lower() in response_lower or addressed_by_terms:
                addressed.append(name)
            else:
                gaps.append(name)

        if not gaps:
            return {
                "status": "COMPLETE",
                "message": f"All {len(root_blockers)} root blockers addressed.",
                "addressed": addressed,
                "gaps": [],
            }

        return {
            "status": "CONTINUE",
            "message": f"{len(gaps)} root blocker(s) not addressed in response.",
            "addressed": addressed,
            "gaps": gaps,
            "suggestion": f"Address: {', '.join(gaps)}",
        }

    def format_for_narrator(self) -> str:
        result = getattr(self, "_last_result", None)
        if result is None:
            return "--- STRUCTURAL TRACE (v3) ---\nNo analysis has been run yet.\n"
        return result.get("trace", "")

    def engine_info(self) -> dict:
        v3_info = self._v3.engine_info()
        return {
            "engine": "ReasoningEngineV25",
            "version": self.VERSION,
            "mode": self.mode,
            "v2": "active",
            "v3": "active",
            "v3_model": v3_info.get("model"),
            "v3_params": v3_info.get("params"),
            "v3_checkpoint": v3_info.get("checkpoint"),
            "v3_device": v3_info.get("device"),
        }

    def get_form(self) -> dict:
        return {
            "engine": "ReasoningEngineV25",
            "version": self.VERSION,
            "type": "hybrid (neural perception + deterministic reasoning)",
            "modes": {
                "full": "V3 neural perception + V2 harness graph analysis (always active)",
            },
            "current_mode": self.mode,
            "input": {
                "text": "Natural language describing dependencies between entities.",
                "blocks": "Pre-extracted V2 blocks (list of dicts with family, entities, roles, source_clause, confidence).",
                "signal_words": {
                    "forward": ["depends on", "needs", "requires", "relies on",
                                "waits for", "blocked by", "must happen after"],
                    "reverse": ["enables", "allows", "unlocks", "must happen before",
                                "makes possible", "comes before"],
                    "conflict": ["contradicts", "conflicts with", "clashes with",
                                 "incompatible with", "opposes"],
                },
            },
            "output": {
                "root_blockers": "Entities/blocks that must be resolved first.",
                "unlock_sequence": "Topological order for resolution.",
                "parallel_work": "Independent items that can proceed in parallel.",
                "has_cycle": "Whether circular dependencies exist.",
                "conflicts": "Detected conflicts between entities (V3 neural).",
                "requirements": "Detected requirements with operators/values (V3 neural).",
                "critical_path_length": "Longest sequential chain (V2 structural).",
                "trace": "LLM-friendly formatted trace.",
            },
        }

    # ── V3 -> V2 conversion ──────────────────────────────────────────────

    def _v3_to_v2_blocks(self, v3_result: dict) -> list:
        blocks = []

        for dep in v3_result.get("dependencies", []):
            src_name = dep.get("src_name", dep.get("src", ""))
            tgt_name = dep.get("tgt_name", dep.get("tgt", ""))
            if not src_name or not tgt_name:
                continue
            blocks.append({
                "family": "dependency",
                "entities": [src_name, tgt_name],
                "roles": {"blocked": src_name, "blocker": tgt_name},
                "source_clause": f"{src_name} depends on {tgt_name}",
                "confidence": 0.85,
            })

        for conflict in v3_result.get("conflicts", []):
            a_name = conflict.get("a_name", conflict.get("a", ""))
            b_name = conflict.get("b_name", conflict.get("b", ""))
            if not a_name or not b_name:
                continue
            blocks.append({
                "family": "conflict",
                "entities": [a_name, b_name],
                "roles": {"initiator": a_name, "opposing": b_name},
                "source_clause": f"{a_name} conflicts with {b_name}",
                "confidence": 0.80,
            })

        return blocks

    # ── V2 execution ─────────────────────────────────────────────────────

    def _run_v2(self, blocks: list) -> dict:
        from .harness import ReasoningEngineV2

        engine = ReasoningEngineV2(blocks)

        root_blockers = []
        for rb in engine.root_blockers:
            ents = sorted(rb.entities, key=str.lower)[:3]
            root_blockers.append({
                "name": ", ".join(ents),
                "entity": ", ".join(ents),
                "impact": rb.impact_score,
                "source": rb.source[:100] if rb.source else "",
                "index": rb.index,
            })

        unlock_sequence = []
        for step_num, step_nodes in engine.resolution_steps:
            if step_num == 0:
                continue
            for n in step_nodes:
                ents = sorted(n.entities, key=str.lower)[:2]
                unlock_sequence.append({
                    "name": ", ".join(ents),
                    "entity": ", ".join(ents),
                    "step": step_num,
                    "is_root": n.is_root_blocker,
                })

        parallel_work = []
        for p in engine.parallel_tracks:
            ents = sorted(p.entities, key=str.lower)[:3]
            parallel_work.append({
                "name": ", ".join(ents),
                "entity": ", ".join(ents),
                "family": p.family,
                "blocks_chain": p.blocks_chain is not None,
                "enables_resolution": p.enables_resolution is not None,
            })

        v2_cycles = engine.circular_dependencies
        has_cycle = len(v2_cycles) > 0
        cycle_nodes = []
        for cycle in v2_cycles:
            for node in cycle:
                ents = sorted(node.entities, key=str.lower)[:2]
                cycle_nodes.append({
                    "name": ", ".join(ents),
                    "entity": ", ".join(ents),
                    "index": node.index,
                })

        trace = engine.format_for_narrator()

        return {
            "trace": trace,
            "root_blockers": root_blockers,
            "unlock_sequence": unlock_sequence,
            "parallel_work": parallel_work,
            "critical_path_length": engine.critical_path_length,
            "has_cycle": has_cycle,
            "cycle_nodes": cycle_nodes,
            "conflicts": self._v2_conflicts(blocks),
            "requirements": [],
            "grounding_level": self._compute_grounding(root_blockers, unlock_sequence, parallel_work),
            "n_blocks": len(blocks),
            "v3_confidence": {},
        }

    def _v2_conflicts(self, blocks: list) -> list:
        conflicts = []
        for block in blocks:
            if block.get("family") != "conflict":
                continue
            entities = block.get("entities", [])
            a = block.get("roles", {}).get("initiator") or (entities[0] if entities else "")
            b = block.get("roles", {}).get("opposing") or (entities[1] if len(entities) > 1 else "")
            conflicts.append({
                "a": a,
                "b": b,
                "a_name": a,
                "b_name": b,
                "source": block.get("source_clause", ""),
                "mode": "deterministic",
            })
        return conflicts

    # ── Neural enrichment ────────────────────────────────────────────────

    def _enrich_with_neural(self, v2_result: dict, v3_result: dict) -> dict:
        merged = dict(v2_result)

        v3_has_cycle = v3_result.get("has_cycle", False)
        v3_cycle_nodes = v3_result.get("cycle_nodes", [])
        if v3_has_cycle:
            merged["has_cycle"] = True
            if v3_cycle_nodes:
                merged["cycle_nodes"] = [
                    {"name": cn.get("name", cn.get("entity", "")),
                     "entity": cn.get("entity", "")}
                    for cn in v3_cycle_nodes
                ]

        v3_conflicts = v3_result.get("conflicts", [])
        if v3_conflicts:
            merged["conflicts"] = [
                {"a": c.get("a", ""), "b": c.get("b", ""),
                 "a_name": c.get("a_name", ""), "b_name": c.get("b_name", "")}
                for c in v3_conflicts
            ]

        v3_requirements = v3_result.get("requirements", [])
        if v3_requirements:
            merged["requirements"] = [
                {"entity": r.get("entity", ""), "operator": r.get("operator"),
                 "value": r.get("value"), "name": r.get("name", "")}
                for r in v3_requirements
            ]

        v3_confidence = {}
        if v3_result.get("dependencies"):
            v3_confidence["dependencies"] = len(v3_result["dependencies"])
        if v3_conflicts:
            v3_confidence["conflicts"] = len(v3_conflicts)
        if v3_requirements:
            v3_confidence["requirements"] = len(v3_requirements)
        if v3_result.get("root_blockers"):
            v3_confidence["root_blockers"] = len(v3_result["root_blockers"])
        merged["v3_confidence"] = v3_confidence

        merged["trace"] = self._build_enriched_trace(v2_result, v3_result)

        return merged

    def _build_enriched_trace(self, v2_result: dict, v3_result: dict) -> str:
        lines = ["--- STRUCTURAL TRACE (v3) ---"]
        lines.append("V3 neural perception + V2 deterministic reasoning.")
        lines.append("This is a map of the situation. Use it to ground your response.")
        lines.append("")

        root_blockers = v2_result.get("root_blockers", [])
        if root_blockers:
            lines.append("ROOT BLOCKERS (must resolve FIRST):")
            for i, rb in enumerate(root_blockers):
                name = rb.get("name", rb.get("entity", ""))
                impact = rb.get("impact", 0)
                lines.append(f"  Block [{i+1}]: {name}")
                if impact > 0:
                    lines.append(f"    Impact: unblocks {impact} downstream steps")
                src = rb.get("source", "")
                if src:
                    lines.append(f"    Evidence: '{src[:80]}'")
        else:
            lines.append("ROOT BLOCKERS: None identified")

        lines.append("")
        unlock_seq = v2_result.get("unlock_sequence", [])
        if unlock_seq:
            lines.append("UNLOCK SEQUENCE (optimal order):")
            for item in unlock_seq:
                name = item.get("name", item.get("entity", ""))
                step = item.get("step", "?")
                action = "Resolve" if item.get("is_root") else "Unblocked"
                lines.append(f"  Step {step}: {action} {name}")
        else:
            lines.append("UNLOCK SEQUENCE: N/A")

        lines.append("")
        parallel = v2_result.get("parallel_work", [])
        if parallel:
            lines.append("PARALLEL WORK (proceed now, independent):")
            for p in parallel:
                name = p.get("name", p.get("entity", ""))
                family = p.get("family", "")
                extra = ""
                if p.get("blocks_chain"):
                    extra = " [BLOCKS CHAIN]"
                if p.get("enables_resolution"):
                    extra = " [ENABLES RESOLUTION]"
                lines.append(f"  [{family}] {name}{extra}")
        else:
            lines.append("PARALLEL WORK: None")

        v3_conflicts = v3_result.get("conflicts", [])
        if v3_conflicts:
            lines.append("")
            lines.append("CONFLICTS (V3 neural detection):")
            for c in v3_conflicts:
                a = c.get("a_name", c.get("a", ""))
                b = c.get("b_name", c.get("b", ""))
                lines.append(f"  {a} CONTRADICTS {b}")

        v3_requirements = v3_result.get("requirements", [])
        if v3_requirements:
            lines.append("")
            lines.append("REQUIREMENTS (V3 neural detection):")
            for r in v3_requirements:
                name = r.get("name", r.get("entity", ""))
                op = r.get("operator", "")
                val = r.get("value", "")
                op_str = ">=" if op == "GEQ" else "<=" if op == "LEQ" else (op or "?")
                val_str = val.replace("VALUE_", "") if val and isinstance(val, str) and val.startswith("VALUE_") else (val or "?")
                lines.append(f"  {name} {op_str} {val_str}")

        has_cycle = v2_result.get("has_cycle", False) or v3_result.get("has_cycle", False)
        lines.append("")
        if has_cycle:
            cycle_nodes = v2_result.get("cycle_nodes", []) or v3_result.get("cycle_nodes", [])
            cycle_names = [cn.get("name", cn.get("entity", "")) for cn in cycle_nodes]
            lines.append(f"CYCLES DETECTED: {', '.join(cycle_names) if cycle_names else 'Yes'}")
            lines.append("  These depend on each other -- use alternative resolution strategy.")
        else:
            lines.append("CYCLES DETECTED: None")

        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _compute_grounding(self, root_blockers, unlock_sequence, parallel_work) -> str:
        n_rb = len(root_blockers)
        n_us = len(unlock_sequence)
        n_pw = len(parallel_work)
        total = n_rb + n_us + n_pw
        if total == 0:
            return "unlocking"
        if n_rb > 0 and n_us > 0:
            return "grounding"
        if n_us > 0:
            return "enhancing"
        return "unlocking"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="pre-reasoning",
        description="Pre-Reasoning v3 -- 12M Neural Engine",
    )
    parser.add_argument("text", nargs="?", help="Problem text to analyze")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional .safetensors weights override")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--form", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    engine = ReasoningEngineV25(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.info:
        print(json.dumps(engine.engine_info(), indent=2))
        return

    if args.form:
        print(json.dumps(engine.get_form(), indent=2))
        return

    if not args.text:
        print("Usage: pre-reasoning 'A enables B. B enables C.'")
        print("       pre-reasoning --info")
        print("       pre-reasoning --form")
        return

    result = engine.analyze(args.text)

    if args.json:
        result.pop("trace", None)
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result["trace"])


if __name__ == "__main__":
    main()
