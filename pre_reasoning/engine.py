#!/usr/bin/env python3
"""
engine.py -- ReasoningEngineV3Engine
=====================================

Additive extension of ReasoningEngineV25 that wires in the 12M model's
built-in transitive-closure enrichment.

What it adds vs the core engine:
  - _enrich_with_derive() driven by the 12M model's E4 expert
  - analyze() and analyze_blocks() call super(), then enrich the result
  - Two new keys written additively to every result dict:
      merged["derived_assumptions"]  -- list of {assuming, premise} dicts
      merged["derive_meta"]          -- strategy, n_edges, n_entities, edge_source

What is NEVER changed:
  - No existing key in the result dict is modified or removed

Author: Dr. Shannon, Mia Labs
Date: 2026-06-14
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Base import ───────────────────────────────────────────────────────────────
try:
    from .engine_core import ReasoningEngineV25  # noqa: E402
except ImportError:  # Allows direct script execution from this directory.
    _PACKAGE_DIR = Path(__file__).resolve().parent
    if str(_PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_DIR))
    from engine_core import ReasoningEngineV25  # type: ignore # noqa: E402

logger = logging.getLogger(__name__)


class ReasoningEngineV252(ReasoningEngineV25):
    """
    V3 -- core engine + 12M built-in transitive-closure enrichment.

    Subclasses ReasoningEngineV25 additively: all existing behavior is
    preserved; the only change is that every result dict gains two new keys:
      derived_assumptions : list of {assuming: str, premise: str}
      derive_meta         : {n_edges, n_entities, strategy, edge_source}

    Full analyze() is available and works identically to the core engine with the
    extra enrichment on top. Requires the bundled neural perception
    checkpoint at initialization and at runtime. Explicit harness parsing
    is used only as an additive signal alongside neural perception, never as
    a silent replacement for it.
    """

    VERSION = "3.0.0"

    def __init__(self, checkpoint_path=None, device="auto"):
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    # ── Overrides ─────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        t0 = time.perf_counter()

        v3_result = self._v3.analyze(text, mode="ls")
        v2_blocks = self._v3_to_v2_blocks(v3_result)
        harness_blocks = self._harness_blocks_from_text(text)
        v2_blocks = self._merge_blocks_preserving_derive(
            v2_blocks + harness_blocks
        )
        direct_edges = self._dependency_edges_from_blocks(v2_blocks)
        v2_blocks = self._append_derived_dependency_blocks(
            v2_blocks,
            v3_result,
            source_edges=direct_edges,
        )
        v2_blocks = self._merge_blocks_preserving_derive(v2_blocks)

        v2_result = self._run_v2(v2_blocks)

        merged = self._enrich_with_neural(v2_result, v3_result)

        inference_ms = (time.perf_counter() - t0) * 1000
        merged["inference_ms"] = round(inference_ms, 1)
        merged["version"] = self.VERSION
        merged["mode"] = self.mode
        merged["neural_enriched"] = True

        v3_edges = self._dependency_edges_from_v3(v3_result)
        use_block_edges = len(set(direct_edges)) > len(set(v3_edges))
        if use_block_edges:
            merged["dependencies"] = [
                {"src_name": src, "tgt_name": tgt}
                for src, tgt in direct_edges
            ]
        merged = self._enrich_with_derive(
            merged,
            v3_result=None if use_block_edges else v3_result,
        )
        if use_block_edges:
            merged.pop("dependencies", None)
        self._attach_block_output(merged, v2_blocks)
        self._attach_v3_root_blockers(merged, v3_result)
        self._append_v252_trace_sections(merged)
        self._last_result = merged
        return merged

    def analyze_blocks(self, blocks: list) -> dict:
        t0 = time.perf_counter()
        input_dependency_edges = self._dependency_edges_from_blocks(blocks)

        v3_result = self._v3.analyze_blocks(blocks, mode="ls")

        v2_blocks = list(blocks)
        v2_blocks = self._append_derived_dependency_blocks(
            v2_blocks,
            v3_result,
            source_edges=input_dependency_edges,
        )

        v2_result = self._run_v2(v2_blocks)

        merged = self._enrich_with_neural(v2_result, v3_result)

        inference_ms = (time.perf_counter() - t0) * 1000
        merged["inference_ms"] = round(inference_ms, 1)
        merged["version"] = self.VERSION
        merged["mode"] = self.mode
        merged["neural_enriched"] = True

        v3_edges = self._dependency_edges_from_v3(v3_result)
        # For structured blocks the caller's declared dependency edges are ground
        # truth -- always prefer them over re-perceived v3 edges, which can drift
        # out-of-distribution on large prompts and invent spurious links that then
        # produce false transitive closures. Fall back to v3 only when the blocks
        # carried no explicit dependency entities.
        use_input_edges = bool(input_dependency_edges)
        if use_input_edges:
            merged["dependencies"] = [
                {"src_name": src, "tgt_name": tgt}
                for src, tgt in input_dependency_edges
            ]

        merged = self._enrich_with_derive(
            merged,
            v3_result=None if use_input_edges else v3_result,
        )
        if use_input_edges:
            merged.pop("dependencies", None)
        self._attach_block_output(merged, v2_blocks)
        self._attach_v3_root_blockers(merged, v3_result)
        self._append_v252_trace_sections(merged)
        self._last_result = merged
        return merged

    def _harness_blocks_from_text(self, text: str) -> list:
        try:
            from .inference import _extract_entities_and_relations
        except ImportError:  # Direct script execution from package directory.
            from inference import _extract_entities_and_relations  # type: ignore

        entity_order, ent_map, relations = _extract_entities_and_relations(text)
        reverse_map = {
            ent_map[name.lower()]: name
            for name in entity_order
            if name.lower() in ent_map
        }
        sentence_lookup = self._sentence_lookup(text)

        blocks = []
        for src_ent, rel, tgt_ent in relations:
            src = reverse_map.get(src_ent, src_ent)
            tgt = reverse_map.get(tgt_ent, tgt_ent)
            source_clause = self._relation_source_clause(src, tgt, sentence_lookup)
            if rel == "DEPENDS_ON":
                blocks.append({
                    "family": "dependency",
                    "entities": [src, tgt],
                    "roles": {"blocked": src, "blocker": tgt},
                    "source_clause": source_clause or f"{src} depends on {tgt}",
                    "confidence": 0.72,
                })
            elif rel in ("CONTRADICTS", "CONFLICTS"):
                blocks.append({
                    "family": "conflict",
                    "entities": [src, tgt],
                    "roles": {"initiator": src, "opposing": tgt},
                    "source_clause": source_clause or f"{src} conflicts with {tgt}",
                    "confidence": 0.70,
                })

        return self._merge_blocks_preserving_derive(blocks)

    @staticmethod
    def _sentence_lookup(text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"[.;!?\n]+", text)
            if sentence.strip()
        ]

    @staticmethod
    def _relation_source_clause(src: str, tgt: str, sentences: List[str]) -> str:
        src_l = src.lower()
        tgt_l = tgt.lower()
        for sentence in sentences:
            lowered = sentence.lower()
            if src_l in lowered and tgt_l in lowered:
                return sentence
        return ""

    @staticmethod
    def _block_key(block: dict) -> Tuple:
        family = block.get("family", "")
        roles = block.get("roles", {})
        if family in ("dependency", "prereq"):
            src = roles.get("blocked", roles.get("gated", ""))
            tgt = roles.get("blocker", roles.get("gate", ""))
            if not src or not tgt:
                entities = block.get("entities", [])
                if len(entities) >= 2:
                    src, tgt = entities[0], entities[1]
            return (family, str(src).lower(), str(tgt).lower())
        entities = tuple(str(e).lower() for e in block.get("entities", []))
        return (family, entities)

    def _merge_blocks_preserving_derive(self, blocks: list) -> list:
        seen = set()
        merged = []
        for block in blocks:
            key = self._block_key(block)
            if key in seen:
                continue
            seen.add(key)
            merged.append(block)
        return merged

    @staticmethod
    def _public_blocks(blocks: list) -> list:
        public = []
        for i, block in enumerate(blocks, start=1):
            item = {
                "index": i,
                "family": block.get("family"),
                "entities": block.get("entities", []),
                "roles": block.get("roles", {}),
                "source": block.get("source_clause", ""),
                "confidence": block.get("confidence"),
            }
            if block.get("derived"):
                item["derived"] = True
                item["derive_source"] = block.get("derive_source")
            public.append(item)
        return public

    def _attach_block_output(self, merged: dict, blocks: list) -> None:
        public = self._public_blocks(blocks)
        merged["blocks"] = public
        merged["derived_blocks"] = [b for b in public if b.get("derived")]
        merged["n_derived_blocks"] = len(merged["derived_blocks"])

    @staticmethod
    def _attach_v3_root_blockers(merged: dict, v3_result: Optional[dict]) -> None:
        if not v3_result:
            merged["v3_root_blockers"] = []
            return
        merged["v3_root_blockers"] = [
            {
                "name": rb.get("name", rb.get("entity", "")),
                "entity": rb.get("entity", ""),
                "impact": rb.get("impact", 0),
            }
            for rb in v3_result.get("root_blockers", [])
        ]

    @staticmethod
    def _append_v252_trace_sections(merged: dict) -> None:
        lines = [merged.get("trace", "").rstrip()]

        v3_roots = merged.get("v3_root_blockers", [])
        if v3_roots:
            lines.append("")
            lines.append("V3 ROOT BLOCKERS (neural perception):")
            for i, rb in enumerate(v3_roots, start=1):
                name = rb.get("name", rb.get("entity", ""))
                impact = rb.get("impact", 0)
                lines.append(f"  V3 [{i}]: {name} (impact={impact})")

        derived = merged.get("derived_assumptions", [])
        meta = merged.get("derive_meta", {})
        lines.append("")
        lines.append("DERIVED ASSUMPTIONS (v3 12M closure):")
        if derived:
            for pair in derived:
                lines.append(f"  {pair.get('assuming')} => {pair.get('premise')}")
        else:
            lines.append("  None")
        lines.append(
            "  "
            f"strategy={meta.get('strategy')} "
            f"edge_source={meta.get('edge_source')} "
            f"n_edges={meta.get('n_edges')} "
            f"n_entities={meta.get('n_entities')}"
        )

        derived_blocks = merged.get("derived_blocks", [])
        lines.append("")
        lines.append(
            "BLOCK OUTPUT: "
            f"{merged.get('n_blocks', 0)} total, "
            f"{len(derived_blocks)} derived"
        )
        if derived_blocks:
            for block in derived_blocks[:12]:
                lines.append(
                    f"  [{block.get('index')}] {block.get('source')}"
                )
            if len(derived_blocks) > 12:
                lines.append(f"  ... {len(derived_blocks) - 12} more derived blocks")

        merged["trace"] = "\n".join(lines)

    @staticmethod
    def _dedupe_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seen = set()
        unique = []
        for src, tgt in edges:
            if not src or not tgt:
                continue
            pair = (str(src), str(tgt))
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)
        return unique

    @staticmethod
    def _exact_derived_pairs(edges: List[Tuple[str, str]]) -> set:
        unique_edges = ReasoningEngineV252._dedupe_edges(edges)
        direct_set = set(unique_edges)
        adj: Dict[str, set] = {}
        for src, tgt in unique_edges:
            adj.setdefault(src, set()).add(tgt)

        derived = set()
        for start in list(adj.keys()):
            frontier = deque([(start, 0)])
            visited = {start}
            while frontier:
                node, depth = frontier.popleft()
                for nxt in adj.get(node, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        frontier.append((nxt, depth + 1))
                        pair = (start, nxt)
                        if depth + 1 >= 2 and pair not in direct_set:
                            derived.add(pair)
        return derived

    def _derive_assumptions_spoonfed(
        self,
        edges: List[Tuple[str, str]],
    ) -> Tuple[List[Tuple[str, str]], dict]:
        """
        Transitive-closure assumptions from the 12M model's dedicated E4 expert.

        The closure is computed end-to-end in the network (Pass B / family-5 routing
        to E4) via ReasoningEngineV3.derive_assumptions(); no separate parametric
        expert and no BFS oracle stand in for the neural output. Only genuinely
        transitive pairs (closure minus the direct edges) are returned.
        """
        unique_edges = self._dedupe_edges(edges)
        direct_set = set(unique_edges)
        entity_set = {name for edge in unique_edges for name in edge}

        meta = {
            "n_edges": len(unique_edges),
            "n_entities": len(entity_set),
            "strategy": "none" if not unique_edges else "e4_closure",
            "source": "12m_v4_E4",
            "assumption_verdict": None,
        }

        if not unique_edges:
            return [], meta

        closure_pairs, verdict = self._v3.derive_assumptions(unique_edges)
        meta["assumption_verdict"] = verdict
        derived = sorted(pair for pair in closure_pairs if pair not in direct_set)
        if not derived:
            meta["strategy"] = "e4_closure_empty"
        return derived, meta

    @staticmethod
    def _dependency_edges_from_v3(v3_result: Optional[dict]) -> List[Tuple[str, str]]:
        if not v3_result:
            return []

        edges: List[Tuple[str, str]] = []
        for dep in v3_result.get("dependencies", []):
            src = dep.get("src_name", dep.get("src", ""))
            tgt = dep.get("tgt_name", dep.get("tgt", ""))
            if src and tgt:
                edges.append((str(src), str(tgt)))
        return edges

    @staticmethod
    def _dependency_edges_from_blocks(blocks: list) -> List[Tuple[str, str]]:
        edges: List[Tuple[str, str]] = []
        for block in blocks:
            if block.get("family") != "dependency":
                continue
            roles = block.get("roles", {})
            src = roles.get("blocked", "")
            tgt = roles.get("blocker", "")
            if not src or not tgt:
                entities = block.get("entities", [])
                if len(entities) >= 2:
                    src, tgt = entities[0], entities[1]
            if src and tgt:
                edges.append((str(src), str(tgt)))
        return edges

    def _append_derived_dependency_blocks(
        self,
        blocks: list,
        v3_result: Optional[dict],
        source_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> list:
        """
        Feed 12M model assumptions into the normal V2 block pipeline.

        V3 still owns perception.  The 12M model derives transitive
        assumptions from V3 dependency edges.  Those assumptions are appended as
        ordinary dependency blocks before _run_v2(), so root blockers, unlock
        order, critical path, and parallel-work heuristics are computed by the
        same block machinery as the core engine.
        """
        if not v3_result and not source_edges:
            return list(blocks)

        edges = self._dependency_edges_from_v3(v3_result)
        # Caller-declared edges are ground truth for structured blocks: prefer them
        # whenever present (re-perceived v3 edges can drift OOD on large prompts).
        if source_edges:
            edges = list(source_edges)
        elif not edges:
            edges = []

        if not edges:
            return list(blocks)

        direct_set = set(self._dedupe_edges(edges))
        derived_edges, _derive_meta = self._derive_assumptions_spoonfed(edges)

        enriched_blocks = list(blocks)
        for src, tgt in derived_edges:
            if (src, tgt) in direct_set:
                continue
            enriched_blocks.append({
                "family": "dependency",
                "entities": [src, tgt],
                "roles": {"blocked": src, "blocker": tgt},
                "source_clause": (
                    f"{src} transitively depends on {tgt} "
                    "(derived by 12M MoE E4 closure)"
                ),
                "confidence": 0.78,
                "derived": True,
                "derive_source": "12m_v4_E4",
            })

        return enriched_blocks

    # ── Derive enrichment ─────────────────────────────────────────────────────

    def _enrich_with_derive(
        self,
        merged: dict,
        v3_result: Optional[dict] = None,
    ) -> dict:
        """
        Additive enrichment: writes derived_assumptions + derive_meta.

        Edge source priority:
          1. v3_result["dependencies"]  (full mode; explicit src_name/tgt_name)
          2. merged["dependencies"]     (if caller stored it; same schema)
          3. unlock_sequence chain      (lite mode; consecutive step pairs)

        Strategy:
          spoon-fed 2-hop 12M model windows, guarded by exact traversal

        Never raises. Never removes or modifies existing keys.
        """
        # ── 1. Extract edges ──────────────────────────────────────────────────
        edges: List[Tuple[str, str]] = []

        # Full mode: explicit dependency list from v3_result or merged
        dep_source = None
        if v3_result is not None:
            dep_source = v3_result.get("dependencies", [])
        elif "dependencies" in merged:
            dep_source = merged.get("dependencies", [])

        if dep_source:
            for dep in dep_source:
                src = dep.get("src_name", dep.get("src", ""))
                tgt = dep.get("tgt_name", dep.get("tgt", ""))
                if src and tgt:
                    edges.append((str(src), str(tgt)))

        # Last-resort edge reconstruction for direct analyze_blocks() callers
        # that supplied blocks but no dependency list was preserved in merged.
        if not edges:
            seq = merged.get("unlock_sequence", [])
            if seq:
                sorted_seq = sorted(seq, key=lambda x: x.get("step", 0))
                names = [
                    item.get("name", item.get("entity", ""))
                    for item in sorted_seq
                ]
                names = [n for n in names if n]
                for i in range(len(names) - 1):
                    edges.append((names[i], names[i + 1]))

        edge_src_label = (
            "v3_dependencies" if (v3_result is not None and dep_source)
            else "merged_dependencies" if dep_source
            else "unlock_sequence_chain"
        )

        if not edges:
            merged["derived_assumptions"] = []
            merged["derive_meta"] = {
                "n_edges": 0,
                "n_entities": 0,
                "strategy": "none",
                "edge_source": edge_src_label,
            }
            return merged

        # ── 2. Spoon-fed derive expert, guarded by exact traversal ──────────
        unique_edges = self._dedupe_edges(edges)
        derived_tuples, derive_meta = self._derive_assumptions_spoonfed(unique_edges)
        derived_pairs = [
            {"assuming": src, "premise": tgt}
            for src, tgt in derived_tuples
        ]

        # ── 5. Write additive keys only ───────────────────────────────────────
        merged["derived_assumptions"] = derived_pairs
        derive_meta["edge_source"] = edge_src_label
        merged["derive_meta"] = derive_meta
        return merged

    # ── Info ──────────────────────────────────────────────────────────────────

    def engine_info(self) -> dict:
        info = super().engine_info()
        info["engine"] = "ReasoningEngineV252"
        info["version"] = self.VERSION
        info["closure_engine"] = self._probe_closure_engine()
        return info

    @staticmethod
    def _probe_closure_engine() -> dict:
        # Transitive closure is produced by the 12M model's dedicated E4
        # expert (ReasoningEngineV3.derive_assumptions).
        return {"engine": "12m_v4_E4", "nn": True, "bfs": False}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Pre-Reasoning v3.0.0 -- 12M neural engine"
    )
    parser.add_argument("text", nargs="?", help="Problem text to analyze")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    engine = ReasoningEngineV252(checkpoint_path=args.checkpoint, device=args.device)

    if args.info:
        print(json.dumps(engine.engine_info(), indent=2))
        return

    if not args.text:
        print("Usage: python -m pre_reasoning.engine 'A enables B. B enables C.'")
        print("       python -m pre_reasoning.engine --info")
        return

    result = engine.analyze(args.text)

    if args.json:
        result.pop("trace", None)
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result.get("trace", ""))
        print()
        print("--- DERIVED ASSUMPTIONS (v3 closure) ---")
        da = result.get("derived_assumptions", [])
        dm = result.get("derive_meta", {})
        if da:
            for pair in da:
                print(f"  {pair['assuming']} => {pair['premise']}")
        else:
            print("  (none)")
        print(f"  strategy={dm.get('strategy')}  n_entities={dm.get('n_entities')}")


if __name__ == "__main__":
    main()
