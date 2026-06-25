"""
harness.py — Nested Decision Engine with Algorithmic Scaling
=================================================================

Upgrades over v1.5 tree builder:
  1. IMPACT SCORING — each node scored by transitive downstream dependencies
  2. RESOLUTION SIMULATION — step-by-step: resolve root blocker -> unblock chain -> next
  3. CROSS-TYPE LINKING — conflicts that block deps, delegations that enable resolution
  4. CRITICAL PATH — longest chain = minimum steps to goal
  5. PARALLEL OPPORTUNITIES — what can proceed at each resolution step
  6. ALGORITHMIC SCALING — O(N^2) for N blocks, handles arbitrary complexity

The engine is pure algorithm (0 ML params). Takes L1 adapter blocks, outputs:
  - Nested decision tree with impact scores
  - Optimal resolution sequence
  - Parallel opportunity windows
  - Critical path length

Pipeline: Neural perception (12M MoE) -> structural blocks -> graph reasoning (0p) -> structural trace

Author: Dr. Shannon, Mia Labs
Date: 2026-03-02
"""

from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, deque
import re


# ── Node ─────────────────────────────────────────────────────────────────────

class DecisionNode:
    """A node in the v2 nested decision tree."""

    def __init__(self, block: Dict, index: int):
        self.block = block
        self.index = index
        self.family = block["family"]
        self.entities = set(
            e.strip() for e in block.get("entities", [])
            if e and e.strip() and e.strip().upper() != "NONE"
        )
        self.roles = block.get("roles", {})
        self.confidence = block.get("confidence", 0.5)
        self.source = block.get("source_clause", "")

        # Tree structure
        self.children: List['DecisionNode'] = []
        self.parent: Optional['DecisionNode'] = None

        # v2 scoring
        self.label = "UNKNOWN"
        self.impact_score = 0       # how many nodes depend on this (transitive)
        self.chain_depth = 0        # depth in its chain (0 = leaf)
        self.critical_path = False  # on the longest chain?
        self.resolution_step = -1   # which step resolves this (-1 = not yet)

        # Cross-type links
        self.blocks_chain: Optional['DecisionNode'] = None   # conflict that blocks a dep
        self.enables_resolution: Optional['DecisionNode'] = None  # delegation that resolves a blocker
        self.mediated_by: Optional['DecisionNode'] = None    # conflict mediated by entity in chain

        # Selective dependency metadata
        self.selective_dep = block.get("selective", None)

    @property
    def depth(self) -> int:
        d = 0
        n = self
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    @property
    def is_root_blocker(self) -> bool:
        return self.label == "ROOT BLOCKER"

    @property
    def is_critical(self) -> bool:
        return self.label in ("ROOT BLOCKER", "CRITICAL CHAIN", "DEPENDENT-CRITICAL")

    def __repr__(self):
        ents = sorted(self.entities, key=str.lower)[:3]
        return f"[{self.family}] {', '.join(ents)} ({self.label}, impact={self.impact_score})"


# ── Entity Matching ──────────────────────────────────────────────────────────

# Common stopwords that must NOT, on their own, link two entities. Without this
# filter, plain-English entities like "the missing skill" and "the cheapest test"
# match on the shared word "the", collapsing distinct nodes onto whichever block is
# processed first. That builds a star instead of a chain and flattens every impact
# score to 1. Stripping these before word-overlap matching lets only *content* words
# form edges, so genuine dependency chains and their transitive impact survive.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "be", "this", "that", "these", "those", "with", "by", "from", "at", "as",
    "it", "its", "their", "our", "your", "his", "her", "we", "depends",
})


def entity_match(a: str, b: str) -> bool:
    """
    Check if two entity strings refer to the same entity.
    Uses word-boundary matching to avoid false positives
    (e.g., "Entity_1" should NOT match "Entity_10").

    Rules:
    1. Exact match (case-insensitive)
    2. Space-separated word overlap ("lead attorney" ~ "attorney")
    3. Substring with word boundary check ("committee" in "curriculum committee")
    """
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    # Exact match
    if a == b:
        return True
    # Word overlap on SPACE-split only (not underscore), ignoring stopwords so a
    # shared article/preposition ("the", "a", "of") can't link unrelated entities.
    # "lead attorney" ~ "attorney" but NOT "entity_1" ~ "entity_2", and NOT
    # "the missing skill" ~ "the cheapest test" on the bare word "the".
    wa = set(a.split()) - _STOPWORDS
    wb = set(b.split()) - _STOPWORDS
    if wa & wb:
        return True
    # Substring match ONLY if the shorter string is a complete word in the longer
    # This prevents "entity_1" matching "entity_10"
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer:
        idx = longer.find(shorter)
        end = idx + len(shorter)
        before_ok = (idx == 0 or not longer[idx-1].isalnum())
        after_ok = (end == len(longer) or not longer[end].isalnum())
        if before_ok and after_ok:
            return True
    return False


def entity_overlap(a: DecisionNode, b: DecisionNode) -> Set[str]:
    """Find shared entities between two nodes."""
    shared = set()
    for ea in a.entities:
        for eb in b.entities:
            if entity_match(ea, eb):
                shared.add(ea)
    return shared


# ── v2 Engine ────────────────────────────────────────────────────────────────

class ReasoningEngineV2:
    """
    Nested Decision Engine — algorithmic reasoning at scale.

    Takes flat L1 adapter blocks, produces:
      1. Dependency tree with impact scores
      2. Optimal resolution sequence
      3. Critical path identification
      4. Cross-type links (conflict-blocks-chain, delegation-enables-resolution)
      5. Parallel opportunity windows per resolution step
    """

    CRITICAL_FAMILIES = {"dependency", "prereq"}
    PARALLEL_FAMILIES = {"conflict", "delegate"}

    def __init__(self, l1_blocks: List[Dict]):
        self.blocks = l1_blocks
        self.nodes = [DecisionNode(blk, i) for i, blk in enumerate(l1_blocks)]
        self._build()

    def _build(self):
        """Full build pipeline."""
        self._build_dependency_edges()
        self._build_entity_overlap_edges()
        self._detect_cross_type_links()
        self._detect_cycles()
        self._label_nodes()
        self._compute_impact_scores()
        self._identify_critical_path()
        self._compute_resolution_sequence()

    # ── Step 1: Dependency edges from roles ───────────────────────────────

    def _build_dependency_edges(self):
        """Build parent-child edges via blocker/blocked role matching."""
        for i, node_a in enumerate(self.nodes):
            if node_a.family not in self.CRITICAL_FAMILIES:
                continue
            for j, node_b in enumerate(self.nodes):
                if i == j or node_b.family not in self.CRITICAL_FAMILIES:
                    continue

                a_blocker = (
                    node_a.roles.get("blocker", "") or
                    node_a.roles.get("requirement", "")
                ).lower().strip()
                b_blocked = (
                    node_b.roles.get("blocked", "") or
                    node_b.roles.get("entity", "")
                ).lower().strip()

                if not a_blocker or not b_blocked:
                    continue

                if entity_match(a_blocker, b_blocked):
                    if node_b.parent is None:
                        node_b.parent = node_a
                        node_a.children.append(node_b)

    # ── Step 2: Entity overlap edges (fallback) ──────────────────────────

    def _build_entity_overlap_edges(self):
        """Broader check for chains not caught by role matching.

        Links by shared (content-word) entities. A node that already has a parent
        may STILL acquire children here -- otherwise a node in the middle of a chain
        could never connect the nodes below it, fragmenting one long chain into
        disjoint pairs and flattening transitive impact. We only require the *child*
        (node_b) to be an orphan, and keep the j > i rule so every edge runs
        low->high index, which keeps the graph acyclic.
        """
        for i, node_a in enumerate(self.nodes):
            if node_a.family not in self.CRITICAL_FAMILIES:
                continue
            for j, node_b in enumerate(self.nodes):
                if i == j or node_b.family not in self.CRITICAL_FAMILIES:
                    continue
                if node_b in node_a.children:
                    continue
                overlap = entity_overlap(node_a, node_b)
                if overlap and node_b.parent is None and j > i:
                    node_b.parent = node_a
                    node_a.children.append(node_b)

    # ── Step 3: Cross-type links ─────────────────────────────────────────

    def _detect_cross_type_links(self):
        """
        Detect cross-type interactions:
        - Conflict that blocks a dependency chain (conflict entity = dep blocked entity)
        - Delegation that enables resolution (delegate task matches blocker condition)
        - Conflict mediator that appears in a dependency chain
        """
        dep_nodes = [n for n in self.nodes if n.family in self.CRITICAL_FAMILIES]
        conflict_nodes = [n for n in self.nodes if n.family == "conflict"]
        delegate_nodes = [n for n in self.nodes if n.family == "delegate"]

        # Conflict -> blocks chain
        for conflict in conflict_nodes:
            conflict_ents = conflict.entities
            for dep in dep_nodes:
                blocked = (dep.roles.get("blocked", "") or
                          dep.roles.get("entity", "")).lower().strip()
                if blocked:
                    for ce in conflict_ents:
                        if entity_match(ce, blocked):
                            conflict.blocks_chain = dep
                            break

        # Delegation -> enables resolution
        for delegate in delegate_nodes:
            # The delegated task might match a blocker's condition
            delegate_task = delegate.roles.get("task", "").lower().strip()
            if not delegate_task:
                # Try to extract task from source clause
                delegate_task = delegate.source.lower() if delegate.source else ""

            for dep in dep_nodes:
                blocker = (dep.roles.get("blocker", "") or
                          dep.roles.get("requirement", "")).lower().strip()
                condition = dep.roles.get("condition", "").lower().strip()

                if blocker and delegate_task:
                    # Check if delegation task relates to resolving the blocker
                    if entity_match(delegate_task, blocker) or entity_match(delegate_task, condition):
                        delegate.enables_resolution = dep
                        break

                # Also check if delegator/delegate entity appears in the dep chain
                for de in delegate.entities:
                    if entity_match(de, blocker):
                        delegate.enables_resolution = dep
                        break

        # Conflict mediator in chain
        for conflict in conflict_nodes:
            mediator = conflict.roles.get("mediator", "").lower().strip()
            if not mediator:
                continue
            for dep in dep_nodes:
                for de in dep.entities:
                    if entity_match(mediator, de):
                        conflict.mediated_by = dep
                        break

    # ── Step 3b: Cycle detection (Tarjan's SCC) ─────────────────────────

    def _detect_cycles(self):
        """
        Detect circular dependencies using Tarjan's SCC algorithm.
        Flags nodes in cycles of size >= 2 as CIRCULAR_DEPENDENCY.
        Stores cycles in self._cycles for format_for_narrator().
        """
        index_counter = [0]
        stack = []
        on_stack = set()
        indices = {}
        lowlinks = {}
        sccs = []

        # Build adjacency: parent -> children (dependency direction)
        # Also include cross-type links as edges
        adj = defaultdict(list)
        for node in self.nodes:
            for child in node.children:
                adj[node.index].append(child.index)
            if node.blocks_chain is not None:
                adj[node.index].append(node.blocks_chain.index)
            if node.enables_resolution is not None:
                adj[node.index].append(node.enables_resolution.index)

        def strongconnect(v):
            indices[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in adj.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) >= 2:
                    sccs.append(scc)

        for node in self.nodes:
            if node.index not in indices:
                strongconnect(node.index)

        self._cycles = sccs

        # Label nodes in cycles
        for cycle in sccs:
            for idx in cycle:
                self.nodes[idx].label = "CIRCULAR_DEPENDENCY"

    # ── Step 4: Label nodes ──────────────────────────────────────────────

    def _label_nodes(self):
        """Label all nodes: ROOT BLOCKER, CRITICAL CHAIN, DEPENDENT-CRITICAL, PARALLEL TRACK.
        Skips nodes already labeled CIRCULAR_DEPENDENCY by _detect_cycles()."""
        roots = [n for n in self.nodes if n.parent is None]

        for root in roots:
            if root.label == "CIRCULAR_DEPENDENCY":
                continue  # already labeled by cycle detection
            if root.family in self.CRITICAL_FAMILIES:
                self._label_chain(root, is_root_chain=True)
            else:
                root.label = "PARALLEL TRACK"
                # Check if this parallel node blocks a chain (cross-type)
                if root.blocks_chain is not None:
                    root.label = "BLOCKING PARALLEL"

    def _label_chain(self, node: DecisionNode, is_root_chain: bool):
        """Recursively label a chain. Skips CIRCULAR_DEPENDENCY nodes."""
        if node.label == "CIRCULAR_DEPENDENCY":
            return
        if not node.children:
            node.label = "ROOT BLOCKER" if is_root_chain else "DEPENDENT-CRITICAL"
        elif node.parent is None:
            node.label = "CRITICAL CHAIN"
        else:
            node.label = "DEPENDENT-CRITICAL"

        for child in node.children:
            self._label_chain(child, is_root_chain)

    # ── Step 5: Impact scores ────────────────────────────────────────────

    def _compute_impact_scores(self):
        """
        Compute transitive impact: how many nodes depend on this one?
        Higher impact = resolving this unblocks more work.
        Includes cross-type impacts (conflict blocking a chain).
        """
        for node in self.nodes:
            node.impact_score = self._count_downstream(node)

    def _count_downstream(self, node: DecisionNode, visited: Set[int] = None) -> int:
        """Count all nodes transitively upstream (depending on this node)."""
        if visited is None:
            visited = set()
        if node.index in visited:
            return 0
        visited.add(node.index)

        count = 0
        # Direct parent depends on this node
        if node.parent is not None and node.parent.index not in visited:
            count += 1 + self._count_downstream(node.parent, visited)

        # Cross-type: if this node enables resolution of a dep, that dep's chain benefits
        if node.enables_resolution is not None:
            target = node.enables_resolution
            if target.index not in visited:
                count += 1 + self._count_downstream(target, visited)

        return count

    # ── Step 6: Critical path ────────────────────────────────────────────

    def _identify_critical_path(self):
        """
        Find the longest chain (critical path).
        Critical path = minimum number of sequential steps to reach goal.
        """
        roots = [n for n in self.nodes if n.parent is None and n.is_critical]
        if not roots:
            return

        max_depth = 0
        deepest_leaf = None

        for root in roots:
            leaf, depth = self._find_deepest_leaf(root)
            if depth > max_depth:
                max_depth = depth
                deepest_leaf = leaf

        # Mark critical path from deepest leaf up to root
        if deepest_leaf:
            node = deepest_leaf
            while node is not None:
                node.critical_path = True
                node.chain_depth = node.depth
                node = node.parent

    def _find_deepest_leaf(self, node: DecisionNode, depth: int = 0) -> Tuple[DecisionNode, int]:
        """Find the deepest leaf in a subtree."""
        if not node.children:
            return node, depth
        best_leaf, best_depth = node, depth
        for child in node.children:
            leaf, d = self._find_deepest_leaf(child, depth + 1)
            if d > best_depth:
                best_leaf, best_depth = leaf, d
        return best_leaf, best_depth

    # ── Step 7: Resolution sequence ──────────────────────────────────────

    def _compute_resolution_sequence(self):
        """
        Compute optimal resolution order using topological sort.
        Resolve ROOT BLOCKERs first (sorted by impact score, highest first).
        Then DEPENDENT-CRITICAL nodes get unblocked automatically.
        Parallel tracks can proceed at any time.
        """
        # Gather root blockers sorted by impact (highest first)
        root_blockers = sorted(
            [n for n in self.nodes if n.label == "ROOT BLOCKER"],
            key=lambda n: (-n.impact_score, n.index)
        )

        step = 1
        resolved = set()

        # Phase 1: Resolve root blockers
        for rb in root_blockers:
            rb.resolution_step = step
            resolved.add(rb.index)
            step += 1

        # Phase 2: Propagate resolution up chains (BFS from resolved nodes)
        queue = deque(root_blockers)
        while queue:
            node = queue.popleft()
            if node.parent is not None and node.parent.index not in resolved:
                parent = node.parent
                # Check if ALL children of parent are resolved
                all_children_resolved = all(c.index in resolved for c in parent.children)
                if all_children_resolved:
                    parent.resolution_step = step
                    resolved.add(parent.index)
                    step += 1
                    queue.append(parent)

        # Phase 3: Parallel tracks get step 0 (can proceed anytime)
        for node in self.nodes:
            if node.label in ("PARALLEL TRACK", "BLOCKING PARALLEL"):
                node.resolution_step = 0

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def root_blockers(self) -> List[DecisionNode]:
        return sorted(
            [n for n in self.nodes if n.label == "ROOT BLOCKER"],
            key=lambda n: (-n.impact_score, n.index)
        )

    @property
    def critical_chain(self) -> List[DecisionNode]:
        return [n for n in self.nodes if n.is_critical]

    @property
    def critical_path_length(self) -> int:
        return max((n.chain_depth for n in self.nodes if n.critical_path), default=0)

    @property
    def parallel_tracks(self) -> List[DecisionNode]:
        return [n for n in self.nodes if n.label in ("PARALLEL TRACK", "BLOCKING PARALLEL")]

    @property
    def resolution_steps(self) -> List[Tuple[int, List[DecisionNode]]]:
        """Get resolution sequence grouped by step number."""
        by_step = defaultdict(list)
        for n in self.nodes:
            if n.resolution_step >= 0:
                by_step[n.resolution_step].append(n)
        return sorted(by_step.items())

    @property
    def circular_dependencies(self) -> List[List[DecisionNode]]:
        """Get all detected circular dependency cycles."""
        cycles = getattr(self, '_cycles', [])
        return [[self.nodes[idx] for idx in cycle] for cycle in cycles]

    @property
    def cross_type_links(self) -> List[Dict]:
        """Get all detected cross-type interactions."""
        links = []
        for n in self.nodes:
            if n.blocks_chain is not None:
                links.append({
                    "type": "conflict_blocks_chain",
                    "source": n,
                    "target": n.blocks_chain,
                })
            if n.enables_resolution is not None:
                links.append({
                    "type": "delegation_enables_resolution",
                    "source": n,
                    "target": n.enables_resolution,
                })
            if n.mediated_by is not None:
                links.append({
                    "type": "mediator_in_chain",
                    "source": n,
                    "target": n.mediated_by,
                })
        return links

    # ── Formatting ───────────────────────────────────────────────────────

    def format_tree(self) -> str:
        """Format the full decision tree as readable text."""
        roots = [n for n in self.nodes if n.parent is None]
        lines = []

        def _fmt(node: DecisionNode, prefix: str, is_last: bool):
            connector = "+-- "
            ents = sorted(node.entities, key=str.lower)[:3]
            ent_str = ", ".join(ents)
            cp = " *CP*" if node.critical_path else ""
            impact = f" impact={node.impact_score}" if node.impact_score > 0 else ""
            step = f" step={node.resolution_step}" if node.resolution_step > 0 else ""
            line = f"{prefix}{connector}[{node.family}] {ent_str} [{node.label}]{cp}{impact}{step}"

            src = node.block.get("source_clause", "")
            if src:
                line += f"\n{prefix}    '{src[:60]}'"

            lines.append(line)

            child_prefix = prefix + ("|   " if not is_last else "    ")
            for ci, child in enumerate(node.children):
                _fmt(child, child_prefix, ci == len(node.children) - 1)

        for ri, root in enumerate(roots):
            _fmt(root, "", ri == len(roots) - 1)

        return "\n".join(lines)

    def format_summary(self) -> str:
        """Format concise reasoning summary."""
        lines = []
        lines.append("REASONING ENGINE v2 SUMMARY:")
        lines.append(f"  Total blocks: {len(self.nodes)}")
        lines.append(f"  Critical path length: {self.critical_path_length} steps")
        lines.append(f"  Root blockers: {len(self.root_blockers)}")
        lines.append(f"  Parallel tracks: {len(self.parallel_tracks)}")
        cycles = self.circular_dependencies
        if cycles:
            lines.append(f"  Circular dependencies: {len(cycles)} cycle(s)")

        # Cross-type links
        ct = self.cross_type_links
        if ct:
            lines.append(f"  Cross-type links: {len(ct)}")
            for link in ct:
                src_ents = sorted(link['source'].entities, key=str.lower)[:2]
                tgt_ents = sorted(link['target'].entities, key=str.lower)[:2]
                lines.append(
                    f"    {link['type']}: "
                    f"{', '.join(src_ents)} -> "
                    f"{', '.join(tgt_ents)}"
                )

        # Root blockers with impact
        if self.root_blockers:
            lines.append("")
            lines.append("  ROOT BLOCKERS (resolve FIRST, sorted by impact):")
            for rb in self.root_blockers:
                ents = sorted(rb.entities, key=str.lower)[:3]
                lines.append(
                    f"    [{rb.index+1}] [{rb.family}] "
                    f"{', '.join(ents)} "
                    f"(impact={rb.impact_score})"
                )
                if rb.source:
                    lines.append(f"        '{rb.source[:70]}'")

        # Resolution sequence
        steps = self.resolution_steps
        if steps:
            lines.append("")
            lines.append("  OPTIMAL RESOLUTION SEQUENCE:")
            for step_num, step_nodes in steps:
                if step_num == 0:
                    continue  # parallel tracks
                node_strs = []
                for n in step_nodes:
                    ents = sorted(n.entities, key=str.lower)[:2]
                    node_strs.append(f"[{n.index+1}] {', '.join(ents)}")
                lines.append(f"    Step {step_num}: {' + '.join(node_strs)}")

        # Parallel opportunities
        parallel = self.parallel_tracks
        if parallel:
            lines.append("")
            lines.append("  PARALLEL (proceed independently at any time):")
            for p in parallel:
                ents = sorted(p.entities, key=str.lower)[:3]
                extra = ""
                if p.blocks_chain:
                    extra = " [!BLOCKS CHAIN]"
                if p.enables_resolution:
                    extra = " [ENABLES RESOLUTION]"
                lines.append(
                    f"    [{p.index+1}] [{p.family}] "
                    f"{', '.join(ents)}{extra}"
                )

        return "\n".join(lines)

    def format_for_narrator(self) -> str:
        """
        Format specifically for LLM narrator consumption.
        Structured, concise, with clear labels for grounding.
        """
        lines = []

        # Framing header — tells the model what the trace IS and how to use it
        lines.append("--- STRUCTURAL TRACE ---")
        lines.append("This is a map of the situation - not a perfect map, but a grounding map.")
        lines.append("It gives you the opportunity to see the whole picture before committing to tokens.")
        lines.append("Now, create the solution on your own using this map.")
        lines.append("")

        # Root blockers
        lines.append("ROOT BLOCKERS (must resolve FIRST):")
        for rb in self.root_blockers:
            ents = sorted(rb.entities, key=str.lower)[:3]
            src = rb.source[:80] if rb.source else ""
            lines.append(f"  Block [{rb.index+1}]: {', '.join(ents)}")
            lines.append(f"    Impact: unblocks {rb.impact_score} downstream steps")
            if src:
                lines.append(f"    Evidence: '{src}'")

        # Resolution sequence
        lines.append("")
        lines.append("UNLOCK SEQUENCE (optimal order):")
        for step_num, step_nodes in self.resolution_steps:
            if step_num == 0:
                continue
            for n in step_nodes:
                ents = sorted(n.entities, key=str.lower)[:2]
                action = "Resolve" if n.is_root_blocker else "Unblocked"
                lines.append(
                    f"  Step {step_num}: {action} [{n.index+1}] "
                    f"{', '.join(ents)}"
                )

        # Parallel
        lines.append("")
        lines.append("PARALLEL WORK (proceed now, independent):")
        for p in self.parallel_tracks:
            ents = sorted(p.entities, key=str.lower)[:3]
            lines.append(f"  Block [{p.index+1}]: [{p.family}] {', '.join(ents)}")

        # Cross-type warnings
        ct = self.cross_type_links
        if ct:
            lines.append("")
            lines.append("CROSS-TYPE INTERACTIONS:")
            for link in ct:
                src_ents = sorted(link['source'].entities, key=str.lower)[:2]
                tgt_ents = sorted(link['target'].entities, key=str.lower)[:2]
                if link['type'] == 'conflict_blocks_chain':
                    lines.append(
                        f"  WARNING: Conflict [{link['source'].index+1}] "
                        f"({', '.join(src_ents)}) "
                        f"may block chain [{link['target'].index+1}]"
                    )
                elif link['type'] == 'delegation_enables_resolution':
                    lines.append(
                        f"  OPPORTUNITY: Delegation [{link['source'].index+1}] "
                        f"({', '.join(src_ents)}) "
                        f"can help resolve [{link['target'].index+1}]"
                    )

        # Circular dependencies
        cycles = self.circular_dependencies
        if cycles:
            lines.append("")
            lines.append("CIRCULAR DEPENDENCIES (cannot resolve sequentially):")
            for ci, cycle in enumerate(cycles):
                ent_parts = []
                for node in cycle:
                    ents = sorted(node.entities, key=str.lower)[:2]
                    ent_parts.append(f"[{node.index+1}] {', '.join(ents)}")
                lines.append(f"  Cycle {ci+1}: {' <-> '.join(ent_parts)}")
                lines.append(f"    These depend on each other — use alternative resolution strategy.")

        # Selective dependencies
        selective_nodes = [n for n in self.nodes if n.selective_dep]
        if selective_nodes:
            lines.append("")
            lines.append("SELECTIVE DEPENDENCIES (not all members are active):")
            for n in selective_nodes:
                sd = n.selective_dep
                ents = sorted(n.entities, key=str.lower)[:3]
                lines.append(f"  Block [{n.index+1}]: {', '.join(ents)}")
                lines.append(f"    ONLY {sd.get('active_pct', '?')}% of {sd.get('subset_of', 'group')} are active")
                if sd.get("distinguisher"):
                    lines.append(f"    Distinguisher: {sd['distinguisher']}")
                if sd.get("description"):
                    lines.append(f"    Detail: {sd['description']}")

        return "\n".join(lines)
