# Eval Results

13.7M trainable parameter MoE model, 35/35 PASS. Checkpoint: `pre-reasoning-12m-v3.safetensors`.

F1-F4 evaluated on 1,650 examples (50 per category, 33 categories). F5 evaluated on 600 examples (400 YES, 200 EMPTY). Seed 7777, greedy decoding.

## F1 -- Dependencies (9/9)

The core family. Detects who depends on whom, in what order, and whether the graph has cycles.

| Metric | Value | Target | Status | What it measures | Why this threshold |
|---|---:|---:|:---:|---|---|
| forward_dep_accuracy | 100.0 | >= 95 | PASS | "A depends on B" correctly emits A DEPENDS_ON B | Core capability, must be near-perfect |
| reverse_dep_accuracy | 100.0 | >= 95 | PASS | "A depends on B" does NOT emit B DEPENDS_ON A | Reversals would invert the entire trace |
| temporal_before_accuracy | 100.0 | >= 90 | PASS | "A before B" correctly maps to temporal ordering | Temporal language is more ambiguous than "depends on" |
| temporal_after_accuracy | 100.0 | >= 90 | PASS | "A after B" correctly maps to reverse temporal ordering | Same as above, reverse direction |
| root_blocker_accuracy | 100.0 | >= 90 | PASS | Identifies the node(s) with zero in-degree (nothing blocks them) | Graph-derived, depends on dependency edges being correct |
| cycle_detection_accuracy | 100.0 | >= 90 | PASS | Detects A->B->A circular dependencies | Cycles are structurally important but rare in inputs |
| false_positive_rate | 0.0 | <= 5 | PASS | % of independent entities falsely linked as dependent | False edges corrupt the entire downstream trace |
| chain_accuracy | 100.0 | >= 90 | PASS | Multi-hop chains (A->B->C->D) correctly ordered end-to-end | Tests compositionality, harder than pairwise |
| cross_modal_consistency | 98.0 | >= 85 | PASS | Same dependency expressed different ways produces same structure | Natural language varies; the model must be robust to phrasing |

## F2 -- Conflicts (4/4)

Detects when two entities are in opposition. Separate from dependencies: "A conflicts with B" is symmetric, not directional.

| Metric | Value | Target | Status | What it measures | Why this threshold |
|---|---:|---:|:---:|---|---|
| conflict_detection_accuracy | 99.5 | >= 90 | PASS | "CTO conflicts with dev" correctly emits a CONFLICTS edge | Must catch real disagreements |
| conflict_pair_precision | 99.1 | >= 85 | PASS | Both entities in the conflict are correctly identified | Lower bar because entity extraction from conflict phrases is harder than from "A depends on B" |
| false_conflict_rate | 0.0 | <= 5 | PASS | % of non-conflicting pairs falsely flagged | False conflicts waste attention on non-issues |
| mixed_separation_accuracy | 100.0 | >= 85 | PASS | When input has both dependencies AND conflicts, they are correctly separated into different families | Cross-family confusion is the main failure mode for mixed inputs |

## F3 -- Requirements (5/5)

Detects numeric thresholds and constraints: "coverage must be >= 80%", "latency must be <= 200ms".

| Metric | Value | Target | Status | What it measures | Why this threshold |
|---|---:|---:|:---:|---|---|
| has_req_verdict_accuracy | 100.0 | >= 90 | PASS | Correctly identifies that a requirement exists in the input | Binary detection, should be reliable |
| operator_accuracy | 90.7 | >= 85 | PASS | >= vs <= vs == operator correctly extracted | Operators are expressed many ways in natural language ("at least", "no more than", "exactly") |
| geq_recall | 92.3 | >= 85 | PASS | "at least X" / ">= X" requirements correctly caught | "At least" is the most common requirement phrasing |
| leq_recall | 89.6 | >= 85 | PASS | "at most X" / "<= X" requirements correctly caught | "At most" / "no more than" is rarer and more ambiguous |
| false_requirement_rate | 0.0 | <= 5 | PASS | % of non-requirement text falsely flagged as a requirement | Numbers appear often without being thresholds |

## F4 -- Conditionals (4/4)

Detects gated dependencies: "if A passes, then B can proceed". Different from plain dependencies because they have a condition.

| Metric | Value | Target | Status | What it measures | Why this threshold |
|---|---:|---:|:---:|---|---|
| conditional_detection_accuracy | 98.7 | >= 85 | PASS | "If X then Y" correctly identified as a conditional edge | Conditional language overlaps heavily with plain dependency language |
| conditional_edge_accuracy | 81.5 | >= 75 | PASS | The direction and binding of the conditional edge is correct | Hardest metric in the suite: "if" clauses have complex scope rules |
| conditional_entity_binding | 99.8 | >= 85 | PASS | The correct entities are bound to the condition vs. the consequence | Entity extraction is reliable even when edge direction is ambiguous |
| false_conditional_rate | 3.5 | <= 10 | PASS | % of non-conditional "if" statements falsely flagged | "If" appears often in non-conditional contexts ("if you want", "if possible") |

## F5 -- Transitive Closure (4/4)

The built-in E4 expert infers implicit assumptions from dependency chains. If A->B and B->C, then A transitively depends on C. This is the assumption the model must detect and surface.

| Metric | Value | Target | Status | What it measures | Why this threshold |
|---|---:|---:|:---:|---|---|
| transitive_recall | 90.1 | >= 90 | PASS | % of true transitive pairs the model finds | Must catch most implicit dependencies or the trace misses hidden pressure |
| transitive_precision | 90.2 | >= 90 | PASS | % of predicted transitive pairs that are actually transitive | False transitive claims add noise to the trace |
| assumption_detection | 100.0 | >= 92 | PASS | When transitive assumptions exist, the model correctly signals HAS_YES | Binary detection gate for downstream consumers |
| assum_empty_exact | 100.0 | >= 90 | PASS | When no transitive assumptions exist, the model correctly signals empty | Prevents hallucinated assumptions on simple graphs |

## Cross-Family Integrity (9/9)

These metrics test that families do not interfere with each other. When the model learns F4, does it forget F1? When F5 tokens appear, do they leak into F1-F3 outputs?

| Metric | Value | Target | Status | What it measures | Why this threshold |
|---|---:|---:|:---:|---|---|
| requirement_entity_detection | 100.0 | >= 85 | PASS | Entities in requirements correctly extracted alongside F1/F2 entities | Cross-family entity resolution |
| dep_retention | 94.0 | >= 85 | PASS | F1 dependency accuracy holds when F2-F5 families are also present in the input | Catastrophic forgetting check: adding new families must not break old ones |
| conflict_retention | 93.3 | >= 85 | PASS | F2 conflict accuracy holds in mixed-family inputs | Same retention check for conflicts |
| requirement_retention | 100.0 | >= 85 | PASS | F3 requirement accuracy holds in mixed-family inputs | Same retention check for requirements |
| cycle_detection_retention | 100.0 | >= 85 | PASS | Cycle detection still works when other families are present | Cycles are rare and fragile to interference |
| f5_token_intrusion_rate | 0.0 | <= 5 | PASS | % of F1-F3 outputs that incorrectly contain F5 tokens | F5 tokens (HAS_YES, HAS_NO) must never leak into non-closure outputs |
| entity_binding_accuracy | 99.7 | >= 90 | PASS | Across all families, entities are bound to the correct roles | Global entity resolution quality |
| structural_format_validity | 100.0 | >= 95 | PASS | Output token sequences are syntactically valid (parseable) | Invalid structure means the output is useless regardless of content |
| normalized_edge_f1 | 99.9 | >= 90 | PASS | F1 score of all predicted edges vs. ground truth across all families | Single aggregate quality number for the full output |

## Methodology

- All metrics are computed on held-out eval sets not seen during training
- F1-F4 use eval_unified with 50 examples per category, 33 categories
- F5 uses a dedicated BFS-oracle board: ground truth transitive closure computed by exact graph traversal, not labels
- Thresholds were set before training based on what a production grounding engine needs to be reliable, not tuned to the model's performance
- Seed 7777 is fixed for reproducibility; results are deterministic under greedy decoding
