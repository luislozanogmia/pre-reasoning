# Why Pre-Reasoning Traces Work: Literature Review

Pre-reasoning traces are an inference-time scaffold: the calling AI extracts
the relevant structure, an external component transforms that structure, and
the AI receives the result before producing its answer. The closest literature
does not describe this exact product, but several research lines explain why
the pattern is plausible.

## 1. Structured intermediate state reduces the burden on generation

[Fast Thinking with Structured Prompts](https://aclanthology.org/2025.ranlp-1.87/)
(Morozov, Chubarova, and Piontkovskaya, RANLP 2025) uses a graph-based
intermediate representation and reports gains on 0.5B and 7B instruction-tuned
models, including under a 25-token answer budget. Its result is directly
relevant to Pre-Reasoning: a model can use an externally supplied structure to
answer without regenerating a long chain of thought.

[Structure-Augmented Reasoning Generation](https://arxiv.org/abs/2506.08364)
(SARG, 2025) extracts relational triples from retrieved documents, builds a
knowledge graph, traverses multi-hop paths, and injects the paths and source
chunks into the generation prompt. It reports higher factual accuracy and
reasoning coherence than flat-context RAG. This supports the design principle
that relationships should be made explicit before the final generation pass.

## 2. Graph structure supports composition and selective computation

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al., NeurIPS
2023) expands reasoning from one sequence into multiple candidate states that
can be evaluated and searched.

[Graph of Thoughts](https://arxiv.org/abs/2308.09687) (Besta et al., AAAI
2024) generalizes this to arbitrary graph operations, allowing information to
flow between non-adjacent reasoning states and enabling aggregation and
refinement.

[Adaptive Graph of Thoughts](https://arxiv.org/abs/2502.05078) (2025) makes
the graph adaptive: it recursively decomposes a problem and expands only the
subproblems that need more computation. The relevant mechanism is not the
specific graph algorithm; it is the explicit representation of dependencies
and the ability to revisit a structured state.

## 3. A second pass can correct forward-only generation errors

[ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023) interleaves
reasoning and actions. External observations return to the model and change
the next reasoning step, rather than forcing the model to rely only on the
state it generated earlier.

[Self-Reflective Generation at Test Time](https://aclanthology.org/2026.acl-long.465/)
(ACL 2026) studies a related loop in which the model detects likely early
errors, performs a bounded internal update, and continues generation. It
reports improvements over direct chain-of-thought and Self-Refine across math,
general reasoning, and code tasks. The paper supports the general claim that
intervening before final emission can reduce cascading errors.

[Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska,
and Khattab, 2025) treats long prompts as an external environment and lets the
model inspect, decompose, and recursively call itself over selected fragments.
This is the closest recent work to the “re-enter the problem with preserved
context” idea: useful state is externalized, selected, and supplied to later
calls instead of being carried only through one uninterrupted token stream.

## 4. Tools and retrieved context make the trace operational

[Think-on-Graph 2.0](https://arxiv.org/abs/2407.10805) (Ma et al.,
ICLR 2025) alternates between graph retrieval and document-context retrieval.
It reports improved knowledge-intensive reasoning and compatibility with
multiple models without fine-tuning. This supports using a trace as a bridge
between retrieval/tool output and the next language-model pass.
