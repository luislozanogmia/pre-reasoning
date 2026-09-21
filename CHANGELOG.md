# Changelog

## v4.0.1

- Adds scheduler-backed Focus Mode with a ten-minute default interval. It exposes an explicit model-facing request to create or update one recurring task in the current chat, including the durable task prompt and duplicate-prevention instruction.
- Keeps `focus.check()` as a local fallback for hosts without scheduling support; the package itself does not start a sleeping process, background thread, or OS crontab entry.
- Redefines `pulse(form_text)` as a fresh checkpoint-backed reasoning pass over the AI's current structured view of the problem.
- Keeps a failed short-form pulse due and returns the existing `REPROMPT_REQUIRED` alarm with the exact five-block form template.
- Renames the former word-presence behavior to `coverage_check()` and `coverage_check_result()`, and marks its result explicitly as lexical and not semantically verified.
- Preserves the former `pulse(form_text, response)` and `pulse_result(analysis, response)` signatures as compatibility aliases.

- Fixes the V4 package loader so the 1M-parameter checkpoint is reconstructed with its exact architecture and loaded with strict tensor validation.
- Routes the public `analyze()` and `pulse()` API through the loaded neural component instead of the accidental parser-only path.
- Removes copied training-framework, optimizer, dataset, and checkpoint-management modules from the distribution; only the inference architecture remains.
- Defines the product boundary explicitly: the calling AI interprets natural language and writes the structured form; the checkpoint executes learned structural operations; the adapter validates, binds, windows, restores names, and renders the trace for the AI's next forward pass.
- Adds a model-decoded adapter: it binds submitted entities to learned placeholders, the checkpoint generates each required operation, and the adapter restores the original names from the generated result.
- Adds neural ablations proving that reversed model output reverses the public relation and disabled generation cannot fall back to parser-only output.
- Revalidated the public execution path at 3,000/3,000 on the frozen compatibility board.
- Replaces raw-prompt examples and hooks with the form-first integration and removes platform-specific sidecars and legacy benchmark artifacts from the core repository.
- Adds a five-block minimum guard: shorter valid forms return a `REPROMPT_REQUIRED` alarm with the submitted form and an attached retry template before neural analysis runs.

## v4.0.0

- Public release of the V4 pre-reasoning engine, replacing the previous V3 release while preserving its structural contract.
- Ships a weights-only 1M checkpoint under a neutral filename; model lineage and training state are not part of the public version.
- Adds capability-aware grouped loss documentation and compact cycle-safe structural traces.
- Known packaging defect: the public API did not execute the bundled 1M checkpoint. Superseded by v4.0.1.

## v3.1.0

CPU inference is 4-6x faster. No API changes, no new dependencies, identical
outputs (verified byte-identical on a 12-prompt reference suite). All
optimizations are pure PyTorch and process-safe: nothing global is modified,
so other models running in the same client process are unaffected.

- KV cache in `generate()`: the prompt is encoded once, then each new token
  runs a single incremental step instead of re-processing the full sequence.
  Long-sequence fallback to the original loop is automatic (> max_seq_len).
- Functional fast path for incremental steps (same tensor ops without
  nn.Module dispatch overhead) under `torch.inference_mode()`.
- Scoped thread pinning: single-token steps run at 4 threads during
  `generate()` only; the caller's `torch.get_num_threads()` setting is
  saved and restored, never changed globally.
- Engine singleton: `get_engine()`/`analyze()` reuse one engine per
  (checkpoint, device) instead of reloading the checkpoint each call.
- Persistent E4 closure-window cache: structured 2-hop windows are
  deterministic (greedy decode, fixed weights), so results are memoized on
  the model instance across calls.
- Escape hatches (env vars, all default-off): `PRE_REASONING_DISABLE_KV=1`
  (original generate loop), `PRE_REASONING_NO_FASTSTEP=1` (module-based
  incremental steps), `PRE_REASONING_THREADS=N` (thread pin override, 0 =
  never touch threading), `PRE_REASONING_NO_ENGINE_CACHE=1` (rebuild engine
  per call), `PRE_REASONING_NO_F5_CACHE=1` (per-pass closure cache only).

## v3.0.0

- Upgraded the V3 engine to the 13.7M trainable-parameter MoE model with five expert groups.
- Transitive closure is now computed by the built-in E4 expert. The external derive_expert package is no longer needed.
- Renamed internal modules: engine.py (was pre_reasoning_v2_5_2.py), engine_core.py (was pre_reasoning_v2_5.py).
- Checkpoint upgraded from the earlier 3M model to the 13.7M model distributed as `pre-reasoning-12m-v3.safetensors`.
- Removed derive_expert sub-package from the distribution.
- Fixed broken imports caused by the module rename.

## v2.5.4

- Added derive-expert neural transitive-closure enrichment to the default engine.
- Bundled `derive_expert` and `thin_expert_d128L3.safetensors` with the package.
- CLI and package-level APIs now route through the enriched engine while preserving the legacy v2.5 engine alias.

## v2.5.3

- Documentation only: removed internal stage labels (V2 / V3) from the public docs (README, CHANGELOG, skill descriptor). The package is presented as a single v2.5 engine: neural perception plus graph reasoning. No code, API, or behavior changes.

## v2.5.2

- Fixed graph linking that flattened transitive impact scores to 1 on plain-English problems.
  - `entity_match` no longer links two entities on shared stopwords ("the", "a", "of", ...); only content words can form an edge. Previously, phrases like "the missing skill" and "the cheapest test" matched on "the", collapsing distinct nodes into a star and flattening impact.
  - `_build_entity_overlap_edges` now lets an already-parented node acquire children (only the child must be an orphan), so a node in the middle of a chain connects the nodes below it instead of fragmenting one long chain into disjoint pairs. The `j > i` rule is kept, so edges run low->high index and the graph stays acyclic.
- Net effect: a 4-node dependency chain now reports transitive impact 0/1/2/3 (was 1/1/1/1), and root blockers are identified correctly.

## v2.5.1

- torch and safetensors are now required dependencies -- the engine always runs in full mode (neural perception + graph reasoning).
- Removed `[neural]` optional extra and deterministic-only fallback.

## v2.5.0

- Neural perception engine: 3M-parameter model trained on reasoning graphs, bundled as weights-only safetensors.
- v2.5 engine: neural perception + graph reasoning.
- Package-level `analyze()` and `pulse()` API.
- CLI entry point: `pre-reasoning "your problem text"`.
- Pytest coverage for analysis, cycles, conflicts, and pulse checks.
- PyPI packaging with torch and safetensors bundled.
- Claude Code adoption docs and agent skill descriptor.
- MIT license.
