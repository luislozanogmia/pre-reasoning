# V4.1 Evaluation

V4 ships the 1M-parameter weights-only checkpoint `pre-reasoning-1m.safetensors`. Evaluation targets its product role: learned structural operations over an AI-authored structured form. It does not test arbitrary-language understanding because that belongs to the calling AI.

| Public-contract board | Result |
|---|---:|
| F1 dependencies and cycles | **600/600** |
| F2 conflicts | **600/600** |
| F3 requirements | **600/600** |
| F4 conditionals | **600/600** |
| F5 transitive closure | **600/600** |
| **Total** | **3,000/3,000** |
| Current runtime and integration tests | **32/32** |

The public-contract board uses fresh multi-token entity names on every case and exact assertions over the returned API fields. The adapter deliberately normalizes those names before inference, so this result measures the shipped product path—form validation, learned operator execution, placeholder restoration, composition, and rendering—not unrestricted lexical generalization by the checkpoint alone.

The public path strictly loads the checkpoint, executes each normalized operation through model generation, decodes the generated result, and restores submitted entity names. Ablations prove that reversing the model's generated dependency reverses the public relation, corrupting a requirement value fails closed, and disabling neural generation cannot fall back to adapter-computed answers.

Focus Mode tests verify the current-chat scheduler request, ten-minute cadence, durable five-block task prompt, and duplicate-prevention instruction. An injected monotonic clock verifies the fallback timer's exact boundary without sleeping, lazy checkpoint loading, fresh checkpoint-backed pulses, interval reset after success, and a latched due state after a short-form alarm.

## V3 comparison

The previous V3 model had 13.7M parameters and a fixed 32-entity vocabulary. V4 has 1M parameters, a **92.6% reduction**. On a 70-statement mixed-family form, V3 retained only 50 of 51 direct dependencies, dropped all 8 conflicts, 6 requirements, and 5 conditionals after its entity limit, missed the real seven-node cycle, and produced a false cycle. V4 preserved every submitted statement, returned the correct cycle and root blocker, and matched the structural baseline.

## Scope

The checkpoint learns dependency direction, transitive composition, cycle status, conflicts, numeric requirement operators, conditionals, schema, and termination. The adapter performs protocol work: form validation, reversible placeholder binding, graph window scheduling, name restoration, and trace rendering.

Raw prose, unrestricted instruction following, and direct copying of arbitrary natural-language entity strings are outside the checkpoint's product contract. The calling AI supplies the form.

Checkpoint SHA-256:

`a1bdb2877f7f4a4ccdf31cbad88c28bb81ce74a5d770f6cf25111cb66a31bd91`
