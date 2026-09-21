# Loss Function Optimization

V4's central training change was separating the cooking recipe from the capability objective. The recipe controls which structured problems the model sees: operators, entity bindings, composition depth, families, and difficulty. The loss function controls which abilities must remain true on every applicable example.

## Before: one token-average objective

```text
token_loss = CrossEntropy(logits, targets)
L_before   = mean(token_loss over supervised tokens)
```

This is a valid language-modeling loss, but frequent/easy tokens can dominate it. A model may lower the average while silently failing copying, composition, cycle status, or termination.

Sequential training exposed a failure in both directions. Copy-first training encouraged the easiest copying shortcut and weakened structural learning. Reasoning-first training produced an efficient circuit that later training reused instead of preserving exact bindings. The objective therefore had to keep both capabilities active on the same examples.

## After: capability-aware grouped objective

```text
token_loss = model(x, targets, loss_reduction="none")
cell_loss  = normalize_each_applicable_span(token_loss, capability_masks)
L_after    = grouped_smooth_max(cell_loss, task_ids, example_ids, span_ids)
```

The grouped smooth-max keeps pressure on the weakest protected capability instead of allowing a strong cell to hide a failed one. The objective is still next-token cross-entropy underneath; the change is how losses are normalized, grouped, and kept simultaneously active.

Each training row is evaluated through capability cells rather than one undifferentiated token average. The cells cover operator execution, exact entity binding, composition, schema validity, family-specific status, and correct termination. Token cross-entropy remains the base signal, but each protected span is normalized first. A token-level smooth-max prevents one wrong character inside a long copied name from disappearing inside the mean loss. The normalized capability cells are then combined with a grouped smooth-max, so the easiest task cannot hide a failed requirement.

The objective is cumulative: new families are added while earlier dependency, binding, schema, and EOS cells stay active through replay. Paired renaming tests whether the model learned placeholders and operators instead of memorizing names. Promotion is behavioral, using raw greedy generation on frozen and fresh boards; aggregate loss alone is insufficient.

This produced the 1M-parameter checkpoint shipped with V4. At runtime the calling AI supplies the structured form; the checkpoint supplies learned structural operations over that form.
