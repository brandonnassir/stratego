# Phase 11B — Agent 3: C1-Feature CNN

## Mission

Determine whether the final C1 move model already preserves enough belief-relevant information and whether the old small belief classifier was mainly an extraction bottleneck.

Use the same Phase 11B training and development data. Do not regenerate it.

Do not use the spent Phase 11 test bank.

# Frozen C1 Feature Seam

Inspect C1 and deliberately select the richest **spatial/token-level representation immediately before the task heads** that can be mapped back to board cells.

Prefer a per-square/per-token representation over an unnecessarily pooled or compressed global vector.

C1 remains completely frozen.

Record exactly which tensor/seam is used.

Cache the frozen C1 features for the Phase 11B training and development positions if this materially speeds training.

Any cache must be derivable from the common public observations plus the accepted frozen C1.

# Model

Build one dedicated belief CNN operating on the frozen C1 spatial features.

Target approximately the **3–5M parameter region**, although using less capacity is acceptable if the natural architecture is smaller.

Conceptually:

```text
public observation
      ->
frozen C1
      ->
per-square C1 representation
      ->
residual belief CNN
      ->
12 rank logits per unresolved square
```

Do not feed raw observation into the specialist. That is Agent 4's experiment.

Choose one reasonable architecture. No sweep.

# Training

Train only the new belief network.

No gradient may update C1.

Use the exact common Phase 11B training corpus and the same supervised hidden-rank objective.

# Time Budget

Maximum:

```text
approximately 1–2 hours total
```

Feature caching, training, evaluation, and report all count toward the budget.

Stop a clearly poor candidate after approximately 20–30 minutes.

Preserve a clearly strong checkpoint early.

# Evaluation

Evaluate on the identical Phase 11B development positions.

Report:

```text
overall CE
remaining-count baseline CE
R_CE
top-1

Phase9-like R_CE
Strategic R_CE
Tactical R_CE
Scout-rush R_CE

parameter count
training wall-clock
time-to-best checkpoint
inference latency
peak memory
```

Also report:

```text
Agent 2 raw-CNN R_CE
Agent 3 C1-feature-CNN R_CE
difference
```

Use earlier reports on disk. Do not rerun prior candidates.

# Interpretation

If:

```text
Agent 3 ~= Agent 2
```

then C1 probably retained substantial belief-relevant information and the original tiny head/extraction was likely the main bottleneck.

If:

```text
Agent 2 substantially > Agent 3
```

then C1 is probably discarding or obscuring belief-specific information and a dedicated raw-observation encoder is preferable.

# Required Interface

Expose:

```text
predict_marginals(public_state)
sample_worlds(public_state, n, seed)
```

# Required Artifacts

Save:

```text
best C1-feature-CNN checkpoint
frozen feature-seam description
exact architecture/config
parameter count
feature-cache metadata if used
learning curve

reports/phase11b/agent_03_summary.json
reports/phase11b/agent_03_report.md
```

Do not change C1 or any upstream evidence.

# Stop Condition

Stop after reporting this candidate.

Do not begin Agent 4 automatically.

Do not claim that Phase 11 has been repaired or that Phase 12 is authorized.
