# Phase 11B — Agent 4: Hybrid Raw+C1 CNN

## Mission

Test whether raw public information and C1's learned strategic representation provide complementary belief signal.

Reuse Agent 1's common Phase 11B corpus and Agent 3's exact frozen C1 feature seam.

Do not change either.

Do not use the spent Phase 11 test bank.

# Inputs

Use two legal/public branches:

```text
raw 127 x 10 x 10 public observation

frozen per-square C1 features
```

No hidden truth may enter either branch.

C1 remains frozen.

# Model

Build one fusion CNN around approximately **3–5M belief parameters**.

A sensible pattern is:

```text
raw observation -> small spatial projection ----\
                                                  -> concatenate/fuse
C1 feature map -> small projection -------------/
                                                  ->
                                           residual CNN
                                                  ->
                                           12 logits/square
```

Use one reasonable configuration.

Do not perform branch-width, fusion-method, depth, or learning-rate sweeps.

# Engineering Question

This model asks whether C1 provides useful high-level strategic abstractions while raw observation restores belief-specific details C1 may have compressed.

Compare directly with:

```text
Agent 2: raw-only CNN
Agent 3: C1-only CNN
```

# Training

Train only the hybrid belief network on the common Phase 11B supervised corpus.

Reuse Agent 3's C1 feature cache if compatible and exact.

# Time Budget

Maximum:

```text
approximately 1–2 hours total
```

Run a brief throughput/sanity pilot first.

Stop after approximately 20–30 minutes if clearly poor.

Preserve the best checkpoint early if strongly competitive.

# Evaluation

Use the identical common development positions.

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

Include a no-rerun comparison table:

```text
old Phase 11 head
Agent 1 best attached head
Agent 2 raw CNN
Agent 3 C1-feature CNN
Agent 4 hybrid
```

# Required Interface

Expose:

```text
predict_marginals(public_state)
sample_worlds(public_state, n, seed)
```

# Required Artifacts

Save:

```text
best hybrid checkpoint
exact architecture/config
parameter count
learning curve

reports/phase11b/agent_04_summary.json
reports/phase11b/agent_04_report.md
```

# Stop Condition

Stop after this candidate's engineering run.

Do not start the Transformer unless separately authorized.

Do not claim that Phase 11 has been repaired or that Phase 12 is authorized.
