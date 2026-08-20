# Phase 11B — Agent 2: Raw-Observation CNN

## Mission

Test whether a dedicated spatial belief specialist that receives the complete legal public observation directly can substantially outperform the attached C1 belief head.

Start from the Phase 11B state containing Agent 1's common training and development corpus.

Do not regenerate or alter those datasets.

Do not use the spent Phase 11 test bank.

# Model

Build **one** dedicated CNN/residual belief model in approximately the **3–5 million parameter** range.

Input:

```text
public 127 x 10 x 10 observation
```

The 127-channel public observation already contains substantial board, movement, history, inventory, behavior, and game-state information.

Do not add privileged information.

A sensible architecture family is:

```text
127 channels
    ->
spatial projection
    ->
residual 3x3 convolution blocks
    ->
per-square representation
    ->
12 rank logits per square
```

Width approximately 160–192 with enough residual blocks to land around 3–5M parameters is reasonable.

Calculate the actual parameter count before the full run.

Choose one sensible configuration. No architecture sweep.

Loss is supervised cross-entropy on unresolved opponent-piece target squares only.

# Engineering Question

This model bypasses C1's learned compression entirely.

The experiment asks:

> Does giving belief inference its own raw-observation spatial specialist solve most of the predictive weakness?

If this model substantially outperforms Agent 1's attached head, that is evidence that belief-specific information is easier to extract directly from the public observation than from C1's final representation.

# Training

Use the exact Agent 1 common training corpus.

Use MPS if the implementation is stable and materially faster.

Run a tiny sanity/throughput pilot first.

Then train only this one architecture.

Save the best development checkpoint whenever it improves materially.

# Time Budget

Maximum:

```text
approximately 1–2 hours total
```

This includes pilot, training, evaluation, and reporting.

Do not automatically consume the full budget.

If after approximately 20–30 minutes the model is clearly uncompetitive and the curve is poor, stop it.

If it is already very strong after 45–60 minutes and the curve has flattened, preserve it and stop.

# Evaluation

Use the exact common Phase 11B development positions.

Report:

```text
CE
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

Include Agent 1's best candidate and the old Phase 11 head in a compact comparison table. Do not rerun them.

# Required Interface

Expose:

```text
predict_marginals(public_state)
sample_worlds(public_state, n, seed)
```

The world-sampling adapter may consume the CNN marginals using the accepted constrained-world machinery through a Phase 11B adapter/import. Do not modify accepted Phase 11 code.

# Required Artifacts

Save:

```text
best raw-CNN checkpoint
exact architecture/config
parameter count
learning curve

reports/phase11b/agent_02_summary.json
reports/phase11b/agent_02_report.md
```

Do not modify Agent 1 artifacts.

# Stop Condition

Stop after this architecture's one engineering run.

Report whether it looks preferable to Agent 1.

Do not begin Agent 3's experiment yourself.

Do not claim that Phase 11 has been repaired or that Phase 12 is authorized.
