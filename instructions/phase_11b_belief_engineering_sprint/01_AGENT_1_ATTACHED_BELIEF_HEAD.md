# Phase 11B — Agent 1: Attached Belief-Head Engineering

## Mission

Determine whether the existing C1 representation already contains enough information for substantially stronger belief prediction and whether the Phase 11 weakness was mainly insufficient dedicated belief optimization or an undersized belief output head.

This is an engineering prototype. Do not alter or reinterpret Phase 11.

## Starting State

Start from the accepted Phase 11 closure state and a **copy** of the accepted Phase 9 checkpoint.

The production Phase 9 checkpoint is read-only.

Do not modify the accepted Phase 9 checkpoint, Phase 10 artifacts, or any Phase 11 contract, report, bank, ledger, sampler, system artifact, validation artifact, or sealed-test evidence.

All new work goes under Phase 11B-specific paths.

# Part 0 — Build the Common Phase 11B Corpus

Create the reusable Phase 11B engineering dataset that Agents 2–5 will also use.

## Training Corpus

```text
2,048 fresh games

512 Phase9-like
512 Strategic
512 Tactical
512 Scout-rush
```

Where practical:

- balance observer color;
- use approximately 50/50 P10-D versus neutral setup source;
- sample at most 16 evenly spaced eligible observer decisions per game.

An eligible observer decision is one where the observer is to act and at least one opponent piece remains unresolved.

## Development Set

```text
512 fresh games

128 Phase9-like
128 Strategic
128 Tactical
128 Scout-rush

up to 4 evenly spaced eligible observer decisions per game
```

The development set must not overlap the training corpus.

Do **not** use `phase11_test_bank_v1`.

## Data Boundary

Store public inputs separately from privileged labels. True hidden ranks are supervised targets only and must never enter the model-input path.

Preserve the common corpus and a compact metadata manifest so later agents can reuse it byte-for-byte.

# Experiment 1A — Existing Belief Output Layer

Load a copy of the accepted Phase 9 C1 model.

Freeze every shared C1 parameter.

Train only the existing belief output layer:

```text
frozen C1 per-piece feature
        ->
existing 128 -> 12 mapping
        ->
hidden-rank probabilities
```

Use supervised hidden-rank cross-entropy.

Do not train policy or value.

Record the learning curve and save the best development checkpoint.

If 1A clearly plateaus after a short run, stop 1A and move immediately to 1B.

# Experiment 1B — Larger Attached MLP

Keep C1 completely frozen.

Replace only the tiny belief mapping with one modest nonlinear MLP in approximately this family:

```text
C1 per-piece feature, width 128
        ->
hidden 256–512
        ->
hidden 256–512
        ->
12 rank logits
```

Choose one sensible architecture. Do not run a hyperparameter sweep.

Train it on exactly the same common corpus.

# Optional Experiment 1C — Final C1 Block + Larger Head

Run 1C only if 1B is clearly promising and sufficient time remains.

In a separate model copy:

- unfreeze only the final C1 block;
- train the larger belief head with it;
- use a smaller learning rate for the unfrozen C1 block.

Do not alter the accepted Phase 9 checkpoint.

# Time Budget

```text
30–90 minutes normally
approximately 2 hours absolute planning ceiling
```

Suggested behavior:

- very short pipeline/throughput sanity pilot first;
- 1A roughly 20–30 minutes if needed;
- give 1B most of the remaining budget;
- run 1C only if 1B justifies it.

Do not automatically consume two hours.

Stop a clearly poor branch after approximately 20–30 minutes.

Save any clearly strong checkpoint immediately.

# Evaluation

Evaluate 1A and 1B, and 1C if run, on the common development positions.

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

parameters added
total belief-related parameters
training wall-clock
time-to-best checkpoint
inference latency
peak memory
```

Also report the unchanged old Phase 11 head as a reference only. Do not reuse the spent Phase 11 test bank.

# Required Interface

The best Agent 1 candidate must expose:

```text
predict_marginals(public_state)
sample_worlds(public_state, n, seed)
```

Use existing accepted constrained-world logic through a Phase 11B adapter/import if useful. Do not modify accepted Phase 11 code.

# Required Artifacts

Preserve:

```text
common Phase11B training corpus
common Phase11B development set
common data manifest

best 1A checkpoint
best 1B checkpoint
best 1C checkpoint if run

training config(s)
concise learning curves

reports/phase11b/agent_01_summary.json
reports/phase11b/agent_01_report.md
```

The summary JSON must contain the standardized leaderboard fields so later agents can compare directly.

# Stop Condition

Stop after reporting which of 1A/1B/1C was best.

Do not begin another architecture.

Do not claim that Phase 11 has been repaired, Phase 12 is authorized, or the old Phase 11 `FAIL` has been overturned.
