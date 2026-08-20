# Phase 11B — Belief Engineering Sprint

## Purpose

Phase 11B is a separate engineering branch inserted after the formally closed Phase 11 `FAIL` and before a newly defined Phase 12.

Its purpose is to rapidly identify a belief architecture that offers the greatest useful improvement in hidden-piece prediction for the lowest practical computational and wall-clock cost.

Phase 11B is not a retroactive override of Phase 11, not a scientific repair phase, and not authorization for Phase 12.

## Preservation Rules

Do not delete, overwrite, reinterpret, or modify any accepted evidence from earlier phases. Preserve at minimum:

- the accepted Phase 9 checkpoint;
- the existing Phase 9 belief head;
- all Phase 10 and earlier accepted artifacts;
- all Phase 11 Agent 1–7 reports;
- `phase11_system_v1`;
- the Phase 11 validation artifacts;
- the spent Phase 11 sealed test bank;
- the Phase 11 access ledger;
- the final Phase 11 `FAIL` classification;
- the Phase 11 sampler, information-safety, reproducibility, and runtime evidence.

Nothing in Phase 11B should prevent a later rigorous belief-repair study.

## Namespace

All new work should live under Phase 11B-specific paths, for example:

```text
checkpoints/phase11b/
data/phase11b/
reports/phase11b/
stratego/belief/phase11b/
tests/belief/phase11b/
```

Every Phase 11B checkpoint/report should record:

```text
phase = phase11b
status = engineering_prototype
phase11_fail_unchanged = true
phase11_test_bank_used = false
phase12_authorized_by_this_artifact = false
```

## Shared Engineering Question

> Given only information legally available to a Stratego player, how accurately can this architecture infer the true ranks of the opponent's hidden pieces?

True hidden ranks may be supervised training labels. They must never be model inputs.

This sprint is supervised belief prediction, not reinforcement learning from game outcomes.

# Common Phase 11B Dataset

Agent 1 creates the common corpus once. Agents 2–5 reuse it unchanged.

Do not use the spent `phase11_test_bank_v1`.

## Training Corpus

```text
2,048 fresh games total

512 Phase9-like
512 Strategic
512 Tactical
512 Scout-rush
```

Where practical:

- balance observer color;
- use approximately 50/50 P10-D versus `neutral_v1` setup source;
- sample at most 16 evenly spaced eligible observer decisions per game.

An eligible observer decision is one where the observer is to act and at least one opponent piece remains unresolved.

## Development Set

```text
512 fresh games total

128 Phase9-like
128 Strategic
128 Tactical
128 Scout-rush
```

Sample at most 4 evenly spaced eligible observer decisions per game.

The development set is an engineering comparison set, not a scientifically sealed bank.

## Canonical Sample Contents

Store at minimum:

```text
public 127 x 10 x 10 observation
public-state id
observer color
behavior stratum
setup source
decision index
hidden target-square mask
true rank labels
remaining public inventory
public legal-rank masks
```

Public inputs and privileged labels must be stored separately.

Do not bake C1 internal features into the canonical dataset. Agents 3 and 4 may derive/cache them from the frozen C1 encoder.

# Shared Metrics

Every candidate must report at minimum:

```text
overall cross-entropy
remaining-count baseline cross-entropy
R_CE = CE_candidate / CE_remaining_count_baseline
top-1 hidden-rank accuracy

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

The old Phase 11 head produced approximately `R_CE ~= 0.975`. This is a reference point, not a hard gate.

The old Phase 11 `0.970` threshold is not sacred in Phase 11B.

# Shared Belief Interface

Every candidate should expose:

```text
predict_marginals(public_state)
    -> 12-way rank probabilities for unresolved opponent pieces

sample_worlds(public_state, n, seed)
    -> complete legal hidden armies
```

Candidates 1–4 may use predicted marginals with the already accepted constrained-world machinery through a Phase 11B adapter/import. Do not modify the accepted Phase 11 sampler.

Candidate 5 may additionally expose its native constrained autoregressive sampler.

# Runtime Policy

No individual pseudo-experiment should be designed around more than approximately two hours total wall-clock.

Agents should:

- run a very short sanity/throughput pilot first;
- stop clearly poor candidates after roughly 20–30 minutes;
- save promising checkpoints early;
- avoid architecture or hyperparameter sweeps;
- avoid automatically consuming the entire two-hour budget.

If a candidate is clearly strong after 45–60 minutes and the curve has flattened, preserve it and stop.

# Experiment Order

```text
1. Retrain / expand current belief head
2. Raw-observation CNN
3. C1-feature CNN
4. Hybrid raw+C1 CNN
5. Scaled Ataraxos-like autoregressive Transformer
```

Do not assume all five must run.

## Sequential Stopping Policy

```text
Agent 1
   ↓
review
   ↓
Agent 2
   ↓
review
   ↓
Compelling engineering winner already?
   ├─ yes -> stop Phase 11B
   └─ no
        ↓
      Agent 3
        ↓
      Agent 4
        ↓
      review
        ↓
Still important uncertainty?
   ├─ no -> stop Phase 11B
   └─ yes -> Agent 5
```

A result around `R_CE 0.94–0.95` at very low inference cost is strong enough that continuing every later experiment may not be worthwhile.

# Engineering Winner Rule

Do not choose solely by lowest cross-entropy.

Discard candidates that fail basic correctness, leak hidden truth, or are clearly no better than the old head.

Among viable candidates:

1. Prefer materially lower overall `R_CE`.
2. Give substantial weight to Scout-rush/generalization performance.
3. If two candidates are within roughly `0.005 R_CE`, prefer the cheaper and simpler model unless the more expensive model has a meaningful unusual-behavior advantage.
4. A dramatically better model can justify additional complexity.
5. Search-integration complexity is part of the decision.

# Phase 11B Exit Artifact

At the end of the sprint, do not rewrite `phase11_system_v1` and do not change the Phase 11 `FAIL`.

Create one additive artifact:

```text
phase11b_engineering_selection_v1
```

containing at minimum:

```text
selected candidate
checkpoint digest
architecture
common-data identity
development metrics
training cost
inference cost
reason selected
all losing candidate references

phase11_final_classification = FAIL
phase11_test_bank_spent = true
scientific_claim = none
```

The selected model is an engineering candidate, not a scientifically accepted replacement.

# Return to the Normal Project

After Phase 11B, resume the rigorous methodology in a newly written Phase 12 contract.

Phase 12 should explicitly acknowledge that Phase 11 formally failed and Phase 11B was an engineering detour that did not scientifically overturn Phase 11.

Use fresh evidence in Phase 12. The old Phase 11 sealed bank remains permanently spent.

Potential Phase 12 comparison:

```text
direct C1 policy
search + remaining-count belief
search + old Phase 11 neural belief
search + selected Phase 11B belief
```

Optionally, for diagnosis only:

```text
search + true hidden information (oracle)
```

The oracle is never deployable.

The practical project question remains:

> Does the improved belief model help search choose moves that win more Stratego games?
