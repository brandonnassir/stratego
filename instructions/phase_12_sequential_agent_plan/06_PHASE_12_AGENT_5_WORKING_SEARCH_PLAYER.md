# Phase 12 — Agent 5
## Working Search-Enhanced Player

## Mission

Turn the best Agent 1C search configuration into a stable working player suitable for the next stages of the project.

This is engineering integration, not scientific final acceptance.

## 1. Production stack

Production must use:

```text
accepted Phase 9 C1
    policy/value

Agent 1C
    belief

selected Phase 12 search preset
    search
```

Do not substitute Agent 1C policy/value outputs.

The oracle provider must be impossible to enable through the normal production path.

## 2. Search modes

Expose at minimum:

```text
direct
tiny search
small search
selected/best search
```

Use explicit named configurations rather than hidden defaults.

## 3. Time controls

Add a per-move search time cap.

If search exceeds the cap, crashes, encounters an internal error, or cannot produce a valid search score:

```text
fall back to the direct accepted Phase 9 policy
```

Do not forfeit or return an illegal action.

Record fallback events.

## 4. Performance optimization

Profile before changing architecture.

High-value optimizations include:

- batching C1 evaluations across worlds;
- reusing root C1 outputs;
- avoiding repeated public-state reconstruction;
- caching sampled root worlds during one search;
- caching invariant legal structures within one search call.

Trunk sharing between Phase 9 C1 and Agent 1C may be considered only if exact equivalence can be demonstrated cheaply.

Do not delay the working player merely to implement shared-trunk optimization.

Two separate small models are acceptable if latency is practical.

## 5. Integration

Expose the player through the existing game/control interface.

At minimum the user should be able to choose:

```text
direct C1
search + Agent1C
```

for machine-vs-machine and human-vs-model games where supported.

The selected mode and search budget should be visible in logs/UI.

## 6. Hidden-information safety

Production search must never use the oracle.

Confirm:

```text
oracle_available_in_production = false
```

Sampled hidden worlds are internal hypothetical states only.

The real root player receives only public information.

## 7. Quick final engineering check

Run only enough games to confirm:

- legal play;
- stable search;
- no hidden-information leak;
- practical move latency;
- deterministic/fixed-seed behavior where expected;
- direct-policy fallback works;
- observed strength is broadly consistent with Agent 4.

Do not turn this into another long validation phase.

## 8. Freeze the engineering candidate

Create:

```text
phase12_search_candidate_v1
```

Record:

```text
accepted Phase 9 checkpoint identity
Agent 1C checkpoint identity
search algorithm version
belief provider = Agent1C
world count
root candidate count
depth
policy regularization
latency
quick match result
fallback behavior
known limitations
phase11_final_classification = FAIL
phase11b_selection = Agent1C
scientific_validation_status = not performed
oracle_available_in_production = false
```

Preserve all Phase 12 agent reports and configs.

## 9. Stop condition

Stop after delivering a stable working search-enhanced player.

Do not automatically launch another long scientific validation phase.

Do not modify or reinterpret the formal Phase 11 result.
