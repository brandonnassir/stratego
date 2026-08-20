# Phase 12 Agent 1 — Minimal Search Engine and Belief Providers

Generated 2026-08-20T05:37:17Z by `scripts/run_phase12_agent01.py`.

Engineering artifact of the Phase 12 rapid search-engineering phase. Sanity work only: no match run, no strength claim, no MEDIUM benchmark.

## 1. Search architecture

```text
search version   phase12_root_world_search_v1
root             one accepted Phase 9 C1 forward -> direct action, priors
candidates       all legal actions if <= 8, else top 8 by Phase 9 policy
                 (the direct action is candidate 0 by construction)
worlds           K hidden worlds sampled once at the root by the provider,
                 duplicates evaluated once and weighted by multiplicity
materialization  root state cloned, hidden opponent ranks overwritten —
                 the accepted anti-leak permutation transformation; the
                 root's observation and legal actions are re-derived in
                 every world and required identical (runtime gate)
rollouts         accepted Phase 9 greedy policy for both sides, batched
                 across all live (candidate, world) sims at each ply
leaf             exact terminal result overrides; otherwise C1 value head
score            S(a) = Q(a) + beta * log(pi(a) + epsilon)
selection        highest score, ties to the lowest normalized action id
```

`beta = 0.1`, `epsilon = 1e-06` — one fixed modest prior, no grid search. `rollout_depth` counts plies after the candidate action; rollouts/action/world = 1 (greedy rollouts are deterministic, so repeats would be identical).

## 2. Model roles and identities

```text
policy/value/rollout/leaf  accepted Phase 9 C1  (state digest f1df694d59e34359...)
agent1c beliefs            agent01_1c_final_block_plus_mlp
                           checkpoint sha256 a125208605f5e68c... (surviving bytes)
original_phase11 beliefs   accepted belief head (digest a9df48a1adcd29b1...)
remaining_count beliefs    remaining_count_belief_v1 / count_uniform_world_sampler_v1
oracle                     true hidden state, offline diagnostic only
```

Agent 1C's policy/value heads are never consulted: the adapter reads only its belief logits, and the move model is a separately loaded, digest-checked accepted C1.

## 3. Belief-provider interface

```text
provider.sample_assignments(public_state, n, seed) -> n x {piece_slot: rank}
provider.predict_marginals(public_state)           -> {piece_slot: 12-vector}
```

Non-oracle providers read a `Phase11BPublicState` (frozen public-state document + 127-channel observation) and structurally cannot see truth. The neural providers are the accepted Phase 11B adapter wrapped unchanged, so their worlds go through the accepted Phase 11 sampler mathematics by import. The oracle uses a separate privileged method, requires `offline_diagnostic=True` at construction, and is refused by the factory and the engine in production configurations.

## 4. Sanity checks

```text
all_four_belief_providers_execute                True
all_selected_actions_legal                       True
all_sampled_worlds_legal                         True
direct_c1_action_always_among_candidates         True
fixed_seeds_reproduce_the_search_action          True
oracle_unavailable_in_production                 True
tiny_latency_measured                            True
small_latency_measured                           True
```

Positions: plan0_ply0, plan0_ply16, plan1_ply24, plan1_ply60, plan2_ply40 — fresh seeded-categorical C1 playouts from accepted setup sources (dev-plan grammar), plus the opening position.

## 5. TINY / SMALL latency (device: cpu)

| provider | preset | mean s/move | median | C1 fwd/move | fwd pos/s | max batch | unique worlds | move-change | reproduced |
|---|---|---|---|---|---|---|---|---|---|
| agent1c | SMALL | 0.328 | 0.325 | 884 | 2729 | 128 | 16.0 | 0.00 | yes |
| agent1c | TINY | 0.125 | 0.124 | 319 | 2538 | 64 | 8.0 | 0.20 | yes |
| oracle | SMALL | 0.042 | 0.042 | 57 | 1346 | 8 | 1.0 | 0.00 | yes |
| oracle | TINY | 0.030 | 0.030 | 41 | 1351 | 8 | 1.0 | 0.00 | yes |
| original_phase11 | SMALL | 0.329 | 0.330 | 888 | 2664 | 128 | 16.0 | 0.00 | yes |
| original_phase11 | TINY | 0.125 | 0.125 | 321 | 2541 | 64 | 8.0 | 0.00 | yes |
| remaining_count | SMALL | 0.324 | 0.324 | 882 | 2672 | 128 | 16.0 | 0.00 | yes |
| remaining_count | TINY | 0.121 | 0.122 | 317 | 2591 | 64 | 8.0 | 0.00 | yes |

Latency is end-to-end per decision (worlds, materialization, rollouts, scoring), averaged over the sanity positions and their repeat runs. `fwd pos/s` is C1 forward positions per second through the whole search stack; the oracle rows run one unique world, which is why they are cheap.

### MPS probe (agent1c only)

| provider | preset | mean s/move | median | C1 fwd/move | fwd pos/s | max batch | unique worlds | move-change | reproduced |
|---|---|---|---|---|---|---|---|---|---|
| agent1c | SMALL | 0.230 | 0.196 | 884 | 3469 | 128 | 16.0 | 0.00 | yes |
| agent1c | TINY | 0.137 | 0.078 | 319 | 1631 | 64 | 8.0 | 0.20 | yes |

MPS forward passes reproduced identical decisions on repeat runs.

## 6. MEDIUM smoke decision (not a benchmark)

```text
provider agent1c  position plan1_ply24  seconds 0.825  c1_forwards 2284  unique_worlds 32  max_batch 256
```

## 7. Cost profile and main observed bottleneck

```text
          agent1c SMALL  forward 0.73  observation 0.10  other 0.17
          agent1c TINY   forward 0.71  observation 0.09  other 0.20
           oracle SMALL  forward 0.83  observation 0.05  other 0.12
           oracle TINY   forward 0.82  observation 0.05  other 0.13
 original_phase11 SMALL  forward 0.74  observation 0.10  other 0.16
 original_phase11 TINY   forward 0.71  observation 0.09  other 0.19
  remaining_count SMALL  forward 0.72  observation 0.10  other 0.18
  remaining_count TINY   forward 0.72  observation 0.09  other 0.18
```

Main observed bottleneck on cpu: c1_forward_passes (mean time fractions — forward 0.75, observation building 0.08, other 0.17).

## 8. Deliverables and status

```text
stratego/search/phase12/{contract,providers,engine}.py
tests/search/test_phase12_{providers,engine}.py  (+ conftest)
reports/phase12/agent_01_report.md
reports/phase12/agent_01_summary.json

phase11_final_classification = FAIL
phase11b_selection           = Agent1C
scientific_validation_status = not performed
oracle_available_in_production = False
```

Stop condition reached: the search core works and every sanity check passes. No large match run was started; Agent 2 (belief-to-decision diagnostic) is not launched automatically.
