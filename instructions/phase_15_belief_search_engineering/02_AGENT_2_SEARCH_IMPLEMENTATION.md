# Phase 15 — Agent 2
## P18/P24 Belief-Guided Search Integration

## Mission

Build and evaluate the complete search players formed from the two frozen Phase 14 move models and the two new Phase 15 belief specialists:

```text
P18 + B18
P18 + B24
P24 + B18
P24 + B24
```

Reuse the working Phase 12 search engine and constrained-world machinery wherever possible. The goal is to select and package the strongest practical complete system, not to redesign search from scratch or produce a scientific claim.

Do **not** train or modify B18/B24. Do **not** alter, pause, stop, kill, restart, or finalize any running Phase 14 task.

## 1. Required starting state

Begin only after Agent 1 has delivered and verified:

```text
P18 source identity and digest
P24 source identity and digest
B18 checkpoint and calibration identity
B24 checkpoint and calibration identity
phase15_belief_corpus_v1 manifest
phase15_search_handoff_v1
passing belief-provider and legal-world tests
```

Verify every handoff digest before implementation or evaluation.

If the handoff is incomplete, implement only code that can be tested with deterministic fixtures, report the exact missing artifact, and stop. Do not manufacture placeholder strength results or repair Agent 1's outputs in place.

## 2. Process boundary

This instruction does not authorize process control.

Before sustained match evaluation or device-heavy profiling:

1. inspect live task status read-only;
2. confirm the work will not compete with a running Phase 14 learner/evaluator/supervisor;
3. if it would compete, complete implementation and CPU-only correctness tests, report `ready_for_compute`, and stop for operator review.

Never signal a process, create an emergency-stop file, edit live run state, rotate live checkpoints, or invoke Phase 14 closeout/finalization.

## 3. Preserve earlier work

Treat all Phase 9–14 artifacts and Agent 1 deliverables as read-only.

In particular:

- do not edit accepted `stratego/search/phase12/` behavior to re-label it as Phase 15;
- do not overwrite `phase12_search_candidate_v1`;
- do not rewrite the Phase 11B closure or Phase 12 reports;
- do not modify P18, P24, B18, B24, their calibration values, or their identity records;
- do not touch the unrelated modified Phase 14 launch manifest;
- do not clean or reset unrelated working-tree changes.

Use additive Phase 15 namespaces:

```text
stratego/search/phase15/
tests/search/phase15/
scripts/run_phase15_agent02.py
checkpoints/phase15/
reports/phase15/
```

Minimal generic fixes to shared code are allowed only if unavoidable, covered by regression tests, and behavior-neutral for the frozen Phase 12 candidate. Prefer adapters and composition.

## 4. Correct model roles

For every combined player:

```text
P18 or P24:
    root policy
    candidate prior
    rollout policy for both simulated sides
    leaf value
    direct fallback

B18 or B24:
    hidden-rank marginals
    legal hidden-world sampling
```

Never use a belief specialist's fine-tuned block for policy or value decisions.

Cross-pairing is intentional. Do not assume B18 belongs only with P18 or B24 only with P24.

## 5. Reuse Phase 12 search mechanics

Start from `phase12_root_world_search_v1` and retain its established mechanics unless a failing correctness test requires a narrowly documented change:

```text
public state
    -> belief provider
    -> sample K complete legal worlds once at root
    -> obtain root policy from selected P18/P24
    -> choose candidate root moves
    -> evaluate every candidate on the same worlds
    -> greedy rollouts using the same selected P18/P24
    -> exact terminal result or selected model's leaf value
    -> average world values
    -> policy-regularized score
    -> highest-scoring legal move
```

Use:

```text
V = P(win) - P(loss)
S(a) = Q(a) + 0.1 * log(pi(a) + 1e-6)
```

Initial candidate rule:

```text
if legal actions <= 8:
    evaluate all legal actions
else:
    evaluate top 8 under the selected P18/P24 policy
```

The direct action from the selected move model must always be included.

Do not tune `beta`, candidate count, depth, and world count simultaneously.

## 6. Phase 15 loaders and providers

Implement digest-bound loaders for:

```text
P18
P24
B18 over its frozen P18 prefix
B24 over its frozen P24 prefix
```

Expose B18/B24 through the same functional belief-provider surface used by Phase 12:

```text
predict_marginals(public_state)
sample_assignments(public_state, n, seed)
```

Use the temperature recorded by Agent 1. Use the accepted constrained-world sampler by adapter/import and preserve every inventory, movement, seed-determinism, and legality check.

Support these provider identities:

```text
remaining_count
b18
b24
oracle   offline diagnostics only
```

The oracle must remain structurally unavailable in production constructors, modes, configs, CLI choices, and serialized production candidates.

## 7. Search configurations

First reproduce the existing presets with the selected move model substituted correctly:

```text
TINY
    worlds       8
    candidates  <= 8
    depth        4 plies

SMALL
    worlds      16
    candidates  <= 8
    depth        6 plies

MEDIUM
    worlds      32
    candidates  <= 8
    depth        8 plies
```

Only after MEDIUM shows a useful improvement and remains below the human-play latency budget may one `STRONG` configuration be tried:

```text
STRONG
    worlds      64
    candidates  <= 12
    depth       10-12 plies
```

Choose one depth in that range from a short latency pilot. This is not permission for a grid search.

Target practical latency:

```text
ordinary strong move       <= 2 seconds preferred
difficult maximum move     <= 5 seconds acceptable
```

Profile before optimizing. Preserve Phase 12 batching, root-output reuse, shared root worlds, duplicate-world weighting, and cooperative deadlines.

## 8. Information boundary and orientation

Every simulated player receives only its legal observation. Sampled truth is internal to a hypothetical world and must never be exposed as root input.

Repeat the hidden-identity permutation test for every policy/belief pairing:

```text
non-oracle production answer must not change
oracle diagnostic is the positive control
```

Do not reuse the contaminated Phase 12 boards as new evidence. Build all new match boards through accepted orientation helpers:

```text
SelectorDraw.oriented(player)
SampledSetup.oriented(player)
or orient_setup(canonical, player)
```

Add the same Red/Blue Flag-row and setup legality gates used by Agent 1. Refuse evaluation if a canonical Blue tuple reaches `create_game()` directly.

## 9. Correctness gate

Before any match pack, prove on fresh positions that:

```text
all four policy/belief pairings load with exact digests
all chosen actions are legal
all sampled worlds are legal
same seed reproduces the same worlds and action
every candidate is evaluated on the same root worlds
direct move is always a candidate
P18 pairings use P18 for policy/value/rollouts/fallback
P24 pairings use P24 for policy/value/rollouts/fallback
B18/B24 are used only for beliefs
oracle is refused in production
deadline and forced-error fallback return the correct direct model's move
Phase 12 frozen-candidate regression tests still pass
```

Run small TINY and SMALL latency probes for all four pairings. Do not start a large match run until the gate passes.

## 10. Engineering comparison arms

Maintain clear names for these arms:

```text
P18 direct
P18 + remaining-count search
P18 + B18 search
P18 + B24 search
P18 + oracle search       offline diagnostic

P24 direct
P24 + remaining-count search
P24 + B18 search
P24 + B24 search
P24 + oracle search       offline diagnostic
```

The oracle answers the ceiling question only:

> If hidden-piece inference were perfect, can this search design use that information?

It is never a deployable arm.

## 11. Stage A — quick decision diagnostic

On one fresh, correctly oriented manifest of tactically meaningful positions, compare each search arm with its own direct move model.

Report:

```text
move-change rate vs direct
agreement with oracle-selected move
legal decision rate
mean/median/p95 latency
C1 forwards per move
world uniqueness
score margin between chosen and runner-up move
```

Interpretation:

```text
oracle helps but learned belief does not
    -> belief/provider quality is limiting

oracle also fails to change useful decisions
    -> search mechanics or value/rollout quality is limiting

learned belief tracks oracle and changes useful decisions
    -> proceed to match comparison
```

Do not scale compute to hide a failed oracle diagnostic.

## 12. Stage B — complete-system match comparison

Use a single fresh paired-board manifest for every arm. Balance colors and setup sources. At minimum include opponents:

```text
P18
P24
accepted Phase 9 anchor
strategic_rule_based
tactical_rule_based
stress_scout_rush
stress_miner_rush
stress_berserker
stress_information_miser
stress_chaos
```

Cover setup families that target observed weaknesses:

```text
balanced_conventional
high_bomb_placement
aggressive_high_rank_front
conservative_high_rank_rear
corner_flag_fortress / near_corner_flag_fortress
distributed_bomb_defense
scout_forward_information
miner_forward
irregular_high_entropy
```

Run a compact engineering pack first. Use enough paired boards to detect large practical differences without turning this into another long scientific phase. Preserve every game record and replay seed.

Report for each arm:

```text
W / D / L
effective win rate
EWR by opponent
EWR by color
EWR by setup source/family
paired delta vs its direct policy
minimum opponent/family score
move latency distribution
fallback rate and reason
move-change rate
search seconds per game
```

No significance claim is required. State the sample size and noise limitation plainly.

## 13. Stage C — budget selection

Take the best one or two learned-belief pairings from Stage B and compare TINY, SMALL, and MEDIUM on the same boards and seeds.

Do not infer that the largest budget is strongest merely because it is larger. Select the cheapest preset that is not meaningfully behind the strongest observed preset and that addresses the aggressive/unusual weakness pack.

Try STRONG only under the condition in section 7.

Record:

```text
EWR improvement per added search second
latency vs forward count
budget-to-budget paired outcomes
human-play practicality
```

## 14. Select the complete player

Select the strongest practical **complete system**, not the direct checkpoint with the best old 128-game headline.

Decision order:

1. reject any system with an information leak, illegal worlds/actions, identity mismatch, or unstable fallback;
2. prefer better overall and worst-stratum match strength on the fresh pack;
3. give specific weight to aggressive, unusual, Scout, Miner/Bomb, and Flag-structure performance;
4. if strength is effectively tied, prefer lower latency and the simpler belief pairing;
5. retain both a practical default and a maximum-strength mode if the slower mode buys a useful observed gain within the 5-second ceiling.

Produce a matrix like:

```text
system       direct EWR   search EWR   worst stratum   median/p95 move   fallback
P18+B18      ...          ...          ...             ...               ...
P18+B24      ...          ...          ...             ...               ...
P24+B18      ...          ...          ...             ...               ...
P24+B24      ...          ...          ...             ...               ...
```

## 15. Working player

Package the selection behind explicit modes. At minimum expose:

```text
p18_direct
p24_direct
selected_search
maximum_strength
```

Keep diagnostic names available for machine evaluation, but do not expose oracle.

Every search mode must have:

- a recorded model/belief/config identity;
- a per-move time cap based on measured p95 with reasonable headroom;
- direct fallback to the same selected P18/P24 move model;
- visible mode, budget, latency, and fallback logging;
- machine-vs-machine and human-play integration where the Phase 12 player already supports them.

On timeout, search error, non-finite score, or illegal result, play the direct legal move. Never forfeit because search failed.

## 16. Required artifacts

Create at minimum:

```text
stratego/search/phase15/...
tests/search/phase15/...
scripts/run_phase15_agent02.py

reports/phase15/agent_02_position_manifest.json
reports/phase15/agent_02_decisions.csv
reports/phase15/agent_02_match_manifest.json
reports/phase15/agent_02_games.jsonl
reports/phase15/agent_02_games.csv
reports/phase15/agent_02_budget_profile.json
reports/phase15/agent_02_system_matrix.json
reports/phase15/agent_02_report.md
reports/phase15/agent_02_summary.json

checkpoints/phase15/phase15_search_candidate_v1.json
```

The frozen candidate must record exact identities/digests for:

```text
selected move model: P18 or P24
selected belief model: B18 or B24
belief calibration
search version and preset
world/candidate/depth budget
policy regularization
expected latency and time cap
direct fallback identity
fresh match-manifest digest
quick engineering results
known limitations
oracle_available_in_production = false
scientific_validation_status = not performed
```

## 17. Completion and stop condition

Finish when:

- all four P/B combinations execute correctly;
- fresh orientation-safe decision and match evidence is preserved;
- count and oracle controls identify whether learned belief is useful;
- a practical search budget and complete player are selected;
- the working player and frozen Phase 15 candidate load by exact digest;
- direct fallback and oracle exclusion are proven.

Then stop and report.

Do not begin another policy-training phase, modify B18/B24, reinterpret old contaminated Phase 12 evidence, or control any running Phase 14 task.
