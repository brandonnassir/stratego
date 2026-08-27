# Phase 16 — Agent 2
## Stochastic search: sample the move, sample the rollouts, measure the cost

## Mission

The deployed player is deterministic end-to-end: argmax over
`S(a) = Q(a) + beta*log(pi(a)+eps)` computed from greedy rollouts. The operator
defeats it partly by learning its predictable decisions. Build and select a
**stochastic play configuration** that keeps machine-measured strength within a
predeclared margin while restoring unpredictability.

Paper grounding: Ataraxos *samples* its move from a KL-anchored search distribution
(π_search ∝ [e^q̂ ρ^α π_θ^β]^(1/(α+β))) and its rollouts follow the move network;
its ablations show the policy-anchor term is worth 485 Elo while raw depth/rollouts
are worth ~29 each. Note that sampling from `softmax(S(a)/τ)` is exactly the
magnet-free form of their distribution, so this change is an alignment with the
paper, not an invention.

Read `00_PHASE_16_OVERVIEW.md` first; its rules bind. Do not edit Phase 12 or
Phase 15 code — import it.

## 1. Process boundary and namespaces

Boundary identical to Agent 1 section 1. Namespaces:

```text
stratego/search/phase16/         tests/search/phase16/
scripts/run_phase16_agent02.py   scripts/play_phase16.py
checkpoints/phase16/             reports/phase16/
```

## 2. What to build

Two independent, seed-deterministic knobs over the frozen engine
(`phase12_root_world_search_v1` via the Phase 15 systems), everything else
byte-identical — candidate rule, world sampling, dedup, caps, fallback, oracle
refusals all unchanged:

1. **Move sampling** — play `a ~ softmax(S(a)/τ)` over the existing candidate set.
   `τ = 0` reproduces the accepted argmax exactly (control). RNG seeded per decision
   from the accepted seed derivation plus the arm id: same seed ⇒ same move.
2. **Rollout sampling** — rollout actions for both sides drawn from the move model's
   distribution at temperature `τ_r`, optionally restricted to the smallest set
   covering top-p = 0.9 probability mass (blunder-tail trim). `τ_r = 0` reproduces
   the accepted greedy rollouts exactly (control).

A test must prove both zero-temperature paths replay a frozen Phase 15 decision
bit-identically, and that nonzero temperatures are reproducible from the seed.

## 3. Stage 1 — position diagnostics (no games)

On 120 fresh orientation-gated replayed positions (Phase 15 pattern), grid:

```text
τ    ∈ {0 (control), 0.15, 0.30, 0.60}
τ_r  ∈ {0 (control), 1.0}
```

at TINY and MEDIUM budgets. Per arm, over 16 reseeded replays per position:

- **repeat rate** — fraction of replays choosing the modal move (argmax control = 1.0);
- **played-move entropy** and agreement with the τ=0 choice;
- **oracle Q-regret** — reuse the Phase 15 mixture-pilot machinery by import
  (shared candidate set makes regret well defined; read excess over the common
  β-floor exactly as that report does).

**Predeclared filter:** an arm survives if its mean oracle Q-regret excess is within
**+0.010** of the τ=0 control. Report the full grid regardless.

## 4. Stage 2 — the match pack

Surviving arms play the paired match pack vs the τ=0 control:

- Boards: Agent 1's `phase16_benchmark_v1` (fallback if its handoff has not landed:
  draw a fresh 60-board balanced set exactly per Phase 15 Stage C rules and name it
  `phase16_agent02_interim_pack_v1`).
- Budgets: TINY and MEDIUM. Same seeds, paired per board.
- **Predeclared selection:** among arms with EWR within **0.05** of the control at
  MEDIUM, select the one with the *lowest repeat rate* (most varied). If none
  qualifies, select `τ=0.15, τ_r=0` if it is within 0.05, else report
  no-viable-stochastic-mode and keep argmax.
- When Agent 1's adversarial pack exists, also score the selected arm on it
  (opponent-side setups) and report the delta vs the deterministic control on the
  same boards.

## 5. Repeat-encounter probe (recorded, not gating)

Selected arm and control each play 20 sequential games vs each of the two strongest
fixed opponents (p18, p24 direct) with a fresh board per game; report the per-index
EWR trend. Fixed bots cannot adapt, so this is a weak proxy — say so. The real
adaptation test is the operator series (Agent 1's protocol / Agent 5's exam).

## 6. `scripts/play_phase16.py`

The human-facing CLI that supersedes `play_phase15.py` (which stays untouched):

- all Phase 15 modes, by import;
- `varied_strength` — selected stochastic configuration at MEDIUM budget;
- `varied_fast` — same configuration at TINY;
- integrates Agent 1's operator logging module when present (fallback: local JSONL
  with the same schema);
- same information boundary as `play_phase15.py`: legal knowledge only, oracle
  refused by name and by absence.

Re-measure move-time caps for the varied modes from an **idle** run; sampling must
not change latency materially (it adds one softmax draw).

## 7. Candidate freeze and handoff

`checkpoints/phase16/phase16_stochastic_candidate_v1.json`: selected `τ`, `τ_r`,
top-p, budgets, caps (idle-measured), the bound engine/model digests (P24, B24,
temperature 1.0), Stage 1/2 headline numbers with pack names, and
`known_limitations` (state plainly: machine packs cannot measure adaptation
resistance; the operator exam does).

## 8. Report

`reports/phase16/agent_02_report.md`, sections mirroring this file, every table
carrying its own game/position counts, no significance claims.
