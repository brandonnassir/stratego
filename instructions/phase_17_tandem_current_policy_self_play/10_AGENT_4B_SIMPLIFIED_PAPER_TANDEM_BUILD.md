# Phase 17 — Agent 4B
## Narrow conversion to the simplified paper-shaped tandem recipe

## Mission

Starting from commit `c2c0365`, convert the completed Agent 4 tandem system to operator
decision D10 with the smallest clear implementation change. Reuse the fixed-transition
runner, current-policy routing, persistence, export, and integrity safety foundation.
Do not rebuild or recertify them.

Read `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` and
`09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md`. D10 governs every conflict with
Agents 1–4, D3/D4/D5/D7-B/D9-B, their handoffs, and older plan sections.

## 1. Required implementation changes

1. Add recipe/config identity `phase17_simple_paper_tandem_v1` and production run ID
   `RUN-2026-B`. Refuse incompatible checkpoints.
2. Ensure production initialization loads exact Phase 9 move weights and creates a
   newly random setup model. No rehearsal setup state may enter production.
3. Replace the active adaptive setup-KL controller with fixed reverse
   `D_KL(current || behavior)` coefficient `0.1`. Telemetry and checkpoints must call
   this a fixed coefficient, not beta/target/controller state.
4. Replace the re-horizoned setup alpha with `0.1 * global_iteration^-0.3`, using the
   same one-based iteration counter as the tandem runner. No floor and no dependence on
   expected run length `N`.
5. Replace D7-B with the printed paper setup advantage:
   `(outcome - E[value]) + alpha * (I - h_behavior)`. Keep `L_h` targeting `I/10`.
6. After each fixed-transition move window, train for five setup epochs on every setup
   episode completed in that window, exactly once. Remove the active fixed quota,
   warm-up minimum, max-age selection, and backlog balancing. Persist only the
   current iteration's not-yet-consumed completed buffer for crash correctness.
7. Generate a fresh 512-per-side setup pool from the current raw setup snapshot every
   global iteration. Refill within an iteration only from that same snapshot.
8. Convert statistical setup/EWR/KL/diversity stop predicates to telemetry warnings.
   Retain D10's integrity and unrecoverable-failure stops.

Do not change the setup architecture, move learner objective, fixed-transition target
logic, policy sampling, current-policy rebind, EMA roles, external bundle semantics,
or exact active-game persistence.

## 2. Keep the code direct

Prefer deleting or bypassing active controller/quota paths over adding another policy
layer. Historical checkpoint readers may remain if removing them would create risk,
but the D10 production config must have one unambiguous path and may not serialize an
adaptive setup controller as though it were active.

Do not add new gate frameworks, threshold registries, diversity certification code,
parameter sweeps, dashboards, or experimental branches.

## 3. Targeted verification only

Run focused tests for:

- fixed setup KL coefficient and reverse direction;
- global-iteration alpha values at iterations 1, 2, and a later iteration;
- printed paper advantage component arithmetic using recorded behavior fields;
- Phase 9 move plus fresh setup initialization and checkpoint refusal;
- all completed setup episodes consumed exactly once for five epochs;
- one short current-policy tandem iteration with legal sampled moves and setups;
- checkpoint round trip without duplicated or lost completed setup outcomes; and
- structural absence of prohibited training participants.

Then run the D10 end-to-end smoke, capped at 30 minutes. Use a non-production run ID
and discard its learned weights. Do not rerun Agent 3's standalone gate or Agent 4's
200-iteration concentration experiment.

## 4. Handoff

Deliver:

```text
reports/phase17/agent_04b_report.md
reports/phase17/phase17_simple_tandem_preflight.json
reports/phase17/phase17_simple_tandem_handoff_v1.json
```

The report should be short: exact source/config/start digests, files changed, targeted
tests, smoke duration, the nine D10 smoke checks, and any issue that could silently
invalidate the 12-hour run. Do not analyze strength, certify diversity, tune the
recipe, or start production.

Set `ready_for_short_launch_check: true` only when the D10 smoke checks pass. Observable
but finite learning behavior is not a blocker.
