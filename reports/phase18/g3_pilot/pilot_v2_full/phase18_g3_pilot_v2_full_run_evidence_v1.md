# Phase 18 Gate G3 — `g3_pilot_v2` full run evidence

Run `G3-PILOT-2026-A`, source commit `8c1baa8ff163c593d72c63b711e905937b557669`,
runtime `output/phase18/runtime/g3_pilot_v2`, execution worktree
`output/phase18/worktrees/g3_pilot_exec_v2` (detached at the source commit).

## Status

Training **complete** for both lineages. The matching and fairness checks **pass**.
Both primary evaluation arms are **complete and reconciled**. The frozen analysis
**could not be run**: `--analyse` fails on a reader defect. See
`phase18_g3_pilot_analysis_blocker_v1.json`.

**The primary contrast has not been computed by any path.** Computing it outside the
frozen analysis, before the reviewer settles the repair, would let the outcome
influence the choice of repair.

## Training

| | candidate | control |
|---|---|---|
| periods | 1..256 once each | 1..256 once each |
| C1 updates | 16,384 | 16,384 |
| C1 global step | 16,384 | 16,384 |
| canonical rows served | 2,097,093 | 2,097,093 |
| live rows served | 2,097,152 | 2,097,152 |
| setup updates | 256 (moved on every one) | 0 |
| setup EMA updates | 256 | 0 |
| setup optimizer steps | — | 0 |
| final bundle | `2162b448017ca844` | `e1c410021f71124f` |
| legality / orientation / attribution / non-finite failures | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Canonical rows are 2,097,093 rather than 256 x 8,192 = 2,097,152 because the canonical
cursor crosses an epoch boundary at period 153, whose planned batch is short (8,133).
Both lineages hit the same boundary at the same period, so the budgets stay equal. The
contract requires *served == planned*, which holds in every period of both lineages.

The control setup model is frozen at every one of its ten bundles: raw and EMA digests
equal the initialisation digest throughout, with zero optimizer steps. The runner also
asserts this internally each period, so any drift would have halted the run.

## Matching (regenerated at period 256)

`matched: true`, `problems: []`. The accepted period-1 semantic live-store identity
remains valid: 1,709 commits per lineage, identical lineage-neutral commit digest,
differing only by the lineage stamp. Raw commit digests differ by construction and are
audit information only.

## Evaluation

Frozen 2,560-case handcrafted-opponent schedule on the established G1 evaluator,
`battleless 200` evaluation rules (P18-A001). Only the two primary arms were run; the
period-0 and period-128 diagnostic arms were **not** evaluated.

| | candidate_final | control_final |
|---|---|---|
| games planned / completed | 2,560 / 2,560 | 2,560 / 2,560 |
| reconciles | true | true |
| errored / illegal / policy errors | 0 / 0 / 0 | 0 / 0 / 0 |

Schedule digests equal across arms; setup-model digests differ; match-id sets identical;
all eight persisted arm-invariant fields agree on every one of the 2,560 paired games.
The two arms played a different own setup in 2,524 of 2,560 games, which is the expected
consequence of the candidate's setup model having learned.

## Operational record

The candidate leg ran 2026-09-03 14:50 to 2026-09-04 01:55 (39,813 s) and completed
without interruption. The control leg started automatically at 01:55 and was killed at
period 169 by a host out-of-memory crash at 08:30. It was resumed from its latest
complete verified bundle (`bundle_0160`, id `fc0f4d017f68`) per the recovery rule; the
nine period records after that bundle, their live stores, 576 C1 rows and the previous
run state were **archived, not deleted**, under
`control/archive/20260904T123848Z_after_bundle_0160/`. No bundle was moved or destroyed.
The control leg then ran periods 161..256 in 13,140 s. Every one of the 20 bundles was
re-verified after the crash.

Per-period cost drifts upward over a leg — about 130 s at period 2 rising to about 212 s
by period 256. Collection plateaus near 117 s; the growing term is C1, slope about
+0.12 s/period. Size future horizons of this shape on roughly 155 s/period, not on the
early rate.

## Files

- `phase18_g3_pilot_matching_final_v1.json` — the regenerated 256-period matching record
- `phase18_g3_pilot_v2_bundle_identities_v1.json` — all 20 bundles, re-verified
- `phase18_g3_pilot_analysis_blocker_v1.json` — the reader defect blocking `--analyse`
- `run_summary_{candidate,control}_final.json`
- `arm_record_{candidate,control}_final.json`
- `{candidate,control}_final_receipts.jsonl` — 2,560 immutable per-game receipts each
- `log_*.txt` — concise period and stage lines

Runtime checkpoints, live `.records` stores and evaluation game chunks are deliberately
not committed.
