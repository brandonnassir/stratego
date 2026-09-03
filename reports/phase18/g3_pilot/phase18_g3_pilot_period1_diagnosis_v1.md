# Phase 18 Agent 6 — G3 pilot v1: diagnosis of the period-1 consequential stop

**Record:** `phase18_g3_pilot_period1_diagnosis_v1.json` (computed read-only from the
immutable `output/phase18/runtime/g3_pilot_v1` evidence; every input file's sha256 is listed there).
**Pilot attempt:** `g3_pilot_v1`, run 2026-09-03 from the published source commit
`c12bed35b8cc0f5a92f87e26ad9af4f0ee7d18c7` (execution worktree `output/phase18/worktrees/g3_pilot_exec`).
**Outcome of the attempt:** candidate and control each completed period 1 and wrote their gate
bundle; `--check-matching` reported `matched: false` on exactly one field,
`period 1 collection/live/commit_digest differs between lineages`, and the launch sequence
stopped as designed. Nothing was resumed.

## 1. What the evidence shows

```text
finding                                                       candidate                  control
commits in period 1 (journal = index)                         1,709                      1,709
ordered game ids                                              identical
trajectory sha256 per commit                                  identical (all 1,709)
selected-decision lists per commit                            identical (all 1,709)
metadata line per commit, `lineage` removed                   identical (all 1,709)
metadata fields that differ                                   ['lineage'] only: "candidate" vs "control"
raw commit_digest (period_0001.done.json)                     413606e45edb…               50d02c2a5e92…
raw commit_digest recomputed from the files                   reproduces                 reproduces
lineage-neutral commit digest (metadata hashed w/o lineage)   d7b48a7b1abd…               d7b48a7b1abd…   (equal)
C1 period 1: updates / keys_digests / live seeds              64, identical across lineages
C1 rows planned                                               8,192 canonical + 8,192 live per lineage
C1 rows recorded as served                                    live_rows_served = 0; canonical_rows_served absent
C1 row batch size (every one of 64 rows)                      256
every other required period-1 matching field                  passed
```

The raw `commit_digest` is `sha256` over `game_id|trajectory_sha256|metadata_sha256` per commit
in order, and `metadata_sha256` is the sha256 of the metadata line the live writer stores. That
line carries `"lineage": "candidate"` in one store and `"lineage": "control"` in the other, so the
1,709 `metadata_sha256` values differ pairwise while the 1,709 game ids and trajectory digests are
equal. Removing exactly that field makes every metadata document identical and the recomputed
digests equal.

## 2. Conclusion

**The period-1 stop was a checker defect, not a gameplay mismatch.** Both lineages played and
committed byte-identical games; the gate required a digest that includes the lineage name to be
equal across differently named lineages, which cannot hold once any game completes. The Stage 6B
smoke pilot did not catch it because its period 1 completes no game (3 slots x 64 plies), so both
stores were empty and their raw digests trivially equal.

A second, independent defect surfaced in the same records: the accepted Phase 8 trainer's metric
row keeps only the cache counters of the batch statistics, so the pilot recorded
`live_rows_served = 0` for a period that served 8,192 live rows (every row consumed a batch of
256 = 128 canonical + 128 live, as the planned counts, the key digests and the batch size show).
This is a telemetry omission; it did not change what was trained.

## 3. What the repair changes and what it does not

The repair (this commit) replaces the cross-lineage raw-digest condition with a lineage-neutral
semantic comparison of the live stores (equal commit counts and order, game ids, trajectory
digests, selected-decision lists and metadata with only `lineage` permitted to differ, which
must read `candidate`/`control` respectively), keeps the raw digest for each store's own file
integrity, reports both raw digests as audit information, records the served canonical and
live counts on every C1 row and in the period record, and fails a period whose served counts
differ from its planned counts. `phase18_g3_telemetry_neutrality_v1.json` proves on CPU that
the telemetry change leaves every training digest, loss, gradient norm and key digest bitwise
identical to the published commit. The setup-learning equation, seeds, schedules, batch
composition, budgets, evaluation rules and checkpoint formats are untouched, and the contract
digest (`7cd75349…`) is unchanged.

The `g3_pilot_v1` runtime, its bundles (candidate `bundle_1 = 92fcf4d2…`, control
`bundle_1 = a4010818…`), logs, launch manifest and failed matching report are preserved unchanged
as evidence. Period 1 is repeated from the repair commit in a fresh namespace
(`output/phase18/runtime/g3_pilot_v2`); the v1 runtime is never resumed.
