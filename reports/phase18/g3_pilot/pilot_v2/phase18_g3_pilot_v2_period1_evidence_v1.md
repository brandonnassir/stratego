# Phase 18 Agent 6 — G3 pilot v2: period 1 repeated from the gate correction

Repeat of the consequential-stop test from the repair commit
`8c1baa8ff163c593d72c63b711e905937b557669` (branch `phase18/g3-stage6b-harness`), in a clean
detached execution worktree `output/phase18/worktrees/g3_pilot_exec_v2` with the fresh runtime
namespace `output/phase18/runtime/g3_pilot_v2`. Exactly three stages ran after the launch
manifest: candidate through period 1, control through period 1, `--check-matching`. No period 2,
no evaluation, none of the remaining 255 periods. The stopped first attempt (`g3_pilot_v1`) is
preserved unchanged; copies of its records are under `../pilot_v1_stopped/`.

Official records at the standard paths: `../phase18_g3_pilot_launch_manifest_v1.json` (bound to
`8c1baa8`, runtime root `g3_pilot_v2`, clean tree) and `../phase18_g3_pilot_matching_v1.json`
(**`matched: true`**, no problems). Machine-checked details: `phase18_g3_pilot_v2_period1_evidence_v1.json`.

```text
result required before stopping                          candidate                     control
matched                                                  true (0 problems)
semantic live-store identity                             PASS: 8/8 checks, 1,709 commits each, neutral digest d7b48a7b1abd… equal
raw live commit digest (audit only, expected unequal)    413606e45edb…                  50d02c2a5e92…
games / outcomes / completed ids / plies / in flight     identical                     identical
C1 training keys (64 keys_digests), live seeds, cursor   identical                     identical
C1 updates                                               64                            64
rows served per update                                   128 canonical + 128 live      128 canonical + 128 live
canonical rows served (planned 8,192)                    8,192                         8,192
live rows served (planned 8,192)                         8,192                         8,192
setup model                                              1 update applied, raw and     0 updates, raw = EMA = init
                                                         EMA moved from init, 0 skips  digest
integrity counters (legality, orientation, attribution,  all zero                      all zero
non-finite, duplicates)
immediately_terminal_setups (telemetry, not a failure)   1                             1
bundle_0 verifies                                        fa7a5faf4c5d…                 bb3b4bf1e292…
bundle_1 verifies, consistent with the period record     3bcca8dc7a25…                 c43710d440f6…
C1 weight digest across lineages                         differs: MPS, reported only (P18-D002), as in v1
```

Cross-run, v1 versus v2 (same run id, namespace, seeds, configuration): period-1 games,
outcomes, live stores (the raw commit digest of each lineage is byte-identical to v1's), C1
keys, live seeds, pool and setup digests reproduced exactly; the C1 weights differ (MPS) with
per-row losses within 1e-6; `live_rows_served` reads 8,192 instead of v1's 0.
