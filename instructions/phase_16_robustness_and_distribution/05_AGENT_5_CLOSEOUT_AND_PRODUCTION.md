# Phase 16 — Agent 5
## Phase 14 closeout, repository unfreeze, production run, operator exam

## Mission

Formally close Phase 14 after its deadline passes, lift the repository freeze
safely, land two phases of untracked work in git, then — gated on Agents 1–4 —
assemble and run the Phase 16 production training and administer the operator exam.

**Nothing in this file executes before `2026-08-28T16:15:34.689Z`.** Until that
instant the freeze in `00_PHASE_16_OVERVIEW.md` §2 binds absolutely. The operator
has decided the run will not be resumed; the remaining window is simply allowed to
expire.

## 1. Phase 14 finalization (first action after the deadline)

1. Verify the deadline has passed (UTC) and no Phase 14 process is running.
2. Follow `PHASE_14_RUNBOOK.md`'s closeout procedure (`--role finalize` path),
   which refuses before the deadline by design. Expected outcome: `closed: true`
   in the run state, final candidate selection recorded by the frozen predeclared
   rule (expected winner: hour-18; its 2,200-game verification —
   0.8086 [0.7924, 0.8248] — already exists in the sidecar evaluation).
3. Record in the report: closure timestamp, selected candidate identity and digest,
   final step/iteration counts, and the location of every preserved artifact
   (external volume preservation dirs, sidecar eval, ledger).
4. Only after `closed: true`: the freeze is lifted.

## 2. Landing the untracked work (ordered, reviewed commits)

Before any commit: run the full pytest suite (must be green) and confirm a current
untracked backup exists (Agent 1 §3). Then, in order:

1. **The manifest**: verify `reports/phase13/phase14_launch_manifest_v1.json` is
   still the only tracked modification and its diff is the expected self-referential
   dirty-list; commit it with a closure message.
2. **Phase 15**: all `phase15` namespaces, scripts, reports, checkpoints-metadata
   (respect any existing repo policy on large binaries — model `.pt` files were
   committed in prior phases only where the phase report says so; follow precedent,
   and if unclear, commit code+reports+manifests and record checkpoint digests in
   the report rather than the bytes).
3. **Phase 16**: this instructions directory plus whatever Wave-0/1 artifacts exist
   at close time, same policy.
4. Never `git clean`. Never rewrite history. One phase per commit, messages naming
   the agent reports that document the content.

## 3. Deferred housekeeping (post-unfreeze, low risk, do in this order)

1. **Dashboard**: apply the preserved out-of-tree
   `dashboard_pid_discovery_fix.patch` (in
   `/Volumes/Brandon_Washington/stratego_phase14_preserved_2026-08-23/`), commit,
   and note the still-open single-threaded keep-alive wedge defect (fix belongs to
   a later maintenance task; a restart clears it).
2. **`play_phase12.py`**: do not silently change accepted behavior. Add a loud
   startup deprecation banner naming the Blue-orientation defect and directing to
   `play_phase16.py` (or `play_phase15.py` until Agent 2 lands). The historical
   defect stays documented, not erased.
3. **`stratego_project_docs 2`**: diff against `stratego_project_docs`. Identical →
   delete the copy. Diverged → report the diffs and reconcile with the operator
   before deleting anything.
4. Optionally reclaim the ~15 GB of regenerable `*_prefix_*.npy` caches once the
   backup is verified; they rebuild via `ensure_caches`.

## 4. The production run (gated)

Preconditions, all required:

```text
agent1  phase16_measurement_handoff_v1 verified (benchmark + adversarial pack)
agent3  phase16_recipe_candidate_v1 with adopt_recipe = true
agent2  phase16_stochastic_candidate_v1 present (any verdict — argmax fallback ok)
agent4  optional; if promoted, its provider joins the play stack, not the training
```

If Agent 3's stop-rule fired, there is no production run: report and hand back.

Assemble `phase16_production_config_v1`:

- the winning recipe's flags and schedules, horizon constants re-derived for the
  longer run (document the mapping from the 6-hour probe's iteration count);
- setup mixture: the shootout winner's (expanded, if `setups_causal` held);
- duration: **48–72 h**, wall-clock deadline recorded at launch, hot/archive/
  candidate cadence per Phase 14 precedent (candidates every 3 h);
- **slope-based stop rule, predeclared at launch**: score every candidate on the
  benchmark quick subset + adversarial stratum; if the 12-hour moving slope of the
  composite falls below +0.005 EWR/12 h for two consecutive checks, stop the run —
  never again buy 50 hours of flatline;
- **candidate selection rule, frozen before launch**:
  `composite = 0.5*benchmark_EWR + 0.3*adversarial_EWR + 0.2*worst_opponent`,
  evaluated on the full packs for the top-3 candidates after the run ends;
- launch discipline per `PHASE_14_RUNBOOK.md`: power assertion, external-volume
  space check, preflight, supervisor, and the read-only monitoring posture.

After selection: hand the winning weights to the play stack — Agent 2's stochastic
configuration on top of the new policy, with belief re-fit: rerun Agent 1 of
Phase 15's recipe (by import of the phase15/phase16 belief pipeline) against the
new policy's self-play to produce its matching belief (and Agent 4's AR form, if
promoted). Re-measure caps idle. Freeze the whole assembly as
`phase16_player_candidate_v1` with digests.

## 4b. Carry-forward constraints from Agent 3 (added 2026-08-27)

Agent 3's shootout returned **STOP**, so §4's production run is **not authorized**
by that result. If the operator later authorizes a run from a redesigned
experiment, three findings bind it:

1. **Size the pack to the margin, not the other way round.** The shootout's
   predeclared 0.03 margin on a 60-board instrument is **0.53 standard errors** —
   a coin flip — and the 60-board and 120-board readings of the *same games*
   disagreed on whether arm B cleared. Before predeclaring any margin, compute the
   standard error of the instrument at the expected EWR and require the margin to
   exceed it (≥1 SE minimum; ≥2 SE to be worth a long run). The same rule applies
   to §5's exam.
2. **Re-derive horizon constants; never inherit them.** Use
   `phase16_recipe_candidate_v1.horizon_evidence` — it stores measured
   seconds/iteration and the rule, deliberately not just the constant. `n_ref = 40`
   is correct only for ~313 iterations.
3. **The historical-opponent pool fragments collection.** The accepted collector
   groups pending decisions by acting checkpoint, so a growing historical pool
   (2 → 13 members) fragmented arm A's 96-game lockstep batch and cut throughput
   **1,996 → 830 plies/s**. `pure_current` arms never fragment. A production run
   that wants Phase 14-style mixture fidelity *and* throughput must cap the pool
   or pad batches across snapshots. Decide deliberately and record the choice.

## 5. The operator exam

Administer per `reports/phase16/operator_protocol_v1.md`:

- 20 games, operator vs `phase16_player_candidate_v1` at its varied maximum
  strength, alternating colors, idle machine, everything logged;
- **pass**: model EWR ≥ 0.50 (draws half);
- on failure: the logs are the diagnosis — report per-game results against the
  operator's setup families (which exploit still works), decision-repeat metrics
  across the series, and a ranked list of what Phase 17 should target. A failed
  exam with a clean diagnosis is a successful phase output; say it plainly.

## 6. Report

`reports/phase16/agent_05_report.md`: closure record, commit log, housekeeping
results, production run summary with its h-curve, the frozen candidate identity,
and the exam outcome. This report closes Phase 16.
