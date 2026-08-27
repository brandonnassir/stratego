# Experiment framework for future work

**Written 2026-08-27.** This file prepares the documentation for the
forthcoming model buildout. It **does not design that buildout**: it contains no
architecture, no hyperparameters, no training schedule and no authorization.
Those are the project owner's to decide.

Scope: **from this point forward, most new work is engineering and direct model
testing** unless a separate, explicitly designed scientific-validation protocol
says otherwise.

Companions: [`STATUS.md`](STATUS.md) · [`PHASE_HISTORY.md`](PHASE_HISTORY.md) ·
[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md).

---

## 1. The default classification is ENGINEERING

Future model variants and training experiments are **engineering experiments**
by default. An engineering experiment:

- exists to make a build decision;
- predeclares its decision rule before seeing numbers;
- reports standard errors and makes **no significance claim**;
- is valid only on its own named evaluation pack;
- **selects**; it does not **demonstrate**.

Nothing is promoted out of this class by accumulating engineering results. The
only way out is §5.

---

## 2. Every experiment must register these nine fields

Before it runs, and in its own machine-readable artifact. An experiment missing
any field is not interpretable later and its numbers should not be cited.

| Field | What it must name |
|---|---|
| 1. **Starting checkpoint** | Exact path **and** model-state digest. "P24" is not sufficient; `622d9e6caa723c93…` is. Include how it was obtained (ledger entry, archive snapshot, prior arm). |
| 2. **Rules version** | Currently `stratego_project_v1`. Any change to two-square, continuous-chasing, battleless limits or the move-safety limit **requires a new rules identifier** (`02_project_ruleset.md` §227). |
| 3. **Observation version** | Currently `observation_v2_1_127ch`. Plus the action encoding (`source_destination_10000_v1`) and the reference engine version (`phase2_1_reference_1.2.0`). |
| 4. **Setup source** | Library and version, **split** (train / validation / test), the sampler or mixture, **and the orientation gate**. Every board reaching `create_game` must pass `red engine row == canonical rank; blue engine row == 9 − canonical rank`. Import the Phase 15 gate; never re-derive it. |
| 5. **Model architecture** | Parameter count and the contract version, not just a nickname. |
| 6. **Training recipe** | Optimizer, learning-rate schedule **with its horizon**, entropy schedule, epochs, EMA, clipping, minibatch, precision, device, opponent mixture, iteration sizing (whole games vs. fixed decision budget), and retention. |
| 7. **Evaluation pack** | Name, version, digest, game/board count, and which subset is the **decision instrument** vs. which are secondary readings. |
| 8. **Evidence classification** | One of `ACCEPTED` / `ENGINEERING` / `INCOMPLETE` / `INTERRUPTED` / `SUPERSEDED` / `CONTAMINATED` / `PENDING` / `HISTORICAL`, stated in the artifact itself. |
| 9. **Run identity** | The run-series name (§6), the seed namespace, and the compute-lock discipline it ran under. |

Follow the existing pattern: a frozen JSON handoff that re-verifies its own
digests against bytes on disk. See
[`phase15_search_handoff_v1.json`](../reports/phase15/phase15_search_handoff_v1.json)
and [`phase16_measurement_handoff_v1.json`](../reports/phase16/phase16_measurement_handoff_v1.json).

---

## 3. Comparison rules

**Cross-pack comparison is prohibited unless a protocol explicitly supports
it.** This is not a stylistic preference. Phase 16 measured a search advantage
of **+0.1375 ± 0.0414** on one 120-board pack and **−0.029 ± 0.036** on another
drawn from the same library split by the same machinery. Two packs disagreeing
by more than their combined noise is a *measured property of this system*, not a
hypothetical.

Consequences to design around:

1. **Compare paired arms on identical boards and identical seeds.** Report the
   paired delta, not two absolute EWRs.
2. **Name the decision instrument in advance**, and stick to it even when a
   secondary reading disagrees. Phase 16's shootout had the quick-60 subset and
   the full-120 pack **reverse each other on the same games**; the verdict stood
   on the pre-named instrument, and the disagreement was reported as evidence
   the margin was unresolvable.
3. **Size the pack to the margin.** A predeclared 0.03 margin against an
   instrument whose standard error is 0.056 is **0.53 SE** — a rule that cannot
   see the difference it asks about. Compute the SE first, then set the margin,
   then set the pack size.
4. **Start comparisons where there is headroom.** The Phase 16 shootout started
   all three arms from the saturated P24 and could not separate them. Phase 14's
   own curve says the learning is in hour 0→6 (+0.0414) and everything after is
   a random walk. Prefer a curve-shape question over "which arm is best".
5. **Consider selecting on the worst stratum, not the mean.** The `min` stratum
   peaked at h=18 and degraded after, while the mean stayed flat.
6. **Latency claims come from idle, single-process runs only.** Pack numbers run
   ~1.8× inflated under worker contention.
7. **Re-fit every borrowed schedule constant to your own horizon.** The Phase 16
   brief transcribed Ataraxos's power-law LR (fitted to ~43,000 iterations) into
   a ~313-iteration run and would have starved two of three arms by 5×. A
   re-horizoning that moves an endpoint is not a re-horizoning.

---

## 4. Portability and storage

New work should not add machine-specific absolute paths to code or to
contracts. Resolve storage roots through a **single indirection** (a pointer
file or one configuration entry) and record the **resolved absolute path in the
run's own evidence**, where it belongs as a fact about that run.

Existing artifacts are left exactly as recorded. Where a historical path is
load-bearing — above all
`/Volumes/Brandon_Washington/stratego_phase14/`, which holds the only copy of
the interrupted run and the source bytes for P18 and P24 — treat it as a
**dependency to preserve**, not text to rewrite. See
[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) §6.3.

---

## 5. What "validated" would require

**A result is not scientifically validated because its tests pass or because an
internal EWR went up.** Neither is it validated by a bigger engineering pack, a
better point estimate, or agreement across several engineering runs.

Validation requires a **separately designed protocol, frozen before any data is
seen**, that at minimum:

- states the hypothesis and the decision rule before the experiment runs;
- names a **held-out evaluation set that has never been scored**, with an
  access ledger proving it;
- fixes the sample size from a power calculation against the claimed margin;
- specifies the statistical test and the acceptance threshold in advance;
- forbids retuning in response to any reading, and records that nothing moved;
- is executed **once**.

Phases 9, 10 and 11 are the shape of this in practice, and Phase 11 is the
worked example of it going against the project — it FAILED and the result was
kept. Note the cost of this discipline: **`phase11_test_bank_v1` is permanently
spent.** A future belief-repair claim needs a *new* sealed bank.

**Human-strength claims require their own frozen human-evaluation protocol.**
The machine packs in this repository structurally cannot produce one: every
opponent is a fixed policy that cannot learn the player's habits, which is
exactly the property the current candidate was selected to resist. Any human
claim must predeclare the opponent pool and how it was recruited, the number of
games, colour/setup balance, adaptation conditions, the pass threshold, and how
opponent skill is characterized — before the first game. The existing
[`operator_protocol_v1.md`](../reports/phase16/operator_protocol_v1.md) is a
starting point for the operator exam specifically; it is not a general
human-strength protocol, and **it has never been run**.

Until then: **internal-bot EWR is not human EWR**, and the project's original
85%-vs-casual-humans target remains **retired and unmeasured**.

---

## 6. Run naming — decision required from the project owner

**Do not reuse the bare name "Phase 14."** It now permanently denotes the
interrupted 59.97-hour run, its 10 candidates, its emergency-stop record, its
still-open run state and its immutable
`2026-08-28T16:15:34.689Z` deadline. Reusing it would make P18/P24, the launch
manifest, the candidate ledger and the frozen selection rule ambiguous, and
would corrupt exactly the history this documentation set exists to protect.

Candidate conventions, for the owner to choose between:

| Option | Example | Reads as |
|---|---|---|
| **A — suffixed restart** | `Phase 14R`, then `Phase 14R2` | "the Phase 14 attempt, restarted." Keeps the lineage visible; risks being mistaken for the original in speech and in filenames. |
| **B — numbered restart series** | `Phase 14-R1`, `Phase 14-R2`, … | Same lineage, but every restart is individually addressable. Slightly more verbose. |
| **C — new run-series identifier** | `RUN-2026-A`, `RUN-2026-B`, … decoupled from phase numbers entirely | Cleanest break. Phase numbers go back to meaning *work packages*; run ids mean *training runs*. Costs one extra mapping table between runs and the phases that produced them. |

**Recommendation, not a decision: Option C**, because the project has already
demonstrated twice that phase numbers drift from the work they name (see
[`PHASE_HISTORY.md`](PHASE_HISTORY.md) §1), and a run identifier that never has
to be renamed is the thing this history most lacked. **The final naming decision
is explicitly left open for the project owner.**

Whichever is chosen, the convention should be recorded here and in
[`STATUS.md`](STATUS.md) before the first new run starts, and every artifact of
that run must carry the identifier in its `artifact` field.

---

## 7. Standing preconditions before any new long run

Not authorization — a checklist of what is currently unmet.

1. **Phase 14 is formally closed** (blocked until 2026-08-28T16:15:34.689Z) and
   the repository freeze is lifted.
2. **The Phase 15–16 work is committed to version control**, so the new run has
   a real baseline to diff against.
3. **A run identifier is chosen** (§6).
4. **The nine fields (§2) are registered** in a frozen artifact before the first
   optimizer step.
5. **The evaluation instrument is named in advance**, and its standard error is
   computed **before** the decision margin is set (§3.3).
6. **The starting checkpoint has headroom**, or the experiment is explicitly
   framed as a curve-shape question rather than a ranking (§3.4).
7. **Someone has written down what result would falsify the hypothesis.**

As of 2026-08-27, items 1–7 are all unmet, and **no new long training run is
authorized** ([`STATUS.md`](STATUS.md) §11).
