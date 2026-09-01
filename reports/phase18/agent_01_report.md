# Phase 18 — Agent 1

## Reproduction boundary, setup-method parity map, and evaluation contract

**2026-08-31.** Read-only work package. No optimizer step was taken, no training
or control run was started, no repository file was committed, reset, cleaned or
deleted, and no accepted Phase 8 or Phase 17 artifact was edited.

**Recommendation: `PROCEED`.** Gate G0 passes on every sub-gate. Two data
dependencies block *later* gates and neither blocks the Phase 8 control.

---

## 1. Findings

### 1.1 Phase 8 is reproducible, and the strongest evidence is bit-exact

All twelve frozen Phase 8 identities were **independently recomputed** rather
than copied from the instruction. All twelve match.

The decisive one: the canonical fresh C1 initialization rebuilds **bit-exactly**
from its seed. `build_candidate_model('C1', seed=2026081302)` produces state
checksum `cfe60bb0…e042b8`, the accepted value, twice in a row. Parameter count
863,959, matching `EXPECTED_C1_PARAMETERS`. The control's starting point is
therefore reconstructible, which is the precondition for G1.

The corpus verifies at **payload** level, not merely at index level. This
mattered: `corpus_content_digest` hashes the commit journal's *recorded* per-game
digests, so on its own it cannot detect payload corruption. I re-read and
re-hashed all 28,000 committed games against their commit records —
**0 trajectory mismatches, 0 metadata mismatches**.

The corpus has been relocated since its manifest was written (the manifest still
names `/Users/brandonwashington/Dev/stratego_phase8/…`). The accepted identity
rule binds corpus by version and digest, never by path, and all four digests
recompute at the current location, so the relocation is benign.

### 1.2 One original Phase 8 gate is fragile, and I flagged it before the run

`random_effective_win_rate_at_least_0_950` is the tightest of the 42 gates. The
accepted result **0.956787 clears its own 0.95 gate by 0.006787 EWR, which is
1.511 binomial SE at n = 2,048.** A faithful reproduction that is genuinely equal
in strength still fails that point-estimate gate roughly **6.5%** of the time.

The reproduction contract therefore predeclares the interpretation *now*, before
any control result exists: failing only this gate, while the paired bootstrap
lower bound still clears 0.90 and the paired delta stays inside margin, is
recorded as a **reproduction tolerance event** and sets G1 to `REVISE`. It is not
recorded as a pass, and this interpretation may not be widened afterwards. By
contrast the vs-initialization gate is robust — it clears by 25.2 SE.

### 1.3 The published implementation materially changes the setup method

The authors' source is available at exactly the pinned commit
`92db29e8…` (remote HEAD equals the pin). I read it and built a 35-row map:
**22 `exact`, 6 `corrected`, 2 `scaled`, 4 `intentional integration divergence`,
1 `not used`.** Three findings change Phase 18's design.

**(a) The paper is internally inconsistent about `h`, and the code settles it.**
Table 19 glosses `h_θt(σ)` as a *predicted conditional entropy* — nats — while
Eq. (1) trains that same symbol toward `H/10`. The printed advantage `α(H − h)`
cannot be implemented from the paper alone without choosing a reading. The
published buffer chooses: `ents = reg_norm * self.ents` with `reg_norm = 10`,
carrying the in-source comment *"Multiplying by reg_norm gives network entropy
prediction"*. Phase 18 uses **`I − 10h`**.

This is not cosmetic. Once `h` converges to `I/10`:

```text
I − 10h  →  0          a centered, mean-zero innovation
I −   h  →  0.9 · I    an uncentered positive bonus proportional to I
```

Phase 17 shipped the second form. Its own module docstring records the
consequence: the entropy term was weighted 2.70 against 1.00 for the outcome
term. Phase 17's reading was defensible from the printed formula; it is now
contradicted by the authors' code and must not be relabelled a faithful
transcription.

**(b) Forced handedness and post-generation reflection appear nowhere in the
paper.** They are code-only (`force_handedness=True` by default, masking the flag
out of the left five columns of every row; then `torch.rand(n) > 0.5` reflection
after generation), and they are inseparable — forcing handedness without
reflecting afterwards would remove one orientation from actual gameplay. Phase 17
could not have known this from the paper.

**(c) The published setup buffer is a TD(λ)/GAE(λ) recursion, not the flat
per-prefix form.** At the shipped defaults `arr_td_lambda = arr_gae_lambda = 1.0`
it telescopes *exactly* to the paper's printed flat form — the outcome channel to
`z̄ − V_k`, the entropy channel to `I_k − 10h_k`. So the Phase 18 contract's math
is correct, but it is correct **as a λ = 1 specialization**, and it silently stops
being correct if anyone sets λ to anything else. The map records this and requires
a guard.

Two further semantics are load-bearing and appear in **neither** the paper nor the
Phase 18 common contract:

- **De-duplication (S10).** Setup identity is a canonical arrangement id, and
  duplicates collapse to the instance with the newest behavior snapshot. Without
  it the same board trains twice with different behavior log-probs and split
  outcome evidence.
- **Window reset (S23).** `counts`, `rewards` and `ready_flags` are zeroed at
  every pool refresh, so outcome aggregation spans exactly one collection period
  and never accumulates across periods.

Smaller items worth carrying: the published piece head is **14-way** (lake and
empty carry count 0 and are permanently masked), not 12; `arr_batch_size = 1024`
is the **optimizer minibatch**, not an accumulated effective batch, so Phase 17's
32-episode minibatches took ~32× more optimizer steps per unit of data; the pool
is **1,024 total / 512 per player** in code against the paper's *1,000 per
player*; `AdamW(weight_decay=0)` is exactly Adam here, so that method question is
**resolved, not open**; and the published loop has SIGUSR1 handling that the
frozen Phase 17 closure lacked entirely.

### 1.4 The scaling arithmetic checks out, from code

Counting the published `ArrangementTransformer` at its defaults (depth 4, embed
512 = 8·8·8, 8 heads, ff 2,048) gives **12,647,954** trainable parameters —
Table 23's "12.6 million" exactly. The move network's shape matches Table 24
likewise. So the numerator of the scaling ratio is now *verified* rather than
paper-stated; the denominator (14.7M) stays paper-stated, because the move
network needs the compiled CUDA extension to instantiate.

```text
proportional target = 12.6M × 0.863959M / 14.7M = 0.7405M
Phase 18 default    = 802,320 (verified in-process, tolerance 0)
excess              = +8.35%
```

The 4/128/4/512 default stands. No architecture sweep is authorized.

### 1.5 The evaluation can separate Claim A, Claim B and the combined system — but not at 120 games

The four-lane factorial does isolate the claims cleanly, and Phase 17 supplies a
directly reusable instrument: the paired per-board `joint − move_only` difference
holds move weights fixed and varies only the player's own setup, which is exactly
the shape of `C0-L − C0-F` and `T-L − T-F`.

Measuring it produced the single most decision-relevant number in this task.
**Pairing buys far less than it looks like it should.** The within-candidate
correlation between the two lanes on the same board is only **0.238**, so the
paired difference SD (**0.5391**) is *larger* than either lane's own SD
(0.42–0.45). Pairing reduces variance by about 24% against fully unpaired — not
by the order of magnitude that near-perfect pairing would give.

| n games | Minimum detectable effect at 80% power |
|---:|---:|
| 12 (a Phase 17 per-opponent stratum) | 0.4360 |
| **120 (a Phase 17 lane)** | **0.1379** |
| 480 | 0.0689 |
| **913** | **0.0500** |
| 2,521 | 0.0300 |

A deterministic bootstrap over the empirical paired-difference pool reproduces
these (type-I measured at 0.048).

**This reframes Phase 17.** Its 120-board lane could not have resolved anything
below ≈0.14 EWR. Its "flat" joint lane (t = 0.04) and its "0 of 24 beat hour 0"
reading are both consistent with any true effect inside roughly ±0.13. It also
independently reproduces the 0.1435 EWR pure-noise spread measured for 25
candidates on that lane size. And Phase 17's reported worst-stratum figures — for
example 0.125 on `phase9_anchor` over 12 games — carry no information at all.

Phase 18's predeclared practical margin is therefore **0.05 EWR**, needing
**≈913 (operationally 1,000) paired games per contrast**, about 8× a Phase 17
lane. A stratum may only be called a regression with ≥200 paired games.

### 1.6 The `unusual_procedural` pack cannot be built from anything that exists

This is the one genuinely blocking discovery, and it blocks Claim B.

- `setup_library_v1` has exactly 16 primary families F00–F15, 500 bases each,
  split 400 train / 50 validation / 50 test.
- The `neutral_v1` sampler profile selects **uniformly over all 16 families**.
- The accepted Phase 8 corpus draws the library **train** split for its train
  corpus, the **validation** split for its validation corpus, and the **test**
  split for its test corpus.

So the **entire 8,000-board library is consumed** by the accepted corpus, and no
library family is unseen. The Phase 16 `targeted_family` source does not rescue
it either — it is documented as *"accepted library bases, family-targeted"*, i.e.
inside the same 16 families. The strongest held-out condition constructible from
existing assets is *familiar family, unseen base*, which is weaker than Claim B
requires.

`unusual_procedural` therefore needs **newly generated families** with a
family-level disjointness proof. `operator_sealed` needs operator-supplied
setups; the evaluation contract already specifies the opaque-manifest form so
that no agent ever needs to read them.

### 1.7 The Phase 17 evaluator retry defect, reproduced

I did not take this on report. I executed it in a scratch directory:
`refusal_receipt()` writes `status: "refused"` with no `bundle_digest` to
`<candidate_id>.result.json`; `existing_result()` finds that file on any later
attempt; the identity check compares `None` against the real manifest digest and
raises *duplicate-conflicting candidate refused* — **permanently**. Nothing in
the repository was modified. It is a required regression case (requirement R9).

---

## 2. Every material Phase 17 → published-method correction

| # | Correction | Phase 17 | Phase 18 | Map row |
|---|---|---|---|---|
| 1 | Entropy units in the advantage | `I − h` | `I − 10h` | S13 |
| 2 | Flag handedness | none | forced to one half during generation | S04 |
| 3 | Post-generation reflection | none | seeded 50% horizontal reflection | S05 |
| 4 | Setups per outcome | 1 fresh setup per game, m = 1 | reusable 1,024 pool, all outcomes averaged per setup | S09, S20 |
| 5 | Effective batch | 32 episodes | 1,024 episodes | S26 |
| 6 | Experimental point | Phase 9 weights, self-play, belief **disabled** | Phase 8 warmstart, supervised, belief **enabled** | S31, S34 |
| 7 | Signal handling | none — SIGTERM killed instantly | cooperative termination at an iteration boundary | S35 |

Corrections 1–5 are the five mandated by common contract §3.1. Each has a
required reduction or parity test named in the map; documentation alone is not a
gate.

---

## 3. Unresolved items and why they matter

| # | Item | Blocks | Does not block | Why it matters |
|---|---|---|---|---|
| U1 | `unusual_procedural` pack does not exist and cannot be built from existing assets | **G4**, Claim B | G1, G2, G3 | Without genuinely unfamiliar families there is no instrument for "does the setup curriculum generalize", and a held-out base of a trained family would silently substitute a weaker claim |
| U2 | `operator_sealed` pack needs operator-supplied setups | **G6** | G1–G4 | Final acceptance has no sealed arm without it |
| U3 | Source closure is identified but not immutable | any run that must bind a closure | Agent 1 | 8 modified/deleted tracked files and 29 untracked paths, all Phase 17 evidence or the Phase 18 package. Recorded and hashed; needs an operator commit-on-branch or a signed dirty-list manifest |
| U4 | Paper's 14.7M move-network count is paper-stated, not verified | nothing | everything | The scaling numerator is verified from code; the denominator needs the compiled CUDA extension. Low materiality — the target moves by <1% under any plausible correction |
| U5 | Canonical/live stream mixture and setup-update cadence undetermined | G4 design | G1–G3 | Derivable from neither the paper nor the code — they have no supervised anchor stream. Must come from a bounded pilot and may never be selected on a full run (map row S33) |
| U6 | No replicate-run variance estimate for the Phase 8 control | interpretation of a marginal G1 | G1 itself | The non-inferiority margins are predeclared judgments, not estimates from a measured run-to-run distribution. The contract requires one replicate before declaring a reproduction failure |

---

## 4. Recommended next experiment

**Run the Phase 8 control (Gate G1).** It is the only bounded question whose
inputs are fully verified, it costs about **1.08 wall hours** on MPS by the
accepted run's own telemetry, and it gates everything else by contract §12.

It also discriminates: if the control reproduces, every later Phase 18 negative
result is attributable to the setup integration rather than to a broken baseline
— which is precisely the ambiguity Phase 17 could not resolve about itself.

The setup parity build (G2) is *also* unblocked and independent of G1, so the
operator may reasonably authorize them in either order or together. I recommend
G1 first because contract §12 makes a failed G1 stop setup work outright, and
because it is one hour.

I have not written either instruction. Per §11 that is not Agent 1's task.

---

## 5. What I verified, by evidence path

| Claim | How | Result |
|---|---|---|
| 12 Phase 8 identities | recomputed from bytes / live source | all match |
| Fresh C1 init reconstructible | rebuilt from seed 2026081302, hashed twice | `cfe60bb0…` bit-exact |
| C1 parameter count | counted on the rebuilt model | 863,959 |
| Corpus integrity | all 28,000 games re-hashed at payload level | 0 mismatches |
| Published source availability | `git ls-remote` then clone at the pin | remote HEAD **equals** the pin |
| Paper setup model = 12.6M | counted the published defaults by hand | 12,647,954 |
| Phase 18 setup model = 802,320 | instantiated `Phase17SetupModel` | 802,320 |
| Power / MDE | Phase 17's own per-case rows, analytic + bootstrap | MDE 0.1379 at n=120 |
| Evaluator retry defect | executed a reproduction in scratch | confirmed |
| Test suite baseline | full run before any edit | **7,480 passed, 3 skipped**, 653 s |

---

## 6. Deliverables

```text
reports/phase18/phase18_process_boundary_v1.json
reports/phase18/phase18_phase8_reproduction_contract_v1.json
reports/phase18/ataraxos_setup_method_map_v2.md
reports/phase18/ataraxos_setup_method_map_v2.json
reports/phase18/phase18_evaluation_contract_v1.json
reports/phase18/agent_01_report.md
reports/phase18/phase18_agent1_handoff_v1.json
reports/phase18/decisions/P18-D001.md
reports/phase18/decisions/P18-D001.json
reports/phase18/decision_index.json
reports/phase18/agent_instruction_index.json
```

Documentation updated after verifying the underlying artifacts:
`stratego_project_docs/STATUS.md` (§13, §14),
`EVIDENCE_INDEX.md` (§7, §6.1),
`PHASE_HISTORY.md` (§13, §14, numbering table),
`05_project_plan.md` and `README.md` (pointers only).

Historical Phase 17 instruction amendments were **preserved unedited**, as
required; the paper-only method map v1 is marked `SUPERSEDED`, not rewritten.
