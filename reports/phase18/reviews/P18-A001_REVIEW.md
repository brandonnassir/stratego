# Review record of P18-A001 — evaluation rule identity

## Decision

**`P18-A001` is accepted and is OPERATIVE for future G3/G4 play evaluation.** On
2026-09-02 the operator approved the corrective Stage 3 commit
`959461fd3ed48cadbc1fb8531fa01a09a8c6c557` and accepted the amended
recommendation. The open item `O-P18-EVALRULES-1` is **resolved** by this
acceptance.

Future Gate G3 and Gate G4 play evaluation uses `EVALUATION_RULES`:
battleless move limit **200**, absolute move limit **4000**, first player **0**,
context **`evaluation`**.

## The canonical evaluation-rules token

```text
stratego_project_v1|board_10x10_v1|first_player=0|battleless_move_limit=200|
absolute_move_limit=4000|two_square_rule_enabled=0|continuous_chasing_rule_enabled=0|
context=evaluation
```

This is the token every G1 receipt already carries — all 16,384 rows, with no
competing token anywhere in `reports/phase18/` — and the one the accepted
`MatchSpec` machinery defaults to.

## What is superseded, and what is not

**Superseded — exactly one field, in one file:**

```text
file   reports/phase18/phase18_evaluation_contract_v1.json
field  cases_seeds_opponents.rule_version
from   "the accepted training rules: battleless_move_limit 100,
        absolute_move_limit 4000, first_player 0"
to     the accepted play-evaluation rules (EVALUATION_RULES): battleless_move_limit
        200, absolute_move_limit 4000, first_player 0, context evaluation
```

No other field of the evaluation contract, and no other artifact, is superseded.

**The original frozen evaluation contract remains unchanged.** It is **not**
edited by this acceptance. Its digest is
`6878bab622c47bbbc21f0d18f88844a5e5a1c2f298acfd3b1e9560705d0ecce9`, unchanged
throughout, and it still verifies against the two places that pin it —
`reports/phase18/decisions/P18-D001.json` and
`reports/phase18/phase18_agent1_handoff_v1.json`. The superseding text lives in
`P18-A001` and in this review; the contract file keeps its bytes, exactly as the
project handles every frozen artifact.

**Training and corpus rules are unchanged.** `TRAINING_RULES`, aliased
`CORPUS_RULES` in `stratego/training/warmstart_contract.py:351`, keeps
`battleless_move_limit = 100`. Corpus generation and training play
(`rule_population.py`, `phase9_collector.py`, `synthetic_corpus.py`) are
unaffected. The project invariant is now recorded explicitly: **train under 100,
evaluate play under 200.**

**Prior results are unchanged.** G1 (`P18-D002`, `P18-D003`) was measured under
battleless 200 and needs no rerun — the amendment records the rule G1 already
used. G2 (`P18-D004`, `P18-D005`) plays no Stratego games and opens no sealed
data. **Gate G1 and Gate G2 both remain closed on their existing evidence**, and
nothing in this acceptance revisits either.

## Scope of this acceptance

**This does not authorize G3 execution.** It settles a rule identity and nothing
else. It designs no gate: no lanes, packs, opponents, seeds, sample sizes,
margins, power plan or decision rule are approved or implied. **Gate G3 remains
unstarted**, and a G3 design instruction with its own decision packet is still
required before any game is played. No Stratego game was played, no model
trained, and no sealed evidence opened in producing this record; the
`unusual_procedural` pack remains unpopulated and still blocks G4 independently.

## What the accepted recommendation does and does not claim

The recommendation rests on three grounds: **comparability** with the closed G1
baseline, the **established project-wide evaluation invariant** (every accepted
play gate since Phase 4, and the `MatchSpec` default), and **avoiding additional
early censoring** of continuations.

It explicitly does **not** rest on a power argument. The amendment's statistical
claims were corrected before acceptance, and the accepted text records:

- reducing the limit from 200 to 100 censors continuations earlier, so the count
  of games terminated as battleless draws **weakly increases or is preserved** and
  cannot decrease;
- **draws are not signal-free** — under the accepted `EWR = (W + 0.5·D) / N`
  (`stratego/evaluation/statistics.py:11`, `98-107`) a draw scores 0.5 and
  contributes to paired effective-win-rate differences, as every `draw = true` row
  in the G1 receipts confirms with `candidate_score = 0.5`;
- a higher draw share does **not** necessarily reduce statistical power;
- **no power advantage for 200 over 100 has been demonstrated**, and the effect of
  the rule change on variance, effect size and power is **unknown** without a
  predeclared comparison.

Any future document citing `P18-A001` must carry this limit with it.

## Process deviation and its forward correction

Recorded in full in
`reports/phase18/process_notes/P18-PN001_STAGE3_PREMATURE_PUBLICATION.md`
(SHA-256 `9c4969e7b09b4efbc659e7e7a4c2495e3d81536341e09b051779ad38f371c23c`):

```text
cac7abf   Stage 2, P18-D005 review          published only after approval    CORRECT
69c97db   Stage 3, P18-A001 proposal        PUSHED BEFORE REVIEW             DEVIATION
959461f   corrective commit                 held local until approved        CORRECT
```

Commit `69c97dbad98718269e875545456b8e2cd208afc2` was pushed to
`origin/phase18/g3-backend-foundation` before the operator had reviewed Stage 3,
contrary to the required hard stop. The steward acted on an ambiguous one-word
approval that followed a multi-stage report instead of confirming which stage it
applied to.

The deviation authorized nothing: `P18-A001` carried the status
`PROPOSED — NOT OPERATIVE` throughout, and `O-P18-EVALRULES-1` stayed open, so
publishing the proposal did not adopt it, changed no active rule, and started no
experiment.

**History was not rewritten.** No amend, reset, rebase, force push or branch
deletion; `69c97db` is preserved exactly as published and remains an ancestor of
the branch tip. The correction was made forward, in commit `959461f`, which
withdrew the overstated statistical claims, clarified one digest's provenance, and
added the process note. This is the same convention the project already uses for
errata: the record is corrected forward, never overwritten.

## Approved hashes

```text
branch            phase18/g3-backend-foundation
remote            origin  https://github.com/brandonnassir/stratego.git

approved commit   959461fd3ed48cadbc1fb8531fa01a09a8c6c557
local SHA         959461fd3ed48cadbc1fb8531fa01a09a8c6c557
remote SHA        959461fd3ed48cadbc1fb8531fa01a09a8c6c557
local == remote   true
push kind         normal non-force fast-forward from 69c97db; no history rewritten
published         2026-09-02T19:56Z  by the Phase 18 Backend Steward

branch lineage    d0fd36d  P18-D005 delivered      (published, approved)
                  cac7abf  P18-D005 review         (published, approved)
                  69c97db  P18-A001 proposed       (published prematurely, preserved)
                  959461f  P18-A001 corrected      (published, approved)  <- accepted
```

The remote SHA was confirmed server-side with `git ls-remote --heads origin` and
again through `refs/remotes/origin/…` after a fresh `git fetch`.

## Artifacts accepted, as delivered

The `P18-A001` files are **preserved as delivered** and are not rewritten. Their
own status line still reads `PROPOSED — NOT OPERATIVE`, which records their state
at delivery; acceptance is recorded here and in the indexes, the same convention
that leaves the `P18-D003`/`P18-D004`/`P18-D005` packets' internal status
untouched.

```text
reports/phase18/amendments/P18-A001_EVALUATION_RULE_IDENTITY.md
    b32a13bec88b017d6903c8ee5f6f58a6642c7576854bd03e12d4a271e22fa7d6
reports/phase18/amendments/P18-A001_EVALUATION_RULE_IDENTITY.json
    0787871b4a61e72bc44cd5b7d8914fe4a48440790ec295b6362a1a5fc7a1f633
reports/phase18/process_notes/P18-PN001_STAGE3_PREMATURE_PUBLICATION.md
    9c4969e7b09b4efbc659e7e7a4c2495e3d81536341e09b051779ad38f371c23c
reports/phase18/phase18_evaluation_contract_v1.json  (frozen, unedited)
    6878bab622c47bbbc21f0d18f88844a5e5a1c2f298acfd3b1e9560705d0ecce9
```

The errata registry `phase18_rule_identity_errata_v1.json` is updated by this
acceptance to mark `O-P18-EVALRULES-1` resolved; the historical proposal record
and the process-note reference inside it are preserved.
