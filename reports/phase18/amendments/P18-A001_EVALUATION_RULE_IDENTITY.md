# P18-A001 — Proposed amendment: the rule identity for future play evaluation

**Status: `PROPOSED — NOT OPERATIVE`.** This document resolves nothing by
existing. It becomes operative only if and when the reviewing chat and the
operator accept it and that acceptance is recorded in the decision trail. Until
then the frozen evaluation contract stands exactly as written.

**Resolves open item:** `O-P18-EVALRULES-1`
(`reports/phase18/phase18_rule_identity_errata_v1.json`, `open_items[0]`).

**Amends:** `reports/phase18/phase18_evaluation_contract_v1.json` — the single
field `cases_seeds_opponents.rule_version`, and nothing else.

**The original frozen contract is preserved unchanged and is not edited by this
proposal.** As with every frozen Phase 18 artifact, the amendment is recorded
alongside it; the contract file keeps its bytes and its digests stay valid.

## The three rule identities, kept distinct

The project defines two rule presets in `stratego/engine/constants.py`. They are
**not** interchangeable, and this amendment concerns only the second.

```text
                            TRAINING_RULES            EVALUATION_RULES
                            (alias CORPUS_RULES)
battleless_move_limit       100                       200
absolute_move_limit         4000                      4000
first_player                RED = 0                   RED = 0
context                     "training"                "evaluation"
rules_version               stratego_project_v1       stratego_project_v1
board_geometry_version      board_10x10_v1            board_10x10_v1
two_square_rule_enabled     False                     False
continuous_chasing_rule…    False                     False
```

The two presets differ in **exactly two fields**: the battleless move limit and
the context label. Every other field is identical, verified field by field
against `stratego/engine/constants.py` lines 273–306 (`RED = 0` at line 48).

1. **Training rule — battleless move limit `100`.** `TRAINING_RULES`, aliased as
   `CORPUS_RULES` in `stratego/training/warmstart_contract.py:351`, governs
   corpus generation and training play. It is used by
   `stratego/training/rule_population.py`,
   `stratego/training/phase9_collector.py` and
   `stratego/training/synthetic_corpus.py`. **This amendment does not
   change it, does not propose changing it, and no training behaviour anywhere
   in the project is affected by adopting this proposal.**

2. **Accepted play-evaluation rule used by G1 — battleless move limit `200`.**
   `EVALUATION_RULES` is what every G1 Stratego game was actually played under.

3. **Absolute move limit `4000` and first player `0`** are *common to both*
   presets. They are not in dispute, are not changed by this amendment, and are
   restated here only so that the amended field names a complete rule identity
   rather than a partial one.

## Evidence that `200` is the accepted play-evaluation rule

Every layer of the accepted G1 evidence agrees, and I verified each one directly
rather than relying on the earlier summary in the errata record.

**Engine constant.** `stratego/engine/constants.py`:
`EVALUATION_RULES = RulesConfig(battleless_move_limit=200, absolute_move_limit=4000, context="evaluation")`,
with `first_player` defaulting to `RED`, and `RED = 0`.

**Accepted match machinery.** `stratego/evaluation/match_spec.py` **defaults**
`rules` to `EVALUATION_RULES` at four separate sites (lines 146, 264, 313, 398).
Play evaluation under the evaluation rules is the accepted default of the match
layer, not a per-run choice.

**Frozen G1 contract.** `phase18_g1_random_confirmation_contract_v1.json`,
`schedule.rules`:

```text
stratego_project_v1|board_10x10_v1|first_player=0|battleless_move_limit=200|
absolute_move_limit=4000|two_square_rule_enabled=0|continuous_chasing_rule_enabled=0|
context=evaluation
```

**Driver.** `scripts/phase18_g1_random_confirmation.py` imports `EVALUATION_RULES`
(line 53), builds the paired schedule with `rules=EVALUATION_RULES` (line 281),
and stamps `rules_token(EVALUATION_RULES)` into every receipt (line 377).

**Receipts.** All **16,384** rows across
`reports/phase18/g1_random_confirmation/reference_receipts.jsonl` (8,192) and
`candidate_receipts.jsonl` (8,192) carry that one identical rules token. Counted
directly: 16,384 of 16,384 rows, zero rows carrying any other token, and
`first_player = 0` on all 16,384 rows. Across the whole of `reports/phase18/`
the token appears 16,386 times and **no competing token exists**.

**Project precedent.** The accepted play gates of Phase 9, 11B, 12 and 15 all
play under `EVALUATION_RULES`; `warmstart_contract.py:349` records that the gates
"keep their own frozen `EVALUATION_RULES`". Training corpora are generated under
`CORPUS_RULES`. The train-under-100 / evaluate-under-200 split is already the
project-wide invariant.

## What the observed games say about the choice

From the accepted G1 receipts and arm summaries, at battleless limit **200**:

```text
lane                                  n        battleless draws     share
G1 confirmation, reference (random)   8,192    128                  1.56%
G1 confirmation, candidate (random)   8,192     84                  1.03%
G1 control, random_gate  reference    2,048     23                  1.12%
G1 control, random_gate  candidate    2,048     15                  0.73%
G1 control, vs_init      reference    1,024    134                 13.09%
G1 control, vs_init      candidate    1,024    137                 13.38%
absolute_move_limit_draw, all lanes  16,384      0                  0.00%
ply length, all 16,384 games          min 9  median 300  p99 812  max 1,336
```

Two facts follow, and only these two are claimed:

- **The absolute limit of 4000 is a safety backstop, not an operating
  parameter.** It never fired once in 16,384 accepted games, and the longest game
  reached 1,336 plies — about a third of the cap. Keeping it at 4000 costs
  nothing and changes nothing.
- **The battleless limit is the binding rule, and it binds unevenly.** Against a
  random opponent it ends about 1% of games; between two comparable trained
  players (`vs_init`) it ends about **13%**. The lanes G3 and G4 would care about
  are contrasts between comparable players, which is the regime where this rule
  is roughly ten times more consequential than in the random lane.

*Directional, not quantified:* a shorter battleless limit can only convert games
that would have been decided after the cap into draws, so moving 200 → 100 can
only increase the draw share, and it would do so most in exactly the
comparable-player lanes where a setup effect must be detected. Draws carry no
signal for a win-rate contrast, so a higher draw share costs power. The size of
that loss is **not** estimated here: quantifying it would require playing games,
which is out of scope and not authorized.

## Recommendation — a single rule identity

**Adopt `EVALUATION_RULES` as the sole rule identity for all future G3 and G4
play evaluation**, written in full as the canonical token:

```text
stratego_project_v1|board_10x10_v1|first_player=0|battleless_move_limit=200|
absolute_move_limit=4000|two_square_rule_enabled=0|continuous_chasing_rule_enabled=0|
context=evaluation
```

Four reasons, in order of weight:

1. **Comparability with the accepted G1 baseline.** G1 is closed and its result
   is the reference point every later play measurement is read against. It was
   measured under battleless 200. Evaluating G3/G4 under 100 would make them
   non-comparable to the accepted baseline, for no gain.
2. **It is already the project-wide invariant.** Every accepted play gate since
   Phase 4 uses these rules, and the accepted `MatchSpec` defaults to them.
   Naming training rules for a play lane is the anomaly, not the rule.
3. **Measurement quality.** The battleless limit binds hardest in the
   comparable-player lanes that G3/G4 contrasts live in; the longer limit leaves
   more games decided and fewer signal-free draws.
4. **Separation of concerns.** Training rules govern corpus generation;
   evaluation rules govern measurement. Keeping one identity for each prevents a
   change on one side from silently altering the other.

The alternative — adopting battleless 100 for play evaluation to match the
training corpus — is **not** recommended: it breaks comparability with the closed
G1 result, departs from every accepted play gate in the project, and increases
the draw share in precisely the lanes where a setup effect must be detected.

## Exactly what this amendment would supersede

**Superseded field** — one field, in one file:

```text
file   reports/phase18/phase18_evaluation_contract_v1.json
field  cases_seeds_opponents.rule_version
from   "the accepted training rules: battleless_move_limit 100,
        absolute_move_limit 4000, first_player 0"
to     "the accepted play-evaluation rules (EVALUATION_RULES):
        battleless_move_limit 200, absolute_move_limit 4000, first_player 0,
        context evaluation; rules_version stratego_project_v1, geometry
        board_10x10_v1, two-square and continuous-chasing rules disabled.
        Training and corpus generation continue to use TRAINING_RULES /
        CORPUS_RULES with battleless_move_limit 100, unchanged."
```

No other field of the evaluation contract is amended. No other artifact is
amended. The contract file itself is **not edited**: if this amendment is
accepted, the superseding text lives here and in the decision trail, and the
frozen file keeps its bytes, exactly as the project handles every frozen artifact.

## Scope and limits

- **Training rules remain unchanged.** `TRAINING_RULES` / `CORPUS_RULES` keep
  `battleless_move_limit = 100`. No corpus, no Phase 8 warmstart, no training
  play is affected in any way.
- **This affects future play evaluation only** — the G3 and G4 factorial play
  lanes, if and when either is authorized. It has no retroactive effect.
- **Prior results are unchanged.** G1 (`P18-D002`, `P18-D003`) was measured under
  battleless 200 and needs no rerun; the amendment records the rule G1 already
  used. G2 (`P18-D004`, `P18-D005`) plays no Stratego games at all, opens no
  sealed data, and is untouched. **Gate G1 and Gate G2 both remain closed on
  their existing evidence.**
- **This does not authorize a G3 run.** It settles a rule identity and nothing
  else. It designs no gate: no lanes, packs, opponents, seeds, sample sizes,
  margins, power plan or decision rule are proposed or implied here. G3 remains
  unstarted, and a G3 design instruction and its own decision packet are still
  required before any game is played.
- **It becomes operative only after review and approval.** Until the reviewing
  chat and the operator accept it and that acceptance is recorded, the frozen
  contract's original wording stands and no measurement may cite this document as
  authority.
- **It opens no sealed evidence.** The `operator_sealed` pack is untouched and
  unread; the `unusual_procedural` pack remains unpopulated and still blocks G4
  independently of this amendment.
