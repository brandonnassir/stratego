# P18-A002 — The persisted-receipt reader: three aliases, reconstructed on load

**Gate G3. Frozen before the result was computed.**

## The defect

`stage_analyse` is the only caller that reads evaluation rows back from disk. It does so
through `read_receipt_rows`, which rebuilds each row as `SimpleNamespace(**receipt_json)`.
Three of the eleven `ARM_INVARIANT_FIELDS` are not persisted under those names:

| invariant field | persisted as |
|---|---|
| `setup_pair_id` | `case_index` |
| `opponent_policy_id` | left of `@` in `opponent_policy` |
| `opponent_policy_version` | right of `@` in `opponent_policy` |

`prove_arm_identity` raises `AttributeError` on `reference.setup_pair_id` before computing
anything; `case_scores` reads the same missing field. The G3 tests never caught it because
they feed the in-memory rows from `evaluate_bundle` straight into the analysis and never
round-trip through `read_receipt_rows`.

This is a checker/reader defect. The receipts, the games, the scores and the training are
unaffected.

## The authorised repair

`read_receipt_rows` reconstructs exactly three aliases while loading each immutable receipt:

- `setup_pair_id := case_index`
- `opponent_policy` split at its **final** `@`: left becomes `opponent_policy_id`,
  right becomes `opponent_policy_version`

`setup_pair_id` and `case_index` are identical by construction: `write_receipts` resolves
`by_case[int(row.setup_pair_id)]` and writes that case's `case_index`.

The reconstructed `opponent_policy_id` **must** equal the separately persisted
`opponent_id`. A disagreement is rejected, never repaired.

### Rejected, not repaired

- a receipt missing `case_index`, `opponent_policy` or `opponent_id`
- an `opponent_policy` with no `@`, or an empty id or version side
- a receipt already carrying an alias name with a conflicting value
- a reconstructed `opponent_policy_id` that differs from `opponent_id`

## Explicitly forbidden

Receipt contents. Rewriting existing receipt files. `write_receipts`. Evaluation cases and
the frozen schedule. Scores and outcome handling. `ARM_INVARIANT_FIELDS`.
`prove_arm_identity`. `case_scores`. The bootstrap calculation. The success rule and the
near-boundary rule. Rerunning training or evaluation, replacing a checkpoint, or running a
diagnostic arm or a second seed.

## Binding

Every stage after `--launch-manifest` re-verifies HEAD and each tracked harness source and
test digest, so this repair necessarily drifts from `8c1baa8`. The original launch manifest
is **preserved** and continues to identify `8c1baa8` as the source that created the training
and evaluation data. A separate, narrowly scoped rebind record relaxes the source-commit
equality check for `--analyse` alone, binds every evidence hash, and authorises no other
stage.

## Ordering

This amendment, the reader repair, the regression tests and the rebind support were
committed and pushed **before** the primary analysis was executed. The repair cannot have
been chosen with knowledge of the outcome.
