# Reviewing-chat audit of P18-D003

## Decision

**Accept Agent 3's `PROCEED` decision. Gate G1 is closed. Gate G2 (setup
implementation parity and synthetic learning) is authorized as the next bounded
question; nothing beyond G2 is authorized.**

Agent 3 produced a complete, exactly reconciled powered confirmation under the
frozen rule, on an independent bank frozen before either arm ran:

```text
reference EWR (accepted checkpoint)    0.9559326171875
candidate EWR (G1 reproduction)        0.9622802734375
paired delta                          +0.00634765625
two-sided 95% paired bootstrap        [+0.00079345703125, +0.01190185546875]
frozen margin                         -0.010
decision rule                         lower endpoint strictly greater -> PASS
paired units                          4,096 (8,192 games per arm, 16,384 total)
failed / missing / retried games       0 / 0 / 0
illegal / non-finite / policy errors   0 / 0 / 0
sealed Phase 8 test examples opened    0
Phase 18 evaluator tests               86 / 86
```

Combined with P18-D002's 42/42 original Phase 8 gates and 7/8 certified paired
margins, the eighth margin (`vs_random_ewr`) is now certified with the same margin,
the same two-sided 95% paired-bootstrap rule, the same 10,000 replicates, and the
same two checkpoint byte-identities. G1 therefore closes on the frozen question and
on nothing else.

## Independent verification

The review independently reproduced the result and checked every identity:

- the local branch `phase18/g1-random-confirmation` resolves to
  `ef7523c1940650c0906d1927e64679e8328a663f`, whose tree contains `P18-D003.json`
  at SHA-256 `57281f6a5c460a9a524bd0538ed44050e2949dd12873401cc0a3f8521d6d8fdb` and
  `P18-D003.md` at `7d930357e2c324d7c9ff3e0622637e5014fdbdba0390e4c06156e95b24266ab7`;
- `G1_CONFIRM_SOURCE_COMMIT 9392c6ec1c948a7c5c91278616f669340f4a6445` and the
  result-evidence commit `c1833ad7a36fb0c98d6fada02326316d50d7f284` exist on that
  branch in the recorded order, and every result artifact repeats the source commit;
- the bank (`24b263d2…`) and schedule (`2111848b…`) digests re-derive from the
  `phase18_g1_random_confirmation_v1` namespace through `derive_stream_seed`;
- the per-row receipts (`84b479b5…` reference, `6a773792…` candidate) recompute both
  arm EWRs and every paired unit score exactly, reproducing the headline numbers
  above to full precision;
- the 4,096-pair result, the interval, and the strict-margin verdict were
  reproduced independently from the receipts;
- the accepted Phase 8 artifacts and both frozen checkpoints hash to their recorded
  identities;
- the protected `reports/phase13/phase14_launch_manifest_v1.json` modification is
  the only dirty path and appears in none of the branch's commits; and
- 86 of 86 Phase 18 evaluator tests pass.

## Reading

The instrument behaved as the frozen power calculation predicted: a half-width of
about 0.0056 at 4,096 pairs against the 0.010 margin, where the original 1,024-pair
bank resolved about 0.0116. P18-D002's single failing margin was a measurement-design
deficit, repaired by sample size alone; nothing about the models, the margin, or the
rule changed between the two measurements.

The review accepts the packet's own separation of readings: non-inferiority is the
supported inference; per-bank variation of a few thousandths between the two banks'
deltas is plausible and untested; and superiority of the reproduction over the
accepted checkpoint is explicitly **not** claimed, because that question was never
frozen and the original bank's near-zero delta sits against it.

## Erratum — rule identity in the packet narrative

`P18-D003.json` records, under `identity.rules`:

```text
accepted evaluation rules (battleless_move_limit 100, absolute_move_limit 4000, first_player 0)
```

The value `100` is a narrative metadata error. The measurement itself used the
accepted **evaluation** rules, whose battleless move limit is **200**, consistently
and verifiably:

```text
engine constant     stratego/engine/constants.py  EVALUATION_RULES.battleless_move_limit = 200
                    (TRAINING_RULES, the Phase 8 corpus rules, carry 100)
driver              scripts/phase18_g1_random_confirmation.py builds the schedule with
                    match_spec.EVALUATION_RULES
frozen contract     phase18_g1_random_confirmation_contract_v1.json schedule.rules =
                    ...|battleless_move_limit=200|...|context=evaluation
receipts            all 16,384 rows of reference_receipts.jsonl and
                    candidate_receipts.jsonl carry the identical token with
                    battleless_move_limit=200 and context=evaluation
```

The frozen G1 result packet is **not** rewritten. The correction lives in this review
and in `reports/phase18/phase18_rule_identity_errata_v1.json`, and G1 does **not**
need to be rerun: the rule actually applied is the one the contract froze and the one
the original Phase 8 play gates use, and it is the same in both arms.

A second, separate fact is recorded as an open item rather than an error:
`reports/phase18/phase18_evaluation_contract_v1.json` names *training* rules
(`battleless_move_limit 100`) as the rule version for the future factorial lanes,
while the G1 play measurements used the evaluation rules (200). That frozen file is
preserved unedited. **Before any real-game G3 or G4 evaluation, the
training-versus-evaluation rule choice for play evaluation must be settled by an
explicit, reviewed amendment.** It does not affect G2, which plays no Stratego games.

## Authorized next question

> Does the local scaled setup-policy implementation match the paper and the pinned
> published implementation at loss, gradient, sampling, aggregation, optimizer,
> checkpoint, and EMA semantics — and can it reliably learn a known synthetic
> setup-reward landscape from outcome-only feedback across three fresh seeds?

This is Gate G2, executed by
`instructions/phase_18_setup_integrated_warmstart/06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md`.
It authorizes only G2. Stratego setup training, Phase 8 trainer integration, G3, a
tandem pilot, and a production run remain unauthorized. No Stratego games or sealed
Phase 8 examples may be opened.

## GitHub publication

The approved branch head was published by the next authorized agent under §7 of the
decision-packet protocol, with a normal non-force push and no publication commit:

```text
remote        origin  https://github.com/brandonnassir/stratego.git
branch        phase18/g1-random-confirmation
local SHA     ef7523c1940650c0906d1927e64679e8328a663f
remote SHA    ef7523c1940650c0906d1927e64679e8328a663f
published     2026-09-02 (UTC), by Phase 18 Agent 4 before any G2 work
```

The remote branch was absent before the push; no divergent history existed and none
was overwritten. The G2 branch `phase18/g2-setup-parity` was created from exactly that
commit.
