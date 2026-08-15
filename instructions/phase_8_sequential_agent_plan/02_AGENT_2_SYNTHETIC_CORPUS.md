# Phase 8 Agent 2 — Deterministic Synthetic Rule-Agent Corpus

## Role

You are **Agent 2**.

Build and finalize the static synthetic game corpus. Do not train C1.

Agent 1 must be PASS.

## Required reading

Read:

```text
00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_8_implementation_report.md
reports/phase_8_data/agent_01_*.json

reports/phase_7_implementation_report.md
reports/phase_4_implementation_report.md

stratego/evaluation/
stratego/training/
stratego/setups/
stratego/replay or trajectory modules
```

Verify every Agent 1 contract digest against live code.

Run the full suite before edits.

## Mission

Generate exactly the frozen:

```text
synthetic_warmstart_corpus_v1
```

using the accepted rule agents and Phase 7 setup sampler.

Expected final counts:

```text
100 ordered policy pairs

train         20,000 games
validation     4,000 games
test           4,000 games
total         28,000 games
```

No neural model makes moves in this corpus.

## Rule-agent behavior

For each game:

1. build the deterministic Phase 7 setup pair from the split-specific setup source;
2. instantiate the exact frozen Red and Blue rule policies;
3. derive their independent action/tie-break seeds from the game identity;
4. run the frozen engine to termination;
5. persist `trajectory_v1`;
6. persist synthetic/setup metadata;
7. commit the game only after both payloads verify.

Do not alter a rule policy to improve corpus outcomes.

## Corpus storage

Preferred:

```text
data/warmstart/synthetic_warmstart_corpus_v1/
    manifest.json
    train/
    validation/
    test/
```

Within each split, use compressed trajectory shards and metadata/journal files consistent with existing repository conventions.

Large data may live on the external volume via a configured root.

The manifest in the repository/report must preserve enough information to reproduce it from scratch.

## Required per-game metadata

At minimum:

```text
corpus_version
corpus_split
synthetic_game_id
ordered_matchup_id
matchup_ordinal

red_policy_id
red_policy_version
red_policy_seed
red_policy_weight

blue_policy_id
blue_policy_version
blue_policy_seed
blue_policy_weight

root/setup seed identity
red setup provenance
blue setup provenance

trajectory record identity
final result
terminal reason
ply/decision count

commit status / content digest
```

The trajectory remains the game-history authority.

## Crash-safe commit design

This is a hard Phase 8 requirement.

A game must not become visible to the training dataset until:

```text
trajectory record exists and verifies
metadata record exists and verifies
commit record exists
```

Use an append-only commit journal, finalized index, transactional store, or an equivalent auditable design.

A crash may leave:

```text
trajectory without metadata
metadata without trajectory
both without commit
```

but the dataset must expose none of them.

On restart:

```text
reconcile all three identity sets
continue from missing logical game ids
never regenerate/duplicate a committed id
```

Finalization must produce zero orphans.

## Required injected-crash tests

Simulate interruption after at least:

1. game finished but before trajectory write;
2. trajectory write before metadata;
3. metadata write before commit;
4. commit flush boundary;
5. shard rollover;
6. process restart.

After resume require:

```text
same final corpus identities
same game contents
no duplicate committed ids
no missing scheduled games
no trainable orphan records
```

A clean uninterrupted mini-corpus and a crash/resumed mini-corpus from the same seed should finalize to the same logical corpus digest.

## Determinism

Prove at minimum:

```text
same game id -> same setup pair
same game id -> same rule-agent action sequence
same game id -> same result

worker count change        0 logical-game differences
enumeration order change   0 logical-game differences
resume boundary change     0 logical-game differences
isolated game rebuild      exact
```

A corpus content digest should be based on stable game ids + trajectory/metadata fingerprints, not on incidental shard filenames.

## Split isolation

Hard requirements:

```text
train game setups       train split only
validation setups       validation split only
test setups             test split only

base-id overlap train/val     0 by Phase 7 contract
base-id overlap train/test    0
base-id overlap val/test      0
```

Also verify no synthetic `game_id` occurs in two splits.

Do not infer split from output directory alone; validate setup provenance.

## Ordered matchup audit

Generate an exact CSV with one row per ordered policy pair per split.

Require:

```text
train        200 each
validation    40 each
test          40 each
```

No cell may borrow games from another cell.

Report game lengths, terminal reasons, W/D/L, and color balance as corpus diagnostics only.

Do not alter the schedule based on outcomes.

## Trajectory correctness

For every game, validate structural decoding.

For a substantial independent sample, reconstruct full games through the frozen replay path.

Recommended:

```text
at least 2,000 complete games across all splits/pairs
at least 500,000 total reconstructed decisions
0 board/observation/legal/result mismatches
```

If all 28,000 can be replayed affordably, do so.

## Setup provenance

Rebuild both setups from Phase 7 provenance for every persisted game if practical; otherwise every game should at least pass fingerprint equality and a large sample should run full provenance rebuild.

Preferred hard evidence:

```text
28,000 / 28,000 setup fingerprints agree
>=2,000 games full setup rebuild
0 mismatches
```

## Performance/storage

Measure:

```text
games/s by split and overall
decisions/s
compression ratio
bytes/game
bytes/decision
total corpus bytes
generation peak RSS
worker count
CPU utilization
setup-source overhead
```

No MPS is required for rule-agent corpus generation.

Do not optimize prematurely unless generation is unexpectedly slow.

## Corpus finalization

The final manifest must include:

```text
corpus version
contract version
policy roster digest
setup library/sampler versions + digest
corpus master seed
split seeds
game counts
ordered matchup counts
trajectory schema
snapshot interval
compression settings
content digest
metadata digest
commit-index digest
storage path
generation commands
```

After finalization, treat corpus bytes as immutable.

## Suggested files

```text
stratego/training/synthetic_corpus.py
stratego/training/corpus_commit.py
stratego/training/rule_population.py

tests/training/test_synthetic_corpus.py
tests/training/test_corpus_resume.py

scripts/run_phase8_agent02.py
```

Reuse existing trajectory/engine/policy code.

## Required artifacts

Create:

```text
reports/phase_8_data/agent_02_corpus_manifest.json
reports/phase_8_data/agent_02_corpus_audit.json
reports/phase_8_data/agent_02_matchup_counts.csv
```

Append report section 2 only.

## STOP / FAIL rules

`BLOCKED` if:

- Agent 1 is not PASS;
- rule policy behavior cannot be deterministically seeded as contracted;
- setup split access cannot be enforced;
- corpus persistence cannot reconcile crashes without changing trajectory semantics.

`FAIL` if the generated corpus does not meet its exact frozen schedule/correctness gates.

Do not generate a different seed because one outcome distribution looks undesirable.

## PASS gates

PASS only if:

- exactly 28,000 scheduled games;
- exact 20k/4k/4k split counts;
- exact 200/40/40 games in all 100 ordered cells;
- 0 illegal actions;
- 0 setup split violations;
- 0 setup fingerprint mismatches;
- 0 duplicate game ids;
- 0 committed orphans/missing payloads;
- crash/resume mini-corpus converges to same logical digest;
- worker/enumeration independence exact;
- replay audit clean;
- corpus manifest/digests written;
- no neural model generated corpus actions;
- Phase 4 policies unchanged;
- Phase 7 library/sampler unchanged;
- full suite green.

## Handoff to Agent 3

Provide:

```text
corpus storage root
manifest/digests
game index
metadata reader
trajectory reader
commit-index reader
policy roster/weights
split access APIs
decision-sampler contract
rebuild-game API
```
