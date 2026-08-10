# Compact Training Trajectory and Reconstruction Specification

## 1. Status

**Status: Accepted Phase 3 baseline.**

Schema identifiers:

- trajectory schema: `trajectory_v1`;
- wire format: `1`;
- magic: `STJ1`.

The trajectory system is training infrastructure. It does not redefine rules, observation semantics, or privileged replay authority.

---

## 2. Design principle

Do not store:

- full `127 x 10 x 10` observation tensor per decision;
- dense 10,000-entry policy probability vector per decision;
- privileged belief targets in model-facing trajectory data.

Store compact facts and reconstruct model inputs/targets through the frozen reference engine.

---

## 3. Game record

A game record contains enough information to reproduce and audit the game:

- game identifier;
- environment identifier;
- generation;
- rules version;
- observation version;
- red setup;
- blue setup;
- first player;
- ordered action sequence;
- terminal result;
- terminal reason;
- final ply;
- collection policy/checkpoint identifiers;
- setup-generator/family identifiers when available;
- periodic compact snapshots.

The authoritative game/replay semantics remain those in `09_public_event_and_replay_schema.md`.

---

## 4. Decision record

For each ply store:

- game identifier/reference;
- ply;
- acting player;
- selected action ID;
- ascending legal action IDs;
- one old/behavior-policy probability per legal action;
- win/draw/loss prediction;
- collection-policy version;
- nearest snapshot reference.

Validation requirements:

- legal IDs are unique and deterministically ordered;
- selected action belongs to legal IDs;
- probability row length matches legal IDs;
- probabilities are finite/nonnegative;
- probability sum is approximately 1;
- value prediction has three finite entries;
- decision belongs to the correct `(environment_id, generation)`.

---

## 5. Probability precision

Accepted Phase 3 baseline:

- old policy probabilities stored as `float32`.

This is the fidelity baseline.

A later training-storage optimization may benchmark `float16` or quantized probability storage, but must explicitly measure:

- policy-update impact;
- importance-ratio error;
- storage savings;
- numerical stability.

Do not silently change trajectory precision.

---

## 6. Snapshot design

Snapshots are compact engine snapshots, not stored observation tensors.

Phase 3 measured:

- interval 16;
- interval 32;
- interval 64.

Accepted initial default:

- **32 plies**.

Controlled benchmark:

| Interval | Raw B/game | Compressed B/game | Reconstruction positions/s | Mean replayed actions | p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 101,421 | 62,668 | 2,095 | 7.5 | 15 |
| **32** | **87,155** | **60,682** | **1,681** | **15.3** | **30** |
| 64 | 80,072 | 59,450 | 1,149 | 30.9 | 61 |

Interval 32 provides the accepted storage/reconstruction balance.

---

## 7. Reconstruction

To reconstruct a historical decision:

```text
game header
+ nearest earlier snapshot
+ subsequent actions
        |
        v
frozen reference state
        |
        +--> observation_v2_1_127ch
        +--> legal action list/mask
        +--> observer public knowledge
        +--> privileged belief target
```

The belief target is returned separately from the policy observation.

No observation-builder path may consume the privileged target.

---

## 8. Reconstruction acceptance evidence

Dedicated Agent 3 gate:

- 2,020 complete games;
- 1,000,162 historical decisions;
- zero state-fingerprint mismatches;
- zero acting-player mismatches;
- zero observation mismatches;
- zero legal-list mismatches;
- 99,978 dense-mask comparisons, zero mismatches;
- zero public-knowledge mismatches;
- zero belief-target mismatches;
- zero selected-action mismatches;
- zero environment/generation/game-identity mismatches.

Integrated Agent 5 gate:

- 11,251 stored decisions reconstructed;
- zero mismatches.

Two-hour soak:

- 411,818 sampled decision reconstructions;
- zero mismatches.

---

## 9. Worker-side collection

In production collection:

- worker owns `GameState`;
- coordinator owns model;
- coordinator writes action/policy/value decision fields;
- worker records the decision before applying the move;
- terminal game is sealed before reset;
- record may be encoded/compressed;
- bulk records may be consumed/discarded rather than permanently archived.

The coordinator should not receive a game object simply to build the trajectory.

---

## 10. Serialization baseline

The accepted Agent 3 implementation uses:

- compact varints;
- signed zigzag encoding where appropriate;
- little-endian `float32`;
- delta-encoded ascending integer sequences;
- string interning within records;
- zlib framing/compression.

Serialization must reject incompatible magic/version/schema rather than guess.

Snapshot codec assumptions are guarded: if the frozen snapshot field set changes, encoding must fail rather than silently omit/interpret fields.

---

## 11. Measured storage

### 11.1 Agent 3 controlled baseline

At 32-ply snapshots on the larger dedicated gate corpus:

- mean raw bytes/game: 93,003;
- mean compressed bytes/game: 64,692;
- mean raw decision bytes: about 154/decision.

### 11.2 Integrated representative-model collection

During the two-hour Agent 5 soak:

- encoded bytes/decision: 187.8;
- encoded bytes/game: 96,965;
- generated trajectory: 11.17 GiB in 2 hours;
- planning rate: approximately 5.59 GiB/hour.

The integrated records are larger because real model policy distributions are stored over the legal set.

---

## 12. Retention policy

Do not permanently retain every trajectory generated during the final 168-hour run.

At 5.59 GiB/hour:

\[
5.59 \times 168 \approx 939\ \text{GiB}.
\]

This nearly fills the nominal 1-terabyte external archive before other artifacts.

Required planning approach:

```text
self-play collection
      |
      v
rolling replay/training buffer
      |
      +--> training consumption
      |
      +--> selected archival
      |
      v
expiration/deletion of consumed bulk data
```

Archive preferentially:

- evaluation games;
- diagnostic/error games;
- unusual/high-value sampled games;
- representative training samples;
- checkpoints;
- manifests/metrics;
- enough data for run reproducibility/debugging.

---

## 13. Reconstruction throughput planning

Random-access reconstruction at interval 32 measured approximately:

- 1,681 positions/s/process.

This is a training-pipeline cost, not a reason to store full observations.

Future training can scale reconstruction by:

- multiple reconstruction workers;
- sequential iteration with state-copy avoidance where safe;
- filtering/sampling before expensive reconstruction when the algorithm allows it;
- caching within a bounded replay buffer.

The final training architecture should measure reconstruction demand alongside optimizer/model throughput.

---

## 14. Versioning

Any change to:

- required decision fields;
- probability precision semantics;
- snapshot encoding;
- legal-action ordering;
- reconstruction interpretation;
- wire format;

requires explicit trajectory/wire-format versioning.

Do not silently reinterpret existing stored records.
