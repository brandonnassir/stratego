# Project Stratego Ruleset

## Purpose

This file is the authoritative rules contract for the first version of the project game engine.

It deliberately differs from the full competitive rules described in the Ataraxos paper. Those deviations are explicit so they do not become accidental implementation errors.

### Current implementation status

As of Phase 2.1, this rules contract is implemented by the frozen Python reference engine:

- implementation: `phase2_1_reference_1.1.0`;
- rules version: `stratego_project_v1`;
- observation version: `observation_v2_1_127ch`;
- action encoding: fixed 10,000-entry source-destination space.

Phase 2.1 validation passed with zero unexplained rule, replay, observation, hidden-information, snapshot, or invariant mismatches. Future changes to game semantics require explicit versioning and differential comparison against the frozen reference engine.

---

## 1. Included rules

The engine will implement:

- 10 by 10 board;
- standard two lake regions;
- 92 occupiable squares;
- 40 pieces per player;
- standard piece inventory;
- private four-row setup;
- alternating turns;
- standard cardinal movement;
- Scout long-range movement;
- immovable Flag and Bomb pieces;
- standard rank combat;
- Spy-versus-Marshal exception;
- Miner-versus-Bomb exception;
- Flag capture victory;
- victory when the opponent has no legal move;
- draw if neither player can legally move;
- persistent public knowledge of identities revealed by combat;
- logically known Scout identity after an observable multi-square Scout move;
- configurable no-battle draw rule;
- configurable absolute move limit as an engineering safety mechanism.

---

## 2. Deliberate exclusions

### Two-square rule: EXCLUDED

The engine will **not** prohibit repeated back-and-forth movement across the same boundary.

### Continuous-chasing rule: EXCLUDED

The engine will **not** track or restrict repeated chase positions.

These are project simplifications chosen to reduce game-engine complexity and historical-state requirements.

### Consequence

Removing both rules creates more opportunity for loops and stalling. Therefore the no-battle draw rule is mandatory for self-play and human evaluation in this project.

---

## 3. No-battle draw rules

### Training

Default:

- **100 consecutive moves without combat -> draw**.

Rationale:

- the Ataraxos paper used 100 during training;
- it discourages unproductive self-play;
- it guarantees practical termination even without the two anti-repetition rules;
- it reduces the amount of training computation spent on stalled games.

### Automated evaluation

Default:

- **200 consecutive moves without combat -> draw**.

### Human browser play

Default:

- **200 consecutive moves without combat -> draw**.

The browser must visibly display the current no-battle counter.

---

## 4. Absolute move limit

Training will support a configurable absolute move limit.

Initial default:

- **4,000 total moves -> draw**.

This is an engineering safety limit, not a standard Stratego rule. The Ataraxos implementation used the same 4,000-move safeguard and reported that it almost never triggered when combined with its 100-move training rule.

Evaluation logs must record when this limit is responsible for a draw.

---

## 5. First player

- Red moves first.
- Training must balance which model/checkpoint/setup is assigned to red and blue.
- Human evaluation results must report color allocation.

---

## 6. Setup legality

A setup is legal only if:

1. all 40 setup squares on that player's side are occupied;
2. only that player's 40 setup squares are used;
3. every piece count exactly matches the official inventory;
4. no piece occupies a lake or non-setup square;
5. exactly one Flag and six Bombs are present;
6. the complete setup is hidden from the opponent at game start.

There are no additional restrictions on where a particular piece type may appear within the four setup rows.

---

## 7. Information visibility

The engine maintains two concepts separately:

### True state

Contains every piece's actual identity and is available only to:

- game transition logic;
- validation/debugging;
- supervised belief targets;
- deterministic replay tools.

### Player observation

Contains only information legally available or logically deducible by the acting player.

A training policy must never receive opponent hidden identities through the observation interface.

---

## 8. Combat knowledge

When combat occurs:

- both piece identities become public;
- public knowledge is persistent for surviving pieces;
- captured identities remain part of public game history and remaining-piece counts.

If an unrevealed piece makes a multi-square Scout move, its identity becomes known as Scout in the project observation state because no other piece can legally make that move.

---

## 9A. Terminal-condition precedence

More than one terminal condition can be satisfied by a single move. The engine resolves them in this fixed order, highest priority first:

1. `flag_capture`
2. `opponent_no_legal_move`
3. `both_no_legal_move_draw`
4. `battleless_move_limit_draw`
5. `absolute_move_limit_draw`

The principle is that genuine Stratego game-ending conditions take precedence over the project-specific training termination limits, which exist only to guarantee practical termination once the two anti-repetition rules were removed.

The order is observable only when a single move both reaches a draw threshold and settles the mobility question. Two consequences are worth stating explicitly:

- combat always resets the no-battle counter to zero, so a capture can never coincide with `battleless_move_limit_draw`;
- a player cannot strand itself with a non-combat move, because the square it just vacated is always available to move back into. `both_no_legal_move_draw` therefore always arises from a move that resolves combat, and so can never coincide with `battleless_move_limit_draw` either.

This section records behaviour that earlier revisions of `stratego_project_v1` left unspecified. It clarifies the existing ruleset rather than changing a stated rule, so the rules version is unchanged. Any future change to this ordering does require a new rules version identifier under section 10.

---

## 9. Terminal result encoding

For reinforcement learning:

- win: `+1` from the winner's perspective;
- loss: `-1` from the loser's perspective;
- draw: `0`.

For headline evaluation:

- win: `1.0` effective win;
- draw: `0.5` effective win;
- loss: `0.0` effective win.

Effective win rate:

\[
\text{effective win rate}
=
\frac{\text{wins}+0.5\times\text{draws}}{\text{games}}.
\]

The project target is **at least 85 percent effective win rate against casual human players**.

---

## 10. Rules versioning

Every game record must contain a `rules_version` identifier.

Initial identifier:

- `stratego_project_v1`

Any future addition of the two-square rule, continuous-chasing rule, alternate no-battle limits, or other game-mechanics changes requires a new version identifier.
