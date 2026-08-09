# Official / Source Rules of Stratego

## Purpose

This document records the Stratego rules that form the source baseline for the project. The primary technical source is **Appendix A, pages 15-16** of Sokota et al. (2025). That appendix distinguishes the core game rules, additional competitive rules, and an additional Strategus online rule.

This document is intentionally separate from the project's implemented rules. See `02_project_ruleset.md` for deliberate project deviations.

---

## 1. Board

- Stratego is played on a **10 by 10 grid**.
- There are **92 occupiable squares**.
- The remaining 8 squares form two non-occupiable lake regions in the center of the board.
- No piece may occupy or move through a lake square.

### Coordinate convention for this project documentation

- Columns: `a` through `j`.
- Rows: `1` through `10`.
- Each player owns the four rows nearest that player during setup.

The coordinate convention is an implementation convention; the 10 by 10 board and lake geometry are source rules.

---

## 2. Armies and setup

Each player begins with **40 pieces** and privately arranges all 40 pieces in the first four rows on that player's side.

The opponent must not know the identities of unrevealed pieces.

### Piece inventory

| Piece | Rank | Count | Normal movement | Special combat rule |
|---|---:|---:|---|---|
| Flag | — | 1 | Immovable | Loses to any attacking piece |
| Spy | 1 | 1 | 1 cardinal square | Defeats the Marshal when the Spy attacks it; otherwise combat is resolved by rank |
| Scout | 2 | 8 | Any positive number of unobstructed squares in one cardinal direction | Combat otherwise resolved by rank |
| Miner | 3 | 5 | 1 cardinal square | Defeats a Bomb when attacking; otherwise combat is resolved by rank |
| Sergeant | 4 | 4 | 1 cardinal square | By rank |
| Lieutenant | 5 | 4 | 1 cardinal square | By rank |
| Captain | 6 | 4 | 1 cardinal square | By rank |
| Major | 7 | 3 | 1 cardinal square | By rank |
| Colonel | 8 | 2 | 1 cardinal square | By rank |
| General | 9 | 1 | 1 cardinal square | By rank |
| Marshal | 10 | 1 | 1 cardinal square | Loses to the Spy when defending against a Spy; otherwise by rank |
| Bomb | — | 6 | Immovable | Defeats every attacking piece except a Miner |

Total pieces per player: **40**.

---

## 3. Turns

- Players alternate turns.
- On a turn, the acting player moves one movable piece.
- A legal ordinary move is cardinal: up, down, left, or right.
- Ordinary movable pieces move exactly one square.
- Pieces do not move diagonally.
- A piece cannot move onto a square occupied by a friendly piece.
- A piece cannot move onto or through a lake.

The paper treats the red player as the first-moving player in its self-play analysis. The project therefore uses red as the first player unless a later compatibility requirement changes this convention.

---

## 4. Scout movement

The Scout is the exception to one-square movement.

A Scout may move any positive number of squares in one cardinal direction provided that:

- every intermediate square is occupiable;
- every intermediate square is empty;
- the Scout does not jump over another piece;
- the Scout does not cross a lake;
- the destination is either empty or occupied by an opponent piece.

A long Scout move may therefore also be an attack if the first occupied square in the movement direction is an opponent piece at the chosen destination.

---

## 5. Immovable pieces

The following pieces never move after setup:

- Flag;
- Bomb.

Because they cannot move, they also cannot initiate combat.

---

## 6. Combat

Combat occurs when the acting player moves a piece onto a square occupied by an opponent piece.

At combat:

- both participating piece identities become known;
- the result is determined by the ranks and special rules;
- at least one participating piece is removed.

### Normal rank resolution

For ranked pieces without a special-case interaction:

- higher rank defeats lower rank;
- equal ranks eliminate both pieces.

### Required implementation interpretation

Appendix A specifies the winner/loser relationship but does not spell out post-combat square occupancy in detail. The project will use the conventional Stratego interpretation:

- if the attacker wins, the attacker occupies the destination square;
- if the defender wins, the defender remains on the destination square and the attacker is removed;
- if the combat is a tie, both pieces are removed and the destination becomes empty.

This interpretation is recorded explicitly so the engine behavior is unambiguous.

---

## 7. Special combat cases

### Spy versus Marshal

- If the **Spy attacks the Marshal**, the Spy wins.
- If the **Marshal attacks the Spy**, normal rank ordering applies and the Marshal wins.

### Miner versus Bomb

- If a **Miner attacks a Bomb**, the Miner wins and the Bomb is removed.
- If any other movable piece attacks a Bomb, the attacking piece loses.

### Flag

- The Flag loses when attacked.
- Capturing the opponent's Flag immediately wins the game.

---

## 8. Winning and drawing

The source rules in Appendix A state that a player wins by either:

1. capturing the opponent's Flag; or
2. leaving the opponent with no legal moves.

The game is a draw if **neither player would possess a legal move**.

---

## 9. Hidden information

Each player's piece identities begin hidden from the opponent.

For the project engine, a piece identity becomes public when a battle reveals it. The identity remains known thereafter while that piece remains on the board. A Scout that makes a multi-square move is also logically identifiable as a Scout from legal movement; the project observation system will preserve that deduction as known information.

The last sentence is a project information-state convention required for a machine-readable observation; it is a deduction from the movement rules rather than a separately stated Appendix A rule.

---

## 10. Additional competitive rules described by the paper

Appendix A identifies two additional rules for competitive play.

### 10.1 Two-square rule

The rule restricts repeated crossing of the same square boundary by the same piece across consecutive turns of that player's movement.

### 10.2 Continuous-chasing rule

The paper defines:

- **Threat:** moving a piece adjacent to an opponent piece.
- **Evade:** moving a piece that was threatened on the preceding turn away from the threatening piece.
- **Chase:** an unbroken sequence of alternating threats and evades.
- **Chasing:** making the threats during such a chase.

The continuous-chasing rule prevents a chasing player from recreating certain earlier positions in the chase.

**Project decision:** both rules are deliberately excluded from the first project ruleset. See `02_project_ruleset.md`.

---

## 11. Additional Strategus online rule described by the paper

The paper separately describes the **200-move rule** used by Strategus:

- a game is declared a draw after 200 consecutive moves without a battle.

The paper trained Ataraxos with a 100-move version of this rule and evaluated against its elite human opponent under the 200-move online rule.

This no-battle rule is important to our project because we are intentionally removing the two anti-repetition competitive rules. The project uses configurable no-battle limits; see `02_project_ruleset.md`.

---

## 12. Time controls

Appendix A notes that competitive Stratego may use time controls. The Ataraxos human evaluation used Strategus 15+3 timing.

Time controls are **not part of the training game-state logic** for our first implementation. The browser interface may add human-facing clocks later without changing the model's rules.

---

## 13. Source notes

Primary source:

- Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, and Gabriele Farina, *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search*, Appendix A, pages 15-16, 2025.

Official publisher reference checked:

- Hasbro, official Stratego product instructions page for Stratego product 04714.

When this document says **source rule**, it refers primarily to Appendix A of the research paper unless otherwise stated.
