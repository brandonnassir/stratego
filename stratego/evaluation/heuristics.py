"""Observer-safe feature extraction and scoring primitives for the baselines.

Specification sources:

- `01_official_rules.md` sections 3-7 (movement, Scout, combat)
- `06_observation_v2_127ch.md` section 3 (what an observer may know)
- `09_public_event_and_replay_schema.md` sections 11, 12, 16 (the public surface)
- Phase 4 Agent 2 instructions ("Required baseline policies", "Hidden-information
  safety")

Why the safety argument is structural, not per-policy
----------------------------------------------------
Everything in this module is a pure function of :class:`PublicView`, the legal
action list, and the rules configuration. Agent 1 proved `PublicView` invariant
under :func:`stratego.engine.permutation.permute_hidden_identities`, and the
legal action list is one of the products the permutation gate protects. A pure
function of invariant inputs is invariant, so no baseline built on this module
can leak a hidden identity by construction -- the differential tests confirm the
property rather than being the only thing establishing it.

Nothing here imports `GameState`, `PieceRecord` or `belief_target`. The one
engine rule this module reaches for is :func:`resolve_combat`, and only ever
with two types the observer legally knows, or with a type drawn from the
*publicly deducible* unresolved-inventory distribution.

Public inference this module does make
--------------------------------------
Two deductions are legal for an observer and are used throughout:

1. `unresolved_opponent_counts` says how many copies of each type remain
   unaccounted for. Attacking an unknown piece therefore has a computable
   expected value, which is a far better signal than a hand-tuned constant.
2. A piece that has moved is neither a Flag nor a Bomb, since both are immovable
   and movement is public. This is exactly the constraint
   :func:`stratego.engine.permutation.permutation_is_valid` enforces, so
   conditioning on it can never distinguish two permutations of the same public
   position.

No other deduction is made. In particular nothing here consults `true_type` for
an opponent piece the observer has not legally seen -- `PublicPiece.piece_type`
is `None` in that case and every code path treats it as unknown.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..engine.actions import decode_action
from ..engine.combat import ATTACKER_WINS, DEFENDER_WINS, resolve_combat
from ..engine.constants import (
    BOARD_ROWS,
    BOMB,
    FLAG,
    IMMOVABLE_TYPES,
    MARSHAL,
    MINER,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    RED,
    SCOUT,
)
from ..engine.coordinates import NEIGHBOURS, RAYS, square_column, square_row
from .policy import PolicyInput, PublicPiece, PublicView

HEURISTICS_VERSION = "phase4_heuristics_v1"

# ---------------------------------------------------------------------------
# Material values
# ---------------------------------------------------------------------------
#
# Indexed by piece type, in the frozen enumeration order of `constants.py`.
# These are evaluation weights, not rules, so they may be revised by Agent 4's
# calibration under a policy-version bump.
#
# The Flag entry deserves a note. Capturing a *known* Flag ends the game and is
# handled by an explicit override (`FLAG_CAPTURE_BONUS`), never by this table.
# The table entry only ever appears inside the expected-value calculation for
# attacking an *unknown* piece, where it encodes "there is some chance this is
# the Flag". A literal win-sized number there would make every speculative
# attack look overwhelming and collapse every tier into a berserker, so the
# entry is sized as "clearly worth more than a Marshal" instead.
PIECE_VALUES: tuple[float, ...] = (
    25.0,  # spy
    10.0,  # scout
    30.0,  # miner
    14.0,  # sergeant
    18.0,  # lieutenant
    24.0,  # captain
    32.0,  # major
    46.0,  # colonel
    70.0,  # general
    100.0,  # marshal
    400.0,  # flag
    22.0,  # bomb
)
assert len(PIECE_VALUES) == NUM_PIECE_TYPES

#: Score for an attack that captures a Flag the observer legally knows about.
#: Large enough that no combination of other components can outweigh it.
FLAG_CAPTURE_BONUS = 1_000_000.0

#: Score for preventing an opponent piece from reaching the observer's own Flag.
FLAG_DEFENCE_BONUS = 100_000.0


def _build_combat_tables() -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Precompute the value of every attacker/defender pairing, both directions.

    `capture[a][d]` is the value to the *attacker* of attacking a defender of
    type `d`; `defence[a][d]` is the value to the *defender* of being attacked.
    They are not negatives of each other, because a mutual destruction trades
    two different pieces.

    Rows for immovable attackers are filled with zeros and are never read: an
    immovable piece generates no action, so it can never be an attacker.
    """
    capture: list[tuple[float, ...]] = []
    defence: list[tuple[float, ...]] = []
    for attacker in range(NUM_PIECE_TYPES):
        capture_row: list[float] = []
        defence_row: list[float] = []
        for defender in range(NUM_PIECE_TYPES):
            if attacker in IMMOVABLE_TYPES:
                capture_row.append(0.0)
                defence_row.append(0.0)
                continue
            outcome = resolve_combat(attacker, defender)
            if outcome == ATTACKER_WINS:
                capture_row.append(PIECE_VALUES[defender])
                defence_row.append(-PIECE_VALUES[defender])
            elif outcome == DEFENDER_WINS:
                capture_row.append(-PIECE_VALUES[attacker])
                defence_row.append(PIECE_VALUES[attacker])
            else:
                trade = PIECE_VALUES[defender] - PIECE_VALUES[attacker]
                capture_row.append(trade)
                defence_row.append(-trade)
        capture.append(tuple(capture_row))
        defence.append(tuple(defence_row))
    return tuple(capture), tuple(defence)


#: `CAPTURE_VALUES[attacker][defender]` -- value to the attacker of that attack.
#: `DEFENCE_VALUES[attacker][defender]` -- value to the defender of being hit.
CAPTURE_VALUES, DEFENCE_VALUES = _build_combat_tables()


def capture_value(attacker_type: int, defender_type: int) -> float:
    """Value to the attacker of attacking a defender of a *known* type."""
    return CAPTURE_VALUES[attacker_type][defender_type]


# ---------------------------------------------------------------------------
# Board orientation
# ---------------------------------------------------------------------------


def advance_progress(square: int, player: int) -> int:
    """How far `square` is from `player`'s own back row, in rows.

    Red sets up on rows 0-3 and advances upward; blue sets up on rows 6-9 and
    advances downward. Expressing both as "rows gained" lets every forward or
    territorial heuristic be written once for both colours.
    """
    row = square_row(square)
    return row if player == RED else (BOARD_ROWS - 1 - row)


def in_own_half(square: int, player: int) -> bool:
    """Whether `square` lies in `player`'s own half of the board."""
    return advance_progress(square, player) < BOARD_ROWS // 2


def manhattan(first: int, second: int) -> int:
    """Orthogonal distance between two squares, ignoring lakes and occupancy.

    An admissible lower bound on the number of moves needed to get from one to
    the other, which is all the approach heuristics need.
    """
    return abs(square_row(first) - square_row(second)) + abs(
        square_column(first) - square_column(second)
    )


# ---------------------------------------------------------------------------
# Candidate moves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateMove:
    """One legal action, decoded into the facts a heuristic actually wants.

    `target_type` is `None` exactly when the defender exists but the observer
    may not know its type. Every consumer must branch on that rather than
    assuming a type is available.
    """

    action_id: int
    source: int
    destination: int
    piece_id: int
    piece_type: int
    distance: int
    is_attack: bool
    target_piece_id: int | None
    target_type: int | None
    target_has_moved: bool
    target_known: bool
    advance: int

    @property
    def is_scout_run(self) -> bool:
        """A Scout move of more than one square."""
        return self.piece_type == SCOUT and self.distance > 1


@dataclass(frozen=True)
class ScoredMove:
    """One candidate with its score, the rule family that dominated, and why.

    `components` is a tuple of `(name, value)` pairs rather than a dict so the
    object stays hashable, ordered and directly comparable in the differential
    tests.
    """

    action_id: int
    score: float
    family: str
    components: tuple[tuple[str, float], ...] = ()

    def component_dict(self, digits: int = 4) -> dict[str, float]:
        return {name: round(value, digits) for name, value in self.components}


# ---------------------------------------------------------------------------
# The per-decision context
# ---------------------------------------------------------------------------


class DecisionContext:
    """Everything a baseline needs about one position, computed once per ply.

    Built purely from a :class:`PublicView` and the legal action list. Scoring a
    single move is then close to constant time, which matters because a
    mid-game Stratego position routinely offers well over a hundred actions and
    a calibration league plays millions of plies.
    """

    __slots__ = (
        "view",
        "me",
        "opponent",
        "moves",
        "own_flag_square",
        "own_flag_attackers",
        "known_opponent_marshal_square",
        "own_miner_count",
        "known_attacker_types",
        "hidden_mover_adjacent",
        "own_support",
        "empty_neighbours",
        "unresolved_counts",
        "unresolved_total",
        "unresolved_bombs",
        "average_hidden_value",
        "own_material",
        "opponent_material",
        "material_edge",
        "battleless_pressure",
        "last_source_of",
        "recent_move_count",
        "_expected_capture_cache",
        "_expected_defence_cache",
    )

    def __init__(self, view: PublicView, legal_actions: "Sequence[int]") -> None:
        self.view = view
        self.me = view.observer
        self.opponent = view.opponent

        pieces = view.pieces
        occupancy = view.occupancy

        # -- own landmarks --------------------------------------------------
        own_flag_square: int | None = None
        own_miner_count = 0
        own_material = 0.0
        for piece_id in view.own_piece_ids:
            piece = pieces[piece_id]
            piece_type = piece.piece_type
            if piece_type is None:  # pragma: no cover - own types are always known
                continue
            own_material += PIECE_VALUES[piece_type]
            if piece_type == FLAG:
                own_flag_square = piece.square
            elif piece_type == MINER:
                own_miner_count += 1
        self.own_flag_square = own_flag_square
        self.own_miner_count = own_miner_count

        # -- unresolved opponent inventory ----------------------------------
        self.unresolved_counts = view.unresolved_opponent_counts
        self.unresolved_bombs = self.unresolved_counts[BOMB]
        unresolved_total = 0
        unresolved_value = 0.0
        for piece_type in range(NUM_PIECE_TYPES):
            count = self.unresolved_counts[piece_type]
            if count > 0:
                unresolved_total += count
                unresolved_value += count * PIECE_VALUES[piece_type]
        self.unresolved_total = unresolved_total
        self.average_hidden_value = (
            unresolved_value / unresolved_total if unresolved_total else 0.0
        )

        # -- opponent landmarks and threat maps -----------------------------
        known_marshal_square: int | None = None
        opponent_material = 0.0

        known_attacker_types: list[list[int]] = [[] for _ in range(NUM_SQUARES)]
        hidden_mover_adjacent = [0] * NUM_SQUARES

        for piece_id in view.opponent_piece_ids:
            piece = pieces[piece_id]
            square = piece.square
            if square is None:  # pragma: no cover - opponent_piece_ids are alive
                continue
            piece_type = piece.piece_type
            if piece_type is None:
                opponent_material += self.average_hidden_value
                if piece.has_moved:
                    # Only a piece that has already moved is certainly able to
                    # attack; an unmoved hidden piece may be immovable.
                    for neighbour in NEIGHBOURS[square]:
                        hidden_mover_adjacent[neighbour] += 1
                continue

            opponent_material += PIECE_VALUES[piece_type]
            if piece_type == FLAG:
                # Reachable only in a terminal state: the sole reveal path is
                # combat, and combat with a Flag ends the game on the same ply.
                # A policy is never asked to decide from a terminal state, so no
                # heuristic here plans around a *located* opponent Flag.
                continue
            if piece_type == BOMB:
                # A revealed Bomb threatens nothing; it can only be attacked.
                continue
            if piece_type == MARSHAL:
                known_marshal_square = square

            if piece_type == SCOUT:
                for ray in RAYS[square]:
                    for reachable in ray:
                        known_attacker_types[reachable].append(piece_type)
                        if occupancy[reachable] is not None:
                            break
            else:
                for neighbour in NEIGHBOURS[square]:
                    known_attacker_types[neighbour].append(piece_type)

        self.known_opponent_marshal_square = known_marshal_square
        self.known_attacker_types = known_attacker_types
        self.hidden_mover_adjacent = hidden_mover_adjacent

        self.own_material = own_material
        self.opponent_material = opponent_material
        self.material_edge = own_material - opponent_material

        # -- own support and free space -------------------------------------
        own_support = [0] * NUM_SQUARES
        for piece_id in view.own_piece_ids:
            piece = pieces[piece_id]
            square = piece.square
            if square is None:  # pragma: no cover - own_piece_ids are alive
                continue
            piece_type = piece.piece_type
            if piece_type is None or piece_type in IMMOVABLE_TYPES:
                # An immovable piece defends nothing; it cannot recapture.
                continue
            for neighbour in NEIGHBOURS[square]:
                own_support[neighbour] += 1
        self.own_support = own_support

        empty_neighbours = [0] * NUM_SQUARES
        for square in range(NUM_SQUARES):
            empty_neighbours[square] = sum(
                1 for neighbour in NEIGHBOURS[square] if occupancy[neighbour] is None
            )
        self.empty_neighbours = empty_neighbours

        # -- recent-move memory, for repetition control ----------------------
        last_source_of: dict[int, int] = {}
        recent_move_count: dict[int, int] = {}
        for move in view.recent_moves:
            if move.player != self.me:
                continue
            last_source_of[move.piece_id] = move.source
            recent_move_count[move.piece_id] = recent_move_count.get(move.piece_id, 0) + 1
        self.last_source_of = last_source_of
        self.recent_move_count = recent_move_count

        limit = view.battleless_move_limit
        self.battleless_pressure = view.battleless_moves / limit if limit else 0.0

        self._expected_capture_cache: dict[tuple[int, bool], float] = {}
        self._expected_defence_cache: dict[tuple[int, bool], float] = {}

        #: Opponent pieces standing next to my own Flag right now. Precomputed
        #: because a scoring pass consults it once per attacking candidate.
        self.own_flag_attackers = self._find_own_flag_attackers()

        self.moves = tuple(self._decode(action_id) for action_id in legal_actions)

    # -- construction helpers ------------------------------------------------

    def _decode(self, action_id: int) -> CandidateMove:
        view = self.view
        source, destination = decode_action(action_id)
        piece_id = view.occupancy[source]
        if piece_id is None:  # pragma: no cover - the engine generated this action
            raise ValueError(f"legal action {action_id} has an empty source square")
        piece = view.pieces[piece_id]
        piece_type = piece.piece_type
        if piece_type is None:  # pragma: no cover - own types are always known
            raise ValueError(f"acting player cannot see the type of its own piece {piece_id}")

        target_piece_id = view.occupancy[destination]
        target: PublicPiece | None = (
            None if target_piece_id is None else view.pieces[target_piece_id]
        )

        # A legal move is always along one rank or one file, so the two deltas
        # never both exceed zero and the sum is the true step count.
        row_delta = abs(square_row(destination) - square_row(source))
        column_delta = abs(square_column(destination) - square_column(source))
        return CandidateMove(
            action_id=action_id,
            source=source,
            destination=destination,
            piece_id=piece_id,
            piece_type=piece_type,
            distance=row_delta + column_delta,
            is_attack=target is not None,
            target_piece_id=target_piece_id,
            target_type=None if target is None else target.piece_type,
            target_has_moved=False if target is None else target.has_moved,
            target_known=False if target is None else target.known,
            advance=(
                advance_progress(destination, self.me) - advance_progress(source, self.me)
            ),
        )

    # -- public inference ----------------------------------------------------

    def expected_capture_value(self, attacker_type: int, defender_has_moved: bool) -> float:
        """Expected value of attacking an unknown defender.

        The expectation runs over the publicly deducible unresolved inventory,
        excluding Flag and Bomb when the defender has already moved. Both facts
        are public, so this is a deduction the observer is entitled to make.
        """
        key = (attacker_type, defender_has_moved)
        cached = self._expected_capture_cache.get(key)
        if cached is not None:
            return cached

        total = 0
        accumulated = 0.0
        row = CAPTURE_VALUES[attacker_type]
        for piece_type in range(NUM_PIECE_TYPES):
            count = self.unresolved_counts[piece_type]
            if count <= 0:
                continue
            if defender_has_moved and piece_type in IMMOVABLE_TYPES:
                continue
            total += count
            accumulated += count * row[piece_type]
        value = accumulated / total if total else 0.0
        self._expected_capture_cache[key] = value
        return value

    def expected_defence_value(self, defender_type: int) -> float:
        """Expected value to me of being attacked by an unknown opponent piece.

        Only pieces that have already moved can attack, so the expectation runs
        over the unresolved inventory with Flag and Bomb excluded.
        """
        key = (defender_type, True)
        cached = self._expected_defence_cache.get(key)
        if cached is not None:
            return cached

        total = 0
        accumulated = 0.0
        for attacker_type in range(NUM_PIECE_TYPES):
            count = self.unresolved_counts[attacker_type]
            if count <= 0 or attacker_type in IMMOVABLE_TYPES:
                continue
            total += count
            accumulated += count * DEFENCE_VALUES[attacker_type][defender_type]
        value = accumulated / total if total else 0.0
        self._expected_defence_cache[key] = value
        return value

    # -- risk ----------------------------------------------------------------

    def known_risk(self, square: int, piece_type: int) -> float:
        """Worst outcome a *known* opponent piece can inflict on `square`.

        Zero or positive means no known attacker profits from attacking. The
        Spy/Marshal inversion falls out of the combat table, so a Marshal
        standing next to a revealed Spy is correctly seen as in danger.
        """
        attackers = self.known_attacker_types[square]
        if not attackers:
            return 0.0
        worst = 0.0
        for attacker_type in attackers:
            value = DEFENCE_VALUES[attacker_type][piece_type]
            if value < worst:
                worst = value
        return worst

    def hidden_risk(self, square: int, piece_type: int) -> float:
        """Expected exposure to *unknown* opponent pieces adjacent to `square`.

        Counts only hidden pieces that have already moved, since an unmoved
        hidden piece may be a Bomb or the Flag and could not attack at all. The
        result is never positive: a favourable draw from the hidden inventory is
        not something to plan around, so only the downside is scored.
        """
        movers = self.hidden_mover_adjacent[square]
        if movers <= 0:
            return 0.0
        expectation = self.expected_defence_value(piece_type)
        return expectation * movers if expectation < 0.0 else 0.0

    def is_known_losing_attack(self, move: CandidateMove) -> bool:
        """Whether the defender's type is known and the attack loses material."""
        if move.target_type is None:
            return False
        return CAPTURE_VALUES[move.piece_type][move.target_type] < 0.0

    def repetition_penalty(self, move: CandidateMove) -> float:
        """How strongly this move looks like shuffling rather than playing.

        Two public signals: returning a piece to the square it just left, and
        moving the same piece repeatedly inside the 16-ply public window.
        """
        penalty = 0.0
        if self.last_source_of.get(move.piece_id) == move.destination:
            penalty += 1.0
        repeats = self.recent_move_count.get(move.piece_id, 0)
        if repeats > 1:
            penalty += 0.25 * (repeats - 1)
        return penalty

    def _find_own_flag_attackers(self) -> frozenset[int]:
        """Live opponent pieces orthogonally adjacent to my own Flag.

        Uses only my own Flag's location -- which I always know -- and public
        occupancy. A neighbour whose type I have legally seen to be a Bomb or a
        Flag is excluded, since neither can ever move onto my Flag. A *hidden*
        neighbour is always counted: assuming an unidentified piece next to my
        Flag is harmless is exactly the optimism that loses games.
        """
        flag_square = self.own_flag_square
        if flag_square is None:
            return frozenset()
        occupancy = self.view.occupancy
        pieces = self.view.pieces
        found = []
        for neighbour in NEIGHBOURS[flag_square]:
            piece_id = occupancy[neighbour]
            if piece_id is None:
                continue
            piece = pieces[piece_id]
            if piece.owner != self.opponent:
                continue
            if piece.piece_type is not None and piece.piece_type in IMMOVABLE_TYPES:
                continue
            found.append(piece_id)
        return frozenset(found)

    def unmoved_opponent_cluster(self, square: int) -> int:
        """Opponent pieces adjacent to `square` that have never moved.

        A public proxy for "defensive zone": a block of pieces that has sat
        still all game is where Bombs and the Flag tend to be. It says nothing
        about any individual piece's type.
        """
        occupancy = self.view.occupancy
        pieces = self.view.pieces
        count = 0
        for neighbour in NEIGHBOURS[square]:
            piece_id = occupancy[neighbour]
            if piece_id is None:
                continue
            piece = pieces[piece_id]
            if piece.owner == self.opponent and not piece.has_moved:
                count += 1
        return count

    def is_exposed(self, piece_id: int) -> bool:
        """Whether the opponent has legally learned this own piece's type."""
        return piece_id in self.view.own_piece_ids_known_to_opponent


def build_context(request: PolicyInput) -> DecisionContext:
    """Build the per-decision context from an observer-safe policy request."""
    return DecisionContext(request.require_public_view(), request.legal_actions)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def rank_moves(scored: "Iterable[ScoredMove]") -> tuple[ScoredMove, ...]:
    """Order candidates by descending score, breaking ties by action identifier.

    The tie-break is applied here, before any sampling, so the candidate list a
    stochastic policy draws from is itself deterministic.
    """
    return tuple(sorted(scored, key=lambda move: (-move.score, move.action_id)))


def select_from_ranked(
    request: PolicyInput,
    ranked: "Sequence[ScoredMove]",
    *,
    margin: float = 0.0,
) -> tuple[ScoredMove, bool]:
    """Pick one candidate, optionally sampling among near-best moves.

    With `margin <= 0` the best-ranked candidate is returned and the policy is
    fully deterministic. With a positive margin, every candidate within `margin`
    of the best score forms a pool -- already in deterministic order -- and one
    is drawn from `request.random_stream()`.

    The margin exists for a concrete reason rather than as flavour: two purely
    deterministic policies facing each other tend to lock into a repeating
    shuffle and hit the battleless-move draw limit, which would leave the whole
    ladder statistically indistinguishable at draws.
    """
    if not ranked:  # pragma: no cover - the contract guarantees a legal action
        raise ValueError("cannot select from an empty candidate list")
    if margin <= 0.0 or len(ranked) == 1:
        return ranked[0], False

    best = ranked[0].score
    limit = best - margin
    pool = [move for move in ranked if move.score >= limit]
    if len(pool) == 1:
        return pool[0], False
    index = request.random_stream().randrange(len(pool))
    return pool[index], True


def build_diagnostics(
    chosen: ScoredMove,
    ranked: "Sequence[ScoredMove]",
    *,
    sampled: bool,
    top_k: int = 3,
    digits: int = 4,
) -> dict:
    """Serialisable, permutation-invariant diagnostics for one decision.

    Contains only action identifiers, scores derived from public information and
    a rule-family label. No opponent type appears, known or otherwise, so this
    payload is safe to compare for equality in the hidden-information tests --
    which is precisely what those tests do.
    """
    return {
        "rule": chosen.family,
        "score": round(chosen.score, digits),
        "components": chosen.component_dict(digits),
        "candidate_count": len(ranked),
        "sampled": sampled,
        "top_candidates": [
            [move.action_id, round(move.score, digits)] for move in ranked[:top_k]
        ],
    }


# ---------------------------------------------------------------------------
# Shared scoring fragments
# ---------------------------------------------------------------------------


def combat_component(context: DecisionContext, move: CandidateMove) -> tuple[float, str]:
    """Value of the combat this move initiates, and the rule family it belongs to.

    Three cases, in the order a human would think about them: a known Flag ends
    the game, a known type resolves exactly, and an unknown type resolves in
    expectation over the public unresolved inventory.

    The first case is a correctness guard rather than a live branch. Under
    `stratego_project_v1` the only reveal path is combat, and combat with a Flag
    ends the game on the same ply, so no policy is ever asked to decide against
    a Flag it has already identified. "Immediate Flag capture" therefore lives
    inside the expectation of the third case, where the unresolved Flag carries
    its share of the value of attacking an unknown piece.
    """
    if not move.is_attack:
        return 0.0, "quiet"
    if move.target_type == FLAG:
        return FLAG_CAPTURE_BONUS, "flag_capture"
    if move.target_type is not None:
        value = capture_value(move.piece_type, move.target_type)
        if value > 0.0:
            return value, "winning_capture"
        if value < 0.0:
            return value, "losing_capture"
        return value, "even_trade"
    return (
        context.expected_capture_value(move.piece_type, move.target_has_moved),
        "speculative_attack",
    )


__all__ = [
    "CAPTURE_VALUES",
    "DEFENCE_VALUES",
    "FLAG_CAPTURE_BONUS",
    "FLAG_DEFENCE_BONUS",
    "HEURISTICS_VERSION",
    "PIECE_VALUES",
    "CandidateMove",
    "DecisionContext",
    "ScoredMove",
    "advance_progress",
    "build_context",
    "build_diagnostics",
    "capture_value",
    "combat_component",
    "in_own_half",
    "manhattan",
    "rank_moves",
    "select_from_ranked",
]
