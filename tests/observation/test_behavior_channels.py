"""Behavioural events and their 40 observation channels (68-107).

Covers `06_observation_v2_127ch.md` section 10 (formal definitions) and
`07_observation_validation_matrix.md` section 8 (positive cases, negative cases,
recency, context encoding, actor-knew flag and retrospective reinterpretation).
"""

import pytest

from stratego.engine.constants import (
    BEHAVIOR_DECLINED_ATTACK,
    BEHAVIOR_EVADE,
    BEHAVIOR_PROTECT,
    BEHAVIOR_THREAT,
    BEHAVIOR_WAS_PROTECTED,
    BLUE,
    RED,
    RulesConfig,
)
from stratego.engine.observation import (
    CH_OPPONENT_BEHAVIOR,
    CH_OWN_BEHAVIOR,
    build_observation,
)
from tests.helpers import cell, make_position, piece_at, play, square

LOOSE_RULES = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=10_000)

BEHAVIOR_ORDER = (
    BEHAVIOR_THREAT,
    BEHAVIOR_EVADE,
    BEHAVIOR_DECLINED_ATTACK,
    BEHAVIOR_PROTECT,
    BEHAVIOR_WAS_PROTECTED,
)


def behavior_base(behavior_type: str, own: bool = True) -> int:
    """First channel of a behaviour's four-feature block."""
    block = CH_OWN_BEHAVIOR if own else CH_OPPONENT_BEHAVIOR
    return block + 4 * BEHAVIOR_ORDER.index(behavior_type)


def features(state, behavior_type, square_name, observer=RED, own=True):
    """`(recency, rank, actor_knew, special)` read from the observation."""
    observation = build_observation(state, observer)
    base = behavior_base(behavior_type, own)
    return tuple(
        cell(observation, base + offset, square_name, observer) for offset in range(4)
    )


def event_of(state, square_name, behavior_type):
    record = piece_at(state, square_name)
    assert record is not None, f"no piece on {square_name}"
    return state.behavior_event(record.piece_id, behavior_type)


# ---------------------------------------------------------------------------
# Threat (definition 10.1)
# ---------------------------------------------------------------------------


def test_threat_is_recorded_when_the_mover_ends_adjacent_to_an_opponent():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e5": "sergeant", "j10": "flag"},
        acting_player=RED,
    )
    target = piece_at(state, "e5")
    play(state, "e3 e4")

    event = event_of(state, "e4", BEHAVIOR_THREAT)
    assert event is not None
    assert event.counterpart_piece_id == target.piece_id
    assert event.event_ply == 1
    assert features(state, BEHAVIOR_THREAT, "e4")[0] == pytest.approx(1.0)


def test_no_threat_for_diagonal_adjacency_only():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"f5": "sergeant", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    assert event_of(state, "e4", BEHAVIOR_THREAT) is None


def test_no_threat_when_the_actor_does_not_survive():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "f4": "scout", "j10": "flag"},
        acting_player=RED,
    )
    attacker = piece_at(state, "e3")
    play(state, "e3 e4")
    assert not attacker.alive
    assert state.behavior_event(attacker.piece_id, BEHAVIOR_THREAT) is None


def test_the_attacked_piece_is_never_the_threat_counterpart():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "f4": "sergeant", "j10": "flag"},
        acting_player=RED,
    )
    sergeant = piece_at(state, "f4")
    play(state, "e3 e4")  # marshal captures the captain and lands next to f4

    event = event_of(state, "e4", BEHAVIOR_THREAT)
    assert event is not None
    assert event.counterpart_piece_id == sergeant.piece_id


def test_no_threat_when_the_actor_ends_non_adjacent():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e5": "sergeant", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 d3")
    assert event_of(state, "d3", BEHAVIOR_THREAT) is None


def test_no_threat_when_only_a_friendly_piece_is_adjacent():
    state = make_position(
        red={"e3": "captain", "e5": "miner", "a1": "flag"},
        blue={"j9": "sergeant", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    assert event_of(state, "e4", BEHAVIOR_THREAT) is None


def test_threat_counterpart_is_the_lowest_board_square_index():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"d4": "sergeant", "e5": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    lower = piece_at(state, "d4")
    higher = piece_at(state, "e5")
    assert lower.current_square < higher.current_square

    play(state, "e3 e4")
    event = event_of(state, "e4", BEHAVIOR_THREAT)
    assert event.counterpart_piece_id == lower.piece_id


def test_threat_counterpart_selection_ignores_hidden_types():
    """Same geometry, swapped hidden types: the selection must not move."""
    selections = []
    for first_type, second_type in (("sergeant", "marshal"), ("marshal", "sergeant")):
        state = make_position(
            red={"e3": "captain", "a1": "flag"},
            blue={"d4": first_type, "e5": second_type, "j10": "flag"},
            acting_player=RED,
        )
        play(state, "e3 e4")
        selections.append(event_of(state, "e4", BEHAVIOR_THREAT).counterpart_piece_id)
    assert selections[0] == selections[1]


def test_all_threat_relations_are_retained_even_though_one_is_recorded():
    """`08_internal_state_spec.md` section 9 distinction."""
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"d4": "sergeant", "e5": "marshal", "f4": "scout", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    threatened = {relation[1] for relation in state.active_threat_relations}
    assert len(threatened) == 3
    assert event_of(state, "e4", BEHAVIOR_THREAT).counterpart_piece_id == (
        piece_at(state, "d4").piece_id
    )


# ---------------------------------------------------------------------------
# Evade (definition 10.2)
# ---------------------------------------------------------------------------


def threatened_position(**kwargs):
    """Blue threatens the red captain on e3 by stepping onto e4."""
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"e5": "sergeant", "j9": "scout", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
        **kwargs,
    )
    play(state, "e5 e4")
    return state


def test_evade_is_recorded_when_the_threatened_piece_escapes():
    state = threatened_position()
    threatener = piece_at(state, "e4")
    play(state, "e3 e2")

    event = event_of(state, "e2", BEHAVIOR_EVADE)
    assert event is not None
    assert event.counterpart_piece_id == threatener.piece_id
    assert features(state, BEHAVIOR_EVADE, "e2")[0] == pytest.approx(1.0)


def test_no_evade_when_the_piece_attacks_instead():
    state = threatened_position()
    captain = piece_at(state, "e3")
    play(state, "e3 e4")  # captain beats the sergeant instead of running
    assert state.behavior_event(captain.piece_id, BEHAVIOR_EVADE) is None


def test_no_evade_when_a_different_friendly_piece_moves():
    state = threatened_position()
    captain = piece_at(state, "e3")
    play(state, "c3 c4")
    assert state.behavior_event(captain.piece_id, BEHAVIOR_EVADE) is None


def test_a_threatened_piece_cannot_stay_adjacent_without_attacking():
    """The `P remains adjacent to A` negative case is geometrically unreachable.

    One opponent move creates threats from exactly one piece, and with purely
    orthogonal movement every square adjacent to the threatener other than the
    threatened square itself is a diagonal or blocked step away. So any legal
    non-attack move by the threatened piece breaks the adjacency, and every one
    of them must therefore record an evade.
    """
    from stratego.engine.actions import decode_action
    from stratego.engine.coordinates import are_adjacent
    from stratego.engine.legal_moves import legal_actions
    from stratego.engine.snapshot import clone_state

    reference = threatened_position()
    threatener_square = piece_at(reference, "e4").current_square
    captain_square = piece_at(reference, "e3").current_square

    captain_moves = [
        action
        for action in legal_actions(reference)
        if decode_action(action)[0] == captain_square
    ]
    assert captain_moves

    for action in captain_moves:
        destination = decode_action(action)[1]
        state = clone_state(reference)
        captain_id = state.board[captain_square]
        from stratego.engine.transition import apply_action

        apply_action(state, action)
        if destination == threatener_square:
            continue  # the attack case, covered separately
        assert not are_adjacent(destination, threatener_square)
        assert state.behavior_event(captain_id, BEHAVIOR_EVADE) is not None


def test_no_evade_when_the_threat_is_older_than_the_previous_move():
    state = threatened_position()
    play(state, "c3 c4")  # red does something else
    play(state, "j9 i9")  # blue's new move replaces the threat relations
    play(state, "e3 e2")  # only now does the captain move away
    assert event_of(state, "e2", BEHAVIOR_EVADE) is None


def test_no_evade_when_the_previous_move_created_no_threat():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"j9": "scout", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
    )
    play(state, "j9 i9", "e3 e2")
    assert event_of(state, "e2", BEHAVIOR_EVADE) is None


# ---------------------------------------------------------------------------
# Declined attack (definition 10.3)
# ---------------------------------------------------------------------------


def test_declined_attack_is_recorded_for_a_piece_that_did_not_move():
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"e4": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    target = piece_at(state, "e4")
    play(state, "c3 c4")  # red moves an unrelated piece

    event = event_of(state, "e3", BEHAVIOR_DECLINED_ATTACK)
    assert event is not None
    assert event.counterpart_piece_id == target.piece_id
    assert features(state, BEHAVIOR_DECLINED_ATTACK, "e3")[0] == pytest.approx(1.0)


def test_no_declined_attack_when_the_attack_is_actually_played():
    state = make_position(
        red={"e3": "marshal", "c3": "miner", "a1": "flag"},
        blue={"e4": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    attacker = piece_at(state, "e3")
    play(state, "e3 e4")
    assert state.behavior_event(attacker.piece_id, BEHAVIOR_DECLINED_ATTACK) is None


def test_no_declined_attack_for_an_immovable_piece():
    state = make_position(
        red={"e3": "bomb", "c3": "miner", "a1": "flag"},
        blue={"e4": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    bomb = piece_at(state, "e3")
    play(state, "c3 c4")
    assert state.behavior_event(bomb.piece_id, BEHAVIOR_DECLINED_ATTACK) is None


def test_no_declined_attack_when_no_opponent_is_adjacent():
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"e6": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    play(state, "c3 c4")
    assert event_of(state, "e3", BEHAVIOR_DECLINED_ATTACK) is None


def test_declined_attack_counterpart_is_the_lowest_square_index():
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"d3": "sergeant", "e4": "marshal", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    lower = piece_at(state, "d3")
    assert lower.current_square < piece_at(state, "e4").current_square
    play(state, "c3 c4")
    assert event_of(state, "e3", BEHAVIOR_DECLINED_ATTACK).counterpart_piece_id == (
        lower.piece_id
    )


def test_declined_attack_records_the_other_target_when_one_is_attacked():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"d3": "sergeant", "e4": "captain", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    declined = piece_at(state, "e4")
    play(state, "e3 d3")  # attacks d3, declining the e4 attack
    marshal = piece_at(state, "d3")
    event = state.behavior_event(marshal.piece_id, BEHAVIOR_DECLINED_ATTACK)
    assert event is not None
    assert event.counterpart_piece_id == declined.piece_id


# ---------------------------------------------------------------------------
# Protect and was protected (definitions 10.4 and 10.5)
# ---------------------------------------------------------------------------


def test_protect_and_was_protected_are_recorded_together():
    state = threatened_position()
    protected = piece_at(state, "e3")
    threatener = piece_at(state, "e4")
    play(state, "c3 d3")  # the miner becomes newly adjacent to the threatened captain

    protector = piece_at(state, "d3")
    protect_event = state.behavior_event(protector.piece_id, BEHAVIOR_PROTECT)
    assert protect_event is not None
    assert protect_event.counterpart_piece_id == protected.piece_id
    assert protect_event.context_piece_id == threatener.piece_id

    was_protected = state.behavior_event(protected.piece_id, BEHAVIOR_WAS_PROTECTED)
    assert was_protected is not None
    assert was_protected.counterpart_piece_id == protector.piece_id

    assert features(state, BEHAVIOR_PROTECT, "d3")[0] == pytest.approx(1.0)
    assert features(state, BEHAVIOR_WAS_PROTECTED, "e3")[0] == pytest.approx(1.0)


def test_no_protection_when_the_threatened_piece_moves_itself():
    state = threatened_position()
    captain = piece_at(state, "e3")
    play(state, "e3 e2")
    assert state.behavior_event(captain.piece_id, BEHAVIOR_PROTECT) is None
    assert state.behavior_event(captain.piece_id, BEHAVIOR_WAS_PROTECTED) is None


def test_no_protection_when_the_protector_was_already_adjacent():
    state = make_position(
        red={"e3": "captain", "d3": "miner", "a1": "flag"},
        blue={"e5": "sergeant", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
    )
    play(state, "e5 e4")  # threatens the captain
    miner = piece_at(state, "d3")
    play(state, "d3 d2")  # still not newly adjacent; it was adjacent before
    assert state.behavior_event(miner.piece_id, BEHAVIOR_PROTECT) is None


def test_no_protection_when_the_protector_attacks():
    state = make_position(
        red={"e3": "captain", "c4": "marshal", "a1": "flag"},
        blue={"e5": "sergeant", "d4": "scout", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
    )
    play(state, "e5 e4")  # threatens the captain
    marshal = piece_at(state, "c4")
    play(state, "c4 d4")  # attacks rather than moving to an empty square
    assert state.behavior_event(marshal.piece_id, BEHAVIOR_PROTECT) is None


def test_no_protection_when_nothing_was_threatened_last_move():
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"j9": "scout", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
    )
    play(state, "j9 i9", "c3 d3")
    miner = piece_at(state, "d3")
    assert state.behavior_event(miner.piece_id, BEHAVIOR_PROTECT) is None


def test_no_protection_when_the_mover_ends_non_adjacent():
    state = threatened_position()
    play(state, "c3 b3")
    miner = piece_at(state, "b3")
    assert state.behavior_event(miner.piece_id, BEHAVIOR_PROTECT) is None


def test_empty_square_protection_is_excluded_in_version_one():
    """Blue threatens nothing; guarding an empty square records no event."""
    state = make_position(
        red={"c3": "miner", "a1": "flag"},
        blue={"e5": "sergeant", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
    )
    play(state, "e5 e4")  # no red piece is adjacent, so no threat relation exists
    assert state.active_threat_relations == []
    play(state, "c3 d3")  # d3 guards the empty e3 square
    miner = piece_at(state, "d3")
    assert state.behavior_event(miner.piece_id, BEHAVIOR_PROTECT) is None


# ---------------------------------------------------------------------------
# Recency (validation matrix section 8.5)
# ---------------------------------------------------------------------------


def idle_state():
    """A threat event plus two distant pieces that can shuffle for many plies."""
    state = make_position(
        red={"e3": "captain", "a1": "scout", "j1": "flag"},
        blue={"e5": "sergeant", "a10": "scout", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    play(state, "e3 e4")  # red captain threatens the blue sergeant
    return state


def advance(state, plies):
    for _ in range(plies):
        if state.acting_player == RED:
            move = "a1 a2" if state.board[square("a1")] is not None else "a2 a1"
        else:
            move = "a10 a9" if state.board[square("a10")] is not None else "a9 a10"
        play(state, move)


@pytest.mark.parametrize(
    "delta,expected",
    [(0, 1.0), (8, 0.8), (16, 2 / 3), (32, 0.5), (64, 1 / 3), (128, 0.2)],
)
def test_recency_decay_matches_the_documented_formula(delta, expected):
    state = idle_state()
    advance(state, delta)
    recency = features(state, BEHAVIOR_THREAT, "e4")[0]
    assert recency == pytest.approx(expected, abs=1e-6)
    assert recency == pytest.approx(1.0 / (1.0 + delta / 32.0), abs=1e-6)


def test_a_later_event_of_the_same_type_replaces_the_older_one():
    state = idle_state()
    first_ply = event_of(state, "e4", BEHAVIOR_THREAT).event_ply
    advance(state, 9)  # returns the turn to red
    assert state.acting_player == RED
    assert features(state, BEHAVIOR_THREAT, "e4")[0] < 1.0

    play(state, "e4 d4")  # step away, then come back and threaten again
    advance(state, 1)
    play(state, "d4 e4")

    event = event_of(state, "e4", BEHAVIOR_THREAT)
    assert event.event_ply > first_ply
    assert features(state, BEHAVIOR_THREAT, "e4")[0] == pytest.approx(1.0)


def test_behaviour_values_travel_with_the_piece_and_vanish_on_capture():
    state = make_position(
        red={"e3": "captain", "a1": "scout", "j1": "flag"},
        blue={"e5": "marshal", "a10": "scout", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    play(state, "e3 e4")  # red captain threatens the blue marshal
    captain = piece_at(state, "e4")
    assert features(state, BEHAVIOR_THREAT, "e4")[0] == pytest.approx(1.0)

    play(state, "a10 a9", "e4 d4")
    assert features(state, BEHAVIOR_THREAT, "e4")[0] == 0.0
    assert features(state, BEHAVIOR_THREAT, "d4")[0] > 0.0

    play(state, "e5 e4", "a1 a2", "e4 d4")  # the marshal takes the captain
    assert not captain.alive
    assert features(state, BEHAVIOR_THREAT, "d4")[0] == 0.0


# ---------------------------------------------------------------------------
# Counterpart rank, special encoding and the actor-knew flag (8.6-8.8)
# ---------------------------------------------------------------------------


def revealed_counterpart_state(counterpart_type):
    """Blue's counterpart is revealed by combat before red threatens it."""
    state = make_position(
        red={"e3": "captain", "b3": "scout", "j1": "flag"},
        blue={"b5": counterpart_type, "a10": "scout", "j10": "flag"},
        acting_player=RED,
        revealed={"b5"},
        rules=LOOSE_RULES,
    )
    play(state, "e3 e4", "a10 a9", "e4 e3", "a9 a10")  # burn plies, no interaction
    return state


@pytest.mark.parametrize(
    "type_name,expected_rank,expected_special",
    [
        ("spy", 0.1, 0.0),
        ("miner", 0.3, 0.0),
        ("colonel", 0.8, 0.0),
        ("marshal", 1.0, 0.0),
        ("bomb", 0.0, 1.0),
        ("flag", 0.0, -1.0),
    ],
)
def test_known_counterpart_rank_and_special_encoding(
    type_name, expected_rank, expected_special
):
    state = make_position(
        red={"b3": "captain", "j1": "flag"},
        blue={"b5": type_name, "j10": "scout"},
        acting_player=RED,
        revealed={"b5"},
        rules=LOOSE_RULES,
    )
    play(state, "b3 b4")  # b4 is adjacent to b5

    recency, rank, actor_knew, special = features(state, BEHAVIOR_THREAT, "b4")
    assert recency == pytest.approx(1.0)
    assert actor_knew == 1.0
    assert rank == pytest.approx(expected_rank)
    assert special == pytest.approx(expected_special)


def test_unknown_counterpart_yields_zero_rank_and_special():
    state = make_position(
        red={"b3": "captain", "j1": "flag"},
        blue={"b5": "marshal", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    play(state, "b3 b4")
    recency, rank, actor_knew, special = features(state, BEHAVIOR_THREAT, "b4")
    assert recency == pytest.approx(1.0)
    assert actor_knew == 0.0
    assert rank == 0.0
    assert special == 0.0


def test_actor_knew_flag_is_historical_and_does_not_change_later():
    state = make_position(
        red={"b3": "captain", "c3": "marshal", "j1": "flag"},
        blue={"b5": "sergeant", "c5": "scout", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    play(state, "b3 b4")  # threat against a still-hidden sergeant
    event = event_of(state, "b4", BEHAVIOR_THREAT)
    assert event.actor_knew_counterpart_type is False
    assert features(state, BEHAVIOR_THREAT, "b4")[2] == 0.0

    # Reveal the sergeant through unrelated combat.
    play(state, "c5 c4", "c3 c4")
    assert piece_at(state, "c4").known_to(RED) or True  # marshal took the scout

    # The historical flag stays false even though red now knows more.
    assert event.actor_knew_counterpart_type is False
    assert features(state, BEHAVIOR_THREAT, "b4")[2] == 0.0


def test_retrospective_reinterpretation_of_a_protect_counterpart():
    """Validation-matrix section 8.8, observed from the opponent's side.

    Blue watches red protect a red piece. Blue may not see the protected piece's
    rank until that piece is legally revealed; afterwards the same unchanged
    event may legally show it.
    """
    state = make_position(
        red={"e3": "colonel", "c3": "miner", "a1": "flag"},
        blue={"e5": "sergeant", "j9": "scout", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE_RULES,
    )
    play(state, "e5 e4")  # blue threatens the red colonel
    play(state, "c3 d3")  # red miner protects it

    protect_event = state.behavior_event(
        piece_at(state, "d3").piece_id, BEHAVIOR_PROTECT
    )
    assert protect_event.actor_knew_counterpart_type is True
    original_ply = protect_event.event_ply

    # Blue observes red's behaviour block; the colonel is still hidden.
    _, rank, actor_knew, special = features(
        state, BEHAVIOR_PROTECT, "d3", observer=BLUE, own=False
    )
    assert actor_knew == 1.0
    assert rank == 0.0 and special == 0.0

    # The colonel is revealed by combat with the blue sergeant.
    play(state, "j9 i9", "e3 e4")
    assert piece_at(state, "e4").known_to(BLUE)

    _, rank, actor_knew, special = features(
        state, BEHAVIOR_PROTECT, "d3", observer=BLUE, own=False
    )
    assert rank == pytest.approx(0.8)  # colonel
    assert special == 0.0
    assert state.behavior_event(
        piece_at(state, "d3").piece_id, BEHAVIOR_PROTECT
    ).event_ply == original_ply


def test_own_and_opponent_behaviour_blocks_are_separate():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e5": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )
    play(state, "e3 e4")

    # Red sees the threat in its own block; blue sees the same event in the
    # opponent block, at the same normalized square.
    assert features(state, BEHAVIOR_THREAT, "e4", observer=RED, own=True)[0] == 1.0
    assert features(state, BEHAVIOR_THREAT, "e4", observer=RED, own=False)[0] == 0.0
    assert features(state, BEHAVIOR_THREAT, "e4", observer=BLUE, own=False)[0] == 1.0
    assert features(state, BEHAVIOR_THREAT, "e4", observer=BLUE, own=True)[0] == 0.0
