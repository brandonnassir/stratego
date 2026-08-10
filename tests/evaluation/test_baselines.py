"""The Phase 4 baseline and stress opponent suite.

Four properties are tested here:

1. the catalogue holds the policies Phase 4 promises, under stable identities;
2. every policy always returns a legal action, in random and crafted positions;
3. every decision is reproducible from `(public input, policy seed, ply)`;
4. each policy actually does the thing its name claims -- the ladder tiers make
   the distinctions their design says they make, and the stress policies produce
   measurably different games.

Hidden-information safety lives in `test_baseline_information_safety.py`.
"""

import json
from collections import Counter

import pytest

from stratego.engine.actions import decode_action, encode_action
from stratego.engine.combat import ATTACKER_WINS, BOTH_REMOVED, DEFENDER_WINS, resolve_combat
from stratego.engine.constants import (
    BLUE,
    BOMB,
    EVALUATION_RULES,
    IMMOVABLE_TYPES,
    MARSHAL,
    MINER,
    NUM_PIECE_TYPES,
    PIECE_TYPE_NAMES,
    RED,
    SCOUT,
    SERGEANT,
    SPY,
)
from stratego.engine.legal_moves import legal_actions
from stratego.engine.random_play import play_random_game_to_ply
from stratego.engine.state import GameState, RecentMove, create_game
from stratego.engine.transition import apply_action
from stratego.evaluation.baselines import (
    BASIC_WEIGHTS,
    LADDER_POLICY_CLASSES,
    STRATEGIC_WEIGHTS,
    TACTICAL_WEIGHTS,
    BasicHeuristicPolicy,
    ScoringPolicy,
    StrategicRuleBasedPolicy,
    TacticalRuleBasedPolicy,
)
from stratego.evaluation.heuristics import (
    CAPTURE_VALUES,
    DEFENCE_VALUES,
    PIECE_VALUES,
    ScoredMove,
    advance_progress,
    build_context,
    manhattan,
    rank_moves,
    select_from_ranked,
)
from stratego.evaluation.match_spec import build_paired_schedule
from stratego.evaluation.policy import PolicyRequirements, build_policy_input
from stratego.evaluation.registry import (
    ALL_POLICY_CLASSES,
    ALL_POLICY_IDS,
    LADDER_POLICY_IDS,
    STRESS_POLICY_IDS,
    UnknownPolicyError,
    build_policies,
    build_policy,
    policy_catalog,
    policy_ref,
)
from stratego.evaluation.setup_bank import SetupBank
from stratego.evaluation.stress import STRESS_POLICY_CLASSES
from tests.helpers import make_position, square

ALL_POLICY_ID_LIST = list(ALL_POLICY_IDS)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bank() -> SetupBank:
    """A small slice of the evaluation bank. Agent 1 tests the bank itself."""
    return SetupBank.generate(8)


def random_positions(count: int, plies=(10, 25, 45, 80, 140)):
    """Seeded nonterminal positions spread across the phases of a game."""
    produced = 0
    for seed in range(500):
        if produced >= count:
            return
        for ply in plies:
            if produced >= count:
                return
            state = play_random_game_to_ply(seed, ply, rules=EVALUATION_RULES)
            if state.terminal or state.total_moves != ply:
                continue
            yield state
            produced += 1


def make_request(state: GameState, policy, seed: int = 90210, **overrides):
    fields = {
        "policy": policy.ref,
        "policy_seed": seed,
        "requirements": policy.requirements,
        "suite_version": "test_suite",
        "match_id": "m-test",
        "paired_unit_id": "u-test",
    }
    fields.update(overrides)
    return build_policy_input(state, **fields)


def play_game(spec, policies, bank, profile_for=None):
    """Play one match through the contract, optionally profiling one side.

    A deliberately minimal loop. Agent 3 owns the real runner; this exists so the
    behavioural assertions below can look at whole games rather than at single
    positions, which is the only level at which "attacks a lot" is meaningful.
    """
    red_setup, blue_setup = spec.resolve_setups(bank)
    state = create_game(red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id)
    profile: Counter = Counter()

    while not state.terminal:
        actor = state.acting_player
        ref = spec.policy_ref_for(actor)
        policy = policies[ref.token]
        legal = legal_actions(state)
        request = build_policy_input(
            state,
            policy=ref,
            policy_seed=spec.policy_seed_for(actor),
            requirements=policy.requirements,
            suite_version=spec.suite_version,
            match_id=spec.match_id,
            paired_unit_id=spec.paired_unit_id,
            legal=legal,
        )
        result = policy.decide_checked(request)

        if profile_for is not None and ref == profile_for:
            source, destination = decode_action(result.selected_action_id)
            mover = state.piece_at(source)
            target = state.piece_at(destination)
            profile["moves"] += 1
            profile[f"type_{PIECE_TYPE_NAMES[mover.true_type]}"] += 1
            if target is not None:
                profile["attacks"] += 1
            if mover.true_type == SCOUT and abs(source - destination) not in (1, 10):
                profile["scout_runs"] += 1

        apply_action(state, result.selected_action_id, legal=legal)

    profile["plies"] = state.total_moves
    profile[f"terminal_{state.terminal_reason}"] += 1
    return state, profile


def profile_matchup(candidate_id: str, opponent_id: str, bank, units: int = 2) -> Counter:
    """Aggregate behavioural counters for `candidate_id` over a few paired games."""
    candidate = build_policy(candidate_id)
    opponent = build_policy(opponent_id)
    policies = {candidate.ref.token: candidate, opponent.ref.token: opponent}
    totals: Counter = Counter()
    schedule = build_paired_schedule(
        candidate.ref, opponent.ref, range(units), setup_bank_version=bank.bank_version
    )
    for unit in schedule:
        for spec in unit.matches:
            _, profile = play_game(spec, policies, bank, profile_for=candidate.ref)
            totals.update(profile)
    return totals


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_the_catalogue_contains_the_full_phase_4_ladder():
    assert LADDER_POLICY_IDS == (
        "random_legal",
        "basic_heuristic",
        "tactical_rule_based",
        "strategic_rule_based",
    )
    assert len(LADDER_POLICY_CLASSES) == 4


def test_the_catalogue_contains_at_least_four_stress_policies():
    # The completion gate requires four; the suite ships six so Agent 4 has room
    # to characterise behaviour on more than one axis.
    assert len(STRESS_POLICY_IDS) >= 4
    assert len(STRESS_POLICY_CLASSES) == len(STRESS_POLICY_IDS)
    assert all(policy_id.startswith("stress_") for policy_id in STRESS_POLICY_IDS)


def test_policy_identifiers_are_unique_and_versioned():
    assert len(set(ALL_POLICY_IDS)) == len(ALL_POLICY_IDS)
    for policy_class in ALL_POLICY_CLASSES:
        assert policy_class.policy_version
        assert policy_class.description


def test_no_policy_borrows_the_contract_fixture_prefix():
    """A `contract_*` name would let an interface fixture enter a league."""
    assert not any(policy_id.startswith("contract_") for policy_id in ALL_POLICY_IDS)


def test_the_catalogue_round_trips_through_identifiers():
    for policy_id in ALL_POLICY_IDS:
        policy = build_policy(policy_id)
        assert policy.policy_id == policy_id
        assert policy.ref == policy_ref(policy_id)


def test_an_unknown_identifier_fails_loudly():
    with pytest.raises(UnknownPolicyError):
        build_policy("strategic_rule_based_v2")


def test_the_catalogue_serialises():
    catalog = policy_catalog()
    assert len(catalog) == len(ALL_POLICY_IDS)
    assert json.loads(json.dumps(catalog)) == catalog
    roles = {entry["policy_id"]: entry["role"] for entry in catalog}
    assert roles["random_legal"] == "ladder"
    assert roles["stress_chaos"] == "stress"


def test_no_policy_requests_more_than_it_needs():
    """A product that is never built can never leak; nothing here wants the tensor."""
    for policy in build_policies():
        assert policy.requirements.observation is False
        assert policy.requirements.legal_action_mask is False


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_every_policy_returns_a_legal_action(policy_id):
    policy = build_policy(policy_id)
    positions = 0
    for state in random_positions(24):
        request = make_request(state, policy)
        assert request.legal_actions, "a nonterminal state must offer a legal action"
        result = policy.decide_checked(request)
        assert result.selected_action_id in request.legal_actions
        positions += 1
    assert positions == 24


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_every_policy_survives_a_whole_game_against_itself_in_spirit(policy_id, bank):
    """A full game exercises openings, mid-game and the endgame in one pass."""
    candidate = build_policy(policy_id)
    opponent = build_policy(
        "random_legal" if policy_id != "random_legal" else "basic_heuristic"
    )
    policies = {candidate.ref.token: candidate, opponent.ref.token: opponent}
    unit = build_paired_schedule(
        candidate.ref, opponent.ref, [0], setup_bank_version=bank.bank_version
    )[0]
    for spec in unit.matches:
        state, _ = play_game(spec, policies, bank)
        assert state.terminal


def test_a_policy_that_returns_an_illegal_action_is_rejected():
    """The contract must catch a broken policy rather than trust it."""

    class BrokenPolicy(ScoringPolicy):
        policy_id = "test_broken"
        policy_version = "1.0.0"
        requirements = PolicyRequirements(public_view=False)

        def decide(self, request):
            illegal = next(
                action for action in range(10_000) if action not in request.legal_actions
            )
            return self.result(request, illegal)

    policy = BrokenPolicy()
    state = next(random_positions(1))
    from stratego.evaluation.policy import PolicyContractError

    with pytest.raises(PolicyContractError):
        policy.decide_checked(make_request(state, policy))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_the_same_input_and_seed_give_the_same_decision(policy_id):
    policy = build_policy(policy_id)
    for state in random_positions(6):
        request = make_request(state, policy)
        first = policy.decide_checked(request)
        for _ in range(4):
            repeat = policy.decide_checked(make_request(state, policy))
            assert repeat.selected_action_id == first.selected_action_id
            assert repeat.diagnostics == first.diagnostics


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_a_fresh_policy_instance_decides_identically(policy_id):
    """No policy may carry state between decisions."""
    for state in random_positions(4):
        first = build_policy(policy_id)
        second = build_policy(policy_id)
        # Drive the first instance through an unrelated position, so a policy
        # that cached anything would diverge here.
        other = next(random_positions(1, plies=(60,)))
        first.decide_checked(make_request(other, first))

        left = first.decide_checked(make_request(state, first))
        right = second.decide_checked(make_request(state, second))
        assert left.selected_action_id == right.selected_action_id
        assert left.diagnostics == right.diagnostics


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_stochastic_policies_actually_consume_their_seed(policy_id):
    """Otherwise `stochastic = True` would be a comment, not a property.

    A scoring policy only reaches for the seed when two or more candidates fall
    inside its margin, which happens in roughly a quarter to all positions
    depending on the policy. Asserting "some position out of N varied" would be
    a flaky way to say that, so the test instead finds a genuine near-tie and
    requires the seed to be what resolves it.
    """
    policy = build_policy(policy_id)
    assert policy.stochastic, f"{policy_id} no longer declares itself stochastic"

    seeds = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    if not isinstance(policy, ScoringPolicy) or policy.selection_margin <= 0.0:
        # Random and Chaos draw from the stream on every decision.
        for state in random_positions(3):
            actions = {
                policy.decide_checked(
                    make_request(state, policy, seed=seed)
                ).selected_action_id
                for seed in seeds
            }
            assert len(actions) > 1
        return

    for state in random_positions(40):
        context = build_context(make_request(state, policy))
        ranked = rank_moves(policy.score(context, move) for move in context.moves)
        best = ranked[0].score
        pool = [move for move in ranked if move.score >= best - policy.selection_margin]
        if len(pool) < 2:
            continue
        actions = {
            policy.decide_checked(make_request(state, policy, seed=seed)).selected_action_id
            for seed in seeds
        }
        assert len(actions) > 1, f"{policy_id} ignored its seed at a genuine near-tie"
        assert actions <= {move.action_id for move in pool}
        return
    pytest.fail(f"no near-tie position found for {policy_id}")


def test_zero_margin_selection_is_fully_deterministic():
    """The deterministic branch of the selector, which stress variants may use."""

    class DeterministicPolicy(BasicHeuristicPolicy):
        policy_id = "test_deterministic_basic"
        policy_version = "1.0.0"
        selection_margin = 0.0
        stochastic = False

    policy = DeterministicPolicy()
    for state in random_positions(6):
        chosen = {
            policy.decide_checked(make_request(state, policy, seed=seed)).selected_action_id
            for seed in (1, 17, 999, 123456)
        }
        assert len(chosen) == 1, "a zero-margin policy must ignore its seed entirely"


def test_ties_are_broken_before_sampling():
    """Ranking is by score then action identifier, so the pool order is fixed."""
    scored = [
        ScoredMove(500, 1.0, "a"),
        ScoredMove(100, 1.0, "b"),
        ScoredMove(300, 2.0, "c"),
        ScoredMove(200, 2.0, "d"),
    ]
    ranked = rank_moves(scored)
    assert [move.action_id for move in ranked] == [200, 300, 100, 500]
    assert rank_moves(reversed(scored)) == ranked


def test_sampling_only_reaches_candidates_inside_the_margin():
    policy = build_policy("basic_heuristic")
    state = next(random_positions(1))
    ranked = rank_moves(
        policy.score(context, move)
        for context in [build_context(make_request(state, policy))]
        for move in context.moves
    )
    best = ranked[0].score
    inside = {move.action_id for move in ranked if move.score >= best - policy.selection_margin}
    request = make_request(state, policy)
    for seed in range(40):
        chosen, _ = select_from_ranked(
            make_request(state, policy, seed=seed), ranked, margin=policy.selection_margin
        )
        assert chosen.action_id in inside
    assert request.legal_actions  # the request really did describe this position


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_diagnostics_are_serialisable_and_stable(policy_id):
    policy = build_policy(policy_id)
    for state in random_positions(4):
        result = policy.decide_checked(make_request(state, policy))
        encoded = json.dumps(result.to_dict(), sort_keys=True)
        assert json.loads(encoded)["diagnostics"] == json.loads(
            json.dumps(dict(result.diagnostics), sort_keys=True)
        )
        assert "rule" in result.diagnostics


#: Every string a diagnostic in this suite is allowed to contain: the top-level
#: keys, the rule-family labels, the score-component names and the Chaos feature
#: names. Some are rare enough not to appear in a small sample, so the test below
#: checks containment rather than equality.
DIAGNOSTIC_VOCABULARY = frozenset(
    {
        # structure
        "rule",
        "score",
        "components",
        "candidate_count",
        "sampled",
        "top_candidates",
        "objective",
        # rule families
        "uniform_legal",
        "quiet",
        "advance",
        "evade",
        "flag_capture",
        "flag_defence",
        "winning_capture",
        "losing_capture",
        "even_trade",
        "speculative_attack",
        "miner_demolition",
        "spy_hunt",
        "scout_probe",
        "pressure",
        "attack",
        "forced_attack",
        "shuffle",
        "chaos",
        "scout_run",
        "scout_step",
        "miner_advance",
        "bomb_demolition",
        "bomb_hunt",
        # score components
        "combat",
        "repetition",
        "known_risk",
        "hidden_risk",
        "support",
        "miner_bomb",
        "scout_information",
        "mobility",
        "miner_preservation",
        "exposure",
        "flag_guard",
        "battleless",
        "scout_preference",
        "non_scout",
        "miner_preference",
        "unmoved_cluster",
        "known_bomb",
        "bomb_candidate",
        "combat_aversion",
        "retreat_preference",
        "stay_home",
        "attack_preference",
        "reveal_aversion",
        "scout_reveal_aversion",
        "already_exposed",
        "conceal_value",
        # chaos objective features
        "distance",
        "lateral",
        "crowding",
        "value",
    }
)


def diagnostic_strings(value):
    """Every string reachable inside a diagnostics payload, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from diagnostic_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from diagnostic_strings(item)


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_diagnostics_draw_from_a_closed_vocabulary(policy_id):
    """Diagnostics must be labels and numbers, never text derived from the state.

    This is the structural half of the diagnostics safety argument: if no string
    in a payload can depend on the position, no string can carry a hidden
    identity. The behavioural half -- that the *numbers* do not vary with hidden
    state either -- is `test_baseline_information_safety.py`, which compares
    whole payloads across a hidden-identity permutation.

    Note that several allowed labels contain a piece-type word (`miner_bomb`,
    `scout_probe`). Each names either the acting player's own piece or a target
    whose type the observer has legally been shown, so none of them can be a
    leak; the permutation suite is what proves that rather than this test.
    """
    policy = build_policy(policy_id)
    for state in random_positions(8):
        result = policy.decide_checked(make_request(state, policy))
        found = set(diagnostic_strings(dict(result.diagnostics)))
        unexpected = found - DIAGNOSTIC_VOCABULARY
        assert not unexpected, f"{policy_id} emitted unrecognised diagnostic text: {unexpected}"


# ---------------------------------------------------------------------------
# Heuristic primitives
# ---------------------------------------------------------------------------


def test_the_capture_tables_agree_with_the_engine_combat_resolver():
    """The tables are a cache of engine behaviour, never a second rule set."""
    for attacker in range(NUM_PIECE_TYPES):
        if attacker in IMMOVABLE_TYPES:
            continue
        for defender in range(NUM_PIECE_TYPES):
            outcome = resolve_combat(attacker, defender)
            capture = CAPTURE_VALUES[attacker][defender]
            defence = DEFENCE_VALUES[attacker][defender]
            if outcome == ATTACKER_WINS:
                assert capture == PIECE_VALUES[defender]
                assert defence == -PIECE_VALUES[defender]
            elif outcome == DEFENDER_WINS:
                assert capture == -PIECE_VALUES[attacker]
                assert defence == PIECE_VALUES[attacker]
            else:
                assert outcome == BOTH_REMOVED
                assert capture == PIECE_VALUES[defender] - PIECE_VALUES[attacker]


def test_the_spy_marshal_inversion_survives_into_the_tables():
    assert CAPTURE_VALUES[SPY][MARSHAL] == PIECE_VALUES[MARSHAL]
    assert CAPTURE_VALUES[MARSHAL][SPY] == PIECE_VALUES[SPY]
    assert DEFENCE_VALUES[SPY][MARSHAL] == -PIECE_VALUES[MARSHAL]


def test_only_a_miner_profits_from_a_known_bomb():
    for attacker in range(NUM_PIECE_TYPES):
        if attacker in IMMOVABLE_TYPES:
            continue
        value = CAPTURE_VALUES[attacker][BOMB]
        assert (value > 0) is (attacker == MINER)


def test_advance_progress_reads_the_same_for_both_colours():
    """Rows gained toward the opponent, so one heuristic serves both sides."""
    assert advance_progress(square("a1"), RED) == 0
    assert advance_progress(square("a10"), RED) == 9
    assert advance_progress(square("a10"), BLUE) == 0
    assert advance_progress(square("a1"), BLUE) == 9


def test_manhattan_matches_a_scout_run_length():
    assert manhattan(square("a1"), square("a5")) == 4
    assert manhattan(square("a1"), square("e1")) == 4
    assert manhattan(square("a1"), square("a1")) == 0


def test_expected_capture_value_excludes_immovables_from_a_moved_defender():
    """A piece that has moved is neither Flag nor Bomb; both facts are public."""
    state = make_position(
        red={"a1": "flag", "e5": "captain"},
        blue={"j10": "flag", "e6": "captain", "f6": "scout"},
        moved={"e6"},
        acting_player=RED,
    )
    policy = build_policy("tactical_rule_based")
    context = build_context(make_request(state, policy))

    moved = context.expected_capture_value(MINER, True)
    unmoved = context.expected_capture_value(MINER, False)
    # A Miner's upside is the Bomb, which only an unmoved piece can be.
    assert unmoved > moved


def test_known_risk_prices_the_spy_marshal_inversion():
    state = make_position(
        red={"a1": "flag", "e5": "marshal"},
        blue={"j10": "flag", "e6": "spy"},
        revealed={"e6"},
        moved={"e6"},
        acting_player=RED,
    )
    policy = build_policy("tactical_rule_based")
    context = build_context(make_request(state, policy))
    # `known_risk` reports only the downside, so a favourable matchup scores
    # zero rather than a positive number: a Spy the Marshal would beat anyway is
    # not a reason to walk toward it.
    assert context.known_risk(square("e5"), MARSHAL) == -PIECE_VALUES[MARSHAL]
    assert context.known_risk(square("e5"), SERGEANT) == 0.0


def test_the_unresolved_inventory_counts_exactly_the_live_hidden_pieces():
    """The expected-value calculation is only sound if this identity holds.

    `unresolved_opponent_counts` is `inventory - pieces the observer has legally
    identified`, counting captured pieces as identified. That equals the live
    hidden pieces only because every capture in this ruleset comes from combat
    and combat reveals both participants. If a future rule ever removed a piece
    without revealing it, every expectation in this module would silently be
    taken over the wrong distribution -- so the identity is asserted rather
    than assumed.
    """
    policy = build_policy("tactical_rule_based")
    for state in random_positions(20):
        request = make_request(state, policy)
        context = build_context(request)
        view = request.require_public_view()
        assert context.unresolved_total == len(view.unresolved_opponent_piece_ids)


def test_a_hidden_piece_that_never_moved_is_not_counted_as_an_attacker():
    """It may be a Bomb or the Flag, neither of which can attack."""
    unmoved = make_position(
        red={"a1": "flag", "e5": "captain"},
        blue={"j10": "flag", "e6": "major"},
        acting_player=RED,
    )
    moved = make_position(
        red={"a1": "flag", "e5": "captain"},
        blue={"j10": "flag", "e6": "major"},
        moved={"e6"},
        acting_player=RED,
    )
    policy = build_policy("tactical_rule_based")
    assert build_context(make_request(unmoved, policy)).hidden_mover_adjacent[square("e5")] == 0
    assert build_context(make_request(moved, policy)).hidden_mover_adjacent[square("e5")] == 1


# ---------------------------------------------------------------------------
# Ladder behaviour
# ---------------------------------------------------------------------------


def scored(policy, state, action_id):
    """Score one specific action, for precise tier comparisons."""
    context = build_context(make_request(state, policy))
    move = next(item for item in context.moves if item.action_id == action_id)
    return policy.score(context, move)


def test_a_known_winning_capture_is_taken():
    state = make_position(
        red={"a1": "flag", "e5": "marshal", "a2": "scout"},
        blue={"j10": "flag", "e6": "captain", "j9": "scout"},
        revealed={"e6"},
        moved={"e6"},
        acting_player=RED,
    )
    capture = encode_action(square("e5"), square("e6"))
    for policy_id in ("basic_heuristic", "tactical_rule_based", "strategic_rule_based"):
        policy = build_policy(policy_id)
        assert policy.decide_checked(make_request(state, policy)).selected_action_id == capture


def test_a_known_losing_capture_is_refused():
    state = make_position(
        red={"a1": "flag", "e5": "scout", "a2": "sergeant"},
        blue={"j10": "flag", "e6": "marshal", "j9": "scout"},
        revealed={"e6"},
        moved={"e6"},
        acting_player=RED,
    )
    suicide = encode_action(square("e5"), square("e6"))
    for policy_id in ("basic_heuristic", "tactical_rule_based", "strategic_rule_based"):
        policy = build_policy(policy_id)
        assert policy.decide_checked(make_request(state, policy)).selected_action_id != suicide


def test_a_miner_is_sent_at_a_bomb_the_observer_has_seen():
    state = make_position(
        red={"a1": "flag", "e5": "miner", "a2": "scout"},
        blue={"j10": "flag", "e6": "bomb", "j9": "scout"},
        revealed={"e6"},
        acting_player=RED,
    )
    demolition = encode_action(square("e5"), square("e6"))
    for policy_id in ("tactical_rule_based", "strategic_rule_based"):
        policy = build_policy(policy_id)
        result = policy.decide_checked(make_request(state, policy))
        assert result.selected_action_id == demolition
        assert result.diagnostics["rule"] == "miner_demolition"


def test_a_piece_standing_next_to_my_own_flag_is_removed_first():
    """Uses only my own Flag's location, which I always legally know."""
    state = make_position(
        red={"a1": "flag", "b2": "sergeant", "e4": "captain"},
        blue={"j10": "flag", "a2": "lieutenant", "j9": "scout"},
        moved={"a2"},
        acting_player=RED,
    )
    defence = encode_action(square("b2"), square("a2"))
    for policy_id in ("tactical_rule_based", "strategic_rule_based"):
        policy = build_policy(policy_id)
        result = policy.decide_checked(make_request(state, policy))
        assert result.selected_action_id == defence
        assert result.diagnostics["rule"] == "flag_defence"


def test_basic_walks_into_a_revealed_marshal_and_tactical_does_not():
    """The concrete difference between the two lowest scoring tiers."""
    state = make_position(
        red={"a1": "flag", "e4": "sergeant"},
        blue={"j10": "flag", "e6": "marshal"},
        revealed={"e6"},
        moved={"e6"},
        acting_player=RED,
    )
    suicidal_advance = encode_action(square("e4"), square("e5"))

    basic = build_policy("basic_heuristic")
    assert basic.decide_checked(make_request(state, basic)).selected_action_id == (
        suicidal_advance
    )

    for policy_id in ("tactical_rule_based", "strategic_rule_based"):
        policy = build_policy(policy_id)
        chosen = policy.decide_checked(make_request(state, policy)).selected_action_id
        assert chosen != suicidal_advance
        assert scored(policy, state, suicidal_advance).score < 0.0


def test_strategic_holds_its_miners_back_while_bombs_are_unresolved():
    """The clearest Strategic-only term: Miner scarcity beyond material value."""
    state = make_position(
        red={"a1": "flag", "e5": "miner", "a2": "scout"},
        # Two live, unrevealed Bombs, so the Miner really is the scarce answer
        # to something still on the board.
        blue={"j10": "flag", "e6": "captain", "j9": "scout", "h8": "bomb", "h7": "bomb"},
        moved={"e6"},
        acting_player=RED,
    )
    speculative = encode_action(square("e5"), square("e6"))
    tactical = scored(build_policy("tactical_rule_based"), state, speculative)
    strategic = scored(build_policy("strategic_rule_based"), state, speculative)
    assert strategic.score < tactical.score
    assert "miner_preservation" in dict(strategic.components)


def test_stepping_a_piece_straight_back_is_penalised():
    """The anti-shuffling term, read off the public 16-ply move window."""
    for weights in (BASIC_WEIGHTS, TACTICAL_WEIGHTS, STRATEGIC_WEIGHTS):
        assert weights.repetition > 0.0

    state = make_position(
        red={"a1": "flag", "e5": "captain"},
        blue={"j10": "flag", "j9": "scout"},
        moved={"e5"},
        acting_player=RED,
        total_moves=4,
    )
    mover = state.piece_at(square("e5"))
    state.recent_moves.append(
        RecentMove(
            ply=3,
            player=RED,
            piece_id=mover.piece_id,
            source=square("e4"),
            destination=square("e5"),
            destination_had_opponent=False,
            target_piece_id=None,
        )
    )

    policy = build_policy("basic_heuristic")
    context = build_context(make_request(state, policy))
    retreat = next(
        move for move in context.moves if move.destination == square("e4")
    )
    # d5 is a lake, so f5 is the sideways step out of e5.
    sidestep = next(
        move for move in context.moves if move.destination == square("f5")
    )
    assert context.repetition_penalty(retreat) >= 1.0
    assert context.repetition_penalty(sidestep) == 0.0
    assert policy.score(context, retreat).score < policy.score(context, sidestep).score


def test_the_tiers_are_a_nesting_not_four_unrelated_agents():
    """Strategic inherits Tactical, and each tier only adds weights."""
    assert issubclass(StrategicRuleBasedPolicy, TacticalRuleBasedPolicy)
    for field in ("known_risk", "hidden_risk", "evade", "support", "miner_bomb", "scout_probe"):
        assert getattr(BASIC_WEIGHTS, field) == 0.0
        assert getattr(TACTICAL_WEIGHTS, field) > 0.0
    for field in ("mobility", "miner_preservation", "exposure", "pressure", "flag_guard"):
        assert getattr(TACTICAL_WEIGHTS, field) == 0.0
        assert getattr(STRATEGIC_WEIGHTS, field) > 0.0


def test_every_strategic_only_weight_reaches_a_real_scoring_term():
    """A weight nobody reads is a silent hole in the design, not a dead line."""
    policy = build_policy("strategic_rule_based")
    fired: set[str] = set()
    for state in random_positions(30):
        context = build_context(make_request(state, policy))
        for move in context.moves:
            fired.update(name for name, _ in policy.score(context, move).components)
    for name in ("mobility", "miner_preservation", "exposure", "pressure"):
        assert name in fired, f"the {name} weight never produced a score component"


# ---------------------------------------------------------------------------
# Stress behaviour
# ---------------------------------------------------------------------------


def test_the_scout_rush_moves_scouts_far_more_than_the_ladder(bank):
    rush = profile_matchup("stress_scout_rush", "random_legal", bank)
    strategic = profile_matchup("strategic_rule_based", "random_legal", bank)
    rush_share = rush["type_scout"] / rush["moves"]
    strategic_share = strategic["type_scout"] / strategic["moves"]
    assert rush_share > 0.6
    assert rush_share > 2 * strategic_share


def test_the_berserker_attacks_far_more_than_the_draw_seeker(bank):
    berserker = profile_matchup("stress_berserker", "random_legal", bank)
    seeker = profile_matchup("stress_draw_seeker", "random_legal", bank)
    berserker_rate = berserker["attacks"] / berserker["moves"]
    seeker_rate = seeker["attacks"] / seeker["moves"]
    assert berserker_rate > 5 * max(seeker_rate, 1e-6)


def test_the_draw_seeker_almost_never_starts_a_fight(bank):
    seeker = profile_matchup("stress_draw_seeker", "random_legal", bank)
    assert seeker["attacks"] / seeker["moves"] < 0.02


def test_the_miner_rush_moves_miners_far_more_than_the_ladder(bank):
    rush = profile_matchup("stress_miner_rush", "random_legal", bank)
    strategic = profile_matchup("strategic_rule_based", "random_legal", bank)
    assert rush["type_miner"] / rush["moves"] > 3 * (
        strategic["type_miner"] / strategic["moves"]
    )


def test_the_information_miser_avoids_the_two_public_reveal_paths(bank):
    """Combat reveals both participants; a multi-square Scout move reveals the Scout."""
    miser = profile_matchup("stress_information_miser", "random_legal", bank)
    berserker = profile_matchup("stress_berserker", "random_legal", bank)
    assert miser["attacks"] / miser["moves"] < berserker["attacks"] / berserker["moves"]
    assert miser["scout_runs"] / miser["moves"] < 0.05


def test_chaos_is_not_just_uniform_random(bank):
    """A random *objective* is not the same thing as a random *move*.

    Both are compared against the same opponent so the difference cannot come
    from who they played. Chaos commits hard to whichever feature its weight
    vector favoured that ply, and `attack` is one of those features, so it
    initiates combat far more often than uniform sampling over the legal set.
    """
    chaos = profile_matchup("stress_chaos", "basic_heuristic", bank)
    uniform = profile_matchup("random_legal", "basic_heuristic", bank)
    chaos_attack_rate = chaos["attacks"] / chaos["moves"]
    uniform_attack_rate = uniform["attacks"] / uniform["moves"]
    assert chaos_attack_rate > 1.5 * uniform_attack_rate


@pytest.mark.parametrize("policy_id", list(STRESS_POLICY_IDS))
def test_every_stress_policy_still_takes_a_flag_capture(policy_id):
    """A stress opponent that declines a won game is noise, not a distribution."""
    state = make_position(
        red={"a1": "flag", "e5": "scout"},
        blue={"e6": "flag", "j9": "scout"},
        revealed={"e6"},
        acting_player=RED,
    )
    policy = build_policy(policy_id)
    result = policy.decide_checked(make_request(state, policy))
    assert result.selected_action_id == encode_action(square("e5"), square("e6"))
    assert result.diagnostics["rule"] == "flag_capture"
