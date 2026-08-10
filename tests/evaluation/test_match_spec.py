"""Match identity and paired scheduling.

The contract under test: a match identifier implies the setups, colours, policy
seeds, first player and rules, and *nothing about how the schedule was executed*
may reach it.
"""

import dataclasses
import random

import pytest

from stratego.engine.constants import BLUE, EVALUATION_RULES, RED, TRAINING_RULES, RulesConfig
from stratego.evaluation.match_spec import (
    DEFAULT_ROOT_SEED,
    EVALUATION_SUITE_VERSION,
    MATCH_SPEC_VERSION,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    ROLE_CANDIDATE,
    ROLE_OPPONENT,
    MatchSpec,
    MatchSpecError,
    PairedUnit,
    build_paired_schedule,
    build_round_robin_schedule,
    derive_policy_seed,
    match_identity_components,
    rules_token,
    schedule_digest,
    schedule_matches,
    shard_schedule,
    sibling_match,
    validate_schedule,
)
from stratego.evaluation.policy import PolicyRef
from stratego.evaluation.setup_bank import SETUP_BANK_VERSION, SetupBank

CANDIDATE = PolicyRef("candidate_policy", "1.2.0")
OPPONENT = PolicyRef("opponent_policy", "0.9.1")
THIRD = PolicyRef("third_policy", "3.0.0")


@pytest.fixture(scope="module")
def bank() -> SetupBank:
    return SetupBank.generate(32)


def spec(**overrides) -> MatchSpec:
    fields = {
        "candidate": CANDIDATE,
        "opponent": OPPONENT,
        "setup_pair_id": 5,
        "candidate_color": RED,
        "replicate": 0,
    }
    fields.update(overrides)
    return MatchSpec(**fields)


# ---------------------------------------------------------------------------
# Identity determinism
# ---------------------------------------------------------------------------


def test_identical_components_give_identical_identifiers():
    assert spec().match_id == spec().match_id
    assert spec().paired_unit_id == spec().paired_unit_id
    assert spec() == spec()


def test_identifiers_are_stable_strings():
    match = spec()
    assert match.match_id.startswith("m-")
    assert match.paired_unit_id.startswith("u-")
    assert match.game_id == match.match_id
    # 12 digest bytes; enough that a league can never collide by accident.
    assert len(match.match_id) == len("m-") + 24


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate": PolicyRef("candidate_policy", "1.2.1")},
        {"candidate": PolicyRef("other_policy", "1.2.0")},
        {"opponent": THIRD},
        {"setup_pair_id": 6},
        {"candidate_color": BLUE},
        {"replicate": 1},
        {"root_seed": DEFAULT_ROOT_SEED + 1},
        {"suite_version": "phase4_evaluation_suite_v2"},
        {"setup_bank_version": "evaluation_setup_bank_v2"},
        {"rules": TRAINING_RULES},
    ],
)
def test_every_identity_component_changes_the_match_id(overrides):
    assert spec(**overrides).match_id != spec().match_id


#: `two_square_rule_enabled` and `continuous_chasing_rule_enabled` are excluded
#: from `stratego_project_v1` and `RulesConfig` refuses to construct with either
#: set, so they cannot be perturbed. They are checked textually instead.
_UNSETTABLE_RULE_FIELDS = ("two_square_rule_enabled", "continuous_chasing_rule_enabled")


def test_the_rules_configuration_is_fully_covered_by_its_token():
    """A match identifier must never silently mean two different rule sets."""
    token = rules_token(EVALUATION_RULES)
    covered = set()
    for field in dataclasses.fields(RulesConfig):
        if field.name in _UNSETTABLE_RULE_FIELDS:
            assert field.name in token
            covered.add(field.name)
            continue
        variant = dataclasses.replace(
            EVALUATION_RULES,
            **{field.name: _perturb(getattr(EVALUATION_RULES, field.name))},
        )
        assert rules_token(variant) != token, f"{field.name} is missing from the token"
        covered.add(field.name)
    assert covered == {field.name for field in dataclasses.fields(RulesConfig)}


def _perturb(value):
    if isinstance(value, int):
        return BLUE if value == RED else value + 1
    return f"{value}_variant"


def test_training_and_evaluation_rules_are_different_matches():
    assert spec(rules=EVALUATION_RULES).match_id != spec(rules=TRAINING_RULES).match_id


def test_policy_seeds_follow_from_the_match_identifier_alone():
    match = spec()
    assert match.candidate_seed == derive_policy_seed(match.match_id, ROLE_CANDIDATE)
    assert match.opponent_seed == derive_policy_seed(match.match_id, ROLE_OPPONENT)
    assert match.candidate_seed != match.opponent_seed


def test_policy_seeds_differ_between_the_two_games_of_a_unit():
    """Otherwise a deterministic-with-seed policy would replay itself."""
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=5)
    assert unit.game_a.candidate_seed != unit.game_b.candidate_seed
    assert unit.game_a.opponent_seed != unit.game_b.opponent_seed


def test_seed_and_reference_lookup_follow_the_colour_assignment():
    match = spec(candidate_color=BLUE)
    assert match.policy_seed_for(BLUE) == match.candidate_seed
    assert match.policy_seed_for(RED) == match.opponent_seed
    assert match.policy_ref_for(BLUE) == CANDIDATE
    assert match.policy_ref_for(RED) == OPPONENT
    assert match.role_for(BLUE) == ROLE_CANDIDATE
    assert match.role_for(RED) == ROLE_OPPONENT


def test_derive_policy_seed_rejects_an_unknown_role():
    with pytest.raises(MatchSpecError):
        derive_policy_seed("m-abc", "spectator")


def test_identity_components_are_exactly_the_documented_list():
    components = match_identity_components(spec())
    assert set(components) == {
        "match_spec_version",
        "suite_version",
        "pairing_mode",
        "candidate",
        "opponent",
        "setup_bank_version",
        "setup_pair_id",
        "candidate_color",
        "replicate",
        "root_seed",
        "rules",
    }
    assert components["match_spec_version"] == MATCH_SPEC_VERSION
    assert components["suite_version"] == EVALUATION_SUITE_VERSION


# ---------------------------------------------------------------------------
# The paired evaluation unit
# ---------------------------------------------------------------------------


def test_a_paired_unit_is_one_red_and_one_blue_assignment():
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=5)
    game_a, game_b = unit.matches
    assert game_a.candidate_color == RED
    assert game_b.candidate_color == BLUE
    assert game_a.opponent_color == BLUE
    assert game_b.opponent_color == RED
    assert game_a.match_id != game_b.match_id
    assert game_a.paired_unit_id == game_b.paired_unit_id == unit.paired_unit_id


def test_the_pairing_holds_the_board_fixed_and_swaps_the_controllers(bank: SetupBank):
    """The defining property of `color_swap_same_board`.

    A setup tuple is stored in each player's own square order, and those orders
    are not symmetric, so the pairing deliberately does not transform the board.
    """
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=5, setup_bank_version=bank.bank_version)
    game_a, game_b = unit.matches
    assert game_a.resolve_setups(bank) == game_b.resolve_setups(bank)
    assert game_a.pairing_mode == PAIRING_COLOR_SWAP_SAME_BOARD


def test_each_policy_moves_first_exactly_once_in_a_unit():
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=5)
    game_a, game_b = unit.matches
    assert game_a.first_player == game_b.first_player == EVALUATION_RULES.first_player
    assert game_a.candidate_moves_first
    assert not game_b.candidate_moves_first


def test_a_unit_reconstructs_exactly_from_either_of_its_matches():
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=5, replicate=2)
    for match in unit.matches:
        assert PairedUnit.from_match(match) == unit
        assert PairedUnit.from_match(match).matches == unit.matches


def test_sibling_match_is_an_involution():
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=5)
    for match in unit.matches:
        assert sibling_match(match).candidate_color == match.opponent_color
        assert sibling_match(sibling_match(match)) == match


# ---------------------------------------------------------------------------
# Execution independence
# ---------------------------------------------------------------------------


def test_shuffling_a_schedule_changes_no_specification():
    units = build_paired_schedule(CANDIDATE, OPPONENT, range(16), replicates=2)
    matches = schedule_matches(units)
    shuffled = list(matches)
    random.Random(20260401).shuffle(shuffled)

    assert shuffled != list(matches)
    assert sorted(shuffled, key=lambda item: item.match_id) == sorted(
        matches, key=lambda item: item.match_id
    )
    assert schedule_digest(shuffled) == schedule_digest(matches)


def test_worker_count_does_not_enter_a_match_identity():
    matches = schedule_matches(build_paired_schedule(CANDIDATE, OPPONENT, range(12)))
    reference = schedule_digest(matches)
    for worker_count in (1, 2, 3, 4, 8, 16):
        shards = shard_schedule(matches, worker_count)
        assert sum(len(shard) for shard in shards) == len(matches)
        rejoined = [match for shard in shards for match in shard]
        assert schedule_digest(rejoined) == reference
        assert {match.match_id for match in rejoined} == {
            match.match_id for match in matches
        }


def test_sharding_rejects_a_non_positive_worker_count():
    with pytest.raises(MatchSpecError):
        shard_schedule(schedule_matches(build_paired_schedule(CANDIDATE, OPPONENT, [0])), 0)


def test_regenerating_a_schedule_reproduces_it_exactly():
    first = schedule_matches(build_paired_schedule(CANDIDATE, OPPONENT, range(8), replicates=3))
    second = schedule_matches(build_paired_schedule(CANDIDATE, OPPONENT, range(8), replicates=3))
    assert first == second
    assert [match.candidate_seed for match in first] == [
        match.candidate_seed for match in second
    ]


def test_schedule_digest_is_sensitive_to_contents():
    base = schedule_matches(build_paired_schedule(CANDIDATE, OPPONENT, range(8)))
    extended = schedule_matches(build_paired_schedule(CANDIDATE, OPPONENT, range(9)))
    assert schedule_digest(base) != schedule_digest(extended)


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------


def test_paired_schedule_covers_every_pair_and_replicate():
    units = build_paired_schedule(CANDIDATE, OPPONENT, range(10), replicates=3)
    assert len(units) == 30
    assert len({unit.paired_unit_id for unit in units}) == 30
    matches = schedule_matches(units)
    assert len(matches) == 60
    assert len({match.match_id for match in matches}) == 60


def test_round_robin_covers_each_unordered_policy_pair_once():
    policies = [CANDIDATE, OPPONENT, THIRD]
    units = build_round_robin_schedule(policies, range(4))
    assert len(units) == 3 * 4
    matchups = {(unit.candidate.token, unit.opponent.token) for unit in units}
    assert len(matchups) == 3
    for left, right in matchups:
        assert (right, left) not in matchups


def test_round_robin_rejects_a_duplicated_policy():
    with pytest.raises(MatchSpecError):
        build_round_robin_schedule([CANDIDATE, CANDIDATE], range(2))


def test_a_policy_cannot_be_scheduled_against_itself():
    with pytest.raises(MatchSpecError):
        build_paired_schedule(CANDIDATE, CANDIDATE, range(2))


def test_replicates_must_be_at_least_one():
    with pytest.raises(MatchSpecError):
        build_paired_schedule(CANDIDATE, OPPONENT, range(2), replicates=0)


def test_validate_schedule_accepts_a_well_formed_schedule(bank: SetupBank):
    units = build_paired_schedule(
        CANDIDATE, OPPONENT, bank.pair_ids, setup_bank_version=bank.bank_version
    )
    assert validate_schedule(schedule_matches(units), bank) == []


def test_validate_schedule_reports_an_unpaired_unit():
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=1)
    problems = validate_schedule([unit.game_a])
    assert problems and "expected exactly one red and one blue" in problems[0]


def test_validate_schedule_reports_a_duplicated_match():
    unit = PairedUnit(CANDIDATE, OPPONENT, setup_pair_id=1)
    problems = validate_schedule([unit.game_a, unit.game_a, unit.game_b])
    assert any("duplicate match_id" in problem for problem in problems)


def test_validate_schedule_reports_a_missing_setup_pair(bank: SetupBank):
    unit = PairedUnit(
        CANDIDATE, OPPONENT, setup_pair_id=9_999, setup_bank_version=bank.bank_version
    )
    problems = validate_schedule(unit.matches, bank)
    assert any("9999" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Validation and serialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_color": 2},
        {"pairing_mode": "rotate_180"},
        {"setup_pair_id": -1},
        {"replicate": -1},
    ],
)
def test_malformed_specifications_are_rejected(overrides):
    with pytest.raises(MatchSpecError):
        spec(**overrides)


def test_resolve_setups_rejects_a_bank_of_the_wrong_version(bank: SetupBank):
    match = spec(setup_bank_version="evaluation_setup_bank_v9")
    with pytest.raises(MatchSpecError):
        match.resolve_setups(bank)


def test_setup_bank_version_defaults_to_the_frozen_bank():
    assert spec().setup_bank_version == SETUP_BANK_VERSION


def test_specification_round_trips_through_a_dictionary():
    match = spec(candidate_color=BLUE, replicate=4)
    payload = match.to_dict()
    assert payload["candidate_color_name"] == "blue"
    assert MatchSpec.from_dict(payload) == match


def test_round_trip_detects_a_mismatched_rules_configuration():
    payload = spec().to_dict()
    with pytest.raises(MatchSpecError):
        MatchSpec.from_dict(payload, rules=TRAINING_RULES)


def test_round_trip_detects_a_tampered_identifier():
    payload = spec().to_dict()
    payload["setup_pair_id"] = 6
    with pytest.raises(MatchSpecError):
        MatchSpec.from_dict(payload)
