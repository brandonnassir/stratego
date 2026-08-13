"""The 16 family contracts: identity, executability, invariance, serialization."""

import json

import pytest

from stratego.engine.setup import validate_setup
from stratego.setups.families import (
    FAMILY_BY_ID,
    FAMILY_BY_KEY,
    FAMILY_CONTRACT_VERSION,
    FAMILY_CONTRACTS,
    FAMILY_IDS,
    FAMILY_KEYS,
    AllOf,
    Clause,
    Condition,
    evaluate_family,
    families_document,
    family_contract,
)
from stratego.setups.identity import SetupLibraryError, reflect_canonical
from stratego.setups.mobility import setup_has_initial_mobility
from stratego.setups.traits import TRAIT_SCHEMA, compute_trait_vector

from .family_fixtures import build_fixture, build_negative_fixture

EXPECTED_FAMILY_TABLE = {
    "F00": "corner_flag_fortress",
    "F01": "near_corner_flag_fortress",
    "F02": "central_back_flag_fortress",
    "F03": "partially_bombed_flag",
    "F04": "lightly_defended_deceptive_flag",
    "F05": "false_fortress_bomb_decoy",
    "F06": "distributed_bomb_defense",
    "F07": "high_bomb_placement",
    "F08": "aggressive_high_rank_front",
    "F09": "conservative_high_rank_rear",
    "F10": "scout_forward_information",
    "F11": "scout_preservation",
    "F12": "miner_forward",
    "F13": "miner_preservation",
    "F14": "balanced_conventional",
    "F15": "irregular_high_entropy",
}


# ---------------------------------------------------------------------------
# Registry identity
# ---------------------------------------------------------------------------


def test_exactly_sixteen_families_with_the_fixed_ids_once_each():
    assert len(FAMILY_CONTRACTS) == 16
    assert FAMILY_IDS == tuple(f"F{index:02d}" for index in range(16))
    assert len(set(FAMILY_IDS)) == 16
    assert len(set(FAMILY_KEYS)) == 16


def test_family_keys_match_the_instruction_table_exactly():
    assert {c.family_id: c.key for c in FAMILY_CONTRACTS} == EXPECTED_FAMILY_TABLE


def test_lookup_by_id_and_key():
    for contract in FAMILY_CONTRACTS:
        assert FAMILY_BY_ID[contract.family_id] is contract
        assert FAMILY_BY_KEY[contract.key] is contract
        assert family_contract(contract.family_id) is contract
    with pytest.raises(SetupLibraryError):
        family_contract("F16")


def test_every_contract_is_fully_documented():
    for contract in FAMILY_CONTRACTS:
        assert contract.display_name
        assert contract.purpose
        assert contract.required  # at least one required clause
        assert contract.allowed_ranges
        assert contract.primary_diagnostics
        assert contract.secondary_expectations
        assert contract.reflection_invariance_rule
        assert contract.perturbation_invariants
        for clause in (*contract.required, *contract.forbidden):
            assert clause.name and clause.description


# ---------------------------------------------------------------------------
# Positive and negative fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_positive_fixture_is_legal_mobile_and_satisfies_its_family(family_id):
    fixture = build_fixture(family_id)
    validate_setup(fixture, 0)
    satisfied, violations = evaluate_family(family_id, fixture)
    assert satisfied, violations
    assert setup_has_initial_mobility(fixture)


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_negative_fixture_is_legal_but_violates_its_family(family_id):
    fixture = build_negative_fixture(family_id)
    validate_setup(fixture, 0)
    satisfied, violations = evaluate_family(family_id, fixture)
    assert not satisfied
    assert violations


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_family_membership_is_reflection_invariant(family_id):
    positive = build_fixture(family_id)
    negative = build_negative_fixture(family_id)
    assert evaluate_family(family_id, reflect_canonical(positive))[0]
    assert not evaluate_family(family_id, reflect_canonical(negative))[0]


def test_contracts_reference_only_reflection_invariant_traits():
    invariant = {f.name for f in TRAIT_SCHEMA if f.reflection_invariant}
    for contract in FAMILY_CONTRACTS:
        for trait in contract.referenced_traits():
            assert trait in invariant, (contract.family_id, trait)


# ---------------------------------------------------------------------------
# Clause algebra
# ---------------------------------------------------------------------------


def test_condition_operators():
    traits = {"flag_rank": 1}
    assert Condition("flag_rank", "==", 1).evaluate(traits)
    assert Condition("flag_rank", "!=", 0).evaluate(traits)
    assert Condition("flag_rank", ">=", 1).evaluate(traits)
    assert Condition("flag_rank", "<=", 1).evaluate(traits)
    assert Condition("flag_rank", ">", 0).evaluate(traits)
    assert Condition("flag_rank", "<", 2).evaluate(traits)
    assert not Condition("flag_rank", "==", 0).evaluate(traits)


def test_condition_rejects_unknown_trait_or_operator():
    with pytest.raises(SetupLibraryError):
        Condition("no_such_trait", "==", 0)
    with pytest.raises(SetupLibraryError):
        Condition("flag_rank", "~", 0)


def test_all_of_is_a_conjunction():
    expression = AllOf(
        (Condition("flag_rank", "==", 0), Condition("flag_orth_bomb_guards", ">=", 2))
    )
    assert expression.evaluate({"flag_rank": 0, "flag_orth_bomb_guards": 2})
    assert not expression.evaluate({"flag_rank": 0, "flag_orth_bomb_guards": 1})


def test_forbidden_clause_semantics_on_f15():
    contract = family_contract("F15")
    conventional = compute_trait_vector(build_fixture("F14"))
    satisfied, violations = contract.evaluate(conventional)
    assert not satisfied
    assert "forbidden:conventional_fortress_signature" in violations


def test_forbidden_violations_are_prefixed_and_required_are_not():
    contract = family_contract("F06")
    traits = compute_trait_vector(build_negative_fixture("F06"))
    _, violations = contract.evaluate(traits)
    assert any(name.startswith("forbidden:") for name in violations)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_families_document_round_trips_and_carries_every_definition():
    document = families_document()
    assert document["family_contract_version"] == FAMILY_CONTRACT_VERSION == "setup_family_v1"
    assert document["family_count"] == 16
    payload = json.dumps(document, sort_keys=True)
    assert json.loads(payload) == json.loads(json.dumps(document, sort_keys=True))
    for entry in document["families"]:
        assert entry["family_id"] in FAMILY_BY_ID
        for clause in (*entry["required"], *entry["forbidden"]):
            assert clause["name"]
            assert clause["description"]
            assert clause["expression"]
            assert clause["formula"]


def test_serialized_expressions_can_be_reevaluated_independently():
    # An auditor must be able to evaluate the artifact's JSON expressions
    # without trusting the Python clause objects.
    def evaluate_expression(expression: dict, traits: dict) -> bool:
        if "all_of" in expression:
            return all(evaluate_expression(term, traits) for term in expression["all_of"])
        actual = traits[expression["trait"]]
        op, value = expression["op"], expression["value"]
        return {
            "==": actual == value,
            "!=": actual != value,
            ">=": actual >= value,
            "<=": actual <= value,
            ">": actual > value,
            "<": actual < value,
        }[op]

    document = families_document()
    for entry in document["families"]:
        family_id = entry["family_id"]
        for canonical in (build_fixture(family_id), build_negative_fixture(family_id)):
            traits = compute_trait_vector(canonical)
            required_ok = all(
                evaluate_expression(clause["expression"], traits)
                for clause in entry["required"]
            )
            forbidden_hit = any(
                evaluate_expression(clause["expression"], traits)
                for clause in entry["forbidden"]
            )
            independent = required_ok and not forbidden_hit
            assert independent == evaluate_family(family_id, canonical)[0]


def test_every_family_defines_a_positive_and_negative_fixture():
    from .family_fixtures import FIXTURE_PLACEMENTS, NEGATIVE_MUTATIONS

    assert set(FIXTURE_PLACEMENTS) == set(FAMILY_IDS)
    assert set(NEGATIVE_MUTATIONS) == set(FAMILY_IDS)
