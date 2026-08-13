"""Library contract: versions, counts, splits, IDs, serialization, perturbation."""

import json
import re
from pathlib import Path

import pytest

from stratego.setups import contracts
from stratego.setups.contracts import (
    BASE_ENTRY_REQUIRED_FIELDS,
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_COUNT,
    PERTURBATION_INVARIANTS,
    PERTURBATION_MAX_HAMMING,
    PERTURBATION_MIN_HAMMING,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    TEST_PER_FAMILY,
    TEST_TOTAL,
    TRAIN_PER_FAMILY,
    TRAIN_TOTAL,
    VALIDATION_PER_FAMILY,
    VALIDATION_TOTAL,
    base_entry_json_line,
    base_setup_id,
    contract_document,
    isolated_rebuild_sample_indices,
    library_digest,
    parse_base_setup_id,
    split_for_base_index,
    validate_perturbation,
)
from stratego.setups.identity import CANONICAL_FILES, SetupLibraryError

from .family_fixtures import build_fixture, build_negative_fixture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Versions and fixed counts
# ---------------------------------------------------------------------------


def test_version_identifiers_are_the_required_v1_names():
    assert SETUP_GENERATOR_CONTRACT_VERSION == "setup_generator_contract_v1"
    assert SETUP_LIBRARY_VERSION == "setup_library_v1"
    assert SETUP_FAMILY_VERSION == "setup_family_v1"
    assert SETUP_TRAIT_VECTOR_VERSION == "setup_trait_vector_v1"


def test_fixed_library_counts_are_exact():
    assert BASE_SETUP_COUNT == 8000
    assert FAMILY_COUNT == 16
    assert BASES_PER_FAMILY == 500
    assert TRAIN_PER_FAMILY == 400
    assert VALIDATION_PER_FAMILY == 50
    assert TEST_PER_FAMILY == 50
    assert TRAIN_TOTAL == 6400
    assert VALIDATION_TOTAL == 800
    assert TEST_TOTAL == 800
    assert TRAIN_TOTAL + VALIDATION_TOTAL + TEST_TOTAL == BASE_SETUP_COUNT


def test_master_seed_is_frozen():
    assert DEFAULT_LIBRARY_MASTER_SEED == 20260813


# ---------------------------------------------------------------------------
# Split rule
# ---------------------------------------------------------------------------


def test_split_boundaries_are_exact():
    assert split_for_base_index(0) == SPLIT_TRAIN
    assert split_for_base_index(399) == SPLIT_TRAIN
    assert split_for_base_index(400) == SPLIT_VALIDATION
    assert split_for_base_index(449) == SPLIT_VALIDATION
    assert split_for_base_index(450) == SPLIT_TEST
    assert split_for_base_index(499) == SPLIT_TEST


def test_split_rule_produces_the_exact_family_quotas():
    splits = [split_for_base_index(index) for index in range(BASES_PER_FAMILY)]
    assert splits.count(SPLIT_TRAIN) == TRAIN_PER_FAMILY
    assert splits.count(SPLIT_VALIDATION) == VALIDATION_PER_FAMILY
    assert splits.count(SPLIT_TEST) == TEST_PER_FAMILY


@pytest.mark.parametrize("bad", [-1, 500, 1000])
def test_split_rule_rejects_out_of_range_indices(bad):
    with pytest.raises(SetupLibraryError):
        split_for_base_index(bad)


# ---------------------------------------------------------------------------
# Stable semantic identifiers
# ---------------------------------------------------------------------------


def test_base_setup_id_format_and_round_trip():
    identifier = base_setup_id("F07", 42)
    assert identifier == "setup_library_v1:F07:042"
    assert parse_base_setup_id(identifier) == ("setup_library_v1", "F07", 42)


def test_base_setup_ids_are_unique_across_the_whole_library_space():
    identifiers = {
        base_setup_id(f"F{family:02d}", index)
        for family in range(16)
        for index in range(0, 500, 25)
    }
    assert len(identifiers) == 16 * 20


@pytest.mark.parametrize(
    "bad",
    [
        "setup_library_v1:F07",
        "setup_library_v2:F07:042",
        "setup_library_v1:F16:042",
        "setup_library_v1:F07:5000",
        "setup_library_v1:F07:42",
        "setup_library_v1:F07:xyz",
    ],
)
def test_parse_rejects_malformed_identifiers(bad):
    with pytest.raises(SetupLibraryError):
        parse_base_setup_id(bad)


def test_base_setup_id_rejects_bad_inputs():
    with pytest.raises(SetupLibraryError):
        base_setup_id("F16", 0)
    with pytest.raises(SetupLibraryError):
        base_setup_id("F00", 500)


# ---------------------------------------------------------------------------
# Serialization and digest
# ---------------------------------------------------------------------------


def _minimal_entry() -> dict:
    return {field: 0 for field in BASE_ENTRY_REQUIRED_FIELDS}


def test_entry_line_is_canonical_json():
    line = base_entry_json_line(_minimal_entry())
    assert "\n" not in line
    assert json.loads(line) == _minimal_entry()
    assert line == json.dumps(_minimal_entry(), sort_keys=True, separators=(",", ":"))


def test_entry_line_rejects_missing_fields():
    entry = _minimal_entry()
    del entry["fingerprint"]
    with pytest.raises(SetupLibraryError):
        base_entry_json_line(entry)


def test_library_digest_is_deterministic_and_order_sensitive():
    pairs = [("id-a", "fp-a"), ("id-b", "fp-b")]
    digest = library_digest(pairs)
    assert digest == library_digest(list(pairs))
    assert digest != library_digest(list(reversed(pairs)))
    assert len(digest) == 64


# ---------------------------------------------------------------------------
# Isolated-rebuild sample
# ---------------------------------------------------------------------------


def test_isolated_rebuild_sample_covers_split_boundaries():
    indices = isolated_rebuild_sample_indices()
    assert len(indices) == 40
    assert len(set(indices)) == 40
    assert all(0 <= index < BASES_PER_FAMILY for index in indices)
    # head, both boundary crossings, and the tail
    for required in (0, 399, 400, 449, 450, 499):
        assert required in indices
    assert FAMILY_COUNT * len(indices) == 640


# ---------------------------------------------------------------------------
# Perturbation invariants
# ---------------------------------------------------------------------------


def test_perturbation_invariants_are_explicit():
    assert PERTURBATION_MIN_HAMMING == 2
    assert PERTURBATION_MAX_HAMMING == 12
    text = " ".join(PERTURBATION_INVARIANTS)
    for required_word in ("inventory", "split", "family", "Flag", "mobility", "seed"):
        assert required_word in text


def _swap(cells: list, a: "tuple[int, int]", b: "tuple[int, int]") -> None:
    index_a = a[0] * CANONICAL_FILES + a[1]
    index_b = b[0] * CANONICAL_FILES + b[1]
    cells[index_a], cells[index_b] = cells[index_b], cells[index_a]


def test_a_legal_perturbation_passes():
    base = build_fixture("F00")
    candidate = list(base)
    # Swap two filler pieces across cells no F00 clause references.
    _swap(candidate, (3, 2), (2, 8))
    if tuple(candidate) == base:  # identical filler types at both cells
        _swap(candidate, (3, 2), (0, 7))
    violations = validate_perturbation(base, tuple(candidate), "F00")
    assert violations == []


def test_identity_output_is_rejected_as_a_perturbation():
    base = build_fixture("F00")
    violations = validate_perturbation(base, base, "F00")
    assert any("below minimum" in violation for violation in violations)


def test_excessive_perturbation_is_rejected():
    base = build_fixture("F00")
    candidate = list(base)
    # Seven disjoint swaps of differing piece types: 14 changed cells.
    swaps = [
        ((2, 0), (2, 2)),  # Scout <-> Miner
        ((2, 1), (2, 3)),  # Scout <-> Bomb
        ((2, 4), (3, 0)),  # Major <-> Scout
        ((2, 5), (3, 1)),  # Major <-> Scout
        ((2, 6), (3, 4)),  # Major <-> Scout
        ((1, 5), (3, 5)),  # Colonel <-> Scout
        ((1, 6), (3, 8)),  # Colonel <-> Scout
    ]
    for cell_a, cell_b in swaps:
        _swap(candidate, cell_a, cell_b)
    violations = validate_perturbation(base, tuple(candidate), "F00")
    assert any("above maximum" in violation for violation in violations)


def test_moving_the_flag_is_rejected():
    base = build_fixture("F00")
    candidate = list(base)
    _swap(candidate, (0, 0), (0, 4))  # Flag corner <-> General
    violations = validate_perturbation(base, tuple(candidate), "F00")
    assert any("Flag moved" in violation for violation in violations)


def test_breaking_the_family_contract_is_rejected():
    base = build_fixture("F03")
    candidate = build_negative_fixture("F03")  # one swap: the guard walks away
    violations = validate_perturbation(base, candidate, "F03")
    assert any("clause violated" in violation for violation in violations)


def test_inventory_corruption_is_rejected():
    base = build_fixture("F00")
    candidate = list(base)
    candidate[5] = candidate[6]  # F00 (0,5) Bomb overwritten by the (0,6) Spy
    assert tuple(candidate) != base
    violations = validate_perturbation(base, tuple(candidate), "F00")
    assert any("inventory" in violation.lower() for violation in violations)


def test_stranding_perturbation_is_rejected():
    # F07 admits a family-valid stranded arrangement: all six Bombs on the
    # open front files. Swapping each fixture Bomb with the Scout on its
    # target cell moves 12 cells (the maximum), keeps the Flag, keeps
    # bomb_front2 >= 4, and strands the owner — so the only violation left is
    # the mobility invariant.
    base = build_fixture("F07")
    candidate = list(base)
    bomb_cells = [(2, 2), (2, 6), (3, 3), (3, 6), (2, 7), (0, 3)]
    front_targets = [(3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (3, 9)]
    for bomb, target in zip(bomb_cells, front_targets):
        _swap(candidate, bomb, target)
    violations = validate_perturbation(base, tuple(candidate), "F07")
    assert violations == ["stranded: no initial legal move for the owner"]


def test_unknown_family_is_rejected():
    base = build_fixture("F00")
    with pytest.raises(SetupLibraryError):
        validate_perturbation(base, build_negative_fixture("F00"), "F16")


# ---------------------------------------------------------------------------
# Contract document and artifact consistency
# ---------------------------------------------------------------------------


def test_contract_document_carries_every_required_section():
    document = contract_document()
    for section in (
        "contract_version",
        "library_version",
        "family_contract_version",
        "trait_schema_version",
        "frozen_stack",
        "library_target",
        "canonical_representation",
        "reflection",
        "identity",
        "split_rule",
        "seeding",
        "quality",
        "serialization_contract",
        "isolated_rebuild_sample",
        "perturbation_invariants",
        "prohibitions",
        "family_contracts",
        "trait_schema",
    ):
        assert section in document, section
    assert document["frozen_stack"]["reference_engine"] == "phase2_1_reference_1.2.0"
    assert document["frozen_stack"]["rules_version"] == "stratego_project_v1"
    json.loads(json.dumps(document, sort_keys=True))


def test_committed_contract_artifact_matches_the_code():
    artifact = REPOSITORY_ROOT / "reports" / "phase_7_data" / "agent_01_setup_contract.json"
    assert artifact.exists(), "Agent 1 must freeze the contract artifact"
    payload = json.loads(artifact.read_text())
    assert payload["contract"] == json.loads(
        json.dumps(contract_document(), sort_keys=True)
    )


def test_committed_thresholds_artifact_matches_the_code():
    from stratego.setups.diversity import DIVERSITY_THRESHOLDS_V1

    artifact = (
        REPOSITORY_ROOT / "reports" / "phase_7_data" / "agent_01_diversity_thresholds.json"
    )
    assert artifact.exists(), "Agent 1 must freeze the thresholds artifact"
    payload = json.loads(artifact.read_text())
    assert payload["thresholds"] == json.loads(
        json.dumps(DIVERSITY_THRESHOLDS_V1.to_dict(), sort_keys=True)
    )


# ---------------------------------------------------------------------------
# No process-randomized hashing anywhere in the package
# ---------------------------------------------------------------------------


def test_setups_package_never_calls_builtin_hash():
    package_dir = REPOSITORY_ROOT / "stratego" / "setups"
    pattern = re.compile(r"(?<![\w.])hash\(")
    for source_file in sorted(package_dir.glob("*.py")):
        text = source_file.read_text()
        assert not pattern.search(text), source_file.name
