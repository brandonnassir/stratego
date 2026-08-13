"""Deterministic base-setup generator: plans, determinism, acceptance stack.

The properties pinned here are the ones the whole library rests on.

**Determinism from identity alone.** A base setup is a pure function of
`(contract, library, master seed, family, index)`. Two calls agree; a rebuild
in a fresh call agrees; a different master seed disagrees. If any of this
breaks, Agent 6's bit-for-bit regeneration and Phase 8's provenance both
become fiction.

**Enumeration independence.** Nothing about generating F03:100 may depend on
whether F00:000 was generated first, or on how many candidates were rejected
anywhere else. That is what makes `rebuild_base_setup` an isolated operation.

**The acceptance stack is Agent 1's, not the generator's.** Construction plans
only propose; engine validation, the frozen family predicate and the frozen
initial-mobility rule dispose. These tests check the disposal path with
crafted invalid, family-violating and stranded arrangements, because in
production those branches fire rarely and would otherwise go untested.
"""

import ast
import random
from pathlib import Path

import pytest

from stratego.engine.constants import BOMB, PIECE_COUNTS
from stratego.engine.setup import random_setup, validate_setup
from stratego.setups import (
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_IDS,
    GENERATOR_VERSION,
    MAX_ATTEMPTS_PER_BASE,
    REJECTION_REASONS,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    BaseSetupEntry,
    LibrarySeedContext,
    base_setup_id,
    canonical_class_representative,
    class_fingerprint,
    compute_trait_vector,
    construct_candidate,
    evaluate_family,
    family_plan,
    generate_base_setup,
    is_canonical_representative,
    plans_document,
    rebuild_base_setup,
    reflect_canonical,
    setup_has_initial_mobility,
    split_for_base_index,
)
from stratego.setups.generator import (
    FAMILY_PLANS,
    REJECTION_ENGINE_INVALID,
    REJECTION_FAMILY_PREDICATE,
    REJECTION_STRANDED,
    BombPlan,
    FamilyPlan,
    FlagPlan,
    GroupPlan,
    _reject_reason,
)
from stratego.setups.identity import SetupLibraryError, canonical_index
from stratego.setups.seed import DEFAULT_SEED_CONTEXT, SEED_CONTEXT_VERSION
from stratego.setups.traits import OPEN_FRONT_FILES

from .family_fixtures import build_fixture, build_negative_fixture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

#: Base indices exercised whenever a test walks every family: the head of
#: train plus both split boundaries, so split-dependent metadata is covered
#: without generating all 8,000 entries in a unit test.
SAMPLE_INDICES = (0, 1, 2, 399, 400, 449, 450, 499)


def _stranded_f07_setup() -> "tuple[int, ...]":
    """A legal, F07-satisfying setup with no initial legal move.

    Built from the F07 fixture by moving all six Bombs onto the six open
    front-rank files — the only files whose forward square is not a lake — so
    the arrangement keeps the official inventory and still satisfies F07
    (Flag in the back two ranks, >= 4 Bombs forward) while being stranded.
    That isolates the mobility rule as the single reason to reject it.
    """
    cells = list(build_fixture("F07"))
    front = [canonical_index(3, file) for file in OPEN_FRONT_FILES]
    for target in front:
        if cells[target] != BOMB:
            donor = next(
                index
                for index, piece in enumerate(cells)
                if piece == BOMB and index not in front
            )
            cells[donor], cells[target] = cells[target], cells[donor]
    assert all(cells[target] == BOMB for target in front)
    return tuple(cells)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class TestFamilyPlans:
    def test_exactly_one_plan_per_frozen_family_in_order(self):
        assert tuple(plan.family_id for plan in FAMILY_PLANS) == FAMILY_IDS

    def test_every_plan_is_reachable_by_family_id(self):
        for family_id in FAMILY_IDS:
            assert family_plan(family_id).family_id == family_id

    def test_an_unknown_family_id_is_rejected(self):
        with pytest.raises(SetupLibraryError):
            family_plan("F99")

    def test_a_plan_that_forbids_every_flag_rank_is_rejected(self):
        with pytest.raises(SetupLibraryError):
            FamilyPlan(
                family_id="F00",
                flag=FlagPlan(rank_weights=(0.0, 0.0, 0.0, 0.0)),
                bombs=BombPlan(),
                groups=(),
            )

    def test_a_bomb_front_half_quota_cannot_be_combined_with_pinned_guards(self):
        # The quota partitions all six Bombs, so a pinned guard would make the
        # partition arithmetic silently wrong rather than loudly impossible.
        with pytest.raises(SetupLibraryError):
            FamilyPlan(
                family_id="F07",
                flag=FlagPlan(rank_weights=(1.0, 0.0, 0.0, 0.0)),
                bombs=BombPlan(guard_choices=(2,), front2_choices=(4,)),
                groups=(),
            )

    def test_duplicate_group_names_are_rejected(self):
        with pytest.raises(SetupLibraryError):
            FamilyPlan(
                family_id="F00",
                flag=FlagPlan(rank_weights=(1.0, 0.0, 0.0, 0.0)),
                bombs=BombPlan(),
                groups=(
                    GroupPlan("scouts", (1,)),
                    GroupPlan("scouts", (1,)),
                ),
            )

    def test_the_plans_document_is_machine_readable_and_complete(self):
        document = plans_document()
        assert document["generator_version"] == GENERATOR_VERSION
        assert document["max_attempts_per_base"] == MAX_ATTEMPTS_PER_BASE
        assert document["rejection_reasons"] == list(REJECTION_REASONS)
        assert document["acceptance_order"] == [
            "engine_validate_setup",
            "family_predicate",
            "initial_mobility",
        ]
        assert [plan["family_id"] for plan in document["plans"]] == list(FAMILY_IDS)
        for plan in document["plans"]:
            assert plan["rationale"], f"{plan['family_id']} plan states no rationale"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_identity_generates_an_identical_entry(self):
        for family_id in FAMILY_IDS:
            first = generate_base_setup(family_id, 7).entry
            second = generate_base_setup(family_id, 7).entry
            assert first.to_dict() == second.to_dict()

    def test_construction_is_a_pure_function_of_its_stream(self):
        for family_id in FAMILY_IDS:
            plan = family_plan(family_id)
            first = construct_candidate(plan, random.Random(99))
            second = construct_candidate(plan, random.Random(99))
            assert first == second

    def test_the_default_seed_context_carries_the_frozen_master_seed(self):
        assert DEFAULT_SEED_CONTEXT.master_seed == DEFAULT_LIBRARY_MASTER_SEED
        assert DEFAULT_SEED_CONTEXT.contract_version == SETUP_GENERATOR_CONTRACT_VERSION
        assert DEFAULT_SEED_CONTEXT.library_version == SETUP_LIBRARY_VERSION
        assert DEFAULT_SEED_CONTEXT.to_dict()["seed_context_version"] == SEED_CONTEXT_VERSION

    def test_a_different_master_seed_changes_every_sampled_base(self):
        alternative = LibrarySeedContext(master_seed=DEFAULT_LIBRARY_MASTER_SEED + 1)
        for family_id in FAMILY_IDS:
            for base_index in (0, 250, 499):
                production = rebuild_base_setup(family_id, base_index)
                changed = rebuild_base_setup(family_id, base_index, alternative)
                assert production.fingerprint != changed.fingerprint
                assert production.generation_seed != changed.generation_seed
                # Identity and split are seed-independent by contract.
                assert production.base_setup_id == changed.base_setup_id
                assert production.split == changed.split

    def test_consecutive_base_indices_receive_unrelated_streams(self):
        seeds = [DEFAULT_SEED_CONTEXT.base_seed("F00", index) for index in range(64)]
        assert len(set(seeds)) == len(seeds)
        assert all(abs(seeds[i + 1] - seeds[i]) > 1 for i in range(len(seeds) - 1))

    def test_attempt_streams_are_distinct_within_a_base(self):
        seeds = [DEFAULT_SEED_CONTEXT.attempt_seed("F09", 42, attempt) for attempt in range(32)]
        assert len(set(seeds)) == len(seeds)

    def test_a_non_integer_master_seed_is_rejected(self):
        with pytest.raises(SetupLibraryError):
            LibrarySeedContext(master_seed="20260813")  # type: ignore[arg-type]

    def test_seed_context_rejects_unknown_identities(self):
        with pytest.raises(SetupLibraryError):
            DEFAULT_SEED_CONTEXT.base_seed("F16", 0)
        with pytest.raises(SetupLibraryError):
            DEFAULT_SEED_CONTEXT.base_seed("F00", BASES_PER_FAMILY)


# ---------------------------------------------------------------------------
# Isolated rebuild
# ---------------------------------------------------------------------------


class TestIsolatedRebuild:
    def test_rebuild_matches_generation_for_every_family_and_boundary(self):
        for family_id in FAMILY_IDS:
            for base_index in SAMPLE_INDICES:
                generated = generate_base_setup(family_id, base_index).entry
                assert rebuild_base_setup(family_id, base_index).to_dict() == generated.to_dict()

    def test_a_late_index_rebuilds_without_generating_its_predecessors(self):
        # A fresh process would call exactly this and nothing else; the test
        # asserts the call is self-contained by comparing against the entry a
        # second independent call produces.
        entry = rebuild_base_setup("F11", 499)
        assert entry.base_setup_id == base_setup_id("F11", 499)
        assert entry.split == split_for_base_index(499)
        assert entry.to_dict() == rebuild_base_setup("F11", 499).to_dict()

    def test_entries_round_trip_through_their_serialized_form(self):
        for family_id in FAMILY_IDS:
            entry = rebuild_base_setup(family_id, 123)
            assert BaseSetupEntry.from_dict(entry.to_dict()).to_dict() == entry.to_dict()

    def test_an_out_of_range_identity_is_rejected(self):
        with pytest.raises(SetupLibraryError):
            generate_base_setup("F00", -1)
        with pytest.raises(SetupLibraryError):
            generate_base_setup("F00", BASES_PER_FAMILY)
        with pytest.raises(SetupLibraryError):
            generate_base_setup("nope", 0)


# ---------------------------------------------------------------------------
# The frozen acceptance stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    """One accepted entry per family at each boundary index (128 entries)."""
    return [
        generate_base_setup(family_id, base_index).entry
        for family_id in FAMILY_IDS
        for base_index in SAMPLE_INDICES
    ]


class TestAcceptedBases:
    def test_every_accepted_base_is_engine_valid_with_the_official_inventory(self, sample):
        for entry in sample:
            validated = validate_setup(entry.canonical_setup, 0)
            assert len(validated) == 40
            for piece_type, required in PIECE_COUNTS.items():
                assert validated.count(piece_type) == required

    def test_every_accepted_base_is_initially_mobile(self, sample):
        for entry in sample:
            assert setup_has_initial_mobility(entry.canonical_setup), entry.base_setup_id

    def test_every_accepted_base_satisfies_its_primary_family_contract(self, sample):
        for entry in sample:
            satisfied, violations = evaluate_family(entry.family_id, entry.canonical_setup)
            assert satisfied, f"{entry.base_setup_id}: {violations}"

    def test_every_stored_setup_is_the_canonical_class_representative(self, sample):
        for entry in sample:
            assert is_canonical_representative(entry.canonical_setup)

    def test_reflection_canonicalizes_back_to_the_stored_base(self, sample):
        for entry in sample:
            setup = entry.canonical_setup
            assert canonical_class_representative(setup) == setup
            assert canonical_class_representative(reflect_canonical(setup)) == setup
            assert class_fingerprint(reflect_canonical(setup)) == entry.fingerprint

    def test_identity_split_and_versions_follow_the_frozen_rules(self, sample):
        for entry in sample:
            assert entry.base_setup_id == base_setup_id(entry.family_id, entry.base_index)
            assert entry.split == split_for_base_index(entry.base_index)
            assert entry.library_version == SETUP_LIBRARY_VERSION
            assert entry.contract_version == SETUP_GENERATOR_CONTRACT_VERSION
            assert entry.family_contract_version == SETUP_FAMILY_VERSION
            assert entry.trait_schema_version == SETUP_TRAIT_VECTOR_VERSION
            assert entry.generator_version == GENERATOR_VERSION
            assert entry.master_seed == DEFAULT_LIBRARY_MASTER_SEED

    def test_recorded_fingerprints_and_traits_match_recomputation(self, sample):
        for entry in sample:
            setup = entry.canonical_setup
            assert entry.fingerprint == class_fingerprint(setup)
            assert entry.trait_vector == compute_trait_vector(setup)
            assert entry.content_fingerprint != entry.reflected_content_fingerprint

    def test_the_accepted_attempt_identity_is_recorded(self, sample):
        for entry in sample:
            assert entry.generation_attempts >= 1
            assert entry.accepted_attempt_index == entry.generation_attempts - 1
            assert entry.accepted_attempt_seed == DEFAULT_SEED_CONTEXT.attempt_seed(
                entry.family_id, entry.base_index, entry.accepted_attempt_index
            )
            assert entry.generation_seed == DEFAULT_SEED_CONTEXT.base_seed(
                entry.family_id, entry.base_index
            )

    def test_families_are_not_all_the_same_shape(self, sample):
        # A generator that ignored its plans would produce one distribution
        # wearing sixteen labels; the defining traits must actually differ.
        by_family = {}
        for entry in sample:
            by_family.setdefault(entry.family_id, []).append(entry.trait_vector)
        assert all(vector["flag_rank"] == 0 for vector in by_family["F00"])
        assert all(vector["flag_edge_distance"] == 0 for vector in by_family["F00"])
        assert all(vector["flag_orth_bomb_guards"] == 1 for vector in by_family["F03"])
        assert all(vector["flag_orth_bomb_guards"] == 0 for vector in by_family["F04"])
        assert all(vector["bomb_adjacent_pairs"] == 0 for vector in by_family["F06"])
        assert all(vector["bomb_front2_count"] >= 4 for vector in by_family["F07"])
        assert all(vector["marshal_rank"] >= 2 for vector in by_family["F08"])
        assert all(vector["marshal_rank"] <= 1 for vector in by_family["F09"])
        assert all(vector["scout_front_rank_count"] >= 3 for vector in by_family["F10"])
        assert all(vector["scout_front_rank_count"] == 0 for vector in by_family["F11"])
        assert all(vector["miner_front2_count"] >= 3 for vector in by_family["F12"])
        assert all(vector["miner_front_rank_count"] == 0 for vector in by_family["F13"])
        assert all(vector["unconventional_feature_count"] >= 2 for vector in by_family["F15"])


# ---------------------------------------------------------------------------
# Rejection accounting
# ---------------------------------------------------------------------------


class TestRejection:
    def test_an_illegal_inventory_is_rejected_as_engine_invalid(self):
        broken = list(build_fixture("F00"))
        broken[0] = BOMB  # seven Bombs, no Flag
        reason, _violations = _reject_reason(tuple(broken), "F00")
        assert reason == REJECTION_ENGINE_INVALID

    def test_a_family_violating_candidate_is_rejected_by_the_frozen_predicate(self):
        for family_id in FAMILY_IDS:
            reason, violations = _reject_reason(build_negative_fixture(family_id), family_id)
            assert reason == REJECTION_FAMILY_PREDICATE
            assert violations

    def test_a_stranded_candidate_is_rejected_even_when_legal(self):
        stranded = _stranded_f07_setup()
        validate_setup(stranded, 0)  # legal under the frozen engine
        satisfied, violations = evaluate_family("F07", stranded)
        assert satisfied, violations  # mobility is the only remaining objection
        assert not setup_has_initial_mobility(stranded)
        assert _reject_reason(stranded, "F07")[0] == REJECTION_STRANDED

    def test_an_accepted_candidate_reports_no_rejection_reason(self):
        for family_id in FAMILY_IDS:
            assert _reject_reason(build_fixture(family_id), family_id) == (None, [])

    def test_rejections_are_counted_by_reason_and_never_hidden(self):
        seen = set()
        for family_id in FAMILY_IDS:
            for base_index in range(40):
                record = generate_base_setup(family_id, base_index)
                seen.update(record.rejections)
                assert sum(record.rejections.values()) == (
                    record.entry.generation_attempts - 1
                )
        assert seen <= set(REJECTION_REASONS)

    def test_the_attempt_budget_is_finite_and_declared(self):
        assert MAX_ATTEMPTS_PER_BASE >= 16
        assert plans_document()["max_attempts_per_base"] == MAX_ATTEMPTS_PER_BASE


# ---------------------------------------------------------------------------
# No outcome or strength signal
# ---------------------------------------------------------------------------


def _module_identifiers(relative_path: str) -> "set[str]":
    tree = ast.parse((REPOSITORY_ROOT / relative_path).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


#: Tokens that would betray an outcome, strength or learned signal in code.
#: Deliberately checked against *identifiers* rather than raw text, so the
#: modules can keep documenting the prohibition in prose.
FORBIDDEN_CODE_TOKENS = (
    "winrate",
    "win_rate",
    "elo",
    "outcome",
    "reward",
    "strength",
    "policy",
    "score",
    "rating",
    "preference",
    "baseline_win",
)


@pytest.mark.parametrize(
    "module_path",
    [
        "stratego/setups/generator.py",
        "stratego/setups/library.py",
        "stratego/setups/seed.py",
    ],
)
def test_no_outcome_or_strength_signal_participates_in_generation(module_path):
    identifiers = _module_identifiers(module_path)
    offenders = sorted(
        name
        for name in identifiers
        if any(token in name.lower() for token in FORBIDDEN_CODE_TOKENS)
    )
    assert offenders == [], f"{module_path} names an outcome/strength signal: {offenders}"


def test_generation_never_consumes_global_rng_state():
    # Seeding the global stream differently must not move a single setup.
    random.seed(1)
    first = rebuild_base_setup("F05", 300).to_dict()
    random.seed(2)
    [random.random() for _ in range(100)]
    assert rebuild_base_setup("F05", 300).to_dict() == first
