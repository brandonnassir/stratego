"""Constrained perturbation: invariants, determinism, and the rejection path.

The properties pinned here are the ones every procedural descendant rests on.

**Structural invariants come from construction, not from repair.** A
perturbation is a sequence of disjoint swaps between cells holding different
piece types, with the Flag cell excluded. That alone forces exact inventory
preservation, a fixed Flag, and canonical Hamming distance exactly `2k`. These
tests check the property directly rather than trusting the frozen validator to
notice a violation after the fact.

**Acceptance is Agent 1's, not this module's.** Operators only propose;
`validate_perturbation` disposes. The rejection branches — family-breaking and
stranded candidates — fire rarely in production, so they are exercised here
with crafted arrangements.

**Determinism through the retry loop.** A perturbation identity must
reproduce not only its accepted descendant but the attempts and rejections it
cost, because that is what makes provenance rebuild exact.

**The identity is the seed, alone.** Agent 1 froze the descendant as a pure
function of `(base_setup_id, sampler_version, perturbation_seed)`, so the
production entry point takes exactly those inputs: the swap count is decoded
from the composite `seed_encoding_v1` seed, the operator mix and retry budget
are version constants, and the internal knobs exist only behind the private
diagnostic entry these tests use to force rare branches.
"""

import ast
import inspect
import random
from pathlib import Path

import pytest

from stratego.engine.constants import (
    BOMB,
    FLAG,
    IMMOVABLE_TYPES,
    LIEUTENANT,
    PIECE_COUNTS,
    SCOUT,
    SERGEANT,
)
from stratego.engine.setup import validate_setup
from stratego.setups import (
    FAMILY_IDS,
    MAX_PERTURBATION_ATTEMPTS,
    MAX_SWAP_COUNT,
    MIN_SWAP_COUNT,
    OPERATOR_NAMES,
    OPERATORS,
    PERTURBATION_SEED_ENCODING,
    PERTURBATION_VERSION,
    apply_swaps,
    compute_trait_vector,
    decode_perturbation_seed,
    encode_perturbation_seed,
    evaluate_family,
    operator_mix_document,
    perturb_setup,
    perturbation_id,
    reflect_canonical,
    setup_has_initial_mobility,
)
from stratego.setups.contracts import (
    PERTURBATION_MAX_HAMMING,
    PERTURBATION_MIN_HAMMING,
    validate_perturbation,
)
from stratego.setups.identity import (
    CANONICAL_FILES,
    SetupLibraryError,
    canonical_index,
    derive_stream_seed,
)
from stratego.setups.perturbation import (
    DEFAULT_OPERATOR_WEIGHTS,
    HAMMING_PER_SWAP,
    REJECTION_CONSTRUCTION,
    REJECTION_FAMILY_PREDICATE,
    REJECTION_REASONS,
    REJECTION_STRANDED,
    _classify,
    _perturb_setup_diagnostic,
    _propose_candidate,
)

from .family_fixtures import build_fixture, build_negative_fixture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent


def _seed(*parts) -> int:
    return derive_stream_seed("test_perturbation", *parts)


def _run(base, family_id: str, swap_count: int, *parts):
    """Perturb through the production identity API with a labelled test seed."""
    return perturb_setup(
        base, family_id, encode_perturbation_seed(swap_count, _seed(*parts))
    )


@pytest.fixture(scope="module")
def fixtures() -> dict:
    """One legal, mobile, contract-satisfying arrangement per family."""
    return {family_id: build_fixture(family_id) for family_id in FAMILY_IDS}


# ---------------------------------------------------------------------------
# The frozen contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_the_swap_window_matches_agent_1s_frozen_hamming_window(self):
        assert MIN_SWAP_COUNT * HAMMING_PER_SWAP == PERTURBATION_MIN_HAMMING
        assert MAX_SWAP_COUNT * HAMMING_PER_SWAP == PERTURBATION_MAX_HAMMING

    def test_operator_names_are_unique_and_weights_form_a_distribution(self):
        assert len(set(OPERATOR_NAMES)) == len(OPERATORS)
        assert len(DEFAULT_OPERATOR_WEIGHTS) == len(OPERATORS)
        assert all(weight > 0.0 for weight in DEFAULT_OPERATOR_WEIGHTS)
        assert abs(sum(DEFAULT_OPERATOR_WEIGHTS) - 1.0) < 1e-9

    def test_the_operator_set_covers_every_assigned_technique(self):
        techniques = {operator.technique for operator in OPERATORS}
        assert techniques == {
            "bounded within-rank swaps",
            "bounded cross-rank swaps",
            "local fortress variation",
            "controlled decoy variation",
            "controlled Scout relocation",
            "controlled Miner relocation",
            "role-compatible piece swaps",
        }

    def test_the_mix_document_restates_the_frozen_bounds(self):
        document = operator_mix_document()
        assert document["perturbation_version"] == PERTURBATION_VERSION
        assert document["seed_encoding"] == PERTURBATION_SEED_ENCODING
        assert document["min_hamming"] == PERTURBATION_MIN_HAMMING
        assert document["max_hamming"] == PERTURBATION_MAX_HAMMING
        assert document["max_attempts"] == MAX_PERTURBATION_ATTEMPTS
        assert document["max_attempts_status"] == (
            "version constant of setup_perturbation_v1"
        )
        assert document["rejection_reasons"] == list(REJECTION_REASONS)
        assert len(document["operators"]) == len(OPERATORS)

    def test_perturbation_ids_name_the_version_and_the_derived_swap_count(self):
        seed = encode_perturbation_seed(3, 255)
        identifier = perturbation_id(seed)
        assert identifier.startswith(f"{PERTURBATION_VERSION}:k3:")
        assert identifier.endswith(f"{seed:016x}")
        assert perturbation_id(encode_perturbation_seed(4, 255)) != identifier

    def test_the_production_identity_has_no_configurable_inputs(self):
        """Agent 1's frozen invariant: the descendant is a pure function of
        `(base_setup_id, sampler_version, perturbation_seed)`. The production
        signature therefore carries no swap count, no operator weights and no
        attempt budget — they are decoded or version constants."""
        assert list(inspect.signature(perturb_setup).parameters) == [
            "base_canonical",
            "family_id",
            "perturbation_seed",
        ]
        assert MAX_PERTURBATION_ATTEMPTS == 64

    def test_an_unknown_family_is_refused(self, fixtures):
        with pytest.raises(SetupLibraryError, match="unknown family"):
            perturb_setup(fixtures["F00"], "F99", encode_perturbation_seed(2, 1))

    def test_a_malformed_arrangement_is_refused(self):
        with pytest.raises(SetupLibraryError, match="canonical entries"):
            perturb_setup((0, 1, 2), "F00", encode_perturbation_seed(2, 1))


# ---------------------------------------------------------------------------
# The composite seed encoding
# ---------------------------------------------------------------------------


class TestSeedEncoding:
    def test_the_roundtrip_is_exact_over_the_whole_window(self):
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            for raw_seed in (0, 1, 255, 2**32, 2**62 - 1):
                seed = encode_perturbation_seed(swap_count, raw_seed)
                assert decode_perturbation_seed(seed) == (swap_count, raw_seed)

    def test_the_mapping_is_a_bijection(self):
        seeds = {
            encode_perturbation_seed(swap_count, raw_seed)
            for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1)
            for raw_seed in range(64)
        }
        assert len(seeds) == 6 * 64

    def test_the_swap_count_is_derived_from_the_seed_not_supplied(self, fixtures):
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            result = _run(fixtures["F14"], "F14", swap_count, "derived", swap_count)
            decoded_count, _raw = decode_perturbation_seed(result.perturbation_seed)
            assert decoded_count == result.swap_count == swap_count

    def test_invalid_low_bits_are_rejected(self):
        for invalid in (6, 7, (123 << 3) | 6, (123 << 3) | 7):
            with pytest.raises(SetupLibraryError, match="invalid"):
                decode_perturbation_seed(invalid)

    @pytest.mark.parametrize("swap_count", [0, -1, MAX_SWAP_COUNT + 1])
    def test_out_of_window_swap_counts_cannot_be_encoded(self, swap_count):
        with pytest.raises(SetupLibraryError, match="swap_count"):
            encode_perturbation_seed(swap_count, 1)

    def test_malformed_seeds_are_rejected(self):
        with pytest.raises(SetupLibraryError):
            decode_perturbation_seed(-1)
        with pytest.raises(SetupLibraryError):
            decode_perturbation_seed(1.5)  # type: ignore[arg-type]
        with pytest.raises(SetupLibraryError):
            decode_perturbation_seed(True)  # type: ignore[arg-type]
        with pytest.raises(SetupLibraryError):
            encode_perturbation_seed(2, -1)
        with pytest.raises(SetupLibraryError):
            encode_perturbation_seed(2, True)  # type: ignore[arg-type]

    def test_perturbing_with_an_invalid_encoding_is_refused(self, fixtures):
        with pytest.raises(SetupLibraryError, match="invalid"):
            perturb_setup(fixtures["F00"], "F00", (5 << 3) | 7)


# ---------------------------------------------------------------------------
# Structural invariants of an accepted descendant
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    @pytest.mark.parametrize("swap_count", range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1))
    def test_accepted_descendants_satisfy_every_frozen_invariant(
        self, fixtures, family_id, swap_count
    ):
        base = fixtures[family_id]
        for trial in range(4):
            result = _run(base, family_id, swap_count, family_id, swap_count, trial)
            assert result.accepted, f"{family_id} k={swap_count} exhausted the budget"
            assert validate_perturbation(base, result.canonical, family_id) == []

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_inventory_is_preserved_exactly(self, fixtures, family_id):
        base = fixtures[family_id]
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            result = _run(base, family_id, swap_count, "inv", family_id, swap_count)
            assert validate_setup(result.canonical, 0) == result.canonical
            for piece_type, expected in PIECE_COUNTS.items():
                assert result.canonical.count(piece_type) == expected

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_the_flag_never_moves(self, fixtures, family_id):
        base = fixtures[family_id]
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            result = _run(base, family_id, swap_count, "flag", family_id, swap_count)
            assert result.canonical.index(FLAG) == base.index(FLAG)

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_hamming_distance_is_exactly_two_per_swap(self, fixtures, family_id):
        base = fixtures[family_id]
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            result = _run(base, family_id, swap_count, "h", family_id, swap_count)
            assert result.hamming_from_base == HAMMING_PER_SWAP * swap_count
            assert PERTURBATION_MIN_HAMMING <= result.hamming_from_base <= PERTURBATION_MAX_HAMMING

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_the_primary_family_still_holds(self, fixtures, family_id):
        base = fixtures[family_id]
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            result = _run(base, family_id, swap_count, "f", family_id, swap_count)
            satisfied, violations = evaluate_family(family_id, result.canonical)
            assert satisfied, violations

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_descendants_are_never_stranded(self, fixtures, family_id):
        base = fixtures[family_id]
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            result = _run(base, family_id, swap_count, "s", family_id, swap_count)
            assert setup_has_initial_mobility(result.canonical)

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_family_membership_survives_reflection(self, fixtures, family_id):
        """Family membership is a property of the reflection class, so a
        descendant and its mirror must agree — the sampler applies reflection
        after perturbation and revalidates only the final orientation."""
        base = fixtures[family_id]
        result = _run(base, family_id, 3, "refl", family_id)
        mirrored = reflect_canonical(result.canonical)
        assert evaluate_family(family_id, mirrored)[0]
        assert setup_has_initial_mobility(mirrored)
        assert reflect_canonical(mirrored) == result.canonical

    def test_swaps_are_disjoint_and_are_recorded_exactly(self, fixtures):
        for family_id in FAMILY_IDS:
            result = _run(fixtures[family_id], family_id, MAX_SWAP_COUNT, "swaps", family_id)
            cells = [cell for swap in result.swaps for cell in swap]
            assert len(cells) == len(set(cells)), "a cell was touched twice"
            assert result.canonical.index(FLAG) not in cells
            assert apply_swaps(result.base_canonical, result.swaps) == result.canonical
            assert len(result.operators_applied) == len(result.swaps) == MAX_SWAP_COUNT
            assert set(result.operators_applied) <= set(OPERATOR_NAMES)

    def test_every_proposal_swaps_two_different_piece_types(self, fixtures):
        base = fixtures["F14"]
        for trial in range(50):
            rng = random.Random(_seed("proposal", trial))
            candidate, _names, swaps = _propose_candidate(
                base, MAX_SWAP_COUNT, rng, DEFAULT_OPERATOR_WEIGHTS
            )
            working = base
            for left, right in swaps:
                assert working[left] != working[right]
                working = apply_swaps(working, ((left, right),))
            assert working == candidate


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_identity_reproduces_everything(self, fixtures):
        for family_id in FAMILY_IDS:
            for swap_count in (1, 3, 6):
                seed = encode_perturbation_seed(
                    swap_count, _seed("det", family_id, swap_count)
                )
                first = perturb_setup(fixtures[family_id], family_id, seed)
                second = perturb_setup(fixtures[family_id], family_id, seed)
                assert first == second
                assert first.to_dict() == second.to_dict()

    def test_rejection_and_retry_are_themselves_reproducible(self, fixtures):
        """The attempts a perturbation costs must reproduce, not just its result."""
        observed = []
        for family_id in FAMILY_IDS:
            for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
                seed = encode_perturbation_seed(
                    swap_count, _seed("retry", family_id, swap_count)
                )
                result = perturb_setup(fixtures[family_id], family_id, seed)
                repeat = perturb_setup(fixtures[family_id], family_id, seed)
                assert result.attempts == repeat.attempts
                assert result.rejections == repeat.rejections
                assert result.accepted_attempt_index == repeat.accepted_attempt_index
                observed.append(result)
        assert any(
            result.attempts > 1 for result in observed
        ), "no retry occurred, so the retry path was not exercised"

    def test_a_different_seed_generally_yields_a_different_descendant(self, fixtures):
        base = fixtures["F14"]
        descendants = {
            _run(base, "F14", 4, "spread", trial).canonical for trial in range(40)
        }
        assert len(descendants) >= 35

    def test_the_encoded_swap_count_changes_the_descendant(self, fixtures):
        """Same raw randomness, different encoded intensity: the composite
        seeds differ in their low bits alone, and the descendants differ."""
        raw_seed = _seed("k")
        canonical = {
            swap_count: perturb_setup(
                fixtures["F06"], "F06", encode_perturbation_seed(swap_count, raw_seed)
            ).canonical
            for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1)
        }
        assert len(set(canonical.values())) == MAX_SWAP_COUNT - MIN_SWAP_COUNT + 1

    def test_no_global_rng_state_is_consumed(self, fixtures):
        seed = encode_perturbation_seed(5, _seed("global"))
        random.seed(1)
        first = perturb_setup(fixtures["F09"], "F09", seed)
        random.seed(999999)
        [random.random() for _ in range(37)]
        second = perturb_setup(fixtures["F09"], "F09", seed)
        assert first == second

    def test_the_diagnostic_entry_matches_production_under_default_knobs(self, fixtures):
        """The private diagnostic exists to force rare branches, not to define
        a second semantics: with default knobs it is exactly the production
        function on the encoded seed."""
        raw_seed = _seed("diag")
        production = perturb_setup(
            fixtures["F12"], "F12", encode_perturbation_seed(4, raw_seed)
        )
        diagnostic = _perturb_setup_diagnostic(fixtures["F12"], "F12", raw_seed, 4)
        assert production == diagnostic

    def test_diagnostic_operator_weights_change_the_proposal_stream(self, fixtures):
        base = fixtures["F12"]
        raw_seed = _seed("weights")
        default = _perturb_setup_diagnostic(base, "F12", raw_seed, 4)
        scout_only = _perturb_setup_diagnostic(
            base, "F12", raw_seed, 4, weights=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        )
        assert set(scout_only.operators_applied) == {"scout_relocation"}
        assert default.canonical != scout_only.canonical

    def test_a_wrong_length_weight_vector_is_refused_by_the_diagnostic(self, fixtures):
        with pytest.raises(SetupLibraryError, match="operator weights"):
            _perturb_setup_diagnostic(fixtures["F00"], "F00", 1, 2, weights=(1.0, 1.0))


# ---------------------------------------------------------------------------
# The rejection path
# ---------------------------------------------------------------------------


def _stranded_setup() -> "tuple[int, ...]":
    """A legal but stranded arrangement: every open front file is immovable.

    The lakes sit on files 2, 3, 6 and 7, so a front-rank piece on any other
    file faces an empty square at ply 0. Filling files 0, 1, 4, 5, 8 and 9 of
    the front rank with the six Bombs leaves only lake-facing movable pieces,
    and the setup zone is full, so no legal move exists.
    """
    cells: list[int] = [SERGEANT] * 40
    for file in (0, 1, 4, 5, 8, 9):
        cells[canonical_index(3, file)] = BOMB
    cells[canonical_index(0, 0)] = FLAG
    remaining = [
        piece_type
        for piece_type, count in PIECE_COUNTS.items()
        for _ in range(count)
        if piece_type not in (BOMB, FLAG)
    ]
    free = [cell for cell in range(40) if cell not in {canonical_index(0, 0)} and cells[cell] != BOMB]
    assert len(free) == len(remaining)
    for cell, piece_type in zip(free, remaining):
        cells[cell] = piece_type
    setup = tuple(cells)
    validate_setup(setup, 0)
    return setup


class TestRejection:
    def test_the_crafted_stranded_arrangement_really_is_stranded(self):
        stranded = _stranded_setup()
        assert validate_setup(stranded, 0) == stranded
        assert not setup_has_initial_mobility(stranded)

    def test_a_stranded_candidate_is_rejected_and_classified(self):
        """The stranded branch fires rarely in production, so it is forced here."""
        stranded = _stranded_setup()
        base = apply_swaps(stranded, ((canonical_index(3, 0), canonical_index(0, 5)),))
        assert setup_has_initial_mobility(base)
        violations = validate_perturbation(base, stranded, "F15")
        assert any(violation.startswith("stranded") for violation in violations)
        assert _classify(["stranded: no initial legal move for the owner"]) == REJECTION_STRANDED

    @pytest.mark.parametrize("family_id", FAMILY_IDS)
    def test_a_family_breaking_candidate_is_rejected(self, fixtures, family_id):
        negative = build_negative_fixture(family_id)
        violations = validate_perturbation(fixtures[family_id], negative, family_id)
        assert violations, f"{family_id}: the negative fixture was accepted"
        assert any(
            violation.startswith(f"family {family_id} clause violated")
            or violation.startswith("hamming distance")
            or violation.startswith("the Flag moved")
            for violation in violations
        )

    def test_every_rejection_reason_is_classifiable(self):
        assert _classify(["inventory/legality: boom"]) in REJECTION_REASONS
        assert _classify(["hamming distance 14 above maximum 12"]) in REJECTION_REASONS
        assert _classify(["the Flag moved off its base cell"]) in REJECTION_REASONS
        assert _classify(["family F03 clause violated: x"]) == REJECTION_FAMILY_PREDICATE
        assert _classify(["stranded: no initial legal move for the owner"]) == REJECTION_STRANDED

    def test_an_unclassifiable_violation_raises_rather_than_being_swallowed(self):
        with pytest.raises(SetupLibraryError, match="unclassifiable"):
            _classify(["something nobody wrote"])

    def test_exhaustion_returns_the_unmodified_base_never_an_invalid_setup(self, fixtures):
        """A zero-attempt budget is refused; a one-attempt budget may exhaust,
        and when it does the base itself is returned, still legal and mobile."""
        with pytest.raises(SetupLibraryError, match="max_attempts"):
            _perturb_setup_diagnostic(fixtures["F00"], "F00", 1, 2, max_attempts=0)

        exhausted = None
        for trial in range(400):
            result = _perturb_setup_diagnostic(
                fixtures["F14"], "F14", _seed("exhaust", trial), 6, max_attempts=1
            )
            if not result.accepted:
                exhausted = result
                break
        assert exhausted is not None, "no single-attempt exhaustion occurred to check"
        assert exhausted.canonical == fixtures["F14"]
        assert exhausted.accepted_attempt_index is None
        assert exhausted.hamming_from_base == 0
        assert sum(exhausted.rejections.values()) == 1
        assert setup_has_initial_mobility(exhausted.canonical)
        assert evaluate_family("F14", exhausted.canonical)[0]

    def test_the_full_budget_never_exhausts_on_the_family_fixtures(self, fixtures):
        for family_id in FAMILY_IDS:
            for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
                result = _run(
                    fixtures[family_id], family_id, swap_count, "budget", family_id, swap_count
                )
                assert result.accepted
                assert result.attempts <= MAX_PERTURBATION_ATTEMPTS

    def test_an_infeasible_operator_is_counted_as_a_construction_rejection(self, fixtures):
        """A weight vector pinned to one narrow operator can run out of legal
        pairs; that must be a counted rejection, not an exception.

        F00 seals its corner Flag with two adjacent Bombs, which are inside the
        decoy operator's Manhattan-4 exclusion, so fewer than six Bombs are
        ever eligible and a six-swap decoy-only proposal cannot complete.
        """
        base = fixtures["F00"]
        result = _perturb_setup_diagnostic(
            base,
            "F00",
            _seed("infeasible"),
            MAX_SWAP_COUNT,
            weights=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            max_attempts=8,
        )
        assert result.rejections.get(REJECTION_CONSTRUCTION, 0) >= 1
        assert not result.accepted
        assert result.canonical == base


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


class TestNoOutcomeDependency:
    def test_no_identifier_names_an_outcome_or_strength_signal(self):
        names = _module_identifiers("stratego/setups/perturbation.py")
        offenders = [
            name
            for name in names
            if any(token in name.lower() for token in FORBIDDEN_CODE_TOKENS)
        ]
        assert offenders == []

    def test_the_module_imports_no_model_or_training_code(self):
        source = (REPOSITORY_ROOT / "stratego/setups/perturbation.py").read_text()
        tree = ast.parse(source)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.ImportFrom):
                modules.add("." * node.level)
        assert not any(
            "model" in module or "training" in module or "evaluation" in module
            for module in modules
        ), modules
