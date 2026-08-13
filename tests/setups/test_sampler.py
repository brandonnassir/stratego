"""The runtime sampler: uniformity, split isolation, rebuild, and validation.

The properties pinned here are the ones Phase 8 will depend on without being
able to see inside the generator.

**Split isolation is structural, not cosmetic.** Changing `split` must change
the *eligible base population*, not relabel an identical base. The frozen
split rule partitions base indices, so these tests check the index ranges, the
disjointness of the reachable base identities, and the inherited split on
every output.

**Uniformity is a contract.** Family selection is uniform over 16 and base
selection is uniform inside the split, so no family gains mass because its
generator needed more candidates.

**Provenance rebuilds exactly.** `rebuild_from_provenance` must reproduce the
setup *and* the provenance — including a requested-but-exhausted perturbation
— or Phase 8's replay story is fiction.

**Nothing invalid escapes.** Every output passes the full final-output stack
recomputed from scratch; a sampler that cannot produce a valid output raises
rather than returning a fallback.
"""

import ast
import csv
import inspect
import json
import random
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from stratego.engine.constants import FLAG, PIECE_COUNTS
from stratego.engine.setup import validate_setup
from stratego.setups import (
    BASES_PER_FAMILY,
    FAMILY_IDS,
    MAX_PERTURBATION_ATTEMPTS,
    MAX_SWAP_COUNT,
    MIN_SWAP_COUNT,
    PERTURBATION_VERSION,
    SPLITS,
    TEST_PER_FAMILY,
    TRAIN_PER_FAMILY,
    VALIDATION_PER_FAMILY,
    decode_perturbation_seed,
    encode_perturbation_seed,
    evaluate_family,
    perturb_setup,
    reflect_canonical,
    setup_has_initial_mobility,
    split_for_base_index,
)
from stratego.setups import perturbation as perturbation_v1
from stratego.setups.contracts import LIBRARY_JSONL_PATH
from stratego.setups.identity import SetupLibraryError, class_fingerprint, orient_setup
from stratego.setups.library import read_library_jsonl
from stratego.setups.sampler import (
    DEFAULT_PROFILE,
    NEUTRAL_PROFILE,
    PERTURBATION_ONLY_PROFILE,
    PROFILES,
    PROVENANCE_FIELDS,
    REFLECTION_ONLY_PROFILE,
    REQUIRED_PROVENANCE_FIELDS,
    SAMPLER_VERSION,
    SPLIT_BASE_RANGES,
    STRESS_CORPUS_VERSION,
    STRESS_OUTPUTS_PER_FAMILY,
    STRESS_SPLIT_OUTPUTS,
    STRESS_TOTAL_OUTPUTS,
    SamplerProfile,
    SetupLibraryIndex,
    build_descendant,
    build_stress_output,
    load_library_index,
    provenance_is_observer_safe,
    provenance_round_trips,
    rebuild_from_provenance,
    sample_setup,
    sampler_contract_document,
    sampler_profile,
    stress_corpus_plan,
    validate_sampled_setup,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
LIBRARY_FILE = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"
CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_04_sampler_contract.json"
STRESS_ARTIFACT = DATA_DIRECTORY / "agent_04_procedural_stress.json"
FAMILY_METRICS_CSV = DATA_DIRECTORY / "agent_04_procedural_family_metrics.csv"

requires_production_library = pytest.mark.skipif(
    not LIBRARY_FILE.exists(),
    reason="production setup library not materialized yet",
)
requires_stress_artifacts = pytest.mark.skipif(
    not (
        CONTRACT_ARTIFACT.exists()
        and STRESS_ARTIFACT.exists()
        and FAMILY_METRICS_CSV.exists()
    ),
    reason="Agent 4 stress artifacts not materialized yet",
)


@pytest.fixture(scope="module")
def index() -> SetupLibraryIndex:
    return load_library_index(str(LIBRARY_FILE))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_the_default_profile_is_neutral(self):
        assert DEFAULT_PROFILE is NEUTRAL_PROFILE
        assert DEFAULT_PROFILE.reflection_probability == 0.5
        assert DEFAULT_PROFILE.perturbation_probability == 0.5
        assert len(set(DEFAULT_PROFILE.intensity_weights)) == 1, (
            "the default intensity mix must be uniform unless it is justified "
            "from structural-diversity evidence"
        )

    def test_the_instrument_profiles_pin_their_single_branch(self):
        assert REFLECTION_ONLY_PROFILE.perturbation_probability == 0.0
        assert PERTURBATION_ONLY_PROFILE.perturbation_probability == 1.0

    def test_profiles_are_looked_up_by_name(self):
        for name, profile in PROFILES.items():
            assert sampler_profile(name) is profile
        with pytest.raises(SetupLibraryError, match="unknown sampler profile"):
            sampler_profile("nope_v9")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"perturbation_probability": 1.5}, "perturbation_probability"),
            ({"reflection_probability": -0.1}, "reflection_probability"),
            ({"intensity_weights": (1.0, 1.0)}, "intensity_weights"),
            ({"intensity_weights": (0.0,) * 6}, "must not be all zero"),
            ({"intensity_weights": (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0)}, "non-negative"),
        ],
    )
    def test_malformed_profiles_are_refused(self, kwargs, match):
        base = {
            "name": "probe",
            "perturbation_probability": 0.5,
            "intensity_weights": (1.0,) * 6,
        }
        with pytest.raises(SetupLibraryError, match=match):
            SamplerProfile(**{**base, **kwargs})

    def test_the_swap_counts_cover_the_frozen_window(self):
        assert DEFAULT_PROFILE.swap_counts == tuple(
            range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1)
        )


# ---------------------------------------------------------------------------
# The library index
# ---------------------------------------------------------------------------


@requires_production_library
class TestLibraryIndex:
    def test_the_index_carries_the_whole_library(self, index):
        assert len(index.entries) == len(FAMILY_IDS) * BASES_PER_FAMILY == 8000
        assert len(index.content_digest) == 64

    def test_eligible_bases_are_split_restricted_and_ordered(self, index):
        expected = {
            "train": TRAIN_PER_FAMILY,
            "validation": VALIDATION_PER_FAMILY,
            "test": TEST_PER_FAMILY,
        }
        for family_id in FAMILY_IDS:
            for split in SPLITS:
                members = index.eligible_bases(family_id, split)
                assert len(members) == expected[split]
                indices = [entry.base_index for entry in members]
                assert indices == sorted(indices)
                start, stop = SPLIT_BASE_RANGES[split]
                assert all(start <= value < stop for value in indices)
                assert all(entry.split == split for entry in members)
                assert all(entry.family_id == family_id for entry in members)

    def test_the_split_index_ranges_partition_the_family(self, index):
        covered = set()
        for split in SPLITS:
            start, stop = SPLIT_BASE_RANGES[split]
            block = set(range(start, stop))
            assert not covered & block
            covered |= block
            assert all(split_for_base_index(value) == split for value in block)
        assert covered == set(range(BASES_PER_FAMILY))

    def test_unknown_lookups_are_refused(self, index):
        with pytest.raises(SetupLibraryError, match="unknown family"):
            index.eligible_bases("F99", "train")
        with pytest.raises(SetupLibraryError, match="unknown split"):
            index.eligible_bases("F00", "holdout")
        with pytest.raises(SetupLibraryError, match="unknown base_setup_id"):
            index.base("setup_library_v1:F00:999")

    def test_a_truncated_library_is_refused(self, index):
        with pytest.raises(SetupLibraryError, match="expected 8000 base entries"):
            SetupLibraryIndex(index.entries[:100])

    def test_a_relabelled_split_is_refused(self, index):
        from dataclasses import replace

        tampered = list(index.entries)
        tampered[0] = replace(tampered[0], split="test")
        with pytest.raises(SetupLibraryError, match="contradicts the frozen split rule"):
            SetupLibraryIndex(tampered)


# ---------------------------------------------------------------------------
# Sampling: determinism, uniformity, orientation
# ---------------------------------------------------------------------------


@requires_production_library
class TestSampling:
    def test_the_same_draw_reproduces_exactly(self, index):
        for seed in range(20):
            first = sample_setup("train", seed, index=index)
            second = sample_setup("train", seed, index=index)
            assert first.canonical == second.canonical
            assert first.provenance == second.provenance

    def test_no_global_rng_state_is_consumed(self, index):
        random.seed(7)
        first = sample_setup("train", 4242, index=index)
        random.seed(11)
        [random.random() for _ in range(53)]
        second = sample_setup("train", 4242, index=index)
        assert first.provenance == second.provenance

    def test_family_selection_is_uniform(self, index):
        draws = 16000
        counts = Counter(
            sample_setup("train", seed, index=index).family_id for seed in range(draws)
        )
        assert set(counts) == set(FAMILY_IDS)
        expected = draws / len(FAMILY_IDS)
        chi_square = sum(
            (count - expected) ** 2 / expected for count in counts.values()
        )
        # 15 degrees of freedom; the 0.999 critical value is ~37.7.
        assert chi_square < 37.7, counts

    def test_base_selection_is_uniform_inside_a_split(self, index):
        draws = 16000
        counts = Counter(
            sample_setup("train", seed, index=index).provenance["base_index"]
            for seed in range(draws)
        )
        assert set(counts) <= set(range(*SPLIT_BASE_RANGES["train"]))
        # Every base index should be reachable and none should dominate.
        assert len(counts) == TRAIN_PER_FAMILY
        expected = draws / TRAIN_PER_FAMILY
        assert max(counts.values()) < 4 * expected

    def test_orientation_is_a_fair_seeded_coin(self, index):
        draws = 8000
        reflected = sum(
            sample_setup("train", seed, index=index).reflection_applied
            for seed in range(draws)
        )
        assert abs(reflected / draws - 0.5) < 0.02

    def test_reflection_is_deterministic_and_correct(self, index):
        entry = index.base("setup_library_v1:F04:007")
        plain = build_descendant(
            entry, reflection_applied=False, perturbation_requested=False
        )
        mirrored = build_descendant(
            entry, reflection_applied=True, perturbation_requested=False
        )
        assert plain.canonical == entry.canonical_setup
        assert mirrored.canonical == reflect_canonical(entry.canonical_setup)
        assert reflect_canonical(mirrored.canonical) == plain.canonical
        assert class_fingerprint(mirrored.canonical) == class_fingerprint(plain.canonical)
        assert mirrored.provenance["final_setup_class_fingerprint"] == entry.fingerprint
        assert (
            mirrored.provenance["final_setup_fingerprint"]
            != plain.provenance["final_setup_fingerprint"]
        )

    def test_the_perturbation_probability_governs_the_perturbed_branch(self, index):
        for profile, expected in (
            (REFLECTION_ONLY_PROFILE, False),
            (PERTURBATION_ONLY_PROFILE, True),
        ):
            for seed in range(50):
                sampled = sample_setup("train", seed, profile=profile, index=index)
                assert sampled.provenance["perturbation_requested"] is expected
                assert sampled.perturbation_applied is expected

    def test_a_profile_can_be_named_by_string(self, index):
        by_object = sample_setup("train", 5, profile=PERTURBATION_ONLY_PROFILE, index=index)
        by_name = sample_setup("train", 5, profile="perturbation_only_v1", index=index)
        assert by_object.provenance == by_name.provenance

    def test_an_unknown_split_is_refused(self, index):
        with pytest.raises(SetupLibraryError, match="unknown split"):
            sample_setup("holdout", 1, index=index)

    def test_the_engine_handoff_orients_without_a_second_convention(self, index):
        sampled = sample_setup("train", 99, index=index)
        for player in (0, 1):
            assert sampled.oriented(player) == orient_setup(sampled.canonical, player)
            assert validate_setup(sampled.oriented(player), player)


# ---------------------------------------------------------------------------
# Split isolation
# ---------------------------------------------------------------------------


@requires_production_library
class TestSplitIsolation:
    def test_changing_the_split_changes_the_eligible_base_set(self, index):
        """Not a relabelling: the reachable base identities are disjoint."""
        reachable = {
            split: {
                sample_setup(split, seed, index=index).base_setup_id
                for seed in range(1500)
            }
            for split in SPLITS
        }
        assert not reachable["train"] & reachable["validation"]
        assert not reachable["train"] & reachable["test"]
        assert not reachable["validation"] & reachable["test"]

    def test_every_output_inherits_its_base_split(self, index):
        for split in SPLITS:
            start, stop = SPLIT_BASE_RANGES[split]
            for seed in range(400):
                sampled = sample_setup(split, seed, index=index)
                assert sampled.split == split
                assert index.base(sampled.base_setup_id).split == split
                assert start <= sampled.provenance["base_index"] < stop
                assert split_for_base_index(sampled.provenance["base_index"]) == split

    def test_a_perturbed_descendant_never_migrates_split(self, index):
        for split in SPLITS:
            for seed in range(200):
                sampled = sample_setup(
                    split, seed, profile=PERTURBATION_ONLY_PROFILE, index=index
                )
                assert sampled.perturbation_applied
                assert sampled.split == split
                assert sampled.provenance["split"] == split

    def test_validation_rejects_a_claimed_split_the_base_does_not_carry(self, index):
        entry = index.base("setup_library_v1:F00:000")
        failures = validate_sampled_setup(
            entry.canonical_setup, entry, "validation", entry.family_id
        )
        assert any("split migration" in failure for failure in failures)

    def test_validation_rejects_a_claimed_family_the_base_does_not_carry(self, index):
        entry = index.base("setup_library_v1:F00:000")
        failures = validate_sampled_setup(
            entry.canonical_setup, entry, entry.split, "F09"
        )
        assert any("family migration" in failure for failure in failures)


# ---------------------------------------------------------------------------
# Final-output validation
# ---------------------------------------------------------------------------


@requires_production_library
class TestFinalValidation:
    def test_every_sampled_output_passes_the_whole_stack(self, index):
        for split in SPLITS:
            for seed in range(150):
                sampled = sample_setup(split, seed, index=index)
                entry = index.base(sampled.base_setup_id)
                assert (
                    validate_sampled_setup(
                        sampled.canonical, entry, split, sampled.family_id
                    )
                    == []
                )
                assert validate_setup(sampled.canonical, 0) == sampled.canonical
                assert evaluate_family(sampled.family_id, sampled.canonical)[0]
                assert setup_has_initial_mobility(sampled.canonical)
                for piece_type, expected in PIECE_COUNTS.items():
                    assert sampled.canonical.count(piece_type) == expected

    def test_a_malformed_setup_is_reported_not_returned(self, index):
        entry = index.base("setup_library_v1:F00:000")
        failures = validate_sampled_setup((1, 2, 3), entry, entry.split, entry.family_id)
        assert failures and "length" in failures[0]

    def test_an_illegal_inventory_is_reported(self, index):
        entry = index.base("setup_library_v1:F00:000")
        broken = (FLAG,) + entry.canonical_setup[1:]
        failures = validate_sampled_setup(broken, entry, entry.split, entry.family_id)
        assert any("inventory/legality" in failure for failure in failures)

    def test_a_perturbation_request_without_its_identity_is_refused(self, index):
        entry = index.base("setup_library_v1:F00:000")
        with pytest.raises(SetupLibraryError, match="perturbation_seed"):
            build_descendant(entry, reflection_applied=False, perturbation_requested=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@requires_production_library
class TestProvenance:
    def test_the_schema_matches_the_record_builder(self, index):
        sampled = sample_setup("train", 3, index=index)
        assert tuple(sampled.provenance) == PROVENANCE_FIELDS
        assert set(REQUIRED_PROVENANCE_FIELDS) <= set(PROVENANCE_FIELDS)

    def test_every_required_field_is_present_on_both_branches(self, index):
        for profile in (REFLECTION_ONLY_PROFILE, PERTURBATION_ONLY_PROFILE):
            sampled = sample_setup("train", 21, profile=profile, index=index)
            assert all(key in sampled.provenance for key in REQUIRED_PROVENANCE_FIELDS)

    def test_provenance_carries_no_outcome_or_strength_field(self, index):
        for seed in range(50):
            sampled = sample_setup("train", seed, index=index)
            assert provenance_is_observer_safe(sampled.provenance) == []

    def test_provenance_survives_a_json_round_trip(self, index):
        for seed in range(50):
            sampled = sample_setup("train", seed, index=index)
            assert provenance_round_trips(sampled.provenance)

    def test_rebuild_from_provenance_is_exact(self, index):
        for split in SPLITS:
            for seed in range(200):
                sampled = sample_setup(split, seed, index=index)
                serialized = json.loads(json.dumps(sampled.provenance))
                rebuilt = rebuild_from_provenance(serialized, index=index)
                assert rebuilt.canonical == sampled.canonical
                assert rebuilt.provenance == sampled.provenance

    def test_rebuild_reproduces_the_perturbation_attempt_history(self, index):
        for seed in range(120):
            sampled = sample_setup(
                "train", seed, profile=PERTURBATION_ONLY_PROFILE, index=index
            )
            rebuilt = rebuild_from_provenance(dict(sampled.provenance), index=index)
            assert rebuilt.perturbation.attempts == sampled.perturbation.attempts
            assert rebuilt.perturbation.rejections == sampled.perturbation.rejections
            assert rebuilt.perturbation.swaps == sampled.perturbation.swaps

    def test_rebuild_refuses_provenance_missing_required_fields(self, index):
        sampled = sample_setup("train", 1, index=index)
        broken = dict(sampled.provenance)
        del broken["final_setup_fingerprint"]
        with pytest.raises(SetupLibraryError, match="missing required fields"):
            rebuild_from_provenance(broken, index=index)

    def test_rebuild_refuses_a_tampered_fingerprint(self, index):
        sampled = sample_setup("train", 2, index=index)
        broken = dict(sampled.provenance)
        broken["final_setup_fingerprint"] = "0" * 64
        with pytest.raises(SetupLibraryError, match="does not match the recorded"):
            rebuild_from_provenance(broken, index=index)

    def test_rebuild_refuses_a_tampered_split_or_family(self, index):
        sampled = sample_setup("train", 4, index=index)
        with pytest.raises(SetupLibraryError, match="contradicts base"):
            rebuild_from_provenance({**sampled.provenance, "split": "test"}, index=index)
        other = next(
            family_id for family_id in FAMILY_IDS if family_id != sampled.family_id
        )
        with pytest.raises(SetupLibraryError, match="contradicts base"):
            rebuild_from_provenance(
                {**sampled.provenance, "primary_family_id": other}, index=index
            )

    def test_the_perturbation_identity_is_a_pure_function_of_the_seed(self, index):
        """Agent 1's frozen invariant, tested end to end: the same
        `(base_setup_id, sampler_version, perturbation_seed)` always yields
        the same canonical descendant, from any caller context."""
        entry = index.base("setup_library_v1:F09:012")
        seed = encode_perturbation_seed(4, 987654)
        results = [
            perturb_setup(entry.canonical_setup, entry.family_id, seed)
            for _ in range(3)
        ]
        assert len({result.canonical for result in results}) == 1
        assert results[0] == results[1] == results[2]

    def test_swap_count_is_derivable_from_the_recorded_seed(self, index):
        """`perturbation_swap_count` is derived metadata: decoding the
        recorded seed must reproduce it exactly, on every perturbed output."""
        for seed in range(80):
            sampled = sample_setup(
                "train", seed, profile=PERTURBATION_ONLY_PROFILE, index=index
            )
            decoded_count, _raw = decode_perturbation_seed(
                sampled.provenance["perturbation_seed"]
            )
            assert decoded_count == sampled.provenance["perturbation_swap_count"]
            assert sampled.provenance["perturbation_hamming_from_base"] == (
                2 * decoded_count
            )

    def test_profile_context_cannot_change_the_descendant(self, index):
        """Once the sampler has emitted the effective perturbation seed, the
        profile is metadata: every caller context produces the same canonical
        descendant, and every record rebuilds to it."""
        entry = index.base("setup_library_v1:F09:012")
        seed = encode_perturbation_seed(4, 987654)
        variants = [
            build_descendant(
                entry,
                reflection_applied=False,
                perturbation_requested=True,
                perturbation_seed=seed,
                profile_name=name,
                draw_seed=draw_seed,
            )
            for draw_seed, name in enumerate(
                ["neutral_v1", "perturbation_only_v1", "reflection_only_v1", "unregistered_probe"]
            )
        ]
        assert len({variant.canonical for variant in variants}) == 1
        assert (
            len({variant.provenance["final_setup_fingerprint"] for variant in variants})
            == 1
        )
        for variant in variants:
            rebuilt = rebuild_from_provenance(dict(variant.provenance), index=index)
            assert rebuilt.canonical == variants[0].canonical

    def test_equal_perturbation_identities_cannot_rebuild_differently(self, index):
        """Two provenance records sharing the complete perturbation identity
        but differing in sampler metadata must rebuild to one descendant."""
        entry = index.base("setup_library_v1:F03:201")
        seed = encode_perturbation_seed(3, 5555)
        first = build_descendant(
            entry,
            reflection_applied=True,
            perturbation_requested=True,
            perturbation_seed=seed,
            profile_name="neutral_v1",
            draw_seed=1,
        )
        second = build_descendant(
            entry,
            reflection_applied=True,
            perturbation_requested=True,
            perturbation_seed=seed,
            profile_name=STRESS_CORPUS_VERSION,
            draw_seed=999_999,
        )
        assert first.canonical == second.canonical
        rebuilt_first = rebuild_from_provenance(dict(first.provenance), index=index)
        rebuilt_second = rebuild_from_provenance(dict(second.provenance), index=index)
        assert rebuilt_first.canonical == rebuilt_second.canonical == first.canonical

    def test_the_retry_budget_is_a_version_constant_not_an_input(self):
        """The accepted candidate is the first one the budget reaches, so the
        budget is perturbation semantics. It is therefore a constant of
        `setup_perturbation_v1`, not a parameter anywhere in production."""
        assert perturbation_v1.MAX_PERTURBATION_ATTEMPTS == 64
        assert "max_attempts" not in inspect.signature(perturb_setup).parameters
        assert list(inspect.signature(build_descendant).parameters) == [
            "base_entry",
            "reflection_applied",
            "perturbation_requested",
            "perturbation_seed",
            "profile_name",
            "draw_seed",
        ]

    def test_tampered_max_attempt_provenance_is_rejected(self, index):
        sampled = sample_setup(
            "train", 7, profile=PERTURBATION_ONLY_PROFILE, index=index
        )
        assert sampled.provenance["perturbation_max_attempts"] == 64
        broken = {**sampled.provenance, "perturbation_max_attempts": 63}
        with pytest.raises(SetupLibraryError, match="version constant"):
            rebuild_from_provenance(broken, index=index)

    def test_tampered_derived_swap_count_provenance_is_rejected(self, index):
        sampled = sample_setup(
            "train", 7, profile=PERTURBATION_ONLY_PROFILE, index=index
        )
        recorded = sampled.provenance["perturbation_swap_count"]
        tampered_count = recorded % MAX_SWAP_COUNT + 1
        assert tampered_count != recorded
        broken = {**sampled.provenance, "perturbation_swap_count": tampered_count}
        with pytest.raises(SetupLibraryError, match="derived metadata"):
            rebuild_from_provenance(broken, index=index)

    def test_an_exhausted_perturbation_is_recorded_honestly_and_rebuilds(self, index):
        """A requested-but-exhausted perturbation must be recorded as not
        applied, return the unmodified base, and replay exactly.

        Production's 64-attempt budget never exhausts on library bases, so
        the branch is forced through a test-only patch of the version
        constant — a private diagnostic mechanism, not a second production
        semantics: the final block proves that a record produced under the
        patched budget is *rejected* by the unpatched production rebuild.
        """
        entry = index.base("setup_library_v1:F14:000")
        exhausted = None
        with mock.patch.object(perturbation_v1, "MAX_PERTURBATION_ATTEMPTS", 1):
            for raw_seed in range(400):
                sampled = build_descendant(
                    entry,
                    reflection_applied=False,
                    perturbation_requested=True,
                    perturbation_seed=encode_perturbation_seed(6, raw_seed),
                )
                assert sampled.provenance["perturbation_requested"] is True
                assert sampled.provenance["perturbation_max_attempts"] == 1
                assert sampled.provenance["perturbation_id"].startswith(
                    PERTURBATION_VERSION
                )
                rebuilt = rebuild_from_provenance(dict(sampled.provenance), index=index)
                assert rebuilt.canonical == sampled.canonical
                assert rebuilt.provenance == sampled.provenance
                if not sampled.perturbation_applied:
                    exhausted = sampled
                    break

        assert exhausted is not None, "no single-attempt exhaustion occurred to check"
        assert exhausted.provenance["perturbation_exhausted"] is True
        assert exhausted.provenance["perturbation_applied"] is False
        assert exhausted.provenance["perturbation_hamming_from_base"] == 0
        assert exhausted.canonical == entry.canonical_setup
        assert setup_has_initial_mobility(exhausted.canonical)
        assert evaluate_family(entry.family_id, exhausted.canonical)[0]

        # Outside the diagnostic patch the truncated-budget record is not a
        # valid production record, and the integrity guard rejects it.
        with pytest.raises(SetupLibraryError, match="version constant"):
            rebuild_from_provenance(dict(exhausted.provenance), index=index)

    def test_every_registered_profile_rebuilds_exactly(self, index):
        for name in PROFILES:
            for seed in range(30):
                sampled = sample_setup("train", seed, profile=name, index=index)
                rebuilt = rebuild_from_provenance(dict(sampled.provenance), index=index)
                assert rebuilt.canonical == sampled.canonical
                assert rebuilt.provenance == sampled.provenance


# ---------------------------------------------------------------------------
# The stress-corpus instrument
# ---------------------------------------------------------------------------


class TestStressCorpusPlan:
    def test_the_nominal_corpus_is_the_required_size(self):
        assert STRESS_SPLIT_OUTPUTS == {"train": 5000, "validation": 625, "test": 625}
        assert STRESS_OUTPUTS_PER_FAMILY == 6250
        assert STRESS_TOTAL_OUTPUTS == 100000
        assert STRESS_TOTAL_OUTPUTS >= 100000

    def test_the_plan_is_deterministic(self):
        outputs = {"train": 40, "validation": 8, "test": 8}
        first = list(stress_corpus_plan(split_outputs=outputs, family_ids=("F03",)))
        second = list(stress_corpus_plan(split_outputs=outputs, family_ids=("F03",)))
        assert first == second

    def test_the_plan_balances_both_branches_and_both_orientations(self):
        outputs = {"train": 400, "validation": 0, "test": 0}
        draws = list(stress_corpus_plan(split_outputs=outputs, family_ids=("F07",)))
        assert len(draws) == 400
        assert sum(draw.reflection_applied for draw in draws) == 200
        assert sum(draw.perturbation_requested for draw in draws) == 200
        combinations = Counter(
            (draw.reflection_applied, draw.perturbation_requested) for draw in draws
        )
        assert set(combinations.values()) == {100}

    def test_the_full_train_segment_is_exactly_balanced(self):
        outputs = {"train": STRESS_SPLIT_OUTPUTS["train"], "validation": 0, "test": 0}
        draws = list(stress_corpus_plan(split_outputs=outputs, family_ids=("F04",)))
        assert len(draws) == 5000
        assert sum(draw.reflection_applied for draw in draws) == 2500
        assert sum(draw.perturbation_requested for draw in draws) == 2500
        combinations = Counter(
            (draw.reflection_applied, draw.perturbation_requested) for draw in draws
        )
        assert set(combinations.values()) == {1250}

    def test_each_base_is_exercised_in_both_orientations_and_both_branches(self):
        """Regression: the base round robin has period `len(eligible)`, and
        400, 50 and 50 are all multiples of 4 — the period of the branch
        counter. Keying the branch bits on `position` alone aliased them onto
        the base index, so every output of a given base carried the same
        orientation and the same branch. No base ever appeared in both
        orientations, which meant a mirror-image duplicate between two
        descendants of one base could not arise even in principle. The `lap`
        offset makes the stride odd and breaks the aliasing.
        """
        for split, count in STRESS_SPLIT_OUTPUTS.items():
            outputs = {name: 0 for name in SPLITS}
            outputs[split] = count
            per_base: dict[int, set] = {}
            for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F06",)):
                per_base.setdefault(draw.base_index, set()).add(
                    (draw.reflection_applied, draw.perturbation_requested)
                )
            assert per_base, split
            for base_index, combinations in per_base.items():
                assert len(combinations) == 4, (
                    f"{split} base {base_index} only ever saw {sorted(combinations)}"
                )

    def test_each_base_sees_several_perturbation_intensities(self):
        outputs = {"train": STRESS_SPLIT_OUTPUTS["train"], "validation": 0, "test": 0}
        per_base: dict[int, set] = {}
        for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F09",)):
            if draw.perturbation_requested:
                per_base.setdefault(draw.base_index, set()).add(draw.swap_count)
        assert len(per_base) == TRAIN_PER_FAMILY
        assert min(len(counts) for counts in per_base.values()) >= 3

    def test_the_plan_covers_the_whole_swap_window(self):
        outputs = {"train": 240, "validation": 0, "test": 0}
        draws = [
            draw
            for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F10",))
            if draw.perturbation_requested
        ]
        assert {draw.swap_count for draw in draws} == set(
            range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1)
        )

    def test_the_plan_stays_inside_the_split_index_ranges(self):
        outputs = {"train": 200, "validation": 60, "test": 60}
        for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F01", "F15")):
            start, stop = SPLIT_BASE_RANGES[draw.split]
            assert start <= draw.base_index < stop
            assert split_for_base_index(draw.base_index) == draw.split
            assert draw.base_setup_id.endswith(f"{draw.base_index:03d}")
            assert f":{draw.family_id}:" in draw.base_setup_id

    def test_the_plan_visits_every_base_of_a_split_round_robin(self):
        outputs = {"train": TRAIN_PER_FAMILY, "validation": 0, "test": 0}
        draws = list(stress_corpus_plan(split_outputs=outputs, family_ids=("F02",)))
        assert sorted(draw.base_index for draw in draws) == list(range(TRAIN_PER_FAMILY))

    def test_perturbation_seeds_are_distinct_per_position(self):
        outputs = {"train": 200, "validation": 0, "test": 0}
        seeds = [
            draw.perturbation_seed
            for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F05",))
        ]
        assert len(set(seeds)) == len(seeds)

    def test_plan_seeds_encode_the_plan_swap_count(self):
        """The plan's `swap_count` is derived metadata like everywhere else:
        decoding the planned composite seed must reproduce it."""
        outputs = {"train": 200, "validation": 24, "test": 24}
        for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F05", "F11")):
            decoded_count, _raw = decode_perturbation_seed(draw.perturbation_seed)
            assert decoded_count == draw.swap_count

    @requires_production_library
    def test_built_outputs_match_their_plan(self, index):
        outputs = {"train": 24, "validation": 8, "test": 8}
        for draw in stress_corpus_plan(split_outputs=outputs, family_ids=("F08", "F13")):
            sampled = build_stress_output(draw, index=index)
            assert sampled.base_setup_id == draw.base_setup_id
            assert sampled.split == draw.split
            assert sampled.family_id == draw.family_id
            assert sampled.reflection_applied == draw.reflection_applied
            assert sampled.provenance["sampler_profile"] == STRESS_CORPUS_VERSION
            assert (
                validate_sampled_setup(
                    sampled.canonical, index.base(draw.base_setup_id), draw.split, draw.family_id
                )
                == []
            )
            assert rebuild_from_provenance(
                dict(sampled.provenance), index=index
            ).canonical == sampled.canonical


# ---------------------------------------------------------------------------
# The contract document
# ---------------------------------------------------------------------------


class TestContractDocument:
    def test_the_document_restates_the_frozen_versions(self):
        document = sampler_contract_document()
        assert document["sampler_version"] == SAMPLER_VERSION
        assert document["perturbation"]["perturbation_version"] == PERTURBATION_VERSION
        assert document["default_profile"] == DEFAULT_PROFILE.name
        assert document["provenance_schema"]["full_fields"] == list(PROVENANCE_FIELDS)
        assert document["provenance_schema"]["required_fields"] == list(
            REQUIRED_PROVENANCE_FIELDS
        )
        assert set(document["profiles"]) == set(PROFILES)
        assert document["stress_corpus"]["total_outputs"] == STRESS_TOTAL_OUTPUTS
        assert len(document["final_output_validation"]) == 7


# ---------------------------------------------------------------------------
# No outcome or strength signal
# ---------------------------------------------------------------------------


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
        tree = ast.parse((REPOSITORY_ROOT / "stratego/setups/sampler.py").read_text())
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
        offenders = [
            name for name in names if any(token in name.lower() for token in FORBIDDEN_CODE_TOKENS)
        ]
        assert offenders == []

    def test_the_sampler_imports_no_model_or_training_code(self):
        tree = ast.parse((REPOSITORY_ROOT / "stratego/setups/sampler.py").read_text())
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        assert not any(
            "model" in module or "training" in module or "evaluation" in module
            for module in modules
        ), modules


# ---------------------------------------------------------------------------
# Stress artifacts (gated until the harness has run)
# ---------------------------------------------------------------------------


@pytest.fixture()
def stress_artifact() -> dict:
    return json.loads(STRESS_ARTIFACT.read_text())


@pytest.fixture()
def contract_artifact() -> dict:
    return json.loads(CONTRACT_ARTIFACT.read_text())


@requires_stress_artifacts
class TestStressArtifacts:
    def test_the_run_reports_pass_with_every_gate_true(self, stress_artifact):
        assert stress_artifact["status"] == "PASS"
        assert stress_artifact["gates_true"] == stress_artifact["gates_total"]
        assert all(stress_artifact["completion_gates"].values())

    def test_the_corpus_meets_the_required_size_and_shape(self, stress_artifact):
        corpus = stress_artifact["corpus"]
        assert corpus["outputs"] >= 100000
        assert set(corpus["outputs_by_family"]) == set(FAMILY_IDS)
        assert set(corpus["outputs_by_family"].values()) == {STRESS_OUTPUTS_PER_FAMILY}
        assert corpus["outputs_by_split"]["train"] == 5000 * len(FAMILY_IDS)
        assert corpus["bases_used"] == 8000
        assert min(corpus["branch_counts"].values()) > 0

    def test_every_hard_requirement_is_zero(self, stress_artifact):
        hard = stress_artifact["hard_requirements"]
        for name, value in hard.items():
            if name == "examples":
                continue
            assert value == 0, f"{name} = {value}"

    def test_no_cross_split_or_cross_base_descendant_leakage(self, stress_artifact):
        analysis = stress_artifact["duplicate_and_leakage_analysis"]
        assert analysis["classes_with_multiple_splits"] == 0
        assert analysis["exact_setups_with_multiple_splits"] == 0
        assert analysis["classes_with_multiple_families"] == 0
        pairwise = stress_artifact["pairwise_class_distance"]
        assert pairwise["cross_split_min_class_distance"] >= pairwise["cross_split_floor"]

    def test_procedural_support_exceeds_the_static_library(self, stress_artifact):
        diversity = stress_artifact["effective_diversity"]
        assert diversity["distinct_class_fingerprints"] > diversity["static_base_classes"]
        assert diversity["procedural_support_multiple"] > 1.0

    def test_the_family_diagonal_is_still_exactly_one(self, stress_artifact):
        diagonal = stress_artifact["overlap_comparison"]["self_satisfaction_diagonal"]
        assert set(diagonal) == set(FAMILY_IDS)
        assert all(value == 1.0 for value in diagonal.values())

    def test_the_artifact_versions_match_the_code(self, stress_artifact, contract_artifact):
        assert stress_artifact["sampler_version"] == SAMPLER_VERSION
        assert stress_artifact["perturbation_version"] == PERTURBATION_VERSION
        assert stress_artifact["stress_corpus_version"] == STRESS_CORPUS_VERSION
        assert contract_artifact["sampler_version"] == SAMPLER_VERSION
        assert contract_artifact["provenance_schema"]["full_fields"] == list(PROVENANCE_FIELDS)

    @requires_production_library
    def test_the_artifact_pins_the_audited_library_digest(self, stress_artifact, index):
        assert stress_artifact["library_digest"] == index.content_digest
        assert contract_digest(stress_artifact) == index.content_digest

    def test_the_family_metrics_csv_agrees_with_the_json(self, stress_artifact):
        with FAMILY_METRICS_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["family_id"] for row in rows] == list(FAMILY_IDS)
        per_family = stress_artifact["family_metrics"]["per_family"]
        for row in rows:
            metrics = per_family[row["family_id"]]
            assert int(row["outputs"]) == metrics["outputs"]
            assert int(row["distinct_class_fingerprints"]) == (
                metrics["distinct_class_fingerprints"]
            )
            assert float(row["self_satisfaction"]) == metrics["self_satisfaction"]

    def test_the_totals_add_up(self, stress_artifact):
        corpus = stress_artifact["corpus"]
        assert sum(corpus["outputs_by_family"].values()) == corpus["outputs"]
        assert sum(corpus["outputs_by_split"].values()) == corpus["outputs"]
        assert sum(corpus["branch_counts"].values()) == corpus["outputs"]
        diversity = stress_artifact["effective_diversity"]
        assert diversity["outputs"] == corpus["outputs"]
        assert (
            diversity["perturbation_applied"] + diversity["perturbation_exhausted"]
            == diversity["perturbation_requested"]
        )


def contract_digest(stress_artifact: dict) -> str:
    return stress_artifact["prerequisite_status"]["observed_digests"]["library_digest"]
