"""Phase 7 Agent 3: the independent auditor and its injected-defect proofs.

Two obligations are tested here. First, a clean library slice must produce
zero findings — the auditor may not cry wolf. Second, every defect class the
Agent 3 instructions name must be caught when deliberately injected: wrong
inventory, illegal placement, stranded setup, wrong family label, exact
duplicate, reflected duplicate, stable-ID collision, split-count mismatch,
cross-split near duplicate, diversity-threshold failure, bad serialization,
bad reflection, and an altered manifest digest.

Injections are built from isolated rebuilds of real production entries
(`rebuild_base_setup`), so the clean baseline is the actual frozen library
content, and each forged variant differs from it in exactly the injected
defect (derived fields are recomputed where the test isolates a single
finding, or left stale where the staleness *is* the defect).

The production artifacts written by `scripts/run_phase7_agent03.py` are
checked by the artifact-gated tests at the bottom once they exist.
"""

import csv
import json
from functools import lru_cache
from pathlib import Path

import pytest

from stratego.engine.constants import BOMB, FLAG, MINER, RED, SCOUT
from stratego.engine.setup import (
    SetupError,
    serialize_setup,
    setup_to_placements,
    validate_setup_placement,
)
from stratego.setups import (
    AUDIT_VERSION,
    BaseSetupEntry,
    audit_library,
    base_entry_json_line,
    canonical_class_representative,
    canonical_index,
    class_fingerprint,
    compute_trait_vector,
    content_fingerprint,
    count_audit,
    duplicate_audit,
    line_format_audit,
    manifest_audit,
    orient_setup,
    overlap_audit,
    per_base_audit,
    read_library_jsonl,
    read_manifest,
    rebuild_base_setup,
    reflect_canonical,
    similarity_audit,
    similarity_cross_check,
    threshold_audit,
)
from stratego.setups.traits import OPEN_FRONT_FILES

from .family_fixtures import build_fixture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
LIBRARY_PATH = REPOSITORY_ROOT / "data" / "setups" / "setup_library_v1.jsonl"
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "setups" / "setup_library_v1_manifest.json"
AUDIT_ARTIFACT = REPOSITORY_ROOT / "reports" / "phase_7_data" / "agent_03_library_audit.json"
FAMILY_METRICS_CSV = REPOSITORY_ROOT / "reports" / "phase_7_data" / "agent_03_family_metrics.csv"
SIMILARITY_CSV = REPOSITORY_ROOT / "reports" / "phase_7_data" / "agent_03_similarity_audit.csv"

requires_production_library = pytest.mark.skipif(
    not (LIBRARY_PATH.exists() and MANIFEST_PATH.exists()),
    reason="production setup_library_v1 not materialized",
)
requires_audit_artifacts = pytest.mark.skipif(
    not (
        AUDIT_ARTIFACT.exists()
        and FAMILY_METRICS_CSV.exists()
        and SIMILARITY_CSV.exists()
    ),
    reason="Agent 3 audit artifacts not materialized yet",
)


# ---------------------------------------------------------------------------
# Clean baseline: isolated rebuilds of real production entries
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _sample_entries() -> "tuple[BaseSetupEntry, ...]":
    """Real production entries across families and both split boundaries."""
    families = ("F00", "F07", "F11", "F15")
    indices = (0, 1, 399, 400, 449, 450, 499)
    return tuple(
        rebuild_base_setup(family_id, base_index)
        for family_id in families
        for base_index in indices
    )


def _entry(family_id: str, base_index: int) -> BaseSetupEntry:
    return next(
        entry
        for entry in _sample_entries()
        if entry.family_id == family_id and entry.base_index == base_index
    )


def _raw_variant(entry: BaseSetupEntry, **overrides) -> BaseSetupEntry:
    """A forged entry whose overridden fields are deliberately inconsistent."""
    payload = entry.to_dict()
    payload.update(overrides)
    return BaseSetupEntry.from_dict(payload)


def _content_variant(entry: BaseSetupEntry, new_setup, **overrides) -> BaseSetupEntry:
    """A forged entry with `new_setup` and honestly recomputed derived fields.

    Fingerprints and traits are recomputed so the only findings a stage can
    raise against the variant are the ones the injected content itself causes.
    """
    setup = tuple(new_setup)
    payload = entry.to_dict()
    payload["canonical_setup"] = serialize_setup(setup)
    payload["fingerprint"] = class_fingerprint(setup)
    payload["content_fingerprint"] = content_fingerprint(setup)
    payload["reflected_content_fingerprint"] = content_fingerprint(
        reflect_canonical(setup)
    )
    payload["trait_vector"] = compute_trait_vector(setup)
    payload.update(overrides)
    return BaseSetupEntry.from_dict(payload)


def _swapped_variant(setup, swaps: int):
    """`setup` with `swaps` disjoint central swaps of distinct movable types.

    Each swap changes exactly two cells, so the Hamming distance to `setup`
    is `2 * swaps`. Only movable pieces in ranks 1-2, files 2-7 move, which
    cannot affect any F00 clause (Flag rank/edge/guards) or initial mobility.
    """
    cells = list(setup)
    candidates = [
        canonical_index(rank, file) for rank in (1, 2) for file in range(2, 8)
    ]
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    for left in candidates:
        if len(chosen) == swaps:
            break
        if left in used or cells[left] in (FLAG, BOMB):
            continue
        for right in candidates:
            if right == left or right in used or cells[right] in (FLAG, BOMB):
                continue
            if cells[left] != cells[right]:
                chosen.append((left, right))
                used.update((left, right))
                break
    assert len(chosen) == swaps, "fixture lacks distinct movable central pieces"
    for left, right in chosen:
        cells[left], cells[right] = cells[right], cells[left]
    return tuple(cells)


def _stranded_f07_setup():
    """A legal, F07-satisfying, immobile arrangement (Bombs on all open files)."""
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
# Positive control: the auditor does not cry wolf
# ---------------------------------------------------------------------------


class TestPositiveControl:
    def test_clean_rebuilt_entries_produce_zero_per_base_findings(self):
        result = per_base_audit(_sample_entries())
        for key, value in result.items():
            if key in ("entry_count", "trait_vectors"):
                continue
            assert value == [], f"unexpected findings under {key}: {value!r}"

    def test_clean_rebuilt_entries_have_no_duplicates(self):
        result = duplicate_audit(_sample_entries())
        assert result["exact_duplicate_groups"] == 0
        assert result["reflection_class_duplicate_groups"] == 0
        assert result["stable_id_collisions"] == 0
        assert result["same_id_different_setup"] == []
        assert result["different_id_same_class_groups"] == []
        assert result["stored_mirror_overlap"] == []
        assert result["cross_split_class_duplicate_groups"] == 0

    def test_canonical_lines_pass_the_line_format_audit(self):
        text = "\n".join(
            base_entry_json_line(entry.to_dict()) for entry in _sample_entries()
        )
        result = line_format_audit(text)
        assert result["line_count"] == len(_sample_entries())
        assert result["serialization_failures"] == 0

    def test_the_similarity_matrix_agrees_with_the_frozen_scalar_metric(self):
        result = similarity_audit(list(_sample_entries()))
        assert result["cross_check"]["mismatches_vs_frozen_metric"] == 0
        assert result["cross_check"]["matrix_symmetric"]

    def test_the_audit_side_reduction_matches_the_frozen_reduction(self):
        entries = list(_sample_entries())
        similarity = similarity_audit(entries)
        thresholds = threshold_audit(entries)
        reconciliation = similarity_cross_check(similarity, thresholds["metrics"])
        assert reconciliation["agrees"], reconciliation["disagreements"]


# ---------------------------------------------------------------------------
# Injected defects: inventory and placement
# ---------------------------------------------------------------------------


class TestInventoryAndPlacementInjection:
    def test_a_wrong_inventory_is_caught(self):
        entry = _entry("F00", 0)
        cells = list(entry.canonical_setup)
        cells[cells.index(SCOUT)] = MINER  # 7 Scouts / 6 Miners: inventory broken
        forged = _raw_variant(entry, canonical_setup=serialize_setup(tuple(cells)))
        result = per_base_audit([forged])
        assert forged.base_setup_id in result["inventory_failures"]
        assert any(
            failure["base_setup_id"] == forged.base_setup_id
            and "inventory" in failure["error"]
            for failure in result["engine_failures"]
        )

    def test_the_placement_validator_the_audit_delegates_to_rejects_bad_squares(self):
        entry = _entry("F00", 0)
        placements = setup_to_placements(
            orient_setup(entry.canonical_setup, RED), RED
        )
        square = next(iter(placements))
        piece = placements.pop(square)
        placements[42] = piece  # a lake square
        with pytest.raises(SetupError):
            validate_setup_placement(placements, RED)
        placements.pop(42)
        placements[55] = piece  # a central non-setup square
        with pytest.raises(SetupError):
            validate_setup_placement(placements, RED)

    def test_a_truncated_setup_string_is_a_serialization_finding(self):
        entry = _entry("F00", 0)
        payload = entry.to_dict()
        payload["canonical_setup"] = payload["canonical_setup"][:-1]
        result = line_format_audit(base_entry_json_line(payload))
        assert result["undeserializable_lines"] == [1]
        assert result["serialization_failures"] >= 1


# ---------------------------------------------------------------------------
# Injected defects: stranded setup and wrong family label
# ---------------------------------------------------------------------------


class TestStrandedAndFamilyInjection:
    def test_a_stranded_setup_is_caught_in_both_orientations(self):
        base = _entry("F07", 0)
        forged = _content_variant(base, _stranded_f07_setup())
        result = per_base_audit([forged])
        assert forged.base_setup_id in result["mobility_failures"]
        assert forged.base_setup_id in result["reflected_mobility_failures"]
        # The arrangement is legal and still satisfies F07, so the finding is
        # isolated to the mobility rule.
        assert result["engine_failures"] == []
        assert result["family_failures"] == []

    def test_a_wrong_family_label_is_caught_with_the_violated_clause(self):
        entry = _entry("F00", 0)  # corner fortress: two orthogonal Bomb guards
        forged = _raw_variant(
            entry,
            family_id="F04",
            family_key="lightly_defended_deceptive_flag",
            base_setup_id="setup_library_v1:F04:000",
        )
        result = per_base_audit([forged])
        failure = next(
            item
            for item in result["family_failures"]
            if item["base_setup_id"] == forged.base_setup_id
        )
        assert "no_orthogonal_guards" in failure["violations"]
        assert any(
            item["base_setup_id"] == forged.base_setup_id
            for item in result["reflected_family_failures"]
        )

    def test_a_tampered_generation_seed_is_caught(self):
        entry = _entry("F00", 0)
        forged = _raw_variant(entry, generation_seed=entry.generation_seed + 1)
        assert forged.base_setup_id in per_base_audit([forged])["seed_failures"]


# ---------------------------------------------------------------------------
# Injected defects: duplicates and identity collisions
# ---------------------------------------------------------------------------


class TestDuplicateInjection:
    def test_an_exact_duplicate_is_caught(self):
        original = _entry("F00", 0)
        forged = _content_variant(_entry("F00", 1), original.canonical_setup)
        result = duplicate_audit([original, forged])
        assert result["exact_duplicate_groups"] == 1
        assert result["reflection_class_duplicate_groups"] == 1
        assert sorted(result["exact_duplicate_members"]) == sorted(
            [original.base_setup_id, forged.base_setup_id]
        )
        assert result["different_id_same_class_groups"]

    def test_duplicate_detection_is_not_limited_to_within_family(self):
        original = _entry("F00", 0)
        forged = _content_variant(_entry("F07", 0), original.canonical_setup)
        result = duplicate_audit([original, forged])
        assert result["exact_duplicate_groups"] == 1

    def test_a_reflected_duplicate_is_caught(self):
        original = _entry("F00", 0)
        forged = _content_variant(
            _entry("F00", 1), reflect_canonical(original.canonical_setup)
        )
        result = duplicate_audit([original, forged])
        assert result["exact_duplicate_groups"] == 0
        assert result["reflection_class_duplicate_groups"] == 1
        assert result["stored_mirror_overlap"]
        # The stored mirror is also a canonicalization violation per entry.
        per_base = per_base_audit([forged])
        assert forged.base_setup_id in per_base["canonicalization_failures"]

    def test_a_stable_id_collision_is_caught(self):
        original = _entry("F00", 0)
        forged = _raw_variant(_entry("F00", 1), base_setup_id=original.base_setup_id)
        result = duplicate_audit([original, forged])
        assert result["stable_id_collisions"] == 1
        assert original.base_setup_id in result["same_id_different_setup"]


# ---------------------------------------------------------------------------
# Injected defects: split handling and cross-split leakage
# ---------------------------------------------------------------------------


class TestSplitAndLeakageInjection:
    def test_a_flipped_split_label_is_caught(self):
        entry = _entry("F00", 0)  # index 0 is train by the frozen rule
        forged = _raw_variant(entry, split="validation")
        counts = count_audit([forged])
        assert not counts["checks"]["family_split_exact"]
        per_base = per_base_audit([forged])
        assert any(
            item["base_setup_id"] == forged.base_setup_id
            for item in per_base["identity_failures"]
        )

    def test_a_cross_split_class_duplicate_is_caught(self):
        train_entry = _entry("F00", 399)
        forged_validation = _content_variant(
            _entry("F00", 400), train_entry.canonical_setup
        )
        result = duplicate_audit([train_entry, forged_validation])
        assert result["cross_split_class_duplicate_groups"] == 1
        assert train_entry.base_setup_id in result["cross_split_class_duplicate_members"]

    def test_a_cross_split_near_duplicate_is_caught_with_offenders_named(self):
        train_entry = _entry("F00", 399)
        near = canonical_class_representative(
            _swapped_variant(train_entry.canonical_setup, swaps=1)
        )
        forged_validation = _content_variant(_entry("F00", 400), near)
        others = [
            entry
            for entry in _sample_entries()
            if entry.base_setup_id
            not in (train_entry.base_setup_id, forged_validation.base_setup_id)
        ]
        entries = [train_entry, forged_validation, *others]

        similarity = similarity_audit(entries)
        assert similarity["cross_split_min_nn_distance"] == 2
        scope = similarity["cross_split"]["train__validation"]["global"]
        assert scope["pairs_below_cross_split_floor"] >= 1
        offender = scope["offending_pairs"][0]
        assert {offender["a"], offender["b"]} == {
            train_entry.base_setup_id,
            forged_validation.base_setup_id,
        }
        assert offender["class_distance"] == 2

        thresholds = threshold_audit(entries)
        assert not thresholds["all_pass"]
        assert any(
            check["check"] == "cross_split_min_nn_distance" and not check["pass"]
            for check in thresholds["checks"]
        )


# ---------------------------------------------------------------------------
# Injected defects: diversity-threshold failure (finding, not repair)
# ---------------------------------------------------------------------------


class TestThresholdFailureInjection:
    def test_a_within_family_distance_floor_failure_is_reported(self):
        original = _entry("F00", 0)
        variant = canonical_class_representative(
            _swapped_variant(original.canonical_setup, swaps=2)
        )
        forged = _content_variant(_entry("F00", 1), variant)
        entries = [original, forged]

        # Both entries still satisfy F00, so this is purely a diversity
        # failure, not a family violation.
        thresholds = threshold_audit(entries)
        assert not thresholds["all_pass"]
        failed_names = {check["check"] for check in thresholds["failed_checks"]}
        assert "F00:min_within_family_nn_distance" in failed_names
        assert "F00:within_family_near_duplicate_fraction" in failed_names
        assert not any(
            check["check"] == "F00:self_satisfaction" and not check["pass"]
            for check in thresholds["checks"]
        )

        similarity = similarity_audit(entries)
        assert similarity["within_family"]["F00"]["min_nn_distance"] == 4
        assert similarity["within_family"]["F00"]["offending_pairs"]


# ---------------------------------------------------------------------------
# Injected defects: serialization and reflection metadata
# ---------------------------------------------------------------------------


class TestSerializationAndReflectionInjection:
    def test_noncanonical_json_lines_are_caught(self):
        entry = _entry("F00", 0)
        line = json.dumps(entry.to_dict(), sort_keys=True)  # spaced separators
        result = line_format_audit(line)
        assert result["noncanonical_lines"] == [1]
        assert result["serialization_failures"] == 1

    def test_a_missing_required_field_is_caught(self):
        payload = _entry("F00", 0).to_dict()
        del payload["fingerprint"]
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert line_format_audit(line)["missing_field_lines"] == [1]

    def test_an_unparseable_line_is_caught(self):
        assert line_format_audit("{not json")["unparseable_lines"] == [1]

    def test_a_corrupted_class_fingerprint_is_caught(self):
        entry = _entry("F00", 0)
        forged = _raw_variant(entry, fingerprint="0" * 64)
        failure = next(
            item
            for item in per_base_audit([forged])["fingerprint_failures"]
            if item["base_setup_id"] == forged.base_setup_id
        )
        assert "fingerprint" in failure["mismatched"]

    def test_a_corrupted_reflected_fingerprint_is_caught(self):
        entry = _entry("F00", 0)
        forged = _raw_variant(entry, reflected_content_fingerprint="0" * 64)
        failure = next(
            item
            for item in per_base_audit([forged])["fingerprint_failures"]
            if item["base_setup_id"] == forged.base_setup_id
        )
        assert "reflected_content_fingerprint" in failure["mismatched"]

    def test_a_tampered_trait_vector_is_caught(self):
        entry = _entry("F00", 0)
        traits = dict(entry.trait_vector)
        traits["flag_rank"] = 3
        forged = _raw_variant(entry, trait_vector=traits)
        assert forged.base_setup_id in per_base_audit([forged])["trait_failures"]


# ---------------------------------------------------------------------------
# Injected defects: manifest digest (production-gated)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def production():
    return read_library_jsonl(LIBRARY_PATH), read_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def artifact():
    return json.loads(AUDIT_ARTIFACT.read_text())


@requires_production_library
class TestManifestInjection:
    def test_the_real_manifest_passes(self, production):
        entries, manifest = production
        assert manifest_audit(entries, manifest)["all_pass"]

    def test_an_altered_manifest_digest_is_caught(self, production):
        entries, manifest = production
        tampered = dict(manifest)
        tampered["manifest_digest"] = "0" * 64
        result = manifest_audit(entries, tampered)
        assert not result["checks"]["manifest_digest_matches"]
        assert not result["all_pass"]

    def test_tampered_manifest_counts_are_caught(self, production):
        entries, manifest = production
        tampered = json.loads(json.dumps(manifest))
        tampered["entry_count"] = 7999
        result = manifest_audit(entries, tampered)
        assert not result["checks"]["entry_count_matches"]
        assert not result["checks"]["manifest_digest_matches"]

    def test_a_forged_handoff_digest_is_caught(self, production):
        entries, manifest = production
        result = manifest_audit(
            entries,
            manifest,
            expected_digests={
                "library_content_digest": "0" * 64,
                "entry_metadata_digest": "0" * 64,
                "manifest_digest": "0" * 64,
            },
        )
        assert not result["checks"]["handoff_library_digest_matches"]
        assert not result["all_pass"]


# ---------------------------------------------------------------------------
# Overlap matrix behaviour
# ---------------------------------------------------------------------------


class TestOverlapAudit:
    def test_the_diagonal_is_exactly_one_for_clean_entries(self):
        result = overlap_audit(list(_sample_entries()))
        assert result["diagonal_failures"] == []
        for family_id in ("F00", "F07", "F11", "F15"):
            assert result["matrix"][family_id][family_id] == 1.0

    def test_a_wrong_family_label_breaks_the_diagonal(self):
        entries = [
            _raw_variant(
                _entry("F00", 0),
                family_id="F04",
                family_key="lightly_defended_deceptive_flag",
                base_setup_id="setup_library_v1:F04:000",
            )
        ]
        result = overlap_audit(entries)
        assert result["diagonal_failures"] == ["F04"]

    def test_large_overlaps_are_attributed_clause_by_clause(self):
        f11_entries = [entry for entry in _sample_entries() if entry.family_id == "F11"]
        result = overlap_audit(f11_entries)
        row = result["matrix"]["F11"]
        assert row["F11"] == 1.0
        for attribution in result["attributions"]:
            assert "target_clause_rates" in attribution
            if attribution["also_satisfies"] == "F15":
                incidence = attribution["f15_feature_incidence_among_satisfying"]
                # F11 forbids front-rank Scouts, which is itself an F15
                # feature, so it must hold for every satisfying member.
                assert incidence["no_front_rank_scouts"] == 1.0


# ---------------------------------------------------------------------------
# Production audit artifacts (gated until the harness has run)
# ---------------------------------------------------------------------------


@requires_audit_artifacts
class TestProductionAuditArtifacts:
    def test_the_audit_reports_pass_with_every_gate_true(self, artifact):
        assert artifact["status"] == "PASS"
        assert artifact["gates_true"] == artifact["gates_total"]
        assert all(artifact["completion_gates"].values())
        assert artifact["audit_version"] == AUDIT_VERSION
        assert artifact["audit"]["status"] == "PASS"
        assert all(gate["pass"] for gate in artifact["audit"]["gates"])

    def test_headline_counts_are_exact(self, artifact):
        assert artifact["setup_count"] == 8000
        assert artifact["split_counts"] == {
            "train": 6400,
            "validation": 800,
            "test": 800,
        }
        assert all(count == 500 for count in artifact["family_counts"].values())

    @requires_production_library
    def test_the_audit_digests_match_the_production_manifest(self, artifact):
        manifest = read_manifest(MANIFEST_PATH)
        assert artifact["library_digest"] == manifest["library_content_digest"]
        assert artifact["entry_metadata_digest"] == manifest["entry_metadata_digest"]
        assert artifact["manifest_digest"] == manifest["manifest_digest"]

    def test_zero_findings_in_every_hard_category(self, artifact):
        per_base = artifact["audit"]["per_base"]
        for key in (
            "inventory_failures",
            "engine_failures",
            "reflected_engine_failures",
            "placement_failures",
            "mobility_failures",
            "reflected_mobility_failures",
            "family_failures",
            "reflected_family_failures",
            "serialization_failures",
            "reflection_roundtrip_failures",
            "canonicalization_failures",
            "fingerprint_failures",
            "trait_failures",
            "identity_failures",
            "seed_failures",
            "version_failures",
            "reflection_symmetric_bases",
        ):
            assert per_base[key] == [], key
        duplicates = artifact["audit"]["duplicates"]
        assert duplicates["exact_duplicate_groups"] == 0
        assert duplicates["reflection_class_duplicate_groups"] == 0
        assert duplicates["stable_id_collisions"] == 0
        assert duplicates["cross_split_class_duplicate_groups"] == 0

    def test_every_frozen_threshold_check_passed(self, artifact):
        thresholds = artifact["audit"]["thresholds"]
        assert thresholds["all_pass"]
        assert thresholds["failed_checks"] == []
        assert thresholds["check_count"] == 199

    def test_the_overlap_diagonal_is_one_and_off_diagonal_is_descriptive(self, artifact):
        overlap = artifact["audit"]["overlap"]
        assert overlap["diagonal_failures"] == []
        largest = overlap["largest_off_diagonal"]
        assert largest is not None and 0.0 <= largest["fraction"] < 1.0

    def test_family_metrics_csv_covers_every_family_and_every_row_passes(self):
        with FAMILY_METRICS_CSV.open() as handle:
            rows = list(csv.DictReader(handle))
        families = {row["family_id"] for row in rows}
        assert families == {f"F{index:02d}" for index in range(16)} | {"ALL"}
        thresholded = [row for row in rows if row["pass"] != ""]
        assert thresholded, "no thresholded rows recorded"
        assert all(row["pass"] == "true" for row in thresholded)
        report_only = [row for row in rows if row["pass"] == ""]
        assert all(row["required"] == "report-only" for row in report_only)

    def test_similarity_csv_reports_zero_cross_split_leakage(self):
        with SIMILARITY_CSV.open() as handle:
            rows = list(csv.DictReader(handle))
        cross_rows = [row for row in rows if row["scope"].startswith("cross_split")]
        assert len(cross_rows) == 3 * 17  # three split pairs x (global + 16 families)
        for row in cross_rows:
            assert row["pairs_below_8"] == "0", row
            assert row["pairs_below_10"] == "0", row
            assert row["offending_ids"] == "", row
        within_rows = [row for row in rows if row["scope"] == "within_family"]
        assert len(within_rows) == 16
        for row in within_rows:
            assert int(row["min"]) >= 6, row
            assert row["offending_ids"] == "", row


# ---------------------------------------------------------------------------
# Full-audit orchestration on a forged mini-library
# ---------------------------------------------------------------------------


class TestAuditLibraryOrchestration:
    def test_a_forged_library_yields_fail_with_named_gates(self):
        original = _entry("F00", 0)
        forged = _content_variant(_entry("F00", 1), original.canonical_setup)
        result = audit_library([original, forged])
        assert result["status"] == "FAIL"
        failing = {gate["metric"] for gate in result["gates"] if not gate["pass"]}
        assert "exact_duplicate_groups" in failing
        assert "total_bases" in failing  # a 2-entry library is not the library

    def test_every_gate_row_carries_metric_required_measured_pass(self):
        original = _entry("F00", 0)
        forged = _content_variant(_entry("F00", 1), original.canonical_setup)
        result = audit_library([original, forged])
        for gate in result["gates"]:
            assert set(gate) == {"metric", "required", "measured", "pass"}
