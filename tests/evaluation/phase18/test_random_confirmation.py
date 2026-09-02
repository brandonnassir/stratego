"""Tests for the Phase 18 G1 random-opponent confirmation (Agent 3).

Every launch requirement in Part C of the Agent 3 instruction has a test here:
frozen checkpoint refusal, deterministic bank generation at exactly 4,096
independent pairs, reflection-class separation from the original bank, one
schedule for both arms, cross-arm case identity, colour-paired bootstrap units,
deterministic 10,000-replicate bootstrap, the strict lower-bound rule, the
failure/missing accounting that can never score as a draw, the retry-safe chunk
path, the unreachability of the sealed Phase 8 test split, and the protection
of the accepted artifacts.

The full 4,096-pair bank and its separation audit run for real: those are the
properties the launch depends on, so they are measured, not sampled.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import phase18_g1_random_confirmation as rc  # noqa: E402

from stratego.engine.constants import BLUE, RED  # noqa: E402
from stratego.evaluation.match_runner import RESULT_ERROR, ERROR_ILLEGAL_ACTION  # noqa: E402
from stratego.evaluation.phase18.confirmation_bank import (  # noqa: E402
    CONFIRMATION_BANK_VERSION,
    CONFIRMATION_PAIRS,
    bank_record,
    bank_root_seed,
    bootstrap_seed,
    build_confirmation_bank,
    pair_canonical_setups,
    pair_class_fingerprint,
    pair_content_fingerprint,
    schedule_root_seed,
    separation_audit,
)
from stratego.evaluation.phase18.noninferiority import paired_unit_delta  # noqa: E402
from stratego.evaluation.phase18.power import (  # noqa: E402
    PowerError,
    noninferiority_power,
    noninferiority_sample_size,
    plan,
)
from stratego.evaluation.setup_bank import SetupBank  # noqa: E402
from stratego.evaluation.statistics import (  # noqa: E402
    StatisticsError,
    build_paired_units,
    synthetic_results,
)
from stratego.setups.identity import (  # noqa: E402
    class_fingerprint,
    content_fingerprint,
    reflect_canonical,
)


@pytest.fixture(scope="module")
def confirmation_bank():
    return build_confirmation_bank()


@pytest.fixture(scope="module")
def original_bank():
    return SetupBank.from_json(rc.REFERENCE_BANK_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def audit(confirmation_bank, original_bank):
    return separation_audit(confirmation_bank, original_bank)


# ---------------------------------------------------------------------------
# Checkpoint digest refusal
# ---------------------------------------------------------------------------


class TestCheckpointRefusal:
    def test_a_wrong_digest_is_blocked(self, tmp_path, monkeypatch):
        impostor = tmp_path / "warmstart_c1_v1.pt"
        impostor.write_bytes(b"not the frozen checkpoint")
        monkeypatch.setattr(rc, "ACCEPTED_CHECKPOINT", impostor)
        with pytest.raises(rc.ConfirmationError, match="BLOCKED.*hashes to"):
            rc.verify_checkpoints()

    def test_a_missing_checkpoint_is_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rc, "CANDIDATE_CHECKPOINT", tmp_path / "gone.pt")
        with pytest.raises(rc.ConfirmationError, match="BLOCKED.*missing"):
            rc.verify_checkpoints()

    def test_the_frozen_paths_carry_the_frozen_identities(self):
        record = rc.verify_checkpoints()
        assert record["reference"]["sha256"] == rc.ACCEPTED_SHA256
        assert record["candidate"]["sha256"] == rc.CANDIDATE_SHA256
        assert record["reference"]["matches_frozen_identity"] is True
        assert record["candidate"]["matches_frozen_identity"] is True


# ---------------------------------------------------------------------------
# Bank: determinism, count, independence
# ---------------------------------------------------------------------------


class TestBankGeneration:
    def test_generation_is_deterministic(self):
        first, second = build_confirmation_bank(64), build_confirmation_bank(64)
        assert first.digest() == second.digest()
        assert [p.to_dict() for p in first.pairs] == [p.to_dict() for p in second.pairs]

    def test_the_namespace_seeds_are_frozen(self):
        """The derived seeds are identities: a change in the derivation chain
        must fail loudly here, not silently rebuild a different bank."""
        assert bank_root_seed() == 6318667561552392983
        assert schedule_root_seed() == 6846249549559311463
        assert bootstrap_seed() == 5234174007292860913

    def test_a_different_root_seed_builds_a_different_bank(self):
        ours = build_confirmation_bank(64)
        other = SetupBank.generate(
            size=64, root_seed=bank_root_seed() + 1, bank_version=CONFIRMATION_BANK_VERSION
        )
        assert ours.digest() != other.digest()

    def test_the_bank_holds_exactly_4096_unique_pairs(self, confirmation_bank):
        assert CONFIRMATION_PAIRS == 4096
        assert len(confirmation_bank.pairs) == 4096
        identifiers = [p.setup_pair_id for p in confirmation_bank.pairs]
        assert identifiers == list(range(4096))

    def test_the_bank_record_is_self_verifying(self):
        record = bank_record(build_confirmation_bank(8))
        assert record["bank_version"] == CONFIRMATION_BANK_VERSION
        assert record["pair_count"] == 8
        assert all(p["derived_seed_recomputes"] for p in record["pairs"])


class TestSeparation:
    def test_the_real_bank_is_independent_of_the_original(self, audit):
        """The launch property itself: zero canonical overlap, zero
        reflection-class overlap, zero internal duplicates, on the full bank."""
        assert audit.confirmation_pairs == 4096
        assert audit.reference_pairs == 1024
        assert audit.reference_bank_version == "evaluation_setup_bank_v1"
        assert audit.internal_content_duplicates == 0
        assert audit.internal_class_duplicates == 0
        assert audit.cross_content_overlap == 0
        assert audit.cross_class_overlap == 0
        assert audit.cross_side_class_overlap == 0
        assert audit.duplicate_pair_ids == 0
        assert audit.reflection_fingerprint_stable is True
        assert audit.structural_problems == []
        assert audit.separated is True

    def test_the_audit_catches_cross_bank_overlap(self, original_bank):
        selfsame = separation_audit(original_bank, original_bank, check_structure=False)
        assert selfsame.cross_content_overlap == 1024
        assert selfsame.cross_class_overlap == 1024
        assert selfsame.separated is False

    def test_the_audit_catches_an_internal_duplicate(self):
        bank = build_confirmation_bank(8)
        twin = dataclasses.replace(
            bank.pairs[1],
            red_setup=bank.pairs[0].red_setup,
            blue_setup=bank.pairs[0].blue_setup,
        )
        doctored = dataclasses.replace(bank, pairs=(bank.pairs[0], twin) + bank.pairs[2:])
        report = separation_audit(
            doctored, build_confirmation_bank(4), check_structure=False
        )
        assert report.internal_content_duplicates == 1
        assert report.internal_class_duplicates == 1
        assert report.separated is False

    def test_a_horizontal_mirror_collides_in_class_space(self, original_bank):
        """Reflection classes are what make the separation meaningful: a
        mirrored board keeps its class fingerprint while its content
        fingerprint moves, so a mirror of an old board cannot slip in."""
        red, _blue = pair_canonical_setups(original_bank.pairs[0])
        mirrored = reflect_canonical(red)
        assert mirrored != tuple(red)
        assert content_fingerprint(mirrored) != content_fingerprint(red)
        assert class_fingerprint(mirrored) == class_fingerprint(red)


# ---------------------------------------------------------------------------
# One schedule, both arms
# ---------------------------------------------------------------------------


class TestSchedule:
    def test_the_schedule_is_deterministic_and_shared(self):
        from stratego.evaluation.match_spec import schedule_digest

        _u1, first, cand, opp = rc.schedule_for(8)
        _u2, second, _c, _o = rc.schedule_for(8)
        assert schedule_digest(first) == schedule_digest(second)
        assert len(first) == 16
        assert cand.token == "phase6_c1_warmstart_greedy@0.2.0+float32"
        assert opp.token == "random_legal@1.0.0"

    def test_every_match_binds_the_confirmation_bank(self, confirmation_bank):
        from stratego.evaluation.match_spec import validate_schedule

        _units, matches, _c, _o = rc.schedule_for(8)
        assert all(m.setup_bank_version == CONFIRMATION_BANK_VERSION for m in matches)
        assert validate_schedule(matches, confirmation_bank) == []

    def test_both_colours_appear_once_per_pair(self):
        _units, matches, _c, _o = rc.schedule_for(8)
        by_pair: dict = {}
        for match in matches:
            by_pair.setdefault(match.setup_pair_id, []).append(match.candidate_color)
        assert all(sorted(colors) == [RED, BLUE] for colors in by_pair.values())


class TestArmIdentity:
    def test_identical_cases_prove_identity(self):
        candidate = synthetic_results([(1.0, 0.0), (0.5, 0.5)])
        reference = synthetic_results([(0.0, 0.0), (1.0, 1.0)])
        proof = rc.prove_arm_identity(candidate, reference)
        assert proof["problems"] == []
        assert proof["identical_match_ids"] is True
        assert proof["field_mismatches"] == {}

    def test_a_moved_seed_is_caught(self):
        candidate = list(synthetic_results([(1.0, 0.0)]))
        reference = synthetic_results([(1.0, 0.0)])
        candidate[0] = dataclasses.replace(candidate[0], candidate_seed=99)
        proof = rc.prove_arm_identity(candidate, reference)
        assert proof["field_mismatches"] == {"candidate_seed": 1}
        assert proof["problems"]

    def test_a_different_opponent_is_caught(self):
        candidate = synthetic_results([(1.0, 0.0)], opponent="other@1.0.0")
        reference = synthetic_results([(1.0, 0.0)])
        proof = rc.prove_arm_identity(candidate, reference)
        assert proof["problems"]

    def test_a_different_case_set_is_caught(self):
        candidate = synthetic_results([(1.0, 0.0), (0.5, 0.5)])
        reference = synthetic_results([(1.0, 0.0)])
        proof = rc.prove_arm_identity(candidate, reference)
        assert any("different match ids" in p for p in proof["problems"])


# ---------------------------------------------------------------------------
# Paired unit semantics and the receipts recomputation
# ---------------------------------------------------------------------------


def _receipt(pair_id: int, color: int, score, *, errored=False, match=None):
    return {
        "match_id": match or f"m{pair_id}c{color}",
        "setup_pair_id": pair_id,
        "candidate_color": color,
        "candidate_score": score,
        "errored": errored,
    }


def _write_receipts(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class TestPairedUnits:
    def test_both_colours_are_one_bootstrap_unit(self):
        units = build_paired_units(synthetic_results([(1.0, 0.0)]))
        assert len(units) == 1
        assert units[0].red_score == 1.0
        assert units[0].blue_score == 0.0
        assert units[0].score == 0.5

    def test_receipts_recompute_the_unit_scores(self, tmp_path):
        path = _write_receipts(
            tmp_path / "r.jsonl",
            [_receipt(0, RED, 1.0), _receipt(0, BLUE, 0.0),
             _receipt(1, RED, 0.5), _receipt(1, BLUE, 0.5)],
        )
        recomputed = rc.recompute_from_receipts(path)
        assert recomputed["rows"] == 4
        assert recomputed["unit_scores"] == {0: 0.5, 1: 0.5}

    def test_a_missing_colour_is_blocked(self, tmp_path):
        path = _write_receipts(tmp_path / "r.jsonl", [_receipt(0, RED, 1.0)])
        with pytest.raises(rc.ConfirmationError, match="missing a colour"):
            rc.recompute_from_receipts(path)

    def test_an_errored_receipt_is_blocked_not_scored(self, tmp_path):
        path = _write_receipts(
            tmp_path / "r.jsonl",
            [_receipt(0, RED, None, errored=True), _receipt(0, BLUE, 0.0)],
        )
        with pytest.raises(rc.ConfirmationError, match="carries no score"):
            rc.recompute_from_receipts(path)

    def test_a_duplicated_colour_is_blocked(self, tmp_path):
        path = _write_receipts(
            tmp_path / "r.jsonl",
            [_receipt(0, RED, 1.0, match="a"), _receipt(0, RED, 0.0, match="b")],
        )
        with pytest.raises(rc.ConfirmationError, match="repeat colour"):
            rc.recompute_from_receipts(path)


# ---------------------------------------------------------------------------
# Bootstrap determinism and the strict decision rule
# ---------------------------------------------------------------------------


class TestDecision:
    def test_ten_thousand_replicates_are_deterministic(self):
        import random

        rng = random.Random(7)
        candidate = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(101)]
        reference = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(101)]
        seed = bootstrap_seed()
        first = paired_unit_delta(candidate, reference, seed=seed, replicates=10_000)
        second = paired_unit_delta(candidate, reference, seed=seed, replicates=10_000)
        assert (first.lower, first.upper, first.delta) == (
            second.lower, second.upper, second.delta,
        )
        assert first.replicates == 10_000
        moved = paired_unit_delta(candidate, reference, seed=seed + 1, replicates=10_000)
        assert (moved.lower, moved.upper) != (first.lower, first.upper)

    def test_the_lower_bound_decides_in_the_right_direction(self):
        above = SimpleNamespace(lower=-0.0099, upper=0.02, delta=0.0)
        below = SimpleNamespace(lower=-0.0101, upper=0.02, delta=0.0)
        assert rc.strict_verdict(above, rc.SIGNED_MARGIN)["non_inferior"] is True
        assert rc.strict_verdict(below, rc.SIGNED_MARGIN)["non_inferior"] is False

    def test_the_rule_is_strict_at_the_margin(self):
        """The instruction says 'strictly greater than -0.010'. The original
        G1 dialect (`assess_margin`) reads `>=`; the confirmation must not."""
        from stratego.evaluation.phase18.noninferiority import (
            DIRECTION_DELTA_MIN,
            DeltaInterval,
            assess_margin,
        )

        at_margin = DeltaInterval(
            candidate=0.95, reference=0.96, delta=-0.01, lower=-0.010, upper=0.0,
            confidence=0.95, replicates=10, seed=1, resampling_unit="paired_unit",
            sample_size=4, method="paired_unit_difference_bootstrap",
        )
        strict = rc.strict_verdict(at_margin, rc.SIGNED_MARGIN)
        original = assess_margin(
            "vs_random_ewr", at_margin, margin=rc.SIGNED_MARGIN,
            direction=DIRECTION_DELTA_MIN,
        )
        assert strict["non_inferior"] is False
        assert original.non_inferior is True  # the dialects differ exactly here

    def test_an_upper_bound_cannot_decide(self):
        generous_upper = SimpleNamespace(lower=-0.02, upper=0.5, delta=0.2)
        assert rc.strict_verdict(generous_upper, rc.SIGNED_MARGIN)["non_inferior"] is False


class TestPower:
    def test_the_frozen_sizing_reproduces_the_audit(self):
        assert noninferiority_sample_size(target_power=0.80) == 2815
        assert noninferiority_sample_size(target_power=0.90) == 3769
        sizing = plan(4096)
        assert sizing.minimum_n == 3769
        assert sizing.frozen_n == 4096
        assert sizing.power_at_frozen_n > 0.90

    def test_the_planning_sd_reproduces_the_original_instrument(self):
        """At 1,024 pairs the half-width must be Agent 2's measured 0.011599,
        or the planning SD is not the instrument the audit reasoned about."""
        assert plan(1024).half_width_at_frozen_n == pytest.approx(0.011599, abs=5e-7)

    def test_power_needs_a_certifiable_configuration(self):
        with pytest.raises(PowerError):
            noninferiority_sample_size(margin=0.0)
        with pytest.raises(PowerError):
            noninferiority_sample_size(true_delta=-0.010)
        with pytest.raises(PowerError):
            noninferiority_power(0)


# ---------------------------------------------------------------------------
# Accounting: failures and gaps can never pass
# ---------------------------------------------------------------------------


def _specs(rows):
    return tuple(SimpleNamespace(match_id=row.match_id) for row in rows)


def _errored(row):
    return dataclasses.replace(
        row,
        candidate_result=RESULT_ERROR,
        candidate_score=None,
        winner=None,
        draw=False,
        policy_error="synthetic failure",
        policy_error_category=ERROR_ILLEGAL_ACTION,
    )


class TestAccounting:
    def test_a_complete_arm_reconciles(self):
        rows = synthetic_results([(1.0, 0.0), (0.5, 0.5)])
        accounting = rc.reconcile(_specs(rows), rows)
        assert accounting["planned"] == 4
        assert accounting["completed"] == 4
        assert accounting["reconciles"] is True
        assert accounting["complete_for_primary"] is True

    def test_a_missing_game_is_missing_not_a_draw(self):
        rows = synthetic_results([(1.0, 0.0), (0.5, 0.5)])
        accounting = rc.reconcile(_specs(rows), rows[:-1])
        assert accounting["missing"] == 1
        assert accounting["completed"] == 3
        assert accounting["reconciles"] is True
        assert accounting["complete_for_primary"] is False

    def test_a_failed_game_is_failed_not_a_draw(self):
        rows = list(synthetic_results([(1.0, 0.0), (0.5, 0.5)]))
        rows[1] = _errored(rows[1])
        accounting = rc.reconcile(_specs(rows), rows)
        assert accounting["failed"] == 1
        assert accounting["completed"] == 3
        assert accounting["complete_for_primary"] is False
        # And the statistics refuse the same rows outright.
        with pytest.raises(StatisticsError, match="policy error"):
            build_paired_units(rows, allow_policy_errors=False)

    def test_an_unplanned_or_duplicated_row_breaks_reconciliation(self):
        rows = synthetic_results([(1.0, 0.0)])
        accounting = rc.reconcile(_specs(rows[:-1]), rows)
        assert accounting["unplanned"] == 1
        assert accounting["reconciles"] is False
        doubled = rc.reconcile(_specs(rows), list(rows) + [rows[0]])
        assert doubled["duplicates"] == 1
        assert doubled["reconciles"] is False


# ---------------------------------------------------------------------------
# The retry-safe chunk path
# ---------------------------------------------------------------------------


def _fake_matches(count):
    return tuple(SimpleNamespace(match_id=f"match-{index:03d}") for index in range(count))


def _fake_runner(journal, fail_on_chunk=None):
    def runner(chunk):
        number = len(journal)
        if fail_on_chunk is not None and number == fail_on_chunk:
            raise RuntimeError("transient worker crash")
        journal.append([spec.match_id for spec in chunk])
        rows = tuple(SimpleNamespace(match_id=spec.match_id) for spec in chunk)
        return rows, {"matches": len(rows)}

    return runner


class TestRetrySafety:
    def test_a_completed_run_is_reused_without_replaying(self, tmp_path):
        matches = _fake_matches(8)
        first_journal: list = []
        first, _ = rc.play_chunks(
            matches, tmp_path, _fake_runner(first_journal), chunk_units=1, label="t"
        )
        assert len(first_journal) == 4
        second_journal: list = []
        second, reports = rc.play_chunks(
            matches, tmp_path, _fake_runner(second_journal), chunk_units=1, label="t"
        )
        assert second_journal == []  # nothing replayed
        assert [row.match_id for row in second] == [row.match_id for row in first]
        assert all(report["reused"] for report in reports)

    def test_a_crash_resumes_by_replaying_only_the_missing_chunks(self, tmp_path):
        matches = _fake_matches(8)
        crashing: list = []
        with pytest.raises(RuntimeError, match="transient worker crash"):
            rc.play_chunks(
                matches, tmp_path, _fake_runner(crashing, fail_on_chunk=2),
                chunk_units=1, label="t",
            )
        survivors = sorted(tmp_path.glob("chunk_*.pkl"))
        assert len(survivors) == 2  # chunks 0 and 1 persisted, nothing deleted
        resumed: list = []
        rows, reports = rc.play_chunks(
            matches, tmp_path, _fake_runner(resumed), chunk_units=1, label="t"
        )
        assert len(resumed) == 2  # only chunks 2 and 3 were played
        assert [row.match_id for row in rows] == [spec.match_id for spec in matches]
        assert [report["reused"] for report in reports] == [True, True, False, False]
        assert len(sorted(tmp_path.glob("chunk_*.pkl"))) == 4

    def test_a_foreign_chunk_file_is_refused_not_scored(self, tmp_path):
        matches = _fake_matches(2)
        journal: list = []
        rc.play_chunks(matches, tmp_path, _fake_runner(journal), chunk_units=1, label="t")
        path = next(tmp_path.glob("chunk_0000_*.pkl"))
        with open(path, "rb") as stream:
            stored = pickle.load(stream)
        stored["results"] = (SimpleNamespace(match_id="someone-else"),)
        with open(path, "wb") as stream:
            pickle.dump(stored, stream)
        with pytest.raises(rc.ConfirmationError, match="refusing to reuse"):
            rc.play_chunks(matches, tmp_path, _fake_runner([]), chunk_units=1, label="t")


# ---------------------------------------------------------------------------
# Sealing and protected artifacts
# ---------------------------------------------------------------------------


class TestSealedPathUnreachable:
    """The confirmation may not open one sealed Phase 8 test example. The
    driver has no code path to the corpus at all, and this pins that."""

    FORBIDDEN = (
        "run_phase8_agent07",
        "stage_test_metrics",
        "synthetic_warmstart_corpus",
        "data/stratego_phase8",
        "warmstart_dataset",
        "load_warmstart_examples",
        "corpus",
    )

    def test_the_driver_references_no_corpus_or_test_split_path(self):
        source = (REPOSITORY_ROOT / "scripts" / "phase18_g1_random_confirmation.py").read_text()
        for token in self.FORBIDDEN:
            assert token not in source, f"driver must not reference {token!r}"

    def test_the_confirmation_modules_reference_no_corpus_path(self):
        for name in ("confirmation_bank.py", "power.py"):
            source = (
                REPOSITORY_ROOT / "stratego" / "evaluation" / "phase18" / name
            ).read_text()
            for token in self.FORBIDDEN:
                assert token not in source, f"{name} must not reference {token!r}"

    def test_the_planned_sealed_access_is_zero(self):
        contract_path = (
            REPOSITORY_ROOT / "reports" / "phase18" / rc.CONTRACT_NAME
        )
        if not contract_path.exists():
            pytest.skip("contract not frozen yet")
        contract = json.loads(contract_path.read_text())
        assert contract["sealed_test_access"]["planned"] == 0


class TestProtectedArtifacts:
    def test_the_protected_set_hashes_to_the_frozen_identities(self):
        digests = rc.protected_digests()
        assert digests["checkpoints/phase8/warmstart_c1_v1.pt"] == rc.ACCEPTED_SHA256
        assert digests["g1_candidate:warmstart_c1_v1.pt"] == rc.CANDIDATE_SHA256
        assert len(digests) == len(rc.PROTECTED) + 1

    def test_a_missing_protected_artifact_is_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rc, "ACCEPTED_INSTALL_ROOT", tmp_path)
        with pytest.raises(rc.ConfirmationError, match="BLOCKED.*missing"):
            rc.protected_digests()

    def test_a_changed_protected_artifact_is_detectable(self, tmp_path, monkeypatch):
        """The run compares before/after digest maps; a byte change flips them."""
        root = tmp_path / "install"
        for name in rc.PROTECTED:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"accepted")
        candidate = tmp_path / "candidate.pt"
        candidate.write_bytes(b"candidate")
        monkeypatch.setattr(rc, "ACCEPTED_INSTALL_ROOT", root)
        monkeypatch.setattr(rc, "CANDIDATE_CHECKPOINT", candidate)
        before = rc.protected_digests()
        (root / rc.PROTECTED[0]).write_bytes(b"tampered")
        after = rc.protected_digests()
        assert before != after


# ---------------------------------------------------------------------------
# The frozen contract, once it exists
# ---------------------------------------------------------------------------


class TestFrozenContract:
    def test_the_committed_contract_matches_the_frozen_design(self, confirmation_bank):
        contract_path = REPOSITORY_ROOT / "reports" / "phase18" / rc.CONTRACT_NAME
        if not contract_path.exists():
            pytest.skip("contract not frozen yet")
        contract = json.loads(contract_path.read_text())
        hypothesis = contract["primary_hypothesis"]
        assert hypothesis["margin"] == -0.010
        assert hypothesis["confidence"] == 0.95
        assert hypothesis["replicates"] == 10_000
        assert "strictly greater than -0.010" in hypothesis["decision_rule"]
        assert contract["bank"]["pair_count"] == 4096
        assert contract["bank"]["digest"] == confirmation_bank.digest()
        assert contract["bank"]["root_seed"] == bank_root_seed()
        assert contract["bootstrap_seed"] == bootstrap_seed()
        assert contract["power"]["minimum_n"] == 3769
        assert contract["power"]["frozen_n"] == 4096
        assert contract["arm_order"] == ["reference", "candidate"]
        assert contract["schedule"]["games_per_arm"] == 8192
        assert contract["schedule"]["total_games_both_arms"] == 16384
        assert contract["bank"]["separation_audit"]["separated"] is True

    def test_fingerprints_bind_the_receipts_to_the_bank(self, confirmation_bank):
        pair = confirmation_bank.pairs[0]
        assert pair_content_fingerprint(pair) != pair_class_fingerprint(pair)
        assert len(pair_content_fingerprint(pair)) == 64
