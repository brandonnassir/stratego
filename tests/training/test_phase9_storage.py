"""Phase 9 Agent 2: rollout storage resolution, and its distance from identity.

Two things are being protected here.

The first is the redirect itself: Phase 9 production rollout bytes are meant
to live on an external volume, and a resolution order that silently prefers
the wrong root would send a 60-iteration run to the boot disk. The order is
the accepted Phase 8 precedent — environment override, then durable pointer
file, then repository default — and each rung is tested in isolation.

The second is the rule that makes the redirect safe: **a path is a
diagnostic, never an identity**. The tests below pin that a schedule digest,
a game id and a scheduled record are all completely unmoved by any storage
configuration, including one pointing at a volume that does not exist.

Nothing here requires the external drive to be plugged in; every filesystem
assertion runs against a temporary directory.
"""

import json
import os
from pathlib import Path

import pytest

from stratego.training import phase9_schedule as psch
from stratego.training import phase9_storage as pstore


@pytest.fixture
def clean_environment(monkeypatch):
    monkeypatch.delenv(pstore.PHASE9_ROLLOUT_ROOT_ENV, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


class TestResolution:
    def test_environment_override_wins(self, clean_environment, tmp_path):
        clean_environment.setenv(pstore.PHASE9_ROLLOUT_ROOT_ENV, str(tmp_path / "ext"))
        assert pstore.default_rollout_root() == tmp_path / "ext"
        assert pstore.describe_rollout_root()["source"] == "environment"

    def test_pointer_file_is_used_when_no_environment_override(
        self, clean_environment, tmp_path
    ):
        pointer = tmp_path / "pointer.txt"
        pointer.write_text(f"{tmp_path / 'recorded'}\n")
        clean_environment.setattr(pstore, "repository_root", lambda: tmp_path)
        clean_environment.setattr(pstore, "PHASE9_ROLLOUT_ROOT_POINTER", "pointer.txt")
        assert pstore.default_rollout_root() == tmp_path / "recorded"
        assert pstore.describe_rollout_root()["source"] == "pointer_file"

    def test_repository_default_is_the_fallback(self, clean_environment, tmp_path):
        clean_environment.setattr(pstore, "repository_root", lambda: tmp_path)
        clean_environment.setattr(pstore, "PHASE9_ROLLOUT_ROOT_POINTER", "absent.txt")
        assert pstore.default_rollout_root() == tmp_path / "data/phase9/rollouts"
        assert pstore.describe_rollout_root()["source"] == "repository_default"

    def test_an_empty_pointer_file_does_not_win(self, clean_environment, tmp_path):
        (tmp_path / "pointer.txt").write_text("   \n")
        clean_environment.setattr(pstore, "repository_root", lambda: tmp_path)
        clean_environment.setattr(pstore, "PHASE9_ROLLOUT_ROOT_POINTER", "pointer.txt")
        # A blank redirect is a configuration mistake, not an instruction to
        # write to the filesystem root.
        assert pstore.default_rollout_root() == tmp_path / "data/phase9/rollouts"

    def test_the_repository_default_is_the_contract_namespace(self):
        assert pstore.DEFAULT_PHASE9_ROLLOUT_ROOT == "data/phase9/rollouts"

    def test_description_is_json_serializable_and_states_the_identity_rule(
        self, clean_environment, tmp_path
    ):
        clean_environment.setenv(pstore.PHASE9_ROLLOUT_ROOT_ENV, str(tmp_path))
        description = pstore.describe_rollout_root()
        assert json.loads(json.dumps(description)) == description
        assert "never an identity" in description["identity_rule"]

    def test_iteration_directories_are_a_layout_convention(self, tmp_path):
        path = pstore.namespace_rollout_directory(tmp_path, "canonical", 7)
        assert path == tmp_path / "canonical" / "iteration_007"
        with pytest.raises(pstore.Phase9StorageError):
            pstore.namespace_rollout_directory(tmp_path, "canonical", 0)


# ---------------------------------------------------------------------------
# The identity rule
# ---------------------------------------------------------------------------


class TestPathIsNotIdentity:
    @pytest.mark.parametrize(
        "root",
        (
            "/Volumes/SomeExternalDrive/data/phase9/rollouts",
            "/completely/nonexistent/volume",
            "data/phase9/rollouts",
        ),
    )
    def test_no_storage_configuration_moves_a_schedule_digest(
        self, clean_environment, root
    ):
        reference_digest = psch.iteration_schedule_digest("canonical", 4)
        reference_record = psch.scheduled_game_record("canonical", 4, "historical", 9)
        clean_environment.setenv(pstore.PHASE9_ROLLOUT_ROOT_ENV, root)
        assert psch.iteration_schedule_digest("canonical", 4) == reference_digest
        assert (
            psch.scheduled_game_record("canonical", 4, "historical", 9)
            == reference_record
        )
        assert psch.population_digest() == psch.population_digest()

    def test_a_game_id_contains_no_path(self):
        record = psch.scheduled_game_record("canonical", 12, "rule", 3)
        serialized = json.dumps(record.to_dict())
        assert "/" not in record.phase9_game_id
        assert "Volumes" not in serialized
        assert "rollouts" not in serialized

    def test_relocation_leaves_the_run_digest_unchanged(
        self, clean_environment, tmp_path
    ):
        before = psch.run_schedule_digest("pilot_p9a")
        clean_environment.setenv(
            pstore.PHASE9_ROLLOUT_ROOT_ENV, str(tmp_path / "elsewhere")
        )
        assert psch.run_schedule_digest("pilot_p9a") == before


# ---------------------------------------------------------------------------
# Capacity, writability and the recommendation
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_volume_diagnostics_probe_an_unborn_directory(self, tmp_path):
        target = tmp_path / "not" / "yet" / "created"
        report = pstore.volume_diagnostics(target)
        assert report["path_exists"] is False
        assert report["existing_ancestor_probed"] == str(tmp_path)
        assert report["free_bytes"] > 0
        assert report["total_bytes"] >= report["free_bytes"]

    def test_write_probe_round_trips_and_cleans_up(self, tmp_path):
        target = tmp_path / "rollouts"
        report = pstore.check_writable(target)
        assert report["writable"]
        assert report["directory_created"]
        assert not report["problems"]
        assert list(target.iterdir()) == []

    def test_write_probe_reports_an_unwritable_location(self, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            report = pstore.check_writable(locked / "rollouts")
            assert not report["writable"]
            assert report["problems"]
        finally:
            locked.chmod(0o700)

    def test_projection_scales_the_measured_phase_8_rate(self):
        projection = pstore.projected_rollout_bytes(172_032)
        per_game = pstore.PHASE8_MEASURED_BYTES_PER_GAME
        assert projection["base_bytes"] == int(per_game * 172_032)
        assert projection["projected_bytes"] == int(
            per_game * 172_032 * pstore.PHASE9_VOLUME_SAFETY_FACTOR
        )
        assert projection["safety_factor"] > 1

    def test_projection_of_no_games_is_zero(self):
        assert pstore.projected_rollout_bytes(0)["projected_bytes"] == 0

    def test_negative_game_counts_are_refused(self):
        with pytest.raises(pstore.Phase9StorageError):
            pstore.projected_rollout_bytes(-1)

    def test_a_writable_volume_with_headroom_is_recommended(self, tmp_path):
        report = pstore.evaluate_storage_target(tmp_path / "rollouts", 1000)
        assert report["recommended"]
        assert not report["problems"]
        assert report["observed_headroom_factor"] > pstore.REQUIRED_HEADROOM_FACTOR

    def test_a_volume_without_headroom_is_not_recommended(self, tmp_path):
        # Ask for a corpus far larger than any local disk: the recommendation
        # must fail on capacity even though the location is perfectly writable.
        report = pstore.evaluate_storage_target(tmp_path / "rollouts", 10**12)
        assert not report["recommended"]
        assert any("headroom" in problem for problem in report["problems"])
        assert report["write_probe"]["writable"]

    def test_evaluation_carries_the_identity_rule_forward(self, tmp_path):
        report = pstore.evaluate_storage_target(tmp_path, 1000)
        assert report["identity_rule"] == pstore.STORAGE_IDENTITY_RULE
        assert json.loads(json.dumps(report)) == report


# ---------------------------------------------------------------------------
# Live volume (only when one is actually mounted)
# ---------------------------------------------------------------------------


EXTERNAL_ROOT = Path("/Volumes")


class TestLiveExternalVolume:
    def test_mounted_external_volumes_report_honest_capacity(self):
        candidates = [
            path
            for path in (EXTERNAL_ROOT.iterdir() if EXTERNAL_ROOT.is_dir() else [])
            if path.is_dir() and not path.is_symlink() and os.path.ismount(path)
        ]
        if not candidates:
            pytest.skip("no external volume is mounted on this machine")
        for volume in candidates:
            report = pstore.volume_diagnostics(volume)
            assert report["total_bytes"] > 0
            assert 0 <= report["free_bytes"] <= report["total_bytes"]
            assert report["mount_point"] == str(volume)
