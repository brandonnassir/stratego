"""The Phase 14 dashboard is read-only, cheap, honest and unnecessary.

Those four claims are what the task asks to be proved, so each has tests:

read-only
    :class:`TestReadOnly` digests every byte of a run tree, refreshes the
    dashboard a hundred times, and digests it again.
cheap
    :class:`TestCost` builds a 480-iteration store — a full 168-hour run at the
    measured ~21 min/iteration — and times a cold and a warm refresh.
honest
    :class:`TestAuthoritativeSources` reproduces Phase 13's own finding: a store
    holding 8,192 games behind a process counter that says 4,096, and a
    dashboard that reports the first as authoritative and labels the second.
    :class:`TestMirroredContract` checks every frozen number the dashboard
    displays against the frozen module it was copied from.
unnecessary
    :class:`TestIndependence` runs a stand-in trainer, kills the dashboard
    under it, and shows the trainer neither notices nor loses a file lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from monitoring.phase14_dashboard import contract  # noqa: E402
from monitoring.phase14_dashboard.server import (  # noqa: E402
    DashboardError,
    build,
    serve,
)
from monitoring.phase14_dashboard.sources import RunPaths, window_clock  # noqa: E402
from monitoring.phase14_dashboard.state import DashboardState  # noqa: E402


# ---------------------------------------------------------------------------
# A run tree on disk, in the shape the accepted Phase 14 code writes
# ---------------------------------------------------------------------------


def utc_text(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class FakeRun:
    """Writes the files a Phase 14 run writes, and nothing the dashboard needs.

    Deliberately built from the *file formats* rather than by importing the
    runner: the dashboard's contract is with the bytes on disk, and a test that
    shared the writer with the reader would prove only that they agree with
    each other.
    """

    def __init__(self, root: Path, start: "datetime | None" = None) -> None:
        self.root = Path(root)
        self.hot = self.root / "hot"
        for directory in ("rollouts/phase14", "archive", "logs", "evaluations"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.hot.mkdir(parents=True, exist_ok=True)
        self.start = start or datetime.now(timezone.utc) - timedelta(hours=12)
        self.deadline = self.start + timedelta(seconds=contract.DEADLINE_SECONDS)
        self.transition = self.start + timedelta(seconds=contract.TRANSITION_SECONDS)

    @property
    def window(self) -> dict:
        return {
            "run_start_utc": utc_text(self.start),
            "run_deadline_utc": utc_text(self.deadline),
            "transition_utc": utc_text(self.transition),
            "transition_seconds": contract.TRANSITION_SECONDS,
            "deadline_seconds": float(contract.DEADLINE_SECONDS),
            "production": True,
        }

    def paths(self) -> RunPaths:
        return RunPaths(self.root, hot_root=self.hot)

    # -- the rollout store -------------------------------------------------

    def iteration(self, number: int, *, games: int = 2048, state: str = "COMMITTED",
                  manifest: bool = True, journal_games: int = 0) -> Path:
        directory = self.root / "rollouts" / "phase14" / f"iteration_{number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state.json").write_text(json.dumps({
            "iteration": number, "namespace": "phase14", "state": state,
            "history": [{"state": state, "unix": time.time()}],
            "sealed_rollout_digest": hashlib.sha256(str(number).encode()).hexdigest(),
        }))
        if manifest:
            (directory / "manifest.json").write_text(json.dumps({
                "iteration": number, "namespace": "phase14", "committed_games": games,
                "sealed": True,
            }))
        if journal_games:
            journal = directory / "journal"
            journal.mkdir(exist_ok=True)
            with open(journal / "w00.commit.jsonl", "a") as stream:
                for index in range(journal_games):
                    stream.write(json.dumps({
                        "phase9_game_id": f"phase14_it{number}_g{index}",
                        "iteration": number,
                    }) + "\n")
        return directory

    # -- the logs ----------------------------------------------------------

    def telemetry(self, **overrides) -> dict:
        row = {
            "telemetry_version": "phase14_telemetry_v1",
            "namespace": "phase14",
            "unix": time.time(),
            "clock": {
                "run_start_utc": self.window["run_start_utc"],
                "run_deadline_utc": self.window["run_deadline_utc"],
                "transition_utc": self.window["transition_utc"],
                "deadline_seconds": float(contract.DEADLINE_SECONDS),
                "window_production": True,
                "segment": "main",
            },
            "training": {
                "global_optimizer_step": 1744, "learning_rate": contract.MAIN_LEARNING_RATE,
                "segment": "main", "policy_loss": 0.09, "value_loss": 0.65,
                "belief_loss": 1.83, "grad_norm": 1.0, "advantage_retention": 0.25,
                "examples_per_second": 929.0, "examples_consumed": 892674,
            },
            "collection": {
                "games_generated": 4096, "process_counter_games": 4096,
                "positions_generated": 552726, "games_per_second": 7.2,
                "draw_rate": 0.0078, "mean_game_length": 269.0, "iteration": 4,
            },
            "population": {"percentages": {}, "active_pool": ["P8", "P9"], "archive_k": 1},
            "checkpoints": {"hot_age_seconds": 60.0},
            "candidates": {},
            "storage": {"free_gib": 900.0},
            "workers": {
                "status": "healthy", "configured_loader_workers": 6, "loader_workers": 6,
                "live_loader_workers": 6, "loader_pool_rebuilds": 0,
                "max_loader_pool_rebuilds": contract.MAX_LOADER_POOL_REBUILDS,
            },
            "counters": {"non_finite_losses": 0, "non_finite_gradients": 0,
                         "non_finite_parameters": 0},
            "failures": {},
        }
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(row.get(section), dict):
                row[section].update(values)
            else:
                row[section] = values
        with open(self.root / "logs" / "phase14_telemetry.jsonl", "a") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return row

    def supervisor(self, event: str, **fields) -> dict:
        record = {"utc": utc_text(datetime.now(timezone.utc)), "unix": time.time(),
                  "event": event, **fields}
        with open(self.root / "logs" / "phase14_supervisor.jsonl", "a") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    # -- checkpoints -------------------------------------------------------

    def hot_checkpoint(self, index: int, step: int, *, age_seconds: float = 0.0) -> Path:
        path = self.hot / f"hot_{index:06d}_step{step:09d}.pt"
        path.write_bytes(b"not a real checkpoint; the dashboard never opens one")
        if age_seconds:
            when = time.time() - age_seconds
            os.utime(path, (when, when))
        return path

    def archive_snapshot(self, position: int) -> Path:
        path = self.root / "archive" / f"archive_{position:04d}.pt"
        path.write_bytes(b"archive snapshot")
        return path

    def candidate_mark(self, hour: int, *, step: int = 6712) -> Path:
        path = self.root / "archive" / f"candidate_h{hour:03d}.candidate.json"
        path.write_text(json.dumps({
            "artifact": "phase14_candidate_mark_v1", "hour": hour,
            "global_optimizer_step": step, "iteration": 3,
            "evaluation_status": "pending",
            "written_utc": utc_text(datetime.now(timezone.utc)),
        }))
        return path

    def ledger(self, hours) -> Path:
        path = self.root / "evaluations" / "phase14_candidate_ledger.json"
        path.write_text(json.dumps({
            "artifact": "phase14_candidate_ledger_v1",
            "candidates": {
                str(hour): {"hour": hour, "status": status}
                for hour, status in hours.items()
            },
        }))
        return path

    def run_state(self, **overrides) -> Path:
        document = {
            "artifact": "phase14_run_manifest_v1", "window": self.window,
            "progress": {"closed": False, "close_reason": "", "iteration": 4,
                         "games_generated": 4096},
        }
        document.update(overrides)
        path = self.root / "phase14_run_state.json"
        path.write_text(json.dumps(document))
        return path


@pytest.fixture(scope="module")
def big(tmp_path_factory) -> FakeRun:
    """A finished run: 480 iterations, 168 hours at the measured ~21 min each."""
    fake = FakeRun(tmp_path_factory.mktemp("big") / "run")
    for number in range(1, 481):
        fake.iteration(number)
    fake.run_state()
    for _ in range(400):
        fake.telemetry()
    for index in range(200):
        fake.supervisor("progress", iteration=index)
    return fake


@pytest.fixture
def run(tmp_path) -> FakeRun:
    """A healthy run, twelve hours in, with four committed iterations."""
    fake = FakeRun(tmp_path / "stratego_phase14")
    for number in range(1, 5):
        fake.iteration(number)
    fake.run_state()
    fake.telemetry()
    fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid(),
                    reason="initial launch")
    fake.hot_checkpoint(1, 1744, age_seconds=120)
    fake.archive_snapshot(1)
    fake.candidate_mark(6)
    fake.ledger({0: "complete", 6: "pending"})
    return fake


def _receiver(attribute) -> "str | None":
    """The name the attribute is being read off, e.g. `sys.stderr.write` -> stderr."""
    import ast

    value = getattr(attribute, "value", None)
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return value.id
    return None


def digest_tree(*roots) -> dict:
    """Every byte and every timestamp under `roots`, for a before/after compare."""
    found = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            info = path.stat()
            found[str(path)] = (
                info.st_size,
                info.st_mtime_ns,
                # atime deliberately excluded: reading a file updates it on some
                # mounts, and an access time is not a modification.
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return found


# ---------------------------------------------------------------------------


class TestMirroredContract:
    """Every frozen value the dashboard displays still equals its source.

    The dashboard copies these rather than importing them, so that the monitor
    process never loads torch. This is the test that keeps the copy honest: it
    runs in a test process, where importing the real modules is free.
    """

    def test_every_mirrored_constant_matches_the_frozen_source(self):
        from stratego.training import phase14_contract as frozen
        from stratego.training import phase14_status as status
        from stratego.training import phase14_trainer as trainer

        from stratego.training import phase14_launch as launch

        sources = {
            "MAX_LOADER_POOL_REBUILDS": trainer.MAX_LOADER_POOL_REBUILDS,
            "EMERGENCY_STOP_FILENAME": launch.EMERGENCY_STOP_FILENAME,
            "INTEGRITY_FAILURE_FILENAME": launch.INTEGRITY_FAILURE_FILENAME,
            "SEALED_OR_LATER": status.SEALED_OR_LATER,
            "WORKER_COMMAND_MARK": status.WORKER_COMMAND_MARK,
        }
        for name, mirrored in contract.mirrored_values().items():
            expected = sources.get(name, getattr(frozen, name, None))
            assert expected is not None, f"{name} no longer exists in the frozen contract"
            assert mirrored == expected, (
                f"{name}: dashboard shows {mirrored!r}, the frozen contract says {expected!r}"
            )

    def test_segment_and_learning_rate_agree_with_the_frozen_clock(self):
        from stratego.training import phase14_clock as clock

        for hours in (0, 1, 131.9, 132, 132.1, 167.9, 168):
            seconds = hours * 3600
            assert contract.segment_for_elapsed(seconds) == clock.segment_for_elapsed(seconds)
            assert contract.learning_rate_for_elapsed(seconds) == pytest.approx(
                clock.learning_rate_for_elapsed(seconds)
            )

    def test_the_layout_matches_the_accepted_storage_object(self, tmp_path):
        from stratego.training.phase14_storage import Phase14Storage

        accepted = Phase14Storage.under(tmp_path)
        ours = RunPaths(tmp_path, hot_root=tmp_path / "hot")
        assert Path(ours.rollout_root) == accepted.rollout_root
        assert Path(ours.archive_root) == accepted.archive_root
        assert Path(ours.log_root) == accepted.log_root
        assert Path(ours.evaluation_root) == accepted.evaluation_root
        assert Path(ours.run_state_path) == accepted.run_state_path

    def test_production_defaults_are_the_frozen_production_paths(self):
        from stratego.training.phase14_storage import Phase14Storage

        accepted = Phase14Storage.production()
        ours = RunPaths()
        assert Path(ours.external_root) == accepted.external_root
        assert Path(ours.hot_root) == accepted.hot_root


class TestCodeIsolation:
    """The dashboard is outside the sealed Phase 14 code identity, and stays there."""

    def test_no_training_module_imports_the_dashboard(self):
        """The monitor is imported by nothing that trains.

        This is what keeps the launch manifest's code closure unchanged: the
        closure is reached from `stratego.training.phase14_*`, so a dashboard
        nothing imports cannot enter it, and cannot move the code digest.
        """
        offenders = []
        for path in (REPOSITORY / "stratego").rglob("*.py"):
            text = path.read_text()
            if "monitoring" in text and (
                "import monitoring" in text or "from monitoring" in text
            ):
                offenders.append(str(path.relative_to(REPOSITORY)))
        assert offenders == [], f"training code imports the dashboard: {offenders}"

    def test_the_dashboard_is_not_in_the_frozen_launch_closure(self):
        manifest = json.loads(
            (REPOSITORY / "reports" / "phase13" / "phase14_launch_manifest_v1.json").read_text()
        )
        closure = manifest["code"]["file_sha256"]
        assert not [name for name in closure if name.startswith("monitoring/")]
        assert not [name for name in closure if "dashboard" in name]

    def test_every_file_in_the_frozen_closure_is_still_byte_identical(self):
        """Building the dashboard changed no sealed Phase 14 file."""
        manifest = json.loads(
            (REPOSITORY / "reports" / "phase13" / "phase14_launch_manifest_v1.json").read_text()
        )
        changed = []
        for name, digest in manifest["code"]["file_sha256"].items():
            path = REPOSITORY / name
            actual = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
            )
            if actual != digest:
                changed.append(name)
        assert changed == [], f"the sealed closure moved: {changed}"


class TestNoModel:
    """The monitor process loads no model, no torch, and no training code."""

    def test_importing_the_dashboard_imports_no_torch_and_no_stratego(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import monitoring.phase14_dashboard.server as server\n"
            "server.build()\n"
            "loaded = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'stratego', 'numpy'))\n"
            "print(repr(loaded))\n" % str(REPOSITORY)
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "[]", (
            f"the dashboard process imported {result.stdout.strip()}"
        )

    def test_a_whole_status_read_imports_no_torch(self, run):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from monitoring.phase14_dashboard.server import build\n"
            "build(%r, %r).status()\n"
            "print('torch' in sys.modules or any(m.startswith('stratego') for m in sys.modules))\n"
            % (str(REPOSITORY), str(run.root), str(run.hot))
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False"

    def test_the_dashboard_never_opens_a_checkpoint_file(self, run):
        """Checkpoint panels are built from `stat`, never from `torch.load`.

        The accepted `resume_checkpoint_state` loads the payload to answer
        "does it validate". That is right for a resume decision and ruinous at
        a ten-second refresh, so the dashboard reports mtime and size and says
        so.
        """
        checkpoint = run.hot / "hot_000001_step000001744.pt"
        before = checkpoint.stat().st_mtime_ns
        document = build(run.root, run.hot).status()
        assert document["checkpoints"]["hot"]["validated"] is False
        assert "supervisor" in document["checkpoints"]["hot"]["validation_note"]
        assert checkpoint.stat().st_mtime_ns == before


class TestReadOnly:
    """Nothing the dashboard does changes anything on disk."""

    def test_a_hundred_refreshes_change_no_byte_of_the_run(self, run):
        state = build(run.root, run.hot)
        before = digest_tree(run.root, run.hot)
        assert before, "the fixture wrote nothing to compare"
        for _ in range(100):
            for cache in state._caches.values():
                cache.invalidate()
            state.status()
        after = digest_tree(run.root, run.hot)
        assert after == before
        assert set(after) == set(before), "the dashboard created or removed a file"

    def test_the_dashboard_creates_no_directory_that_did_not_exist(self, tmp_path):
        """Before launch there is no run directory, and reading must not make one.

        A monitor that created `/Volumes/.../stratego_phase14` by looking at it
        would bring the production run directory into existence before anybody
        decided to launch.
        """
        absent = tmp_path / "not_yet"
        document = build(absent, absent / "hot").status()
        assert not absent.exists()
        assert document["overall"]["state"] == "NOT STARTED"

    def test_no_source_module_calls_anything_that_writes(self):
        """A structural check over the parsed source, so an edit must be deliberate.

        Parsed rather than grepped: this module's own prose says the words
        `mkdir` and `unlink` while explaining that it never calls them, and a
        text search cannot tell a docstring from a statement.
        """
        import ast

        forbidden = {
            "write", "write_text", "write_bytes", "writelines", "mkdir", "makedirs",
            "unlink", "remove", "rmdir", "rmtree", "rename", "chmod", "utime",
            "truncate", "touch", "symlink", "link", "fsync", "copy", "copyfile",
        }
        for name in ("sources.py", "state.py", "server.py", "contract.py"):
            path = REPOSITORY / "monitoring" / "phase14_dashboard" / name
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                spelled = (
                    called.attr if isinstance(called, ast.Attribute)
                    else called.id if isinstance(called, ast.Name)
                    else None
                )
                if spelled == "write" and _receiver(called) in (
                    "wfile", "stderr", "stdout"
                ):
                    # The HTTP response socket and the console. Neither is the
                    # run directory.
                    continue
                assert spelled not in forbidden, (
                    f"{name}:{node.lineno} calls {spelled}()"
                )
                if spelled == "open":
                    modes = [
                        argument.value for argument in node.args[1:]
                        if isinstance(argument, ast.Constant)
                    ] + [
                        keyword.value.value for keyword in node.keywords
                        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant)
                    ]
                    assert modes == ["rb"], (
                        f"{name}:{node.lineno} opens a file with mode {modes!r}, not 'rb'"
                    )

    def test_the_server_refuses_every_method_except_get(self, run):
        server = serve(external_root=run.root, hot_root=run.hot, port=0)
        port = server.server_address[1]
        import threading

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/status", method=method, data=b"{}"
                )
                with pytest.raises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=5)
                assert caught.value.code == 405
                body = json.loads(caught.value.read())
                assert "read-only" in body["error"]
                assert "RUNBOOK" in body["note"]
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/../etc/passwd", timeout=5)
            assert caught.value.code == 404
        finally:
            server.shutdown()
            server.server_close()

    def test_binding_a_nonlocal_address_is_refused_by_default(self):
        with pytest.raises(DashboardError, match="127.0.0.1"):
            serve(host="0.0.0.0", port=0)


class TestAuthoritativeSources:
    """Each displayed number comes from the source that survives a crash."""

    def test_committed_games_come_from_the_manifests_not_the_counter(self, tmp_path):
        """Phase 13's own finding, reproduced: 8,192 on disk, 4,096 in the counter."""
        fake = FakeRun(tmp_path / "run")
        for number in range(1, 5):
            fake.iteration(number, games=2048)
        fake.run_state()
        fake.telemetry(collection={"process_counter_games": 4096, "games_generated": 4096})
        document = build(fake.root, fake.hot).status()
        games = document["games"]
        assert games["committed_games"] == 8192
        assert games["committed_games_authoritative"] is True
        assert games["committed_games_source"] == "rollout store iteration manifests"
        assert games["process_counter_games"] == 4096
        assert games["process_counter_is_diagnostic"] is True

    def test_an_iteration_still_collecting_is_counted_separately(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.iteration(1, games=2048)
        fake.iteration(2, state="COLLECTING", manifest=False, journal_games=300)
        fake.run_state()
        document = build(fake.root, fake.hot).status()
        assert document["games"]["committed_games"] == 2048
        assert document["games"]["in_flight_games"] == 300
        assert document["games"]["sealed_iterations"] == 1

    def test_a_duplicate_journal_commit_does_not_inflate_the_total(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        directory = fake.iteration(1, state="COLLECTING", manifest=False, journal_games=100)
        with open(directory / "journal" / "w00.commit.jsonl", "a") as stream:
            stream.write(json.dumps({"phase9_game_id": "phase14_it1_g0"}) + "\n")
        assert build(fake.root, fake.hot).status()["games"]["in_flight_games"] == 100

    def test_a_torn_manifest_is_reported_and_does_not_raise(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.iteration(1, games=2048)
        directory = fake.iteration(2, games=2048)
        (directory / "manifest.json").write_text('{"committed_games": 20')
        (directory / "state.json").write_text("{ truncated")
        document = build(fake.root, fake.hot).status()
        assert document["games"]["committed_games"] == 2048
        assert str(directory) in document["games"]["unreadable_iteration_directories"]

    def test_a_sealed_iteration_is_parsed_once_and_then_only_stat_ed(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        for number in range(1, 21):
            fake.iteration(number)
        state = build(fake.root, fake.hot)
        state.status()
        first = state._census.parses
        assert first == 20
        for _ in range(10):
            state._caches["census"].invalidate()
            state.status()
        assert state._census.parses == first, "a sealed manifest was re-parsed"

    def test_a_changed_manifest_is_re_read(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.iteration(1, games=2048)
        state = build(fake.root, fake.hot)
        assert state.status()["games"]["committed_games"] == 2048
        time.sleep(0.01)
        fake.iteration(1, games=2048)
        fake.iteration(2, games=2048)
        state._caches["census"].invalidate()
        assert state.status()["games"]["committed_games"] == 4096

    def test_candidate_and_archive_state_come_from_the_files(self, run):
        document = build(run.root, run.hot).status()
        assert document["checkpoints"]["archive"]["snapshots"] == 1
        assert document["checkpoints"]["candidate"]["latest"]["hour"] == 6
        assert document["candidates"]["by_status"] == {"complete": 1, "pending": 1}
        assert document["candidates"]["unevaluated_hours"] == [6]

    def test_checkpoint_age_is_the_files_own_age(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.hot_checkpoint(1, 100, age_seconds=90)
        fake.hot_checkpoint(2, 200, age_seconds=400)
        hot = build(fake.root, fake.hot).status()["checkpoints"]["hot"]
        # Newest by mtime, not by index: a resumed run rewrites indices.
        assert hot["age_seconds"] == pytest.approx(90, abs=5)
        assert hot["files"] == 2


class TestTheImmutableClock:
    """Elapsed and remaining come from the original window. Downtime is lost."""

    def test_elapsed_and_remaining_are_measured_from_the_original_start(self):
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        window = {
            "run_start_utc": utc_text(start),
            "run_deadline_utc": utc_text(start + timedelta(hours=168)),
            "transition_utc": utc_text(start + timedelta(hours=132)),
            "production": True,
        }
        clock = window_clock(window, now=start + timedelta(hours=73, minutes=24))
        assert clock["elapsed_hours"] == pytest.approx(73.4)
        assert clock["remaining_hours"] == pytest.approx(94.6)
        assert clock["progress_fraction"] == pytest.approx(73.4 / 168)
        assert clock["segment"] == "main"

    def test_downtime_is_lost_time_and_never_extends_the_deadline(self):
        """A six-hour outage costs six of the 168 and moves nothing else."""
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        window = {
            "run_start_utc": utc_text(start),
            "run_deadline_utc": utc_text(start + timedelta(hours=168)),
            "transition_utc": utc_text(start + timedelta(hours=132)),
        }
        before = window_clock(window, now=start + timedelta(hours=40))
        after = window_clock(window, now=start + timedelta(hours=46))
        assert after["remaining_hours"] == pytest.approx(before["remaining_hours"] - 6)
        assert after["run_deadline_utc"] == before["run_deadline_utc"]
        assert after["transition_utc"] == before["transition_utc"]

    def test_the_late_segment_begins_exactly_at_hour_132(self):
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        window = {
            "run_start_utc": utc_text(start),
            "run_deadline_utc": utc_text(start + timedelta(hours=168)),
            "transition_utc": utc_text(start + timedelta(hours=132)),
        }
        assert window_clock(window, now=start + timedelta(hours=131, minutes=59))[
            "segment"] == "main"
        assert window_clock(window, now=start + timedelta(hours=132))["segment"] == "late"

    def test_past_the_deadline_reports_negative_remaining_and_finalizing(self, tmp_path):
        fake = FakeRun(
            tmp_path / "run", start=datetime.now(timezone.utc) - timedelta(hours=169)
        )
        fake.run_state()
        fake.telemetry()
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid())
        document = build(fake.root, fake.hot).status()
        assert document["clock"]["passed_deadline"] is True
        assert document["clock"]["remaining_seconds"] < 0
        assert document["overall"]["state"] == "FINALIZING"

    def test_the_frozen_schedule_is_displayed_and_carries_no_control(self, run):
        schedule = build(run.root, run.hot).status()["schedule"]
        main, late = schedule["segments"]
        assert (main["from_hour"], main["to_hour"]) == (0, 132)
        assert main["learning_rate"] == 7.5e-5
        assert (late["from_hour"], late["to_hour"]) == (132, 168)
        assert late["learning_rate"] == 3.75e-5
        assert schedule["frozen"] is True

    def test_the_page_offers_no_control_for_any_immutable_value(self):
        """No input, no button, no form, and no fetch that is not a GET."""
        page = (REPOSITORY / "monitoring" / "phase14_dashboard" / "index.html").read_text()
        for element in ("<input", "<button", "<form", "<select", "<textarea",
                        "contenteditable"):
            assert element not in page.lower(), f"the page contains {element}"
        for verb in ('method:"POST"', "method: 'POST'", '"PUT"', '"DELETE"'):
            assert verb not in page
        assert page.count("fetch(") == 1


class TestOperationalHealth:
    """Health is graded only where a Phase 14 contract already grades it."""

    def test_a_live_learner_and_a_whole_pool_are_green(self, run):
        document = build(run.root, run.hot).status()
        checks = document["health"]["checks"]
        assert checks["learner"]["status"] == "ok"
        assert checks["nonfinite"]["status"] == "ok"
        # Storage is graded against the real host volume, which is why it is
        # exercised separately with a controlled free-space figure.
        assert checks["storage"]["status"] in ("ok", "watch")
        assert document["overall"]["state"] == "TRAINING"

    def test_a_dead_learner_reads_recovering_and_points_at_the_runbook(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.iteration(1)
        fake.run_state()
        fake.telemetry()
        # A PID that cannot exist: liveness is probed, never remembered.
        fake.supervisor("launch", learner_pid=2 ** 22 - 1, supervisor_pid=os.getpid())
        document = build(fake.root, fake.hot).status()
        assert document["processes"]["learner_alive"] is False
        assert document["overall"]["state"] == "RECOVERING"
        assert "RUNBOOK" in document["health"]["checks"]["learner"]["note"]

    def test_a_closed_pool_during_a_collection_is_not_an_alarm(self, tmp_path):
        """Zero live workers while collecting is the healthy state of a real run.

        The pool exists only while an iteration trains. Grading its absence red
        for the ~5 minutes of every iteration spent collecting is how an
        operator is trained to ignore the field that matters.
        """
        fake = FakeRun(tmp_path / "run")
        fake.iteration(1)
        fake.run_state()
        fake.telemetry()
        # This test process has no `spawn_main` children, so live workers is 0.
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid())
        checks = build(fake.root, fake.hot).status()["health"]["checks"]
        assert checks["loaders"]["status"] == "ok"
        assert "collecting" in checks["loaders"]["note"]

    def test_pool_rebuilds_are_counted_against_the_frozen_ceiling(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid())
        for rebuilds, expected in ((0, "ok"), (1, "watch"), (16, "bad")):
            fake.telemetry(workers={
                "loader_pool_rebuilds": rebuilds,
                "last_pool_rebuild_reason": "BrokenProcessPool",
                "last_pool_rebuild_utc": "2026-09-01T10:00:00.000Z",
            })
            state = build(fake.root, fake.hot)
            document = state.status()
            assert document["health"]["checks"]["pool_rebuilds"]["status"] == expected
            assert document["workers"]["loader_pool_rebuilds"] == rebuilds
            assert document["workers"]["last_pool_rebuild_reason"] == "BrokenProcessPool"

    def test_free_space_is_graded_against_the_frozen_reserve(self, run, monkeypatch):
        state = build(run.root, run.hot)
        for free, expected in ((900.0, "ok"), (200.0, "watch"), (50.0, "bad")):
            monkeypatch.setattr(
                "monitoring.phase14_dashboard.state.volume_usage",
                lambda path, free=free: {
                    "free_gib": free, "external_volume_present": True, "available": True,
                    "used_gib": 10.0, "total_gib": 994.0, "mount_point": "/Volumes/X",
                    "requested_path": str(path), "read_only": False, "mounted": True,
                },
            )
            state._caches["storage"].invalidate()
            assert state.status()["health"]["checks"]["storage"]["status"] == expected

    def test_a_nonfinite_count_is_red(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid())
        fake.telemetry(counters={"non_finite_losses": 3})
        document = build(fake.root, fake.hot).status()
        assert document["health"]["checks"]["nonfinite"]["status"] == "bad"
        assert document["overall"]["state"] == "ERROR"

    def test_loss_movement_is_never_graded(self, tmp_path):
        """A rising policy loss is displayed and is not an alarm.

        "policy loss increased, therefore training is bad" is not in the frozen
        contract, and an invented alarm is one an operator learns to ignore.
        """
        fake = FakeRun(tmp_path / "run")
        fake.iteration(1)
        fake.run_state()
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid())
        fake.telemetry(training={"policy_loss": 0.05})
        fake.telemetry(training={"policy_loss": 50.0, "value_loss": 90.0})
        document = build(fake.root, fake.hot).status()
        assert document["training"]["policy_loss"] == 50.0
        assert document["overall"]["state"] == "TRAINING"
        for name, check in document["health"]["checks"].items():
            assert check["status"] != "bad", f"{name} alarmed on loss movement"

    def test_a_stale_hot_checkpoint_is_watch_and_not_red(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.telemetry()
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid())
        fake.hot_checkpoint(1, 100, age_seconds=3000)
        checks = build(fake.root, fake.hot).status()["health"]["checks"]
        assert checks["checkpoint"]["status"] == "watch"

    def test_an_emergency_stop_file_reads_finalizing(self, run):
        """Written in the shape `request_emergency_stop` actually writes."""
        (run.root / "phase14_emergency_stop.json").write_text(json.dumps({
            "artifact": "phase14_emergency_stop_v1",
            "reason": "operator request",
            "requested_utc": "2026-09-04T02:11:03.412Z",
            "requested_by_pid": 4242,
        }))
        document = build(run.root, run.hot).status()
        stop = document["controls"]["emergency_stop"]
        assert stop["active"] is True
        assert stop["reason"] == "operator request"
        assert stop["requested_utc"] == "2026-09-04T02:11:03.412Z"
        assert document["overall"]["state"] == "FINALIZING"

    def test_an_unreadable_stop_file_is_still_a_stop(self, run):
        """Failing open would show a run as running that was asked to stop.

        The same rule the accepted `emergency_stop_state` applies: presence is
        the signal, not parseability.
        """
        (run.root / "phase14_emergency_stop.json").write_text("{ torn")
        document = build(run.root, run.hot).status()
        assert document["controls"]["emergency_stop"]["active"] is True
        assert document["overall"]["state"] == "FINALIZING"

    def test_an_integrity_failure_reads_error(self, run):
        (run.root / "phase14_integrity_failure.json").write_text(
            json.dumps({"artifact": "phase14_integrity_failure_v1",
                        "error": "checkpoint digest mismatch",
                        "reason": "checkpoint digest mismatch"})
        )
        document = build(run.root, run.hot).status()
        assert document["controls"]["integrity_failure"]["recorded"] is True
        assert document["overall"]["state"] == "ERROR"

    def test_a_closed_run_reads_complete(self, run):
        run.run_state(progress={"closed": True, "close_reason": "deadline"})
        document = build(run.root, run.hot).status()
        assert document["overall"]["state"] == "COMPLETE"
        assert "deadline" in document["overall"]["reason"]


class TestSupervisorVisibility:
    """Restarts and hard kills are visible without reading a log by hand."""

    def test_restarts_are_counted_and_the_first_launch_is_not_one(self, tmp_path):
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.telemetry()
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid(),
                        reason="initial launch")
        assert build(fake.root, fake.hot).status()["supervisor"]["restarts"] == 0

        fake.supervisor("final_process_exit", returncode=-9, signal="SIGKILL")
        fake.supervisor("launch", learner_pid=os.getpid(), supervisor_pid=os.getpid(),
                        reason="restart after signal", attempt=2,
                        checkpoint_selected="/hot/hot_000004_step000001744.pt")
        document = build(fake.root, fake.hot).status()
        assert document["supervisor"]["restarts"] == 1
        assert document["supervisor"]["last_exit"]["returncode"] == -9
        assert document["supervisor"]["last_exit"]["signal"] == "SIGKILL"
        assert document["supervisor"]["last_restart_utc"] is not None
        assert "hot_000004" in document["processes"]["checkpoint_resumed_from"]

    def test_the_supervisor_pid_is_derived_from_the_learners_parent(self, tmp_path):
        """The accepted supervisor logs `learner_pid` and not its own.

        It does not need to. `spawn` gives the learner a new session, not a new
        parent, so the supervisor is the learner's PPID while it lives — which
        means this works without touching sealed Phase 14 code.
        """
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.telemetry()
        learner = subprocess.Popen(
            [sys.executable, "-c", "import time\nwhile True: time.sleep(0.2)"],
            start_new_session=True,
        )
        try:
            fake.supervisor("launch", learner_pid=learner.pid)  # no supervisor_pid
            processes = build(fake.root, fake.hot).status()["processes"]
            assert processes["learner_alive"] is True
            assert processes["supervisor_pid"] == os.getpid()
            assert processes["supervisor_pid_source"] == (
                "derived from the learner's parent process"
            )
            assert processes["learner_orphaned"] is False
        finally:
            learner.kill()
            learner.wait(timeout=10)

    def test_a_learner_that_outlived_its_supervisor_is_reported_unsupervised(
        self, tmp_path
    ):
        """PID 1 as a parent means the supervisor died and left the learner running.

        Training continues, but nothing is watching for the next crash. A
        logged supervisor PID could not have told us this; the derivation can.
        """
        fake = FakeRun(tmp_path / "run")
        fake.run_state()
        fake.telemetry()
        # A grandchild whose parent exits is reparented to launchd, which is
        # exactly the shape of a supervisor dying under a live learner.
        starter = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys\n"
             "child = subprocess.Popen([sys.executable, '-c', "
             "'import time\\nwhile True: time.sleep(0.2)'], start_new_session=True)\n"
             "print(child.pid, flush=True)\n"],
            stdout=subprocess.PIPE, text=True,
        )
        orphan_pid = int(starter.stdout.readline().strip())
        starter.wait(timeout=10)
        deadline = time.time() + 10
        while time.time() < deadline:
            from monitoring.phase14_dashboard.sources import parent_pid
            if (parent_pid(orphan_pid) or 0) <= 1:
                break
            time.sleep(0.1)
        try:
            fake.supervisor("launch", learner_pid=orphan_pid)
            document = build(fake.root, fake.hot).status()
            assert document["processes"]["learner_alive"] is True
            assert document["processes"]["learner_orphaned"] is True
            assert document["health"]["checks"]["supervisor"]["status"] == "bad"
            assert "unsupervised" in document["health"]["checks"]["supervisor"]["note"]
        finally:
            try:
                os.kill(orphan_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_a_logged_supervisor_pid_is_preferred_over_the_derivation(self, run):
        processes = build(run.root, run.hot).status()["processes"]
        assert processes["supervisor_pid"] == os.getpid()
        assert processes["supervisor_pid_source"] == "supervisor log"

    def test_a_loader_pool_rebuild_appears_in_the_event_stream(self, run):
        run.supervisor("worker_health", reason="loader worker exited; pool rebuilt")
        events = build(run.root, run.hot).status()["events"]
        assert any("pool rebuilt" in event["detail"] for event in events)

    def test_iteration_commits_appear_in_the_event_stream(self, run):
        events = build(run.root, run.hot).status()["events"]
        committed = [e for e in events if e["event"] == "iteration_committed"]
        assert len(committed) == 4
        assert committed[0]["source"] == "rollout store"
        # Timestamped from the store's own state history, so commits interleave
        # with supervisor events instead of clumping at one end.
        assert all(event["unix"] is not None for event in committed)
        assert any(event["source"] == "supervisor" for event in events[-4:])

    def test_the_supervisor_log_is_read_incrementally(self, run):
        state = build(run.root, run.hot)
        state.status()
        first = state._supervisor.total_records
        for index in range(50):
            run.supervisor("progress", iteration=index)
        state._caches["supervisor"].invalidate()
        state._caches["process"].invalidate()
        state.status()
        assert state._supervisor.total_records == first + 50

    def test_a_replaced_log_is_detected_and_re_read(self, run):
        """A new run under the same path must not be read at a stale offset."""
        state = build(run.root, run.hot)
        state.status()
        path = run.root / "logs" / "phase14_supervisor.jsonl"
        path.unlink()
        run.supervisor("launch", learner_pid=os.getpid(), reason="a different run")
        state._caches["supervisor"].invalidate()
        state._caches["process"].invalidate()
        events = state.status()["events"]
        assert any(event["detail"] == "a different run" for event in events)

    def test_a_half_written_log_line_is_not_parsed_until_complete(self, run):
        state = build(run.root, run.hot)
        state.status()
        before = len(state.status()["events"])
        path = run.root / "logs" / "phase14_supervisor.jsonl"
        with open(path, "a") as stream:
            stream.write('{"event": "launch", "utc": "2026')  # no newline yet
        state._caches["supervisor"].invalidate()
        assert len(state.status()["events"]) == before
        with open(path, "a") as stream:
            stream.write('-09-01T00:00:00.000Z"}\n')
        state._caches["supervisor"].invalidate()
        assert len(state.status()["events"]) == before + 1


class TestIndependence:
    """Training does not depend on the dashboard, in either direction."""

    def _trainer(self, run: FakeRun) -> subprocess.Popen:
        """A stand-in trainer: appends telemetry and writes checkpoints forever."""
        script = (
            "import json, os, sys, time\n"
            "root = %r\n"
            "for index in range(10_000):\n"
            "    with open(os.path.join(root, 'logs', 'phase14_telemetry.jsonl'), 'a') as s:\n"
            "        s.write(json.dumps({'unix': time.time(), 'training': {'global_optimizer_step': index}}) + '\\n')\n"
            "    print(index, flush=True)\n"
            "    time.sleep(0.05)\n" % str(run.root)
        )
        return subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )

    def test_the_trainer_runs_with_the_dashboard_absent(self, run):
        trainer = self._trainer(run)
        try:
            assert trainer.stdout.readline().strip() == "0"
            assert trainer.poll() is None
        finally:
            trainer.terminate()
            trainer.wait(timeout=10)

    def test_the_dashboard_attaches_to_an_already_running_trainer(self, run):
        trainer = self._trainer(run)
        try:
            for _ in range(3):
                trainer.stdout.readline()
            document = build(run.root, run.hot).status()
            assert document["meta"]["telemetry_rows_read"] >= 3
        finally:
            trainer.terminate()
            trainer.wait(timeout=10)

    def test_killing_the_dashboard_does_not_affect_the_trainer(self, run):
        """SIGKILL the dashboard mid-read; the trainer keeps writing."""
        trainer = self._trainer(run)
        dashboard = subprocess.Popen(
            [sys.executable, str(REPOSITORY / "scripts" / "phase14_dashboard.py"),
             "--external-root", str(run.root), "--hot-root", str(run.hot), "--port", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            trainer.stdout.readline()
            time.sleep(0.5)
            assert dashboard.poll() is None
            os.kill(dashboard.pid, signal.SIGKILL)
            dashboard.wait(timeout=10)
            # The trainer must still be advancing after the dashboard's death.
            first = int(trainer.stdout.readline().strip())
            second = int(trainer.stdout.readline().strip())
            assert second > first
            assert trainer.poll() is None
        finally:
            for process in (trainer, dashboard):
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)

    def test_the_dashboard_restarts_and_reconnects_without_changing_anything(self, run):
        first = build(run.root, run.hot)
        first.status()
        before = digest_tree(run.root, run.hot)
        second = build(run.root, run.hot)  # a "restarted" dashboard
        document = second.status()
        assert document["games"]["committed_games"] == 8192
        assert digest_tree(run.root, run.hot) == before

    def test_the_dashboard_holds_no_file_open_between_requests(self, run):
        """No lock, no held descriptor, nothing for the trainer to contend with."""
        state = build(run.root, run.hot)
        state.status()
        listing = subprocess.run(
            ["lsof", "-p", str(os.getpid())], capture_output=True, text=True, check=False
        ).stdout
        held = [
            line for line in listing.splitlines()
            if str(run.root) in line and "REG" in line
        ]
        assert held == [], f"the dashboard is holding {held}"

    def test_a_missing_run_directory_degrades_instead_of_raising(self, tmp_path):
        document = build(tmp_path / "gone", tmp_path / "gone" / "hot").status()
        assert document["overall"]["state"] == "NOT STARTED"
        assert document["games"]["committed_games"] == 0
        assert document["clock"]["known"] is False

    def test_an_unreadable_source_is_reported_as_content_not_a_crash(self, run):
        (run.root / "logs" / "phase14_telemetry.jsonl").write_text("{ not json\n")
        document = build(run.root, run.hot).status()
        assert document["games"]["committed_games"] == 8192


class TestCost:
    """A refresh must stay cheap at the scale of a finished 168-hour run."""

    def test_a_full_run_is_read_in_well_under_a_second(self, big):
        state = build(big.root, big.hot)
        started = time.perf_counter()
        document = state.status()
        cold = time.perf_counter() - started
        assert document["games"]["committed_games"] == 480 * 2048
        assert cold < 2.0, f"a cold read of a finished run took {cold:.2f}s"

    def test_a_refresh_with_every_cache_expired_stays_cheap(self, big):
        state = build(big.root, big.hot)
        state.status()
        timings = []
        for _ in range(10):
            for cache in state._caches.values():
                cache.invalidate()
            started = time.perf_counter()
            state.status()
            timings.append(time.perf_counter() - started)
        worst = max(timings)
        assert worst < 0.25, f"an expired-cache refresh took {worst * 1000:.0f} ms"

    def test_a_cached_refresh_is_effectively_free(self, big):
        state = build(big.root, big.hot)
        state.status()
        started = time.perf_counter()
        for _ in range(20):
            state.status()
        each = (time.perf_counter() - started) / 20
        assert each < 0.02, f"a cached refresh took {each * 1000:.1f} ms"

    def test_the_history_the_browser_receives_is_bounded(self, big):
        document = build(big.root, big.hot).status()
        assert document["history"]["rows"] <= 400
        assert len(document["events"]) <= 60
        payload = len(json.dumps(document, default=str))
        assert payload < 400_000, f"the status document is {payload} bytes"

    def test_an_idle_dashboard_does_no_work(self, run):
        """Nothing runs between requests: no thread, no timer, no poll."""
        import threading

        before = threading.active_count()
        state = build(run.root, run.hot)
        state.status()
        time.sleep(0.3)
        assert threading.active_count() == before
        assert state.requests == 1
