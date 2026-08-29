"""Agent 4C: the six narrow provenance and resume corrections.

Each test here names the defect it pins rather than the feature it exercises,
because every one of these passed review as "working" before: the run started
with no source identity at all, the resumed log quietly lost a row, the parent
link skipped a generation, four predicates were read on a cadence and then
written nowhere, the P6 baseline was rebuilt from a post-resume sample, and a
completed game's outcome could be dropped with a counter incremented.

Nothing here changes the recipe. The learning rates, schedules, epochs, budget
and population are the D10 values and no test asserts about strength.
"""

from __future__ import annotations

import json
import types

import pytest

from stratego.training.phase17.checkpoint import (
    Phase17CheckpointError,
    read_joint_checkpoint,
)
from stratego.training.phase17.runner import TandemConfig, TandemRunner
from stratego.training.phase17.setup_episode import attach_setup_episodes
from stratego.training.phase17.source_identity import (
    PRODUCTION_SOURCE_ROOTS,
    Phase17SourceError,
    expand_roots,
    production_source_closure,
    require_source_digest,
    source_closure,
    verify_source_digest,
)
from stratego.training.phase17.supervisor import (
    MODE_INTEGRATION,
    CollapseSupervisor,
    Phase17SupervisorError,
    SUPERVISOR_VERSION,
)
from stratego.training.phase17.telemetry import (
    REQUIRED_MOVE_KEYS,
    REQUIRED_SETUP_KEYS,
    REQUIRED_SYSTEM_KEYS,
    TelemetryWriter,
    read_rows,
)

RUN_ID = "RUN-TEST-4C"
SOURCE_A = "a" * 64
SOURCE_B = "b" * 64


def tiny_config(run_id: str = RUN_ID) -> TandemConfig:
    return TandemConfig(
        run_id=run_id,
        total_iterations=20,
        move_budget=400,
        population=8,
        pool_size_per_side=16,
        setup_minibatch_episodes=4,
        move_minibatch_size=64,
    )


def _training_session():
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from run_phase17_training import TrainingSession

    return TrainingSession


def session_at(directory, *, source_digest: str = SOURCE_A, **overrides):
    TrainingSession = _training_session()
    overrides.setdefault("reading_every", 0)
    return TrainingSession(
        tiny_config(),
        directory=directory,
        supervisor_mode=MODE_INTEGRATION,
        source_digest=source_digest,
        **overrides,
    )


def make_row(index: int, *, iteration: "int | None" = None) -> dict:
    """A schema-complete row whose `system.iteration` is addressable."""
    system = {key: index for key in REQUIRED_SYSTEM_KEYS}
    system["iteration"] = index if iteration is None else iteration
    return {
        "move": {key: index for key in REQUIRED_MOVE_KEYS},
        "setup": {key: index for key in REQUIRED_SETUP_KEYS},
        "system": system,
    }


# ---------------------------------------------------------------------------
# Correction 1: the production source-closure digest is mandatory and non-empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   ", None, 0, "not-a-digest", "ab" * 33])
def test_an_empty_or_malformed_production_source_digest_is_refused(value):
    """The empty digest is the dangerous one: it matches every other empty one.

    A run that bound `""` would satisfy every checkpoint, export and resume
    identity check it has, because each of those compares its stored digest to
    the process's, and `"" == ""`.
    """
    with pytest.raises(Phase17SourceError):
        require_source_digest(value)


def test_a_session_cannot_be_constructed_without_a_source_digest(tmp_path):
    TrainingSession = _training_session()
    with pytest.raises(Phase17SourceError, match="no production source digest"):
        TrainingSession(
            tiny_config(),
            directory=tmp_path,
            supervisor_mode=MODE_INTEGRATION,
            source_digest="",
            reading_every=0,
        )
    with pytest.raises(TypeError):
        TrainingSession(
            tiny_config(),
            directory=tmp_path,
            supervisor_mode=MODE_INTEGRATION,
            reading_every=0,
        )


def test_the_production_closure_covers_the_move_half_it_used_to_omit():
    """The pre-correction closure was a hand-written list of eleven paths.

    It omitted `move_trainer.py`, `transition_collector.py`,
    `transition_targets.py`, `move_loss.py`, `move_start.py`, `move_snapshot.py`
    and `setup_sampling.py` -- every one of which changes what the run does --
    and included the smoke script, which cannot. Expanding the package is what
    makes a module added later covered without anybody remembering to add it.
    """
    members = set(expand_roots())
    for name in (
        "stratego/training/phase17/move_trainer.py",
        "stratego/training/phase17/move_loss.py",
        "stratego/training/phase17/move_start.py",
        "stratego/training/phase17/move_snapshot.py",
        "stratego/training/phase17/transition_collector.py",
        "stratego/training/phase17/transition_targets.py",
        "stratego/training/phase17/setup_sampling.py",
        "stratego/training/phase17/setup_learning.py",
        "scripts/run_phase17_training.py",
    ):
        assert name in members, f"{name} is production code outside the closure"
    assert "scripts/run_phase17_d10_smoke.py" not in members, (
        "the smoke is not loaded by production; including it would force a "
        "refreeze for a change that cannot reach the run"
    )
    assert not any("__pycache__" in name for name in members)


def test_editing_any_closure_member_moves_the_digest(tmp_path):
    """A source identity that does not move is a source identity that lies."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("A = 1\n")
    (root / "pkg" / "b.py").write_text("B = 2\n")
    (root / "entry.py").write_text("import pkg\n")
    roots = ("pkg", "entry.py")

    baseline = source_closure(roots, root=root)
    assert baseline["file_count"] == 3
    for member in ("pkg/a.py", "pkg/b.py", "entry.py"):
        original = (root / member).read_text()
        (root / member).write_text(original + "# edited\n")
        assert source_closure(roots, root=root)["closure_digest"] != baseline[
            "closure_digest"
        ], f"editing {member} did not move the closure digest"
        (root / member).write_text(original)
    assert source_closure(roots, root=root)["closure_digest"] == baseline["closure_digest"]

    # A module added later is covered without anybody maintaining a list.
    (root / "pkg" / "c.py").write_text("C = 3\n")
    assert source_closure(roots, root=root)["closure_digest"] != baseline["closure_digest"]


def test_the_working_tree_must_match_the_authorized_digest():
    closure = production_source_closure()
    assert verify_source_digest(closure["closure_digest"])["closure_digest"] == (
        closure["closure_digest"]
    )
    with pytest.raises(Phase17SourceError, match="not the authorized"):
        verify_source_digest("0" * 64)


def test_no_frozen_digest_is_hard_coded_in_the_source_identity_module():
    """Agent 6 supplies the frozen value; this module must not carry one."""
    from pathlib import Path

    text = Path("stratego/training/phase17/source_identity.py").read_text()
    literals = [
        token
        for token in text.replace('"', " ").replace("'", " ").split()
        if len(token) == 64 and all(c in "0123456789abcdef" for c in token)
    ]
    assert not literals, f"a frozen digest is compiled into the module: {literals}"


# ---------------------------------------------------------------------------
# Correction 1: source identity is part of the run identity, end to end
# ---------------------------------------------------------------------------


def test_source_identity_changes_the_run_identity(tmp_path):
    one = session_at(tmp_path / "one", source_digest=SOURCE_A)
    two = session_at(tmp_path / "two", source_digest=SOURCE_B)
    try:
        assert one.config_digest == two.config_digest, "same config, by construction"
        assert one.run_digest != two.run_digest, (
            "the same config under different code is a different run"
        )
    finally:
        one.close()
        two.close()


def test_source_identity_survives_checkpoint_export_and_resume(tmp_path):
    directory = tmp_path / "run"
    session = session_at(directory, source_digest=SOURCE_A)
    hour_zero = session.export_hour_zero()
    step = session.step()
    session.close()

    checkpoint_path = step["checkpoint"]["path"]

    # The checkpoint carries it, and reading under a different source refuses.
    payload = read_joint_checkpoint(
        checkpoint_path, run_id=RUN_ID, source_digest=SOURCE_A
    )
    assert payload["source_digest"] == SOURCE_A
    with pytest.raises(Phase17CheckpointError, match="was written under source"):
        read_joint_checkpoint(checkpoint_path, run_id=RUN_ID, source_digest=SOURCE_B)

    # The paired export carries it too, which is what lets Agent 5 reject a
    # candidate that is not bound to the corrected source.
    import torch

    bundle = torch.load(
        directory / "exports" / f"{hour_zero['candidate_id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert bundle["manifest"]["source_digest"] == SOURCE_A

    # And a resume under the wrong source is refused before anything is loaded.
    wrong = session_at(directory, source_digest=SOURCE_B)
    try:
        with pytest.raises(Phase17CheckpointError, match="was written under source"):
            wrong.resume(checkpoint_path)
    finally:
        wrong.close()


# ---------------------------------------------------------------------------
# Correction 2: the checkpointed iteration keeps its telemetry row
# ---------------------------------------------------------------------------


def test_the_pending_row_is_adopted_once_and_later_rows_are_truncated(tmp_path):
    """The unit-level statement of the defect and of the rule that replaces it.

    Iteration 2's row is written after iteration 2's checkpoint, so it sits
    past the checkpointed offset. Iteration 3's row sits past it too. Exactly
    one of them belongs to the checkpoint being resumed.
    """
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id=RUN_ID)
    writer.append(make_row(0, iteration=1))
    position = writer.position(pending_row_iteration=2)  # checkpoint of it. 2
    writer.append(make_row(1, iteration=2))  # the checkpointed row
    writer.append(make_row(2, iteration=3))  # genuinely uncheckpointed
    writer.close()

    resumed = TelemetryWriter.resume(position, run_id=RUN_ID)
    rows = read_rows(path)
    assert [row["system"]["iteration"] for row in rows] == [1, 2], (
        "the checkpointed row must survive and the later one must not"
    )
    assert resumed.records == 2
    assert resumed.offset == path.stat().st_size

    resumed.append(make_row(2, iteration=3))
    resumed.close()
    rows = read_rows(path)
    assert [row["record_index"] for row in rows] == [0, 1, 2]
    assert [row["system"]["iteration"] for row in rows] == [1, 2, 3]


def test_a_pending_row_that_never_landed_is_not_invented(tmp_path):
    """The crash-before-append case: the row is re-produced, not duplicated."""
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id=RUN_ID)
    writer.append(make_row(0, iteration=1))
    position = writer.position(pending_row_iteration=2)
    writer.close()

    resumed = TelemetryWriter.resume(position, run_id=RUN_ID)
    assert resumed.records == 1
    resumed.append(make_row(1, iteration=2))
    resumed.close()
    rows = read_rows(path)
    assert [row["system"]["iteration"] for row in rows] == [1, 2]


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda line: line[: len(line) // 2], id="partial_line"),
        pytest.param(lambda line: b"{not json}\n", id="unparseable"),
    ],
)
def test_a_tail_that_is_not_a_complete_pending_row_is_discarded(tmp_path, corrupt):
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id=RUN_ID)
    writer.append(make_row(0, iteration=1))
    position = writer.position(pending_row_iteration=2)
    writer.close()
    with path.open("ab") as handle:
        handle.write(corrupt(json.dumps(make_row(1, iteration=2)).encode() + b"\n"))

    resumed = TelemetryWriter.resume(position, run_id=RUN_ID)
    assert resumed.records == 1
    assert path.stat().st_size == position["offset"]
    resumed.close()


def test_a_row_from_a_different_iteration_is_not_adopted(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id=RUN_ID)
    writer.append(make_row(0, iteration=1))
    position = writer.position(pending_row_iteration=2)
    writer.append(make_row(1, iteration=7))
    writer.close()

    resumed = TelemetryWriter.resume(position, run_id=RUN_ID)
    assert resumed.records == 1, "only the named pending iteration may be adopted"
    resumed.close()


def test_a_live_resume_keeps_every_checkpointed_iteration_row_exactly_once(tmp_path):
    """The live statement: three real iterations, a real resume from the second.

    Before this correction the resumed log held iterations 1, 2, 4 at record
    indices 0, 1, 2 -- iteration 3's row deleted although its weights were
    checkpointed and restored, and `record_index` no longer tracking anything.
    """
    directory = tmp_path / "run"
    session = session_at(directory)
    session.export_hour_zero()
    steps = [session.step() for _ in range(3)]
    session.close()

    log = session.telemetry.path
    assert [row["system"]["iteration"] for row in read_rows(log)] == [1, 2, 3]

    resumed = session_at(directory)
    resumed.resume(steps[1]["checkpoint"]["path"])
    assert [row["system"]["iteration"] for row in read_rows(log)] == [1, 2], (
        "iteration 2 was checkpointed and must survive; iteration 3 was not "
        "part of this checkpoint and must not"
    )
    # `checkpoint=False`: this resume deliberately goes back to an OLDER
    # checkpoint while generation 3 is still on disk, and a checkpoint here
    # would collide with it. That refusal is the accepted behavior -- an
    # accepted checkpoint is never overwritten -- and it is not what this test
    # is about.
    resumed.step(checkpoint=False)
    resumed.close()

    rows = read_rows(log)
    assert [row["system"]["iteration"] for row in rows] == [1, 2, 3]
    assert [row["record_index"] for row in rows] == [0, 1, 2]
    iterations = [row["system"]["iteration"] for row in rows]
    assert len(iterations) == len(set(iterations)), "no iteration is recorded twice"


# ---------------------------------------------------------------------------
# Correction 3: the next checkpoint's parent is the checkpoint that was loaded
# ---------------------------------------------------------------------------


def test_the_checkpoint_after_a_resume_links_to_the_one_it_loaded(tmp_path):
    directory = tmp_path / "run"
    session = session_at(directory)
    session.export_hour_zero()
    steps = [session.step() for _ in range(2)]
    session.close()
    loaded = steps[1]["checkpoint"]
    assert loaded["generation"] == 2

    resumed = session_at(directory)
    resumed.resume(loaded["path"])
    assert resumed.parent_checkpoint == loaded, (
        "the resume itself must adopt the loaded checkpoint as the parent"
    )
    after = resumed.step()["checkpoint"]
    resumed.close()

    assert after["generation"] == 3
    payload = read_joint_checkpoint(
        after["path"], run_id=RUN_ID, source_digest=SOURCE_A
    )
    parent = payload["parent_checkpoint_identity"]
    assert parent["generation"] == 2, (
        "generation 3's parent used to be generation 1: the resume copied the "
        "LOADED checkpoint's parent instead of the loaded checkpoint"
    )
    assert parent["path"] == loaded["path"]
    assert parent["payload_digest"] == loaded["payload_digest"]
    assert parent["file_sha256"] == loaded["file_sha256"]
    assert parent["iteration"] == loaded["iteration"]


def test_an_export_after_a_resume_binds_the_loaded_checkpoint(tmp_path):
    """Same link, on the artifact Agent 5 actually evaluates."""
    directory = tmp_path / "run"
    session = session_at(directory)
    session.export_hour_zero()
    step = session.step()
    session.close()

    resumed = session_at(directory)
    resumed.resume(step["checkpoint"]["path"])
    candidate = resumed._export(9)
    resumed.close()

    import torch

    bundle = torch.load(
        directory / "exports" / f"{candidate['candidate_id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert bundle["manifest"]["parent_checkpoint_identity"]["generation"] == 1
    assert bundle["manifest"]["parent_checkpoint_identity"]["path"] == (
        step["checkpoint"]["path"]
    )


# ---------------------------------------------------------------------------
# Correction 4: cadence warnings reach the row; the P6 baseline survives resume
# ---------------------------------------------------------------------------


def test_a_cadence_warning_appears_in_the_row_of_the_iteration_that_tripped_it(tmp_path):
    """P5 is read on the reading cadence, after `run_iteration` has returned.

    Its verdict therefore was not in `result.verdicts` and reached no row at
    all: the predicate could trip on every iteration of the 12-hour run and the
    telemetry would never mention it.
    """
    session = session_at(tmp_path, reading_every=1, reading_samples=8)
    session.export_hour_zero()
    # Raise P5's floor above any achievable support so the reading trips it.
    # A threshold, not a recipe value: the supervisor may not write either, and
    # this only decides whether the warning is generated, not what happens next.
    session.runner.supervisor.flag_support_floor = 1e9
    row = session.step(checkpoint=False)["row"]
    session.close()

    warnings = {verdict["code"] for verdict in row["system"]["warnings"]}
    assert "P5" in warnings, "a tripped cadence predicate must reach its own row"
    p5 = [v for v in row["system"]["warnings"] if v["code"] == "P5"][0]
    assert p5["severity"] == "warning"
    assert p5["stops_the_run"] is False
    assert p5["evidence"]["flag_effective_support"] < p5["evidence"]["floor"]
    # D10 section 7: a statistical predicate can never appear as a stop, even
    # though it "fires" on one observation.
    assert row["system"]["stop_predicates"] == []


def test_every_tripping_cadence_predicate_is_carried_into_the_row(tmp_path):
    """P4, P5 and P6 forced to trip together; all three must reach one row.

    Each is generated at a different point of `_cadence_guards` -- P6 before
    the reading, P4 and P5 inside it -- so carrying one out is not evidence
    that the others are carried. The thresholds are moved, never the recipe:
    what changes is whether a warning is generated, not what happens next.
    """
    session = session_at(tmp_path, reading_every=1, reading_samples=8)
    session.export_hour_zero()
    supervisor = session.runner.supervisor
    supervisor.setup_entropy_floor = 1e9  # P4
    supervisor.flag_support_floor = 1e9  # P5
    supervisor.move_entropy_first_hour_closed = True  # P6, against a fixed median
    supervisor.move_entropy_first_hour_median = 1e9
    row = session.step(checkpoint=False)["row"]
    session.close()

    carried = {verdict["code"] for verdict in row["system"]["warnings"]}
    assert {"P4", "P5", "P6"} <= carried, (
        f"a cadence predicate tripped but never reached its row: {carried}"
    )
    assert row["system"]["stop_predicates"] == [], "D10 section 7: these never stop"
    # P7 cannot be forced here -- D10 drains the buffer every iteration, so
    # "episodes available and no update" does not arise -- but it must still be
    # observed on every iteration, which is what makes its silence readable.
    assert "P7" in {verdict["code"] for verdict in supervisor.verdicts}


def test_the_row_carries_verdicts_the_iteration_itself_did_not_produce(tmp_path):
    """The merge, stated directly: cadence verdicts are additional, not a copy."""
    session = session_at(tmp_path, reading_every=1, reading_samples=8)
    session.export_hour_zero()
    session.runner.supervisor.flag_support_floor = 1e9
    step = session.step(checkpoint=False)
    session.close()
    from_iteration = {verdict["code"] for verdict in step["result"].verdicts}
    in_row = {verdict["code"] for verdict in step["row"]["system"]["warnings"]}
    assert "P5" not in from_iteration, (
        "P5 is read on the reading cadence, after run_iteration returned"
    )
    assert "P5" in in_row


def test_the_first_hour_entropy_baseline_survives_a_resume(tmp_path):
    """P6's floor is 25% of the first hour's median, fixed once.

    The samples used to live on the session object, which a resume rebuilds
    from nothing: a run resumed inside its first hour would fix the median from
    the post-resume readings alone, and a run resumed after it would start
    collecting a second, different one.
    """
    directory = tmp_path / "run"
    session = session_at(directory)
    session.export_hour_zero()
    step = session.step()
    before = list(session.first_hour_move_entropies)
    session.close()
    assert before and before[-1] > 0.0
    assert not session.first_hour_median_set, "well inside the first hour"

    resumed = session_at(directory)
    resumed.resume(step["checkpoint"]["path"])
    assert resumed.first_hour_move_entropies == before, (
        "the first-hour sample set must come back from the checkpoint"
    )
    assert not resumed.first_hour_median_set
    resumed.close()


def test_a_fixed_first_hour_median_is_never_recomputed(tmp_path):
    """Once the hour closes the baseline is frozen, across a resume included."""
    supervisor = CollapseSupervisor(
        run_id=RUN_ID, mode=MODE_INTEGRATION, setup_entropy_baseline=1.5
    )
    for value in (1.0, 2.0, 3.0):
        assert supervisor.note_first_hour_move_entropy(
            value, first_hour_complete=False
        ) is None
    assert supervisor.note_first_hour_move_entropy(4.0, first_hour_complete=True) == 2.5
    assert supervisor.move_entropy_first_hour_closed
    # Later readings do not move it, and do not join the sample set.
    assert supervisor.note_first_hour_move_entropy(99.0, first_hour_complete=True) is None
    assert supervisor.move_entropy_first_hour_median == 2.5
    assert supervisor.move_entropy_first_hour_samples == [1.0, 2.0, 3.0, 4.0]

    restored = CollapseSupervisor(
        run_id=RUN_ID, mode=MODE_INTEGRATION, setup_entropy_baseline=1.5
    )
    restored.load_state_document(supervisor.state_document())
    assert restored.move_entropy_first_hour_median == 2.5
    assert restored.move_entropy_first_hour_closed
    assert restored.move_entropy_first_hour_samples == [1.0, 2.0, 3.0, 4.0]
    assert restored.note_first_hour_move_entropy(0.001, first_hour_complete=True) is None
    assert restored.move_entropy_first_hour_median == 2.5


def test_a_supervisor_state_without_the_first_hour_samples_is_refused():
    """A v2 state document cannot answer the question, so it is not resumed."""
    supervisor = CollapseSupervisor(
        run_id=RUN_ID, mode=MODE_INTEGRATION, setup_entropy_baseline=1.5
    )
    document = supervisor.state_document()
    assert document["supervisor_version"] == SUPERVISOR_VERSION
    del document["move_entropy_first_hour_samples"]
    with pytest.raises(Phase17SupervisorError, match="first_hour_samples"):
        supervisor.load_state_document(document)


# ---------------------------------------------------------------------------
# Correction 5: a rejected completed setup episode arms the integrity stop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    return TandemRunner(tiny_config("RUN-TEST-4C-ENQ"), supervisor_mode=MODE_INTEGRATION)


def _completed_pair(red_samples, blue_samples, index: int, run_id: str):
    pair = attach_setup_episodes(
        red_samples[index],
        blue_samples[index],
        run_id=run_id,
        game_id=f"4c-game-{index}",
    )
    return pair.complete("red_win")


def test_a_duplicate_completed_episode_arms_the_integrity_stop(
    runner, red_samples, blue_samples
):
    episodes = _completed_pair(red_samples, blue_samples, 0, runner.config.run_id)
    supervisor = runner.supervisor
    assert not supervisor.should_stop

    runner._enqueue(episodes[0])
    assert not supervisor.should_stop, "the first arrival is ordinary work"
    assert supervisor.predicates["I7"].trips == 0

    runner._enqueue(episodes[0])  # the same outcome offered twice
    assert supervisor.should_stop, (
        "a lost or duplicated setup outcome is a D10 section 7 integrity stop, "
        "not a counted rejection the run walks past"
    )
    record = supervisor.stop_record()
    assert record["code"] == "I7"
    assert record["evidence"]["rejected_episode"]["reason"] == (
        "duplicate (run_id, game_id, color)"
    )
    # Recorded for diagnosis, as the instruction allows -- and stopping too.
    assert runner.enqueue_rejections[-1]["reason"] == (
        "duplicate (run_id, game_id, color)"
    )
    assert runner.enqueue_rejections[-1]["identity"] == episodes[0].identity()


def test_an_incomplete_episode_also_arms_it(runner, red_samples, blue_samples):
    pair = attach_setup_episodes(
        red_samples[1], blue_samples[1], run_id=runner.config.run_id, game_id="4c-open"
    )
    runner.supervisor.predicates["I7"].consecutive = 0
    runner.supervisor.stopped = None
    runner._enqueue(pair.red)  # never completed: no terminal result
    assert runner.supervisor.should_stop
    assert runner.supervisor.stop_record()["evidence"]["rejected_episode"][
        "reason"
    ] == "episode has no terminal result"


def test_the_rejection_verdict_reaches_the_iterations_telemetry(
    runner, red_samples, blue_samples
):
    """Armed mid-window, surfaced at the boundary.

    `_enqueue` runs inside the collector, so its verdict is not in the list
    `run_iteration` builds. Without the drain the run would stop with no row
    saying which iteration armed it.
    """
    episodes = _completed_pair(red_samples, blue_samples, 2, runner.config.run_id)
    runner._enqueue(episodes[0])
    runner._enqueue(episodes[0])
    assert runner._mid_window_verdicts, "the verdict is held for the boundary"

    result = types.SimpleNamespace(
        window=types.SimpleNamespace(transitions_harvested=runner.config.move_budget),
        move_update=types.SimpleNamespace(
            means={
                "mean_loss_total": 0.5,
                "mean_policy_entropy": 1.0,
                "mean_behavior_kl": 0.01,
            },
            learning_rate=1e-4,
        ),
        setup_update=None,
    )
    verdicts = runner._supervise(result)
    codes = [verdict["code"] for verdict in verdicts]
    assert "I7" in codes
    surfaced = [verdict for verdict in verdicts if verdict["code"] == "I7"][0]
    assert surfaced["stops_the_run"] is True
    assert not runner._mid_window_verdicts, "drained exactly once"
    assert "I7" not in [v["code"] for v in runner._supervise(result)]
