"""Real Phase 6 candidates inside the Phase 3 pipeline, under `model_contract_v2`.

`scripts/run_phase6_agent04.py` runs these same properties at acceptance scale
over four candidates for half an hour. These are the small, fast versions, so a
regression in the frame conversion, the recording path or the storage accounting
surfaces in an ordinary test run instead of only after a long harness.

Metal is used when available and the processor stands in when it is not: what is
under test here is the *pipeline*, not the device. The acceptance harness refuses
to substitute the processor; that distinction is deliberate.
"""

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stratego.engine.constants import ACTION_SPACE_SIZE, PLAYERS  # noqa: E402
from stratego.model.action_frame import (  # noqa: E402
    absolute_action_to_model,
    absolute_legal_mask_to_model,
    model_action_to_absolute,
)
from stratego.model.architecture_configs import (  # noqa: E402
    ARCHITECTURE_FAMILY,
    FAMILY_INITIALIZATION_SEED,
    candidate_config,
)
from stratego.model.contract import MODEL_CONTRACT_VERSION  # noqa: E402
from stratego.training import phase6_pipeline_benchmark as p6  # noqa: E402
from stratego.training.coordinator import (  # noqa: E402
    ACTION_FRAME_ABSOLUTE,
    ACTION_FRAME_NORMALIZED,
    ActionFrameMismatchError,
    CoordinatorConfig,
    CoordinatorError,
    NormalizedActionFrame,
    SelfPlayCoordinator,
)
from stratego.training.shared_buffers import (  # noqa: E402
    NO_ACTING_PLAYER,
    STATUS_ACTIVE,
)
from stratego.training.trajectory import (  # noqa: E402
    DecisionRecord,
    GameRecord,
    TRAJECTORY_FORMAT_VERSION,
    TRAJECTORY_VERSION,
    decode_game_record,
    encode_game_record,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PRECISION = "float32" if DEVICE == "cpu" else "float16"

#: The smallest candidate. Every property under test is a property of the
#: pipeline and the frame, not of a particular width, and C0 keeps the suite fast.
CANDIDATE = "C0"


def small_config(**overrides) -> CoordinatorConfig:
    settings = {
        "workers": 2,
        "environments": 16,
        "inference_batch_size": 8,
        "precision": PRECISION,
        "detailed_timing": False,
        "root_seed": 77_004,
    }
    settings.update(overrides)
    return p6.candidate_configuration(CANDIDATE, **settings)


@pytest.fixture(scope="module")
def frame() -> NormalizedActionFrame:
    return NormalizedActionFrame(torch.device("cpu"))


# ---------------------------------------------------------------------------
# The batch frame conversion agrees with Agent 1, everywhere
# ---------------------------------------------------------------------------


def test_the_batch_tables_are_agent_ones_tables(frame):
    """The coordinator must not hold a second, independent frame convention."""
    for player in PLAYERS:
        expected_forward = np.array(
            [absolute_action_to_model(action, player) for action in range(ACTION_SPACE_SIZE)],
            dtype=np.int64,
        )
        expected_inverse = np.array(
            [model_action_to_absolute(action, player) for action in range(ACTION_SPACE_SIZE)],
            dtype=np.int64,
        )
        assert np.array_equal(frame.to_model_host[player], expected_forward)
        assert np.array_equal(frame.to_absolute_host[player], expected_inverse)


def test_every_table_is_a_permutation_and_inverts(frame):
    identity = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    for player in PLAYERS:
        forward = frame.to_model_host[player]
        inverse = frame.to_absolute_host[player]
        assert np.array_equal(np.sort(forward), identity)
        assert np.array_equal(np.sort(inverse), identity)
        assert np.array_equal(inverse[forward], identity)
        assert np.array_equal(forward[inverse], identity)


def test_red_is_the_identity_and_blue_is_not(frame):
    """If both were the identity the conversion would be untested by everything."""
    assert frame.is_identity[PLAYERS[0]] is True
    assert frame.is_identity[PLAYERS[1]] is False


def test_masks_to_model_matches_the_single_row_converter(frame):
    """The whole-batch permutation equals Agent 1's mask converter, row by row."""
    generator = np.random.default_rng(4)
    acting = np.array([0, 1, 1, 0, 1], dtype=np.int8)
    masks = generator.integers(0, 2, size=(5, ACTION_SPACE_SIZE), dtype=np.uint8)
    device_masks = torch.from_numpy(masks.astype(bool))

    converted = frame.masks_to_model(device_masks.clone(), frame.split_rows(acting))

    for row in range(masks.shape[0]):
        expected = absolute_legal_mask_to_model(masks[row], int(acting[row]))
        assert np.array_equal(converted[row].numpy(), expected.astype(bool))


def test_action_ids_to_model_matches_the_single_action_converter(frame):
    acting = np.array([1, 0], dtype=np.int8)
    ids = torch.tensor([[10, 4049, 99], [10, 4049, 99]], dtype=torch.long)

    converted = frame.action_ids_to_model(ids, frame.split_rows(acting))

    assert converted[0].tolist() == [
        absolute_action_to_model(action, 1) for action in (10, 4049, 99)
    ]
    assert converted[1].tolist() == [10, 4049, 99]


def test_actions_to_absolute_inverts_the_selection(frame):
    acting = np.array([1, 0, 1], dtype=np.int8)
    model_actions = torch.tensor([9989, 4049, 5950], dtype=torch.long)

    absolute = frame.actions_to_absolute(model_actions, frame.split_rows(acting))

    assert absolute.tolist() == [
        model_action_to_absolute(9989, 1),
        4049,
        model_action_to_absolute(5950, 1),
    ]


def test_a_row_without_an_acting_player_is_refused(frame):
    """A normalized action is meaningless without knowing whose frame it is in."""
    acting = np.array([0, NO_ACTING_PLAYER, 1], dtype=np.int8)
    with pytest.raises(ActionFrameMismatchError, match="acting player"):
        frame.split_rows(acting)


def test_a_frame_failure_is_classified_as_a_frame_failure():
    assert p6.classify_failure(ActionFrameMismatchError("x")) == p6.FAILURE_FRAME
    assert (
        p6.classify_failure(CoordinatorError("the published legality mask forbids: y"))
        == p6.FAILURE_ILLEGAL
    )


# ---------------------------------------------------------------------------
# The conversion in the live coordinator path
# ---------------------------------------------------------------------------


def test_the_normalized_path_only_ever_applies_legal_absolute_actions():
    """The engine's published mask is the authority, whatever frame the model used."""
    config = small_config()
    coordinator = p6.open_candidate_coordinator(CANDIDATE, config, device=DEVICE)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        for _ in range(10):
            masks = buffers.legal_mask.copy()
            statuses = buffers.status.copy()
            coordinator.step()
            for slot in range(config.num_environments):
                if statuses[slot] != STATUS_ACTIVE:
                    continue
                action = int(coordinator.last_actions[slot])
                assert 0 <= action < ACTION_SPACE_SIZE
                assert masks[slot, action] == 1, (
                    f"slot {slot}: absolute action {action} was not in the published mask"
                )
    finally:
        coordinator.shutdown()


def test_the_selected_normalized_action_inverts_to_the_applied_action():
    """The round trip the whole contract rests on, taken from the live pipeline."""
    config = small_config()
    coordinator = p6.open_candidate_coordinator(CANDIDATE, config, device=DEVICE)
    coordinator.start()
    differing = 0
    checked = 0
    try:
        buffers = coordinator.pool.buffers
        for _ in range(10):
            acting = buffers.acting_player.copy()
            statuses = buffers.status.copy()
            coordinator.step()
            for slot in range(config.num_environments):
                if statuses[slot] != STATUS_ACTIVE:
                    continue
                player = int(acting[slot])
                model_action = int(coordinator.last_model_actions[slot])
                absolute = int(coordinator.last_actions[slot])
                assert model_action >= 0
                assert model_action_to_absolute(model_action, player) == absolute
                assert absolute_action_to_model(absolute, player) == model_action
                checked += 1
                if model_action != absolute:
                    differing += 1
    finally:
        coordinator.shutdown()
    assert checked > 0
    # Without this the test would pass just as happily against a coordinator that
    # performed no conversion at all: for red the two identifiers coincide, so
    # only a blue row proves the permutation is really being applied.
    assert differing > 0, "no row ever needed a conversion; the frame path is untested"


def test_the_absolute_frame_path_is_left_exactly_as_phase_three_had_it():
    config = CoordinatorConfig(
        num_environments=16,
        num_workers=2,
        inference_batch_size=8,
        precision=PRECISION,
        detailed_timing=False,
        root_seed=77_010,
    )
    assert config.action_frame == ACTION_FRAME_ABSOLUTE
    coordinator = SelfPlayCoordinator(config, device=DEVICE)
    coordinator.start()
    try:
        assert coordinator.frame is None
        metrics = coordinator.step()
        assert metrics.frame_seconds == 0.0
        # No normalized identifier exists in the absolute frame, and the
        # coordinator says so rather than inventing one.
        assert set(coordinator.last_model_actions.tolist()) == {-1}
    finally:
        coordinator.shutdown()


def test_an_unknown_action_frame_is_refused():
    with pytest.raises(ValueError, match="action frame"):
        CoordinatorConfig(16, 2, 8, action_frame="rotated_squares")


# ---------------------------------------------------------------------------
# Illegal action rejection stays loud
# ---------------------------------------------------------------------------


def test_an_illegal_selection_raises_instead_of_being_repaired(monkeypatch):
    """Never substitute a legal move after a bad selection -- fail, loudly."""
    import stratego.training.coordinator as coordinator_module

    def always_pick_zero(policy_logits, legal_mask, *, generator=None):
        return torch.zeros(policy_logits.shape[0], dtype=torch.long, device=policy_logits.device)

    config = small_config()
    coordinator = p6.open_candidate_coordinator(CANDIDATE, config, device=DEVICE)
    coordinator.start()
    try:
        # Action 0 is `a1 -> a1`, a move to its own square, so it is never legal.
        monkeypatch.setattr(coordinator_module, "sample_dense", always_pick_zero)
        with pytest.raises(CoordinatorError) as raised:
            coordinator.step()
        assert "legality mask forbids" in str(raised.value)
        assert p6.classify_failure(raised.value) == p6.FAILURE_ILLEGAL
    finally:
        coordinator.shutdown()


def test_the_forward_and_inverse_tables_coincide_for_an_involution(frame):
    """Why the *skipped*-inverse control below is the meaningful one.

    Blue's transform is `square -> 99 - square` applied to source and destination
    alike, which makes the whole action permutation `a -> 9999 - a`: its own
    inverse. So a coordinator that used the forward table where it meant the
    inverse would still be correct here, and a negative control built on swapping
    them would prove nothing. Pinned explicitly, because the day the geometry
    stops being an involution this assertion is the one that should fail and send
    somebody back to that control.
    """
    for player in PLAYERS:
        assert np.array_equal(frame.to_model_host[player], frame.to_absolute_host[player])


def test_forgetting_the_inverse_conversion_would_be_caught(monkeypatch):
    """A negative control for the frame itself.

    The failure `model_contract_v2` actually risks is a normalized identifier
    reaching the engine unconverted: the model is shown normalized legality and
    picks a normalized action, and if nothing inverts it, blue's move arrives
    180 degrees wrong. The coordinator's own check against the published absolute
    mask must refuse it rather than letting a worker apply it.
    """
    config = small_config()
    coordinator = p6.open_candidate_coordinator(CANDIDATE, config, device=DEVICE)
    coordinator.start()
    try:
        assert coordinator.frame is not None
        monkeypatch.setattr(
            coordinator.frame,
            "actions_to_absolute",
            lambda model_actions, assignment: model_actions,
        )
        with pytest.raises(CoordinatorError, match="legality mask forbids"):
            for _ in range(12):
                coordinator.step()
    finally:
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# One MPS owner
# ---------------------------------------------------------------------------


def test_the_benchmark_module_does_not_move_metal_into_a_worker():
    """Importing the worker layer must not pull the model or Metal in with it."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stratego.training.worker_pool; "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_only_the_coordinator_holds_the_model_and_the_frame():
    config = small_config()
    coordinator = p6.open_candidate_coordinator(CANDIDATE, config, device=DEVICE)
    coordinator.start()
    try:
        assert coordinator.model is not None
        assert coordinator.frame is not None
        pool = coordinator.pool
        assert not hasattr(pool, "model")
        assert not hasattr(pool, "frame")
        # The pool's recording configuration carries version strings and counters
        # and nothing that could reach a device.
        assert not any(
            isinstance(value, torch.Tensor) for value in vars(pool.recording).values()
        )
    finally:
        coordinator.shutdown()


def test_a_candidate_and_a_representative_config_cannot_both_be_given():
    from stratego.training.representative_model import RepresentativeConfig

    with pytest.raises(CoordinatorError, match="not both"):
        SelfPlayCoordinator(
            small_config(),
            device=DEVICE,
            model=p6.build_pipeline_candidate(CANDIDATE),
            model_config=RepresentativeConfig(),
        )


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------


def test_the_candidate_is_agent_twos_candidate_unmodified():
    model = p6.build_pipeline_candidate("C1")
    configuration = candidate_config("C1")
    assert model.config == configuration
    assert model.config.digest() == configuration.digest()
    assert model.architecture_id == ARCHITECTURE_FAMILY
    assert model.initialisation_seed == FAMILY_INITIALIZATION_SEED
    assert model.parameter_count() == 863_959  # Agent 3's recorded figure


def test_two_builds_of_a_candidate_are_bit_identical():
    left = p6.build_pipeline_candidate(CANDIDATE).state_dict()
    right = p6.build_pipeline_candidate(CANDIDATE).state_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_identity_carries_the_contract_version():
    identity = p6.candidate_identity(p6.build_pipeline_candidate(CANDIDATE))
    assert identity["model_contract_version"] == MODEL_CONTRACT_VERSION
    assert identity["candidate_id"] == CANDIDATE


def test_a_candidate_coordinator_refuses_the_absolute_frame():
    config = CoordinatorConfig(16, 2, 8, action_frame=ACTION_FRAME_ABSOLUTE)
    with pytest.raises(CoordinatorError, match="normalized"):
        p6.open_candidate_coordinator(CANDIDATE, config, device=DEVICE)


# ---------------------------------------------------------------------------
# The trajectory schema is unchanged
# ---------------------------------------------------------------------------


def test_the_trajectory_schema_did_not_move():
    """Agent 4 supplies a real policy into `trajectory_v1`; it does not change it."""
    assert TRAJECTORY_VERSION == "trajectory_v1"
    assert TRAJECTORY_FORMAT_VERSION == 1
    assert set(DecisionRecord.__dataclass_fields__) == {
        "game_id",
        "ply",
        "acting_player",
        "selected_action_id",
        "legal_action_ids",
        "old_probabilities",
        "win_draw_loss_prediction",
        "collection_policy_version",
        "snapshot_reference",
    }
    assert set(GameRecord.__dataclass_fields__) == {
        "game_id",
        "environment_id",
        "generation",
        "trajectory_version",
        "rules_version",
        "observation_version",
        "implementation_version",
        "red_setup",
        "blue_setup",
        "first_player",
        "setup_family",
        "setup_id",
        "board_geometry_version",
        "battleless_move_limit",
        "absolute_move_limit",
        "rules_context",
        "terminal_result",
        "terminal_reason",
        "final_ply",
        "collection_policy_version",
        "collection_checkpoint_id",
        "root_seed",
        "slot_seed",
        "snapshot_interval",
        "actions",
        "snapshots",
        "decisions",
    }
    # No belief field anywhere: privileged targets stay out of ordinary records.
    assert not any(
        "belief" in name
        for name in (*DecisionRecord.__dataclass_fields__, *GameRecord.__dataclass_fields__)
    )


def test_the_collection_policy_version_names_the_candidate():
    version = p6.collection_policy_version("C2", "float16")
    assert version == "phase6_candidate_c2_float16_v1"
    # Distinct from Agent 3's and from Phase 3's corpora.
    assert version != "synthetic_hash_policy_v1"
    assert version != "end_to_end_representative_probe_v1"


def test_stored_records_are_stamped_with_the_candidates_policy_version():
    config = small_config(environments=24, record_trajectories=True, retain_games=2)
    row = p6.measure_candidate_configuration(
        CANDIDATE, config, seconds=1.5, mode=p6.MODE_RECORDING, device=DEVICE, warmup_steps=2
    )
    retained = row.pop("retained_records", ())
    assert row["status"] == "ok", row.get("error")
    assert retained, "the recording run produced no sealed game"
    record = decode_game_record(retained[0])
    assert record.collection_policy_version == config.collection_policy_version
    assert record.trajectory_version == TRAJECTORY_VERSION
    assert record.snapshot_interval == config.snapshot_interval
    assert all(
        decision.collection_policy_version == config.collection_policy_version
        for decision in record.decisions
    )


# ---------------------------------------------------------------------------
# v2 reconstruction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorded_games():
    """A handful of real games recorded through a real candidate."""
    config = p6.candidate_configuration(
        CANDIDATE,
        workers=2,
        environments=24,
        inference_batch_size=24,
        precision=PRECISION,
        record_trajectories=True,
        detailed_timing=False,
        verify_target_decisions=60,
        max_concurrent_verifications=4,
        retain_games=3,
        root_seed=77_021,
    )
    row = p6.measure_candidate_configuration(
        CANDIDATE, config, seconds=6.0, mode=p6.MODE_RECORDING, device=DEVICE, warmup_steps=2
    )
    retained = row.pop("retained_records", ())
    if row["status"] != "ok" or not retained:
        pytest.skip(f"no games were sealed in the time budget: {row.get('error')}")
    return {"row": row, "payloads": retained, "config": config}


def test_live_decisions_reconstruct_without_a_mismatch(recorded_games):
    row = recorded_games["row"]
    assert row["verified_decisions"] > 0
    assert row["reconstruction_mismatches"] == 0


def test_stored_games_rebuild_in_both_frames(recorded_games):
    report = p6.reconstruct_stored_games(recorded_games["payloads"])
    assert report.games_sampled > 0
    assert report.decisions_sampled > 0
    assert report.codec_round_trips == report.games_sampled
    assert report.total_mismatches == 0, report.details
    assert report.belief_targets_found_in_record == 0


def test_the_recorded_distribution_is_the_candidates_own(recorded_games):
    """Proves the probabilities were gathered through the frame, not past it.

    A record stores one probability per legal action in ascending *absolute*
    order, while the model produced them in the normalized frame. Re-running the
    candidate on the rebuilt observation and gathering through Agent 1's
    converter must land on the same numbers; a frame error here would move mass
    onto entirely different moves.
    """
    model = p6.build_pipeline_candidate(CANDIDATE).to(
        device=torch.device(DEVICE),
        dtype=torch.float32 if PRECISION == "float32" else torch.float16,
    )
    model.eval()
    report = p6.reconstruct_stored_games(
        recorded_games["payloads"],
        model=model,
        device=torch.device(DEVICE),
        dtype=torch.float32 if PRECISION == "float32" else torch.float16,
        max_decisions_per_game=25,
    )
    assert report.policy_reevaluations > 0
    assert report.policy_reevaluation_over_tolerance == 0
    assert report.policy_reevaluation_max_deviation <= p6.POLICY_REEVALUATION_TOLERANCE


def test_reconstruction_notices_a_tampered_selected_action(recorded_games):
    """A negative control: the checker must not pass anything handed to it."""
    record = decode_game_record(recorded_games["payloads"][0])
    target = next(
        (
            decision
            for decision in record.decisions
            if len(decision.legal_action_ids) > 1
        ),
        None,
    )
    if target is None:
        pytest.skip("no stored decision had more than one legal action")
    other = next(
        action for action in target.legal_action_ids if action != target.selected_action_id
    )
    tampered = GameRecord(
        **{
            **{
                name: getattr(record, name)
                for name in GameRecord.__dataclass_fields__
                if name != "decisions"
            },
            "decisions": tuple(
                DecisionRecord(
                    **{
                        **{
                            name: getattr(decision, name)
                            for name in DecisionRecord.__dataclass_fields__
                            if name != "selected_action_id"
                        },
                        "selected_action_id": (
                            other if decision is target else decision.selected_action_id
                        ),
                    }
                )
                for decision in record.decisions
            ),
        }
    )
    report = p6.reconstruct_stored_games([encode_game_record(tampered)])
    assert report.total_mismatches > 0
    assert report.absolute_selection_mismatches > 0 or report.schema_problems > 0


# ---------------------------------------------------------------------------
# Storage accounting
# ---------------------------------------------------------------------------


def test_storage_projection_is_plain_arithmetic():
    gib = p6.BYTES_PER_GIB
    projection = p6.storage_projection(record_bytes=gib, seconds=3600.0, label="one")
    assert projection["gib_per_hour"] == pytest.approx(1.0)
    assert projection["gib_per_24_hours"] == pytest.approx(24.0)
    assert projection["gib_per_168_hours"] == pytest.approx(168.0)
    assert projection["bytes_per_second"] == pytest.approx(gib / 3600.0)
    assert projection["bytes_per_168_hours"] == pytest.approx(168.0 * gib)


def test_storage_projection_compares_against_both_volumes():
    projection = p6.storage_projection(record_bytes=p6.BYTES_PER_GIB, seconds=3600.0)
    assert projection["internal_free_bytes"] == p6.INTERNAL_FREE_BYTES
    assert projection["external_free_bytes"] == p6.EXTERNAL_FREE_BYTES
    assert projection["fraction_of_internal_free"] == pytest.approx(
        168 * p6.BYTES_PER_GIB / p6.INTERNAL_FREE_BYTES
    )
    assert projection["fits_internal_uncompressed"] is False
    assert projection["fits_external_uncompressed"] is True


def test_a_storage_rate_needs_a_measured_duration():
    with pytest.raises(ValueError, match="positive"):
        p6.storage_projection(record_bytes=1, seconds=0.0)


def test_a_storage_rate_needs_a_recording_configuration():
    with pytest.raises(ValueError, match="recording configuration"):
        p6.measure_storage_rate(CANDIDATE, small_config(), device=DEVICE)


def test_the_storage_rate_excludes_its_warmup():
    """The cold-start transient must be dropped, not averaged in.

    A trajectory is written only when a game is sealed and every environment
    starts at ply 0, so the opening steps of a run record decisions whose bytes
    do not exist yet. Dividing whole-run bytes by whole-run seconds understates
    the sustained rate by an order of magnitude at production scale, so the
    measured window has to start after the warmup and the reported figures have
    to be differences across that window rather than cumulative totals.
    """
    config = p6.candidate_configuration(
        CANDIDATE,
        workers=2,
        environments=16,
        inference_batch_size=16,
        precision=PRECISION,
        record_trajectories=True,
        detailed_timing=False,
        root_seed=77_033,
    )
    measurement = p6.measure_storage_rate(
        CANDIDATE, config, warmup_steps=40, measure_steps=40, sample_steps=10, device=DEVICE
    )

    warmup = [s for s in measurement["samples"] if not s["in_measured_window"]]
    measured = [s for s in measurement["samples"] if s["in_measured_window"]]
    assert warmup and measured
    assert all(sample["step"] <= 40 for sample in warmup)
    assert all(sample["step"] > 40 for sample in measured)
    assert measurement["measured_steps"] == 40

    # The headline totals are differences across the measured window, so they
    # must be strictly smaller than the whole run's cumulative totals.
    assert measurement["steady_state_record_bytes"] <= measurement["total_record_bytes"]
    assert measurement["measured_seconds"] < measurement["total_run_seconds"]
    assert measurement["steady_state_gib_per_hour"] == pytest.approx(
        measurement["steady_state_bytes_per_second"] * 3600 / p6.BYTES_PER_GIB
    )
    # And the naive figure is reported alongside, so the gap stays visible.
    assert "cumulative_gib_per_hour_if_naively_divided" in measurement


def test_the_storage_warmup_is_counted_in_steps_not_seconds():
    """A slower candidate must not get the *less* settled measurement.

    Desynchronizing the slots takes about two mean game lengths of simulated
    time, which is a number of global steps. A seconds-based warmup would give a
    candidate half as fast half as many steps to settle in.
    """
    assert isinstance(p6.STORAGE_WARMUP_STEPS, int)
    assert p6.STORAGE_WARMUP_STEPS >= 1000
    assert p6.STORAGE_MEASURE_STEPS >= 500


def test_recording_rows_account_for_every_stored_byte(recorded_games):
    row = recorded_games["row"]
    assert row["trajectory_bytes"] > 0
    assert row["trajectory_records"] > 0
    assert row["trajectory_decisions"] > 0
    assert row["snapshot_count"] >= row["trajectory_records"]
    assert row["bytes_per_decision"] == pytest.approx(
        row["trajectory_bytes"] / row["trajectory_decisions"]
    )
    assert row["bytes_per_game"] == pytest.approx(
        row["trajectory_bytes"] / row["trajectory_records"]
    )
    projection = p6.storage_projection(
        record_bytes=row["trajectory_bytes"], seconds=row["duration_seconds"]
    )
    assert row["gib_per_hour"] == pytest.approx(projection["gib_per_hour"])


def test_the_retained_sample_is_not_just_the_shortest_games():
    """Retention must be a reservoir, not "the first N".

    Every environment starts at ply 0, so the first games to seal are the
    shortest of the whole run. Keeping those would hand the reconstruction check
    and the storage measurement a sample of ten-ply games when a production game
    runs to about five hundred plies -- measured here at 417 decisions a game
    against 10 for the first-sealed sample.
    """
    from stratego.training.worker_pool import offer_to_reservoir

    capacity = 6
    stream = list(range(1000))
    means = []
    for seed in range(40):
        rng = __import__("random").Random(seed)
        retained: list[int] = []
        for seen, item in enumerate(stream, start=1):
            offer_to_reservoir(rng, retained, item, capacity=capacity, seen=seen)
        assert len(retained) == capacity
        means.append(sum(retained) / capacity)

    # A first-N policy would give exactly 2.5 every time; a uniform sample of
    # 0..999 averages 499.5. The gap is the whole point of the change.
    assert sum(means) / len(means) == pytest.approx(499.5, abs=90.0)
    assert min(means) > 2.5


def test_a_reservoir_smaller_than_its_capacity_keeps_everything():
    from stratego.training.worker_pool import offer_to_reservoir

    rng = __import__("random").Random(0)
    retained: list[int] = []
    for seen, item in enumerate(range(3), start=1):
        offer_to_reservoir(rng, retained, item, capacity=6, seen=seen)
    assert retained == [0, 1, 2]


def test_a_zero_capacity_reservoir_keeps_nothing():
    from stratego.training.worker_pool import offer_to_reservoir

    rng = __import__("random").Random(0)
    retained: list[int] = []
    offer_to_reservoir(rng, retained, 1, capacity=0, seen=1)
    assert retained == []


def test_a_collection_only_row_stores_nothing():
    config = small_config(environments=24)
    row = p6.measure_candidate_configuration(
        CANDIDATE, config, seconds=1.0, mode=p6.MODE_COLLECTION, device=DEVICE, warmup_steps=2
    )
    assert row["status"] == "ok", row.get("error")
    assert row["trajectory_bytes"] == 0
    assert row["gib_per_hour"] == 0.0
    assert row["trajectory_write_fraction"] == 0.0


# ---------------------------------------------------------------------------
# The finalist rule
# ---------------------------------------------------------------------------


def _summary(candidate_id, parameters, recording, **overrides):
    summary = {
        "candidate_id": candidate_id,
        "parameters": parameters,
        "standalone_inference_positions_per_second": 10_000.0,
        "standalone_training_examples_per_second": 1_000.0,
        "collection_positions_per_second": recording * 1.2,
        "recording_positions_per_second": recording,
        "gib_per_hour": 5.0,
        "process_rss_bytes": 1,
        "metal_memory_bytes": 1,
        "numerically_stable_float16": True,
        "bottleneck_ratio": 4.0,
    }
    summary.update(overrides)
    return summary


def test_the_finalist_rule_cannot_read_playing_strength():
    for key in p6.FINALIST_INPUT_KEYS:
        for forbidden in p6.FORBIDDEN_INPUT_SUBSTRINGS:
            assert forbidden not in key, f"{key} exposes {forbidden}"


def test_a_strength_field_cannot_change_the_finalists():
    honest = [
        _summary("C0", 123_223, 12_000.0),
        _summary("C1", 863_959, 9_000.0),
        _summary("C2", 1_922_519, 6_000.0),
        _summary("C3", 2_812_247, 4_000.0),
    ]
    poisoned = [
        {**summary, "win_rate": 0.99 if summary["candidate_id"] == "C2" else 0.01}
        for summary in honest
    ]
    assert (
        p6.recommend_finalists(honest)["finalist_ids"]
        == p6.recommend_finalists(poisoned)["finalist_ids"]
    )
    assert "win_rate" not in p6.finalist_inputs(poisoned[0])


def test_between_two_and_three_finalists_are_chosen():
    result = p6.recommend_finalists(
        [
            _summary("C0", 123_223, 12_000.0),
            _summary("C1", 863_959, 9_000.0),
            _summary("C2", 1_922_519, 6_000.0),
            _summary("C3", 2_812_247, 4_000.0),
        ]
    )
    assert 2 <= len(result["finalist_ids"]) <= 3
    # The span, not the top three: smallest and largest must both be present.
    assert "C0" in result["finalist_ids"]
    assert "C3" in result["finalist_ids"]


def test_an_unstable_candidate_is_excluded():
    result = p6.recommend_finalists(
        [
            _summary("C0", 123_223, 12_000.0),
            _summary("C1", 863_959, 9_000.0),
            _summary("C2", 1_922_519, 6_000.0, numerically_stable_float16=False),
        ]
    )
    assert "C2" not in result["finalist_ids"]
    assert result["verdicts"]["C2"] == "EXCLUDED"


def test_a_dominated_candidate_is_not_a_finalist():
    result = p6.recommend_finalists(
        [
            _summary("C0", 500_000, 9_000.0),
            # Same throughput and memory, strictly more capacity: C0 is dominated.
            _summary("C1", 900_000, 9_000.0),
            _summary("C2", 1_922_519, 6_000.0),
        ]
    )
    assert result["verdicts"]["C0"] == "DOMINATED"
    assert "C0" not in result["finalist_ids"]


def test_a_candidate_that_never_recorded_is_excluded():
    result = p6.recommend_finalists(
        [
            _summary("C0", 123_223, 12_000.0),
            _summary("C1", 863_959, 9_000.0),
            _summary("C2", 1_922_519, 0.0),
        ]
    )
    assert result["verdicts"]["C2"] == "EXCLUDED"
    assert "no sustained production-recording throughput" in result["reasons"]["C2"]
