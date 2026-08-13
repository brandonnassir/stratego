"""Stillborn games through the training stack (`phase2_1_reference_1.2.0`).

A "stillborn" game is one that is terminal at creation because the first
player is stranded. Engine-level behaviour is pinned in
`tests/engine/test_initial_mobility.py`; this module covers everything the
Phase 6B abort exposed above the engine:

- the exact production case `(root_seed 60006, environment 112, generation
  98)` that aborted the soak at t = 8,981 s;
- `BatchSimulator` creating and resetting through a stillborn generation;
- the worker pool publishing the formerly failing state as TERMINAL rather
  than active-with-zero-legality, sealing its outcome into the shared `last_*`
  fields, and persisting its zero-decision record;
- `trajectory_v1` round-tripping a zero-decision record without any schema
  change.

The fixture seeds were found by exhaustive scan and are pinned forever:
`157345` strands red at (env 0, gen 0); `151139` strands red at (env 0,
gen 2); `1032652` strands *blue* at (env 0, gen 0), which must NOT be terminal
at creation. Neighbouring generations of each are ordinary mobile games.
"""

import pytest

from stratego.engine.constants import (
    BLUE,
    RED,
    TERMINAL_OPPONENT_NO_LEGAL_MOVE,
)
from stratego.engine.legal_moves import has_legal_action
from stratego.engine.state import state_fingerprint
from stratego.training.batch_simulation import (
    BatchSimulator,
    BatchTerminalStateError,
    derive_slot_seed,
    slot_game_id,
)
from stratego.training.reconstruction import reconstruct_state
from stratego.training.shard_writer import shard_paths, verify_shard
from stratego.training.shared_buffers import (
    NO_ACTING_PLAYER,
    STATUS_ACTIVE,
    STATUS_TERMINAL,
    terminal_reason_name,
)
from stratego.training.trajectory import (
    builder_for_slot,
    decode_game_record,
    encode_game_record,
    validate_game_record,
)
from stratego.training.worker_pool import RecordingConfig, WorkerPool

#: The aborted soak's exact failing identity.
SOAK_ROOT_SEED = 60006
SOAK_ENVIRONMENT = 112
SOAK_GENERATION = 98

#: Exhaustively located fixture seeds (see module docstring).
RED_STRANDED_AT_START = 157345
RED_STRANDED_AT_GENERATION_2 = 151139
BLUE_STRANDED_AT_START = 1032652


@pytest.fixture(scope="module")
def simulator():
    """The soak's slot 112 advanced to generation 98, exactly as it built it."""
    simulator = BatchSimulator(
        1, root_seed=SOAK_ROOT_SEED, first_environment_id=SOAK_ENVIRONMENT
    )
    for _ in range(SOAK_GENERATION):
        simulator.reset_slots([0])
    return simulator


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One real worker-pool run over the stillborn fixture seed.

    Startup publishes the stillborn generation-0 game; the single step seals
    its outcome and record and resets the slot; shutdown closes the shard.
    """
    directory = tmp_path_factory.mktemp("stillborn_shards")
    pool = WorkerPool(
        1,
        1,
        root_seed=RED_STRANDED_AT_START,
        recording=RecordingConfig(
            enabled=True,
            output_directory=str(directory),
            run_id="stillborn",
            compress_records=True,
        ),
    )
    result = {"directory": directory}
    try:
        pool.start()
        result["startup"] = {
            "status": int(pool.buffers.status[0]),
            "terminal": int(pool.buffers.terminal[0]),
            "legal_count": int(pool.buffers.legal_count[0]),
            "acting_player": int(pool.buffers.acting_player[0]),
            "generation": int(pool.buffers.generation[0]),
            "episode_count": int(pool.buffers.episode_count[0]),
        }
        pool.clear_actions()
        report = pool.step(apply_actions=True, auto_reset=True)
        result["step"] = report
        result["totals"] = pool.recording_totals()
        result["after"] = {
            "status": int(pool.buffers.status[0]),
            "generation": int(pool.buffers.generation[0]),
            "episode_count": int(pool.buffers.episode_count[0]),
            "last_terminal_reason": terminal_reason_name(
                int(pool.buffers.last_terminal_reason[0])
            ),
            "last_winner": int(pool.buffers.last_winner[0]),
            "last_total_moves": int(pool.buffers.last_total_moves[0]),
            "last_result_blue": float(pool.buffers.last_result_blue[0]),
        }
    finally:
        result["shutdown"] = pool.shutdown()
    return result


class TestTheExactProductionCase:
    """`batch60006-env000112-gen000098`, rebuilt exactly as the soak built it."""

    def test_generation_98_is_terminal_at_creation(self, simulator):
        assert simulator.generation(0) == SOAK_GENERATION
        assert simulator.environment_id(0) == SOAK_ENVIRONMENT
        assert simulator.is_terminal(0) is True
        state = simulator.game_state(0)
        assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert state.winner == BLUE
        assert state.total_moves == 0
        assert has_legal_action(state, RED) is False
        assert has_legal_action(state, BLUE) is True

    def test_the_published_legality_products_are_empty_and_consistent(self, simulator):
        assert simulator.legal_actions(0) == []
        assert not simulator.legal_action_mask(0).any()
        assert simulator.acting_player(0) == NO_ACTING_PLAYER

    def test_the_outcome_names_the_winner_and_reason(self, simulator):
        outcome = simulator.outcome(0)
        assert outcome.terminal is True
        assert outcome.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert outcome.winner == BLUE
        assert outcome.total_moves == 0
        assert outcome.result_for_red == -1.0
        assert outcome.result_for_blue == 1.0

    def test_the_identity_is_the_soak_abort_identity(self, simulator):
        assert simulator.game_id(0) == slot_game_id(
            SOAK_ROOT_SEED, SOAK_ENVIRONMENT, SOAK_GENERATION
        )
        assert simulator.slot_seed(0) == derive_slot_seed(
            SOAK_ROOT_SEED, SOAK_ENVIRONMENT, SOAK_GENERATION
        )


class TestBatchSimulatorLifecycle:
    def test_a_stillborn_generation_is_reached_and_left_by_ordinary_resets(self):
        simulator = BatchSimulator(1, root_seed=RED_STRANDED_AT_GENERATION_2)
        assert simulator.is_terminal(0) is False  # generation 0 is ordinary
        simulator.reset_slots([0])
        assert simulator.is_terminal(0) is False  # generation 1 is ordinary
        simulator.reset_slots([0])
        assert simulator.generation(0) == 2
        assert simulator.is_terminal(0) is True  # the stranded generation
        assert simulator.active_slots() == ()
        assert simulator.finished_slots() == (0,)
        simulator.reset_slots([0])
        assert simulator.generation(0) == 3
        assert simulator.is_terminal(0) is False  # and out the other side

    def test_stepping_a_stillborn_slot_is_refused_like_any_terminal_slot(self):
        simulator = BatchSimulator(1, root_seed=RED_STRANDED_AT_START)
        assert simulator.is_terminal(0) is True
        with pytest.raises(BatchTerminalStateError):
            simulator.step({0: 0})

    def test_a_stranded_second_player_is_not_terminal_at_creation(self):
        simulator = BatchSimulator(1, root_seed=BLUE_STRANDED_AT_START)
        state = simulator.game_state(0)
        assert state.terminal is False
        assert has_legal_action(state, RED) is True
        assert has_legal_action(state, BLUE) is False
        assert len(simulator.legal_actions(0)) > 0


class TestZeroDecisionTrajectoryRecord:
    """`trajectory_v1` carries a stillborn game without any schema change."""

    @pytest.fixture()
    def record(self):
        simulator = BatchSimulator(1, root_seed=RED_STRANDED_AT_START)
        builder = builder_for_slot(simulator, 0)
        return builder.finish(simulator.game_state(0))

    def test_the_record_validates_and_round_trips(self, record):
        assert validate_game_record(record) == []
        assert record.final_ply == 0
        assert record.actions == ()
        assert record.decisions == ()
        assert len(record.snapshots) == 1
        assert record.snapshots[0].ply == 0
        decoded = decode_game_record(encode_game_record(record))
        assert validate_game_record(decoded) == []
        assert decoded.game_id == record.game_id
        assert decoded.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert decoded.terminal_result == "blue_win"

    def test_the_record_reconstructs_its_own_terminal_state(self, record):
        decoded = decode_game_record(encode_game_record(record))
        state, replayed = reconstruct_state(decoded, 0)
        assert replayed == 0
        assert state.terminal is True
        assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert state.winner == BLUE
        # The recorded setups regenerate the identical position.
        simulator = BatchSimulator(1, root_seed=RED_STRANDED_AT_START)
        assert decoded.red_setup == simulator.setups(0)[0]
        assert decoded.blue_setup == simulator.setups(0)[1]
        assert state_fingerprint(state, include_history=False) == state_fingerprint(
            simulator.game_state(0), include_history=False
        )


class TestWorkerPoolPublishesAndSealsStillbornGames:
    """The formerly failing publish path, end to end through real processes."""

    def test_startup_publishes_the_stillborn_slot_as_terminal_not_active(self, run):
        startup = run["startup"]
        assert startup["status"] == STATUS_TERMINAL
        assert startup["status"] != STATUS_ACTIVE
        assert startup["terminal"] == 1
        assert startup["legal_count"] == 0
        assert startup["acting_player"] == NO_ACTING_PLAYER
        assert startup["generation"] == 0

    def test_the_first_step_seals_the_outcome_and_moves_on(self, run):
        assert run["startup"]["episode_count"] == 0
        after = run["after"]
        assert after["episode_count"] == 1
        assert after["last_terminal_reason"] == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert after["last_winner"] == BLUE
        assert after["last_total_moves"] == 0
        assert after["last_result_blue"] == 1.0
        assert after["generation"] == 1
        assert after["status"] == STATUS_ACTIVE  # the next game is ordinary

    def test_the_stillborn_game_is_counted_not_dropped(self, run):
        totals = run["totals"]
        assert totals["total_stillborn_games"] == 1
        assert totals["total_games_recorded"] == 1
        assert totals["total_records_persisted"] == 1
        assert totals["total_decisions_recorded"] == 0
        shutdown = run["shutdown"]
        assert shutdown["total_stillborn_games"] == 1
        assert shutdown["total_games_recorded"] == 1

    def test_the_persisted_record_is_the_stillborn_game(self, run):
        paths = shard_paths(run["directory"])
        assert len(paths) == 1
        verified = verify_shard(paths[0], decode=True, keep_records=True)
        assert verified["ok"], verified["problems"]
        assert verified["record_count"] == 1
        record = verified["records"][0]
        assert record.game_id == slot_game_id(RED_STRANDED_AT_START, 0, 0)
        assert record.final_ply == 0
        assert record.decisions == ()
        assert record.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert record.terminal_result == "blue_win"
        assert validate_game_record(record) == []
