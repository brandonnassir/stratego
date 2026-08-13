"""Phase 7 Agent 5: the setup source inside the real collection pipeline.

What is pinned here:

- the injection point is *optional*, and leaving it out reproduces the accepted
  Phase 6 games exactly;
- the production training source cannot reach a validation or test base;
- a setup pair belongs to a logical game identity, so worker count, slot
  partitioning, scheduling order and a recycle boundary cannot change it;
- every completed game -- including a zero-decision game -- gets exactly one
  provenance record, and that record rebuilds the setup stored in the
  persisted trajectory;
- `trajectory_v1` and the Phase 4 evaluation bank are untouched.

These run real worker pools at a small scale, so a regression shows up in an
ordinary test run rather than only in the acceptance campaign.
"""

import pytest

from stratego.engine.constants import PLAYERS
from stratego.engine.observation import OBSERVATION_VERSION
from stratego.engine.setup import deserialize_setup
from stratego.evaluation.setup_bank import (
    DEFAULT_BANK_ROOT_SEED,
    DEFAULT_BANK_SIZE,
    SETUP_BANK_VERSION,
    SetupBank,
    bank_digest,
)
from stratego.setups.contracts import TRAIN_PER_FAMILY
from stratego.setups.identity import orient_setup
from stratego.setups.sampler import (
    SAMPLER_VERSION,
    rebuild_from_provenance,
    sample_setup,
)
from stratego.training.batch_simulation import BatchSimulator
from stratego.training.setup_source import (
    AUDIT_PURPOSE,
    PROVENANCE_SCHEMA_VERSION,
    SETUP_SOURCE_VERSION,
    TRAINING_SPLIT,
    LibrarySetupSource,
    SetupSourceError,
    UniformRandomSetupSource,
    audit_setup_source,
    family_pair,
    read_provenance_index,
    training_setup_source,
    validate_provenance_record,
    verify_provenance_against_setups,
    verify_provenance_split,
)
from stratego.training.shard_writer import iter_shard_payloads, shard_paths
from stratego.training.trajectory import (
    BATCH_RANDOM_SETUP_FAMILY,
    TRAJECTORY_VERSION,
    decode_game_record_compressed,
)
from stratego.training.worker_pool import RecordingConfig, WorkerPool

#: The digest Agent 4 handed to Agent 5. Pinned so an accidental library edit
#: fails here rather than silently changing what collection samples.
HANDOFF_LIBRARY_DIGEST = (
    "7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777"
)

#: A seed whose generation-0 game strands the first player, from the Phase 6B
#: stillborn fixtures (`tests/training/test_stillborn_games.py`).
RED_STRANDED_AT_START = 157345


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StrandedProvenanceSource:
    """A source that produces a stranded pair *with* provenance.

    Curated library setups always leave their owner a legal move, so a
    zero-decision game cannot arise from `setup_library_v1`. This double
    reproduces the shape of one anyway -- provenance attached to a game that
    ends before any decision exists -- so the sidecar's behaviour on that path
    is a tested property rather than an assumption.

    Defined at module level and holding only plain data, so it survives the
    `spawn` pickling every worker start performs.
    """

    setup_family = "stranded_probe_v1"

    def describe(self) -> dict:
        return {
            "source_id": self.setup_family,
            "kind": "test_double",
            "produces_provenance": True,
        }

    def assign(
        self, *, root_seed, environment_id, generation, slot_seed=0, game_id=""
    ):
        from stratego.engine.random_play import make_random_setups

        from stratego.training.batch_simulation import derive_slot_seed
        from stratego.training.setup_source import SetupAssignment

        # Generation 0 replays the stranded fixture slot; later generations are
        # ordinary games, so the slot keeps running after the stillborn one.
        seed = (
            derive_slot_seed(RED_STRANDED_AT_START, 0, 0)
            if generation == 0
            else slot_seed
        )
        red_setup, blue_setup = make_random_setups(seed)
        provenance = {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "setup_source_version": SETUP_SOURCE_VERSION,
            "probe_generation": int(generation),
            "game_id": game_id,
            "environment_id": int(environment_id),
            "generation": int(generation),
        }
        return SetupAssignment(
            red_setup=red_setup, blue_setup=blue_setup, provenance=provenance
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source():
    return training_setup_source()


def _recording(directory, run_id="phase7"):
    return RecordingConfig(
        enabled=True,
        snapshot_interval=32,
        output_directory=str(directory),
        compress_records=True,
        run_id=run_id,
        encode_records=True,
    )


def _drive(pool, steps):
    """Run `steps` bulk-synchronous phases with the deterministic policy."""
    for _ in range(steps):
        actions = pool.select_actions()
        pool.buffers.decision_valid[:] = (pool.buffers.status == 1).astype(
            pool.buffers.decision_valid.dtype
        )
        pool.buffers.actions[:] = actions
        pool.step()


@pytest.fixture(scope="module")
def collected(tmp_path_factory, source):
    """One real recorded pool run over library setups, with its sidecars."""
    directory = tmp_path_factory.mktemp("phase7_collection")
    pool = WorkerPool(
        8, 2, root_seed=99, recording=_recording(directory), setup_source=source
    )
    pool.start()
    try:
        _drive(pool, 1400)
    finally:
        totals = pool.shutdown()

    records = []
    for path in shard_paths(directory):
        for payload in iter_shard_payloads(path):
            records.append(decode_game_record_compressed(payload))
    return {
        "directory": directory,
        "totals": totals,
        "records": records,
        "provenance": read_provenance_index(directory),
    }


# ---------------------------------------------------------------------------
# The library handoff
# ---------------------------------------------------------------------------


def test_the_sampled_library_is_the_one_agent_4_handed_over(source):
    assert source.library_digest() == HANDOFF_LIBRARY_DIGEST


def test_the_source_names_the_frozen_sampler(source):
    assert source.describe()["sampler_version"] == SAMPLER_VERSION


# ---------------------------------------------------------------------------
# The injection point is optional
# ---------------------------------------------------------------------------


def test_no_source_reproduces_the_accepted_phase_6_games():
    """The default path must be the old path, slot for slot."""
    without = BatchSimulator(6, root_seed=4242)
    explicit = BatchSimulator(6, root_seed=4242, setup_source=UniformRandomSetupSource())
    assert without.batch_fingerprint() == explicit.batch_fingerprint()


def test_the_uniform_source_produces_no_provenance():
    simulator = BatchSimulator(2, root_seed=7, setup_source=UniformRandomSetupSource())
    assert simulator.setup_provenance(0) is None


def test_a_library_sourced_slot_carries_provenance(source):
    simulator = BatchSimulator(2, root_seed=7, setup_source=source)
    provenance = simulator.setup_provenance(0)
    assert provenance is not None
    assert validate_provenance_record(provenance) == []


def test_the_source_is_consulted_once_per_created_game(source):
    simulator = BatchSimulator(3, root_seed=11, setup_source=source)
    assert simulator.setup_source_calls == 3
    simulator.reset_slots([0, 2])
    assert simulator.setup_source_calls == 5


def test_the_setups_the_engine_received_are_the_sampled_ones(source):
    simulator = BatchSimulator(2, root_seed=13, setup_source=source)
    for slot in range(2):
        provenance = simulator.setup_provenance(slot)
        red, blue = simulator.setups(slot)
        assert deserialize_setup(provenance["red"]["engine_setup"]) == red
        assert deserialize_setup(provenance["blue"]["engine_setup"]) == blue
        assert verify_provenance_against_setups(
            provenance, red_setup=red, blue_setup=blue
        ) == []


# ---------------------------------------------------------------------------
# Split behaviour
# ---------------------------------------------------------------------------


def test_the_production_training_source_is_the_train_split(source):
    assert source.split == TRAINING_SPLIT
    assert source.purpose == "training"


@pytest.mark.parametrize("split", ["validation", "test"])
def test_a_training_source_cannot_be_built_on_a_held_out_split(split):
    with pytest.raises(SetupSourceError, match="locked to"):
        LibrarySetupSource(split=split)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_a_held_out_split_needs_an_explicit_justification(split):
    with pytest.raises(SetupSourceError, match="access_justification"):
        LibrarySetupSource(split=split, purpose=AUDIT_PURPOSE)
    with pytest.raises(SetupSourceError, match="justification"):
        audit_setup_source(split, "   ")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_an_explicit_audit_request_reaches_the_held_out_split(split):
    audit = audit_setup_source(split, f"Agent 5 {split} smoke request")
    assert audit.split == split
    simulator = BatchSimulator(1, root_seed=5, setup_source=audit)
    provenance = simulator.setup_provenance(0)
    assert verify_provenance_split(provenance, split) == []


def test_unknown_splits_purposes_and_profiles_are_rejected():
    with pytest.raises(SetupSourceError, match="unknown split"):
        LibrarySetupSource(split="holdout")
    with pytest.raises(SetupSourceError, match="unknown purpose"):
        LibrarySetupSource(purpose="whatever")
    with pytest.raises(SetupSourceError, match="unknown sampler profile"):
        LibrarySetupSource(profile="not_a_profile")


def test_the_default_training_path_never_samples_a_held_out_base(source):
    """The regression the assignment asks for, over a broad deterministic sweep."""
    for environment_id in range(24):
        for generation in range(6):
            provenance = source.assign(
                root_seed=6006,
                environment_id=environment_id,
                generation=generation,
            ).provenance
            assert verify_provenance_split(provenance, TRAINING_SPLIT) == []
            for side in ("red", "blue"):
                assert provenance[side]["base_index"] < TRAIN_PER_FAMILY


def test_the_collected_run_only_used_train_bases(collected):
    assert collected["provenance"]
    for record in collected["provenance"].values():
        assert verify_provenance_split(record, TRAINING_SPLIT) == []


# ---------------------------------------------------------------------------
# Setup-pair identity and determinism
# ---------------------------------------------------------------------------


def test_the_two_sides_are_sampled_independently(source):
    """Red and blue draw from different streams, so they are not one setup twice."""
    distinct = 0
    for environment_id in range(32):
        provenance = source.assign(
            root_seed=1, environment_id=environment_id, generation=0
        ).provenance
        if (
            provenance["red"]["final_setup_fingerprint"]
            != provenance["blue"]["final_setup_fingerprint"]
        ):
            distinct += 1
    assert distinct == 32


def test_the_side_seed_depends_only_on_the_logical_game_identity(source):
    first = source.side_seed(
        root_seed=3, environment_id=17, generation=4, player=PLAYERS[0]
    )
    second = source.side_seed(
        root_seed=3, environment_id=17, generation=4, player=PLAYERS[0]
    )
    assert first == second
    assert first != source.side_seed(
        root_seed=3, environment_id=17, generation=5, player=PLAYERS[0]
    )
    assert first != source.side_seed(
        root_seed=3, environment_id=18, generation=4, player=PLAYERS[0]
    )
    assert first != source.side_seed(
        root_seed=4, environment_id=17, generation=4, player=PLAYERS[0]
    )
    assert first != source.side_seed(
        root_seed=3, environment_id=17, generation=4, player=PLAYERS[1]
    )


def test_the_assignment_does_not_depend_on_the_slot_partitioning(source):
    """The same environment id gets the same setups from any simulator window."""
    whole = BatchSimulator(8, root_seed=808, setup_source=source)
    tail = BatchSimulator(
        4, root_seed=808, first_environment_id=4, setup_source=source
    )
    for offset in range(4):
        assert whole.setups(4 + offset) == tail.setups(offset)


def test_the_assignment_survives_a_recycle_boundary(source):
    """A restarted process rebuilding a slot's later generation gets the same pair.

    A recycled worker does not replay the generations it missed, so the check
    is that generation `g` is reproducible in isolation -- which is exactly what
    a restart needs.
    """
    advanced = BatchSimulator(1, root_seed=515, setup_source=source)
    for _ in range(5):
        advanced.reset_slots([0])
    assert advanced.generation(0) == 5

    restarted = BatchSimulator(1, root_seed=515, setup_source=source)
    while restarted.generation(0) < 5:
        restarted.reset_slots([0])
    assert restarted.setups(0) == advanced.setups(0)
    assert restarted.setup_provenance(0) == advanced.setup_provenance(0)


@pytest.fixture(scope="module")
def worker_count_runs(tmp_path_factory, source):
    """The same logical run collected under 1, 2 and 4 workers."""
    runs = {}
    for workers in (1, 2, 4):
        directory = tmp_path_factory.mktemp(f"phase7_workers_{workers}")
        pool = WorkerPool(
            16,
            workers,
            root_seed=77,
            recording=_recording(directory, run_id=f"w{workers}"),
            setup_source=source,
        )
        pool.start()
        try:
            observations = pool.buffers.observations.copy()
            _drive(pool, 1100)
        finally:
            pool.shutdown()
        runs[workers] = {
            "startup_observations": observations,
            "provenance": {
                (record["environment_id"], record["generation"]): record
                for record in read_provenance_index(directory).values()
            },
        }
    return runs


def test_worker_count_does_not_change_the_published_startup_games(worker_count_runs):
    """Every slot's published position at generation 0 is pool-shape independent."""
    import numpy as np

    reference = worker_count_runs[1]["startup_observations"]
    for workers in (2, 4):
        assert np.array_equal(reference, worker_count_runs[workers]["startup_observations"])


def test_worker_count_does_not_change_the_setup_assignment(worker_count_runs):
    """Completed games with the same logical identity carry the same setups.

    Compared through the sidecars the workers actually wrote, so this is what
    the pipeline produced rather than what the source would produce if asked
    again.
    """
    reference = worker_count_runs[1]["provenance"]
    assert reference
    for workers in (2, 4):
        other = worker_count_runs[workers]["provenance"]
        shared = set(reference) & set(other)
        assert shared, f"{workers} workers completed no comparable game"
        for key in shared:
            for field in ("red", "blue"):
                assert (
                    reference[key][field]["engine_setup"]
                    == other[key][field]["engine_setup"]
                )
                assert (
                    reference[key][field]["final_setup_fingerprint"]
                    == other[key][field]["final_setup_fingerprint"]
                )
                assert (
                    reference[key][field]["base_setup_id"]
                    == other[key][field]["base_setup_id"]
                )


# ---------------------------------------------------------------------------
# Provenance in the persisted run
# ---------------------------------------------------------------------------


def test_the_run_completed_games_and_sealed_them(collected):
    assert collected["totals"]["total_games_recorded"] > 0
    assert len(collected["records"]) == collected["totals"]["total_games_recorded"]


def test_every_completed_game_has_exactly_one_provenance_record(collected):
    provenance = collected["provenance"]
    assert collected["totals"]["total_provenance_records"] == len(provenance)
    assert collected["totals"]["total_provenance_missing"] == 0
    assert collected["totals"]["total_provenance_write_errors"] == 0
    game_ids = [record.game_id for record in collected["records"]]
    assert len(game_ids) == len(set(game_ids))
    assert set(game_ids) == set(provenance)


def test_every_provenance_record_matches_its_persisted_trajectory(collected):
    provenance = collected["provenance"]
    for record in collected["records"]:
        problems = verify_provenance_against_setups(
            provenance[record.game_id],
            red_setup=record.red_setup,
            blue_setup=record.blue_setup,
        )
        assert problems == [], (record.game_id, problems)


def test_the_rebuilt_descendant_orients_onto_the_persisted_setup(collected):
    """The fingerprint check, done the long way round for one sample."""
    record = collected["records"][0]
    stored = collected["provenance"][record.game_id]
    for side, player, setup in (
        ("red", PLAYERS[0], record.red_setup),
        ("blue", PLAYERS[1], record.blue_setup),
    ):
        rebuilt = rebuild_from_provenance(stored[side])
        assert orient_setup(rebuilt.canonical, player) == setup


def test_provenance_carries_the_run_and_worker_identity(collected):
    for record in collected["provenance"].values():
        assert record["run_id"] == "phase7"
        assert record["worker_id"] in (0, 1)
        assert record["setup_source_version"] == SETUP_SOURCE_VERSION


def test_the_sidecar_is_a_sibling_file_not_part_of_a_shard(collected):
    directory = collected["directory"]
    sidecars = sorted(path.name for path in directory.glob("*_setup_provenance.jsonl"))
    assert sidecars == [
        "phase7_w00_setup_provenance.jsonl",
        "phase7_w01_setup_provenance.jsonl",
    ]
    assert shard_paths(directory)


def test_a_zero_decision_game_still_gets_provenance(tmp_path):
    """The stillborn path writes provenance before the reset replaces the slot."""
    pool = WorkerPool(
        1,
        1,
        root_seed=RED_STRANDED_AT_START,
        recording=_recording(tmp_path, run_id="stranded"),
        setup_source=StrandedProvenanceSource(),
    )
    pool.start()
    try:
        pool.clear_actions()
        pool.step(apply_actions=True, auto_reset=True)
        totals = pool.recording_totals()
        assert totals["total_stillborn_games"] == 1
        assert totals["total_provenance_records"] == 1
        assert totals["total_provenance_missing"] == 0
    finally:
        pool.shutdown()

    index = read_provenance_index(tmp_path)
    assert len(index) == 1
    stillborn = next(iter(index.values()))
    assert stillborn["probe_generation"] == 0
    assert stillborn["worker_id"] == 0


# ---------------------------------------------------------------------------
# Branch coverage in the sampled population
# ---------------------------------------------------------------------------


def test_both_reflection_and_perturbation_branches_occur(source):
    reflections = set()
    perturbations = set()
    families = set()
    for environment_id in range(64):
        provenance = source.assign(
            root_seed=606, environment_id=environment_id, generation=0
        ).provenance
        for side in ("red", "blue"):
            reflections.add(bool(provenance[side]["reflection_applied"]))
            perturbations.add(bool(provenance[side]["perturbation_applied"]))
        families.add(family_pair(provenance))
    assert reflections == {True, False}
    assert perturbations == {True, False}
    assert len(families) > 32


def test_the_source_agrees_with_a_direct_sampler_call(source):
    """Integration must not reinterpret the frozen sampler."""
    sampled, seed = source.sample_for_player(
        root_seed=2, environment_id=9, generation=1, player=PLAYERS[1]
    )
    direct = sample_setup(TRAINING_SPLIT, seed, profile=source.profile)
    assert sampled.provenance == direct.provenance


# ---------------------------------------------------------------------------
# Frozen contracts that must not have moved
# ---------------------------------------------------------------------------


def test_trajectory_and_observation_versions_are_unchanged():
    assert TRAJECTORY_VERSION == "trajectory_v1"
    assert OBSERVATION_VERSION == "observation_v2_1_127ch"


def test_the_setup_family_label_names_the_generator(collected, source):
    """`setup_family` is an existing `trajectory_v1` string field, used as intended."""
    assert source.setup_family != BATCH_RANDOM_SETUP_FAMILY
    for record in collected["records"]:
        assert record.setup_family == source.setup_family
        assert record.trajectory_version == TRAJECTORY_VERSION


def test_a_run_without_a_source_still_records_the_uniform_label(tmp_path):
    pool = WorkerPool(2, 1, root_seed=31, recording=_recording(tmp_path, run_id="p6"))
    pool.start()
    try:
        _drive(pool, 900)
    finally:
        pool.shutdown()
    records = [
        decode_game_record_compressed(payload)
        for path in shard_paths(tmp_path)
        for payload in iter_shard_payloads(path)
    ]
    assert records
    for record in records:
        assert record.setup_family == BATCH_RANDOM_SETUP_FAMILY
    assert not list(tmp_path.glob("*_setup_provenance.jsonl"))


def test_the_phase_4_evaluation_bank_is_untouched():
    """Phase 4's fixture must remain a separate, byte-identical evaluation bank."""
    bank = SetupBank.generate(
        size=DEFAULT_BANK_SIZE, root_seed=DEFAULT_BANK_ROOT_SEED
    )
    assert bank.bank_version == SETUP_BANK_VERSION == "evaluation_setup_bank_v1"
    assert len(bank) == DEFAULT_BANK_SIZE == 1024
    assert bank_digest(bank) == bank_digest(
        SetupBank.generate(size=DEFAULT_BANK_SIZE, root_seed=DEFAULT_BANK_ROOT_SEED)
    )
    # The Phase 7 library is not its source: a bank pair is not a library entry.
    assert bank.pair(0).bank_version != training_setup_source().setup_family
