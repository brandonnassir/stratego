"""Parallel neural evaluation must not change a single game.

What is under test
------------------
Phase 4 already proved that a *baseline* schedule survives being spread across
workers. The new risk in Phase 6 is the model: a checkpoint, a device, and a
batching design sitting between the engine and the action. So these tests check
four separate claims, and each one can fail independently:

1. the inference owner is the only holder of the checkpoint, and it loads it
   once no matter how many runs or workers it serves;
2. the payload that crosses the process boundary carries nothing privileged;
3. the remote decision path is the *same* decision path as the serial Phase 5
   adapter -- not merely a path that happens to agree today;
4. a failure anywhere in that chain is loud and never becomes a move.

Cost note: every multi-worker test starts `spawn` processes, which costs about a
second per pool on macOS. The schedules here are therefore small -- the mechanism
is what is being tested, and the large 1/2/4/8/shuffled sweep lives in
`scripts/run_phase6_agent05.py`. C0 is used rather than the C1 the acceptance run
uses because the topology under test is identical and C0 is seven times cheaper.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from stratego.engine.constants import ACTION_SPACE_SIZE, BLUE, RED
from stratego.evaluation.match_runner import (
    ERROR_POLICY_EXCEPTION,
    ON_POLICY_ERROR_QUARANTINE,
    RESULT_ERROR,
    PolicyFailure,
    compare_results,
    play_match,
    run_schedule,
)
from stratego.evaluation.match_spec import build_paired_schedule, schedule_matches
from stratego.evaluation.neural_worker import (
    BATCH_POLICY_ARRIVAL,
    BATCH_POLICY_SINGLE,
    DECISION_MODE_CATEGORICAL,
    DECISION_MODE_GREEDY,
    NEURAL_DECISION_RULE_VERSION,
    REQUEST_FIELDS,
    InferenceFailure,
    InferenceOwner,
    InferenceRequest,
    InferenceResponse,
    LocalInferenceChannel,
    NeuralEvaluationError,
    RemoteInferenceError,
    RemoteNeuralPolicy,
    checkpoint_load_count,
    field_level_mismatches,
    neural_policy_ref,
    run_neural_schedule,
    sweep_digests,
)
from stratego.evaluation.registry import ALL_POLICY_IDS, policy_ref
from stratego.evaluation.setup_bank import SetupBank
from stratego.model import policy_adapter
from stratego.model.checkpoint import CheckpointError, save_checkpoint
from stratego.model.policy_adapter import NeuralCheckpointPolicy
from stratego.model.production_model import build_candidate_model

CANDIDATE = "C0"
FAMILY_SEED = 20250601

#: 2 setup pairs against 2 opponents = 8 matches, both colours of each pair.
#: Small enough for several process pools, wide enough that a colour- or
#: opponent-specific bug cannot hide.
PAIR_IDS = range(2)
OPPONENTS = ("basic_heuristic", "tactical_rule_based")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A real C0 checkpoint on disk, written once for the module."""
    directory = tmp_path_factory.mktemp("phase6_neural_worker")
    model = build_candidate_model(CANDIDATE, seed=FAMILY_SEED)
    return save_checkpoint(
        model,
        directory / "phase6_c0.pt",
        training_metrics={"note": "untrained Phase 6 candidate; strength is meaningless"},
    )


@pytest.fixture(scope="module")
def greedy_ref():
    return neural_policy_ref(CANDIDATE, decision_mode=DECISION_MODE_GREEDY)


@pytest.fixture(scope="module")
def sampled_ref():
    return neural_policy_ref(CANDIDATE, decision_mode=DECISION_MODE_CATEGORICAL)


@pytest.fixture(scope="module")
def bank():
    return SetupBank.generate(size=max(PAIR_IDS) + 1)


@pytest.fixture(scope="module")
def schedule(greedy_ref):
    matches = []
    for opponent in OPPONENTS:
        matches.extend(
            schedule_matches(build_paired_schedule(greedy_ref, policy_ref(opponent), PAIR_IDS))
        )
    return tuple(matches)


@pytest.fixture(scope="module")
def owner(checkpoint):
    """One long-lived owner for every test in this module.

    Deliberately module-scoped: "the checkpoint is loaded once per owner however
    many runs it serves" is a property that only means something if an owner
    actually serves several runs.
    """
    instance = InferenceOwner(checkpoint, device="cpu", name="test_owner")
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def runs(schedule, bank, owner):
    """The same schedule at three worker counts plus a shuffled input order."""
    shuffled = tuple(reversed(schedule))
    return {
        "1": run_neural_schedule(schedule, bank, owner, worker_count=1),
        "2": run_neural_schedule(schedule, bank, owner, worker_count=2),
        "3": run_neural_schedule(schedule, bank, owner, worker_count=3),
        "3_shuffled": run_neural_schedule(shuffled, bank, owner, worker_count=3),
    }


def one_request(schedule, bank, greedy_ref):
    """A genuine first-ply request, captured from a real game."""
    captured = {}

    class Capture(RemoteNeuralPolicy):
        def decide(self, request):
            captured.setdefault("input", request)
            captured.setdefault("payload", InferenceRequest.from_policy_input(request))
            return super().decide(request)

    return captured, Capture


# ---------------------------------------------------------------------------
# 1. Inference owner lifecycle and checkpoint loading
# ---------------------------------------------------------------------------


def test_the_owner_loads_its_checkpoint_exactly_once(owner, runs):
    assert owner.checkpoint_load_count == 1
    for label, run in runs.items():
        assert run.inference["checkpoint_load_count"] == 1, label


def test_one_owner_serves_many_runs_without_reloading(owner, runs):
    """Four runs, one load. This is the "long-lived" half of the requirement."""
    assert len(runs) == 4
    assert owner.checkpoint_load_count == 1
    assert sum(run.decisions for run in runs.values()) > 0


def test_game_workers_never_load_a_checkpoint(runs):
    for label, run in runs.items():
        assert run.worker_checkpoint_loads == 0, label


def test_spawned_game_workers_do_not_import_torch(runs):
    """The MPS-ownership topology, checked from inside the workers."""
    for label in ("2", "3", "3_shuffled"):
        run = runs[label]
        assert run.workers_importing_torch == 0, label
        for report in run.workers:
            assert report["modules"]["model_modules_imported"] == [], label


def test_the_owner_reports_the_checkpoint_identity(owner, checkpoint):
    identity = owner.identity()
    assert identity["checkpoint_path"] == str(checkpoint)
    assert identity["model_architecture_id"] == "stratego_transformer_v1"
    assert identity["model_contract_version"] == "model_contract_v2"
    assert identity["policy_action_frame"] == "perspective_normalized_squares"
    assert identity["engine_action_frame"] == "absolute_engine_squares"
    assert identity["parameter_count"] == 123223
    assert identity["checkpoint_load_count"] == 1
    assert identity["checkpoint_file_digest"]


def test_a_closed_owner_refuses_to_serve(checkpoint):
    instance = InferenceOwner(checkpoint, device="cpu", name="closed")
    instance.close()
    instance.close()  # idempotent
    with pytest.raises(NeuralEvaluationError, match="is closed"):
        instance.serve_batch([])


def test_the_module_counter_notices_every_load(checkpoint):
    before = checkpoint_load_count()
    first = InferenceOwner(checkpoint, device="cpu", name="counted_a")
    second = InferenceOwner(checkpoint, device="cpu", name="counted_b")
    assert checkpoint_load_count() == before + 2
    first.close()
    second.close()


def test_the_decision_constants_have_not_drifted_from_the_adapter():
    """This module re-declares three strings so a worker need not import torch."""
    assert DECISION_MODE_GREEDY == policy_adapter.DECISION_MODE_GREEDY
    assert DECISION_MODE_CATEGORICAL == policy_adapter.DECISION_MODE_CATEGORICAL
    assert NEURAL_DECISION_RULE_VERSION == policy_adapter.NEURAL_POLICY_VERSION


# ---------------------------------------------------------------------------
# 2. Observer-safe payload
# ---------------------------------------------------------------------------


def test_the_request_carries_exactly_the_declared_fields():
    assert tuple(InferenceRequest.__dataclass_fields__) == REQUEST_FIELDS


def test_a_captured_request_reaches_no_privileged_object(schedule, bank, owner, greedy_ref):
    """Walk the whole object graph of a real request and refuse the forbidden."""
    captured, capture_class = one_request(schedule, bank, greedy_ref)
    policy = capture_class(greedy_ref, LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY)
    play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})

    payload = captured["payload"]
    forbidden = {
        "GameState",
        "PieceRecord",
        "ReplayRecord",
        "MoveRecord",
        "SetupBank",
        "SetupPair",
        "MatchSpec",
        "PolicyInput",
    }
    seen: set[int] = set()
    types: set[str] = set()

    def walk(value, depth=0):
        if id(value) in seen or depth > 8:
            return
        seen.add(id(value))
        types.add(type(value).__name__)
        if isinstance(value, np.ndarray):
            if value.base is not None:
                walk(value.base, depth + 1)
            return
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(key, depth + 1)
                walk(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item, depth + 1)
            return
        for name in getattr(value, "__dataclass_fields__", ()) or ():
            walk(getattr(value, name), depth + 1)
        for item in vars(value).values() if hasattr(value, "__dict__") else ():
            walk(item, depth + 1)

    walk(payload)
    assert forbidden.isdisjoint(types), sorted(types & forbidden)
    assert types <= {"InferenceRequest", "str", "int", "tuple", "ndarray"}, sorted(types)


def test_the_request_arrays_do_not_alias_engine_memory(schedule, bank, owner, greedy_ref):
    captured, capture_class = one_request(schedule, bank, greedy_ref)
    policy = capture_class(greedy_ref, LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY)
    play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})

    payload = captured["payload"]
    source = captured["input"]
    assert payload.observation.base is None
    assert payload.legal_action_mask.base is None
    assert payload.observation is not source.observation
    assert payload.legal_action_mask is not source.legal_action_mask
    # Same values, different memory.
    assert np.array_equal(payload.observation, source.observation)
    assert np.array_equal(payload.legal_action_mask, source.legal_action_mask)
    assert payload.observation.flags.writeable


def test_the_pickled_request_names_no_privileged_class(schedule, bank, owner, greedy_ref):
    """The transport's own view: what actually crosses the process boundary."""
    captured, capture_class = one_request(schedule, bank, greedy_ref)
    policy = capture_class(greedy_ref, LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY)
    play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})

    blob = pickle.dumps(captured["payload"])
    for name in (b"GameState", b"PieceRecord", b"ReplayRecord", b"stratego.engine.state"):
        assert name not in blob
    restored = pickle.loads(blob)
    assert restored.digest() == captured["payload"].digest()


def test_the_remote_policy_declares_only_two_products():
    requirements = RemoteNeuralPolicy.requirements
    assert requirements.observation is True
    assert requirements.legal_action_mask is True
    assert requirements.public_view is False
    assert requirements.public_events is False
    assert requirements.public_setup is False


def test_the_remote_policy_holds_no_weights(owner, greedy_ref):
    policy = RemoteNeuralPolicy(greedy_ref, LocalInferenceChannel(owner))
    assert policy.describe()["holds_model_weights"] is False
    assert not any(isinstance(value, torch.nn.Module) for value in vars(policy).values())


# ---------------------------------------------------------------------------
# 3. The decision path is the serial adapter's decision path
# ---------------------------------------------------------------------------


def test_the_remote_path_reproduces_the_in_process_adapter(schedule, bank, owner, checkpoint, greedy_ref):
    """The strongest correctness statement available: the parallel path is not a
    second implementation that agrees, it is the same rules reached remotely."""

    class DirectGreedy(NeuralCheckpointPolicy):
        policy_id = greedy_ref.policy_id
        policy_version = greedy_ref.policy_version
        decision_mode = DECISION_MODE_GREEDY

    direct = DirectGreedy.from_checkpoint(checkpoint, device="cpu")
    serial = run_schedule(schedule, bank, policies={greedy_ref.token: direct}, worker_count=1)
    remote = run_neural_schedule(schedule, bank, owner, worker_count=2)
    assert compare_results(serial.results, remote.results) == []
    assert serial.results_digest == remote.results_digest


def test_the_per_decision_seed_survives_the_round_trip(schedule, bank, owner, greedy_ref):
    seen = []

    class Recording(RemoteNeuralPolicy):
        def decide(self, request):
            result = super().decide(request)
            seen.append((request.ply, request.decision_seed, result.decision_seed))
            return result

    policy = Recording(greedy_ref, LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY)
    play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})

    from stratego.evaluation.policy import derive_decision_seed

    expected_seed = schedule[0].policy_seed_for(schedule[0].candidate_color)
    assert seen
    for ply, requested, returned in seen:
        assert requested == derive_decision_seed(expected_seed, ply)
        assert returned == requested


def test_a_crossed_answer_is_refused(owner, greedy_ref, schedule, bank):
    """A response for the wrong request must never become a move."""

    class Crossing(LocalInferenceChannel):
        def infer(self, request):
            answer = super().infer(request)
            return InferenceResponse(
                request_id="m-not-this-one#0",
                decision_seed=answer.decision_seed,
                absolute_action_id=answer.absolute_action_id,
                model_action_id=answer.model_action_id,
            )

    policy = RemoteNeuralPolicy(greedy_ref, Crossing(owner), decision_mode=DECISION_MODE_GREEDY)
    with pytest.raises(PolicyFailure, match="while this worker asked about") as failure:
        play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})
    assert isinstance(failure.value.__cause__, RemoteInferenceError)
    assert failure.value.category == ERROR_POLICY_EXCEPTION


def test_an_answer_under_the_wrong_seed_is_refused(owner, greedy_ref, schedule, bank):
    class Reseeding(LocalInferenceChannel):
        def infer(self, request):
            answer = super().infer(request)
            return InferenceResponse(
                request_id=answer.request_id,
                decision_seed=answer.decision_seed + 1,
                absolute_action_id=answer.absolute_action_id,
                model_action_id=answer.model_action_id,
            )

    policy = RemoteNeuralPolicy(greedy_ref, Reseeding(owner), decision_mode=DECISION_MODE_GREEDY)
    with pytest.raises(PolicyFailure, match="came back under seed") as failure:
        play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})
    assert isinstance(failure.value.__cause__, RemoteInferenceError)


# ---------------------------------------------------------------------------
# 4. Deterministic batching, ordering and worker-count independence
# ---------------------------------------------------------------------------


def test_the_worker_count_does_not_change_a_single_game(runs):
    assert field_level_mismatches(runs, baseline="1") == []


def test_shuffling_the_schedule_input_changes_nothing(runs):
    assert compare_results(runs["3"].results, runs["3_shuffled"].results) == []
    assert runs["3_shuffled"].results_digest == runs["3"].results_digest
    assert runs["3_shuffled"].schedule_digest == runs["3"].schedule_digest


def test_the_sweep_has_one_results_digest_and_one_replay_digest_set(runs):
    digests = sweep_digests(runs)
    assert digests["distinct_results_digests"] == 1
    assert digests["distinct_replay_digest_sets"] == 1


@pytest.mark.parametrize(
    "field",
    ["match_id", "paired_unit_id", "red_setup", "blue_setup", "candidate_seed",
     "opponent_seed", "replay_digest", "winner", "terminal_reason", "plies"],
)
def test_the_named_gate_fields_are_identical_across_worker_counts(runs, field):
    baseline = {row.match_id: getattr(row, field) for row in runs["1"].results}
    for label, run in runs.items():
        assert {row.match_id: getattr(row, field) for row in run.results} == baseline, label


def test_absolute_action_histories_are_identical_across_worker_counts(runs):
    baseline = {row.match_id: row.action_history for row in runs["1"].results}
    for label, run in runs.items():
        assert {row.match_id: row.action_history for row in run.results} == baseline, label
        assert all(row.action_history for row in run.results), label


def test_the_chunk_count_does_not_change_results(schedule, bank, owner):
    baseline = run_neural_schedule(schedule, bank, owner, worker_count=2, chunks_per_worker=1)
    crowded = run_neural_schedule(schedule, bank, owner, worker_count=2, chunks_per_worker=8)
    assert compare_results(baseline.results, crowded.results) == []
    assert crowded.chunk_count > baseline.chunk_count


def test_every_run_plays_both_colours_and_both_opponents(runs, greedy_ref):
    for label, run in runs.items():
        assert {row.candidate_color for row in run.results} == {RED, BLUE}, label
        assert {row.opponent_policy_id for row in run.results} == set(OPPONENTS), label
        assert all(row.candidate_policy_id == greedy_ref.policy_id for row in run.results)


def test_the_default_batch_policy_serves_one_request_per_forward(runs):
    for label, run in runs.items():
        assert run.batch_policy == BATCH_POLICY_SINGLE, label
        assert run.inference["max_batch_size_seen"] == 1, label
        assert set(run.inference["batch_size_histogram"]) == {"1"}, label


def test_requests_are_dispatched_in_canonical_identity_order():
    """Ordering is by identity, never by arrival."""
    def make(match_id, ply):
        return InferenceRequest(
            request_id=InferenceRequest.request_id_for(match_id, ply),
            match_id=match_id,
            paired_unit_id="u-1",
            ply=ply,
            acting_player=RED,
            decision_seed=1,
            observation=np.zeros((127, 10, 10), dtype=np.float32),
            legal_actions=(0,),
            legal_action_mask=np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8),
        )

    arrival = [make("m-b", 2), make("m-a", 4), make("m-b", 0), make("m-a", 1)]
    ordered = [request.request_id for request in sorted(arrival, key=lambda r: r.sort_key)]
    assert ordered == ["m-a#1", "m-a#4", "m-b#0", "m-b#2"]
    assert ordered == [
        request.request_id for request in sorted(reversed(arrival), key=lambda r: r.sort_key)
    ]


def test_an_arrival_batched_owner_must_declare_a_batch_size(checkpoint):
    with pytest.raises(NeuralEvaluationError, match="contradicts it"):
        InferenceOwner(
            checkpoint, device="cpu", batch_policy=BATCH_POLICY_SINGLE, max_batch_size=8
        )
    with pytest.raises(NeuralEvaluationError, match="unknown batch policy"):
        InferenceOwner(checkpoint, device="cpu", batch_policy="whatever")


def test_arrival_batching_is_available_and_measured(checkpoint, schedule, bank):
    """The batched path exists as a performance instrument. Whether it agrees
    with the deterministic path is measured, not assumed -- so this test only
    requires that a batch larger than one actually happened."""
    batched = InferenceOwner(
        checkpoint,
        device="cpu",
        batch_policy=BATCH_POLICY_ARRIVAL,
        max_batch_size=4,
        name="batched",
    )
    try:
        run = run_neural_schedule(schedule, bank, batched, worker_count=4)
    finally:
        batched.close()
    assert run.batch_policy == BATCH_POLICY_ARRIVAL
    assert run.inference["max_batch_size_seen"] >= 2
    assert run.matches_run == len(schedule)


# ---------------------------------------------------------------------------
# 5. Seeded categorical mode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sampled_runs(checkpoint, bank, sampled_ref):
    matches = schedule_matches(
        build_paired_schedule(sampled_ref, policy_ref("basic_heuristic"), PAIR_IDS)
    )
    owner = InferenceOwner(
        checkpoint, device="cpu", decision_mode=DECISION_MODE_CATEGORICAL, name="sampled_owner"
    )
    try:
        yield matches, {
            "1": run_neural_schedule(matches, bank, owner, worker_count=1),
            "3": run_neural_schedule(matches, bank, owner, worker_count=3),
            "3_shuffled": run_neural_schedule(
                tuple(reversed(matches)), bank, owner, worker_count=3
            ),
        }, owner
    finally:
        owner.close()


def test_the_seeded_categorical_mode_reproduces_across_worker_counts(sampled_runs):
    _, runs, _ = sampled_runs
    assert field_level_mismatches(runs, baseline="1") == []
    assert sweep_digests(runs)["distinct_results_digests"] == 1


def test_the_seeded_categorical_owner_loads_once(sampled_runs):
    _, _, owner = sampled_runs
    assert owner.checkpoint_load_count == 1


def test_the_stochastic_path_is_not_secretly_the_greedy_path(sampled_runs, schedule, bank, owner, greedy_ref):
    """Guards against the sampled gate accidentally re-testing greedy."""
    matches, runs, _ = sampled_runs
    greedy = run_neural_schedule(
        schedule_matches(build_paired_schedule(greedy_ref, policy_ref("basic_heuristic"), PAIR_IDS)),
        bank,
        owner,
        worker_count=1,
    )
    sampled = runs["1"]
    assert sampled.results_digest != greedy.results_digest
    greedy_actions = {row.setup_pair_id: row.action_history for row in greedy.results}
    sampled_actions = {row.setup_pair_id: row.action_history for row in sampled.results}
    assert any(
        greedy_actions[pair_id] != sampled_actions[pair_id] for pair_id in greedy_actions
    )


def test_the_sampled_policy_is_marked_stochastic(owner, sampled_ref):
    policy = RemoteNeuralPolicy(
        sampled_ref, LocalInferenceChannel(owner), decision_mode=DECISION_MODE_CATEGORICAL
    )
    assert policy.stochastic is True
    assert RemoteNeuralPolicy(sampled_ref, LocalInferenceChannel(owner)).stochastic is False


# ---------------------------------------------------------------------------
# 6. Failure behaviour -- loud, and never a substituted move
# ---------------------------------------------------------------------------


def test_a_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises((CheckpointError, FileNotFoundError, OSError)):
        InferenceOwner(tmp_path / "absent.pt", device="cpu")


def test_a_contract_v1_checkpoint_is_refused(repository_root):
    """The shipped Phase 5 fixture is a `model_contract_v1` file."""
    legacy = repository_root / "checkpoints" / "integration_model_v1.pt"
    if not legacy.exists():  # pragma: no cover - artifact is in the repository
        pytest.skip("the Phase 5 v1 fixture is not present")
    with pytest.raises(CheckpointError):
        InferenceOwner(legacy, device="cpu")


def test_a_checkpoint_for_a_different_candidate_is_refused(checkpoint):
    from stratego.model.architecture_configs import candidate_config

    with pytest.raises(CheckpointError):
        InferenceOwner(
            checkpoint, device="cpu", expected_configuration=candidate_config("C3")
        )


def test_a_corrupted_checkpoint_is_refused(checkpoint, tmp_path):
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(checkpoint.read_bytes()[: checkpoint.stat().st_size // 2])
    with pytest.raises(Exception) as failure:
        InferenceOwner(truncated, device="cpu")
    assert not isinstance(failure.value, AssertionError)


def test_a_malformed_request_is_refused_without_an_action(owner):
    good = InferenceRequest(
        request_id="m-x#0",
        match_id="m-x",
        paired_unit_id="u-x",
        ply=0,
        acting_player=RED,
        decision_seed=7,
        observation=np.zeros((127, 10, 10), dtype=np.float32),
        legal_actions=(101,),
        legal_action_mask=_mask([101]),
    )
    cases = {
        "shape": {"observation": np.zeros((3, 10, 10), dtype=np.float32)},
        "player": {"acting_player": 7},
        "mask": {"legal_action_mask": np.zeros(5, dtype=np.uint8)},
        "empty": {"legal_actions": ()},
        "nonfinite": {"observation": np.full((127, 10, 10), np.nan, dtype=np.float32)},
        "seed": {"decision_seed": -3},
    }
    for label, override in cases.items():
        from dataclasses import replace

        answer = owner.serve(replace(good, **override))
        assert isinstance(answer, InferenceFailure), label
        assert not hasattr(answer, "absolute_action_id"), label
    # A well-formed request still works afterwards: one bad request must not
    # poison the owner.
    assert isinstance(owner.serve(good), InferenceResponse)


def test_a_non_numpy_request_is_refused(owner):
    answer = owner.serve_batch(["not a request"])[0]
    assert isinstance(answer, InferenceFailure)
    assert "InferenceRequest" in answer.message


def test_a_non_finite_model_output_refuses_rather_than_choosing(owner, checkpoint):
    """The owner must not pick "the best finite logit"."""
    poisoned = InferenceOwner(checkpoint, device="cpu", name="poisoned")
    try:
        with torch.no_grad():
            poisoned.model.policy_source_bias.fill_(float("nan"))
        answer = poisoned.serve(
            InferenceRequest(
                request_id="m-nan#0",
                match_id="m-nan",
                paired_unit_id="u-nan",
                ply=0,
                acting_player=RED,
                decision_seed=1,
                observation=np.zeros((127, 10, 10), dtype=np.float32),
                legal_actions=(101, 202),
                legal_action_mask=_mask([101, 202]),
            )
        )
    finally:
        poisoned.close()
    assert isinstance(answer, InferenceFailure)
    assert "non-finite" in answer.message


def _fault_after(count: int):
    calls = {"n": 0}

    def explode(requests):
        calls["n"] += 1
        if calls["n"] > count:
            raise RuntimeError("the inference coordinator fell over")

    return explode


def test_an_owner_fault_aborts_a_serial_run_loudly(checkpoint, schedule, bank):
    """In-process, the classified policy failure is what escapes -- the run stops
    at the first bad decision rather than finishing with a guessed move."""
    faulty = InferenceOwner(checkpoint, device="cpu", name="faulty_serial")
    faulty.fault_hook = _fault_after(3)
    try:
        with pytest.raises(PolicyFailure, match="fell over") as failure:
            run_neural_schedule(schedule, bank, faulty, worker_count=1)
    finally:
        faulty.close()
    assert isinstance(failure.value.__cause__, RemoteInferenceError)
    assert failure.value.category == ERROR_POLICY_EXCEPTION
    assert faulty.aborted


def test_an_owner_fault_aborts_a_parallel_run_loudly(checkpoint, schedule, bank):
    """Across processes, the parent collects the workers' failures and refuses to
    return a partial result set."""
    faulty = InferenceOwner(checkpoint, device="cpu", name="faulty_parallel")
    faulty.fault_hook = _fault_after(3)
    try:
        with pytest.raises(NeuralEvaluationError) as failure:
            run_neural_schedule(schedule, bank, faulty, worker_count=2)
    finally:
        faulty.close()
    assert "game workers failed" in str(failure.value)
    assert "fell over" in str(failure.value)
    assert faulty.aborted


def test_an_owner_fault_is_never_answered_with_a_legal_move(checkpoint, schedule, bank, greedy_ref):
    """After a fault every further request is refused. No move is substituted."""
    faulty = InferenceOwner(checkpoint, device="cpu", name="faulty_quarantine")
    faulty.fault_hook = lambda requests: (_ for _ in ()).throw(RuntimeError("device lost"))
    try:
        policy = RemoteNeuralPolicy(
            greedy_ref, LocalInferenceChannel(faulty), decision_mode=DECISION_MODE_GREEDY
        )
        row = play_match(
            schedule[0],
            bank=bank,
            policies={greedy_ref.token: policy},
            on_policy_error=ON_POLICY_ERROR_QUARANTINE,
        )
    finally:
        faulty.close()
    assert row.candidate_result == RESULT_ERROR
    assert row.candidate_score is None
    assert row.policy_error_category == ERROR_POLICY_EXCEPTION
    assert row.terminal_reason == "policy_error"
    assert "device lost" in row.policy_error
    # The candidate never moved, so no action of its own is in the history.
    assert policy.decisions == 0


def test_a_refusal_raises_rather_than_returning_a_legal_action(owner, greedy_ref, schedule, bank):
    class Refusing(LocalInferenceChannel):
        def infer(self, request):
            return InferenceFailure(request.request_id, "Injected", "refused on purpose")

    policy = RemoteNeuralPolicy(greedy_ref, Refusing(owner), decision_mode=DECISION_MODE_GREEDY)
    with pytest.raises(PolicyFailure, match="refused on purpose") as failure:
        play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})
    assert isinstance(failure.value.__cause__, RemoteInferenceError)
    assert policy.decisions == 0


def test_a_timeout_is_reported_rather_than_guessed(owner, greedy_ref, schedule, bank):
    """A silent owner must stop the match, not produce a plausible move."""
    import queue as queue_module

    class Silent:
        transport = "silent"

        def infer(self, request):
            raise RemoteInferenceError(
                f"the inference owner did not answer request {request.request_id!r} "
                "within 0s; refusing to continue this match"
            )

        def stats(self):
            return {}

    policy = RemoteNeuralPolicy(greedy_ref, Silent(), decision_mode=DECISION_MODE_GREEDY)
    with pytest.raises(PolicyFailure, match="did not answer request") as failure:
        play_match(schedule[0], bank=bank, policies={greedy_ref.token: policy})
    assert isinstance(failure.value.__cause__, RemoteInferenceError)
    assert policy.decisions == 0
    assert queue_module.Empty is not None


def test_a_normalized_action_outside_the_legal_set_is_refused(owner, monkeypatch):
    """The conversion boundary refuses rather than submitting an illegal move."""
    from stratego.model import policy_adapter as adapter

    monkeypatch.setattr(adapter, "model_action_to_absolute", lambda action, player: 9999)
    answer = owner.serve(
        InferenceRequest(
            request_id="m-frame#0",
            match_id="m-frame",
            paired_unit_id="u-frame",
            ply=0,
            acting_player=RED,
            decision_seed=3,
            observation=np.zeros((127, 10, 10), dtype=np.float32),
            legal_actions=(101, 202),
            legal_action_mask=_mask([101, 202]),
        )
    )
    assert isinstance(answer, InferenceFailure)
    assert "did not declare legal" in answer.message


def test_a_schedule_that_does_not_name_the_served_policy_is_refused(checkpoint, bank, greedy_ref):
    matches = schedule_matches(
        build_paired_schedule(greedy_ref, policy_ref("basic_heuristic"), PAIR_IDS)
    )
    instance = InferenceOwner(checkpoint, device="cpu", name="mismatched")
    try:
        with pytest.raises(NeuralEvaluationError, match="do not name"):
            run_neural_schedule(
                matches, bank, instance, worker_count=1, policy_ref=policy_ref("random_legal")
            )
    finally:
        instance.close()


def test_a_worker_that_dies_without_reporting_fails_the_run():
    """The service loop must not wait forever for a worker that will never speak.

    Exercised directly rather than by crashing a real process: what needs
    proving is that a dead-but-still-active worker eventually raises, and that
    the grace period tolerates a message still in the queue's feeder thread.
    """
    from stratego.evaluation.neural_worker import WORKER_DEATH_GRACE_SECONDS, _check_liveness

    class DeadProcess:
        exitcode = -9

        def is_alive(self):
            return False

    class LiveProcess:
        exitcode = None

        def is_alive(self):
            return True

    processes = [DeadProcess()]
    dead_since: dict = {}
    # First sighting is inside the grace period and must not raise.
    _check_liveness(processes, {0}, dead_since)
    assert 0 in dead_since

    dead_since[0] -= WORKER_DEATH_GRACE_SECONDS + 1
    with pytest.raises(NeuralEvaluationError, match="without returning its results"):
        _check_liveness(processes, {0}, dead_since)

    # A live worker is never accused, and a worker that reports back is cleared.
    _check_liveness([LiveProcess()], {0}, {})
    _check_liveness(processes, set(), {})


def test_an_empty_or_mixed_schedule_is_refused(bank, owner, schedule):
    with pytest.raises(NeuralEvaluationError, match="empty schedule"):
        run_neural_schedule((), bank, owner)
    with pytest.raises(NeuralEvaluationError, match="worker_count must be"):
        run_neural_schedule(schedule, bank, owner, worker_count=0)


# ---------------------------------------------------------------------------
# 7. The same topology on the real device
# ---------------------------------------------------------------------------
#
# The tests above run the owner on the CPU because process spawning, not the
# forward pass, is what they cost. Metal is where the acceptance run happens, so
# the two device-specific claims are checked here directly: that a single-request
# forward is bit-stable on Metal, and that the worker count still cannot reach a
# game when the model lives on the GPU.

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal is not available on this machine"
)


@requires_mps
def test_a_single_request_forward_is_bit_stable_on_metal(checkpoint):
    """Repeating one request must give the identical logit row every time.

    This is the numerical assumption the `single_request` batch policy rests on:
    the model input is a pure function of the request, so the output must be too.
    """
    metal = InferenceOwner(checkpoint, device="mps", name="metal_stability")
    request = InferenceRequest(
        request_id="m-metal#0",
        match_id="m-metal",
        paired_unit_id="u-metal",
        ply=0,
        acting_player=BLUE,
        decision_seed=11,
        observation=np.linspace(0, 1, 127 * 100, dtype=np.float32).reshape(127, 10, 10),
        legal_actions=(101, 202, 303),
        legal_action_mask=_mask([101, 202, 303]),
    )
    try:
        answers = [metal.serve(request) for _ in range(16)]
    finally:
        metal.close()
    assert all(isinstance(answer, InferenceResponse) for answer in answers)
    logits = {answer.diagnostics["selected_logit"] for answer in answers}
    assert len(logits) == 1, logits
    assert len({answer.absolute_action_id for answer in answers}) == 1


@requires_mps
def test_the_batch_invariance_probe_answers_the_question_it_claims_to(checkpoint):
    """The probe must compare like with like: same requests, same owner, raw rows."""
    metal = InferenceOwner(checkpoint, device="mps", name="metal_probe")
    requests = [
        InferenceRequest(
            request_id=f"m-probe#{index}",
            match_id="m-probe",
            paired_unit_id="u-probe",
            ply=index,
            acting_player=RED if index % 2 == 0 else BLUE,
            decision_seed=index + 1,
            observation=np.linspace(index, index + 1, 127 * 100, dtype=np.float32).reshape(
                127, 10, 10
            ),
            legal_actions=(101, 202, 303, 404),
            legal_action_mask=_mask([101, 202, 303, 404]),
        )
        for index in range(4)
    ]
    try:
        alone = [metal.probe_policy_logits([request])[0] for request in requests]
        together = metal.probe_policy_logits(requests)
    finally:
        metal.close()
    assert len(together) == 4
    assert all(row.shape == (ACTION_SPACE_SIZE,) for row in together)
    assert all(row.device.type == "cpu" and row.dtype is torch.float32 for row in together)
    # Whether the rows agree is the measurement; that the probe returns
    # comparable objects is what this test pins down.
    assert all(alone[index].shape == together[index].shape for index in range(4))


@requires_mps
def test_the_probe_refuses_a_malformed_request(checkpoint):
    metal = InferenceOwner(checkpoint, device="mps", name="metal_probe_guard")
    try:
        with pytest.raises(NeuralEvaluationError, match="observation of shape"):
            metal.probe_policy_logits(
                [
                    InferenceRequest(
                        request_id="m-bad#0",
                        match_id="m-bad",
                        paired_unit_id="u-bad",
                        ply=0,
                        acting_player=RED,
                        decision_seed=1,
                        observation=np.zeros((3, 10, 10), dtype=np.float32),
                        legal_actions=(101,),
                        legal_action_mask=_mask([101]),
                    )
                ]
            )
    finally:
        metal.close()


@requires_mps
def test_the_worker_count_does_not_change_a_game_on_metal(checkpoint, schedule, bank):
    metal = InferenceOwner(checkpoint, device="mps", name="metal_sweep")
    try:
        metal_runs = {
            "1": run_neural_schedule(schedule, bank, metal, worker_count=1),
            "4": run_neural_schedule(schedule, bank, metal, worker_count=4),
            "4_shuffled": run_neural_schedule(
                tuple(reversed(schedule)), bank, metal, worker_count=4
            ),
        }
        assert metal.checkpoint_load_count == 1
    finally:
        metal.close()
    assert field_level_mismatches(metal_runs, baseline="1") == []
    assert sweep_digests(metal_runs)["distinct_results_digests"] == 1
    assert sweep_digests(metal_runs)["distinct_replay_digest_sets"] == 1
    for label, run in metal_runs.items():
        assert run.policy_errors == 0, label
        assert run.illegal_policy_actions == 0, label


# ---------------------------------------------------------------------------
# 8. Phase 4 identities are untouched
# ---------------------------------------------------------------------------


def test_the_neural_policies_are_not_in_the_phase_4_catalogue(greedy_ref, sampled_ref):
    assert greedy_ref.policy_id not in ALL_POLICY_IDS
    assert sampled_ref.policy_id not in ALL_POLICY_IDS
    assert len(ALL_POLICY_IDS) == 10


def test_the_policy_identity_names_the_candidate_mode_and_precision():
    assert neural_policy_ref("C1").token == "phase6_c1_greedy@0.2.0+float32"
    assert (
        neural_policy_ref("C1", decision_mode=DECISION_MODE_CATEGORICAL, dtype_name="float16").token
        == "phase6_c1_sampled@0.2.0+float16"
    )
    with pytest.raises(NeuralEvaluationError, match="unknown decision mode"):
        neural_policy_ref("C1", decision_mode="argmax")


def test_the_result_rows_name_the_neural_policy(runs, greedy_ref):
    for row in runs["1"].results:
        assert row.candidate_policy_id == greedy_ref.policy_id
        assert row.candidate_policy_version == greedy_ref.policy_version
        assert row.pairing_mode == "color_swap_same_board"
        assert row.setup_bank_version.startswith("evaluation_setup_bank")


def test_no_run_reports_an_illegal_action_or_a_policy_failure(runs):
    for label, run in runs.items():
        assert run.policy_errors == 0, label
        assert run.illegal_policy_actions == 0, label
        assert run.inference["failures_returned"] == 0, label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask(actions):
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    mask[np.asarray(actions, dtype=np.int64)] = 1
    return mask


@pytest.fixture(scope="module")
def repository_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
