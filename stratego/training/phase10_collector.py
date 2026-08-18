"""Phase 10 Agent 2: playing the frozen 16,384-game setup-outcome corpus.

Specification sources:

- `02_AGENT_2_SETUP_OUTCOME_CORPUS.md` (whole document)
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Controlled setup-outcome
  corpus")
- `stratego/training/phase10_schedule.py` — the Agent 1 freeze this executes

What this module does and does not decide
-----------------------------------------
Nothing here chooses anything. The schedule is Agent 1's; the setups are the
frozen Phase 7 sampler's; the move policy is the accepted Phase 9 checkpoint
played greedily. This module's whole job is to turn 16,384 already-determined
logical games into 16,384 digest-checked outcome records, and then to prove —
by replaying them — that it did.

The Phase 9 weights are read, never written
-------------------------------------------
`checkpoints/phase9/selfplay_c1_v1.pt` is a `phase9_checkpoint_v1` payload,
which the evaluation loader does not accept. Phase 9 Agent 8 already solved
this and its solution is reused verbatim: export the payload's weights to the
frozen evaluation format in a scratch location, prove the export is
*bitwise* identical to the source's tensors, and run inference against the
export. The accepted file is opened read-only and hashed before collection
and after sealing.

Why CPU rather than the MPS owner
---------------------------------
The one-MPS-owner topology in :mod:`stratego.evaluation.neural_worker` exists
because the machine has one GPU, and it makes every game worker wait on a
single serial stream of batch-1 forward passes. On this workload — 864k
parameters, one position per decision — a CPU forward is measurably faster
than an MPS forward, so a pure-CPU worker is both quicker and free of the
owner bottleneck: each worker owns its own model and nothing is shared.

That is an operational choice, not a contract change: the frozen behaviour is
`greedy / float32 / single_request / no search`, which names no device. It is
recorded in every run's diagnostics, and
:func:`device_agreement_probe` measures the claim it rests on — that the two
devices choose the same actions — rather than assuming it. CPU float32 with a
fixed thread count is additionally bit-exact run to run, which is what makes
the replay audit a real check instead of a tolerance.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

from ..setups.traits import TRAIT_SCHEMA_VERSION, compute_trait_vector
from .phase10_outcome_store import (
    OUTCOME_RECORD_VERSION,
    OutcomeReader,
    OutcomeStoreError,
    OutcomeWriter,
    build_stored_record,
    canonical_json,
    next_segment,
    reconcile_corpus,
)
from .phase10_schedule import (
    CORPUS_MOVE_BEHAVIOR,
    CORPUS_SAMPLER_PROFILE,
    CORPUS_SPLIT,
    CORPUS_VERSION,
    GAMES_PER_ORDERED_PAIR,
    ORDERED_FAMILY_PAIRS,
    RESULT_TARGETS,
    TOTAL_CORPUS_GAMES,
    enumerate_schedule,
    ordered_family_pairs,
    rebuild_game,
    resolve_side,
)

#: The move-policy identity every corpus record is played under. Distinct from
#: any Phase 9 evaluation identity because these are Phase 10 corpus games, and
#: a stored row must never be mistakable for a Phase 9 gate row.
CORPUS_MOVE_POLICY_ID = "phase10_corpus_move_v1"

#: The evaluation-suite version corpus matches carry.
CORPUS_SUITE_VERSION = "phase10_setup_outcome_corpus_v1"

#: The collection device. Recorded, measured, and not part of any identity.
CORPUS_DEVICE = "cpu"
CORPUS_TORCH_THREADS = 1

#: Result tokens, from the Red perspective.
RESULT_RED_WIN = "red_win"
RESULT_DRAW = "draw"
RESULT_RED_LOSS = "red_loss"


class Phase10CollectorError(RuntimeError):
    """Raised when a Phase 10 corpus collection condition is violated."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def corpus_policy_ref():
    """The `PolicyRef` both sides of every corpus game play under."""
    from ..evaluation.neural_worker import neural_policy_ref
    from ..model.policy_adapter import DECISION_MODE_GREEDY

    return neural_policy_ref(
        CORPUS_MOVE_POLICY_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name="float32"
    )


def file_sha256(path: "str | Path", *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def export_evaluation_weights(source: "str | Path", export_path: "str | Path") -> dict:
    """Export a `phase9_checkpoint_v1` file to the frozen evaluation format.

    The accepted Phase 9 Agent 8 procedure, unchanged: the source is opened
    read-only, and the export is refused unless every tensor round-trips
    bitwise. The returned identity is what every worker then checks its own
    loaded model against.
    """
    import torch

    from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ..model.checkpoint import load_checkpoint, save_checkpoint
    from .phase9_behavior import state_dict_digest
    from .phase9_checkpoint import model_from_payload, read_phase9_payload

    source = Path(source)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(source)

    payload = read_phase9_payload(source)
    model = model_from_payload(payload)
    state_digest = state_dict_digest(model)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    save_checkpoint(model, export_path)
    reloaded, _metadata = load_checkpoint(
        export_path,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    source_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    bitwise = set(source_state) == set(reloaded_state) and all(
        torch.equal(source_state[name], reloaded_state[name]) for name in source_state
    )
    if not bitwise:
        raise Phase10CollectorError(
            f"the evaluation export of {source} changed the weights; collection is BLOCKED"
        )
    if state_dict_digest(reloaded) != state_digest:
        raise Phase10CollectorError(
            f"the evaluation export of {source} changed the model-state digest"
        )
    if file_sha256(source) != source_sha:
        raise Phase10CollectorError(f"{source} changed while it was being exported")
    del model, reloaded, payload
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "export_path": str(export_path),
        "export_sha256": file_sha256(export_path),
        "model_state_digest": state_digest,
        "parameters": int(parameters),
        "bitwise_identical": True,
    }


def load_corpus_owner(export_path: "str | Path", *, device: str = CORPUS_DEVICE, name: str = "phase10_corpus"):
    """One long-lived inference owner holding the exported Phase 9 weights."""
    from ..evaluation.neural_worker import InferenceOwner
    from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ..model.policy_adapter import DECISION_MODE_GREEDY

    return InferenceOwner(
        Path(export_path),
        decision_mode=DECISION_MODE_GREEDY,
        device=device,
        dtype="float32",
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name=name,
    )


def owner_state_digest(owner) -> str:
    """The model-state digest of the weights an owner actually loaded."""
    from .phase9_behavior import state_dict_digest

    return state_dict_digest(owner.model)


def corpus_policy(owner):
    """The policy object both sides of a corpus game share."""
    from ..evaluation.neural_worker import LocalInferenceChannel, RemoteNeuralPolicy
    from ..model.policy_adapter import DECISION_MODE_GREEDY

    return RemoteNeuralPolicy(
        corpus_policy_ref(), LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY
    )


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

#: The frozen schedule index of every game id, so a `MatchSpec` identity is a
#: pure function of the logical game and never of arrival order.
def _schedule_index() -> dict:
    return {game.game_id: index for index, game in enumerate(enumerate_schedule())}


_SCHEDULE_INDEX: "dict | None" = None


def schedule_index(game_id: str) -> int:
    global _SCHEDULE_INDEX
    if _SCHEDULE_INDEX is None:
        _SCHEDULE_INDEX = _schedule_index()
    try:
        return _SCHEDULE_INDEX[game_id]
    except KeyError as error:
        raise Phase10CollectorError(f"{game_id} is not a scheduled corpus game") from error


def corpus_match_spec(game_id: str):
    """The fully determined `MatchSpec` of one corpus game.

    Both sides name the same policy reference, which is the literal statement
    that this is self-play under one frozen checkpoint. The colour assignment
    is fixed — the schedule already orders the family pair, so a colour swap
    would double-count `(red family, blue family)` as its own mirror.
    """
    from ..engine.constants import RED
    from ..evaluation.match_spec import MatchSpec

    game = rebuild_game(game_id)
    ref = corpus_policy_ref()
    return MatchSpec(
        candidate=ref,
        opponent=ref,
        setup_pair_id=schedule_index(game_id),
        candidate_color=RED,
        root_seed=game.match_seed,
        suite_version=CORPUS_SUITE_VERSION,
        setup_bank_version=CORPUS_VERSION,
    )


def resolve_game_setups(game_id: str, index=None) -> dict:
    """Both sides of one corpus game, rebuilt from the game id alone."""
    red, red_attempt, red_seed = resolve_side(game_id, "red", index=index)
    blue, blue_attempt, blue_seed = resolve_side(game_id, "blue", index=index)
    return {
        "red": {"sampled": red, "attempt": red_attempt, "seed": red_seed},
        "blue": {"sampled": blue, "attempt": blue_attempt, "seed": blue_seed},
    }


def play_corpus_game(game_id: str, policy, index=None, *, sides=None):
    """`(MatchResult, sides)` for one corpus game."""
    from ..engine.constants import BLUE, RED
    from ..evaluation.match_runner import ON_POLICY_ERROR_RAISE, play_match

    resolved = resolve_game_setups(game_id, index) if sides is None else sides
    spec = corpus_match_spec(game_id)
    result = play_match(
        spec,
        setups=(
            resolved["red"]["sampled"].oriented(RED),
            resolved["blue"]["sampled"].oriented(BLUE),
        ),
        policies={spec.candidate.token: policy},
        record_actions=False,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    return result, resolved


def result_token(result) -> str:
    """The Red-perspective outcome token of a played match."""
    from ..engine.constants import BLUE, RED

    if result.draw or result.winner is None:
        return RESULT_DRAW
    if result.winner == RED:
        return RESULT_RED_WIN
    if result.winner == BLUE:
        return RESULT_RED_LOSS
    raise Phase10CollectorError(f"unknown winner {result.winner!r}")


def trait_identity(sampled, index) -> dict:
    """The trait-vector identity of one side: its base's and its descendant's.

    The utility contract's `x(s)` is the *base*'s frozen 35-field vector, so
    that digest is what Agent 3 will need; the descendant's is recorded beside
    it so a record names the arrangement that was actually played. Both are
    structural descriptors of an arrangement — never a score, a strength
    signal or an outcome.
    """
    base = index.base(sampled.base_setup_id)
    return {
        "trait_schema_version": TRAIT_SCHEMA_VERSION,
        "base_trait_digest": hashlib.sha256(canonical_json(base.trait_vector).encode()).hexdigest(),
        "final_trait_digest": hashlib.sha256(
            canonical_json(compute_trait_vector(sampled.canonical)).encode()
        ).hexdigest(),
    }


def build_record(
    game_id: str,
    result,
    sides: dict,
    *,
    index,
    identity: dict,
) -> dict:
    """The two-halved stored record of one played corpus game."""
    from ..engine.constants import RED

    game = rebuild_game(game_id)
    token = result_token(result)
    red = sides["red"]
    blue = sides["blue"]
    if red["sampled"].family_id != game.red_family or blue["sampled"].family_id != game.blue_family:
        raise Phase10CollectorError(
            f"{game_id}: resolved families "
            f"({red['sampled'].family_id}, {blue['sampled'].family_id}) contradict the "
            f"schedule ({game.red_family}, {game.blue_family})"
        )
    for color, side in (("red", red), ("blue", blue)):
        if side["sampled"].split != CORPUS_SPLIT:
            raise Phase10CollectorError(
                f"{game_id} {color}: split {side['sampled'].split!r} is not "
                f"{CORPUS_SPLIT!r}; a held-out base in the corpus is BLOCKED"
            )

    setup_section = {
        "corpus_version": CORPUS_VERSION,
        "record_version": OUTCOME_RECORD_VERSION,
        "game_id": game_id,
        "red_family": game.red_family,
        "blue_family": game.blue_family,
        "ordinal": game.ordinal,
        "split": CORPUS_SPLIT,
        "match_seed": game.match_seed,
        "red_setup_draw_seed": red["seed"],
        "blue_setup_draw_seed": blue["seed"],
        "red_setup_attempt": red["attempt"],
        "blue_setup_attempt": blue["attempt"],
        "red_base_setup_id": red["sampled"].base_setup_id,
        "blue_base_setup_id": blue["sampled"].base_setup_id,
        "red_provenance": dict(red["sampled"].provenance),
        "blue_provenance": dict(blue["sampled"].provenance),
        "red_final_fingerprint": red["sampled"].provenance["final_setup_fingerprint"],
        "blue_final_fingerprint": blue["sampled"].provenance["final_setup_fingerprint"],
        "red_trait_identity": trait_identity(red["sampled"], index),
        "blue_trait_identity": trait_identity(blue["sampled"], index),
        "trait_schema_version": TRAIT_SCHEMA_VERSION,
        "library_content_digest": identity["library_content_digest"],
        "corpus_contract_digest": identity["corpus_contract_digest"],
        "outcome_schedule_digest": identity["outcome_schedule_digest"],
        "contract_bundle_digest": identity["contract_bundle_digest"],
    }
    outcome_section = {
        "result": token,
        "winner": None if token == RESULT_DRAW else ("red" if token == RESULT_RED_WIN else "blue"),
        "red_score": RESULT_TARGETS[token],
        "plies": int(result.plies),
        "decisions": int(result.decisions),
        "terminal_reason": result.terminal_reason,
        "move_policy_identity": corpus_policy_ref().token,
        "move_checkpoint_sha256": identity["phase9_checkpoint_sha256"],
        "move_model_state_digest": identity["phase9_model_state_digest"],
    }
    if result.candidate_color != RED:  # pragma: no cover - fixed by construction
        raise Phase10CollectorError(f"{game_id}: candidate colour is not Red")
    return build_stored_record(setup_section, outcome_section)


def corpus_identity(export: dict) -> dict:
    """The digests every record carries, computed once per process."""
    from ..setups.sampler import load_library_index
    from . import phase10_contract as contract
    from .phase10_schedule import corpus_contract_document, schedule_digest

    return {
        "library_content_digest": load_library_index().content_digest,
        "corpus_contract_digest": contract.document_digest(corpus_contract_document()),
        "outcome_schedule_digest": schedule_digest(),
        "contract_bundle_digest": contract.contract_bundle_digest(),
        "phase9_checkpoint_sha256": export["source_sha256"],
        "phase9_model_state_digest": export["model_state_digest"],
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def partition(game_ids, worker_count: int) -> list:
    """Round-robin assignment of games to workers.

    Which worker plays which game is an operational detail: the canonical
    corpus order is `sorted(game_id)`, so two runs partitioned differently
    converge to the same logical corpus. The tests prove that rather than
    assert it.
    """
    if worker_count < 1:
        raise Phase10CollectorError(f"worker_count must be at least 1, got {worker_count}")
    buckets: list = [[] for _ in range(worker_count)]
    for index, game_id in enumerate(game_ids):
        buckets[index % worker_count].append(game_id)
    return buckets


def collect_slice(
    root: "str | Path",
    game_ids,
    *,
    segment: int,
    worker_id: int,
    export_path: "str | Path",
    device: str = CORPUS_DEVICE,
    torch_threads: int = CORPUS_TORCH_THREADS,
    expected_state_digest: "str | None" = None,
    crash_hook=None,
    progress=None,
) -> dict:
    """Play and commit one worker's games into its own file set.

    Runs in whichever process calls it — the parent for `worker_count=1`, a
    spawned child otherwise — and is the only thing that appends to its file
    set.
    """
    import torch

    from ..setups.sampler import load_library_index

    torch.set_num_threads(int(torch_threads))
    game_ids = list(game_ids)
    started = time.perf_counter()

    if _WORKER_IDENTITY is None:
        raise Phase10CollectorError(
            "collect_slice needs a corpus identity; call set_identity first"
        )
    identity = dict(_WORKER_IDENTITY)

    owner = load_corpus_owner(export_path, device=device, name=f"phase10_w{worker_id:02d}")
    state_digest = owner_state_digest(owner)
    if expected_state_digest is not None and state_digest != expected_state_digest:
        raise Phase10CollectorError(
            f"worker {worker_id} loaded model-state {state_digest}, expected "
            f"{expected_state_digest}; the move-policy identity would be a lie"
        )
    if identity["phase9_model_state_digest"] != state_digest:
        raise Phase10CollectorError(
            f"worker {worker_id} would stamp model-state "
            f"{identity['phase9_model_state_digest']} onto records played by {state_digest}"
        )
    policy = corpus_policy(owner)
    index = load_library_index()

    writer = OutcomeWriter(root, segment=segment, worker_id=worker_id, crash_hook=crash_hook)
    plies = 0
    decisions = 0
    try:
        for position, game_id in enumerate(game_ids):
            result, sides = play_corpus_game(game_id, policy, index)
            record = build_record(game_id, result, sides, index=index, identity=identity)
            writer.write_record(record)
            plies += int(result.plies)
            decisions += int(result.decisions)
            if progress is not None:
                progress(worker_id, position + 1, len(game_ids))
    finally:
        stats = writer.close()
        owner.close()

    elapsed = time.perf_counter() - started
    stats.update(
        {
            "games": len(game_ids),
            "plies": plies,
            "decisions": decisions,
            "wall_clock_seconds": elapsed,
            "games_per_second": len(game_ids) / elapsed if elapsed else 0.0,
            "decisions_per_second": decisions / elapsed if elapsed else 0.0,
            "model_state_digest": state_digest,
            "device": device,
            "torch_threads": int(torch_threads),
            "policy_decisions": policy.decisions,
            "inference_failures": owner.stats().get("failures_returned", 0),
            "checkpoint_loads": owner.checkpoint_load_count,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )
    return stats


#: The identity block a worker process stamps onto every record it writes.
#: Set once per process before `collect_slice`; a child sets it from the
#: payload the parent sent, so no worker recomputes — and therefore cannot
#: disagree about — the digests the records carry.
_WORKER_IDENTITY: "dict | None" = None


def set_identity(identity: dict) -> None:
    global _WORKER_IDENTITY
    _WORKER_IDENTITY = dict(identity)


def _peak_rss_bytes() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux kilobytes.
    return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


def _worker_main(payload: dict, queue) -> None:
    """One spawned collection worker."""
    report = {"worker_id": payload["worker_id"], "status": "ok"}
    try:
        set_identity(payload["identity"])
        report["stats"] = collect_slice(
            payload["root"],
            payload["game_ids"],
            segment=payload["segment"],
            worker_id=payload["worker_id"],
            export_path=payload["export_path"],
            device=payload["device"],
            torch_threads=payload["torch_threads"],
            expected_state_digest=payload["expected_state_digest"],
        )
    except BaseException as error:  # noqa: BLE001 -- reported to the parent verbatim
        import traceback

        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    queue.put(report)


def _drain_reports(queue, processes, *, poll_seconds: float = 1.0) -> list:
    """Collect one report per worker, refusing to block on a dead one.

    A worker that dies without reporting — an OOM kill, a segfault — would
    otherwise hang the parent forever on `queue.get()`. Liveness is checked
    between polls so the run fails with the exit code instead.
    """
    import queue as queue_module

    reports: list = []
    outstanding = {process.pid: process for process in processes}
    while len(reports) < len(processes):
        try:
            reports.append(queue.get(timeout=poll_seconds))
            continue
        except queue_module.Empty:
            pass
        dead = [
            process
            for process in outstanding.values()
            if process.exitcode is not None and process.exitcode != 0
        ]
        if dead and len(reports) < len(processes):
            codes = ", ".join(f"pid {p.pid} exit {p.exitcode}" for p in dead)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise Phase10CollectorError(
                f"collection worker(s) died without reporting ({codes}); the committed "
                "games are intact and a resumed run will replay only the missing ones"
            )
    return reports


def collect_corpus(
    root: "str | Path",
    *,
    export: dict,
    game_ids=None,
    worker_count: int = 1,
    device: str = CORPUS_DEVICE,
    torch_threads: int = CORPUS_TORCH_THREADS,
    progress=None,
) -> dict:
    """Collect every missing scheduled game into `root`, then report.

    Reconciles first, so an interrupted attempt's uncommitted tail is gone
    before anything new is written, and only plays the games no commit record
    claims. Calling it twice in a row is therefore a no-op the second time.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    recovery = reconcile_corpus(root)
    committed = set(recovery["committed"])

    scheduled = [game.game_id for game in enumerate_schedule()]
    if game_ids is not None:
        requested = list(game_ids)
        unknown = [game_id for game_id in requested if game_id not in set(scheduled)]
        if unknown:
            raise Phase10CollectorError(f"{len(unknown)} requested ids are not scheduled: {unknown[:3]}")
        scheduled = requested
    missing = [game_id for game_id in scheduled if game_id not in committed]

    identity = corpus_identity(export)
    segment = next_segment(root)
    started = time.perf_counter()
    reports: list = []

    if missing:
        buckets = [bucket for bucket in partition(missing, worker_count) if bucket]
        if len(buckets) == 1 and worker_count == 1:
            set_identity(identity)
            reports.append(
                {
                    "worker_id": 0,
                    "status": "ok",
                    "stats": collect_slice(
                        root,
                        buckets[0],
                        segment=segment,
                        worker_id=0,
                        export_path=export["export_path"],
                        device=device,
                        torch_threads=torch_threads,
                        expected_state_digest=export["model_state_digest"],
                        progress=progress,
                    ),
                }
            )
        else:
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            processes = []
            for worker_id, bucket in enumerate(buckets):
                payload = {
                    "root": str(root),
                    "game_ids": bucket,
                    "segment": segment,
                    "worker_id": worker_id,
                    "export_path": str(export["export_path"]),
                    "device": device,
                    "torch_threads": torch_threads,
                    "expected_state_digest": export["model_state_digest"],
                    "identity": identity,
                }
                process = context.Process(target=_worker_main, args=(payload, queue), daemon=False)
                process.start()
                processes.append(process)
            reports.extend(_drain_reports(queue, processes))
            for process in processes:
                process.join()
            failed = [report for report in reports if report["status"] != "ok"]
            if failed:
                raise Phase10CollectorError(
                    f"{len(failed)} collection worker(s) failed; the first says: "
                    f"{failed[0].get('error')}\n{failed[0].get('traceback', '')}"
                )

    elapsed = time.perf_counter() - started
    reader = OutcomeReader(root)
    games = sum(report["stats"]["games"] for report in reports)
    plies = sum(report["stats"]["plies"] for report in reports)
    decisions = sum(report["stats"]["decisions"] for report in reports)
    return {
        "corpus_version": CORPUS_VERSION,
        "root": str(root),
        "segment": segment,
        "worker_count": len(reports),
        "recovery": {
            key: value for key, value in recovery.items() if key != "committed"
        },
        "already_committed": len(committed),
        "games_played": games,
        "plies_played": plies,
        "decisions_played": decisions,
        "committed_games": len(reader),
        "wall_clock_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed and games else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed and decisions else 0.0,
        "peak_worker_rss_bytes": max(
            (report["stats"].get("peak_rss_bytes", 0) for report in reports), default=0
        ),
        "inference_failures": sum(
            report["stats"].get("inference_failures", 0) for report in reports
        ),
        "checkpoint_loads": sum(report["stats"].get("checkpoint_loads", 0) for report in reports),
        "device": device,
        "torch_threads": int(torch_threads),
        "workers": [report["stats"] for report in reports],
        "identity": identity,
    }


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_corpus_balance(root: "str | Path") -> dict:
    """Recompute every structural property the instruction requires.

    Reads stored bytes only: it plays no game and loads no model, so it is
    exactly as valid run by a reviewer as it is run by the collector.
    """
    from ..setups.sampler import load_library_index

    reader = OutcomeReader(root)
    index = load_library_index()
    scheduled = {game.game_id: game for game in enumerate_schedule()}

    pair_counts: dict = {}
    commit_digests: set = set()
    duplicate_commits: list = []
    split_violations: list = []
    provenance_mismatches: list = []
    policy_mismatches: list = []
    unscheduled: list = []
    result_counts = {token: 0 for token in RESULT_TARGETS}
    terminal_reasons: dict = {}
    plies: list = []
    policy_identities: set = set()
    model_digests: set = set()
    checkpoint_digests: set = set()
    base_ids: set = set()
    fingerprints: set = set()

    for record in reader.iter_records():
        game_id = record["game_id"]
        game = scheduled.get(game_id)
        if game is None:
            unscheduled.append(game_id)
            continue
        pair = (record["red_family"], record["blue_family"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if record["commit_digest"] in commit_digests:
            duplicate_commits.append(game_id)
        commit_digests.add(record["commit_digest"])
        if record["split"] != CORPUS_SPLIT:
            split_violations.append(game_id)
        result_counts[record["result"]] = result_counts.get(record["result"], 0) + 1
        terminal_reasons[record["terminal_reason"]] = (
            terminal_reasons.get(record["terminal_reason"], 0) + 1
        )
        plies.append(int(record["plies"]))
        policy_identities.add(record["move_policy_identity"])
        model_digests.add(record["move_model_state_digest"])
        checkpoint_digests.add(record["move_checkpoint_sha256"])

        for color in ("red", "blue"):
            provenance = record[f"{color}_provenance"]
            base_ids.add(provenance["base_setup_id"])
            fingerprints.add(provenance["final_setup_fingerprint"])
            entry = index.base(provenance["base_setup_id"])
            if entry.split != CORPUS_SPLIT:
                split_violations.append(f"{game_id}:{color}")
            if provenance["primary_family_id"] != getattr(game, f"{color}_family"):
                provenance_mismatches.append(f"{game_id}:{color}: family")
            if record[f"{color}_base_setup_id"] != provenance["base_setup_id"]:
                provenance_mismatches.append(f"{game_id}:{color}: base id")
            if record[f"{color}_final_fingerprint"] != provenance["final_setup_fingerprint"]:
                provenance_mismatches.append(f"{game_id}:{color}: fingerprint")
            if record[f"{color}_setup_draw_seed"] != provenance["draw_seed"]:
                provenance_mismatches.append(f"{game_id}:{color}: draw seed")
            if provenance["sampler_profile"] != CORPUS_SAMPLER_PROFILE:
                provenance_mismatches.append(f"{game_id}:{color}: profile")
        if record["red_score"] != RESULT_TARGETS[record["result"]]:
            policy_mismatches.append(f"{game_id}: score")

    expected_policy = corpus_policy_ref().token
    if policy_identities - {expected_policy}:
        policy_mismatches.append(f"unexpected policy identities: {sorted(policy_identities)}")

    checks = {
        "total_games_exact": len(reader) == TOTAL_CORPUS_GAMES,
        "ordered_pairs_exact": len(pair_counts) == ORDERED_FAMILY_PAIRS,
        "ordered_pairs_complete": set(pair_counts) == set(ordered_family_pairs()),
        "games_per_pair_exact": all(
            count == GAMES_PER_ORDERED_PAIR for count in pair_counts.values()
        ),
        "duplicate_game_ids_zero": len(set(reader.game_ids)) == len(reader.game_ids),
        "duplicate_commit_identities_zero": not duplicate_commits
        and not reader.duplicate_committed_ids,
        "every_game_scheduled": not unscheduled,
        "train_split_violations_zero": not split_violations,
        "setup_provenance_mismatches_zero": not provenance_mismatches,
        "policy_identity_mismatches_zero": not policy_mismatches
        and len(model_digests) == 1
        and len(checkpoint_digests) == 1,
    }

    return {
        "corpus_version": CORPUS_VERSION,
        "committed_games": len(reader),
        "ordered_pair_count": len(pair_counts),
        "games_per_ordered_pair": sorted(set(pair_counts.values())),
        "distinct_base_setups_used": len(base_ids),
        "distinct_final_fingerprints": len(fingerprints),
        "result_counts": result_counts,
        "result_rates": {
            token: count / max(len(reader), 1) for token, count in result_counts.items()
        },
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "ply_summary": {
            "min": min(plies) if plies else 0,
            "max": max(plies) if plies else 0,
            "mean": sum(plies) / len(plies) if plies else 0.0,
            "total": sum(plies),
        },
        "move_policy_identities": sorted(policy_identities),
        "move_model_state_digests": sorted(model_digests),
        "move_checkpoint_sha256": sorted(checkpoint_digests),
        "unscheduled_game_ids": sorted(unscheduled)[:32],
        "split_violations": sorted(set(split_violations))[:32],
        "setup_provenance_mismatches": sorted(set(provenance_mismatches))[:32],
        "policy_identity_mismatches": sorted(set(policy_mismatches))[:32],
        "duplicate_commit_identities": sorted(set(duplicate_commits))[:32],
        "checks": checks,
        "all_pass": all(checks.values()),
        "pair_counts": {f"{red}|{blue}": count for (red, blue), count in sorted(pair_counts.items())},
    }


def audit_setup_reconstruction(root: "str | Path", *, sample_every: int = 1) -> dict:
    """Rebuild every sampled final setup from its stored provenance alone."""
    from ..setups.sampler import load_library_index, rebuild_from_provenance

    reader = OutcomeReader(root)
    index = load_library_index()
    mismatches: list = []
    checked = 0
    for position, game_id in enumerate(reader.game_ids):
        if position % max(int(sample_every), 1):
            continue
        record = reader.record(game_id)
        for color in ("red", "blue"):
            provenance = record[f"{color}_provenance"]
            rebuilt = rebuild_from_provenance(provenance, index)
            checked += 1
            if rebuilt.provenance["final_setup_fingerprint"] != record[f"{color}_final_fingerprint"]:
                mismatches.append(f"{game_id}:{color}")
            if rebuilt.base_setup_id != record[f"{color}_base_setup_id"]:
                mismatches.append(f"{game_id}:{color}: base")
    return {
        "sides_rebuilt": checked,
        "games_checked": checked // 2,
        "mismatches": sorted(set(mismatches))[:32],
        "all_pass": not mismatches,
    }


def replay_audit(
    root: "str | Path",
    *,
    export: dict,
    sample: int = 2048,
    device: str = CORPUS_DEVICE,
    torch_threads: int = CORPUS_TORCH_THREADS,
    worker_count: int = 1,
) -> dict:
    """Replay stored games end to end and require identical outcomes.

    The sample is a deterministic stride over the canonical order, so it is
    reproducible and spreads across every ordered family pair rather than
    clustering in whichever pairs a worker happened to finish first.
    """
    reader = OutcomeReader(root)
    total = len(reader)
    if total == 0:
        raise Phase10CollectorError("nothing to replay: the corpus is empty")
    count = min(int(sample), total)
    stride = max(total // count, 1)
    chosen = [reader.game_ids[position] for position in range(0, total, stride)][:count]
    stored = {game_id: reader.record(game_id) for game_id in chosen}

    started = time.perf_counter()
    if worker_count > 1:
        replays = _replay_across_processes(
            chosen, export=export, device=device, torch_threads=torch_threads,
            worker_count=worker_count,
        )
    else:
        replays = replay_slice(chosen, export=export, device=device, torch_threads=torch_threads)
    elapsed = time.perf_counter() - started

    mismatches: list = []
    families: set = set()
    pairs: set = set()
    for game_id in chosen:
        record = stored[game_id]
        observed = replays[game_id]
        families.add(record["red_family"])
        families.add(record["blue_family"])
        pairs.add((record["red_family"], record["blue_family"]))
        for field in ("result", "plies", "terminal_reason"):
            if observed[field] != record[field]:
                mismatches.append(
                    f"{game_id}: {field} {observed[field]!r} != stored {record[field]!r}"
                )
        if observed["red_final_fingerprint"] != record["red_final_fingerprint"]:
            mismatches.append(f"{game_id}: red fingerprint")
        if observed["blue_final_fingerprint"] != record["blue_final_fingerprint"]:
            mismatches.append(f"{game_id}: blue fingerprint")

    from ..setups.families import FAMILY_IDS

    checks = {
        "outcomes_identical": not mismatches,
        "sample_meets_minimum": count >= min(2048, total),
        "all_families_covered": set(families) == set(FAMILY_IDS),
    }
    return {
        "replayed_games": count,
        "corpus_games": total,
        "stride": stride,
        "distinct_ordered_pairs_covered": len(pairs),
        "families_covered": len(families),
        "mismatches": mismatches[:32],
        "wall_clock_seconds": elapsed,
        "games_per_second": count / elapsed if elapsed else 0.0,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def replay_slice(game_ids, *, export: dict, device: str = CORPUS_DEVICE, torch_threads: int = CORPUS_TORCH_THREADS) -> dict:
    """Replay games from their identity alone, returning primitive outcomes."""
    import torch

    from ..setups.sampler import load_library_index

    torch.set_num_threads(int(torch_threads))
    owner = load_corpus_owner(export["export_path"], device=device, name="phase10_replay")
    if owner_state_digest(owner) != export["model_state_digest"]:
        raise Phase10CollectorError("the replay owner loaded different weights")
    policy = corpus_policy(owner)
    index = load_library_index()
    observed: dict = {}
    try:
        for game_id in game_ids:
            result, sides = play_corpus_game(game_id, policy, index)
            observed[game_id] = {
                "result": result_token(result),
                "plies": int(result.plies),
                "terminal_reason": result.terminal_reason,
                "red_final_fingerprint": sides["red"]["sampled"].provenance["final_setup_fingerprint"],
                "blue_final_fingerprint": sides["blue"]["sampled"].provenance["final_setup_fingerprint"],
            }
    finally:
        owner.close()
    return observed


def _replay_worker_main(payload: dict, queue) -> None:
    report = {"worker_id": payload["worker_id"], "status": "ok"}
    try:
        report["observed"] = replay_slice(
            payload["game_ids"],
            export=payload["export"],
            device=payload["device"],
            torch_threads=payload["torch_threads"],
        )
    except BaseException as error:  # noqa: BLE001 -- reported to the parent verbatim
        import traceback

        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    queue.put(report)


def _replay_across_processes(game_ids, *, export: dict, device: str, torch_threads: int, worker_count: int) -> dict:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    buckets = [bucket for bucket in partition(list(game_ids), worker_count) if bucket]
    processes = []
    for worker_id, bucket in enumerate(buckets):
        payload = {
            "worker_id": worker_id,
            "game_ids": bucket,
            "export": export,
            "device": device,
            "torch_threads": torch_threads,
        }
        process = context.Process(target=_replay_worker_main, args=(payload, queue), daemon=False)
        process.start()
        processes.append(process)
    observed: dict = {}
    reports = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    failed = [report for report in reports if report["status"] != "ok"]
    if failed:
        raise Phase10CollectorError(
            f"{len(failed)} replay worker(s) failed; the first says: {failed[0].get('error')}\n"
            f"{failed[0].get('traceback', '')}"
        )
    for report in reports:
        observed.update(report["observed"])
    return observed


def wrong_checkpoint_negative_control(
    root: "str | Path",
    *,
    export: dict,
    wrong_export: dict,
    sample: int = 32,
    device: str = CORPUS_DEVICE,
) -> dict:
    """Require the verifier to *fail* when the wrong weights are used.

    A replay audit that passes no matter which checkpoint played is not an
    audit. This runs the same verifier against deliberately wrong weights and
    treats a clean pass as the failure.
    """
    reader = OutcomeReader(root)
    total = len(reader)
    count = min(int(sample), total)
    stride = max(total // count, 1)
    chosen = [reader.game_ids[position] for position in range(0, total, stride)][:count]
    stored = {game_id: reader.record(game_id) for game_id in chosen}

    identity_rejected = None
    try:
        replay_slice(chosen, export={**wrong_export, "model_state_digest": export["model_state_digest"]}, device=device)
        identity_rejected = False
    except Phase10CollectorError:
        identity_rejected = True

    observed = replay_slice(chosen, export=wrong_export, device=device)
    differing = [
        game_id
        for game_id in chosen
        if any(
            observed[game_id][field] != stored[game_id][field]
            for field in ("result", "plies", "terminal_reason")
        )
    ]
    checks = {
        "policy_identity_check_fires": bool(identity_rejected),
        "result_verifier_fires": bool(differing),
    }
    return {
        "sampled_games": count,
        "wrong_model_state_digest": wrong_export["model_state_digest"],
        "accepted_model_state_digest": export["model_state_digest"],
        "games_with_different_outcome": len(differing),
        "difference_rate": len(differing) / count if count else 0.0,
        "first_differences": differing[:8],
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def device_agreement_probe(
    game_ids,
    *,
    export: dict,
    devices=("cpu", "mps"),
    torch_threads: int = CORPUS_TORCH_THREADS,
) -> dict:
    """Do the available devices choose the same games? Measured, not assumed."""
    import torch

    available = []
    for device in devices:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        available.append(device)
    observed = {
        device: replay_slice(list(game_ids), export=export, device=device, torch_threads=torch_threads)
        for device in available
    }
    reference = available[0]
    disagreements: dict = {}
    for device in available[1:]:
        differing = [
            game_id
            for game_id in game_ids
            if any(
                observed[device][game_id][field] != observed[reference][game_id][field]
                for field in ("result", "plies", "terminal_reason")
            )
        ]
        disagreements[device] = differing
    return {
        "reference_device": reference,
        "devices": available,
        "games": len(list(game_ids)),
        "disagreements": {device: len(rows) for device, rows in disagreements.items()},
        "first_disagreements": {device: rows[:4] for device, rows in disagreements.items()},
        "all_agree": all(not rows for rows in disagreements.values()),
    }


def family_pair_rows(root: "str | Path") -> list:
    """One diagnostic row per ordered family pair, for the CSV artifact."""
    reader = OutcomeReader(root)
    rows: dict = {}
    for record in reader.iter_records():
        key = (record["red_family"], record["blue_family"])
        row = rows.setdefault(
            key,
            {
                "red_family": key[0],
                "blue_family": key[1],
                "games": 0,
                "red_wins": 0,
                "draws": 0,
                "red_losses": 0,
                "total_plies": 0,
                "distinct_red_bases": set(),
                "distinct_blue_bases": set(),
            },
        )
        row["games"] += 1
        row["total_plies"] += int(record["plies"])
        row["distinct_red_bases"].add(record["red_base_setup_id"])
        row["distinct_blue_bases"].add(record["blue_base_setup_id"])
        if record["result"] == RESULT_RED_WIN:
            row["red_wins"] += 1
        elif record["result"] == RESULT_DRAW:
            row["draws"] += 1
        else:
            row["red_losses"] += 1
    ordered = []
    for key in ordered_family_pairs():
        row = rows.get(key)
        if row is None:
            continue
        games = max(row["games"], 1)
        ordered.append(
            {
                "red_family": row["red_family"],
                "blue_family": row["blue_family"],
                "games": row["games"],
                "red_wins": row["red_wins"],
                "draws": row["draws"],
                "red_losses": row["red_losses"],
                "red_score": (row["red_wins"] + 0.5 * row["draws"]) / games,
                "mean_plies": row["total_plies"] / games,
                "distinct_red_bases": len(row["distinct_red_bases"]),
                "distinct_blue_bases": len(row["distinct_blue_bases"]),
            }
        )
    return ordered


def collection_contract_document(export: dict) -> dict:
    """What Agent 2 executed, as a machine-readable record."""
    return {
        "corpus_version": CORPUS_VERSION,
        "record_version": OUTCOME_RECORD_VERSION,
        "move_policy_identity": corpus_policy_ref().token,
        "move_behavior": dict(CORPUS_MOVE_BEHAVIOR),
        "device": CORPUS_DEVICE,
        "torch_threads": CORPUS_TORCH_THREADS,
        "device_is_identity": False,
        "split": CORPUS_SPLIT,
        "sampler_profile": CORPUS_SAMPLER_PROFILE,
        "total_games": TOTAL_CORPUS_GAMES,
        "phase9_checkpoint_sha256": export["source_sha256"],
        "phase9_model_state_digest": export["model_state_digest"],
        "phase9_parameters": export["parameters"],
        "phase9_export_bitwise_identical": export["bitwise_identical"],
        "optimizer_steps": 0,
        "learning_performed": "none: no gradient, no optimizer, no parameter write",
    }


__all__ = [
    "CORPUS_DEVICE",
    "CORPUS_MOVE_POLICY_ID",
    "CORPUS_SUITE_VERSION",
    "CORPUS_TORCH_THREADS",
    "Phase10CollectorError",
    "RESULT_DRAW",
    "RESULT_RED_LOSS",
    "RESULT_RED_WIN",
    "audit_corpus_balance",
    "audit_setup_reconstruction",
    "build_record",
    "collect_corpus",
    "collect_slice",
    "collection_contract_document",
    "corpus_identity",
    "corpus_match_spec",
    "corpus_policy",
    "corpus_policy_ref",
    "device_agreement_probe",
    "export_evaluation_weights",
    "family_pair_rows",
    "file_sha256",
    "load_corpus_owner",
    "owner_state_digest",
    "partition",
    "play_corpus_game",
    "replay_audit",
    "replay_slice",
    "resolve_game_setups",
    "result_token",
    "schedule_index",
    "set_identity",
    "trait_identity",
    "wrong_checkpoint_negative_control",
]
