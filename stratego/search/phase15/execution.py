"""Phase 15 Agent 2: running the packs, one process per core.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 2, 12, 13.

Why this exists at all
----------------------
A Phase 15 game is long — the pilot measured a mean of ~400 plies and a tail
past 1,600 — so one TINY search game costs about half a minute and the whole
Stage B pack costs hours in one process. Section 12 asks for a *compact
engineering pack*, not a small one, so the fix is throughput rather than
fewer boards: each worker loads its own copy of the frozen models once and
then plays whole games, which is embarrassingly parallel because a game
shares no state with any other game.

Determinism survives the parallelism
------------------------------------
Nothing here touches a seed. Every stream a game consumes is derived from its
board id, so a game's result does not depend on which worker played it, on
how many workers there were, or on the order they finished in. Results are
sorted back into pack order before anything reads them, and a serial run
(`workers=1`, the in-process path) produces the identical rows — which is
what makes the parallel path checkable rather than merely fast.

The process boundary
--------------------
Section 2 does not authorize process control, and nothing here does any: the
pool is this agent's own children, created and joined by this agent. No
signal is sent to anything it did not start, and no Phase 14 file is opened.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

from .contract import Phase15SearchError, parse_board_id

#: The execution identity a report records.
EXECUTION_VERSION = "phase15_pack_execution_v1"

#: Per-worker torch thread count. One process per core beats one process
#: with many threads for this workload (batch sizes are small and the
#: bottleneck is the Python game loop), and it keeps the workers from
#: fighting each other for the same cores.
WORKER_TORCH_THREADS = 1

_STATE: dict = {}


class Phase15ExecutionError(Phase15SearchError):
    """A pack could not be executed."""


@dataclass(frozen=True)
class Task:
    """One (arm, preset, board) unit of work."""

    arm_id: str
    preset_name: str
    board_id: str
    probe: bool = False
    time_cap: "float | None" = None
    keep_moves: bool = False

    @property
    def key(self) -> tuple:
        return (self.arm_id, self.preset_name, self.board_id)


def _worker_init(
    root: str, device: str, with_anchor: bool, *, configure_threads: bool = True
) -> None:
    """Load one worker's own copy of every frozen model, exactly once.

    `configure_threads` is false on the in-process serial path. A worker is a
    fresh process that exits when the pool closes, so pinning its thread count
    costs nothing; doing the same in the caller's own process would mutate
    global state that outlives this call. The thread count changes speed, never
    results — the engine consumes no randomness and its rollouts are
    deterministic greedy — so the serial path stays a true reference run.

    Nothing here touches `os.environ`. An earlier version set
    `OMP_NUM_THREADS`, which leaked out of the serial path and broke the
    accepted worker-pool tests two files away; torch's own thread setting is
    sufficient and is process-local.
    """
    import torch

    if configure_threads:
        torch.set_num_threads(WORKER_TORCH_THREADS)

    from .boards import Phase15MatchSetupSources
    from .loaders import load_all
    from .matchplay import build_owners

    models = load_all(root=root, device=device, with_anchor=with_anchor)
    _STATE["root"] = root
    _STATE["device"] = device
    _STATE["models"] = models
    _STATE["owners"] = build_owners(models, device=device)
    _STATE["sources"] = Phase15MatchSetupSources()
    _STATE["systems"] = {}
    _STATE["plans"] = {}


def _system(arm_id: str, preset_name: str):
    from .contract import pairing as pairing_of
    from .systems import build_engine

    key = (arm_id, preset_name)
    bundle = _STATE["systems"].get(key)
    if bundle is None:
        target = pairing_of(arm_id)
        bundle = build_engine(
            target,
            _STATE["models"],
            preset_name,
            production=target.kind != "diagnostic",
            device=_STATE["device"],
        )
        _STATE["systems"][key] = bundle
    return bundle


def _plan(board_identifier: str):
    from .boards import board_plan

    plan = _STATE["plans"].get(board_identifier)
    if plan is None:
        fields = parse_board_id(board_identifier)
        plan = board_plan(
            fields["opponent"],
            fields["setup_source"],
            fields["color"],
            fields["ordinal"],
            _STATE["sources"],
        )
        _STATE["plans"][board_identifier] = plan
    return plan


def run_task(task: Task) -> dict:
    """Play one board with one arm. Pure function of the task."""
    from .matchplay import SeatProbe, play_board
    from .systems import build_seat

    bundle = _system(task.arm_id, task.preset_name)
    plan = _plan(task.board_id)
    seat = build_seat(bundle, _STATE["owners"], time_cap=task.time_cap)
    probe = None
    if task.probe:
        reference = getattr(seat, "direct_policy", None) or getattr(seat, "policy", None)
        probe = SeatProbe(
            reference=reference,
            expects_hidden_truth=bundle.pairing.kind == "diagnostic",
        )
    started = time.perf_counter()
    record = play_board(
        plan,
        seat,
        _STATE["owners"],
        probe=probe,
        preset_id=task.preset_name,
        keep_moves=task.keep_moves,
    )
    row = record.row()
    row["preset_id"] = task.preset_name
    row["wall_seconds"] = round(time.perf_counter() - started, 4)
    if task.keep_moves:
        # The played move sequence, so two rungs' games can be compared move
        # for move up to the ply where they first part company.
        row["actions"] = [int(move["action_id"]) for move in record.moves]
    return {
        "row": row,
        "move_seconds": [round(value, 6) for value in record.move_seconds],
        "fallback_reasons": dict(getattr(seat, "fallbacks", {}) or {}),
        "probe": probe.summary() if probe is not None else None,
    }


def run_pack(
    tasks: "list[Task]",
    *,
    root: str = ".",
    device: str = "cpu",
    workers: int = 8,
    with_anchor: bool = True,
    progress=None,
    keep_moves: bool = False,
) -> "list[dict]":
    """Run every task, in pack order, over `workers` processes.

    `workers=1` runs in this process through the same code path, so a serial
    reference run needs no separate implementation.
    """
    tasks = list(tasks)
    if keep_moves:
        tasks = [
            task if task.keep_moves else replace(task, keep_moves=True)
            for task in tasks
        ]
    if not tasks:
        return []
    if int(workers) <= 1:
        _worker_init(str(root), device, with_anchor, configure_threads=False)
        results = []
        for index, task in enumerate(tasks):
            results.append(run_task(task))
            if progress is not None:
                progress(index + 1, len(tasks), results[-1])
        return results

    if with_anchor:
        # The anchor export is written once, here, before any worker exists.
        # Ten processes exporting to one path race on its temporary file.
        from .loaders import ensure_phase9_anchor

        ensure_phase9_anchor(root=str(root))

    results: "list[dict | None]" = [None] * len(tasks)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(str(root), device, with_anchor),
    ) as pool:
        futures = {pool.submit(run_task, task): index for index, task in enumerate(tasks)}
        for future in _as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, len(tasks), results[index])
    missing = [index for index, value in enumerate(results) if value is None]
    if missing:  # pragma: no cover - a future that neither returned nor raised
        raise Phase15ExecutionError(f"{len(missing)} tasks produced no result")
    return results


def _as_completed(futures):
    from concurrent.futures import as_completed

    return as_completed(futures)


__all__ = [
    "EXECUTION_VERSION",
    "Phase15ExecutionError",
    "Task",
    "run_pack",
    "run_task",
]
