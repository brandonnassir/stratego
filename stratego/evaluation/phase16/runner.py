"""Phase 16 Agent 1: the scoring runner.

`score_on_benchmark(mode_or_provider, preset, workers, subset=None)` scores

1. any Phase 15 production system by pairing id (`p24_direct`, `p24_b24`, …
   — the working player's machine modes), or
2. any object implementing the Phase 15 decision-seat interface, supplied as
   a factory so Agents 2/3/4 plug in without new glue:

   ```python
   {"factory": "package.module:build_seat",   # importable path
    "kwargs": {...},                           # JSON-serialisable options
    "arm_id": "agent2_sampled_tiny"}          # the reported arm id
   ```

   The factory is called once per worker as
   ``factory(models=<Phase15Models>, owners=<owner dict>, preset=<str>,
   device=<str>, **kwargs)`` and must return a seat: an object with
   ``arm_id``, ``pairing`` (a Phase 15 `Pairing`) and
   ``decide(state, legal, spec, plan) -> (action_id, record_dict)``.

Execution mirrors Phase 15's accepted pack executor: one process per worker,
each loading its own digest-checked copy of the frozen stack exactly once;
every stream a game consumes derives from the board id, so results are
independent of worker count and completion order. `workers=1` runs in-process
through the same code path.

Results are appended to a JSONL file as games finish, and a rerun of the
same pack **resumes**: rows already on disk are not replayed. That is what
makes an hours-long pack safe to run in a background shell.

The oracle is refused by name here as well: a diagnostic pairing cannot be
scored through this runner.
"""

from __future__ import annotations

import importlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ...search.phase15.contract import pairing as pairing_of
from .contract import (
    ADVERSARIAL_BASELINE_VERSION,
    BENCHMARK_VERSION,
    Phase16MeasurementError,
    QUICK_SUBSET_NAME,
    RUNNER_VERSION,
)

#: Per-worker torch thread count, as the accepted Phase 15 executor pins it.
WORKER_TORCH_THREADS = 1

_STATE: dict = {}


class Phase16RunnerError(Phase16MeasurementError):
    """A Phase 16 pack could not be executed."""


@dataclass(frozen=True)
class Task16:
    """One (seat, preset, board) unit of work. Pure data, so it pickles."""

    seat_spec: "str | tuple"
    preset_name: str
    board_id: str
    keep_moves: bool = False

    @property
    def arm_id(self) -> str:
        if isinstance(self.seat_spec, str):
            return self.seat_spec
        return self.seat_spec[2]

    @property
    def key(self) -> tuple:
        return (self.arm_id, self.preset_name, self.board_id)


def normalize_seat_spec(mode_or_provider) -> "str | tuple":
    """A pickleable seat specification, with the oracle refused by name."""
    if isinstance(mode_or_provider, str):
        target = pairing_of(mode_or_provider)  # refuses unknown names
        if target.kind == "diagnostic":
            raise Phase16RunnerError(
                f"{mode_or_provider!r} is an offline diagnostic and cannot be "
                "scored on the benchmark; production pairings only"
            )
        return target.pairing_id
    if isinstance(mode_or_provider, dict):
        factory = mode_or_provider.get("factory")
        if not isinstance(factory, str) or ":" not in factory:
            raise Phase16RunnerError(
                "a provider spec needs factory='package.module:callable'"
            )
        kwargs = mode_or_provider.get("kwargs") or {}
        arm_id = mode_or_provider.get("arm_id")
        if not arm_id:
            raise Phase16RunnerError("a provider spec needs an arm_id")
        return (factory, json.dumps(kwargs, sort_keys=True), str(arm_id))
    raise Phase16RunnerError(
        "mode_or_provider must be a Phase 15 production pairing id or a "
        "{'factory': 'module:callable', 'kwargs': {...}, 'arm_id': ...} spec"
    )


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def _worker_init(root: str, device: str, *, configure_threads: bool = True) -> None:
    """Load one worker's own digest-checked copy of the frozen stack."""
    import torch

    if configure_threads:
        torch.set_num_threads(WORKER_TORCH_THREADS)

    from ...search.phase15.boards import Phase15MatchSetupSources
    from ...search.phase15.loaders import load_all
    from ...search.phase15.matchplay import build_owners

    models = load_all(root=root, device=device, with_anchor=True)
    _STATE["root"] = root
    _STATE["device"] = device
    _STATE["models"] = models
    _STATE["owners"] = build_owners(models, device=device)
    _STATE["sources"] = Phase15MatchSetupSources()
    _STATE["library"] = None
    _STATE["seats"] = {}
    _STATE["plans"] = {}


def _resolve_plan(board_id: str):
    plan = _STATE["plans"].get(board_id)
    if plan is not None:
        return plan
    if board_id.startswith(BENCHMARK_VERSION + "|"):
        from .benchmark import benchmark_board_plan
        from .contract import parse_benchmark_board_id

        fields = parse_benchmark_board_id(board_id)
        plan = benchmark_board_plan(
            fields["opponent"],
            fields["setup_source"],
            fields["color"],
            fields["ordinal"],
            _STATE["sources"],
        )
    elif board_id.startswith(ADVERSARIAL_BASELINE_VERSION + "|"):
        from .adversarial import load_library
        from .baseline import baseline_board_plan
        from .contract import parse_adversarial_board_id

        if _STATE["library"] is None:
            _STATE["library"] = load_library(root=_STATE["root"])
        fields = parse_adversarial_board_id(board_id)
        plan = baseline_board_plan(
            fields["arm"], fields["pair_index"], _STATE["library"], _STATE["sources"]
        )
    else:
        raise Phase16RunnerError(f"unrecognised Phase 16 board id: {board_id!r}")
    if plan.board_id != board_id:
        raise Phase16RunnerError(
            f"rebuilt plan carries id {plan.board_id!r}, expected {board_id!r}"
        )
    _STATE["plans"][board_id] = plan
    return plan


def _resolve_seat(seat_spec, preset_name: str):
    key = (seat_spec, preset_name)
    seat = _STATE["seats"].get(key)
    if seat is not None:
        return seat
    if isinstance(seat_spec, str):
        from ...search.phase15.systems import build_engine, build_seat

        target = pairing_of(seat_spec)
        if target.kind == "diagnostic":  # defence in depth; normalize refuses first
            raise Phase16RunnerError(f"{seat_spec!r} is an offline diagnostic")
        bundle = build_engine(
            target, _STATE["models"], preset_name, production=True, device=_STATE["device"]
        )
        seat = build_seat(bundle, _STATE["owners"])
    else:
        factory_path, kwargs_json, arm_id = seat_spec
        module_name, _, attribute = factory_path.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute)
        seat = factory(
            models=_STATE["models"],
            owners=_STATE["owners"],
            preset=preset_name,
            device=_STATE["device"],
            **json.loads(kwargs_json),
        )
        for required in ("decide", "pairing", "arm_id"):
            if not hasattr(seat, required):
                raise Phase16RunnerError(
                    f"the seat from {factory_path} lacks {required!r}; the Phase 15 "
                    "decision-seat interface requires decide/pairing/arm_id"
                )
        provider = getattr(seat, "provider", None) or getattr(
            getattr(seat, "engine", None), "provider", None
        )
        if getattr(provider, "uses_hidden_truth", False):
            raise Phase16RunnerError(
                f"the seat from {factory_path} carries a hidden-truth provider; "
                "the benchmark scores production configurations only"
            )
        seat.arm_id = arm_id
    _STATE["seats"][key] = seat
    return seat


def run_task16(task: Task16) -> dict:
    """Play one board with one seat. Pure function of the task."""
    from ...search.phase15.matchplay import play_board

    seat = _resolve_seat(task.seat_spec, task.preset_name)
    plan = _resolve_plan(task.board_id)
    started = time.perf_counter()
    record = play_board(
        plan,
        seat,
        _STATE["owners"],
        probe=None,
        preset_id=task.preset_name,
        keep_moves=task.keep_moves,
    )
    row = record.row()
    row["arm_id"] = task.arm_id
    row["preset_id"] = task.preset_name
    row["wall_seconds"] = round(time.perf_counter() - started, 4)
    result = {
        "row": row,
        "move_seconds": [round(value, 6) for value in record.move_seconds],
        "fallback_reasons": dict(getattr(seat, "fallbacks", {}) or {}),
    }
    if task.keep_moves:
        result["actions"] = [int(move["action_id"]) for move in record.moves]
    return result


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def load_results(out_path: "Path | str | None") -> "dict[tuple, dict]":
    """Rows already on disk, keyed by (arm, preset, board)."""
    if out_path is None:
        return {}
    path = Path(out_path)
    if not path.is_file():
        return {}
    finished: dict[tuple, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            row = entry["row"]
            finished[(row["arm_id"], row["preset_id"], row["board_id"])] = entry
    return finished


def run_pack16(
    tasks: "list[Task16]",
    *,
    root: str = ".",
    device: str = "cpu",
    workers: int = 8,
    out_path: "Path | str | None" = None,
    progress=None,
) -> "list[dict]":
    """Run every task, resuming from `out_path`, in pack order."""
    tasks = list(tasks)
    finished = load_results(out_path)
    todo = [task for task in tasks if task.key not in finished]
    handle = None
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")

    def _record(task: Task16, result: dict) -> None:
        finished[task.key] = result
        if handle is not None:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
        if progress is not None:
            progress(len(finished), len(tasks), result)

    try:
        if todo:
            if int(workers) <= 1:
                _worker_init(str(root), device, configure_threads=False)
                for task in todo:
                    _record(task, run_task16(task))
            else:
                from ...search.phase15.loaders import ensure_phase9_anchor

                ensure_phase9_anchor(root=str(root))
                with ProcessPoolExecutor(
                    max_workers=int(workers),
                    initializer=_worker_init,
                    initargs=(str(root), device),
                ) as pool:
                    futures = {pool.submit(run_task16, task): task for task in todo}
                    for future in as_completed(futures):
                        _record(futures[future], future.result())
    finally:
        if handle is not None:
            handle.close()
    missing = [task.key for task in tasks if task.key not in finished]
    if missing:
        raise Phase16RunnerError(f"{len(missing)} tasks produced no result")
    return [finished[task.key] for task in tasks]


# ---------------------------------------------------------------------------
# score_on_benchmark
# ---------------------------------------------------------------------------


def resolve_subset(manifest: dict, subset) -> "list[str]":
    """The board ids a subset names. `None` = the full pack."""
    from .benchmark import quick_subset_ids

    all_ids = [row["board_id"] for row in manifest["boards"]]
    if subset is None:
        return all_ids
    if subset == QUICK_SUBSET_NAME:
        return quick_subset_ids(manifest)
    requested = list(subset)
    known = set(all_ids)
    unknown = [board for board in requested if board not in known]
    if unknown:
        raise Phase16RunnerError(
            f"{len(unknown)} requested boards are not in the benchmark, first: "
            f"{unknown[0]!r}"
        )
    return requested


def score_on_benchmark(
    mode_or_provider,
    preset: str = "TINY",
    workers: int = 8,
    subset=None,
    *,
    root: str = ".",
    device: str = "cpu",
    manifest: "dict | None" = None,
    out_path: "Path | str | None" = None,
    progress=None,
) -> dict:
    """Score one system (or provider seat) on `phase16_benchmark_v1`.

    Direct pairings ignore the preset; it is recorded as `direct` so a row's
    identity never claims a budget that was not used.
    """
    from ...search.phase15.analysis import arm_summary
    from .benchmark import load_benchmark_manifest

    manifest = load_benchmark_manifest(root=root) if manifest is None else manifest
    seat_spec = normalize_seat_spec(mode_or_provider)
    preset_name = preset
    if isinstance(seat_spec, str) and pairing_of(seat_spec).kind == "direct":
        preset_name = "direct"
    boards = resolve_subset(manifest, subset)
    tasks = [Task16(seat_spec, preset_name, board) for board in boards]
    results = run_pack16(
        tasks,
        root=root,
        device=device,
        workers=workers,
        out_path=out_path,
        progress=progress,
    )
    rows = [entry["row"] for entry in results]
    move_seconds = {entry["row"]["board_id"]: entry["move_seconds"] for entry in results}
    return {
        "runner_version": RUNNER_VERSION,
        "pack": manifest["artifact"],
        "manifest_digest": manifest["manifest_digest"],
        "subset": QUICK_SUBSET_NAME if subset == QUICK_SUBSET_NAME else (
            "full" if subset is None else "custom"
        ),
        "preset": preset_name,
        "games": len(rows),
        "summary": arm_summary(rows, move_seconds),
        "rows": rows,
        "latency_note": (
            "pack move times carry multi-process scheduler contention "
            "(~1.8x, Phase 15 measured); latency claims come only from idle "
            "single-process runs"
        ),
    }


__all__ = [
    "Phase16RunnerError",
    "RUNNER_VERSION",
    "Task16",
    "load_results",
    "normalize_seat_spec",
    "resolve_subset",
    "run_pack16",
    "run_task16",
    "score_on_benchmark",
]
