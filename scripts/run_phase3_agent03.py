#!/usr/bin/env python3
"""Phase 3 Agent 3 acceptance harness.

Runs the trajectory-storage gates that are too slow for the ordinary pytest run
and writes `reports/phase_3_data/agent_03_trajectory_reconstruction.json` plus
`reports/phase_3_data/agent_03_snapshot_interval_raw.csv`:

- the exact-reconstruction gate: at least 1,000,000 historical decisions
  rebuilt from `game record + nearest snapshot + subsequent actions` and
  compared field by field against digests taken from the live game;
- the snapshot-interval benchmark over 16, 32 and 64 plies, on the *same*
  deterministic games, measuring storage and random-access reconstruction cost;
- the sparse decision-storage checks on every stored decision;
- the public event-stream check, which needs a full replay from ply 0 and is
  therefore run per game rather than per decision;
- the automated test suite summary.

Dense 10,000-entry legality masks are compared on a stratified subset: a
million dense comparisons would dominate the runtime without testing anything
the legal-action lists do not already cover. Legal-action *lists* are compared
for every one of the 1,000,000 decisions. Both counts are reported.

No neural network, no PyTorch, no Metal: this agent stores and rebuilds
positions only.

Usage:

    python scripts/run_phase3_agent03.py                 # full acceptance run
    python scripts/run_phase3_agent03.py --quick         # fast smoke run
    python scripts/run_phase3_agent03.py --skip-pytest   # measurements only
    python scripts/run_phase3_agent03.py --skip-benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
    TRAINING_RULES,
)
from stratego.engine.events import filter_events_for_observer  # noqa: E402
from stratego.engine.replay import rebuild_final_state  # noqa: E402
from stratego.training.batch_simulation import BatchSimulator  # noqa: E402
from stratego.training.reconstruction import (  # noqa: E402
    RECONSTRUCTION_VERSION,
    compare_digests,
    digest_live_decision,
    digest_reconstructed_decision,
    reconstruct_decision,
    reconstruct_state,
)
from stratego.training.serialization import SERIALIZATION_VERSION  # noqa: E402
from stratego.training.trajectory import (  # noqa: E402
    DEFAULT_SNAPSHOT_INTERVAL,
    SUPPORTED_SNAPSHOT_INTERVALS,
    SYNTHETIC_POLICY_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    TRAJECTORY_VERSION,
    collect_games,
    decode_game_record,
    encode_game_record,
    encode_game_record_compressed,
    validate_game_record,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_3_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_03_trajectory_reconstruction.json"
DEFAULT_RAW_OUTPUT = DATA_DIRECTORY / "agent_03_snapshot_interval_raw.csv"

AGENT_01_DATA = DATA_DIRECTORY / "agent_01_batch_equivalence.json"
AGENT_02_DATA = DATA_DIRECTORY / "agent_02_shared_memory_scaling.json"

REPORT_TARGET_DECISIONS = 1_000_000

# One dense mask compared per this many decisions. 10 keeps roughly 100,000
# dense comparisons, which is a large stratified subset without letting a
# 10,000-entry array per decision dominate the run.
DEFAULT_MASK_STRIDE = 10

# Ply buckets the dense-mask subset is reported against, so the stratification
# is visible rather than asserted.
_PLY_BUCKETS = (0, 16, 32, 64, 128, 256, 512)


def _ply_bucket(ply: int) -> str:
    label = f"{_PLY_BUCKETS[-1]}+"
    for lower, upper in zip(_PLY_BUCKETS, _PLY_BUCKETS[1:]):
        if lower <= ply < upper:
            label = f"{lower}-{upper - 1}"
            break
    return label


def peak_memory_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux reports kilobytes.
    return usage if sys.platform == "darwin" else usage * 1024


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def read_prerequisites() -> tuple[bool, list[str], dict]:
    """Agent 1 and Agent 2 must both be `PASS` before this agent may run."""
    problems: list[str] = []
    summary: dict = {}
    for name, path in (("agent_01", AGENT_01_DATA), ("agent_02", AGENT_02_DATA)):
        if not path.exists():
            problems.append(f"{name}: missing data file {path}")
            summary[name] = {"status": "MISSING"}
            continue
        payload = json.loads(path.read_text())
        status = payload.get("status")
        summary[name] = {
            "status": status,
            "implementation_version": payload.get("implementation_version"),
            "observation_version": payload.get("observation_version"),
            "rules_version": payload.get("rules_version"),
        }
        if status != "PASS":
            problems.append(f"{name}: status is {status!r}, not PASS")
        for field, expected in (
            ("implementation_version", IMPLEMENTATION_VERSION),
            ("observation_version", OBSERVATION_VERSION),
            ("rules_version", RULES_VERSION),
        ):
            if payload.get(field) != expected:
                problems.append(
                    f"{name}: {field} is {payload.get(field)!r}, expected {expected!r}"
                )
    return not problems, problems, summary


# ---------------------------------------------------------------------------
# Stage 1: exact reconstruction
# ---------------------------------------------------------------------------


class ReconstructionGate:
    """Collect games, then rebuild and compare every decision they contain.

    Games are verified as they finish and then dropped, so peak memory tracks
    the number of games in flight rather than the size of the corpus.
    """

    def __init__(
        self,
        *,
        target_decisions: int,
        environments: int,
        root_seed: int,
        snapshot_interval: int,
        mask_stride: int,
        event_stream_games: int,
    ) -> None:
        self.target_decisions = target_decisions
        self.environments = environments
        self.root_seed = root_seed
        self.snapshot_interval = snapshot_interval
        self.mask_stride = mask_stride
        self.event_stream_games = event_stream_games

        self.live: dict[str, list] = {}
        self.live_events: dict[str, tuple[bytes, bytes] | None] = {}
        self.collected_decisions = 0

        self.games_generated = 0
        self.decisions_stored = 0
        self.reconstructed = 0
        self.mismatch_counts: Counter = Counter()
        self.mismatch_details: list[dict] = []
        self.legal_mask_checks = 0
        self.mask_bucket_counts: Counter = Counter()
        self.replayed_actions: list[int] = []
        self.record_problems: list[dict] = []
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.snapshot_bytes = 0
        self.decision_bytes = 0
        self.terminal_reasons: Counter = Counter()
        self.terminal_results: Counter = Counter()
        self.game_lengths: list[int] = []
        self.legal_set_sizes: list[int] = []
        self.event_stream_checks = 0
        self.event_stream_mismatches = 0
        self.codec_round_trip_failures = 0
        self.collection_seconds = 0.0
        self.verification_seconds = 0.0

    # -- live capture -----------------------------------------------------

    def _on_decision(self, state, decision, builder) -> None:
        dense = self.collected_decisions % self.mask_stride == 0
        self.live.setdefault(state.game_id, []).append(
            digest_live_decision(
                state,
                decision,
                environment_id=builder.environment_id,
                generation=builder.generation,
                dense_mask=dense,
                legal_action_ids=decision.legal_action_ids,
            )
        )
        self.collected_decisions += 1

    def _on_game_finished(self, record, state) -> None:
        """Capture the live public event streams for the sampled games.

        A compact snapshot deliberately carries no derived event log
        (`08_internal_state_spec.md` section 15), so the public event *stream*
        is checked by replaying the record from ply 0 rather than from a
        snapshot. That is a whole-game operation, so it is sampled per game.
        """
        if self.games_generated < self.event_stream_games:
            self.live_events[record.game_id] = (
                _event_stream_digest(state, 0),
                _event_stream_digest(state, 1),
            )
        else:
            self.live_events[record.game_id] = None

    # -- verification -----------------------------------------------------

    def _verify_game(self, record) -> None:
        live_digests = self.live.pop(record.game_id)
        live_events = self.live_events.pop(record.game_id, None)

        raw = encode_game_record(record)
        compressed = encode_game_record_compressed(record)
        decoded = decode_game_record(raw)
        if decoded != record:
            self.codec_round_trip_failures += 1
            self.mismatch_details.append(
                {"game_id": record.game_id, "category": "codec_round_trip"}
            )

        problems = validate_game_record(decoded)
        if problems:
            self.record_problems.append({"game_id": record.game_id, "problems": problems[:8]})

        self.raw_bytes += len(raw)
        self.compressed_bytes += len(compressed)
        self.snapshot_bytes += decoded.snapshot_bytes
        self.decision_bytes += _decision_payload_bytes(decoded)
        self.games_generated += 1
        self.game_lengths.append(decoded.final_ply)
        self.terminal_reasons[decoded.terminal_reason] += 1
        self.terminal_results[decoded.terminal_result] += 1

        # Everything below is verified against the *decoded* record, so the
        # comparison covers the codec as well as the reconstruction.
        for decision in decoded.decisions:
            live = live_digests[decision.ply]
            dense = live.legal_mask is not None
            rebuilt = reconstruct_decision(decoded, decision.ply, dense_mask=dense)
            digest = digest_reconstructed_decision(rebuilt, decision)
            mismatches = compare_digests(live, digest)
            self.reconstructed += 1
            self.replayed_actions.append(rebuilt.replayed_actions)
            self.legal_set_sizes.append(len(rebuilt.legal_action_ids))
            if dense:
                self.legal_mask_checks += 1
                self.mask_bucket_counts[_ply_bucket(decision.ply)] += 1
            for category, attribute in mismatches:
                self.mismatch_counts[category] += 1
                if len(self.mismatch_details) < 50:
                    self.mismatch_details.append(
                        {
                            "game_id": decoded.game_id,
                            "ply": decision.ply,
                            "category": category,
                            "field": attribute,
                        }
                    )

        self.decisions_stored += len(decoded.decisions)

        if live_events is not None:
            self._verify_event_stream(decoded, live_events)

    def _verify_event_stream(self, record, live_events) -> None:
        replayed = rebuild_final_state(record.to_replay_record())
        self.event_stream_checks += 1
        for observer, expected in enumerate(live_events):
            if _event_stream_digest(replayed, observer) != expected:
                self.event_stream_mismatches += 1
                self.mismatch_details.append(
                    {
                        "game_id": record.game_id,
                        "category": "public_event_stream",
                        "observer": observer,
                    }
                )

    # -- driver -----------------------------------------------------------

    def run(self, *, progress_every: int = 50_000) -> None:
        simulator = BatchSimulator(
            self.environments, root_seed=self.root_seed, rules=TRAINING_RULES
        )
        stream = collect_games(
            simulator,
            # An upper bound only: the loop stops on the decision target.
            games=10**9,
            snapshot_interval=self.snapshot_interval,
            collection_policy_version=SYNTHETIC_POLICY_VERSION,
            on_decision=self._on_decision,
            on_game_finished=self._on_game_finished,
        )
        started = time.perf_counter()
        last_report = 0
        for record in stream:
            collected_at = time.perf_counter()
            self._verify_game(record)
            now = time.perf_counter()
            self.verification_seconds += now - collected_at
            self.collection_seconds = (now - started) - self.verification_seconds
            if self.reconstructed - last_report >= progress_every:
                last_report = self.reconstructed
                rate = self.reconstructed / max(now - started, 1e-9)
                print(
                    f"  {self.reconstructed:>9,} decisions | "
                    f"{self.games_generated:>6,} games | "
                    f"{rate:7.0f} decisions/s | "
                    f"{sum(self.mismatch_counts.values())} mismatches",
                    flush=True,
                )
            if self.reconstructed >= self.target_decisions:
                break

    # -- report -----------------------------------------------------------

    def summary(self) -> dict:
        replayed = self.replayed_actions
        return {
            "games_generated": self.games_generated,
            "decisions_stored": self.decisions_stored,
            "historical_decisions_reconstructed": self.reconstructed,
            "legal_mask_checks": self.legal_mask_checks,
            "legal_mask_stride": self.mask_stride,
            "legal_mask_ply_buckets": dict(sorted(self.mask_bucket_counts.items())),
            "public_event_stream_checks": self.event_stream_checks,
            "public_event_stream_mismatches": self.event_stream_mismatches,
            "codec_round_trip_failures": self.codec_round_trip_failures,
            "record_validation_problems": self.record_problems[:20],
            "record_validation_problem_games": len(self.record_problems),
            "mean_game_plies": statistics.fmean(self.game_lengths) if self.game_lengths else 0.0,
            "median_game_plies": statistics.median(self.game_lengths) if self.game_lengths else 0,
            "mean_legal_actions_per_decision": (
                statistics.fmean(self.legal_set_sizes) if self.legal_set_sizes else 0.0
            ),
            "max_legal_actions_per_decision": max(self.legal_set_sizes, default=0),
            "mean_replayed_actions_per_reconstruction": (
                statistics.fmean(replayed) if replayed else 0.0
            ),
            "p95_replayed_actions_per_reconstruction": _percentile(replayed, 95),
            "terminal_reason_counts": dict(self.terminal_reasons),
            "terminal_result_counts": dict(self.terminal_results),
            "collection_seconds": self.collection_seconds,
            "verification_seconds": self.verification_seconds,
            "verified_decisions_per_second": (
                self.reconstructed / self.verification_seconds
                if self.verification_seconds
                else 0.0
            ),
        }

    def mismatch_summary(self) -> dict:
        counts = self.mismatch_counts
        return {
            "state_mismatches": counts.get("state", 0),
            "observation_mismatches": counts.get("observation", 0),
            "legal_list_mismatches": counts.get("legal_list", 0),
            "legal_mask_mismatches": counts.get("legal_mask", 0),
            "belief_target_mismatches": counts.get("belief_target", 0),
            "public_knowledge_mismatches": counts.get("public_knowledge", 0),
            "acting_player_mismatches": counts.get("acting_player", 0),
            "selected_action_mismatches": counts.get("selected_action", 0),
            "identity_generation_mismatches": counts.get("identity_generation", 0),
            "total_mismatches": sum(counts.values())
            + self.event_stream_mismatches
            + self.codec_round_trip_failures,
            "mismatch_details": self.mismatch_details[:50],
        }


def _corpus_digest(records: "list") -> str:
    """Digest of the games in a benchmark corpus, ignoring snapshot cadence."""
    import hashlib

    hasher = hashlib.blake2b(digest_size=8)
    for record in records:
        hasher.update(record.game_id.encode())
        hasher.update(repr(record.actions).encode())
    return hasher.hexdigest()


def _event_stream_digest(state, observer: int) -> bytes:
    import hashlib

    payload = repr(filter_events_for_observer(state.events, observer)).encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def _decision_payload_bytes(record) -> int:
    """Bytes attributable to the decisions alone, isolated from the header.

    Measured by encoding the record twice, once with an empty decision list.
    """
    without = dataclass_replace(record, decisions=())
    return len(encode_game_record(record)) - len(encode_game_record(without))


def _percentile(values: "list[int] | list[float]", percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# Stage 2: snapshot-interval benchmark
# ---------------------------------------------------------------------------


def benchmark_interval(
    interval: int,
    *,
    games: int,
    environments: int,
    root_seed: int,
    sample_positions: int,
) -> dict:
    """Storage and random-access reconstruction cost at one snapshot interval.

    The same `root_seed` is used for every interval, so all three measure the
    identical set of games and differ only in snapshot cadence.
    """
    simulator = BatchSimulator(environments, root_seed=root_seed, rules=TRAINING_RULES)
    records = []
    for record in collect_games(simulator, games=games, snapshot_interval=interval):
        records.append(record)

    raw_sizes = [len(encode_game_record(record)) for record in records]
    compressed_sizes = [len(encode_game_record_compressed(record)) for record in records]
    snapshot_sizes = [record.snapshot_bytes for record in records]
    decision_sizes = [_decision_payload_bytes(record) for record in records]
    plies = [record.final_ply for record in records]
    snapshot_counts = [len(record.snapshots) for record in records]

    # Random-access reconstruction: one snapshot restore plus a replay per
    # position, which is what a training sampler actually does. Positions are
    # walked with a stride co-prime with the interval so the sample is spread
    # evenly across the distance-from-snapshot distribution rather than
    # clustering on snapshot plies.
    targets: list[tuple[int, int]] = []
    stride = 7
    offset = 0
    while len(targets) < sample_positions:
        added = 0
        for index, record in enumerate(records):
            ply = offset
            while ply < record.final_ply and len(targets) < sample_positions:
                targets.append((index, ply))
                ply += stride
                added += 1
        if not added:
            break
        offset += 1
        if offset >= stride:
            break

    replayed_counts: list[int] = []
    started = time.perf_counter()
    for index, ply in targets:
        _, replayed = reconstruct_state(records[index], ply)
        replayed_counts.append(replayed)
    state_seconds = time.perf_counter() - started

    # The full sampler path: state plus observation, legal actions and the
    # privileged belief target.
    started = time.perf_counter()
    for index, ply in targets:
        reconstruct_decision(records[index], ply, include_public_knowledge=False)
    full_seconds = time.perf_counter() - started

    return {
        "snapshot_interval": interval,
        # Identical across intervals by construction: the collection policy is a
        # function of `(game_id, ply)` only, so the same root seed replays the
        # same games and the byte comparison is like for like. Recorded so that
        # the claim is checked rather than asserted.
        "corpus_digest": _corpus_digest(records),
        "games": len(records),
        "total_plies": sum(plies),
        "mean_game_plies": statistics.fmean(plies) if plies else 0.0,
        "mean_snapshots_per_game": (
            statistics.fmean(snapshot_counts) if snapshot_counts else 0.0
        ),
        "mean_bytes_per_game": statistics.fmean(raw_sizes) if raw_sizes else 0.0,
        "median_bytes_per_game": statistics.median(raw_sizes) if raw_sizes else 0.0,
        "mean_compressed_bytes_per_game": (
            statistics.fmean(compressed_sizes) if compressed_sizes else 0.0
        ),
        "median_compressed_bytes_per_game": (
            statistics.median(compressed_sizes) if compressed_sizes else 0.0
        ),
        "mean_snapshot_bytes_per_game": (
            statistics.fmean(snapshot_sizes) if snapshot_sizes else 0.0
        ),
        "mean_decision_bytes_per_game": (
            statistics.fmean(decision_sizes) if decision_sizes else 0.0
        ),
        "mean_bytes_per_decision": (
            sum(decision_sizes) / sum(plies) if sum(plies) else 0.0
        ),
        "sampled_positions": len(targets),
        "state_reconstruction_positions_per_second": (
            len(targets) / state_seconds if state_seconds else 0.0
        ),
        "reconstruction_positions_per_second": (
            len(targets) / full_seconds if full_seconds else 0.0
        ),
        "mean_replayed_actions_per_reconstruction": (
            statistics.fmean(replayed_counts) if replayed_counts else 0.0
        ),
        "p95_replayed_actions_per_reconstruction": _percentile(replayed_counts, 95),
        "max_replayed_actions_per_reconstruction": max(replayed_counts, default=0),
        "estimated_million_game_bytes": (
            statistics.fmean(raw_sizes) * 1_000_000 if raw_sizes else 0.0
        ),
        "estimated_million_game_compressed_bytes": (
            statistics.fmean(compressed_sizes) * 1_000_000 if compressed_sizes else 0.0
        ),
    }


# Storage the recommendation is willing to spend, as a fraction above the
# cheapest measured interval, in exchange for faster reconstruction. Stated
# here rather than buried in the function because it is the judgement the
# recommendation turns on.
STORAGE_BUDGET_FRACTION = 0.15


def recommend_interval(results: "list[dict]") -> tuple[int, str]:
    """Choose an interval from measured storage and reconstruction cost.

    Neither extreme is taken automatically. A shorter interval always
    reconstructs faster and always stores more, so the rule sets an explicit
    storage budget -- `STORAGE_BUDGET_FRACTION` above the cheapest measured
    interval -- and takes the fastest interval that fits inside it. That
    rejects the fastest interval when its storage cost is disproportionate, and
    rejects the cheapest interval when a modest amount of storage buys a large
    amount of throughput.
    """
    cheapest_bytes = min(result["mean_bytes_per_game"] for result in results)
    best_rate = max(result["reconstruction_positions_per_second"] for result in results)
    budget = cheapest_bytes * (1.0 + STORAGE_BUDGET_FRACTION)

    affordable = [result for result in results if result["mean_bytes_per_game"] <= budget]
    chosen = max(
        affordable or results,
        key=lambda result: result["reconstruction_positions_per_second"],
    )
    interval = chosen["snapshot_interval"]

    fastest = max(results, key=lambda result: result["reconstruction_positions_per_second"])
    cheapest = min(results, key=lambda result: result["mean_bytes_per_game"])
    overhead = chosen["mean_bytes_per_game"] / cheapest_bytes - 1.0 if cheapest_bytes else 0.0
    compressed_overhead = (
        chosen["mean_compressed_bytes_per_game"] / cheapest["mean_compressed_bytes_per_game"]
        - 1.0
        if cheapest["mean_compressed_bytes_per_game"]
        else 0.0
    )
    speedup = (
        chosen["reconstruction_positions_per_second"]
        / cheapest["reconstruction_positions_per_second"]
        if cheapest["reconstruction_positions_per_second"]
        else 0.0
    )

    parts = [
        f"Interval {interval} stores {chosen['mean_bytes_per_game']:.0f} raw bytes/game "
        f"({chosen['mean_snapshot_bytes_per_game']:.0f} of them snapshots), "
        f"{overhead * 100:.1f} percent above the cheapest measured interval "
        f"({cheapest['snapshot_interval']}) raw and {compressed_overhead * 100:.1f} percent "
        f"above it compressed, and reconstructs at "
        f"{chosen['reconstruction_positions_per_second']:.0f} positions/second, "
        f"{speedup:.2f}x the cheapest interval's rate, replaying "
        f"{chosen['mean_replayed_actions_per_reconstruction']:.1f} actions on average "
        f"(p95 {chosen['p95_replayed_actions_per_reconstruction']:.0f}).",
        f"Storage budget for the choice: {STORAGE_BUDGET_FRACTION * 100:.0f} percent above "
        f"the cheapest measured interval.",
    ]
    if fastest["snapshot_interval"] != interval:
        fastest_overhead = (
            fastest["mean_bytes_per_game"] / cheapest_bytes - 1.0 if cheapest_bytes else 0.0
        )
        parts.append(
            f"Interval {fastest['snapshot_interval']} is faster "
            f"({fastest['reconstruction_positions_per_second']:.0f} positions/second) but "
            f"costs {fastest_overhead * 100:.1f} percent more raw storage than the cheapest "
            "interval, which is outside the budget; it was not chosen on speed alone."
        )
    return interval, " ".join(parts)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0}
    for key, pattern in (
        ("passed", r"(\d+) passed"),
        ("failed", r"(\d+) failed"),
        ("errors", r"(\d+) error"),
        ("skipped", r"(\d+) skipped"),
        ("xfailed", r"(\d+) xfailed"),
    ):
        match = re.search(pattern, output)
        if match:
            counts[key] = int(match.group(1))
    failures = [line for line in output.splitlines() if line.startswith("FAILED")]
    return {
        "test_exit_code": completed.returncode,
        "test_passed": counts["passed"],
        "test_failed": counts["failed"],
        "test_errors": counts["errors"],
        "test_skipped": counts["skipped"],
        "test_expected_failures": counts["xfailed"],
        "test_total": sum(counts.values()),
        "test_seconds": elapsed,
        "test_failure_lines": failures[:20],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--decisions", type=int, default=REPORT_TARGET_DECISIONS)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--root-seed", type=int, default=30_003)
    parser.add_argument(
        "--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL
    )
    parser.add_argument("--mask-stride", type=int, default=DEFAULT_MASK_STRIDE)
    parser.add_argument("--event-stream-games", type=int, default=200)
    parser.add_argument("--benchmark-games", type=int, default=200)
    parser.add_argument("--benchmark-positions", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    options = parser.parse_args()

    if options.quick:
        options.decisions = 4_000
        options.environments = 8
        options.event_stream_games = 10
        options.benchmark_games = 12
        options.benchmark_positions = 600

    options.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Phase 3 Agent 3 - trajectory storage and reconstruction")
    print(f"repository: {REPOSITORY_ROOT}")
    print(f"target decisions: {options.decisions:,}")
    print()

    ready, prerequisite_problems, prerequisite_summary = read_prerequisites()
    if not ready:
        print("BLOCKED: prerequisites are not satisfied")
        for problem in prerequisite_problems:
            print(f"  - {problem}")
        report = {
            "agent": "agent_03_trajectory_reconstruction",
            "status": "BLOCKED",
            "blocking_reasons": prerequisite_problems,
            "prerequisites": prerequisite_summary,
            "trajectory_version": TRAJECTORY_VERSION,
            "files_created": [],
            "files_modified": [],
        }
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 2

    print("prerequisites: agent_01 PASS, agent_02 PASS")
    print()

    print(f"stage 1: exact reconstruction at snapshot interval {options.snapshot_interval}")
    gate = ReconstructionGate(
        target_decisions=options.decisions,
        environments=options.environments,
        root_seed=options.root_seed,
        snapshot_interval=options.snapshot_interval,
        mask_stride=options.mask_stride,
        event_stream_games=options.event_stream_games,
    )
    gate.run(progress_every=1_000 if options.quick else 50_000)
    gate_summary = gate.summary()
    mismatches = gate.mismatch_summary()
    print(
        f"  {gate_summary['historical_decisions_reconstructed']:,} decisions from "
        f"{gate_summary['games_generated']:,} games, "
        f"{mismatches['total_mismatches']} mismatches"
    )
    print(
        f"  dense mask comparisons: {gate_summary['legal_mask_checks']:,}; "
        f"public event streams: {gate_summary['public_event_stream_checks']:,}"
    )
    print()

    interval_results: list[dict] = []
    benchmark_corpus_identical = None
    if not options.skip_benchmark:
        print("stage 2: snapshot-interval benchmark")
        for interval in SUPPORTED_SNAPSHOT_INTERVALS:
            result = benchmark_interval(
                interval,
                games=options.benchmark_games,
                environments=min(options.environments, 16),
                root_seed=options.root_seed + 1,
                sample_positions=options.benchmark_positions,
            )
            interval_results.append(result)
            print(
                f"  interval {interval:>3}: "
                f"{result['mean_bytes_per_game']:>9,.0f} raw B/game | "
                f"{result['mean_compressed_bytes_per_game']:>9,.0f} zlib B/game | "
                f"{result['mean_snapshot_bytes_per_game']:>8,.0f} snapshot B/game | "
                f"{result['reconstruction_positions_per_second']:>7,.0f} pos/s | "
                f"mean replay {result['mean_replayed_actions_per_reconstruction']:>5.1f} | "
                f"p95 {result['p95_replayed_actions_per_reconstruction']:>5.1f}"
            )
        benchmark_corpus_identical = (
            len({result["corpus_digest"] for result in interval_results}) == 1
        )
        print(f"  identical corpus across intervals: {benchmark_corpus_identical}")
        print()

    if interval_results:
        recommended, rationale = recommend_interval(interval_results)
        chosen = next(
            result
            for result in interval_results
            if result["snapshot_interval"] == recommended
        )
    else:
        recommended, rationale = options.snapshot_interval, "benchmark skipped"
        chosen = {}

    tests = run_pytest() if not options.skip_pytest else {}
    if tests:
        print(
            f"tests: {tests['test_passed']} passed, {tests['test_failed']} failed, "
            f"{tests['test_errors']} errors in {tests['test_seconds']:.1f}s"
        )
        print()

    mean_bytes_per_game = (
        gate.raw_bytes / gate.games_generated if gate.games_generated else 0.0
    )
    mean_compressed_per_game = (
        gate.compressed_bytes / gate.games_generated if gate.games_generated else 0.0
    )

    gate_clean = (
        mismatches["total_mismatches"] == 0
        and gate.record_problems == []
        and gate.codec_round_trip_failures == 0
    )
    reached_target = gate.reconstructed >= options.decisions
    tests_clean = not tests or (
        tests["test_exit_code"] == 0 and tests["test_failed"] == 0 and tests["test_errors"] == 0
    )
    benchmark_complete = options.skip_benchmark or (
        len(interval_results) == len(SUPPORTED_SNAPSHOT_INTERVALS)
        and benchmark_corpus_identical is not False
    )

    status = (
        "PASS"
        if gate_clean and reached_target and tests_clean and benchmark_complete
        else "FAIL"
    )

    report = {
        "agent": "agent_03_trajectory_reconstruction",
        "status": status,
        "quick_mode": options.quick,
        "trajectory_version": TRAJECTORY_VERSION,
        "trajectory_format_version": TRAJECTORY_FORMAT_VERSION,
        "serialization_version": SERIALIZATION_VERSION,
        "reconstruction_version": RECONSTRUCTION_VERSION,
        "collection_policy_version": SYNTHETIC_POLICY_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "rules_version": RULES_VERSION,
        "prerequisites": prerequisite_summary,
        "target_decisions": options.decisions,
        "root_seed": options.root_seed,
        "environments": options.environments,
        "default_snapshot_interval": options.snapshot_interval,
        "snapshot_intervals_tested": list(SUPPORTED_SNAPSHOT_INTERVALS),
        "snapshot_interval_results": interval_results,
        "benchmark_corpus_identical": benchmark_corpus_identical,
        "recommended_snapshot_interval": recommended,
        "recommended_snapshot_interval_rationale": rationale,
        "mean_replay_bytes": mean_bytes_per_game,
        "mean_replay_compressed_bytes": mean_compressed_per_game,
        "mean_snapshot_bytes": (
            gate.snapshot_bytes / gate.games_generated if gate.games_generated else 0.0
        ),
        "mean_decision_bytes": (
            gate.decision_bytes / gate.decisions_stored if gate.decisions_stored else 0.0
        ),
        "mean_decision_bytes_per_game": (
            gate.decision_bytes / gate.games_generated if gate.games_generated else 0.0
        ),
        "estimated_million_game_bytes": mean_bytes_per_game * 1_000_000,
        "estimated_million_game_compressed_bytes": mean_compressed_per_game * 1_000_000,
        "reconstruction_positions_per_second": chosen.get(
            "reconstruction_positions_per_second",
            gate_summary["verified_decisions_per_second"],
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory_bytes(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "files_created": [
            "stratego/training/serialization.py",
            "stratego/training/trajectory.py",
            "stratego/training/reconstruction.py",
            "tests/training/test_trajectory.py",
            "tests/training/test_reconstruction.py",
            "scripts/run_phase3_agent03.py",
            "reports/phase_3_data/agent_03_trajectory_reconstruction.json",
            "reports/phase_3_data/agent_03_snapshot_interval_raw.csv",
        ],
        "files_modified": [
            "stratego/training/__init__.py",
            "reports/phase_3_implementation_report.md",
        ],
    }
    report.update(gate_summary)
    report.update(mismatches)
    report.update(tests)

    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if interval_results:
        with options.raw_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(interval_results[0]))
            writer.writeheader()
            writer.writerows(interval_results)

    print(f"status: {status}")
    print(f"recommended snapshot interval: {recommended}")
    print(f"wrote {options.output}")
    if interval_results:
        print(f"wrote {options.raw_output}")
    print(f"elapsed: {report['elapsed_seconds']:.1f}s")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
