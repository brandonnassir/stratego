"""Phase 11 Agent 5: the integrated validation pipeline and final-test entry point.

Specification sources:

- `05_AGENT_5_INTEGRATED_VALIDATION_FREEZE.md` sections "Integrated
  validation" and "Freeze implementation identity"
- Agent 1's eight frozen contracts (bank, metrics, sampler, acceptance)
- Agent 4's frozen production request shape (`phase11_repro.execute_request`)

One pipeline, two banks
-----------------------
Before Agent 5 the Phase 11 pipeline existed only as four agent harnesses:
Agent 2 played the bank and scored it, Agent 3 audited the sampler, Agent 4
proved safety, reproducibility and runtime. Nothing named the *whole*
computation, so "run it again on the sealed test bank" had no referent.

This module is that referent. :func:`run_phase11_pipeline` takes a bank
name and runs every scored stage of Phase 11 end to end — generate,
targets, score, metrics, slices, sampler checks, evidence binding, gate
quantities — by calling the already-accepted Agent 2/3/4 modules. It adds
no arithmetic of its own: every number it produces comes from
`phase11_runner`, `phase11_records`, `phase11_evaluator`,
`phase11_sampler` or `phase11_repro`, which are byte-frozen upstream.

Agent 5 runs it on `phase11_validation_bank_v1`. Agent 7 runs it once on
`phase11_test_bank_v1`.

The seal is structural
----------------------
:data:`SEALED_BANKS` names the banks a caller may not score by accident.
Running the pipeline on one requires passing `sealed_bank_authorized=True`
explicitly, so no default argument, no typo and no loop variable can open
the test bank: a caller that has not written the word has not opened it.
Agent 5 never passes it.

What this module may not do
---------------------------
It holds no weight, no calibration, no threshold, no bin edge, no baseline
and no sampler rule. Every such quantity is imported from the frozen
contract or the frozen implementation module, so a change to this file
cannot move one.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..training.phase11_contract import (
    BELIEF_SAMPLER_VERSION,
    EVALUATOR_VERSION,
    OPPONENT_STRATA,
    PROGRESS_BUCKET_NAMES,
    Phase11ContractError,
    REMAINING_COUNT_BASELINE_VERSION,
    SAMPLER_ZERO_TOLERANCE_COUNTERS,
    TEST_BANK_GAMES,
    TEST_BANK_VERSION,
    VALIDATION_BANK_GAMES,
    VALIDATION_BANK_VERSION,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gate_d,
    evaluate_gate_e,
    evaluate_gate_f,
    evaluate_gate_g,
    evaluate_gate_h,
    progress_bucket,
)

#: The Agent 5 implementation freeze. Agents 6 and 7 name this version.
PIPELINE_VERSION = "phase11_validation_freeze_v1"

#: The single entry point the sealed final test calls.
FINAL_TEST_ENTRY_POINT = "stratego.evaluation.phase11_pipeline.run_phase11_pipeline"

#: The scored stages, in execution order. A pipeline result carries one
#: block per name and the acceptance gate checks the tuple, so a stage
#: cannot be quietly dropped on the sealed run.
PIPELINE_STAGES = (
    "generate",
    "targets",
    "score",
    "metrics",
    "slices",
    "sampler_checks",
    "bound_evidence",
    "gate_quantities",
)

#: Banks whose scored access requires explicit authorization. Agent 5 and
#: Agent 6 never authorize; Agent 7 authorizes exactly once.
SEALED_BANKS = (TEST_BANK_VERSION,)

#: `bank name -> (bank version, expected games, bank artifact)`.
BANK_BINDINGS = {
    "validation": {
        "bank_version": VALIDATION_BANK_VERSION,
        "games_expected": VALIDATION_BANK_GAMES,
        "bank_artifact": "reports/phase_11_data/agent_01_validation_bank.json",
        "store_relative_root": "data/phase11/agent05/validation_predictions",
    },
    "test": {
        "bank_version": TEST_BANK_VERSION,
        "games_expected": TEST_BANK_GAMES,
        "bank_artifact": "reports/phase_11_data/agent_01_test_bank.json",
        "store_relative_root": "data/phase11/agent07/test_predictions",
    },
}

# ---------------------------------------------------------------------------
# The frozen integrated sample schedule
# ---------------------------------------------------------------------------

#: Eligible recorded decisions sampled per game, evenly spaced — the same
#: `floor(k * E / n)` spacing rule Agent 3 froze for the large audit and
#: Agent 1 froze for the benchmark cells. Four per game over 1,024
#: validation games gives 4,096 states.
SAMPLE_DECISIONS_PER_GAME = 4

#: World ordinals per scheduled state: the Phase 12 production request
#: shape Agent 4 measured and Gate G bounds (`forward + 64 worlds`). The
#: integrated pass therefore samples the same object the runtime ceiling
#: is stated about, not a smaller stand-in.
SAMPLE_WORLD_ORDINALS = 64

#: The integrated pass must clear the frozen large-audit world floor on its
#: own bytes, so `sampler_evidence_bound` is a reproduction and not only a
#: citation of Agent 3.
SAMPLE_WORLD_FLOOR = 250_000

class Phase11PipelineError(Phase11ContractError):
    """The integrated pipeline refused to run or produced an invalid block."""


class Phase11SealError(Phase11PipelineError):
    """A sealed bank was reached without explicit authorization."""


# ---------------------------------------------------------------------------
# 1. Generate — play the bank through the frozen paths
# ---------------------------------------------------------------------------


def bank_binding(bank: str) -> dict:
    if bank not in BANK_BINDINGS:
        raise Phase11PipelineError(
            f"unknown Phase 11 bank {bank!r}; the frozen banks are "
            f"{sorted(BANK_BINDINGS)}"
        )
    return dict(BANK_BINDINGS[bank])


def assert_seal(bank: str, *, sealed_bank_authorized: bool) -> None:
    """Refuse a sealed bank unless the caller wrote the authorization.

    The refusal is structural: the default is `False`, and the pipeline
    never derives the flag from a bank name, an environment variable or an
    artifact. Agent 5's completion gates read this function's behaviour,
    not a promise about it.
    """
    binding = bank_binding(bank)
    if binding["bank_version"] in SEALED_BANKS and not sealed_bank_authorized:
        raise Phase11SealError(
            f"{binding['bank_version']} is sealed: scored access needs an "
            "explicit sealed_bank_authorized=True from the agent that owns "
            "the first sealed evaluation (Agent 7). Agent 5 and Agent 6 never "
            "authorize it."
        )


def build_owners(repository_root: Path, export_path: Path, *, device: str = "cpu"):
    """The two long-lived inference owners the run needs.

    Exactly the Agent 2 construction: the accepted Phase 9 export on the
    frozen CPU/float32/greedy backend, plus the Phase 8 anchor seat.
    """
    from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ..training.phase10_collector import export_evaluation_weights
    from ..training.phase11_contract import ACCEPTED_ANCHOR_EXPORT_PATH
    from .neural_worker import DECISION_MODE_GREEDY, InferenceOwner
    from .phase11_belief import Phase11BeliefOwner

    checkpoint = Path(repository_root) / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
    export = export_evaluation_weights(checkpoint, Path(export_path))
    common = {
        "decision_mode": DECISION_MODE_GREEDY,
        "device": device,
        "dtype": "float32",
        "expected_architecture_id": ARCHITECTURE_FAMILY,
        "expected_configuration": candidate_config("C1"),
    }
    owners = {
        "phase9": Phase11BeliefOwner(
            Path(export_path), name="phase11_observer", **common
        ),
        "anchor": InferenceOwner(
            Path(repository_root) / ACCEPTED_ANCHOR_EXPORT_PATH,
            name="phase11_anchor",
            **common,
        ),
    }
    return owners, export


def generate_bank(
    cases,
    owners: dict,
    *,
    bank: str,
    root: Path,
    device: str = "cpu",
    torch_threads: int = 1,
    progress=None,
) -> dict:
    """Play every game of one bank and write the prediction store.

    Stage 1 (`generate`) and stage 2 (`targets`) of the pipeline: the games
    are played by `phase11_runner.play_validation_game`, the public shard is
    written before any truth exists, and `privileged_truth_pass` runs
    afterwards on its own replay. The manifest this returns is the same
    logical object Agent 2 wrote — same keys, same values — so its digest is
    directly comparable.

    Returns `(manifest, elapsed seconds)`; the elapsed time is deliberately
    outside the manifest, which is hashed into the store identity.
    """
    from ..training.phase10_collector import owner_state_digest
    from ..training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    )
    from . import phase11_records as records
    from . import phase11_runner as runner

    binding = bank_binding(bank)
    bank_version = binding["bank_version"]
    model_id = records.model_identity(
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST, ACCEPTED_BELIEF_HEAD_DIGEST
    )
    observed = owner_state_digest(owners["phase9"])
    if observed != ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
        raise Phase11PipelineError(
            f"the loaded observer weights digest {observed} != accepted"
        )

    started = time.perf_counter()
    entries: list[dict] = []
    truth_summaries: list[dict] = []
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    terminal_reasons: dict[str, int] = {}
    decisions = events = 0
    request_digests = hashlib.sha256()
    for position, case in enumerate(cases):
        for game_index in (0, 1):
            plan, result, recorder, observer = runner.play_validation_game(
                case, game_index, owners, bank_version
            )
            entry = records.write_public_shard(root, recorder)
            arrays = records.read_public_shard(root, plan.game_id)
            truth = runner.privileged_truth_pass(plan, result, arrays)
            truth_entry = records.write_truth_shard(
                root, plan.game_id, truth["true_rank_index"]
            )
            entry["truth_shard_digest"] = truth_entry["truth_shard_digest"]
            entries.append(entry)
            truth_summaries.append(
                {key: value for key, value in truth.items() if key != "true_rank_index"}
            )
            outcomes[result.candidate_result] = (
                outcomes.get(result.candidate_result, 0) + 1
            )
            terminal_reasons[result.terminal_reason] = (
                terminal_reasons.get(result.terminal_reason, 0) + 1
            )
            decisions += recorder.decisions
            events += recorder.events
            for digest in observer.request_digests:
                request_digests.update(digest.encode())
        if progress is not None and (position + 1) % 32 == 0:
            progress(position + 1, len(cases), events, time.perf_counter() - started)

    manifest = {
        "store_version": records.PREDICTION_STORE_VERSION,
        "record_version": records.PREDICTION_RECORD_VERSION,
        "run_version": runner.PHASE11_RUN_VERSION,
        "bank_version": bank_version,
        "bank_digest": None,  # filled by the caller from the frozen bank
        "model_identity": model_id,
        "eval_device": device,
        "torch_threads": int(torch_threads),
        "games": len(entries),
        "observer_decisions": decisions,
        "prediction_events": events,
        "games_expected": int(binding["games_expected"]),
        "complete_bank": len(entries) == int(binding["games_expected"]),
        "belief_forwards": owners["phase9"].belief_forwards,
        "belief_rows": owners["phase9"].belief_rows,
        "request_digest_rollup": request_digests.hexdigest(),
        "outcomes": outcomes,
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "games_index": sorted(entries, key=lambda item: item["game_id"]),
        "truth_pass": {
            "games": len(truth_summaries),
            "identity_mismatches": sum(
                row["identity_mismatches"] for row in truth_summaries
            ),
            "alignment_mismatches": sum(
                row["alignment_mismatches"] for row in truth_summaries
            ),
            "count_mismatches": sum(row["count_mismatches"] for row in truth_summaries),
            "mask_mismatches": sum(row["mask_mismatches"] for row in truth_summaries),
            "unlabelled_events": sum(
                row["unlabelled_events"] for row in truth_summaries
            ),
            "verified_decisions": sum(
                row["verified_decisions"] for row in truth_summaries
            ),
        },
    }
    # `manifest_digest` excludes only paths and timestamps, so a duration
    # would enter the store identity. The elapsed time is returned beside
    # the manifest instead of inside it.
    return manifest, round(time.perf_counter() - started, 3)


# ---------------------------------------------------------------------------
# 2. Score — learned and remaining-count baseline over the recorded rows
# ---------------------------------------------------------------------------


#: The per-game manifest fields that are logical store content. Everything
#: a replay determines, and nothing a clock does.
STORE_CONTENT_FIELDS = (
    "bank_version",
    "case_id",
    "case_index",
    "decisions",
    "empty_decisions",
    "events",
    "game_id",
    "game_index",
    "match_id",
    "match_seed",
    "observer_color",
    "observer_decisions",
    "observer_result",
    "opponent_setup_source",
    "opponent_stratum",
    "plies",
    "public_shard_digest",
    "record_version",
    "replay_digest",
    "store_version",
    "terminal_reason",
    "truth_shard_digest",
)

#: The top-level manifest fields that are logical store content. `games`,
#: the two counts and the request rollup identify the run; `games_index` is
#: digested per game through :data:`STORE_CONTENT_FIELDS`.
STORE_CONTENT_MANIFEST_FIELDS = (
    "bank_digest",
    "bank_version",
    "belief_forwards",
    "belief_rows",
    "complete_bank",
    "games",
    "games_expected",
    "model_identity",
    "observer_decisions",
    "prediction_events",
    "record_version",
    "request_digest_rollup",
    "run_version",
    "store_version",
)


def store_content_digest(manifest: dict) -> str:
    """A content-only identity of one prediction store.

    The accepted `phase11_records.manifest_digest` excludes paths and
    timestamps but **not** the per-game `forward_seconds`, so it embeds a
    wall-clock duration and is not reproducible across two executions of
    the same frozen bank. This digest covers exactly the fields a replay
    determines — the public and truth shard content hashes, the replay
    digest, the decision and event counts, the seeds, the terminal reason
    and the run-level rollup — and therefore *is* a pure function of the
    frozen bank and the frozen model.

    It is an addition, never a replacement: the frozen manifest digest is
    still computed, stored and reported unchanged.
    """
    hasher = hashlib.sha256()
    hasher.update(b"phase11_store_content_digest_v1")
    for name in STORE_CONTENT_MANIFEST_FIELDS:
        hasher.update(f"|{name}={manifest[name]}".encode())
    entries = sorted(manifest["games_index"], key=lambda item: item["game_id"])
    hasher.update(f"|games={len(entries)}".encode())
    for entry in entries:
        for name in STORE_CONTENT_FIELDS:
            hasher.update(f"|{name}={entry[name]}".encode())
    return hasher.hexdigest()


def scored_blocks(root: Path, manifest: dict) -> tuple:
    """Every game's scored block, in the frozen game-id order.

    The learned vector is the recorded logits' float64 softmax and the
    baseline is `remaining_count_belief_v1` rebuilt from the recorded public
    counts and mask — neither reads a truth shard until `score_matrix` is
    handed one, which happens after both vectors exist.
    """
    from .phase11_audit import independent_scores
    from .phase11_baselines import remaining_count_distribution
    from .phase11_belief import softmax_float64
    from .phase11_evaluator import score_matrix
    from . import phase11_records as records

    bucket_index = {name: index for index, name in enumerate(PROGRESS_BUCKET_NAMES)}
    blocks = []
    audit_deviations: dict = {}
    log_floor_events = 0
    for entry in manifest["games_index"]:
        arrays = records.read_public_shard(root, entry["game_id"])
        truth = records.read_truth_shard(root, entry["game_id"])
        size = int(truth.size)
        identity = {
            "case_id": entry["case_id"],
            "opponent_stratum": entry["opponent_stratum"],
            "opponent_setup_source": entry["opponent_setup_source"],
            "observer_color": entry["observer_color"],
        }
        if size == 0:
            blocks.append(
                {
                    **identity,
                    "true_rank": np.zeros(0, dtype=np.int64),
                    "bucket_index": np.zeros(0, dtype=np.int8),
                    "piece_moved": np.zeros(0, dtype=np.uint8),
                    "learned": None,
                    "baseline": None,
                }
            )
            continue
        logits = arrays["belief_logits"]
        learned = np.stack([softmax_float64(row) for row in logits])
        offsets = arrays["event_offset"]
        counts = arrays["remaining_counts"]
        masks = arrays["legal_rank_mask"]
        baseline = np.empty_like(learned)
        buckets = np.empty(size, dtype=np.int8)
        for decision in range(int(arrays["decision_index"].size)):
            start, stop = int(offsets[decision]), int(offsets[decision + 1])
            if stop <= start:
                continue
            bucket = bucket_index[
                progress_bucket(int(arrays["decision_index"][decision]))
            ]
            buckets[start:stop] = bucket
            for cursor in range(start, stop):
                baseline[cursor] = remaining_count_distribution(
                    counts[decision], masks[cursor]
                )
        learned_scores = score_matrix(learned, truth)
        baseline_scores = score_matrix(baseline, truth)
        log_floor_events += learned_scores["log_floor_events"]

        audit = independent_scores(learned, truth)
        for name, value in audit.items():
            deviation = float(
                np.abs(np.asarray(learned_scores[name]) - np.asarray(value)).max()
            )
            audit_deviations[name] = max(audit_deviations.get(name, 0.0), deviation)

        blocks.append(
            {
                **identity,
                "true_rank": truth.astype(np.int64),
                "bucket_index": buckets,
                "piece_moved": arrays["piece_moved"],
                "learned": learned_scores,
                "baseline": baseline_scores,
            }
        )
    return blocks, audit_deviations, int(log_floor_events)


def bank_metrics(blocks, bank: str) -> dict:
    """The frozen overall block and every mandatory diagnostic slice."""
    from .phase11_evaluator import (
        all_finite,
        build_scored_events,
        overall_metrics,
        slice_metrics,
    )

    table = build_scored_events(blocks)
    overall = overall_metrics(table, bank)
    slices = slice_metrics(table, bank)
    nonfinite = all_finite({"overall": overall, "slices": slices})
    return {
        "table": table,
        "overall": overall,
        "slices": slices,
        "nonfinite_paths": nonfinite,
        "metrics_finite": not nonfinite,
    }


# ---------------------------------------------------------------------------
# 3. The integrated sampler pass
# ---------------------------------------------------------------------------


def _evenly_spaced(values, count: int) -> list:
    """The frozen `floor(k * n / take)` spacing rule, Agent 1's exactly."""
    if not values:
        return []
    take = min(int(count), len(values))
    return [values[(index * len(values)) // take] for index in range(take)]


def frozen_sample_schedule(rows, *, decisions_per_game: int = SAMPLE_DECISIONS_PER_GAME) -> list:
    """The frozen integrated sample schedule of one bank.

    Per game, the evenly spaced eligible recorded decisions — eligible
    meaning at least one hidden target, which `decision_table` already
    enforces. Every game contributes, so the schedule inherits the bank's
    stratum, colour and setup-source balance exactly and no cell can be
    over-represented by a selection accident.
    """
    by_game: dict[str, list] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)
    schedule = []
    for game_id in sorted(by_game):
        ordered = sorted(by_game[game_id], key=lambda item: item["decision_index"])
        for row in _evenly_spaced(ordered, decisions_per_game):
            schedule.append(dict(row))
    for ordinal, row in enumerate(schedule):
        row["schedule_ordinal"] = ordinal
        row["request_ordinal"] = ordinal
        row["request_id"] = (
            f"{PIPELINE_VERSION}|sched={ordinal:06d}|{row['public_state_identity'][:16]}"
        )
    return schedule


def sampler_checks(
    owner,
    schedule,
    *,
    action_histories: dict,
    setups: dict,
    world_ordinals: int = SAMPLE_WORLD_ORDINALS,
    progress=None,
) -> dict:
    """Run the frozen production request over the whole sample schedule.

    Each scheduled state is replayed from public bytes alone, one belief
    forward is taken, and `world_ordinals` complete worlds are sampled
    through `belief_sampler_v1`. Every world is checked against the frozen
    validation stack by the sampler itself — `sample_belief_world` raises on
    a finding — and re-checked here so the counters are reported rather than
    only asserted.
    """
    from .phase11_baselines import validate_world
    from .phase11_repro import execute_request, replay_state

    counters = {name: 0 for name in SAMPLER_ZERO_TOLERANCE_COUNTERS}
    worlds_total = 0
    distinct_states = set()
    distinct_worlds: dict[str, set] = {}
    digest_rollup = hashlib.sha256()
    digest_rollup.update(f"{PIPELINE_VERSION}|sampler_checks".encode())
    started = time.perf_counter()
    forward_ns = sampling_ns = 0

    for position, row in enumerate(schedule):
        spec = {
            **row,
            **setups[(row["case_id"], row["game_index"])],
            "action_history": action_histories[row["game_id"]],
        }
        state, observer = replay_state(spec)
        result, parts = execute_request(
            owner,
            spec,
            world_count=int(world_ordinals),
            state=state,
            observer=observer,
            collect=True,
        )
        document = parts["document"]
        identity = result.public_state_identity
        distinct_states.add(identity)
        bucket = distinct_worlds.setdefault(identity, set())
        for world in parts["worlds"]:
            worlds_total += 1
            check = validate_world(document, world)
            for name, value in check["counters"].items():
                counters[name] += int(value)
            if world["sampler_version"] != BELIEF_SAMPLER_VERSION:
                counters["provenance_mismatches"] += 1
            if world["public_state_identity"] != identity:
                counters["provenance_mismatches"] += 1
            counters["dead_end_events"] += int(world["dead_end_events"])
            bucket.add(
                tuple(sorted((int(k), int(v)) for k, v in world["assignment"].items()))
            )
        for row_probabilities in parts["probabilities"].values():
            values = np.asarray(row_probabilities, dtype=np.float64)
            if not np.isfinite(values).all():
                counters["nonfinite_probability_rows"] += 1
        digest_rollup.update(f"|{result.request_id}={result.digest}".encode())
        forward_ns += int(result.forward_ns)
        sampling_ns += int(result.sampling_ns)
        if progress is not None and (position + 1) % 256 == 0:
            progress(position + 1, len(schedule), worlds_total,
                     time.perf_counter() - started)

    distinct_counts = [len(value) for value in distinct_worlds.values()]
    return {
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "schedule_version": PIPELINE_VERSION,
        "requests": len(schedule),
        "world_ordinals_per_request": int(world_ordinals),
        "worlds": worlds_total,
        "world_floor": SAMPLE_WORLD_FLOOR,
        "meets_world_floor": worlds_total >= SAMPLE_WORLD_FLOOR,
        "distinct_public_states": len(distinct_states),
        "counters": counters,
        "all_counters_zero": all(value == 0 for value in counters.values()),
        "request_rollup_digest": digest_rollup.hexdigest(),
        "mean_distinct_worlds_per_state": (
            float(sum(distinct_counts) / len(distinct_counts))
            if distinct_counts
            else float("nan")
        ),
        "forward_seconds": round(forward_ns / 1e9, 3),
        "sampling_seconds": round(sampling_ns / 1e9, 3),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# 4. Gate quantities and the eight gates
# ---------------------------------------------------------------------------


def gate_quantities(overall: dict, slices: dict) -> dict:
    """Every gate-bearing quantity Gates A-D read, named exactly once.

    Pulled from the metric block by path so a renamed key is an error and
    never a silently substituted number.
    """
    metrics = overall["metrics"]
    strata = slices["opponent_stratum"]
    missing = [name for name in OPPONENT_STRATA if name not in strata]
    if missing:
        raise Phase11PipelineError(f"the stratum slice is missing {missing}")
    return {
        "r_ce": float(metrics["r_ce"]["point"]),
        "r_ce_lower": float(metrics["r_ce"]["lower"]),
        "r_ce_upper": float(metrics["r_ce"]["upper"]),
        "ce_delta_point": float(metrics["ce_delta"]["point"]),
        "ce_delta_upper": float(metrics["ce_delta"]["upper"]),
        "delta_top1": float(metrics["top1_delta"]["point"]),
        "delta_top1_lower": float(metrics["top1_delta"]["lower"]),
        "brier_delta_upper": float(metrics["brier_delta"]["upper"]),
        "ece_overall": float(overall["ece_learned"]["ece"]),
        "stratum_ece": {
            name: float(strata[name]["ece_learned"]["ece"]) for name in OPPONENT_STRATA
        },
        "stratum_r_ce": {
            name: float(strata[name]["r_ce"]["point"]) for name in OPPONENT_STRATA
        },
    }


def evaluate_all_gates(
    quantities: dict,
    *,
    sampler_counters: dict,
    safety_counters: dict,
    leg_exact: dict,
    p95_forward_64_ms: float,
    preservation: dict,
) -> dict:
    """The eight hard gates, each from the frozen contract evaluator."""
    gates = {
        "A": evaluate_gate_a(quantities["r_ce"], quantities["ce_delta_upper"]),
        "B": evaluate_gate_b(
            quantities["delta_top1"], quantities["delta_top1_lower"]
        ),
        "C": evaluate_gate_c(
            quantities["ece_overall"],
            quantities["stratum_ece"],
            quantities["brier_delta_upper"],
        ),
        "D": evaluate_gate_d(quantities["stratum_r_ce"]),
        "E": evaluate_gate_e(dict(sampler_counters)),
        "F": evaluate_gate_f(dict(safety_counters)),
        "G": evaluate_gate_g(dict(leg_exact), float(p95_forward_64_ms)),
        "H": evaluate_gate_h(dict(preservation)),
    }
    return gates


# ---------------------------------------------------------------------------
# 5. The entry point
# ---------------------------------------------------------------------------


def _slots_per_game(rows) -> dict:
    counts: dict = {}
    for row in rows:
        counts[row["game_id"]] = counts.get(row["game_id"], 0) + 1
    return counts


def schedule_accounting(
    rows, schedule, games: int, *, decisions_per_game: int = SAMPLE_DECISIONS_PER_GAME
) -> dict:
    """Exactly what the frozen schedule rule could and did take.

    Three different denominators matter and are reported separately:

    - **nominal** — `games * decisions_per_game`, the size if every game
      offered the full quota;
    - **attainable** — `sum(min(quota, eligible decisions))`, the size the
      rule can actually reach, since a game offers only the decisions at
      which the observer faced a hidden target;
    - **realized** — what the schedule holds.

    A game with no eligible decision at all has nothing to contribute and
    is counted, not silently folded into a shortfall. The rule's guarantee
    is that every *eligible* game contributes, and realized == attainable
    is the check of it.
    """
    eligible = _slots_per_game(rows)
    taken = _slots_per_game(schedule)
    attainable = sum(min(int(decisions_per_game), count) for count in eligible.values())
    return {
        "decisions_per_game": int(decisions_per_game),
        "games": int(games),
        "games_with_eligible_decisions": len(eligible),
        "games_without_eligible_decisions": int(games) - len(eligible),
        "games_contributing": len(taken),
        "games_below_quota": len(
            [count for count in eligible.values() if count < int(decisions_per_game)]
        ),
        "schedule_slots_nominal": int(games) * int(decisions_per_game),
        "schedule_slots_attainable": attainable,
        "schedule_slots_realized": len(schedule),
        "every_eligible_game_contributes": len(taken) == len(eligible),
        "realized_equals_attainable": len(schedule) == attainable,
    }


def _bank_cases(repository_root: Path, bank: str) -> tuple:
    binding = bank_binding(bank)
    payload = json.loads(
        (Path(repository_root) / binding["bank_artifact"]).read_text()
    )
    return payload["cases"], payload["manifest"]


def _action_histories(root: Path, manifest: dict) -> dict:
    from . import phase11_records as records

    histories = {}
    for entry in manifest["games_index"]:
        arrays = records.read_public_shard(root, entry["game_id"])
        histories[entry["game_id"]] = [
            int(value) for value in arrays["action_history"]
        ]
    return histories


def run_phase11_pipeline(
    bank: str,
    repository_root,
    *,
    bound_evidence: dict,
    preservation: dict,
    store_root=None,
    export_path=None,
    device: str = "cpu",
    torch_threads: int = 1,
    sealed_bank_authorized: bool = False,
    limit_cases: "int | None" = None,
    progress=None,
) -> dict:
    """The complete Phase 11 scored pipeline over one bank.

    The eight :data:`PIPELINE_STAGES`, in order, over the frozen paths:
    play the bank, take the privileged targets afterwards, score the
    learned head against `remaining_count_belief_v1`, compute the frozen
    metric block and every mandatory slice with their case bootstraps, run
    `belief_sampler_v1` over the frozen sample schedule, bind the safety /
    reproducibility / runtime evidence the caller supplies, and recompute
    every hard-gate quantity the bank can produce.

    `bound_evidence` and `preservation` are the caller's, because Gates F,
    G and H are statements about the *agent's own run environment* and this
    module must not go and re-derive them from a file it happens to find.
    Agent 5's harness passes the recomputed Agent 3/4 evidence; Agent 7
    passes its own.

    Returns one result document. Nothing is written except the prediction
    store; the caller owns every artifact.
    """
    from . import phase11_records as records
    from .phase11_repro import decision_table

    assert_seal(bank, sealed_bank_authorized=sealed_bank_authorized)
    repository_root = Path(repository_root)
    binding = bank_binding(bank)
    cases, bank_manifest = _bank_cases(repository_root, bank)
    if limit_cases:
        cases = cases[: int(limit_cases)]
    root = (
        Path(store_root)
        if store_root is not None
        else repository_root / binding["store_relative_root"]
    )
    export = (
        Path(export_path)
        if export_path is not None
        else repository_root / "checkpoints" / "phase11" / "agent05" / "eval_phase9_c1.pt"
    )
    export.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    stages: dict = {}

    owners, export_summary = build_owners(repository_root, export, device=device)
    try:
        manifest, generate_seconds = generate_bank(
            cases,
            owners,
            bank=bank,
            root=root,
            device=device,
            torch_threads=torch_threads,
            progress=None if progress is None else progress("generate"),
        )
        manifest["bank_digest"] = bank_manifest["bank_digest"]
        manifest["manifest_digest"] = records.manifest_digest(manifest)
        records.write_manifest(root, manifest)
        stages["generate"] = {
            "games": manifest["games"],
            "games_expected": manifest["games_expected"],
            "complete_bank": manifest["complete_bank"],
            "observer_decisions": manifest["observer_decisions"],
            "prediction_events": manifest["prediction_events"],
            "belief_forwards": manifest["belief_forwards"],
            "manifest_digest": manifest["manifest_digest"],
            "store_content_digest": store_content_digest(manifest),
            "request_digest_rollup": manifest["request_digest_rollup"],
            "outcomes_report_only": manifest["outcomes"],
            "terminal_reasons_report_only": manifest["terminal_reasons"],
            "export": export_summary,
            "wall_clock_seconds": generate_seconds,
        }
        stages["targets"] = dict(manifest["truth_pass"])
        stages["targets"]["targets_read_after_prediction"] = True

        blocks, audit_deviations, log_floor_events = scored_blocks(root, manifest)
        stages["score"] = {
            "games_scored": len(blocks),
            "log_floor_events": log_floor_events,
            "per_event_audit_max_deviation": audit_deviations,
        }

        computed = bank_metrics(blocks, bank)
        stages["metrics"] = {
            "evaluator_version": EVALUATOR_VERSION,
            "events": computed["overall"]["events"],
            "cases_with_events": computed["overall"]["cases_with_events"],
            "cases_without_events": computed["overall"]["cases_without_events"],
            "metrics_finite": computed["metrics_finite"],
            "nonfinite_paths": computed["nonfinite_paths"],
        }
        stages["slices"] = {
            "slice_keys": sorted(computed["slices"]),
            "strata": sorted(computed["slices"]["opponent_stratum"]),
        }

        rows = decision_table(
            root,
            manifest,
            {case["case_id"]: case for case in cases},
        )
        schedule = frozen_sample_schedule(rows)
        histories = _action_histories(root, manifest)
        setups = {}
        for case in cases:
            for game_index in (0, 1):
                game = case["games"][str(game_index)]
                observer_setup = [int(v) for v in game["observer"]["setup"]]
                opponent_setup = [int(v) for v in game["opponent"]["setup"]]
                setups[(case["case_id"], game_index)] = (
                    {"red_setup": observer_setup, "blue_setup": opponent_setup}
                    if game["observer_color"] == "red"
                    else {"red_setup": opponent_setup, "blue_setup": observer_setup}
                )
        stages["sampler_checks"] = sampler_checks(
            owners["phase9"],
            schedule,
            action_histories=histories,
            setups=setups,
            progress=None if progress is None else progress("sampler_checks"),
        )
        # A game offers only the decisions at which the observer faced a
        # hidden target, so the realized schedule is at most the nominal
        # one. The shortfall is accounted for exactly and never made up
        # from another game.
        stages["sampler_checks"]["schedule_accounting"] = schedule_accounting(
            rows, schedule, int(manifest["games"])
        )
    finally:
        owners["phase9"].close()
        owners["anchor"].close()

    stages["bound_evidence"] = {
        "information_safety_version": bound_evidence["information_safety_version"],
        "artifacts": bound_evidence["artifacts"],
        "safety_counters": bound_evidence["safety_counters"],
        "leg_exact": bound_evidence["leg_exact"],
        "p95_forward_64_ms": bound_evidence["runtime"]["p95_forward_64_ms"],
        "sampler_audit_counters": bound_evidence["sampler_audit_counters"],
        "sampler_audit_worlds": bound_evidence["sampler_audit_worlds"],
    }

    quantities = gate_quantities(computed["overall"], computed["slices"])
    # Gate E reads the union of the two independent sampler passes: Agent
    # 3's large audit and this run's integrated schedule. A counter that
    # fired in either is non-zero here.
    combined_sampler = {
        name: int(stages["sampler_checks"]["counters"][name])
        + int(bound_evidence["sampler_audit_counters"].get(name, 0))
        for name in SAMPLER_ZERO_TOLERANCE_COUNTERS
    }
    gates = evaluate_all_gates(
        quantities,
        sampler_counters=combined_sampler,
        safety_counters=bound_evidence["safety_counters"],
        leg_exact=bound_evidence["leg_exact"],
        p95_forward_64_ms=bound_evidence["runtime"]["p95_forward_64_ms"],
        preservation=preservation,
    )
    stages["gate_quantities"] = {
        "quantities": quantities,
        "combined_sampler_counters": combined_sampler,
        "gates": gates,
    }

    if tuple(stages) != PIPELINE_STAGES:
        raise Phase11PipelineError(
            f"the pipeline produced stages {tuple(stages)}, expected "
            f"{PIPELINE_STAGES}"
        )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "bank": bank,
        "bank_version": binding["bank_version"],
        "bank_digest": bank_manifest["bank_digest"],
        "sealed_bank_authorized": bool(sealed_bank_authorized),
        "store_root": str(root),
        "manifest": manifest,
        "blocks": blocks,
        "table": computed["table"],
        "overall": computed["overall"],
        "slices": computed["slices"],
        "schedule": schedule,
        "stages": stages,
        "gates": gates,
        "gate_quantities": quantities,
        "complete": True,
        "smoke_run": bool(limit_cases),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# 6. The implementation freeze identity
# ---------------------------------------------------------------------------

#: The tracked implementation whose bytes are the Phase 11 freeze. Agents 6
#: and 7 re-hash exactly this list; a change to any one of them is a
#: different implementation and must be reported as such.
FROZEN_IMPLEMENTATION_MODULES = (
    "stratego/evaluation/phase11_audit.py",
    "stratego/evaluation/phase11_banks.py",
    "stratego/evaluation/phase11_baselines.py",
    "stratego/evaluation/phase11_belief.py",
    "stratego/evaluation/phase11_evaluator.py",
    "stratego/evaluation/phase11_pipeline.py",
    "stratego/evaluation/phase11_public_state.py",
    "stratego/evaluation/phase11_records.py",
    "stratego/evaluation/phase11_recompute.py",
    "stratego/evaluation/phase11_repro.py",
    "stratego/evaluation/phase11_runner.py",
    "stratego/evaluation/phase11_safety.py",
    "stratego/evaluation/phase11_sampler.py",
    "stratego/evaluation/phase11_sampler_audit.py",
    "stratego/evaluation/phase11_streams.py",
    "stratego/training/phase11_contract.py",
    "stratego/training/phase11_seed.py",
)


def module_sha256(repository_root: Path, names=FROZEN_IMPLEMENTATION_MODULES) -> dict:
    digests = {}
    for name in sorted(names):
        hasher = hashlib.sha256()
        with open(Path(repository_root) / name, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(block)
        digests[name] = hasher.hexdigest()
    return digests


def freeze_identity(payload: dict) -> str:
    """The digest of the Phase 11 implementation freeze.

    Over the logical freeze document only: versions, module bytes, model
    and sampler identity, statistics and runtime configuration. No path, no
    timestamp, no volume — those are diagnostics, never identity.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def implementation_freeze(
    repository_root: Path,
    *,
    belief_head_digest: str,
    model_state_digest: str,
    contract_bundle_digest: str,
    validation_bank_digest: str,
    test_bank_digest: str,
    runtime: dict,
    bound_evidence: dict,
) -> dict:
    """The single frozen Phase 11 implementation identity, as a document."""
    from ..training.phase11_contract import (
        BOOTSTRAP_CONFIDENCE,
        BOOTSTRAP_REPLICATES,
        ECE_SPECIFICATION,
        WORLD_BASELINE_VERSION,
    )
    from .phase11_recompute import RECOMPUTE_VERSION

    document = {
        "freeze_version": PIPELINE_VERSION,
        "final_test_entry_point": FINAL_TEST_ENTRY_POINT,
        "pipeline_stages": list(PIPELINE_STAGES),
        "sealed_banks": list(SEALED_BANKS),
        "belief_head_identity": {
            "belief_head_digest": belief_head_digest,
            "model_state_digest": model_state_digest,
            "belief_owner": "stratego.evaluation.phase11_belief.Phase11BeliefOwner",
            "request_type": "stratego.evaluation.phase11_belief.Phase11BeliefRequest",
        },
        "evaluator_version": EVALUATOR_VERSION,
        "remaining_count_baseline": REMAINING_COUNT_BASELINE_VERSION,
        "world_baseline": WORLD_BASELINE_VERSION,
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "sampler_entry_point": "stratego.evaluation.phase11_sampler.sample_belief_world",
        "information_safety_version": bound_evidence["information_safety_version"],
        "statistics_version": {
            "bootstrap_replicates": int(BOOTSTRAP_REPLICATES),
            "bootstrap_confidence": float(BOOTSTRAP_CONFIDENCE),
            "resampling_unit": "logical paired case, both colour games pooled",
            "ece_bins": int(ECE_SPECIFICATION["bins"]),
            "independent_recompute_version": RECOMPUTE_VERSION,
        },
        "runtime_backend": {
            "backend": runtime["backend"],
            "dtype": runtime["dtype"],
            "torch_threads": int(runtime["torch_threads"]),
            "process_model": runtime["process_model"],
            "measured_p95_forward_64_ms": float(runtime["p95_forward_64_ms"]),
            "ceiling_ms": float(runtime["ceiling_ms"]),
        },
        "sample_schedule": {
            "decisions_per_game": SAMPLE_DECISIONS_PER_GAME,
            "world_ordinals_per_request": SAMPLE_WORLD_ORDINALS,
            "spacing_rule": "floor(k * E / n) over the eligible recorded decisions",
            "world_floor": SAMPLE_WORLD_FLOOR,
        },
        "contract_bundle_digest": contract_bundle_digest,
        "bank_digests": {
            "phase11_validation_bank_v1": validation_bank_digest,
            "phase11_test_bank_v1": test_bank_digest,
        },
        "module_sha256": module_sha256(repository_root),
        "bound_evidence": bound_evidence["artifacts"],
    }
    document["freeze_digest"] = freeze_identity(document)
    return document


__all__ = [
    "BANK_BINDINGS",
    "FINAL_TEST_ENTRY_POINT",
    "FROZEN_IMPLEMENTATION_MODULES",
    "PIPELINE_STAGES",
    "PIPELINE_VERSION",
    "Phase11PipelineError",
    "Phase11SealError",
    "SAMPLE_DECISIONS_PER_GAME",
    "SAMPLE_WORLD_FLOOR",
    "SAMPLE_WORLD_ORDINALS",
    "SEALED_BANKS",
    "STORE_CONTENT_FIELDS",
    "STORE_CONTENT_MANIFEST_FIELDS",
    "assert_seal",
    "bank_binding",
    "bank_metrics",
    "build_owners",
    "evaluate_all_gates",
    "freeze_identity",
    "frozen_sample_schedule",
    "gate_quantities",
    "generate_bank",
    "implementation_freeze",
    "module_sha256",
    "sampler_checks",
    "schedule_accounting",
    "run_phase11_pipeline",
    "scored_blocks",
    "store_content_digest",
]
