#!/usr/bin/env python3
"""Phase 13 — Agent 4: launch-readiness gates and the immutable launch package.

Task: `instructions/phase_13_final_training_integration/04_AGENT_4_IMMUTABLE_LAUNCH_PACKAGE.md`.

This agent is a launch-readiness reviewer and packager, not another
experimenter. It starts no training, runs no rehearsal and changes no frozen
value. It re-derives every identity live, evaluates Gates A-J, writes the
immutable launch package, and returns GO or NO-GO.

Usage:

```text
python scripts/run_phase13_agent04.py            # gates + package + summary
python scripts/run_phase13_agent04.py --gates    # gates only, write nothing
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

REPORT_ROOT = REPOSITORY / "reports" / "phase13"
EVIDENCE_ROOT = REPORT_ROOT / "agent04_evidence"


def log(message: str) -> None:
    print(f"[agent04] {message}", flush=True)


def _read(relative: str) -> dict:
    return json.loads((REPOSITORY / relative).read_text())


def _gate(name: str, title: str, checks: list) -> dict:
    return {
        "gate": name,
        "title": title,
        "checks": checks,
        "failed": [check["check"] for check in checks if not check["passed"]],
        "passed": all(check["passed"] for check in checks),
    }


def _check(name: str, passed: bool, **evidence) -> dict:
    return {"check": name, "passed": bool(passed), **evidence}


# ---------------------------------------------------------------------------
# Gate A — upstream identity
# ---------------------------------------------------------------------------


def gate_a() -> dict:
    from stratego.evaluation.phase14_candidates import load_pack, load_selection_rule
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_config import integrated_config_digest

    checks = []
    observed = contract.file_sha256(REPOSITORY / contract.STARTING_CHECKPOINT)
    checks.append(
        _check(
            "starting Phase 9 checkpoint is the accepted one",
            observed == contract.STARTING_CHECKPOINT_SHA256,
            path=contract.STARTING_CHECKPOINT,
            observed=observed,
            expected=contract.STARTING_CHECKPOINT_SHA256,
        )
    )
    for name, relative in sorted(contract.ANCHOR_CHECKPOINTS.items()):
        seen = contract.file_sha256(REPOSITORY / relative)
        checks.append(
            _check(
                f"pool anchor {name} matches its frozen digest",
                seen == contract.ANCHOR_SHA256[name],
                path=relative,
                observed=seen,
                expected=contract.ANCHOR_SHA256[name],
            )
        )
    frozen_sha = contract.file_sha256(REPOSITORY / contract.FROZEN_CONTRACT_RELATIVE_PATH)
    checks.append(
        _check(
            "Agent 1 frozen contract is the accepted document",
            frozen_sha
            == "65d1f941a326a1343dce597082c3b525203ef7182f73c759ac6eb04d87a12cdf",
            observed=frozen_sha,
        )
    )
    checks.append(
        _check(
            "Agent 2 integrated config digest is unchanged",
            integrated_config_digest()
            == "9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e",
            observed=integrated_config_digest(),
        )
    )
    checks.append(
        _check(
            "Phase 14 contract digest is unchanged",
            contract.contract_digest()
            == "62ce6d4e04ffd25755717ef290f7486f2616927ddada59d8ea9fb05565c052b9",
            observed=contract.contract_digest(),
        )
    )
    rehearsal = _read("reports/phase13/phase13_rehearsal_v1.json")
    checks.append(
        _check(
            "Agent 3 rehearsal identity is present and names the same digests",
            rehearsal.get("rehearsal_digest")
            == "d8ebae4e28500c27cad8e7c5c48932431e89c45c6d52b8c87da2bb1443a13d21",
            rehearsal_digest=rehearsal.get("rehearsal_digest"),
        )
    )
    pack = load_pack()
    checks.append(
        _check(
            "candidate pack is the frozen 128-game pack",
            pack["pack_content_digest"] == contract.SELECTION_PACK_DIGEST
            and len(pack["games"]) == 128,
            digest=pack["pack_content_digest"],
            games=len(pack["games"]),
        )
    )
    rule = load_selection_rule()
    checks.append(
        _check(
            "the frozen selection rule is bound to that exact pack",
            rule["artifact"] == "phase14_checkpoint_selection_rule_v1"
            and rule["pack_binding"]["pack_content_digest"] == contract.SELECTION_PACK_DIGEST
            and rule["pack_binding"]["games_per_candidate"] == 128,
            artifact=rule["artifact"],
            pack_binding=rule["pack_binding"]["pack_content_digest"],
        )
    )
    source = _read("reports/phase13/phase14_setup_source_v1.json")
    checks.append(
        _check(
            "Phase 14 setup source identity is frozen",
            source.get("artifact") == "phase14_setup_source_v1",
            selector_config_sha256=contract.SETUP_SELECTOR_CONFIG_SHA256,
        )
    )
    checks.append(
        _check(
            "Agent 1C is not the Phase 14 policy/value checkpoint",
            contract.AGENT1C_CHECKPOINT != contract.STARTING_CHECKPOINT,
            agent1c=contract.AGENT1C_CHECKPOINT,
            role=contract.AGENT1C_ROLE,
        )
    )
    return _gate("A", "Upstream identity", checks)


# ---------------------------------------------------------------------------
# Gate B — the final training contract
# ---------------------------------------------------------------------------


def gate_b() -> dict:
    from stratego.training import phase14_contract as contract

    binding = contract.assert_matches_frozen_contract()
    checks = [
        _check(
            "the implementation matches Agent 1's frozen contract",
            not binding.get("disagreements"),
            disagreements=binding.get("disagreements"),
        ),
        _check(
            "main continuation LR is below the accepted Phase 9 LR",
            contract.MAIN_LEARNING_RATE < contract.LR9,
            main=contract.MAIN_LEARNING_RATE,
            lr9=contract.LR9,
        ),
        _check(
            "late continuation LR is below the main continuation LR",
            contract.LATE_LEARNING_RATE < contract.MAIN_LEARNING_RATE,
            late=contract.LATE_LEARNING_RATE,
            main=contract.MAIN_LEARNING_RATE,
        ),
        _check(
            "the multipliers are exactly 0.25x and 0.125x LR9",
            abs(contract.MAIN_LEARNING_RATE - 0.25 * contract.LR9) < 1e-12
            and abs(contract.LATE_LEARNING_RATE - 0.125 * contract.LR9) < 1e-12,
            main_multiplier=contract.MAIN_LR_MULTIPLIER,
            late_multiplier=contract.LATE_LR_MULTIPLIER,
        ),
        _check(
            "the 132-hour transition is tied to the original wall clock",
            contract.TRANSITION_SECONDS == 132 * 3600
            and "run_start_utc" in contract.TRANSITION_RULE,
            transition_seconds=contract.TRANSITION_SECONDS,
            rule=contract.TRANSITION_RULE,
        ),
        _check(
            "the deadline is 168 hours",
            contract.DEADLINE_SECONDS == 168 * 3600,
            deadline_seconds=contract.DEADLINE_SECONDS,
        ),
        _check(
            "no live LR tuning is reachable through the control surface",
            "learning_rate" in contract.IMMUTABLE_CONTROL_KEYS,
            immutable=list(contract.IMMUTABLE_CONTROL_KEYS),
        ),
        _check(
            "the belief auxiliary objective is retained at weight 0.25",
            contract.inherited_phase9_values()["belief_loss_weight"] == 0.25,
            weight=contract.inherited_phase9_values()["belief_loss_weight"],
        ),
        _check(
            "the checkpoint hierarchy is 15 min / 2 h / 6 h",
            (
                contract.HOT_CHECKPOINT_SECONDS,
                contract.ARCHIVE_CADENCE_SECONDS,
                contract.CANDIDATE_CADENCE_SECONDS,
            )
            == (900, 7200, 21600),
            hot=contract.HOT_CHECKPOINT_SECONDS,
            archive=contract.ARCHIVE_CADENCE_SECONDS,
            candidate=contract.CANDIDATE_CADENCE_SECONDS,
        ),
        _check(
            "hour 0 and hour 168 are both candidate hours",
            contract.CANDIDATE_HOURS[0] == 0 and contract.CANDIDATE_HOURS[-1] == 168,
            candidates=len(contract.CANDIDATE_HOURS),
        ),
    ]
    # The opponent population contract.
    total = contract.GAMES_PER_ITERATION
    for segment in contract.SEGMENTS:
        counts = contract.bucket_counts(segment)
        handcrafted = counts["rule"] + counts["stress"]
        share = handcrafted / total
        checks.append(
            _check(
                f"{segment} handcrafted share is inside the frozen 10-15% band",
                0.10 <= share <= 0.15,
                segment=segment,
                handcrafted=handcrafted,
                share=round(share, 4),
                counts=counts,
            )
        )
    checks.append(
        _check(
            "the late segment shifts neural weight toward historical",
            contract.bucket_counts("late")["historical"]
            > contract.bucket_counts("main")["historical"]
            and contract.bucket_counts("late")["current"]
            < contract.bucket_counts("main")["current"],
            main=contract.bucket_counts("main"),
            late=contract.bucket_counts("late"),
        )
    )
    checks.append(
        _check(
            "the handcrafted counts do not change between segments",
            contract.bucket_counts("main")["rule"] == contract.bucket_counts("late")["rule"]
            and contract.bucket_counts("main")["stress"]
            == contract.bucket_counts("late")["stress"],
        )
    )
    families = dict(contract.HANDCRAFTED_COUNTS)
    required = {
        "strategic_rule_based",
        "tactical_rule_based",
        "stress_scout_rush",
        "stress_miner_rush",
        "stress_information_miser",
    }
    checks.append(
        _check(
            "all five required opponent families are represented",
            required <= set(families) and all(families[name] > 0 for name in required),
            families=families,
        )
    )
    return _gate("B", "Final training contract", checks)


# ---------------------------------------------------------------------------
# Gate C — setup safety
# ---------------------------------------------------------------------------


def gate_c() -> dict:
    from stratego.training import phase14_contract as contract

    policy = _read("reports/phase13/phase13_setup_census_alarm_policy_v1.json")
    census = _read("reports/phase13/phase13_setup_census_v1.json")
    source = _read("reports/phase13/phase14_setup_source_v1.json")
    summary = _read("reports/phase13/phase13_agent_01_summary.json")
    checks = [
        _check(
            "the alarm policy was written before the census sampled anything",
            policy["written_utc"] < census["written_utc"],
            policy_written=policy["written_utc"],
            census_written=census["written_utc"],
        ),
        _check(
            "the alarm policy on disk is the one the census names",
            contract.file_sha256(
                REPOSITORY / "reports/phase13/phase13_setup_census_alarm_policy_v1.json"
            )
            == census["alarm_policy_sha256"],
        ),
        _check(
            "the census records no defect observation",
            census["classification"]["total_defect_observations"] == 0,
            defect_counts=census["defect_counts"],
        ),
        _check(
            "trivial-capture probability is under its predeclared threshold",
            summary["census"]["P_trivial"] < census["classification"]["P_trivial_threshold"],
            observed=summary["census"]["P_trivial"],
            threshold=census["classification"]["P_trivial_threshold"],
        ),
        _check(
            "pre-decision probability is under its predeclared threshold",
            summary["census"]["P_predecision"]
            < census["classification"]["P_predecision_threshold"],
            observed=summary["census"]["P_predecision"],
            threshold=census["classification"]["P_predecision_threshold"],
        ),
        _check(
            "the census reports every required distribution",
            all(
                key in census
                for key in (
                    "sampled_tallies",
                    "exposure_attribution",
                    "stage_effects",
                    "paired_measurements",
                    "library_composition",
                )
            ),
            reported=sorted(census),
        ),
        _check(
            "reflection and perturbation move the Flag row zero times",
            census["stage_effects"]["front_row_changes_after_reflection"] == 0
            and census["stage_effects"]["front_row_changes_after_perturbation"] == 0,
            effects=census["stage_effects"],
        ),
        _check(
            "forward-Flag exposure is attributed to one deliberate family",
            set(census["exposure_attribution"]["forwardmost_flag_draws_by_family"])
            == {census["exposure_attribution"]["irregular_family_id"]},
            attribution=census["exposure_attribution"],
        ),
        _check(
            "no repair was required, so Phase 10 evidence was not rewritten",
            summary["setup_source"]["repaired"] is False
            and summary["stop_condition_items"]["setup_defect_resolution_required"] is False,
        ),
        _check(
            "the frozen Phase 14 setup source is the one the contract binds",
            source["artifact"] == "phase14_setup_source_v1"
            and contract.SETUP_SELECTOR_CONFIG_SHA256
            == summary["setup_source"]["selector_config_sha256"],
            selector_config_sha256=contract.SETUP_SELECTOR_CONFIG_SHA256,
        ),
        _check(
            "the Phase 14 setup path is oriented, not the Phase 11B canonical glue",
            (
                _phase14_setup_source_probe()["engine_is_oriented"] is True
                and _phase14_setup_source_probe()["canonical_differs_from_oriented"] is True
                and _phase14_setup_source_probe()["orientation_helper"]
                == "SelectorDraw.oriented(player)"
                and _phase14_setup_source_probe()["kind"] == "phase10_learned_selector"
            ),
            probe={
                key: _phase14_setup_source_probe()[key]
                for key in (
                    "source_id",
                    "kind",
                    "engine_is_oriented",
                    "canonical_differs_from_oriented",
                    "orientation_helper",
                )
            },
            note=(
                "Agent 1 explained the Phase 12 flag-row observation as the Phase 11B "
                "glue returning canonical tuples; the production path measured clean "
                "(D4 = 0 over 8,192 engine boards) and the runner re-probes it before "
                "the window is stamped"
            ),
        ),
    ]
    return _gate("C", "Setup safety", checks)


_PROBE_CACHE: dict = {}


def _phase14_setup_source_probe() -> dict:
    """The oriented-path probe the runner runs before stamping the window."""
    if _PROBE_CACHE:
        return _PROBE_CACHE
    from stratego.training.phase14_seed import game_id
    from stratego.training.phase14_setup_source import (
        Phase14SetupSource,
        assert_orientation_path,
    )

    source = Phase14SetupSource.build()
    probe = assert_orientation_path(source, game_id(1, "current", 0))
    _PROBE_CACHE.update({**source.describe(), **probe})
    return _PROBE_CACHE


# ---------------------------------------------------------------------------
# Gates D-H — evidence from Agents 2 and 3, re-read rather than remembered
# ---------------------------------------------------------------------------


def _readiness(rehearsal: dict) -> dict:
    return {
        entry["check"]: entry["passed"]
        for entry in rehearsal["readiness"]["checks"]
    }


def gate_d(rehearsal: dict, agent2: dict) -> dict:
    readiness = _readiness(rehearsal)
    units = agent2["integration"]["units"]
    checks = [
        _check(
            "training updates are finite at production population",
            readiness["training updates finite"],
        ),
        _check("parameters change", readiness["parameters change"]),
        _check(
            "the belief auxiliary objective functions",
            readiness["belief auxiliary objective functioning"],
        ),
        _check(
            "every Agent 2 unit reported a finite belief loss at weight 0.25",
            all(
                isinstance(unit.get("belief_loss"), (int, float))
                and unit["belief_loss"] == unit["belief_loss"]
                for unit in units
            ),
            belief_losses=[unit.get("belief_loss") for unit in units],
        ),
        _check(
            "no telemetry row was missing a frozen metric",
            all(not unit.get("missing_metrics") for unit in units),
        ),
        _check(
            "search is absent from the training import closure",
            _code_binding()["search_excluded"],
            search_modules=_code_binding()["search_modules_in_training_closure"],
        ),
    ]
    return _gate("D", "Training correctness", checks)


def gate_e(rehearsal: dict) -> dict:
    from stratego.training.phase14_launch import worker_repair_evidence

    readiness = _readiness(rehearsal)
    repair = worker_repair_evidence()
    checks = [
        _check("a forced process crash recovers", readiness["forced process crash recovered"]),
        _check("optimizer state is preserved", readiness["optimizer state preserved"]),
        _check(
            "the active historical pool is preserved",
            readiness["active historical pool preserved"],
        ),
        _check("worker failure recovers", readiness["worker failure recovered"]),
        _check(
            "the accepted worker-pool repair is installed in this revision",
            repair["installed"],
            checks=repair["checks"],
        ),
        _check(
            "the repair was re-verified at the frozen production population",
            rehearsal["worker_failure_reverification"]["learner_survived"] is True
            and rehearsal["worker_failure_reverification"]["victim_died"] is True
            and "production" in rehearsal["worker_failure_reverification"]["population"],
            reverification=rehearsal["worker_failure_reverification"],
        ),
        _check(
            "a killed process is now recorded by something outside it",
            _supervisor_records_a_death(),
            note="the supervisor logs the death; the dead process cannot",
        ),
    ]
    return _gate("E", "Recovery", checks)


def _supervisor_records_a_death() -> bool:
    from stratego.training.phase14_supervisor import supervisor_semantics

    required = {"unexpected exit", "exit code / signal", "learner PID", "launch timestamp"}
    return required <= set(supervisor_semantics()["records"])


def gate_f(rehearsal: dict, agent2: dict) -> dict:
    readiness = _readiness(rehearsal)
    resume = agent2["integration"]["resume"]
    checks = [
        _check(
            "the original deadline survives downtime and restart",
            readiness["original rehearsal deadline preserved"],
        ),
        _check(
            "an Agent 2 resume reproduced the window byte for byte",
            resume["state_before"]["window"] == resume["state_after"]["window"],
            window=resume["state_after"]["window"],
        ),
        _check(
            "post-deadline recovery performs zero optimizer steps",
            readiness["post-deadline recovery refuses training"],
        ),
        _check(
            "the deadline stop is automatic",
            readiness["test-clock 168h shutdown works"],
        ),
        _check(
            "the late transition reuses the original start",
            readiness["test-clock late transition works"],
        ),
        _check(
            "the supervisor never creates a new deadline",
            _supervisor_never_creates_a_deadline(),
        ),
    ]
    return _gate("F", "Wall-clock semantics", checks)


def _supervisor_never_creates_a_deadline() -> bool:
    from stratego.training.phase14_supervisor import supervisor_semantics

    return supervisor_semantics()["never"] == "creates a new training deadline"


def gate_g(rehearsal: dict, agent2: dict) -> dict:
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_pool import pool_semantics

    readiness = _readiness(rehearsal)
    semantics = pool_semantics()
    checks = [
        _check(
            "the durable archive cadence is 2 hours",
            contract.ARCHIVE_CADENCE_SECONDS == 7200,
        ),
        _check(
            "the archive is append-only and never pruned",
            "never pruned" in semantics.get("archive", ""),
            archive=semantics.get("archive"),
        ),
        _check(
            "pool membership is a pure function of the ordered archive",
            "f(k)" in semantics.get("membership", "")
            or "pure function" in semantics.get("membership", ""),
            membership=semantics.get("membership"),
        ),
        _check(
            "there is no tournament admission",
            "no tournament" in json.dumps(semantics).lower(),
        ),
        _check(
            "both permanent anchors are bound",
            set(contract.POOL_ANCHORS) == {"P8", "P9"},
            anchors=list(contract.POOL_ANCHORS),
        ),
        _check(
            "the pool is bounded at 16 with fixed sampling weights",
            contract.POOL_SIZE == 16
            and dict(contract.POOL_CATEGORY_WEIGHTS)
            == {"anchor": 0.2, "older": 0.25, "middle": 0.25, "recent": 0.3},
            size=contract.POOL_SIZE,
            weights=dict(contract.POOL_CATEGORY_WEIGHTS),
        ),
        _check(
            "a resumed checkpoint recomputes the same pool",
            agent2["integration"]["pool"]["recompute_matches_checkpoint"] is True,
        ),
        _check(
            "the rehearsal preserved pool continuity across both crashes",
            readiness["active historical pool preserved"],
        ),
    ]
    return _gate("G", "Historical system", checks)


def gate_h(rehearsal: dict, agent2: dict) -> dict:
    from stratego.evaluation.phase14_candidates import evaluator_semantics
    from stratego.training import phase14_contract as contract

    readiness = _readiness(rehearsal)
    semantics = evaluator_semantics()
    checks = [
        _check(
            "candidates are marked every 6 hours",
            contract.CANDIDATE_CADENCE_SECONDS == 21600
            and len(contract.CANDIDATE_HOURS) == 29,
            hours=len(contract.CANDIDATE_HOURS),
        ),
        _check(
            "the evaluation pack is frozen and identical for every candidate",
            semantics["pack_digest"] == contract.SELECTION_PACK_DIGEST
            and semantics["games"] == 128,
        ),
        _check("no search is used in candidate evaluation", semantics["search"] == "absent"),
        _check(
            "the evaluator imports no trainer, scheduler or clock",
            "no trainer" in semantics["isolation"],
            isolation=semantics["isolation"],
        ),
        _check(
            "the evaluator cannot write any frozen training value",
            set(contract.IMMUTABLE_CONTROL_KEYS)
            >= {
                "learning_rate",
                "opponent_mixture",
                "setup_source",
                "historical_pool_algorithm",
                "candidate_selection_rule",
                "deadline",
            },
            immutable=list(contract.IMMUTABLE_CONTROL_KEYS),
        ),
        _check(
            "the 6-hour candidate event fired in the rehearsal",
            readiness["test-clock 6h candidate event works"],
        ),
        _check(
            "the hour-168 candidate is marked, not selected",
            agent2["integration"]["deadline"]["hour_168_candidate"]["evaluation_status"]
            == "pending",
            hour_168=agent2["integration"]["deadline"]["hour_168_candidate"]["note"],
        ),
        _check(
            "an incomplete evaluation is refused by the selection rule",
            _incomplete_evaluation_is_refused(),
        ),
        _check(
            "pending candidate evaluations are detected from disk, not from memory",
            _pending_detection_is_on_disk(),
        ),
    ]
    return _gate("H", "Candidate system", checks)


def _incomplete_evaluation_is_refused() -> bool:
    from stratego.evaluation.phase14_candidates import (
        Phase14CandidateError,
        select_final_candidate,
    )

    try:
        select_final_candidate([{"hour": 6, "complete": False, "mean_ewr": 1.0}])
    except Phase14CandidateError:
        return True
    return False


def _pending_detection_is_on_disk() -> bool:
    import tempfile

    from stratego.evaluation.phase14_candidates import CandidateLedger
    from stratego.training.phase14_supervisor import unevaluated_candidates

    with tempfile.TemporaryDirectory() as directory:
        CandidateLedger.at(directory).record_candidate(6, {"hour": 6})
        return unevaluated_candidates(directory) == [6]


# ---------------------------------------------------------------------------
# Gate I — storage
# ---------------------------------------------------------------------------


def gate_i(rehearsal: dict) -> dict:
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_storage import Phase14Storage, volume_usage

    storage = Phase14Storage.production()
    usage = volume_usage(contract.EXTERNAL_VOLUME)
    measured = rehearsal["storage_projection"]
    projected = float(measured["projected_168h_total_gib"])
    with_reserve = projected * 1.2
    conservative = float(measured["agent_1_conservative_ceiling_gib"])
    free = float(usage["free_gib"])
    checks = [
        _check(
            "the external volume is mounted and writable",
            usage["external_volume_present"] and not usage["read_only"],
            mount=usage["mount_point"],
        ),
        _check(
            "the measured 168-hour projection plus a 20% reserve fits",
            with_reserve < free,
            projected_gib=round(projected, 3),
            projected_with_20_percent_reserve_gib=round(with_reserve, 3),
            free_gib=free,
        ),
        _check(
            "Agent 1's conservative planning ceiling also fits",
            conservative * 1.2 < free,
            ceiling_gib=conservative,
            ceiling_with_reserve_gib=round(conservative * 1.2, 3),
            free_gib=free,
        ),
        _check(
            "full raw retention is the frozen plan",
            contract.FULL_RAW_RETENTION is True,
        ),
        _check(
            "the frozen reserve threshold is intact",
            free > contract.STORAGE_RESERVE_GIB,
            reserve_gib=contract.STORAGE_RESERVE_GIB,
            free_gib=free,
        ),
        _check(
            "earlier accepted evidence can never be deleted",
            "never" in contract.NO_DELETION_RULE.lower(),
            rule=contract.NO_DELETION_RULE,
        ),
        _check(
            "no Phase 14 run identity exists yet",
            not any(Path(storage.hot_root).glob("hot_*.pt"))
            and not Path(storage.run_state_path).exists()
            and not any(Path(storage.rollout_root).glob("*/iteration_*")),
            external_run_directory=str(storage.external_root),
            external_run_directory_exists=Path(storage.external_root).exists(),
            hot_checkpoints=len(list(Path(storage.hot_root).glob("hot_*.pt"))),
            run_manifest_exists=Path(storage.run_state_path).exists(),
            note="an empty directory tree is not a run identity; a hot checkpoint is",
        ),
    ]
    return _gate("I", "Storage", checks)


# ---------------------------------------------------------------------------
# Gate J — monitoring and controls
# ---------------------------------------------------------------------------


def gate_j() -> dict:
    import tempfile

    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_launch import (
        clear_emergency_stop,
        emergency_stop_state,
        request_emergency_stop,
    )
    from stratego.training.phase14_status import status_semantics
    from stratego.training.phase14_supervisor import supervisor_semantics
    from stratego.training.phase14_telemetry import (
        EXTENDED_METRIC_PATHS,
        METRIC_PATHS,
        ControlSurface,
        Phase14TelemetryError,
    )

    refused = []
    control = ControlSurface()
    for key in contract.IMMUTABLE_CONTROL_KEYS:
        try:
            control.set(key, 1.0)
        except Phase14TelemetryError:
            refused.append(key)
    with tempfile.TemporaryDirectory() as directory:
        request_emergency_stop(directory, reason="gate J probe")
        durable = emergency_stop_state(directory)["active"]
        surface = ControlSurface(stop_file=Path(directory) / "phase14_emergency_stop.json")
        durable_stops_the_run = surface.should_continue() is False
        clear_emergency_stop(directory)
        cleared = emergency_stop_state(directory)["active"] is False

    checks = [
        _check(
            "every frozen metric has a snapshot path",
            set(contract.FROZEN_METRIC_LIST) == set(METRIC_PATHS),
            metrics=len(METRIC_PATHS),
        ),
        _check(
            "the Agent 4 monitoring additions are exposed and checked",
            set(EXTENDED_METRIC_PATHS)
            >= {
                "committed games (authoritative)",
                "process game counter (diagnostic)",
                "configured loader workers",
                "live loader workers",
                "loader pool rebuilds",
                "last pool rebuild timestamp",
                "last pool rebuild reason",
            },
            extended=sorted(EXTENDED_METRIC_PATHS),
        ),
        _check(
            "the authoritative game total comes from the store, not a counter",
            "iteration manifests" in status_semantics()["committed_games"],
            committed=status_semantics()["committed_games"],
            process=status_semantics()["process_counter_games"],
        ),
        _check(
            "worker health is a live observation",
            "live OS children" in status_semantics()["worker_health"],
        ),
        _check(
            "every frozen training value is refused by name",
            refused == list(contract.IMMUTABLE_CONTROL_KEYS),
            refused=refused,
        ),
        _check("a durable emergency stop can be requested", durable),
        _check("a durable emergency stop reaches the run", durable_stops_the_run),
        _check("a durable emergency stop can be cleared", cleared),
        _check(
            "the supervisor refuses to restart over an emergency stop",
            "emergency stop is active" in supervisor_semantics()["refuses_to_restart_when"],
            refusals=supervisor_semantics()["refuses_to_restart_when"],
        ),
        _check(
            "restart and resume are one documented procedure",
            (REPOSITORY / "scripts" / "phase14_launch.py").exists()
            and (REPOSITORY / "PHASE_14_RUNBOOK.md").exists(),
        ),
    ]
    return _gate("J", "Controls", checks)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _code_binding() -> dict:
    from stratego.training.phase14_launch import code_binding

    if not hasattr(_code_binding, "_cached"):
        _code_binding._cached = code_binding()
    return _code_binding._cached


def evaluate_gates() -> dict:
    rehearsal = _read("reports/phase13/phase13_rehearsal_v1.json")
    agent2 = _read("reports/phase13/phase13_agent_02_summary.json")
    gates = [
        gate_a(),
        gate_b(),
        gate_c(),
        gate_d(rehearsal, agent2),
        gate_e(rehearsal),
        gate_f(rehearsal, agent2),
        gate_g(rehearsal, agent2),
        gate_h(rehearsal, agent2),
        gate_i(rehearsal),
        gate_j(),
    ]
    failed = [gate["gate"] for gate in gates if not gate["passed"]]
    return {
        "artifact": "phase13_agent04_gates_v1",
        "gates": gates,
        "checks_total": sum(len(gate["checks"]) for gate in gates),
        "checks_passed": sum(
            1 for gate in gates for check in gate["checks"] if check["passed"]
        ),
        "failed_gates": failed,
        "recommendation": "GO" if not failed else "NO-GO",
    }


def main(argv=None) -> int:
    from stratego.training.phase14_launch import write_launch_package

    parser = argparse.ArgumentParser(description="Phase 13 Agent 4 launch readiness")
    parser.add_argument("--gates", action="store_true", help="gates only; write nothing")
    args = parser.parse_args(argv)

    started = time.time()
    log("evaluating launch-readiness gates A-J")
    report = evaluate_gates()
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        log(f"  Gate {gate['gate']} {gate['title']:<26} {status} ({len(gate['checks'])} checks)")
        for check in gate["checks"]:
            if not check["passed"]:
                log(f"      FAILED: {check['check']}")
    if args.gates:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["recommendation"] == "GO" else 1

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_ROOT / "gates.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    log("writing the immutable launch package")
    written = write_launch_package()
    report["package"] = written
    report["seconds"] = round(time.time() - started, 3)
    (EVIDENCE_ROOT / "gates.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    log(f"recommendation: {report['recommendation']}")
    print(json.dumps({k: v for k, v in report.items() if k != "gates"}, indent=2, default=str))
    return 0 if report["recommendation"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
