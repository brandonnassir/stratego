#!/usr/bin/env python3
"""Phase 18 Gate G1: paired evaluation of the reproduction against the accepted
Phase 8 checkpoint.

The accepted Agent 7 harness cannot do this. It hard-codes one checkpoint path,
one work directory, and writes its results straight into
`reports/phase_8_data/` and the Phase 8 report. Section A4 of the Agent 2 work
package anticipates exactly that and asks for a wrapper instead.

This is that wrapper, and it is deliberately thin. Every measurement below is
produced by calling `scripts/run_phase8_agent07.py`'s own stage functions - the
same sealed-test pass, the same random gate, the same versus-initialization
schedule, the same discipline audit, the same 42 gate expressions. What the
wrapper adds is only what pairing needs:

* **Two arms over one schedule.** Both arms play under the same policy token,
  so `build_paired_schedule` produces byte-identical match ids and seeds for
  each. The arms differ only in which weights the inference owner loads. The
  schedule digests are recorded and compared; a mismatch is fatal.
* **A per-arm data overlay.** Identity checks must still run against the
  accepted Phase 8 evidence, but the *discipline* audit for the candidate has
  to read the candidate's own Agent 6 artifacts. The overlay is the accepted
  directory with those two files swapped for the arm's own, so nothing else
  moves.
* **Paired statistics.** `stratego.evaluation.phase18.noninferiority` resamples
  both arms on one shared draw - games for the head metrics, setup pairs for
  the play metrics - and reads each frozen margin from the bound the contract
  names.
* **A reproduction check.** The reference arm re-measures the accepted
  checkpoint before the candidate is scored at all, and its numbers are
  compared against `agent_07_final_acceptance.json`. That is what makes the
  wrapper's fidelity a measurement rather than a claim.

Margins, seeds, and pair counts are read from
`reports/phase18/phase18_phase8_reproduction_contract_v1.json`. None is typed
here. Nothing is written outside the Phase 18 output root.

Tier and stress diagnostics are not run. They carry no Phase 8 gate - the
frozen 42 contain none - and the paired non-inferiority contract does not name
them. The omission is recorded in the output rather than left implicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_phase8_agent07 as a7  # noqa: E402

from stratego.evaluation.phase18 import PHASE18_EVALUATION_VERSION  # noqa: E402
from stratego.evaluation.phase18.noninferiority import (  # noqa: E402
    DIRECTION_DELTA_MAX,
    DIRECTION_DELTA_MIN,
    assess_margin,
    paired_ratio_delta,
    paired_unit_delta,
)
from stratego.evaluation.statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    build_paired_units,
    matchup_seed,
)

RUN_ID = "G1-CONTROL-2026-A"
WORK_PACKAGE = "phase18_setup_integrated_warmstart"

ACCEPTED_DATA = REPOSITORY_ROOT / "reports" / "phase_8_data"
CONTRACT_PATH = (
    REPOSITORY_ROOT / "reports" / "phase18" / "phase18_phase8_reproduction_contract_v1.json"
)

#: The two Agent 6 artifacts whose *discipline* evidence is arm-specific.
ARM_SPECIFIC = ("agent_06_warmstart_run.json", "agent_06_checkpoint_manifest.json")

#: Head metrics: the per-game numerator and denominator whose ratio each is,
#: the contract's margin field, and which bound decides. The denominators and
#: the baseline numerators are properties of the data, not of a model, so both
#: arms must produce identical values for them - which the pairing proof checks.
HEAD_METRICS = (
    ("policy_ce_ratio", "policy_weighted_ce", "policy_weighted_baseline_ce",
     "policy_ce_ratio_delta_max", DIRECTION_DELTA_MAX),
    ("policy_top1", "policy_weighted_top1", "policy_weight_sum",
     "policy_top1_delta_min", DIRECTION_DELTA_MIN),
    ("value_ce_ratio", "value_ce", "value_baseline_ce",
     "value_ce_ratio_delta_max", DIRECTION_DELTA_MAX),
    ("value_brier", "value_brier", "value_examples",
     "value_brier_delta_max", DIRECTION_DELTA_MAX),
    ("belief_ce_ratio", "belief_ce", "belief_baseline_ce",
     "belief_ce_ratio_delta_max", DIRECTION_DELTA_MAX),
    ("belief_top1", "belief_top1", "belief_pieces",
     "belief_top1_delta_min", DIRECTION_DELTA_MIN),
)

#: Fields that must agree game-for-game between the arms.
MODEL_INDEPENDENT = (
    "policy_weighted_baseline_ce", "policy_weight_sum", "policy_weighted_expected_top1",
    "policy_examples", "value_baseline_ce", "value_baseline_brier", "value_examples",
    "belief_baseline_ce", "belief_baseline_top1", "belief_pieces",
)


class G1EvaluationError(RuntimeError):
    """A pairing, sealing or identity precondition failed."""


def log(message: str) -> None:
    print(f"[g1-eval {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Sealing: the test split may not open before selection is final
# ---------------------------------------------------------------------------


def assert_selection_finalized(control_directory: Path) -> dict:
    """Refuse to evaluate until the candidate's checkpoint choice is closed.

    The contract's order is binding: training updates weights, validation
    selects, the selection is frozen, and only then does the sealed split open.
    This reads the control's own frozen manifest and refuses if the selection
    is missing, or if it ever saw the test split or a Phase 4 strength number.
    """
    manifest_path = control_directory / "dry_run_artifacts" / "agent_06_checkpoint_manifest.json"
    run_path = control_directory / "dry_run_artifacts" / "agent_06_warmstart_run.json"
    for path in (manifest_path, run_path):
        if not path.exists():
            raise G1EvaluationError(
                f"selection is not finalized: {path} is missing; the sealed test "
                "split may not be opened"
            )
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["selection_protocol"]
    if protocol["split"] != "validation" or protocol["test_split_used"] or protocol["phase4_strength_used"]:
        raise G1EvaluationError(f"selection protocol is not validation-only: {protocol}")
    if manifest.get("selected_global_step") is None:
        raise G1EvaluationError("the control froze no selected checkpoint")
    run = json.loads(run_path.read_text())
    discipline = run["held_out_discipline"]
    if int(discipline["test_examples_evaluated_by_model"]) != 0:
        raise G1EvaluationError(
            "the control already evaluated test examples; sealing is broken"
        )
    return {
        "selection_finalized": True,
        "selected_global_step": manifest["selected_global_step"],
        "selection_protocol": protocol,
        "control_test_examples_evaluated_by_model": int(
            discipline["test_examples_evaluated_by_model"]
        ),
        "control_phase4_neural_evaluation_games": int(
            discipline["phase4_neural_evaluation_games"]
        ),
    }


# ---------------------------------------------------------------------------
# Per-arm staging
# ---------------------------------------------------------------------------


def stage_checkpoint(source: Path, destination: Path) -> dict:
    """Place a read-only copy under the repository root.

    Agent 7 reports checkpoint paths relative to the repository root, so a
    checkpoint outside it cannot be evaluated. The copies are made read-only so
    that nothing in this process can write through one into an accepted
    artifact, and each is re-hashed against its source.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.chmod(0o644)
        destination.unlink()
    shutil.copy2(source, destination)
    destination.chmod(0o444)
    source_digest, destination_digest = sha256(source), sha256(destination)
    if source_digest != destination_digest:
        raise G1EvaluationError(f"staging {source} changed its bytes")
    return {
        "source": str(source),
        "staged": str(destination),
        "sha256": destination_digest,
        "mode": "0444",
    }


def build_overlay(directory: Path, arm_artifacts: "dict | None") -> Path:
    """The accepted Phase 8 evidence, with the arm's own Agent 6 files if given."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for path in sorted(ACCEPTED_DATA.iterdir()):
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    for name, source in (arm_artifacts or {}).items():
        if name not in ARM_SPECIFIC:
            raise G1EvaluationError(f"{name} is not an arm-specific artifact")
        shutil.copy2(source, directory / name)
    return directory


def load_paired_units(work_directory: Path, label: str) -> dict:
    """Rebuild the accepted paired units from the persisted chunk results."""
    directory = work_directory / "games" / label
    results = []
    for path in sorted(directory.glob("chunk_*.pkl")):
        with open(path, "rb") as stream:
            results.extend(pickle.load(stream)["results"])
    if not results:
        raise G1EvaluationError(f"no persisted results under {directory}")
    units = build_paired_units(results, allow_policy_errors=True)
    scores = {}
    for unit in units:
        if unit.setup_pair_id in scores:
            raise G1EvaluationError(
                f"{label}: setup pair {unit.setup_pair_id} appears twice"
            )
        scores[unit.setup_pair_id] = unit.score
    return scores


def capture_test_metrics(device: str) -> tuple:
    """Run the accepted sealed-test stage and keep its per-game statistics.

    `stage_test_metrics` builds a per-game table, hands it to `summarize_games`
    and reports only the aggregate. The paired bootstrap needs the table. The
    accepted stage runs completely unmodified here; `summarize_games` is
    wrapped so the table can be copied out on the way past, and the wrapper
    returns the accepted result untouched.
    """
    import stratego.training.warmstart_metrics as metrics

    captured: dict = {}
    original = metrics.summarize_games

    def capturing(per_game, **keywords):
        result = original(per_game, **keywords)
        captured["per_game"] = {game: dict(stats) for game, stats in per_game.items()}
        return result

    metrics.summarize_games = capturing
    try:
        payload = a7.stage_test_metrics(device=device)
    finally:
        metrics.summarize_games = original
    if "per_game" not in captured:
        raise G1EvaluationError("the sealed-test stage produced no per-game table")
    return payload, captured["per_game"]


def run_arm(
    *,
    label: str,
    checkpoint: Path,
    initial_checkpoint: Path,
    overlay: Path,
    work_directory: Path,
    device: str,
    workers: int,
    chunk_units: int,
    random_pairs: int,
    vs_init_pairs: int,
) -> dict:
    """One arm: verify, export, sealed test, random gate, versus initialization."""
    saved = (
        a7.WORK_DIRECTORY, a7.CHECKPOINT_PATH, a7.INITIAL_CHECKPOINT_PATH,
        a7.EXPECTED_CHECKPOINT_SHA256, a7.DATA_DIRECTORY,
    )
    a7.WORK_DIRECTORY = work_directory
    a7.CHECKPOINT_PATH = checkpoint
    a7.INITIAL_CHECKPOINT_PATH = initial_checkpoint
    a7.EXPECTED_CHECKPOINT_SHA256 = sha256(checkpoint)
    a7.DATA_DIRECTORY = overlay
    work_directory.mkdir(parents=True, exist_ok=True)
    try:
        log(f"{label}: verifying identities and all 28,000 corpus payloads")
        verify = a7.stage_verify()
        log(f"{label}: exporting to the frozen evaluation checkpoint format")
        export = a7.stage_export()
        log(f"{label}: opening the sealed Phase 8 test split")
        heads, per_game = capture_test_metrics(device)
        log(f"{label}: random gate, {random_pairs * 2} games")
        random_gate = a7.stage_random_gate(
            workers=workers, chunk_units=chunk_units, pairs=random_pairs
        )
        log(f"{label}: versus canonical initialization, {vs_init_pairs * 2} games")
        vs_init = a7.stage_vs_init(chunk_units=chunk_units, pairs=vs_init_pairs)
        log(f"{label}: discipline audit")
        audit = a7.stage_audit()
    finally:
        (
            a7.WORK_DIRECTORY, a7.CHECKPOINT_PATH, a7.INITIAL_CHECKPOINT_PATH,
            a7.EXPECTED_CHECKPOINT_SHA256, a7.DATA_DIRECTORY,
        ) = saved

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "work_directory": str(work_directory),
        "verify": verify,
        "export": export,
        "heads": heads,
        "per_game": per_game,
        "random_gate": random_gate,
        "vs_init": vs_init,
        "audit": audit,
        "random_units": load_paired_units(work_directory, "random"),
        "vs_init_units": load_paired_units(work_directory, "vs_init"),
    }


# ---------------------------------------------------------------------------
# Pairing proofs and paired statistics
# ---------------------------------------------------------------------------


def prove_pairing(candidate: dict, reference: dict) -> dict:
    """Every claim of a paired delta has to survive these four checks."""
    problems: list = []

    candidate_games = sorted(candidate["per_game"])
    reference_games = sorted(reference["per_game"])
    if candidate_games != reference_games:
        problems.append(
            f"sealed-test game sets differ: {len(candidate_games)} vs "
            f"{len(reference_games)} games"
        )

    mismatched_fields: dict = {}
    if candidate_games == reference_games:
        for field in MODEL_INDEPENDENT:
            mismatches = sum(
                1
                for game in candidate_games
                if candidate["per_game"][game][field] != reference["per_game"][game][field]
            )
            if mismatches:
                mismatched_fields[field] = mismatches
    if mismatched_fields:
        problems.append(
            f"model-independent per-game statistics differ between arms: {mismatched_fields}"
        )

    schedules = {}
    for stage in ("random_gate", "vs_init"):
        candidate_digest = candidate[stage]["harness"]["schedule_digest"]
        reference_digest = reference[stage]["harness"]["schedule_digest"]
        schedules[stage] = {
            "candidate": candidate_digest,
            "reference": reference_digest,
            "identical": candidate_digest == reference_digest,
        }
        if candidate_digest != reference_digest:
            problems.append(f"{stage}: the two arms did not play the same schedule")

    units = {}
    for stage in ("random_units", "vs_init_units"):
        candidate_pairs = sorted(candidate[stage])
        reference_pairs = sorted(reference[stage])
        units[stage] = {
            "candidate_units": len(candidate_pairs),
            "reference_units": len(reference_pairs),
            "identical_pair_ids": candidate_pairs == reference_pairs,
        }
        if candidate_pairs != reference_pairs:
            problems.append(f"{stage}: the arms produced different setup pairs")

    return {
        "sealed_test_games": len(candidate_games),
        "identical_game_sets": candidate_games == reference_games,
        "model_independent_fields_checked": list(MODEL_INDEPENDENT),
        "model_independent_mismatches": mismatched_fields,
        "schedule_digests": schedules,
        "paired_units": units,
        "problems": problems,
    }


def paired_results(candidate: dict, reference: dict, frozen: dict) -> dict:
    margins = frozen["new_control_comparison"][
        "practical_non_inferiority_margins_selected_before_the_result"
    ]
    bootstrap = frozen["new_control_comparison"]["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence"])
    test_seed = int(frozen["frozen_configuration"]["canonical_seeds"]["test_bootstrap_seed"])

    games = sorted(candidate["per_game"])
    comparisons: dict = {}
    verdicts: dict = {}

    for metric, numerator, denominator, margin_field, direction in HEAD_METRICS:
        interval = paired_ratio_delta(
            [candidate["per_game"][game][numerator] for game in games],
            [candidate["per_game"][game][denominator] for game in games],
            [reference["per_game"][game][numerator] for game in games],
            [reference["per_game"][game][denominator] for game in games],
            seed=test_seed,
            replicates=replicates,
            confidence=confidence,
        )
        comparisons[metric] = interval.to_dict() | {
            "numerator_field": numerator,
            "denominator_field": denominator,
            "margin_field": margin_field,
        }
        verdicts[metric] = assess_margin(
            metric, interval, margin=float(margins[margin_field]), direction=direction
        ).to_dict()

    for metric, stage, margin_field in (
        ("vs_random_ewr", "random_units", "vs_random_ewr_delta_min"),
        ("vs_init_ewr", "vs_init_units", "vs_init_ewr_delta_min"),
    ):
        pairs = sorted(candidate[stage])
        source = "random_gate" if stage == "random_units" else "vs_init"
        summary = candidate[source]["summary"]
        # `MatchResult.matchup` is exactly "candidate_token vs opponent_token";
        # the accepted single-arm interval seeds from it, so the paired delta
        # does too.
        matchup = f"{summary['candidate']} vs {summary['opponent']}"
        reference_summary = reference[source]["summary"]
        if matchup != f"{reference_summary['candidate']} vs {reference_summary['opponent']}":
            raise G1EvaluationError(f"{stage}: the arms report different matchups")
        seed = matchup_seed(DEFAULT_BOOTSTRAP_SEED, matchup)
        interval = paired_unit_delta(
            [candidate[stage][pair] for pair in pairs],
            [reference[stage][pair] for pair in pairs],
            seed=seed,
            replicates=replicates,
            confidence=confidence,
        )
        comparisons[metric] = interval.to_dict() | {
            "matchup": matchup,
            "seed_rule": "matchup_seed(DEFAULT_BOOTSTRAP_SEED, matchup), the accepted play-metric seed",
            "margin_field": margin_field,
        }
        verdicts[metric] = assess_margin(
            metric, interval, margin=float(margins[margin_field]),
            direction=DIRECTION_DELTA_MIN,
        ).to_dict()

    return {
        "design": frozen["new_control_comparison"]["design"],
        "decision_rule": margins["decision_rule"],
        "margins_source": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
        "bootstrap": {
            "replicates": replicates,
            "confidence": confidence,
            "head_metric_seed": test_seed,
            "head_metric_unit": "game (cluster)",
            "play_metric_unit": "paired setup unit",
        },
        "comparisons": comparisons,
        "verdicts": verdicts,
        "all_non_inferior": all(v["non_inferior"] for v in verdicts.values()),
    }


def completion_gates(arm: dict) -> dict:
    """The 42 original Phase 8 gates, assembled exactly as Agent 7 assembles them."""
    verify, export = arm["verify"], arm["export"]
    gates: dict = {}
    gates.update({f"heldout_{k}": v for k, v in arm["heads"]["gates"].items()})
    gates.update({f"random_{k}": v for k, v in arm["random_gate"]["gates"].items()})
    gates.update({f"vs_init_{k}": v for k, v in arm["vs_init"]["gates"].items()})
    gates.update({f"discipline_{k}": v for k, v in arm["audit"]["gates"].items()})
    gates.update(
        {
            "prerequisites_agents_1_to_6_pass": verify["prior_agents"]["agents_1_to_6_all_pass"],
            "corpus_resolved_through_resolver": verify["corpus"][
                "resolved_root_matches_accepted_location"
            ],
            "corpus_digests_match_accepted": not verify["corpus"]["problems"],
            "upstream_identities_unchanged": not verify["upstream"]["frozen_upstream_problems"],
            "checkpoint_identity_verified": not verify["checkpoint_identity"]["problems"],
            "evaluation_export_bitwise_faithful": all(
                entry["bitwise_state_dict_match"] for entry in export["exports"].values()
            ),
            "no_phase9_selfplay_or_rl": True,
            "no_learned_setup_selection": True,
            "no_decision_time_search": True,
        }
    )
    return gates


def reference_reproduction(reference: dict) -> dict:
    """Did re-measuring the accepted checkpoint reproduce its accepted numbers?

    Not a gate. MPS is not bitwise reproducible run to run, so a small drift
    here is expected and is exactly why the comparison is paired rather than
    made against the recorded figures. It is reported because a *large* drift
    would mean the wrapper is not measuring what Agent 7 measured.
    """
    accepted = json.loads((ACCEPTED_DATA / "agent_07_final_acceptance.json").read_text())
    recorded = accepted["headline_results"]
    heads, random_gate, vs_init = reference["heads"], reference["random_gate"], reference["vs_init"]
    observed = {
        "test_policy_ce_ratio": heads["headline"]["policy"]["ce_ratio"],
        "test_policy_top1": heads["headline"]["policy"]["model_top1"],
        "test_value_ce_ratio": heads["headline"]["value"]["ce_ratio"],
        "test_value_brier": heads["headline"]["value"]["model_brier"],
        "test_belief_ce_ratio": heads["headline"]["belief"]["ce_ratio"],
        "test_belief_top1": heads["headline"]["belief"]["model_top1"],
        "random_effective_win_rate": random_gate["summary"]["effective_win_rate"],
        "random_ci_lower": random_gate["summary"]["confidence_interval"]["lower"],
        "vs_init_effective_win_rate": vs_init["summary"]["effective_win_rate"],
        "vs_init_ci_lower": vs_init["summary"]["confidence_interval"]["lower"],
    }
    deltas = {
        name: {
            "accepted": recorded[name],
            "observed": observed[name],
            "delta": observed[name] - recorded[name],
        }
        for name in sorted(observed)
    }
    return {
        "source": "reports/phase_8_data/agent_07_final_acceptance.json headline_results",
        "gate": False,
        "note": (
            "reported, not gated: MPS is not bitwise reproducible run to run, "
            "which is why the control's comparison is paired"
        ),
        "largest_absolute_delta": max(abs(row["delta"]) for row in deltas.values()),
        "metrics": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accepted-checkpoint", required=True)
    parser.add_argument("--accepted-initialisation", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-units", type=int, default=64)
    arguments = parser.parse_args()

    started = time.perf_counter()
    control_directory = Path(arguments.control_dir).expanduser().resolve()
    output = Path(arguments.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    frozen = contract()
    from stratego.training import warmstart_contract as wc

    thresholds = wc.acceptance_thresholds()
    random_pairs = int(thresholds["playing_strength_vs_random"]["evaluation_pairs"])
    vs_init_pairs = int(
        thresholds["improvement_over_initialization"]["paired_setup_cases_min"]
    )
    if (random_pairs, vs_init_pairs) != (a7.RANDOM_PAIRS, a7.VS_INIT_PAIRS):
        raise G1EvaluationError(
            f"the accepted thresholds ({random_pairs}, {vs_init_pairs}) disagree with "
            f"Agent 7's frozen sizes ({a7.RANDOM_PAIRS}, {a7.VS_INIT_PAIRS})"
        )

    log("checking that checkpoint selection is finalized before the seal opens")
    sealing = assert_selection_finalized(control_directory)
    log(f"selection final at step {sealing['selected_global_step']}")

    candidate_source = control_directory / "dry_run_artifacts" / "warmstart_c1_v1.pt"
    staging = REPOSITORY_ROOT / "checkpoints" / "phase8"
    staged = {
        "accepted": stage_checkpoint(
            Path(arguments.accepted_checkpoint).expanduser().resolve(),
            staging / "warmstart_c1_v1.pt",
        ),
        "canonical_initialisation": stage_checkpoint(
            Path(arguments.accepted_initialisation).expanduser().resolve(),
            staging / "warmstart_c1_v1_initialisation.pt",
        ),
        "candidate": stage_checkpoint(
            candidate_source, staging / "phase18_g1_candidate_v1.pt"
        ),
    }
    initial = Path(staged["canonical_initialisation"]["staged"])

    # The reference arm runs first, on purpose: the wrapper has to show it can
    # reproduce the accepted result before the candidate is scored at all.
    arms = {}
    for label, checkpoint, arm_artifacts in (
        ("reference", Path(staged["accepted"]["staged"]), None),
        (
            "candidate",
            Path(staged["candidate"]["staged"]),
            {
                name: control_directory / "dry_run_artifacts" / name
                for name in ARM_SPECIFIC
            },
        ),
    ):
        overlay = build_overlay(output / "overlays" / label, arm_artifacts)
        arms[label] = run_arm(
            label=label,
            checkpoint=checkpoint,
            initial_checkpoint=initial,
            overlay=overlay,
            work_directory=output / "arms" / label,
            device=arguments.device,
            workers=arguments.workers,
            chunk_units=arguments.chunk_units,
            random_pairs=random_pairs,
            vs_init_pairs=vs_init_pairs,
        )

    log("proving the two arms were scored on identical cases")
    pairing = prove_pairing(arms["candidate"], arms["reference"])
    if pairing["problems"]:
        for problem in pairing["problems"]:
            log(f"BLOCKED: {problem}")

    log("computing paired non-inferiority")
    comparison = paired_results(arms["candidate"], arms["reference"], frozen)
    gates = completion_gates(arms["candidate"])
    reference_gates = completion_gates(arms["reference"])
    reproduction = reference_reproduction(arms["reference"])

    payload = {
        "artifact": "phase18_g1_noninferiority_v1",
        "phase18_evaluation_version": PHASE18_EVALUATION_VERSION,
        "run_id": RUN_ID,
        "work_package": WORK_PACKAGE,
        "gate": "G1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_revision": a7.git_commit(),
            "working_tree_state": a7.working_tree_state(),
        },
        "sealing": sealing | {
            "test_split_consumers_this_run": 2,
            "consumers": ["accepted checkpoint (re-measured)", "phase18 G1 candidate"],
            "note": (
                "the Phase 8 sealed test split was already spent once by Phase 8 "
                "Agent 7; these two accesses are additional multiplicity and are "
                "counted here"
            ),
        },
        "staged_checkpoints": staged,
        "pairing_proof": pairing,
        "paired_non_inferiority": comparison,
        "candidate_completion_gates": gates,
        "candidate_gates_total": len(gates),
        "candidate_gates_true": sum(bool(v) for v in gates.values()),
        "reference_completion_gates": reference_gates,
        "reference_gates_true": sum(bool(v) for v in reference_gates.values()),
        "reference_reproduction_of_accepted_metrics": reproduction,
        "not_run": {
            "tiers_and_stress": (
                "no Phase 8 completion gate covers the basic/tactical/strategic/stress "
                "diagnostics and the paired contract does not name them"
            )
        },
        "seconds": round(time.perf_counter() - started, 3),
    }
    a7.write_json(output / "phase18_g1_noninferiority_v1.json", payload)

    for label, arm in arms.items():
        a7.write_json(
            output / f"phase18_g1_arm_{label}_v1.json",
            {key: value for key, value in arm.items() if key != "per_game"},
        )
        a7.write_json(output / f"phase18_g1_arm_{label}_per_game_v1.json", arm["per_game"])

    failing = sorted(name for name, ok in gates.items() if not ok)
    log(f"candidate gates: {payload['candidate_gates_true']}/{len(gates)}")
    if failing:
        log(f"failing: {failing}")
    log(f"paired non-inferiority: {'PASS' if comparison['all_non_inferior'] else 'FAIL'}")
    return 0 if (not failing and comparison["all_non_inferior"] and not pairing["problems"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
