"""Phase 18 Stage 6B: the minimal G1 evaluator adapter for joint bundles
(G3-ENG-04) and the candidate-versus-control analysis (G3-ENG-05).

```text
cases        160 library validation bases (400..409, ten per family) x the
             eight frozen handcrafted opponents x two colours = 2,560 per arm;
             case index = ((base_slot * 8) + opponent) * 2 + colour, so the
             index parity is the candidate's colour
own setup    every arm resolves ITS OWN setup per case from ITS OWN bundle's
             setup model (the EMA; equal to the initial model in the control)
             through the accepted generate_pool path under the shared case
             seeds (purpose 'g3_evaluation'; identical uniforms in every arm,
             different models give different boards)
opponent     the library base, oriented to the opponent's colour
schedule     one MatchSpec per case under EVALUATION_RULES (P18-A001), with a
             policy token shared by every arm, so match ids, seeds, colours,
             opponents and opponent formations are identical across arms and
             the schedule digest is one number (gate G7)
play         the accepted G1 harness: InferenceOwner + run_neural_schedule in
             retry-safe chunks, one immutable receipt per game, and the
             planned = completed + failed + missing accounting
analysis     per case EWR (draw = 0.5), paired candidate - control difference,
             per-base means, a stratified cluster bootstrap over bases within
             families with the finite-stratum rescaling sqrt(n_f / (n_f - 1)),
             PROCEED iff the 95% lower bound > 0 and the point >= 0.05
```

The bundle id is stamped into every receipt and the arm record, not into the
policy token: a per-arm token would change every match id and seed and break
the pairing the design requires (section 4.1, gate G7).
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...engine.constants import BLUE, PLAYER_NAMES, RED
from ...engine.setup import serialize_setup, validate_setup
from ...evaluation.match_runner import ERROR_ILLEGAL_ACTION, ON_POLICY_ERROR_QUARANTINE
from ...evaluation.match_spec import MatchSpec, rules_token, schedule_digest
from ...evaluation.neural_worker import (
    BATCH_POLICY_SINGLE,
    DECISION_MODE_GREEDY,
    NEURAL_WORKER_VERSION,
    neural_policy_ref,
)
from ...evaluation.registry import policy_ref
from ...evaluation.setup_bank import SetupBank, SetupPair
from ...setups.contracts import LIBRARY_JSONL_PATH
from ...setups.identity import class_fingerprint, content_fingerprint, orient_setup
from ...setups.library import read_library_jsonl
from .g3_bundle import C1_NAME, load_setup_trainer, read_manifest, verify_bundle
from .g3_contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    EVALUATION_BASE_INDICES,
    G3_EVALUATION_VERSION,
    HANDCRAFTED_OPPONENTS,
    PLAY_EVALUATION_RULES,
    PRIMARY_MARGIN,
    Phase18G3Error,
    Phase18G3LineageError,
    PilotConfig,
    assert_base_index_is_evaluable,
    evaluation_bootstrap_seed,
    evaluation_schedule_seed,
)
from .setup_contract import file_sha256
from .setup_model import state_dict_digest
from .setup_sampling import generate_pool

G3_EVALUATION_BANK_VERSION = "phase18_g3_evaluation_bank_v1"
G3_EVALUATION_BANK_FAMILY = "phase18_g3_own_setup_vs_library_base_v1"
#: One token for every arm (design 4.1 / gate G7); the bundle id lives in the receipts.
CANDIDATE_TOKEN_ID = "g3_bundle"
GATE_DTYPE = "float32"
OWN_SETUP_PURPOSE = "g3_evaluation"


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationCase:
    case_index: int
    base_slot: int
    family_id: str
    base_index: int
    base_setup_id: str
    opponent_index: int
    opponent_id: str
    colour: int
    base_canonical: tuple
    base_content_fingerprint: str
    base_class_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "case_index": int(self.case_index),
            "base_slot": int(self.base_slot),
            "family_id": self.family_id,
            "base_index": int(self.base_index),
            "base_setup_id": self.base_setup_id,
            "opponent_index": int(self.opponent_index),
            "opponent_id": self.opponent_id,
            "colour": int(self.colour),
            "colour_name": PLAYER_NAMES[self.colour],
            "base_content_fingerprint": self.base_content_fingerprint,
            "base_class_fingerprint": self.base_class_fingerprint,
        }


def load_evaluation_bases(library_path=LIBRARY_JSONL_PATH, *, base_indices=EVALUATION_BASE_INDICES, families=None) -> list:
    """The opponent formations: validation bases 400..409 of every family.

    Every index is checked through `assert_base_index_is_evaluable`, so the
    reserved 410..449 and the sealed 450..499 cannot be opened by mistake.
    """
    wanted = {assert_base_index_is_evaluable(index) for index in base_indices}
    entries = [
        entry
        for entry in read_library_jsonl(library_path)
        if entry.base_index in wanted and (families is None or entry.family_id in set(families))
    ]
    entries.sort(key=lambda entry: (entry.family_id, entry.base_index))
    if not entries:
        raise Phase18G3Error("no evaluation base was selected")
    counts: dict = {}
    for entry in entries:
        counts[entry.family_id] = counts.get(entry.family_id, 0) + 1
    if len(set(counts.values())) != 1:
        raise Phase18G3Error(f"families hold unequal base counts: {counts}")
    return entries


def build_cases(bases, *, opponents=HANDCRAFTED_OPPONENTS) -> list:
    """Every (base, opponent, colour) case; the index parity is the colour."""
    for opponent_id in opponents:
        policy_ref(opponent_id)  # unknown ids refuse here, before any game
    cases = []
    for base_slot, entry in enumerate(bases):
        assert_base_index_is_evaluable(entry.base_index)
        canonical = tuple(int(v) for v in entry.canonical_setup)
        for opponent_index, opponent_id in enumerate(opponents):
            for colour in (RED, BLUE):
                index = ((base_slot * len(opponents)) + opponent_index) * 2 + colour
                assert index % 2 == colour
                cases.append(
                    EvaluationCase(
                        case_index=index,
                        base_slot=base_slot,
                        family_id=entry.family_id,
                        base_index=int(entry.base_index),
                        base_setup_id=entry.base_setup_id,
                        opponent_index=opponent_index,
                        opponent_id=opponent_id,
                        colour=colour,
                        base_canonical=canonical,
                        base_content_fingerprint=content_fingerprint(canonical),
                        base_class_fingerprint=class_fingerprint(canonical),
                    )
                )
    return cases


def cases_digest(cases) -> str:
    hasher = hashlib.sha256()
    for case in cases:
        hasher.update(json.dumps(case.to_dict(), sort_keys=True, separators=(",", ":")).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# One arm's own setups, bank and schedule
# ---------------------------------------------------------------------------


def resolve_own_setups(setup_model, cases, *, namespace: str, seed_index: int, device: str = "cpu"):
    """One own setup per case from `setup_model`, under the shared case seeds.

    `generate_pool` assigns the red lane to even indices and the blue lane to
    odd ones, which is exactly the case colour parity; each sample is checked
    against its case.
    """
    generation = generate_pool(
        setup_model,
        namespace=namespace,
        seed_index=int(seed_index),
        snapshot_iteration=0,
        snapshot_digest=state_dict_digest(setup_model),
        count=len(cases),
        device=device,
        purpose=OWN_SETUP_PURPOSE,
    )
    for case, sample in zip(cases, generation.samples):
        if int(sample.index) != int(case.case_index) or int(sample.lane) != int(case.colour):
            raise Phase18G3Error(f"case {case.case_index}: the sampled own setup is lane {sample.lane}, index {sample.index}")
    if generation.telemetry["legality_failures"] or generation.telemetry["orientation_failures"]:
        raise Phase18G3Error("an own setup failed legality or orientation (gates G1 / G2)")
    return generation


def build_arm_bank(cases, own_samples) -> SetupBank:
    """The arm's board per case: the own setup on the candidate's colour, the
    library base oriented to the opponent's colour."""
    pairs = []
    for case, sample in zip(cases, own_samples):
        base_engine = validate_setup(orient_setup(case.base_canonical, RED if case.colour == BLUE else BLUE), RED if case.colour == BLUE else BLUE)
        own_engine = validate_setup(tuple(int(v) for v in sample.engine_setup), case.colour)
        red_setup, blue_setup = (own_engine, base_engine) if case.colour == RED else (base_engine, own_engine)
        pairs.append(
            SetupPair(
                setup_pair_id=int(case.case_index),
                red_setup=red_setup,
                blue_setup=blue_setup,
                generation_seed=int(sample.root_seed),
                bank_version=G3_EVALUATION_BANK_VERSION,
                generation_family=G3_EVALUATION_BANK_FAMILY,
            )
        )
    return SetupBank(bank_version=G3_EVALUATION_BANK_VERSION, root_seed=0, generation_family=G3_EVALUATION_BANK_FAMILY, pairs=tuple(pairs))


def candidate_policy_ref():
    return neural_policy_ref(CANDIDATE_TOKEN_ID, dtype_name=GATE_DTYPE)


def build_schedule(cases, *, namespace: str) -> tuple:
    """One MatchSpec per case; identical for every arm of the pilot."""
    candidate = candidate_policy_ref()
    root_seed = evaluation_schedule_seed(namespace)
    matches = []
    for case in cases:
        matches.append(
            MatchSpec(
                candidate=candidate,
                opponent=policy_ref(case.opponent_id),
                setup_pair_id=int(case.case_index),
                candidate_color=int(case.colour),
                replicate=0,
                root_seed=root_seed,
                setup_bank_version=G3_EVALUATION_BANK_VERSION,
                rules=PLAY_EVALUATION_RULES,
            )
        )
    return tuple(matches)


def schedule_record(cases, matches, *, namespace: str) -> dict:
    return {
        "evaluation_version": G3_EVALUATION_VERSION,
        "digest": schedule_digest(matches),
        "matches": len(matches),
        "cases": len(cases),
        "cases_digest": cases_digest(cases),
        "bases": len({case.base_slot for case in cases}),
        "families": sorted({case.family_id for case in cases}),
        "opponents": list(dict.fromkeys(case.opponent_id for case in cases)),
        "candidate_token": candidate_policy_ref().token,
        "root_seed": evaluation_schedule_seed(namespace),
        "rules": rules_token(PLAY_EVALUATION_RULES),
        "decision_mode": DECISION_MODE_GREEDY,
        "dtype": GATE_DTYPE,
        "batch_policy": BATCH_POLICY_SINGLE,
        "neural_worker_version": NEURAL_WORKER_VERSION,
        "own_setup_purpose": OWN_SETUP_PURPOSE,
        "note": "every arm plays this schedule under one shared token; the bundle id is stamped in the receipts",
    }


# ---------------------------------------------------------------------------
# Component matching (gate G5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentTag:
    """Which bundle a component came from; two tags must agree to be paired."""

    component: str
    bundle_id: str
    run_id: str
    lineage: str
    period: int


def component_tag(manifest: dict, component: str) -> ComponentTag:
    return ComponentTag(component=component, bundle_id=manifest["bundle_id"], run_id=manifest["run_id"], lineage=manifest["lineage"], period=int(manifest["period"]))


def assert_internally_matched(c1: ComponentTag, setup: ComponentTag) -> None:
    """The one place a C1 and a setup model are allowed to meet."""
    if c1.lineage != setup.lineage:
        raise Phase18G3LineageError(
            f"refusing to pair the {c1.lineage} C1 with the {setup.lineage} setup model; components are never crossed"
        )
    if (c1.run_id, c1.period, c1.bundle_id) != (setup.run_id, setup.period, setup.bundle_id):
        raise Phase18G3Error(
            f"refusing to pair C1 from bundle {c1.bundle_id[:12]} (period {c1.period}) with the setup model from "
            f"bundle {setup.bundle_id[:12]} (period {setup.period}); a bundle is evaluated whole"
        )


# ---------------------------------------------------------------------------
# Export, play, receipts, accounting
# ---------------------------------------------------------------------------


def export_bundle_c1(bundle_directory, destination) -> dict:
    """Bridge the bundle's C1 into the frozen evaluation checkpoint format."""
    import torch

    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.checkpoint import load_checkpoint, save_checkpoint
    from ..warmstart_checkpoint import load_model_for_evaluation

    source = Path(bundle_directory) / C1_NAME
    model, metadata = load_model_for_evaluation(source, device="cpu")
    candidate_id = metadata["model"]["model_configuration"]["candidate_id"]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, destination)
    reloaded, exported = load_checkpoint(
        destination, expected_architecture_id=ARCHITECTURE_FAMILY, expected_configuration=candidate_config(candidate_id)
    )
    source_state, reloaded_state = model.state_dict(), reloaded.state_dict()
    bitwise = set(source_state) == set(reloaded_state) and all(torch.equal(source_state[n], reloaded_state[n]) for n in source_state)
    if not bitwise:
        raise Phase18G3Error(f"exporting {source} changed the weights")
    return {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "export": str(destination),
        "export_sha256": file_sha256(destination),
        "state_dict_digest": exported.get("state_dict_digest"),
        "c1_state_digest": state_dict_digest(model),
        "candidate_id": candidate_id,
        "bitwise_state_dict_match": True,
        "parameter_count": int(reloaded.parameter_count()),
    }


def play_chunks(matches: tuple, directory: Path, runner, *, chunk_units: int, label: str, log=print) -> tuple:
    """The G1 retry-safe chunk path, transcribed: a completed chunk is reused
    byte-for-byte, keyed by position and its own schedule digest."""
    directory.mkdir(parents=True, exist_ok=True)
    results, reports = [], []
    step = max(1, int(chunk_units))
    for index in range(0, len(matches), step):
        chunk = matches[index : index + step]
        number = index // step
        digest = schedule_digest(chunk)[:16]
        path = directory / f"chunk_{number:04d}_{digest}.pkl"
        if path.exists():
            with open(path, "rb") as stream:
                stored = pickle.load(stream)
            expected_ids = sorted(spec.match_id for spec in chunk)
            stored_ids = sorted(row.match_id for row in stored["results"])
            if stored_ids != expected_ids:
                raise Phase18G3Error(f"{label} chunk {number} at {path} holds different match ids than the schedule")
            results.extend(stored["results"])
            reports.append(stored["report"] | {"reused": True})
            continue
        chunk_results, report = runner(chunk)
        report = report | {"chunk": number, "reused": False}
        with open(path, "wb") as stream:
            pickle.dump({"results": chunk_results, "report": report}, stream)
        results.extend(chunk_results)
        reports.append(report)
        log(f"{label}: {len(results)}/{len(matches)} games (chunk {number})")
    return tuple(results), reports


def reconcile(matches: tuple, rows) -> dict:
    """`planned = completed + failed + missing` (the G1 rule, transcribed)."""
    planned_ids = [spec.match_id for spec in matches]
    planned_set = set(planned_ids)
    if len(planned_set) != len(planned_ids):
        raise Phase18G3Error("the schedule repeats a match id")
    by_id: dict = {}
    duplicates = []
    for row in rows:
        if row.match_id in by_id:
            duplicates.append(row.match_id)
        else:
            by_id[row.match_id] = row
    unplanned = sorted(set(by_id) - planned_set)
    missing = sorted(planned_set - set(by_id))
    failed = sorted(match_id for match_id, row in by_id.items() if row.errored)
    completed = [match_id for match_id, row in by_id.items() if match_id in planned_set and not row.errored]
    reconciles = len(planned_ids) == len(completed) + len(failed) + len(missing)
    return {
        "planned": len(planned_ids),
        "completed": len(completed),
        "failed": len(failed),
        "missing": len(missing),
        "unplanned": len(unplanned),
        "duplicates": len(duplicates),
        "failed_ids": failed[:20],
        "missing_ids": missing[:20],
        "reconciles": reconciles and not unplanned and not duplicates,
        "complete_for_primary": reconciles and not failed and not missing and not unplanned and not duplicates,
        "rule": "planned = completed + failed + missing; failures and retries are never draws and never passes",
    }


def write_receipts(path: Path, *, rows, cases, own_samples, arm: dict) -> dict:
    """One immutable, self-sufficient row per game."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_case = {case.case_index: (case, sample) for case, sample in zip(cases, own_samples)}
    written = 0
    with open(path, "w") as stream:
        for row in sorted(rows, key=lambda r: r.match_id):
            case, sample = by_case[int(row.setup_pair_id)]
            stream.write(
                json.dumps(
                    {
                        "evaluation_version": G3_EVALUATION_VERSION,
                        "run_id": arm["run_id"],
                        "arm": arm["label"],
                        "lineage": arm["lineage"],
                        "bundle_id": arm["bundle_id"],
                        "bundle_period": arm["bundle_period"],
                        "c1_state_digest": arm["c1_state_digest"],
                        "setup_model_digest": arm["setup_model_digest"],
                        "match_id": row.match_id,
                        "paired_unit_id": row.paired_unit_id,
                        "case_index": int(case.case_index),
                        "family_id": case.family_id,
                        "base_index": int(case.base_index),
                        "base_setup_id": case.base_setup_id,
                        "opponent_id": case.opponent_id,
                        "candidate_color": int(row.candidate_color),
                        "own_setup_content_fingerprint": sample.content_fingerprint,
                        "own_setup_class_fingerprint": sample.class_fingerprint,
                        "own_setup_reflected": bool(sample.reflected),
                        "own_setup_root_seed": int(sample.root_seed),
                        "base_content_fingerprint": case.base_content_fingerprint,
                        "setup_bank_version": row.setup_bank_version,
                        "red_setup": row.red_setup,
                        "blue_setup": row.blue_setup,
                        "first_player": row.first_player,
                        "candidate_seed": row.candidate_seed,
                        "opponent_seed": row.opponent_seed,
                        "root_seed": row.root_seed,
                        "candidate_policy": f"{row.candidate_policy_id}@{row.candidate_policy_version}",
                        "opponent_policy": f"{row.opponent_policy_id}@{row.opponent_policy_version}",
                        "rules": row.rules,
                        "winner": row.winner,
                        "draw": row.draw,
                        "candidate_result": row.candidate_result,
                        "candidate_score": row.candidate_score,
                        "terminal_reason": row.terminal_reason,
                        "plies": row.plies,
                        "decisions": row.decisions,
                        "replay_digest": row.replay_digest,
                        "errored": row.errored,
                        "policy_error": row.policy_error,
                        "policy_error_category": row.policy_error_category,
                        "illegal_action": row.policy_error_category == ERROR_ILLEGAL_ACTION,
                        "wall_clock_seconds": row.wall_clock_seconds,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
    return {"path": str(path), "rows": written, "sha256": file_sha256(path)}


#: Per-game fields identical across arms for one match id (the case, not the model).
ARM_INVARIANT_FIELDS = (
    "paired_unit_id",
    "setup_pair_id",
    "setup_bank_version",
    "candidate_color",
    "first_player",
    "root_seed",
    "candidate_seed",
    "opponent_seed",
    "opponent_policy_id",
    "opponent_policy_version",
    "rules",
)


def prove_arm_identity(rows_by_arm: dict, cases) -> dict:
    """Every arm played the same cases: identical match ids, seeds, colours,
    opponents, rules, and the opponent's formation; only the own setup differs."""
    labels = list(rows_by_arm)
    by_arm = {label: {row.match_id: row for row in rows} for label, rows in rows_by_arm.items()}
    problems: list = []
    ids = {label: sorted(by_arm[label]) for label in labels}
    if any(ids[label] != ids[labels[0]] for label in labels):
        problems.append("the arms played different match ids")
    mismatches: dict = {}
    base_mismatches = 0
    case_by_index = {case.case_index: case for case in cases}
    if not problems:
        for match_id in ids[labels[0]]:
            reference = by_arm[labels[0]][match_id]
            case = case_by_index[int(reference.setup_pair_id)]
            opponent_colour = BLUE if case.colour == RED else RED
            for label in labels[1:]:
                other = by_arm[label][match_id]
                for field in ARM_INVARIANT_FIELDS:
                    if getattr(other, field) != getattr(reference, field):
                        mismatches[field] = mismatches.get(field, 0) + 1
            for label in labels:
                row = by_arm[label][match_id]
                played = row.blue_setup if opponent_colour == BLUE else row.red_setup
                expected = serialize_setup(orient_setup(case.base_canonical, opponent_colour))
                if played != expected:
                    base_mismatches += 1
    if mismatches:
        problems.append(f"case identity differs between arms: {mismatches}")
    if base_mismatches:
        problems.append(f"{base_mismatches} games did not play the case's library formation for the opponent")
    return {"arms": labels, "rows_per_arm": {label: len(by_arm[label]) for label in labels}, "fields_checked": list(ARM_INVARIANT_FIELDS), "field_mismatches": mismatches, "opponent_formation_mismatches": base_mismatches, "problems": problems}


def evaluate_bundle(
    bundle_directory,
    *,
    config: PilotConfig,
    lineage: str,
    label: str,
    cases,
    work,
    device: str = "cpu",
    workers: int = 1,
    chunk_units: int = 64,
    setup_device: str = "cpu",
    log=print,
) -> dict:
    """Play the frozen schedule with one bundle's own C1 and own setup model."""
    from ...evaluation.neural_worker import InferenceOwner, run_neural_schedule
    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    bundle_directory = Path(bundle_directory)
    work = Path(work)
    manifest = verify_bundle(bundle_directory, expected_run_id=config.run_id, expected_lineage=lineage, expected_matched_digest=config.matched_digest())
    c1_tag = component_tag(manifest, "c1")
    setup_tag = component_tag(manifest, "setup_ema")
    assert_internally_matched(c1_tag, setup_tag)

    started = time.perf_counter()
    trainer, _setup_manifest = load_setup_trainer(bundle_directory, config.with_lineage(lineage), device=setup_device)
    setup_model = trainer.evaluation_model(device=setup_device)
    setup_model_digest = state_dict_digest(setup_model)
    if setup_model_digest != manifest["components"]["setup_ema"]["state_digest"]:
        raise Phase18G3Error("the bundle's evaluation setup model is not its EMA component")
    generation = resolve_own_setups(setup_model, cases, namespace=config.namespace, seed_index=config.seed_index, device=setup_device)
    bank = build_arm_bank(cases, generation.samples)
    matches = build_schedule(cases, namespace=config.namespace)
    for match in matches:
        match.resolve_setups(bank)

    work.mkdir(parents=True, exist_ok=True)
    export = export_bundle_c1(bundle_directory, work / "eval_weights.pt")
    if export["c1_state_digest"] != manifest["components"]["c1"]["state_digest"]:
        raise Phase18G3Error("the exported C1 is not the bundle's C1 component")
    owner = InferenceOwner(
        work / "eval_weights.pt",
        decision_mode=DECISION_MODE_GREEDY,
        device=device,
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config(export["candidate_id"]),
        name=f"phase18_g3_{label}",
    )
    candidate_ref = candidate_policy_ref()

    def runner(chunk):
        run = run_neural_schedule(chunk, bank, owner, policy_ref=candidate_ref, worker_count=int(workers), record_actions=False, on_policy_error=ON_POLICY_ERROR_QUARANTINE)
        report = {
            "matches": run.matches_run,
            "decisions": run.decisions,
            "wall_clock_seconds": round(run.wall_clock_seconds, 3),
            "policy_errors": run.policy_errors,
            "illegal_policy_actions": run.illegal_policy_actions,
            "worker_checkpoint_loads": run.worker_checkpoint_loads,
            "results_digest": run.results_digest,
        }
        return run.results, report

    try:
        results, reports = play_chunks(matches, work / "games", runner, chunk_units=chunk_units, label=label, log=log)
        owner_identity = owner.identity()
    finally:
        owner.close()

    arm = {
        "evaluation_version": G3_EVALUATION_VERSION,
        "label": label,
        "run_id": config.run_id,
        "lineage": lineage,
        "bundle": str(bundle_directory),
        "bundle_id": manifest["bundle_id"],
        "bundle_period": int(manifest["period"]),
        "c1_state_digest": manifest["components"]["c1"]["state_digest"],
        "setup_model_digest": setup_model_digest,
        "setup_model_role": "EMA component of the same bundle" + ("" if manifest["setup_updates_enabled"] else " (frozen initial model)"),
        "component_tags": {"c1": c1_tag.__dict__, "setup": setup_tag.__dict__},
    }
    receipts = write_receipts(work / "receipts.jsonl", rows=results, cases=cases, own_samples=generation.samples, arm=arm)
    accounting = reconcile(matches, results)
    record = arm | {
        "schedule": schedule_record(cases, matches, namespace=config.namespace),
        "bank_digest": bank.digest(),
        "own_setups": {
            "count": len(generation.samples),
            "distinct_content_fingerprints": generation.telemetry["distinct_content_fingerprints"],
            "distinct_class_fingerprints": generation.telemetry["distinct_class_fingerprints"],
            "reflected_fraction": generation.telemetry["reflected_fraction"],
            "legality_failures": generation.telemetry["legality_failures"],
            "orientation_failures": generation.telemetry["orientation_failures"],
            "immediately_terminal_count": generation.telemetry["immediately_terminal_count"],
            "digest": hashlib.sha256("\n".join(s.content_fingerprint for s in generation.samples).encode()).hexdigest(),
        },
        "export": export,
        "owner_identity": owner_identity,
        "chunks": reports,
        "accounting": accounting,
        "receipts": receipts,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    (work / "arm.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    if not accounting["reconciles"]:
        raise Phase18G3Error(f"the {label} arm does not reconcile: {accounting}")
    return record, results


# ---------------------------------------------------------------------------
# Analysis: the stratified cluster bootstrap and the decision rule
# ---------------------------------------------------------------------------


def case_scores(rows, cases) -> np.ndarray:
    """EWR per case (draw = 0.5), in case order; every case must be scored."""
    by_pair = {}
    for row in rows:
        if row.errored or row.candidate_score is None:
            raise Phase18G3Error(f"receipt {row.match_id} carries no score; the primary analysis is invalid")
        if int(row.setup_pair_id) in by_pair:
            raise Phase18G3Error(f"case {row.setup_pair_id} was scored twice")
        by_pair[int(row.setup_pair_id)] = float(row.candidate_score)
    missing = [case.case_index for case in cases if case.case_index not in by_pair]
    if missing:
        raise Phase18G3Error(f"{len(missing)} cases are missing a score (first {missing[:5]})")
    return np.array([by_pair[case.case_index] for case in cases], dtype=np.float64)


def stratified_cluster_bootstrap(per_base: np.ndarray, families: np.ndarray, *, replicates: int, seed: int) -> np.ndarray:
    """Resample bases with replacement WITHIN each family, carrying each base's
    full per-base mean, with the finite-stratum rescaling sqrt(n_f / (n_f - 1))
    about the family centre (the superseded design's validated T5 procedure;
    equal family sizes reduce it to the 6A formula)."""
    rng = np.random.default_rng(int(seed) % (2**32))
    labels = sorted(set(families.tolist()))
    groups = [np.nonzero(families == label)[0] for label in labels]
    total = per_base.size
    means = np.zeros(int(replicates), dtype=np.float64)
    for group in groups:
        n_f = group.size
        if n_f < 2:
            raise Phase18G3Error("every family needs at least two bases for the stratified bootstrap")
        values = per_base[group]
        draws = values[rng.integers(0, n_f, size=(int(replicates), n_f))].mean(axis=1)
        centre = values.mean()
        rescaled = centre + (draws - centre) * math.sqrt(n_f / (n_f - 1.0))
        means += rescaled * (n_f / total)
    return means


def direct_per_base_standard_error(per_base: np.ndarray, families: np.ndarray) -> float:
    labels = sorted(set(families.tolist()))
    variances = [per_base[families == label].var(ddof=1) for label in labels]
    return float(math.sqrt(float(np.mean(variances)) / per_base.size))


def paired_analysis(candidate_rows, control_rows, cases, *, namespace: str, margin: float = PRIMARY_MARGIN, replicates: int = BOOTSTRAP_REPLICATES, confidence: float = BOOTSTRAP_CONFIDENCE) -> dict:
    """The primary contrast EWR(candidate) - EWR(control), paired by case."""
    candidate = case_scores(candidate_rows, cases)
    control = case_scores(control_rows, cases)
    difference = candidate - control
    base_slots = np.array([case.base_slot for case in cases])
    slots = sorted(set(base_slots.tolist()))
    per_base = np.array([difference[base_slots == slot].mean() for slot in slots])
    family_of_slot = {case.base_slot: case.family_id for case in cases}
    families = np.array([family_of_slot[slot] for slot in slots])
    cases_per_base = {slot: int((base_slots == slot).sum()) for slot in slots}
    if len(set(cases_per_base.values())) != 1:
        raise Phase18G3Error(f"bases carry unequal case counts: {cases_per_base}")
    seed = evaluation_bootstrap_seed(namespace)
    resampled = stratified_cluster_bootstrap(per_base, families, replicates=replicates, seed=seed)
    resampled.sort()
    tail = (1.0 - confidence) / 2.0
    lower = float(np.quantile(resampled, tail))
    upper = float(np.quantile(resampled, 1.0 - tail))
    point = float(per_base.mean())
    lower_above_zero = lower > 0.0
    point_at_margin = point >= margin
    interval_contains_margin = lower <= margin <= upper

    def breakdown(key):
        groups: dict = {}
        for case, value in zip(cases, difference):
            groups.setdefault(getattr(case, key), []).append(float(value))
        return {str(k): {"n": len(v), "mean_difference": float(np.mean(v))} for k, v in sorted(groups.items())}

    return {
        "evaluation_version": G3_EVALUATION_VERSION,
        "contrast": "EWR(candidate_final) - EWR(control_final), paired by case",
        "cases": len(cases),
        "bases": len(slots),
        "families": len(set(families.tolist())),
        "cases_per_base": next(iter(cases_per_base.values())),
        "candidate_ewr": float(candidate.mean()),
        "control_ewr": float(control.mean()),
        "point": point,
        "lower": lower,
        "upper": upper,
        "confidence": float(confidence),
        "replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap": "stratified cluster bootstrap over bases within families, finite-stratum rescaling sqrt(n_f/(n_f-1))",
        "bootstrap_standard_error": float(resampled.std(ddof=1)),
        "direct_per_base_standard_error": direct_per_base_standard_error(per_base, families),
        "per_base_sd": float(per_base.std(ddof=1)),
        "margin": float(margin),
        "rule": "PROCEED requires the 95% lower bound above zero AND the point estimate at least the margin",
        "lower_above_zero": bool(lower_above_zero),
        "point_at_margin": bool(point_at_margin),
        "passes": bool(lower_above_zero and point_at_margin),
        "near_boundary": bool(interval_contains_margin),
        "near_boundary_rule": "a second seed is a conditional follow-up when the interval contains the margin (design 4.4); never pooled to rescue a failed primary",
        "cases_on_which_arms_differ": int((difference != 0.0).sum()),
        "by_opponent": breakdown("opponent_id"),
        "by_colour": breakdown("colour"),
        "by_family": breakdown("family_id"),
    }


RECEIPT_ALIASES = ("setup_pair_id", "opponent_policy_id", "opponent_policy_version")


def _receipt_row_aliases(receipt: dict, *, path, line_number: int) -> dict:
    """The three ARM_INVARIANT_FIELDS the receipt stores under other names (P18-A002).

    `setup_pair_id` is persisted as `case_index` -- identical by construction, because
    `write_receipts` resolves `by_case[int(row.setup_pair_id)]` and writes that case's
    `case_index`. The opponent policy is persisted as the single string
    `"<id>@<version>"`. Nothing else is reconstructed, and nothing is repaired: a
    receipt that does not support the reconstruction is rejected.
    """
    where = f"{path} line {line_number}"
    for field in ("case_index", "opponent_policy", "opponent_id"):
        if receipt.get(field) is None:
            raise Phase18G3Error(f"{where}: receipt is missing {field!r}; it cannot be read for analysis")

    combined = str(receipt["opponent_policy"])
    identifier, separator, version = combined.rpartition("@")
    if not separator or not identifier or not version:
        raise Phase18G3Error(
            f"{where}: opponent_policy {combined!r} is not '<id>@<version>'; refusing to guess the split"
        )
    if identifier != str(receipt["opponent_id"]):
        raise Phase18G3Error(
            f"{where}: opponent_policy id {identifier!r} disagrees with the persisted opponent_id "
            f"{receipt['opponent_id']!r}"
        )

    aliases = {
        "setup_pair_id": int(receipt["case_index"]),
        "opponent_policy_id": identifier,
        "opponent_policy_version": version,
    }
    for name, value in aliases.items():
        if name in receipt and receipt[name] != value:
            raise Phase18G3Error(
                f"{where}: receipt carries {name}={receipt[name]!r}, conflicting with the "
                f"reconstruction {value!r}"
            )
    return aliases


def read_receipt_rows(path) -> list:
    """Receipts back as the minimal row objects the analysis reads.

    The receipts are immutable and are never rewritten. Three ARM_INVARIANT_FIELDS are
    persisted under other names, so they are reconstructed here on load, under P18-A002.
    """
    from types import SimpleNamespace

    rows = []
    with open(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            receipt = json.loads(line)
            aliases = _receipt_row_aliases(receipt, path=path, line_number=line_number)
            rows.append(SimpleNamespace(**(receipt | aliases)))
    return rows


__all__ = [
    "ARM_INVARIANT_FIELDS",
    "RECEIPT_ALIASES",
    "CANDIDATE_TOKEN_ID",
    "ComponentTag",
    "EvaluationCase",
    "G3_EVALUATION_BANK_VERSION",
    "GATE_DTYPE",
    "OWN_SETUP_PURPOSE",
    "assert_internally_matched",
    "build_arm_bank",
    "build_cases",
    "build_schedule",
    "candidate_policy_ref",
    "case_scores",
    "cases_digest",
    "component_tag",
    "direct_per_base_standard_error",
    "evaluate_bundle",
    "export_bundle_c1",
    "load_evaluation_bases",
    "paired_analysis",
    "play_chunks",
    "prove_arm_identity",
    "read_receipt_rows",
    "reconcile",
    "resolve_own_setups",
    "schedule_record",
    "stratified_cluster_bootstrap",
    "write_receipts",
]
