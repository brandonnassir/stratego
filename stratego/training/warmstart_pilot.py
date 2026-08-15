"""Phase 8 Agent 5: `warmstart_pilot_v1` — bounded pilot selection and the
frozen `warmstart_train_config_v1`.

Specification sources:

- `05_AGENT_5_PILOT_SELECTION.md` (fairness, metrics, selection, freeze)
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 20-22, 25, 28
- Agent 1's frozen pilot matrix and selection contract
  (:data:`stratego.training.warmstart_contract.PILOT_CANDIDATES`,
  :func:`stratego.training.warmstart_contract.pilot_matrix`)

What this module is and is not
------------------------------
It is the *decision* layer: the arithmetic of Agent 1's frozen
`selection_score`, the hard-veto predicate, the tie-break order, and the
serialization of the one winning configuration. Every number it consumes is
produced by Agent 4's trainer and validation pass; nothing here trains, and
nothing here may invent a hyperparameter. The candidate matrix is read from
the live contract and cross-checked against Agent 1's accepted artifact, so a
drifted matrix is a stop rather than a silently different search.

Why the selection functions are pure
------------------------------------
`select_winner` takes plain records and returns a decision. That is what makes
"the winner is reproducible from the CSV" a testable property instead of a
claim: the acceptance harness writes per-checkpoint rows, and the suite
re-reads those rows, re-runs the same function, and requires the same winner.
A selection that can only be reproduced by re-running 30,000 optimizer updates
is not auditable.

Held-out discipline, measured rather than asserted
--------------------------------------------------
`record_model_input_access` instruments
:meth:`WarmstartBatch.model_input` — the single boundary where a batch becomes
model input — and tallies examples by corpus split. Agent 5's artifact
therefore reports an observed count of test examples fed to a model, not an
intention. `record_phase4_access` does the same for the Phase 4 evaluation
entry points plus the neural checkpoint-load counter, so "Phase 4 neural
evaluation games = 0" is likewise a measurement.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field

from .warmstart_checkpoint import (
    WARMSTART_CHECKPOINT_VERSION,
    WARMSTART_TRAINER_VERSION,
)
from .warmstart_contract import (
    DEVELOPMENT_BUDGET,
    PILOT_CANDIDATE_LIMIT,
    PILOT_CANDIDATES,
    PILOT_FIXED_CONTROLS,
    PILOT_SELECTION,
    WARMSTART_EVAL_VERSION,
    WARMSTART_EXAMPLE_VERSION,
    pilot_matrix,
)
from .warmstart_dataset import TRAIN_ORDER_VERSION, WarmstartBatch
from .warmstart_loss import WARMSTART_LOSS_VERSION
from .warmstart_metrics import WARMSTART_METRICS_VERSION
from .warmstart_seed import DECISION_SAMPLER_VERSION, SYNTHETIC_CORPUS_VERSION

#: This module's implementation version.
WARMSTART_PILOT_VERSION = "warmstart_pilot_v1"

#: The frozen train-configuration contract Agent 5 hands to Agent 6.
WARMSTART_TRAIN_CONFIG_VERSION = "warmstart_train_config_v1"

#: Agent 1's frozen budgets, restated here as module constants so a harness
#: cannot quietly run a different number of updates than the contract allows.
PILOT_UPDATE_BUDGET = int(DEVELOPMENT_BUDGET["pilot_updates_per_config_max"])
FINAL_UPDATE_BUDGET_MAX = int(DEVELOPMENT_BUDGET["final_run_optimizer_steps_max"])

#: Agent 1's frozen validation cadence for pilots.
PILOT_VALIDATION_CADENCE = int(PILOT_FIXED_CONTROLS["validation_cadence_updates"])

#: The frozen component-ratio veto threshold at the final pilot checkpoint.
RATIO_VETO_THRESHOLD = 1.05

#: The three ratios that make up the selection score, in score order.
RATIO_FIELDS = ("policy_ce_ratio", "value_ce_ratio", "belief_ce_ratio")

#: Machine-readable hard-veto reason codes. The strings are the artifact's
#: vocabulary; `veto_reasons` is the only thing that emits them.
VETO_NON_FINITE = "non_finite_loss_gradient_or_parameter"
VETO_TARGET_MISMATCH = "target_mismatch"
VETO_SPLIT_LEAK = "data_split_leak"
VETO_CHECKPOINT_FAILURE = "checkpoint_or_resume_failure"
VETO_RATIO_ABOVE_THRESHOLD = "component_ratio_above_1.05_at_final_checkpoint"
VETO_INCOMPLETE_BUDGET = "did_not_complete_the_frozen_update_budget"
VETO_MISSING_FINAL_SCORE = "no_selection_score_at_the_final_checkpoint"

#: Agent 1's tie-break order, restated as the sort key's documentation.
TIE_BREAK_ORDER = tuple(PILOT_SELECTION["tie_break_order"])


class WarmstartPilotError(RuntimeError):
    """A pilot/selection precondition failed. Always raised, never warned."""


# ---------------------------------------------------------------------------
# The candidate matrix
# ---------------------------------------------------------------------------


def frozen_candidate_matrix() -> tuple:
    """Agent 1's frozen candidates, in Agent 1's order, as plain dicts."""
    return tuple(dict(entry) for entry in PILOT_CANDIDATES)


def candidate_matrix_digest() -> str:
    """SHA-256 over the frozen matrix plus its fixed controls.

    Two harnesses that print the same digest ran the same search space.
    """
    matrix = pilot_matrix()
    canonical = json.dumps(
        {
            "candidates": matrix["candidates"],
            "fixed_controls": matrix["fixed_controls"],
            "candidate_limit": PILOT_CANDIDATE_LIMIT,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_candidate_matrix(recorded_matrix: "dict | None" = None) -> list:
    """Every matrix-exactness check, as a list of problems (empty = clean).

    `recorded_matrix` is Agent 1's accepted artifact payload
    (`contract["pilot_matrix"]`). When given, the live matrix must equal it
    field for field: a matrix that drifted after acceptance is a stop, not a
    new search space.
    """
    problems: list = []
    live = frozen_candidate_matrix()

    if len(live) > PILOT_CANDIDATE_LIMIT:
        problems.append(
            f"the live matrix has {len(live)} candidates, above the frozen "
            f"limit of {PILOT_CANDIDATE_LIMIT}"
        )
    identifiers = [entry["candidate_id"] for entry in live]
    if len(set(identifiers)) != len(identifiers):
        problems.append(f"duplicate candidate ids in the live matrix: {identifiers}")

    allowed = set(PILOT_SELECTION.get("allowed_dimensions", ()))
    for entry in live:
        varying = {
            name
            for name in entry
            if name not in ("candidate_id", "loss_profile")
        }
        unexpected = varying - {
            "learning_rate",
            "lambda_policy",
            "lambda_value",
            "lambda_belief",
        }
        if unexpected:
            problems.append(
                f"{entry['candidate_id']} varies {sorted(unexpected)}, outside the "
                f"allowed dimensions {sorted(allowed)}"
            )

    if recorded_matrix is not None:
        # Compare through `pilot_matrix()`, the JSON-safe projection of the
        # live contract, so a tuple-vs-list encoding cannot read as drift.
        serializable = pilot_matrix()
        recorded = [dict(entry) for entry in recorded_matrix.get("candidates", ())]
        if recorded != serializable["candidates"]:
            problems.append(
                "the live candidate matrix differs from Agent 1's accepted "
                f"artifact: live {serializable['candidates']} vs recorded {recorded}"
            )
        recorded_controls = dict(recorded_matrix.get("fixed_controls", {}))
        live_controls = dict(serializable["fixed_controls"])
        if recorded_controls != live_controls:
            differing = sorted(
                name
                for name in set(recorded_controls) | set(live_controls)
                if recorded_controls.get(name) != live_controls.get(name)
            )
            problems.append(
                f"fixed controls differ from Agent 1's artifact in: {differing}"
            )
        recorded_limit = recorded_matrix.get("candidate_limit")
        if recorded_limit is not None and int(recorded_limit) != PILOT_CANDIDATE_LIMIT:
            problems.append(
                f"candidate limit {recorded_limit} != live {PILOT_CANDIDATE_LIMIT}"
            )
    return problems


def model_state_checksum(state: dict) -> str:
    """SHA-256 over a model `state_dict`'s names, shapes and float32 bytes.

    Every pilot must start from the identical canonical C1 initialization;
    this is the number that says so. Tensors are compared as CPU float32 so
    the checksum names the initialization, not the device that holds it.
    """
    import torch

    hasher = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        values = tensor.detach().to("cpu", torch.float32).contiguous()
        hasher.update(name.encode())
        hasher.update(str(tuple(values.shape)).encode())
        hasher.update(values.numpy().tobytes())
    return hasher.hexdigest()


def batch_sequence_digest(keys_digests) -> str:
    """SHA-256 over one run's ordered per-step batch-identity digests.

    Equal digests mean two candidates consumed the same ordered batch
    identities — Agent 5's data-order fairness evidence, folded to one
    comparable string instead of tens of thousands of rows.
    """
    hasher = hashlib.sha256()
    for index, digest in enumerate(keys_digests):
        hasher.update(f"{index}|{digest}\n".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Selection arithmetic
# ---------------------------------------------------------------------------


def selection_score(ratios) -> "float | None":
    """Agent 1's frozen `mean(r_policy, r_value, r_belief)`; lower is better.

    `None` when any component is undefined — a partially measured checkpoint
    has no score, and a scoreless final checkpoint is a veto, never a pass.
    """
    values = list(ratios)
    if len(values) != len(RATIO_FIELDS):
        raise WarmstartPilotError(
            f"the selection score needs exactly {len(RATIO_FIELDS)} ratios, "
            f"got {len(values)}"
        )
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def selection_score_from_metrics(metrics: dict) -> "float | None":
    """`selection_score` read off a checkpoint record's ratio fields."""
    return selection_score(metrics.get(name) for name in RATIO_FIELDS)


def veto_reasons(record: dict) -> tuple:
    """Every hard-veto reason a candidate record trips, in contract order.

    Empty means the candidate is eligible for selection. The counters come
    from the trainer (`WarmstartTrainer.counters`); the ratio check is Agent
    1's `> 1.05 at the final pilot checkpoint` rule applied to the record's
    final-checkpoint metrics.
    """
    reasons: list = []
    counters = dict(record.get("counters", {}))

    non_finite = (
        int(counters.get("non_finite_losses", 0))
        + int(counters.get("non_finite_gradients", 0))
        + int(counters.get("non_finite_parameters", 0))
    )
    if non_finite or record.get("non_finite_failure"):
        reasons.append(VETO_NON_FINITE)
    if int(counters.get("illegal_targets", 0)) or record.get("target_mismatch"):
        reasons.append(VETO_TARGET_MISMATCH)
    if int(counters.get("data_mismatches", 0)) or record.get("split_leak"):
        reasons.append(VETO_SPLIT_LEAK)
    if int(counters.get("checkpoint_errors", 0)) or record.get("checkpoint_failure"):
        reasons.append(VETO_CHECKPOINT_FAILURE)

    completed = record.get("completed_updates")
    budget = int(record.get("update_budget", PILOT_UPDATE_BUDGET))
    if completed is None or int(completed) < budget:
        reasons.append(VETO_INCOMPLETE_BUDGET)

    final = dict(record.get("final_checkpoint", {}))
    if final.get("selection_score") is None:
        reasons.append(VETO_MISSING_FINAL_SCORE)
    for name in RATIO_FIELDS:
        value = final.get(name)
        if value is not None and float(value) > RATIO_VETO_THRESHOLD:
            reasons.append(f"{VETO_RATIO_ABOVE_THRESHOLD}:{name}={float(value):.6f}")
    return tuple(reasons)


def _sort_key(record: dict) -> tuple:
    final = dict(record.get("final_checkpoint", {}))
    return (
        float(final["selection_score"]),
        float(final["policy_ce_ratio"]),
        # An unmeasured throughput sorts last on this key rather than raising:
        # the two keys above have already decided every real comparison.
        -float(record.get("examples_per_second") or 0.0),
        str(record.get("candidate_id", "")),
    )


def rank_candidates(records) -> list:
    """Non-vetoed records in Agent 1's tie-break order, best first.

    The trailing `candidate_id` component makes the order total even if two
    candidates tie on all three contract keys; whether it was ever needed is
    reported by `select_winner`, because a real tie there would be a fact the
    reviewer should see rather than a coin flip the harness hides.
    """
    eligible = [record for record in records if not veto_reasons(record)]
    return sorted(eligible, key=_sort_key)


def select_winner(records) -> dict:
    """Apply veto, then Agent 1's tie-break order; return the full decision.

    Deterministic and pure: the same records always yield the same winner.
    When every candidate is vetoed the decision's `status` is `BLOCKED` and
    `winner` is `None` — the contract's instruction is to report that for
    review, never to broaden the search.
    """
    records = [dict(record) for record in records]
    for record in records:
        record["veto_reasons"] = list(veto_reasons(record))
    ranked = rank_candidates(records)
    vetoed = [record for record in records if record["veto_reasons"]]

    decision = {
        "pilot_version": WARMSTART_PILOT_VERSION,
        "score_definition": PILOT_SELECTION["score"],
        "score_checkpoint": PILOT_SELECTION["score_checkpoint"],
        "tie_break_order": list(TIE_BREAK_ORDER),
        "ratio_veto_threshold": RATIO_VETO_THRESHOLD,
        "candidates_considered": len(records),
        "candidates_vetoed": len(vetoed),
        "candidates_eligible": len(ranked),
        "vetoed": [
            {"candidate_id": record["candidate_id"], "reasons": record["veto_reasons"]}
            for record in records
            if record["veto_reasons"]
        ],
        "ranking": [
            {
                "rank": position + 1,
                "candidate_id": record["candidate_id"],
                "selection_score": record["final_checkpoint"]["selection_score"],
                "policy_ce_ratio": record["final_checkpoint"]["policy_ce_ratio"],
                "value_ce_ratio": record["final_checkpoint"]["value_ce_ratio"],
                "belief_ce_ratio": record["final_checkpoint"]["belief_ce_ratio"],
                "examples_per_second": record.get("examples_per_second"),
            }
            for position, record in enumerate(ranked)
        ],
    }
    if not ranked:
        decision["status"] = "BLOCKED"
        decision["winner"] = None
        decision["tie_break_used"] = None
        decision["margin_to_runner_up"] = None
        decision["reason"] = (
            "every predeclared candidate was vetoed; report BLOCKED for review "
            "rather than broadening the search"
        )
        return decision

    best, *rest = ranked
    runner_up = rest[0] if rest else None
    first_key, second_key = _sort_key(best), (_sort_key(runner_up) if runner_up else None)
    if second_key is None:
        tie_break_used = None
    elif first_key[0] != second_key[0]:
        tie_break_used = "selection_score"
    elif first_key[1] != second_key[1]:
        tie_break_used = "policy_ce_ratio"
    elif first_key[2] != second_key[2]:
        tie_break_used = "examples_per_second"
    else:
        tie_break_used = "candidate_id_determinism_fallback"

    decision["status"] = "PASS"
    decision["winner"] = best["candidate_id"]
    decision["winner_selection_score"] = best["final_checkpoint"]["selection_score"]
    decision["tie_break_used"] = tie_break_used
    decision["margin_to_runner_up"] = (
        float(runner_up["final_checkpoint"]["selection_score"])
        - float(best["final_checkpoint"]["selection_score"])
        if runner_up is not None
        else None
    )
    decision["runner_up"] = runner_up["candidate_id"] if runner_up is not None else None
    return decision


# ---------------------------------------------------------------------------
# Reproducing the decision from the published CSV
# ---------------------------------------------------------------------------

#: The scope label of the rows the selection reads: one full-validation-split
#: pass at the final pilot checkpoint per candidate.
SELECTION_SCOPE = "full_validation_split"

#: The scope label of the cadence rows: the evenly spread fixed-size passes
#: that produce the training curve. Never the selection input.
CADENCE_SCOPE = "cadence_spread"


def records_from_rows(rows) -> list:
    """Rebuild selection records from `agent_05_pilot_runs.csv` rows.

    The published CSV is the auditable record: this reads the selection-scope
    row of each candidate plus the per-candidate veto columns those rows
    carry, so `select_winner(records_from_rows(rows))` reproduces the frozen
    decision from the artifact alone.
    """

    def number(value):
        text = str(value).strip()
        return float(text) if text not in ("", "None", "nan") else None

    records: dict = {}
    for row in rows:
        if str(row.get("validation_scope")) != SELECTION_SCOPE:
            continue
        candidate_id = str(row["candidate_id"])
        if candidate_id in records:
            raise WarmstartPilotError(
                f"{candidate_id} has more than one {SELECTION_SCOPE} row"
            )
        records[candidate_id] = {
            "candidate_id": candidate_id,
            "completed_updates": int(float(row["global_step"])),
            "update_budget": int(float(row["update_budget"])),
            "examples_per_second": number(row["examples_per_second"]),
            "counters": {
                "non_finite_losses": int(float(row["non_finite_losses"])),
                "non_finite_gradients": int(float(row["non_finite_gradients"])),
                "non_finite_parameters": int(float(row["non_finite_parameters"])),
                "illegal_targets": int(float(row["illegal_targets"])),
                "data_mismatches": int(float(row["data_mismatches"])),
                "checkpoint_errors": int(float(row["checkpoint_errors"])),
            },
            "final_checkpoint": {
                "global_step": int(float(row["global_step"])),
                "policy_ce_ratio": number(row["policy_ce_ratio"]),
                "value_ce_ratio": number(row["value_ce_ratio"]),
                "belief_ce_ratio": number(row["belief_ce_ratio"]),
                "selection_score": number(row["selection_score"]),
            },
        }
    return [records[key] for key in sorted(records)]


# ---------------------------------------------------------------------------
# Freezing `warmstart_train_config_v1`
# ---------------------------------------------------------------------------

#: Every field `05_AGENT_5_PILOT_SELECTION.md` requires the frozen config to
#: serialize. Checked structurally, so an incomplete freeze cannot ship.
REQUIRED_TRAIN_CONFIG_FIELDS = (
    "model_candidate",
    "model_config_digest",
    "model_init_seed",
    "trainer_version",
    "checkpoint_version",
    "example_version",
    "corpus_version",
    "corpus_digests",
    "batch_size",
    "optimizer",
    "learning_rate",
    "adam_betas",
    "adam_epsilon",
    "weight_decay",
    "gradient_clip_norm",
    "lr_schedule",
    "warmup_steps",
    "lambda_policy",
    "lambda_value",
    "lambda_belief",
    "train_shuffle_seed",
    "train_order_version",
    "max_final_updates",
    "validation_cadence_updates",
    "checkpoint_cadence_updates",
    "best_checkpoint_metric",
    "early_stop_rule",
    "loader_topology",
    "device",
    "precision",
)


def build_frozen_train_config(
    *,
    winner_candidate_id: str,
    train_config_identity: dict,
    train_config_digest: str,
    model_config_digest: str,
    expected_fresh_init_checksum: str,
    corpus_identity: dict,
    max_final_updates: int,
    checkpoint_cadence_updates: int,
    best_checkpoint_metric: str,
    early_stop_rule: dict,
    loader_topology: dict,
    seeds: dict,
    validation_batches: "int | None",
) -> dict:
    """Serialize the one winning configuration Agent 6 must run verbatim.

    Every hyperparameter is copied from the frozen candidate and the frozen
    fixed controls — this function has no way to express a value the matrix
    does not already contain — and the run-shape fields Agent 5 owns
    (final budget, cadences, best-checkpoint metric, early stop, loader
    topology) are validated against Agent 1's development budget.
    """
    frozen = {entry["candidate_id"]: entry for entry in PILOT_CANDIDATES}.get(
        winner_candidate_id
    )
    if frozen is None:
        raise WarmstartPilotError(
            f"{winner_candidate_id!r} is not one of Agent 1's frozen candidates"
        )
    budget = int(max_final_updates)
    if budget < 1 or budget > FINAL_UPDATE_BUDGET_MAX:
        raise WarmstartPilotError(
            f"the final update budget {budget} is outside the frozen limit "
            f"1..{FINAL_UPDATE_BUDGET_MAX}"
        )
    if int(checkpoint_cadence_updates) < 1:
        raise WarmstartPilotError("checkpoint cadence must be >= 1")

    controls = PILOT_FIXED_CONTROLS
    config = {
        "model_candidate": controls["model"],
        "model_config_digest": str(model_config_digest),
        "model_init_seed": int(controls["model_init_seed"]),
        "expected_fresh_init_checksum": str(expected_fresh_init_checksum),
        "trainer_version": WARMSTART_TRAINER_VERSION,
        "checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
        "example_version": WARMSTART_EXAMPLE_VERSION,
        "loss_version": WARMSTART_LOSS_VERSION,
        "metrics_version": WARMSTART_METRICS_VERSION,
        "eval_version": WARMSTART_EVAL_VERSION,
        "sampler_version": DECISION_SAMPLER_VERSION,
        "corpus_version": SYNTHETIC_CORPUS_VERSION,
        "corpus_digests": dict(corpus_identity),
        "batch_size": int(controls["batch_size"]),
        "optimizer": controls["optimizer"],
        "learning_rate": float(frozen["learning_rate"]),
        "adam_betas": [float(value) for value in controls["adam_betas"]],
        "adam_epsilon": float(controls["adam_epsilon"]),
        "weight_decay": float(controls["weight_decay"]),
        "gradient_clip_norm": float(controls["gradient_clip_norm"]),
        "lr_schedule": controls["lr_schedule"],
        "warmup_steps": int(train_config_identity["warmup_steps"]),
        "lambda_policy": float(frozen["lambda_policy"]),
        "lambda_value": float(frozen["lambda_value"]),
        "lambda_belief": float(frozen["lambda_belief"]),
        "loss_profile": frozen["loss_profile"],
        "train_split": "train",
        "train_shuffle_seed": int(seeds["train_order_seed"]),
        "train_order_version": TRAIN_ORDER_VERSION,
        "train_order": train_config_identity["order"],
        "max_final_updates": budget,
        "validation_split": "validation",
        "validation_cadence_updates": int(
            train_config_identity["validation_cadence_updates"]
        ),
        "validation_batches": (
            int(validation_batches) if validation_batches is not None else None
        ),
        "validation_selection": train_config_identity["validation_selection"],
        "checkpoint_cadence_updates": int(checkpoint_cadence_updates),
        "best_checkpoint_metric": str(best_checkpoint_metric),
        "early_stop_rule": dict(early_stop_rule),
        "loader_topology": dict(loader_topology),
        "device": controls["device"],
        "precision": controls["precision"],
    }
    missing = [name for name in REQUIRED_TRAIN_CONFIG_FIELDS if name not in config]
    if missing:
        raise WarmstartPilotError(
            f"the frozen train config is incomplete, missing: {missing}"
        )

    payload = {
        "train_config_version": WARMSTART_TRAIN_CONFIG_VERSION,
        "pilot_version": WARMSTART_PILOT_VERSION,
        "frozen_by": "phase_8_agent_5",
        "winning_candidate_id": winner_candidate_id,
        "candidate_matrix_digest": candidate_matrix_digest(),
        "trainer_construction": (
            "stratego.training.warmstart_trainer.WarmstartTrainConfig."
            f"from_pilot_candidate({winner_candidate_id!r}, device="
            f"{controls['device']!r}, validation_batches={validation_batches!r})"
        ),
        "trainer_config_identity": dict(train_config_identity),
        "trainer_config_digest": str(train_config_digest),
        "canonical_seeds": dict(seeds),
        "config": config,
        "agent_6_rules": [
            "start from a fresh reconstruction of the canonical C1 "
            "initialization; never continue a pilot checkpoint",
            "do not tune any field of this configuration",
            "select the best checkpoint by validation only",
            "the sealed test split and the Phase 4 bank stay closed until "
            "Agent 7",
        ],
    }
    payload["train_config_digest"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_frozen_train_config(payload: dict) -> list:
    """Completeness/versioning checks on a frozen config, as problems."""
    problems: list = []
    if payload.get("train_config_version") != WARMSTART_TRAIN_CONFIG_VERSION:
        problems.append(
            f"train_config_version is {payload.get('train_config_version')!r}, "
            f"expected {WARMSTART_TRAIN_CONFIG_VERSION!r}"
        )
    config = dict(payload.get("config", {}))
    missing = [name for name in REQUIRED_TRAIN_CONFIG_FIELDS if name not in config]
    if missing:
        problems.append(f"frozen config is missing required fields: {missing}")
    budget = config.get("max_final_updates")
    if budget is None or not 1 <= int(budget) <= FINAL_UPDATE_BUDGET_MAX:
        problems.append(
            f"max_final_updates {budget!r} is outside 1..{FINAL_UPDATE_BUDGET_MAX}"
        )
    winner = payload.get("winning_candidate_id")
    frozen = {entry["candidate_id"]: entry for entry in PILOT_CANDIDATES}.get(winner)
    if frozen is None:
        problems.append(f"winning candidate {winner!r} is not in the frozen matrix")
    else:
        for name in ("learning_rate", "lambda_policy", "lambda_value", "lambda_belief"):
            if config.get(name) != frozen[name]:
                problems.append(
                    f"frozen config {name}={config.get(name)!r} != candidate "
                    f"{frozen[name]!r}"
                )
    recomputed = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed != payload.get("train_config_digest"):
        problems.append("train_config_digest does not match the serialized config")
    return problems


# ---------------------------------------------------------------------------
# Held-out access, measured
# ---------------------------------------------------------------------------


@dataclass
class ModelInputAccessLog:
    """Examples that crossed the model-input boundary, tallied by split."""

    examples_by_split: dict = field(default_factory=dict)
    batches_by_split: dict = field(default_factory=dict)

    def record(self, corpus_splits) -> None:
        for split in corpus_splits:
            self.examples_by_split[split] = self.examples_by_split.get(split, 0) + 1
        if corpus_splits:
            split = corpus_splits[0]
            self.batches_by_split[split] = self.batches_by_split.get(split, 0) + 1

    @property
    def test_examples(self) -> int:
        return int(self.examples_by_split.get("test", 0))

    def to_dict(self) -> dict:
        return {
            "boundary": (
                "stratego.training.warmstart_dataset.WarmstartBatch.model_input"
            ),
            "examples_by_split": dict(sorted(self.examples_by_split.items())),
            "batches_by_split": dict(sorted(self.batches_by_split.items())),
            "test_examples_evaluated_by_model": self.test_examples,
        }


@contextmanager
def record_model_input_access(log: "ModelInputAccessLog | None" = None):
    """Tally every batch that becomes model input while the block runs.

    Instruments the one method that hands observations to a model, so the
    resulting count is an observation of what happened rather than a claim
    about what was intended.
    """
    log = log if log is not None else ModelInputAccessLog()
    original = WarmstartBatch.model_input

    def instrumented(self):
        log.record(self.corpus_splits)
        return original(self)

    WarmstartBatch.model_input = instrumented
    try:
        yield log
    finally:
        WarmstartBatch.model_input = original


@dataclass
class PhaseFourAccessLog:
    """Phase 4 evaluation entry points invoked while the block runs."""

    play_match_calls: int = 0
    run_schedule_calls: int = 0
    run_neural_schedule_calls: int = 0
    neural_checkpoint_loads_before: int = 0
    neural_checkpoint_loads_after: int = 0

    @property
    def neural_evaluation_games(self) -> int:
        """Every way a Phase 8 process could have played an evaluation game."""
        return int(
            self.play_match_calls
            + self.run_schedule_calls
            + self.run_neural_schedule_calls
        )

    def to_dict(self) -> dict:
        return {
            "instrumented": [
                "stratego.evaluation.match_runner.play_match",
                "stratego.evaluation.match_runner.run_schedule",
                "stratego.evaluation.neural_worker.run_neural_schedule",
                "stratego.evaluation.neural_worker.checkpoint_load_count",
            ],
            "play_match_calls": self.play_match_calls,
            "run_schedule_calls": self.run_schedule_calls,
            "run_neural_schedule_calls": self.run_neural_schedule_calls,
            "neural_checkpoint_loads": int(
                self.neural_checkpoint_loads_after - self.neural_checkpoint_loads_before
            ),
            "phase4_neural_evaluation_games": self.neural_evaluation_games,
        }


@contextmanager
def record_phase4_access(log: "PhaseFourAccessLog | None" = None):
    """Count Phase 4 evaluation-game entry points while the block runs.

    Agent 5 must play zero. Wrapping the entry points rather than blocking
    them keeps the evidence honest: a call would be counted and reported, not
    silently swallowed.
    """
    from ..evaluation import match_runner, neural_worker

    log = log if log is not None else PhaseFourAccessLog()
    log.neural_checkpoint_loads_before = int(neural_worker.checkpoint_load_count())
    originals = {
        (match_runner, "play_match"): match_runner.play_match,
        (match_runner, "run_schedule"): match_runner.run_schedule,
        (neural_worker, "run_neural_schedule"): neural_worker.run_neural_schedule,
    }
    counters = {
        "play_match": "play_match_calls",
        "run_schedule": "run_schedule_calls",
        "run_neural_schedule": "run_neural_schedule_calls",
    }

    def wrap(original, attribute):
        def instrumented(*arguments, **keywords):
            setattr(log, counters[attribute], getattr(log, counters[attribute]) + 1)
            return original(*arguments, **keywords)

        return instrumented

    for (module, attribute), original in originals.items():
        setattr(module, attribute, wrap(original, attribute))
    try:
        yield log
    finally:
        for (module, attribute), original in originals.items():
            setattr(module, attribute, original)
        log.neural_checkpoint_loads_after = int(neural_worker.checkpoint_load_count())


__all__ = [
    "CADENCE_SCOPE",
    "FINAL_UPDATE_BUDGET_MAX",
    "PILOT_CANDIDATE_LIMIT",
    "PILOT_UPDATE_BUDGET",
    "PILOT_VALIDATION_CADENCE",
    "RATIO_FIELDS",
    "RATIO_VETO_THRESHOLD",
    "REQUIRED_TRAIN_CONFIG_FIELDS",
    "SELECTION_SCOPE",
    "TIE_BREAK_ORDER",
    "WARMSTART_PILOT_VERSION",
    "WARMSTART_TRAIN_CONFIG_VERSION",
    "ModelInputAccessLog",
    "PhaseFourAccessLog",
    "WarmstartPilotError",
    "batch_sequence_digest",
    "build_frozen_train_config",
    "candidate_matrix_digest",
    "frozen_candidate_matrix",
    "model_state_checksum",
    "rank_candidates",
    "record_model_input_access",
    "record_phase4_access",
    "records_from_rows",
    "select_winner",
    "selection_score",
    "selection_score_from_metrics",
    "veto_reasons",
    "verify_candidate_matrix",
    "verify_frozen_train_config",
]
