"""Phase 8 Agent 1: the frozen warm-start training contract.

Specification sources:

- `01_AGENT_1_WARMSTART_CONTRACT.md` (everything this module freezes)
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 6-28 (frozen upstream
  stack, targets, baselines, loss, pilot budget, acceptance thresholds,
  held-out discipline)

What "frozen" means here
------------------------
Every learning-design decision of Phase 8 is stated in this module **before**
any synthetic production corpus exists and before any meaningful optimizer
step has run: the teacher roster and its policy-supervision weights, the
ordered matchup schedule, the setup-source configuration per corpus split,
the game identity and its seeds (:mod:`stratego.training.warmstart_seed`),
the decision sampler, the `warmstart_example_v1` schema, the target and
baseline semantics, the loss normalization, the bounded pilot matrix with its
selection score, the final acceptance thresholds, and the held-out sealing
rules. Agents 2-7 read these values; they do not choose them. A different
value is a reviewed new version of this contract, never an in-place edit.

Nothing in this module generates corpus games, builds training examples, or
touches model weights. The only executable behaviour is verification (does
the live repository still match the frozen expectation?) and construction of
the already-accepted setup sources through their frozen Phase 7 entry points.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..engine.constants import (
    IMPLEMENTATION_VERSION,
    RULES_VERSION,
    TRAINING_RULES,
    RulesConfig,
)
from ..evaluation.registry import (
    ALL_POLICY_IDS,
    LADDER_POLICY_IDS,
    POLICY_INDEX,
    STRESS_POLICY_IDS,
    policy_catalog,
)
from ..evaluation.setup_bank import (
    DEFAULT_BANK_ROOT_SEED,
    DEFAULT_BANK_SIZE,
    SETUP_BANK_VERSION,
)
from ..model.architecture_configs import candidate_config
from ..model.contract import (
    ACTION_ENCODING_VERSION,
    BELIEF_IGNORE_INDEX,
    BELIEF_TYPE_COUNT,
    ENGINE_ACTION_FRAME,
    MODEL_ACTION_FRAME,
    MODEL_CONTRACT_VERSION,
    OBSERVATION_CHANNELS,
    OBSERVATION_VERSION,
    VALUE_CLASS_ORDER,
    VALUE_DRAW_INDEX,
    VALUE_LOSS_INDEX,
    VALUE_WIN_INDEX,
)
from ..setups.contracts import SETUP_LIBRARY_VERSION
from ..setups.perturbation import PERTURBATION_SEED_ENCODING, PERTURBATION_VERSION
from ..setups.sampler import DEFAULT_PROFILE, SAMPLER_VERSION, load_library_index
from .belief_targets import BELIEF_TARGET_SQUARE_FRAME, BELIEF_TARGET_VERSION
from .setup_source import (
    LibrarySetupSource,
    PROVENANCE_SCHEMA_VERSION,
    SETUP_SOURCE_VERSION,
    audit_setup_source,
    training_setup_source,
)
from .trajectory import DEFAULT_SNAPSHOT_INTERVAL, TRAJECTORY_VERSION
from .warmstart_seed import (
    CANONICAL_C1_INIT_SEED,
    CANONICAL_SEEDS,
    CORPUS_SPLITS,
    DECISION_SAMPLER_VERSION,
    GAMES_PER_CELL,
    MAX_DECISIONS_PER_GAME,
    SYNTHETIC_CORPUS_VERSION,
    TEST_BOOTSTRAP_SEED,
    VALIDATION_BOOTSTRAP_SEED,
    synthetic_game_id,
)

#: The whole Phase 8 learning-design contract. Any semantic change to any
#: frozen value below is a new contract version after review.
WARMSTART_TRAINING_CONTRACT_VERSION = "warmstart_training_contract_v1"

#: The training-example schema (section `EXAMPLE_FIELDS`).
WARMSTART_EXAMPLE_VERSION = "warmstart_example_v1"

#: The evaluation contract: metrics, baselines, aggregation and bootstrap.
WARMSTART_EVAL_VERSION = "warmstart_eval_v1"

#: Frozen upstream identities Phase 8 depends on and must not change.
EXPECTED_ENGINE_VERSION = "phase2_1_reference_1.2.0"
EXPECTED_RULES_VERSION = "stratego_project_v1"
EXPECTED_OBSERVATION_VERSION = "observation_v2_1_127ch"
EXPECTED_MODEL_CONTRACT_VERSION = "model_contract_v2"
EXPECTED_ACTION_ENCODING_VERSION = "source_destination_10000_v1"
EXPECTED_TRAJECTORY_VERSION = "trajectory_v1"
EXPECTED_SNAPSHOT_INTERVAL = 32
EXPECTED_C1_PARAMETERS = 863959
EXPECTED_C1_CONFIG_DIGEST = (
    "31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d"
)
EXPECTED_LIBRARY_DIGEST = (
    "7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777"
)
EXPECTED_PHASE4_BANK_DIGEST = (
    "5fe5f98750ca2bd90ee75a74b3ba024bf753342872ae5472f13eb7afbb674266"
)
EXPECTED_SETUP_PROFILE = "neutral_v1"


class WarmstartContractError(RuntimeError):
    """Raised when the frozen Phase 8 contract is violated or unusable."""


# ---------------------------------------------------------------------------
# Teacher population — the frozen Phase 4 roster and its supervision weights
# ---------------------------------------------------------------------------

#: The exact accepted Phase 4 roster, in the frozen matchup-schedule order
#: (the live registry's own ladder-then-stress order). Copied literally so a
#: registry drift is a detectable BLOCKED condition rather than a silent
#: schedule change.
EXPECTED_TEACHER_ROSTER = (
    ("random_legal", "1.0.0", "tier_random"),
    ("basic_heuristic", "1.0.0", "tier_basic"),
    ("tactical_rule_based", "1.0.0", "tier_tactical"),
    ("strategic_rule_based", "1.1.0", "tier_strategic"),
    ("stress_scout_rush", "1.0.0", "stress"),
    ("stress_miner_rush", "1.0.0", "stress"),
    ("stress_draw_seeker", "1.0.0", "stress"),
    ("stress_berserker", "1.0.0", "stress"),
    ("stress_information_miser", "1.0.0", "stress"),
    ("stress_chaos", "1.0.0", "stress"),
)

EXPECTED_TEACHER_COUNT = 10

#: Frozen policy-supervision weights by role (common contract section 10).
#: Random/stress decisions carry no policy gradient but remain fully eligible
#: for value and belief supervision.
ROLE_POLICY_WEIGHTS = {
    "tier_strategic": 1.0,
    "tier_tactical": 1.0,
    "tier_basic": 0.5,
    "tier_random": 0.0,
    "stress": 0.0,
}

POLICY_SUPERVISION_WEIGHTS = {
    policy_id: ROLE_POLICY_WEIGHTS[role] for policy_id, _, role in EXPECTED_TEACHER_ROSTER
}

#: Every decision, whatever its acting policy, supervises value and belief.
VALUE_SUPERVISION_ELIGIBLE = True
BELIEF_SUPERVISION_ELIGIBLE = True


def policy_weight(policy_id: str) -> float:
    """The frozen policy-supervision weight of one teacher policy."""
    try:
        return POLICY_SUPERVISION_WEIGHTS[policy_id]
    except KeyError:
        raise WarmstartContractError(
            f"{policy_id!r} is not in the frozen Phase 8 teacher roster"
        ) from None


def teacher_tokens() -> tuple:
    """The ten `id@version` tokens in frozen matchup-schedule order."""
    return tuple(
        f"{policy_id}@{version}" for policy_id, version, _ in EXPECTED_TEACHER_ROSTER
    )


def teacher_population() -> list:
    """The frozen teacher table, joined with the live catalogue entries.

    Each row records identity, version, implementation path, the behaviour
    contract (stochastic flag plus the seeded selection semantics accepted in
    Phase 4), the role, and the frozen policy-supervision weight.
    """
    catalog = {entry["policy_id"]: entry for entry in policy_catalog()}
    population = []
    for policy_id, version, role in EXPECTED_TEACHER_ROSTER:
        entry = catalog.get(policy_id, {})
        policy_class = POLICY_INDEX.get(policy_id)
        module = policy_class.__module__ if policy_class is not None else None
        selection_margin = getattr(policy_class, "selection_margin", None)
        population.append(
            {
                "policy_id": policy_id,
                "policy_version": version,
                "implementation_path": (
                    module.replace(".", "/") + ".py" if module else None
                ),
                "role": role,
                "policy_weight": ROLE_POLICY_WEIGHTS[role],
                "value_supervision": VALUE_SUPERVISION_ELIGIBLE,
                "belief_supervision": BELIEF_SUPERVISION_ELIGIBLE,
                "stochastic": entry.get("stochastic"),
                "behavior_contract": {
                    "interface_version": entry.get("interface_version"),
                    "selection_margin": (
                        float(selection_margin) if selection_margin is not None else None
                    ),
                    "ranking_tie_break": "descending score, then ascending action id",
                    "decision_rng": (
                        "random.Random(decision_seed); decision_seed = "
                        "derive_decision_seed(policy_seed, ply) [frozen Phase 4 "
                        "blake2b person 'strat-dec']"
                    ),
                    "policy_seed_source": (
                        "warmstart_seed.red_policy_seed / blue_policy_seed of the "
                        "synthetic game id (domains 'policy:red' / 'policy:blue')"
                    ),
                },
                "description": entry.get("description"),
            }
        )
    return population


def verify_teacher_roster() -> list:
    """Every discrepancy between the frozen roster and the live registry."""
    problems = []
    live = tuple(
        (policy_id, POLICY_INDEX[policy_id].policy_version) for policy_id in ALL_POLICY_IDS
    )
    expected = tuple((policy_id, version) for policy_id, version, _ in EXPECTED_TEACHER_ROSTER)
    if len(live) != EXPECTED_TEACHER_COUNT:
        problems.append(
            f"live Phase 4 registry has {len(live)} policies, expected "
            f"{EXPECTED_TEACHER_COUNT}"
        )
    if live != expected:
        problems.append(
            f"live roster {live!r} differs from the frozen expectation {expected!r}"
        )
    if len(LADDER_POLICY_IDS) != 4:
        problems.append(f"expected 4 ladder policies, found {len(LADDER_POLICY_IDS)}")
    if len(STRESS_POLICY_IDS) != 6:
        problems.append(f"expected 6 stress policies, found {len(STRESS_POLICY_IDS)}")
    for policy_id, _, role in EXPECTED_TEACHER_ROSTER:
        is_stress = policy_id in STRESS_POLICY_IDS
        if is_stress != (role == "stress"):
            problems.append(f"{policy_id}: frozen role {role!r} disagrees with the registry")
    return problems


# ---------------------------------------------------------------------------
# Ordered matchup schedule
# ---------------------------------------------------------------------------

EXPECTED_CELL_COUNT = 100
SCHEDULE_TOTALS = {"train": 20000, "validation": 4000, "test": 4000, "total": 28000}


def ordered_matchup_cells() -> list:
    """The 100 ordered (red, blue) teacher cells in frozen red-major order.

    ``cell_index = red_index * 10 + blue_index`` over the frozen roster
    order. Self-play cells (red == blue policy) are included: the two sides
    still receive independent seeds and setups.
    """
    tokens = teacher_tokens()
    cells = []
    for red_index, red_token in enumerate(tokens):
        for blue_index, blue_token in enumerate(tokens):
            cells.append(
                {
                    "cell_index": red_index * len(tokens) + blue_index,
                    "red_index": red_index,
                    "blue_index": blue_index,
                    "red_token": red_token,
                    "blue_token": blue_token,
                }
            )
    return cells


def matchup_schedule() -> dict:
    """The complete frozen schedule: every cell, every split, exact counts."""
    return {
        "cells": ordered_matchup_cells(),
        "games_per_cell": dict(GAMES_PER_CELL),
        "totals": dict(SCHEDULE_TOTALS),
        "counting_rule": (
            "exact per-cell counts; matchup counts are scheduled, never the "
            "outcome of random sampling"
        ),
    }


def iter_game_identities(split: str):
    """Every logical game identity of one split, schedule order.

    Yields ``(cell_index, red_token, blue_token, ordinal, game_id)``. The
    order is cell-major then ordinal, which is a *schedule* order only; the
    training-time example order is owned by the frozen shuffle streams, and
    corpus determinism is per-game, so generation may proceed in any order.
    """
    if split not in CORPUS_SPLITS:
        raise WarmstartContractError(f"unknown corpus split: {split!r}")
    for cell in ordered_matchup_cells():
        for ordinal in range(GAMES_PER_CELL[split]):
            yield (
                cell["cell_index"],
                cell["red_token"],
                cell["blue_token"],
                ordinal,
                synthetic_game_id(split, cell["red_token"], cell["blue_token"], ordinal),
            )


# ---------------------------------------------------------------------------
# Setup sources and corpus game rules
# ---------------------------------------------------------------------------

#: Written justifications the audit entry point requires, exactly as the
#: assignment states them.
HELD_OUT_SETUP_JUSTIFICATIONS = {
    "validation": "Phase 8 held-out warm-start validation corpus",
    "test": "Phase 8 sealed held-out warm-start test corpus",
}

#: Frozen constants of the per-game `assign` call. The per-game root seed is
#: `warmstart_seed.setup_root_seed(game_id)`; with environment and generation
#: pinned to zero, per-game independence comes entirely from the
#: domain-separated root seed, and red/blue independence comes from the
#: sampler's own accepted per-side derivation.
SETUP_SOURCE_ENVIRONMENT_ID = 0
SETUP_SOURCE_GENERATION = 0

#: Corpus games run under the frozen training rules context (the same rules
#: the accepted Phase 6/7 collection pipeline uses). The Phase 4 evaluation
#: gates keep their own frozen `EVALUATION_RULES` via the untouched Phase 4
#: match machinery.
CORPUS_RULES: RulesConfig = TRAINING_RULES


def corpus_setup_source(split: str) -> LibrarySetupSource:
    """The frozen setup source of one corpus split.

    ``train`` goes through the production entry point (hard-wired to the
    train split); ``validation``/``test`` go through the audit entry point
    with the frozen written justification. No other construction path is
    part of the Phase 8 contract.
    """
    if split == "train":
        return training_setup_source(EXPECTED_SETUP_PROFILE)
    if split in HELD_OUT_SETUP_JUSTIFICATIONS:
        return audit_setup_source(
            split,
            HELD_OUT_SETUP_JUSTIFICATIONS[split],
            profile=EXPECTED_SETUP_PROFILE,
        )
    raise WarmstartContractError(f"unknown corpus split: {split!r}")


def setup_source_configuration() -> dict:
    """The serializable frozen setup-source contract, per split."""
    per_split = {}
    for split in CORPUS_SPLITS:
        entry = {
            "split": split,
            "entry_point": (
                f"training_setup_source({EXPECTED_SETUP_PROFILE!r})"
                if split == "train"
                else (
                    f"audit_setup_source({split!r}, "
                    f"{HELD_OUT_SETUP_JUSTIFICATIONS[split]!r}, "
                    f"profile={EXPECTED_SETUP_PROFILE!r})"
                )
            ),
            "profile": EXPECTED_SETUP_PROFILE,
            "access_justification": HELD_OUT_SETUP_JUSTIFICATIONS.get(split, ""),
            "assign_call": {
                "root_seed": "warmstart_seed.setup_root_seed(game_id)",
                "environment_id": SETUP_SOURCE_ENVIRONMENT_ID,
                "generation": SETUP_SOURCE_GENERATION,
                "game_id": "the synthetic game id",
            },
        }
        per_split[split] = entry
    return {
        "setup_library_version": SETUP_LIBRARY_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "setup_source_version": SETUP_SOURCE_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "profile": EXPECTED_SETUP_PROFILE,
        "per_split": per_split,
        "independence": (
            "each game samples its red and blue setups independently through "
            "the sampler's frozen per-side stream derivation"
        ),
        "weighting": "no setup family receives outcome-based weighting",
    }


# ---------------------------------------------------------------------------
# Training-example schema (`warmstart_example_v1`)
# ---------------------------------------------------------------------------

#: Exact field names, types and shapes of one reconstructed training example.
#: `model_input` marks the only tensor the network consumes; `loss_input`
#: marks tensors consumed by loss/metric computation; everything else is
#: metadata that must never reach the model input.
EXAMPLE_FIELDS = (
    {
        "name": "observation",
        "dtype": "float32",
        "shape": (OBSERVATION_CHANNELS, 10, 10),
        "model_input": True,
        "loss_input": False,
        "description": "observation_v2_1_127ch for the acting player",
    },
    {
        "name": "legal_mask",
        "dtype": "bool",
        "shape": (10000,),
        "model_input": False,
        "loss_input": True,
        "description": (
            "legal actions in the model (perspective-normalized) frame; used "
            "by the masked policy loss / action adapter, never concatenated "
            "into the observation"
        ),
    },
    {
        "name": "acting_player",
        "dtype": "int8",
        "shape": (),
        "model_input": False,
        "loss_input": False,
        "description": "engine player constant (RED=0, BLUE=1) before conversion",
    },
    {
        "name": "policy_action_abs",
        "dtype": "int32",
        "shape": (),
        "model_input": False,
        "loss_input": False,
        "description": "the action the rule policy chose, absolute engine id",
    },
    {
        "name": "policy_action_model",
        "dtype": "int64",
        "shape": (),
        "model_input": False,
        "loss_input": True,
        "description": (
            "the same action in the perspective-normalized model frame via "
            "the frozen absolute_action_to_model conversion"
        ),
    },
    {
        "name": "policy_weight",
        "dtype": "float32",
        "shape": (),
        "model_input": False,
        "loss_input": True,
        "description": "frozen supervision weight of the acting rule policy",
    },
    {
        "name": "value_target",
        "dtype": "int64",
        "shape": (),
        "model_input": False,
        "loss_input": True,
        "description": "WIN=0 / DRAW=1 / LOSS=2 from the acting player's perspective",
    },
    {
        "name": "belief_target",
        "dtype": "int64",
        "shape": (100,),
        "model_input": False,
        "loss_input": True,
        "description": (
            "dense_belief_target_v1 labels: true piece-type index on exactly "
            "the unresolved hidden opponent squares (model frame), "
            f"{BELIEF_IGNORE_INDEX} elsewhere"
        ),
    },
    {
        "name": "belief_mask",
        "dtype": "bool",
        "shape": (100,),
        "model_input": False,
        "loss_input": True,
        "description": "true exactly where belief is supervised",
    },
    {
        "name": "game_id",
        "dtype": "str",
        "shape": (),
        "model_input": False,
        "loss_input": False,
        "description": "the synthetic game id",
    },
    {
        "name": "decision_index",
        "dtype": "int32",
        "shape": (),
        "model_input": False,
        "loss_input": False,
        "description": "0-based ply index of this decision in the game",
    },
    {
        "name": "source_policy_id",
        "dtype": "str",
        "shape": (),
        "model_input": False,
        "loss_input": False,
        "description": "the acting rule policy's id",
    },
    {
        "name": "corpus_split",
        "dtype": "str",
        "shape": (),
        "model_input": False,
        "loss_input": False,
        "description": "train / validation / test",
    },
)


def example_schema() -> dict:
    """The serializable `warmstart_example_v1` schema."""
    return {
        "example_version": WARMSTART_EXAMPLE_VERSION,
        "fields": [dict(field, shape=list(field["shape"])) for field in EXAMPLE_FIELDS],
        "model_input_fields": ["observation"],
        "privileged_boundary": (
            "privileged metadata (true hidden types, provenance, teacher "
            "identity) may ride on the example object but must never enter "
            "the model-input tensor"
        ),
    }


# ---------------------------------------------------------------------------
# Target semantics
# ---------------------------------------------------------------------------


def target_semantics() -> dict:
    """The frozen policy / value / belief target definitions."""
    return {
        "decision_definition": (
            "one decision = one recorded action of the trajectory, indexed "
            "0..T-1 in ply order across both players; the acting player of "
            "the ply is the observer of every target"
        ),
        "policy": {
            "target": "the actual legal action chosen by the acting rule policy",
            "frame": MODEL_ACTION_FRAME,
            "engine_frame": ENGINE_ACTION_FRAME,
            "conversion": "stratego.model.action_frame.absolute_action_to_model",
            "loss": "masked cross entropy over legal model-frame actions",
            "masking": (
                "illegal actions are excluded from the policy normalization "
                "via masked_policy_log_probabilities (finite fill -1e9)"
            ),
            "zero_weight_rule": (
                "decisions with policy_weight == 0 contribute no policy "
                "gradient and still supervise value and belief"
            ),
        },
        "value": {
            "classes": list(VALUE_CLASS_ORDER),
            "indices": {
                "WIN": VALUE_WIN_INDEX,
                "DRAW": VALUE_DRAW_INDEX,
                "LOSS": VALUE_LOSS_INDEX,
            },
            "perspective": "acting player",
            "mapping": (
                "terminal winner == acting player -> WIN; draw -> DRAW; "
                "terminal winner == opponent -> LOSS (engine "
                "terminal_result_label is the outcome authority)"
            ),
            "loss": "categorical cross entropy over [WIN, DRAW, LOSS]",
            "bootstrapping": "none in Phase 8 (no model-value targets)",
        },
        "belief": {
            "belief_target_version": BELIEF_TARGET_VERSION,
            "square_frame": BELIEF_TARGET_SQUARE_FRAME,
            "type_count": BELIEF_TYPE_COUNT,
            "ignore_index": BELIEF_IGNORE_INDEX,
            "supervised": (
                "opponent pieces still unresolved to the acting player: "
                "target square = perspective-normalized current square, "
                "target class = true piece type"
            ),
            "excluded": (
                "own pieces, empty and lake squares, captured pieces, and "
                "opponent pieces whose identity is already legally known"
            ),
            "loss": "hidden-only masked cross entropy (per supervised square)",
            "privilege_rule": (
                "true types come from privileged replay state only after the "
                "public observation is constructed; they never enter any "
                "model input"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Baselines and the evaluation contract (`warmstart_eval_v1`)
# ---------------------------------------------------------------------------

#: Probability floor inside every baseline/metric logarithm.
METRIC_LOG_EPSILON = 1e-12

#: Game-level bootstrap: replicates and confidence level, frozen.
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_CONFIDENCE = 0.95


def evaluation_contract() -> dict:
    """`warmstart_eval_v1`: metrics, baselines, aggregation, bootstrap."""
    return {
        "eval_version": WARMSTART_EVAL_VERSION,
        "log_epsilon": METRIC_LOG_EPSILON,
        "policy_metric": {
            "population": "policy-supervised examples only (policy_weight > 0)",
            "weighting": (
                "weighted by policy_weight, normalized by the sum of weights "
                "— identical for the model and the baseline, matching the "
                "training-loss normalization"
            ),
            "model_ce": (
                "-log p_model(chosen action) with masked log-softmax over "
                "legal model-frame actions"
            ),
            "top1": "argmax over legal actions; ties broken by lowest action id",
        },
        "policy_baseline": {
            "definition": "uniform over legal actions: p(a) = 1 / legal_count",
            "cross_entropy": "ln(legal_count) per example",
            "expected_top1": "mean(1 / legal_count) over the same population",
        },
        "value_baseline": {
            "definition": (
                "one constant W/D/L distribution: class frequencies fitted on "
                "train selected examples only; validation and test reuse that "
                "frozen train prior"
            ),
            "cross_entropy": "-ln(max(prior[target], epsilon))",
            "brier": "sum_k (p_k - onehot_k)^2, mean over examples",
            "accuracy_tie_break": "lowest class index among tied maxima",
        },
        "belief_baseline": {
            "definition": (
                "observable unresolved-inventory marginal: p(type=t) = "
                "unresolved_remaining_count[t] / total_unresolved_remaining, "
                "computed from the acting player's observable information and "
                "applied independently to each unresolved hidden opponent piece"
            ),
            "cross_entropy": "-ln(max(p[true_type], epsilon)) per supervised piece",
            "top1_tie_break": "lowest piece-type index among tied maxima",
        },
        "aggregation_units": {
            "policy": "per decision (policy-supervised, weighted as above)",
            "value": "per selected decision",
            "belief": "per supervised hidden piece",
        },
        "bootstrap": {
            "unit": "game (all of a game's decisions/pieces resampled together)",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "method": (
                "numpy.random.default_rng(bootstrap_seed) draws one index "
                "matrix integers(0, n_games, size=(replicates, n_games)); "
                "each replicate recomputes the metric from per-game sufficient "
                "statistics; interval = 2.5/97.5 percentiles"
            ),
            "pairing": (
                "ratio and difference intervals reuse the same index matrix "
                "for model and baseline (paired at the game level)"
            ),
            "seeds": {
                "validation": VALIDATION_BOOTSTRAP_SEED,
                "test": TEST_BOOTSTRAP_SEED,
            },
        },
        "reporting": (
            "point estimate, 95% CI, number of games, number of decisions, "
            "number of hidden-piece targets"
        ),
    }


# ---------------------------------------------------------------------------
# Loss normalization (for Agent 4's trainer)
# ---------------------------------------------------------------------------


def loss_semantics() -> dict:
    """Frozen per-batch loss normalization (common contract section 18)."""
    return {
        "combination": "L = lambda_policy*L_policy + lambda_value*L_value + lambda_belief*L_belief",
        "policy": (
            "L_policy = sum_i(weight_i * CE_i) / sum_i(weight_i) over the "
            "batch's policy-eligible examples; a batch whose weights sum to "
            "zero contributes L_policy = 0"
        ),
        "value": "L_value = mean CE over the batch's selected decisions",
        "belief": (
            "L_belief = sum of per-square CE over supervised squares / "
            "max(supervised square count, 1) — the frozen "
            "stratego.model.losses.belief_loss normalization, so hidden-piece "
            "count cannot silently scale the belief head's influence"
        ),
    }


# ---------------------------------------------------------------------------
# Pilot candidate matrix and selection
# ---------------------------------------------------------------------------

#: Everything every pilot candidate must share, frozen.
PILOT_FIXED_CONTROLS = {
    "model": "C1",
    "precision": "float32",
    "device": "mps",
    "batch_size": 256,
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "lr_schedule": "linear_warmup_500_steps_then_constant",
    "model_init_seed": CANONICAL_C1_INIT_SEED,
    "update_budget": 5000,
    "validation_cadence_updates": 500,
    "corpus": SYNTHETIC_CORPUS_VERSION,
    "example_universe": (
        "the train split's frozen selected-decision universe under "
        f"{DECISION_SAMPLER_VERSION}"
    ),
}

#: The two frozen loss-weight profiles (lambda_policy, lambda_value,
#: lambda_belief).
PILOT_LOSS_PROFILES = {
    "balanced": {"lambda_policy": 1.0, "lambda_value": 1.0, "lambda_belief": 1.0},
    "policy_led": {"lambda_policy": 1.0, "lambda_value": 0.5, "lambda_belief": 0.5},
}

#: The three frozen learning rates.
PILOT_LEARNING_RATES = (1e-3, 3e-4, 1e-4)

#: 3 learning rates x 2 loss-weight profiles = 6 candidates, at the cap.
PILOT_CANDIDATES = tuple(
    {
        "candidate_id": f"ws_pilot_lr{rate:.0e}_{profile}".replace("e-0", "e-"),
        "learning_rate": rate,
        "loss_profile": profile,
        **PILOT_LOSS_PROFILES[profile],
    }
    for rate in PILOT_LEARNING_RATES
    for profile in PILOT_LOSS_PROFILES
)

PILOT_CANDIDATE_LIMIT = 6

#: Selection score and veto rules (common contract section 21), frozen.
PILOT_SELECTION = {
    "score": (
        "mean(r_policy, r_value, r_belief); r_head = validation head CE / "
        "frozen head baseline CE (warmstart_eval_v1 semantics); lower is better"
    ),
    "score_checkpoint": "the final pilot checkpoint (update 5000)",
    "hard_veto": [
        "non-finite loss/gradient/parameter",
        "target mismatch",
        "data split leak",
        "checkpoint/resume failure",
        "any component ratio > 1.05 at the final pilot checkpoint",
    ],
    "tie_break_order": [
        "lower selection score",
        "lower validation policy ratio",
        "higher measured training examples/s",
    ],
    "forbidden_evidence": [
        "test metrics",
        "Phase 4 game strength",
        "architecture changes",
        "teacher weights",
        "setup sampling",
    ],
}

#: Development budget (common contract section 28).
DEVELOPMENT_BUDGET = {
    "pilot_candidates_max": PILOT_CANDIDATE_LIMIT,
    "pilot_updates_per_config_max": 5000,
    "final_run_optimizer_steps_max": 25000,
}


def pilot_matrix() -> dict:
    """The complete frozen pilot contract."""
    if len(PILOT_CANDIDATES) > PILOT_CANDIDATE_LIMIT:
        raise WarmstartContractError(
            f"{len(PILOT_CANDIDATES)} pilot candidates exceed the frozen limit "
            f"of {PILOT_CANDIDATE_LIMIT}"
        )
    return {
        "fixed_controls": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in PILOT_FIXED_CONTROLS.items()
        },
        "candidates": [dict(candidate) for candidate in PILOT_CANDIDATES],
        "candidate_limit": PILOT_CANDIDATE_LIMIT,
        "allowed_dimensions": ["learning_rate", "loss_weights"],
        "selection": dict(PILOT_SELECTION),
        "development_budget": dict(DEVELOPMENT_BUDGET),
    }


# ---------------------------------------------------------------------------
# Final acceptance thresholds (frozen verbatim from the common contract)
# ---------------------------------------------------------------------------


def acceptance_thresholds() -> dict:
    """The machine-readable Phase 8 acceptance gates. Never relaxed later."""
    return {
        "playing_strength_vs_random": {
            "opponent": "random_legal (frozen Phase 4 random tier)",
            "setup_bank": SETUP_BANK_VERSION,
            "setup_bank_digest": EXPECTED_PHASE4_BANK_DIGEST,
            "evaluation_pairs": 1024,
            "games": 2048,
            "effective_win_rate_min": 0.950,
            "red_effective_win_rate_min": 0.900,
            "blue_effective_win_rate_min": 0.900,
            "paired_bootstrap_lower_bound_exclusive": 0.900,
            "illegal_moves_max": 0,
            "model_failures_max": 0,
            "non_finite_outputs_max": 0,
        },
        "improvement_over_initialization": {
            "opponent": (
                "canonical untrained C1: build_candidate_model('C1', "
                f"seed={CANONICAL_C1_INIT_SEED})"
            ),
            "paired_setup_cases_min": 512,
            "games_min": 1024,
            "effective_win_rate_min": 0.700,
            "paired_bootstrap_lower_bound_exclusive": 0.550,
        },
        "policy_learning": {
            "population": "sealed synthetic test examples",
            "ce_ratio_vs_uniform_legal_max": 0.90,
            "top1_must_beat_uniform_expected_top1": True,
        },
        "value_learning": {
            "population": "sealed synthetic test examples",
            "ce_ratio_vs_train_prior_max": 0.98,
            "brier_must_beat_train_prior": True,
        },
        "belief_learning": {
            "population": "hidden-only sealed test targets",
            "ce_ratio_vs_remaining_count_prior_max": 0.98,
            "top1_must_beat_remaining_count_prior": True,
        },
        "stability": {
            "finite_logits_fraction_required": 1.0,
            "max_legal_probability_threshold": 0.999,
            "fraction_above_threshold_max_exclusive": 0.95,
            "report": "normalized legal-action entropy distribution",
        },
        "statistics": {
            "bootstrap_unit": "game",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
        "development_budget": dict(DEVELOPMENT_BUDGET),
    }


# ---------------------------------------------------------------------------
# Held-out sealing — testable access rules
# ---------------------------------------------------------------------------

FINAL_EVALUATION_AGENT = 7

#: Test-corpus purposes. Structural integrity work is always allowed; any
#: model inference against test examples is Agent 7's alone.
TEST_CORPUS_ALLOWED_ALWAYS = ("structural_audit",)
TEST_CORPUS_AGENT7_ONLY = ("final_evaluation",)
TEST_CORPUS_PROHIBITED_BEFORE_7 = (
    "model_inference",
    "model_metric",
    "checkpoint_selection",
    "hyperparameter_selection",
    "early_stopping",
)

#: Phase 4 bank purposes. The accepted non-neural regression suite keeps
#: running; neural playing-strength evaluation is Agent 7's final gate only.
PHASE4_BANK_ALLOWED_ALWAYS = ("non_neural_regression",)
PHASE4_BANK_AGENT7_ONLY = ("final_random_evaluation", "final_ladder_evaluation")
PHASE4_BANK_PROHIBITED_BEFORE_7 = (
    "neural_playing_strength",
    "pilot_selection",
    "config_selection",
    "checkpoint_selection",
)


class HeldOutAccessError(WarmstartContractError):
    """A sealed resource was requested for a prohibited purpose."""


@dataclass(frozen=True)
class HeldOutAccess:
    """One authorized access to a sealed Phase 8 resource."""

    resource: str
    purpose: str
    phase8_agent: int


def check_test_corpus_access(purpose: str, *, phase8_agent: int) -> HeldOutAccess:
    """Authorize (or refuse) one access to the sealed test corpus.

    Pure and stateless: agents 1-6 may run structural audits; every purpose
    that runs the model or informs a selection raises before Agent 7; Agent 7
    may run the final evaluation. Statelessness is what keeps the rule
    testable now and still satisfiable after Agent 6 freezes the checkpoint.
    """
    agent = int(phase8_agent)
    if not 1 <= agent <= 7:
        raise HeldOutAccessError(f"unknown Phase 8 agent: {phase8_agent!r}")
    if purpose in TEST_CORPUS_ALLOWED_ALWAYS:
        return HeldOutAccess("test_corpus", purpose, agent)
    if purpose in TEST_CORPUS_AGENT7_ONLY:
        if agent == FINAL_EVALUATION_AGENT:
            return HeldOutAccess("test_corpus", purpose, agent)
        raise HeldOutAccessError(
            f"test-corpus purpose {purpose!r} is sealed until Agent "
            f"{FINAL_EVALUATION_AGENT}; agent {agent} may not open it"
        )
    if purpose in TEST_CORPUS_PROHIBITED_BEFORE_7:
        raise HeldOutAccessError(
            f"test-corpus purpose {purpose!r} is prohibited before Agent "
            f"{FINAL_EVALUATION_AGENT}; the sealed gate for Agent 7 is "
            f"'final_evaluation'"
        )
    raise HeldOutAccessError(f"unknown test-corpus purpose: {purpose!r}")


def check_phase4_bank_access(purpose: str, *, phase8_agent: int) -> HeldOutAccess:
    """Authorize (or refuse) one access to the frozen Phase 4 bank."""
    agent = int(phase8_agent)
    if not 1 <= agent <= 7:
        raise HeldOutAccessError(f"unknown Phase 8 agent: {phase8_agent!r}")
    if purpose in PHASE4_BANK_ALLOWED_ALWAYS:
        return HeldOutAccess("phase4_bank", purpose, agent)
    if purpose in PHASE4_BANK_AGENT7_ONLY:
        if agent == FINAL_EVALUATION_AGENT:
            return HeldOutAccess("phase4_bank", purpose, agent)
        raise HeldOutAccessError(
            f"Phase 4 bank purpose {purpose!r} is sealed until Agent "
            f"{FINAL_EVALUATION_AGENT}; agent {agent} may not open it"
        )
    if purpose in PHASE4_BANK_PROHIBITED_BEFORE_7:
        raise HeldOutAccessError(
            f"Phase 4 bank purpose {purpose!r} is prohibited before Agent "
            f"{FINAL_EVALUATION_AGENT}: neural playing strength must not "
            f"select pilots, configurations, or checkpoints"
        )
    raise HeldOutAccessError(f"unknown Phase 4 bank purpose: {purpose!r}")


def sealing_rules() -> dict:
    """The serializable held-out access policy."""
    return {
        "test_corpus": {
            "allowed_always": list(TEST_CORPUS_ALLOWED_ALWAYS),
            "agent7_only": list(TEST_CORPUS_AGENT7_ONLY),
            "prohibited_before_agent_7": list(TEST_CORPUS_PROHIBITED_BEFORE_7),
        },
        "phase4_bank": {
            "allowed_always": list(PHASE4_BANK_ALLOWED_ALWAYS),
            "agent7_only": list(PHASE4_BANK_AGENT7_ONLY),
            "prohibited_before_agent_7": list(PHASE4_BANK_PROHIBITED_BEFORE_7),
        },
        "enforcement": (
            "stratego.training.warmstart_contract.check_test_corpus_access / "
            "check_phase4_bank_access; pure, stateless, regression-tested"
        ),
        "validation_corpus": (
            "may select pilot configuration, best checkpoint and early "
            "stopping; may never update weights"
        ),
        "train_corpus": (
            "may update weights, fit the value prior, and drive optimization "
            "statistics"
        ),
    }


# ---------------------------------------------------------------------------
# Corpus storage expectations (for Agent 2)
# ---------------------------------------------------------------------------


def corpus_storage_schema() -> dict:
    """What Agent 2 must persist for every committed game."""
    return {
        "preferred_root": "data/warmstart/synthetic_warmstart_corpus_v1/",
        "redirect": (
            "the root may be redirected to the external volume by explicit "
            "configuration; the manifest records the actual location and "
            "free space"
        ),
        "trajectory": {
            "version": TRAJECTORY_VERSION,
            "snapshot_interval": DEFAULT_SNAPSHOT_INTERVAL,
            "role": "replay authority: setups, actions, terminal state",
        },
        "synthetic_metadata_fields": [
            "game_id",
            "corpus_version",
            "corpus_split",
            "cell_index",
            "matchup_ordinal",
            "red_policy_id",
            "red_policy_version",
            "blue_policy_id",
            "blue_policy_version",
            "red_policy_seed",
            "blue_policy_seed",
            "setup_root_seed",
            "setup_provenance (both sides, setup_provenance_v1)",
            "policy_weight_red",
            "policy_weight_blue",
            "terminal_result",
            "terminal_reason",
            "total_decisions",
        ],
        "commit_rule": (
            "a game is trainable only once a commit record exists, written "
            "after both the trajectory payload and the synthetic/setup "
            "metadata exist and verify; resume reconciles persisted, "
            "metadata and commit ids, never duplicates a committed game, "
            "never exposes an orphan, and deterministically rebuilds or "
            "discards uncommitted work"
        ),
        "finalization_requirements": {
            "orphan_trajectory_records": 0,
            "orphan_metadata_records": 0,
            "duplicate_committed_ids": 0,
            "missing_committed_records": 0,
        },
    }


# ---------------------------------------------------------------------------
# Frozen upstream verification
# ---------------------------------------------------------------------------


def verify_frozen_upstream(*, include_library_digest: bool = True) -> list:
    """Every disagreement between the frozen expectations and live source.

    The Phase 4 bank digest is intentionally not recomputed here (bank
    generation is seconds of work); the Agent 1 runner regenerates and checks
    it explicitly and records the observation in the artifact.
    """
    checks = [
        ("rules_version", EXPECTED_RULES_VERSION, RULES_VERSION),
        ("engine_version", EXPECTED_ENGINE_VERSION, IMPLEMENTATION_VERSION),
        ("observation_version", EXPECTED_OBSERVATION_VERSION, OBSERVATION_VERSION),
        ("observation_channels", 127, OBSERVATION_CHANNELS),
        ("model_contract_version", EXPECTED_MODEL_CONTRACT_VERSION, MODEL_CONTRACT_VERSION),
        ("action_encoding_version", EXPECTED_ACTION_ENCODING_VERSION, ACTION_ENCODING_VERSION),
        ("model_action_frame", "perspective_normalized_squares", MODEL_ACTION_FRAME),
        ("engine_action_frame", "absolute_engine_squares", ENGINE_ACTION_FRAME),
        ("trajectory_version", EXPECTED_TRAJECTORY_VERSION, TRAJECTORY_VERSION),
        ("snapshot_interval", EXPECTED_SNAPSHOT_INTERVAL, DEFAULT_SNAPSHOT_INTERVAL),
        ("setup_bank_version", "evaluation_setup_bank_v1", SETUP_BANK_VERSION),
        ("setup_bank_size", 1024, DEFAULT_BANK_SIZE),
        ("setup_bank_root_seed", 20260101, DEFAULT_BANK_ROOT_SEED),
        ("setup_library_version", "setup_library_v1", SETUP_LIBRARY_VERSION),
        ("sampler_version", "setup_sampler_v1", SAMPLER_VERSION),
        ("perturbation_version", "setup_perturbation_v1", PERTURBATION_VERSION),
        ("perturbation_seed_encoding", "seed_encoding_v1", PERTURBATION_SEED_ENCODING),
        ("setup_source_version", "setup_source_v1", SETUP_SOURCE_VERSION),
        ("default_profile", EXPECTED_SETUP_PROFILE, DEFAULT_PROFILE.name),
        ("profile_reflection_probability", 0.5, DEFAULT_PROFILE.reflection_probability),
        ("profile_perturbation_probability", 0.5, DEFAULT_PROFILE.perturbation_probability),
        ("profile_swap_counts", (1, 2, 3, 4, 5, 6), tuple(DEFAULT_PROFILE.swap_counts)),
        ("c1_config_digest", EXPECTED_C1_CONFIG_DIGEST, candidate_config("C1").digest()),
    ]
    if include_library_digest:
        checks.append(
            ("library_digest", EXPECTED_LIBRARY_DIGEST, load_library_index().content_digest)
        )
    return [
        f"{name}: expected {expected!r}, found {observed!r}"
        for name, expected, observed in checks
        if expected != observed
    ]


# ---------------------------------------------------------------------------
# The complete serialized contract
# ---------------------------------------------------------------------------


def contract_document() -> dict:
    """The full frozen `warmstart_training_contract_v1` as one document."""
    return {
        "contract_version": WARMSTART_TRAINING_CONTRACT_VERSION,
        "corpus_version": SYNTHETIC_CORPUS_VERSION,
        "decision_sampler_version": DECISION_SAMPLER_VERSION,
        "example_version": WARMSTART_EXAMPLE_VERSION,
        "eval_version": WARMSTART_EVAL_VERSION,
        "frozen_upstream": {
            "rules": EXPECTED_RULES_VERSION,
            "reference_engine": EXPECTED_ENGINE_VERSION,
            "observation": EXPECTED_OBSERVATION_VERSION,
            "model_contract": EXPECTED_MODEL_CONTRACT_VERSION,
            "action_encoding": EXPECTED_ACTION_ENCODING_VERSION,
            "trajectory": EXPECTED_TRAJECTORY_VERSION,
            "snapshot_interval": EXPECTED_SNAPSHOT_INTERVAL,
            "c1_parameters": EXPECTED_C1_PARAMETERS,
            "c1_config_digest": EXPECTED_C1_CONFIG_DIGEST,
            "phase4_bank": SETUP_BANK_VERSION,
            "phase4_bank_digest": EXPECTED_PHASE4_BANK_DIGEST,
            "setup_library": SETUP_LIBRARY_VERSION,
            "setup_library_digest": EXPECTED_LIBRARY_DIGEST,
            "setup_sampler": SAMPLER_VERSION,
            "setup_perturbation": PERTURBATION_VERSION,
            "seed_encoding": PERTURBATION_SEED_ENCODING,
            "setup_source": SETUP_SOURCE_VERSION,
            "setup_profile": EXPECTED_SETUP_PROFILE,
        },
        "corpus_rules_config": {
            "context": CORPUS_RULES.context,
            "battleless_move_limit": CORPUS_RULES.battleless_move_limit,
            "absolute_move_limit": CORPUS_RULES.absolute_move_limit,
            "note": (
                "corpus games use the frozen TRAINING_RULES; Phase 4 strength "
                "gates keep the frozen EVALUATION_RULES via the untouched "
                "Phase 4 machinery"
            ),
        },
        "teacher_population": teacher_population(),
        "policy_supervision_weights": dict(POLICY_SUPERVISION_WEIGHTS),
        "matchup_schedule": matchup_schedule(),
        "setup_sources": setup_source_configuration(),
        "game_identity": {
            "id_function": "stratego.training.warmstart_seed.synthetic_game_id",
            "fields": [
                "corpus_version",
                "corpus_master_seed",
                "split",
                "red policy id@version",
                "blue policy id@version",
                "per-cell game ordinal",
            ],
            "format_example": synthetic_game_id(
                "train", "random_legal@1.0.0", "random_legal@1.0.0", 0
            ),
            "seed_domains": {
                "setup_root": "setup identity / setup-source root seed",
                "policy:red": "red rule-policy match-level seed",
                "policy:blue": "blue rule-policy match-level seed",
                "decision_sampler": "per-bin decision-selection streams",
            },
            "per_ply_policy_randomness": (
                "derive_decision_seed(policy_seed, ply) — the frozen Phase 4 "
                "per-ply derivation"
            ),
            "rng_rule": "no global RNG cursor; every stream is a pure function of identity",
        },
        "canonical_seeds": dict(CANONICAL_SEEDS),
        "decision_sampler": {
            "version": DECISION_SAMPLER_VERSION,
            "max_decisions_per_game": MAX_DECISIONS_PER_GAME,
            "short_game_rule": "T <= 64 selects every decision index 0..T-1",
            "long_game_rule": (
                "T > 64: bin b covers [floor(b*T/64), floor((b+1)*T/64)); "
                "selected index = lo + (decision_bin_seed(game_id, b) % "
                "(hi - lo)); bins are disjoint and ascending, so selection is "
                "without replacement and already sorted"
            ),
            "outcome_independence": (
                "game outcome, teacher strength, future value and model "
                "predictions never influence selection"
            ),
        },
        "example_schema": example_schema(),
        "target_semantics": target_semantics(),
        "evaluation_contract": evaluation_contract(),
        "loss_semantics": loss_semantics(),
        "pilot_matrix": pilot_matrix(),
        "acceptance_thresholds": acceptance_thresholds(),
        "sealing_rules": sealing_rules(),
        "corpus_storage_schema": corpus_storage_schema(),
        "phase_boundaries": {
            "not_reinforcement_learning": (
                "no self-play, PPO, advantage estimation, KL regularization, "
                "dynamic damping, EMA stabilization, learned setup selection, "
                "search targets, or human data in Phase 8"
            ),
            "phase_9_receives": (
                "frozen warm-start checkpoint + digests + curves + sealed-test "
                "results + regeneration instructions"
            ),
        },
    }


def contract_digest() -> str:
    """SHA-256 over the canonical JSON of the frozen contract document."""
    canonical = json.dumps(contract_document(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "BELIEF_SUPERVISION_ELIGIBLE",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATES",
    "CORPUS_RULES",
    "DEVELOPMENT_BUDGET",
    "EXAMPLE_FIELDS",
    "EXPECTED_C1_CONFIG_DIGEST",
    "EXPECTED_C1_PARAMETERS",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_LIBRARY_DIGEST",
    "EXPECTED_PHASE4_BANK_DIGEST",
    "EXPECTED_SETUP_PROFILE",
    "EXPECTED_TEACHER_COUNT",
    "EXPECTED_TEACHER_ROSTER",
    "FINAL_EVALUATION_AGENT",
    "HELD_OUT_SETUP_JUSTIFICATIONS",
    "HeldOutAccess",
    "HeldOutAccessError",
    "METRIC_LOG_EPSILON",
    "PHASE4_BANK_AGENT7_ONLY",
    "PHASE4_BANK_ALLOWED_ALWAYS",
    "PHASE4_BANK_PROHIBITED_BEFORE_7",
    "PILOT_CANDIDATES",
    "PILOT_CANDIDATE_LIMIT",
    "PILOT_FIXED_CONTROLS",
    "PILOT_LEARNING_RATES",
    "PILOT_LOSS_PROFILES",
    "PILOT_SELECTION",
    "POLICY_SUPERVISION_WEIGHTS",
    "ROLE_POLICY_WEIGHTS",
    "SCHEDULE_TOTALS",
    "SETUP_SOURCE_ENVIRONMENT_ID",
    "SETUP_SOURCE_GENERATION",
    "TEST_CORPUS_AGENT7_ONLY",
    "TEST_CORPUS_ALLOWED_ALWAYS",
    "TEST_CORPUS_PROHIBITED_BEFORE_7",
    "VALUE_SUPERVISION_ELIGIBLE",
    "WARMSTART_EVAL_VERSION",
    "WARMSTART_EXAMPLE_VERSION",
    "WARMSTART_TRAINING_CONTRACT_VERSION",
    "WarmstartContractError",
    "acceptance_thresholds",
    "check_phase4_bank_access",
    "check_test_corpus_access",
    "contract_digest",
    "contract_document",
    "corpus_setup_source",
    "corpus_storage_schema",
    "evaluation_contract",
    "example_schema",
    "iter_game_identities",
    "loss_semantics",
    "matchup_schedule",
    "ordered_matchup_cells",
    "pilot_matrix",
    "policy_weight",
    "sealing_rules",
    "setup_source_configuration",
    "target_semantics",
    "teacher_population",
    "teacher_tokens",
    "verify_frozen_upstream",
    "verify_teacher_roster",
]
