"""Optional Phase 10B: playing evaluation cells, resumably.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 15, 16 and 20.

One work unit is one `(arm, matchup, case slice)`. Units are addressed by
identity and written once, so a re-run plays only what is missing and a
resumed pass converges to the same rows. Nothing here decides a threshold or
a verdict — it produces primitive per-game rows and hands them to
:mod:`stratego.evaluation.phase10b_acceptance`, which recomputes everything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
from ..training.phase10b_contract import HEAD_TO_HEAD_MATCHUPS
from .match_runner import ON_POLICY_ERROR_QUARANTINE
from .phase10b_eval import (
    ARM_BASELINE,
    ARM_CANDIDATE,
    BASELINE_MOVE_POLICY_ID,
    CANDIDATE_MOVE_POLICY_ID,
    EVAL_DTYPE,
    EXTERNAL_OPPONENT_POLICY_IDS,
    NEURAL_OPPONENT_MATCHUP,
    PHASE8_ANCHOR_CANDIDATE_ID,
    Phase10BEvalError,
    game_setups,
    own_side_draws,
    play_cell_game,
)


@dataclass
class ArmPolicies:
    """The two neural arms and the Phase 8 anchor, as live policy objects."""

    candidate_ref: object
    candidate_policy: object
    baseline_ref: object
    baseline_policy: object
    anchor_ref: object
    anchor_policy: object
    rule_policies: dict


def build_arm_policies(
    *,
    candidate_export: "str | Path",
    baseline_export: "str | Path",
    anchor_export: "str | Path",
    device: str = "mps",
    label: str = "p10b",
):
    """Load both arms and every catalogued opponent. Returns `(arms, close)`.

    Both arms play greedy float32 `single_request` — the accepted evaluation
    move behaviour, no search — and each holds its own inference owner so a
    row can never be attributed to the other checkpoint's weights.
    """
    from .neural_worker import (
        DECISION_MODE_GREEDY,
        InferenceOwner,
        LocalInferenceChannel,
        RemoteNeuralPolicy,
        neural_policy_ref,
    )
    from .registry import build_policy

    owners = {}
    for name, path in (
        ("candidate", candidate_export),
        ("baseline", baseline_export),
        ("anchor", anchor_export),
    ):
        owners[name] = InferenceOwner(
            Path(path),
            decision_mode=DECISION_MODE_GREEDY,
            device=device,
            dtype=EVAL_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"{label}_{name}",
        )

    def policy_for(name, policy_id):
        ref = neural_policy_ref(
            policy_id, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
        )
        return ref, RemoteNeuralPolicy(
            ref, LocalInferenceChannel(owners[name]), decision_mode=DECISION_MODE_GREEDY
        )

    candidate_ref, candidate_policy = policy_for("candidate", CANDIDATE_MOVE_POLICY_ID)
    baseline_ref, baseline_policy = policy_for("baseline", BASELINE_MOVE_POLICY_ID)
    anchor_ref, anchor_policy = policy_for("anchor", PHASE8_ANCHOR_CANDIDATE_ID)

    arms = ArmPolicies(
        candidate_ref=candidate_ref,
        candidate_policy=candidate_policy,
        baseline_ref=baseline_ref,
        baseline_policy=baseline_policy,
        anchor_ref=anchor_ref,
        anchor_policy=anchor_policy,
        rule_policies={
            matchup: build_policy(policy_id)
            for matchup, policy_id in EXTERNAL_OPPONENT_POLICY_IDS.items()
        },
    )

    def close():
        for owner in owners.values():
            owner.close()

    return arms, close


def _seats(arms: ArmPolicies, arm: str, matchup: str):
    """`(own_ref, own_policy, opponent_ref, opponent_policy)` for one cell."""
    from .registry import policy_ref

    if arm == ARM_CANDIDATE:
        own_ref, own_policy = arms.candidate_ref, arms.candidate_policy
    elif arm == ARM_BASELINE:
        own_ref, own_policy = arms.baseline_ref, arms.baseline_policy
    else:
        raise Phase10BEvalError(f"unknown arm {arm!r}")

    if matchup in HEAD_TO_HEAD_MATCHUPS:
        if arm != ARM_CANDIDATE:
            raise Phase10BEvalError(
                f"{matchup} is head-to-head; only the candidate arm owns it"
            )
        return own_ref, own_policy, arms.baseline_ref, arms.baseline_policy
    if matchup == NEURAL_OPPONENT_MATCHUP:
        return own_ref, own_policy, arms.anchor_ref, arms.anchor_policy
    return (
        own_ref,
        own_policy,
        policy_ref(EXTERNAL_OPPONENT_POLICY_IDS[matchup]),
        arms.rule_policies[matchup],
    )


def run_cell(
    cases,
    arm: str,
    matchup: str,
    *,
    arms: ArmPolicies,
    setup_source,
    record_actions: bool = False,
    on_policy_error: str = ON_POLICY_ERROR_QUARANTINE,
) -> dict:
    """Play every game of one `(arm, matchup)` cell over `cases`."""
    own_ref, own_policy, opponent_ref, opponent_policy = _seats(arms, arm, matchup)
    rows = []
    started = time.perf_counter()
    for case in cases:
        own = own_side_draws(setup_source, case, matchup)
        for setup_row in game_setups(case, matchup, own):
            spec, result = play_cell_game(
                case,
                setup_row,
                matchup,
                arm=arm,
                own_ref=own_ref,
                opponent_ref=opponent_ref,
                own_policy=own_policy,
                opponent_policy=opponent_policy,
                record_actions=record_actions,
                on_policy_error=on_policy_error,
            )
            draw = own[setup_row["own_color"]]
            rows.append(
                {
                    "match_id": result.match_id,
                    "case_id": case.case_id,
                    "case_family": case.family_id,
                    "case_index": case.case_index,
                    "game_index": setup_row["game_index"],
                    "own_color": setup_row["own_color"],
                    "arm": arm,
                    "matchup": matchup,
                    "setup_source": draw.source,
                    "candidate_result": result.candidate_result,
                    "score": None if result.errored else float(result.candidate_score),
                    "terminal_reason": result.terminal_reason,
                    "plies": int(result.plies),
                    "decisions": int(result.decisions),
                    "replay_digest": result.replay_digest,
                    "own_fingerprint": draw.final_setup_fingerprint,
                    "own_base_setup_id": draw.base_setup_id,
                    "own_family_id": draw.family_id,
                    "own_branch": draw.branch,
                    "policy_error": result.policy_error,
                    "policy_error_category": result.policy_error_category,
                    "setup_bank_version": result.setup_bank_version,
                    "root_seed": int(spec.root_seed),
                }
            )
    return {
        "arm": arm,
        "matchup": matchup,
        "cases": len(cases),
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }


__all__ = ["ArmPolicies", "build_arm_policies", "run_cell"]
