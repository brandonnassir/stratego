"""Phase 17 Agent 5: the cross-machine identity fixture.

Specification source: Agent 5 instruction section 5 -- "Platform floating-point
differences must be measured and documented, not assumed bit-identical."

Why this exists at all
-----------------------
Both machines score the candidate by playing games in which every decision is
an **argmax**. Argmax is discrete: a 1e-7 disagreement in one logit is either
completely invisible or it flips a move and changes the rest of the game. So
"the numbers are close" is not a claim that can be checked by comparing two
EWRs -- two machines could agree on 0.7542 while having played different games,
and could disagree on the EWR while being numerically identical.

The fixture therefore compares four things, at three different strengths:

```text
identities        model-state, pack, evaluator-source digests   exact
setup tokens      all 120 generated boards under fixed seeds    exact
game actions      every action of N whole games, both lanes     exact
logits            float32 policy logits at N fixed positions    MEASURED
```

The first three are yes/no. The fourth is the only one that is allowed to
differ, and its observed maximum difference is *recorded* rather than compared
against a hope. If the actions match exactly, the logit delta is below the
threshold that matters no matter what its value is; if the actions differ, the
logit delta tells the operator whether the cause is float drift or a genuinely
different model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .candidate import (
    build_move_owner,
    build_setup_model,
    materialize_move_checkpoint,
    verify_bundle,
)
from .contract import (
    CROSS_MACHINE_PROBABILITY_TOLERANCE,
    EVALUATOR_VERSION,
    LANE_JOINT,
    LANE_MOVE_ONLY,
    Phase17EvaluationError,
    json_digest,
)
from .lanes import Phase17CandidateSeat, generate_joint_setups, joint_plan
from .opponents import build_opponent_owners, verify_opponent_files
from .pack import load_composite_pack, plan_from_row

FIXTURE_VERSION = "phase17_cross_machine_fixture_v1"

#: How many whole games per lane the fixture replays action-for-action.
DEFAULT_FIXTURE_CASES = 8


def _probe_logits(owner, plan, owners) -> dict:
    """Float32 policy logits at the opening position of one board."""
    from ...engine.legal_moves import legal_actions
    from ...engine.state import create_game
    from ...evaluation.match_spec import EVALUATION_RULES
    from ...evaluation.neural_worker import (
        DECISION_MODE_GREEDY,
        InferenceRequest,
        LocalInferenceChannel,
        RemoteNeuralPolicy,
    )
    from ...evaluation.policy import build_policy_input
    from ...search.phase15.matchplay import build_spec, opponent_seat, player_ref

    reference, _ = opponent_seat(plan, owners)
    spec = build_spec(plan, reference)
    state = create_game(
        plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id=spec.game_id
    )
    legal = legal_actions(state)
    policy = RemoteNeuralPolicy(
        player_ref(), LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY
    )
    request = build_policy_input(
        state,
        policy=player_ref(),
        policy_seed=spec.policy_seed_for(state.acting_player),
        requirements=policy.requirements,
        suite_version=spec.suite_version,
        match_id=spec.match_id,
        paired_unit_id=spec.paired_unit_id,
        legal=legal,
    )
    logits = owner.probe_policy_logits([InferenceRequest.from_policy_input(request)])[0]
    selected = int(policy.decide_checked(request).selected_action_id)
    values = [float(value) for value in logits.tolist()]
    return {
        "board_id": plan.board_id,
        "acting_player": int(state.acting_player),
        "legal_action_count": len(legal),
        "legal_actions": [int(action) for action in legal],
        "policy_logits": values,
        "policy_logits_digest": json_digest([repr(value) for value in values]),
        "selected_action": selected,
    }


def build_fixture(
    bundle_path: "Path | str",
    *,
    root: "Path | str" = ".",
    pack_path: "Path | str",
    expected_pack_digest: "str | None" = None,
    cases: int = DEFAULT_FIXTURE_CASES,
) -> dict:
    """Everything two machines must agree on, computed from fixed inputs."""
    from ...search.phase15.matchplay import play_board
    from .evaluator import evaluator_source_digest, host_identity, utc_now

    verified, payload = verify_bundle(bundle_path)
    pack = load_composite_pack(pack_path, expected_digest=expected_pack_digest)
    opponents = verify_opponent_files(root=root)
    owners = build_opponent_owners(root=root)

    with tempfile.TemporaryDirectory(prefix="phase17_fixture_") as scratch:
        checkpoint = materialize_move_checkpoint(payload, scratch, verified=verified)
        owner = build_move_owner(checkpoint, name="phase17_fixture_candidate")
        seat = Phase17CandidateSeat(owner, verified.candidate_id)

        setup_model = build_setup_model(payload, verified=verified)
        generated = generate_joint_setups(
            setup_model,
            pack,
            setup_digest=verified.setup_ema_model_state_digest,
            iteration=verified.iteration,
        )

        move_rows = pack["lanes"][LANE_MOVE_ONLY]["cases"][: int(cases)]
        joint_rows = pack["lanes"][LANE_JOINT]["cases"][: int(cases)]

        probes = [_probe_logits(owner, plan_from_row(row), owners) for row in move_rows]

        games = []
        for lane, rows in ((LANE_MOVE_ONLY, move_rows), (LANE_JOINT, joint_rows)):
            for row in rows:
                plan = (
                    plan_from_row(row)
                    if lane == LANE_MOVE_ONLY
                    else joint_plan(row, generated)
                )
                record = play_board(
                    plan, seat, owners, probe=None, preset_id=lane, keep_moves=True
                )
                actions = [int(move["action_id"]) for move in record.moves]
                games.append({
                    "lane": lane,
                    "board_id": row["board_id"],
                    "plies": int(record.plies),
                    "player_decisions": int(record.player_decisions),
                    "outcome": record.outcome,
                    "effective_score": float(record.effective_score),
                    "terminal_reason": record.terminal_reason,
                    "candidate_actions": actions,
                    "candidate_actions_digest": json_digest(actions),
                })

    fixture = {
        "fixture_version": FIXTURE_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "work_package": "phase17",
        "built_utc": utc_now(),
        "host_identity": host_identity(),
        "cases": int(cases),
        "identities": {
            "candidate_id": verified.candidate_id,
            "run_id": verified.run_id,
            "bundle_file_sha256": verified.file_sha256,
            "bundle_manifest_digest": verified.manifest_digest,
            "move_ema_model_state_digest": verified.move_ema_model_state_digest,
            "setup_ema_model_state_digest": verified.setup_ema_model_state_digest,
            "benchmark_pack_digest": pack["pack_digest"],
            "evaluator_source_digest": evaluator_source_digest(root=root),
            "opponent_file_sha256": {
                name: record["file_sha256"] for name, record in opponents.items()
            },
        },
        "setup_tokens": {
            board_id: {
                "canonical_setup": record["canonical_setup"],
                "canonical_fingerprint": record["canonical_fingerprint"],
                "root_seed": record["root_seed"],
            }
            for board_id, record in sorted(generated.items())
        },
        "logit_probes": probes,
        "games": games,
        "tolerance": CROSS_MACHINE_PROBABILITY_TOLERANCE,
    }
    fixture["fixture_digest"] = json_digest(
        {key: value for key, value in fixture.items()
         if key not in ("fixture_digest", "host_identity", "built_utc")}
    )
    return fixture


def compare_fixtures(left: dict, right: dict) -> dict:
    """Measure the difference between two machines' fixtures."""
    for side in (left, right):
        if side.get("fixture_version") != FIXTURE_VERSION:
            raise Phase17EvaluationError(
                f"not a {FIXTURE_VERSION} document: {side.get('fixture_version')!r}"
            )

    findings: dict = {
        "left_host": (left.get("host_identity") or {}).get("hostname"),
        "right_host": (right.get("host_identity") or {}).get("hostname"),
        "left_torch": (left.get("host_identity") or {}).get("torch"),
        "right_torch": (right.get("host_identity") or {}).get("torch"),
        "mismatches": [],
    }

    for key, value in left["identities"].items():
        if right["identities"].get(key) != value:
            findings["mismatches"].append(f"identity.{key}")
    findings["identities_match"] = not findings["mismatches"]

    left_tokens = left.get("setup_tokens") or {}
    right_tokens = right.get("setup_tokens") or {}
    token_diffs = [
        board_id
        for board_id in sorted(set(left_tokens) | set(right_tokens))
        if left_tokens.get(board_id, {}).get("canonical_fingerprint")
        != right_tokens.get(board_id, {}).get("canonical_fingerprint")
    ]
    findings["setup_boards_compared"] = len(left_tokens)
    findings["setup_tokens_match"] = not token_diffs
    findings["setup_token_mismatches"] = token_diffs[:10]

    left_games = {(row["lane"], row["board_id"]): row for row in left.get("games", [])}
    right_games = {(row["lane"], row["board_id"]): row for row in right.get("games", [])}
    action_diffs, score_diffs = [], []
    for key in sorted(set(left_games) | set(right_games)):
        one, two = left_games.get(key), right_games.get(key)
        if one is None or two is None:
            action_diffs.append(f"{key}: missing on one side")
            continue
        if one["candidate_actions"] != two["candidate_actions"]:
            first = next(
                (
                    index
                    for index, (a, b) in enumerate(
                        zip(one["candidate_actions"], two["candidate_actions"])
                    )
                    if a != b
                ),
                min(len(one["candidate_actions"]), len(two["candidate_actions"])),
            )
            action_diffs.append(f"{key[0]}/{key[1]}: first divergence at decision {first}")
        if one["effective_score"] != two["effective_score"]:
            score_diffs.append(f"{key[0]}/{key[1]}")
    findings["games_compared"] = len(left_games)
    findings["game_actions_match"] = not action_diffs
    findings["game_action_mismatches"] = action_diffs[:10]
    findings["final_scoring_match"] = not score_diffs
    findings["scoring_mismatches"] = score_diffs[:10]

    left_probes = {row["board_id"]: row for row in left.get("logit_probes", [])}
    right_probes = {row["board_id"]: row for row in right.get("logit_probes", [])}
    max_logit_delta = 0.0
    legal_mismatch = []
    decision_mismatch = []
    for board_id in sorted(set(left_probes) | set(right_probes)):
        one, two = left_probes.get(board_id), right_probes.get(board_id)
        if one is None or two is None:
            legal_mismatch.append(f"{board_id}: missing on one side")
            continue
        if one["legal_actions"] != two["legal_actions"]:
            legal_mismatch.append(board_id)
        if one["selected_action"] != two["selected_action"]:
            decision_mismatch.append(board_id)
        for a, b in zip(one["policy_logits"], two["policy_logits"]):
            max_logit_delta = max(max_logit_delta, abs(float(a) - float(b)))
    findings["positions_probed"] = len(left_probes)
    findings["legal_moves_match"] = not legal_mismatch
    findings["legal_move_mismatches"] = legal_mismatch[:10]
    findings["probe_decisions_match"] = not decision_mismatch
    findings["probe_decision_mismatches"] = decision_mismatch[:10]
    findings["max_abs_logit_delta"] = max_logit_delta
    findings["bit_identical_logits"] = max_logit_delta == 0.0
    findings["tolerance"] = left.get("tolerance", CROSS_MACHINE_PROBABILITY_TOLERANCE)
    findings["within_tolerance"] = max_logit_delta <= float(findings["tolerance"])

    findings["identical"] = (
        findings["identities_match"]
        and findings["setup_tokens_match"]
        and findings["game_actions_match"]
        and findings["final_scoring_match"]
        and findings["legal_moves_match"]
        and findings["probe_decisions_match"]
    )
    findings["pass"] = findings["identical"] and findings["within_tolerance"]
    findings["interpretation"] = (
        "the two machines played identical games from identical weights"
        if findings["identical"]
        else "the two machines DIVERGED; see the mismatch lists before trusting any lane result"
    )
    return findings


__all__ = [
    "DEFAULT_FIXTURE_CASES",
    "FIXTURE_VERSION",
    "build_fixture",
    "compare_fixtures",
]
