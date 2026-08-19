"""Phase 10 Agent 7: the sealed final-test evaluation.

Specification sources:

- `07_AGENT_7_FINAL_ACCEPTANCE_AND_FREEZE.md` ("Open final bank once",
  "Required final matchups", "Recompute hard gates independently")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Phase 10 evaluation
  banks", "Final acceptance gates", "Statistics")

What one final-test game is
---------------------------
Identical in structure to a validation game — the shared primitives
(:func:`~stratego.evaluation.phase10_validation.game_setups`,
:class:`~stratego.evaluation.phase10_validation.FrozenSeedPolicy`,
:func:`~stratego.evaluation.phase10_validation.neutral_own_side`,
:func:`~stratego.evaluation.phase10_validation.learned_own_side`) are reused
unchanged — but the case comes from `phase10_test_bank_v1` and the game
identity carries the *test* bank version, so no final-test stream, cache
path or stored row can collide with a validation one.

Selection is closed
-------------------
Agent 5 permanently selected P10-D and Agent 6 froze it into
`phase10_system_v1`. This module therefore refuses to evaluate any learned
candidate other than the pinned winner: the final test measures the single
frozen system against the fixed `neutral_v1` baseline, and a second
candidate on the sealed bank would be exactly the reopened selection the
contract forbids.

Sealing
-------
`final_cases()` is the only test-bank entry point here. Constructing the
cases is the `structural_audit` purpose every agent is allowed; *playing*
them is the `final_evaluation` purpose reserved to Agent 7, and the caller
(the Agent 7 harness) records that access in its ledger before the first
game.
"""

from __future__ import annotations

from ..engine.constants import BLUE, EVALUATION_RULES, RED
from ..training.phase10_contract import (
    MATCHUP_LEARNED_VS_NEUTRAL,
    MATCHUP_TOKENS,
    NEUTRAL_PROFILE_NAME,
    TEST_BANK_VERSION,
    Phase10ContractError,
)
from ..training.phase10_seed import CASE_GAME_COLOR, CASE_GAME_INDICES, case_match_seed
from ..training.phase10_soak import SELECTED_CANDIDATE_ID
from .match_runner import ON_POLICY_ERROR_RAISE, play_match
from .match_spec import MatchSpec
from .phase10_validation import (
    ARM_LEARNED,
    ARM_NEUTRAL,
    ARMS,
    FrozenSeedPolicy,
    Phase10ValidationError,
    single_game_bank,
)

FINAL_BANK = "test"

#: Version tag of the final-test game-identity scheme, distinct from the
#: validation scheme so a stored row names the evaluation that produced it.
PHASE10_FINAL_VERSION = "phase10_final_eval_v1"

#: The matchups the neutral baseline arm plays on the final bank — every
#: externally-opposed matchup; the direct matchup is the comparison itself.
FINAL_NEUTRAL_ARM_MATCHUPS = tuple(
    token for token in MATCHUP_TOKENS if token != MATCHUP_LEARNED_VS_NEUTRAL
)


class Phase10FinalError(Phase10ContractError):
    """Raised when a final-test evaluation is asked for something illegal."""


def final_cases():
    """The 512 frozen test cases, rebuilt from their identity.

    The only bank entry point in this module, fixed to the sealed test bank:
    no caller can steer it anywhere else, and the harness records the access
    purpose before using what it returns.
    """
    from .phase10_banks import build_phase10_bank

    cases, manifest = build_phase10_bank(FINAL_BANK)
    return cases, manifest


def final_cell_token(arm: str, candidate_id: "str | None", matchup: str) -> str:
    """The identity of one `(arm, candidate, matchup)` final-test cell.

    Mirrors the validation cell token with the test bank version, which is
    what keeps every final-test `match_id`, cache path and stored row
    disjoint from every validation one. The learned arm is additionally
    pinned to the permanently selected candidate: selection is closed and no
    other candidate may reach the sealed bank.
    """
    if arm not in ARMS:
        raise Phase10FinalError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    if matchup not in MATCHUP_TOKENS:
        raise Phase10FinalError(f"unknown matchup token {matchup!r}")
    if arm == ARM_LEARNED:
        if candidate_id != SELECTED_CANDIDATE_ID:
            raise Phase10FinalError(
                f"the final test evaluates the permanently selected "
                f"{SELECTED_CANDIDATE_ID} only, not {candidate_id!r}; selection is "
                "closed"
            )
        selector = candidate_id
    else:
        if candidate_id:
            raise Phase10FinalError(
                "the neutral arm is the fixed baseline and carries no candidate id"
            )
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            raise Phase10FinalError(
                "the direct matchup has no separate neutral arm: it is the comparison"
            )
        selector = NEUTRAL_PROFILE_NAME
    return f"{TEST_BANK_VERSION}|{arm}|{selector}|{matchup}"


def build_final_spec(
    case,
    game_index: int,
    matchup: str,
    *,
    arm: str,
    candidate_id: "str | None",
    own_ref,
    opponent_ref,
    rules=None,
) -> MatchSpec:
    """The completely determined specification of one final-test game.

    `root_seed` is the frozen Agent 1 match seed of the *test* case, so the
    game's identity descends from the sealed bank rather than from anything
    Agent 7 chose; `setup_bank_version` carries the final cell token, so
    identity is candidate- and arm-specific exactly as on the validation
    bank.
    """
    if game_index not in CASE_GAME_INDICES:
        raise Phase10FinalError(f"unknown game index {game_index!r}")
    if case.bank_version != TEST_BANK_VERSION:
        raise Phase10FinalError(
            f"{case.case_id}: final-test specs are built over {TEST_BANK_VERSION} "
            f"cases only, got bank {case.bank_version!r}"
        )
    own_color = CASE_GAME_COLOR[game_index]
    return MatchSpec(
        candidate=own_ref,
        opponent=opponent_ref,
        setup_pair_id=int(case.case_index),
        candidate_color=RED if own_color == "red" else BLUE,
        replicate=game_index,
        root_seed=case_match_seed(case.case_id, game_index, matchup),
        suite_version=PHASE10_FINAL_VERSION,
        setup_bank_version=final_cell_token(arm, candidate_id, matchup),
        rules=EVALUATION_RULES if rules is None else rules,
    )


def play_final_game(
    case,
    row: dict,
    matchup: str,
    *,
    arm: str,
    candidate_id: "str | None",
    own_ref,
    opponent_ref,
    own_policy,
    opponent_policy,
    record_actions: bool = True,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
):
    """Play one final-test game and return `(spec, result)`.

    The opponent plays on the frozen arm-independent
    `case_match_seed(case_id, game_index, matchup)` through
    :class:`FrozenSeedPolicy`, exactly as on the validation bank. Actions
    are recorded by default so the final replay/safety audit can re-apply
    every move through a fresh engine rather than sampling.
    """
    game_index = int(row["game_index"])
    spec = build_final_spec(
        case,
        game_index,
        matchup,
        arm=arm,
        candidate_id=candidate_id,
        own_ref=own_ref,
        opponent_ref=opponent_ref,
    )
    policies = {own_ref.token: own_policy}
    if opponent_ref.token != own_ref.token:
        if opponent_policy is None:
            raise Phase10ValidationError(
                f"{matchup}: an external opponent needs a policy object"
            )
        policies[opponent_ref.token] = FrozenSeedPolicy(
            opponent_policy, case_match_seed(case.case_id, game_index, matchup)
        )
    result = play_match(
        spec,
        bank=single_game_bank(spec, row["red_setup"], row["blue_setup"]),
        policies=policies,
        record_actions=record_actions,
        on_policy_error=on_policy_error,
    )
    return spec, result


__all__ = [
    "FINAL_BANK",
    "FINAL_NEUTRAL_ARM_MATCHUPS",
    "PHASE10_FINAL_VERSION",
    "Phase10FinalError",
    "SELECTED_CANDIDATE_ID",
    "build_final_spec",
    "final_cases",
    "final_cell_token",
    "play_final_game",
]
