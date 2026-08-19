"""Optional Phase 10B: the two-arm paired evaluation.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 15, 17, 20 and 21.

What differs between the two arms
---------------------------------
Exactly one thing: the move policy. Phase 10 varied the *setup source* and
held the checkpoint fixed; Phase 10B does the reverse. On a given case both
arms read the same frozen selector seeds, draw the same own-side arrangements
through the same frozen P10-D selector, face the same held-out opponent
arrangement, and give a rule-based opponent the same match seed. The delta a
gate reads is therefore attributable to the checkpoint and to nothing else.

The two comparison shapes
-------------------------
`direct_p10d` and `neutral_rollback` are head-to-head: the Phase 10B
checkpoint plays the accepted Phase 9 checkpoint, both sides drawing from the
same setup source, and the measured quantity is the candidate's own expected
win rate. The five externally-opposed matchups are two-cell: each arm plays
the same external opponent on the same cases, and the measured quantity is the
paired difference.

The Phase 9 arm is a constant
-----------------------------
The accepted Phase 9 checkpoint never changes during Phase 10B, so its cells
on a given bank are computed once and reused by every scheduled validation
pass. That is a caching decision about identical logical games, not a
statistical shortcut: the rows are byte-identical to recomputing them.
"""

from __future__ import annotations

import dataclasses

from ..engine.constants import BLUE, EVALUATION_RULES, RED
from ..evaluation.policy import Policy, PolicyResult, derive_decision_seed
from ..setups.identity import content_fingerprint
from ..training.phase10b_contract import (
    HEAD_TO_HEAD_MATCHUPS,
    MATCHUP_BASIC,
    MATCHUP_DIRECT,
    MATCHUP_NEUTRAL,
    MATCHUP_PHASE8,
    MATCHUP_RANDOM,
    MATCHUP_STRATEGIC,
    MATCHUP_TACTICAL,
    MATCHUP_TOKENS,
    PAIRED_DELTA_MATCHUPS,
    Phase10BContractError,
)
from .match_runner import ON_POLICY_ERROR_RAISE, play_match
from .match_spec import MatchSpec
from .phase10b_banks import CASE_GAME_COLOR, CASE_GAME_INDICES, COLORS
from .setup_bank import SetupBank, SetupPair

PHASE10B_EVAL_VERSION = "phase10b_eval_v1"

#: The two arms. `phase9` is the fixed accepted baseline, never a candidate.
ARM_CANDIDATE = "phase10b"
ARM_BASELINE = "phase9"
ARMS = (ARM_CANDIDATE, ARM_BASELINE)

#: The setup source each matchup puts under the evaluated arm's own seat.
MATCHUP_SETUP_SOURCE = {
    MATCHUP_DIRECT: "p10d",
    MATCHUP_NEUTRAL: "neutral_v1",
    MATCHUP_STRATEGIC: "p10d",
    MATCHUP_TACTICAL: "p10d",
    MATCHUP_PHASE8: "p10d",
    MATCHUP_RANDOM: "p10d",
    MATCHUP_BASIC: "p10d",
}

#: The catalogued Phase 4 opponent of each externally-opposed matchup.
EXTERNAL_OPPONENT_POLICY_IDS = {
    MATCHUP_STRATEGIC: "strategic_rule_based",
    MATCHUP_TACTICAL: "tactical_rule_based",
    MATCHUP_RANDOM: "random_legal",
    MATCHUP_BASIC: "basic_heuristic",
}

#: The one externally-opposed matchup whose opponent is a checkpoint.
NEURAL_OPPONENT_MATCHUP = MATCHUP_PHASE8
PHASE8_ANCHOR_CANDIDATE_ID = "c1_warmstart"

#: The evaluation identities the two arms play under.
CANDIDATE_MOVE_POLICY_ID = "phase10b_eval_move_v1"
BASELINE_MOVE_POLICY_ID = "phase10b_eval_phase9_move_v1"

EVAL_DTYPE = "float32"

_RESULT_SCORES = {"win": 1.0, "draw": 0.5, "loss": 0.0}


class Phase10BEvalError(Phase10BContractError):
    """Raised when a Phase 10B evaluation is asked for something illegal."""


# ---------------------------------------------------------------------------
# The frozen-seed opponent wrapper
# ---------------------------------------------------------------------------


class FrozenSeedPolicy(Policy):
    """`inner`, playing on the frozen Phase 10B match seed.

    The wrapper exists for one property: a rule-based opponent's randomness is
    a pure function of `(bank, case, game index, matchup)` and is therefore
    identical in the Phase 10B arm and the Phase 9 arm. It replaces only the
    two seed fields of the request and delegates every decision, so the
    wrapped policy's rule is untouched.
    """

    def __init__(self, inner: Policy, policy_seed: int):
        self._inner = inner
        self._policy_seed = int(policy_seed)
        self.policy_id = inner.policy_id
        self.policy_version = inner.policy_version
        self.requirements = inner.requirements
        self.stochastic = inner.stochastic
        self.description = inner.description

    @property
    def frozen_policy_seed(self) -> int:
        return self._policy_seed

    @property
    def inner(self) -> Policy:
        return self._inner

    def decide(self, request) -> PolicyResult:
        reseeded = dataclasses.replace(
            request,
            policy_seed=self._policy_seed,
            decision_seed=derive_decision_seed(self._policy_seed, request.ply),
        )
        result = self._inner.decide(reseeded)
        return PolicyResult(
            selected_action_id=result.selected_action_id,
            policy=request.policy,
            decision_seed=request.decision_seed,
            diagnostics=dict(result.diagnostics)
            | {
                "phase10b_frozen_policy_seed": self._policy_seed,
                "phase10b_frozen_decision_seed": reseeded.decision_seed,
            },
        )

    def describe(self) -> dict:
        described = self._inner.describe()
        described["phase10b_frozen_policy_seed"] = self._policy_seed
        described["phase10b_seed_source"] = "case_match_seed(bank, case, game, matchup)"
        return described


# ---------------------------------------------------------------------------
# Own-side setups
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OwnSideDraw:
    """One seat's own-side arrangement for one `(case, colour)`."""

    source: str
    color: str
    selector_seed: int
    canonical: tuple
    base_setup_id: str
    family_id: str
    branch: "str | None"
    final_setup_fingerprint: str

    def oriented(self, player: int) -> tuple:
        from ..setups.identity import orient_setup

        return orient_setup(self.canonical, player)


def production_selector(*, index=None, scorer=None):
    """The frozen P10-D selector, as the evaluation's own-side source.

    The evaluation addresses a draw by `(case, colour)` rather than by a
    rollout game id, so it holds the accepted
    :class:`~stratego.training.phase10_selector.LearnedSetupSource` directly
    instead of the training adapter around it. Same selector, same utility,
    same scaler, same temperature, same mixture.
    """
    from ..training.phase10b_setup_source import Phase10BSetupSource

    return Phase10BSetupSource.build(index=index, scorer=scorer).source


def neutral_draw(case, color: str) -> OwnSideDraw:
    """The `neutral_v1` own-side draw, straight through the accepted sampler.

    Rebuilt live and then required to match the fingerprint the bank froze, so
    a moved sampler is caught here rather than quietly changing the arrangement
    every rollback measurement rests on.
    """
    from ..training.phase10_selector import neutral_baseline_draw

    seed = int(case.selector_seeds[color])
    sampled = neutral_baseline_draw(case.split, seed)
    fingerprint = content_fingerprint(sampled.canonical)
    frozen = str(case.neutral_provenance[color]["final_setup_fingerprint"])
    if fingerprint != frozen:
        raise Phase10BEvalError(
            f"{case.case_id}/{color}: the live neutral_v1 draw fingerprints "
            f"{fingerprint} but the frozen case records {frozen}"
        )
    return OwnSideDraw(
        source="neutral_v1",
        color=color,
        selector_seed=seed,
        canonical=tuple(sampled.canonical),
        base_setup_id=sampled.base_setup_id,
        family_id=sampled.family_id,
        branch=None,
        final_setup_fingerprint=fingerprint,
    )


def p10d_draw(source, case, color: str) -> OwnSideDraw:
    """The frozen P10-D own-side draw, through the accepted production selector."""
    from ..training.phase10_selector import SelectorRequest

    seed = int(case.selector_seeds[color])
    draw = source.draw(
        SelectorRequest(split=case.split, color=color, selector_seed=seed)
    )
    return OwnSideDraw(
        source="p10d",
        color=color,
        selector_seed=seed,
        canonical=tuple(draw.setup.canonical),
        base_setup_id=draw.base_setup_id,
        family_id=draw.family_id,
        branch=draw.branch,
        final_setup_fingerprint=content_fingerprint(draw.setup.canonical),
    )


def own_side_draws(source, case, matchup: str) -> dict:
    """Both colours' own-side draws for one matchup, by colour."""
    if matchup not in MATCHUP_TOKENS:
        raise Phase10BEvalError(f"unknown matchup token {matchup!r}")
    if MATCHUP_SETUP_SOURCE[matchup] == "neutral_v1":
        return {color: neutral_draw(case, color) for color in COLORS}
    return {color: p10d_draw(source, case, color) for color in COLORS}


# ---------------------------------------------------------------------------
# Game identity
# ---------------------------------------------------------------------------


def cell_token(bank_version: str, arm: str, matchup: str) -> str:
    """The identity of one `(arm, matchup)` evaluation cell.

    Carried in `MatchSpec.setup_bank_version`, which is part of `match_id`.
    That is what makes a game — and therefore its cache path, its replay digest
    and its stored row — arm-specific without touching the frozen opponent
    seed.
    """
    if arm not in ARMS:
        raise Phase10BEvalError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    if matchup not in MATCHUP_TOKENS:
        raise Phase10BEvalError(f"unknown matchup token {matchup!r}")
    if arm == ARM_BASELINE and matchup in HEAD_TO_HEAD_MATCHUPS:
        raise Phase10BEvalError(
            f"{matchup} is head-to-head: it has no separate Phase 9 arm, it *is* "
            "the comparison"
        )
    return f"{bank_version}|{arm}|{matchup}"


def _player_of(color: str) -> int:
    if color == "red":
        return RED
    if color == "blue":
        return BLUE
    raise Phase10BEvalError(f"unknown colour {color!r}")


def _other_player(player: int) -> int:
    return BLUE if player == RED else RED


def game_setups(case, matchup: str, own: dict):
    """`(red_setup, blue_setup, opposing draw)` for both games of one case.

    Game 0 seats the evaluated arm as Red, game 1 as Blue. The opposing seat
    holds the frozen held-out opponent arrangement, except in a head-to-head
    matchup where it holds the *other colour's* own-side draw from the same
    source — so both checkpoints face the same setup distribution.
    """
    rows = []
    for game_index in CASE_GAME_INDICES:
        own_color = CASE_GAME_COLOR[game_index]
        own_player = _player_of(own_color)
        other_player = _other_player(own_player)
        own_setup = own[own_color].oriented(own_player)
        if matchup in HEAD_TO_HEAD_MATCHUPS:
            other_color = CASE_GAME_COLOR[1 - game_index]
            opposing = own[other_color]
            other_setup = opposing.oriented(other_player)
        else:
            opposing = None
            other_setup = case.oriented_opponent(other_player)
        red_setup, blue_setup = (
            (own_setup, other_setup)
            if own_player == RED
            else (other_setup, own_setup)
        )
        rows.append(
            {
                "game_index": game_index,
                "own_color": own_color,
                "red_setup": tuple(red_setup),
                "blue_setup": tuple(blue_setup),
                "opposing_draw": opposing,
            }
        )
    return tuple(rows)


def build_spec(case, game_index: int, matchup: str, *, arm: str, own_ref, opponent_ref,
               rules=EVALUATION_RULES) -> MatchSpec:
    """The completely determined specification of one Phase 10B evaluation game."""
    if game_index not in CASE_GAME_INDICES:
        raise Phase10BEvalError(f"unknown game index {game_index!r}")
    own_color = CASE_GAME_COLOR[game_index]
    return MatchSpec(
        candidate=own_ref,
        opponent=opponent_ref,
        setup_pair_id=int(case.case_index),
        candidate_color=_player_of(own_color),
        replicate=game_index,
        root_seed=int(case.match_seeds[matchup][game_index]),
        suite_version=PHASE10B_EVAL_VERSION,
        setup_bank_version=cell_token(case.bank_version, arm, matchup),
        rules=rules,
    )


def single_game_bank(spec: MatchSpec, red_setup, blue_setup) -> SetupBank:
    """A one-pair bank holding exactly this game's position."""
    pair = SetupPair(
        setup_pair_id=spec.setup_pair_id,
        red_setup=tuple(red_setup),
        blue_setup=tuple(blue_setup),
        generation_seed=spec.root_seed,
        bank_version=spec.setup_bank_version,
        generation_family=PHASE10B_EVAL_VERSION,
    )
    return SetupBank(
        bank_version=spec.setup_bank_version,
        root_seed=spec.root_seed,
        generation_family=PHASE10B_EVAL_VERSION,
        pairs=(pair,),
    )


def play_cell_game(
    case,
    row: dict,
    matchup: str,
    *,
    arm: str,
    own_ref,
    opponent_ref,
    own_policy: Policy,
    opponent_policy: "Policy | None",
    rules=EVALUATION_RULES,
    record_actions: bool = False,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
):
    """Play one Phase 10B evaluation game and return `(spec, result)`."""
    game_index = int(row["game_index"])
    spec = build_spec(
        case, game_index, matchup, arm=arm, own_ref=own_ref, opponent_ref=opponent_ref,
        rules=rules,
    )
    policies = {own_ref.token: own_policy}
    if opponent_ref.token != own_ref.token:
        if opponent_policy is None:
            raise Phase10BEvalError(
                f"{matchup}: the opposing seat needs a policy object"
            )
        # Every opposing seat is wrapped, exactly as the accepted Phase 10
        # harness does: a greedy neural opponent consumes no randomness, so the
        # wrapper is a no-op there and the seeding rule stays one rule.
        seed = int(case.match_seeds[matchup][game_index])
        policies[opponent_ref.token] = FrozenSeedPolicy(opponent_policy, seed)
    result = play_match(
        spec,
        bank=single_game_bank(spec, row["red_setup"], row["blue_setup"]),
        policies=policies,
        record_actions=record_actions,
        on_policy_error=on_policy_error,
    )
    return spec, result


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------


def game_score(result) -> float:
    """The evaluated arm's score, from the accepted result label."""
    try:
        return _RESULT_SCORES[result.candidate_result]
    except KeyError:
        raise Phase10BEvalError(
            f"{result.match_id}: candidate_result {result.candidate_result!r} has "
            "no Phase 10B score; an errored game is a correctness failure, not a "
            "loss"
        ) from None


def case_scores(rows) -> dict:
    """`{case_id: (game 0 score, game 1 score)}` for one cell's rows."""
    by_case: dict = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[int(row["game_index"])] = row["score"]
    scores: dict = {}
    for case_id, games in by_case.items():
        missing = [index for index in CASE_GAME_INDICES if index not in games]
        if missing:
            raise Phase10BEvalError(f"{case_id}: missing game(s) {missing}")
        if any(games[index] is None for index in CASE_GAME_INDICES):
            raise Phase10BEvalError(f"{case_id}: an errored game has no score")
        scores[case_id] = tuple(float(games[index]) for index in CASE_GAME_INDICES)
    return scores


def case_means(rows) -> dict:
    """`{case_id: mean of the two colour-swapped games}` — the bootstrap unit."""
    return {
        case_id: (pair[0] + pair[1]) / 2.0
        for case_id, pair in case_scores(rows).items()
    }


def expected_win_rate(rows) -> float:
    means = case_means(rows)
    if not means:
        raise Phase10BEvalError("no cases to score")
    return sum(means.values()) / len(means)


def color_split(rows) -> dict:
    """Per-seat expected win rate, for the Gate E per-colour guards."""
    totals = {"red": [0.0, 0], "blue": [0.0, 0]}
    for row in rows:
        if row["score"] is None:
            raise Phase10BEvalError(f"{row['match_id']}: an errored game has no score")
        bucket = totals[row["own_color"]]
        bucket[0] += float(row["score"])
        bucket[1] += 1
    return {
        color: (total / count if count else None)
        for color, (total, count) in totals.items()
    }


def counts_from_rows(rows) -> dict:
    outcomes: dict = {}
    for row in rows:
        outcomes[row["candidate_result"]] = outcomes.get(row["candidate_result"], 0) + 1
    return outcomes


def safety_counters(rows) -> dict:
    """Every correctness counter the gates read, from primitive stored rows."""
    return {
        "games": len(rows),
        "errored_games": sum(1 for row in rows if row["candidate_result"] == "error"),
        "policy_errors": sum(1 for row in rows if row.get("policy_error")),
        "missing_scores": sum(1 for row in rows if row["score"] is None),
        "distinct_replay_digests": len({row["replay_digest"] for row in rows}),
    }


def cells_for_bank(bank_version: str) -> tuple:
    """Every `(arm, matchup)` cell one Phase 10B bank has to produce."""
    cells = [(ARM_CANDIDATE, matchup) for matchup in MATCHUP_TOKENS]
    cells.extend((ARM_BASELINE, matchup) for matchup in PAIRED_DELTA_MATCHUPS)
    return tuple(cells)


def baseline_cells() -> tuple:
    """The cells whose rows depend only on the accepted, unchanging Phase 9 model."""
    return tuple((ARM_BASELINE, matchup) for matchup in PAIRED_DELTA_MATCHUPS)


__all__ = [
    "ARMS",
    "ARM_BASELINE",
    "ARM_CANDIDATE",
    "BASELINE_MOVE_POLICY_ID",
    "CANDIDATE_MOVE_POLICY_ID",
    "EVAL_DTYPE",
    "EXTERNAL_OPPONENT_POLICY_IDS",
    "FrozenSeedPolicy",
    "MATCHUP_SETUP_SOURCE",
    "NEURAL_OPPONENT_MATCHUP",
    "OwnSideDraw",
    "PHASE10B_EVAL_VERSION",
    "PHASE8_ANCHOR_CANDIDATE_ID",
    "Phase10BEvalError",
    "baseline_cells",
    "build_spec",
    "case_means",
    "case_scores",
    "cell_token",
    "cells_for_bank",
    "color_split",
    "counts_from_rows",
    "expected_win_rate",
    "game_score",
    "game_setups",
    "neutral_draw",
    "own_side_draws",
    "p10d_draw",
    "play_cell_game",
    "production_selector",
    "safety_counters",
    "single_game_bank",
]
