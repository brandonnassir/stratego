"""Phase 10 Agent 5: the bounded validation evaluation.

Specification sources:

- `05_AGENT_5_BOUNDED_VALIDATION_SELECTION.md` ("Validation protocol",
  "Primitive metrics", "Eligibility", "Selection score")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Validation matchups",
  "Candidate-selection score", "Statistics")

What one validation game is
---------------------------
A Phase 10 case fixes everything except the setup under test. This module
turns a case into the two colour-paired games of one `(arm, candidate,
matchup)` cell:

```text
game 0      the evaluated selector plays Red
game 1      the evaluated selector plays Blue
own side    learned arm  -> LearnedSetupSource.draw(split, colour, case seed)
            neutral arm  -> neutral_baseline_draw(split, case seed)
other side  learned_vs_neutral -> the neutral_v1 draw of the *other* colour
            every other matchup -> the case's frozen held-out opponent setup
```

`learned_vs_neutral` is the one matchup with no external opponent: it has
two sides and two selectors, so the held-out opponent setup has no seat at
the table and the neutral side plays the case's frozen `neutral_v1` draw
for the colour it was dealt. Every other matchup seats the frozen held-out
opponent setup opposite the selector under test, identically in both arms.

Two seeds, and why the opponent's is not the runner's
-----------------------------------------------------
Agent 1 froze `case_match_seed(case_id, game_index, matchup_token)` as "one
seed per (case, game index, matchup), **independent of arm and candidate**,
so a rule-based opponent draws identical randomness in both arms". The
accepted runner derives a side's seed from `match_id`, and `match_id` here
must stay candidate-specific — Agent 5 is required to keep cache and game
identities candidate-specific. The two requirements are about different
things and are met separately:

```text
game identity      candidate-specific, through MatchSpec.setup_bank_version
opponent seed      the frozen case_match_seed, through FrozenSeedPolicy
```

:class:`FrozenSeedPolicy` is a thin delegating wrapper that replaces the
match-level policy seed of the side it wraps. It changes no decision rule
and keeps the wrapped policy's identity, so the recorded row still names
the accepted baseline. The selector-under-test side never needs it: in all
six matchups that side is the accepted Phase 9 checkpoint playing greedy,
which reads no seed at all.

What this module does not do
----------------------------
It fits nothing, refits nothing, defines no candidate, and cannot reach
`phase10_test_bank_v1`: :func:`validation_cases` is the only bank entry
point here and it builds the validation bank alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass

from ..engine.constants import BLUE, EVALUATION_RULES, RED, RulesConfig
from ..setups.identity import content_fingerprint
from ..training.phase10_contract import (
    EVAL_MOVE_BEHAVIOR,
    MATCHUP_BASIC,
    MATCHUP_LEARNED_VS_NEUTRAL,
    MATCHUP_PHASE8_ANCHOR,
    MATCHUP_RANDOM,
    MATCHUP_STRATEGIC,
    MATCHUP_TACTICAL,
    MATCHUP_TOKENS,
    NEUTRAL_PROFILE_NAME,
    Phase10ContractError,
    VALIDATION_BANK_VERSION,
)
from ..training.phase10_seed import (
    CASE_GAME_COLOR,
    CASE_GAME_INDICES,
    COLORS,
    case_match_seed,
)
from .match_runner import (
    ERROR_CONTRACT_VIOLATION,
    ERROR_ENGINE_REJECTED,
    ERROR_ILLEGAL_ACTION,
    ERROR_POLICY_EXCEPTION,
    ON_POLICY_ERROR_RAISE,
    play_match,
)
from .match_spec import MatchSpec
from .policy import Policy, PolicyResult, derive_decision_seed
from .setup_bank import SetupBank, SetupPair

#: The two arms of every paired comparison. `neutral` is the fixed baseline,
#: never a seventh candidate.
ARM_LEARNED = "learned"
ARM_NEUTRAL = "neutral"
ARMS = (ARM_LEARNED, ARM_NEUTRAL)

#: The matchups the neutral baseline arm is also evaluated on. The direct
#: matchup is excluded because it *is* the comparison: it has no second arm.
NEUTRAL_ARM_MATCHUPS = tuple(
    token for token in MATCHUP_TOKENS if token != MATCHUP_LEARNED_VS_NEUTRAL
)

#: The catalogued opponent of each externally-opposed matchup.
EXTERNAL_OPPONENT_POLICY_IDS = {
    MATCHUP_STRATEGIC: "strategic_rule_based",
    MATCHUP_TACTICAL: "tactical_rule_based",
    MATCHUP_RANDOM: "random_legal",
    MATCHUP_BASIC: "basic_heuristic",
}

#: The one externally-opposed matchup whose opponent is a checkpoint rather
#: than a catalogued rule-based policy, which is why it is absent above.
NEURAL_OPPONENT_MATCHUP = MATCHUP_PHASE8_ANCHOR

#: The Phase 8 anchor's accepted evaluation identity, unchanged from Phase 9.
PHASE8_ANCHOR_CANDIDATE_ID = "c1_warmstart"

#: The evaluation identity the accepted Phase 9 checkpoint plays under here.
#: It names the same weights Agent 2 collected the corpus with; the id is
#: distinct only because the role is.
PHASE10_EVAL_MOVE_POLICY_ID = "phase10_eval_move_v1"

#: Every move decision in every matchup, from the frozen contract.
EVAL_DTYPE = EVAL_MOVE_BEHAVIOR["dtype"]

VALIDATION_BANK = "validation"

#: Version tag of this module's game-identity scheme, so a stored row names
#: the scheme that produced it.
PHASE10_VALIDATION_VERSION = "phase10_validation_eval_v1"


class Phase10ValidationError(Phase10ContractError):
    """Raised when a validation evaluation is asked for something illegal."""


# ---------------------------------------------------------------------------
# The frozen-seed opponent wrapper
# ---------------------------------------------------------------------------


class FrozenSeedPolicy(Policy):
    """`inner`, playing on the frozen Phase 10 match seed.

    The wrapper exists for one frozen property: the opponent's randomness is
    a pure function of `(case, game index, matchup, ply)` and therefore
    identical in the learned arm and the neutral arm, and identical across
    all six candidates. It replaces only the two seed fields of the request
    and delegates every decision, so the wrapped policy's rule is untouched.

    The returned result is re-stamped with the *outer* request's decision
    seed because that is what the accepted validator compares against; the
    seed the decision was actually taken on is recorded in the diagnostics
    rather than hidden.
    """

    def __init__(self, inner: Policy, policy_seed: int):
        self._inner = inner
        self._policy_seed = int(policy_seed)
        # Instance attributes shadow the class-level identity, so the wrapper
        # answers to the wrapped policy's token: a stored row names the
        # accepted baseline, not a Phase 10 variant of it.
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
                "phase10_frozen_policy_seed": self._policy_seed,
                "phase10_frozen_decision_seed": reseeded.decision_seed,
            },
        )

    def describe(self) -> dict:
        described = self._inner.describe()
        described["phase10_frozen_policy_seed"] = self._policy_seed
        described["phase10_seed_source"] = "case_match_seed(case_id, game_index, matchup)"
        return described


# ---------------------------------------------------------------------------
# Bank access — validation only
# ---------------------------------------------------------------------------


def validation_cases():
    """The 128 frozen validation cases, rebuilt from their identity.

    The only bank entry point in this module. It takes no bank argument, so
    no caller can steer it at `phase10_test_bank_v1`.
    """
    from .phase10_banks import build_phase10_bank

    cases, manifest = build_phase10_bank(VALIDATION_BANK)
    return cases, manifest


# ---------------------------------------------------------------------------
# Own-side setups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnSideDraw:
    """One arm's own-side setup for one `(case, colour)`."""

    arm: str
    candidate_id: "str | None"
    color: str
    selector_seed: int
    canonical: "tuple[int, ...]"
    base_setup_id: str
    family_id: str
    branch: "str | None"
    final_setup_fingerprint: str

    def oriented(self, player: int) -> "tuple[int, ...]":
        from ..setups.identity import orient_setup

        return orient_setup(self.canonical, player)


def neutral_own_side(case, color: str) -> OwnSideDraw:
    """The baseline arm's own-side draw: the accepted Phase 7 sampler.

    Rebuilt live rather than read out of the case, then required to match
    the fingerprint Agent 1 froze — so a moved sampler is caught here rather
    than quietly changing the baseline every delta is measured against.
    """
    from ..training.phase10_selector import neutral_baseline_draw

    seed = int(case.selector_seeds[color])
    sampled = neutral_baseline_draw(case.split, seed)
    fingerprint = content_fingerprint(sampled.canonical)
    frozen = str(case.neutral_provenance[color]["final_setup_fingerprint"])
    if fingerprint != frozen:
        raise Phase10ValidationError(
            f"{case.case_id}/{color}: the live neutral_v1 draw fingerprints "
            f"{fingerprint} but the frozen case records {frozen}"
        )
    return OwnSideDraw(
        arm=ARM_NEUTRAL,
        candidate_id=None,
        color=color,
        selector_seed=seed,
        canonical=tuple(sampled.canonical),
        base_setup_id=sampled.base_setup_id,
        family_id=sampled.family_id,
        branch=None,
        final_setup_fingerprint=fingerprint,
    )


def learned_own_side(source, case, color: str) -> OwnSideDraw:
    """The candidate arm's own-side draw, through the production selector."""
    from ..training.phase10_selector import SelectorRequest

    seed = int(case.selector_seeds[color])
    draw = source.draw(
        SelectorRequest(split=case.split, color=color, selector_seed=seed)
    )
    return OwnSideDraw(
        arm=ARM_LEARNED,
        candidate_id=source.candidate.candidate_id,
        color=color,
        selector_seed=seed,
        canonical=tuple(draw.setup.canonical),
        base_setup_id=draw.base_setup_id,
        family_id=draw.family_id,
        branch=draw.branch,
        final_setup_fingerprint=content_fingerprint(draw.setup.canonical),
    )


# ---------------------------------------------------------------------------
# Game identity
# ---------------------------------------------------------------------------


def cell_token(arm: str, candidate_id: "str | None", matchup: str) -> str:
    """The identity of one `(arm, candidate, matchup)` evaluation cell.

    Carried in `MatchSpec.setup_bank_version`, which is part of `match_id`.
    That is what makes a game — and therefore its cache path, its replay
    digest and its stored row — candidate-specific, as Agent 5 requires,
    without touching the frozen opponent seed.
    """
    if arm not in ARMS:
        raise Phase10ValidationError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    if matchup not in MATCHUP_TOKENS:
        raise Phase10ValidationError(f"unknown matchup token {matchup!r}")
    if arm == ARM_LEARNED and not candidate_id:
        raise Phase10ValidationError("the learned arm needs a candidate id")
    if arm == ARM_NEUTRAL:
        if candidate_id:
            raise Phase10ValidationError(
                "the neutral arm is the fixed baseline and carries no candidate id"
            )
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            raise Phase10ValidationError(
                "the direct matchup has no separate neutral arm: it is the comparison"
            )
    selector = candidate_id if arm == ARM_LEARNED else NEUTRAL_PROFILE_NAME
    return f"{VALIDATION_BANK_VERSION}|{arm}|{selector}|{matchup}"


def game_setups(case, matchup: str, own: "dict[str, OwnSideDraw]"):
    """`(red_setup, blue_setup, opposing_side)` for both games of one case.

    `own` maps colour to the evaluated arm's own-side draw. Game 0 seats the
    evaluated selector as Red, game 1 as Blue; the opposing seat holds the
    frozen held-out opponent setup, except in the direct matchup where it
    holds the neutral draw of the colour that seat was dealt.
    """
    rows = []
    for game_index in CASE_GAME_INDICES:
        own_color = CASE_GAME_COLOR[game_index]
        own_player = _player_of(own_color)
        other_player = _other_player(own_player)
        own_setup = own[own_color].oriented(own_player)
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            other_color = CASE_GAME_COLOR[1 - game_index]
            opposing = neutral_own_side(case, other_color)
            other_setup = opposing.oriented(other_player)
        else:
            opposing = None
            other_setup = case.oriented_opponent(other_player)
        red_setup, blue_setup = (
            (own_setup, other_setup) if own_player == _player_of("red") else (other_setup, own_setup)
        )
        rows.append(
            {
                "game_index": game_index,
                "own_color": own_color,
                "red_setup": tuple(red_setup),
                "blue_setup": tuple(blue_setup),
                "opposing_neutral": opposing,
            }
        )
    return tuple(rows)


def _player_of(color: str) -> int:
    if color == "red":
        return RED
    if color == "blue":
        return BLUE
    raise Phase10ValidationError(f"unknown colour {color!r}; expected one of {list(COLORS)}")


def _other_player(player: int) -> int:
    return BLUE if player == RED else RED


def build_spec(
    case,
    game_index: int,
    matchup: str,
    *,
    arm: str,
    candidate_id: "str | None",
    own_ref,
    opponent_ref,
    rules: RulesConfig = EVALUATION_RULES,
) -> MatchSpec:
    """The completely determined specification of one validation game.

    `root_seed` is the frozen Phase 10 match seed, so the game's identity
    descends from Agent 1's schedule rather than from anything Agent 5
    chose; `setup_bank_version` carries the cell, so identity is also
    candidate-specific.
    """
    if game_index not in CASE_GAME_INDICES:
        raise Phase10ValidationError(f"unknown game index {game_index!r}")
    own_color = CASE_GAME_COLOR[game_index]
    return MatchSpec(
        candidate=own_ref,
        opponent=opponent_ref,
        setup_pair_id=int(case.case_index),
        candidate_color=_player_of(own_color),
        replicate=game_index,
        root_seed=case_match_seed(case.case_id, game_index, matchup),
        suite_version=PHASE10_VALIDATION_VERSION,
        setup_bank_version=cell_token(arm, candidate_id, matchup),
        rules=rules,
    )


def single_game_bank(spec: MatchSpec, red_setup, blue_setup) -> SetupBank:
    """A one-pair bank holding exactly this game's position.

    `play_match` also accepts raw setups, but routing through the accepted
    bank type keeps `MatchResult.setup_bank_version` meaningful and makes
    the cell token part of the stored row.
    """
    pair = SetupPair(
        setup_pair_id=spec.setup_pair_id,
        red_setup=tuple(red_setup),
        blue_setup=tuple(blue_setup),
        generation_seed=spec.root_seed,
        bank_version=spec.setup_bank_version,
        generation_family=PHASE10_VALIDATION_VERSION,
    )
    return SetupBank(
        bank_version=spec.setup_bank_version,
        root_seed=spec.root_seed,
        generation_family=PHASE10_VALIDATION_VERSION,
        pairs=(pair,),
    )


# ---------------------------------------------------------------------------
# Playing one cell
# ---------------------------------------------------------------------------


def play_cell_game(
    case,
    row: dict,
    matchup: str,
    *,
    arm: str,
    candidate_id: "str | None",
    own_ref,
    opponent_ref,
    own_policy: Policy,
    opponent_policy: "Policy | None",
    rules: RulesConfig = EVALUATION_RULES,
    record_actions: bool = False,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
):
    """Play one validation game and return `(spec, result)`.

    The opponent is always wrapped on the frozen match seed. When both sides
    are the same policy reference — the direct matchup, which is the same
    checkpoint on both sides — the runner resolves one object for both
    seats, so no wrapper is applied and no seed is read: greedy neural play
    consumes none.
    """
    game_index = int(row["game_index"])
    spec = build_spec(
        case,
        game_index,
        matchup,
        arm=arm,
        candidate_id=candidate_id,
        own_ref=own_ref,
        opponent_ref=opponent_ref,
        rules=rules,
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


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------

#: The score of one game from the evaluated selector's perspective.
_RESULT_SCORES = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def game_score(result) -> float:
    """The evaluated side's score, from the accepted result label."""
    try:
        return _RESULT_SCORES[result.candidate_result]
    except KeyError:
        raise Phase10ValidationError(
            f"{result.match_id}: candidate_result {result.candidate_result!r} has no "
            "Phase 10 score; an errored game is a correctness failure, not a loss"
        ) from None


def case_game_pairs(rows: "dict[str, dict[int, dict]]", case_ids) -> "tuple[tuple[float, float], ...]":
    """`(game 0 score, game 1 score)` per case, in the frozen case order."""
    pairs = []
    for case_id in case_ids:
        by_index = rows.get(case_id)
        if by_index is None or set(by_index) != set(CASE_GAME_INDICES):
            raise Phase10ValidationError(
                f"{case_id}: expected both colour-paired games, got "
                f"{sorted(by_index or ())}"
            )
        pairs.append(tuple(float(by_index[index]["score"]) for index in CASE_GAME_INDICES))
    return tuple(pairs)


def counts_from_rows(rows) -> dict:
    """W/D/L and EWR over a flat sequence of stored game rows."""
    wins = sum(1 for row in rows if row["score"] == 1.0)
    draws = sum(1 for row in rows if row["score"] == 0.5)
    losses = sum(1 for row in rows if row["score"] == 0.0)
    total = wins + draws + losses
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "ewr": (wins + 0.5 * draws) / total if total else float("nan"),
    }


def color_split(rows) -> dict:
    """W/D/L and EWR of the evaluated side, split by the colour it played."""
    return {
        color: counts_from_rows([row for row in rows if row["own_color"] == color])
        for color in COLORS
    }


def family_split(rows) -> dict:
    """EWR by the case's opponent-setup family, the frozen case family."""
    families = sorted({row["case_family"] for row in rows})
    return {
        family: counts_from_rows([row for row in rows if row["case_family"] == family])
        for family in families
    }


def terminal_reasons(rows) -> dict:
    counts: dict = {}
    for row in rows:
        counts[row["terminal_reason"]] = counts.get(row["terminal_reason"], 0) + 1
    return dict(sorted(counts.items()))


def length_summary(rows) -> dict:
    plies = sorted(int(row["plies"]) for row in rows)
    if not plies:
        return {"games": 0}
    total = len(plies)
    mean = sum(plies) / total
    return {
        "games": total,
        "min": plies[0],
        "max": plies[-1],
        "mean": mean,
        "median": (
            plies[total // 2]
            if total % 2
            else (plies[total // 2 - 1] + plies[total // 2]) / 2.0
        ),
        "total": sum(plies),
    }


def safety_counters(rows) -> dict:
    """Every zero-tolerance counter this evaluation can observe."""
    return {
        "policy_errors": sum(1 for row in rows if row.get("policy_error")),
        "illegal_actions": sum(
            1 for row in rows if row.get("policy_error_category") == ERROR_ILLEGAL_ACTION
        ),
        "engine_rejections": sum(
            1 for row in rows if row.get("policy_error_category") == ERROR_ENGINE_REJECTED
        ),
        "policy_exceptions": sum(
            1 for row in rows if row.get("policy_error_category") == ERROR_POLICY_EXCEPTION
        ),
        "contract_violations": sum(
            1 for row in rows if row.get("policy_error_category") == ERROR_CONTRACT_VIOLATION
        ),
        "non_finite_scores": sum(
            1 for row in rows if not math.isfinite(float(row["score"]))
        ),
        "illegal_setups": sum(1 for row in rows if row.get("setup_invalid")),
        "unscored_games": sum(1 for row in rows if row.get("score") is None),
    }


def rows_digest(rows) -> str:
    """A canonical digest over every game outcome a cell recorded."""
    payload = "\n".join(
        "|".join(
            (
                str(row["match_id"]),
                str(row["case_id"]),
                str(row["game_index"]),
                str(row["own_color"]),
                str(row["score"]),
                str(row["terminal_reason"]),
                str(row["plies"]),
                str(row["replay_digest"]),
            )
        )
        for row in sorted(rows, key=lambda entry: (entry["case_id"], entry["game_index"]))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# The Phase 9 fingerprint landing diagnostic — report-only, never a gate
# ---------------------------------------------------------------------------


def landing_counts(rows, isolation_set) -> dict:
    """How many produced own-side setups land in the Phase 9 held-out set.

    Agent 1's standing obligation, at the granularity Agent 5 owes:
    candidate x arm x matchup x bank, reporting count and rate. It is
    recorded and never read: no retry, score, eligibility check, tie-break
    or acceptance gate consults it. Rejecting such a draw at evaluation time
    would distort the very mixed distribution the diversity contract is
    stated over, which is why Agent 1 forbade it.
    """
    total = len(rows)
    landed = sum(1 for row in rows if row["own_fingerprint"] in isolation_set)
    return {
        "games": total,
        "landings": landed,
        "rate": (landed / total) if total else 0.0,
        "gate": False,
        "use": "report_only",
    }


__all__ = [
    "ARMS",
    "ARM_LEARNED",
    "ARM_NEUTRAL",
    "EXTERNAL_OPPONENT_POLICY_IDS",
    "FrozenSeedPolicy",
    "NEURAL_OPPONENT_MATCHUP",
    "NEUTRAL_ARM_MATCHUPS",
    "OwnSideDraw",
    "EVAL_DTYPE",
    "PHASE10_EVAL_MOVE_POLICY_ID",
    "PHASE10_VALIDATION_VERSION",
    "PHASE8_ANCHOR_CANDIDATE_ID",
    "Phase10ValidationError",
    "build_spec",
    "case_game_pairs",
    "cell_token",
    "color_split",
    "counts_from_rows",
    "family_split",
    "game_score",
    "game_setups",
    "landing_counts",
    "learned_own_side",
    "length_summary",
    "neutral_own_side",
    "play_cell_game",
    "rows_digest",
    "safety_counters",
    "single_game_bank",
    "terminal_reasons",
    "validation_cases",
]
