"""The Phase 12 search-budget ladder: presets, efficiency and the stopping rule.

Specification source: `05_PHASE_12_AGENT_4_BUDGET_SCALING.md` — three
sequential presets rather than a grid, one compact fixed match pack with a
stable opponent mix, the section 4 metric list, and the section 5 stopping
rule.

What this module is for
-----------------------
Agent 3 showed that search at SMALL beats the direct accepted Phase 9
player. The remaining question is not *whether* to search but *how much to
pay for it*, and that question has two halves which this module keeps
apart:

- the ladder itself — which budgets get played, in what order, by which
  arm — lives in :func:`ladder_arms` and :data:`LADDER_PRESET_NAMES`;
- the reading of the results — efficiency, the stopping rule, and the
  choice of a practical operating point — lives in :func:`ladder_analysis`,
  :func:`stopping_rule` and :func:`select_operating_point`, all pure
  functions over :class:`BudgetPoint` values.

The second half is pure on purpose. A stopping rule that is only ever
exercised by a two-hour match run is a rule nobody can check; these take
plain numbers and are tested directly.

Thresholds are engineering judgements, not measurements
-------------------------------------------------------
Every constant below is a stated choice. They are named, gathered here and
echoed into the report so a reader disagreeing with the rule can see
exactly which number to move, rather than reverse-engineering it from a
verdict. The instruction asks for an engineering trend, not a powered
experiment, so these are deliberately coarse.

The gated larger setting
------------------------
`LARGE` (64 worlds, depth 10) exists so the stopping rule has something
concrete to refuse. The instruction allows it only if MEDIUM already
produces meaningful additional strength at acceptable latency, so it is
defined here rather than added to Agent 1's frozen preset table: nothing in
:mod:`.contract` is edited, and `search_preset` still names exactly the
three instructed presets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .contract import (
    PROVIDER_AGENT1C,
    Phase12SearchConfig,
    Phase12SearchError,
    SEARCH_PRESETS,
)
from .matchplay import ARM_DIRECT, MatchArm

#: The identity of this ladder. Any change to the presets played, the arm
#: definitions or the stopping-rule thresholds is a new version.
BUDGET_VERSION = "phase12_budget_ladder_v1"

#: The three instructed presets, cheapest first. Played in this order and
#: reported in it; the instruction is explicit that this is a ladder and
#: not a grid, so nothing here crosses worlds against depth against beta.
LADDER_PRESET_NAMES = ("TINY", "SMALL", "MEDIUM")

#: The larger setting the instruction gates: "64 worlds and depth 10-12".
#: Defined, costed and refused-or-allowed by the stopping rule; never a
#: default. Depth 10 is the cheap end of the instructed range.
PRESET_LARGE = Phase12SearchConfig("LARGE", worlds=64, rollout_depth=10)

#: From Agent 4 onward Agent 1C is *the* production belief provider; the
#: other providers are comparison references and do not appear on the
#: ladder. One provider, three budgets, one variable.
LADDER_PROVIDER = PROVIDER_AGENT1C


# ---------------------------------------------------------------------------
# Engineering thresholds
# ---------------------------------------------------------------------------

#: An EWR difference this agent is willing to read as a real gain. At 64
#: games a per-game score has a standard error near 0.06, so anything
#: smaller is a record of what happened rather than a strength ordering.
#: Deliberately the same margin Agent 3 used, so the two agents' verdicts
#: are spoken in one language.
MEANINGFUL_EWR_GAIN = 0.10

#: The latency a human opponent should not have to wait past on a normal
#: move. A chosen comfort line, not a measured one.
COMFORTABLE_MOVE_SECONDS = 1.0

#: The latency at which human play stops being practical at all.
IMPRACTICAL_MOVE_SECONDS = 5.0

#: "Latency rises much faster than strength": the next preset must not cost
#: more than this multiple of the search seconds per game for each unit of
#: the meaningful gain it buys. Expressed as a ratio so it scales with the
#: ladder rather than with any one preset's absolute cost.
LATENCY_TO_STRENGTH_RATIO = 3.0

#: "The next preset would consume disproportionate compute": a preset
#: costing more than this multiple of the whole ladder played so far is not
#: worth spending on an engineering trend.
DISPROPORTIONATE_COMPUTE_MULTIPLE = 1.0


class Phase12BudgetError(Phase12SearchError):
    """A budget ladder could not be built or read."""


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def ladder_config(preset_name: str) -> Phase12SearchConfig:
    """The search configuration for one rung, instructed or gated."""
    if preset_name == PRESET_LARGE.preset_id:
        return PRESET_LARGE
    if preset_name not in SEARCH_PRESETS:
        known = ", ".join((*sorted(SEARCH_PRESETS), PRESET_LARGE.preset_id))
        raise Phase12BudgetError(f"unknown ladder preset {preset_name!r}; known: {known}")
    return SEARCH_PRESETS[preset_name]


def ladder_arm(preset_name: str) -> MatchArm:
    """The Agent 1C search arm for one rung.

    One arm per budget, all carrying the same provider, so a difference
    between two arms is a difference of budget and of nothing else.
    """
    config = ladder_config(preset_name)
    return MatchArm(
        arm_id=f"search_agent1c_{config.preset_id.lower()}",
        kind="search",
        label=f"search + agent1c @ {config.preset_id}",
        provider_id=LADDER_PROVIDER,
    )


def preset_of_arm(arm_id: str) -> str:
    """The rung an arm id names, inverting :func:`ladder_arm`.

    Callers need the budget behind an arm in three places — building the
    engine, labelling the rung and costing the next one — and deriving it
    by string surgery on the id in each of them is how a rung ends up
    running a budget its label denies.
    """
    for name in (*SEARCH_PRESETS, PRESET_LARGE.preset_id):
        if ladder_arm(name).arm_id == arm_id:
            return name
    raise Phase12BudgetError(f"{arm_id!r} is not a budget ladder arm")


def ladder_arms(preset_names=LADDER_PRESET_NAMES) -> "tuple[MatchArm, ...]":
    """The reference arm followed by one search arm per rung.

    The direct accepted Phase 9 seat is on the ladder because every metric
    the instruction asks for is a *delta*: EWR gain, extra search seconds,
    strength bought per second. Without a zero-search arm on the same
    boards those deltas would be quoted against another match set.
    """
    names = tuple(preset_names)
    if not names:
        raise Phase12BudgetError("a budget ladder needs at least one preset")
    if len(set(names)) != len(names):
        raise Phase12BudgetError(f"duplicate presets on the ladder: {names}")
    return (ARM_DIRECT, *(ladder_arm(name) for name in names))


def relative_cost(config: Phase12SearchConfig, other: Phase12SearchConfig) -> float:
    """`config`'s forward-pass cost as a multiple of `other`'s.

    A static estimate — worlds x candidates x plies-to-leaf — used only to
    say what an unplayed rung *would* cost. Measured seconds always win
    over this number where both exist; a real game deviates because
    duplicate worlds are evaluated once and terminal rollouts stop early.
    """
    def forwards(item: Phase12SearchConfig) -> float:
        return float(item.worlds * item.max_root_candidates * (item.rollout_depth + 1))

    denominator = forwards(other)
    if denominator <= 0:  # pragma: no cover - defensive
        raise Phase12BudgetError("a preset with no forward cost cannot be a baseline")
    return forwards(config) / denominator


# ---------------------------------------------------------------------------
# One measured rung
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetPoint:
    """One preset's measured result, reduced to what the rule needs.

    `search_seconds_per_game` is the player seat's own time, not the game's
    wall clock: the opponent's seconds are the same in every arm and would
    dilute the very quantity the efficiency metric is dividing by.
    """

    preset_id: str
    worlds: int
    rollout_depth: int
    max_root_candidates: int
    games: int
    ewr: float
    move_seconds_median: float
    move_seconds_p95: float
    search_seconds_per_game: float
    forwards_per_move: float
    #: Any run-time defect that makes this rung's number untrustworthy:
    #: a probe failure, an illegal action, a refused decision.
    unstable: bool = False
    instability: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if self.games < 1:
            raise Phase12BudgetError(f"{self.preset_id}: a rung needs at least one game")
        if not 0.0 <= self.ewr <= 1.0:
            raise Phase12BudgetError(f"{self.preset_id}: EWR {self.ewr} outside [0, 1]")
        if self.search_seconds_per_game < 0.0:
            raise Phase12BudgetError(f"{self.preset_id}: negative search seconds")
        # A rung whose label disagrees with its budget would silently
        # mis-project what the next rung costs, which is the one number the
        # stopping rule cannot afford to get wrong.
        known = SEARCH_PRESETS.get(self.preset_id)
        if self.preset_id == PRESET_LARGE.preset_id:
            known = PRESET_LARGE
        if known is not None:
            carried = (self.worlds, self.rollout_depth, self.max_root_candidates)
            expected = (known.worlds, known.rollout_depth, known.max_root_candidates)
            if carried != expected:
                raise Phase12BudgetError(
                    f"{self.preset_id}: rung carries budget {carried} but the preset "
                    f"of that name is {expected}"
                )

    @property
    def config(self) -> Phase12SearchConfig:
        return Phase12SearchConfig(
            preset_id=self.preset_id,
            worlds=self.worlds,
            rollout_depth=self.rollout_depth,
            max_root_candidates=self.max_root_candidates,
        )

    def describe(self) -> dict:
        return {
            "preset_id": self.preset_id,
            "worlds": self.worlds,
            "rollout_depth": self.rollout_depth,
            "max_root_candidates": self.max_root_candidates,
            "games": self.games,
            "ewr": self.ewr,
            "move_seconds_median": self.move_seconds_median,
            "move_seconds_p95": self.move_seconds_p95,
            "search_seconds_per_game": self.search_seconds_per_game,
            "forwards_per_move": self.forwards_per_move,
            "unstable": self.unstable,
            "instability": list(self.instability),
        }


# ---------------------------------------------------------------------------
# Reading the ladder
# ---------------------------------------------------------------------------


def ladder_analysis(
    points: "list[BudgetPoint]",
    *,
    reference_ewr: "float | None" = None,
    reference_seconds_per_game: float = 0.0,
) -> "list[dict]":
    """Per-rung deltas and efficiency, in ladder order.

    Two efficiencies are reported for every rung because they answer
    different questions. `ewr_gain_per_search_second_vs_reference` is what
    the whole search buys over playing directly — the number that decides
    whether to search at all. `ewr_gain_per_extra_search_second` is what
    *this rung* buys over the rung below it — the number that decides
    whether to climb, and the one the stopping rule reads.
    """
    rows: list = []
    previous: "BudgetPoint | None" = None
    for point in points:
        gain_vs_reference = (
            None if reference_ewr is None else point.ewr - reference_ewr
        )
        extra_vs_reference = point.search_seconds_per_game - reference_seconds_per_game
        delta_ewr = None if previous is None else point.ewr - previous.ewr
        extra_seconds = (
            None
            if previous is None
            else point.search_seconds_per_game - previous.search_seconds_per_game
        )
        rows.append(
            {
                **point.describe(),
                "previous_preset_id": None if previous is None else previous.preset_id,
                "delta_ewr_from_previous": delta_ewr,
                "extra_search_seconds_per_game": extra_seconds,
                "ewr_gain_per_extra_search_second": (
                    delta_ewr / extra_seconds
                    if delta_ewr is not None and extra_seconds
                    else None
                ),
                "ewr_gain_vs_reference": gain_vs_reference,
                "extra_search_seconds_vs_reference": extra_vs_reference,
                "ewr_gain_per_search_second_vs_reference": (
                    gain_vs_reference / extra_vs_reference
                    if gain_vs_reference is not None and extra_vs_reference
                    else None
                ),
                "search_seconds_multiple_of_previous": (
                    point.search_seconds_per_game / previous.search_seconds_per_game
                    if previous is not None and previous.search_seconds_per_game
                    else None
                ),
            }
        )
        previous = point
    return rows


def select_operating_point(
    points: "list[BudgetPoint]", *, margin: float = MEANINGFUL_EWR_GAIN
) -> dict:
    """The cheapest rung that is not meaningfully weaker than the best one.

    The rule is deliberately biased towards the cheap end. Every rung on
    this ladder plays the same boards with the same provider, so a rung
    that leads by less than `margin` has not been shown to be better — and
    buying latency for a difference the match set cannot resolve is exactly
    what section 5 tells this agent not to do.
    """
    if not points:
        raise Phase12BudgetError("cannot select an operating point from an empty ladder")
    usable = [point for point in points if not point.unstable]
    excluded = [point.preset_id for point in points if point.unstable]
    if not usable:
        raise Phase12BudgetError("every rung on the ladder was marked unstable")
    best = max(usable, key=lambda point: point.ewr)
    within = [point for point in usable if best.ewr - point.ewr <= margin]
    # `points` is in ladder (cheapest-first) order, and `within` preserves it.
    selected = within[0]
    return {
        "selected_preset_id": selected.preset_id,
        "selected": selected.describe(),
        "strongest_preset_id": best.preset_id,
        "strongest_ewr": best.ewr,
        "margin": margin,
        "ewr_behind_strongest": best.ewr - selected.ewr,
        "presets_within_margin": [point.preset_id for point in within],
        "excluded_unstable": excluded,
        "rule": (
            "cheapest rung whose EWR is within the engineering margin of the "
            "strongest rung; unstable rungs are excluded"
        ),
    }


def stopping_rule(
    points: "list[BudgetPoint]",
    *,
    next_config: "Phase12SearchConfig | None" = PRESET_LARGE,
    margin: float = MEANINGFUL_EWR_GAIN,
    comfortable_seconds: float = COMFORTABLE_MOVE_SECONDS,
    impractical_seconds: float = IMPRACTICAL_MOVE_SECONDS,
    latency_ratio: float = LATENCY_TO_STRENGTH_RATIO,
    disproportionate_multiple: float = DISPROPORTIONATE_COMPUTE_MULTIPLE,
) -> dict:
    """Section 5, condition by condition, over the rungs actually played.

    Returns every condition with its evidence whether it fired or not, so
    the report can show the rule being applied rather than its conclusion.
    Scaling stops if *any* condition fires — they are reasons to stop, not
    a score to be traded off.
    """
    if not points:
        raise Phase12BudgetError("the stopping rule needs at least one measured rung")
    top = points[-1]
    previous = points[-2] if len(points) > 1 else None
    next_cost_multiple = (
        None if next_config is None else relative_cost(next_config, top.config)
    )
    ladder_seconds = sum(point.search_seconds_per_game for point in points)
    projected_next_seconds = (
        None
        if next_cost_multiple is None
        else top.search_seconds_per_game * next_cost_multiple
    )

    conditions: dict = {}

    delta = None if previous is None else top.ewr - previous.ewr
    conditions["strength_clearly_stopped_improving"] = {
        "fired": bool(previous is not None and delta is not None and delta < margin),
        "top_preset": top.preset_id,
        "previous_preset": None if previous is None else previous.preset_id,
        "delta_ewr_from_previous": delta,
        "margin": margin,
        "reading": (
            "no previous rung to compare against"
            if previous is None
            else f"{top.preset_id} - {previous.preset_id} = {delta:+.4f} EWR "
            f"against a {margin:.2f} margin"
        ),
    }

    extra_seconds = (
        None
        if previous is None
        else top.search_seconds_per_game - previous.search_seconds_per_game
    )
    cost_ratio = (
        None
        if not extra_seconds or not previous or not previous.search_seconds_per_game
        else extra_seconds / previous.search_seconds_per_game
    )
    gain_ratio = None if delta is None else delta / margin
    conditions["latency_rises_much_faster_than_strength"] = {
        "fired": bool(
            cost_ratio is not None
            and gain_ratio is not None
            and cost_ratio > latency_ratio * max(gain_ratio, 0.0)
        ),
        "extra_search_seconds_per_game": extra_seconds,
        "relative_cost_increase": cost_ratio,
        "gain_in_margins": gain_ratio,
        "allowed_ratio": latency_ratio,
        "reading": (
            "no previous rung to compare against"
            if cost_ratio is None or gain_ratio is None
            else f"cost {cost_ratio:+.2f}x for {gain_ratio:+.2f} margins of strength; "
            f"the rule allows {latency_ratio:.1f}x per margin"
        ),
    }

    conditions["human_play_latency_impractical"] = {
        "fired": bool(top.move_seconds_median >= impractical_seconds),
        "median_move_seconds": top.move_seconds_median,
        "p95_move_seconds": top.move_seconds_p95,
        "comfortable_seconds": comfortable_seconds,
        "impractical_seconds": impractical_seconds,
        "past_comfort": bool(top.move_seconds_median > comfortable_seconds),
        "projected_next_median_move_seconds": (
            None
            if next_cost_multiple is None
            else top.move_seconds_median * next_cost_multiple
        ),
        "reading": (
            f"{top.preset_id} decides in {top.move_seconds_median:.3f} s median "
            f"(p95 {top.move_seconds_p95:.3f} s) against a "
            f"{comfortable_seconds:.1f} s comfort line and a "
            f"{impractical_seconds:.1f} s practicality line"
        ),
    }

    operating_point = select_operating_point(points, margin=margin)
    obvious = bool(
        len(points) >= 2
        and operating_point["selected_preset_id"] != top.preset_id
    )
    conditions["useful_operating_point_already_obvious"] = {
        "fired": obvious,
        "selected_preset_id": operating_point["selected_preset_id"],
        "presets_within_margin": operating_point["presets_within_margin"],
        "reading": (
            f"the practical point is {operating_point['selected_preset_id']}, "
            + (
                "below the top rung played — a larger rung would be bought for a "
                "difference this match set cannot resolve"
                if obvious
                else "the top rung played"
            )
        ),
    }

    unstable = [point for point in points if point.unstable]
    conditions["larger_search_creates_instability"] = {
        "fired": bool(unstable),
        "unstable_presets": [point.preset_id for point in unstable],
        "instability": {
            point.preset_id: list(point.instability) for point in unstable
        },
        "reading": (
            "no rung reported a defect"
            if not unstable
            else "defects on " + ", ".join(point.preset_id for point in unstable)
        ),
    }

    disproportionate = bool(
        projected_next_seconds is not None
        and projected_next_seconds > disproportionate_multiple * ladder_seconds
    )
    conditions["next_preset_consumes_disproportionate_compute"] = {
        "fired": disproportionate,
        "next_preset_id": None if next_config is None else next_config.preset_id,
        "next_cost_multiple_of_top": next_cost_multiple,
        "projected_search_seconds_per_game": projected_next_seconds,
        "ladder_search_seconds_per_game": ladder_seconds,
        "allowed_multiple": disproportionate_multiple,
        "reading": (
            "no larger preset is defined"
            if next_config is None
            else f"{next_config.preset_id} projects "
            f"{projected_next_seconds:.1f} s of search per game, against "
            f"{ladder_seconds:.1f} s for the whole ladder played so far"
        ),
    }

    fired = [name for name, block in conditions.items() if block["fired"]]
    return {
        "budget_version": BUDGET_VERSION,
        "thresholds": {
            "meaningful_ewr_gain": margin,
            "comfortable_move_seconds": comfortable_seconds,
            "impractical_move_seconds": impractical_seconds,
            "latency_to_strength_ratio": latency_ratio,
            "disproportionate_compute_multiple": disproportionate_multiple,
        },
        "presets_played": [point.preset_id for point in points],
        "next_preset_considered": None if next_config is None else next_config.describe(),
        "conditions": conditions,
        "conditions_fired": fired,
        "stop_scaling": bool(fired),
        "operating_point": operating_point,
        "statement": (
            f"stop scaling after {top.preset_id}: " + ", ".join(fired)
            if fired
            else f"no stopping condition fired after {top.preset_id}; a larger "
            "preset is allowed by the rule"
        ),
    }


__all__ = [
    "BUDGET_VERSION",
    "BudgetPoint",
    "COMFORTABLE_MOVE_SECONDS",
    "DISPROPORTIONATE_COMPUTE_MULTIPLE",
    "IMPRACTICAL_MOVE_SECONDS",
    "LADDER_PRESET_NAMES",
    "LADDER_PROVIDER",
    "LATENCY_TO_STRENGTH_RATIO",
    "MEANINGFUL_EWR_GAIN",
    "PRESET_LARGE",
    "Phase12BudgetError",
    "ladder_analysis",
    "ladder_arm",
    "ladder_arms",
    "ladder_config",
    "preset_of_arm",
    "relative_cost",
    "select_operating_point",
    "stopping_rule",
]
