"""The four-tier Phase 4 baseline ladder.

Specification source: Phase 4 Agent 2 instructions ("Required baseline
policies").

The ladder is deliberately a *nesting*, not four unrelated agents:

```text
Random      no information at all
Basic       known captures, forward progress, anti-shuffling
Tactical    Basic + expected-value attacks, threat/evasion, Flag tactics
Strategic   Tactical + material preservation, Scout information, Flag defence,
            territorial pressure, draw-counter awareness
```

Each tier adds terms to the tier below rather than replacing it, so a strength
inversion means a specific added term is wrong and can be found by disabling it.
That is what makes the tiers useful as an evaluation ladder: a checkpoint that
beats Tactical but loses to Strategic has been told something meaningful.

None of these is an attempt at strong Stratego. There is no search, no belief
model, no opponent modelling and no learning. Every decision is one pass over
the legal action list scoring each move independently.

Information safety
------------------
Every policy here reads `request.require_public_view()` and nothing else, via
:mod:`stratego.evaluation.heuristics`. See that module's docstring for why the
hidden-information property is structural.

Tuning
------
Weights live in :class:`HeuristicWeights` instances at module level rather than
inline, so Agent 4 can recalibrate by editing one table and bumping the affected
`policy_version`. A weight change without a version bump would silently
invalidate every stored match identity that names the policy.
"""

from dataclasses import dataclass
from typing import ClassVar

from ..engine.constants import BOMB, FLAG, MINER, SPY
from ..engine.coordinates import NEIGHBOURS
from .heuristics import (
    FLAG_CAPTURE_BONUS,
    FLAG_DEFENCE_BONUS,
    CandidateMove,
    DecisionContext,
    ScoredMove,
    build_context,
    build_diagnostics,
    capture_value,
    combat_component,
    manhattan,
    rank_moves,
    select_from_ranked,
)
from .policy import Policy, PolicyInput, PolicyRequirements, PolicyResult

BASELINE_SUITE_VERSION = "phase4_baseline_suite_v1"


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicWeights:
    """Multipliers for the scoring terms, shared by every scoring baseline.

    A tier simply leaves the terms it does not use at zero, so one table
    describes the whole ladder and the diff between two tiers is readable.
    """

    #: Multiplier on the combat term (exact when the defender's type is known,
    #: expected over the public unresolved inventory when it is not).
    combat: float = 1.0
    #: Flat value used by Basic in place of the expected-value calculation.
    speculative_attack: float = 2.0
    #: Score per row gained toward the opponent's home rows.
    advance: float = 1.0
    #: Penalty multiplier on the anti-shuffling signal.
    repetition: float = 3.0
    #: Multiplier on the risk of the destination square from *known* attackers.
    known_risk: float = 1.0
    #: Multiplier on expected exposure to *hidden* pieces that have moved.
    hidden_risk: float = 0.5
    #: Reward for stepping off a square a known attacker is profiting from.
    evade: float = 0.8
    #: Reward per friendly piece defending the destination, when it is risky.
    support: float = 1.5
    #: Premium for sending a Miner at a Bomb the observer has legally seen.
    miner_bomb: float = 12.0
    #: Reward per square of approach by the Spy toward a revealed Marshal.
    spy_hunt: float = 1.5
    #: Reward for a multi-square Scout move that probes an unknown piece.
    scout_probe: float = 4.0
    #: Reward per empty square adjacent to the destination.
    mobility: float = 0.4
    #: Penalty per unresolved Bomb for risking a Miner speculatively.
    miner_preservation: float = 1.2
    #: Penalty for advancing an identified piece the opponent can still answer,
    #: scaled by the expected cost of being attacked by an unknown mover.
    exposure: float = 0.06
    #: Reward for pressing squares surrounded by never-moved opponent pieces.
    pressure: float = 1.5
    #: Penalty for stripping the last movable defender from beside my own Flag.
    flag_guard: float = 25.0
    #: Multiplier on the draw-counter term, scaled by battleless pressure.
    battleless: float = 8.0


#: Basic: known captures, forward progress, and not shuffling. Nothing else.
BASIC_WEIGHTS = HeuristicWeights(
    combat=1.0,
    speculative_attack=2.0,
    advance=1.0,
    repetition=3.0,
    known_risk=0.0,
    hidden_risk=0.0,
    evade=0.0,
    support=0.0,
    miner_bomb=0.0,
    spy_hunt=0.0,
    scout_probe=0.0,
    mobility=0.0,
    miner_preservation=0.0,
    exposure=0.0,
    pressure=0.0,
    flag_guard=0.0,
    battleless=0.0,
)

#: Tactical: adds local combat reasoning -- expected-value attacks, threat and
#: evasion, Miner/Bomb and Spy/Marshal knowledge, Scout probes, Flag tactics.
TACTICAL_WEIGHTS = HeuristicWeights(
    combat=1.0,
    advance=1.0,
    repetition=3.0,
    known_risk=1.0,
    hidden_risk=0.5,
    evade=0.8,
    support=1.5,
    miner_bomb=12.0,
    spy_hunt=1.5,
    scout_probe=4.0,
    mobility=0.0,
    miner_preservation=0.0,
    exposure=0.0,
    pressure=0.0,
    flag_guard=0.0,
    battleless=0.0,
)

#: Strategic: Tactical plus the longer-horizon terms.
STRATEGIC_WEIGHTS = HeuristicWeights(
    combat=1.0,
    advance=1.0,
    repetition=3.0,
    known_risk=1.0,
    hidden_risk=0.6,
    evade=0.9,
    support=1.5,
    miner_bomb=12.0,
    spy_hunt=1.5,
    scout_probe=5.0,
    mobility=0.4,
    miner_preservation=1.2,
    exposure=0.06,
    pressure=1.5,
    flag_guard=25.0,
    battleless=8.0,
)


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class RandomLegalPolicy(Policy):
    """Uniform over the legal action set. The absolute floor of the ladder.

    Declares no requirements at all, which makes it the cheapest policy in the
    suite and also the strongest possible statement about information safety:
    it is handed nothing beyond the legal action list, so there is nothing it
    could leak.
    """

    policy_id = "random_legal"
    policy_version = "1.0.0"
    requirements = PolicyRequirements(public_view=False)
    stochastic = True
    description = "Uniform random legal move; performance floor and stochastic control."

    def decide(self, request: PolicyInput) -> PolicyResult:
        actions = request.legal_actions
        index = request.random_stream().randrange(len(actions))
        return self.result(
            request,
            actions[index],
            {
                "rule": "uniform_legal",
                "candidate_count": len(actions),
                "sampled": True,
            },
        )


# ---------------------------------------------------------------------------
# The scoring baselines
# ---------------------------------------------------------------------------


class ScoringPolicy(Policy):
    """Shared machinery for every baseline that scores each legal move.

    Subclasses implement :meth:`score` for one move. The decide loop, the
    deterministic ordering, the optional near-best sampling and the diagnostics
    are identical across tiers so that a strength difference can only come from
    the scoring itself.
    """

    weights: ClassVar[HeuristicWeights] = HeuristicWeights()
    #: Candidates within this much of the best score form the sampling pool.
    #: Zero makes the policy fully deterministic.
    selection_margin: ClassVar[float] = 0.0
    requirements = PolicyRequirements(public_view=True)

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        raise NotImplementedError

    def decide(self, request: PolicyInput) -> PolicyResult:
        context = build_context(request)
        ranked = rank_moves(self.score(context, move) for move in context.moves)
        chosen, sampled = select_from_ranked(request, ranked, margin=self.selection_margin)
        return self.result(
            request, chosen.action_id, build_diagnostics(chosen, ranked, sampled=sampled)
        )


class BasicHeuristicPolicy(ScoringPolicy):
    """Deliberately modest public-information scoring.

    Three ideas only: take a capture that is known to win, refuse one that is
    known to lose, and otherwise walk forward without shuffling the same piece
    back and forth. Attacks on unknown pieces get a flat nudge rather than the
    expected-value calculation the higher tiers use, which is the main reason
    Basic bleeds material into Bombs and higher ranks.
    """

    policy_id = "basic_heuristic"
    policy_version = "1.0.0"
    weights = BASIC_WEIGHTS
    selection_margin = 0.75
    stochastic = True
    description = "Known captures, forward progress, anti-shuffling. No threat model."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        weights = self.weights
        components: list[tuple[str, float]] = []
        score = 0.0
        family = "quiet"

        if move.is_attack:
            if move.target_type == FLAG:
                return ScoredMove(
                    move.action_id,
                    FLAG_CAPTURE_BONUS,
                    "flag_capture",
                    (("flag_capture", FLAG_CAPTURE_BONUS),),
                )
            if move.target_type is not None:
                value = weights.combat * capture_value(move.piece_type, move.target_type)
                family = (
                    "winning_capture"
                    if value > 0.0
                    else "losing_capture"
                    if value < 0.0
                    else "even_trade"
                )
            else:
                value = weights.combat * weights.speculative_attack
                family = "speculative_attack"
            score += value
            components.append(("combat", value))

        if move.advance and weights.advance:
            advance = weights.advance * move.advance
            score += advance
            components.append(("advance", advance))
            if family == "quiet":
                family = "advance"

        penalty = context.repetition_penalty(move)
        if penalty and weights.repetition:
            value = -weights.repetition * penalty
            score += value
            components.append(("repetition", value))

        return ScoredMove(move.action_id, score, family, tuple(components))


class TacticalRuleBasedPolicy(ScoringPolicy):
    """Local combat reasoning over public facts.

    What it adds over Basic:

    - attacks on unknown pieces are valued in *expectation* over the publicly
      deducible unresolved inventory instead of by a flat constant, so probing a
      board still full of Bombs is correctly unattractive;
    - the destination square is priced for danger from known attackers and, more
      weakly, from hidden pieces that have already moved;
    - stepping a piece off a square a known attacker is profiting from is
      rewarded;
    - a Miner is pushed at a Bomb the observer has legally seen, and the
      Spy/Marshal inversion falls out of the combat table without a special case;
    - a multi-square Scout move that probes an unknown piece is cheap
      information and is rewarded as such;
    - capturing a piece standing next to my own Flag outranks everything except
      capturing the opponent's Flag.
    """

    policy_id = "tactical_rule_based"
    policy_version = "1.0.0"
    weights = TACTICAL_WEIGHTS
    selection_margin = 0.5
    stochastic = True
    description = "Expected-value combat, threat and evasion, Flag and Miner tactics."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        components: list[tuple[str, float]] = []
        score, family = self._tactical_terms(context, move, components)
        return ScoredMove(move.action_id, score, family, tuple(components))

    def _tactical_terms(
        self,
        context: DecisionContext,
        move: CandidateMove,
        components: list[tuple[str, float]],
    ) -> tuple[float, str]:
        """Score the tactical layer, appending each nonzero component.

        Split out from :meth:`score` because Strategic builds directly on top of
        it; keeping one implementation is what makes the tiers a true nesting.
        """
        weights = self.weights
        score = 0.0

        combat, family = combat_component(context, move)
        if family == "flag_capture":
            components.append(("flag_capture", FLAG_CAPTURE_BONUS))
            return FLAG_CAPTURE_BONUS, family
        if combat:
            value = weights.combat * combat
            score += value
            components.append(("combat", value))

        # Removing a piece from beside my own Flag. Only worth doing if the
        # capture is not a known loss, which would leave the attacker in place
        # and cost me a piece as well.
        if move.is_attack and weights.combat:
            if (
                move.target_piece_id in context.own_flag_attackers
                and not context.is_known_losing_attack(move)
            ):
                score += FLAG_DEFENCE_BONUS
                components.append(("flag_defence", FLAG_DEFENCE_BONUS))
                return score, "flag_defence"

        if move.target_type == BOMB and move.piece_type == MINER and weights.miner_bomb:
            score += weights.miner_bomb
            components.append(("miner_bomb", weights.miner_bomb))
            family = "miner_demolition"

        if weights.known_risk:
            risk = context.known_risk(move.destination, move.piece_type)
            if risk:
                value = weights.known_risk * risk
                score += value
                components.append(("known_risk", value))

        if weights.hidden_risk:
            risk = context.hidden_risk(move.destination, move.piece_type)
            if risk:
                value = weights.hidden_risk * risk
                score += value
                components.append(("hidden_risk", value))

        if weights.evade:
            standing_risk = context.known_risk(move.source, move.piece_type)
            if standing_risk < 0.0:
                value = weights.evade * -standing_risk
                score += value
                components.append(("evade", value))
                if family == "quiet":
                    family = "evade"

        if weights.support:
            destination_risk = context.known_risk(move.destination, move.piece_type)
            if destination_risk < 0.0:
                value = weights.support * context.own_support[move.destination]
                if value:
                    score += value
                    components.append(("support", value))

        if weights.spy_hunt and move.piece_type == SPY:
            # The Spy beats a Marshal only by attacking it, so closing on a
            # Marshal the opponent has already revealed is worth doing.
            marshal_square = context.known_opponent_marshal_square
            if marshal_square is not None:
                gain = manhattan(move.source, marshal_square) - manhattan(
                    move.destination, marshal_square
                )
                if gain:
                    value = weights.spy_hunt * gain
                    score += value
                    components.append(("spy_hunt", value))
                    if family in ("quiet", "evade"):
                        family = "spy_hunt"

        if weights.scout_probe and move.is_scout_run and move.is_attack and not move.target_known:
            score += weights.scout_probe
            components.append(("scout_probe", weights.scout_probe))
            family = "scout_probe"

        if move.advance and weights.advance:
            value = weights.advance * move.advance
            score += value
            components.append(("advance", value))
            if family == "quiet":
                family = "advance"

        penalty = context.repetition_penalty(move)
        if penalty and weights.repetition:
            value = -weights.repetition * penalty
            score += value
            components.append(("repetition", value))

        return score, family


class StrategicRuleBasedPolicy(TacticalRuleBasedPolicy):
    """Tactical plus longer-horizon public-information judgement.

    What it adds over Tactical:

    - **Miner preservation.** Every unresolved Bomb makes a Miner more valuable
      than its material price, so speculative Miner attacks and risky Miner
      moves are taxed in proportion to how many Bombs remain unaccounted for.
    - **Scout information value.** Scout runs are worth more while the opponent
      inventory is largely unresolved and taper off as it resolves.
    - **Territorial pressure.** Squares surrounded by never-moved opponent
      pieces are where a defensive block sits. Pressing them uses `has_moved`,
      which is public, and says nothing about any individual piece's type.
    - **Exposure control.** An identified piece the opponent can still answer is
      a target; advancing it is discouraged in proportion to how badly the
      unresolved inventory answers it. A revealed Marshal is barely restrained
      because almost nothing beats it; a revealed Spy or Miner is restrained a
      great deal.
    - **Flag defence.** Stripping the last movable defender from beside my own
      Flag is penalised, using only my own Flag's location.
    - **Draw-counter awareness.** As the battleless counter runs down, a policy
      that is ahead on public material is pushed toward forcing combat and one
      that is behind is pushed toward quiet moves.
    """

    policy_id = "strategic_rule_based"
    #: 1.1.0 re-prices the exposure term by vulnerability rather than by
    #: material value (Phase 4 Agent 4 calibration). At 1.0.0 that term made
    #: Strategic measurably *weaker* than Tactical, 0.472 EWR over 1,024 paired
    #: units; at 1.1.0 it is 0.556. Nothing else changed, and no weight moved.
    policy_version = "1.1.0"
    weights = STRATEGIC_WEIGHTS
    selection_margin = 0.5
    stochastic = True
    description = (
        "Tactical plus material preservation, Scout information, territorial "
        "pressure, Flag defence and draw-counter awareness."
    )

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        components: list[tuple[str, float]] = []
        score, family = self._tactical_terms(context, move, components)
        if family in ("flag_capture", "flag_defence"):
            # Both are game-deciding and must not be diluted by positional terms.
            return ScoredMove(move.action_id, score, family, tuple(components))

        weights = self.weights

        if weights.miner_preservation and move.piece_type == MINER:
            # Miners are the only answer to a Bomb, so their scarcity value rises
            # as Bombs stay unresolved and as Miners are spent.
            premium = (
                weights.miner_preservation
                * context.unresolved_bombs
                / max(1, context.own_miner_count)
            )
            if premium:
                exposure = 0.0
                if move.is_attack and move.target_type is None:
                    # A speculative Miner attack risks the piece that is the only
                    # answer to the Bombs still on the board.
                    exposure += 1.0
                if context.known_risk(move.destination, MINER) < 0.0:
                    exposure += 1.0
                if exposure:
                    value = -premium * exposure
                    score += value
                    components.append(("miner_preservation", value))

        if weights.scout_probe and move.is_scout_run and context.unresolved_total:
            # Information is worth most while little is known and decays as the
            # opponent's inventory resolves.
            share = context.unresolved_total / 40.0
            value = weights.scout_probe * share * 0.5
            score += value
            components.append(("scout_information", value))

        if weights.pressure:
            cluster = context.unmoved_opponent_cluster(move.destination)
            if cluster:
                # A Miner has the most to gain from pressing an unmoved block,
                # since that is where Bombs sit; everyone else gains only the
                # information and pays for it in the risk terms above.
                scale = 2.0 if move.piece_type == MINER else 1.0
                value = weights.pressure * cluster * scale
                score += value
                components.append(("pressure", value))
                if family in ("quiet", "advance"):
                    family = "pressure"

        if weights.mobility:
            # Space: a piece with room around it keeps its options, and a piece
            # boxed into a corner is the one that ends up with no legal move.
            value = weights.mobility * context.empty_neighbours[move.destination]
            score += value
            components.append(("mobility", value))

        if weights.exposure and move.advance > 0 and context.is_exposed(move.piece_id):
            # Priced by how badly the identified piece can actually be answered
            # from the unresolved inventory, not by what it is worth. The two
            # are anti-correlated: the Marshal is the most valuable piece on the
            # board and also the least answerable, so pricing exposure by value
            # taxed the Marshal six times as hard as the Spy while the Spy is
            # the piece that actually dies once named. Worse, a piece becomes
            # identified by *winning* a fight, so a value-priced penalty froze
            # exactly the attackers that had just proved they win.
            # `expected_defence_value` is negative only while the opponent still
            # holds something that beats this type, and it is a public deduction
            # over the unresolved inventory, so it is invariant under
            # `permute_hidden_identities` like every other term here.
            vulnerability = context.expected_defence_value(move.piece_type)
            if vulnerability < 0.0:
                value = weights.exposure * vulnerability * move.advance
                score += value
                components.append(("exposure", value))

        if weights.flag_guard:
            value = self._flag_guard_term(context, move)
            if value:
                score += value
                components.append(("flag_guard", value))

        if weights.battleless and context.battleless_pressure > 0.5:
            # Only bites in the second half of the battleless window, so normal
            # play is unaffected.
            urgency = weights.battleless * (context.battleless_pressure - 0.5) * 2.0
            ahead = context.material_edge > 0.0
            value = 0.0
            if ahead and move.is_attack:
                value = urgency
            elif not ahead and not move.is_attack:
                value = urgency * 0.5
            if value:
                score += value
                components.append(("battleless", value))

        return ScoredMove(move.action_id, score, family, tuple(components))

    def _flag_guard_term(self, context: DecisionContext, move: CandidateMove) -> float:
        """Penalise walking the last movable defender away from my own Flag."""
        flag_square = context.own_flag_square
        if flag_square is None:
            return 0.0
        if move.source not in NEIGHBOURS[flag_square]:
            return 0.0
        if move.destination in NEIGHBOURS[flag_square]:
            # Still guarding; sidestepping around the Flag is fine.
            return 0.0

        occupancy = context.view.occupancy
        pieces = context.view.pieces
        guards = 0
        for neighbour in NEIGHBOURS[flag_square]:
            piece_id = occupancy[neighbour]
            if piece_id is None:
                continue
            piece = pieces[piece_id]
            if piece.owner != context.me or piece.piece_type is None:
                continue
            if piece.piece_type in (FLAG, BOMB):
                # A Bomb shields the Flag but cannot recapture an attacker.
                continue
            guards += 1
        return -self.weights.flag_guard if guards <= 1 else 0.0


#: The ladder, weakest first. Agent 4 calibrates the tiers; Agent 2 only asserts
#: that they are distinct policies built from a nested set of heuristics.
LADDER_POLICY_CLASSES: tuple[type[Policy], ...] = (
    RandomLegalPolicy,
    BasicHeuristicPolicy,
    TacticalRuleBasedPolicy,
    StrategicRuleBasedPolicy,
)


__all__ = [
    "BASELINE_SUITE_VERSION",
    "BASIC_WEIGHTS",
    "LADDER_POLICY_CLASSES",
    "STRATEGIC_WEIGHTS",
    "TACTICAL_WEIGHTS",
    "BasicHeuristicPolicy",
    "HeuristicWeights",
    "RandomLegalPolicy",
    "ScoringPolicy",
    "StrategicRuleBasedPolicy",
    "TacticalRuleBasedPolicy",
]
