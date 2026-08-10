"""Stress and unusual opponents for the Phase 4 evaluation suite.

Specification source: Phase 4 Agent 2 instructions ("Required stress/unusual
policies").

These six policies are not attempts at strength. Each takes one idea to an
extreme so the games it produces sit somewhere the ladder never goes:

```text
stress_scout_rush           Scout-heavy aggression; fast reveals, fast losses
stress_miner_rush           Miner rush at never-moved blocks; Bomb hunting
stress_draw_seeker          refuses combat; drives the battleless-move draw
stress_berserker            attacks on sight; ignores risk entirely
stress_information_miser    hoards information; avoids reveals and combat
stress_chaos                fresh random objective every ply; high entropy
```

Why they exist
--------------
A ladder of four policies that all play "sensible" Stratego produces a narrow
band of game shapes. A checkpoint tuned against only that band can look strong
and still fall apart against an opponent that opens with eight Scout runs or
never initiates combat at all. These policies exist to widen the distribution --
game length, attack rate, reveal rate, draw rate and Flag-capture rate all move
substantially -- so brittleness shows up in evaluation rather than later.

They are held to exactly the same contract as the ladder: observer-safe input
only, always a legal action, reproducible from `(public input, policy seed,
ply)`, and permutation-invariant decisions and diagnostics.
"""

from typing import ClassVar

from ..engine.constants import BOMB, FLAG, MINER, SCOUT
from .heuristics import (
    FLAG_CAPTURE_BONUS,
    PIECE_VALUES,
    CandidateMove,
    DecisionContext,
    ScoredMove,
    build_context,
    build_diagnostics,
    capture_value,
    in_own_half,
    rank_moves,
    select_from_ranked,
)
from .baselines import ScoringPolicy
from .policy import Policy, PolicyInput, PolicyRequirements, PolicyResult

#: Applied wherever a stress policy must express "essentially never do this"
#: while still leaving the move selectable when it is the only legal one.
STRONG_AVERSION = 500.0


def _flag_capture_override(move: CandidateMove) -> "ScoredMove | None":
    """Take a known Flag whatever else the policy believes.

    Even a policy built to refuse combat should not walk past a won game; a
    stress opponent that declines a mate is noise, not a useful distribution.
    """
    if move.target_type == FLAG:
        return ScoredMove(
            move.action_id,
            FLAG_CAPTURE_BONUS,
            "flag_capture",
            (("flag_capture", FLAG_CAPTURE_BONUS),),
        )
    return None


class ScoutRushPolicy(ScoringPolicy):
    """Scout-heavy aggression: move Scouts, far, at anything.

    Produces very high Scout-move frequency and an early collapse of the
    opponent's information advantage, at the cost of throwing away eight pieces.
    A policy that only knows how to punish cautious openings will be exposed.
    """

    policy_id = "stress_scout_rush"
    policy_version = "1.0.0"
    selection_margin = 1.0
    stochastic = True
    description = "Scout-heavy aggression: long Scout runs and constant probing."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        override = _flag_capture_override(move)
        if override is not None:
            return override

        components: list[tuple[str, float]] = []
        score = 0.0
        family = "quiet"

        if move.piece_type == SCOUT:
            value = 30.0 + 4.0 * move.distance
            score += value
            components.append(("scout_preference", value))
            family = "scout_run" if move.distance > 1 else "scout_step"
        else:
            score -= 20.0
            components.append(("non_scout", -20.0))

        if move.is_attack:
            if move.target_type is not None:
                value = capture_value(move.piece_type, move.target_type)
            else:
                value = 12.0
            score += value
            components.append(("combat", value))
            if family == "quiet":
                family = "attack"

        if move.advance:
            score += 2.0 * move.advance
            components.append(("advance", 2.0 * move.advance))

        penalty = context.repetition_penalty(move)
        if penalty:
            score -= 6.0 * penalty
            components.append(("repetition", -6.0 * penalty))

        return ScoredMove(move.action_id, score, family, tuple(components))


class MinerRushPolicy(ScoringPolicy):
    """Miner rush and aggressive Bomb hunting.

    Drives Miners at blocks of never-moved opponent pieces -- a public
    inference, since `has_moved` is public and says nothing about type -- and
    attacks anything unmoved it reaches. It spends its Miners fast and often
    resolves the opponent's Bomb structure far earlier than the ladder does.
    """

    policy_id = "stress_miner_rush"
    policy_version = "1.0.0"
    selection_margin = 1.0
    stochastic = True
    description = "Miner rush at never-moved blocks; aggressive Bomb hunting."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        override = _flag_capture_override(move)
        if override is not None:
            return override

        components: list[tuple[str, float]] = []
        score = 0.0
        family = "quiet"

        if move.piece_type == MINER:
            score += 25.0
            components.append(("miner_preference", 25.0))
            family = "miner_advance"
            cluster = context.unmoved_opponent_cluster(move.destination)
            if cluster:
                value = 10.0 * cluster
                score += value
                components.append(("unmoved_cluster", value))
            if move.is_attack:
                if move.target_type == BOMB:
                    score += 60.0
                    components.append(("known_bomb", 60.0))
                    family = "bomb_demolition"
                elif move.target_type is None and not move.target_has_moved:
                    # An unmoved unknown piece is the best Bomb candidate the
                    # public record offers.
                    score += 35.0
                    components.append(("bomb_candidate", 35.0))
                    family = "bomb_hunt"

        if move.is_attack:
            if move.target_type is not None:
                value = capture_value(move.piece_type, move.target_type)
            else:
                value = context.expected_capture_value(move.piece_type, move.target_has_moved)
            score += value
            components.append(("combat", value))
            if family == "quiet":
                family = "attack"

        if move.advance:
            score += 1.5 * move.advance
            components.append(("advance", 1.5 * move.advance))

        penalty = context.repetition_penalty(move)
        if penalty:
            score -= 6.0 * penalty
            components.append(("repetition", -6.0 * penalty))

        return ScoredMove(move.action_id, score, family, tuple(components))


class DrawSeekerPolicy(ScoringPolicy):
    """Defensive and draw-seeking: refuses combat and stays home.

    The battleless-move counter only resets on combat, so a policy that never
    initiates one and keeps its pieces in its own half pushes hard toward
    `battleless_move_limit_draw`. It is the natural probe for a checkpoint that
    has learned to win but not to make progress.
    """

    policy_id = "stress_draw_seeker"
    policy_version = "1.0.0"
    selection_margin = 1.5
    stochastic = True
    description = "Defensive draw-seeker: avoids all combat and stays in its own half."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        override = _flag_capture_override(move)
        if override is not None:
            return override

        components: list[tuple[str, float]] = []
        score = 0.0
        family = "shuffle"

        if move.is_attack:
            score -= STRONG_AVERSION
            components.append(("combat_aversion", -STRONG_AVERSION))
            family = "forced_attack"

        if move.advance > 0:
            value = -4.0 * move.advance
            score += value
            components.append(("retreat_preference", value))
        elif move.advance < 0:
            value = 1.0 * -move.advance
            score += value
            components.append(("retreat_preference", value))

        if in_own_half(move.destination, context.me):
            score += 5.0
            components.append(("stay_home", 5.0))

        risk = context.known_risk(move.destination, move.piece_type)
        if risk:
            score += 2.0 * risk
            components.append(("known_risk", 2.0 * risk))

        # Deliberately does not penalise repetition: shuffling is the plan.
        return ScoredMove(move.action_id, score, family, tuple(components))


class BerserkerPolicy(ScoringPolicy):
    """High-pressure attacker: takes every fight, prices none of them.

    Ignores the risk terms entirely and values any attack above any quiet move.
    Produces the shortest games and the highest combat rate in the suite, which
    makes it the fastest way to find a policy that only performs well when it is
    allowed to develop quietly.
    """

    policy_id = "stress_berserker"
    policy_version = "1.0.0"
    selection_margin = 1.0
    stochastic = True
    description = "Attacks on sight and charges forward; no risk assessment at all."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        override = _flag_capture_override(move)
        if override is not None:
            return override

        components: list[tuple[str, float]] = []
        score = 0.0
        family = "advance"

        if move.is_attack:
            score += 60.0
            components.append(("attack_preference", 60.0))
            family = "attack"
            if move.target_type is not None:
                value = 0.25 * capture_value(move.piece_type, move.target_type)
                score += value
                components.append(("combat", value))

        if move.advance:
            score += 4.0 * move.advance
            components.append(("advance", 4.0 * move.advance))

        # The only thing it dislikes is standing still, so it keeps charging
        # rather than oscillating in place.
        penalty = context.repetition_penalty(move)
        if penalty:
            score -= 8.0 * penalty
            components.append(("repetition", -8.0 * penalty))

        return ScoredMove(move.action_id, score, family, tuple(components))


class InformationMiserPolicy(ScoringPolicy):
    """Low-information-conservation: reveals as little as possible.

    Two public facts reveal a type in this ruleset -- combat reveals both
    participants, and a multi-square Scout move reveals the Scout. This policy
    avoids both, and prefers to move pieces whose type the opponent has already
    learned, since those cost nothing further to expose.

    The resulting games are long, quiet and unusually opaque, which is a
    genuinely different regime from anything the ladder generates.
    """

    policy_id = "stress_information_miser"
    policy_version = "1.0.0"
    selection_margin = 1.0
    stochastic = True
    description = "Hoards information: avoids combat, Scout runs and fresh reveals."

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        override = _flag_capture_override(move)
        if override is not None:
            return override

        components: list[tuple[str, float]] = []
        score = 0.0
        family = "quiet"

        if move.is_attack:
            # Combat reveals both participants unconditionally.
            penalty = -40.0 - 0.5 * PIECE_VALUES[move.piece_type]
            score += penalty
            components.append(("reveal_aversion", penalty))
            family = "forced_attack"

        if move.is_scout_run:
            # A multi-square Scout move reveals the Scout to the opponent.
            score -= 25.0
            components.append(("scout_reveal_aversion", -25.0))

        if context.is_exposed(move.piece_id):
            # Already identified, so moving it leaks nothing new.
            score += 12.0
            components.append(("already_exposed", 12.0))
        else:
            value = -0.15 * PIECE_VALUES[move.piece_type]
            score += value
            components.append(("conceal_value", value))

        if move.advance:
            score += 0.5 * move.advance
            components.append(("advance", 0.5 * move.advance))

        risk = context.known_risk(move.destination, move.piece_type)
        if risk:
            score += risk
            components.append(("known_risk", risk))

        penalty = context.repetition_penalty(move)
        if penalty:
            score -= 2.0 * penalty
            components.append(("repetition", -2.0 * penalty))

        return ScoredMove(move.action_id, score, family, tuple(components))


class ChaosPolicy(Policy):
    """Deliberately irregular: a fresh random objective every ply.

    Each decision draws its own weight vector over a handful of crude public
    features and then plays the best move *for that objective*. The result is
    neither uniform noise nor a consistent strategy: it is coherent within a
    ply and incoherent across plies, which produces action distributions no
    fixed policy generates.

    It overrides `decide` rather than subclassing `ScoringPolicy` because the
    weights themselves come from the decision stream, and the whole point is
    that they change every ply.
    """

    policy_id = "stress_chaos"
    policy_version = "1.0.0"
    requirements = PolicyRequirements(public_view=True)
    stochastic = True
    description = "Draws a fresh random objective each ply; maximum behavioural entropy."

    #: Feature names, fixed so the diagnostics are stable and comparable.
    FEATURES: ClassVar[tuple[str, ...]] = (
        "attack",
        "advance",
        "distance",
        "lateral",
        "crowding",
        "value",
    )

    def decide(self, request: PolicyInput) -> PolicyResult:
        context = build_context(request)
        # One stream, drawn once: `random_stream()` reseeds on every call, so
        # taking it twice would produce the same numbers twice.
        rng = request.random_stream()
        weights = tuple(rng.uniform(-1.0, 1.0) for _ in self.FEATURES)

        scored = []
        for move in context.moves:
            override = _flag_capture_override(move)
            if override is not None:
                scored.append(override)
                continue
            features = (
                1.0 if move.is_attack else 0.0,
                move.advance / 9.0,
                move.distance / 9.0,
                1.0 if move.advance == 0 else 0.0,
                context.empty_neighbours[move.destination] / 4.0,
                PIECE_VALUES[move.piece_type] / 100.0,
            )
            score = sum(weight * feature for weight, feature in zip(weights, features))
            scored.append(ScoredMove(move.action_id, score, "chaos", ()))

        ranked = rank_moves(scored)
        chosen, sampled = select_from_ranked(request, ranked, margin=0.0)
        diagnostics = build_diagnostics(chosen, ranked, sampled=sampled)
        diagnostics["objective"] = {
            name: round(weight, 4) for name, weight in zip(self.FEATURES, weights)
        }
        return self.result(request, chosen.action_id, diagnostics)


#: The stress suite. Order is fixed so report tables and the data file are
#: stable across runs.
STRESS_POLICY_CLASSES: tuple[type[Policy], ...] = (
    ScoutRushPolicy,
    MinerRushPolicy,
    DrawSeekerPolicy,
    BerserkerPolicy,
    InformationMiserPolicy,
    ChaosPolicy,
)


__all__ = [
    "STRESS_POLICY_CLASSES",
    "STRONG_AVERSION",
    "BerserkerPolicy",
    "ChaosPolicy",
    "DrawSeekerPolicy",
    "InformationMiserPolicy",
    "MinerRushPolicy",
    "ScoutRushPolicy",
]
