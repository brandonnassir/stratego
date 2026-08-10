"""Phase 4 Agent 4 calibration helpers.

Three things Agent 4 needs that are too large to leave in an acceptance script,
and that a later checkpoint evaluation will want to rerun unchanged:

1. **A parallel policy-level hidden-information audit.** Agents 1 and 2 each ran
   a permutation sweep in the low thousands; the Phase 4 gate is 100,000 valid
   trials across every catalogued policy. At roughly 7 ms per trial that is
   twelve minutes in one process, so the audit is decomposed into deterministic
   chunks that can run in a pool. A chunk's work is fixed by
   ``(root_seed, chunk_index)`` alone, so the number of workers changes only how
   long the audit takes -- exactly the property the match runner already has.

2. **Behavioural profiling by replay.** The stress characterisation needs attack
   rates, Scout and Miner usage, reveal frequency and action-distribution
   diversity, and :class:`~stratego.evaluation.match_runner.MatchResult` carries
   none of them -- it is an outcome row. Rather than reimplement the game loop
   and risk profiling something subtly different from what the league played,
   this module replays a league row's own stored action history through the
   engine and counts. No policy is consulted during a replay, so the profile is
   by construction a description of the games the league actually played.

3. **Strength-tier partitioning.** Turning a table of paired confidence
   intervals into "how many statistically distinguishable tiers are there" is
   the Phase 4 gate, and it is a judgement that should be one auditable function
   rather than prose in a report.

Nothing here is a policy, and nothing here plays a policy against privileged
information: the audit builds its policy inputs through
:func:`~stratego.evaluation.policy.build_policy_input` like the runner does, and
the profiler reads the privileged state only *after* a game is over, to count
what happened.

This module adds no rules. The engine remains the authority on legality, combat,
terminal precedence and knowledge.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..engine.actions import decode_action
from ..engine.constants import (
    DRAW_TERMINAL_REASONS,
    EVALUATION_RULES,
    MINER,
    PIECE_TYPE_NAMES,
    PIECES_PER_PLAYER,
    RED,
    SCOUT,
    TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
    TERMINAL_FLAG_CAPTURE,
)
from ..engine.coordinates import square_column, square_row
from ..engine.legal_moves import legal_actions
from ..engine.permutation import (
    belief_targets_differ,
    hidden_opponent_piece_ids,
    permute_hidden_identities,
)
from ..engine.random_play import make_random_setups, select_random_action
from ..engine.setup import deserialize_setup
from ..engine.snapshot import clone_state
from ..engine.state import GameState, create_game
from ..engine.transition import apply_action
from .baselines import ScoringPolicy
from .heuristics import build_context, rank_moves
from .match_spec import EVALUATION_SUITE_VERSION
from .policy import Policy, build_policy_input
from .registry import ALL_POLICY_IDS, build_policies, build_policy

if TYPE_CHECKING:  # pragma: no cover
    from .match_runner import MatchResult

CALIBRATION_VERSION = "phase4_calibration_v1"

#: Plies at which a sampled game is snapshotted for the audit. Ten checkpoints
#: rather than Agent 2's seven, because how much is hidden -- and therefore what
#: a leak could look like -- changes completely across a game: at ply 4 almost
#: the whole opponent army is unresolved, by ply 200 much of it is not.
AUDIT_PLIES: tuple[int, ...] = (4, 8, 15, 24, 40, 60, 85, 115, 150, 200)

#: How audit positions are reached. ``random_walk`` is cheap and spreads widely
#: over the state space; ``baseline_play`` costs about three times as much but
#: produces the *kind* of position a league actually visits -- more revealed
#: pieces, a partly resolved inventory, pieces clustered where baselines fight.
#: A leak that only manifests in realistic positions would hide from a purely
#: random sweep, so the audit uses both.
POSITION_SOURCES: tuple[str, ...] = ("random_walk", "baseline_play")

#: How many deterministic slices an audit is cut into. Fixed rather than derived
#: from the worker count, so the audit's findings do not depend on the machine.
DEFAULT_AUDIT_CHUNKS = 32

#: Movement-direction buckets, own-relative so that pooling a policy's red and
#: blue games does not cancel "forward" against "forward".
DIRECTION_LABELS: tuple[str, ...] = (
    "forward_step",
    "forward_run",
    "backward_step",
    "backward_run",
    "lateral_step",
    "lateral_run",
)


# ---------------------------------------------------------------------------
# Position sampling
# ---------------------------------------------------------------------------


def _audit_request(state: GameState, policy: Policy, seed: int):
    """An observer-safe policy input for the acting player.

    Identical in shape to what the match runner builds. The audit deliberately
    uses one fixed policy seed for both halves of a trial: a differing action
    then means the *position* was distinguishable, which is the thing under test.
    """
    return build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=seed,
        requirements=policy.requirements,
        suite_version=EVALUATION_SUITE_VERSION,
        match_id="audit",
        paired_unit_id="audit",
    )


def sample_positions(
    seed: int,
    *,
    source: str = "random_walk",
    plies: "Sequence[int]" = AUDIT_PLIES,
    policy_ids: "Sequence[str] | None" = None,
) -> list[GameState]:
    """Nonterminal positions from one game, snapshotted at each checkpoint ply.

    One game yields up to ``len(plies)`` positions, which is far cheaper than
    replaying a game per position for the same phase coverage. Every position is
    a :func:`clone_state`, so a consumer that mutates one cannot disturb another.

    ``source="baseline_play"`` walks the game with two catalogued policies chosen
    deterministically from ``seed`` instead of uniform random moves.
    """
    if source not in POSITION_SOURCES:
        raise ValueError(f"unknown position source {source!r}; expected one of {POSITION_SOURCES}")

    red_setup, blue_setup = make_random_setups(seed)
    state = create_game(
        red_setup, blue_setup, rules=EVALUATION_RULES, game_id=f"audit-{source}-{seed}"
    )
    rng = random.Random(seed * 2_654_435_761 % (2**61))

    walkers: dict[int, Policy] | None = None
    if source == "baseline_play":
        catalogue = list(policy_ids or ALL_POLICY_IDS)
        # Deterministic from the seed, and the two are always distinct so a
        # position is never the product of one policy playing itself.
        first = catalogue[seed % len(catalogue)]
        second = catalogue[(seed // len(catalogue) + 1) % len(catalogue)]
        if second == first:
            second = catalogue[(catalogue.index(first) + 1) % len(catalogue)]
        walkers = {0: build_policy(first), 1: build_policy(second)}

    targets = set(plies)
    limit = max(plies)
    positions: list[GameState] = []
    while not state.terminal and state.total_moves < limit:
        legal = legal_actions(state)
        if walkers is None:
            action_id = select_random_action(state, rng, legal)
        else:
            policy = walkers[state.acting_player]
            request = _audit_request(
                state, policy, seed=(seed * 7 + state.total_moves) % (2**31)
            )
            action_id = policy.decide_checked(request).selected_action_id
        apply_action(state, action_id, legal=legal)
        if state.total_moves in targets and not state.terminal:
            positions.append(clone_state(state))
    return positions


# ---------------------------------------------------------------------------
# Hidden-information audit
# ---------------------------------------------------------------------------


def _empty_audit() -> dict:
    return {
        "trials": 0,
        "policy_comparisons": 0,
        "score_vector_comparisons": 0,
        "action_mismatches": 0,
        "diagnostic_mismatches": 0,
        "score_vector_mismatches": 0,
        "public_view_mismatches": 0,
        "legal_action_mismatches": 0,
        "positive_control_trials": 0,
        "positive_control_failures": 0,
        "leak_detector_failures": 0,
        "positions_skipped_unchanged": 0,
        "games_sampled": 0,
        "hidden_pieces_permuted": 0,
        "mismatch_detail": [],
        "trials_by_ply": {},
        "trials_by_source": {},
        "trials_by_policy": {},
    }


def audit_chunk(payload: Mapping[str, Any]) -> dict:
    """One deterministic slice of the hidden-information audit.

    A module-level function taking a plain mapping so it can be dispatched to a
    process pool. Everything it does follows from ``root_seed``, ``chunk_index``
    and ``trials``, so the audit total is independent of how the chunks were
    distributed.

    Per trial: clone the privileged state, permute the true identities of the
    unresolved hidden opponent pieces while preserving every public constraint,
    then ask every policy to decide on both states with the *same* policy seed
    and compare. A policy that reads only observer-legal facts cannot tell the
    two apart, so any difference is a leak.

    Two guards keep the count honest. The **positive control** requires the
    privileged belief target to actually differ, so a trial where the permutation
    did nothing cannot inflate the total. The **leak detector** requires the
    hidden true types themselves to differ, which proves the pair of states is
    one a leaking policy really could distinguish.
    """
    root_seed = int(payload["root_seed"])
    chunk_index = int(payload["chunk_index"])
    target = int(payload["trials"])
    source = str(payload["source"])
    policy_ids = tuple(payload.get("policy_ids") or ALL_POLICY_IDS)
    plies = tuple(payload.get("plies") or AUDIT_PLIES)
    decision_seed = int(payload.get("decision_seed", 20260407))
    detail_limit = int(payload.get("detail_limit", 12))

    policies = build_policies(policy_ids)
    totals = _empty_audit()
    if target <= 0:
        return totals

    rng = random.Random((root_seed * 1_000_003 + chunk_index) % (2**61))
    trials_by_ply: Counter = Counter()
    trials_by_policy: Counter = Counter()
    detail: list[str] = totals["mismatch_detail"]

    # Distinct game seeds per chunk, so two chunks never audit the same position.
    game_seed = root_seed + chunk_index * 1_000_000
    while totals["trials"] < target:
        positions = sample_positions(
            game_seed, source=source, plies=plies, policy_ids=policy_ids
        )
        game_seed += 1
        totals["games_sampled"] += 1
        if not positions:
            continue

        for state in positions:
            if totals["trials"] >= target:
                break
            observer = state.acting_player
            clone, info = permute_hidden_identities(state, observer, rng)
            if not info["valid"] or not info["changed"]:
                totals["positions_skipped_unchanged"] += 1
                continue

            totals["positive_control_trials"] += 1
            if not belief_targets_differ(state, clone, observer):
                totals["positive_control_failures"] += 1

            hidden = hidden_opponent_piece_ids(state, observer)
            if [state.pieces[pid].true_type for pid in hidden] == [
                clone.pieces[pid].true_type for pid in hidden
            ]:
                totals["leak_detector_failures"] += 1

            totals["trials"] += 1
            totals["hidden_pieces_permuted"] += int(info["hidden_pieces"])
            trials_by_ply[state.total_moves] += 1

            # Public products, compared once per trial rather than per policy.
            left_legal = legal_actions(state)
            right_legal = legal_actions(clone)
            if left_legal != right_legal:
                totals["legal_action_mismatches"] += 1
                if len(detail) < detail_limit:
                    detail.append(f"legal actions differ at ply {state.total_moves}")

            for policy in policies:
                left_request = _audit_request(state, policy, decision_seed)
                right_request = _audit_request(clone, policy, decision_seed)
                if left_request.public_view != right_request.public_view:
                    totals["public_view_mismatches"] += 1
                    if len(detail) < detail_limit:
                        detail.append(
                            f"{policy.policy_id} public view differs at ply {state.total_moves}"
                        )

                left = policy.decide_checked(left_request)
                right = policy.decide_checked(right_request)
                totals["policy_comparisons"] += 1
                trials_by_policy[policy.policy_id] += 1

                if left.selected_action_id != right.selected_action_id:
                    totals["action_mismatches"] += 1
                    if len(detail) < detail_limit:
                        detail.append(
                            f"{policy.policy_id} chose differently at ply {state.total_moves}"
                        )
                if left.diagnostics != right.diagnostics:
                    totals["diagnostic_mismatches"] += 1
                    if len(detail) < detail_limit:
                        detail.append(
                            f"{policy.policy_id} diagnostics differ at ply {state.total_moves}"
                        )

                if isinstance(policy, ScoringPolicy):
                    # Strictly stronger than comparing the argmax: two score
                    # vectors can share a maximum and differ everywhere else.
                    left_context = build_context(left_request)
                    right_context = build_context(right_request)
                    totals["score_vector_comparisons"] += 1
                    if rank_moves(
                        policy.score(left_context, move) for move in left_context.moves
                    ) != rank_moves(
                        policy.score(right_context, move) for move in right_context.moves
                    ):
                        totals["score_vector_mismatches"] += 1
                        if len(detail) < detail_limit:
                            detail.append(
                                f"{policy.policy_id} score vector differs at ply "
                                f"{state.total_moves}"
                            )

    totals["trials_by_ply"] = dict(sorted(trials_by_ply.items()))
    totals["trials_by_source"] = {source: totals["trials"]}
    totals["trials_by_policy"] = dict(sorted(trials_by_policy.items()))
    return totals


def _sort_key(name: str) -> "tuple[int, float, str]":
    """Numeric where the key is a number, alphabetical otherwise.

    `trials_by_ply` is keyed by ply and `trials_by_policy` by name, and both pass
    through JSON where every key is a string. Sorting them lexicographically would
    put ply 115 before ply 15.
    """
    return (0, float(name), "") if name.lstrip("-").isdigit() else (1, 0.0, name)


def merge_audit_results(chunks: "Sequence[Mapping[str, Any]]") -> dict:
    """Sum chunk totals into one audit report."""
    merged = _empty_audit()
    counters = {
        "trials_by_ply": Counter(),
        "trials_by_source": Counter(),
        "trials_by_policy": Counter(),
    }
    for chunk in chunks:
        for key, value in chunk.items():
            if key in counters:
                # Keys become strings because these dicts are serialised to JSON,
                # where an integer key would come back as a string anyway.
                counters[key].update({str(name): int(count) for name, count in value.items()})
            elif key == "mismatch_detail":
                merged[key].extend(value)
            else:
                merged[key] += value
    for key, counter in counters.items():
        merged[key] = dict(sorted(counter.items(), key=lambda item: _sort_key(item[0])))
    merged["mismatch_detail"] = merged["mismatch_detail"][:40]
    merged["total_mismatches"] = (
        merged["action_mismatches"]
        + merged["diagnostic_mismatches"]
        + merged["score_vector_mismatches"]
        + merged["public_view_mismatches"]
        + merged["legal_action_mismatches"]
    )
    return merged


def audit_payloads(
    target_trials: int,
    *,
    root_seed: int = 20260407,
    policy_ids: "Sequence[str] | None" = None,
    sources: "Sequence[str]" = POSITION_SOURCES,
    plies: "Sequence[int]" = AUDIT_PLIES,
    chunks: int = DEFAULT_AUDIT_CHUNKS,
) -> list[dict]:
    """The exact chunk decomposition an audit will run.

    Deliberately independent of the worker count. The audit's whole claim is that
    a policy cannot see hidden state, and that claim is worth nothing if the
    evidence changes shape depending on how many cores were free -- so the number
    of chunks is an explicit parameter and ``workers`` only decides how many run
    at once. The trial counts sum to ``target_trials`` exactly.
    """
    if target_trials <= 0:
        raise ValueError(f"target_trials must be positive, got {target_trials}")
    if chunks < 1:
        raise ValueError(f"chunks must be at least 1, got {chunks}")

    source_list = list(sources)
    if not source_list:
        raise ValueError("an audit needs at least one position source")
    chunks_per_source = max(1, chunks // len(source_list))
    per_source = target_trials // len(source_list)

    payloads: list[dict] = []
    index = 0
    for position, source in enumerate(source_list):
        share = (
            target_trials - per_source * (len(source_list) - 1)
            if position == len(source_list) - 1
            else per_source
        )
        base, extra = divmod(share, chunks_per_source)
        for slot in range(chunks_per_source):
            trials = base + (1 if slot < extra else 0)
            if trials <= 0:
                continue
            payloads.append(
                {
                    "root_seed": root_seed,
                    "chunk_index": index,
                    "trials": trials,
                    "source": source,
                    "policy_ids": list(policy_ids or ALL_POLICY_IDS),
                    "plies": list(plies),
                }
            )
            index += 1
    return payloads


def run_hidden_information_audit(
    target_trials: int,
    *,
    workers: int = 1,
    root_seed: int = 20260407,
    policy_ids: "Sequence[str] | None" = None,
    sources: "Sequence[str]" = POSITION_SOURCES,
    plies: "Sequence[int]" = AUDIT_PLIES,
    chunks: int = DEFAULT_AUDIT_CHUNKS,
) -> dict:
    """Run the audit to exactly `target_trials` valid trials.

    The merged report is a pure function of
    ``(root_seed, target_trials, sources, plies, policy_ids, chunks)``. It does
    **not** depend on ``workers``, which only sets the pool size, so a laptop and
    a build machine produce the same audit.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    payloads = audit_payloads(
        target_trials,
        root_seed=root_seed,
        policy_ids=policy_ids,
        sources=sources,
        plies=plies,
        chunks=chunks,
    )
    if workers == 1:
        results = [audit_chunk(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(audit_chunk, payloads))

    merged = merge_audit_results(results)
    merged["calibration_version"] = CALIBRATION_VERSION
    merged["root_seed"] = root_seed
    merged["chunk_count"] = len(payloads)
    merged["chunks_requested"] = chunks
    merged["sources"] = list(sources)
    merged["plies"] = list(plies)
    merged["policies_audited"] = list(policy_ids or ALL_POLICY_IDS)
    return merged


# ---------------------------------------------------------------------------
# Behavioural profiling
# ---------------------------------------------------------------------------


@dataclass
class BehaviorCounters:
    """Raw per-policy counters accumulated over replayed games."""

    counts: Counter = field(default_factory=Counter)

    def merge(self, other: "BehaviorCounters | Counter | Mapping[str, int]") -> None:
        source = other.counts if isinstance(other, BehaviorCounters) else other
        self.counts.update(source)


def _direction_bucket(source: int, destination: int, player: int) -> str:
    """Own-relative direction and length class of one move."""
    row_delta = square_row(destination) - square_row(source)
    column_delta = square_column(destination) - square_column(source)
    distance = abs(row_delta) + abs(column_delta)
    length = "run" if distance > 1 else "step"
    if row_delta == 0:
        return f"lateral_{length}"
    # Red's home rows are the low row indices, so red advances as the row index
    # grows and blue advances as it shrinks.
    forward = row_delta > 0 if player == RED else row_delta < 0
    return f"{'forward' if forward else 'backward'}_{length}"


def profile_replay(result: "MatchResult") -> dict[str, Counter]:
    """Behavioural counters for both players of one stored match.

    Replays the row's own action history through the engine, so the profile
    describes precisely the game the league played rather than a re-simulation.
    No policy is consulted. The privileged state is read only for the mover's own
    piece type and, once the game is over, for which pieces the opponent legally
    learned -- both of which are facts *about* the finished game, not inputs to a
    decision.

    Returns one counter per policy token.
    """
    if result.action_history is None:
        raise ValueError(f"match {result.match_id} has no stored action history to profile")
    if result.errored:
        raise ValueError(f"match {result.match_id} errored and has no behaviour to profile")

    tokens = {
        result.candidate_color: result.candidate.token,
        1 - result.candidate_color: result.opponent.token,
    }
    counters: dict[str, Counter] = {token: Counter() for token in tokens.values()}

    state = create_game(
        deserialize_setup(result.red_setup),
        deserialize_setup(result.blue_setup),
        rules=result.rules_config(),
        game_id=result.match_id,
    )
    for action_id in result.action_history:
        player = state.acting_player
        counter = counters[tokens[player]]
        source, destination = decode_action(action_id)
        mover = state.piece_at(source)
        target = state.piece_at(destination)
        distance = abs(square_row(destination) - square_row(source)) + abs(
            square_column(destination) - square_column(source)
        )
        counter["moves"] += 1
        counter[f"piece_{PIECE_TYPE_NAMES[mover.true_type]}"] += 1
        counter[f"direction_{_direction_bucket(source, destination, player)}"] += 1
        counter["distance"] += distance
        if target is not None:
            counter["attacks"] += 1
            if mover.true_type == MINER:
                counter["miner_attacks"] += 1
        if mover.true_type == SCOUT and distance > 1:
            counter["scout_runs"] += 1
        apply_action(state, action_id)

    for player, token in tokens.items():
        counter = counters[token]
        counter["games"] += 1
        counter["plies"] += state.total_moves
        counter[f"terminal_{state.terminal_reason}"] += 1
        if state.terminal_reason in DRAW_TERMINAL_REASONS:
            counter["draws"] += 1
        elif state.winner == player:
            counter["wins"] += 1
        else:
            counter["losses"] += 1
        if state.terminal_reason == TERMINAL_FLAG_CAPTURE and state.winner == player:
            counter["flag_captures"] += 1
        counter["own_pieces_revealed"] += sum(
            1 for record in state.pieces_of(player) if record.known_to(1 - player)
        )
    return counters


def entropy_bits(counts: "Sequence[int]") -> float:
    """Shannon entropy in bits of a count distribution."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count <= 0:
            continue
        share = count / total
        value -= share * math.log2(share)
    return value


def summarise_behavior(counter: Mapping[str, int]) -> dict:
    """Turn raw counters into the rates the stress characterisation reports.

    Metric definitions match Agent 2's ``summarise_profile`` so the two tables
    are directly comparable, plus a movement-entropy term over own-relative
    direction and length buckets.
    """
    moves = max(counter.get("moves", 0), 1)
    games = max(counter.get("games", 0), 1)
    piece_counts = [counter.get(f"piece_{name}", 0) for name in PIECE_TYPE_NAMES]
    direction_counts = [counter.get(f"direction_{label}", 0) for label in DIRECTION_LABELS]
    return {
        "games": counter.get("games", 0),
        "moves": counter.get("moves", 0),
        "mean_game_plies": counter.get("plies", 0) / games,
        "attack_rate": counter.get("attacks", 0) / moves,
        "scout_move_rate": counter.get("piece_scout", 0) / moves,
        "scout_run_rate": counter.get("scout_runs", 0) / moves,
        "miner_move_rate": counter.get("piece_miner", 0) / moves,
        "miner_attack_rate": counter.get("miner_attacks", 0) / moves,
        "mean_move_distance": counter.get("distance", 0) / moves,
        "piece_type_entropy_bits": entropy_bits(piece_counts),
        "movement_entropy_bits": entropy_bits(direction_counts),
        "own_reveal_rate": counter.get("own_pieces_revealed", 0) / (games * PIECES_PER_PLAYER),
        "draw_rate": counter.get("draws", 0) / games,
        "battleless_draw_rate": counter.get(f"terminal_{TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW}", 0)
        / games,
        "flag_capture_win_rate": counter.get("flag_captures", 0) / games,
        "effective_win_rate": (counter.get("wins", 0) + 0.5 * counter.get("draws", 0)) / games,
        "wins": counter.get("wins", 0),
        "draws": counter.get("draws", 0),
        "losses": counter.get("losses", 0),
        "direction_shares": {
            label: counter.get(f"direction_{label}", 0) / moves for label in DIRECTION_LABELS
        },
    }


#: Metrics the stress characterisation compares against the Strategic baseline.
STRESS_COMPARISON_METRICS: tuple[str, ...] = (
    "attack_rate",
    "scout_move_rate",
    "miner_move_rate",
    "mean_game_plies",
    "battleless_draw_rate",
    "flag_capture_win_rate",
    "own_reveal_rate",
    "piece_type_entropy_bits",
    "movement_entropy_bits",
)


def behavior_divergence(
    profile: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    threshold: float = 0.35,
) -> dict:
    """How far one policy's behaviour sits from the Strategic baseline's.

    Reported as a relative difference per metric, using the symmetric
    denominator ``max(|a|, |b|, eps)`` so a metric that is zero for one policy
    and positive for the other reads as a full separation rather than dividing by
    zero. ``threshold`` is the relative gap counted as "materially different";
    0.35 is a deliberately blunt line, and the raw numbers are always reported
    beside the verdict.
    """
    differences: dict[str, float] = {}
    for metric in STRESS_COMPARISON_METRICS:
        mine = float(profile.get(metric, 0.0))
        theirs = float(reference.get(metric, 0.0))
        scale = max(abs(mine), abs(theirs), 1e-9)
        differences[metric] = (mine - theirs) / scale
    separated = sorted(
        metric for metric, value in differences.items() if abs(value) >= threshold
    )
    return {
        "relative_differences": differences,
        "metrics_beyond_threshold": separated,
        "threshold": threshold,
        "materially_different": bool(separated),
        "largest_metric": max(differences, key=lambda key: abs(differences[key])),
        "largest_relative_difference": max(abs(value) for value in differences.values()),
    }


# ---------------------------------------------------------------------------
# Strength tiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrengthTier:
    """One rung of the calibrated ladder."""

    rank: int
    members: tuple[str, ...]
    pooled_effective_win_rate: float

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "members": list(self.members),
            "pooled_effective_win_rate": self.pooled_effective_win_rate,
        }


def _direct_result(
    summaries: Mapping[str, Mapping[str, Any]], first: str, second: str
) -> "tuple[float, bool] | None":
    """`first`'s effective win rate against `second`, and whether it separates.

    A matchup is stored under one ordering only, so the reversed orientation is
    recovered by complementing the rate and mirroring the interval.
    """
    forward = summaries.get(f"{first} vs {second}")
    if forward is not None:
        interval = forward["confidence_interval"]
        return float(forward["effective_win_rate"]), bool(
            interval["lower"] > 0.5 or interval["upper"] < 0.5
        )
    backward = summaries.get(f"{second} vs {first}")
    if backward is not None:
        interval = backward["confidence_interval"]
        return 1.0 - float(backward["effective_win_rate"]), bool(
            interval["lower"] > 0.5 or interval["upper"] < 0.5
        )
    return None


def strength_tiers(
    policy_tokens: "Sequence[str]",
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Partition policies into statistically distinguishable strength tiers.

    Two policies share a tier when their direct paired comparison does **not**
    separate them -- the interval on their head-to-head effective win rate
    contains 0.5. A new tier opens at the first policy that is separated from
    every member of the tier being built. The result therefore answers the Phase 4
    gate directly: the tier count is the number of levels the evidence supports,
    not a count of policies.

    Ordering is by pooled effective win rate across the policies considered,
    with the policy token as a tiebreak so the partition is deterministic.
    ``fully_ordered`` reports whether every cross-tier pair separates, which is
    the stronger claim that the tiers are a genuine ladder rather than merely a
    chain of adjacent separations.
    """
    tokens = list(policy_tokens)
    pooled: dict[str, float] = {}
    for token in tokens:
        rates = [
            _direct_result(summaries, token, other)[0]
            for other in tokens
            if other != token and _direct_result(summaries, token, other) is not None
        ]
        pooled[token] = sum(rates) / len(rates) if rates else 0.0

    ordered = sorted(tokens, key=lambda token: (-pooled[token], token))
    tiers: list[list[str]] = []
    missing: list[str] = []
    for token in ordered:
        if not tiers:
            tiers.append([token])
            continue
        current = tiers[-1]
        separated_from_all = True
        for member in current:
            direct = _direct_result(summaries, member, token)
            if direct is None:
                missing.append(f"{member} vs {token}")
                separated_from_all = False
                break
            # Separated *above*, not merely separated: a token that separates in
            # the other direction is not a weaker tier, it is a non-transitivity,
            # and it belongs in the same tier so `fully_ordered` reports it.
            if not (direct[1] and direct[0] > 0.5):
                separated_from_all = False
                break
        if separated_from_all:
            tiers.append([token])
        else:
            current.append(token)

    built = tuple(
        StrengthTier(
            rank=index + 1,
            members=tuple(sorted(members)),
            pooled_effective_win_rate=sum(pooled[m] for m in members) / len(members),
        )
        for index, members in enumerate(tiers)
    )

    cross_pairs = 0
    cross_separated = 0
    unseparated: list[str] = []
    for i, upper in enumerate(built):
        for lower in built[i + 1 :]:
            for stronger in upper.members:
                for weaker in lower.members:
                    direct = _direct_result(summaries, stronger, weaker)
                    cross_pairs += 1
                    if direct is not None and direct[1] and direct[0] > 0.5:
                        cross_separated += 1
                    else:
                        unseparated.append(f"{stronger} vs {weaker}")

    return {
        "calibration_version": CALIBRATION_VERSION,
        "tier_count": len(built),
        "tiers": [tier.to_dict() for tier in built],
        "membership": {
            token: index + 1
            for index, tier in enumerate(built)
            for token in tier.members
        },
        "pooled_effective_win_rates": {token: pooled[token] for token in ordered},
        "cross_tier_pairs": cross_pairs,
        "cross_tier_pairs_separated": cross_separated,
        "fully_ordered": cross_pairs == cross_separated,
        "unseparated_cross_tier_pairs": unseparated,
        "missing_comparisons": sorted(set(missing)),
    }


__all__ = [
    "AUDIT_PLIES",
    "CALIBRATION_VERSION",
    "DEFAULT_AUDIT_CHUNKS",
    "DIRECTION_LABELS",
    "POSITION_SOURCES",
    "STRESS_COMPARISON_METRICS",
    "BehaviorCounters",
    "StrengthTier",
    "audit_chunk",
    "audit_payloads",
    "behavior_divergence",
    "entropy_bits",
    "merge_audit_results",
    "profile_replay",
    "run_hidden_information_audit",
    "sample_positions",
    "strength_tiers",
    "summarise_behavior",
]
