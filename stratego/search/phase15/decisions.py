"""Phase 15 Agent 2 section 11: Stage A, the quick decision diagnostic.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 10, 11.

The question Stage A answers
----------------------------
Before spending hours of match compute, find out whether belief-guided search
*changes decisions at all*, and whether the learned belief tracks the oracle
ceiling. Section 11's reading is a three-way fork:

```text
oracle helps, learned belief does not   -> belief/provider quality is limiting
oracle also fails to change decisions   -> search mechanics or value is limiting
learned belief tracks the oracle        -> proceed to the match comparison
```

Every arm decides on the *same* replayed positions with the same seed, so a
difference between two arms is a difference between two decision procedures
rather than between two samples.

What "agreement with oracle" measures
--------------------------------------
The oracle is run on the same position with the same budget and the same
seed, and its selected action is the reference. A production arm agreeing
with it more often than the direct model does is the arm recovering some of
the information a perfect belief would give. That is a diagnostic quantity,
not a strength claim: the oracle plays a game it is not allowed to play.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ...engine.legal_moves import legal_actions
from .contract import (
    DIAGNOSTIC_PAIRING_IDS,
    MOVE_MODELS,
    PRODUCTION_PAIRING_IDS,
    Phase15SearchError,
    pairing as pairing_of,
)

#: The decision-diagnostic identity.
DECISION_VERSION = "phase15_decision_diagnostic_v1"

#: The seed every arm uses on every position, so the arms differ only in the
#: procedure and never in the randomness.
DECISION_SEED = 20260824


class Phase15DecisionError(Phase15SearchError):
    """A Stage A diagnostic could not be run."""


@dataclass(frozen=True)
class DecisionRow:
    """One arm's decision on one position."""

    position_id: str
    arm_id: str
    move_model: str
    provider: "str | None"
    preset_id: str
    ply: int
    unresolved: int
    legal_actions: int
    action_id: int
    direct_action_id: int
    move_changed: bool
    matches_oracle: "bool | None"
    seconds: float
    c1_forwards: int
    unique_worlds: "int | None"
    worlds_requested: "int | None"
    score_margin: "float | None"
    legal: bool

    def row(self) -> dict:
        return {
            "position_id": self.position_id,
            "arm_id": self.arm_id,
            "move_model": self.move_model,
            "provider": self.provider,
            "preset_id": self.preset_id,
            "ply": self.ply,
            "unresolved": self.unresolved,
            "legal_actions": self.legal_actions,
            "action_id": self.action_id,
            "direct_action_id": self.direct_action_id,
            "move_changed": int(self.move_changed),
            "matches_oracle": (
                "" if self.matches_oracle is None else int(self.matches_oracle)
            ),
            "seconds": round(self.seconds, 5),
            "c1_forwards": self.c1_forwards,
            "unique_worlds": "" if self.unique_worlds is None else self.unique_worlds,
            "worlds_requested": (
                "" if self.worlds_requested is None else self.worlds_requested
            ),
            "score_margin": (
                "" if self.score_margin is None else round(self.score_margin, 6)
            ),
            "legal": int(self.legal),
        }


def _margin(decision) -> "float | None":
    scores = sorted((candidate.score for candidate in decision.candidates), reverse=True)
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def run_decisions(
    models,
    replayed,
    *,
    preset: str = "TINY",
    arms=None,
    include_oracle: bool = True,
    seed: int = DECISION_SEED,
    progress=None,
) -> "list[DecisionRow]":
    """Every arm on every replayed position. `replayed` is `(row, state, plan)`.

    The oracle arms are built with `production=False` and run last, so their
    answers are available as the agreement reference for the production arms
    of the same move model.
    """
    from .player import Phase15SearchPlayer
    from .systems import build_engine

    arm_ids = list(PRODUCTION_PAIRING_IDS if arms is None else arms)
    oracle_ids = list(DIAGNOSTIC_PAIRING_IDS) if include_oracle else []

    engines = {
        arm_id: build_engine(arm_id, models, preset)
        for arm_id in arm_ids
        if pairing_of(arm_id).kind != "direct"
    }
    for arm_id in oracle_ids:
        engines[arm_id] = build_engine(arm_id, models, preset, production=False)
    direct_player = Phase15SearchPlayer(
        {"p18_b18": build_engine("p18_b18", models, preset)}, models, mode="p18_direct"
    )

    rows: list[DecisionRow] = []
    for index, (position, state, plan) in enumerate(replayed):
        legal = set(legal_actions(state))
        # 1. the oracle references, one per move model
        oracle_action = {}
        for arm_id in oracle_ids:
            target = pairing_of(arm_id)
            started = time.perf_counter()
            decision = engines[arm_id].engine.choose_action(state, seed=seed)
            elapsed = time.perf_counter() - started
            oracle_action[target.move_model] = int(decision.selected_action_id)
            rows.append(
                DecisionRow(
                    position_id=position["position_id"],
                    arm_id=arm_id,
                    move_model=target.move_model,
                    provider=target.provider,
                    preset_id=preset,
                    ply=int(position["ply"]),
                    unresolved=int(position["unresolved"]),
                    legal_actions=len(legal),
                    action_id=int(decision.selected_action_id),
                    direct_action_id=int(decision.direct_action_id),
                    move_changed=bool(decision.move_changed),
                    matches_oracle=True,
                    seconds=elapsed,
                    c1_forwards=int(decision.c1_forwards),
                    unique_worlds=int(decision.unique_worlds),
                    worlds_requested=int(decision.worlds_requested),
                    score_margin=_margin(decision),
                    legal=int(decision.selected_action_id) in legal,
                )
            )
        # 2. every production arm
        for arm_id in arm_ids:
            target = pairing_of(arm_id)
            reference = oracle_action.get(target.move_model)
            if target.kind == "direct":
                started = time.perf_counter()
                action = direct_player.direct_action(
                    state, sorted(legal), move_model=target.move_model
                )
                elapsed = time.perf_counter() - started
                rows.append(
                    DecisionRow(
                        position_id=position["position_id"],
                        arm_id=arm_id,
                        move_model=target.move_model,
                        provider=None,
                        preset_id="direct",
                        ply=int(position["ply"]),
                        unresolved=int(position["unresolved"]),
                        legal_actions=len(legal),
                        action_id=int(action),
                        direct_action_id=int(action),
                        move_changed=False,
                        matches_oracle=(
                            None if reference is None else int(action) == reference
                        ),
                        seconds=elapsed,
                        c1_forwards=1,
                        unique_worlds=None,
                        worlds_requested=None,
                        score_margin=None,
                        legal=int(action) in legal,
                    )
                )
                continue
            started = time.perf_counter()
            decision = engines[arm_id].engine.choose_action(state, seed=seed)
            elapsed = time.perf_counter() - started
            selected = int(decision.selected_action_id)
            rows.append(
                DecisionRow(
                    position_id=position["position_id"],
                    arm_id=arm_id,
                    move_model=target.move_model,
                    provider=target.provider,
                    preset_id=preset,
                    ply=int(position["ply"]),
                    unresolved=int(position["unresolved"]),
                    legal_actions=len(legal),
                    action_id=selected,
                    direct_action_id=int(decision.direct_action_id),
                    move_changed=bool(decision.move_changed),
                    matches_oracle=None if reference is None else selected == reference,
                    seconds=elapsed,
                    c1_forwards=int(decision.c1_forwards),
                    unique_worlds=int(decision.unique_worlds),
                    worlds_requested=int(decision.worlds_requested),
                    score_margin=_margin(decision),
                    legal=selected in legal,
                )
            )
        if progress is not None:
            progress(index + 1, len(replayed), len(rows))
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(rows: "list[DecisionRow]") -> dict:
    """Section 11's table, one entry per arm."""
    by_arm: dict[str, list[DecisionRow]] = {}
    for row in rows:
        by_arm.setdefault(row.arm_id, []).append(row)
    report = {}
    for arm_id, entries in by_arm.items():
        seconds = np.asarray([entry.seconds for entry in entries], dtype=np.float64)
        changed = [entry.move_changed for entry in entries]
        agreement = [
            entry.matches_oracle for entry in entries if entry.matches_oracle is not None
        ]
        margins = [
            entry.score_margin for entry in entries if entry.score_margin is not None
        ]
        unique = [
            entry.unique_worlds for entry in entries if entry.unique_worlds is not None
        ]
        requested = [
            entry.worlds_requested
            for entry in entries
            if entry.worlds_requested is not None
        ]
        report[arm_id] = {
            "decisions": len(entries),
            "move_change_rate_vs_direct": round(float(np.mean(changed)), 5),
            "oracle_agreement": (
                round(float(np.mean(agreement)), 5) if agreement else None
            ),
            "oracle_agreement_n": len(agreement),
            "legal_decision_rate": round(
                float(np.mean([entry.legal for entry in entries])), 5
            ),
            "mean_seconds": round(float(seconds.mean()), 5),
            "median_seconds": round(float(np.median(seconds)), 5),
            "p95_seconds": round(float(np.percentile(seconds, 95)), 5),
            "max_seconds": round(float(seconds.max()), 5),
            "mean_c1_forwards": round(
                float(np.mean([entry.c1_forwards for entry in entries])), 1
            ),
            "mean_unique_worlds": round(float(np.mean(unique)), 3) if unique else None,
            "world_uniqueness": (
                round(float(np.sum(unique) / np.sum(requested)), 5)
                if unique and np.sum(requested)
                else None
            ),
            "mean_score_margin": round(float(np.mean(margins)), 6) if margins else None,
            "median_score_margin": (
                round(float(np.median(margins)), 6) if margins else None
            ),
        }
    return report


def interpret(summary: dict) -> dict:
    """Section 11's three-way reading, stated per move model."""
    readings = {}
    for move_model in MOVE_MODELS:
        oracle = summary.get(f"{move_model}_oracle") or {}
        count = summary.get(f"{move_model}_remaining_count") or {}
        learned = {
            name: summary[f"{move_model}_{name}"]
            for name in ("b18", "b24")
            if f"{move_model}_{name}" in summary
        }
        oracle_change = oracle.get("move_change_rate_vs_direct")
        learned_change = (
            max(entry["move_change_rate_vs_direct"] for entry in learned.values())
            if learned
            else None
        )
        best_agreement = (
            max(
                entry["oracle_agreement"]
                for entry in learned.values()
                if entry["oracle_agreement"] is not None
            )
            if learned
            else None
        )
        count_agreement = count.get("oracle_agreement")
        if oracle_change is not None and oracle_change < 0.02:
            reading = "search_mechanics_or_value_limiting"
            note = (
                "even a perfect belief barely changes the decision, so the limit is "
                "the search design or the leaf value, not belief quality"
            )
        elif learned_change is not None and learned_change < 0.5 * oracle_change:
            reading = "belief_quality_limiting"
            note = (
                "the oracle changes decisions but the learned belief changes far "
                "fewer, so the provider is the binding constraint"
            )
        else:
            reading = "learned_belief_tracks_oracle"
            note = "the learned belief changes decisions at a rate comparable to the oracle"
        readings[move_model] = {
            "reading": reading,
            "note": note,
            "oracle_move_change_rate": oracle_change,
            "best_learned_move_change_rate": learned_change,
            "count_move_change_rate": count.get("move_change_rate_vs_direct"),
            "best_learned_oracle_agreement": best_agreement,
            "count_oracle_agreement": count_agreement,
            "direct_oracle_agreement": (
                summary.get(f"{move_model}_direct") or {}
            ).get("oracle_agreement"),
        }
    return readings


__all__ = [
    "DECISION_SEED",
    "DECISION_VERSION",
    "DecisionRow",
    "Phase15DecisionError",
    "interpret",
    "run_decisions",
    "summarize",
]
