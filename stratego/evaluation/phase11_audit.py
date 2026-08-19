"""Phase 11 Agent 2: independent recomputation and the negative controls.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` sections 3, 7 and 8

An audit that shares an implementation audits nothing
-----------------------------------------------------
Three layers, each deliberately unlike the one it checks.

1. **Independent formulas, 100% of events.** :func:`independent_scores`
   recomputes CE, top-1, Brier, entropy and the true-rank probability from
   the stored primitives using different arithmetic from the evaluator's:
   the Brier score comes from the algebraic identity rather than the
   explicit twelve-term sum, top-1 from an explicit maximum scan rather
   than `argmax`, the softmax from a plain exponential rather than the
   max-shifted one. Agreement to float64 tolerance is then evidence and
   not a tautology.

2. **A pure-Python scalar path, on a deterministic sample.**
   :func:`scalar_recompute` walks the logical records with `math` and
   lists, no NumPy anywhere, and rebuilds the case aggregates and the
   pooled ECE from them.

3. **The engine's own counts.** :func:`baseline_edge_case_audit` replays
   games and checks `remaining_count_belief_v1` against
   `PublicView.unresolved_opponent_counts` — the accepted Phase 4
   implementation of the same public quantity, written long before Phase
   11 and by a different route — while classifying the edge cases the
   instruction names: moved unknowns, revealed ranks, captures, near
   exhaustion, one-legal-rank pieces and public Scout deductions.

Negative controls
-----------------
:func:`run_negative_controls` breaks the pipeline in each of the six ways
the instruction lists and requires each break to be *detected*. A control
that does not fire is a finding, because it means the corresponding real
mistake would also pass unnoticed.
"""

from __future__ import annotations

import math

import numpy as np

from ..engine.constants import PLAYER_NAMES
from ..engine.legal_moves import legal_actions
from ..engine.observation import build_observation
from ..engine.state import create_game
from ..engine.transition import apply_action
from ..training.phase11_contract import (
    LOG_PROBABILITY_FLOOR,
    Phase11ContractError,
    RANK_COUNT,
    RANK_INITIAL_COUNTS,
)
from .match_spec import EVALUATION_RULES
from .phase11_baselines import (
    Phase11BaselineError,
    check_count_conservation,
    remaining_count_belief,
    remaining_count_distribution,
    remaining_counts,
)
from .phase11_belief import Phase11BeliefError, Phase11BeliefRequest, softmax_float64
from .phase11_evaluator import expected_calibration_error
from .phase11_public_state import (
    build_public_state_document,
    hidden_opponent_pieces,
    legal_rank_mask,
    public_state_identity,
)
from .policy import build_public_view

#: The audit identity recorded on the artifact.
AUDIT_VERSION = "phase11_agent02_audit_v1"

#: The six negative controls the instruction requires, in its order.
NEGATIVE_CONTROLS = (
    "reversed_rank_mapping",
    "wrong_true_rank_label",
    "wrong_remaining_inventory",
    "known_pieces_in_hidden_denominator",
    "hidden_truth_injected_into_request",
    "permuted_probability_columns",
)

#: The baseline edge cases section 3 requires to be exercised.
BASELINE_EDGE_CASES = (
    "moved_unknown",
    "revealed_rank",
    "capture",
    "near_endgame_exhaustion",
    "single_legal_rank",
    "public_scout_deduction",
)

#: Agreement tolerance between two float64 formulations of one metric.
AUDIT_TOLERANCE = 1e-9


class Phase11AuditError(Phase11ContractError):
    """An audit could not be run."""


# ---------------------------------------------------------------------------
# Layer 1 — independent formulas, every event
# ---------------------------------------------------------------------------


def independent_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax without the max shift: a different float path, same limit."""
    raw = np.exp(np.asarray(logits, dtype=np.float64))
    return raw / raw.sum(axis=-1, keepdims=True)


def independent_scores(probabilities: np.ndarray, true_rank: np.ndarray) -> dict:
    """CE, top-1, Brier, entropy and true-rank probability, differently."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    true_rank = np.asarray(true_rank, dtype=np.int64)
    rows = np.arange(probabilities.shape[0])
    true_probability = probabilities[rows, true_rank]
    ce = np.negative(np.log(np.where(
        true_probability > LOG_PROBABILITY_FLOOR, true_probability, LOG_PROBABILITY_FLOOR
    )))
    # Explicit maximum scan rather than argmax, first occurrence wins ties.
    best = probabilities[:, 0].copy()
    best_index = np.zeros(probabilities.shape[0], dtype=np.int64)
    for rank in range(1, RANK_COUNT):
        better = probabilities[:, rank] > best
        best = np.where(better, probabilities[:, rank], best)
        best_index = np.where(better, rank, best_index)
    top1 = (best_index == true_rank).astype(np.float64)
    # The algebraic identity, not the twelve-term sum.
    brier = (probabilities * probabilities).sum(axis=1) - 2.0 * true_probability + 1.0
    safe = np.where(probabilities > 0.0, probabilities, 1.0)
    entropy = -(probabilities * np.log(safe)).sum(axis=1)
    return {
        "ce": ce,
        "top1": top1,
        "brier": brier,
        "entropy": entropy,
        "true_rank_probability": true_probability,
        "confidence": best,
    }


def compare_scores(primary: dict, audit: dict, names=None) -> dict:
    """Max absolute deviation per metric between the two implementations."""
    names = names or ("ce", "top1", "brier", "entropy", "true_rank_probability", "confidence")
    deviations = {}
    for name in names:
        left = np.asarray(primary[name], dtype=np.float64)
        right = np.asarray(audit[name], dtype=np.float64)
        deviations[name] = float(np.abs(left - right).max()) if left.size else 0.0
    return {
        "max_deviation": deviations,
        "tolerance": AUDIT_TOLERANCE,
        "within_tolerance": all(
            value <= AUDIT_TOLERANCE for value in deviations.values()
        ),
    }


# ---------------------------------------------------------------------------
# Layer 2 — the pure-Python scalar path
# ---------------------------------------------------------------------------


def scalar_event_metrics(probabilities: "list[float]", true_rank: int) -> dict:
    """One event, in plain Python. No NumPy in this function, deliberately."""
    if len(probabilities) != RANK_COUNT:
        raise Phase11AuditError(f"a probability row has length {len(probabilities)}")
    true_probability = probabilities[true_rank]
    ce = -math.log(max(true_probability, LOG_PROBABILITY_FLOOR))
    best = probabilities[0]
    best_index = 0
    for rank in range(1, RANK_COUNT):
        if probabilities[rank] > best:
            best = probabilities[rank]
            best_index = rank
    brier = 0.0
    entropy = 0.0
    for rank in range(RANK_COUNT):
        target = 1.0 if rank == true_rank else 0.0
        brier += (probabilities[rank] - target) ** 2
        if probabilities[rank] > 0.0:
            entropy -= probabilities[rank] * math.log(probabilities[rank])
    return {
        "ce": ce,
        "top1": 1.0 if best_index == true_rank else 0.0,
        "brier": brier,
        "entropy": entropy,
        "true_rank_probability": true_probability,
        "confidence": best,
    }


def scalar_recompute(records) -> dict:
    """Case aggregates and pooled ECE from the logical records, scalar-only."""
    per_case: dict[str, dict] = {}
    confidence: list[float] = []
    correct: list[float] = []
    events = 0
    for record in records:
        true_rank = record["true_rank_index"]
        if true_rank is None:
            raise Phase11AuditError("the scalar audit needs scored records")
        learned = scalar_event_metrics(record["learned_probabilities"], int(true_rank))
        baseline = scalar_event_metrics(record["baseline_probabilities"], int(true_rank))
        bucket = per_case.setdefault(
            record["case_id"],
            {"events": 0, **{f"{name}_{side}": 0.0
                             for side in ("learned", "baseline")
                             for name in ("ce", "top1", "brier", "entropy")}},
        )
        bucket["events"] += 1
        for name in ("ce", "top1", "brier", "entropy"):
            bucket[f"{name}_learned"] += learned[name]
            bucket[f"{name}_baseline"] += baseline[name]
        confidence.append(learned["confidence"])
        correct.append(learned["top1"])
        events += 1

    aggregates = {}
    for case_id, bucket in per_case.items():
        count = bucket["events"]
        aggregates[case_id] = {
            name: bucket[name] / count
            for name in bucket
            if name != "events"
        }
    overall = {}
    if aggregates:
        for name in next(iter(aggregates.values())):
            overall[name] = sum(row[name] for row in aggregates.values()) / len(aggregates)
        overall["r_ce"] = overall["ce_learned"] / overall["ce_baseline"]
        overall["top1_delta"] = overall["top1_learned"] - overall["top1_baseline"]
        overall["brier_delta"] = overall["brier_learned"] - overall["brier_baseline"]
        overall["ce_delta"] = overall["ce_learned"] - overall["ce_baseline"]
    return {
        "events": events,
        "cases": len(aggregates),
        "case_aggregates": aggregates,
        "overall": overall,
        "ece_learned": expected_calibration_error(
            np.asarray(confidence), np.asarray(correct)
        )["ece"],
    }


# ---------------------------------------------------------------------------
# Layer 3 — the engine's own counts, and the edge cases
# ---------------------------------------------------------------------------


def replay_documents(plan, action_history):
    """Yield `(state, document)` at every observer decision of one game.

    The single replay primitive the three audits share. `action_history` is
    the game's public move list, stored in its own shard, so an audit needs
    no match runner and no policy — only the engine.
    """
    observer = 0 if plan.observer_color == "red" else 1
    state = create_game(
        plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id=plan.game_id
    )
    for action in action_history:
        if state.terminal:  # pragma: no cover - the history stops at terminal
            break
        if state.acting_player == observer:
            view = build_public_view(state, observer)
            document = build_public_state_document(
                view, build_observation(state, observer)
            )
            yield state, view, document
        apply_action(state, int(action))


def baseline_edge_case_audit(replays, *, collect_reveal_document: bool = True) -> dict:
    """Replay games and check the baseline against the engine's own counts.

    `replays` is an iterable of `(plan, action_history)`. For every observer
    decision the audit rebuilds the public document, compares
    `remaining_count_belief_v1`'s inventory against
    `PublicView.unresolved_opponent_counts`, re-derives every mask, and
    classifies which of the frozen edge cases the position exercises.
    """
    seen = {name: 0 for name in BASELINE_EDGE_CASES}
    findings: list[str] = []
    decisions = 0
    pieces = 0
    count_mismatches = 0
    mask_mismatches = 0
    conservation_failures = 0
    zero_true_probability = 0
    distribution_mismatches = 0
    reveal_document = None
    reveal_observation = None

    for plan, action_history in replays:
        observer = 0 if plan.observer_color == "red" else 1
        for state, view, document in replay_documents(plan, action_history):
            decisions += 1
            counts = remaining_counts(document)
            if tuple(counts) != tuple(
                int(value) for value in view.unresolved_opponent_counts
            ):
                count_mismatches += 1
                findings.append(
                    f"{plan.game_id}@{state.total_moves}: inventory disagrees with "
                    "PublicView.unresolved_opponent_counts"
                )
            if not check_count_conservation(document, counts)["conserved"]:
                conservation_failures += 1

            if any(count == 0 for count in counts):
                seen["near_endgame_exhaustion"] += 1
            revealed = [
                piece
                for piece in document["pieces"]
                if piece["owner_color"] != document["observer_color"]
                and piece["known_to_observer"]
            ]
            if revealed:
                seen["revealed_rank"] += 1
            # The reveal document exists for the "known pieces in the hidden
            # denominator" control, which needs a *live* known opponent piece:
            # a captured one is already outside every denominator.
            if (
                collect_reveal_document
                and reveal_document is None
                and any(piece["alive"] for piece in revealed)
            ):
                reveal_document = document
                reveal_observation = build_observation(state, observer)
            if any(not piece["alive"] for piece in document["pieces"]):
                seen["capture"] += 1
            for record in state.pieces:
                reason = (
                    record.reveal_reason_red if observer == 0 else record.reveal_reason_blue
                )
                if reason == "scout_multisquare":
                    seen["public_scout_deduction"] += 1
                    break

            distributions = remaining_count_belief(document)
            for piece in hidden_opponent_pieces(document):
                slot = int(piece["piece_slot"])
                pieces += 1
                mask = legal_rank_mask(bool(piece["has_moved"]))
                if piece["has_moved"]:
                    seen["moved_unknown"] += 1
                    if mask[10] or mask[11]:
                        mask_mismatches += 1
                elif mask != (1,) * RANK_COUNT:
                    mask_mismatches += 1
                legal = [
                    rank for rank in range(RANK_COUNT) if mask[rank] and counts[rank] > 0
                ]
                if len(legal) == 1:
                    seen["single_legal_rank"] += 1
                expected = remaining_count_distribution(counts, mask)
                if float(np.abs(distributions[slot] - expected).max()) > 0.0:
                    distribution_mismatches += 1
                true_type = int(state.pieces[_opponent_piece_id(observer, slot)].true_type)
                if distributions[slot][true_type] <= 0.0:
                    zero_true_probability += 1
                    findings.append(
                        f"{plan.game_id}@{state.total_moves}: the baseline gave the "
                        f"true rank zero mass on slot {slot}"
                    )

    # Edge-case *coverage* is reported, not a correctness verdict: whether
    # the frozen bank happens to reach a one-legal-rank position is a fact
    # about the bank, and the deterministic construction of each edge case
    # lives in the suite (`tests/evaluation/test_phase11_baselines.py`),
    # where it cannot depend on what the games did.
    missing = [name for name, count in seen.items() if count == 0]
    return {
        "audit_version": AUDIT_VERSION,
        "decisions": decisions,
        "hidden_pieces": pieces,
        "edge_cases_seen": seen,
        "edge_cases_missing": missing,
        "count_mismatches": count_mismatches,
        "mask_mismatches": mask_mismatches,
        "conservation_failures": conservation_failures,
        "distribution_mismatches": distribution_mismatches,
        "baseline_zero_on_true_rank": zero_true_probability,
        "reveal_document": reveal_document,
        "reveal_observation": reveal_observation,
        "edge_case_coverage_complete": not missing,
        "findings": findings,
        "pass": not findings
        and count_mismatches == 0
        and mask_mismatches == 0
        and conservation_failures == 0
        and distribution_mismatches == 0
        and zero_true_probability == 0,
    }


def world_baseline_audit(replays, *, worlds_per_state: int = 8, max_states: int = 512) -> dict:
    """Sample and validate `count_uniform_world_sampler_v1` worlds.

    The structural fallback baseline has to produce *complete legal* worlds,
    so every sample is checked against the frozen validation stack and every
    zero-tolerance counter is required to stay at zero. This is Agent 2's
    baseline obligation, not Agent 3's large audit.
    """
    from .phase11_baselines import (
        COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
        WORLD_COUNTER_NAMES,
        sample_world,
        validate_world,
    )

    counters = {name: 0 for name in WORLD_COUNTER_NAMES}
    findings: list[str] = []
    states = 0
    worlds = 0
    distinct_by_state: list[int] = []
    truth_recovered = 0
    for plan, action_history in replays:
        observer = 0 if plan.observer_color == "red" else 1
        for state, _view, document in replay_documents(plan, action_history):
            if states >= max_states:
                break
            states += 1
            truth = {
                int(piece["piece_slot"]): int(
                    state.pieces[_opponent_piece_id(observer, int(piece["piece_slot"]))].true_type
                )
                for piece in hidden_opponent_pieces(document)
            }
            seen_worlds = set()
            for ordinal in range(worlds_per_state):
                world = sample_world(document, ordinal)
                worlds += 1
                check = validate_world(document, world)
                for name, value in check["counters"].items():
                    counters[name] += value
                findings.extend(check["findings"][:1])
                assignment = tuple(sorted(world["assignment"].items()))
                seen_worlds.add(assignment)
                if world["assignment"] == truth:
                    truth_recovered += 1
                rebuilt = sample_world(document, ordinal)
                if rebuilt["assignment"] != world["assignment"]:
                    counters["provenance_mismatches"] += 1
                    findings.append(f"{plan.game_id}: a world did not re-derive")
            distinct_by_state.append(len(seen_worlds))
        if states >= max_states:
            break
    return {
        "audit_version": AUDIT_VERSION,
        "sampler_version": COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
        "public_states": states,
        "worlds": worlds,
        "worlds_per_state": worlds_per_state,
        "counters": counters,
        "all_counters_zero": all(value == 0 for value in counters.values()),
        "mean_distinct_worlds_per_state": (
            sum(distinct_by_state) / len(distinct_by_state) if distinct_by_state else 0.0
        ),
        "worlds_equal_to_truth_report_only": truth_recovered,
        "findings": findings[:20],
        "pass": all(value == 0 for value in counters.values()) and not findings,
    }


def _opponent_piece_id(observer: int, slot: int) -> int:
    from ..engine.pieces import make_piece_id

    return make_piece_id(1 - observer, slot)


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def run_negative_controls(sample: dict) -> dict:
    """Break the pipeline six ways; each break must be detected.

    `sample` carries a real scored block: `probabilities` `[n, 12]`,
    `true_rank` `[n]`, one public-state `document`, and the `observation`
    that document commits to.
    """
    probabilities = np.asarray(sample["probabilities"], dtype=np.float64)
    true_rank = np.asarray(sample["true_rank"], dtype=np.int64)
    document = sample["document"]
    observation = sample["observation"]
    controls: list[dict] = []

    baseline_scores = independent_scores(probabilities, true_rank)
    reference_ce = float(baseline_scores["ce"].mean())
    reference_top1 = float(baseline_scores["top1"].mean())

    # 1. reversed rank mapping
    reversed_scores = independent_scores(probabilities[:, ::-1].copy(), true_rank)
    controls.append(
        _control(
            "reversed_rank_mapping",
            abs(float(reversed_scores["ce"].mean()) - reference_ce) > AUDIT_TOLERANCE,
            {
                "reference_ce": reference_ce,
                "corrupted_ce": float(reversed_scores["ce"].mean()),
            },
        )
    )

    # 2. wrong true-rank label
    wrong_label = (true_rank + 1) % RANK_COUNT
    wrong_scores = independent_scores(probabilities, wrong_label)
    controls.append(
        _control(
            "wrong_true_rank_label",
            abs(float(wrong_scores["ce"].mean()) - reference_ce) > AUDIT_TOLERANCE
            and abs(float(wrong_scores["top1"].mean()) - reference_top1) > AUDIT_TOLERANCE,
            {
                "reference_top1": reference_top1,
                "corrupted_top1": float(wrong_scores["top1"].mean()),
            },
        )
    )

    # 3. wrong remaining inventory
    counts = list(remaining_counts(document))
    corrupted = list(counts)
    corrupted[1] += 1
    honest = remaining_count_distribution(counts, (1,) * RANK_COUNT)
    tampered = remaining_count_distribution(corrupted, (1,) * RANK_COUNT)
    conservation = check_count_conservation(document)
    tampered_total = sum(corrupted)
    controls.append(
        _control(
            "wrong_remaining_inventory",
            float(np.abs(honest - tampered).max()) > AUDIT_TOLERANCE
            and conservation["conserved"]
            and tampered_total != conservation["unresolved_pieces"],
            {
                "honest_total": int(sum(counts)),
                "tampered_total": int(tampered_total),
                "unresolved_pieces": conservation["unresolved_pieces"],
            },
        )
    )

    # 4. publicly known pieces included in the hidden denominator.
    #    The honest inventory stays honest — the error being modelled is
    #    scoring a *known* opponent piece as if it were a hidden target,
    #    which inflates the denominator without freeing any count.
    honest_hidden = hidden_opponent_pieces(document)
    known_alive = [
        piece
        for piece in document["pieces"]
        if piece["owner_color"] != document["observer_color"]
        and piece["alive"]
        and piece["known_to_observer"]
    ]
    honest_check = check_count_conservation(document, counts)
    polluted_denominator = len(honest_hidden) + len(known_alive)
    detected = (
        len(known_alive) > 0
        and honest_check["conserved"]
        and polluted_denominator != int(sum(counts))
    )
    detail = {
        "known_alive_opponent_pieces": len(known_alive),
        "honest_hidden": len(honest_hidden),
        "polluted_denominator": polluted_denominator,
        "remaining_total": int(sum(counts)),
        "honest_conserved": honest_check["conserved"],
    }
    if not known_alive:
        detail["note"] = (
            "the sampled position reveals no live opponent rank, so the "
            "control has nothing to detect here"
        )
    controls.append(_control("known_pieces_in_hidden_denominator", detected, detail))

    # 5. hidden truth injected into the production request
    injected = False
    message = None
    for field, value in (
        ("true_rank_index", 3),
        ("opponent_truth", {"0": 5}),
        ("private_piece_table", []),
        ("storage_path", "/tmp/x"),
    ):
        payload = {
            "request_version": "phase11_belief_request_v1",
            "request_id": "negative-control",
            "observer_color": document["observer_color"],
            "public_state_document": document,
            "observation": observation,
            field: value,
        }
        try:
            Phase11BeliefRequest.from_payload(payload)
        except Phase11BeliefError as error:
            injected = True
            message = str(error)
        else:  # pragma: no cover - a leak would land here
            injected = False
            message = f"the request accepted {field!r}"
            break
    controls.append(
        _control("hidden_truth_injected_into_request", injected, {"refusal": message})
    )

    # 6. permuted probability columns
    permuted = np.roll(probabilities, 1, axis=1)
    permuted_scores = independent_scores(permuted, true_rank)
    controls.append(
        _control(
            "permuted_probability_columns",
            abs(float(permuted_scores["ce"].mean()) - reference_ce) > AUDIT_TOLERANCE,
            {
                "reference_ce": reference_ce,
                "corrupted_ce": float(permuted_scores["ce"].mean()),
            },
        )
    )

    names = tuple(control["control"] for control in controls)
    if names != NEGATIVE_CONTROLS:
        raise Phase11AuditError(f"negative controls drifted: {names}")
    return {
        "audit_version": AUDIT_VERSION,
        "controls": controls,
        "all_fire": all(control["fired"] for control in controls),
        "events_scored": int(true_rank.size),
    }


def _control(name: str, fired: bool, detail: dict) -> dict:
    return {"control": name, "fired": bool(fired), "detail": detail}


__all__ = [
    "AUDIT_TOLERANCE",
    "AUDIT_VERSION",
    "BASELINE_EDGE_CASES",
    "NEGATIVE_CONTROLS",
    "Phase11AuditError",
    "baseline_edge_case_audit",
    "compare_scores",
    "replay_documents",
    "world_baseline_audit",
    "independent_scores",
    "independent_softmax",
    "run_negative_controls",
    "scalar_event_metrics",
    "scalar_recompute",
]
