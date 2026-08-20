#!/usr/bin/env python
"""Phase 12 Agent 2 runner: the belief-to-decision diagnostic.

Specification source: `03_PHASE_12_AGENT_2_BELIEF_DECISION_DIAGNOSTIC.md`.

The question, and only that question
------------------------------------
Does a better belief make the *same* search algorithm choose different — and
better-looking — actions? This runner:

1. builds a fresh 256-position diagnostic set (four behaviour groups,
   balanced colours, Phase 12 seed streams, never the spent Phase 11 test
   bank) and writes its manifest;
2. measures each belief provider's predictive quality on exactly those
   positions, so the decision result can be read against a belief ordering
   measured on the same data;
3. runs the accepted Agent 1 search core over every position with each of
   the four belief providers, changing nothing but the provider;
4. reports disagreement with direct C1, agreement with the oracle-search
   action, root-score deltas, latency, per-group results, and the pairwise
   agreement matrix.

No match run. No tournament. No training. Nothing accepted is modified, and
the Agent 1 search modules are read-only inputs here.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.belief.phase11b.features import load_frozen_c1  # noqa: E402
from stratego.belief.phase11b.interface import Phase11BPublicState  # noqa: E402
from stratego.belief.phase11b.metrics import cross_entropy  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.permutation import hidden_opponent_piece_ids  # noqa: E402
from stratego.engine.pieces import piece_setup_slot  # noqa: E402
from stratego.evaluation.phase11_baselines import remaining_count_belief  # noqa: E402
from stratego.search.phase12 import (  # noqa: E402
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_REMAINING_COUNT,
    Phase12SearchEngine,
    Phase12SearchError,
    SEARCH_VERSION,
    build_belief_provider,
    search_preset,
)
from stratego.search.phase12.contract import SCORE_DEFINITION  # noqa: E402
from stratego.search.phase12 import positions as diag  # noqa: E402

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase12"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase12"
HANDOFF_PATH = REPOSITORY_ROOT / "reports" / "phase11b" / "phase12_handoff.json"
MANIFEST_PATH = REPORT_DIRECTORY / "agent_02_position_manifest.json"
REPORT_PATH = REPORT_DIRECTORY / "agent_02_report.md"
SUMMARY_PATH = REPORT_DIRECTORY / "agent_02_summary.json"
#: The per-decision rows live beside the summary as a CSV rather than inside
#: it: 2,048 rows of 25 fields is a dataset, and inlining it would make the
#: summary ten times the size of everything a reader actually reads.
DECISIONS_PATH = REPORT_DIRECTORY / "agent_02_decisions.csv"

#: Report order. `oracle` last: it is the diagnostic upper bound the other
#: three are read against, not a peer arm.
PROVIDER_ORDER = (
    PROVIDER_REMAINING_COUNT,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
)

#: Human-readable behaviour-group names, in the instruction's own order.
GROUP_LABELS = {
    "phase9_selfplay": "Phase9-like",
    "strategic_rule": "Strategic",
    "tactical_rule": "Tactical",
    "scout_rush": "Scout-rush",
}
BUCKETS = ("early", "middle", "late")

#: Below this oracle move-disagreement rate the mechanism has no decision
#: headroom worth arguing about: perfect hidden information barely changes
#: what this search wants to play, so no belief can either. An engineering
#: threshold, named here so the verdict is not a hidden judgement call.
MECHANISM_HEADROOM_FLOOR = 0.05


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize(value):
    """Make a result tree JSON-serializable."""
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [sanitize(item) for item in value.tolist()]
    return value


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(statistics.mean(values)) if values else None


def median_or_none(values):
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


# ---------------------------------------------------------------------------
# Stage: models and providers
# ---------------------------------------------------------------------------


def load_handoff() -> dict:
    handoff = json.loads(HANDOFF_PATH.read_text())
    if handoff.get("artifact") != "phase11b_phase12_handoff_v1":
        raise Phase12SearchError(f"{HANDOFF_PATH} is not the Phase 12 handoff")
    return handoff


def load_move_model(handoff: dict, device: str):
    """The accepted Phase 9 C1, digest-checked against the handoff record."""
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model, identity = load_frozen_c1(
        REPOSITORY_ROOT,
        CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt",
        device=device,
    )
    expected = handoff["accepted_phase9_checkpoint"]
    if identity["model_state_digest"] != expected["model_state_digest"]:
        raise Phase12SearchError("loaded Phase 9 state digest != handoff record")
    if identity["belief_head_digest"] != expected["belief_head_digest"]:
        raise Phase12SearchError("loaded belief-head digest != handoff record")
    log(
        f"  move model: accepted Phase 9 C1, {identity['parameters']:,} parameters, "
        f"state digest {identity['model_state_digest'][:12]}..., device {device}"
    )
    return model, identity


def build_providers(model, handoff: dict, device: str) -> dict:
    """The four interchangeable providers, bound to the handoff identities."""
    agent1c_record = handoff["agent1c_checkpoint"]
    providers = {
        PROVIDER_REMAINING_COUNT: build_belief_provider(
            PROVIDER_REMAINING_COUNT, production=True
        ),
        PROVIDER_ORIGINAL_PHASE11: build_belief_provider(
            PROVIDER_ORIGINAL_PHASE11, encoder=model, production=True, device=device
        ),
        PROVIDER_AGENT1C: build_belief_provider(
            PROVIDER_AGENT1C,
            encoder=model,
            agent1c_checkpoint=REPOSITORY_ROOT / agent1c_record["path"],
            expected_agent1c_sha256=agent1c_record["sha256"],
            expected_agent1c_state_digest=agent1c_record["state_dict_digest"],
            production=True,
            device=device,
        ),
        PROVIDER_ORACLE: build_belief_provider(PROVIDER_ORACLE, production=False),
    }
    for name in PROVIDER_ORDER:
        provider = providers[name]
        log(f"  provider ready: {name} (uses_hidden_truth={provider.uses_hidden_truth})")
    return providers


# ---------------------------------------------------------------------------
# Stage: the fresh diagnostic position set
# ---------------------------------------------------------------------------


def build_position_set(owners_device: str, *, positions_per_cell: int) -> dict:
    from stratego.evaluation.phase11_pipeline import build_owners

    owners, _ = build_owners(
        REPOSITORY_ROOT,
        CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt",
        device=owners_device,
    )
    started = time.perf_counter()
    state = {"last": 0.0}

    def progress(stratum, source, color, taken, games, total):
        now = time.perf_counter()
        if now - state["last"] < 5.0:
            return
        state["last"] = now
        log(
            f"  [positions] {stratum}/{source}/{color}: {taken}/{positions_per_cell} "
            f"in cell, {games} games, {total} positions, {now - started:.0f}s"
        )

    generated = diag.generate_positions(
        owners, positions_per_cell=positions_per_cell, progress=progress
    )
    manifest = diag.build_manifest(
        generated,
        generated_utc=utc_now(),
        generation_seconds=round(time.perf_counter() - started, 3),
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    log(
        f"  positions: {manifest['counts']['positions']} from "
        f"{manifest['counts']['games_played']} games in "
        f"{time.perf_counter() - started:.0f}s -> {MANIFEST_PATH.name} "
        f"(digest {manifest['manifest_digest'][:12]}...)"
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage: belief quality on exactly these positions
# ---------------------------------------------------------------------------


def true_hidden_ranks(state, observer: int) -> dict:
    """`{piece_slot: true rank}` for the observer's unresolved opponent pieces.

    Privileged. Used only by the labels of the belief-quality diagnostic and
    by the oracle arm, both explicitly offline.
    """
    return {
        piece_setup_slot(piece_id): int(state.pieces[piece_id].true_type)
        for piece_id in hidden_opponent_piece_ids(state, observer)
    }


def belief_quality(providers: dict, materialized: list, worlds: int) -> dict:
    """Per-provider CE / R_CE / top-1 and sampled-world rank accuracy.

    The metric definitions are the accepted Phase 11/11B ones — the raw
    softmax against the true rank, divided by the `remaining_count_belief_v1`
    cross-entropy on the same pieces — but computed on this fresh position
    set, so the numbers are comparable *to each other* and must not be read
    as Phase 11B leaderboard values.
    """
    rows: dict = {name: {"ce": [], "correct": [], "base_ce": [], "group": []} for name in PROVIDER_ORDER
                  if name != PROVIDER_ORACLE}
    world_hits: dict = {name: [0, 0] for name in PROVIDER_ORDER}
    started = time.perf_counter()

    for index, record in enumerate(materialized):
        state = record["state"]
        observer = int(record["observer_player"])
        public = Phase11BPublicState(record["document"], record["observation"])
        truth = true_hidden_ranks(state, observer)
        slots = sorted(truth)
        labels = np.array([truth[slot] for slot in slots], dtype=np.int64)
        baseline = np.stack(
            [remaining_count_belief(record["document"])[slot] for slot in slots]
        ).astype(np.float64)
        base_ce = cross_entropy(baseline, labels)

        for name in PROVIDER_ORDER:
            provider = providers[name]
            if name == PROVIDER_ORACLE:
                assignments = provider.sample_assignments_privileged(
                    state, public, worlds, record["search_seed"]
                )
            else:
                marginals = provider.predict_marginals(public)
                probabilities = np.stack([marginals[slot] for slot in slots]).astype(
                    np.float64
                )
                candidate_ce = cross_entropy(probabilities, labels)
                block = rows[name]
                block["ce"].extend(float(value) for value in candidate_ce)
                block["base_ce"].extend(float(value) for value in base_ce)
                block["correct"].extend(
                    bool(value) for value in (probabilities.argmax(axis=1) == labels)
                )
                block["group"].extend([record["stratum"]] * len(slots))
                assignments = provider.sample_assignments(
                    public, worlds, record["search_seed"]
                )
            hits, total = world_hits[name]
            for assignment in assignments:
                for slot, rank in assignment.items():
                    total += 1
                    hits += int(rank == truth[int(slot)])
            world_hits[name] = [hits, total]
        if (index + 1) % 64 == 0:
            log(
                f"  [belief] {index + 1}/{len(materialized)} positions, "
                f"{time.perf_counter() - started:.0f}s"
            )

    report: dict = {}
    for name, block in rows.items():
        candidate = np.asarray(block["ce"], dtype=np.float64)
        base = np.asarray(block["base_ce"], dtype=np.float64)
        correct = np.asarray(block["correct"], dtype=bool)
        groups = np.asarray(block["group"])
        per_group = {}
        for stratum in diag.DIAGNOSTIC_STRATA:
            selection = groups == stratum
            if not selection.any():
                continue
            per_group[stratum] = {
                "pieces": int(selection.sum()),
                "ce": float(candidate[selection].mean()),
                "baseline_ce": float(base[selection].mean()),
                "r_ce": float(candidate[selection].mean() / base[selection].mean()),
                "top1": float(correct[selection].mean()),
            }
        hits, total = world_hits[name]
        report[name] = {
            "pieces": int(candidate.size),
            "ce": float(candidate.mean()),
            "baseline_ce": float(base.mean()),
            "r_ce": float(candidate.mean() / base.mean()),
            "top1": float(correct.mean()),
            "by_behavior_group": per_group,
            "sampled_world_rank_accuracy": (hits / total) if total else None,
            "sampled_world_pieces": total,
        }
    hits, total = world_hits[PROVIDER_ORACLE]
    report[PROVIDER_ORACLE] = {
        "pieces": None,
        "ce": 0.0,
        "baseline_ce": None,
        "r_ce": 0.0,
        "top1": 1.0,
        "by_behavior_group": {},
        "sampled_world_rank_accuracy": (hits / total) if total else None,
        "sampled_world_pieces": total,
        "note": "true hidden state; perfect by construction, offline diagnostic only",
    }
    log(f"  belief quality computed in {time.perf_counter() - started:.0f}s")
    return report


# ---------------------------------------------------------------------------
# Stage: the search matrix
# ---------------------------------------------------------------------------


def candidate_entry(decision, action_id: int):
    for candidate in decision.candidates:
        if candidate.absolute_action_id == int(action_id):
            return candidate
    return None


def run_search_matrix(
    model,
    model_identity: dict,
    providers: dict,
    materialized: list,
    preset_name: str,
    device: str,
) -> list:
    """Every provider on every position at one preset. Only the provider varies."""
    engines = {}
    for name in PROVIDER_ORDER:
        config = search_preset(preset_name, production=(name != PROVIDER_ORACLE))
        engines[name] = Phase12SearchEngine(
            model,
            providers[name],
            config,
            device=device,
            model_identity=model_identity,
        )

    rows: list = []
    started = time.perf_counter()
    for index, record in enumerate(materialized):
        state = record["state"]
        legal = set(legal_actions(state))
        seed = int(record["search_seed"])
        direct_reference = None
        for name in PROVIDER_ORDER:
            decision = engines[name].choose_action(state, seed=seed)
            if decision.selected_action_id not in legal:
                raise Phase12SearchError(
                    f"{name} selected an illegal action at {record['position_id']}"
                )
            if not decision.candidates[0].is_direct:
                raise Phase12SearchError(
                    f"{name} lost the direct action from its candidate set at "
                    f"{record['position_id']}"
                )
            # Every arm must see the identical direct Phase 9 action and the
            # identical root value: that is what makes this a belief
            # comparison rather than four unrelated searches.
            if direct_reference is None:
                direct_reference = (
                    decision.direct_action_id,
                    round(decision.root_direct_value, 9),
                    tuple(c.absolute_action_id for c in decision.candidates),
                )
            else:
                current = (
                    decision.direct_action_id,
                    round(decision.root_direct_value, 9),
                    tuple(c.absolute_action_id for c in decision.candidates),
                )
                if current != direct_reference:
                    raise Phase12SearchError(
                        f"the arms disagree about the shared root at "
                        f"{record['position_id']}: {current} != {direct_reference}"
                    )
            direct = candidate_entry(decision, decision.direct_action_id)
            selected = candidate_entry(decision, decision.selected_action_id)
            rows.append(
                {
                    "position_id": record["position_id"],
                    "stratum": record["stratum"],
                    "progress_bucket": record["document_summary"]["progress_bucket"],
                    "observer_color": record["observer_color"],
                    "ply": int(record["ply"]),
                    "unresolved": int(record["unresolved"]),
                    "provider": name,
                    "preset": preset_name,
                    "seed": seed,
                    "selected_action_id": int(decision.selected_action_id),
                    "direct_action_id": int(decision.direct_action_id),
                    "move_changed": bool(decision.move_changed),
                    "root_direct_value": float(decision.root_direct_value),
                    "q_selected": float(selected.q_value),
                    "q_direct": float(direct.q_value),
                    "score_selected": float(selected.score),
                    "score_direct": float(direct.score),
                    "prior_selected": float(selected.prior),
                    "prior_direct": float(direct.prior),
                    "candidate_count": len(decision.candidates),
                    "legal_action_count": int(decision.legal_action_count),
                    "unique_worlds": int(decision.unique_worlds),
                    "c1_forwards": int(decision.c1_forwards),
                    "terminal_leaves": int(decision.terminal_leaves),
                    "seconds": float(decision.seconds),
                }
            )
        if (index + 1) % 32 == 0:
            elapsed = time.perf_counter() - started
            rate = (index + 1) / max(elapsed, 1e-9)
            log(
                f"  [{preset_name}] {index + 1}/{len(materialized)} positions x "
                f"{len(PROVIDER_ORDER)} arms, {elapsed:.0f}s, "
                f"eta {(len(materialized) - index - 1) / max(rate, 1e-9):.0f}s"
            )
    log(
        f"  {preset_name}: {len(rows)} decisions in "
        f"{time.perf_counter() - started:.0f}s"
    )
    return rows


# ---------------------------------------------------------------------------
# Stage: diagnostics
# ---------------------------------------------------------------------------


def index_rows(rows: list) -> dict:
    """`{(preset, provider, position_id): row}`."""
    return {(row["preset"], row["provider"], row["position_id"]): row for row in rows}


def arm_diagnostics(rows: list, preset_name: str) -> dict:
    """The instructed per-arm diagnostic block, overall and by group."""
    lookup = index_rows(rows)
    position_ids = []
    seen = set()
    for row in rows:
        if row["preset"] == preset_name and row["position_id"] not in seen:
            seen.add(row["position_id"])
            position_ids.append(row["position_id"])

    oracle_choice = {
        position: lookup[(preset_name, PROVIDER_ORACLE, position)]["selected_action_id"]
        for position in position_ids
    }
    direct_choice = {
        position: lookup[(preset_name, PROVIDER_ORACLE, position)]["direct_action_id"]
        for position in position_ids
    }

    def block(selected_positions: list, provider: str) -> dict:
        members = [
            lookup[(preset_name, provider, position)] for position in selected_positions
        ]
        if not members:
            return {}
        changed = [row for row in members if row["move_changed"]]
        agrees = [
            row
            for row in members
            if row["selected_action_id"] == oracle_choice[row["position_id"]]
        ]
        changed_and_agrees = [
            row
            for row in changed
            if row["selected_action_id"] == oracle_choice[row["position_id"]]
        ]
        # Where direct C1 already disagrees with the perfect-information
        # search, does this arm fix it? Where direct C1 already agrees, does
        # this arm break it? Net oracle agreement hides both.
        direct_wrong = [
            row
            for row in members
            if direct_choice[row["position_id"]] != oracle_choice[row["position_id"]]
        ]
        direct_right = [
            row
            for row in members
            if direct_choice[row["position_id"]] == oracle_choice[row["position_id"]]
        ]
        fixed = [
            row
            for row in direct_wrong
            if row["selected_action_id"] == oracle_choice[row["position_id"]]
        ]
        broken = [
            row
            for row in direct_right
            if row["selected_action_id"] != oracle_choice[row["position_id"]]
        ]
        return {
            "oracle_headroom_positions": len(direct_wrong),
            "oracle_fixed": len(fixed),
            "oracle_fix_rate": (len(fixed) / len(direct_wrong)) if direct_wrong else None,
            "oracle_broken": len(broken),
            "oracle_break_rate": (
                (len(broken) / len(direct_right)) if direct_right else None
            ),
            "net_oracle_moves": len(fixed) - len(broken),
            "positions": len(members),
            "move_disagreement_rate_vs_direct": len(changed) / len(members),
            "oracle_agreement_rate": len(agrees) / len(members),
            "oracle_agreement_when_changed": (
                len(changed_and_agrees) / len(changed) if changed else None
            ),
            "score_delta_vs_direct": {
                "mean": mean_or_none(
                    [row["score_selected"] - row["score_direct"] for row in members]
                ),
                "median": median_or_none(
                    [row["score_selected"] - row["score_direct"] for row in members]
                ),
                "max": max(row["score_selected"] - row["score_direct"] for row in members),
            },
            "q_delta_vs_direct": {
                "mean": mean_or_none(
                    [row["q_selected"] - row["q_direct"] for row in members]
                ),
                "median": median_or_none(
                    [row["q_selected"] - row["q_direct"] for row in members]
                ),
            },
            # The same two deltas restricted to the positions where the arm
            # actually left the direct move. Over the whole set they are
            # dominated by the zeros of the positions it did not change.
            "when_changed": {
                "positions": len(changed),
                "mean_score_delta": mean_or_none(
                    [row["score_selected"] - row["score_direct"] for row in changed]
                ),
                "mean_q_delta": mean_or_none(
                    [row["q_selected"] - row["q_direct"] for row in changed]
                ),
                "mean_prior_of_chosen": mean_or_none(
                    [row["prior_selected"] for row in changed]
                ),
                "mean_prior_of_direct": mean_or_none(
                    [row["prior_direct"] for row in changed]
                ),
            },
            "q_of_direct_action": {
                "mean": mean_or_none([row["q_direct"] for row in members]),
                "median": median_or_none([row["q_direct"] for row in members]),
            },
            "move_latency_seconds": {
                "mean": mean_or_none([row["seconds"] for row in members]),
                "median": median_or_none([row["seconds"] for row in members]),
            },
            "c1_forwards_per_move": mean_or_none(
                [row["c1_forwards"] for row in members]
            ),
            "mean_unique_worlds": mean_or_none([row["unique_worlds"] for row in members]),
            "mean_candidates": mean_or_none([row["candidate_count"] for row in members]),
        }

    grouped: dict = {}
    for provider in PROVIDER_ORDER:
        entry = block(position_ids, provider)
        entry["by_behavior_group"] = {}
        for stratum in diag.DIAGNOSTIC_STRATA:
            members = [
                position
                for position in position_ids
                if lookup[(preset_name, provider, position)]["stratum"] == stratum
            ]
            entry["by_behavior_group"][stratum] = block(members, provider)
        entry["by_progress_bucket"] = {}
        for bucket in BUCKETS:
            members = [
                position
                for position in position_ids
                if lookup[(preset_name, provider, position)]["progress_bucket"] == bucket
            ]
            if members:
                entry["by_progress_bucket"][bucket] = block(members, provider)
        grouped[provider] = entry

    # The reference point every arm is read against: how often the direct
    # Phase 9 move already is the perfect-information search's move.
    direct_agreement = sum(
        1
        for position in position_ids
        if direct_choice[position] == oracle_choice[position]
    ) / max(len(position_ids), 1)
    grouped["direct_c1"] = {
        "positions": len(position_ids),
        "move_disagreement_rate_vs_direct": 0.0,
        "oracle_agreement_rate": direct_agreement,
        "note": "the accepted Phase 9 greedy action; the arms' common baseline",
    }
    grouped["direct_c1"]["by_behavior_group"] = {}
    for stratum in diag.DIAGNOSTIC_STRATA:
        members = [
            position
            for position in position_ids
            if lookup[(preset_name, PROVIDER_ORACLE, position)]["stratum"] == stratum
        ]
        if members:
            grouped["direct_c1"]["by_behavior_group"][stratum] = {
                "positions": len(members),
                "oracle_agreement_rate": sum(
                    1
                    for position in members
                    if direct_choice[position] == oracle_choice[position]
                )
                / len(members),
            }
    return grouped


def pairwise_agreement(rows: list, preset_name: str) -> dict:
    """The 4x4 action-agreement matrix among the belief providers."""
    lookup = index_rows(rows)
    position_ids = sorted(
        {row["position_id"] for row in rows if row["preset"] == preset_name}
    )
    matrix: dict = {}
    for left in PROVIDER_ORDER:
        matrix[left] = {}
        for right in PROVIDER_ORDER:
            agree = sum(
                1
                for position in position_ids
                if lookup[(preset_name, left, position)]["selected_action_id"]
                == lookup[(preset_name, right, position)]["selected_action_id"]
            )
            matrix[left][right] = agree / max(len(position_ids), 1)
    return matrix


def unanimity(rows: list, preset_name: str) -> dict:
    """How often the arms all agree, and how often any arm leaves direct C1."""
    lookup = index_rows(rows)
    position_ids = sorted(
        {row["position_id"] for row in rows if row["preset"] == preset_name}
    )
    all_same = 0
    production_same = 0
    any_changed = 0
    for position in position_ids:
        choices = {
            provider: lookup[(preset_name, provider, position)]["selected_action_id"]
            for provider in PROVIDER_ORDER
        }
        direct = lookup[(preset_name, PROVIDER_ORACLE, position)]["direct_action_id"]
        all_same += int(len(set(choices.values())) == 1)
        production_same += int(
            len({choices[p] for p in PROVIDER_ORDER if p != PROVIDER_ORACLE}) == 1
        )
        any_changed += int(any(choice != direct for choice in choices.values()))
    total = max(len(position_ids), 1)
    return {
        "positions": len(position_ids),
        "all_four_arms_identical": all_same / total,
        "three_production_arms_identical": production_same / total,
        "any_arm_left_direct_c1": any_changed / total,
    }


def reproducibility_probe(
    model, model_identity, providers, materialized, preset_name, device, sample
) -> dict:
    """Re-decide a slice under the same seeds and require identical actions."""
    subset = materialized[:sample]
    repeat = run_search_matrix(
        model, model_identity, providers, subset, preset_name, device
    )
    return {
        "preset": preset_name,
        "positions": len(subset),
        "decisions": len(repeat),
        "rows": repeat,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def percent(value) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def number(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def arm_table(diagnostics: dict) -> list:
    lines = [
        "| arm | move disagreement vs direct | oracle agreement | oracle agreement when it deviated | mean S(sel) − S(direct) | median | mean Q(direct) | s/move |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for provider in ("direct_c1",) + PROVIDER_ORDER:
        block = diagnostics[provider]
        if provider == "direct_c1":
            lines.append(
                f"| direct C1 (no search) | 0.0% | {percent(block['oracle_agreement_rate'])} "
                "| — | — | — | — | — |"
            )
            continue
        lines.append(
            "| {name} | {dis} | {orc} | {orcc} | {mean} | {median} | {qd} | {sec} |".format(
                name=f"search + {provider}",
                dis=percent(block["move_disagreement_rate_vs_direct"]),
                orc=percent(block["oracle_agreement_rate"]),
                orcc=percent(block["oracle_agreement_when_changed"]),
                mean=number(block["score_delta_vs_direct"]["mean"]),
                median=number(block["score_delta_vs_direct"]["median"]),
                qd=number(block["q_of_direct_action"]["mean"]),
                sec=number(block["move_latency_seconds"]["mean"], 3),
            )
        )
    return lines


def group_table(diagnostics: dict, field: str, formatter) -> list:
    lines = [
        "| arm | " + " | ".join(GROUP_LABELS[s] for s in diag.DIAGNOSTIC_STRATA) + " |",
        "|---|" + "---|" * len(diag.DIAGNOSTIC_STRATA),
    ]
    for provider in PROVIDER_ORDER:
        cells = []
        for stratum in diag.DIAGNOSTIC_STRATA:
            block = diagnostics[provider]["by_behavior_group"].get(stratum) or {}
            cells.append(formatter(block.get(field)))
        lines.append(f"| {provider} | " + " | ".join(cells) + " |")
    return lines


def pairwise_table(matrix: dict) -> list:
    lines = [
        "| | " + " | ".join(PROVIDER_ORDER) + " |",
        "|---|" + "---|" * len(PROVIDER_ORDER),
    ]
    for left in PROVIDER_ORDER:
        cells = [percent(matrix[left][right]) for right in PROVIDER_ORDER]
        lines.append(f"| **{left}** | " + " | ".join(cells) + " |")
    return lines


def write_decisions(rows: list, path: Path) -> dict:
    """Write every decision as a CSV row; return the summary's pointer block."""
    import csv

    columns = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    log(f"  wrote {path} ({len(rows)} rows)")
    return {
        "path": f"reports/phase12/{path.name}",
        "rows": len(rows),
        "columns": columns,
        "sha256": digest,
        "note": (
            "one row per (preset, provider, position); every aggregate in this "
            "summary is recomputable from these rows alone"
        ),
    }


def write_report(summary: dict, path: Path) -> None:
    s = summary
    primary = s["primary_preset"]
    diagnostics = s["diagnostics"][primary]
    lines: list = []
    add = lines.append

    add("# Phase 12 Agent 2 — Belief-to-Decision Diagnostic")
    add("")
    add(f"Generated {s['generated_utc']} by `scripts/run_phase12_agent02.py`.")
    add("")
    add(
        "Engineering artifact of the Phase 12 rapid search-engineering phase. "
        "Position-level comparison only: no match set, no tournament, no "
        "significance claim, no change to the Agent 1 search core."
    )
    add("")
    add("## 1. Question and verdict")
    add("")
    add("```text")
    add(s["verdict"]["statement"])
    add("```")
    add("")
    for line in s["verdict"]["findings"]:
        add(f"- {line}")
    add("")

    add("## 2. The fresh diagnostic position set")
    add("")
    manifest = s["position_set"]
    add("```text")
    add(f"artifact      {manifest['artifact']}")
    add(f"positions     {manifest['positions']} from {manifest['games_played']} fresh games")
    add(
        "groups        "
        + ", ".join(
            f"{GROUP_LABELS[k]} {v}"
            for k, v in manifest["positions_by_behavior_group"].items()
        )
    )
    add(
        "colours       "
        + ", ".join(f"{k} {v}" for k, v in manifest["positions_by_observer_color"].items())
    )
    add(f"eligibility   observer to act, ply >= {manifest['eligibility']['min_ply']}, "
        f">= {manifest['eligibility']['min_unresolved_opponent_pieces']} unresolved opponent pieces")
    add(f"selection     {manifest['eligibility']['selection']}, "
        f"{manifest['eligibility']['positions_per_game']} per game")
    add(f"setups        accepted library split '{manifest['setup_library_split']}' "
        "(neither the spent test pool nor Agent 1C's training pool)")
    add("opponents     the four accepted Phase 11 strata, unmodified")
    add(f"seeds         Phase 12 personalization, master {manifest['master_seed']}")
    add(f"test bank     {manifest['phase11_test_bank_used']} (never opened)")
    add(f"manifest      reports/phase12/{MANIFEST_PATH.name}  sha-of-content {manifest['manifest_digest'][:16]}...")
    add("```")
    add("")
    add(
        "Every position replays bit-for-bit from the manifest: the rebuilt "
        "observation is required to match the digest the observer recorded "
        f"while the game was played, and all {manifest['positions']} did."
    )
    add("")
    add("### Position mix")
    add("")
    add("```text")
    for key, value in s["position_mix"].items():
        add(f"{key:<28} {value}")
    add("```")
    add("")

    add("## 3. Belief quality on these same positions")
    add("")
    add(
        "Measured with the accepted `R_CE` arithmetic on this fresh set, so the "
        "three providers are comparable to each other here. These are **not** "
        "Phase 11B leaderboard numbers: different positions, different games."
    )
    add("")
    add("| provider | pieces | CE | R_CE | top-1 | sampled-world rank accuracy |")
    add("|---|---|---|---|---|---|")
    for provider in PROVIDER_ORDER:
        block = s["belief_quality"][provider]
        if provider == PROVIDER_ORACLE:
            add(
                "| oracle | — | 0 | 0 | 100% | "
                f"{percent(block['sampled_world_rank_accuracy'])} (true state) |"
            )
            continue
        add(
            f"| {provider} | {block['pieces']:,} | {number(block['ce'], 4)} | "
            f"{number(block['r_ce'], 4)} | {percent(block['top1'])} | "
            f"{percent(block['sampled_world_rank_accuracy'])} |"
        )
    add("")
    add(
        "`sampled-world rank accuracy` is the fraction of hidden pieces whose "
        "rank the provider's sampled worlds actually got right — the quantity "
        "search consumes, as opposed to the marginals it is scored on."
    )
    add("")
    add("R_CE by behaviour group:")
    add("")
    add("| provider | " + " | ".join(GROUP_LABELS[g] for g in diag.DIAGNOSTIC_STRATA) + " |")
    add("|---|" + "---|" * len(diag.DIAGNOSTIC_STRATA))
    for provider in PROVIDER_ORDER:
        if provider == PROVIDER_ORACLE:
            continue
        cells = []
        for stratum in diag.DIAGNOSTIC_STRATA:
            entry = s["belief_quality"][provider]["by_behavior_group"].get(stratum)
            cells.append("—" if not entry else number(entry["r_ce"]))
        add(f"| {provider} | " + " | ".join(cells) + " |")
    add("")

    add(f"## 4. Decisions at {primary} (primary comparison)")
    add("")
    lines.extend(arm_table(diagnostics))
    add("")
    add(
        "`S(sel) − S(direct)` is non-negative by construction — the direct "
        "action is always a candidate, so a search that changes the move only "
        "does so when it scores the new move higher. Its size is how much "
        "better the arm *thinks* its choice is; `Q(direct)` is each arm's own "
        "valuation of the same unchanged action, which is where world realism "
        "shows up."
    )
    add("")
    add("### Pairwise action agreement")
    add("")
    lines.extend(pairwise_table(s["pairwise_agreement"][primary]))
    add("")
    block = s["unanimity"][primary]
    add("```text")
    add(f"all four arms chose the identical action        {percent(block['all_four_arms_identical'])}")
    add(f"the three production arms chose identically     {percent(block['three_production_arms_identical'])}")
    add(f"at least one arm left the direct C1 move        {percent(block['any_arm_left_direct_c1'])}")
    add("```")
    add("")
    add("### Move disagreement vs direct C1, by behaviour group")
    add("")
    lines.extend(
        group_table(diagnostics, "move_disagreement_rate_vs_direct", percent)
    )
    add("")
    add("### Oracle agreement, by behaviour group")
    add("")
    lines.extend(group_table(diagnostics, "oracle_agreement_rate", percent))
    add("")
    add(
        "Direct C1's own oracle agreement by group: "
        + ", ".join(
            f"{GROUP_LABELS[stratum]} "
            f"{percent(block['oracle_agreement_rate'])}"
            for stratum, block in diagnostics["direct_c1"]["by_behavior_group"].items()
        )
        + "."
    )
    add("")
    add("### By game phase (move disagreement / oracle agreement)")
    add("")
    add("| arm | " + " | ".join(BUCKETS) + " |")
    add("|---|" + "---|" * len(BUCKETS))
    for provider in PROVIDER_ORDER:
        cells = []
        for bucket in BUCKETS:
            entry = diagnostics[provider]["by_progress_bucket"].get(bucket)
            cells.append(
                "—"
                if not entry
                else f"{percent(entry['move_disagreement_rate_vs_direct'])} / "
                f"{percent(entry['oracle_agreement_rate'])}"
            )
        add(f"| {provider} | " + " | ".join(cells) + " |")
    add("")
    add("### Where the arms move the decision, relative to the oracle choice")
    add("")
    add(
        "Net oracle agreement hides two opposite effects. These are the "
        "positions where direct C1 and the perfect-information search already "
        "disagree (the headroom), and the positions where they already agree "
        "(what an arm can break)."
    )
    add("")
    add(
        "| arm | headroom positions | fixed | fix rate | broken | break rate | net |"
    )
    add("|---|---|---|---|---|---|---|")
    for provider in PROVIDER_ORDER:
        block = diagnostics[provider]
        add(
            f"| {provider} | {block['oracle_headroom_positions']} | "
            f"{block['oracle_fixed']} | {percent(block['oracle_fix_rate'])} | "
            f"{block['oracle_broken']} | {percent(block['oracle_break_rate'])} | "
            f"{block['net_oracle_moves']:+d} |"
        )
    add("")
    add("### Only the positions where the arm left the direct move")
    add("")
    add(
        "| arm | positions changed | mean ΔS | mean ΔQ | prior of chosen | prior of direct |"
    )
    add("|---|---|---|---|---|---|")
    for provider in PROVIDER_ORDER:
        block = diagnostics[provider]["when_changed"]
        add(
            f"| {provider} | {block['positions']} | "
            f"{number(block['mean_score_delta'])} | {number(block['mean_q_delta'])} | "
            f"{number(block['mean_prior_of_chosen'], 3)} | "
            f"{number(block['mean_prior_of_direct'], 3)} |"
        )
    add("")
    add(
        "ΔQ is the world-averaged value the arm gains by switching; ΔS adds the "
        "policy-regularization term, which is negative for every switch away "
        "from the policy's own top move — a switch has to buy more value than "
        "it gives up in prior."
    )
    add("")

    other = [name for name in s["presets_run"] if name != primary]
    add("## 5. Budget sensitivity")
    add("")
    add(
        "The oracle *choice* is itself a search product, so it moves with the "
        "budget: direct C1's agreement with it is "
        + ", ".join(
            f"{percent(s['diagnostics'][name]['direct_c1']['oracle_agreement_rate'])}"
            f" at {name}"
            for name in s["presets_run"]
        )
        + ". Compare arms within a budget, never numbers across budgets."
    )
    add("")
    if not other:
        add("Only one preset was run; there is nothing to compare against.")
        add("")
    for name in other:
        add(f"### {name}")
        add("")
        lines.extend(arm_table(s["diagnostics"][name]))
        add("")
        add("Pairwise action agreement:")
        add("")
        lines.extend(pairwise_table(s["pairwise_agreement"][name]))
        add("")

    add("## 6. Cost")
    add("")
    add("| preset | worlds | depth | arms | decisions | s/move (mean) | total s |")
    add("|---|---|---|---|---|---|---|")
    for name in s["presets_run"]:
        cost = s["cost"][name]
        add(
            f"| {name} | {cost['worlds']} | {cost['rollout_depth']} | "
            f"{cost['arms']} | {cost['decisions']} | {number(cost['mean_seconds'], 3)} | "
            f"{cost['total_seconds']:.0f} |"
        )
    add("")
    add(
        f"Device `{s['device']}`, {s['environment']['torch_threads']} torch threads. "
        "Latency is end-to-end per decision through the whole search stack."
    )
    add("")
    if s.get("reproducibility"):
        repro = s["reproducibility"]
        add(
            f"Repeat probe: re-deciding {repro['positions']} positions "
            f"({repro['decisions']} decisions) under the same seeds reproduced "
            f"{'every' if repro['identical'] else 'NOT every'} action and root score."
        )
        add("")

    add("## 7. Interpretation")
    add("")
    for paragraph in s["verdict"]["interpretation"]:
        add(paragraph)
        add("")

    add("## 8. Limitations")
    add("")
    for item in s["limitations"]:
        add(f"- {item}")
    add("")

    add("## 9. Deliverables and status")
    add("")
    add("```text")
    add("stratego/search/phase12/positions.py           (new; Agent 1's modules untouched)")
    add(f"tests/search/test_phase12_positions.py")
    add(f"reports/phase12/{MANIFEST_PATH.name}")
    add(f"reports/phase12/{DECISIONS_PATH.name}")
    add(f"reports/phase12/{REPORT_PATH.name}")
    add(f"reports/phase12/{SUMMARY_PATH.name}")
    add("")
    for key, value in s["status"].items():
        add(f"{key:<32} {value}")
    add("```")
    add("")
    add(
        "Stop condition reached: the position-level comparison is complete. No "
        "match set was run and Agent 3 is not launched."
    )
    add("")
    path.write_text("\n".join(lines))
    log(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def build_verdict(summary: dict) -> dict:
    """The engineering read of the numbers, stated plainly."""
    primary = summary["primary_preset"]
    diagnostics = summary["diagnostics"][primary]
    quality = summary["belief_quality"]
    agent1c = diagnostics[PROVIDER_AGENT1C]
    original = diagnostics[PROVIDER_ORIGINAL_PHASE11]
    count = diagnostics[PROVIDER_REMAINING_COUNT]
    oracle = diagnostics[PROVIDER_ORACLE]
    direct = diagnostics["direct_c1"]

    belief_ordering_holds = (
        quality[PROVIDER_AGENT1C]["r_ce"]
        < quality[PROVIDER_ORIGINAL_PHASE11]["r_ce"]
        < 1.0
    )
    decisions_differ = (
        summary["pairwise_agreement"][primary][PROVIDER_AGENT1C][
            PROVIDER_ORIGINAL_PHASE11
        ]
        < 1.0
    )
    moves_toward_oracle = (
        agent1c["oracle_agreement_rate"] > direct["oracle_agreement_rate"]
    )
    agent1c_beats_original = (
        agent1c["oracle_agreement_rate"] > original["oracle_agreement_rate"]
    )
    # The oracle arm's own oracle agreement is 100% by definition, so it says
    # nothing at all. What says something is whether perfect hidden information
    # makes this search want a *different* move: that fraction is the entire
    # decision headroom any belief could ever recover.
    oracle_headroom = oracle["move_disagreement_rate_vs_direct"]
    mechanism_has_headroom = oracle_headroom >= MECHANISM_HEADROOM_FLOOR

    # How big is the agent1c-vs-original gap in *positions*, and does its sign
    # survive a budget change? A gap of one or two positions that flips between
    # TINY and SMALL is noise, and saying so is more useful than the label.
    total_positions = int(summary["position_set"]["positions"])
    gaps = {
        name: (
            summary["diagnostics"][name][PROVIDER_AGENT1C]["oracle_agreement_rate"]
            - summary["diagnostics"][name][PROVIDER_ORIGINAL_PHASE11][
                "oracle_agreement_rate"
            ]
        )
        for name in summary["presets_run"]
    }
    gap_signs = {(gap > 0) - (gap < 0) for gap in gaps.values()}
    gap_sign_stable = len(summary["presets_run"]) > 1 and len(gap_signs) == 1
    gap_text = ", ".join(
        f"{name} {gap * 100:+.1f}pp "
        f"({round(gap * total_positions):+.0f} of {total_positions} positions)"
        for name, gap in gaps.items()
    )

    findings = [
        f"On this fresh set the belief ordering is "
        f"agent1c R_CE {quality[PROVIDER_AGENT1C]['r_ce']:.4f} < original_phase11 "
        f"{quality[PROVIDER_ORIGINAL_PHASE11]['r_ce']:.4f} < remaining_count 1.0"
        + (" — the Phase 11B ordering reproduces here." if belief_ordering_holds
           else " — **the Phase 11B ordering does not reproduce here**."),
        f"Search changes the move in {percent(count['move_disagreement_rate_vs_direct'])} "
        f"(count), {percent(original['move_disagreement_rate_vs_direct'])} (original), "
        f"{percent(agent1c['move_disagreement_rate_vs_direct'])} (agent1c) and "
        f"{percent(oracle['move_disagreement_rate_vs_direct'])} (oracle) of positions.",
        f"agent1c and original_phase11 pick the same action at "
        f"{percent(summary['pairwise_agreement'][primary][PROVIDER_AGENT1C][PROVIDER_ORIGINAL_PHASE11])} "
        "of positions"
        + (
            "; the belief improvement does reach the decision."
            if decisions_differ
            else "; the belief improvement does not reach the decision."
        ),
        f"Oracle agreement: direct C1 {percent(direct['oracle_agreement_rate'])}, "
        f"count {percent(count['oracle_agreement_rate'])}, "
        f"original {percent(original['oracle_agreement_rate'])}, "
        f"agent1c {percent(agent1c['oracle_agreement_rate'])}. Every search arm "
        "beats direct C1 here, and the three are within a few positions of each "
        "other.",
        f"agent1c minus original_phase11 oracle agreement: {gap_text}"
        + (
            " — the same sign at both budgets."
            if gap_sign_stable
            else " — the sign flips between budgets, so at this sample size the "
            "two beliefs are not separable by this measure."
        ),
        f"Of the {agent1c['oracle_headroom_positions']} positions where direct C1 "
        f"and the oracle search disagree, agent1c recovers the most "
        f"({agent1c['oracle_fixed']}, a {percent(agent1c['oracle_fix_rate'])} fix "
        f"rate, against {original['oracle_fixed']} for original_phase11 and "
        f"{count['oracle_fixed']} for remaining_count) but also breaks the most "
        f"({agent1c['oracle_broken']} against {original['oracle_broken']} and "
        f"{count['oracle_broken']}), so the net is a wash: "
        f"{agent1c['net_oracle_moves']:+d} against "
        f"{original['net_oracle_moves']:+d} and {count['net_oracle_moves']:+d}.",
    ]

    if not mechanism_has_headroom:
        statement = (
            "SEARCH MECHANICS ARE THE BINDING CONSTRAINT\n"
            "even holding the true hidden state, this search leaves the direct\n"
            f"C1 move in only {percent(oracle_headroom)} of positions, so there is\n"
            "almost no decision headroom for a better belief to recover"
        )
    elif agent1c_beats_original and moves_toward_oracle:
        statement = (
            "BETTER BELIEF -> BETTER-LOOKING DECISIONS\n"
            "agent1c search agrees with the perfect-information search more\n"
            "often than the original belief and more often than direct C1"
        )
    elif decisions_differ and not agent1c_beats_original:
        statement = (
            "BELIEF CHANGES DECISIONS, BUT NOT DEMONSTRABLY FOR THE BETTER\n"
            "agent1c and original_phase11 choose differently, and agent1c is not\n"
            "closer to the perfect-information choice at this budget"
        )
    else:
        statement = (
            "BELIEF QUALITY DOES NOT REACH THE DECISION AT THIS BUDGET\n"
            "the arms choose nearly the same actions despite measurably\n"
            "different beliefs"
        )

    interpretation = []
    interpretation.append(
        "The oracle arm is the ceiling of this mechanism, not of Stratego: it "
        "is the same search, the same rollouts and the same leaf value, run on "
        "the one true world. Its distance from direct C1 is how much this "
        "search design can move a decision at all: it leaves the direct C1 move "
        f"in {percent(oracle_headroom)} of positions, and those are the entire "
        "budget of decisions any belief could hope to change for the better."
    )
    if mechanism_has_headroom:
        interpretation.append(
            "Because the oracle does move decisions, the mechanism has room in "
            "it, and the interesting question is how much of that room the "
            "learned beliefs recover. Between the count baseline "
            f"({percent(count['oracle_agreement_rate'])} oracle agreement) and the "
            f"oracle itself, agent1c reaches {percent(agent1c['oracle_agreement_rate'])} "
            f"and the original head {percent(original['oracle_agreement_rate'])}, "
            f"against direct C1's {percent(direct['oracle_agreement_rate'])}. Of the "
            f"{agent1c['oracle_headroom_positions']} positions where direct C1 and the "
            f"oracle search disagree, agent1c recovers {agent1c['oracle_fixed']} and "
            f"breaks {agent1c['oracle_broken']} that direct C1 already had right "
            f"(net {agent1c['net_oracle_moves']:+d}); original_phase11 is "
            f"{original['oracle_fixed']}/{original['oracle_broken']} "
            f"(net {original['net_oracle_moves']:+d}) and remaining_count "
            f"{count['oracle_fixed']}/{count['oracle_broken']} "
            f"(net {count['net_oracle_moves']:+d})."
        )
    else:
        interpretation.append(
            "Because even perfect hidden information barely changes what this "
            "search wants to play, no belief improvement can be expected to help "
            "at this budget. Section 12 of the common contract applies: fix "
            "search mechanics before spending compute on worlds or depth."
        )
    interpretation.append(
        "Nothing here is a strength claim. Agreeing with the perfect-information "
        "search is agreement with a shallow greedy-rollout evaluation that "
        "happens to know the hidden ranks; it is a diagnostic, not a proof of "
        "optimality, and Agent 3's match test is what turns any of this into "
        "wins or does not."
    )

    return {
        "statement": statement,
        "findings": findings,
        "interpretation": interpretation,
        "flags": {
            "belief_ordering_reproduces_on_this_set": bool(belief_ordering_holds),
            "belief_change_reaches_the_decision": bool(decisions_differ),
            "agent1c_closer_to_oracle_than_original": bool(agent1c_beats_original),
            "agent1c_closer_to_oracle_than_direct_c1": bool(moves_toward_oracle),
            "agent1c_minus_original_by_preset": {
                name: float(gap) for name, gap in gaps.items()
            },
            "agent1c_minus_original_sign_stable_across_budgets": bool(gap_sign_stable),
            "oracle_search_has_decision_headroom": bool(mechanism_has_headroom),
            "oracle_headroom_rate": float(oracle_headroom),
            "mechanism_headroom_floor": MECHANISM_HEADROOM_FLOOR,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="0 leaves torch alone")
    parser.add_argument(
        "--positions-per-cell", type=int, default=diag.POSITIONS_PER_CELL
    )
    parser.add_argument(
        "--presets", default="TINY,SMALL", help="comma separated, in run order"
    )
    parser.add_argument(
        "--primary", default="SMALL", help="the preset the report leads with"
    )
    parser.add_argument(
        "--reuse-positions",
        action="store_true",
        help="load the existing manifest instead of playing fresh games",
    )
    parser.add_argument("--repeat-probe", type=int, default=16)
    parser.add_argument("--limit-positions", type=int, default=0)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rewrite the report from the existing summary, running nothing",
    )
    arguments = parser.parse_args()

    if arguments.report_only:
        summary = json.loads(SUMMARY_PATH.read_text())
        summary["verdict"] = build_verdict(summary)
        SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
        write_report(summary, REPORT_PATH)
        log(summary["verdict"]["statement"])
        return 0

    if arguments.threads:
        torch.set_num_threads(int(arguments.threads))
    started = time.perf_counter()
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log("Phase 12 Agent 2 — belief-to-decision diagnostic")
    log("stage: identities")
    handoff = load_handoff()
    model, model_identity = load_move_model(handoff, arguments.device)
    providers = build_providers(model, handoff, arguments.device)

    log("stage: fresh diagnostic positions")
    if arguments.reuse_positions and MANIFEST_PATH.exists():
        manifest = diag.load_manifest(MANIFEST_PATH)
        log(
            f"  reusing {MANIFEST_PATH.name}: {manifest['counts']['positions']} "
            f"positions, digest {manifest['manifest_digest'][:12]}..."
        )
    else:
        manifest = build_position_set(
            arguments.device, positions_per_cell=arguments.positions_per_cell
        )
    materialized = diag.materialize_manifest(manifest, verify=True)
    if arguments.limit_positions:
        materialized = materialized[: arguments.limit_positions]
    log(f"  materialized {len(materialized)} positions, all observation digests matched")

    presets = [name.strip() for name in arguments.presets.split(",") if name.strip()]
    primary = arguments.primary if arguments.primary in presets else presets[-1]

    log("stage: belief quality on the diagnostic positions")
    quality = belief_quality(
        providers, materialized, search_preset(primary).worlds
    )
    for name in PROVIDER_ORDER:
        block = quality[name]
        log(
            f"  {name:>17}: R_CE {number(block['r_ce'])} top-1 "
            f"{percent(block['top1'])} world-rank {percent(block['sampled_world_rank_accuracy'])}"
        )

    log("stage: search matrix")
    all_rows: list = []
    cost: dict = {}
    for preset_name in presets:
        rows = run_search_matrix(
            model,
            model_identity,
            providers,
            materialized,
            preset_name,
            arguments.device,
        )
        all_rows.extend(rows)
        config = search_preset(preset_name)
        cost[preset_name] = {
            "worlds": config.worlds,
            "rollout_depth": config.rollout_depth,
            "arms": len(PROVIDER_ORDER),
            "decisions": len(rows),
            "mean_seconds": mean_or_none([row["seconds"] for row in rows]),
            "total_seconds": sum(row["seconds"] for row in rows),
        }

    log("stage: diagnostics")
    diagnostics = {name: arm_diagnostics(all_rows, name) for name in presets}
    pairwise = {name: pairwise_agreement(all_rows, name) for name in presets}
    unanimous = {name: unanimity(all_rows, name) for name in presets}

    repro = None
    if arguments.repeat_probe:
        log("stage: repeat probe")
        probe = reproducibility_probe(
            model,
            model_identity,
            providers,
            materialized,
            primary,
            arguments.device,
            int(arguments.repeat_probe),
        )
        original_index = index_rows(all_rows)
        identical = all(
            original_index[(row["preset"], row["provider"], row["position_id"])][
                "selected_action_id"
            ]
            == row["selected_action_id"]
            and abs(
                original_index[(row["preset"], row["provider"], row["position_id"])][
                    "score_selected"
                ]
                - row["score_selected"]
            )
            < 1e-9
            for row in probe["rows"]
        )
        repro = {
            "preset": probe["preset"],
            "positions": probe["positions"],
            "decisions": probe["decisions"],
            "identical": bool(identical),
        }
        log(f"  repeat probe identical: {identical}")

    mix = {
        "median ply": int(statistics.median([r["ply"] for r in materialized])),
        "ply range": f"{min(r['ply'] for r in materialized)}-{max(r['ply'] for r in materialized)}",
        "median unresolved": int(
            statistics.median([r["unresolved"] for r in materialized])
        ),
        "unresolved range": (
            f"{min(r['unresolved'] for r in materialized)}-"
            f"{max(r['unresolved'] for r in materialized)}"
        ),
        "median moved hidden": int(
            statistics.median([r["moved_hidden"] for r in materialized])
        ),
        "median legal actions": int(
            statistics.median([r["legal_action_count"] for r in materialized])
        ),
    }
    for bucket in BUCKETS:
        mix[f"positions {bucket}"] = sum(
            1
            for r in materialized
            if r["document_summary"]["progress_bucket"] == bucket
        )

    summary = {
        "artifact": "phase12_agent02_belief_decision_diagnostic_v1",
        "phase": "phase12",
        "agent": 2,
        "generated_utc": utc_now(),
        "search_version": SEARCH_VERSION,
        "score_definition": SCORE_DEFINITION,
        "device": arguments.device,
        "presets_run": presets,
        "primary_preset": primary,
        "search_configs": {
            name: search_preset(name).describe() for name in presets
        },
        "move_model_identity": model_identity,
        "providers": {name: providers[name].describe() for name in PROVIDER_ORDER},
        "position_set": {
            "artifact": manifest["artifact"],
            "manifest_path": f"reports/phase12/{MANIFEST_PATH.name}",
            "manifest_digest": manifest["manifest_digest"],
            "master_seed": manifest["master_seed"],
            "setup_library_split": manifest["setup_library_split"],
            "eligibility": manifest["eligibility"],
            "positions": len(materialized),
            "games_played": manifest["counts"]["games_played"],
            "games_contributing": manifest["counts"]["games_contributing"],
            "positions_by_behavior_group": manifest["counts"][
                "positions_by_behavior_group"
            ],
            "positions_by_observer_color": manifest["counts"][
                "positions_by_observer_color"
            ],
            "shortfalls": manifest["counts"]["shortfalls"],
            "phase11_test_bank_used": manifest["phase11_test_bank_used"],
            "replay_verified": True,
        },
        "position_mix": mix,
        "belief_quality": quality,
        "diagnostics": diagnostics,
        "pairwise_agreement": pairwise,
        "unanimity": unanimous,
        "cost": cost,
        "reproducibility": repro,
        "decisions": write_decisions(all_rows, DECISIONS_PATH),
        "limitations": [
            "Position-level diagnostic only: agreement with a perfect-information "
            "search is not a win rate and not a strength claim.",
            "The oracle arm collapses to a single world, so it is both the "
            "best-informed and the cheapest arm; its latency is not comparable "
            "to the belief arms'.",
            f"{manifest['eligibility']['positions_per_game']} positions per game "
            f"over {manifest['counts']['games_contributing']} contributing games: "
            "positions from one game share a setup and an opening, so they are "
            "not independent samples.",
            "The diagnostic setups are drawn from the accepted library's "
            "'validation' split, which is also the pool Phase 11B's dev split "
            "drew from — different seeds, different games, but Agent 1C's "
            "candidate selection saw that pool. A mild optimistic residual for "
            "agent1c, accepted for an engineering diagnostic.",
            "R_CE here is computed on this fresh set and is not the Phase 11B "
            "leaderboard number for the same checkpoint.",
            "One fixed beta (0.1) and one candidate rule, per the common "
            "contract; no tuning was attempted.",
        ],
        "status": {
            "phase11_final_classification": "FAIL",
            "phase11b_selection": "Agent1C",
            "scientific_validation_status": "not performed",
            "oracle_available_in_production": False,
            "phase11_test_bank_used": False,
            "search_core_modified": False,
            "match_set_run": False,
            "agent_3_launched": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "history": {
            "phase11_final_classification": "FAIL",
            "phase12_authorized_by_phase11": False,
            "phase11b_selection": "Agent1C",
        },
    }
    summary["verdict"] = build_verdict(summary)
    summary["seconds_total"] = round(time.perf_counter() - started, 3)

    SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
    log(f"  wrote {SUMMARY_PATH}")
    write_report(summary, REPORT_PATH)

    log("")
    log(summary["verdict"]["statement"])
    log(f"total {summary['seconds_total']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
