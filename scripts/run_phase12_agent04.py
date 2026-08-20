#!/usr/bin/env python
"""Phase 12 Agent 4 runner: how much search is worth paying for.

Specification source: `05_PHASE_12_AGENT_4_BUDGET_SCALING.md`.

The question, and only that question
------------------------------------
Agent 3 found that search at SMALL beats the direct accepted Phase 9 player
with every belief provider tried. What it could not do is separate the
providers, or say what budget is worth its latency. From here Agent 1C is
*the* production belief provider, so this run holds the provider fixed and
moves one variable — the budget.

1. Build the match pack: 64 boards, the same four accepted opponent
   behaviours, the same two accepted setup sources, balanced colours, the
   accepted library's `validation` split, never the spent Phase 11 bank.
   Agent 3's 32 boards are ordinals 0-1 of this set, so the SMALL rung
   replays them exactly and the two agents can be checked against each
   other rather than merely compared.
2. Play every board with every rung: the direct accepted Phase 9 seat as
   the zero-search anchor, then Agent 1C search at TINY, SMALL and MEDIUM.
   A ladder, not a grid: beta, the candidate rule and the search version
   are the same at every rung.
3. Report the section 4 metrics per rung, apply the section 5 stopping
   rule condition by condition, and name one practical operating point.

Board-major, resumable, single process
--------------------------------------
Board-major so a run stopped early still has every rung on exactly the same
boards. Single process because per-move latency is half of what this agent
is measuring, and latency taken while four workers contend for ten cores is
not the latency a player would see. That costs wall clock and buys a number
the report can stand behind.

Nothing accepted is modified. The Agent 1 search core, the Agent 3 match
apparatus and the accepted evaluation stack are read-only inputs; the only
new module is the budget ladder this agent needs to state its rule in.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.belief.phase11b.corpus import Phase11BSetupSources  # noqa: E402
from stratego.belief.phase11b.features import load_frozen_c1  # noqa: E402
from stratego.evaluation.neural_worker import (  # noqa: E402
    DECISION_MODE_GREEDY,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from stratego.evaluation.policy import PolicyRef  # noqa: E402
from stratego.training.phase10_acceptance import effective_win_rate  # noqa: E402
from stratego.search.phase12 import (  # noqa: E402
    PROVIDER_AGENT1C,
    Phase12SearchEngine,
    Phase12SearchError,
    SEARCH_VERSION,
    build_belief_provider,
)
from stratego.search.phase12.contract import SCORE_DEFINITION  # noqa: E402
from stratego.search.phase12 import budget as bd  # noqa: E402
from stratego.search.phase12 import matchplay as mp  # noqa: E402

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase12"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase12"
HANDOFF_PATH = REPOSITORY_ROOT / "reports" / "phase11b" / "phase12_handoff.json"
CONFIG_PATH = REPORT_DIRECTORY / "agent_04_ladder_config.json"
GAMES_JSONL_PATH = REPORT_DIRECTORY / "agent_04_games.jsonl"
GAMES_CSV_PATH = REPORT_DIRECTORY / "agent_04_games.csv"
REPORT_PATH = REPORT_DIRECTORY / "agent_04_report.md"
SUMMARY_PATH = REPORT_DIRECTORY / "agent_04_summary.json"

#: Agent 2's verified position manifest, reused for the cost profile, and
#: this agent's own profile output.
AGENT2_MANIFEST_PATH = REPORT_DIRECTORY / "agent_02_position_manifest.json"
PROFILE_PATH = REPORT_DIRECTORY / "agent_04_profile.json"

#: Agent 3's game rows, used for the cross-agent reproduction check.
AGENT3_GAMES_PATH = REPORT_DIRECTORY / "agent_03_games.jsonl"
AGENT3_SMALL_ARM = mp.ARM_AGENT1C.arm_id

#: 16 games per opponent — 64 boards, twice Agent 3's set, with Agent 3's
#: boards as ordinals 0-1. Agent 3's own limitation was sample size, not
#: world count: three belief providers there spread one game of 32. The
#: budget question deserves the extra games more than it deserves a fourth
#: rung.
GAMES_PER_OPPONENT = 16

GROUP_LABELS = {
    "phase9_selfplay": "Phase 9 direct",
    "strategic_rule": "Strategic",
    "tactical_rule": "Tactical",
    "scout_rush": "Scout-rush",
}


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize(value):
    """JSON-safe: numpy scalars, tuples, sets and Paths."""
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, set):
        return sorted(sanitize(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if hasattr(value, "item"):
        return sanitize(value.item())
    return str(value)


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def median_or_none(values):
    values = [float(value) for value in values if value is not None]
    return statistics.median(values) if values else None


def quantile(values, fraction: float):
    """The `fraction` quantile by nearest-rank, robust to tiny samples."""
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    index = max(0, min(len(values) - 1, int(round(fraction * (len(values) - 1)))))
    return values[index]


# ---------------------------------------------------------------------------
# Stage: identities and seats
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


def build_agent1c_provider(model, handoff: dict, device: str):
    """The one production belief provider this agent uses, digest-checked."""
    record = handoff["agent1c_checkpoint"]
    return build_belief_provider(
        PROVIDER_AGENT1C,
        encoder=model,
        agent1c_checkpoint=REPOSITORY_ROOT / record["path"],
        expected_agent1c_sha256=record["sha256"],
        expected_agent1c_state_digest=record["state_dict_digest"],
        production=True,
        device=device,
    )


def build_ladder_seats(arms, *, model, identity, provider, owners, device: str):
    """One seat per rung. Every search rung shares one provider instance.

    Sharing is safe and deliberate: the provider is stateless between
    calls and seeded per decision, so two rungs asking for worlds at the
    same board and ply get the same world stream — the same common random
    numbers the arms of Agent 3 shared, now across budgets. It also means
    the belief model is loaded and digest-checked exactly once.
    """
    seats = {}
    for arm in arms:
        if arm.kind == "direct":
            seats[arm.arm_id] = mp.DirectSeat(arm, owners)
            continue
        config = bd.ladder_config(bd.preset_of_arm(arm.arm_id))
        if config.production is not True:  # pragma: no cover - defensive
            raise Phase12SearchError(f"{arm.arm_id}: a ladder rung must be production")
        engine = Phase12SearchEngine(
            model, provider, config, device=device, model_identity=identity
        )
        seats[arm.arm_id] = mp.SearchSeat(arm, engine)
    return seats


def probe_reference(owners) -> RemoteNeuralPolicy:
    """The accepted direct player, used only inside :class:`SeatProbe`."""
    return RemoteNeuralPolicy(
        PolicyRef("phase12_budget_probe_reference_v1", mp.MATCH_VERSION),
        LocalInferenceChannel(owners["phase9"]),
        decision_mode=DECISION_MODE_GREEDY,
    )


# ---------------------------------------------------------------------------
# Stage: play
# ---------------------------------------------------------------------------


def game_payload(record: mp.GameRecord) -> dict:
    """One resumable JSONL row: the flat result plus the per-move arrays.

    `move_worlds` is Agent 4's addition to Agent 3's payload — worlds per
    move is one of the section 4 metrics, and after de-duplication it is a
    measurement rather than a restatement of the preset.
    """
    payload = record.row()
    payload["move_seconds"] = [round(float(row["seconds"]), 5) for row in record.moves]
    payload["move_forwards"] = [int(row["c1_forwards"] or 0) for row in record.moves]
    payload["move_legal_actions"] = [int(row["legal_actions"]) for row in record.moves]
    payload["move_changed"] = [
        None if row["move_changed"] is None else int(row["move_changed"])
        for row in record.moves
    ]
    payload["move_worlds"] = [
        None if row.get("unique_worlds") is None else int(row["unique_worlds"])
        for row in record.moves
    ]
    return payload


def load_completed(path: Path) -> dict:
    """Rows already on disk, keyed by `(board_id, arm_id)`."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["board_id"], row["arm_id"])] = row
    return rows


def play_stage(plans, arms, seats, owners, *, probes, resume: bool, max_seconds: float):
    """Play every board with every rung, board-major, streaming to JSONL."""
    completed = load_completed(GAMES_JSONL_PATH) if resume else {}
    if completed:
        log(f"  resuming: {len(completed)} game rows already on disk")
    elif GAMES_JSONL_PATH.exists():
        GAMES_JSONL_PATH.unlink()

    rows: list = list(completed.values())
    started = time.perf_counter()
    played = 0
    stopped_early = False
    boards_done = 0

    with GAMES_JSONL_PATH.open("a") as stream:
        for plan_index, plan in enumerate(plans):
            elapsed = time.perf_counter() - started
            if max_seconds and elapsed > max_seconds and plan_index:
                log(
                    f"  wall-clock cap reached after {boards_done}/{len(plans)} boards "
                    f"({elapsed:.0f}s); stopping at a board boundary"
                )
                stopped_early = True
                break
            outcomes = []
            for arm in arms:
                key = (plan.board_id, arm.arm_id)
                if key in completed:
                    outcomes.append((arm, completed[key]))
                    continue
                record = mp.play_arm_game(
                    plan, seats[arm.arm_id], owners, probe=probes.get(arm.arm_id)
                )
                payload = game_payload(record)
                stream.write(json.dumps(payload) + "\n")
                stream.flush()
                rows.append(payload)
                outcomes.append((arm, payload))
                played += 1
            boards_done += 1
            elapsed = time.perf_counter() - started
            log(
                f"  [{boards_done}/{len(plans)}] {plan.stratum}/{plan.setup_source}/"
                f"{plan.player_color}/g{plan.ordinal:02d} "
                + " ".join(
                    f"{arm.arm_id.replace('search_agent1c_', '').replace('_c1', '')}="
                    f"{row['outcome'][0].upper()}"
                    for arm, row in outcomes
                )
                + f"  {elapsed:.0f}s"
            )
    log(f"  played {played} games this run, {len(rows)} rows total")
    return rows, stopped_early


# ---------------------------------------------------------------------------
# Stage: analysis
# ---------------------------------------------------------------------------


def tally(rows) -> dict:
    scores = [float(row["effective_score"]) for row in rows]
    return {
        "games": len(rows),
        "wins": sum(1 for row in rows if row["outcome"] == "win"),
        "draws": sum(1 for row in rows if row["outcome"] == "draw"),
        "losses": sum(1 for row in rows if row["outcome"] == "loss"),
        "ewr": effective_win_rate(scores) if scores else None,
    }


def arm_rows(rows, arm_id) -> list:
    return sorted(
        (row for row in rows if row["arm_id"] == arm_id), key=lambda row: row["board_id"]
    )


def paired_comparison(mine, other) -> dict:
    """Per-board score differences against another rung on the same boards."""
    index = {row["board_id"]: row for row in other}
    deltas = []
    better = same = worse = 0
    for row in mine:
        partner = index.get(row["board_id"])
        if partner is None:
            continue
        delta = float(row["effective_score"]) - float(partner["effective_score"])
        deltas.append(delta)
        if delta > 0:
            better += 1
        elif delta < 0:
            worse += 1
        else:
            same += 1
    if not deltas:
        return {"boards": 0}
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "boards": len(deltas),
        "mean_score_delta": statistics.fmean(deltas),
        "stdev": stdev,
        "standard_error": (stdev / (len(deltas) ** 0.5)) if len(deltas) > 1 else None,
        "better": better,
        "same": same,
        "worse": worse,
    }


def arm_summary(rows, arm, *, direct_rows, contested_boards) -> dict:
    """Every section 4 metric for one rung, plus the slices that read it."""
    mine = arm_rows(rows, arm.arm_id)
    block = {
        "arm_id": arm.arm_id,
        "label": arm.label,
        "kind": arm.kind,
        "provider_id": arm.provider_id,
    }
    block.update(tally(mine))
    block["by_opponent"] = {
        stratum: tally([row for row in mine if row["stratum"] == stratum])
        for stratum in mp.MATCH_STRATA
    }
    block["by_color"] = {
        color: tally([row for row in mine if row["player_color"] == color])
        for color in mp.MATCH_COLORS
    }
    block["by_setup_source"] = {
        source: tally([row for row in mine if row["setup_source"] == source])
        for source in mp.MATCH_SOURCES
    }

    latencies = [second for row in mine for second in row["move_seconds"]]
    forwards = [count for row in mine for count in row["move_forwards"]]
    worlds = [
        count for row in mine for count in row.get("move_worlds", []) if count is not None
    ]
    changed = [flag for row in mine for flag in row["move_changed"] if flag is not None]
    decisions = sum(int(row["player_decisions"]) for row in mine)
    player_seconds = sum(float(row["player_seconds"]) for row in mine)
    game_seconds = [float(row["seconds"]) for row in mine]

    block["plies_mean"] = mean_or_none([row["plies"] for row in mine])
    block["plies_median"] = median_or_none([row["plies"] for row in mine])
    block["player_decisions"] = decisions
    block["player_decisions_per_game"] = decisions / len(mine) if mine else None
    block["game_seconds_mean"] = mean_or_none(game_seconds)
    block["game_seconds_median"] = median_or_none(game_seconds)
    block["games_per_hour"] = (
        3600.0 / statistics.fmean(game_seconds) if game_seconds else None
    )
    block["player_seconds_total"] = player_seconds
    block["search_seconds_per_game"] = player_seconds / len(mine) if mine else None
    block["move_seconds_mean"] = mean_or_none(latencies)
    block["move_seconds_median"] = median_or_none(latencies)
    block["move_seconds_p95"] = quantile(latencies, 0.95)
    block["move_seconds_p99"] = quantile(latencies, 0.99)
    block["move_seconds_max"] = max(latencies) if latencies else None
    block["search_calls"] = decisions if arm.kind == "search" else 0
    block["c1_forwards_total"] = sum(forwards)
    block["c1_forwards_per_move"] = mean_or_none(forwards)
    block["forward_positions_per_second"] = (
        sum(forwards) / player_seconds if player_seconds else None
    )
    block["worlds_per_move"] = mean_or_none(worlds)
    block["move_change_rate"] = (sum(changed) / len(changed)) if changed else None
    block["move_change_rate_by_opponent"] = {}
    for stratum in mp.MATCH_STRATA:
        flags = [
            flag
            for row in mine
            if row["stratum"] == stratum
            for flag in row["move_changed"]
            if flag is not None
        ]
        block["move_change_rate_by_opponent"][stratum] = (
            (sum(flags) / len(flags)) if flags else None
        )

    contested = [row for row in mine if row["board_id"] in contested_boards]
    block["contested"] = tally(contested)
    reasons: dict = {}
    for row in mine:
        reasons[row["terminal_reason"]] = reasons.get(row["terminal_reason"], 0) + 1
    block["terminal_reasons"] = dict(sorted(reasons.items(), key=lambda item: -item[1]))
    block["paired_vs_direct"] = paired_comparison(mine, direct_rows)
    block["outcome_standard_error"] = outcome_noise_scale(mine)
    return block


def outcome_noise_scale(rows) -> "float | None":
    """The standard error of one rung's EWR at this sample size.

    Descriptive, not an inference: no significance claim is made anywhere
    in this agent. It is here so the ladder's steps can be read against the
    resolution of the match pack that produced them.
    """
    scores = [float(row["effective_score"]) for row in rows]
    if len(scores) < 2:
        return None
    return statistics.stdev(scores) / (len(scores) ** 0.5)


def contested_board_set(rows, arms) -> dict:
    """Boards whose result the player's own decisions could have changed.

    A board where the seat never got a decision — the opponent's opening
    reached a flag first — returns the same result for every rung by
    construction. Those boards are not dropped from the headline EWR: they
    are real games of the match pack and every rung is charged for them
    equally. They are counted here because they bound what the pack can
    resolve, and because a reader comparing two rungs should know how many
    of the boards were never in play.
    """
    board_ids = sorted({row["board_id"] for row in rows})
    arm_ids = {arm.arm_id for arm in arms}
    by_board: dict = {board: [] for board in board_ids}
    for row in rows:
        by_board[row["board_id"]].append(row)

    uncontested, quick = [], []
    for board in board_ids:
        here = by_board[board]
        if {row["arm_id"] for row in here} != arm_ids:
            continue
        decisions = [int(row["player_decisions"]) for row in here]
        if max(decisions) == 0:
            uncontested.append(board)
        if max(decisions) <= 3:
            quick.append(board)
    contested = [board for board in board_ids if board not in set(uncontested)]
    return {
        "boards": len(board_ids),
        "contested": set(contested),
        "contested_boards": len(contested),
        "uncontested_boards": len(uncontested),
        "uncontested_board_ids": uncontested,
        "boards_decided_within_three_player_decisions": len(quick),
        "quick_board_ids": quick,
    }


def budget_points(summary_by_arm: dict, arms, probes: dict) -> "list[bd.BudgetPoint]":
    """One :class:`BudgetPoint` per rung, in ladder order."""
    points = []
    for arm in arms:
        if arm.kind != "search":
            continue
        block = summary_by_arm[arm.arm_id]
        config = bd.ladder_config(bd.preset_of_arm(arm.arm_id))
        instability = []
        probe = probes.get(arm.arm_id)
        if probe and probe.get("failures"):
            instability.append(
                f"{len(probe['failures'])} match-time boundary probe failure(s)"
            )
        points.append(
            bd.BudgetPoint(
                preset_id=config.preset_id,
                worlds=config.worlds,
                rollout_depth=config.rollout_depth,
                max_root_candidates=config.max_root_candidates,
                games=block["games"],
                ewr=block["ewr"],
                move_seconds_median=block["move_seconds_median"],
                move_seconds_p95=block["move_seconds_p95"],
                search_seconds_per_game=block["search_seconds_per_game"],
                forwards_per_move=block["c1_forwards_per_move"],
                unstable=bool(instability),
                instability=tuple(instability),
            )
        )
    return points


def reproduction_check(rows) -> dict:
    """Does the SMALL rung replay Agent 3's `search_agent1c` arm exactly?

    Agent 3's 32 boards are ordinals 0-1 of this pack and every seed in the
    match apparatus is a pure function of the board and the ply, so the
    SMALL rung should reproduce Agent 3's agent1c games move for move. This
    is the cheapest available check that the two agents are running the
    same system, and it costs nothing: the games are played anyway.
    """
    if not AGENT3_GAMES_PATH.exists():
        return {"available": False, "reason": f"{AGENT3_GAMES_PATH.name} not found"}
    theirs = {}
    for line in AGENT3_GAMES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["arm_id"] == AGENT3_SMALL_ARM:
            theirs[row["board_id"]] = row
    mine = {
        row["board_id"]: row
        for row in rows
        if row["arm_id"] == bd.ladder_arm("SMALL").arm_id
    }
    shared = sorted(set(theirs) & set(mine))
    fields = ("outcome", "plies", "player_decisions", "c1_forwards", "move_changes")
    mismatches = []
    for board in shared:
        here, there = mine[board], theirs[board]
        differing = {
            field: [there[field], here[field]]
            for field in fields
            if there[field] != here[field]
        }
        if differing:
            mismatches.append({"board_id": board, "fields": differing})
    return {
        "available": True,
        "compared_arm": AGENT3_SMALL_ARM,
        "against": bd.ladder_arm("SMALL").arm_id,
        "shared_boards": len(shared),
        "fields_compared": list(fields),
        "identical_boards": len(shared) - len(mismatches),
        "mismatches": mismatches,
        "reproduced": bool(shared) and not mismatches,
    }


# ---------------------------------------------------------------------------
# Stage: cost profile
# ---------------------------------------------------------------------------


def profile_stage(configs, *, model, identity, provider, device: str, positions: int) -> dict:
    """Where each rung's decision time actually goes.

    Section 6 asks for a profile before any redesign, and the match run
    cannot supply one: it measures whole decisions, not their parts. This
    re-searches a fixed sample of Agent 2's manifest positions — replayed
    and observation-digest-verified by the accepted apparatus, so the
    positions are the ones Agent 2 studied rather than fresh playouts — and
    reports the split for every rung on the same positions.

    Run separately from the match, single process and otherwise idle, for
    the same reason the match is: a fraction measured under contention is a
    fraction of the wrong denominator.
    """
    from stratego.search.phase12 import positions as pos

    manifest = pos.load_manifest(AGENT2_MANIFEST_PATH)
    materialized = pos.materialize_manifest(manifest, verify=True)
    sample = pos.spread(materialized, positions)
    log(
        f"  {len(sample)} positions from {AGENT2_MANIFEST_PATH.name} "
        f"({len(materialized)} available, observation digests verified)"
    )

    rungs: dict = {}
    for config in configs:
        engine = Phase12SearchEngine(
            model, provider, config, device=device, model_identity=identity
        )
        seconds, forward, observation, forwards, worlds = [], [], [], [], []
        for entry in sample:
            decision = engine.choose_action(
                entry["state"], seed=int(entry["search_seed"])
            )
            seconds.append(float(decision.seconds))
            forward.append(float(decision.forward_seconds))
            observation.append(float(decision.observation_seconds))
            forwards.append(int(decision.c1_forwards))
            worlds.append(int(decision.unique_worlds))
        total = sum(seconds)
        rungs[config.preset_id] = {
            "preset_id": config.preset_id,
            "worlds": config.worlds,
            "rollout_depth": config.rollout_depth,
            "positions": len(sample),
            "seconds_mean": mean_or_none(seconds),
            "seconds_median": median_or_none(seconds),
            "forward_fraction": (sum(forward) / total) if total else None,
            "observation_fraction": (sum(observation) / total) if total else None,
            "other_fraction": (
                (total - sum(forward) - sum(observation)) / total if total else None
            ),
            "c1_forwards_per_move": mean_or_none(forwards),
            "unique_worlds_per_move": mean_or_none(worlds),
            "forward_positions_per_second": (sum(forwards) / total) if total else None,
        }
        log(
            f"  {config.preset_id:<7} {rungs[config.preset_id]['seconds_mean']:.3f} s/move  "
            f"forward {rungs[config.preset_id]['forward_fraction']:.2f}  "
            f"observation {rungs[config.preset_id]['observation_fraction']:.2f}  "
            f"other {rungs[config.preset_id]['other_fraction']:.2f}"
        )
    return {
        "artifact": "phase12_agent04_cost_profile_v1",
        "generated_utc": utc_now(),
        "source_manifest": str(AGENT2_MANIFEST_PATH.relative_to(REPOSITORY_ROOT)),
        "manifest_digest": manifest.get("manifest_digest"),
        "belief_provider": bd.LADDER_PROVIDER,
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "rungs": rungs,
        "bottleneck": max(
            ("c1_forward_passes", "observation_building", "other"),
            key=lambda name: sum(
                row[
                    {
                        "c1_forward_passes": "forward_fraction",
                        "observation_building": "observation_fraction",
                        "other": "other_fraction",
                    }[name]
                ]
                or 0.0
                for row in rungs.values()
            ),
        ),
    }


def load_profile() -> "dict | None":
    if not PROFILE_PATH.exists():
        return None
    profile = json.loads(PROFILE_PATH.read_text())
    if profile.get("artifact") != "phase12_agent04_cost_profile_v1":
        return None
    return profile


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------


def percent(value) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def number(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def signed(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):+.{digits}f}"


def record_text(block: dict) -> str:
    return f"{block['wins']} / {block['draws']} / {block['losses']}"


def count_text(count: int, noun: str = "game") -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def table(header, rows) -> "list[str]":
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    lines.append("")
    return lines


def games_text(delta: float, games: int) -> str:
    """An EWR delta restated as games of the pack, which is what it is."""
    return f"{abs(delta) * games:.1f} game{'' if abs(delta) * games == 1 else 's'} of {games}"


def write_report(summary: dict, path: Path) -> None:
    ladder = summary["ladder"]
    arms = summary["arms"]
    order = summary["arm_order"]
    rule = summary["stopping_rule"]
    pack = summary["match_pack"]
    games = pack["boards"]
    direct = arms[mp.ARM_DIRECT.arm_id]
    verdict = summary["verdict"]

    lines: list = []
    lines.append("# Phase 12 Agent 4 — Search Budget Scaling with Agent 1C")
    lines.append("")
    lines.append(
        f"Generated {summary['generated_utc']} by `scripts/run_phase12_agent04.py`."
    )
    lines.append("")
    lines.append(
        "Engineering artifact of the Phase 12 rapid search-engineering phase. One "
        "provider, three budgets, one fixed match pack: no grid, no tuning, no sealed "
        "test bank, and no significance claim."
    )
    lines.append("")

    lines.append("## 1. Question and verdict")
    lines.append("")
    lines.append("```text")
    lines.append(verdict["headline"])
    lines.append(verdict["statement"])
    lines.append("```")
    lines.append("")
    lines.extend(verdict["bullets"])
    lines.append("")

    lines.append("## 2. Ladder configuration")
    lines.append("")
    lines.append("```text")
    configuration = summary["ladder_configuration"]
    for key, value in configuration.items():
        lines.append(f"{key:<24}{value}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Every rung plays the identical boards under identical opponent seeds and "
        "identical per-ply world seeds, with one shared Agent 1C provider instance, so "
        "two rungs differ only in the budget they spend on the same worlds."
    )
    lines.append("")

    lines.append("## 3. The ladder")
    lines.append("")
    lines.append(
        "Effective win rate is the accepted definition — the mean per-game score, win 1, "
        "draw 0.5, loss 0."
    )
    lines.append("")
    rows = [
        [
            direct["label"],
            "—",
            record_text(direct),
            number(direct["ewr"]),
            "—",
            "—",
            number(direct["move_seconds_median"], 3),
            number(direct["move_seconds_p95"], 3),
            f"{direct['c1_forwards_per_move']:.1f}",
            "—",
        ]
    ]
    for row in ladder:
        block = arms[row["arm_id"]]
        rows.append(
            [
                block["label"],
                f"{row['worlds']}w / d{row['rollout_depth']}",
                record_text(block),
                number(block["ewr"]),
                signed(row["ewr_gain_vs_reference"]),
                signed(row["delta_ewr_from_previous"]),
                number(block["move_seconds_median"], 3),
                number(block["move_seconds_p95"], 3),
                f"{block['c1_forwards_per_move']:.1f}",
                number(block["worlds_per_move"], 1),
            ]
        )
    lines.extend(
        table(
            [
                "rung",
                "budget",
                "W / D / L",
                "EWR",
                "vs direct",
                "delta from previous",
                "median s/move",
                "p95 s/move",
                "C1 fwd/move",
                "worlds/move",
            ],
            rows,
        )
    )
    lines.append(
        f"At {games} games per rung an EWR standard error of about "
        f"{number(summary['noise']['unpaired_standard_error'], 3)} is unavoidable, and "
        f"the paired per-board comparison carries about "
        f"{number(summary['noise']['paired_standard_error'], 3)}. The engineering margin "
        f"below which this agent refuses to read an ordering is "
        f"{number(rule['thresholds']['meaningful_ewr_gain'], 2)}."
    )
    lines.append("")

    lines.append("### Throughput and cost")
    lines.append("")
    rows = []
    for arm_id in order:
        block = arms[arm_id]
        rows.append(
            [
                block["label"],
                number(block["search_seconds_per_game"], 2),
                number(block["game_seconds_mean"], 1),
                number(block["games_per_hour"], 1),
                f"{block['player_decisions_per_game']:.1f}",
                number(block["forward_positions_per_second"], 0),
                percent(block["move_change_rate"]),
            ]
        )
    lines.extend(
        table(
            [
                "rung",
                "search s/game",
                "game wall-clock s",
                "games/hour",
                "player decisions/game",
                "fwd positions/s",
                "move-change vs direct",
            ],
            rows,
        )
    )

    lines.append("### Strength bought per unit of search time")
    lines.append("")
    rows = []
    for row in ladder:
        rows.append(
            [
                row["preset_id"],
                signed(row["ewr_gain_vs_reference"]),
                number(row["extra_search_seconds_vs_reference"], 1),
                signed(row["ewr_gain_per_search_second_vs_reference"], 5),
                signed(row["delta_ewr_from_previous"]),
                number(row["extra_search_seconds_per_game"], 1),
                signed(row["ewr_gain_per_extra_search_second"], 5),
            ]
        )
    lines.extend(
        table(
            [
                "rung",
                "EWR vs direct",
                "extra s/game vs direct",
                "EWR per search second",
                "delta EWR from previous",
                "extra s/game vs previous",
                "EWR per extra search second",
            ],
            rows,
        )
    )
    lines.append(
        "The two efficiency columns answer different questions: the first is what "
        "searching at all buys over playing directly, the second is what climbing one "
        "rung buys over the rung below it. The stopping rule reads the second."
    )
    lines.append("")

    lines.append("### EWR by opponent")
    lines.append("")
    rows = []
    for arm_id in order:
        block = arms[arm_id]
        rows.append(
            [block["label"]]
            + [
                number(block["by_opponent"][stratum]["ewr"], 3)
                for stratum in mp.MATCH_STRATA
            ]
            + [number(block["ewr"], 3)]
        )
    lines.extend(
        table(
            ["rung"] + [GROUP_LABELS[s] for s in mp.MATCH_STRATA] + ["overall"], rows
        )
    )

    lines.append("### EWR by colour and setup source")
    lines.append("")
    rows = []
    for arm_id in order:
        block = arms[arm_id]
        rows.append(
            [block["label"]]
            + [number(block["by_color"][color]["ewr"], 3) for color in mp.MATCH_COLORS]
            + [
                number(block["by_setup_source"][source]["ewr"], 3)
                for source in mp.MATCH_SOURCES
            ]
        )
    lines.extend(
        table(
            ["rung", "red", "blue"]
            + [f"{source} opponent" for source in mp.MATCH_SOURCES],
            rows,
        )
    )

    lines.append("## 4. Paired reading of the same boards")
    lines.append("")
    lines.append(
        "Every rung played every board, so the informative comparison is per board. A "
        "board both rungs resolved the same way carries no information about the "
        "difference between them."
    )
    lines.append("")
    rows = []
    for entry in summary["paired"]:
        rows.append(
            [
                entry["label"],
                entry["boards"],
                f"{entry['better']} / {entry['same']} / {entry['worse']}",
                signed(entry["mean_score_delta"]),
                number(entry["standard_error"], 3),
                games_text(entry["mean_score_delta"], entry["boards"]),
            ]
        )
    lines.extend(
        table(
            [
                "comparison",
                "boards",
                "better / same / worse",
                "paired delta",
                "standard error",
                "size of the delta",
            ],
            rows,
        )
    )

    lines.append("### Boards the seat never got to play")
    lines.append("")
    lines.append("```text")
    lines.append(f"boards in the pack                          {pack['boards']}")
    lines.append(
        f"decided before the player's first decision  {pack['uncontested_boards']}"
    )
    lines.append(
        f"decided within three player decisions       "
        f"{pack['boards_decided_within_three_player_decisions']}"
    )
    lines.append(
        f"contested boards                            {pack['contested_boards']}"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "These boards are kept in every headline number — they are real games and every "
        "rung is charged for them equally — but they are the same result for every rung "
        "by construction, so they set a floor on how much of the pack can separate two "
        "budgets. The cause is the setup library, not the search: "
        f"{pack['front_row_flag_boards']} of {pack['boards']} boards place a flag on a "
        "front row, where an opening scout down an open file can reach it."
    )
    lines.append("")
    rows = []
    for arm_id in order:
        block = arms[arm_id]
        rows.append(
            [
                block["label"],
                record_text(block),
                number(block["ewr"]),
                f"{block['contested']['wins']} / {block['contested']['draws']} / "
                f"{block['contested']['losses']}",
                number(block["contested"]["ewr"]),
            ]
        )
    lines.extend(
        table(
            [
                "rung",
                "W / D / L (all)",
                "EWR (all)",
                "W / D / L (contested)",
                "EWR (contested)",
            ],
            rows,
        )
    )

    lines.append("## 5. Stopping rule")
    lines.append("")
    lines.append("```text")
    for key, value in rule["thresholds"].items():
        lines.append(f"{key:<40}{value}")
    lines.append("```")
    lines.append("")
    rows = []
    for name, block in rule["conditions"].items():
        rows.append(
            [name.replace("_", " "), "FIRED" if block["fired"] else "no", block["reading"]]
        )
    lines.extend(table(["condition", "fired", "evidence"], rows))
    lines.append(f"```text\n{rule['statement']}\n```")
    lines.append("")

    lines.append("## 6. Selected practical operating point")
    lines.append("")
    lines.append("```text")
    for key, value in summary["operating_point_record"].items():
        lines.append(f"{key:<28}{value}")
    lines.append("```")
    lines.append("")
    lines.append(rule["operating_point"]["rule"] + ".")
    lines.append("")

    lines.append("## 7. Cost profile")
    lines.append("")
    profile = summary.get("profile")
    if not profile:
        lines.append(
            "Not run. Re-run `scripts/run_phase12_agent04.py --profile-only` and rebuild "
            "the report with `--analysis-only` to fill this section."
        )
        lines.append("")
    else:
        lines.append(
            "Section 6 asks for a profile before any redesign. Measured in a separate "
            "single-process pass over "
            f"{next(iter(profile['rungs'].values()))['positions']} positions from Agent "
            "2's manifest, replayed and observation-digest-verified by the accepted "
            "apparatus, so every rung is profiled on the same positions."
        )
        lines.append("")
        rows = [
            [
                row["preset_id"],
                number(row["seconds_mean"], 3),
                number(row["forward_fraction"], 2),
                number(row["observation_fraction"], 2),
                number(row["other_fraction"], 2),
                f"{row['c1_forwards_per_move']:.0f}",
                number(row["unique_worlds_per_move"], 1),
                number(row["forward_positions_per_second"], 0),
            ]
            for row in profile["rungs"].values()
        ]
        lines.extend(
            table(
                [
                    "rung",
                    "s/move",
                    "forward",
                    "observation",
                    "other",
                    "C1 fwd/move",
                    "unique worlds",
                    "fwd positions/s",
                ],
                rows,
            )
        )
        lines.append(
            f"Main observed bottleneck: {profile['bottleneck'].replace('_', ' ')}. The "
            "optimizations section 6 names are already in the Agent 1 engine — worlds "
            "are sampled once at the root and de-duplicated, the root C1 outputs are "
            "computed once and reused by every candidate, and rollout forwards are "
            "batched across all live (candidate, world) simulations at each ply — so "
            "there is no cheap structural win left to take, and none is needed to make "
            "the selected operating point viable."
        )
        lines.append("")

    lines.append("## 8. Cross-agent reproduction check")
    lines.append("")
    reproduction = summary["reproduction"]
    if not reproduction.get("available"):
        lines.append(f"Not run: {reproduction.get('reason')}.")
    else:
        lines.append("```text")
        lines.append(
            f"Agent 3 arm            {reproduction['compared_arm']}\n"
            f"Agent 4 rung           {reproduction['against']}\n"
            f"shared boards          {reproduction['shared_boards']}\n"
            f"identical boards       {reproduction['identical_boards']}\n"
            f"mismatching boards     {len(reproduction['mismatches'])}\n"
            f"fields compared        {', '.join(reproduction['fields_compared'])}"
        )
        lines.append("```")
        lines.append("")
        if reproduction["reproduced"]:
            lines.append(
                "The SMALL rung replayed Agent 3's `search_agent1c` games exactly — same "
                "outcome, same ply count, same decision count, same forward count, same "
                "move changes — on all "
                f"{reproduction['shared_boards']} boards the two packs share. The two "
                "agents are running the same system."
            )
        else:
            lines.append(
                f"{len(reproduction['mismatches'])} of {reproduction['shared_boards']} "
                "shared boards did not replay identically. That is a defect: with the "
                "same board, the same seeds and the same budget these games should be "
                "identical."
            )
    lines.append("")

    lines.append("## 9. Match-time boundary probe")
    lines.append("")
    lines.append(
        "Each seat was re-asked a sample of its own decisions on a state whose hidden "
        "opponent identities had been permuted by the accepted "
        "`permute_hidden_identities`, and required to answer identically; the search "
        "seats were additionally required to agree with the accepted direct player on "
        "what the direct Phase 9 action was."
    )
    lines.append("")
    rows = []
    for arm_id in order:
        probe = summary["probes"].get(arm_id)
        if probe is None:
            continue
        # A production seat that changed its answer under permutation is a
        # failure, not a sensitivity count, so the two live in different
        # fields; the column the reader wants is their sum.
        changed_answer = probe["permutation_sensitive"] + sum(
            1
            for failure in probe["failures"]
            if failure["check"] == "permutation_invariance"
        )
        rows.append(
            [
                arms[arm_id]["label"],
                probe["permutation_checks"],
                probe["permutation_assignments_changed"],
                changed_answer,
                probe["direct_agreement_checks"],
                len(probe["failures"]),
            ]
        )
    lines.extend(
        table(
            [
                "rung",
                "permutation checks",
                "assignments actually changed",
                "answer changed",
                "direct-agreement checks",
                "failures",
            ],
            rows,
        )
    )
    lines.append(
        "No production rung may change its answer under permutation; Agent 3's oracle "
        "arm was the positive control for that check and is not replayed here."
    )
    lines.append("")

    lines.append("## 10. Interpretation")
    lines.append("")
    for paragraph in summary["interpretation"]:
        lines.append(paragraph)
        lines.append("")

    lines.append("## 11. Limitations")
    lines.append("")
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 12. Deliverables and status")
    lines.append("")
    lines.append("```text")
    for item in summary["deliverables"]:
        lines.append(item)
    lines.append("")
    for key, value in summary["status"].items():
        lines.append(f"{key:<34}{value}")
    lines.append("```")
    lines.append("")
    lines.append(summary["stop_condition"])
    lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Stage: verdict and interpretation
# ---------------------------------------------------------------------------


def build_verdict(summary: dict) -> dict:
    """What the ladder showed, in the instruction's own terms.

    The verdict is derived from the stopping rule and the ladder table, not
    written over them: if the numbers change, this changes with them.
    """
    ladder = summary["ladder"]
    arms = summary["arms"]
    rule = summary["stopping_rule"]
    margin = rule["thresholds"]["meaningful_ewr_gain"]
    games = summary["match_pack"]["boards"]
    direct = arms[mp.ARM_DIRECT.arm_id]
    selected_id = rule["operating_point"]["selected_preset_id"]
    selected_arm = bd.ladder_arm(selected_id).arm_id
    selected = arms[selected_arm]
    selected_row = next(row for row in ladder if row["preset_id"] == selected_id)
    steps = [row for row in ladder if row["delta_ewr_from_previous"] is not None]
    biggest = max(
        (row for row in ladder),
        key=lambda row: row["ewr"],
    )
    spread = max(row["ewr"] for row in ladder) - min(row["ewr"] for row in ladder)
    any_step_meaningful = any(row["delta_ewr_from_previous"] >= margin for row in steps)
    all_above_direct = all(row["ewr_gain_vs_reference"] > 0 for row in ladder)

    if any_step_meaningful:
        headline = "CLIMBING THE LADDER BOUGHT STRENGTH"
    elif spread < margin:
        headline = "THE THREE BUDGETS DID NOT SEPARATE"
    else:
        headline = "THE LADDER IS NOT MONOTONE IN BUDGET"

    statement = (
        f"practical operating point {selected_id} "
        f"({selected_row['worlds']} worlds, depth {selected_row['rollout_depth']}, "
        f"{number(selected['move_seconds_median'], 3)} s/move median, EWR "
        f"{number(selected['ewr'])} against direct {number(direct['ewr'])})"
    )

    bullets = []
    bullets.append(
        f"- Direct accepted Phase 9 C1 scored EWR {number(direct['ewr'])} "
        f"({record_text(direct)}) over {count_text(games)}; "
        + ", ".join(
            f"{arms[row['arm_id']]['label']} {number(row['ewr'])} "
            f"({record_text(arms[row['arm_id']])})"
            for row in ladder
        )
        + "."
    )
    if all_above_direct:
        bullets.append(
            f"- Every rung finished above direct C1, by "
            f"{signed(min(row['ewr_gain_vs_reference'] for row in ladder))} to "
            f"{signed(max(row['ewr_gain_vs_reference'] for row in ladder))} EWR, which "
            "reproduces Agent 3's direction on twice the boards."
        )
    else:
        behind = [
            row
            for row in ladder
            if row["ewr_gain_vs_reference"] is not None
            and row["ewr_gain_vs_reference"] < 0
        ]
        level = [
            row for row in ladder if row["ewr_gain_vs_reference"] == 0
        ]
        parts = []
        if behind:
            parts.append(
                "behind it at "
                + ", ".join(
                    f"{row['preset_id']} {signed(row['ewr_gain_vs_reference'])}"
                    for row in behind
                )
            )
        if level:
            parts.append(
                "level with it at " + ", ".join(row["preset_id"] for row in level)
            )
        bullets.append("- Not every rung beat direct C1: " + "; ".join(parts) + ".")
    bullets.append(
        "- Step by step up the ladder: "
        + ", ".join(
            f"{row['previous_preset_id']} → {row['preset_id']} "
            f"{signed(row['delta_ewr_from_previous'])} EWR for "
            f"{number(row['extra_search_seconds_per_game'], 1)} more search seconds "
            "per game"
            for row in steps
        )
        + f". The whole ladder spans {number(spread)} EWR "
        f"({games_text(spread, games)}), against a {number(margin, 2)} engineering "
        "margin."
    )
    bullets.append(
        f"- Latency across the ladder: "
        + ", ".join(
            f"{row['preset_id']} {number(arms[row['arm_id']]['move_seconds_median'], 3)} s "
            f"median / {number(arms[row['arm_id']]['move_seconds_p95'], 3)} s p95"
            for row in ladder
        )
        + f", against {number(direct['move_seconds_median'], 4)} s for direct C1."
    )
    bullets.append(
        f"- The strongest rung on this pack was {biggest['preset_id']} at EWR "
        f"{number(biggest['ewr'])}; the selected operating point is {selected_id}, "
        f"{number(rule['operating_point']['ewr_behind_strongest'])} EWR behind it and "
        f"{number(selected['search_seconds_per_game'], 1)} s of search per game."
    )
    bullets.append(
        "- Stopping rule: "
        + (
            ", ".join(rule["conditions_fired"])
            if rule["conditions_fired"]
            else "no condition fired"
        )
        + "."
    )
    return {
        "headline": headline,
        "statement": statement,
        "bullets": bullets,
        "selected_preset_id": selected_id,
        "selected_arm_id": selected_arm,
        "any_step_meaningful": any_step_meaningful,
        "all_rungs_above_direct": all_above_direct,
        "ladder_ewr_spread": spread,
    }


def build_interpretation(summary: dict) -> "list[str]":
    """The prose reading, derived from the same numbers as the verdict."""
    ladder = summary["ladder"]
    arms = summary["arms"]
    rule = summary["stopping_rule"]
    verdict = summary["verdict"]
    margin = rule["thresholds"]["meaningful_ewr_gain"]
    games = summary["match_pack"]["boards"]
    direct = arms[mp.ARM_DIRECT.arm_id]
    selected_id = verdict["selected_preset_id"]
    selected = arms[verdict["selected_arm_id"]]
    steps = [row for row in ladder if row["delta_ewr_from_previous"] is not None]
    cheapest, dearest = ladder[0], ladder[-1]

    paragraphs: list = []
    paragraphs.append(
        f"The ladder cost what it was expected to cost. {cheapest['preset_id']} decides "
        f"in {number(arms[cheapest['arm_id']]['move_seconds_median'], 3)} s and "
        f"{dearest['preset_id']} in "
        f"{number(arms[dearest['arm_id']]['move_seconds_median'], 3)} s — a factor of "
        f"{number(arms[dearest['arm_id']]['move_seconds_median'] / arms[cheapest['arm_id']]['move_seconds_median'], 1)} "
        f"for a factor of {number(bd.relative_cost(bd.ladder_config(dearest['preset_id']), bd.ladder_config(cheapest['preset_id'])), 1)} "
        "in forward passes, which is the arithmetic of worlds times candidates times "
        "plies and not a surprise. Latency is the half of this agent's question that "
        "the match pack measures precisely."
    )
    if verdict["any_step_meaningful"]:
        gained = [row for row in steps if row["delta_ewr_from_previous"] >= margin]
        paragraphs.append(
            "Strength moved with budget: "
            + ", ".join(
                f"{row['previous_preset_id']} → {row['preset_id']} "
                f"{signed(row['delta_ewr_from_previous'])} EWR"
                for row in gained
            )
            + f", at or above the {number(margin, 2)} margin. That is the branch of the "
            "instruction where a larger rung is worth considering, and the stopping "
            "rule is what decides whether to take it."
        )
    else:
        paragraphs.append(
            f"Strength did not move with budget. The whole ladder spans "
            f"{number(verdict['ladder_ewr_spread'])} EWR — "
            f"{games_text(verdict['ladder_ewr_spread'], games)} — against an unpaired "
            f"standard error near "
            f"{number(summary['noise']['unpaired_standard_error'], 3)} and a "
            f"{number(margin, 2)} margin, so the ordering among rungs is a record of "
            "what happened and not a ranking. Buying more worlds and more depth did not "
            "buy games on this pack."
        )
    paragraphs.append(
        f"That makes the practical operating point {selected_id}: "
        f"{number(selected['search_seconds_per_game'], 1)} s of search per game, "
        f"{number(selected['move_seconds_median'], 3)} s per decision at the median and "
        f"{number(selected['move_seconds_p95'], 3)} s at p95, EWR {number(selected['ewr'])} "
        f"against direct C1's {number(direct['ewr'])}. It is chosen by the stated rule — "
        "the cheapest rung not meaningfully behind the strongest — and not by picking "
        "the largest number in the EWR column."
    )
    strongest_id = rule["operating_point"]["strongest_preset_id"]
    if strongest_id != selected_id:
        strongest = arms[bd.ladder_arm(strongest_id).arm_id]
        paragraphs.append(
            f"What that choice gives up should be stated rather than buried. The "
            f"strongest rung on this pack was {strongest_id} at EWR "
            f"{number(strongest['ewr'])}, "
            f"{number(rule['operating_point']['ewr_behind_strongest'])} EWR — "
            f"{games_text(rule['operating_point']['ewr_behind_strongest'], games)} — "
            f"ahead of {selected_id}, and it costs "
            f"{number(strongest['move_seconds_median'], 3)} s per decision against "
            f"{number(selected['move_seconds_median'], 3)} s. The rule prefers the "
            "cheaper rung because a lead smaller than the margin has not been shown to "
            "be a lead at all; a reader who believes the pack resolves finer than that "
            f"should read {strongest_id} as the operating point instead, and pay the "
            f"{number(strongest['search_seconds_per_game'] - selected['search_seconds_per_game'], 1)} "
            "extra search seconds per game for it."
        )

    paragraphs.append(
        f"The pack itself is the limiting instrument, and it is worth saying how. "
        f"{summary['match_pack']['uncontested_boards']} of {games} boards were decided "
        "before the player seat had a single decision, and "
        f"{summary['match_pack']['boards_decided_within_three_player_decisions']} within "
        "three of them: an opening scout reaching a flag placed on a front row. Those "
        "boards return the same result for every rung whatever it spends, so the "
        "effective sample separating two budgets is smaller than the headline game count."
    )
    return paragraphs


# ---------------------------------------------------------------------------
# Stage: assembly
# ---------------------------------------------------------------------------


def write_games_csv(rows, path: Path) -> dict:
    """The flat per-game table, without the per-move arrays."""
    ordered = sorted(rows, key=lambda row: (row["board_id"], row["arm_id"]))
    fields = [key for key in ordered[0] if not key.startswith("move_")] if ordered else []
    fields = [field for field in fields if field != "moves"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    return {"path": str(path.relative_to(REPOSITORY_ROOT)), "rows": len(ordered), "fields": fields}


def front_row_flag_boards(plans) -> int:
    """Boards where either side's flag sits on its own front row.

    Reported as the structural cause of the uncontested boards. The check is
    static — it does not ask whether the file happens to be open — so it is
    an upper bound on the exposure, not a prediction of instant losses.
    """
    import numpy as np
    from stratego.engine.constants import FLAG

    def exposed(setup, color) -> bool:
        rows = np.asarray(setup).reshape(4, 10)
        front = rows[3] if color == "red" else rows[0]
        return bool((front == FLAG).any())

    count = 0
    for plan in plans:
        player = plan.red_setup if plan.player_color == "red" else plan.blue_setup
        opponent = plan.red_setup if plan.opponent_color == "red" else plan.blue_setup
        if exposed(player, plan.player_color) or exposed(opponent, plan.opponent_color):
            count += 1
    return count


def build_ladder_configuration(plans, arms, configs, *, device: str, threads: int) -> dict:
    return {
        "artifact": "phase12_agent04_budget_ladder_v1",
        "budget_version": bd.BUDGET_VERSION,
        "search_version": SEARCH_VERSION,
        "score_definition": SCORE_DEFINITION,
        "belief_provider": bd.LADDER_PROVIDER,
        "rungs": "  ".join(
            f"{config.preset_id}(worlds {config.worlds}, depth {config.rollout_depth}, "
            f"candidates <= {config.max_root_candidates})"
            for config in configs
        ),
        "beta": configs[0].beta if configs else None,
        "reference_arm": mp.ARM_DIRECT.label,
        "opponents": ", ".join(GROUP_LABELS[s] for s in mp.MATCH_STRATA),
        "boards": len(plans),
        "games_played": len(plans) * len(arms),
        "setups": f"accepted library split '{mp.MATCH_LIBRARY_SPLIT}'",
        "rules": "EVALUATION_RULES (accepted)",
        "master seed": mp.MATCH_MASTER_SEED,
        "match_version": mp.MATCH_VERSION,
        "test bank": "False (never opened)",
        "device": f"{device}, {threads} torch threads",
    }


def build_summary(
    rows,
    plans,
    arms,
    configs,
    seats: dict,
    probes: dict,
    *,
    model_identity: dict,
    device: str,
    threads: int,
    seconds_total: float,
    stopped_early: bool,
    games_csv: dict,
    handoff: dict,
) -> dict:
    order = [arm.arm_id for arm in arms]
    pack = contested_board_set(rows, arms)
    contested = pack.pop("contested")
    pack["front_row_flag_boards"] = front_row_flag_boards(plans)
    direct_rows = arm_rows(rows, mp.ARM_DIRECT.arm_id)

    by_arm = {
        arm.arm_id: arm_summary(
            rows, arm, direct_rows=direct_rows, contested_boards=contested
        )
        for arm in arms
    }
    points = budget_points(by_arm, arms, probes)
    direct_block = by_arm[mp.ARM_DIRECT.arm_id]
    analysis = bd.ladder_analysis(
        points,
        reference_ewr=direct_block["ewr"],
        reference_seconds_per_game=direct_block["search_seconds_per_game"],
    )
    for row in analysis:
        row["arm_id"] = bd.ladder_arm(row["preset_id"]).arm_id
    rule = bd.stopping_rule(points)

    paired = []
    for arm in arms:
        if arm.kind != "search":
            continue
        block = dict(by_arm[arm.arm_id]["paired_vs_direct"])
        block["label"] = f"{arm.label} vs direct C1"
        paired.append(block)
    for previous, current in zip(arms[1:], arms[2:]):
        block = dict(
            paired_comparison(arm_rows(rows, current.arm_id), arm_rows(rows, previous.arm_id))
        )
        block["label"] = f"{current.label} vs {previous.label}"
        paired.append(block)
    if len(arms) > 3:
        block = dict(
            paired_comparison(arm_rows(rows, arms[-1].arm_id), arm_rows(rows, arms[1].arm_id))
        )
        block["label"] = f"{arms[-1].label} vs {arms[1].label}"
        paired.append(block)

    selected_id = rule["operating_point"]["selected_preset_id"]
    selected_config = bd.ladder_config(selected_id)
    selected_block = by_arm[bd.ladder_arm(selected_id).arm_id]

    summary = {
        "artifact": "phase12_agent04_budget_scaling_v1",
        "phase": "phase12",
        "agent": 4,
        "generated_utc": utc_now(),
        "search_version": SEARCH_VERSION,
        "budget_version": bd.BUDGET_VERSION,
        "score_definition": SCORE_DEFINITION,
        "device": device,
        "arm_order": order,
        "ladder_configuration": build_ladder_configuration(
            plans, arms, configs, device=device, threads=threads
        ),
        "move_model_identity": model_identity,
        "belief_model_identity": handoff["agent1c_checkpoint"],
        "seats": seats,
        "probes": probes,
        "match_pack": pack,
        "arms": by_arm,
        "ladder": analysis,
        "paired": paired,
        "stopping_rule": rule,
        "profile": load_profile(),
        "reproduction": reproduction_check(rows),
        "noise": {
            "unpaired_standard_error": mean_or_none(
                [by_arm[arm.arm_id]["outcome_standard_error"] for arm in arms]
            ),
            "paired_standard_error": mean_or_none(
                [entry.get("standard_error") for entry in paired]
            ),
            "note": (
                "descriptive resolution of the match pack; no significance claim is "
                "made anywhere in this agent"
            ),
        },
        "operating_point_record": {
            "worlds": selected_config.worlds,
            "root candidates": f"<= {selected_config.max_root_candidates}",
            "depth": selected_config.rollout_depth,
            "policy regularization": (
                f"S(a) = Q(a) + {selected_config.beta} * log(pi(a) + "
                f"{selected_config.epsilon})"
            ),
            "belief provider": bd.LADDER_PROVIDER,
            "expected move latency": (
                f"{number(selected_block['move_seconds_median'], 3)} s median, "
                f"{number(selected_block['move_seconds_p95'], 3)} s p95, "
                f"{number(selected_block['move_seconds_max'], 3)} s max"
            ),
            "quick strength result": (
                f"EWR {number(selected_block['ewr'])} "
                f"({record_text(selected_block)}) over {count_text(pack['boards'])} against "
                f"direct C1 {number(direct_block['ewr'])} "
                f"({record_text(direct_block)})"
            ),
            "search seconds per game": number(
                selected_block["search_seconds_per_game"], 1
            ),
            "games per hour": number(selected_block["games_per_hour"], 1),
        },
        "runtime": {
            "seconds_total": seconds_total,
            "stopped_early": stopped_early,
            "games": len(rows),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": threads,
            "games_csv": games_csv,
        },
        "deliverables": [
            "stratego/search/phase12/budget.py           (new; Agent 1 and 3 modules untouched)",
            "tests/search/test_phase12_budget.py",
            "reports/phase12/agent_04_ladder_config.json",
            "reports/phase12/agent_04_games.jsonl",
            "reports/phase12/agent_04_games.csv",
            "reports/phase12/agent_04_profile.json",
            "reports/phase12/agent_04_report.md",
            "reports/phase12/agent_04_summary.json",
        ],
        "status": {
            "phase11_final_classification": "FAIL",
            "phase11b_selection": "Agent1C",
            "scientific_validation_status": "not performed",
            "oracle_available_in_production": False,
            "phase11_test_bank_used": False,
            "search_core_modified": False,
            "belief_provider": bd.LADDER_PROVIDER,
            "presets_played": ", ".join(point.preset_id for point in points),
            "budget_above_medium_used": any(
                point.preset_id == bd.PRESET_LARGE.preset_id for point in points
            ),
            "selected_operating_point": selected_id,
            "production_integration_started": False,
            "agent_5_launched": False,
        },
    }
    summary["verdict"] = build_verdict(summary)
    summary["interpretation"] = build_interpretation(summary)
    summary["limitations"] = [
        f"{count_text(pack['boards'])} per rung is an engineering sample, not a powered "
        "experiment: an EWR difference below the stated noise scale is not evidence of "
        "an ordering between budgets.",
        "One provider (Agent 1C), one beta, one candidate rule, one search version. The "
        "ladder moves worlds and depth together, as the instruction's presets do, so it "
        "cannot say which of the two bought or failed to buy anything.",
        "The rungs share boards, opponent seeds and per-ply world seeds, which removes "
        "setup variance but leaves them correlated: paired numbers and unpaired numbers "
        "must not be mixed.",
        f"{pack['uncontested_boards']} boards were decided before the player's first "
        "decision and return the same result for every rung, so the pack's effective "
        "resolution is below its game count.",
        "Latency is single-process on cpu with the stated thread count. It is the "
        "latency of this machine and this device, not a portable number, and a parallel "
        "match harness would change it.",
        "The stopping-rule thresholds are engineering judgements stated in "
        "`stratego/search/phase12/budget.py`, not measurements. A reader who disagrees "
        "with the operating point should move a threshold and re-read the table.",
        f"Setups come from the accepted library's '{mp.MATCH_LIBRARY_SPLIT}' split, the "
        "same pool Phase 11B's dev split drew from, so a mild optimistic residual for "
        "agent1c is accepted for an engineering match test.",
    ]
    summary["stop_condition"] = (
        "Stop condition reached: a practical operating point has been identified. "
        + (
            "No preset above MEDIUM was run. "
            if not summary["status"]["budget_above_medium_used"]
            else ""
        )
        + "Production integration was not begun and Agent 5 is not launched."
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_arms(names: str):
    """The ladder in cheapest-first order, from a preset-name list."""
    presets = tuple(name.strip().upper() for name in names.split(",") if name.strip())
    return bd.ladder_arms(presets), [bd.ladder_config(name) for name in presets]


def rebuild_from_disk(arguments) -> int:
    """Re-run the analysis and the report over the rows already on disk."""
    arms, configs = resolve_arms(arguments.presets)
    rows = list(load_completed(GAMES_JSONL_PATH).values())
    if not rows:
        raise Phase12SearchError(f"no game rows on disk at {GAMES_JSONL_PATH}")
    arm_ids = {arm.arm_id for arm in arms}
    finished = {
        board
        for board in {row["board_id"] for row in rows}
        if {row["arm_id"] for row in rows if row["board_id"] == board} == arm_ids
    }
    rows = [row for row in rows if row["board_id"] in finished]
    sources = Phase11BSetupSources()
    plans = [
        plan
        for plan in mp.match_plans(
            sources, games_per_opponent=arguments.games_per_opponent
        )
        if plan.board_id in finished
    ]
    log(f"  {len(finished)} complete boards, {len(rows)} games")
    handoff = load_handoff()
    previous = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary = build_summary(
        rows,
        plans,
        arms,
        configs,
        previous.get("seats", {}),
        previous.get("probes", {}),
        model_identity=previous.get("move_model_identity", {}),
        device=arguments.device,
        threads=torch.get_num_threads(),
        seconds_total=float(previous.get("runtime", {}).get("seconds_total", 0.0)),
        stopped_early=bool(previous.get("runtime", {}).get("stopped_early", False)),
        games_csv=write_games_csv(rows, GAMES_CSV_PATH),
        handoff=handoff,
    )
    SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
    write_report(summary, REPORT_PATH)
    log(summary["verdict"]["headline"])
    log(summary["verdict"]["statement"])
    return 0


def profile_only(arguments) -> int:
    """Measure the per-rung cost split; play no games."""
    log("Phase 12 Agent 4 — cost profile")
    handoff = load_handoff()
    model, identity = load_move_model(handoff, arguments.device)
    provider = build_agent1c_provider(model, handoff, arguments.device)
    _, configs = resolve_arms(arguments.presets)
    profile = profile_stage(
        configs,
        model=model,
        identity=identity,
        provider=provider,
        device=arguments.device,
        positions=arguments.profile_positions,
    )
    PROFILE_PATH.write_text(json.dumps(sanitize(profile), indent=1) + "\n")
    log(f"  wrote {PROFILE_PATH.name}; rebuild the report with --analysis-only")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="0 leaves torch alone")
    parser.add_argument(
        "--presets",
        default=",".join(bd.LADDER_PRESET_NAMES),
        help="ladder rungs, cheapest first",
    )
    parser.add_argument(
        "--games-per-opponent",
        type=int,
        default=GAMES_PER_OPPONENT,
        help="must be a multiple of 4 so colours and setup sources stay balanced",
    )
    parser.add_argument("--probe-interval", type=int, default=24)
    parser.add_argument("--probe-budget", type=int, default=16)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop at a board boundary after this much wall clock (0 = no cap)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep game rows already on disk and play only what is missing",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="rebuild the analysis and report from the rows on disk",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="measure the per-rung cost split and write the profile, playing no games",
    )
    parser.add_argument("--profile-positions", type=int, default=12)
    arguments = parser.parse_args()

    if arguments.threads:
        torch.set_num_threads(int(arguments.threads))
    if arguments.analysis_only:
        return rebuild_from_disk(arguments)
    if arguments.profile_only:
        return profile_only(arguments)

    started = time.perf_counter()
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log("Phase 12 Agent 4 — search budget scaling with Agent 1C")
    log("stage: identities")
    handoff = load_handoff()
    model, model_identity = load_move_model(handoff, arguments.device)

    from stratego.evaluation.phase11_pipeline import build_owners

    owners, _ = build_owners(
        REPOSITORY_ROOT,
        CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt",
        device=arguments.device,
    )
    provider = build_agent1c_provider(model, handoff, arguments.device)
    log(
        f"  belief provider: {provider.provider_id}, checkpoint "
        f"{handoff['agent1c_checkpoint']['sha256'][:12]}..., one instance shared by "
        "every rung"
    )

    arms, configs = resolve_arms(arguments.presets)
    for config in configs:
        log(
            f"  rung: {config.preset_id} worlds {config.worlds} depth "
            f"{config.rollout_depth} candidates <= {config.max_root_candidates} "
            f"beta {config.beta}"
        )
    seats = build_ladder_seats(
        arms,
        model=model,
        identity=model_identity,
        provider=provider,
        owners=owners,
        device=arguments.device,
    )
    reference = probe_reference(owners)
    probes = {
        arm.arm_id: mp.SeatProbe(
            reference=reference if arm.kind == "search" else None,
            interval=arguments.probe_interval,
            budget=arguments.probe_budget,
        )
        for arm in arms
    }
    log(f"  seats ready: {', '.join(arm.arm_id for arm in arms)}")

    log("stage: match pack")
    sources = Phase11BSetupSources()
    plans = mp.match_plans(sources, games_per_opponent=arguments.games_per_opponent)
    log(
        f"  {len(plans)} boards: {len(mp.MATCH_STRATA)} opponents x "
        f"{arguments.games_per_opponent} games, balanced over "
        f"{len(mp.MATCH_SOURCES)} setup sources and {len(mp.MATCH_COLORS)} colours"
    )
    CONFIG_PATH.write_text(
        json.dumps(
            sanitize(
                build_ladder_configuration(
                    plans,
                    arms,
                    configs,
                    device=arguments.device,
                    threads=torch.get_num_threads(),
                )
            ),
            indent=1,
        )
        + "\n"
    )
    log(f"  ladder configuration -> {CONFIG_PATH.name}")

    log(f"stage: play ({len(plans) * len(arms)} games)")
    rows, stopped_early = play_stage(
        plans,
        arms,
        seats,
        owners,
        probes=probes,
        resume=arguments.resume,
        max_seconds=arguments.max_seconds,
    )

    log("stage: analysis")
    # Only boards where every rung finished. A board one rung played and
    # another did not would silently unbalance the ladder, which is the one
    # thing a paired comparison cannot afford.
    arm_ids = {arm.arm_id for arm in arms}
    finished = {
        board
        for board in {row["board_id"] for row in rows}
        if {row["arm_id"] for row in rows if row["board_id"] == board} == arm_ids
    }
    dropped = len({row["board_id"] for row in rows}) - len(finished)
    if dropped:
        log(f"  dropping {dropped} board(s) that no rung set completed")
    rows = [row for row in rows if row["board_id"] in finished]
    plans = [plan for plan in plans if plan.board_id in finished]
    log(f"  {len(finished)} complete boards, {len(rows)} games")

    summary = build_summary(
        rows,
        plans,
        arms,
        configs,
        {arm.arm_id: seats[arm.arm_id].describe() for arm in arms},
        {arm.arm_id: probes[arm.arm_id].summary() for arm in arms},
        model_identity=model_identity,
        device=arguments.device,
        threads=torch.get_num_threads(),
        seconds_total=time.perf_counter() - started,
        stopped_early=stopped_early,
        games_csv=write_games_csv(rows, GAMES_CSV_PATH),
        handoff=handoff,
    )
    SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
    write_report(summary, REPORT_PATH)

    for arm_id in summary["arm_order"]:
        block = summary["arms"][arm_id]
        log(
            f"  {block['label']:<30} {record_text(block):>12}  EWR "
            f"{number(block['ewr'])}  {number(block['move_seconds_median'], 3)} s/move "
            f"median"
        )
    failures = [
        arm_id for arm_id, block in summary["probes"].items() if block["failures"]
    ]
    if failures:
        log(f"  PROBE FAILURES: {failures}")
    if summary["reproduction"].get("available") and not summary["reproduction"]["reproduced"]:
        log("  REPRODUCTION MISMATCH against Agent 3")
    log(f"  {summary['verdict']['headline']}")
    log(f"  {summary['verdict']['statement']}")
    log(f"  wrote {REPORT_PATH.name} and {SUMMARY_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
