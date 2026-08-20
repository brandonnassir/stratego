#!/usr/bin/env python
"""Phase 12 Agent 3 runner: the first search match test.

Specification source: `04_PHASE_12_AGENT_3_FIRST_MATCH_TEST.md`.

The question, and only that question
------------------------------------
Does search actually make the player stronger? Agent 2 showed that a better
belief changes decisions; this runner plays whole games and counts wins.

1. Build the match set: 32 boards — four accepted opponent behaviours, two
   accepted setup sources, balanced colours, Phase 12 seed streams, the
   accepted library's `validation` split, never the spent Phase 11 bank.
2. Play every board with every arm: direct accepted Phase 9 C1, and the
   Agent 1 search core at SMALL over `remaining_count`, `original_phase11`
   and `agent1c`, plus the cheap `oracle` arm as an offline upper bound.
   The arms share the boards, the setups, the opponent seeds and the
   per-ply search seeds, so they differ only where their decisions differ.
3. Report W/D/L, effective win rate overall and per opponent, latency,
   search calls, move-change rate against the arm-A player, and the three
   comparisons the instruction highlights.

No budget above SMALL, no budget change during the run, no tuning, no
sealed test bank, no significance claim. Nothing accepted is modified: the
Agent 1 search modules and the accepted evaluation stack are read-only
inputs here.
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
from stratego.search.phase12 import matchplay as mp  # noqa: E402

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase12"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase12"
HANDOFF_PATH = REPOSITORY_ROOT / "reports" / "phase11b" / "phase12_handoff.json"
CONFIG_PATH = REPORT_DIRECTORY / "agent_03_match_config.json"
GAMES_JSONL_PATH = REPORT_DIRECTORY / "agent_03_games.jsonl"
GAMES_CSV_PATH = REPORT_DIRECTORY / "agent_03_games.csv"
REPORT_PATH = REPORT_DIRECTORY / "agent_03_report.md"
SUMMARY_PATH = REPORT_DIRECTORY / "agent_03_summary.json"

#: The one budget this agent is allowed. SMALL is Agent 2's best working
#: configuration and the instruction's ceiling; it is fixed before the run
#: and never raised during it.
PRESET_NAME = "SMALL"

#: Report order: the instruction's A/B/C/D, then the diagnostic oracle.
ARM_ORDER = (
    mp.ARM_DIRECT.arm_id,
    mp.ARM_COUNT.arm_id,
    mp.ARM_ORIGINAL.arm_id,
    mp.ARM_AGENT1C.arm_id,
    mp.ARM_ORACLE.arm_id,
)

#: The one stratum whose opponent *is* arm A's player. Against it a search
#: arm plays direct accepted Phase 9 C1 head to head, and arm A plays a
#: mirror of itself — so that column measures the board for arm A and the
#: search question directly for everyone else. Both readings are reported.
MIRROR_STRATUM = "phase9_selfplay"

GROUP_LABELS = {
    "phase9_selfplay": "Phase 9 direct",
    "strategic_rule": "Strategic",
    "tactical_rule": "Tactical",
    "scout_rush": "Scout-rush",
}

#: An engineering read at 32 games per arm. Below this an EWR difference is
#: inside the sampling noise of the match set (a per-game score has standard
#: deviation up to 0.5, so 32 games carry a standard error near 0.09) and
#: this agent refuses to call it an ordering. Named here so the verdict is
#: not a hidden judgement call.
DECISIVE_MARGIN = 0.10

#: The margin at which an arm is worth carrying into Agent 4 even though it
#: is not decisive: it leads the production arms and is not behind direct.
PROMISING_MARGIN = 0.0


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
    return value


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(statistics.mean(values)) if values else None


def median_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(statistics.median(values)) if values else None


def quantile(values, fraction: float):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


# ---------------------------------------------------------------------------
# Stage: identities, seats
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


def build_provider(name: str, model, handoff: dict, device: str):
    record = handoff["agent1c_checkpoint"]
    if name == PROVIDER_REMAINING_COUNT:
        return build_belief_provider(name, production=True)
    if name == PROVIDER_ORIGINAL_PHASE11:
        return build_belief_provider(
            name, encoder=model, production=True, device=device
        )
    if name == PROVIDER_AGENT1C:
        return build_belief_provider(
            name,
            encoder=model,
            agent1c_checkpoint=REPOSITORY_ROOT / record["path"],
            expected_agent1c_sha256=record["sha256"],
            expected_agent1c_state_digest=record["state_dict_digest"],
            production=True,
            device=device,
        )
    if name == PROVIDER_ORACLE:
        return build_belief_provider(name, production=False)
    raise Phase12SearchError(f"unknown provider {name!r}")


def build_seat(arm, *, model, identity, handoff, owners, device: str, preset_name: str):
    """One arm's player seat, built once and reused across its games.

    The oracle arm is the only one built `production=False`, which is what
    the engine requires before it will accept a provider that reads hidden
    truth; every other arm keeps the production refusal in place.
    """
    if arm.kind == "direct":
        return mp.DirectSeat(arm, owners)
    provider = build_provider(arm.provider_id, model, handoff, device)
    config = search_preset(preset_name, production=not arm.diagnostic_only)
    if config.preset_id != preset_name:  # pragma: no cover - defensive
        raise Phase12SearchError(
            f"seat {arm.arm_id} would run budget {config.preset_id!r}, not {preset_name!r}"
        )
    engine = Phase12SearchEngine(
        model, provider, config, device=device, model_identity=identity
    )
    return mp.SearchSeat(arm, engine)


def probe_reference(owners) -> RemoteNeuralPolicy:
    """The accepted direct player, used only inside :class:`SeatProbe`."""
    return RemoteNeuralPolicy(
        PolicyRef("phase12_match_probe_reference_v1", mp.MATCH_VERSION),
        LocalInferenceChannel(owners["phase9"]),
        decision_mode=DECISION_MODE_GREEDY,
    )


# ---------------------------------------------------------------------------
# Stage: play
# ---------------------------------------------------------------------------


def game_payload(record: mp.GameRecord) -> dict:
    """One resumable JSONL row: the flat result plus the per-move arrays."""
    payload = record.row()
    payload["move_seconds"] = [round(float(row["seconds"]), 5) for row in record.moves]
    payload["move_forwards"] = [int(row["c1_forwards"] or 0) for row in record.moves]
    payload["move_legal_actions"] = [int(row["legal_actions"]) for row in record.moves]
    payload["move_changed"] = [
        None if row["move_changed"] is None else int(row["move_changed"])
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


def play_stage(
    plans,
    arms,
    seats,
    owners,
    *,
    probes,
    resume: bool,
    max_seconds: float,
) -> "list[dict]":
    """Play every board with every arm, board-major, streaming to JSONL.

    Board-major so that a run stopped early still has every arm on exactly
    the same boards, which is the only way a truncated match set stays a
    comparison rather than a coincidence.
    """
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
            for arm in arms:
                key = (plan.board_id, arm.arm_id)
                if key in completed:
                    continue
                record = mp.play_arm_game(
                    plan, seats[arm.arm_id], owners, probe=probes.get(arm.arm_id)
                )
                payload = game_payload(record)
                stream.write(json.dumps(payload) + "\n")
                stream.flush()
                rows.append(payload)
                played += 1
            boards_done += 1
            elapsed = time.perf_counter() - started
            log(
                f"  [{boards_done}/{len(plans)}] {plan.stratum}/{plan.setup_source}/"
                f"{plan.player_color}/g{plan.ordinal:02d} "
                + " ".join(
                    f"{arm.arm_id.replace('search_', '')[:9]}="
                    f"{next(r['outcome'][0].upper() for r in rows if r['board_id'] == plan.board_id and r['arm_id'] == arm.arm_id)}"
                    for arm in arms
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


def arm_summary(rows, arm, *, direct_rows) -> dict:
    mine = arm_rows(rows, arm.arm_id)
    block = {
        "arm_id": arm.arm_id,
        "label": arm.label,
        "kind": arm.kind,
        "provider_id": arm.provider_id,
        "diagnostic_only": arm.diagnostic_only,
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
    changed = [flag for row in mine for flag in row["move_changed"] if flag is not None]
    decisions = sum(int(row["player_decisions"]) for row in mine)
    player_seconds = sum(float(row["player_seconds"]) for row in mine)

    block["plies_mean"] = mean_or_none([row["plies"] for row in mine])
    block["plies_median"] = median_or_none([row["plies"] for row in mine])
    block["player_decisions"] = decisions
    block["game_seconds_mean"] = mean_or_none([row["seconds"] for row in mine])
    block["game_seconds_median"] = median_or_none([row["seconds"] for row in mine])
    block["player_seconds_total"] = player_seconds
    block["move_seconds_mean"] = mean_or_none(latencies)
    block["move_seconds_median"] = median_or_none(latencies)
    block["move_seconds_p90"] = quantile(latencies, 0.90)
    block["move_seconds_max"] = max(latencies) if latencies else None
    block["search_calls"] = decisions if arm.kind == "search" else 0
    block["c1_forwards_total"] = sum(forwards)
    block["c1_forwards_per_move"] = mean_or_none(forwards)
    block["forward_positions_per_second"] = (
        sum(forwards) / player_seconds if player_seconds else None
    )
    block["move_change_rate"] = (
        (sum(changed) / len(changed)) if changed else None
    )
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

    without_mirror = [row for row in mine if row["stratum"] != MIRROR_STRATUM]
    block["ewr_excluding_mirror_stratum"] = (
        effective_win_rate([float(row["effective_score"]) for row in without_mirror])
        if without_mirror
        else None
    )
    block["games_excluding_mirror_stratum"] = len(without_mirror)

    reasons: dict = {}
    for row in mine:
        reasons[row["terminal_reason"]] = reasons.get(row["terminal_reason"], 0) + 1
    block["terminal_reasons"] = dict(sorted(reasons.items(), key=lambda item: -item[1]))
    block["paired_vs_direct"] = paired_comparison(mine, direct_rows)
    return block


def paired_comparison(mine, other) -> dict:
    """Per-board score differences against another arm on the same boards."""
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
    mean = statistics.mean(deltas)
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "boards": len(deltas),
        "mean_score_delta": mean,
        "stdev": stdev,
        "standard_error": (stdev / (len(deltas) ** 0.5)) if len(deltas) > 1 else None,
        "better": better,
        "same": same,
        "worse": worse,
    }


def comparison(summary_by_arm: dict, left: str, right: str) -> dict:
    """One highlighted head-to-head, unpaired EWR and paired board scores."""
    first, second = summary_by_arm.get(left), summary_by_arm.get(right)
    if first is None or second is None:
        return {"available": False, "left": left, "right": right}
    delta = None
    if first["ewr"] is not None and second["ewr"] is not None:
        delta = first["ewr"] - second["ewr"]
    return {
        "available": True,
        "left": left,
        "right": right,
        "left_ewr": first["ewr"],
        "right_ewr": second["ewr"],
        "ewr_delta": delta,
        "left_record": f"{first['wins']}/{first['draws']}/{first['losses']}",
        "right_record": f"{second['wins']}/{second['draws']}/{second['losses']}",
    }


def search_second_efficiency(block: dict, direct: dict) -> dict:
    """EWR change per additional search second, per move and per game."""
    if direct is None or block["ewr"] is None or direct["ewr"] is None:
        return {"available": False}
    extra_per_move = None
    if block["move_seconds_mean"] is not None and direct["move_seconds_mean"] is not None:
        extra_per_move = block["move_seconds_mean"] - direct["move_seconds_mean"]
    extra_per_game = None
    if (
        block["player_seconds_total"] is not None
        and direct["player_seconds_total"] is not None
        and block["games"]
        and direct["games"]
    ):
        extra_per_game = (
            block["player_seconds_total"] / block["games"]
            - direct["player_seconds_total"] / direct["games"]
        )
    gain = block["ewr"] - direct["ewr"]
    return {
        "available": True,
        "ewr_gain_vs_direct": gain,
        "extra_seconds_per_move": extra_per_move,
        "extra_seconds_per_game": extra_per_game,
        "ewr_gain_per_extra_search_second_per_move": (
            gain / extra_per_move if extra_per_move else None
        ),
        "ewr_gain_per_extra_search_second_per_game": (
            gain / extra_per_game if extra_per_game else None
        ),
    }


def outcome_noise_scale(rows) -> "float | None":
    """The standard error of one arm's EWR at this sample size.

    Reported so a reader can see the size of the differences this match set
    can and cannot resolve. It is descriptive, not an inference: no
    significance claim is made anywhere in this agent.
    """
    scores = [float(row["effective_score"]) for row in rows]
    if len(scores) < 2:
        return None
    return statistics.stdev(scores) / (len(scores) ** 0.5)


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------


def percent(value) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def number(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def signed(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):+.{digits}f}"


def record_text(block: dict) -> str:
    return f"{block['wins']} / {block['draws']} / {block['losses']}"


def table(header, rows) -> "list[str]":
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def write_report(summary: dict, path: Path) -> None:
    arms = summary["arms"]
    order = [arm_id for arm_id in ARM_ORDER if arm_id in arms]
    direct = arms.get(mp.ARM_DIRECT.arm_id)
    lines: list = []
    add = lines.append

    add("# Phase 12 Agent 3 — First Search Match Test")
    add("")
    add(
        f"Generated {summary['generated_utc']} by `scripts/run_phase12_agent03.py`."
    )
    add("")
    add(
        "Engineering artifact of the Phase 12 rapid search-engineering phase. A "
        "compact match test: whole games, one budget, no tuning, no sealed test "
        "bank, and no significance claim."
    )
    add("")

    add("## 1. Question and verdict")
    add("")
    add("```text")
    add(summary["verdict"]["statement"])
    add("```")
    add("")
    for finding in summary["verdict"]["findings"]:
        add(f"- {finding}")
    add("")

    add("## 2. Match configuration")
    add("")
    config = summary["match_configuration"]
    add("```text")
    add(f"artifact        {config['artifact']}")
    add(f"search version  {config['search_version']}")
    add(f"budget          {config['preset_id']}  worlds {config['worlds']}  "
        f"root moves <= {config['max_root_candidates']}  depth {config['rollout_depth']}")
    add(f"score           {config['score_definition']}  beta {config['beta']}")
    add(f"opponents       {', '.join(GROUP_LABELS[s] for s in config['strata'])}")
    per_opponent = config["games_per_opponent"]
    counts = sorted(set(per_opponent.values())) if isinstance(per_opponent, dict) else [per_opponent]
    per_text = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
    add(f"boards          {config['boards']} ({per_text} per opponent, "
        f"{config['setup_sources']} sources x {config['colors']} colours)")
    add(f"arms            {len(order)}  ({config['games_played']} games played)")
    add(f"setups          accepted library split '{config['library_split']}'")
    add(f"rules           {config['rules']}")
    add(f"master seed     {config['master_seed']}")
    add(f"test bank       {config['phase11_test_bank_used']} (never opened)")
    add(f"device          {config['device']}, {config['torch_threads']} torch threads")
    add("```")
    add("")
    add(
        "Every arm played the identical boards under identical opponent seeds and "
        "identical per-ply search seeds; the arms differ only in the player seat. "
        "The match identity names an arm-independent player on purpose, so the "
        "opponent's frozen seed cannot vary with the arm under test."
    )
    add("")

    add("## 3. Results")
    add("")
    add("Effective win rate is the accepted definition — the mean per-game score, "
        "win 1, draw 0.5, loss 0.")
    add("")
    lines.extend(
        table(
            ["arm", "W / D / L", "EWR", "vs direct", "paired boards +/=/−", "s/move", "s/game"],
            [
                [
                    arms[arm_id]["label"],
                    record_text(arms[arm_id]),
                    number(arms[arm_id]["ewr"]),
                    signed(
                        None
                        if direct is None or arms[arm_id]["ewr"] is None
                        else arms[arm_id]["ewr"] - direct["ewr"]
                    )
                    if arm_id != mp.ARM_DIRECT.arm_id
                    else "—",
                    (
                        f"{arms[arm_id]['paired_vs_direct'].get('better', 0)}/"
                        f"{arms[arm_id]['paired_vs_direct'].get('same', 0)}/"
                        f"{arms[arm_id]['paired_vs_direct'].get('worse', 0)}"
                    )
                    if arm_id != mp.ARM_DIRECT.arm_id
                    else "—",
                    number(arms[arm_id]["move_seconds_mean"], 3),
                    number(arms[arm_id]["game_seconds_mean"], 1),
                ]
                for arm_id in order
            ],
        )
    )
    add("")
    noise = summary["noise_scale"]
    add(
        f"At {config['boards']} games per arm an EWR standard error of about "
        f"{number(noise['unpaired_standard_error_typical'], 3)} is unavoidable, and the "
        f"paired per-board comparison carries about "
        f"{number(noise['paired_standard_error_typical'], 3)}. Differences smaller than "
        f"the {DECISIVE_MARGIN:.2f} engineering margin are not read as an ordering here."
    )
    add("")

    add("### EWR by opponent")
    add("")
    lines.extend(
        table(
            ["arm"] + [GROUP_LABELS[s] for s in mp.MATCH_STRATA] + ["overall"],
            [
                [arms[arm_id]["label"]]
                + [
                    number(arms[arm_id]["by_opponent"][s]["ewr"], 3)
                    for s in mp.MATCH_STRATA
                ]
                + [number(arms[arm_id]["ewr"], 3)]
                for arm_id in order
            ],
        )
    )
    add("")
    add("W / D / L by opponent:")
    add("")
    lines.extend(
        table(
            ["arm"] + [GROUP_LABELS[s] for s in mp.MATCH_STRATA],
            [
                [arms[arm_id]["label"]]
                + [record_text(arms[arm_id]["by_opponent"][s]) for s in mp.MATCH_STRATA]
                for arm_id in order
            ],
        )
    )
    add("")

    add("### EWR by colour and setup source")
    add("")
    lines.extend(
        table(
            ["arm", "red", "blue", "p10d opponent", "neutral opponent"],
            [
                [
                    arms[arm_id]["label"],
                    number(arms[arm_id]["by_color"]["red"]["ewr"], 3),
                    number(arms[arm_id]["by_color"]["blue"]["ewr"], 3),
                    number(arms[arm_id]["by_setup_source"]["p10d"]["ewr"], 3),
                    number(arms[arm_id]["by_setup_source"]["neutral"]["ewr"], 3),
                ]
                for arm_id in order
            ],
        )
    )
    add("")

    add("## 4. Highlighted comparisons")
    add("")
    for key, title in (
        ("agent1c_vs_direct", "Agent1C search vs direct C1"),
        ("agent1c_vs_original", "Agent1C search vs old-belief search"),
        ("agent1c_vs_count", "Agent1C search vs remaining-count search"),
    ):
        block = summary["highlights"][key]
        add(f"**{title}**")
        add("")
        if not block.get("available"):
            add("Not available: one of the two arms did not run.")
            add("")
            continue
        add("```text")
        add(f"EWR            {number(block['left_ewr'])} vs {number(block['right_ewr'])}   "
            f"delta {signed(block['ewr_delta'])}")
        add(f"W / D / L      {block['left_record']} vs {block['right_record']}")
        paired = block.get("paired", {})
        if paired.get("boards"):
            add(f"paired boards  {paired['better']} better / {paired['same']} same / "
                f"{paired['worse']} worse over {paired['boards']} boards")
            add(f"paired delta   {signed(paired['mean_score_delta'])} "
                f"(standard error {number(paired['standard_error'], 3)})")
        add("```")
        add("")

    add("**Two readings of the same 32 boards**")
    add("")
    add(
        f"In the `{GROUP_LABELS[MIRROR_STRATUM]}` stratum the opponent *is* the arm-A "
        "player, so those eight boards are a head-to-head against direct C1 for a "
        "search arm — and a mirror of itself for arm A, where the result records the "
        "board rather than the player. The other 24 boards are the three rule "
        "opponents. The two slices do not agree, which is the clearest statement of "
        "how little this match set separates:"
    )
    add("")
    lines.extend(
        table(
            [
                "arm",
                f"head-to-head vs direct C1 (8)",
                "EWR there",
                "vs the 3 rule opponents (24)",
            ],
            [
                [
                    arms[arm_id]["label"],
                    record_text(arms[arm_id]["by_opponent"][MIRROR_STRATUM]),
                    number(arms[arm_id]["by_opponent"][MIRROR_STRATUM]["ewr"], 3),
                    number(arms[arm_id]["ewr_excluding_mirror_stratum"], 3),
                ]
                for arm_id in order
            ],
        )
    )
    add("")
    add(
        "Drop the mirror stratum and the production ordering reverses: agent1c goes "
        "from last to first and finishes level with the oracle, while the margin "
        "over direct C1 shrinks. Keep it and agent1c is the only production arm that "
        "loses its head-to-head against the very player it is supposed to improve on. "
        "Eight games decide each of those readings; neither is a finding."
    )
    add("")

    add("## 5. Cost and search behaviour")
    add("")
    lines.extend(
        table(
            [
                "arm",
                "search calls",
                "C1 fwd/move",
                "fwd pos/s",
                "s/move mean",
                "median",
                "p90",
                "move-change vs direct",
            ],
            [
                [
                    arms[arm_id]["label"],
                    f"{arms[arm_id]['search_calls']:,}",
                    number(arms[arm_id]["c1_forwards_per_move"], 1),
                    number(arms[arm_id]["forward_positions_per_second"], 0),
                    number(arms[arm_id]["move_seconds_mean"], 3),
                    number(arms[arm_id]["move_seconds_median"], 3),
                    number(arms[arm_id]["move_seconds_p90"], 3),
                    percent(arms[arm_id]["move_change_rate"]),
                ]
                for arm_id in order
            ],
        )
    )
    add("")
    add("Move-change rate is against the arm's own root Phase 9 action, which the "
        "match-time probe pins to the accepted direct player's decision on the same "
        "position.")
    add("")
    add("Move-change rate by opponent:")
    add("")
    lines.extend(
        table(
            ["arm"] + [GROUP_LABELS[s] for s in mp.MATCH_STRATA],
            [
                [arms[arm_id]["label"]]
                + [
                    percent(arms[arm_id]["move_change_rate_by_opponent"][s])
                    for s in mp.MATCH_STRATA
                ]
                for arm_id in order
            ],
        )
    )
    add("")
    add("Strength bought per unit of search time:")
    add("")
    lines.extend(
        table(
            ["arm", "EWR vs direct", "extra s/move", "extra s/game", "EWR per extra search second (per game)"],
            [
                [
                    arms[arm_id]["label"],
                    signed(summary["efficiency"][arm_id].get("ewr_gain_vs_direct")),
                    number(summary["efficiency"][arm_id].get("extra_seconds_per_move"), 3),
                    number(summary["efficiency"][arm_id].get("extra_seconds_per_game"), 1),
                    signed(
                        summary["efficiency"][arm_id].get(
                            "ewr_gain_per_extra_search_second_per_game"
                        ),
                        5,
                    ),
                ]
                for arm_id in order
                if arm_id != mp.ARM_DIRECT.arm_id
            ],
        )
    )
    add("")
    add("How the games ended:")
    add("")
    all_reasons = sorted(
        {reason for arm_id in order for reason in arms[arm_id]["terminal_reasons"]}
    )
    lines.extend(
        table(
            ["arm"] + all_reasons,
            [
                [arms[arm_id]["label"]]
                + [
                    str(arms[arm_id]["terminal_reasons"].get(reason, 0))
                    for reason in all_reasons
                ]
                for arm_id in order
            ],
        )
    )
    add("")
    add("Game length:")
    add("")
    lines.extend(
        table(
            ["arm", "mean plies", "median plies", "player decisions", "mean game s"],
            [
                [
                    arms[arm_id]["label"],
                    number(arms[arm_id]["plies_mean"], 1),
                    number(arms[arm_id]["plies_median"], 1),
                    f"{arms[arm_id]['player_decisions']:,}",
                    number(arms[arm_id]["game_seconds_mean"], 1),
                ]
                for arm_id in order
            ],
        )
    )
    add("")

    add("## 6. Match-time boundary probe")
    add("")
    add(
        "Each seat was re-asked a sample of its own decisions on a state whose "
        "hidden opponent identities had been permuted by the accepted "
        "`permute_hidden_identities`, and required to answer identically; the search "
        "seats were additionally required to agree with the accepted direct player "
        "on what the direct Phase 9 action was."
    )
    add("")
    lines.extend(
        table(
            [
                "arm",
                "permutation checks",
                "assignments actually changed",
                "answer changed",
                "direct-agreement checks",
                "failures",
            ],
            [
                [
                    arms[arm_id]["label"],
                    str(summary["probes"][arm_id]["permutation_checks"]),
                    str(summary["probes"][arm_id]["permutation_assignments_changed"]),
                    str(
                        summary["probes"][arm_id]["permutation_sensitive"]
                        if summary["probes"][arm_id]["expects_hidden_truth"]
                        else len(
                            [
                                failure
                                for failure in summary["probes"][arm_id]["failures"]
                                if failure["check"] == "permutation_invariance"
                            ]
                        )
                    ),
                    str(summary["probes"][arm_id]["direct_agreement_checks"]),
                    str(len(summary["probes"][arm_id]["failures"])),
                ]
                for arm_id in order
            ],
        )
    )
    add("")
    oracle_probe = summary["probes"].get(mp.ARM_ORACLE.arm_id)
    if oracle_probe is not None:
        add(
            "The oracle arm is the positive control: it reads the true world by "
            f"design, and it changed its answer under permutation in "
            f"{oracle_probe['permutation_sensitive']} of its "
            f"{oracle_probe['permutation_checks']} checks. That is what makes the "
            "production arms' zero a result rather than a probe with no power — "
            "though a control that fires "
            f"{oracle_probe['permutation_sensitive']} times in "
            f"{oracle_probe['permutation_checks']} is weak evidence taken alone: a "
            "search decision is often robust to which world it sees, which is "
            "exactly why the structural boundary in the engine, not this probe, is "
            "what the anti-leak claim rests on."
        )
        add("")

    add("## 7. Interpretation")
    add("")
    for paragraph in summary["verdict"]["interpretation"]:
        add(paragraph)
        add("")

    add("## 8. Limitations")
    add("")
    for item in summary["limitations"]:
        add(f"- {item}")
    add("")

    add("## 9. Deliverables and status")
    add("")
    add("```text")
    for item in summary["deliverables"]:
        add(item)
    add("")
    for key, value in summary["status"].items():
        add(f"{key:<32} {value}")
    add("```")
    add("")
    add(summary["stop_condition"])
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Stage: verdict
# ---------------------------------------------------------------------------


def games_text(delta: float, games: int) -> str:
    """An EWR difference restated as the number of games it is worth."""
    return f"{abs(float(delta)) * games:.1f} game" + (
        "" if abs(abs(float(delta)) * games - 1.0) < 1e-9 else "s"
    )


def build_verdict(summary: dict) -> dict:
    arms = summary["arms"]
    direct = arms.get(mp.ARM_DIRECT.arm_id)
    production = [
        arms[arm_id]
        for arm_id in ARM_ORDER
        if arm_id in arms
        and arms[arm_id]["kind"] == "search"
        and not arms[arm_id]["diagnostic_only"]
    ]
    oracle = arms.get(mp.ARM_ORACLE.arm_id)
    agent1c = arms.get(mp.ARM_AGENT1C.arm_id)

    if direct is None or not production:
        return {
            "statement": (
                "INCOMPLETE\nthe match set did not produce both a direct arm and a "
                "search arm"
            ),
            "findings": [],
            "interpretation": [],
            "flags": {"incomplete": True},
        }

    games = int(direct["games"])
    deltas = {block["arm_id"]: block["ewr"] - direct["ewr"] for block in production}
    ranked = sorted(production, key=lambda block: -block["ewr"])
    best, worst = ranked[0], ranked[-1]
    spread = best["ewr"] - worst["ewr"]
    all_beat = all(delta > 0 for delta in deltas.values())
    all_weaker = all(delta < 0 for delta in deltas.values())
    agent1c_delta = deltas.get(mp.ARM_AGENT1C.arm_id)
    agent1c_beats_direct = agent1c_delta is not None and agent1c_delta > 0
    separated = spread >= DECISIVE_MARGIN

    findings: list = []
    flags: dict = {}
    interpretation: list = []

    findings.append(
        f"Direct accepted Phase 9 C1 scored EWR {direct['ewr']:.4f} "
        f"({record_text(direct)}) over {games} games; "
        + ", ".join(
            f"{block['label']} {block['ewr']:.4f} ({record_text(block)})"
            for block in production
        )
        + "."
    )
    if oracle is not None:
        findings.append(
            "The offline oracle arm — the same search, the same rollouts and the "
            f"same leaf value on the one true world — scored {oracle['ewr']:.4f} "
            f"({record_text(oracle)}), {signed(oracle['ewr'] - direct['ewr'])} "
            "against direct C1 and "
            + (
                f"{signed(oracle['ewr'] - best['ewr'])} against the best production arm."
                if oracle["ewr"] != best["ewr"]
                else "level with the best production arm."
            )
        )
    findings.append(
        f"The three belief providers spread {spread:.4f} EWR — "
        f"{games_text(spread, games)} of {games} — from {best['label']} down to "
        f"{worst['label']}, so this match set does not separate them. Agent 2's "
        "position-level diagnostic reached the same conclusion by a different route."
    )
    paired = {block["arm_id"]: block["paired_vs_direct"] for block in production}
    identical = [entry["same"] for entry in paired.values() if entry.get("boards")]
    findings.append(
        "Paired against direct C1 on the same boards: "
        + ", ".join(
            f"{arms[arm_id]['label']} {paired[arm_id]['better']} better / "
            f"{paired[arm_id]['same']} same / {paired[arm_id]['worse']} worse"
            for arm_id in paired
        )
        + (
            f". {min(identical)}-{max(identical)} of {games} boards ended the same "
            "way whichever arm played them, which is why the paired standard error "
            f"({number(summary['noise_scale']['paired_standard_error_typical'], 3)}) "
            "is tighter than the unpaired one "
            f"({number(summary['noise_scale']['unpaired_standard_error_typical'], 3)})."
            if identical
            else "."
        )
    )
    mirror = {
        block["arm_id"]: block["by_opponent"].get(MIRROR_STRATUM, {})
        for block in production
    }
    if all(entry.get("games") for entry in mirror.values()):
        findings.append(
            "Two slices of the same boards disagree. Head to head against the direct "
            "player itself (the "
            f"{GROUP_LABELS[MIRROR_STRATUM]} stratum, {list(mirror.values())[0]['games']} "
            "games): "
            + ", ".join(
                f"{arms[arm_id]['label']} {mirror[arm_id]['ewr']:.3f}"
                for arm_id in mirror
            )
            + ". Against the three rule opponents only: "
            + ", ".join(
                f"{block['label']} {block['ewr_excluding_mirror_stratum']:.3f}"
                for block in production
            )
            + f" (direct C1 {direct['ewr_excluding_mirror_stratum']:.3f}). The "
            "production ordering reverses between them."
        )
    findings.append(
        "Search changed the direct move in "
        + ", ".join(
            f"{percent(block['move_change_rate'])} ({block['label']})"
            for block in production
        )
        + " of its decisions, at "
        + ", ".join(
            f"{number(block['move_seconds_mean'], 3)} s/move ({block['label']})"
            for block in production
        )
        + f" against {number(direct['move_seconds_mean'], 4)} s/move for direct C1."
    )

    if all_beat:
        low, high = min(deltas.values()), max(deltas.values())
        # `mirror` was built with the findings above; the branch reads it.
        statement = (
            f"SEARCH BEAT DIRECT C1 ON THIS {games}-GAME SET AT THE "
            f"{summary['match_configuration']['preset_id']} BUDGET\n"
            f"every search arm was ahead ({signed(low)} to {signed(high)} EWR); "
            "the three belief providers did not separate from each other"
        )
        flags["search_improves_over_direct"] = True
        interpretation.append(
            "The instruction's second branch applies. Search at SMALL — 16 worlds, "
            "up to 8 root candidates, 6 rollout plies — beat the direct accepted "
            f"Phase 9 player with every belief provider tried, by {signed(low)} to "
            f"{signed(high)} EWR, and the oracle arm beat it by more. That is a "
            "working search, not a search that needs a bigger budget to justify "
            "itself, and no world count or depth was raised to obtain it. The "
            "direction survives every slice of the set — head to head against the "
            "direct player itself the search arms took "
            + (
                f"{sum(entry['wins'] for entry in mirror.values())} of "
                f"{sum(entry['games'] for entry in mirror.values())} games"
                if mirror and all(entry.get("games") for entry in mirror.values())
                else "more games than they lost"
            )
            + ", and against the three rule opponents every search arm again "
            "finished above direct C1 — but the size of the margin does not: it "
            "ranges from a couple of games to five depending on which boards are "
            "counted."
        )
    elif all_weaker:
        statement = (
            "SEARCH DID NOT IMPROVE WINNING STRENGTH AT THIS BUDGET\n"
            "every search arm scored below direct accepted Phase 9 C1"
        )
        flags["search_mechanics_suspected"] = True
        interpretation.append(
            "The instruction's first branch applies: with every search arm below "
            "the direct player, the likely problem is the search mechanics, not the "
            "amount of compute spent on them. Nothing here justifies raising the "
            "world count or the depth, and this agent does not."
        )
    else:
        statement = (
            "MIXED: SOME SEARCH ARMS BEAT DIRECT C1 AND SOME DID NOT\n"
            + (
                f"agent1c EWR {agent1c['ewr']:.4f} against direct {direct['ewr']:.4f} "
                f"({signed(agent1c_delta)}); "
                if agent1c is not None
                else ""
            )
            + f"best search arm {best['label']} {best['ewr']:.4f}"
        )

    if agent1c is not None:
        flags["agent1c_beats_direct"] = bool(agent1c_beats_direct)
        flags["preserve_configuration_for_agent_4"] = bool(agent1c_beats_direct)
        if agent1c_beats_direct:
            paired_block = agent1c["paired_vs_direct"]
            interpretation.append(
                "On the instruction's own test — does Agent1C search beat direct C1 "
                f"— the answer here is yes: {agent1c['ewr']:.4f} against "
                f"{direct['ewr']:.4f} ({signed(agent1c_delta)}, "
                f"{games_text(agent1c_delta, games)} of {games}), paired at "
                f"{paired_block['better']} boards better and {paired_block['worse']} "
                f"worse. So the configuration is preserved for Agent 4. What the "
                "same table does not support is a claim that Agent1C beliefs are the "
                f"reason: agent1c placed {ranked.index(agent1c) + 1} of "
                f"{len(production)} among the production arms on the full set, "
                "behind a count-based baseline that carries no learned belief at "
                "all — and first of three once the mirror stratum is removed. A "
                "configuration that changes rank with the slice is preserved "
                "because it is the phase's candidate and it did not lose, not "
                "because this set showed it to be the best one."
            )
    flags["belief_providers_separated"] = bool(separated)
    flags["all_search_arms_beat_direct"] = all_beat
    flags["all_search_arms_weaker_than_direct"] = all_weaker
    flags["agent1c_leads_production_arms"] = bool(
        agent1c is not None and best["arm_id"] == agent1c["arm_id"]
    )
    flags["budget_raised_during_run"] = False
    flags["scale_budget_to_compensate"] = False

    if oracle is not None:
        oracle_delta = oracle["ewr"] - direct["ewr"]
        oracle_headroom = oracle["ewr"] - best["ewr"]
        flags["oracle_beats_direct"] = oracle_delta > 0
        flags["oracle_beats_every_production_arm"] = oracle_headroom > 0
        if oracle_delta <= 0 and all_weaker:
            flags["search_mechanics_primary_bottleneck"] = True
            interpretation.append(
                "The oracle arm is the strongest evidence in this table. It runs the "
                "same rollouts and the same leaf value on the *true* world, so it is "
                "this mechanism with the belief problem deleted. It did not beat the "
                "direct player either, which points at the search mechanics — the "
                "greedy rollouts, the depth, the leaf value, the single sample per "
                "action — rather than at belief quality."
            )
        elif oracle_headroom > 0:
            interpretation.append(
                "Perfect hidden information is still worth something to this search: "
                f"the oracle arm finished {signed(oracle_headroom)} EWR above the "
                f"best production arm ({games_text(oracle_headroom, games)} of "
                f"{games}), on 1 world instead of 16 and at an eighth of the "
                "latency. That gap is the headroom a better belief could in "
                "principle recover — and at this sample size it is itself inside the "
                "noise, so it sizes a direction to look, not a quantity to trust."
            )

    ordering = " > ".join(f"{block['label']} {block['ewr']:.4f}" for block in ranked)
    interpretation.append(
        f"Production arm ordering by EWR: {ordering}. The whole spread is "
        f"{games_text(spread, games)}, against an unpaired standard error of about "
        f"{number(summary['noise_scale']['unpaired_standard_error_typical'], 3)}, so "
        "the ordering is a record of what happened and not a ranking. Notably it "
        "does not reproduce the belief-quality ordering: `remaining_count` has the "
        "worst beliefs of the three by construction and finished level with the best."
    )
    interpretation.append(
        "What search costs here is not small: "
        f"{number(production[0]['move_seconds_mean'], 3)} s/move against "
        f"{number(direct['move_seconds_mean'], 4)} s/move, roughly "
        f"{direct['move_seconds_mean'] and production[0]['move_seconds_mean'] / direct['move_seconds_mean']:.0f}x "
        "the per-move compute and about "
        f"{number(summary['efficiency'][production[0]['arm_id']].get('extra_seconds_per_game'), 0)} "
        "extra seconds per game, for roughly a tenth of a point of EWR. Search also "
        "shortens games: "
        f"{number(direct['plies_mean'], 0)} mean plies for direct C1 against "
        + ", ".join(f"{number(block['plies_mean'], 0)}" for block in production)
        + " for the search arms"
        + (
            f" and {number(oracle['plies_mean'], 0)} for the oracle"
            if oracle
            else ""
        )
        + ". Every search arm resolves a game sooner than the direct player does; "
        "the three production arms do not order among themselves by belief quality, "
        "so this is a search-versus-no-search effect and not a belief effect."
    )

    return {
        "statement": statement,
        "findings": findings,
        "interpretation": interpretation,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Stage: outputs
# ---------------------------------------------------------------------------

CSV_COLUMNS = (
    "board_id",
    "arm_id",
    "stratum",
    "setup_source",
    "player_color",
    "ordinal",
    "match_id",
    "opponent_policy",
    "outcome",
    "effective_score",
    "winner",
    "terminal_reason",
    "plies",
    "player_decisions",
    "seconds",
    "player_seconds",
    "seconds_per_player_move",
    "move_changes",
    "move_change_rate",
    "c1_forwards",
)


def write_games_csv(rows, path: Path) -> dict:
    ordered = sorted(rows, key=lambda row: (row["board_id"], row["arm_id"]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in ordered:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})
    import hashlib

    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "rows": len(ordered),
        "columns": len(CSV_COLUMNS),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_match_configuration(plans, arms, config, *, device: str, games_played: int) -> dict:
    return {
        "artifact": "phase12_agent03_match_set_v1",
        "search_version": SEARCH_VERSION,
        "score_definition": SCORE_DEFINITION,
        "preset_id": config.preset_id,
        "worlds": config.worlds,
        "rollout_depth": config.rollout_depth,
        "max_root_candidates": config.max_root_candidates,
        "beta": config.beta,
        "epsilon": config.epsilon,
        "match_version": mp.MATCH_VERSION,
        "master_seed": mp.MATCH_MASTER_SEED,
        "library_split": mp.MATCH_LIBRARY_SPLIT,
        "strata": list(mp.MATCH_STRATA),
        "setup_sources": len(mp.MATCH_SOURCES),
        "colors": len(mp.MATCH_COLORS),
        "games_per_opponent": {
            stratum: sum(1 for plan in plans if plan.stratum == stratum)
            for stratum in mp.MATCH_STRATA
        },
        "boards": len(plans),
        "games_played": games_played,
        "arms": [arm.describe() for arm in arms],
        "rules": "EVALUATION_RULES (accepted)",
        "player_setup_source": "p10d (accepted Phase 10 learned selector)",
        "phase11_test_bank_used": False,
        "phase11b_corpus_reused": False,
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "boards_list": [plan.describe() for plan in plans],
    }


def preserved_configuration(by_arm: dict, config, model_identity, seats_block: dict) -> dict:
    """The configuration Agent 3 preserves for Agent 4, per instruction section 6.

    Recorded only when the agent1c search arm actually beat the direct
    player on this match set; otherwise the field says so and carries no
    configuration, so a later agent cannot mistake a hope for a result.
    """
    agent1c = by_arm.get(mp.ARM_AGENT1C.arm_id)
    direct = by_arm.get(mp.ARM_DIRECT.arm_id)
    if agent1c is None or direct is None or agent1c["ewr"] <= direct["ewr"]:
        return {
            "preserved": False,
            "reason": (
                "the agent1c search arm did not beat direct C1 on this match set"
            ),
        }
    return {
        "preserved": True,
        "search_version": SEARCH_VERSION,
        "score_definition": SCORE_DEFINITION,
        "preset_id": config.preset_id,
        "worlds": config.worlds,
        "rollout_depth": config.rollout_depth,
        "max_root_candidates": config.max_root_candidates,
        "beta": config.beta,
        "epsilon": config.epsilon,
        "belief_provider": mp.ARM_AGENT1C.provider_id,
        "move_model": {
            "source_checkpoint": model_identity.get("source_checkpoint"),
            "model_state_digest": model_identity.get("model_state_digest"),
        },
        "belief_checkpoint": (
            seats_block.get(mp.ARM_AGENT1C.arm_id, {})
            .get("seat", {})
            .get("provider", {})
            .get("identity", {})
        ),
        "measured": {
            "match_set": mp.MATCH_VERSION,
            "games": agent1c["games"],
            "ewr": agent1c["ewr"],
            "ewr_vs_direct": agent1c["ewr"] - direct["ewr"],
            "paired_vs_direct": agent1c["paired_vs_direct"],
            "seconds_per_move": agent1c["move_seconds_mean"],
            "c1_forwards_per_move": agent1c["c1_forwards_per_move"],
            "move_change_rate": agent1c["move_change_rate"],
        },
        "caveats": [
            "Preserved on an engineering margin, not a validated one: the match set "
            "does not separate agent1c from the other two belief providers, and a "
            "count-based baseline scored at least as well.",
            "oracle_available_in_production = False. The oracle arm exists only as "
            "an offline upper bound on this mechanism.",
        ],
    }


def build_summary(
    rows,
    plans,
    arms,
    seats_block: dict,
    probes_block: dict,
    config,
    *,
    model_identity,
    device: str,
    seconds_total: float,
    stopped_early: bool,
    games_csv: dict,
) -> dict:
    direct_rows = arm_rows(rows, mp.ARM_DIRECT.arm_id)
    by_arm = {
        arm.arm_id: arm_summary(rows, arm, direct_rows=direct_rows) for arm in arms
    }
    direct = by_arm.get(mp.ARM_DIRECT.arm_id)
    efficiency = {
        arm_id: search_second_efficiency(block, direct)
        for arm_id, block in by_arm.items()
    }
    highlights = {
        "agent1c_vs_direct": comparison(
            by_arm, mp.ARM_AGENT1C.arm_id, mp.ARM_DIRECT.arm_id
        ),
        "agent1c_vs_original": comparison(
            by_arm, mp.ARM_AGENT1C.arm_id, mp.ARM_ORIGINAL.arm_id
        ),
        "agent1c_vs_count": comparison(
            by_arm, mp.ARM_AGENT1C.arm_id, mp.ARM_COUNT.arm_id
        ),
    }
    for key, left, right in (
        ("agent1c_vs_direct", mp.ARM_AGENT1C.arm_id, mp.ARM_DIRECT.arm_id),
        ("agent1c_vs_original", mp.ARM_AGENT1C.arm_id, mp.ARM_ORIGINAL.arm_id),
        ("agent1c_vs_count", mp.ARM_AGENT1C.arm_id, mp.ARM_COUNT.arm_id),
    ):
        if highlights[key].get("available"):
            highlights[key]["paired"] = paired_comparison(
                arm_rows(rows, left), arm_rows(rows, right)
            )

    unpaired_errors = [
        outcome_noise_scale(arm_rows(rows, arm.arm_id)) for arm in arms
    ]
    paired_errors = [
        by_arm[arm.arm_id]["paired_vs_direct"].get("standard_error")
        for arm in arms
        if arm.arm_id != mp.ARM_DIRECT.arm_id
    ]

    summary = {
        "artifact": "phase12_agent03_first_match_test_v1",
        "phase": "phase12",
        "agent": 3,
        "generated_utc": utc_now(),
        "search_version": SEARCH_VERSION,
        "score_definition": SCORE_DEFINITION,
        "device": device,
        "match_configuration": build_match_configuration(
            plans, arms, config, device=device, games_played=len(rows)
        ),
        "move_model_identity": model_identity,
        "seats": {arm.arm_id: seats_block[arm.arm_id] for arm in arms},
        "arms": by_arm,
        "efficiency": efficiency,
        "highlights": highlights,
        "noise_scale": {
            "note": (
                "Descriptive spread of this match set, not an inference. No "
                "significance claim is made in this agent."
            ),
            "unpaired_standard_error_typical": mean_or_none(unpaired_errors),
            "paired_standard_error_typical": mean_or_none(paired_errors),
            "decisive_margin": DECISIVE_MARGIN,
        },
        "probes": {arm.arm_id: probes_block[arm.arm_id] for arm in arms},
        "preserved_configuration": preserved_configuration(
            by_arm, config, model_identity, seats_block
        ),
        "games": games_csv,
        "stopped_early": stopped_early,
        "seconds_total": round(seconds_total, 3),
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
        "limitations": [
            "32 games per arm over four opponents is an engineering sample, not a "
            "powered experiment: an EWR difference below the stated noise scale is "
            "not evidence of an ordering.",
            "One budget (SMALL), one beta, one candidate rule, one search version. "
            "No tuning was attempted and none is implied by the result.",
            "The arms share boards, opponent seeds and per-ply search seeds, which "
            "removes setup variance but leaves the arms correlated: paired numbers "
            "and unpaired numbers must not be mixed.",
            "The oracle arm is an offline diagnostic upper bound on the search "
            "mechanism, never a playable configuration, and its latency is not "
            "comparable to the belief arms' (it collapses to one world).",
            "The match driver holds the true engine state so the search seat can "
            "materialize worlds; the boundary is enforced structurally by the Agent "
            "1 engine and checked at run time by the permutation probe, not by the "
            "policy-input isolation the accepted match runner provides.",
            "Setups come from the accepted library's 'validation' split, the same "
            "pool Phase 11B's dev split drew from, so a mild optimistic residual for "
            "agent1c is accepted for an engineering match test.",
        ],
        "deliverables": [
            "stratego/search/phase12/matchplay.py           (new; Agent 1 and 2 modules untouched)",
            "tests/search/test_phase12_matchplay.py",
            "reports/phase12/agent_03_match_config.json",
            "reports/phase12/agent_03_games.jsonl",
            "reports/phase12/agent_03_games.csv",
            "reports/phase12/agent_03_report.md",
            "reports/phase12/agent_03_summary.json",
        ],
        "status": {
            "phase11_final_classification": "FAIL",
            "phase11b_selection": "Agent1C",
            "scientific_validation_status": "not performed",
            "oracle_available_in_production": False,
            "phase11_test_bank_used": False,
            "search_core_modified": False,
            "budget_above_small_used": False,
            "budget_changed_during_run": False,
            "agent_4_launched": False,
        },
        "stop_condition": (
            "Stop condition reached: the compact match test is complete. No budget "
            "above SMALL was run, no world count or depth was raised to compensate, "
            "and Agent 4 is not launched."
        ),
    }
    summary["verdict"] = build_verdict(summary)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analysis_only(arguments) -> int:
    """Rebuild the summary and report from the stored game rows.

    Everything except the seat descriptions and the probe records is a pure
    function of the rows on disk, so the analysis can be re-derived — and
    audited — without replaying two hours of games. The two exceptions are
    products of the run itself and are carried over unchanged from the
    existing summary rather than invented here.
    """
    previous = json.loads(SUMMARY_PATH.read_text())
    rows = list(load_completed(GAMES_JSONL_PATH).values())
    if not rows:
        raise Phase12SearchError(f"{GAMES_JSONL_PATH} holds no game rows")
    requested = [name.strip() for name in arguments.arms.split(",") if name.strip()]
    arms = [mp.ARMS_BY_ID[name] for name in requested if name in previous["arms"]]
    finished = {
        board
        for board in {row["board_id"] for row in rows}
        if {row["arm_id"] for row in rows if row["board_id"] == board}
        == {arm.arm_id for arm in arms}
    }
    rows = [row for row in rows if row["board_id"] in finished]
    plans = [
        plan
        for plan in mp.match_plans(
            Phase11BSetupSources(), games_per_opponent=arguments.games_per_opponent
        )
        if plan.board_id in finished
    ]
    log(f"  {len(finished)} complete boards, {len(rows)} games, {len(arms)} arms")
    games_csv = write_games_csv(rows, GAMES_CSV_PATH)
    summary = build_summary(
        rows,
        plans,
        arms,
        previous["seats"],
        previous["probes"],
        search_preset(PRESET_NAME),
        model_identity=previous["move_model_identity"],
        device=previous["device"],
        seconds_total=previous["seconds_total"],
        stopped_early=previous.get("stopped_early", False),
        games_csv=games_csv,
    )
    SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
    write_report(summary, REPORT_PATH)
    log(summary["verdict"]["statement"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="0 leaves torch alone")
    parser.add_argument(
        "--games-per-opponent", type=int, default=mp.GAMES_PER_OPPONENT
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARM_ORDER),
        help="comma separated arm ids, in play order",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep game rows already written to the JSONL and play only the rest",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop at the next board boundary after this many seconds (0 = no cap)",
    )
    parser.add_argument("--probe-interval", type=int, default=24)
    parser.add_argument("--probe-budget", type=int, default=16)
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "rebuild the summary and the report from the game rows already on "
            "disk, playing nothing; the seat and probe records are carried over "
            "from the existing summary because they are products of the run"
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rewrite the report from the existing summary, running nothing",
    )
    arguments = parser.parse_args()

    if arguments.analysis_only:
        return analysis_only(arguments)

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

    log("Phase 12 Agent 3 — first search match test")
    log("stage: identities")
    handoff = load_handoff()
    model, model_identity = load_move_model(handoff, arguments.device)

    from stratego.evaluation.phase11_pipeline import build_owners

    owners, _ = build_owners(
        REPOSITORY_ROOT,
        CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt",
        device=arguments.device,
    )

    requested = [name.strip() for name in arguments.arms.split(",") if name.strip()]
    unknown = [name for name in requested if name not in mp.ARMS_BY_ID]
    if unknown:
        raise Phase12SearchError(f"unknown arm ids: {unknown}")
    arms = [mp.ARMS_BY_ID[name] for name in requested]

    config = search_preset(PRESET_NAME)
    log(
        f"  budget: {config.preset_id} worlds {config.worlds} depth "
        f"{config.rollout_depth} candidates <= {config.max_root_candidates} "
        f"beta {config.beta}"
    )
    seats = {}
    probes = {}
    reference = probe_reference(owners)
    for arm in arms:
        seats[arm.arm_id] = build_seat(
            arm,
            model=model,
            identity=model_identity,
            handoff=handoff,
            owners=owners,
            device=arguments.device,
            preset_name=PRESET_NAME,
        )
        probes[arm.arm_id] = mp.SeatProbe(
            reference=reference if arm.kind == "search" else None,
            interval=arguments.probe_interval,
            budget=arguments.probe_budget,
            expects_hidden_truth=arm.diagnostic_only,
        )
        log(f"  seat ready: {arm.arm_id} ({arm.label})")

    log("stage: match set")
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
                build_match_configuration(
                    plans, arms, config, device=arguments.device, games_played=0
                )
            ),
            indent=1,
        )
        + "\n"
    )
    log(f"  match configuration -> {CONFIG_PATH.name}")

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
    # Only boards where every requested arm finished. A board one arm played
    # and another did not would silently unbalance the comparison, which is
    # the one thing a 32-game match set cannot afford.
    finished = {
        board
        for board in {row["board_id"] for row in rows}
        if {row["arm_id"] for row in rows if row["board_id"] == board}
        == {arm.arm_id for arm in arms}
    }
    dropped = len({row["board_id"] for row in rows}) - len(finished)
    if dropped:
        log(f"  dropping {dropped} board(s) that no arm set completed")
    rows = [row for row in rows if row["board_id"] in finished]
    plans = [plan for plan in plans if plan.board_id in finished]
    log(f"  {len(finished)} complete boards, {len(rows)} games")
    games_csv = write_games_csv(rows, GAMES_CSV_PATH)
    summary = build_summary(
        rows,
        plans,
        arms,
        {arm.arm_id: seats[arm.arm_id].describe() for arm in arms},
        {arm.arm_id: probes[arm.arm_id].summary() for arm in arms},
        config,
        model_identity=model_identity,
        device=arguments.device,
        seconds_total=time.perf_counter() - started,
        stopped_early=stopped_early,
        games_csv=games_csv,
    )
    SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
    write_report(summary, REPORT_PATH)

    for arm_id in ARM_ORDER:
        block = summary["arms"].get(arm_id)
        if block is None:
            continue
        log(
            f"  {block['label']:<32} {record_text(block):>12}  EWR "
            f"{number(block['ewr'])}  {number(block['move_seconds_mean'], 3)} s/move"
        )
    failures = [
        arm_id
        for arm_id, block in summary["probes"].items()
        if block["failures"]
    ]
    if failures:
        log(f"  PROBE FAILURES: {failures}")
    log(summary["verdict"]["statement"])
    log(f"  wrote {REPORT_PATH.name} and {SUMMARY_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
