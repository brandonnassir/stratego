#!/usr/bin/env python
"""Phase 12 Agent 5 runner: productionize the working search player.

Specification source: `06_PHASE_12_AGENT_5_WORKING_SEARCH_PLAYER.md`.

Engineering integration, not another experiment
-----------------------------------------------
Agents 1-4 built, validated and budgeted the search. This runner does the
one thing left: prove the productionized player — accepted Phase 9 C1
policy/value, Agent 1C beliefs, TINY search, a 0.5 s time cap, direct-C1
fallback — loads the right bytes, plays legally, falls back correctly, and
is selectable through the project's seats and the human CLI. Then it
freezes `phase12_search_candidate_v1` and stops.

The quick checks are exactly the instructed list. The smoke set is 16
boards (4 per opponent, ordinal 0 of the accepted match pack) played by the
working player in `direct` and `tiny` modes — boards Agent 4 already
played, so beyond "games complete legally" the run checks that the
productionized seat replays Agent 4's games move for move. No new
tournament, no strength pack, no Agent 3/4 rerun.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.state import create_game  # noqa: E402
from stratego.evaluation.match_spec import EVALUATION_RULES  # noqa: E402
from stratego.evaluation.neural_worker import (  # noqa: E402
    DECISION_MODE_GREEDY,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from stratego.evaluation.policy import PolicyRef  # noqa: E402
from stratego.training.phase10_acceptance import effective_win_rate  # noqa: E402
from stratego.search.phase12 import matchplay as mp  # noqa: E402
from stratego.search.phase12 import player as pl  # noqa: E402
from stratego.search.phase12.contract import (  # noqa: E402
    PROVIDER_ORACLE,
    Phase12SearchError,
    Phase12SearchTimeout,
    SEARCH_PRESETS,
)
from stratego.search.phase12.engine import Phase12SearchEngine  # noqa: E402
from stratego.search.phase12.providers import (  # noqa: E402
    OracleBeliefProvider,
    build_belief_provider,
)

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase12"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase12"
AGENT4_SUMMARY_PATH = REPORT_DIRECTORY / "agent_04_summary.json"
AGENT4_GAMES_PATH = REPORT_DIRECTORY / "agent_04_games.jsonl"
SMOKE_JSONL_PATH = REPORT_DIRECTORY / "agent_05_smoke_games.jsonl"
SMOKE_CSV_PATH = REPORT_DIRECTORY / "agent_05_smoke_games.csv"
REPORT_PATH = REPORT_DIRECTORY / "agent_05_report.md"
SUMMARY_PATH = REPORT_DIRECTORY / "agent_05_summary.json"
CANDIDATE_PATH = CHECKPOINT_DIRECTORY / "phase12_search_candidate_v1.json"
CLI_PATH = REPOSITORY_ROOT / "scripts" / "play_phase12.py"

#: 4 games per opponent = the 16 ordinal-0 boards of the accepted match
#: pack, all of them inside Agent 4's 64-board set.
SMOKE_GAMES_PER_OPPONENT = 4

#: Working-player arm -> the Agent 4 arm whose games it must replay.
REPLAY_OF = {
    "player_direct": "direct_c1",
    "player_search_tiny": "search_agent1c_tiny",
    "player_search_medium": "search_agent1c_medium",
}
REPLAY_FIELDS = ("outcome", "plies", "player_decisions", "c1_forwards", "move_changes")

#: Boards the MEDIUM maximum-strength candidate replays (a spot check, not
#: a pack): the first contested strategic_rule boards of ordinal 0.
MEDIUM_SMOKE_BOARDS = 2


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize(value):
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


def quantile(values, fraction: float):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    index = max(0, min(len(values) - 1, int(round(fraction * (len(values) - 1)))))
    return values[index]


def number(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------
# Stage: quick integration checks
# ---------------------------------------------------------------------------


class Checks:
    """The instructed check list, each one recorded pass/fail with detail."""

    def __init__(self):
        self.results: list = []

    def run(self, name: str, function) -> None:
        try:
            detail = function()
            self.results.append({"check": name, "passed": True, "detail": detail})
            log(f"  PASS  {name}: {detail}")
        except Exception as error:  # noqa: BLE001 - every failure must be listed
            self.results.append(
                {"check": name, "passed": False, "detail": f"{type(error).__name__}: {error}"}
            )
            log(f"  FAIL  {name}: {type(error).__name__}: {error}")

    @property
    def all_passed(self) -> bool:
        return all(entry["passed"] for entry in self.results)

    def summary(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "checks": {entry["check"]: entry["passed"] for entry in self.results},
            "details": list(self.results),
        }


def expect_refusal(function, description: str) -> str:
    try:
        function()
    except Phase12SearchError:
        return f"refused: {description}"
    raise AssertionError(f"NOT refused: {description}")


def check_state(sources):
    """A real opening position from the accepted setup library."""
    plan = mp.match_plan("strategic_rule", "p10d", "red", 0, sources)
    return create_game(
        plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id="agent05_check"
    )


def run_quick_checks(player, identities, sources, arguments) -> Checks:
    checks = Checks()
    handoff = identities["handoff"]
    state = check_state(sources)
    legal = set(legal_actions(state))

    def loads():
        described = player.describe()
        assert described["player_version"] == pl.PLAYER_VERSION
        assert described["mode"] == pl.DEFAULT_MODE == "tiny"
        assert set(described["presets"]) == {"tiny", "small", "medium"}
        return (
            f"{pl.PLAYER_VERSION}, default mode tiny "
            f"(TINY: 8 worlds, depth 4, cap {player.time_caps['tiny']}s)"
        )

    checks.run("search_player_loads", loads)

    def phase9_identity():
        observed = identities["move_model_identity"]["model_state_digest"]
        expected = handoff["accepted_phase9_checkpoint"]["model_state_digest"]
        assert observed == expected, f"{observed} != {expected}"
        return f"model_state_digest {observed[:16]}… == handoff"

    checks.run("phase9_checkpoint_identity", phase9_identity)

    def agent1c_identity():
        observed = identities["belief_model_identity"]["checkpoint_sha256"]
        expected = handoff["agent1c_checkpoint"]["sha256"]
        assert observed == expected, f"{observed} != {expected}"
        state_digest = identities["belief_model_identity"]["state_dict_digest"]
        assert state_digest == handoff["agent1c_checkpoint"]["state_dict_digest"]
        return f"sha256 {observed[:16]}… == handoff, state digest bound"

    checks.run("agent1c_checkpoint_identity", agent1c_identity)

    def oracle_blocked():
        details = [
            expect_refusal(
                lambda: build_belief_provider(PROVIDER_ORACLE, production=True),
                "provider factory under production=True",
            ),
            expect_refusal(
                lambda: pl.Phase12SearchPlayer(
                    player.model, OracleBeliefProvider(offline_diagnostic=True)
                ),
                "Phase12SearchPlayer constructor",
            ),
            expect_refusal(
                lambda: Phase12SearchEngine(
                    player.model,
                    OracleBeliefProvider(offline_diagnostic=True),
                    SEARCH_PRESETS["TINY"],
                ),
                "engine under a production config",
            ),
            expect_refusal(lambda: pl.check_mode("oracle"), "player mode named 'oracle'"),
        ]
        assert pl.ORACLE_AVAILABLE_IN_PRODUCTION is False
        return f"{len(details)} independent refusals"

    checks.run("oracle_cannot_enter_production", oracle_blocked)

    def direct_works():
        first = player.decide(state, mode="direct")
        second = player.decide(state, mode="direct")
        assert first.action_id in legal and first.fallback_reason is None
        assert first.action_id == second.action_id, "direct mode is not deterministic"
        assert first.used_search is False
        return f"legal, deterministic, {first.seconds:.4f}s"

    checks.run("direct_mode_works", direct_works)

    def tiny_works():
        decision = player.decide(state, seed=2026082005)
        assert decision.action_id in legal
        assert decision.used_search is True and decision.fallback_reason is None
        assert decision.search.unique_worlds >= 1
        assert decision.seconds < player.time_caps["tiny"], "TINY ran past its cap"
        return (
            f"legal, {decision.search.unique_worlds} unique worlds, "
            f"{decision.search.c1_forwards} forwards, {decision.seconds:.3f}s "
            f"(cap {player.time_caps['tiny']}s)"
        )

    checks.run("normal_tiny_search_works", tiny_works)

    def deterministic():
        first = player.decide(state, seed=2026082005)
        second = player.decide(state, seed=2026082005)
        assert first.action_id == second.action_id
        assert first.search.c1_forwards == second.search.c1_forwards
        return "same seed, same action, same forward count"

    checks.run("search_decision_deterministic", deterministic)

    def deadline_neutral():
        engine = player.engines["tiny"]
        bare = engine.choose_action(state, seed=7)
        roomy = engine.choose_action(state, seed=7, deadline=time.perf_counter() + 60.0)
        assert bare.selected_action_id == roomy.selected_action_id
        assert bare.c1_forwards == roomy.c1_forwards
        assert [c.score for c in bare.candidates] == [c.score for c in roomy.candidates]
        return "deadline off/roomy: bit-identical decision on the accepted weights"

    checks.run("deadline_checks_are_behavior_neutral", deadline_neutral)

    def timeout_fallback():
        direct_reference = player.decide(state, mode="direct").action_id
        before = player.fallback_counts[pl.FALLBACK_TIMEOUT]
        original = player.time_caps["tiny"]
        player.time_caps["tiny"] = 1e-4
        try:
            decision = player.decide(state, seed=2026082005)
        finally:
            player.time_caps["tiny"] = original
        assert decision.fallback_reason == pl.FALLBACK_TIMEOUT
        assert decision.used_search is False
        assert decision.action_id == direct_reference
        assert player.fallback_counts[pl.FALLBACK_TIMEOUT] == before + 1
        return "0.1 ms cap trips; direct accepted action played and counted"

    checks.run("timeout_fallback_works", timeout_fallback)

    def forced_error_fallback():
        class Broken:
            config = SEARCH_PRESETS["TINY"]

            def choose_action(self, state, *, seed, deadline=None):
                raise Phase12SearchError("forced integration-check failure")

        from types import SimpleNamespace

        original = player.engines["tiny"]
        outcomes = []
        try:
            player.engines["tiny"] = Broken()
            decision = player.decide(state, seed=1)
            outcomes.append((decision.fallback_reason, decision.action_id in legal))

            class NonFinite(Broken):
                def choose_action(self, state, *, seed, deadline=None):
                    return SimpleNamespace(
                        selected_action_id=int(min(legal)),
                        direct_action_id=int(min(legal)),
                        move_changed=False,
                        candidates=(SimpleNamespace(score=float("nan")),),
                    )

            player.engines["tiny"] = NonFinite()
            decision = player.decide(state, seed=1)
            outcomes.append((decision.fallback_reason, decision.action_id in legal))

            class Illegal(Broken):
                def choose_action(self, state, *, seed, deadline=None):
                    return SimpleNamespace(
                        selected_action_id=-1,
                        direct_action_id=-1,
                        move_changed=False,
                        candidates=(SimpleNamespace(score=0.0),),
                    )

            player.engines["tiny"] = Illegal()
            decision = player.decide(state, seed=1)
            outcomes.append((decision.fallback_reason, decision.action_id in legal))
        finally:
            player.engines["tiny"] = original
        assert outcomes == [
            (pl.FALLBACK_SEARCH_ERROR, True),
            (pl.FALLBACK_NON_FINITE, True),
            (pl.FALLBACK_ILLEGAL_ACTION, True),
        ], outcomes
        confirm = player.decide(state, seed=2026082005)
        assert confirm.used_search is True and confirm.fallback_reason is None
        return "search_error, non_finite_score, illegal_action all fall back legally"

    checks.run("forced_error_fallback_works", forced_error_fallback)

    def seats_select_modes():
        direct_seat = pl.Phase12PlayerSeat(player, "direct")
        tiny_seat = pl.Phase12PlayerSeat(player, "tiny")
        assert direct_seat.arm.arm_id == "player_direct"
        assert tiny_seat.arm.arm_id == "player_search_tiny"
        assert tiny_seat.arm.provider_id == "agent1c"
        previous = player.set_mode("small")
        assert player.status()["mode"] == "small"
        player.set_mode(previous)
        return "machine seats select direct/tiny; set_mode('small') visible in status"

    checks.run("machine_interface_selects_search_mode", seats_select_modes)

    def cli_selects_modes():
        spec = importlib.util.spec_from_file_location("play_phase12", CLI_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.build_parser()
        arguments = parser.parse_args(["--red", "human", "--blue", "small"])
        assert arguments.red == "human" and arguments.blue == "small"
        arguments = parser.parse_args([])
        assert arguments.red == "human" and arguments.blue == "tiny"
        assert "oracle" not in module.SEAT_CHOICES
        try:
            parser.parse_args(["--blue", "oracle"])
        except SystemExit:
            pass
        else:
            raise AssertionError("the CLI accepted an oracle seat")
        return (
            f"seats {module.SEAT_CHOICES}; defaults human vs tiny; oracle not a choice"
        )

    checks.run("human_interface_selects_search_mode", cli_selects_modes)
    return checks


# ---------------------------------------------------------------------------
# Stage: smoke games
# ---------------------------------------------------------------------------


def game_payload(record: mp.GameRecord) -> dict:
    payload = record.row()
    payload["move_seconds"] = [round(float(row["seconds"]), 5) for row in record.moves]
    payload["move_forwards"] = [int(row["c1_forwards"] or 0) for row in record.moves]
    payload["move_changed"] = [
        None if row["move_changed"] is None else int(row["move_changed"])
        for row in record.moves
    ]
    payload["move_fallbacks"] = [row.get("fallback_reason") for row in record.moves]
    payload["fallbacks"] = sum(1 for row in record.moves if row.get("fallback_reason"))
    return payload


def probe_reference(owners) -> RemoteNeuralPolicy:
    return RemoteNeuralPolicy(
        PolicyRef("phase12_agent05_probe_reference_v1", mp.MATCH_VERSION),
        LocalInferenceChannel(owners["phase9"]),
        decision_mode=DECISION_MODE_GREEDY,
    )


def play_smoke(player, sources, arguments) -> tuple:
    """The 16 ordinal-0 boards, direct then tiny, through the accepted loop."""
    from stratego.evaluation.phase11_pipeline import build_owners

    owners, _ = build_owners(
        REPOSITORY_ROOT,
        CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt",
        device=arguments.device,
    )
    plans = mp.match_plans(sources, games_per_opponent=arguments.games_per_opponent)
    seats = {
        "player_direct": (pl.Phase12PlayerSeat(player, "direct"), plans),
        "player_search_tiny": (pl.Phase12PlayerSeat(player, "tiny"), plans),
    }
    if arguments.medium_boards:
        # The MEDIUM maximum-strength candidate: replay a couple of
        # contested boards, enough to prove the mode end to end without
        # growing a tournament. strategic_rule ordinal 0 is contested on
        # every cell in the Agent 4 record.
        medium_plans = [
            plan for plan in plans if plan.stratum == "strategic_rule"
        ][: arguments.medium_boards]
        seats["player_search_medium"] = (
            pl.Phase12PlayerSeat(player, "medium"), medium_plans,
        )
    reference = probe_reference(owners)
    probes = {
        arm_id: mp.SeatProbe(
            reference=reference if seat.kind == "search" else None,
            interval=24,
            budget=8,
        )
        for arm_id, (seat, _) in seats.items()
    }
    log(
        "  boards per mode: "
        + ", ".join(f"{arm_id} {len(mode_plans)}" for arm_id, (_, mode_plans) in seats.items())
    )
    rows = []
    started = time.perf_counter()
    with SMOKE_JSONL_PATH.open("w") as stream:
        for index, plan in enumerate(plans):
            outcomes = []
            for arm_id, (seat, mode_plans) in seats.items():
                if plan not in mode_plans:
                    continue
                record = mp.play_arm_game(plan, seat, owners, probe=probes[arm_id])
                payload = game_payload(record)
                stream.write(json.dumps(payload) + "\n")
                rows.append(payload)
                outcomes.append(f"{arm_id.split('_')[-1]}={payload['outcome'][0].upper()}")
            log(
                f"  [{index + 1}/{len(plans)}] {plan.stratum}/{plan.setup_source}/"
                f"{plan.player_color} {' '.join(outcomes)} "
                f"{time.perf_counter() - started:.0f}s"
            )
    with SMOKE_CSV_PATH.open("w", newline="") as stream:
        fields = [key for key in rows[0] if not key.startswith("move_")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["board_id"], r["arm_id"])):
            writer.writerow(row)
    return rows, {arm_id: probe.summary() for arm_id, probe in probes.items()}


def smoke_analysis(rows, player) -> dict:
    """Legality, completion, latency, fallbacks, and the Agent 4 replay."""
    by_arm: dict = {}
    for arm_id in REPLAY_OF:
        mine = [row for row in rows if row["arm_id"] == arm_id]
        if not mine:
            continue
        latencies = [second for row in mine for second in row["move_seconds"]]
        scores = [float(row["effective_score"]) for row in mine]
        by_arm[arm_id] = {
            "games": len(mine),
            "wins": sum(1 for row in mine if row["outcome"] == "win"),
            "draws": sum(1 for row in mine if row["outcome"] == "draw"),
            "losses": sum(1 for row in mine if row["outcome"] == "loss"),
            "ewr": effective_win_rate(scores) if scores else None,
            "player_decisions": sum(int(row["player_decisions"]) for row in mine),
            "fallbacks": sum(int(row["fallbacks"]) for row in mine),
            "move_seconds_median": statistics.median(latencies) if latencies else None,
            "move_seconds_p95": quantile(latencies, 0.95),
            "move_seconds_max": max(latencies) if latencies else None,
            "terminal_reasons": sorted({row["terminal_reason"] for row in mine}),
        }

    theirs: dict = {}
    if AGENT4_GAMES_PATH.exists():
        for line in AGENT4_GAMES_PATH.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            theirs[(row["arm_id"], row["board_id"])] = row
    replay = {}
    for arm_id, agent4_arm in REPLAY_OF.items():
        mine = [row for row in rows if row["arm_id"] == arm_id]
        if not mine:
            continue
        mismatches = []
        shared = 0
        for row in mine:
            partner = theirs.get((agent4_arm, row["board_id"]))
            if partner is None:
                continue
            shared += 1
            differing = {
                f: [partner[f], row[f]] for f in REPLAY_FIELDS if partner[f] != row[f]
            }
            if differing:
                mismatches.append({"board_id": row["board_id"], "fields": differing})
        replay[arm_id] = {
            "against": agent4_arm,
            "shared_boards": shared,
            "identical_boards": shared - len(mismatches),
            "mismatches": mismatches,
            "reproduced": shared > 0 and not mismatches,
        }
    return {
        "arms": by_arm,
        "replay_of_agent4": replay,
        "player_fallback_counts": dict(player.fallback_counts),
        "all_games_completed": all(
            row["outcome"] in ("win", "draw", "loss") for row in rows
        ),
        "smoke_fallbacks_total": sum(int(row["fallbacks"]) for row in rows),
    }


# ---------------------------------------------------------------------------
# Stage: the candidate artifact
# ---------------------------------------------------------------------------


def load_agent4_operating_point() -> dict:
    summary = json.loads(AGENT4_SUMMARY_PATH.read_text())
    rule = summary["stopping_rule"]["operating_point"]
    if rule["selected_preset_id"] != "TINY":
        raise Phase12SearchError(
            f"Agent 4 selected {rule['selected_preset_id']}, not TINY; the frozen "
            "default in the instruction no longer matches the record"
        )
    selected = rule["selected"]
    direct = summary["arms"]["direct_c1"]
    point = {
        "source": str(AGENT4_SUMMARY_PATH.relative_to(REPOSITORY_ROOT)),
        "preset_id": selected["preset_id"],
        "games": selected["games"],
        "ewr": selected["ewr"],
        "direct_ewr": direct["ewr"],
        "move_seconds_median": selected["move_seconds_median"],
        "move_seconds_p95": selected["move_seconds_p95"],
        "move_seconds_max": summary["arms"]["search_agent1c_tiny"]["move_seconds_max"],
        "search_seconds_per_game": selected["search_seconds_per_game"],
    }
    # The instruction's headline numbers must be these numbers, rounded.
    literals = {
        "Agent4_quick_EWR": (round(point["ewr"], 4), 0.6406),
        "Agent4_direct_EWR": (round(point["direct_ewr"], 4), 0.5234),
        "expected_latency_median": (round(point["move_seconds_median"], 3), 0.126),
        "expected_latency_p95": (round(point["move_seconds_p95"], 3), 0.138),
    }
    for name, (observed, instructed) in literals.items():
        if observed != instructed:
            raise Phase12SearchError(
                f"{name}: Agent 4 record rounds to {observed}, instruction says "
                f"{instructed}; refusing to freeze a mismatched artifact"
            )
    return point


def load_agent4_medium_point() -> dict:
    """The MEDIUM rung's exact Agent 4 numbers, for the max-strength block."""
    summary = json.loads(AGENT4_SUMMARY_PATH.read_text())
    medium = summary["arms"]["search_agent1c_medium"]
    point = {
        "source": str(AGENT4_SUMMARY_PATH.relative_to(REPOSITORY_ROOT)),
        "preset_id": "MEDIUM",
        "games": medium["games"],
        "ewr": medium["ewr"],
        "move_seconds_median": medium["move_seconds_median"],
        "move_seconds_p95": medium["move_seconds_p95"],
        "move_seconds_max": medium["move_seconds_max"],
        "search_seconds_per_game": medium["search_seconds_per_game"],
    }
    if round(point["ewr"], 4) != 0.6875:
        raise Phase12SearchError(
            f"MEDIUM EWR rounds to {round(point['ewr'], 4)}, expected 0.6875; the "
            "maximum-strength designation no longer matches the Agent 4 record"
        )
    return point


KNOWN_LIMITATIONS = [
    "Strength numbers are the Agent 4 engineering sample (64 games per rung): "
    "no significance claim, and scientific validation has not been performed.",
    "The Agent 4 ladder did not separate TINY/SMALL/MEDIUM within the 0.10 "
    "engineering margin; TINY is the cheapest rung not meaningfully behind the "
    "strongest on that pack, not a proven optimum.",
    "Latency, and therefore the 0.5 s cap's headroom, were measured on this "
    "machine (cpu, single process); a different device should re-derive the cap "
    "from its own profile, keeping the ~3.6x-over-p95 intent.",
    "The accepted setup library places a flag on a front row on 47 of 64 match "
    "boards, so part of every pack is decided by opening scout races that no "
    "search budget can influence.",
    "Agent 1C was trained on setups from the same accepted library family the "
    "match packs draw from; a mild optimistic residual is accepted for "
    "engineering purposes.",
    "The time cap makes the fallback wall-clock-dependent by design; search "
    "decisions themselves are seed-deterministic.",
]


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------


def write_report(summary: dict) -> None:
    checks = summary["quick_checks"]
    smoke = summary["smoke"]
    candidate = summary["candidate"]
    lines: list = []
    lines.append("# Phase 12 Agent 5 — Working Search-Enhanced Player")
    lines.append("")
    lines.append(f"Generated {summary['generated_utc']} by `scripts/run_phase12_agent05.py`.")
    lines.append("")
    lines.append(
        "Engineering integration of the accepted TINY + Agent 1C configuration into "
        "the project's one working player. No new experiment: quick checks, a 16-board "
        "smoke set already inside Agent 4's pack, and the frozen "
        "`phase12_search_candidate_v1` artifact."
    )
    lines.append("")

    lines.append("## 1. The production stack")
    lines.append("")
    lines.append("```text")
    lines.append("accepted Phase 9 C1   policy + value")
    lines.append("Agent 1C              belief only")
    lines.append(f"search                {candidate['search_version']} @ TINY")
    lines.append("")
    move = candidate["move_model_identity"]
    belief = candidate["belief_model_identity"]
    lines.append(f"phase9 source sha256     {move.get('source_sha256')}")
    lines.append(f"phase9 state digest      {move.get('model_state_digest')}")
    lines.append(f"agent1c sha256           {belief.get('checkpoint_sha256')}")
    lines.append(f"agent1c state digest     {belief.get('state_dict_digest')}")
    lines.append(f"candidate config digest  {candidate['candidate_config_digest']}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Digest-checked at load against the Phase 11B handoff record; the loader "
        "(`stratego.search.phase12.player.load_search_player`) refuses unbound bytes."
    )
    lines.append("")

    lines.append("## 2. Modes, time cap, fallback")
    lines.append("")
    lines.append("```text")
    lines.append("modes            direct | tiny (default) | small | medium (max-strength)")
    lines.append("time caps        tiny 0.5 s   small 1.5 s   medium 3.5 s")
    lines.append("cap rationale    TINY observed 0.126 s median / 0.138 s p95 / 0.193 s max;")
    lines.append("                 0.5 s = 3.6x p95, 2.6x max — headroom, not the p95 itself")
    lines.append(f"fallback         {candidate['fallback_policy']}")
    lines.append("fallback fires   timeout | search_error | unexpected_error |")
    lines.append("                 non_finite_score | illegal_action | direct_error (last resort)")
    lines.append("```")
    lines.append("")
    lines.append(
        "The cap is enforced cooperatively inside the engine (`deadline` parameter — "
        "additive, default off, spot-checked bit-identical on the accepted weights) "
        "and re-checked on completion. Every fallback is counted by reason and logged; "
        "the player never forfeits and never emits an illegal action because search "
        "failed. SMALL remains an engineering/debug mode; the oracle is not a mode "
        "and cannot be constructed into the player (four independent refusals, "
        "checked below)."
    )
    lines.append("")
    maximum = candidate.get("maximum_strength_candidate")
    if maximum:
        lines.append(
            f"**Maximum-strength candidate (by project direction): "
            f"`{maximum['mode']}` (MEDIUM — {maximum['worlds']} worlds, depth "
            f"{maximum['depth']}, cap {maximum['time_cap_seconds']} s).** Agent 4 "
            f"measured it at EWR {number(maximum['agent4_exact']['ewr'])} "
            f"({maximum['expected_latency_median']} median, "
            f"{maximum['expected_latency_p95']} p95), a "
            f"{number(maximum['ewr_lead_over_selected'])} EWR lead over TINY that "
            "sits inside the 0.10 engineering margin — the strongest observed "
            "configuration, not a validated ordering. TINY remains the production "
            "default."
        )
        lines.append("")

    lines.append("## 3. Integration surfaces")
    lines.append("")
    lines.append("```text")
    lines.append("machine vs machine   Phase12PlayerSeat(player, 'direct' | 'tiny' | ...)")
    lines.append("                     through the accepted matchplay driver (play_arm_game)")
    lines.append("human play           scripts/play_phase12.py --red human --blue tiny")
    lines.append("status/logs          player.status(), logger 'stratego.phase12.player'")
    lines.append("```")
    lines.append("")
    lines.append(
        "Search seats draw per-ply world seeds from the accepted match stream, which "
        "is what makes the smoke games below a replay of Agent 4 rather than merely "
        "similar games. The CLI renders only the human's legal observation (unrevealed "
        "opponent pieces stay hidden), shows the active mode, budget and per-move "
        "latency, and accepts direct/tiny/small/medium for either seat."
    )
    lines.append("")

    lines.append("## 4. Quick integration checks")
    lines.append("")
    lines.append("| check | result | detail |")
    lines.append("|---|---|---|")
    for entry in checks["details"]:
        lines.append(
            f"| {entry['check']} | {'PASS' if entry['passed'] else 'FAIL'} | "
            f"{entry['detail']} |"
        )
    lines.append("")

    lines.append("## 5. Smoke set: 16 boards, direct and tiny")
    lines.append("")
    if smoke.get("skipped"):
        lines.append("Skipped this run (`--skip-smoke`); quick checks only.")
        lines.append("")
        smoke = {"arms": {}, "replay_of_agent4": {}, "smoke_fallbacks_total": 0}
    lines.append(
        "Ordinal-0 boards of the accepted match pack (4 per opponent, balanced "
        "sources and colours) — all 16 inside Agent 4's 64-board set, so the working "
        "player's games can be required to replay Agent 4's move for move. The "
        "`medium` maximum-strength candidate replays a 2-board contested spot check "
        "of the same kind (its strength number is Agent 4's, not these 2 games)."
    )
    lines.append("")
    lines.append("| mode | W / D / L | EWR | decisions | fallbacks | median s/move | p95 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm_id, block in smoke["arms"].items():
        lines.append(
            f"| {arm_id} | {block['wins']} / {block['draws']} / {block['losses']} | "
            f"{number(block['ewr'])} | {block['player_decisions']} | "
            f"{block['fallbacks']} | {number(block['move_seconds_median'], 3)} | "
            f"{number(block['move_seconds_p95'], 3)} | "
            f"{number(block['move_seconds_max'], 3)} |"
        )
    lines.append("")
    for arm_id, block in smoke["replay_of_agent4"].items():
        verdict = "replayed exactly" if block["reproduced"] else "MISMATCH"
        lines.append(
            f"- `{arm_id}` vs Agent 4 `{block['against']}`: "
            f"{block['identical_boards']}/{block['shared_boards']} boards identical "
            f"on {', '.join(REPLAY_FIELDS)} — {verdict}."
        )
    lines.append(
        f"- Boundary probes: "
        + "; ".join(
            f"{arm_id} {probe['permutation_checks']} permutation + "
            f"{probe['direct_agreement_checks']} direct-agreement checks, "
            f"{len(probe['failures'])} failures"
            for arm_id, probe in summary["probes"].items()
        )
        + "."
    )
    lines.append(
        f"- Fallbacks during smoke games: {smoke['smoke_fallbacks_total']} "
        f"(cap headroom held; the timeout/error fallbacks were exercised in the "
        "checks above, by force)."
    )
    lines.append("")

    lines.append("## 6. The frozen engineering candidate")
    lines.append("")
    lines.append("```text")
    for key in (
        "artifact", "move_model", "belief_model", "search_version", "selected_preset",
        "worlds", "root_candidates", "depth", "beta", "epsilon",
        "expected_latency_median", "expected_latency_p95", "Agent4_quick_EWR",
        "Agent4_direct_EWR", "time_cap_seconds", "fallback_policy",
        "oracle_available_in_production", "phase11_final_classification",
        "phase11b_selection", "scientific_validation_status",
    ):
        lines.append(f"{key:<32}{candidate[key]}")
    if maximum:
        lines.append("")
        lines.append("maximum_strength_candidate:")
        for key in (
            "mode", "preset", "worlds", "depth", "time_cap_seconds",
            "expected_latency_median", "expected_latency_p95", "Agent4_quick_EWR",
            "ewr_lead_over_selected",
        ):
            lines.append(f"  {key:<30}{maximum[key]}")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Written to `{CANDIDATE_PATH.relative_to(REPOSITORY_ROOT)}` with the full "
        "identity blocks (paths, sha256, state digests, dev metrics) and the exact "
        "un-rounded Agent 4 numbers the headline strings derive from."
    )
    lines.append("")

    lines.append("## 7. Known limitations")
    lines.append("")
    for item in candidate["known_limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 8. Deliverables and status")
    lines.append("")
    lines.append("```text")
    for item in summary["deliverables"]:
        lines.append(item)
    lines.append("")
    for key, value in summary["status"].items():
        lines.append(f"{key:<38}{value}")
    lines.append("```")
    lines.append("")
    lines.append(summary["stop_condition"])
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threads", type=int, default=10,
        help="torch threads; 10 matches the Agent 4 latency/replay environment",
    )
    parser.add_argument(
        "--games-per-opponent", type=int, default=SMOKE_GAMES_PER_OPPONENT,
        help="smoke games per opponent (multiple of 4)",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="run only the quick checks and the artifact (no games)",
    )
    parser.add_argument(
        "--medium-boards", type=int, default=MEDIUM_SMOKE_BOARDS,
        help="boards the MEDIUM maximum-strength candidate replays (0 disables)",
    )
    arguments = parser.parse_args()
    if arguments.threads:
        torch.set_num_threads(int(arguments.threads))
    started = time.perf_counter()
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log("Phase 12 Agent 5 — working search-enhanced player")
    log("stage: identities")
    player, identities = pl.load_search_player(
        REPOSITORY_ROOT, mode=pl.DEFAULT_MODE, device=arguments.device
    )
    move_identity = identities["move_model_identity"]
    belief_identity = identities["belief_model_identity"]
    log(
        f"  move model: accepted Phase 9 C1, state digest "
        f"{move_identity['model_state_digest'][:12]}…"
    )
    log(
        f"  belief model: agent1c, sha256 "
        f"{belief_identity['checkpoint_sha256'][:12]}…"
    )
    sources = Phase11BSetupSources()

    log("stage: quick integration checks")
    checks = run_quick_checks(player, identities, sources, arguments)

    probes: dict = {}
    if arguments.skip_smoke:
        smoke = {"skipped": True}
        rows = []
    else:
        log("stage: smoke games")
        rows, probes = play_smoke(player, sources, arguments)
        smoke = smoke_analysis(rows, player)
        smoke["skipped"] = False
        for arm_id, block in smoke["arms"].items():
            log(
                f"  {arm_id:<22} {block['wins']}/{block['draws']}/{block['losses']}  "
                f"EWR {number(block['ewr'])}  median {number(block['move_seconds_median'], 3)}s  "
                f"fallbacks {block['fallbacks']}"
            )
        for arm_id, block in smoke["replay_of_agent4"].items():
            log(
                f"  replay vs {block['against']}: "
                f"{block['identical_boards']}/{block['shared_boards']} identical"
            )

    log("stage: candidate artifact")
    agent4 = load_agent4_operating_point()
    candidate = pl.build_candidate_record(
        move_model_identity=move_identity,
        belief_model_identity=belief_identity,
        agent4=agent4,
        agent4_medium=load_agent4_medium_point(),
        generated_utc=utc_now(),
        environment={
            "device": arguments.device,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        quick_checks=checks.summary()["checks"],
        known_limitations=KNOWN_LIMITATIONS,
    )
    CANDIDATE_PATH.write_text(json.dumps(sanitize(candidate), indent=1) + "\n")
    log(f"  wrote {CANDIDATE_PATH.relative_to(REPOSITORY_ROOT)}")

    replay_ok = arguments.skip_smoke or all(
        block["reproduced"] for block in smoke["replay_of_agent4"].values()
    )
    probes_ok = all(not block["failures"] for block in probes.values())
    stable = checks.all_passed and replay_ok and probes_ok

    summary = {
        "artifact": "phase12_agent05_working_player_v1",
        "phase": "phase12",
        "agent": 5,
        "generated_utc": utc_now(),
        "player_version": pl.PLAYER_VERSION,
        "search_version": candidate["search_version"],
        "device": arguments.device,
        "quick_checks": checks.summary(),
        "smoke": smoke,
        "probes": probes,
        "candidate": candidate,
        "player_status_final": player.status(),
        "runtime": {
            "seconds_total": time.perf_counter() - started,
            "smoke_games": len(rows),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "deliverables": [
            "stratego/search/phase12/player.py            (new: the working player)",
            "stratego/search/phase12/engine.py            (additive deadline parameter, default off)",
            "stratego/search/phase12/contract.py          (Phase12SearchTimeout)",
            "scripts/play_phase12.py                      (new: human/machine CLI)",
            "scripts/run_phase12_agent05.py",
            "tests/search/test_phase12_player.py",
            "checkpoints/phase12/phase12_search_candidate_v1.json",
            "reports/phase12/agent_05_smoke_games.jsonl",
            "reports/phase12/agent_05_smoke_games.csv",
            "reports/phase12/agent_05_report.md",
            "reports/phase12/agent_05_summary.json",
        ],
        "status": {
            "phase11_final_classification": "FAIL",
            "phase11b_selection": "Agent1C",
            "scientific_validation_status": "not performed",
            "oracle_available_in_production": False,
            "phase11_test_bank_used": False,
            "search_core_modified": (
                "additive only: cooperative deadline parameter, default off, "
                "bit-identity spot-checked and Agent 4 games replayed"
            ),
            "selected_operating_point": "TINY",
            "production_default_mode": pl.DEFAULT_MODE,
            "maximum_strength_candidate": f"MEDIUM (mode '{pl.MAX_STRENGTH_MODE}')",
            "time_cap_seconds": pl.MODE_TIME_CAP_SECONDS[pl.DEFAULT_MODE],
            "working_player_delivered": stable,
            "quick_checks_all_passed": checks.all_passed,
            "agent4_replay_exact": replay_ok,
            "scientific_validation_started": False,
            "final_training_started": False,
        },
        "stop_condition": (
            "Stop condition reached: the working search player and "
            "`phase12_search_candidate_v1` are delivered. No further validation "
            "phase and no final training were started."
            if stable
            else "STOP WITH DEFECTS: see failed checks / replay mismatches above."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(sanitize(summary), indent=1) + "\n")
    write_report(summary)
    log(f"  wrote {REPORT_PATH.name} and {SUMMARY_PATH.name}")
    log(summary["stop_condition"])
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
