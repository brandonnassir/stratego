"""Predeclared rules and readers: Stage 1 summary/filter, Stage 2 selection,
probe analysis, cap rule, pack construction — all hermetic."""

import pytest

from stratego.search.phase16.contract import (
    CONTROL_ARM,
    INTERIM_PACK_ORDINAL,
    arm_name,
    parse_arm_name,
)
from stratego.search.phase16.diagnostics import (
    ORACLE_ARM_16,
    apply_stage1_filter,
    position_cells_16,
    summarize_stage1,
)
from stratego.search.phase16.matchpack import (
    analyse_probe,
    decide_time_caps,
    select_configuration,
)


def _grid_rows(arm, preset, actions_by_position, regret, control_action, tau, tau_r):
    """Synthetic Stage 1 rows: one position -> a list of replay actions."""
    rows = []
    for position, actions in actions_by_position.items():
        for replay, action in enumerate(actions):
            rows.append(
                {
                    "position_id": position,
                    "preset_id": preset,
                    "arm": arm,
                    "tau": tau,
                    "tau_r": tau_r,
                    "replay": replay,
                    "action_id": action,
                    "argmax_action_id": actions[0],
                    "direct_action_id": 1,
                    "changed_from_argmax": int(action != actions[0]),
                    "move_changed_vs_direct": int(action != 1),
                    "matches_control": int(action == control_action),
                    "matches_oracle": int(action == 9),
                    "oracle_q_regret": regret,
                    "legal": 1,
                    "search_seconds": 0.1,
                    "c1_forwards": 100,
                    "unique_worlds": 8,
                    "candidates": 4,
                }
            )
    return rows


def _oracle_rows(preset, floor):
    return [
        {
            "position_id": position,
            "preset_id": preset,
            "arm": ORACLE_ARM_16,
            "replay": 0,
            "ply": 12,
            "unresolved": 8,
            "action_id": 9,
            "oracle_q_regret": floor,
            "legal": 1,
            "seconds": 0.05,
            "candidates": 4,
        }
        for position in ("p1", "p2")
    ]


class TestStage1Summary:
    def _rows(self):
        rows = _oracle_rows("TINY", 0.02)
        # control: same action every replay
        rows += _grid_rows(
            CONTROL_ARM, "TINY", {"p1": [5] * 4, "p2": [5] * 4}, 0.03, 5, 0.0, 0.0
        )
        # a varied arm: half the replays elsewhere, slightly worse regret
        rows += _grid_rows(
            arm_name(0.3, 0.0), "TINY", {"p1": [5, 5, 7, 7], "p2": [5, 7, 5, 7]},
            0.035, 5, 0.3, 0.0,
        )
        return rows

    def test_repeat_rate_and_agreement(self):
        summary = summarize_stage1(self._rows())
        control = summary["arms"][f"{CONTROL_ARM}|TINY"]
        varied = summary["arms"][f"{arm_name(0.3, 0.0)}|TINY"]
        assert control["repeat_rate"] == 1.0
        assert control["agreement_with_tau0"] == 1.0
        assert control["played_move_entropy_nats"] == 0.0
        assert varied["repeat_rate"] == 0.5
        assert varied["agreement_with_tau0"] == 0.5
        assert varied["played_move_entropy_nats"] > 0.6

    def test_regret_excess_reads_against_the_floor(self):
        summary = summarize_stage1(self._rows())
        assert summary["oracle_regret_floor_by_preset"]["TINY"] == 0.02
        control = summary["arms"][f"{CONTROL_ARM}|TINY"]
        assert control["oracle_q_regret_excess_over_floor"] == pytest.approx(0.01)

    def test_filter_margin(self):
        summary = summarize_stage1(self._rows())
        verdict = apply_stage1_filter(summary, budgets=("TINY",), margin=0.010)
        # varied excess 0.015, control 0.010 -> delta 0.005 <= 0.010: survives
        assert verdict["verdicts"][arm_name(0.3, 0.0)]["survives"] is True
        tight = apply_stage1_filter(summary, budgets=("TINY",), margin=0.004)
        assert tight["verdicts"][arm_name(0.3, 0.0)]["survives"] is False
        assert tight["verdicts"][CONTROL_ARM]["survives"] is True
        assert CONTROL_ARM in tight["survivors"]


def _stage2(arm, preset, ewr, games=60):
    return {
        "arm": f"{arm}|{preset}",
        "games": games,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "ewr": ewr,
        "paired_vs_reference": {"delta": 0.0, "standard_error": 0.01},
    }


def _stage1_summary(entries):
    return {"arms": entries}


class TestSelection:
    def _stage2_report(self, ewrs):
        report = {f"{CONTROL_ARM}|MEDIUM": _stage2(CONTROL_ARM, "MEDIUM", ewrs[CONTROL_ARM])}
        for arm, ewr in ewrs.items():
            if arm != CONTROL_ARM:
                report[f"{arm}|MEDIUM"] = _stage2(arm, "MEDIUM", ewr)
        return report

    def _stage1(self, repeat_rates):
        entries = {}
        for arm, rate in repeat_rates.items():
            tau, tau_r = (0.0, 0.0) if arm == CONTROL_ARM else parse_arm_name(arm)
            entries[f"{arm}|MEDIUM"] = {
                "arm": arm,
                "repeat_rate": rate,
                "tau": tau,
                "tau_r": tau_r,
            }
        return _stage1_summary(entries)

    def test_lowest_repeat_rate_wins_within_margin(self):
        a15 = arm_name(0.15, 0.0)
        a30 = arm_name(0.3, 0.0)
        report = self._stage2_report({CONTROL_ARM: 0.90, a15: 0.88, a30: 0.87})
        stage1 = self._stage1({CONTROL_ARM: 1.0, a15: 0.8, a30: 0.6})
        result = select_configuration(report, stage1, margin=0.05)
        assert result["selected_arm"] == a30
        assert result["stochastic_mode_viable"] is True
        assert result["selected_tau"] == 0.3

    def test_arm_outside_margin_cannot_win(self):
        a15 = arm_name(0.15, 0.0)
        a60 = arm_name(0.6, 0.0)
        report = self._stage2_report({CONTROL_ARM: 0.90, a15: 0.88, a60: 0.80})
        stage1 = self._stage1({CONTROL_ARM: 1.0, a15: 0.8, a60: 0.3})
        result = select_configuration(report, stage1, margin=0.05)
        assert result["selected_arm"] == a15  # a60 is varied but too weak

    def test_named_fallback_when_no_qualifier_has_stage1_numbers(self):
        a15 = arm_name(0.15, 0.0)
        report = self._stage2_report({CONTROL_ARM: 0.90, a15: 0.88})
        stage1 = _stage1_summary({})  # no repeat rates at all
        result = select_configuration(report, stage1, margin=0.05)
        assert result["selected_arm"] == a15
        assert "fallback" in result["reason"]

    def test_no_viable_mode_keeps_argmax(self):
        a15 = arm_name(0.15, 0.0)
        report = self._stage2_report({CONTROL_ARM: 0.90, a15: 0.80})
        stage1 = self._stage1({CONTROL_ARM: 1.0, a15: 0.8})
        result = select_configuration(report, stage1, margin=0.05)
        assert result["selected_arm"] == CONTROL_ARM
        assert result["stochastic_mode_viable"] is False
        assert result["selected_tau"] == 0.0

    def test_better_than_control_qualifies(self):
        a15 = arm_name(0.15, 0.0)
        report = self._stage2_report({CONTROL_ARM: 0.85, a15: 0.90})
        stage1 = self._stage1({CONTROL_ARM: 1.0, a15: 0.8})
        result = select_configuration(report, stage1, margin=0.05)
        assert result["selected_arm"] == a15


class TestProbeAnalysis:
    def _entries(self, arm, scores_by_index):
        from stratego.search.phase16.contract import PROBE_ORDINAL_BASE

        entries = []
        for index, scores in scores_by_index.items():
            for score in scores:
                entries.append(
                    {
                        "row": {
                            "arm_id": arm,
                            "ordinal": PROBE_ORDINAL_BASE + index,
                            "effective_score": score,
                        }
                    }
                )
        return entries

    def test_trend_and_halves(self):
        entries = self._entries(
            CONTROL_ARM, {0: [1.0, 1.0], 1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.0, 0.0]}
        )
        report = analyse_probe(entries)
        entry = report["arms"][CONTROL_ARM]
        assert entry["games"] == 8
        assert entry["ewr"] == 0.5
        assert entry["ewr_slope_per_game_index"] < 0
        assert entry["halves"]["first_half_ewr"] == 0.75
        assert entry["halves"]["second_half_ewr"] == 0.25
        assert "cannot adapt" in report["note"]


class TestCaps:
    def test_caps_kept_when_latency_unchanged(self):
        result = decide_time_caps(
            {"TINY": {"p95_seconds_per_move": 0.26}, "MEDIUM": {"p95_seconds_per_move": 1.80}},
            {"TINY": 0.91, "MEDIUM": 5.0},
            {"TINY": {"p95": 0.259}, "MEDIUM": {"p95": 1.782}},
        )
        assert result["caps_seconds"] == {"TINY": 0.91, "MEDIUM": 5.0}
        assert result["changed"] is False

    def test_caps_rederived_when_latency_grew(self):
        result = decide_time_caps(
            {"TINY": {"p95_seconds_per_move": 0.40}, "MEDIUM": {"p95_seconds_per_move": 1.80}},
            {"TINY": 0.91, "MEDIUM": 5.0},
            {"TINY": {"p95": 0.259}, "MEDIUM": {"p95": 1.782}},
        )
        assert result["changed"] is True
        assert result["caps_seconds"]["TINY"] == pytest.approx(1.4, abs=0.01)
        assert result["caps_seconds"]["MEDIUM"] == 5.0

    def test_ceiling_binds(self):
        result = decide_time_caps(
            {"MEDIUM": {"p95_seconds_per_move": 3.0}},
            {"MEDIUM": 5.0},
            {"MEDIUM": {"p95": 1.782}},
        )
        assert result["caps_seconds"]["MEDIUM"] == 5.0


class TestPackConstruction:
    def test_position_cells_are_fresh_and_balanced(self):
        cells = position_cells_16(30)
        ordinals = {cell[4] for cell in cells}
        assert min(ordinals) == 200 and len(ordinals) == 30
        opponents = [cell[1] for cell in cells]
        assert all(opponents.count(name) == 3 for name in set(opponents))
        observers = {cell[0] for cell in cells}
        assert observers == {"p24"}

    def test_interim_ordinal_collides_with_nothing(self):
        # Phase 15 Stage B: 0-1; Stage C / deep pack: 0; Phase 15 positions:
        # 100-114; Phase 16 positions: 200+; probe: 300+.
        assert INTERIM_PACK_ORDINAL == 2

    def test_interim_board_ids_parse_and_cover_the_cells(self):
        from stratego.search.phase15.boards import BOARD_CELLS
        from stratego.search.phase15.contract import parse_board_id
        from stratego.search.phase16.contract import INTERIM_PACK_ORDINAL as ORD
        from stratego.search.phase15.contract import board_id
        from stratego.search.phase16.matchpack import interim_pack_plans  # noqa: F401

        # Identity-level check without drawing setups (drawing is exercised in
        # the runner): every cell yields a well-formed fresh board id.
        from stratego.search.phase15.boards import requested_family

        seen = set()
        for index, (opponent, source, color) in enumerate(BOARD_CELLS):
            family = requested_family(source, index, ORD)
            identifier = board_id(opponent, source, family, color, ORD)
            fields = parse_board_id(identifier)
            assert fields["ordinal"] == ORD
            seen.add(identifier)
        assert len(seen) == 60
