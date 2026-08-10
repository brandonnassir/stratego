"""Phase 4 Agent 4 acceptance tests.

Four things this module pins that nothing else does.

**The exposure recalibration.** Agent 4 changed exactly one rule in the baseline
suite: `strategic_rule_based`'s exposure term is now priced by how badly an
identified piece can be answered rather than by what it is worth. The old rule
was anti-correlated with vulnerability and made Strategic measurably *weaker*
than Tactical. These tests assert the corrected direction on crafted positions,
so a future edit that reverts to material pricing fails here rather than being
found again by a 15-minute league.

**The tier partition.** `strength_tiers` turns a table of paired confidence
intervals into the Phase 4 gate's answer. Its edge cases -- ties collapsing into
one tier, a missing comparison, a cross-tier pair that does not separate -- are
tested against hand-built tables with known answers, because a partition function
that silently over-counts tiers would make the gate meaningless.

**Audit determinism.** The hidden-information audit's findings must not depend on
how many cores were free. The audit is therefore cut into a fixed number of
chunks and this module proves a serial run and a pooled run produce byte-identical
reports.

**A fast permutation regression.** The full audit is 100,000 trials and belongs in
`scripts/run_phase4_agent04.py`, not in a per-commit suite. A few hundred trials
here keep the property under continuous test at a cost the suite can afford.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.engine.constants import (
    BLUE,
    BOMB,
    FLAG,
    MARSHAL,
    MINER,
    RED,
    SCOUT,
    SPY,
)
from stratego.engine.state import state_fingerprint
from stratego.evaluation.baselines import STRATEGIC_WEIGHTS
from stratego.evaluation.calibration import (
    AUDIT_PLIES,
    CALIBRATION_VERSION,
    POSITION_SOURCES,
    audit_chunk,
    audit_payloads,
    behavior_divergence,
    entropy_bits,
    merge_audit_results,
    profile_replay,
    run_hidden_information_audit,
    sample_positions,
    strength_tiers,
    summarise_behavior,
)
from stratego.evaluation.heuristics import PIECE_VALUES, build_context
from stratego.evaluation.match_runner import run_schedule
from stratego.evaluation.policy import build_policy_input
from stratego.evaluation.registry import (
    ALL_POLICY_IDS,
    LADDER_POLICY_IDS,
    build_policy,
    policy_ref,
)
from stratego.evaluation.scheduler import build_matchup_schedule
from stratego.evaluation.setup_bank import SetupBank

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports/phase_4_data"


@pytest.fixture(scope="module")
def small_bank() -> SetupBank:
    return SetupBank.generate(size=8)


def request_for(state, policy, seed: int = 4242):
    return build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=seed,
        requirements=policy.requirements,
        suite_version="test",
        match_id="test",
        paired_unit_id="test",
    )


# ---------------------------------------------------------------------------
# The exposure recalibration
# ---------------------------------------------------------------------------


def test_strategic_is_at_the_recalibrated_version():
    """The rule change is versioned, because a match identity names the version."""
    assert policy_ref("strategic_rule_based").policy_version == "1.1.0"
    assert STRATEGIC_WEIGHTS.exposure > 0.0


def _exposure_component(policy, context, move) -> float:
    return sum(value for name, value in policy.score(context, move).components if name == "exposure")


def test_exposure_penalises_the_answerable_piece_not_the_valuable_one():
    """A revealed Spy advancing must be restrained harder than a revealed Marshal.

    This is the whole content of the recalibration. Under the old rule the
    ordering was strictly reversed, because the Marshal's material value is four
    times the Spy's while the Marshal is the piece the opponent can least answer.
    """
    policy = build_policy("strategic_rule_based")
    positions = sample_positions(5, source="random_walk")
    context = build_context(request_for(positions[len(positions) // 2], policy))

    spy_cost = context.expected_defence_value(SPY)
    marshal_cost = context.expected_defence_value(MARSHAL)
    miner_cost = context.expected_defence_value(MINER)

    # The public deduction the term now rests on.
    assert spy_cost < 0.0, "the Spy must be answerable while high ranks remain"
    assert miner_cost < 0.0
    assert marshal_cost > spy_cost
    # And the ordering the old rule got backwards.
    assert PIECE_VALUES[MARSHAL] > PIECE_VALUES[SPY]
    assert abs(STRATEGIC_WEIGHTS.exposure * spy_cost) > abs(
        STRATEGIC_WEIGHTS.exposure * min(marshal_cost, 0.0)
    )


def test_exposure_is_silent_for_a_piece_nothing_answers():
    """No penalty at all once the term's premise fails.

    A piece the unresolved inventory cannot beat is not made a target by being
    identified, so the component must be absent rather than merely small.
    """
    policy = build_policy("strategic_rule_based")
    positions = sample_positions(9, source="random_walk")
    context = build_context(request_for(positions[0], policy))
    assert context.expected_defence_value(MARSHAL) > 0.0

    for move in context.moves:
        if move.piece_type == MARSHAL and move.advance > 0 and context.is_exposed(move.piece_id):
            assert _exposure_component(policy, context, move) == 0.0


def test_exposure_only_fires_on_an_identified_piece_that_advances():
    """Both preconditions are load-bearing: identified, and moving forward."""
    policy = build_policy("strategic_rule_based")
    for seed in (3, 5, 8, 13):
        for state in sample_positions(seed, source="baseline_play"):
            context = build_context(request_for(state, policy))
            for move in context.moves:
                component = _exposure_component(policy, context, move)
                if component == 0.0:
                    continue
                assert context.is_exposed(move.piece_id)
                assert move.advance > 0
                assert context.expected_defence_value(move.piece_type) < 0.0
                assert component < 0.0


def test_the_exposure_term_still_reaches_a_real_score_component():
    """A recalibrated term that never fires would be a silent removal."""
    policy = build_policy("strategic_rule_based")
    fired = 0
    for seed in range(20):
        for state in sample_positions(seed, source="baseline_play"):
            context = build_context(request_for(state, policy))
            fired += sum(
                1 for move in context.moves if _exposure_component(policy, context, move) != 0.0
            )
        if fired:
            break
    assert fired > 0, "the recalibrated exposure term never produced a component"


def test_exposure_uses_only_permutation_invariant_inputs():
    """`expected_defence_value` reads the unresolved inventory, which is public."""
    from stratego.engine.permutation import permute_hidden_identities
    import random

    policy = build_policy("strategic_rule_based")
    rng = random.Random(77)
    compared = 0
    for seed in range(6):
        for state in sample_positions(seed, source="random_walk"):
            clone, info = permute_hidden_identities(state, state.acting_player, rng)
            if not info["valid"] or not info["changed"]:
                continue
            left = build_context(request_for(state, policy))
            right = build_context(request_for(clone, policy))
            for piece_type in (SPY, SCOUT, MINER, MARSHAL, FLAG, BOMB):
                assert left.expected_defence_value(piece_type) == right.expected_defence_value(
                    piece_type
                )
            compared += 1
    assert compared > 0


# ---------------------------------------------------------------------------
# Strength tiers
# ---------------------------------------------------------------------------


def interval(lower: float, upper: float) -> dict:
    return {"lower": lower, "upper": upper, "width": upper - lower}


def matchup(candidate: str, opponent: str, rate: float, lower: float, upper: float) -> dict:
    return {
        "candidate": candidate,
        "opponent": opponent,
        "effective_win_rate": rate,
        "confidence_interval": interval(lower, upper),
    }


def test_a_total_order_gives_one_tier_per_policy():
    summaries = {
        "a vs b": matchup("a", "b", 0.80, 0.76, 0.84),
        "a vs c": matchup("a", "c", 0.95, 0.92, 0.98),
        "b vs c": matchup("b", "c", 0.70, 0.66, 0.74),
    }
    tiers = strength_tiers(["a", "b", "c"], summaries)
    assert tiers["tier_count"] == 3
    assert [tier["members"] for tier in tiers["tiers"]] == [["a"], ["b"], ["c"]]
    assert tiers["fully_ordered"] is True
    assert tiers["unseparated_cross_tier_pairs"] == []
    assert tiers["calibration_version"] == CALIBRATION_VERSION


def test_two_indistinguishable_policies_share_a_tier():
    """The gate counts levels the evidence supports, not policies."""
    summaries = {
        "a vs b": matchup("a", "b", 0.52, 0.47, 0.57),
        "a vs c": matchup("a", "c", 0.90, 0.86, 0.94),
        "b vs c": matchup("b", "c", 0.88, 0.84, 0.92),
    }
    tiers = strength_tiers(["a", "b", "c"], summaries)
    assert tiers["tier_count"] == 2
    assert tiers["membership"] == {"a": 1, "b": 1, "c": 2}
    assert tiers["fully_ordered"] is True


def test_a_reversed_matchup_orientation_is_recovered():
    """A matchup is stored once; the other orientation must still be readable."""
    forward = strength_tiers(
        ["a", "b"], {"a vs b": matchup("a", "b", 0.80, 0.75, 0.85)}
    )
    backward = strength_tiers(
        ["a", "b"], {"b vs a": matchup("b", "a", 0.20, 0.15, 0.25)}
    )
    assert forward["tier_count"] == backward["tier_count"] == 2
    assert forward["membership"] == backward["membership"] == {"a": 1, "b": 2}
    assert forward["pooled_effective_win_rates"]["a"] == pytest.approx(
        backward["pooled_effective_win_rates"]["a"]
    )


def test_a_missing_comparison_is_reported_not_assumed():
    tiers = strength_tiers(
        ["a", "b", "c"],
        {
            "a vs b": matchup("a", "b", 0.80, 0.75, 0.85),
            "a vs c": matchup("a", "c", 0.90, 0.86, 0.94),
        },
    )
    assert tiers["missing_comparisons"] == ["b vs c"]


def test_a_non_transitive_result_is_not_reported_as_a_clean_ladder():
    """Three tiers by adjacency, but the top does not beat the bottom.

    `fully_ordered` exists precisely so a rock-paper-scissors league cannot be
    reported as a ladder. The tier count can still be three; the claim that the
    tiers are ordered must fail.
    """
    summaries = {
        "a vs b": matchup("a", "b", 0.80, 0.75, 0.85),
        "b vs c": matchup("b", "c", 0.80, 0.75, 0.85),
        "a vs c": matchup("a", "c", 0.50, 0.45, 0.55),
    }
    tiers = strength_tiers(["a", "b", "c"], summaries)
    assert tiers["tier_count"] == 3
    assert tiers["fully_ordered"] is False
    assert "a vs c" in tiers["unseparated_cross_tier_pairs"]


def test_a_perfect_cycle_has_no_tiers_at_all():
    """A tier must separate *above* the one below, not merely separate from it.

    Rock-paper-scissors: `y` beats `x`, `z` beats `y`, `x` beats `z`, every pair
    decisively and every pooled rate identical. No strength ordering exists, so
    the honest answer is one tier. A partition that asked only "is this pair
    separated?" would answer three tiers and call them fully ordered, which is
    the failure mode this guards -- and the reason the Phase 4 gate reads the
    tier count rather than a rating table.
    """
    summaries = {
        "x vs y": matchup("x", "y", 0.10, 0.06, 0.14),  # y beats x
        "y vs z": matchup("y", "z", 0.10, 0.06, 0.14),  # z beats y
        "x vs z": matchup("x", "z", 0.90, 0.86, 0.94),  # x beats z
    }
    tiers = strength_tiers(["x", "y", "z"], summaries)
    assert tiers["pooled_effective_win_rates"] == {"x": 0.5, "y": 0.5, "z": 0.5}
    assert tiers["tier_count"] == 1
    assert tiers["tiers"][0]["members"] == ["x", "y", "z"]
    # One tier has no cross-tier pair, so ordering is trivially satisfied; the
    # content of the finding is the tier count.
    assert tiers["cross_tier_pairs"] == 0


def test_the_partition_is_deterministic_under_input_order():
    summaries = {
        "a vs b": matchup("a", "b", 0.80, 0.75, 0.85),
        "a vs c": matchup("a", "c", 0.90, 0.86, 0.94),
        "b vs c": matchup("b", "c", 0.70, 0.66, 0.74),
    }
    first = strength_tiers(["a", "b", "c"], summaries)
    second = strength_tiers(["c", "a", "b"], summaries)
    assert first["tiers"] == second["tiers"]
    assert first["membership"] == second["membership"]


# ---------------------------------------------------------------------------
# Audit determinism
# ---------------------------------------------------------------------------


def test_audit_chunks_sum_to_the_requested_trials():
    payloads = audit_payloads(500, chunks=16)
    assert sum(payload["trials"] for payload in payloads) == 500
    assert len({payload["chunk_index"] for payload in payloads}) == len(payloads)
    assert {payload["source"] for payload in payloads} == set(POSITION_SOURCES)


def test_audit_decomposition_does_not_depend_on_the_worker_count():
    """The chunking is an explicit parameter, not a function of the machine."""
    assert audit_payloads(300, chunks=8) == audit_payloads(300, chunks=8)


def test_the_same_chunk_payload_reproduces_its_totals():
    payload = {
        "root_seed": 99,
        "chunk_index": 3,
        "trials": 20,
        "source": "random_walk",
        "policy_ids": list(ALL_POLICY_IDS),
        "plies": list(AUDIT_PLIES),
    }
    first = audit_chunk(dict(payload))
    second = audit_chunk(dict(payload))
    assert first == second


def test_a_pooled_audit_matches_a_serial_one_exactly():
    """The evidence must be identical whether or not a pool was used."""
    serial = run_hidden_information_audit(120, workers=1, chunks=4, root_seed=17)
    pooled = run_hidden_information_audit(120, workers=4, chunks=4, root_seed=17)
    for key in (
        "trials",
        "policy_comparisons",
        "score_vector_comparisons",
        "total_mismatches",
        "positive_control_trials",
        "positive_control_failures",
        "leak_detector_failures",
        "positions_skipped_unchanged",
        "games_sampled",
        "trials_by_ply",
        "trials_by_policy",
        "trials_by_source",
    ):
        assert serial[key] == pooled[key], f"{key} depends on the worker count"


def test_a_short_audit_finds_no_leak_and_is_not_vacuous():
    """The per-commit floor under the 100,000-trial acceptance audit."""
    audit = run_hidden_information_audit(200, workers=1, chunks=4, root_seed=23)
    assert audit["trials"] == 200
    assert audit["total_mismatches"] == 0
    assert audit["action_mismatches"] == 0
    assert audit["diagnostic_mismatches"] == 0
    assert audit["score_vector_mismatches"] == 0
    assert audit["public_view_mismatches"] == 0
    assert audit["legal_action_mismatches"] == 0
    # Not vacuous: every trial really did permute something privileged.
    assert audit["positive_control_trials"] == audit["trials"]
    assert audit["positive_control_failures"] == 0
    assert audit["leak_detector_failures"] == 0
    assert len(audit["trials_by_policy"]) == len(ALL_POLICY_IDS)
    assert len(audit["trials_by_ply"]) >= 4
    assert audit["policy_comparisons"] == audit["trials"] * len(ALL_POLICY_IDS)


def test_merging_audit_chunks_adds_rather_than_overwrites():
    left = audit_chunk(
        {
            "root_seed": 5,
            "chunk_index": 0,
            "trials": 10,
            "source": "random_walk",
            "policy_ids": list(ALL_POLICY_IDS),
            "plies": list(AUDIT_PLIES),
        }
    )
    right = audit_chunk(
        {
            "root_seed": 5,
            "chunk_index": 1,
            "trials": 10,
            "source": "baseline_play",
            "policy_ids": list(ALL_POLICY_IDS),
            "plies": list(AUDIT_PLIES),
        }
    )
    merged = merge_audit_results([left, right])
    assert merged["trials"] == left["trials"] + right["trials"]
    assert merged["policy_comparisons"] == left["policy_comparisons"] + right["policy_comparisons"]
    assert set(merged["trials_by_source"]) == {"random_walk", "baseline_play"}
    assert merged["total_mismatches"] == 0


# ---------------------------------------------------------------------------
# Position sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", POSITION_SOURCES)
def test_sampled_positions_are_deterministic_and_nonterminal(source: str):
    first = sample_positions(31, source=source)
    second = sample_positions(31, source=source)
    assert [state.total_moves for state in first] == [state.total_moves for state in second]
    assert [state_fingerprint(state) for state in first] == [
        state_fingerprint(state) for state in second
    ]
    assert first
    for state in first:
        assert not state.terminal
        assert state.total_moves in AUDIT_PLIES


def test_random_walk_and_baseline_play_produce_different_positions():
    """Two sources, or the audit is only testing one kind of position."""
    walk = sample_positions(41, source="random_walk")
    play = sample_positions(41, source="baseline_play")
    assert [state_fingerprint(state) for state in walk] != [
        state_fingerprint(state) for state in play
    ]


def test_an_unknown_position_source_is_rejected():
    with pytest.raises(ValueError, match="unknown position source"):
        sample_positions(1, source="wishful_thinking")


# ---------------------------------------------------------------------------
# Behaviour profiling
# ---------------------------------------------------------------------------


def test_profiling_by_replay_agrees_with_the_row_it_replays(small_bank):
    """The profile describes the league's own game, not a re-simulation."""
    schedule = build_matchup_schedule("tactical_rule_based", "stress_scout_rush", list(range(2)))
    run = run_schedule(schedule.matches, small_bank, worker_count=1, record_actions=True)
    for row in run.results:
        counters = profile_replay(row)
        assert set(counters) == {row.candidate.token, row.opponent.token}
        total_moves = sum(counter["moves"] for counter in counters.values())
        assert total_moves == row.plies
        for counter in counters.values():
            assert counter["plies"] == row.plies
            assert counter["games"] == 1
            assert counter["wins"] + counter["draws"] + counter["losses"] == 1


def test_profiling_refuses_a_row_with_no_history(small_bank):
    schedule = build_matchup_schedule("basic_heuristic", "random_legal", list(range(1)))
    run = run_schedule(schedule.matches, small_bank, worker_count=1, record_actions=False)
    with pytest.raises(ValueError, match="no stored action history"):
        profile_replay(run.results[0])


def test_the_scout_rush_still_separates_from_the_ladder_by_replay(small_bank):
    """A behavioural claim the stress characterisation rests on."""
    schedule = build_matchup_schedule("stress_scout_rush", "strategic_rule_based", list(range(4)))
    run = run_schedule(schedule.matches, small_bank, worker_count=1, record_actions=True)
    from collections import Counter

    pooled: dict[str, Counter] = {}
    for row in run.results:
        for token, counter in profile_replay(row).items():
            pooled.setdefault(token, Counter()).update(counter)
    rush = summarise_behavior(pooled[policy_ref("stress_scout_rush").token])
    ladder = summarise_behavior(pooled[policy_ref("strategic_rule_based").token])
    assert rush["scout_move_rate"] > 2 * ladder["scout_move_rate"]
    assert behavior_divergence(rush, ladder)["materially_different"] is True


def test_behaviour_summary_rates_are_well_formed():
    counter = {
        "games": 2,
        "moves": 10,
        "plies": 20,
        "attacks": 3,
        "piece_scout": 4,
        "piece_miner": 1,
        "scout_runs": 2,
        "miner_attacks": 1,
        "distance": 14,
        "wins": 1,
        "draws": 1,
        "losses": 0,
        "own_pieces_revealed": 8,
        "direction_forward_step": 6,
        "direction_lateral_step": 4,
    }
    summary = summarise_behavior(counter)
    assert summary["attack_rate"] == pytest.approx(0.3)
    assert summary["scout_move_rate"] == pytest.approx(0.4)
    assert summary["mean_game_plies"] == pytest.approx(10.0)
    assert summary["effective_win_rate"] == pytest.approx(0.75)
    assert summary["own_reveal_rate"] == pytest.approx(8 / (2 * 40))
    assert summary["movement_entropy_bits"] == pytest.approx(entropy_bits([6, 4]))
    assert sum(summary["direction_shares"].values()) == pytest.approx(1.0)


def test_an_identical_profile_is_not_materially_different():
    profile = summarise_behavior({"games": 1, "moves": 4, "plies": 4, "attacks": 1})
    assert behavior_divergence(profile, profile)["materially_different"] is False
    assert behavior_divergence(profile, profile)["largest_relative_difference"] == 0.0


def test_a_zero_versus_positive_metric_reads_as_a_full_separation():
    """The symmetric denominator must not divide by zero on a refusing policy."""
    quiet = summarise_behavior({"games": 1, "moves": 10, "plies": 10, "attacks": 0})
    busy = summarise_behavior({"games": 1, "moves": 10, "plies": 10, "attacks": 5})
    divergence = behavior_divergence(quiet, busy)
    assert divergence["relative_differences"]["attack_rate"] == pytest.approx(-1.0)
    assert divergence["materially_different"] is True


def test_entropy_of_a_degenerate_distribution_is_zero():
    assert entropy_bits([]) == 0.0
    assert entropy_bits([0, 0]) == 0.0
    assert entropy_bits([7]) == 0.0
    assert entropy_bits([1, 1]) == pytest.approx(1.0)
    assert entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)


def test_direction_buckets_are_own_relative(small_bank):
    """Forward must mean forward for both colours, or pooling cancels it out."""
    from stratego.evaluation.calibration import _direction_bucket
    from stratego.engine.coordinates import square_index

    forward_red = _direction_bucket(square_index(1, 0), square_index(2, 0), RED)
    forward_blue = _direction_bucket(square_index(8, 0), square_index(7, 0), BLUE)
    assert forward_red == forward_blue == "forward_step"
    assert _direction_bucket(square_index(2, 0), square_index(1, 0), RED) == "backward_step"
    assert _direction_bucket(square_index(2, 0), square_index(2, 3), RED) == "lateral_run"


# ---------------------------------------------------------------------------
# The published acceptance artefacts
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DATA_DIRECTORY / "agent_04_calibration_security.json").exists(),
    reason="the Agent 4 acceptance harness has not been run in this checkout",
)
def test_the_published_acceptance_record_is_internally_consistent():
    """Guards the artefact the Phase 4 decision is read from.

    Deliberately checks consistency rather than re-deriving the numbers: the run
    that produced them takes tens of minutes and belongs in the harness. What must
    not happen is a data file whose headline disagrees with its own gates.
    """
    payload = json.loads(
        (DATA_DIRECTORY / "agent_04_calibration_security.json").read_text(encoding="utf-8")
    )
    required = {
        "agent",
        "status",
        "hidden_information_trials",
        "hidden_information_mismatches",
        "positive_control_trials",
        "positive_control_failures",
        "policies_audited",
        "league_match_count",
        "league_paired_unit_count",
        "core_policy_results",
        "pairwise_effective_win_rates",
        "pairwise_confidence_intervals",
        "strength_tier_count",
        "strength_tier_membership",
        "stress_behavior_metrics",
        "reproducibility_rerun_matches",
        "reproducibility_mismatches",
        "policy_tuning_iterations",
        "phase4_decision",
        "test_total",
        "test_passed",
        "test_failed",
        "files_created",
        "files_modified",
    }
    assert required <= set(payload), sorted(required - set(payload))
    assert payload["agent"] == "agent_04"
    assert payload["status"] == payload["phase4_decision"]
    if payload["quick"]:
        pytest.skip("a --quick run reports FAIL by design")

    assert payload["status"] == "PASS"
    assert all(payload["completion_gates"].values())
    assert payload["hidden_information_trials"] >= 100_000
    assert payload["hidden_information_mismatches"] == 0
    assert payload["positive_control_failures"] == 0
    assert payload["reproducibility_mismatches"] == 0
    assert payload["strength_tier_count"] >= 3
    assert len(payload["policies_audited"]) == len(ALL_POLICY_IDS)
    assert set(payload["strength_tier_membership"]) == {
        policy_ref(pid).token for pid in LADDER_POLICY_IDS
    }
    # The floor of the ladder is the highest tier number.
    assert payload["strength_tier_membership"][policy_ref("random_legal").token] == max(
        payload["strength_tier_membership"].values()
    )
    for matchup, ci in payload["pairwise_confidence_intervals"].items():
        assert ci["resampling_unit"] == "paired_unit", matchup
        assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0, matchup
