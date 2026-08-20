"""Phase 13 Agent 1 artifact tests.

Mechanical checks of the frozen Phase 13/14 contract chain:

- the setup-census alarm policy existed before sampling and the census binds
  its exact bytes;
- the census and pack content digests recompute from their own payloads;
- the 128-game selection pack has the frozen shape and every board passes the
  engine's own setup validation;
- the contract's arithmetic (LRs, mixtures, schedule) is exact;
- the frozen active-pool membership function is implementable, deterministic
  and well-formed for every archive size a 168-hour run can produce.

Torch-free on purpose: these are artifact tests, not model tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase13"

POLICY_PATH = REPORT_DIR / "phase13_setup_census_alarm_policy_v1.json"
CENSUS_PATH = REPORT_DIR / "phase13_setup_census_v1.json"
CONTRACT_PATH = REPORT_DIR / "phase13_final_training_contract_v1.json"
SETUP_SOURCE_PATH = REPORT_DIR / "phase14_setup_source_v1.json"
PACK_PATH = REPORT_DIR / "phase14_checkpoint_selection_pack_v1.json"
RULE_PATH = REPORT_DIR / "phase14_checkpoint_selection_rule_v1.json"

pytestmark = pytest.mark.skipif(
    not CENSUS_PATH.exists(), reason="phase13 agent 1 artifacts not present"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Census chain
# ---------------------------------------------------------------------------


def test_alarm_policy_precedes_and_binds_census():
    policy = _load(POLICY_PATH)
    census = _load(CENSUS_PATH)
    assert census["alarm_policy_sha256"] == hashlib.sha256(
        POLICY_PATH.read_bytes()
    ).hexdigest()
    assert census["alarm_policy_written_utc"] == policy["written_utc"]
    assert census["alarm_policy_written_utc"] <= census["written_utc"]
    thresholds = policy["classification"]["pathology"]["thresholds"]
    assert census["classification"]["P_trivial_threshold"] == thresholds[
        "P_trivial_pathological_at_or_above"
    ]
    assert census["classification"]["P_predecision_threshold"] == thresholds[
        "P_predecision_pathological_at_or_above"
    ]


def test_census_content_digest_recomputes():
    census = _load(CENSUS_PATH)
    payload = {
        k: v
        for k, v in census.items()
        if k not in ("written_utc", "elapsed_seconds", "census_content_digest")
    }
    assert census["census_content_digest"] == _canonical_digest(payload)


def test_census_classification_is_mechanical():
    census = _load(CENSUS_PATH)
    paired = census["paired_measurements"]
    defects = sum(census["defect_counts"].values())
    p_trivial = paired["either_immediate"] / paired["boards"]
    p_predecision = paired["predecision_engine"] / paired["boards"]
    assert paired["P_trivial"] == round(p_trivial, 6)
    assert paired["P_predecision"] == round(p_predecision, 6)
    result = census["classification"]["result"]
    if defects:
        assert result == "DEFECT"
    elif (
        p_trivial >= census["classification"]["P_trivial_threshold"]
        or p_predecision >= census["classification"]["P_predecision_threshold"]
    ):
        assert result == "PATHOLOGY"
    else:
        assert result in ("VALID_BUT_STRATEGICALLY_POOR", "NO_FINDING")


def test_census_sample_sizes_meet_policy_minima():
    census = _load(CENSUS_PATH)
    sizes = census["sample_sizes"]
    assert sizes["production_mixture_draws"] >= 10000
    assert sizes["neutral_v1_draws"] >= 10000
    assert sizes["learned_branch_draws"] >= 10000
    assert sizes["paired_boards"] >= 8192
    assert (
        sizes["learned_branch_draws"] + sizes["neutral_branch_draws"]
        == sizes["production_mixture_draws"]
    )


def test_census_stage_invariants_hold():
    census = _load(CENSUS_PATH)
    effects = census["stage_effects"]
    assert effects["front_row_changes_after_reflection"] == 0
    assert effects["front_row_changes_after_perturbation"] == 0
    attribution = census["exposure_attribution"]
    families = set(attribution["forwardmost_flag_draws_by_family"])
    assert families <= {"F15"}


def test_setup_source_binds_census():
    source = _load(SETUP_SOURCE_PATH)
    census = _load(CENSUS_PATH)
    assert (
        source["census_binding"]["census_content_digest"]
        == census["census_content_digest"]
    )
    assert (
        source["census_binding"]["alarm_policy_sha256"] == census["alarm_policy_sha256"]
    )
    assert source["definition"]["mixture"] == {"neutral_v1": 0.35, "p10d_learned": 0.65}


# ---------------------------------------------------------------------------
# Selection pack and rule
# ---------------------------------------------------------------------------


def test_pack_shape_and_digest():
    pack = _load(PACK_PATH)
    games = pack["games"]
    assert len(games) == 128
    payload = {k: v for k, v in pack.items() if k not in ("written_utc", "pack_content_digest")}
    assert pack["pack_content_digest"] == _canonical_digest(payload)
    by_opponent: dict = {}
    for game in games:
        by_opponent.setdefault(game["opponent"], []).append(game["candidate_color"])
    assert set(by_opponent) == {
        "phase9_anchor",
        "strategic_rule_based",
        "tactical_rule_based",
        "stress_scout_rush",
    }
    for colors in by_opponent.values():
        assert len(colors) == 32
        assert colors.count("red") == 16 and colors.count("blue") == 16
    ordinals = [game["setup_ordinal"] for game in games]
    assert len(set(ordinals)) == 128
    assert min(ordinals) >= 1_000_000  # disjoint from census ordinals 0..8191


def test_pack_boards_pass_engine_validation():
    from stratego.engine.constants import BLUE, FLAG, RED
    from stratego.engine.setup import validate_setup

    pack = _load(PACK_PATH)
    for game in pack["games"]:
        red = tuple(game["red"]["oriented_engine_setup"])
        blue = tuple(game["blue"]["oriented_engine_setup"])
        validate_setup(red, RED)
        validate_setup(blue, BLUE)
        assert red.count(FLAG) == 1 and blue.count(FLAG) == 1


def test_selection_rule_binds_pack():
    rule = _load(RULE_PATH)
    pack = _load(PACK_PATH)
    assert rule["pack_binding"]["pack_content_digest"] == pack["pack_content_digest"]
    assert rule["pack_binding"]["games_per_candidate"] == len(pack["games"])
    assert rule["tie_break_2"].startswith("if still exactly tied, select the later")


# ---------------------------------------------------------------------------
# Contract arithmetic
# ---------------------------------------------------------------------------


def test_contract_learning_rates_exact():
    contract = _load(CONTRACT_PATH)
    lrs = contract["continuation_learning_rate"]
    assert lrs["LR9"] == 0.0003
    assert lrs["main_continuation_LR"] == 0.25 * 0.0003
    assert lrs["late_continuation_LR"] == 0.125 * 0.0003


def test_contract_mixture_counts_exact():
    contract = _load(CONTRACT_PATH)
    mixture = contract["opponent_mixture"]
    for segment in ("main_segment_counts_per_2048", "late_segment_counts_per_2048"):
        counts = mixture[segment]
        handcrafted = sum(counts["handcrafted"].values())
        total = counts["current_neural"] + counts["historical_neural"] + handcrafted
        assert total == 2048
        neural = (counts["current_neural"] + counts["historical_neural"]) / 2048
        assert 0.85 <= neural <= 0.90
        assert 0.10 <= handcrafted / 2048 <= 0.15
        assert set(counts["handcrafted"]) == {
            "strategic_rule_based",
            "tactical_rule_based",
            "stress_scout_rush",
            "stress_miner_rush",
            "stress_information_miser",
        }


def test_contract_schedule_and_deadline():
    contract = _load(CONTRACT_PATH)
    schedule = contract["main_late_schedule"]
    assert schedule["main_segment_hours"] + schedule["late_segment_hours"] == 168
    assert 0.75 <= schedule["main_segment_hours"] / 168 <= 0.80
    assert "604800" in contract["wall_clock_contract"]["run_deadline_utc"]
    assert contract["phase9_retrieved_values"]["ema_behavior"][
        "present_in_accepted_phase9"
    ] is False


def test_contract_binds_accepted_checkpoint_bytes():
    contract = _load(CONTRACT_PATH)
    start = contract["starting_model"]
    checkpoint = REPO_ROOT / start["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == start["file_sha256"]


# ---------------------------------------------------------------------------
# Active-pool membership function (reference implementation of the frozen text)
# ---------------------------------------------------------------------------


def _membership(k: int) -> dict:
    if k <= 14:
        older = list(range(1, k // 3 + 1))
        middle = list(range(k // 3 + 1, 2 * k // 3 + 1))
        recent = list(range(2 * k // 3 + 1, k + 1))
    else:
        recent = list(range(k - 5, k + 1))
        remaining = list(range(1, k - 5))
        half = math.ceil(len(remaining) / 2)
        older_half, middle_half = remaining[:half], remaining[half:]
        older = [older_half[i * (len(older_half) - 1) // 3] for i in range(4)]
        middle = [middle_half[i * (len(middle_half) - 1) // 3] for i in range(4)]
    return {"older": older, "middle": middle, "recent": recent}


def test_pool_membership_function_well_formed():
    for k in range(0, 90):  # a 168h run archives 84 snapshots
        bands = _membership(k)
        members = bands["older"] + bands["middle"] + bands["recent"]
        assert len(members) == len(set(members)), f"duplicate members at k={k}"
        assert all(1 <= j <= k for j in members)
        if k <= 14:
            assert len(members) == k
        else:
            assert len(bands["older"]) == 4
            assert len(bands["middle"]) == 4
            assert bands["recent"] == list(range(k - 5, k + 1))
            assert max(bands["older"], default=0) < min(bands["middle"])
            assert max(bands["middle"]) < min(bands["recent"])
        assert bands == _membership(k)  # pure function


def test_pool_partition_counts_exact():
    # largest-remainder partition of the historical bucket, as frozen
    def partition(total: int, weights: list) -> list:
        shares = [total * w for w in weights]
        counts = [int(share) for share in shares]
        remainder = total - sum(counts)
        order = sorted(
            range(len(weights)), key=lambda i: (-(shares[i] - counts[i]), i)
        )
        for i in order[:remainder]:
            counts[i] += 1
        return counts

    for total in (615, 984):
        # anchors only (k = 0): whole share to the two anchors
        counts = partition(total, [0.5, 0.5])
        assert sum(counts) == total
        # full pool: 2 anchors at 0.10, older/middle 4 at 0.0625, recent 6 at 0.05
        weights = [0.10, 0.10] + [0.25 / 4] * 4 + [0.25 / 4] * 4 + [0.30 / 6] * 6
        assert abs(sum(weights) - 1.0) < 1e-12
        counts = partition(total, weights)
        assert sum(counts) == total
        assert all(count >= 0 for count in counts)
