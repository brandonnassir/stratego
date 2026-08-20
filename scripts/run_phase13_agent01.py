"""Phase 13 Agent 1: setup census and candidate-selection pack.

Two stages, both deterministic:

``--stage census``
    The one short exposed-Flag setup-distribution census of
    `01_AGENT_1_FINAL_TRAINING_CONTRACT_AND_SETUP_CENSUS.md` section 15.
    Refuses to run unless `phase13_setup_census_alarm_policy_v1.json`
    already exists (the alarm criteria must predate sampling) and embeds
    that file's sha256 in the census artifact.

``--stage pack``
    Builds `phase14_checkpoint_selection_pack_v1.json`, the fixed 128-game
    direct-policy candidate evaluation pack of section 11. Refuses to run
    unless `phase14_setup_source_v1.json` exists (the pack draws its boards
    from the frozen production source).

Every draw goes through the accepted entry points only:
`LearnedSetupSource.draw` (production mixture), `neutral_baseline_draw`
(accepted sampler), `SelectorDraw.oriented`/`SampledSetup.oriented`
(accepted orientation), `create_game` + `legal_actions` (frozen engine).
Seeds come from the frozen Phase 10 `selector_audit_seed` domain; the
census uses ordinals 0..8191 per color and the pack uses ordinals
1_000_000..1_000_127, so the two ranges cannot collide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stratego.engine import create_game, legal_actions  # noqa: E402
from stratego.engine.actions import decode_action  # noqa: E402
from stratego.engine.constants import (  # noqa: E402
    BLUE,
    EVALUATION_RULES,
    FLAG,
    RED,
    SCOUT,
    SETUP_SQUARES,
)
from stratego.setups.families import FAMILY_BY_ID  # noqa: E402
from stratego.setups.identity import reflect_canonical  # noqa: E402
from stratego.setups.sampler import load_library_index  # noqa: E402
from stratego.training.phase10_seed import selector_audit_seed  # noqa: E402
from stratego.training.phase10_selector import (  # noqa: E402
    LearnedSetupSource,
    SelectorRequest,
    candidate,
    load_scorer,
    neutral_baseline_draw,
    neutral_branch_matches_accepted_sampler,
    split_base_entries,
)
from stratego.training.warmstart_contract import CORPUS_RULES  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "phase13"
ALARM_POLICY_PATH = REPORT_DIR / "phase13_setup_census_alarm_policy_v1.json"
CENSUS_PATH = REPORT_DIR / "phase13_setup_census_v1.json"
SETUP_SOURCE_PATH = REPORT_DIR / "phase14_setup_source_v1.json"
PACK_PATH = REPORT_DIR / "phase14_checkpoint_selection_pack_v1.json"

CENSUS_SPLIT = "train"
CENSUS_CANDIDATE = "P10-D"
CENSUS_ORDINALS = 8192            # per color -> 16,384 mixture + 16,384 neutral draws
PAIRED_BOARDS = 8192
LEARNED_MINIMUM = 10000
ADAPTER_CHECK_COUNT = 1024        # D6 neutral-branch bit-equality subsample
COLORS = ("red", "blue")

OPEN_FILES = (0, 1, 4, 5, 8, 9)
FRONT_RANK = 3

PACK_ORDINAL_BASE = 1_000_000
PACK_GAMES_PER_OPPONENT = 32
PACK_OPPONENTS = (
    "phase9_anchor",
    "strategic_rule_based",
    "tactical_rule_based",
    "stress_scout_rush",
)
PACK_SEED_PERSON = b"strat-p14"

# Pathology thresholds restated from the alarm policy; the census artifact
# also re-reads them from the policy file and refuses on any mismatch, so
# they cannot drift apart silently.
P_TRIVIAL_THRESHOLD = 0.05
P_PREDECISION_THRESHOLD = 0.025


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Flag metrics
# ---------------------------------------------------------------------------


def flag_metrics(canonical: tuple) -> dict:
    cell = canonical.index(FLAG)
    rank, file = cell // 10, cell % 10
    front = rank == FRONT_RANK
    return {
        "cell": cell,
        "rank": rank,
        "file": file,
        "front": front,
        "scout_lane": front and file in OPEN_FILES,
    }


class StageTally:
    """Aggregated Flag metrics of one (source, stage, color) band."""

    def __init__(self) -> None:
        self.count = 0
        self.rank_hist = [0, 0, 0, 0]
        self.file_hist = [0] * 10
        self.cell_hist = [0] * 40
        self.front = 0
        self.scout_lane = 0

    def add(self, metrics: dict) -> None:
        self.count += 1
        self.rank_hist[metrics["rank"]] += 1
        self.file_hist[metrics["file"]] += 1
        self.cell_hist[metrics["cell"]] += 1
        self.front += int(metrics["front"])
        self.scout_lane += int(metrics["scout_lane"])

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "flag_rank_histogram": self.rank_hist,
            "flag_file_histogram": self.file_hist,
            "flag_cell_histogram": self.cell_hist,
            "forwardmost_row_flags": self.front,
            "forwardmost_row_rate": round(self.front / self.count, 6) if self.count else 0.0,
            "scout_lane_exposed_flags": self.scout_lane,
            "scout_lane_exposed_rate": round(self.scout_lane / self.count, 6) if self.count else 0.0,
        }


# ---------------------------------------------------------------------------
# Census stage
# ---------------------------------------------------------------------------


def run_census() -> int:
    if not ALARM_POLICY_PATH.exists():
        print("REFUSED: the alarm policy must be written before census sampling:")
        print(f"  missing {ALARM_POLICY_PATH}")
        return 2
    policy = json.loads(ALARM_POLICY_PATH.read_text())
    policy_digest = sha256_of(ALARM_POLICY_PATH)
    thresholds = policy["classification"]["pathology"]["thresholds"]
    if (
        thresholds["P_trivial_pathological_at_or_above"] != P_TRIVIAL_THRESHOLD
        or thresholds["P_predecision_pathological_at_or_above"] != P_PREDECISION_THRESHOLD
    ):
        print("REFUSED: the policy file's thresholds differ from this runner's copy")
        return 2

    started = time.time()
    index = load_library_index()
    scorer = load_scorer()
    source = LearnedSetupSource(candidate(CENSUS_CANDIDATE), scorer, index)

    # -- library composition (read-only enumeration, attribution context) --
    composition: dict = {}
    for entry in index.entries:
        band = composition.setdefault(
            entry.split, {}
        ).setdefault(entry.family_id, {"count": 0, "flag_rank_histogram": [0, 0, 0, 0]})
        band["count"] += 1
        band["flag_rank_histogram"][entry.canonical_setup.index(FLAG) // 10] += 1

    # -- exact per-branch expectations from the frozen distributions --------
    entries = split_base_entries(CENSUS_SPLIT, index)
    base_front = [entry.canonical_setup.index(FLAG) // 10 == FRONT_RANK for entry in entries]
    base_lane = [
        front and (entries[i].canonical_setup.index(FLAG) % 10 in OPEN_FILES)
        for i, front in enumerate(base_front)
    ]
    irregular_ids = {
        entry.base_setup_id for entry in entries if entry.family_id == "F15"
    }
    exact_expectations = {}
    for color in COLORS:
        distribution = source.distribution(color, CENSUS_SPLIT)
        bands = {}
        for name, vector in (
            ("neutral_v1", distribution.p_neutral),
            ("p10d_learned", distribution.p_learned),
            ("production_mixture", distribution.p_mixed),
        ):
            front_mass = float(sum(p for p, f in zip(vector, base_front) if f))
            lane_mass = float(sum(p for p, f in zip(vector, base_lane) if f))
            family_mass: dict = {}
            for probability, entry in zip(vector, entries):
                family_mass[entry.family_id] = family_mass.get(entry.family_id, 0.0) + float(
                    probability
                )
            bands[name] = {
                "expected_forwardmost_row_rate": round(front_mass, 6),
                "expected_scout_lane_exposed_rate": round(lane_mass, 6),
                "family_mass": {k: round(v, 6) for k, v in sorted(family_mass.items())},
            }
        exact_expectations[color] = bands

    # -- sampled draws -------------------------------------------------------
    tallies: dict = {}

    def tally(source_name: str, stage: str, color: str) -> StageTally:
        return tallies.setdefault((source_name, stage, color), StageTally())

    defects: dict = {name: [] for name in ("D1", "D2", "D5", "D6")}
    defect_counts = {name: 0 for name in ("D1", "D2", "D3", "D4", "D5", "D6")}

    def record_defect(name: str, detail: dict) -> None:
        defect_counts[name] += 1
        if len(defects.setdefault(name, [])) < 5:
            defects[name].append(detail)

    mixture_draws: dict = {}
    branch_counts = {"neutral": 0, "learned": 0}
    exposure_by_family: dict = {}
    exposure_by_branch = {"neutral": 0, "learned": 0}
    adapter_checked = 0
    reflection_applied_count = 0
    perturbation_applied_count = 0
    stage_front_deltas = {"reflection": 0, "perturbation": 0}
    ordinals_used = 0

    def measure_draw(source_name: str, color: str, draw_like, base_entry, provenance) -> dict:
        nonlocal reflection_applied_count, perturbation_applied_count
        base = tuple(base_entry.canonical_setup)
        reflected = provenance["reflection_applied"]
        post_reflection = reflect_canonical(base) if reflected else base
        final = tuple(draw_like.canonical)
        base_m = flag_metrics(base)
        post_m = flag_metrics(post_reflection)
        final_m = flag_metrics(final)
        tally(source_name, "base", color).add(base_m)
        tally(source_name, "post_reflection", color).add(post_m)
        tally(source_name, "final", color).add(final_m)
        if source_name == "production_mixture":
            reflection_applied_count += int(bool(reflected))
            perturbation_applied_count += int(bool(provenance.get("perturbation_applied")))
        # stage deltas (must be zero unless a defect exists)
        if post_m["front"] != base_m["front"]:
            stage_front_deltas["reflection"] += 1
        if final_m["front"] != post_m["front"]:
            stage_front_deltas["perturbation"] += 1
        # D1: Flag may only mirror files under reflection, never move otherwise
        expected_file = 9 - base_m["file"] if reflected else base_m["file"]
        if final_m["rank"] != base_m["rank"] or final_m["file"] != expected_file:
            record_defect(
                "D1",
                {
                    "source": source_name,
                    "color": color,
                    "base_setup_id": base_entry.base_setup_id,
                    "base_flag_cell": base_m["cell"],
                    "final_flag_cell": final_m["cell"],
                    "reflection_applied": bool(reflected),
                },
            )
        # D2: family contract range on the final arrangement
        low, high = FAMILY_BY_ID[base_entry.family_id].allowed_ranges.get("flag_rank", (0, 3))
        if not low <= final_m["rank"] <= high:
            record_defect(
                "D2",
                {
                    "source": source_name,
                    "color": color,
                    "family_id": base_entry.family_id,
                    "base_setup_id": base_entry.base_setup_id,
                    "final_flag_rank": final_m["rank"],
                    "allowed": [low, high],
                },
            )
        # D5: independent mirror check of the accepted reflection helper
        if reflected:
            for rank in range(4):
                for file in range(10):
                    if post_reflection[rank * 10 + file] != base[rank * 10 + (9 - file)]:
                        record_defect(
                            "D5",
                            {
                                "source": source_name,
                                "color": color,
                                "base_setup_id": base_entry.base_setup_id,
                                "rank": rank,
                                "file": file,
                            },
                        )
                        break
                else:
                    continue
                break
        return final_m

    def census_pass(ordinal: int) -> None:
        nonlocal adapter_checked
        for color in COLORS:
            seed = selector_audit_seed(CENSUS_CANDIDATE, CENSUS_SPLIT, color, ordinal)
            draw = source.draw(
                SelectorRequest(split=CENSUS_SPLIT, color=color, selector_seed=seed)
            )
            mixture_draws[(color, ordinal)] = draw
            branch_counts[draw.branch] += 1
            base_entry = index.base(draw.base_setup_id)
            final_m = measure_draw(
                "production_mixture", color, draw.setup, base_entry, draw.setup_provenance
            )
            band = "learned" if draw.branch == "learned" else "neutral"
            tally(f"branch_{band}", "final", color).add(final_m)
            if final_m["front"]:
                exposure_by_branch[band] += 1
                exposure_by_family[draw.family_id] = exposure_by_family.get(draw.family_id, 0) + 1
            if draw.branch == "neutral" and adapter_checked < ADAPTER_CHECK_COUNT:
                adapter_checked += 1
                findings = neutral_branch_matches_accepted_sampler(draw, index)
                if findings:
                    record_defect(
                        "D6",
                        {
                            "color": color,
                            "ordinal": ordinal,
                            "base_setup_id": draw.base_setup_id,
                            "findings": findings[:3],
                        },
                    )
            # standalone accepted-sampler source under the same seed
            neutral = neutral_baseline_draw(CENSUS_SPLIT, seed, index)
            neutral_base = index.base(neutral.provenance["base_setup_id"])
            measure_draw("neutral_v1", color, neutral, neutral_base, neutral.provenance)

    for ordinal in range(CENSUS_ORDINALS):
        census_pass(ordinal)
    ordinals_used = CENSUS_ORDINALS
    while branch_counts["learned"] < LEARNED_MINIMUM:
        census_pass(ordinals_used)
        ordinals_used += 1

    # -- paired boards through the production orientation path ---------------
    paired = {
        "boards": 0,
        "blue_flag_immediate": 0,
        "red_flag_ply2_geometric": 0,
        "either_immediate": 0,
        "predecision_engine": 0,
        "engine_geometry_mismatches": 0,
        "either_by_branch_pair": {},
    }
    red_squares = SETUP_SQUARES[RED]
    blue_squares = SETUP_SQUARES[BLUE]

    for board in range(PAIRED_BOARDS):
        red_draw = mixture_draws[("red", board)]
        blue_draw = mixture_draws[("blue", board)]
        oriented_red = red_draw.oriented(RED)
        oriented_blue = blue_draw.oriented(BLUE)
        try:
            state = create_game(
                oriented_red,
                oriented_blue,
                rules=CORPUS_RULES,
                game_id=f"phase13-census-{board:05d}",
            )
        except Exception as error:  # noqa: BLE001 - defect evidence, not control flow
            record_defect("D3", {"board": board, "error": str(error)[:200]})
            continue
        red_flag_square = red_squares[oriented_red.index(FLAG)]
        blue_flag_square = blue_squares[oriented_blue.index(FLAG)]
        # D4: engine-frame Flag rows must agree with the orientation map
        red_rank = tuple(red_draw.setup.canonical).index(FLAG) // 10
        blue_rank = tuple(blue_draw.setup.canonical).index(FLAG) // 10
        red_record = state.piece_at(red_flag_square)
        blue_record = state.piece_at(blue_flag_square)
        red_ok = (
            red_flag_square // 10 == red_rank
            and red_record is not None
            and red_record.true_type == FLAG
            and red_record.owner == RED
        )
        blue_ok = (
            blue_flag_square // 10 == 9 - blue_rank
            and blue_record is not None
            and blue_record.true_type == FLAG
            and blue_record.owner == BLUE
        )
        if not (red_ok and blue_ok):
            record_defect(
                "D4",
                {
                    "board": board,
                    "red_flag_square": red_flag_square,
                    "blue_flag_square": blue_flag_square,
                    "red_canonical_rank": red_rank,
                    "blue_canonical_rank": blue_rank,
                },
            )
        paired["boards"] += 1
        blue_flag_immediate = any(
            oriented_red[30 + file] == SCOUT and blue_flag_square == 60 + file
            for file in OPEN_FILES
        )
        red_flag_immediate = any(
            oriented_blue[file] == SCOUT and red_flag_square == 30 + file
            for file in OPEN_FILES
        )
        predecision = any(
            decode_action(action)[1] == blue_flag_square for action in legal_actions(state)
        )
        if predecision != blue_flag_immediate:
            paired["engine_geometry_mismatches"] += 1
        paired["blue_flag_immediate"] += int(blue_flag_immediate)
        paired["red_flag_ply2_geometric"] += int(red_flag_immediate)
        paired["either_immediate"] += int(blue_flag_immediate or red_flag_immediate)
        paired["predecision_engine"] += int(predecision)
        if blue_flag_immediate or red_flag_immediate:
            key = f"{red_draw.branch}+{blue_draw.branch}"
            paired["either_by_branch_pair"][key] = paired["either_by_branch_pair"].get(key, 0) + 1

    boards = max(paired["boards"], 1)
    p_trivial = paired["either_immediate"] / boards
    p_predecision = paired["predecision_engine"] / boards

    total_defects = sum(defect_counts.values())
    pathology = p_trivial >= P_TRIVIAL_THRESHOLD or p_predecision >= P_PREDECISION_THRESHOLD
    if total_defects:
        classification = "DEFECT"
    elif pathology:
        classification = "PATHOLOGY"
    else:
        front_rate = tallies[("production_mixture", "final", "red")].front + tallies[
            ("production_mixture", "final", "blue")
        ].front
        classification = (
            "VALID_BUT_STRATEGICALLY_POOR" if front_rate else "NO_FINDING"
        )

    artifact = {
        "artifact": "phase13_setup_census_v1",
        "phase": 13,
        "agent": 1,
        "written_utc": utc_now(),
        "alarm_policy_sha256": policy_digest,
        "alarm_policy_written_utc": policy["written_utc"],
        "runner": "scripts/run_phase13_agent01.py --stage census",
        "split": CENSUS_SPLIT,
        "selector_candidate": CENSUS_CANDIDATE,
        "library_content_digest": index.content_digest,
        "seed_domain": (
            f"selector_audit_seed({CENSUS_CANDIDATE!r}, {CENSUS_SPLIT!r}, color, ordinal); "
            f"ordinals 0..{ordinals_used - 1} per color; paired board k = (red k, blue k); "
            "identical seeds drive the standalone neutral_v1 draws"
        ),
        "sample_sizes": {
            "ordinals_per_color": ordinals_used,
            "production_mixture_draws": 2 * ordinals_used,
            "neutral_v1_draws": 2 * ordinals_used,
            "learned_branch_draws": branch_counts["learned"],
            "neutral_branch_draws": branch_counts["neutral"],
            "paired_boards": paired["boards"],
            "adapter_equality_checks": adapter_checked,
        },
        "library_composition": composition,
        "exact_distribution_expectations": exact_expectations,
        "sampled_tallies": {
            f"{source_name}|{stage}|{color}": band.to_dict()
            for (source_name, stage, color), band in sorted(tallies.items())
        },
        "stage_effects": {
            "reflection_applied_mixture_draws": reflection_applied_count,
            "perturbation_applied_mixture_draws": perturbation_applied_count,
            "front_row_changes_after_reflection": stage_front_deltas["reflection"],
            "front_row_changes_after_perturbation": stage_front_deltas["perturbation"],
            "reading": (
                "reflection mirrors files only and perturbation pins the Flag cell, so both "
                "deltas must be exactly 0 in a defect-free run; the numbers above are measured, "
                "not assumed"
            ),
        },
        "exposure_attribution": {
            "forwardmost_flag_draws_by_branch": exposure_by_branch,
            "forwardmost_flag_draws_by_family": dict(sorted(exposure_by_family.items())),
            "irregular_family_id": "F15",
            "irregular_train_bases": len(irregular_ids),
        },
        "paired_measurements": {
            **{k: v for k, v in paired.items() if k != "either_by_branch_pair"},
            "either_by_branch_pair": dict(sorted(paired["either_by_branch_pair"].items())),
            "P_trivial": round(p_trivial, 6),
            "P_predecision": round(p_predecision, 6),
        },
        "defect_counts": defect_counts,
        "defect_examples": {k: v for k, v in defects.items() if v},
        "classification": {
            "result": classification,
            "P_trivial_threshold": P_TRIVIAL_THRESHOLD,
            "P_predecision_threshold": P_PREDECISION_THRESHOLD,
            "total_defect_observations": total_defects,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    measurement_payload = {
        k: v for k, v in artifact.items() if k not in ("written_utc", "elapsed_seconds")
    }
    artifact["census_content_digest"] = canonical_json_digest(measurement_payload)
    CENSUS_PATH.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"census written: {CENSUS_PATH}")
    print(
        f"classification={classification} P_trivial={p_trivial:.6f} "
        f"P_predecision={p_predecision:.6f} defects={total_defects} "
        f"learned_draws={branch_counts['learned']} elapsed={artifact['elapsed_seconds']}s"
    )
    return 0


# ---------------------------------------------------------------------------
# Pack stage
# ---------------------------------------------------------------------------


def pack_seed(token: str) -> int:
    digest = hashlib.blake2b(
        token.encode(), digest_size=8, person=PACK_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big")


def run_pack() -> int:
    if not SETUP_SOURCE_PATH.exists():
        print("REFUSED: freeze phase14_setup_source_v1 before building the pack:")
        print(f"  missing {SETUP_SOURCE_PATH}")
        return 2
    setup_source_doc = json.loads(SETUP_SOURCE_PATH.read_text())
    census = json.loads(CENSUS_PATH.read_text()) if CENSUS_PATH.exists() else None

    index = load_library_index()
    source = LearnedSetupSource(candidate(CENSUS_CANDIDATE), load_scorer(), index)

    games = []
    for opponent_index, opponent in enumerate(PACK_OPPONENTS):
        for game_index in range(PACK_GAMES_PER_OPPONENT):
            ordinal = PACK_ORDINAL_BASE + opponent_index * PACK_GAMES_PER_OPPONENT + game_index
            game_id = f"phase14_selection_pack_v1:{opponent}:{game_index:02d}"
            candidate_color = "red" if game_index % 2 == 0 else "blue"
            setups = {}
            for color in COLORS:
                seed = selector_audit_seed(CENSUS_CANDIDATE, CENSUS_SPLIT, color, ordinal)
                draw = source.draw(
                    SelectorRequest(split=CENSUS_SPLIT, color=color, selector_seed=seed)
                )
                player = RED if color == "red" else BLUE
                oriented = draw.oriented(player)
                setups[color] = {
                    "selector_seed": seed,
                    "branch": draw.branch,
                    "base_setup_id": draw.base_setup_id,
                    "final_setup_fingerprint": draw.final_setup_fingerprint,
                    "oriented_engine_setup": list(oriented),
                }
            games.append(
                {
                    "game_id": game_id,
                    "opponent": opponent,
                    "candidate_color": candidate_color,
                    "setup_ordinal": ordinal,
                    "red": setups["red"],
                    "blue": setups["blue"],
                    "opponent_decision_seed": pack_seed(f"{game_id}|role=opponent"),
                    "candidate_decision_seed": pack_seed(f"{game_id}|role=candidate"),
                }
            )

    pack = {
        "artifact": "phase14_checkpoint_selection_pack_v1",
        "phase": 13,
        "agent": 1,
        "written_utc": utc_now(),
        "runner": "scripts/run_phase13_agent01.py --stage pack",
        "purpose": (
            "fixed direct-policy engineering evaluation pack for Phase 14 candidate "
            "checkpoints; monitoring/selection infrastructure only — it may never change "
            "ongoing Phase 14 training, stop it early, or extend the deadline"
        ),
        "games_per_candidate": len(games),
        "opponents": {
            "phase9_anchor": {
                "kind": "neural",
                "checkpoint": "checkpoints/phase9/selfplay_c1_v1.pt",
                "sha256": "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea",
                "decision_rule": "greedy float32, fixed inference batch shape",
            },
            "strategic_rule_based": {"kind": "rule", "roster_id": "strategic_rule_based"},
            "tactical_rule_based": {"kind": "rule", "roster_id": "tactical_rule_based"},
            "stress_scout_rush": {"kind": "stress", "roster_id": "stress_scout_rush"},
        },
        "candidate_decision_rule": "greedy float32, fixed inference batch shape",
        "rules": {
            "rules_version": EVALUATION_RULES.rules_version,
            "board_geometry_version": EVALUATION_RULES.board_geometry_version,
            "first_player": EVALUATION_RULES.first_player,
            "battleless_move_limit": EVALUATION_RULES.battleless_move_limit,
            "absolute_move_limit": EVALUATION_RULES.absolute_move_limit,
            "two_square_rule_enabled": EVALUATION_RULES.two_square_rule_enabled,
            "continuous_chasing_rule_enabled": EVALUATION_RULES.continuous_chasing_rule_enabled,
            "context": "evaluation",
        },
        "setup_source": {
            "identity": setup_source_doc["identity"],
            "document_sha256": sha256_of(SETUP_SOURCE_PATH),
            "seed_domain": (
                f"selector_audit_seed({CENSUS_CANDIDATE!r}, {CENSUS_SPLIT!r}, color, "
                f"{PACK_ORDINAL_BASE}+k) — disjoint from the census ordinals"
            ),
        },
        "census_binding": {
            "census_content_digest": census["census_content_digest"] if census else None,
        },
        "scoring": {
            "ewr": "wins + 0.5*draws, per opponent stratum of 32 games",
            "colors": "16 candidate-red / 16 candidate-blue per stratum (game_index parity)",
        },
        "evaluation_implementation": (
            "the accepted evaluation machinery: engine create_game under the rules above, "
            "greedy neural decisions, frozen rule/stress policies from the accepted rosters "
            "seeded by opponent_decision_seed; Agent 2 binds the exact module identities and "
            "may not substitute policies, rules, boards, colors or seeds"
        ),
        "games": games,
    }
    pack["pack_content_digest"] = canonical_json_digest(
        {k: v for k, v in pack.items() if k != "written_utc"}
    )
    PACK_PATH.write_text(json.dumps(pack, indent=1) + "\n")
    front_rows = sum(
        1
        for game in games
        for color, offset in (("red", 30), ("blue", 0))
        if any(game[color]["oriented_engine_setup"][offset + f] == FLAG for f in range(10))
    )
    print(f"pack written: {PACK_PATH}")
    print(
        f"games={len(games)} pack_digest={pack['pack_content_digest'][:16]}... "
        f"front_row_flag_sides={front_rows}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("census", "pack"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "census":
        return run_census()
    return run_pack()


if __name__ == "__main__":
    raise SystemExit(main())
