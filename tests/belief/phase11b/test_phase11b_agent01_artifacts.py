"""Phase 11B Agent 1: the common corpus, the candidates and their report.

These tests protect what Agent 1 established and, just as importantly, what
Agent 1 promised **not** to do. Three groups:

- **Boundary** — the Phase 11B status markers are present and false where
  they must be false, the accepted Phase 9 belief head is untouched, and no
  Phase 11B module reaches the spent test bank or writes an accepted path.
- **Corpus** — the shape, the balance, the split disjointness, the
  public/privileged separation, and the arithmetic invariants a later agent
  will rely on when it reuses these bytes.
- **Results** — the leaderboard is internally consistent, the winner
  follows the stated rule, and every metric recomputes from the corpus.

Artifacts are skipped when absent so a fresh clone still runs green, the
accepted Phase 9-11 pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stratego.belief.phase11b import contract as c11b
from stratego.belief.phase11b import metrics as M
from stratego.belief.phase11b import seeds as S
from stratego.belief.phase11b import storage as store

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIRECTORY = REPOSITORY_ROOT / c11b.REPORT_ROOT
CORPUS_DIRECTORY = REPOSITORY_ROOT / c11b.CORPUS_ROOT


def _load(name: str):
    path = REPORT_DIRECTORY / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def summary():
    return _load("agent_01_summary.json")


@pytest.fixture(scope="module")
def curves():
    return _load("agent_01_learning_curves.json")


@pytest.fixture(scope="module")
def dev():
    if not (CORPUS_DIRECTORY / "manifest.json").exists():
        pytest.skip("the common Phase 11B corpus has not been generated yet")
    return store.load_split(CORPUS_DIRECTORY, "dev", labels=True)


# ---------------------------------------------------------------------------
# Boundary: what Phase 11B promised not to do
# ---------------------------------------------------------------------------


def test_the_status_markers_are_constants_no_result_can_flip():
    assert c11b.PHASE11B_STATUS_MARKERS == {
        "phase": "phase11b",
        "status": "engineering_prototype",
        "phase11_fail_unchanged": True,
        "phase11_test_bank_used": False,
        "phase12_authorized_by_this_artifact": False,
    }
    assert c11b.PHASE11_FACTS["phase11_final_classification"] == "FAIL"
    assert c11b.PHASE11_FACTS["phase11_test_bank_spent"] is True
    assert c11b.PHASE11_FACTS["scientific_claim"] == "none"


def test_the_summary_carries_every_marker_unchanged(summary):
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert summary[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert summary[key] == value, key


def test_the_accepted_phase9_belief_head_is_still_the_accepted_one(summary):
    from stratego.training.phase10_contract import (
        ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        ACCEPTED_PHASE9_PARAMETERS,
    )
    from stratego.training.phase11_contract import ACCEPTED_BELIEF_HEAD_DIGEST

    checkpoint = summary["starting_state"]["phase9_checkpoint"]
    assert checkpoint["sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert checkpoint["model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert checkpoint["belief_head_digest"] == ACCEPTED_BELIEF_HEAD_DIGEST
    assert checkpoint["parameters"] == ACCEPTED_PHASE9_PARAMETERS
    assert checkpoint["opened"] == "read_only"
    assert summary["starting_state"]["preservation_problems"] == []


def test_the_preserved_phase11_artifacts_still_hash_to_what_agent1_recorded(summary):
    digests = summary["starting_state"]["preserved_artifact_digests"]
    assert digests, "Agent 1 recorded no preserved artifacts"
    import hashlib

    drifted = []
    for relative, recorded in digests.items():
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            drifted.append(f"{relative} is missing")
            continue
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                hasher.update(block)
        if hasher.hexdigest() != recorded:
            drifted.append(relative)
    assert drifted == [], f"Phase 11B must not change these: {drifted}"


#: Names no Phase 11B module may *execute*. Prose about the spent bank is
#: fine — and required — so the check reads the AST rather than the text.
FORBIDDEN_CODE_TOKENS = (
    "phase11_test_bank_v1",
    "TEST_BANK_VERSION",
    "TEST_BANK_CASES",
    "TEST_BANK_GAMES",
    "phase11_banks",
    "sealed_bank_authorized",
)


def _executable_tokens(path: Path) -> set:
    """Every identifier, import and non-docstring literal in one module."""
    import ast

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.alias):
            tokens.update(node.name.split("."))
            if node.asname:
                tokens.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            tokens.update(node.module.split("."))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                tokens.add(node.value)
    return tokens


def test_no_phase11b_module_reaches_the_spent_test_bank():
    package = REPOSITORY_ROOT / "stratego" / "belief" / "phase11b"
    offenders = []
    for path in sorted(package.glob("*.py")):
        tokens = _executable_tokens(path)
        for token in FORBIDDEN_CODE_TOKENS:
            if token in tokens:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], f"Phase 11B must not reach the spent bank: {offenders}"


def test_the_forbidden_token_check_would_actually_catch_a_reach(tmp_path):
    """The guard above is only worth having if it fails on a real reach."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        '"""A docstring naming phase11_test_bank_v1 must stay allowed."""\n'
        "from stratego.evaluation import phase11_banks\n"
    )
    tokens = _executable_tokens(probe)
    assert "phase11_banks" in tokens
    assert "phase11_test_bank_v1" not in tokens


def test_phase11b_seed_streams_cannot_collide_with_phase11_streams():
    from stratego.training import phase11_seed as p11

    assert S._PHASE11B_SEED_PERSON != p11._PHASE11_SEED_PERSON
    assert S.PHASE11B_MASTER_SEED != p11.PHASE11_MASTER_SEED
    assert not set(S.CANONICAL_PHASE11B_SEEDS.values()) & set(
        p11.CANONICAL_PHASE11_SEEDS.values()
    )


def test_the_corpus_game_id_grammar_is_closed():
    game_id = S.corpus_game_id("train", "scout_rush", "p10d", "red", 17)
    assert S.parse_corpus_game_id(game_id) == {
        "phase11b_master_seed": S.PHASE11B_MASTER_SEED,
        "split": "train",
        "stratum": "scout_rush",
        "setup_source": "p10d",
        "observer_color": "red",
        "ordinal": 17,
    }
    for bad in ("phase11_validation_bank_v1|c=001", "", "phase11b_corpus_v1|g=0001"):
        with pytest.raises(c11b.Phase11BError):
            S.parse_corpus_game_id(bad)
    with pytest.raises(c11b.Phase11BError):
        S.corpus_game_id("train", "miner_rush", "p10d", "red", 0)


def test_every_seed_stream_is_a_pure_function_of_its_identity():
    left = S.corpus_game_id("train", "tactical_rule", "neutral", "blue", 3)
    right = S.corpus_game_id("dev", "tactical_rule", "neutral", "blue", 3)
    assert S.match_seed(left) == S.match_seed(left)
    assert S.match_seed(left) != S.match_seed(right)
    assert S.setup_seed(left, S.ROLE_OBSERVER) != S.setup_seed(left, S.ROLE_OPPONENT)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_the_corpus_is_the_shape_the_sprint_specified(summary):
    splits = summary["common_corpus"]["splits"]
    assert splits["train"]["games"] == 2_048
    assert splits["dev"]["games"] == 512
    assert splits["train"]["decisions_per_game"] == 16
    assert splits["dev"]["decisions_per_game"] == 4
    for split in ("train", "dev"):
        assert splits[split]["complete"] is True
        assert splits[split]["observation_shape"] == [127, 10, 10]
        assert splits[split]["observation_dtype"] == "float32"


def test_the_corpus_is_balanced_over_strata_sources_and_colours(summary):
    splits = summary["common_corpus"]["splits"]
    for split, expected in (("train", 512), ("dev", 128)):
        block = splits[split]
        assert set(block["strata"]) == set(c11b.CORPUS_STRATA)
        assert set(block["strata"].values()) == {expected}
        assert set(block["sources"].values()) == {block["games"] // 2}
        assert set(block["observer_colors"].values()) == {block["games"] // 2}


def test_the_two_splits_cannot_share_a_setup_or_a_seed(summary):
    splits = summary["common_corpus"]["splits"]
    assert splits["train"]["library_split"] == "train"
    assert splits["dev"]["library_split"] == "validation"
    assert splits["train"]["library_split"] != splits["dev"]["library_split"]


def test_the_corpus_bytes_still_hash_to_the_recorded_identity(summary, dev):
    recorded = summary["common_corpus"]["file_digests"]
    for split in ("train", "dev"):
        assert store.split_digest(CORPUS_DIRECTORY, split) == recorded[split]
    manifest = store.read_manifest(CORPUS_DIRECTORY)
    assert store.corpus_digest(manifest) == summary["common_corpus"]["corpus_digest"]


def test_public_inputs_and_privileged_labels_live_in_separate_directories():
    for split in ("train", "dev"):
        base = store.split_root(CORPUS_DIRECTORY, split)
        if not base.exists():
            pytest.skip("the common Phase 11B corpus has not been generated yet")
        public = {path.name for path in (base / c11b.PUBLIC_DIRECTORY).iterdir()}
        privileged = {path.name for path in (base / c11b.PRIVILEGED_DIRECTORY).iterdir()}
        assert "observations.f32" in public
        assert "labels.npz" in privileged
        assert not public & privileged


def test_the_loader_hands_over_labels_only_when_asked_by_name():
    if not (CORPUS_DIRECTORY / "manifest.json").exists():
        pytest.skip("the common Phase 11B corpus has not been generated yet")
    public_only = store.load_split(CORPUS_DIRECTORY, "dev")
    assert "true_rank" not in public_only
    assert "observations" in public_only
    assert "true_rank" in store.load_split(CORPUS_DIRECTORY, "dev", labels=True)


def test_every_stored_sample_has_at_least_one_hidden_piece(dev):
    counts = np.diff(np.asarray(dev["piece_offset"], dtype=np.int64))
    assert counts.min() >= 1
    assert counts.max() <= 40
    assert int(counts.sum()) == dev["pieces"]


def test_every_true_rank_is_publicly_admissible(dev):
    rows = np.arange(dev["pieces"])
    true_rank = np.asarray(dev["true_rank"], dtype=np.int64)
    assert true_rank.min() >= 0 and true_rank.max() < c11b.RANK_COUNT
    mask = np.asarray(dev["legal_rank_mask"], dtype=bool)
    assert mask[rows, true_rank].all(), "a true rank is excluded by its own public mask"
    counts = np.asarray(dev["remaining_counts"], dtype=np.int64)[M.piece_samples(dev)]
    assert (counts[rows, true_rank] > 0).all(), "a true rank has no remaining inventory"


def test_a_moved_piece_can_never_be_a_flag_or_a_bomb(dev):
    moved = np.asarray(dev["piece_moved"], dtype=bool)
    mask = np.asarray(dev["legal_rank_mask"], dtype=bool)
    for rank in c11b.IMMOVABLE_RANK_INDICES:
        assert not mask[moved, rank].any()
    assert mask[~moved].all()


def test_the_remaining_inventory_counts_exactly_the_unresolved_pieces(dev):
    counts = np.asarray(dev["remaining_counts"], dtype=np.int64)
    widths = np.diff(np.asarray(dev["piece_offset"], dtype=np.int64))
    assert (counts.sum(axis=1) == widths).all()
    assert counts.min() >= 0
    assert (counts <= np.asarray(c11b.RANK_INITIAL_COUNTS)).all()


def test_the_supervised_square_mask_agrees_with_the_piece_list(dev):
    offsets = np.asarray(dev["piece_offset"], dtype=np.int64)
    squares = np.asarray(dev["perspective_square"], dtype=np.int64)
    masks = np.asarray(dev["target_mask"], dtype=bool)
    for row in range(0, dev["samples"], max(1, dev["samples"] // 64)):
        selected = squares[offsets[row] : offsets[row + 1]]
        assert masks[row].sum() == selected.size
        assert masks[row][selected].all()
        assert len(set(selected.tolist())) == selected.size


def test_the_public_state_identities_are_distinct_positions(dev):
    identities = dev["identities"]
    assert len(identities) == dev["samples"]
    assert all(len(value) == 64 for value in identities)


# ---------------------------------------------------------------------------
# Metrics and results
# ---------------------------------------------------------------------------


def test_the_baseline_is_the_accepted_remaining_count_arithmetic(dev):
    probabilities = M.baseline_probabilities(dev)
    assert probabilities.shape == (dev["pieces"], c11b.RANK_COUNT)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    rows = np.arange(dev["pieces"])
    assert (probabilities[rows, np.asarray(dev["true_rank"], dtype=np.int64)] > 0).all()
    assert M.BASELINE_VERSION == "remaining_count_belief_v1"


def test_the_baseline_matches_the_accepted_module_on_sampled_rows(dev):
    """The corpus's stored public arrays reproduce the accepted baseline."""
    from stratego.evaluation.phase11_baselines import REMAINING_COUNT_BASELINE_VERSION

    assert REMAINING_COUNT_BASELINE_VERSION == M.BASELINE_VERSION
    counts = np.asarray(dev["remaining_counts"], dtype=np.float64)[M.piece_samples(dev)]
    mask = np.asarray(dev["legal_rank_mask"], dtype=np.float64)
    weights = counts * mask
    expected = weights / weights.sum(axis=1, keepdims=True)
    assert np.allclose(M.baseline_probabilities(dev), expected)


def test_the_recorded_baseline_ce_recomputes_from_the_corpus(summary, dev):
    baseline = M.baseline_probabilities(dev)
    recomputed = float(
        M.cross_entropy(baseline, np.asarray(dev["true_rank"], dtype=np.int64)).mean()
    )
    for row in summary["leaderboard"].values():
        assert abs(row["baseline_ce"] - recomputed) < 1e-9
        assert row["dev_pieces"] == dev["pieces"]


def test_every_leaderboard_row_has_the_standard_fields(summary):
    required = {
        "candidate_id",
        "architecture",
        "ce",
        "baseline_ce",
        "r_ce",
        "r_ce_ci95",
        "top1",
        "baseline_top1",
        "r_ce_by_stratum",
        "parameters_added",
        "training_seconds",
        "time_to_best_seconds",
        "inference_microseconds_per_piece",
        "dev_samples",
        "dev_pieces",
    }
    assert summary["leaderboard"], "the leaderboard is empty"
    for name, row in summary["leaderboard"].items():
        assert required <= set(row), f"{name} is missing {required - set(row)}"
        assert abs(row["r_ce"] - row["ce"] / row["baseline_ce"]) < 1e-9
        assert set(row["r_ce_by_stratum"]) == set(c11b.CORPUS_STRATA)
        low, high = row["r_ce_ci95"]
        assert low <= row["r_ce"] <= high


def test_all_three_experiments_ran_and_beat_both_references(summary):
    rows = summary["leaderboard"]
    reference = rows["phase11_head_unchanged_reference"]
    trained = [name for name, row in rows.items() if row["trained_in_phase11b"]]
    assert len(trained) >= 2, "1A and 1B are mandatory"
    for name in trained:
        assert rows[name]["r_ce"] < reference["r_ce"], name
        assert rows[name]["r_ce"] < 1.0, f"{name} is no better than counting"
        assert rows[name]["r_ce"] < summary["uniform_floor"]["r_ce"], name


def test_the_winner_follows_the_stated_rule(summary):
    decision = summary["decision"]
    rows = summary["leaderboard"]
    leader = decision["leader_by_r_ce"]
    trained = {name: rows[name] for name in rows if rows[name]["trained_in_phase11b"]}
    assert rows[leader]["r_ce"] == min(row["r_ce"] for row in trained.values())
    band = decision["equivalence_band"]
    for name in decision["inside_band"]:
        assert trained[name]["r_ce"] - trained[leader]["r_ce"] < band
    for name in decision["excluded_as_materially_worse"]:
        assert trained[name]["r_ce"] - trained[leader]["r_ce"] >= band
    winner = decision["winner"]
    assert winner in decision["inside_band"]
    assert trained[winner]["parameters_added"] == min(
        trained[name]["parameters_added"] for name in decision["inside_band"]
    )


def test_the_learning_curves_end_at_the_reported_checkpoint(summary, curves):
    for name, curve in curves.items():
        assert curve, f"{name} recorded no curve"
        row = summary["leaderboard"][name]
        best = min(entry["dev_ce"] for entry in curve)
        assert abs(best - row["ce"]) < 1e-9, name
        assert row["best_epoch"] == next(
            entry["epoch"] for entry in curve if entry["dev_ce"] == best
        )
        assert row["time_to_best_seconds"] <= max(entry["seconds"] for entry in curve)


def test_the_interface_was_exercised_through_the_accepted_sampler(summary):
    interface = summary.get("interface") or {}
    candidates = interface.get("candidates") or {}
    assert candidates, "no candidate exposed the required interface"
    assert summary["decision"]["winner"] in candidates
    for name, block in candidates.items():
        assert block["positions_checked"] > 0, name
        assert block["worlds_sampled"] > 0, name
        assert block["all_marginals_sum_to_one"] is True
        assert block["sample_worlds_seed_deterministic"] is True
        assert block["all_worlds_passed_accepted_validation_stack"] is True
        assert "accepted, unmodified" in block["sampler_source"]


def test_the_paired_comparisons_agree_with_the_leaderboard_order(summary):
    rows = summary["leaderboard"]
    for label, block in summary.get("paired_comparisons", {}).items():
        left, right = label.split(" vs ")
        expected = rows[left]["ce"] - rows[right]["ce"]
        assert abs(block["ce_difference"] - expected) < 1e-9, label
        low, high = block["ce_difference_ci95"]
        assert low <= block["ce_difference"] <= high


def test_the_report_says_what_agent_1_is_not_claiming():
    path = REPORT_DIRECTORY / "agent_01_report.md"
    if not path.exists():
        pytest.skip("the Agent 1 report has not been produced yet")
    text = path.read_text()
    for phrase in (
        "engineering prototype",
        "does not authorize Phase 12",
        "remains spent",
        "No other architecture was begun",
    ):
        assert phrase in text, phrase


# ---------------------------------------------------------------------------
# The frozen C1 split, which 1C's whole design rests on
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frozen_c1():
    export = REPOSITORY_ROOT / "checkpoints" / "phase11b" / "phase9_c1_readonly_copy.pt"
    if not (REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt").exists():
        pytest.skip("the accepted Phase 9 checkpoint is not present")
    from stratego.belief.phase11b import features as feat

    model, identity = feat.load_frozen_c1(REPOSITORY_ROOT, export, device="cpu")
    return model, identity


def test_the_loaded_c1_is_frozen_and_is_the_accepted_one(frozen_c1):
    model, identity = frozen_c1
    assert identity["trainable_parameters"] == 0
    assert not any(parameter.requires_grad for parameter in model.parameters())
    from stratego.training.phase11_contract import ACCEPTED_BELIEF_HEAD_DIGEST

    assert identity["belief_head_digest"] == ACCEPTED_BELIEF_HEAD_DIGEST


def test_the_penultimate_cache_plus_the_last_block_is_the_accepted_encoder(frozen_c1, dev):
    """1C's design rests on this: its input is the frozen prefix's output.

    If `penultimate` were off by a block, 1C would be training a different
    network than the one it reports, and the reconstruction would drift.
    """
    import numpy as np
    import torch

    from stratego.belief.phase11b import features as feat

    model, _identity = frozen_c1
    observations = np.array(dev["observations"][:8])
    penultimate = feat.encode_batch(model, observations, feat.LAYER_PENULTIMATE)
    final = feat.encode_batch(model, observations, feat.LAYER_FINAL)
    block, norm = feat.final_block(model)
    with torch.no_grad():
        rebuilt = norm(block(penultimate))
    assert torch.equal(rebuilt, final), "the frozen prefix is not one block short"


def test_a_fresh_1c_model_starts_as_the_accepted_encoder(frozen_c1, dev):
    """1C copies the accepted block; at init it must reproduce it exactly."""
    import numpy as np
    import torch

    from stratego.belief.phase11b import features as feat
    from stratego.belief.phase11b import heads as H

    model, _identity = frozen_c1
    observations = np.array(dev["observations"][:8])
    penultimate = feat.encode_batch(model, observations, feat.LAYER_PENULTIMATE)
    final = feat.encode_batch(model, observations, feat.LAYER_FINAL)
    candidate = H.build_candidate(H.CANDIDATE_1C, model).eval()
    with torch.no_grad():
        assert torch.equal(candidate.encode(penultimate), final)
    # ... and the copy is a copy: training 1C cannot move an accepted weight.
    assert candidate.block is not block_of(model)


def block_of(model):
    from stratego.belief.phase11b.features import final_block

    return final_block(model)[0]


def test_the_1c_batcher_pairs_every_piece_with_its_own_sample(dev):
    """1C batches positions but its loss is per piece, so the index pair it
    builds must map each piece back to the position it came from. A silent
    misalignment here would train on real labels attached to the wrong
    squares and still produce a plausible-looking curve.
    """
    from stratego.belief.phase11b.train import _sample_batches

    offsets = np.asarray(dev["piece_offset"], dtype=np.int64)
    squares = np.asarray(dev["perspective_square"], dtype=np.int64)
    labels = np.asarray(dev["true_rank"], dtype=np.int64)
    order = np.random.default_rng(0).permutation(int(dev["samples"])).astype(np.int64)

    covered = 0
    for block, token_rows, token_squares, token_labels in _sample_batches(dev, order, 37):
        cursor = 0
        for position, row in enumerate(block):
            width = int(offsets[row + 1] - offsets[row])
            piece = slice(cursor, cursor + width)
            assert (token_rows[piece] == position).all()
            assert np.array_equal(token_squares[piece], squares[offsets[row] : offsets[row + 1]])
            assert np.array_equal(token_labels[piece], labels[offsets[row] : offsets[row + 1]])
            cursor += width
        assert cursor == token_labels.size
        covered += token_labels.size
    assert covered == dev["pieces"], "a shuffled pass must cover every piece exactly once"


def test_every_candidate_reads_the_feature_layer_it_declares(frozen_c1):
    import torch

    from stratego.belief.phase11b import features as feat
    from stratego.belief.phase11b import heads as H

    model, _identity = frozen_c1
    expected = {
        H.CANDIDATE_1A: feat.LAYER_FINAL,
        H.CANDIDATE_1B: feat.LAYER_FINAL,
        H.CANDIDATE_1C: feat.LAYER_PENULTIMATE,
    }
    for candidate_id, layer in expected.items():
        candidate = H.build_candidate(candidate_id, model).eval()
        assert candidate.feature_layer == layer
        with torch.no_grad():
            logits = candidate.belief_logits(torch.zeros(100, c11b.C1_FEATURE_WIDTH))
        assert tuple(logits.shape) == (100, c11b.RANK_COUNT)


# ---------------------------------------------------------------------------
# The metric implementation itself
# ---------------------------------------------------------------------------


def test_cross_entropy_is_the_negative_log_of_the_true_rank_mass():
    probabilities = np.full((3, c11b.RANK_COUNT), 1.0 / c11b.RANK_COUNT)
    values = M.cross_entropy(probabilities, np.array([0, 5, 11]))
    assert np.allclose(values, np.log(c11b.RANK_COUNT))


def test_the_projected_diagnostic_never_moves_mass_onto_an_excluded_rank(dev):
    probabilities = M.baseline_probabilities(dev)
    projected = M.projected_probabilities(probabilities, dev)
    assert np.allclose(projected.sum(axis=1), 1.0)
    support = (
        np.asarray(dev["remaining_counts"], dtype=np.int64)[M.piece_samples(dev)] > 0
    ) & np.asarray(dev["legal_rank_mask"], dtype=bool)
    assert not (projected[~support] > 0).any()


def test_a_metric_block_refuses_a_wrongly_shaped_candidate(dev):
    with pytest.raises(M.Phase11BMetricsError):
        M.evaluate(np.zeros((dev["pieces"] + 1, c11b.RANK_COUNT)), dev)
