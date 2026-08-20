"""Phase 11B Agent 4: the hybrid CNN, its two inputs, its results, its report.

These tests protect what Agent 4 measured and, just as importantly, what
Agent 4 promised **not** to do. Five groups:

- **Boundary** — the Phase 11B status markers survive on every artifact, the
  common corpus was reused byte-for-byte, no Agent 1, 2, 3 or Phase 11
  artifact moved, Agent 3's field cache was reused rather than rebuilt, and
  no Agent 4 module reaches the spent test bank.
- **Inputs** — the seam is Agent 3's unchanged, the fused tensor is exactly
  the public observation and Agent 3's field side by side with nothing
  between them, and neither branch can carry a label.
- **Model** — the parameter count is inside the instructed band and inside
  1,440 of Agents 2 and 3, the tower is Agent 2's by import rather than by
  resemblance, the two entry points compute the same function, and C1
  carries no gradient.
- **Interface** — the required two methods exist, produce probability
  vectors, and reach worlds only through the accepted Phase 11 sampler.
- **Results** — every leaderboard number recomputes from the corpus, the
  complementarity reading is arithmetic on reported numbers, the table
  `04_AGENT_4` asks for is in the order it asks for, and the verdict follows
  the stated rule.

Artifacts are skipped when absent so a fresh clone still runs green, the
accepted Phase 9-11 pattern.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stratego.belief.phase11b import contract as c11b
from stratego.belief.phase11b import metrics as M
from stratego.belief.phase11b import storage as store

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIRECTORY = REPOSITORY_ROOT / c11b.REPORT_ROOT
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / c11b.CHECKPOINT_ROOT
CORPUS_DIRECTORY = REPOSITORY_ROOT / c11b.CORPUS_ROOT
CANDIDATE_4 = "agent04_hybrid_raw_c1_cnn"
CANDIDATE_3 = "agent03_c1_feature_cnn"
CANDIDATE_2 = "agent02_raw_observation_cnn"
CANDIDATE_1B = "agent01_1b_attached_mlp_head"

AGENT4_SOURCES = (
    "stratego/belief/phase11b/hybrid_cnn.py",
    "scripts/run_phase11b_agent04.py",
    "scripts/_phase11b_agent04_report.py",
)

#: Names that would mean an Agent 4 module had reached the spent Phase 11
#: test bank. Matched against executable tokens only, never docstrings.
FORBIDDEN_CODE_TOKENS = frozenset(
    {
        "phase11_test_bank_v1",
        "agent_01_test_bank",
        "phase11_banks",
        "load_test_bank",
        "sealed_test_bank",
    }
)


def _load(name: str):
    path = REPORT_DIRECTORY / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def summary():
    return _load("agent_04_summary.json")


@pytest.fixture(scope="module")
def curve():
    return _load("agent_04_learning_curve.json")


@pytest.fixture(scope="module")
def agent3():
    return _load("agent_03_summary.json")


@pytest.fixture(scope="module")
def agent2():
    return _load("agent_02_summary.json")


@pytest.fixture(scope="module")
def agent1():
    return _load("agent_01_summary.json")


@pytest.fixture(scope="module")
def dev():
    if not (CORPUS_DIRECTORY / "manifest.json").exists():
        pytest.skip("the common Phase 11B corpus has not been generated yet")
    return store.load_split(CORPUS_DIRECTORY, "dev", labels=True)


@pytest.fixture(scope="module")
def hybrid_model():
    from stratego.belief.phase11b.hybrid_cnn import build_hybrid_cnn

    return build_hybrid_cnn(seed=17)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Boundary: what Agent 4 promised not to do
# ---------------------------------------------------------------------------


def test_the_summary_carries_every_status_marker_unchanged(summary):
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert summary[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert summary[key] == value, key
    assert summary["agent"] == 4


def test_the_curve_and_the_checkpoint_carry_the_markers_too(summary, curve):
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert curve[key] == value, key
    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 4 checkpoint is not present")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert payload[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert payload[key] == value, key


def test_the_common_corpus_was_reused_not_regenerated(summary, agent1, agent2, agent3):
    corpus = summary["common_corpus"]
    for earlier in (agent1, agent2, agent3):
        assert corpus["corpus_digest"] == earlier["common_corpus"]["corpus_digest"]
    assert corpus["corpus_digest_matches_manifest"] is True
    assert corpus["file_digest_drift"] == []
    assert summary["preservation"]["corpus_regenerated"] is False
    for split in c11b.CORPUS_SPLITS:
        assert corpus["file_digests"][split] == agent1["common_corpus"]["file_digests"][split]


def test_the_corpus_bytes_still_hash_to_what_agent4_scored_on(summary, dev):
    """The recorded identity is only worth having if it still holds."""
    recomputed = {
        split: store.split_digest(CORPUS_DIRECTORY, split) for split in c11b.CORPUS_SPLITS
    }
    assert recomputed == summary["common_corpus"]["file_digests"]


def test_no_earlier_or_phase11_artifact_moved(summary):
    drifted = []
    for relative, digest in summary["preserved_artifact_digests"].items():
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            drifted.append(f"{relative}: missing")
        elif _sha256(path) != digest:
            drifted.append(f"{relative}: changed")
    assert drifted == [], f"Agent 4 must leave these untouched: {drifted}"
    preservation = summary["preservation"]
    assert preservation["agent1_artifacts_modified"] is False
    assert preservation["agent2_artifacts_modified"] is False
    assert preservation["agent3_artifacts_modified"] is False
    assert preservation["c1_modified"] is False
    assert preservation["artifacts_unchanged_since_agent3"] is True
    assert preservation["phase11_test_bank_opened"] is False


def test_the_preserved_set_covers_every_module_agent4_builds_on(summary):
    """A preservation claim is only as good as the list behind it."""
    preserved = set(summary["preserved_artifact_digests"])
    for relative in (
        "stratego/belief/phase11b/features.py",
        "stratego/belief/phase11b/heads.py",
        "stratego/belief/phase11b/interface.py",
        "stratego/belief/phase11b/metrics.py",
        "stratego/belief/phase11b/raw_cnn.py",
        "stratego/belief/phase11b/raw_train.py",
        "stratego/belief/phase11b/feature_seam.py",
        "stratego/belief/phase11b/feature_cnn.py",
        "stratego/model/production_model.py",
        "reports/phase11b/agent_01_summary.json",
        "reports/phase11b/agent_02_summary.json",
        "reports/phase11b/agent_03_summary.json",
    ):
        assert relative in preserved, relative


def _executable_tokens(path: Path) -> set:
    """Every name and string constant that is *not* a docstring."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.alias):
            tokens.add(node.name.split(".")[-1])
            if node.asname:
                tokens.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            tokens.update(node.module.split("."))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                tokens.add(node.value)
    return tokens


def test_no_agent4_module_reaches_the_spent_test_bank():
    offenders = []
    for relative in AGENT4_SOURCES:
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            continue
        tokens = _executable_tokens(path)
        for token in FORBIDDEN_CODE_TOKENS:
            if token in tokens:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], f"Agent 4 must not reach the spent bank: {offenders}"


# ---------------------------------------------------------------------------
# Inputs: Agent 3's seam unchanged, and a fusion that is only its two sources
# ---------------------------------------------------------------------------


def test_the_seam_is_agent3s_and_was_not_modified(summary, agent3):
    from stratego.belief.phase11b import feature_seam as fs

    assert summary["frozen_seam"] == agent3["frozen_seam"]
    assert summary["frozen_seam"] == fs.SEAM_DESCRIPTION
    assert summary["frozen_seam_matches_agent3_record"] is True
    assert summary["frozen_seam_source"].startswith("agent_03")
    assert summary["feature_cache"]["c1_field_rebuilt"] is False
    assert summary["preservation"]["agent3_field_cache_rebuilt"] is False


def test_the_reused_field_cache_still_digests_to_what_agent3_published(summary, agent3):
    reused = summary["feature_cache"]["c1_field_reused_from_agent3"]
    published = agent3["feature_cache"]["caches"]
    for split, block in reused.items():
        assert block["digest"] == published[split]["digest"], split
        assert block["digest_matches_agent3"] is True, split
        assert block["rebuilt_by_agent4"] is False, split


def test_the_reused_field_re_encodes_from_the_public_observations(summary):
    for split, block in summary["feature_cache"]["c1_field_verification"].items():
        assert block["bit_identical"] is True, split
        assert block["max_absolute_difference"] == 0.0, split
        assert block["inputs"] == ["public observation", "accepted frozen C1 weights"]


def test_the_fused_input_is_exactly_its_two_sources(summary):
    for split, block in summary["feature_cache"]["fused_input_verification"].items():
        assert block["bit_identical"] is True, split
        assert block["raw_half_is_the_corpus_observation"] is True, split
        assert block["c1_half_is_agent3s_field"] is True, split
        assert block["raw_half_max_absolute_difference"] == 0.0, split
        assert block["c1_half_max_absolute_difference"] == 0.0, split


def test_the_fused_cache_metadata_says_it_holds_no_labels(summary):
    for split, block in summary["feature_cache"]["fused_input"].items():
        assert block["contains_labels"] is False, split
        assert block["channel_layout"]["public_observation"] == [0, 127], split
        assert block["channel_layout"]["frozen_c1_field"] == [127, 255], split
        assert "privileged" not in block["path"], split


def test_the_fusion_on_disk_re_derives_from_the_corpus_and_agent3s_cache(summary, dev):
    """The claim "this is only its two sources" re-checked against the bytes."""
    import torch

    from stratego.belief.phase11b import feature_seam as fs
    from stratego.belief.phase11b.hybrid_cnn import fuse_arrays, load_fused_cache

    fused_path = REPOSITORY_ROOT / summary["feature_cache"]["fused_input"]["dev"]["path"]
    field_path = REPOSITORY_ROOT / (
        summary["feature_cache"]["c1_field_reused_from_agent3"]["dev"]["path"]
    )
    if not (fused_path.exists() and field_path.exists()):
        pytest.skip("the Agent 4 fused cache or Agent 3's field cache is not present")
    samples = int(dev["samples"])
    cache = load_fused_cache(fused_path, expected_samples=samples)
    field = fs.load_field_cache(field_path, expected_samples=samples)
    rows = np.linspace(0, samples - 1, num=24, dtype=np.int64)
    rebuilt = fuse_arrays(
        np.asarray(dev["observations"])[rows], np.asarray(field[rows], dtype=np.float32)
    )
    stored = np.asarray(cache[rows], dtype=np.float32)
    assert np.array_equal(rebuilt, stored)
    # And each half really is the object it claims to be.
    assert np.array_equal(stored[:, :127], np.asarray(dev["observations"])[rows])
    assert np.array_equal(
        torch.from_numpy(stored[:, 127:]),
        fs.field_to_planes(torch.from_numpy(np.asarray(field[rows], dtype=np.float32))),
    )


def test_the_true_ranks_are_not_reachable_from_either_branch(dev):
    """The labels live somewhere the model's inputs are not."""
    from stratego.belief.phase11b.hybrid_cnn import FUSED_SHAPE

    assert "true_rank" in dev
    public_only = store.load_split(CORPUS_DIRECTORY, "dev", labels=False)
    assert "true_rank" not in public_only
    # The fused tensor's width is fully accounted for by the two public
    # sources, so there is no spare channel a label could occupy.
    assert FUSED_SHAPE[0] == c11b.OBSERVATION_SHAPE[0] + c11b.C1_FEATURE_WIDTH


# ---------------------------------------------------------------------------
# Model: capacity, inheritance, and the two entry points
# ---------------------------------------------------------------------------


def test_the_parameter_count_is_inside_the_instructed_band(hybrid_model):
    from stratego.belief.phase11b.hybrid_cnn import parameter_breakdown

    total = parameter_breakdown(hybrid_model)["total"]
    assert 3_000_000 <= total <= 5_000_000


def test_the_capacity_matches_agents_2_and_3_so_the_input_is_the_variable(summary):
    from stratego.belief.phase11b.hybrid_cnn import parameter_breakdown, build_hybrid_cnn

    total = parameter_breakdown(build_hybrid_cnn(seed=1))["total"]
    held = summary["complementarity"]["capacity_held_fixed"]
    assert held["agent04_hybrid_raw_c1_cnn"] == total
    assert abs(total - held["agent02_raw_observation_cnn"]) <= 1_500
    assert abs(total - held["agent03_c1_feature_cnn"]) <= 1_500


def test_the_reported_parameter_count_is_the_model_that_was_trained(summary, hybrid_model):
    from stratego.belief.phase11b.hybrid_cnn import parameter_count

    row = summary["leaderboard"][CANDIDATE_4]
    assert row["parameters"] == parameter_count(hybrid_model)
    assert row["parameters_trained"] == row["parameters"]
    assert summary["pilot"]["parameters"]["total"] == row["parameters"]


def test_the_tower_and_readout_are_agent2s_by_import_not_by_resemblance():
    """A shared block class, not a copy that could drift."""
    from stratego.belief.phase11b import hybrid_cnn as hc
    from stratego.belief.phase11b import raw_cnn as rc

    assert hc.ResidualBlock is rc.ResidualBlock
    source = (REPOSITORY_ROOT / "stratego/belief/phase11b/hybrid_cnn.py").read_text()
    tree = ast.parse(source)
    declared = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert "ResidualBlock" not in declared
    model = hc.build_hybrid_cnn(seed=3)
    reference = rc.build_raw_cnn(seed=3)
    assert model.width == reference.width
    assert model.blocks_count == reference.blocks_count
    assert model.readout_width == reference.readout_width
    assert hc.parameter_count(model.blocks) == rc.parameter_count(reference.blocks)
    assert hc.parameter_count(model.readout) == rc.parameter_count(reference.readout)


def test_the_two_branches_split_the_inherited_width_evenly(hybrid_model):
    from stratego.belief.phase11b import raw_cnn as rc

    assert hybrid_model.raw_branch_width == hybrid_model.c1_branch_width
    assert (
        hybrid_model.raw_branch_width + hybrid_model.c1_branch_width
        == rc.RAW_CNN_WIDTH
        == hybrid_model.width
    )


def test_the_model_takes_only_public_inputs(hybrid_model):
    """Neither entry point has an argument a hidden rank could arrive in."""
    import inspect

    forward = inspect.signature(hybrid_model.forward)
    assert list(forward.parameters) == ["fused"]
    parts = inspect.signature(hybrid_model.forward_parts)
    assert list(parts.parameters) == ["observations", "field"]


def test_the_two_entry_points_compute_the_same_function(hybrid_model):
    """`forward(fused)` and `forward_parts(obs, field)` must not drift."""
    import torch

    from stratego.belief.phase11b.feature_seam import field_to_planes
    from stratego.belief.phase11b.hybrid_cnn import fuse_arrays

    generator = np.random.default_rng(4)
    observations = generator.random((3, *c11b.OBSERVATION_SHAPE)).astype(np.float32)
    field = generator.random((3, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)).astype(np.float32)
    hybrid_model.eval()
    with torch.no_grad():
        by_parts = hybrid_model.forward_parts(
            torch.from_numpy(observations), torch.from_numpy(field)
        )
        by_fused = hybrid_model(torch.from_numpy(fuse_arrays(observations, field)))
    assert torch.equal(by_parts, by_fused)
    # And the numpy fusion agrees with the accepted torch layout.
    assert np.array_equal(
        fuse_arrays(observations, field)[:, 127:],
        field_to_planes(torch.from_numpy(field)).numpy(),
    )


def test_the_model_refuses_an_input_of_the_wrong_width(hybrid_model):
    import torch

    from stratego.belief.phase11b.hybrid_cnn import Phase11BHybridCNNError

    for shape in ((2, 127, 10, 10), (2, 128, 10, 10), (2, 255, 10)):
        with pytest.raises(Phase11BHybridCNNError):
            hybrid_model(torch.zeros(*shape))
    with pytest.raises(Phase11BHybridCNNError):
        hybrid_model.forward_parts(torch.zeros(2, 128, 10, 10), torch.zeros(2, 100, 128))


def test_the_per_square_readout_uses_the_accepted_token_order(hybrid_model):
    """Square `s` of the output is row-major square `s`, as the corpus indexes."""
    import torch

    hybrid_model.eval()
    fused = torch.from_numpy(
        np.random.default_rng(9).random((2, 255, 10, 10)).astype(np.float32)
    )
    with torch.no_grad():
        planes = hybrid_model(fused)
        per_square = hybrid_model.per_square_logits(fused)
    for square in (0, 1, 10, 47, 99):
        row, column = divmod(square, 10)
        assert torch.equal(per_square[:, square], planes[:, :, row, column])


def test_the_gather_visits_each_piece_own_square(dev, hybrid_model):
    """`logits_at` must select each supervised piece's own square."""
    import torch

    from stratego.belief.phase11b.train import _sample_batches

    rows = np.arange(4, dtype=np.int64)
    hybrid_model.eval()
    fused = torch.from_numpy(
        np.random.default_rng(11).random((4, 255, 10, 10)).astype(np.float32)
    )
    for block, token_rows, token_squares, _labels in _sample_batches(dev, rows, 4):
        with torch.no_grad():
            gathered = hybrid_model.logits_at(
                fused, torch.from_numpy(token_rows), torch.from_numpy(token_squares)
            )
            per_square = hybrid_model.per_square_logits(fused)
        for index, (row, square) in enumerate(zip(token_rows, token_squares)):
            assert torch.equal(gathered[index], per_square[int(row), int(square)])
        assert gathered.shape[1] == c11b.RANK_COUNT
        break


def test_the_split_view_swaps_the_input_array_and_nothing_else(dev):
    from stratego.belief.phase11b.hybrid_cnn import FUSED_SHAPE, hybrid_split_view

    fused = np.zeros((int(dev["samples"]), *FUSED_SHAPE), dtype=np.float32)
    view = hybrid_split_view(dev, fused)
    assert view["observations"] is fused
    assert view["model_input"] == "public_observation_and_frozen_c1_field"
    assert view["public_observations"] is dev["observations"]
    for key, value in dev.items():
        if key == "observations":
            continue
        assert view[key] is value, key


def test_no_c1_parameter_is_trainable_in_this_experiment(summary):
    assert summary["training"]["c1_parameters_updated"] == 0
    assert summary["feature_cache"]["gradients_reaching_c1"] is False
    assert summary["leaderboard"][CANDIDATE_4]["retrains_accepted_c1_weights"] is False
    breakdown = summary["pilot"]["parameters"]
    assert breakdown["trainable_c1_parameters"] == 0


def test_the_saved_checkpoint_loads_back_into_the_shape_it_was_saved_with(summary):
    from stratego.belief.phase11b.hybrid_cnn import load_hybrid_cnn, parameter_count

    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 4 checkpoint is not present")
    assert _sha256(path) == summary["checkpoint"]["sha256"]
    model, payload = load_hybrid_cnn(path)
    assert payload["candidate_id"] == CANDIDATE_4
    assert parameter_count(model) == summary["leaderboard"][CANDIDATE_4]["parameters"]
    assert payload["architecture"] == summary["leaderboard"][CANDIDATE_4]["architecture"]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def test_the_interface_exposes_exactly_the_two_required_methods():
    from stratego.belief.phase11b.hybrid_cnn import HybridBeliefModel

    for name in ("predict_marginals", "sample_worlds"):
        assert callable(getattr(HybridBeliefModel, name, None)), name


def test_the_adapter_inherits_the_accepted_sampler_path_rather_than_forking_it():
    from stratego.belief.phase11b.hybrid_cnn import HybridBeliefModel
    from stratego.belief.phase11b.interface import Phase11BBeliefModel

    assert issubclass(HybridBeliefModel, Phase11BBeliefModel)
    # `sample_worlds` is not overridden, so the accepted sampler path is the
    # inherited one and cannot have been quietly replaced.
    assert "sample_worlds" not in vars(HybridBeliefModel)


def test_the_deployed_path_reads_the_same_layer_the_cache_was_built_from():
    from stratego.belief.phase11b import features as feat
    from stratego.belief.phase11b import feature_seam as fs
    from stratego.belief.phase11b.hybrid_cnn import HybridBeliefCNN

    assert HybridBeliefCNN.feature_layer == feat.LAYER_FINAL
    assert HybridBeliefCNN.seam_id == fs.SEAM_ID


def test_the_interface_block_records_a_run_through_the_accepted_sampler(summary):
    interface = summary["interface"]
    assert interface["positions_checked"] > 0
    assert interface["worlds_sampled"] > 0
    assert interface["all_marginals_sum_to_one"] is True
    assert interface["sample_worlds_seed_deterministic"] is True
    assert interface["all_worlds_passed_accepted_validation_stack"] is True
    assert interface["sampler_source"].endswith("(accepted, unmodified)")
    assert interface["describe"]["reads_hidden_truth"] is False
    assert interface["describe"]["consumes_c1_features"] is True
    assert interface["describe"]["consumes_public_observation"] is True
    assert interface["describe"]["c1_frozen"] is True


def test_the_marginals_of_the_trained_model_are_probability_vectors(summary, dev):
    """The deployed path, on real corpus observations."""
    import torch

    from stratego.belief.phase11b.hybrid_cnn import load_hybrid_cnn

    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 4 checkpoint is not present")
    from stratego.belief.phase11b import features as feat

    model, _payload = load_hybrid_cnn(path)
    encoder, _identity = feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device="cpu"
    )
    observations = np.asarray(dev["observations"])[:4].astype(np.float32)
    field = feat.encode_batch(encoder, observations, feat.LAYER_FINAL)
    with torch.no_grad():
        logits = model.per_square_logits_from_parts(torch.from_numpy(observations), field)
        probabilities = torch.softmax(logits.to(torch.float64), dim=2).numpy()
    assert probabilities.shape == (4, c11b.NUM_SQUARES, c11b.RANK_COUNT)
    assert np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-9)
    assert np.isfinite(probabilities).all()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_the_leaderboard_row_has_every_shared_metric_the_sprint_requires(summary):
    row = summary["leaderboard"][CANDIDATE_4]
    for field in (
        "ce",
        "baseline_ce",
        "r_ce",
        "top1",
        "r_ce_by_stratum",
        "parameters",
        "training_seconds",
        "time_to_best_seconds",
        "inference_microseconds_per_piece",
    ):
        assert row.get(field) is not None, field
    for stratum in c11b.CORPUS_STRATA:
        assert stratum in row["r_ce_by_stratum"], stratum


def test_the_reported_r_ce_is_the_reported_ce_over_the_reported_baseline(summary):
    row = summary["leaderboard"][CANDIDATE_4]
    assert row["r_ce"] == pytest.approx(row["ce"] / row["baseline_ce"], rel=1e-12)


def test_the_denominator_is_the_corpus_own_remaining_count_baseline(summary, dev):
    baseline = M.baseline_probabilities(dev)
    recomputed = float(
        M.cross_entropy(baseline, np.asarray(dev["true_rank"], dtype=np.int64)).mean()
    )
    assert recomputed == pytest.approx(
        summary["leaderboard"][CANDIDATE_4]["baseline_ce"], rel=1e-12
    )


def test_every_candidate_on_the_board_divides_by_that_same_denominator(summary):
    row = summary["leaderboard"][CANDIDATE_4]
    for name, block in summary["earlier_reference_rows"].items():
        assert block["baseline_ce"] == pytest.approx(row["baseline_ce"], rel=1e-12), name


def test_the_reported_r_ce_recomputes_from_the_checkpoint_and_the_corpus(summary, dev):
    """The headline number, re-derived end to end from what is on disk."""
    import torch

    from stratego.belief.phase11b.hybrid_cnn import (
        hybrid_split_view,
        load_fused_cache,
        load_hybrid_cnn,
    )
    from stratego.belief.phase11b.raw_train import (
        predict_probabilities_raw,
        stage_observations,
    )

    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    cache = REPOSITORY_ROOT / summary["feature_cache"]["fused_input"]["dev"]["path"]
    if not (path.exists() and cache.exists()):
        pytest.skip("the Agent 4 checkpoint or fused dev cache is not present")
    model, _payload = load_hybrid_cnn(path)
    fused = load_fused_cache(cache, expected_samples=int(dev["samples"]))
    staged = stage_observations(hybrid_split_view(dev, fused), "cpu", on_device=False)
    with torch.no_grad():
        probabilities = predict_probabilities_raw(model, staged, dev, device="cpu")
    metrics = M.evaluate(probabilities, dev, bootstrap_resamples=50)
    row = summary["leaderboard"][CANDIDATE_4]
    assert metrics["r_ce"] == pytest.approx(row["r_ce"], abs=1e-9)
    assert metrics["top1"] == pytest.approx(row["top1"], abs=1e-9)


def test_the_earlier_rows_are_quoted_verbatim_not_recomputed(summary, agent1, agent2, agent3):
    combined = {
        **agent1["leaderboard"],
        **agent2["leaderboard"],
        **agent3["leaderboard"],
    }
    for name, block in summary["earlier_reference_rows"].items():
        assert block == combined[name], name
    assert summary["experiment"]["prior_candidates_rerun"] is False


def test_the_earlier_checkpoints_reproduce_the_numbers_their_agents_reported(summary):
    reproduced = summary["earlier_reproduction"]
    # The rows the complementarity reading turns on must both be there.
    for required in (CANDIDATE_2, CANDIDATE_3):
        assert required in reproduced, required
    for name, block in reproduced.items():
        reported = block["r_ce_reported_by_its_agent"]
        if reported is None:
            continue
        assert block["absolute_difference"] < 1.0e-3, name
        assert block["source"].startswith("loaded read-only")


def test_the_two_backends_agree_on_the_reported_number(summary):
    agreement = summary["backend_agreement"]
    assert agreement["scoring_backend"] == "cpu"
    assert agreement["absolute_difference"] < 1.0e-6
    assert summary["leaderboard"][CANDIDATE_4]["r_ce"] == pytest.approx(
        agreement["r_ce_cpu"], rel=1e-12
    )


def test_one_architecture_and_one_configuration_were_declared(summary):
    experiment = summary["experiment"]
    assert experiment["architectures_trained"] == 1
    assert experiment["optimization_configurations_declared"] == 1
    for forbidden in (
        "architecture_sweep",
        "branch_width_sweep",
        "fusion_method_sweep",
        "depth_sweep",
        "learning_rate_sweep",
        "hyperparameter_sweep",
    ):
        assert experiment[forbidden] is False, forbidden
    assert summary["training"]["architectures_trained"] == 1
    assert summary["training"]["configurations_declared"] == 1


def test_the_kept_checkpoint_is_the_best_probe_of_its_whole_run(summary, curve):
    rows = curve["curve"]
    best = min(rows, key=lambda row: row["dev_ce"])
    assert best["step"] == summary["training"]["best_step"]
    assert summary["training"]["overfitting"]["dev_ce_best"] == pytest.approx(
        best["dev_ce"], rel=1e-12
    )


def test_the_repeat_run_is_a_diagnostic_and_never_the_reported_candidate(summary):
    repeat = summary["repeat_run"]
    assert repeat["is_the_reported_candidate"] is False
    assert repeat["checkpoint_written"] is False
    assert repeat["identical_seed"] is True
    assert all(repeat["identical_config"].values())
    assert repeat["reported_r_ce"] == pytest.approx(
        summary["training"]["overfitting"]["dev_ce_best"]
        / summary["leaderboard"][CANDIDATE_4]["baseline_ce"],
        rel=1e-9,
    )


def test_the_repeat_bounds_the_run_to_run_spread(summary):
    """A gap the leaderboard turns on must be larger than this candidate's noise."""
    repeat = summary["repeat_run"]
    spread = repeat["absolute_r_ce_difference"]
    row = summary["leaderboard"][CANDIDATE_4]
    against_leader = abs(row["r_ce"] - summary["decision"]["leader_r_ce"])
    assert spread < against_leader


def test_the_supervised_loss_used_no_policy_value_or_outcome_term(summary):
    training = summary["training"]
    assert training["policy_or_value_terms"] is False
    assert training["game_outcome_used"] is False
    assert training["loss"].startswith("supervised hidden-rank cross-entropy")


def test_the_complementarity_block_is_arithmetic_on_reported_numbers(summary, agent2, agent3):
    block = summary["complementarity"]
    assert block["agent2_raw_only_r_ce"] == agent2["leaderboard"][CANDIDATE_2]["r_ce"]
    assert block["agent3_c1_only_r_ce"] == agent3["leaderboard"][CANDIDATE_3]["r_ce"]
    assert block["agent4_hybrid_r_ce"] == summary["leaderboard"][CANDIDATE_4]["r_ce"]
    assert block["hybrid_minus_agent2_raw_only"] == pytest.approx(
        block["agent4_hybrid_r_ce"] - block["agent2_raw_only_r_ce"], rel=1e-12
    )
    assert block["hybrid_minus_agent3_c1_only"] == pytest.approx(
        block["agent4_hybrid_r_ce"] - block["agent3_c1_only_r_ce"], rel=1e-12
    )
    assert block["agent2_rerun"] is False
    assert block["agent3_rerun"] is False


def test_the_complementarity_reference_is_the_better_single_source(summary):
    """Beating only the weaker branch would show nothing."""
    block = summary["complementarity"]
    better = min(block["agent2_raw_only_r_ce"], block["agent3_c1_only_r_ce"])
    assert block["better_single_source_r_ce"] == pytest.approx(better, rel=1e-12)
    assert block["hybrid_minus_better_single_source"] == pytest.approx(
        block["agent4_hybrid_r_ce"] - better, rel=1e-12
    )


def test_the_complementarity_verdict_follows_the_declared_rule(summary):
    block = summary["complementarity"]
    delta = block["hybrid_minus_better_single_source"]
    band = block["equivalence_band"]
    assert block["complementary"] is bool(delta < -band)
    if block["complementary"]:
        assert block["interpretation"].startswith("complementary")
    elif abs(delta) <= band:
        assert block["interpretation"].startswith("not complementary")
    else:
        assert block["interpretation"].startswith("the fusion is worse")


def test_the_required_table_is_present_in_the_instructions_own_order(summary):
    labels = [entry["label"] for entry in summary["comparison_table"]["rows"]]
    assert labels == [
        "old Phase 11 head",
        "Agent 1 best attached head",
        "Agent 2 raw CNN",
        "Agent 3 C1-feature CNN",
        "Agent 4 hybrid",
    ]
    assert summary["comparison_table"]["prior_candidates_rerun"] is False
    for entry in summary["comparison_table"]["rows"]:
        assert entry["rerun_by_agent4"] is False, entry["candidate_id"]
        assert entry["r_ce"] is not None, entry["candidate_id"]


def test_the_table_quotes_agent1s_actual_best_row(summary, agent1):
    table = summary["comparison_table"]
    best = min(agent1["leaderboard"], key=lambda name: agent1["leaderboard"][name]["r_ce"])
    assert table["agent1_best_resolved_as"] == best
    quoted = next(
        entry for entry in table["rows"] if entry["label"] == "Agent 1 best attached head"
    )
    assert quoted["candidate_id"] == best
    assert quoted["r_ce"] == agent1["leaderboard"][best]["r_ce"]


def test_the_verdict_follows_the_stated_engineering_rule(summary):
    decision = summary["decision"]
    row = summary["leaderboard"][CANDIDATE_4]
    everyone = {
        **{name: block["r_ce"] for name, block in summary["earlier_reference_rows"].items()},
        CANDIDATE_4: row["r_ce"],
    }
    leader = min(everyone, key=lambda name: everyone[name])
    assert decision["leader_by_r_ce"] == leader
    assert decision["leader_r_ce"] == pytest.approx(everyone[leader], rel=1e-12)
    assert decision["agent4_is_the_leader"] is (leader == CANDIDATE_4)
    delta = decision["agent4_minus_best_earlier_r_ce"]
    assert decision["agent4_materially_better_than_best_earlier"] is bool(
        delta < -decision["equivalence_band"]
    )


def test_the_paired_comparisons_agree_with_the_leaderboard_ordering(summary):
    """A distinguishable paired difference must point the way the board does."""
    row = summary["leaderboard"][CANDIDATE_4]
    for name, block in summary["paired_comparisons"].items():
        other = name.split(" vs ")[1]
        earlier = summary["earlier_reference_rows"].get(other)
        if earlier is None or not block["distinguishable"]:
            continue
        assert block["left_lower_ce"] is bool(row["r_ce"] < earlier["r_ce"]), name


def test_the_report_says_what_agent_4_is_not_claiming():
    path = REPORT_DIRECTORY / "agent_04_report.md"
    if not path.exists():
        pytest.skip("the Agent 4 report has not been produced yet")
    # The report is hard-wrapped, so a claim can straddle a newline: compare
    # against the whitespace-collapsed text rather than the raw file.
    text = " ".join(path.read_text().split())
    for phrase in (
        "engineering prototype",
        "does not repair Phase 11",
        "does not overturn the Phase 11 `FAIL`",
        "does not authorize Phase 12",
        "`phase11_test_bank_v1` was not opened; it remains spent",
    ):
        assert phrase in text, phrase


def test_the_report_records_the_stop_condition_and_does_not_start_agent_5(summary):
    stop = summary["stop_condition"]
    assert "was not begun" in stop
    assert "FAIL" in stop
    assert "spent" in stop
    assert summary["phase12_authorized_by_this_artifact"] is False
