"""Phase 11B Agent 3: the C1-feature CNN, its seam, its results, its report.

These tests protect what Agent 3 measured and, just as importantly, what
Agent 3 promised **not** to do. Five groups:

- **Boundary** — the Phase 11B status markers survive on every artifact, the
  common corpus was reused byte-for-byte, no Agent 1, Agent 2 or Phase 11
  artifact moved, and no Agent 3 module reaches the spent test bank.
- **Seam** — the recorded seam really is the tensor `ProductionModel.encode`
  returns, it is per-square and unpooled, it maps back to board cells in the
  accepted token order, and the cached field re-derives from the public
  observations under the accepted frozen weights.
- **Model** — the parameter count is inside the instructed band, the only
  input is the frozen field, no raw observation path exists, C1 carries no
  gradient, and the read-out uses the accepted token order.
- **Interface** — the required two methods exist, produce probability
  vectors, and reach worlds only through the accepted Phase 11 sampler.
- **Results** — every leaderboard number recomputes from the corpus, the
  Agent 2 contrast the instruction asks for is arithmetic on two reported
  numbers, and the verdict follows the stated rule.

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
CANDIDATE_3 = "agent03_c1_feature_cnn"
CANDIDATE_2 = "agent02_raw_observation_cnn"
CANDIDATE_1B = "agent01_1b_attached_mlp_head"

AGENT3_SOURCES = (
    "stratego/belief/phase11b/feature_seam.py",
    "stratego/belief/phase11b/feature_cnn.py",
    "scripts/run_phase11b_agent03.py",
    "scripts/_phase11b_agent03_report.py",
)

#: Names that would mean an Agent 3 module had reached the spent Phase 11
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
    return _load("agent_03_summary.json")


@pytest.fixture(scope="module")
def curve():
    return _load("agent_03_learning_curve.json")


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


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Boundary: what Agent 3 promised not to do
# ---------------------------------------------------------------------------


def test_the_summary_carries_every_status_marker_unchanged(summary):
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert summary[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert summary[key] == value, key
    assert summary["agent"] == 3


def test_the_curve_and_the_checkpoint_carry_the_markers_too(summary, curve):
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert curve[key] == value, key
    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 3 checkpoint is not present")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert payload[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert payload[key] == value, key


def test_the_common_corpus_was_reused_not_regenerated(summary, agent1, agent2):
    corpus = summary["common_corpus"]
    assert corpus["corpus_digest"] == agent1["common_corpus"]["corpus_digest"]
    assert corpus["corpus_digest"] == agent2["common_corpus"]["corpus_digest"]
    assert corpus["corpus_digest_matches_manifest"] is True
    assert corpus["file_digest_drift"] == []
    assert summary["preservation"]["corpus_regenerated"] is False
    for split in c11b.CORPUS_SPLITS:
        assert corpus["file_digests"][split] == agent1["common_corpus"]["file_digests"][split]


def test_the_corpus_bytes_still_hash_to_what_agent3_scored_on(summary, dev):
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
    assert drifted == [], f"Agent 3 must leave these untouched: {drifted}"
    preservation = summary["preservation"]
    assert preservation["agent1_artifacts_modified"] is False
    assert preservation["agent2_artifacts_modified"] is False
    assert preservation["c1_modified"] is False
    assert preservation["artifacts_unchanged_since_agent2"] is True
    assert preservation["phase11_test_bank_opened"] is False


def test_the_preserved_set_actually_covers_agent1_and_agent2s_modules(summary):
    """A preservation claim is only as good as the list behind it."""
    preserved = set(summary["preserved_artifact_digests"])
    for relative in (
        "stratego/belief/phase11b/features.py",
        "stratego/belief/phase11b/heads.py",
        "stratego/belief/phase11b/interface.py",
        "stratego/belief/phase11b/metrics.py",
        "stratego/belief/phase11b/raw_cnn.py",
        "stratego/belief/phase11b/raw_train.py",
        "stratego/model/production_model.py",
        "reports/phase11b/agent_01_summary.json",
        "reports/phase11b/agent_02_summary.json",
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


def test_no_agent3_module_reaches_the_spent_test_bank():
    offenders = []
    for relative in AGENT3_SOURCES:
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            continue
        tokens = _executable_tokens(path)
        for token in FORBIDDEN_CODE_TOKENS:
            if token in tokens:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], f"Agent 3 must not reach the spent bank: {offenders}"


# ---------------------------------------------------------------------------
# Seam: which tensor, and is the cache really a function of public inputs
# ---------------------------------------------------------------------------


def test_the_recorded_seam_is_per_square_and_not_pooled():
    from stratego.belief.phase11b import feature_seam as fs

    description = fs.SEAM_DESCRIPTION
    assert description["tensor"] == "ProductionModel.encode(tokens)"
    assert description["is_per_square"] is True
    assert description["is_pooled"] is False
    assert description["shape"][1:] == [c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH]
    assert description["c1_frozen"] is True
    assert description["gradients_reaching_c1"] is False
    # The instruction's own warning: not a pooled or compressed global vector.
    rejected = description["alternatives_rejected"]
    assert "hidden.mean(dim=1)" in rejected
    assert "belief_output(hidden)" in rejected


def test_the_summary_records_the_same_seam_the_module_declares(summary):
    from stratego.belief.phase11b import feature_seam as fs

    assert summary["frozen_seam"] == fs.SEAM_DESCRIPTION
    assert summary["leaderboard"][CANDIDATE_3]["seam_id"] == fs.SEAM_ID


def test_the_seam_call_returns_exactly_what_encode_returns():
    """`encode_batch(..., 'final')` is `encode`, not a re-implementation."""
    import torch

    from stratego.belief.phase11b import features as feat
    from stratego.model.architecture_configs import candidate_config
    from stratego.model.production_model import build_candidate_model
    from stratego.model.tokenization import observation_to_tokens

    model = build_candidate_model(candidate_config("C1"), seed=11)
    model.eval()
    observations = np.random.default_rng(3).normal(size=(2, *c11b.OBSERVATION_SHAPE))
    observations = observations.astype(np.float32)
    with torch.no_grad():
        direct = model.encode(observation_to_tokens(torch.from_numpy(observations)))
        through_seam = feat.encode_batch(model, observations, feat.LAYER_FINAL)
    assert tuple(direct.shape) == (2, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)
    assert torch.equal(direct, through_seam)


def test_the_seam_is_what_the_task_heads_read():
    """All three heads consume `encode`'s output; that is why it is the seam."""
    import inspect

    from stratego.model.production_model import ProductionModel

    source = inspect.getsource(ProductionModel.forward)
    assert "hidden = self.encode(tokens)" in source
    assert "self.belief_output(hidden)" in source


def test_the_field_maps_back_to_board_cells_in_the_accepted_order():
    import torch

    from stratego.belief.phase11b.feature_seam import field_to_planes

    field = torch.arange(
        2 * c11b.NUM_SQUARES * c11b.C1_FEATURE_WIDTH, dtype=torch.float32
    ).reshape(2, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)
    planes = field_to_planes(field)
    assert tuple(planes.shape) == (2, c11b.C1_FEATURE_WIDTH, 10, 10)
    for square in (0, 1, 9, 10, 41, 99):
        assert torch.equal(planes[1, :, square // 10, square % 10], field[1, square])
    # And contiguous, which the convolution and batch-norm backward kernels
    # on MPS require of a strided reshape.
    assert planes.is_contiguous()


def test_the_field_layout_is_the_accepted_tokenization_inverse():
    """The same operation `tokens_to_observation` performs, at width 128."""
    import torch

    from stratego.belief.phase11b.feature_seam import field_to_planes
    from stratego.model.tokenization import observation_to_tokens, tokens_to_observation

    observation = torch.randn(2, *c11b.OBSERVATION_SHAPE)
    tokens = observation_to_tokens(observation)
    assert torch.equal(tokens_to_observation(tokens), observation)
    # field_to_planes is that inverse with 128 channels instead of 127.
    field = torch.randn(2, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)
    expected = field.transpose(1, 2).reshape(2, c11b.C1_FEATURE_WIDTH, 10, 10)
    assert torch.equal(field_to_planes(field), expected)


def test_the_cache_metadata_says_it_holds_no_labels(summary):
    caches = summary["feature_cache"]["caches"]
    assert set(caches) == set(c11b.CORPUS_SPLITS)
    for split, block in caches.items():
        assert block["contains_labels"] is False
        assert block["derived_from"] == "public observations + accepted frozen C1"
        assert block["shape"][1:] == [c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH]
        assert block["shape"][0] == summary["common_corpus"]["splits"][split]["samples"]
    assert summary["feature_cache"]["gradients_reaching_c1"] is False


def test_the_harness_recorded_a_bit_identical_re_derivation(summary):
    for split, block in summary["feature_cache"]["verification"].items():
        assert block["bit_identical"] is True, split
        assert block["max_absolute_difference"] == 0.0, split
        assert block["rows_checked"] > 0, split
        assert block["inputs"] == ["public observation", "accepted frozen C1 weights"]


def test_the_cache_on_disk_re_derives_from_the_public_observations(summary, dev):
    """The cache is only legitimate if it is a function of nothing else."""
    from stratego.belief.phase11b import feature_seam as fs
    from stratego.belief.phase11b.features import load_frozen_c1

    path = REPOSITORY_ROOT / summary["feature_cache"]["caches"]["dev"]["path"]
    if not path.exists():
        pytest.skip("the Agent 3 dev field cache is not present")
    export = CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt"
    if not export.exists():
        pytest.skip("the read-only Phase 9 export is not present")
    model, _identity = load_frozen_c1(REPOSITORY_ROOT, export, device="cpu")
    cache = fs.load_field_cache(path, expected_samples=int(dev["samples"]))
    checked = fs.verify_field_cache(model, dev, cache, rows=16, seed=7)
    assert checked["bit_identical"] is True
    assert checked["max_absolute_difference"] == 0.0


# ---------------------------------------------------------------------------
# Model: capacity band, the input boundary, and the frozen encoder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_model():
    from stratego.belief.phase11b.feature_cnn import build_feature_cnn

    return build_feature_cnn(seed=20260819)


def test_the_parameter_count_is_inside_the_instructed_band(feature_model):
    from stratego.belief.phase11b.feature_cnn import parameter_breakdown

    breakdown = parameter_breakdown(feature_model)
    assert 3_000_000 <= breakdown["total"] <= 5_000_000
    assert breakdown["total"] == (
        breakdown["stem"] + breakdown["residual_tower"] + breakdown["readout"]
    )
    assert breakdown["input_channels"] == c11b.C1_FEATURE_WIDTH
    assert breakdown["trainable_c1_parameters"] == 0


def test_the_reported_parameter_count_is_the_model_that_was_trained(summary, feature_model):
    from stratego.belief.phase11b.feature_cnn import parameter_count

    row = summary["leaderboard"][CANDIDATE_3]
    assert row["parameters"] == parameter_count(feature_model)
    assert row["parameters_trained"] == row["parameters"]
    assert summary["training"]["c1_parameters_updated"] == 0


def test_the_tower_is_agent2s_so_the_comparison_is_representation_only(summary, agent2):
    """The two candidates must differ in input, not in capacity."""
    pilot = summary["pilot"]
    agent2_row = agent2["leaderboard"][CANDIDATE_2]
    assert pilot["agent2_parameters"] == agent2_row["parameters"]
    breakdown = pilot["parameters"]
    agent2_breakdown = agent2["pilot"]["parameters"]
    for key in ("width", "blocks", "readout_width"):
        assert breakdown[key] == agent2_breakdown[key], key
    assert breakdown["residual_tower"] == agent2_breakdown["residual_tower"]
    assert breakdown["readout"] == agent2_breakdown["readout"]
    # The only difference is the stem's input channels: 128 against 127.
    assert pilot["parameters_minus_agent2"] == breakdown["stem"] - agent2_breakdown["stem"]
    assert pilot["parameters_minus_agent2"] == 160 * 9


def test_the_block_class_is_imported_from_agent2_not_redeclared():
    from stratego.belief.phase11b import feature_cnn, raw_cnn

    assert feature_cnn.ResidualBlock is raw_cnn.ResidualBlock


def test_the_model_takes_the_frozen_field_and_nothing_else(feature_model):
    import inspect

    signature = inspect.signature(feature_model.forward)
    assert list(signature.parameters) == ["field"]
    # No raw-observation path: feeding the observation in is Agent 4's
    # experiment, and the specialist has nowhere to put it.
    assert _no_raw_observation_path(feature_model)


def _no_raw_observation_path(model) -> bool:
    """No submodule of the specialist accepts the 127-channel observation."""
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 127:
            return False
    return True


def test_the_model_refuses_anything_that_is_not_a_100x128_field(feature_model):
    import torch

    from stratego.belief.phase11b.feature_cnn import Phase11BFeatureCNNError

    for bad in (
        torch.zeros(2, *c11b.OBSERVATION_SHAPE),
        torch.zeros(2, c11b.NUM_SQUARES, 127),
        torch.zeros(c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH),
    ):
        with pytest.raises(Phase11BFeatureCNNError):
            feature_model(bad)


def test_the_per_square_readout_uses_the_accepted_token_order(feature_model):
    import torch

    feature_model.eval()
    field = torch.randn(3, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)
    with torch.no_grad():
        planes = feature_model(field)
        per_square = feature_model.per_square_logits(field)
    assert tuple(per_square.shape) == (3, c11b.NUM_SQUARES, c11b.RANK_COUNT)
    for square in (0, 7, 41, 99):
        assert torch.allclose(per_square[2, square], planes[2, :, square // 10, square % 10])


def test_belief_logits_is_the_per_square_field_for_one_position(feature_model):
    import torch

    feature_model.eval()
    field = torch.randn(1, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)
    with torch.no_grad():
        assert torch.allclose(
            feature_model.belief_logits(field[0]), feature_model.per_square_logits(field)[0]
        )


def test_the_gather_visits_each_piece_own_square(dev, feature_model):
    import torch

    from stratego.belief.phase11b.train import _sample_batches

    rows = np.arange(4, dtype=np.int64)
    block, token_rows, token_squares, labels = next(_sample_batches(dev, rows, 4))
    offsets = np.asarray(dev["piece_offset"], dtype=np.int64)
    squares = np.asarray(dev["perspective_square"], dtype=np.int64)
    expected = np.concatenate(
        [squares[offsets[row] : offsets[row + 1]] for row in block]
    )
    assert np.array_equal(token_squares, expected)
    assert labels.shape == token_squares.shape
    field = torch.randn(4, c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH)
    feature_model.eval()
    with torch.no_grad():
        gathered = feature_model.logits_at(
            field, torch.from_numpy(token_rows), torch.from_numpy(token_squares)
        )
        full = feature_model.per_square_logits(field)
    for index in range(min(8, gathered.shape[0])):
        assert torch.allclose(
            gathered[index], full[int(token_rows[index]), int(token_squares[index])]
        )


def test_the_split_view_swaps_the_input_array_and_nothing_else(dev):
    """Agent 3 reuses Agent 2's trainer by giving it a different input."""
    from stratego.belief.phase11b.feature_cnn import feature_split_view

    field = np.zeros(
        (int(dev["samples"]), c11b.NUM_SQUARES, c11b.C1_FEATURE_WIDTH), dtype=np.float32
    )
    view = feature_split_view(dev, field)
    assert view["observations"] is field
    assert view["public_observations"] is dev["observations"]
    for key in ("true_rank", "piece_offset", "perspective_square", "stratum", "game_ordinal"):
        assert view[key] is dev[key], key


def test_the_saved_checkpoint_loads_back_into_the_shape_it_was_saved_with(summary):
    from stratego.belief.phase11b.feature_cnn import load_feature_cnn, parameter_count

    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 3 checkpoint is not present")
    model, payload = load_feature_cnn(path)
    assert parameter_count(model) == payload["parameters"]
    assert payload["candidate_id"] == CANDIDATE_3
    assert payload["frozen_c1"]["retrained"] is False
    assert payload["corpus_digest"] == summary["common_corpus"]["corpus_digest"]
    assert _sha256(path) == summary["checkpoint"]["sha256"]


# ---------------------------------------------------------------------------
# Interface: the two required methods, through the accepted sampler
# ---------------------------------------------------------------------------


def test_the_interface_exposes_exactly_the_two_required_methods():
    from stratego.belief.phase11b.feature_cnn import C1FeatureBeliefModel

    for name in ("predict_marginals", "sample_worlds"):
        assert callable(getattr(C1FeatureBeliefModel, name))


def test_the_adapter_inherits_the_accepted_sampler_path_rather_than_forking_it():
    from stratego.belief.phase11b.feature_cnn import C1FeatureBeliefModel
    from stratego.belief.phase11b.interface import Phase11BBeliefModel

    assert issubclass(C1FeatureBeliefModel, Phase11BBeliefModel)
    # `sample_worlds` and `predict_marginals` are Agent 1's code, not a copy.
    assert C1FeatureBeliefModel.sample_worlds is Phase11BBeliefModel.sample_worlds
    assert C1FeatureBeliefModel.predict_marginals is Phase11BBeliefModel.predict_marginals


def test_the_deployed_path_reads_the_same_layer_the_cache_was_built_from():
    from stratego.belief.phase11b import features as feat
    from stratego.belief.phase11b.feature_cnn import C1FeatureBeliefCNN

    assert C1FeatureBeliefCNN.feature_layer == feat.LAYER_FINAL


def test_the_interface_block_records_a_run_through_the_accepted_sampler(summary):
    interface = summary["interface"]
    assert interface["candidate_id"] == CANDIDATE_3
    assert interface["positions_checked"] > 0
    assert interface["worlds_sampled"] > 0
    assert interface["all_marginals_sum_to_one"] is True
    assert interface["sample_worlds_seed_deterministic"] is True
    assert interface["all_worlds_passed_accepted_validation_stack"] is True
    assert interface["sampler_source"].startswith("stratego.evaluation.phase11_sampler")
    assert interface["describe"]["reads_hidden_truth"] is False
    assert interface["describe"]["consumes_c1_features"] is True
    assert interface["describe"]["consumes_public_observation"] is False
    assert interface["describe"]["c1_frozen"] is True


def test_the_marginals_of_the_trained_model_are_probability_vectors(summary, dev):
    """The trained checkpoint, on real development fields."""
    import torch

    from stratego.belief.phase11b import feature_seam as fs
    from stratego.belief.phase11b.feature_cnn import load_feature_cnn

    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    cache = REPOSITORY_ROOT / summary["feature_cache"]["caches"]["dev"]["path"]
    if not (path.exists() and cache.exists()):
        pytest.skip("the Agent 3 checkpoint or dev cache is not present")
    model, _payload = load_feature_cnn(path)
    field = fs.load_field_cache(cache, expected_samples=int(dev["samples"]))
    batch = torch.from_numpy(np.array(field[:8], dtype=np.float32, copy=True))
    with torch.no_grad():
        probabilities = torch.softmax(
            model.per_square_logits(batch).to(torch.float64), dim=-1
        ).numpy()
    assert probabilities.shape == (8, c11b.NUM_SQUARES, c11b.RANK_COUNT)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.isfinite(probabilities).all()


# ---------------------------------------------------------------------------
# Results: the numbers, and the rule that read them
# ---------------------------------------------------------------------------


def test_the_leaderboard_row_has_every_shared_metric_the_sprint_requires(summary):
    row = summary["leaderboard"][CANDIDATE_3]
    for key in (
        "ce",
        "baseline_ce",
        "r_ce",
        "top1",
        "parameters",
        "training_seconds",
        "time_to_best_seconds",
        "inference_microseconds_per_piece",
    ):
        assert row[key] is not None, key
    for stratum in c11b.CORPUS_STRATA:
        assert stratum in row["r_ce_by_stratum"], stratum


def test_the_reported_r_ce_is_the_reported_ce_over_the_reported_baseline(summary):
    row = summary["leaderboard"][CANDIDATE_3]
    assert row["r_ce"] == pytest.approx(row["ce"] / row["baseline_ce"], rel=1e-12)


def test_the_denominator_is_the_corpus_own_remaining_count_baseline(summary, dev):
    baseline = M.baseline_probabilities(dev)
    recomputed = float(
        M.cross_entropy(baseline, np.asarray(dev["true_rank"], dtype=np.int64)).mean()
    )
    assert summary["leaderboard"][CANDIDATE_3]["baseline_ce"] == pytest.approx(
        recomputed, rel=1e-12
    )


def test_every_candidate_on_the_board_divides_by_that_same_denominator(summary):
    row = summary["leaderboard"][CANDIDATE_3]
    for name, block in summary["earlier_reference_rows"].items():
        assert block["baseline_ce"] == pytest.approx(row["baseline_ce"], rel=1e-12), name


def test_the_reported_r_ce_recomputes_from_the_checkpoint_and_the_corpus(summary, dev):
    """The headline number, re-derived end to end from what is on disk."""
    import torch

    from stratego.belief.phase11b import feature_seam as fs
    from stratego.belief.phase11b.feature_cnn import feature_split_view, load_feature_cnn
    from stratego.belief.phase11b.raw_train import (
        predict_probabilities_raw,
        stage_observations,
    )

    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    cache = REPOSITORY_ROOT / summary["feature_cache"]["caches"]["dev"]["path"]
    if not (path.exists() and cache.exists()):
        pytest.skip("the Agent 3 checkpoint or dev cache is not present")
    model, _payload = load_feature_cnn(path)
    field = fs.load_field_cache(cache, expected_samples=int(dev["samples"]))
    staged = stage_observations(feature_split_view(dev, field), "cpu", on_device=False)
    with torch.no_grad():
        probabilities = predict_probabilities_raw(model, staged, dev, device="cpu")
    metrics = M.evaluate(probabilities, dev, bootstrap_resamples=50)
    assert metrics["r_ce"] == pytest.approx(
        summary["leaderboard"][CANDIDATE_3]["r_ce"], abs=1e-9
    )
    assert metrics["top1"] == pytest.approx(
        summary["leaderboard"][CANDIDATE_3]["top1"], abs=1e-9
    )


def test_the_earlier_rows_are_quoted_verbatim_not_recomputed(summary, agent1, agent2):
    combined = {**agent1["leaderboard"], **agent2["leaderboard"]}
    for name, block in summary["earlier_reference_rows"].items():
        assert block == combined[name], name
    assert summary["experiment"]["prior_candidates_rerun"] is False


def test_the_earlier_checkpoints_reproduce_the_numbers_their_agents_reported(summary):
    reproduced = summary["earlier_reproduction"]
    # The two rows the comparison actually turns on must both be there.
    for required in (CANDIDATE_2, CANDIDATE_1B):
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
    assert summary["leaderboard"][CANDIDATE_3]["r_ce"] == pytest.approx(
        agreement["r_ce_cpu"], rel=1e-12
    )


def test_one_architecture_and_one_configuration_were_declared(summary):
    experiment = summary["experiment"]
    assert experiment["architectures_trained"] == 1
    assert experiment["architecture_sweep"] is False
    assert experiment["hyperparameter_sweep"] is False
    assert experiment["optimization_configurations_declared"] == 1
    assert summary["training"]["configurations_declared"] == 1


def test_the_kept_checkpoint_is_the_best_probe_of_its_whole_run(summary, curve):
    rows = curve["curve"]
    best = min(rows, key=lambda row: row["dev_ce"])
    assert best["step"] == summary["training"]["best_step"]
    assert best["dev_ce"] == pytest.approx(
        summary["training"]["overfitting"]["dev_ce_best"], rel=1e-12
    )
    # The best probe is the one the reported metrics came from.
    assert summary["leaderboard"][CANDIDATE_3]["ce"] == pytest.approx(
        best["dev_ce"], abs=1e-6
    )


def test_the_sub_epoch_probe_ran_at_the_declared_cadence(summary, curve):
    """Agent 2's warning, inherited: an epoch-granular probe misses the optimum."""
    rows = curve["curve"]
    declared = int(curve["config"]["evaluations_per_epoch"])
    epochs = sorted({row["epoch"] for row in rows})
    for epoch in epochs[:-1]:
        probes = [row for row in rows if row["epoch"] == epoch]
        assert len(probes) >= declared, epoch
    epoch_only = min(row["dev_ce"] for row in rows if not row["sub_epoch"])
    assert epoch_only >= summary["training"]["overfitting"]["dev_ce_best"]


def test_the_repeat_run_is_a_diagnostic_and_never_the_reported_candidate(summary):
    repeat = summary.get("repeat_run")
    if repeat is None:
        pytest.skip("the repeat diagnostic was not run")
    assert repeat["is_the_reported_candidate"] is False
    assert repeat["checkpoint_written"] is False
    assert repeat["identical_seed"] is True
    assert all(repeat["identical_config"].values()), repeat["identical_config"]
    # The reported row is the training run's, not the repeat's.
    assert repeat["reported_r_ce"] == pytest.approx(
        summary["training"]["overfitting"]["dev_ce_best"]
        / summary["leaderboard"][CANDIDATE_3]["baseline_ce"],
        rel=1e-9,
    )


def test_the_repeat_bounds_the_run_to_run_spread(summary):
    repeat = summary.get("repeat_run")
    if repeat is None:
        pytest.skip("the repeat diagnostic was not run")
    spread = repeat["absolute_r_ce_difference"]
    assert spread == pytest.approx(
        abs(repeat["reported_r_ce"] - repeat["repeated_r_ce"]), rel=1e-12
    )
    # Two orders of magnitude below the smallest gap the leaderboard turns on.
    contrast = summary["agent2_contrast"]
    assert spread < abs(contrast["difference_agent3_minus_agent2"]) / 100


def test_the_supervised_loss_used_no_policy_value_or_outcome_term(summary):
    training = summary["training"]
    assert training["policy_or_value_terms"] is False
    assert training["game_outcome_used"] is False
    assert "cross-entropy" in training["loss"]


def test_the_agent2_contrast_is_arithmetic_on_two_reported_numbers(summary, agent2):
    """The table `03_AGENT_3` asks for, checked against its own sources."""
    contrast = summary["agent2_contrast"]
    row = summary["leaderboard"][CANDIDATE_3]
    assert contrast["agent2_raw_cnn_r_ce"] == pytest.approx(
        agent2["leaderboard"][CANDIDATE_2]["r_ce"], rel=1e-12
    )
    assert contrast["agent3_c1_feature_cnn_r_ce"] == pytest.approx(row["r_ce"], rel=1e-12)
    assert contrast["difference_agent3_minus_agent2"] == pytest.approx(
        contrast["agent3_c1_feature_cnn_r_ce"] - contrast["agent2_raw_cnn_r_ce"], rel=1e-9
    )
    assert contrast["agent2_rerun"] is False
    assert contrast["within_equivalence_band"] == (
        abs(contrast["difference_agent3_minus_agent2"]) <= contrast["equivalence_band"]
    )


def test_the_interpretation_follows_the_instructions_own_rule(summary):
    contrast = summary["agent2_contrast"]
    difference = contrast["difference_agent3_minus_agent2"]
    band = contrast["equivalence_band"]
    reading = contrast["interpretation"]
    if abs(difference) <= band:
        assert "retained" in reading
    elif difference < -band:
        assert "Agent 3 is materially better" in reading
    else:
        assert "Agent 2 is materially better" in reading


def test_the_verdict_follows_the_stated_engineering_rule(summary):
    decision = summary["decision"]
    row = summary["leaderboard"][CANDIDATE_3]
    everyone = {
        **{name: block["r_ce"] for name, block in summary["earlier_reference_rows"].items()},
        CANDIDATE_3: row["r_ce"],
    }
    assert decision["leader_by_r_ce"] == min(everyone, key=lambda name: everyone[name])
    assert decision["leader_r_ce"] == pytest.approx(min(everyone.values()), rel=1e-12)
    for name in decision["within_band_of_leader"]:
        assert everyone[name] - decision["leader_r_ce"] <= decision["equivalence_band"]
    better = decision["agent3_materially_better_than_best_earlier"]
    assert better == bool(
        decision["agent3_minus_best_earlier_r_ce"] < -decision["equivalence_band"]
    )


def test_the_paired_comparisons_agree_with_the_leaderboard_ordering(summary):
    row = summary["leaderboard"][CANDIDATE_3]
    for key, block in summary["paired_comparisons"].items():
        other = key.split(" vs ")[1]
        reference = summary["earlier_reference_rows"].get(other)
        if reference is None or not block["distinguishable"]:
            continue
        assert (block["ce_difference"] < 0) == (row["ce"] < reference["ce"]), key


def test_the_report_says_what_agent_3_is_not_claiming():
    path = REPORT_DIRECTORY / "agent_03_report.md"
    if not path.exists():
        pytest.skip("agent_03_report.md has not been produced yet")
    # The report is hard-wrapped, so a claim can straddle a newline: compare
    # against the whitespace-collapsed text rather than the raw file.
    text = " ".join(path.read_text().split())
    for phrase in (
        "engineering prototype",
        "does not overturn the Phase 11 `FAIL`",
        "does not authorize Phase 12",
        "remains spent",
    ):
        assert phrase in text, phrase


def test_the_report_records_the_seam_it_chose_and_what_it_rejected():
    path = REPORT_DIRECTORY / "agent_03_report.md"
    if not path.exists():
        pytest.skip("agent_03_report.md has not been produced yet")
    text = path.read_text()
    assert "ProductionModel.encode(tokens)" in text
    assert "hidden.mean(dim=1)" in text
    assert "penultimate" in text


def test_the_report_carries_the_comparison_the_instruction_asks_for(summary):
    path = REPORT_DIRECTORY / "agent_03_report.md"
    if not path.exists():
        pytest.skip("agent_03_report.md has not been produced yet")
    text = path.read_text()
    contrast = summary["agent2_contrast"]
    assert f"{contrast['agent2_raw_cnn_r_ce']:.4f}" in text
    assert f"{contrast['agent3_c1_feature_cnn_r_ce']:.4f}" in text
    assert f"{contrast['difference_agent3_minus_agent2']:+.4f}" in text


def test_the_stop_condition_does_not_begin_agent_4(summary):
    stop = summary["stop_condition"]
    assert "Agent 4" in stop
    assert "not begun" in stop
    assert summary["phase12_authorized_by_this_artifact"] is False
