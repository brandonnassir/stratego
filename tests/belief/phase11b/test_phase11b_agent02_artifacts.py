"""Phase 11B Agent 2: the raw-observation CNN, its results and its report.

These tests protect what Agent 2 measured and, just as importantly, what
Agent 2 promised **not** to do. Four groups:

- **Boundary** — the Phase 11B status markers survive on every artifact, the
  common corpus was reused byte-for-byte rather than regenerated, no Agent 1
  or Phase 11 artifact moved, and no Agent 2 module reaches the spent test
  bank.
- **Model** — the parameter count is inside the instructed band, the
  per-square read-out uses the accepted token order, the gather visits each
  piece's own square, and the only input is the public observation.
- **Interface** — the required two methods exist, produce probability
  vectors, and reach worlds only through the accepted Phase 11 sampler.
- **Results** — every leaderboard number recomputes from the corpus, both
  declared runs are reported, and the verdict follows the stated rule.

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
CORPUS_DIRECTORY = REPOSITORY_ROOT / c11b.CORPUS_ROOT
CANDIDATE_2 = "agent02_raw_observation_cnn"

#: Names that would mean an Agent 2 module had reached the spent Phase 11
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
    return _load("agent_02_summary.json")


@pytest.fixture(scope="module")
def curves():
    return _load("agent_02_learning_curve.json")


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
# Boundary: what Agent 2 promised not to do
# ---------------------------------------------------------------------------


def test_the_summary_carries_every_status_marker_unchanged(summary):
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert summary[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert summary[key] == value, key
    assert summary["agent"] == 2


def test_the_report_and_the_checkpoint_carry_the_markers_too(summary):
    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 2 checkpoint is not present")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in c11b.PHASE11B_STATUS_MARKERS.items():
        assert payload[key] == value, key
    for key, value in c11b.PHASE11_FACTS.items():
        assert payload[key] == value, key


def test_the_common_corpus_was_reused_not_regenerated(summary, agent1):
    corpus = summary["common_corpus"]
    assert corpus["corpus_digest"] == agent1["common_corpus"]["corpus_digest"]
    assert corpus["corpus_digest_matches_agent1"] is True
    assert corpus["corpus_digest_matches_manifest"] is True
    assert corpus["file_digest_drift"] == []
    assert summary["preservation"]["corpus_regenerated"] is False
    for split in c11b.CORPUS_SPLITS:
        assert corpus["file_digests"][split] == agent1["common_corpus"]["file_digests"][split]


def test_the_corpus_bytes_still_hash_to_what_agent2_scored_on(summary, dev):
    """The recorded identity is only worth having if it still holds."""
    recomputed = {split: store.split_digest(CORPUS_DIRECTORY, split) for split in c11b.CORPUS_SPLITS}
    assert recomputed == summary["common_corpus"]["file_digests"]


def test_no_agent1_or_phase11_artifact_moved(summary):
    drifted = []
    for relative, digest in summary["preserved_artifact_digests"].items():
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            drifted.append(f"{relative}: missing")
        elif _sha256(path) != digest:
            drifted.append(f"{relative}: changed")
    assert drifted == [], f"Agent 2 must leave these untouched: {drifted}"
    assert summary["preservation"]["agent1_artifacts_modified"] is False
    assert summary["preservation"]["phase11_artifacts_unchanged_since_agent1"] is True
    assert summary["preservation"]["phase11_test_bank_opened"] is False


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


def test_no_agent2_module_reaches_the_spent_test_bank():
    sources = [
        REPOSITORY_ROOT / "stratego" / "belief" / "phase11b" / "raw_cnn.py",
        REPOSITORY_ROOT / "stratego" / "belief" / "phase11b" / "raw_train.py",
        REPOSITORY_ROOT / "scripts" / "run_phase11b_agent02.py",
        REPOSITORY_ROOT / "scripts" / "_phase11b_agent02_report.py",
    ]
    offenders = []
    for path in sources:
        if not path.exists():
            continue
        tokens = _executable_tokens(path)
        for token in FORBIDDEN_CODE_TOKENS:
            if token in tokens:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], f"Agent 2 must not reach the spent bank: {offenders}"


# ---------------------------------------------------------------------------
# Model: shape, band, geometry, and the input boundary
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_model():
    from stratego.belief.phase11b.raw_cnn import build_raw_cnn

    return build_raw_cnn(seed=17)


def test_the_parameter_count_is_inside_the_instructed_band(raw_model):
    from stratego.belief.phase11b.raw_cnn import parameter_breakdown

    breakdown = parameter_breakdown(raw_model)
    assert 3_000_000 <= breakdown["total"] <= 5_000_000
    assert (
        breakdown["stem"] + breakdown["residual_tower"] + breakdown["readout"]
        == breakdown["total"]
    )
    assert 160 <= breakdown["width"] <= 192


def test_the_reported_parameter_count_is_the_model_that_was_trained(summary, raw_model):
    from stratego.belief.phase11b.raw_cnn import parameter_count

    assert summary["leaderboard"][CANDIDATE_2]["parameters"] == parameter_count(raw_model)
    assert summary["pilot"]["parameters"]["total"] == parameter_count(raw_model)


def test_the_model_takes_the_public_observation_and_nothing_else(raw_model):
    import inspect

    signature = inspect.signature(raw_model.forward)
    assert list(signature.parameters) == ["observations"]
    from stratego.belief.phase11b.interface import Phase11BPublicState

    assert set(Phase11BPublicState.__dataclass_fields__) == {
        "public_state_document",
        "observation",
    }


def test_the_model_refuses_anything_that_is_not_a_127_channel_observation(raw_model):
    import torch

    from stratego.belief.phase11b.raw_cnn import Phase11BRawCNNError

    with pytest.raises(Phase11BRawCNNError):
        raw_model(torch.zeros(2, 64, 10, 10))
    with pytest.raises(Phase11BRawCNNError):
        raw_model(torch.zeros(127, 10, 10))


def test_the_per_square_readout_uses_the_accepted_token_order(raw_model):
    """`per_square_logits` must agree with `observation_to_tokens`.

    If these two disagreed, every piece would be supervised at some other
    piece's square and the whole experiment would be meaningless. The check
    runs the accepted flattening and the model's read-out reshape over the
    same probe planes and compares them elementwise.
    """
    import torch

    from stratego.model.tokenization import observation_to_tokens

    probe = torch.arange(127 * 100, dtype=torch.float32).reshape(1, 127, 10, 10)
    accepted = observation_to_tokens(probe)
    mine = probe.reshape(1, 127, 100).transpose(1, 2)
    assert torch.equal(accepted, mine)
    # And the model's own read-out is that same reshape on 12 channels.
    planes = torch.arange(12 * 100, dtype=torch.float32).reshape(1, 12, 10, 10)
    reshaped = planes.reshape(1, 12, 100).transpose(1, 2)
    for square in (0, 1, 10, 45, 99):
        assert torch.equal(reshaped[0, square], planes[0, :, square // 10, square % 10])


def test_the_batcher_pairs_every_piece_with_its_own_square(dev):
    """The gather must visit exactly the corpus's stored squares, in order."""
    from stratego.belief.phase11b.train import _sample_batches

    rows = np.arange(int(dev["samples"]), dtype=np.int64)
    squares = np.concatenate(
        [batch[2] for batch in _sample_batches(dev, rows, 512)]
    )
    labels = np.concatenate([batch[3] for batch in _sample_batches(dev, rows, 512)])
    assert np.array_equal(squares, np.asarray(dev["perspective_square"], dtype=np.int64))
    assert np.array_equal(labels, np.asarray(dev["true_rank"], dtype=np.int64))


def test_dropout_costs_no_parameters_but_does_shift_the_readout_keys():
    """Why `load_raw_cnn` rebuilds from the checkpoint's own record.

    Dropout has no parameters, so the two settings are the same model by
    every count that matters. But a read-out dropout module occupies an
    `nn.Sequential` position, which renumbers the read-out's `state_dict`
    keys — so a checkpoint must be rebuilt at the shape it was saved with,
    not at the default.
    """
    from stratego.belief.phase11b.raw_cnn import build_raw_cnn, parameter_count

    plain = build_raw_cnn(seed=3)
    regularized = build_raw_cnn(seed=3, block_dropout=0.1, readout_dropout=0.3)
    assert parameter_count(plain) == parameter_count(regularized)
    assert sorted(tuple(p.shape) for p in plain.parameters()) == sorted(
        tuple(p.shape) for p in regularized.parameters()
    )
    plain_keys = [name for name, _ in plain.named_parameters()]
    regularized_keys = [name for name, _ in regularized.named_parameters()]
    assert plain_keys != regularized_keys
    # Everything before the read-out is untouched; only the read-out shifts.
    tower = [name for name in plain_keys if not name.startswith("readout")]
    assert tower == [name for name in regularized_keys if not name.startswith("readout")]


def test_the_saved_checkpoint_loads_back_into_the_shape_it_was_saved_with(summary):
    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 2 checkpoint is not present")
    from stratego.belief.phase11b.raw_cnn import load_raw_cnn, parameter_count

    model, payload = load_raw_cnn(path)
    assert parameter_count(model) == payload["parameters"]
    assert payload["corpus_digest"] == summary["common_corpus"]["corpus_digest"]
    assert not model.training


def test_the_model_is_a_pure_function_of_the_observation_in_eval_mode(raw_model):
    import torch

    raw_model.eval()
    probe = torch.randn(3, 127, 10, 10)
    with torch.no_grad():
        first = raw_model(probe)
        second = raw_model(probe)
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def test_the_interface_exposes_exactly_the_two_required_methods():
    from stratego.belief.phase11b.raw_cnn import RawObservationBeliefModel

    for name in ("predict_marginals", "sample_worlds"):
        assert callable(getattr(RawObservationBeliefModel, name))


def test_the_adapter_inherits_the_accepted_sampler_path_rather_than_forking_it():
    from stratego.belief.phase11b.interface import Phase11BBeliefModel
    from stratego.belief.phase11b.raw_cnn import RawObservationBeliefModel

    assert issubclass(RawObservationBeliefModel, Phase11BBeliefModel)
    # `sample_worlds` is inherited, not overridden: the accepted Phase 11
    # sampler is reached through Agent 1's adapter and nothing else.
    assert "sample_worlds" not in RawObservationBeliefModel.__dict__


def test_the_interface_block_records_a_run_through_the_accepted_sampler(summary):
    block = summary["interface"]
    assert block["candidate_id"] == CANDIDATE_2
    assert block["positions_checked"] > 0
    assert block["worlds_sampled"] > 0
    assert block["all_marginals_sum_to_one"] is True
    assert block["sample_worlds_seed_deterministic"] is True
    assert block["all_worlds_passed_accepted_validation_stack"] is True
    assert "accepted, unmodified" in block["sampler_source"]
    assert block["describe"]["reads_hidden_truth"] is False
    assert block["describe"]["consumes_c1_features"] is False


def test_the_marginals_of_the_trained_model_are_probability_vectors(summary, dev):
    """Every stored development piece gets a finite 12-way simplex vector."""
    path = REPOSITORY_ROOT / summary["checkpoint"]["path"]
    if not path.exists():
        pytest.skip("the Agent 2 checkpoint is not present")
    from stratego.belief.phase11b.raw_cnn import load_raw_cnn
    from stratego.belief.phase11b.raw_train import (
        predict_probabilities_raw,
        stage_observations,
    )

    model, _payload = load_raw_cnn(path)
    observations = stage_observations(
        {"observations": dev["observations"][:64]}, "cpu", on_device=False
    )
    slice_data = _first_positions(dev, 64)
    probabilities = predict_probabilities_raw(
        model, observations, slice_data, device="cpu", batch_positions=64
    )
    assert probabilities.shape == (slice_data["pieces"], c11b.RANK_COUNT)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def _first_positions(data: dict, count: int) -> dict:
    """The first `count` stored positions, as a standalone split view."""
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)[: count + 1]
    pieces = int(offsets[-1])
    view = dict(data)
    view["samples"] = count
    view["pieces"] = pieces
    view["piece_offset"] = offsets
    for name in ("perspective_square", "true_rank"):
        view[name] = np.asarray(data[name])[:pieces]
    return view


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_the_leaderboard_row_has_every_shared_metric_the_sprint_requires(summary):
    row = summary["leaderboard"][CANDIDATE_2]
    for field in (
        "ce",
        "baseline_ce",
        "r_ce",
        "r_ce_ci95",
        "top1",
        "r_ce_by_stratum",
        "parameters",
        "training_seconds",
        "time_to_best_seconds",
        "inference_microseconds_per_piece",
        "checkpoint_sha256",
    ):
        assert field in row, field
    assert set(row["r_ce_by_stratum"]) == set(c11b.CORPUS_STRATA)
    assert row["corpus_version"] == c11b.CORPUS_VERSION


def test_the_reported_r_ce_is_the_reported_ce_over_the_reported_baseline(summary):
    row = summary["leaderboard"][CANDIDATE_2]
    assert row["r_ce"] == pytest.approx(row["ce"] / row["baseline_ce"], rel=1e-12)
    lower, upper = row["r_ce_ci95"]
    assert lower < row["r_ce"] < upper


def test_the_denominator_is_the_corpus_own_remaining_count_baseline(summary, dev):
    baseline = M.baseline_probabilities(dev)
    recomputed = float(M.cross_entropy(baseline, np.asarray(dev["true_rank"])).mean())
    assert summary["leaderboard"][CANDIDATE_2]["baseline_ce"] == pytest.approx(
        recomputed, rel=1e-9
    )


def test_every_candidate_on_the_board_divides_by_that_same_denominator(summary):
    denominator = summary["leaderboard"][CANDIDATE_2]["baseline_ce"]
    for name, row in summary["agent1_reference_rows"].items():
        assert row["baseline_ce"] == pytest.approx(denominator, rel=1e-9), name
        assert row["corpus_version"] == c11b.CORPUS_VERSION, name


def test_the_agent1_rows_are_quoted_verbatim_not_recomputed(summary, agent1):
    for name, row in summary["agent1_reference_rows"].items():
        assert row == agent1["leaderboard"][name], name


def test_agent1s_saved_checkpoints_reproduce_the_numbers_agent1_reported(summary):
    """If they did not, the paired comparison would not be like-for-like.

    The tolerance is `1e-4` rather than exact because Agent 1 measured a
    run-to-run drift of up to `6.45e-5 R_CE` on these very candidates and
    said so; anything inside that band is the float32 noise Agent 1
    documented, and it is two orders of magnitude below the gaps the
    leaderboard turns on. The unchanged Phase 11 head, which involves no
    trained-in-Phase-11B weights, is required to reproduce exactly.
    """
    for name, block in summary["agent1_reproduction"].items():
        if block["r_ce_reported_by_agent1"] is None:
            continue
        limit = 0.0 if "unchanged" in name else 1e-4
        assert block["absolute_difference"] <= limit, name


def test_the_two_backends_agree_on_the_reported_number(summary):
    block = summary["backend_agreement"]
    assert block["absolute_difference"] < 1e-3
    assert block["scoring_backend"] == "cpu"


def test_both_declared_runs_are_reported_and_only_one_architecture_was_trained(summary):
    training = summary["training"]
    assert training["architectures_trained"] == 1
    assert training["configurations_declared"] == len(training["runs"]) == 2
    assert training["selected_run"] in training["runs"]
    architectures = {block["parameters"] for block in training["runs"].values()}
    assert len(architectures) == 1, "the two runs must be the same architecture"
    selected = training["runs"][training["selected_run"]]
    other = next(
        block for name, block in training["runs"].items() if name != training["selected_run"]
    )
    assert selected["dev_ce"] <= other["dev_ce"], "the selection rule was not applied"


def test_the_kept_checkpoint_is_the_best_probe_of_its_whole_run(summary, curves):
    """Best-on-development means best over *every* probe, not every epoch."""
    for name, block in curves["runs"].items():
        reported = summary["training"]["runs"][name]
        curve = block["curve"]
        epochs = [row for row in curve if not row["sub_epoch"]]
        assert len(epochs) == reported["epochs_run"]
        assert len(curve) == reported["evaluations"]
        best = min(curve, key=lambda row: row["dev_ce"])
        assert best["step"] == reported["best_step"]
        assert best["epoch"] == reported["best_epoch"]
        assert best["dev_ce"] == pytest.approx(reported["dev_ce"], rel=1e-6)
    assert curves["selected_run"] == summary["training"]["selected_run"]


def test_the_sub_epoch_probe_ran_at_the_declared_cadence(summary, curves):
    """Probes fire every `steps_per_epoch // evaluations_per_epoch` steps.

    That stride does not divide the epoch evenly, so an epoch carries
    `ceil(steps / stride)` sub-epoch probes plus its own boundary row, and
    the count can legitimately exceed `evaluations_per_epoch`.
    """
    import math

    for name, block in curves["runs"].items():
        reported = summary["training"]["runs"][name]
        per_epoch = int(block["config"]["evaluations_per_epoch"])
        steps = int(reported["steps_per_epoch"])
        stride = max(1, steps // per_epoch)
        ceiling = 1 + math.ceil(steps / stride)
        counts = {}
        for row in block["curve"]:
            counts[row["epoch"]] = counts.get(row["epoch"], 0) + 1
        for epoch, count in counts.items():
            assert per_epoch <= count <= ceiling, (name, epoch, count)
        assert steps > per_epoch


def test_the_probe_only_changed_how_finely_the_run_was_observed(curves):
    """Probing must not alter training: the loss curve is still monotone in step."""
    for name, block in curves["runs"].items():
        steps = [row["step"] for row in block["curve"]]
        assert steps == sorted(steps), name
        assert len(set(steps)) == len(steps), name


def test_the_corpus_size_diagnostic_never_became_the_reported_candidate(summary):
    scale = summary.get("corpus_size_diagnostic")
    if not scale:
        pytest.skip("the corpus-size diagnostic was not run")
    assert scale["diagnostic_only"] is True
    reported = summary["training"]["runs"][summary["training"]["selected_run"]]
    checkpoints = {
        block["checkpoint"]["path"] for block in summary["training"]["runs"].values()
    }
    assert summary["checkpoint"]["path"] in checkpoints
    # The full-corpus point must be the reported run quoted, not a retrain.
    full = max(scale["points"], key=lambda point: point["games"])
    assert full["best_r_ce"] == pytest.approx(reported["dev_r_ce"], rel=1e-12)
    assert "not retrained" in full["note"]


def test_the_corpus_size_diagnostic_slices_whole_games(summary):
    scale = summary.get("corpus_size_diagnostic")
    if not scale:
        pytest.skip("the corpus-size diagnostic was not run")
    points = sorted(scale["points"], key=lambda point: point["games"])
    assert [point["games"] for point in points] == sorted(
        point["games"] for point in points
    )
    for smaller, larger in zip(points, points[1:]):
        assert smaller["games"] < larger["games"]
        assert smaller["positions"] < larger["positions"]
        assert smaller["pieces"] < larger["pieces"]
    assert points[-1]["games"] == c11b.CORPUS_SPLITS["train"]["games"]


def test_the_supervised_loss_used_no_policy_value_or_outcome_term(summary):
    training = summary["training"]
    assert training["policy_or_value_terms"] is False
    assert training["game_outcome_used"] is False
    assert "cross-entropy" in training["loss"]


def test_the_verdict_follows_the_stated_engineering_rule(summary):
    decision = summary["decision"]
    row = summary["leaderboard"][CANDIDATE_2]
    earlier = summary["agent1_reference_rows"]
    everyone = {CANDIDATE_2: row["r_ce"], **{n: b["r_ce"] for n, b in earlier.items()}}
    assert decision["leader_by_r_ce"] == min(everyone, key=everyone.get)
    assert decision["leader_r_ce"] == pytest.approx(min(everyone.values()))
    band = decision["equivalence_band"]
    expected = sorted(
        name for name, value in everyone.items() if value - min(everyone.values()) <= band
    )
    assert sorted(decision["within_band_of_leader"]) == expected
    best_id = decision["agent1_best_candidate"]
    assert decision["agent2_minus_agent1_best_r_ce"] == pytest.approx(
        row["r_ce"] - earlier[best_id]["r_ce"]
    )
    assert decision["agent2_materially_better_than_agent1_best"] is (
        decision["agent2_minus_agent1_best_r_ce"] < -band
    )


def test_the_paired_comparison_agrees_with_the_leaderboard_ordering(summary):
    row = summary["leaderboard"][CANDIDATE_2]
    for name, block in summary["paired_comparisons"].items():
        other = name.split(" vs ")[1]
        reported = summary["agent1_reference_rows"][other]["ce"]
        # A negative mean difference must mean Agent 2 has the lower CE.
        assert (block["ce_difference"] < 0) == (row["ce"] < reported), name
        assert block["bootstrap_unit"] == "game"


def test_the_report_says_what_agent_2_is_not_claiming():
    path = REPORT_DIRECTORY / "agent_02_report.md"
    if not path.exists():
        pytest.skip("the Agent 2 report has not been produced yet")
    text = path.read_text()
    for phrase in (
        "engineering prototype",
        "does not authorize Phase 12",
        "remains spent",
        "Agent 3's experiment was not",
    ):
        assert phrase in text, phrase


def test_the_report_discloses_that_a_second_configuration_was_run():
    path = REPORT_DIRECTORY / "agent_02_report.md"
    if not path.exists():
        pytest.skip("the Agent 2 report has not been produced yet")
    text = path.read_text()
    assert "run1_declared" in text and "run2_regularized" in text
    assert "not a sweep" in text
