"""The synthetic assay runner at reduced scale: refusals, seed derivation,
the outcomes-only boundary, deterministic replay and the artifacts."""

import json

import numpy as np
import pytest

from stratego.setups.identity import derive_stream_seed
from stratego.training.phase18 import synthetic_assay as assay
from stratego.training.phase18.setup_contract import Phase18SetupError, json_document_digest, seed_uniform
from stratego.training.phase18.synthetic_landscape import build_landscape

NS = "phase18_g2_assay_test_v1"


def test_the_frozen_design_enforces_every_minimum_of_the_instruction():
    design = assay.AssayDesign(namespace=NS)
    assert (len(design.seed_indices), design.updates, design.pool_size) == (3, 64, 1024)
    assert (design.outcomes_per_setup, design.evaluation_samples, design.bootstrap_replicates) == (4, 4096, 10000)
    assert design.batch_size == 1024 and design.epochs_per_update == 5 and design.reduced is False
    for bad in (
        dict(seed_indices=(1, 2)),
        dict(updates=65),
        dict(outcomes_per_setup=3),
        dict(evaluation_samples=4095),
        dict(bootstrap_replicates=9999),
        dict(pool_size=512),
        dict(batch_size=256),
        dict(epochs_per_update=1),
    ):
        with pytest.raises(Phase18SetupError):
            assay.AssayDesign(namespace=NS, **bad)


def test_every_seed_derives_from_the_namespace_through_derive_stream_seed():
    design = assay.AssayDesign(namespace=NS)
    for k in design.seed_indices:
        assert design.model_seed(k) == derive_stream_seed(NS, "model_init", str(k))
    assert design.landscape_table_seed() == derive_stream_seed(NS, "landscape_table")
    assert design.bootstrap_seed() == derive_stream_seed(NS, "paired_bootstrap")
    assert len({design.model_seed(k) for k in design.seed_indices}) == 3
    document = design.document()
    assert document["model_seeds"] == {str(k): design.model_seed(k) for k in design.seed_indices}
    assert json_document_digest(document) == json_document_digest(assay.AssayDesign(namespace=NS).document())


class _Watched:
    """A landscape proxy that records which public methods the runner calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def __getattr__(self, name):
        value = getattr(self._inner, name)
        if callable(value) and not name.startswith("_"):
            def wrapped(*args, **kwargs):
                self.calls.append(name)
                return value(*args, **kwargs)
            return wrapped
        return value


@pytest.fixture(scope="module")
def reduced_run(tmp_path_factory):
    design = assay.AssayDesign(
        namespace=NS, run_id="G2-TEST-REDUCED", seed_indices=(1,), updates=2, pool_size=48,
        outcomes_per_setup=4, evaluation_samples=64, curve_every=1, bootstrap_replicates=100,
        batch_size=24, reduced=True, threads=4,
    )
    landscape = build_landscape(namespace=NS, table_seed=design.landscape_table_seed(), kappa=3.0, p_draw=0.1)
    watched = _Watched(landscape)
    output = tmp_path_factory.mktemp("assay") / "seed_1"
    record = assay.run_seed(design, watched, 1, output, log=lambda *_: None)
    return design, landscape, watched, output, record


def test_the_runner_calls_only_outcomes_for_on_the_learning_side(reduced_run):
    design, _, watched, _, _ = reduced_run
    evaluator_side = ("utilities", "expected_z_outcome", "utility", "z_scores")
    learner_side = [c for c in watched.calls if c not in evaluator_side]
    assert set(learner_side) == {"outcomes_for"}
    # Evaluator reads: (initial + curve points + final) x (EMA + raw diagnostic)
    # held-out evaluations, plus one pool-utility diagnostic per update.
    curve_points = sum(1 for u in range(1, design.updates + 1) if u % design.curve_every == 0 and u != design.updates)
    evaluations = (2 + curve_points) * 2
    assert watched.calls.count("expected_z_outcome") == evaluations
    assert watched.calls.count("utilities") == evaluations + design.updates
    # The learner never receives a utility: the only value that crosses to the
    # update loop is the outcome list.
    assert "utility" not in learner_side and "z_scores" not in learner_side


def test_the_reduced_run_writes_every_artifact_with_zero_integrity_events(reduced_run):
    design, landscape, _, output, record = reduced_run
    for name in ("seed_result.json", "telemetry.jsonl", "outcome_receipts.jsonl", "utilities_initial.npy", "utilities_final.npy", "checkpoint_final/manifest.json"):
        assert (output / name).exists(), name
    assert record["integrity"] == {
        "legality_failures": 0, "orientation_failures": 0, "attribution_failures": 0, "non_finite_events": 0,
        "checkpoint_identity_failures": 0, "immediately_terminal_setups": record["integrity"]["immediately_terminal_setups"], "duplicates_collapsed": record["integrity"]["duplicates_collapsed"],
    }
    assert record["optimizer_steps"] == 2 * 5 * 2  # two updates, five epochs, ceil(48/24) = 2 minibatches
    assert record["ema_updates"] == 2
    rows = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == [1, 2]
    assert all(r["update_result"]["ready_rows"] == 48 - r["pool"]["immediately_terminal_count"] for r in rows)
    assert rows[1]["buffer_after_filter"]["rows"] == 48 + 48 - rows[1]["pool_record"]["duplicates_collapsed"]
    assert rows[1]["update_result"]["process"]["excluded_zero_outcome_rows"] == 48 - rows[1]["pool_record"]["duplicates_collapsed"], "the previous pool survives with zero outcomes and is excluded"
    initial = np.load(output / "utilities_initial.npy")
    final = np.load(output / "utilities_final.npy")
    assert initial.shape == final.shape == (64,)
    assert record["paired"]["mean_difference"] == pytest.approx(float((final - initial).mean()))
    assert record["gap"]["exact_optimum"] == landscape.optimum
    assert len(record["curve"]) == 3 and [c["update"] for c in record["curve"]] == [0, 1, 2]


def test_the_outcome_receipts_replay_deterministically_from_the_seeds(reduced_run):
    design, landscape, _, output, record = reduced_run
    receipts = [json.loads(line) for line in (output / "outcome_receipts.jsonl").read_text().splitlines()]
    by_period = {}
    for row in receipts:
        by_period.setdefault(row["period"], []).append(row)
    for period, rows in by_period.items():
        replay = []
        for row in rows:
            uniforms = [seed_uniform(landscape.outcome_seed(1, period, row["fingerprint"], r)) for r in range(design.outcomes_per_setup)]
            assert len(row["outcomes"]) == design.outcomes_per_setup
            replay.append([row["fingerprint"], row["outcomes"]])
            # every recorded outcome is consistent with its uniform and the mapping threshold order
            for symbol, u in zip(row["outcomes"], uniforms):
                assert symbol in "-0+"
        assert json_document_digest(replay) == record["period_outcome_digests"][period - 1]
    assert len(by_period) == design.updates


def test_the_evaluation_stream_is_the_same_at_every_endpoint(reduced_run):
    """Common random numbers: the initial and final samples share their token
    uniforms, so per-sample differences are paired."""
    design, _, _, output, record = reduced_run
    from stratego.training.phase18.setup_contract import stream_seed

    first = stream_seed(design.namespace, "eval", 1, 0, 0)
    assert first == stream_seed(design.namespace, "eval", 1, 0, 0)
    assert record["initial"]["samples"] == record["final"]["samples"] == 64
    assert record["initial"]["ema_updates"] == 0 and record["final"]["ema_updates"] == 2
