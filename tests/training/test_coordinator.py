"""Coordinator unit surface: legality construction, configuration, decision rule.

The pieces here run without a device wherever possible, so a regression in the
legality maths or the decision rule shows up in an ordinary test run rather than
only inside a Metal benchmark. The tests that genuinely need the pipeline live
in `test_end_to_end_pipeline.py`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stratego.engine.constants import ACTION_SPACE_SIZE  # noqa: E402
from stratego.training.coordinator import (  # noqa: E402
    DTYPE_BY_NAME,
    LEGALITY_MODES,
    PRECISIONS,
    CoordinatorConfig,
    CoordinatorError,
    RunTotals,
    StepMetrics,
    compact_legality_from_masks,
)
from stratego.training.end_to_end_benchmark import (  # noqa: E402
    DECISION_BUILD_BACKEND,
    DECISION_KEEP_PYTHON,
    DECISION_KEEP_PYTHON_OPTIONAL,
    ENVIRONMENT_COUNTS,
    INFERENCE_BATCH_SIZES,
    WORKER_COUNTS,
    build_screening_plan,
    compute_ratio,
    decide_backend,
)
from stratego.training.shared_buffers import POLICY_CAPACITY  # noqa: E402


# ---------------------------------------------------------------------------
# Compact legality
# ---------------------------------------------------------------------------


def _random_masks(rows: int, seed: int = 0, max_legal: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.uint8)
    for row in range(rows):
        count = int(rng.integers(1, max_legal + 1))
        columns = rng.choice(ACTION_SPACE_SIZE, size=count, replace=False)
        masks[row, columns] = 1
    return masks


def test_compact_legality_matches_per_row_nonzero():
    """The vectorised build must equal the obvious per-row construction."""
    masks = _random_masks(64, seed=7)
    action_ids, valid, counts = compact_legality_from_masks(masks, capacity=64)
    for row in range(masks.shape[0]):
        expected = np.flatnonzero(masks[row])
        assert counts[row] == len(expected)
        assert valid[row].sum() == len(expected)
        assert action_ids[row, : len(expected)].tolist() == expected.tolist()
        # Padding must be marked invalid, whatever it happens to contain.
        assert not valid[row, len(expected) :].any()


def test_compact_legality_is_ascending():
    """Ascending order is the contract a worker relies on to line probabilities up."""
    masks = _random_masks(32, seed=11)
    action_ids, valid, counts = compact_legality_from_masks(masks, capacity=64)
    for row in range(masks.shape[0]):
        count = int(counts[row])
        entries = action_ids[row, :count]
        assert np.all(np.diff(entries) > 0)


def test_compact_legality_raises_above_capacity():
    """A row that does not fit must fail loudly, never silently lose a move."""
    masks = np.zeros((2, ACTION_SPACE_SIZE), dtype=np.uint8)
    masks[0, :10] = 1
    masks[1, :40] = 1
    with pytest.raises(CoordinatorError, match="above the compact capacity"):
        compact_legality_from_masks(masks, capacity=32)


def test_compact_legality_rejects_wrong_shape():
    with pytest.raises(ValueError):
        compact_legality_from_masks(np.zeros((4, 9), dtype=np.uint8), capacity=8)


def test_compact_legality_handles_single_row():
    masks = np.zeros((1, ACTION_SPACE_SIZE), dtype=np.uint8)
    masks[0, [3, 17, 9000]] = 1
    action_ids, valid, counts = compact_legality_from_masks(masks, capacity=8)
    assert counts.tolist() == [3]
    assert action_ids[0, :3].tolist() == [3, 17, 9000]
    assert valid[0, :3].all()


# ---------------------------------------------------------------------------
# Gumbel sampling: the illegal-action regression
# ---------------------------------------------------------------------------
#
# `torch.rand` draws from [0, 1). A `u` of exactly 0 used to make the Gumbel
# noise `+inf`, which added to the `-inf` at an illegal entry gives `NaN`, and
# `argmax` ranks `NaN` above everything -- so the sample landed on an action the
# engine had declared illegal. It happens about once in 17 million draws, which
# a short benchmark never sees and a sustained self-play run always does.


def test_gumbel_noise_is_finite_even_when_the_uniform_draw_is_zero():
    from stratego.training.representative_model import _gumbel_noise

    noise = _gumbel_noise((4096, 512), torch.device("cpu"), None)
    assert torch.isfinite(noise).all()


def test_gumbel_noise_is_finite_over_many_draws():
    """Enough draws that a zero is likely, on the backend that produces them."""
    from stratego.training.representative_model import _gumbel_noise

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20_250_809)
    for _ in range(8):
        noise = _gumbel_noise((10_000, 512), torch.device("cpu"), generator)
        assert torch.isfinite(noise).all()


def test_a_zero_uniform_would_have_produced_positive_infinity():
    """Pin the mechanism itself, so a future rewrite cannot reintroduce it."""
    zero = torch.zeros((1,), dtype=torch.float32)
    unguarded = -torch.log(-torch.log1p(-zero.clamp(max=1.0 - 1e-7)))
    assert torch.isposinf(unguarded).all()
    # And that infinity is what turns an illegal entry into the argmax.
    combined = torch.tensor([float("-inf")]) + unguarded
    assert torch.isnan(combined).all()
    assert int(torch.argmax(torch.tensor([float("nan"), 1.0, 2.0]))) == 0


def test_sampling_never_leaves_the_legal_set():
    """The property that actually matters, over many masked draws."""
    from stratego.training.representative_model import sample_dense

    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(4321)
    rows = 256
    mask = torch.zeros((rows, ACTION_SPACE_SIZE), dtype=torch.bool)
    for row in range(rows):
        columns = torch.randperm(ACTION_SPACE_SIZE)[: 2 + (row % 40)]
        mask[row, columns] = True
    for _ in range(20):
        logits = torch.randn((rows, ACTION_SPACE_SIZE), dtype=torch.float16)
        actions = sample_dense(logits, mask, generator=generator)
        assert mask.gather(1, actions.unsqueeze(1)).squeeze(1).all()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_configuration_rejects_unknown_precision_and_legality():
    with pytest.raises(ValueError, match="unknown precision"):
        CoordinatorConfig(64, 2, 64, precision="float8")
    with pytest.raises(ValueError, match="unknown legality"):
        CoordinatorConfig(64, 2, 64, legality="sparse")


def test_configuration_rejects_capacity_beyond_the_shared_row():
    """The compact width has to fit the shared policy row or a decision is lost."""
    with pytest.raises(ValueError, match="exceeds the shared policy row"):
        CoordinatorConfig(64, 2, 64, compact_capacity=POLICY_CAPACITY + 8)


def test_configuration_rejects_non_positive_batch():
    with pytest.raises(ValueError, match="inference batch size must be positive"):
        CoordinatorConfig(64, 2, 0)


def test_every_precision_and_legality_mode_is_known():
    assert set(PRECISIONS) == set(DTYPE_BY_NAME)
    assert set(LEGALITY_MODES) == {"dense", "compact"}


def test_configuration_label_is_distinct_per_dimension():
    base = CoordinatorConfig(256, 4, 128)
    variants = [
        CoordinatorConfig(256, 4, 128, precision="float32"),
        CoordinatorConfig(256, 4, 128, legality="compact"),
        CoordinatorConfig(256, 4, 256),
        CoordinatorConfig(256, 6, 128),
        CoordinatorConfig(512, 4, 128),
    ]
    labels = {base.label} | {variant.label for variant in variants}
    assert len(labels) == len(variants) + 1


# ---------------------------------------------------------------------------
# Metrics accounting
# ---------------------------------------------------------------------------


def test_run_totals_accumulate_every_stage():
    totals = RunTotals()
    for index in range(3):
        totals.add(
            StepMetrics(
                step=index,
                positions=10,
                transitions=9,
                chunks=2,
                wall_seconds=0.5,
                inference_seconds=0.3,
                worker_seconds=0.1,
            )
        )
    assert totals.steps == 3
    assert totals.positions == 30
    assert totals.transitions == 27
    assert totals.chunks == 6
    assert totals.wall_seconds == pytest.approx(1.5)
    assert totals.inference_seconds == pytest.approx(0.9)
    assert len(totals.step_latencies) == 3


def test_coordinator_seconds_is_the_sum_of_the_coordinator_stages():
    metrics = StepMetrics(
        observation_seconds=1.0,
        legality_seconds=2.0,
        transfer_seconds=3.0,
        inference_seconds=4.0,
        sampling_seconds=5.0,
        writeback_seconds=6.0,
        worker_seconds=100.0,
    )
    # Worker time is deliberately excluded: it is what the coordinator waits on.
    assert metrics.coordinator_seconds == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio, expected, required",
    [
        (4.0, DECISION_KEEP_PYTHON, False),
        (2.0, DECISION_KEEP_PYTHON, False),
        (1.99, DECISION_KEEP_PYTHON_OPTIONAL, False),
        (1.25, DECISION_KEEP_PYTHON_OPTIONAL, False),
        (1.24, DECISION_BUILD_BACKEND, True),
        (0.5, DECISION_BUILD_BACKEND, True),
    ],
)
def test_decision_rule_boundaries(ratio, expected, required):
    """The thresholds are inclusive at 2.0 and at 1.25, as specified."""
    decision, needs_backend = decide_backend(ratio)
    assert decision == expected
    assert needs_backend is required


def test_compute_ratio_reports_the_decision_and_the_inputs():
    result = compute_ratio(90_000.0, 15_000.0)
    assert result["R"] == pytest.approx(6.0)
    assert result["backend_decision"] == DECISION_KEEP_PYTHON
    assert result["optimized_backend_required"] is False
    assert result["simulation_pipeline_positions_per_second"] == 90_000.0
    assert result["representative_model_inference_positions_per_second"] == 15_000.0


def test_compute_ratio_rejects_a_zero_denominator():
    with pytest.raises(CoordinatorError):
        compute_ratio(90_000.0, 0.0)


# ---------------------------------------------------------------------------
# Screening plan
# ---------------------------------------------------------------------------


def test_screening_plan_covers_every_required_dimension_value():
    """A screened subset is allowed; skipping a required value is not."""
    plan = build_screening_plan()
    assert {point["num_workers"] for point in plan} >= set(WORKER_COUNTS)
    assert {point["num_environments"] for point in plan} >= set(ENVIRONMENT_COUNTS)
    assert {point["inference_batch_size"] for point in plan} >= set(
        INFERENCE_BATCH_SIZES
    )


def test_screening_plan_is_much_smaller_than_the_cartesian_product():
    plan = build_screening_plan()
    cartesian = len(WORKER_COUNTS) * len(ENVIRONMENT_COUNTS) * len(INFERENCE_BATCH_SIZES)
    assert len(plan) < cartesian // 4


def test_screening_plan_has_no_duplicate_points():
    plan = build_screening_plan()
    keys = [
        (
            point["num_workers"],
            point["num_environments"],
            point["inference_batch_size"],
            point["precision"],
            point["legality"],
        )
        for point in plan
    ]
    assert len(keys) == len(set(keys))


def test_screening_plan_never_asks_for_more_workers_than_environments():
    for point in build_screening_plan():
        assert point["num_environments"] >= point["num_workers"]


def test_screening_plan_includes_both_baselines():
    plan = build_screening_plan(precision="float16", legality="dense")
    assert any(point["precision"] == "float32" for point in plan)
    assert any(point["legality"] == "compact" for point in plan)
