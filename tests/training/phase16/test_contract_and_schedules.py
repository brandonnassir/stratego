"""Phase 16 Agent 3: the contract, the seed streams and the two schedules."""

import pytest

from stratego.training.phase16 import contract as C
from stratego.training.phase16 import schedules as S


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def test_the_three_arms_are_the_ones_section_4_declares():
    assert [arm.arm_id for arm in C.SHOOTOUT_ARMS] == [
        "a_control",
        "b_damped",
        "c_damped_plus",
    ]
    a, b, c = C.SHOOTOUT_ARMS
    # A isolates infrastructure: the window collector on Phase 14's recipe.
    assert (a.lr_schedule, a.entropy_schedule, a.epochs, a.ema) == (
        C.LR_CONSTANT,
        C.ENTROPY_CONSTANT,
        2,
        False,
    )
    assert (a.lr_constant, a.entropy_constant) == (7.5e-5, 0.001)
    assert a.opponents == C.OPPONENTS_PHASE14_MIXTURE
    # B isolates the damping package.
    assert (b.lr_schedule, b.entropy_schedule, b.epochs, b.ema) == (
        C.LR_POWER_LAW,
        C.ENTROPY_ANNEALED,
        1,
        True,
    )
    assert b.opponents == C.OPPONENTS_PURE_CURRENT
    # C adds distribution and nothing else.
    assert c.setups == C.SETUPS_EXPANDED
    assert b.setups == C.SETUPS_LIBRARY
    assert c.to_dict() | {"arm_id": b.arm_id, "label": b.label, "setups": b.setups} == (
        b.to_dict() | {"arm_id": b.arm_id, "label": b.label}
    )


def test_arm_digests_separate_the_arms():
    digests = {arm.arm_id: arm.digest() for arm in C.SHOOTOUT_ARMS}
    assert len(set(digests.values())) == 3
    # and a digest moves when any flag moves
    assert C.ARM_B.replace(epochs=2).digest() != C.ARM_B.digest()


def test_an_arm_refuses_a_configuration_outside_the_contract():
    for change in (
        {"lr_schedule": "cosine"},
        {"entropy_schedule": "linear"},
        {"opponents": "anchor_only"},
        {"setups": "adversarial_only"},
        {"epochs": 3},
        {"population": 0},
        {"ema_decay": 1.0},
        {"window_decisions": 4},
    ):
        with pytest.raises(C.Phase16TrainingError):
            C.ARM_B.replace(**change)


def test_arm_lookup_names_the_known_arms():
    assert C.arm("b_damped") is C.ARM_B
    with pytest.raises(C.Phase16TrainingError):
        C.arm("d_wishful")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_game_ids_round_trip_and_are_phase16_only():
    identifier = C.game_id("b_damped", 7, 42)
    assert C.parse_game_id(identifier) == {"arm": "b_damped", "slot": 7, "draw": 42}
    assert str(C.TRAINING_MASTER_SEED) in identifier
    for bad in (
        "phase14_rollout_v1|ms=20260820141|ns=phase14|it=0001|b=current|g=0000",
        "phase16_game_v1|ms=1|arm=b|slot=0001|draw=000001",
        "nonsense",
    ):
        with pytest.raises(C.Phase16TrainingError):
            C.parse_game_id(bad)


def test_game_id_refuses_out_of_format_ordinals():
    for slot, draw in ((-1, 0), (10_000, 0), (0, 1_000_000)):
        with pytest.raises(C.Phase16TrainingError):
            C.game_id("b_damped", slot, draw)


def test_seed_streams_are_domain_separated_and_deterministic():
    first = C.derive_train_seed(C.DOMAIN_SETUP_SIDE, "g", "red")
    assert first == C.derive_train_seed(C.DOMAIN_SETUP_SIDE, "g", "red")
    assert first != C.derive_train_seed(C.DOMAIN_ACTION_SAMPLING, "g", "red")
    assert first != C.derive_train_seed(C.DOMAIN_SETUP_SIDE, "g", "blue")
    assert 0 <= first < (1 << 63)
    with pytest.raises(C.Phase16TrainingError):
        C.derive_train_seed("not_a_domain", "g")
    with pytest.raises(C.Phase16TrainingError):
        C.derive_train_seed(C.DOMAIN_SETUP_SIDE, "has:colon")


def test_a_phase16_seed_is_not_a_phase14_seed():
    """Overview section 6: Phase 16 draws from `phase16.agent3`, not Phase 14."""
    from stratego.training.phase14_seed import derive_seed as phase14_seed

    ours = C.derive_train_seed(C.DOMAIN_ACTION_SAMPLING, "shared", 3)
    theirs = phase14_seed("action_sampling", "shared", 3)
    assert ours != theirs


def test_uniform_from_seed_stays_in_the_unit_interval():
    values = [C.uniform_from_seed(C.derive_train_seed(C.DOMAIN_GAME_DRAW, n)) for n in range(200)]
    assert all(0.0 <= value < 1.0 for value in values)
    assert 0.35 < sum(values) / len(values) < 0.65


# ---------------------------------------------------------------------------
# The objective is inherited, not restated
# ---------------------------------------------------------------------------


def test_every_objective_constant_comes_from_the_accepted_contract():
    from stratego.training import phase9_contract as P9

    values = C.inherited_phase9_values()
    assert values["ppo_clip_epsilon"] == P9.PPO_CLIP_EPSILON == 0.20
    assert values["lambda_A"] == P9.LAMBDA_ADVANTAGE == 0.5
    assert values["lambda_V"] == P9.LAMBDA_VALUE == 0.8
    assert values["advantage_filter_quantile"] == P9.ADVANTAGE_FILTER_QUANTILE == 0.75
    assert values["advantage_filter_floor"] == P9.ADVANTAGE_FILTER_FLOOR == 0.01
    assert values["value_loss_weight"] == P9.VALUE_LOSS_WEIGHT == 0.5
    assert values["belief_loss_weight"] == P9.BELIEF_LOSS_WEIGHT == 0.25
    assert values["minibatch_size"] == P9.MINIBATCH_SIZE == 512


def test_contract_digest_is_stable_and_moves_with_the_document():
    assert C.contract_digest() == C.contract_digest()
    document = C.contract_document()
    assert document["arms"].keys() == {"a_control", "b_damped", "c_damped_plus"}
    assert document["collection"]["search"].startswith("absent")
    assert "no search in the training loop" in document["non_goals"]


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_power_law_learning_rate_matches_the_written_formula():
    value = S.power_law_learning_rate(3, lr_max=1.5e-4, lr_min=1.5e-5, exponent=1.1)
    assert value == pytest.approx(1.5e-4 * 3 ** -1.1)
    # clamped at both ends
    assert S.power_law_learning_rate(1, lr_max=1.5e-4, lr_min=1.5e-5, exponent=1.1) == 1.5e-4
    assert S.power_law_learning_rate(
        10_000, lr_max=1.5e-4, lr_min=1.5e-5, exponent=1.1
    ) == 1.5e-5


def test_annealed_entropy_matches_the_written_formula_and_floors():
    assert S.annealed_entropy(1, start=0.005, floor=0.001, exponent=0.3) == pytest.approx(0.005)
    assert S.annealed_entropy(4, start=0.005, floor=0.001, exponent=0.3) == pytest.approx(
        0.005 * 4 ** -0.3
    )
    assert S.annealed_entropy(10**6, start=0.005, floor=0.001, exponent=0.3) == 0.001


def test_the_annealed_schedule_starts_at_phase9_and_ends_at_phase14():
    """Section 2.3: Phase 14 ran the terminal floor from step 0."""
    assert S.entropy_coefficient_for(C.ARM_B, 1) == C.DEFAULT_ENTROPY_START == 0.005
    assert S.entropy_coefficient_for(C.ARM_A, 1) == C.PHASE14_CONSTANT_ENTROPY == 0.001
    assert S.entropy_coefficient_for(C.ARM_A, 500) == 0.001


def test_both_schedules_are_monotone_non_increasing():
    for n in range(1, 60):
        assert S.learning_rate_for(C.ARM_B, n) >= S.learning_rate_for(C.ARM_B, n + 1)
        assert S.entropy_coefficient_for(C.ARM_B, n) >= S.entropy_coefficient_for(C.ARM_B, n + 1)


def test_the_control_arm_holds_phase14s_two_numbers_at_every_iteration():
    for n in (1, 5, 50, 500):
        assert S.learning_rate_for(C.ARM_A, n) == C.PHASE14_CONSTANT_LR
        assert S.entropy_coefficient_for(C.ARM_A, n) == C.PHASE14_CONSTANT_ENTROPY


def test_the_schedule_index_is_the_iteration_not_the_step():
    for bad in (0, -1, True, 1.0):
        with pytest.raises(C.Phase16TrainingError):
            S.learning_rate_for(C.ARM_B, bad)
        with pytest.raises(C.Phase16TrainingError):
            S.entropy_coefficient_for(C.ARM_B, bad)


def test_schedule_curve_reports_what_the_run_config_records():
    curve = S.schedule_curve(C.ARM_B, 5)
    assert [row["iteration"] for row in curve] == [1, 2, 3, 4, 5]
    assert curve[0]["learning_rate"] == C.DEFAULT_LR_MAX
    assert all(row["lr_schedule"] == C.LR_POWER_LAW for row in curve)


# ---------------------------------------------------------------------------
# The horizon amendment (2026-08-26)
# ---------------------------------------------------------------------------


def test_the_reference_iteration_defaults_to_the_briefs_own_formula():
    """`n_ref = 1` must reproduce section 2.3 exactly, or the default lies."""
    for n in (1, 2, 5, 9, 50):
        assert S.power_law_learning_rate(
            n, lr_max=1.5e-4, lr_min=1.5e-5, exponent=1.1, reference=1
        ) == pytest.approx(
            min(max(1.5e-4 * n ** -1.1, 1.5e-5), 1.5e-4)
        )
    assert C.ARM_A.lr_reference == 1  # the control is untouched by the amendment


def test_the_amendment_moves_the_floor_from_iteration_9_to_the_end_of_the_run():
    """The defect and its fix, as one arithmetic statement."""
    unamended = C.ARM_B.replace(lr_reference=1)
    assert S.floor_iteration(unamended) == 9
    assert S.floor_iteration(C.ARM_B) == 325
    assert C.PLANNED_ITERATIONS == 313
    # the floor arrives after the run ends, not at 3% of it
    assert S.floor_iteration(C.ARM_B) > C.PLANNED_ITERATIONS
    assert S.floor_iteration(unamended) < 0.05 * C.PLANNED_ITERATIONS


def test_the_reference_is_an_eighth_of_the_planned_horizon():
    import math

    assert C.LR_REFERENCE_ITERATION == math.ceil(
        C.LR_HORIZON_FRACTION * C.PLANNED_ITERATIONS
    )
    for arm in (C.ARM_B, C.ARM_C):
        assert arm.lr_reference == C.LR_REFERENCE_ITERATION
        assert arm.planned_iterations == C.PLANNED_ITERATIONS


def test_the_amended_schedule_holds_the_maximum_then_decays_then_floors():
    arm = C.ARM_B
    assert S.learning_rate_for(arm, 1) == C.DEFAULT_LR_MAX
    assert S.learning_rate_for(arm, arm.lr_reference) == C.DEFAULT_LR_MAX
    assert S.learning_rate_for(arm, arm.lr_reference + 1) < C.DEFAULT_LR_MAX
    middle = S.learning_rate_for(arm, C.PLANNED_ITERATIONS // 2)
    assert C.DEFAULT_LR_MIN < middle < C.DEFAULT_LR_MAX
    assert S.learning_rate_for(arm, C.PLANNED_ITERATIONS) == pytest.approx(
        C.DEFAULT_LR_MIN, rel=0.1
    )


def test_the_amended_arm_is_no_longer_starved_against_the_control():
    """Mean LR over the planned run, which is what the defect actually broke."""
    horizon = range(1, C.PLANNED_ITERATIONS + 1)
    control = C.PHASE14_CONSTANT_LR
    unamended = sum(
        S.learning_rate_for(C.ARM_B.replace(lr_reference=1), n) for n in horizon
    ) / C.PLANNED_ITERATIONS
    amended = sum(S.learning_rate_for(C.ARM_B, n) for n in horizon) / C.PLANNED_ITERATIONS
    assert unamended < control / 4       # the defect: >4x below the control
    assert control / 2 < amended < control  # the fix: the same order of magnitude


def test_the_entropy_anneal_is_deliberately_not_re_horizoned():
    """It already spans most of the run and lands on the briefed floor."""
    arm = C.ARM_B
    assert S.entropy_coefficient_for(arm, 1) == C.DEFAULT_ENTROPY_START == 0.005
    reaches = next(
        n
        for n in range(1, C.PLANNED_ITERATIONS + 1)
        if S.entropy_coefficient_for(arm, n) == C.DEFAULT_ENTROPY_FLOOR
    )
    assert 0.6 < reaches / C.PLANNED_ITERATIONS < 0.75
    # The re-horizoning first proposed (n_ref_H = 13) breaks both endpoints:
    # it starts above the accepted Phase 9 level section 2.3 restores, and it
    # never reaches the terminal floor within the run.
    starts_at = 0.005 * (1 / 13) ** -0.3
    ends_at = 0.005 * (C.PLANNED_ITERATIONS / 13) ** -0.3
    assert starts_at > C.DEFAULT_ENTROPY_START
    assert ends_at > C.DEFAULT_ENTROPY_FLOOR


def test_the_amendment_is_recorded_in_the_contract_document():
    amendment = C.contract_document()["schedule_amendment"]
    assert amendment["section"].endswith("section 2.3")
    assert "n_ref" in amendment["change"]
    assert amendment["planned_iterations"] == C.PLANNED_ITERATIONS
    assert "arm A in its entirety" in amendment["unchanged"]
    assert amendment["entropy_not_re_horizoned"]
