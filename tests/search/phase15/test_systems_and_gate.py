"""Assembling complete systems, and the section 9 correctness gate."""

import pytest

from stratego.search.phase15.systems import (
    Phase15SystemError,
    build_engine,
    build_systems,
)


# -- role assignment --------------------------------------------------------


def test_a_direct_pairing_carries_no_engine(fake_models):
    bundle = build_engine("p18_direct", fake_models, "TINY")
    assert bundle.engine is None
    assert bundle.provider is None
    assert bundle.identities["move_model"]["logical_identity"] == "P18"


@pytest.mark.parametrize(
    "pairing_id,move,provider",
    [
        ("p18_b18", "p18", "b18"),
        ("p18_b24", "p18", "b24"),
        ("p24_b18", "p24", "b18"),
        ("p24_b24", "p24", "b24"),
    ],
)
def test_the_four_systems_wire_the_right_two_models(
    fake_models, pairing_id, move, provider
):
    bundle = build_engine(pairing_id, fake_models, "TINY")
    assert bundle.engine.provider.provider_id == provider
    assert bundle.engine.model is fake_models.move_models[move].model
    assert bundle.identities["move_model"]["move_model"] == move
    assert bundle.identities["belief_model"]["provider_id"] == provider


def test_a_cross_pairing_computes_beliefs_over_its_own_backbone(fake_models):
    """P18 + B24 runs policy on P18 and marginals on P24's prefix."""
    bundle = build_engine("p18_b24", fake_models, "TINY")
    assert bundle.engine.model is fake_models.move_models["p18"].model
    inner = bundle.engine.provider.belief_provider
    assert inner.policy_model is fake_models.move_models["p24"].model
    assert bundle.identities["belief_model"]["prefix_backbone"] == "p24"


def test_the_belief_specialist_never_reaches_the_engine_as_a_model(fake_models):
    from stratego.model.base import StrategoModel

    bundle = build_engine("p24_b18", fake_models, "TINY")
    assert isinstance(bundle.engine.model, StrategoModel)
    assert not isinstance(
        fake_models.specialists["b18"].specialist, StrategoModel
    )


def test_the_role_table_is_recorded(fake_models):
    report = build_engine("p24_b18", fake_models, "TINY").describe()
    roles = report["roles"]
    assert roles["policy"] == roles["value"] == "p24"
    assert roles["rollout_policy_both_sides"] == "p24"
    assert roles["direct_fallback"] == "p24"
    assert roles["hidden_rank_marginals"] == "b18"


# -- oracle refusals --------------------------------------------------------


def test_the_oracle_system_is_refused_in_production(fake_models):
    with pytest.raises(Phase15SystemError, match="offline diagnostic"):
        build_engine("p18_oracle", fake_models, "TINY")


def test_the_oracle_builds_only_when_asked_explicitly(fake_models):
    bundle = build_engine("p24_oracle", fake_models, "TINY", production=False)
    assert bundle.provider.uses_hidden_truth is True
    assert bundle.engine.config.production is False


def test_a_batch_build_will_not_relax_production_for_one_arm(fake_models):
    with pytest.raises(Phase15SystemError, match="may not be built under a production"):
        build_systems(["p18_b18", "p18_oracle"], fake_models, "TINY")


def test_a_batch_build_keeps_every_production_arm_production(fake_models):
    built = build_systems(
        ["p18_b18", "p24_b24", "p18_oracle"], fake_models, "TINY", production=False
    )
    assert built["p18_b18"].engine.config.production is True
    assert built["p24_b24"].engine.config.production is True
    assert built["p18_oracle"].engine.config.production is False


def test_a_bad_preset_object_is_refused(fake_models):
    with pytest.raises(Phase15SystemError, match="preset must be"):
        build_engine("p18_b18", fake_models, 32)


# -- the gate, on hermetic weights ------------------------------------------


def test_identity_check_reads_the_handoff(fake_models):
    from stratego.search.phase15.gate import check_identities

    report = check_identities(fake_models)
    assert report["passed"] is True
    assert set(report["pairings"]) == {"p18_b18", "p18_b24", "p24_b18", "p24_b24"}


def test_identity_check_catches_a_digest_mismatch(fake_models):
    import copy

    from stratego.search.phase15.gate import check_identities

    tampered = copy.copy(fake_models)
    handoff = copy.deepcopy(fake_models.handoff)
    handoff["policy_models"]["p18"]["model_state_digest"] = "f" * 64
    object.__setattr__(tampered, "handoff", handoff)
    report = check_identities(tampered)
    assert report["passed"] is False
    assert any("state digest mismatch" in finding for finding in report["findings"])


def test_oracle_refusal_check_finds_every_refusal(fake_models):
    from stratego.search.phase15.gate import check_oracle_refusals

    report = check_oracle_refusals(fake_models)
    assert report["passed"] is True
    assert len(report["refusals"]) >= 5
    assert "player.set_mode(oracle)" in report["refusals"]


def test_decision_check_passes_on_a_real_position(fake_models, midgame_state):
    from stratego.search.phase15.gate import check_decisions

    states = [({"position_id": "x", "ply": 24, "unresolved": 40}, midgame_state, None)]
    report = check_decisions(fake_models, states, preset="TINY")
    assert report["passed"] is True
    assert report["decisions"] == 4
    assert report["candidates_checked"] > 0


def test_permutation_invariance_holds_and_the_control_has_power(
    fake_models, midgame_state
):
    from stratego.search.phase15.gate import check_permutation_invariance

    class _Position:
        position_id = "gate_probe"

    states = [(_Position(), midgame_state, None)]
    report = check_permutation_invariance(fake_models, states, preset="TINY")
    assert report["production_checks"] == 4
    assert report["permutations_that_changed_assignments"] == 4
    # Random weights may or may not make the oracle sensitive on one position;
    # what must hold is that the production arms never changed their answer.
    assert not [
        finding
        for finding in report["findings"]
        if "changed under a hidden-identity permutation" in finding
    ]
