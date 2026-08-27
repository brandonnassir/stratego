"""Digest-bound loading, and the belief providers behind the accepted surface."""

import json

import pytest

from stratego.search.phase15.loaders import (
    Phase15LoadError,
    load_belief_specialist,
    load_handoff,
    load_move_model,
)
from stratego.search.phase15.providers import (
    Phase15ProviderError,
    Phase15SpecialistProvider,
    build_phase15_provider,
)


# -- the handoff ------------------------------------------------------------


def test_handoff_loads_and_names_every_model(models):
    handoff = models.handoff
    assert handoff["artifact"] == "phase15_search_handoff_v1"
    assert set(handoff["policy_models"]) >= {"p18", "p24"}
    assert set(handoff["belief_models"]) >= {"b18", "b24"}


def test_a_foreign_document_is_not_a_handoff(tmp_path):
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps({"artifact": "something_else"}))
    with pytest.raises(Phase15LoadError, match="is not a phase15_search_handoff_v1"):
        load_handoff(path)


def test_a_missing_handoff_is_refused(tmp_path):
    with pytest.raises(Phase15LoadError, match="no Agent 1 handoff"):
        load_handoff(tmp_path / "absent.json")


def test_an_incomplete_handoff_is_refused(tmp_path, models):
    payload = json.loads(json.dumps(models.handoff))
    del payload["belief_models"]["b24"]
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(Phase15LoadError, match="missing \\['b24'\\]"):
        load_handoff(path)


# -- digest binding ---------------------------------------------------------


def test_move_models_match_their_recorded_digests(models):
    for name in ("p18", "p24"):
        identity = models.move_models[name].identity
        record = models.handoff["policy_models"][name]
        assert identity["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert identity["model_state_digest"] == record["model_state_digest"]
        assert identity["trained_by_phase15"] is False


def test_a_wrong_sha256_refuses_the_load(models, repository_root):
    handoff = json.loads(json.dumps(models.handoff))
    handoff["policy_models"]["p18"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(Phase15LoadError, match="refusing to load unbound bytes"):
        load_move_model("p18", handoff, root=repository_root)


def test_a_wrong_state_digest_refuses_the_load(models, repository_root):
    handoff = json.loads(json.dumps(models.handoff))
    handoff["policy_models"]["p24"]["model_state_digest"] = "f" * 64
    with pytest.raises(Phase15LoadError, match="loaded model-state digest"):
        load_move_model("p24", handoff, root=repository_root)


def test_specialists_are_bound_to_their_own_prefix(models):
    assert models.specialists["b18"].backbone == "p18"
    assert models.specialists["b24"].backbone == "p24"
    for name in ("b18", "b24"):
        identity = models.specialists[name].identity
        assert identity["holds_policy_parameters"] is False
        assert identity["holds_value_parameters"] is False


def test_a_specialist_will_not_load_over_the_wrong_backbone(models, repository_root):
    """B18 over P24's prefix must be refused by the checkpoint itself."""
    from stratego.belief.phase15.checkpoint import Phase15CheckpointError, load_specialist

    path = models.specialists["b18"].identity["checkpoint_path"]
    with pytest.raises(Phase15CheckpointError, match="records source"):
        load_specialist(path, models.move_models["p24"].model)


def test_the_handoff_binding_is_cross_checked(models, repository_root):
    handoff = json.loads(json.dumps(models.handoff))
    handoff["belief_models"]["b18"]["bound_policy"] = "p24"
    with pytest.raises(Phase15LoadError, match="the contract binds it to"):
        load_belief_specialist("b18", handoff, models.move_models, root=repository_root)


def test_a_disagreeing_temperature_refuses_the_load(models, repository_root):
    handoff = json.loads(json.dumps(models.handoff))
    handoff["belief_models"]["b24"]["calibration"]["applied_temperature"] = 2.5
    with pytest.raises(Phase15LoadError, match="applied temperature"):
        load_belief_specialist("b24", handoff, models.move_models, root=repository_root)


def test_the_recorded_temperature_is_the_one_agent_1_applied(models):
    for name in ("b18", "b24"):
        identity = models.specialists[name].identity
        record = models.handoff["belief_models"][name]["calibration"]
        assert identity["applied_temperature"] == record["applied_temperature"]
        assert identity["keep_calibrated"] == record["keep_calibrated"]


# -- providers --------------------------------------------------------------


def test_every_production_provider_builds(models):
    for name in ("remaining_count", "b18", "b24"):
        provider = build_phase15_provider(name, models)
        assert provider.provider_id == name
        assert provider.uses_hidden_truth is False


def test_a_specialist_provider_is_an_accepted_phase12_provider(models):
    from stratego.search.phase12.providers import Phase12BeliefProvider

    provider = build_phase15_provider("b18", models)
    assert isinstance(provider, Phase12BeliefProvider)
    assert isinstance(provider, Phase15SpecialistProvider)


def test_the_oracle_is_refused_in_production(models):
    with pytest.raises(Phase15ProviderError, match="not an available belief provider"):
        build_phase15_provider("oracle", models)


def test_the_oracle_needs_both_switches(models):
    with pytest.raises(Phase15ProviderError, match="offline diagnostic"):
        build_phase15_provider("oracle", models, production=False)
    provider = build_phase15_provider(
        "oracle", models, production=False, offline_diagnostic=True
    )
    assert provider.uses_hidden_truth is True


def test_an_unknown_provider_name_is_refused(models):
    with pytest.raises(Phase15ProviderError, match="unknown belief provider"):
        build_phase15_provider("b99", models)


def test_a_learned_provider_needs_the_models():
    with pytest.raises(Phase15ProviderError, match="needs the loaded Phase 15 models"):
        build_phase15_provider("b18", None)


def test_the_adapter_refuses_a_foreign_object():
    with pytest.raises(Phase15ProviderError, match="wraps a Phase15BeliefProvider"):
        Phase15SpecialistProvider(object(), provider_id="b18")


def test_the_adapter_refuses_a_non_specialist_id(models):
    provider = build_phase15_provider("b18", models)
    with pytest.raises(Phase15ProviderError, match="a Phase 15 specialist provider"):
        Phase15SpecialistProvider(provider.belief_provider, provider_id="remaining_count")


# -- provider behaviour on a real position ----------------------------------


def test_marginals_are_probability_vectors_over_the_hidden_pieces(models, midgame_state):
    from tests.search.conftest import public_state_for

    public = public_state_for(midgame_state)
    for name in ("b18", "b24"):
        marginals = build_phase15_provider(name, models).predict_marginals(public)
        assert marginals
        for row in marginals.values():
            assert row.shape == (12,)
            assert abs(float(row.sum()) - 1.0) < 1e-9
            assert (row >= 0).all()


def test_the_same_seed_reproduces_the_same_worlds(models, midgame_state):
    from tests.search.conftest import public_state_for

    public = public_state_for(midgame_state)
    provider = build_phase15_provider("b24", models)
    first = provider.sample_assignments(public, 8, 11)
    assert first == provider.sample_assignments(public, 8, 11)
    assert first != provider.sample_assignments(public, 8, 12)


def test_a_bad_world_count_is_refused(models, midgame_state):
    from tests.search.conftest import public_state_for

    provider = build_phase15_provider("b18", models)
    public = public_state_for(midgame_state)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(Phase15ProviderError, match="positive int"):
            provider.sample_assignments(public, bad, 1)


def test_learned_and_count_providers_disagree_about_worlds(models, midgame_state):
    """If they agreed everywhere the belief model would not be consulted."""
    from tests.search.conftest import public_state_for

    public = public_state_for(midgame_state)
    learned = build_phase15_provider("b18", models).sample_assignments(public, 16, 5)
    count = build_phase15_provider("remaining_count", models).sample_assignments(public, 16, 5)
    assert learned != count
