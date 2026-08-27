"""Shared fixtures for the Phase 15 Agent 2 search tests.

Two kinds of fixture, on purpose.

The *hermetic* ones build a randomly initialized C1 and a randomly
initialized belief specialist over it, so the mechanics — legality,
determinism, role separation, fallback, refusals — are tested for any
weights and the tests stay fast.

The *bound* ones load the real frozen Phase 15 stack from the Agent 1
handoff. They are what proves the digests, the calibration binding and the
loader refusals on the actual delivered bytes, and they skip cleanly when
the handoff or its checkpoints are not present.
"""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repository_root():
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def handoff_available():
    from stratego.search.phase15.loaders import DEFAULT_HANDOFF_PATH

    path = REPOSITORY_ROOT / DEFAULT_HANDOFF_PATH
    if not path.is_file():
        return False
    import json

    handoff = json.loads(path.read_text())
    for section, keys in (
        ("policy_models", ("p18", "p24")),
        ("belief_models", ("b18", "b24")),
    ):
        for key in keys:
            record = (handoff.get(section) or {}).get(key)
            if record is None:
                return False
            if not (REPOSITORY_ROOT / record["checkpoint_path"]).is_file():
                return False
    return True


@pytest.fixture(scope="session")
def models(handoff_available):
    """The real frozen Phase 15 stack, loaded once."""
    if not handoff_available:
        pytest.skip("the Phase 15 Agent 1 handoff or its checkpoints are not present")
    from stratego.search.phase15.loaders import load_all

    return load_all(root=REPOSITORY_ROOT, device="cpu", with_anchor=False)


@pytest.fixture(scope="session")
def setup_sources():
    from stratego.search.phase15.boards import Phase15MatchSetupSources

    return Phase15MatchSetupSources()


@pytest.fixture(scope="session")
def fake_models():
    """A hermetic stand-in for `Phase15Models`, with random weights.

    Same shape as the real object — two move models, two specialists, the
    same identity keys — so every code path that reads identities works, but
    nothing is loaded from disk and no digest is real.
    """
    from stratego.belief.phase15.heads import Phase15BeliefSpecialist
    from stratego.model.production_model import ProductionModel
    from stratego.search.phase15.loaders import LoadedMoveModel, LoadedSpecialist, Phase15Models
    from stratego.training.phase9_behavior import state_dict_digest

    move_models = {}
    specialists = {}
    for name, specialist_id in (("p18", "b18"), ("p24", "b24")):
        model = ProductionModel("C1")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        digest = state_dict_digest(model)
        move_models[name] = LoadedMoveModel(
            move_model=name,
            model=model,
            identity={
                "move_model": name,
                "logical_identity": name.upper(),
                "checkpoint_path": f"<hermetic {name}>",
                "checkpoint_sha256": "0" * 64,
                "model_state_digest": digest,
                "global_optimizer_step": 0,
                "phase14_candidate_hour": int(name[1:]),
                "phase14_archive_sha256": "0" * 64,
                "architecture_id": getattr(model, "architecture_id", None),
                "role": "policy, value, rollout policy for both sides, direct fallback",
                "trained_by_phase15": False,
            },
        )
        specialist = Phase15BeliefSpecialist.from_policy(model, specialist_id=specialist_id)
        specialist.eval()
        for parameter in specialist.parameters():
            parameter.requires_grad_(False)
        specialists[specialist_id] = LoadedSpecialist(
            provider_id=specialist_id,
            specialist=specialist,
            backbone=name,
            identity={
                "provider_id": specialist_id,
                "checkpoint_path": f"<hermetic {specialist_id}>",
                "checkpoint_sha256": "0" * 64,
                "state_digest": "0" * 64,
                "architecture_version": "phase15_belief_specialist_v1",
                "prefix_backbone": name,
                "prefix_backbone_state_digest": digest,
                "applied_temperature": 1.0,
                "fitted_temperature": 1.0,
                "keep_calibrated": False,
                "holds_policy_parameters": False,
                "holds_value_parameters": False,
                "corpus": {},
                "role": "hidden-rank marginals and legal hidden-world sampling only",
            },
        )
    return Phase15Models(
        handoff={
            "artifact": "phase15_search_handoff_v1",
            "policy_models": {
                name: loaded.identity for name, loaded in move_models.items()
            },
            "belief_models": {
                name: loaded.identity for name, loaded in specialists.items()
            },
            "corpus": {"corpus_digest": "0" * 64, "corpus_version": "hermetic"},
        },
        handoff_path="<hermetic>",
        move_models=move_models,
        specialists=specialists,
    )


@pytest.fixture(scope="session")
def midgame_state():
    from tests.helpers import nonterminal_state

    return nonterminal_state(24, first_seed=3)
