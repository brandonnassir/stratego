"""Phase 15 Agent 2 section 4: building one complete system.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 4, 5, 7.

The role table, made mechanical
-------------------------------
```text
P18 or P24     root policy, candidate prior, rollout policy for both sides,
               leaf value, direct fallback
B18 or B24     hidden-rank marginals, legal hidden-world sampling
```

:func:`build_engine` is the only place a Phase 15 engine is constructed, and
it takes the move model from `models.move_models[pairing.move_model]` and
the provider from `pairing.provider` — two different lookups that cannot be
crossed. The belief specialist's fine-tuned block never reaches the engine at
all: the engine holds a `StrategoModel`, and a `Phase15BeliefSpecialist` is
not one.

Cross-pairing is intentional
----------------------------
`P18 + B24` builds B24's marginals over **P24's** frozen prefix — the only
backbone its checkpoint will load against — and runs the search's policy,
value and rollouts on **P18**. Two models are held; each does exactly one
job. :meth:`SystemBundle.describe` records both so a report can never be read
as if one model did both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..phase12.contract import Phase12SearchConfig
from ..phase12.engine import Phase12SearchEngine
from .contract import (
    LEARNED_PROVIDERS,
    PROVIDER_ORACLE,
    Pairing,
    Phase15SearchError,
    pairing as pairing_of,
    preset as preset_of,
)
from .matchplay import DirectSeat, SearchSeat
from .providers import build_phase15_provider

#: The system-assembly identity.
SYSTEM_VERSION = "phase15_complete_system_v1"


class Phase15SystemError(Phase15SearchError):
    """A complete system could not be assembled."""


@dataclass
class SystemBundle:
    """One assembled complete system, plus everything a report needs."""

    pairing: Pairing
    config: "Phase12SearchConfig | None"
    engine: "Phase12SearchEngine | None"
    provider: object = None
    identities: dict = field(default_factory=dict)

    @property
    def pairing_id(self) -> str:
        return self.pairing.pairing_id

    def describe(self) -> dict:
        report = {
            "system_version": SYSTEM_VERSION,
            "pairing": self.pairing.describe(),
            "move_model_identity": dict(self.identities.get("move_model") or {}),
            "belief_model_identity": dict(self.identities.get("belief_model") or {}),
            "roles": {
                "policy": self.pairing.move_model,
                "value": self.pairing.move_model,
                "rollout_policy_both_sides": self.pairing.move_model,
                "direct_fallback": self.pairing.move_model,
                "hidden_rank_marginals": self.pairing.provider,
                "hidden_world_sampling": self.pairing.provider,
            },
        }
        if self.engine is not None:
            report["engine"] = self.engine.describe()
        return report


def build_engine(
    pairing: "Pairing | str",
    models,
    preset: "Phase12SearchConfig | str" = "TINY",
    *,
    production: bool = True,
    device: str = "cpu",
) -> SystemBundle:
    """Assemble one search system. Direct pairings return an engine-free bundle.

    `production=True` refuses the oracle three times over: the provider
    factory refuses the name, the engine refuses a hidden-truth provider, and
    the config carries `production=True` into both.
    """
    if isinstance(pairing, str):
        pairing = pairing_of(pairing)
    if pairing.kind == "direct":
        move = models.move_models[pairing.move_model]
        return SystemBundle(
            pairing=pairing,
            config=None,
            engine=None,
            provider=None,
            identities={"move_model": dict(move.identity), "belief_model": {}},
        )
    if isinstance(preset, str):
        preset = preset_of(preset)
    if not isinstance(preset, Phase12SearchConfig):
        raise Phase15SystemError(
            f"preset must be a name or a Phase12SearchConfig, got {type(preset).__name__}"
        )
    wants_oracle = pairing.provider == PROVIDER_ORACLE
    if wants_oracle and production:
        raise Phase15SystemError(
            f"{pairing.pairing_id} is an offline diagnostic; build it with "
            "production=False or do not build it at all"
        )
    from dataclasses import replace

    config = replace(preset, production=bool(production))
    provider = build_phase15_provider(
        pairing.provider,
        models,
        production=production,
        offline_diagnostic=wants_oracle,
        device=device,
    )
    move = models.move_models[pairing.move_model]
    engine = Phase12SearchEngine(
        move.model,
        provider,
        config,
        device=device,
        model_identity=dict(move.identity),
    )
    belief_identity: dict = {}
    if pairing.provider in LEARNED_PROVIDERS:
        belief_identity = dict(models.specialists[pairing.provider].identity)
    elif not wants_oracle:
        belief_identity = {"provider_id": pairing.provider, "learned": False}
    return SystemBundle(
        pairing=pairing,
        config=config,
        engine=engine,
        provider=provider,
        identities={
            "move_model": dict(move.identity),
            "belief_model": belief_identity,
        },
    )


def build_seat(
    bundle: SystemBundle, owners: dict, *, time_cap: "float | None" = None
):
    """The match seat for one assembled system."""
    if bundle.pairing.kind == "direct":
        return DirectSeat(bundle.pairing, owners)
    return SearchSeat(bundle.pairing, bundle.engine, owners=owners, time_cap=time_cap)


def build_systems(
    pairing_ids,
    models,
    preset: "Phase12SearchConfig | str" = "TINY",
    *,
    production: bool = True,
    device: str = "cpu",
) -> "dict[str, SystemBundle]":
    """Assemble several systems at once, keyed by pairing id.

    A diagnostic pairing in the list forces `production=False` for *that*
    system only; the production arms keep their own production configuration,
    so one offline diagnostic can never relax another arm's refusal.
    """
    built: dict[str, SystemBundle] = {}
    for pairing_id in pairing_ids:
        target = pairing_of(pairing_id) if isinstance(pairing_id, str) else pairing_id
        diagnostic = target.kind == "diagnostic"
        if diagnostic and production:
            raise Phase15SystemError(
                f"{target.pairing_id} may not be built under a production "
                "configuration; ask for it explicitly with production=False"
            )
        built[target.pairing_id] = build_engine(
            target, models, preset, production=not diagnostic, device=device
        )
    return built


__all__ = [
    "SYSTEM_VERSION",
    "Phase15SystemError",
    "SystemBundle",
    "build_engine",
    "build_seat",
    "build_systems",
]
