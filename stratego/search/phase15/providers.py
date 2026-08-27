"""Phase 15 Agent 2 section 6: B18/B24 behind the accepted provider surface.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` section 6.

An adapter, not a fork
----------------------
The accepted Phase 12 engine calls a provider through exactly two names:

```text
predict_marginals(public_state)
sample_assignments(public_state, n, seed)
```

Agent 1's :class:`~stratego.belief.phase15.interface.Phase15BeliefProvider`
already implements both — including `sample_assignments`, which it exposes
specifically so a Phase 15 specialist drops into the accepted engine. What
it is not is a :class:`~stratego.search.phase12.providers.Phase12BeliefProvider`
subclass, and the engine's constructor requires one. This module is that one
missing edge: :class:`Phase15SpecialistProvider` inherits the accepted
abstract base and delegates every call, unchanged, to Agent 1's object.

Nothing is re-implemented. The marginals are Agent 1's, the sampler is
`stratego.evaluation.phase11_sampler.sample_belief_world` reached by import,
and every inventory, movement-impossibility, seed-determinism and legality
check in that accepted stack runs exactly as it did for Agent 1's own
interface checks.

The other two providers are imported outright
---------------------------------------------
`remaining_count` and `oracle` are the accepted Phase 12 classes, imported
and unmodified — the count-uniform skeleton with its full validation, and
the offline oracle with its three independent refusals. Phase 15 adds no
fourth kind of provider.

Where the oracle is refused
---------------------------
:func:`build_phase15_provider` refuses `oracle` unless `production=False`
*and* the caller passes `offline_diagnostic=True`, which is one more refusal
than Phase 12 required; the engine refuses it again at construction whenever
`config.production` is true; and the working player's mode table has no
entry that reaches this factory with the oracle name at all.
"""

from __future__ import annotations

from ..phase12.providers import (
    OracleBeliefProvider,
    Phase12BeliefProvider,
    RemainingCountBeliefProvider,
)
from .contract import (
    ALL_PROVIDERS,
    LEARNED_PROVIDERS,
    PRODUCTION_PROVIDERS,
    PROVIDER_ORACLE,
    PROVIDER_REMAINING_COUNT,
    Phase15SearchError,
)

#: The provider-layer identity a report and the frozen candidate record.
PROVIDER_VERSION = "phase15_belief_provider_v1"


class Phase15ProviderError(Phase15SearchError):
    """A Phase 15 belief provider was refused or misconfigured."""


class Phase15SpecialistProvider(Phase12BeliefProvider):
    """B18 or B24 as an accepted Phase 12 belief provider.

    Holds Agent 1's provider object and forwards both interface methods to
    it verbatim. `uses_hidden_truth` is `False` and is not settable, so the
    engine will never hand this object the privileged `GameState`.
    """

    uses_hidden_truth = False

    def __init__(self, belief_provider, *, provider_id: str, identity: "dict | None" = None):
        from ...belief.phase15.interface import Phase15BeliefProvider

        if provider_id not in LEARNED_PROVIDERS:
            raise Phase15ProviderError(
                f"a Phase 15 specialist provider is one of {list(LEARNED_PROVIDERS)}, "
                f"got {provider_id!r}"
            )
        if not isinstance(belief_provider, Phase15BeliefProvider):
            raise Phase15ProviderError(
                "Phase15SpecialistProvider wraps a Phase15BeliefProvider, got "
                f"{type(belief_provider).__name__}"
            )
        if getattr(belief_provider, "uses_hidden_truth", False):
            raise Phase15ProviderError(  # pragma: no cover - structural
                "the wrapped belief provider claims to read hidden truth"
            )
        self.provider_id = provider_id
        self.belief_provider = belief_provider
        self.identity = dict(identity or {})

    def predict_marginals(self, public) -> dict:
        return self.belief_provider.predict_marginals(public)

    def sample_assignments(self, public, n: int, seed: int) -> "list[dict[int, int]]":
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise Phase15ProviderError(f"n must be a positive int, got {n!r}")
        return self.belief_provider.sample_assignments(public, int(n), int(seed))

    def describe(self) -> dict:
        report = super().describe()
        report.update(
            {
                "provider_version": PROVIDER_VERSION,
                "adapter": self.belief_provider.describe(),
                "identity": dict(self.identity),
            }
        )
        return report


def build_specialist_provider(
    provider_id: str,
    loaded_specialist,
    move_models: dict,
    *,
    device: str = "cpu",
) -> Phase15SpecialistProvider:
    """Wrap one loaded specialist over its own frozen prefix.

    `loaded_specialist.backbone` names the model whose three-block prefix the
    specialist was fine-tuned on; that is the backbone the marginals must be
    computed with, whatever move model the search is going to use for policy
    and value.
    """
    from ...belief.phase15.interface import Phase15BeliefProvider

    backbone = move_models[loaded_specialist.backbone]
    inner = Phase15BeliefProvider(
        backbone.model,
        loaded_specialist.specialist,
        provider_id=provider_id,
        identity=dict(loaded_specialist.identity),
        device=device,
        # Agent 1 recorded `keep_calibrated: false` and an applied
        # temperature of 1.0 for both specialists; running the calibrated
        # path with T = 1.0 is the recorded configuration, and the loader
        # already refused a checkpoint whose temperature disagreed.
        calibrated=True,
    )
    return Phase15SpecialistProvider(
        inner, provider_id=provider_id, identity=dict(loaded_specialist.identity)
    )


def build_phase15_provider(
    name: str,
    models=None,
    *,
    production: bool = True,
    offline_diagnostic: bool = False,
    device: str = "cpu",
) -> Phase12BeliefProvider:
    """Build one Phase 15 belief provider by its contract name.

    `models` is a :class:`~stratego.search.phase15.loaders.Phase15Models` and
    is required for the two learned providers. With `production=True` (the
    default) the oracle name is refused outright; with `production=False` it
    is refused unless the caller *also* passes `offline_diagnostic=True`.
    """
    if name not in ALL_PROVIDERS:
        raise Phase15ProviderError(
            f"unknown belief provider {name!r}; Phase 15 provider names are "
            f"{list(ALL_PROVIDERS)}"
        )
    if name == PROVIDER_ORACLE:
        if production:
            raise Phase15ProviderError(
                "oracle is not an available belief provider in a production "
                f"configuration; production providers are {list(PRODUCTION_PROVIDERS)}"
            )
        if offline_diagnostic is not True:
            raise Phase15ProviderError(
                "the oracle is an offline diagnostic; pass offline_diagnostic=True "
                "or do not ask for it at all"
            )
        return OracleBeliefProvider(offline_diagnostic=True)
    if name == PROVIDER_REMAINING_COUNT:
        return RemainingCountBeliefProvider()
    if models is None:
        raise Phase15ProviderError(f"{name} needs the loaded Phase 15 models")
    try:
        loaded = models.specialists[name]
    except KeyError:
        raise Phase15ProviderError(f"{name} was not loaded from the handoff") from None
    return build_specialist_provider(
        name, loaded, models.move_models, device=device
    )


__all__ = [
    "OracleBeliefProvider",
    "PROVIDER_VERSION",
    "Phase12BeliefProvider",
    "Phase15ProviderError",
    "Phase15SpecialistProvider",
    "RemainingCountBeliefProvider",
    "build_phase15_provider",
    "build_specialist_provider",
]
