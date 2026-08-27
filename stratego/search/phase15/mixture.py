"""Phase 15 follow-up: mixing B24 with the robust remaining-count belief.

One narrow question, asked of one system:

> The deeper-search pilot found that LARGE and XLARGE both came out
> **worse** than MEDIUM (-0.075 paired EWR each) while the *oracle* at the
> same budgets got **better** (+0.042 at LARGE). That points at the sampled
> world distribution rather than the search mechanics. Can a mixture of the
> learned B24 marginals with the robust count marginals,
>
> .. math:: b_{mix} = \\lambda\\, b_{B24} + (1 - \\lambda)\\, b_{count}
>
> recover the deeper rung's regression?

What is held fixed
------------------
Everything except the marginal vector handed to the sampler. The P24 move
model, its digest, the B24 specialist, its digest and its applied
temperature, the candidate rule, `beta`, `epsilon`, world deduplication,
every per-decision seed, the legal-world sampler and the search engine are
the frozen objects, reached by import. This module adds one provider and
nothing else: it does not touch :mod:`~stratego.search.phase12.engine`,
:mod:`~stratego.search.phase15.contract` or any accepted preset.

Why the pairing is duck-typed rather than registered
-----------------------------------------------------
:class:`~stratego.search.phase15.contract.Pairing` refuses a provider name
that is not in the frozen `ALL_PROVIDERS` table, and widening that table
would change what every other Phase 15 module and test sees. A pilot may
not do that. :class:`MixturePairing` carries the four fields the match seat
actually reads — `pairing_id`, `move_model`, `provider`, `kind` — so the
accepted :class:`~stratego.search.phase15.matchplay.SearchSeat` accepts it
without the frozen table growing an entry.

The two endpoints are not what their names suggest
---------------------------------------------------
`lambda = 1.00` is B24's own marginals through B24's own sampler path, and
is expected to reproduce the frozen `p24_b24` arm decision for decision;
:func:`stage1_positions` measures that rather than assuming it.

`lambda = 0.00` is **not** the accepted `remaining_count` provider. That
provider draws from the count-uniform skeleton
(`weight = remaining_count`); a mixture at `lambda = 0` feeds the count
marginals to the *learned* sampler, whose weighting is
`learned_probability * remaining_count` — so its effective weight is
proportional to `count^2` and it concentrates harder on the plentiful
ranks than the baseline does. Keeping one sampler across the whole sweep is
what makes the sweep a measurement of `lambda`; the accepted count provider
is therefore carried alongside as its own reference arm, and the two are
reported separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ...belief.phase15.interface import Phase15PublicState
from ...belief.phase15.seeds import DOMAIN_INTERFACE, derive_phase15_seed
from ...evaluation.phase11_sampler import Phase11SamplerRequest, sample_belief_world
from ...training.phase11_contract import BELIEF_SAMPLER_VERSION, RANK_COUNT
from ...training.phase11_seed import MAX_SAMPLE_ORDINAL_FORMAT
from ..phase12.contract import Phase12SearchConfig
from ..phase12.engine import Phase12SearchEngine
from ..phase12.providers import Phase12BeliefProvider, RemainingCountBeliefProvider
from .contract import Phase15SearchError, preset as preset_of
from .providers import build_specialist_provider
from .systems import SystemBundle

#: The pilot identity every artifact records.
MIXTURE_VERSION = "phase15_belief_mixture_pilot_v1"

#: The five instructed mixture weights, count-only first.
MIXTURE_LAMBDAS = (0.00, 0.25, 0.50, 0.75, 1.00)

#: The one system the mixture is defined over, and its two roles.
MIXTURE_MOVE_MODEL = "p24"
MIXTURE_LEARNED_PROVIDER = "b24"

#: Stage 1 asks its question at the deeper rung only; MEDIUM is carried as
#: the incumbent reference, not as a rung of the sweep.
MIXTURE_STAGE1_PRESET = "LARGE"
MIXTURE_REFERENCE_PRESET = "MEDIUM"

#: The fixed per-position seed. The same constant the deeper-search pilot's
#: own decision-divergence measurement used, so the two are comparable.
MIXTURE_DECISION_SEED = 20260824

#: How much of LARGE's -0.075 paired regression a mixture must give back
#: before it is worth adopting. Half of it, and no less: a pilot this small
#: cannot resolve a quarter of an EWR-point of anything.
MIXTURE_RECOVERY_FRACTION = 0.5


class Phase15MixtureError(Phase15SearchError):
    """A belief mixture was refused, or a mixture invariant was violated."""


def lambda_token(lam: float) -> str:
    """`0.25 -> 'l025'`. A filename-safe, sortable name for one weight."""
    value = float(lam)
    if not 0.0 <= value <= 1.0:
        raise Phase15MixtureError(f"lambda must lie in [0, 1], got {lam!r}")
    scaled = round(value * 100.0)
    if abs(value * 100.0 - scaled) > 1e-9:
        raise Phase15MixtureError(
            f"lambda {lam!r} is not expressible in whole percent; the pilot's "
            "grid is deliberately coarse"
        )
    return f"l{int(scaled):03d}"


def mixture_provider_id(lam: float) -> str:
    return f"{MIXTURE_LEARNED_PROVIDER}_count_mix_{lambda_token(lam)}"


def mixture_arm_id(lam: float) -> str:
    return f"{MIXTURE_MOVE_MODEL}_mix_{lambda_token(lam)}"


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


class MixtureBeliefProvider(Phase12BeliefProvider):
    """`lambda * B24 + (1 - lambda) * count`, normalized, then sampled.

    Reads a :class:`Phase15PublicState` and nothing else, exactly like the
    two providers it combines, so it cannot see a hidden rank. The sampler
    is the accepted `sample_belief_world`, reached by import, with the
    accepted ordinal derivation: at `lambda = 1` the ordinals, the sampler
    and the marginals are all B24's, so the worlds are B24's worlds.
    """

    uses_hidden_truth = False

    def __init__(self, learned, count, *, lam: float, identity: "dict | None" = None):
        self.lam = float(lam)
        if not 0.0 <= self.lam <= 1.0:
            raise Phase15MixtureError(f"lambda must lie in [0, 1], got {lam!r}")
        if getattr(learned, "uses_hidden_truth", False) or getattr(
            count, "uses_hidden_truth", False
        ):
            raise Phase15MixtureError(
                "a mixture component claims to read hidden truth"
            )
        for name, component in (("learned", learned), ("count", count)):
            if not hasattr(component, "predict_marginals"):
                raise Phase15MixtureError(
                    f"the {name} component has no predict_marginals"
                )
        self.learned = learned
        self.count = count
        self.provider_id = mixture_provider_id(self.lam)
        self.identity = dict(identity or {})

    # -- the mixture ------------------------------------------------------

    def predict_marginals(self, public) -> dict:
        if not isinstance(public, Phase15PublicState):
            raise Phase15MixtureError(
                "predict_marginals accepts only a Phase15PublicState, got "
                f"{type(public).__name__}"
            )
        learned = self.learned.predict_marginals(public)
        counts = self.count.predict_marginals(public)
        if set(learned) != set(counts):
            raise Phase15MixtureError(
                "the learned and count marginals cover different hidden pieces: "
                f"{sorted(set(learned) ^ set(counts))[:6]}"
            )
        lam = self.lam
        mixed: dict[int, np.ndarray] = {}
        for slot, learned_row in learned.items():
            left = np.asarray(learned_row, dtype=np.float64)
            right = np.asarray(counts[slot], dtype=np.float64)
            if left.shape != (RANK_COUNT,) or right.shape != (RANK_COUNT,):
                raise Phase15MixtureError(
                    f"slot {slot}: a component marginal is not a {RANK_COUNT}-vector"
                )
            row = lam * left + (1.0 - lam) * right
            total = float(row.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise Phase15MixtureError(
                    f"slot {slot}: the mixed marginal has no positive mass"
                )
            # Instructed explicitly, and exact at `total == 1.0`: IEEE-754
            # division by one is the identity, so the lambda = 1 endpoint is
            # not perturbed by normalizing something already normalized.
            row = row / total
            if not np.isfinite(row).all() or (row < 0.0).any():
                raise Phase15MixtureError(
                    f"slot {slot}: the normalized mixture is not a distribution"
                )
            mixed[int(slot)] = row
        return mixed

    # -- the accepted sampler, unchanged ----------------------------------

    def sample_assignments(self, public, n: int, seed: int) -> "list[dict[int, int]]":
        """`n` legal assignments from the mixed marginals.

        The ordinal walk is
        :meth:`stratego.belief.phase15.interface.Phase15BeliefProvider.sample_worlds`'s
        walk, restated over the mixed marginals rather than reached through
        a method that would recompute B24's. Same domain, same label, same
        modulus, so `lambda = 1` lands on the same ordinals B24 would.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise Phase15MixtureError(f"n must be a positive int, got {n!r}")
        marginals = self.predict_marginals(public)
        start = derive_phase15_seed(DOMAIN_INTERFACE, "world", int(seed)) % (
            MAX_SAMPLE_ORDINAL_FORMAT + 1
        )
        assignments = []
        for offset in range(int(n)):
            ordinal = (start + offset) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
            world = sample_belief_world(
                Phase11SamplerRequest(
                    sampler_version=BELIEF_SAMPLER_VERSION,
                    public_state_document=public.public_state_document,
                    learned_probabilities=marginals,
                    sample_ordinal=int(ordinal),
                )
            )
            assignments.append(
                {int(slot): int(rank) for slot, rank in world["assignment"].items()}
            )
        return assignments

    def describe(self) -> dict:
        report = super().describe()
        report.update(
            {
                "mixture_version": MIXTURE_VERSION,
                "lambda": self.lam,
                "components": {
                    "learned": getattr(self.learned, "provider_id", None),
                    "count": getattr(self.count, "provider_id", None),
                },
                "rule": "normalize(lambda * b_b24 + (1 - lambda) * b_count)",
                "sampler": BELIEF_SAMPLER_VERSION,
                "sampler_source": (
                    "stratego.evaluation.phase11_sampler (accepted, unmodified, "
                    "by import)"
                ),
                "identity": dict(self.identity),
            }
        )
        return report


# ---------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixturePairing:
    """A mixture arm, shaped like the frozen `Pairing` the match seat reads.

    Deliberately *not* registered in the frozen pairing table: see the module
    docstring. It carries `describe()` so a report records the mixture the
    same way it records an accepted arm.
    """

    pairing_id: str
    move_model: str
    provider: str
    lam: float
    kind: str = "search"
    description: str = ""

    @property
    def is_learned(self) -> bool:
        return True

    def describe(self) -> dict:
        return {
            "pairing_id": self.pairing_id,
            "move_model": self.move_model,
            "provider": self.provider,
            "kind": self.kind,
            "is_learned_belief": True,
            "lambda": self.lam,
            "description": self.description,
        }


def mixture_pairing(lam: float) -> MixturePairing:
    return MixturePairing(
        pairing_id=mixture_arm_id(lam),
        move_model=MIXTURE_MOVE_MODEL,
        provider=mixture_provider_id(lam),
        lam=float(lam),
        description=(
            f"P24 + search over {lam:.2f}*B24 + {1.0 - float(lam):.2f}*remaining_count"
        ),
    )


def build_mixture_bundle(
    models,
    lam: float,
    preset: "Phase12SearchConfig | str" = MIXTURE_STAGE1_PRESET,
    *,
    device: str = "cpu",
) -> SystemBundle:
    """One assembled mixture system, over the frozen P24/B24 bytes."""
    if isinstance(preset, str):
        preset = preset_of(preset)
    if not isinstance(preset, Phase12SearchConfig):
        raise Phase15MixtureError(
            f"preset must be a name or a Phase12SearchConfig, got {type(preset).__name__}"
        )
    loaded = models.specialists[MIXTURE_LEARNED_PROVIDER]
    learned = build_specialist_provider(
        MIXTURE_LEARNED_PROVIDER, loaded, models.move_models, device=device
    )
    provider = MixtureBeliefProvider(
        learned,
        RemainingCountBeliefProvider(),
        lam=lam,
        identity=dict(loaded.identity),
    )
    move = models.move_models[MIXTURE_MOVE_MODEL]
    # `production=True`: a mixture is a production-shaped arm, and the engine
    # refuses a hidden-truth provider under it.
    config = replace(preset, production=True)
    engine = Phase12SearchEngine(
        move.model,
        provider,
        config,
        device=device,
        model_identity=dict(move.identity),
    )
    return SystemBundle(
        pairing=mixture_pairing(lam),
        config=config,
        engine=engine,
        provider=provider,
        identities={
            "move_model": dict(move.identity),
            "belief_model": dict(loaded.identity),
        },
    )


# ---------------------------------------------------------------------------
# Control checks
# ---------------------------------------------------------------------------


def check_frozen_identity(models, candidate: dict) -> dict:
    """The pilot runs on the frozen candidate's exact bytes, or not at all.

    The same four digests and the same temperature the deeper-search pilot
    bound itself to, restated here so this pilot is checkable on its own.
    """
    move = models.move_models[MIXTURE_MOVE_MODEL].identity
    belief = models.specialists[MIXTURE_LEARNED_PROVIDER].identity
    findings = []
    checks = {
        "move_model_checkpoint_sha256": (
            move["checkpoint_sha256"],
            candidate["move_model"]["checkpoint_sha256"],
        ),
        "move_model_state_digest": (
            move["model_state_digest"],
            candidate["move_model"]["model_state_digest"],
        ),
        "belief_checkpoint_sha256": (
            belief["checkpoint_sha256"],
            candidate["belief_model"]["checkpoint_sha256"],
        ),
        "belief_state_digest": (
            belief["state_digest"],
            candidate["belief_model"]["state_digest"],
        ),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            findings.append(f"{name}: loaded {observed} != frozen {expected}")
    if float(belief["applied_temperature"]) != float(
        candidate["belief_calibration"]["applied_temperature"]
    ):
        findings.append("the applied belief temperature differs from the frozen record")
    return {
        "passed": not findings,
        "findings": findings,
        "move_model_state_digest": move["model_state_digest"],
        "belief_state_digest": belief["state_digest"],
        "applied_temperature": belief["applied_temperature"],
    }


def check_configuration_invariants(
    presets=(MIXTURE_REFERENCE_PRESET, MIXTURE_STAGE1_PRESET),
) -> dict:
    """Only the marginal vector may differ. The budgets are the frozen ones."""
    baseline = preset_of(MIXTURE_REFERENCE_PRESET)
    findings = []
    rows = {}
    for name in presets:
        config = preset_of(name)
        for field in ("max_root_candidates", "beta", "epsilon", "deduplicate_worlds"):
            if getattr(config, field) != getattr(baseline, field):
                findings.append(
                    f"{name}: {field} is {getattr(config, field)!r}, MEDIUM's is "
                    f"{getattr(baseline, field)!r}; the pilot may not change it"
                )
        rows[name] = {
            "worlds": config.worlds,
            "rollout_depth": config.rollout_depth,
            "max_root_candidates": config.max_root_candidates,
            "beta": config.beta,
            "epsilon": config.epsilon,
            "deduplicate_worlds": config.deduplicate_worlds,
        }
    return {"passed": not findings, "findings": findings, "presets": rows}


def check_mixture_algebra(models, states, *, lambdas=MIXTURE_LAMBDAS) -> dict:
    """The mixture is the stated formula, and its endpoints are the components.

    Checked on real public states rather than on synthetic vectors: the
    thing that can go wrong is the two component providers disagreeing about
    which pieces are hidden, which only a real document exposes.
    """
    from .gate import _public_state

    loaded = models.specialists[MIXTURE_LEARNED_PROVIDER]
    learned = build_specialist_provider(
        MIXTURE_LEARNED_PROVIDER, loaded, models.move_models
    )
    counts = RemainingCountBeliefProvider()
    providers = {lam: MixtureBeliefProvider(learned, counts, lam=lam) for lam in lambdas}

    findings = []
    checked = 0
    worst_endpoint_error = 0.0
    for _row, state, _plan in states:
        public = _public_state(state)
        left = learned.predict_marginals(public)
        right = counts.predict_marginals(public)
        for lam, provider in providers.items():
            mixed = provider.predict_marginals(public)
            if set(mixed) != set(left):
                findings.append(f"lambda={lam}: the mixture covers different pieces")
                continue
            for slot, row in mixed.items():
                expected = lam * np.asarray(left[slot]) + (1.0 - lam) * np.asarray(
                    right[slot]
                )
                expected = expected / expected.sum()
                if not np.allclose(row, expected, rtol=0.0, atol=1e-12):
                    findings.append(
                        f"lambda={lam}, slot {slot}: the mixture is not the formula"
                    )
                if abs(float(row.sum()) - 1.0) > 1e-12:
                    findings.append(f"lambda={lam}, slot {slot}: not normalized")
            if lam == 1.0:
                worst_endpoint_error = max(
                    worst_endpoint_error,
                    max(
                        float(np.abs(mixed[slot] - np.asarray(left[slot])).max())
                        for slot in mixed
                    )
                    if mixed
                    else 0.0,
                )
            if lam == 0.0:
                worst_endpoint_error = max(
                    worst_endpoint_error,
                    max(
                        float(np.abs(mixed[slot] - np.asarray(right[slot])).max())
                        for slot in mixed
                    )
                    if mixed
                    else 0.0,
                )
            checked += 1
    return {
        "passed": not findings,
        "findings": findings[:8],
        "positions": len(states),
        "marginal_sets_checked": checked,
        "worst_endpoint_deviation": worst_endpoint_error,
    }


def check_determinism(models, states, *, lambdas=MIXTURE_LAMBDAS, seed: int = 606) -> dict:
    """A fixed seed must reproduce the same action, worlds and Q values."""
    findings = []
    per_lambda = {}
    from ...engine.legal_moves import legal_actions

    for lam in lambdas:
        bundle = build_mixture_bundle(models, lam)
        legal_all = 0
        for _row, state, _plan in states:
            legal = set(legal_actions(state))
            first = bundle.engine.choose_action(state, seed=seed)
            again = bundle.engine.choose_action(state, seed=seed)
            token = lambda_token(lam)
            if first.selected_action_id != again.selected_action_id:
                findings.append(f"{token}: the same seed chose two different actions")
            if first.world_weights != again.world_weights:
                findings.append(f"{token}: the same seed sampled different worlds")
            if [round(c.q_value, 12) for c in first.candidates] != [
                round(c.q_value, 12) for c in again.candidates
            ]:
                findings.append(f"{token}: the same seed produced different Q values")
            if first.selected_action_id not in legal:
                findings.append(f"{token}: selected an illegal action")
            else:
                legal_all += 1
            if sum(first.world_weights) != first.worlds_requested:
                findings.append(f"{token}: world weights do not sum to the budget")
            if not any(candidate.is_direct for candidate in first.candidates):
                findings.append(f"{token}: the direct action was not a candidate")
        per_lambda[lambda_token(lam)] = {
            "positions": len(states),
            "legal_decisions": legal_all,
        }
    return {"passed": not findings, "findings": findings[:8], "lambdas": per_lambda}


def check_worlds_legal(models, states, *, lambdas=MIXTURE_LAMBDAS) -> dict:
    """Every world a mixture samples passes the accepted validation stack."""
    from ...evaluation.phase11_baselines import validate_world
    from .gate import _public_state

    findings = []
    checked = {}
    for lam in lambdas:
        bundle = build_mixture_bundle(models, lam)
        worlds_seen = 0
        for _row, state, _plan in states:
            public = _public_state(state)
            marginals = bundle.provider.predict_marginals(public)
            start = derive_phase15_seed(
                DOMAIN_INTERFACE, "world", MIXTURE_DECISION_SEED
            ) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
            for offset in range(bundle.config.worlds):
                ordinal = (start + offset) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
                world = sample_belief_world(
                    Phase11SamplerRequest(
                        sampler_version=BELIEF_SAMPLER_VERSION,
                        public_state_document=public.public_state_document,
                        learned_probabilities=marginals,
                        sample_ordinal=int(ordinal),
                    )
                )
                report = validate_world(public.public_state_document, world)
                worlds_seen += 1
                if not report["valid"]:
                    findings.append(
                        f"{lambda_token(lam)}: a world failed the accepted "
                        f"validation stack: {report['findings'][:2]}"
                    )
                    break
        checked[lambda_token(lam)] = worlds_seen
    return {"passed": not findings, "findings": findings[:8], "worlds_checked": checked}


__all__ = [
    "MIXTURE_DECISION_SEED",
    "MIXTURE_LAMBDAS",
    "MIXTURE_LEARNED_PROVIDER",
    "MIXTURE_MOVE_MODEL",
    "MIXTURE_RECOVERY_FRACTION",
    "MIXTURE_REFERENCE_PRESET",
    "MIXTURE_STAGE1_PRESET",
    "MIXTURE_VERSION",
    "MixtureBeliefProvider",
    "MixturePairing",
    "Phase15MixtureError",
    "build_mixture_bundle",
    "check_configuration_invariants",
    "check_determinism",
    "check_frozen_identity",
    "check_mixture_algebra",
    "check_worlds_legal",
    "lambda_token",
    "mixture_arm_id",
    "mixture_pairing",
    "mixture_provider_id",
]
