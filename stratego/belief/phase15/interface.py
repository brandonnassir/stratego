"""Phase 15 Agent 1 section 12: the belief/sampler interface.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 12.

```text
predict_marginals(public_state)      -> 12-way rank probabilities
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

Adapter, not a fork
-------------------
`sample_worlds` builds an accepted :class:`Phase11SamplerRequest` and calls
the accepted `sample_belief_world`. Every constraint — the completion
feasibility guard, the `learned_probability * remaining_count` weighting,
the frozen categorical walk, the full validation stack — is the accepted
Phase 11 code, imported and unmodified. Phase 15 supplies marginals and
nothing else: it does not sample pieces independently and it does not
touch the accepted inventory or movement-impossibility constraints.

The public-state type is the accepted one
------------------------------------------
:class:`Phase11BPublicState` is reused by import rather than restated. It
holds exactly the two public objects the accepted `Phase11BeliefRequest`
holds — the frozen public-state document and the 127-channel observation —
and its constructor refuses a document that is not a
`phase11_public_state_v1`. There is no field a true rank could arrive in,
and reusing it is what makes a Phase 15 provider drop straight into the
accepted Phase 12 search engine. It lives in `belief/phase11b/interface.py`,
which imports nothing from the mis-oriented `belief/phase11b/corpus.py`.

The seed, and why it becomes an ordinal
---------------------------------------
The accepted sampler derives its randomness from `(sampler_version,
public_state_identity, sample_ordinal)` and takes no caller seed. Rather
than reach into that derivation, the adapter maps the requested `seed` onto
a starting sample ordinal through the Phase 15 seed stream and walks `n`
consecutive ordinals from there. The result is seed-deterministic for the
caller and bit-identical to what the accepted sampler would have produced
at those ordinals.
"""

from __future__ import annotations

import numpy as np

from ...belief.phase11b.interface import Phase11BPublicState
from ...engine.constants import PLAYER_NAMES, PLAYERS
from ...engine.coordinates import to_perspective
from ...evaluation.phase11_public_state import hidden_opponent_pieces
from ...evaluation.phase11_sampler import Phase11SamplerRequest, sample_belief_world
from ...training.phase11_contract import BELIEF_SAMPLER_VERSION
from ...training.phase11_seed import MAX_SAMPLE_ORDINAL_FORMAT
from .contract import RANK_COUNT, Phase15Error
from .features import encode_prefix
from .seeds import DOMAIN_INTERFACE, derive_phase15_seed

#: The interface identity every Phase 15 specialist exposes.
BELIEF_INTERFACE_VERSION = "phase15_belief_interface_v1"

#: The accepted public-state type, re-exported so a caller never has to
#: know it came from the Phase 11B namespace.
Phase15PublicState = Phase11BPublicState

_PLAYER_BY_COLOR = {PLAYER_NAMES[player]: player for player in PLAYERS}


class Phase15InterfaceError(Phase15Error):
    """A belief request was refused, or an interface invariant was violated."""


class Phase15BeliefProvider:
    """One specialist plus its frozen backbone, as a belief provider.

    Holds the frozen policy model (for the three-block prefix) and the
    trained specialist (for the last block, the norm, the MLP and the
    temperature), and exposes the two section 12 methods. It reads a
    :class:`Phase15PublicState` and nothing else, so it cannot see a hidden
    rank even if a caller wanted it to.
    """

    interface_version = BELIEF_INTERFACE_VERSION
    uses_hidden_truth = False

    def __init__(
        self,
        policy_model,
        specialist,
        *,
        provider_id: str,
        identity: dict | None = None,
        device: str = "cpu",
        calibrated: bool = True,
    ) -> None:
        import torch

        self._torch = torch
        self.policy_model = policy_model.eval()
        self.specialist = specialist.eval()
        self.provider_id = str(provider_id)
        self.identity = dict(identity or {})
        self.device = torch.device(device)
        self.calibrated = bool(calibrated)
        for parameter in self.policy_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.specialist.parameters():
            parameter.requires_grad_(False)

    # -- section 12 --------------------------------------------------------

    def predict_marginals(self, public_state) -> dict:
        """12-way rank probabilities for every unresolved opponent piece.

        Keyed by setup slot, exactly the key the accepted sampler wants.
        The vector is the float64 softmax of the specialist's logits at the
        piece's perspective-normalized square — the accepted convention, no
        masking and no epsilon — divided by the fitted temperature when the
        provider is calibrated.
        """
        if not isinstance(public_state, Phase15PublicState):
            raise Phase15InterfaceError(
                "predict_marginals accepts only a Phase15PublicState, got "
                f"{type(public_state).__name__}"
            )
        torch = self._torch
        document = public_state.public_state_document
        observer = _PLAYER_BY_COLOR.get(document.get("observer_color"))
        if observer is None:
            raise Phase15InterfaceError("the public document names no valid observer")
        hidden = hidden_opponent_pieces(document)
        if not hidden:
            return {}
        squares = [
            to_perspective(int(piece["current_square"]), observer) for piece in hidden
        ]
        batch = np.array(public_state.observation, dtype=np.float32, copy=True)[None]
        with torch.no_grad():
            tokens = encode_prefix(self.policy_model, batch)
            logits = self.specialist.belief_logits(tokens[0])[squares]
            if self.calibrated:
                logits = self.specialist.calibrated_logits(logits)
            # Move first, then widen: casting to float64 on a device that
            # has no float64 silently degrades the result.
            probabilities = torch.softmax(
                logits.detach().cpu().to(torch.float64), dim=1
            ).numpy()
        marginals = {
            int(piece["piece_slot"]): np.asarray(row, dtype=np.float64)
            for piece, row in zip(hidden, probabilities)
        }
        for slot, row in marginals.items():
            if row.shape != (RANK_COUNT,) or not np.isfinite(row).all():
                raise Phase15InterfaceError(f"slot {slot} produced a malformed marginal")
        return marginals

    def sample_worlds(self, public_state, n: int, seed: int) -> "list[dict]":
        """`n` complete legal hidden armies, through the accepted sampler."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise Phase15InterfaceError(f"n must be a positive int, got {n!r}")
        marginals = self.predict_marginals(public_state)
        start = derive_phase15_seed(DOMAIN_INTERFACE, "world", int(seed)) % (
            MAX_SAMPLE_ORDINAL_FORMAT + 1
        )
        worlds = []
        for offset in range(int(n)):
            ordinal = (start + offset) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
            worlds.append(
                sample_belief_world(
                    Phase11SamplerRequest(
                        sampler_version=BELIEF_SAMPLER_VERSION,
                        public_state_document=public_state.public_state_document,
                        learned_probabilities=marginals,
                        sample_ordinal=int(ordinal),
                    )
                )
            )
        return worlds

    # -- Phase 12 search-engine compatibility ------------------------------

    def sample_assignments(self, public_state, n: int, seed: int) -> "list[dict]":
        """`{piece_slot: rank}` per world — the accepted engine's vocabulary.

        The accepted Phase 12 search engine calls providers through this
        name; `sample_worlds` is section 12's name for the same call. Both
        exist so the handoff can bind either without an adapter in between.
        """
        return [
            {int(slot): int(rank) for slot, rank in world["assignment"].items()}
            for world in self.sample_worlds(public_state, n, seed)
        ]

    def describe(self) -> dict:
        return {
            "interface_version": BELIEF_INTERFACE_VERSION,
            "provider_id": self.provider_id,
            "sampler": BELIEF_SAMPLER_VERSION,
            "sampler_source": (
                "stratego.evaluation.phase11_sampler (accepted, unmodified, by import)"
            ),
            "device": str(self.device),
            "calibrated": self.calibrated,
            "temperature": self.specialist.temperature,
            "uses_hidden_truth": False,
            "reads_hidden_truth": False,
            "independent_per_piece_sampling": False,
            "identity": dict(self.identity),
        }


__all__ = [
    "BELIEF_INTERFACE_VERSION",
    "Phase15BeliefProvider",
    "Phase15InterfaceError",
    "Phase15PublicState",
]
