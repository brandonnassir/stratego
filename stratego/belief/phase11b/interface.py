"""Phase 11B shared belief interface, and its adapter to the accepted sampler.

Specification sources:

- `00_PHASE_11B_OVERVIEW.md` ("Shared Belief Interface")
- `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Required Interface")

```text
predict_marginals(public_state) -> {piece_slot: 12-way rank probabilities}
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

Adapter, not a fork
-------------------
`sample_worlds` builds an accepted :class:`Phase11SamplerRequest` and calls
the accepted `sample_belief_world`. Every constraint — the completion
feasibility guard, the `learned_probability * remaining_count` weighting,
the frozen categorical walk, the full validation stack — is the accepted
Phase 11 code, imported and unmodified. Phase 11B supplies marginals and
nothing else.

The seed, and why it becomes an ordinal
---------------------------------------
The accepted sampler derives its randomness from
`(sampler_version, public_state_identity, sample_ordinal)` and takes no
caller seed. Rather than reach into that derivation, the adapter maps the
requested `seed` onto a starting sample ordinal through the Phase 11B seed
stream and walks `n` consecutive ordinals from there. The result is
seed-deterministic for the caller and bit-identical to what the accepted
sampler would have produced at those ordinals.

The state this interface accepts cannot carry truth
----------------------------------------------------
:class:`Phase11BPublicState` holds exactly the two public objects the
accepted `Phase11BeliefRequest` holds — the frozen public-state document
and the 127-channel observation — and its constructor refuses a document
that is not a `phase11_public_state_v1`. There is no field a true rank
could arrive in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...evaluation.phase11_public_state import (
    PUBLIC_STATE_DOCUMENT_VERSION,
    hidden_opponent_pieces,
    public_state_identity,
)
from ...evaluation.phase11_sampler import Phase11SamplerRequest, sample_belief_world
from ...training.phase11_contract import BELIEF_SAMPLER_VERSION
from ...training.phase11_seed import MAX_SAMPLE_ORDINAL_FORMAT
from .contract import OBSERVATION_SHAPE, RANK_COUNT, Phase11BError
from .seeds import DOMAIN_INTERFACE, derive_phase11b_seed

#: The interface identity every Phase 11B candidate exposes.
BELIEF_INTERFACE_VERSION = "phase11b_belief_interface_v1"


class Phase11BInterfaceError(Phase11BError):
    """A belief request was refused, or an interface invariant was violated."""


@dataclass(frozen=True)
class Phase11BPublicState:
    """The two public objects a Phase 11B belief model may read."""

    public_state_document: dict
    observation: np.ndarray

    def __post_init__(self) -> None:
        document = self.public_state_document
        if not isinstance(document, dict):
            raise Phase11BInterfaceError("public_state_document must be a mapping")
        if document.get("document_version") != PUBLIC_STATE_DOCUMENT_VERSION:
            raise Phase11BInterfaceError(
                f"public_state_document is not a {PUBLIC_STATE_DOCUMENT_VERSION!r} document"
            )
        observation = np.asarray(self.observation)
        if observation.shape != OBSERVATION_SHAPE:
            raise Phase11BInterfaceError(
                f"observation is {observation.shape}, expected {OBSERVATION_SHAPE}"
            )

    @property
    def identity(self) -> str:
        return public_state_identity(self.public_state_document)


class Phase11BBeliefModel:
    """A frozen C1 encoder plus one trained Phase 11B head, as a belief model.

    Holds the encoder and the head, and exposes the two interface methods.
    It reads a :class:`Phase11BPublicState` and nothing else, so it cannot
    see a hidden rank even if a caller wanted it to.
    """

    interface_version = BELIEF_INTERFACE_VERSION

    def __init__(self, encoder, head, *, candidate_id: str, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self.encoder = encoder.eval()
        self.head = head.eval()
        self.candidate_id = str(candidate_id)
        self.device = torch.device(device)

    def _features(self, state: Phase11BPublicState):
        """`[100, 128]` frozen C1 features of one public state.

        Which layer depends on the head: a head that ends at the accepted
        encoder reads the encoder's output, and a head that carries its own
        copy of the last encoder block reads that block's *input*, so the
        frozen prefix is computed once either way.
        """
        from .features import CACHE_LAYERS, LAYER_FINAL, encode_batch

        layer = getattr(self.head, "feature_layer", LAYER_FINAL)
        if layer not in CACHE_LAYERS:  # pragma: no cover - defensive
            raise Phase11BInterfaceError(f"unknown feature layer {layer!r}")
        batch = np.array(state.observation, dtype=np.float32, copy=True)[None]
        return encode_batch(self.encoder, batch, layer)[0]

    def predict_marginals(self, public_state: Phase11BPublicState) -> dict:
        """12-way rank probabilities for every unresolved opponent piece.

        Keyed by setup slot, exactly the key the accepted sampler wants.
        The vector is the raw float64 softmax of the head's logits at the
        piece's perspective-normalized square — the accepted convention,
        no masking and no epsilon.
        """
        import torch

        from ...engine.coordinates import to_perspective
        from ...engine.constants import PLAYER_NAMES, PLAYERS

        if not isinstance(public_state, Phase11BPublicState):
            raise Phase11BInterfaceError(
                "predict_marginals accepts only a Phase11BPublicState, got "
                f"{type(public_state).__name__}"
            )
        document = public_state.public_state_document
        observer = next(
            player
            for player in PLAYERS
            if PLAYER_NAMES[player] == document["observer_color"]
        )
        features = self._features(public_state)
        hidden = hidden_opponent_pieces(document)
        if not hidden:
            return {}
        squares = [
            to_perspective(int(piece["current_square"]), observer) for piece in hidden
        ]
        with torch.no_grad():
            logits = self.head.belief_logits(features)[squares]
            # Move first, then widen: casting to float64 on a device that has
            # no float64 silently degrades the result.
            probabilities = torch.softmax(
                logits.detach().cpu().to(torch.float64), dim=1
            ).numpy()
        marginals = {
            int(piece["piece_slot"]): np.asarray(row, dtype=np.float64)
            for piece, row in zip(hidden, probabilities)
        }
        for slot, row in marginals.items():
            if row.shape != (RANK_COUNT,) or not np.isfinite(row).all():
                raise Phase11BInterfaceError(f"slot {slot} produced a malformed marginal")
        return marginals

    def sample_worlds(
        self, public_state: Phase11BPublicState, n: int, seed: int
    ) -> "list[dict]":
        """`n` complete legal hidden armies, through the accepted sampler."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise Phase11BInterfaceError(f"n must be a positive int, got {n!r}")
        marginals = self.predict_marginals(public_state)
        start = derive_phase11b_seed(DOMAIN_INTERFACE, "world", int(seed)) % (
            MAX_SAMPLE_ORDINAL_FORMAT + 1
        )
        worlds = []
        for offset in range(int(n)):
            ordinal = (start + offset) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
            request = Phase11SamplerRequest(
                sampler_version=BELIEF_SAMPLER_VERSION,
                public_state_document=public_state.public_state_document,
                learned_probabilities=marginals,
                sample_ordinal=int(ordinal),
            )
            worlds.append(sample_belief_world(request))
        return worlds

    def describe(self) -> dict:
        return {
            "interface_version": BELIEF_INTERFACE_VERSION,
            "candidate_id": self.candidate_id,
            "sampler": BELIEF_SAMPLER_VERSION,
            "sampler_source": "stratego.evaluation.phase11_sampler (accepted, unmodified)",
            "device": str(self.device),
            "reads_hidden_truth": False,
        }


__all__ = [
    "BELIEF_INTERFACE_VERSION",
    "Phase11BBeliefModel",
    "Phase11BInterfaceError",
    "Phase11BPublicState",
]
