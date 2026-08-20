"""Phase 12 belief providers: four sources of hidden worlds, one interface.

Specification sources:

- `00_PHASE_12_SEQUENCE_AND_COMMON_CONTRACT.md` section 5
- `02_PHASE_12_AGENT_1_SEARCH_CORE.md` section 3

Adapters, not forks
-------------------
The two neural providers are the accepted Phase 11B
:class:`~stratego.belief.phase11b.interface.Phase11BBeliefModel` adapter
wrapped unchanged, so their marginals and their worlds go through the
accepted Phase 11 sampler mathematics by import:

- `original_phase11` is :class:`ExistingBeliefHead` initialized **from the
  accepted Phase 9 weights** (the frozen `belief_output` linear layer) and
  digest-checked against `ACCEPTED_BELIEF_HEAD_DIGEST` — the accepted
  Phase 11 belief head, not Phase 11B's retrained 1A candidate;
- `agent1c` is the selected Phase 11B candidate loaded from its surviving
  checkpoint bytes.

`remaining_count` uses the accepted count-uniform skeleton
(:func:`stratego.evaluation.phase11_baselines.sample_world`) and validates
every world through the accepted stack, exactly like the learned sampler
does for its own worlds.

The oracle is different on purpose
----------------------------------
Every non-oracle provider reads a
:class:`~stratego.belief.phase11b.interface.Phase11BPublicState` and nothing
else — a type with no field a true rank could arrive in. The oracle is the
one explicitly-offline diagnostic exception: it reads the privileged
:class:`~stratego.engine.state.GameState` through a separate method name,
must be constructed with `offline_diagnostic=True`, is refused by
:func:`build_belief_provider` under a production configuration, and is
refused again by the engine when `config.production` is true.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np

from ...belief.phase11b.heads import CANDIDATE_1C, ExistingBeliefHead, build_candidate
from ...belief.phase11b.features import belief_head_digest
from ...belief.phase11b.interface import Phase11BBeliefModel, Phase11BPublicState
from ...engine.constants import PLAYER_NAMES, PLAYERS
from ...engine.permutation import hidden_opponent_piece_ids
from ...engine.pieces import piece_setup_slot
from ...engine.state import GameState
from ...evaluation.phase11_baselines import (
    COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
    REMAINING_COUNT_BASELINE_VERSION,
    remaining_count_belief,
    remaining_counts,
    sample_world,
    validate_world,
)
from ...evaluation.phase11_public_state import hidden_opponent_pieces
from ...training.phase11_contract import (
    ACCEPTED_BELIEF_HEAD_DIGEST,
    RANK_COUNT,
)
from ...training.phase11_seed import MAX_SAMPLE_ORDINAL_FORMAT
from .contract import (
    DOMAIN_COUNT_WORLDS,
    PRODUCTION_PROVIDERS,
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_REMAINING_COUNT,
    Phase12SearchError,
    derive_phase12_seed,
)

#: Colour token -> player id, for reading a public document's observer.
_PLAYER_BY_COLOR = {PLAYER_NAMES[player]: player for player in PLAYERS}


class Phase12BeliefProvider(ABC):
    """One interchangeable source of complete hidden worlds for search.

    An *assignment* is `{opponent piece_slot: rank index}` covering exactly
    the live opponent pieces whose rank the observer may not know — the
    accepted sampler's own output vocabulary.
    """

    provider_id: ClassVar[str]
    #: True only for the oracle. The engine hands the privileged state to a
    #: provider if and only if this is set, and only outside production.
    uses_hidden_truth: ClassVar[bool] = False

    @abstractmethod
    def predict_marginals(self, public: Phase11BPublicState) -> dict:
        """`{piece_slot: 12-way rank probabilities}` for the hidden pieces."""

    @abstractmethod
    def sample_assignments(
        self, public: Phase11BPublicState, n: int, seed: int
    ) -> "list[dict[int, int]]":
        """`n` complete legal assignments, deterministic in `seed`."""

    def describe(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "uses_hidden_truth": self.uses_hidden_truth,
        }


def _check_positive_count(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise Phase12SearchError(f"n must be a positive int, got {n!r}")
    return int(n)


class RemainingCountBeliefProvider(Phase12BeliefProvider):
    """The count-based baseline: accepted skeleton, count-only weighting."""

    provider_id = PROVIDER_REMAINING_COUNT

    def predict_marginals(self, public: Phase11BPublicState) -> dict:
        return remaining_count_belief(public.public_state_document)

    def sample_assignments(
        self, public: Phase11BPublicState, n: int, seed: int
    ) -> "list[dict[int, int]]":
        n = _check_positive_count(n)
        document = public.public_state_document
        start = derive_phase12_seed(DOMAIN_COUNT_WORLDS, "world", int(seed)) % (
            MAX_SAMPLE_ORDINAL_FORMAT + 1
        )
        assignments = []
        for offset in range(n):
            ordinal = (start + offset) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
            world = sample_world(document, ordinal)
            # The count-uniform path does not validate internally (the
            # learned sampler does); hold it to the same standard here.
            check = validate_world(document, world)
            if not check["valid"]:
                raise Phase12SearchError(
                    "a count-uniform world failed the accepted validation "
                    f"stack: {check['findings'][:3]}"
                )
            assignments.append(
                {int(slot): int(rank) for slot, rank in world["assignment"].items()}
            )
        return assignments

    def describe(self) -> dict:
        report = super().describe()
        report.update(
            {
                "marginals_version": REMAINING_COUNT_BASELINE_VERSION,
                "world_sampler_version": COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
                "sampler_source": "stratego.evaluation.phase11_baselines (accepted, unmodified)",
            }
        )
        return report


class AdapterNeuralBeliefProvider(Phase12BeliefProvider):
    """A neural belief model behind the accepted Phase 11B adapter.

    Marginals and worlds are the adapter's own, unchanged: the frozen C1
    prefix, the head's logits at each hidden piece's normalized square, and
    the accepted learned sampler with its full validation stack.
    """

    def __init__(
        self, belief_model: Phase11BBeliefModel, *, provider_id: str, identity: dict
    ) -> None:
        if provider_id not in (PROVIDER_ORIGINAL_PHASE11, PROVIDER_AGENT1C):
            raise Phase12SearchError(
                f"unknown neural provider id {provider_id!r}"
            )
        if not isinstance(belief_model, Phase11BBeliefModel):
            raise Phase12SearchError(
                "AdapterNeuralBeliefProvider wraps a Phase11BBeliefModel, got "
                f"{type(belief_model).__name__}"
            )
        self.provider_id = provider_id
        self.belief_model = belief_model
        self.identity = dict(identity)

    def predict_marginals(self, public: Phase11BPublicState) -> dict:
        return self.belief_model.predict_marginals(public)

    def sample_assignments(
        self, public: Phase11BPublicState, n: int, seed: int
    ) -> "list[dict[int, int]]":
        n = _check_positive_count(n)
        worlds = self.belief_model.sample_worlds(public, n, int(seed))
        return [
            {int(slot): int(rank) for slot, rank in world["assignment"].items()}
            for world in worlds
        ]

    def describe(self) -> dict:
        report = super().describe()
        report["adapter"] = self.belief_model.describe()
        report["identity"] = dict(self.identity)
        return report


class OracleBeliefProvider(Phase12BeliefProvider):
    """True hidden information. Offline diagnostic upper bound, and only that.

    Refuses to exist unless constructed with `offline_diagnostic=True`;
    refuses `sample_assignments` (it has no public path); and cross-checks
    the privileged state against the public document before answering, so
    it can never silently pair one position's truth with another's search.
    """

    provider_id = PROVIDER_ORACLE
    uses_hidden_truth = True

    def __init__(self, *, offline_diagnostic: bool = False) -> None:
        if offline_diagnostic is not True:
            raise Phase12SearchError(
                "the oracle provider is an offline diagnostic; construct it "
                "with offline_diagnostic=True or do not construct it at all"
            )

    def predict_marginals(self, public: Phase11BPublicState) -> dict:
        raise Phase12SearchError(
            "the oracle has no public marginals; it is not a belief model"
        )

    def sample_assignments(
        self, public: Phase11BPublicState, n: int, seed: int
    ) -> "list[dict[int, int]]":
        raise Phase12SearchError(
            "the oracle cannot sample from public state; the engine must call "
            "sample_assignments_privileged outside production"
        )

    def sample_assignments_privileged(
        self, state: GameState, public: Phase11BPublicState, n: int, seed: int
    ) -> "list[dict[int, int]]":
        n = _check_positive_count(n)
        document = public.public_state_document
        observer = _PLAYER_BY_COLOR.get(document.get("observer_color"))
        if observer is None:
            raise Phase12SearchError("the public document names no valid observer")

        document_hidden = {
            int(piece["piece_slot"]): piece for piece in hidden_opponent_pieces(document)
        }
        truth: dict[int, int] = {}
        state_hidden = hidden_opponent_piece_ids(state, observer)
        for piece_id in state_hidden:
            record = state.pieces[piece_id]
            slot = piece_setup_slot(piece_id)
            entry = document_hidden.get(slot)
            if (
                entry is None
                or int(entry["current_square"]) != int(record.current_square)
                or bool(entry["has_moved"]) != bool(record.has_moved)
            ):
                raise Phase12SearchError(
                    "the privileged state and the public document disagree "
                    f"about hidden slot {slot}; refusing to answer"
                )
            truth[slot] = int(record.true_type)
        if set(truth) != set(document_hidden):
            raise Phase12SearchError(
                "the privileged state and the public document disagree about "
                "which opponent pieces are unresolved"
            )

        observed = [0] * RANK_COUNT
        for rank in truth.values():
            observed[rank] += 1
        if tuple(observed) != tuple(remaining_counts(document)):
            raise Phase12SearchError(
                "the true hidden army does not match the public remaining "
                "inventory; the state and document describe different games"
            )
        # There is exactly one true world; the requested n copies keep the
        # provider interchangeable, and the engine's deduplication collapses
        # them back into one weighted world.
        return [dict(truth) for _ in range(n)]

    def describe(self) -> dict:
        report = super().describe()
        report.update(
            {
                "role": "offline diagnostic upper bound",
                "available_in_production": False,
            }
        )
        return report


# ---------------------------------------------------------------------------
# Loading the neural heads
# ---------------------------------------------------------------------------


def _torch_device(device):
    import torch

    return torch.device(device)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_original_phase11_head(encoder) -> tuple:
    """`(head, identity)`: the accepted Phase 9 belief head, digest-checked.

    Built through the Phase 11B `ExistingBeliefHead.from_accepted` copy so
    the accepted checkpoint is never mutated, then verified byte-for-byte
    against the accepted `belief_head_digest` recipe.
    """
    head = ExistingBeliefHead.from_accepted(encoder).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    digest = belief_head_digest(head)
    if digest != ACCEPTED_BELIEF_HEAD_DIGEST:
        raise Phase12SearchError(
            f"the copied original belief head digests to {digest}, not the "
            f"accepted {ACCEPTED_BELIEF_HEAD_DIGEST}"
        )
    identity = {
        "head": "linear(128->12) belief_output from the accepted Phase 9 checkpoint",
        "belief_head_digest": digest,
        "feature_layer": head.feature_layer,
    }
    return head, identity


def load_agent1c_head(
    encoder,
    checkpoint_path,
    *,
    expected_sha256: "str | None" = None,
    expected_state_digest: "str | None" = None,
) -> tuple:
    """`(head, identity)`: the selected Phase 11B Agent 1C belief model.

    The checkpoint file is read-only input; pass the handoff's `sha256` and
    `state_dict_digest` to bind the load to the surviving bytes.
    """
    import torch

    from ...training.phase9_behavior import state_dict_digest

    path = Path(checkpoint_path)
    if not path.exists():
        raise Phase12SearchError(f"agent1c checkpoint {path} does not exist")
    file_digest = _file_sha256(path)
    if expected_sha256 is not None and file_digest != expected_sha256:
        raise Phase12SearchError(
            f"agent1c checkpoint {path} has sha256 {file_digest}, expected "
            f"{expected_sha256}; refusing to load unbound bytes"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("candidate_id") != CANDIDATE_1C:
        raise Phase12SearchError(
            f"checkpoint {path} describes candidate "
            f"{payload.get('candidate_id')!r}, expected {CANDIDATE_1C!r}"
        )
    head = build_candidate(CANDIDATE_1C, encoder)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    state_digest = state_dict_digest(head)
    if expected_state_digest is not None and state_digest != expected_state_digest:
        raise Phase12SearchError(
            f"agent1c weights digest to {state_digest}, expected "
            f"{expected_state_digest}"
        )
    identity = {
        "candidate_id": CANDIDATE_1C,
        "architecture": payload.get("architecture"),
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_digest,
        "state_dict_digest": state_digest,
        "dev_metrics": dict(payload.get("dev_metrics") or {}),
        "feature_layer": head.feature_layer,
    }
    return head, identity


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


def build_belief_provider(
    name: str,
    *,
    encoder=None,
    agent1c_checkpoint=None,
    expected_agent1c_sha256: "str | None" = None,
    expected_agent1c_state_digest: "str | None" = None,
    production: bool = True,
    device: str = "cpu",
) -> Phase12BeliefProvider:
    """Build one provider by its contract name.

    `encoder` is the frozen accepted Phase 9 C1 model and is required for
    the two neural providers. With `production=True` (the default) the
    oracle name is refused outright.
    """
    if name == PROVIDER_ORACLE:
        if production:
            raise Phase12SearchError(
                "oracle is not an available belief provider in a production "
                f"configuration; production providers are {PRODUCTION_PROVIDERS}"
            )
        return OracleBeliefProvider(offline_diagnostic=True)
    if name == PROVIDER_REMAINING_COUNT:
        return RemainingCountBeliefProvider()
    if name == PROVIDER_ORIGINAL_PHASE11:
        if encoder is None:
            raise Phase12SearchError("original_phase11 needs the frozen C1 encoder")
        head, identity = load_original_phase11_head(encoder)
        # Digests are taken on the CPU (their recipe casts there anyway);
        # the head then joins the encoder's device for inference.
        head = head.to(_torch_device(device))
        model = Phase11BBeliefModel(
            encoder, head, candidate_id="phase11_accepted_belief_head", device=device
        )
        return AdapterNeuralBeliefProvider(
            model, provider_id=PROVIDER_ORIGINAL_PHASE11, identity=identity
        )
    if name == PROVIDER_AGENT1C:
        if encoder is None:
            raise Phase12SearchError("agent1c needs the frozen C1 encoder")
        if agent1c_checkpoint is None:
            raise Phase12SearchError("agent1c needs its checkpoint path")
        head, identity = load_agent1c_head(
            encoder,
            agent1c_checkpoint,
            expected_sha256=expected_agent1c_sha256,
            expected_state_digest=expected_agent1c_state_digest,
        )
        head = head.to(_torch_device(device))
        model = Phase11BBeliefModel(
            encoder, head, candidate_id=CANDIDATE_1C, device=device
        )
        return AdapterNeuralBeliefProvider(
            model, provider_id=PROVIDER_AGENT1C, identity=identity
        )
    raise Phase12SearchError(
        f"unknown belief provider {name!r}; contract names are "
        f"{PRODUCTION_PROVIDERS + (PROVIDER_ORACLE,)}"
    )


__all__ = [
    "AdapterNeuralBeliefProvider",
    "OracleBeliefProvider",
    "Phase12BeliefProvider",
    "RemainingCountBeliefProvider",
    "build_belief_provider",
    "load_agent1c_head",
    "load_original_phase11_head",
]
