"""Phase 11B Part 0: the common corpus Agents 2-5 reuse unchanged.

Specification sources:

- `00_PHASE_11B_OVERVIEW.md` ("Common Phase 11B Dataset", "Canonical Sample
  Contents")
- `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Part 0", "Data Boundary")

Fresh games, accepted machinery
-------------------------------
Phase 11B owns no game loop and no opponent. Every corpus game is played by
the accepted `match_runner.play_match` with the accepted `EVALUATION_RULES`,
the accepted Phase 9 greedy seat as observer, and the accepted Phase 11
stratum opponent wrapped in the accepted `FrozenSeedPolicy`. The only new
thing is the identity: Phase 11B seeds, so the games are fresh and the spent
`phase11_test_bank_v1` is never opened.

Two passes, and truth only in the second
----------------------------------------
The Phase 11 pattern, reused. The **public pass** plays the game: the
observer policy sees a `PolicyInput`, records the ply, the unresolved-piece
count and a digest of its own observation, and is structurally unable to
reach a hidden rank. The **privileged pass** then replays the recorded
action history through the engine, rebuilds each selected decision from
scratch, checks the rebuilt observation against the digest the public pass
recorded — a bit-for-bit alignment proof, not a spot check — and only then
reads `dense_belief_target`.

Nothing flows backwards: the privileged pass's outputs are the label arrays,
written under `privileged/`, and no model input is derived from them.

An eligible observer decision
-----------------------------
One where the observer is to act and at least one opponent piece remains
unresolved. Ineligible decisions carry no belief target at all, so training
on them would be training on an empty loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ...engine.constants import BLUE, RED
from ...engine.observation import build_observation
from ...engine.state import create_game
from ...engine.transition import apply_action
from ...evaluation.match_runner import ON_POLICY_ERROR_RAISE, play_match
from ...evaluation.match_spec import EVALUATION_RULES, MatchSpec
from ...evaluation.neural_worker import (
    DECISION_MODE_GREEDY,
    InferenceRequest,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from ...evaluation.phase10_validation import FrozenSeedPolicy
from ...evaluation.phase11_baselines import remaining_counts
from ...evaluation.phase11_public_state import (
    build_public_state_document,
    hidden_opponent_pieces,
    legal_rank_mask,
    public_state_identity,
)
from ...evaluation.policy import Policy, PolicyRef, PolicyRequirements, build_public_view
from ...evaluation.registry import build_policy
from ...evaluation.setup_bank import SetupBank, SetupPair
from ...engine.coordinates import to_perspective
from ...training.belief_targets import dense_belief_target
from .contract import (
    CELLS,
    CORPUS_SPLITS,
    OBSERVATION_SHAPE,
    STRATUM_POLICY_IDS,
    Phase11BError,
)
from .seeds import (
    ROLE_OBSERVER,
    ROLE_OPPONENT,
    corpus_game_id,
    match_seed,
    setup_seed,
)

#: The suite identity every Phase 11B corpus game is played under.
CORPUS_RUN_VERSION = "phase11b_corpus_generation_v1"

#: The observer seat's policy identity, distinct from the Phase 9 opponent
#: seat's so `play_match` resolves two objects even in the self-play cell.
OBSERVER_POLICY_ID = "phase11b_corpus_observer_v1"
PHASE9_OPPONENT_POLICY_ID = "phase11b_phase9_opponent_v1"

_PLAYER_OF = {"red": RED, "blue": BLUE}


class Phase11BCorpusError(Phase11BError):
    """A corpus game could not be played, extracted or verified."""


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusGamePlan:
    """Everything one corpus game needs, resolved from its identity alone."""

    game_id: str
    split: str
    ordinal: int
    stratum: str
    setup_source: str
    observer_color: str
    opponent_color: str
    match_seed: int
    red_setup: tuple
    blue_setup: tuple

    @property
    def game_index(self) -> int:
        """Row index of this game inside its split, cell-major."""
        return self.ordinal * len(CELLS) + CELLS.index(
            (self.stratum, self.setup_source, self.observer_color)
        )


class Phase11BSetupSources:
    """The two accepted setup sources, loaded once and reused.

    `p10d` is the accepted Phase 10 learned selector and `neutral` the
    accepted Phase 7 `neutral_v1` baseline draw — both called through their
    accepted entry points with a Phase 11B seed. Phase 11B redefines
    neither.
    """

    def __init__(self) -> None:
        from ...setups.sampler import load_library_index
        from ...training.phase10_selector import LearnedSetupSource, candidate, load_scorer
        from ...training.phase11_contract import ACCEPTED_SELECTOR_CANDIDATE_ID

        self.index = load_library_index()
        self.scorer = load_scorer()
        self.learned = LearnedSetupSource(
            candidate(ACCEPTED_SELECTOR_CANDIDATE_ID), self.scorer, self.index
        )

    def draw(self, source: str, library_split: str, color: str, seed: int) -> tuple:
        from ...training.phase10_selector import SelectorRequest, neutral_baseline_draw

        if source == "p10d":
            drawn = self.learned.draw(
                SelectorRequest(split=library_split, color=color, selector_seed=int(seed))
            ).setup
        elif source == "neutral":
            drawn = neutral_baseline_draw(library_split, int(seed), self.index)
        else:  # pragma: no cover - guarded by the id grammar
            raise Phase11BCorpusError(f"unknown setup source {source!r}")
        return tuple(drawn.canonical)


def corpus_plans(
    split: str, sources: "Phase11BSetupSources | None" = None, *, limit: int | None = None
) -> "list[CorpusGamePlan]":
    """Every game of one split, cell-major, resolved from its id.

    Cell-major means a truncated run (`limit`) is still balanced over
    strata, sources and observer colours, which is what makes the pilot a
    scaled-down corpus rather than a biased corner of one.
    """
    if split not in CORPUS_SPLITS:
        raise Phase11BCorpusError(f"unknown split {split!r}")
    specification = CORPUS_SPLITS[split]
    per_cell = specification["games"] // len(CELLS)
    library_split = specification["library_split"]
    if sources is None:
        sources = Phase11BSetupSources()

    plans: list[CorpusGamePlan] = []
    for ordinal in range(per_cell):
        for stratum, source, observer_color in CELLS:
            game_id = corpus_game_id(split, stratum, source, observer_color, ordinal)
            opponent_color = "blue" if observer_color == "red" else "red"
            observer_setup = sources.draw(
                "p10d", library_split, observer_color, setup_seed(game_id, ROLE_OBSERVER)
            )
            opponent_setup = sources.draw(
                source, library_split, opponent_color, setup_seed(game_id, ROLE_OPPONENT)
            )
            red, blue = (
                (observer_setup, opponent_setup)
                if observer_color == "red"
                else (opponent_setup, observer_setup)
            )
            plans.append(
                CorpusGamePlan(
                    game_id=game_id,
                    split=split,
                    ordinal=ordinal,
                    stratum=stratum,
                    setup_source=source,
                    observer_color=observer_color,
                    opponent_color=opponent_color,
                    match_seed=match_seed(game_id),
                    red_setup=red,
                    blue_setup=blue,
                )
            )
            if limit is not None and len(plans) >= int(limit):
                return plans
    return plans


# ---------------------------------------------------------------------------
# The public pass
# ---------------------------------------------------------------------------


def observer_ref() -> PolicyRef:
    return PolicyRef(policy_id=OBSERVER_POLICY_ID, policy_version=CORPUS_RUN_VERSION)


class CorpusObserverPolicy(Policy):
    """The accepted Phase 9 greedy seat, logging public decision facts.

    Records the ply, the unresolved-opponent-piece count and a digest of
    its own observation. It reads a `PolicyInput` and nothing else, so it
    is structurally incapable of seeing a hidden rank — and it records no
    probability of any kind, because Phase 11B trains its own heads and
    must not bake the old head's output into the corpus.
    """

    requirements = PolicyRequirements(
        observation=True, legal_action_mask=True, public_view=True
    )
    description = (
        "Phase 11B corpus observer: the accepted Phase 9 greedy decision, "
        "logging only public decision facts."
    )

    def __init__(self, ref: PolicyRef, owner) -> None:
        self.policy_id = ref.policy_id
        self.policy_version = ref.policy_version
        self.stochastic = False
        self.owner = owner
        self.decisions: list[dict] = []

    @property
    def ref(self) -> PolicyRef:
        return PolicyRef(policy_id=self.policy_id, policy_version=self.policy_version)

    def decide(self, request):
        view = request.require_public_view()
        observation = request.require_observation()
        payload = InferenceRequest.from_policy_input(request)
        response = self.owner.serve(payload)
        if not hasattr(response, "absolute_action_id"):
            raise Phase11BCorpusError(
                f"the observer owner refused a decision: {getattr(response, 'message', response)!r}"
            )
        unresolved = len(view.unresolved_opponent_piece_ids)
        self.decisions.append(
            {
                "ply": int(request.ply),
                "unresolved": unresolved,
                "observation_sha256": hashlib.sha256(
                    np.ascontiguousarray(observation, dtype=np.float32).tobytes()
                ).hexdigest(),
            }
        )
        return self.result(
            request,
            response.absolute_action_id,
            {"phase11b_unresolved_opponent_pieces": unresolved},
        )

    def describe(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "decision_mode": DECISION_MODE_GREEDY,
            "holds_model_weights": False,
            "records_predictions": False,
        }


def build_spec(plan: CorpusGamePlan, opponent: PolicyRef) -> MatchSpec:
    return MatchSpec(
        candidate=observer_ref(),
        opponent=opponent,
        setup_pair_id=plan.game_index,
        candidate_color=_PLAYER_OF[plan.observer_color],
        replicate=plan.ordinal,
        root_seed=plan.match_seed,
        suite_version=CORPUS_RUN_VERSION,
        setup_bank_version=(
            f"{CORPUS_RUN_VERSION}|sp={plan.split}|st={plan.stratum}|src={plan.setup_source}"
        ),
        rules=EVALUATION_RULES,
    )


def single_game_bank(spec: MatchSpec, plan: CorpusGamePlan) -> SetupBank:
    pair = SetupPair(
        setup_pair_id=spec.setup_pair_id,
        red_setup=plan.red_setup,
        blue_setup=plan.blue_setup,
        generation_seed=spec.root_seed,
        bank_version=spec.setup_bank_version,
        generation_family=CORPUS_RUN_VERSION,
    )
    return SetupBank(
        bank_version=spec.setup_bank_version,
        root_seed=spec.root_seed,
        generation_family=CORPUS_RUN_VERSION,
        pairs=(pair,),
    )


def opponent_seat(plan: CorpusGamePlan, owners: dict):
    """`(ref, policy)` for the opponent stratum of this game."""
    policy_id = STRATUM_POLICY_IDS[plan.stratum]
    if policy_id is None:
        ref = PolicyRef(
            policy_id=PHASE9_OPPONENT_POLICY_ID, policy_version=CORPUS_RUN_VERSION
        )
        return ref, RemoteNeuralPolicy(
            ref,
            LocalInferenceChannel(owners["phase9"]),
            decision_mode=DECISION_MODE_GREEDY,
        )
    policy = build_policy(policy_id)
    return policy.ref, policy


def play_corpus_game(plan: CorpusGamePlan, owners: dict):
    """Play one corpus game. Returns `(result, decisions)` — public only."""
    opponent_ref, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_ref)
    observer = CorpusObserverPolicy(observer_ref(), owners["phase9"])
    policies = {observer_ref().token: observer}
    if opponent_ref.token != observer_ref().token:
        policies[opponent_ref.token] = FrozenSeedPolicy(opponent_policy, plan.match_seed)
    result = play_match(
        spec,
        bank=single_game_bank(spec, plan),
        policies=policies,
        record_actions=True,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    if result.errored:  # pragma: no cover - raises above under RAISE
        raise Phase11BCorpusError(f"{plan.game_id} errored: {result.policy_error}")
    return result, observer.decisions


def evenly_spaced(values: "list", count: int) -> "list":
    """At most `count` evenly spaced elements, **both endpoints included**.

    Deliberately *not* the accepted Phase 11 `_evenly_spaced` rule, which is
    `values[(k * n) // take]`: that rule always starts at the first element
    and never reaches the last, so it systematically omits each game's final
    eligible decision. Phase 11 could afford that — it was measuring a fixed
    head over a whole bank. A training corpus cannot: the late-game
    positions it drops are exactly the ones with the fewest unresolved
    pieces and the most public evidence, which is where a belief model has
    the most to learn.

    A Phase 11B slice therefore does not mean the same thing as a Phase 11
    slice, and the two must not be compared decision-for-decision.
    """
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    step = (len(values) - 1) / (count - 1) if count > 1 else 0.0
    return [values[int(round(index * step))] for index in range(count)]


def select_decisions(decisions: "list[dict]", per_game: int) -> "list[dict]":
    """The evenly spaced eligible observer decisions of one game."""
    eligible = [row for row in decisions if row["unresolved"] > 0]
    return evenly_spaced(eligible, int(per_game))


# ---------------------------------------------------------------------------
# The privileged pass
# ---------------------------------------------------------------------------


@dataclass
class ExtractedGame:
    """One game's contribution to the corpus: its selected samples."""

    game_id: str
    plies: int
    samples: list


def privileged_extract(
    plan: CorpusGamePlan, result, selected: "list[dict]"
) -> ExtractedGame:
    """Replay one game and build the samples of its selected decisions.

    Reads true ranks. Runs only after the game is over and every public
    fact already exists, and writes nothing back into any model input.
    """
    history = result.action_history
    if history is None:  # pragma: no cover - record_actions is always True
        raise Phase11BCorpusError(f"{plan.game_id} recorded no action history")
    observer = _PLAYER_OF[plan.observer_color]
    wanted = {int(row["ply"]): row for row in selected}

    state = create_game(
        plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id=plan.game_id
    )
    samples: list[dict] = []
    for action in history:
        if state.terminal:  # pragma: no cover - the history stops at terminal
            break
        ply = int(state.total_moves)
        if state.acting_player == observer and ply in wanted:
            samples.append(_build_sample(plan, state, observer, ply, wanted.pop(ply)))
        apply_action(state, int(action))

    if len(samples) != len(selected):
        raise Phase11BCorpusError(
            f"{plan.game_id}: replay produced {len(samples)} of {len(selected)} samples"
        )
    return ExtractedGame(
        game_id=plan.game_id, plies=int(result.plies), samples=samples
    )


def _build_sample(
    plan: CorpusGamePlan, state, observer: int, ply: int, row: dict
) -> dict:
    """Public arrays and privileged labels for one decision.

    The rebuilt observation is checked against the digest the public pass
    recorded before anything privileged is read, so a sample can only ever
    carry the labels of the position it belongs to.
    """
    observation = build_observation(state, observer)
    if observation.shape != OBSERVATION_SHAPE:  # pragma: no cover - engine invariant
        raise Phase11BCorpusError(f"observation shape {observation.shape} is not frozen")
    digest = hashlib.sha256(
        np.ascontiguousarray(observation, dtype=np.float32).tobytes()
    ).hexdigest()
    if digest != row["observation_sha256"]:
        raise Phase11BCorpusError(
            f"{plan.game_id} ply {ply}: replayed observation {digest[:16]} != "
            f"recorded {row['observation_sha256'][:16]}"
        )

    view = build_public_view(state, observer)
    document = build_public_state_document(view, observation)
    identity = public_state_identity(document)
    counts = remaining_counts(document)
    hidden = hidden_opponent_pieces(document)
    if len(hidden) != int(row["unresolved"]):
        raise Phase11BCorpusError(
            f"{plan.game_id} ply {ply}: {len(hidden)} hidden pieces on replay, "
            f"{row['unresolved']} recorded"
        )
    if not hidden:  # pragma: no cover - selection filters these out
        raise Phase11BCorpusError(f"{plan.game_id} ply {ply} has no hidden piece")

    labels, target_mask = dense_belief_target(state, observer)

    pieces = []
    for piece in hidden:
        square = int(piece["current_square"])
        normalized = to_perspective(square, observer)
        if not target_mask[normalized]:
            raise Phase11BCorpusError(
                f"{plan.game_id} ply {ply}: hidden piece at {square} is not a "
                "supervised square in the engine's belief target"
            )
        true_rank = int(labels[normalized])
        moved = bool(piece["has_moved"])
        mask = legal_rank_mask(moved)
        if not mask[true_rank]:
            raise Phase11BCorpusError(
                f"{plan.game_id} ply {ply}: true rank {true_rank} is excluded by "
                "its own public legal-rank mask"
            )
        if counts[true_rank] <= 0:
            raise Phase11BCorpusError(
                f"{plan.game_id} ply {ply}: true rank {true_rank} has no remaining "
                "public inventory"
            )
        pieces.append(
            {
                "piece_slot": int(piece["piece_slot"]),
                "piece_square": square,
                "perspective_square": int(normalized),
                "piece_moved": moved,
                "legal_rank_mask": mask,
                "true_rank": true_rank,
            }
        )
    if int(target_mask.sum()) != len(pieces):
        raise Phase11BCorpusError(
            f"{plan.game_id} ply {ply}: engine supervises {int(target_mask.sum())} "
            f"squares, the public document names {len(pieces)}"
        )

    return {
        "game_id": plan.game_id,
        "split": plan.split,
        "stratum": plan.stratum,
        "setup_source": plan.setup_source,
        "observer_color": plan.observer_color,
        "decision_index": int(ply),
        "total_moves": int(document["total_moves"]),
        "public_state_identity": identity,
        "observation": observation,
        "target_mask": np.asarray(target_mask, dtype=bool),
        "remaining_counts": tuple(int(value) for value in counts),
        "pieces": pieces,
    }


__all__ = [
    "CORPUS_RUN_VERSION",
    "CorpusGamePlan",
    "CorpusObserverPolicy",
    "ExtractedGame",
    "Phase11BCorpusError",
    "Phase11BSetupSources",
    "corpus_plans",
    "evenly_spaced",
    "play_corpus_game",
    "privileged_extract",
    "select_decisions",
]
