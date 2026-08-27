"""Phase 15 Agent 1 sections 5-7: the orientation-safe belief corpus.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` sections 5, 6, 7.

A new collector, not the old one
--------------------------------
`stratego/belief/phase11b/corpus.py` is not imported by this module, not by
anything it imports, and not by the driver that runs it. Its
`Phase11BSetupSources.draw` returns canonical own-orientation tuples and its
glue hands them to `create_game` for Blue; every board Phase 15 builds
instead comes out of :class:`~.setups.Phase15SetupSources`, whose single
exit is :func:`~.orientation.oriented_for`.

The stopping unit is positions
------------------------------
Section 5: "The stopping unit is eligible observer positions, not completed
games." :func:`plan_cycle` emits a deterministically shuffled cycle whose
composition matches the section 6 mixture exactly, and the driver stops
when the split's position budget is met — so a run that ends early ends on
a balanced prefix rather than on a truncated corner of the design.

Two passes, and truth only in the second
----------------------------------------
The **public pass** plays the game: the observer reads a `PolicyInput`,
records the ply, the unresolved-piece count and a sha256 of its own
observation, and is structurally unable to reach a hidden rank. The
**privileged replay pass** rebuilds each selected decision from the action
history, checks the rebuilt observation against the digest the public pass
recorded — a bit-for-bit alignment proof, not a spot check — and only then
reads `dense_belief_target`. Nothing flows backwards.

The accepted termination cap is preserved
-----------------------------------------
Games are played under `EVALUATION_RULES` unchanged, so the accepted
`battleless_move_limit=200` / `absolute_move_limit=4000` caps bound every
trajectory and no pathological game can monopolize collection. Section 5
permits retiring a trajectory early; this collector instead plays each game
to its accepted termination, because evenly spaced sampling is defined over
the game's complete eligible list and the accepted cap already bounds the
cost. That is a choice, and the manifest records it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ...engine.constants import BLUE, PLAYER_NAMES, RED
from ...engine.coordinates import to_perspective
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
from ...training.belief_targets import dense_belief_target
from .contract import (
    CORPUS_SPLITS,
    DECISIONS_PER_GAME,
    LIBRARY_PARTITION,
    LIBRARY_SPLIT,
    OBSERVATION_SHAPE,
    OPPONENT_MIXTURE,
    OBSERVER_MIXTURE,
    RULE_OPPONENT_POLICY_IDS,
    SETUP_MIXTURE,
    Phase15Error,
)
from .orientation import assert_engine_orientation
from .seeds import (
    ROLE_OBSERVER,
    ROLE_OPPONENT,
    corpus_game_id,
    derive_phase15_seed,
    match_seed,
    setup_seed,
    DOMAIN_MATCH,
)

#: The suite identity every Phase 15 corpus game is played under.
CORPUS_RUN_VERSION = "phase15_belief_corpus_generation_v1"

#: The observer seat's policy identity, distinct from the opponent seat's so
#: `play_match` resolves two objects even in a self-play cell.
OBSERVER_POLICY_ID = "phase15_corpus_observer_v1"
NEURAL_OPPONENT_POLICY_ID = "phase15_corpus_neural_opponent_v1"

#: The decision mode of both neural seats. The accepted evaluation
#: convention; diversity comes from the setup mixture, not from sampling.
DECISION_MODE = DECISION_MODE_GREEDY

_PLAYER_OF = {"red": RED, "blue": BLUE}

#: The mixture is expressed in 5% units, so one cycle realises every share
#: exactly rather than approximately.
_MIXTURE_UNIT = 0.05


class Phase15CorpusError(Phase15Error):
    """A corpus game could not be planned, played, extracted or verified."""


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusGamePlan:
    """Everything one corpus game needs, resolved from its identity alone."""

    game_id: str
    split: str
    ordinal: int
    observer_model: str
    opponent: str
    setup_source: str
    observer_color: str
    opponent_color: str
    match_seed: int
    red_setup: "tuple[int, ...]"
    blue_setup: "tuple[int, ...]"
    observer_family_key: str
    opponent_family_key: str
    observer_base_setup_id: str
    opponent_base_setup_id: str
    observer_setup_branch: "str | None"
    opponent_setup_branch: "str | None"


def _units(mixture: dict) -> "list[str]":
    """A mixture expressed as a list of 5% units, in declared order."""
    units: list[str] = []
    for name, share in mixture.items():
        count = round(float(share) / _MIXTURE_UNIT)
        if abs(count * _MIXTURE_UNIT - float(share)) > 1e-9:
            raise Phase15CorpusError(
                f"share {share} of {name!r} is not a whole multiple of {_MIXTURE_UNIT}"
            )
        units.extend([name] * int(count))
    return units


def plan_cycle(split: str) -> "list[tuple[str, str, str]]":
    """One balanced cycle of `(observer, opponent, setup source)` cells.

    The cross product of the three mixtures in 5% units, so the cycle's
    composition equals the section 6 design exactly. The cycle is then
    permuted by a fixed Phase 15 stream, which is what makes an *early*
    prefix of a run balanced too — a corpus that stops on its position
    budget mid-cycle is still a fair sample of the design.
    """
    if split not in CORPUS_SPLITS:
        raise Phase15CorpusError(f"unknown split {split!r}")
    cells = [
        (observer, opponent, source)
        for observer in _units(OBSERVER_MIXTURE)
        for opponent in _units(OPPONENT_MIXTURE)
        for source in _units(SETUP_MIXTURE)
    ]
    order = np.random.default_rng(
        derive_phase15_seed(DOMAIN_MATCH, "plan_cycle", split) % (2**63)
    ).permutation(len(cells))
    return [cells[int(index)] for index in order]


def iter_plans(split: str, sources, *, limit: "int | None" = None):
    """Yield corpus game plans forever, cycle after cycle.

    Each cell is emitted as two consecutive games — observer Red then
    observer Blue — so colour is balanced inside every cell at every even
    prefix, which is section 6's "balance Red/Blue observer colour within
    every major cell" made structural.
    """
    library_split = LIBRARY_SPLIT[split]
    partition = LIBRARY_PARTITION[split]
    cycle = plan_cycle(split)
    produced = 0
    ordinal = 0
    while limit is None or produced < int(limit):
        for observer_model, opponent, setup_source in cycle:
            for observer_color in ("red", "blue"):
                opponent_color = "blue" if observer_color == "red" else "red"
                game_id = corpus_game_id(
                    split, observer_model, opponent, setup_source, observer_color, ordinal
                )
                observer_draw = sources.draw(
                    setup_source,
                    library_split,
                    observer_color,
                    setup_seed(game_id, ROLE_OBSERVER),
                    partition,
                )
                opponent_draw = sources.draw(
                    setup_source,
                    library_split,
                    opponent_color,
                    setup_seed(game_id, ROLE_OPPONENT),
                    partition,
                )
                red_draw, blue_draw = (
                    (observer_draw, opponent_draw)
                    if observer_color == "red"
                    else (opponent_draw, observer_draw)
                )
                # The production-path orientation assertion. `oriented_for`
                # already ran inside the source; this is the second, explicit
                # check on the exact tuples that reach `create_game`.
                assert_engine_orientation(red_draw.canonical, red_draw.engine, RED)
                assert_engine_orientation(blue_draw.canonical, blue_draw.engine, BLUE)
                yield CorpusGamePlan(
                    game_id=game_id,
                    split=split,
                    ordinal=ordinal,
                    observer_model=observer_model,
                    opponent=opponent,
                    setup_source=setup_source,
                    observer_color=observer_color,
                    opponent_color=opponent_color,
                    match_seed=match_seed(game_id),
                    red_setup=red_draw.engine,
                    blue_setup=blue_draw.engine,
                    observer_family_key=observer_draw.family_key,
                    opponent_family_key=opponent_draw.family_key,
                    observer_base_setup_id=observer_draw.base_setup_id,
                    opponent_base_setup_id=opponent_draw.base_setup_id,
                    observer_setup_branch=observer_draw.branch,
                    opponent_setup_branch=opponent_draw.branch,
                )
                produced += 1
                if limit is not None and produced >= int(limit):
                    return
            ordinal += 1


# ---------------------------------------------------------------------------
# The public pass
# ---------------------------------------------------------------------------


def observer_ref() -> PolicyRef:
    return PolicyRef(policy_id=OBSERVER_POLICY_ID, policy_version=CORPUS_RUN_VERSION)


class CorpusObserverPolicy(Policy):
    """The observer seat: a frozen P18/P24 decision, logging public facts.

    Records the ply, the unresolved-opponent-piece count and a digest of its
    own observation. It reads a `PolicyInput` and nothing else, so it is
    structurally incapable of seeing a hidden rank — and it records no
    probability of any kind, because Phase 15 trains its own belief heads
    and must not bake a policy's own belief output into the corpus.
    """

    requirements = PolicyRequirements(
        observation=True, legal_action_mask=True, public_view=True
    )
    description = (
        "Phase 15 corpus observer: a frozen Phase 14 candidate's greedy "
        "decision, logging only public decision facts."
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
        response = self.owner.serve(InferenceRequest.from_policy_input(request))
        if not hasattr(response, "absolute_action_id"):
            raise Phase15CorpusError(
                "the observer owner refused a decision: "
                f"{getattr(response, 'message', response)!r}"
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
            {"phase15_unresolved_opponent_pieces": unresolved},
        )

    def describe(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "decision_mode": DECISION_MODE,
            "holds_model_weights": False,
            "records_predictions": False,
        }


def build_spec(plan: CorpusGamePlan, opponent: PolicyRef) -> MatchSpec:
    return MatchSpec(
        candidate=observer_ref(),
        opponent=opponent,
        setup_pair_id=plan.ordinal,
        candidate_color=_PLAYER_OF[plan.observer_color],
        replicate=plan.ordinal,
        root_seed=plan.match_seed,
        suite_version=CORPUS_RUN_VERSION,
        setup_bank_version=(
            f"{CORPUS_RUN_VERSION}|sp={plan.split}|obs={plan.observer_model}"
            f"|opp={plan.opponent}|src={plan.setup_source}"
        ),
        rules=EVALUATION_RULES,
    )


def opponent_seat(plan: CorpusGamePlan, owners: dict):
    """`(ref, policy)` for the opponent of this game."""
    if plan.opponent in owners:
        ref = PolicyRef(
            policy_id=f"{NEURAL_OPPONENT_POLICY_ID}|{plan.opponent}",
            policy_version=CORPUS_RUN_VERSION,
        )
        return ref, RemoteNeuralPolicy(
            ref,
            LocalInferenceChannel(owners[plan.opponent]),
            decision_mode=DECISION_MODE,
        )
    policy = build_policy(RULE_OPPONENT_POLICY_IDS[plan.opponent])
    return policy.ref, policy


def play_corpus_game(plan: CorpusGamePlan, owners: dict):
    """Play one corpus game. Returns `(result, decisions)` — public only."""
    opponent_reference, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_reference)
    observer = CorpusObserverPolicy(observer_ref(), owners[plan.observer_model])
    policies = {observer_ref().token: observer}
    if opponent_reference.token != observer_ref().token:
        policies[opponent_reference.token] = (
            opponent_policy
            if plan.opponent in owners
            else FrozenSeedPolicy(opponent_policy, plan.match_seed)
        )
    result = play_match(
        spec,
        setups=(plan.red_setup, plan.blue_setup),
        policies=policies,
        record_actions=True,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    if result.errored:  # pragma: no cover - raises above under RAISE
        raise Phase15CorpusError(f"{plan.game_id} errored: {result.policy_error}")
    return result, observer.decisions


def evenly_spaced(values: "list", count: int) -> "list":
    """At most `count` evenly spaced elements, **both endpoints included**.

    The Phase 11B rule, restated in this namespace rather than imported, so
    the new collector shares no module with the contaminated one. It is
    deliberately *not* the accepted Phase 11 `values[(k * n) // take]` rule:
    that always starts at the first element and never reaches the last, so
    it systematically omits each game's final eligible decision — exactly
    the late positions with the fewest unresolved pieces and the most public
    evidence, where a belief model has the most to learn.
    """
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    step = (len(values) - 1) / (count - 1) if count > 1 else 0.0
    return [values[int(round(index * step))] for index in range(count)]


def select_decisions(
    decisions: "list[dict]", per_game: int = DECISIONS_PER_GAME
) -> "list[dict]":
    """The evenly spaced eligible observer decisions of one game.

    Eligibility here is the public half of section 5's definition — the
    observer is to act and at least one opponent piece is unresolved. The
    other two halves (public and privileged agree exactly; a non-empty
    legal hidden-rank target exists) are proved in the privileged pass,
    where the truth needed to check them is available.
    """
    eligible = [row for row in decisions if row["unresolved"] > 0]
    return evenly_spaced(eligible, int(per_game))


# ---------------------------------------------------------------------------
# The privileged replay pass
# ---------------------------------------------------------------------------


@dataclass
class ExtractedGame:
    """One game's contribution to the corpus: its selected samples."""

    game_id: str
    plies: int
    eligible_decisions: int
    samples: list


def privileged_extract(
    plan: CorpusGamePlan, result, selected: "list[dict]", eligible: int
) -> ExtractedGame:
    """Replay one game and build the samples of its selected decisions.

    Reads true ranks. Runs only after the game is over and every public
    fact already exists, and writes nothing back into any model input.
    """
    history = result.action_history
    if history is None:  # pragma: no cover - record_actions is always True
        raise Phase15CorpusError(f"{plan.game_id} recorded no action history")
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
        raise Phase15CorpusError(
            f"{plan.game_id}: replay produced {len(samples)} of {len(selected)} samples"
        )
    return ExtractedGame(
        game_id=plan.game_id,
        plies=int(result.plies),
        eligible_decisions=int(eligible),
        samples=samples,
    )


def _build_sample(
    plan: CorpusGamePlan, state, observer: int, ply: int, row: dict
) -> dict:
    """Public arrays and privileged labels for one decision.

    The rebuilt observation is checked against the digest the public pass
    recorded *before* anything privileged is read, so a sample can only ever
    carry the labels of the position it belongs to.
    """
    observation = build_observation(state, observer)
    if observation.shape != OBSERVATION_SHAPE:  # pragma: no cover - engine invariant
        raise Phase15CorpusError(f"observation shape {observation.shape} is not frozen")
    digest = hashlib.sha256(
        np.ascontiguousarray(observation, dtype=np.float32).tobytes()
    ).hexdigest()
    if digest != row["observation_sha256"]:
        raise Phase15CorpusError(
            f"{plan.game_id} ply {ply}: replayed observation {digest[:16]} != "
            f"recorded {row['observation_sha256'][:16]}"
        )

    view = build_public_view(state, observer)
    document = build_public_state_document(view, observation)
    identity = public_state_identity(document)
    counts = remaining_counts(document)
    hidden = hidden_opponent_pieces(document)
    if len(hidden) != int(row["unresolved"]):
        raise Phase15CorpusError(
            f"{plan.game_id} ply {ply}: {len(hidden)} hidden pieces on replay, "
            f"{row['unresolved']} recorded"
        )
    if not hidden:  # pragma: no cover - selection filters these out
        raise Phase15CorpusError(f"{plan.game_id} ply {ply} has no hidden piece")

    labels, target_mask = dense_belief_target(state, observer)

    pieces = []
    for piece in hidden:
        square = int(piece["current_square"])
        normalized = to_perspective(square, observer)
        if not target_mask[normalized]:
            raise Phase15CorpusError(
                f"{plan.game_id} ply {ply}: hidden piece at {square} is not a "
                "supervised square in the engine's belief target"
            )
        true_rank = int(labels[normalized])
        moved = bool(piece["has_moved"])
        mask = legal_rank_mask(moved)
        if not mask[true_rank]:
            raise Phase15CorpusError(
                f"{plan.game_id} ply {ply}: true rank {true_rank} is excluded by "
                "its own public legal-rank mask"
            )
        if counts[true_rank] <= 0:
            raise Phase15CorpusError(
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
        raise Phase15CorpusError(
            f"{plan.game_id} ply {ply}: engine supervises {int(target_mask.sum())} "
            f"squares, the public document names {len(pieces)}"
        )

    return {
        "game_id": plan.game_id,
        "split": plan.split,
        "observer_model": plan.observer_model,
        "opponent": plan.opponent,
        "setup_source": plan.setup_source,
        "observer_family_key": plan.observer_family_key,
        "opponent_family_key": plan.opponent_family_key,
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
    "DECISION_MODE",
    "CorpusGamePlan",
    "CorpusObserverPolicy",
    "ExtractedGame",
    "Phase15CorpusError",
    "build_spec",
    "evenly_spaced",
    "iter_plans",
    "observer_ref",
    "opponent_seat",
    "plan_cycle",
    "play_corpus_game",
    "privileged_extract",
    "select_decisions",
]
