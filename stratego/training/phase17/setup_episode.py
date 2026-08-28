"""Phase 17: the setup episode, the completed-episode buffer, and the runner API.

Specification sources:

- `09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md` sections 4 and 7
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` sections 8 and 10
- `reports/phase17/ataraxos_method_map_v1.md` rows S05, S06, S17

Module name
-----------
The episode schema and its pending buffer live here, generation lives in
`setup_sampling`, and the update lives in `setup_learning`. Every path is
inside the `phase17` setup namespace and none collides with the move half's
files.

The four things the runner must not be able to do
--------------------------------------------------
Silent dropping, double consumption, outcome rebinding and behavior-identity
mismatch must be *impossible or fatal*. Each has a mechanism here, not a
convention:

- dropping: every departure from the buffer increments a named counter, and
  the whole buffer is drained every iteration so nothing can age out of it;
- double consumption: an episode's `state` advances one way only, and
  `consume_all` refuses anything already `consumed`;
- outcome rebinding: `complete` refuses a second, different result;
- behavior identity: a batch is refused if its episodes carry a
  `setup_model_state_digest` that no longer matches what the caller declared.

An episode records the raw snapshot that generated it and keeps it: an old
completed game is valid only with its attached behavior identity, so the ratio
denominator is always the recorded probability and never a re-run of the latest
network.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ...engine.constants import BLUE, PLAYERS, RED
from .setup_contract import (
    ORIENTATION_RULE_VERSION,
    SETUP_EPISODE_SCHEMA_VERSION,
    SETUP_PREFIXES,
    SETUP_BUFFER_VERSION,
    Phase17SetupError,
    json_document_digest,
)
from .setup_sampling import SampledSetup

TERMINAL_RESULTS = ("red_win", "blue_win", "draw")
EPISODE_STATES = ("open", "queued", "consumed", "rejected")

#: W/D/L class order, shared with the value head. Index 0 is a win *for the
#: episode's owner*, so the same head reads correctly for both colours.
WDL_CLASSES = ("win", "draw", "loss")


def utc_now() -> str:
    """ISO-8601 UTC with a trailing Z, second resolution (encoding rules)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def outcome_for(terminal_result: str, owner_perspective: int) -> int:
    """`+1` win / `0` draw / `-1` loss, from the setup owner's own side.

    The sign is taken from `owner_perspective` rather than from a global
    winner field, because both sides of one game produce episodes and each
    must see the same game with the opposite sign.
    """
    if terminal_result not in TERMINAL_RESULTS:
        raise Phase17SetupError(f"unknown terminal result: {terminal_result!r}")
    if owner_perspective not in PLAYERS:
        raise Phase17SetupError(f"unknown owner perspective: {owner_perspective!r}")
    if terminal_result == "draw":
        return 0
    winner = RED if terminal_result == "red_win" else BLUE
    return 1 if winner == owner_perspective else -1


def wdl_class(outcome: int) -> int:
    """Map `+1 / 0 / -1` onto the value head's class index."""
    if outcome == 1:
        return 0
    if outcome == 0:
        return 1
    if outcome == -1:
        return 2
    raise Phase17SetupError(f"outcome must be -1, 0 or +1, got {outcome!r}")


@dataclass
class SetupEpisode:
    """One side's setup for one game, from sampling to outcome.

    Schema `phase17_setup_episode_v1`. Field names and meanings are Agent 1's
    and are not reinterpreted here.
    """

    run_id: str
    game_id: str
    color: int
    owner_perspective: int
    setup_model_state_digest: str
    setup_snapshot_iteration: int
    setup_root_seed: int
    per_token_seeds: tuple
    canonical_setup: tuple
    engine_setup: tuple
    canonical_fingerprint: str
    engine_fingerprint: str
    reflection_class_fingerprint: str
    orientation_rule_version: str
    tokens: np.ndarray
    inventory_masks: np.ndarray
    behavior_probabilities: np.ndarray
    behavior_log_probabilities: np.ndarray
    suffix_information_content: np.ndarray
    prefix_wdl_predictions: np.ndarray
    prefix_conditional_entropy_predictions: np.ndarray
    terminal_result: str | None = None
    outcome: int | None = None
    completed_utc: str | None = None
    enqueued_utc: str | None = None
    consumed_utc: str | None = None
    consumed_in_setup_iteration: int | None = None
    policy_age_iterations: int | None = None
    state: str = "open"
    rejected_reason: str | None = None
    schema_version: str = SETUP_EPISODE_SCHEMA_VERSION

    # -- construction ----------------------------------------------------

    @classmethod
    def create(
        cls, sample: SampledSetup, *, run_id: str, game_id: str
    ) -> "SetupEpisode":
        """Build an open episode from a sampled setup without recomputation."""
        from ...setups.identity import content_fingerprint

        return cls(
            run_id=run_id,
            game_id=game_id,
            color=int(sample.color),
            owner_perspective=int(sample.color),
            setup_model_state_digest=sample.setup_model_state_digest,
            setup_snapshot_iteration=int(sample.setup_snapshot_iteration),
            setup_root_seed=int(sample.root_seed),
            per_token_seeds=tuple(sample.per_token_seeds),
            canonical_setup=tuple(sample.canonical_setup),
            engine_setup=tuple(sample.engine_setup),
            canonical_fingerprint=sample.canonical_fingerprint,
            engine_fingerprint=content_fingerprint(sample.engine_setup),
            reflection_class_fingerprint=sample.reflection_class_fingerprint,
            orientation_rule_version=sample.orientation_rule_version,
            tokens=np.asarray(sample.tokens, dtype=np.int8),
            inventory_masks=np.asarray(sample.inventory_masks, dtype=bool),
            behavior_probabilities=np.asarray(sample.behavior_probabilities, dtype=np.float32),
            behavior_log_probabilities=np.asarray(
                sample.behavior_log_probabilities, dtype=np.float32
            ),
            suffix_information_content=np.asarray(
                sample.suffix_information_content, dtype=np.float32
            ),
            prefix_wdl_predictions=np.asarray(sample.prefix_wdl_predictions, dtype=np.float32),
            prefix_conditional_entropy_predictions=np.asarray(
                sample.prefix_conditional_entropy_predictions, dtype=np.float32
            ),
        )

    def __post_init__(self) -> None:
        if self.color not in PLAYERS:
            raise Phase17SetupError(f"unknown colour: {self.color!r}")
        if self.owner_perspective != self.color:
            raise Phase17SetupError(
                "owner_perspective must equal color; it is recorded separately "
                "only because the outcome sign is taken from it"
            )
        if self.orientation_rule_version != ORIENTATION_RULE_VERSION:
            raise Phase17SetupError(
                f"episode carries orientation rule {self.orientation_rule_version!r}, "
                f"contract freezes {ORIENTATION_RULE_VERSION!r}"
            )
        for name, expected in (
            ("tokens", (SETUP_PREFIXES,)),
            ("inventory_masks", (SETUP_PREFIXES, 12)),
            ("behavior_probabilities", (SETUP_PREFIXES, 12)),
            ("behavior_log_probabilities", (SETUP_PREFIXES,)),
            ("suffix_information_content", (SETUP_PREFIXES,)),
            ("prefix_wdl_predictions", (SETUP_PREFIXES, 3)),
            ("prefix_conditional_entropy_predictions", (SETUP_PREFIXES,)),
        ):
            actual = tuple(np.asarray(getattr(self, name)).shape)
            if actual != expected:
                raise Phase17SetupError(f"{name} has shape {actual}, expected {expected}")
        if len(self.per_token_seeds) != SETUP_PREFIXES:
            raise Phase17SetupError("per_token_seeds must hold one seed per prefix")

    # -- lifecycle -------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        return self.terminal_result is not None

    def complete(self, terminal_result: str) -> "SetupEpisode":
        """Bind the terminal result. Refuses a second, different result.

        Outcome rebinding is a section 8 impossibility, so this is fatal
        rather than idempotent-with-a-warning: a game whose result changed
        means two different games were confused for one another.
        """
        resolved = outcome_for(terminal_result, self.owner_perspective)
        if self.terminal_result is not None:
            if self.terminal_result != terminal_result:
                raise Phase17SetupError(
                    f"episode {self.game_id}/{self.color} already bound to "
                    f"{self.terminal_result!r}; refusing to rebind to {terminal_result!r}"
                )
            return self
        self.terminal_result = terminal_result
        self.outcome = resolved
        self.completed_utc = utc_now()
        return self

    def policy_age(self, current_setup_iteration: int) -> int:
        """How many setup updates have landed since this episode was drawn."""
        return int(current_setup_iteration) - int(self.setup_snapshot_iteration)

    def compatible_with(self, setup_model_state_digest: str) -> bool:
        """Whether this episode's behavior identity is the given snapshot.

        Deliberately *not* a training precondition: an episode is off-policy
        by construction (row S05), and the ratio denominator is its own
        recorded probability. This exists so a caller can report policy age
        and so a batch can refuse an episode whose identity was overwritten.
        """
        return self.setup_model_state_digest == setup_model_state_digest

    # -- serialization ---------------------------------------------------

    def to_document(self, include_arrays: bool = True) -> dict:
        """The JSON-safe record. Arrays become lists; nothing is dropped."""
        document = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "color": int(self.color),
            "owner_perspective": int(self.owner_perspective),
            "setup_model_state_digest": self.setup_model_state_digest,
            "setup_snapshot_iteration": int(self.setup_snapshot_iteration),
            "setup_root_seed": int(self.setup_root_seed),
            "canonical_setup": [int(value) for value in self.canonical_setup],
            "engine_setup": [int(value) for value in self.engine_setup],
            "canonical_fingerprint": self.canonical_fingerprint,
            "engine_fingerprint": self.engine_fingerprint,
            "reflection_class_fingerprint": self.reflection_class_fingerprint,
            "orientation_rule_version": self.orientation_rule_version,
            "terminal_result": self.terminal_result,
            "outcome": None if self.outcome is None else int(self.outcome),
            "completed_utc": self.completed_utc,
            "enqueued_utc": self.enqueued_utc,
            "consumed_utc": self.consumed_utc,
            "consumed_in_setup_iteration": self.consumed_in_setup_iteration,
            "policy_age_iterations": self.policy_age_iterations,
            "state": self.state,
            "rejected_reason": self.rejected_reason,
        }
        if include_arrays:
            document.update(
                {
                    "per_token_seeds": [int(value) for value in self.per_token_seeds],
                    "tokens": [int(value) for value in self.tokens],
                    "inventory_masks": self.inventory_masks.astype(np.uint8).tolist(),
                    "behavior_probabilities": self.behavior_probabilities.astype(
                        np.float32
                    ).tolist(),
                    "behavior_log_probabilities": self.behavior_log_probabilities.astype(
                        np.float32
                    ).tolist(),
                    "suffix_information_content": self.suffix_information_content.astype(
                        np.float32
                    ).tolist(),
                    "prefix_wdl_predictions": self.prefix_wdl_predictions.astype(
                        np.float32
                    ).tolist(),
                    "prefix_conditional_entropy_predictions": (
                        self.prefix_conditional_entropy_predictions.astype(np.float32).tolist()
                    ),
                }
            )
        return document

    @classmethod
    def from_document(cls, document: dict) -> "SetupEpisode":
        """Rebuild an episode, refusing anything the schema does not cover."""
        version = document.get("schema_version")
        if version != SETUP_EPISODE_SCHEMA_VERSION:
            raise Phase17SetupError(
                f"episode schema {version!r}, expected {SETUP_EPISODE_SCHEMA_VERSION!r}"
            )
        required = (
            "per_token_seeds",
            "tokens",
            "inventory_masks",
            "behavior_probabilities",
            "behavior_log_probabilities",
            "suffix_information_content",
            "prefix_wdl_predictions",
            "prefix_conditional_entropy_predictions",
        )
        for name in required:
            if name not in document:
                raise Phase17SetupError(f"episode document is missing {name!r}")
        episode = cls(
            run_id=document["run_id"],
            game_id=document["game_id"],
            color=int(document["color"]),
            owner_perspective=int(document["owner_perspective"]),
            setup_model_state_digest=document["setup_model_state_digest"],
            setup_snapshot_iteration=int(document["setup_snapshot_iteration"]),
            setup_root_seed=int(document["setup_root_seed"]),
            per_token_seeds=tuple(int(value) for value in document["per_token_seeds"]),
            canonical_setup=tuple(int(value) for value in document["canonical_setup"]),
            engine_setup=tuple(int(value) for value in document["engine_setup"]),
            canonical_fingerprint=document["canonical_fingerprint"],
            engine_fingerprint=document["engine_fingerprint"],
            reflection_class_fingerprint=document["reflection_class_fingerprint"],
            orientation_rule_version=document["orientation_rule_version"],
            tokens=np.array(document["tokens"], dtype=np.int8),
            inventory_masks=np.array(document["inventory_masks"], dtype=bool),
            behavior_probabilities=np.array(
                document["behavior_probabilities"], dtype=np.float32
            ),
            behavior_log_probabilities=np.array(
                document["behavior_log_probabilities"], dtype=np.float32
            ),
            suffix_information_content=np.array(
                document["suffix_information_content"], dtype=np.float32
            ),
            prefix_wdl_predictions=np.array(document["prefix_wdl_predictions"], dtype=np.float32),
            prefix_conditional_entropy_predictions=np.array(
                document["prefix_conditional_entropy_predictions"], dtype=np.float32
            ),
        )
        episode.terminal_result = document["terminal_result"]
        episode.outcome = document["outcome"]
        episode.completed_utc = document["completed_utc"]
        episode.enqueued_utc = document["enqueued_utc"]
        episode.consumed_utc = document["consumed_utc"]
        episode.consumed_in_setup_iteration = document["consumed_in_setup_iteration"]
        episode.policy_age_iterations = document["policy_age_iterations"]
        episode.state = document["state"]
        episode.rejected_reason = document["rejected_reason"]
        return episode

    def identity(self) -> str:
        """A content digest over the episode's identity fields."""
        return json_document_digest(self.to_document(include_arrays=False))


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


@dataclass
class QueueTelemetry:
    """Everything the completed-episode buffer records each iteration."""

    depth: int
    oldest_age: int | None
    mean_age: float | None
    max_policy_age: int | None
    enqueued_count: int
    consumed_count: int
    rejected_count: int


class SetupEpisodeQueue:
    """The pending buffer of completed setup episodes, each consumed once.

    Under operator decision D10 this is a *pending buffer*, not a backlog. Every
    episode whose game completed during the current fixed-transition iteration
    is trained on in that iteration and then removed, so the buffer is empty
    between iterations and holds at most one window's arrivals while a window is
    open. It exists so that a crash cannot lose or duplicate an outcome before
    the iteration closes -- nothing more.

    What D10 removed, and why none of it is still here
    ---------------------------------------------------
    Agent 3's version was a bounded FIFO that raised at a frozen capacity, and
    Agent 4 wrapped it in a fixed quota, a two-budget warm-up and a backlog
    alarm. All of that existed to keep a *fixed-size* batch unbiased while the
    queue carried episodes across iterations. A total drain has no such
    problem: the batch is exactly what arrived, so there is no count to hold
    constant, no arrival-rate margin to maintain, and no depth for an alarm to
    watch. Keeping the capacity would only reintroduce a way for a large window
    to kill the run.

    Duplicate detection is scoped to what is pending, for the same reason. A
    consumed episode's game has been retired and its `(run, game, colour)` key
    can never be offered again -- game ids only ever gain a new draw number --
    so a whole-run ledger would add several hundred thousand keys to every
    checkpoint and detect nothing a pending-set check does not.
    """

    version = SETUP_BUFFER_VERSION

    def __init__(self) -> None:
        self._queue: "deque[SetupEpisode]" = deque()
        self._seen: set = set()
        self.enqueued_count = 0
        self.consumed_count = 0
        self.rejected_count = 0
        self.rejections: "list[dict]" = []

    def __len__(self) -> int:
        return len(self._queue)

    @staticmethod
    def key(episode: SetupEpisode) -> tuple:
        return (episode.run_id, episode.game_id, int(episode.color))

    def _reject(self, episode: SetupEpisode, reason: str) -> None:
        episode.state = "rejected"
        episode.rejected_reason = reason
        self.rejected_count += 1
        self.rejections.append(
            {
                "game_id": episode.game_id,
                "color": int(episode.color),
                "reason": reason,
                "utc": utc_now(),
            }
        )

    def enqueue(self, episode: SetupEpisode) -> bool:
        """Validate and enqueue one completed episode.

        Returns whether it entered the buffer. A refusal is *recorded* with a
        reason -- the one thing this class may never do is discard quietly.
        """
        if not episode.is_complete:
            self._reject(episode, "episode has no terminal result")
            return False
        if self.key(episode) in self._seen:
            self._reject(episode, "duplicate (run_id, game_id, color)")
            return False
        if episode.state == "consumed":
            self._reject(episode, "already consumed")
            return False
        episode.state = "queued"
        episode.enqueued_utc = utc_now()
        self._queue.append(episode)
        self._seen.add(self.key(episode))
        self.enqueued_count += 1
        return True

    def consume_all(self, *, setup_iteration: int) -> "list[SetupEpisode]":
        """Take every pending episode, in FIFO order, each exactly once.

        The whole buffer, never a slice: D10 section 4 trains on every episode
        whose game completed during the iteration, with both sides represented.
        An empty buffer returns an empty list and the caller records a skipped
        setup update.
        """
        taken: "list[SetupEpisode]" = []
        while self._queue:
            episode = self._queue.popleft()
            if episode.state == "consumed":
                raise Phase17SetupError(
                    f"episode {episode.game_id}/{episode.color} is already consumed"
                )
            episode.state = "consumed"
            episode.consumed_utc = utc_now()
            episode.consumed_in_setup_iteration = int(setup_iteration)
            episode.policy_age_iterations = episode.policy_age(setup_iteration)
            taken.append(episode)
        self.consumed_count += len(taken)
        self._seen = {self.key(episode) for episode in self._queue}
        return taken

    def ages(self, setup_iteration: int) -> "list[int]":
        return [episode.policy_age(setup_iteration) for episode in self._queue]

    def telemetry(self, setup_iteration: int) -> QueueTelemetry:
        ages = self.ages(setup_iteration)
        return QueueTelemetry(
            depth=len(self._queue),
            oldest_age=max(ages) if ages else None,
            mean_age=float(sum(ages) / len(ages)) if ages else None,
            max_policy_age=max(ages) if ages else None,
            enqueued_count=self.enqueued_count,
            consumed_count=self.consumed_count,
            rejected_count=self.rejected_count,
        )

    def state_document(self) -> dict:
        """The pending buffer, for the paired checkpoint.

        `seen` is exactly the pending set, so a resume can neither lose an
        outcome that had arrived nor accept it twice.
        """
        return {
            "buffer_version": self.version,
            "enqueued_count": self.enqueued_count,
            "consumed_count": self.consumed_count,
            "rejected_count": self.rejected_count,
            "rejections": list(self.rejections),
            "seen": [list(key) for key in sorted(self._seen)],
            "episodes": [episode.to_document() for episode in self._queue],
        }

    @classmethod
    def from_state_document(cls, document: dict) -> "SetupEpisodeQueue":
        version = document.get("buffer_version")
        if version != SETUP_BUFFER_VERSION:
            raise Phase17SetupError(
                f"completed-setup buffer schema {version!r}, expected "
                f"{SETUP_BUFFER_VERSION!r}"
            )
        queue = cls()
        queue.enqueued_count = int(document["enqueued_count"])
        queue.consumed_count = int(document["consumed_count"])
        queue.rejected_count = int(document["rejected_count"])
        queue.rejections = list(document["rejections"])
        queue._seen = {tuple(key) for key in document["seen"]}
        for entry in document["episodes"]:
            queue._queue.append(SetupEpisode.from_document(entry))
        if {queue.key(episode) for episode in queue._queue} != queue._seen:
            raise Phase17SetupError(
                "the restored buffer's duplicate ledger does not match its own "
                "pending episodes; refusing a state that could lose or "
                "duplicate a completed setup outcome"
            )
        return queue


# ---------------------------------------------------------------------------
# The Agent 4-facing operations (section 8)
# ---------------------------------------------------------------------------


@dataclass
class GameSetupEpisodes:
    """The two episodes of one game, held together so neither can be orphaned."""

    game_id: str
    red: SetupEpisode
    blue: SetupEpisode
    completed: bool = False

    def both(self) -> "list[SetupEpisode]":
        return [self.red, self.blue]

    def complete(self, terminal_result: str) -> "list[SetupEpisode]":
        """Bind the same result to both sides, from their own perspectives.

        One call, two episodes: binding them separately is how a game ends up
        with a Red episode that saw a win and a Blue episode that never
        completed at all.
        """
        self.red.complete(terminal_result)
        self.blue.complete(terminal_result)
        self.completed = True
        return self.both()

    def engine_setups(self) -> "tuple[tuple[int, ...], tuple[int, ...]]":
        """`(red_setup, blue_setup)` ready for `create_game`, in engine frame."""
        return self.red.engine_setup, self.blue.engine_setup


def attach_setup_episodes(
    red_sample: SampledSetup, blue_sample: SampledSetup, *, run_id: str, game_id: str
) -> GameSetupEpisodes:
    """Create and attach both setup episodes of one game."""
    if int(red_sample.color) != RED or int(blue_sample.color) != BLUE:
        raise Phase17SetupError(
            "attach_setup_episodes expects a RED sample and a BLUE sample, in that order"
        )
    return GameSetupEpisodes(
        game_id=game_id,
        red=SetupEpisode.create(red_sample, run_id=run_id, game_id=game_id),
        blue=SetupEpisode.create(blue_sample, run_id=run_id, game_id=game_id),
    )


__all__ = [
    "EPISODE_STATES",
    "GameSetupEpisodes",
    "QueueTelemetry",
    "SetupEpisode",
    "SetupEpisodeQueue",
    "TERMINAL_RESULTS",
    "WDL_CLASSES",
    "attach_setup_episodes",
    "outcome_for",
    "utc_now",
    "wdl_class",
]
