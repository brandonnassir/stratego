"""Phase 10 Agent 1: the frozen 16,384-game setup-outcome corpus schedule.

Specification sources:

- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Controlled setup-outcome
  corpus")
- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze the 16,384-game
  outcome schedule", "Handoff to Agent 2")

The schedule is arithmetic, not sampling
----------------------------------------
```text
16 x 16 = 256 ordered family pairs
64 games per ordered family pair
16,384 games total
```

Counts are *scheduled*, never drawn: :func:`enumerate_schedule` is a pure
nested loop over the frozen family order and the frozen per-pair ordinal
range, so the corpus's shape cannot depend on a seed, a worker count, or an
arrival order. Ordering matters — `(F03, F11)` and `(F11, F03)` are two
distinct scheduled pairs — because the red-first intercept is a real effect
the utility fit has to be able to separate from setup quality.

This module never touches the filesystem and never resolves a path;
:mod:`stratego.training.phase10_storage` owns that, and the separation is
the structural proof that a corpus written at one path and copied elsewhere
is the same corpus.

Family-conditioned side draws
-----------------------------
Both sides of a game draw from the **train split only**, each conditioned on
its scheduled family, by the accepted Phase 9 bank rule reused verbatim:
walk ``attempt = 0, 1, 2, ...`` through the frozen `setup_sampler_v1` and
accept the first draw whose primary family matches. Every accepted draw is a
complete, untouched sampler output — base choice, reflection, perturbation,
validation stack and provenance included — so the conditional distribution
is exactly `neutral_v1` given the family and the corpus inherits the frozen
Phase 7 semantics wholesale. Acceptance probability is 1/16 per attempt; the
frozen 2,048-attempt ceiling has failure probability below 1e-57 and exists
only to make a broken library loud.

What the corpus is for, and what it may never be
------------------------------------------------
These 16,384 games are the *only* Phase 10 utility-training data. They are
played entirely on train-split setups, so no held-out base can reach either
utility model. No Phase 10 agent may use a corpus outcome to select a
candidate — selection happens on the validation bank and final acceptance on
the sealed test bank.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..setups.contracts import SPLIT_TRAIN
from ..setups.families import FAMILY_IDS
from .phase10_seed import (
    COLORS,
    PHASE10_MASTER_SEED,
    PHASE10_OUTCOME_VERSION,
    Phase10SeedError,
    corpus_match_seed,
    corpus_setup_seed,
    parse_phase10_game_id,
    phase10_game_id,
)

#: Version of the controlled setup-outcome corpus.
CORPUS_VERSION = "phase10_setup_outcome_corpus_v1"

#: The frozen schedule arithmetic.
FAMILY_COUNT = len(FAMILY_IDS)
ORDERED_FAMILY_PAIRS = FAMILY_COUNT * FAMILY_COUNT
GAMES_PER_ORDERED_PAIR = 64
TOTAL_CORPUS_GAMES = ORDERED_FAMILY_PAIRS * GAMES_PER_ORDERED_PAIR
assert (FAMILY_COUNT, ORDERED_FAMILY_PAIRS, TOTAL_CORPUS_GAMES) == (16, 256, 16_384)

#: Corpus setups come from the train split and nowhere else.
CORPUS_SPLIT = SPLIT_TRAIN

#: The frozen sampler profile of every corpus side draw.
CORPUS_SAMPLER_PROFILE = "neutral_v1"

#: Frozen family-rejection ceiling, the accepted Phase 9 bank value.
MAX_FAMILY_ATTEMPTS = 2048

#: Both sides play the accepted Phase 9 checkpoint under one frozen
#: behaviour. Recorded as text so the corpus record names the behaviour
#: rather than relying on a default somewhere in the inference stack.
CORPUS_MOVE_BEHAVIOR = {
    "policy": "accepted Phase 9 checkpoint, both sides",
    "checkpoint": "checkpoints/phase9/selfplay_c1_v1.pt",
    "decision_mode": "greedy",
    "dtype": "float32",
    "batch_policy": "single_request",
    "search": "none",
    "optimizer_steps": 0,
}


class Phase10ScheduleError(ValueError):
    """Raised when a Phase 10 corpus schedule condition is violated."""


@dataclass(frozen=True)
class ScheduledGame:
    """One logical corpus game, complete before any outcome exists."""

    game_id: str
    red_family: str
    blue_family: str
    ordinal: int
    match_seed: int

    def side_family(self, color: str) -> str:
        if color == "red":
            return self.red_family
        if color == "blue":
            return self.blue_family
        raise Phase10ScheduleError(f"colour must be one of {list(COLORS)}, got {color!r}")

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "red_family": self.red_family,
            "blue_family": self.blue_family,
            "ordinal": self.ordinal,
            "match_seed": self.match_seed,
            "split": CORPUS_SPLIT,
        }


def ordered_family_pairs() -> "tuple[tuple[str, str], ...]":
    """The 256 ordered `(red family, blue family)` pairs, in frozen order."""
    return tuple(
        (red_family, blue_family)
        for red_family in FAMILY_IDS
        for blue_family in FAMILY_IDS
    )


def enumerate_schedule() -> "tuple[ScheduledGame, ...]":
    """Every scheduled corpus game, in the frozen deterministic order.

    Pure arithmetic over frozen constants: no seed chooses a count, no path
    appears anywhere, and the result is byte-identical in every process.
    """
    games: list[ScheduledGame] = []
    for red_family, blue_family in ordered_family_pairs():
        for ordinal in range(GAMES_PER_ORDERED_PAIR):
            game_id = phase10_game_id(red_family, blue_family, ordinal)
            games.append(
                ScheduledGame(
                    game_id=game_id,
                    red_family=red_family,
                    blue_family=blue_family,
                    ordinal=ordinal,
                    match_seed=corpus_match_seed(game_id),
                )
            )
    return tuple(games)


def rebuild_game(game_id: str) -> ScheduledGame:
    """One scheduled game rebuilt from its identifier alone.

    The crash/resume primitive handed to Agent 2: a missing game is
    regenerated without enumerating, or even knowing about, any other game.
    """
    fields = parse_phase10_game_id(game_id)
    if not 0 <= fields["ordinal"] < GAMES_PER_ORDERED_PAIR:
        raise Phase10ScheduleError(
            f"{game_id}: ordinal {fields['ordinal']} is outside "
            f"0..{GAMES_PER_ORDERED_PAIR - 1}"
        )
    for key in ("red_family", "blue_family"):
        if fields[key] not in FAMILY_IDS:
            raise Phase10ScheduleError(f"{game_id}: unknown {key} {fields[key]!r}")
    return ScheduledGame(
        game_id=game_id,
        red_family=fields["red_family"],
        blue_family=fields["blue_family"],
        ordinal=fields["ordinal"],
        match_seed=corpus_match_seed(game_id),
    )


def resolve_side(game_id: str, color: str, *, index=None):
    """`(SampledSetup, attempt, draw_seed)` for one side of one corpus game.

    Deterministic rejection over the untouched frozen sampler: the first
    attempt whose primary family equals the side's scheduled family wins.
    Imported lazily so that this module — the logical schedule — stays free
    of the library-loading machinery and can be imported by anything.
    """
    from ..setups.sampler import load_library_index, sample_setup

    game = rebuild_game(game_id)
    family_id = game.side_family(color)
    library = load_library_index() if index is None else index
    for attempt in range(MAX_FAMILY_ATTEMPTS):
        seed = corpus_setup_seed(game_id, color, attempt)
        sampled = sample_setup(
            CORPUS_SPLIT, seed, profile=CORPUS_SAMPLER_PROFILE, index=library
        )
        if sampled.family_id == family_id:
            return sampled, attempt, seed
    raise Phase10ScheduleError(
        f"{game_id} {color}: no {family_id} draw within {MAX_FAMILY_ATTEMPTS} "
        "attempts; the library or sampler has drifted (BLOCKED)"
    )


# ---------------------------------------------------------------------------
# The outcome record
# ---------------------------------------------------------------------------

#: The frozen outcome-record schema. Every field is required, and together
#: they are enough to replay one game independently: the identity, both
#: setups' complete provenance, the played result, and the digests that name
#: the software and data the game ran against.
OUTCOME_RECORD_FIELDS = (
    ("corpus_version", "the corpus version this record belongs to"),
    ("game_id", "the logical Phase 10 outcome game id"),
    ("red_family", "scheduled Red setup family"),
    ("blue_family", "scheduled Blue setup family"),
    ("ordinal", "the game's ordinal inside its ordered family pair"),
    ("split", "the setup split both sides drew from; always 'train'"),
    ("match_seed", "the frozen match-level randomness seed"),
    ("red_setup_draw_seed", "the accepted sampler draw seed of the Red side"),
    ("blue_setup_draw_seed", "the accepted sampler draw seed of the Blue side"),
    ("red_setup_attempt", "the accepted family-rejection attempt index for Red"),
    ("blue_setup_attempt", "the accepted family-rejection attempt index for Blue"),
    ("red_base_setup_id", "the Red side's Phase 7 base identity"),
    ("blue_base_setup_id", "the Blue side's Phase 7 base identity"),
    ("red_provenance", "the Red side's complete setup_sampler_v1 provenance record"),
    ("blue_provenance", "the Blue side's complete setup_sampler_v1 provenance record"),
    ("trait_schema_version", "the trait-vector identity both sides were scored under"),
    ("result", "'red_win', 'draw' or 'red_loss', from the Red perspective"),
    ("red_score", "the Red-perspective target 1.0 / 0.5 / 0.0"),
    ("plies", "game length in plies"),
    ("terminal_reason", "the engine's terminal reason token"),
    ("move_policy_identity", "the move-policy identity both sides played under"),
    ("move_checkpoint_sha256", "the accepted Phase 9 checkpoint file SHA-256"),
    ("move_model_state_digest", "the accepted Phase 9 model-state digest"),
    ("library_content_digest", "the Phase 7 library content digest in force"),
    ("payload_digest", "SHA-256 over this record's canonical outcome payload"),
    ("metadata_digest", "SHA-256 over the shard/segment metadata this record sits in"),
    ("commit_digest", "the corpus commit identity that sealed this record"),
)

#: Result tokens and their frozen Red-perspective targets.
RESULT_TARGETS = {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}


def outcome_record_schema() -> dict:
    """The machine-readable outcome-record schema handed to Agent 2."""
    return {
        "corpus_version": CORPUS_VERSION,
        "fields": [{"name": name, "description": text} for name, text in OUTCOME_RECORD_FIELDS],
        "field_count": len(OUTCOME_RECORD_FIELDS),
        "result_targets": dict(RESULT_TARGETS),
        "target_orientation": "red perspective",
        "replay_claim": (
            "game_id rebuilds both side draw seeds; each side's provenance rebuilds "
            "its exact final setup through the frozen sampler; the move-policy "
            "identity and checkpoint digests name the weights that moved; so any "
            "single record replays independently of every other record"
        ),
        "forbidden_fields": (
            "no opponent-private information, no model score, no strength signal, "
            "and no physical storage path may appear in a record"
        ),
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_schedule(schedule: "tuple[ScheduledGame, ...] | None" = None) -> dict:
    """Recompute every structural property of the frozen schedule.

    Proves exact pair counts, exact totals, id uniqueness, seed uniqueness,
    train-only use, and rebuild exactness from identity alone. Plays no game
    and reads no outcome.
    """
    games = enumerate_schedule() if schedule is None else tuple(schedule)

    pair_counts: dict = {}
    game_ids: list[str] = []
    match_seeds: list[int] = []
    setup_seeds: list[int] = []
    rebuild_failures: list[str] = []
    split_violations: list[str] = []

    for game in games:
        pair_counts[(game.red_family, game.blue_family)] = (
            pair_counts.get((game.red_family, game.blue_family), 0) + 1
        )
        game_ids.append(game.game_id)
        match_seeds.append(game.match_seed)
        for color in COLORS:
            setup_seeds.append(corpus_setup_seed(game.game_id, color, 0))
        if rebuild_game(game.game_id) != game:
            rebuild_failures.append(f"{game.game_id}: isolated rebuild differs")

    expected_pairs = set(ordered_family_pairs())
    distinct_ids = len(set(game_ids))
    distinct_match = len(set(match_seeds))
    distinct_setup = len(set(setup_seeds))

    checks = {
        "total_games_exact": len(games) == TOTAL_CORPUS_GAMES,
        "ordered_pair_count_exact": len(pair_counts) == ORDERED_FAMILY_PAIRS,
        "ordered_pairs_complete": set(pair_counts) == expected_pairs,
        "games_per_pair_exact": all(
            count == GAMES_PER_ORDERED_PAIR for count in pair_counts.values()
        ),
        "game_ids_unique": distinct_ids == len(game_ids),
        "match_seeds_unique": distinct_match == len(match_seeds),
        "setup_seeds_unique": distinct_setup == len(setup_seeds),
        "seed_streams_disjoint": not (set(match_seeds) & set(setup_seeds)),
        "isolated_rebuild_exact": not rebuild_failures,
        "train_split_only": not split_violations and CORPUS_SPLIT == SPLIT_TRAIN,
        "no_path_in_identity": all("/" not in game.game_id for game in games),
    }

    return {
        "corpus_version": CORPUS_VERSION,
        "total_games": len(games),
        "ordered_pair_count": len(pair_counts),
        "games_per_ordered_pair": sorted(set(pair_counts.values())),
        "distinct_game_ids": distinct_ids,
        "distinct_match_seeds": distinct_match,
        "distinct_first_attempt_setup_seeds": distinct_setup,
        "rebuild_failures": rebuild_failures,
        "split_violations": split_violations,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def schedule_digest(schedule: "tuple[ScheduledGame, ...] | None" = None) -> str:
    """SHA-256 over the complete frozen schedule's canonical JSON."""
    games = enumerate_schedule() if schedule is None else tuple(schedule)
    payload = {
        "corpus_version": CORPUS_VERSION,
        "outcome_version": PHASE10_OUTCOME_VERSION,
        "master_seed": PHASE10_MASTER_SEED,
        "games": [game.to_dict() for game in games],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def corpus_contract_document() -> dict:
    """`phase10_setup_outcome_corpus_v1` — the complete frozen corpus contract."""
    return {
        "corpus_version": CORPUS_VERSION,
        "outcome_version": PHASE10_OUTCOME_VERSION,
        "master_seed": PHASE10_MASTER_SEED,
        "family_count": FAMILY_COUNT,
        "family_order": list(FAMILY_IDS),
        "ordered_family_pairs": ORDERED_FAMILY_PAIRS,
        "games_per_ordered_pair": GAMES_PER_ORDERED_PAIR,
        "total_games": TOTAL_CORPUS_GAMES,
        "arithmetic": "256 ordered family pairs x 64 games = 16,384",
        "split": CORPUS_SPLIT,
        "held_out_bases_used": 0,
        "sampler_profile": CORPUS_SAMPLER_PROFILE,
        "side_draw_rule": (
            "side draw = first attempt k with sample_setup('train', "
            "corpus_setup_seed(game_id, color, k), profile='neutral_v1').family_id "
            "== the side's scheduled family"
        ),
        "max_family_attempts": MAX_FAMILY_ATTEMPTS,
        "conditional_distribution": (
            "exactly neutral_v1 conditioned on the scheduled family: the sampler's "
            "family draw is uniform and independent of its base, reflection, "
            "perturbation and intensity streams, so rejecting on family leaves "
            "every other accepted Phase 7 decision untouched"
        ),
        "move_behavior": dict(CORPUS_MOVE_BEHAVIOR),
        "outcome_record_schema": outcome_record_schema(),
        "identity_independence": (
            "no game id, seed or record field depends on worker count, task "
            "arrival order, process id, wall clock, or a physical storage path"
        ),
        "selection_prohibition": (
            "corpus outcomes fit the two utility models and nothing else; no "
            "candidate may be selected, and no threshold changed, from them"
        ),
    }


__all__ = [
    "CORPUS_MOVE_BEHAVIOR",
    "CORPUS_SAMPLER_PROFILE",
    "CORPUS_SPLIT",
    "CORPUS_VERSION",
    "FAMILY_COUNT",
    "GAMES_PER_ORDERED_PAIR",
    "MAX_FAMILY_ATTEMPTS",
    "ORDERED_FAMILY_PAIRS",
    "OUTCOME_RECORD_FIELDS",
    "RESULT_TARGETS",
    "TOTAL_CORPUS_GAMES",
    "Phase10ScheduleError",
    "ScheduledGame",
    "audit_schedule",
    "corpus_contract_document",
    "enumerate_schedule",
    "ordered_family_pairs",
    "outcome_record_schema",
    "rebuild_game",
    "resolve_side",
    "schedule_digest",
]
