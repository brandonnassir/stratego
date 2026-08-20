"""The fresh Phase 12 diagnostic position set.

Specification source: `03_PHASE_12_AGENT_2_BELIEF_DECISION_DIAGNOSTIC.md`
section 2 — roughly 256 positions, 64 per behaviour group, not the spent
Phase 11 sealed test bank, preferring positions with meaningful unresolved
opponent information, balanced colours where practical.

Fresh by construction, accepted by import
-----------------------------------------
Phase 12 owns no game loop, no opponent and no setup generator. Every
diagnostic game is played by the accepted `match_runner.play_match` under
the accepted `EVALUATION_RULES`, with the accepted Phase 9 greedy seat as
observer and the accepted Phase 11 stratum opponent wrapped in the accepted
`FrozenSeedPolicy`. The setups come from the two accepted sources through
the Phase 11B `Phase11BSetupSources` wrapper.

The only new thing is the identity. Every stream here is derived under the
Phase 12 personalization (``strat-p12``) from a Phase 12 master seed, so a
diagnostic game cannot coincide with a Phase 11 bank game, a Phase 11B
corpus game or a Phase 11 soak game even if two ids happened to spell the
same text. Nothing opens the spent `phase11_test_bank_v1`.

The setup-library split
-----------------------
`validation`. The accepted library's `test` split is the pool the spent
Phase 11 sealed test bank drew from and stays closed; its `train` split is
the pool Agent 1C's belief training corpus drew from. `validation` is the
one split that is neither, which is what a fresh decision diagnostic wants.
The Phase 11B *dev* split also drew from `validation` — different seeds,
different draws, different games, but the same underlying arrangement pool
that Agent 1C's candidate selection saw. That residual is named in the
report rather than hidden; it is a mild optimistic bias for `agent1c` and
this engineering diagnostic accepts it.

An eligible position
--------------------
The observer is to act, at least `MIN_UNRESOLVED` opponent pieces are still
unresolved, and the game is at least `MIN_PLY` plies old. The ply floor is
what makes the information "meaningful": at ply 0 every provider is looking
at the same untouched inventory and there is nothing for a belief model to
be better *at*. The unresolved floor keeps positions where the hidden state
still matters to a decision.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from ...belief.phase11b.contract import (
    CORPUS_COLORS,
    CORPUS_SOURCES,
    CORPUS_STRATA,
    STRATUM_POLICY_IDS,
)
from ...belief.phase11b.corpus import CorpusObserverPolicy, Phase11BSetupSources
from ...engine.constants import BLUE, RED
from ...engine.legal_moves import legal_actions
from ...engine.observation import build_observation
from ...engine.snapshot import clone_state
from ...engine.state import create_game
from ...engine.transition import apply_action
from ...evaluation.match_runner import ON_POLICY_ERROR_RAISE, play_match
from ...evaluation.match_spec import EVALUATION_RULES, MatchSpec
from ...evaluation.neural_worker import (
    DECISION_MODE_GREEDY,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from ...evaluation.phase10_validation import FrozenSeedPolicy
from ...evaluation.phase11_public_state import (
    build_public_state_document,
    document_summary,
)
from ...evaluation.policy import PolicyRef, build_public_view
from ...evaluation.registry import build_policy
from ...evaluation.setup_bank import SetupBank, SetupPair
from .contract import Phase12SearchError, derive_phase12_seed

#: The identity of this position set. Any change to the cells, the
#: eligibility rule, the selection rule or the seed derivation is a new
#: version, never a silent edit.
DIAGNOSTIC_VERSION = "phase12_diagnostic_positions_v1"

#: The Phase 12 diagnostic master seed, folded into every stream below.
DIAGNOSTIC_MASTER_SEED = 2026082002

#: The accepted setup-library split these games draw from. See the module
#: docstring: neither the spent test pool nor Agent 1C's training pool.
DIAGNOSTIC_LIBRARY_SPLIT = "validation"

#: Seed domains. Distinct strings, so the observer's setup draw, the
#: opponent's setup draw, the match randomness and the per-position search
#: seed are four independent streams.
DOMAIN_OBSERVER_SETUP = "diag_observer_setup"
DOMAIN_OPPONENT_SETUP = "diag_opponent_setup"
DOMAIN_MATCH = "diag_match"
DOMAIN_SEARCH = "diag_search"

#: The four behaviour groups of the instruction — Phase9-like, Strategic,
#: Tactical, Scout-rush — imported rather than re-spelled, so a diagnostic
#: group is literally an accepted Phase 11 stratum.
DIAGNOSTIC_STRATA = tuple(CORPUS_STRATA)
DIAGNOSTIC_SOURCES = tuple(CORPUS_SOURCES)
DIAGNOSTIC_COLORS = tuple(CORPUS_COLORS)

#: (stratum x setup source x observer colour), cell-major. Balance over all
#: three is a property of the id space, not of a post-hoc filter.
DIAGNOSTIC_CELLS = tuple(
    (stratum, source, color)
    for stratum in DIAGNOSTIC_STRATA
    for source in DIAGNOSTIC_SOURCES
    for color in DIAGNOSTIC_COLORS
)

#: Eligibility floors. See the module docstring.
MIN_PLY = 12
MIN_UNRESOLVED = 4

#: The instructed shape: 256 positions, 64 per behaviour group.
POSITIONS_PER_CELL = 16
POSITIONS_PER_GAME = 2
#: How many game ordinals one cell may walk before giving up. Games can end
#: too early to contribute (a three-ply loss yields nothing), so the walk
#: continues rather than leaving a cell short.
MAX_ORDINALS_PER_CELL = 64

#: Identities carried by every diagnostic game.
DIAGNOSTIC_RUN_VERSION = "phase12_diagnostic_generation_v1"
OBSERVER_POLICY_ID = "phase12_diagnostic_observer_v1"
PHASE9_OPPONENT_POLICY_ID = "phase12_diagnostic_phase9_opponent_v1"

MAX_GAME_ORDINAL_FORMAT = 9999

_PLAYER_OF = {"red": RED, "blue": BLUE}

_GAME_ID_PATTERN = re.compile(
    rf"^phase12_diag_v1\|ms=(?P<master>[0-9]+)"
    rf"\|st=(?P<stratum>{'|'.join(DIAGNOSTIC_STRATA)})"
    rf"\|src=(?P<source>{'|'.join(DIAGNOSTIC_SOURCES)})"
    rf"\|obs=(?P<color>{'|'.join(DIAGNOSTIC_COLORS)})"
    rf"\|g=(?P<ordinal>[0-9]{{4}})$"
)


class Phase12PositionError(Phase12SearchError):
    """A diagnostic game could not be played, selected from, or replayed."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def diagnostic_game_id(
    stratum: str, source: str, observer_color: str, ordinal: int
) -> str:
    """The stable identifier of one diagnostic game.

    ```text
    phase12_diag_v1|ms=2026082002|st=tactical_rule|src=p10d|obs=red|g=0007
    ```
    """
    if stratum not in DIAGNOSTIC_STRATA:
        raise Phase12PositionError(f"unknown stratum {stratum!r}")
    if source not in DIAGNOSTIC_SOURCES:
        raise Phase12PositionError(f"unknown setup source {source!r}")
    if observer_color not in DIAGNOSTIC_COLORS:
        raise Phase12PositionError(f"unknown observer colour {observer_color!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase12PositionError(f"ordinal must be an int, got {ordinal!r}")
    if not 0 <= ordinal <= MAX_GAME_ORDINAL_FORMAT:
        raise Phase12PositionError(f"ordinal {ordinal} outside 0..{MAX_GAME_ORDINAL_FORMAT}")
    game_id = (
        f"phase12_diag_v1|ms={DIAGNOSTIC_MASTER_SEED}|st={stratum}|src={source}"
        f"|obs={observer_color}|g={ordinal:04d}"
    )
    if _GAME_ID_PATTERN.match(game_id) is None:  # pragma: no cover - defensive
        raise Phase12PositionError(f"constructed a malformed game id: {game_id!r}")
    return game_id


def parse_diagnostic_game_id(game_id: str) -> dict:
    """The identity fields of a diagnostic game id, validated."""
    match = _GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase12PositionError(f"malformed Phase 12 diagnostic game id: {game_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != DIAGNOSTIC_MASTER_SEED:
        raise Phase12PositionError(
            f"game id names master seed {fields['master']}, expected "
            f"{DIAGNOSTIC_MASTER_SEED}"
        )
    return {
        "master_seed": int(fields["master"]),
        "stratum": fields["stratum"],
        "setup_source": fields["source"],
        "observer_color": fields["color"],
        "ordinal": int(fields["ordinal"]),
    }


def diagnostic_seed(domain: str, game_id: str, ordinal: int = 0) -> int:
    """A 63-bit Phase 12 diagnostic stream value.

    `derive_phase12_seed` is the Phase 12 personalization Agent 1 froze; the
    right shift matches the accepted phases' convention of handing policies
    and samplers a non-negative 63-bit seed.
    """
    return derive_phase12_seed(domain, game_id, int(ordinal)) >> 1


def position_id(game_id: str, ply: int) -> str:
    return f"{game_id}|ply={int(ply):04d}"


def search_seed_for(position_identifier: str) -> int:
    """The one search seed every arm uses at this position.

    Derived from the position alone, so the seed policy is identical across
    belief providers by construction rather than by discipline.
    """
    return diagnostic_seed(DOMAIN_SEARCH, position_identifier)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticGamePlan:
    """Everything one diagnostic game needs, resolved from its identity."""

    game_id: str
    stratum: str
    setup_source: str
    observer_color: str
    opponent_color: str
    ordinal: int
    cell_index: int
    match_seed: int
    red_setup: tuple
    blue_setup: tuple

    @property
    def observer(self) -> int:
        return _PLAYER_OF[self.observer_color]

    def describe(self) -> dict:
        return {
            "game_id": self.game_id,
            "stratum": self.stratum,
            "setup_source": self.setup_source,
            "observer_color": self.observer_color,
            "ordinal": self.ordinal,
            "match_seed": self.match_seed,
        }


def diagnostic_plan(
    stratum: str,
    source: str,
    observer_color: str,
    ordinal: int,
    sources: Phase11BSetupSources,
) -> DiagnosticGamePlan:
    """Resolve one game's setups and seeds from its identity alone."""
    game_id = diagnostic_game_id(stratum, source, observer_color, ordinal)
    opponent_color = "blue" if observer_color == "red" else "red"
    observer_setup = sources.draw(
        "p10d",
        DIAGNOSTIC_LIBRARY_SPLIT,
        observer_color,
        diagnostic_seed(DOMAIN_OBSERVER_SETUP, game_id),
    )
    opponent_setup = sources.draw(
        source,
        DIAGNOSTIC_LIBRARY_SPLIT,
        opponent_color,
        diagnostic_seed(DOMAIN_OPPONENT_SETUP, game_id),
    )
    red, blue = (
        (observer_setup, opponent_setup)
        if observer_color == "red"
        else (opponent_setup, observer_setup)
    )
    return DiagnosticGamePlan(
        game_id=game_id,
        stratum=stratum,
        setup_source=source,
        observer_color=observer_color,
        opponent_color=opponent_color,
        ordinal=ordinal,
        cell_index=DIAGNOSTIC_CELLS.index((stratum, source, observer_color)),
        match_seed=diagnostic_seed(DOMAIN_MATCH, game_id),
        red_setup=tuple(red),
        blue_setup=tuple(blue),
    )


# ---------------------------------------------------------------------------
# Playing one game
# ---------------------------------------------------------------------------


def observer_ref() -> PolicyRef:
    return PolicyRef(
        policy_id=OBSERVER_POLICY_ID, policy_version=DIAGNOSTIC_RUN_VERSION
    )


class DiagnosticObserverPolicy(CorpusObserverPolicy):
    """The accepted Phase 9 greedy seat, logging public decision facts.

    The Phase 11B corpus observer, reused unchanged except for one extra
    public field: how many of the unresolved opponent pieces have publicly
    moved. That is the observer's own measure of how much evidence the
    position carries, and it is read off the public view, so this policy is
    no closer to a hidden rank than its parent.
    """

    description = (
        "Phase 12 diagnostic observer: the accepted Phase 9 greedy decision, "
        "logging only public decision facts."
    )

    def decide(self, request):
        result = super().decide(request)
        view = request.require_public_view()
        moved = sum(
            1
            for piece_id in view.unresolved_opponent_piece_ids
            if view.piece(piece_id).has_moved
        )
        self.decisions[-1]["moved_hidden"] = int(moved)
        self.decisions[-1]["legal_actions"] = int(
            np.count_nonzero(request.require_legal_action_mask())
        )
        return result


def build_spec(plan: DiagnosticGamePlan, opponent: PolicyRef) -> MatchSpec:
    return MatchSpec(
        candidate=observer_ref(),
        opponent=opponent,
        setup_pair_id=plan.cell_index,
        candidate_color=plan.observer,
        replicate=plan.ordinal,
        root_seed=plan.match_seed,
        suite_version=DIAGNOSTIC_RUN_VERSION,
        setup_bank_version=(
            f"{DIAGNOSTIC_RUN_VERSION}|st={plan.stratum}|src={plan.setup_source}"
        ),
        rules=EVALUATION_RULES,
    )


def single_game_bank(spec: MatchSpec, plan: DiagnosticGamePlan) -> SetupBank:
    pair = SetupPair(
        setup_pair_id=spec.setup_pair_id,
        red_setup=plan.red_setup,
        blue_setup=plan.blue_setup,
        generation_seed=spec.root_seed,
        bank_version=spec.setup_bank_version,
        generation_family=DIAGNOSTIC_RUN_VERSION,
    )
    return SetupBank(
        bank_version=spec.setup_bank_version,
        root_seed=spec.root_seed,
        generation_family=DIAGNOSTIC_RUN_VERSION,
        pairs=(pair,),
    )


def opponent_seat(plan: DiagnosticGamePlan, owners: dict):
    """`(ref, policy)` for the opponent behaviour group of this game."""
    policy_id = STRATUM_POLICY_IDS[plan.stratum]
    if policy_id is None:
        ref = PolicyRef(
            policy_id=PHASE9_OPPONENT_POLICY_ID, policy_version=DIAGNOSTIC_RUN_VERSION
        )
        return ref, RemoteNeuralPolicy(
            ref,
            LocalInferenceChannel(owners["phase9"]),
            decision_mode=DECISION_MODE_GREEDY,
        )
    policy = build_policy(policy_id)
    return policy.ref, policy


def play_diagnostic_game(plan: DiagnosticGamePlan, owners: dict):
    """Play one diagnostic game. Returns `(result, decisions)` — public only."""
    opponent_reference, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_reference)
    observer = DiagnosticObserverPolicy(observer_ref(), owners["phase9"])
    policies = {observer_ref().token: observer}
    if opponent_reference.token != observer_ref().token:
        policies[opponent_reference.token] = FrozenSeedPolicy(
            opponent_policy, plan.match_seed
        )
    result = play_match(
        spec,
        bank=single_game_bank(spec, plan),
        policies=policies,
        record_actions=True,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    if result.errored:  # pragma: no cover - raises above under RAISE
        raise Phase12PositionError(f"{plan.game_id} errored: {result.policy_error}")
    return result, observer.decisions


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def eligible_decisions(
    decisions: "list[dict]",
    *,
    min_ply: int = MIN_PLY,
    min_unresolved: int = MIN_UNRESOLVED,
) -> "list[dict]":
    """The observer decisions that carry meaningful unresolved information."""
    return [
        row
        for row in decisions
        if int(row["ply"]) >= int(min_ply)
        and int(row["unresolved"]) >= int(min_unresolved)
    ]


def spread(values: "list", count: int) -> "list":
    """At most `count` elements at the quantile midpoints of `values`.

    Deliberately neither of the two rules the accepted phases use. The
    Phase 11 slice always starts at the first element and never reaches the
    last; the Phase 11B slice includes both endpoints. Either one would
    distort *this* set, because a game's eligible list is ordered by ply:
    the first element is always the eligibility floor and the last is always
    the game's final decision, so an endpoint rule with two picks per game
    would put half the diagnostic set at ply 12 and the other half at the
    end of a game. Midpoints — `(2k+1)/2n` through the list — spread the
    picks over the interior instead, which is where a belief actually has
    both evidence to use and a game left to decide.
    """
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    return [values[((2 * index + 1) * len(values)) // (2 * count)] for index in range(count)]


def select_positions(
    decisions: "list[dict]",
    *,
    per_game: int = POSITIONS_PER_GAME,
    min_ply: int = MIN_PLY,
    min_unresolved: int = MIN_UNRESOLVED,
) -> "list[dict]":
    """The evenly spaced eligible observer decisions of one game."""
    return spread(
        eligible_decisions(
            decisions, min_ply=min_ply, min_unresolved=min_unresolved
        ),
        int(per_game),
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def observation_digest(observation: np.ndarray) -> str:
    """The Phase 11B corpus recipe, so a replay check compares like with like."""
    return hashlib.sha256(
        np.ascontiguousarray(observation, dtype=np.float32).tobytes()
    ).hexdigest()


def replay_positions(
    game_id: str,
    red_setup,
    blue_setup,
    action_history,
    observer: int,
    plies: "list[int]",
) -> "list[dict]":
    """Rebuild the wanted plies of one game from its recorded action history.

    Returns `[{ply, state, observation, document}]` in ply order. The states
    are independent and playable; each carries true ranks, which is exactly
    what the oracle arm and the belief-quality diagnostic need and what
    every other arm is structurally unable to read.
    """
    remaining = sorted({int(ply) for ply in plies})
    if not remaining:
        return []
    state = create_game(
        tuple(red_setup), tuple(blue_setup), rules=EVALUATION_RULES, game_id=game_id
    )
    actions = [int(action) for action in action_history]
    index = 0
    rebuilt: list[dict] = []
    while remaining:
        if int(state.total_moves) == remaining[0]:
            if state.terminal:
                raise Phase12PositionError(
                    f"{game_id} ply {remaining[0]} is terminal; not a decision"
                )
            if state.acting_player != observer:
                raise Phase12PositionError(
                    f"{game_id} ply {remaining[0]} is not an observer decision"
                )
            # Cloned, so the walk can keep replaying without mutating a
            # position that has already been handed out.
            captured = clone_state(state)
            observation = build_observation(captured, observer)
            document = build_public_state_document(
                build_public_view(captured, observer), observation
            )
            rebuilt.append(
                {
                    "ply": remaining.pop(0),
                    "state": captured,
                    "observation": observation,
                    "document": document,
                }
            )
            continue
        if state.terminal or index >= len(actions):
            raise Phase12PositionError(
                f"{game_id} action history never reached plies {remaining}"
            )
        apply_action(state, actions[index])
        index += 1
    return rebuilt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_positions(
    owners: dict,
    *,
    positions_per_cell: int = POSITIONS_PER_CELL,
    per_game: int = POSITIONS_PER_GAME,
    min_ply: int = MIN_PLY,
    min_unresolved: int = MIN_UNRESOLVED,
    max_ordinals_per_cell: int = MAX_ORDINALS_PER_CELL,
    sources: "Phase11BSetupSources | None" = None,
    progress=None,
) -> dict:
    """Play the diagnostic games cell by cell and select their positions.

    Each cell walks game ordinals from zero and takes up to `per_game`
    eligible positions per game until it has `positions_per_cell` or the
    ordinal cap is reached. Games that end too early to carry an eligible
    decision are skipped rather than shrinking the cell, so the cell balance
    survives the tail of the game-length distribution.
    """
    if sources is None:
        sources = Phase11BSetupSources()
    games: list[dict] = []
    positions: list[dict] = []
    shortfalls: list[dict] = []

    for stratum, source, color in DIAGNOSTIC_CELLS:
        taken = 0
        ordinal = 0
        while taken < positions_per_cell and ordinal < max_ordinals_per_cell:
            plan = diagnostic_plan(stratum, source, color, ordinal, sources)
            ordinal += 1
            result, decisions = play_diagnostic_game(plan, owners)
            chosen = select_positions(
                decisions,
                per_game=min(per_game, positions_per_cell - taken),
                min_ply=min_ply,
                min_unresolved=min_unresolved,
            )
            history = [int(action) for action in (result.action_history or ())]
            games.append(
                {
                    "game_id": plan.game_id,
                    "stratum": plan.stratum,
                    "setup_source": plan.setup_source,
                    "observer_color": plan.observer_color,
                    "observer_player": int(plan.observer),
                    "ordinal": plan.ordinal,
                    "match_seed": plan.match_seed,
                    "opponent_policy_id": STRATUM_POLICY_IDS[plan.stratum]
                    or PHASE9_OPPONENT_POLICY_ID,
                    "red_setup": [int(value) for value in plan.red_setup],
                    "blue_setup": [int(value) for value in plan.blue_setup],
                    "plies": len(history),
                    "observer_decisions": len(decisions),
                    "eligible_decisions": len(
                        eligible_decisions(
                            decisions, min_ply=min_ply, min_unresolved=min_unresolved
                        )
                    ),
                    "contributed": len(chosen),
                    "action_history": history,
                }
            )
            for row in chosen:
                positions.append(
                    {
                        "position_id": position_id(plan.game_id, row["ply"]),
                        "game_id": plan.game_id,
                        "ply": int(row["ply"]),
                        "stratum": plan.stratum,
                        "setup_source": plan.setup_source,
                        "observer_color": plan.observer_color,
                        "observer_player": int(plan.observer),
                        "unresolved": int(row["unresolved"]),
                        "moved_hidden": int(row.get("moved_hidden", -1)),
                        "legal_actions": int(row.get("legal_actions", -1)),
                        "observation_sha256": row["observation_sha256"],
                    }
                )
            taken += len(chosen)
            if progress is not None:
                progress(stratum, source, color, taken, len(games), len(positions))
        if taken < positions_per_cell:
            shortfalls.append(
                {
                    "cell": [stratum, source, color],
                    "wanted": positions_per_cell,
                    "got": taken,
                    "ordinals_walked": ordinal,
                }
            )
    return {"games": games, "positions": positions, "shortfalls": shortfalls}


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def manifest_digest(manifest: dict) -> str:
    """sha256 of the manifest's canonical JSON, excluding the digest field."""
    import json

    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_manifest(generated: dict, *, generated_utc: str, **extra) -> dict:
    """The self-contained, replayable description of the position set."""
    positions = generated["positions"]
    by_stratum: dict = {}
    by_color: dict = {}
    for row in positions:
        by_stratum[row["stratum"]] = by_stratum.get(row["stratum"], 0) + 1
        by_color[row["observer_color"]] = by_color.get(row["observer_color"], 0) + 1
    manifest = {
        "artifact": DIAGNOSTIC_VERSION,
        "phase": "phase12",
        "generated_utc": generated_utc,
        "generated_by": "Phase 12 Agent 2 (belief-to-decision diagnostic)",
        "master_seed": DIAGNOSTIC_MASTER_SEED,
        "run_version": DIAGNOSTIC_RUN_VERSION,
        "setup_library_split": DIAGNOSTIC_LIBRARY_SPLIT,
        "rules_version": EVALUATION_RULES.rules_version,
        "phase11_test_bank_used": False,
        "phase11b_corpus_reused": False,
        "eligibility": {
            "observer_to_act": True,
            "min_ply": MIN_PLY,
            "min_unresolved_opponent_pieces": MIN_UNRESOLVED,
            "selection": "eligible decisions at the quantile midpoints of each game",
            "positions_per_game": POSITIONS_PER_GAME,
        },
        "cells": [list(cell) for cell in DIAGNOSTIC_CELLS],
        "counts": {
            "games_played": len(generated["games"]),
            "games_contributing": sum(
                1 for game in generated["games"] if game["contributed"]
            ),
            "positions": len(positions),
            "positions_by_behavior_group": by_stratum,
            "positions_by_observer_color": by_color,
            "shortfalls": generated["shortfalls"],
        },
        "positions": positions,
        "games": generated["games"],
    }
    manifest.update(extra)
    manifest["manifest_digest"] = manifest_digest(manifest)
    return manifest


def load_manifest(path) -> dict:
    import json
    from pathlib import Path

    manifest = json.loads(Path(path).read_text())
    if manifest.get("artifact") != DIAGNOSTIC_VERSION:
        raise Phase12PositionError(
            f"{path} is not a {DIAGNOSTIC_VERSION} manifest"
        )
    recorded = manifest.get("manifest_digest")
    recomputed = manifest_digest(manifest)
    if recorded != recomputed:
        raise Phase12PositionError(
            f"{path} digests to {recomputed}, not the recorded {recorded}"
        )
    return manifest


def materialize_manifest(manifest: dict, *, verify: bool = True) -> "list[dict]":
    """Rebuild every manifest position as a playable state, in manifest order.

    With `verify` on, each rebuilt observation is required to digest to the
    value the public pass recorded while the game was being played — a
    bit-for-bit alignment proof that the replayed position is the position
    that was selected, not merely a position at the same ply.
    """
    games = {game["game_id"]: game for game in manifest["games"]}
    wanted: dict = {}
    for row in manifest["positions"]:
        wanted.setdefault(row["game_id"], []).append(int(row["ply"]))

    rebuilt: dict = {}
    for game_id, plies in wanted.items():
        game = games.get(game_id)
        if game is None:
            raise Phase12PositionError(f"manifest position names unknown game {game_id}")
        for entry in replay_positions(
            game_id,
            game["red_setup"],
            game["blue_setup"],
            game["action_history"],
            int(game["observer_player"]),
            plies,
        ):
            rebuilt[position_id(game_id, entry["ply"])] = entry

    materialized: list[dict] = []
    for row in manifest["positions"]:
        entry = rebuilt[row["position_id"]]
        if verify:
            digest = observation_digest(entry["observation"])
            if digest != row["observation_sha256"]:
                raise Phase12PositionError(
                    f"{row['position_id']} replayed to observation {digest[:16]}..., "
                    f"recorded {row['observation_sha256'][:16]}..."
                )
        record = dict(row)
        record.update(
            {
                "state": entry["state"],
                "observation": entry["observation"],
                "document": entry["document"],
                "document_summary": document_summary(entry["document"]),
                "search_seed": search_seed_for(row["position_id"]),
                "legal_action_count": len(legal_actions(entry["state"])),
            }
        )
        materialized.append(record)
    return materialized


__all__ = [
    "DIAGNOSTIC_CELLS",
    "DIAGNOSTIC_COLORS",
    "DIAGNOSTIC_LIBRARY_SPLIT",
    "DIAGNOSTIC_MASTER_SEED",
    "DIAGNOSTIC_RUN_VERSION",
    "DIAGNOSTIC_SOURCES",
    "DIAGNOSTIC_STRATA",
    "DIAGNOSTIC_VERSION",
    "DiagnosticGamePlan",
    "DiagnosticObserverPolicy",
    "MIN_PLY",
    "MIN_UNRESOLVED",
    "POSITIONS_PER_CELL",
    "POSITIONS_PER_GAME",
    "Phase12PositionError",
    "build_manifest",
    "diagnostic_game_id",
    "diagnostic_plan",
    "diagnostic_seed",
    "eligible_decisions",
    "generate_positions",
    "load_manifest",
    "manifest_digest",
    "materialize_manifest",
    "observation_digest",
    "parse_diagnostic_game_id",
    "play_diagnostic_game",
    "position_id",
    "replay_positions",
    "search_seed_for",
    "select_positions",
    "spread",
]
