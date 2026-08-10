"""Independently stepped reference games and the batch/reference comparator.

`tests/training/test_batch_simulation.py` runs this at a small deterministic
scale and `scripts/run_phase3_agent01.py` runs it at full acceptance scale, so
the fast suite and the acceptance gate cannot drift apart.

The reference side never touches `stratego.training`: it builds a `GameState`
with `create_game` and advances it with `apply_action`, exactly as the Phase 2
suite does. The only thing the two sides share is the seed derivation, which is
the definition of "deterministically seed each slot" rather than a shortcut.

Comparison points, evaluated for **every** state a tested action produces:

- full state fingerprint (`include_history=False` per state, and the complete
  `include_history=True` fingerprint once per finished game, which covers the
  whole event log and action history without re-hashing it every ply);
- acting player, both from the state and from the batch API's dense vector;
- legal-action list;
- dense 10,000-entry legality mask, including the stacked batch read;
- both players' observations;
- both players' public board views and the observer-filtered public event
  stream of every step;
- both players' privileged belief targets;
- terminal reason, result and winner.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from stratego.engine.constants import PLAYERS, RulesConfig, TRAINING_RULES
from stratego.engine.events import filter_events_for_observer, public_board_view
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import belief_target, build_observation
from stratego.engine.random_play import make_random_setups
from stratego.engine.state import GameState, create_game, state_fingerprint
from stratego.engine.transition import apply_action
from stratego.training.batch_simulation import (
    BatchSimulator,
    derive_slot_seed,
    slot_game_id,
)


def reference_game(
    root_seed: int,
    environment_id: int,
    generation: int,
    rules: RulesConfig = TRAINING_RULES,
) -> GameState:
    """The game a batch slot must contain, built straight from the engine."""
    seed = derive_slot_seed(root_seed, environment_id, generation)
    red_setup, blue_setup = make_random_setups(seed)
    return create_game(
        red_setup,
        blue_setup,
        rules=rules,
        game_id=slot_game_id(root_seed, environment_id, generation),
    )


def choose_action(
    root_seed: int, environment_id: int, generation: int, ply: int, legal: list[int]
) -> int:
    """Deterministic action choice that depends on nothing but its arguments.

    Both sides select from lists that have already been asserted equal, so the
    choice cannot mask a legality difference.
    """
    payload = f"{root_seed}:{environment_id}:{generation}:{ply}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"stratego-pick").digest()
    return legal[int.from_bytes(digest, "big") % len(legal)]


def compare_state(
    simulator: BatchSimulator,
    slot: int,
    reference: GameState,
    *,
    stacked_observation: np.ndarray | None = None,
    stacked_mask: np.ndarray | None = None,
    stacked_acting_player: int | None = None,
    include_public_views: bool = True,
) -> list[str]:
    """Compare one batch slot against its reference game. Empty list means equal."""
    problems: list[str] = []
    batch_state = simulator.game_state(slot)

    batch_core = state_fingerprint(batch_state, include_history=False)
    reference_core = state_fingerprint(reference, include_history=False)
    if batch_core != reference_core:
        problems.append("state fingerprint differs")

    if batch_state.acting_player != reference.acting_player:
        problems.append(
            f"acting player differs: batch {batch_state.acting_player} "
            f"vs reference {reference.acting_player}"
        )
    expected_acting = -1 if reference.terminal else reference.acting_player
    if simulator.acting_player(slot) != expected_acting:
        problems.append(
            f"batch acting_player() reports {simulator.acting_player(slot)}, "
            f"expected {expected_acting}"
        )
    if stacked_acting_player is not None and int(stacked_acting_player) != expected_acting:
        problems.append(
            f"stacked acting-player entry {int(stacked_acting_player)} != {expected_acting}"
        )

    reference_legal = legal_actions(reference)
    batch_legal = simulator.legal_actions(slot)
    if batch_legal != reference_legal:
        problems.append(
            f"legal-action list differs: batch {len(batch_legal)} entries "
            f"vs reference {len(reference_legal)}"
        )

    reference_mask = legal_action_mask(reference, reference_legal)
    if not np.array_equal(simulator.legal_action_mask(slot), reference_mask):
        problems.append("dense legal mask differs")
    if stacked_mask is not None and not np.array_equal(stacked_mask, reference_mask):
        problems.append("stacked dense legal mask differs")
    # The mask and the list are generated from the same source inside the
    # engine, so this catches a batch layer that reordered or filtered either.
    if np.flatnonzero(reference_mask).tolist() != reference_legal:
        problems.append("reference mask and list disagree")

    for observer in PLAYERS:
        batch_observation = simulator.observation(slot, observer)
        if not np.array_equal(batch_observation, build_observation(reference, observer)):
            problems.append(f"observation differs for observer {observer}")
    if stacked_observation is not None:
        expected = build_observation(reference, reference.acting_player)
        if not np.array_equal(stacked_observation, expected):
            problems.append("stacked acting-player observation differs")

    for observer in PLAYERS:
        if simulator.belief_targets(slot, observer) != belief_target(reference, observer):
            problems.append(f"belief target differs for observer {observer}")

    if include_public_views:
        for observer in PLAYERS:
            if simulator.public_board(slot, observer) != public_board_view(
                reference, observer
            ):
                problems.append(f"public board view differs for observer {observer}")

    outcome = simulator.outcome(slot)
    if outcome.terminal != reference.terminal:
        problems.append("terminal flag differs")
    expected_reason = reference.terminal_reason if reference.terminal else "not_terminal"
    if outcome.terminal_reason != expected_reason:
        problems.append(
            f"terminal reason differs: batch {outcome.terminal_reason} "
            f"vs reference {expected_reason}"
        )
    if outcome.winner != reference.winner or outcome.is_draw != reference.is_draw:
        problems.append("terminal result differs")
    if outcome.total_moves != reference.total_moves:
        problems.append("total move counter differs")
    if reference.terminal:
        if outcome.result_for_red != reference.result_for(PLAYERS[0]):
            problems.append("red result differs")
        if outcome.result_for_blue != reference.result_for(PLAYERS[1]):
            problems.append("blue result differs")
        # One full-history comparison per game covers the complete derived event
        # log and action history.
        if simulator.slot_fingerprint(slot)[-1] != state_fingerprint(reference):
            problems.append("full-history state fingerprint differs")
        for observer in PLAYERS:
            if simulator.public_events(slot, observer) != filter_events_for_observer(
                reference.events, observer
            ):
                problems.append(f"public event stream differs for observer {observer}")

    return problems


def compare_step_events(
    batch_events: "tuple[dict, ...]", reference_events: list[dict]
) -> list[str]:
    """Compare the derived events one action emitted, including public filtering."""
    problems: list[str] = []
    if list(batch_events) != list(reference_events):
        problems.append("step event stream differs")
        return problems
    for observer in PLAYERS:
        if filter_events_for_observer(
            list(batch_events), observer
        ) != filter_events_for_observer(reference_events, observer):
            problems.append(f"filtered step events differ for observer {observer}")
    return problems


@dataclass
class DifferentialReport:
    """Aggregate result of one differential-equivalence run."""

    comparisons: int = 0
    mismatches: int = 0
    mismatch_details: list[dict] = field(default_factory=list)
    batch_steps: int = 0
    games_completed: int = 0
    resets: int = 0
    generation_errors: int = 0
    ordinary_moves: int = 0
    scout_multisquare_moves: int = 0
    combats: int = 0
    reveals: int = 0
    behavior_counts: Counter = field(default_factory=Counter)
    terminal_reason_counts: Counter = field(default_factory=Counter)
    max_ply: int = 0

    def as_dict(self) -> dict:
        return {
            "state_action_comparisons": self.comparisons,
            "equivalence_mismatches": self.mismatches,
            "mismatch_details": self.mismatch_details,
            "batch_steps": self.batch_steps,
            "games_completed": self.games_completed,
            "resets": self.resets,
            "generation_errors": self.generation_errors,
            "ordinary_moves": self.ordinary_moves,
            "scout_multisquare_moves": self.scout_multisquare_moves,
            "combats": self.combats,
            "identity_reveals": self.reveals,
            "behavior_types_observed": dict(sorted(self.behavior_counts.items())),
            "terminal_reason_counts": dict(sorted(self.terminal_reason_counts.items())),
            "max_ply_reached": self.max_ply,
        }


def _record_events(report: DifferentialReport, events: "tuple[dict, ...]") -> None:
    for event in events:
        kind = event["event_type"]
        if kind == "move":
            if event["distance"] >= 2:
                report.scout_multisquare_moves += 1
            else:
                report.ordinary_moves += 1
        elif kind == "combat":
            report.combats += 1
        elif kind == "identity_reveal":
            report.reveals += 1
        elif kind == "behavior":
            report.behavior_counts[event["behavior_type"]] += 1


def run_differential(
    *,
    num_environments: int,
    root_seed: int,
    target_comparisons: int,
    rules: RulesConfig = TRAINING_RULES,
    include_public_views: bool = True,
    stop_on_mismatch: bool = True,
    max_mismatch_details: int = 5,
    progress=None,
) -> DifferentialReport:
    """Drive a batch and an independent reference set in lockstep.

    Every state is compared exactly once: a non-terminal state during the bulk
    read that precedes its action, and a terminal state in the pass that runs
    before its slot is reset. Finished slots are reset independently, and their
    reference games are rebuilt from the slot's new generation, so resets are
    part of the compared behaviour rather than a gap in it.
    """
    report = DifferentialReport()
    simulator = BatchSimulator(num_environments, root_seed=root_seed, rules=rules)
    references = {
        slot: reference_game(root_seed, simulator.environment_id(slot), 0, rules)
        for slot in range(num_environments)
    }

    def note(slot: int, problems: list[str], stage: str) -> bool:
        if not problems:
            return True
        report.mismatches += len(problems)
        if len(report.mismatch_details) < max_mismatch_details:
            report.mismatch_details.append(
                {
                    "stage": stage,
                    "slot": slot,
                    "environment_id": simulator.environment_id(slot),
                    "generation": simulator.generation(slot),
                    "game_id": simulator.game_id(slot),
                    "ply": simulator.game_state(slot).total_moves,
                    "action_history": list(simulator.game_state(slot).action_history),
                    "problems": problems,
                }
            )
        return False

    while report.comparisons < target_comparisons:
        active = simulator.active_slots()
        if not active:  # pragma: no cover - a reset always follows a finish
            break

        # ---- bulk read phase --------------------------------------------
        observations = simulator.observations(active)
        masks = simulator.legal_action_masks(active)
        acting = simulator.acting_players(active)

        actions: dict[int, int] = {}
        for position, slot in enumerate(active):
            reference = references[slot]
            problems = compare_state(
                simulator,
                slot,
                reference,
                stacked_observation=observations[position],
                stacked_mask=masks[position],
                stacked_acting_player=acting[position],
                include_public_views=include_public_views,
            )
            if not note(slot, problems, "pre_step") and stop_on_mismatch:
                return report
            actions[slot] = choose_action(
                root_seed,
                simulator.environment_id(slot),
                simulator.generation(slot),
                reference.total_moves,
                simulator.legal_actions(slot),
            )

        # ---- apply phase -------------------------------------------------
        result = simulator.step(actions)
        report.batch_steps += 1
        for slot in active:
            reference_events = apply_action(references[slot], actions[slot])
            problems = compare_step_events(result.events[slot], reference_events)
            if not note(slot, problems, "step_events") and stop_on_mismatch:
                return report
            _record_events(report, result.events[slot])
            report.comparisons += 1
            report.max_ply = max(report.max_ply, references[slot].total_moves)

        # ---- terminal collection ----------------------------------------
        for slot in result.newly_terminal:
            problems = compare_state(
                simulator,
                slot,
                references[slot],
                include_public_views=include_public_views,
            )
            if not note(slot, problems, "terminal") and stop_on_mismatch:
                return report
            report.games_completed += 1
            report.terminal_reason_counts[simulator.outcome(slot).terminal_reason] += 1

        # ---- independent reset ------------------------------------------
        finished = simulator.finished_slots()
        if finished:
            previous = {slot: simulator.generation(slot) for slot in finished}
            untouched = [slot for slot in range(num_environments) if slot not in set(finished)]
            # History-free fingerprints here: this check runs on every reset in
            # every batch, so hashing each neighbour's whole event log would make
            # the run cost grow with game length. `scripts/run_phase3_agent01.py`
            # runs the full-history isolation check in its dedicated reset trials.
            before = {
                slot: simulator.slot_fingerprint(slot, include_history=False)
                for slot in untouched
            }
            simulator.reset_slots(finished)
            for slot in untouched:
                if simulator.slot_fingerprint(slot, include_history=False) != before[slot]:
                    if not note(slot, ["slot changed during another slot's reset"], "reset"):
                        if stop_on_mismatch:
                            return report
            for slot in finished:
                if simulator.generation(slot) != previous[slot] + 1:
                    report.generation_errors += 1
                references[slot] = reference_game(
                    root_seed,
                    simulator.environment_id(slot),
                    simulator.generation(slot),
                    rules,
                )
                report.resets += 1

        if progress is not None:
            progress(report)

    return report
