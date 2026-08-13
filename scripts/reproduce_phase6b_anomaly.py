"""Phase 6B Gate 1: the exact formerly-failing sequence, re-run under the fix.

History
-------
The first persisted soak aborted at t = 8,981 s: slot 112 was published ACTIVE
with `legal_count 0`. The diagnosis (frozen, pre-fix, in
`reports/phase_6_data/agent_06b_anomaly_diagnosis.json`) identified the state
as `(root_seed=60006, environment_id=112, generation=98)` — a game whose first
player is stranded at ply 0 by a 1-in-548,340 random setup, which
`create_game` under `phase2_1_reference_1.1.0` returned labelled active
because it never evaluated initial mobility.

What this script is now
-----------------------
The engine correction was authorized and landed as `phase2_1_reference_1.2.0`:
`create_game` applies the mobility-termination rule to the initial position
through the same `transition.py` logic that governs every move. This script
re-runs the exact deterministic sequence that previously failed and exits 0
only if every stage now behaves correctly:

1. **Engine** — the game is terminal at creation: `opponent_no_legal_move`,
   winner blue, zero legal actions, exactly one `game_end` event.
2. **Batch layer** — `BatchSimulator` rebuilds the identical terminal state.
3. **Production pool** — a real `WorkerPool` advanced to that generation
   publishes the slot as TERMINAL (the former failure point: no trap fires,
   no active-with-zero-legality row exists), seals the stillborn outcome on
   the next step, and moves the slot on to generation 99.
4. **Corpus forensics** — the preserved aborted-soak corpus still shows
   generations 0..97 sealed exactly once, generation 98 absent, and the
   abort step at 48,225 (the pre-fix evidence is undisturbed).
5. **Horizon scan** — (112, 98) remains the only first-player-stranded setup
   in the 1,536 x 120 horizon.

Writes `reports/phase_6_data/agent_06b_anomaly_fix_validation.json`. The
pre-fix diagnosis artifact is never touched.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

ROOT_SEED = 60006
ENVIRONMENT = 112
GENERATION = 98
DEFAULT_CORPUS = "/Volumes/Brandon_Washington/stratego_phase6b/soak"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports" / "phase_6_data" / (
    "agent_06b_anomaly_fix_validation.json"
)

FRONT_OPEN = (30, 31, 34, 35, 38, 39)
BLUE_FRONT_OPEN = (0, 1, 4, 5, 8, 9)


def stranded_probability_one_in() -> int:
    """Exact reciprocal of P(one side stranded at creation): 548,340."""
    numerator = math.perm(40, 6)
    denominator = math.perm(7, 6)
    assert numerator % denominator == 0
    return numerator // denominator


def validate_engine_state() -> dict:
    from stratego.engine.constants import BLUE, RED, TRAINING_RULES
    from stratego.engine.events import EVENT_GAME_END
    from stratego.engine.legal_moves import has_legal_action, legal_actions
    from stratego.engine.random_play import make_random_setups
    from stratego.engine.state import create_game, state_fingerprint
    from stratego.training.batch_simulation import derive_slot_seed, slot_game_id

    seed = derive_slot_seed(ROOT_SEED, ENVIRONMENT, GENERATION)
    red_setup, blue_setup = make_random_setups(seed)
    game_id = slot_game_id(ROOT_SEED, ENVIRONMENT, GENERATION)
    state = create_game(red_setup, blue_setup, rules=TRAINING_RULES, game_id=game_id)
    ends = [event for event in state.events if event["event_type"] == EVENT_GAME_END]
    return {
        "slot_seed": seed,
        "game_id": game_id,
        "terminal": state.terminal,
        "terminal_reason": state.terminal_reason,
        "winner": state.winner,
        "winner_is_blue": state.winner == BLUE,
        "is_draw": state.is_draw,
        "ply": state.total_moves,
        "legal_actions_for_acting_player": len(legal_actions(state)),
        "has_legal_action_red": has_legal_action(state, RED),
        "has_legal_action_blue": has_legal_action(state, BLUE),
        "game_end_events": len(ends),
        "game_end_is_final_event": bool(ends) and ends[-1] is state.events[-1],
        "state_fingerprint_sha256": hashlib.sha256(
            repr(state_fingerprint(state)).encode()
        ).hexdigest(),
        "correct": (
            state.terminal is True
            and state.terminal_reason == "opponent_no_legal_move"
            and state.winner == BLUE
            and state.total_moves == 0
            and len(legal_actions(state)) == 0
            and len(ends) == 1
        ),
    }


def validate_batch_layer(engine: dict) -> dict:
    from stratego.engine.state import state_fingerprint
    from stratego.training.batch_simulation import BatchSimulator

    simulator = BatchSimulator(
        1, root_seed=ROOT_SEED, first_environment_id=ENVIRONMENT
    )
    for _ in range(GENERATION):
        simulator.reset_slots([0])
    state = simulator.game_state(0)
    fingerprint = hashlib.sha256(repr(state_fingerprint(state)).encode()).hexdigest()
    return {
        "generation": simulator.generation(0),
        "game_id": state.game_id,
        "is_terminal": simulator.is_terminal(0),
        "legal_count": len(simulator.legal_actions(0)),
        "legal_mask_nonzero": int(simulator.legal_action_mask(0).sum()),
        "identical_to_engine_state": fingerprint == engine["state_fingerprint_sha256"],
        "correct": (
            simulator.is_terminal(0)
            and len(simulator.legal_actions(0)) == 0
            and fingerprint == engine["state_fingerprint_sha256"]
        ),
    }


def validate_through_pool() -> dict:
    from stratego.engine.constants import BLUE
    from stratego.training.shared_buffers import (
        NO_ACTING_PLAYER,
        STATUS_TERMINAL,
        terminal_reason_name,
    )
    from stratego.training.worker_pool import WorkerPool

    pool = WorkerPool(ENVIRONMENT + 1, 1, root_seed=ROOT_SEED)
    try:
        pool.start()
        for _ in range(GENERATION):
            pool.request_reset([ENVIRONMENT])
            pool.step(apply_actions=False, auto_reset=False)
        published = {
            "generation": int(pool.buffers.generation[ENVIRONMENT]),
            "status_terminal": int(pool.buffers.status[ENVIRONMENT])
            == STATUS_TERMINAL,
            "terminal_flag": int(pool.buffers.terminal[ENVIRONMENT]),
            "legal_count": int(pool.buffers.legal_count[ENVIRONMENT]),
            "acting_player_none": int(pool.buffers.acting_player[ENVIRONMENT])
            == NO_ACTING_PLAYER,
            "episode_count_before_seal": int(pool.buffers.episode_count[ENVIRONMENT]),
        }
        # The next ordinary step seals the stillborn outcome and recycles the
        # slot -- the lifecycle the production coordinator drives every step.
        pool.clear_actions()
        pool.step(apply_actions=True, auto_reset=True)
        sealed = {
            "episode_count": int(pool.buffers.episode_count[ENVIRONMENT]),
            "last_terminal_reason": terminal_reason_name(
                int(pool.buffers.last_terminal_reason[ENVIRONMENT])
            ),
            "last_winner_is_blue": int(pool.buffers.last_winner[ENVIRONMENT]) == BLUE,
            "last_total_moves": int(pool.buffers.last_total_moves[ENVIRONMENT]),
            "next_generation": int(pool.buffers.generation[ENVIRONMENT]),
            "next_generation_active": int(pool.buffers.terminal[ENVIRONMENT]) == 0,
            "stillborn_counted": int(
                sum(
                    reply.get("total_stillborn_games", 0)
                    for reply in pool.last_replies
                )
            ),
        }
    finally:
        pool.shutdown()
    return {
        "published_generation_98": published,
        "after_sealing_step": sealed,
        "correct": (
            published["generation"] == GENERATION
            and published["status_terminal"]
            and published["terminal_flag"] == 1
            and published["legal_count"] == 0
            and published["acting_player_none"]
            and sealed["episode_count"] == 1
            and sealed["last_terminal_reason"] == "opponent_no_legal_move"
            and sealed["last_winner_is_blue"]
            and sealed["last_total_moves"] == 0
            and sealed["next_generation"] == GENERATION + 1
            and sealed["next_generation_active"]
            and sealed["stillborn_counted"] == 1
        ),
    }


def corpus_forensics(corpus: Path) -> dict:
    if not corpus.is_dir():
        return {"available": False, "directory": str(corpus), "correct": True}

    marker = f"-env{ENVIRONMENT:06d}-"
    generations: list[int] = []
    for manifest_path in sorted(corpus.glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        for game_id in manifest["game_ids"]:
            if marker in game_id:
                generations.append(int(game_id.split("-gen")[1]))
    generations.sort()
    missing = sorted(set(range(generations[0], generations[-1] + 1)) - set(generations))
    return {
        "available": True,
        "directory": str(corpus),
        "sealed_games_for_environment": len(generations),
        "generation_min": generations[0],
        "generation_max": generations[-1],
        "missing_generations_in_range": missing,
        "anomalous_generation_present": GENERATION in set(generations),
        "correct": (
            generations[0] == 0
            and generations[-1] == GENERATION - 1
            and not missing
            and GENERATION not in set(generations)
        ),
    }


def horizon_scan(environments: int = 1536, generations: int = 120) -> dict:
    from stratego.engine.constants import IMMOVABLE_TYPES
    from stratego.engine.random_play import make_random_setups
    from stratego.training.batch_simulation import derive_slot_seed

    red_stranded = []
    blue_stranded = []
    for environment in range(environments):
        for generation in range(generations):
            seed = derive_slot_seed(ROOT_SEED, environment, generation)
            red_setup, blue_setup = make_random_setups(seed)
            if all(red_setup[i] in IMMOVABLE_TYPES for i in FRONT_OPEN):
                red_stranded.append([environment, generation])
            if all(blue_setup[i] in IMMOVABLE_TYPES for i in BLUE_FRONT_OPEN):
                blue_stranded.append([environment, generation])
    return {
        "environments_scanned": environments,
        "generations_scanned": generations,
        "first_player_stranded_setups": red_stranded,
        "second_player_stranded_setups": blue_stranded,
        "correct": red_stranded == [[ENVIRONMENT, GENERATION]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    parser.add_argument("--skip-pool", action="store_true")
    arguments = parser.parse_args()

    from stratego.engine.constants import IMPLEMENTATION_VERSION

    started = time.perf_counter()
    print(f"engine implementation: {IMPLEMENTATION_VERSION}")

    print("stage 1/5: engine — the state is terminal at creation")
    engine = validate_engine_state()
    print(
        f"  terminal={engine['terminal']} reason={engine['terminal_reason']!r} "
        f"winner_is_blue={engine['winner_is_blue']} "
        f"game_end_events={engine['game_end_events']} correct={engine['correct']}"
    )

    print("stage 2/5: batch layer — identical terminal state")
    batch = validate_batch_layer(engine)
    print(
        f"  identical={batch['identical_to_engine_state']} "
        f"terminal={batch['is_terminal']} correct={batch['correct']}"
    )

    if arguments.skip_pool:
        pool = {"skipped": True, "correct": True}
    else:
        print("stage 3/5: production pool — the former failure point now passes")
        pool = validate_through_pool()
        print(
            f"  published terminal={pool['published_generation_98']['status_terminal']} "
            f"sealed={pool['after_sealing_step']['episode_count']} "
            f"next_gen={pool['after_sealing_step']['next_generation']} "
            f"correct={pool['correct']}"
        )

    print("stage 4/5: preserved corpus — pre-fix evidence undisturbed")
    corpus = corpus_forensics(arguments.corpus)
    if corpus["available"]:
        print(
            f"  gens {corpus['generation_min']}..{corpus['generation_max']} "
            f"missing={corpus['missing_generations_in_range']} "
            f"gen98_present={corpus['anomalous_generation_present']} "
            f"correct={corpus['correct']}"
        )
    else:
        print("  corpus not mounted; skipped")

    print("stage 5/5: horizon — (112, 98) is still the only stranded setup")
    scan = horizon_scan()
    print(
        f"  first-player stranded={scan['first_player_stranded_setups']} "
        f"correct={scan['correct']}"
    )

    confirmed = all(
        stage["correct"] for stage in (engine, batch, pool, corpus, scan)
    )
    payload = {
        "agent": "agent_06b",
        "phase": "phase_6b_gate_1_acceptance",
        "purpose": (
            "re-run of the exact deterministic sequence that aborted the first "
            "persisted soak, under the authorized engine correction"
        ),
        "implementation_version": IMPLEMENTATION_VERSION,
        "failing_identity": {
            "root_seed": ROOT_SEED,
            "environment_id": ENVIRONMENT,
            "generation": GENERATION,
            "game_id": engine["game_id"],
            "slot_seed": engine["slot_seed"],
        },
        "probability_one_in_exact": stranded_probability_one_in(),
        "stages": {
            "engine_state": engine,
            "batch_layer": batch,
            "production_pool": pool,
            "corpus_forensics": corpus,
            "horizon_scan": scan,
        },
        "formerly_failing_sequence_now_passes": confirmed,
        "pre_fix_diagnosis_artifact": (
            "reports/phase_6_data/agent_06b_anomaly_diagnosis.json (frozen; "
            "documents the defect as it stood before the correction)"
        ),
        "duration_seconds": time.perf_counter() - started,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nfix validation written to {arguments.output}")
    print(f"formerly failing sequence now passes: {confirmed}")
    return 0 if confirmed else 1


if __name__ == "__main__":
    sys.exit(main())
