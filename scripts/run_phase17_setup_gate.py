#!/usr/bin/env python3
"""Phase 17 Agent 3: the extended setup gate (G-S).

Specification sources:

- `03_AGENT_3_AUTOREGRESSIVE_SETUP_NETWORK.md` sections 6, 7 and 9
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` section 12 ("Setup gate")
- `reports/phase17/phase17_contract_handoff_v1.json` -> `gates.gates[G-S]`

What this gate establishes, and what it does not
------------------------------------------------
It establishes legality, masking, orientation, causality, reproducibility,
diversity, the outcome-to-update path, checkpoint identity, and five-epoch
throughput. It does **not** establish setup strength: the soak's move policy
is a uniform-random legal fixture, so a stronger setup distribution has
nothing to be stronger *against*. Section 7 says so explicitly and the report
repeats it.

The move fixture lives here rather than in the library
------------------------------------------------------
Common contract section 5 bars rule-based agents from collection and training.
The random-move fixture below is a gate instrument: it exists to finish games
so that outcomes exist, and `tests/training/phase17/test_setup_structure.py`
asserts that nothing in `stratego/training/phase17/` can reach it.

Usage
-----
```text
python scripts/run_phase17_setup_gate.py --run-id RUN-2026-A
python scripts/run_phase17_setup_gate.py --quick        # a bounded rehearsal
```
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import BLUE, FLAG, RED, TRAINING_RULES  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.setup import validate_setup  # noqa: E402
from stratego.engine.state import create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.belief.phase15.orientation import (  # noqa: E402
    Phase15OrientationError,
    assert_engine_orientation,
)
from stratego.training.phase9_behavior import state_dict_digest  # noqa: E402
from stratego.training.phase17.setup_contract import (  # noqa: E402
    ORIENTATION_RULE_VERSION,
    SETUP_ENTROPY_BONUS_COEFFICIENT,
    SETUP_KL_PINNED_FRACTION_LIMIT,
    SETUP_CONTRACT_VERSION,
    SETUP_EPISODE_SCHEMA_VERSION,
    SETUP_EQUATION_VERSION,
    SETUP_MODEL_VERSION,
    SETUP_PREFIXES,
    SETUP_QUEUE_VERSION,
    Phase17SetupError,
    Phase17SetupGenerationError,
    SetupTrainingConfig,
    file_sha256,
    json_document_digest,
)
from stratego.training.phase17.setup_episode import (  # noqa: E402
    SetupEpisode,
    attach_setup_episodes,
    outcome_for,
)
from stratego.training.phase17.setup_learning import (  # noqa: E402
    SetupTrainer,
    advantage_terms,
    build_batch,
    setup_advantage,
    setup_batch_loss,
)
from stratego.training.phase17.setup_metrics import (  # noqa: E402
    DiversityAlarms,
    diversity_profile,
)
from stratego.training.phase17.setup_model import (  # noqa: E402
    Phase17SetupModel,
    assert_architecture,
    build_setup_model,
    count_parameters,
)
from stratego.training.phase17.setup_sampling import (  # noqa: E402
    generate_setups,
    masked_probabilities,
    to_engine_setup,
)

#: Common contract section 13's flag floor, and a reflection-class uniqueness
#: floor. The class floor is deliberately not 1.0: a handful of collisions in a
#: 320-sample pool is sampling, not collapse.
FLAG_EFFECTIVE_SUPPORT_FLOOR = 4.0
CLASS_UNIQUENESS_FLOOR = 0.95

SETUP_SOURCES = (
    "stratego/training/phase17/setup_contract.py",
    "stratego/training/phase17/setup_model.py",
    "stratego/training/phase17/setup_sampling.py",
    "stratego/training/phase17/setup_episode.py",
    "stratego/training/phase17/setup_learning.py",
    "stratego/training/phase17/setup_metrics.py",
    "stratego/training/phase17/__init__.py",
    "scripts/run_phase17_setup_gate.py",
)

TEST_SOURCES = (
    "tests/training/phase17/conftest.py",
    "tests/training/phase17/test_setup_model.py",
    "tests/training/phase17/test_setup_sampling.py",
    "tests/training/phase17/test_setup_episode.py",
    "tests/training/phase17/test_setup_learning.py",
    "tests/training/phase17/test_setup_metrics.py",
    "tests/training/phase17/test_setup_structure.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def peak_memory_mib() -> float:
    """Process high-water mark. `ru_maxrss` is bytes on Darwin, KiB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


class Recorder:
    """Collects named checks so a failure is reported, not raised away."""

    def __init__(self) -> None:
        self.checks: "list[dict]" = []

    def record(self, check_id: str, name: str, passed: bool, detail=None) -> bool:
        self.checks.append(
            {"id": check_id, "name": name, "passed": bool(passed), "detail": detail}
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_id} {name}", flush=True)
        if not passed and detail is not None:
            print(f"         {detail}", flush=True)
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        return all(check["passed"] for check in self.checks)

    def failures(self) -> "list[dict]":
        return [check for check in self.checks if not check["passed"]]


# ---------------------------------------------------------------------------
# The move fixture -- a gate instrument, never a training participant
# ---------------------------------------------------------------------------


def play_random_game(red_setup, blue_setup, game_id: str, seed: int, ply_cap: int) -> dict:
    """Play one game to termination with uniform-random legal moves."""
    state = create_game(red_setup, blue_setup, TRAINING_RULES, game_id=game_id)
    rng = random.Random(seed)
    plies = 0
    while not state.terminal and plies < ply_cap:
        actions = legal_actions(state)
        if not actions:
            break
        apply_action(state, actions[rng.randrange(len(actions))])
        plies += 1
    if not state.terminal:
        return {"terminal_result": None, "plies": plies, "reason": "ply_cap"}
    if state.winner is None:
        result = "draw"
    else:
        result = "red_win" if state.winner == RED else "blue_win"
    return {"terminal_result": result, "plies": plies, "reason": state.terminal_reason}


# ---------------------------------------------------------------------------
# Section 7: hard correctness
# ---------------------------------------------------------------------------


def hard_correctness(
    model: Phase17SetupModel,
    digest: str,
    run_id: str,
    sample_count: int,
    recorder: Recorder,
    rehearsal: bool = False,
) -> dict:
    print("\n== S1-S5 hard correctness ==", flush=True)
    per_side = sample_count // 2
    started = time.perf_counter()

    samples: "dict[int, list]" = {}
    generation_seconds: "dict[int, float]" = {}
    for color in (RED, BLUE):
        game_ids = [f"gate-{color}-{index}" for index in range(per_side)]
        color_started = time.perf_counter()
        drawn: "list" = []
        for start in range(0, per_side, 512):
            drawn.extend(
                generate_setups(
                    model,
                    run_id=run_id,
                    game_ids=game_ids[start : start + 512],
                    color=color,
                    model_state_digest=digest,
                    snapshot_iteration=0,
                )
            )
        generation_seconds[color] = time.perf_counter() - color_started
        samples[color] = drawn

    total = len(samples[RED]) + len(samples[BLUE])
    required = 5000 if not rehearsal else total
    recorder.record(
        "S1",
        f"at least {required:,} samples split across colours ({len(samples[RED])} red / "
        f"{len(samples[BLUE])} blue)",
        total >= required and len(samples[RED]) > 0 and len(samples[BLUE]) > 0,
        {"total": total, "required": required, "rehearsal": rehearsal},
    )

    # S2 -- inventory, legality, placement, lake and orientation.
    failures = {"inventory": 0, "orientation": 0, "game_creation": 0}
    flag_rows = {RED: [], BLUE: []}
    for color in (RED, BLUE):
        for sample in samples[color]:
            try:
                validate_setup(sample.canonical_setup, color)
                validate_setup(sample.engine_setup, color)
            except Exception:
                failures["inventory"] += 1
                continue
            try:
                assert_engine_orientation(sample.canonical_setup, sample.engine_setup, color)
            except Phase15OrientationError:
                failures["orientation"] += 1
                continue
            from stratego.engine.constants import SETUP_SQUARES

            flag_rows[color].append(SETUP_SQUARES[color][sample.engine_setup.index(FLAG)] // 10)

    creation_checked = 0
    for red, blue in zip(samples[RED], samples[BLUE]):
        try:
            create_game(red.engine_setup, blue.engine_setup, TRAINING_RULES, game_id="check")
            creation_checked += 1
        except Exception:
            failures["game_creation"] += 1

    recorder.record(
        "S2",
        "zero inventory, legality, placement, lake or orientation failures",
        sum(failures.values()) == 0,
        failures,
    )
    recorder.record(
        "S2b",
        f"every generated board round-trips through create_game ({creation_checked} boards)",
        failures["game_creation"] == 0,
        {"checked": creation_checked},
    )

    # The Phase 11B symptom, measured rather than assumed.
    blue_front_row = float(np.mean([row == 6 for row in flag_rows[BLUE]])) if flag_rows[BLUE] else 1.0
    red_front_row = float(np.mean([row == 3 for row in flag_rows[RED]])) if flag_rows[RED] else 1.0
    recorder.record(
        "S2c",
        "Blue flag-row distribution is not the mis-oriented one",
        blue_front_row < 0.5,
        {
            "blue_flag_on_engine_row_6_fraction": blue_front_row,
            "red_flag_on_engine_row_3_fraction": red_front_row,
            "reference_old_glue_blue_front_row_fraction": 0.770,
        },
    )

    # S3 -- adversarial masking at every later prefix.
    adversarial_failures = 0
    for color in (RED, BLUE):
        for sample in samples[color][:200]:
            tokens = np.asarray(sample.tokens, dtype=np.int64)
            for prefix in range(1, SETUP_PREFIXES):
                exhausted = ~sample.inventory_masks[prefix]
                if not exhausted.any():
                    continue
                logits = torch.full((1, 12), -60.0)
                logits[0, torch.as_tensor(np.nonzero(exhausted)[0])] = 1e6
                mask = torch.as_tensor(sample.inventory_masks[prefix]).unsqueeze(0)
                probabilities = masked_probabilities(logits, mask)
                if float(probabilities[0][torch.as_tensor(exhausted)].max()) != 0.0:
                    adversarial_failures += 1
    recorder.record(
        "S3",
        "adversarial logits cannot sample an exhausted type at any later prefix",
        adversarial_failures == 0,
        {"failures": adversarial_failures},
    )

    # S4 -- autoregressive causality.
    probe = torch.randint(0, 12, (8, SETUP_PREFIXES + 1), dtype=torch.long)
    probe[:, 0] = 12
    mutated = probe.clone()
    mutated[:, 21:] = (mutated[:, 21:] + 5) % 12
    with torch.no_grad():
        first = model(probe)
        second = model(mutated)
    causal = all(
        torch.allclose(first[name][:, :20], second[name][:, :20], atol=1e-6)
        for name in ("piece_logits", "wdl_logits", "conditional_entropy")
    )
    visible = not torch.allclose(first["piece_logits"][:, 25], second["piece_logits"][:, 25])
    recorder.record(
        "S4",
        "prefix k's outputs do not depend on tokens > k (and the probe is not vacuous)",
        causal and visible,
        {"earlier_prefixes_unchanged": causal, "later_prefixes_changed": visible},
    )

    # S5 -- deterministic trace, and a changed seed domain changing the draw.
    #
    # Two claims, deliberately separated. The setup and its token trace replay
    # EXACTLY from (run, game, side) at any batch shape -- that is the property
    # the contract needs, and it is exact because the inverse-CDF draw is
    # keyed by the token's own seed rather than by a shared generator.
    #
    # The recorded float32 probabilities are exact only at the SAME batch
    # shape: a batched GEMM is not shape-invariant, so redrawing one chain
    # alone reproduces the same tokens through arithmetic that differs in the
    # last ulp. This has no training consequence -- the ratio denominator is
    # the recorded probability, never a recomputation (section 5) -- but it is
    # measured and reported rather than assumed away, because a verifier that
    # re-derives a setup at a different batch size must expect it.
    single = generate_setups(
        model, run_id=run_id, game_ids=[f"gate-{RED}-0"], color=RED,
        model_state_digest=digest, snapshot_iteration=0,
    )[0]
    same_shape = generate_setups(
        model, run_id=run_id, game_ids=[f"gate-{RED}-{index}" for index in range(min(512, per_side))],
        color=RED, model_state_digest=digest, snapshot_iteration=0,
    )[0]
    reference = samples[RED][0]

    trace_exact = (
        single.canonical_setup == reference.canonical_setup
        and single.engine_setup == reference.engine_setup
        and single.per_token_seeds == reference.per_token_seeds
        and np.array_equal(single.tokens, reference.tokens)
        and np.array_equal(single.inventory_masks, reference.inventory_masks)
    )
    probability_delta = float(
        np.abs(single.behavior_probabilities - reference.behavior_probabilities).max()
    )
    same_shape_bitwise = (
        same_shape.canonical_setup == reference.canonical_setup
        and np.array_equal(
            same_shape.behavior_probabilities, reference.behavior_probabilities
        )
        and np.array_equal(
            same_shape.behavior_log_probabilities, reference.behavior_log_probabilities
        )
    )
    changed = reference.canonical_setup != samples[RED][1].canonical_setup

    recorder.record(
        "S5",
        "identical snapshot and seeds replay the setup and trace exactly; a changed "
        "seed domain redraws",
        trace_exact and changed,
        {
            "trace_bitwise_identical_at_any_batch_shape": trace_exact,
            "changed_seed_redraws": changed,
        },
    )
    recorder.record(
        "S5b",
        "a same-batch-shape replay is bitwise identical in every recorded field",
        same_shape_bitwise,
        {"same_shape_bitwise_identical": same_shape_bitwise},
    )
    recorder.record(
        "S5c",
        "a cross-batch-shape replay agrees on tokens and matches probabilities to float32 ulp",
        probability_delta < 1e-6,
        {
            "max_probability_delta": probability_delta,
            "note": (
                "batched GEMM is not batch-shape-invariant; the tokens are unaffected "
                "because each draw is keyed by its own seed, and training never "
                "recomputes the behavior probability"
            ),
        },
    )

    # No library or repair fallback is reachable -- asserted over the import
    # graph by tests/training/phase17/test_setup_structure.py, and here as a
    # behavioural refusal: a broken setup raises rather than being repaired.
    repaired = False
    try:
        to_engine_setup(tuple([FLAG] * 40), RED)
        repaired = True
    except Phase17SetupGenerationError:
        pass
    recorder.record(
        "S2d",
        "an illegal setup is refused, never repaired or replaced from a library",
        not repaired,
    )

    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "sample_count": total,
        "generation_seconds": {str(k): v for k, v in generation_seconds.items()},
        "seconds": elapsed,
        "failures": failures,
        "blue_flag_engine_row_histogram": {
            str(row): int(count)
            for row, count in zip(*np.unique(flag_rows[BLUE], return_counts=True))
        },
        "red_flag_engine_row_histogram": {
            str(row): int(count)
            for row, count in zip(*np.unique(flag_rows[RED], return_counts=True))
        },
    }


# ---------------------------------------------------------------------------
# Section 7: learning and outcome binding
# ---------------------------------------------------------------------------


def outcome_binding(samples, run_id: str, recorder: Recorder) -> dict:
    print("\n== S7-S8 outcome binding and reward flip ==", flush=True)
    signs = {}
    for result in ("red_win", "blue_win", "draw"):
        pair = attach_setup_episodes(
            samples[RED][0], samples[BLUE][0], run_id=run_id, game_id=f"sign-{result}"
        )
        pair.complete(result)
        signs[result] = {"red": pair.red.outcome, "blue": pair.blue.outcome}
    expected = {
        "red_win": {"red": 1, "blue": -1},
        "blue_win": {"red": -1, "blue": 1},
        "draw": {"red": 0, "blue": 0},
    }
    recorder.record("S7", "Red win, Blue win and draw signs bind independently", signs == expected, signs)

    rebind_refused = False
    pair = attach_setup_episodes(
        samples[RED][1], samples[BLUE][1], run_id=run_id, game_id="rebind"
    )
    pair.complete("red_win")
    try:
        pair.red.complete("blue_win")
    except Phase17SetupError:
        rebind_refused = True
    recorder.record("S7b", "an outcome cannot be rebound to a different result", rebind_refused)

    # S8 -- the reward-flip gradient test, isolated so it is exact.
    config = SetupTrainingConfig(run_id=run_id, total_iterations=626, device="cpu")
    model = build_setup_model(seed=808)
    neutral = np.full((SETUP_PREFIXES, 3), 1.0 / 3.0, dtype=np.float32)
    episodes = []
    for index in range(8):
        game = attach_setup_episodes(
            samples[RED][index], samples[BLUE][index], run_id=run_id, game_id=f"flip-{index}"
        )
        episodes.extend(game.complete("red_win" if index % 2 else "blue_win"))
    for episode in episodes:
        episode.prefix_wdl_predictions = neutral.copy()

    def policy_gradient(sign: int, clip_epsilon: float):
        flipped = []
        for episode in episodes:
            clone = copy.deepcopy(episode)
            clone.outcome = sign * clone.outcome
            flipped.append(clone)
        batch = build_batch(flipped, alpha=0.0)
        _, terms = setup_batch_loss(
            model, batch, config=config.replace(ppo_clip_epsilon=clip_epsilon), beta=0.0
        )
        model.zero_grad(set_to_none=True)
        terms["policy_loss"].backward()
        return model.piece_head.weight.grad.detach().clone(), float(terms["clip_fraction"].detach())

    forward, clip_fraction = policy_gradient(1, 1e6)
    reversed_, _ = policy_gradient(-1, 1e6)
    exact = bool(torch.allclose(forward, -reversed_, atol=1e-6)) and float(forward.abs().max()) > 0
    _, clipped_fraction = policy_gradient(1, 0.2)
    recorder.record(
        "S8",
        "reversing only the outcomes negates the policy gradient on the surrogate",
        exact,
        {
            "gradient_norm": float(forward.norm()),
            "clip_fraction_unclipped_probe": clip_fraction,
            "clip_fraction_at_epsilon_0.2": clipped_fraction,
            "note": (
                "asserted on the unclipped surrogate. -min(r*delta, clip(r)*delta) is "
                "asymmetric in the sign of delta, so once the policy is off the behavior "
                "snapshot a flip changes which rows are gradient-active rather than "
                "negating the gradient"
            ),
        },
    )
    return {"signs": signs, "reward_flip_exact": exact}


# ---------------------------------------------------------------------------
# Section 7: the real short soak
# ---------------------------------------------------------------------------


def soak(
    config: SetupTrainingConfig,
    *,
    iterations: int,
    games_per_iteration: int,
    consume: int,
    ply_cap: int,
    diversity_every: int,
    seed: int,
    recorder: Recorder,
) -> dict:
    print(
        f"\n== S9 real short soak: {iterations} iterations x {games_per_iteration} games ==",
        flush=True,
    )
    model = build_setup_model(device=config.device, seed=seed)
    trainer = SetupTrainer(model, config)
    initial_digest = state_dict_digest(model)

    rows: "list[dict]" = []
    diversity_checks: "list[dict]" = []
    baseline_profile = None
    alarms = None
    results_seen = {"red_win": 0, "blue_win": 0, "draw": 0, "ply_cap": 0}
    game_plies: "list[int]" = []
    generation_seconds = 0.0
    play_seconds = 0.0
    update_seconds = 0.0
    started = time.perf_counter()

    for iteration in range(1, iterations + 1):
        snapshot_digest = state_dict_digest(trainer.model)
        game_ids = [f"soak-{iteration}-{index}" for index in range(games_per_iteration)]

        mark = time.perf_counter()
        red = generate_setups(
            trainer.model, run_id=config.run_id, game_ids=game_ids, color=RED,
            model_state_digest=snapshot_digest, snapshot_iteration=trainer.setup_iteration,
        )
        blue = generate_setups(
            trainer.model, run_id=config.run_id, game_ids=game_ids, color=BLUE,
            model_state_digest=snapshot_digest, snapshot_iteration=trainer.setup_iteration,
        )
        generation_seconds += time.perf_counter() - mark

        if baseline_profile is None:
            baseline_profile = diversity_profile(
                [sample.canonical_setup for sample in red + blue],
                behavior_probabilities=np.stack(
                    [sample.behavior_probabilities for sample in red + blue]
                ),
                suffix_information=np.stack(
                    [sample.suffix_information_content for sample in red + blue]
                ),
                label="soak_iteration_1",
            )
            alarms = DiversityAlarms.from_baseline(baseline_profile)

        mark = time.perf_counter()
        for index, game_id in enumerate(game_ids):
            pair = attach_setup_episodes(
                red[index], blue[index], run_id=config.run_id, game_id=game_id
            )
            outcome = play_random_game(
                *pair.engine_setups(), game_id=game_id,
                seed=(hash((iteration, index)) if False else iteration * 100_003 + index),
                ply_cap=ply_cap,
            )
            game_plies.append(outcome["plies"])
            if outcome["terminal_result"] is None:
                results_seen["ply_cap"] += 1
                continue
            results_seen[outcome["terminal_result"]] += 1
            for episode in pair.complete(outcome["terminal_result"]):
                trainer.queue.enqueue(episode)
        play_seconds += time.perf_counter() - mark

        mark = time.perf_counter()
        result = trainer.update(batch_episodes=consume)
        update_seconds += time.perf_counter() - mark

        row = {
            "iteration": iteration,
            "alpha": result.alpha,
            "skipped": result.skipped,
            "episodes_consumed": result.episodes_consumed,
            "optimizer_steps": result.optimizer_steps,
            "beta_after": result.beta_after,
            "gradient_norm_mean": result.gradient_norm_mean,
            "queue_depth": (result.queue or {}).get("depth"),
            "queue_skip_count": (result.queue or {}).get("skip_count"),
        }
        if not result.skipped:
            final_epoch = result.epochs[-1]
            row.update(
                {
                    "policy_loss": final_epoch["policy_loss"],
                    "value_loss": final_epoch["value_loss"],
                    "conditional_entropy_loss": final_epoch["conditional_entropy_loss"],
                    "behavior_kl": final_epoch["mean_behavior_kl"],
                    "total_loss": final_epoch["total_loss"],
                    "clip_fraction": final_epoch["clip_fraction"],
                    "model_mean_prefix_entropy_nats": final_epoch["mean_prefix_entropy_nats"],
                    "advantage_mean": final_epoch["advantage_mean"],
                    "outcome_term_abs_mean": result.advantage_telemetry["outcome_term_abs_mean"],
                    "entropy_term_abs_mean": result.advantage_telemetry["entropy_term_abs_mean"],
                    "digest_changed": result.digest_before != result.digest_after,
                }
            )
        rows.append(row)

        if diversity_every and (iteration % diversity_every == 0 or iteration == iterations):
            profile = diversity_profile(
                [sample.canonical_setup for sample in red + blue],
                behavior_probabilities=np.stack(
                    [sample.behavior_probabilities for sample in red + blue]
                ),
                suffix_information=np.stack(
                    [sample.suffix_information_content for sample in red + blue]
                ),
                label=f"soak_iteration_{iteration}",
            )
            verdict = alarms.evaluate(profile)
            diversity_checks.append({"iteration": iteration, "profile": profile, "verdict": verdict})
            print(
                f"  iter {iteration:>4}  alpha {result.alpha:.4f}  "
                f"H {profile['mean_prefix_entropy_nats']:.4f}  "
                f"flag_eff {profile['flag_effective_support']:.2f}  "
                f"classuniq {profile['reflection_class_unique_fraction']:.3f}  "
                f"KL {row.get('behavior_kl', float('nan')):.5f}  "
                f"beta {result.beta_after:.4f}  [{verdict['status']}]",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    final_digest = state_dict_digest(trainer.model)
    trained = [row for row in rows if not row["skipped"]]
    kl_target = config.kl_target
    kl_hard_limit = config.kl_hard_limit
    kl_bounds = tuple(config.kl_beta_bounds)

    recorder.record(
        "S9",
        "completed outcomes produce optimizer steps, gradients and a changed raw digest",
        bool(trained)
        and all(row["optimizer_steps"] > 0 for row in trained)
        and all(row["gradient_norm_mean"] > 0.0 for row in trained)
        and all(row["digest_changed"] for row in trained)
        and initial_digest != final_digest,
        {
            "iterations_trained": len(trained),
            "iterations_skipped": len(rows) - len(trained),
            "initial_digest": initial_digest,
            "final_digest": final_digest,
        },
    )

    last = diversity_checks[-1]["profile"] if diversity_checks else baseline_profile
    hard_trips = [check for check in diversity_checks if check["verdict"]["status"] == "hard"]

    # Common contract section 13: one low reading is a WARNING; the stop needs
    # the floor held for three consecutive checks. Two refinements matter.
    #
    # First, only regular-cadence checks count. The final check is forced on
    # whatever iteration the soak ends at, which can land one iteration after
    # the previous one -- two near-simultaneous readings are one observation,
    # not two, and letting them both count would weaken the very rule that
    # exists to reject a noisy single reading.
    #
    # Second, the run must be CONSECUTIVE. Counting total hard readings would
    # stop on three isolated dips spread across the run, which section 13
    # explicitly declines to do.
    on_cadence = [
        check
        for check in diversity_checks
        if diversity_every and check["iteration"] % diversity_every == 0
    ]
    longest_run = current_run = 0
    for check in on_cadence:
        current_run = current_run + 1 if check["verdict"]["status"] == "hard" else 0
        longest_run = max(longest_run, current_run)
    consecutive_required = alarms.consecutive_checks

    recorder.record(
        "S6",
        f"entropy and diversity do not collapse over the soak (floor held for "
        f"{consecutive_required} consecutive checks is a stop)",
        longest_run < consecutive_required,
        {
            "baseline_mean_prefix_entropy_nats": baseline_profile["mean_prefix_entropy_nats"],
            "final_mean_prefix_entropy_nats": last["mean_prefix_entropy_nats"],
            "hard_floor": alarms.hard_mean_prefix_entropy_nats,
            "hard_readings_total": len(hard_trips),
            "hard_readings_on_cadence": [
                check["iteration"]
                for check in on_cadence
                if check["verdict"]["status"] == "hard"
            ],
            "longest_consecutive_hard_run_on_cadence": longest_run,
            "consecutive_required_to_stop": consecutive_required,
            "off_cadence_checks_excluded": [
                check["iteration"]
                for check in diversity_checks
                if diversity_every and check["iteration"] % diversity_every != 0
            ],
            "rule": "common contract section 13 -- one reading is a warning, not a stop",
        },
    )

    flag_supports = [check["profile"]["flag_effective_support"] for check in diversity_checks]
    class_fractions = [
        check["profile"]["reflection_class_unique_fraction"] for check in diversity_checks
    ]
    recorder.record(
        "S6b",
        f"flag effective support stays at or above {FLAG_EFFECTIVE_SUPPORT_FLOOR}",
        bool(flag_supports) and min(flag_supports) >= FLAG_EFFECTIVE_SUPPORT_FLOOR,
        {
            "minimum": min(flag_supports) if flag_supports else None,
            "final": flag_supports[-1] if flag_supports else None,
            "floor": FLAG_EFFECTIVE_SUPPORT_FLOOR,
        },
    )
    recorder.record(
        "S6c",
        "reflection-class diversity stays healthy across the soak",
        bool(class_fractions) and min(class_fractions) >= CLASS_UNIQUENESS_FLOOR,
        {
            "minimum": min(class_fractions) if class_fractions else None,
            "final": class_fractions[-1] if class_fractions else None,
            "floor": CLASS_UNIQUENESS_FLOOR,
            "note": (
                "measured by reflection class, not exact tuple -- mirrored copies "
                "inflate raw uniqueness and would hide a collapse"
            ),
        },
    )

    # D5: the controller must both stay under its hard limit AND actually act.
    kls = [row["behavior_kl"] for row in trained]
    betas = [row["beta_after"] for row in trained]
    low, high = kl_bounds
    at_low = sum(1 for beta in betas if beta <= low * 1.0000001) / len(betas)
    at_high = sum(1 for beta in betas if beta >= high * 0.9999999) / len(betas)
    over_hard = [row["iteration"] for row in trained if row["behavior_kl"] > kl_hard_limit]
    recorder.record(
        "S12",
        f"setup KL stays below its hard limit ({kl_hard_limit}) and the controller is "
        f"not effectively pinned",
        not over_hard
        and max(at_low, at_high) < SETUP_KL_PINNED_FRACTION_LIMIT,
        {
            "kl_mean": float(np.mean(kls)),
            "kl_p95": float(np.quantile(kls, 0.95)),
            "kl_max": float(np.max(kls)),
            "kl_hard_limit": kl_hard_limit,
            "iterations_over_hard_limit": over_hard,
            "beta_min": float(np.min(betas)),
            "beta_max": float(np.max(betas)),
            "beta_final": betas[-1],
            "fraction_at_lower_bound": at_low,
            "fraction_at_upper_bound": at_high,
            "pinned_fraction_limit": SETUP_KL_PINNED_FRACTION_LIMIT,
            "note": "a controller living at a bound is not regulating anything",
        },
    )

    return {
        "iterations": iterations,
        "games_per_iteration": games_per_iteration,
        "kl_controller": {
            "target": kl_target,
            "hard_limit": kl_hard_limit,
            "bounds": list(kl_bounds),
            "fraction_at_lower_bound": at_low,
            "fraction_at_upper_bound": at_high,
            "beta_min": float(np.min(betas)),
            "beta_max": float(np.max(betas)),
            "beta_final": betas[-1],
            "kl_mean": float(np.mean(kls)),
            "kl_p95": float(np.quantile(kls, 0.95)),
            "kl_max": float(np.max(kls)),
        },
        "consume": consume,
        "ply_cap": ply_cap,
        "seconds": elapsed,
        "generation_seconds": generation_seconds,
        "play_seconds": play_seconds,
        "update_seconds": update_seconds,
        "game_results": results_seen,
        "game_plies_mean": float(np.mean(game_plies)) if game_plies else None,
        "game_plies_max": int(np.max(game_plies)) if game_plies else None,
        "initial_digest": initial_digest,
        "final_digest": final_digest,
        "rows": rows,
        "baseline_profile": baseline_profile,
        "diversity_checks": diversity_checks,
        "alarms": alarms.document() if alarms else None,
        "trainer": trainer,
    }


# ---------------------------------------------------------------------------
# Section 7: the checkpoint round trip
# ---------------------------------------------------------------------------


def checkpoint_round_trip(trainer: SetupTrainer, config: SetupTrainingConfig, recorder: Recorder) -> dict:
    print("\n== S10 checkpoint round trip ==", flush=True)
    document = trainer.state_document()
    restored = SetupTrainer(build_setup_model(device=config.device, seed=1), config)
    restored.load_state_document(document)

    raw_ok = state_dict_digest(restored.model) == document["setup_raw_model_state_digest"]
    ema_ok = state_dict_digest(restored.ema.as_model()) == document["setup_ema_model_state_digest"]
    controller_ok = (
        restored.controller.beta == trainer.controller.beta
        and restored.controller.direction == trainer.controller.direction
    )
    queue_ok = len(restored.queue) == len(trainer.queue)
    optimizer_ok = (
        restored.optimizer.state_dict()["param_groups"]
        == trainer.optimizer.state_dict()["param_groups"]
    )
    episodes_ok = [episode.identity() for episode in restored.queue._queue] == [
        episode.identity() for episode in trainer.queue._queue
    ]

    refusals = {}
    for name, mutate in (
        ("foreign_run", lambda d: d.__setitem__("run_id", "RUN-OTHER")),
        ("foreign_config", lambda d: d.__setitem__("config_digest", "0" * 64)),
        ("partial", lambda d: d.pop("setup_optimizer_state")),
    ):
        broken = dict(document)
        mutate(broken)
        try:
            SetupTrainer(build_setup_model(device=config.device, seed=2), config).load_state_document(
                broken
            )
            refusals[name] = False
        except Phase17SetupError:
            refusals[name] = True

    recorder.record(
        "S10",
        "raw, EMA, optimizer, setup KL, queue and episode data round-trip",
        all([raw_ok, ema_ok, controller_ok, queue_ok, optimizer_ok, episodes_ok]),
        {
            "raw": raw_ok, "ema": ema_ok, "kl_controller": controller_ok,
            "queue": queue_ok, "optimizer": optimizer_ok, "episodes": episodes_ok,
        },
    )
    recorder.record(
        "S10b",
        "a foreign run, a foreign config digest and a partial state are all refused",
        all(refusals.values()),
        refusals,
    )
    return {"round_trip": {"raw": raw_ok, "ema": ema_ok, "queue": queue_ok}, "refusals": refusals}


# ---------------------------------------------------------------------------
# Section 6: the throughput decision
# ---------------------------------------------------------------------------


def throughput(
    config: SetupTrainingConfig,
    *,
    devices: "list[str]",
    pool_sizes: "list[int]",
    batch_episodes: int,
    repeats: int,
    recorder: Recorder,
) -> dict:
    print("\n== S11 five-epoch throughput ==", flush=True)
    measurements = {}
    for device in devices:
        model = build_setup_model(device=device, seed=6)
        digest = state_dict_digest(model)
        device_config = config.replace(device=device)

        pools = {}
        for size in pool_sizes:
            game_ids = [f"tp-{size}-{index}" for index in range(size)]
            mark = time.perf_counter()
            generate_setups(
                model, run_id=config.run_id, game_ids=game_ids, color=RED,
                model_state_digest=digest, snapshot_iteration=0,
            )
            seconds = time.perf_counter() - mark
            pools[str(size)] = {
                "seconds": seconds,
                "samples_per_second": size / seconds,
                "seconds_per_side_pool": seconds,
                "seconds_for_both_sides": seconds * 2,
            }
            print(f"  {device}: pool {size:>5} -> {seconds:6.2f}s ({size/seconds:6.1f}/s)", flush=True)

        game_ids = [f"tpu-{index}" for index in range(batch_episodes // 2)]
        red = generate_setups(
            model, run_id=config.run_id, game_ids=game_ids, color=RED,
            model_state_digest=digest, snapshot_iteration=0,
        )
        blue = generate_setups(
            model, run_id=config.run_id, game_ids=game_ids, color=BLUE,
            model_state_digest=digest, snapshot_iteration=0,
        )
        episodes = []
        for index, game_id in enumerate(game_ids):
            pair = attach_setup_episodes(red[index], blue[index], run_id=config.run_id, game_id=game_id)
            episodes.extend(pair.complete(["red_win", "blue_win", "draw"][index % 3]))

        epoch_timings = {}
        for epochs in (1, 5):
            durations = []
            for repeat in range(repeats):
                trainer = SetupTrainer(
                    build_setup_model(device=device, seed=6),
                    device_config.replace(epochs_per_iteration=epochs),
                )
                for episode in episodes:
                    trainer.queue.enqueue(copy.deepcopy(episode))
                mark = time.perf_counter()
                trainer.update(batch_episodes=len(episodes))
                durations.append(time.perf_counter() - mark)
            epoch_timings[f"{epochs}_epoch"] = {
                "seconds_mean": float(np.mean(durations)),
                "seconds_min": float(np.min(durations)),
                "episodes": len(episodes),
                "repeats": repeats,
            }
            print(
                f"  {device}: {epochs} epoch(s) on {len(episodes)} episodes -> "
                f"{np.mean(durations):.3f}s",
                flush=True,
            )
        measurements[device] = {
            "pool_generation": pools,
            "update": epoch_timings,
            "five_epoch_overhead_seconds": (
                epoch_timings["5_epoch"]["seconds_mean"] - epoch_timings["1_epoch"]["seconds_mean"]
            ),
        }

    peak = peak_memory_mib()
    recorder.record(
        "S11",
        "five-epoch setup throughput measured (generation, forward/backward, total)",
        bool(measurements),
        {"devices": list(measurements)},
    )
    return {"measurements": measurements, "peak_memory_mib": peak}


#: Five epochs are affordable while the setup half stays well inside the move
#: cadence. A third of the run would be a real argument for fewer; anything
#: near a tenth is not.
FIVE_EPOCH_AFFORDABLE_FRACTION = 0.25


def project_budget(measurement: dict, *, pool_size: int, run_hours: float, iterations: int) -> dict:
    """What the measured setup cost implies for a 12-hour tandem run."""
    pool = measurement["pool_generation"][str(pool_size)]["seconds_for_both_sides"]
    update = measurement["update"]["5_epoch"]["seconds_mean"]
    per_iteration = pool + update
    total = per_iteration * iterations
    budget = run_hours * 3600.0
    return {
        "pool_size_per_side": pool_size,
        "assumed_iterations": iterations,
        "setup_seconds_per_iteration": per_iteration,
        "generation_seconds_per_iteration": pool,
        "five_epoch_update_seconds_per_iteration": update,
        "projected_setup_seconds_over_run": total,
        "projected_setup_hours_over_run": total / 3600.0,
        "run_budget_hours": run_hours,
        "projected_fraction_of_run": total / budget,
        "one_epoch_alternative_fraction_of_run": (
            (pool + measurement["update"]["1_epoch"]["seconds_mean"]) * iterations / budget
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 17 Agent 3 setup gate (G-S)")
    parser.add_argument("--run-id", default="RUN-2026-A")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--total-iterations", type=int, default=626)
    parser.add_argument("--soak-iterations", type=int, default=626)
    parser.add_argument("--soak-games", type=int, default=96)
    parser.add_argument("--soak-consume", type=int, default=192)
    parser.add_argument("--minibatch-episodes", type=int, default=64)
    parser.add_argument("--ply-cap", type=int, default=1500)
    parser.add_argument("--diversity-every", type=int, default=25)
    parser.add_argument("--throughput-devices", default="cpu,mps")
    parser.add_argument("--throughput-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1717)
    parser.add_argument("--reports", default="reports/phase17")
    parser.add_argument("--quick", action="store_true", help="a bounded rehearsal, not the gate")
    arguments = parser.parse_args(argv)

    if arguments.quick:
        # A rehearsal must never be able to overwrite a real gate artifact.
        # `--quick` differs from the gate only in scale, so its output lands in
        # the same shape at the same default path -- and a 55-minute run is
        # then one careless command away from being replaced by a 6-second
        # one. The rehearsal gets its own directory unless the caller named a
        # destination explicitly.
        if arguments.reports == parser.get_default("reports"):
            arguments.reports = "reports/phase17/rehearsal"
        arguments.samples = 400
        arguments.soak_iterations = 6
        arguments.soak_games = 8
        arguments.soak_consume = 16
        arguments.minibatch_episodes = 8
        arguments.diversity_every = 3
        arguments.throughput_devices = "cpu"
        arguments.throughput_repeats = 1

    torch.manual_seed(arguments.seed)
    recorder = Recorder()
    started = time.perf_counter()
    started_utc = utc_now()

    config = SetupTrainingConfig(
        run_id=arguments.run_id,
        total_iterations=arguments.total_iterations,
        device=arguments.device,
        minibatch_episodes=arguments.minibatch_episodes,
    )

    print(f"Phase 17 setup gate  run_id={arguments.run_id}  started {started_utc}", flush=True)
    print(f"  config_digest {config.config_digest()}", flush=True)

    print("\n== S0 architecture ==", flush=True)
    model = build_setup_model(device=arguments.device, seed=arguments.seed)
    architecture = assert_architecture(model)
    digest = state_dict_digest(model)
    recorder.record(
        "S0",
        f"parameter count is the frozen 802,320 (observed {count_parameters(model)})",
        count_parameters(model) == 802_320,
        architecture,
    )

    correctness = hard_correctness(
        model, digest, arguments.run_id, arguments.samples, recorder, rehearsal=arguments.quick
    )
    samples = correctness.pop("samples")

    print("\n== S6 initial diversity ==", flush=True)
    initial_profile = diversity_profile(
        [sample.canonical_setup for sample in samples[RED] + samples[BLUE]],
        behavior_probabilities=np.stack(
            [sample.behavior_probabilities for sample in samples[RED] + samples[BLUE]]
        ),
        suffix_information=np.stack(
            [sample.suffix_information_content for sample in samples[RED] + samples[BLUE]]
        ),
        label="initial_masked_model",
    )
    for name in (
        "reflection_class_unique_fraction",
        "mean_prefix_entropy_nats",
        "mean_class_distance",
        "flag_effective_support",
        "bomb_pattern_unique",
        "mean_top_token_concentration",
    ):
        print(f"  {name}: {initial_profile[name]}", flush=True)
    recorder.record(
        "S6a",
        "the initial distribution is diverse by reflection class, not only exactly",
        initial_profile["reflection_class_unique_fraction"] > 0.99,
        {
            "exact": initial_profile["exact_unique_fraction"],
            "reflection_class": initial_profile["reflection_class_unique_fraction"],
        },
    )

    binding = outcome_binding(samples, arguments.run_id, recorder)

    soak_result = soak(
        config,
        iterations=arguments.soak_iterations,
        games_per_iteration=arguments.soak_games,
        consume=arguments.soak_consume,
        ply_cap=arguments.ply_cap,
        diversity_every=arguments.diversity_every,
        seed=arguments.seed,
        recorder=recorder,
    )
    trainer = soak_result.pop("trainer")

    round_trip = checkpoint_round_trip(trainer, config, recorder)

    devices = [name for name in arguments.throughput_devices.split(",") if name]
    devices = [name for name in devices if name != "mps" or torch.backends.mps.is_available()]
    throughput_result = throughput(
        config,
        devices=devices,
        pool_sizes=[512, 1000] if not arguments.quick else [64],
        batch_episodes=arguments.soak_consume,
        repeats=arguments.throughput_repeats,
        recorder=recorder,
    )
    projections = {
        device: project_budget(
            measurement,
            pool_size=512 if not arguments.quick else 64,
            run_hours=12.0,
            iterations=arguments.total_iterations,
        )
        for device, measurement in throughput_result["measurements"].items()
    }

    reference = projections.get("cpu") or next(iter(projections.values()))
    recorder.record(
        "S11b",
        f"five setup epochs remain affordable "
        f"({reference['projected_fraction_of_run']:.1%} of a 12-hour run)",
        reference["projected_fraction_of_run"] < FIVE_EPOCH_AFFORDABLE_FRACTION,
        {
            "five_epoch_fraction_of_run": reference["projected_fraction_of_run"],
            "one_epoch_fraction_of_run": reference["one_epoch_alternative_fraction_of_run"],
            "limit": FIVE_EPOCH_AFFORDABLE_FRACTION,
            "epochs_retained": 5,
        },
    )

    elapsed = time.perf_counter() - started
    reports = REPOSITORY_ROOT / arguments.reports
    reports.mkdir(parents=True, exist_ok=True)

    # Belt and braces: refuse to replace a full-gate artifact with a rehearsal
    # even if the caller pointed --reports at the real directory by hand.
    existing = reports / "agent_03_setup_gate.json"
    if arguments.quick and existing.is_file():
        try:
            previous = json.loads(existing.read_text())
        except (OSError, ValueError):
            previous = {}
        if not previous.get("quick_rehearsal", True):
            raise SystemExit(
                f"refusing to overwrite the full-gate artifact at {existing} with a "
                f"--quick rehearsal (it recorded {previous.get('soak', {}).get('iterations')} "
                "soak iterations). Pass --reports <other-dir> if that is really what you want."
            )

    source_digests = {
        path: file_sha256(REPOSITORY_ROOT / path)
        for path in SETUP_SOURCES + TEST_SOURCES
        if (REPOSITORY_ROOT / path).is_file()
    }
    source_digest = json_document_digest(source_digests)

    gate = {
        "artifact": "agent_03_setup_gate.json",
        "gate": "G-S",
        "work_package": "phase17",
        "run_id": arguments.run_id,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "quick_rehearsal": bool(arguments.quick),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": arguments.device,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "arguments": vars(arguments),
        "config": config.document(),
        "config_digest": config.config_digest(),
        "architecture": architecture,
        "setup_equation_version": SETUP_EQUATION_VERSION,
        "entropy_bonus": {
            "form": "0.9 * alpha * (I/10)",
            "coefficient": SETUP_ENTROPY_BONUS_COEFFICIENT,
            "centered": False,
            "operator_decision": "D7-B",
            "retired_form": "alpha * (I/10 - h)  [D4, retired by D7-B]",
            "head_retained": "the conditional-entropy head and L_h are retained for "
                             "telemetry and paper alignment; the advantage no longer reads h",
        },
        "initial_model_state_digest": digest,
        "source_digests": source_digests,
        "source_digest": source_digest,
        "result": (
            ("REHEARSAL_PASS" if recorder.all_passed else "REHEARSAL_FAIL")
            if arguments.quick
            else ("PASS" if recorder.all_passed else "FAIL")
        ),
        "checks": recorder.checks,
        "failures": recorder.failures(),
        "hard_correctness": correctness,
        "initial_diversity_profile": initial_profile,
        "outcome_binding": binding,
        "soak": soak_result,
        "checkpoint_round_trip": round_trip,
        "throughput": throughput_result,
        "budget_projection": projections,
        "not_established": [
            "setup strength: the soak's move policy is a uniform-random legal fixture",
            "any claim about production EWR, which this gate does not measure",
            "the production value of N, which Agent 4 freezes at preflight",
        ],
    }
    gate_path = reports / "agent_03_setup_gate.json"
    gate_path.write_text(json.dumps(gate, indent=1, sort_keys=True, default=str))

    throughput_document = {
        "artifact": "agent_03_setup_throughput.json",
        "work_package": "phase17",
        "run_id": arguments.run_id,
        "recorded_utc": utc_now(),
        "host": gate["host"],
        "epochs_per_iteration": config.epochs_per_iteration,
        "epochs_reduction_rule": (
            "five epochs is the paper default; fewer only with a measured case and "
            "the operator's decision. Silent reduction is prohibited."
        ),
        "measurements": throughput_result["measurements"],
        "peak_memory_mib": throughput_result["peak_memory_mib"],
        "budget_projection": projections,
        "soak_cost": {
            "generation_seconds": soak_result["generation_seconds"],
            "play_seconds": soak_result["play_seconds"],
            "update_seconds": soak_result["update_seconds"],
            "iterations": soak_result["iterations"],
        },
    }
    throughput_path = reports / "agent_03_setup_throughput.json"
    throughput_path.write_text(json.dumps(throughput_document, indent=1, sort_keys=True, default=str))

    print(f"\n{'='*72}")
    print(f"GATE G-S: {gate['result']}  ({elapsed/60.0:.1f} minutes)")
    for failure in recorder.failures():
        print(f"  FAILED {failure['id']} {failure['name']}")
    for written in (gate_path, throughput_path):
        try:
            shown = written.relative_to(REPOSITORY_ROOT)
        except ValueError:
            shown = written
        print(f"  wrote {shown}")
    print("=" * 72)
    return 0 if recorder.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
