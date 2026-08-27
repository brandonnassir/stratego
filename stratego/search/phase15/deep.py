"""Phase 15 Agent 2 follow-up: the deeper-search pilot.

One narrow question, asked of one system:

> Does buying roughly 2-4x more search compute make the selected
> **P24 + B24** meaningfully stronger than MEDIUM, and what does it cost?

What is held fixed
------------------
Everything except the world count and the rollout depth. The move model, the
belief specialist, their digests, the candidate rule, `beta`, `epsilon`, the
score definition, the board pack, the opponents and every per-decision seed
are the ones already frozen and already measured. `LARGE` and `XLARGE` inherit
`max_root_candidates`, `beta` and `epsilon` from the same defaults MEDIUM
uses, by not passing them at all, so "candidate handling and regularization
unchanged" is a property of the configuration rather than a promise.

Why MEDIUM is not re-run
------------------------
MEDIUM was already played on exactly this board list, with exactly these
per-decision seeds, in Stage C. Replaying it would consume 70 minutes to
reproduce rows that are a deterministic function of inputs that have not
changed. :func:`check_medium_reproduces` re-plays a sample of those games
instead and requires the stored outcome, ply count and full action sequence
to come back identical — which is a stronger statement than a fresh run
would have made, because it also proves the reuse is sound.

"% moves differing from MEDIUM", carefully
-------------------------------------------
Inside a match game this quantity is ill-defined: the moment a rung plays a
different move the two games diverge and later positions are no longer
comparable, so a naive per-ply comparison measures divergence of *positions*,
not of decisions. The pilot therefore measures decision divergence on the
fixed diagnostic position manifest, where every rung answers the same
question, and reports the match-game divergence separately as the ply at
which the two games first part company.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ...engine.legal_moves import legal_actions
from .contract import (
    DEEP_MEANINGFUL_GAIN_HIGH,
    DEEP_MEANINGFUL_GAIN_LOW,
    DEEP_PILOT_PAIRING,
    DEEP_PILOT_PRESET_NAMES,
    DEEP_PILOT_VERSION,
    Phase15SearchError,
    naive_compute_units,
    preset as preset_of,
)

#: The rung the pilot compares everything against.
BASELINE_PRESET = DEEP_PILOT_PRESET_NAMES[0]


class Phase15DeepError(Phase15SearchError):
    """A deeper-search pilot check failed."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def check_frozen_identity(models, candidate: dict) -> dict:
    """The pilot runs on the frozen candidate's exact bytes, or not at all."""
    pairing = candidate["selected_system"]["pairing_id"]
    if pairing != DEEP_PILOT_PAIRING:
        raise Phase15DeepError(
            f"the pilot is defined for {DEEP_PILOT_PAIRING}, the frozen candidate "
            f"selects {pairing}"
        )
    move = models.move_models["p24"].identity
    belief = models.specialists["b24"].identity
    findings = []
    checks = {
        "move_model_checkpoint_sha256": (
            move["checkpoint_sha256"],
            candidate["move_model"]["checkpoint_sha256"],
        ),
        "move_model_state_digest": (
            move["model_state_digest"],
            candidate["move_model"]["model_state_digest"],
        ),
        "belief_checkpoint_sha256": (
            belief["checkpoint_sha256"],
            candidate["belief_model"]["checkpoint_sha256"],
        ),
        "belief_state_digest": (
            belief["state_digest"],
            candidate["belief_model"]["state_digest"],
        ),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            findings.append(f"{name}: loaded {observed} != frozen {expected}")
    if float(belief["applied_temperature"]) != float(
        candidate["belief_calibration"]["applied_temperature"]
    ):
        findings.append("the applied belief temperature differs from the frozen record")
    return {
        "passed": not findings,
        "findings": findings,
        "pairing_id": pairing,
        "move_model_state_digest": move["model_state_digest"],
        "belief_state_digest": belief["state_digest"],
        "applied_temperature": belief["applied_temperature"],
    }


def check_configuration_invariants() -> dict:
    """The pilot's own control: only worlds and depth may differ from MEDIUM."""
    baseline = preset_of(BASELINE_PRESET)
    findings = []
    rows = {}
    for name in DEEP_PILOT_PRESET_NAMES:
        config = preset_of(name)
        for field in ("max_root_candidates", "beta", "epsilon", "deduplicate_worlds"):
            if getattr(config, field) != getattr(baseline, field):
                findings.append(
                    f"{name}: {field} is {getattr(config, field)!r}, MEDIUM's is "
                    f"{getattr(baseline, field)!r}; the pilot may not change it"
                )
        if name != BASELINE_PRESET:
            if config.worlds <= baseline.worlds:
                findings.append(f"{name}: worlds did not grow beyond MEDIUM's")
            if config.rollout_depth < baseline.rollout_depth:
                findings.append(f"{name}: depth fell below MEDIUM's")
        rows[name] = {
            "worlds": config.worlds,
            "rollout_depth": config.rollout_depth,
            "max_root_candidates": config.max_root_candidates,
            "beta": config.beta,
            "epsilon": config.epsilon,
            "naive_compute_units": naive_compute_units(config),
            "naive_ratio_vs_medium": round(
                naive_compute_units(config) / naive_compute_units(baseline), 3
            ),
        }
    return {"passed": not findings, "findings": findings, "rungs": rows}


# ---------------------------------------------------------------------------
# Determinism, legality, worlds
# ---------------------------------------------------------------------------


def check_determinism(models, states, *, presets=DEEP_PILOT_PRESET_NAMES, seed: int = 606) -> dict:
    """A fixed seed must reproduce the same action, worlds and scores.

    Checked at *every* rung, not just the cheap one: more worlds means more
    draws from the accepted sampler and a longer rollout batch, which is
    exactly where a determinism defect would first appear.
    """
    from .systems import build_engine

    findings = []
    observed = {"decisions": 0, "repeats": 0}
    per_rung = {}
    for name in presets:
        bundle = build_engine(DEEP_PILOT_PAIRING, models, name)
        legal_all = 0
        for _row, state, _plan in states:
            legal = set(legal_actions(state))
            first = bundle.engine.choose_action(state, seed=seed)
            again = bundle.engine.choose_action(state, seed=seed)
            observed["decisions"] += 1
            observed["repeats"] += 1
            if first.selected_action_id != again.selected_action_id:
                findings.append(f"{name}: the same seed chose two different actions")
            if first.world_weights != again.world_weights:
                findings.append(f"{name}: the same seed sampled different worlds")
            if [round(c.q_value, 12) for c in first.candidates] != [
                round(c.q_value, 12) for c in again.candidates
            ]:
                findings.append(f"{name}: the same seed produced different Q values")
            if first.selected_action_id not in legal:
                findings.append(f"{name}: selected an illegal action")
            else:
                legal_all += 1
            if sum(first.world_weights) != first.worlds_requested:
                findings.append(f"{name}: world weights do not sum to the budget")
            if not any(candidate.is_direct for candidate in first.candidates):
                findings.append(f"{name}: the direct action was not a candidate")
            widths = {len(candidate.world_values) for candidate in first.candidates}
            if widths != {first.unique_worlds}:
                findings.append(
                    f"{name}: candidates were not all evaluated on the same worlds"
                )
        per_rung[name] = {"positions": len(states), "legal_decisions": legal_all}
    return {"passed": not findings, "findings": findings, "rungs": per_rung, **observed}


def check_worlds_legal(models, states, *, presets=DEEP_PILOT_PRESET_NAMES) -> dict:
    """Every sampled world stays legal at the larger world counts."""
    from ...evaluation.phase11_baselines import validate_world
    from ...evaluation.phase11_sampler import Phase11SamplerRequest, sample_belief_world
    from ...training.phase11_contract import BELIEF_SAMPLER_VERSION
    from .gate import _public_state
    from .systems import build_engine

    findings = []
    checked = {}
    for name in presets:
        bundle = build_engine(DEEP_PILOT_PAIRING, models, name)
        worlds_seen = 0
        for _row, state, _plan in states:
            public = _public_state(state)
            marginals = bundle.provider.predict_marginals(public)
            for offset in range(bundle.config.worlds):
                world = sample_belief_world(
                    Phase11SamplerRequest(
                        sampler_version=BELIEF_SAMPLER_VERSION,
                        public_state_document=public.public_state_document,
                        learned_probabilities=marginals,
                        sample_ordinal=offset,
                    )
                )
                report = validate_world(public.public_state_document, world)
                worlds_seen += 1
                if not report["valid"]:
                    findings.append(
                        f"{name}: a world failed the accepted validation stack: "
                        f"{report['findings'][:2]}"
                    )
                    break
        checked[name] = worlds_seen
    return {"passed": not findings, "findings": findings, "worlds_checked": checked}


# ---------------------------------------------------------------------------
# Idle latency
# ---------------------------------------------------------------------------


def latency_pilot(models, states, *, presets=DEEP_PILOT_PRESET_NAMES, seed: int = 20260824) -> dict:
    """One idle-machine latency run per rung, with measured forward counts."""
    from .systems import build_engine

    profiles = {}
    for name in presets:
        bundle = build_engine(DEEP_PILOT_PAIRING, models, name)
        timings, forwards, unique = [], [], []
        for _row, state, _plan in states:
            started = time.perf_counter()
            decision = bundle.engine.choose_action(state, seed=seed)
            timings.append(time.perf_counter() - started)
            forwards.append(decision.c1_forwards)
            unique.append(decision.unique_worlds / decision.worlds_requested)
        array = np.asarray(timings, dtype=np.float64)
        profiles[name] = {
            "preset_id": name,
            "decisions": len(timings),
            "mean_seconds_per_move": round(float(array.mean()), 5),
            "median_seconds_per_move": round(float(np.median(array)), 5),
            "p95_seconds_per_move": round(float(np.percentile(array, 95)), 5),
            "max_seconds_per_move": round(float(array.max()), 5),
            "mean_c1_forwards": round(float(np.mean(forwards)), 1),
            "mean_world_uniqueness": round(float(np.mean(unique)), 4),
        }
    baseline = profiles[BASELINE_PRESET]
    for name, entry in profiles.items():
        entry["measured_forward_ratio_vs_medium"] = round(
            entry["mean_c1_forwards"] / baseline["mean_c1_forwards"], 3
        )
        entry["measured_latency_ratio_vs_medium"] = round(
            entry["median_seconds_per_move"] / baseline["median_seconds_per_move"], 3
        )
        entry["naive_ratio_vs_medium"] = round(
            naive_compute_units(preset_of(name))
            / naive_compute_units(preset_of(BASELINE_PRESET)),
            3,
        )
    return profiles


# ---------------------------------------------------------------------------
# Decision divergence
# ---------------------------------------------------------------------------


def decision_divergence(
    models, states, *, presets=DEEP_PILOT_PRESET_NAMES, seed: int = 20260824
) -> dict:
    """How often each rung picks a different move than MEDIUM, position by position.

    The comparable measure: every rung answers the same fixed question, so a
    difference is a difference of decision procedures rather than of positions
    reached.
    """
    from .systems import build_engine

    actions: dict[str, list] = {}
    for name in presets:
        bundle = build_engine(DEEP_PILOT_PAIRING, models, name)
        actions[name] = [
            int(bundle.engine.choose_action(state, seed=seed).selected_action_id)
            for _row, state, _plan in states
        ]
    baseline = actions[BASELINE_PRESET]
    report = {}
    for name, chosen in actions.items():
        differing = sum(1 for left, right in zip(chosen, baseline) if left != right)
        report[name] = {
            "positions": len(chosen),
            "moves_differing_from_medium": differing,
            "fraction_differing_from_medium": round(differing / len(chosen), 5)
            if chosen
            else None,
        }
    return report


def first_divergence(rows_by_preset: dict) -> dict:
    """Where each rung's match games first part company with MEDIUM's.

    A game-level companion to :func:`decision_divergence`: it says how deep
    into a game the extra search takes before it changes anything, which the
    position-level number cannot.
    """
    baseline = {
        row["board_id"]: row.get("actions") or []
        for row in rows_by_preset.get(BASELINE_PRESET, [])
    }
    report = {}
    for name, rows in rows_by_preset.items():
        if name == BASELINE_PRESET:
            continue
        plies, identical = [], 0
        for row in rows:
            reference = baseline.get(row["board_id"])
            if not reference:
                continue
            mine = row.get("actions") or []
            index = 0
            while index < min(len(mine), len(reference)) and mine[index] == reference[index]:
                index += 1
            if index == len(mine) == len(reference):
                identical += 1
            else:
                plies.append(index)
        report[name] = {
            "games_compared": len(plies) + identical,
            "games_identical_to_medium": identical,
            "games_that_diverged": len(plies),
            "median_first_divergence_ply": (
                int(np.median(plies)) if plies else None
            ),
            "min_first_divergence_ply": int(min(plies)) if plies else None,
        }
    return report


# ---------------------------------------------------------------------------
# Reading the pack
# ---------------------------------------------------------------------------


def analyse_rungs(entries: "list[dict]", idle_latency: dict, divergence: dict) -> dict:
    """Every quantity the pilot was asked to report, per rung."""
    from .analysis import arm_summary, paired_delta

    by_preset: dict[str, list] = {}
    seconds_by_preset: dict[str, dict] = {}
    fallbacks_by_preset: dict[str, dict] = {}
    for entry in entries:
        row = entry["row"]
        by_preset.setdefault(row["preset_id"], []).append(row)
        seconds_by_preset.setdefault(row["preset_id"], {})[row["board_id"]] = (
            entry.get("move_seconds") or []
        )
        for reason, count in (entry.get("fallback_reasons") or {}).items():
            bucket = fallbacks_by_preset.setdefault(row["preset_id"], {})
            bucket[reason] = bucket.get(reason, 0) + int(count)

    baseline_rows = by_preset.get(BASELINE_PRESET, [])
    rungs = {}
    for name, rows in by_preset.items():
        summary = arm_summary(rows, seconds_by_preset.get(name, {}))
        rungs[name] = {
            "preset_id": name,
            "games": summary["games"],
            "wins": summary["wins"],
            "draws": summary["draws"],
            "losses": summary["losses"],
            "ewr": summary["ewr"],
            "paired_vs_medium": (
                paired_delta(rows, baseline_rows) if name != BASELINE_PRESET else None
            ),
            "worst_opponent": summary["min_opponent"],
            "ewr_by_opponent": summary["ewr_by_opponent"],
            "worst_family": summary["min_family"],
            "weakness_pack_family_ewr": summary["weakness_pack_family_ewr"],
            "pack_latency": {
                "mean_seconds_per_move": summary.get("mean_seconds_per_move"),
                "median_seconds_per_move": summary.get("median_seconds_per_move"),
                "p95_seconds_per_move": summary.get("p95_seconds_per_move"),
                "max_seconds_per_move": summary.get("max_seconds_per_move"),
                "note": "measured under ten-way process contention",
            },
            "idle_latency": idle_latency.get(name, {}),
            "moves_differing_from_medium": divergence.get(name, {}),
            "fallbacks": summary["fallbacks"],
            "fallback_rate": summary["fallback_rate"],
            "fallback_reasons": fallbacks_by_preset.get(name, {}),
            "player_decisions": summary["player_decisions"],
            "search_seconds_per_game": summary["search_seconds_per_game"],
            "mean_c1_forwards_per_move": summary.get("mean_c1_forwards_per_move"),
            "mean_plies": summary["mean_plies"],
        }
    return rungs


def check_medium_reproduces(fresh: "list[dict]", stage_c: "list[dict]") -> dict:
    """The freshly played MEDIUM rows must equal Stage C's, board for board.

    A cross-run determinism proof that costs nothing extra: the same boards,
    the same seeds and the same frozen weights were played twice, hours apart,
    by two different pack invocations. Anything but exact agreement on outcome
    and ply count would mean some input was not as fixed as it is documented
    to be.
    """
    left = {row["board_id"]: row for row in fresh}
    right = {row["board_id"]: row for row in stage_c}
    shared = sorted(set(left) & set(right))
    findings = []
    for board in shared:
        for field in ("effective_score", "outcome", "plies", "player_decisions"):
            if left[board][field] != right[board][field]:
                findings.append(
                    f"{board}: {field} {left[board][field]!r} != Stage C's "
                    f"{right[board][field]!r}"
                )
    return {
        "passed": not findings and bool(shared),
        "boards_compared": len(shared),
        "findings": findings[:10],
        "note": (
            "the same 60 boards played twice, in two separate pack runs, with the "
            "same seeds and the same frozen weights"
        ),
    }


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepVerdict:
    """The pilot's answer, and the reason for it."""

    recommendation: str
    reason: str
    detail: dict

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "reason": self.reason,
            **self.detail,
        }


def decide(rungs: dict, *, low: float = DEEP_MEANINGFUL_GAIN_LOW, high: float = DEEP_MEANINGFUL_GAIN_HIGH) -> DeepVerdict:
    """Apply the pilot's decision rule to the measured rungs.

    ```text
    LARGE or XLARGE gains >= ~0.03-0.05 without absurd latency -> recommend it
    gain tiny or noisy while latency multiplies                -> keep MEDIUM
    LARGE improves but XLARGE does not                         -> choose LARGE
    ```

    "Absurd latency" is read against the 5 s ceiling the working player already
    enforces: a rung whose p95 move cannot fit inside the cap is not a mode
    that could be shipped, whatever it scores.
    """
    from .contract import ACCEPTABLE_MOVE_SECONDS

    large = rungs.get("LARGE", {})
    xlarge = rungs.get("XLARGE", {})

    def gain(entry):
        return (entry.get("paired_vs_medium") or {}).get("delta")

    def standard_error(entry):
        return (entry.get("paired_vs_medium") or {}).get("standard_error")

    def shippable(entry):
        p95 = (entry.get("idle_latency") or {}).get("p95_seconds_per_move")
        return p95 is not None and p95 <= ACCEPTABLE_MOVE_SECONDS

    large_gain, xlarge_gain = gain(large), gain(xlarge)
    detail = {
        "large_gain": large_gain,
        "large_standard_error": standard_error(large),
        "large_p95_seconds": (large.get("idle_latency") or {}).get("p95_seconds_per_move"),
        "large_fits_latency_ceiling": shippable(large),
        "xlarge_gain": xlarge_gain,
        "xlarge_standard_error": standard_error(xlarge),
        "xlarge_p95_seconds": (xlarge.get("idle_latency") or {}).get("p95_seconds_per_move"),
        "xlarge_fits_latency_ceiling": shippable(xlarge),
        "meaningful_gain_band": [low, high],
        "latency_ceiling_seconds": ACCEPTABLE_MOVE_SECONDS,
        "rule": (
            "recommend the stronger rung when it gains >= the meaningful band and "
            "its p95 move fits the ceiling; keep MEDIUM when the gain is tiny or "
            "noisy while latency multiplies; prefer LARGE when LARGE improves and "
            "XLARGE does not"
        ),
    }

    def meaningful(value):
        return value is not None and value >= low

    large_ok = meaningful(large_gain) and shippable(large)
    xlarge_ok = meaningful(xlarge_gain) and shippable(xlarge)

    if xlarge_ok and large_ok:
        if (xlarge_gain - large_gain) >= low:
            return DeepVerdict(
                "XLARGE",
                "both rungs clear the band and XLARGE adds a further meaningful gain "
                "over LARGE while still fitting the latency ceiling",
                detail,
            )
        return DeepVerdict(
            "LARGE",
            "both rungs clear the band but XLARGE adds nothing meaningful over "
            "LARGE, so the cheaper of the two is the one worth shipping",
            detail,
        )
    if large_ok and not xlarge_ok:
        return DeepVerdict(
            "LARGE",
            "LARGE clears the meaningful band and fits the latency ceiling; XLARGE "
            "does not (either its gain is not meaningful or its p95 move cannot fit "
            "inside the cap)",
            detail,
        )
    if xlarge_ok and not large_ok:
        return DeepVerdict(
            "XLARGE",
            "XLARGE clears the meaningful band and fits the latency ceiling while "
            "LARGE does not",
            detail,
        )
    # "No meaningful gain" and "measurably worse" are different answers and
    # must not be reported as the same one. When both point estimates are
    # negative the pilot has not merely failed to find a gain — it has found
    # that the extra compute costs strength.
    regressions = [value for value in (large_gain, xlarge_gain) if value is not None and value < 0]
    if regressions and len(regressions) == len([v for v in (large_gain, xlarge_gain) if v is not None]):
        detail["both_rungs_regressed"] = True
        return DeepVerdict(
            "MEDIUM",
            "both stronger rungs came out *worse* than MEDIUM on the paired pack "
            f"(LARGE {large_gain:+.4f}, XLARGE {xlarge_gain:+.4f}), so the extra "
            "compute does not merely fail to pay for itself — the point estimate "
            "is a regression, and MEDIUM stands",
            detail,
        )
    detail["both_rungs_regressed"] = False
    return DeepVerdict(
        "MEDIUM",
        "neither rung buys a gain at or above the meaningful band within the "
        "latency ceiling, so the extra compute is not worth spending and MEDIUM "
        "stands",
        detail,
    )


__all__ = [
    "BASELINE_PRESET",
    "DEEP_PILOT_VERSION",
    "DeepVerdict",
    "Phase15DeepError",
    "check_configuration_invariants",
    "check_determinism",
    "check_frozen_identity",
    "analyse_rungs",
    "check_medium_reproduces",
    "check_worlds_legal",
    "decide",
    "decision_divergence",
    "first_divergence",
    "latency_pilot",
]
