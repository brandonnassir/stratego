#!/usr/bin/env python
"""Phase 12 Agent 1 runner: search core sanity checks and latency.

Specification source: `02_PHASE_12_AGENT_1_SEARCH_CORE.md` (sections 10-12).

Sanity work only. This runner:

1. loads the accepted Phase 9 C1 (digest-checked) as the move model;
2. builds all four belief providers, binding `agent1c` to the surviving
   checkpoint bytes recorded in the Phase 11B handoff;
3. plays a small set of fresh sanity positions from accepted setup sources;
4. runs every instructed sanity check;
5. measures TINY and SMALL latency per provider (plus one MEDIUM smoke
   decision — explicitly not a MEDIUM benchmark);
6. writes `reports/phase12/agent_01_report.md` and
   `reports/phase12/agent_01_summary.json`.

No match run. No new training. Nothing accepted is modified.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.belief.phase11b.features import load_frozen_c1  # noqa: E402
from stratego.belief.phase11b.interface import Phase11BPublicState  # noqa: E402
from stratego.engine.constants import EVALUATION_RULES  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.state import create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.evaluation.phase11_baselines import validate_world  # noqa: E402
from stratego.evaluation.phase11_public_state import (  # noqa: E402
    build_public_state_document,
    document_summary,
)
from stratego.evaluation.policy import build_policy_input, build_public_view  # noqa: E402
from stratego.model.policy_adapter import SeededCategoricalNeuralPolicy  # noqa: E402
from stratego.search.phase12 import (  # noqa: E402
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_REMAINING_COUNT,
    Phase12SearchEngine,
    Phase12SearchError,
    SEARCH_VERSION,
    build_belief_provider,
    search_preset,
)
from stratego.search.phase12.contract import SCORE_DEFINITION  # noqa: E402

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase12"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase12"
HANDOFF_PATH = REPOSITORY_ROOT / "reports" / "phase11b" / "phase12_handoff.json"

PROVIDER_ORDER = (
    PROVIDER_REMAINING_COUNT,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
)

#: `(plan index, target ply)` of each sanity position. Setups come from the
#: accepted setup sources through the Phase 11B dev-plan grammar; playouts
#: are fresh (seeded-categorical C1), so these positions belong to no
#: existing corpus, bank or soak.
POSITION_SPECS = ((0, 0), (0, 16), (1, 24), (1, 60), (2, 40), (3, 88))

SANITY_SEED = 2026082001


def log(message: str) -> None:
    print(message, flush=True)


def sanitize(value):
    """Make a result tree JSON-serializable."""
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [sanitize(item) for item in value.tolist()]
    return value


# ---------------------------------------------------------------------------
# Stage: models and providers
# ---------------------------------------------------------------------------


def load_move_model(handoff: dict, device: str):
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model, identity = load_frozen_c1(
        REPOSITORY_ROOT,
        CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt",
        device=device,
    )
    expected = handoff["accepted_phase9_checkpoint"]
    if identity["model_state_digest"] != expected["model_state_digest"]:
        raise Phase12SearchError("loaded Phase 9 state digest != handoff record")
    if identity["belief_head_digest"] != expected["belief_head_digest"]:
        raise Phase12SearchError("loaded belief-head digest != handoff record")
    log(
        f"  move model: accepted Phase 9 C1, {identity['parameters']:,} parameters, "
        f"state digest {identity['model_state_digest'][:12]}..., device {device}"
    )
    return model, identity


def build_providers(model, handoff: dict, device: str) -> dict:
    agent1c_record = handoff["agent1c_checkpoint"]
    providers = {
        PROVIDER_REMAINING_COUNT: build_belief_provider(
            PROVIDER_REMAINING_COUNT, production=True
        ),
        PROVIDER_ORIGINAL_PHASE11: build_belief_provider(
            PROVIDER_ORIGINAL_PHASE11, encoder=model, production=True, device=device
        ),
        PROVIDER_AGENT1C: build_belief_provider(
            PROVIDER_AGENT1C,
            encoder=model,
            agent1c_checkpoint=REPOSITORY_ROOT / agent1c_record["path"],
            expected_agent1c_sha256=agent1c_record["sha256"],
            expected_agent1c_state_digest=agent1c_record["state_dict_digest"],
            production=True,
            device=device,
        ),
        PROVIDER_ORACLE: build_belief_provider(PROVIDER_ORACLE, production=False),
    }
    for name, provider in providers.items():
        log(f"  provider ready: {name} (uses_hidden_truth={provider.uses_hidden_truth})")
    return providers


# ---------------------------------------------------------------------------
# Stage: sanity positions
# ---------------------------------------------------------------------------


def build_positions(model, device: str, specs) -> list:
    """Fresh deterministic positions from accepted setup sources."""
    from stratego.belief.phase11b.corpus import Phase11BSetupSources, corpus_plans

    sources = Phase11BSetupSources()
    plans = corpus_plans("dev", sources, limit=8)
    policy = SeededCategoricalNeuralPolicy(model, device=device)

    positions = []
    for plan_index, target_ply in specs:
        plan = plans[plan_index]
        state = create_game(
            plan.red_setup,
            plan.blue_setup,
            rules=EVALUATION_RULES,
            game_id=f"phase12-sanity-{plan.game_id}-ply{target_ply}",
        )
        while not state.terminal and state.total_moves < target_ply:
            request = build_policy_input(
                state,
                policy=policy.ref,
                policy_seed=SANITY_SEED + 100 * plan_index + state.acting_player,
                requirements=policy.requirements,
            )
            result = policy.decide_checked(request)
            apply_action(state, result.selected_action_id)
        if state.terminal:
            log(
                f"  position plan{plan_index}/ply{target_ply}: game ended early at "
                f"ply {state.total_moves}; skipped"
            )
            continue
        observer = state.acting_player
        observation = build_observation(state, observer)
        document = build_public_state_document(
            build_public_view(state, observer), observation
        )
        positions.append(
            {
                "label": f"plan{plan_index}_ply{state.total_moves}",
                "state": state,
                "public": Phase11BPublicState(document, observation),
                "document_summary": document_summary(document),
            }
        )
        log(
            f"  position {positions[-1]['label']}: "
            f"{positions[-1]['document_summary']['hidden_opponent_pieces']} hidden "
            f"pieces, {len(legal_actions(state))} legal actions"
        )
    if not positions:
        raise Phase12SearchError("no sanity positions survived; cannot proceed")
    return positions


# ---------------------------------------------------------------------------
# Stage: direct world-legality probe
# ---------------------------------------------------------------------------


def world_legality_probe(providers: dict, positions: list, worlds: int = 8) -> dict:
    """Re-validate sampled worlds through the accepted stack, explicitly.

    The neural providers' sampler already validates internally and the
    count provider validates in its adapter; this probe is independent
    evidence for the report, run on full world documents.
    """
    from stratego.evaluation.phase11_baselines import sample_world

    blocks = {}
    for name, provider in providers.items():
        checked = 0
        for position in positions:
            public = position["public"]
            document = public.public_state_document
            if name == PROVIDER_ORACLE:
                assignments = provider.sample_assignments_privileged(
                    position["state"], public, 1, SANITY_SEED
                )
                checked += len(assignments)  # inventory checks run inside
                continue
            if name == PROVIDER_REMAINING_COUNT:
                from stratego.search.phase12.contract import (
                    DOMAIN_COUNT_WORLDS,
                    derive_phase12_seed,
                )
                from stratego.training.phase11_seed import MAX_SAMPLE_ORDINAL_FORMAT

                start = derive_phase12_seed(
                    DOMAIN_COUNT_WORLDS, "world", SANITY_SEED
                ) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
                for offset in range(worlds):
                    world = sample_world(
                        document, (start + offset) % (MAX_SAMPLE_ORDINAL_FORMAT + 1)
                    )
                    check = validate_world(document, world)
                    if not check["valid"]:
                        raise Phase12SearchError(
                            f"{name} produced an invalid world: {check['findings'][:3]}"
                        )
                    checked += 1
                continue
            for world in provider.belief_model.sample_worlds(public, worlds, SANITY_SEED):
                check = validate_world(document, world)
                if not check["valid"]:
                    raise Phase12SearchError(
                        f"{name} produced an invalid world: {check['findings'][:3]}"
                    )
                checked += 1
        blocks[name] = {"worlds_checked": checked, "all_valid": True}
        log(f"  world probe: {name} -> {checked} worlds, all valid")
    return blocks


# ---------------------------------------------------------------------------
# Stage: the search matrix
# ---------------------------------------------------------------------------


def run_matrix(
    model,
    model_identity: dict,
    providers: dict,
    positions: list,
    presets,
    device: str,
) -> list:
    rows = []
    for provider_name in PROVIDER_ORDER:
        if provider_name not in providers:
            continue
        provider = providers[provider_name]
        for preset_name in presets:
            config = search_preset(
                preset_name, production=(provider_name != PROVIDER_ORACLE)
            )
            engine = Phase12SearchEngine(
                model,
                provider,
                config,
                device=device,
                model_identity=model_identity,
            )
            for position in positions:
                state = position["state"]
                decision = engine.choose_action(state, seed=SANITY_SEED)
                legal = set(legal_actions(state))
                if decision.selected_action_id not in legal:
                    raise Phase12SearchError("selected action is not legal")
                if not decision.candidates[0].is_direct:
                    raise Phase12SearchError("direct action missing from candidates")
                repeat = engine.choose_action(state, seed=SANITY_SEED)
                reproduced = (
                    repeat.selected_action_id == decision.selected_action_id
                    and [c.world_values for c in repeat.candidates]
                    == [c.world_values for c in decision.candidates]
                )
                row = decision.summary()
                row.update(
                    {
                        "position": position["label"],
                        "device": device,
                        "reproduced": bool(reproduced),
                        "repeat_seconds": repeat.seconds,
                        "mean_forward_batch": (
                            float(np.mean(decision.forward_batch_sizes))
                        ),
                    }
                )
                rows.append(row)
            log(
                f"  matrix: {provider_name:>17} {preset_name:<6} on {device}: "
                f"{len(positions)} positions x 2 runs done"
            )
    return rows


def summarize_matrix(rows: list) -> dict:
    groups: dict = {}
    for row in rows:
        key = (row["provider_id"], row["preset_id"], row["device"])
        groups.setdefault(key, []).append(row)
    summary = {}
    for (provider, preset, device), members in groups.items():
        seconds = [m["seconds"] for m in members] + [m["repeat_seconds"] for m in members]
        forwards = [m["c1_forwards"] for m in members]
        forward_seconds = [m["forward_seconds"] for m in members]
        observation_seconds = [m["observation_seconds"] for m in members]
        totals = [m["seconds"] for m in members]
        summary[f"{provider}|{preset}|{device}"] = {
            "provider": provider,
            "preset": preset,
            "device": device,
            "decisions_timed": len(seconds),
            "move_latency_seconds": {
                "mean": statistics.mean(seconds),
                "median": statistics.median(seconds),
                "min": min(seconds),
                "max": max(seconds),
            },
            "c1_forwards_per_move": {
                "mean": statistics.mean(forwards),
                "min": min(forwards),
                "max": max(forwards),
            },
            "forward_positions_per_second": (
                sum(forwards) / sum(totals) if sum(totals) else None
            ),
            "moves_per_minute": 60.0 / statistics.mean(seconds),
            "max_forward_batch": max(m["max_forward_batch"] for m in members),
            "mean_forward_batch": statistics.mean(
                m["mean_forward_batch"] for m in members
            ),
            "mean_unique_worlds": statistics.mean(m["unique_worlds"] for m in members),
            "move_change_rate_vs_direct": statistics.mean(
                1.0 if m["move_changed"] else 0.0 for m in members
            ),
            "seed_reproduced_all": all(m["reproduced"] for m in members),
            "time_fraction_forward": sum(forward_seconds) / sum(totals),
            "time_fraction_observation": sum(observation_seconds) / sum(totals),
            "time_fraction_other": 1.0
            - (sum(forward_seconds) + sum(observation_seconds)) / sum(totals),
        }
    return summary


# ---------------------------------------------------------------------------
# Stage: structural oracle rejection
# ---------------------------------------------------------------------------


def oracle_rejection_evidence(model, providers: dict) -> dict:
    factory_refused = False
    try:
        build_belief_provider(PROVIDER_ORACLE, production=True)
    except Phase12SearchError:
        factory_refused = True
    engine_refused = False
    try:
        Phase12SearchEngine(
            model, providers[PROVIDER_ORACLE], search_preset("TINY"), device="cpu"
        )
    except Phase12SearchError:
        engine_refused = True
    constructor_refused = False
    try:
        from stratego.search.phase12.providers import OracleBeliefProvider

        OracleBeliefProvider()
    except Phase12SearchError:
        constructor_refused = True
    evidence = {
        "factory_refuses_oracle_in_production": factory_refused,
        "production_engine_refuses_oracle_provider": engine_refused,
        "oracle_constructor_requires_offline_flag": constructor_refused,
        "oracle_unavailable_in_production": (
            factory_refused and engine_refused and constructor_refused
        ),
    }
    log(f"  oracle rejection: {evidence}")
    return evidence


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def format_latency_table(summary: dict, device: str) -> list:
    lines = [
        "| provider | preset | mean s/move | median | C1 fwd/move | fwd pos/s | max batch | unique worlds | move-change | reproduced |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(summary):
        block = summary[key]
        if block["device"] != device:
            continue
        lines.append(
            "| {provider} | {preset} | {mean:.3f} | {median:.3f} | {fwd:.0f} | "
            "{pps:.0f} | {batch:.0f} | {worlds:.1f} | {change:.2f} | {repro} |".format(
                provider=block["provider"],
                preset=block["preset"],
                mean=block["move_latency_seconds"]["mean"],
                median=block["move_latency_seconds"]["median"],
                fwd=block["c1_forwards_per_move"]["mean"],
                pps=block["forward_positions_per_second"],
                batch=block["max_forward_batch"],
                worlds=block["mean_unique_worlds"],
                change=block["move_change_rate_vs_direct"],
                repro="yes" if block["seed_reproduced_all"] else "NO",
            )
        )
    return lines


def write_report(summary: dict, path: Path) -> None:
    s = summary
    lines: list[str] = []
    add = lines.append
    add("# Phase 12 Agent 1 — Minimal Search Engine and Belief Providers")
    add("")
    add(f"Generated {s['generated_utc']} by `scripts/run_phase12_agent01.py`.")
    add("")
    add("Engineering artifact of the Phase 12 rapid search-engineering phase. "
        "Sanity work only: no match run, no strength claim, no MEDIUM benchmark.")
    add("")
    add("## 1. Search architecture")
    add("")
    add("```text")
    add(f"search version   {SEARCH_VERSION}")
    add("root             one accepted Phase 9 C1 forward -> direct action, priors")
    add("candidates       all legal actions if <= 8, else top 8 by Phase 9 policy")
    add("                 (the direct action is candidate 0 by construction)")
    add("worlds           K hidden worlds sampled once at the root by the provider,")
    add("                 duplicates evaluated once and weighted by multiplicity")
    add("materialization  root state cloned, hidden opponent ranks overwritten —")
    add("                 the accepted anti-leak permutation transformation; the")
    add("                 root's observation and legal actions are re-derived in")
    add("                 every world and required identical (runtime gate)")
    add("rollouts         accepted Phase 9 greedy policy for both sides, batched")
    add("                 across all live (candidate, world) sims at each ply")
    add("leaf             exact terminal result overrides; otherwise C1 value head")
    add(f"score            {SCORE_DEFINITION}")
    add("selection        highest score, ties to the lowest normalized action id")
    add("```")
    add("")
    config = s["search_config"]
    add(f"`beta = {config['beta']}`, `epsilon = {config['epsilon']}` — one fixed "
        "modest prior, no grid search. `rollout_depth` counts plies after the "
        "candidate action; rollouts/action/world = 1 (greedy rollouts are "
        "deterministic, so repeats would be identical).")
    add("")
    add("## 2. Model roles and identities")
    add("")
    add("```text")
    identity = s["move_model_identity"]
    add(f"policy/value/rollout/leaf  accepted Phase 9 C1  (state digest {identity['model_state_digest'][:16]}...)")
    agent1c = s["providers"]["agent1c"]["identity"]
    add(f"agent1c beliefs            {agent1c['candidate_id']}")
    add(f"                           checkpoint sha256 {agent1c['checkpoint_sha256'][:16]}... (surviving bytes)")
    original = s["providers"]["original_phase11"]["identity"]
    add(f"original_phase11 beliefs   accepted belief head (digest {original['belief_head_digest'][:16]}...)")
    add("remaining_count beliefs    remaining_count_belief_v1 / count_uniform_world_sampler_v1")
    add("oracle                     true hidden state, offline diagnostic only")
    add("```")
    add("")
    add("Agent 1C's policy/value heads are never consulted: the adapter reads "
        "only its belief logits, and the move model is a separately loaded, "
        "digest-checked accepted C1.")
    add("")
    add("## 3. Belief-provider interface")
    add("")
    add("```text")
    add("provider.sample_assignments(public_state, n, seed) -> n x {piece_slot: rank}")
    add("provider.predict_marginals(public_state)           -> {piece_slot: 12-vector}")
    add("```")
    add("")
    add("Non-oracle providers read a `Phase11BPublicState` (frozen public-state "
        "document + 127-channel observation) and structurally cannot see truth. "
        "The neural providers are the accepted Phase 11B adapter wrapped "
        "unchanged, so their worlds go through the accepted Phase 11 sampler "
        "mathematics by import. The oracle uses a separate privileged method, "
        "requires `offline_diagnostic=True` at construction, and is refused by "
        "the factory and the engine in production configurations.")
    add("")
    add("## 4. Sanity checks")
    add("")
    add("```text")
    for name, value in s["sanity_checks"].items():
        add(f"{name:<48} {value}")
    add("```")
    add("")
    add(f"Positions: {', '.join(p['label'] for p in s['positions'])} — fresh "
        "seeded-categorical C1 playouts from accepted setup sources (dev-plan "
        "grammar), plus the opening position.")
    add("")
    add("## 5. TINY / SMALL latency (device: " + s["primary_device"] + ")")
    add("")
    lines.extend(format_latency_table(s["matrix_summary"], s["primary_device"]))
    add("")
    add("Latency is end-to-end per decision (worlds, materialization, rollouts, "
        "scoring), averaged over the sanity positions and their repeat runs. "
        "`fwd pos/s` is C1 forward positions per second through the whole "
        "search stack; the oracle rows run one unique world, which is why they "
        "are cheap.")
    if s.get("mps_matrix_ran"):
        add("")
        add("### MPS probe (agent1c only)")
        add("")
        lines.extend(format_latency_table(s["matrix_summary"], "mps"))
        add("")
        add(s["mps_note"])
    add("")
    add("## 6. MEDIUM smoke decision (not a benchmark)")
    add("")
    medium = s["medium_smoke"]
    add("```text")
    add(f"provider {medium['provider_id']}  position {medium['position']}  "
        f"seconds {medium['seconds']:.3f}  c1_forwards {medium['c1_forwards']}  "
        f"unique_worlds {medium['unique_worlds']}  max_batch {medium['max_forward_batch']}")
    add("```")
    add("")
    add("## 7. Cost profile and main observed bottleneck")
    add("")
    add("```text")
    for key in sorted(s["matrix_summary"]):
        block = s["matrix_summary"][key]
        if block["device"] != s["primary_device"]:
            continue
        add(
            f"{block['provider']:>17} {block['preset']:<6} "
            f"forward {block['time_fraction_forward']:.2f}  "
            f"observation {block['time_fraction_observation']:.2f}  "
            f"other {block['time_fraction_other']:.2f}"
        )
    add("```")
    add("")
    add(s["bottleneck_note"])
    add("")
    add("## 8. Deliverables and status")
    add("")
    add("```text")
    add("stratego/search/phase12/{contract,providers,engine}.py")
    add("tests/search/test_phase12_{providers,engine}.py  (+ conftest)")
    add("reports/phase12/agent_01_report.md")
    add("reports/phase12/agent_01_summary.json")
    add("")
    add(f"phase11_final_classification = {s['history']['phase11_final_classification']}")
    add(f"phase11b_selection           = {s['history']['phase11b_selection']}")
    add(f"scientific_validation_status = {s['history']['scientific_validation_status']}")
    add(f"oracle_available_in_production = {s['history']['oracle_available_in_production']}")
    add("```")
    add("")
    add("Stop condition reached: the search core works and every sanity check "
        "passes. No large match run was started; Agent 2 (belief-to-decision "
        "diagnostic) is not launched automatically.")
    add("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="primary device (default cpu)")
    parser.add_argument("--skip-mps", action="store_true", help="skip the MPS probe")
    parser.add_argument("--quick", action="store_true", help="3 positions, TINY only")
    arguments = parser.parse_args()

    started = time.perf_counter()
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    handoff = json.loads(HANDOFF_PATH.read_text())

    log("[1/7] loading the accepted Phase 9 C1 move model ...")
    model, model_identity = load_move_model(handoff, arguments.device)

    log("[2/7] building the four belief providers ...")
    providers = build_providers(model, handoff, arguments.device)

    log("[3/7] building sanity positions ...")
    specs = POSITION_SPECS[:3] if arguments.quick else POSITION_SPECS
    positions = build_positions(model, arguments.device, specs)

    log("[4/7] world-legality probe ...")
    world_probe = world_legality_probe(providers, positions)

    log("[5/7] oracle rejection evidence ...")
    oracle_evidence = oracle_rejection_evidence(model, providers)

    log("[6/7] search matrix (TINY/SMALL x providers) ...")
    presets = ("TINY",) if arguments.quick else ("TINY", "SMALL")
    rows = run_matrix(
        model, model_identity, providers, positions, presets, arguments.device
    )

    mps_rows: list = []
    mps_note = "MPS probe skipped."
    mps_ran = False
    if not arguments.skip_mps and torch.backends.mps.is_available():
        log("  MPS probe: agent1c on TINY/SMALL ...")
        mps_model, mps_identity = load_move_model(handoff, "mps")
        mps_providers = {
            PROVIDER_AGENT1C: build_belief_provider(
                PROVIDER_AGENT1C,
                encoder=mps_model,
                agent1c_checkpoint=REPOSITORY_ROOT
                / handoff["agent1c_checkpoint"]["path"],
                expected_agent1c_sha256=handoff["agent1c_checkpoint"]["sha256"],
                expected_agent1c_state_digest=handoff["agent1c_checkpoint"][
                    "state_dict_digest"
                ],
                production=True,
                device="mps",
            )
        }
        mps_rows = run_matrix(
            mps_model, mps_identity, mps_providers, positions, presets, "mps"
        )
        reproduced = all(row["reproduced"] for row in mps_rows)
        mps_ran = True
        mps_note = (
            "MPS forward passes reproduced identical decisions on repeat runs."
            if reproduced
            else "MPS repeat runs did NOT reproduce identical decisions; CPU is "
            "the reproducibility device."
        )

    log("[7/7] MEDIUM smoke decision (single, not a benchmark) ...")
    medium_engine = Phase12SearchEngine(
        model,
        providers[PROVIDER_AGENT1C],
        search_preset("MEDIUM"),
        device=arguments.device,
        model_identity=model_identity,
    )
    medium_position = positions[min(2, len(positions) - 1)]
    medium_decision = medium_engine.choose_action(
        medium_position["state"], seed=SANITY_SEED
    )
    medium_smoke = medium_decision.summary()
    medium_smoke["position"] = medium_position["label"]

    all_rows = rows + mps_rows
    matrix_summary = summarize_matrix(all_rows)

    primary = {
        key: block
        for key, block in matrix_summary.items()
        if block["device"] == arguments.device
    }
    fractions_forward = statistics.mean(
        block["time_fraction_forward"] for block in primary.values()
    )
    fractions_observation = statistics.mean(
        block["time_fraction_observation"] for block in primary.values()
    )
    if fractions_forward >= max(fractions_observation, 1 - fractions_forward - fractions_observation):
        bottleneck = "c1_forward_passes"
    elif fractions_observation >= 1 - fractions_forward - fractions_observation:
        bottleneck = "python_observation_building"
    else:
        bottleneck = "python_engine_overhead"
    bottleneck_note = (
        f"Main observed bottleneck on {arguments.device}: {bottleneck} "
        f"(mean time fractions — forward {fractions_forward:.2f}, observation "
        f"building {fractions_observation:.2f}, other "
        f"{1 - fractions_forward - fractions_observation:.2f})."
    )

    sanity_checks = {
        "all_four_belief_providers_execute": sorted(
            {row["provider_id"] for row in rows}
        )
        == sorted(PROVIDER_ORDER),
        "all_selected_actions_legal": True,  # asserted inside run_matrix
        "all_sampled_worlds_legal": all(
            block["all_valid"] for block in world_probe.values()
        ),
        "direct_c1_action_always_among_candidates": True,  # asserted inside run_matrix
        "fixed_seeds_reproduce_the_search_action": all(
            row["reproduced"] for row in rows
        ),
        "oracle_unavailable_in_production": oracle_evidence[
            "oracle_unavailable_in_production"
        ],
        "tiny_latency_measured": any(row["preset_id"] == "TINY" for row in rows),
        "small_latency_measured": any(row["preset_id"] == "SMALL" for row in rows),
    }
    required = dict(sanity_checks)
    if arguments.quick:
        # Quick mode runs TINY only; SMALL is simply not requested.
        required.pop("small_latency_measured")
        sanity_checks["small_latency_measured"] = "skipped (--quick)"
    if not all(required.values()):
        raise Phase12SearchError(f"sanity checks failed: {required}")

    summary = {
        "artifact": "phase12_agent01_summary",
        "phase": "phase12",
        "agent": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_version": SEARCH_VERSION,
        "score_definition": SCORE_DEFINITION,
        "search_config": search_preset("TINY").describe(),
        "presets": {
            name: search_preset(name).describe() for name in ("TINY", "SMALL", "MEDIUM")
        },
        "move_model_identity": model_identity,
        "providers": {name: provider.describe() for name, provider in providers.items()},
        "positions": [
            {"label": p["label"], "document_summary": p["document_summary"]}
            for p in positions
        ],
        "sanity_checks": sanity_checks,
        "world_legality_probe": world_probe,
        "oracle_rejection": oracle_evidence,
        "matrix_rows": all_rows,
        "matrix_summary": matrix_summary,
        "medium_smoke": medium_smoke,
        "primary_device": arguments.device,
        "mps_matrix_ran": mps_ran,
        "mps_note": mps_note,
        "bottleneck": bottleneck,
        "bottleneck_note": bottleneck_note,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "history": {
            "phase11_final_classification": "FAIL",
            "phase12_authorized_by_phase11": False,
            "phase11b_selection": "Agent1C",
            "scientific_validation_status": "not performed",
            "oracle_available_in_production": False,
        },
        "seconds_total": round(time.perf_counter() - started, 3),
    }

    summary_path = REPORT_DIRECTORY / "agent_01_summary.json"
    summary_path.write_text(json.dumps(sanitize(summary), indent=1))
    report_path = REPORT_DIRECTORY / "agent_01_report.md"
    write_report(sanitize(summary), report_path)
    log(f"\nwrote {summary_path.relative_to(REPOSITORY_ROOT)}")
    log(f"wrote {report_path.relative_to(REPOSITORY_ROOT)}")
    log(f"total {summary['seconds_total']:.1f}s; all sanity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
