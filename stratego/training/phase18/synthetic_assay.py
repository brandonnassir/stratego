"""Phase 18 Gate G2: the three-seed synthetic known-reward learning assay.

One seed, once
--------------
```text
model    = fresh 802,320-parameter setup model from model_seed(namespace, k)
trainer  = AdamW(lr 5e-5, wd 0) + EMA(0.999); buffer(storage_duration)
evaluate the EMA at 0 updates on the held-out evaluation stream  -> U_initial
for update u in 1..U:
    pool   = generate_pool(RAW model, 1,024, snapshot u-1)     # forced handedness, seeded reflection,
                                                              # orientation-gated per lane
    buffer.add_pool(pool, period u)                            # de-dup, counts reset
    for each pooled setup with an opening move:
        outcomes = landscape.outcomes_for(..., replicates)     # seeded W/D/L only
        buffer.add_outcome(fingerprint, z) for each             # running mean per exact setup
    trainer.update(buffer, u)                                   # 5 epochs x ceil(ready/1,024) steps; EMA once
    buffer.filter(u)                                            # retention window
evaluate the EMA after update U on the SAME evaluation stream   -> U_final
save raw / optimizer / EMA as three files
```

The evaluation stream uses common random numbers: every endpoint draws the
same 4,096 per-token uniforms, so `U_final[i] - U_initial[i]` is a paired
difference at sample `i`. The decision reads only the initial and final
endpoints; intermediate curve points are telemetry.

The landscape's utility never reaches the learner: the only landscape method
called inside the update loop is `outcomes_for`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from .setup_buffer import SetupBuffer
from .setup_contract import (
    SETUP_BATCH_SIZE,
    SETUP_EPOCHS_PER_UPDATE,
    SETUP_POOL_SIZE,
    Phase18SetupError,
    SetupTrainingConfig,
    json_document_digest,
    model_seed,
    stream_seed,
)
from .setup_learning import SetupTrainer
from .setup_model import build_setup_model, state_dict_digest
from .setup_sampling import generate_pool
from .synthetic_landscape import SyntheticLandscape

ASSAY_VERSION = "phase18_g2_synthetic_assay_v1"


@dataclass(frozen=True)
class AssayDesign:
    """Everything the assay freezes before the first optimizer step."""

    namespace: str = "phase18_g2_setup_parity_v1"
    run_id: str = "G2-SYNTHETIC-ASSAY-2026-A"
    seed_indices: tuple = (1, 2, 3)
    updates: int = 64
    pool_size: int = SETUP_POOL_SIZE
    outcomes_per_setup: int = 4
    storage_duration: int = 1
    epochs_per_update: int = SETUP_EPOCHS_PER_UPDATE
    batch_size: int = SETUP_BATCH_SIZE
    evaluation_samples: int = 4096
    curve_every: int = 8
    bootstrap_replicates: int = 10_000
    bootstrap_confidence: float = 0.95
    gap_closure_threshold: float = 0.10
    landscape_kappa: float = 3.0
    landscape_p_draw: float = 0.10
    device: str = "cpu"
    threads: int = 4
    #: Tests may run a reduced design; the frozen assay must have `reduced=False`,
    #: which enforces every minimum of the instruction.
    reduced: bool = False

    def __post_init__(self) -> None:
        if self.updates < 1 or self.updates > 64:
            raise Phase18SetupError("the update budget is 1..64")
        if self.outcomes_per_setup < 4:
            raise Phase18SetupError("at least four outcomes per eligible setup")
        if self.reduced:
            return
        if len(self.seed_indices) < 3:
            raise Phase18SetupError("the assay needs at least three seeds")
        if self.pool_size < SETUP_POOL_SIZE or self.batch_size != SETUP_BATCH_SIZE:
            raise Phase18SetupError("1,024 ready setups per update and a 1,024-setup optimizer batch")
        if self.epochs_per_update != SETUP_EPOCHS_PER_UPDATE:
            raise Phase18SetupError("five epochs per update")
        if self.evaluation_samples < 4096:
            raise Phase18SetupError("at least 4,096 held-out EMA setups per endpoint")
        if self.bootstrap_replicates < 10_000:
            raise Phase18SetupError("10,000 bootstrap replicates")

    def landscape_table_seed(self) -> int:
        return stream_seed(self.namespace, "landscape_table")

    def bootstrap_seed(self) -> int:
        return stream_seed(self.namespace, "paired_bootstrap")

    def model_seed(self, seed_index: int) -> int:
        return model_seed(self.namespace, seed_index)

    def training_config(self) -> SetupTrainingConfig:
        return SetupTrainingConfig(
            run_id=self.run_id,
            device=self.device,
            epochs_per_update=self.epochs_per_update,
            batch_size=self.batch_size,
            pool_size=self.pool_size,
        )

    def document(self) -> dict:
        return {
            "assay_version": ASSAY_VERSION,
            **asdict(self),
            "seed_indices": list(self.seed_indices),
            "model_seeds": {str(k): self.model_seed(k) for k in self.seed_indices},
            "landscape_table_seed": self.landscape_table_seed(),
            "bootstrap_seed": self.bootstrap_seed(),
            "seed_function": "stratego.setups.identity.derive_stream_seed",
            "seed_streams": {
                "model_init": "derive_stream_seed(namespace, 'model_init', k)",
                "pool_tokens": "derive_stream_seed(namespace, 'pool', k, snapshot, index) -> per-prefix derive_stream_seed('phase18_setup_token', root, prefix)",
                "pool_reflection": "derive_stream_seed(namespace, 'reflection', k, snapshot, index)   (independent of the token stream)",
                "outcomes": "derive_stream_seed(namespace, 'outcome', k, period, content_fingerprint, replicate)",
                "shuffle": "derive_stream_seed(namespace, 'shuffle', k, update, epoch)",
                "evaluation_tokens": "derive_stream_seed(namespace, 'eval', k, 0, index) -> per-prefix token seeds; snapshot fixed at 0 so every endpoint shares the same uniforms (common random numbers)",
                "evaluation_reflection": "derive_stream_seed(namespace, 'eval_reflection', k, 0, index)",
                "landscape_table": "derive_stream_seed(namespace, 'landscape_table')",
                "paired_bootstrap": "derive_stream_seed(namespace, 'paired_bootstrap')",
            },
            "training_config_digest": self.training_config().config_digest(),
            "checkpoint_rule": "the decision reads the EMA after the final fixed update only; intermediate curve points are telemetry and never select a checkpoint",
            "evaluation_rule": "expected utility = mean landscape utility over the held-out EMA samples; the same evaluation stream at every endpoint gives paired per-sample differences",
        }


def evaluate_ema(trainer: SetupTrainer, landscape: SyntheticLandscape, design: AssayDesign, seed_index: int, label: str) -> dict:
    """Score the EMA model on the held-out evaluation stream (the decision model)."""
    return evaluate_policy(trainer.evaluation_model(device=design.device), landscape, design, seed_index, label, ema_updates=trainer.ema.updates)


def evaluate_raw(trainer: SetupTrainer, landscape: SyntheticLandscape, design: AssayDesign, seed_index: int, label: str) -> dict:
    """DIAGNOSTIC ONLY: score the RAW model on the same held-out stream.

    The raw model is the generation actor, never the evaluation model (S28);
    its held-out utility is recorded so the learner's own trajectory can be
    read beside the EMA's, and it decides nothing.
    """
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        return evaluate_policy(trainer.model, landscape, design, seed_index, label, ema_updates=None)
    finally:
        trainer.model.train(was_training)


def evaluate_policy(policy, landscape: SyntheticLandscape, design: AssayDesign, seed_index: int, label: str, *, ema_updates) -> dict:
    """Score one policy on the held-out evaluation stream (common random numbers)."""
    ema_model = policy
    digest = state_dict_digest(ema_model)
    generation = generate_pool(
        ema_model,
        namespace=design.namespace,
        seed_index=seed_index,
        snapshot_iteration=0,
        snapshot_digest=digest,
        count=design.evaluation_samples,
        purpose="eval",
        device=design.device,
    )
    boards = np.array([s.played_canonical for s in generation.samples], dtype=np.int64)
    values = landscape.utilities(boards)
    expected_z = landscape.expected_z_outcome(boards)
    fingerprints = [s.content_fingerprint for s in generation.samples]
    classes = {s.class_fingerprint for s in generation.samples}
    return {
        "label": label,
        "ema_digest": digest,
        "ema_updates": ema_updates,
        "samples": int(len(generation.samples)),
        "mean_utility": float(values.mean()),
        "sd_utility": float(values.std(ddof=1)),
        "mean_z": float(((values - landscape.uniform_mean) / landscape.uniform_sd).mean()),
        "mean_expected_outcome": float(expected_z.mean()),
        "utilities": values.astype(np.float64),
        "distinct_content_fingerprints": len(set(fingerprints)),
        "distinct_class_fingerprints": len(classes),
        "generation_telemetry": generation.telemetry,
    }


def run_seed(design: AssayDesign, landscape: SyntheticLandscape, seed_index: int, output: Path, *, log=print) -> dict:
    """Run one frozen seed of the assay and write its artifacts under `output`."""
    torch.set_num_threads(int(design.threads))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = design.training_config()
    started = time.perf_counter()

    model = build_setup_model(device=design.device, seed=design.model_seed(seed_index))
    initial_digest = state_dict_digest(model)
    trainer = SetupTrainer(model, config, namespace=design.namespace, seed_index=seed_index)
    buffer = SetupBuffer(storage_duration=design.storage_duration, device=design.device)
    integrity = {
        "legality_failures": 0,
        "orientation_failures": 0,
        "attribution_failures": 0,
        "non_finite_events": 0,
        "checkpoint_identity_failures": 0,
        "immediately_terminal_setups": 0,
        "duplicates_collapsed": 0,
    }

    initial = evaluate_ema(trainer, landscape, design, seed_index, "initial")
    np.save(output / "utilities_initial.npy", initial["utilities"])
    initial_raw = evaluate_raw(trainer, landscape, design, seed_index, "initial_raw")
    np.save(output / "utilities_initial_raw.npy", initial_raw["utilities"])
    log(f"seed {seed_index}: initial EMA utility {initial['mean_utility']:.4f} (z {initial['mean_z']:+.3f})")
    curve = [{"update": 0, "mean_utility": initial["mean_utility"], "mean_z": initial["mean_z"], "ema_digest": initial["ema_digest"],
              "raw_mean_utility": initial_raw["mean_utility"], "raw_mean_z": initial_raw["mean_z"]}]

    receipts_path = output / "outcome_receipts.jsonl"
    telemetry_path = output / "telemetry.jsonl"
    period_digests = []
    with open(receipts_path, "w") as receipts, open(telemetry_path, "w") as telemetry:
        for update in range(1, design.updates + 1):
            tick = time.perf_counter()
            actor = trainer.generation_actor
            snapshot_digest = state_dict_digest(actor)
            generation = generate_pool(
                actor,
                namespace=design.namespace,
                seed_index=seed_index,
                snapshot_iteration=update - 1,
                snapshot_digest=snapshot_digest,
                count=design.pool_size,
                force_handedness=config.force_handedness,
                reflection_probability=config.reflection_probability,
                device=design.device,
            )
            integrity["legality_failures"] += generation.telemetry["legality_failures"]
            integrity["orientation_failures"] += generation.telemetry["orientation_failures"]
            integrity["immediately_terminal_setups"] += generation.telemetry["immediately_terminal_count"]
            pool_record = buffer.add_pool(generation.samples, period=update)
            integrity["duplicates_collapsed"] += pool_record["duplicates_collapsed"]

            period_rows = []
            outcomes_added = 0
            for sample in generation.samples:
                if not sample.opening_move:
                    continue  # S24: filtered from play; never trained, never a draw
                outcomes = landscape.outcomes_for(
                    sample.played_canonical,
                    seed_index=seed_index,
                    period=update,
                    fingerprint=sample.content_fingerprint,
                    replicates=design.outcomes_per_setup,
                )
                outcomes_added += buffer.add_outcomes((sample.content_fingerprint, z) for z in outcomes)
                period_rows.append(
                    {
                        "period": update,
                        "index": sample.index,
                        "fingerprint": sample.content_fingerprint,
                        "class": sample.class_fingerprint,
                        "reflected": sample.reflected,
                        "lane": sample.lane,
                        "snapshot": snapshot_digest,
                        "outcomes": "".join({-1: "-", 0: "0", 1: "+"}[z] for z in outcomes),
                    }
                )
            for row in period_rows:
                receipts.write(json.dumps(row, sort_keys=True) + "\n")
            period_digest = json_document_digest([[r["fingerprint"], r["outcomes"]] for r in period_rows])
            period_digests.append(period_digest)

            result = trainer.update(buffer, global_iteration=update)
            integrity["non_finite_events"] += result.non_finite_events
            retained = buffer.filter(update)
            integrity["attribution_failures"] += buffer.attribution_failures

            # DIAGNOSTIC: the landscape utility of the pool the RAW actor just
            # played. Computed after the update from the played boards; it is
            # never handed to the learner.
            pool_boards = np.array([s.played_canonical for s in generation.samples], dtype=np.int64)
            pool_utilities = landscape.utilities(pool_boards)
            row = {
                "seed_index": seed_index,
                "update": update,
                "seconds": round(time.perf_counter() - tick, 3),
                "pool_mean_utility_diagnostic": float(pool_utilities.mean()),
                "pool_mean_z_diagnostic": float(((pool_utilities - landscape.uniform_mean) / landscape.uniform_sd).mean()),
                "pool": generation.telemetry,
                "pool_record": pool_record,
                "outcomes_added": outcomes_added,
                "period_outcome_digest": period_digest,
                "update_result": result.document(),
                "buffer_after_filter": retained,
            }
            telemetry.write(json.dumps(row, sort_keys=True) + "\n")
            if update % design.curve_every == 0 and update != design.updates:
                point = evaluate_ema(trainer, landscape, design, seed_index, f"curve_{update}")
                raw_point = evaluate_raw(trainer, landscape, design, seed_index, f"curve_{update}_raw")
                curve.append({"update": update, "mean_utility": point["mean_utility"], "mean_z": point["mean_z"], "ema_digest": point["ema_digest"],
                              "raw_mean_utility": raw_point["mean_utility"], "raw_mean_z": raw_point["mean_z"]})
                log(f"seed {seed_index}: update {update}/{design.updates} EMA utility {point['mean_utility']:.4f} (z {point['mean_z']:+.3f}); raw {raw_point['mean_utility']:.4f} (z {raw_point['mean_z']:+.3f})")
            elif update % 4 == 0:
                log(f"seed {seed_index}: update {update}/{design.updates} done in {row['seconds']} s")

    final = evaluate_ema(trainer, landscape, design, seed_index, "final")
    np.save(output / "utilities_final.npy", final["utilities"])
    final_raw = evaluate_raw(trainer, landscape, design, seed_index, "final_raw")
    np.save(output / "utilities_final_raw.npy", final_raw["utilities"])
    curve.append({"update": design.updates, "mean_utility": final["mean_utility"], "mean_z": final["mean_z"], "ema_digest": final["ema_digest"],
                  "raw_mean_utility": final_raw["mean_utility"], "raw_mean_z": final_raw["mean_z"]})
    log(f"seed {seed_index}: final EMA utility {final['mean_utility']:.4f} (z {final['mean_z']:+.3f}); raw {final_raw['mean_utility']:.4f} (z {final_raw['mean_z']:+.3f})")

    manifest = trainer.save_checkpoint(output / "checkpoint_final")
    reloaded, _ = SetupTrainer.load_checkpoint(
        output / "checkpoint_final", config, namespace=design.namespace, seed_index=seed_index, device=design.device
    )
    if state_dict_digest(reloaded.model) != state_dict_digest(trainer.model):
        integrity["checkpoint_identity_failures"] += 1
    if state_dict_digest(reloaded.ema.as_model(device="cpu")) != final["ema_digest"]:
        integrity["checkpoint_identity_failures"] += 1

    differences = final["utilities"] - initial["utilities"]
    raw_differences = final_raw["utilities"] - initial_raw["utilities"]
    gap = landscape.optimum - initial["mean_utility"]
    raw_gap = landscape.optimum - initial_raw["mean_utility"]
    record = {
        "assay_version": ASSAY_VERSION,
        "namespace": design.namespace,
        "run_id": design.run_id,
        "seed_index": seed_index,
        "model_seed": design.model_seed(seed_index),
        "initial_raw_digest": initial_digest,
        "updates": design.updates,
        "optimizer_steps": trainer.optimizer_step_count,
        "ema_updates": trainer.ema.updates,
        "initial": {k: v for k, v in initial.items() if k != "utilities"},
        "final": {k: v for k, v in final.items() if k != "utilities"},
        "curve": curve,
        "paired": {
            "samples": int(differences.size),
            "mean_difference": float(differences.mean()),
            "sd_difference": float(differences.std(ddof=1)),
            "fraction_improved": float((differences > 0).mean()),
        },
        "gap": {
            "exact_optimum": landscape.optimum,
            "initial_to_optimum": float(gap),
            "closed": float(final["mean_utility"] - initial["mean_utility"]),
            "fraction_closed": float((final["mean_utility"] - initial["mean_utility"]) / gap) if gap > 0 else None,
            "ema_retained_initial_fraction": float(0.999 ** design.updates),
        },
        "raw_diagnostic": {
            "role": "DIAGNOSTIC ONLY - the raw generation actor's held-out utility on the same evaluation stream; it never decides the gate",
            "initial": {k: v for k, v in initial_raw.items() if k != "utilities"},
            "final": {k: v for k, v in final_raw.items() if k != "utilities"},
            "paired": {
                "samples": int(raw_differences.size),
                "mean_difference": float(raw_differences.mean()),
                "sd_difference": float(raw_differences.std(ddof=1)),
                "fraction_improved": float((raw_differences > 0).mean()),
            },
            "gap": {
                "initial_to_optimum": float(raw_gap),
                "closed": float(final_raw["mean_utility"] - initial_raw["mean_utility"]),
                "fraction_closed": float((final_raw["mean_utility"] - initial_raw["mean_utility"]) / raw_gap) if raw_gap > 0 else None,
            },
        },
        "integrity": integrity,
        "period_outcome_digests": period_digests,
        "outcome_receipts": {"path": str(receipts_path), "sha256": _file_sha256(receipts_path)},
        "telemetry": {"path": str(telemetry_path), "sha256": _file_sha256(telemetry_path)},
        "checkpoint": manifest,
        "utilities": {
            "initial": {"path": str(output / "utilities_initial.npy"), "sha256": _file_sha256(output / "utilities_initial.npy")},
            "final": {"path": str(output / "utilities_final.npy"), "sha256": _file_sha256(output / "utilities_final.npy")},
            "initial_raw": {"path": str(output / "utilities_initial_raw.npy"), "sha256": _file_sha256(output / "utilities_initial_raw.npy")},
            "final_raw": {"path": str(output / "utilities_final_raw.npy"), "sha256": _file_sha256(output / "utilities_final_raw.npy")},
        },
        "wall_seconds": round(time.perf_counter() - started, 3),
        "threads": int(design.threads),
        "device": design.device,
    }
    (output / "seed_result.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


def _file_sha256(path: Path) -> str:
    from .setup_contract import file_sha256

    return file_sha256(path)


__all__ = ["ASSAY_VERSION", "AssayDesign", "evaluate_ema", "evaluate_policy", "evaluate_raw", "run_seed"]
