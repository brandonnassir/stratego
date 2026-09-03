"""Phase 18 Stage 6B: the joint period loop with the lineage switch
(G3-ENG-02) and the joint bundle cadence (G3-ENG-03).

One `LineageRunner` drives one lineage. Per period `p` (design 2.3):

```text
(1) pool       the raw setup actor samples the period's pool under the shared
               pool seeds (snapshot_iteration = p - 1 in BOTH lineages); the
               buffer adds it (S10 de-duplication, counts reset, S23)
(2) collect    every slot advances T plies from the pool; completed games
               attribute two outcomes and commit their trajectory to the live
               store; the G4 accounting identity is checked
(3) C1         K supervised updates on the canonical/live mixture through the
               accepted trainer, the live half drawn from the last
               `live_retention_periods` finalised periods
(4) setup      candidate only: one setup update (five epochs over the ready
               rows) and one EMA update; a period with no ready row records an
               explicit skip. Control: nothing; the raw and EMA digests are
               asserted equal to the recorded initial version
(5) filter     rows older than the buffer storage duration expire
(6) bundle     every `bundle_cadence_periods` periods and at the end
```

Both lineages run this code verbatim; `config.setup_updates_enabled` is the
only branch, and it enters no seed. Every period writes one JSON record with
the digests the matching check and the gates read.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..warmstart_checkpoint import CorpusIdentity
from ..warmstart_pilot import model_state_checksum
from .g3_buffer_state import buffer_state_digest, restore_buffer_state
from .g3_bundle import (
    C1_NAME,
    SETUP_DIRECTORY,
    load_collector_state,
    load_setup_trainer,
    read_manifest,
    verify_bundle,
    write_bundle,
)
from .g3_c1 import JointC1Trainer
from .g3_collector import PeriodCollector
from .g3_contract import (
    G3_DESIGN_COMMIT,
    G3_HARNESS_VERSION,
    LINEAGES,
    Phase18G3Error,
    PilotConfig,
)
from .g3_live_store import LiveRecordReader, compare_live_periods, discard_periods_after
from .setup_buffer import SetupBuffer
from .setup_learning import SetupTrainer
from .setup_model import build_setup_model, state_dict_digest
from .setup_sampling import generate_pool

PERIODS_NAME = "periods.jsonl"
C1_ROWS_NAME = "c1_rows.jsonl"
INIT_NAME = "init.json"
STATE_NAME = "run_state.json"
BUNDLES_DIRECTORY = "bundles"
LIVE_DIRECTORY = "live"
ARCHIVE_DIRECTORY = "archive"


def bundle_name(period: int) -> str:
    return f"bundle_{int(period):04d}"


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def pool_content_digest(samples) -> str:
    """Order-independent digest over the played boards of one pool."""
    import hashlib

    fingerprints = sorted(sample.content_fingerprint for sample in samples)
    return hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()


class LineageRunner:
    """One lineage of the pilot: state, the period loop and the bundle cadence."""

    def __init__(
        self,
        config: PilotConfig,
        *,
        run_root,
        corpus_root,
        corpus_identity: CorpusIdentity,
        value_prior=None,
        require_complete_split: bool = True,
        log=print,
    ) -> None:
        self.config = config
        self.run_root = Path(run_root)
        self.lineage_root = self.run_root / config.lineage
        self.live_root = self.lineage_root / LIVE_DIRECTORY
        self.bundles_root = self.lineage_root / BUNDLES_DIRECTORY
        self.corpus_root = Path(corpus_root)
        self.corpus_identity = corpus_identity
        self.value_prior = value_prior
        self.require_complete_split = bool(require_complete_split)
        self.log = log
        self.period = 0
        self.c1: JointC1Trainer | None = None
        self.setup_trainer: SetupTrainer | None = None
        self.buffer: SetupBuffer | None = None
        self.collector: PeriodCollector | None = None
        self.init_record: dict = {}
        self.last_bundle_id: str | None = None
        self.last_bundle_period: int | None = None
        self.resume_record: dict | None = None
        self.setup_skips = 0
        self.integrity = {
            "legality_failures": 0,
            "orientation_failures": 0,
            "attribution_failures": 0,
            "non_finite_events": 0,
            "duplicates_collapsed": 0,
            "immediately_terminal_setups": 0,
        }
        torch.set_num_threads(int(config.threads))

    # -- construction ---------------------------------------------------------

    @classmethod
    def fresh(cls, config: PilotConfig, **keywords) -> "LineageRunner":
        runner = cls(config, **keywords)
        if runner.lineage_root.exists() and any(runner.lineage_root.iterdir()):
            raise Phase18G3Error(f"{runner.lineage_root} is not empty; a lineage is never restarted from scratch over old files")
        runner.lineage_root.mkdir(parents=True, exist_ok=True)
        runner.live_root.mkdir(parents=True, exist_ok=True)
        runner.bundles_root.mkdir(parents=True, exist_ok=True)

        runner.c1 = JointC1Trainer(
            config.c1_train_config,
            runner.corpus_identity,
            pilot=config,
            live_root=runner.live_root,
            root=runner.corpus_root,
            require_complete_split=runner.require_complete_split,
            value_prior=runner.value_prior,
            run_label=f"phase18_g3_{config.lineage}",
        )
        setup_model = build_setup_model(device=config.setup_device, seed=config.setup_init_seed())
        runner.setup_trainer = SetupTrainer(setup_model, config.setup_config(), namespace=config.namespace, seed_index=config.seed_index)
        runner.buffer = SetupBuffer(storage_duration=config.buffer_storage_periods, device=config.setup_device)
        runner.collector = PeriodCollector(config, runner.buffer, live_root=runner.live_root)
        runner.init_record = {
            "harness_version": G3_HARNESS_VERSION,
            "design_commit": G3_DESIGN_COMMIT,
            "run_id": config.run_id,
            "lineage": config.lineage,
            "setup_updates_enabled": config.setup_updates_enabled,
            "config_digest": config.config_digest(),
            "matched_digest": config.matched_digest(),
            "c1_init_state_digest": state_dict_digest(runner.c1.model),
            "c1_init_checksum": model_state_checksum(runner.c1.model.state_dict()),
            "c1_init_seed": int(config.c1_train_config.model_init_seed),
            "c1_parameter_count": int(sum(p.numel() for p in runner.c1.model.parameters())),
            "setup_init_seed": int(config.setup_init_seed()),
            "setup_init_state_digest": state_dict_digest(setup_model),
            "setup_ema_init_state_digest": state_dict_digest(runner.setup_trainer.ema.as_model(device="cpu")),
            "setup_parameter_count": int(sum(p.numel() for p in setup_model.parameters())),
            "corpus_identity": runner.corpus_identity.to_dict(),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (runner.lineage_root / INIT_NAME).write_text(json.dumps(runner.init_record, indent=1, sort_keys=True) + "\n")
        runner._write_bundle(0, telemetry={"note": "bundle_0: the recorded initial components, before any update"})
        runner._write_state()
        return runner

    @classmethod
    def resume(cls, config: PilotConfig, *, bundle_directory=None, **keywords) -> "LineageRunner":
        """Continue a lineage from its latest complete, verified bundle.

        `bundle_directory=None` selects the newest bundle that verifies
        (`select_resume_bundle`); `run_state.json` is never trusted for the
        choice. Everything the lineage produced after that bundle (live
        periods, period records, C1 update rows, unverifiable or partial
        bundles, the old state file) is moved into a dated archive under the
        lineage root, and the active records are restored to exactly one entry
        per completed period through the bundle before the replay begins.
        """
        runner = cls(config, **keywords)
        if bundle_directory is None:
            selection = select_resume_bundle(runner.lineage_root, config)
            bundle_directory = Path(selection["bundle"])
        else:
            bundle_directory = Path(bundle_directory)
            selection = {"bundle": str(bundle_directory), "period": None, "skipped": [], "selected_by": "caller"}
        manifest = verify_bundle(
            bundle_directory,
            expected_run_id=config.run_id,
            expected_lineage=config.lineage,
            expected_matched_digest=config.matched_digest(),
        )
        if manifest["config_digest"] != config.config_digest():
            raise Phase18G3Error("the bundle was written under a different lineage configuration")
        init_path = runner.lineage_root / INIT_NAME
        if not init_path.exists():
            raise Phase18G3Error(f"{init_path} is missing; the lineage cannot prove its initial identity")
        runner.init_record = json.loads(init_path.read_text())
        period = int(manifest["period"])
        selection["period"] = period
        archive = archive_progress_after(runner.lineage_root, period, updates_per_period=config.c1_updates_per_period)
        reader = LiveRecordReader(runner.live_root)
        recorded = {entry["period"]: entry for entry in manifest["live_periods"]}
        if set(reader.periods()) != set(recorded):
            raise Phase18G3Error(
                f"the live store holds periods {reader.periods()} but the bundle names {sorted(recorded)}"
            )
        for live_period, entry in recorded.items():
            verification = reader.verify_period(live_period)
            if not verification["verified"] or reader.summary(live_period)["commit_digest"] != entry["commit_digest"]:
                raise Phase18G3Error(f"live period {live_period} does not match the bundle: {verification}")

        runner.c1 = JointC1Trainer.resume(
            bundle_directory / C1_NAME,
            config=config.c1_train_config,
            corpus_identity=runner.corpus_identity,
            pilot=config,
            live_root=runner.live_root,
            root=runner.corpus_root,
            require_complete_split=runner.require_complete_split,
            value_prior=runner.value_prior,
            run_label=f"phase18_g3_{config.lineage}",
        )
        if state_dict_digest(runner.c1.model) != manifest["components"]["c1"]["state_digest"]:
            raise Phase18G3Error("the restored C1 weights do not reproduce the bundle's digest")
        if runner.c1.global_step != int(manifest["counters"]["c1_global_step"]):
            raise Phase18G3Error("the restored C1 step count disagrees with the manifest")
        runner.setup_trainer, _setup_manifest = load_setup_trainer(bundle_directory, config)
        if state_dict_digest(runner.setup_trainer.ema.as_model(device="cpu")) != manifest["components"]["setup_ema"]["state_digest"]:
            raise Phase18G3Error("the restored setup EMA does not reproduce the bundle's digest")
        state = load_collector_state(bundle_directory)
        runner.buffer = restore_buffer_state(state["buffer"], device=config.setup_device)
        if buffer_state_digest(runner.buffer) != state["buffer_state_digest"]:
            raise Phase18G3Error("the restored setup buffer does not reproduce its digest")
        runner.collector = PeriodCollector(config, runner.buffer, live_root=runner.live_root)
        restored = runner.collector.restore(state["collector"])
        if runner.collector.periods_completed != period:
            raise Phase18G3Error(
                f"the collector completed {runner.collector.periods_completed} periods, the bundle is period {period}"
            )
        records = read_period_records(runner.lineage_root)
        if [int(r["period"]) for r in records] != list(range(1, period + 1)):
            raise Phase18G3Error("the period records do not hold exactly one entry per completed period through the bundle")
        runner.period = period
        runner.last_bundle_id = manifest["bundle_id"]
        runner.last_bundle_period = period
        runner.setup_skips = int(manifest["telemetry"].get("setup_skips", 0))
        runner.integrity = dict(manifest["telemetry"].get("integrity", runner.integrity))
        runner.resume_record = {
            "bundle": str(bundle_directory),
            "bundle_id": manifest["bundle_id"],
            "period": period,
            "selection": selection,
            "archive": archive,
            "games_restored": restored["games_restored"],
            "c1_global_step": int(runner.c1.global_step),
            "resumed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        runner._write_state()
        return runner

    # -- the period loop --------------------------------------------------------

    def run_period(self) -> dict:
        config = self.config
        period = self.period + 1
        if period > config.periods:
            raise Phase18G3Error(f"period {period} is beyond the bounded horizon of {config.periods}; nothing continues automatically")
        assert self.c1 is not None and self.setup_trainer is not None and self.buffer is not None and self.collector is not None
        started = time.perf_counter()
        seconds: dict = {}

        # (1) the pool
        tick = time.perf_counter()
        actor = self.setup_trainer.generation_actor
        snapshot_digest = state_dict_digest(actor)
        generation = generate_pool(
            actor,
            namespace=config.namespace,
            seed_index=config.seed_index,
            snapshot_iteration=period - 1,
            snapshot_digest=snapshot_digest,
            count=config.pool_size,
            force_handedness=config.setup_config().force_handedness,
            reflection_probability=config.setup_config().reflection_probability,
            device=config.setup_device,
        )
        self.integrity["legality_failures"] += int(generation.telemetry["legality_failures"])
        self.integrity["orientation_failures"] += int(generation.telemetry["orientation_failures"])
        self.integrity["immediately_terminal_setups"] += int(generation.telemetry["immediately_terminal_count"])
        pool_record = self.buffer.add_pool(generation.samples, period=period)
        self.integrity["duplicates_collapsed"] += int(pool_record["duplicates_collapsed"])
        seconds["pool"] = time.perf_counter() - tick

        # (2) collection
        tick = time.perf_counter()
        self.collector.begin_period(period, generation.samples, snapshot_digest=snapshot_digest)
        self.collector.run_period()
        collection = self.collector.end_period()
        seconds["collection"] = time.perf_counter() - tick

        # (3) K C1 updates on the mixture
        tick = time.perf_counter()
        reader = LiveRecordReader(self.live_root)
        retained_periods = [q for q in reader.periods() if q > period - config.live_retention_periods]
        universe = reader.universe(retained_periods)
        c1_rows, c1_record = self.c1.train_period(period=period, live_universe=universe, updates=config.c1_updates_per_period)
        c1_record["retained_live_periods"] = retained_periods
        c1_record["losses"] = {
            "total_mean": float(np.mean([row["loss_total"] for row in c1_rows])) if "loss_total" in c1_rows[0] else None,
            "all_finite": bool(all(np.isfinite([v for v in row.values() if isinstance(v, float)]).all() for row in c1_rows)),
        }
        c1_record["validation_entries"] = len(self.c1.validation_history)
        seconds["c1"] = time.perf_counter() - tick

        # (4) the lineage switch
        tick = time.perf_counter()
        setup_record: dict
        if config.setup_updates_enabled:
            if self.buffer.ready_count() > 0:
                result = self.setup_trainer.update(self.buffer, global_iteration=period)
                self.integrity["non_finite_events"] += int(result.non_finite_events)
                setup_record = {"applied": True, "skipped": False, "update": result.document()}
                if result.optimizer_steps > 0 and result.digest_after == result.digest_before:
                    raise Phase18G3Error(f"period {period}: {result.optimizer_steps} setup steps ran but the raw digest did not move")
            else:
                self.setup_skips += 1
                setup_record = {"applied": False, "skipped": True, "reason": "no pooled setup received a completed outcome in this period"}
        else:
            setup_record = {"applied": False, "skipped": False, "reason": "control lineage: the setup model is frozen"}
            self._assert_control_frozen(period)
        seconds["setup"] = time.perf_counter() - tick

        # (5) retention
        retained = self.buffer.filter(period)
        self.integrity["attribution_failures"] = int(self.buffer.attribution_failures + self.collector.attribution_failures)

        raw_digest = state_dict_digest(self.setup_trainer.model)
        ema_digest = state_dict_digest(self.setup_trainer.ema.as_model(device="cpu"))
        record = {
            "harness_version": G3_HARNESS_VERSION,
            "run_id": config.run_id,
            "lineage": config.lineage,
            "setup_updates_enabled": config.setup_updates_enabled,
            "period": period,
            "pool": {
                "snapshot_digest": snapshot_digest,
                "snapshot_iteration": period - 1,
                "content_digest": pool_content_digest(generation.samples),
                "telemetry": generation.telemetry,
                "record": pool_record,
                "distinct_reflection_classes": int(generation.telemetry["distinct_class_fingerprints"]),
            },
            "collection": collection,
            "c1": c1_record,
            "c1_state_digest": state_dict_digest(self.c1.model),
            "c1_global_step": int(self.c1.global_step),
            "c1_counters": dict(self.c1.counters),
            "setup": setup_record,
            "setup_raw_digest": raw_digest,
            "setup_ema_digest": ema_digest,
            "setup_updates": int(self.setup_trainer.updates),
            "setup_optimizer_steps": int(self.setup_trainer.optimizer_step_count),
            "setup_ema_updates": int(self.setup_trainer.ema.updates),
            "setup_skips": int(self.setup_skips),
            "buffer": self.buffer.telemetry() | {"filter": retained, "state_digest": buffer_state_digest(self.buffer)},
            "integrity": dict(self.integrity),
            "seconds": {k: round(v, 3) for k, v in seconds.items()},
            "rss_bytes": _rss_bytes(),
        }
        self.period = period
        bundle_due = self.bundle_due(period)
        record["bundle_planned"] = bundle_due
        record["seconds"]["total"] = round(time.perf_counter() - started, 3)
        # Records first, bundle second: a bundle at period p then implies the
        # records and C1 rows of periods 1..p exist, which recovery relies on.
        self._append(PERIODS_NAME, record)
        with (self.lineage_root / C1_ROWS_NAME).open("a") as handle:
            for row in c1_rows:
                handle.write(json.dumps(_json_safe({"period": period, **row}), default=str) + "\n")
        bundle_written = None
        if bundle_due:
            tick = time.perf_counter()
            manifest = self._write_bundle(period, telemetry={"period_record": {k: record[k] for k in ("collection", "c1_state_digest", "setup_raw_digest", "setup_ema_digest", "integrity")}, "setup_skips": self.setup_skips, "integrity": dict(self.integrity)})
            seconds["bundle"] = time.perf_counter() - tick
            bundle_written = {"path": str(self.bundle_path(period)), "bundle_id": manifest["bundle_id"], "seconds": round(seconds["bundle"], 3)}
        self._write_state()
        self.log(
            f"[{config.lineage}] period {period}/{config.periods}: games {collection['completed']} "
            f"(in flight {collection['in_flight_at_end']}), c1 step {self.c1.global_step}, "
            f"setup {'update' if setup_record['applied'] else ('SKIP' if setup_record['skipped'] else 'frozen')}, "
            f"{record['seconds']['total']} s"
            + (f", bundle {bundle_written['bundle_id'][:12]}" if bundle_written else "")
        )
        record["bundle"] = bundle_written
        return record

    def bundle_due(self, period: int) -> bool:
        """Bundle after period 1 (the consequential-stop gate), at every cadence
        multiple, and at the horizon."""
        config = self.config
        return int(period) == 1 or int(period) % config.bundle_cadence_periods == 0 or int(period) == config.periods

    def run(self, *, periods: "int | None" = None) -> list:
        """Run `periods` more periods (default: to the bounded horizon)."""
        target = self.config.periods if periods is None else min(self.config.periods, self.period + int(periods))
        records = []
        while self.period < target:
            records.append(self.run_period())
        return records

    # -- helpers ------------------------------------------------------------------

    def _assert_control_frozen(self, period: int) -> None:
        assert self.setup_trainer is not None
        raw = state_dict_digest(self.setup_trainer.model)
        ema = state_dict_digest(self.setup_trainer.ema.as_model(device="cpu"))
        expected = self.init_record["setup_init_state_digest"]
        if raw != expected or ema != expected:
            raise Phase18G3Error(
                f"period {period}: the control setup model moved (raw {raw[:12]}, ema {ema[:12]}, init {expected[:12]})"
            )
        if self.setup_trainer.updates or self.setup_trainer.optimizer_step_count or self.setup_trainer.ema.updates:
            raise Phase18G3Error("the control setup trainer recorded an update")

    def bundle_path(self, period: int) -> Path:
        return self.bundles_root / bundle_name(period)

    def _write_bundle(self, period: int, *, telemetry: dict) -> dict:
        assert self.c1 is not None and self.setup_trainer is not None and self.buffer is not None and self.collector is not None
        manifest = write_bundle(
            self.bundle_path(period),
            config=self.config,
            period=period,
            c1_trainer=self.c1,
            setup_trainer=self.setup_trainer,
            buffer=self.buffer,
            collector=self.collector,
            live_reader=LiveRecordReader(self.live_root),
            telemetry=_json_safe(telemetry),
            parent_bundle_id=self.last_bundle_id,
        )
        self.last_bundle_id = manifest["bundle_id"]
        self.last_bundle_period = int(period)
        return manifest

    def _append(self, name: str, record: dict) -> None:
        with (self.lineage_root / name).open("a") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True, default=str) + "\n")

    def _write_state(self) -> None:
        state = {
            "harness_version": G3_HARNESS_VERSION,
            "run_id": self.config.run_id,
            "lineage": self.config.lineage,
            "period": int(self.period),
            "periods": int(self.config.periods),
            "last_bundle_id": self.last_bundle_id,
            "last_bundle_period": self.last_bundle_period,
            "last_bundle": str(self.bundle_path(self.last_bundle_period)) if self.last_bundle_period is not None else None,
            "resume": self.resume_record,
            "c1_global_step": int(self.c1.global_step) if self.c1 is not None else None,
            "setup_updates": int(self.setup_trainer.updates) if self.setup_trainer is not None else None,
            "integrity": dict(self.integrity),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = self.lineage_root / STATE_NAME
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)

    def close(self) -> None:
        if self.c1 is not None:
            self.c1.close()


# ---------------------------------------------------------------------------
# Reading a lineage back
# ---------------------------------------------------------------------------


def read_period_records(lineage_root) -> list:
    path = Path(lineage_root) / PERIODS_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_init_record(lineage_root) -> dict:
    return json.loads((Path(lineage_root) / INIT_NAME).read_text())


# ---------------------------------------------------------------------------
# The matching check (design 2.2: identical by construction through period 1)
# ---------------------------------------------------------------------------

#: Period-record fields that must be identical between the lineages in period 1
#: whatever the C1 device, because nothing on their path touches C1 weights.
#: The live store's RAW commit digest is deliberately absent: it hashes the
#: metadata line, which carries the lineage stamp, so it differs between the
#: lineages by construction whenever a game completed (the first pilot's
#: period-1 stop). The live store is compared semantically instead
#: (`compare_live_periods`), and both raw digests are reported for audit.
MATCHED_PERIOD_FIELDS = (
    ("pool", "content_digest"),
    ("pool", "snapshot_digest"),
    ("collection", "started"),
    ("collection", "completed"),
    ("collection", "in_flight_at_end"),
    ("collection", "plies_advanced"),
    ("collection", "completed_game_ids_digest"),
    ("collection", "outcome_records_digest"),
    ("collection", "live", "games"),
    ("collection", "live", "selected_examples"),
    ("c1", "live_universe_digest"),
    ("c1", "keys_digests"),
    ("c1", "updates_planned"),
    ("c1", "updates_completed"),
    ("c1", "live_seeds"),
    ("c1", "canonical_rows_planned"),
    ("c1", "live_rows_planned"),
    ("c1", "canonical_rows_served"),
    ("c1", "live_rows_served"),
    ("buffer", "rows"),
)


def _lookup(record: dict, path: tuple):
    value = record
    for key in path:
        value = value[key]
    return value


def matching_check(run_root, *, c1_device: str) -> dict:
    """Compare the two lineages before setup learning can have separated them.

    Exact identity is required for the initial components (bundle_0 differs
    only in lineage) and for every period-1 quantity that does not pass
    through a C1 forward/backward pass. The period-1 live stores are compared
    semantically (same games, same trajectories, same selected decisions,
    same metadata apart from the lineage stamp: `compare_live_periods`);
    their raw commit digests are reported under `live_store` as audit
    information and are expected to differ. Each lineage's period-1 C1 rows
    must have served exactly the planned canonical and live counts. The
    period-1 C1 weight digest is required to be identical on CPU (bit-exact
    with fixed threads) and is reported, not required, on MPS (P18-D002: MPS
    training is not run-to-run bitwise reproducible), because a period-1
    difference there is not evidence of an implementation defect.
    """
    run_root = Path(run_root)
    inits = {lineage: read_init_record(run_root / lineage) for lineage in LINEAGES}
    periods = {lineage: read_period_records(run_root / lineage) for lineage in LINEAGES}
    problems: list = []
    report: dict = {"c1_device": c1_device, "c1_exact_required": not str(c1_device).startswith("mps")}

    init_fields = (
        "matched_digest",
        "c1_init_state_digest",
        "c1_init_checksum",
        "setup_init_seed",
        "setup_init_state_digest",
        "setup_ema_init_state_digest",
        "corpus_identity",
    )
    report["init"] = {}
    for name in init_fields:
        same = inits["candidate"][name] == inits["control"][name]
        report["init"][name] = same
        if not same:
            problems.append(f"init {name} differs between lineages")
    if inits["candidate"]["setup_updates_enabled"] is not True or inits["control"]["setup_updates_enabled"] is not False:
        problems.append("the lineage switch is not set as candidate=True, control=False")

    bundle_zero = {}
    for lineage in LINEAGES:
        path = run_root / lineage / BUNDLES_DIRECTORY / bundle_name(0)
        manifest = read_manifest(path)
        bundle_zero[lineage] = manifest
    from .g3_bundle import compare_bundles

    report["bundle_0"] = compare_bundles(bundle_zero["candidate"], bundle_zero["control"])
    for name, same in report["bundle_0"].items():
        if name in ("c1", "setup_raw", "setup_ema") and not same:
            problems.append(f"bundle_0 component {name} differs between lineages")
    report["bundle_0"]["lineage_differs"] = bundle_zero["candidate"]["lineage"] != bundle_zero["control"]["lineage"]

    first = {lineage: next((r for r in periods[lineage] if r["period"] == 1), None) for lineage in LINEAGES}
    report["period_1"] = {}
    if any(first[lineage] is None for lineage in LINEAGES):
        problems.append("period 1 has not completed in both lineages")
    else:
        for path in MATCHED_PERIOD_FIELDS:
            try:
                same = _lookup(first["candidate"], path) == _lookup(first["control"], path)
            except KeyError:
                same = False
            report["period_1"]["/".join(path)] = same
            if not same:
                problems.append(f"period 1 {'/'.join(path)} differs between lineages")
        c1_same = first["candidate"]["c1_state_digest"] == first["control"]["c1_state_digest"]
        report["period_1"]["c1_state_digest"] = c1_same
        if not c1_same and report["c1_exact_required"]:
            problems.append("period 1 C1 weight digests differ on a CPU run (an implementation defect)")
        report["period_1"]["c1_state_digest_note"] = (
            "required identical (CPU, fixed threads)" if report["c1_exact_required"] else "reported only (MPS is not bitwise reproducible, P18-D002)"
        )
        # Before any setup update both lineages generated period 1 from the init.
        for lineage in LINEAGES:
            if first[lineage]["pool"]["snapshot_digest"] != inits[lineage]["setup_init_state_digest"]:
                problems.append(f"{lineage}: the period-1 pool was not sampled by the initial setup model")

        # The live stores: semantic identity is the condition; the raw digests are audit.
        try:
            live = compare_live_periods(
                run_root / "candidate" / LIVE_DIRECTORY,
                run_root / "control" / LIVE_DIRECTORY,
                1,
                lineage_a="candidate",
                lineage_b="control",
            )
        except Phase18G3Error as error:
            live = {"matched": False, "problems": [str(error)], "raw_commit_digest": None, "raw_commit_digest_equal": None, "semantic": None}
        report["live_store"] = live
        report["period_1"]["collection/live/semantic_identity"] = bool(live["matched"])
        if not live["matched"]:
            problems.append("period 1 live stores are not semantically identical: " + "; ".join(live["problems"]))
        recorded = True
        if live.get("raw_commit_digest"):
            for lineage in LINEAGES:
                if first[lineage]["collection"]["live"]["commit_digest"] != live["raw_commit_digest"][lineage]:
                    recorded = False
                    problems.append(f"{lineage}: the period record's raw live commit digest is not the store's own")
        report["period_1"]["collection/live/raw_commit_digest_recorded"] = recorded

        # Each lineage served exactly the planned mixture in period 1.
        served = True
        for lineage in LINEAGES:
            c1 = first[lineage]["c1"]
            ok = (
                c1.get("live_rows_served") is not None
                and c1.get("canonical_rows_served") is not None
                and int(c1["live_rows_served"]) == int(c1["live_rows_planned"])
                and int(c1["canonical_rows_served"]) == int(c1["canonical_rows_planned"])
            )
            if not ok:
                served = False
                problems.append(
                    f"{lineage}: period 1 served {c1.get('canonical_rows_served')} canonical / {c1.get('live_rows_served')} live rows, "
                    f"planned {c1.get('canonical_rows_planned')} / {c1.get('live_rows_planned')}"
                )
        report["period_1"]["c1/served_rows_equal_planned"] = served

    # The equal gameplay-update budget, over every completed period.
    budget = {}
    for lineage in LINEAGES:
        records = periods[lineage]
        budget[lineage] = {
            "periods": len(records),
            "c1_updates": int(sum(r["c1"]["updates_completed"] for r in records)),
            "c1_global_step": int(records[-1]["c1_global_step"]) if records else 0,
            "canonical_rows_served": int(sum(int(r["c1"].get("canonical_rows_served") or 0) for r in records)),
            "live_rows_served": int(sum(int(r["c1"].get("live_rows_served") or 0) for r in records)),
            "setup_updates": int(records[-1]["setup_updates"]) if records else 0,
            "setup_ema_updates": int(records[-1]["setup_ema_updates"]) if records else 0,
        }
    report["budget"] = budget
    common = min(budget["candidate"]["periods"], budget["control"]["periods"])
    if common:
        for lineage in LINEAGES:
            expected = common * periods[lineage][0]["c1"]["updates_planned"]
            observed = int(sum(r["c1"]["updates_completed"] for r in periods[lineage][:common]))
            if observed != expected:
                problems.append(f"{lineage}: {observed} C1 updates over {common} periods, expected {expected}")
        for name in ("canonical_rows_served", "live_rows_served"):
            counts = {lineage: int(sum(int(r["c1"].get(name) or 0) for r in periods[lineage][:common])) for lineage in LINEAGES}
            if counts["candidate"] != counts["control"]:
                problems.append(f"{name} over the common {common} periods differs: {counts}")
        if budget["control"]["setup_updates"] or budget["control"]["setup_ema_updates"]:
            problems.append("the control lineage recorded a setup or EMA update")
        control_digests = {(r["setup_raw_digest"], r["setup_ema_digest"]) for r in periods["control"]}
        init_digest = inits["control"]["setup_init_state_digest"]
        if control_digests != {(init_digest, init_digest)}:
            problems.append("the control setup digests moved away from the initial version")
        report["control_setup_digest_constant"] = control_digests == {(init_digest, init_digest)}
        applied = [r for r in periods["candidate"] if r["setup"]["applied"]]
        report["candidate_setup_updates_applied"] = len(applied)
        moved = all(r["setup"]["update"]["raw_digest_after"] != r["setup"]["update"]["raw_digest_before"] for r in applied)
        report["candidate_setup_moved_on_every_applied_update"] = moved
        if applied and not moved:
            problems.append("a candidate setup update left the raw digest unchanged")
        report["candidate_setup_skips"] = int(sum(1 for r in periods["candidate"] if r["setup"]["skipped"]))
    report["problems"] = problems
    report["matched"] = not problems
    return report




# ---------------------------------------------------------------------------
# Recovery: the latest verified bundle, and the archive of everything after it
# ---------------------------------------------------------------------------


def read_c1_rows(lineage_root) -> list:
    path = Path(lineage_root) / C1_ROWS_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_jsonl(path: Path, period: int, *, key: str = "period") -> tuple:
    """`(kept, archived, partial_line)`: entries with `key <= period`, the
    rest, and a malformed trailing line (an interrupted append) if there is
    one. A malformed line anywhere else is corruption and is refused."""
    if not path.exists():
        return [], [], None
    lines = path.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    kept, archived = [], []
    partial = None
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                partial = raw
                continue
            raise Phase18G3Error(f"{path} line {index + 1} is not JSON; the record file is corrupt")
        (kept if int(entry[key]) <= int(period) else archived).append(entry)
    return kept, archived, partial


def _rewrite_jsonl(path: Path, entries: list) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(_json_safe(entry), sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_resume_bundle(lineage_root, config: PilotConfig) -> dict:
    """The newest bundle that verifies whole, whatever `run_state.json` says.

    Bundles are tried from the highest period downwards; a partial directory,
    a tampered component or a foreign identity is skipped and recorded, never
    repaired. No verified bundle at all is fatal.
    """
    bundles_root = Path(lineage_root) / BUNDLES_DIRECTORY
    candidates = []
    for path in bundles_root.glob("bundle_*") if bundles_root.exists() else ():
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        stem = path.name[len("bundle_") :]
        if stem.isdigit():
            candidates.append((int(stem), path))
    candidates.sort(reverse=True)
    skipped = []
    for period, path in candidates:
        try:
            manifest = verify_bundle(
                path,
                expected_run_id=config.run_id,
                expected_lineage=config.lineage,
                expected_period=period,
                expected_matched_digest=config.matched_digest(),
            )
        except Phase18G3Error as error:
            skipped.append({"bundle": str(path), "period": period, "problem": str(error)})
            continue
        return {
            "bundle": str(path),
            "period": period,
            "bundle_id": manifest["bundle_id"],
            "skipped": skipped,
            "selected_by": "latest complete verified bundle",
        }
    raise Phase18G3Error(f"no complete verified bundle under {bundles_root}; skipped: {skipped}")


def archive_progress_after(lineage_root, period: int, *, updates_per_period: int) -> dict:
    """Move everything the lineage produced after bundle `period` into a dated
    archive and restore the active records to exactly periods 1..`period`.

    Validated before anything moves: the kept period records must be exactly
    one per period 1..`period` in order, and the kept C1 rows exactly
    `updates_per_period` per period. Nothing is deleted.
    """
    lineage_root = Path(lineage_root)
    n = int(period)
    periods_path = lineage_root / PERIODS_NAME
    rows_path = lineage_root / C1_ROWS_NAME
    kept_records, tail_records, partial_record = _split_jsonl(periods_path, n)
    kept_periods = [int(r["period"]) for r in kept_records]
    if kept_periods != list(range(1, n + 1)):
        raise Phase18G3Error(
            f"{periods_path} does not hold exactly one record for each of periods 1..{n} (found {kept_periods[:12]})"
        )
    kept_rows, tail_rows, partial_row = _split_jsonl(rows_path, n)
    counts = Counter(int(r["period"]) for r in kept_rows)
    expected = {p: int(updates_per_period) for p in range(1, n + 1)}
    if dict(counts) != expected:
        raise Phase18G3Error(
            f"{rows_path} does not hold exactly {updates_per_period} C1 rows for each of periods 1..{n} (found {dict(counts)})"
        )
    live_root = lineage_root / LIVE_DIRECTORY
    later_live = [p for p in (live_root.glob("period_*") if live_root.exists() else ()) if ".orphaned" not in p.name and p.name[len("period_") : len("period_") + 4].isdigit() and int(p.name[len("period_") : len("period_") + 4]) > n]
    bundles_root = lineage_root / BUNDLES_DIRECTORY
    later_bundles = []
    for path in bundles_root.glob("bundle_*") if bundles_root.exists() else ():
        stem = path.name[len("bundle_") :].split(".")[0]
        if stem.isdigit() and (int(stem) > n or path.name.endswith(".partial")):
            later_bundles.append(path)
    record = {
        "archive": None,
        "restored_through_period": n,
        "live_periods_moved": [],
        "period_records_archived": len(tail_records),
        "c1_rows_archived": len(tail_rows),
        "partial_lines_archived": int(partial_record is not None) + int(partial_row is not None),
        "bundles_moved": [],
        "previous_state_archived": False,
    }
    if not (tail_records or tail_rows or partial_record or partial_row or later_live or later_bundles):
        return record
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = lineage_root / ARCHIVE_DIRECTORY / f"{stamp}_after_bundle_{n:04d}"
    suffix = 0
    while archive.exists():
        suffix += 1
        archive = lineage_root / ARCHIVE_DIRECTORY / f"{stamp}_after_bundle_{n:04d}_{suffix}"
    archive.mkdir(parents=True)
    record["archive"] = str(archive)
    record["live_periods_moved"] = discard_periods_after(live_root, n, destination=archive / LIVE_DIRECTORY)
    if tail_records or partial_record:
        with (archive / "periods_after.jsonl").open("w") as handle:
            for entry in tail_records:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            if partial_record is not None:
                (archive / "periods_after.partial_line").write_text(partial_record)
        _rewrite_jsonl(periods_path, kept_records)
    if tail_rows or partial_row:
        with (archive / "c1_rows_after.jsonl").open("w") as handle:
            for entry in tail_rows:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            if partial_row is not None:
                (archive / "c1_rows_after.partial_line").write_text(partial_row)
        _rewrite_jsonl(rows_path, kept_rows)
    for path in sorted(later_bundles):
        target = archive / BUNDLES_DIRECTORY / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        record["bundles_moved"].append({"from": str(path), "to": str(target)})
    state_path = lineage_root / STATE_NAME
    if state_path.exists():
        shutil.copy2(state_path, archive / STATE_NAME)
        record["previous_state_archived"] = True
    return record


# ---------------------------------------------------------------------------
# The fairness conditions every decision depends on
# ---------------------------------------------------------------------------


def fairness_conditions(run_root, *, periods: int, updates_per_period: int, matching, contract_sha256) -> dict:
    """Every condition the primary comparison needs before it can decide.

    A missing or failed lineage-matching record is a failed condition, never
    telemetry. Every other condition reads the lineages' own records and
    final bundles, so a decision cannot rest on an incomplete, duplicated or
    unequal history.
    """
    run_root = Path(run_root)
    periods = int(periods)
    conditions: dict = {}
    problems: list = []

    matched = bool(matching) and matching.get("matched") is True
    bound = bool(matching) and (contract_sha256 is None or matching.get("contract_sha256") == contract_sha256)
    conditions["matching_record_present"] = bool(matching)
    conditions["matching_record_contract_bound"] = bool(matching) and bound
    conditions["matching_record_matched"] = matched
    if not matching:
        problems.append("the lineage-matching record is missing; it is required evidence, not telemetry")
    else:
        if not bound:
            problems.append("the lineage-matching record was written against a different contract")
        if not matched:
            problems.append(f"the lineage-matching record reports problems: {matching.get('problems')}")

    records = {}
    inits = {}
    budgets = {}
    for lineage in LINEAGES:
        root = run_root / lineage
        records[lineage] = read_period_records(root)
        inits[lineage] = read_init_record(root) if (root / INIT_NAME).exists() else None
        ps = [int(r["period"]) for r in records[lineage]]
        exact = ps == list(range(1, periods + 1))
        conditions[f"{lineage}_periods_exactly_1_to_{periods}"] = exact
        if not exact:
            problems.append(f"{lineage}: period records are {ps[:12]}..., not exactly 1..{periods} once each in order")
        rows = read_c1_rows(root)
        counts = Counter(int(r["period"]) for r in rows)
        rows_exact = dict(counts) == {p: int(updates_per_period) for p in range(1, periods + 1)}
        conditions[f"{lineage}_c1_rows_exactly_{updates_per_period}_per_period"] = rows_exact
        if not rows_exact:
            problems.append(f"{lineage}: C1 update rows per period are {dict(sorted(counts.items()))[:1] if False else dict(sorted(counts.items()))}, not {updates_per_period} for each of 1..{periods}")
        budgets[lineage] = {
            "completed_updates": int(sum(int(r["c1"]["updates_completed"]) for r in records[lineage])),
            "final_global_step": int(records[lineage][-1]["c1_global_step"]) if records[lineage] else None,
        }
    expected_budget = periods * int(updates_per_period)
    equal = (
        budgets["candidate"] == budgets["control"]
        and budgets["candidate"]["completed_updates"] == expected_budget
        and budgets["candidate"]["final_global_step"] == expected_budget
    )
    conditions["equal_completed_c1_update_budget"] = bool(equal)
    if not equal:
        problems.append(f"C1 update budgets are not equal and complete: {budgets} (expected {expected_budget} each)")

    control = records["control"]
    init_control = inits["control"]
    if control and init_control:
        last = control[-1]
        zero = int(last["setup_updates"]) == 0 and int(last["setup_ema_updates"]) == 0 and int(last["setup_optimizer_steps"]) == 0
        digest = init_control["setup_init_state_digest"]
        unchanged = all(r["setup_raw_digest"] == digest and r["setup_ema_digest"] == digest for r in control)
        conditions["control_zero_setup_and_ema_updates"] = bool(zero)
        conditions["control_setup_digest_unchanged"] = bool(unchanged)
        if not zero:
            problems.append("the control lineage recorded a setup, optimizer or EMA update")
        if not unchanged:
            problems.append("the control setup digest moved away from the initial version")
    else:
        conditions["control_zero_setup_and_ema_updates"] = False
        conditions["control_setup_digest_unchanged"] = False
        problems.append("the control lineage has no records or no init record")

    candidate = records["candidate"]
    init_candidate = inits["candidate"]
    if candidate and init_candidate:
        applied = [
            r for r in candidate
            if r["setup"]["applied"] and int(r["setup"]["update"]["optimizer_steps"]) >= 1
            and r["setup"]["update"]["raw_digest_after"] != r["setup"]["update"]["raw_digest_before"]
        ]
        conditions["candidate_at_least_one_real_setup_update"] = bool(applied)
        if not applied:
            problems.append("the candidate lineage applied no real setup update")
        digest = init_candidate["setup_init_state_digest"]
        final = candidate[-1]
        moved = final["setup_raw_digest"] != digest and final["setup_ema_digest"] != digest
        conditions["candidate_final_setup_digests_differ_from_init"] = bool(moved)
        if not moved:
            problems.append("the candidate's final raw or EMA setup digest equals its initialisation")
    else:
        conditions["candidate_at_least_one_real_setup_update"] = False
        conditions["candidate_final_setup_digests_differ_from_init"] = False
        problems.append("the candidate lineage has no records or no init record")

    final_bundles = {}
    for lineage in LINEAGES:
        init = inits[lineage]
        path = run_root / lineage / BUNDLES_DIRECTORY / bundle_name(periods)
        try:
            if init is None:
                raise Phase18G3Error("no init record")
            manifest = verify_bundle(path, expected_run_id=init["run_id"], expected_lineage=lineage, expected_period=periods, expected_matched_digest=init["matched_digest"])
            last = records[lineage][-1] if records[lineage] else None
            consistent = (
                last is not None
                and int(manifest["counters"]["c1_global_step"]) == expected_budget
                and manifest["components"]["c1"]["state_digest"] == last["c1_state_digest"]
                and manifest["components"]["setup_raw"]["state_digest"] == last["setup_raw_digest"]
                and manifest["components"]["setup_ema"]["state_digest"] == last["setup_ema_digest"]
            )
            final_bundles[lineage] = {"bundle_id": manifest["bundle_id"], "verified": True, "consistent_with_records": bool(consistent)}
            if not consistent:
                problems.append(f"{lineage}: the final bundle disagrees with the final period record")
        except Phase18G3Error as error:
            final_bundles[lineage] = {"bundle_id": None, "verified": False, "consistent_with_records": False, "problem": str(error)}
            problems.append(f"{lineage}: the final bundle {path.name} is missing or does not verify: {error}")
        conditions[f"{lineage}_final_bundle_complete"] = bool(final_bundles[lineage]["verified"] and final_bundles[lineage]["consistent_with_records"])
    return {
        "periods": periods,
        "updates_per_period": int(updates_per_period),
        "conditions": conditions,
        "budgets": budgets,
        "final_bundles": final_bundles,
        "problems": problems,
        "all_hold": all(conditions.values()) and not problems,
    }


def decision_input(analysis: dict, gates: dict, fairness: dict) -> dict:
    """PROCEED only when the primary rule passes AND every computable gate AND
    every fairness condition hold; a failed fairness condition or gate blocks
    the decision outright, because the comparison is then not the frozen one."""
    computable = {name: value for name, value in gates.items() if value is not None}
    failed_gates = sorted(name for name, value in computable.items() if not value)
    if not fairness["all_hold"]:
        decision, basis = "BLOCKED", f"a fairness condition failed: {fairness['problems']}"
    elif failed_gates:
        decision, basis = "BLOCKED", f"gate(s) failed: {failed_gates}"
    elif analysis["passes"]:
        decision, basis = "PROCEED", "the frozen rule passes with every gate and fairness condition holding"
    elif analysis["near_boundary"]:
        decision, basis = "NEAR_BOUNDARY", "the interval contains the margin; a conditional second seed is the design's follow-up"
    else:
        decision, basis = "FAIL", "the frozen rule does not pass"
    return {
        "decision": decision,
        "basis": basis,
        "primary_passes": bool(analysis["passes"]),
        "near_boundary": bool(analysis["near_boundary"]),
        "failed_gates": failed_gates,
        "fairness_all_hold": bool(fairness["all_hold"]),
        "fairness_problems": list(fairness["problems"]),
        "uncomputed_gates": sorted(name for name, value in gates.items() if value is None),
    }

__all__ = [
    "ARCHIVE_DIRECTORY",
    "BUNDLES_DIRECTORY",
    "C1_ROWS_NAME",
    "INIT_NAME",
    "LIVE_DIRECTORY",
    "LineageRunner",
    "MATCHED_PERIOD_FIELDS",
    "PERIODS_NAME",
    "STATE_NAME",
    "archive_progress_after",
    "bundle_name",
    "decision_input",
    "fairness_conditions",
    "matching_check",
    "pool_content_digest",
    "read_c1_rows",
    "read_init_record",
    "read_period_records",
    "select_resume_bundle",
]
