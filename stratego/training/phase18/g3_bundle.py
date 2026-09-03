"""Phase 18 Stage 6B: the joint checkpoint bundle (G3-ENG-03, design section 3).

```text
<bundle>/manifest.json     run, LINEAGE, setup_updates_enabled, seed, period,
                           counters, every component's sha256 and state digest,
                           the live periods it knows about, bundle_id
<bundle>/c1.pt             warmstart_checkpoint_v1: C1 weights, AdamW state,
                           scheduler, data cursor, global step, RNG (unchanged
                           accepted format, written by the accepted writer)
<bundle>/setup/            SetupTrainer.save_checkpoint: raw.pt, optimizer.pt,
                           ema.pt, manifest.json (unchanged accepted format)
<bundle>/collector.pt      the slot population (Phase 17 codec, compressed),
                           the setup buffer's exact state, the period counters
```

Rules (gate G5): a bundle is written and loaded only as a whole; every
component's digest must equal the digest bound in the manifest; a load
refuses a bundle whose manifest names another run, lineage or period; no
component of one bundle is ever paired with a component of another. The
`bundle_id` is the sha256 over the manifest without the id itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

from ..warmstart_checkpoint import read_warmstart_payload, validate_warmstart_payload
from .g3_contract import (
    G3_BUNDLE_VERSION,
    G3_DESIGN_COMMIT,
    G3_HARNESS_VERSION,
    LINEAGES,
    Phase18G3Error,
    Phase18G3LineageError,
    PilotConfig,
)
from .g3_buffer_state import buffer_state_digest, capture_buffer_state
from .setup_contract import SETUP_CHECKPOINT_VERSION, file_sha256, json_document_digest
from .setup_learning import SetupTrainer
from .setup_model import state_dict_digest

MANIFEST_NAME = "manifest.json"
C1_NAME = "c1.pt"
SETUP_DIRECTORY = "setup"
COLLECTOR_NAME = "collector.pt"
COMPONENT_NAMES = ("c1", "setup_raw", "setup_ema", "setup_optimizer", "setup_manifest", "collector")


def bundle_identity(manifest: dict) -> str:
    """sha256 over the manifest without `bundle_id`."""
    return json_document_digest({key: value for key, value in manifest.items() if key != "bundle_id"})


def c1_state_digest(model) -> str:
    """The accepted parameter digest of the C1 model."""
    return state_dict_digest(model)


def write_bundle(
    directory,
    *,
    config: PilotConfig,
    period: int,
    c1_trainer,
    setup_trainer: SetupTrainer,
    buffer,
    collector,
    live_reader,
    telemetry: dict,
    parent_bundle_id: "str | None",
) -> dict:
    """Write one complete joint bundle. Never overwrites an existing one."""
    target = Path(directory)
    if target.exists():
        raise Phase18G3Error(f"{target} already exists; a bundle is never overwritten")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise Phase18G3Error(f"{partial} exists from an interrupted write; remove it deliberately first")
    partial.mkdir(parents=True)

    c1_record = c1_trainer.save_checkpoint(partial / C1_NAME)
    c1_payload = read_warmstart_payload(partial / C1_NAME)
    c1_metadata = validate_warmstart_payload(c1_payload, source=str(partial / C1_NAME))
    setup_manifest = setup_trainer.save_checkpoint(partial / SETUP_DIRECTORY)

    collector_state = collector.capture()
    buffer_state = capture_buffer_state(buffer)
    torch.save({"collector": collector_state, "buffer": buffer_state, "buffer_state_digest": buffer_state_digest(buffer)}, partial / COLLECTOR_NAME)

    live_periods = []
    for live_period in live_reader.periods():
        if int(live_period) > int(period):
            raise Phase18G3Error(
                f"the live store holds period {live_period} beyond the bundle period {period}; "
                "a bundle may only reference periods that precede it"
            )
        summary = live_reader.summary(live_period)
        live_periods.append(
            {
                "period": int(live_period),
                "games": int(summary["games"]),
                "selected_examples": int(summary["selected_examples"]),
                "commit_digest": summary["commit_digest"],
                "files": summary["files"],
            }
        )

    setup_directory = partial / SETUP_DIRECTORY
    manifest = {
        "bundle_version": G3_BUNDLE_VERSION,
        "harness_version": G3_HARNESS_VERSION,
        "design_commit": G3_DESIGN_COMMIT,
        "run_id": config.run_id,
        "lineage": config.lineage,
        "setup_updates_enabled": bool(config.setup_updates_enabled),
        "seed_index": int(config.seed_index),
        "namespace": config.namespace,
        "period": int(period),
        "config_digest": config.config_digest(),
        "matched_digest": config.matched_digest(),
        "config": config.document(),
        "counters": {
            "c1_global_step": int(c1_trainer.global_step),
            "c1_examples_consumed": int(c1_trainer.examples_consumed),
            "c1_cursor": c1_trainer.cursor.to_dict(),
            "c1_validation_entries": len(c1_trainer.validation_history),
            "setup_updates": int(setup_trainer.updates),
            "setup_optimizer_steps": int(setup_trainer.optimizer_step_count),
            "setup_ema_updates": int(setup_trainer.ema.updates),
            "setup_non_finite_events": int(setup_trainer.non_finite_events),
            "collector": collector.telemetry(),
            "buffer": buffer.telemetry(),
        },
        "components": {
            "c1": {
                "file": C1_NAME,
                "sha256": file_sha256(partial / C1_NAME),
                "bytes": int(c1_record["bytes"]),
                "integrity_digest": c1_record["integrity_digest"],
                "state_digest": c1_state_digest(c1_trainer.model),
                "train_config_digest": c1_metadata["train_config_digest"],
                "corpus_identity": c1_metadata["corpus_identity"],
                "global_step": int(c1_metadata["global_step"]),
            },
            "setup_raw": {
                "file": f"{SETUP_DIRECTORY}/{setup_manifest['raw']['file']}",
                "sha256": setup_manifest["raw"]["sha256"],
                "state_digest": setup_manifest["raw"]["state_digest"],
            },
            "setup_ema": {
                "file": f"{SETUP_DIRECTORY}/{setup_manifest['ema']['file']}",
                "sha256": setup_manifest["ema"]["sha256"],
                "state_digest": setup_manifest["ema"]["state_digest"],
            },
            "setup_optimizer": {
                "file": f"{SETUP_DIRECTORY}/{setup_manifest['optimizer']['file']}",
                "sha256": setup_manifest["optimizer"]["sha256"],
            },
            "setup_manifest": {
                "file": f"{SETUP_DIRECTORY}/manifest.json",
                "sha256": file_sha256(setup_directory / "manifest.json"),
                "config_digest": setup_manifest["config_digest"],
            },
            "collector": {
                "file": COLLECTOR_NAME,
                "sha256": file_sha256(partial / COLLECTOR_NAME),
                "buffer_state_digest": buffer_state_digest(buffer),
                "active_games": int(collector_state["active_games"]),
            },
        },
        "live_periods": live_periods,
        "parent_bundle_id": parent_bundle_id,
        "telemetry": dict(telemetry),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["bundle_id"] = bundle_identity(manifest)
    manifest_path = partial / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    with manifest_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, target)
    directory_handle = os.open(str(target.parent), os.O_RDONLY)
    try:
        os.fsync(directory_handle)
    finally:
        os.close(directory_handle)
    verify_bundle(target, expected_run_id=config.run_id, expected_lineage=config.lineage, expected_period=period)
    return manifest


def read_manifest(directory) -> dict:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise Phase18G3Error(f"no bundle manifest at {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("bundle_version") != G3_BUNDLE_VERSION:
        raise Phase18G3Error(f"{path}: bundle version {manifest.get('bundle_version')!r} is not {G3_BUNDLE_VERSION!r}")
    if manifest.get("lineage") not in LINEAGES:
        raise Phase18G3Error(f"{path}: unknown lineage {manifest.get('lineage')!r}")
    if bundle_identity(manifest) != manifest.get("bundle_id"):
        raise Phase18G3Error(f"{path}: the manifest does not reproduce its own bundle_id")
    return manifest


def verify_bundle(
    directory,
    *,
    expected_run_id: "str | None" = None,
    expected_lineage: "str | None" = None,
    expected_period: "int | None" = None,
    expected_matched_digest: "str | None" = None,
) -> dict:
    """Re-hash every component and check the identity the caller expects."""
    target = Path(directory)
    manifest = read_manifest(target)
    problems = []
    for name in COMPONENT_NAMES:
        entry = manifest["components"].get(name)
        if entry is None:
            problems.append(f"component {name} is missing from the manifest")
            continue
        path = target / entry["file"]
        if not path.exists():
            problems.append(f"component {name}: {path} is missing")
            continue
        observed = file_sha256(path)
        if observed != entry["sha256"]:
            problems.append(f"component {name}: digest {observed} != bound {entry['sha256']}")
    if problems:
        raise Phase18G3Error(f"bundle {target} does not verify: {problems}")
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise Phase18G3Error(f"bundle {target} belongs to run {manifest['run_id']!r}, not {expected_run_id!r}")
    if expected_lineage is not None and manifest["lineage"] != expected_lineage:
        raise Phase18G3LineageError(
            f"bundle {target} belongs to the {manifest['lineage']} lineage, not {expected_lineage}; "
            "components are never paired across lineages"
        )
    if expected_period is not None and int(manifest["period"]) != int(expected_period):
        raise Phase18G3Error(f"bundle {target} is period {manifest['period']}, not {expected_period}")
    if expected_matched_digest is not None and manifest["matched_digest"] != expected_matched_digest:
        raise Phase18G3Error(f"bundle {target} was written under a different matched configuration")
    # The setup half must be internally consistent with its own manifest.
    setup_manifest = json.loads((target / SETUP_DIRECTORY / "manifest.json").read_text())
    if setup_manifest["checkpoint_version"] != SETUP_CHECKPOINT_VERSION:
        raise Phase18G3Error(f"bundle {target}: setup checkpoint version {setup_manifest['checkpoint_version']!r}")
    if setup_manifest["run_id"] != manifest["run_id"]:
        raise Phase18G3Error(f"bundle {target}: the setup half names run {setup_manifest['run_id']!r}")
    if manifest["components"]["setup_raw"]["state_digest"] != setup_manifest["raw"]["state_digest"]:
        raise Phase18G3Error(f"bundle {target}: the setup raw digest disagrees with the setup manifest")
    if manifest["components"]["setup_ema"]["state_digest"] != setup_manifest["ema"]["state_digest"]:
        raise Phase18G3Error(f"bundle {target}: the setup EMA digest disagrees with the setup manifest")
    if not manifest["setup_updates_enabled"]:
        counters = manifest["counters"]
        if counters["setup_updates"] or counters["setup_optimizer_steps"] or counters["setup_ema_updates"]:
            raise Phase18G3Error(f"bundle {target}: a control bundle records setup updates")
        if setup_manifest["raw"]["state_digest"] != setup_manifest["ema"]["state_digest"]:
            raise Phase18G3Error(f"bundle {target}: a control bundle's raw and EMA setup weights differ")
    return manifest


def load_c1_payload(directory) -> dict:
    """The validated warmstart checkpoint payload of a verified bundle."""
    payload = read_warmstart_payload(Path(directory) / C1_NAME)
    validate_warmstart_payload(payload, source=str(Path(directory) / C1_NAME))
    return payload


def load_setup_trainer(directory, config: PilotConfig, *, device: "str | None" = None) -> tuple:
    """The bundle's setup trainer (raw, optimizer, EMA) through the accepted loader."""
    manifest = verify_bundle(directory, expected_run_id=config.run_id, expected_lineage=config.lineage)
    trainer, setup_manifest = SetupTrainer.load_checkpoint(
        Path(directory) / SETUP_DIRECTORY,
        config.setup_config(),
        namespace=config.namespace,
        seed_index=config.seed_index,
        device=device or config.setup_device,
    )
    if state_dict_digest(trainer.model) != manifest["components"]["setup_raw"]["state_digest"]:
        raise Phase18G3Error("the restored setup raw weights do not reproduce the bundle's digest")
    return trainer, setup_manifest


def load_collector_state(directory) -> dict:
    payload = torch.load(Path(directory) / COLLECTOR_NAME, map_location="cpu", weights_only=False)
    for key in ("collector", "buffer", "buffer_state_digest"):
        if key not in payload:
            raise Phase18G3Error(f"collector state is missing {key!r}")
    return payload


def bundle_components_digest(manifest: dict) -> str:
    """One digest over every bound component digest, for cross-bundle comparison."""
    hasher = hashlib.sha256()
    for name in COMPONENT_NAMES:
        entry = manifest["components"][name]
        hasher.update(f"{name}|{entry['sha256']}|{entry.get('state_digest', '')}\n".encode())
    return hasher.hexdigest()


def compare_bundles(manifest_a: dict, manifest_b: dict) -> dict:
    """Which components two bundles share, by state digest where one exists."""
    same = {}
    for name in COMPONENT_NAMES:
        a, b = manifest_a["components"][name], manifest_b["components"][name]
        key = "state_digest" if "state_digest" in a else "sha256"
        same[name] = a[key] == b[key]
    return same


__all__ = [
    "C1_NAME",
    "COLLECTOR_NAME",
    "COMPONENT_NAMES",
    "MANIFEST_NAME",
    "SETUP_DIRECTORY",
    "bundle_components_digest",
    "bundle_identity",
    "c1_state_digest",
    "compare_bundles",
    "load_c1_payload",
    "load_collector_state",
    "load_setup_trainer",
    "read_manifest",
    "verify_bundle",
    "write_bundle",
]
