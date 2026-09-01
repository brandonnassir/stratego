"""Phase 17 Agent 5: publication, transfer and receipt protocol.

Specification source: Agent 5 instruction section 4.

The operator chose authenticated pull (the contract's second-preference
topology): the trainer publishes into an outbox and never blocks on the
network; the MacBook connects in, takes what it finds, and returns receipts.
A pull worker gets backlog semantics for free -- if it is late, or was asleep,
or the network dropped, it simply finds more work waiting, all of it still
bound to the candidate identity it was published under.

The SSH surface is one forced command
--------------------------------------
The evaluating machine's key is pinned with
`restrict,from="<ip>",command="<this endpoint>"`, so it cannot open a shell and
cannot reach the repository. Five verbs exist -- `ping`, `list`, `get`,
`get-static`, `put-receipt` -- and every path argument is matched against
`SAFE_NAME` and re-checked to resolve inside its own directory. A verb that is
not in the table is refused rather than passed to a shell.

Nothing is visible until it is complete
----------------------------------------
Publication stages under a `.partial` name in the same directory, fsyncs, and
`os.replace`s onto the final name; the index is rewritten the same way and only
*after* the bundle is in place. A puller that lists between the two sees a
smaller index, never a name whose bytes are still arriving. The remote stages
its download the same way and verifies sha256 before it renames -- so a
truncated transfer is a retry, not a wrong number.

A late result is late, never relabelled
----------------------------------------
The ledger records the candidate a result belongs to and the nominal cadence
slot that candidate was published for. Contract section 11: an old result is
never re-attributed to a newer timestamp, and a missed slot is recorded as
skipped rather than backfilled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .contract import (
    TRANSPORT_PROTOCOL_VERSION,
    Phase17TransportError,
    file_sha256,
    json_digest,
)

#: Every name that may appear in a verb argument.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: The transport tree, deliberately OUTSIDE the repository: the forced command
#: cannot reach source, checkpoints or reports even by defect.
DEFAULT_TRANSPORT_ROOT = Path.home() / "stratego_phase17_transport"

OUTBOX = "outbox"
INBOX = "inbox"
STATIC = "static"
INDEX_NAME = "index.json"
LEDGER_NAME = "ledger.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def transport_paths(root: "Path | str" = DEFAULT_TRANSPORT_ROOT) -> dict:
    root = Path(root)
    return {
        "root": root,
        "outbox": root / OUTBOX,
        "inbox": root / INBOX,
        "static": root / STATIC,
        "index": root / OUTBOX / INDEX_NAME,
        "ledger": root / LEDGER_NAME,
    }


def ensure_transport(root: "Path | str" = DEFAULT_TRANSPORT_ROOT) -> dict:
    paths = transport_paths(root)
    for key in ("root", "outbox", "inbox", "static"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if not paths["index"].is_file():
        write_atomic_json(paths["index"], empty_index())
    return paths


def empty_index() -> dict:
    return {
        "protocol": TRANSPORT_PROTOCOL_VERSION,
        "updated_utc": utc_now(),
        "candidates": [],
        "static": [],
    }


def safe_child(directory: Path, name: str) -> Path:
    """A name-checked path that must resolve inside `directory`."""
    if not SAFE_NAME.match(name or ""):
        raise Phase17TransportError(f"unsafe name {name!r}")
    target = (directory / name).resolve()
    if directory.resolve() not in target.parents:
        raise Phase17TransportError(f"{name!r} escapes {directory}")
    return target


def write_atomic_bytes(target: "Path | str", data: bytes) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".partial"
    )
    with os.fdopen(handle, "wb") as sink:
        sink.write(data)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, target)
    return target


def write_atomic_json(target: "Path | str", payload: dict) -> Path:
    return write_atomic_bytes(
        target, (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    )


def append_ledger(root: "Path | str", row: dict) -> None:
    paths = transport_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    with paths["ledger"].open("a") as handle:
        handle.write(json.dumps(dict(row, recorded_utc=utc_now()), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_ledger(root: "Path | str") -> list:
    paths = transport_paths(root)
    if not paths["ledger"].is_file():
        return []
    return [
        json.loads(line)
        for line in paths["ledger"].read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Source side: publish
# ---------------------------------------------------------------------------


def publish_candidate(
    bundle_path: "Path | str",
    *,
    root: "Path | str" = DEFAULT_TRANSPORT_ROOT,
    manifest: "dict | None" = None,
    nominal_slot_seconds: "int | None" = None,
    pack_digest: str = "",
) -> dict:
    """Copy one immutable candidate into the outbox and index it.

    Staged, fsynced, renamed, and only then indexed -- so `list` can never name
    a candidate whose bytes are still in flight.
    """
    paths = ensure_transport(root)
    source = Path(bundle_path)
    if not source.is_file():
        raise Phase17TransportError(f"no bundle at {source}")
    candidate_id = source.stem
    if not SAFE_NAME.match(candidate_id):
        raise Phase17TransportError(f"unsafe candidate id {candidate_id!r}")

    target = paths["outbox"] / f"{candidate_id}.pt"
    digest = file_sha256(source)
    if target.exists():
        existing = file_sha256(target)
        if existing != digest:
            raise Phase17TransportError(
                f"{candidate_id} is already published with sha256 {existing}; "
                f"this bundle is {digest}. Candidates are immutable and a name "
                "is never reused."
            )
    else:
        handle, temporary = tempfile.mkstemp(
            dir=str(paths["outbox"]), prefix=target.name + ".", suffix=".partial"
        )
        os.close(handle)
        try:
            shutil.copyfile(source, temporary)
            with open(temporary, "rb") as sink:
                os.fsync(sink.fileno())
            staged = file_sha256(temporary)
            if staged != digest:
                raise Phase17TransportError(
                    f"{candidate_id}: staged copy digests to {staged}, source is {digest}"
                )
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    record = {
        "candidate_id": candidate_id,
        "file": target.name,
        "file_sha256": digest,
        "bytes": target.stat().st_size,
        "published_utc": utc_now(),
        "pack_digest": pack_digest,
        "nominal_slot_seconds": nominal_slot_seconds,
        "run_id": (manifest or {}).get("run_id"),
        "candidate_index": (manifest or {}).get("candidate_index"),
        "iteration": (manifest or {}).get("iteration"),
        "manifest_digest": (manifest or {}).get("_digest"),
        "elapsed_active_training_seconds": (manifest or {}).get(
            "elapsed_active_training_seconds"
        ),
    }
    index = json.loads(paths["index"].read_text())
    index["candidates"] = [
        row for row in index.get("candidates", []) if row["candidate_id"] != candidate_id
    ] + [record]
    index["candidates"].sort(key=lambda row: row["candidate_id"])
    index["updated_utc"] = utc_now()
    index["static"] = [
        {
            "name": item.name,
            "file_sha256": file_sha256(item),
            "bytes": item.stat().st_size,
        }
        for item in sorted(paths["static"].iterdir())
        if item.is_file() and not item.name.endswith(".partial")
    ]
    index["protocol"] = TRANSPORT_PROTOCOL_VERSION
    write_atomic_json(paths["index"], index)
    append_ledger(root, {"event": "published", **record})
    return record


def refresh_static_index(root: "Path | str" = DEFAULT_TRANSPORT_ROOT) -> dict:
    paths = ensure_transport(root)
    index = json.loads(paths["index"].read_text())
    index["static"] = [
        {
            "name": item.name,
            "file_sha256": file_sha256(item),
            "bytes": item.stat().st_size,
        }
        for item in sorted(paths["static"].iterdir())
        if item.is_file() and not item.name.endswith(".partial")
    ]
    index["updated_utc"] = utc_now()
    index["protocol"] = TRANSPORT_PROTOCOL_VERSION
    write_atomic_json(paths["index"], index)
    return index


# ---------------------------------------------------------------------------
# Source side: receipts
# ---------------------------------------------------------------------------


def ingest_receipt(
    payload: bytes,
    *,
    root: "Path | str" = DEFAULT_TRANSPORT_ROOT,
    candidate_id: str,
) -> dict:
    """Store a returned receipt atomically and verify it against the index."""
    from .evaluator import verify_receipt

    paths = ensure_transport(root)
    if not SAFE_NAME.match(candidate_id):
        raise Phase17TransportError(f"unsafe candidate id {candidate_id!r}")
    try:
        receipt = json.loads(payload.decode())
    except Exception as error:  # noqa: BLE001
        raise Phase17TransportError(f"receipt is not JSON: {error}") from error
    if receipt.get("candidate_id") != candidate_id:
        raise Phase17TransportError(
            f"receipt names candidate {receipt.get('candidate_id')!r} but was "
            f"returned as {candidate_id!r}; refusing a mis-attributed result"
        )

    index = json.loads(paths["index"].read_text())
    published = {row["candidate_id"]: row for row in index.get("candidates", [])}
    expected = {}
    if candidate_id in published:
        expected["bundle_file_sha256"] = published[candidate_id]["file_sha256"]
        if published[candidate_id].get("pack_digest"):
            expected["benchmark_pack_digest"] = published[candidate_id]["pack_digest"]
        if published[candidate_id].get("run_id"):
            expected["run_id"] = published[candidate_id]["run_id"]
    findings = verify_receipt(receipt, expected=expected if receipt.get("status") == "ok" else None)

    target = paths["inbox"] / f"{candidate_id}.receipt.json"
    write_atomic_bytes(target, payload)
    row = {
        "event": "receipt",
        "candidate_id": candidate_id,
        "status": receipt.get("status"),
        "eligible": findings["eligible"],
        "mismatches": findings["mismatches"],
        "receipt_digest": receipt.get("receipt_digest"),
        "result_digest": receipt.get("result_digest"),
        "runtime_seconds": receipt.get("runtime_seconds"),
        "evaluator_host": (receipt.get("host_identity") or {}).get("hostname"),
        "path": str(target),
    }
    append_ledger(root, row)
    return row


def queue_status(root: "Path | str" = DEFAULT_TRANSPORT_ROOT) -> dict:
    """Explicit backlog: published, receipted, outstanding, refused."""
    paths = ensure_transport(root)
    index = json.loads(paths["index"].read_text())
    published = [row["candidate_id"] for row in index.get("candidates", [])]
    receipts = {}
    for item in sorted(paths["inbox"].glob("*.receipt.json")):
        receipts[item.name[: -len(".receipt.json")]] = json.loads(item.read_text())
    outstanding = [name for name in published if name not in receipts]
    refused = [
        name for name, receipt in receipts.items() if receipt.get("status") != "ok"
    ]
    return {
        "protocol": TRANSPORT_PROTOCOL_VERSION,
        "published": len(published),
        "receipted": len(receipts),
        "outstanding": outstanding,
        "backlog_depth": len(outstanding),
        "refused": refused,
        "note": (
            "a late result stays bound to its own candidate; a missed slot is "
            "recorded as skipped and is never backfilled from a later reading"
        ),
        "checked_utc": utc_now(),
    }


__all__ = [
    "DEFAULT_TRANSPORT_ROOT",
    "SAFE_NAME",
    "append_ledger",
    "empty_index",
    "ensure_transport",
    "ingest_receipt",
    "publish_candidate",
    "queue_status",
    "read_ledger",
    "refresh_static_index",
    "safe_child",
    "transport_paths",
    "utc_now",
    "write_atomic_bytes",
    "write_atomic_json",
]
