#!/usr/bin/env python3
"""Phase 17 Agent 5: the evaluating MacBook's pull worker.

Specification source: Agent 5 instruction sections 4 and 7.

The operator chose authenticated pull, so this is the only moving part on the
evaluating machine. It connects to the trainer's forced-command endpoint, takes
whatever candidates are published and not yet receipted, evaluates them in
publication order, and returns a receipt for each.

```text
bootstrap   fetch and verify the one-time static payload
once        one poll cycle: list, fetch, evaluate, return receipts
poll        `once` on an interval, until stopped
status      what this machine has done and what it still owes
```

Backlog is explicit and never relabelled
-----------------------------------------
If this machine was asleep, offline, or simply slow, `list` returns more work
than one slot's worth and the worker grinds through it *in publication order*,
each result still bound to the candidate identity it was published under.
Contract section 11 forbids attributing an old result to a newer nominal time,
so the worker has no code path that could: a receipt carries the candidate id
it evaluated, and the trainer refuses a receipt whose candidate id does not
match the name it was returned under.

Every failure produces a receipt
---------------------------------
A refused bundle, a hash mismatch, a disk-full, an evaluation crash: each ends
as a `status: refused` receipt with its reason, returned like any other. An
absent result is never read as a pass -- Agent 7 sees a refusal row, not
silence.

Nothing here may change the benchmark
--------------------------------------
Section 7: after the gate the worker may retry transport or evaluation, but may
not change the pack, the worker count, the inference mode, the weights or the
candidate identity. The pack digest it was bootstrapped with is pinned in its
state file and passed to every evaluation as `--expect-pack-digest`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stratego.evaluation.phase17.contract import (  # noqa: E402
    TRANSPORT_PROTOCOL_VERSION,
    Phase17EvaluationError,
    file_sha256,
)

DEFAULT_STATE = Path.home() / "phase17_eval"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Endpoint:
    """The trainer, reachable only through its forced command."""

    def __init__(self, host: str, user: str, key: str, port: int = 22) -> None:
        self.host, self.user, self.key, self.port = host, user, key, int(port)

    def _base(self) -> list:
        return ["ssh", *SSH_OPTIONS, "-p", str(self.port), "-i", self.key,
                f"{self.user}@{self.host}"]

    def call(self, verb: str, *, stdin: "bytes | None" = None, binary: bool = False):
        result = subprocess.run(
            [*self._base(), verb], input=stdin, capture_output=True, timeout=600
        )
        if result.returncode != 0:
            raise Phase17EvaluationError(
                f"endpoint {verb!r} failed ({result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout if binary else json.loads(result.stdout.decode())

    def stream_to(self, verb: str, target: Path) -> int:
        """Fetch bytes into `target`, which the caller stages and renames."""
        with target.open("wb") as sink:
            result = subprocess.run(
                [*self._base(), verb], stdout=sink, stderr=subprocess.PIPE, timeout=1800
            )
        if result.returncode != 0:
            target.unlink(missing_ok=True)
            raise Phase17EvaluationError(
                f"endpoint {verb!r} failed ({result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return target.stat().st_size


def load_state(directory: Path) -> dict:
    path = directory / "state.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"protocol": TRANSPORT_PROTOCOL_VERSION, "candidates": {}, "pack_digest": None}


def save_state(directory: Path, state: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    staging = path.with_suffix(".json.partial")
    staging.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    staging.replace(path)


def log(directory: Path, row: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.log.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(row, utc=utc_now()), sort_keys=True) + "\n")
        handle.flush()


def fetch_candidate(endpoint: Endpoint, record: dict, directory: Path) -> Path:
    """Stage, verify, then publish locally. A truncated pull is a retry."""
    bundles = directory / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    final = bundles / f"{record['candidate_id']}.pt"
    if final.is_file() and file_sha256(final) == record["file_sha256"]:
        return final
    staging = bundles / f"{record['candidate_id']}.pt.partial"
    staging.unlink(missing_ok=True)
    size = endpoint.stream_to(f"get {record['candidate_id']}", staging)
    if size != int(record["bytes"]):
        staging.unlink(missing_ok=True)
        raise Phase17EvaluationError(
            f"{record['candidate_id']}: fetched {size} bytes, index says "
            f"{record['bytes']}; partial transfer"
        )
    observed = file_sha256(staging)
    if observed != record["file_sha256"]:
        staging.unlink(missing_ok=True)
        raise Phase17EvaluationError(
            f"{record['candidate_id']}: fetched sha256 {observed}, index says "
            f"{record['file_sha256']}"
        )
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, final)
    return final


def evaluate(record: dict, bundle: Path, directory: Path, args) -> dict:
    """Run the evaluator out of process, so a crash is a refusal not a stop."""
    results = directory / "results"
    results.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts" / "run_phase17_eval.py"), "evaluate",
        str(bundle),
        "--pack", args.pack,
        "--results", str(results),
        "--workers", str(args.workers),
        "--expect-file-sha256", record["file_sha256"],
        "--expect-candidate-id", record["candidate_id"],
    ]
    if args.expect_pack_digest:
        command += ["--expect-pack-digest", args.expect_pack_digest]
    if record.get("run_id"):
        command += ["--expect-run-id", record["run_id"]]
    subprocess.run(command, capture_output=True, timeout=args.timeout)
    target = results / f"{record['candidate_id']}.result.json"
    if not target.is_file():
        from stratego.evaluation.phase17.evaluator import refusal_receipt

        receipt = refusal_receipt(
            candidate_id=record["candidate_id"],
            reason="EvaluationProducedNoResult",
            detail="the evaluator exited without writing a result file",
            root=str(ROOT),
        )
        target.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return json.loads(target.read_text())


def role_bootstrap(args) -> int:
    directory = Path(args.state)
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = Endpoint(args.host, args.user, args.key, args.port)
    ping = endpoint.call("ping")
    if ping.get("protocol") != TRANSPORT_PROTOCOL_VERSION:
        raise SystemExit(f"endpoint speaks {ping.get('protocol')!r}, worker speaks "
                         f"{TRANSPORT_PROTOCOL_VERSION!r}")
    index = endpoint.call("list")
    static = directory / "static"
    static.mkdir(parents=True, exist_ok=True)
    fetched = []
    for item in index.get("static", []):
        target = static / item["name"]
        if target.is_file() and file_sha256(target) == item["file_sha256"]:
            fetched.append({"name": item["name"], "cached": True})
            continue
        staging = target.with_suffix(target.suffix + ".partial")
        endpoint.stream_to(f"get-static {item['name']}", staging)
        observed = file_sha256(staging)
        if observed != item["file_sha256"]:
            staging.unlink(missing_ok=True)
            raise SystemExit(f"{item['name']}: sha256 {observed} != {item['file_sha256']}")
        os.replace(staging, target)
        fetched.append({"name": item["name"], "bytes": target.stat().st_size})
    state = load_state(directory)
    state["endpoint"] = {"host": args.host, "user": args.user, "port": args.port}
    state["bootstrapped_utc"] = utc_now()
    state["pack_digest"] = args.expect_pack_digest
    save_state(directory, state)
    print(json.dumps({"ping": ping, "static": fetched, "state": str(directory)}, indent=1))
    return 0


def role_once(args) -> int:
    directory = Path(args.state)
    endpoint = Endpoint(args.host, args.user, args.key, args.port)
    state = load_state(directory)
    index = endpoint.call("list")
    published = index.get("candidates", [])
    done = state.setdefault("candidates", {})

    outstanding = [row for row in published if done.get(row["candidate_id"], {}).get("returned") is not True]
    summary = {"polled_utc": utc_now(), "published": len(published),
               "backlog_depth": len(outstanding), "processed": [], "failed": []}

    for record in outstanding:
        candidate = record["candidate_id"]
        try:
            bundle = fetch_candidate(endpoint, record, directory)
            receipt = evaluate(record, bundle, directory, args)
            payload = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode()
            row = endpoint.call(f"put-receipt {candidate}", stdin=payload)
            done[candidate] = {
                "returned": True, "status": receipt.get("status"),
                "eligible": row.get("eligible"),
                "runtime_seconds": receipt.get("runtime_seconds"),
                "returned_utc": utc_now(),
            }
            summary["processed"].append({
                "candidate_id": candidate, "status": receipt.get("status"),
                "eligible": row.get("eligible"),
                "runtime_seconds": receipt.get("runtime_seconds"),
            })
            log(directory, {"event": "returned", "candidate_id": candidate, **done[candidate]})
        except Exception as error:  # noqa: BLE001 - every failure becomes visible
            attempts = done.get(candidate, {}).get("attempts", 0) + 1
            done[candidate] = {"returned": False, "attempts": attempts,
                               "last_error": str(error), "last_attempt_utc": utc_now()}
            summary["failed"].append({"candidate_id": candidate, "attempts": attempts,
                                      "error": str(error)})
            log(directory, {"event": "failed", "candidate_id": candidate, "error": str(error)})
        save_state(directory, state)

    summary["still_outstanding"] = [
        row["candidate_id"] for row in published
        if done.get(row["candidate_id"], {}).get("returned") is not True
    ]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def role_poll(args) -> int:
    while True:
        try:
            role_once(args)
        except Exception as error:  # noqa: BLE001 - a poll failure is never fatal
            log(Path(args.state), {"event": "poll_error", "error": str(error)})
            print(json.dumps({"poll_error": str(error), "utc": utc_now()}))
        time.sleep(max(int(args.interval), 5))


def role_status(args) -> int:
    directory = Path(args.state)
    state = load_state(directory)
    done = state.get("candidates", {})
    print(json.dumps({
        "state": str(directory),
        "pack_digest": state.get("pack_digest"),
        "bootstrapped_utc": state.get("bootstrapped_utc"),
        "returned": sum(1 for row in done.values() if row.get("returned")),
        "outstanding": [name for name, row in sorted(done.items()) if not row.get("returned")],
        "refused": [name for name, row in sorted(done.items()) if row.get("status") == "refused"],
    }, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--key", default=str(Path.home() / ".ssh" / "phase17_eval"))
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--pack", default="data/phase17/phase17_composite_benchmark_v1.json")
    parser.add_argument("--expect-pack-digest", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    sub = parser.add_subparsers(dest="role", required=True)
    for name, handler in (("bootstrap", role_bootstrap), ("once", role_once),
                          ("status", role_status)):
        block = sub.add_parser(name)
        block.set_defaults(handler=handler)
    poll = sub.add_parser("poll")
    poll.add_argument("--interval", type=int, default=60)
    poll.set_defaults(handler=role_poll)
    args = parser.parse_args()
    if args.role in ("bootstrap", "once", "poll") and not (args.host and args.user):
        raise SystemExit("--host and --user are required for this role")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
