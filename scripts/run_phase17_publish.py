#!/usr/bin/env python3
"""Phase 17 Agent 5: the training computer's publication side.

```text
static     build the one-time static payload into the transport outbox
publish    publish one immutable candidate bundle for the evaluator to pull
status     published / receipted / outstanding, with the refusal list
ledger     the append-only transport ledger
key-line   the exact authorized_keys line for the evaluator's public key
```

Agent 7 uses `publish` after each 30-minute export and `status` to see backlog.
Nothing here evaluates anything, and nothing here imports torch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stratego.evaluation.phase17.contract import file_sha256, json_digest  # noqa: E402
from stratego.evaluation.phase17.opponents import (  # noqa: E402
    NEURAL_OPPONENT_FILES,
    verify_opponent_files,
)
from stratego.evaluation.phase17.pack import DEFAULT_PACK_PATH  # noqa: E402
from stratego.evaluation.phase17.transport import (  # noqa: E402
    DEFAULT_TRANSPORT_ROOT,
    ensure_transport,
    publish_candidate,
    queue_status,
    read_ledger,
    refresh_static_index,
    transport_paths,
)

SOURCE_ARCHIVE = "phase17_eval_source.tar.gz"

#: What the evaluating machine needs from this repository. The whole package
#: rather than a hand-curated import closure: 6.5 MB of .py is cheaper than the
#: risk of a closure that is subtly short of one module.
SOURCE_INCLUDES = ("stratego",)
SOURCE_SCRIPTS = ("run_phase17_eval.py", "run_phase17_eval_worker.py")


def _tar_filter(info: tarfile.TarInfo):
    name = Path(info.name).name
    if "__pycache__" in info.name or name.endswith((".pyc", ".pyo")):
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0  # a byte-identical archive from identical sources
    return info


def role_static(args) -> int:
    paths = ensure_transport(args.transport)
    static = paths["static"]

    archive = static / SOURCE_ARCHIVE
    staging = archive.with_suffix(archive.suffix + ".partial")
    with tarfile.open(staging, "w:gz") as tar:
        for name in SOURCE_INCLUDES:
            tar.add(ROOT / name, arcname=name, filter=_tar_filter)
        for name in SOURCE_SCRIPTS:
            tar.add(ROOT / "scripts" / name, arcname=f"scripts/{name}", filter=_tar_filter)
    staging.replace(archive)

    pack_source = Path(args.pack)
    if not pack_source.is_file():
        raise SystemExit(f"no composite pack at {pack_source}; run `run_phase17_eval.py pack` first")
    (static / pack_source.name).write_bytes(pack_source.read_bytes())

    verify_opponent_files(root=ROOT)
    for name, record in NEURAL_OPPONENT_FILES.items():
        source = ROOT / record["path"]
        (static / source.name).write_bytes(source.read_bytes())

    index = refresh_static_index(args.transport)
    payload_digest = json_digest(
        sorted((item["name"], item["file_sha256"]) for item in index["static"])
    )
    manifest = {
        "payload_digest": payload_digest,
        "files": index["static"],
        "total_bytes": sum(item["bytes"] for item in index["static"]),
        "install": {
            "python": "3.13",
            "pins": ["torch==2.13.0", "numpy==2.5.1"],
            "device": "cpu",
            "layout": "extract the source archive; the pack and opponent files "
                      "go to data/phase17/ and checkpoints/phase15/ respectively",
        },
    }
    (static / "payload_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    index = refresh_static_index(args.transport)
    print(json.dumps({
        "static": str(static),
        "payload_digest": payload_digest,
        "files": index["static"],
        "total_bytes": manifest["total_bytes"],
    }, indent=1))
    return 0


def role_publish(args) -> int:
    manifest = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    record = publish_candidate(
        args.bundle,
        root=args.transport,
        manifest=manifest,
        nominal_slot_seconds=args.slot_seconds,
        pack_digest=args.pack_digest or "",
    )
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


def role_status(args) -> int:
    print(json.dumps(queue_status(args.transport), indent=1, sort_keys=True))
    return 0


def role_ledger(args) -> int:
    for row in read_ledger(args.transport):
        print(json.dumps(row, sort_keys=True))
    return 0


def role_key_line(args) -> int:
    interpreter = args.interpreter or str(ROOT / ".venv" / "bin" / "python")
    endpoint = ROOT / "scripts" / "phase17_transport_endpoint.py"
    line = (
        f'restrict,from="{args.evaluator_ip}",'
        f'command="{interpreter} {endpoint}" {args.public_key.strip()}'
    )
    print(line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", default=str(DEFAULT_TRANSPORT_ROOT))
    sub = parser.add_subparsers(dest="role", required=True)

    static = sub.add_parser("static")
    static.add_argument("--pack", default=str(DEFAULT_PACK_PATH))
    static.set_defaults(handler=role_static)

    publish = sub.add_parser("publish")
    publish.add_argument("bundle")
    publish.add_argument("--manifest", default=None)
    publish.add_argument("--pack-digest", default=None)
    publish.add_argument("--slot-seconds", type=int, default=None)
    publish.set_defaults(handler=role_publish)

    status = sub.add_parser("status")
    status.set_defaults(handler=role_status)

    ledger = sub.add_parser("ledger")
    ledger.set_defaults(handler=role_ledger)

    key = sub.add_parser("key-line")
    key.add_argument("public_key")
    key.add_argument("--evaluator-ip", required=True)
    key.add_argument("--interpreter", default=None)
    key.set_defaults(handler=role_key_line)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
