#!/usr/bin/env python3
"""Phase 17 Agent 5: the SSH forced-command endpoint on the training computer.

Pinned in `~/.ssh/authorized_keys` as

```text
restrict,from="<evaluator ip>",command="<interpreter> <this file>" ssh-ed25519 AAAA... phase17-eval
```

so the evaluating MacBook's key can do these five things and nothing else. It
cannot open a shell, cannot run another command, and cannot see the repository:
the transport tree lives outside it.

```text
ping                      protocol + host identity, as JSON
list                      the published index, as JSON
get <candidate_id>        the candidate's bytes, to stdout
get-static <name>         one static payload file's bytes, to stdout
put-receipt <candidate>   read a receipt on stdin, store it atomically, verify
```

Deliberately torch-free
------------------------
Importing `torch` costs 205 MB of resident memory, and this runs on the machine
in the middle of a 12-hour training run, once per candidate, plus every poll.
Nothing here needs a tensor, so nothing here imports one.

Refusals are silent about internals
------------------------------------
An unknown verb, an unsafe name or a path that escapes its directory produces a
short refusal on stderr and a non-zero exit. It does not echo the attempted
path back, and it never falls through to a shell.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stratego.evaluation.phase17.contract import (  # noqa: E402
    TRANSPORT_PROTOCOL_VERSION,
    Phase17TransportError,
)
from stratego.evaluation.phase17.transport import (  # noqa: E402
    DEFAULT_TRANSPORT_ROOT,
    ensure_transport,
    ingest_receipt,
    queue_status,
    safe_child,
)

CHUNK = 1 << 20
MAX_RECEIPT_BYTES = 4 << 20


def _root() -> Path:
    return Path(os.environ.get("PHASE17_TRANSPORT_ROOT", DEFAULT_TRANSPORT_ROOT))


def _emit_json(payload: dict) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


def verb_ping(_args) -> int:
    import platform
    import socket

    return _emit_json({
        "protocol": TRANSPORT_PROTOCOL_VERSION,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "transport_root": str(_root()),
        "queue": queue_status(_root()),
    })


def verb_list(_args) -> int:
    paths = ensure_transport(_root())
    sys.stdout.write(paths["index"].read_text())
    sys.stdout.flush()
    return 0


def _stream(target: Path) -> int:
    if not target.is_file():
        sys.stderr.write("not published\n")
        return 4
    with target.open("rb") as source:
        while True:
            block = source.read(CHUNK)
            if not block:
                break
            sys.stdout.buffer.write(block)
    sys.stdout.buffer.flush()
    return 0


def verb_get(args) -> int:
    if len(args) != 1:
        sys.stderr.write("get takes one candidate id\n")
        return 2
    paths = ensure_transport(_root())
    return _stream(safe_child(paths["outbox"], f"{args[0]}.pt"))


def verb_get_static(args) -> int:
    if len(args) != 1:
        sys.stderr.write("get-static takes one name\n")
        return 2
    paths = ensure_transport(_root())
    return _stream(safe_child(paths["static"], args[0]))


def verb_put_receipt(args) -> int:
    if len(args) != 1:
        sys.stderr.write("put-receipt takes one candidate id\n")
        return 2
    payload = sys.stdin.buffer.read(MAX_RECEIPT_BYTES + 1)
    if len(payload) > MAX_RECEIPT_BYTES:
        sys.stderr.write("receipt too large\n")
        return 5
    row = ingest_receipt(payload, root=_root(), candidate_id=args[0])
    return _emit_json(row)


VERBS = {
    "ping": verb_ping,
    "list": verb_list,
    "get": verb_get,
    "get-static": verb_get_static,
    "put-receipt": verb_put_receipt,
}


def main() -> int:
    original = os.environ.get("SSH_ORIGINAL_COMMAND")
    raw = original if original is not None else " ".join(sys.argv[1:])
    try:
        parts = shlex.split(raw or "")
    except ValueError:
        sys.stderr.write("unparseable command\n")
        return 2
    if not parts:
        sys.stderr.write(f"expected one of: {', '.join(sorted(VERBS))}\n")
        return 2
    handler = VERBS.get(parts[0])
    if handler is None:
        sys.stderr.write("refused\n")
        return 2
    try:
        return int(handler(parts[1:]))
    except Phase17TransportError as error:
        sys.stderr.write(f"refused: {error}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
