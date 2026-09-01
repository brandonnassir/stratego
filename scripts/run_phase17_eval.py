#!/usr/bin/env python3
"""Phase 17 Agent 5: the external evaluation CLI.

Roles
-----
```text
pack            materialize/verify the composite benchmark (SOURCE side only:
                needs the accepted setup library)
evaluate        verify one candidate bundle and run both lanes; writes a result
                and prints the receipt
fixture         the cross-machine identity fixture (contract section 5)
verify-receipt  re-verify a returned receipt at the source; no result is
                eligible until this passes
```

This file is the only entry point the remote worker calls, so the set of things
that can be asked of the evaluating machine is exactly the set of roles above.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stratego.evaluation.phase17.contract import (  # noqa: E402
    COMPOSITE_PACK_ID,
    LANE_JOINT,
    LANE_MOVE_ONLY,
    Phase17EvaluationError,
)
from stratego.evaluation.phase17.evaluator import (  # noqa: E402
    evaluate_candidate,
    evaluator_source_digest,
    host_identity,
    refusal_receipt,
    verify_receipt,
)
from stratego.evaluation.phase17.fixture import (  # noqa: E402
    DEFAULT_FIXTURE_CASES,
    build_fixture,
    compare_fixtures,
)
from stratego.evaluation.phase17.pack import (  # noqa: E402
    DEFAULT_PACK_PATH,
    build_composite_pack,
    load_composite_pack,
    write_composite_pack,
)


def role_pack(args) -> int:
    if args.verify_only:
        pack = load_composite_pack(args.pack, expected_digest=args.expect_digest)
        print(json.dumps({
            "pack_id": pack["pack_id"],
            "pack_digest": pack["pack_digest"],
            "lanes": {lane: pack["lanes"][lane]["case_count"] for lane in pack["lanes"]},
            "verified": True,
        }, indent=1))
        return 0
    payload = build_composite_pack(root=args.root)
    path = write_composite_pack(payload, args.pack)
    reloaded = load_composite_pack(path, expected_digest=payload["pack_digest"])
    print(json.dumps({
        "pack_id": reloaded["pack_id"],
        "pack_digest": reloaded["pack_digest"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "library_redraw_proof": reloaded["library_redraw_proof"],
        "lanes": {lane: reloaded["lanes"][lane]["case_count"] for lane in reloaded["lanes"]},
    }, indent=1))
    return 0


def role_evaluate(args) -> int:
    lanes = tuple(args.lanes.split(",")) if args.lanes else (LANE_MOVE_ONLY, LANE_JOINT)
    for lane in lanes:
        if lane not in (LANE_MOVE_ONLY, LANE_JOINT):
            raise SystemExit(f"unknown lane {lane!r}")
    try:
        receipt = evaluate_candidate(
            args.bundle,
            root=args.root,
            pack_path=args.pack,
            expected_pack_digest=args.expect_pack_digest,
            results_directory=args.results,
            workers=args.workers,
            expected_file_sha256=args.expect_file_sha256,
            expected_run_id=args.expect_run_id,
            expected_candidate_id=args.expect_candidate_id,
            lanes=lanes,
            payload_digest=args.payload_digest,
        )
    except Phase17EvaluationError as error:
        receipt = refusal_receipt(
            candidate_id=args.expect_candidate_id or Path(args.bundle).stem,
            reason=type(error).__name__,
            detail=str(error),
            root=args.root,
        )
        Path(args.results).mkdir(parents=True, exist_ok=True)
        target = Path(args.results) / f"{receipt['candidate_id']}.result.json"
        target.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=1, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


def role_verify_receipt(args) -> int:
    receipt = json.loads(Path(args.receipt).read_text())
    expected = json.loads(args.expect) if args.expect else None
    findings = verify_receipt(receipt, expected=expected)
    print(json.dumps({
        "candidate_id": receipt.get("candidate_id"),
        "status": receipt.get("status"),
        **findings,
    }, indent=1))
    return 0 if findings["eligible"] else 3


def role_fixture(args) -> int:
    fixture = build_fixture(
        args.bundle,
        root=args.root,
        pack_path=args.pack,
        expected_pack_digest=args.expect_pack_digest,
        cases=args.cases,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "fixture_digest": fixture["fixture_digest"],
        "host": fixture["host_identity"]["hostname"],
        "torch": fixture["host_identity"]["torch"],
        "python": fixture["host_identity"]["python"],
        "positions_probed": len(fixture["logit_probes"]),
        "setup_boards": len(fixture["setup_tokens"]),
        "games": len(fixture["games"]),
        "out": str(args.out),
    }, indent=1))
    return 0


def role_compare_fixture(args) -> int:
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    findings = compare_fixtures(left, right)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(findings, indent=1, sort_keys=True) + "\n")
    print(json.dumps(findings, indent=1, sort_keys=True))
    return 0 if findings["pass"] else 4


def role_identity(args) -> int:
    print(json.dumps({
        "host_identity": host_identity(),
        "evaluator_source_digest": evaluator_source_digest(root=args.root),
        "composite_pack_id": COMPOSITE_PACK_ID,
    }, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    sub = parser.add_subparsers(dest="role", required=True)

    pack = sub.add_parser("pack", help="materialize or verify the composite pack")
    pack.add_argument("--pack", default=str(DEFAULT_PACK_PATH))
    pack.add_argument("--verify-only", action="store_true")
    pack.add_argument("--expect-digest", default=None)
    pack.set_defaults(handler=role_pack)

    run = sub.add_parser("evaluate", help="evaluate one candidate bundle")
    run.add_argument("bundle")
    run.add_argument("--pack", default=str(DEFAULT_PACK_PATH))
    run.add_argument("--expect-pack-digest", default=None)
    run.add_argument("--results", default="results")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--lanes", default=None)
    run.add_argument("--expect-file-sha256", default=None)
    run.add_argument("--expect-run-id", default=None)
    run.add_argument("--expect-candidate-id", default=None)
    run.add_argument("--payload-digest", default=None)
    run.set_defaults(handler=role_evaluate)

    check = sub.add_parser("verify-receipt", help="re-verify a returned receipt")
    check.add_argument("receipt")
    check.add_argument("--expect", default=None, help="JSON of fields that must match")
    check.set_defaults(handler=role_verify_receipt)

    fix = sub.add_parser("fixture", help="build the cross-machine identity fixture")
    fix.add_argument("bundle")
    fix.add_argument("--pack", default=str(DEFAULT_PACK_PATH))
    fix.add_argument("--expect-pack-digest", default=None)
    fix.add_argument("--cases", type=int, default=DEFAULT_FIXTURE_CASES)
    fix.add_argument("--out", required=True)
    fix.set_defaults(handler=role_fixture)

    cmp_ = sub.add_parser("compare-fixture", help="compare two machines' fixtures")
    cmp_.add_argument("left")
    cmp_.add_argument("right")
    cmp_.add_argument("--out", default=None)
    cmp_.set_defaults(handler=role_compare_fixture)

    who = sub.add_parser("identity", help="print this machine's evaluator identity")
    who.set_defaults(handler=role_identity)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
