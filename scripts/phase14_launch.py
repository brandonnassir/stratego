#!/usr/bin/env python3
"""Phase 14: the production launch entry point.

Bound by `reports/phase13/phase14_launch_manifest_v1.json`. This one script is
both the launch procedure and the resume procedure, because they are the same
procedure: the learner always calls `start_or_resume()`, and a run that already
has a valid hot checkpoint is resumed against its **original** window. Nothing
here can create a 168-hour deadline a second time.

Roles
-----

```text
supervisor   the default. Launches the learner, watches it from outside,
             restarts it when that is the right thing to do, and runs pending
             candidate evaluations out of band.
learner      one training process. The thing the supervisor launches.
finalize     one closeout process for a run whose deadline has passed: it
             resumes, takes zero optimizer steps, and finalizes.
```

Usage:

```text
python scripts/phase14_launch.py                      # launch (or resume) Phase 14
python scripts/phase14_launch.py --preflight-only     # check, do not launch
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def log(message: str) -> None:
    print(f"[phase14] {message}", flush=True)


def _storage(args):
    from stratego.training.phase14_storage import Phase14Storage

    if args.external_root is None and args.hot_root is None:
        return Phase14Storage.production()
    return Phase14Storage.under(args.external_root, hot_root=args.hot_root)


# ---------------------------------------------------------------------------
# Role: the learner (the process that trains)
# ---------------------------------------------------------------------------


def learner_main(args) -> int:
    """One Phase 14 training process. Started, killed, restarted; never retuned."""
    from stratego.training.phase14_clock import SystemClock
    from stratego.training.phase14_contract import PRODUCTION_POPULATION
    from stratego.training.phase14_launch import (
        assert_bound_launch_code,
        record_integrity_failure,
    )
    from stratego.training.phase14_runner import (
        MODE_PRODUCTION,
        Phase14IntegrityError,
        Phase14Runner,
    )
    from stratego.training.phase9_trainer import LoaderTopology

    storage = _storage(args)
    # Checked here as well as in the supervisor: a learner started by hand
    # bypasses the supervisor, and the code binding is the whole reason the
    # accepted worker-pool repair can be proved present at all.
    assert_bound_launch_code()
    runner = Phase14Runner(
        storage,
        clock=SystemClock(),
        mode=MODE_PRODUCTION,
        device=args.device,
        inference_device=args.device,
        topology=LoaderTopology(workers=args.loader_workers),
        games_in_flight=args.games_in_flight,
        inference_batch_shape=args.inference_batch_shape,
        population=PRODUCTION_POPULATION,
    )
    try:
        report = runner.start_or_resume()
        log(
            f"pid {os.getpid()} "
            f"{'started' if report.get('started') else 'resumed'} "
            f"deadline {report.get('run_deadline_utc')}"
        )
        if args.role == "finalize":
            # A closeout launch: the deadline has passed and the only thing
            # left is to finalize. `run()` is not called at all, so the "zero
            # optimizer steps" property does not depend on the loop noticing.
            if not runner.controller.expired():
                log("closeout asked for, but the deadline has not passed; refusing")
                return 2
            final = runner.finalize(reason="post-deadline closeout")
            log(f"closed: {final['reason']}")
            return 0
        result = runner.run()
        log(
            f"run stopped because {result['stopped_because']} at step "
            f"{result['global_optimizer_step']} ({result['elapsed_hours']:.2f} h)"
        )
        if runner.controller.expired():
            final = runner.finalize(reason=result["stopped_because"])
            log(f"closed: {final['reason']}")
        else:
            runner.run_manifest(reason=result["stopped_because"])
        return 0
    except Phase14IntegrityError as error:
        # The one failure a restart must not paper over. Recorded on disk so
        # the supervisor refuses to relaunch over it.
        record_integrity_failure(
            storage.external_root, error=str(error), traceback_text=traceback.format_exc()
        )
        log(f"UNRECOVERABLE integrity failure: {error}")
        return 3
    finally:
        if runner.trainer is not None:
            runner.trainer.close()


# ---------------------------------------------------------------------------
# Role: the supervisor
# ---------------------------------------------------------------------------


def supervisor_main(args) -> int:
    from stratego.training.phase14_launch import load_launch_manifest
    from stratego.training.phase14_supervisor import Phase14Supervisor, SupervisorPolicy

    storage = _storage(args)
    manifest = load_launch_manifest()
    supervisor = Phase14Supervisor(
        storage,
        manifest=manifest,
        python=sys.executable,
        learner_script=Path(__file__).resolve(),
        evaluator_script=REPOSITORY / "scripts" / "phase14_evaluate_candidates.py",
        repository=REPOSITORY,
        policy=SupervisorPolicy(
            max_consecutive_restarts=args.max_consecutive_restarts,
            poll_seconds=args.poll_seconds,
        ),
    )
    report = supervisor.preflight()
    log(
        f"preflight passed: revision {report['code']['revision'][:12]} "
        f"code digest {report['code']['code_digest'][:12]} "
        f"manifest {report['manifest_digest'][:12]}"
    )
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    supervisor.launch(reason="Phase 14 launch")
    try:
        final = supervisor.supervise(max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        log("interrupted; asking the learner to stop")
        supervisor.stop_child()
        return 130
    log(f"supervisor finished: {final['state']['stopped_because']}")
    print(json.dumps(final, indent=2, sort_keys=True, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 14 launch / resume")
    parser.add_argument("--role", choices=("supervisor", "learner", "finalize"), default="supervisor")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--hot-root", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--loader-workers", type=int, default=6)
    parser.add_argument("--games-in-flight", type=int, default=96)
    parser.add_argument("--inference-batch-shape", type=int, default=64)
    parser.add_argument("--max-consecutive-restarts", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.role in ("learner", "finalize"):
        return learner_main(args)
    return supervisor_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
