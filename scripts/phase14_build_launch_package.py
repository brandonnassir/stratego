#!/usr/bin/env python3
"""Phase 14: rebuild the immutable launch package.

Writes `phase14_final_training_config_v1` and `phase14_launch_manifest_v1` from
the **live** modules and the accepted artifacts on disk, and binds the code
revision they were built over.

This is the deliberate act the launch check exists to require. Phase 14 refuses
to launch on a code revision the manifest does not name, and the only way to
name a new one is to run this — after which the pre-launch checklist re-verifies
every identity before anything is started.

Usage:

```text
python scripts/phase14_build_launch_package.py
python scripts/phase14_build_launch_package.py --verify   # check, write nothing
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def main(argv=None) -> int:
    from stratego.training.phase14_launch import (
        Phase14LaunchError,
        assert_bound_launch_code,
        write_launch_package,
    )

    parser = argparse.ArgumentParser(description="Rebuild the Phase 14 launch package")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        try:
            report = assert_bound_launch_code()
        except Phase14LaunchError as error:
            print(json.dumps({"verified": False, "error": str(error)}, indent=2))
            return 1
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    written = write_launch_package()
    print(json.dumps(written, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
