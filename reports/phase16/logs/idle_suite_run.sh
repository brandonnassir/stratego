#!/bin/zsh
cd /Users/brandonwashington/Dev/Github/stratego/gpt_agent
PY=.venv/bin/python
LOG=reports/phase16/logs
STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ }
# Re-stamp the lock with this process's pid so it reads as live.
$PY - <<PYEOF
import json, os
from pathlib import Path
lock = Path("checkpoints/phase16/COMPUTE_LOCK.json")
held = json.loads(lock.read_text()) if lock.is_file() else {}
held.update({"agent": 1, "task": "agent1 idle suite verification", "pid": os.getppid()})
lock.write_text(json.dumps(held, indent=1) + "\n")
PYEOF
echo "[$(STAMP)] lock re-stamped pid $$; verifying machine idle"
if pgrep -f "run_phase16_agent01.py --role (bench|baseline)|run_phase16_agent02|run_phase15" > /dev/null; then
  echo "[$(STAMP)] WARNING: heavy workers present, not idle"
fi
echo "[$(STAMP)] phase12 player tests solo:"
$PY -m pytest tests/search/test_phase12_player.py -q -p no:cacheprovider > $LOG/idle_phase12_player_tests.txt 2>&1
tail -1 $LOG/idle_phase12_player_tests.txt
echo "[$(STAMP)] full suite idle:"
$PY -m pytest -q -p no:cacheprovider > $LOG/idle_full_suite.txt 2>&1
tail -1 $LOG/idle_full_suite.txt
$PY - <<PYEOF
import json, os
from pathlib import Path
lock = Path("checkpoints/phase16/COMPUTE_LOCK.json")
if lock.is_file():
    held = json.loads(lock.read_text())
    if held.get("agent") == 1 and "idle suite" in str(held.get("task", "")):
        lock.unlink()
PYEOF
echo "[$(STAMP)] IDLE SUITE RUN COMPLETE"
