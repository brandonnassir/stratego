#!/bin/zsh
# Phase 16 Agent 1: idle re-verification of the pytest suite.
# Waits for the Agent 1 pack chain to finish, acquires the compute lock
# (waiting politely on any live holder, e.g. Agent 2's packs), waits until no
# heavy match workers remain, then reruns the phase12 player tests solo and
# the full suite on an idle machine. Results land in reports/phase16/logs/.
cd /Users/brandonwashington/Dev/Github/stratego/gpt_agent
PY=.venv/bin/python
LOG=reports/phase16/logs
STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ }

# 1. Wait for the Agent 1 chain to be done.
while pgrep -f "run_packs.sh" > /dev/null; do sleep 60; done
echo "[$(STAMP)] agent1 pack chain finished"

# 2. Acquire the compute lock, waiting on any live holder.
$PY - <<'EOF'
import json, os, time
from pathlib import Path
lock = Path("checkpoints/phase16/COMPUTE_LOCK.json")

def alive(pid):
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True

while lock.is_file():
    held = json.loads(lock.read_text())
    if held.get("pid") is None or not alive(held["pid"]):
        break
    time.sleep(60)
lock.write_text(json.dumps({
    "agent": 1,
    "task": "agent1 idle suite verification (phase12 player tests + full suite)",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "expected_hours": 0.5,
    "pid": os.getppid(),
}, indent=1) + "\n")
EOF
echo "[$(STAMP)] compute lock acquired for idle suite verification"

# 3. Wait until no heavy match workers remain (idle machine).
while pgrep -f "run_phase16_agent01.py --role (bench|baseline)|run_phase15_agent02|run_phase15_mixture|run_phase16_agent02" > /dev/null; do
  sleep 30
done
sleep 10
echo "[$(STAMP)] machine idle; running phase12 player tests solo"

$PY -m pytest tests/search/test_phase12_player.py -q -p no:cacheprovider > $LOG/idle_phase12_player_tests.txt 2>&1
tail -1 $LOG/idle_phase12_player_tests.txt

echo "[$(STAMP)] running the full suite idle"
$PY -m pytest -q -p no:cacheprovider > $LOG/idle_full_suite.txt 2>&1
tail -1 $LOG/idle_full_suite.txt

# 4. Release the lock (only if it is ours).
$PY - <<'EOF'
import json, os
from pathlib import Path
lock = Path("checkpoints/phase16/COMPUTE_LOCK.json")
if lock.is_file():
    held = json.loads(lock.read_text())
    if held.get("agent") == 1 and "idle suite" in str(held.get("task", "")):
        lock.unlink()
EOF
echo "[$(STAMP)] IDLE SUITE CHECK COMPLETE"
