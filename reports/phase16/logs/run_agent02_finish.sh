#!/bin/zsh
# Phase 16 Agent 2 finish chain: idle suite + CLI verify (under the lock),
# then the adversarial-delta benchscore at MEDIUM (takes its own lock).
cd /Users/brandonwashington/Dev/Github/stratego/gpt_agent
PY=.venv/bin/python
LOCK=checkpoints/phase16/COMPUTE_LOCK.json

echo "[$(date -u +%FT%TZ)] finish chain starting; waiting for a free lock"
while [ -f $LOCK ]; do sleep 60; done
cat > $LOCK <<EOF
{"agent": 2, "task": "agent02 idle full suite + cli verify", "started_utc": "$(date -u +%FT%TZ)", "expected_hours": 0.4, "pid": $$}
EOF
echo "[$(date -u +%FT%TZ)] lock taken for the idle window (pid $$)"

echo "[$(date -u +%FT%TZ)] running the full pytest suite (idle machine)"
START=$(date +%s)
$PY -m pytest -q -p no:cacheprovider > reports/phase16/logs/full_suite_agent02.txt 2>&1
SUITE_RC=$?
END=$(date +%s)
tail -3 reports/phase16/logs/full_suite_agent02.txt
$PY - "$START" "$END" <<'PYEOF'
import json, re, sys, time
start, end = int(sys.argv[1]), int(sys.argv[2])
tail = open("reports/phase16/logs/full_suite_agent02.txt").read()[-4000:]
match = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?", tail)
failed = re.search(r"(\d+) failed", tail)
payload = {
    "passed": int(match.group(1)) if match else None,
    "skipped": int(match.group(2) or 0) if match else None,
    "failed": int(failed.group(1)) if failed else 0,
    "minutes": round((end - start) / 60, 1),
    "ran_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)),
    "context": "idle machine, single process, under the compute lock",
    "log": "reports/phase16/logs/full_suite_agent02.txt",
}
open("reports/phase16/agent_02_full_suite.json", "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("suite json:", payload)
PYEOF

echo "[$(date -u +%FT%TZ)] CLI end-to-end verification (varied_fast vs varied_strength)"
$PY scripts/play_phase16.py --red varied_fast --blue varied_strength --setup-seed 3 \
  > reports/phase16/logs/cli_verify.txt 2>&1
CLI_RC=$?
tail -6 reports/phase16/logs/cli_verify.txt
echo "[$(date -u +%FT%TZ)] cli verify rc=$CLI_RC"

rm -f $LOCK
echo "[$(date -u +%FT%TZ)] idle-window lock released"

echo "[$(date -u +%FT%TZ)] benchscore MEDIUM (adversarial delta; takes its own lock)"
$PY scripts/run_phase16_agent02.py --role benchscore --preset MEDIUM --workers 10 --wait-lock 720
BENCH_RC=$?
echo "[$(date -u +%FT%TZ)] benchscore rc=$BENCH_RC"
echo "AGENT02 FINISH COMPLETE (suite=$SUITE_RC cli=$CLI_RC bench=$BENCH_RC)"
