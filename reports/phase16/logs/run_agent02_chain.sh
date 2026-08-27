#!/bin/zsh
# Phase 16 Agent 2 compute chain. Waits for Agent 1's pack chain (priority),
# then runs positions -> stage1 (TINY, MEDIUM) -> stage2 (TINY, MEDIUM) ->
# probe (MEDIUM) -> idle latency, each under the shared COMPUTE_LOCK.
cd /Users/brandonwashington/Dev/Github/stratego/gpt_agent
PY=.venv/bin/python
echo "[$(date -u +%FT%TZ)] agent02 chain starting; waiting for agent 1's pack chain"
while pgrep -f run_packs.sh > /dev/null; do
  if grep -q "PACK CHAIN COMPLETE" reports/phase16/logs/packs_20260825.log 2>/dev/null; then break; fi
  sleep 300
done
echo "[$(date -u +%FT%TZ)] agent 1's chain finished or exited; proceeding"
set -x
$PY scripts/run_phase16_agent02.py --role positions --wait-lock 720 || exit 1
$PY scripts/run_phase16_agent02.py --role stage1 --budget TINY --workers 10 --wait-lock 720 || exit 1
$PY scripts/run_phase16_agent02.py --role stage1 --budget MEDIUM --workers 10 --wait-lock 720 || exit 1
$PY scripts/run_phase16_agent02.py --role stage2 --budget TINY --workers 10 --wait-lock 720 || exit 1
$PY scripts/run_phase16_agent02.py --role stage2 --budget MEDIUM --workers 10 --wait-lock 720 || exit 1
$PY scripts/run_phase16_agent02.py --role probe --preset MEDIUM --workers 10 --wait-lock 720 || exit 1
set +x
# idle window for the latency pilot: lock free, then a settling minute
while [ -f checkpoints/phase16/COMPUTE_LOCK.json ]; do sleep 120; done
sleep 60
set -x
$PY scripts/run_phase16_agent02.py --role latency --threads 1 || exit 1
set +x
echo "AGENT02 CHAIN COMPLETE"
