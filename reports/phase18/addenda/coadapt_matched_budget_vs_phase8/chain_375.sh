#!/bin/bash
ROOT=/Users/brandonwashington/Dev/Github/stratego/gpt_agent; W=$ROOT/output/phase18/runtime/addendum_coadapt; PY=$ROOT/.venv/bin/python
say(){ echo "[chain375 $(date '+%H:%M:%S')] $*" | tee -a "$W/logs/chain_375.log"; }
cd "$ROOT/output/phase18/worktrees/g3-stage6b"
until [ -f "$W/runtime/control/bundles/bundle_0375/manifest.json" ]; do
  say "resuming lineage toward 375"; caffeinate -i "$PY" "$W/coadapt_lineage.py" --resume --horizon 375 >> "$W/logs/resume_375_stdout.log" 2>&1
  [ -f "$W/runtime/control/bundles/bundle_0375/manifest.json" ] || { say "lineage exited before 375; relaunching from the latest bundle in 60 s"; sleep 60; }
done
until [ -f "$W/analysis_p375.json" ] && grep -q "EVAL P375 DONE" "$W/logs/eval_p375.log" 2>/dev/null; do
  say "evaluating bundle_0375"; caffeinate -i "$PY" "$W/coadapt_eval_final.py" 375 >> "$W/logs/eval_375_stdout.log" 2>&1
  grep -q "EVAL P375 DONE" "$W/logs/eval_p375.log" 2>/dev/null || { say "evaluator exited early; retrying in 60 s"; sleep 60; }
done
touch "$W/DONE_375"; say "ALL DONE"
