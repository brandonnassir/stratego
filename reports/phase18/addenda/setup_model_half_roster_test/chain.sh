#!/bin/bash
ROOT=/Users/brandonwashington/Dev/Github/stratego/gpt_agent; H=$ROOT/output/phase18/runtime/addendum_half_roster; PY=$ROOT/.venv/bin/python
cd "$ROOT/output/phase18/worktrees/g3-stage6b"
say(){ echo "[chain $(date '+%H:%M:%S')] $*" | tee -a "$H/chain.log"; }
for h in A B; do
  until [ -f "$H/$h/ckpt_1024/manifest.json" ]; do
    say "training half $h"; caffeinate -i "$PY" "$H/train_half.py" --half "$h" --periods 1024 >> "$H/$h/train_stdout.log" 2>&1
    [ -f "$H/$h/ckpt_1024/manifest.json" ] || { say "half $h trainer exited early; retrying from checkpoint in 60 s"; sleep 60; }
  done
  until [ "$(ls "$H/$h"/eval/rows_*_ckpt_1024.jsonl 2>/dev/null | wc -l | tr -d ' ')" = "8" ]; do
    say "evaluating half $h"; caffeinate -i "$PY" "$H/evaluate_half.py" --half "$h" 0 256 384 512 640 768 896 1024 >> "$H/$h/eval_stdout.log" 2>&1
    [ "$(ls "$H/$h"/eval/rows_*_ckpt_1024.jsonl 2>/dev/null | wc -l | tr -d ' ')" = "8" ] || { say "half $h evaluator exited early; retrying in 60 s"; sleep 60; }
  done
  say "half $h COMPLETE"
done
touch "$H/ALL_DONE"; say "ALL DONE"
