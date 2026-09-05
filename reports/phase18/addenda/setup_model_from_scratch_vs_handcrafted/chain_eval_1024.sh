#!/bin/bash
ROOT=/Users/brandonwashington/Dev/Github/stratego/gpt_agent
OUT=$ROOT/output/phase18/runtime/addendum_library_setup_from_scratch
until [ -f "$OUT/ckpt_1024/manifest.json" ] && ! pgrep -f "train_from_scratch.py --periods 1024" >/dev/null; do
  if grep -qE "Traceback|RuntimeError" "$OUT/train_stdout.log" 2>/dev/null; then echo "[chain] training failed; not evaluating"; exit 1; fi
  sleep 30
done
echo "[chain $(date '+%H:%M:%S')] ckpt_1024 present and trainer exited; starting evaluation"
cd "$ROOT/output/phase18/worktrees/g3-stage6b"
exec caffeinate -i "$ROOT/.venv/bin/python" "$OUT/evaluate.py" 0 256 384 512 640 768 896 1024
