#!/bin/bash
# Reap orphaned spawn workers every 60 s; if free<15% twice or swap>3GB, kill the running stage
# (the chain relaunches it from its checkpoint / completed arms). Exit when the chain writes ALL_DONE.
ROOT=/Users/brandonwashington/Dev/Github/stratego/gpt_agent; H=$ROOT/output/phase18/runtime/addendum_half_roster; LOG=$H/memory_watch.log
FREE_MIN=15; WARN_AT=30; SWAP_MAX_GB=3; low=0; warned=0
say(){ echo "[memwatch $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
freepct(){ memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%","",$5); print $5}'; }
swapgb(){ sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); printf "%.2f", v/1024}}'; }
reap(){ local n=0; for q in $(ps -eo pid,ppid,command | awk '$2==1 && /multiprocessing.spawn/ {print $1}'); do kill -KILL "$q" 2>/dev/null && n=$((n+1)); done; [ "$n" -gt 0 ] && say "REAPED $n orphaned worker(s)"; }
say "started"
while true; do
  [ -f "$H/ALL_DONE" ] && { say "chain finished; exiting"; exit 0; }
  reap; f=$(freepct); s=$(swapgb); f=${f:-100}; s=${s:-0}
  stage="idle"; pgrep -f "train_half.py" >/dev/null && stage="train"; pgrep -f "evaluate_half.py" >/dev/null && stage="eval"
  say "free ${f}% swap ${s}GB workers $(pgrep -f multiprocessing.spawn | wc -l | tr -d ' ') stage ${stage}"
  if [ "$f" -lt "$WARN_AT" ] && [ "$warned" -eq 0 ]; then say "WARN free ${f}%"; warned=1; fi; [ "$f" -ge "$WARN_AT" ] && warned=0
  if [ "$f" -lt "$FREE_MIN" ]; then low=$((low+1)); else low=0; fi
  hit=$(awk -v s="$s" -v m="$SWAP_MAX_GB" 'BEGIN{print (s>m)?1:0}')
  if { [ "$low" -ge 2 ] || [ "$hit" -eq 1 ]; } && [ "$stage" != "idle" ]; then
    say "RESTART: killing ${stage} (free ${f}% swap ${s}GB); the chain relaunches from checkpoint"
    for pat in train_half.py evaluate_half.py; do for p in $(pgrep -f "$pat"); do for c in $(pgrep -P "$p"); do kill -KILL "$c" 2>/dev/null; done; kill -KILL "$p" 2>/dev/null; done; done
    sleep 5; reap; low=0; sleep 120
  fi
  sleep 60
done
