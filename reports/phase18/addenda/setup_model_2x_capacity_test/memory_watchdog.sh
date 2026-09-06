#!/bin/bash
# Memory watchdog for the 2x setup-model run. Every 60 s: log free%, swap and the trainer's RSS.
# If free memory stays below FREE_MIN% for two consecutive checks, or swap use exceeds SWAP_MAX_GB,
# kill the running stage (trainer or evaluator) and its workers, let memory settle, and relaunch —
# the trainer resumes from its latest checkpoint, the evaluator from its completed arms.
ROOT=/Users/brandonwashington/Dev/Github/stratego/gpt_agent
X2=$ROOT/output/phase18/runtime/addendum_library_setup_from_scratch_2x
LOG=$X2/memory_watch.log; FREE_MIN=15; WARN_AT=30; SWAP_MAX_GB=3; low_streak=0; warned=0
TRAIN="train_from_scratch_2x.py --periods 1024"; EVAL="evaluate_2x.py"
say(){ echo "[memwatch $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
freepct(){ memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%","",$5); print $5}'; }
swapgb(){ sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); printf "%.2f", v/1024}}'; }
rss_gb(){ ps -eo rss,command | grep -E "train_from_scratch_2x.py --periods|evaluate_2x.py" | grep -v grep | awk '{s+=$1} END {printf "%.2f", s/1048576}'; }
workers(){ pgrep -f "multiprocessing.spawn" | wc -l | tr -d " "; }
reap_orphans(){ local n=0; for q in $(ps -eo pid,ppid,command | awk '$2==1 && /multiprocessing.spawn/ {print $1}'); do kill -KILL "$q" 2>/dev/null && n=$((n+1)); done; [ "$n" -gt 0 ] && say "REAPED $n orphaned worker process(es) (PPID 1)"; }
kill_tree(){ # $1 = pattern
  for p in $(pgrep -f "$1"); do for c in $(pgrep -P "$p"); do kill -TERM "$c" 2>/dev/null; done; kill -TERM "$p" 2>/dev/null; done
  for i in $(seq 1 30); do pgrep -f "$1" >/dev/null || break; sleep 1; done
  for p in $(pgrep -f "$1"); do for c in $(pgrep -P "$p"); do kill -KILL "$c" 2>/dev/null; done; kill -KILL "$p" 2>/dev/null; done
  pkill -KILL -f "multiprocessing.spawn" 2>/dev/null; pkill -KILL -f "multiprocessing.resource_tracker" 2>/dev/null; sleep 5
}
say "started: restart when free<${FREE_MIN}% twice or swap>${SWAP_MAX_GB}GB; warn at free<${WARN_AT}%"
while true; do
  f=$(freepct); s=$(swapgb); r=$(rss_gb); f=${f:-100}; s=${s:-0}
  stage="idle"; pgrep -f "$TRAIN" >/dev/null && stage="train"; pgrep -f "$EVAL" >/dev/null && stage="eval"
  reap_orphans
  say "free ${f}% swap ${s}GB parent_rss ${r}GB workers $(workers) stage ${stage}"
  if [ "$(ls "$X2"/eval/rows_*_ckpt_1024.jsonl 2>/dev/null | wc -l | tr -d " ")" = "8" ] && [ "$stage" = "idle" ]; then say "final evaluation complete (8 ckpt_1024 arms); watchdog exiting"; exit 0; fi
  if [ "$f" -lt "$WARN_AT" ] && [ "$warned" -eq 0 ]; then say "WARN free memory ${f}% (swap ${s}GB)"; warned=1; fi
  [ "$f" -ge "$WARN_AT" ] && warned=0
  if [ "$f" -lt "$FREE_MIN" ]; then low_streak=$((low_streak+1)); else low_streak=0; fi
  swap_hit=$(awk -v s="$s" -v m="$SWAP_MAX_GB" 'BEGIN{print (s>m)?1:0}')
  if { [ "$low_streak" -ge 2 ] || [ "$swap_hit" -eq 1 ]; } && [ "$stage" != "idle" ]; then
    say "RESTART triggered (free ${f}% streak ${low_streak}, swap ${s}GB) — killing ${stage} and relaunching from checkpoint"
    cd "$ROOT/output/phase18/worktrees/g3-stage6b"
    if [ "$stage" = "train" ]; then
      kill_tree "$TRAIN"; say "after kill: free $(freepct)% swap $(swapgb)GB"
      nohup caffeinate -i "$ROOT/.venv/bin/python" "$X2/train_from_scratch_2x.py" --periods 1024 >> "$X2/train_stdout.log" 2>&1 &
      say "trainer relaunched (pid $!) — resumes from $(cat "$X2/state.json" 2>/dev/null | tr -d '\n' | cut -c1-40)"
    else
      pkill -TERM -f chain_eval.sh 2>/dev/null; kill_tree "$EVAL"; say "after kill: free $(freepct)% swap $(swapgb)GB"
      nohup caffeinate -i "$ROOT/.venv/bin/python" "$X2/evaluate_2x.py" 0 256 384 512 640 768 896 1024 >> "$X2/eval_stdout.log" 2>&1 &
      say "evaluator relaunched (pid $!) — completed arms are reused"
    fi
    low_streak=0; warned=0; sleep 120
  fi
  sleep 60
done
