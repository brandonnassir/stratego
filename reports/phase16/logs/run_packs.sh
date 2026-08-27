#!/bin/zsh
# Phase 16 Agent 1 pack chain, priority order per the brief:
# benchmark TINY baselines before MEDIUM; adversarial arm 2 before arm 3.
cd /Users/brandonwashington/Dev/Github/stratego/gpt_agent
PY=.venv/bin/python
set -x
$PY scripts/run_phase16_agent01.py --role bench --arm p24_direct --preset direct --workers 10 --expected-hours 0.2 --quiet
$PY scripts/run_phase16_agent01.py --role bench --arm p24_b24 --preset TINY --workers 10 --expected-hours 0.5 --quiet
$PY scripts/run_phase16_agent01.py --role baseline --preset TINY --arms benchmark_control,adversarial_opponent,adversarial_both --workers 10 --expected-hours 1.0 --quiet
$PY scripts/run_phase16_agent01.py --role analyse
$PY scripts/run_phase16_agent01.py --role bench --arm p24_b24 --preset MEDIUM --workers 10 --expected-hours 2.5 --quiet
$PY scripts/run_phase16_agent01.py --role analyse
$PY scripts/run_phase16_agent01.py --role baseline --preset MEDIUM --arms benchmark_control,adversarial_opponent --workers 10 --expected-hours 4.0 --quiet
$PY scripts/run_phase16_agent01.py --role analyse
$PY scripts/run_phase16_agent01.py --role baseline --preset MEDIUM --arms benchmark_control,adversarial_opponent,adversarial_both --workers 10 --expected-hours 2.0 --quiet
$PY scripts/run_phase16_agent01.py --role analyse
echo "PACK CHAIN COMPLETE"
