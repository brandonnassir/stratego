# Phase 9 Sequential Agent Plan — Manifest

Execute in this order:

```text
00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md
01_AGENT_1_RL_CONTRACT_AND_EVAL_BANKS.md
02_AGENT_2_POPULATION_AND_OPPONENT_SCHEDULER.md
03_AGENT_3_SELFPLAY_COLLECTOR_AND_ROLLOUT_STORE.md
04_AGENT_4_RL_TARGETS_ADVANTAGES_AND_ANTILEAK.md
05_AGENT_5_PPO_TRAINER_AND_RESUME.md
06_AGENT_6_BOUNDED_RL_PILOT_SELECTION.md
07_AGENT_7_CANONICAL_POPULATION_SELFPLAY_RUN.md
08_AGENT_8_FINAL_ACCEPTANCE_AND_FREEZE.md
```

After each agent completes, return its report to the reviewing chat. Do not authorize the next agent until the previous one is formally accepted.

All agent instructions preserve the Phase 8 corpus resolver rule: downstream code must call `synthetic_corpus.default_corpus_root()` rather than embedding the current absolute corpus path. Corpus identity is version + accepted digests, not location.
