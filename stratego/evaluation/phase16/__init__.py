"""Phase 16 Agent 1: measurement instruments.

Specification sources:
`instructions/phase_16_robustness_and_distribution/00_PHASE_16_OVERVIEW.md`,
`instructions/phase_16_robustness_and_distribution/01_AGENT_1_MEASUREMENT_AND_OPERATOR_EXAM.md`.

What lives here
---------------
```text
contract.py      identities, seed streams, arm and family names
benchmark.py     phase16_benchmark_v1 — the canonical machine-opponent pack
adversarial.py   phase16_adversarial_setups_v1 — the pack that models the operator
runner.py        score_on_benchmark and the parallel pack executor
operator_log.py  the operator game log (JSONL) and setup harvesting
```

Nothing accepted is modified. Phase 15's match machinery (`matchplay`,
`systems`, `boards`, `analysis`, the loaders and the orientation gate) is
consumed strictly by import; behaviour that must differ is built here.
"""
