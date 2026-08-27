"""Phase 16 Agent 2: stochastic search over the frozen Phase 15 stack.

Two independent, seed-deterministic knobs over the accepted engine
(`phase12_root_world_search_v1` reached through the Phase 15 systems):

1. **move sampling** — the played move is drawn from
   ``softmax(S(a)/tau)`` over the engine's own candidate set; ``tau = 0``
   plays the frozen argmax exactly;
2. **rollout sampling** — rollout actions for both sides are drawn from
   the move model's distribution at temperature ``tau_r``, restricted to
   the smallest legal set covering top-p probability mass; ``tau_r = 0``
   is the frozen greedy rollout exactly.

Everything else — candidate rule, world sampling, deduplication, caps,
fallback, oracle refusals — is byte-identical to the accepted Phase 15
configuration. Nothing in the Phase 2-15 namespaces is edited; accepted
behaviour is reached by import and new behaviour lives here.
"""
