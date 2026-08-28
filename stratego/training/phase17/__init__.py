"""Phase 17: tandem current-policy self-play.

Additive namespace. Nothing here edits, wraps in place, or overwrites an
accepted Phase 2-16 module: the engine, the observation, the trajectory
builder, the orientation helper, the diversity primitives, the setup identity
frame, the Phase 9 target recursions and the Phase 9 objective's components are
all imported unmodified, and everything Phase 17 changes is rebuilt under this
package.

The move half (Agent 2) is:

```text
move_contract.py         constants, versions, seeds, the schedule horizon,
                         the identity plumbing, the participant refusals
move_start.py            the exact Phase 9 loader, and the fresh optimizer /
                         KL controller / EMA / schedule state
move_snapshot.py         the live current-policy cell, the seating that reads
                         it per decision, and the categorical action sampler
move_loss.py             the objective, per-row maskable, with no belief term
move_trainer.py          the one-epoch update
transition_schema.py     phase17_move_transition_v1 and its validator
transition_targets.py    the tailed recursions, the seat-trace carry state,
                         gate G-M4a, and the divergence telemetry
transition_collector.py  the true fixed-transition window collector
```

The setup half (Agent 3) is:

```text
setup_contract.py   frozen constants, identities, seeds, refusals
setup_model.py      the 802,320-parameter causal decoder
setup_sampling.py   inventory masking, orientation, batched generation, pools
setup_episode.py    the episode schema, its FIFO queue, the Agent 4 API
setup_learning.py   the setup update: advantage, losses, KL controller, EMA
setup_metrics.py    diversity and entropy measurement
```

This module deliberately re-exports nothing. The two halves are owned by
different agents and are consumed directly by module path; a package `__init__`
that imported one half would put it in the other's import closure, which is
exactly what the move half's structural no-search gate measures.
"""
