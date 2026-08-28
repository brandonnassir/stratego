"""Phase 17: tandem current-policy self-play.

Additive namespace. Nothing here edits, wraps in place, or overwrites an
accepted Phase 2-16 module: the engine, the observation, the trajectory
builder, the orientation helper, the diversity primitives, the setup identity
frame, the Phase 9 target recursions and the Phase 9 objective's components are
all imported unmodified, and everything Phase 17 changes is rebuilt under this
package.

The move half is:

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

The setup half is:

```text
setup_contract.py   frozen constants, identities, seeds, refusals
setup_model.py      the 802,320-parameter causal decoder
setup_sampling.py   inventory masking, orientation, batched generation, pools
setup_episode.py    the episode schema, the pending buffer, the runner API
setup_learning.py   the setup update: advantage, losses, fixed KL, EMA
setup_metrics.py    diversity and entropy measurement
```

The tandem layer on top of the two halves is:

```text
runner.py           the ten contractual steps of one tandem iteration
checkpoint.py       the atomic paired move/setup checkpoint
supervisor.py       the integrity stops and the statistical warnings
telemetry.py        the durable append-only JSONL row log
export.py           the paired EMA evaluation bundle
```

The active recipe is operator decision D10's `phase17_simple_paper_tandem_v1`:
the paper's printed setup advantage, a fixed reverse-KL coefficient of 0.1,
`alpha(n) = 0.1 * n**-0.3` on the shared global iteration, and every completed
setup episode trained exactly once per fixed-transition window.

This module deliberately re-exports nothing. The two halves are owned by
different agents and are consumed directly by module path; a package `__init__`
that imported one half would put it in the other's import closure, which is
exactly what the move half's structural no-search gate measures.
"""
