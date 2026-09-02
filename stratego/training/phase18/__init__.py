"""Phase 18: the setup-policy implementation at published-method parity.

Additive namespace. Nothing here edits, wraps in place or overwrites an
accepted Phase 2-17 module. The engine, the setup identity frame, the accepted
orientation helper and the accepted parameter digest are imported unmodified;
everything Phase 18 changes about the setup learner is rebuilt under this
package, checked row by row against
`reports/phase18/ataraxos_setup_method_map_v2.json` (S01-S30) and against the
authors' published implementation at commit
`92db29e8ffc323b1b8a2804b5c3f84695d036b05`.

```text
setup_contract.py        frozen constants, identities, seeds, refusals
setup_model.py           the 802,320-parameter causal decoder
setup_sampling.py        inventory + handedness masking, seeded generation,
                         post-generation reflection, orientation boundary
setup_buffer.py          reusable pools, identity/de-duplication, repeated
                         outcome aggregation, the flat advantage, minibatches
setup_learning.py        the loss, AdamW(wd=0), clipping, EMA, checkpoints
reference_oracle.py      the implementation-independent parity oracle
synthetic_landscape.py   the frozen known-reward landscape (Gate G2 assay)
synthetic_assay.py       the three-seed synthetic learning assay runner
coverage.py              the machine-readable S01-S30 coverage table
```

The five Phase 17 corrections mandated by the common contract (section 3.1)
are each a concrete mechanism here, not a note: the advantage residual is
`I - 10h` (setup_buffer.process), the Flag is forced into one horizontal half
during generation and a seeded 50% reflection follows (setup_sampling),
outcomes are aggregated per exact setup and snapshot before the update
(setup_buffer), the optimizer minibatch is 1,024 setups (setup_learning), and
the experimental point is left to the tandem stage, which this package does
not touch.

This module deliberately re-exports nothing.
"""

#: The package identity recorded in every Phase 18 setup artifact.
PHASE18_SETUP_PACKAGE_VERSION = "phase18_setup_v1"

__all__ = ["PHASE18_SETUP_PACKAGE_VERSION"]
