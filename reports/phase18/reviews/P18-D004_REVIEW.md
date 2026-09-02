# Review record of P18-D004

## Decision

**`P18-D004` is accepted as `REVISE`.** On 2026-09-02 the operator ordered the
publication of the reviewed Agent 4 branch, a bounded correction to the packet's
wording, and one bounded confirmation experiment
(`instructions/phase_18_setup_integrated_warmstart/07_AGENT_5_G2_RAW_ACTOR_CONFIRMATION.md`,
SHA-256 `afd411fa097528796d843055a02774f73acecab71d134c61e484ef9777f159af`). The decision itself, every identity and every
number in the packet stand.

```text
parity                          30/30 method-map rows from recorded outcomes; oracle PASS
                                (loss terms 1.8e-15, gradients 2e-10); 86/86 + 122/122 tests
integrity                       0 legality / orientation / attribution / non-finite /
                                checkpoint-identity events across three seeds
raw actor gap closure           20.92% / 18.50% / 14.82%   pooled 95% [+10.6338, +10.9089]
EMA gap closure                  0.28% /  0.52% /  0.35%   pooled 95% [+0.1588, +0.2949]
EMA retained initial fraction   0.999^64 = 0.937975  (~1,000-update time constant)
EMA movement / raw displacement 1.3% / 2.8% / 2.4%
replay                          landscape, first-period outcomes and both endpoints bitwise
sealed Phase 8 access           0        Stratego games       0
```

## Identities checked before publication

- `phase18/g2-setup-parity` resolved to `6afa13bed355884a3327d2661fd739784260dc2b`,
  the reviewed HEAD, before the push; the only dirty path in the canonical checkout
  was the protected `reports/phase13/phase14_launch_manifest_v1.json`, which was
  neither staged, edited, restored nor stashed.
- At that commit `P18-D004.md` hashes to `d6e90e041e9f66bb3d79c74fb40541e6fa94d9a39f2ba064a217ec33aba86cfc`
  and `P18-D004.json` to `36fb4a3d9b8afd4a087cf16dbcd6f834962cafffb8e1cf359c1acfeb102f6679`, the digests the
  decision index recorded at delivery.
- `G2_SOURCE_COMMIT 354a4cad55a88dca6dcb24a21cf79cecc130008f`, the verification
  commit `11b5558`, the launch-manifest commit `6f5297e` and the result-evidence
  commit `6623621` are on the branch in the recorded order.

## GitHub publication

Published under section 7 of the decision-packet protocol with a normal non-force
push and no publication commit:

```text
remote        origin  https://github.com/brandonnassir/stratego.git
branch        phase18/g2-setup-parity
local SHA     6afa13bed355884a3327d2661fd739784260dc2b
remote SHA    6afa13bed355884a3327d2661fd739784260dc2b
published     2026-09-02T14:28:00Z  by Phase 18 Agent 5
```

The remote branch was absent beforehand; no divergent history existed and none was
overwritten. `phase18/g2-raw-confirmation` was created from exactly that commit.

## Corrections to the packet wording

The operator required four corrections, applied by Agent 5 on
`phase18/g2-raw-confirmation` to `P18-D004.md`, `P18-D004.json` and
`agent_04_report.md`. The originals are byte-identical at the published commit and
the original wording of every corrected sentence is preserved in the JSON packet
under `corrections.items`.

1. **Withdrawn.** The packet claimed the EMA criterion "cannot be met inside the
   64-update budget by arithmetic" and was "unreachable at this budget", and that
   the evaluation model "cannot see" / "cannot reflect" the learning. The decay
   arithmetic gives the fraction of the initial parameters the EMA still carries
   and its time constant; it does not bound the EMA's held-out utility regardless
   of learner behaviour. The frozen contract's phrase "whatever the raw learner
   does" (which the packet quotes verbatim) is superseded by the same reading; the
   contract file is frozen and unedited.
2. **Stated.** Decay 0.999 retained `0.999^64 = 0.937975` of the initial parameter
   contribution after 64 once-per-update blends, giving the EMA an approximately
   1,000-update time constant (`1 / (1 - 0.999)`).
3. **Stated.** The frozen 64-update assay empirically showed severe EMA lag: the raw
   actor learned strongly (20.9%, 18.5%, 14.8% of the gap) while the EMA captured
   only a small portion of the observed change (1.3%, 2.8%, 2.4% of the raw
   displacement).
4. **Stated.** The two development smokes informed the raw-actor diagnostic and the
   instrument-defect interpretation; they did not change the frozen primary
   landscape, budget, threshold or run seeds.

Corrected digests on `phase18/g2-raw-confirmation`:

```text
P18-D004.md          295cb9340c56802f2d61dc154cea1bbb6b00e86a98ea4a4a4d7a86bb838b9c0f
P18-D004.json        bf6744d546d5756a84bef396eef6e8bb8b330be55941999e0547983f0e9f2cb1
agent_04_report.md   78d66f98003e8745db74d9f7290a80413a177dfeaf1acff3e0eccf126edb2b4b
```

## Authorized next question

> Using the parity-correct setup-learning method, does the raw generation actor
> reliably learn an independently generated synthetic setup landscape within 64
> setup updates?

This is the bounded G2 raw-actor confirmation executed by instruction 07: a new
landscape table from the same family and methodology, a new artifact namespace,
three fresh seed streams derived from the reviewed base commit and the namespace,
the unchanged G2 learning method, the raw actor as the primary endpoint of this
synthetic trainability assay only, and the EMA recorded as secondary telemetry that
cannot change the decision. The EMA remains the required evaluation/deployment
model for every later Stratego-facing stage. A pass closes only the synthetic
trainability portion of G2 and authorizes designing the next gate; it authorizes
neither G3 nor the full warmstart. No Stratego game and no sealed Phase 8 example
may be opened.
