"""The Phase 7 base-library seed context: `setup_library_seed_v1`.

Specification sources:

- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (determinism and identity)
- `02_AGENT_2_BASE_LIBRARY_GENERATOR.md` (master seed and isolated
  regeneration)

Agent 1 froze the seed *derivation* in `identity.py`
(:func:`derive_base_seed`, :func:`derive_attempt_seed`). This module adds the
one thing generation needs on top: a single immutable object that carries the
four identity inputs — contract version, library version, master seed and the
generator version — so no call site ever mixes seed inputs by hand, and so the
exact derivation can be serialized into the library manifest.

The derivation is a pure hash of identity inputs:

```text
base_seed    = blake2b('strat-lb7', "contract:library:master_seed:family:index")
attempt_seed = blake2b('strat-at7', "base_seed:attempt")
```

Nothing about it depends on how many setups were generated before, which
family is being generated, or in what order the library is enumerated. Adding
a rejected candidate to F03 therefore cannot move any other base setup —
the property that makes `rebuild_base_setup(family_id, base_index)` exact.
"""

import random
from dataclasses import dataclass

from .contracts import (
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
)
from .families import FAMILY_BY_ID
from .identity import SetupLibraryError, derive_attempt_seed, derive_base_seed

#: Version identifier of the seed-context contract itself. A semantic change
#: to the derivation is a new identifier, never a silent reinterpretation.
SEED_CONTEXT_VERSION = "setup_library_seed_v1"


@dataclass(frozen=True)
class LibrarySeedContext:
    """The complete randomness identity of one materialized library.

    Two contexts with equal fields produce byte-identical libraries; a context
    differing in any field — most obviously `master_seed` — defines a
    different library, which is exactly why the master seed is frozen
    alongside the contract version.
    """

    master_seed: int = DEFAULT_LIBRARY_MASTER_SEED
    contract_version: str = SETUP_GENERATOR_CONTRACT_VERSION
    library_version: str = SETUP_LIBRARY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.master_seed, int) or isinstance(self.master_seed, bool):
            raise SetupLibraryError(f"master_seed must be an int, got {self.master_seed!r}")

    def base_seed(self, family_id: str, base_index: int) -> int:
        """The generation seed of one base identity.

        A pure function of `(contract, library, master_seed, family, index)`,
        so any process can rebuild any single base without generating any
        other base.
        """
        if family_id not in FAMILY_BY_ID:
            raise SetupLibraryError(f"unknown family id: {family_id!r}")
        if not 0 <= base_index < BASES_PER_FAMILY:
            raise SetupLibraryError(
                f"base_index must be in 0..{BASES_PER_FAMILY - 1}, got {base_index}"
            )
        return derive_base_seed(
            self.contract_version,
            self.library_version,
            self.master_seed,
            family_id,
            base_index,
        )

    def attempt_seed(self, family_id: str, base_index: int, attempt: int) -> int:
        """The seed of rejection-sampling attempt `attempt` for one base."""
        return derive_attempt_seed(self.base_seed(family_id, base_index), attempt)

    def attempt_rng(self, family_id: str, base_index: int, attempt: int) -> random.Random:
        """A private RNG stream for one candidate attempt.

        Every random decision in candidate construction comes from this
        stream, so a candidate is a pure function of its attempt identity and
        no global RNG state is ever consumed.
        """
        return random.Random(self.attempt_seed(family_id, base_index, attempt))

    def to_dict(self) -> dict:
        """The machine-readable derivation record for the library manifest."""
        return {
            "seed_context_version": SEED_CONTEXT_VERSION,
            "master_seed": self.master_seed,
            "contract_version": self.contract_version,
            "library_version": self.library_version,
            "base_seed_derivation": (
                "blake2b(person='strat-lb7', digest_size=8) over "
                "'contract_version:library_version:master_seed:family_id:base_index', "
                "big-endian, right-shifted one bit"
            ),
            "attempt_seed_derivation": (
                "blake2b(person='strat-at7', digest_size=8) over "
                "'base_seed:attempt', big-endian, right-shifted one bit"
            ),
            "candidate_rule": (
                "attempts 0, 1, 2, ... are drawn from independent streams and "
                "the first candidate satisfying the frozen contract is "
                "accepted; rejection is local to the base identity"
            ),
            "enumeration_independence": (
                "no seed input names another base setup, another family, or "
                "any generation counter, so adding or removing a rejected "
                "candidate anywhere cannot change any other accepted base"
            ),
        }


#: The canonical Phase 7 seed context: the frozen master seed with the frozen
#: contract and library versions.
DEFAULT_SEED_CONTEXT = LibrarySeedContext()
