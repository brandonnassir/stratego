"""Phase 9 Agent 1: deterministic, family-balanced held-out evaluation banks.

Specification sources:

- `01_AGENT_1_RL_CONTRACT_AND_EVAL_BANKS.md` ("Evaluation banks", "Sealing")
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Validation and final-test
  banks")

What a bank is
--------------
A Phase 9 bank is an ordinary frozen :class:`SetupBank` — the same value
object the accepted Phase 4 match machinery already consumes — whose pairs
are built from the Phase 7 setup library's held-out splits instead of the
Phase 4 structured generator:

```text
phase9_validation_bank_v1   128 cases   validation split   8 per family
phase9_test_bank_v1         512 cases   test split        32 per family
```

Case identity is family-major: ``setup_pair_id = family_index *
cases_per_family + case_ordinal`` over the frozen family order `F00..F15`,
so family balance is a property of the id space, not of any draw.

Family-pure cases
-----------------
Both sides of a case draw from the case's family. The case family is then
unambiguous — setup-family EWR attributes cleanly — and because the
evaluated candidate swaps colours inside every `color_swap_same_board`
paired unit, family purity is the only colour-symmetric choice.

Deterministic rejection over the frozen sampler
-----------------------------------------------
The accepted `setup_sampler_v1` draws its family uniformly; this module
never bypasses or reweights it. A case side walks ``attempt = 0, 1, 2, ...``
through ``sample_setup(split, eval_bank_draw_seed(bank_version, family_id,
case_ordinal, side, attempt), profile='neutral_v1')`` and accepts the first
draw whose ``primary_family_id`` equals the case family. Every accepted
draw is a complete, untouched sampler output (base choice, perturbation,
reflection, validation stack and provenance included), so the bank inherits
the sampler's frozen semantics wholesale. Acceptance probability is 1/16
per attempt; the frozen 2,048-attempt ceiling has failure probability below
1e-57 and exists only to make a broken library loud.

Held-out access
---------------
Sampling the validation/test splits here is an explicit, justified
evaluation/audit request in the sense of the Phase 7 split rules: the
justification strings below are recorded in both manifests, split isolation
is re-audited from provenance, and no game is ever played during
construction, so no outcome can select a case.
"""

from __future__ import annotations

import hashlib
import json
import time

# This module is deliberately NOT re-exported from `stratego.evaluation`'s
# package __init__: its contract imports reach `stratego.model` (and so
# torch), and spawned game workers import evaluation modules — the package
# import must stay pure-engine so `workers_importing_torch` stays 0.
# Import it directly, function-scoped in anything that spawns workers.

from ..setups.contracts import (
    SPLIT_TEST,
    SPLIT_VALIDATION,
    parse_base_setup_id,
    split_for_base_index,
)
from ..setups.families import FAMILY_IDS
from ..setups.sampler import (
    SAMPLER_VERSION,
    load_library_index,
    rebuild_from_provenance,
    sample_setup,
)
from ..training.phase9_contract import (
    BANK_GENERATION_FAMILY,
    BANK_MAX_ATTEMPTS_PER_SIDE,
    Phase9ContractError,
    TEST_BANK_CASES,
    TEST_BANK_VERSION,
    TEST_CASES_PER_FAMILY,
    VALIDATION_BANK_CASES,
    VALIDATION_BANK_VERSION,
    VALIDATION_CASES_PER_FAMILY,
)
from ..training.phase9_seed import (
    PHASE9_MASTER_SEED,
    derive_phase9_seed,
    eval_bank_draw_seed,
)
from ..training.warmstart_contract import EXPECTED_SETUP_PROFILE
from .setup_bank import SetupBank, SetupPair, bank_digest, validate_setup_pair

#: The two Phase 9 banks, keyed by their short name.
BANK_SPECIFICATIONS = {
    "validation": {
        "bank_version": VALIDATION_BANK_VERSION,
        "split": SPLIT_VALIDATION,
        "cases_per_family": VALIDATION_CASES_PER_FAMILY,
        "case_count": VALIDATION_BANK_CASES,
        "access_justification": (
            "Phase 9 validation bank: frozen model-selection evaluation "
            "cases (Agent 1 construction)"
        ),
    },
    "test": {
        "bank_version": TEST_BANK_VERSION,
        "split": SPLIT_TEST,
        "cases_per_family": TEST_CASES_PER_FAMILY,
        "case_count": TEST_BANK_CASES,
        "access_justification": (
            "Phase 9 sealed final-test bank: frozen final-evaluation cases "
            "(Agent 1 construction; structural audit only before Agent 8)"
        ),
    },
}

#: Manifest keys deliberately outside the manifest digest: run measurements,
#: not bank identity.
MANIFEST_VOLATILE_KEYS = ("construction_run", "manifest_digest")

MANIFEST_DIGEST_DOMAIN = "stratego_phase9_bank_manifest_v1"

_SIDES = ("red", "blue")
_SIDE_PLAYER = {"red": 0, "blue": 1}


class Phase9BankError(Phase9ContractError):
    """A Phase 9 bank failed construction or audit."""


def bank_specification(bank: str) -> dict:
    try:
        return dict(BANK_SPECIFICATIONS[bank])
    except KeyError:
        raise Phase9BankError(
            f"unknown Phase 9 bank {bank!r}; expected one of "
            f"{sorted(BANK_SPECIFICATIONS)}"
        ) from None


def case_family(setup_pair_id: int, cases_per_family: int) -> str:
    """The frozen family of one case id: family-major over `F00..F15`."""
    family_index, _ = divmod(int(setup_pair_id), int(cases_per_family))
    if not 0 <= family_index < len(FAMILY_IDS):
        raise Phase9BankError(
            f"setup_pair_id {setup_pair_id} is outside the "
            f"{len(FAMILY_IDS) * cases_per_family}-case id space"
        )
    return FAMILY_IDS[family_index]


def resolve_case_side(
    bank_version: str,
    split: str,
    family_id: str,
    case_ordinal: int,
    side: str,
    *,
    index=None,
):
    """`(SampledSetup, attempt, draw_seed)` for one side of one bank case.

    Deterministic rejection over the untouched frozen sampler: the first
    attempt whose primary family matches the case family wins. Raises after
    the frozen attempt ceiling, which an honest library cannot reach.
    """
    if side not in _SIDES:
        raise Phase9BankError(f"side must be one of {_SIDES}, got {side!r}")
    library = load_library_index() if index is None else index
    for attempt in range(BANK_MAX_ATTEMPTS_PER_SIDE):
        seed = eval_bank_draw_seed(bank_version, family_id, case_ordinal, side, attempt)
        sampled = sample_setup(
            split, seed, profile=EXPECTED_SETUP_PROFILE, index=library
        )
        if sampled.family_id == family_id:
            return sampled, attempt, seed
    raise Phase9BankError(
        f"{bank_version} {family_id} case {case_ordinal} {side}: no "
        f"family-matching draw within {BANK_MAX_ATTEMPTS_PER_SIDE} attempts; "
        "the library or sampler has drifted (BLOCKED)"
    )


def build_case(
    bank: str, setup_pair_id: int, *, index=None
) -> tuple[SetupPair, dict]:
    """One bank case and its provenance record, from identity alone."""
    specification = bank_specification(bank)
    family_id = case_family(setup_pair_id, specification["cases_per_family"])
    case_ordinal = int(setup_pair_id) % specification["cases_per_family"]
    library = load_library_index() if index is None else index

    sides: dict = {}
    setups: dict = {}
    for side in _SIDES:
        sampled, attempt, seed = resolve_case_side(
            specification["bank_version"],
            specification["split"],
            family_id,
            case_ordinal,
            side,
            index=library,
        )
        player = _SIDE_PLAYER[side]
        setups[side] = sampled.oriented(player)
        sides[side] = {
            "accepted_attempt": attempt,
            "accepted_draw_seed": seed,
            **sampled.provenance,
        }

    pair = SetupPair(
        setup_pair_id=int(setup_pair_id),
        red_setup=setups["red"],
        blue_setup=setups["blue"],
        generation_seed=derive_phase9_seed(
            "eval_bank", specification["bank_version"], family_id, case_ordinal
        ),
        bank_version=specification["bank_version"],
        generation_family=BANK_GENERATION_FAMILY,
    )
    provenance = {
        "setup_pair_id": int(setup_pair_id),
        "family_id": family_id,
        "case_ordinal": case_ordinal,
        "split": specification["split"],
        "red": sides["red"],
        "blue": sides["blue"],
    }
    return pair, provenance


def build_phase9_bank(bank: str) -> tuple[SetupBank, dict]:
    """The complete frozen bank and its manifest.

    Reproducible from frozen constants alone; two builds yield identical
    bytes and identical digests, which the audit re-proves by rebuilding a
    deterministic sample of cases.
    """
    specification = bank_specification(bank)
    library = load_library_index()
    started = time.time()

    pairs = []
    provenance_records = []
    attempt_histogram: dict = {}
    for setup_pair_id in range(specification["case_count"]):
        pair, provenance = build_case(bank, setup_pair_id, index=library)
        pairs.append(pair)
        provenance_records.append(provenance)
        for side in _SIDES:
            attempts = provenance[side]["accepted_attempt"]
            attempt_histogram[attempts] = attempt_histogram.get(attempts, 0) + 1

    built = SetupBank(
        bank_version=specification["bank_version"],
        root_seed=PHASE9_MASTER_SEED,
        generation_family=BANK_GENERATION_FAMILY,
        pairs=tuple(pairs),
    )

    manifest = {
        "bank": bank,
        "bank_version": specification["bank_version"],
        "generation_family": BANK_GENERATION_FAMILY,
        "split": specification["split"],
        "case_count": specification["case_count"],
        "cases_per_family": specification["cases_per_family"],
        "family_ids": list(FAMILY_IDS),
        "case_id_rule": (
            "setup_pair_id = family_index * cases_per_family + case_ordinal "
            "over families F00..F15"
        ),
        "family_purity": "both sides of a case draw from the case's family",
        "sampler_version": SAMPLER_VERSION,
        "sampler_profile": EXPECTED_SETUP_PROFILE,
        "library_content_digest": library.content_digest,
        "draw_rule": (
            "side draw = first attempt k with sample_setup(split, "
            "eval_bank_draw_seed(bank_version, family_id, case_ordinal, "
            "side, k), profile='neutral_v1').family_id == case family"
        ),
        "max_attempts_per_side": BANK_MAX_ATTEMPTS_PER_SIDE,
        "access_justification": specification["access_justification"],
        "no_outcome_selection": (
            "construction plays no game and reads no strength signal; draws "
            "are rejected only for family identity"
        ),
        "bank_digest": bank_digest(built),
        "attempt_histogram": {
            str(attempts): count for attempts, count in sorted(attempt_histogram.items())
        },
        "case_provenance": provenance_records,
        "construction_run": {
            "duration_seconds": round(time.time() - started, 3),
        },
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return built, manifest


def manifest_digest(manifest: dict) -> str:
    """SHA-256 over the manifest's identity fields (volatile keys excluded)."""
    stable = {
        key: value for key, value in manifest.items() if key not in MANIFEST_VOLATILE_KEYS
    }
    payload = MANIFEST_DIGEST_DOMAIN + "\n" + json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Structural audit
# ---------------------------------------------------------------------------


def audit_phase9_bank(
    bank: str,
    built: SetupBank,
    manifest: dict,
    *,
    rebuild_sample_every: int = 16,
) -> dict:
    """Recompute every structural bank property from stored content.

    Structural only: no game is played, no model is loaded, no outcome is
    read — this is exactly the `structural_audit` purpose the sealing rules
    allow every agent. `rebuild_sample_every` controls the isolated-rebuild
    spot check (every Nth case is rebuilt from identity alone and compared
    byte for byte); 1 rebuilds every case.
    """
    specification = bank_specification(bank)
    library = load_library_index()

    failures: list = []
    engine_failures: list = []
    family_counts: dict = {family_id: 0 for family_id in FAMILY_IDS}
    split_violations: list = []
    purity_violations: list = []
    provenance_mismatches: list = []
    rebuild_mismatches: list = []

    records = {record["setup_pair_id"]: record for record in manifest["case_provenance"]}
    if len(records) != len(built.pairs):
        failures.append(
            f"{len(records)} provenance records for {len(built.pairs)} pairs"
        )

    expected_ids = tuple(range(specification["case_count"]))
    if built.pair_ids != expected_ids:
        failures.append("setup_pair_ids are not exactly 0..case_count-1 in order")

    for pair in built.pairs:
        if pair.bank_version != specification["bank_version"]:
            failures.append(f"pair {pair.setup_pair_id}: bank_version {pair.bank_version!r}")
        if pair.generation_family != BANK_GENERATION_FAMILY:
            failures.append(
                f"pair {pair.setup_pair_id}: generation_family {pair.generation_family!r}"
            )
        engine_failures.extend(validate_setup_pair(pair))

        expected_family = case_family(
            pair.setup_pair_id, specification["cases_per_family"]
        )
        family_counts[expected_family] += 1

        record = records.get(pair.setup_pair_id)
        if record is None:
            failures.append(f"pair {pair.setup_pair_id}: no provenance record")
            continue
        if record["family_id"] != expected_family:
            failures.append(
                f"pair {pair.setup_pair_id}: provenance family {record['family_id']!r} "
                f"disagrees with the id rule {expected_family!r}"
            )

        for side, stored_setup in (("red", pair.red_setup), ("blue", pair.blue_setup)):
            side_record = record[side]
            if side_record["primary_family_id"] != expected_family:
                purity_violations.append(
                    f"pair {pair.setup_pair_id} {side}: family "
                    f"{side_record['primary_family_id']!r}"
                )
            if side_record["split"] != specification["split"]:
                split_violations.append(
                    f"pair {pair.setup_pair_id} {side}: split {side_record['split']!r}"
                )
            _, _, base_index = parse_base_setup_id(side_record["base_setup_id"])
            if split_for_base_index(base_index) != specification["split"]:
                split_violations.append(
                    f"pair {pair.setup_pair_id} {side}: base index {base_index} is "
                    f"not a {specification['split']!r} base"
                )
            try:
                rebuilt = rebuild_from_provenance(side_record, index=library)
            except Exception as error:  # noqa: BLE001 - a failed rebuild is a finding
                provenance_mismatches.append(
                    f"pair {pair.setup_pair_id} {side}: rebuild failed: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            if rebuilt.oriented(_SIDE_PLAYER[side]) != stored_setup:
                provenance_mismatches.append(
                    f"pair {pair.setup_pair_id} {side}: provenance does not rebuild "
                    "the stored setup"
                )

        if pair.setup_pair_id % max(1, int(rebuild_sample_every)) == 0:
            rebuilt_pair, rebuilt_record = build_case(
                bank, pair.setup_pair_id, index=library
            )
            if rebuilt_pair != pair:
                rebuild_mismatches.append(
                    f"pair {pair.setup_pair_id}: isolated rebuild differs"
                )
            elif rebuilt_record != record:
                rebuild_mismatches.append(
                    f"pair {pair.setup_pair_id}: isolated provenance rebuild differs"
                )

    observed_digest = bank_digest(built)
    positions = {(pair.red_setup, pair.blue_setup) for pair in built.pairs}

    checks = {
        "case_count_exact": len(built.pairs) == specification["case_count"],
        "pair_ids_contiguous": built.pair_ids == expected_ids,
        "family_balance_exact": all(
            count == specification["cases_per_family"] for count in family_counts.values()
        ),
        "family_purity": not purity_violations,
        "split_isolation": not split_violations,
        "engine_valid": not engine_failures,
        "provenance_rebuilds": not provenance_mismatches,
        "isolated_rebuild_exact": not rebuild_mismatches,
        "distinct_positions": len(positions) == len(built.pairs),
        "digest_matches_manifest": observed_digest == manifest["bank_digest"],
        "manifest_digest_consistent": manifest_digest(manifest)
        == manifest["manifest_digest"],
        "no_structural_failures": not failures,
    }

    return {
        "bank": bank,
        "bank_version": specification["bank_version"],
        "case_count": len(built.pairs),
        "family_counts": family_counts,
        "bank_digest": observed_digest,
        "failures": failures,
        "engine_failures": engine_failures,
        "purity_violations": purity_violations,
        "split_violations": split_violations,
        "provenance_mismatches": provenance_mismatches,
        "rebuild_mismatches": rebuild_mismatches,
        "rebuild_sample_every": int(rebuild_sample_every),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


__all__ = [
    "BANK_SPECIFICATIONS",
    "MANIFEST_DIGEST_DOMAIN",
    "MANIFEST_VOLATILE_KEYS",
    "Phase9BankError",
    "audit_phase9_bank",
    "bank_specification",
    "build_case",
    "build_phase9_bank",
    "case_family",
    "manifest_digest",
    "resolve_case_side",
]
