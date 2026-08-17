"""Phase 9: `phase9_operational_amendment_v1` — the reviewed 15-hour ceiling.

Specification source: the reviewing chat's Agent 6 review resolution, which
formally accepted the bounded pilot selection (P9-C the unique winner) and
authorized **one** narrow change: the canonical *operational* wall-clock
ceiling moves from 43,200 s (12 h) to 54,000 s (15 h).

Why this is a separate module and a separate identity
-----------------------------------------------------
`phase9_contract.CANONICAL_WALL_CLOCK_CEILING_HOURS` stays `12` and
`phase9_contract.contract_digest()` stays
`ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34`. That is
not deference to convention — it is a correctness requirement:

```text
the contract digest is stamped into every phase9_rollout_store_v1 metadata
sidecar (57,344 committed pilot games) and into every phase9_checkpoint_v1
```

Editing the frozen contract in place would change that digest, and every
sealed rollout and every checkpoint written so far would fail the identity
checks that `bind_sealed_rollout` and `validate_phase9_payload` run — the
accepted Agents 3-5 evidence would stop verifying against the library that
produced it. So the amendment is layered *beside* the contract as its own
review-authorized identity, carrying the digest of the exact base contract it
amends, and history is preserved rather than rewritten.

Scope: operational, not experimental
------------------------------------
A wall-clock ceiling is an operations budget, not a learning-design decision.
It appears in no loss, no schedule, no seed, no target and no acceptance
threshold, and it is absent from `Phase9TrainConfig.identity()` — which is
why the trainer runtime identity digest is provably unchanged by this
amendment (:func:`runtime_identity_is_unaffected` measures that rather than
asserting it).

Everything else stays exactly as frozen and accepted:

```text
60 RL iterations              2,048 scheduled games / iteration
2 optimizer epochs / rollout  twelve validation passes (cadence 5)
archive cadence 5             P9-C: LR 3e-4, initial KL beta 0.005
population mixture            all eight Phase 9 seeds
selection score + tie-break   final acceptance gates A-H
the sealed phase9_test_bank_v1 remains sealed until Agent 8
```
"""

from __future__ import annotations

import hashlib
import json

from .phase9_contract import (
    ARCHIVE_CADENCE_ITERATIONS,
    CANONICAL_GAMES_PER_ITERATION,
    CANONICAL_ITERATIONS,
    CANONICAL_WALL_CLOCK_CEILING_HOURS,
    EPOCHS_PER_ROLLOUT,
    PHASE9_RL_CONTRACT_VERSION,
    VALIDATION_CADENCE_ITERATIONS,
    Phase9ContractError,
    contract_digest,
)

#: The amendment's own frozen identity.
PHASE9_OPERATIONAL_AMENDMENT_VERSION = "phase9_operational_amendment_v1"

#: The exact base contract this amendment applies to. A drift here means the
#: amendment is being read against a contract it was never reviewed against.
AMENDED_CONTRACT_VERSION = PHASE9_RL_CONTRACT_VERSION
AMENDED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)

#: The historical value, preserved. Agent 6 measured its projection against
#: this ceiling and correctly returned BLOCKED; that record stands.
HISTORICAL_CEILING_HOURS = 12
HISTORICAL_CEILING_SECONDS = 43_200

#: The reviewed operational ceiling.
AMENDED_CEILING_HOURS = 15
AMENDED_CEILING_SECONDS = 54_000

#: Who authorized it, and on what evidence.
AMENDMENT_AUTHORIZATION = {
    "authorized_by": "Phase 9 reviewing chat, Agent 6 review resolution",
    "authorized_on": "2026-08-17",
    "trigger": (
        "Agent 6 returned BLOCKED — CANONICAL WALL-CLOCK CONTRACT REQUIRES "
        "REVIEW because the winner-specific measured projection of the frozen "
        "60 x 2,048 canonical run (45,697 s mean-decisions / 46,757 s "
        "peak-decisions basis) exceeds the 43,200 s operational ceiling by "
        "42-59 minutes"
    ),
    "resolution": (
        "the pilot-selection result is formally accepted and P9-C remains the "
        "unique frozen winner; the operational ceiling alone is raised to "
        "54,000 s so the frozen experiment can run unchanged"
    ),
    "explicitly_not_authorized": [
        "rerunning, retraining or reevaluating any pilot",
        "additional pilot training or model selection",
        "changing the 60 iterations",
        "changing the 2,048 games per iteration",
        "changing the two optimizer epochs per rollout",
        "skipping or reducing any of the twelve validation passes",
        "changing the archive cadence",
        "changing the P9-C hyperparameters",
        "changing the population mixture or any seed",
        "changing the selection rule or the acceptance thresholds",
        "opening the sealed final-test bank",
        "editing the frozen phase9_rl_contract_v1 in place",
    ],
}

#: The one field this amendment changes, stated as a field path so a
#: reconciliation can be computed rather than described.
AMENDED_FIELD = "wall_clock_ceiling_hours"


class Phase9AmendmentError(Phase9ContractError):
    """The operational amendment is being used outside its reviewed scope."""


def amended_ceiling_seconds() -> int:
    """The operational ceiling Agent 7 runs under."""
    return AMENDED_CEILING_SECONDS


def unchanged_manifest() -> dict:
    """Everything the amendment leaves exactly as frozen, read from source.

    Read from `phase9_contract` rather than restated, so a later edit to a
    frozen quantity cannot hide behind this amendment's paperwork.
    """
    return {
        "canonical_iterations": CANONICAL_ITERATIONS,
        "canonical_games_per_iteration": CANONICAL_GAMES_PER_ITERATION,
        "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
        "validation_cadence_iterations": VALIDATION_CADENCE_ITERATIONS,
        "validation_passes": CANONICAL_ITERATIONS // VALIDATION_CADENCE_ITERATIONS,
        "archive_cadence_iterations": ARCHIVE_CADENCE_ITERATIONS,
        "winning_candidate_id": "P9-C",
        "learning_rate": 3e-4,
        "initial_kl_beta": 0.005,
        "population_version": "phase9_population_v1",
        "schedule_version": "phase9_rollout_schedule_v1",
        "acceptance_version": "phase9_acceptance_v1",
        "selection_rule": (
            "S = 0.45*E_strategic + 0.35*E_tactical + 0.20*E_phase8_anchor at "
            "the frozen final validation pass, with the frozen tie-break"
        ),
        "final_test_bank": "phase9_test_bank_v1 remains sealed until Agent 8",
    }


def amendment_document() -> dict:
    """The complete serializable `phase9_operational_amendment_v1`."""
    return {
        "amendment_version": PHASE9_OPERATIONAL_AMENDMENT_VERSION,
        "amends": {
            "contract_version": AMENDED_CONTRACT_VERSION,
            "contract_digest": AMENDED_CONTRACT_DIGEST,
            "in_place_edit": False,
            "rule": (
                "the base contract is not modified; its digest is stamped into "
                "every sealed rollout sidecar and every checkpoint, so an "
                "in-place edit would invalidate the accepted Agents 3-5 "
                "evidence"
            ),
        },
        "authorization": dict(AMENDMENT_AUTHORIZATION),
        "change": {
            "field": AMENDED_FIELD,
            "scope": "operational budget only; not a learning-design decision",
            "from_hours": HISTORICAL_CEILING_HOURS,
            "from_seconds": HISTORICAL_CEILING_SECONDS,
            "to_hours": AMENDED_CEILING_HOURS,
            "to_seconds": AMENDED_CEILING_SECONDS,
            "historical_value_preserved": True,
            "ceiling_rule": (
                "the amended ceiling remains an operational maximum, not "
                "permission to shorten the logical contract silently; an "
                "incomplete run still reports incomplete/blocked rather than "
                "pretending completion"
            ),
        },
        "unchanged": unchanged_manifest(),
        "affects_trainer_runtime_identity": False,
        "runtime_identity_rationale": (
            "Phase9TrainConfig.identity() carries no wall-clock field and its "
            "contract_digest entry reads the unmodified base contract, so the "
            "trainer runtime identity digest is unchanged by construction"
        ),
    }


def amendment_digest() -> str:
    """SHA-256 over the canonical JSON of the amendment document."""
    canonical = json.dumps(amendment_document(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_base_contract_untouched() -> list:
    """Every way the live contract disagrees with what this amendment assumes.

    The amendment is only meaningful while the thing it amends is unchanged,
    so this is measured on use rather than trusted.
    """
    problems: list = []
    observed = contract_digest()
    if observed != AMENDED_CONTRACT_DIGEST:
        problems.append(
            f"base contract digest {observed} != the amended-against "
            f"{AMENDED_CONTRACT_DIGEST}; the frozen contract was edited"
        )
    if CANONICAL_WALL_CLOCK_CEILING_HOURS != HISTORICAL_CEILING_HOURS:
        problems.append(
            f"phase9_contract still holds the historical ceiling as "
            f"{CANONICAL_WALL_CLOCK_CEILING_HOURS} h; the amendment requires it "
            f"to remain {HISTORICAL_CEILING_HOURS} h so history is preserved"
        )
    return problems


def runtime_identity_is_unaffected(identity_before: dict, identity_after: dict) -> dict:
    """Measure — never assert — that the trainer runtime identity is unchanged."""
    differing = sorted(
        key
        for key in set(identity_before) | set(identity_after)
        if identity_before.get(key) != identity_after.get(key)
    )
    return {
        "fields_compared": len(set(identity_before) | set(identity_after)),
        "differing_fields": differing,
        "unchanged": not differing,
        "carries_a_wall_clock_field": any(
            "wall_clock" in key or "ceiling" in key for key in identity_after
        ),
    }


def apply_to_train_config_document(document: dict) -> dict:
    """The amended train-config document: one field changed, nothing else.

    Returns a new document rather than mutating, so the original document (and
    therefore the original digest) stays available for the reconciliation.
    """
    if AMENDED_FIELD not in document:
        raise Phase9AmendmentError(
            f"train-config document has no {AMENDED_FIELD!r} field to amend"
        )
    if document[AMENDED_FIELD] != HISTORICAL_CEILING_HOURS:
        raise Phase9AmendmentError(
            f"train-config document holds {AMENDED_FIELD}="
            f"{document[AMENDED_FIELD]!r}; the amendment applies to the "
            f"historical {HISTORICAL_CEILING_HOURS}"
        )
    amended = dict(document)
    amended[AMENDED_FIELD] = AMENDED_CEILING_HOURS
    return amended


def reconcile_documents(original: dict, amended: dict) -> dict:
    """Field-by-field reconciliation of the original and amended documents."""
    keys = sorted(set(original) | set(amended))
    differing = [key for key in keys if original.get(key) != amended.get(key)]
    return {
        "fields_compared": len(keys),
        "changed_fields": [
            {
                "field": key,
                "original": original.get(key),
                "amended": amended.get(key),
            }
            for key in differing
        ],
        "unchanged_field_count": len(keys) - len(differing),
        "only_the_wall_clock_ceiling_changed": differing == [AMENDED_FIELD],
        "rule": (
            "the amended document is the accepted document with exactly one "
            "operational field rewritten; every learning-design field is "
            "byte-identical"
        ),
    }


__all__ = [
    "AMENDED_CEILING_HOURS",
    "AMENDED_CEILING_SECONDS",
    "AMENDED_CONTRACT_DIGEST",
    "AMENDED_CONTRACT_VERSION",
    "AMENDED_FIELD",
    "AMENDMENT_AUTHORIZATION",
    "HISTORICAL_CEILING_HOURS",
    "HISTORICAL_CEILING_SECONDS",
    "PHASE9_OPERATIONAL_AMENDMENT_VERSION",
    "Phase9AmendmentError",
    "amended_ceiling_seconds",
    "amendment_digest",
    "amendment_document",
    "apply_to_train_config_document",
    "reconcile_documents",
    "runtime_identity_is_unaffected",
    "unchanged_manifest",
    "verify_base_contract_untouched",
]
