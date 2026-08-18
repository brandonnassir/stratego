"""Phase 9: `phase9_operational_amendment_v2` — the reviewed 24-hour ceiling.

Specification source: the reviewing chat's Agent 7 review update, which
accepted that the canonical policy materially changed the runtime
distribution relative to the Agent 6 pilot projection — an **operational**
finding, not a training failure — and authorized **one** narrow change: the
canonical wall-clock ceiling moves from 54,000 s (15 h) to 86,400 s (24 h).

Why a third identity instead of an edit
---------------------------------------
Two earlier values are load-bearing history and neither may be rewritten:

```text
phase9_rl_contract_v1        12 h / 43,200 s   digest ad3dba3c…
phase9_operational_amendment_v1  15 h / 54,000 s   digest ee4b0507…
```

The contract digest is stamped into every `phase9_rollout_store_v1` metadata
sidecar and every `phase9_checkpoint_v1` this phase has written — by the time
this amendment exists, that includes the canonical run's own committed
iterations. Editing it would invalidate the very rollouts and checkpoints the
run is resuming from. `phase9_operational_amendment_v1` is the record of the
first review decision and of Agent 6's measured 12-hour BLOCKED finding; that
record stands on its own terms. So this amendment layers *beside* both,
carrying the digest of the exact base it amends, and
:func:`verify_chain_untouched` measures on every use that neither earlier
identity has moved.

Scope: operational, not experimental
------------------------------------
A wall-clock ceiling is an operations budget. It appears in no loss, no
schedule, no seed, no target and no acceptance threshold, and it is absent
from `Phase9TrainConfig.identity()` — so the trainer runtime identity digest
`77af4d45…` is unchanged by construction, exactly as it was under v1.

Everything scientific stays as frozen and accepted:

```text
60 RL iterations              2,048 scheduled games / iteration
2 optimizer epochs / rollout  twelve validation passes (cadence 5)
archive cadence 5             P9-C: LR 3e-4, initial KL beta 0.005
population mixture            all eight Phase 9 seeds
selection score + tie-break   final acceptance gates A-H
the sealed phase9_test_bank_v1 remains sealed until Agent 8
```

The ceiling is a maximum, never a training target: the run stops immediately
after iteration 60 and its required bookkeeping, and unused allowance is not
spent on extra games, extra updates, additional validation, hyperparameter
changes or experimentation.
"""

from __future__ import annotations

import hashlib
import json

from .phase9_amendment import (
    AMENDED_CEILING_HOURS as V1_CEILING_HOURS,
    AMENDED_CEILING_SECONDS as V1_CEILING_SECONDS,
    AMENDED_CONTRACT_DIGEST,
    AMENDED_FIELD,
    HISTORICAL_CEILING_HOURS as CONTRACT_CEILING_HOURS,
    HISTORICAL_CEILING_SECONDS as CONTRACT_CEILING_SECONDS,
    PHASE9_OPERATIONAL_AMENDMENT_VERSION as V1_VERSION,
    Phase9AmendmentError,
    amendment_digest as v1_amendment_digest,
    unchanged_manifest,
    verify_base_contract_untouched,
)
from .phase9_contract import (
    CANONICAL_WALL_CLOCK_CEILING_HOURS,
    Phase9ContractError,
)

#: This amendment's own frozen identity.
PHASE9_OPERATIONAL_AMENDMENT_V2_VERSION = "phase9_operational_amendment_v2"

#: The exact base this amendment applies to: v1, which itself amends the
#: contract. A drift in either is an amendment being read against something
#: it was never reviewed against.
AMENDED_AMENDMENT_VERSION = V1_VERSION
AMENDED_AMENDMENT_DIGEST = (
    "ee4b05078c676128f78c8e5c31bd10ce4f0841e34a57c4c7c3fca6616e083ac4"
)

#: The reviewed operational ceiling.
AMENDED_CEILING_HOURS = 24
AMENDED_CEILING_SECONDS = 86_400

#: Who authorized it, and on what evidence.
AMENDMENT_AUTHORIZATION = {
    "authorized_by": "Phase 9 reviewing chat, Agent 7 review update",
    "authorized_on": "2026-08-17",
    "trigger": (
        "the canonical run's measured cost per iteration rose from ~670 s "
        "early to ~900 s by iteration 25 as the learner-decision count per "
        "iteration grew from 282,414 to 324,990 and total plies from 380,564 "
        "to 431,214; at 25 committed iterations and 20,425 s consumed, every "
        "cost model projected the frozen 60-iteration experiment past the "
        "54,000 s ceiling (flat-cost model: halt during iteration 59; "
        "linear-trend model: halt during iteration 52)"
    ),
    "finding": (
        "the canonical policy materially changed the runtime distribution "
        "relative to the Agent 6 pilot projection, which was measured on "
        "1,024-game pilot iterations that never ran long enough to observe "
        "the change. This is an operational finding, not a training failure: "
        "no hard stop fired, no guard was breached, and the frozen validation "
        "score improved at every cadence point"
    ),
    "resolution": (
        "the running canonical experiment continues unchanged; the "
        "operational ceiling alone is raised to 86,400 s"
    ),
    "game_length_interpretation": (
        "increasing game length is reported as an observed runtime/behavioral "
        "change and is not by itself evidence of stronger play; strength "
        "claims remain grounded in the frozen validation results and the "
        "later Agent 8 final-test evaluation"
    ),
    "explicitly_not_authorized": [
        "extra games beyond the frozen 2,048 per iteration",
        "extra optimizer updates beyond the frozen two epochs",
        "additional validation passes beyond the frozen cadence",
        "additional archive members beyond the frozen cadence",
        "hyperparameter changes of any kind",
        "experimentation with unused wall-clock allowance",
        "changing the 60 iterations",
        "changing the population mixture or any seed",
        "changing the selection rule or the acceptance thresholds",
        "opening the sealed final-test bank",
        "editing phase9_rl_contract_v1 in place",
        "editing phase9_operational_amendment_v1 in place",
    ],
}


class Phase9AmendmentV2Error(Phase9AmendmentError):
    """The v2 operational amendment is being used outside its reviewed scope."""


def amended_ceiling_seconds() -> int:
    """The operational ceiling the canonical run now runs under."""
    return AMENDED_CEILING_SECONDS


def ceiling_history() -> list:
    """Every ceiling this phase has held, oldest first, all preserved."""
    return [
        {
            "authority": "phase9_rl_contract_v1",
            "digest": AMENDED_CONTRACT_DIGEST,
            "hours": CONTRACT_CEILING_HOURS,
            "seconds": CONTRACT_CEILING_SECONDS,
            "status": "original frozen value; preserved unedited",
        },
        {
            "authority": V1_VERSION,
            "digest": AMENDED_AMENDMENT_DIGEST,
            "hours": V1_CEILING_HOURS,
            "seconds": V1_CEILING_SECONDS,
            "status": (
                "first review-authorized operational amendment; preserved "
                "unedited, and the authority Agent 6's BLOCKED finding was "
                "resolved under"
            ),
        },
        {
            "authority": PHASE9_OPERATIONAL_AMENDMENT_V2_VERSION,
            "digest": None,  # filled by `amendment_digest()` at read time
            "hours": AMENDED_CEILING_HOURS,
            "seconds": AMENDED_CEILING_SECONDS,
            "status": "in force for the canonical run",
        },
    ]


def amendment_document() -> dict:
    """The complete serializable `phase9_operational_amendment_v2`."""
    return {
        "amendment_version": PHASE9_OPERATIONAL_AMENDMENT_V2_VERSION,
        "amends": {
            "amendment_version": AMENDED_AMENDMENT_VERSION,
            "amendment_digest": AMENDED_AMENDMENT_DIGEST,
            "base_contract_version": "phase9_rl_contract_v1",
            "base_contract_digest": AMENDED_CONTRACT_DIGEST,
            "in_place_edit": False,
            "rule": (
                "neither the base contract nor v1 is modified; the contract "
                "digest is stamped into every sealed rollout sidecar and every "
                "checkpoint the canonical run has already written, so an "
                "in-place edit would invalidate the state this run resumes from"
            ),
        },
        "authorization": dict(AMENDMENT_AUTHORIZATION),
        "change": {
            "field": AMENDED_FIELD,
            "scope": "operational budget only; not a learning-design decision",
            "from_hours": V1_CEILING_HOURS,
            "from_seconds": V1_CEILING_SECONDS,
            "to_hours": AMENDED_CEILING_HOURS,
            "to_seconds": AMENDED_CEILING_SECONDS,
            "historical_values_preserved": True,
            "ceiling_rule": (
                "the amended ceiling is an operational maximum, not a training "
                "target: the run stops immediately after iteration 60 and its "
                "required bookkeeping, and unused allowance is never spent on "
                "extra games, updates, validation, archive members or "
                "experimentation; an incomplete run still reports "
                "incomplete/blocked rather than pretending completion"
            ),
        },
        "ceiling_history": [
            {key: value for key, value in entry.items() if key != "digest"}
            | ({"digest": entry["digest"]} if entry["digest"] else {})
            for entry in ceiling_history()
        ],
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


def verify_chain_untouched() -> list:
    """Every way the live contract or v1 disagrees with what v2 assumes.

    A layered amendment is only meaningful while everything beneath it is
    unchanged, so the whole chain is measured on use rather than trusted.
    """
    problems = list(verify_base_contract_untouched())
    observed_v1 = v1_amendment_digest()
    if observed_v1 != AMENDED_AMENDMENT_DIGEST:
        problems.append(
            f"phase9_operational_amendment_v1 digest {observed_v1} != the "
            f"amended-against {AMENDED_AMENDMENT_DIGEST}; v1 was edited"
        )
    if V1_CEILING_SECONDS != 54_000:
        problems.append(
            f"phase9_operational_amendment_v1 no longer holds 54,000 s "
            f"({V1_CEILING_SECONDS}); v2 requires its value preserved"
        )
    if CANONICAL_WALL_CLOCK_CEILING_HOURS != CONTRACT_CEILING_HOURS:
        problems.append(
            f"phase9_contract still holds the original ceiling as "
            f"{CANONICAL_WALL_CLOCK_CEILING_HOURS} h; v2 requires it to remain "
            f"{CONTRACT_CEILING_HOURS} h so history is preserved"
        )
    return problems


def apply_to_train_config_document(document: dict) -> dict:
    """The v2-amended train-config document: one field changed, nothing else.

    Takes the *v1-amended* document (15 h) and returns a new one at 24 h, so
    both earlier documents and their digests remain available.
    """
    if AMENDED_FIELD not in document:
        raise Phase9AmendmentV2Error(
            f"train-config document has no {AMENDED_FIELD!r} field to amend"
        )
    if document[AMENDED_FIELD] != V1_CEILING_HOURS:
        raise Phase9AmendmentV2Error(
            f"train-config document holds {AMENDED_FIELD}="
            f"{document[AMENDED_FIELD]!r}; v2 applies to the v1-amended "
            f"{V1_CEILING_HOURS}"
        )
    amended = dict(document)
    amended[AMENDED_FIELD] = AMENDED_CEILING_HOURS
    return amended


def reconcile_documents(previous: dict, amended: dict) -> dict:
    """Field-by-field reconciliation of the v1-amended and v2-amended documents."""
    keys = sorted(set(previous) | set(amended))
    differing = [key for key in keys if previous.get(key) != amended.get(key)]
    return {
        "fields_compared": len(keys),
        "changed_fields": [
            {"field": key, "previous": previous.get(key), "amended": amended.get(key)}
            for key in differing
        ],
        "unchanged_field_count": len(keys) - len(differing),
        "only_the_wall_clock_ceiling_changed": differing == [AMENDED_FIELD],
        "rule": (
            "the v2 document is the v1-amended document with exactly one "
            "operational field rewritten; every learning-design field is "
            "byte-identical, and both earlier documents remain addressable"
        ),
    }


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


__all__ = [
    "AMENDED_AMENDMENT_DIGEST",
    "AMENDED_AMENDMENT_VERSION",
    "AMENDED_CEILING_HOURS",
    "AMENDED_CEILING_SECONDS",
    "AMENDMENT_AUTHORIZATION",
    "CONTRACT_CEILING_HOURS",
    "CONTRACT_CEILING_SECONDS",
    "PHASE9_OPERATIONAL_AMENDMENT_V2_VERSION",
    "Phase9AmendmentV2Error",
    "V1_CEILING_HOURS",
    "V1_CEILING_SECONDS",
    "amended_ceiling_seconds",
    "amendment_digest",
    "amendment_document",
    "apply_to_train_config_document",
    "ceiling_history",
    "reconcile_documents",
    "runtime_identity_is_unaffected",
    "verify_chain_untouched",
]
