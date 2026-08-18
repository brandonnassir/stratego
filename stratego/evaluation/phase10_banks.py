"""Phase 10 Agent 1: the frozen paired setup-selection evaluation banks.

Specification sources:

- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Build/freeze both
  evaluation banks", "Isolation audit", "Seal test-bank access")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Phase 10 evaluation
  banks", "Validation matchups")

What a Phase 10 case is
-----------------------
A Phase 9 bank case fixed *both* setups, because both sides were policies.
A Phase 10 case cannot: the whole experiment is about which setup a selector
chooses, so one side of every case is produced at evaluation time by the
selector under test. A case therefore fixes everything *except* that:

```text
one held-out opponent setup      family-conditioned, frozen, plays in every
                                 matchup and in both arms
two selector draw seeds          one per colour, identical for the learned
                                 candidate and the neutral_v1 baseline
two neutral own-side draws       what the baseline arm plays, derived from
                                 those same seeds; frozen and auditable
per-matchup match seeds          independent of arm and candidate
```

The colour pairing is frozen: the selector under test plays Red in game 0
and Blue in game 1 against the same opponent setup, so first-move advantage
cancels inside a case while setup-quality asymmetry — the thing being
measured — deliberately does not. The bootstrap unit is the case, scoring
the mean of its two games; a learned-minus-neutral difference is paired on
that unit.

```text
phase10_validation_bank_v1   128 cases   validation split    8 per family
phase10_test_bank_v1         512 cases   test split         32 per family
```

Case identity is family-major over the frozen family order `F00..F15`, so
family balance is a property of the id space rather than of any draw.

Deterministic rejection over the frozen sampler
-----------------------------------------------
The accepted `setup_sampler_v1` draws its family uniformly; this module
never bypasses or reweights it. An opponent setup walks
``attempt = 0, 1, 2, ...`` through
``sample_setup(split, case_opponent_setup_seed(case_id, attempt),
profile='neutral_v1')`` and accepts the first draw that is *family-correct*
and *isolation-clean*. A selector seed walks its own attempts and accepts
the first whose `neutral_v1` own-side draw is isolation-clean and distinct
from the case's other frozen setups. Both walks are pure functions of the
case identity and one fixed external fingerprint set, so a case rebuilds in
isolation and construction order can never matter.

Isolation
---------
Phase 10 does not claim a wholly unseen base-template universe — earlier
phases already used the same held-out base pool — so it claims exactly what
it can prove: new logical case ids, new Phase 10 seeds, new procedural
descendants, and **zero exact final-setup fingerprint overlap with the
Phase 9 held-out evaluation banks over every arrangement a Phase 10 case
fixes**. That set is the opponent setup and both neutral own-side draws:
every setup determined before a selector exists.

A learned selector's own-side draw is the one arrangement Agent 1 cannot
enumerate, because no learned selector exists yet. Rejecting such a draw at
evaluation time would distort the very mixed distribution the diversity
contract is stated over, so it is handled as a recorded report-only
diagnostic instead — :func:`phase9_isolation_set` is exported precisely so
Agents 5-7 can count those landings without inventing their own set.

Construction plays no game, loads no model and reads no outcome. That is
what makes building the sealed test bank here legal: it is the
`structural_audit` purpose the sealing rules allow every agent.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..setups.contracts import (
    SPLIT_TEST,
    SPLIT_VALIDATION,
    parse_base_setup_id,
    split_for_base_index,
)
from ..setups.families import FAMILY_IDS
from ..setups.identity import content_fingerprint
from ..setups.sampler import (
    SAMPLER_VERSION,
    load_library_index,
    rebuild_from_provenance,
    sample_setup,
)
from ..training.phase10_contract import (
    BANK_MAX_ATTEMPTS,
    MATCHUP_TOKENS,
    NEUTRAL_PROFILE_NAME,
    Phase10ContractError,
    TEST_BANK_CASES,
    TEST_BANK_VERSION,
    TEST_CASES_PER_FAMILY,
    VALIDATION_BANK_CASES,
    VALIDATION_BANK_VERSION,
    VALIDATION_CASES_PER_FAMILY,
)
from ..training.phase10_seed import (
    CASE_GAME_COLOR,
    CASE_GAME_INDICES,
    COLORS,
    PHASE10_MASTER_SEED,
    case_match_seed,
    case_opponent_setup_seed,
    case_selector_seed,
    parse_phase10_case_id,
    phase10_case_id,
)

#: The two Phase 10 banks, keyed by their short name.
BANK_SPECIFICATIONS = {
    "validation": {
        "bank_version": VALIDATION_BANK_VERSION,
        "split": SPLIT_VALIDATION,
        "cases_per_family": VALIDATION_CASES_PER_FAMILY,
        "case_count": VALIDATION_BANK_CASES,
        "access_justification": (
            "Phase 10 validation bank: frozen candidate-selection evaluation "
            "cases (Agent 1 construction)"
        ),
    },
    "test": {
        "bank_version": TEST_BANK_VERSION,
        "split": SPLIT_TEST,
        "cases_per_family": TEST_CASES_PER_FAMILY,
        "case_count": TEST_BANK_CASES,
        "access_justification": (
            "Phase 10 sealed final-test bank: frozen final-evaluation cases "
            "(Agent 1 structural construction; structural audit only before "
            "Agent 7)"
        ),
    },
}

#: The accepted Phase 9 held-out bank artifacts whose final-setup
#: fingerprints Phase 10 must not reuse.
PHASE9_BANK_ARTIFACTS = (
    "reports/phase_9_data/agent_01_validation_bank.json",
    "reports/phase_9_data/agent_01_test_bank.json",
)

#: Manifest keys deliberately outside the manifest digest: run measurements,
#: not bank identity.
MANIFEST_VOLATILE_KEYS = ("construction_run", "manifest_digest")

MANIFEST_DIGEST_DOMAIN = "stratego_phase10_bank_manifest_v1"
BANK_DIGEST_DOMAIN = "stratego_phase10_bank_v1"

_SIDE_PLAYER = {"red": 0, "blue": 1}


class Phase10BankError(Phase10ContractError):
    """A Phase 10 evaluation bank failed construction or audit."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bank_specification(bank: str) -> dict:
    try:
        return dict(BANK_SPECIFICATIONS[bank])
    except KeyError:
        raise Phase10BankError(
            f"unknown Phase 10 bank {bank!r}; expected one of "
            f"{sorted(BANK_SPECIFICATIONS)}"
        ) from None


# ---------------------------------------------------------------------------
# The Phase 9 isolation set
# ---------------------------------------------------------------------------


def phase9_isolation_set(root: "Path | None" = None) -> "tuple[frozenset[str], dict]":
    """Every Phase 9 held-out evaluation final-setup fingerprint, and its identity.

    Read from the accepted Phase 9 Agent 1 bank artifacts rather than
    recomputed, because those artifacts *are* the accepted record of which
    arrangements Phase 9 evaluated on. The returned manifest pins the set
    itself with a digest, so the rejection rule a case was built under is as
    identifiable as the case.
    """
    base = repository_root() if root is None else Path(root)
    fingerprints: set[str] = set()
    sources = []
    for relative in PHASE9_BANK_ARTIFACTS:
        path = base / relative
        if not path.exists():
            raise Phase10BankError(
                f"accepted Phase 9 bank artifact {relative} is missing; the "
                "Phase 10 isolation set cannot be established (BLOCKED)"
            )
        payload = json.loads(path.read_text())
        records = payload["manifest"]["case_provenance"]
        before = len(fingerprints)
        for record in records:
            for side in ("red", "blue"):
                fingerprints.add(str(record[side]["final_setup_fingerprint"]))
        sources.append(
            {
                "artifact": relative,
                "bank_version": payload["manifest"]["bank_version"],
                "split": payload["manifest"]["split"],
                "case_count": len(records),
                "sides": 2 * len(records),
                "new_fingerprints": len(fingerprints) - before,
            }
        )
    ordered = sorted(fingerprints)
    manifest = {
        "isolation_set_version": "phase9_heldout_final_setup_fingerprints_v1",
        "sources": sources,
        "distinct_fingerprints": len(ordered),
        "set_digest": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "rule": (
            "no arrangement a Phase 10 case fixes may have a final setup "
            "fingerprint in this set"
        ),
    }
    return frozenset(ordered), manifest


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase10Case:
    """One frozen Phase 10 evaluation case."""

    case_id: str
    bank: str
    bank_version: str
    split: str
    family_id: str
    case_ordinal: int
    case_index: int
    #: The fixed held-out opponent setup, in the canonical own-orientation frame.
    opponent_setup: "tuple[int, ...]"
    opponent_attempt: int
    opponent_draw_seed: int
    opponent_provenance: dict
    #: Accepted selector draw seed per colour, and the attempt that produced it.
    selector_seeds: dict
    selector_seed_attempts: dict
    #: The `neutral_v1` baseline's own-side draw per colour.
    neutral_provenance: dict
    #: Per-matchup, per-game match seeds; arm- and candidate-independent.
    match_seeds: dict

    def oriented_opponent(self, player: int) -> "tuple[int, ...]":
        """The engine-ready opponent setup for `player`."""
        from ..setups.identity import orient_setup

        return orient_setup(self.opponent_setup, player)

    @property
    def frozen_fingerprints(self) -> "tuple[str, ...]":
        """Every final-setup fingerprint this case fixes, in a stable order."""
        return (
            str(self.opponent_provenance["final_setup_fingerprint"]),
            str(self.neutral_provenance["red"]["final_setup_fingerprint"]),
            str(self.neutral_provenance["blue"]["final_setup_fingerprint"]),
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "bank": self.bank,
            "bank_version": self.bank_version,
            "split": self.split,
            "family_id": self.family_id,
            "case_ordinal": self.case_ordinal,
            "case_index": self.case_index,
            "bootstrap_unit": self.case_id,
            "colour_pairing": {
                str(index): CASE_GAME_COLOR[index] for index in CASE_GAME_INDICES
            },
            "opponent_setup": list(self.opponent_setup),
            "opponent_attempt": self.opponent_attempt,
            "opponent_draw_seed": self.opponent_draw_seed,
            "opponent_provenance": self.opponent_provenance,
            "selector_seeds": dict(self.selector_seeds),
            "selector_seed_attempts": dict(self.selector_seed_attempts),
            "neutral_provenance": {
                color: self.neutral_provenance[color] for color in COLORS
            },
            "match_seeds": {
                token: {str(index): seeds[index] for index in CASE_GAME_INDICES}
                for token, seeds in sorted(self.match_seeds.items())
            },
            "frozen_fingerprints": list(self.frozen_fingerprints),
        }


def case_family(case_index: int, cases_per_family: int) -> str:
    """The frozen family of one case index: family-major over `F00..F15`."""
    family_index, _ = divmod(int(case_index), int(cases_per_family))
    if not 0 <= family_index < len(FAMILY_IDS):
        raise Phase10BankError(
            f"case_index {case_index} is outside the "
            f"{len(FAMILY_IDS) * cases_per_family}-case id space"
        )
    return FAMILY_IDS[family_index]


def _match_seeds(case_id: str) -> dict:
    return {
        token: {index: case_match_seed(case_id, index, token) for index in CASE_GAME_INDICES}
        for token in MATCHUP_TOKENS
    }


def resolve_opponent_setup(
    case_id: str,
    split: str,
    family_id: str,
    isolation: "frozenset[str]",
    *,
    index=None,
):
    """`(SampledSetup, attempt, seed)` for one case's fixed opponent setup.

    Accepts the first attempt that is family-correct and whose final setup
    fingerprint is outside the Phase 9 held-out set. Both conditions depend
    only on the case identity and one fixed external set, so the walk is
    order-independent and rebuilds exactly.
    """
    library = load_library_index() if index is None else index
    for attempt in range(BANK_MAX_ATTEMPTS):
        seed = case_opponent_setup_seed(case_id, attempt)
        sampled = sample_setup(
            split, seed, profile=NEUTRAL_PROFILE_NAME, index=library
        )
        if sampled.family_id != family_id:
            continue
        if sampled.provenance["final_setup_fingerprint"] in isolation:
            continue
        return sampled, attempt, seed
    raise Phase10BankError(
        f"{case_id}: no family-correct, isolation-clean opponent draw within "
        f"{BANK_MAX_ATTEMPTS} attempts; the library, sampler or isolation set "
        "has drifted (BLOCKED)"
    )


def resolve_selector_seed(
    case_id: str,
    split: str,
    color: str,
    isolation: "frozenset[str]",
    excluded: "frozenset[str]",
    *,
    index=None,
):
    """`(SampledSetup, attempt, seed)` for one case-colour's selector seed.

    The accepted seed is the first attempt whose **`neutral_v1` baseline
    draw** is isolation-clean and distinct from the case's already-fixed
    arrangements. The walk reads only the baseline — a quantity fixed before
    any selector exists — so it is arm-independent by construction and can
    neither know about nor favour a learned candidate.
    """
    if color not in COLORS:
        raise Phase10BankError(f"colour must be one of {list(COLORS)}, got {color!r}")
    library = load_library_index() if index is None else index
    for attempt in range(BANK_MAX_ATTEMPTS):
        seed = case_selector_seed(case_id, color, attempt)
        sampled = sample_setup(
            split, seed, profile=NEUTRAL_PROFILE_NAME, index=library
        )
        fingerprint = sampled.provenance["final_setup_fingerprint"]
        if fingerprint in isolation or fingerprint in excluded:
            continue
        return sampled, attempt, seed
    raise Phase10BankError(
        f"{case_id} {color}: no isolation-clean, distinct baseline draw within "
        f"{BANK_MAX_ATTEMPTS} attempts (BLOCKED)"
    )


def build_case(
    bank: str,
    case_index: int,
    isolation: "frozenset[str]",
    *,
    index=None,
) -> Phase10Case:
    """One bank case, built from its identity and the fixed isolation set alone."""
    specification = bank_specification(bank)
    family_id = case_family(case_index, specification["cases_per_family"])
    case_ordinal = int(case_index) % specification["cases_per_family"]
    case_id = phase10_case_id(specification["bank_version"], family_id, case_ordinal)
    library = load_library_index() if index is None else index
    split = specification["split"]

    opponent, opponent_attempt, opponent_seed = resolve_opponent_setup(
        case_id, split, family_id, isolation, index=library
    )

    excluded = {opponent.provenance["final_setup_fingerprint"]}
    selector_seeds: dict = {}
    selector_attempts: dict = {}
    neutral_provenance: dict = {}
    for color in COLORS:
        baseline, attempt, seed = resolve_selector_seed(
            case_id, split, color, isolation, frozenset(excluded), index=library
        )
        selector_seeds[color] = seed
        selector_attempts[color] = attempt
        neutral_provenance[color] = dict(baseline.provenance)
        excluded.add(baseline.provenance["final_setup_fingerprint"])

    return Phase10Case(
        case_id=case_id,
        bank=bank,
        bank_version=specification["bank_version"],
        split=split,
        family_id=family_id,
        case_ordinal=case_ordinal,
        case_index=int(case_index),
        opponent_setup=tuple(opponent.canonical),
        opponent_attempt=opponent_attempt,
        opponent_draw_seed=opponent_seed,
        opponent_provenance=dict(opponent.provenance),
        selector_seeds=selector_seeds,
        selector_seed_attempts=selector_attempts,
        neutral_provenance=neutral_provenance,
        match_seeds=_match_seeds(case_id),
    )


def bank_digest(cases: "tuple[Phase10Case, ...]") -> str:
    """SHA-256 over the bank's complete case content — its stable identity."""
    payload = {
        "domain": BANK_DIGEST_DOMAIN,
        "master_seed": PHASE10_MASTER_SEED,
        "cases": [case.to_dict() for case in cases],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def manifest_digest(manifest: dict) -> str:
    """SHA-256 over the manifest's identity fields (volatile keys excluded)."""
    stable = {
        key: value for key, value in manifest.items() if key not in MANIFEST_VOLATILE_KEYS
    }
    payload = MANIFEST_DIGEST_DOMAIN + "\n" + json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_phase10_bank(
    bank: str,
    isolation: "frozenset[str] | None" = None,
    isolation_manifest: "dict | None" = None,
) -> "tuple[tuple[Phase10Case, ...], dict]":
    """The complete frozen bank and its manifest.

    Reproducible from frozen constants and the accepted Phase 9 artifacts
    alone; two builds yield identical bytes and identical digests, which the
    audit re-proves by rebuilding a deterministic sample of cases.
    """
    specification = bank_specification(bank)
    if isolation is None:
        isolation, isolation_manifest = phase9_isolation_set()
    library = load_library_index()
    started = time.time()

    cases = tuple(
        build_case(bank, case_index, isolation, index=library)
        for case_index in range(specification["case_count"])
    )

    opponent_attempts: dict = {}
    selector_attempts: dict = {}
    for case in cases:
        opponent_attempts[case.opponent_attempt] = (
            opponent_attempts.get(case.opponent_attempt, 0) + 1
        )
        for color in COLORS:
            attempt = case.selector_seed_attempts[color]
            selector_attempts[attempt] = selector_attempts.get(attempt, 0) + 1

    manifest = {
        "bank": bank,
        "bank_version": specification["bank_version"],
        "split": specification["split"],
        "case_count": specification["case_count"],
        "cases_per_opponent_family": specification["cases_per_family"],
        "family_ids": list(FAMILY_IDS),
        "case_id_rule": (
            "case_index = family_index * cases_per_family + case_ordinal over "
            "families F00..F15; case_id = "
            "'<bank_version>|ms=<master>|f=<family>|c=<ordinal:03d>'"
        ),
        "case_family_meaning": "the fixed opponent setup's primary family",
        "colour_pairing": (
            "the selector under test plays Red in game 0 and Blue in game 1 "
            "against the same fixed opponent setup"
        ),
        "bootstrap_unit": "the logical case, scoring the mean of its two games",
        "sampler_version": SAMPLER_VERSION,
        "sampler_profile": NEUTRAL_PROFILE_NAME,
        "library_content_digest": library.content_digest,
        "opponent_draw_rule": (
            "first attempt k with sample_setup(split, "
            "case_opponent_setup_seed(case_id, k), profile='neutral_v1') whose "
            "primary family is the case family and whose final setup fingerprint "
            "is outside the Phase 9 held-out set"
        ),
        "selector_seed_rule": (
            "first attempt k whose neutral_v1 draw for "
            "case_selector_seed(case_id, colour, k) is outside the Phase 9 "
            "held-out set and distinct from the case's other frozen setups; the "
            "walk reads the baseline only, so it is arm-independent"
        ),
        "match_seed_rule": (
            "case_match_seed(case_id, game_index, matchup_token), independent of "
            "arm and candidate"
        ),
        "matchup_tokens": list(MATCHUP_TOKENS),
        "max_attempts": BANK_MAX_ATTEMPTS,
        "isolation": isolation_manifest,
        "access_justification": specification["access_justification"],
        "no_outcome_selection": (
            "construction plays no game, loads no model and reads no strength "
            "signal; draws are rejected only for family identity, fingerprint "
            "isolation and within-case distinctness"
        ),
        "bank_digest": bank_digest(cases),
        "opponent_attempt_histogram": {
            str(attempt): count for attempt, count in sorted(opponent_attempts.items())
        },
        "selector_seed_attempt_histogram": {
            str(attempt): count for attempt, count in sorted(selector_attempts.items())
        },
        "construction_run": {"duration_seconds": round(time.time() - started, 3)},
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return cases, manifest


# ---------------------------------------------------------------------------
# Structural audit
# ---------------------------------------------------------------------------


def audit_phase10_bank(
    bank: str,
    cases: "tuple[Phase10Case, ...]",
    manifest: dict,
    isolation: "frozenset[str] | None" = None,
    *,
    rebuild_sample_every: int = 16,
) -> dict:
    """Recompute every structural bank property from stored content.

    Structural only: no game is played, no model is loaded, no outcome is
    read — exactly the `structural_audit` purpose the sealing rules allow
    every agent, which is what makes auditing the sealed test bank legal
    here. `rebuild_sample_every` controls the isolated-rebuild spot check;
    1 rebuilds every case.
    """
    specification = bank_specification(bank)
    if isolation is None:
        isolation, _ = phase9_isolation_set()
    library = load_library_index()

    failures: list = []
    family_counts: dict = {family_id: 0 for family_id in FAMILY_IDS}
    split_violations: list = []
    family_violations: list = []
    provenance_mismatches: list = []
    rebuild_mismatches: list = []
    isolation_overlaps: list = []
    within_case_duplicates: list = []
    seed_reuse: list = []
    engine_failures: list = []

    all_seeds: dict = {}
    fingerprint_counts: dict = {}

    for case in cases:
        expected_family = case_family(case.case_index, specification["cases_per_family"])
        family_counts[expected_family] += 1
        if case.family_id != expected_family:
            failures.append(
                f"{case.case_id}: family {case.family_id!r} disagrees with the id "
                f"rule {expected_family!r}"
            )
        parsed = parse_phase10_case_id(case.case_id)
        if parsed["bank_version"] != specification["bank_version"]:
            failures.append(f"{case.case_id}: bank version {parsed['bank_version']!r}")
        if parsed["case_ordinal"] != case.case_ordinal:
            failures.append(f"{case.case_id}: ordinal disagrees with the id")

        records = [("opponent", case.opponent_provenance)] + [
            (color, case.neutral_provenance[color]) for color in COLORS
        ]
        for label, record in records:
            if record["split"] != specification["split"]:
                split_violations.append(
                    f"{case.case_id} {label}: split {record['split']!r}"
                )
            _, _, base_index = parse_base_setup_id(record["base_setup_id"])
            if split_for_base_index(base_index) != specification["split"]:
                split_violations.append(
                    f"{case.case_id} {label}: base index {base_index} is not a "
                    f"{specification['split']!r} base"
                )
            if record["final_setup_fingerprint"] in isolation:
                isolation_overlaps.append(f"{case.case_id} {label}")
            fingerprint_counts[record["final_setup_fingerprint"]] = (
                fingerprint_counts.get(record["final_setup_fingerprint"], 0) + 1
            )
            try:
                rebuilt = rebuild_from_provenance(record, index=library)
            except Exception as error:  # noqa: BLE001 - a failed rebuild is a finding
                provenance_mismatches.append(
                    f"{case.case_id} {label}: rebuild failed: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            if content_fingerprint(rebuilt.canonical) != record["final_setup_fingerprint"]:
                provenance_mismatches.append(
                    f"{case.case_id} {label}: provenance does not rebuild its "
                    "recorded fingerprint"
                )

        if case.opponent_provenance["primary_family_id"] != expected_family:
            family_violations.append(
                f"{case.case_id} opponent: family "
                f"{case.opponent_provenance['primary_family_id']!r}"
            )
        if tuple(case.opponent_setup) != tuple(
            rebuild_from_provenance(case.opponent_provenance, index=library).canonical
        ):
            provenance_mismatches.append(
                f"{case.case_id} opponent: stored setup differs from its rebuild"
            )
        engine_failures.extend(
            f"{case.case_id} opponent: {failure}"
            for failure in _validate_case_setup(case.opponent_setup)
        )

        fingerprints = case.frozen_fingerprints
        if len(set(fingerprints)) != len(fingerprints):
            within_case_duplicates.append(case.case_id)

        for color in COLORS:
            key = ("selector", case.selector_seeds[color])
            if key in all_seeds:
                seed_reuse.append(f"{case.case_id} {color} selector seed reused")
            all_seeds[key] = case.case_id
        opponent_key = ("opponent", case.opponent_draw_seed)
        if opponent_key in all_seeds:
            seed_reuse.append(f"{case.case_id} opponent draw seed reused")
        all_seeds[opponent_key] = case.case_id
        for token, seeds in case.match_seeds.items():
            for game_index, seed in seeds.items():
                key = ("match", seed)
                if key in all_seeds:
                    seed_reuse.append(
                        f"{case.case_id} {token} game {game_index} match seed reused"
                    )
                all_seeds[key] = case.case_id

        if case.case_index % max(1, int(rebuild_sample_every)) == 0:
            rebuilt_case = build_case(bank, case.case_index, isolation, index=library)
            if rebuilt_case != case:
                rebuild_mismatches.append(f"{case.case_id}: isolated rebuild differs")

    observed_digest = bank_digest(cases)
    repeated = {
        fingerprint: count
        for fingerprint, count in fingerprint_counts.items()
        if count > 1
    }

    checks = {
        "case_count_exact": len(cases) == specification["case_count"],
        "case_indices_contiguous": tuple(case.case_index for case in cases)
        == tuple(range(specification["case_count"])),
        "family_balance_exact": all(
            count == specification["cases_per_family"] for count in family_counts.values()
        ),
        "opponent_family_correct": not family_violations,
        "split_isolation": not split_violations,
        "engine_valid": not engine_failures,
        "provenance_rebuilds": not provenance_mismatches,
        "isolated_rebuild_exact": not rebuild_mismatches,
        "phase9_fingerprint_overlap_zero": not isolation_overlaps,
        "within_case_fingerprints_distinct": not within_case_duplicates,
        "seeds_unique": not seed_reuse,
        "digest_matches_manifest": observed_digest == manifest["bank_digest"],
        "manifest_digest_consistent": manifest_digest(manifest)
        == manifest["manifest_digest"],
        "no_structural_failures": not failures,
    }

    return {
        "bank": bank,
        "bank_version": specification["bank_version"],
        "case_count": len(cases),
        "family_counts": family_counts,
        "bank_digest": observed_digest,
        "frozen_setups_per_case": 3,
        "frozen_setups_total": 3 * len(cases),
        "distinct_frozen_fingerprints": len(fingerprint_counts),
        "repeated_within_bank_fingerprints": len(repeated),
        "repeated_within_bank_note": (
            "a repeat across two different cases is a property of the frozen "
            "sampler's support, not an isolation failure; only within-case "
            "duplication and Phase 9 overlap are rejected"
        ),
        "failures": failures,
        "engine_failures": engine_failures,
        "family_violations": family_violations,
        "split_violations": split_violations,
        "provenance_mismatches": provenance_mismatches,
        "rebuild_mismatches": rebuild_mismatches,
        "isolation_overlaps": isolation_overlaps,
        "within_case_duplicates": within_case_duplicates,
        "seed_reuse": seed_reuse,
        "rebuild_sample_every": int(rebuild_sample_every),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _validate_case_setup(canonical: "tuple[int, ...]") -> "list[str]":
    """Engine legality of one stored arrangement, recomputed from scratch."""
    from ..engine.setup import validate_setup
    from ..setups.mobility import setup_has_initial_mobility

    failures: list[str] = []
    try:
        validate_setup(tuple(canonical), 0)
    except Exception as error:  # noqa: BLE001 - an invalid setup is a finding
        failures.append(f"inventory/legality: {type(error).__name__}: {error}")
        return failures
    if not setup_has_initial_mobility(tuple(canonical)):
        failures.append("stranded: no initial legal move for the owner")
    return failures


def cross_bank_isolation(
    validation_cases: "tuple[Phase10Case, ...]",
    test_cases: "tuple[Phase10Case, ...]",
) -> dict:
    """Fingerprint overlap between the two Phase 10 banks.

    Required to be empty. The two banks draw from disjoint Phase 7 splits, so
    an overlap could only come from two different bases producing the same
    perturbed arrangement — vanishingly unlikely, and precisely the kind of
    assumption worth checking rather than asserting.
    """
    validation_fingerprints = {
        fingerprint for case in validation_cases for fingerprint in case.frozen_fingerprints
    }
    test_fingerprints = {
        fingerprint for case in test_cases for fingerprint in case.frozen_fingerprints
    }
    overlap = sorted(validation_fingerprints & test_fingerprints)
    return {
        "validation_distinct_fingerprints": len(validation_fingerprints),
        "test_distinct_fingerprints": len(test_fingerprints),
        "overlap_count": len(overlap),
        "overlap": overlap,
        "zero_overlap": not overlap,
    }


__all__ = [
    "BANK_DIGEST_DOMAIN",
    "BANK_SPECIFICATIONS",
    "MANIFEST_DIGEST_DOMAIN",
    "MANIFEST_VOLATILE_KEYS",
    "PHASE9_BANK_ARTIFACTS",
    "Phase10BankError",
    "Phase10Case",
    "audit_phase10_bank",
    "bank_digest",
    "bank_specification",
    "build_case",
    "build_phase10_bank",
    "case_family",
    "cross_bank_isolation",
    "manifest_digest",
    "phase9_isolation_set",
    "repository_root",
    "resolve_opponent_setup",
    "resolve_selector_seed",
]
