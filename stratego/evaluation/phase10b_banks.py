"""Optional Phase 10B: the frozen validation and sealed test banks.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 16 and 20.

What a Phase 10B case is
------------------------
One logical paired case fixes everything about a comparison except the move
policy: the two per-colour selector seeds both arms draw their own side from,
one held-out opponent arrangement for the externally-opposed matchups, the
deterministic colour pairing over the two games, and the per-matchup match
seeds a rule-based opponent's randomness descends from. Two arms evaluated on
the same case therefore differ in exactly one thing — the checkpoint — which
is what makes the paired delta mean what the gates say it means.

Isolation
---------
A case may fix no arrangement whose final-setup fingerprint appears in the
accepted Phase 9 held-out banks or in either accepted Phase 10 bank. Phase 10B
does not claim an unseen base-template universe — the Phase 7 held-out splits
have been used before — but it does guarantee new case ids, new seeds, new
procedural descendants and zero exact final-setup fingerprint overlap with the
earlier held-out evaluation sets.

Sealing
-------
`build_bank("validation")` is open from the moment the contract is frozen.
`build_bank("test")` may be *constructed* and structurally audited at any
time; playing its cases is reserved to the single sealed final evaluation, and
the caller records that access before the first game.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..setups.contracts import SPLIT_TEST, SPLIT_VALIDATION
from ..setups.families import FAMILY_IDS
from ..setups.identity import content_fingerprint, orient_setup
from ..setups.sampler import SAMPLER_VERSION, load_library_index, sample_setup
from ..training.phase10b_contract import (
    MATCHUP_TOKENS,
    PHASE10B_TEST_BANK_VERSION,
    PHASE10B_VALIDATION_BANK_VERSION,
    Phase10BContractError,
    TEST_CASES_PER_MATCHUP,
    VALIDATION_CASES_PER_MATCHUP,
)
from ..training.phase10b_seed import (
    case_match_seed,
    case_opponent_seed,
    case_selector_seed,
    derive_seed,
    DOMAIN_VALIDATION_CASE,
)

NEUTRAL_PROFILE_NAME = "neutral_v1"
COLORS = ("red", "blue")
CASE_GAME_INDICES = (0, 1)

#: Game 0 seats the evaluated arm as Red, game 1 as Blue.
CASE_GAME_COLOR = {0: "red", 1: "blue"}

BANK_MAX_ATTEMPTS = 4096

BANK_SPECIFICATIONS = {
    "validation": {
        "bank_version": PHASE10B_VALIDATION_BANK_VERSION,
        "split": SPLIT_VALIDATION,
        "case_count": VALIDATION_CASES_PER_MATCHUP,
        "cases_per_family": VALIDATION_CASES_PER_MATCHUP // len(FAMILY_IDS),
        "access_justification": (
            "Phase 10B validation bank: frozen checkpoint-selection cases for "
            "the optional setup-conditioned fine-tuning experiment"
        ),
    },
    "test": {
        "bank_version": PHASE10B_TEST_BANK_VERSION,
        "split": SPLIT_TEST,
        "case_count": TEST_CASES_PER_MATCHUP,
        "cases_per_family": TEST_CASES_PER_MATCHUP // len(FAMILY_IDS),
        "access_justification": (
            "Phase 10B sealed final-test bank: structural construction and "
            "audit; outcome evaluation is reserved to the single sealed final "
            "evaluation"
        ),
    },
}

#: Held-out evaluation artifacts whose final-setup fingerprints Phase 10B may
#: not reuse. The Phase 9 pair is the accepted Phase 9 record; the Phase 10
#: pair is the formally closed Phase 10 one.
ISOLATION_ARTIFACTS = (
    ("phase9", "reports/phase_9_data/agent_01_validation_bank.json"),
    ("phase9", "reports/phase_9_data/agent_01_test_bank.json"),
    ("phase10", "reports/phase_10_data/agent_01_validation_bank.json"),
    ("phase10", "reports/phase_10_data/agent_01_test_bank.json"),
)

MANIFEST_VOLATILE_KEYS = ("construction_run", "manifest_digest")


class Phase10BBankError(Phase10BContractError):
    """A Phase 10B evaluation bank failed construction or audit."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bank_specification(bank: str) -> dict:
    try:
        return dict(BANK_SPECIFICATIONS[bank])
    except KeyError:
        raise Phase10BBankError(
            f"unknown Phase 10B bank {bank!r}; expected one of "
            f"{sorted(BANK_SPECIFICATIONS)}"
        ) from None


# ---------------------------------------------------------------------------
# The isolation set
# ---------------------------------------------------------------------------


def _fingerprints_from_artifact(payload: dict) -> list:
    """Every final-setup fingerprint one accepted bank artifact records.

    Phase 9 and Phase 10 wrote structurally different manifests, so both
    shapes are read explicitly rather than guessed at: a silently-empty
    isolation source would weaken the guarantee without failing anything.
    """
    manifest = payload.get("manifest", payload)
    found: list = []
    records = manifest.get("case_provenance")
    if records:
        for record in records:
            for side in COLORS:
                if side in record and "final_setup_fingerprint" in record[side]:
                    found.append(str(record[side]["final_setup_fingerprint"]))
    for container in (payload.get("cases"), manifest.get("cases")):
        for case in container or ():
            for value in case.get("frozen_fingerprints", ()):
                found.append(str(value))
    return found


def isolation_set(root: "Path | None" = None) -> "tuple[frozenset, dict]":
    """Every earlier held-out final-setup fingerprint, and the set's identity."""
    base = repository_root() if root is None else Path(root)
    fingerprints: set = set()
    sources = []
    for phase, relative in ISOLATION_ARTIFACTS:
        path = base / relative
        if not path.exists():
            raise Phase10BBankError(
                f"held-out bank artifact {relative} is missing; the Phase 10B "
                "isolation set cannot be established (BLOCKED)"
            )
        payload = json.loads(path.read_text())
        found = _fingerprints_from_artifact(payload)
        if not found:
            raise Phase10BBankError(
                f"{relative} contributed no final-setup fingerprints; the "
                "isolation set would be silently weaker than it claims (BLOCKED)"
            )
        before = len(fingerprints)
        fingerprints.update(found)
        sources.append(
            {
                "phase": phase,
                "artifact": relative,
                "fingerprints_read": len(found),
                "new_fingerprints": len(fingerprints) - before,
            }
        )
    ordered = sorted(fingerprints)
    manifest = {
        "isolation_set_version": "phase10b_heldout_fingerprints_v1",
        "sources": sources,
        "distinct_fingerprints": len(ordered),
        "set_digest": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "rule": (
            "no arrangement a Phase 10B case fixes may have a final setup "
            "fingerprint that any accepted Phase 9 or Phase 10 held-out "
            "evaluation case already fixed"
        ),
    }
    return frozenset(ordered), manifest


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def case_id(bank_version: str, family_id: str, ordinal: int) -> str:
    return f"{bank_version}|f={family_id}|c={int(ordinal):04d}"


@dataclass(frozen=True)
class Phase10BCase:
    """One frozen Phase 10B evaluation case."""

    case_id: str
    bank: str
    bank_version: str
    split: str
    family_id: str
    case_ordinal: int
    case_index: int
    opponent_setup: "tuple[int, ...]"
    opponent_attempt: int
    opponent_draw_seed: int
    opponent_provenance: dict
    selector_seeds: dict
    selector_seed_attempts: dict
    neutral_provenance: dict
    match_seeds: dict

    def oriented_opponent(self, player: int) -> "tuple[int, ...]":
        return orient_setup(self.opponent_setup, player)

    @property
    def frozen_fingerprints(self) -> tuple:
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
    family_index, _ = divmod(int(case_index), int(cases_per_family))
    if not 0 <= family_index < len(FAMILY_IDS):
        raise Phase10BBankError(
            f"case_index {case_index} is outside the "
            f"{len(FAMILY_IDS) * cases_per_family}-case id space"
        )
    return FAMILY_IDS[family_index]


def _attempt_seed(base: int, attempt: int) -> int:
    """The attempt-indexed descendant of one frozen case-side seed."""
    return derive_seed(DOMAIN_VALIDATION_CASE, "attempt", int(base), int(attempt))


def resolve_opponent_setup(
    identifier: str,
    bank_version: str,
    split: str,
    family_id: str,
    isolation: frozenset,
    *,
    index=None,
):
    """`(SampledSetup, attempt, seed)` for one case's fixed opponent setup.

    Accepts the first attempt that is family-correct and isolation-clean. Both
    conditions depend only on the case identity and one fixed external set, so
    the walk is order-independent and rebuilds exactly.
    """
    library = load_library_index() if index is None else index
    base = case_opponent_seed(bank_version, identifier, "red")
    for attempt in range(BANK_MAX_ATTEMPTS):
        seed = _attempt_seed(base, attempt)
        sampled = sample_setup(split, seed, profile=NEUTRAL_PROFILE_NAME, index=library)
        if sampled.family_id != family_id:
            continue
        if sampled.provenance["final_setup_fingerprint"] in isolation:
            continue
        return sampled, attempt, seed
    raise Phase10BBankError(
        f"{identifier}: no family-correct, isolation-clean opponent draw within "
        f"{BANK_MAX_ATTEMPTS} attempts (BLOCKED)"
    )


def resolve_selector_seed(
    identifier: str,
    bank_version: str,
    split: str,
    color: str,
    isolation: frozenset,
    excluded: frozenset,
    *,
    index=None,
):
    """`(SampledSetup, attempt, seed)` for one case-colour's selector seed.

    The accepted seed is the first attempt whose **`neutral_v1` baseline draw**
    is isolation-clean and distinct from the case's already-fixed
    arrangements. The walk reads only the baseline — a quantity fixed before
    any selector or any checkpoint exists — so it is arm-independent by
    construction and cannot favour either the Phase 10B checkpoint or the
    accepted Phase 9 one.
    """
    if color not in COLORS:
        raise Phase10BBankError(f"colour must be one of {list(COLORS)}, got {color!r}")
    library = load_library_index() if index is None else index
    base = case_selector_seed(bank_version, identifier, color)
    for attempt in range(BANK_MAX_ATTEMPTS):
        seed = _attempt_seed(base, attempt)
        sampled = sample_setup(split, seed, profile=NEUTRAL_PROFILE_NAME, index=library)
        fingerprint = sampled.provenance["final_setup_fingerprint"]
        if fingerprint in isolation or fingerprint in excluded:
            continue
        return sampled, attempt, seed
    raise Phase10BBankError(
        f"{identifier} {color}: no isolation-clean, distinct baseline draw within "
        f"{BANK_MAX_ATTEMPTS} attempts (BLOCKED)"
    )


def build_case(bank: str, case_index: int, isolation: frozenset, *, index=None) -> Phase10BCase:
    """One bank case, built from its identity and the fixed isolation set alone."""
    specification = bank_specification(bank)
    bank_version = specification["bank_version"]
    family_id = case_family(case_index, specification["cases_per_family"])
    ordinal = int(case_index) % specification["cases_per_family"]
    identifier = case_id(bank_version, family_id, ordinal)
    library = load_library_index() if index is None else index
    split = specification["split"]

    opponent, opponent_attempt, opponent_seed = resolve_opponent_setup(
        identifier, bank_version, split, family_id, isolation, index=library
    )
    excluded = {opponent.provenance["final_setup_fingerprint"]}
    selector_seeds: dict = {}
    selector_attempts: dict = {}
    neutral_provenance: dict = {}
    for color in COLORS:
        baseline, attempt, seed = resolve_selector_seed(
            identifier, bank_version, split, color, isolation, frozenset(excluded),
            index=library,
        )
        selector_seeds[color] = seed
        selector_attempts[color] = attempt
        neutral_provenance[color] = dict(baseline.provenance)
        excluded.add(baseline.provenance["final_setup_fingerprint"])

    return Phase10BCase(
        case_id=identifier,
        bank=bank,
        bank_version=bank_version,
        split=split,
        family_id=family_id,
        case_ordinal=ordinal,
        case_index=int(case_index),
        opponent_setup=tuple(opponent.canonical),
        opponent_attempt=opponent_attempt,
        opponent_draw_seed=opponent_seed,
        opponent_provenance=dict(opponent.provenance),
        selector_seeds=selector_seeds,
        selector_seed_attempts=selector_attempts,
        neutral_provenance=neutral_provenance,
        match_seeds={
            token: {
                index_: case_match_seed(bank_version, identifier, index_, token)
                for index_ in CASE_GAME_INDICES
            }
            for token in MATCHUP_TOKENS
        },
    )


# ---------------------------------------------------------------------------
# Whole banks
# ---------------------------------------------------------------------------


def bank_digest(cases) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"phase10b_bank_v1\n")
    for case in cases:
        hasher.update(
            json.dumps(case.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def manifest_digest(manifest: dict) -> str:
    stable = {
        key: value
        for key, value in manifest.items()
        if key not in MANIFEST_VOLATILE_KEYS
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_bank(bank: str, *, root: "Path | None" = None, index=None):
    """The complete frozen bank and its manifest, rebuilt from identity alone."""
    specification = bank_specification(bank)
    isolation, isolation_manifest = isolation_set(root)
    library = load_library_index() if index is None else index
    started = time.perf_counter()
    cases = tuple(
        build_case(bank, case_index, isolation, index=library)
        for case_index in range(specification["case_count"])
    )
    seconds = time.perf_counter() - started

    fingerprints: list = []
    for case in cases:
        fingerprints.extend(case.frozen_fingerprints)
    families: dict = {}
    for case in cases:
        families[case.family_id] = families.get(case.family_id, 0) + 1

    manifest = {
        "bank": bank,
        "bank_version": specification["bank_version"],
        "split": specification["split"],
        "case_count": len(cases),
        "cases_per_family": specification["cases_per_family"],
        "family_counts": families,
        "games_per_case": len(CASE_GAME_INDICES),
        "colour_pairing": {str(k): v for k, v in CASE_GAME_COLOR.items()},
        "matchups": list(MATCHUP_TOKENS),
        "sampler_version": SAMPLER_VERSION,
        "sampler_profile": NEUTRAL_PROFILE_NAME,
        "library_content_digest": library.content_digest,
        "access_justification": specification["access_justification"],
        "isolation": isolation_manifest,
        "distinct_frozen_fingerprints": len(set(fingerprints)),
        "total_frozen_fingerprints": len(fingerprints),
        "bootstrap_unit": "logical case",
        "bank_digest": bank_digest(cases),
        "construction_run": {"seconds": seconds},
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return cases, manifest


def audit_bank(cases, manifest: dict, *, root: "Path | None" = None) -> dict:
    """Every way a built Phase 10B bank can fail its own contract."""
    specification = bank_specification(manifest["bank"])
    isolation, _isolation_manifest = isolation_set(root)
    problems: list = []

    if len(cases) != specification["case_count"]:
        problems.append(
            f"{len(cases)} cases, the frozen bank holds {specification['case_count']}"
        )
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        problems.append("duplicate case ids")

    families: dict = {}
    for case in cases:
        families[case.family_id] = families.get(case.family_id, 0) + 1
    expected_per_family = specification["cases_per_family"]
    unbalanced = {
        family: count
        for family, count in families.items()
        if count != expected_per_family
    }
    if unbalanced or len(families) != len(FAMILY_IDS):
        problems.append(
            f"family balance is {families}, expected {expected_per_family} for each "
            f"of {len(FAMILY_IDS)} families"
        )

    fingerprints: list = []
    for case in cases:
        for value in case.frozen_fingerprints:
            fingerprints.append(value)
            if value in isolation:
                problems.append(f"{case.case_id}: fingerprint {value} is in the isolation set")
        if len(set(case.frozen_fingerprints)) != len(case.frozen_fingerprints):
            problems.append(f"{case.case_id}: repeats an arrangement within the case")
        if case.split != specification["split"]:
            problems.append(f"{case.case_id}: split {case.split!r}")
        if content_fingerprint(case.opponent_setup) != str(
            case.opponent_provenance["final_setup_fingerprint"]
        ):
            problems.append(f"{case.case_id}: opponent setup does not match its provenance")
        for token in MATCHUP_TOKENS:
            if token not in case.match_seeds:
                problems.append(f"{case.case_id}: no match seed for {token!r}")
        seeds = {
            seed
            for token in case.match_seeds
            for seed in case.match_seeds[token].values()
        }
        if len(seeds) != len(MATCHUP_TOKENS) * len(CASE_GAME_INDICES):
            problems.append(f"{case.case_id}: match seeds collide across matchups/games")

    if manifest["bank_digest"] != bank_digest(cases):
        problems.append("the manifest's bank digest is not this case set's")

    return {
        "bank": manifest["bank"],
        "bank_version": manifest["bank_version"],
        "cases": len(cases),
        "family_counts": families,
        "distinct_fingerprints": len(set(fingerprints)),
        "total_fingerprints": len(fingerprints),
        "isolation_clean": not any("isolation set" in problem for problem in problems),
        "problems": problems,
    }


def cross_bank_isolation(validation_cases, test_cases) -> dict:
    """Prove the two Phase 10B banks fix no arrangement in common."""
    left = {value for case in validation_cases for value in case.frozen_fingerprints}
    right = {value for case in test_cases for value in case.frozen_fingerprints}
    shared = sorted(left & right)
    return {
        "validation_fingerprints": len(left),
        "test_fingerprints": len(right),
        "shared": len(shared),
        "examples": shared[:5],
        "disjoint": not shared,
    }


__all__ = [
    "BANK_SPECIFICATIONS",
    "CASE_GAME_COLOR",
    "CASE_GAME_INDICES",
    "COLORS",
    "NEUTRAL_PROFILE_NAME",
    "Phase10BBankError",
    "Phase10BCase",
    "audit_bank",
    "bank_digest",
    "bank_specification",
    "build_bank",
    "build_case",
    "case_family",
    "case_id",
    "cross_bank_isolation",
    "isolation_set",
    "manifest_digest",
]
