"""Materialization, verification and manifest of `setup_library_v1`.

Specification sources:

- `02_AGENT_2_BASE_LIBRARY_GENERATOR.md` (global uniqueness, split handling,
  materialized library, base-generation diversity checks)
- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (canonical Phase 7 data files)
- Agent 1's frozen serialization contract in `contracts.py`

What lives here
---------------
`generator.py` knows how to build one base setup from its identity. This
module turns that into the production artifact: it enumerates the library in
canonical order, enforces the library-wide acceptance gates, writes the JSONL
and its manifest with deterministic bytes, and re-verifies everything from
stored content rather than from generation counters.

Global uniqueness is a gate, not a generation input
---------------------------------------------------
Agent 1 froze cross-base independence: a base setup may never condition on
another base's outcome, because that is what makes isolated rebuild exact. So
this module does not reroll a colliding base — it raises. A duplicate
canonical fingerprint anywhere in the 8,000 entries (including across
families), a repeated stable id, or a repeated exact arrangement is a BLOCKED
finding for review, exactly as the contract requires. The frozen master seed
produces none.

Determinism of the bytes
------------------------
Entries are emitted in the frozen file order (families `F00..F15`, base index
ascending) using the frozen canonical line format, so rewriting the same
library reproduces the same bytes and the same digests. Timestamps, durations
and host measurements live in the manifest's `generation_run` section, which
is deliberately outside the manifest-digest domain.
"""

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..engine.setup import SetupError, validate_setup
from .contracts import (
    BASE_ENTRY_REQUIRED_FIELDS,
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    SPLITS,
    TEST_PER_FAMILY,
    TRAIN_PER_FAMILY,
    VALIDATION_PER_FAMILY,
    base_entry_json_line,
    base_setup_id,
    library_digest,
    split_for_base_index,
)
from .families import FAMILY_BY_ID, FAMILY_IDS, family_contract
from .identity import (
    SetupLibraryError,
    canonical_class_representative,
    class_fingerprint,
    content_fingerprint,
    is_canonical_representative,
    reflect_canonical,
)
from .mobility import setup_has_initial_mobility
from .seed import DEFAULT_SEED_CONTEXT, LibrarySeedContext
from .traits import compute_trait_vector
from .generator import (
    GENERATOR_VERSION,
    REJECTION_REASONS,
    BaseSetupEntry,
    generate_base_setup,
    plans_document,
)

#: Domain prefix of the entry-metadata digest, so it can never be confused
#: with the identity digest defined by Agent 1's `library_digest`.
ENTRY_METADATA_DIGEST_DOMAIN = "stratego_setup_library_entries_v1"

#: Domain prefix of the manifest digest.
MANIFEST_DIGEST_DOMAIN = "stratego_setup_library_manifest_v1"

#: Manifest sections deliberately excluded from the manifest digest: they are
#: run measurements (wall time, timestamp, host), not library identity.
MANIFEST_VOLATILE_KEYS = ("generation_run", "manifest_digest")

#: Substrings that would betray an outcome, strength or learned signal in a
#: serialized entry. Phase 7 selects setups structurally, so none may appear.
#: `verify_library` checks the actual serialized field names against this list.
FORBIDDEN_ENTRY_FIELD_TOKENS = (
    "win",
    "loss",
    "elo",
    "rating",
    "outcome",
    "result",
    "reward",
    "score",
    "policy",
    "value_",
    "strength",
    "preference",
)


def library_order() -> "list[tuple[str, int]]":
    """The frozen enumeration order: families `F00..F15`, index ascending."""
    return [
        (family_id, base_index)
        for family_id in FAMILY_IDS
        for base_index in range(BASES_PER_FAMILY)
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryGenerationResult:
    """A fully generated library plus its deterministic generation statistics."""

    entries: "tuple[BaseSetupEntry, ...]"
    seed_context: LibrarySeedContext
    #: `family_id -> reason -> count` over every rejected candidate attempt.
    rejections_by_family: dict = field(default_factory=dict)
    #: `attempts -> number of bases accepted at that attempt count`.
    attempt_histogram: dict = field(default_factory=dict)
    duration_seconds: float = 0.0

    def rejections_by_reason(self) -> dict:
        totals: Counter = Counter()
        for reasons in self.rejections_by_family.values():
            totals.update(reasons)
        return {reason: totals.get(reason, 0) for reason in REJECTION_REASONS}

    def total_attempts(self) -> int:
        return sum(entry.generation_attempts for entry in self.entries)

    def attempts_per_accepted_base(self) -> float:
        return round(self.total_attempts() / len(self.entries), 6) if self.entries else 0.0

    def rejection_rate_by_family(self) -> dict:
        rates = {}
        for family_id in FAMILY_IDS:
            rejected = sum(self.rejections_by_family.get(family_id, {}).values())
            accepted = sum(1 for entry in self.entries if entry.family_id == family_id)
            attempts = rejected + accepted
            rates[family_id] = round(rejected / attempts, 6) if attempts else 0.0
        return rates


def _enforce_global_uniqueness(entries: "list[BaseSetupEntry]") -> None:
    """Reject a library that repeats an identity anywhere, including cross-family.

    Raises rather than regenerating: under the frozen contract a collision is
    a BLOCKED finding, because rerolling one base against another base's
    outcome would destroy isolated rebuild.
    """
    seen_fingerprint: dict[str, str] = {}
    seen_setup: dict[tuple, str] = {}
    seen_identifier: set[str] = set()
    for entry in entries:
        if entry.base_setup_id in seen_identifier:
            raise SetupLibraryError(
                f"stable id collision: {entry.base_setup_id} occurs twice"
            )
        seen_identifier.add(entry.base_setup_id)
        previous = seen_fingerprint.get(entry.fingerprint)
        if previous is not None:
            raise SetupLibraryError(
                "reflection-class fingerprint collision between "
                f"{previous} and {entry.base_setup_id}: under the frozen "
                "cross-base-independence rule this is a BLOCKED finding, not "
                "a licence to reroll"
            )
        seen_fingerprint[entry.fingerprint] = entry.base_setup_id
        previous_setup = seen_setup.get(entry.canonical_setup)
        if previous_setup is not None:  # pragma: no cover - implied by the above
            raise SetupLibraryError(
                f"exact duplicate arrangement between {previous_setup} and "
                f"{entry.base_setup_id}"
            )
        seen_setup[entry.canonical_setup] = entry.base_setup_id


def generate_library(
    seed_context: LibrarySeedContext = DEFAULT_SEED_CONTEXT,
    order: "list[tuple[str, int]] | None" = None,
    progress=None,
) -> LibraryGenerationResult:
    """Generate the complete base library and enforce the library-wide gates.

    `order` exists for the enumeration-order regression: generating in a
    shuffled order must produce exactly the same entries, because no base
    conditions on any other. The returned entries are always sorted back into
    the frozen file order regardless of the order they were generated in.
    """
    started = time.time()
    requested = library_order() if order is None else list(order)
    rejections_by_family: dict = {}
    attempt_histogram: Counter = Counter()
    entries: list[BaseSetupEntry] = []

    for position, (family_id, base_index) in enumerate(requested):
        record = generate_base_setup(family_id, base_index, seed_context)
        entries.append(record.entry)
        attempt_histogram[record.entry.generation_attempts] += 1
        if record.rejections:
            family_totals = rejections_by_family.setdefault(family_id, {})
            for reason, count in record.rejections.items():
                family_totals[reason] = family_totals.get(reason, 0) + count
        if progress is not None and (position + 1) % 500 == 0:
            progress(position + 1, len(requested))

    entries.sort(key=lambda entry: (FAMILY_IDS.index(entry.family_id), entry.base_index))
    _enforce_global_uniqueness(entries)

    return LibraryGenerationResult(
        entries=tuple(entries),
        seed_context=seed_context,
        rejections_by_family=rejections_by_family,
        attempt_histogram=dict(sorted(attempt_histogram.items())),
        duration_seconds=round(time.time() - started, 3),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def entry_lines(entries: "list[BaseSetupEntry] | tuple[BaseSetupEntry, ...]") -> "list[str]":
    """The canonical JSONL lines of a library, in the order supplied."""
    return [base_entry_json_line(entry.to_dict()) for entry in entries]


def write_library_jsonl(path, entries) -> int:
    """Write the library JSONL deterministically. Returns the byte count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(entry_lines(entries)) + "\n"
    target.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def read_library_jsonl(path) -> "list[BaseSetupEntry]":
    """Read a materialized library back into entries."""
    text = Path(path).read_text(encoding="utf-8")
    return [
        BaseSetupEntry.from_dict(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]


def library_content_digest(entries) -> str:
    """Agent 1's library identity digest over `base_setup_id:fingerprint`."""
    return library_digest([(entry.base_setup_id, entry.fingerprint) for entry in entries])


def entry_metadata_digest(entries) -> str:
    """SHA-256 over the exact serialized entry lines.

    Stronger than the identity digest: it pins every metadata field — seeds,
    attempt indices, traits, splits — not just the accepted content, so a
    regeneration that changed any recorded provenance is detected even if the
    setups themselves matched.
    """
    payload = f"{ENTRY_METADATA_DIGEST_DOMAIN}\n" + "\n".join(entry_lines(entries))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _counts(entries) -> dict:
    family_counts = {family_id: 0 for family_id in FAMILY_IDS}
    split_counts = {split: 0 for split in SPLITS}
    family_split_counts = {
        family_id: {split: 0 for split in SPLITS} for family_id in FAMILY_IDS
    }
    for entry in entries:
        family_counts[entry.family_id] += 1
        split_counts[entry.split] += 1
        family_split_counts[entry.family_id][entry.split] += 1
    return {
        "family_counts": family_counts,
        "split_counts": split_counts,
        "family_split_counts": family_split_counts,
    }


def manifest_digest(manifest: dict) -> str:
    """SHA-256 over the manifest's identity fields.

    `generation_run` (timestamp, wall time, host measurements) is excluded, so
    the digest is a property of the library, not of the machine that built it:
    two independent regenerations of the same library agree exactly.
    """
    stable = {
        key: value for key, value in manifest.items() if key not in MANIFEST_VOLATILE_KEYS
    }
    payload = MANIFEST_DIGEST_DOMAIN + "\n" + json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    result: LibraryGenerationResult,
    command: str,
    library_bytes: "int | None" = None,
    peak_rss_bytes: "int | None" = None,
    timestamp: "str | None" = None,
) -> dict:
    """The `setup_library_v1` manifest for the materialized library."""
    entries = result.entries
    counts = _counts(entries)
    manifest = {
        "library_version": SETUP_LIBRARY_VERSION,
        "generator_contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "family_version": SETUP_FAMILY_VERSION,
        "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
        "generator_version": GENERATOR_VERSION,
        "master_seed": result.seed_context.master_seed,
        "seed_derivation": result.seed_context.to_dict(),
        "entry_count": len(entries),
        "family_counts": counts["family_counts"],
        "split_counts": counts["split_counts"],
        "family_split_counts": counts["family_split_counts"],
        "library_content_digest": library_content_digest(entries),
        "entry_metadata_digest": entry_metadata_digest(entries),
        "library_jsonl_path": LIBRARY_JSONL_PATH,
        "library_manifest_path": LIBRARY_MANIFEST_PATH,
        "entry_fields": list(BASE_ENTRY_REQUIRED_FIELDS),
        "entry_line_format": 'json.dumps(entry, sort_keys=True, separators=(",", ":"))',
        "file_order": "families F00..F15, base index ascending within each family",
        "split_rule": (
            f"base_index 0..{TRAIN_PER_FAMILY - 1} train, "
            f"{TRAIN_PER_FAMILY}..{TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY - 1} validation, "
            f"{TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY}..{BASES_PER_FAMILY - 1} test"
        ),
        "rejection_counts": {
            "by_reason": result.rejections_by_reason(),
            "by_family": {
                family_id: dict(sorted(reasons.items()))
                for family_id, reasons in sorted(result.rejections_by_family.items())
            },
            "total": sum(result.rejections_by_reason().values()),
        },
        "attempt_statistics": {
            "total_attempts": result.total_attempts(),
            "attempts_per_accepted_base": result.attempts_per_accepted_base(),
            "attempt_histogram": {
                str(attempts): count for attempts, count in result.attempt_histogram.items()
            },
            "rejection_rate_by_family": result.rejection_rate_by_family(),
            "max_attempts_budget": plans_document()["max_attempts_per_base"],
        },
        "generation_run": {
            "command": command,
            "duration_seconds": result.duration_seconds,
            "timestamp": timestamp,
            "library_bytes": library_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        },
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return manifest


def write_manifest(path, manifest: dict) -> int:
    """Write the manifest as sorted, indented JSON. Returns the byte count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    target.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def read_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Generation-time verification
# ---------------------------------------------------------------------------


def _trait_distributions(entries) -> dict:
    """Basic per-family trait distributions, recomputed from content."""
    per_family: dict = {}
    for family_id in FAMILY_IDS:
        members = [entry for entry in entries if entry.family_id == family_id]
        if not members:
            continue
        vectors = [compute_trait_vector(entry.canonical_setup) for entry in members]
        summary: dict = {
            "member_count": len(members),
            "flag_rank_histogram": [
                sum(1 for vector in vectors if vector["flag_rank"] == rank) for rank in range(4)
            ],
            "flag_edge_distance_histogram": [
                sum(1 for vector in vectors if vector["flag_edge_distance"] == distance)
                for distance in range(5)
            ],
        }
        for trait in (
            "flag_orth_bomb_guards",
            "bomb_front2_count",
            "bomb_distinct_files",
            "scout_front2_count",
            "scout_front_rank_count",
            "miner_front2_count",
            "high_front2_count",
            "movable_front_rank_count",
            "unconventional_feature_count",
        ):
            values = [vector[trait] for vector in vectors]
            summary[trait] = {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 6),
            }
        per_family[family_id] = summary
    return per_family


def verify_library(entries) -> dict:
    """Recompute every generation-time correctness check from stored content.

    Deliberately independent of the generator's own counters: content is
    re-validated through the frozen engine, re-evaluated against the frozen
    family contracts, re-canonicalized, re-fingerprinted, and its identity and
    split re-derived from the base index. Agent 3 owns the independent audit
    verdict; this exists so Agent 2 cannot knowingly hand over a broken
    library.
    """
    counts = _counts(entries)

    exact_groups: dict = {}
    class_groups: dict = {}
    content_groups: dict = {}
    identifier_groups: dict = {}
    engine_invalid: list[str] = []
    stranded: list[str] = []
    family_violations: list[dict] = []
    non_canonical: list[str] = []
    reflection_failures: list[str] = []
    identity_mismatches: list[dict] = []
    metadata_mismatches: list[dict] = []

    for entry in entries:
        setup = entry.canonical_setup
        exact_groups.setdefault(setup, []).append(entry.base_setup_id)
        recomputed_class = class_fingerprint(setup)
        class_groups.setdefault(recomputed_class, []).append(entry.base_setup_id)
        content_groups.setdefault(content_fingerprint(setup), []).append(entry.base_setup_id)
        identifier_groups.setdefault(entry.base_setup_id, []).append(entry.base_setup_id)

        try:
            validate_setup(setup, 0)
        except SetupError as error:
            engine_invalid.append(f"{entry.base_setup_id}: {error}")

        traits = compute_trait_vector(setup)
        satisfied, violations = family_contract(entry.family_id).evaluate(traits)
        if not satisfied:
            family_violations.append(
                {"base_setup_id": entry.base_setup_id, "violations": violations}
            )
        if not setup_has_initial_mobility(setup):
            stranded.append(entry.base_setup_id)
        if not is_canonical_representative(setup):
            non_canonical.append(entry.base_setup_id)
        if (
            canonical_class_representative(setup) != setup
            or canonical_class_representative(reflect_canonical(setup)) != setup
        ):
            reflection_failures.append(entry.base_setup_id)

        expected_identifier = base_setup_id(entry.family_id, entry.base_index)
        expected_split = split_for_base_index(entry.base_index)
        if entry.base_setup_id != expected_identifier or entry.split != expected_split:
            identity_mismatches.append(
                {
                    "base_setup_id": entry.base_setup_id,
                    "expected_id": expected_identifier,
                    "split": entry.split,
                    "expected_split": expected_split,
                }
            )
        if (
            entry.fingerprint != recomputed_class
            or entry.content_fingerprint != content_fingerprint(setup)
            or entry.reflected_content_fingerprint
            != content_fingerprint(reflect_canonical(setup))
            or entry.trait_vector != traits
            or entry.family_key != FAMILY_BY_ID[entry.family_id].key
            or entry.library_version != SETUP_LIBRARY_VERSION
            or entry.contract_version != SETUP_GENERATOR_CONTRACT_VERSION
            or entry.family_contract_version != SETUP_FAMILY_VERSION
            or entry.trait_schema_version != SETUP_TRAIT_VECTOR_VERSION
        ):
            metadata_mismatches.append({"base_setup_id": entry.base_setup_id})

    serialized_field_names = sorted(
        set().union(*(set(entry.to_dict()) | set(entry.trait_vector) for entry in entries))
        if entries
        else set()
    )
    forbidden_fields = [
        name
        for name in serialized_field_names
        if any(token in name.lower() for token in FORBIDDEN_ENTRY_FIELD_TOKENS)
    ]

    checks = {
        "entry_count_exact": len(entries) == BASE_SETUP_COUNT,
        "family_counts_exact": all(
            count == BASES_PER_FAMILY for count in counts["family_counts"].values()
        ),
        "family_split_counts_exact": all(
            row == {"train": TRAIN_PER_FAMILY, "validation": VALIDATION_PER_FAMILY, "test": TEST_PER_FAMILY}
            for row in counts["family_split_counts"].values()
        ),
        "no_exact_duplicates": all(len(group) == 1 for group in exact_groups.values()),
        "no_reflection_duplicates": all(len(group) == 1 for group in class_groups.values()),
        "no_content_fingerprint_collisions": all(
            len(group) == 1 for group in content_groups.values()
        ),
        "no_stable_id_collisions": all(
            len(group) == 1 for group in identifier_groups.values()
        ),
        "no_engine_invalid": not engine_invalid,
        "no_stranded": not stranded,
        "no_family_violations": not family_violations,
        "all_entries_canonical": not non_canonical,
        "reflection_roundtrip_clean": not reflection_failures,
        "identity_and_split_rule_exact": not identity_mismatches,
        "entry_metadata_consistent": not metadata_mismatches,
        "no_outcome_or_strength_fields": not forbidden_fields,
    }

    return {
        "entry_count": len(entries),
        **counts,
        "exact_duplicate_groups": sum(1 for group in exact_groups.values() if len(group) > 1),
        "reflection_duplicate_groups": sum(
            1 for group in class_groups.values() if len(group) > 1
        ),
        "content_fingerprint_collisions": sum(
            1 for group in content_groups.values() if len(group) > 1
        ),
        "stable_id_collisions": sum(
            1 for group in identifier_groups.values() if len(group) > 1
        ),
        "engine_invalid": engine_invalid,
        "stranded": stranded,
        "family_violations": family_violations,
        "non_canonical_entries": non_canonical,
        "reflection_roundtrip_failures": reflection_failures,
        "identity_mismatches": identity_mismatches,
        "metadata_mismatches": metadata_mismatches,
        "forbidden_entry_fields": forbidden_fields,
        "trait_distributions": _trait_distributions(entries),
        "checks": checks,
        "all_pass": all(checks.values()),
    }
