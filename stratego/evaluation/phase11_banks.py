"""Phase 11 Agent 1: the frozen paired belief-validation evaluation banks.

Specification sources:

- `01_AGENT_1_CONTRACTS_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze banks", "Freeze
  observer/opponent semantics", "Access ledger")
- `00_PHASE_11_SEQUENCE_AND_COMMON_CONTRACT.md` ("Phase 11 banks")

What a Phase 11 case is
-----------------------
A Phase 11 case fixes *everything* about its two games before any game is
played, because both setup sources are already frozen production artifacts:
the observer's accepted P10-D draw and the opponent's stratum draw (P10-D or
`neutral_v1`) are deterministic functions of the frozen seeds. A case
therefore materializes:

```text
game 0   observer Red  (P10-D draw)   vs   opponent Blue (source draw)
game 1   observer Blue (P10-D draw)   vs   opponent Red  (source draw)
per game one frozen match seed        rule opponents' randomness root
```

Each seat of each game draws from its source *conditioned on its own
colour* — mirroring one arrangement across colours would distort the P10-D
colour-conditional distribution the belief system must be measured under.
For the same reason there is **no rejection of any kind**: Phase 11 selects
nothing, so draws are pure first-attempt draws from the frozen sources, and
every arrangement is exactly what production would have produced (an
Agent 1 reading, recorded in the bank contract).

```text
phase11_validation_bank_v1   512 cases    validation split   8 strata x 2 sources x 32
phase11_test_bank_v1       2,048 cases    test split         8 strata x 2 sources x 128
```

Case identity is (stratum, source)-cell-major over the frozen stratum and
source orders, so balance over strata, sources and colours is a property of
the id space rather than of any draw.

Construction plays no game, runs no neural inference and reads no outcome.
Building and structurally auditing the sealed test bank here is the
`structural_build`/`structural_audit` purpose the sealing rules allow, and
every touch is written to the append-only Phase 11 bank access ledger.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..setups.contracts import parse_base_setup_id, split_for_base_index
from ..setups.identity import content_fingerprint
from ..setups.sampler import load_library_index, rebuild_from_provenance
from ..training.phase10_selector import (
    LearnedSetupSource,
    SelectorRequest,
    candidate,
    load_scorer,
    neutral_baseline_draw,
)
from ..training.phase11_contract import (
    ACCEPTED_SELECTOR_CANDIDATE_ID,
    ACCEPTED_SELECTOR_CONFIG_SHA256,
    ACCEPTED_TRAIT_SCALER_DIGEST,
    ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
    BANK_SPLITS,
    LEDGER_ENTRY_FIELDS,
    LEDGER_RELATIVE_PATH,
    LEDGER_VERSION,
    Phase11ContractError,
    TEST_BANK_CASES,
    TEST_BANK_VERSION,
    TEST_CASES_PER_CELL,
    VALIDATION_BANK_CASES,
    VALIDATION_BANK_VERSION,
    VALIDATION_CASES_PER_CELL,
)
from ..training.phase11_seed import (
    CASE_GAME_INDICES,
    CASE_GAME_OBSERVER_COLOR,
    CASE_GAME_OPPONENT_COLOR,
    OPPONENT_STRATA,
    PHASE11_MASTER_SEED,
    ROLE_OBSERVER,
    ROLE_OPPONENT,
    SETUP_SOURCES,
    SOURCE_NEUTRAL,
    SOURCE_P10D,
    case_setup_seed,
    game_match_seed,
    parse_phase11_case_id,
    phase11_case_id,
    phase11_game_id,
)

#: The two Phase 11 banks, keyed by their short name.
BANK_SPECIFICATIONS = {
    "validation": {
        "bank_version": VALIDATION_BANK_VERSION,
        "split": BANK_SPLITS["validation"],
        "cases_per_cell": VALIDATION_CASES_PER_CELL,
        "case_count": VALIDATION_BANK_CASES,
        "access_justification": (
            "Phase 11 validation bank: frozen belief-validation cases "
            "(Agent 1 structural construction)"
        ),
    },
    "test": {
        "bank_version": TEST_BANK_VERSION,
        "split": BANK_SPLITS["test"],
        "cases_per_cell": TEST_CASES_PER_CELL,
        "case_count": TEST_BANK_CASES,
        "access_justification": (
            "Phase 11 sealed final-test bank: frozen final-evaluation cases "
            "(Agent 1 structural construction; structural access only before "
            "Agent 7)"
        ),
    },
}

MANIFEST_VOLATILE_KEYS = ("construction_run", "manifest_digest")
MANIFEST_DIGEST_DOMAIN = "stratego_phase11_bank_manifest_v1"
BANK_DIGEST_DOMAIN = "stratego_phase11_bank_v1"


class Phase11BankError(Phase11ContractError):
    """A Phase 11 evaluation bank failed construction or audit."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bank_specification(bank: str) -> dict:
    try:
        return dict(BANK_SPECIFICATIONS[bank])
    except KeyError:
        raise Phase11BankError(
            f"unknown Phase 11 bank {bank!r}; expected one of "
            f"{sorted(BANK_SPECIFICATIONS)}"
        ) from None


# ---------------------------------------------------------------------------
# The frozen setup sources
# ---------------------------------------------------------------------------


class Phase11SetupSources:
    """The two frozen setup sources a bank build draws from.

    Construction verifies the accepted P10-D utility artifact from live
    bytes before a single draw exists: a moved coefficient, scaler or
    config file is BLOCKED, never silently consumed.
    """

    def __init__(self, index=None) -> None:
        self.index = load_library_index() if index is None else index
        problems = self.verify_selector_artifacts()
        if problems:
            raise Phase11BankError(
                "the accepted P10-D selector artifacts moved (BLOCKED): "
                + "; ".join(problems)
            )
        self.scorer = load_scorer()
        self.learned = LearnedSetupSource(
            candidate(ACCEPTED_SELECTOR_CANDIDATE_ID), self.scorer, self.index
        )

    @staticmethod
    def verify_selector_artifacts(root: "Path | None" = None) -> "list[str]":
        """Live-byte verification of the frozen selector identity chain."""
        base = repository_root() if root is None else Path(root)
        problems: list[str] = []
        config_path = base / "reports" / "phase_10_data" / "agent_05_frozen_selector_config.json"
        utility_path = base / "checkpoints" / "phase10" / "setup_utility_v1.json"
        for path in (config_path, utility_path):
            if not path.exists():
                problems.append(f"{path.name} is missing")
        if problems:
            return problems
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_sha != ACCEPTED_SELECTOR_CONFIG_SHA256:
            problems.append(
                f"selector config SHA {config_sha} != accepted "
                f"{ACCEPTED_SELECTOR_CONFIG_SHA256}"
            )
        artifact = json.loads(utility_path.read_text())
        observed_coefficient = artifact["models"]["model_T"]["coefficient_digest"]
        if observed_coefficient != ACCEPTED_UTILITY_COEFFICIENT_DIGEST:
            problems.append(
                f"model_T coefficient digest {observed_coefficient} != accepted"
            )
        if artifact["scaler_digest"] != ACCEPTED_TRAIT_SCALER_DIGEST:
            problems.append(
                f"trait scaler digest {artifact['scaler_digest']} != accepted"
            )
        return problems

    def draw(self, source: str, split: str, color: str, seed: int) -> dict:
        """One frozen setup-draw record for one seat.

        `source` is either the P10-D token or the neutral token from the
        frozen source enumeration. The record stores the complete draw
        identity, the canonical arrangement and the accepted sampler
        provenance, so an auditor can re-derive the draw or rebuild the
        arrangement from provenance alone.
        """
        if source == SOURCE_P10D:
            draw = self.learned.draw(
                SelectorRequest(split=split, color=color, selector_seed=int(seed))
            )
            sampled = draw.setup
            record = {
                "source": SOURCE_P10D,
                "selector_identity": draw.selector_identity,
                "candidate_id": draw.candidate_id,
                "branch": draw.branch,
                "branch_uniform": draw.branch_uniform,
                "base_uniform": draw.base_uniform,
            }
        elif source == SOURCE_NEUTRAL:
            sampled = neutral_baseline_draw(split, int(seed), self.index)
            record = {"source": SOURCE_NEUTRAL}
        else:
            raise Phase11BankError(
                f"unknown setup source {source!r}; expected one of {list(SETUP_SOURCES)}"
            )
        record.update(
            {
                "split": split,
                "color": color,
                "setup_seed": int(seed),
                "base_setup_id": sampled.provenance["base_setup_id"],
                "family_id": sampled.provenance["primary_family_id"],
                "final_setup_fingerprint": sampled.provenance["final_setup_fingerprint"],
                "setup": list(sampled.canonical),
                "provenance": dict(sampled.provenance),
            }
        )
        return record


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase11Case:
    """One frozen Phase 11 paired belief-validation case."""

    case_id: str
    bank: str
    bank_version: str
    split: str
    stratum: str
    setup_source: str
    case_ordinal: int
    case_index: int
    #: Per game index: colours, match seed and both frozen setup draws.
    games: dict

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "bank": self.bank,
            "bank_version": self.bank_version,
            "split": self.split,
            "stratum": self.stratum,
            "setup_source": self.setup_source,
            "case_ordinal": self.case_ordinal,
            "case_index": self.case_index,
            "bootstrap_unit": self.case_id,
            "games": {
                str(index): dict(self.games[index]) for index in CASE_GAME_INDICES
            },
        }


def case_cell(case_index: int, cases_per_cell: int) -> "tuple[str, str, int]":
    """`(stratum, source, ordinal)` of one case index — cell-major, frozen."""
    if not isinstance(case_index, int) or isinstance(case_index, bool) or case_index < 0:
        raise Phase11BankError(f"case_index must be a non-negative int, got {case_index!r}")
    cell_index, ordinal = divmod(int(case_index), int(cases_per_cell))
    stratum_index, source_index = divmod(cell_index, len(SETUP_SOURCES))
    if stratum_index >= len(OPPONENT_STRATA):
        raise Phase11BankError(
            f"case_index {case_index} is outside the "
            f"{len(OPPONENT_STRATA) * len(SETUP_SOURCES) * cases_per_cell}-case id space"
        )
    return OPPONENT_STRATA[stratum_index], SETUP_SOURCES[source_index], ordinal


def build_case(bank: str, case_index: int, sources: "Phase11SetupSources | None" = None) -> Phase11Case:
    """One bank case, built from its identity and the frozen sources alone."""
    specification = bank_specification(bank)
    stratum, source, ordinal = case_cell(case_index, specification["cases_per_cell"])
    case_id = phase11_case_id(specification["bank_version"], stratum, source, ordinal)
    split = specification["split"]
    if sources is None:
        sources = Phase11SetupSources()

    games: dict = {}
    for game_index in CASE_GAME_INDICES:
        observer_color = CASE_GAME_OBSERVER_COLOR[game_index]
        opponent_color = CASE_GAME_OPPONENT_COLOR[game_index]
        game_id = phase11_game_id(case_id, game_index)
        observer_record = sources.draw(
            SOURCE_P10D,
            split,
            observer_color,
            case_setup_seed(case_id, game_index, ROLE_OBSERVER),
        )
        opponent_record = sources.draw(
            source,
            split,
            opponent_color,
            case_setup_seed(case_id, game_index, ROLE_OPPONENT),
        )
        games[game_index] = {
            "game_id": game_id,
            "game_index": game_index,
            "observer_color": observer_color,
            "opponent_color": opponent_color,
            "match_seed": game_match_seed(game_id),
            "observer": observer_record,
            "opponent": opponent_record,
        }

    return Phase11Case(
        case_id=case_id,
        bank=bank,
        bank_version=specification["bank_version"],
        split=split,
        stratum=stratum,
        setup_source=source,
        case_ordinal=ordinal,
        case_index=int(case_index),
        games=games,
    )


def bank_digest(cases: "tuple[Phase11Case, ...]") -> str:
    """SHA-256 over the bank's complete case content — its stable identity."""
    payload = {
        "domain": BANK_DIGEST_DOMAIN,
        "master_seed": PHASE11_MASTER_SEED,
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


def build_phase11_bank(
    bank: str, sources: "Phase11SetupSources | None" = None
) -> "tuple[tuple[Phase11Case, ...], dict]":
    """The complete frozen bank and its manifest.

    Reproducible from frozen constants and the accepted upstream artifacts
    alone; two builds yield identical bytes and identical digests, which
    the audit re-proves by rebuilding a deterministic sample of cases in
    isolation.
    """
    specification = bank_specification(bank)
    if sources is None:
        sources = Phase11SetupSources()
    started = time.time()

    cases = tuple(
        build_case(bank, case_index, sources)
        for case_index in range(specification["case_count"])
    )

    branch_histogram: dict = {}
    for case in cases:
        for game in case.games.values():
            for role in (ROLE_OBSERVER, ROLE_OPPONENT):
                branch = game[role].get("branch")
                if branch is not None:
                    branch_histogram[branch] = branch_histogram.get(branch, 0) + 1

    manifest = {
        "bank": bank,
        "bank_version": specification["bank_version"],
        "split": specification["split"],
        "case_count": specification["case_count"],
        "game_count": 2 * specification["case_count"],
        "cases_per_cell": specification["cases_per_cell"],
        "cases_per_stratum": specification["cases_per_cell"] * len(SETUP_SOURCES),
        "stratum_order": list(OPPONENT_STRATA),
        "setup_sources": list(SETUP_SOURCES),
        "case_id_rule": (
            "case_index = ((stratum_index * 2) + source_index) * cases_per_cell "
            "+ ordinal; case_id = '<bank_version>|ms=<master>|st=<stratum>"
            "|src=<source>|c=<ordinal:03d>'"
        ),
        "colour_pairing": (
            "the observer plays Red in game 0 and Blue in game 1 against the "
            "same opponent stratum"
        ),
        "bootstrap_unit": "the logical case, both colour games together",
        "observer_setup_source": (
            "learned_setup_source_v1 P10-D, constant across both banks"
        ),
        "setup_draw_rule": (
            "each seat of each game draws from its frozen source conditioned "
            "on its colour under case_setup_seed(case_id, game_index, role); "
            "pure first-attempt draws, no rejection of any kind"
        ),
        "match_seed_rule": "game_match_seed(game_id)",
        "selector_config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
        "utility_coefficient_digest": ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
        "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
        "library_content_digest": sources.index.content_digest,
        "p10d_branch_histogram": {
            branch: count for branch, count in sorted(branch_histogram.items())
        },
        "access_justification": specification["access_justification"],
        "no_outcome_selection": (
            "construction plays no game, runs no neural inference and reads "
            "no strength signal; there is no rejection, so no draw was "
            "conditioned on anything but its frozen identity"
        ),
        "bank_digest": bank_digest(cases),
        "construction_run": {"duration_seconds": round(time.time() - started, 3)},
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return cases, manifest


# ---------------------------------------------------------------------------
# Structural audit
# ---------------------------------------------------------------------------


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


def audit_phase11_bank(
    bank: str,
    cases: "tuple[Phase11Case, ...]",
    manifest: dict,
    sources: "Phase11SetupSources | None" = None,
    *,
    rebuild_sample_every: int = 16,
) -> dict:
    """Recompute every structural bank property from stored content.

    Structural only: no game is played, no neural model is loaded, no
    outcome is read. `rebuild_sample_every` controls the isolated-rebuild
    spot check; 1 rebuilds every case.
    """
    specification = bank_specification(bank)
    if sources is None:
        sources = Phase11SetupSources()
    library = sources.index

    failures: list = []
    cell_counts: dict = {}
    stratum_counts: dict = {stratum: 0 for stratum in OPPONENT_STRATA}
    source_counts: dict = {source: 0 for source in SETUP_SOURCES}
    split_violations: list = []
    provenance_mismatches: list = []
    draw_mismatches: list = []
    rebuild_mismatches: list = []
    engine_failures: list = []
    seed_reuse: list = []
    observer_source_violations: list = []
    color_pairing_violations: list = []

    all_seeds: dict = {}
    fingerprint_counts: dict = {}

    for case in cases:
        expected_stratum, expected_source, expected_ordinal = case_cell(
            case.case_index, specification["cases_per_cell"]
        )
        stratum_counts[expected_stratum] += 1
        source_counts[expected_source] += 1
        cell = (expected_stratum, expected_source)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        if (case.stratum, case.setup_source, case.case_ordinal) != (
            expected_stratum,
            expected_source,
            expected_ordinal,
        ):
            failures.append(f"{case.case_id}: cell disagrees with the id rule")
        parsed = parse_phase11_case_id(case.case_id)
        if parsed["bank_version"] != specification["bank_version"]:
            failures.append(f"{case.case_id}: bank version {parsed['bank_version']!r}")
        if (
            parsed["stratum"] != case.stratum
            or parsed["setup_source"] != case.setup_source
            or parsed["case_ordinal"] != case.case_ordinal
        ):
            failures.append(f"{case.case_id}: stored fields disagree with the id")

        for game_index in CASE_GAME_INDICES:
            game = case.games[game_index]
            expected_observer = CASE_GAME_OBSERVER_COLOR[game_index]
            expected_opponent = CASE_GAME_OPPONENT_COLOR[game_index]
            if (
                game["observer_color"] != expected_observer
                or game["opponent_color"] != expected_opponent
            ):
                color_pairing_violations.append(f"{case.case_id} game {game_index}")
            if game["game_id"] != phase11_game_id(case.case_id, game_index):
                failures.append(f"{case.case_id} game {game_index}: game id")
            expected_match_seed = game_match_seed(game["game_id"])
            if game["match_seed"] != expected_match_seed:
                failures.append(f"{case.case_id} game {game_index}: match seed")
            match_key = ("match", game["match_seed"])
            if match_key in all_seeds:
                seed_reuse.append(f"{case.case_id} game {game_index} match seed reused")
            all_seeds[match_key] = case.case_id

            for role, expected_color, expected_draw_source in (
                (ROLE_OBSERVER, expected_observer, SOURCE_P10D),
                (ROLE_OPPONENT, expected_opponent, case.setup_source),
            ):
                record = game[role]
                if record["source"] != expected_draw_source:
                    if role == ROLE_OBSERVER:
                        observer_source_violations.append(
                            f"{case.case_id} game {game_index}"
                        )
                    else:
                        failures.append(
                            f"{case.case_id} game {game_index}: opponent source "
                            f"{record['source']!r}"
                        )
                if record["color"] != expected_color:
                    color_pairing_violations.append(
                        f"{case.case_id} game {game_index} {role}"
                    )
                expected_seed = case_setup_seed(case.case_id, game_index, role)
                if record["setup_seed"] != expected_seed:
                    failures.append(
                        f"{case.case_id} game {game_index} {role}: setup seed"
                    )
                seed_key = (f"setup_{role}", record["setup_seed"])
                if seed_key in all_seeds:
                    seed_reuse.append(
                        f"{case.case_id} game {game_index} {role} setup seed reused"
                    )
                all_seeds[seed_key] = case.case_id

                if record["split"] != specification["split"]:
                    split_violations.append(
                        f"{case.case_id} game {game_index} {role}: split "
                        f"{record['split']!r}"
                    )
                _, _, base_index = parse_base_setup_id(record["base_setup_id"])
                if split_for_base_index(base_index) != specification["split"]:
                    split_violations.append(
                        f"{case.case_id} game {game_index} {role}: base index "
                        f"{base_index} is not a {specification['split']!r} base"
                    )

                fingerprint = record["final_setup_fingerprint"]
                fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1
                try:
                    rebuilt = rebuild_from_provenance(record["provenance"], index=library)
                except Exception as error:  # noqa: BLE001 - a failed rebuild is a finding
                    provenance_mismatches.append(
                        f"{case.case_id} game {game_index} {role}: rebuild failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                if tuple(rebuilt.canonical) != tuple(record["setup"]):
                    provenance_mismatches.append(
                        f"{case.case_id} game {game_index} {role}: stored setup "
                        "differs from its provenance rebuild"
                    )
                if content_fingerprint(tuple(record["setup"])) != fingerprint:
                    provenance_mismatches.append(
                        f"{case.case_id} game {game_index} {role}: fingerprint "
                        "does not match the stored setup"
                    )
                engine_failures.extend(
                    f"{case.case_id} game {game_index} {role}: {failure}"
                    for failure in _validate_case_setup(tuple(record["setup"]))
                )
                redraw = sources.draw(
                    record["source"], record["split"], record["color"], record["setup_seed"]
                )
                if redraw != record:
                    draw_mismatches.append(
                        f"{case.case_id} game {game_index} {role}: independent "
                        "re-draw differs from the stored record"
                    )

        if case.case_index % max(1, int(rebuild_sample_every)) == 0:
            rebuilt_case = build_case(bank, case.case_index, sources)
            if rebuilt_case != case:
                rebuild_mismatches.append(f"{case.case_id}: isolated rebuild differs")

    observed_digest = bank_digest(cases)
    repeated = {
        fingerprint: count for fingerprint, count in fingerprint_counts.items() if count > 1
    }

    checks = {
        "case_count_exact": len(cases) == specification["case_count"],
        "case_indices_contiguous": tuple(case.case_index for case in cases)
        == tuple(range(specification["case_count"])),
        "stratum_balance_exact": all(
            count == specification["cases_per_cell"] * len(SETUP_SOURCES)
            for count in stratum_counts.values()
        ),
        "source_balance_exact": all(
            count == specification["cases_per_cell"] * len(OPPONENT_STRATA)
            for count in source_counts.values()
        ),
        "cell_balance_exact": len(cell_counts) == len(OPPONENT_STRATA) * len(SETUP_SOURCES)
        and all(count == specification["cases_per_cell"] for count in cell_counts.values()),
        "colour_pairing_exact": not color_pairing_violations,
        "observer_source_constant": not observer_source_violations,
        "split_isolation": not split_violations,
        "engine_valid": not engine_failures,
        "provenance_rebuilds": not provenance_mismatches,
        "independent_redraw_exact": not draw_mismatches,
        "isolated_rebuild_exact": not rebuild_mismatches,
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
        "game_count": 2 * len(cases),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "cell_counts": {
            f"{stratum}/{source}": count
            for (stratum, source), count in sorted(cell_counts.items())
        },
        "bank_digest": observed_digest,
        "setup_draws_total": 4 * len(cases),
        "distinct_fingerprints": len(fingerprint_counts),
        "repeated_fingerprints": len(repeated),
        "repeated_fingerprint_note": (
            "a repeat across cases is a property of the frozen sources' "
            "support, not a failure: Phase 11 rejects nothing by design"
        ),
        "failures": failures,
        "engine_failures": engine_failures,
        "split_violations": split_violations,
        "provenance_mismatches": provenance_mismatches,
        "draw_mismatches": draw_mismatches,
        "rebuild_mismatches": rebuild_mismatches,
        "seed_reuse": seed_reuse,
        "observer_source_violations": observer_source_violations,
        "color_pairing_violations": color_pairing_violations,
        "rebuild_sample_every": int(rebuild_sample_every),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def cross_bank_checks(
    validation_cases: "tuple[Phase11Case, ...]",
    test_cases: "tuple[Phase11Case, ...]",
) -> dict:
    """Logical-id, seed and fingerprint disjointness between the two banks.

    Id and seed disjointness is required (the banks' version tokens make it
    structural; this proves it). Fingerprint overlap is also required to be
    zero here: the banks draw from disjoint Phase 7 splits, so an overlap
    could only mean two different bases producing one perturbed arrangement
    — worth checking rather than asserting.
    """

    def _ids(cases):
        values = set()
        for case in cases:
            values.add(case.case_id)
            for game in case.games.values():
                values.add(game["game_id"])
        return values

    def _seeds(cases):
        values = []
        for case in cases:
            for game in case.games.values():
                values.append(game["match_seed"])
                values.append(game[ROLE_OBSERVER]["setup_seed"])
                values.append(game[ROLE_OPPONENT]["setup_seed"])
        return values

    def _fingerprints(cases):
        return {
            game[role]["final_setup_fingerprint"]
            for case in cases
            for game in case.games.values()
            for role in (ROLE_OBSERVER, ROLE_OPPONENT)
        }

    validation_ids, test_ids = _ids(validation_cases), _ids(test_cases)
    validation_seeds, test_seeds = _seeds(validation_cases), _seeds(test_cases)
    validation_fps, test_fps = _fingerprints(validation_cases), _fingerprints(test_cases)
    id_overlap = sorted(validation_ids & test_ids)
    seed_overlap = sorted(set(validation_seeds) & set(test_seeds))
    fingerprint_overlap = sorted(validation_fps & test_fps)
    return {
        "validation_ids": len(validation_ids),
        "test_ids": len(test_ids),
        "id_overlap": id_overlap,
        "seed_overlap": seed_overlap,
        "validation_fingerprints": len(validation_fps),
        "test_fingerprints": len(test_fps),
        "fingerprint_overlap": fingerprint_overlap,
        "zero_overlap": not (id_overlap or seed_overlap or fingerprint_overlap),
    }


# ---------------------------------------------------------------------------
# The append-only bank access ledger
# ---------------------------------------------------------------------------


def ledger_path(root: "Path | None" = None) -> Path:
    base = repository_root() if root is None else Path(root)
    return base / LEDGER_RELATIVE_PATH


def ledger_entry(
    agent: int,
    stage: str,
    bank_version: str,
    purpose: str,
    *,
    structural_only: bool,
    neural_inference_count: int = 0,
    scored_prediction_count: int = 0,
    privileged_truth_count: int = 0,
    outcome_count: int = 0,
) -> dict:
    """One well-formed ledger entry, in the frozen field order."""
    entry = {
        "ledger_version": LEDGER_VERSION,
        "agent": int(agent),
        "stage": str(stage),
        "bank_version": str(bank_version),
        "purpose": str(purpose),
        "structural_only": bool(structural_only),
        "neural_inference_count": int(neural_inference_count),
        "scored_prediction_count": int(scored_prediction_count),
        "privileged_truth_count": int(privileged_truth_count),
        "outcome_count": int(outcome_count),
    }
    if tuple(entry) != LEDGER_ENTRY_FIELDS:
        raise Phase11BankError("ledger entry fields drifted from the frozen schema")
    return entry


def append_ledger_entries(entries: "list[dict]", root: "Path | None" = None) -> Path:
    """Append entries to the ledger file. Append-only, never rewritten."""
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        for entry in entries:
            if tuple(entry) != LEDGER_ENTRY_FIELDS:
                raise Phase11BankError(
                    "refusing to append a ledger entry with drifted fields"
                )
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def read_ledger(root: "Path | None" = None) -> "list[dict]":
    """Every ledger entry, in append order."""
    path = ledger_path(root)
    if not path.exists():
        return []
    entries = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("ledger_version") != LEDGER_VERSION:
            raise Phase11BankError(
                f"ledger line {line_number} carries version "
                f"{entry.get('ledger_version')!r}, expected {LEDGER_VERSION!r}"
            )
        entries.append(entry)
    return entries


def verify_test_bank_sealed(entries: "list[dict] | None" = None, root: "Path | None" = None) -> dict:
    """The pre-Agent-7 sealing proof over the ledger.

    Every test-bank entry must be structural-only with all four counters
    zero. Returns the summary; raises nothing — callers gate on the
    booleans so a violation is reported, not swallowed.
    """
    entries = read_ledger(root) if entries is None else entries
    test_entries = [
        entry for entry in entries if entry["bank_version"] == TEST_BANK_VERSION
    ]
    violations = [
        entry
        for entry in test_entries
        if not entry["structural_only"]
        or entry["neural_inference_count"] != 0
        or entry["scored_prediction_count"] != 0
        or entry["privileged_truth_count"] != 0
        or entry["outcome_count"] != 0
    ]
    return {
        "ledger_entries": len(entries),
        "test_bank_entries": len(test_entries),
        "violations": violations,
        "test_bank_structural_only": not violations,
        "scored_prediction_total": sum(
            entry["scored_prediction_count"] for entry in test_entries
        ),
        "privileged_truth_total": sum(
            entry["privileged_truth_count"] for entry in test_entries
        ),
        "outcome_total": sum(entry["outcome_count"] for entry in test_entries),
        "neural_inference_total": sum(
            entry["neural_inference_count"] for entry in test_entries
        ),
    }


__all__ = [
    "BANK_DIGEST_DOMAIN",
    "BANK_SPECIFICATIONS",
    "MANIFEST_DIGEST_DOMAIN",
    "MANIFEST_VOLATILE_KEYS",
    "Phase11BankError",
    "Phase11Case",
    "Phase11SetupSources",
    "append_ledger_entries",
    "audit_phase11_bank",
    "bank_digest",
    "bank_specification",
    "build_case",
    "build_phase11_bank",
    "case_cell",
    "cross_bank_checks",
    "ledger_entry",
    "ledger_path",
    "manifest_digest",
    "read_ledger",
    "repository_root",
    "verify_test_bank_sealed",
]
