"""Phase 10 Agent 6: the integration soak of the frozen production system.

Specification sources:

- `06_AGENT_6_INTEGRATION_SOAK_AND_PRODUCTION_FREEZE.md` ("Integration soak",
  "Per-game integration checks", "Parallelism/restart", "Storage/throughput",
  "Actual-game diversity diagnostics", "Outcome diagnostics")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Selector semantics",
  "Storage/path semantics", "Phase 10 root seeds")

What this module is
-------------------
The machinery that plays **actual games** through the frozen production
system — the accepted Phase 9 move policy on both sides and the selected
`learned_setup_source_v1` configuration (P10-D) supplying both initial
setups — at operational scale, under parallel workers, process restarts and
crash recovery, and then audits every committed game against its own logical
identity. It selects nothing, fits nothing, changes nothing: Agent 5's
selection is closed, every soak outcome is report-only, and the only frozen
decision this module *executes* is "play the system as selected and prove it
is operationally safe and reproducible".

The soak namespace
------------------
Soak games live in their own identity namespace, disjoint by construction
from every other Phase 10 stream:

```text
game id     phase10_soak_v1|ms=2026081801|g=<ordinal:05d>
selector    soak_selector stream, rooted at the frozen selector-draw root
match       soak_match stream, rooted at the frozen Phase 10 master root
```

Seeds are derived exactly as the frozen Agent 1 derivation does —
``blake2b(person='strat-s10', digest_size=8)`` over colon-joined identity
text, big-endian, right-shifted one bit — but under the payload prefix
``phase10_soak_v1`` instead of ``phase10_identity_v1``. Two payloads with
different first tokens are never equal, so no soak stream can collide with
any frozen Agent 1 stream; :func:`soak_seed_collision_audit` additionally
proves disjointness empirically over the materialized Phase 10 id space
rather than leaving it as an argument. No derivation reads worker count,
arrival order, process id, wall clock or a storage path, so a soak game is
addressable by id, resume is exact set subtraction by game id, and
re-sharding across workers or restarts cannot move a single draw.

The store
---------
One append-only journal file per (segment, worker), holding one canonical
JSON line per committed game:

```text
<root>/journal/seg0000_w00.soak.jsonl
```

The commit rule is the accepted Phase 8/10 sentence — *a game becomes
visible only when its commit line exists* — collapsed to its one-file form:
a line is a commit exactly when it is newline-terminated, parses, carries
every envelope field, and its payload digest matches its record. A process
killed mid-write leaves a torn tail that :func:`reconcile_soak` truncates on
the next open, so the store only ever contains committed games. A sealed
soak store is immutable, and its content digest is computed over the
committed payload digests in canonical game-id order — path-independent by
construction, exactly as the corpus seal is.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .phase10_contract import (
    ACCEPTED_PHASE9_CHECKPOINT_SHA256,
    ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    LEARNED_SETUP_SOURCE_VERSION,
    NEUTRAL_MIXTURE_WEIGHT,
)
from .phase10_seed import (
    COLORS,
    PHASE10_MASTER_SEED,
    SELECTOR_DRAW_SEED,
    Phase10SeedError,
    selector_base_uniform,
    selector_branch_uniform,
)
from .phase10_selector import (
    ALLOWED_REQUEST_FIELDS,
    BRANCH_LEARNED,
    BRANCH_NEUTRAL,
    LearnedSetupSource,
    Phase10SelectorError,
    SelectorRequest,
    candidate,
    classify_construction_failure,
    learned_branch_shares_phase7_decisions,
    load_scorer,
    neutral_branch_matches_accepted_sampler,
)

#: The soak workload version: the frozen Agent 6 integration exercise.
SOAK_VERSION = "phase10_integration_soak_v1"

#: The soak identity version: the game-id prefix and the seed payload prefix.
#: Distinct from `phase10_identity_v1` on purpose — the distinct first token
#: is what makes a soak stream structurally unable to collide with a frozen
#: Agent 1 stream.
SOAK_ID_VERSION = "phase10_soak_v1"

#: One stored soak record's layout version.
SOAK_RECORD_VERSION = "phase10_soak_record_v1"

#: The soak commit protocol: one canonical JSON envelope line per game,
#: newline-committed, digest-bound, torn tails truncated on reconcile.
SOAK_COMMIT_VERSION = "phase10_soak_commit_v1"

#: The soak plays the production system, so its split is the production
#: split: train only. Validation cases and test cases never enter the soak.
SOAK_SPLIT = "train"

#: The frozen soak volume: at least 8,192 complete games.
SOAK_TOTAL_GAMES = 8_192

#: The move-policy identity soak records carry. Distinct from the corpus and
#: from every Phase 9 identity so a soak row can never be mistaken for either.
SOAK_MOVE_POLICY_ID = "phase10_soak_move_v1"

#: The permanently selected configuration (Agent 5, frozen). The soak module
#: pins the id; the harness verifies it against the frozen config artifact's
#: bytes before any game is played.
SELECTED_CANDIDATE_ID = "P10-D"

#: SHA-256 of `reports/phase_10_data/agent_05_frozen_selector_config.json`.
SELECTED_CONFIG_SHA256 = (
    "6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668"
)

#: The two soak stream domains and their frozen roots. Selector draws hang
#: off the frozen selector-draw root because that is what they are; match
#: seeds hang off the phase master root, the phase-wide root every Phase 10
#: logical id already folds in. Neither reading adds a root.
SOAK_DOMAIN_SELECTOR = "soak_selector"
SOAK_DOMAIN_MATCH = "soak_match"
SOAK_DOMAIN_ROOTS = {
    SOAK_DOMAIN_SELECTOR: SELECTOR_DRAW_SEED,
    SOAK_DOMAIN_MATCH: PHASE10_MASTER_SEED,
}

#: Result tokens, from the Red perspective — the corpus tokens, reused.
SOAK_RESULT_TARGETS = {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}

_SOAK_SEED_PERSON = b"strat-s10"

_SOAK_ID_PATTERN = re.compile(
    rf"^{SOAK_ID_VERSION}\|ms=(?P<master>[0-9]+)\|g=(?P<ordinal>[0-9]{{5}})$"
)

MAX_SOAK_ORDINAL_FORMAT = 99_999

JOURNAL_DIRECTORY = "journal"
JOURNAL_SUFFIX = ".soak.jsonl"
STATE_FILENAME = "soak_state.json"
SEAL_FILENAME = "soak_seal.json"

STATE_COLLECTING = "COLLECTING"
STATE_SEALED = "SEALED"

_FILE_SET_PATTERN = re.compile(r"^seg(?P<segment>\d{4})_w(?P<worker>\d{2})$")

#: Every field of a stored soak record. A closed set: an unexpected field is
#: a rejected record, and a missing one is too.
SOAK_RECORD_FIELDS = (
    "soak_version",
    "record_version",
    "game_id",
    "ordinal",
    "split",
    "selector_config_sha256",
    "candidate_id",
    "selector_identity",
    "match_seed",
    "red_selector_request",
    "blue_selector_request",
    "red_selector_provenance",
    "blue_selector_provenance",
    "red_setup_provenance",
    "blue_setup_provenance",
    "red_base_setup_id",
    "blue_base_setup_id",
    "red_family",
    "blue_family",
    "red_final_fingerprint",
    "blue_final_fingerprint",
    "result",
    "winner",
    "red_score",
    "plies",
    "decisions",
    "terminal_reason",
    "move_policy_identity",
    "move_checkpoint_sha256",
    "move_model_state_digest",
    "library_content_digest",
    "contract_bundle_digest",
)

#: Every field of a commit envelope line.
SOAK_ENVELOPE_FIELDS = (
    "commit_version",
    "game_id",
    "payload_sha256",
    "committed_unix",
    "record",
)

#: The audit's zero-tolerance counters. The selector-audit list, plus the
#: soak's own integration counters; a missing counter is a failure, not a
#: pass.
SOAK_AUDIT_COUNTERS = (
    "illegal_setups",
    "inventory_errors",
    "stranded_sampled_setups",
    "split_violations",
    "provenance_mismatches",
    "determinism_mismatches",
    "non_finite_selector_values",
    "selector_identity_mismatches",
    "seed_derivation_mismatches",
    "hidden_opponent_input_fields",
    "outcome_inconsistencies",
    "unscheduled_game_ids",
)


class Phase10SoakError(RuntimeError):
    """Raised when a soak identity, record or store condition is violated."""


# ---------------------------------------------------------------------------
# Identity: game ids and seeds
# ---------------------------------------------------------------------------


def soak_game_id(ordinal: int) -> str:
    """The stable identifier of one logical soak game.

    ```text
    phase10_soak_v1|ms=2026081801|g=00042
    ```

    A pure function of the frozen master seed and the game ordinal. Worker
    count, arrival order and restart boundaries appear nowhere, which is what
    makes resume exact set subtraction by game id.
    """
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase10SoakError(f"soak ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_SOAK_ORDINAL_FORMAT:
        raise Phase10SoakError(
            f"soak ordinal {ordinal} is outside 0..{MAX_SOAK_ORDINAL_FORMAT}"
        )
    return f"{SOAK_ID_VERSION}|ms={PHASE10_MASTER_SEED}|g={ordinal:05d}"


def parse_soak_game_id(game_id: str) -> dict:
    """The identity fields of a soak game id, validated."""
    match = _SOAK_ID_PATTERN.match(game_id) if isinstance(game_id, str) else None
    if match is None:
        raise Phase10SoakError(f"malformed Phase 10 soak game id: {game_id!r}")
    if int(match["master"]) != PHASE10_MASTER_SEED:
        raise Phase10SoakError(
            f"soak game id names master seed {match['master']}, expected "
            f"{PHASE10_MASTER_SEED}"
        )
    return {
        "soak_id_version": SOAK_ID_VERSION,
        "phase10_master_seed": int(match["master"]),
        "ordinal": int(match["ordinal"]),
    }


def soak_game_ids(total: int = SOAK_TOTAL_GAMES) -> tuple:
    """The complete soak schedule: ordinals 0..total-1, in canonical order."""
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise Phase10SoakError(f"soak volume must be a positive int, got {total!r}")
    return tuple(soak_game_id(ordinal) for ordinal in range(total))


def derive_soak_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one soak stream.

    The frozen Agent 1 derivation, applied under the soak identity version:
    the payload's first token is ``phase10_soak_v1`` where every frozen
    Agent 1 stream's is ``phase10_identity_v1``, so equal payloads — and
    therefore colliding streams — are impossible across the two namespaces.
    """
    if domain not in SOAK_DOMAIN_ROOTS:
        raise Phase10SoakError(f"unknown soak stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase10SoakError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
    payload = ":".join(
        [
            SOAK_ID_VERSION,
            domain,
            str(SOAK_DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_SOAK_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def soak_selector_seed(game_id: str, color: str) -> int:
    """The selector seed of one soak side's production draw."""
    parse_soak_game_id(game_id)
    if color not in COLORS:
        raise Phase10SoakError(f"colour must be one of {list(COLORS)}, got {color!r}")
    return derive_soak_seed(SOAK_DOMAIN_SELECTOR, game_id, color)


def soak_match_seed(game_id: str) -> int:
    """The match-level randomness seed of one soak game.

    Both sides play the accepted Phase 9 checkpoint greedily, so no policy
    consumes this stream; it is frozen anyway so a soak record names a match
    seed a replay must reproduce — the corpus rule, reused.
    """
    parse_soak_game_id(game_id)
    return derive_soak_seed(SOAK_DOMAIN_MATCH, game_id)


def soak_seed_collision_audit(total: int = SOAK_TOTAL_GAMES) -> dict:
    """Prove the soak streams are collision-free and disjoint from the frozen
    Phase 10 streams, empirically over the materialized id space.

    The structural argument — distinct payload first tokens — is checked by
    enumeration rather than trusted: every soak selector/match seed is
    compared against every corpus setup/match seed, every bank
    opponent/selector/match seed, and the selected candidate's train-split
    audit seeds.
    """
    from ..setups.families import FAMILY_IDS
    from .phase10_seed import (
        case_match_seed,
        case_opponent_setup_seed,
        case_selector_seed,
        corpus_match_seed,
        corpus_setup_seed,
        phase10_case_id,
        phase10_game_id,
        selector_audit_seed,
        stream_collision_audit,
    )

    ids = soak_game_ids(total)
    streams: dict = {
        "soak_selector": [
            soak_selector_seed(game_id, color) for game_id in ids for color in COLORS
        ],
        "soak_match": [soak_match_seed(game_id) for game_id in ids],
    }

    corpus_ids = [
        phase10_game_id(red, blue, ordinal)
        for red in FAMILY_IDS
        for blue in FAMILY_IDS
        for ordinal in range(64)
    ]
    streams["corpus_setup_attempt0"] = [
        corpus_setup_seed(game_id, color, 0) for game_id in corpus_ids for color in COLORS
    ]
    streams["corpus_match"] = [corpus_match_seed(game_id) for game_id in corpus_ids]

    case_ids = [
        phase10_case_id("phase10_validation_bank_v1", family, ordinal)
        for family in FAMILY_IDS
        for ordinal in range(8)
    ] + [
        phase10_case_id("phase10_test_bank_v1", family, ordinal)
        for family in FAMILY_IDS
        for ordinal in range(32)
    ]
    streams["bank_opponent_attempt0"] = [
        case_opponent_setup_seed(case_id, 0) for case_id in case_ids
    ]
    streams["bank_selector_attempt0"] = [
        case_selector_seed(case_id, color, 0) for case_id in case_ids for color in COLORS
    ]
    streams["bank_match_sample"] = [
        case_match_seed(case_id, game_index, "learned_vs_neutral")
        for case_id in case_ids
        for game_index in (0, 1)
    ]
    streams["selector_audit_selected_train"] = [
        selector_audit_seed(SELECTED_CANDIDATE_ID, SOAK_SPLIT, color, ordinal)
        for color in COLORS
        for ordinal in range(0, 100_000, 7)
    ]
    return stream_collision_audit(streams)


# ---------------------------------------------------------------------------
# The store: one canonical JSON envelope line per committed game
# ---------------------------------------------------------------------------


def canonical_json(payload) -> str:
    """The one canonical text form used for every digest in this module."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def record_payload_sha256(record: dict) -> str:
    """SHA-256 over a record's canonical JSON — the line's binding digest."""
    return hashlib.sha256(canonical_json(record).encode()).hexdigest()


def journal_directory(root: "str | Path") -> Path:
    return Path(root) / JOURNAL_DIRECTORY


def file_set_name(segment: int, worker_id: int) -> str:
    return f"seg{int(segment):04d}_w{int(worker_id):02d}"


def state_path(root: "str | Path") -> Path:
    return Path(root) / STATE_FILENAME


def seal_path(root: "str | Path") -> Path:
    return Path(root) / SEAL_FILENAME


def read_soak_state(root: "str | Path") -> str:
    path = state_path(root)
    if not path.exists():
        return STATE_COLLECTING
    payload = json.loads(path.read_text())
    state = str(payload.get("state"))
    if state not in (STATE_COLLECTING, STATE_SEALED):
        raise Phase10SoakError(f"{path}: unknown soak state {state!r}")
    return state


def require_soak_collecting(root: "str | Path") -> None:
    """Refuse any mutation of a sealed soak store. A seal is not advisory."""
    if read_soak_state(root) == STATE_SEALED:
        raise Phase10SoakError(
            f"{root} is SEALED; a sealed soak store is immutable and must not be "
            "appended to, truncated or re-collected"
        )


def validate_soak_record(record: dict) -> dict:
    """One stored soak record, validated field by field against the closed set."""
    if not isinstance(record, dict):
        raise Phase10SoakError(f"a soak record is a mapping, got {type(record).__name__}")
    missing = [name for name in SOAK_RECORD_FIELDS if name not in record]
    extra = [name for name in record if name not in SOAK_RECORD_FIELDS]
    if missing or extra:
        raise Phase10SoakError(
            f"soak record is malformed: missing={missing} unexpected={extra}"
        )
    if record["soak_version"] != SOAK_VERSION:
        raise Phase10SoakError(
            f"record names soak {record['soak_version']!r}, not {SOAK_VERSION!r}"
        )
    if record["record_version"] != SOAK_RECORD_VERSION:
        raise Phase10SoakError(
            f"record names layout {record['record_version']!r}, not "
            f"{SOAK_RECORD_VERSION!r}"
        )
    identity = parse_soak_game_id(record["game_id"])
    if record["ordinal"] != identity["ordinal"]:
        raise Phase10SoakError(
            f"{record['game_id']}: stored ordinal {record['ordinal']} contradicts the id"
        )
    if record["split"] != SOAK_SPLIT:
        raise Phase10SoakError(
            f"{record['game_id']}: split {record['split']!r} is not {SOAK_SPLIT!r}"
        )
    result = record["result"]
    if result not in SOAK_RESULT_TARGETS:
        raise Phase10SoakError(f"unknown result token {result!r}")
    if float(record["red_score"]) != SOAK_RESULT_TARGETS[result]:
        raise Phase10SoakError(
            f"result {result!r} carries red_score {record['red_score']!r}, not the "
            f"frozen target {SOAK_RESULT_TARGETS[result]}"
        )
    expected_winner = (
        None if result == "draw" else ("red" if result == "red_win" else "blue")
    )
    if record["winner"] != expected_winner:
        raise Phase10SoakError(
            f"result {result!r} carries winner {record['winner']!r}"
        )
    for side in ("red", "blue"):
        request = record[f"{side}_selector_request"]
        if not isinstance(request, dict) or set(request) != set(ALLOWED_REQUEST_FIELDS):
            raise Phase10SoakError(
                f"{record['game_id']} {side}: selector request fields "
                f"{sorted(request) if isinstance(request, dict) else request!r} are not "
                f"exactly the frozen allowlist {sorted(ALLOWED_REQUEST_FIELDS)}"
            )
    return dict(record)


class SoakWriter:
    """The append-only writer of one (segment, worker) soak journal.

    Not shared and not thread-safe: one instance belongs to one process,
    which is the only thing that ever appends to this file. Every line is
    flushed as it is written, so a commit is durable at the newline and a
    SIGKILL anywhere leaves at worst one torn, invisible tail line.
    """

    def __init__(self, root: "str | Path", *, segment: int, worker_id: int) -> None:
        self.root = Path(root)
        self.segment = int(segment)
        self.worker_id = int(worker_id)
        require_soak_collecting(self.root)
        self.name = file_set_name(self.segment, self.worker_id)
        self.journal_path = journal_directory(self.root) / f"{self.name}{JOURNAL_SUFFIX}"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if self.journal_path.exists():
            raise Phase10SoakError(
                f"file set {self.name} already exists; a resumed run must open a "
                "fresh segment rather than append to a reconciled one"
            )
        self._handle = open(self.journal_path, "ab")
        self.games_written = 0
        self.bytes_written = 0
        self.write_seconds = 0.0

    def write_record(self, record: dict) -> dict:
        """Validate, envelope and commit one soak record."""
        record = validate_soak_record(record)
        envelope = {
            "commit_version": SOAK_COMMIT_VERSION,
            "game_id": record["game_id"],
            "payload_sha256": record_payload_sha256(record),
            "committed_unix": time.time(),
            "record": record,
        }
        line = (canonical_json(envelope) + "\n").encode()
        started = time.perf_counter()
        self._handle.write(line)
        self._handle.flush()
        self.write_seconds += time.perf_counter() - started
        self.games_written += 1
        self.bytes_written += len(line)
        return envelope

    def close(self) -> dict:
        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        self._handle = None
        return self.stats()

    def stats(self) -> dict:
        return {
            "file_set": self.name,
            "segment": self.segment,
            "worker_id": self.worker_id,
            "games_written": self.games_written,
            "bytes_written": self.bytes_written,
            "write_seconds": self.write_seconds,
        }


def read_soak_journal(path: "str | Path") -> tuple:
    """`(envelopes, valid_bytes)` for one soak journal.

    Only newline-terminated lines that parse, carry every envelope field,
    name the frozen commit version and whose payload digest matches their
    record are commits; the first line that fails any of that ends the valid
    prefix, so a process killed mid-write contributes nothing.
    """
    path = Path(path)
    if not path.exists():
        return ([], 0)
    raw = path.read_bytes()
    envelopes: list = []
    offset = 0
    valid = 0
    while True:
        index = raw.find(b"\n", offset)
        if index < 0:
            break
        line = raw[offset:index]
        offset = index + 1
        text = line.strip()
        if not text:
            valid = offset
            continue
        try:
            payload = json.loads(text.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
        if not isinstance(payload, dict) or any(
            key not in payload for key in SOAK_ENVELOPE_FIELDS
        ):
            break
        if payload["commit_version"] != SOAK_COMMIT_VERSION:
            raise Phase10SoakError(
                f"{path}: commit protocol {payload['commit_version']!r} is not "
                f"{SOAK_COMMIT_VERSION!r}"
            )
        if record_payload_sha256(payload["record"]) != payload["payload_sha256"]:
            break
        if payload["record"].get("game_id") != payload["game_id"]:
            break
        envelopes.append(payload)
        valid = offset
    return (envelopes, valid)


def _file_sets(root: "str | Path") -> list:
    resolved = []
    for path in sorted(journal_directory(root).glob(f"*{JOURNAL_SUFFIX}")):
        name = path.name[: -len(JOURNAL_SUFFIX)]
        match = _FILE_SET_PATTERN.match(name)
        if match is None:
            raise Phase10SoakError(f"unrecognized soak file set name: {name!r}")
        resolved.append((int(match["segment"]), int(match["worker"]), name))
    return resolved


def next_soak_segment(root: "str | Path") -> int:
    """The first segment number no file set has used. A resumed run always
    writes a fresh segment, so the attempt boundary is readable off the
    filenames."""
    highest = -1
    for segment, _worker, _name in _file_sets(root):
        highest = max(highest, segment)
    return highest + 1


def reconcile_soak(root: "str | Path") -> dict:
    """Truncate every torn journal tail and index the committed games.

    The only function that ever removes soak bytes, and it removes exactly
    the bytes no commit line claims — by the write order, only work that was
    interrupted before it became visible.
    """
    root = Path(root)
    require_soak_collecting(root)
    committed: dict = {}
    duplicates: list = []
    reports: list = []
    for _segment, _worker, name in _file_sets(root):
        path = journal_directory(root) / f"{name}{JOURNAL_SUFFIX}"
        envelopes, valid = read_soak_journal(path)
        size = path.stat().st_size
        discarded = size - valid
        if discarded:
            with path.open("r+b") as handle:
                handle.truncate(valid)
                handle.flush()
                os.fsync(handle.fileno())
        reports.append(
            {
                "file_set": name,
                "committed_games": len(envelopes),
                "journal_bytes_discarded": discarded,
                "torn_tail": discarded > 0,
            }
        )
        for envelope in envelopes:
            game_id = envelope["game_id"]
            if game_id in committed:
                duplicates.append(game_id)
            committed[game_id] = envelope
    return {
        "commit_version": SOAK_COMMIT_VERSION,
        "committed": committed,
        "committed_count": len(committed),
        "duplicate_committed_ids": sorted(set(duplicates)),
        "file_sets": reports,
        "bytes_discarded": sum(report["journal_bytes_discarded"] for report in reports),
    }


def soak_committed_count(root: "str | Path") -> int:
    """Committed games, read-only: no truncation, valid lines only."""
    count = 0
    for _segment, _worker, name in _file_sets(root):
        envelopes, _valid = read_soak_journal(
            journal_directory(root) / f"{name}{JOURNAL_SUFFIX}"
        )
        count += len(envelopes)
    return count


class SoakReader:
    """Read-only access to a committed soak store, in canonical game-id order."""

    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)
        self.state = read_soak_state(self.root)
        self._envelopes: dict = {}
        duplicates: list = []
        self.journal_bytes = 0
        self.file_set_count = 0
        for _segment, _worker, name in _file_sets(self.root):
            path = journal_directory(self.root) / f"{name}{JOURNAL_SUFFIX}"
            envelopes, valid = read_soak_journal(path)
            self.journal_bytes += valid
            self.file_set_count += 1
            for envelope in envelopes:
                game_id = envelope["game_id"]
                if game_id in self._envelopes:
                    duplicates.append(game_id)
                self._envelopes[game_id] = envelope
        self.duplicate_committed_ids = sorted(set(duplicates))
        self.game_ids = tuple(sorted(self._envelopes))

    def __len__(self) -> int:
        return len(self._envelopes)

    def envelope(self, game_id: str) -> dict:
        try:
            return self._envelopes[game_id]
        except KeyError as error:
            raise Phase10SoakError(f"{game_id} is not committed in {self.root}") from error

    def record(self, game_id: str) -> dict:
        envelope = self.envelope(game_id)
        record = envelope["record"]
        if record_payload_sha256(record) != envelope["payload_sha256"]:
            raise Phase10SoakError(
                f"{game_id}: stored payload digest disagrees with its envelope"
            )
        return record

    def iter_records(self):
        for game_id in self.game_ids:
            yield self.record(game_id)

    def storage_summary(self) -> dict:
        games = max(len(self), 1)
        return {
            "committed_games": len(self),
            "file_sets": self.file_set_count,
            "journal_bytes": self.journal_bytes,
            "total_bytes": self.journal_bytes,
            "bytes_per_game": self.journal_bytes / games,
        }


def soak_content_digest(root: "str | Path") -> str:
    """SHA-256 over every committed payload digest, in canonical game-id order.

    Path-independent by construction: the same records copied to another
    volume, or rewritten by a differently partitioned run, produce the same
    value.
    """
    reader = SoakReader(root)
    digest = hashlib.sha256()
    digest.update(f"{SOAK_VERSION}|{SOAK_RECORD_VERSION}|{len(reader)}".encode())
    for game_id in reader.game_ids:
        digest.update(f"|{game_id}|{reader.envelope(game_id)['payload_sha256']}".encode())
    return digest.hexdigest()


def seal_soak(root: "str | Path", *, expected_games: int, extra: "dict | None" = None) -> dict:
    """Move the soak store `COLLECTING -> SEALED` and freeze its content identity."""
    root = Path(root)
    if read_soak_state(root) == STATE_SEALED:
        raise Phase10SoakError(f"{root} is already SEALED")
    reader = SoakReader(root)
    if reader.duplicate_committed_ids:
        raise Phase10SoakError(
            f"refusing to seal {root}: duplicate committed ids "
            f"{reader.duplicate_committed_ids[:4]}"
        )
    if len(reader) != int(expected_games):
        raise Phase10SoakError(
            f"refusing to seal {root}: {len(reader)} committed games, expected "
            f"{expected_games}"
        )
    for game_id in reader.game_ids:
        validate_soak_record(reader.record(game_id))
    digest = soak_content_digest(root)
    seal = {
        "soak_version": SOAK_VERSION,
        "record_version": SOAK_RECORD_VERSION,
        "commit_version": SOAK_COMMIT_VERSION,
        "committed_games": len(reader),
        "content_digest": digest,
        "sealed_unix": time.time(),
        **(extra or {}),
    }
    seal_path(root).write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    state_payload = {
        "soak_version": SOAK_VERSION,
        "commit_version": SOAK_COMMIT_VERSION,
        "state": STATE_SEALED,
        "content_digest": digest,
        "committed_games": len(reader),
        "updated_unix": time.time(),
    }
    state_path(root).write_text(json.dumps(state_payload, indent=2, sort_keys=True) + "\n")
    return seal


def read_soak_seal(root: "str | Path") -> dict:
    path = seal_path(root)
    if not path.exists():
        raise Phase10SoakError(f"{root} carries no soak seal")
    return json.loads(path.read_text())


def verify_soak_seal(root: "str | Path") -> dict:
    """Recompute a sealed soak store's content digest against its seal."""
    seal = read_soak_seal(root)
    observed = soak_content_digest(root)
    reader = SoakReader(root)
    checks = {
        "state_is_sealed": read_soak_state(root) == STATE_SEALED,
        "content_digest_matches": observed == seal["content_digest"],
        "committed_games_match": len(reader) == int(seal["committed_games"]),
        "soak_version_matches": seal["soak_version"] == SOAK_VERSION,
    }
    return {
        "seal": seal,
        "observed_content_digest": observed,
        "observed_committed_games": len(reader),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Storage resolution: where soak bytes live, and why that is not identity
# ---------------------------------------------------------------------------

DEFAULT_PHASE10_SOAK_ROOT = "data/phase10/soak"
PHASE10_SOAK_ROOT_ENV = "STRATEGO_PHASE10_SOAK_ROOT"
PHASE10_SOAK_ROOT_POINTER = "data/phase10_soak_root.txt"


def default_soak_root() -> Path:
    """Where this installation keeps soak journal bytes.

    The accepted Phase 10 resolution order, applied to the soak namespace:
    environment override, then the durable pointer file, then the repository
    default. Identity never contains the result — the soak content digest is
    computed over committed payload digests alone.
    """
    from .phase10_storage import repository_root

    configured = os.environ.get(PHASE10_SOAK_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    pointer = repository_root() / PHASE10_SOAK_ROOT_POINTER
    if pointer.exists():
        recorded = pointer.read_text().strip()
        if recorded:
            return Path(recorded).expanduser()
    return repository_root() / DEFAULT_PHASE10_SOAK_ROOT


def describe_soak_root() -> dict:
    """Which redirect (if any) chose the soak root, for the manifest."""
    from .phase10_storage import STORAGE_IDENTITY_RULE, repository_root

    configured = os.environ.get(PHASE10_SOAK_ROOT_ENV, "").strip()
    pointer = repository_root() / PHASE10_SOAK_ROOT_POINTER
    recorded = pointer.read_text().strip() if pointer.exists() else ""
    if configured:
        source = "environment"
    elif recorded:
        source = "pointer_file"
    else:
        source = "repository_default"
    return {
        "root": str(default_soak_root()),
        "source": source,
        "environment_variable": PHASE10_SOAK_ROOT_ENV,
        "environment_value": configured,
        "pointer_file": str(pointer),
        "pointer_value": recorded,
        "repository_default": str(repository_root() / DEFAULT_PHASE10_SOAK_ROOT),
        "identity_rule": STORAGE_IDENTITY_RULE,
    }


def probe_volume_health(root: "str | Path") -> dict:
    """Mount safety, capacity and an actual write/fsync/read-back probe."""
    from .phase10_storage import check_corpus_root

    root = Path(root)
    findings = check_corpus_root(root)
    probe = {"attempted": False, "ok": False, "error": None}
    if findings["usable"]:
        probe["attempted"] = True
        try:
            root.mkdir(parents=True, exist_ok=True)
            token = os.urandom(16).hex()
            path = root / f".volume_probe_{os.getpid()}"
            with path.open("w") as handle:
                handle.write(token)
                handle.flush()
                os.fsync(handle.fileno())
            probe["ok"] = path.read_text() == token
            path.unlink()
        except OSError as error:  # pragma: no cover - depends on the host volume
            probe["error"] = f"{type(error).__name__}: {error}"
    return {**findings, "write_probe": probe}


# ---------------------------------------------------------------------------
# Playing one soak game
# ---------------------------------------------------------------------------


def selected_candidate():
    """The permanently selected configuration, as a frozen candidate object."""
    return candidate(SELECTED_CANDIDATE_ID)


def selected_selector_identity() -> str:
    return selected_candidate().selector_identity


def build_soak_source(scorer=None, index=None) -> LearnedSetupSource:
    """The production `learned_setup_source_v1` under the selected config."""
    from ..setups.sampler import load_library_index

    return LearnedSetupSource(
        selected_candidate(),
        load_scorer() if scorer is None else scorer,
        load_library_index() if index is None else index,
    )


def soak_policy_ref():
    """The `PolicyRef` both sides of every soak game play under."""
    from ..evaluation.neural_worker import neural_policy_ref
    from ..model.policy_adapter import DECISION_MODE_GREEDY

    return neural_policy_ref(
        SOAK_MOVE_POLICY_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name="float32"
    )


def soak_match_spec(game_id: str):
    """The fully determined `MatchSpec` of one soak game.

    Both sides name the same policy reference — self-play under one frozen
    checkpoint — and `setup_bank_version` carries the soak version plus the
    frozen selector identity, so the match id is a cryptographic statement of
    *which* production configuration supplied the setups.
    """
    from ..engine.constants import RED
    from ..evaluation.match_spec import MatchSpec

    identity = parse_soak_game_id(game_id)
    ref = soak_policy_ref()
    return MatchSpec(
        candidate=ref,
        opponent=ref,
        setup_pair_id=identity["ordinal"],
        candidate_color=RED,
        root_seed=soak_match_seed(game_id),
        suite_version=SOAK_VERSION,
        setup_bank_version=f"{SOAK_VERSION}|{selected_selector_identity()}",
    )


def draw_soak_sides(source: LearnedSetupSource, game_id: str) -> dict:
    """Both sides' production draws, through the audited request boundary.

    Requests are built with `SelectorRequest.from_payload` — the entry point
    that rejects any field outside the frozen allowlist — so every soak draw
    exercises the same information-safety wall production traffic would.
    """
    sides: dict = {}
    for color in COLORS:
        payload = {
            "split": SOAK_SPLIT,
            "color": color,
            "selector_seed": soak_selector_seed(game_id, color),
        }
        sides[color] = {
            "request": dict(payload),
            "draw": source.draw(SelectorRequest.from_payload(payload)),
        }
    return sides


def play_soak_game(game_id: str, policy, source: LearnedSetupSource, *, sides=None):
    """`(MatchResult, sides)` for one soak game."""
    from ..engine.constants import BLUE, RED
    from ..evaluation.match_runner import ON_POLICY_ERROR_RAISE, play_match

    resolved = draw_soak_sides(source, game_id) if sides is None else sides
    spec = soak_match_spec(game_id)
    result = play_match(
        spec,
        setups=(
            resolved["red"]["draw"].oriented(RED),
            resolved["blue"]["draw"].oriented(BLUE),
        ),
        policies={spec.candidate.token: policy},
        record_actions=False,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    return result, resolved


def soak_result_token(result) -> str:
    """The Red-perspective outcome token of a played soak match."""
    from ..engine.constants import BLUE, RED

    if result.draw or result.winner is None:
        return "draw"
    if result.winner == RED:
        return "red_win"
    if result.winner == BLUE:
        return "red_loss"
    raise Phase10SoakError(f"unknown winner {result.winner!r}")


def soak_identity_block(export: dict) -> dict:
    """The digests every soak record carries, computed once per run."""
    from ..setups.sampler import load_library_index
    from . import phase10_contract as contract

    return {
        "library_content_digest": load_library_index().content_digest,
        "contract_bundle_digest": contract.contract_bundle_digest(),
        "phase9_checkpoint_sha256": export["source_sha256"],
        "phase9_model_state_digest": export["model_state_digest"],
        "selector_config_sha256": SELECTED_CONFIG_SHA256,
        "selector_identity": selected_selector_identity(),
        "candidate_id": SELECTED_CANDIDATE_ID,
    }


def build_soak_record(game_id: str, result, sides: dict, *, identity: dict) -> dict:
    """The flat stored record of one played soak game."""
    from ..engine.constants import RED

    token = soak_result_token(result)
    if result.candidate_color != RED:  # pragma: no cover - fixed by construction
        raise Phase10SoakError(f"{game_id}: candidate colour is not Red")
    parsed = parse_soak_game_id(game_id)
    record = {
        "soak_version": SOAK_VERSION,
        "record_version": SOAK_RECORD_VERSION,
        "game_id": game_id,
        "ordinal": parsed["ordinal"],
        "split": SOAK_SPLIT,
        "selector_config_sha256": identity["selector_config_sha256"],
        "candidate_id": identity["candidate_id"],
        "selector_identity": identity["selector_identity"],
        "match_seed": soak_match_seed(game_id),
        "result": token,
        "winner": None if token == "draw" else ("red" if token == "red_win" else "blue"),
        "red_score": SOAK_RESULT_TARGETS[token],
        "plies": int(result.plies),
        "decisions": int(result.decisions),
        "terminal_reason": result.terminal_reason,
        "move_policy_identity": soak_policy_ref().token,
        "move_checkpoint_sha256": identity["phase9_checkpoint_sha256"],
        "move_model_state_digest": identity["phase9_model_state_digest"],
        "library_content_digest": identity["library_content_digest"],
        "contract_bundle_digest": identity["contract_bundle_digest"],
    }
    for color in COLORS:
        draw = sides[color]["draw"]
        if draw.split != SOAK_SPLIT:
            raise Phase10SoakError(
                f"{game_id} {color}: draw split {draw.split!r} is not {SOAK_SPLIT!r}"
            )
        if draw.selector_identity != identity["selector_identity"]:
            raise Phase10SoakError(
                f"{game_id} {color}: draw identity {draw.selector_identity!r} is not "
                f"the selected configuration"
            )
        record[f"{color}_selector_request"] = dict(sides[color]["request"])
        record[f"{color}_selector_provenance"] = draw.selector_provenance()
        record[f"{color}_setup_provenance"] = dict(draw.setup_provenance)
        record[f"{color}_base_setup_id"] = draw.base_setup_id
        record[f"{color}_family"] = draw.family_id
        record[f"{color}_final_fingerprint"] = draw.final_setup_fingerprint
    return validate_soak_record(record)


# ---------------------------------------------------------------------------
# Collection: parallel workers, restart, resume by logical game id
# ---------------------------------------------------------------------------


def _peak_rss_bytes() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux kilobytes.
    return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


def collect_soak_slice(
    root: "str | Path",
    game_ids,
    *,
    segment: int,
    worker_id: int,
    export_path: "str | Path",
    identity: dict,
    device: str = "cpu",
    torch_threads: int = 1,
    expected_state_digest: "str | None" = None,
    progress=None,
) -> dict:
    """Play and commit one worker's soak games into its own file set."""
    import torch

    from .phase10_collector import load_corpus_owner, owner_state_digest

    torch.set_num_threads(int(torch_threads))
    game_ids = list(game_ids)
    started = time.perf_counter()

    owner = load_corpus_owner(export_path, device=device, name=f"phase10_soak_w{worker_id:02d}")
    state_digest = owner_state_digest(owner)
    if expected_state_digest is not None and state_digest != expected_state_digest:
        raise Phase10SoakError(
            f"worker {worker_id} loaded model-state {state_digest}, expected "
            f"{expected_state_digest}; the move-policy identity would be a lie"
        )
    if identity["phase9_model_state_digest"] != state_digest:
        raise Phase10SoakError(
            f"worker {worker_id} would stamp model-state "
            f"{identity['phase9_model_state_digest']} onto records played by {state_digest}"
        )
    from ..evaluation.neural_worker import LocalInferenceChannel, RemoteNeuralPolicy
    from ..model.policy_adapter import DECISION_MODE_GREEDY

    policy = RemoteNeuralPolicy(
        soak_policy_ref(), LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY
    )
    source = build_soak_source()

    writer = SoakWriter(root, segment=segment, worker_id=worker_id)
    plies = 0
    decisions = 0
    try:
        for position, game_id in enumerate(game_ids):
            result, sides = play_soak_game(game_id, policy, source)
            record = build_soak_record(game_id, result, sides, identity=identity)
            writer.write_record(record)
            plies += int(result.plies)
            decisions += int(result.decisions)
            if progress is not None:
                progress(worker_id, position + 1, len(game_ids))
    finally:
        stats = writer.close()
        owner.close()

    elapsed = time.perf_counter() - started
    stats.update(
        {
            "games": len(game_ids),
            "plies": plies,
            "decisions": decisions,
            "wall_clock_seconds": elapsed,
            "games_per_second": len(game_ids) / elapsed if elapsed else 0.0,
            "decisions_per_second": decisions / elapsed if elapsed else 0.0,
            "model_state_digest": state_digest,
            "device": device,
            "torch_threads": int(torch_threads),
            "policy_decisions": policy.decisions,
            "inference_failures": owner.stats().get("failures_returned", 0),
            "checkpoint_loads": owner.checkpoint_load_count,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )
    return stats


def _soak_worker_main(payload: dict, queue) -> None:
    """One spawned soak collection worker."""
    report = {"worker_id": payload["worker_id"], "status": "ok", "pid": os.getpid()}
    try:
        report["stats"] = collect_soak_slice(
            payload["root"],
            payload["game_ids"],
            segment=payload["segment"],
            worker_id=payload["worker_id"],
            export_path=payload["export_path"],
            identity=payload["identity"],
            device=payload["device"],
            torch_threads=payload["torch_threads"],
            expected_state_digest=payload["expected_state_digest"],
        )
    except BaseException as error:  # noqa: BLE001 -- reported to the parent verbatim
        import traceback

        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    queue.put(report)


def _drain_soak_reports(
    queue, processes, *, poll_seconds: float = 1.0, expect_kill: bool = False
) -> list:
    """Collect one report per worker, refusing to block on a dead one.

    Under an armed kill timer (`expect_kill`), dead workers are the timer
    doing its job an instant before it reaches this process too, so the drain
    keeps waiting for its own SIGKILL instead of racing it to an error.
    """
    import queue as queue_module

    reports: list = []
    while len(reports) < len(processes):
        try:
            reports.append(queue.get(timeout=poll_seconds))
            continue
        except queue_module.Empty:
            pass
        if expect_kill:
            continue
        dead = [
            process
            for process in processes
            if process.exitcode is not None and process.exitcode != 0
        ]
        if dead and len(reports) < len(processes):
            codes = ", ".join(f"pid {p.pid} exit {p.exitcode}" for p in dead)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise Phase10SoakError(
                f"soak worker(s) died without reporting ({codes}); the committed games "
                "are intact and a resumed run will play only the missing ones"
            )
    return reports


def collect_soak(
    root: "str | Path",
    *,
    export: dict,
    total: int = SOAK_TOTAL_GAMES,
    worker_count: int = 1,
    limit: "int | None" = None,
    device: str = "cpu",
    torch_threads: int = 1,
    kill_after_seconds: "float | None" = None,
    progress=None,
) -> dict:
    """Collect missing scheduled soak games into `root`, then report.

    Reconciles first, so an interrupted attempt's uncommitted tail is gone
    before anything new is written, and plays only the games no commit line
    claims — resume is exact set subtraction by logical game id. ``limit``
    bounds how many missing games this invocation plays (the controlled-
    restart lever); ``kill_after_seconds`` arms a hard SIGKILL of every
    worker and then of this process (the crash-restart lever) — neither
    changes which logical games exist, only which attempt plays them.
    """
    import multiprocessing

    from .phase10_collector import partition

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    recovery = reconcile_soak(root)
    committed = set(recovery["committed"])

    scheduled = list(soak_game_ids(total))
    missing = [game_id for game_id in scheduled if game_id not in committed]
    if limit is not None:
        missing = missing[: max(int(limit), 0)]

    identity = soak_identity_block(export)
    segment = next_soak_segment(root)
    started = time.perf_counter()
    reports: list = []

    if missing:
        buckets = [bucket for bucket in partition(missing, worker_count) if bucket]
        if len(buckets) == 1 and worker_count == 1 and kill_after_seconds is None:
            reports.append(
                {
                    "worker_id": 0,
                    "status": "ok",
                    "pid": os.getpid(),
                    "stats": collect_soak_slice(
                        root,
                        buckets[0],
                        segment=segment,
                        worker_id=0,
                        export_path=export["export_path"],
                        identity=identity,
                        device=device,
                        torch_threads=torch_threads,
                        expected_state_digest=export["model_state_digest"],
                        progress=progress,
                    ),
                }
            )
        else:
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            processes = []
            for worker_id, bucket in enumerate(buckets):
                payload = {
                    "root": str(root),
                    "game_ids": bucket,
                    "segment": segment,
                    "worker_id": worker_id,
                    "export_path": str(export["export_path"]),
                    "identity": identity,
                    "device": device,
                    "torch_threads": torch_threads,
                    "expected_state_digest": export["model_state_digest"],
                }
                process = context.Process(
                    target=_soak_worker_main, args=(payload, queue), daemon=False
                )
                process.start()
                processes.append(process)
            if kill_after_seconds is not None:
                import signal
                import threading

                def _hard_kill() -> None:  # pragma: no cover - exercised via subprocess
                    for process in processes:
                        if process.pid is not None and process.is_alive():
                            try:
                                os.kill(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    os.kill(os.getpid(), signal.SIGKILL)

                timer = threading.Timer(float(kill_after_seconds), _hard_kill)
                timer.daemon = True
                timer.start()
            reports.extend(
                _drain_soak_reports(
                    queue, processes, expect_kill=kill_after_seconds is not None
                )
            )
            for process in processes:
                process.join()
            failed = [report for report in reports if report["status"] != "ok"]
            if failed:
                raise Phase10SoakError(
                    f"{len(failed)} soak worker(s) failed; the first says: "
                    f"{failed[0].get('error')}\n{failed[0].get('traceback', '')}"
                )

    elapsed = time.perf_counter() - started
    games = sum(report["stats"]["games"] for report in reports)
    plies = sum(report["stats"]["plies"] for report in reports)
    decisions = sum(report["stats"]["decisions"] for report in reports)
    return {
        "soak_version": SOAK_VERSION,
        "root": str(root),
        "segment": segment,
        "parent_pid": os.getpid(),
        "worker_count": len(reports),
        "worker_pids": [report.get("pid") for report in reports],
        "recovery": {key: value for key, value in recovery.items() if key != "committed"},
        "already_committed": len(committed),
        "scheduled_games": len(scheduled),
        "requested_this_run": len(missing),
        "games_played": games,
        "plies_played": plies,
        "decisions_played": decisions,
        "committed_after": soak_committed_count(root),
        "wall_clock_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed and games else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed and decisions else 0.0,
        "peak_worker_rss_bytes": max(
            (report["stats"].get("peak_rss_bytes", 0) for report in reports), default=0
        ),
        "inference_failures": sum(
            report["stats"].get("inference_failures", 0) for report in reports
        ),
        "checkpoint_loads": sum(
            report["stats"].get("checkpoint_loads", 0) for report in reports
        ),
        "device": device,
        "torch_threads": int(torch_threads),
        "workers": [report["stats"] for report in reports],
        "identity": identity,
    }


# ---------------------------------------------------------------------------
# The per-game integration audit
# ---------------------------------------------------------------------------


def hidden_input_positive_control() -> dict:
    """Prove the information-safety wall fires, not just that it exists."""
    attempts = {
        "opponent_family": {"split": SOAK_SPLIT, "color": "red", "selector_seed": 1, "opponent_family": "F00"},
        "opponent_base_id": {"split": SOAK_SPLIT, "color": "red", "selector_seed": 1, "opponent_base_id": "x"},
        "outcome": {"split": SOAK_SPLIT, "color": "red", "selector_seed": 1, "outcome": 1.0},
        "storage_path": {"split": SOAK_SPLIT, "color": "red", "selector_seed": 1, "path": "/Volumes/x"},
    }
    rejected = {}
    for name, payload in attempts.items():
        try:
            SelectorRequest.from_payload(payload)
            rejected[name] = False
        except Phase10SelectorError:
            rejected[name] = True
    return {"attempts": len(attempts), "rejected": rejected, "all_rejected": all(rejected.values())}


def verify_soak_game(
    record: dict,
    source: LearnedSetupSource,
    *,
    scheduled: "set[str] | None" = None,
    cross_check_accepted_sampler: bool = True,
) -> dict:
    """Everything one committed soak game must satisfy, recomputed from its id.

    The per-game integration checks of the Agent 6 instruction, mechanized:
    selector identity, requested split, base membership, seed re-derivation,
    branch/uniform agreement, full deterministic redraw, provenance rebuild,
    final-setup legality and inventory, stranded-setup binding, outcome
    consistency, and the closed request field set.
    """
    from ..setups.families import FAMILY_IDS
    from ..setups.sampler import rebuild_from_provenance, validate_sampled_setup

    counters = {name: 0 for name in SOAK_AUDIT_COUNTERS}
    findings: list = []

    try:
        identity = parse_soak_game_id(record["game_id"])
    except Phase10SoakError as error:
        counters["unscheduled_game_ids"] += 1
        return {"counters": counters, "findings": [str(error)], "ok": False}
    game_id = record["game_id"]
    if scheduled is not None and game_id not in scheduled:
        counters["unscheduled_game_ids"] += 1
        findings.append(f"{game_id} is not a scheduled soak game")
    if record["ordinal"] != identity["ordinal"]:
        counters["outcome_inconsistencies"] += 1
        findings.append("stored ordinal contradicts the game id")

    expected_identity = selected_selector_identity()
    if (
        record["selector_identity"] != expected_identity
        or record["candidate_id"] != SELECTED_CANDIDATE_ID
        or record["selector_config_sha256"] != SELECTED_CONFIG_SHA256
    ):
        counters["selector_identity_mismatches"] += 1
        findings.append("record does not name the permanently selected configuration")
    if record["match_seed"] != soak_match_seed(game_id):
        counters["seed_derivation_mismatches"] += 1
        findings.append("stored match seed disagrees with its derivation")

    if record["split"] != SOAK_SPLIT:
        counters["split_violations"] += 1
        findings.append(f"record split {record['split']!r} is not {SOAK_SPLIT!r}")

    for color in COLORS:
        request = record[f"{color}_selector_request"]
        extra = sorted(set(request) - set(ALLOWED_REQUEST_FIELDS))
        if extra:
            counters["hidden_opponent_input_fields"] += 1
            findings.append(f"{color}: request carries illegal fields {extra}")
            continue
        if request["split"] != SOAK_SPLIT or request["color"] != color:
            counters["split_violations"] += 1
            findings.append(f"{color}: request names split/colour it was not dealt")
        expected_seed = soak_selector_seed(game_id, color)
        if request["selector_seed"] != expected_seed:
            counters["seed_derivation_mismatches"] += 1
            findings.append(f"{color}: selector seed disagrees with its derivation")
            continue

        selector_provenance = record[f"{color}_selector_provenance"]
        setup_provenance = record[f"{color}_setup_provenance"]

        branch_uniform = selector_branch_uniform(
            expected_identity, SOAK_SPLIT, color, expected_seed
        )
        expected_branch = (
            BRANCH_NEUTRAL if branch_uniform < NEUTRAL_MIXTURE_WEIGHT else BRANCH_LEARNED
        )
        if not math.isfinite(branch_uniform):
            counters["non_finite_selector_values"] += 1
            findings.append(f"{color}: branch uniform is not finite")
        if selector_provenance["branch"] != expected_branch:
            counters["determinism_mismatches"] += 1
            findings.append(f"{color}: stored branch contradicts the branch coin")
        if selector_provenance["branch_uniform"] != branch_uniform:
            counters["determinism_mismatches"] += 1
            findings.append(f"{color}: stored branch uniform disagrees with derivation")
        if expected_branch == BRANCH_LEARNED:
            base_uniform = selector_base_uniform(
                expected_identity, SOAK_SPLIT, color, expected_seed
            )
            if selector_provenance["base_uniform"] != base_uniform:
                counters["determinism_mismatches"] += 1
                findings.append(f"{color}: stored base uniform disagrees with derivation")

        # The deterministic redraw: the draw is a pure function of its
        # logical identity, so an independent process must reproduce it
        # exactly — base, branch, family, fingerprint and both provenance
        # halves. This is the restart/topology invariance statement, checked
        # per game rather than asserted.
        redraw = source.draw(
            SelectorRequest(split=SOAK_SPLIT, color=color, selector_seed=expected_seed)
        )
        if (
            redraw.base_setup_id != record[f"{color}_base_setup_id"]
            or redraw.branch != selector_provenance["branch"]
            or redraw.family_id != record[f"{color}_family"]
            or redraw.final_setup_fingerprint != record[f"{color}_final_fingerprint"]
            or redraw.selector_provenance() != selector_provenance
            or dict(redraw.setup_provenance) != setup_provenance
        ):
            counters["determinism_mismatches"] += 1
            findings.append(f"{color}: independent redraw does not reproduce the record")

        try:
            rebuilt = rebuild_from_provenance(setup_provenance, source.index)
        except Exception as error:  # noqa: BLE001 -- classified, not swallowed
            counters["provenance_mismatches"] += 1
            findings.append(f"{color}: provenance rebuild failed: {error}")
            continue
        if (
            rebuilt.provenance["final_setup_fingerprint"]
            != record[f"{color}_final_fingerprint"]
            or rebuilt.base_setup_id != record[f"{color}_base_setup_id"]
            or dict(rebuilt.provenance) != setup_provenance
        ):
            counters["provenance_mismatches"] += 1
            findings.append(f"{color}: provenance does not rebuild to the identical setup")

        base_entry = source.index.base(record[f"{color}_base_setup_id"])
        if base_entry.split != SOAK_SPLIT:
            counters["split_violations"] += 1
            findings.append(
                f"{color}: base {record[f'{color}_base_setup_id']} is not a train base"
            )
        if record[f"{color}_family"] not in FAMILY_IDS:
            counters["illegal_setups"] += 1
            findings.append(f"{color}: unknown family {record[f'{color}_family']!r}")

        failures = validate_sampled_setup(
            rebuilt.canonical, base_entry, SOAK_SPLIT, record[f"{color}_family"]
        )
        for failure in failures:
            counters[classify_construction_failure(failure)] += 1
            findings.append(f"{color}: {failure}")

        if selector_provenance["final_setup_fingerprint"] != record[f"{color}_final_fingerprint"]:
            counters["stranded_sampled_setups"] += 1
            findings.append(
                f"{color}: the drawn setup and the played setup are not the same "
                "arrangement — a sampled setup was stranded"
            )

        if cross_check_accepted_sampler:
            drift = (
                neutral_branch_matches_accepted_sampler(redraw, source.index)
                if redraw.branch == BRANCH_NEUTRAL
                else learned_branch_shares_phase7_decisions(redraw, source.index)
            )
            if drift:
                counters["provenance_mismatches"] += 1
                findings.extend(f"{color}: {finding}" for finding in drift)

    if record["result"] not in SOAK_RESULT_TARGETS:
        counters["outcome_inconsistencies"] += 1
        findings.append(f"unknown result {record['result']!r}")
    elif float(record["red_score"]) != SOAK_RESULT_TARGETS[record["result"]]:
        counters["outcome_inconsistencies"] += 1
        findings.append("red_score contradicts the result token")
    if record["plies"] <= 0 or record["decisions"] <= 0 or not record["terminal_reason"]:
        counters["outcome_inconsistencies"] += 1
        findings.append("plies/decisions/terminal_reason are not a played game's")

    return {"counters": counters, "findings": findings, "ok": not findings}


def _empirical_family_diagnostic(counts: dict, exact_family_mass: dict) -> dict:
    """Empirical vs exact family frequencies for one colour, with the frozen
    sampling-noise expectation `0.5*sqrt(2/pi)*sum(sqrt(p(1-p)/N))`."""
    import numpy as np

    from ..setups.families import FAMILY_IDS

    total = sum(counts.values())
    empirical = np.array(
        [counts.get(family_id, 0) / total for family_id in FAMILY_IDS], dtype=np.float64
    )
    exact = np.array([exact_family_mass[family_id] for family_id in FAMILY_IDS], dtype=np.float64)
    tv = float(0.5 * np.abs(empirical - exact).sum())
    noise = float(0.5 * math.sqrt(2.0 / math.pi) * np.sqrt(exact * (1 - exact) / total).sum())
    return {
        "draws": total,
        "empirical_family_frequencies": {
            family_id: float(value) for family_id, value in zip(FAMILY_IDS, empirical)
        },
        "exact_family_probabilities": {
            family_id: float(value) for family_id, value in zip(FAMILY_IDS, exact)
        },
        "family_total_variation": tv,
        "sampling_noise_expectation": noise,
        "tv_to_noise_ratio": tv / noise if noise else float("inf"),
        "rule": (
            "material disagreement beyond sampling expectations is an implementation "
            "problem; empirical total variation remains a report-only diagnostic"
        ),
    }


def audit_soak_records(
    records,
    *,
    source: "LearnedSetupSource | None" = None,
    scheduled: "set[str] | None" = None,
    cross_check_accepted_sampler: bool = True,
) -> dict:
    """Run the per-game audit over an iterable of records and aggregate."""
    source = build_soak_source() if source is None else source
    counters = {name: 0 for name in SOAK_AUDIT_COUNTERS}
    findings: list = []
    games = 0
    seen_ids: set = set()
    duplicate_ids: list = []

    family_counts = {"red": {}, "blue": {}}
    base_ids = {"red": set(), "blue": set()}
    branch_counts = {"red": {}, "blue": {}}
    reflection = {"red": 0, "blue": 0}
    perturbation = {"red": 0, "blue": 0}
    swap_counts: dict = {}
    fingerprints: set = set()
    fingerprint_list: list = []
    result_counts = {token: 0 for token in SOAK_RESULT_TARGETS}
    terminal_reasons: dict = {}
    plies: list = []
    policy_identities: set = set()
    model_digests: set = set()
    checkpoint_digests: set = set()

    for record in records:
        games += 1
        game_id = record["game_id"]
        if game_id in seen_ids:
            duplicate_ids.append(game_id)
        seen_ids.add(game_id)
        verdict = verify_soak_game(
            record,
            source,
            scheduled=scheduled,
            cross_check_accepted_sampler=cross_check_accepted_sampler,
        )
        for name, value in verdict["counters"].items():
            counters[name] += value
        if verdict["findings"]:
            findings.extend(f"{game_id}: {finding}" for finding in verdict["findings"])

        for color in COLORS:
            family = record[f"{color}_family"]
            family_counts[color][family] = family_counts[color].get(family, 0) + 1
            base_ids[color].add(record[f"{color}_base_setup_id"])
            provenance = record[f"{color}_setup_provenance"]
            branch = record[f"{color}_selector_provenance"]["branch"]
            branch_counts[color][branch] = branch_counts[color].get(branch, 0) + 1
            if provenance["reflection_applied"]:
                reflection[color] += 1
            if provenance["perturbation_requested"]:
                perturbation[color] += 1
                swap = provenance["perturbation_swap_count"]
                swap_counts[str(swap)] = swap_counts.get(str(swap), 0) + 1
            fingerprints.add(record[f"{color}_final_fingerprint"])
            fingerprint_list.append(record[f"{color}_final_fingerprint"])
        result_counts[record["result"]] = result_counts.get(record["result"], 0) + 1
        terminal_reasons[record["terminal_reason"]] = (
            terminal_reasons.get(record["terminal_reason"], 0) + 1
        )
        plies.append(int(record["plies"]))
        policy_identities.add(record["move_policy_identity"])
        model_digests.add(record["move_model_state_digest"])
        checkpoint_digests.add(record["move_checkpoint_sha256"])

    return {
        "games_audited": games,
        "counters": counters,
        "findings": findings[:64],
        "finding_count": len(findings),
        "duplicate_ids_in_audit": sorted(set(duplicate_ids))[:16],
        "family_counts": family_counts,
        "base_ids": {color: sorted(values) for color, values in base_ids.items()},
        "branch_counts": branch_counts,
        "reflection_counts": reflection,
        "perturbation_counts": perturbation,
        "swap_counts": swap_counts,
        "distinct_final_fingerprints": len(fingerprints),
        "final_fingerprints": fingerprint_list,
        "result_counts": result_counts,
        "terminal_reasons": terminal_reasons,
        "plies": plies,
        "move_policy_identities": sorted(policy_identities),
        "move_model_state_digests": sorted(model_digests),
        "move_checkpoint_sha256": sorted(checkpoint_digests),
    }


def _merge_counts(target: dict, source: dict) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def merge_soak_audits(parts: list) -> dict:
    """Combine per-shard audit outputs into one aggregate."""
    merged = None
    for part in parts:
        if merged is None:
            merged = {
                "games_audited": 0,
                "counters": {name: 0 for name in SOAK_AUDIT_COUNTERS},
                "findings": [],
                "finding_count": 0,
                "duplicate_ids_in_audit": [],
                "family_counts": {"red": {}, "blue": {}},
                "base_ids": {"red": set(), "blue": set()},
                "branch_counts": {"red": {}, "blue": {}},
                "reflection_counts": {"red": 0, "blue": 0},
                "perturbation_counts": {"red": 0, "blue": 0},
                "swap_counts": {},
                "distinct_final_fingerprints": 0,
                "final_fingerprints": [],
                "result_counts": {token: 0 for token in SOAK_RESULT_TARGETS},
                "terminal_reasons": {},
                "plies": [],
                "move_policy_identities": set(),
                "move_model_state_digests": set(),
                "move_checkpoint_sha256": set(),
            }
        merged["games_audited"] += part["games_audited"]
        _merge_counts(merged["counters"], part["counters"])
        merged["findings"].extend(part["findings"])
        merged["finding_count"] += part["finding_count"]
        merged["duplicate_ids_in_audit"].extend(part["duplicate_ids_in_audit"])
        for color in COLORS:
            _merge_counts(merged["family_counts"][color], part["family_counts"][color])
            merged["base_ids"][color].update(part["base_ids"][color])
            _merge_counts(merged["branch_counts"][color], part["branch_counts"][color])
            merged["reflection_counts"][color] += part["reflection_counts"][color]
            merged["perturbation_counts"][color] += part["perturbation_counts"][color]
        _merge_counts(merged["swap_counts"], part["swap_counts"])
        merged["final_fingerprints"].extend(part["final_fingerprints"])
        _merge_counts(merged["result_counts"], part["result_counts"])
        _merge_counts(merged["terminal_reasons"], part["terminal_reasons"])
        merged["plies"].extend(part["plies"])
        merged["move_policy_identities"].update(part["move_policy_identities"])
        merged["move_model_state_digests"].update(part["move_model_state_digests"])
        merged["move_checkpoint_sha256"].update(part["move_checkpoint_sha256"])
    if merged is None:
        raise Phase10SoakError("nothing to merge: no audit shards")
    merged["findings"] = merged["findings"][:64]
    merged["duplicate_ids_in_audit"] = sorted(set(merged["duplicate_ids_in_audit"]))[:16]
    merged["distinct_final_fingerprints"] = len(set(merged["final_fingerprints"]))
    merged["base_ids"] = {
        color: sorted(values) for color, values in merged["base_ids"].items()
    }
    merged["move_policy_identities"] = sorted(merged["move_policy_identities"])
    merged["move_model_state_digests"] = sorted(merged["move_model_state_digests"])
    merged["move_checkpoint_sha256"] = sorted(merged["move_checkpoint_sha256"])
    return merged


def soak_diagnostics(audit: dict, *, isolation: "frozenset[str] | None" = None) -> dict:
    """The report-only diagnostic block computed from a finished audit.

    Family entropy and effective families use empirical frequencies from
    actual games; the hard diversity acceptance remains Agent 4's exact
    distribution metrics, and soak frequencies cannot override them.
    """
    import numpy as np

    from ..setups.families import FAMILY_IDS

    source = build_soak_source()
    diagnostics: dict = {"per_color": {}}
    for color in COLORS:
        counts = audit["family_counts"][color]
        total = sum(counts.values())
        frequencies = np.array(
            [counts.get(family_id, 0) / total for family_id in FAMILY_IDS],
            dtype=np.float64,
        )
        positive = frequencies[frequencies > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        distribution = source.distribution(color, SOAK_SPLIT)
        exact_mass = {
            family_id: float(value)
            for family_id, value in zip(FAMILY_IDS, distribution.family_probabilities())
        }
        sides = audit["branch_counts"][color]
        branch_total = sum(sides.values())
        diagnostics["per_color"][color] = {
            "draws": total,
            "families_seen": int((frequencies > 0.0).sum()),
            "family_frequencies": {
                family_id: float(value)
                for family_id, value in zip(FAMILY_IDS, frequencies)
            },
            "family_entropy_nats": entropy,
            "normalized_family_entropy": entropy / math.log(len(FAMILY_IDS)),
            "effective_families": math.exp(entropy),
            "distinct_bases": len(audit["base_ids"][color]),
            "branch_counts": dict(sides),
            "neutral_branch_rate": sides.get(BRANCH_NEUTRAL, 0) / branch_total,
            "reflection_rate": audit["reflection_counts"][color] / total,
            "perturbation_rate": audit["perturbation_counts"][color] / total,
            "empirical_vs_exact": _empirical_family_diagnostic(counts, exact_mass),
        }
    swap_total = sum(audit["swap_counts"].values())
    diagnostics["swap_count_distribution"] = {
        "counts": dict(sorted(audit["swap_counts"].items())),
        "total_perturbed_sides": swap_total,
        "rates": {
            key: value / swap_total for key, value in sorted(audit["swap_counts"].items())
        }
        if swap_total
        else {},
    }
    diagnostics["unique_final_setups"] = audit["distinct_final_fingerprints"]
    diagnostics["total_sides"] = 2 * audit["games_audited"]
    if isolation is not None:
        landings = [
            fingerprint
            for fingerprint in audit["final_fingerprints"]
            if fingerprint in isolation
        ]
        diagnostics["phase9_fingerprint_landings"] = {
            "isolation_set_size": len(isolation),
            "sides_checked": len(audit["final_fingerprints"]),
            "landings": len(landings),
            "landing_rate": len(landings) / max(len(audit["final_fingerprints"]), 1),
            "role": (
                "report-only diagnostic; never used for rejection sampling or "
                "selection — rejecting at draw time would distort the frozen "
                "mixed distribution"
            ),
        }
    plies = audit["plies"]
    diagnostics["outcomes"] = {
        "result_counts": dict(audit["result_counts"]),
        "result_rates": {
            token: count / max(audit["games_audited"], 1)
            for token, count in audit["result_counts"].items()
        },
        "terminal_reasons": dict(sorted(audit["terminal_reasons"].items())),
        "ply_summary": {
            "min": min(plies) if plies else 0,
            "max": max(plies) if plies else 0,
            "mean": sum(plies) / len(plies) if plies else 0.0,
            "total": sum(plies),
        },
        "role": "report-only; may not change candidate, coefficients, T, mixture or thresholds",
    }
    return diagnostics


# ---------------------------------------------------------------------------
# Replay probe: end-to-end reproducibility under a different topology
# ---------------------------------------------------------------------------


def replay_soak_slice(
    game_ids,
    *,
    export: dict,
    device: str = "cpu",
    torch_threads: int = 1,
) -> dict:
    """Replay soak games from their identity alone, returning primitives."""
    import torch

    from .phase10_collector import load_corpus_owner, owner_state_digest

    torch.set_num_threads(int(torch_threads))
    owner = load_corpus_owner(export["export_path"], device=device, name="phase10_soak_replay")
    if owner_state_digest(owner) != export["model_state_digest"]:
        raise Phase10SoakError("the soak replay owner loaded different weights")
    from ..evaluation.neural_worker import LocalInferenceChannel, RemoteNeuralPolicy
    from ..model.policy_adapter import DECISION_MODE_GREEDY

    policy = RemoteNeuralPolicy(
        soak_policy_ref(), LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY
    )
    source = build_soak_source()
    observed: dict = {}
    try:
        for game_id in game_ids:
            result, sides = play_soak_game(game_id, policy, source)
            observed[game_id] = {
                "result": soak_result_token(result),
                "plies": int(result.plies),
                "terminal_reason": result.terminal_reason,
                "red_final_fingerprint": sides["red"]["draw"].final_setup_fingerprint,
                "blue_final_fingerprint": sides["blue"]["draw"].final_setup_fingerprint,
            }
    finally:
        owner.close()
    return observed


def _soak_replay_worker_main(payload: dict, queue) -> None:
    report = {"worker_id": payload["worker_id"], "status": "ok"}
    try:
        report["observed"] = replay_soak_slice(
            payload["game_ids"],
            export=payload["export"],
            device=payload["device"],
            torch_threads=payload["torch_threads"],
        )
    except BaseException as error:  # noqa: BLE001 -- reported to the parent verbatim
        import traceback

        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    queue.put(report)


def replay_soak_probe(
    root: "str | Path",
    *,
    export: dict,
    sample: int = 256,
    worker_count: int = 5,
    device: str = "cpu",
    torch_threads: int = 1,
) -> dict:
    """Replay a deterministic stride sample under a different worker topology
    and require byte-identical primitives against the committed records."""
    import multiprocessing

    from .phase10_collector import partition

    reader = SoakReader(root)
    total = len(reader)
    if total == 0:
        raise Phase10SoakError("nothing to replay: the soak store is empty")
    count = min(int(sample), total)
    stride = max(total // count, 1)
    chosen = [reader.game_ids[position] for position in range(0, total, stride)][:count]
    stored = {game_id: reader.record(game_id) for game_id in chosen}

    started = time.perf_counter()
    if worker_count > 1:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        buckets = [bucket for bucket in partition(chosen, worker_count) if bucket]
        processes = []
        for worker_id, bucket in enumerate(buckets):
            payload = {
                "worker_id": worker_id,
                "game_ids": bucket,
                "export": export,
                "device": device,
                "torch_threads": torch_threads,
            }
            process = context.Process(
                target=_soak_replay_worker_main, args=(payload, queue), daemon=False
            )
            process.start()
            processes.append(process)
        observed: dict = {}
        reports = [queue.get() for _ in processes]
        for process in processes:
            process.join()
        failed = [report for report in reports if report["status"] != "ok"]
        if failed:
            raise Phase10SoakError(
                f"{len(failed)} replay worker(s) failed; the first says: "
                f"{failed[0].get('error')}\n{failed[0].get('traceback', '')}"
            )
        for report in reports:
            observed.update(report["observed"])
    else:
        observed = replay_soak_slice(
            chosen, export=export, device=device, torch_threads=torch_threads
        )
    elapsed = time.perf_counter() - started

    mismatches: list = []
    for game_id in chosen:
        record = stored[game_id]
        replayed = observed[game_id]
        for field in ("result", "plies", "terminal_reason"):
            if replayed[field] != record[field]:
                mismatches.append(
                    f"{game_id}: {field} {replayed[field]!r} != stored {record[field]!r}"
                )
        for color in COLORS:
            if replayed[f"{color}_final_fingerprint"] != record[f"{color}_final_fingerprint"]:
                mismatches.append(f"{game_id}: {color} fingerprint")
    return {
        "replayed_games": count,
        "store_games": total,
        "stride": stride,
        "worker_count": worker_count,
        "wall_clock_seconds": elapsed,
        "games_per_second": count / elapsed if elapsed else 0.0,
        "mismatches": mismatches[:32],
        "all_identical": not mismatches,
    }


__all__ = [
    "DEFAULT_PHASE10_SOAK_ROOT",
    "MAX_SOAK_ORDINAL_FORMAT",
    "PHASE10_SOAK_ROOT_ENV",
    "PHASE10_SOAK_ROOT_POINTER",
    "SELECTED_CANDIDATE_ID",
    "SELECTED_CONFIG_SHA256",
    "SOAK_AUDIT_COUNTERS",
    "SOAK_COMMIT_VERSION",
    "SOAK_DOMAIN_MATCH",
    "SOAK_DOMAIN_ROOTS",
    "SOAK_DOMAIN_SELECTOR",
    "SOAK_ENVELOPE_FIELDS",
    "SOAK_ID_VERSION",
    "SOAK_MOVE_POLICY_ID",
    "SOAK_RECORD_FIELDS",
    "SOAK_RECORD_VERSION",
    "SOAK_RESULT_TARGETS",
    "SOAK_SPLIT",
    "SOAK_TOTAL_GAMES",
    "SOAK_VERSION",
    "Phase10SoakError",
    "SoakReader",
    "SoakWriter",
    "audit_soak_records",
    "build_soak_record",
    "build_soak_source",
    "canonical_json",
    "collect_soak",
    "collect_soak_slice",
    "default_soak_root",
    "derive_soak_seed",
    "describe_soak_root",
    "draw_soak_sides",
    "file_set_name",
    "hidden_input_positive_control",
    "merge_soak_audits",
    "next_soak_segment",
    "parse_soak_game_id",
    "play_soak_game",
    "probe_volume_health",
    "read_soak_journal",
    "read_soak_seal",
    "read_soak_state",
    "reconcile_soak",
    "record_payload_sha256",
    "replay_soak_probe",
    "replay_soak_slice",
    "require_soak_collecting",
    "seal_soak",
    "selected_candidate",
    "selected_selector_identity",
    "soak_committed_count",
    "soak_content_digest",
    "soak_diagnostics",
    "soak_game_id",
    "soak_game_ids",
    "soak_identity_block",
    "soak_match_seed",
    "soak_match_spec",
    "soak_policy_ref",
    "soak_result_token",
    "soak_seed_collision_audit",
    "soak_selector_seed",
    "validate_soak_record",
    "verify_soak_game",
]
