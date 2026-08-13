"""Phase 7 Agent 5: the pluggable setup source for production collection.

Specification sources:

- `05_AGENT_5_PIPELINE_INTEGRATION.md` (pluggable setup source, setup-pair
  sampling, split behaviour, provenance sidecar, observer safety)
- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (split permanence, provenance
  design, observer-safety boundary, trajectory/persistence rules)

What this is
------------
One narrow interface between the frozen Phase 6 collection pipeline and the
accepted Phase 7 setup sampler. `BatchSimulator` asks a setup source for the
two setups of every game it creates; everything else about the pipeline --
the coordinator, the workers, the bulk-synchronous cycle, the action frames,
`trajectory_v1`, the shard container -- is untouched.

Two sources exist:

```text
UniformRandomSetupSource   the frozen Phase 6 behaviour, exactly
LibrarySetupSource         setup_library_v1 + setup_sampler_v1
```

The default is `UniformRandomSetupSource`, so a caller that does not ask for
Phase 7 setups gets the accepted Phase 6 games byte for byte.

Setup-pair identity
-------------------
A setup pair belongs to a *logical game*, not to a worker, a slot position or
a scheduling order. The identity is the triple the Phase 3 batch layer already
made sufficient to rebuild any game in isolation:

```text
(root_seed, environment_id, generation)
    -> red side seed
    -> blue side seed
```

`environment_id` is the global slot index regardless of how the pool is
partitioned, and `generation` counts that slot's games, so changing the worker
count, the slot-to-worker mapping, the scheduling order or a recycle boundary
cannot change which setups a logical game receives. Each side draws from its
own domain-separated stream, so the two sides are independent and neither
consumes a shared cursor in worker-arrival order.

Splits
------
`training_setup_source()` is the production training entry point and is
hard-wired to `split="train"`. A validation or test source cannot be built by
accident: `purpose` must be `evaluation_audit` *and* an explicit written
justification must be supplied. There is no code path in which a routine
training run reaches a validation or test base.

Provenance
----------
Sampling produces a per-player provenance record. It never enters
`trajectory_v1`, `observation_v2_1_127ch`, the shared buffers or any model
input: it is written to a per-worker JSONL sidecar next to the shard files and
keyed by `game_id`. The setup stored inside the trajectory remains the replay
authority; provenance is diagnostic/training metadata that rebuilds the sample
through :func:`stratego.setups.rebuild_from_provenance`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..engine.constants import PLAYERS, PLAYER_NAMES
from ..engine.random_play import make_random_setups
from ..engine.setup import serialize_setup
from ..setups.contracts import LIBRARY_JSONL_PATH, SETUP_LIBRARY_VERSION, SPLITS
from ..setups.identity import derive_stream_seed
from ..setups.sampler import (
    DEFAULT_PROFILE,
    REQUIRED_PROVENANCE_FIELDS,
    SAMPLER_VERSION,
    load_library_index,
    rebuild_from_provenance,
    sample_setup,
    sampler_profile,
)

#: Version identifier of this integration. A change to the side-seed
#: derivation, the provenance schema or the split rule is a new identifier.
SETUP_SOURCE_VERSION = "setup_source_v1"

#: The identifier `UniformRandomSetupSource` reports. It is deliberately the
#: same string `trajectory_v1` already stamps into `setup_family` for uniform
#: random placement, because it names the same generator.
UNIFORM_RANDOM_SOURCE = "batch_random_uniform_v1"

#: The only split a production training run may sample from.
TRAINING_SPLIT = "train"

#: The two accepted purposes. `training` is the default production path and is
#: locked to `TRAINING_SPLIT`; `evaluation_audit` is the explicit request a
#: validation/test source requires.
TRAINING_PURPOSE = "training"
AUDIT_PURPOSE = "evaluation_audit"
PURPOSES = (TRAINING_PURPOSE, AUDIT_PURPOSE)

#: Sidecar file naming. One file per worker, alongside that worker's shards.
PROVENANCE_SUFFIX = "_setup_provenance.jsonl"
PROVENANCE_SCHEMA_VERSION = "setup_provenance_v1"

#: Per-game keys of a sidecar record, in emission order.
PROVENANCE_RECORD_FIELDS = (
    "provenance_schema_version",
    "setup_source_version",
    "setup_library_version",
    "sampler_version",
    "sampler_profile",
    "split",
    "run_id",
    "worker_id",
    "game_id",
    "environment_id",
    "generation",
    "root_seed",
    "red",
    "blue",
)

#: Per-player keys this integration adds on top of the sampler's own 27-field
#: provenance record. `engine_setup` is the exact 40-tuple the frozen engine
#: received, so a persisted trajectory can be checked against it directly
#: rather than through a re-derived orientation.
PROVENANCE_PLAYER_EXTRA_FIELDS = (
    "player",
    "player_name",
    "side_seed",
    "engine_setup",
)

#: The minimum per-player provenance the Agent 5 assignment names. Every one of
#: these is already part of the frozen `setup_sampler_v1` provenance schema, so
#: the sidecar stores that record verbatim rather than a lossy projection.
REQUIRED_PLAYER_PROVENANCE_FIELDS = (
    "setup_library_version",
    "sampler_version",
    "primary_family_id",
    "base_setup_id",
    "split",
    "reflection_applied",
    "perturbation_applied",
    "perturbation_seed",
    "final_setup_fingerprint",
)


class SetupSourceError(RuntimeError):
    """A setup source was configured or used outside its contract."""


# ---------------------------------------------------------------------------
# What a source returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupAssignment:
    """The two engine-order setups of one game, plus optional provenance.

    `red_setup` and `blue_setup` are what `create_game` receives: the frozen
    engine's own 40-entry setup order for each player. `provenance` is `None`
    for the uniform random source, which has nothing to trace.
    """

    red_setup: tuple[int, ...]
    blue_setup: tuple[int, ...]
    provenance: dict | None = None

    def setup_for(self, player: int) -> tuple[int, ...]:
        if player == PLAYERS[0]:
            return self.red_setup
        if player == PLAYERS[1]:
            return self.blue_setup
        raise SetupSourceError(f"unknown player: {player!r}")


# ---------------------------------------------------------------------------
# The frozen Phase 6 default
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformRandomSetupSource:
    """The accepted Phase 6 setup generator, wrapped in the source interface.

    Exists so the injection point has a default that is provably the old
    behaviour: it calls `make_random_setups(slot_seed)` with the same slot seed
    the batch layer already derived, and produces no provenance.
    """

    source_id: str = UNIFORM_RANDOM_SOURCE

    @property
    def setup_family(self) -> str:
        """The `trajectory_v1` `setup_family` label for games from this source."""
        return UNIFORM_RANDOM_SOURCE

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "setup_source_version": SETUP_SOURCE_VERSION,
            "kind": "uniform_random",
            "produces_provenance": False,
        }

    def assign(
        self,
        *,
        root_seed: int,
        environment_id: int,
        generation: int,
        slot_seed: int,
        game_id: str = "",
    ) -> SetupAssignment:
        red_setup, blue_setup = make_random_setups(slot_seed)
        return SetupAssignment(red_setup=red_setup, blue_setup=blue_setup)


# ---------------------------------------------------------------------------
# The Phase 7 library source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibrarySetupSource:
    """`setup_library_v1` + `setup_sampler_v1` as a collection setup source.

    A small frozen value object holding configuration only, so it crosses a
    `spawn` boundary cheaply and every worker rebuilds the identical source.
    The 8,000-entry library itself is loaded lazily per process through the
    sampler's own cached, read-only index.

    Splits are enforced in `__post_init__` rather than at sampling time: a
    validation/test source cannot be constructed by a training caller at all,
    so there is no path from routine collection to a held-out base.
    """

    split: str = TRAINING_SPLIT
    profile: str = DEFAULT_PROFILE.name
    purpose: str = TRAINING_PURPOSE
    access_justification: str = ""
    library_path: str = LIBRARY_JSONL_PATH
    source_version: str = SETUP_SOURCE_VERSION

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise SetupSourceError(f"unknown split: {self.split!r}")
        if self.purpose not in PURPOSES:
            raise SetupSourceError(
                f"unknown purpose {self.purpose!r}; expected one of {list(PURPOSES)}"
            )
        if self.purpose == TRAINING_PURPOSE and self.split != TRAINING_SPLIT:
            raise SetupSourceError(
                f"the production training setup source is locked to "
                f"split={TRAINING_SPLIT!r}; {self.split!r} is a held-out split and "
                f"needs purpose={AUDIT_PURPOSE!r} with an explicit "
                f"access_justification"
            )
        if self.split != TRAINING_SPLIT and not self.access_justification.strip():
            raise SetupSourceError(
                f"sampling the {self.split!r} split requires a non-empty "
                f"access_justification naming the evaluation/audit request"
            )
        # Rejects an unknown profile name at construction, in the parent
        # process, instead of inside every spawned worker.
        try:
            sampler_profile(self.profile)
        except Exception as error:  # noqa: BLE001 - re-raised at this boundary
            raise SetupSourceError(str(error)) from error

    # -- description -------------------------------------------------------

    @property
    def setup_family(self) -> str:
        """The `trajectory_v1` `setup_family` label for games from this source.

        A string field `trajectory_v1` already carries; the value names the
        generator, exactly as `batch_random_uniform_v1` does. Per-game family
        identity lives in the sidecar, not here.
        """
        return f"{SETUP_LIBRARY_VERSION}_{SAMPLER_VERSION}_{self.split}"

    def describe(self) -> dict:
        return {
            "source_id": self.setup_family,
            "setup_source_version": self.source_version,
            "kind": "setup_library",
            "setup_library_version": SETUP_LIBRARY_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "sampler_profile": self.profile,
            "split": self.split,
            "purpose": self.purpose,
            "access_justification": self.access_justification,
            "library_path": self.library_path,
            "produces_provenance": True,
            "side_seed_derivation": (
                "derive_stream_seed('setup_source_v1:side', split, profile, "
                "root_seed, environment_id, generation, player_name); one "
                "domain-separated stream per (logical game, side)"
            ),
        }

    def library_digest(self) -> str:
        return load_library_index(self.library_path).content_digest

    # -- sampling ----------------------------------------------------------

    def side_seed(
        self, *, root_seed: int, environment_id: int, generation: int, player: int
    ) -> int:
        """The draw seed of one side of one logical game.

        Depends only on the run's root seed, the logical game identity and the
        side. Worker count, slot-to-worker mapping, arrival order and recycle
        boundaries are all absent from the derivation, which is what makes the
        assignment schedule-independent.
        """
        if player not in PLAYERS:
            raise SetupSourceError(f"unknown player: {player!r}")
        return derive_stream_seed(
            f"{self.source_version}:side",
            self.split,
            self.profile,
            int(root_seed),
            int(environment_id),
            int(generation),
            PLAYER_NAMES[player],
        )

    def sample_for_player(
        self, *, root_seed: int, environment_id: int, generation: int, player: int
    ):
        """`(SampledSetup, side_seed)` for one side of one logical game."""
        seed = self.side_seed(
            root_seed=root_seed,
            environment_id=environment_id,
            generation=generation,
            player=player,
        )
        sampled = sample_setup(
            self.split,
            seed,
            profile=self.profile,
            index=load_library_index(self.library_path),
        )
        return sampled, seed

    def assign(
        self,
        *,
        root_seed: int,
        environment_id: int,
        generation: int,
        slot_seed: int = 0,
        game_id: str = "",
    ) -> SetupAssignment:
        """Both setups of one logical game, with provenance.

        The two sides are sampled independently from the requested split. The
        engine-order tuples come from the sampler's own orientation helper, so
        this module never re-derives the board convention.
        """
        sides: dict[int, dict] = {}
        engine_setups: dict[int, tuple[int, ...]] = {}
        for player in PLAYERS:
            sampled, seed = self.sample_for_player(
                root_seed=root_seed,
                environment_id=environment_id,
                generation=generation,
                player=player,
            )
            engine_setup = sampled.oriented(player)
            engine_setups[player] = engine_setup
            sides[player] = {
                "player": int(player),
                "player_name": PLAYER_NAMES[player],
                "side_seed": int(seed),
                "engine_setup": serialize_setup(engine_setup),
                **sampled.provenance,
            }

        provenance = {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "setup_source_version": self.source_version,
            "setup_library_version": SETUP_LIBRARY_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "sampler_profile": self.profile,
            "split": self.split,
            "run_id": "",
            "worker_id": -1,
            "game_id": game_id,
            "environment_id": int(environment_id),
            "generation": int(generation),
            "root_seed": int(root_seed),
            "red": sides[PLAYERS[0]],
            "blue": sides[PLAYERS[1]],
        }
        return SetupAssignment(
            red_setup=engine_setups[PLAYERS[0]],
            blue_setup=engine_setups[PLAYERS[1]],
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Construction entry points
# ---------------------------------------------------------------------------


def training_setup_source(profile: str = DEFAULT_PROFILE.name) -> LibrarySetupSource:
    """The production training setup source. Always the train split.

    The split is not a parameter. A caller that wants a held-out split has to
    go through :func:`audit_setup_source` and say why, which is what keeps
    validation and test bases out of routine collection.
    """
    return LibrarySetupSource(
        split=TRAINING_SPLIT, profile=profile, purpose=TRAINING_PURPOSE
    )


def audit_setup_source(
    split: str,
    justification: str,
    profile: str = DEFAULT_PROFILE.name,
) -> LibrarySetupSource:
    """An explicitly requested evaluation/audit setup source.

    The only way to reach `validation` or `test`. `justification` is recorded
    in every provenance record the source produces, so a held-out sample can
    always be traced back to the request that authorised it.
    """
    if not str(justification).strip():
        raise SetupSourceError(
            "an evaluation/audit setup source requires an explicit justification"
        )
    return LibrarySetupSource(
        split=split,
        profile=profile,
        purpose=AUDIT_PURPOSE,
        access_justification=str(justification),
    )


def default_setup_source() -> UniformRandomSetupSource:
    """The frozen Phase 6 default, returned explicitly for callers that want it."""
    return UniformRandomSetupSource()


def describe_setup_source(source) -> dict:
    if source is None:
        return UniformRandomSetupSource().describe()
    return source.describe()


# ---------------------------------------------------------------------------
# The provenance sidecar
# ---------------------------------------------------------------------------


def provenance_path(directory, *, run_id: str, worker_id: int) -> Path:
    return Path(directory) / f"{run_id}_w{int(worker_id):02d}{PROVENANCE_SUFFIX}"


def validate_provenance_record(record: dict) -> list[str]:
    """Every schema violation in one sidecar record.

    Checks the record's own shape and both player sub-records, including the
    minimum per-player fields the assignment names and the full frozen
    `setup_sampler_v1` required set. Returns an empty list for a good record.
    """
    problems: list[str] = []
    for key in PROVENANCE_RECORD_FIELDS:
        if key not in record:
            problems.append(f"missing record field {key!r}")
    for side in ("red", "blue"):
        player = record.get(side)
        if not isinstance(player, dict):
            problems.append(f"{side}: player provenance is missing or not an object")
            continue
        for key in PROVENANCE_PLAYER_EXTRA_FIELDS:
            if key not in player:
                problems.append(f"{side}: missing integration field {key!r}")
        for key in REQUIRED_PLAYER_PROVENANCE_FIELDS:
            if key not in player:
                problems.append(f"{side}: missing required provenance field {key!r}")
        for key in REQUIRED_PROVENANCE_FIELDS:
            if key not in player:
                problems.append(f"{side}: missing sampler provenance field {key!r}")
        if player.get("split") != record.get("split"):
            problems.append(
                f"{side}: player split {player.get('split')!r} disagrees with the "
                f"record split {record.get('split')!r}"
            )
    return problems


class SetupProvenanceWriter:
    """Append-only per-worker provenance sidecar.

    One instance belongs to one worker process, next to that worker's shard
    writer, and is written synchronously as each game is sealed. Deliberately
    a separate file rather than a `trajectory_v1` change: the record format,
    its bytes and its decoder are untouched, and a Phase 6 shard remains
    readable by everything that could read it before.
    """

    def __init__(self, directory, *, run_id: str, worker_id: int) -> None:
        self.directory = Path(directory)
        self.run_id = str(run_id)
        self.worker_id = int(worker_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = provenance_path(
            self.directory, run_id=self.run_id, worker_id=self.worker_id
        )
        self.records_written = 0
        self.bytes_written = 0
        self.write_errors = 0
        self.write_seconds = 0.0
        self.error_details: list[str] = []
        self._handle = open(self.path, "a", encoding="utf-8")

    def write(self, provenance: dict, *, game_id: str) -> int:
        """Append one game's provenance. Returns the bytes written."""
        if self._handle is None:
            raise SetupSourceError("this provenance writer is closed")
        record = dict(provenance)
        record["run_id"] = self.run_id
        record["worker_id"] = self.worker_id
        record["game_id"] = game_id
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        started = time.perf_counter()
        try:
            self._handle.write(line)
            self._handle.flush()
        except OSError as error:  # pragma: no cover - filesystem failure
            self.write_errors += 1
            self.error_details.append(f"{type(error).__name__}: {error}")
            self.write_seconds += time.perf_counter() - started
            return 0
        self.write_seconds += time.perf_counter() - started
        written = len(line.encode())
        self.records_written += 1
        self.bytes_written += written
        return written

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError:  # pragma: no cover - shutdown must not mask a fault
            pass
        finally:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def stats(self) -> dict:
        return {
            "provenance_path": str(self.path),
            "records_written": self.records_written,
            "bytes_written": self.bytes_written,
            "write_errors": self.write_errors,
            "write_seconds": self.write_seconds,
            "error_details": list(self.error_details),
        }


def provenance_paths(directory) -> list[Path]:
    return sorted(Path(directory).glob(f"*{PROVENANCE_SUFFIX}"))


def iter_provenance_records(directory):
    """Every sidecar record under `directory`, file order then line order."""
    for path in provenance_paths(directory):
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError as error:
                    raise SetupSourceError(
                        f"{path}:{line_number}: malformed provenance line: {error}"
                    ) from error


def read_provenance_index(directory) -> dict:
    """Sidecar records keyed by `game_id`.

    Raises on a duplicate `game_id`: two provenance records for one game would
    make the mapping ambiguous, and the batch layer's identity triple already
    guarantees they cannot legitimately collide.
    """
    index: dict[str, dict] = {}
    for record in iter_provenance_records(directory):
        game_id = str(record.get("game_id", ""))
        if not game_id:
            raise SetupSourceError("a provenance record carries no game_id")
        if game_id in index:
            raise SetupSourceError(f"duplicate provenance record for game {game_id!r}")
        index[game_id] = record
    return index


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_provenance_against_setups(
    record: dict,
    *,
    red_setup: "tuple[int, ...]",
    blue_setup: "tuple[int, ...]",
    library_path: str = LIBRARY_JSONL_PATH,
) -> list[str]:
    """Every mismatch between a provenance record and a pair of true setups.

    The strong form of the check the assignment asks for. For each player it
    rebuilds the sampled descendant from provenance alone through the frozen
    `setup_sampler_v1` rebuild path, re-orients it for that player, and
    requires:

    ```text
    rebuilt engine setup   == the setup actually stored in the trajectory
    rebuilt fingerprint    == the recorded final_setup_fingerprint
    recorded engine_setup  == the setup actually stored in the trajectory
    ```

    so a provenance record cannot agree with the trajectory by naming it while
    disagreeing about which library entry produced it.
    """
    problems = list(validate_provenance_record(record))
    if problems:
        return problems

    index = load_library_index(library_path)
    stored = {PLAYERS[0]: tuple(red_setup), PLAYERS[1]: tuple(blue_setup)}
    for side, player in (("red", PLAYERS[0]), ("blue", PLAYERS[1])):
        player_record = record[side]
        if int(player_record["player"]) != int(player):
            problems.append(
                f"{side}: provenance names player {player_record['player']}, "
                f"expected {player}"
            )
        try:
            rebuilt = rebuild_from_provenance(player_record, index=index)
        except Exception as error:  # noqa: BLE001 - a failed rebuild is a finding
            problems.append(f"{side}: rebuild failed: {type(error).__name__}: {error}")
            continue
        rebuilt_engine = rebuilt.oriented(player)
        if rebuilt_engine != stored[player]:
            problems.append(
                f"{side}: the setup rebuilt from provenance is not the setup stored "
                f"in the trajectory"
            )
        if serialize_setup(stored[player]) != player_record["engine_setup"]:
            problems.append(
                f"{side}: recorded engine_setup does not match the stored setup"
            )
        if (
            rebuilt.provenance["final_setup_fingerprint"]
            != player_record["final_setup_fingerprint"]
        ):
            problems.append(f"{side}: final_setup_fingerprint does not rebuild")
        if rebuilt.split != player_record["split"]:
            problems.append(f"{side}: rebuilt split disagrees with the record")
        if rebuilt.family_id != player_record["primary_family_id"]:
            problems.append(f"{side}: rebuilt family disagrees with the record")
        if rebuilt.base_setup_id != player_record["base_setup_id"]:
            problems.append(f"{side}: rebuilt base id disagrees with the record")
    return problems


def verify_provenance_split(record: dict, expected_split: str) -> list[str]:
    """Split violations in one provenance record, both sides."""
    problems: list[str] = []
    if record.get("split") != expected_split:
        problems.append(
            f"record split {record.get('split')!r} is not {expected_split!r}"
        )
    for side in ("red", "blue"):
        player = record.get(side) or {}
        if player.get("split") != expected_split:
            problems.append(
                f"{side}: split {player.get('split')!r} is not {expected_split!r}"
            )
    return problems


def family_pair(record: dict) -> tuple[str, str]:
    """The ordered `(red family, blue family)` of one provenance record."""
    return (
        str(record["red"]["primary_family_id"]),
        str(record["blue"]["primary_family_id"]),
    )


__all__ = [
    "AUDIT_PURPOSE",
    "PROVENANCE_PLAYER_EXTRA_FIELDS",
    "PROVENANCE_RECORD_FIELDS",
    "PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_SUFFIX",
    "PURPOSES",
    "REQUIRED_PLAYER_PROVENANCE_FIELDS",
    "SETUP_SOURCE_VERSION",
    "TRAINING_PURPOSE",
    "TRAINING_SPLIT",
    "UNIFORM_RANDOM_SOURCE",
    "LibrarySetupSource",
    "SetupAssignment",
    "SetupProvenanceWriter",
    "SetupSourceError",
    "UniformRandomSetupSource",
    "audit_setup_source",
    "default_setup_source",
    "describe_setup_source",
    "family_pair",
    "iter_provenance_records",
    "provenance_path",
    "provenance_paths",
    "read_provenance_index",
    "training_setup_source",
    "validate_provenance_record",
    "verify_provenance_against_setups",
    "verify_provenance_split",
]
