"""Phase 14: the historical archive and the bounded active pool.

Specification source: the frozen
`historical_archive_and_active_pool.active_pool_algorithm phase14_active_pool_v1`,
via `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 7 and 8.

Two objects, one of them derived
--------------------------------
The **archive** is the durable, append-only, ordered list of every 2-hour
snapshot the run has written. Nothing is ever pruned from it, and its order is
its identity.

The **active pool** is a pure function `f(k)` of that ordering: 2 permanent
anchors plus up to 14 snapshots, chosen by age band alone. No tournament, no
result, no strength estimate reaches this module — the only inputs are how many
snapshots exist and in what order they arrived. That is what makes the
resume check in :func:`assert_pool_matches` meaningful: after a crash the pool
is recomputed from the archive and must equal the pool the checkpoint recorded,
and if it does not, the run stops rather than training against a different
opponent distribution than it believes.

Exact counts, not sampling
--------------------------
The historical bucket is *partitioned* into per-member game counts by the
largest-remainder rule, with exact ties broken by canonical member order
rotated by the iteration index. Sampling would let the realized mix drift from
the frozen percentages within an iteration; partitioning cannot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction

from .phase14_contract import (
    ANCHOR_CHECKPOINTS,
    ANCHOR_SHA256,
    CATEGORY_ANCHOR,
    CATEGORY_MIDDLE,
    CATEGORY_OLDER,
    CATEGORY_RECENT,
    PHASE14_POOL_VERSION,
    POOL_ANCHORS,
    POOL_CATEGORY_WEIGHTS,
    POOL_MIDDLE_SLOTS,
    POOL_OLDER_SLOTS,
    POOL_RECENT_SLOTS,
    POOL_SIZE,
    POOL_SNAPSHOT_CATEGORIES,
    POOL_SNAPSHOT_SLOTS,
)


class Phase14PoolError(RuntimeError):
    """Raised when an archive or active-pool request is not well formed."""


ANCHOR_TOKEN_PREFIX = "phase14_anchor_v1"
ARCHIVE_TOKEN_PREFIX = "phase14_archive_v1"


def snapshot_identity(position: int) -> str:
    """The logical identity of the archive's `position`-th snapshot.

    Position, not the 2-hour mark: the pool is a function of *ordering*, and a
    mark that was crossed while one long iteration was in flight would leave a
    gap in a mark-numbered scheme. The mark is recorded on the entry.
    """
    if not isinstance(position, int) or isinstance(position, bool) or position < 1:
        raise Phase14PoolError(f"archive position must be an int >= 1, got {position!r}")
    return f"S{position:04d}"


def historical_policy_token(identity: str) -> str:
    """The stored `collection_policy_version` of a historical side."""
    if identity in POOL_ANCHORS:
        return f"{ANCHOR_TOKEN_PREFIX}|{identity}"
    return f"{ARCHIVE_TOKEN_PREFIX}|{identity}"


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveEntry:
    """One durable snapshot, addressable by identity and by real bytes."""

    position: int
    identity: str
    archive_mark: int
    path: str
    sha256: str
    model_state_digest: str
    elapsed_seconds: float
    written_utc: str
    iteration: int
    global_optimizer_step: int

    def to_dict(self) -> dict:
        return {
            "position": int(self.position),
            "identity": self.identity,
            "archive_mark": int(self.archive_mark),
            "path": self.path,
            "sha256": self.sha256,
            "model_state_digest": self.model_state_digest,
            "elapsed_seconds": float(self.elapsed_seconds),
            "written_utc": self.written_utc,
            "iteration": int(self.iteration),
            "global_optimizer_step": int(self.global_optimizer_step),
        }

    @staticmethod
    def from_dict(payload: dict) -> "ArchiveEntry":
        return ArchiveEntry(
            position=int(payload["position"]),
            identity=str(payload["identity"]),
            archive_mark=int(payload["archive_mark"]),
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            model_state_digest=str(payload["model_state_digest"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            written_utc=str(payload["written_utc"]),
            iteration=int(payload["iteration"]),
            global_optimizer_step=int(payload["global_optimizer_step"]),
        )


@dataclass
class HistoricalArchive:
    """Every durable Phase 14 snapshot, in the order it was written.

    Append-only by construction: there is no removal method, because the
    contract forbids pruning an archive entry merely because it is not an
    active opponent. Membership of the *pool* is what changes.
    """

    entries: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def k(self) -> int:
        return len(self.entries)

    def append(
        self,
        *,
        archive_mark: int,
        path: str,
        sha256: str,
        model_state_digest: str,
        elapsed_seconds: float,
        written_utc: str,
        iteration: int,
        global_optimizer_step: int,
    ) -> ArchiveEntry:
        position = len(self.entries) + 1
        entry = ArchiveEntry(
            position=position,
            identity=snapshot_identity(position),
            archive_mark=int(archive_mark),
            path=str(path),
            sha256=str(sha256),
            model_state_digest=str(model_state_digest),
            elapsed_seconds=float(elapsed_seconds),
            written_utc=str(written_utc),
            iteration=int(iteration),
            global_optimizer_step=int(global_optimizer_step),
        )
        self.entries.append(entry)
        return entry

    def entry(self, identity: str) -> ArchiveEntry:
        for candidate in self.entries:
            if candidate.identity == identity:
                return candidate
        raise Phase14PoolError(f"the archive holds no snapshot {identity!r}")

    def at(self, position: int) -> ArchiveEntry:
        if not 1 <= position <= len(self.entries):
            raise Phase14PoolError(
                f"archive position {position} is outside 1..{len(self.entries)}"
            )
        return self.entries[position - 1]

    def identities(self) -> tuple:
        return tuple(entry.identity for entry in self.entries)

    def digest_map(self) -> dict:
        return {entry.identity: entry.sha256 for entry in self.entries}

    def to_dict(self) -> dict:
        return {
            "archive_version": "phase14_archive_v1",
            "k": self.k,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @staticmethod
    def from_dict(payload: dict) -> "HistoricalArchive":
        entries = [ArchiveEntry.from_dict(item) for item in payload.get("entries", [])]
        for index, entry in enumerate(entries, start=1):
            if entry.position != index or entry.identity != snapshot_identity(index):
                raise Phase14PoolError(
                    f"archive entry {index} claims position {entry.position} / identity "
                    f"{entry.identity!r}; the archive's order is its identity"
                )
        return HistoricalArchive(entries=entries)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


# ---------------------------------------------------------------------------
# f(k): the membership function
# ---------------------------------------------------------------------------


def _quantile_picks(values: list, slots: int) -> tuple:
    """`slots` evenly spaced picks from `values`, endpoints included.

    The frozen spelling: `values[floor(i*(len-1)/(slots-1))]` for i in
    0..slots-1. With `len >= slots` the picks are distinct, which the caller
    guarantees by only using this above k = 14.
    """
    if not values:
        return ()
    if len(values) <= slots:
        return tuple(values)
    last = len(values) - 1
    return tuple(values[(index * last) // (slots - 1)] for index in range(slots))


def active_pool_positions(k: int) -> dict:
    """The archive positions in each age band at archive size `k`.

    Below 15 snapshots every snapshot is active and the bands are contiguous
    thirds; at 15 and above the six newest are `recent`, and the remainder is
    split in half and quantile-sampled four ways each. Empty bands are legal
    and are exactly what a young run has.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise Phase14PoolError(f"archive size must be a non-negative int, got {k!r}")
    if k == 0:
        return {CATEGORY_OLDER: (), CATEGORY_MIDDLE: (), CATEGORY_RECENT: ()}
    if k <= POOL_SNAPSHOT_SLOTS:
        first = k // 3
        second = (2 * k) // 3
        return {
            CATEGORY_OLDER: tuple(range(1, first + 1)),
            CATEGORY_MIDDLE: tuple(range(first + 1, second + 1)),
            CATEGORY_RECENT: tuple(range(second + 1, k + 1)),
        }
    recent = tuple(range(k - POOL_RECENT_SLOTS + 1, k + 1))
    remainder = list(range(1, k - POOL_RECENT_SLOTS + 1))
    half = -(-len(remainder) // 2)  # ceil
    older_half = remainder[:half]
    middle_half = remainder[half:]
    return {
        CATEGORY_OLDER: _quantile_picks(older_half, POOL_OLDER_SLOTS),
        CATEGORY_MIDDLE: _quantile_picks(middle_half, POOL_MIDDLE_SLOTS),
        CATEGORY_RECENT: recent,
    }


@dataclass(frozen=True)
class ActivePool:
    """The pool of one moment: anchors plus the age-banded snapshot picks.

    Frozen and comparable: :meth:`digest` is what a hot checkpoint stores and
    what a resume recomputes, so "the same logical point in the run" is a
    check rather than an assumption.
    """

    k: int
    categories: dict
    checkpoints: dict

    @staticmethod
    def for_archive(archive: HistoricalArchive) -> "ActivePool":
        positions = active_pool_positions(archive.k)
        categories = {CATEGORY_ANCHOR: tuple(POOL_ANCHORS)}
        checkpoints = {
            name: {"path": ANCHOR_CHECKPOINTS[name], "sha256": ANCHOR_SHA256[name]}
            for name in POOL_ANCHORS
        }
        for category in POOL_SNAPSHOT_CATEGORIES:
            members = []
            for position in positions[category]:
                entry = archive.at(position)
                members.append(entry.identity)
                checkpoints[entry.identity] = {
                    "path": entry.path,
                    "sha256": entry.sha256,
                }
            categories[category] = tuple(members)
        pool = ActivePool(k=archive.k, categories=categories, checkpoints=checkpoints)
        pool.validate()
        return pool

    # -- shape -------------------------------------------------------------

    def validate(self) -> None:
        members = self.members()
        if len(members) != len(set(members)):
            raise Phase14PoolError(f"the active pool repeats a member: {members}")
        if len(members) > POOL_SIZE:
            raise Phase14PoolError(
                f"the active pool holds {len(members)} members, above the frozen "
                f"bound of {POOL_SIZE}"
            )
        snapshots = sum(len(self.categories[name]) for name in POOL_SNAPSHOT_CATEGORIES)
        if snapshots > POOL_SNAPSHOT_SLOTS:
            raise Phase14PoolError(
                f"the active pool holds {snapshots} snapshots, above the frozen "
                f"{POOL_SNAPSHOT_SLOTS}"
            )
        if tuple(self.categories[CATEGORY_ANCHOR]) != tuple(POOL_ANCHORS):
            raise Phase14PoolError(
                f"the anchors are {self.categories[CATEGORY_ANCHOR]}, not the "
                f"permanent {list(POOL_ANCHORS)}"
            )

    def members(self) -> tuple:
        """Canonical member order: P8, P9, older, middle, recent — each ascending."""
        ordered: list = []
        for category in (CATEGORY_ANCHOR,) + POOL_SNAPSHOT_CATEGORIES:
            ordered.extend(self.categories.get(category, ()))
        return tuple(ordered)

    def category_of(self, identity: str) -> str:
        for category, members in self.categories.items():
            if identity in members:
                return category
        raise Phase14PoolError(f"{identity!r} is not an active pool member")

    def checkpoint_for(self, identity: str) -> dict:
        if identity not in self.checkpoints:
            raise Phase14PoolError(f"{identity!r} has no bound checkpoint")
        return dict(self.checkpoints[identity])

    # -- weights -----------------------------------------------------------

    def category_weights(self) -> dict:
        """The frozen category weights after empty-category redistribution.

        An empty snapshot category's weight goes to the non-empty snapshot
        categories in proportion to their frozen weights; with no snapshots at
        all the whole historical share goes to the anchors. Anchors are never
        empty, so the anchor share is only ever *raised* by that last rule.
        """
        frozen = {name: Fraction(str(POOL_CATEGORY_WEIGHTS[name])) for name in POOL_CATEGORY_WEIGHTS}
        occupied = [
            name for name in POOL_SNAPSHOT_CATEGORIES if self.categories.get(name)
        ]
        if not occupied:
            return {CATEGORY_ANCHOR: Fraction(1)}
        spare = sum(
            frozen[name]
            for name in POOL_SNAPSHOT_CATEGORIES
            if name not in occupied
        )
        base = sum(frozen[name] for name in occupied)
        weights = {CATEGORY_ANCHOR: frozen[CATEGORY_ANCHOR]}
        for name in occupied:
            weights[name] = frozen[name] + (spare * frozen[name] / base if spare else Fraction(0))
        return weights

    def member_weights(self) -> dict:
        """Per-member sampling weight: a category's weight split equally."""
        categories = self.category_weights()
        weights: dict = {}
        for category, weight in categories.items():
            members = self.categories.get(category, ())
            if not members:
                continue
            share = weight / len(members)
            for identity in members:
                weights[identity] = share
        total = sum(weights.values())
        if total != Fraction(1):
            raise Phase14PoolError(
                f"the member weights sum to {float(total)!r}, not 1"
            )
        return weights

    # -- identity ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "pool_version": PHASE14_POOL_VERSION,
            "k": int(self.k),
            "categories": {
                category: list(members) for category, members in self.categories.items()
            },
            "checkpoints": {
                identity: dict(binding) for identity, binding in self.checkpoints.items()
            },
            "member_weights": {
                identity: float(weight) for identity, weight in self.member_weights().items()
            },
        }

    @staticmethod
    def from_dict(payload: dict) -> "ActivePool":
        pool = ActivePool(
            k=int(payload["k"]),
            categories={
                category: tuple(members)
                for category, members in payload["categories"].items()
            },
            checkpoints={
                identity: dict(binding)
                for identity, binding in payload["checkpoints"].items()
            },
        )
        pool.validate()
        return pool

    def digest(self) -> str:
        """The comparable identity of one pool: membership and bindings."""
        return hashlib.sha256(
            json.dumps(
                {
                    "pool_version": PHASE14_POOL_VERSION,
                    "k": int(self.k),
                    "categories": {
                        category: list(members)
                        for category, members in sorted(self.categories.items())
                    },
                    "checkpoints": {
                        identity: dict(binding)
                        for identity, binding in sorted(self.checkpoints.items())
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def assert_pool_matches(archive: HistoricalArchive, recorded: dict) -> ActivePool:
    """Refuse to continue unless f(archive) is the pool the checkpoint recorded.

    The frozen resume rule. A run that resumed against a differently-composed
    pool would still look healthy in every metric while training against a
    different opponent distribution than its own checkpoint claims, so the
    disagreement has to be fatal rather than logged.
    """
    recomputed = ActivePool.for_archive(archive)
    saved = ActivePool.from_dict(recorded)
    if recomputed.digest() != saved.digest():
        raise Phase14PoolError(
            "the active pool recomputed from the archive is not the checkpointed "
            f"pool: recomputed {recomputed.members()} (k={recomputed.k}) != "
            f"recorded {saved.members()} (k={saved.k})"
        )
    return recomputed


# ---------------------------------------------------------------------------
# Exact per-member game counts
# ---------------------------------------------------------------------------


def exact_member_counts(total: int, pool: ActivePool, iteration: int) -> dict:
    """Partition `total` historical games across the pool, exactly.

    Largest remainder over exact rational shares: every member takes
    `floor(total * w)`, and the leftover games go one each to the largest
    fractional parts. Exact ties — which the frozen weights produce constantly,
    since members inside a category share a weight — are broken by canonical
    member order rotated left by `iteration mod tie_group_size`, so the same
    member does not collect the remainder every iteration.

    Rational arithmetic, not float: with float shares two members of one
    category can differ in the last bit and silently stop being a tie.
    """
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise Phase14PoolError(f"the historical bucket must be an int >= 0, got {total!r}")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        raise Phase14PoolError(f"iteration must be an int >= 1, got {iteration!r}")
    weights = pool.member_weights()
    order = [identity for identity in pool.members() if identity in weights]

    counts: dict = {}
    fractions: dict = {}
    for identity in order:
        share = Fraction(total) * weights[identity]
        floor = int(share)
        counts[identity] = floor
        fractions[identity] = share - floor
    remaining = total - sum(counts.values())

    groups: dict = {}
    for identity in order:
        groups.setdefault(fractions[identity], []).append(identity)
    ranked: list = []
    for value in sorted(groups, reverse=True):
        group = groups[value]
        offset = iteration % len(group)
        ranked.extend(group[offset:] + group[:offset])

    for identity in ranked[:remaining]:
        counts[identity] += 1
    if sum(counts.values()) != total:  # pragma: no cover - arithmetic invariant
        raise Phase14PoolError(
            f"the partition assigned {sum(counts.values())} of {total} games"
        )
    return counts


def member_ordinal_ranges(total: int, pool: ActivePool, iteration: int) -> tuple:
    """`((identity, start, stop), ...)` over the historical bucket's ordinals.

    Contiguous half-open ranges in canonical member order. Turning the counts
    into ranges is what lets a single ordinal name its opponent with no state:
    :func:`member_for_ordinal` is a lookup, so a crashed game regenerates its
    own opponent from its id.
    """
    counts = exact_member_counts(total, pool, iteration)
    ranges: list = []
    cursor = 0
    for identity in pool.members():
        count = counts.get(identity, 0)
        if count == 0:
            continue
        ranges.append((identity, cursor, cursor + count))
        cursor += count
    return tuple(ranges)


def member_for_ordinal(ordinal: int, total: int, pool: ActivePool, iteration: int) -> str:
    """The pool member that plays historical-bucket ordinal `ordinal`."""
    for identity, start, stop in member_ordinal_ranges(total, pool, iteration):
        if start <= ordinal < stop:
            return identity
    raise Phase14PoolError(
        f"historical ordinal {ordinal} is outside 0..{total - 1}"
    )


def realized_shares(total: int, pool: ActivePool, iteration: int) -> dict:
    """What fraction of the historical bucket each category actually received.

    Reported, not enforced: the partition is exact by construction, and this is
    how telemetry shows the mix without inviting anybody to adjust it.
    """
    counts = exact_member_counts(total, pool, iteration)
    shares: dict = {}
    for category, members in pool.categories.items():
        got = sum(counts.get(identity, 0) for identity in members)
        shares[category] = {
            "games": got,
            "share": (got / total) if total else 0.0,
            "members": list(members),
        }
    return shares


def pool_semantics() -> dict:
    return {
        "pool_version": PHASE14_POOL_VERSION,
        "size": POOL_SIZE,
        "anchors": list(POOL_ANCHORS),
        "snapshot_slots": POOL_SNAPSHOT_SLOTS,
        "bands": {
            CATEGORY_OLDER: POOL_OLDER_SLOTS,
            CATEGORY_MIDDLE: POOL_MIDDLE_SLOTS,
            CATEGORY_RECENT: POOL_RECENT_SLOTS,
        },
        "weights": dict(POOL_CATEGORY_WEIGHTS),
        "membership": "pure function f(k) of the ordered archive; no tournament",
        "counts": "exact largest-remainder partition, ties rotated by iteration",
        "archive": "append-only; entries are never pruned for inactivity",
    }
