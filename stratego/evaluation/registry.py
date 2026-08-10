"""The named catalogue of Phase 4 opponents.

Specification source: Phase 4 Agent 2 instructions ("Required baseline
policies", "Required stress/unusual policies").

Agents 3 and 4 schedule matches from :class:`PolicyRef` tokens that arrive from
a schedule, a data file or a command line. They therefore need one place that
turns `policy_id` back into a live policy object, and one place that enumerates
"every opponent Phase 4 defines". Spreading either across modules is how a
policy quietly drops out of an audit that claims to cover all of them.

The catalogue deliberately excludes the `contract_*` fixtures from
:mod:`stratego.evaluation.policy`. Those exist to test the interface and have no
strategy at all; including them in a league would report a meaningless strength
number for something that is not an opponent.
"""

from collections.abc import Iterable, Sequence

from .baselines import (
    BASELINE_SUITE_VERSION,
    LADDER_POLICY_CLASSES,
    BasicHeuristicPolicy,
    RandomLegalPolicy,
    StrategicRuleBasedPolicy,
    TacticalRuleBasedPolicy,
)
from .policy import Policy, PolicyRef
from .stress import STRESS_POLICY_CLASSES

#: The four-tier ladder, weakest first. This ordering is the hypothesis Agent 4
#: tests; it is not an assertion Agent 2 makes about measured strength.
LADDER_POLICY_IDS: tuple[str, ...] = tuple(
    policy_class.policy_id for policy_class in LADDER_POLICY_CLASSES
)

STRESS_POLICY_IDS: tuple[str, ...] = tuple(
    policy_class.policy_id for policy_class in STRESS_POLICY_CLASSES
)

ALL_POLICY_CLASSES: tuple[type[Policy], ...] = LADDER_POLICY_CLASSES + STRESS_POLICY_CLASSES

ALL_POLICY_IDS: tuple[str, ...] = LADDER_POLICY_IDS + STRESS_POLICY_IDS


def _build_index() -> dict[str, type[Policy]]:
    index: dict[str, type[Policy]] = {}
    for policy_class in ALL_POLICY_CLASSES:
        policy_id = policy_class.policy_id
        if policy_id in index:  # pragma: no cover - guarded by the uniqueness test
            raise ValueError(f"duplicate policy_id in the Phase 4 catalogue: {policy_id}")
        if policy_id.startswith("contract_"):  # pragma: no cover - naming rule
            raise ValueError(
                f"{policy_id!r} uses the reserved contract-fixture prefix; a real "
                "opponent must not be mistakable for an interface fixture"
            )
        index[policy_id] = policy_class
    return index


POLICY_INDEX: dict[str, type[Policy]] = _build_index()


class UnknownPolicyError(KeyError):
    """Raised when a policy identifier is not in the Phase 4 catalogue."""


def build_policy(policy_id: str) -> Policy:
    """Instantiate a catalogued policy by identifier."""
    try:
        policy_class = POLICY_INDEX[policy_id]
    except KeyError:
        raise UnknownPolicyError(
            f"unknown policy_id {policy_id!r}; known identifiers are "
            f"{', '.join(ALL_POLICY_IDS)}"
        ) from None
    return policy_class()


def build_policies(policy_ids: "Iterable[str] | None" = None) -> tuple[Policy, ...]:
    """Instantiate several policies, defaulting to the whole catalogue."""
    identifiers = ALL_POLICY_IDS if policy_ids is None else tuple(policy_ids)
    return tuple(build_policy(policy_id) for policy_id in identifiers)


def policy_ref(policy_id: str) -> PolicyRef:
    """The `id@version` reference a schedule needs, without instantiating."""
    return PolicyRef(policy_id, POLICY_INDEX[policy_id].policy_version)


def policy_refs(policy_ids: "Iterable[str] | None" = None) -> tuple[PolicyRef, ...]:
    identifiers = ALL_POLICY_IDS if policy_ids is None else tuple(policy_ids)
    return tuple(policy_ref(policy_id) for policy_id in identifiers)


def policy_catalog(policy_ids: "Sequence[str] | None" = None) -> list[dict]:
    """Serialisable description of the catalogue, for reports and data files."""
    identifiers = ALL_POLICY_IDS if policy_ids is None else tuple(policy_ids)
    catalog = []
    for policy_id in identifiers:
        policy = build_policy(policy_id)
        entry = policy.describe()
        entry["role"] = "ladder" if policy_id in LADDER_POLICY_IDS else "stress"
        entry["baseline_suite_version"] = BASELINE_SUITE_VERSION
        catalog.append(entry)
    return catalog


__all__ = [
    "ALL_POLICY_CLASSES",
    "ALL_POLICY_IDS",
    "LADDER_POLICY_CLASSES",
    "LADDER_POLICY_IDS",
    "POLICY_INDEX",
    "STRESS_POLICY_CLASSES",
    "STRESS_POLICY_IDS",
    "BasicHeuristicPolicy",
    "RandomLegalPolicy",
    "StrategicRuleBasedPolicy",
    "TacticalRuleBasedPolicy",
    "UnknownPolicyError",
    "build_policies",
    "build_policy",
    "policy_catalog",
    "policy_ref",
    "policy_refs",
]
