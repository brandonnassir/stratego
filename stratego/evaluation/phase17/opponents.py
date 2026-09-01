"""Phase 17 Agent 5: the evaluation-only opponent roster, bound to bytes.

Specification source: common contract section 11 -- "Historical, Phase 9,
rule-based and stress opponents are evaluation instruments only."

Three of the ten opponents are frozen neural checkpoints and seven are accepted
catalogue policies resolved by id, so the roster costs 10.4 MB of weights and
nothing else. The digests below are not documentation: `build_opponent_owners`
recomputes each file's sha256 before a tensor is read, so an opponent that was
replaced, truncated in transit or rebuilt cannot quietly become the thing a
candidate is scored against.
"""

from __future__ import annotations

from pathlib import Path

from .contract import (
    EVALUATION_DEVICE,
    EVALUATION_DTYPE,
    Phase17EvaluationError,
    file_sha256,
)

#: The three neural opponents: relative path, sha256, bytes.
NEURAL_OPPONENT_FILES = {
    "p18": {
        "path": "checkpoints/phase15/p18_source_readonly.pt",
        "file_sha256": (
            "aa2cc39b3867264e939c5361b32f5d10b8a3e5e268e2d99635f8d9bc00ec2412"
        ),
        "bytes": 3477949,
    },
    "p24": {
        "path": "checkpoints/phase15/p24_source_readonly.pt",
        "file_sha256": (
            "9bf256a9b085176bf48c1eca424fa10cef109f09c90999b23be62e685e917fb1"
        ),
        "bytes": 3477949,
    },
    "phase9_anchor": {
        "path": "checkpoints/phase15/phase9_anchor_readonly.pt",
        "file_sha256": (
            "ed0f5198b19f29331cd21fd2e422bd7d910b945a56b99678439e092f02950407"
        ),
        "bytes": 3479085,
    },
}

#: The seven rule and stress opponents are code, not weights.
CODE_OPPONENTS = (
    "strategic_rule_based",
    "tactical_rule_based",
    "stress_scout_rush",
    "stress_miner_rush",
    "stress_berserker",
    "stress_information_miser",
    "stress_chaos",
)


def verify_opponent_files(*, root: "Path | str" = ".") -> dict:
    """Recompute every neural opponent's sha256 before anything loads it."""
    root = Path(root)
    verified = {}
    for name, record in NEURAL_OPPONENT_FILES.items():
        target = root / record["path"]
        if not target.is_file():
            raise Phase17EvaluationError(
                f"opponent {name}: no checkpoint at {target}; the static payload "
                "is incomplete"
            )
        observed = file_sha256(target)
        if observed != record["file_sha256"]:
            raise Phase17EvaluationError(
                f"opponent {name}: {target} has sha256 {observed}, the roster "
                f"binds {record['file_sha256']}"
            )
        verified[name] = {
            "path": str(target),
            "file_sha256": observed,
            "bytes": target.stat().st_size,
        }
    return verified


def build_opponent_owners(*, root: "Path | str" = ".") -> dict:
    """One long-lived greedy inference owner per neural opponent.

    The same construction `stratego.search.phase15.matchplay.build_owners`
    performs, without routing through `load_all` -- which would also load the
    B18/B24 belief specialists this phase has no use for.
    """
    from ...evaluation.neural_worker import DECISION_MODE_GREEDY, InferenceOwner
    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    verified = verify_opponent_files(root=root)
    return {
        name: InferenceOwner(
            record["path"],
            decision_mode=DECISION_MODE_GREEDY,
            device=EVALUATION_DEVICE,
            dtype=EVALUATION_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"phase17_eval_{name}",
        )
        for name, record in verified.items()
    }


__all__ = [
    "CODE_OPPONENTS",
    "NEURAL_OPPONENT_FILES",
    "build_opponent_owners",
    "verify_opponent_files",
]
