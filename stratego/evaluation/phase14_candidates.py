"""Phase 14: the fixed-pack candidate evaluator and the selection rule.

Specification source: `reports/phase13/phase14_checkpoint_selection_pack_v1.json`
and `..._selection_rule_v1.json` (both FROZEN), via
`02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` section 10.

Monitoring, not control
-----------------------
This module measures candidates. It cannot change training, and the shape of
the code is what makes that true rather than a promise: nothing here imports
the trainer, the collector, the scheduler or the clock, and no function returns
a value the training loop consults. A candidate evaluation that fails leaves a
`failed` row in the ledger, the candidate itself untouched on the archive, and
the run going.

The pack is not a parameter
---------------------------
Every candidate is played on the *same* 128 games: same boards, same colours,
same opponents, same seeds. :func:`load_pack` recomputes the frozen
`pack_content_digest` from the document and refuses a pack that does not match,
so "the same pack" is checked at every use rather than assumed from a filename.

No search
---------
Candidates are direct policies: greedy float32 through the accepted
`RemoteNeuralPolicy`. Nothing under `stratego.search` is imported here, and the
spent Phase 11 sealed test bank is not touched.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

from ..engine.constants import BLUE, EVALUATION_RULES, RED
from ..training.phase14_contract import (
    SELECTION_GAMES_PER_STRATUM,
    SELECTION_PACK_DIGEST,
    SELECTION_PACK_GAMES,
    SELECTION_PACK_RELATIVE_PATH,
    SELECTION_RULE,
    SELECTION_RULE_RELATIVE_PATH,
    SELECTION_STRATA,
    repository_root,
)
from .match_runner import play_match
from .match_spec import MatchSpec
from .neural_worker import (
    DECISION_MODE_GREEDY,
    InferenceOwner,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from .phase10b_eval import FrozenSeedPolicy
from .policy import PolicyRef
from .registry import build_policy

PHASE14_EVALUATOR_VERSION = "phase14_candidate_evaluator_v1"

CANDIDATE_POLICY_ID = "phase14_candidate"
ANCHOR_POLICY_ID = "phase9_anchor"

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUSES = (STATUS_PENDING, STATUS_COMPLETE, STATUS_FAILED)


class Phase14CandidateError(RuntimeError):
    """Raised when a candidate evaluation cannot run as the pack specifies."""


# ---------------------------------------------------------------------------
# The frozen pack
# ---------------------------------------------------------------------------


def canonical_json_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pack_path() -> Path:
    return repository_root() / SELECTION_PACK_RELATIVE_PATH


def load_pack(path=None) -> dict:
    """Read the frozen pack and re-derive its content digest.

    The digest covers everything except `written_utc`, exactly as Agent 1
    computed it. Recomputing rather than reading the recorded value is the
    difference between "this is the frozen pack" and "this file says it is".
    """
    target = Path(path) if path is not None else pack_path()
    if not target.exists():
        raise Phase14CandidateError(f"the frozen selection pack is missing at {target}")
    pack = json.loads(target.read_text())
    recorded = pack.get("pack_content_digest")
    recomputed = canonical_json_digest(
        {
            key: value
            for key, value in pack.items()
            if key not in ("written_utc", "pack_content_digest")
        }
    )
    if recomputed != recorded:
        raise Phase14CandidateError(
            f"{target}: recomputed pack digest {recomputed} != recorded {recorded}"
        )
    if recorded != SELECTION_PACK_DIGEST:
        raise Phase14CandidateError(
            f"{target}: pack digest {recorded} != the contracted {SELECTION_PACK_DIGEST}"
        )
    if len(pack["games"]) != SELECTION_PACK_GAMES:
        raise Phase14CandidateError(
            f"{target}: {len(pack['games'])} games, not the contracted "
            f"{SELECTION_PACK_GAMES}"
        )
    return pack


def load_selection_rule(path=None) -> dict:
    target = Path(path) if path is not None else repository_root() / SELECTION_RULE_RELATIVE_PATH
    if not target.exists():
        raise Phase14CandidateError(f"the frozen selection rule is missing at {target}")
    rule = json.loads(target.read_text())
    if rule["pack_binding"]["pack_content_digest"] != SELECTION_PACK_DIGEST:
        raise Phase14CandidateError(
            "the selection rule is bound to a different pack than the contract names"
        )
    return rule


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------


def _candidate_ref() -> PolicyRef:
    return PolicyRef(policy_id=CANDIDATE_POLICY_ID, policy_version=PHASE14_EVALUATOR_VERSION)


def _anchor_ref() -> PolicyRef:
    return PolicyRef(policy_id=ANCHOR_POLICY_ID, policy_version=PHASE14_EVALUATOR_VERSION)


class _Seats:
    """The loaded seats of one evaluation pass, closed together.

    One inference owner per set of weights for the whole 128 games: the accepted
    "checkpoint loads per long-lived inference owner = 1" shape, which is also
    the only affordable one when a candidate arrives every six hours.
    """

    def __init__(self, candidate_weights, *, anchor_weights, device: str = "mps") -> None:
        self.candidate_owner = InferenceOwner(
            candidate_weights,
            decision_mode=DECISION_MODE_GREEDY,
            device=device,
            name="phase14_candidate",
        )
        self.anchor_owner = InferenceOwner(
            anchor_weights,
            decision_mode=DECISION_MODE_GREEDY,
            device=device,
            name="phase9_anchor",
        )
        self.candidate_policy = RemoteNeuralPolicy(
            _candidate_ref(),
            LocalInferenceChannel(self.candidate_owner),
            decision_mode=DECISION_MODE_GREEDY,
        )
        self.anchor_policy = RemoteNeuralPolicy(
            _anchor_ref(),
            LocalInferenceChannel(self.anchor_owner),
            decision_mode=DECISION_MODE_GREEDY,
        )
        self._rules: dict = {}

    def opponent(self, stratum: str):
        """`(ref, policy)` for one stratum's opponent seat."""
        if stratum == "phase9_anchor":
            return _anchor_ref(), self.anchor_policy
        if stratum not in self._rules:
            self._rules[stratum] = build_policy(stratum)
        policy = self._rules[stratum]
        return policy.ref, policy

    def identity(self) -> dict:
        return {
            "candidate": self.candidate_owner.identity(),
            "anchor": self.anchor_owner.identity(),
        }

    def close(self) -> None:
        for owner in (self.candidate_owner, self.anchor_owner):
            close = getattr(owner, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 - a closed seat is best-effort
                    pass


# ---------------------------------------------------------------------------
# Playing the pack
# ---------------------------------------------------------------------------


def _spec_for(game: dict, opponent_ref: PolicyRef) -> MatchSpec:
    color = RED if game["candidate_color"] == "red" else BLUE
    return MatchSpec(
        candidate=_candidate_ref(),
        opponent=opponent_ref,
        setup_pair_id=int(game["setup_ordinal"]),
        candidate_color=color,
        replicate=0,
        root_seed=int(game["candidate_decision_seed"]),
        suite_version=PHASE14_EVALUATOR_VERSION,
        setup_bank_version=f"phase14_selection_pack_v1|{game['opponent']}",
        rules=EVALUATION_RULES,
    )


def play_pack_game(game: dict, seats: _Seats) -> dict:
    """Play one frozen pack game and return its scored row.

    The setups come straight from the pack's stored `oriented_engine_setup`
    tuples, so the boards are the frozen ones rather than boards re-derived
    through a selector that would have to be identically configured.
    """
    opponent_ref, opponent_policy = seats.opponent(game["opponent"])
    spec = _spec_for(game, opponent_ref)
    candidate = FrozenSeedPolicy(
        seats.candidate_policy, int(game["candidate_decision_seed"])
    )
    opponent = FrozenSeedPolicy(opponent_policy, int(game["opponent_decision_seed"]))
    result = play_match(
        spec,
        setups=(
            tuple(game["red"]["oriented_engine_setup"]),
            tuple(game["blue"]["oriented_engine_setup"]),
        ),
        policies={
            spec.candidate.token: candidate,
            spec.opponent.token: opponent,
        },
        record_actions=False,
    )
    return {
        "game_id": game["game_id"],
        "opponent": game["opponent"],
        "candidate_color": game["candidate_color"],
        "candidate_result": result.candidate_result,
        "candidate_score": result.candidate_score,
        "draw": bool(result.draw),
        "terminal_reason": result.terminal_reason,
        "plies": int(result.plies),
        "decisions": int(result.decisions),
        "replay_digest": result.replay_digest,
        "seconds": float(result.wall_clock_seconds),
    }


def score_rows(rows) -> dict:
    """Per-stratum and overall EWR, exactly as the frozen rule defines them."""
    by_stratum: dict = {}
    for row in rows:
        by_stratum.setdefault(row["opponent"], []).append(row)
    strata: dict = {}
    for stratum, games in sorted(by_stratum.items()):
        scores = [float(row["candidate_score"] or 0.0) for row in games]
        wins = sum(1 for row in games if row["candidate_result"] == "win")
        draws = sum(1 for row in games if row["draw"])
        strata[stratum] = {
            "games": len(games),
            "wins": wins,
            "draws": draws,
            "losses": len(games) - wins - draws,
            "ewr": (wins + 0.5 * draws) / len(games) if games else 0.0,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
        }
    present = [name for name in SELECTION_STRATA if name in strata]
    mean = (
        sum(strata[name]["ewr"] for name in present) / len(present) if present else 0.0
    )
    return {
        "strata": strata,
        "mean_ewr": mean,
        "min_stratum_ewr": min((strata[name]["ewr"] for name in present), default=0.0),
        "games": len(rows),
        "complete": sorted(strata) == sorted(SELECTION_STRATA)
        and all(entry["games"] == SELECTION_GAMES_PER_STRATUM for entry in strata.values()),
    }


def evaluate_candidate(
    candidate_weights,
    *,
    anchor_weights,
    pack: "dict | None" = None,
    device: str = "mps",
    limit: "int | None" = None,
    progress=None,
) -> dict:
    """Play the whole frozen pack with one candidate and score it.

    `limit` exists for integration tests, which need to prove the evaluator
    runs without paying for 128 games; a limited run is marked `complete:
    False` and the selection rule refuses it.
    """
    pack = pack or load_pack()
    games = list(pack["games"])
    if limit is not None:
        games = games[: int(limit)]
    seats = _Seats(candidate_weights, anchor_weights=anchor_weights, device=device)
    started = time.perf_counter()
    rows: list = []
    try:
        for index, game in enumerate(games, start=1):
            rows.append(play_pack_game(game, seats))
            if progress is not None:
                progress(index, len(games))
    finally:
        seats.close()
    scored = score_rows(rows)
    scored.update(
        {
            "evaluator_version": PHASE14_EVALUATOR_VERSION,
            "pack_content_digest": pack["pack_content_digest"],
            "candidate_weights": str(candidate_weights),
            "anchor_weights": str(anchor_weights),
            "device": device,
            "seconds": time.perf_counter() - started,
            "games_played": len(rows),
            "games_in_pack": len(pack["games"]),
            "search_used": False,
            "rows": rows,
        }
    )
    return scored


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CandidateLedger:
    """Every candidate's evaluation status, on disk, survivable across restart.

    The ledger is the mechanism behind "evaluation failure != training failure":
    a candidate is recorded the moment it is marked, its evaluation is a
    separate row that may be `pending` for hours or days, and a failed attempt
    stores its reason and stays re-runnable on the identical pack.
    """

    path: Path

    @staticmethod
    def at(directory) -> "CandidateLedger":
        return CandidateLedger(path=Path(directory) / "phase14_candidate_ledger.json")

    def read(self) -> dict:
        if not self.path.exists():
            return {
                "artifact": "phase14_candidate_ledger_v1",
                "pack_content_digest": SELECTION_PACK_DIGEST,
                "selection_rule": SELECTION_RULE,
                "candidates": {},
            }
        return json.loads(self.path.read_text())

    def write(self, document: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return document

    def record_candidate(self, hour: int, mark: dict) -> dict:
        document = self.read()
        entry = document["candidates"].setdefault(str(int(hour)), {})
        entry.update(
            {
                "hour": int(hour),
                "mark": mark,
                "status": entry.get("status", STATUS_PENDING),
            }
        )
        return self.write(document)

    def record_result(self, hour: int, result: dict) -> dict:
        document = self.read()
        entry = document["candidates"].setdefault(str(int(hour)), {"hour": int(hour)})
        entry.update(
            {
                "status": STATUS_COMPLETE,
                "mean_ewr": result["mean_ewr"],
                "min_stratum_ewr": result["min_stratum_ewr"],
                "strata": {
                    name: entry_["ewr"] for name, entry_ in result["strata"].items()
                },
                "detail": {
                    name: dict(entry_) for name, entry_ in result["strata"].items()
                },
                "games_played": result["games_played"],
                "complete": bool(result["complete"]),
                "pack_content_digest": result["pack_content_digest"],
                "seconds": result["seconds"],
                "attempts": int(entry.get("attempts", 0)) + 1,
                "error": None,
            }
        )
        return self.write(document)

    def record_failure(self, hour: int, error: str) -> dict:
        """A failed evaluation. The candidate is preserved; training continues."""
        document = self.read()
        entry = document["candidates"].setdefault(str(int(hour)), {"hour": int(hour)})
        entry.update(
            {
                "status": STATUS_FAILED,
                "error": str(error)[:2000],
                "attempts": int(entry.get("attempts", 0)) + 1,
                "rerunnable": True,
                "note": "the candidate is preserved and reruns later on the same pack",
            }
        )
        return self.write(document)

    def pending(self) -> list:
        document = self.read()
        return sorted(
            (
                entry
                for entry in document["candidates"].values()
                if entry.get("status") != STATUS_COMPLETE
            ),
            key=lambda entry: int(entry["hour"]),
        )

    def completed(self) -> list:
        document = self.read()
        return sorted(
            (
                entry
                for entry in document["candidates"].values()
                if entry.get("status") == STATUS_COMPLETE and entry.get("complete")
            ),
            key=lambda entry: int(entry["hour"]),
        )

    def status_summary(self) -> dict:
        document = self.read()
        counts: dict = {status: 0 for status in STATUSES}
        for entry in document["candidates"].values():
            counts[entry.get("status", STATUS_PENDING)] = (
                counts.get(entry.get("status", STATUS_PENDING), 0) + 1
            )
        return {
            "ledger": str(self.path),
            "candidates": len(document["candidates"]),
            "by_status": counts,
            "pack_content_digest": document.get("pack_content_digest"),
        }


# ---------------------------------------------------------------------------
# The frozen selection rule
# ---------------------------------------------------------------------------


def select_final_candidate(entries) -> dict:
    """Apply the frozen rule: mean EWR, then min-stratum EWR, then later hour.

    Exact arithmetic on the frozen pack results only — no confidence intervals,
    no tests, no reweighting. Incomplete evaluations are refused rather than
    compared, because a candidate scored on 40 games is not comparable with one
    scored on 128.
    """
    usable = [entry for entry in entries if entry.get("complete")]
    if not usable:
        raise Phase14CandidateError(
            "no candidate has a complete evaluation on the frozen pack"
        )
    ranked = sorted(
        usable,
        key=lambda entry: (
            float(entry["mean_ewr"]),
            float(entry["min_stratum_ewr"]),
            int(entry["hour"]),
        ),
        reverse=True,
    )
    winner = ranked[0]
    return {
        "selected_hour": int(winner["hour"]),
        "mean_ewr": float(winner["mean_ewr"]),
        "min_stratum_ewr": float(winner["min_stratum_ewr"]),
        "rule": SELECTION_RULE,
        "candidates_considered": len(usable),
        "ranking": [
            {
                "hour": int(entry["hour"]),
                "mean_ewr": float(entry["mean_ewr"]),
                "min_stratum_ewr": float(entry["min_stratum_ewr"]),
            }
            for entry in ranked
        ],
    }


def evaluator_semantics() -> dict:
    return {
        "evaluator_version": PHASE14_EVALUATOR_VERSION,
        "pack_digest": SELECTION_PACK_DIGEST,
        "games": SELECTION_PACK_GAMES,
        "strata": list(SELECTION_STRATA),
        "decision_rule": "greedy float32 through the accepted RemoteNeuralPolicy",
        "match_runner": "stratego.evaluation.match_runner.play_match, accepted",
        "seeds": "the pack's own opponent/candidate decision seeds, pinned",
        "isolation": (
            "no trainer, collector, scheduler or clock is imported; a failure "
            "records a ledger row and nothing else"
        ),
        "search": "absent",
        "selection_rule": SELECTION_RULE,
    }
