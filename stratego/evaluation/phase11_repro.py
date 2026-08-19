"""Phase 11 Agent 4: the frozen request set, its purity, and the benchmark.

Specification sources:

- `04_AGENT_4_INFO_SAFETY_REPRO_RUNTIME.md` ("Part B — topology/restart
  reproducibility", "Part C — performance benchmark")
- Agent 1's `phase11_information_safety_v1.reproducibility` (the 2,048
  request rule, the request content, the eight legs, the canonical
  comparison, the purity statement) and `.runtime_benchmark` (the frozen
  backend, the 48-cell state rule, the four configurations, the timer, the
  warmup and the 500 ms ceiling)

One request, defined once
-------------------------
A Phase 11 request is "one belief forward plus complete worlds for sample
ordinals 0..63". :func:`execute_request` is the only implementation of that
sentence in the repository: the topology legs, the restart legs and the
runtime benchmark all call it, so "the benchmark measured something other
than what the topology legs compared" is not a state this code can reach.
Its result is summarised by :func:`request_digest`, a SHA-256 over the raw
bytes of the belief logits, the learned probabilities, the public
legal-rank masks, every sampled world and every provenance field — no
rounding, no tolerance, no field left out.

Selection consumes no randomness
--------------------------------
Both frozen sets are hash-order rules over the Agent 2 prediction store,
exactly as Agent 1 wrote them:

- the topology request set takes, per opponent stratum, the distinct
  validation public states ordered by `public_state_identity` and keeps the
  first 256 — 8 x 256 = 2,048;
- the benchmark takes 10 states per (stratum x colour x progress bucket)
  cell over the 48 cells, ordering each cell's distinct states by
  unresolved-piece count then identity and taking evenly spaced picks, so
  unresolved-count variation is covered without a draw.

Neither rule touches a seed stream, a wall clock or a worker count, and
both are computed from the recorded store rather than from a replay, so the
frozen sets can be rebuilt and re-verified from the store alone.

Purity is structural here too
-----------------------------
:func:`execute_request` rebuilds its position by replaying the game's public
action history from the initial setup on every call. It keeps no cache
between requests, holds no module-level state, constructs its own belief
request and its own sampler requests, and reads no worker id, process id,
ordinal-in-batch, path or clock into any derivation. Two calls with the same
`request_ordinal` therefore produce the same bytes whether they are the
first call of a fresh interpreter or the two-thousandth call of a pool
worker — which is the property the eight legs measure.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from ..training.phase11_contract import (
    BELIEF_REQUEST_VERSION,
    BELIEF_SAMPLER_VERSION,
    PROGRESS_BUCKET_NAMES,
    Phase11ContractError,
    RANK_COUNT,
    progress_bucket,
)
from ..training.phase11_seed import (
    BENCHMARK_STATES_PER_CELL,
    COLORS,
    OPPONENT_STRATA,
    REPRO_REQUEST_COUNT,
    phase11_benchmark_state_id,
    phase11_repro_request_id,
)

#: The frozen world count of one topology/replay request, from the Agent 1
#: `request_content` sentence ("sample ordinals 0..63").
REQUEST_WORLD_COUNT = 64

#: The frozen benchmark configurations, in the Agent 1 order, as world
#: counts. `forward_only` samples nothing.
BENCHMARK_CONFIGURATIONS = (
    ("forward_only", 0),
    ("forward_plus_16_worlds", 16),
    ("forward_plus_32_worlds", 32),
    ("forward_plus_64_worlds", 64),
)

#: The frozen benchmark warmups: 32 global, then 1 discarded per state.
BENCHMARK_GLOBAL_WARMUPS = 32
BENCHMARK_STATE_WARMUPS = 1

#: The gate quantity's configuration name.
GATE_CONFIGURATION = "forward_plus_64_worlds"

#: Requests per stratum in the frozen topology set.
REQUESTS_PER_STRATUM = REPRO_REQUEST_COUNT // len(OPPONENT_STRATA)


class Phase11ReproError(Phase11ContractError):
    """A frozen set could not be built, or a request could not be executed."""


# ---------------------------------------------------------------------------
# The recorded decision index, read from the Agent 2 prediction store
# ---------------------------------------------------------------------------


def decision_table(root, manifest: dict, cases: dict) -> "list[dict]":
    """Every recorded observer decision, as the frozen selection rules see it.

    Built from the store's public shards and the frozen bank — never from a
    replay, and never from an outcome field. One row per decision, carrying
    only the four facts the two selection rules order on: the stratum, the
    observer colour, the progress bucket and the unresolved-piece count,
    plus the recorded public-state identity that names the state itself.
    """
    from .phase11_records import read_public_shard

    rows = []
    for entry in sorted(manifest["games_index"], key=lambda item: item["game_id"]):
        game_id = entry["game_id"]
        case = cases[entry["case_id"]]
        game = case["games"][str(int(entry["game_index"]))]
        observer_color = game["observer_color"]
        arrays = read_public_shard(root, game_id)
        offsets = np.asarray(arrays["event_offset"], dtype=np.int64)
        identities = arrays["public_state_identity"]
        decisions = np.asarray(arrays["decision_index"], dtype=np.int64)
        for position in range(len(decisions)):
            unresolved = int(offsets[position + 1] - offsets[position])
            if unresolved <= 0:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "case_id": entry["case_id"],
                    "game_index": int(entry["game_index"]),
                    "observer_color": observer_color,
                    "opponent_stratum": entry["opponent_stratum"],
                    "opponent_setup_source": entry["opponent_setup_source"],
                    "decision_index": int(decisions[position]),
                    "public_state_identity": bytes(identities[position]).hex(),
                    "unresolved_pieces": unresolved,
                    "progress_bucket": progress_bucket(int(decisions[position])),
                }
            )
    return rows


def game_setups(cases: dict, case_id: str, game_index: int) -> dict:
    """The red/blue setups of one bank game, as the replay needs them."""
    game = cases[case_id]["games"][str(int(game_index))]
    observer_setup = [int(value) for value in game["observer"]["setup"]]
    opponent_setup = [int(value) for value in game["opponent"]["setup"]]
    if game["observer_color"] == "red":
        return {"red_setup": observer_setup, "blue_setup": opponent_setup}
    return {"red_setup": opponent_setup, "blue_setup": observer_setup}


# ---------------------------------------------------------------------------
# The two frozen sets
# ---------------------------------------------------------------------------


def _first_row_per_identity(rows) -> dict:
    """One representative row per distinct public-state identity.

    Ties are broken by `(game_id, decision_index)`, so "the distinct
    validation public states" is a well-defined set of rows and not a
    dictionary-insertion accident.
    """
    chosen: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (item["game_id"], item["decision_index"])):
        chosen.setdefault(row["public_state_identity"], row)
    return chosen


def frozen_repro_requests(rows) -> "list[dict]":
    """The 2,048 frozen topology requests, by the Agent 1 hash-order rule."""
    requests = []
    for stratum in OPPONENT_STRATA:
        distinct = _first_row_per_identity(
            row for row in rows if row["opponent_stratum"] == stratum
        )
        ordered = [distinct[key] for key in sorted(distinct)]
        if len(ordered) < REQUESTS_PER_STRATUM:
            raise Phase11ReproError(
                f"stratum {stratum!r} offers {len(ordered)} distinct public states, "
                f"the frozen rule needs {REQUESTS_PER_STRATUM}"
            )
        requests.extend(ordered[:REQUESTS_PER_STRATUM])
    if len(requests) != REPRO_REQUEST_COUNT:
        raise Phase11ReproError(
            f"the frozen request set holds {len(requests)} requests, expected "
            f"{REPRO_REQUEST_COUNT}"
        )
    frozen = []
    for ordinal, row in enumerate(requests):
        frozen.append(
            {
                "request_ordinal": ordinal,
                "request_id": phase11_repro_request_id(ordinal),
                **row,
            }
        )
    return frozen


def _evenly_spaced(values, count: int) -> list:
    if not values:
        return []
    take = min(count, len(values))
    return [values[(index * len(values)) // take] for index in range(take)]


def frozen_benchmark_states(rows) -> "list[dict]":
    """The frozen benchmark states, by the Agent 1 48-cell rule.

    A cell with fewer distinct states than the per-cell quota contributes
    what it has; the shortfall is recorded by the caller rather than being
    made up from another cell, because the rule's point is coverage of the
    cell, not a round number.
    """
    selected = []
    cells = []
    for stratum in OPPONENT_STRATA:
        for color in COLORS:
            for bucket in PROGRESS_BUCKET_NAMES:
                distinct = _first_row_per_identity(
                    row
                    for row in rows
                    if row["opponent_stratum"] == stratum
                    and row["observer_color"] == color
                    and row["progress_bucket"] == bucket
                )
                ordered = sorted(
                    distinct.values(),
                    key=lambda item: (
                        item["unresolved_pieces"],
                        item["public_state_identity"],
                    ),
                )
                picks = _evenly_spaced(ordered, BENCHMARK_STATES_PER_CELL)
                cells.append(
                    {
                        "stratum": stratum,
                        "observer_color": color,
                        "progress_bucket": bucket,
                        "distinct_states": len(ordered),
                        "selected": len(picks),
                    }
                )
                selected.extend(picks)
    states = []
    for ordinal, row in enumerate(selected):
        states.append(
            {
                "state_ordinal": ordinal,
                "benchmark_state_id": phase11_benchmark_state_id(ordinal),
                **row,
            }
        )
    return states, cells


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------


def request_digest(
    logits: dict, probabilities: dict, masks: dict, worlds: "list[dict]"
) -> str:
    """The canonical digest of one executed request.

    Beliefs, masks, worlds and provenance, over raw bytes and frozen field
    order. This is the only quantity the eight legs compare, and one
    differing byte in any of the four fails Gate G.
    """
    import json

    hasher = hashlib.sha256()
    hasher.update(b"phase11_repro_request_digest_v1")
    hasher.update(f"|slots={len(logits)}".encode())
    for slot in sorted(logits):
        hasher.update(f"|slot={int(slot)}".encode())
        hasher.update(np.asarray(logits[slot], dtype=np.float32).tobytes())
        hasher.update(np.asarray(probabilities[slot], dtype=np.float64).tobytes())
        mask = np.asarray(masks[slot], dtype=np.uint8)
        if mask.shape != (RANK_COUNT,):
            raise Phase11ReproError(f"slot {slot} mask has shape {mask.shape}")
        hasher.update(mask.tobytes())
    hasher.update(f"|worlds={len(worlds)}".encode())
    for world in worlds:
        hasher.update(b"|world|")
        hasher.update(
            json.dumps(
                {str(key): value for key, value in world.items()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return hasher.hexdigest()


def replay_state(spec: dict):
    """Rebuild one request's position from public bytes alone.

    The setups come from the frozen bank and the action history from the
    game's recorded public shard, both of which every player saw. Nothing
    survives between calls: the game is created and replayed from scratch,
    which is what makes a request's result independent of what ran before
    it.
    """
    from ..engine.constants import BLUE, RED
    from ..engine.state import create_game
    from ..engine.transition import apply_action
    from .match_spec import EVALUATION_RULES

    observer = RED if spec["observer_color"] == "red" else BLUE
    state = create_game(
        tuple(spec["red_setup"]),
        tuple(spec["blue_setup"]),
        rules=EVALUATION_RULES,
        game_id=spec["game_id"],
    )
    target = int(spec["decision_index"])
    for action in spec["action_history"]:
        if state.terminal:
            break
        if state.acting_player == observer and int(state.total_moves) == target:
            return state, observer
        apply_action(state, int(action))
    if (
        not state.terminal
        and state.acting_player == observer
        and int(state.total_moves) == target
    ):
        return state, observer
    raise Phase11ReproError(
        f"{spec['game_id']}: the replay never reached observer decision {target}"
    )


@dataclass(frozen=True)
class RequestResult:
    """One executed request: its digest, its parts, and its component times."""

    request_ordinal: int
    request_id: str
    public_state_identity: str
    hidden_pieces: int
    digest: str
    document_ns: int
    forward_ns: int
    sampling_ns: int
    total_ns: int
    worlds: int

    def as_row(self) -> dict:
        return {
            "request_ordinal": self.request_ordinal,
            "request_id": self.request_id,
            "public_state_identity": self.public_state_identity,
            "hidden_pieces": self.hidden_pieces,
            "digest": self.digest,
            "document_ns": self.document_ns,
            "forward_ns": self.forward_ns,
            "sampling_ns": self.sampling_ns,
            "total_ns": self.total_ns,
            "worlds": self.worlds,
        }


def execute_request(
    owner,
    spec: dict,
    *,
    world_count: int = REQUEST_WORLD_COUNT,
    state=None,
    observer=None,
    collect: bool = False,
):
    """One belief forward plus `world_count` complete worlds, timed.

    `state` may be supplied when the caller has already replayed the
    position (the benchmark does, so the replay is not measured); the
    topology legs pass nothing, so each leg replays every request itself.
    """
    from ..engine.observation import build_observation
    from .neural_worker import InferenceRequest
    from .phase11_belief import Phase11BeliefRequest, softmax_float64
    from .phase11_public_state import (
        build_public_state_document,
        hidden_opponent_pieces,
        legal_rank_mask,
    )
    from .phase11_sampler import Phase11SamplerRequest, sample_belief_world
    from .policy import PolicyRef, PolicyRequirements, build_policy_input

    if state is None:
        state, observer = replay_state(spec)

    requirements = PolicyRequirements(
        observation=True, legal_action_mask=True, public_view=True
    )
    reference = PolicyRef(
        policy_id="phase11_repro_observer", policy_version=BELIEF_REQUEST_VERSION
    )

    start = time.perf_counter_ns()
    policy_input = build_policy_input(
        state,
        policy=reference,
        policy_seed=0,
        requirements=requirements,
        match_id=spec["request_id"],
        game_id=spec["game_id"],
    )
    view = policy_input.require_public_view()
    observation = policy_input.require_observation()
    document = build_public_state_document(view, observation)
    belief_request = Phase11BeliefRequest(
        request_version=BELIEF_REQUEST_VERSION,
        request_id=spec["request_id"],
        observer_color=spec["observer_color"],
        public_state_document=document,
        observation=observation,
    )
    payload = InferenceRequest.from_policy_input(policy_input)
    after_document = time.perf_counter_ns()

    _response, prediction, _elapsed = owner.serve_decision(payload, belief_request)
    after_forward = time.perf_counter_ns()

    logits = prediction.belief_logits
    probabilities = {slot: softmax_float64(row) for slot, row in logits.items()}
    masks = {
        int(piece["piece_slot"]): legal_rank_mask(bool(piece["has_moved"]))
        for piece in hidden_opponent_pieces(document)
    }
    worlds = []
    for ordinal in range(int(world_count)):
        sampler_request = Phase11SamplerRequest(
            sampler_version=BELIEF_SAMPLER_VERSION,
            public_state_document=document,
            learned_probabilities=probabilities,
            sample_ordinal=ordinal,
        )
        worlds.append(sample_belief_world(sampler_request))
    end = time.perf_counter_ns()

    identity = document and belief_request.public_state_identity
    if identity != spec["public_state_identity"]:
        raise Phase11ReproError(
            f"{spec['request_id']}: replayed identity {identity} does not match the "
            f"frozen {spec['public_state_identity']}"
        )
    result = RequestResult(
        request_ordinal=int(spec["request_ordinal"]),
        request_id=spec["request_id"],
        public_state_identity=identity,
        hidden_pieces=len(logits),
        digest=request_digest(logits, probabilities, masks, worlds),
        document_ns=after_document - start,
        forward_ns=after_forward - after_document,
        sampling_ns=end - after_forward,
        total_ns=end - start,
        worlds=len(worlds),
    )
    if not collect:
        return result
    return result, {
        "document": document,
        "logits": logits,
        "probabilities": probabilities,
        "masks": masks,
        "worlds": worlds,
        "observation": np.array(observation, dtype=np.float32, copy=True)
        if observation is not None
        else None,
    }


def build_owner(export_path, *, device: str = "cpu", dtype: str = "float32"):
    """The accepted Phase 11 belief owner on the frozen benchmark backend."""
    from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from .neural_worker import DECISION_MODE_GREEDY
    from .phase11_belief import Phase11BeliefOwner

    return Phase11BeliefOwner(
        export_path,
        name="phase11_agent04",
        decision_mode=DECISION_MODE_GREEDY,
        device=device,
        dtype=dtype,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )


# ---------------------------------------------------------------------------
# Timing statistics
# ---------------------------------------------------------------------------


def timing_statistics(values_ms) -> dict:
    """Median/p90/p95/p99/max, on the accepted quantile definition."""
    from .statistics import quantile

    ordered = sorted(float(value) for value in values_ms)
    if not ordered:
        raise Phase11ReproError("no timing samples")
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "median_ms": quantile(ordered, 0.5),
        "p90_ms": quantile(ordered, 0.90),
        "p95_ms": quantile(ordered, 0.95),
        "p99_ms": quantile(ordered, 0.99),
        "max_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
        "all_finite": all(
            value == value and value not in (float("inf"), float("-inf"))
            for value in ordered
        ),
    }


def resident_set_bytes() -> int:
    """Peak RSS of this process, in bytes."""
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


__all__ = [
    "BENCHMARK_CONFIGURATIONS",
    "BENCHMARK_GLOBAL_WARMUPS",
    "BENCHMARK_STATE_WARMUPS",
    "GATE_CONFIGURATION",
    "REQUESTS_PER_STRATUM",
    "REQUEST_WORLD_COUNT",
    "Phase11ReproError",
    "RequestResult",
    "build_owner",
    "decision_table",
    "execute_request",
    "frozen_benchmark_states",
    "frozen_repro_requests",
    "game_setups",
    "replay_state",
    "request_digest",
    "resident_set_bytes",
    "timing_statistics",
]
