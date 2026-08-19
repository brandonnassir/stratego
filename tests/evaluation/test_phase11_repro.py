"""Phase 11 Agent 4: the frozen request set, its digest, and the benchmark.

The topology legs and the runtime benchmark are only as good as the two
things tested here: the selection rules must be deterministic hash-order
rules that consume no randomness, and the canonical request digest must
move whenever any of the four things it covers moves. Both are checked on
synthetic decision tables where the correct answer is known, so neither
needs the recorded store to be present.
"""

import numpy as np
import pytest

from stratego.evaluation.phase11_repro import (
    BENCHMARK_CONFIGURATIONS,
    BENCHMARK_GLOBAL_WARMUPS,
    BENCHMARK_STATE_WARMUPS,
    GATE_CONFIGURATION,
    REQUESTS_PER_STRATUM,
    REQUEST_WORLD_COUNT,
    Phase11ReproError,
    frozen_benchmark_states,
    frozen_repro_requests,
    request_digest,
    resident_set_bytes,
    timing_statistics,
)
from stratego.training.phase11_contract import (
    PROGRESS_BUCKET_NAMES,
    RANK_COUNT,
    REPRODUCIBILITY_TOPOLOGY_LEGS,
    RUNTIME_BENCHMARK_CONFIGURATION,
)
from stratego.training.phase11_seed import (
    BENCHMARK_CELL_COUNT,
    BENCHMARK_STATES_PER_CELL,
    BENCHMARK_STATE_COUNT,
    COLORS,
    OPPONENT_STRATA,
    REPRO_REQUEST_COUNT,
)


def synthetic_rows(per_cell=BENCHMARK_STATES_PER_CELL * 6):
    """A decision table with enough distinct states in every frozen cell."""
    rows = []
    counter = 0
    for stratum in OPPONENT_STRATA:
        for color in COLORS:
            for bucket, decision in zip(PROGRESS_BUCKET_NAMES, (5, 60, 200)):
                for index in range(per_cell):
                    counter += 1
                    rows.append(
                        {
                            "game_id": f"g-{stratum}-{color}-{bucket}-{index:03d}",
                            "case_id": f"c-{stratum}",
                            "game_index": index % 2,
                            "observer_color": color,
                            "opponent_stratum": stratum,
                            "opponent_setup_source": "p10d" if index % 2 else "neutral",
                            "decision_index": decision + index,
                            "public_state_identity": f"{counter:064x}",
                            "unresolved_pieces": 1 + (index % 40),
                            "progress_bucket": bucket,
                        }
                    )
    return rows


# ---------------------------------------------------------------------------
# The frozen topology request set
# ---------------------------------------------------------------------------


def test_the_request_set_is_the_frozen_size_and_balance():
    requests = frozen_repro_requests(synthetic_rows())
    assert len(requests) == REPRO_REQUEST_COUNT == 2048
    by_stratum = {stratum: 0 for stratum in OPPONENT_STRATA}
    for request in requests:
        by_stratum[request["opponent_stratum"]] += 1
    assert set(by_stratum.values()) == {REQUESTS_PER_STRATUM}
    assert len({request["public_state_identity"] for request in requests}) == len(
        requests
    )


def test_the_request_set_is_hash_ordered_within_each_stratum():
    requests = frozen_repro_requests(synthetic_rows())
    for stratum in OPPONENT_STRATA:
        identities = [
            request["public_state_identity"]
            for request in requests
            if request["opponent_stratum"] == stratum
        ]
        assert identities == sorted(identities)


def test_the_request_set_consumes_no_randomness_and_ignores_row_order():
    rows = synthetic_rows()
    forward = frozen_repro_requests(rows)
    backward = frozen_repro_requests(list(reversed(rows)))
    assert [row["public_state_identity"] for row in forward] == [
        row["public_state_identity"] for row in backward
    ]
    assert [row["request_id"] for row in forward] == [
        row["request_id"] for row in backward
    ]


def test_a_stratum_short_of_states_is_refused_rather_than_padded():
    rows = [row for row in synthetic_rows() if row["opponent_stratum"] != "miner_rush"]
    rows.extend(
        row for row in synthetic_rows() if row["opponent_stratum"] == "miner_rush"
    )
    thin = [
        row
        for row in rows
        if row["opponent_stratum"] != "miner_rush"
        or int(row["public_state_identity"], 16) % 97 == 0
    ]
    with pytest.raises(Phase11ReproError):
        frozen_repro_requests(thin)


def test_request_ids_are_the_frozen_ordinal_identities():
    requests = frozen_repro_requests(synthetic_rows())
    for ordinal, request in enumerate(requests):
        assert request["request_ordinal"] == ordinal
        assert request["request_id"].endswith(f"n={ordinal:05d}")


# ---------------------------------------------------------------------------
# The frozen benchmark state set
# ---------------------------------------------------------------------------


def test_the_benchmark_covers_every_cell_at_the_frozen_quota():
    states, cells = frozen_benchmark_states(synthetic_rows())
    assert len(states) == BENCHMARK_STATE_COUNT == 480
    assert len(cells) == BENCHMARK_CELL_COUNT == 48
    assert all(cell["selected"] == BENCHMARK_STATES_PER_CELL for cell in cells)
    assert len({state["public_state_identity"] for state in states}) == len(states)


def test_the_benchmark_orders_each_cell_by_unresolved_count_then_identity():
    rows = synthetic_rows()
    states, _cells = frozen_benchmark_states(rows)
    cell = [
        state
        for state in states
        if state["opponent_stratum"] == OPPONENT_STRATA[0]
        and state["observer_color"] == COLORS[0]
        and state["progress_bucket"] == PROGRESS_BUCKET_NAMES[0]
    ]
    keys = [
        (state["unresolved_pieces"], state["public_state_identity"]) for state in cell
    ]
    assert keys == sorted(keys)


def test_the_benchmark_spans_unresolved_piece_counts():
    states, _cells = frozen_benchmark_states(synthetic_rows())
    counts = {state["unresolved_pieces"] for state in states}
    assert len(counts) >= 8


def test_a_short_cell_contributes_what_it_has_and_is_recorded():
    rows = [
        row
        for row in synthetic_rows()
        if not (
            row["opponent_stratum"] == OPPONENT_STRATA[1]
            and row["observer_color"] == COLORS[1]
            and row["progress_bucket"] == PROGRESS_BUCKET_NAMES[2]
            and row["unresolved_pieces"] > 4
        )
    ]
    states, cells = frozen_benchmark_states(rows)
    short = [cell for cell in cells if cell["selected"] < BENCHMARK_STATES_PER_CELL]
    assert short, "the thinned cell should be recorded as short"
    assert len(states) < BENCHMARK_STATE_COUNT


def test_benchmark_state_ids_are_the_frozen_ordinal_identities():
    states, _cells = frozen_benchmark_states(synthetic_rows())
    for ordinal, state in enumerate(states):
        assert state["state_ordinal"] == ordinal
        assert state["benchmark_state_id"].endswith(f"n={ordinal:03d}")


# ---------------------------------------------------------------------------
# The canonical request digest
# ---------------------------------------------------------------------------


def a_request():
    logits = {3: np.arange(RANK_COUNT, dtype=np.float32)}
    probabilities = {3: np.full(RANK_COUNT, 1.0 / RANK_COUNT, dtype=np.float64)}
    masks = {3: tuple([1] * RANK_COUNT)}
    worlds = [
        {
            "sample_token": "t",
            "sampler_version": "belief_sampler_v1",
            "public_state_identity": "ab" * 32,
            "belief_model_label": "selfplay_c1_v1",
            "sample_ordinal": ordinal,
            "piece_order": [3],
            "fallback_steps": [],
            "assignment": {3: 4},
            "dead_end_events": 0,
        }
        for ordinal in range(2)
    ]
    return logits, probabilities, masks, worlds


def test_the_request_digest_is_stable_for_identical_inputs():
    first = request_digest(*a_request())
    second = request_digest(*a_request())
    assert first == second


@pytest.mark.parametrize(
    "mutation",
    ["logits", "probabilities", "masks", "world_assignment", "world_provenance"],
)
def test_the_request_digest_moves_on_any_covered_change(mutation):
    logits, probabilities, masks, worlds = a_request()
    baseline = request_digest(logits, probabilities, masks, worlds)
    if mutation == "logits":
        # Index 1 holds 1.0: a float32 ulp there survives the cast, where a
        # denormal step away from 0.0 would not.
        logits = {3: logits[3].copy()}
        logits[3][1] = np.nextafter(logits[3][1], np.float32(2.0))
    elif mutation == "probabilities":
        probabilities = {3: probabilities[3].copy()}
        probabilities[3][0] = np.nextafter(probabilities[3][0], 1.0)
    elif mutation == "masks":
        masks = {3: tuple([1] * 10 + [0, 0])}
    elif mutation == "world_assignment":
        worlds = [dict(world) for world in worlds]
        worlds[0]["assignment"] = {3: 5}
    else:
        worlds = [dict(world) for world in worlds]
        worlds[0]["sample_token"] = "other"
    assert request_digest(logits, probabilities, masks, worlds) != baseline


def test_the_request_digest_refuses_a_mask_of_the_wrong_width():
    logits, probabilities, _masks, worlds = a_request()
    with pytest.raises(Phase11ReproError):
        request_digest(logits, probabilities, {3: (1, 1, 1)}, worlds)


def test_dropping_a_world_changes_the_digest():
    logits, probabilities, masks, worlds = a_request()
    baseline = request_digest(logits, probabilities, masks, worlds)
    assert request_digest(logits, probabilities, masks, worlds[:1]) != baseline


# ---------------------------------------------------------------------------
# The frozen benchmark configuration and timing statistics
# ---------------------------------------------------------------------------


def test_the_measured_configurations_are_the_frozen_ones():
    names = [name for name, _ in BENCHMARK_CONFIGURATIONS]
    assert tuple(names) == tuple(
        RUNTIME_BENCHMARK_CONFIGURATION["measured_configurations"]
    )
    assert dict(BENCHMARK_CONFIGURATIONS)["forward_plus_64_worlds"] == 64
    assert GATE_CONFIGURATION == "forward_plus_64_worlds"
    assert REQUEST_WORLD_COUNT == 64


def test_the_frozen_backend_is_cpu_float32_single_thread():
    assert RUNTIME_BENCHMARK_CONFIGURATION["backend"] == "cpu"
    assert RUNTIME_BENCHMARK_CONFIGURATION["dtype"] == "float32"
    assert RUNTIME_BENCHMARK_CONFIGURATION["torch_threads"] == 1
    assert RUNTIME_BENCHMARK_CONFIGURATION["ceiling_ms"] == 500.0


def test_the_warmups_are_recorded_and_non_zero():
    assert BENCHMARK_GLOBAL_WARMUPS == 32
    assert BENCHMARK_STATE_WARMUPS == 1


def test_the_eight_topology_legs_are_the_frozen_ones():
    assert len(REPRODUCIBILITY_TOPOLOGY_LEGS) == 8
    assert "kill_resume_set_subtraction" in REPRODUCIBILITY_TOPOLOGY_LEGS
    assert "fresh_process" in REPRODUCIBILITY_TOPOLOGY_LEGS


def test_timing_statistics_are_the_accepted_quantiles():
    values = [float(index) for index in range(1, 101)]
    stats = timing_statistics(values)
    assert stats["count"] == 100
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0
    assert 50.0 <= stats["median_ms"] <= 51.0
    assert 95.0 <= stats["p95_ms"] <= 96.0
    assert stats["p90_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert stats["all_finite"] is True


def test_timing_statistics_refuse_an_empty_sample():
    with pytest.raises(Phase11ReproError):
        timing_statistics([])


def test_resident_set_bytes_is_a_positive_measurement():
    assert resident_set_bytes() > 0
