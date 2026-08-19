"""Phase 11 Agent 1: frozen seeds, identities and domain separation."""

import pytest

from stratego.training import phase11_seed as ps


# ---------------------------------------------------------------------------
# Root seeds and domains
# ---------------------------------------------------------------------------


def test_the_eight_root_seeds_are_exactly_the_common_contract():
    assert ps.CANONICAL_PHASE11_SEEDS == {
        "phase11_master_seed": 2026081901,
        "bank_schedule_seed": 2026081902,
        "match_randomness_seed": 2026081903,
        "world_sampling_seed": 2026081904,
        "information_safety_seed": 2026081905,
        "repro_runtime_seed": 2026081906,
        "validation_bootstrap_seed": 2026081907,
        "test_bootstrap_seed": 2026081908,
    }


def test_every_domain_has_a_root_and_documented_parts():
    document = ps.seed_derivation_document()
    assert set(document["domains"]) == set(ps.STREAM_DOMAINS)
    for domain in ps.STREAM_DOMAINS:
        assert document["domains"][domain]["root_seed"] == ps.DOMAIN_ROOTS[domain]
        assert document["domains"][domain]["identity_parts"]


def test_domain_separation_two_domains_same_parts_differ():
    a = ps.derive_phase11_seed(ps.DOMAIN_BANK_OBSERVER_SETUP, "x", 0)
    b = ps.derive_phase11_seed(ps.DOMAIN_BANK_OPPONENT_SETUP, "x", 0)
    c = ps.derive_phase11_seed(ps.DOMAIN_BANK_MATCH, "x", 0)
    assert len({a, b, c}) == 3


def test_derivation_is_deterministic_and_part_sensitive():
    a = ps.derive_phase11_seed(ps.DOMAIN_WORLD_ORDER, "token", 3)
    assert a == ps.derive_phase11_seed(ps.DOMAIN_WORLD_ORDER, "token", 3)
    assert a != ps.derive_phase11_seed(ps.DOMAIN_WORLD_ORDER, "token", 4)
    assert a != ps.derive_phase11_seed(ps.DOMAIN_WORLD_ORDER, "tokem", 3)
    assert 0 <= a < 1 << 63


def test_colon_in_string_parts_is_rejected():
    with pytest.raises(ps.Phase11SeedError):
        ps.derive_phase11_seed(ps.DOMAIN_WORLD_ORDER, "a:b", 0)


def test_unknown_domain_and_bad_part_types_are_rejected():
    with pytest.raises(ps.Phase11SeedError):
        ps.derive_phase11_seed("no_such_domain", 1)
    with pytest.raises(ps.Phase11SeedError):
        ps.derive_phase11_seed(ps.DOMAIN_BANK_MATCH, 1.5)
    with pytest.raises(ps.Phase11SeedError):
        ps.derive_phase11_seed(ps.DOMAIN_BANK_MATCH, True)


def test_unit_uniform_is_exact_and_the_tail_edge_is_documented():
    assert ps.unit_uniform(0) == 0.0
    assert ps.unit_uniform(1 << 62) == 0.5
    # The frozen convention's honest edge: the extreme top of the seed range
    # rounds to exactly 1.0 under float64, which the frozen inverse-CDF
    # walks absorb with their last-element tail guard.
    assert ps.unit_uniform((1 << 63) - 1) == 1.0
    assert ps.unit_uniform((1 << 63) - (1 << 10)) < 1.0
    with pytest.raises(ps.Phase11SeedError):
        ps.unit_uniform(-1)


# ---------------------------------------------------------------------------
# Case, game and prediction identity
# ---------------------------------------------------------------------------


def test_case_id_round_trip():
    case_id = ps.phase11_case_id(
        "phase11_validation_bank_v1", ps.STRATUM_SCOUT_RUSH, ps.SOURCE_P10D, 17
    )
    assert case_id == (
        "phase11_validation_bank_v1|ms=2026081901|st=scout_rush|src=p10d|c=017"
    )
    fields = ps.parse_phase11_case_id(case_id)
    assert fields["bank_version"] == "phase11_validation_bank_v1"
    assert fields["stratum"] == "scout_rush"
    assert fields["setup_source"] == "p10d"
    assert fields["case_ordinal"] == 17


def test_case_id_rejects_bad_inputs():
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_case_id("phase11_validation_bank_v1", "no_such_stratum", "p10d", 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_case_id("phase11_validation_bank_v1", ps.STRATUM_BASIC, "learned", 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_case_id("phase11_validation_bank_v1", ps.STRATUM_BASIC, "p10d", 1000)
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_case_id("setup_bank_v1", ps.STRATUM_BASIC, "p10d", 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.parse_phase11_case_id(
            "phase11_validation_bank_v1|ms=2026089999|st=basic_rule|src=p10d|c=000"
        )


def test_game_id_round_trip_and_colour_pairing():
    case_id = ps.phase11_case_id(
        "phase11_test_bank_v1", ps.STRATUM_PHASE9, ps.SOURCE_NEUTRAL, 127
    )
    game0 = ps.phase11_game_id(case_id, 0)
    game1 = ps.phase11_game_id(case_id, 1)
    assert game0.endswith("|g=0") and game1.endswith("|g=1")
    fields0 = ps.parse_phase11_game_id(game0)
    fields1 = ps.parse_phase11_game_id(game1)
    assert fields0["observer_color"] == "red" and fields0["opponent_color"] == "blue"
    assert fields1["observer_color"] == "blue" and fields1["opponent_color"] == "red"
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_game_id(case_id, 2)


def test_prediction_id_round_trip():
    case_id = ps.phase11_case_id(
        "phase11_validation_bank_v1", ps.STRATUM_MINER_RUSH, ps.SOURCE_NEUTRAL, 3
    )
    game_id = ps.phase11_game_id(case_id, 1)
    prediction_id = ps.phase11_prediction_id(game_id, 217, 39)
    assert prediction_id.endswith("|d=0217|p=39")
    fields = ps.parse_phase11_prediction_id(prediction_id)
    assert fields["decision_index"] == 217
    assert fields["piece_slot"] == 39
    assert fields["observer_color"] == "blue"
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_prediction_id(game_id, 10_000, 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_prediction_id(game_id, 0, 40)


# ---------------------------------------------------------------------------
# Bank streams
# ---------------------------------------------------------------------------


def _sample_case_id():
    return ps.phase11_case_id(
        "phase11_validation_bank_v1", ps.STRATUM_TACTICAL, ps.SOURCE_P10D, 5
    )


def test_case_setup_seeds_are_role_and_game_separated():
    case_id = _sample_case_id()
    seeds = {
        (game, role): ps.case_setup_seed(case_id, game, role)
        for game in ps.CASE_GAME_INDICES
        for role in ps.SETUP_ROLES
    }
    assert len(set(seeds.values())) == 4
    with pytest.raises(ps.Phase11SeedError):
        ps.case_setup_seed(case_id, 0, "spectator")
    with pytest.raises(ps.Phase11SeedError):
        ps.case_setup_seed(case_id, 2, ps.ROLE_OBSERVER)


def test_match_seed_requires_a_game_id():
    case_id = _sample_case_id()
    game0 = ps.phase11_game_id(case_id, 0)
    game1 = ps.phase11_game_id(case_id, 1)
    assert ps.game_match_seed(game0) != ps.game_match_seed(game1)
    with pytest.raises(ps.Phase11SeedError):
        ps.game_match_seed(case_id)


# ---------------------------------------------------------------------------
# World-sample identity
# ---------------------------------------------------------------------------


def test_sample_token_round_trip_and_streams():
    state = "ab" * 32
    token = ps.phase11_sample_token("belief_sampler_v1", state, 63)
    fields = ps.parse_phase11_sample_token(token)
    assert fields["sampler_version"] == "belief_sampler_v1"
    assert fields["public_state_identity"] == state
    assert fields["sample_ordinal"] == 63
    seed = ps.world_sample_seed(token)
    assert seed == ps.world_sample_seed(token)
    other = ps.phase11_sample_token("count_uniform_world_sampler_v1", state, 63)
    assert ps.world_sample_seed(other) != seed
    keys = {slot: ps.world_order_key(token, slot) for slot in range(40)}
    assert len(set(keys.values())) == 40
    uniforms = [ps.world_categorical_uniform(token, step) for step in range(40)]
    assert all(0.0 <= value < 1.0 for value in uniforms)
    assert len(set(uniforms)) == 40


def test_sample_token_rejects_bad_inputs():
    state = "cd" * 32
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_sample_token("Belief", state, 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_sample_token("belief_sampler_v1", "not-hex", 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_sample_token("belief_sampler_v1", state, 100_000)
    token = ps.phase11_sample_token("belief_sampler_v1", state, 0)
    with pytest.raises(ps.Phase11SeedError):
        ps.world_order_key(token, 40)
    with pytest.raises(ps.Phase11SeedError):
        ps.world_categorical_uniform(token, -1)


# ---------------------------------------------------------------------------
# Safety, reproducibility, benchmark, bootstrap
# ---------------------------------------------------------------------------


def test_safety_trial_identity_and_purposes():
    trial = ps.phase11_safety_trial_id(49_999)
    fields = ps.parse_phase11_safety_trial_id(trial)
    assert fields["trial_ordinal"] == 49_999
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_safety_trial_id(50_000)
    seeds = {
        purpose: ps.safety_trial_seed(trial, purpose, 0)
        for purpose in ps.SAFETY_PURPOSES
    }
    assert len(set(seeds.values())) == 3
    with pytest.raises(ps.Phase11SeedError):
        ps.safety_trial_seed(trial, "state_pick", 0)


def test_repro_and_benchmark_identity():
    request = ps.phase11_repro_request_id(2_047)
    assert ps.parse_phase11_repro_request_id(request)["request_ordinal"] == 2_047
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_repro_request_id(2_048)
    state = ps.phase11_benchmark_state_id(479)
    assert state.endswith("|n=479")
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_benchmark_state_id(480)
    assert ps.BENCHMARK_STATE_COUNT == 480 == ps.BENCHMARK_STATES_PER_CELL * ps.BENCHMARK_CELL_COUNT
    assert ps.repro_schedule_seed("replay", 0) != ps.benchmark_seed("replay", 0)


def test_bootstrap_roots_and_stream_separation():
    assert ps.bootstrap_root("validation") == 2026081907
    assert ps.bootstrap_root("test") == 2026081908
    with pytest.raises(ps.Phase11SeedError):
        ps.bootstrap_root("train")
    validation = ps.bootstrap_stream_seed("validation", "ce_delta")
    test = ps.bootstrap_stream_seed("test", "ce_delta")
    stratum = ps.bootstrap_stream_seed("test", "ce_delta|st=scout_rush")
    assert len({validation, test, stratum}) == 3


# ---------------------------------------------------------------------------
# Soak identity
# ---------------------------------------------------------------------------


def test_soak_arithmetic_and_identity():
    assert ps.SOAK_GAME_COUNT == 1_024
    assert ps.SOAK_REQUEST_COUNT == 8_192
    game = ps.phase11_soak_game_id(ps.STRATUM_INFORMATION_MISER, 127)
    fields = ps.parse_phase11_soak_game_id(game)
    assert fields["observer_color"] == "blue"  # odd ordinal
    assert ps.parse_phase11_soak_game_id(
        ps.phase11_soak_game_id(ps.STRATUM_BASIC, 0)
    )["observer_color"] == "red"
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_soak_game_id(ps.STRATUM_BASIC, 128)
    request = ps.phase11_soak_request_id(game, 7)
    assert ps.parse_phase11_soak_request_id(request)["request_ordinal"] == 7
    with pytest.raises(ps.Phase11SeedError):
        ps.phase11_soak_request_id(game, 8)
    assert ps.soak_setup_seed(game, ps.ROLE_OBSERVER) != ps.soak_setup_seed(
        game, ps.ROLE_OPPONENT
    )
    assert ps.soak_match_seed(game) == ps.soak_match_seed(game)


def test_soak_streams_cannot_collide_with_bank_streams_by_construction():
    """Distinct domain tokens, same roots — the frozen reading, proved."""
    case_id = _sample_case_id()
    game = ps.phase11_soak_game_id(ps.STRATUM_TACTICAL, 5)
    bank = {ps.case_setup_seed(case_id, g, r) for g in (0, 1) for r in ps.SETUP_ROLES}
    soak = {ps.soak_setup_seed(game, r) for r in ps.SETUP_ROLES}
    assert not bank & soak


# ---------------------------------------------------------------------------
# Collision audit
# ---------------------------------------------------------------------------


def test_stream_collision_audit_reports_duplicates_and_cross_collisions():
    clean = ps.stream_collision_audit({"a": [1, 2, 3], "b": [4, 5]})
    assert clean["no_collisions"] and clean["total_seeds"] == 5
    dirty = ps.stream_collision_audit({"a": [1, 1], "b": [1]})
    assert not dirty["no_collisions"]
    assert any("duplicate" in finding for finding in dirty["findings"])
    assert any("collides" in finding for finding in dirty["findings"])
