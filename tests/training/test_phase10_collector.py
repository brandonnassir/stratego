"""The Phase 10 corpus collector: identity, provenance and the audits.

The neural half of collection — the exported Phase 9 weights, the greedy
forward passes, the 16,384 real games — is evidence the Agent 2 harness
produces once and records in its artifacts; the artifact tests check that.
What is checked here is everything that decides *what a record means*:

- a corpus match's identity is a pure function of its logical game id;
- both sides resolve to the scheduled family, from the train split, through
  the untouched frozen sampler;
- a built record carries the frozen 27 fields, keeps its setup and outcome
  halves apart, and rebuilds its setups from provenance alone;
- the balance audit counts what the instruction says it counts, and fails
  when the corpus is wrong rather than only when it is empty.

None of it loads a checkpoint, so the whole module runs in well under a
second.
"""

import pytest

from stratego.training import phase10_collector as collector
from stratego.training import phase10_outcome_store as store
from stratego.training.phase10_schedule import (
    CORPUS_SPLIT,
    GAMES_PER_ORDERED_PAIR,
    ORDERED_FAMILY_PAIRS,
    TOTAL_CORPUS_GAMES,
    enumerate_schedule,
    ordered_family_pairs,
    rebuild_game,
)


@pytest.fixture(scope="module")
def library():
    from stratego.setups.sampler import load_library_index

    return load_library_index()


@pytest.fixture(scope="module")
def identity(library):
    from stratego.training import phase10_contract as contract
    from stratego.training.phase10_schedule import corpus_contract_document, schedule_digest

    return {
        "library_content_digest": library.content_digest,
        "corpus_contract_digest": contract.document_digest(corpus_contract_document()),
        "outcome_schedule_digest": schedule_digest(),
        "contract_bundle_digest": contract.contract_bundle_digest(),
        "phase9_checkpoint_sha256": "d" * 64,
        "phase9_model_state_digest": "e" * 64,
    }


class StubResult:
    """The primitive outcome fields `build_record` reads off a `MatchResult`."""

    def __init__(self, winner, draw, plies, reason="flag_capture"):
        from stratego.engine.constants import RED

        self.winner = winner
        self.draw = draw
        self.plies = plies
        self.decisions = plies
        self.terminal_reason = reason
        self.candidate_color = RED


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_schedule_index_covers_the_whole_frozen_schedule():
    schedule = enumerate_schedule()
    assert len(schedule) == TOTAL_CORPUS_GAMES
    assert collector.schedule_index(schedule[0].game_id) == 0
    assert collector.schedule_index(schedule[-1].game_id) == TOTAL_CORPUS_GAMES - 1


def test_an_unscheduled_game_id_has_no_index():
    with pytest.raises(collector.Phase10CollectorError, match="not a scheduled"):
        collector.schedule_index("phase10_outcome_v1|ms=2026081801|rf=F00|bf=F00|g=99")


def test_match_identity_is_a_pure_function_of_the_game_id():
    game_id = enumerate_schedule()[7].game_id
    first = collector.corpus_match_spec(game_id)
    second = collector.corpus_match_spec(game_id)
    assert first.match_id == second.match_id
    assert first.paired_unit_id == second.paired_unit_id
    assert first.root_seed == rebuild_game(game_id).match_seed


def test_both_sides_of_a_corpus_game_name_the_same_policy():
    spec = collector.corpus_match_spec(enumerate_schedule()[0].game_id)
    assert spec.candidate == spec.opponent == collector.corpus_policy_ref()


def test_distinct_games_get_distinct_match_identities():
    schedule = enumerate_schedule()
    ids = {collector.corpus_match_spec(game.game_id).match_id for game in schedule[:256]}
    assert len(ids) == 256


def test_the_policy_reference_pins_the_decision_rule_and_dtype():
    ref = collector.corpus_policy_ref()
    assert collector.CORPUS_MOVE_POLICY_ID in ref.policy_id
    assert "greedy" in ref.policy_id
    assert ref.policy_version.endswith("+float32")


# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 4_097, 9_000, TOTAL_CORPUS_GAMES - 1])
def test_resolved_sides_match_the_scheduled_families_from_the_train_split(library, position):
    game = enumerate_schedule()[position]
    sides = collector.resolve_game_setups(game.game_id, library)
    assert sides["red"]["sampled"].family_id == game.red_family
    assert sides["blue"]["sampled"].family_id == game.blue_family
    for color in ("red", "blue"):
        sampled = sides[color]["sampled"]
        assert sampled.split == CORPUS_SPLIT
        assert library.base(sampled.base_setup_id).split == CORPUS_SPLIT
        assert sampled.provenance["sampler_profile"] == "neutral_v1"


def test_side_resolution_is_deterministic(library):
    game_id = enumerate_schedule()[123].game_id
    first = collector.resolve_game_setups(game_id, library)
    second = collector.resolve_game_setups(game_id, library)
    for color in ("red", "blue"):
        assert first[color]["seed"] == second[color]["seed"]
        assert first[color]["attempt"] == second[color]["attempt"]
        assert first[color]["sampled"].canonical == second[color]["sampled"].canonical


def test_resolved_setups_rebuild_from_provenance_alone(library):
    from stratego.setups.sampler import rebuild_from_provenance

    sides = collector.resolve_game_setups(enumerate_schedule()[500].game_id, library)
    for color in ("red", "blue"):
        sampled = sides[color]["sampled"]
        rebuilt = rebuild_from_provenance(sampled.provenance, library)
        assert rebuilt.canonical == sampled.canonical
        assert rebuilt.base_setup_id == sampled.base_setup_id


def test_both_sides_produce_engine_ready_setups(library):
    from stratego.engine.constants import BLUE, RED
    from stratego.engine.state import create_game

    game = enumerate_schedule()[0]
    sides = collector.resolve_game_setups(game.game_id, library)
    state = create_game(
        sides["red"]["sampled"].oriented(RED), sides["blue"]["sampled"].oriented(BLUE)
    )
    assert not state.terminal


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_result_token_maps_every_outcome():
    from stratego.engine.constants import BLUE, RED

    assert collector.result_token(StubResult(RED, False, 10)) == collector.RESULT_RED_WIN
    assert collector.result_token(StubResult(BLUE, False, 10)) == collector.RESULT_RED_LOSS
    assert collector.result_token(StubResult(None, True, 10)) == collector.RESULT_DRAW


def build(position, result, library, identity):
    game = enumerate_schedule()[position]
    sides = collector.resolve_game_setups(game.game_id, library)
    return collector.build_record(game.game_id, result, sides, index=library, identity=identity)


def test_a_built_record_carries_every_frozen_field(library, identity):
    from stratego.engine.constants import RED

    record = build(0, StubResult(RED, False, 314), library, identity)
    assembled = dict(record["setup"])
    assembled.update(record["outcome"])
    for name in store.FROZEN_RECORD_FIELDS:
        if name in store.DERIVED_RECORD_FIELDS:
            continue
        assert name in assembled, name


def test_a_built_record_records_the_scheduled_identity(library, identity):
    from stratego.engine.constants import BLUE

    game = enumerate_schedule()[4_000]
    record = build(4_000, StubResult(BLUE, False, 88), library, identity)
    setup = record["setup"]
    assert setup["game_id"] == game.game_id
    assert setup["red_family"] == game.red_family
    assert setup["blue_family"] == game.blue_family
    assert setup["ordinal"] == game.ordinal
    assert setup["match_seed"] == game.match_seed
    assert setup["split"] == CORPUS_SPLIT
    assert record["outcome"]["result"] == collector.RESULT_RED_LOSS
    assert record["outcome"]["red_score"] == 0.0
    assert record["outcome"]["winner"] == "blue"


def test_a_built_record_carries_no_strength_signal(library, identity):
    from stratego.engine.constants import RED

    record = build(0, StubResult(RED, False, 10), library, identity)
    text = store.canonical_json(record).lower()
    for forbidden in ("value_logit", "policy_logit", "win_probability", "elo", "score_estimate"):
        assert forbidden not in text
    # No physical path may appear in a record either.
    assert "/volumes/" not in text
    assert "/users/" not in text


def test_trait_identity_names_both_the_base_and_the_descendant(library):
    sides = collector.resolve_game_setups(enumerate_schedule()[0].game_id, library)
    trait = collector.trait_identity(sides["red"]["sampled"], library)
    assert set(trait) == {"trait_schema_version", "base_trait_digest", "final_trait_digest"}
    assert len(trait["base_trait_digest"]) == 64


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def test_partition_covers_every_game_exactly_once():
    ids = [game.game_id for game in enumerate_schedule()[:100]]
    for worker_count in (1, 3, 7, 12):
        buckets = collector.partition(ids, worker_count)
        assert sum(len(bucket) for bucket in buckets) == len(ids)
        assert sorted(sum(buckets, [])) == sorted(ids)


def test_partition_refuses_a_nonsensical_worker_count():
    with pytest.raises(collector.Phase10CollectorError, match="at least 1"):
        collector.partition(["a"], 0)


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _mini_corpus(root, library, identity, *, games=8):
    """A structurally real store of `games` records, no checkpoint involved."""
    from stratego.engine.constants import BLUE, RED

    writer = store.OutcomeWriter(root, segment=0, worker_id=0)
    schedule = enumerate_schedule()
    for position in range(games):
        game = schedule[position]
        sides = collector.resolve_game_setups(game.game_id, library)
        result = StubResult([RED, BLUE, None][position % 3], position % 3 == 2, 100 + position)
        writer.write_record(
            collector.build_record(game.game_id, result, sides, index=library, identity=identity)
        )
    writer.close()


def test_balance_audit_reads_a_real_store(tmp_path, library, identity):
    _mini_corpus(tmp_path, library, identity)
    audit = collector.audit_corpus_balance(tmp_path)
    # A partial corpus is not 16,384 games, so the size checks are false and
    # everything about the records themselves is true.
    assert not audit["checks"]["total_games_exact"]
    assert audit["checks"]["train_split_violations_zero"]
    assert audit["checks"]["setup_provenance_mismatches_zero"]
    assert audit["checks"]["duplicate_game_ids_zero"]
    assert audit["checks"]["duplicate_commit_identities_zero"]
    assert audit["checks"]["every_game_scheduled"]
    assert audit["committed_games"] == 8
    assert sum(audit["result_counts"].values()) == 8


def test_balance_audit_catches_a_provenance_mismatch(tmp_path, library, identity):
    """A record whose stored base id contradicts its own provenance is a finding."""
    from stratego.engine.constants import RED

    game = enumerate_schedule()[0]
    sides = collector.resolve_game_setups(game.game_id, library)
    record = collector.build_record(
        game.game_id, StubResult(RED, False, 10), sides, index=library, identity=identity
    )
    record["setup"]["red_base_setup_id"] = "setup_library_v1:F00:1"
    writer = store.OutcomeWriter(tmp_path, segment=0, worker_id=0)
    writer.write_record(record)
    writer.close()

    audit = collector.audit_corpus_balance(tmp_path)
    assert not audit["checks"]["setup_provenance_mismatches_zero"]
    assert not audit["all_pass"]


def test_build_record_refuses_a_side_whose_family_contradicts_the_schedule(library, identity):
    from stratego.engine.constants import RED

    schedule = enumerate_schedule()
    sides = collector.resolve_game_setups(schedule[0].game_id, library)
    other = collector.resolve_game_setups(schedule[4_000].game_id, library)
    sides["red"] = other["red"]
    with pytest.raises(collector.Phase10CollectorError, match="contradict the schedule"):
        collector.build_record(
            schedule[0].game_id, StubResult(RED, False, 10), sides, index=library, identity=identity
        )


def test_reconstruction_audit_rebuilds_every_stored_setup(tmp_path, library, identity):
    _mini_corpus(tmp_path, library, identity)
    audit = collector.audit_setup_reconstruction(tmp_path)
    assert audit["all_pass"]
    assert audit["sides_rebuilt"] == 16


def test_family_pair_rows_summarize_without_ranking(tmp_path, library, identity):
    _mini_corpus(tmp_path, library, identity)
    rows = collector.family_pair_rows(tmp_path)
    assert rows
    total = sum(row["games"] for row in rows)
    assert total == 8
    for row in rows:
        assert row["red_wins"] + row["draws"] + row["red_losses"] == row["games"]
        assert 0.0 <= row["red_score"] <= 1.0


def test_the_frozen_schedule_arithmetic_is_what_the_collector_executes():
    assert ORDERED_FAMILY_PAIRS * GAMES_PER_ORDERED_PAIR == TOTAL_CORPUS_GAMES == 16_384
    assert len(set(ordered_family_pairs())) == ORDERED_FAMILY_PAIRS
