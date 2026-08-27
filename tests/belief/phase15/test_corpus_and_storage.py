"""Phase 15 Agent 1 sections 5-7: the collector and the store.

Two things must hold for the corpus to be evidence rather than data: the
public/privileged boundary must be a property of the code rather than a
convention, and a truncated run must still be a balanced sample of the
design. These tests check both, plus the storage round trip the trainer,
the metrics and the handoff all depend on.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from stratego.belief.phase15 import contract as C
from stratego.belief.phase15 import corpus as K
from stratego.belief.phase15 import storage as store
from stratego.belief.phase15 import verify as V
from stratego.belief.phase15.setups import Phase15SetupSources
from stratego.engine.constants import BLUE, RED
from stratego.evaluation.match_spec import EVALUATION_RULES


# ---------------------------------------------------------------------------
# The plan cycle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cycle():
    return K.plan_cycle("train")


def test_the_cycle_realises_the_mixture_exactly(cycle):
    total = len(cycle)
    for position, mixture in (
        (0, C.OBSERVER_MIXTURE),
        (1, C.OPPONENT_MIXTURE),
        (2, C.SETUP_MIXTURE),
    ):
        counts = collections.Counter(cell[position] for cell in cycle)
        for name, share in mixture.items():
            assert counts[name] / total == pytest.approx(share, abs=1e-12)


def test_a_prefix_of_the_cycle_is_still_close_to_the_design(cycle):
    prefix = cycle[: len(cycle) // 2]
    counts = collections.Counter(cell[1] for cell in prefix)
    for name, share in C.OPPONENT_MIXTURE.items():
        assert counts[name] / len(prefix) == pytest.approx(share, abs=0.02)


def test_the_cycle_is_deterministic(cycle):
    assert K.plan_cycle("train") == cycle


def test_different_splits_get_different_cycles(cycle):
    assert K.plan_cycle("development") != cycle


def test_a_mixture_that_is_not_a_whole_percentage_unit_is_refused(monkeypatch):
    monkeypatch.setattr(K, "OPPONENT_MIXTURE", {"p18": 0.333, "p24": 0.667})
    with pytest.raises(K.Phase15CorpusError):
        K.plan_cycle("train")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plans():
    return list(K.iter_plans("train", Phase15SetupSources(), limit=64))


def test_plans_alternate_observer_colour_inside_a_cell(plans):
    for index in range(0, len(plans) - 1, 2):
        left, right = plans[index], plans[index + 1]
        assert (left.observer_model, left.opponent, left.setup_source) == (
            right.observer_model,
            right.opponent,
            right.setup_source,
        )
        assert {left.observer_color, right.observer_color} == {"red", "blue"}


def test_every_planned_board_passes_the_orientation_assertion(plans):
    from stratego.belief.phase15.orientation import assert_engine_orientation
    from stratego.setups.identity import deorient_setup

    for plan in plans[:16]:
        assert_engine_orientation(
            deorient_setup(plan.red_setup, RED), plan.red_setup, RED
        )
        assert_engine_orientation(
            deorient_setup(plan.blue_setup, BLUE), plan.blue_setup, BLUE
        )


def test_plans_have_unique_game_ids(plans):
    identifiers = [plan.game_id for plan in plans]
    assert len(set(identifiers)) == len(identifiers)


def test_every_plan_is_a_pure_function_of_its_identity(plans):
    again = list(K.iter_plans("train", Phase15SetupSources(), limit=8))
    for left, right in zip(plans[:8], again):
        assert left == right


# ---------------------------------------------------------------------------
# Even spacing
# ---------------------------------------------------------------------------


def test_evenly_spaced_includes_both_endpoints():
    values = list(range(100))
    picked = K.evenly_spaced(values, 16)
    assert len(picked) == 16
    assert picked[0] == 0
    assert picked[-1] == 99


def test_evenly_spaced_returns_everything_when_it_is_short():
    assert K.evenly_spaced([1, 2, 3], 16) == [1, 2, 3]
    assert K.evenly_spaced([], 16) == []
    assert K.evenly_spaced([1, 2], 0) == []


def test_select_decisions_keeps_only_eligible_decisions():
    decisions = [
        {"ply": 0, "unresolved": 0},
        {"ply": 2, "unresolved": 3},
        {"ply": 4, "unresolved": 1},
    ]
    picked = K.select_decisions(decisions, 8)
    assert [row["ply"] for row in picked] == [2, 4]


# ---------------------------------------------------------------------------
# The accepted termination cap
# ---------------------------------------------------------------------------


def test_games_are_played_under_the_accepted_evaluation_rules(plans):
    spec = K.build_spec(plans[0], K.observer_ref())
    assert spec.rules is EVALUATION_RULES
    assert EVALUATION_RULES.battleless_move_limit == 200
    assert EVALUATION_RULES.absolute_move_limit == 4000


# ---------------------------------------------------------------------------
# The public/privileged boundary
# ---------------------------------------------------------------------------


def test_the_observer_policy_records_no_hidden_information():
    requirements = K.CorpusObserverPolicy.requirements
    assert requirements.observation is True
    assert requirements.public_view is True
    # A `PolicyRequirements` has no field through which truth could arrive.
    assert not any(
        "hidden" in name or "truth" in name
        for name in type(requirements).__dataclass_fields__
    )


def test_the_privileged_arrays_are_exactly_one_name():
    assert list(store.PRIVILEGED_ARRAYS) == ["true_rank"]
    assert "true_rank" not in store.PUBLIC_SAMPLE_ARRAYS
    assert "true_rank" not in store.PUBLIC_PIECE_ARRAYS


def test_loading_a_split_does_not_return_labels_by_default(tmp_path):
    data = _tiny_split(tmp_path)
    assert "true_rank" not in store.load_split(tmp_path, "train")
    assert "true_rank" in store.load_split(tmp_path, "train", labels=True)
    assert data["samples"] == 2


# ---------------------------------------------------------------------------
# Storage round trip
# ---------------------------------------------------------------------------


class _FakeResult:
    plies = 40
    winner = 0
    draw = False


def _fake_plan(ordinal: int, split: str = "train"):
    from stratego.belief.phase15.seeds import corpus_game_id, match_seed

    game_id = corpus_game_id(split, "p18", "p24", "neutral_v1", "red", ordinal)
    return K.CorpusGamePlan(
        game_id=game_id,
        split=split,
        ordinal=ordinal,
        observer_model="p18",
        opponent="p24",
        setup_source="neutral_v1",
        observer_color="red",
        opponent_color="blue",
        match_seed=match_seed(game_id),
        red_setup=tuple(range(40)),
        blue_setup=tuple(range(40)),
        observer_family_key="balanced_conventional",
        opponent_family_key="high_bomb_placement",
        observer_base_setup_id="setup_library_v1:F14:000",
        opponent_base_setup_id="setup_library_v1:F07:000",
        observer_setup_branch=None,
        opponent_setup_branch=None,
    )


def _fake_sample(plan, decision_index: int, pieces: int):
    counts = [4] * 12
    return {
        "game_id": plan.game_id,
        "split": plan.split,
        "observer_model": plan.observer_model,
        "opponent": plan.opponent,
        "setup_source": plan.setup_source,
        "observer_family_key": plan.observer_family_key,
        "opponent_family_key": plan.opponent_family_key,
        "observer_color": plan.observer_color,
        "decision_index": decision_index,
        "total_moves": decision_index,
        "public_state_identity": f"{plan.game_id}#{decision_index}",
        "observation": np.zeros(C.OBSERVATION_SHAPE, dtype=np.float32),
        "target_mask": np.zeros(100, dtype=bool),
        "remaining_counts": tuple(counts),
        "pieces": [
            {
                "piece_slot": index,
                "piece_square": index,
                "perspective_square": index,
                "piece_moved": False,
                "legal_rank_mask": np.ones(12, dtype=bool),
                "true_rank": index % 12,
            }
            for index in range(pieces)
        ],
    }


def _tiny_split(root, split: str = "train", games: int = 2, pieces: int = 3):
    writer = store.SplitWriter(root, split)
    for ordinal in range(games):
        plan = _fake_plan(ordinal, split)
        writer.add_game(
            plan,
            _FakeResult(),
            eligible=8,
            samples=[_fake_sample(plan, 2 * ordinal, pieces)],
        )
    return writer.close()


def test_a_split_round_trips_through_disk(tmp_path):
    block = _tiny_split(tmp_path, games=3, pieces=5)
    assert block["samples"] == 3
    assert block["pieces"] == 15
    data = store.load_split(tmp_path, "train", labels=True)
    assert data["samples"] == 3
    assert data["pieces"] == 15
    assert data["observations"].shape == (3, *C.OBSERVATION_SHAPE)
    assert list(np.asarray(data["piece_offset"])) == [0, 5, 10, 15]
    assert len(data["game_ids"]) == 3
    assert len(data["public_state_identities"]) == 3


def test_csr_offsets_slice_each_sample_exactly(tmp_path):
    _tiny_split(tmp_path, games=2, pieces=4)
    data = store.load_split(tmp_path, "train", labels=True)
    offsets = np.asarray(data["piece_offset"])
    for row in range(data["samples"]):
        low, high = int(offsets[row]), int(offsets[row + 1])
        assert high - low == 4
        assert list(np.asarray(data["true_rank"])[low:high]) == [0, 1, 2, 3]


def test_the_privileged_directory_carries_its_warning(tmp_path):
    _tiny_split(tmp_path)
    note = (tmp_path / "train" / C.PRIVILEGED_DIRECTORY / "README.txt").read_text()
    assert "SUPERVISED" in note
    assert "never enter a model-input path" in note


def test_the_corpus_digest_ignores_wall_clock(tmp_path):
    block = _tiny_split(tmp_path)
    block["file_digests"] = store.split_digest(tmp_path, "train")
    manifest = {
        "corpus_version": C.CORPUS_VERSION,
        "corpus_format_version": store.CORPUS_FORMAT_VERSION,
        "run_version": K.CORPUS_RUN_VERSION,
        "identity_version": "phase15_identity_v1",
        "seeds": {"a": 1},
        "splits": {"train": block},
    }
    first = store.corpus_digest(manifest)
    manifest["generation_seconds"] = {"train": 12.5}
    assert store.corpus_digest(manifest) == first


def test_unknown_splits_are_refused(tmp_path):
    with pytest.raises(store.Phase15StorageError):
        store.SplitWriter(tmp_path, "nope")
    with pytest.raises(store.Phase15StorageError):
        store.split_root(tmp_path, "nope")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_disjointness_passes_on_genuinely_separate_splits(tmp_path):
    _tiny_split(tmp_path, "train")
    _tiny_split(tmp_path, "development")
    splits = {
        name: store.load_split(tmp_path, name, labels=True)
        for name in ("train", "development")
    }
    # The split lives inside the game id and inside every derived identity,
    # so two splits are disjoint by construction. The check confirms it.
    report = V.disjointness_report(splits)
    assert report["disjoint"] is True
    assert report["pairs"]["train|development"]["shared_game_ids"] == 0


def test_disjointness_detects_a_shared_public_state_identity(tmp_path):
    _tiny_split(tmp_path, "train")
    _tiny_split(tmp_path, "development")
    splits = {
        name: dict(store.load_split(tmp_path, name, labels=True))
        for name in ("train", "development")
    }
    leaked = splits["train"]["public_state_identities"][0]
    splits["development"]["public_state_identities"] = [leaked] + list(
        splits["development"]["public_state_identities"][1:]
    )
    with pytest.raises(V.Phase15VerificationError):
        V.disjointness_report(splits)


def test_a_repeated_identity_inside_one_split_is_counted_not_refused(tmp_path):
    # Section 7 asks for the *splits* to be disjoint. A repeat inside a split
    # is a duplicated opening position — two games whose observer drew the
    # same base setup — which is a fact worth reporting, not a defect.
    _tiny_split(tmp_path, "train", games=3)
    splits = {"train": dict(store.load_split(tmp_path, "train", labels=True))}
    identities = list(splits["train"]["public_state_identities"])
    identities[1] = identities[0]
    splits["train"]["public_state_identities"] = identities
    report = V.disjointness_report(splits)
    assert report["disjoint"] is True
    block = report["within_split"]["train"]
    assert block["repeated_public_state_identities"] == 1
    assert block["repeated_fraction"] == pytest.approx(1 / 3)


def test_a_repeated_game_id_is_still_refused(tmp_path):
    _tiny_split(tmp_path, "train", games=3)
    splits = {"train": dict(store.load_split(tmp_path, "train", labels=True))}
    identifiers = list(splits["train"]["game_ids"])
    identifiers[1] = identifiers[0]
    splits["train"]["game_ids"] = identifiers
    with pytest.raises(V.Phase15VerificationError):
        V.disjointness_report(splits)


def test_label_verification_rejects_an_inadmissible_rank(tmp_path):
    _tiny_split(tmp_path, pieces=2)
    data = dict(store.load_split(tmp_path, "train", labels=True))
    mask = np.array(data["legal_rank_mask"], dtype=bool)
    mask[0, int(np.asarray(data["true_rank"])[0])] = False
    data["legal_rank_mask"] = mask
    with pytest.raises(V.Phase15VerificationError):
        V.label_report(data)


def test_label_verification_rejects_a_moved_flag(tmp_path):
    _tiny_split(tmp_path, pieces=12)
    data = dict(store.load_split(tmp_path, "train", labels=True))
    moved = np.array(data["piece_moved"], dtype=bool)
    ranks = np.asarray(data["true_rank"])
    moved[int(np.argmax(ranks >= 10))] = True
    data["piece_moved"] = moved
    with pytest.raises(V.Phase15VerificationError):
        V.label_report(data)


def test_mixture_report_counts_positions_not_games(tmp_path):
    _tiny_split(tmp_path, games=4, pieces=2)
    data = store.load_split(tmp_path, "train", labels=True)
    report = V.mixture_report(data)
    assert report["positions"] == 4
    assert report["games"] == 4
    assert report["observer_model"]["p18"]["positions"] == 4
    assert report["observer_color"]["red"]["fraction"] == 1.0
