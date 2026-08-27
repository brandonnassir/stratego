"""Phase 15 Agent 1 section 4: the orientation gate.

The Phase 11B defect — `Phase11BSetupSources.draw` returning canonical
own-orientation tuples that the old glue handed straight to `create_game`
for Blue — is the reason this phase exists. These tests prove the rule the
gate enforces, prove the gate catches the exact old mistake, and prove that
no Phase 15 module imports the glue that made it.
"""

from __future__ import annotations

import pytest

from stratego.belief.phase15 import orientation as O
from stratego.engine.constants import (
    BLUE,
    BOARD_COLUMNS,
    FLAG,
    PIECE_COUNTS,
    RED,
    SETUP_SQUARES,
)
from stratego.setups.identity import (
    CANONICAL_CELLS,
    canonical_rank_file,
    orient_setup,
)
from stratego.setups.sampler import load_library_index, sample_setup


@pytest.fixture(scope="module")
def library():
    return load_library_index()


@pytest.fixture(scope="module")
def canonical_pair(library):
    return (
        tuple(sample_setup("train", 11, index=library).canonical),
        tuple(sample_setup("train", 12, index=library).canonical),
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_red_engine_row_equals_canonical_rank():
    for rank in range(4):
        assert O.expected_engine_row(rank, RED) == rank


def test_blue_engine_row_is_nine_minus_canonical_rank():
    for rank in range(4):
        assert O.expected_engine_row(rank, BLUE) == 9 - rank


def test_expected_row_covers_exactly_each_players_setup_rows():
    red = {O.expected_engine_row(rank, RED) for rank in range(4)}
    blue = {O.expected_engine_row(rank, BLUE) for rank in range(4)}
    assert red == {0, 1, 2, 3}
    assert blue == {6, 7, 8, 9}
    assert red == {square // BOARD_COLUMNS for square in SETUP_SQUARES[RED]}
    assert blue == {square // BOARD_COLUMNS for square in SETUP_SQUARES[BLUE]}


def test_expected_row_refuses_an_out_of_range_rank():
    with pytest.raises(O.Phase15OrientationError):
        O.expected_engine_row(4, RED)


# ---------------------------------------------------------------------------
# The assertion on the production path
# ---------------------------------------------------------------------------


def test_oriented_for_agrees_with_the_accepted_helper(canonical_pair):
    red, blue = canonical_pair
    assert O.oriented_for(red, RED) == orient_setup(red, RED)
    assert O.oriented_for(blue, BLUE) == orient_setup(blue, BLUE)


def test_red_orientation_is_the_identity(canonical_pair):
    red, _blue = canonical_pair
    assert O.oriented_for(red, RED) == red


def test_blue_orientation_reverses_the_rank_rows(canonical_pair):
    _red, blue = canonical_pair
    oriented = O.oriented_for(blue, BLUE)
    for index in range(CANONICAL_CELLS):
        rank, file = canonical_rank_file(index)
        assert oriented[(3 - rank) * BOARD_COLUMNS + file] == blue[index]


def test_the_assertion_rejects_a_raw_blue_canonical_tuple(canonical_pair):
    _red, blue = canonical_pair
    assume_not_palindromic = orient_setup(blue, BLUE) != blue
    assert assume_not_palindromic, "this fixture cannot demonstrate the defect"
    with pytest.raises(O.Phase15OrientationError):
        O.assert_engine_orientation(blue, blue, BLUE)


def test_the_assertion_rejects_a_swapped_pair(canonical_pair):
    red, blue = canonical_pair
    with pytest.raises(O.Phase15OrientationError):
        O.assert_engine_orientation(red, O.oriented_for(blue, BLUE), BLUE)


def test_the_assertion_rejects_a_wrong_length_setup(canonical_pair):
    red, _blue = canonical_pair
    with pytest.raises(O.Phase15OrientationError):
        O.assert_engine_orientation(red[:-1], red, RED)


# ---------------------------------------------------------------------------
# The bulk checks
# ---------------------------------------------------------------------------


def test_check_board_reports_flag_inventory_and_pairing(canonical_pair):
    red, blue = canonical_pair
    report = O.check_board(red, blue)
    assert report["paired_mirror"] is True
    assert report["red"]["inventory_exact"] is True
    assert report["blue"]["inventory_exact"] is True
    assert report["red"]["flag"]["row"] in {0, 1, 2, 3}
    assert report["blue"]["flag"]["row"] in {6, 7, 8, 9}


def test_flag_rows_mirror_for_the_same_canonical_setup(canonical_pair):
    red, _blue = canonical_pair
    as_red = O.oriented_for(red, RED)
    as_blue = O.oriented_for(red, BLUE)
    red_row = SETUP_SQUARES[RED][as_red.index(FLAG)] // BOARD_COLUMNS
    blue_row = SETUP_SQUARES[BLUE][as_blue.index(FLAG)] // BOARD_COLUMNS
    assert red_row + blue_row == 9


def test_every_oriented_army_holds_the_official_inventory(library):
    for ordinal in range(24):
        canonical = tuple(sample_setup("train", 900 + ordinal, index=library).canonical)
        for player in (RED, BLUE):
            engine = O.oriented_for(canonical, player)
            counts: dict[int, int] = {}
            for piece_type in engine:
                counts[piece_type] = counts.get(piece_type, 0) + 1
            assert counts == PIECE_COUNTS


def test_negative_canary_detects_the_phase11b_mistake():
    canary = O.negative_canary()
    assert canary["detected"] is True
    assert canary["canary"] == "blue_canonical_passed_directly"


def test_the_gate_passes_and_quantifies_the_old_defect():
    gate = O.orientation_gate(boards=64)
    assert gate["passed"] is True
    assert gate["armies_checked"] == 128
    assert gate["inventory_exact"] is True
    assert gate["paired_orientation"] is True
    assert gate["negative_canary"]["detected"] is True
    # The counterfactual is the point: under the old glue a Blue army whose
    # flag sits on its own back rank shows a front-row flag, which is what
    # Phase 12 saw on 47 of 64 boards.
    counterfactual = gate["defect_counterfactual"]
    assert counterfactual["rate"] > 0.5
    assert gate["front_row_flag_rate"] < 0.1
    assert counterfactual["rate"] > gate["front_row_flag_rate"] * 5


# ---------------------------------------------------------------------------
# The glue that must never come back
# ---------------------------------------------------------------------------


def test_no_phase15_module_imports_the_mis_oriented_corpus_glue():
    from pathlib import Path

    package = Path(O.__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            if "phase11b.corpus" in stripped or "phase11b import corpus" in stripped:
                offenders.append(f"{path.name}: {stripped}")
            if "Phase11BSetupSources" in stripped or "corpus_plans" in stripped:
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == []
