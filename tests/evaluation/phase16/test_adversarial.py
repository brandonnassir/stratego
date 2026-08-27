"""The adversarial setup library: authoring, gating, storage, harvest."""

import json

import pytest

from stratego.engine.constants import BOMB, FLAG, PIECE_COUNTS
from stratego.evaluation.phase16.adversarial import (
    append_harvest_setup,
    author_setup,
    build_library_document,
    library_entry,
    load_library,
    save_library,
    setup_properties,
    validate_setup,
)
from stratego.evaluation.phase16.contract import (
    ADVERSARIAL_FAMILIES,
    AUTHORED_FAMILIES,
    FAMILY_OPERATOR_HARVEST,
    Phase16MeasurementError,
    SETUPS_PER_FAMILY,
)
from stratego.setups.identity import canonical_neighbours, canonical_rank_file


class TestAuthoring:
    def test_counts(self, library_document):
        assert library_document["setup_count"] == 96
        for family in AUTHORED_FAMILIES:
            assert library_document["families"][family]["setup_count"] == SETUPS_PER_FAMILY

    def test_operator_harvest_present_but_empty(self, library_document):
        block = library_document["families"][FAMILY_OPERATOR_HARVEST]
        assert block["setup_count"] == 0
        assert block["authored"] is False

    def test_every_family_documented(self, library_document):
        for family in ADVERSARIAL_FAMILIES:
            assert library_document["families"][family]["description"]

    def test_deterministic(self):
        first = author_setup("scout_screen", 4)
        second = author_setup("scout_screen", 4)
        assert first["canonical_setup"] == second["canonical_setup"]

    def test_no_duplicates(self, library_families):
        seen = set()
        for entries in library_families.values():
            for entry in entries:
                key = tuple(entry["canonical_setup"])
                assert key not in seen
                seen.add(key)

    def test_exact_inventory_everywhere(self, library_families):
        for entries in library_families.values():
            for entry in entries:
                counts: dict = {}
                for piece in entry["canonical_setup"]:
                    counts[piece] = counts.get(piece, 0) + 1
                assert counts == PIECE_COUNTS

    def test_every_entry_passes_the_imported_gate(self, library_families):
        for entries in library_families.values():
            for entry in entries:
                report = validate_setup(tuple(entry["canonical_setup"]))
                assert report["paired_mirror"] is True

    def test_unknown_family_refused(self):
        with pytest.raises(Phase16MeasurementError):
            author_setup("operator_harvest", 0)
        with pytest.raises(Phase16MeasurementError):
            author_setup("bombed_corner_flag", SETUPS_PER_FAMILY)


class TestFamilySignatures:
    def test_bombed_corner_flag(self, library_families):
        for entry in library_families["bombed_corner_flag"]:
            facts = entry["properties"]
            assert facts["flag_rank"] == 0
            assert facts["flag_file"] in (0, 9)
            assert facts["bombs_adjacent_to_flag"] == facts["flag_neighbours"]

    def test_bombed_center_flag(self, library_families):
        for entry in library_families["bombed_center_flag"]:
            facts = entry["properties"]
            assert facts["flag_rank"] == 0
            assert 3 <= facts["flag_file"] <= 6
            assert facts["bombs_adjacent_to_flag"] == facts["flag_neighbours"]

    def test_scout_screen(self, library_families):
        for entry in library_families["scout_screen"]:
            assert entry["properties"]["scouts_on_front_rank"] >= 6

    def test_aggressive_marshal(self, library_families):
        for entry in library_families["aggressive_marshal"]:
            assert entry["properties"]["marshal_rank"] == 3

    def test_spy_shadow(self, library_families):
        for entry in library_families["spy_shadow"]:
            facts = entry["properties"]
            assert facts["spy_rank"] >= 2
            assert facts["spy_general_distance"] <= 2

    def test_miner_wall(self, library_families):
        for entry in library_families["miner_wall"]:
            facts = entry["properties"]
            assert facts["miner_ranks"] == [2, 2, 2, 2, 2]
            assert facts["miner_file_spread"] >= 6

    def test_decoy_flag_structure(self, library_families):
        for entry in library_families["decoy_flag_structure"]:
            canonical = entry["canonical_setup"]
            facts = entry["properties"]
            assert facts["bombs_adjacent_to_flag"] == 0
            corner = 0 if facts["flag_file"] == 0 else 9
            decoy_index = 9 - corner  # rank 0, opposite file
            assert canonical[decoy_index] != FLAG
            assert all(
                canonical[cell] == BOMB for cell in canonical_neighbours(decoy_index)
            )

    def test_free_novelty_is_unconventional(self, library_families):
        for entry in library_families["free_novelty"]:
            facts = entry["properties"]
            assert (
                facts["flag_rank"] > 0
                or facts["bombs_on_front_rank"] >= 2
                or (facts["bombs_adjacent_to_flag"] == 0 and facts["marshal_rank"] == 0)
            )


class TestStorage:
    def test_round_trip(self, tmp_path, library_document):
        path = tmp_path / "library.json"
        save_library(library_document, path)
        loaded = load_library(path)
        assert loaded["library_digest"] == library_document["library_digest"]
        assert loaded["authored_digest"] == library_document["authored_digest"]

    def test_tamper_refused(self, tmp_path, library_document):
        path = tmp_path / "library.json"
        save_library(library_document, path)
        tampered = json.loads(path.read_text())
        cells = tampered["families"]["scout_screen"]["setups"][0]["canonical_setup"]
        cells[cells.index(FLAG)] = BOMB  # guaranteed content change
        path.write_text(json.dumps(tampered))
        with pytest.raises(Phase16MeasurementError):
            load_library(path)

    def test_library_entry_lookup(self, library_document):
        entry = library_entry(library_document, "miner_wall", 3)
        assert entry["family"] == "miner_wall"
        with pytest.raises(Phase16MeasurementError):
            library_entry(library_document, "miner_wall", 99)


class TestHarvest:
    def test_append_dedup_and_digests(self, library_document):
        document = json.loads(json.dumps(library_document))  # deep copy
        donor = tuple(
            library_entry(document, "bombed_corner_flag", 0)["canonical_setup"]
        )
        before_authored = document["authored_digest"]
        before_library = document["library_digest"]
        entry = append_harvest_setup(
            document, donor, provenance={"note": "test"}, captured_utc="t"
        )
        assert entry is not None
        assert entry["family"] == FAMILY_OPERATOR_HARVEST
        assert document["harvest_revision"] == 1
        assert document["setup_count"] == 97
        assert document["authored_digest"] == before_authored
        assert document["library_digest"] != before_library
        # The identical tuple again: deduplicated, nothing changes.
        again = append_harvest_setup(
            document, donor, provenance={"note": "test"}, captured_utc="t"
        )
        assert again is None
        assert document["harvest_revision"] == 1
        assert document["setup_count"] == 97

    def test_invalid_setup_refused(self, library_document):
        document = json.loads(json.dumps(library_document))
        bad = list(library_entry(document, "scout_screen", 0)["canonical_setup"])
        bad[bad.index(FLAG)] = BOMB  # no flag, 7 bombs: breaks the inventory
        with pytest.raises(Exception):
            append_harvest_setup(
                document, tuple(bad), provenance={}, captured_utc="t"
            )


class TestProperties:
    def test_setup_properties_shape(self, library_families):
        entry = library_families["balanced" if False else "scout_screen"][0]
        facts = setup_properties(tuple(entry["canonical_setup"]))
        flag_index = entry["canonical_setup"].index(FLAG)
        assert (facts["flag_rank"], facts["flag_file"]) == canonical_rank_file(flag_index)
