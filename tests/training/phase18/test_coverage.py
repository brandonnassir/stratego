"""The S01-S30 coverage table: every cited symbol resolves, every cited test
exists, and a row is complete only from recorded test outcomes."""

import json

from stratego.training.phase18.coverage import COVERAGE, G2_ROWS, OUT_OF_SCOPE, attach_test_outcomes, verify_coverage


def test_every_g2_row_resolves_its_symbols_and_tests(repository_root):
    report = verify_coverage(repository_root)
    assert report["problems"] == []
    assert report["verified"]
    assert set(report["rows"]) == set(G2_ROWS) == set(COVERAGE)
    for row in report["rows"].values():
        assert row["implementation"] and row["tests"]


def test_rows_s31_to_s35_are_listed_as_out_of_scope_and_never_complete(repository_root):
    report = verify_coverage(repository_root)
    assert set(report["out_of_scope"]) == {"S31", "S32", "S33", "S34", "S35"} == set(OUT_OF_SCOPE)
    assert all(entry["complete"] is False for entry in report["out_of_scope"].values())


def test_a_row_is_complete_only_when_every_cited_test_passed(repository_root):
    report = verify_coverage(repository_root)
    every_test = {test for row in report["rows"].values() for test in row["tests"]}
    passing = {test: "passed" for test in every_test}
    complete = attach_test_outcomes(json.loads(json.dumps(report)), passing)
    assert complete["all_g2_rows_complete"] and complete["rows_complete"] == 30
    one_failed = dict(passing)
    victim = next(iter(report["rows"]["S13"]["tests"]))
    one_failed[victim] = "failed"
    partial = attach_test_outcomes(json.loads(json.dumps(report)), one_failed)
    assert not partial["rows"]["S13"]["complete"] and not partial["all_g2_rows_complete"]
    missing = attach_test_outcomes(json.loads(json.dumps(report)), {})
    assert missing["rows_complete"] == 0, "no recorded run, no complete row"


def test_the_method_map_statuses_are_carried_not_restated(repository_root):
    method_map = json.loads((repository_root / "reports/phase18/ataraxos_setup_method_map_v2.json").read_text())
    statuses = {row["id"]: row["status"] for row in method_map["rows"]}
    report = verify_coverage(repository_root)
    for row_id, row in report["rows"].items():
        assert row["map_status"] == statuses[row_id]
    assert statuses["S13"] == "corrected" and statuses["S04"] == "corrected"
