"""Raw-result files, replay sidecars, and Markdown tables.

Specification source: Phase 4 Agent 3 instructions ("Raw match result",
"Statistics", "League rating").

Three artefact kinds, deliberately separated:

- **`agent_0X_*.csv`** -- one row per match, no action histories. This is the
  table a human or a spreadsheet reads, and the one Agent 4's calibration
  inherits.
- **`agent_0X_*.jsonl`** -- one engine :class:`ReplayRecord` per line, keyed by
  `match_id`. Written only when asked. The row's `replay_digest` is what proves
  two runs agree; the sidecar is what lets someone *look at* a disagreement.
- **Markdown tables** -- rendered from the statistics summaries for the shared
  Phase 4 report.

Why the histories live in a sidecar
----------------------------------
A match is already fully reproducible from `match_id` plus the bank version:
setups come from the bank, both policy seeds derive from `match_id`, and each
decision seed derives from the policy seed and the ply. The action history is a
*verification* artefact, not a reproduction requirement. Agent 4's league will
run tens of thousands of games at roughly 300 plies each, so embedding every
history inline would add millions of integers to a file whose main job is to be
read. Keeping the digest inline and the histories beside it preserves detection
of any divergence at a fixed 64 bytes per row.
"""

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..engine.replay import ReplayRecord
from .match_runner import MatchResult, MatchRunnerError, RunSummary
from .statistics import LeagueRatings

REPORTING_VERSION = "evaluation_reporting_v1"

#: CSV column order. Fixed rather than derived from a dict so a later field
#: addition appends instead of silently reordering an existing file.
CSV_COLUMNS: tuple[str, ...] = (
    "match_id",
    "paired_unit_id",
    "candidate_policy_id",
    "candidate_policy_version",
    "opponent_policy_id",
    "opponent_policy_version",
    "candidate_color",
    "candidate_color_name",
    "setup_pair_id",
    "replicate",
    "root_seed",
    "candidate_seed",
    "opponent_seed",
    "winner",
    "winner_name",
    "draw",
    "candidate_result",
    "candidate_score",
    "terminal_reason",
    "plies",
    "decisions",
    "replay_digest",
    "replay_reference",
    "wall_clock_seconds",
    "policy_error_category",
    "policy_error_role",
    "policy_error_policy",
    "policy_error_ply",
    "policy_error",
    "suite_version",
    "pairing_mode",
    "setup_bank_version",
    "rules",
    "first_player",
    "red_setup",
    "blue_setup",
    "schema_version",
    "runner_version",
)


def _ensure_parent(path: "str | Path") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Raw results
# ---------------------------------------------------------------------------


def result_row(result: MatchResult) -> dict:
    """One match as a flat, CSV-safe mapping. Action histories are excluded."""
    payload = result.to_dict()
    payload.pop("action_history", None)
    payload.pop("rules_payload", None)
    return {column: payload.get(column) for column in CSV_COLUMNS}


def write_results_csv(path: "str | Path", results: "Iterable[MatchResult]") -> Path:
    """Write the raw match table, sorted by `match_id` for a stable diff."""
    rows = sorted(results, key=lambda row: row.match_id)
    target = _ensure_parent(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(result_row(row))
    return target


def write_results_json(
    path: "str | Path", results: "Iterable[MatchResult]", *, include_actions: bool = False
) -> Path:
    """Write raw results as JSON, optionally with action histories inline."""
    rows = sorted(results, key=lambda row: row.match_id)
    payload = []
    for row in rows:
        entry = row.to_dict()
        if not include_actions:
            entry["action_history"] = None
        payload.append(entry)
    target = _ensure_parent(path)
    target.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return target


def write_json(path: "str | Path", payload: Mapping[str, Any], *, indent: int = 2) -> Path:
    """Write a machine-readable summary. Indented, because a human reads it too."""
    target = _ensure_parent(path)
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=indent, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return target


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


# ---------------------------------------------------------------------------
# Replay sidecar
# ---------------------------------------------------------------------------


def write_replays_jsonl(path: "str | Path", results: "Iterable[MatchResult]") -> tuple[Path, int]:
    """Write one replay record per line, keyed by `match_id`.

    Requires an inline action history on every scored row: a digest-only row has
    nothing to write, and silently skipping it would produce a sidecar that looks
    complete. Errored (quarantined) matches are unfinished and are skipped with no
    complaint, since there is no completed game to record.
    """
    rows = sorted(results, key=lambda row: row.match_id)
    target = _ensure_parent(path)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.errored:
                continue
            if row.action_history is None:
                raise MatchRunnerError(
                    f"match {row.match_id} has no inline action history, so no replay can "
                    "be written; run with record_actions=True"
                )
            line = {
                "match_id": row.match_id,
                "paired_unit_id": row.paired_unit_id,
                "replay_digest": row.replay_digest,
                "replay": json.loads(row.replay_record().to_json()),
            }
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
            written += 1
    return target, written


def read_replays_jsonl(path: "str | Path") -> dict[str, ReplayRecord]:
    """Load a replay sidecar into `match_id -> ReplayRecord`."""
    records: dict[str, ReplayRecord] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            entry = json.loads(text)
            match_id = entry["match_id"]
            if match_id in records:
                raise MatchRunnerError(
                    f"{path}: match {match_id} appears twice (line {number})"
                )
            records[match_id] = ReplayRecord(**entry["replay"])
    return records


def attach_replay_reference(
    results: "Iterable[MatchResult]", reference: str, *, drop_actions: bool = True
) -> tuple[MatchResult, ...]:
    """Point every row at the sidecar that holds its replay.

    With `drop_actions` the inline histories are removed at the same time, which
    is the intended pairing: the history lives in exactly one place, and the row
    says where.
    """
    updated = []
    for row in sorted(results, key=lambda item: item.match_id):
        row = row.with_replay_reference(reference)
        updated.append(row.without_action_history() if drop_actions else row)
    return tuple(updated)


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------


def _table(headers: "Sequence[str]", rows: "Sequence[Sequence[Any]]") -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _interval(summary: Mapping[str, Any]) -> str:
    interval = summary["confidence_interval"]
    return f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"


def render_matchup_table(summaries: "Mapping[str, Mapping[str, Any]]") -> str:
    """Effective win rate with its paired interval, one row per matchup."""
    rows = []
    for matchup in sorted(summaries):
        summary = summaries[matchup]
        rows.append(
            [
                summary["candidate"],
                summary["opponent"],
                summary["paired_units"],
                summary["games"],
                f"{summary['effective_win_rate']:.3f}",
                _interval(summary),
                f"{summary['wins']}/{summary['draws']}/{summary['losses']}",
                "yes" if summary["separated_from_even"] else "no",
            ]
        )
    return _table(
        ("Candidate", "Opponent", "Units", "Games", "EWR", "95% CI", "W/D/L", "Separated"),
        rows,
    )


def render_color_table(summaries: "Mapping[str, Mapping[str, Any]]") -> str:
    """Candidate performance split by colour, one row per matchup."""
    rows = []
    for matchup in sorted(summaries):
        summary = summaries[matchup]
        colors = summary["color_split"]
        difference = colors.get("difference_red_minus_blue")
        rows.append(
            [
                summary["candidate"],
                summary["opponent"],
                _rate(colors["red"]),
                _rate(colors["blue"]),
                "" if difference is None else f"{difference:+.3f}",
            ]
        )
    return _table(
        ("Candidate", "Opponent", "EWR as red", "EWR as blue", "Red - blue"), rows
    )


def _rate(entry: Mapping[str, Any]) -> str:
    rate = entry.get("effective_win_rate")
    if rate is None:
        return "n/a"
    return f"{rate:.3f} ({entry['wins']}/{entry['draws']}/{entry['losses']})"


def render_terminal_table(frequencies: Mapping[str, Any]) -> str:
    """How games ended, most frequent first."""
    counts = frequencies["counts"]
    shares = frequencies["shares"]
    rows = [[reason, count, f"{shares[reason]:.3f}"] for reason, count in counts.items()]
    return _table(("Terminal reason", "Games", "Share"), rows)


def render_ply_table(summaries: "Mapping[str, Mapping[str, Any]]") -> str:
    """Game length by matchup."""
    rows = []
    for matchup in sorted(summaries):
        summary = summaries[matchup]
        plies = summary["plies"]
        rows.append(
            [
                summary["candidate"],
                summary["opponent"],
                f"{plies['mean']:.0f}" if plies["mean"] is not None else "",
                f"{plies['median']:.0f}" if plies["median"] is not None else "",
                plies["minimum"],
                plies["maximum"],
            ]
        )
    return _table(("Candidate", "Opponent", "Mean plies", "Median", "Min", "Max"), rows)


def render_league_table(ratings: "LeagueRatings | Mapping[str, Any]") -> str:
    """Bradley-Terry ranking on an Elo-like scale, strongest first."""
    payload = ratings.to_dict() if isinstance(ratings, LeagueRatings) else dict(ratings)
    rows = []
    for rank, token in enumerate(payload["ranking"], start=1):
        rows.append(
            [
                rank,
                token,
                f"{payload['ratings'][token]:.1f}",
                f"{payload['strengths'][token]:.4f}",
                payload["games"][token],
            ]
        )
    return _table(("Rank", "Policy", "Rating", "BT strength", "Games"), rows)


def render_worker_table(runs: "Sequence[Mapping[str, Any]]") -> str:
    """The parallel reproducibility sweep: one row per worker count."""
    rows = []
    for entry in runs:
        rows.append(
            [
                entry["worker_count"],
                entry.get("chunk_count", ""),
                entry["matches_run"],
                f"{entry['wall_clock_seconds']:.1f}",
                f"{entry.get('speedup', 1.0):.2f}x" if entry.get("speedup") else "",
                entry["results_digest"][:12],
                entry.get("mismatches", 0),
            ]
        )
    return _table(
        ("Workers", "Chunks", "Matches", "Seconds", "Speedup", "Results digest", "Mismatches"),
        rows,
    )


def render_run_report(summary: Mapping[str, Any], *, title: str = "Evaluation run") -> str:
    """A complete Markdown block for one run's statistics."""
    sections = [f"### {title}", ""]
    sections.append(
        f"{summary['matches']} matches, {summary['paired_units']} paired units, "
        f"{summary['matchups']} matchups, {summary['policy_errors']} policy errors."
    )
    bootstrap = summary["bootstrap"]
    sections.append(
        f"Intervals: {bootstrap['confidence']:.0%} percentile bootstrap over "
        f"{bootstrap['resampling_unit']}s, {bootstrap['resamples']:,} resamples, "
        f"base seed {bootstrap['base_seed']}."
    )
    sections += ["", "#### Effective win rate", "", render_matchup_table(summary["per_matchup"])]
    sections += ["", "#### Colour split", "", render_color_table(summary["per_matchup"])]
    sections += ["", "#### Game length", "", render_ply_table(summary["per_matchup"])]
    sections += ["", "#### Terminal reasons", "", render_terminal_table(summary["terminal_reasons"])]
    if "league" in summary:
        sections += ["", "#### League rating (secondary)", "", render_league_table(summary["league"])]
    if summary.get("problems"):
        sections += ["", "#### Problems", ""]
        sections += [f"- {problem}" for problem in summary["problems"]]
    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def run_manifest(
    run: RunSummary,
    *,
    schedule_manifest: "Mapping[str, Any] | None" = None,
    extra: "Mapping[str, Any] | None" = None,
) -> dict:
    """Everything needed to identify a run, without the rows themselves."""
    payload: dict[str, Any] = {
        "reporting_version": REPORTING_VERSION,
        "run": run.summary_dict(),
    }
    if schedule_manifest is not None:
        payload["schedule"] = dict(schedule_manifest)
    if extra:
        payload.update(dict(extra))
    return payload


__all__ = [
    "CSV_COLUMNS",
    "REPORTING_VERSION",
    "attach_replay_reference",
    "read_replays_jsonl",
    "render_color_table",
    "render_league_table",
    "render_matchup_table",
    "render_ply_table",
    "render_run_report",
    "render_terminal_table",
    "render_worker_table",
    "result_row",
    "run_manifest",
    "write_json",
    "write_replays_jsonl",
    "write_results_csv",
    "write_results_json",
]
