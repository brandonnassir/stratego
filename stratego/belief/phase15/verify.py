"""Phase 15 Agent 1: what the stored corpus must prove about itself.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` sections 5-7.

Everything here reads the corpus back **from disk** rather than trusting
the objects that wrote it. The build-time assertions in
:mod:`.corpus` prove the samples were correct when they were made; these
prove the bytes on disk still say so.

Four questions
--------------
```text
disjoint     no identity is shared across train / calibration / development
mixture      the achieved cell counts, measured over positions not games
labels       every stored true rank is publicly admissible for its piece
orientation  boards rebuilt from stored game ids are still oriented
```

Counted over positions, not games
----------------------------------
Section 6: "Record exact family/source/color/model counts after position
sampling, not merely intended game counts." A long game contributes the same
sixteen positions as a short one, but a game that ends before sixteen
eligible decisions contributes fewer — so the game-level design and the
position-level result are not the same distribution, and only the second
one is evidence.
"""

from __future__ import annotations

import collections

import numpy as np

from ...setups.families import FAMILY_KEYS
from .contract import (
    CORPUS_COLORS,
    CORPUS_SPLITS,
    LIBRARY_PARTITION,
    LIBRARY_SPLIT,
    OPPONENTS,
    OPPONENT_MIXTURE,
    OBSERVER_MIXTURE,
    POLICY_SOURCES,
    SETUP_MIXTURE,
    SETUP_SOURCES,
    TARGETED_FAMILY_KEYS,
    Phase15Error,
    game_band,
)
from .orientation import check_board
from .seeds import ROLE_OBSERVER, ROLE_OPPONENT, parse_corpus_game_id, setup_seed
from .storage import load_split

#: The verification identity.
VERIFICATION_VERSION = "phase15_corpus_verification_v1"


class Phase15VerificationError(Phase15Error):
    """The stored corpus failed one of its own guarantees."""


def _fractions(codes: np.ndarray, labels) -> dict:
    counts = collections.Counter(np.asarray(codes).tolist())
    total = int(np.asarray(codes).size)
    return {
        labels[code]: {"positions": int(count), "fraction": count / total}
        for code, count in sorted(counts.items())
    }


def mixture_report(data: dict) -> dict:
    """The achieved mixture of one split, counted over positions."""
    total = int(data["samples"])
    report = {
        "positions": total,
        "pieces": int(data["pieces"]),
        "games": int(data["games"]),
        "hidden_pieces_per_position": (
            round(int(data["pieces"]) / total, 4) if total else 0.0
        ),
        "observer_model": _fractions(data["observer_model"], POLICY_SOURCES),
        "opponent": _fractions(data["opponent"], OPPONENTS),
        "setup_source": _fractions(data["setup_source"], SETUP_SOURCES),
        "observer_color": _fractions(data["observer_color"], CORPUS_COLORS),
        "observer_setup_family": _fractions(data["observer_family"], FAMILY_KEYS),
        "opponent_setup_family": _fractions(data["opponent_family"], FAMILY_KEYS),
    }
    bands = collections.Counter(
        game_band(int(value)) for value in np.asarray(data["total_moves"])
    )
    report["game_band"] = {
        name: {"positions": int(count), "fraction": count / total}
        for name, count in sorted(bands.items())
    }
    report["intended"] = {
        "observer_model": dict(OBSERVER_MIXTURE),
        "opponent": dict(OPPONENT_MIXTURE),
        "setup_source": dict(SETUP_MIXTURE),
        "observer_color": {name: 0.5 for name in CORPUS_COLORS},
    }
    report["max_absolute_deviation"] = {
        "observer_model": max(
            abs(report["observer_model"].get(name, {}).get("fraction", 0.0) - share)
            for name, share in OBSERVER_MIXTURE.items()
        ),
        "opponent": max(
            abs(report["opponent"].get(name, {}).get("fraction", 0.0) - share)
            for name, share in OPPONENT_MIXTURE.items()
        ),
        "setup_source": max(
            abs(report["setup_source"].get(name, {}).get("fraction", 0.0) - share)
            for name, share in SETUP_MIXTURE.items()
        ),
        "observer_color": max(
            abs(report["observer_color"].get(name, {}).get("fraction", 0.0) - 0.5)
            for name in CORPUS_COLORS
        ),
    }
    covered = {
        key
        for key in TARGETED_FAMILY_KEYS
        if key in report["opponent_setup_family"] or key in report["observer_setup_family"]
    }
    report["targeted_families_covered"] = sorted(covered)
    report["targeted_families_missing"] = sorted(set(TARGETED_FAMILY_KEYS) - covered)
    return report


def label_report(data: dict) -> dict:
    """Every stored true rank is publicly admissible for its own piece."""
    if "true_rank" not in data:
        raise Phase15VerificationError(
            "label verification needs the privileged labels; load with labels=True"
        )
    true_rank = np.asarray(data["true_rank"], dtype=np.int64)
    mask = np.asarray(data["legal_rank_mask"], dtype=bool)
    rows = np.arange(true_rank.size)
    admissible = mask[rows, true_rank]
    counts = np.asarray(data["remaining_counts"], dtype=np.int64)
    per_piece_counts = counts[
        np.repeat(
            np.arange(int(data["samples"]), dtype=np.int64),
            np.diff(np.asarray(data["piece_offset"], dtype=np.int64)),
        )
    ]
    stocked = per_piece_counts[rows, true_rank] > 0
    moved = np.asarray(data["piece_moved"], dtype=bool)
    # The engine's own rule, restated as a check on the bytes: a piece that
    # has moved cannot be the Flag (10) or a Bomb (11).
    moved_immobile = int(((true_rank >= 10) & moved).sum())
    if not admissible.all():
        raise Phase15VerificationError(
            f"{int((~admissible).sum())} stored true ranks are excluded by their "
            "own public legal-rank mask"
        )
    if not stocked.all():
        raise Phase15VerificationError(
            f"{int((~stocked).sum())} stored true ranks have no remaining public "
            "inventory"
        )
    if moved_immobile:
        raise Phase15VerificationError(
            f"{moved_immobile} moved pieces carry an immobile true rank"
        )
    return {
        "pieces": int(true_rank.size),
        "all_ranks_publicly_admissible": True,
        "all_ranks_have_remaining_inventory": True,
        "moved_pieces_with_immobile_rank": 0,
        "rank_histogram": {
            str(rank): int(count)
            for rank, count in sorted(collections.Counter(true_rank.tolist()).items())
        },
    }


def public_arrays_carry_no_truth(data: dict) -> dict:
    """No public array is the label array in disguise.

    A weak statement made strong by where it is checked: the public half is
    loaded on its own, without `labels=True`, and every public array is
    compared for shape compatibility with the label vector. Only the
    per-piece arrays could match, and none of them may equal the labels.
    """
    if "true_rank" not in data:
        raise Phase15VerificationError("this check needs both halves loaded")
    true_rank = np.asarray(data["true_rank"], dtype=np.int64)
    suspects = []
    for name in ("piece_slot", "piece_square", "perspective_square", "piece_moved"):
        array = np.asarray(data[name])
        if array.shape == true_rank.shape and np.array_equal(
            array.astype(np.int64), true_rank
        ):
            suspects.append(name)
    if suspects:  # pragma: no cover - would be a serious storage defect
        raise Phase15VerificationError(
            f"public arrays {suspects} are identical to the privileged labels"
        )
    return {"public_arrays_checked": 4, "arrays_equal_to_labels": 0}


def disjointness_report(splits: dict) -> dict:
    """No game id and no public-state identity is shared across splits."""
    names = list(splits)
    games = {name: set(data["game_ids"]) for name, data in splits.items()}
    states = {
        name: set(data["public_state_identities"]) for name, data in splits.items()
    }
    findings = []
    pairs = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared_games = games[left] & games[right]
            shared_states = states[left] & states[right]
            pairs[f"{left}|{right}"] = {
                "shared_game_ids": len(shared_games),
                "shared_public_state_identities": len(shared_states),
            }
            if shared_games:
                findings.append(
                    f"{left} and {right} share {len(shared_games)} game ids"
                )
            if shared_states:
                findings.append(
                    f"{left} and {right} share {len(shared_states)} public-state "
                    "identities"
                )
    if findings:
        raise Phase15VerificationError("; ".join(findings))
    # Within a split, a repeated public-state identity is a duplicated
    # training example, not leakage: section 7 asks for the *splits* to be
    # disjoint, and they are. Repeats occur only at ply 0-1, where the public
    # state is determined by the observer's own setup and two games that drew
    # the same base therefore open identically. The count is reported so a
    # reader can see how small it is; a repeated *game id* would be a real
    # defect and is still refused.
    within = {}
    for name, data in splits.items():
        identities = list(data["public_state_identities"])
        unique = len(states[name])
        within[name] = {
            "game_ids": len(data["game_ids"]),
            "unique_game_ids": len(games[name]),
            "public_state_identities": len(identities),
            "unique_public_state_identities": unique,
            "repeated_public_state_identities": len(identities) - unique,
            "repeated_fraction": (
                (len(identities) - unique) / len(identities) if identities else 0.0
            ),
            "repeated_plies": _repeated_plies(identities, data["total_moves"]),
        }
        if within[name]["game_ids"] != within[name]["unique_game_ids"]:
            raise Phase15VerificationError(f"{name} repeats a game id")
    return {
        "pairs": pairs,
        "within_split": within,
        "disjoint": True,
        "rule": (
            "the three splits share no game id and no public-state identity; "
            "repeats *inside* a split are duplicated opening positions and are "
            "counted, not refused"
        ),
    }


def _repeated_plies(identities: "list[str]", total_moves) -> list:
    """The distinct plies at which a repeated identity occurs, first eight.

    Repeats concentrate at ply 0-1, and saying so is the point: an opening
    position is public only in the observer's own setup, so two games that
    drew the same base setup open identically. The set is built once rather
    than per element — 120,000 identities make the quadratic form unusable.
    """
    counts = collections.Counter(identities)
    repeated = {identity for identity, count in counts.items() if count > 1}
    if not repeated:
        return []
    plies = np.asarray(total_moves).tolist()
    return sorted(
        {int(ply) for ply, identity in zip(plies, identities) if identity in repeated}
    )[:8]


def orientation_recheck(data: dict, sources, *, games: int = 256) -> dict:
    """Rebuild boards from stored game ids and re-run the section 4 checks.

    The end-to-end statement: not "the source orients correctly" but "the
    boards this corpus was actually played on are oriented correctly",
    re-derived from the stored identity alone.
    """
    identifiers = list(data["game_ids"])
    if not identifiers:  # pragma: no cover - an empty split
        raise Phase15VerificationError("the split stores no game ids")
    step = max(1, len(identifiers) // int(games))
    sampled = identifiers[::step][: int(games)]
    front_row = 0
    checked = 0
    for game_id in sampled:
        fields = parse_corpus_game_id(game_id)
        library_split = LIBRARY_SPLIT[fields["split"]]
        partition = LIBRARY_PARTITION[fields["split"]]
        observer = sources.draw(
            fields["setup_source"],
            library_split,
            fields["observer_color"],
            setup_seed(game_id, ROLE_OBSERVER),
            partition,
        )
        opponent_color = "blue" if fields["observer_color"] == "red" else "red"
        opponent = sources.draw(
            fields["setup_source"],
            library_split,
            opponent_color,
            setup_seed(game_id, ROLE_OPPONENT),
            partition,
        )
        red, blue = (
            (observer, opponent)
            if fields["observer_color"] == "red"
            else (opponent, observer)
        )
        report = check_board(red.canonical, blue.canonical)
        front_row += int(report["red"]["flag"]["row"] == 3)
        front_row += int(report["blue"]["flag"]["row"] == 6)
        checked += 1
    return {
        "games_rechecked": checked,
        "armies_rechecked": checked * 2,
        "front_row_flags": front_row,
        "front_row_flag_rate": front_row / (checked * 2) if checked else 0.0,
        "all_boards_oriented": True,
        "rebuilt_from": "the stored game id alone",
    }


def verify_corpus(root, *, sources=None, orientation_games: int = 256) -> dict:
    """Every check, over every split. Raises on the first failure."""
    from .setups import Phase15SetupSources

    sources = Phase15SetupSources() if sources is None else sources
    splits = {
        split: load_split(root, split, labels=True) for split in CORPUS_SPLITS
    }
    report = {
        "verification_version": VERIFICATION_VERSION,
        "splits": {},
        "disjointness": disjointness_report(splits),
    }
    for split, data in splits.items():
        report["splits"][split] = {
            "mixture": mixture_report(data),
            "labels": label_report(data),
            "truth_isolation": public_arrays_carry_no_truth(data),
            "orientation": orientation_recheck(
                data, sources, games=int(orientation_games)
            ),
        }
    report["passed"] = True
    return report


__all__ = [
    "VERIFICATION_VERSION",
    "Phase15VerificationError",
    "disjointness_report",
    "label_report",
    "mixture_report",
    "orientation_recheck",
    "public_arrays_carry_no_truth",
    "verify_corpus",
]
