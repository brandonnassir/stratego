"""Shared Phase 8 training fixtures.

The warm-start example/dataset tests all need a small committed corpus with
every supervision-weight class represented. Generating it once per session
keeps the suite's cost at a handful of played games.
"""

from __future__ import annotations

import pytest

from stratego.training import synthetic_corpus as sc
from stratego.training.warmstart_seed import synthetic_game_id

#: The mini corpus: one game per interesting weight pairing, across all three
#: splits, kept short by choosing cheap teachers where the pairing allows it.
WARMSTART_MINI_GAME_IDS = (
    synthetic_game_id("train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 0),
    synthetic_game_id("train", "tactical_rule_based@1.0.0", "basic_heuristic@1.0.0", 0),
    synthetic_game_id("train", "random_legal@1.0.0", "random_legal@1.0.0", 0),
    synthetic_game_id("validation", "basic_heuristic@1.0.0", "stress_chaos@1.0.0", 0),
    synthetic_game_id("validation", "stress_draw_seeker@1.0.0", "strategic_rule_based@1.1.0", 1),
    synthetic_game_id("test", "random_legal@1.0.0", "stress_scout_rush@1.0.0", 0),
)


@pytest.fixture(scope="session")
def warmstart_mini_corpus(tmp_path_factory):
    """`(root, game_ids)` of a committed six-game corpus, generated once."""
    root = tmp_path_factory.mktemp("warmstart_mini_corpus")
    sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=WARMSTART_MINI_GAME_IDS
    )
    return root, WARMSTART_MINI_GAME_IDS
