"""Phase 17 Agent 3 sections 3 and 7: structural refusals.

Section 3 forbids any setup library, authored template, Phase 10 selector or
library fallback in training, and section 7 requires that no library or repair
fallback be *reachable*. Reachability is a property of the import graph, so it
is asserted over the source rather than exercised through a code path -- a
fallback that is never taken is still reachable, and a later edit that adds
one must fail here.
"""

import ast
from pathlib import Path

import pytest

SETUP_MODULES = (
    "setup_contract.py",
    "setup_model.py",
    "setup_sampling.py",
    "setup_episode.py",
    "setup_learning.py",
    "setup_metrics.py",
)

#: Every accepted source of pre-made setups. None may be reachable from the
#: Phase 17 setup half's training path.
FORBIDDEN_SETUP_SOURCES = (
    "stratego.setups.library",
    "stratego.setups.sampler",
    "stratego.setups.perturbation",
    "stratego.evaluation.setup_bank",
    "stratego.evaluation.phase10_banks",
    "stratego.training.phase10_selector",
    "stratego.training.phase16.setups",
    "stratego.belief.phase11b.corpus",
    "stratego.belief.phase15.setups",
)

#: Search must not be reachable from collection or training at all.
FORBIDDEN_SEARCH = ("search", "mcts", "phase12")


@pytest.fixture(scope="module")
def package_root() -> Path:
    return Path(__file__).resolve().parents[3] / "stratego" / "training" / "phase17"


def _imported_modules(path: Path) -> "set[str]":
    tree = ast.parse(path.read_text())
    found: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Resolve the relative form so `...setups.library` is comparable.
            prefix = "stratego.training.phase17"
            if node.level:
                parts = prefix.split(".")
                prefix = ".".join(parts[: len(parts) - (node.level - 1)] or ["stratego"])
            module = f"{prefix}.{node.module}" if node.module else prefix
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def test_no_setup_library_is_reachable_from_the_setup_half(package_root):
    for name in SETUP_MODULES:
        imported = _imported_modules(package_root / name)
        for forbidden in FORBIDDEN_SETUP_SOURCES:
            offending = [module for module in imported if module.startswith(forbidden)]
            assert not offending, f"{name} reaches a setup library: {offending}"


def test_no_search_is_reachable_from_the_setup_half(package_root):
    for name in SETUP_MODULES:
        source = (package_root / name).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.unparse(node).lower()
                for forbidden in FORBIDDEN_SEARCH:
                    assert forbidden not in text, f"{name} imports {forbidden}: {text}"


def test_no_move_policy_or_opponent_enters_the_setup_library(package_root):
    """The gate's random-move fixture is a fixture and must stay outside."""
    for name in SETUP_MODULES:
        imported = _imported_modules(package_root / name)
        assert not [m for m in imported if "random_play" in m], name
        assert not [m for m in imported if "phase9_trainer" in m], name


def test_the_orientation_helper_is_imported_never_re_derived(package_root):
    source = (package_root / "setup_sampling.py").read_text()
    assert "from ...belief.phase15.orientation import assert_engine_orientation" in source
    assert "from ...setups.identity import" in source
    # The rule itself must not be restated as a local expression.
    assert "9 - rank" not in source
    assert "9 - canonical_rank" not in source


def test_the_accepted_objective_is_not_edited_or_wrapped(package_root):
    """Agent 1: Phase 17's per-row-maskable loss is a NEW phase17 function."""
    for name in SETUP_MODULES:
        imported = _imported_modules(package_root / name)
        assert not [m for m in imported if "phase9_loss" in m], name


def test_only_the_accepted_digest_function_is_used(package_root):
    """Two functions in this repository are named `state_dict_digest`."""
    source = (package_root / "setup_learning.py").read_text()
    assert "from ..phase9_behavior import state_dict_digest" in source
    assert "def state_dict_digest" not in source


def test_the_setup_half_writes_into_no_earlier_phase_namespace(package_root):
    for name in SETUP_MODULES:
        source = (package_root / name).read_text()
        for phase in ("checkpoints/phase9", "checkpoints/phase14", "reports/phase16"):
            assert phase not in source, f"{name} references {phase}"
