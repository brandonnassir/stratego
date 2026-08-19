"""Phase 10 Agent 5: the bounded validation evaluation.

These tests pin the structural claims Agent 5's selection rests on. They
play no neural game: the expensive evidence lives in the acceptance
artifact, and what belongs here is the part that must not silently drift —
the learned branch's ladder, the frozen-seed wrapper's neutrality, the
case-to-game construction, and the fact that this module cannot reach the
sealed test bank.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.evaluation import phase10_validation as validation
from stratego.evaluation.match_runner import play_match
from stratego.evaluation.phase10_banks import build_phase10_bank
from stratego.evaluation.registry import build_policy, policy_ref
from stratego.setups.identity import content_fingerprint
from stratego.training import phase10_selector as selector_module
from stratego.training.phase10_contract import (
    LEARNED_MIXTURE_WEIGHT,
    MATCHUP_LEARNED_VS_NEUTRAL,
    MATCHUP_TOKENS,
    NEUTRAL_MIXTURE_WEIGHT,
)
from stratego.training.phase10_seed import case_match_seed
from stratego.training.phase10_selector import (
    LearnedSetupSource,
    candidate,
    load_library_index,
    load_scorer,
)


@pytest.fixture(scope="module")
def cases():
    built, _manifest = build_phase10_bank("validation")
    return built


@pytest.fixture(scope="module")
def source():
    return LearnedSetupSource(candidate("P10-D"), load_scorer(), load_library_index())


def _parsed(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


# ---------------------------------------------------------------------------
# The learned branch, as Agent 5 independently verified it
# ---------------------------------------------------------------------------


def test_the_branch_decision_happens_exactly_once_in_the_production_draw():
    """The 0.35/0.65 choice is made once, at the branch, and nowhere else.

    Agent 5's standing obligation before its first strength game. The check
    is on the source rather than on behaviour, because "exactly once" is a
    statement about the code path: a second comparison somewhere else would
    re-apply the neutral weight without changing any single draw's type.
    """
    tree = _parsed(selector_module.LearnedSetupSource.draw)
    named = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert named.count("selector_branch_uniform") == 1
    assert named.count("selector_base_uniform") == 1

    comparisons = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and (
            "NEUTRAL_MIXTURE_WEIGHT" in ast.unparse(node)
            or "LEARNED_MIXTURE_WEIGHT" in ast.unparse(node)
        )
    ]
    assert comparisons == ["branch_uniform < NEUTRAL_MIXTURE_WEIGHT"]

    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in (0.35, 0.65, 0.5775, 0.4225)
    ]
    assert literals == []


def test_the_inverse_cdf_walk_cannot_reach_the_mixed_vector():
    """`base_index_for_uniform` reads the softmax ladder and nothing else."""
    tree = _parsed(selector_module.SelectorDistribution.base_index_for_uniform)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "cumulative_learned" in attributes
    assert "p_mixed" not in attributes
    assert "p_neutral" not in attributes


def test_one_branch_coin_per_draw_and_a_base_uniform_only_on_the_learned_branch(
    source, monkeypatch
):
    """The runtime counterpart of the structural reading."""
    counters = {"branch": 0, "base": 0}
    original_branch = selector_module.selector_branch_uniform
    original_base = selector_module.selector_base_uniform

    def counting_branch(*arguments, **keywords):
        counters["branch"] += 1
        return original_branch(*arguments, **keywords)

    def counting_base(*arguments, **keywords):
        counters["base"] += 1
        return original_base(*arguments, **keywords)

    monkeypatch.setattr(selector_module, "selector_branch_uniform", counting_branch)
    monkeypatch.setattr(selector_module, "selector_base_uniform", counting_base)
    draws = [
        source.draw(
            selector_module.SelectorRequest(
                split="validation", color="blue", selector_seed=seed
            )
        )
        for seed in range(1, 301)
    ]
    learned = sum(1 for draw in draws if draw.branch == "learned")
    assert counters["branch"] == len(draws)
    assert counters["base"] == learned
    assert 0 < learned < len(draws)


def test_the_p_mixed_ladder_reproduces_the_superseded_double_mixing(source):
    """The negative control, as an assertion rather than a diagnostic.

    Walking `cumsum(p_mixed)` after a branch coin that has already applied
    the 0.35 neutral weight realizes `0.5775*neutral + 0.4225*learned`. The
    test states that in closed form and requires the production ladder to
    realize the frozen mixture instead, so a regression cannot pass by
    looking approximately right.
    """
    distribution = source.distribution("blue", "validation")

    production = (
        NEUTRAL_MIXTURE_WEIGHT * distribution.p_neutral
        + LEARNED_MIXTURE_WEIGHT * distribution.p_learned
    )
    assert np.array_equal(production, distribution.p_mixed)

    # Walking cumsum(v) with a uniform realizes v, so the shadow ladder
    # realizes p_mixed inside the learned branch.
    defective = (
        NEUTRAL_MIXTURE_WEIGHT * distribution.p_neutral
        + LEARNED_MIXTURE_WEIGHT * distribution.p_mixed
    )
    expected = (
        NEUTRAL_MIXTURE_WEIGHT + LEARNED_MIXTURE_WEIGHT * NEUTRAL_MIXTURE_WEIGHT
    ) * distribution.p_neutral + (
        LEARNED_MIXTURE_WEIGHT * LEARNED_MIXTURE_WEIGHT
    ) * distribution.p_learned
    assert np.allclose(defective, expected, rtol=0, atol=1e-18)

    families = 16
    per_family = distribution.base_count // families

    def family_view(vector):
        return np.asarray(vector).reshape(families, per_family).sum(axis=1)

    def total_variation(left, right) -> float:
        return 0.5 * float(np.abs(family_view(left) - family_view(right)).sum())

    assert total_variation(production, distribution.p_mixed) == 0.0
    assert total_variation(defective, distribution.p_mixed) > 0.02


# ---------------------------------------------------------------------------
# The frozen-seed opponent wrapper
# ---------------------------------------------------------------------------


def test_the_frozen_seed_wrapper_keeps_the_wrapped_policy_identity():
    """A stored row must name the accepted baseline, not a Phase 10 variant."""
    inner = build_policy("strategic_rule_based")
    wrapped = validation.FrozenSeedPolicy(inner, 12345)
    assert wrapped.ref == inner.ref
    assert wrapped.requirements == inner.requirements
    assert wrapped.frozen_policy_seed == 12345
    assert wrapped.describe()["phase10_frozen_policy_seed"] == 12345


def test_the_frozen_seed_wrapper_is_a_no_op_on_the_runners_own_seed(cases, source):
    """Handed the seed the runner would have used, the wrapper changes nothing.

    This is what licenses the wrapper: it substitutes a seed and does
    nothing else, so a game played through it differs from the accepted
    runner's game only where the frozen Phase 10 seed differs from the
    derived one.
    """
    case = cases[0]
    own = {color: validation.learned_own_side(source, case, color) for color in ("red", "blue")}
    row = validation.game_setups(case, "vs_strategic", own)[0]
    opponent_ref = policy_ref("strategic_rule_based")
    own_ref = policy_ref("basic_heuristic")  # a cheap stand-in for the neural side
    spec = validation.build_spec(
        case,
        row["game_index"],
        "vs_strategic",
        arm="learned",
        candidate_id="P10-D",
        own_ref=own_ref,
        opponent_ref=opponent_ref,
    )
    bank = validation.single_game_bank(spec, row["red_setup"], row["blue_setup"])
    plain = play_match(
        spec,
        bank=bank,
        policies={
            own_ref.token: build_policy("basic_heuristic"),
            opponent_ref.token: build_policy("strategic_rule_based"),
        },
        record_actions=True,
    )
    wrapped = play_match(
        spec,
        bank=bank,
        policies={
            own_ref.token: build_policy("basic_heuristic"),
            opponent_ref.token: validation.FrozenSeedPolicy(
                build_policy("strategic_rule_based"), spec.opponent_seed
            ),
        },
        record_actions=True,
    )
    assert plain.comparable() == wrapped.comparable()


def test_the_opponent_seed_is_the_frozen_case_match_seed(cases, source):
    """Arm- and candidate-independent, exactly as Agent 1 froze it."""
    case = cases[0]
    seeds = set()
    for arm, candidate_id in (("learned", "P10-D"), ("learned", "P10-A"), ("neutral", None)):
        seeds.add(
            validation.FrozenSeedPolicy(
                build_policy("random_legal"), case_match_seed(case.case_id, 0, "vs_random")
            ).frozen_policy_seed
        )
        assert validation.cell_token(arm, candidate_id, "vs_random").endswith("vs_random")
    assert len(seeds) == 1


# ---------------------------------------------------------------------------
# Case -> game construction
# ---------------------------------------------------------------------------


def test_game_identity_is_candidate_specific_but_the_frozen_seed_is_not(cases):
    """Two cells never share a game identity; both descend from one seed."""
    case = cases[3]
    identities = set()
    root_seeds = set()
    for candidate_id in ("P10-A", "P10-D"):
        for matchup in ("vs_strategic", "vs_random"):
            spec = validation.build_spec(
                case,
                0,
                matchup,
                arm="learned",
                candidate_id=candidate_id,
                own_ref=policy_ref("basic_heuristic"),
                opponent_ref=policy_ref("strategic_rule_based"),
            )
            identities.add(spec.match_id)
            root_seeds.add((matchup, spec.root_seed))
    assert len(identities) == 4
    # One frozen seed per matchup, shared by every candidate.
    assert len({seed for _token, seed in root_seeds}) == 2
    for matchup, seed in root_seeds:
        assert seed == case_match_seed(case.case_id, 0, matchup)


def test_the_direct_matchup_seats_two_selectors_and_no_held_out_setup(cases, source):
    """`learned_vs_neutral` has two sides, so the opponent setup has no seat."""
    case = cases[5]
    own = {color: validation.learned_own_side(source, case, color) for color in ("red", "blue")}
    rows = validation.game_setups(case, MATCHUP_LEARNED_VS_NEUTRAL, own)
    assert [row["own_color"] for row in rows] == ["red", "blue"]
    opponent = {
        case.oriented_opponent(RED),
        case.oriented_opponent(BLUE),
    }
    for row in rows:
        assert row["opposing_neutral"] is not None
        assert row["red_setup"] not in opponent or row["blue_setup"] not in opponent
        # game 0: learned is Red; game 1: learned is Blue.
        player = RED if row["own_color"] == "red" else BLUE
        assert (
            row["red_setup"] if player == RED else row["blue_setup"]
        ) == own[row["own_color"]].oriented(player)


def test_every_other_matchup_seats_the_frozen_held_out_opponent_setup(cases, source):
    case = cases[7]
    own = {color: validation.learned_own_side(source, case, color) for color in ("red", "blue")}
    for matchup in MATCHUP_TOKENS:
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            continue
        for row in validation.game_setups(case, matchup, own):
            opposing = BLUE if row["own_color"] == "red" else RED
            seated = row["blue_setup"] if opposing == BLUE else row["red_setup"]
            assert seated == case.oriented_opponent(opposing)


def test_the_neutral_arm_is_the_accepted_sampler_and_matches_the_frozen_case(cases):
    """A moved sampler stops the run rather than shifting every delta."""
    for case in cases[:8]:
        for color in ("red", "blue"):
            draw = validation.neutral_own_side(case, color)
            assert draw.arm == validation.ARM_NEUTRAL
            assert draw.candidate_id is None
            assert draw.final_setup_fingerprint == (
                case.neutral_provenance[color]["final_setup_fingerprint"]
            )
            assert draw.final_setup_fingerprint == content_fingerprint(draw.canonical)


def test_the_learned_arm_uses_the_cases_own_selector_seed(cases, source):
    for case in cases[:8]:
        for color in ("red", "blue"):
            draw = validation.learned_own_side(source, case, color)
            assert draw.selector_seed == case.selector_seeds[color]
            assert draw.candidate_id == "P10-D"
            assert draw.branch in ("neutral", "learned")


# ---------------------------------------------------------------------------
# Sealing and scope
# ---------------------------------------------------------------------------


def test_the_module_has_no_route_to_the_sealed_test_bank():
    """`validation_cases` takes no bank argument, so it cannot be steered.

    Checked against the code rather than the text: the docstring names the
    sealed bank precisely to say it is unreachable, so a substring search
    would fail on the sentence that makes the claim. What matters is that
    every call to the bank builder passes the validation constant.
    """
    signature = inspect.signature(validation.validation_cases)
    assert list(signature.parameters) == []

    tree = ast.parse(inspect.getsource(validation))
    builder_arguments = [
        [ast.unparse(argument) for argument in node.args]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_phase10_bank"
    ]
    assert builder_arguments == [["VALIDATION_BANK"]]
    assert validation.VALIDATION_BANK == "validation"

    # `clean=False`: the cleaned form differs from the raw constant, and it
    # is the raw constants that are being filtered.
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
    }
    code_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    ]
    assert not any("test_bank" in text or text == "test" for text in code_strings)


def test_the_neutral_arm_is_not_a_seventh_candidate():
    """The baseline has no candidate id and no direct-matchup cell."""
    with pytest.raises(validation.Phase10ValidationError):
        validation.cell_token(validation.ARM_NEUTRAL, "P10-D", "vs_random")
    with pytest.raises(validation.Phase10ValidationError):
        validation.cell_token(validation.ARM_NEUTRAL, None, MATCHUP_LEARNED_VS_NEUTRAL)
    with pytest.raises(validation.Phase10ValidationError):
        validation.cell_token(validation.ARM_LEARNED, None, "vs_random")
    assert validation.NEUTRAL_ARM_MATCHUPS == tuple(
        token for token in MATCHUP_TOKENS if token != MATCHUP_LEARNED_VS_NEUTRAL
    )


def test_the_landing_diagnostic_is_marked_report_only():
    rows = [
        {"own_fingerprint": "a"},
        {"own_fingerprint": "b"},
        {"own_fingerprint": "c"},
        {"own_fingerprint": "d"},
    ]
    counts = validation.landing_counts(rows, frozenset({"a"}))
    assert counts == {"games": 4, "landings": 1, "rate": 0.25, "gate": False, "use": "report_only"}


def test_an_errored_game_is_never_scored_as_a_loss():
    class Errored:
        match_id = "m"
        candidate_result = "error"

    with pytest.raises(validation.Phase10ValidationError):
        validation.game_score(Errored())
