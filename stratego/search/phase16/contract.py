"""Phase 16 Agent 2: the frozen identities of stochastic search.

Specification source:
`instructions/phase_16_robustness_and_distribution/02_AGENT_2_STOCHASTIC_SEARCH.md`

What this module freezes
------------------------
The temperature grids, the arm naming, the predeclared decision margins and
the Phase 16 seed streams. The search algorithm, the score definition, the
presets and the pairing table are all imported from the accepted Phase 15
contract — Phase 16 changes *how a move is drawn from the frozen scores*,
never how the scores are computed.

The seed streams
----------------
A new blake2b personalization, ``strat-p16s``, with new domain roots,
following the accepted Phase 15 derivation pattern exactly. The engine's
*world-sampling* seed is untouched: match play keeps the accepted
``search_seed_for(board_id, ply)`` stream and the Stage 1 diagnostic keeps
the accepted fixed ``DECISION_SEED``. The two new streams below exist only
for the two new draws — the root move sample and the rollout samples — so a
Phase 16 arm at zero temperature consumes exactly the randomness the frozen
player consumed: none.
"""

from __future__ import annotations

import hashlib

# Accepted identities, imported rather than restated.
from ..phase15.contract import (  # noqa: F401  (re-exported for phase16 readers)
    ACCEPTABLE_MOVE_SECONDS,
    PHASE15_SCORE_DEFINITION,
    PHASE15_SEARCH_VERSION,
    PREFERRED_MOVE_SECONDS,
    Phase15SearchError,
    preset as phase15_preset,
    search_seed_for,
)
from ..phase15.decisions import DECISION_SEED  # noqa: F401  (the accepted Stage A seed)

#: The Phase 16 stochastic-search identity. Any change to the sampling
#: definitions, the grids, the seed streams or the filter margins is a new
#: version, never a silent edit.
STOCHASTIC_VERSION = "phase16_stochastic_search_v1"

#: The identity stamped on a decision whose *rollouts* were sampled. A
#: decision produced with ``tau_r = 0`` keeps the accepted
#: `phase12_root_world_search_v1` identity because it is bit-identical to it.
ROLLOUT_SEARCH_VERSION = "phase16_sampled_rollout_search_v1"

#: The frozen candidate this agent delivers.
CANDIDATE_ARTIFACT_16 = "phase16_stochastic_candidate_v1"

#: The one production pairing under study — the Phase 15 selection.
STOCHASTIC_PAIRING = "p24_b24"

#: Where a report says this work sits.
PHASE16_STATUS_MARKERS = {
    "phase": "phase_16",
    "agent": "agent_02",
    "status": "engineering_deliverable_not_a_strength_claim",
    "scientific_validation_status": "not performed",
}


class Phase16StochasticError(Phase15SearchError):
    """A Phase 16 stochastic-search request was refused, or an invariant
    was violated. Subclasses the accepted Phase 15 error so every caller
    that already handles `Phase12SearchError` keeps working unchanged."""


# ---------------------------------------------------------------------------
# The grids (brief section 3) and predeclared margins
# ---------------------------------------------------------------------------

#: Move-sampling temperatures. 0 is the argmax control.
MOVE_TAUS = (0.0, 0.15, 0.30, 0.60)

#: Rollout-sampling temperatures. 0 is the greedy control.
ROLLOUT_TAUS = (0.0, 1.0)

#: The blunder-tail trim: rollout sampling is restricted to the smallest
#: legal set covering this probability mass (brief section 2).
ROLLOUT_TOP_P = 0.9

#: The two Stage 1 / Stage 2 budgets.
STAGE_BUDGETS = ("TINY", "MEDIUM")

#: Reseeded replays per (arm, position) in Stage 1.
STAGE1_REPLAYS = 16

#: Stage 1 predeclared filter (brief section 3): an arm survives if its mean
#: oracle Q-regret excess is within this margin of the tau=0 control's.
REGRET_EXCESS_MARGIN = 0.010

#: Stage 2 predeclared selection margin (brief section 4): an arm qualifies
#: if its MEDIUM EWR is within this of the tau=0 control's.
STAGE2_EWR_MARGIN = 0.05

#: Brief section 4's named fallback arm when no arm qualifies outright.
FALLBACK_TAU = 0.15
FALLBACK_TAU_R = 0.0


def tau_token(value: float) -> str:
    """`0.15 -> 't015'`; stable, collision-free for the declared grids."""
    scaled = round(float(value) * 100)
    if abs(float(value) * 100 - scaled) > 1e-9 or scaled < 0 or scaled > 999:
        raise Phase16StochasticError(
            f"temperature {value!r} does not have an exact 2-decimal token"
        )
    return f"t{scaled:03d}"


def tau_r_token(value: float) -> str:
    """`1.0 -> 'r100'`."""
    return "r" + tau_token(value)[1:]


def arm_name(tau: float, tau_r: float) -> str:
    """The stable arm identifier, e.g. `stoch_t015_r100`."""
    return f"stoch_{tau_token(tau)}_{tau_r_token(tau_r)}"


def parse_arm_name(arm_id: str) -> "tuple[float, float]":
    """`'stoch_t015_r100' -> (0.15, 1.0)`, refusing malformed names."""
    import re

    match = re.fullmatch(r"stoch_t(\d{3})_r(\d{3})", str(arm_id))
    if match is None:
        raise Phase16StochasticError(f"malformed Phase 16 arm id: {arm_id!r}")
    tau = int(match.group(1)) / 100.0
    tau_r = int(match.group(2)) / 100.0
    if arm_name(tau, tau_r) != arm_id:  # pragma: no cover - defensive
        raise Phase16StochasticError(f"arm id {arm_id!r} does not round-trip")
    return tau, tau_r


#: The argmax/greedy control's arm id.
CONTROL_ARM = arm_name(0.0, 0.0)


def grid_arms(taus=MOVE_TAUS, rollout_taus=ROLLOUT_TAUS) -> "list[tuple[float, float]]":
    """The full Stage 1 grid, control first, `(tau, tau_r)` pairs."""
    return [(float(t), float(r)) for r in rollout_taus for t in taus]


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------

#: The fresh Stage 1 position pack (Phase 15 pattern, new ordinals).
POSITION_PACK_VERSION = "phase16_agent02_positions_v1"

#: Board ordinal base of the Stage 1 position games. Phase 15 Stage B used
#: ordinals 0-1, its diagnostic pack 100-114; 200+ cannot collide with either.
POSITION_ORDINAL_BASE_16 = 200

#: The Stage 1 grid artifact.
STAGE1_VERSION = "phase16_stochastic_stage1_v1"

#: The Stage 2 fallback pack (brief section 4): a fresh 60-board balanced set
#: drawn exactly per the Phase 15 Stage C rules — one board per
#: (opponent x source x colour) cell — at a fresh ordinal.
INTERIM_PACK_VERSION = "phase16_agent02_interim_pack_v1"
INTERIM_PACK_ORDINAL = 2

#: The Stage 2 artifact.
STAGE2_VERSION = "phase16_stochastic_stage2_v1"

#: The repeat-encounter probe (brief section 5).
PROBE_VERSION = "phase16_repeat_encounter_probe_v1"
PROBE_ORDINAL_BASE = 300
PROBE_GAMES_PER_OPPONENT = 20
PROBE_OPPONENTS = ("p18", "p24")

#: Agent 1's declared handoff; used for Stage 2 boards when it has landed.
MEASUREMENT_HANDOFF_ARTIFACT = "phase16_measurement_handoff_v1"

# ---------------------------------------------------------------------------
# The working modes (brief section 6)
# ---------------------------------------------------------------------------

MODE_VARIED_STRENGTH = "varied_strength"
MODE_VARIED_FAST = "varied_fast"
VARIED_MODES = (MODE_VARIED_STRENGTH, MODE_VARIED_FAST)
VARIED_MODE_PRESETS = {MODE_VARIED_STRENGTH: "MEDIUM", MODE_VARIED_FAST: "TINY"}

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 16 *stochastic* stream. Distinct
#: from Phase 15's ``strat-p15s`` and every earlier phase.
PHASE16_STOCHASTIC_PERSON = b"strat-p16s"

#: Any change to the payload layout, the personalization or the roots is a
#: new identity version.
STOCHASTIC_IDENTITY_VERSION = "phase16_stochastic_identity_v1"

#: Master seed of this agent, folded into every stream below.
STOCHASTIC_MASTER_SEED = 2026082520

DOMAIN_MOVE_SAMPLE = "stoch_move_sample"
DOMAIN_ROLLOUT_SAMPLE = "stoch_rollout_sample"

STREAM_DOMAINS_16 = (DOMAIN_MOVE_SAMPLE, DOMAIN_ROLLOUT_SAMPLE)

DOMAIN_ROOTS_16 = {
    DOMAIN_MOVE_SAMPLE: 2026082521,
    DOMAIN_ROLLOUT_SAMPLE: 2026082522,
}
assert set(DOMAIN_ROOTS_16) == set(STREAM_DOMAINS_16)


def derive_stochastic_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 16 stochastic stream.

    The accepted Phase 15 derivation pattern (blake2b-8, personalized,
    version-prefixed, ':'-joined) rebuilt under the Phase 16 personalization
    so no Phase 12/14/15 stream can coincide with a Phase 16 stream even
    where two identity strings spell the same text.
    """
    if domain not in STREAM_DOMAINS_16:
        raise Phase16StochasticError(f"unknown Phase 16 stochastic domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase16StochasticError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase16StochasticError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            STOCHASTIC_IDENTITY_VERSION,
            domain,
            str(DOMAIN_ROOTS_16[domain]),
            str(STOCHASTIC_MASTER_SEED),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=PHASE16_STOCHASTIC_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def move_sample_seed(tau: float, tau_r: float, identifier: str, ply: int, replay: int = 0) -> int:
    """The root move-draw stream of one decision.

    Keyed by the full arm (both temperatures), the board/position identity,
    the ply and — Stage 1 only — the replay index. `tau = 0` consumes no
    randomness at all; the seed is defined anyway so logs can record it.
    """
    return derive_stochastic_seed(
        DOMAIN_MOVE_SAMPLE,
        tau_token(tau),
        tau_r_token(tau_r),
        str(identifier),
        int(ply),
        int(replay),
    )


def rollout_sample_seed(tau_r: float, top_p: float, identifier: str, ply: int, replay: int = 0) -> int:
    """The rollout-draw stream of one decision.

    Deliberately keyed by the *rollout* configuration only, not by the move
    temperature: every move-sampling arm at the same `tau_r` shares the same
    underlying searches, so a Stage 1 difference between two `tau` arms is a
    pure move-sampling effect on identical scores.
    """
    return derive_stochastic_seed(
        DOMAIN_ROLLOUT_SAMPLE,
        tau_r_token(tau_r),
        tau_token(top_p),
        str(identifier),
        int(ply),
        int(replay),
    )


__all__ = [
    "ACCEPTABLE_MOVE_SECONDS",
    "CANDIDATE_ARTIFACT_16",
    "CONTROL_ARM",
    "DECISION_SEED",
    "DOMAIN_MOVE_SAMPLE",
    "DOMAIN_ROLLOUT_SAMPLE",
    "FALLBACK_TAU",
    "FALLBACK_TAU_R",
    "INTERIM_PACK_ORDINAL",
    "INTERIM_PACK_VERSION",
    "MEASUREMENT_HANDOFF_ARTIFACT",
    "MODE_VARIED_FAST",
    "MODE_VARIED_STRENGTH",
    "MOVE_TAUS",
    "PHASE16_STATUS_MARKERS",
    "POSITION_ORDINAL_BASE_16",
    "POSITION_PACK_VERSION",
    "PROBE_GAMES_PER_OPPONENT",
    "PROBE_OPPONENTS",
    "PROBE_ORDINAL_BASE",
    "PROBE_VERSION",
    "REGRET_EXCESS_MARGIN",
    "ROLLOUT_SEARCH_VERSION",
    "ROLLOUT_TAUS",
    "ROLLOUT_TOP_P",
    "STAGE1_REPLAYS",
    "STAGE1_VERSION",
    "STAGE2_EWR_MARGIN",
    "STAGE2_VERSION",
    "STAGE_BUDGETS",
    "STOCHASTIC_MASTER_SEED",
    "STOCHASTIC_PAIRING",
    "STOCHASTIC_VERSION",
    "VARIED_MODES",
    "VARIED_MODE_PRESETS",
    "Phase16StochasticError",
    "arm_name",
    "derive_stochastic_seed",
    "parse_arm_name",
    "grid_arms",
    "move_sample_seed",
    "rollout_sample_seed",
    "tau_r_token",
    "tau_token",
]
