"""Stratego Phase Two reference engine.

Public entry points, grouped by the specification area they implement:

- geometry and encoding: :mod:`constants`, :mod:`coordinates`, :mod:`actions`
- pieces and setup: :mod:`pieces`, :mod:`setup`
- state and rules: :mod:`state`, :mod:`legal_moves`, :mod:`combat`, :mod:`transition`
- information: :mod:`behavior`, :mod:`observation`, :mod:`events`
- tooling: :mod:`replay`, :mod:`snapshot`, :mod:`invariants`, :mod:`random_play`,
  :mod:`permutation`
"""

from .actions import (
    ACTION_SPACE_SIZE,
    action_from_perspective,
    action_to_perspective,
    decode_action,
    describe_action,
    encode_action,
)
from .combat import ATTACKER_WINS, BOTH_REMOVED, DEFENDER_WINS, resolve_combat
from .constants import (
    BLUE,
    EVALUATION_RULES,
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RED,
    RULES_VERSION,
    RulesConfig,
    TRAINING_RULES,
)
from .events import filter_events_for_observer, public_board_view, public_setup_view
from .invariants import InvariantViolation, capture_baseline, capture_knowledge, check_invariants
from .legal_moves import adjacent_attack_opportunities, legal_action_mask, legal_actions
from .observation import (
    belief_target,
    build_observation,
    observation_and_mask,
    observation_channel_metadata,
    observation_metadata_document,
)
from .permutation import hidden_opponent_piece_ids, permute_hidden_identities
from .random_play import generate_random_games, play_random_game, select_random_action
from .replay import ReplayRecord, build_replay_record, rebuild_final_state, replay_plies
from .setup import SetupError, random_setup, validate_setup, validate_setup_placement
from .snapshot import clone_state, create_snapshot, restore_snapshot
from .state import GameState, create_game, render_board, state_fingerprint
from .transition import IllegalActionError, TerminalStateError, apply_action

__all__ = [
    "ACTION_SPACE_SIZE",
    "ATTACKER_WINS",
    "BLUE",
    "BOTH_REMOVED",
    "DEFENDER_WINS",
    "EVALUATION_RULES",
    "GameState",
    "IMPLEMENTATION_VERSION",
    "IllegalActionError",
    "InvariantViolation",
    "OBSERVATION_VERSION",
    "RED",
    "RULES_VERSION",
    "ReplayRecord",
    "RulesConfig",
    "SetupError",
    "TRAINING_RULES",
    "TerminalStateError",
    "action_from_perspective",
    "action_to_perspective",
    "adjacent_attack_opportunities",
    "apply_action",
    "belief_target",
    "build_observation",
    "build_replay_record",
    "capture_baseline",
    "capture_knowledge",
    "check_invariants",
    "clone_state",
    "create_game",
    "create_snapshot",
    "decode_action",
    "describe_action",
    "encode_action",
    "filter_events_for_observer",
    "generate_random_games",
    "hidden_opponent_piece_ids",
    "legal_action_mask",
    "legal_actions",
    "observation_and_mask",
    "observation_channel_metadata",
    "observation_metadata_document",
    "permute_hidden_identities",
    "play_random_game",
    "public_board_view",
    "public_setup_view",
    "rebuild_final_state",
    "render_board",
    "replay_plies",
    "resolve_combat",
    "restore_snapshot",
    "select_random_action",
    "state_fingerprint",
    "validate_setup",
    "validate_setup_placement",
]
