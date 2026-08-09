"""Action encoding for the fixed 10,000-entry source-destination action space.

Specification sources:

- `03_game_engine_spec.md` section 8
- `06_observation_v2_127ch.md` section 14 (the mask is a separate model input)

`action_id = 100 * source + destination`, so the space is exactly 10,000 wide and
covers ordinary moves and long Scout moves without a special action type.

Absolute (engine) indices are authoritative. Perspective-normalized identifiers
exist only for the benefit of a future single network that plays both colours,
and are produced by mapping both endpoints through the observation's coordinate
normalization.
"""

from .constants import ACTION_SPACE_SIZE, NUM_SQUARES
from .coordinates import to_perspective


def encode_action(source: int, destination: int) -> int:
    """Pack a source/destination pair into a dense action identifier."""
    if not 0 <= source < NUM_SQUARES:
        raise ValueError(f"source square out of range: {source}")
    if not 0 <= destination < NUM_SQUARES:
        raise ValueError(f"destination square out of range: {destination}")
    return NUM_SQUARES * source + destination


def decode_action(action_id: int) -> tuple[int, int]:
    """Unpack an action identifier into `(source, destination)`."""
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError(f"action id out of range: {action_id}")
    return divmod(action_id, NUM_SQUARES)


def action_source(action_id: int) -> int:
    return action_id // NUM_SQUARES


def action_destination(action_id: int) -> int:
    return action_id % NUM_SQUARES


def action_to_perspective(action_id: int, observer: int) -> int:
    """Rewrite an absolute action identifier into the observer's normalized frame."""
    source, destination = decode_action(action_id)
    return encode_action(
        to_perspective(source, observer), to_perspective(destination, observer)
    )


def action_from_perspective(action_id: int, observer: int) -> int:
    """Inverse of :func:`action_to_perspective`.

    The underlying square transform is an involution, so the implementation is
    the same mapping; the separate name keeps call sites readable.
    """
    return action_to_perspective(action_id, observer)


def describe_action(action_id: int) -> str:
    """Human-readable form such as `a4->a5`, used by tests and the report."""
    from .coordinates import square_name  # local import avoids a cycle at import time

    source, destination = decode_action(action_id)
    return f"{square_name(source)}->{square_name(destination)}"
