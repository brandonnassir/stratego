"""Phase 7 Agent 5: setup provenance must never cross the model boundary.

The training system is *allowed* to know the true setups -- they are part of
privileged replay and training state, and `trajectory_v1` has always stored
both of them. What the move policy must never receive is the opponent's true
setup or any setup provenance that names hidden identities:

```text
opponent family id
opponent base setup id
opponent perturbation seed
opponent reflected bit
true opponent setup
```

The proof here has three independent legs:

1. **Transport.** The shared buffers are the only channel between a worker and
   the coordinator. Their field list is fixed, numeric and provenance-free, and
   the observation rows they carry are byte-identical to
   `build_observation(state, mover)` -- so the transport carries one
   perspective and nothing else.
2. **The neural request.** The model is called with exactly one tensor, and
   that tensor is the observation block. Captured from a real coordinator run
   and compared against the buffer itself.
3. **Reachability.** An object-graph walk from the captured model inputs, and
   from the coordinator object as a whole, finds no live provenance value:
   no family id, base setup id, perturbation seed, fingerprint or setup string
   of any game in flight.
"""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stratego.engine.constants import (  # noqa: E402
    OBSERVATION_CHANNELS,
    OBSERVATION_SHAPE,
    OBSERVATION_VERSION,
    PLAYERS,
)
from stratego.engine.observation import (  # noqa: E402
    build_observation,
    observation_metadata_document,
)
from stratego.engine.state import create_game  # noqa: E402
from stratego.setups.sampler import provenance_is_observer_safe  # noqa: E402
from stratego.training.coordinator import (  # noqa: E402
    CoordinatorConfig,
    SelfPlayCoordinator,
)
from stratego.training.setup_source import (  # noqa: E402
    PROVENANCE_PLAYER_EXTRA_FIELDS,
    training_setup_source,
)
from stratego.training.shared_buffers import (  # noqa: E402
    COORDINATOR_WRITTEN_FIELDS,
    WORKER_WRITTEN_FIELDS,
)
from stratego.training.trajectory import GameRecord  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#: Provenance field names that would betray a hidden identity if any of them
#: ever appeared on the transport or in a model input.
FORBIDDEN_PROVENANCE_FIELDS = (
    "primary_family_id",
    "family_key",
    "base_setup_id",
    "base_index",
    "base_fingerprint",
    "perturbation_seed",
    "perturbation_id",
    "reflection_applied",
    "final_setup",
    "final_setup_fingerprint",
    "final_setup_class_fingerprint",
    "setup_provenance",
    "red_setup",
    "blue_setup",
    "engine_setup",
    "side_seed",
)

#: Node cap for the reachability walk. Large enough to exhaust the graphs under
#: test and small enough that a runaway traversal fails loudly instead of
#: hanging.
REACHABILITY_NODE_LIMIT = 400_000

_ATOMIC = (bytes, bytearray, int, float, bool, complex, type(None))

#: Code-shaped nodes the walk stops at. Following any of them leaves the
#: object's own data and enters the module graph.
_NOT_FOLLOWED = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.CodeType,
    types.FrameType,
    types.TracebackType,
    np.ndarray,
    torch.Tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reachable_strings(roots, *, limit: int = REACHABILITY_NODE_LIMIT) -> set:
    """Every `str` an object *holds*, transitively.

    A bounded walk over data edges only: containers, dictionaries and instance
    attributes. Code objects are deliberately not followed -- a class, function
    or module in the graph is an import, and following one reaches the whole
    interpreter, which would make the walk prove nothing. What is left is
    exactly the question being asked: which values does this object carry?

    Raises if the node cap is hit, so a silent truncation cannot turn into a
    passing test.
    """
    seen: set[int] = set()
    found: set[str] = set()
    stack = list(roots)
    visited = 0
    while stack:
        item = stack.pop()
        identifier = id(item)
        if identifier in seen:
            continue
        seen.add(identifier)
        visited += 1
        if visited > limit:  # pragma: no cover - guard against a runaway walk
            raise AssertionError("reachability walk exceeded its node limit")

        if isinstance(item, str):
            found.add(item)
            continue
        if isinstance(item, _ATOMIC) or isinstance(item, _NOT_FOLLOWED):
            continue
        if isinstance(item, dict):
            for key, value in item.items():
                stack.append(key)
                stack.append(value)
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
            continue
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            stack.append(attributes)
        for name in getattr(type(item), "__slots__", ()) or ():
            if isinstance(name, str) and hasattr(item, name):
                stack.append(getattr(item, name))
    return found


def live_provenance_values(source, root_seed: int, environments: int) -> set:
    """Every provenance string of the generation-0 games of a run."""
    values: set[str] = set()
    for environment_id in range(environments):
        provenance = source.assign(
            root_seed=root_seed, environment_id=environment_id, generation=0
        ).provenance
        for side in ("red", "blue"):
            player = provenance[side]
            values.add(str(player["primary_family_id"]))
            values.add(str(player["base_setup_id"]))
            values.add(str(player["final_setup_fingerprint"]))
            values.add(str(player["final_setup"]))
            values.add(str(player["engine_setup"]))
    return values


class _RecordingModel(torch.nn.Module):
    """Wraps the real model and records exactly what the coordinator passes it."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls: list[tuple] = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        return self.inner(*args, **kwargs)


@pytest.fixture(scope="module")
def source():
    return training_setup_source()


@pytest.fixture(scope="module")
def run(source):
    """One real coordinator run over library setups, with the model calls kept."""
    config = CoordinatorConfig(
        num_environments=16,
        num_workers=2,
        inference_batch_size=8,
        root_seed=70_007,
        precision="float32" if DEVICE == "cpu" else "float16",
        detailed_timing=False,
        setup_source=source,
    )
    coordinator = SelfPlayCoordinator(config, device=DEVICE)
    coordinator.model = _RecordingModel(coordinator.model).to(
        device=coordinator.device, dtype=coordinator.dtype
    )
    coordinator.start()
    try:
        # The buffer state the first step will read, captured before it runs.
        startup = coordinator.pool.buffers.observations.copy()
        environment_ids = coordinator.pool.buffers.environment_id.copy()
        generations = coordinator.pool.buffers.generation.copy()
        coordinator.model.calls.clear()
        coordinator.step()
        first_step_calls = list(coordinator.model.calls)
        for _ in range(2):
            coordinator.step()
        calls = list(coordinator.model.calls)
        strings = reachable_strings([coordinator])
    finally:
        coordinator.shutdown()
    return {
        "config": config,
        "calls": calls,
        "first_step_calls": first_step_calls,
        "dtype": coordinator.dtype,
        "startup_observations": startup,
        "environment_ids": environment_ids,
        "generations": generations,
        "coordinator_strings": strings,
    }


# ---------------------------------------------------------------------------
# 1. The transport
# ---------------------------------------------------------------------------


def test_no_setup_provenance_field_exists_on_the_transport():
    published = set(WORKER_WRITTEN_FIELDS) | set(COORDINATOR_WRITTEN_FIELDS)
    for forbidden in FORBIDDEN_PROVENANCE_FIELDS:
        assert forbidden not in published


def test_the_published_observation_is_the_state_and_nothing_else(run, source):
    """Byte-identical to a locally rebuilt game's observation for the mover."""
    config = run["config"]
    checked = 0
    for slot in range(config.num_environments):
        environment_id = int(run["environment_ids"][slot])
        generation = int(run["generations"][slot])
        assignment = source.assign(
            root_seed=config.root_seed,
            environment_id=environment_id,
            generation=generation,
        )
        state = create_game(
            assignment.red_setup,
            assignment.blue_setup,
            rules=config.rules,
            game_id=f"probe-{environment_id}-{generation}",
        )
        if state.terminal:
            continue
        expected = build_observation(state, state.acting_player)
        assert np.array_equal(run["startup_observations"][slot], expected)
        checked += 1
    assert checked == config.num_environments


# ---------------------------------------------------------------------------
# 2. The neural request
# ---------------------------------------------------------------------------


def test_the_model_is_called_with_one_observation_tensor_and_nothing_else(run):
    assert run["calls"], "the coordinator never called the model"
    for args, kwargs in run["calls"]:
        assert kwargs == {}
        assert len(args) == 1
        tokens = args[0]
        assert isinstance(tokens, torch.Tensor)
        # `(rows, tokens, channels)`: the observation block, token-major.
        assert tokens.ndim == 3
        assert tokens.shape[1] == OBSERVATION_SHAPE[1] * OBSERVATION_SHAPE[2]
        assert tokens.shape[2] == OBSERVATION_CHANNELS


def test_the_inference_input_is_the_observation_block(run):
    """The tensor the model saw is the transported observation, element for element."""
    config = run["config"]
    tokens = run["first_step_calls"][0][0][0].detach().cpu()
    rows = tokens.shape[0]
    assert rows == config.inference_batch_size
    expected = (
        torch.from_numpy(
            np.ascontiguousarray(
                run["startup_observations"][:rows].reshape(
                    rows, OBSERVATION_CHANNELS, -1
                )
            )
        )
        .transpose(1, 2)
        .contiguous()
        .to(run["dtype"])
        .cpu()
    )
    assert torch.equal(tokens, expected)


def test_no_provenance_is_reachable_from_the_model_inputs(run, source):
    forbidden = live_provenance_values(
        source, run["config"].root_seed, run["config"].num_environments
    )
    for args, kwargs in run["calls"]:
        strings = reachable_strings(list(args) + list(kwargs.values()))
        assert not (strings & forbidden)
        assert not (strings & set(FORBIDDEN_PROVENANCE_FIELDS))


def test_the_reachability_walk_would_find_provenance_if_it_were_there(run, source):
    """Positive control: the negative results above are not vacuous.

    The same walk, over the same inputs with one provenance record attached,
    must report every value the real walk reported absent.
    """
    provenance = source.assign(
        root_seed=run["config"].root_seed, environment_id=0, generation=0
    ).provenance
    forbidden = live_provenance_values(source, run["config"].root_seed, 1)
    args, kwargs = run["calls"][0]

    clean = reachable_strings(list(args) + list(kwargs.values()))
    assert not (clean & forbidden)

    contaminated = reachable_strings(
        [{"inputs": list(args), "leak": provenance}]
    )
    assert forbidden <= contaminated
    assert "base_setup_id" in contaminated

    # And the coordinator walk is equally capable of seeing one.
    assert forbidden <= reachable_strings([{"coordinator": provenance}])


# ---------------------------------------------------------------------------
# 3. Reachability from the coordinator
# ---------------------------------------------------------------------------


def test_no_live_provenance_is_reachable_from_the_coordinator(run, source):
    """The coordinator process holds configuration, never a sampled identity.

    Provenance is produced inside the workers and written to their own
    sidecars; nothing carries it back across a pipe.
    """
    forbidden = live_provenance_values(
        source, run["config"].root_seed, run["config"].num_environments
    )
    assert not (run["coordinator_strings"] & forbidden)


def test_the_coordinator_holds_the_source_configuration_only(run):
    """The split and profile are configuration and may be known; identities are not."""
    strings = run["coordinator_strings"]
    assert "train" in strings
    for forbidden in ("base_setup_id", "final_setup_fingerprint", "perturbation_seed"):
        assert forbidden not in strings


# ---------------------------------------------------------------------------
# Frozen contracts
# ---------------------------------------------------------------------------


def test_the_observation_contract_gained_no_phase_7_channel():
    assert OBSERVATION_VERSION == "observation_v2_1_127ch"
    assert OBSERVATION_CHANNELS == 127
    assert OBSERVATION_SHAPE == (127, 10, 10)
    document = observation_metadata_document()
    assert document["observation_version"] == OBSERVATION_VERSION
    assert len(document["channels"]) == 127
    names = {entry["name"] for entry in document["channels"]}
    for forbidden in FORBIDDEN_PROVENANCE_FIELDS:
        assert forbidden not in names
    assert not [name for name in names if "family" in name or "provenance" in name]


def test_the_trajectory_record_carries_no_setup_provenance():
    """`trajectory_v1` keeps the true setups it always kept, and nothing new."""
    fields = {field.name for field in GameRecord.__dataclass_fields__.values()}
    assert {"red_setup", "blue_setup", "setup_family", "setup_id"} <= fields
    for forbidden in (
        "primary_family_id",
        "base_setup_id",
        "perturbation_seed",
        "reflection_applied",
        "final_setup_fingerprint",
        "setup_provenance",
        "split",
    ):
        assert forbidden not in fields
    for forbidden in PROVENANCE_PLAYER_EXTRA_FIELDS:
        assert forbidden not in fields


def test_the_setup_family_label_names_no_individual_setup(source):
    """The one `trajectory_v1` string this integration sets reveals nothing hidden."""
    label = source.setup_family
    assert "F0" not in label and "F1" not in label
    provenance = source.assign(
        root_seed=1, environment_id=0, generation=0
    ).provenance
    for side in ("red", "blue"):
        assert provenance[side]["base_setup_id"] not in label
        assert provenance[side]["final_setup_fingerprint"] not in label


def test_provenance_carries_no_outcome_or_strength_signal(source):
    """Phase 7 provenance is structural metadata; it must name no result."""
    provenance = source.assign(
        root_seed=2, environment_id=1, generation=0
    ).provenance
    for side in ("red", "blue"):
        assert provenance_is_observer_safe(provenance[side]) == []


def test_a_player_cannot_read_the_opponents_provenance_from_its_own(source):
    """The two sides' records are separate objects with independent identities."""
    provenance = source.assign(
        root_seed=3, environment_id=2, generation=0
    ).provenance
    red, blue = provenance["red"], provenance["blue"]
    assert red["player"] == PLAYERS[0]
    assert blue["player"] == PLAYERS[1]
    assert red is not blue
    assert red["side_seed"] != blue["side_seed"]
    # Nothing in one side's record names the other's setup.
    assert blue["final_setup"] not in red.values()
    assert red["final_setup"] not in blue.values()
