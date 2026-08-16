"""Regression: the two frozen train-config digests are distinct namespaces.

Agent 5's accepted artifact records two SHA-256 values:

```text
train_config_digest    the frozen `config` document (warmstart_train_config_v1)
trainer_config_digest  WarmstartTrainConfig.identity() (the runtime object)
```

Agent 6's first report issue printed the *runtime* digest under the heading
`warmstart_train_config_v1`, which reads as a contradiction of Agent 5's
recorded value and prompted an acceptance review. The configuration actually
run was never wrong; the label was.

These tests pin the distinction so no later agent can quietly conflate the two,
and so that a genuine drift — a runtime configuration that stops matching the
frozen payload — fails loudly instead of hiding behind whichever digest is
quoted.
"""

import hashlib
import json
from pathlib import Path

import pytest

from stratego.training import warmstart_contract as wc
from stratego.training.warmstart_pilot import verify_frozen_train_config
from stratego.training.warmstart_seed import CANONICAL_SEEDS
from stratego.training.warmstart_trainer import WarmstartTrainConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = (
    REPOSITORY_ROOT / "reports" / "phase_8_data" / "agent_05_frozen_train_config.json"
)

pytestmark = pytest.mark.skipif(
    not FROZEN_CONFIG_PATH.exists(),
    reason="Agent 5's frozen train config has not been produced yet",
)


def canonical_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(FROZEN_CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def runtime_config(frozen) -> WarmstartTrainConfig:
    # Device is deliberately CPU here: the digest covers `device`, so the
    # fixture rebuilds the frozen construction exactly, which names mps.
    return WarmstartTrainConfig.from_pilot_candidate(
        frozen["winning_candidate_id"],
        device=frozen["trainer_config_identity"]["device"],
        validation_batches=frozen["trainer_config_identity"]["validation_batches"],
    )


def test_frozen_payload_verifies(frozen):
    assert verify_frozen_train_config(frozen) == []


def test_document_digest_covers_the_config_document(frozen):
    """`train_config_digest` hashes payload['config'], nothing else."""
    assert canonical_digest(frozen["config"]) == frozen["train_config_digest"]


def test_runtime_digest_covers_the_trainer_identity(frozen, runtime_config):
    """`trainer_config_digest` hashes WarmstartTrainConfig.identity()."""
    assert runtime_config.digest() == frozen["trainer_config_digest"]
    assert canonical_digest(runtime_config.identity()) == frozen["trainer_config_digest"]


def test_the_two_digests_are_different_namespaces(frozen, runtime_config):
    """They cover different objects, so equality was never possible.

    This is the regression proper: a future change that made one digest stand
    in for the other — or that collapsed the two field sets — would break here
    rather than in a report sentence.
    """
    document = frozen["config"]
    identity = runtime_config.identity()
    assert frozen["train_config_digest"] != frozen["trainer_config_digest"]
    assert set(document) - set(identity), "document must carry fields the runtime lacks"
    assert set(identity) - set(document), "runtime must carry fields the document lacks"


def test_runtime_identity_equals_the_recorded_one(frozen, runtime_config):
    assert runtime_config.identity() == frozen["trainer_config_identity"]


@pytest.mark.parametrize(
    "document_field,runtime_field",
    [
        ("learning_rate", "learning_rate"),
        ("batch_size", "batch_size"),
        ("weight_decay", "weight_decay"),
        ("gradient_clip_norm", "gradient_clip_norm"),
        ("warmup_steps", "warmup_steps"),
        ("lambda_policy", "lambda_policy"),
        ("lambda_value", "lambda_value"),
        ("lambda_belief", "lambda_belief"),
        ("model_candidate", "model_candidate"),
        ("model_init_seed", "model_init_seed"),
        ("precision", "precision"),
        ("device", "device"),
        ("lr_schedule", "lr_schedule"),
        ("validation_cadence_updates", "validation_cadence_updates"),
        ("validation_batches", "validation_batches"),
        ("validation_split", "validation_split"),
        ("validation_selection", "validation_selection"),
        # The naming bridge between Agent 5's vocabulary and the trainer's.
        ("train_split", "split"),
        ("train_order", "order"),
    ],
)
def test_shared_fields_agree_across_the_bridge(
    frozen, runtime_config, document_field, runtime_field
):
    assert frozen["config"][document_field] == runtime_config.identity()[runtime_field]


def test_adam_betas_bridge(frozen, runtime_config):
    identity = runtime_config.identity()
    assert list(frozen["config"]["adam_betas"]) == [
        identity["adam_beta1"],
        identity["adam_beta2"],
    ]


@pytest.mark.parametrize(
    "document_field,live_value",
    [
        ("train_shuffle_seed", CANONICAL_SEEDS["train_order_seed"]),
        ("model_config_digest", wc.EXPECTED_C1_CONFIG_DIGEST),
    ],
)
def test_document_only_fields_match_their_live_source(
    frozen, document_field, live_value
):
    """Fields the runtime identity does not carry still bind the run."""
    assert frozen["config"][document_field] == live_value
