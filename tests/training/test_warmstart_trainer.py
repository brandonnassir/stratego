"""Phase 8 Agent 4: trainer semantics on the mini corpus.

CPU + C0 scale: these tests prove the *logic* — frozen-candidate enforcement,
exact resume, validation isolation, worker-count independence — while the
full-scale MPS evidence (1,000-step split run, soak, throughput) lives in
`scripts/run_phase8_agent04.py` and its artifacts.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from stratego.training.warmstart_checkpoint import (
    CorpusIdentity,
    WarmstartCheckpointCompatibilityError,
    measure_corpus_identity,
)
from stratego.training.warmstart_contract import HeldOutAccessError
from stratego.training.warmstart_dataset import WarmstartDataset
from stratego.training.warmstart_metrics import run_validation
from stratego.training.warmstart_trainer import (
    STEP_METRIC_COLUMNS,
    LoaderTopology,
    WarmstartTrainConfig,
    WarmstartTrainer,
    WarmstartTrainerError,
    pilot_candidate_ids,
    unit_test_config,
)

#: Hermetic value prior for tests: the frozen production prior lives in the
#: Agent 3 artifact and is irrelevant to the logic under test here.
UNIFORM_PRIOR = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


@pytest.fixture(scope="module")
def mini(warmstart_mini_corpus):
    root, _game_ids = warmstart_mini_corpus
    return root, measure_corpus_identity(root)


def build_trainer(root, identity, *, workers: int = 1, **config_overrides):
    config = unit_test_config(**config_overrides)
    return WarmstartTrainer(
        config,
        identity,
        root=root,
        topology=LoaderTopology(workers=workers, prefetch=1, record_cache_size=16),
        require_complete_split=False,
        value_prior=UNIFORM_PRIOR,
    )


class TestConfigurationDiscipline:
    def test_every_frozen_candidate_constructs(self):
        for candidate_id in pilot_candidate_ids():
            config = WarmstartTrainConfig.from_pilot_candidate(candidate_id, device="cpu")
            assert config.candidate_id == candidate_id
            assert config.model_candidate == "C1"
            assert config.batch_size == 256
            assert config.lr_schedule == "linear_warmup_500_steps_then_constant"

    def test_unknown_candidate_is_refused(self):
        with pytest.raises(WarmstartTrainerError, match="frozen"):
            WarmstartTrainConfig.from_pilot_candidate("ws_pilot_lr5e-4_balanced")

    @pytest.mark.parametrize(
        "field, value",
        [
            ("learning_rate", 5e-4),
            ("batch_size", 512),
            ("weight_decay", 0.1),
            ("warmup_steps", 100),
            ("lambda_belief", 2.0),
            ("model_init_seed", 1),
        ],
    )
    def test_off_matrix_values_are_refused(self, field, value):
        config = WarmstartTrainConfig.from_pilot_candidate(
            "ws_pilot_lr3e-4_balanced", device="cpu"
        )
        with pytest.raises(WarmstartTrainerError, match="frozen"):
            dataclasses.replace(config, **{field: value})

    def test_unit_test_scope_is_fenced(self):
        with pytest.raises(WarmstartTrainerError, match="unittest_"):
            unit_test_config(candidate_id="pilot_lookalike")
        with pytest.raises(WarmstartTrainerError, match="C0"):
            unit_test_config(model_candidate="C2")

    def test_test_split_never_enters_a_config(self):
        with pytest.raises(WarmstartTrainerError, match="test"):
            unit_test_config(split="test")
        with pytest.raises(WarmstartTrainerError, match="test"):
            unit_test_config(validation_split="test")

    def test_float32_is_the_only_precision(self):
        with pytest.raises(WarmstartTrainerError, match="float32"):
            dataclasses.replace(
                unit_test_config(), precision="float16"
            )

    def test_trainer_requires_a_corpus_identity(self, mini):
        root, identity = mini
        with pytest.raises(WarmstartTrainerError, match="CorpusIdentity"):
            WarmstartTrainer(unit_test_config(), identity.to_dict(), root=root)


class TestTrainingUpdates:
    def test_updates_advance_state_and_report_every_metric(self, mini):
        root, identity = mini
        trainer = build_trainer(root, identity, learning_rate=1e-3, warmup_steps=3)
        with trainer:
            before = trainer.parameter_snapshot()
            rows = trainer.train_updates(3)
        assert trainer.global_step == 3
        assert trainer.examples_consumed == 24
        after = trainer.parameter_snapshot()
        assert any(
            not torch.equal(before[name], after[name]) for name in before
        ), "three optimizer updates left every parameter untouched"
        for row in rows:
            for column in STEP_METRIC_COLUMNS:
                assert column in row, f"metric row is missing {column}"
        # The versioned warmup: update k uses base_lr * min(1, k/3).
        base = trainer.config.learning_rate
        observed = [row["learning_rate"] for row in rows]
        assert observed == pytest.approx([base / 3, base * 2 / 3, base])
        assert all(value == 0 for value in trainer.counters.values())

    def test_worker_pool_serves_bit_identical_batches(self, mini):
        root, identity = mini

        def digests(workers: int) -> list:
            trainer = build_trainer(root, identity, workers=workers)
            with trainer:
                rows = trainer.train_updates(3, capture_batch_digests=True)
            return [(row["keys_digest"], row["batch_digest"]) for row in rows]

        assert digests(1) == digests(2)


class TestResumeEquivalence:
    def test_split_run_equals_uninterrupted_run_exactly(self, mini, tmp_path):
        root, identity = mini
        overrides = {
            "learning_rate": 1e-3,
            "warmup_steps": 3,
            "validation_cadence_updates": 2,
            "validation_batches": 1,
        }

        uninterrupted = build_trainer(root, identity, **overrides)
        with uninterrupted:
            straight_rows = uninterrupted.train_updates(6, capture_batch_digests=True)

        first = build_trainer(root, identity, **overrides)
        with first:
            split_rows = first.train_updates(3, capture_batch_digests=True)
            checkpoint = tmp_path / "split.ckpt"
            first.save_checkpoint(checkpoint)
            config = first.config
        del first

        resumed = WarmstartTrainer.resume(
            checkpoint,
            config=config,
            corpus_identity=identity,
            root=root,
            topology=LoaderTopology(workers=1, prefetch=1, record_cache_size=16),
            require_complete_split=False,
            value_prior=UNIFORM_PRIOR,
        )
        with resumed:
            split_rows += resumed.train_updates(3, capture_batch_digests=True)

        # Same batch identities and bytes at every compared step.
        assert [row["keys_digest"] for row in split_rows] == [
            row["keys_digest"] for row in straight_rows
        ]
        assert [row["batch_digest"] for row in split_rows] == [
            row["batch_digest"] for row in straight_rows
        ]
        # Same learning-rate trajectory through the warmup boundary.
        assert [row["learning_rate"] for row in split_rows] == pytest.approx(
            [row["learning_rate"] for row in straight_rows]
        )
        # Same logical state: step, examples, cursor, scheduler, best logic,
        # validation cadence and optimizer state structure.
        assert resumed.state_summary() == uninterrupted.state_summary()
        # CPU float32 is exactly deterministic here: bitwise-equal parameters.
        resumed_parameters = resumed.parameter_snapshot()
        for name, tensor in uninterrupted.parameter_snapshot().items():
            assert torch.equal(resumed_parameters[name], tensor), name
        # Bitwise-equal optimizer moments too.
        straight_state = uninterrupted.optimizer.state_dict()["state"]
        resumed_state = resumed.optimizer.state_dict()["state"]
        assert straight_state.keys() == resumed_state.keys()
        for index in straight_state:
            for key, value in straight_state[index].items():
                other = resumed_state[index][key]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value.cpu(), other.cpu()), (index, key)
                else:
                    assert value == other, (index, key)
        # The validation history is identical in every logical field; the
        # recorded wall-clock seconds are diagnostics and legitimately differ.
        def logical(history):
            return [
                {key: value for key, value in entry.items() if key != "seconds"}
                for entry in history
            ]

        assert logical(resumed.validation_history) == logical(
            uninterrupted.validation_history
        )

    def test_resume_refuses_a_foreign_configuration(self, mini, tmp_path):
        root, identity = mini
        trainer = build_trainer(root, identity, candidate_id="unittest_run_a")
        with trainer:
            trainer.train_updates(1)
            checkpoint = tmp_path / "run_a.ckpt"
            trainer.save_checkpoint(checkpoint)
        foreign = unit_test_config(candidate_id="unittest_run_b")
        with pytest.raises(WarmstartCheckpointCompatibilityError, match="candidate_id"):
            WarmstartTrainer.resume(
                checkpoint,
                config=foreign,
                corpus_identity=identity,
                root=root,
                require_complete_split=False,
                value_prior=UNIFORM_PRIOR,
            )

    def test_resume_refuses_a_different_corpus(self, mini, tmp_path):
        root, identity = mini
        trainer = build_trainer(root, identity)
        with trainer:
            trainer.train_updates(1)
            checkpoint = tmp_path / "corpus_bound.ckpt"
            trainer.save_checkpoint(checkpoint)
            config = trainer.config
        drifted = CorpusIdentity(
            corpus_version=identity.corpus_version,
            content_digest="f" * 64,
            metadata_digest=identity.metadata_digest,
            commit_index_digest=identity.commit_index_digest,
        )
        with pytest.raises(WarmstartCheckpointCompatibilityError, match="corpus"):
            WarmstartTrainer.resume(
                checkpoint,
                config=config,
                corpus_identity=drifted,
                root=root,
                require_complete_split=False,
                value_prior=UNIFORM_PRIOR,
            )


class TestValidationIsolation:
    def test_validation_mutates_no_training_state(self, mini):
        root, identity = mini
        trainer = build_trainer(
            root, identity, validation_cadence_updates=1000000, validation_batches=1
        )
        with trainer:
            trainer.train_updates(2)
            parameters_before = trainer.parameter_snapshot()
            optimizer_before = trainer.optimizer.state_dict()
            cursor_before = trainer.cursor
            scheduler_before = trainer.scheduler.state_dict()
            step_before = trainer.global_step
            examples_before = trainer.examples_consumed
            assert trainer.model.training

            entry = trainer.run_cadence_validation()
            assert entry["examples"] > 0

            assert trainer.model.training, "validation left the model out of train mode"
            assert trainer.cursor == cursor_before
            assert trainer.global_step == step_before
            assert trainer.examples_consumed == examples_before
            assert trainer.scheduler.state_dict() == scheduler_before
            parameters_after = trainer.parameter_snapshot()
            for name, tensor in parameters_before.items():
                assert torch.equal(parameters_after[name], tensor), name
            optimizer_after = trainer.optimizer.state_dict()
            assert optimizer_before["param_groups"] == optimizer_after["param_groups"]
            for index in optimizer_before["state"]:
                for key, value in optimizer_before["state"][index].items():
                    other = optimizer_after["state"][index][key]
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value, other), (index, key)
                    else:
                        assert value == other, (index, key)

    def test_validation_uses_train_and_validation_only(self, mini):
        root, identity = mini
        trainer = build_trainer(root, identity)
        # The training split feeds updates; the validation split feeds metrics;
        # the sealed test split is refused by the frozen access gate.
        model = trainer.model
        dataset = WarmstartDataset(root, record_cache_size=8, require_complete_split=False)
        with pytest.raises(HeldOutAccessError, match="sealed"):
            run_validation(
                model,
                dataset,
                split="test",
                value_prior=UNIFORM_PRIOR,
                batches=1,
                batch_size=4,
            )
        trainer.close()

    def test_spread_positions_cover_the_split(self):
        from stratego.training.warmstart_metrics import spread_batch_positions

        positions = spread_batch_positions(1_000_000, 256, 8)
        assert len(positions) == 8
        assert list(positions) == sorted(set(positions))
        assert positions[0] == 0
        assert all(position % 256 == 0 for position in positions)
        assert positions[-1] <= 1_000_000 - 256
        # The selection spans the grid rather than sampling a prefix — the
        # sequential order is cell-major, and the first cells are the
        # policy-unsupervised random-vs-random games.
        assert positions[-1] > 500_000
        # More batches than the grid holds degrades to the whole grid.
        assert list(spread_batch_positions(20, 8, 5)) == [0, 8]

    def test_validation_restores_eval_mode_too(self, mini):
        root, identity = mini
        trainer = build_trainer(root, identity, validation_batches=1)
        with trainer:
            trainer.model.eval()
            run_validation(
                trainer.model,
                trainer.validation_dataset,
                split="validation",
                value_prior=UNIFORM_PRIOR,
                batches=1,
                batch_size=4,
            )
            assert not trainer.model.training


class TestCheckpointRoundtrip:
    def test_checkpoint_carries_the_complete_logical_state(self, mini, tmp_path):
        root, identity = mini
        trainer = build_trainer(
            root, identity, validation_cadence_updates=2, validation_batches=1
        )
        with trainer:
            trainer.train_updates(2)
            path = tmp_path / "roundtrip.ckpt"
            trainer.save_checkpoint(path)
            from stratego.training.warmstart_checkpoint import load_warmstart_checkpoint

            restored = load_warmstart_checkpoint(
                path,
                expected_train_config=trainer.config.identity(),
                expected_train_config_digest=trainer.config.digest(),
                expected_corpus_identity=identity,
            )
            assert restored["global_step"] == trainer.global_step
            assert restored["examples_consumed"] == trainer.examples_consumed
            assert restored["cursor"] == trainer.cursor
            assert restored["best_validation"] == trainer.best_validation
            assert restored["validation_history"] == trainer.validation_history
            metadata = restored["metadata"]
            # The corpus is identified by digests; the resolved root is
            # diagnostic only and lives outside the identity block.
            assert "content_digest" in metadata["corpus_identity"]
            assert "resolved_corpus_root" not in metadata["corpus_identity"]
            assert "resolved_corpus_root" in metadata["diagnostics"]
            assert trainer.counters["checkpoint_errors"] == 0
