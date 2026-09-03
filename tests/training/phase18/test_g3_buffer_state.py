"""Stage 6B: the setup buffer's exact state capture for the joint bundle,
without touching the accepted buffer module."""

import numpy as np
import pytest
import torch

from stratego.training.phase18.g3_buffer_state import (
    buffer_state_digest,
    capture_buffer_state,
    restore_buffer_state,
    sample_digest,
    sample_document,
    sample_from_document,
)
from stratego.training.phase18.g3_contract import Phase18G3Error


def test_capture_and_restore_are_exact_and_the_restored_buffer_processes_identically(filled_buffer):
    state = capture_buffer_state(filled_buffer)
    restored = restore_buffer_state(state)
    assert buffer_state_digest(restored) == buffer_state_digest(filled_buffer)
    assert len(restored) == len(filled_buffer) and restored.ready_count() == filled_buffer.ready_count()
    assert restored.telemetry() == filled_buffer.telemetry()
    a = filled_buffer.process(alpha=0.1)
    b = restored.process(alpha=0.1)
    assert np.array_equal(a.indices, b.indices) and np.array_equal(a.advantage, b.advantage)
    batch_a = next(filled_buffer.minibatches(16, seed=3))
    batch_b = next(restored.minibatches(16, seed=3))
    assert batch_a.fingerprints == batch_b.fingerprints
    assert torch.equal(batch_a.behavior_log_probs, batch_b.behavior_log_probs)
    for sample in filled_buffer.samples:
        document = sample_document(sample)
        again = sample_from_document(document)
        assert sample_digest(document) == sample_digest(sample_document(again))
        assert again.content_fingerprint == sample.content_fingerprint


def test_the_restored_buffer_keeps_attributing_and_filtering(filled_buffer):
    restored = restore_buffer_state(capture_buffer_state(filled_buffer))
    fingerprint = filled_buffer.samples[0].content_fingerprint
    before = restored.outcome_record(fingerprint)["count"]
    restored.add_outcome(fingerprint, 1)
    assert restored.outcome_record(fingerprint)["count"] == before + 1
    assert restored.filter(5)["rows"] == 0 and restored.need_pool


def test_a_foreign_buffer_version_is_refused(filled_buffer):
    state = capture_buffer_state(filled_buffer)
    with pytest.raises(Phase18G3Error, match="version"):
        restore_buffer_state(dict(state, buffer_version="other"))
