"""Shared fixtures for the Phase 16 Agent 1 measurement tests.

Everything here is hermetic: the accepted setup library and selector load
from their frozen files (as every phase's tests already do), but no torch
model is loaded and no game is played. Instrument identity, gating and
storage are what these tests pin.
"""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repository_root():
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def setup_sources():
    from stratego.search.phase15.boards import Phase15MatchSetupSources

    return Phase15MatchSetupSources()


@pytest.fixture(scope="session")
def library_families():
    from stratego.evaluation.phase16.adversarial import author_library

    return author_library()


@pytest.fixture(scope="session")
def library_document(library_families):
    from stratego.evaluation.phase16.adversarial import build_library_document

    families = {family: list(entries) for family, entries in library_families.items()}
    return build_library_document(families, generated_utc="2026-08-25T00:00:00Z")
