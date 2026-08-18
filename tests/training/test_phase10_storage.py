"""Regression: Phase 10 storage resolves like Phase 8/9 and never invents a path.

The common contract's rule is blunt: a pointer naming an absent external
volume is `BLOCKED`, never a silent fallback to an internal directory. These
tests pin that behaviour and the identity rule that makes a relocated corpus
the same corpus.
"""

from pathlib import Path

from stratego.training import phase10_storage as st


class TestResolution:
    def test_resolution_order_is_the_accepted_precedent(self):
        policy = st.storage_policy_document()
        assert policy["resolution_order"] == [
            "STRATEGO_PHASE10_CORPUS_ROOT",
            "data/phase10_corpus_root.txt",
            "data/phase10/corpus",
        ]

    def test_environment_override_wins(self, monkeypatch):
        monkeypatch.setenv(st.PHASE10_CORPUS_ROOT_ENV, "/tmp/phase10-somewhere")
        assert st.default_corpus_root() == Path("/tmp/phase10-somewhere")
        assert st.describe_corpus_root()["source"] == "environment"

    def test_the_repository_default_is_used_when_nothing_redirects(self, monkeypatch):
        monkeypatch.delenv(st.PHASE10_CORPUS_ROOT_ENV, raising=False)
        description = st.describe_corpus_root()
        if not description["pointer_value"]:
            assert description["source"] == "repository_default"
            assert description["root"].endswith(st.DEFAULT_PHASE10_CORPUS_ROOT)


class TestMountSafety:
    def test_an_absent_external_volume_is_blocked(self, monkeypatch):
        monkeypatch.setenv(
            st.PHASE10_CORPUS_ROOT_ENV, "/Volumes/NotMountedPhase10/corpus"
        )
        report = st.check_corpus_root()
        assert not report["usable"]
        assert report["blocked"]
        assert "BLOCKED" in report["blocked"][0]
        assert report["external_volume"] == "/Volumes/NotMountedPhase10"
        assert report["external_volume_mounted"] is False

    def test_a_local_root_is_usable(self, monkeypatch):
        monkeypatch.setenv(st.PHASE10_CORPUS_ROOT_ENV, str(Path.cwd() / "data"))
        report = st.check_corpus_root()
        assert report["usable"]
        assert report["external_volume"] is None


class TestIdentityRule:
    def test_a_path_is_never_an_identity(self):
        policy = st.storage_policy_document()
        assert policy["logical_identity_is_path_independent"] is True
        assert "never an identity" in policy["identity_rule"]
        assert "BLOCKED" in policy["absent_external_volume_rule"]

    def test_the_schedule_module_imports_nothing_that_reads_a_path(self):
        """The structural proof of the identity rule, checked on the imports.

        `phase10_schedule` is the logical schedule; if it cannot even import
        the filesystem, no derivation of it can depend on one.
        """
        import ast
        import inspect

        from stratego.training import phase10_schedule

        tree = ast.parse(inspect.getsource(phase10_schedule))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        assert not {"os", "pathlib", "Path"} & imported
        assert not any("phase10_storage" in name for name in imported)
