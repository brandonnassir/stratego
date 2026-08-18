"""The Phase 10 Agent 1 freeze, pinned.

These digests were computed once, at the Agent 1 freeze, before any Phase 10
outcome game was played and before either utility model was fit. They live in
their own module so the values that *are* the freeze sit apart from the tests
that check them, and so a later agent can import the freeze rather than
recompute it.

A failing pin is not something to update. It means a frozen decision moved,
which is a new contract version after review.
"""

CONTRACT_DIGESTS = {
    "phase10_setup_contract_v1":
        "94a1d17161fc936b8f11ed10289fe3fd4aed7bab484dac927d5baa035cc935ad",
    "phase10_setup_outcome_corpus_v1":
        "951025f102dab1a103d02f21e5df414265bc594b37bd02283a64fe02585fe6d5",
    "phase10_setup_utility_v1":
        "2778ddea8bb1c85b998a3abaefaf794816bc9b6eb476010b44d040087758f456",
    "phase10_setup_selector_v1":
        "8a3459fbfb88a45f207fe0965dd6c743524ef16168a78b8ab748ff4efd2bd0b2",
    "phase10_selector_schedule_v1":
        "30ad8ede3fe342d071a5a5d7dc65510bf6cdea3ff20c70554d3e181d97b86dc4",
    "phase10_eval_bank_v1":
        "8e4158426e783f55590086164e9e5fccbd331373b04e1e36a9b7358aaf87f22b",
    "phase10_acceptance_v1":
        "a76f79b7a710f327d2ee097aa922203f1e19ec7bb7619d5baac559e73af7e88b",
    "phase10_system_v1":
        "a8b44e1a12bcc31ed446d031c188129dc82584ed64086601ed9b9edb7830a48d",
}

CONTRACT_BUNDLE_DIGEST = (
    "1cfa5b4667bb75bfb9b323f450ec23d5f812dba629e80a9bce0b19dabb02b395"
)

TRAIT_SCALER_DIGEST = (
    "fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9"
)

OUTCOME_SCHEDULE_DIGEST = (
    "1a49f05032e300a8ecef81aa09776ed0d0766149576afb8eaa74a97e974e98b0"
)

PHASE9_ISOLATION_SET_DIGEST = (
    "c714c6e721e65d2624b34b27a529fa95f69369d0f1070d31b134d1b69aac16ce"
)
PHASE9_ISOLATION_SET_SIZE = 1184

BANK_DIGESTS = {
    "validation": "a37ff113d03a0f67e760e447a462cc0d0d8de83f063d395715aeb77be355657f",
    "test": "be04b891ba5ab142aacbd937ab24f79054843310e8d28f6b8cbee65daef931ad",
}

BANK_MANIFEST_DIGESTS = {
    "validation": "459cef36d7032beb8fc9665efa7692dac3c40c68109e9f0bcdefa6141bd0906e",
    "test": "c6f21bcdb829fe77b208e49d9960b05a1b65bcf1dc7944d3f10420bea132a755",
}

