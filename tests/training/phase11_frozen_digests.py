"""The Phase 11 Agent 1 freeze, pinned.

These digests were computed once, at the Agent 1 freeze, before any Phase 11
prediction was scored, any world was sampled and any test outcome existed.
They live in their own module so the values that *are* the freeze sit apart
from the tests that check them, and so a later agent can import the freeze
rather than recompute it.

A failing pin is not something to update. It means a frozen decision moved,
which is a new contract version after review.
"""

CONTRACT_DIGESTS = {
    "phase11_belief_contract_v1":
        "13c4607619cca5fde621980b6ffa155d3c01378f000d5869319a069286186f75",
    "phase11_belief_baseline_v1":
        "c017d51f78e8f7f5976abec62aa259fd8918508810c42bc41e88470c5998c197",
    "phase11_belief_bank_v1":
        "874a2513427aebde69c9d31c9d06c6562d3daf2faa3f0dc4cd916656a175516d",
    "phase11_belief_metrics_v1":
        "a2f7e5b4cc3944194d3b735f96ad805413670648ffd46acaac3a6a7f436368cb",
    "phase11_belief_sampler_v1":
        "a113d2e9588a6c4d7c2dcff954773e693ae876d19465904e4b277e86675afca9",
    "phase11_information_safety_v1":
        "1b8160d544b5ee71eb1b03be025a868e7298ace1d61c486524e631ba68faab4d",
    "phase11_acceptance_v1":
        "0121ecaac6849a59d78798833ec419f9ff12c14f8720f1bef259960f42c01fe5",
    "phase11_system_v1":
        "9aa22d45ab85b65d5ed14e40288ef7cd4c3226e8d66f52508d9717929ac1adfe",
}

CONTRACT_BUNDLE_DIGEST = (
    "ad16f921c602c1e1eb4975bee31fa6d1dff8dd4afdd09c332d9deaa94712192d"
)

#: The belief head inside the accepted Phase 9 checkpoint, derived from live
#: tensor bytes (`belief_output.bias`, `belief_output.weight`) under the
#: accepted state-digest recipe.
BELIEF_HEAD_DIGEST = (
    "a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7"
)

BANK_DIGESTS = {
    "validation": "bba6860549c05ebd59487d83d205e9d18b2109ab143d3816afbe793a13a04023",
    "test": "566ac35214ac04d5928af2f2975308a03bb78eb2a19e2ea05e6367f839eff404",
}

BANK_MANIFEST_DIGESTS = {
    "validation": "d83ab48516e03a74695a04d68dcda6f17fbf02cb468b6785a3d91627b0534173",
    "test": "360a687d5a6ed2623d50a88cb1fe392dee85064f15f84fc61f13752b6ddca3b0",
}
