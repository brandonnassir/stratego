"""Phase 8 Agent 1: frozen corpus identity, seeds, and decision sampling.

Specification sources:

- `01_AGENT_1_WARMSTART_CONTRACT.md` ("Corpus game identity", "Freeze Phase 8
  seeds", "Decision sampler")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 12, 14, 22 (game
  identity, deterministic per-game decision sampling, canonical seeds)

What lives here and why
-----------------------
Everything in this module is *identity*: the frozen Phase 8 seeds, the
synthetic game identifier, the per-game domain-separated stream seeds, and
the deterministic decision sampler. None of it knows what a policy, a setup
or a model is — :mod:`stratego.training.warmstart_contract` layers the
learning-design contract on top of these identities and is the only intended
importer. Keeping identity free of contract knowledge means a seed can never
depend on a measurement, which is the property the whole phase leans on.

Seeds were chosen before any Phase 8 corpus, pilot, or model result existed.
They follow the repository's date-seed precedent (`20260101` for the Phase 4
bank, `20260813` for the Phase 7 library): the Phase 8 block is the freeze
date `20260813` extended with a two-digit role suffix, giving each canonical
seed a distinct, self-describing value that was fixed by the calendar rather
than by anything measured.

Derivation
----------
All streams come from one helper, :func:`derive_warmstart_seed`, a
domain-separated ``blake2b`` hash with the Phase 8 personalization tag
``strat-ws8`` — distinct from every earlier tag (``strat-lb7``/``strat-at7``/
``strat-st7`` for the Phase 7 library, ``strat-bnk``/``strat-sid`` for the
Phase 4 bank, ``strat-pls``/``strat-dec`` for match/decision seeds), so no
Phase 8 stream can collide with any accepted upstream stream. There is no
global RNG cursor anywhere: every consumer receives a pure function of the
game identity and a frozen domain string.

The four per-game domains the common contract names are:

```text
setup_root          the setup-source root seed of one logical game
policy:red          the red rule-policy match-level seed
policy:blue         the blue rule-policy match-level seed
decision_sampler    the per-bin decision-selection streams
```

Per-ply rule-policy randomness stays on the frozen Phase 4 path: a policy
receives its match-level seed from here and each ply's stream is
``derive_decision_seed(policy_seed, ply)`` exactly as accepted in Phase 4.
"""

from __future__ import annotations

import hashlib
import re

#: The Phase 8 synthetic corpus identity. A change to the game-id format, the
#: seed derivation, or the split schedule is a new corpus version, never a
#: silent edit.
SYNTHETIC_CORPUS_VERSION = "synthetic_warmstart_corpus_v1"

#: The deterministic per-game decision sampler named by the common contract.
DECISION_SAMPLER_VERSION = "warmstart_decision_sampler_v1"

#: Maximum decisions selected from one game. A game with more decisions is
#: covered by exactly this many stratified bins.
MAX_DECISIONS_PER_GAME = 64

#: Corpus splits and the frozen number of games per ordered matchup cell.
#: 100 ordered cells x (200 + 40 + 40) = 28,000 games.
CORPUS_SPLITS = ("train", "validation", "test")
GAMES_PER_CELL = {"train": 200, "validation": 40, "test": 40}

# ---------------------------------------------------------------------------
# Canonical Phase 8 seeds — chosen before any result, recorded here first.
# ---------------------------------------------------------------------------

#: Root of every corpus-generation stream (setups, rule policies, decision
#: sampling). Folded into every synthetic game id.
CORPUS_MASTER_SEED = 2026081301

#: The canonical C1 initialization. Agent 6's production run must begin from
#: `build_candidate_model("C1", seed=CANONICAL_C1_INIT_SEED)` — a fresh
#: reconstruction, never a pilot checkpoint — and the same reconstruction is
#: the "canonical untrained C1" opponent of acceptance gate 26.2.
CANONICAL_C1_INIT_SEED = 2026081302

#: Root of the training-order/shuffle streams (one derived stream per epoch).
TRAIN_ORDER_SEED = 2026081303

#: Namespace roots keeping pilot streams and the final production run's
#: streams disjoint from each other and from everything above.
PILOT_NAMESPACE_SEED = 2026081304
FINAL_RUN_NAMESPACE_SEED = 2026081305

#: Frozen game-level bootstrap seeds for held-out statistics.
VALIDATION_BOOTSTRAP_SEED = 2026081306
TEST_BOOTSTRAP_SEED = 2026081307

CANONICAL_SEEDS = {
    "corpus_master_seed": CORPUS_MASTER_SEED,
    "canonical_c1_init_seed": CANONICAL_C1_INIT_SEED,
    "train_order_seed": TRAIN_ORDER_SEED,
    "pilot_namespace_seed": PILOT_NAMESPACE_SEED,
    "final_run_namespace_seed": FINAL_RUN_NAMESPACE_SEED,
    "validation_bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
    "test_bootstrap_seed": TEST_BOOTSTRAP_SEED,
}

# ---------------------------------------------------------------------------
# Stream domains
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 8 warm-start stream.
_WARMSTART_SEED_PERSON = b"strat-ws8"

DOMAIN_SETUP_ROOT = "setup_root"
DOMAIN_RED_POLICY = "policy:red"
DOMAIN_BLUE_POLICY = "policy:blue"
DOMAIN_DECISION_SAMPLER = "decision_sampler"
DOMAIN_TRAIN_ORDER = "train_order"
DOMAIN_PILOT = "pilot"
DOMAIN_FINAL_RUN = "final_run"
DOMAIN_BOOTSTRAP = "bootstrap"

STREAM_DOMAINS = (
    DOMAIN_SETUP_ROOT,
    DOMAIN_RED_POLICY,
    DOMAIN_BLUE_POLICY,
    DOMAIN_DECISION_SAMPLER,
    DOMAIN_TRAIN_ORDER,
    DOMAIN_PILOT,
    DOMAIN_FINAL_RUN,
    DOMAIN_BOOTSTRAP,
)


class WarmstartSeedError(ValueError):
    """Raised when a Phase 8 identity or seed request is malformed."""


def derive_warmstart_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 8 stream.

    ``domain`` must be one of :data:`STREAM_DOMAINS`; ``parts`` are the
    identity inputs of the stream. The payload is the colon-joined text of
    domain and parts under the ``strat-ws8`` personalization, so equal
    identities always agree and any change to any identity input yields an
    unrelated stream.
    """
    if domain not in STREAM_DOMAINS:
        raise WarmstartSeedError(f"unknown warm-start stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise WarmstartSeedError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
    payload = ":".join([SYNTHETIC_CORPUS_VERSION, domain, *[str(part) for part in parts]])
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_WARMSTART_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# Synthetic game identity
# ---------------------------------------------------------------------------

#: `policy_id@policy_version` as produced by `PolicyRef.token`. The pattern is
#: deliberately narrow: identifiers never contain `|`, `=` or `:`, so the
#: pipe-delimited game id below parses unambiguously.
_POLICY_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9_]*@[0-9]+\.[0-9]+\.[0-9]+$")

_GAME_ID_PATTERN = re.compile(
    r"^(?P<corpus>[a-z0-9_]+)\|ms=(?P<master>[0-9]+)\|split=(?P<split>[a-z]+)"
    r"\|red=(?P<red>[^|]+)\|blue=(?P<blue>[^|]+)\|g=(?P<ordinal>[0-9]{4})$"
)


def _require_policy_token(token: str, side: str) -> str:
    if not _POLICY_TOKEN_PATTERN.match(token):
        raise WarmstartSeedError(
            f"{side} policy token {token!r} is not a canonical 'id@version' token"
        )
    return token


def _require_split(split: str) -> str:
    if split not in CORPUS_SPLITS:
        raise WarmstartSeedError(
            f"unknown corpus split {split!r}; expected one of {list(CORPUS_SPLITS)}"
        )
    return split


def synthetic_game_id(
    split: str, red_token: str, blue_token: str, ordinal: int
) -> str:
    """The stable identifier of one logical corpus game.

    A pure function of exactly the identity fields the common contract
    requires — corpus version, split, red policy id/version, blue policy
    id/version, per-cell game ordinal, and the frozen corpus master seed —
    in a fixed, parseable ``key=value`` pipe format:

    ```text
    synthetic_warmstart_corpus_v1|ms=2026081301|split=train|
        red=strategic_rule_based@1.1.0|blue=random_legal@1.0.0|g=0137
    ```

    Worker count, process partitioning, arrival order and resume boundaries
    appear nowhere, which is what makes the identity schedule-independent.
    """
    _require_split(split)
    _require_policy_token(red_token, "red")
    _require_policy_token(blue_token, "blue")
    games = GAMES_PER_CELL[split]
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise WarmstartSeedError(f"game ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal < games:
        raise WarmstartSeedError(
            f"game ordinal {ordinal} is outside 0..{games - 1} for split {split!r}"
        )
    return (
        f"{SYNTHETIC_CORPUS_VERSION}|ms={CORPUS_MASTER_SEED}|split={split}"
        f"|red={red_token}|blue={blue_token}|g={ordinal:04d}"
    )


def parse_synthetic_game_id(game_id: str) -> dict:
    """The identity fields of a synthetic game id, validated.

    Raises on anything that is not exactly a well-formed id of this corpus
    version under the frozen master seed, so a foreign or tampered identifier
    can never be mistaken for a corpus game.
    """
    match = _GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise WarmstartSeedError(f"malformed synthetic game id: {game_id!r}")
    fields = match.groupdict()
    if fields["corpus"] != SYNTHETIC_CORPUS_VERSION:
        raise WarmstartSeedError(
            f"game id names corpus {fields['corpus']!r}, expected "
            f"{SYNTHETIC_CORPUS_VERSION!r}"
        )
    if int(fields["master"]) != CORPUS_MASTER_SEED:
        raise WarmstartSeedError(
            f"game id names master seed {fields['master']}, expected {CORPUS_MASTER_SEED}"
        )
    split = _require_split(fields["split"])
    red = _require_policy_token(fields["red"], "red")
    blue = _require_policy_token(fields["blue"], "blue")
    ordinal = int(fields["ordinal"])
    if ordinal >= GAMES_PER_CELL[split]:
        raise WarmstartSeedError(
            f"game id ordinal {ordinal} is outside the {split!r} schedule"
        )
    return {
        "corpus_version": fields["corpus"],
        "corpus_master_seed": int(fields["master"]),
        "split": split,
        "red_token": red,
        "blue_token": blue,
        "ordinal": ordinal,
    }


# ---------------------------------------------------------------------------
# Per-game domain-separated stream seeds
# ---------------------------------------------------------------------------


def setup_root_seed(game_id: str) -> int:
    """The setup-source root seed of one logical game.

    Passed as ``root_seed`` to the frozen `setup_source_v1` ``assign`` call
    with the frozen constants ``environment_id=0, generation=0``; the
    sampler's own accepted side derivation then gives red and blue their
    independent domain-separated draw streams.
    """
    parse_synthetic_game_id(game_id)
    return derive_warmstart_seed(DOMAIN_SETUP_ROOT, game_id)


def red_policy_seed(game_id: str) -> int:
    """The match-level seed of the red rule policy in one game."""
    parse_synthetic_game_id(game_id)
    return derive_warmstart_seed(DOMAIN_RED_POLICY, game_id)


def blue_policy_seed(game_id: str) -> int:
    """The match-level seed of the blue rule policy in one game."""
    parse_synthetic_game_id(game_id)
    return derive_warmstart_seed(DOMAIN_BLUE_POLICY, game_id)


def game_seeds(game_id: str) -> dict:
    """Every per-game stream seed, keyed by its contract name."""
    return {
        "setup_root_seed": setup_root_seed(game_id),
        "red_policy_seed": red_policy_seed(game_id),
        "blue_policy_seed": blue_policy_seed(game_id),
    }


# ---------------------------------------------------------------------------
# Deterministic per-game decision sampling
# ---------------------------------------------------------------------------


def decision_bin_bounds(total_decisions: int) -> tuple:
    """The 64 contiguous near-equal bins over decision indices ``[0, T)``.

    Bin ``b`` covers ``[floor(b*T/64), floor((b+1)*T/64))`` in integer
    arithmetic. For ``T >= 64`` every bin is non-empty, the bins are disjoint,
    and their union is exactly ``[0, T)`` — the three properties Agent 3's
    independent reimplementation must reproduce. Only defined for ``T`` above
    the selection cap; a shorter game selects every decision and has no bins.
    """
    total = int(total_decisions)
    if total <= MAX_DECISIONS_PER_GAME:
        raise WarmstartSeedError(
            f"binning is undefined for {total} <= {MAX_DECISIONS_PER_GAME} decisions; "
            "short games select every decision"
        )
    bins = MAX_DECISIONS_PER_GAME
    return tuple(
        ((index * total) // bins, ((index + 1) * total) // bins) for index in range(bins)
    )


def decision_bin_seed(game_id: str, bin_index: int) -> int:
    """The domain-separated stream of one stratified bin of one game."""
    parse_synthetic_game_id(game_id)
    if not 0 <= int(bin_index) < MAX_DECISIONS_PER_GAME:
        raise WarmstartSeedError(
            f"bin index {bin_index} is outside 0..{MAX_DECISIONS_PER_GAME - 1}"
        )
    return derive_warmstart_seed(DOMAIN_DECISION_SAMPLER, game_id, int(bin_index))


def selected_decision_indices(game_id: str, total_decisions: int) -> tuple:
    """`warmstart_decision_sampler_v1`: the selected decisions of one game.

    ```text
    T <= 0      ()                    (a zero-decision game trains nothing)
    T <= 64     (0, 1, ..., T-1)      (every decision)
    T >  64     one index per bin:    lo + (decision_bin_seed(game_id, b)
                                            % (hi - lo))
    ```

    Bins are disjoint and ascending, so selection is without replacement and
    the result is strictly increasing (already sorted) by construction. The
    per-bin modulo draw is exact determinism, not a statistical claim; with
    63-bit streams and bin widths below 2**12 the residual modulo bias is
    far below anything measurable. Game outcome, teacher strength, future
    value and model predictions appear nowhere in the selection.
    """
    total = int(total_decisions)
    if total < 0:
        raise WarmstartSeedError(f"total_decisions must be >= 0, got {total}")
    if total == 0:
        return ()
    if total <= MAX_DECISIONS_PER_GAME:
        return tuple(range(total))
    selected = []
    for bin_index, (low, high) in enumerate(decision_bin_bounds(total)):
        width = high - low
        draw = decision_bin_seed(game_id, bin_index) % width
        selected.append(low + draw)
    return tuple(selected)


# ---------------------------------------------------------------------------
# Training-order, pilot, final-run and bootstrap streams
# ---------------------------------------------------------------------------


def train_order_seed(epoch: int) -> int:
    """The shuffle seed of one training epoch over the selected-example universe."""
    if int(epoch) < 0:
        raise WarmstartSeedError(f"epoch must be >= 0, got {epoch}")
    return derive_warmstart_seed(DOMAIN_TRAIN_ORDER, TRAIN_ORDER_SEED, int(epoch))


def pilot_stream_seed(candidate_id: str, purpose: str) -> int:
    """A named stream inside one pilot candidate's namespace."""
    if not str(candidate_id).strip() or not str(purpose).strip():
        raise WarmstartSeedError("pilot streams need a candidate_id and a purpose")
    return derive_warmstart_seed(
        DOMAIN_PILOT, PILOT_NAMESPACE_SEED, str(candidate_id), str(purpose)
    )


def final_run_stream_seed(purpose: str) -> int:
    """A named stream inside the canonical production run's namespace."""
    if not str(purpose).strip():
        raise WarmstartSeedError("final-run streams need a purpose")
    return derive_warmstart_seed(DOMAIN_FINAL_RUN, FINAL_RUN_NAMESPACE_SEED, str(purpose))


def bootstrap_seed(split: str) -> int:
    """The frozen game-level bootstrap seed of one held-out split."""
    if split == "validation":
        return VALIDATION_BOOTSTRAP_SEED
    if split == "test":
        return TEST_BOOTSTRAP_SEED
    raise WarmstartSeedError(
        f"bootstrap seeds exist for the held-out splits only, not {split!r}"
    )


__all__ = [
    "CANONICAL_C1_INIT_SEED",
    "CANONICAL_SEEDS",
    "CORPUS_MASTER_SEED",
    "CORPUS_SPLITS",
    "DECISION_SAMPLER_VERSION",
    "DOMAIN_BLUE_POLICY",
    "DOMAIN_BOOTSTRAP",
    "DOMAIN_DECISION_SAMPLER",
    "DOMAIN_FINAL_RUN",
    "DOMAIN_PILOT",
    "DOMAIN_RED_POLICY",
    "DOMAIN_SETUP_ROOT",
    "DOMAIN_TRAIN_ORDER",
    "FINAL_RUN_NAMESPACE_SEED",
    "GAMES_PER_CELL",
    "MAX_DECISIONS_PER_GAME",
    "PILOT_NAMESPACE_SEED",
    "STREAM_DOMAINS",
    "SYNTHETIC_CORPUS_VERSION",
    "TEST_BOOTSTRAP_SEED",
    "TRAIN_ORDER_SEED",
    "VALIDATION_BOOTSTRAP_SEED",
    "WarmstartSeedError",
    "blue_policy_seed",
    "bootstrap_seed",
    "decision_bin_bounds",
    "decision_bin_seed",
    "derive_warmstart_seed",
    "final_run_stream_seed",
    "game_seeds",
    "parse_synthetic_game_id",
    "pilot_stream_seed",
    "red_policy_seed",
    "selected_decision_indices",
    "setup_root_seed",
    "synthetic_game_id",
    "train_order_seed",
]
