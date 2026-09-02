"""Phase 18 Gate G2: the frozen synthetic known-reward setup landscape.

What the learner sees, and what it does not
-------------------------------------------
The landscape assigns every legal canonical 40-piece setup an additive
utility from a fixed piece-type-by-square table and maps that utility to a
seeded win/draw/loss outcome. The setup-learning path receives ONLY the
outcomes: `outcomes_for` is the single method the assay calls on the training
side, and it returns integers in {-1, 0, +1}. The utility, the table and the
exact optimum are read only by the evaluator, which scores the EMA model's
samples after the fact.

```text
U(s)        = sum_k T[t_k, k]                      T: [12, 40], reflection-symmetric
z(s)        = (U(s) - mu_uniform) / sigma_uniform   exact moments under a uniform
                                                     random legal setup
P(draw)     = p_draw                                 fixed
P(win)      = (1 - p_draw) * sigmoid(kappa * z)
P(loss)     = (1 - p_draw) * (1 - sigmoid(kappa * z))
outcome     = one uniform per (period, setup, replicate) seed through
              derive_stream_seed, thresholded loss | draw | win
```

Reflection invariance
---------------------
`T[t, rank, file] == T[t, rank, 9 - file]` by construction (the left five
files are drawn and mirrored), so a setup and its mirror have identical
utility and the learner cannot be rewarded for handedness.

Exact optimum
-------------
Maximising `U` subject to the inventory is a linear assignment of 40 piece
slots to 40 squares. It is solved by the Hungarian algorithm on the expanded
40 x 40 cost matrix, and the result is CERTIFIED independently by linear
programming duality: the returned potentials `(u, v)` satisfy
`u_i + v_j <= cost_ij` for every slot and square, and `sum(u) + sum(v)` equals
the assignment's cost, which proves optimality without trusting the solver.

Uniform baseline
----------------
Under a uniform random legal setup (a uniform random permutation of the 40
piece slots over the 40 squares) the mean of `U` is exact and its variance is
Hoeffding's permutation-statistic formula, also exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...engine.constants import NUM_PIECE_TYPES, PIECE_COUNTS, PIECES_PER_PLAYER
from ...engine.setup import validate_setup
from ...setups.identity import CANONICAL_FILES, CANONICAL_RANKS, class_fingerprint, content_fingerprint
from .setup_contract import Phase18SetupError, json_document_digest, seed_uniform, stream_seed

LANDSCAPE_VERSION = "phase18_g2_synthetic_landscape_v1"

#: Piece slots in inventory order: 40 entries, type of each slot.
SLOT_TYPES = tuple(piece_type for piece_type in range(NUM_PIECE_TYPES) for _ in range(PIECE_COUNTS[piece_type]))
assert len(SLOT_TYPES) == PIECES_PER_PLAYER


@dataclass(frozen=True)
class OutcomeMapping:
    kappa: float
    p_draw: float

    def probabilities(self, z) -> np.ndarray:
        """`[..., 3]` in the published (loss, draw, win) order."""
        z = np.asarray(z, dtype=np.float64)
        win_share = 1.0 / (1.0 + np.exp(-self.kappa * z))
        p_win = (1.0 - self.p_draw) * win_share
        p_loss = (1.0 - self.p_draw) * (1.0 - win_share)
        p_draw = np.full_like(z, self.p_draw)
        return np.stack([p_loss, p_draw, p_win], axis=-1)

    def outcome(self, z: float, uniform: float) -> int:
        p_loss, p_draw, _ = self.probabilities(z)
        if uniform < p_loss:
            return -1
        if uniform < p_loss + p_draw:
            return 0
        return 1

    def document(self) -> dict:
        return {
            "form": "P(win) = (1 - p_draw) * sigmoid(kappa * z), P(loss) = (1 - p_draw) * (1 - sigmoid(kappa * z)), P(draw) = p_draw; z = (U - mu_uniform) / sigma_uniform",
            "kappa": self.kappa,
            "p_draw": self.p_draw,
            "draw_rule": "one uniform u per (period, setup fingerprint, replicate) seed: u < P(loss) -> -1; u < P(loss) + P(draw) -> 0; else +1",
        }


def build_table(seed: int) -> np.ndarray:
    """The reflection-symmetric `[12, 40]` utility table from one seed."""
    generator = np.random.default_rng(int(seed))
    half = generator.standard_normal((NUM_PIECE_TYPES, CANONICAL_RANKS, CANONICAL_FILES // 2))
    half = np.round(half, 6)
    full = np.concatenate([half, half[:, :, ::-1]], axis=2)
    return full.reshape(NUM_PIECE_TYPES, PIECES_PER_PLAYER)


def utility(table: np.ndarray, canonical) -> float:
    entries = tuple(int(v) for v in canonical)
    validate_setup(entries, 0)
    return float(sum(table[piece, square] for square, piece in enumerate(entries)))


def utilities(table: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Vectorised utility for `[N, 40]` legal canonical boards."""
    array = np.asarray(boards, dtype=np.int64)
    squares = np.arange(PIECES_PER_PLAYER)
    return table[array, squares[None, :]].sum(axis=1)


def uniform_moments(table: np.ndarray) -> dict:
    """Exact mean and variance of `U` under a uniform random legal setup.

    `a[i, j] = T[type(slot i), j]`; `U = sum_i a[i, pi(i)]` for a uniform random
    permutation `pi`. Mean `= sum_ij a_ij / n`; variance by Hoeffding (1951):
    `(1 / (n - 1)) * sum_ij (a_ij - a_i. - a_.j + a_..)^2`.
    """
    a = np.stack([table[t] for t in SLOT_TYPES])              # [40 slots, 40 squares]
    n = a.shape[0]
    row = a.mean(axis=1, keepdims=True)
    col = a.mean(axis=0, keepdims=True)
    grand = a.mean()
    d = a - row - col + grand
    mean = float(a.sum() / n)
    variance = float((d ** 2).sum() / (n - 1))
    return {"mean": mean, "variance": variance, "sd": math.sqrt(variance), "formula": "Hoeffding permutation statistic"}


def hungarian_minimum(cost) -> tuple:
    """Kuhn-Munkres with potentials on a square matrix. Returns
    `(assignment, u, v)` with `assignment[i]` the column of row `i` and dual
    potentials satisfying `u_i + v_j <= cost_ij`."""
    matrix = [[float(x) for x in row] for row in cost]
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise Phase18SetupError("hungarian_minimum needs a square matrix")
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment, u[1:], v[1:]


def exact_optimum(table: np.ndarray, *, tolerance: float = 1e-9) -> dict:
    """The maximum utility over legal setups, with a duality certificate."""
    a = np.stack([table[t] for t in SLOT_TYPES])              # [40 slots, 40 squares]
    cost = (-a).tolist()                                       # minimise -U
    assignment, u, v = hungarian_minimum(cost)
    if sorted(assignment) != list(range(PIECES_PER_PLAYER)):
        raise Phase18SetupError("the assignment is not a permutation")
    board = [None] * PIECES_PER_PLAYER
    for slot, square in enumerate(assignment):
        board[square] = SLOT_TYPES[slot]
    canonical = tuple(int(t) for t in board)
    validate_setup(canonical, 0)
    optimum = utility(table, canonical)
    # Certificate: dual feasibility and zero duality gap, checked by arithmetic.
    gap_violations = 0
    worst = 0.0
    for i in range(PIECES_PER_PLAYER):
        for j in range(PIECES_PER_PLAYER):
            slack = cost[i][j] - u[i] - v[j]
            if slack < -tolerance:
                gap_violations += 1
            worst = min(worst, slack)
    dual_value = -(sum(u) + sum(v))
    certified = gap_violations == 0 and abs(dual_value - optimum) <= 1e-6
    return {
        "optimum": optimum,
        "optimal_setup": list(canonical),
        "optimal_setup_content_fingerprint": content_fingerprint(canonical),
        "optimal_setup_class_fingerprint": class_fingerprint(canonical),
        "certificate": {
            "method": "LP duality: u_i + v_j <= cost_ij for all (i, j) and sum(u) + sum(v) == assignment cost",
            "dual_feasibility_violations": gap_violations,
            "worst_slack": worst,
            "dual_value": dual_value,
            "primal_value": optimum,
            "gap": abs(dual_value - optimum),
            "certified": certified,
        },
    }


@dataclass(frozen=True)
class SyntheticLandscape:
    version: str
    table_seed: int
    table: np.ndarray
    mapping: OutcomeMapping
    uniform_mean: float
    uniform_sd: float
    optimum: float
    optimal_setup: tuple
    namespace: str

    # -- evaluation side (utility visible) ---------------------------------

    def utility(self, canonical) -> float:
        return utility(self.table, canonical)

    def utilities(self, boards) -> np.ndarray:
        return utilities(self.table, boards)

    def z_scores(self, boards) -> np.ndarray:
        return (self.utilities(boards) - self.uniform_mean) / self.uniform_sd

    # -- learning side (outcomes only) ---------------------------------------

    def outcome_seed(self, seed_index: int, period: int, fingerprint: str, replicate: int) -> int:
        return stream_seed(self.namespace, "outcome", int(seed_index), int(period), fingerprint, int(replicate))

    def _hidden_z(self, canonical) -> float:
        """The landscape's own standardised utility, used only to draw outcomes.
        Never returned to a caller."""
        return (utility(self.table, canonical) - self.uniform_mean) / self.uniform_sd

    def outcomes_for(self, canonical, *, seed_index: int, period: int, fingerprint: str, replicates: int) -> list:
        """The ONLY landscape method the setup-learning path may call. Returns
        `replicates` independent seeded outcomes in {-1, 0, +1} and nothing
        else: no utility, no probability, no gradient."""
        z = self._hidden_z(canonical)
        return [
            self.mapping.outcome(z, seed_uniform(self.outcome_seed(seed_index, period, fingerprint, r)))
            for r in range(int(replicates))
        ]

    def expected_z_outcome(self, boards) -> np.ndarray:
        """E[outcome] = P(win) - P(loss) for the evaluator."""
        probabilities = self.mapping.probabilities(self.z_scores(boards))
        return probabilities[..., 2] - probabilities[..., 0]

    def document(self) -> dict:
        moments = uniform_moments(self.table)
        optimum = exact_optimum(self.table)
        return {
            "landscape_version": self.version,
            "namespace": self.namespace,
            "table_seed": self.table_seed,
            "table_shape": [NUM_PIECE_TYPES, PIECES_PER_PLAYER],
            "table_rows": "piece type 0..11 (spy, scout, miner, sergeant, lieutenant, captain, major, colonel, general, marshal, flag, bomb)",
            "table_columns": "canonical square 0..39, rank-major (rank 0 = own back rank), file 0..9",
            "table": [[float(x) for x in row] for row in self.table],
            "reflection_invariant": bool(np.array_equal(self.table.reshape(NUM_PIECE_TYPES, CANONICAL_RANKS, CANONICAL_FILES), self.table.reshape(NUM_PIECE_TYPES, CANONICAL_RANKS, CANONICAL_FILES)[:, :, ::-1])),
            "uniform_baseline": moments,
            "exact_optimum": optimum,
            "outcome_mapping": self.mapping.document(),
            "outcome_seed_rule": f"derive_stream_seed('{self.namespace}', 'outcome', seed_index, period, content_fingerprint, replicate) -> seed_uniform",
            "learner_interface": "outcomes_for(canonical, seed_index, period, fingerprint, replicates) -> list of -1/0/+1; utility, table and optimum are evaluator-only",
            "table_digest": json_document_digest([[float(x) for x in row] for row in self.table]),
        }

    def digest(self) -> str:
        return json_document_digest(self.document())


def build_landscape(*, namespace: str, table_seed: int, kappa: float, p_draw: float) -> SyntheticLandscape:
    table = build_table(table_seed)
    moments = uniform_moments(table)
    optimum = exact_optimum(table)
    if not optimum["certificate"]["certified"]:
        raise Phase18SetupError("the exact optimum could not be certified")
    return SyntheticLandscape(
        version=LANDSCAPE_VERSION,
        table_seed=int(table_seed),
        table=table,
        mapping=OutcomeMapping(kappa=float(kappa), p_draw=float(p_draw)),
        uniform_mean=moments["mean"],
        uniform_sd=moments["sd"],
        optimum=optimum["optimum"],
        optimal_setup=tuple(optimum["optimal_setup"]),
        namespace=namespace,
    )


def landscape_from_document(document: dict) -> SyntheticLandscape:
    """Rebuild a frozen landscape from its document and verify its digest."""
    rebuilt = build_landscape(
        namespace=document["namespace"],
        table_seed=int(document["table_seed"]),
        kappa=float(document["outcome_mapping"]["kappa"]),
        p_draw=float(document["outcome_mapping"]["p_draw"]),
    )
    if rebuilt.document()["table_digest"] != document["table_digest"]:
        raise Phase18SetupError("the rebuilt landscape table does not match the frozen document")
    if rebuilt.version != document["landscape_version"]:
        raise Phase18SetupError("landscape version mismatch")
    return rebuilt


__all__ = [
    "LANDSCAPE_VERSION",
    "OutcomeMapping",
    "SLOT_TYPES",
    "SyntheticLandscape",
    "build_landscape",
    "build_table",
    "exact_optimum",
    "hungarian_minimum",
    "landscape_from_document",
    "uniform_moments",
    "utilities",
    "utility",
]
