# Phase 11B — Agent 5: Scaled Ataraxos-Like Autoregressive Belief Transformer

## Mission

Build one compact Ataraxos-like joint belief model and determine whether explicitly modeling correlations across the entire hidden army offers a large enough gain to justify its additional complexity.

This is the most expensive Phase 11B candidate.

Run it only if the user authorizes it after reviewing Agents 1–4.

Do not use the spent Phase 11 test bank.

# Target Size

Target approximately:

```text
3–5 million parameters
```

A sensible starting region is:

```text
d_model       ~224
encoder       4 blocks
decoder       2 blocks
attention     4 heads
FF width      ~896
dropout       ~0.2
```

Explicitly calculate the final parameter count before training.

Small dimensional adjustment is allowed solely to place the model sensibly within approximately 3–5M parameters.

Do not perform an architecture sweep.

# Encoder

Consume only legally public information.

A practical starting representation is:

```text
10 x 10 public board
    ->
127-channel cell features
    ->
projected board-cell tokens
    +
positional information
    ->
Transformer encoder
```

No hidden truth may enter the encoder.

# Decoder

Use a fixed, public, deterministic order for unresolved opponent pieces.

A suitable default is ascending current board-square index.

At decoder step `j`, predict:

```text
P(h_j | public information, h_1, ..., h_(j-1))
```

During training use teacher forcing with the true earlier hidden ranks as labels/context.

True ranks are supervised labels only.

# Hard Public Legality

At every decoder step enforce legally known constraints.

At minimum:

- if `remaining_count(rank) == 0`, force that rank probability to zero;
- if an unresolved piece has publicly moved, Flag and Bomb must be masked;
- respect every other frozen public legal-rank exclusion available from the public state;
- decrement remaining rank counts as decoder assignments are made;
- renormalize across legal ranks.

The Transformer should not waste capacity relearning arithmetic/game constraints already known exactly.

# Training Objective

Use supervised autoregressive hidden-army negative log-likelihood:

```text
L =
  - sum_j log P(
        true_rank_j
        | public_information,
          true_rank_1 ... true_rank_(j-1)
    )
```

Do not optimize from game outcome.

# Comparable Engineering R_CE

Do not compare teacher-forced decoder loss directly with Agents 1–4.

For each common development position, estimate inference-time marginal probabilities using:

```text
64 ancestral decoder trajectories
```

For each piece:

1. sample prefixes from the model itself;
2. at each decoder step retain the complete conditional rank-probability vector;
3. average those conditional vectors across the 64 sampled prefixes.

This produces an approximate inference-time marginal:

```text
P(h_i | public information)
```

without conditioning on the true previous hidden ranks.

Compute CE, R_CE, and top-1 from these averaged inference-time marginals.

Also report teacher-forced NLL separately as a training diagnostic.

Teacher-forced NLL must not be used to rank the Transformer against the CNNs.

# Time Budget

Maximum:

```text
approximately 2 hours total
```

This is a pilot, not a convergence run.

Run a very short throughput/sanity pilot first.

Stop after approximately 20–30 minutes if the learning curve is obviously poor.

If the model is clearly strong after 45–60 minutes, preserve its best checkpoint immediately.

If it is still improving when the two-hour budget is reached, stop anyway, preserve the best checkpoint, and report that longer training may be justified later.

# Evaluation

Use exactly the common Phase 11B development positions.

Report:

```text
inference-time marginal CE
remaining-count baseline CE
inference-time marginal R_CE
top-1

Phase9-like R_CE
Strategic R_CE
Tactical R_CE
Scout-rush R_CE

teacher-forced NLL (diagnostic only)

parameter count
training wall-clock
time-to-best checkpoint
inference latency
peak memory
```

Also report the cost of producing the 64-trajectory marginal estimate so the search-integration implication is explicit.

# Required Interface

Expose:

```text
predict_marginals(public_state)
sample_worlds(public_state, n, seed)
```

`sample_worlds` should use the Transformer's native constrained autoregressive decoder.

# Final Leaderboard

Without rerunning any earlier model, load Agents 1–4 standardized summary JSON files and produce:

```text
reports/phase11b/leaderboard.md
reports/phase11b/leaderboard.json
```

Compare:

```text
old Phase 11 head
Agent 1 best attached head
Agent 2 raw CNN
Agent 3 C1-feature CNN
Agent 4 hybrid
Agent 5 autoregressive Transformer
```

Do not automatically declare the minimum-CE model the winner.

Highlight:

- overall predictive improvement;
- Scout-rush/generalization;
- inference cost;
- training cost;
- parameter count;
- memory;
- implementation complexity;
- suitability for search.

# Required Artifacts

Save:

```text
best Transformer checkpoint
exact architecture/config
parameter-count calculation
learning curve

reports/phase11b/agent_05_summary.json
reports/phase11b/agent_05_report.md

reports/phase11b/leaderboard.md
reports/phase11b/leaderboard.json
```

# Stop Condition

Stop after reporting.

Do not begin Phase 12 automatically.

Do not claim that Phase 11 has been repaired or scientifically overturned.
