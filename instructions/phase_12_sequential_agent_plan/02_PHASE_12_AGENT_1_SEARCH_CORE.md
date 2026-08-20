# Phase 12 — Agent 1
## Minimal Search Engine and Belief Providers

## Mission

Build the smallest correct search implementation capable of answering whether belief-guided search can improve C1's move selection.

Do not perform a major strength experiment yet.

## 1. Preserve existing evidence

Do not modify accepted Phase 9, Phase 10, Phase 11, or Phase 11B artifacts.

All new work belongs under Phase 12-specific paths.

## 2. Model roles

Use:

```text
accepted Phase 9 C1:
    policy
    value

Agent 1C:
    belief only
```

Do not use Agent 1C's policy or value outputs as the production move model.

## 3. Common belief-provider interface

Implement:

```text
remaining_count
original_phase11
agent1c
oracle
```

For neural providers, obtain marginals and pass them through existing constrained-world machinery by adapter/import. Do not modify accepted Phase 11 sampler mathematics.

The `oracle` provider may use true hidden state only for offline diagnostics.

The production configuration must structurally reject or omit `oracle`.

## 4. Minimal search algorithm

For each move:

1. obtain current public state;
2. obtain Phase 9 root policy/value outputs;
3. sample `K` complete hidden worlds once;
4. choose candidate root actions;
5. evaluate every candidate on the same worlds;
6. apply the candidate action in each world;
7. rollout using accepted Phase 9 greedy policy for both sides;
8. stop at configured depth or terminal;
9. use exact terminal result when terminal;
10. otherwise use Phase 9 WDL value at leaf;
11. convert WDL to scalar root-player value;
12. average values across worlds;
13. apply modest direct-policy regularization;
14. choose highest-scoring legal root action.

Use:

```text
V = P(win) - P(loss)
```

## 5. Information boundary

A sampled world is one hypothetical hidden state.

During rollout, each simulated player receives only legally available information. The root player must not directly receive sampled opponent ranks.

The oracle is an explicitly offline diagnostic exception.

## 6. Candidate root actions

```text
if legal_actions <= 8:
    candidates = all legal actions
else:
    candidates = top 8 legal actions under Phase 9 policy
```

The direct Phase 9 action must always be present.

## 7. Policy regularization

Implement:

```text
S(a) = Q(a) + beta * log(pi(a) + epsilon)
```

Choose one modest fixed `beta`. No grid search.

## 8. Presets

```text
TINY
worlds 8
candidates <= 8
depth 4

SMALL
worlds 16
candidates <= 8
depth 6

MEDIUM
worlds 32
candidates <= 8
depth 8
```

Do not run a broad MEDIUM benchmark yet.

## 9. Performance implementation

Reuse existing batched inference/coordinator infrastructure where practical.

Prefer batching Phase 9 C1 inference across worlds at each rollout ply.

Correctness comes before optimization.

Do not delay the prototype for advanced caching or trunk sharing.

## 10. Sanity work only

Use a small set of hand-picked and fresh positions.

Establish:

```text
all four belief providers execute
all selected actions are legal
all sampled worlds are legal
direct C1 action always appears among root candidates
fixed seeds reproduce the same search action
oracle is unavailable in production configuration
TINY latency is measured
SMALL latency is measured
```

No large match run.

## 11. Report

Report:

```text
search architecture
search score definition
belief-provider interface
TINY latency
SMALL latency
C1 forwards per move
batch sizes
positions/second
main observed bottleneck
```

## 12. Deliverables

Create:

```text
Phase 12 search module(s)
Phase 12 belief-provider adapter(s)
TINY/SMALL/MEDIUM configs
small correctness tests
reports/phase12/agent_01_report.md
reports/phase12/agent_01_summary.json
```

## 13. Stop condition

Stop after the search core is working and sanity checks pass.

Do not launch a large match run.

Do not begin Agent 2 automatically.
