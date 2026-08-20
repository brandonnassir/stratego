# Phase 12 — Agent 4
## Search Budget Scaling with Agent 1C

## Mission

Assuming Agent 3 shows that search is at least promising, determine how much search is worth paying for.

From this point forward use Agent 1C as the production engineering belief provider.

Other belief systems remain comparison references only.

## 1. Prerequisite

Only run this agent if Agent 3 found evidence that search is useful or plausibly useful.

If Agent 3 concluded the search algorithm itself is broken or consistently weaker than direct C1, stop and redesign search instead.

## 2. Sequential presets

Do not run a full grid.

### TINY

```text
worlds 8
candidates <= 8
depth 4
```

### SMALL

```text
worlds 16
candidates <= 8
depth 6
```

### MEDIUM

```text
worlds 32
candidates <= 8
depth 8
```

Only test a larger setting such as 64 worlds and depth 10–12 if MEDIUM produces meaningful additional strength at acceptable latency.

## 3. Evaluation pack

Use one compact fixed engineering match pack.

Do not create a large tournament.

Keep opponent mix stable across presets.

Use enough games to establish the engineering trend, not publication-quality confidence.

## 4. Metrics

For every preset report:

```text
W / D / L
EWR
delta EWR from previous preset
median move latency
p95 move latency
C1 forwards per move
worlds per move
games per hour
average game wall-clock
```

Also calculate a simple engineering-efficiency quantity such as:

```text
EWR improvement / additional search second
```

## 5. Stopping rule

Stop scaling when:

- strength clearly stops improving;
- latency rises much faster than strength;
- human-play latency becomes impractical;
- the useful operating point is already obvious;
- larger search creates instability;
- the next preset would consume disproportionate compute.

Do not spend compute merely because a larger preset exists.

## 6. Performance notes

Profile before redesigning.

High-value optimizations may include batching across worlds, reusing root C1 outputs, avoiding repeated public-state reconstruction, and caching one root belief/world set during one search.

Do not implement major structural optimization unless necessary to make the selected practical preset viable.

## 7. Deliverables

Create:

```text
budget-scaling results
selected practical preset
reports/phase12/agent_04_report.md
reports/phase12/agent_04_summary.json
```

Record:

```text
worlds
root candidates
depth
policy regularization
expected move latency
quick strength result
```

## 8. Stop condition

Stop once a practical operating point has been identified.

Do not begin production integration automatically.
