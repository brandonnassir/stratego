# Phase 12 Agent 5 — Working Search-Enhanced Player

Generated 2026-08-20T18:17:30Z by `scripts/run_phase12_agent05.py`.

Engineering integration of the accepted TINY + Agent 1C configuration into the project's one working player. No new experiment: quick checks, a 16-board smoke set already inside Agent 4's pack, and the frozen `phase12_search_candidate_v1` artifact.

## 1. The production stack

```text
accepted Phase 9 C1   policy + value
Agent 1C              belief only
search                phase12_root_world_search_v1 @ TINY

phase9 source sha256     dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
phase9 state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
agent1c sha256           a125208605f5e68c897214016e1803718439755e6286e5a185447636ffcd9fad
agent1c state digest     69104cd98c66ae1715b93990cc949b50ebad47bf66177b0d38eb6c958db7c2b8
candidate config digest  b33c486af457af18a2ecf2dba7b6338310a3b53e5a6d1c8b3562be7ed6c7f25f
```

Digest-checked at load against the Phase 11B handoff record; the loader (`stratego.search.phase12.player.load_search_player`) refuses unbound bytes.

## 2. Modes, time cap, fallback

```text
modes            direct | tiny (default) | small | medium (max-strength)
time caps        tiny 0.5 s   small 1.5 s   medium 3.5 s
cap rationale    TINY observed 0.126 s median / 0.138 s p95 / 0.193 s max;
                 0.5 s = 3.6x p95, 2.6x max — headroom, not the p95 itself
fallback         direct accepted Phase 9 C1
fallback fires   timeout | search_error | unexpected_error |
                 non_finite_score | illegal_action | direct_error (last resort)
```

The cap is enforced cooperatively inside the engine (`deadline` parameter — additive, default off, spot-checked bit-identical on the accepted weights) and re-checked on completion. Every fallback is counted by reason and logged; the player never forfeits and never emits an illegal action because search failed. SMALL remains an engineering/debug mode; the oracle is not a mode and cannot be constructed into the player (four independent refusals, checked below).

**Maximum-strength candidate (by project direction): `medium` (MEDIUM — 32 worlds, depth 8, cap 3.5 s).** Agent 4 measured it at EWR 0.6875 (0.846 s/move median, 0.916 s/move p95), a 0.0469 EWR lead over TINY that sits inside the 0.10 engineering margin — the strongest observed configuration, not a validated ordering. TINY remains the production default.

## 3. Integration surfaces

```text
machine vs machine   Phase12PlayerSeat(player, 'direct' | 'tiny' | ...)
                     through the accepted matchplay driver (play_arm_game)
human play           scripts/play_phase12.py --red human --blue tiny
status/logs          player.status(), logger 'stratego.phase12.player'
```

Search seats draw per-ply world seeds from the accepted match stream, which is what makes the smoke games below a replay of Agent 4 rather than merely similar games. The CLI renders only the human's legal observation (unrevealed opponent pieces stay hidden), shows the active mode, budget and per-move latency, and accepts direct/tiny/small/medium for either seat.

## 4. Quick integration checks

| check | result | detail |
|---|---|---|
| search_player_loads | PASS | phase12_search_player_v1, default mode tiny (TINY: 8 worlds, depth 4, cap 0.5s) |
| phase9_checkpoint_identity | PASS | model_state_digest f1df694d59e34359… == handoff |
| agent1c_checkpoint_identity | PASS | sha256 a125208605f5e68c… == handoff, state digest bound |
| oracle_cannot_enter_production | PASS | 4 independent refusals |
| direct_mode_works | PASS | legal, deterministic, 0.0059s |
| normal_tiny_search_works | PASS | legal, 8 unique worlds, 201 forwards, 0.094s (cap 0.5s) |
| search_decision_deterministic | PASS | same seed, same action, same forward count |
| deadline_checks_are_behavior_neutral | PASS | deadline off/roomy: bit-identical decision on the accepted weights |
| timeout_fallback_works | PASS | 0.1 ms cap trips; direct accepted action played and counted |
| forced_error_fallback_works | PASS | search_error, non_finite_score, illegal_action all fall back legally |
| machine_interface_selects_search_mode | PASS | machine seats select direct/tiny; set_mode('small') visible in status |
| human_interface_selects_search_mode | PASS | seats ('human', 'direct', 'tiny', 'small', 'medium'); defaults human vs tiny; oracle not a choice |

## 5. Smoke set: 16 boards, direct and tiny

Ordinal-0 boards of the accepted match pack (4 per opponent, balanced sources and colours) — all 16 inside Agent 4's 64-board set, so the working player's games can be required to replay Agent 4's move for move. The `medium` maximum-strength candidate replays a 2-board contested spot check of the same kind (its strength number is Agent 4's, not these 2 games).

| mode | W / D / L | EWR | decisions | fallbacks | median s/move | p95 | max |
|---|---|---|---|---|---|---|---|
| player_direct | 8 / 1 / 7 | 0.5312 | 1901 | 0 | 0.002 | 0.002 | 0.003 |
| player_search_tiny | 10 / 0 / 6 | 0.6250 | 1174 | 0 | 0.123 | 0.137 | 0.181 |
| player_search_medium | 2 / 0 / 0 | 1.0000 | 206 | 0 | 0.813 | 0.865 | 0.911 |

- `player_direct` vs Agent 4 `direct_c1`: 16/16 boards identical on outcome, plies, player_decisions, c1_forwards, move_changes — replayed exactly.
- `player_search_tiny` vs Agent 4 `search_agent1c_tiny`: 16/16 boards identical on outcome, plies, player_decisions, c1_forwards, move_changes — replayed exactly.
- `player_search_medium` vs Agent 4 `search_agent1c_medium`: 2/2 boards identical on outcome, plies, player_decisions, c1_forwards, move_changes — replayed exactly.
- Boundary probes: player_direct 8 permutation + 0 direct-agreement checks, 0 failures; player_search_tiny 8 permutation + 8 direct-agreement checks, 0 failures; player_search_medium 8 permutation + 8 direct-agreement checks, 0 failures.
- Fallbacks during smoke games: 0 (cap headroom held; the timeout/error fallbacks were exercised in the checks above, by force).

## 6. The frozen engineering candidate

```text
artifact                        phase12_search_candidate_v1
move_model                      accepted Phase 9 C1
belief_model                    Agent1C
search_version                  phase12_root_world_search_v1
selected_preset                 TINY
worlds                          8
root_candidates                 <= 8
depth                           4
beta                            0.1
epsilon                         1e-06
expected_latency_median         0.126 s/move
expected_latency_p95            0.138 s/move
Agent4_quick_EWR                0.6406
Agent4_direct_EWR               0.5234
time_cap_seconds                0.5
fallback_policy                 direct accepted Phase 9 C1
oracle_available_in_production  False
phase11_final_classification    FAIL
phase11b_selection              Agent1C
scientific_validation_status    not performed

maximum_strength_candidate:
  mode                          medium
  preset                        MEDIUM
  worlds                        32
  depth                         8
  time_cap_seconds              3.5
  expected_latency_median       0.846 s/move
  expected_latency_p95          0.916 s/move
  Agent4_quick_EWR              0.6875
  ewr_lead_over_selected        0.0469
```

Written to `checkpoints/phase12/phase12_search_candidate_v1.json` with the full identity blocks (paths, sha256, state digests, dev metrics) and the exact un-rounded Agent 4 numbers the headline strings derive from.

## 7. Known limitations

- Strength numbers are the Agent 4 engineering sample (64 games per rung): no significance claim, and scientific validation has not been performed.
- The Agent 4 ladder did not separate TINY/SMALL/MEDIUM within the 0.10 engineering margin; TINY is the cheapest rung not meaningfully behind the strongest on that pack, not a proven optimum.
- Latency, and therefore the 0.5 s cap's headroom, were measured on this machine (cpu, single process); a different device should re-derive the cap from its own profile, keeping the ~3.6x-over-p95 intent.
- The accepted setup library places a flag on a front row on 47 of 64 match boards, so part of every pack is decided by opening scout races that no search budget can influence.
- Agent 1C was trained on setups from the same accepted library family the match packs draw from; a mild optimistic residual is accepted for engineering purposes.
- The time cap makes the fallback wall-clock-dependent by design; search decisions themselves are seed-deterministic.

## 8. Deliverables and status

```text
stratego/search/phase12/player.py            (new: the working player)
stratego/search/phase12/engine.py            (additive deadline parameter, default off)
stratego/search/phase12/contract.py          (Phase12SearchTimeout)
scripts/play_phase12.py                      (new: human/machine CLI)
scripts/run_phase12_agent05.py
tests/search/test_phase12_player.py
checkpoints/phase12/phase12_search_candidate_v1.json
reports/phase12/agent_05_smoke_games.jsonl
reports/phase12/agent_05_smoke_games.csv
reports/phase12/agent_05_report.md
reports/phase12/agent_05_summary.json

phase11_final_classification          FAIL
phase11b_selection                    Agent1C
scientific_validation_status          not performed
oracle_available_in_production        False
phase11_test_bank_used                False
search_core_modified                  additive only: cooperative deadline parameter, default off, bit-identity spot-checked and Agent 4 games replayed
selected_operating_point              TINY
production_default_mode               tiny
maximum_strength_candidate            MEDIUM (mode 'medium')
time_cap_seconds                      0.5
working_player_delivered              True
quick_checks_all_passed               True
agent4_replay_exact                   True
scientific_validation_started         False
final_training_started                False
```

Stop condition reached: the working search player and `phase12_search_candidate_v1` are delivered. No further validation phase and no final training were started.
