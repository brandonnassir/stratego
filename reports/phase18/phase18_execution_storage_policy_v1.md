# Phase 18 execution-storage policy (v1)

Adopted 2026-09-02 by the Phase 18 Backend Steward, before Gate G3 work begins.
It governs **where future Phase 18 auxiliary execution data is written**. It
changes no result, no contract and no gate status, and it authorizes no run.

## Locations

```text
main repository       /Users/brandonwashington/Dev/Github/stratego/gpt_agent
future worktrees      output/phase18/worktrees/     (git-ignored)
future runtime data   output/phase18/runtime/       (git-ignored)
tracked evidence      unchanged, in its existing repository locations
```

1. **The main repository remains at
   `/Users/brandonwashington/Dev/Github/stratego/gpt_agent`.** It does not move.

2. **Future detached execution worktrees must be created beneath
   `output/phase18/worktrees/`**, one directory per frozen source commit.

3. **Future untracked runtime artifacts should be written beneath
   `output/phase18/runtime/`** — outcome receipts, telemetry, held-out arrays,
   checkpoint objects and similar working bytes that are reproducible from a
   frozen contract.

4. **Tracked reports, contracts, decisions, manifests and compact evidence remain
   in their existing repository locations** — `reports/phase18/` and its
   subdirectories, `instructions/`, `scripts/`, `stratego_project_docs/`. This
   policy does not relocate any tracked artifact.

5. **Every future launch manifest must record its exact worktree and runtime
   paths**, absolute, as recorded facts of the run, alongside the frozen source
   commit it was created from.

6. **Worktrees must be created from a frozen source commit and verified clean
   before execution.** The frozen commit is recorded first; the worktree is
   created detached at that commit; `git status` in the worktree is confirmed
   clean; only then may a run start.

`.gitignore` carries exactly two narrowly scoped entries for this,
`output/phase18/worktrees/` and `output/phase18/runtime/`. **The `output/`
directory itself is deliberately not ignored.** Other existing ignore rules
still apply inside it: the repository-wide `*.pdf` rule means PDFs under
`output/pdf/` remain ignored unless they are explicitly force-added or a narrow
negation is adopted separately. Non-PDF deliverables remain visible unless
another existing pattern matches them.

## Historical worktrees — inventory

Four Phase 18 execution worktrees exist from Gates G1 and G2. All four are
**registered, detached, and Git-clean under ordinary `git status`**: they have
zero modified, staged, or non-ignored untracked entries at their frozen source
commits. Ignored content is not represented by that statement and must be
inventoried separately before any archival action.

| worktree (in `/Users/brandonwashington/Dev/Github/stratego/`) | frozen commit | size | state |
|---|---|---|---|
| `gpt_agent_phase18_g1_exec` | `66b733ad92324751e30bd7e2a5e373129cbe87c3` | 307 MiB | detached, Git-clean; ignored checkpoints/evaluation material present |
| `gpt_agent_phase18_g1_confirm_exec` | `9392c6ec1c948a7c5c91278616f669340f4a6445` | 169 MiB | detached, Git-clean; ignored cache material present |
| `gpt_agent_phase18_g2_exec` | `354a4cad55a88dca6dcb24a21cf79cecc130008f` | 194 MiB | detached, Git-clean; ignored cache material present |
| `gpt_agent_phase18_g2_raw_exec` | `ccddceda27015f47d26879802b4b55653c8fdf18` | 194 MiB | detached, Git-clean; ignored cache material present |

Total footprint **864 MiB** (884,588 KiB). These four sit beside the main
repository, not beneath `output/phase18/worktrees/`, because they predate this
policy.

The ignored material in `gpt_agent_phase18_g1_exec` includes Phase 18 G1
checkpoint and evaluation outputs that may be unique. The other three worktrees
contain at least ignored cache material. Ordinary clean-status checks therefore
do not authorize deletion or relocation.

## Why they stay where they are

**They remain in place because accepted evidence records their absolute
locations.** Those paths are not incidental: they are recorded facts inside
accepted, digest-pinned artifacts. Across the tracked tree there are dozens of
references to the four absolute worktree paths, including:

- all four launch manifests — `phase18_g1_launch_manifest_v1.json`,
  `phase18_g1_random_confirmation_launch_v1.json`,
  `phase18_g2_launch_manifest_v1.json`,
  `phase18_g2_raw_confirmation_launch_manifest_v1.json`;
- the accepted decision packets `P18-D002`, `P18-D003`, `P18-D004`, `P18-D005`;
- per-seed result records, arm records, the G2 contracts and
  `phase18_process_boundary_v2.json`.

Moving or deleting a worktree would make those recorded paths false while their
digests still verify, so the evidence would state a location that no longer
exists. This policy therefore applies to **future** execution data only, and is
not retroactive.

## No historical worktree may be moved or removed without an archival migration

**No historical worktree may be moved, copied, deleted, pruned or unregistered
except through a separate archival migration** that is proposed, reviewed and
approved on its own. That migration must:

1. **check ignored and untracked files** in each worktree, so nothing that exists
   only there is discovered after the fact;
2. **preserve unique checkpoints** — any checkpoint or artifact that exists
   nowhere else must be retained, not regenerated on the assumption that it is
   reproducible;
3. **record old-to-new path mappings** for every relocated directory, so the
   absolute paths in accepted evidence remain resolvable;
4. **verify hashes** before and after the move, proving the bytes are identical;
   and
5. **receive explicit approval before any removal.** Verification precedes
   deletion; deletion is the last step and is never bundled with the move.

Nothing in this policy step moved, copied, deleted, pruned or unregistered any
worktree. All four remain registered and untouched.
