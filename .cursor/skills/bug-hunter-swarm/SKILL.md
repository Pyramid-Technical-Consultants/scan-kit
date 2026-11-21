---
name: bug-hunter-swarm
description: Sweep the codebase for latent bugs with rotating fresh-context hunter subagents, verify every candidate independently, and land the confirmed ones as batched fix PRs — one per round. Use when the user asks to hunt bugs, sweep for defects, run the bug swarm, or do a bug-hunting round.
---

# Bug hunter swarm

Each round runs a small swarm of read-only hunter subagents in parallel, each with a different lens and scope, then an orchestrator (you) verifies candidates, fixes the confirmed bugs with regression tests, and lands them as one PR. Rounds repeat with rotated lenses until the swarm goes quiet.

## Round shape

1. **Hunt** — launch 4-6 `subagent_explore` hunters **in parallel** (`is_background=true`). Each gets a scope (crate/files) and one lens. Give agents raw instructions and file paths, never prior findings — fresh context avoids anchoring.
2. **Verify** — for every reported candidate, confirm it yourself against current `develop` before fixing: read the code path, and where possible write a failing test or a concrete reproduction. Reject anything that is a style nit, a guess, or intentional behavior. A "bug" you cannot reproduce or explain exactly is not confirmed.
3. **Fix + regress** — implement the fix with the smallest correct diff (ponytail rules). Every confirmed bug gets a test that fails without the fix. Bug fixes belong to the crate that owns the behavior.
4. **PR** — exactly **one PR per round**, targeting `develop`. All confirmed fixes go on a single `fix/bughunt-round-N` branch as a few focused commits (one per defect cluster). Do not split the round into per-scope or per-crate PRs — the extra CI runs and review cycles cost more than they isolate. Aim for ≤ ~10 fixes per round so the diff stays reviewable; carry the rest to the next round's log.
5. **Review + merge** — run a fresh-context `subagent_explore` review of the fix PR (same pattern as other PRs), address findings, wait for the required **Test** check, merge, delete the branch.
6. **Rotate** — next round uses different lenses and/or different scopes so the same ground is not re-trod. Keep notes in `ai-scratch/bug-hunt/` (git-ignored): confirmed bugs, rejected candidates, and why — the log dedupes future rounds and may be passed to hunters as a do-not-report list.

## Hunter brief template

Each hunter gets one message with:

- Repo path `C:\Users\MattNichols\Projects\scan-kit`, note it is read-only — no edits.
- Its scope: a crate (`crates/scan-kit-io`), a module, or a file list — keep scopes disjoint so hunters do not collide.
- Its lens (from the rotation below) with 2-3 concrete examples of what to look for.
- Output contract: numbered findings, each with `file:line`, the exact defect, why it is wrong (not just suspicious), a suggested minimal fix, and a reproduction sketch (input or sequence that triggers it). Order by confidence. Explicitly tell it: report only defects it can explain end-to-end; "looks weird" is not a finding.

## Lens rotation

Pick 4-6 per round; do not repeat a lens+scope pair across consecutive rounds.

- **Boundary & numeric** — off-by-one, index/slice bounds, `as` casts that wrap or saturate, `usize`/`u32` overflow, float `NaN`/inf handling, empty input.
- **Error & cancellation** — swallowed errors, `?` dropped silently, state left inconsistent after an error, cancel paths that leak tasks/threads/mapped buffers.
- **State & lifecycle** — stale caches (`kept_runs`, fingerprints), generation/task mismatches, values reused after their inputs changed.
- **Concurrency & GPU queueing** — shared-queue ordering assumptions, map/unmap and staging-buffer lifecycle, atomics/ordering, submission-index bookkeeping.
- **Caller-contract** — invariants callers must hold that are unchecked, panics reachable from untrusted input (MCP args, DICOM, CSV, log files, remote paths).
- **Parsing & IO** — lenient parsers that mis-assign columns, truncation that silently corrupts, path handling on Windows/Unix, partial reads/writes.
- **Resource & cleanup** — unbounded growth, files/sockets/threads left open, temp dirs, guards dropped early.
- **Test-effectiveness** — assertions that cannot fail, tests that do not exercise the invariant they name, fixtures that drifted from the real data shape.

## Scope guidance

- Good first scopes: `crates/scan-kit-io` (parsers, remote, sqlite, DICOM-adjacent), `crates/scan-kit-compute/src/task.rs` (task/lifecycle), `crates/scan-kit-compute/src/present.rs` (MCP arg handling), `apps/desktop/src` (frontend state).
- Physics is in-scope for correctness bugs only — never "improve" `mc_transport.wgsl` physics. A transport-affecting fix runs `cargo run -p scan-kit-compute --bin mc-validate -- fast` (and `patient` for voxel paths) before the PR merges.

## Fix quality bar

- Reproduce first. If a candidate cannot be turned into a failing test or a concrete trace, drop it for this round.
- Root cause at the shared function, not a per-caller patch.
- One bug = one focused change. Do not refactor adjacent code in a bug PR.
- No `#[allow]`, no skipped tests to get green.
- Full static-analysis list before the PR (`cargo fmt/clippy/test`, `deny`, `machete`; `npm run build/test/lint/knip` when `apps/desktop` changed).

## When the swarm goes quiet

Two consecutive rounds with zero confirmed bugs means the codebase is hunted out at these lenses — stop, report the tally (confirmed / rejected / fixed), and keep the `ai-scratch/bug-hunt/` log for the next sweep.
