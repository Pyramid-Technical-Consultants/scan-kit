# Agent notes

## Working alongside other agents

- When another agent may be editing the same checkout, do work in a `git
  worktree` and let it use its own `target/` (the default). Never point
  `CARGO_TARGET_DIR` at another checkout's `target/`: cargo does not
  distinguish same-name path dependencies across worktrees, so concurrent
  builds overwrite each other's rlibs and produce phantom compile errors.
- A fresh worktree pays one cold dependency build. After that,
  `cargo test -p <crate>` is incremental and fast; reserve
  `cargo test --workspace` and clippy for handback per
  `.cursor/skills/static-analysis/SKILL.md`.
