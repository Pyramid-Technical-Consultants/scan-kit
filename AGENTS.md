# Working in this repository

## Parallel checkouts and worktrees

Multiple checkouts of this workspace — for example, `git worktree`s used for
parallel work — must not share a Cargo target directory. Cargo's build
fingerprints do not account for the checkout location of path dependencies,
so concurrent builds in a shared `target/` overwrite each other's artifacts
and produce confusing compile errors. Leave `CARGO_TARGET_DIR` unset so each
checkout builds into its own `target/`.

A fresh checkout pays one full dependency build; subsequent builds are
incremental. During development, prefer `cargo test -p <crate>` for the
crate you are changing, and run the full workspace checks before submitting —
see `.cursor/skills/static-analysis/SKILL.md`.
