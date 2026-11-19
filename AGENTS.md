# Working in this repository

- Keep each checkout/worktree on its own `target/` — never share
  `CARGO_TARGET_DIR`. Cargo does not distinguish path dependencies across
  checkouts, so concurrent builds produce mismatched artifacts.
- Iterate with `cargo test -p <crate>`; run the full workspace checks before
  submitting (see `.cursor/skills/static-analysis/SKILL.md`).
