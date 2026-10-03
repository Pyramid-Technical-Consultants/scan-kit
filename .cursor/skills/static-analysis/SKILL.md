---
name: static-analysis
description: Run Scan Kit's formatting, lint, test, and dependency checks. Use when handing back Rust or desktop work, or when asked to lint, find dead code, or check dependencies.
---

# Static analysis

These are the same checks CI runs. On PowerShell, run each command separately. Do not chain them with `&&`.

From the repository root:

```bash
cargo fmt --all --check
cargo clippy --workspace --exclude scan-kit-desktop --all-targets --locked -- -D warnings
cargo clippy -p scan-kit-plot --no-deps --target wasm32-unknown-unknown -- -D warnings
cargo test --workspace --exclude scan-kit-desktop --locked
cargo deny check
cargo machete --with-metadata
```

Linux CI excludes `scan-kit-desktop` because the Tauri shell needs WebKitGTK. On Windows, when that crate builds, also run:

```bash
cargo clippy -p scan-kit-desktop --all-targets --locked -- -D warnings
```

From `apps/desktop`:

```bash
npm run build
npm test
npm run lint
npm run knip
```

`npm run build` already runs `tsc`. `npm run lint` is Oxlint (correctness, including React hooks). `npm run knip` reports unused files, exports, and dependencies.

`cargo deny` and `cargo machete` are installed tools, not workspace crates. If the commands are missing, install the versions CI uses: `cargo install cargo-deny --locked --version 0.20.2` and `cargo install cargo-machete --locked --version 0.9.2`.

Do not silence a lint with `#[allow]` or an Oxlint disable, and do not skip a failing test to get green. A Knip or cargo-machete false positive gets a one-line ignore in that tool's config that names the reason. Oxlint skips generated `src/wasm` and copied `src/components/ui`. Knip skips gitignored `src/wasm` and ignores the copied shadcn tree.
