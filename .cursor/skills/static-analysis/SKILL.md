---
name: static-analysis
description: Run Scan Kit's formatting, lint, test, and dependency checks. While editing, check the crate you changed. Run the full list once before a pull request, or when asked to lint, find dead code, or check dependencies.
---

# Static analysis

While editing, check the crate you changed. Run the full list once before the pull request. That list is what CI runs. On PowerShell, run each command separately. Do not chain them with `&&`.

## While editing

`cargo fmt --all` from the repository root, then clippy and test that crate:

```bash
cargo clippy -p scan-kit-io --all-targets --locked -- -D warnings
cargo test -p scan-kit-io --locked
```

Use the package name. When `Cargo.lock` is unchanged, `--offline` skips the registry.

When the change touches a public type, function, or trait, also check the crates that depend on it:

- `scan-kit-core`: dicom, plot, io, compute, mcp
- `scan-kit-dicom`: io, compute, mcp
- `scan-kit-io`: compute, mcp
- `scan-kit-plot`: compute
- `scan-kit-compute`: mcp

```bash
cargo check -p scan-kit-compute -p scan-kit-mcp --locked
```

Add the wasm32 plot clippy when `scan-kit-plot` or `scan-kit-core` changed. Add the desktop clippy and the `apps/desktop` npm commands when `apps/desktop` changed. Add `cargo deny check` and `cargo machete --with-metadata` when `Cargo.toml`, `Cargo.lock`, or `deny.toml` changed.

## Before the pull request

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

`npm run build` already runs `tsc`. `npm run lint` is Oxlint (correctness, including React hooks). `npm run knip` reports unused files, exports, and dependencies. `npm test` is `vitest run` followed by `node --test scripts/keep-up.test.mjs`. A path passed to `npm test` is forwarded to that second command and fails. Run `npm test` with no extra arguments, or run `npx vitest run <file>` on its own.

`cargo deny` and `cargo machete` are installed tools, not workspace crates. If the commands are missing, install the versions CI uses: `cargo install cargo-deny --locked --version 0.20.2` and `cargo install cargo-machete --locked --version 0.9.2`.

The required GitHub check is the job named Test. CodeQL is not required. Poll the Test job. `gh pr checks --watch` stays pending while CodeQL is still running.

Do not silence a lint with `#[allow]` or an Oxlint disable, and do not skip a failing test to get green. A Knip or cargo-machete false positive gets a one-line ignore in that tool's config that names the reason. Oxlint skips generated `src/wasm` and copied `src/components/ui`. Knip skips gitignored `src/wasm` and ignores the copied shadcn tree.
