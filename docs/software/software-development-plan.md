# Software development plan

| | |
|---|---|
| Document | SK-SDP |
| Standard | IEC 62304:2006+AMD1:2015 |
| Safety class | A |
| Product | Scan Kit |

Git history is the revision record for this document and for the rest of the software configuration. This plan does not keep a separate revision table.

## Scope

This plan covers Scan Kit 2, the Rust and Tauri replacement of the Python application, developed in this repository. Phase 1 establishes the workspace, the desktop shell, the MCP server, and these planning records. Later changes port behavior into Rust and delete the Python that only that behavior needed.

`scan_kit/__init__.py` remains the version of the released Python product until `develop` is promoted to `main` as 2.0. The Rust workspace version is `2.0.0-dev` until that promotion. `main` keeps the last complete 1.x line. After the first port change deletes Python, a `develop` build is no longer a full 1.x application.

## Safety class

Scan Kit is IEC 62304 Class A. The rationale and the intended-use assumptions are in [safety-classification.md](safety-classification.md). Risk management for software is that classification record plus the problem-resolution process below. A change that contradicts the intended use reopens the classification before the related behavior is ported.

## Development process

Work lands on `develop` through pull requests. `main` receives only a promotion of `develop` for a release.

A change that adds or alters behavior:

1. States or updates a software requirement in [requirements.md](requirements.md) with an id `SK-REQ-###`.
2. Implements the behavior once, in `scan-kit-core`, and registers it in the tool catalog. The desktop shell and the MCP server call that function. They do not grow a second implementation.
3. Verifies it with a unit test named or commented with the requirement id, and records the link in [traceability.md](traceability.md).
4. Deletes the Python modules and tests that exist only for the ported behavior, in the same change. A Python module that unported code still imports stays until its last caller is gone. The Python tree is the unported backlog. There is no second status list and no PyO3 bridge.

The engineering rules for crates, tools, and the user interface are in [architecture.md](architecture.md). That guide is not an IEC 62304 architectural design deliverable. Class A does not require one.

## What Class A requires, and what this project also does

IEC 62304 activities that apply to Class A, and how this repository meets them:

| Activity | How it is met |
|---|---|
| 5.1 Software development planning | This plan |
| 5.2 Software requirements analysis | [requirements.md](requirements.md), updated as behavior is added |
| 5.5 Software unit implementation | Rust functions in the workspace crates |
| 5.7 Software system testing | The desktop and MCP binaries build in CI. System tests grow when ported behavior has a user-visible flow |
| 5.8 Software release | Tag-driven GitHub Release. Python-only until the last Python application code is removed, then the Rust artifacts |
| 6 Software maintenance | This plan remains in force after the first 2.0 release. Fixes follow the same change path |
| 7 Software risk management | [safety-classification.md](safety-classification.md) |
| 8 Software configuration management | Git, pull requests, and CI, as below |
| 9 Software problem resolution | GitHub issues, as below |

Class A does not require a software architectural design (5.3), a software detailed design (5.4), a formal software unit verification process (5.5.2 through 5.5.5), or software integration testing (5.6). This project still runs unit tests, `cargo fmt`, and `cargo clippy -D warnings` on every pull request, because those checks are how the port stays reviewable. They are extra controls, not a claim that the software is Class B or Class C.

## Configuration management

The configuration items are the files in this git repository, including requirements, this plan, source, tests, lockfiles, and the SOUP list.

- One branch of record for day-to-day work: `develop`.
- Identity of a change is the git commit. Identity of a release is the `v*` tag.
- A pull request is the review and approval of a change before it merges.
- CI on the pull request must pass before merge.
- The repository holds one implementation of a capability. Porting retires the Python implementation in the same pull request.
- Direct dependencies are recorded in [soup.md](soup.md) in the same change that adds them. Lockfiles pin the versions CI builds.

Build outputs (`target/`, `node_modules/`, `dist/`, Python `dist/`) are not configuration items.

## Verification

Pull requests to `develop` or `main` run:

- `cargo fmt --all --check`
- `cargo clippy --workspace --exclude scan-kit-desktop --all-targets --locked -- -D warnings` (the Tauri shell needs WebKitGTK, which the Linux job does not install)
- `cargo clippy -p scan-kit-plot --no-deps --target wasm32-unknown-unknown -- -D warnings`
- `cargo test --workspace --exclude scan-kit-desktop --locked`
- `cargo deny check` (advisories, licenses, and crate sources)
- `cargo machete --with-metadata` (unused direct Cargo dependencies)
- in `apps/desktop`: `npm run build` (wasm plot, `tsc`, and the Vite bundle), `npm test` (Vitest), `npm run lint` (Oxlint), and `npm run knip`

`cargo deny` and `cargo machete` are development tools installed in the Rust job. They are not linked into the binaries. The command list an agent runs is the same one, in [.cursor/skills/static-analysis/SKILL.md](../../.cursor/skills/static-analysis/SKILL.md).

Pushes to `develop` or `main` also upload preview desktop and MCP binaries. Those artifacts are not a GitHub Release.

Python tests keep running against whatever `scan_kit/` and `tests/` remain. A port change deletes the tests that only covered the removed modules, and the remaining Python tests must pass.

## Release

Until the Python application is gone, a `v*` tag publishes the Python executables only, and the tag must match `__version__` in `scan_kit/__init__.py`. The Rust workspace version is not that release.

The change that deletes the last of the Python application also removes the Python release job and points tag-driven releases at the Rust desktop artifacts. That change updates this plan and the version source of truth in the same pull request.

Known residual anomalies at release are the open GitHub issues labeled for that release.

## Problem resolution

Problems are GitHub issues in this repository. An issue that reports a software defect states what happened, what was expected, and the version (Python `__version__` or the Rust workspace version). The fixing change links the issue. Closing the issue is the record that it was resolved. Pull requests are not a second problem log.

## Maintenance

After 2.0 is released, fixes, environmental changes, and SOUP updates use the same pull-request path, requirements, tests, and SOUP list. Behavior that is still Python is unported, not a maintained second product.

## SOUP

Software of unknown provenance is every direct dependency this project does not develop. The list, the purpose of each item, and the rule for adding one are in [soup.md](soup.md).

## Tools used to develop the software

Rust (stable, pinned by `rust-toolchain.toml`), Cargo, the Tauri CLI, Node.js, GitHub Actions, cargo-deny, cargo-machete, Oxlint, and Knip. These tools are development tools. They are not part of the released medical-device software except where a dependency is linked into the binaries and listed as SOUP.
