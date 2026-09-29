# Software of unknown provenance

| | |
|---|---|
| Document | SK-SOUP |
| Standard | IEC 62304:2006+AMD1:2015 |
| Safety class | A |
| Product | Scan Kit 2 |

Git history is the revision record.

SOUP is every direct dependency this project does not develop and that is linked into, or required to build, the Scan Kit 2 binaries. Transitive dependencies are pinned by the lockfiles (`Cargo.lock`, `apps/desktop/package-lock.json`) and are not repeated here.

## Rule

A change that adds a direct dependency updates this list in the same change, with the version, the purpose, and where the requirement comes from (Cargo.toml, package.json, or the Rust toolchain file). Removing a direct dependency removes its row in that same change.

Versions below are the ones locked for the phase 1 workspace. Anomaly review for a SOUP upgrade is the pull request that bumps the lockfile: the author records any known defect that affects Scan Kit in the pull request, or states that they found none that affect the used interface.

## Rust workspace

| SOUP | Version | Purpose |
|---|---|---|
| Rust | stable, pinned by `rust-toolchain.toml` | Language and standard library |
| tokio | see `crates/scan-kit-mcp/Cargo.toml` | Async runtime for the MCP server |
| serde / serde_json | see `crates/scan-kit-core/Cargo.toml` | Tool input and output |
| rmcp | see `crates/scan-kit-mcp/Cargo.toml` | MCP protocol and stdio transport |
| tauri | see `apps/desktop/src-tauri/Cargo.toml` | Desktop shell |

## Desktop frontend

| SOUP | Version | Purpose |
|---|---|---|
| React | see `apps/desktop/package.json` | User interface |
| Tailwind CSS | see `apps/desktop/package.json` | shadcn theme utilities |
| Radix UI packages pulled in by shadcn | see `apps/desktop/package.json` | Behavior of shadcn components |

shadcn component source copied into `apps/desktop/src/components/ui/` is project software, not SOUP. The packages those components import are SOUP and are listed through `package.json`.
