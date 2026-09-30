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

Versions below are the ones locked for the workspace. Anomaly review for a SOUP upgrade is the pull request that bumps the lockfile: the author records any known defect that affects Scan Kit in the pull request, or states that they found none that affect the used interface.

## Rust workspace

| SOUP | Version | Purpose |
|---|---|---|
| Rust | stable, pinned by `rust-toolchain.toml` | Language and standard library |
| tokio | see `Cargo.toml` | Async runtime for the MCP server, plot-frame readback, and the compute test |
| serde / serde_json | see `Cargo.toml` | Tool input and output |
| rmcp | see `crates/scan-kit-mcp/Cargo.toml` | MCP protocol and stdio transport |
| tauri | see `apps/desktop/src-tauri/Cargo.toml` | Desktop shell |
| tauri-plugin-dialog | see `apps/desktop/src-tauri/Cargo.toml` | Open Data Folder dialog |
| rusqlite (bundled SQLite) | see `crates/scan-kit-io/Cargo.toml` | Existing `~/.scan-kit/scan-kit.sqlite` file |
| zip | see `crates/scan-kit-io/Cargo.toml` | `.zip` session archives, read without unpacking |
| tar | see `crates/scan-kit-io/Cargo.toml` | `.tar` session archives, read without unpacking |
| flate2 | see `crates/scan-kit-io/Cargo.toml` | `.tar.gz` and `.tgz` session archives |
| bzip2 | see `crates/scan-kit-io/Cargo.toml` | `.tar.bz2` session archives |
| xz2 | see `crates/scan-kit-io/Cargo.toml` | `.tar.xz` session archives |
| csv | see `crates/scan-kit-io/Cargo.toml` | Quoted fields and header drift in session files |
| wgpu | see `crates/scan-kit-compute/Cargo.toml` | The only GPU and compute library |
| naga | see `crates/scan-kit-compute/Cargo.toml` | Compile the compute shader when no GPU adapter is present |
| fontdue | see `crates/scan-kit-compute/Cargo.toml` | Rasterize plot labels into one glyph atlas |

Plot text uses the vendored Source Sans 3 regular face in `crates/scan-kit-compute/assets/SourceSans3-Regular.ttf` (SIL Open Font License, Latin 400 from fontsource 5.2.8). It is compiled into the compute crate with `include_bytes`. It is not a Cargo dependency.

## Desktop frontend

| SOUP | Version | Purpose |
|---|---|---|
| React | see `apps/desktop/package.json` | User interface |
| Tailwind CSS | see `apps/desktop/package.json` | shadcn theme utilities |
| Base UI (`@base-ui/react`) | see `apps/desktop/package.json` | Behavior of shadcn components |
| `@glideapps/glide-data-grid` | see `apps/desktop/package.json` | Session list and later column views |
| `@tauri-apps/plugin-dialog` | see `apps/desktop/package.json` | Open Data Folder dialog |
| sonner | see `apps/desktop/package.json` | Toast notices that do not move the layout |
| next-themes | see `apps/desktop/package.json` | Theme value read by the stock toast toaster. The app stays dark |
| vitest | see `apps/desktop/package.json` | Desktop UI checks. Not linked into the product binary |
| happy-dom | see `apps/desktop/package.json` | DOM for those checks. Not linked into the product binary |

shadcn component source copied into `apps/desktop/src/components/ui/` is project software, not SOUP. The packages those components import are SOUP and are listed through `package.json`.
