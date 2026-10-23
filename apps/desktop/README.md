# Scan Kit desktop

The Tauri 2 desktop shell for Scan Kit. React + TypeScript in `src/`, the
Rust side in `src-tauri/`. Plot drawing lives in `scan-kit-plot` (wasm) — do
not draw plots in TypeScript.

```bash
npm ci
npm run dev      # vite dev server for `npm run tauri dev`
npm test         # vitest + keep-up node tests
npm run build    # typecheck + wasm + production bundle
```

See the repo-root `README.md` for the product overview.
