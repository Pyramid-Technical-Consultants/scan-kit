import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { keepUp } from "./keep-up.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const desktop = path.resolve(dir, "..");

const wasm = spawn("npm", ["run", "wasm"], {
  cwd: desktop,
  stdio: "inherit",
  shell: true,
});
const wasmCode = await new Promise((resolve) => {
  wasm.on("error", () => resolve(1));
  wasm.on("exit", (code) => resolve(code ?? 1));
});
if (wasmCode !== 0) {
  process.exit(wasmCode);
}

const vite = path.join(desktop, "node_modules/vite/bin/vite.js");
await keepUp(process.execPath, [vite], { name: "Vite", restartOnClean: true });
