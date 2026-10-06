import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { keepUp } from "./keep-up.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const tauriJs = path.join(dir, "../node_modules/@tauri-apps/cli/tauri.js");
const args = process.argv.slice(2);

if (args[0] === "dev") {
  await keepUp(process.execPath, [tauriJs, ...args], { name: "Tauri dev" });
} else {
  const child = spawn(process.execPath, [tauriJs, ...args], { stdio: "inherit" });
  child.on("error", () => process.exit(1));
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}
