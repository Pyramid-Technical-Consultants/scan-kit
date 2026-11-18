import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { keepUp } from "./keep-up.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const tauriJs = path.join(dir, "../node_modules/@tauri-apps/cli/tauri.js");
const args = process.argv.slice(2);

// Tauri copies binaries/scan-kit-mcp-<host> into the installer. The file is
// a build output, so `npm run tauri build` produces it and dev does not.
function stageMcpSidecar() {
  const repo = path.resolve(dir, "../../..");
  execFileSync("cargo", ["build", "--release", "--locked", "-p", "scan-kit-mcp"], {
    cwd: repo,
    stdio: "inherit",
  });
  const host = /^host: (.+)$/m
    .exec(execFileSync("rustc", ["-vV"], { encoding: "utf8" }))?.[1]
    ?.trim();
  if (!host) {
    throw new Error("rustc -vV did not report a host triple");
  }
  const ext = process.platform === "win32" ? ".exe" : "";
  const destDir = path.join(dir, "../src-tauri/binaries");
  fs.mkdirSync(destDir, { recursive: true });
  fs.copyFileSync(
    path.join(repo, "target", "release", `scan-kit-mcp${ext}`),
    path.join(destDir, `scan-kit-mcp-${host}${ext}`),
  );
}

if (args[0] === "build") {
  stageMcpSidecar();
}

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
