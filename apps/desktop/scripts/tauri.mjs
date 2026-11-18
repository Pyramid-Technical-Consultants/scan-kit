import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { keepUp } from "./keep-up.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const tauriJs = path.join(dir, "../node_modules/@tauri-apps/cli/tauri.js");
const args = process.argv.slice(2);

// Tauri's build script copies binaries/scan-kit-mcp-<host> next to the app for
// both `tauri dev` and `tauri build`. The file is a build output. Dev keeps a
// debug binary once it exists; a bundle always restages the release binary.
function stageMcpSidecar(release) {
  const repo = path.resolve(dir, "../../..");
  const host = /^host: (.+)$/m
    .exec(execFileSync("rustc", ["-vV"], { encoding: "utf8" }))?.[1]
    ?.trim();
  if (!host) {
    throw new Error("rustc -vV did not report a host triple");
  }
  const ext = process.platform === "win32" ? ".exe" : "";
  const destDir = path.join(dir, "../src-tauri/binaries");
  const dest = path.join(destDir, `scan-kit-mcp-${host}${ext}`);
  if (!release && fs.existsSync(dest)) {
    return;
  }
  execFileSync(
    "cargo",
    ["build", "--locked", "-p", "scan-kit-mcp", ...(release ? ["--release"] : [])],
    { cwd: repo, stdio: "inherit" },
  );
  fs.mkdirSync(destDir, { recursive: true });
  fs.copyFileSync(
    path.join(repo, "target", release ? "release" : "debug", `scan-kit-mcp${ext}`),
    dest,
  );
}

if (args[0] === "build") {
  stageMcpSidecar(true);
}

if (args[0] === "dev") {
  stageMcpSidecar(false);
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
