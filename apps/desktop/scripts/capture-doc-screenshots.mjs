#!/usr/bin/env node
/**
 * Capture documentation screenshots of the Tauri desktop app.
 *
 * Launches a built binary with WebView2 CDP remote debugging enabled, then
 * drives it through the `window.__scanKitShots` automation surface in App.tsx /
 * AnalysisView.tsx. No npm dependencies: Node's built-in WebSocket and fetch
 * speak CDP directly.
 *
 * Usage:
 *   node scripts/capture-doc-screenshots.mjs [options]
 *
 * Options:
 *   --data <dir>   Session data location       (default: ../../test_data/logs)
 *   --exe <path>   App binary                  (default: release, then debug, in target/)
 *   --out <dir>    Output folder               (default: ../../docs/images)
 *   --only <text>  Only shots whose name contains <text> (repeatable, or comma list)
 *   --probe <view> Open a view, print its control ids/options as JSON, and exit
 *   --port <n>     CDP port                    (default: 9333)
 *   --keep-open    Leave the app running when done
 *   --list         Print the shot table and exit
 *
 * The app must have been built first (`cargo build -p scan-kit-desktop` or a
 * release build). The window stays visible while shots are taken.
 */

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");

// --- Shot table -------------------------------------------------------------
// Session ids come from the local test_data folder. `controls` maps control id
// -> option id; run with --only and `window.__scanKitShots.view.controls()` in
// DevTools (or the log below) to discover ids when a view's options change.

const G3_A = "1943968267";
const G3_B = "1091134775";
const LAUNCHER_SESSIONS = ["656350661", G3_A];

const LAUNCHER_SIZE = [1400, 760];
const VIEW_SIZE = [1400, 860];

const G2 = "590658542";

const SHOTS = [
  { name: "launcher-data-analysis", size: LAUNCHER_SIZE, tab: "Data Analysis", sessions: LAUNCHER_SESSIONS },
  { name: "launcher-plan-synthesis", size: LAUNCHER_SIZE, tab: "Plan Synthesis" },
  { name: "launcher-plan-runner", size: LAUNCHER_SIZE, tab: "Plan Runner" },
  { name: "launcher-config-tuning", size: LAUNCHER_SIZE, tab: "Configuration Tuning" },
  {
    name: "view-distribution-explorer",
    size: VIEW_SIZE,
    view: "Distribution",
    sessions: [G3_A, G3_B],
    controls: { xy: "position", plan: "On" },
  },
  {
    name: "view-sigma-energy",
    size: VIEW_SIZE,
    view: "Bins",
    sessions: [G3_A],
    controls: { y: "sigma:iso", x: "energy" },
  },
  {
    name: "view-dose-ratios-energy",
    size: VIEW_SIZE,
    view: "Bins",
    sessions: [G3_A, G3_B],
    controls: { y: "dose_ratio", x: "energy", corr: "On" },
  },
  {
    name: "view-ic-timeslice-replay",
    size: VIEW_SIZE,
    view: "Timeline",
    sessions: [G3_A],
    controls: { y: "ic_current" },
  },
  {
    name: "view-magnetic-field-replay",
    size: VIEW_SIZE,
    view: "Timeline",
    sessions: [G3_B],
    controls: { y: "probe_field" },
  },
  {
    name: "view-dose-volume",
    size: VIEW_SIZE,
    view: "Volumetric",
    sessions: ["1093436476"],
  },
  {
    name: "view-session-log-compare",
    size: VIEW_SIZE,
    view: "Session Log Compare",
    sessions: ["1264446129", "1264491515"],
  },
  {
    name: "view-amplifier-correlation",
    size: VIEW_SIZE,
    view: "Distribution",
    sessions: [G2],
    controls: { xy: "amplifier", draw: "scatter" },
  },
];

// --- Args -------------------------------------------------------------------

function parseArgs(argv) {
  const args = { data: join(ROOT, "test_data", "logs"), out: join(ROOT, "docs", "images"), port: 9333, only: [], exe: null, keepOpen: false, list: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--keep-open") args.keepOpen = true;
    else if (arg === "--list") args.list = true;
    else if (arg === "--probe") args.probe = argv[++i];
    else if (arg === "--only") args.only.push(...argv[++i].split(","));
    else if (arg === "--data") args.data = resolve(argv[++i]);
    else if (arg === "--out") args.out = resolve(argv[++i]);
    else if (arg === "--exe") args.exe = resolve(argv[++i]);
    else if (arg === "--port") args.port = Number(argv[++i]);
    else if (arg === "--help" || arg === "-h") {
      console.log(readHeader());
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function readHeader() {
  return "node scripts/capture-doc-screenshots.mjs [--data dir] [--out dir] [--exe path] [--only text] [--probe view] [--port n] [--keep-open] [--list]";
}

// --- CDP client -------------------------------------------------------------

async function findPageTarget(port, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = targets.find((t) => t.type === "page" && /localhost|tauri/i.test(t.url));
      if (page) return page;
    } catch {
      // WebView2 debugging endpoint not up yet.
    }
    await sleep(300);
  }
  throw new Error("Timed out waiting for the app's webview on the CDP port");
}

function connectCdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let seq = 0;
  const pending = new Map();
  const ready = new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = (e) => rej(new Error(`CDP websocket failed: ${e.message ?? "unknown"}`));
  });
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    const slot = pending.get(msg.id);
    if (slot) {
      pending.delete(msg.id);
      if (msg.error) slot.reject(new Error(`${slot.method}: ${msg.error.message}`));
      else slot.resolve(msg.result);
    }
  };
  return {
    ready,
    send(method, params = {}) {
      const id = ++seq;
      return new Promise((res, rej) => {
        pending.set(id, { resolve: res, reject: rej, method });
        ws.send(JSON.stringify({ id, method, params }));
      });
    },
    close: () => ws.close(),
  };
}

function makeEval(cdp) {
  return async function evaluate(expression) {
    const result = await cdp.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      const text = result.exceptionDetails.exception?.description ?? result.exceptionDetails.text;
      throw new Error(`In-page evaluation failed: ${text}\n${expression}`);
    }
    return result.result.value;
  };
}

async function waitFor(evaluate, expression, { timeoutMs = 60_000, label = expression } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

// --- Main -------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.list) {
    for (const shot of SHOTS) console.log(`  ${shot.name}`);
    return;
  }

  const wanted = args.only.length > 0 ? SHOTS.filter((s) => args.only.some((part) => s.name.includes(part))) : SHOTS;
  if (wanted.length === 0) throw new Error(`No shots match --only ${args.only.join(",")}`);

  const exe =
    args.exe ??
    [join(ROOT, "target", "release", "scan-kit-desktop.exe"), join(ROOT, "target", "debug", "scan-kit-desktop.exe")].find(
      existsSync,
    );
  if (exe == null) {
    throw new Error("No built binary found. Run `cargo build -p scan-kit-desktop` first, or pass --exe.");
  }
  if (!existsSync(args.data)) throw new Error(`Data folder not found: ${args.data} (pass --data)`);
  mkdirSync(args.out, { recursive: true });

  console.log(`Launching ${exe}`);
  const child = spawn(exe, [], {
    env: { ...process.env, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${args.port}` },
    stdio: "ignore",
  });
  let devServer = null;
  try {
    const target = await findPageTarget(args.port);
    const cdp = connectCdp(target.webSocketDebuggerUrl);
    await cdp.ready;
    const evaluate = makeEval(cdp);
    const shots = (expr) => evaluate(`window.__scanKitShots.${expr}`);

    // A `cargo build` debug binary points at the vite dev server, not embedded
    // assets — if the page is the devUrl and nothing answers there, start it.
    // (When vite is down the webview shows chrome-error://, so check the
    // target's configured URL, not location.href.)
    if (/localhost:1420/.test(target.url)) {
      const up = await fetch("http://localhost:1420").then((r) => r.ok, () => false);
      if (!up) {
        console.log("Debug binary uses the vite devUrl — starting `npm run dev`");
        devServer = spawn("npm run dev", { cwd: resolve(HERE, ".."), stdio: "ignore", shell: true });
        const deadline = Date.now() + 180_000;
        while (Date.now() < deadline) {
          if (await fetch("http://localhost:1420").then((r) => r.ok, () => false)) break;
          await sleep(500);
        }
      }
      await cdp.send("Page.enable");
      await cdp.send("Page.reload", { ignoreCache: true });
      await sleep(500);
    }

    await waitFor(evaluate, "!!window.__scanKitShots", { timeoutMs: 120_000, label: "the app to finish mounting" });

    // Remember the window geometry so it can be restored when we're done.
    const savedGeometry = await shots(`invoke("scan_kit_window_geometry", {})`);

    await shots(`openLocation(${JSON.stringify(args.data)})`);
    await waitFor(evaluate, "(window.__scanKitShots.sessionIds() ?? []).length > 0", {
      timeoutMs: 120_000,
      label: "session rows to appear",
    });

    const available = new Set(await shots("sessionIds()"));

    if (args.probe) {
      await shots(`selectSessions(${JSON.stringify([[...available][0]])})`);
      await shots(`openAnalysis(${JSON.stringify(args.probe)})`);
      await waitFor(evaluate, "!!window.__scanKitShots.view", { label: `${args.probe} to open` });
      await waitFor(evaluate, "(window.__scanKitShots.view.controls() ?? []).length > 0", {
        timeoutMs: 180_000,
        label: `${args.probe} controls`,
      });
      console.log(JSON.stringify(await shots("view.controls()"), null, 2));
      return;
    }

    const missing = [...new Set(wanted.flatMap((s) => s.sessions ?? []))].filter((id) => !available.has(id));
    if (missing.length > 0) {
      console.warn(`Warning: sessions not found in ${args.data}: ${missing.join(", ")} — falling back to first rows`);
    }

    let inView = false;
    for (const shot of wanted) {
      process.stdout.write(`  ${shot.name} … `);

      if (inView) {
        await shots("back()");
        inView = false;
        await sleep(400);
      }
      if (shot.size) await shots(`setSize(${shot.size[0]}, ${shot.size[1]})`);

      const sessionIds = (shot.sessions ?? []).filter((id) => available.has(id));
      if (shot.sessions != null) {
        await shots(`selectSessions(${JSON.stringify(sessionIds.length > 0 ? sessionIds : [...available].slice(0, 2))})`);
        await sleep(300);
      }

      if (shot.view) {
        await shots(`openAnalysis(${JSON.stringify(shot.view)})`);
        await waitFor(evaluate, "!!window.__scanKitShots.view?.ready", { label: `${shot.view} to open` });
        for (const [control, option] of Object.entries(shot.controls ?? {})) {
          // The view object is re-assigned as AnalysisView state changes; retry
          // across the brief gap instead of racing it.
          for (let attempt = 0; ; attempt++) {
            try {
              await shots(`view.apply(${JSON.stringify(control)}, ${JSON.stringify(option)})`);
              break;
            } catch (error) {
              if (attempt >= 20) throw error;
              await sleep(250);
            }
          }
          // Let the applied option start a reload before checking readiness.
          await sleep(400);
        }
        await waitFor(evaluate, "window.__scanKitShots.view?.ready?.() === true", {
          timeoutMs: 180_000,
          label: `${shot.view} plot`,
        });
        inView = true;
      } else {
        await shots(`selectTab(${JSON.stringify(shot.tab)})`);
      }

      await sleep(shot.settleMs ?? 1200);
      // Docs shouldn't leak the local machine path; show a relative-looking one.
      await evaluate(`(() => {
        const prefix = ${JSON.stringify(args.data.replaceAll("\\", "/"))};
        for (const el of document.querySelectorAll("[title]")) {
          const t = el.getAttribute("title") ?? "";
          if (t.replaceAll("\\\\", "/").startsWith(prefix + "/")) {
            const short = "test_data/" + t.replaceAll("\\\\", "/").slice(prefix.length + 1);
            el.setAttribute("title", short);
            if ((el.textContent ?? "").trim() === t) el.textContent = short;
          }
        }
      })()`);
      const png = await cdp.send("Page.captureScreenshot", { format: "png" });
      const file = join(args.out, `${shot.name}.png`);
      writeFileSync(file, Buffer.from(png.data, "base64"));
      console.log(file);
    }

    if (savedGeometry) {
      await shots(
        `invoke("scan_kit_set_window_geometry", { width: ${savedGeometry.width}, height: ${savedGeometry.height}, x: ${savedGeometry.x}, y: ${savedGeometry.y} })`,
      );
    }
    cdp.close();
  } finally {
    if (!args.keepOpen) {
      child.kill("SIGTERM");
    }
    if (devServer != null) {
      // npm run dev is shell-wrapped; kill the whole tree so vite dies too.
      spawn("taskkill", ["/pid", String(devServer.pid), "/T", "/F"], { stdio: "ignore" });
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
