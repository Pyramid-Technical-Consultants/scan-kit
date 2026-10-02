import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { notify, notifyError } from "@/notify";

type Enables = { start: boolean; pause: boolean; stop: boolean; reset: boolean };

type RunnerView = {
  connected: boolean;
  state: string;
  subtitle: string;
  progress: number | null;
  point_progress: number | null;
  point: string;
  energy: string;
  layer: string;
  elapsed: string;
  permit: string;
  points: string;
  enables: Enables;
  coach: string;
  host?: string;
  version?: string;
  device_type?: string;
  hint?: string;
  message?: string;
};

type Catalog = {
  host: string | null;
  file_dir: string | null;
  dest?: string;
  connected: boolean;
  view: RunnerView;
};

const IDLE: RunnerView = {
  connected: false,
  state: "—",
  subtitle: "Connect, upload a plan, then press Start",
  progress: null,
  point_progress: null,
  point: "—",
  energy: "—",
  layer: "—",
  elapsed: "—",
  permit: "—",
  points: "—",
  enables: { start: false, pause: false, stop: false, reset: false },
  coach: "Enter the RCI IP and click Connect.",
};

export function PlanRunner() {
  const [host, setHost] = useState("");
  const [fileDir, setFileDir] = useState<string | null>(null);
  const [csvPath, setCsvPath] = useState("");
  const [dest, setDest] = useState("");
  const [view, setView] = useState<RunnerView>(IDLE);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const inflight = useRef(false);
  const hasPlan = useRef(false);
  const destRef = useRef("");

  useEffect(() => {
    void invoke<Catalog>("scan_kit_runner_catalog")
      .then((next) => {
        setHost(next.host ?? "");
        setFileDir(next.file_dir);
        if (next.dest) {
          setDest(next.dest);
        }
        setView(next.view);
      })
      .catch((reason: unknown) => notifyError(reason));
  }, []);

  useEffect(() => {
    hasPlan.current = csvPath.length > 0;
    destRef.current = dest;
  }, [csvPath, dest]);

  useEffect(() => {
    if (!view.connected) {
      return;
    }
    let stop = false;
    const id = window.setInterval(() => {
      if (stop || busyRef.current || inflight.current) {
        return;
      }
      inflight.current = true;
      void invoke<RunnerView>("scan_kit_runner_status", {
        hasPlan: hasPlan.current,
        dest: destRef.current,
      })
        .then((next) => {
          if (!stop) {
            setView(next);
          }
        })
        .catch((reason: unknown) => {
          if (!stop) {
            setView((current) => ({ ...current, connected: false }));
            notifyError(reason, "runner-status");
          }
        })
        .finally(() => {
          inflight.current = false;
        });
    }, 1000);
    return () => {
      stop = true;
      window.clearInterval(id);
    };
  }, [view.connected]);

  async function connect() {
    const text = host.trim();
    if (text.length === 0) {
      notifyError("Enter an RCI host IP or URL.");
      return;
    }
    await run(async () => {
      const next = await invoke<RunnerView>("scan_kit_runner_connect", { host: text });
      setView(next);
      if (next.host) {
        setHost(next.host);
      }
    });
  }

  async function disconnect() {
    await run(async () => {
      const next = await invoke<Catalog>("scan_kit_runner_disconnect");
      setView(next.view);
    });
  }

  async function browsePlan() {
    const selected = await open({
      title: "input_map.csv",
      defaultPath: fileDir ?? undefined,
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    if (typeof selected !== "string") {
      return;
    }
    setCsvPath(selected);
    const parent = parentDir(selected);
    if (parent != null) {
      setFileDir(parent);
      void invoke("scan_kit_runner_remember", { fileDir: parent }).catch((reason: unknown) =>
        notifyError(reason),
      );
    }
  }

  async function upload() {
    if (csvPath.length === 0) {
      return;
    }
    await run(async () => {
      const next = await invoke<RunnerView>("scan_kit_runner_upload", { path: csvPath });
      setView(next);
      if (next.message) {
        notify(next.message);
      }
    });
  }

  async function press(action: string) {
    await run(async () => {
      setView(await invoke<RunnerView>("scan_kit_runner_control", { action }));
    });
  }

  async function browseZip() {
    const selected = await save({
      title: "Session zip",
      defaultPath: dest || (fileDir != null ? `${fileDir}/session.zip` : "session.zip"),
      filters: [{ name: "Zip", extensions: ["zip"] }],
    });
    if (typeof selected !== "string") {
      return;
    }
    const path = selected.toLowerCase().endsWith(".zip") ? selected : `${selected}.zip`;
    setDest(path);
    const parent = parentDir(path);
    if (parent != null) {
      setFileDir(parent);
      void invoke("scan_kit_runner_remember", { fileDir: parent }).catch((reason: unknown) =>
        notifyError(reason),
      );
    }
  }

  async function download() {
    if (!dest.toLowerCase().endsWith(".zip")) {
      notifyError("Choose a local .zip path for the session.");
      return;
    }
    await run(async () => {
      const saved = await invoke<{ message: string }>("scan_kit_runner_download", { dest });
      notify(saved.message);
    });
  }

  async function run(action: () => Promise<void>) {
    busyRef.current = true;
    setBusy(true);
    try {
      await action();
    } catch (reason: unknown) {
      notifyError(reason);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  const identity = view.connected
    ? view.device_type && view.device_type.length > 0
      ? view.device_type
      : "IGX"
    : "Not connected";
  const detail = view.connected
    ? [view.host, view.version].filter((part) => part && part.length > 0).join("  ·  ")
    : "Enter an RCI IP or URL";
  const csvName = csvPath.length > 0 ? fileName(csvPath) : "No plan selected";

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto p-4">
      <div className="flex max-w-3xl flex-col gap-3">
        <Field>
          <FieldLabel>RCI host</FieldLabel>
          <div className="flex gap-2">
            <Input
              value={host}
              placeholder="RCI IP — 192.168.100.184"
              onChange={(event) => setHost(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  void connect();
                }
              }}
            />
            <Button type="button" onClick={() => void connect()} disabled={busy || view.connected}>
              Connect
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => void disconnect()}
              disabled={busy || !view.connected}
            >
              Disconnect
            </Button>
          </div>
        </Field>
        <p className="text-sm">
          <span className="font-medium">{identity}</span>
          <span className="text-muted-foreground"> {detail}</span>
        </p>
      </div>

      <div className="flex max-w-3xl flex-wrap items-center gap-2">
        <p className="min-w-40 flex-1 text-sm">{csvName}</p>
        <Button type="button" variant="outline" onClick={() => void browsePlan()} disabled={busy}>
          Browse
        </Button>
        <Button
          type="button"
          onClick={() => void upload()}
          disabled={busy || !view.connected || csvPath.length === 0}
        >
          Upload to RCI
        </Button>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button type="button" onClick={() => void press("start")} disabled={busy || !view.enables.start}>
          Start
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => void press("pause")}
          disabled={busy || !view.enables.pause}
        >
          Pause
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => void press("stop")}
          disabled={busy || !view.enables.stop}
        >
          Stop
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => void press("reset")}
          disabled={busy || !view.enables.reset}
        >
          Reset
        </Button>
      </div>

      <div className="flex max-w-3xl flex-col gap-2">
        <Field>
          <FieldLabel>Session zip</FieldLabel>
          <div className="flex gap-2">
            <Input value={dest} placeholder="session.zip" onChange={(event) => setDest(event.target.value)} />
            <Button type="button" variant="outline" onClick={() => void browseZip()} disabled={busy}>
              Browse
            </Button>
            <Button
              type="button"
              onClick={() => void download()}
              disabled={busy || !view.connected || !dest.toLowerCase().endsWith(".zip")}
            >
              Download
            </Button>
          </div>
        </Field>
        {view.hint ? <p className="text-muted-foreground text-sm">{view.hint}</p> : null}
      </div>

      <div className="flex max-w-3xl flex-col gap-3">
        <div>
          <p className="text-lg font-medium">{view.state}</p>
          <p className="text-muted-foreground text-sm">{view.subtitle}</p>
        </div>
        <label className="flex flex-col gap-1 text-sm">
          Overall
          <progress max={100} value={view.progress ?? 0} />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Point
          <progress max={100} value={view.point_progress ?? 0} />
        </label>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
          <Metric label="Control point" value={view.point} />
          <Metric label="Energy" value={view.energy} />
          <Metric label="Layer" value={view.layer} />
          <Metric label="Elapsed" value={view.elapsed} />
          <Metric label="Start permit" value={view.permit} />
          <Metric label="Points" value={view.points} />
        </dl>
        <p className="text-sm">{view.coach}</p>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function parentDir(path: string): string | null {
  const trimmed = path.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index > 0 ? trimmed.slice(0, index) : null;
}

function fileName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index >= 0 ? trimmed.slice(index + 1) : trimmed;
}
