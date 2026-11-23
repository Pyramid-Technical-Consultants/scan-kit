import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  CircleCheck,
  Download,
  FolderOpen,
  Pause,
  Play,
  RotateCcw,
  Square,
  Unplug,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldDescription, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { DevicePicker } from "@/components/device-picker";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Progress, ProgressLabel, ProgressValue } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { notify, notifyError, notifySaved } from "@/notify";
import { usePageLoad } from "@/page-load";
import { SidePane } from "@/SidePane";
import { driveTask, type Report } from "@/task-client";

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
  session_id?: string;
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
  const [copyBusy, setCopyBusy] = useState(false);
  const [copyReport, setCopyReport] = useState<Report | null>(null);
  const copying = copyBusy || (copyReport != null && !copyReport.finished);
  const [waiting, setWaiting] = useState(true);
  usePageLoad(waiting || busy || copying, copyReport?.done ?? 0, copyReport?.total ?? 0);
  const copyToken = useRef<object>({});
  const copyBusyRef = useRef(false);
  const busyRef = useRef(false);
  const inflight = useRef(false);
  const statusFailures = useRef(0);
  // Bumped by every control/connect action — a status poll issued before the
  // action must not overwrite the action's fresher view when it lands late.
  const viewSeq = useRef(0);
  const hasPlan = useRef(false);
  const destRef = useRef("");

  useEffect(() => {
    return () => {
      copyToken.current = {};
    };
  }, []);

  useEffect(() => {
    let live = true;
    // A Connect landing before the catalog resolves must not be stomped back
    // to the Offline view — gate the remembered state on the poll sequence.
    const seq = viewSeq.current;
    void invoke<Catalog>("scan_kit_runner_catalog")
      .then((next) => {
        if (!live) {
          return;
        }
        setWaiting(false);
        if (viewSeq.current !== seq) {
          return;
        }
        setHost(next.host ?? "");
        setFileDir(next.file_dir);
        if (next.dest) {
          setDest(next.dest);
        }
        setView(next.view);
      })
      .catch((reason: unknown) => {
        if (!live) {
          return;
        }
        setWaiting(false);
        notifyError(reason);
      });

    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    hasPlan.current = csvPath.length > 0;
    destRef.current = dest;
  }, [csvPath, dest]);

  useEffect(() => {
    let stop = false;
    // Offline still probes slowly  a transient status failure must
    // recover on its own, not pin the view to "Offline" forever.
    const id = window.setInterval(() => {
      if (stop || busyRef.current || inflight.current) {
        return;
      }
      inflight.current = true;
      const seq = viewSeq.current;
      void invoke<RunnerView>("scan_kit_runner_status", {
        hasPlan: hasPlan.current,
        dest: destRef.current,
      })
        .then((next) => {
          if (!stop && seq === viewSeq.current) {
            statusFailures.current = 0;
            setView(next);
          }
        })
        .catch((reason: unknown) => {
          if (!stop && seq === viewSeq.current) {
            // One transient status error is not a disconnect — only drop the
            // view after a few consecutive failures.
            statusFailures.current += 1;
            if (statusFailures.current >= 3) {
              setView((current) => ({ ...current, connected: false }));
            }
            notifyError(reason, "runner-status");
          }
        })
        .finally(() => {
          inflight.current = false;
        });
    }, view.connected ? 1000 : 5000);
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
      statusFailures.current = 0;
      setView(next);
      if (next.host) {
        setHost(next.host);
      }
    });
  }

  async function disconnect() {
    await run(async () => {
      const next = await invoke<Catalog>("scan_kit_runner_disconnect");
      statusFailures.current = 0;
      setView(next.view);
    });
  }

  async function browsePlan() {
    const selected = await open({
      title: "input_map.csv",
      defaultPath: fileDir ?? undefined,
      filters: [{ name: "CSV", extensions: ["csv"] }],
    }).catch((reason: unknown) => {
      notifyError(reason);
      return null;
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
    const stem = view.session_id && view.session_id.length > 0 ? view.session_id : "session";
    const fallback = fileDir != null ? `${fileDir}/${stem}.zip` : `${stem}.zip`;
    const selected = await save({
      title: "Session zip",
      defaultPath: dest || fallback,
      filters: [{ name: "Zip", extensions: ["zip"] }],
    }).catch((reason: unknown) => {
      notifyError(reason);
      return null;
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
    // The backend trims before its own check — match it so a pasted
    // trailing space doesn't disable a valid destination.
    if (!dest.trim().toLowerCase().endsWith(".zip")) {
      notifyError("Choose a local .zip path for the session.");
      return;
    }
    // State has not flushed for a same-frame second click — check the ref.
    if (copyBusyRef.current) {
      return;
    }
    copyBusyRef.current = true;
    setCopyBusy(true);
    try {
      const mine = {};
      copyToken.current = mine;
      // The backend trims dest and reports capture warnings in the payload -
      // the toast must point at the file it actually wrote.
      let savedPath: string | null = null;
      let savedMessage: string | null = null;
      await driveTask(
        {
          view: "runner_copy",
          path: dest,
          sessionIds: [],
          options: { dest },
          background: [],
          foreground: [],
          palette: [],
        },
        (report, payload) => {
          if (copyToken.current === mine) {
            setCopyReport(report.finished ? null : report);
            if (report.finished && payload != null) {
              try {
                const done = JSON.parse(new TextDecoder().decode(payload)) as {
                  path?: string;
                  message?: string;
                };
                savedPath = typeof done.path === "string" ? done.path : null;
                savedMessage = typeof done.message === "string" ? done.message : null;
              } catch {
                // Fall through to the local dest on an unreadable payload.
              }
            }
          }
        },
        () => copyToken.current !== mine,
      ).finally(() => {
        // A failed copy throws out of driveTask — without this the bar and
        // hairline keep reporting the last fraction forever.
        if (copyToken.current === mine) {
          setCopyReport(null);
        }
      });
      if (copyToken.current !== mine) {
        return;
      }
      notifySaved(savedPath ?? dest, {
        title: "Session saved",
        description: savedMessage ?? undefined,
      });
    } catch (reason: unknown) {
      notifyError(reason);
    } finally {
      // The copy streams over HTTP for minutes — holding busyRef for it
      // would freeze Stop/Pause and skip every status poll.
      copyBusyRef.current = false;
      setCopyBusy(false);
    }
  }

  async function run(action: () => Promise<void>) {
    // busy state hasn't flushed for a same-frame second click — the ref is
    // synchronous, so check it, not busy.
    if (busyRef.current) {
      return;
    }
    viewSeq.current += 1;
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

  const connected = view.connected;
  const device = view.device_type && view.device_type.length > 0 ? view.device_type : "IGX";
  const detail = [view.host, view.version].filter((part) => part && part.length > 0).join("  ·  ");
  const csvName = csvPath.length > 0 ? fileName(csvPath) : "No plan selected";
  const pointsReady = connected && view.points === "Valid";
  const copyPercent =
    copyReport != null && copyReport.total > 0
      ? Math.min(100, (100 * copyReport.done) / copyReport.total)
      : null;

  return (
    <SidePane
      main={
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto p-4">
          <Card className="mx-auto w-full max-w-3xl">
            <CardHeader className="border-b">
              <CardTitle className="flex items-center gap-2.5 text-lg">
                {connected ? view.state : "Offline"}
                <Badge variant={stateVariant(view)}>{connected ? "Live" : "Offline"}</Badge>
              </CardTitle>
              <CardDescription>{view.subtitle}</CardDescription>
              <CardAction>
                <Badge variant={connected ? "secondary" : "outline"} title={detail || undefined}>
                  {connected ? device : "Not connected"}
                </Badge>
              </CardAction>
            </CardHeader>
            <CardContent className="flex flex-col gap-5">
              <div className="flex flex-wrap gap-2" role="group" aria-label="Run controls">
                <Button
                  type="button"
                  size="lg"
                  onClick={() => void press("start")}
                  disabled={busy || !view.enables.start}
                >
                  <Play />
                  Start
                </Button>
                <Button
                  type="button"
                  size="lg"
                  variant="secondary"
                  onClick={() => void press("pause")}
                  disabled={busy || !view.enables.pause}
                >
                  <Pause />
                  Pause
                </Button>
                <Button
                  type="button"
                  size="lg"
                  variant="destructive"
                  onClick={() => void press("stop")}
                  disabled={busy || !view.enables.stop}
                >
                  <Square />
                  Stop
                </Button>
                <Button
                  type="button"
                  size="lg"
                  variant="outline"
                  onClick={() => void press("reset")}
                  disabled={busy || !view.enables.reset}
                >
                  <RotateCcw />
                  Reset
                </Button>
              </div>
              <Separator />
              <div className="grid gap-4 sm:grid-cols-2">
                <Progress value={view.progress}>
                  <ProgressLabel>Overall</ProgressLabel>
                  <ProgressValue>{(formatted) => formatted ?? "—"}</ProgressValue>
                </Progress>
                <Progress value={view.point_progress}>
                  <ProgressLabel>Point</ProgressLabel>
                  <ProgressValue>{(formatted) => formatted ?? "—"}</ProgressValue>
                </Progress>
              </div>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
                <Metric label="Control point" value={view.point} />
                <Metric label="Energy" value={view.energy} />
                <Metric label="Layer" value={view.layer} />
                <Metric label="Elapsed" value={view.elapsed} />
                <Metric label="Start permit">
                  <Badge variant={view.permit === "Granted" ? "default" : "outline"}>
                    {view.permit}
                  </Badge>
                </Metric>
                <Metric label="Points">
                  <Badge variant={view.points === "Valid" ? "secondary" : "outline"}>
                    {view.points}
                  </Badge>
                </Metric>
              </dl>
            </CardContent>
            <CardFooter className="text-muted-foreground text-sm">{view.coach}</CardFooter>
          </Card>
        </div>
      }
      side={
        <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-x-hidden overflow-y-auto p-3">
          <FieldSet className="gap-3 rounded-lg border border-border p-3">
            <FieldLegend variant="label" className="w-full">
              <span className="flex items-center justify-between gap-2">
                Controller
                {connected ? (
                  <Badge variant="secondary">
                    <CircleCheck data-icon="inline-start" />
                    Connected
                  </Badge>
                ) : (
                  <Badge variant="outline">Step 1</Badge>
                )}
              </span>
            </FieldLegend>
            <Field>
              <FieldLabel>RCI host</FieldLabel>
              <DevicePicker
                value={host}
                onChange={setHost}
                onConnect={() => void connect()}
                disabled={connected}
                busy={busy}
              />
              <FieldDescription>
                IP address or the controller URL, such as http://192.168.100.184/io/ — or scan
                the network to find a device.
              </FieldDescription>
            </Field>
            {connected ? (
              <div className="flex items-center gap-2">
                <p className="min-w-0 flex-1 truncate text-sm" title={`${device} ${detail}`}>
                  <span className="font-medium">{device}</span>{" "}
                  <span className="text-muted-foreground">{detail}</span>
                </p>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => void disconnect()}
                  disabled={busy}
                >
                  <Unplug />
                  Disconnect
                </Button>
              </div>
            ) : null}
          </FieldSet>

          <FieldSet className="gap-3 rounded-lg border border-border p-3">
            <FieldLegend variant="label" className="w-full">
              <span className="flex items-center justify-between gap-2">
                Plan
                {pointsReady ? (
                  <Badge variant="secondary">
                    <CircleCheck data-icon="inline-start" />
                    Loaded
                  </Badge>
                ) : (
                  <Badge variant="outline">Step 2</Badge>
                )}
              </span>
            </FieldLegend>
            <div className="flex items-center gap-2">
              <p
                className={`min-w-0 flex-1 truncate text-sm ${csvPath.length > 0 ? "" : "text-muted-foreground"}`}
                title={csvPath.length > 0 ? csvPath : undefined}
              >
                {csvName}
              </p>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => void browsePlan()}
                disabled={busy}
              >
                <FolderOpen />
                Browse
              </Button>
            </div>
            <Button
              type="button"
              onClick={() => void upload()}
              disabled={busy || !connected || csvPath.length === 0}
            >
              <Upload />
              Upload to RCI
            </Button>
            {connected ? (
              <p className="text-muted-foreground flex items-center gap-2 text-sm">
                Points on controller
                <Badge variant={pointsReady ? "secondary" : "outline"}>{view.points}</Badge>
              </p>
            ) : null}
          </FieldSet>

          <FieldSet className="gap-3 rounded-lg border border-border p-3">
            <FieldLegend variant="label" className="w-full">
              <span className="flex items-center justify-between gap-2">
                Session
                <Badge variant="outline">Step 3</Badge>
              </span>
            </FieldLegend>
            <Field>
              <FieldLabel>
              Session zip
              {view.session_id ? ` · ${view.session_id}` : ""}
            </FieldLabel>
              <InputGroup>
                <InputGroupInput
                  value={dest}
                  placeholder={
                    view.session_id && view.session_id.length > 0
                      ? `${view.session_id}.zip`
                      : "session.zip"
                  }
                  onChange={(event) => setDest(event.target.value)}
                />
                <InputGroupAddon align="inline-end">
                  <InputGroupButton
                    onClick={() => void browseZip()}
                    disabled={busy || copying}
                  >
                    Browse
                  </InputGroupButton>
                </InputGroupAddon>
              </InputGroup>
              <FieldDescription>
                {view.hint ?? "After the run, download the session folder as a zip."}
              </FieldDescription>
            </Field>
            {copyReport != null ? (
              <Progress value={copyPercent}>
                <ProgressLabel>Downloading</ProgressLabel>
                <ProgressValue />
              </Progress>
            ) : null}
            <Button
              type="button"
              disabled={busy || copying || !connected || !dest.trim().toLowerCase().endsWith(".zip")}
              onClick={() => void download()}
            >
              <Download />
              Download
            </Button>
          </FieldSet>
        </div>
      }
    />
  );
}

function stateVariant(view: RunnerView): "default" | "secondary" | "destructive" | "outline" {
  if (!view.connected) {
    return "outline";
  }
  const state = view.state.toLowerCase();
  if (state.includes("fault") || state.includes("error")) {
    return "destructive";
  }
  if (["dosing", "active", "running", "completed"].includes(state)) {
    return "default";
  }
  return "secondary";
}

function Metric({
  label,
  value,
  children,
}: {
  label: string;
  value?: string;
  children?: React.ReactNode;
}) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="mt-0.5">{children ?? <span className="tabular-nums">{value}</span>}</dd>
    </div>
  );
}

function parentDir(path: string): string | null {
  const trimmed = path.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  // A drive-root child has its separator at index 1 - the parent is
  // "C:\", not the drive-relative "C:".
  if (index === 1 && trimmed[1] === ":") {
    return trimmed.slice(0, 2) + "\\";
  }
  return index > 0 ? trimmed.slice(0, index) : null;
}

function fileName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index >= 0 ? trimmed.slice(index + 1) : trimmed;
}
