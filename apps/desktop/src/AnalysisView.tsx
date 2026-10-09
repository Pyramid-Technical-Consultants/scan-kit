import { createElement, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ArrowLeft } from "lucide-react";
import {
  DataEditor,
  GridCellKind,
  type GridCell,
  type GridColumn,
  type Item,
} from "@glideapps/glide-data-grid";
import { gridTheme, tokenColor } from "@/grid-theme";

import {
  applyOption,
  controlDisabled,
  controlSections,
  localLinePlot,
  paintSpec,
  parseSegments,
  playheadReplay,
  sceneOptions,
  segmentChoices,
  type ControlSlot,
  type GrainMemory,
} from "@/analysis-controls";
import { SegmentList } from "@/segment-list";
import { AnalysisMenu, analysisId, analysisName } from "@/analysis-menu";
import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Field, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { SessionList } from "@/session-list";
import { optionIcon } from "@/option-icons";
import {
  applyLevelPaint,
  backingSize,
  plotHeader,
  shownHeader,
  type PlotHeader,
  type ViewControl,
} from "@/plot-header";
import { usePageLoad } from "@/page-load";
import { sessionColor, shownSessionIds } from "@/session-colors";
import { acceptReport, bytesOf, parsePoll, type Report } from "@/task-client";
import { dismissNotice, notifyError } from "@/notify";
import { ScrubBar, useScrub, type Scrub } from "@/scrub-bar";
import { DoseBoard } from "@/DoseBoard";
import { SidePane } from "@/SidePane";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type Plotter = import("@/wasm/scan_kit_plot.js").WebPlot;

function paintPlot(plot: Plotter, spec: string): string {
  const raw = plot.paint(spec);
  return typeof raw === "string" ? raw : "";
}

function rangeBound(
  options: readonly { id: string; label: string }[],
  id: string,
  fallback: number,
): number {
  const found = options.find((option) => option.id === id);
  const value = Number(found?.label);
  return Number.isFinite(value) ? value : fallback;
}

function formatMm(value: number): string {
  return value
    .toFixed(4)
    .replace(/0+$/, "")
    .replace(/\.$/, "");
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  quick,
  disabled,
  onApply,
}: {
  label: string;
  value: string;
  min: number;
  max: number;
  step: number;
  quick: string | null;
  disabled: boolean;
  onApply: (value: string) => void;
}) {
  const [draft, setDraft] = useState(value);
  const [from, setFrom] = useState(value);
  if (from !== value) {
    setFrom(value);
    setDraft(value);
  }
  const commit = (raw: string) => {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      setDraft(value);
      return;
    }
    const text = formatMm(Math.min(max, Math.max(min, parsed)));
    setDraft(text);
    if (text !== value) {
      onApply(text);
    }
  };
  return (
    <Field orientation="horizontal" className={disabled ? "opacity-50" : undefined}>
      <FieldLabel className="flex-none! shrink-0 whitespace-nowrap">{label}</FieldLabel>
      <Input
        type="number"
        className="w-24 flex-none"
        min={min}
        max={max}
        step={step}
        value={draft}
        disabled={disabled}
        aria-label={label}
        onChange={(event) => {
          const next = event.target.value;
          const parsed = Number(next);
          const previous = Number(draft);
          setDraft(next);
          if (
            Number.isFinite(parsed) &&
            Number.isFinite(previous) &&
            Math.abs(Math.abs(parsed - previous) - step) < 1e-4
          ) {
            commit(next);
          }
        }}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            commit(event.currentTarget.value);
          }
        }}
      />
      {quick != null ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() => {
            setDraft(quick);
            if (quick !== value) {
              onApply(quick);
            }
          }}
        >
          {quick} mm
        </Button>
      ) : null}
    </Field>
  );
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function downloadText(name: string, text: string, type: string) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function exportStudyReport(table: { columns: readonly string[]; rows: readonly (readonly string[])[] }) {
  const summary = table.rows.filter((row) => row[0] !== "DVH" && row[0] !== "curve");
  const body = summary
    .map((row) => `<tr><td>${escapeHtml(row[0] ?? "")}</td><td>${escapeHtml(row[1] ?? "")}</td></tr>`)
    .join("\n");
  downloadText(
    "scan-kit-report.html",
    `<!DOCTYPE html>\n<meta charset="utf-8">\n<title>Scan Kit dose report</title>\n<table>\n<tr><th>Item</th><th>Value</th></tr>\n${body}\n</table>\n`,
    "text/html",
  );
  const curves = table.rows.filter((row) => row[0] === "DVH" || row[0] === "curve");
  if (curves.length === 0) {
    return;
  }
  const lines = ["structure,dose_gy,volume"];
  let name = "";
  for (const row of curves) {
    if (row[0] === "DVH") {
      name = row[1] ?? "";
      continue;
    }
    const [dose, volume] = (row[1] ?? "").split(",");
    const cell = name.includes(",") || name.includes('"') ? `"${name.replace(/"/g, '""')}"` : name;
    lines.push(`${cell},${dose ?? ""},${volume ?? ""}`);
  }
  downloadText("scan-kit-dvh.csv", `${lines.join("\n")}\n`, "text/csv");
}

// A canvas keeps the first context it is given, and StrictMode mounts twice,
// so each canvas gets one WebPlot for its lifetime.
const plotters = new WeakMap<HTMLCanvasElement, Promise<Plotter>>();

// The generated glue has one set of exports. Calling its init once per canvas,
// in parallel, builds six modules and the last one replaces the others, so
// every plot draws into a heap that is already gone.
let plotModule: Promise<typeof import("@/wasm/scan_kit_plot.js")> | null = null;

function loadPlotModule(): Promise<typeof import("@/wasm/scan_kit_plot.js")> {
  if (plotModule == null) {
    plotModule = import("@/wasm/scan_kit_plot.js")
      .then(async (wasm) => {
        await wasm.default();
        return wasm;
      })
      .catch((error: unknown) => {
        plotModule = null;
        throw error;
      });
  }
  return plotModule;
}

function plotterFor(node: HTMLCanvasElement): Promise<Plotter> {
  let pending = plotters.get(node);
  if (pending == null) {
    pending = loadPlotModule().then(async (wasm) => {
      const plot = await wasm.WebPlot.create(node);
      console.info(`scan-kit plot backend: ${plot.backend()}`);
      return plot;
    });
    plotters.set(node, pending);
  }
  return pending;
}

function parseColor(value: string): [number, number, number, number] {
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (context == null) {
    return [0.145, 0.145, 0.145, 1];
  }
  // Stock shadcn tokens are oklch(). getComputedStyle keeps that form, and a
  // regex for rgb() was sending the fallback gray for every channel.
  context.fillStyle = value;
  context.fillRect(0, 0, 1, 1);
  const pixel = context.getImageData(0, 0, 1, 1).data;
  return [pixel[0] / 255, pixel[1] / 255, pixel[2] / 255, pixel[3] / 255];
}

function palette(order: readonly string[], shown: readonly string[]): number[][] {
  return shown.map((id) => parseColor(sessionColor(Math.max(0, order.indexOf(id)))));
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

function ChoiceIcon({ name, icon }: { name: string; icon?: string }) {
  const found =
    (icon != null && icon.length > 0 ? optionIcon(icon) : undefined) ?? optionIcon(name);
  return found == null ? null : createElement(found);
}

function ChoiceFace({
  label,
  detail,
  icon,
}: {
  label: string;
  detail: string;
  icon: string;
}) {
  return (
    <span className="flex min-w-0 items-center gap-1.5" title={detail.length > 0 ? detail : undefined}>
      <ChoiceIcon name={label} icon={icon} />
      <span className="truncate">{label}</span>
    </span>
  );
}

function ChoiceSelect({
  control,
  value,
  disabled,
  fit = false,
  onChange,
}: {
  control: ViewControl;
  value: string;
  disabled?: boolean;
  fit?: boolean;
  onChange: (value: string) => void;
}) {
  const items = control.options.map((option) => ({ ...option, value: option.label }));
  const selected = items.find((item) => item.value === value) ?? items[0];
  return (
    <Select
      items={items}
      value={value}
      disabled={disabled}
      onValueChange={(next) => {
        if (next != null) {
          onChange(next);
        }
      }}
    >
      <SelectTrigger size="sm" className={fit ? "w-fit max-w-44 cursor-pointer" : "w-full cursor-pointer"}>
        <SelectValue>
          {selected == null ? null : (
            <ChoiceFace label={selected.label} detail={selected.detail} icon={selected.icon} />
          )}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              <ChoiceFace label={item.label} detail={item.detail} icon={item.icon} />
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}

function slotRows(slots: readonly ControlSlot[]): { inline: boolean; slots: ControlSlot[] }[] {
  const rows: { inline: boolean; slots: ControlSlot[] }[] = [];
  for (const slot of slots) {
    const last = rows[rows.length - 1];
    if (slot.kind === "check" && last != null && last.slots.every((item) => item.kind === "check")) {
      last.slots.push(slot);
      last.inline = true;
      continue;
    }
    rows.push({ inline: false, slots: [slot] });
  }
  return rows;
}

export function AnalysisView({
  viewId,
  folder,
  sessions,
  onBack,
  onOpenView,
}: {
  viewId: string;
  folder: string;
  sessions: readonly { id: string; note: string }[];
  onBack: () => void;
  onOpenView: (viewId: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const plotter = useRef<Plotter | null>(null);
  const payload = useRef<Uint8Array | null>(null);
  const hold = useRef(0);
  const taskId = useRef(0);
  const [taskReport, setTaskReport] = useState<Report | null>(null);
  const [arrivedKey, setArrivedKey] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const frame = useRef(0);
  const openSeq = useRef(0);
  const [options, setOptions] = useState<Record<string, string>>({});
  const optionsRef = useRef(options);
  const paintRef = useRef("");
  const [plotEpoch, setPlotEpoch] = useState(0);
  const lineToken = useRef("");
  const nextToken = useRef("");
  const [lineEpoch, setLineEpoch] = useState(0);
  const [plotHost, setPlotHost] = useState(0);
  const grains = useRef<GrainMemory>({});
  const [hidden, setHidden] = useState<string[]>([]);
  const [meta, setMeta] = useState<PlotHeader | null>(null);
  const metaRef = useRef<PlotHeader | null>(null);
  const [settled, setSettled] = useState(0);
  const scrubControl = meta?.controls.find((control) => control.kind === "scrub");
  const wasOn = useRef(false);
  const playhead = useRef<Scrub>({ on: false, at: 0, end: 0, speed: 1, window: "second", layers: [] });
  const slideRef = useRef<(scrub: Scrub, force: boolean) => void>(() => undefined);
  const playback = useScrub(
    scrubControl?.value,
    settled,
    (text) => {
      setOptions((current) => (current.scrub === text ? current : { ...current, scrub: text }));
    },
    viewId === "timeline" || viewId === "distribution",
    (next) => {
      playhead.current = next;
      slideRef.current(next, true);
    },
    (viewId === "timeline" && playheadReplay(options)) || viewId === "distribution",
  );
  const [plotError, setPlotError] = useState<string | null>(null);
  const [quiet, setQuiet] = useState<string | null>(null);
  const [studyPath, setStudyPath] = useState<string | null>(null);
  const [study, setStudy] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 960, height: 640 });
  const shown = (meta?.panels.length ?? 0) > 0;
  const viewRef = useRef(viewId);
  const dosePlots = useRef<(Plotter | null)[]>([null, null, null, null, null, null]);
  const picture = useRef<Uint8Array | null>(null);

  const requestDraw = () => {
    if (frame.current !== 0) {
      return;
    }
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      const plots =
        viewRef.current === "volumetric"
          ? dosePlots.current.filter((plot): plot is Plotter => plot != null)
          : plotter.current == null
            ? []
            : [plotter.current];
      for (const plot of plots) {
        try {
          plot.render();
        } catch (reason) {
          notifyError(messageOf(reason), "analysis");
        }
      }
    });
  };

  const fitCanvas = () => {
    const node = canvas.current;
    if (node == null) {
      return;
    }
    const rect = node.getBoundingClientRect();
    const next = backingSize(rect.width, rect.height, window.devicePixelRatio);
    // The plotter configures the drawing buffer. Assigning width here resets it.
    if (plotter.current == null && (node.width !== next.width || node.height !== next.height)) {
      node.width = next.width;
      node.height = next.height;
    }
    plotter.current?.resize(next.width, next.height);
    requestDraw();
  };

  const afterDoseInput = (index: number) => {
    const source = dosePlots.current[index];
    if (source == null) {
      return;
    }
    const cursor = Array.from(source.dose_cursor());
    let shared = false;
    if (cursor.length === 3) {
      for (let otherIndex = 0; otherIndex < dosePlots.current.length; otherIndex += 1) {
        if (otherIndex === index) {
          continue;
        }
        const other = dosePlots.current[otherIndex];
        if (other == null) {
          continue;
        }
        const had = Array.from(other.dose_cursor());
        if (
          had.length === 3 &&
          had[0] === cursor[0] &&
          had[1] === cursor[1] &&
          had[2] === cursor[2]
        ) {
          continue;
        }
        other.set_dose_cursor(cursor[0] ?? 0, cursor[1] ?? 0, cursor[2] ?? 0);
        shared = true;
      }
    }
    if (shared) {
      requestDraw();
      return;
    }
    try {
      source.render();
    } catch (reason) {
      notifyError(messageOf(reason), "analysis");
    }
  };

  useEffect(() => {
    slideRef.current = (scrub, force) => {
      if (viewId !== "timeline" && viewId !== "distribution") {
        return;
      }
      const plot = plotter.current as
        | (Plotter & { follow?: (on: boolean, lo: number, hi: number, force: boolean) => void })
        | null;
      if (plot?.follow == null) {
        return;
      }
      if (!scrub.on) {
        if (wasOn.current) {
          plot.follow(false, 0, 0, true);
          requestDraw();
        }
        wasOn.current = false;
        return;
      }
      wasOn.current = true;
      const lo = scrub.window === "before" ? 0 : Math.max(0, scrub.at - 1);
      plot.follow(true, lo, scrub.at, force);
      requestDraw();
    };
  }, [viewId]);

  const loadPlot = (plot: Plotter, index: number | null): { painted: string; stop: boolean } => {
    const bytes = payload.current ?? picture.current;
    if (bytes == null) {
      return { painted: "", stop: true };
    }
    picture.current = bytes;
    try {
      plot.load(bytes);
      const chosen = optionsRef.current;
      if ((index == null || index === 4) && localLinePlot(chosen.plot0 ?? "")) {
        plot.set_line(4, chosen.plot0);
      }
      if ((index == null || index === 5) && localLinePlot(chosen.plot1 ?? "")) {
        plot.set_line(5, chosen.plot1);
      }
      const painted = paintRef.current.length > 0 ? paintPlot(plot, paintRef.current) : "";
      if (nextToken.current !== "") {
        lineToken.current = nextToken.current;
      }
      if (index == null && playhead.current.on) {
        slideRef.current(playhead.current, false);
      }
      if (index == null) {
        fitCanvas();
      }
      plot.render();
      payload.current = null;
      return { painted, stop: false };
    } catch (reason) {
      // The picture kept its lines, but this payload left them out. Ask again
      // with the full traces.
      if (lineToken.current !== "") {
        lineToken.current = "";
        nextToken.current = "";
        setLineEpoch((epoch) => epoch + 1);
        return { painted: "", stop: true };
      }
      notifyError(messageOf(reason), "analysis");
      return { painted: "", stop: true };
    }
  };

  const loadPayload = (): string => {
    if (payload.current != null) {
      picture.current = payload.current;
    }
    if (picture.current == null) {
      return "";
    }
    const jobs: [Plotter, number | null][] =
      viewRef.current === "volumetric"
        ? dosePlots.current.flatMap((plot, index) => (plot == null ? [] : [[plot, index] as [Plotter, number]]))
        : plotter.current == null
          ? []
          : [[plotter.current, null]];
    if (jobs.length === 0) {
      return "";
    }
    let painted = "";
    for (const [plot, index] of jobs) {
      const loaded = loadPlot(plot, index);
      if (loaded.painted.length > 0) {
        painted = loaded.painted;
      }
      if (loaded.stop) {
        return painted;
      }
    }
    return painted;
  };

  useEffect(() => {
    const node = host.current;
    if (node == null) {
      return;
    }
    const measure = () => {
      const rect = node.getBoundingClientRect();
      setSize({
        width: Math.max(16, Math.round(rect.width)),
        height: Math.max(16, Math.round(rect.height)),
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (viewId === "volumetric") {
      return;
    }
    const node = canvas.current;
    // The canvas stays in the tree while hidden. A surface created at that
    // 0×0 box, then resized, drops the only frame and the view stays black.
    if (node == null || !shown) {
      return;
    }
    let live = true;
    const observer = new ResizeObserver(fitCanvas);
    observer.observe(node);
    fitCanvas();
    plotterFor(node)
      .then((plot) => {
        if (!live) {
          return;
        }
        plotter.current = plot;
        setPlotEpoch((epoch) => epoch + 1);
        fitCanvas();
        loadPayload();
      })
      .catch((reason: unknown) => notifyError(messageOf(reason), "analysis"));
    return () => {
      live = false;
      observer.disconnect();
      plotter.current = null;
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, [shown, viewId]);

  useEffect(() => {
    viewRef.current = viewId;
    if (shown) {
      fitCanvas();
    }
  }, [shown, viewId]);

  const orderKey = sessions.map((session) => session.id).join("\0");
  const hiddenKey = hidden.join("\0");
  const sceneKey = viewId === "volumetric" ? JSON.stringify(sceneOptions(options)) : options;
  const requestKey = JSON.stringify([
    viewId,
    folder,
    orderKey,
    hiddenKey,
    sceneKey,
    studyPath ?? "",
    lineEpoch,
    plotHost,
  ]);
  const loading = arrivedKey !== requestKey || opening;
  usePageLoad(loading, taskReport?.done ?? 0, taskReport?.total ?? 0);
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);
  useEffect(() => {
    const mine = hold.current + 1;
    hold.current = mine;
    const ticket = openSeq.current + 1;
    openSeq.current = ticket;
    const order = orderKey === "" ? [] : orderKey.split("\0");
    const shown = shownSessionIds(order, hiddenKey === "" ? [] : hiddenKey.split("\0"));
    let stop = false;
    const timer = window.setTimeout(() => {
      if (stop) {
        return;
      }
      const plotOptions: Record<string, string> = { ...optionsRef.current };
      if (viewId === "timeline" || viewId === "distribution") {
        const head = playhead.current;
        plotOptions.scrub = JSON.stringify({
          on: head.on,
          at: head.at,
          end: head.end,
          speed: head.speed,
          window: head.window,
        });
      }
      if (viewId === "volumetric" && studyPath != null) {
        plotOptions.study = studyPath;
      }
      if (lineToken.current !== "") {
        plotOptions._lines = lineToken.current;
      }
      if ((metaRef.current?.panels.length ?? 0) === 0) {
        setQuiet(null);
      }
      setPlotError(null);
      void (async () => {
        const started = await invoke<{ task: number; generation: number }>("scan_kit_start", {
          view: viewId,
          path: folder,
          sessionIds: shown,
          options: plotOptions,
          background: parseColor(tokenColor("--background")),
          foreground: parseColor(tokenColor("--foreground")),
          palette: palette(order, shown),
        });
        if (stop || openSeq.current !== ticket) {
          await invoke("scan_kit_cancel", { task: started.task });
          return;
        }
        taskId.current = started.task;
        for (;;) {
          if (stop || openSeq.current !== ticket) {
            return;
          }
          const raw = await invoke<ArrayBuffer | Uint8Array>("scan_kit_poll", { task: started.task });
          if (stop || openSeq.current !== ticket) {
            return;
          }
          const parsed = parsePoll(bytesOf(raw));
          if (!acceptReport(parsed.report, started.task, started.generation)) {
            continue;
          }
          setTaskReport(parsed.report.finished ? null : parsed.report);
          if (parsed.report.finished && openSeq.current === ticket) {
            setArrivedKey(requestKey);
            setSettled((count) => count + 1);
          }
          if (parsed.payload != null) {
            setPlotError(null);
            const parsedHeader = plotHeader(parsed.payload);
            nextToken.current = parsedHeader.lineToken;
            if (parsedHeader.panels.length > 0) {
              payload.current = parsed.payload;
              const header = applyLevelPaint(parsedHeader, loadPayload()) ?? parsedHeader;
              const chosen = shownHeader(metaRef.current, header);
              if (chosen !== metaRef.current) {
                metaRef.current = chosen;
                setMeta(chosen);
              }
              setQuiet(null);
              dismissNotice("analysis");
            } else if (parsed.report.finished && parsed.report.phase === "done") {
              const current = metaRef.current;
              const sameView =
                current != null && current.panels.length > 0 && current.title === parsedHeader.title;
              if (!sameView) {
                if (current != null) {
                  metaRef.current = null;
                  setMeta(null);
                }
                setQuiet("No samples in this window.");
              }
            }
          }
          if (parsed.report.finished) {
            if (parsed.report.phase === "failed") {
              const message = parsed.report.note.length > 0 ? parsed.report.note : "The plot failed.";
              setPlotError(message);
              notifyError(message, "analysis");
            }
            return;
          }
        }
      })().catch((reason: unknown) => {
        if (openSeq.current === ticket) {
          const message = messageOf(reason);
          setArrivedKey(requestKey);
          setSettled((count) => count + 1);
          setPlotError(message);
          notifyError(message, "analysis");
        }
      });
    }, 150);
    return () => {
      stop = true;
      window.clearTimeout(timer);
      const id = taskId.current;
      queueMicrotask(() => {
        if (hold.current === mine && id !== 0) {
          taskId.current = 0;
          void invoke("scan_kit_cancel", { task: id });
        }
      });
    };
  }, [viewId, folder, orderKey, hiddenKey, sceneKey, studyPath, lineEpoch, plotHost, requestKey]);

  useEffect(() => {
    if (viewId === "volumetric") {
      return;
    }
    const node = canvas.current;
    if (node == null) {
      return;
    }
    let dragging = false;
    const locate = (event: { clientX: number; clientY: number }) => {
      const rect = node.getBoundingClientRect();
      const sx = rect.width > 0 ? node.width / rect.width : 1;
      const sy = rect.height > 0 ? node.height / rect.height : 1;
      return {
        x: (event.clientX - rect.left) * sx,
        y: (event.clientY - rect.top) * sy,
        sx,
        sy,
      };
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const point = locate(event);
      plotter.current?.zoom(point.x, point.y, event.deltaY);
      requestDraw();
    };
    const onDown = (event: PointerEvent) => {
      dragging = true;
      node.setPointerCapture(event.pointerId);
      node.focus({ preventScroll: true });
    };
    const onUp = () => {
      dragging = false;
    };
    const onMove = (event: PointerEvent) => {
      if (!dragging) {
        return;
      }
      const point = locate(event);
      plotter.current?.pan(
        point.x,
        point.y,
        event.movementX * point.sx,
        event.movementY * point.sy,
        event.buttons,
        event.shiftKey,
      );
      requestDraw();
    };
    const onDouble = () => {
      plotter.current?.reset();
      requestDraw();
    };
    const onKey = (event: KeyboardEvent) => {
      const used = plotter.current?.dose_key(event.key, event.ctrlKey) ?? false;
      if (!used) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      requestDraw();
    };
    const onMenu = (event: Event) => {
      event.preventDefault();
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("pointerdown", onDown);
    node.addEventListener("pointerup", onUp);
    node.addEventListener("pointercancel", onUp);
    node.addEventListener("pointermove", onMove);
    node.addEventListener("dblclick", onDouble);
    node.addEventListener("keydown", onKey);
    node.addEventListener("contextmenu", onMenu);
    return () => {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("pointerdown", onDown);
      node.removeEventListener("pointerup", onUp);
      node.removeEventListener("pointercancel", onUp);
      node.removeEventListener("pointermove", onMove);
      node.removeEventListener("dblclick", onDouble);
      node.removeEventListener("keydown", onKey);
      node.removeEventListener("contextmenu", onMenu);
    };
  }, [viewId, shown]);

  const controls = meta?.controls ?? [];
  const resolved: Record<string, string> = {};
  for (const control of controls) {
    const stored = options[control.id];
    if (control.kind === "segments") {
      resolved[control.id] = stored != null && parseSegments(stored) != null ? stored : control.value;
      continue;
    }
    if (control.kind === "range") {
      const numeric = Number(stored);
      resolved[control.id] = stored != null && Number.isFinite(numeric) ? stored : control.value;
      continue;
    }
    if (control.kind === "number") {
      const numeric = Number(String(stored ?? "").replace(/mm/gi, "").trim());
      resolved[control.id] =
        stored != null && Number.isFinite(numeric) ? formatMm(numeric) : control.value;
      continue;
    }
    if (control.kind === "text") {
      resolved[control.id] = stored != null && stored.trim().length > 0 ? stored : control.value;
      continue;
    }
    const known = control.options.some((option) => option.label === stored);
    resolved[control.id] = stored != null && known ? stored : control.value;
  }
  const dosePaint = viewId === "volumetric" ? paintSpec(resolved, options) : "";
  useEffect(() => {
    paintRef.current = dosePaint;
  }, [dosePaint]);
  useEffect(() => {
    if (dosePaint.length === 0) {
      return;
    }
    const plots =
      viewRef.current === "volumetric"
        ? dosePlots.current.filter((plot): plot is Plotter => plot != null)
        : plotter.current == null
          ? []
          : [plotter.current];
    if (plots.length === 0) {
      return;
    }
    let raw = "";
    for (const plot of plots) {
      const next = paintPlot(plot, dosePaint);
      if (next.length > 0) {
        raw = next;
      }
    }
    requestDraw();
    setMeta((current) => {
      const next = applyLevelPaint(current, raw);
      if (next !== current) {
        metaRef.current = next;
      }
      return next;
    });
  }, [dosePaint, plotEpoch]);
  const byId = new Map(controls.map((control) => [control.id, control]));
  const sections = controlSections(controls).filter(
    (section) => viewId !== "volumetric" || (section.title !== "Cell" && section.title !== "Plot"),
  );
  const apply = (id: string, value: string) => {
    const next = applyOption(optionsRef.current, resolved, grains.current, id, value);
    optionsRef.current = next;
    setOptions(next);
    const panel = id === "plot0" ? 4 : id === "plot1" ? 5 : -1;
    if (panel >= 0 && localLinePlot(value)) {
      const plot = viewId === "volumetric" ? dosePlots.current[panel] : plotter.current;
      const swapped = plot?.set_line(panel, value) ?? false;
      if (swapped) {
        plot?.render();
      } else {
        setPlotHost((epoch) => epoch + 1);
      }
    } else if (panel >= 0) {
      setPlotHost((epoch) => epoch + 1);
    }
  };

  const table = meta?.table;
  const gridRows = (table?.rows ?? []).filter((row) => row[0] !== "DVH" && row[0] !== "curve");
  const columns: GridColumn[] =
    table?.columns.map((title) => ({ title, width: 180 })) ?? [];
  const getCellContent = ([col, row]: Item): GridCell => ({
    kind: GridCellKind.Text,
    data: gridRows[row]?.[col] ?? "",
    displayData: gridRows[row]?.[col] ?? "",
    allowOverlay: false,
  });

  const renderSlot = (slot: ControlSlot, inline = false) => {
    const control = byId.get(slot.id);
    if (control == null) {
      return null;
    }
    const value = resolved[slot.id] ?? control.value;
    const label = control.label;
    const disabled = controlDisabled(slot.id, resolved);
    if (slot.kind === "segments") {
      return (
        <SegmentList
          key={slot.id}
          value={value}
          kinds={control.options.map((option) => ({
            id: option.id.length > 0 ? option.id : option.label,
            label: option.label,
          }))}
          onChange={(next) => apply(slot.id, next)}
        />
      );
    }
    if (slot.kind === "range") {
      const min = rangeBound(control.options, "min", 0);
      const max = rangeBound(control.options, "max", 1);
      const step = rangeBound(control.options, "step", 0.01);
      const numeric = Number(value);
      const current = Number.isFinite(numeric) ? numeric : min;
      return (
        <Field key={slot.id} orientation="horizontal">
          <FieldLabel className="flex-none! shrink-0 whitespace-nowrap">{label}</FieldLabel>
          <Slider
            className="min-w-0 flex-1"
            min={min}
            max={max}
            step={step}
            value={[current]}
            disabled={disabled}
            aria-label={label}
            onValueChange={(next) => {
              const level = Array.isArray(next) ? next[0] : next;
              if (typeof level === "number") {
                apply(slot.id, String(level));
              }
            }}
          />
        </Field>
      );
    }
    if (slot.kind === "number") {
      const quick = control.options.find((option) => option.id === "quick")?.label ?? null;
      return (
        <NumberField
          key={slot.id}
          label={label}
          value={value}
          min={rangeBound(control.options, "min", 0.25)}
          max={rangeBound(control.options, "max", 10)}
          step={rangeBound(control.options, "step", 0.1)}
          quick={quick}
          disabled={disabled}
          onApply={(next) => apply(slot.id, next)}
        />
      );
    }
    if (slot.kind === "text") {
      return (
        <Field key={slot.id} orientation="vertical">
          <FieldLabel>{label}</FieldLabel>
          <Textarea
            value={value}
            disabled={disabled}
            aria-label={label}
            onChange={(event) => apply(slot.id, event.target.value)}
          />
        </Field>
      );
    }
    if (slot.kind === "radio") {
      return (
        <Field
          key={slot.id}
          orientation="horizontal"
          className={disabled ? "opacity-50" : undefined}
        >
          <FieldLabel className="flex-none! shrink-0 whitespace-nowrap">{label}</FieldLabel>
          <ButtonSegmentGroup
            options={control.options}
            value={value}
            disabled={disabled}
            label={label}
            onChange={(next) => apply(slot.id, next)}
          />
        </Field>
      );
    }
    if (slot.kind === "check") {
      return (
        <Field
          key={slot.id}
          orientation="horizontal"
          className={
            [inline ? "min-w-0 flex-1" : null, disabled ? "opacity-50" : null]
              .filter((item) => item != null)
              .join(" ") || undefined
          }
        >
          <Checkbox
            id={`analysis-${slot.id}`}
            className="cursor-pointer"
            checked={value === "On"}
            disabled={disabled}
            onCheckedChange={(checked) => apply(slot.id, checked ? "On" : "Off")}
          />
          <FieldLabel
            className="cursor-pointer [&_svg:not([class*='size-'])]:size-4"
            htmlFor={`analysis-${slot.id}`}
          >
            {inline ? <ChoiceIcon name={label} /> : null}
            {label}
          </FieldLabel>
        </Field>
      );
    }
    const widget = segmentChoices(control.options) ? (
      <ButtonSegmentGroup
        options={control.options}
        value={value}
        disabled={disabled}
        onChange={(next) => apply(slot.id, next)}
      />
    ) : (
      <ChoiceSelect
        control={control}
        value={value}
        disabled={disabled}
        onChange={(next) => apply(slot.id, next)}
      />
    );
    return (
      <Field
        key={slot.id}
        orientation="horizontal"
        className={disabled ? "opacity-50" : undefined}
      >
        <FieldLabel className="flex-none! shrink-0 whitespace-nowrap">{label}</FieldLabel>
        <div className="min-w-0 flex-1">{widget}</div>
      </Field>
    );
  };

  return (
    <SidePane
      main={
      <div ref={host} className="bg-background relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        {table != null && gridRows.length > 0 ? (
          <DataEditor
            width={size.width}
            height={shown ? Math.min(240, size.height) : size.height}
            columns={columns}
            rows={gridRows.length}
            getCellContent={getCellContent}
            theme={gridTheme()}
            rowMarkers="none"
          />
        ) : null}
        {shown || gridRows.length > 0 ? null : (
          <div className="text-muted-foreground flex flex-1 items-center justify-center px-6 text-center text-sm">
            {plotError ?? quiet ?? "Loading plot…"}
          </div>
        )}
        {viewId === "volumetric" ? (
          <DoseBoard
            shown={shown}
            controls={controls}
            resolved={resolved}
            plots={dosePlots}
            acquire={plotterFor}
            onChange={apply}
            onAction={(panel, action) => {
              dosePlots.current[panel]?.dose_action(panel, action);
              afterDoseInput(panel);
            }}
            onReady={(index) => {
              const plot = dosePlots.current[index];
              if (plot == null) {
                return;
              }
              const loaded = loadPlot(plot, index);
              if (loaded.painted.length === 0) {
                return;
              }
              setMeta((current) => {
                const next = applyLevelPaint(current, loaded.painted);
                if (next !== current) {
                  metaRef.current = next;
                }
                return next;
              });
            }}
            onInput={afterDoseInput}
            onError={(reason) => notifyError(messageOf(reason), "analysis")}
            renderChoice={(control, value, change) => (
              <ChoiceSelect control={control} value={value} disabled={false} fit onChange={change} />
            )}
          />
        ) : (
          <div className={shown ? "relative min-h-0 flex-1" : "hidden"}>
            <canvas ref={canvas} tabIndex={0} className="absolute inset-0 h-full w-full touch-none" />
          </div>
        )}
        {playback.shown ? (
          <ScrubBar
            scrub={playback.scrub}
            playing={playback.playing}
            onChange={playback.update}
            onTogglePlay={playback.togglePlay}
          />
        ) : null}
      </div>
      }
      side={
        <>
        <div className="flex shrink-0 items-center gap-2 p-3 pb-0">
          <Button type="button" variant="outline" className="min-w-0 flex-1" onClick={onBack}>
            <ArrowLeft />
            Sessions
          </Button>
          <AnalysisMenu
            omit={analysisName(viewId)}
            onOpen={(name) => {
              const id = analysisId(name);
              if (id != null) {
                onOpenView(id);
              }
            }}
          />
        </div>
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
        <SessionList
          sessions={sessions}
          isChecked={(id) => !hidden.includes(id)}
          onCheckedChange={(id, next) => {
            setHidden((current) => (next ? current.filter((item) => item !== id) : [...current, id]));
          }}
        />
        {sections.map((section, index) => (
          <FieldSet key={`${section.title}-${index}`} className="gap-2 rounded-lg border border-border p-3">
            <FieldLegend variant="label">{section.title}</FieldLegend>
            {slotRows(section.slots).map((row) => {
              const lead = row.slots[0];
              if (lead == null) {
                return null;
              }
              return row.inline ? (
                <div key={lead.id} className="flex flex-row gap-3">
                  {row.slots.map((slot) => renderSlot(slot, true))}
                </div>
              ) : (
                renderSlot(lead)
              );
            })}
          </FieldSet>
        ))}
        {viewId === "volumetric" ? (
          <Button
            variant="outline"
            onClick={() => {
              void invoke<{ save_dir: string | null }>("scan_kit_phantom_catalog")
                .catch(() => ({ save_dir: null }))
                .then((catalog) =>
                  open({
                    directory: true,
                    title: "DICOM study",
                    defaultPath: catalog.save_dir ?? undefined,
                  }),
                )
                .then((selected) => {
                if (typeof selected !== "string") {
                  return;
                }
                setOpening(true);
                void invoke<{ report: string }>("scan_kit_open_study", { path: selected })
                  .then((opened) => {
                    setStudyPath(selected);
                    setStudy(opened.report);
                    setOpening(false);
                  })
                  .catch((reason: unknown) => {
                    setOpening(false);
                    notifyError(reason);
                  });
              });
            }}
          >
            Open Study
          </Button>
        ) : null}
        {viewId === "volumetric" && studyPath != null ? (
          <Button
            variant="outline"
            onClick={() => {
              setStudyPath(null);
              setStudy(null);
            }}
          >
            Close Study
          </Button>
        ) : null}
        {viewId === "volumetric" && study != null && table != null && table.rows.length > 0 ? (
          <Button variant="outline" onClick={() => exportStudyReport(table)}>
            Export report
          </Button>
        ) : null}
        {study != null ? <pre className="text-muted-foreground text-xs whitespace-pre-wrap">{study}</pre> : null}
        </div>
        </>
      }
    />
  );
}
