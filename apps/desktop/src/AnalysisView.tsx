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
  parseSegments,
  playheadReplay,
  segmentChoices,
  type ControlSlot,
  type GrainMemory,
} from "@/analysis-controls";
import { SegmentList } from "@/segment-list";
import { AnalysisMenu, analysisId, analysisName } from "@/analysis-menu";
import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { SessionList } from "@/session-list";
import { optionIcon } from "@/option-icons";
import { backingSize, plotHeader, shownHeader, type PlotHeader, type ViewControl } from "@/plot-header";
import { usePageLoad } from "@/page-load";
import { sessionColor, shownSessionIds } from "@/session-colors";
import { acceptReport, bytesOf, parsePoll, type Report } from "@/task-client";
import { dismissNotice, notifyError } from "@/notify";
import { ScrubBar, useScrub, type Scrub } from "@/scrub-bar";
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

type DoseFrame = { x: number; y: number; w: number; h: number };

function isDoseFrame(value: unknown): value is DoseFrame {
  if (typeof value !== "object" || value == null) {
    return false;
  }
  const frame = value as DoseFrame;
  return [frame.x, frame.y, frame.w, frame.h].every(
    (item) => typeof item === "number" && Number.isFinite(item),
  );
}

function exportStudyReport(table: { columns: readonly string[]; rows: readonly (readonly string[])[] }) {
  const lines = [table.columns.join("\t"), ...table.rows.map((row) => row.join("\t"))];
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "scan-kit-report.txt";
  link.click();
  URL.revokeObjectURL(url);
}

function canvasScale(node: HTMLCanvasElement | null): { sx: number; sy: number } {
  if (node == null) {
    return { sx: 1, sy: 1 };
  }
  const rect = node.getBoundingClientRect();
  return {
    sx: rect.width > 0 ? node.width / rect.width : 1,
    sy: rect.height > 0 ? node.height / rect.height : 1,
  };
}

function DoseChrome({
  controls,
  frames,
  scale,
  resolved,
  onChange,
  onAction,
}: {
  controls: readonly ViewControl[];
  frames: readonly DoseFrame[];
  scale: { sx: number; sy: number };
  resolved: Readonly<Record<string, string>>;
  onChange: (id: string, value: string) => void;
  onAction: (panel: number, action: string) => void;
}) {
  const sx = scale.sx > 0 ? scale.sx : 1;
  const sy = scale.sy > 0 ? scale.sy : 1;
  const items = [
    { id: "cell0", panel: 0, kind: "cell" },
    { id: "cell1", panel: 1, kind: "cell" },
    { id: "cell2", panel: 2, kind: "cell" },
    { id: "cell3", panel: 3, kind: "cell" },
    { id: "plot0", panel: 4, kind: "plot" },
    { id: "plot1", panel: 5, kind: "plot" },
  ] as const;
  return (
    <>
      {items.map((item) => {
        const control = controls.find((entry) => entry.id === item.id);
        const frame = frames[item.panel];
        if (control == null || frame == null || frame.w < 8 || frame.h < 8) {
          return null;
        }
        const value = resolved[item.id] ?? control.value;
        const slice = item.kind === "cell" && value !== "3D";
        const profile = item.kind === "plot" && value !== "DVH" && value !== "Gamma histogram";
        return (
          <div
            key={item.id}
            className="pointer-events-none absolute flex items-start justify-between"
            style={{ left: frame.x / sx, top: frame.y / sy, width: frame.w / sx }}
          >
            <div className="pointer-events-auto flex">
              {slice ? (
                <Button type="button" size="sm" variant="outline" onClick={() => onAction(item.panel, "rotate")}>
                  90°
                </Button>
              ) : null}
              {slice || profile ? (
                <Button type="button" size="sm" variant="outline" onClick={() => onAction(item.panel, "integral")}>
                  ∫
                </Button>
              ) : null}
            </div>
            <div className="pointer-events-auto min-w-0">
              <ChoiceSelect
                control={control}
                value={value}
                disabled={false}
                onChange={(next) => onChange(item.id, next)}
              />
            </div>
          </div>
        );
      })}
    </>
  );
}

// A canvas keeps the first context it is given, and StrictMode mounts twice,
// so each canvas gets one WebPlot for its lifetime.
const plotters = new WeakMap<HTMLCanvasElement, Promise<Plotter>>();

function plotterFor(node: HTMLCanvasElement): Promise<Plotter> {
  let pending = plotters.get(node);
  if (pending == null) {
    pending = import("@/wasm/scan_kit_plot.js").then(async (wasm) => {
      await wasm.default();
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
  onChange,
}: {
  control: ViewControl;
  value: string;
  disabled?: boolean;
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
      <SelectTrigger size="sm" className="w-full cursor-pointer">
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
  const [loading, setLoading] = useState(true);
  usePageLoad(loading, taskReport?.done ?? 0, taskReport?.total ?? 0);
  const frame = useRef(0);
  const openSeq = useRef(0);
  const [options, setOptions] = useState<Record<string, string>>({});
  const lineToken = useRef("");
  const nextToken = useRef("");
  const [lineEpoch, setLineEpoch] = useState(0);
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
  const [frames, setFrames] = useState<DoseFrame[]>([]);
  const [frameScale, setFrameScale] = useState({ sx: 1, sy: 1 });
  const shown = (meta?.panels.length ?? 0) > 0;

  const requestDraw = () => {
    if (frame.current !== 0) {
      return;
    }
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      try {
        plotter.current?.render();
      } catch (reason) {
        notifyError(messageOf(reason), "analysis");
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
    if (
      plotter.current == null &&
      (node.width !== next.width || node.height !== next.height)
    ) {
      node.width = next.width;
      node.height = next.height;
    }
    plotter.current?.resize(next.width, next.height);
    requestDraw();
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

  const loadPayload = () => {
    const plot = plotter.current;
    const bytes = payload.current;
    if (plot == null || bytes == null) {
      return;
    }
    payload.current = null;
    try {
      plot.load(bytes);
      if (nextToken.current !== "") {
        lineToken.current = nextToken.current;
      }
      if (playhead.current.on) {
        slideRef.current(playhead.current, false);
      }
      fitCanvas();
      plot.render();
    } catch (reason) {
      // The picture kept its lines, but this payload left them out. Ask again
      // with the full traces.
      if (lineToken.current !== "") {
        lineToken.current = "";
        nextToken.current = "";
        setLineEpoch((epoch) => epoch + 1);
        return;
      }
      notifyError(messageOf(reason), "analysis");
    }
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
  }, [shown]);

  useEffect(() => {
    if (shown) {
      fitCanvas();
    }
  }, [shown]);

  const orderKey = sessions.map((session) => session.id).join("\0");
  const hiddenKey = hidden.join("\0");
  useEffect(() => {
    const mine = hold.current + 1;
    hold.current = mine;
    const ticket = openSeq.current + 1;
    openSeq.current = ticket;
    const order = orderKey === "" ? [] : orderKey.split("\0");
    const shown = shownSessionIds(order, hiddenKey === "" ? [] : hiddenKey.split("\0"));
    const plotOptions: Record<string, string> = { ...options };
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
    let stop = false;
    let reveal = 0;
    const timer = window.setTimeout(() => {
      if (stop) {
        return;
      }
      const picture = (metaRef.current?.panels.length ?? 0) > 0;
      if (!picture) {
        setQuiet(null);
      }
      // A picture already on screen stays up. The hairline waits so a fast
      // refresh, such as a playhead step, does not flash over it.
      reveal = window.setTimeout(() => {
        if (!stop && openSeq.current === ticket) {
          setLoading(true);
        }
      }, picture ? 250 : 0);
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
          window.clearTimeout(reveal);
          await invoke("scan_kit_cancel", { task: started.task });
          return;
        }
        taskId.current = started.task;
        for (;;) {
          if (stop || openSeq.current !== ticket) {
            window.clearTimeout(reveal);
            return;
          }
          const raw = await invoke<ArrayBuffer | Uint8Array>("scan_kit_poll", { task: started.task });
          if (stop || openSeq.current !== ticket) {
            window.clearTimeout(reveal);
            return;
          }
          const parsed = parsePoll(bytesOf(raw));
          if (!acceptReport(parsed.report, started.task, started.generation)) {
            continue;
          }
          setTaskReport(parsed.report.finished ? null : parsed.report);
          if (parsed.report.finished && openSeq.current === ticket) {
            window.clearTimeout(reveal);
            setLoading(false);
            setSettled((count) => count + 1);
          }
          if (parsed.payload != null) {
            setPlotError(null);
            const header = plotHeader(parsed.payload);
            nextToken.current = header.lineToken;
            const chosen = shownHeader(metaRef.current, header);
            if (header.panels.length > 0) {
              if (chosen !== metaRef.current) {
                metaRef.current = chosen;
                setMeta(chosen);
              }
              setQuiet(null);
              payload.current = parsed.payload;
              loadPayload();
              dismissNotice("analysis");
            } else if (parsed.report.finished && parsed.report.phase === "done") {
              const current = metaRef.current;
              const sameView =
                current != null && current.panels.length > 0 && current.title === header.title;
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
        window.clearTimeout(reveal);
        if (openSeq.current === ticket) {
          const message = messageOf(reason);
          setLoading(false);
          setSettled((count) => count + 1);
          setPlotError(message);
          notifyError(message, "analysis");
        }
      });
    }, 150);
    return () => {
      stop = true;
      window.clearTimeout(reveal);
      window.clearTimeout(timer);
      const id = taskId.current;
      queueMicrotask(() => {
        if (hold.current === mine && id !== 0) {
          taskId.current = 0;
          void invoke("scan_kit_cancel", { task: id });
        }
      });
    };
  }, [viewId, folder, orderKey, hiddenKey, options, studyPath, lineEpoch]);

  useEffect(() => {
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
    };
    const onUp = () => {
      dragging = false;
    };
    const onMove = (event: PointerEvent) => {
      if (!dragging) {
        return;
      }
      const point = locate(event);
      plotter.current?.pan(point.x, point.y, event.movementX * point.sx, event.movementY * point.sy);
      requestDraw();
    };
    const onDouble = () => {
      plotter.current?.reset();
      requestDraw();
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("pointerdown", onDown);
    node.addEventListener("pointerup", onUp);
    node.addEventListener("pointercancel", onUp);
    node.addEventListener("pointermove", onMove);
    node.addEventListener("dblclick", onDouble);
    return () => {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("pointerdown", onDown);
      node.removeEventListener("pointerup", onUp);
      node.removeEventListener("pointercancel", onUp);
      node.removeEventListener("pointermove", onMove);
      node.removeEventListener("dblclick", onDouble);
    };
  }, []);

  useEffect(() => {
    const plot = plotter.current;
    const node = canvas.current;
    if (node != null) {
      setFrameScale(canvasScale(node));
    }
    if (viewId !== "volumetric" || plot == null) {
      setFrames([]);
      return;
    }
    try {
      const parsed: unknown = JSON.parse(plot.frames());
      setFrames(Array.isArray(parsed) ? parsed.filter(isDoseFrame) : []);
    } catch {
      setFrames([]);
    }
  }, [meta, size, viewId, settled]);

  const controls = meta?.controls ?? [];
  const resolved: Record<string, string> = {};
  for (const control of controls) {
    const stored = options[control.id];
    if (control.kind === "segments") {
      resolved[control.id] = stored != null && parseSegments(stored) != null ? stored : control.value;
      continue;
    }
    const known = control.options.some((option) => option.label === stored);
    resolved[control.id] = stored != null && known ? stored : control.value;
  }
  const byId = new Map(controls.map((control) => [control.id, control]));
  const sections = controlSections(controls).filter(
    (section) => viewId !== "volumetric" || (section.title !== "Cell" && section.title !== "Plot"),
  );
  const apply = (id: string, value: string) => {
    setOptions((current) => applyOption(current, resolved, grains.current, id, value));
  };

  const table = meta?.table;
  const columns: GridColumn[] =
    table?.columns.map((title) => ({ title, width: 180 })) ?? [];
  const getCellContent = ([col, row]: Item): GridCell => ({
    kind: GridCellKind.Text,
    data: table?.rows[row]?.[col] ?? "",
    displayData: table?.rows[row]?.[col] ?? "",
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
        {table != null && table.rows.length > 0 ? (
          <DataEditor
            width={size.width}
            height={shown ? Math.min(240, size.height) : size.height}
            columns={columns}
            rows={table.rows.length}
            getCellContent={getCellContent}
            theme={gridTheme()}
            rowMarkers="none"
          />
        ) : null}
        {shown || (table?.rows.length ?? 0) > 0 ? null : (
          <div className="text-muted-foreground flex flex-1 items-center justify-center px-6 text-center text-sm">
            {plotError ?? quiet ?? "Loading plot…"}
          </div>
        )}
        <div className={shown ? "relative min-h-0 flex-1" : "hidden"}>
          <canvas ref={canvas} className="absolute inset-0 h-full w-full touch-none" />
          {viewId === "volumetric" ? (
            <DoseChrome
              controls={controls}
              frames={frames}
              scale={frameScale}
              resolved={resolved}
              onChange={apply}
              onAction={(panel, action) => {
                plotter.current?.dose_action(panel, action);
                plotter.current?.render();
              }}
            />
          ) : null}
        </div>
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
                setLoading(true);
                void invoke<{ report: string }>("scan_kit_open_study", { path: selected })
                  .then((opened) => {
                    setStudyPath(selected);
                    setStudy(opened.report);
                  })
                  .catch((reason: unknown) => {
                    setLoading(false);
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
