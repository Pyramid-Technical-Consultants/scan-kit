import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ArrowLeft, Download, Play } from "lucide-react";
import {
  DataEditor,
  GridCellKind,
  getDefaultTheme,
  type GridCell,
  type GridColumn,
  type Item,
  type Theme,
} from "@glideapps/glide-data-grid";

import { controlDisabled, controlSections, segmentChoices, type ControlSlot } from "@/analysis-controls";
import { AnalysisMenu, analysisId, analysisName } from "@/analysis-menu";
import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldContent, FieldDescription, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { optionIcon } from "@/option-icons";
import { backingSize, plotHeader, type PlotHeader, type ViewControl } from "@/plot-header";
import { sessionColor, shownSessionIds } from "@/session-colors";
import { dismissNotice, notifyError } from "@/notify";
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

function tokenColor(name: string): string {
  const probe = document.createElement("span");
  probe.style.color = `var(${name})`;
  document.body.append(probe);
  const resolved = getComputedStyle(probe).color;
  probe.remove();
  return resolved;
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

function gridTheme(): Theme {
  const base = getDefaultTheme();
  const foreground = tokenColor("--foreground");
  const muted = tokenColor("--muted-foreground");
  const card = tokenColor("--card");
  return {
    ...base,
    textDark: foreground,
    textMedium: muted,
    textHeader: foreground,
    bgCell: tokenColor("--background"),
    bgHeader: card,
    bgHeaderHovered: tokenColor("--muted"),
    borderColor: tokenColor("--border"),
    horizontalBorderColor: tokenColor("--border"),
  };
}

function wavBlob(samples: number[]): Blob {
  const rate = 8000;
  const held: number[] = [];
  for (const sample of samples) {
    for (let i = 0; i < 8; i += 1) {
      held.push(sample);
    }
  }
  const bytes = new ArrayBuffer(44 + held.length * 2);
  const view = new DataView(bytes);
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + held.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, held.length * 2, true);
  held.forEach((sample, index) => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + index * 2, clipped * 0x7fff, true);
  });
  return new Blob([bytes], { type: "audio/wav" });
}

async function playSamples(samples: number[]) {
  const rate = 8000;
  const context = new AudioContext({ sampleRate: rate });
  const buffer = context.createBuffer(1, samples.length * 8, rate);
  const channel = buffer.getChannelData(0);
  samples.forEach((sample, index) => {
    for (let i = 0; i < 8; i += 1) {
      channel[index * 8 + i] = sample;
    }
  });
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.onended = () => {
    void context.close();
  };
  source.start();
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
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
  const items = control.options.map((option) => ({ label: option, value: option }));
  const Icon = optionIcon(value);
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
        {Icon == null ? null : <Icon />}
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          {items.map((item) => {
            const ItemIcon = optionIcon(item.value);
            return (
              <SelectItem key={item.value} value={item.value}>
                {ItemIcon == null ? null : <ItemIcon />}
                {item.label}
              </SelectItem>
            );
          })}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
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
  const frame = useRef(0);
  const openSeq = useRef(0);
  const [options, setOptions] = useState<Record<string, string>>({});
  const [hidden, setHidden] = useState<string[]>([]);
  const [meta, setMeta] = useState<PlotHeader | null>(null);
  const [plotError, setPlotError] = useState<string | null>(null);
  const [studyPath, setStudyPath] = useState<string | null>(null);
  const [study, setStudy] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 960, height: 640 });
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

  const loadPayload = () => {
    const plot = plotter.current;
    const bytes = payload.current;
    if (plot == null || bytes == null) {
      return;
    }
    payload.current = null;
    try {
      plot.load(bytes);
    } catch (reason) {
      notifyError(messageOf(reason), "analysis");
      return;
    }
    fitCanvas();
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
    const ticket = openSeq.current + 1;
    openSeq.current = ticket;
    const order = orderKey === "" ? [] : orderKey.split("\0");
    const shown = shownSessionIds(order, hiddenKey === "" ? [] : hiddenKey.split("\0"));
    const plotOptions =
      viewId === "dose_volume" && studyPath != null ? { ...options, study: studyPath } : options;
    const timer = window.setTimeout(() => {
      setPlotError(null);
      void invoke<ArrayBuffer | Uint8Array>("scan_kit_open_plot", {
        view: viewId,
        path: folder,
        sessionIds: shown,
        options: plotOptions,
        background: parseColor(tokenColor("--background")),
        foreground: parseColor(tokenColor("--foreground")),
        palette: palette(order, shown),
      })
        .then((result) => {
          if (openSeq.current !== ticket) {
            return;
          }
          const bytes = result instanceof Uint8Array ? result : new Uint8Array(result);
          setPlotError(null);
          setMeta(plotHeader(bytes));
          payload.current = bytes;
          loadPayload();
          dismissNotice("analysis");
        })
        .catch((reason: unknown) => {
          if (openSeq.current === ticket) {
            const message = messageOf(reason);
            setPlotError(message);
            notifyError(message, "analysis");
          }
        });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [viewId, folder, orderKey, hiddenKey, options, studyPath]);

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

  const controls = meta?.controls ?? [];
  const resolved: Record<string, string> = {};
  for (const control of controls) {
    const stored = options[control.id];
    resolved[control.id] =
      stored != null && control.options.includes(stored) ? stored : control.value;
  }
  const byId = new Map(controls.map((control) => [control.id, control]));
  const sections = controlSections(
    viewId,
    controls.map((control) => control.id),
  );
  const apply = (id: string, value: string) => {
    setOptions((current) => ({ ...current, [id]: value }));
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

  const renderSlot = (slot: ControlSlot) => {
    const control = byId.get(slot.id);
    if (control == null) {
      return null;
    }
    const value = resolved[slot.id] ?? control.value;
    const label = slot.label ?? control.label;
    const disabled = controlDisabled(slot.id, resolved);
    if (slot.kind === "check") {
      return (
        <Field key={slot.id} orientation="horizontal" className={disabled ? "opacity-50" : undefined}>
          <Checkbox
            id={`analysis-${slot.id}`}
            className="cursor-pointer"
            checked={value === "On"}
            disabled={disabled}
            onCheckedChange={(checked) => apply(slot.id, checked ? "On" : "Off")}
          />
          <FieldLabel className="cursor-pointer" htmlFor={`analysis-${slot.id}`}>
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
    if (slot.kind === "bare") {
      return <div key={slot.id}>{widget}</div>;
    }
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
      <div ref={host} className="bg-background flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
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
        {shown ? null : (
          <div className="text-muted-foreground flex flex-1 items-center justify-center px-6 text-center text-sm">
            {plotError ?? "Loading plot…"}
          </div>
        )}
        <div className={shown ? "relative min-h-0 flex-1" : "hidden"}>
          <canvas ref={canvas} className="absolute inset-0 h-full w-full touch-none" />
        </div>
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
        <FieldSet className="gap-2 rounded-lg border border-border p-3">
          <FieldLegend variant="label">Sessions</FieldLegend>
          {sessions.map((session, index) => {
            const checked = !hidden.includes(session.id);
            const color = sessionColor(index);
            const inputId = `analysis-session-${session.id}`;
            return (
              <Field key={session.id} orientation="horizontal">
                <Checkbox
                  id={inputId}
                  className="cursor-pointer"
                  checked={checked}
                  aria-label={`Include ${session.id}`}
                  title={
                    checked
                      ? `Session color in plots: ${color}`
                      : `Hidden. Check to draw ${session.id} in ${color}`
                  }
                  style={
                    checked
                      ? { backgroundColor: color, borderColor: color, color: "#fff" }
                      : { borderColor: color }
                  }
                  onCheckedChange={(next) => {
                    setHidden((current) =>
                      next ? current.filter((id) => id !== session.id) : [...current, session.id],
                    );
                  }}
                />
                <FieldContent className="min-w-0">
                  <FieldLabel htmlFor={inputId} className="w-full cursor-pointer truncate">
                    {session.id}
                  </FieldLabel>
                  {session.note === "" ? null : (
                    <FieldDescription className="truncate">{session.note}</FieldDescription>
                  )}
                </FieldContent>
              </Field>
            );
          })}
        </FieldSet>
        {sections.map((section) => (
          <FieldSet key={section.title} className="gap-2 rounded-lg border border-border p-3">
            <FieldLegend variant="label">{section.title}</FieldLegend>
            {section.slots.map((slot) => renderSlot(slot))}
          </FieldSet>
        ))}
        {(meta?.samples.length ?? 0) > 0 ? (
          <div className="flex flex-col gap-2">
            <Button
              variant="secondary"
              onClick={() => {
                if (meta != null) {
                  void playSamples(meta.samples);
                }
              }}
            >
              <Play />
              Play
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                if (meta == null) {
                  return;
                }
                const url = URL.createObjectURL(wavBlob(meta.samples));
                const link = document.createElement("a");
                link.href = url;
                link.download = "scan-kit.wav";
                link.click();
                URL.revokeObjectURL(url);
              }}
            >
              <Download />
              Export WAV
            </Button>
          </div>
        ) : null}
        {viewId === "dose_volume" ? (
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
                void invoke<{ report: string }>("scan_kit_open_study", { path: selected })
                  .then((opened) => {
                    setStudyPath(selected);
                    setStudy(opened.report);
                  })
                  .catch((reason: unknown) => notifyError(reason));
              });
            }}
          >
            Open Study
          </Button>
        ) : null}
        {viewId === "dose_volume" && studyPath != null ? (
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
        {study != null ? <pre className="text-muted-foreground text-xs whitespace-pre-wrap">{study}</pre> : null}
        </div>
        </>
      }
    />
  );
}
