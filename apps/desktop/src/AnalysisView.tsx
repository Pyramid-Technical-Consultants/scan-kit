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

import { Button } from "@/components/ui/button";
import { backingSize, plotHeader, type PlotHeader } from "@/plot-header";
import { sessionColor } from "@/session-colors";
import { dismissNotice, notifyError } from "@/notify";
import { Field, FieldLabel } from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const BINNED_PRESETS: Record<string, Record<string, string>> = {
  "Dose error vs Energy": {
    metric: "Dose Error (%)",
    x: "Energy",
    glyph: "Box",
    trend: "On",
    hist: "On",
    corr: "On",
  },
  "Dose error mean vs Energy": {
    metric: "Dose Error (%)",
    x: "Energy",
    glyph: "Mean",
    trend: "On",
    hist: "On",
    corr: "On",
  },
  "Dose ratios vs Energy": {
    metric: "Dose Ratios",
    x: "Energy",
    glyph: "Box",
    trend: "On",
    corr: "On",
  },
  "Dose rate vs Energy": {
    metric: "Dose Rate (MU/s)",
    x: "Energy",
    glyph: "Mean",
    trend: "On",
  },
  "Current ratios vs Energy": {
    metric: "Current Ratios (%)",
    x: "Energy",
    glyph: "Mean",
    trend: "On",
  },
  "IC current vs Energy": {
    metric: "IC Current (nA)",
    x: "Energy",
    glyph: "Box",
    trend: "On",
  },
  "Position error vs Energy": {
    metric: "Position Error (mm)",
    x: "Energy",
    glyph: "Violin",
    trend: "Off",
  },
  "Sigma vs Energy": {
    metric: "Sigma (mm)",
    x: "Energy",
    glyph: "Violin",
    trend: "Off",
  },
  "IC2-IC1 position vs Energy": {
    metric: "IC2-IC1 Position (mm)",
    x: "Energy",
    glyph: "Violin",
    trend: "Off",
  },
  "Spot time vs Energy": {
    metric: "Spot Delivery Time",
    x: "Energy",
    glyph: "Box",
    trend: "On",
  },
  "Dose error vs Target MU": {
    metric: "Dose Error (%)",
    x: "Target MU",
    glyph: "Box",
    trend: "On",
    hist: "On",
    corr: "On",
  },
  "Dose ratios vs Spot time": {
    metric: "Dose Ratios",
    x: "Spot time",
    glyph: "Box",
    trend: "On",
    corr: "On",
  },
  "Dose ratios vs Beam radius": {
    metric: "Dose Ratios",
    x: "Radius",
    glyph: "Box",
    trend: "On",
    corr: "On",
  },
};

type Readout = {
  x: number;
  y: number;
  series: number | null;
};

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

function palette(sessionIds: readonly string[]): number[][] {
  const count = Math.max(sessionIds.length, 1);
  return Array.from({ length: count }, (_, index) => parseColor(sessionColor(index)));
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

export function AnalysisView({
  viewId,
  folder,
  sessionIds,
  onBack,
}: {
  viewId: string;
  folder: string;
  sessionIds: string[];
  onBack: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const plotter = useRef<Plotter | null>(null);
  const payload = useRef<Uint8Array | null>(null);
  const frame = useRef(0);
  const openSeq = useRef(0);
  const hoverNode = useRef<HTMLParagraphElement>(null);
  const [options, setOptions] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState<PlotHeader | null>(null);
  const [study, setStudy] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 960, height: 640 });
  const shown = (meta?.panels.length ?? 0) > 0;

  // The helpers below only read refs, so the listeners registered once keep working.
  const showReadout = (next: Readout | null) => {
    const node = hoverNode.current;
    if (node == null) {
      return;
    }
    if (next == null) {
      node.hidden = true;
      return;
    }
    node.hidden = false;
    const series = next.series == null ? "" : `  #${next.series + 1}`;
    node.textContent = `${next.x.toPrecision(4)}, ${next.y.toPrecision(4)}${series}`;
  };

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

  const readHover = (x: number, y: number) => {
    const hit = plotter.current?.hover(x, y);
    showReadout(hit == null ? null : { x: hit.x, y: hit.y, series: hit.series ?? null });
    hit?.free();
  };

  const fitCanvas = () => {
    const node = canvas.current;
    if (node == null) {
      return;
    }
    const rect = node.getBoundingClientRect();
    const next = backingSize(rect.width, rect.height, window.devicePixelRatio);
    if (node.width !== next.width || node.height !== next.height) {
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
    if (node == null) {
      return;
    }
    let live = true;
    const observer = new ResizeObserver(fitCanvas);
    observer.observe(node);
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
  }, []);

  useEffect(() => {
    const ticket = openSeq.current + 1;
    openSeq.current = ticket;
    const timer = window.setTimeout(() => {
      void invoke<ArrayBuffer | Uint8Array>("scan_kit_open_plot", {
        view: viewId,
        path: folder,
        sessionIds,
        options,
        background: parseColor(tokenColor("--background")),
        foreground: parseColor(tokenColor("--foreground")),
        palette: palette(sessionIds),
      })
        .then((result) => {
          if (openSeq.current !== ticket) {
            return;
          }
          const bytes = result instanceof Uint8Array ? result : new Uint8Array(result);
          setMeta(plotHeader(bytes));
          payload.current = bytes;
          loadPayload();
          showReadout(null);
          dismissNotice("analysis");
        })
        .catch((reason: unknown) => {
          if (openSeq.current === ticket) {
            notifyError(messageOf(reason), "analysis");
          }
        });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [viewId, folder, sessionIds, options]);

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
      readHover(point.x, point.y);
    };
    const onDown = (event: PointerEvent) => {
      dragging = true;
      node.setPointerCapture(event.pointerId);
    };
    const onUp = () => {
      dragging = false;
    };
    const onMove = (event: PointerEvent) => {
      const point = locate(event);
      if (dragging) {
        plotter.current?.pan(point.x, point.y, event.movementX * point.sx, event.movementY * point.sy);
        requestDraw();
      }
      readHover(point.x, point.y);
    };
    const onLeave = () => showReadout(null);
    const onDouble = () => {
      plotter.current?.reset();
      requestDraw();
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("pointerdown", onDown);
    node.addEventListener("pointerup", onUp);
    node.addEventListener("pointercancel", onUp);
    node.addEventListener("pointermove", onMove);
    node.addEventListener("pointerleave", onLeave);
    node.addEventListener("dblclick", onDouble);
    return () => {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("pointerdown", onDown);
      node.removeEventListener("pointerup", onUp);
      node.removeEventListener("pointercancel", onUp);
      node.removeEventListener("pointermove", onMove);
      node.removeEventListener("pointerleave", onLeave);
      node.removeEventListener("dblclick", onDouble);
    };
  }, []);

  const table = meta?.table;
  const columns: GridColumn[] =
    table?.columns.map((title) => ({ title, width: 180 })) ?? [];
  const getCellContent = ([col, row]: Item): GridCell => ({
    kind: GridCellKind.Text,
    data: table?.rows[row]?.[col] ?? "",
    displayData: table?.rows[row]?.[col] ?? "",
    allowOverlay: false,
  });

  return (
    <div className="flex min-h-0 flex-1 overflow-hidden">
      <aside className="flex min-h-0 w-56 shrink-0 flex-col gap-3 overflow-y-auto border-r border-border p-3">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft />
          Sessions
        </Button>
        <h2 className="text-sm font-medium">{meta?.title ?? "Analysis"}</h2>
        {(meta?.controls ?? []).map((control) => {
          const items = control.options.map((option) => ({ label: option, value: option }));
          const stored = options[control.id];
          const value = stored != null && control.options.includes(stored) ? stored : control.value;
          return (
            <Field key={control.id}>
              <FieldLabel>{control.label}</FieldLabel>
              <Select
                items={items}
                value={value}
                onValueChange={(value) => {
                  if (value == null) {
                    return;
                  }
                  if (control.id === "preset") {
                    const preset = BINNED_PRESETS[value];
                    setOptions((current) => ({
                      ...current,
                      hist: "Off",
                      corr: "Off",
                      trend: "On",
                      ...preset,
                      preset: value,
                    }));
                    return;
                  }
                  setOptions((current) => ({ ...current, [control.id]: value }));
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {items.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
          );
        })}
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
              void open({ directory: true, title: "DICOM study" }).then((selected) => {
                if (typeof selected !== "string") {
                  return;
                }
                void invoke<{ report: string }>("scan_kit_open_study", { path: selected })
                  .then((opened) => setStudy(opened.report))
                  .catch((reason: unknown) => notifyError(reason));
              });
            }}
          >
            Open study
          </Button>
        ) : null}
        {study != null ? <pre className="text-muted-foreground text-xs whitespace-pre-wrap">{study}</pre> : null}
      </aside>
      <div ref={host} className="bg-background relative min-h-0 flex-1">
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
        <canvas ref={canvas} className={shown ? "h-full w-full touch-none" : "hidden"} />
        <p
          ref={hoverNode}
          hidden
          className="text-muted-foreground pointer-events-none absolute bottom-2 left-2 text-xs tabular-nums"
        />
      </div>
    </div>
  );
}
