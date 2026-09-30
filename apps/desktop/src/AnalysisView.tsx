import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
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
import { Field, FieldLabel } from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export type ViewControl = {
  id: string;
  label: string;
  options: string[];
  value: string;
};

type ViewTable = {
  columns: string[];
  rows: string[][];
};

type ViewFrame = {
  title: string;
  width: number;
  height: number;
  rgba_base64: string;
  controls: ViewControl[];
  table: ViewTable | null;
  samples: number[];
};

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

function palette(): number[][] {
  return ["--chart-1", "--chart-2", "--chart-3", "--chart-4", "--chart-5"].map((name) =>
    parseColor(tokenColor(name)),
  );
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
  const request = useRef(0);
  const [options, setOptions] = useState<Record<string, string>>({});
  const [frame, setFrame] = useState<ViewFrame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [study, setStudy] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 960, height: 640 });
  const plotWidth = Math.max(16, Math.round(size.width * (window.devicePixelRatio || 1)));
  const plotHeight = Math.max(16, Math.round(size.height * (window.devicePixelRatio || 1)));

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
    const id = request.current + 1;
    request.current = id;
    const timer = window.setTimeout(() => {
      void invoke<ViewFrame>("scan_kit_run_view", {
        view: viewId,
        path: folder,
        sessionIds,
        options,
        width: plotWidth,
        height: plotHeight,
        background: parseColor(tokenColor("--background")),
        foreground: parseColor(tokenColor("--foreground")),
        palette: palette(),
      })
        .then((next) => {
          if (request.current === id) {
            setFrame(next);
            setError(null);
          }
        })
        .catch((reason: unknown) => {
          if (request.current === id) {
            setError(reason instanceof Error ? reason.message : String(reason));
          }
        });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [viewId, folder, sessionIds, options, plotWidth, plotHeight]);

  useEffect(() => {
    const node = canvas.current;
    if (node == null || frame == null || frame.width === 0 || frame.rgba_base64 === "") {
      return;
    }
    const bytes = Uint8Array.from(atob(frame.rgba_base64), (char) => char.charCodeAt(0));
    if (bytes.length < frame.width * frame.height * 4) {
      return;
    }
    const context = node.getContext("2d");
    if (context == null) {
      return;
    }
    node.width = frame.width;
    node.height = frame.height;
    context.putImageData(new ImageData(new Uint8ClampedArray(bytes), frame.width, frame.height), 0, 0);
  }, [frame]);

  const table = frame?.table;
  const columns: GridColumn[] =
    table?.columns.map((title) => ({ title, width: 180 })) ?? [];
  const getCellContent = ([col, row]: Item): GridCell => ({
    kind: GridCellKind.Text,
    data: table?.rows[row]?.[col] ?? "",
    displayData: table?.rows[row]?.[col] ?? "",
    allowOverlay: false,
  });

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-56 shrink-0 flex-col gap-3 border-r border-border p-3">
        <Button variant="outline" onClick={onBack}>
          Sessions
        </Button>
        <h2 className="text-sm font-medium">{frame?.title ?? "Analysis"}</h2>
        {(frame?.controls ?? []).map((control) => {
          const items = control.options.map((option) => ({ label: option, value: option }));
          return (
            <Field key={control.id}>
              <FieldLabel>{control.label}</FieldLabel>
              <Select
                items={items}
                value={options[control.id] ?? control.value}
                onValueChange={(value) => {
                  if (value == null) {
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
        {(frame?.samples.length ?? 0) > 0 ? (
          <div className="flex flex-col gap-2">
            <Button
              variant="secondary"
              onClick={() => {
                if (frame != null) {
                  void playSamples(frame.samples);
                }
              }}
            >
              Play
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                if (frame == null) {
                  return;
                }
                const url = URL.createObjectURL(wavBlob(frame.samples));
                const link = document.createElement("a");
                link.href = url;
                link.download = "scan-kit.wav";
                link.click();
                URL.revokeObjectURL(url);
              }}
            >
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
                  .catch((reason: unknown) =>
                    setError(reason instanceof Error ? reason.message : String(reason)),
                  );
              });
            }}
          >
            Open study
          </Button>
        ) : null}
        {study != null ? <pre className="text-muted-foreground text-xs whitespace-pre-wrap">{study}</pre> : null}
        {error != null ? <p className="text-destructive text-sm">{error}</p> : null}
      </aside>
      <div ref={host} className="bg-background min-h-0 flex-1">
        {table != null && table.rows.length > 0 ? (
          <DataEditor
            width={size.width}
            height={frame != null && frame.width > 0 ? Math.min(240, size.height) : size.height}
            columns={columns}
            rows={table.rows.length}
            getCellContent={getCellContent}
            theme={gridTheme()}
            rowMarkers="none"
          />
        ) : null}
        <canvas
          ref={canvas}
          className={frame != null && frame.width > 0 ? "h-full w-full" : "hidden"}
        />
      </div>
    </div>
  );
}
