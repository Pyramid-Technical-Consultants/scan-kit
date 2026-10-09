import { useEffect, useRef, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { backingSize, type ViewControl } from "@/plot-header";
import { Splitter } from "@/splitter";

type Plotter = import("@/wasm/scan_kit_plot.js").WebPlot;

const CELLS = [
  { id: "cell0", panel: 0, kind: "cell" },
  { id: "cell1", panel: 1, kind: "cell" },
  { id: "cell2", panel: 2, kind: "cell" },
  { id: "cell3", panel: 3, kind: "cell" },
  { id: "plot0", panel: 4, kind: "plot" },
  { id: "plot1", panel: 5, kind: "plot" },
] as const;

const ROW_WEIGHTS = [1.35, 1.35, 1];
const COL_MIN = 0.18;
const COL_MAX = 0.82;

function DoseCell({
  panel,
  grow,
  shown,
  toolbar,
  plots,
  acquire,
  onReady,
  onInput,
  onError,
}: {
  panel: number;
  grow: number;
  shown: boolean;
  toolbar: ReactNode;
  plots: { current: (Plotter | null)[] };
  acquire: (node: HTMLCanvasElement) => Promise<Plotter>;
  onReady: (panel: number) => void;
  onInput: (panel: number) => void;
  onError: (reason: unknown) => void;
}) {
  const nodeRef = useRef<HTMLCanvasElement>(null);
  const readyRef = useRef(onReady);
  const inputRef = useRef(onInput);
  const errorRef = useRef(onError);
  useEffect(() => {
    readyRef.current = onReady;
    inputRef.current = onInput;
    errorRef.current = onError;
  }, [onReady, onInput, onError]);

  useEffect(() => {
    const node = nodeRef.current;
    if (node == null || !shown) {
      return;
    }
    const slots = plots.current;
    let live = true;
    let opening = false;
    const fit = () => {
      const rect = node.getBoundingClientRect();
      // A surface opened on a 0×0 canvas, then resized, never presents a frame.
      if (rect.width < 1 || rect.height < 1) {
        return;
      }
      const next = backingSize(rect.width, rect.height, window.devicePixelRatio);
      const plot = plots.current[panel];
      if (plot == null) {
        if (opening) {
          return;
        }
        // Set the drawing buffer before the context exists. Assigning it later
        // drops the only WebGL context this canvas will get.
        if (node.width !== next.width || node.height !== next.height) {
          node.width = next.width;
          node.height = next.height;
        }
        opening = true;
        acquire(node)
          .then((created) => {
            if (!live) {
              return;
            }
            const box = node.getBoundingClientRect();
            const sized =
              box.width < 1 || box.height < 1
                ? next
                : backingSize(box.width, box.height, window.devicePixelRatio);
            created.solo(panel);
            created.resize(sized.width, sized.height);
            plots.current[panel] = created;
            readyRef.current(panel);
          })
          .catch((reason: unknown) => errorRef.current(reason));
        return;
      }
      plot.resize(next.width, next.height);
      inputRef.current(panel);
    };
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
      const plot = plots.current[panel];
      if (plot == null) {
        return;
      }
      event.preventDefault();
      const point = locate(event);
      plot.zoom(point.x, point.y, event.deltaY);
      inputRef.current(panel);
    };
    let dragging = false;
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
      const plot = plots.current[panel];
      if (plot == null) {
        return;
      }
      const point = locate(event);
      plot.pan(
        point.x,
        point.y,
        event.movementX * point.sx,
        event.movementY * point.sy,
        event.buttons,
        event.shiftKey,
      );
      inputRef.current(panel);
    };
    const onDouble = () => {
      plots.current[panel]?.reset();
      inputRef.current(panel);
    };
    const onKey = (event: KeyboardEvent) => {
      const used = plots.current[panel]?.dose_key(event.key, event.ctrlKey) ?? false;
      if (!used) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      inputRef.current(panel);
    };
    const onMenu = (event: Event) => {
      event.preventDefault();
    };
    const observer = new ResizeObserver(fit);
    observer.observe(node);
    fit();
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("pointerdown", onDown);
    node.addEventListener("pointerup", onUp);
    node.addEventListener("pointercancel", onUp);
    node.addEventListener("pointermove", onMove);
    node.addEventListener("dblclick", onDouble);
    node.addEventListener("keydown", onKey);
    node.addEventListener("contextmenu", onMenu);
    return () => {
      live = false;
      slots[panel] = null;
      observer.disconnect();
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("pointerdown", onDown);
      node.removeEventListener("pointerup", onUp);
      node.removeEventListener("pointercancel", onUp);
      node.removeEventListener("pointermove", onMove);
      node.removeEventListener("dblclick", onDouble);
      node.removeEventListener("keydown", onKey);
      node.removeEventListener("contextmenu", onMenu);
    };
  }, [shown, panel, plots, acquire]);

  return (
    <div style={{ flex: `${grow} 1 0px` }} className="flex min-h-0 min-w-0 flex-col">
      {toolbar}
      <div className="relative min-h-0 flex-1">
        <canvas ref={nodeRef} tabIndex={0} className="absolute inset-0 h-full w-full touch-none" />
      </div>
    </div>
  );
}

export function DoseBoard({
  shown,
  controls,
  resolved,
  plots,
  acquire,
  onChange,
  onAction,
  onReady,
  onInput,
  onError,
  renderChoice,
}: {
  shown: boolean;
  controls: readonly ViewControl[];
  resolved: Readonly<Record<string, string>>;
  plots: { current: (Plotter | null)[] };
  acquire: (node: HTMLCanvasElement) => Promise<Plotter>;
  onChange: (id: string, value: string) => void;
  onAction: (panel: number, action: string) => void;
  onReady: (panel: number) => void;
  onInput: (panel: number) => void;
  onError: (reason: unknown) => void;
  renderChoice: (control: ViewControl, value: string, onChange: (value: string) => void) => ReactNode;
}) {
  const shell = useRef<HTMLDivElement>(null);
  const rowNodes = useRef<(HTMLDivElement | null)[]>([]);
  const gesture = useRef({ rows: ROW_WEIGHTS, pair: 1, index: 0, col: 0.5 });
  const [rows, setRows] = useState(ROW_WEIGHTS);
  const [col, setCol] = useState(0.5);

  return (
    <div
      ref={shell}
      className={shown ? "flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden" : "hidden"}
    >
      {rows.map((weight, row) => (
        <div key={row} className="contents">
          {row > 0 ? (
            <Splitter
              orientation="horizontal"
              label="Resize row"
              now={(rows[row - 1] / Math.max(1e-6, rows[row - 1] + weight)) * 100}
              min={15}
              onStart={() => {
                const above = rowNodes.current[row - 1]?.getBoundingClientRect().height ?? 1;
                const below = rowNodes.current[row]?.getBoundingClientRect().height ?? 1;
                gesture.current = {
                  rows: [...rows],
                  pair: Math.max(1, above + below),
                  index: row - 1,
                  col,
                };
              }}
              onMove={(delta) => {
                const start = gesture.current.rows;
                const index = gesture.current.index;
                const pair = start[index] + start[index + 1];
                const minShare = Math.min(0.85, 72 / gesture.current.pair);
                const share = Math.min(
                  1 - minShare,
                  Math.max(minShare, start[index] / pair + delta / gesture.current.pair),
                );
                const next = [...start];
                next[index] = pair * share;
                next[index + 1] = pair * (1 - share);
                setRows(next);
              }}
            />
          ) : null}
          <div
            ref={(node) => {
              rowNodes.current[row] = node;
            }}
            style={{ flex: `${weight} 1 0px` }}
            className="flex min-h-0 min-w-0"
          >
            {[0, 1].map((column) => {
              const item = CELLS[row * 2 + column];
              const control = controls.find((entry) => entry.id === item.id);
              const value = control == null ? "" : (resolved[item.id] ?? control.value);
              const slice = item.kind === "cell" && value !== "3D";
              const profile =
                item.kind === "plot" && value !== "DVH" && !value.toLowerCase().startsWith("gamma");
              return (
                <div key={item.id} className="contents">
                  {column === 1 ? (
                    <Splitter
                      orientation="vertical"
                      label="Resize column"
                      now={col * 100}
                      min={18}
                      onStart={() => {
                        gesture.current.col = col;
                      }}
                      onMove={(delta) => {
                        const width = Math.max(1, (shell.current?.clientWidth ?? 1) - 18);
                        const next = gesture.current.col + delta / width;
                        setCol(Math.min(COL_MAX, Math.max(COL_MIN, next)));
                      }}
                    />
                  ) : null}
                  <DoseCell
                    panel={item.panel}
                    grow={column === 0 ? col : 1 - col}
                    shown={shown}
                    plots={plots}
                    acquire={acquire}
                    onReady={onReady}
                    onInput={onInput}
                    onError={onError}
                    toolbar={
                      control == null ? null : (
                        <div className="flex shrink-0 items-center justify-end gap-1 border-b border-border px-1 py-1">
                          {slice ? (
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              aria-label="Rotate 90 degrees"
                              onClick={() => onAction(item.panel, "rotate")}
                            >
                              90°
                            </Button>
                          ) : null}
                          {slice || profile ? (
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              aria-label="Integral"
                              onClick={() => onAction(item.panel, "integral")}
                            >
                              ∫
                            </Button>
                          ) : null}
                          {renderChoice(control, value, (next) => onChange(item.id, next))}
                        </div>
                      )
                    }
                  />
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
