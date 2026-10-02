import { useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

const SIDE_MIN = 220;
const SIDE_DEFAULT = 420;
const CONTENT_MIN = 240;

function clampSide(parentWidth: number, next: number): number {
  const max = Math.max(SIDE_MIN, parentWidth - CONTENT_MIN);
  return Math.round(Math.min(max, Math.max(SIDE_MIN, next)));
}

export function SidePane({ main, side }: { main: ReactNode; side: ReactNode }) {
  const shell = useRef<HTMLDivElement>(null);
  const [sideWidth, setSideWidth] = useState(SIDE_DEFAULT);

  const resizeSide = (next: number) => {
    const parent = shell.current?.getBoundingClientRect().width ?? window.innerWidth;
    setSideWidth(clampSide(parent, next));
  };

  const onSplitterDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const startX = event.clientX;
    const startWidth = sideWidth;
    const onMove = (ev: PointerEvent) => {
      const parent = shell.current?.getBoundingClientRect().width ?? window.innerWidth;
      setSideWidth(clampSide(parent, startWidth + (startX - ev.clientX)));
    };
    const onUp = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  };

  return (
    <div ref={shell} className="flex min-h-0 flex-1 overflow-hidden">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{main}</div>
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize configuration"
        aria-valuemin={SIDE_MIN}
        aria-valuenow={sideWidth}
        tabIndex={0}
        className="bg-border w-1.5 shrink-0 cursor-col-resize touch-none focus-visible:bg-ring"
        onPointerDown={onSplitterDown}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            resizeSide(sideWidth + 16);
          } else if (event.key === "ArrowRight") {
            resizeSide(sideWidth - 16);
          }
        }}
      />
      <aside style={{ width: sideWidth }} className="flex min-h-0 shrink-0 flex-col">
        {side}
      </aside>
    </div>
  );
}
