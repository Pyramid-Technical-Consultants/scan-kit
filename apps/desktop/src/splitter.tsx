import { useRef, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";

export function Splitter({
  orientation,
  label,
  now,
  min,
  onStart,
  onMove,
}: {
  orientation: "vertical" | "horizontal";
  label: string;
  now: number;
  min: number;
  onStart: () => void;
  /** Pointer travel from the drag start, in CSS pixels. Positive is right or down. */
  onMove: (delta: number) => void;
}) {
  const origin = useRef(0);
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    origin.current = orientation === "vertical" ? event.clientX : event.clientY;
    onStart();
    const onPointerMove = (ev: PointerEvent) => {
      const at = orientation === "vertical" ? ev.clientX : ev.clientY;
      onMove(at - origin.current);
    };
    const onUp = () => {
      handle.removeEventListener("pointermove", onPointerMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step =
      event.key === "ArrowRight" || event.key === "ArrowDown"
        ? 16
        : event.key === "ArrowLeft" || event.key === "ArrowUp"
          ? -16
          : 0;
    if (step === 0) {
      return;
    }
    event.preventDefault();
    onStart();
    onMove(step);
  };
  const vertical = orientation === "vertical";
  return (
    <div
      role="separator"
      aria-orientation={orientation}
      aria-label={label}
      aria-valuemin={min}
      aria-valuenow={Math.round(now)}
      tabIndex={0}
      className={
        vertical
          ? "bg-border w-1.5 shrink-0 cursor-col-resize touch-none focus-visible:bg-ring"
          : "bg-border h-1.5 shrink-0 cursor-row-resize touch-none focus-visible:bg-ring"
      }
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
    />
  );
}
