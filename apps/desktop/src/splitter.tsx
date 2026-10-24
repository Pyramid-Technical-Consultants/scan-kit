import { useRef, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { GripHorizontal, GripVertical } from "lucide-react";
import { cn } from "cn";

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
  const Grip = vertical ? GripVertical : GripHorizontal;
  return (
    <div
      role="separator"
      aria-orientation={orientation}
      aria-label={label}
      aria-valuemin={min}
      aria-valuenow={Math.round(now)}
      tabIndex={0}
      className={cn(
        "group relative flex shrink-0 touch-none items-center justify-center outline-none",
        "focus-visible:ring-ring focus-visible:ring-1 focus-visible:ring-offset-1",
        vertical ? "w-2 cursor-col-resize" : "h-2 cursor-row-resize",
      )}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
    >
      <div
        className={cn(
          "bg-border transition-colors group-hover:bg-ring/50 group-active:bg-ring",
          vertical ? "h-full w-px" : "h-px w-full",
        )}
      />
      <div
        className={cn(
          "bg-background border-border text-muted-foreground absolute z-10 flex items-center justify-center rounded-sm border transition-colors",
          "group-hover:border-ring/50 group-hover:text-foreground group-active:border-ring",
          vertical ? "h-4 w-3" : "h-3 w-4",
        )}
      >
        <Grip className="size-2.5" />
      </div>
    </div>
  );
}
