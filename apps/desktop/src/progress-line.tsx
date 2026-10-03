import { hairlineFraction } from "@/task-client";

/// A hairline for a task report. Unknown length pulses. A fraction fills.
/// It does not take clicks. The parent is `relative`.
export function ProgressHairline({ done, total }: { done: number; total: number }) {
  const fraction = hairlineFraction(done, total);
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-0.5 overflow-hidden">
      {fraction == null ? (
        <div className="bg-primary h-full w-1/3 animate-pulse" />
      ) : (
        <div className="bg-primary h-full" style={{ width: `${fraction * 100}%` }} />
      )}
    </div>
  );
}
