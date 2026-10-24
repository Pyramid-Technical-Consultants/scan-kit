import { useRef, useState, type ReactNode } from "react";

import { Splitter } from "@/splitter";

const SIDE_MIN = 220;
const SIDE_DEFAULT = 420;
const CONTENT_MIN = 240;

function clampSide(parentWidth: number, next: number): number {
  const max = Math.max(SIDE_MIN, parentWidth - CONTENT_MIN);
  return Math.round(Math.min(max, Math.max(SIDE_MIN, next)));
}

export function SidePane({ main, side }: { main: ReactNode; side: ReactNode }) {
  const shell = useRef<HTMLDivElement>(null);
  const drag = useRef(SIDE_DEFAULT);
  const [sideWidth, setSideWidth] = useState(SIDE_DEFAULT);

  return (
    <div ref={shell} className="flex min-h-0 flex-1 overflow-hidden">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{main}</div>
      <Splitter
        orientation="vertical"
        label="Resize configuration"
        now={sideWidth}
        min={SIDE_MIN}
        onStart={() => {
          drag.current = sideWidth;
        }}
        onMove={(delta) => {
          const parent = shell.current?.getBoundingClientRect().width ?? window.innerWidth;
          setSideWidth(clampSide(parent, drag.current - delta));
        }}
      />
      <aside style={{ width: sideWidth }} className="flex min-h-0 shrink-0 flex-col">
        {side}
      </aside>
    </div>
  );
}
