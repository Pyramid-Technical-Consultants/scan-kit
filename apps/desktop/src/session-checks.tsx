import { useLayoutEffect, useRef, useState, type RefObject } from "react";
import type { DataEditorRef } from "@glideapps/glide-data-grid";

import { Checkbox } from "@/components/ui/checkbox";

export const MAX_SELECTED = 5;

export function headerCheck(
  rowCount: number,
  selectedCount: number,
): { checked: boolean; indeterminate: boolean } {
  if (selectedCount <= 0 || rowCount <= 0) {
    return { checked: false, indeterminate: false };
  }
  if (selectedCount >= rowCount) {
    return { checked: true, indeterminate: false };
  }
  return { checked: false, indeterminate: true };
}

/** A click on the header checkbox. Indeterminate clicks report checked, so the mixed state clears once the cap is full. */
export function headerWillFill(rowCount: number, selectedCount: number): boolean {
  if (rowCount === 0 || selectedCount === 0) {
    return true;
  }
  if (selectedCount >= rowCount) {
    return false;
  }
  if (rowCount > MAX_SELECTED && selectedCount >= MAX_SELECTED) {
    return false;
  }
  return true;
}

export function nextSessionSelection(
  idsInOrder: readonly string[],
  selected: readonly string[],
  target: string | "all",
  checked: boolean,
): { ids: string[]; capped: boolean } {
  if (target === "all") {
    if (!checked) {
      return { ids: [], capped: false };
    }
    return {
      ids: idsInOrder.slice(0, MAX_SELECTED),
      capped: idsInOrder.length > MAX_SELECTED,
    };
  }
  if (!checked) {
    return { ids: selected.filter((id) => id !== target), capped: false };
  }
  if (selected.includes(target)) {
    return { ids: [...selected], capped: false };
  }
  if (selected.length >= MAX_SELECTED) {
    return { ids: [...selected], capped: true };
  }
  return { ids: [...selected, target], capped: false };
}

type CheckBox = {
  key: string;
  kind: "header" | "row";
  sessionId?: string;
  left: number;
  top: number;
  width: number;
  height: number;
};

export function UseCheckLayer({
  boxes,
  selected,
  header,
  onRow,
  onHeader,
}: {
  boxes: readonly CheckBox[];
  selected: ReadonlySet<string>;
  header: { checked: boolean; indeterminate: boolean };
  onRow: (sessionId: string, checked: boolean) => void;
  onHeader: (checked: boolean) => void;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      {boxes.map((box) => (
        <div
          key={box.key}
          className="pointer-events-none absolute flex items-center justify-center"
          style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
        >
          <span className="pointer-events-auto">
            {box.kind === "header" ? (
              <Checkbox
                checked={header.checked}
                indeterminate={header.indeterminate}
                aria-label="Select all"
                onCheckedChange={(checked) => onHeader(checked)}
              />
            ) : (
              <Checkbox
                checked={box.sessionId != null && selected.has(box.sessionId)}
                aria-label="Select row"
                onCheckedChange={(checked) => {
                  if (box.sessionId != null) {
                    onRow(box.sessionId, checked);
                  }
                }}
              />
            )}
          </span>
        </div>
      ))}
    </div>
  );
}

export function UseColumnChecks({
  gridRef,
  hostRef,
  sessionIds,
  region,
  ready,
  onRow,
  onHeader,
  selected,
  header,
}: {
  gridRef: RefObject<DataEditorRef | null>;
  hostRef: RefObject<HTMLElement | null>;
  sessionIds: readonly string[];
  region: { y: number; height: number };
  ready: boolean;
  onRow: (sessionId: string, checked: boolean) => void;
  onHeader: (checked: boolean) => void;
  selected: ReadonlySet<string>;
  header: { checked: boolean; indeterminate: boolean };
}) {
  const [boxes, setBoxes] = useState<CheckBox[]>([]);
  const [pass, setPass] = useState(0);
  const stamp = useRef("");
  useLayoutEffect(() => {
    const grid = gridRef.current;
    const host = hostRef.current;
    if (!ready || grid == null || host == null) {
      return;
    }
    const origin = host.getBoundingClientRect();
    const place = (bounds: { x: number; y: number; width: number; height: number } | undefined) => {
      if (bounds == null) {
        return null;
      }
      const bottom = bounds.y + bounds.height;
      const right = bounds.x + bounds.width;
      if (bottom <= origin.top || bounds.y >= origin.bottom || right <= origin.left || bounds.x >= origin.right) {
        return null;
      }
      return {
        left: bounds.x - origin.left,
        top: bounds.y - origin.top,
        width: bounds.width,
        height: bounds.height,
      };
    };
    const next: CheckBox[] = [];
    const headerBox = place(grid.getBounds(0, -1));
    if (headerBox != null) {
      next.push({ key: "header", kind: "header", ...headerBox });
    }
    const start = Math.max(0, region.y);
    const end = Math.min(sessionIds.length, region.y + region.height);
    for (let row = start; row < end; row += 1) {
      const sessionId = sessionIds[row];
      const box = place(grid.getBounds(0, row));
      if (sessionId == null || box == null) {
        continue;
      }
      next.push({ key: sessionId, kind: "row", sessionId, ...box });
    }
    const nextStamp = next.map((box) => `${box.key}:${box.left}:${box.top}`).join(";");
    if (nextStamp !== stamp.current) {
      stamp.current = nextStamp;
      setBoxes(next);
    }
    if (headerBox == null && pass < 4) {
      const frame = requestAnimationFrame(() => setPass((value) => value + 1));
      return () => cancelAnimationFrame(frame);
    }
    return undefined;
  }, [gridRef, hostRef, pass, ready, sessionIds, region]);
  return (
    <UseCheckLayer boxes={boxes} selected={selected} header={header} onRow={onRow} onHeader={onHeader} />
  );
}
