import { useLayoutEffect, useRef, useState, type RefObject } from "react";
import type { DataEditorRef } from "@glideapps/glide-data-grid";

import { Checkbox } from "@/components/ui/checkbox";
import { sessionSwatch } from "@/session-colors";

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

/** Skip frames Glide has not laid out yet. Those bounds are NaN and break the overlay. */
export function checkboxFrame(
  bounds: { x: number; y: number; width: number; height: number } | undefined,
  origin: { left: number; top: number; right: number; bottom: number },
): { left: number; top: number; width: number; height: number } | null {
  if (bounds == null) {
    return null;
  }
  const { x, y, width, height } = bounds;
  if (![x, y, width, height, origin.left, origin.top].every(Number.isFinite) || width <= 0 || height <= 0) {
    return null;
  }
  const bottom = y + height;
  const right = x + width;
  if (bottom <= origin.top || y >= origin.bottom || right <= origin.left || x >= origin.right) {
    return null;
  }
  return {
    left: x - origin.left,
    top: y - origin.top,
    width,
    height,
  };
}

export function UseCheckLayer({
  boxes,
  order,
  header,
  onRow,
  onHeader,
}: {
  boxes: readonly CheckBox[];
  order: readonly string[];
  header: { checked: boolean; indeterminate: boolean };
  onRow: (sessionId: string, checked: boolean) => void;
  onHeader: (checked: boolean) => void;
}) {
  const selected = new Set(order);
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
            ) : box.sessionId != null ? (
              <SessionCheck
                sessionId={box.sessionId}
                order={order}
                checked={selected.has(box.sessionId)}
                onCheckedChange={(checked) => {
                  if (box.sessionId != null) {
                    onRow(box.sessionId, checked);
                  }
                }}
              />
            ) : null}
          </span>
        </div>
      ))}
    </div>
  );
}

function SessionCheck({
  sessionId,
  order,
  checked,
  onCheckedChange,
}: {
  sessionId: string;
  order: readonly string[];
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const swatch = sessionSwatch(order, sessionId);
  return (
    <Checkbox
      checked={checked}
      aria-label="Select row"
      title={swatch.label}
      style={
        checked
          ? { backgroundColor: swatch.color, borderColor: swatch.color, color: "#fff" }
          : undefined
      }
      onCheckedChange={onCheckedChange}
    />
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
  header,
  order,
}: {
  gridRef: RefObject<DataEditorRef | null>;
  hostRef: RefObject<HTMLElement | null>;
  sessionIds: readonly string[];
  region: { y: number; height: number };
  ready: boolean;
  onRow: (sessionId: string, checked: boolean) => void;
  onHeader: (checked: boolean) => void;
  header: { checked: boolean; indeterminate: boolean };
  order: readonly string[];
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
    const next: CheckBox[] = [];
    const headerBox = checkboxFrame(grid.getBounds(0, -1), origin);
    if (headerBox != null) {
      next.push({ key: "header", kind: "header", ...headerBox });
    }
    const start = Math.max(0, region.y);
    const end = Math.min(sessionIds.length, region.y + region.height);
    for (let row = start; row < end; row += 1) {
      const sessionId = sessionIds[row];
      const box = checkboxFrame(grid.getBounds(0, row), origin);
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
    <UseCheckLayer boxes={boxes} order={order} header={header} onRow={onRow} onHeader={onHeader} />
  );
}
