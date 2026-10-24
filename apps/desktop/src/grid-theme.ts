import { useCallback, useMemo, useState } from "react";
import { getDefaultTheme, type GridColumn, type Theme } from "@glideapps/glide-data-grid";

export function tokenColor(name: string, percent?: number): string {
  const probe = document.createElement("span");
  probe.style.color =
    percent == null
      ? `var(${name})`
      : `color-mix(in oklch, var(${name}) ${percent}%, transparent)`;
  document.body.append(probe);
  const resolved = getComputedStyle(probe).color;
  probe.remove();
  return resolved;
}

export type CellTone = "good" | "warn" | "bad" | "change";

// Verdict hues have no theme tokens; only red comes from --destructive. The
// background stays a faint wash so the cell reads as tinted, and the text
// carries the verdict color.
export function toneColors(): Record<CellTone, { bg: string; fg: string }> {
  return {
    good: { bg: "rgba(74, 222, 128, 0.08)", fg: "#4ade80" },
    warn: { bg: "rgba(251, 191, 36, 0.08)", fg: "#fbbf24" },
    bad: { bg: tokenColor("--destructive", 10), fg: tokenColor("--destructive") },
    change: { bg: "rgba(147, 197, 253, 0.08)", fg: "#93c5fd" },
  };
}

// Column resize overrides keyed by column id/title so the widths survive
// column-array rebuilds (sort arrows, filtering, new preview results).
export function useResizableColumns(columns: GridColumn[]): {
  columns: GridColumn[];
  onColumnResize: (column: GridColumn, newSize: number, colIndex: number) => void;
} {
  const [widths, setWidths] = useState<Record<string, number>>({});
  const sized = useMemo(
    () =>
      columns.map((column, index) => {
        const width = widths[column.id ?? column.title ?? String(index)];
        return width == null ? column : { ...column, width };
      }),
    [columns, widths],
  );
  const onColumnResize = useCallback(
    (column: GridColumn, newSize: number, colIndex: number) => {
      const key = column.id ?? columns[colIndex]?.id ?? column.title ?? String(colIndex);
      setWidths((current) => ({ ...current, [key]: newSize }));
    },
    [columns],
  );
  return { columns: sized, onColumnResize };
}

export function gridTheme(): Theme {
  const base = getDefaultTheme();
  const foreground = tokenColor("--foreground");
  const muted = tokenColor("--muted-foreground");
  const card = tokenColor("--card");
  const accent = tokenColor("--accent");
  const border = tokenColor("--border");
  const wash = tokenColor("--muted");
  return {
    ...base,
    accentColor: foreground,
    accentFg: tokenColor("--background"),
    accentLight: tokenColor("--foreground", 16),
    textDark: foreground,
    textMedium: muted,
    textLight: muted,
    textBubble: foreground,
    textHeader: foreground,
    textHeaderSelected: tokenColor("--background"),
    bgIconHeader: card,
    fgIconHeader: foreground,
    bgCell: tokenColor("--background"),
    bgCellMedium: card,
    bgHeader: card,
    bgHeaderHasFocus: wash,
    bgHeaderHovered: wash,
    bgBubble: card,
    bgBubbleSelected: accent,
    bgSearchResult: wash,
    borderColor: border,
    horizontalBorderColor: border,
    drilldownBorder: border,
    linkColor: accent,
    fontFamily: getComputedStyle(document.documentElement).fontFamily,
    checkboxMaxSize: 16,
    roundingRadius: 4,
  };
}
