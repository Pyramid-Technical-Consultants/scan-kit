/** Seaborn "deep" palette. Index 0 is the first checked session. */
const SESSION_COLORS = [
  "#4C72B0",
  "#DD8452",
  "#55A868",
  "#C44E52",
  "#8172B3",
  "#937860",
  "#DA8BC3",
  "#8C8C8C",
] as const;

export const UNCHECKED_SWATCH = "#d0d0d0";

export function sessionColor(index: number): string {
  const count = SESSION_COLORS.length;
  const wrapped = ((index % count) + count) % count;
  return SESSION_COLORS[wrapped];
}

/** Drawn sessions, in selection order. A hidden session keeps its color for the others. */
export function shownSessionIds(order: readonly string[], hidden: readonly string[]): string[] {
  const dropped = new Set(hidden);
  return order.filter((id) => !dropped.has(id));
}

export function selectionFromLibrary(
  rows: readonly { session_id: string; selected: boolean }[],
  stored?: readonly string[],
): string[] {
  const present = rows.filter((row) => row.selected).map((row) => row.session_id);
  if (stored == null) {
    return present;
  }
  const known = new Set(present);
  return stored.filter((id) => known.has(id));
}

export function sessionSwatch(
  order: readonly string[],
  sessionId: string,
): { color: string; label: string } {
  const index = order.indexOf(sessionId);
  if (index < 0) {
    return {
      color: UNCHECKED_SWATCH,
      label: "Not used in plots — check to assign a plot color by order",
    };
  }
  const color = sessionColor(index);
  return {
    color,
    label: `Session color in plots: ${color} (${index + 1} of ${order.length})`,
  };
}
