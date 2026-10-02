import { getDefaultTheme, type Theme } from "@glideapps/glide-data-grid";

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
