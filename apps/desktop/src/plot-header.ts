export type ViewControl = {
  id: string;
  label: string;
  options: string[];
  value: string;
};

export type ViewTable = {
  columns: string[];
  rows: string[][];
};

export type PlotHeader = {
  title: string;
  controls: ViewControl[];
  table: ViewTable | null;
  samples: number[];
  panels: unknown[];
};

/** `scan_kit_open_plot` bytes: a little-endian `u32` JSON length, the JSON header, then marks. */
export function plotHeader(bytes: Uint8Array): PlotHeader {
  if (bytes.byteLength < 4) {
    throw new Error("plot payload is empty");
  }
  const length = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  if (bytes.byteLength < 4 + length) {
    throw new Error("plot payload is truncated");
  }
  return JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length))) as PlotHeader;
}

/** Canvas backing size in device pixels for a CSS box. */
export function backingSize(cssWidth: number, cssHeight: number, ratio: number): { width: number; height: number } {
  const scale = ratio > 0 ? ratio : 1;
  return {
    width: Math.max(16, Math.round(cssWidth * scale)),
    height: Math.max(16, Math.round(cssHeight * scale)),
  };
}
