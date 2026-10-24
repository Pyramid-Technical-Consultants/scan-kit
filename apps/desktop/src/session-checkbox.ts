/** Painted to match the shadcn checkbox: 16px, 4px corners, input border, colored fill. */
const CHECK_SIZE = 16;
const CHECK_RADIUS = 4;

export type CheckPaint = {
  border: string;
  idle: string;
  mark: string;
  header: string;
};

export function checkboxRect(cell: { x: number; y: number; width: number; height: number }): {
  x: number;
  y: number;
  size: number;
} {
  return {
    x: cell.x + (cell.width - CHECK_SIZE) / 2,
    y: cell.y + (cell.height - CHECK_SIZE) / 2,
    size: CHECK_SIZE,
  };
}

export function drawSessionCheckbox(
  ctx: CanvasRenderingContext2D,
  cell: { x: number; y: number; width: number; height: number },
  mode: "on" | "off" | "mixed",
  color: string,
  paint: CheckPaint,
): void {
  const box = checkboxRect(cell);
  const fill = mode === "off" ? paint.idle : mode === "mixed" ? paint.header : color;
  const stroke = mode === "off" ? paint.border : fill;
  ctx.beginPath();
  ctx.roundRect(box.x + 0.5, box.y + 0.5, box.size - 1, box.size - 1, CHECK_RADIUS);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  if (mode === "off") {
    return;
  }
  ctx.beginPath();
  ctx.strokeStyle = paint.mark;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (mode === "mixed") {
    ctx.lineWidth = 1.5;
    ctx.moveTo(box.x + 4, box.y + box.size / 2);
    ctx.lineTo(box.x + box.size - 4, box.y + box.size / 2);
  } else {
    // Lucide check, scaled into the 14px icon the shadcn checkbox uses.
    const scale = 14 / 24;
    const originX = box.x + 1;
    const originY = box.y + 1;
    ctx.lineWidth = 2 * scale;
    ctx.moveTo(originX + 20 * scale, originY + 6 * scale);
    ctx.lineTo(originX + 9 * scale, originY + 17 * scale);
    ctx.lineTo(originX + 4 * scale, originY + 12 * scale);
  }
  ctx.stroke();
}
