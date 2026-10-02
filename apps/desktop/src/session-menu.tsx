import { Copy, SlidersHorizontal } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

/** Glide cell bounds are already viewport coordinates. `localEvent` is the click inside that cell. */
export function sessionMenuPoint(
  bounds: { x: number; y: number },
  localEventX: number,
  localEventY: number,
): { x: number; y: number } {
  return { x: bounds.x + localEventX, y: bounds.y + localEventY };
}

export function SessionContextMenu({
  sessionId,
  x,
  y,
  rowIds,
  rowsSelected,
  onClose,
  onCopy,
  onTune,
  onToggleRows,
}: {
  sessionId: string;
  x: number;
  y: number;
  rowIds: readonly string[];
  rowsSelected: boolean;
  onClose: () => void;
  onCopy: (sessionId: string) => void;
  onTune: () => void;
  onToggleRows: () => void;
}) {
  const rowLabel = rowIds.length === 1 ? "Row" : "Rows";
  return (
    <DropdownMenu
      open
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <DropdownMenuTrigger
        className="size-px p-0 opacity-0"
        style={{ position: "fixed", left: x, top: y }}
      />
      <DropdownMenuContent className="w-max">
        <DropdownMenuGroup>
          <DropdownMenuLabel className="whitespace-nowrap">{sessionId}</DropdownMenuLabel>
          <DropdownMenuCheckboxItem
            className="whitespace-nowrap"
            checked={rowsSelected}
            onCheckedChange={() => {
              onToggleRows();
              onClose();
            }}
          >
            {rowsSelected ? "Deselect" : "Select"} {rowLabel}
          </DropdownMenuCheckboxItem>
          <DropdownMenuItem
            className="whitespace-nowrap"
            onClick={() => {
              onCopy(sessionId);
              onClose();
            }}
          >
            <Copy />
            Copy Session ID
          </DropdownMenuItem>
          <DropdownMenuItem
            className="whitespace-nowrap"
            onClick={() => {
              onClose();
              onTune();
            }}
          >
            <SlidersHorizontal />
            Open in Config Tuning…
          </DropdownMenuItem>
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
