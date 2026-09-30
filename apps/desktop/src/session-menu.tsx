import {
  DropdownMenu,
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
  onClose,
  onCopy,
  onTune,
}: {
  sessionId: string;
  x: number;
  y: number;
  onClose: () => void;
  onCopy: (sessionId: string) => void;
  onTune: () => void;
}) {
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
      <DropdownMenuContent>
        <DropdownMenuGroup>
          <DropdownMenuLabel>{sessionId}</DropdownMenuLabel>
          <DropdownMenuItem
            onClick={() => {
              onCopy(sessionId);
              onClose();
            }}
          >
            Copy Session ID
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => {
              onClose();
              onTune();
            }}
          >
            Open in Config Tuning…
          </DropdownMenuItem>
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
