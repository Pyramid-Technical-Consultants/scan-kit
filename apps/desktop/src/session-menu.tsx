import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

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
      <DropdownMenuTrigger className="fixed size-px p-0 opacity-0" style={{ left: x, top: y }} />
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
