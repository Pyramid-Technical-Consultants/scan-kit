import { useEffect, useRef, useState } from "react";
import { Copy, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { clearLog, logLines, subscribeLog } from "@/debug-log";

export function DebugLog() {
  const [lines, setLines] = useState<readonly string[]>(() => logLines());
  const view = useRef<HTMLPreElement>(null);

  useEffect(() => subscribeLog(() => setLines(logLines().slice())), []);

  useEffect(() => {
    const node = view.current;
    // Stick to the tail only while the user is already at it - yanking a
    // mid-scroll read on every line makes a chatty task log unreadable.
    if (node != null && node.scrollHeight - node.scrollTop - node.clientHeight < 24) {
      node.scrollTop = node.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
      <div className="flex items-center gap-2">
        <p className="text-muted-foreground min-w-0 flex-1 text-sm">
          Console, warnings, and uncaught errors from this window.
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            // The clipboard rejects without focus/permission — don't let that
            // land in the log being copied.
            void navigator.clipboard.writeText(lines.join("\n")).catch(() => {});
          }}
        >
          <Copy />
          Copy
        </Button>
        <Button variant="outline" size="sm" onClick={() => clearLog()}>
          <Trash2 />
          Clear
        </Button>
      </div>
      <pre
        ref={view}
        className="border-border min-h-0 flex-1 overflow-auto rounded-lg border p-2 font-mono text-xs"
      >
        {lines.join("\n")}
      </pre>
    </div>
  );
}
