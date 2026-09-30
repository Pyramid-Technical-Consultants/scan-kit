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
    if (node != null) {
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
            void navigator.clipboard.writeText(lines.join("\n"));
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
