import { useEffect, useRef, useState } from "react";
import { Copy, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { clearLog, logLines, subscribeLog } from "@/debug-log";

export function DebugLog() {
  const [lines, setLines] = useState<readonly string[]>(() => logLines());
  const view = useRef<HTMLPreElement>(null);
  // Whether the user is at the tail has to be measured before the append
  // commits — after commit, scrollHeight already includes the new lines, so
  // a multi-line burst (any error stack) would measure its own height, fail
  // the at-tail check, and never stick again. Track it in onScroll instead.
  const atTail = useRef(true);

  useEffect(() => subscribeLog(() => setLines(logLines().slice())), []);

  useEffect(() => {
    const node = view.current;
    if (node != null && atTail.current) {
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
            void navigator.clipboard?.writeText(lines.join("\n")).catch(() => {});
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
        onScroll={(event) => {
          const node = event.currentTarget;
          atTail.current = node.scrollHeight - node.scrollTop - node.clientHeight < 24;
        }}
        className="border-border min-h-0 flex-1 overflow-auto rounded-lg border p-2 font-mono text-xs"
      >
        {lines.join("\n")}
      </pre>
    </div>
  );
}
