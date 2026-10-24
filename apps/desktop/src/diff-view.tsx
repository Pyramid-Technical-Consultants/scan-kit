import { useMemo, useState } from "react";
import {
  ChevronRight,
  CircleCheck,
  FileCheck2,
  FileDiff,
  FileMinus2,
  FilePen,
  FilePlus2,
  TriangleAlert,
} from "lucide-react";
import { cn } from "cn";

import { Badge } from "@/components/ui/badge";
import { toneColors, type CellTone } from "@/grid-theme";

type DiffLine = {
  kind: "context" | "add" | "del" | "gap";
  text?: string;
  old_no?: number | null;
  new_no?: number | null;
  count?: number;
};

type DiffFile = {
  rel: string;
  status: "changed" | "added" | "removed" | "same" | "binary" | "large";
  edited?: boolean;
  adds: number;
  dels: number;
  lines: DiffLine[];
};

export type DiffSession = {
  id: string;
  root?: string;
  warning?: string;
  self?: boolean;
  files: DiffFile[];
};

export type DiffResult = {
  base: string;
  sessions: DiffSession[];
};

const STATUS_META: Record<string, { label: string; Icon: typeof FileDiff; tone?: CellTone }> = {
  changed: { label: "Modified", Icon: FileDiff, tone: "warn" },
  added: { label: "Added", Icon: FilePlus2, tone: "good" },
  removed: { label: "Removed", Icon: FileMinus2, tone: "bad" },
  binary: { label: "Binary", Icon: TriangleAlert },
  large: { label: "Too large", Icon: TriangleAlert },
};

export function DiffView({ result }: { result: DiffResult }) {
  const tones = useMemo(() => toneColors(), []);
  const sessions = result.sessions.filter((session) => session.self !== true);
  if (sessions.length === 0) {
    return (
      <div className="text-muted-foreground flex min-h-0 flex-1 items-center justify-center p-3 text-sm">
        The selected sessions own the loaded configuration — nothing to diff.
      </div>
    );
  }
  return (
    <div className="flex min-h-0 flex-1 gap-3 overflow-auto p-3">
      {sessions.map((session) => (
        <SessionColumn key={session.id} session={session} tones={tones} />
      ))}
    </div>
  );
}

function SessionColumn({
  session,
  tones,
}: {
  session: DiffSession;
  tones: Record<CellTone, { bg: string; fg: string }>;
}) {
  const [showSame, setShowSame] = useState(false);
  const differing = session.files.filter((file) => file.status !== "same");
  const identical = session.files.filter((file) => file.status === "same");
  return (
    <section className="flex min-w-72 flex-1 flex-col overflow-hidden rounded-lg border">
      <header className="shrink-0 border-b px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-sm font-medium" title={session.id}>
            {session.id}
          </span>
          {session.warning == null ? (
            <span className="text-muted-foreground shrink-0 text-xs">
              {differing.length} differ · {identical.length} identical
            </span>
          ) : null}
        </div>
        {session.root != null ? (
          <p className="text-muted-foreground truncate text-xs" title={session.root}>
            {session.root}
          </p>
        ) : null}
      </header>
      {session.warning != null ? (
        <p className="flex items-center gap-2 px-3 py-2 text-sm" style={{ color: tones.warn.fg }}>
          <TriangleAlert className="size-4 shrink-0" />
          {session.warning}
        </p>
      ) : (
        <div className="min-h-0 flex-1 divide-y overflow-y-auto">
          {differing.map((file) => (
            <FileDiffSection key={file.rel} file={file} tones={tones} />
          ))}
          {differing.length === 0 ? (
            <p className="text-muted-foreground flex items-center gap-2 px-3 py-2 text-sm">
              <CircleCheck className="size-4 shrink-0" style={{ color: tones.good.fg }} />
              All files identical
            </p>
          ) : null}
          {identical.length > 0 ? (
            <div>
              <button
                type="button"
                className="text-muted-foreground hover:bg-muted/50 flex w-full cursor-pointer items-center gap-1.5 px-3 py-1.5 text-xs"
                onClick={() => setShowSame((shown) => !shown)}
              >
                <ChevronRight className={cn("size-3.5 transition-transform", showSame && "rotate-90")} />
                {identical.length} identical file{identical.length === 1 ? "" : "s"}
              </button>
              {showSame
                ? identical.map((file) => (
                    <p
                      key={file.rel}
                      className="text-muted-foreground flex items-center gap-1.5 truncate px-3 py-0.5 font-mono text-xs"
                      title={file.rel}
                    >
                      <FileCheck2 className="size-3.5 shrink-0" />
                      {file.rel}
                    </p>
                  ))
                : null}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}

function FileDiffSection({
  file,
  tones,
}: {
  file: DiffFile;
  tones: Record<CellTone, { bg: string; fg: string }>;
}) {
  const [open, setOpen] = useState(true);
  const meta = STATUS_META[file.status] ?? STATUS_META.changed;
  const Icon = meta.Icon;
  const tone = meta.tone != null ? tones[meta.tone] : null;
  return (
    <div>
      <button
        type="button"
        className="hover:bg-muted/50 flex w-full cursor-pointer items-center gap-2 px-3 py-1.5"
        onClick={() => setOpen((shown) => !shown)}
      >
        <ChevronRight className={cn("size-3.5 shrink-0 transition-transform", open && "rotate-90")} />
        <span className="min-w-0 flex-1 truncate text-left font-mono text-xs" title={file.rel}>
          {file.rel}
        </span>
        {file.dels > 0 ? <span className="shrink-0 font-mono text-xs" style={{ color: tones.bad.fg }}>−{file.dels}</span> : null}
        {file.adds > 0 ? <span className="shrink-0 font-mono text-xs" style={{ color: tones.good.fg }}>+{file.adds}</span> : null}
        {file.edited === true ? (
          <Badge variant="outline" className="shrink-0" style={{ color: tones.change.fg }}>
            <FilePen data-icon="inline-start" />
            unsaved
          </Badge>
        ) : null}
        <Badge variant="outline" className="shrink-0" style={tone != null ? { color: tone.fg } : undefined}>
          <Icon data-icon="inline-start" />
          {meta.label}
        </Badge>
      </button>
      {open && file.lines.length > 0 ? (
        <div
          className="grid border-t py-1 font-mono text-xs leading-5"
          style={{ gridTemplateColumns: "2.5rem 2.5rem 1rem minmax(0, 1fr)" }}
        >
          {file.lines.map((line, index) => (
            <DiffLineRow key={index} line={line} tones={tones} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function DiffLineRow({
  line,
  tones,
}: {
  line: DiffLine;
  tones: Record<CellTone, { bg: string; fg: string }>;
}) {
  if (line.kind === "gap") {
    return (
      <span className="bg-muted/40 text-muted-foreground col-span-full px-3 py-0.5 text-center text-[10px]">
        ⋮ {line.count ?? 0} unchanged lines
      </span>
    );
  }
  const tone = line.kind === "add" ? tones.good : line.kind === "del" ? tones.bad : null;
  const marker = line.kind === "add" ? "+" : line.kind === "del" ? "−" : "";
  const style = tone != null ? { backgroundColor: tone.bg } : undefined;
  const colored = tone != null ? { color: tone.fg } : undefined;
  return (
    <>
      <span
        className="text-muted-foreground select-none px-1 text-right"
        style={style}
      >
        {line.old_no ?? ""}
      </span>
      <span
        className="text-muted-foreground select-none px-1 text-right"
        style={style}
      >
        {line.new_no ?? ""}
      </span>
      <span className="select-none text-center" style={{ ...style, ...colored }}>
        {marker}
      </span>
      <span className="whitespace-pre px-1" style={{ ...style, ...colored }}>
        {line.text}
      </span>
    </>
  );
}
