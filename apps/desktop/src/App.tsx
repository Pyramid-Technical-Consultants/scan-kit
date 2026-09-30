import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { PhysicalPosition, PhysicalSize } from "@tauri-apps/api/dpi";
import { availableMonitors, getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";
import {
  DataEditor,
  GridCellKind,
  getDefaultTheme,
  type EditableGridCell,
  type GridCell,
  type GridColumn,
  type Item,
  type Theme,
} from "@glideapps/glide-data-grid";
import "@glideapps/glide-data-grid/dist/index.css";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { SessionContextMenu } from "@/session-menu";
import {
  Menubar,
  MenubarCheckboxItem,
  MenubarContent,
  MenubarGroup,
  MenubarItem,
  MenubarLabel,
  MenubarMenu,
  MenubarSeparator,
  MenubarTrigger,
} from "@/components/ui/menubar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AnalysisView } from "@/AnalysisView";

const MAX_SELECTED = 5;

const LAUNCHER_VIEWS = [
  "Data Analysis",
  "Plan Synthesis",
  "Phantom Synthesis",
  "Plan Runner",
  "Configuration Tuning",
  "Debug",
] as const;

type LauncherView = (typeof LAUNCHER_VIEWS)[number];

function isLauncherView(value: string): value is LauncherView {
  return (LAUNCHER_VIEWS as readonly string[]).includes(value);
}

const ANALYSIS_GROUPS = [
  {
    title: "Unified Views",
    names: [
      "Binned Summary",
      "Distribution Explorer",
      "Timeslice Replay",
      "FFT Explorer",
      "Audio Explorer",
      "IC Beam Trajectory (3D)",
      "Dose Volume",
      "Session Log Compare",
    ],
  },
  {
    title: "Specialized Analysis",
    names: [
      "Beam Error Motion vs Energy",
      "Dose Accumulation",
      "Beam-Off Ramp-Down",
      "IC HV Transient Test",
      "Amplifier Command Correlations",
      "IC Peak Amplitude — Beam-Off (G3)",
    ],
  },
] as const;

const ANALYSIS_IDS: Record<string, string> = {
  "Binned Summary": "binned_summary",
  "Distribution Explorer": "distribution",
  "Timeslice Replay": "timeslice_replay",
  "FFT Explorer": "ic_fft_analysis",
  "Audio Explorer": "ic_audio_player",
  "IC Beam Trajectory (3D)": "trajectory",
  "Dose Volume": "dose_volume",
  "Session Log Compare": "session_log_compare",
  "Beam Error Motion vs Energy": "beam_motion_energy",
  "Dose Accumulation": "dose_accumulation",
  "Beam-Off Ramp-Down": "beam_off_rampdown",
  "IC HV Transient Test": "ic_hv_transient",
  "Amplifier Command Correlations": "amplifier_correlation",
  "IC Peak Amplitude — Beam-Off (G3)": "ic_peak_amplitude_beam_off",
};

type SortKey =
  | "session_id"
  | "date"
  | "mu"
  | "extent"
  | "layers"
  | "time"
  | "room"
  | "config"
  | "note";

type Sort = { key: SortKey; direction: "asc" | "desc" };

const COLUMN_SORT: Array<SortKey | null> = [
  null,
  "session_id",
  "date",
  "mu",
  "extent",
  "layers",
  "time",
  "room",
  "config",
  "note",
];

const COLUMN_TITLES = [
  "Use",
  "Session ID",
  "Date",
  "MU",
  "Ext.",
  "Lyr.",
  "Time",
  "RM",
  "Config",
  "Note",
];

type LibraryRow = {
  session_id: string;
  storage_path: string;
  selected: boolean;
  note: string;
  date: string;
  date_iso: string | null;
  mu: string;
  mu_value: number | null;
  extent: string;
  extent_value: number | null;
  layers: string;
  layers_value: number | null;
  time: string;
  time_value: number | null;
  room: string;
  room_value: number | null;
  config: string;
};

type About = {
  title: string;
  product_line: string;
  description: string;
  source_lead: string;
  github_label: string;
  github_url: string;
  maintainer: string;
  website_label: string;
  website_url: string;
  support_lead: string;
  support_email: string;
  copyright: string;
  license_lead: string;
  license_label: string;
  license_url: string;
};

type NoteEdit = { sessionId: string; before: string; after: string };

type SessionMenu = { sessionId: string; x: number; y: number };

type Geometry = {
  width: number | null;
  height: number | null;
  x: number | null;
  y: number | null;
};

const MIN_WINDOW_WIDTH = 800;
const MIN_WINDOW_HEIGHT = 480;

type MonitorRect = {
  position: { x: number; y: number };
  size: { width: number; height: number };
};

function geometryOnScreen(
  geometry: Geometry,
  monitors: MonitorRect[],
): geometry is { width: number; height: number; x: number; y: number } {
  const { width, height, x, y } = geometry;
  if (width == null || height == null || x == null || y == null) {
    return false;
  }
  if (width < MIN_WINDOW_WIDTH || height < MIN_WINDOW_HEIGHT || x <= -16000 || y <= -16000) {
    return false;
  }
  return monitors.some((monitor) => {
    const overlapW =
      Math.min(x + width, monitor.position.x + monitor.size.width) -
      Math.max(x, monitor.position.x);
    const overlapH =
      Math.min(y + height, monitor.position.y + monitor.size.height) -
      Math.max(y, monitor.position.y);
    return overlapW >= 80 && overlapH >= 80;
  });
}

function tokenColor(name: string): string {
  const probe = document.createElement("span");
  probe.style.color = `var(${name})`;
  document.body.append(probe);
  const resolved = getComputedStyle(probe).color;
  probe.remove();
  return resolved;
}

function gridTheme(): Theme {
  const base = getDefaultTheme();
  const foreground = tokenColor("--foreground");
  const muted = tokenColor("--muted-foreground");
  const card = tokenColor("--card");
  const accent = tokenColor("--accent");
  const accentFg = tokenColor("--accent-foreground");
  const border = tokenColor("--border");
  const wash = tokenColor("--muted");
  return {
    ...base,
    accentColor: accent,
    accentFg,
    accentLight: wash,
    textDark: foreground,
    textMedium: muted,
    textLight: muted,
    textBubble: foreground,
    textHeader: foreground,
    textHeaderSelected: accentFg,
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
    drilldownBorder: border,
    linkColor: accent,
    fontFamily: getComputedStyle(document.documentElement).fontFamily,
  };
}

function sortValue(row: LibraryRow, key: SortKey): string | number | null {
  switch (key) {
    case "date":
      return row.date_iso;
    case "mu":
      return row.mu_value;
    case "extent":
      return row.extent_value;
    case "layers":
      return row.layers_value;
    case "time":
      return row.time_value;
    case "room":
      return row.room_value;
    case "session_id":
      return row.session_id;
    case "config":
      return row.config;
    case "note":
      return row.note;
  }
}

function compareRows(left: LibraryRow, right: LibraryRow, sort: Sort): number {
  const a = sortValue(left, sort.key);
  const b = sortValue(right, sort.key);
  if (a == null && b == null) {
    return 0;
  }
  if (a == null) {
    return 1;
  }
  if (b == null) {
    return -1;
  }
  const order =
    typeof a === "number" && typeof b === "number"
      ? a - b
      : String(a).localeCompare(String(b));
  return sort.direction === "asc" ? order : -order;
}

function textCell(value: string, editable: boolean): GridCell {
  return {
    kind: GridCellKind.Text,
    data: value,
    displayData: value,
    allowOverlay: editable,
    readonly: !editable,
    copyData: value,
  };
}

function messageText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function App() {
  const [folder, setFolder] = useState<string | null>(null);
  const [rows, setRows] = useState<LibraryRow[]>([]);
  const [sort, setSort] = useState<Sort>({ key: "date", direction: "desc" });
  const [message, setMessage] = useState<string | null>(null);
  const [about, setAbout] = useState<About | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [undo, setUndo] = useState<NoteEdit[]>([]);
  const [redo, setRedo] = useState<NoteEdit[]>([]);
  const [theme, setTheme] = useState<Theme | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [tab, setTab] = useState<LauncherView>("Data Analysis");
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [sessionMenu, setSessionMenu] = useState<SessionMenu | null>(null);
  const selectedIds = useMemo(
    () => rows.filter((row) => row.selected).map((row) => row.session_id),
    [rows],
  );
  const canAnalyze = folder != null && selectedIds.length >= 1 && selectedIds.length <= MAX_SELECTED;
  const host = useRef<HTMLDivElement>(null);
  const folderRef = useRef<string | null>(null);
  folderRef.current = folder;

  const order = useMemo(() => {
    const indexes = rows.map((_, index) => index);
    indexes.sort((a, b) => compareRows(rows[a], rows[b], sort));
    return indexes;
  }, [rows, sort]);

  const columns = useMemo<GridColumn[]>(
    () =>
      COLUMN_TITLES.map((title, index) => {
        const key = COLUMN_SORT[index];
        const marked =
          key != null && sort.key === key
            ? `${title} ${sort.direction === "desc" ? "↓" : "↑"}`
            : title;
        return {
          title: marked,
          id: key ?? "use",
          width: index === 0 ? 64 : index === 8 || index === 9 ? 220 : 110,
          grow: index === 8 || index === 9 ? 1 : 0,
        };
      }),
    [sort],
  );

  const loadFolder = useCallback(async (path: string) => {
    const opened = await invoke<{ root: string; rows: LibraryRow[] }>(
      "scan_kit_open_library",
      { path },
    );
    setFolder(opened.root);
    setRows(opened.rows);
    setMessage(null);
    setUndo([]);
    setRedo([]);
  }, []);

  useEffect(() => {
    setTheme(gridTheme());
    let active = true;
    invoke<About>("scan_kit_about")
      .then((value) => {
        if (active) {
          setAbout(value);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setMessage(messageText(error));
        }
      });
    invoke<string | null>("scan_kit_last_main_tab")
      .then((saved) => {
        if (active && saved != null && isLauncherView(saved)) {
          setTab(saved);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setMessage(messageText(error));
        }
      });
    invoke<string | null>("scan_kit_last_data_dir")
      .then((path) => {
        if (active && path) {
          return loadFolder(path);
        }
        return undefined;
      })
      .catch((error: unknown) => {
        if (active) {
          setMessage(messageText(error));
        }
      });
    return () => {
      active = false;
    };
  }, [loadFolder]);

  useEffect(() => {
    const node = host.current;
    if (node == null) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry == null) {
        return;
      }
      setSize({
        width: Math.floor(entry.contentRect.width),
        height: Math.floor(entry.contentRect.height),
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [tab]);

  useEffect(() => {
    let timer = 0;
    let unlisten: Array<() => void> = [];
    let active = true;
    void (async () => {
      try {
        const win = getCurrentWindow();
        const persistGeometry = async () => {
          if ((await win.isMinimized()) || (await win.isMaximized()) || (await win.isFullscreen())) {
            return;
          }
          const outer = await win.outerSize();
          const position = await win.outerPosition();
          const next = {
            width: outer.width,
            height: outer.height,
            x: position.x,
            y: position.y,
          };
          if (!geometryOnScreen(next, await availableMonitors())) {
            return;
          }
          await invoke("scan_kit_set_window_geometry", next);
        };
        const save = () => {
          window.clearTimeout(timer);
          timer = window.setTimeout(() => {
            void persistGeometry();
          }, 400);
        };
        const geometry = await invoke<Geometry>("scan_kit_window_geometry");
        if (active && geometryOnScreen(geometry, await availableMonitors())) {
          await win.setSize(new PhysicalSize(geometry.width, geometry.height));
          await win.setPosition(new PhysicalPosition(geometry.x, geometry.y));
        } else if (active) {
          await persistGeometry();
        }
        const resized = await win.onResized(save);
        const moved = await win.onMoved(save);
        const closing = await win.onCloseRequested(async (event) => {
          event.preventDefault();
          window.clearTimeout(timer);
          try {
            await persistGeometry();
          } catch {
            // A failed geometry write still lets the window close.
          }
          await win.destroy();
        });
        if (active) {
          unlisten = [resized, moved, closing];
        } else {
          resized();
          moved();
          closing();
        }
      } catch {
        // Vite in a browser has no window plugin. The menu still renders.
      }
    })();
    return () => {
      active = false;
      window.clearTimeout(timer);
      unlisten.forEach((stop) => stop());
    };
  }, []);

  const selectTab = useCallback((next: string) => {
    if (!isLauncherView(next)) {
      return;
    }
    setTab(next);
    void invoke("scan_kit_set_last_main_tab", { tab: next }).catch((error: unknown) => {
      setMessage(messageText(error));
    });
  }, []);

  const writeNote = useCallback(async (sessionId: string, note: string) => {
    const path = folderRef.current;
    if (path == null) {
      return;
    }
    await invoke("scan_kit_set_note", { path, sessionId, note });
    setRows((current) =>
      current.map((row) => (row.session_id === sessionId ? { ...row, note } : row)),
    );
  }, []);

  const onCellEdited = useCallback(
    (cell: Item, newValue: EditableGridCell) => {
      const [col, rowIndex] = cell;
      const source = order[rowIndex];
      const row = source == null ? undefined : rows[source];
      if (row == null || folder == null) {
        return;
      }
      if (col === 0 && newValue.kind === GridCellKind.Boolean) {
        if (typeof newValue.data !== "boolean") {
          return;
        }
        const selected = rows.filter((item) => item.selected).map((item) => item.session_id);
        const next = newValue.data
          ? selected.includes(row.session_id)
            ? selected
            : [...selected, row.session_id]
          : selected.filter((id) => id !== row.session_id);
        if (next.length > MAX_SELECTED) {
          setMessage("At most five sessions can be selected.");
          return;
        }
        void invoke("scan_kit_select_sessions", { path: folder, sessionIds: next })
          .then(() => {
            setRows((current) =>
              current.map((item) => ({
                ...item,
                selected: next.includes(item.session_id),
              })),
            );
            setMessage(null);
          })
          .catch((error: unknown) => setMessage(messageText(error)));
        return;
      }
      if (col === 9 && newValue.kind === GridCellKind.Text && newValue.data !== row.note) {
        const edit = { sessionId: row.session_id, before: row.note, after: newValue.data };
        void writeNote(row.session_id, newValue.data)
          .then(() => {
            setUndo((stack) => [...stack, edit]);
            setRedo([]);
            setMessage(null);
          })
          .catch((error: unknown) => setMessage(messageText(error)));
      }
    },
    [folder, order, rows, writeNote],
  );

  const getCellContent = useCallback(
    (cell: Item): GridCell => {
      const [col, rowIndex] = cell;
      const source = order[rowIndex];
      const row = source == null ? undefined : rows[source];
      if (row == null) {
        return textCell("", false);
      }
      if (col === 0) {
        return {
          kind: GridCellKind.Boolean,
          data: row.selected,
          allowOverlay: false,
          readonly: false,
          copyData: row.selected ? "1" : "0",
        };
      }
      const value = [
        row.session_id,
        row.date,
        row.mu,
        row.extent,
        row.layers,
        row.time,
        row.room,
        row.config,
        row.note,
      ][col - 1];
      return textCell(value ?? "", col === 9);
    },
    [order, rows],
  );

  async function chooseFolder() {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Open Data Folder",
      });
      if (typeof selected === "string") {
        await loadFolder(selected);
      }
    } catch (error) {
      setMessage(messageText(error));
    }
  }

  async function applyNote(edit: NoteEdit, note: string) {
    await writeNote(edit.sessionId, note);
  }

  return (
    <div className="flex h-svh flex-col bg-background text-foreground">
      <Menubar className="w-full rounded-none border-0 border-b px-2">
        <MenubarMenu>
          <MenubarTrigger>File</MenubarTrigger>
          <MenubarContent>
            <MenubarItem onClick={() => void chooseFolder()}>Open Data Folder</MenubarItem>
            <MenubarItem
              disabled={folder == null}
              onClick={() => {
                if (folder != null) {
                  void loadFolder(folder).catch((error: unknown) => setMessage(messageText(error)));
                }
              }}
            >
              Refresh Sessions
            </MenubarItem>
            <MenubarSeparator />
            <MenubarItem
              onClick={() => {
                void getCurrentWindow()
                  .close()
                  .catch((error: unknown) => setMessage(messageText(error)));
              }}
            >
              Exit
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>Edit</MenubarTrigger>
          <MenubarContent>
            <MenubarItem
              disabled={undo.length === 0}
              onClick={() => {
                const edit = undo[undo.length - 1];
                if (edit == null) {
                  return;
                }
                void applyNote(edit, edit.before)
                  .then(() => {
                    setUndo((stack) => stack.slice(0, -1));
                    setRedo((stack) => [...stack, edit]);
                  })
                  .catch((error: unknown) => setMessage(messageText(error)));
              }}
            >
              Undo
            </MenubarItem>
            <MenubarItem
              disabled={redo.length === 0}
              onClick={() => {
                const edit = redo[redo.length - 1];
                if (edit == null) {
                  return;
                }
                void applyNote(edit, edit.after)
                  .then(() => {
                    setRedo((stack) => stack.slice(0, -1));
                    setUndo((stack) => [...stack, edit]);
                  })
                  .catch((error: unknown) => setMessage(messageText(error)));
              }}
            >
              Redo
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>View</MenubarTrigger>
          <MenubarContent>
            {LAUNCHER_VIEWS.map((name) => (
              <MenubarCheckboxItem
                key={name}
                checked={tab === name}
                onCheckedChange={(checked) => {
                  if (checked) {
                    selectTab(name);
                  }
                }}
              >
                {name}
              </MenubarCheckboxItem>
            ))}
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>Analysis</MenubarTrigger>
          <MenubarContent>
            {ANALYSIS_GROUPS.map((group, index) => (
              <MenubarGroup key={group.title}>
                {index > 0 ? <MenubarSeparator /> : null}
                <MenubarLabel>{group.title}</MenubarLabel>
                {group.names.map((name) => (
                  <MenubarItem
                    key={name}
                    disabled={!canAnalyze}
                    onClick={() => {
                      setAnalysis(ANALYSIS_IDS[name]);
                      selectTab("Data Analysis");
                    }}
                  >
                    {name}
                  </MenubarItem>
                ))}
              </MenubarGroup>
            ))}
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>Help</MenubarTrigger>
          <MenubarContent>
            <MenubarItem onClick={() => setAboutOpen(true)}>About</MenubarItem>
          </MenubarContent>
        </MenubarMenu>
      </Menubar>

      <Tabs
        value={tab}
        onValueChange={(value) => {
          if (typeof value === "string") {
            selectTab(value);
          }
        }}
        className="min-h-0 flex-1 gap-0"
      >
        <TabsList variant="line" className="w-full justify-start">
          {LAUNCHER_VIEWS.map((name) => (
            <TabsTrigger key={name} value={name} className="flex-none">
              {name}
            </TabsTrigger>
          ))}
        </TabsList>
        <TabsContent value="Data Analysis" className="flex min-h-0 flex-col">
          {analysis != null && folder != null ? (
            <AnalysisView
              viewId={analysis}
              folder={folder}
              sessionIds={selectedIds}
              onBack={() => setAnalysis(null)}
            />
          ) : (
          <>
          <div className="text-muted-foreground truncate px-3 py-2 text-sm">
            {folder ?? "Open a data folder to list sessions."}
            {folder != null && rows.length === 0 ? " — No sessions in this folder." : ""}
            {rows.length > 0 ? ` — ${rows.length} sessions` : ""}
          </div>
          {message != null ? (
            <div className="text-destructive px-3 pb-2 text-sm">{message}</div>
          ) : null}
          <div
            ref={host}
            className="min-h-0 flex-1"
            onContextMenu={(event) => event.preventDefault()}
          >
            {theme != null && size.width > 0 && size.height > 0 ? (
              <DataEditor
                width={size.width}
                height={size.height}
                columns={columns}
                rows={rows.length}
                getCellContent={getCellContent}
                onCellEdited={onCellEdited}
                onCellContextMenu={([, rowIndex], event) => {
                  event.preventDefault();
                  const source = order[rowIndex];
                  const row = source == null ? undefined : rows[source];
                  const box = host.current?.getBoundingClientRect();
                  if (row == null || box == null) {
                    return;
                  }
                  setSessionMenu({
                    sessionId: row.session_id,
                    x: box.left + event.bounds.x + event.localEventX,
                    y: box.top + event.bounds.y + event.localEventY,
                  });
                }}
                onHeaderClicked={(col) => {
                  const key = COLUMN_SORT[col];
                  if (key == null) {
                    return;
                  }
                  setSort((current) =>
                    current.key === key
                      ? { key, direction: current.direction === "desc" ? "asc" : "desc" }
                      : { key, direction: key === "date" ? "desc" : "asc" },
                  );
                }}
                theme={theme}
                rowMarkers="none"
                smoothScrollX
                smoothScrollY
              />
            ) : null}
            {sessionMenu != null ? (
              <SessionContextMenu
                sessionId={sessionMenu.sessionId}
                x={sessionMenu.x}
                y={sessionMenu.y}
                onClose={() => setSessionMenu(null)}
                onCopy={(id) => {
                  void navigator.clipboard.writeText(id).then(
                    () => setMessage(`Copied session ${id}`),
                    () => setMessage(id),
                  );
                }}
                onTune={() => selectTab("Configuration Tuning")}
              />
            ) : null}
          </div>
          </>
          )}
        </TabsContent>
        {LAUNCHER_VIEWS.filter((name) => name !== "Data Analysis").map((name) => (
          <TabsContent key={name} value={name}>
            <p className="text-muted-foreground px-3 py-2">{name} is not in this preview yet.</p>
          </TabsContent>
        ))}
      </Tabs>

      <Dialog open={aboutOpen} onOpenChange={setAboutOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{about?.title ?? "About Scan Kit"}</DialogTitle>
            <DialogDescription>{about?.product_line}</DialogDescription>
          </DialogHeader>
          {about != null ? (
            <div className="flex flex-col gap-3 text-sm">
              <p>{about.description}</p>
              <p>
                {about.source_lead}{" "}
                <a
                  className="underline underline-offset-4"
                  href={about.github_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {about.github_label}
                </a>
                .
              </p>
              <p>
                {about.maintainer}
                <br />
                <a
                  className="underline underline-offset-4"
                  href={about.website_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {about.website_label}
                </a>
              </p>
              <p>
                {about.support_lead}{" "}
                <a className="underline underline-offset-4" href={`mailto:${about.support_email}`}>
                  {about.support_email}
                </a>
              </p>
              <p>
                {about.copyright}
                <br />
                {about.license_lead}{" "}
                <a
                  className="underline underline-offset-4"
                  href={about.license_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {about.license_label}
                </a>
                .
              </p>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
