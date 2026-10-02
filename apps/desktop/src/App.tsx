import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Bug,
  Combine,
  CircleHelp,
  Eye,
  FolderOpen,
  Info,
  LogOut,
  Play,
  Redo2,
  RefreshCw,
  SquarePen,
  SlidersHorizontal,
  Table2,
  Undo2,
  X,
  type LucideIcon,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { PhysicalPosition, PhysicalSize } from "@tauri-apps/api/dpi";
import { availableMonitors, getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";
import { emptyGridSelection, type GridSelection, type Theme } from "@glideapps/glide-data-grid";
import "@glideapps/glide-data-grid/dist/index.css";

import { gridTheme } from "@/grid-theme";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { AnalysisIcon, AnalysisMenu, PRIMARY_ANALYSES, analysisId } from "@/analysis-menu";
import { Button } from "@/components/ui/button";
import { DebugLog } from "@/DebugLog";
import { installDebugLog } from "@/debug-log";
import { dismissNotice, logError, notify, notifyError } from "@/notify";
import { selectionFromLibrary } from "@/session-colors";
import { type CheckPaint } from "@/session-checkbox";
import { SessionContextMenu } from "@/session-menu";
import {
  MAX_SELECTED,
  headerWillFill,
  nextSessionSelection,
  toggleListedSessions,
} from "@/session-checks";
import {
  checkPaint,
  compareRows,
  LibraryTable,
  type LibraryRow,
  type NoteEdit,
  type SessionMenu,
  type Sort,
  type SortKey,
} from "@/LibraryTable";
import {
  Menubar,
  MenubarCheckboxItem,
  MenubarContent,
  MenubarItem,
  MenubarMenu,
  MenubarSeparator,
  MenubarTrigger,
} from "@/components/ui/menubar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster } from "@/components/ui/toast";
import { AnalysisView } from "@/AnalysisView";
import { ConfigTuning } from "@/ConfigTuning";
import { PhantomSynthesis } from "@/PhantomSynthesis";
import { PlanRunner } from "@/PlanRunner";
import { PlanSynthesis } from "@/PlanSynthesis";

const TAB_ICONS: Record<string, LucideIcon> = {
  "Data Analysis": Table2,
  "Plan Synthesis": Combine,
  "Phantom Synthesis": Box,
  "Plan Runner": Play,
  "Configuration Tuning": SlidersHorizontal,
  Debug: Bug,
};

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

function TabIcon({ name }: { name: string }) {
  const Icon = TAB_ICONS[name];
  if (Icon == null) {
    return null;
  }
  return <Icon className="size-4" />;
}

export default function App() {
  const [folder, setFolder] = useState<string | null>(null);
  const [rows, setRows] = useState<LibraryRow[]>([]);
  const [sort, setSort] = useState<Sort>({ key: "date", direction: "desc" });
  const [about, setAbout] = useState<About | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [undo, setUndo] = useState<NoteEdit[]>([]);
  const [redo, setRedo] = useState<NoteEdit[]>([]);
  const [gridSelection, setGridSelection] = useState<GridSelection>(emptyGridSelection);
  const [theme, setTheme] = useState<Theme | null>(null);
  const [checks, setChecks] = useState<CheckPaint | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [tab, setTab] = useState<LauncherView>("Data Analysis");
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [sessionMenu, setSessionMenu] = useState<SessionMenu | null>(null);
  const [selectionOrder, setSelectionOrder] = useState<string[]>([]);
  const selectedIds = selectionOrder;
  const canAnalyze = folder != null && selectedIds.length >= 1 && selectedIds.length <= MAX_SELECTED;
  const host = useRef<HTMLDivElement>(null);
  const folderRef = useRef<string | null>(null);
  folderRef.current = folder;

  const order = useMemo(() => {
    const indexes = rows.map((_, index) => index);
    indexes.sort((a, b) => compareRows(rows[a], rows[b], sort));
    return indexes;
  }, [rows, sort]);

  const displayIds = useMemo(
    () =>
      order.flatMap((index) => {
        const id = rows[index]?.session_id;
        return id == null ? [] : [id];
      }),
    [order, rows],
  );

  const loadFolder = useCallback(async (path: string) => {
    const opened = await invoke<{ root: string; rows: LibraryRow[]; selected?: string[] }>(
      "scan_kit_open_library",
      { path },
    );
    setFolder(opened.root);
    setRows(opened.rows);
    setSelectionOrder(selectionFromLibrary(opened.rows, opened.selected));
    dismissNotice();
    setUndo([]);
    setRedo([]);
  }, []);

  useEffect(() => {
    installDebugLog();
    setTheme(gridTheme());
    setChecks(checkPaint());
    let active = true;
    invoke<About>("scan_kit_about")
      .then((value) => {
        if (active) {
          setAbout(value);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          logError(error);
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
          logError(error);
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
          notifyError(error);
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
    // The grid unmounts while a view is open. Measure again when Sessions brings it back,
    // or the table stays at 0×0 and looks like the button did nothing.
    const measure = () => {
      const rect = node.getBoundingClientRect();
      setSize({
        width: Math.floor(rect.width),
        height: Math.floor(rect.height),
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [tab, analysis]);

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
      notifyError(error);
    });
  }, []);

  const openAnalysis = useCallback(
    (name: string) => {
      const id = analysisId(name);
      if (id == null) {
        return;
      }
      setAnalysis(id);
      selectTab("Data Analysis");
    },
    [selectTab],
  );

  const commitSelection = useCallback((next: { ids: string[]; capped: boolean }) => {
    const path = folderRef.current;
    if (path == null) {
      return;
    }
    const same =
      selectionOrder.length === next.ids.length &&
      selectionOrder.every((id, index) => id === next.ids[index]);
    if (same) {
      if (next.capped) {
        notify("At most five sessions can be selected.", "selection");
      }
      return;
    }
    void invoke("scan_kit_select_sessions", { path, sessionIds: next.ids })
      .then(() => {
        const chosen = new Set(next.ids);
        setSelectionOrder(next.ids);
        setRows((items) => items.map((item) => ({ ...item, selected: chosen.has(item.session_id) })));
        if (next.capped) {
          notify("At most five sessions can be selected.", "selection");
        } else {
          dismissNotice("selection");
        }
      })
      .catch((error: unknown) => notifyError(error));
  }, [selectionOrder]);

  const onRowCheck = useCallback(
    (sessionId: string, checked: boolean) => {
      commitSelection(nextSessionSelection(displayIds, selectedIds, sessionId, checked));
    },
    [commitSelection, displayIds, selectedIds],
  );

  const onHeaderCheck = useCallback(() => {
    commitSelection(
      nextSessionSelection(
        displayIds,
        selectedIds,
        "all",
        headerWillFill(rows.length, selectedIds.length),
      ),
    );
  }, [commitSelection, displayIds, rows.length, selectedIds]);

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

  const commitNote = useCallback(
    (edit: NoteEdit) => {
      void writeNote(edit.sessionId, edit.after)
        .then(() => {
          setUndo((stack) => [...stack, edit]);
          setRedo([]);
          dismissNotice();
        })
        .catch((error: unknown) => notifyError(error));
    },
    [writeNote],
  );

  const changeSort = useCallback((key: SortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === "desc" ? "asc" : "desc" }
        : { key, direction: key === "date" ? "desc" : "asc" },
    );
  }, []);

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
      notifyError(error);
    }
  }

  async function applyNote(edit: NoteEdit, note: string) {
    await writeNote(edit.sessionId, note);
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background text-foreground">
      <Tabs
        value={tab}
        onValueChange={(value) => {
          if (typeof value === "string") {
            selectTab(value);
          }
        }}
        className="min-h-0 flex-1 gap-0 overflow-hidden"
      >
      <div className="flex shrink-0 items-center border-b">
      <Menubar className="shrink-0 rounded-none border-0 px-2">
        <MenubarMenu>
          <MenubarTrigger className="gap-1.5">
            <FolderOpen className="size-4" />
            File
          </MenubarTrigger>
          <MenubarContent className="w-max">
            <MenubarItem className="whitespace-nowrap" onClick={() => void chooseFolder()}>
              <FolderOpen />
              Open Data Folder
            </MenubarItem>
            <MenubarItem
              className="whitespace-nowrap"
              disabled={folder == null}
              onClick={() => {
                if (folder != null) {
                  void loadFolder(folder).catch((error: unknown) => notifyError(error));
                }
              }}
            >
              <RefreshCw />
              Refresh Sessions
            </MenubarItem>
            <MenubarSeparator />
            <MenubarItem
              className="whitespace-nowrap"
              onClick={() => {
                void getCurrentWindow()
                  .close()
                  .catch((error: unknown) => notifyError(error));
              }}
            >
              <LogOut />
              Exit
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger className="gap-1.5">
            <SquarePen className="size-4" />
            Edit
          </MenubarTrigger>
          <MenubarContent className="w-max">
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
                  .catch((error: unknown) => notifyError(error));
              }}
            >
              <Undo2 />
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
                  .catch((error: unknown) => notifyError(error));
              }}
            >
              <Redo2 />
              Redo
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger className="gap-1.5">
            <Eye className="size-4" />
            View
          </MenubarTrigger>
          <MenubarContent className="w-max">
            {LAUNCHER_VIEWS.map((name) => (
              <MenubarCheckboxItem
                key={name}
                className="whitespace-nowrap"
                checked={tab === name}
                onCheckedChange={(checked) => {
                  if (checked) {
                    selectTab(name);
                  }
                }}
              >
                <TabIcon name={name} />
                {name}
              </MenubarCheckboxItem>
            ))}
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger className="gap-1.5">
            <CircleHelp className="size-4" />
            Help
          </MenubarTrigger>
          <MenubarContent>
            <MenubarItem onClick={() => setAboutOpen(true)}>
              <Info />
              About
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
      </Menubar>
        <TabsList variant="line" className="ml-auto h-8 shrink-0 justify-end px-2">
          {LAUNCHER_VIEWS.map((name) => (
            <TabsTrigger key={name} value={name} className="flex-none px-2.5 after:bottom-0">
              <TabIcon name={name} />
              {name}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>
        <TabsContent value="Data Analysis" className="flex min-h-0 flex-col overflow-hidden">
          {analysis != null && folder != null ? (
            <AnalysisView
              viewId={analysis}
              folder={folder}
              sessions={selectedIds.map((id) => ({
                id,
                note: rows.find((row) => row.session_id === id)?.note ?? "",
              }))}
              onBack={() => setAnalysis(null)}
              onOpenView={setAnalysis}
            />
          ) : (
          <>
          <div className="flex items-center gap-2 px-3 py-2">
            <Button
              variant="outline"
              size="sm"
              disabled={selectedIds.length === 0}
              aria-label="Clear selection"
              onClick={() =>
                commitSelection(nextSessionSelection(displayIds, selectedIds, "all", false))
              }
            >
              <X />
              Clear
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={folder == null}
              onClick={() => {
                if (folder != null) {
                  void loadFolder(folder).catch((error: unknown) => notifyError(error));
                }
              }}
            >
              <RefreshCw />
              Refresh
            </Button>
            <div className="ml-auto flex items-center gap-2">
              {PRIMARY_ANALYSES.map((name) => (
                <Button key={name} size="sm" disabled={!canAnalyze} onClick={() => openAnalysis(name)}>
                  <AnalysisIcon name={name} />
                  {name}
                </Button>
              ))}
              <AnalysisMenu
                disabled={!canAnalyze}
                omit={PRIMARY_ANALYSES}
                onOpen={openAnalysis}
              />
            </div>
          </div>
          <div
            ref={host}
            className="relative min-h-0 flex-1 overflow-hidden"
            onContextMenu={(event) => event.preventDefault()}
          >
            {theme != null && size.width > 0 && size.height > 0 ? (
              <LibraryTable
                rows={rows}
                order={order}
                displayIds={displayIds}
                sort={sort}
                theme={theme}
                width={size.width}
                height={size.height}
                checks={checks}
                selectedIds={selectedIds}
                folder={folder}
                gridSelection={gridSelection}
                onGridSelectionChange={setGridSelection}
                onSort={changeSort}
                onRowCheck={onRowCheck}
                onHeaderCheck={onHeaderCheck}
                onCommitNote={commitNote}
                onOpenMenu={setSessionMenu}
              />
            ) : null}
            {rows.length === 0 ? (
              <p className="text-muted-foreground pointer-events-none absolute inset-x-0 top-12 text-center text-sm">
                {folder == null
                  ? "Open a data folder to list sessions."
                  : "No sessions in this folder."}
              </p>
            ) : null}
            {sessionMenu != null ? (
              <SessionContextMenu
                sessionId={sessionMenu.sessionId}
                x={sessionMenu.x}
                y={sessionMenu.y}
                rowIds={sessionMenu.rowIds}
                rowsSelected={
                  sessionMenu.rowIds.length > 0 &&
                  sessionMenu.rowIds.every((id) => selectedIds.includes(id))
                }
                onClose={() => setSessionMenu(null)}
                onCopy={(id) => {
                  void navigator.clipboard.writeText(id).then(
                    () => notify(`Copied session ${id}`),
                    () => notifyError("Could not copy the session id"),
                  );
                }}
                onTune={() => selectTab("Configuration Tuning")}
                onToggleRows={() => {
                  commitSelection(toggleListedSessions(selectedIds, sessionMenu.rowIds));
                }}
              />
            ) : null}
          </div>
          </>
          )}
        </TabsContent>
        <TabsContent value="Debug" className="flex min-h-0 flex-col overflow-hidden">
          <DebugLog />
        </TabsContent>
        <TabsContent value="Plan Synthesis" className="flex min-h-0 flex-col overflow-hidden">
          <PlanSynthesis />
        </TabsContent>
        <TabsContent value="Phantom Synthesis" className="flex min-h-0 flex-col overflow-hidden">
          <PhantomSynthesis />
        </TabsContent>
        <TabsContent value="Plan Runner" className="flex min-h-0 flex-col overflow-hidden">
          <PlanRunner />
        </TabsContent>
        <TabsContent value="Configuration Tuning" className="flex min-h-0 flex-col overflow-hidden">
          <ConfigTuning folder={folder ?? ""} selectedIds={selectedIds} />
        </TabsContent>
        {LAUNCHER_VIEWS.filter((name) => name !== "Data Analysis" && name !== "Debug" && name !== "Plan Synthesis" && name !== "Phantom Synthesis" && name !== "Configuration Tuning" && name !== "Plan Runner").map((name) => (
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
      <Toaster />
    </div>
  );
}
