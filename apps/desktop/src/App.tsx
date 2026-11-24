import { createElement, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Bug,
  Combine,
  CircleHelp,
  Eye,
  FolderOpen,
  Info,
  Plus,
  Trash2,
  Loader2,
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
import { LogicalSize, availableMonitors, getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";
import { emptyGridSelection, type GridSelection, type Theme } from "@glideapps/glide-data-grid";
import "@glideapps/glide-data-grid/dist/index.css";

import { gridTheme } from "@/grid-theme";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { AnalysisIcon, AnalysisMenu, PRIMARY_ANALYSES, analysisId } from "@/analysis-menu";
import { optionIcon } from "@/option-icons";
import { Button } from "@/components/ui/button";
import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { ExamTable, type ExamRow } from "@/ExamTable";
import { DebugLog } from "@/DebugLog";
import { installDebugLog } from "@/debug-log";
import { dismissNotice, logError, notify, notifyError } from "@/notify";
import { usePageLoad } from "@/page-load";
import { driveTask, type Report } from "@/task-client";
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
import { withoutEdit } from "@/note-history";
import {
  Menubar,
  MenubarCheckboxItem,
  MenubarContent,
  MenubarItem,
  MenubarMenu,
  MenubarSeparator,
  MenubarShortcut,
  MenubarTrigger,
} from "@/components/ui/menubar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster } from "@/components/ui/toast";
import { AnalysisView } from "@/AnalysisView";
import { ConfigTuning } from "@/ConfigTuning";
import { PhantomSynthesis } from "@/PhantomSynthesis";
import { PlanRunner } from "@/PlanRunner";
import { PlanSynthesis } from "@/PlanSynthesis";

function typingTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      target.tagName === "INPUT" ||
      target.tagName === "TEXTAREA" ||
      target.tagName === "SELECT")
  );
}

function layerOpen(target: EventTarget | null): boolean {
  if (
    target instanceof Element &&
    target.closest("[role='dialog'], [role='menu'], [role='listbox']") != null
  ) {
    return true;
  }
  return (
    document.querySelector(
      "[role='dialog'][data-open], [role='menu'][data-open], [role='listbox'][data-open]",
    ) != null
  );
}

function authenticationRequired(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.startsWith("authentication required");
}

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

/** Mirror of `location::is_remote_location`: URI `scheme://`, >1 char, not `file`. */
function isRemoteSpec(spec: string): boolean {
  const match = /^([A-Za-z][A-Za-z0-9+.-]*):\/\//.exec(spec.trim());
  if (match == null) {
    return false;
  }
  const scheme = match[1].toLowerCase();
  return scheme.length > 1 && scheme !== "file";
}

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
  third_party: ThirdParty[];
};

type ThirdParty = {
  name: string;
  license: string;
  attribution: string;
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

function emptyCatalogMessage(catalog: "Sessions" | "Exams", locations: number): string {
  if (locations === 0) {
    return catalog === "Sessions"
      ? "Add a data location to list sessions."
      : "Add a data location to list exams.";
  }
  return catalog === "Sessions"
    ? "No sessions in the saved locations."
    : "No DICOM exams in the saved locations.";
}

function CatalogPending({ name }: { name: "Sessions" | "Exams" }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center" role="status">
      <Loader2 className="text-muted-foreground size-8 animate-spin" />
      <span className="sr-only">Loading {name.toLowerCase()}</span>
    </div>
  );
}

function CatalogEmpty({
  name,
  message,
  onLocations,
}: {
  name: "Sessions" | "Exams";
  message: string;
  onLocations: () => void;
}) {
  const icon = optionIcon(name);
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6">
      <div className="flex max-w-sm flex-col items-center gap-3 text-center">
        {icon == null
          ? null
          : createElement(icon, { className: "text-muted-foreground size-8" })}
        <p className="text-muted-foreground text-sm">{message}</p>
        <Button onClick={onLocations}>
          <FolderOpen />
          Locations
        </Button>
      </div>
    </div>
  );
}

function TabIcon({ name }: { name: string }) {
  const Icon = TAB_ICONS[name];
  if (Icon == null) {
    return null;
  }
  return <Icon className="size-4" />;
}

type CatalogPayload = {
  locations: string[];
  rows: LibraryRow[];
  exams: ExamRow[];
  selected: string[];
};

export default function App() {
  const [locations, setLocations] = useState<string[]>([]);
  const [locationsKnown, setLocationsKnown] = useState(false);
  const [locationDraft, setLocationDraft] = useState("");
  const [locationsOpen, setLocationsOpen] = useState(false);
  const [catalog, setCatalog] = useState<"Sessions" | "Exams">("Sessions");
  const [passwordFor, setPasswordFor] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [rows, setRows] = useState<LibraryRow[]>([]);
  const [exams, setExams] = useState<ExamRow[]>([]);
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
  const [trashConfirm, setTrashConfirm] = useState<{
    sessionId: string;
    folderId?: string;
    library: string;
    remote: boolean;
    paths: string[];
  } | null>(null);
  const [selectionOrder, setSelectionOrder] = useState<string[]>([]);
  useEffect(() => {
    selectionLive.current = selectionOrder;
  }, [selectionOrder]);
  const [libraryReport, setLibraryReport] = useState<Report | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(false);
  usePageLoad(libraryLoading, libraryReport?.done ?? 0, libraryReport?.total ?? 0);
  const libraryToken = useRef<object>({});
  const catalogEpoch = useRef(0);
  const selectionLive = useRef<string[]>([]);
  const selectionWrites = useRef(0);
  const rowsRef = useRef(rows);
  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);
  // The optimistic set during an in-flight write — a separate ref because
  // selectionLive is effect-synced, and effect-touched refs may only be
  // written inside effects.
  const selectionPending = useRef<string[] | null>(null);
  const selectionSeq = useRef(0);
  const loadDepth = useRef(0);
  const noteBusy = useRef(false);
  const trashing = useRef(false);
  const pendingLocations = useRef<string[]>([]);
  const passwordSubmitted = useRef(false);
  const selectedIds = selectionOrder;
  // Sessions can live in any indexed location — anchor analysis to the
  // first selected row's own library, not locations[0].
  const folder = rows.find((row) => selectedIds.includes(row.session_id))?.library ?? locations[0] ?? null;
  const canAnalyze = folder != null && selectedIds.length >= 1 && selectedIds.length <= MAX_SELECTED;
  const noRows = catalog === "Sessions" ? rows.length === 0 : exams.length === 0;
  const catalogWaiting = noRows && (!locationsKnown || libraryLoading);
  const emptyMessage =
    catalogWaiting || !noRows ? null : emptyCatalogMessage(catalog, locations.length);
  const host = useRef<HTMLDivElement>(null);

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

  // Positional selection is meaningless when the displayed order changes —
  // a re-sort permutes order without touching rows, so key on displayIds.
  // A rows array rebuilt in place (checkbox toggle, note commit) keeps the
  // sequence and must not wipe the highlight.
  const rowKey = displayIds.join("\0");
  const [prevKey, setPrevKey] = useState(rowKey);
  if (prevKey !== rowKey) {
    setPrevKey(rowKey);
    setGridSelection(emptyGridSelection);
  }

  const beginCatalogLoad = useCallback(() => {
    loadDepth.current += 1;
    setLibraryLoading(true);
  }, []);

  const endCatalogLoad = useCallback(() => {
    loadDepth.current -= 1;
    if (loadDepth.current <= 0) {
      loadDepth.current = 0;
      setLibraryLoading(false);
    }
  }, []);

  const loadFolder = useCallback(
    async (path: string) => {
      const mine = {};
      libraryToken.current = mine;
      beginCatalogLoad();
      try {
        await driveTask(
          {
            view: "library",
            path,
            sessionIds: [],
            options: {},
            background: [],
            foreground: [],
            palette: [],
          },
          (report) => {
            if (libraryToken.current !== mine) {
              return;
            }
            setLibraryReport(report.finished ? null : report);
          },
          () => libraryToken.current !== mine,
        );
      } finally {
        // A failed load throws out of driveTask — drop its last fraction so
        // the next load does not open on the previous one's progress.
        if (libraryToken.current === mine) {
          setLibraryReport(null);
        }
        endCatalogLoad();
      }
    },
    [beginCatalogLoad, endCatalogLoad],
  );

  const applyCatalog = useCallback((payload: CatalogPayload) => {
    setLocations(payload.locations);
    setExams(payload.exams);
    // A selection write is in flight — its optimistic state is newer than this
    // payload, so keep it instead of reverting the checkmark mid-write.
    if (selectionWrites.current > 0) {
      const live = new Set(selectionPending.current ?? selectionLive.current);
      setRows(payload.rows.map((item) => ({ ...item, selected: live.has(item.session_id) })));
    } else {
      setRows(payload.rows);
      setSelectionOrder(payload.selected);
    }
  }, []);

  const pullCatalog = useCallback(
    async (epoch?: number) => {
      const payload = await invoke<CatalogPayload>("scan_kit_read_catalog");
      // A superseding open/indexMany that started during this pull — even
      // one that later failed — must not lose to this stale snapshot.
      if (epoch != null && catalogEpoch.current !== epoch) {
        return;
      }
      applyCatalog(payload);
    },
    [applyCatalog],
  );

  const indexLocation = useCallback(
    async (path: string): Promise<"ok" | "auth" | "error"> => {
      try {
        await loadFolder(path);
        return "ok";
      } catch (error: unknown) {
        if (authenticationRequired(error)) {
          setPassword("");
          setPasswordFor(path);
          return "auth";
        }
        notifyError(error);
        return "error";
      }
    },
    [loadFolder],
  );

  const openLocation = useCallback(
    async (path: string) => {
      // Newest action wins — a running indexMany stops at the next root
      // instead of reclaiming the token and cancelling this load.
      const epoch = ++catalogEpoch.current;
      beginCatalogLoad();
      try {
        const result = await indexLocation(path);
        // A superseding open/indexMany must not let this stale pull
        // overwrite the newer catalog.
        if (result !== "ok" || catalogEpoch.current !== epoch) {
          return;
        }
        try {
          await pullCatalog(epoch);
          setUndo([]);
          setRedo([]);
          setLocationDraft("");
        } catch (error: unknown) {
          notifyError(error);
        }
      } finally {
        endCatalogLoad();
      }
    },
    [beginCatalogLoad, endCatalogLoad, indexLocation, pullCatalog],
  );

  const indexMany = useCallback(
    async (roots: string[]) => {
      const epoch = ++catalogEpoch.current;
      beginCatalogLoad();
      let stopped = -1;
      try {
        for (let index = 0; index < roots.length; index += 1) {
          const root = roots[index];
          if (root == null) {
            continue;
          }
          if (catalogEpoch.current !== epoch) {
            break;
          }
          const result = await indexLocation(root);
          if (result === "auth") {
            stopped = index;
            break;
          }
        }
        if (catalogEpoch.current === epoch) {
          pendingLocations.current = stopped >= 0 ? roots.slice(stopped + 1) : [];
          try {
            await pullCatalog(epoch);
            dismissNotice();
            setUndo([]);
            setRedo([]);
          } catch (error: unknown) {
            notifyError(error);
          }
        } else {
          // A superseded run must not leave stale pending roots — the next
          // password submit would re-index removed locations.
          pendingLocations.current = [];
        }
      } finally {
        endCatalogLoad();
      }
    },
    [beginCatalogLoad, endCatalogLoad, indexLocation, pullCatalog],
  );

  const refreshLocations = useCallback(async () => {
    await indexMany(locations);
  }, [indexMany, locations]);

  const removeLocation = useCallback(
    async (path: string) => {
      // Bump first — a queued indexMany would re-register the forgotten
      // root inside index_library's transaction and republish it.
      const epoch = ++catalogEpoch.current;
      pendingLocations.current = [];
      try {
        const payload = await invoke<CatalogPayload>("scan_kit_forget_data_dir", { path });
        // A newer open/indexMany that landed during the invoke already
        // pulled — its catalog is fresher than this pre-commit snapshot.
        if (catalogEpoch.current === epoch) {
          applyCatalog(payload);
          setUndo([]);
          setRedo([]);
          setSessionMenu(null);
          setTrashConfirm(null);
        }
      } catch (error: unknown) {
        notifyError(error);
      }
    },
    [applyCatalog],
  );

  useEffect(() => {
    installDebugLog();
    let active = true;
    // Theme reads CSS variables from the document, so it waits until after this commit.
    const frame = requestAnimationFrame(() => {
      if (!active) {
        return;
      }
      setTheme(gridTheme());
      setChecks(checkPaint());
    });
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
    invoke<string[]>("scan_kit_data_dirs")
      .then(async (roots) => {
        if (!active) {
          return;
        }
        setLocations(roots);
        setLocationsKnown(true);
        if (roots.length === 0) {
          return;
        }
        await indexMany(roots);
      })
      .catch((error: unknown) => {
        if (active) {
          setLocationsKnown(true);
          notifyError(error);
        }
      });
    return () => {
      active = false;
      libraryToken.current = {};
      cancelAnimationFrame(frame);
    };
  }, [indexMany]);

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
            // Geometry persistence is best-effort — a rejection (teardown,
            // absent plugin) must not land as an unhandled rejection.
            void persistGeometry().catch(() => {});
          }, 400);
        };
        const geometry = await invoke<Geometry>("scan_kit_window_geometry");
        if (active && geometryOnScreen(geometry, await availableMonitors())) {
          await win.setSize(new PhysicalSize(geometry.width, geometry.height));
          await win.setPosition(new PhysicalPosition(geometry.x, geometry.y));
        } else if (active) {
          await persistGeometry();
        }
        // Push each unlisten as its registration lands — a mid-sequence
        // throw must still undo the ones already wired up.
        unlisten.push(await win.onResized(save));
        unlisten.push(await win.onMoved(save));
        unlisten.push(
          await win.onCloseRequested(async (event) => {
            event.preventDefault();
            window.clearTimeout(timer);
            try {
              // The store write can wait on a scan's lock for seconds — cap
              // it or closing the window visibly stalls behind an index.
              await Promise.race([
                persistGeometry(),
                new Promise<void>((resolve) => window.setTimeout(resolve, 1500)),
              ]);
            } catch {
              // A failed geometry write still lets the window close.
            } finally {
              await win.destroy().catch(() => {});
            }
          }),
        );
        if (!active) {
          unlisten.forEach((fn) => fn());
          unlisten = [];
        }
      } catch {
        // Undo whatever did register before the throw. Vite in a browser
        // has no window plugin; the menu still renders.
        unlisten.forEach((fn) => fn());
        unlisten = [];
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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || layerOpen(event.target)) {
        return;
      }
      const key = event.key.toLowerCase();
      // Alt/Meta make the chord somebody else's — Ctrl+Alt+Q must not quit,
      // and AltGr reports ctrlKey+altKey on Windows layouts.
      const ctrl = event.ctrlKey && !event.altKey && !event.metaKey;
      const typing = typingTarget(event.target);
      if (event.key === "Escape") {
        if (event.repeat || typing) {
          return;
        }
        event.preventDefault();
        void getCurrentWindow()
          .close()
          .catch((error: unknown) => notifyError(error));
        return;
      }
      if (key === "f5") {
        if (event.repeat || typing) {
          return;
        }
        event.preventDefault();
        void refreshLocations();
        return;
      }
      if (!ctrl || typing) {
        return;
      }
      if (key === "o") {
        if (event.repeat) {
          return;
        }
        event.preventDefault();
        void chooseFolder();
        return;
      }
      if (key === "q") {
        if (event.repeat) {
          return;
        }
        event.preventDefault();
        void getCurrentWindow()
          .close()
          .catch((error: unknown) => notifyError(error));
        return;
      }
      if (key === "1" && !event.shiftKey) {
        if (!event.repeat) {
          selectTab("Data Analysis");
        }
        event.preventDefault();
        return;
      }
      if (key === "6" && !event.shiftKey) {
        if (!event.repeat) {
          selectTab("Debug");
        }
        event.preventDefault();
        return;
      }
      if (key === "z" && event.shiftKey) {
        redoNote();
        event.preventDefault();
        return;
      }
      if (key === "y") {
        redoNote();
        event.preventDefault();
        return;
      }
      if (key === "z") {
        undoNote();
        event.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

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

  const requestTrash = useCallback(
    async (sessionId: string, library: string) => {
      const row = rows.find((item) => item.session_id === sessionId && item.library === library);
      const path = row?.library ?? library;
      if (path == null) {
        return;
      }
      try {
        const result = await invoke<{ paths: string[] }>("scan_kit_session_storage_paths", {
          path,
          sessionId: row?.folder_id ?? sessionId,
        });
        if (result.paths.length === 0) {
          notify(`Nothing on disk for session ${sessionId}`);
          return;
        }
        setTrashConfirm({
          sessionId,
          folderId: row?.folder_id,
          library,
          remote: isRemoteSpec(library),
          paths: result.paths,
        });
      } catch (error) {
        notifyError(error);
      }
    },
    [rows, setTrashConfirm],
  );

  const confirmTrash = useCallback(async () => {
    const pending = trashConfirm;
    if (pending == null || trashing.current) {
      return;
    }
    trashing.current = true;
    // A catalog pull in flight snapshots the row before the backend removes
    // it — supersede it so the trashed session cannot resurrect.
    ++catalogEpoch.current;
    setTrashConfirm(null);
    try {
      const result = await invoke<{ removed: number; errors: string[] }>(
        "scan_kit_trash_session",
        { path: pending.library, sessionId: pending.folderId ?? pending.sessionId },
      );
      if (result.errors.length > 0) {
        // Files remain on disk — the catalog row stays (and keeps its
        // note/selection) so a later retry sees the same session.
        notifyError(`Some session files could not be removed:\n${result.errors.join("\n")}`);
      } else if (result.removed === 0) {
        notify(`No files found for session ${pending.sessionId}`);
      } else {
        setRows((items) =>
          items.filter(
            (item) => item.session_id !== pending.sessionId || item.library !== pending.library,
          ),
        );
        setSelectionOrder((ids) => ids.filter((id) => id !== pending.sessionId));
        notify(
          pending.remote
            ? `Deleted session ${pending.sessionId}`
            : `Moved session ${pending.sessionId} to the Recycle Bin`,
        );
      }
    } catch (error) {
      notifyError(error);
    } finally {
      trashing.current = false;
    }
  }, [trashConfirm, setTrashConfirm]);

  const commitSelection = useCallback((next: { ids: string[]; capped: boolean }) => {
    if (locations.length === 0) {
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
    // Apply before the invokes so a second check computes from this set, not
    // stale state; the remote write catches up and a failure rolls back.
    const before = selectionOrder;
    const chosen = new Set(next.ids);
    // The pull-side guard reads selectionPending synchronously — a passive
    // effect would leave the old set visible to a catalog that lands mid-write.
    selectionPending.current = next.ids;
    setSelectionOrder(next.ids);
    setRows((items) => items.map((item) => ({ ...item, selected: chosen.has(item.session_id) })));
    const write = ++selectionSeq.current;
    const catalogAt = catalogEpoch.current;
    void (async () => {
      selectionWrites.current += 1;
      try {
        for (const root of locations) {
          // A newer selection supersedes this loop — stop before the next
          // root so the last write to land is always the newest. A location
          // forgotten mid-write must not get a ghost selection persisted.
          if (selectionSeq.current !== write || catalogEpoch.current !== catalogAt) {
            return;
          }
          const sessionIds = rowsRef.current.flatMap((row) => {
            if (!chosen.has(row.session_id) || (row.library ?? locations[0]) !== root) {
              return [];
            }
            return [row.folder_id ?? row.session_id];
          });
          await invoke("scan_kit_select_sessions", { path: root, sessionIds });
        }
        if (next.capped) {
          notify("At most five sessions can be selected.", "selection");
        } else {
          dismissNotice("selection");
        }
      } finally {
        selectionWrites.current -= 1;
        if (selectionWrites.current === 0) {
          selectionPending.current = null;
          // A pull that landed mid-write may have shrunk the catalog -
          // drop optimistic ids that no longer list, unless a newer
          // commit already superseded this one.
          if (selectionSeq.current === write) {
            const listed = new Set(rowsRef.current.map((row) => row.session_id));
            setSelectionOrder((ids) => ids.filter((id) => listed.has(id)));
          }
        }
      }
    })().catch((error: unknown) => {
      // Roll back only if nothing newer landed — the write token is set
      // synchronously at apply time, so a later commit supersedes this
      // failure even when it has already settled.
      const stillMine = selectionSeq.current === write;
      if (stillMine) {
        // A pull that landed mid-write may have shrunk the catalog — only
        // restore ids that still list, like the drain filter does.
        const listed = new Set(rowsRef.current.map((row) => row.session_id));
        const kept = before.filter((id) => listed.has(id));
        const restore = new Set(kept);
        setSelectionOrder(kept);
        setRows((items) =>
          items.map((item) => ({ ...item, selected: restore.has(item.session_id) })),
        );
        // Only this generation's failure is news — a superseded write's
        // error would toast after the newer commit already won.
        notifyError(error);
      }
    });
  }, [locations, selectionOrder]);

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
        headerWillFill(rows.length, rows.filter((row) => selectedIds.includes(row.session_id)).length),
      ),
    );
  }, [commitSelection, displayIds, rows, selectedIds]);

  // Automation surface for scripts/capture-doc-screenshots.mjs (CDP-driven doc shots).
  // The container outlives re-renders; AnalysisView parks its own `view` entry in it.
  useEffect(() => {
    const win = window as unknown as { __scanKitShots?: Record<string, unknown> };
    win.__scanKitShots ??= {};
    return () => {
      delete win.__scanKitShots;
    };
  }, []);
  useEffect(() => {
    const win = window as unknown as { __scanKitShots?: Record<string, unknown> };
    const shots = (win.__scanKitShots ??= {});
    Object.assign(shots, {
      invoke,
      openLocation,
      openAnalysis,
      back: () => setAnalysis(null),
      selectTab,
      selectSessions: (ids: string[]) =>
        commitSelection({ ids: ids.slice(0, MAX_SELECTED), capped: false }),
      sessionIds: () => rows.map((row) => row.session_id),
      setSize: (width: number, height: number) =>
        void getCurrentWindow().setSize(new LogicalSize(width, height)),
    });
  }, [openLocation, openAnalysis, selectTab, commitSelection, rows]);

  const writeNote = useCallback(async (sessionId: string, library: string, note: string) => {
    // session_id is not unique across locations - the note must commit to
    // the copy the user clicked, not the first catalog match.
    const row = rows.find((item) => item.session_id === sessionId && item.library === library);
    const path = row?.library ?? library;
    if (path == null) {
      return;
    }
    await invoke("scan_kit_set_note", { path, sessionId: row?.folder_id ?? sessionId, note });
    setRows((current) =>
      current.map((item) =>
        item.session_id === sessionId && item.library === library ? { ...item, note } : item,
      ),
    );
  }, [rows]);

  // Two rapid edits must push undo entries in user order — invoke
    // completion order is not guaranteed.
  const noteChain = useRef<Promise<unknown>>(Promise.resolve());
  const commitNote = useCallback(
    (edit: NoteEdit): Promise<boolean> => {
      const run = noteChain.current.then(() =>
        writeNote(edit.sessionId, edit.library, edit.after),
      );
      noteChain.current = run.catch(() => undefined);
      return run
        .then(() => {
          setUndo((stack) => [...stack, edit]);
          setRedo([]);
          dismissNotice();
          return true;
        })
        .catch((error: unknown) => {
          notifyError(error);
          return false;
        });
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
        await openLocation(selected);
      }
    } catch (error) {
      notifyError(error);
    }
  }

  async function applyNote(edit: NoteEdit, note: string) {
    // Same chain commitNote uses — an undo must not race an in-flight
    // commit or the persisted text and the stacks diverge.
    const run = noteChain.current.then(() =>
      writeNote(edit.sessionId, edit.library, note),
    );
    noteChain.current = run.catch(() => undefined);
    await run;
  }

  function undoNote() {
    const edit = undo[undo.length - 1];
    if (edit == null || noteBusy.current) {
      return;
    }
    noteBusy.current = true;
    void applyNote(edit, edit.before)
      .then(() => {
        setUndo((stack) => withoutEdit(stack, edit));
        setRedo((stack) => [...stack, edit]);
      })
      .catch((error: unknown) => notifyError(error))
      .finally(() => {
        noteBusy.current = false;
      });
  }

  function redoNote() {
    const edit = redo[redo.length - 1];
    if (edit == null || noteBusy.current) {
      return;
    }
    noteBusy.current = true;
    void applyNote(edit, edit.after)
      .then(() => {
        setRedo((stack) => withoutEdit(stack, edit));
        setUndo((stack) => [...stack, edit]);
      })
      .catch((error: unknown) => notifyError(error))
      .finally(() => {
        noteBusy.current = false;
      });
  }

  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden bg-background text-foreground">
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
              <MenubarShortcut>Ctrl+O</MenubarShortcut>
            </MenubarItem>
            <MenubarItem
              className="whitespace-nowrap"
              disabled={folder == null}
              onClick={() => {
                if (folder != null) {
                  void openLocation(folder);
                }
              }}
            >
              <RefreshCw />
              Refresh Sessions
              <MenubarShortcut>F5</MenubarShortcut>
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
              <MenubarShortcut>Ctrl+Q</MenubarShortcut>
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger className="gap-1.5">
            <SquarePen className="size-4" />
            Edit
          </MenubarTrigger>
          <MenubarContent className="w-max">
            <MenubarItem disabled={undo.length === 0} onClick={undoNote}>
              <Undo2 />
              Undo
              <MenubarShortcut>Ctrl+Z</MenubarShortcut>
            </MenubarItem>
            <MenubarItem disabled={redo.length === 0} onClick={redoNote}>
              <Redo2 />
              Redo
              <MenubarShortcut>Ctrl+Y</MenubarShortcut>
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
                {name === "Data Analysis" ? <MenubarShortcut>Ctrl+1</MenubarShortcut> : null}
                {name === "Debug" ? <MenubarShortcut>Ctrl+6</MenubarShortcut> : null}
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
              key={analysis}
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
              disabled={locations.length === 0}
              onClick={() => {
                void refreshLocations();
              }}
            >
              <RefreshCw />
              Refresh
            </Button>
            <Button variant="outline" size="sm" onClick={() => setLocationsOpen(true)}>
              <FolderOpen />
              Locations
            </Button>
            <ButtonSegmentGroup
              label="Catalog"
              className="w-fit shrink-0"
              options={["Sessions", "Exams"]}
              value={catalog}
              onChange={(value) => {
                if (value === "Sessions" || value === "Exams") {
                  setCatalog(value);
                  setSessionMenu(null);
                }
              }}
            />
            <div className="ml-auto flex items-center gap-2">
              {catalog === "Sessions"
                ? PRIMARY_ANALYSES.map((name) => (
                    <Button key={name} size="sm" disabled={!canAnalyze} onClick={() => openAnalysis(name)}>
                      <AnalysisIcon name={name} />
                      {name}
                    </Button>
                  ))
                : null}
              {catalog === "Sessions" ? (
                <AnalysisMenu
                  disabled={!canAnalyze}
                  omit={PRIMARY_ANALYSES}
                  onOpen={openAnalysis}
                />
              ) : null}
            </div>
          </div>
          <div
            ref={host}
            className="relative min-h-0 flex-1 overflow-hidden"
            onContextMenu={(event) => event.preventDefault()}
          >
            {!catalogWaiting && emptyMessage == null && theme != null && size.width > 0 && size.height > 0 && catalog === "Sessions" ? (
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
            {!catalogWaiting && emptyMessage == null && theme != null && size.width > 0 && size.height > 0 && catalog === "Exams" ? (
              <ExamTable rows={exams} theme={theme} width={size.width} height={size.height} />
            ) : null}
            {catalogWaiting ? <CatalogPending name={catalog} /> : null}
            {emptyMessage != null ? (
              <CatalogEmpty
                name={catalog}
                message={emptyMessage}
                onLocations={() => setLocationsOpen(true)}
              />
            ) : null}
            {catalog === "Sessions" && sessionMenu != null ? (
              <SessionContextMenu
                sessionId={sessionMenu.sessionId}
                x={sessionMenu.x}
                y={sessionMenu.y}
                rowIds={sessionMenu.rowIds}
                rowsSelected={
                  sessionMenu.rowIds.length > 0 &&
                  sessionMenu.rowIds.every((id) => selectedIds.includes(id))
                }
                remote={isRemoteSpec(
                  rows.find(
                    (item) =>
                      item.session_id === sessionMenu.sessionId &&
                      item.library === sessionMenu.library,
                  )?.library ??
                    locations[0] ??
                    "",
                )}
                onClose={() => setSessionMenu(null)}
                onCopy={(id) => {
                  void navigator.clipboard.writeText(id).then(
                    () => notify(`Copied session ${id}`),
                    () => notifyError("Could not copy the session id"),
                  );
                }}
                onTune={() => {
                  if (!sessionMenu.rowIds.every((id) => selectedIds.includes(id))) {
                    commitSelection(toggleListedSessions(selectedIds, sessionMenu.rowIds));
                  }
                  selectTab("Configuration Tuning");
                }}
                onToggleRows={() => {
                  commitSelection(toggleListedSessions(selectedIds, sessionMenu.rowIds));
                }}
                onTrash={() => void requestTrash(sessionMenu.sessionId, sessionMenu.library)}
              />
            ) : null}
          </div>
          </>
          )}
        </TabsContent>
        <TabsContent value="Debug" className="flex min-h-0 flex-col overflow-hidden">
          <DebugLog />
        </TabsContent>
        <TabsContent value="Plan Synthesis" keepMounted className="flex min-h-0 flex-col overflow-hidden">
          <PlanSynthesis />
        </TabsContent>
        <TabsContent value="Phantom Synthesis" keepMounted className="flex min-h-0 flex-col overflow-hidden">
          <PhantomSynthesis />
        </TabsContent>
        <TabsContent value="Plan Runner" keepMounted className="flex min-h-0 flex-col overflow-hidden">
          <PlanRunner />
        </TabsContent>
        <TabsContent value="Configuration Tuning" keepMounted className="flex min-h-0 flex-col overflow-hidden">
          <ConfigTuning
            folder={folder ?? ""}
            sessions={selectedIds.map((id) => ({
              id,
              note: rows.find((row) => row.session_id === id)?.note ?? "",
            }))}
            selectedIds={selectedIds}
          />
        </TabsContent>
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
              {(about.third_party?.length ?? 0) > 0 ? (
                <div>
                  <p className="font-medium">Third-party software and data</p>
                  <ul className="list-disc pl-5 text-muted-foreground">
                    {(about.third_party ?? []).map((item) => (
                      <li key={item.name}>
                        {item.name} — {item.license}. {item.attribution}.
                      </li>
                    ))}
                  </ul>
                  <p className="mt-1 text-muted-foreground">
                    Full license texts install alongside the app in the <code>licenses</code> folder.
                  </p>
                </div>
              ) : null}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
      <Dialog open={trashConfirm != null} onOpenChange={(open) => !open && setTrashConfirm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {trashConfirm?.remote ? "Delete remote session" : "Move session to Recycle Bin"}
            </DialogTitle>
            <DialogDescription>
              {trashConfirm?.remote
                ? `Permanently delete session ${trashConfirm.sessionId} from the remote host?`
                : `Move session ${trashConfirm?.sessionId} to the Recycle Bin?`}
            </DialogDescription>
          </DialogHeader>
          {trashConfirm != null ? (
            <div className="flex flex-col gap-2 text-sm">
              <p>These items will be removed:</p>
              <ul className="max-h-40 overflow-y-auto font-mono text-xs">
                {trashConfirm.paths.map((path) => (
                  <li key={path}>{path}</li>
                ))}
              </ul>
              <p>
                {trashConfirm.remote
                  ? "They are not sent to the Recycle Bin."
                  : "You can restore them from the Recycle Bin if needed."}
              </p>
              {selectedIds.includes(trashConfirm.sessionId) ? (
                <p>
                  This session is currently selected for analysis. Open plot windows may fail
                  after the files are removed.
                </p>
              ) : null}
            </div>
          ) : null}
          <DialogFooter>
            <DialogClose render={<Button variant="outline" />}>Cancel</DialogClose>
            <Button variant="destructive" onClick={() => void confirmTrash()}>
              {trashConfirm?.remote ? "Delete" : "Move to Recycle Bin"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={locationsOpen} onOpenChange={setLocationsOpen}>
        <DialogContent className="sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>Locations</DialogTitle>
            <DialogDescription>
              Sessions and exams from every location are listed together.
            </DialogDescription>
          </DialogHeader>
          <div className="flex max-h-60 flex-col gap-2 overflow-auto">
            {locations.length === 0 ? (
              <p className="text-muted-foreground text-sm">No saved locations.</p>
            ) : (
              locations.map((root) => (
                <div key={root} className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-sm" title={root}>
                    {root}
                  </span>
                  <Button variant="outline" size="sm" onClick={() => void removeLocation(root)}>
                    <Trash2 />
                    Remove
                  </Button>
                </div>
              ))
            )}
          </div>
          <form
            className="flex items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              const spec = locationDraft.trim();
              if (spec.length > 0) {
                void openLocation(spec);
              }
            }}
          >
            <Input
              value={locationDraft}
              placeholder="Folder, UNC, or sftp://user@host/path"
              aria-label="Folder, UNC, or sftp://user@host/path"
              onChange={(event) => setLocationDraft(event.target.value)}
            />
            <Button type="submit">
              <Plus />
              Add
            </Button>
          </form>
          <DialogFooter>
            <Button variant="outline" onClick={() => void chooseFolder()}>
              <FolderOpen />
              Folder
            </Button>
            <DialogClose render={<Button variant="outline" />}>
              <X />
              Close
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={passwordFor != null}
        onOpenChange={(open) => {
          if (!open) {
            pendingLocations.current = [];
            passwordSubmitted.current = false;
            setPasswordFor(null);
            setPassword("");
          }
        }}
      >
        <DialogContent>
          <form
            className="grid gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              if (passwordFor == null) {
                return;
              }
              const spec = passwordFor;
              passwordSubmitted.current = true;
              void invoke("scan_kit_remember_password", { path: spec, password })
                .then(() => {
                  // Closing the dialog mid-invoke cleared the pending list —
                  // indexing now would resurrect a location the user cancelled.
                  if (!passwordSubmitted.current) {
                    return;
                  }
                  setPassword("");
                  setPasswordFor(null);
                  const rest = pendingLocations.current;
                  pendingLocations.current = [];
                  return indexMany([spec, ...rest]);
                })
                .catch((error: unknown) => notifyError(error));
            }}
          >
            <DialogHeader>
              <DialogTitle>Password</DialogTitle>
              <DialogDescription>
                {passwordFor?.replace(/^([a-z][a-z0-9+.-]*:\/\/)[^@/]*@/i, "$1") ?? ""} needs a password. It stays in memory for this session.
              </DialogDescription>
            </DialogHeader>
            <Input
              type="password"
              autoComplete="off"
              aria-label="Password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
            <DialogFooter>
              <DialogClose render={<Button variant="outline" />}>
                <X />
                Cancel
              </DialogClose>
              <Button type="submit">
                <FolderOpen />
                Open
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
      <Toaster />
    </div>
  );
}
