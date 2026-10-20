import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  Check,
  CircleCheck,
  Eye,
  EyeOff,
  FileQuestion,
  FolderOpen,
  Save,
  TriangleAlert,
} from "lucide-react";
import {
  DataEditor,
  GridCellKind,
  type GridCell,
  type GridColumn,
  type Item,
} from "@glideapps/glide-data-grid";
import { gridTheme, toneColors, useResizableColumns, type CellTone } from "@/grid-theme";
import "@glideapps/glide-data-grid/dist/index.css";

import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Toggle } from "@/components/ui/toggle";
import { FieldError, FieldLegend, FieldSet } from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CatalogField, catalogShown } from "@/CatalogField";
import {
  editCell,
  editField,
  FormTree,
  forwardWheel,
  type FormDoc,
} from "@/ConfigForm";
import { DiffView, type DiffSession } from "@/diff-view";
import { notify, notifyError, notifySaved } from "@/notify";
import { optionIcon } from "@/option-icons";
import { usePageLoad } from "@/page-load";
import { SessionList, type SessionListItem } from "@/session-list";
import { SidePane } from "@/SidePane";
import { Splitter } from "@/splitter";

type Choice = { value: string; label: string; tooltip?: string };
type Param = {
  key: string;
  label: string;
  kind: string;
  default?: unknown;
  minimum?: number;
  step?: number;
  suffix?: string;
  tooltip?: string;
  choices?: Choice[];
  visible_when?: Record<string, string[]>;
};
type Workflow = {
  id: string;
  name: string;
  description: string;
  sessions?: boolean;
  diff?: boolean;
  params: Param[];
};
type Catalog = {
  workflows: Workflow[];
  config_dir: string | null;
  hide_unused: boolean;
};
type Integrity = { status: string; label: string };
type Opened = { path: string | null; files: string[]; hide_unused: boolean };
type Loaded = { path: string; xml: string; form: FormDoc; integrity: Integrity };
type PreviewCell = { text: string; tone?: CellTone };
type TuneResult = {
  xml: string;
  form: FormDoc;
  summary: string;
  warnings: string[];
  columns: string[];
  rows: PreviewCell[][];
  changed: boolean;
  mu_totals?: number[];
  base?: string;
  edited?: string;
  sessions?: DiffSession[];
};
type HeldPreview = { stamp: string; result: TuneResult };

const PREVIEW_MS = 200;
const PREVIEW_MIN = 96;
const PREVIEW_RATIO = 0.3;
const EDITOR_MIN = 200;

const TONE_LABEL: Record<CellTone, string> = {
  good: "improvement",
  warn: "review",
  bad: "downgrade",
  change: "will change",
};
const TONE_ORDER: readonly CellTone[] = ["good", "warn", "bad", "change"];

function applyPrompt(workflow: Workflow, params: Record<string, unknown>, ids: string[]): string {
  const source =
    ids.length === 1 ? `session ${ids[0]}` : ids.length > 1 ? `${ids.length} sessions` : "the selected sessions";
  const tail = "The configuration stays unsaved until you save.";
  if (workflow.id === "sigma_tuning") {
    const tolerance = params.sigma_tolerance_percent ?? 20;
    const headroom = params.sigma_lower_headroom_percent ?? 1;
    return `Rewrite beam sigma K0 values in devices.xml using data from ${source}?\n\nEach band is tuned to the ±${tolerance}% tolerance window with ${headroom}% lower headroom above the smallest observed sigma.\n\n${tail}`;
  }
  if (workflow.id === "position_offset_tuning") {
    const dataSource = params.data_source ?? "spot";
    return `Rewrite zero offset at iso values in devices.xml using ${dataSource} data from ${source}?\n\n${tail}`;
  }
  if (workflow.id === "config_diff") {
    return "The configuration diff is read-only.";
  }
  if (workflow.id === "ic_distance_tuning") {
    return `Rewrite source to device distance and zero offset in devices.xml using spot data from ${source}?\n\nThis assumes delivery at isocenter is correct and the chambers are mis-scaled. Review the proposed distance, then re-run Sigma Tuning.\n\n${tail}`;
  }
  const primary = String(params.primary_ic ?? "ic1").toUpperCase();
  if (params.primary_mode === "unchanged" || params.primary_mode == null) {
    return `Rewrite secondary IC K_MU values in devices.xml so they agree with ${primary} using data from ${source}?\n\nPrimary ${primary} K_MU is left unchanged.\n\n${tail}`;
  }
  return `Rewrite K_MU values in devices.xml using data from ${source}?\n\nThis changes the primary (${primary}) K_MU, which changes future delivered charge for the same CHARGE_REQ.\n\n${tail}`;
}

function defaultsOf(workflow: Workflow): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const param of workflow.params) {
    values[param.key] = param.default;
  }
  return values;
}

export { fieldBoxClass, sourceColumn } from "@/ConfigForm";

export function ConfigTuning({
  folder,
  sessions,
  selectedIds,
}: {
  folder: string;
  sessions: readonly SessionListItem[];
  selectedIds: string[];
}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [workflowId, setWorkflowId] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [configDir, setConfigDir] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [file, setFile] = useState("");
  const [xml, setXml] = useState("");
  const [form, setForm] = useState<FormDoc | null>(null);
  const [integrity, setIntegrity] = useState<Integrity | null>(null);
  const [hideUnused, setHideUnused] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [preview, setPreview] = useState<HeldPreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [waiting, setWaiting] = useState(true);
  const [opening, setOpening] = useState(false);
  usePageLoad(waiting || busy || previewing || opening);
  const openSeq = useRef(0);
  const previewGen = useRef(0);
  const autoMu = useRef<number | null>(null);
  const dirtyRef = useRef(false);
  const leftPane = useRef<HTMLDivElement>(null);
  const previewDrag = useRef(PREVIEW_RATIO);
  const [previewFrac, setPreviewFrac] = useState(PREVIEW_RATIO);
  const [excluded, setExcluded] = useState<string[]>([]);
  const tones = useMemo(() => toneColors(), []);
  useEffect(() => {
    dirtyRef.current = dirty;
  });
  const sessionKey = selectedIds.join("|");
  const workflow = catalog?.workflows.find((item) => item.id === workflowId) ?? catalog?.workflows[0];
  const picksSessions = workflow?.sessions === true;
  const sessionIds = useMemo(
    () => selectedIds.filter((id) => !excluded.includes(id)),
    [excluded, selectedIds],
  );
  const needsSessions = picksSessions && sessionIds.length === 0;
  // G2 configurations have no IC_3 ion chamber, so IC3 cannot be the primary.
  const hasIc3 = useMemo(() => /name\s*=\s*"IC_3/i.test(xml), [xml]);
  const viewParams = useMemo(
    () =>
      !hasIc3 && params.primary_ic === "ic3"
        ? { ...params, primary_ic: "ic1" }
        : params,
    [hasIc3, params],
  );
  const previewRequest = useMemo(() => {
    const isDiff = workflow?.diff === true;
    const ready =
      workflow != null &&
      folder.length > 0 &&
      (!picksSessions || sessionIds.length > 0) &&
      (isDiff ? configDir.length > 0 : xml.length > 0);
    if (!ready || workflow == null) {
      return null;
    }
    const formValue = dirty ? form : null;
    return {
      stamp: JSON.stringify({
        workflow: workflow.id,
        params: viewParams,
        sessions: sessionIds.join("|"),
        folder,
        xml,
        form: formValue,
        configDir,
        file,
      }),
      workflowId: workflow.id,
      diff: isDiff,
      xml,
      form: formValue,
      folder,
      sessionIds,
      params: viewParams,
      configDir,
      file,
    };
  }, [configDir, dirty, file, folder, form, viewParams, picksSessions, sessionIds, workflow, xml]);
  const matched = preview != null && preview.stamp === previewRequest?.stamp ? preview.result : null;

  useEffect(() => {
    let cancel = false;
    void invoke<Catalog>("scan_kit_config_catalog")
      .then((loaded) => {
        if (cancel) {
          return;
        }
        setCatalog(loaded);
        const first = loaded.workflows[0];
        if (first != null) {
          setWorkflowId(first.id);
          setParams(defaultsOf(first));
        }
        setHideUnused(loaded.hide_unused);
        setWaiting(false);
      })
      .catch((caught: unknown) => {
        if (!cancel) {
          setWaiting(false);
          notifyError(caught);
        }
      });
    return () => {
      cancel = true;
    };
  }, []);

  useEffect(() => {
    const seq = ++openSeq.current;
    let cancel = false;
    const sessionId = sessionKey.split("|")[0] ?? "";
    void (async () => {
      if (dirtyRef.current && !window.confirm("Discard unsaved edits in this file?")) {
        return;
      }
      setOpening(true);
      try {
        const opened = await invoke<Opened>("scan_kit_config_open", {
          dataDir: folder,
          sessionId,
        });
        if (cancel || seq !== openSeq.current || opened.path == null) {
          return;
        }
        setConfigDir(opened.path);
        setFiles(opened.files);
        const preferred = opened.files.find((name) => name.endsWith("devices.xml")) ?? opened.files[0];
        if (preferred != null) {
          if (await loadFile(joinPath(opened.path, preferred), () => cancel || seq !== openSeq.current)) {
            setFile(preferred);
          }
        }
      } finally {
        if (!cancel) {
          setOpening(false);
        }
      }
    })().catch((caught: unknown) => {
      if (!cancel) {
        notifyError(caught);
      }
    });
    return () => {
      cancel = true;
    };
  }, [folder, sessionKey]);

  useEffect(() => {
    const mine = previewGen.current + 1;
    previewGen.current = mine;
    if (previewRequest == null) {
      const arm = window.setTimeout(() => {
        if (previewGen.current === mine) {
          setPreviewing(false);
        }
      }, 0);
      return () => window.clearTimeout(arm);
    }
    let cancel = false;
    const request = previewRequest;
    const timer = window.setTimeout(() => {
      if (cancel || previewGen.current !== mine) {
        return;
      }
      setError(null);
      setPreviewing(true);
      void invoke<TuneResult>(request.diff ? "scan_kit_config_diff" : "scan_kit_config_tune", {
        workflow: request.workflowId,
        configDir: request.configDir,
        file: request.file,
        xml: request.xml,
        form: request.form,
        dataDir: request.folder,
        sessionIds: request.sessionIds,
        params: request.params,
      })
        .then((result) => {
          if (!cancel) {
            setPreview({ stamp: request.stamp, result });
            setError(null);
            const totals = result.mu_totals;
            const index = ["ic1", "ic2", "ic3"].indexOf(String(request.params.primary_ic ?? "ic1"));
            const total = index >= 0 ? totals?.[index] : undefined;
            if (request.workflowId === "kmu_tuning" && total != null && Number.isFinite(total)) {
              setParams((current) => {
                const held = Number(current.known_mu);
                if ((!(held > 0) || held === autoMu.current) && held !== total) {
                  autoMu.current = total;
                  return { ...current, known_mu: total };
                }
                return current;
              });
            }
          }
        })
        .catch((caught: unknown) => {
          if (!cancel) {
            const message = caught instanceof Error ? caught.message : String(caught);
            setError(message);
            notifyError(caught);
          }
        })
        .finally(() => {
          if (previewGen.current === mine) {
            setPreviewing(false);
          }
        });
    }, PREVIEW_MS);
    return () => {
      cancel = true;
      window.clearTimeout(timer);
    };
  }, [previewRequest]);

  async function loadFile(path: string, cancelled: () => boolean = () => false): Promise<boolean> {
    setOpening(true);
    try {
      const loaded = await invoke<Loaded>("scan_kit_config_form", { path });
      if (cancelled()) {
        return false;
      }
      setXml(loaded.xml);
      setForm(loaded.form);
      setIntegrity(loaded.integrity);
      setDirty(false);
      setPreview(null);
      return true;
    } finally {
      if (!cancelled()) {
        setOpening(false);
      }
    }
  }

  async function browse() {
    const selected = await open({ directory: true, multiple: false, title: "Configuration folder" });
    if (typeof selected !== "string") {
      return;
    }
    if (dirty && !window.confirm("Discard unsaved edits in this file?")) {
      return;
    }
    const seq = ++openSeq.current;
    setBusy(true);
    try {
      const opened = await invoke<Opened>("scan_kit_config_open", { path: selected });
      if (seq !== openSeq.current) {
        return;
      }
      setConfigDir(opened.path ?? "");
      setFiles(opened.files);
      const preferred = opened.files.find((name) => name.endsWith("devices.xml")) ?? opened.files[0] ?? "";
      if (preferred.length > 0 && opened.path != null) {
        if (await loadFile(joinPath(opened.path, preferred), () => seq !== openSeq.current)) {
          setFile(preferred);
        }
      } else {
        setXml("");
        setForm(null);
        setIntegrity(null);
        setDirty(false);
        setPreview(null);
      }
    } catch (caught) {
      notifyError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function chooseFile(rel: string) {
    if (dirty && !window.confirm("Discard unsaved edits in this file?")) {
      return;
    }
    const seq = ++openSeq.current;
    try {
      if (await loadFile(joinPath(configDir, rel), () => seq !== openSeq.current)) {
        setFile(rel);
      }
    } catch (caught) {
      notifyError(caught);
    }
  }

  function changeField(id: string, value: string) {
    setForm((current) => (current == null ? current : { ...current, nodes: editField(current.nodes, id, value) }));
    setDirty(true);
  }

  function changeCell(id: string, cell: Item, value: string) {
    setForm((current) =>
      current == null
        ? current
        : { ...current, nodes: editCell(current.nodes, id, cell[1], cell[0], value) },
    );
    setDirty(true);
  }

  async function toggleHide(next: boolean) {
    setHideUnused(next);
    try {
      await invoke("scan_kit_config_hide", { hideUnused: next });
    } catch (caught) {
      notifyError(caught);
    }
  }

  function applyPreview() {
    if (workflow == null || matched == null || !matched.changed) {
      return;
    }
    if (!window.confirm(applyPrompt(workflow, viewParams, sessionIds))) {
      return;
    }
    setXml(matched.xml);
    setForm(matched.form);
    setDirty(true);
    notify(matched.summary);
  }

  async function save() {
    if (configDir.length === 0 || file.length === 0) {
      return;
    }
    const selected = await open({
      directory: true,
      multiple: false,
      title: "Save configuration folder",
      defaultPath: configDir,
    });
    if (typeof selected !== "string") {
      return;
    }
    setBusy(true);
    try {
      let nextXml = xml;
      let nextForm = form;
      if (dirty && form != null) {
        const applied = await invoke<{ xml: string; form: FormDoc }>("scan_kit_config_apply", {
          xml,
          form,
        });
        nextXml = applied.xml;
        nextForm = applied.form;
      }
      await invoke("scan_kit_config_save", {
        source: configDir,
        dest: selected,
        files: [{ rel: file, xml: nextXml }],
        hideUnused,
      });
      setConfigDir(selected);
      setXml(nextXml);
      setForm(nextForm);
      setDirty(false);
      const opened = await invoke<Opened>("scan_kit_config_open", { path: selected });
      setFiles(opened.files);
      if (nextForm != null) {
        const loaded = await invoke<Loaded>("scan_kit_config_form", {
          path: joinPath(selected, file),
        });
        setIntegrity(loaded.integrity);
      }
      notifySaved(selected, {
        title: "Configuration saved",
        description: "Refreshed .md5 sidecars.",
        folder: true,
      });
    } catch (caught) {
      notifyError(caught);
    } finally {
      setBusy(false);
    }
  }

  const isDiff = workflow?.diff === true;
  const previewStatus =
    folder.length === 0
      ? "No session data folder."
      : needsSessions
        ? "Select a session to preview proposed values."
        : isDiff && configDir.length === 0
          ? "Open a configuration folder to diff."
          : !isDiff && xml.length === 0
            ? "Open devices.xml to preview proposed values."
            : matched != null
              ? matched.summary
              : isDiff
                ? "Diffing configurations."
                : "Previewing proposed values.";
  const shownError = previewRequest != null ? error : null;

  const fileItems = files.map((name) => ({ value: name, label: name }));
  return (
    <SidePane
      main={
        <div ref={leftPane} className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center gap-2 border-b px-3 py-2">
            <span
              className={`min-w-0 flex-1 truncate text-sm ${configDir.length > 0 ? "" : "text-muted-foreground"}`}
              title={configDir.length > 0 ? configDir : "No configuration folder open"}
            >
              {configDir.length > 0 ? configDir : "No configuration folder open"}
            </span>
            <Select
              items={fileItems}
              value={file}
              onValueChange={(value) => {
                if (value != null) {
                  void chooseFile(value);
                }
              }}
            >
              <SelectTrigger
                size="sm"
                className="w-56 shrink-0"
                aria-label="XML file"
                disabled={busy || opening}
              >
                <SelectValue placeholder="XML file" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {files.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2">
              {integrity == null ? null : <IntegrityMark status={integrity.status} label={integrity.label} />}
              <Separator orientation="vertical" />
              <Toggle
                variant="outline"
                size="sm"
                pressed={hideUnused}
                aria-label="Hide unused map2map XML"
                title="Hide unused map2map XML"
                onPressedChange={(next) => void toggleHide(next)}
              >
                {hideUnused ? <EyeOff /> : <Eye />}
                Hide Unused
              </Toggle>
              <Button
                type="button"
                variant="outline"
                size="sm"
                title={configDir.length > 0 ? configDir : "Open configuration folder"}
                disabled={busy}
                onClick={() => void browse()}
              >
                <FolderOpen />
                Browse
              </Button>
              <Button
                type="button"
                variant={dirty ? "default" : "outline"}
                size="sm"
                aria-label="Save folder"
                title="Save folder"
                disabled={busy || file.length === 0}
                onClick={() => void save()}
              >
                <Save />
                Save
              </Button>
            </div>
          </div>
          <div
            data-form-scroll
            className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-x-hidden overflow-y-auto p-3"
            style={{ overflowAnchor: "none" }}
          >
            {form != null ? (
              <FormTree
                nodes={form.nodes}
                hideUnused={hideUnused}
                onField={changeField}
                onCell={changeCell}
              />
            ) : (
              <p className="text-muted-foreground text-sm">Open a configuration folder to edit devices.xml.</p>
            )}
          </div>
          <Splitter
            orientation="horizontal"
            label="Resize preview"
            now={previewFrac * 100}
            min={10}
            onStart={() => {
              previewDrag.current = previewFrac;
            }}
            onMove={(delta) => {
              const parent = leftPane.current?.getBoundingClientRect().height ?? window.innerHeight;
              setPreviewFrac(clampPreviewFrac(parent, previewDrag.current - delta / parent));
            }}
          />
          <div style={{ flexBasis: `${previewFrac * 100}%` }} className="flex min-h-0 shrink-0 grow-0 flex-col">
            <div className="flex shrink-0 items-center gap-2 border-b px-3 py-2">
              <p
                className={
                  shownError != null
                    ? "text-destructive min-w-0 flex-1 truncate text-sm"
                    : matched != null
                      ? "min-w-0 flex-1 truncate text-sm"
                      : "text-muted-foreground min-w-0 flex-1 truncate text-sm"
                }
                title={shownError ?? matched?.summary ?? previewStatus}
              >
                {shownError ?? matched?.summary ?? previewStatus}
              </p>
              {isDiff ? null : (
                <Button
                  type="button"
                  size="sm"
                  className="shrink-0"
                  onClick={applyPreview}
                  disabled={busy || previewing || matched == null || !matched.changed}
                  title="Apply the preview to devices.xml. The file stays unsaved until you save."
                >
                  <Check />
                  Apply
                </Button>
              )}
            </div>
            {!isDiff && matched != null && matched.columns.length > 0 ? (
              <div className="text-muted-foreground flex shrink-0 items-center gap-3 border-b px-3 py-1 text-xs">
                {TONE_ORDER.map((tone) => (
                  <span key={tone} className="flex items-center gap-1">
                    <span
                      className="border-foreground/25 size-2.5 rounded-sm border"
                      style={{ backgroundColor: tones[tone].fg }}
                    />
                    {TONE_LABEL[tone]}
                  </span>
                ))}
              </div>
            ) : null}
            {matched != null && matched.warnings.length > 0 ? (
              <FieldError className="shrink-0 px-3 pt-1">{matched.warnings.join(" ")}</FieldError>
            ) : null}
            {isDiff && matched != null && matched.sessions != null ? (
              <DiffView
                result={{ base: matched.base ?? "", sessions: matched.sessions }}
              />
            ) : null}
            {!isDiff && matched != null && matched.columns.length > 0 ? (
              <div data-form-scroll className="min-h-0 flex-1 overflow-y-auto p-3">
                <PreviewGrid columns={matched.columns} rows={matched.rows} />
              </div>
            ) : null}
          </div>
        </div>
      }
      side={
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
          <FieldSet className="min-w-0 gap-2 rounded-lg border border-border p-3">
            <FieldLegend variant="label">Workflow</FieldLegend>
            <ButtonSegmentGroup
              orientation="vertical"
              label="Workflow"
              options={(catalog?.workflows ?? []).map((item) => ({
                id: item.id,
                label: item.name,
                detail: item.description,
              }))}
              value={workflow?.id ?? ""}
              onChange={(name) => {
                const next = catalog?.workflows.find((item) => item.name === name);
                if (next == null) {
                  return;
                }
                setWorkflowId(next.id);
                setParams(defaultsOf(next));
                setPreview(null);
              }}
            />
            {workflow != null ? <p className="text-muted-foreground text-sm">{workflow.description}</p> : null}
            {(workflow?.params ?? [])
              .filter((param) => catalogShown(param, viewParams))
              .map((param) => {
                const shown =
                  param.key === "primary_ic" && !hasIc3
                    ? { ...param, choices: param.choices?.filter((choice) => choice.value !== "ic3") }
                    : param;
                return (
                  <CatalogField
                    key={param.key}
                    param={shown}
                    value={viewParams[param.key]}
                    beside
                    icon={(label) => {
                      const Icon = optionIcon(label);
                      return Icon == null ? null : <Icon />;
                    }}
                    onChange={(value) => setParams((current) => ({ ...current, [param.key]: value }))}
                  />
                );
              })}
          </FieldSet>
          {picksSessions ? (
            <SessionList
              sessions={sessions}
              isChecked={(id) => !excluded.includes(id)}
              onCheckedChange={(id, checked) =>
                setExcluded((current) =>
                  checked ? current.filter((item) => item !== id) : [...current, id],
                )
              }
            />
          ) : null}
        </div>
      }
    />
  );
}

function IntegrityMark({ status, label }: { status: string; label: string }) {
  const Icon = status === "OK" ? CircleCheck : status === "HASH_ERR" ? TriangleAlert : FileQuestion;
  return (
    <span className="text-muted-foreground inline-flex" title={label} aria-label={label}>
      <Icon className="size-4" />
    </span>
  );
}

function PreviewGrid({ columns, rows }: { columns: string[]; rows: PreviewCell[][] }) {
  const host = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const theme = useMemo(() => gridTheme(), []);
  const tones = useMemo(() => toneColors(), []);
  const baseColumns = useMemo<GridColumn[]>(
    () => columns.map((title) => ({ title, id: title, width: 150 })),
    [columns],
  );
  const { columns: gridColumns, onColumnResize } = useResizableColumns(baseColumns);
  useEffect(() => {
    const node = host.current;
    if (node == null) {
      return;
    }
    const measure = () => setSize({ width: node.clientWidth, height: node.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  function getCell([column, row]: Item): GridCell {
    const cell = rows[row]?.[column];
    const text = cell?.text ?? "";
    const tone = cell?.tone != null ? tones[cell.tone] : undefined;
    return {
      kind: GridCellKind.Text,
      data: text,
      displayData: text,
      allowOverlay: false,
      readonly: true,
      themeOverride:
        tone == null
          ? undefined
          : { bgCell: tone.bg, textDark: tone.fg, accentLight: tone.bg },
    };
  }
  const height = size.height;
  return (
    <div
      ref={host}
      className="h-full w-full min-w-0 overflow-hidden"
      style={{ overflowAnchor: "none" }}
      onWheel={forwardWheel}
    >
      {size.width > 0 && height > 0 ? (
        <DataEditor
          width={size.width}
          height={height}
          columns={gridColumns}
          rows={rows.length}
          getCellContent={getCell}
          theme={theme}
          rowMarkers="number"
          scrollToActiveCell={false}
          onColumnResize={onColumnResize}
        />
      ) : null}
    </div>
  );
}

function clampPreviewFrac(parentHeight: number, next: number): number {
  if (!(parentHeight > 0)) {
    return next;
  }
  const min = PREVIEW_MIN / parentHeight;
  const max = Math.max(min, 1 - EDITOR_MIN / parentHeight);
  return Math.min(max, Math.max(min, next));
}

function joinPath(dir: string, name: string): string {
  const sep = dir.includes("\\") ? "\\" : "/";
  return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`;
}
