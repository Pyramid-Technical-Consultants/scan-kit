import { useEffect, useMemo, useRef, useState, type WheelEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  ChevronDown,
  ChevronRight,
  CircleCheck,
  Eye,
  EyeOff,
  FileQuestion,
  FolderOpen,
  Save,
  TriangleAlert,
  Undo2,
} from "lucide-react";
import {
  DataEditor,
  GridCellKind,
  getDefaultTheme,
  type GridCell,
  type GridColumn,
  type Item,
  type Theme,
} from "@glideapps/glide-data-grid";
import "@glideapps/glide-data-grid/dist/index.css";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Toggle } from "@/components/ui/toggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { notify, notifyError, notifySaved } from "@/notify";
import { SidePane } from "@/SidePane";

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
type Workflow = { id: string; name: string; description: string; params: Param[] };
type Catalog = {
  workflows: Workflow[];
  config_dir: string | null;
  hide_unused: boolean;
};
type Integrity = { status: string; label: string };
type FormField = {
  id: string;
  label: string;
  kind: string;
  value: string;
  dead?: boolean;
  tooltip?: string;
};
type FormColumn = { name: string; label: string; dead?: boolean; tooltip?: string };
type FormNode = {
  kind: string;
  title?: string;
  dead?: boolean;
  nodes?: FormNode[];
  fields?: FormField[];
  id?: string;
  indices?: number[];
  columns?: FormColumn[];
  rows?: string[][];
  collapsible?: boolean;
};
type FormDoc = { root: string; nodes: FormNode[] };
type Opened = { path: string | null; files: string[]; hide_unused: boolean };
type Loaded = { path: string; xml: string; form: FormDoc; integrity: Integrity };
type TuneResult = {
  xml: string;
  form: FormDoc;
  summary: string;
  warnings: string[];
  columns: string[];
  rows: string[][];
  changed: boolean;
};
type HeldPreview = { stamp: string; result: TuneResult };

const PREVIEW_MS = 200;

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

function shown(param: Param, values: Record<string, unknown>): boolean {
  if (param.visible_when == null) {
    return true;
  }
  return Object.entries(param.visible_when).every(([key, allowed]) =>
    allowed.some((item) => item === values[key]),
  );
}

function shortChoices(param: Param): boolean {
  const labels = param.choices?.map((choice) => choice.label) ?? [];
  return (
    labels.length >= 2 &&
    labels.length <= 3 &&
    labels.every((label) => label.length >= 1 && label.length <= 10)
  );
}

function tokenColor(name: string, percent?: number): string {
  const probe = document.createElement("span");
  probe.style.color =
    percent == null
      ? `var(${name})`
      : `color-mix(in oklch, var(${name}) ${percent}%, transparent)`;
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
  const border = tokenColor("--border");
  const wash = tokenColor("--muted");
  return {
    ...base,
    accentColor: foreground,
    accentFg: tokenColor("--background"),
    accentLight: tokenColor("--foreground", 16),
    textDark: foreground,
    textMedium: muted,
    textLight: muted,
    textBubble: foreground,
    textHeader: foreground,
    textHeaderSelected: tokenColor("--background"),
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
    roundingRadius: 4,
  };
}

function mapNodes(nodes: FormNode[], visit: (node: FormNode) => FormNode): FormNode[] {
  return nodes.map((node) => {
    const next = node.nodes == null ? node : { ...node, nodes: mapNodes(node.nodes, visit) };
    return visit(next);
  });
}

function editField(nodes: FormNode[], id: string, value: string): FormNode[] {
  return mapNodes(nodes, (node) => {
    if (node.fields == null) {
      return node;
    }
    return {
      ...node,
      fields: node.fields.map((field) => (field.id === id ? { ...field, value } : field)),
    };
  });
}

export function fieldBoxClass(field: { kind: string; value: string }): string {
  if (field.kind === "bool") {
    return "w-fit";
  }
  if (field.kind === "int" || field.kind === "float") {
    return "w-28";
  }
  if (field.value.length > 40) {
    return "w-full max-w-lg";
  }
  if (field.value.length > 16) {
    return "w-56";
  }
  return "w-36";
}

function gridHeight(rows: number): number {
  return Math.min(360, 36 + Math.max(rows, 1) * 28);
}

function forwardWheel(event: WheelEvent<HTMLDivElement>) {
  if (event.deltaY === 0) {
    return;
  }
  const scroller = event.currentTarget.querySelector(".dvn-scroller");
  const pane = event.currentTarget.closest("[data-form-scroll]");
  if (!(pane instanceof HTMLElement) || !(scroller instanceof HTMLElement)) {
    return;
  }
  const max = scroller.scrollHeight - scroller.clientHeight;
  const atTop = scroller.scrollTop <= 0;
  const atBottom = scroller.scrollTop >= max - 1;
  const stuck = max <= 1 || (event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom);
  if (!stuck) {
    return;
  }
  event.preventDefault();
  pane.scrollTop += event.deltaY;
}

export function sourceColumn(
  columns: readonly { dead?: boolean }[],
  visible: number,
  hideUnused: boolean,
): number {
  if (!hideUnused) {
    return visible;
  }
  let seen = 0;
  for (let index = 0; index < columns.length; index += 1) {
    if (columns[index]?.dead) {
      continue;
    }
    if (seen === visible) {
      return index;
    }
    seen += 1;
  }
  return visible;
}

function editCell(nodes: FormNode[], id: string, row: number, column: number, value: string): FormNode[] {
  return mapNodes(nodes, (node) => {
    if (node.kind !== "table" || node.id !== id || node.rows == null) {
      return node;
    }
    const rows = node.rows.map((cells, index) => {
      if (index !== row) {
        return cells;
      }
      const next = cells.slice();
      next[column] = value;
      return next;
    });
    return { ...node, rows };
  });
}

export function ConfigTuning({
  folder,
  selectedIds,
}: {
  folder: string;
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
  const openSeq = useRef(0);
  const dirtyRef = useRef(false);
  dirtyRef.current = dirty;
  const sessionKey = selectedIds.join("|");
  const workflow = catalog?.workflows.find((item) => item.id === workflowId) ?? catalog?.workflows[0];
  const previewStamp = JSON.stringify({
    workflow: workflow?.id ?? "",
    params,
    sessions: sessionKey,
    folder,
    xml,
    form: dirty ? form : null,
  });
  const matched = preview != null && preview.stamp === previewStamp ? preview.result : null;

  useEffect(() => {
    const seq = ++openSeq.current;
    let cancel = false;
    void (async () => {
      if (catalog == null) {
        const loaded = await invoke<Catalog>("scan_kit_config_catalog");
        if (cancel || seq !== openSeq.current) {
          return;
        }
        setCatalog(loaded);
        const first = loaded.workflows[0];
        if (first != null) {
          setWorkflowId(first.id);
          setParams(defaultsOf(first));
        }
        setHideUnused(loaded.hide_unused);
      }
      if (dirtyRef.current && !window.confirm("Discard unsaved edits in this file?")) {
        return;
      }
      const opened = await invoke<Opened>("scan_kit_config_open", {
        dataDir: folder,
        sessionId: selectedIds[0] ?? "",
      });
      if (cancel || seq !== openSeq.current || opened.path == null) {
        return;
      }
      setConfigDir(opened.path);
      setFiles(opened.files);
      const preferred = opened.files.find((name) => name.endsWith("devices.xml")) ?? opened.files[0];
      if (preferred != null) {
        setFile(preferred);
        await loadFile(joinPath(opened.path, preferred), cancel);
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
    if (workflow == null || xml.length === 0 || selectedIds.length === 0 || folder.length === 0) {
      return;
    }
    let cancel = false;
    const stamp = previewStamp;
    const timer = window.setTimeout(() => {
      setPreviewing(true);
      void invoke<TuneResult>("scan_kit_config_tune", {
        workflow: workflow.id,
        xml,
        form: dirty ? form : null,
        dataDir: folder,
        sessionIds: selectedIds,
        params,
      })
        .then((result) => {
          if (!cancel) {
            setPreview({ stamp, result });
            setError(null);
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
          if (!cancel) {
            setPreviewing(false);
          }
        });
    }, PREVIEW_MS);
    return () => {
      cancel = true;
      window.clearTimeout(timer);
    };
  }, [previewStamp]);

  async function loadFile(path: string, cancel = false) {
    const loaded = await invoke<Loaded>("scan_kit_config_form", { path });
    if (cancel) {
      return;
    }
    setXml(loaded.xml);
    setForm(loaded.form);
    setIntegrity(loaded.integrity);
    setDirty(false);
    setPreview(null);
  }

  async function browse() {
    const selected = await open({ directory: true, multiple: false, title: "Configuration folder" });
    if (typeof selected !== "string") {
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
      setFile(preferred);
      if (preferred.length > 0 && opened.path != null) {
        await loadFile(joinPath(opened.path, preferred));
      } else {
        setXml("");
        setForm(null);
        setIntegrity(null);
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
    setFile(rel);
    try {
      await loadFile(joinPath(configDir, rel));
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
    if (!window.confirm(applyPrompt(workflow, params, selectedIds))) {
      return;
    }
    setXml(matched.xml);
    setForm(matched.form);
    setDirty(true);
    notify(matched.summary);
  }

  async function revert() {
    if (configDir.length === 0 || file.length === 0 || !dirty) {
      return;
    }
    if (!window.confirm("Discard unsaved edits in this file?")) {
      return;
    }
    try {
      await loadFile(joinPath(configDir, file));
    } catch (caught) {
      notifyError(caught);
    }
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

  const sessionLine =
    selectedIds.length === 1
      ? `Session ${selectedIds[0]}`
      : selectedIds.length > 1
        ? `${selectedIds.length} sessions`
        : "";
  const previewStatus =
    selectedIds.length === 0
      ? "Select a session to preview proposed values."
      : xml.length === 0
        ? "Open devices.xml to preview proposed values."
        : matched != null
          ? matched.summary
          : "Previewing proposed values.";

  const fileItems = files.map((name) => ({ value: name, label: name }));
  return (
    <SidePane
      main={
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center gap-2 border-b px-3 py-2">
            <Select
              items={fileItems}
              value={file}
              onValueChange={(value) => {
                if (value != null) {
                  void chooseFile(value);
                }
              }}
            >
              <SelectTrigger size="sm" className="w-56" aria-label="XML file">
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
            <div className="ml-auto flex items-center gap-1">
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
              </Toggle>
              <Button
                type="button"
                variant="outline"
                size="icon-sm"
                aria-label="Configuration folder"
                title={configDir.length > 0 ? configDir : "Open configuration folder"}
                disabled={busy}
                onClick={() => void browse()}
              >
                <FolderOpen />
              </Button>
              <Button
                type="button"
                variant="outline"
                size="icon-sm"
                aria-label="Revert"
                title="Revert"
                disabled={busy || !dirty}
                onClick={() => void revert()}
              >
                <Undo2 />
              </Button>
              <Button
                type="button"
                variant={dirty ? "default" : "outline"}
                size="icon-sm"
                aria-label="Save folder"
                title="Save folder"
                disabled={busy || file.length === 0}
                onClick={() => void save()}
              >
                <Save />
              </Button>
            </div>
          </div>
          <div
            data-form-scroll
            className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-x-hidden overflow-y-auto p-3"
            style={{ overflowAnchor: "none" }}
          >
            {matched != null ? <p className="text-sm">{matched.summary}</p> : null}
            {matched != null && matched.columns.length > 0 ? (
              <PreviewGrid columns={matched.columns} rows={matched.rows} />
            ) : null}
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
        </div>
      }
      side={
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <FieldSet>
        <FieldLegend>Auto tune</FieldLegend>
        <FieldGroup>
          <Field>
            <FieldLabel>Workflow</FieldLabel>
            <Select
              items={(catalog?.workflows ?? []).map((item) => ({ value: item.id, label: item.name }))}
              value={workflow?.id ?? ""}
              onValueChange={(value) => {
                const next = catalog?.workflows.find((item) => item.id === value);
                if (next == null) {
                  return;
                }
                setWorkflowId(next.id);
                setParams(defaultsOf(next));
                setPreview(null);
              }}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Workflow" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {(catalog?.workflows ?? []).map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </Field>
          {workflow != null ? <p className="text-muted-foreground text-sm">{workflow.description}</p> : null}
          {sessionLine.length > 0 ? <p className="text-muted-foreground text-sm">{sessionLine}</p> : null}
          {(workflow?.params ?? []).filter((param) => shown(param, params)).map((param) => (
            <ParamField
              key={param.key}
              param={param}
              value={params[param.key]}
              onChange={(value) => setParams((current) => ({ ...current, [param.key]: value }))}
            />
          ))}
          {error != null ? <FieldError>{error}</FieldError> : <p className="text-sm">{previewStatus}</p>}
          {matched != null && matched.warnings.length > 0 ? (
            <FieldError>{matched.warnings.join(" ")}</FieldError>
          ) : null}
          <Button
            type="button"
            onClick={applyPreview}
            disabled={busy || previewing || matched == null || !matched.changed}
            title="Apply the preview to devices.xml. The file stays unsaved until you save."
          >
            Apply
          </Button>
        </FieldGroup>
      </FieldSet>
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

function FormTree({
  nodes,
  hideUnused,
  onField,
  onCell,
}: {
  nodes: FormNode[];
  hideUnused: boolean;
  onField: (id: string, value: string) => void;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {nodes.map((node, index) => (
        <FormNodeView
          key={`${node.kind}-${node.id ?? node.title ?? index}`}
          node={node}
          hideUnused={hideUnused}
          onField={onField}
          onCell={onCell}
        />
      ))}
    </div>
  );
}

function FormNodeView({
  node,
  hideUnused,
  onField,
  onCell,
}: {
  node: FormNode;
  hideUnused: boolean;
  onField: (id: string, value: string) => void;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  if (hideUnused && node.dead) {
    return null;
  }
  if (node.kind === "section") {
    return <FormSection node={node} hideUnused={hideUnused} onField={onField} onCell={onCell} />;
  }
  if (node.kind === "fields") {
    const fields = (node.fields ?? []).filter((field) => !(hideUnused && field.dead));
    if (fields.length === 0) {
      return null;
    }
    return (
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        {fields.map((field) =>
          field.kind === "bool" ? (
            <Field key={field.id} orientation="horizontal" className={fieldBoxClass(field)}>
              <Checkbox
                id={field.id}
                checked={field.value.toLowerCase() === "true"}
                onCheckedChange={(next) => onField(field.id, next === true ? "true" : "false")}
              />
              <FieldLabel htmlFor={field.id} title={field.tooltip}>
                {field.label}
              </FieldLabel>
            </Field>
          ) : (
            <Field key={field.id} className={fieldBoxClass(field)}>
              <FieldLabel title={field.tooltip}>{field.label}</FieldLabel>
              <Input value={field.value} onChange={(event) => onField(field.id, event.target.value)} />
            </Field>
          ),
        )}
      </div>
    );
  }
  if (node.kind === "table" && node.id != null) {
    return (
      <XmlGrid
        id={node.id}
        title={node.title ?? "Table"}
        columns={node.columns ?? []}
        rows={node.rows ?? []}
        hideUnused={hideUnused}
        onCell={onCell}
      />
    );
  }
  return null;
}

function FormSection({
  node,
  hideUnused,
  onField,
  onCell,
}: {
  node: FormNode;
  hideUnused: boolean;
  onField: (id: string, value: string) => void;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  const [open, setOpen] = useState(!node.collapsible);
  const body =
    !node.collapsible || open ? (
      <CardContent>
        <FormTree nodes={node.nodes ?? []} hideUnused={hideUnused} onField={onField} onCell={onCell} />
      </CardContent>
    ) : null;
  return (
    <Card size="sm">
      <CardHeader>
        {node.collapsible ? (
          <Button
            type="button"
            variant="ghost"
            className="w-full justify-start"
            aria-expanded={open}
            onClick={() => setOpen((current) => !current)}
          >
            {open ? <ChevronDown /> : <ChevronRight />}
            {node.title}
          </Button>
        ) : (
          <CardTitle>{node.title}</CardTitle>
        )}
      </CardHeader>
      {body}
    </Card>
  );
}

function XmlGrid({
  id,
  title,
  columns,
  rows,
  hideUnused,
  onCell,
}: {
  id: string;
  title: string;
  columns: FormColumn[];
  rows: string[][];
  hideUnused: boolean;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const theme = useMemo(() => gridTheme(), []);
  const shown = useMemo(
    () =>
      columns
        .map((column, index) => ({ column, index }))
        .filter((item) => !(hideUnused && item.column.dead)),
    [columns, hideUnused],
  );
  const gridColumns = useMemo<GridColumn[]>(
    () =>
      shown.map((item) => ({
        title: item.column.label,
        id: item.column.name,
        width: Math.min(180, Math.max(88, item.column.label.length * 9 + 28)),
      })),
    [shown],
  );
  useEffect(() => {
    const node = host.current;
    if (node == null) {
      return;
    }
    const measure = () => setWidth(node.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  function getCell([column, row]: Item): GridCell {
    const source = shown[column]?.index ?? column;
    const text = rows[row]?.[source] ?? "";
    return {
      kind: GridCellKind.Text,
      data: text,
      displayData: text,
      allowOverlay: true,
    };
  }
  const height = gridHeight(rows.length);
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <p className="text-muted-foreground text-sm">{title}</p>
      <div
        ref={host}
        className="w-full min-w-0 overflow-hidden"
        style={{ height, overflowAnchor: "none" }}
        onWheel={forwardWheel}
      >
        {width > 0 ? (
          <DataEditor
            width={width}
            height={height}
            columns={gridColumns}
            rows={rows.length}
            getCellContent={getCell}
            theme={theme}
            rowMarkers="number"
            scrollToActiveCell={false}
            onCellEdited={(cell, value) => {
              if (value.kind === GridCellKind.Text) {
                const source = sourceColumn(columns, cell[0], hideUnused);
                onCell(id, [source, cell[1]], value.data);
              }
            }}
          />
        ) : null}
      </div>
    </div>
  );
}

function PreviewGrid({ columns, rows }: { columns: string[]; rows: string[][] }) {
  const host = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const theme = useMemo(() => gridTheme(), []);
  const gridColumns = useMemo<GridColumn[]>(
    () => columns.map((title) => ({ title, id: title, width: 150 })),
    [columns],
  );
  useEffect(() => {
    const node = host.current;
    if (node == null) {
      return;
    }
    const measure = () => setWidth(node.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  function getCell([column, row]: Item): GridCell {
    const text = rows[row]?.[column] ?? "";
    return { kind: GridCellKind.Text, data: text, displayData: text, allowOverlay: false, readonly: true };
  }
  const height = gridHeight(rows.length);
  return (
    <div
      ref={host}
      className="w-full min-w-0 overflow-hidden"
      style={{ height, overflowAnchor: "none" }}
      onWheel={forwardWheel}
    >
      {width > 0 ? (
        <DataEditor
          width={width}
          height={height}
          columns={gridColumns}
          rows={rows.length}
          getCellContent={getCell}
          theme={theme}
          rowMarkers="number"
          scrollToActiveCell={false}
        />
      ) : null}
    </div>
  );
}

function ParamField({
  param,
  value,
  onChange,
}: {
  param: Param;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (param.kind === "choice" && shortChoices(param)) {
    return (
      <Field>
        <FieldLabel title={param.tooltip}>{param.label}</FieldLabel>
        <ToggleGroup
          variant="outline"
          spacing={0}
          size="sm"
          className="w-full"
          value={typeof value === "string" ? [value] : []}
          onValueChange={(next) => {
            const current = typeof value === "string" ? value : "";
            const picked = next.find((item) => item !== current) ?? next[0];
            if (picked != null) {
              onChange(picked);
            }
          }}
        >
          {(param.choices ?? []).map((choice) => (
            <ToggleGroupItem
              key={choice.value}
              value={choice.value}
              title={choice.tooltip}
              className="min-w-0 flex-1 cursor-pointer"
            >
              {choice.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>
    );
  }
  if (param.kind === "choice") {
    return (
      <Field>
        <FieldLabel title={param.tooltip}>{param.label}</FieldLabel>
        <Select
          items={(param.choices ?? []).map((choice) => ({ value: choice.value, label: choice.label }))}
          value={typeof value === "string" ? value : ""}
          onValueChange={(next) => {
            if (next != null) {
              onChange(next);
            }
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {(param.choices ?? []).map((choice) => (
                <SelectItem key={choice.value} value={choice.value} title={choice.tooltip}>
                  {choice.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </Field>
    );
  }
  return (
    <Field>
      <FieldLabel title={param.tooltip}>{param.label}</FieldLabel>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          inputMode="decimal"
          min={param.minimum}
          step={param.step ?? "any"}
          title={param.tooltip}
          value={value == null ? "" : String(value)}
          onChange={(event) => {
            const text = event.target.value;
            const number = Number(text);
            onChange(Number.isFinite(number) && text.trim() !== "" ? number : text);
          }}
        />
        {param.suffix != null ? <span className="text-muted-foreground text-sm">{param.suffix}</span> : null}
      </div>
    </Field>
  );
}

function joinPath(dir: string, name: string): string {
  const sep = dir.includes("\\") ? "\\" : "/";
  return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`;
}
