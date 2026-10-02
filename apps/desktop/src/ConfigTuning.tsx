import { useEffect, useMemo, useRef, useState, type WheelEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ChevronDown, ChevronRight } from "lucide-react";
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
import { notify, notifyError } from "@/notify";
import { SidePane } from "@/SidePane";

type Choice = { value: string; label: string };
type Param = {
  key: string;
  label: string;
  kind: string;
  default?: unknown;
  minimum?: number;
  step?: number;
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
  const [preview, setPreview] = useState<TuneResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const openSeq = useRef(0);
  const dirtyRef = useRef(false);
  dirtyRef.current = dirty;
  const sessionKey = selectedIds.join("|");
  const workflow = catalog?.workflows.find((item) => item.id === workflowId) ?? catalog?.workflows[0];

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

  async function runTune() {
    if (workflow == null || xml.length === 0) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await invoke<TuneResult>("scan_kit_config_tune", {
        workflow: workflow.id,
        xml,
        form: dirty ? form : null,
        dataDir: folder,
        sessionIds: selectedIds,
        params,
      });
      setXml(result.xml);
      setForm(result.form);
      setPreview(result);
      setDirty(result.changed || dirty);
      if (result.warnings.length > 0) {
        setError(result.warnings.join(" "));
      }
      notify(result.summary);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      notifyError(caught);
    } finally {
      setBusy(false);
    }
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
      notify("Saved the configuration and refreshed .md5 sidecars.");
    } catch (caught) {
      notifyError(caught);
    } finally {
      setBusy(false);
    }
  }

  const sessionLine =
    selectedIds.length === 0
      ? "Select sessions in Data Analysis, then tune."
      : selectedIds.length === 1
        ? `Tuning ${selectedIds[0]}`
        : `Tuning ${selectedIds.length} sessions`;

  return (
    <SidePane
      main={
        <div
          data-form-scroll
          className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-x-hidden overflow-y-auto p-3"
          style={{ overflowAnchor: "none" }}
        >
          {preview != null ? <p className="text-sm">{preview.summary}</p> : null}
          {preview != null && preview.columns.length > 0 ? (
            <PreviewGrid columns={preview.columns} rows={preview.rows} />
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
      }
      side={
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <Field>
        <FieldLabel>Configuration folder</FieldLabel>
        <div className="flex gap-2">
          <Input value={configDir} readOnly />
          <Button type="button" variant="outline" onClick={() => void browse()} disabled={busy}>
            Browse
          </Button>
        </div>
      </Field>
      <Field>
        <FieldLabel>File</FieldLabel>
        <Select
          items={files.map((name) => ({ value: name, label: name }))}
          value={file}
          onValueChange={(value) => {
            if (value != null) {
              void chooseFile(value);
            }
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Choose an XML file" />
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
      </Field>
      {integrity != null ? <p className="text-muted-foreground text-sm">{integrity.label}</p> : null}
      <Field orientation="horizontal">
        <Checkbox
          id="hide-unused"
          checked={hideUnused}
          onCheckedChange={(next) => void toggleHide(next === true)}
        />
        <FieldLabel htmlFor="hide-unused">Hide unused map2map XML</FieldLabel>
      </Field>
      <div className="flex gap-2">
        <Button type="button" variant="outline" onClick={() => void save()} disabled={busy || file.length === 0}>
          Save folder
        </Button>
        <Button type="button" variant="outline" onClick={() => void revert()} disabled={busy || !dirty}>
          Revert
        </Button>
      </div>
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
          <p className="text-muted-foreground text-sm">
            {sessionLine}
            {folder.length > 0 ? ` · ${folder}` : ""}
          </p>
          {(workflow?.params ?? []).filter((param) => shown(param, params)).map((param) => (
            <ParamField
              key={param.key}
              param={param}
              value={params[param.key]}
              onChange={(value) => setParams((current) => ({ ...current, [param.key]: value }))}
            />
          ))}
          <Button type="button" onClick={() => void runTune()} disabled={busy || xml.length === 0}>
            Tune
          </Button>
          {error != null ? <FieldError>{error}</FieldError> : null}
        </FieldGroup>
      </FieldSet>
        </div>
      }
    />
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
        <FieldLabel>{param.label}</FieldLabel>
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
            <ToggleGroupItem key={choice.value} value={choice.value} className="min-w-0 flex-1 cursor-pointer">
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
        <FieldLabel>{param.label}</FieldLabel>
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
                <SelectItem key={choice.value} value={choice.value}>
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
      <FieldLabel>{param.label}</FieldLabel>
      <Input
        value={value == null ? "" : String(value)}
        onChange={(event) => {
          const text = event.target.value;
          const number = Number(text);
          onChange(Number.isFinite(number) && text.trim() !== "" ? number : text);
        }}
      />
    </Field>
  );
}

function joinPath(dir: string, name: string): string {
  const sep = dir.includes("\\") ? "\\" : "/";
  return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`;
}
