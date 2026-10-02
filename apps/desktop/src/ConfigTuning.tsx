import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
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
  const workflow = catalog?.workflows.find((item) => item.id === workflowId) ?? catalog?.workflows[0];

  useEffect(() => {
    let cancel = false;
    invoke<Catalog>("scan_kit_config_catalog")
      .then(async (loaded) => {
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
        const opened = await invoke<Opened>("scan_kit_config_open", { path: loaded.config_dir });
        if (cancel || opened.path == null) {
          return;
        }
        setConfigDir(opened.path);
        setFiles(opened.files);
        const preferred = opened.files.find((name) => name.endsWith("devices.xml")) ?? opened.files[0];
        if (preferred != null) {
          setFile(preferred);
          await loadFile(joinPath(opened.path, preferred), cancel);
        }
      })
      .catch((caught: unknown) => {
        if (!cancel) {
          notifyError(caught);
        }
      });
    return () => {
      cancel = true;
    };
  }, []);

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
    setBusy(true);
    try {
      const opened = await invoke<Opened>("scan_kit_config_open", { path: selected });
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
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-auto p-3">
      <div className="flex flex-wrap items-end gap-2">
        <Field className="min-w-64 flex-1">
          <FieldLabel>Configuration folder</FieldLabel>
          <Input value={configDir} readOnly />
        </Field>
        <Button type="button" variant="outline" onClick={() => void browse()} disabled={busy}>
          Browse
        </Button>
        <Field orientation="horizontal">
          <Checkbox
            id="hide-unused"
            checked={hideUnused}
            onCheckedChange={(next) => void toggleHide(next === true)}
          />
          <FieldLabel htmlFor="hide-unused">Hide unused map2map XML</FieldLabel>
        </Field>
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <Field className="min-w-64">
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
        <Button type="button" variant="outline" onClick={() => void save()} disabled={busy || file.length === 0}>
          Save folder
        </Button>
        {integrity != null ? <p className="text-muted-foreground text-sm">{integrity.label}</p> : null}
      </div>
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
      {preview != null ? <p className="text-sm">{preview.summary}</p> : null}
      {preview != null && preview.columns.length > 0 ? (
        <PreviewGrid columns={preview.columns} rows={preview.rows} />
      ) : null}
    </div>
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
    return (
      <FieldSet>
        <FieldLegend>{node.title}</FieldLegend>
        <FormTree nodes={node.nodes ?? []} hideUnused={hideUnused} onField={onField} onCell={onCell} />
      </FieldSet>
    );
  }
  if (node.kind === "fields") {
    const fields = (node.fields ?? []).filter((field) => !(hideUnused && field.dead));
    if (fields.length === 0) {
      return null;
    }
    return (
      <FieldGroup>
        {fields.map((field) => (
          <Field key={field.id}>
            <FieldLabel title={field.tooltip}>{field.label}</FieldLabel>
            {field.kind === "bool" ? (
              <Checkbox
                checked={field.value.toLowerCase() === "true"}
                onCheckedChange={(next) => onField(field.id, next === true ? "true" : "false")}
              />
            ) : (
              <Input value={field.value} onChange={(event) => onField(field.id, event.target.value)} />
            )}
          </Field>
        ))}
      </FieldGroup>
    );
  }
  if (node.kind === "table" && node.id != null) {
    const columns = (node.columns ?? []).filter((column) => !(hideUnused && column.dead));
    return (
      <XmlGrid
        id={node.id}
        title={node.title ?? "Table"}
        columns={columns}
        rows={node.rows ?? []}
        onCell={onCell}
      />
    );
  }
  return null;
}

function XmlGrid({
  id,
  title,
  columns,
  rows,
  onCell,
}: {
  id: string;
  title: string;
  columns: FormColumn[];
  rows: string[][];
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const theme = useMemo(() => gridTheme(), []);
  const gridColumns = useMemo<GridColumn[]>(
    () => columns.map((column) => ({ title: column.label, id: column.name, width: 140 })),
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
    return {
      kind: GridCellKind.Text,
      data: rows[row]?.[column] ?? "",
      displayData: rows[row]?.[column] ?? "",
      allowOverlay: true,
    };
  }
  return (
    <div className="flex flex-col gap-1">
      <p className="text-sm font-medium">{title}</p>
      <div ref={host} style={{ height: Math.min(360, 36 + rows.length * 28) }}>
        {width > 0 ? (
          <DataEditor
            width={width}
            height={Math.min(360, 36 + rows.length * 28)}
            columns={gridColumns}
            rows={rows.length}
            getCellContent={getCell}
            theme={theme}
            rowMarkers="number"
            onCellEdited={(cell, value) => {
              if (value.kind === GridCellKind.Text) {
                onCell(id, cell, value.data);
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
  return (
    <div ref={host} className="min-h-40" style={{ height: Math.min(420, 36 + rows.length * 28) }}>
      {width > 0 ? (
        <DataEditor
          width={width}
          height={Math.min(420, 36 + Math.max(rows.length, 1) * 28)}
          columns={gridColumns}
          rows={rows.length}
          getCellContent={getCell}
          theme={theme}
          rowMarkers="number"
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
