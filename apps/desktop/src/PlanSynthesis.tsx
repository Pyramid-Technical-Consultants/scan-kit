import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
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
type QuickSet = { label: string; values: Record<string, number> };
type Preset = { label: string; energies: number[] };
type Param = {
  key: string;
  label: string;
  kind: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  step?: number;
  suffix?: string;
  sub_label?: string;
  row_partner?: string;
  field_set?: string;
  choices?: Choice[];
  visible_when?: Record<string, Array<string | number | boolean>>;
  quick_sets?: QuickSet[];
  presets?: Preset[];
};
type Template = { id: string; name: string; description: string; params: Param[] };
type Catalog = { templates: Template[]; save_dir: string | null };
type PlanResult = {
  summary: string;
  filename: string;
  columns: string[];
  rows: string[][];
  csv: string;
};

const FIELD_SETS: Array<[string, string]> = [
  ["source", "Source"],
  ["energy", "Energy"],
  ["geometry", "Geometry"],
  ["weight", "Weight"],
];

function defaultsOf(template: Template): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const param of template.params) {
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

function energyText(energy: number): string {
  return `${String(energy)} MeV`;
}

function parentDir(path: string): string {
  return path.replace(/[/\\][^/\\]+$/, "");
}

function joinPath(dir: string, name: string): string {
  if (dir.length === 0) {
    return name;
  }
  const sep = dir.includes("\\") ? "\\" : "/";
  return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`;
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

export function PlanSynthesis() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [templateId, setTemplateId] = useState("zero_field");
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [plan, setPlan] = useState<PlanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saveDir, setSaveDir] = useState("");
  const [size, setSize] = useState({ width: 0, height: 0 });
  const host = useRef<HTMLDivElement>(null);
  const template = catalog?.templates.find((item) => item.id === templateId) ?? catalog?.templates[0];

  useEffect(() => {
    let cancel = false;
    invoke<Catalog>("scan_kit_plan_catalog")
      .then((loaded) => {
        if (cancel) {
          return;
        }
        const first = loaded.templates[0];
        setCatalog(loaded);
        if (first != null) {
          setTemplateId(first.id);
          setValues(defaultsOf(first));
        }
        setSaveDir(loaded.save_dir ?? "");
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
  }, [plan]);

  const columns = useMemo<GridColumn[]>(
    () => (plan?.columns ?? []).map((title) => ({ title, id: title, width: 130 })),
    [plan],
  );
  const theme = useMemo(() => (plan == null ? null : gridTheme()), [plan]);

  function setValue(key: string, value: unknown) {
    setValues((current) => ({ ...current, [key]: value }));
  }

  function selectTemplate(next: string) {
    const found = catalog?.templates.find((item) => item.id === next);
    if (found == null) {
      return;
    }
    setTemplateId(found.id);
    setValues(defaultsOf(found));
    setPlan(null);
    setError(null);
  }

  async function generate() {
    if (template == null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await invoke<PlanResult>("scan_kit_synthesize_plan", {
        template: template.id,
        params: values,
      });
      setPlan(result);
    } catch (caught) {
      setPlan(null);
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  async function browse(param: Param) {
    const selected = await open({
      multiple: false,
      title: param.label,
      filters:
        param.key === "pld_path"
          ? [{ name: "IBA PLD", extensions: ["pld"] }]
          : [{ name: "RT Ion DICOM", extensions: ["dcm"] }],
    });
    if (typeof selected === "string") {
      setValue(param.key, selected);
    }
  }

  async function savePlan() {
    if (plan == null || template == null) {
      return;
    }
    const selected = await save({
      title: "Save Input Map",
      defaultPath: joinPath(saveDir, plan.filename),
      filters: [{ name: "Input map", extensions: ["csv"] }],
    });
    if (typeof selected !== "string") {
      return;
    }
    try {
      await invoke("scan_kit_synthesize_plan", {
        template: template.id,
        params: values,
        path: selected,
        csv: plan.csv,
      });
      setSaveDir(parentDir(selected));
      notify(`Saved ${selected}`);
    } catch (caught) {
      notifyError(caught);
    }
  }

  function getCellContent([col, row]: Item): GridCell {
    const text = plan?.rows[row]?.[col] ?? "";
    return { kind: GridCellKind.Text, data: text, displayData: text, allowOverlay: false };
  }

  if (catalog == null || template == null) {
    return <p className="text-muted-foreground px-3 py-2">Loading templates…</p>;
  }

  const items = catalog.templates.map((item) => ({ value: item.id, label: item.name }));
  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex w-[26rem] shrink-0 flex-col gap-3 overflow-auto border-r p-3">
        <Field>
          <FieldLabel>Template</FieldLabel>
          <Select items={items} value={template.id} onValueChange={(next) => next != null && selectTemplate(next)}>
            <SelectTrigger className="w-full cursor-pointer">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {items.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </Field>
        <p className="text-muted-foreground text-sm">{template.description}</p>
        {FIELD_SETS.map(([setId, title]) => {
          const params = template.params.filter(
            (param) => (param.field_set ?? "geometry") === setId && shown(param, values) && param.row_partner == null,
          );
          if (params.length === 0) {
            return null;
          }
          return (
            <FieldSet key={setId}>
              <FieldLegend variant="label">{title}</FieldLegend>
              <FieldGroup>
                {params.map((param) => {
                  const partner = template.params.find(
                    (item) => item.row_partner === param.key && shown(item, values),
                  );
                  return (
                    <div key={param.key} className="flex flex-col gap-2">
                      <div className={partner == null ? undefined : "grid grid-cols-2 gap-2"}>
                        <ParamControl
                          param={param}
                          values={values}
                          onChange={setValue}
                          onBrowse={() => void browse(param)}
                        />
                        {partner == null ? null : (
                          <ParamControl
                            param={partner}
                            values={values}
                            onChange={setValue}
                            onBrowse={() => void browse(partner)}
                          />
                        )}
                      </div>
                      {param.quick_sets == null || param.quick_sets.length === 0 ? null : (
                        <div className="flex flex-wrap gap-1">
                          {param.quick_sets.map((quick) => (
                            <Button
                              key={quick.label}
                              type="button"
                              size="sm"
                              variant="outline"
                              onClick={() => setValues((current) => ({ ...current, ...quick.values }))}
                            >
                              {quick.label}
                            </Button>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </FieldGroup>
            </FieldSet>
          );
        })}
        {error == null ? null : <FieldError>{error}</FieldError>}
        <Button type="button" disabled={busy} onClick={() => void generate()}>
          {busy ? "Generating…" : "Generate"}
        </Button>
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b px-3 py-2">
          <p className="text-sm">{plan?.summary ?? "No plan generated yet."}</p>
          <Button className="ml-auto" type="button" size="sm" disabled={plan == null} onClick={() => void savePlan()}>
            Save CSV
          </Button>
        </div>
        <div ref={host} className="min-h-0 flex-1">
          {plan != null && theme != null && size.width > 0 && size.height > 0 ? (
            <DataEditor
              width={size.width}
              height={size.height}
              columns={columns}
              rows={plan.rows.length}
              rowHeight={28}
              headerHeight={32}
              getCellContent={getCellContent}
              theme={theme}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ParamControl({
  param,
  values,
  onChange,
  onBrowse,
}: {
  param: Param;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
  onBrowse: () => void;
}) {
  const label = param.sub_label == null || param.sub_label.length === 0 ? param.label : `${param.label} ${param.sub_label}`;
  if (param.kind === "energy_multiselect") {
    const selected = Array.isArray(values[param.key]) ? (values[param.key] as number[]) : [];
    const catalog = Array.isArray(param.default) ? [...(param.default as number[])].reverse() : [];
    return (
      <Field>
        <FieldLabel>{param.label}</FieldLabel>
        <div className="flex flex-wrap gap-1">
          {(param.presets ?? []).map((preset) => (
            <Button
              key={preset.label}
              type="button"
              size="sm"
              variant="outline"
              onClick={() => onChange(param.key, preset.energies)}
            >
              {preset.label}
            </Button>
          ))}
        </div>
        <p className="text-muted-foreground text-sm">
          {selected.length === 0
            ? "No layers selected"
            : selected.length === 1
              ? "1 layer selected"
              : `${selected.length} layers selected`}
        </p>
        <div className="flex max-h-48 flex-col gap-1 overflow-auto">
          {catalog.map((energy) => {
            const id = `energy-${energy}`;
            const checked = selected.includes(energy);
            return (
              <Field key={energy} orientation="horizontal">
                <Checkbox
                  id={id}
                  className="cursor-pointer"
                  checked={checked}
                  onCheckedChange={(next) => {
                    const without = selected.filter((item) => item !== energy);
                    onChange(param.key, next ? [...without, energy] : without);
                  }}
                />
                <FieldLabel className="cursor-pointer" htmlFor={id}>
                  {energyText(energy)}
                </FieldLabel>
              </Field>
            );
          })}
        </div>
      </Field>
    );
  }
  if (param.kind === "bool") {
    const id = `plan-${param.key}`;
    return (
      <Field orientation="horizontal">
        <Checkbox
          id={id}
          className="cursor-pointer"
          checked={values[param.key] === true}
          onCheckedChange={(checked) => onChange(param.key, checked === true)}
        />
        <FieldLabel className="cursor-pointer" htmlFor={id}>
          {param.label}
        </FieldLabel>
      </Field>
    );
  }
  if (param.kind === "file_path") {
    return (
      <Field>
        <FieldLabel>{param.label}</FieldLabel>
        <div className="flex gap-2">
          <Input readOnly value={String(values[param.key] ?? "")} />
          <Button type="button" variant="outline" onClick={onBrowse}>
            Browse
          </Button>
        </div>
      </Field>
    );
  }
  if ((param.kind === "choice" || param.kind === "button_group") && param.choices != null) {
    if (shortChoices(param)) {
      const current = String(values[param.key] ?? param.choices[0]?.value ?? "");
      return (
        <Field>
          <FieldLabel>{param.label}</FieldLabel>
          <ToggleGroup
            variant="outline"
            spacing={0}
            size="sm"
            className="w-full"
            value={[current]}
            onValueChange={(next) => {
              const picked = next.find((item) => item !== current) ?? next[0];
              if (picked != null) {
                onChange(param.key, picked);
              }
            }}
          >
            {param.choices.map((choice) => (
              <ToggleGroupItem key={choice.value} value={choice.value} className="min-w-0 flex-1 cursor-pointer">
                {choice.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Field>
      );
    }
    const items = param.choices.map((choice) => ({ value: choice.value, label: choice.label }));
    return (
      <Field>
        <FieldLabel>{param.label}</FieldLabel>
        <Select
          items={items}
          value={String(values[param.key] ?? "")}
          onValueChange={(next) => {
            if (next != null) {
              onChange(param.key, next);
            }
          }}
        >
          <SelectTrigger className="w-full cursor-pointer">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {items.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
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
      <FieldLabel>{label}</FieldLabel>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          value={values[param.key] == null ? "" : String(values[param.key])}
          min={param.minimum}
          max={param.maximum}
          step={param.step}
          onChange={(event) => {
            const next = event.target.value;
            onChange(param.key, next === "" ? "" : Number(next));
          }}
        />
        {param.suffix == null || param.suffix.length === 0 ? null : (
          <span className="text-muted-foreground text-sm">{param.suffix}</span>
        )}
      </div>
    </Field>
  );
}
