import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  ArrowDownLeft,
  ArrowDownRight,
  ArrowRight,
  ArrowUpLeft,
  ArrowUpRight,
  Combine,
  CornerUpLeft,
  Crosshair,
  Dices,
  Download,
  File,
  FileText,
  FolderOpen,
  Hash,
  Layers,
  ListChecks,
  ListOrdered,
  ListX,
  Lock,
  MoveHorizontal,
  MoveVertical,
  Route,
  Rows3,
  Scale,
  Shuffle,
  Square,
  type LucideIcon,
} from "lucide-react";
import {
  DataEditor,
  GridCellKind,
  type GridCell,
  type GridColumn,
  type Item,
} from "@glideapps/glide-data-grid";
import { gridTheme } from "@/grid-theme";
import "@glideapps/glide-data-grid/dist/index.css";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
  InputGroupText,
} from "@/components/ui/input-group";
import { optionIcon } from "@/option-icons";
import { CatalogField, catalogShown } from "@/CatalogField";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { notifyError, notifySaved } from "@/notify";
import { SidePane } from "@/SidePane";

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
  ["energy", "Energy (MeV)"],
  ["geometry", "Geometry"],
  ["weight", "Weight"],
];

const LABEL_CLASS = "w-28 shrink-0 flex-none! leading-snug";

const CHOICE_ICONS: Record<string, LucideIcon> = {
  X: MoveHorizontal,
  Y: MoveVertical,
  "Top Left": ArrowUpLeft,
  "Top Right": ArrowUpRight,
  "Bottom Left": ArrowDownLeft,
  "Bottom Right": ArrowDownRight,
  "Reset to Corner": CornerUpLeft,
  "Continue from End": ArrowRight,
  "Plan Order": ListOrdered,
  "Minimize Travel": Route,
  Fixed: Lock,
  "Random Range": Shuffle,
  "Even per Layer": Layers,
  "Even Total": Scale,
  "Random Total": Dices,
};

const PRESET_ORDER = ["Select All", "Clear All", "Whole MeV Steps", "10 MeV Steps"];

function presetRank(label: string): number {
  const rank = PRESET_ORDER.indexOf(label);
  return rank < 0 ? PRESET_ORDER.length : rank;
}

const ENERGY_PRESETS: Record<string, { label: string; title: string; icon: LucideIcon }> = {
  "Select All": { label: "All", title: "Select every layer", icon: ListChecks },
  "Whole MeV Steps": { label: "Whole MeV", title: "Integer MeV layers", icon: Hash },
  "10 MeV Steps": { label: "10 MeV", title: "Every 10 MeV", icon: Rows3 },
  "Clear All": { label: "None", title: "Clear the selection", icon: ListX },
};

const TEMPLATE_ICONS: Record<string, LucideIcon> = {
  zero_field: Crosshair,
  rectangular_field: Square,
  dicom_rt_plan: FileText,
  iba_pld_plan: File,
};

function defaultsOf(template: Template): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const param of template.params) {
    values[param.key] = param.default;
  }
  return values;
}

function labelParts(label: string): { name: string; unit?: string } {
  const match = /^(.*)\s+\(([^)]+)\)$/.exec(label);
  if (match == null) {
    return { name: label };
  }
  return { name: match[1], unit: match[2] };
}

function unitOf(param: Param): string | undefined {
  const unit =
    param.suffix != null && param.suffix.length > 0 ? param.suffix : labelParts(param.label).unit;
  if (unit == null || labelParts(param.label).name.toLowerCase().includes(unit.toLowerCase())) {
    return undefined;
  }
  return unit;
}

function numericKind(kind: string): boolean {
  return kind === "int" || kind === "float";
}

function choiceIcon(label: string): LucideIcon | undefined {
  return CHOICE_ICONS[label] ?? optionIcon(label);
}

function sameNumbers(left: number[], right: number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const a = [...left].sort((x, y) => x - y);
  const b = [...right].sort((x, y) => x - y);
  return a.every((value, index) => value === b[index]);
}

function fileName(path: string): string {
  const cut = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return cut >= 0 ? path.slice(cut + 1) : path;
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
      notifySaved(selected, { title: "Input map saved" });
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

  const applyQuick = (next: Record<string, number>) => {
    setValues((current) => ({ ...current, ...next }));
  };
  return (
    <SidePane
      side={
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-x-hidden overflow-y-auto p-3">
            <FieldSet className="min-w-0 gap-2 rounded-lg border border-border p-3">
              <FieldLegend variant="label">Template</FieldLegend>
              <ToggleGroup
                variant="outline"
                orientation="vertical"
                spacing={2}
                className="w-full min-w-0 flex-col items-stretch"
                value={[template.id]}
                onValueChange={(next) => {
                  const picked = next.find((item) => item !== template.id) ?? next[0];
                  if (picked != null) {
                    selectTemplate(picked);
                  }
                }}
              >
                {catalog.templates.map((item) => {
                  const Icon = TEMPLATE_ICONS[item.id];
                  return (
                    <ToggleGroupItem
                      key={item.id}
                      value={item.id}
                      className="w-full min-w-0 justify-start gap-3 overflow-hidden px-2.5 font-normal"
                    >
                      <span className="flex shrink-0 items-center gap-1.5 font-medium">
                        {Icon == null ? null : <Icon />}
                        {item.name}
                      </span>
                      <span
                        className="text-muted-foreground ml-auto min-w-0 flex-1 truncate text-right font-normal"
                        title={item.description}
                      >
                        {item.description}
                      </span>
                    </ToggleGroupItem>
                  );
                })}
              </ToggleGroup>
            </FieldSet>
            {FIELD_SETS.map(([setId, title]) => {
              const params = template.params.filter(
                (param) =>
                  (param.field_set ?? "geometry") === setId && catalogShown(param, values) && param.row_partner == null,
              );
              if (params.length === 0) {
                return null;
              }
              return (
                <FieldSet key={setId} className="gap-2 rounded-lg border border-border p-3">
                  <FieldLegend variant="label">{title}</FieldLegend>
                  <FieldGroup className="gap-3">
                    {params.map((param) => {
                      const partner = template.params.find(
                        (item) => item.row_partner === param.key && catalogShown(item, values),
                      );
                      if (numericKind(param.kind)) {
                        return (
                          <NumberRow
                            key={param.key}
                            param={param}
                            partner={partner != null && numericKind(partner.kind) ? partner : undefined}
                            values={values}
                            onChange={setValue}
                            onQuick={applyQuick}
                          />
                        );
                      }
                      return (
                        <ParamControl
                          key={param.key}
                          param={param}
                          values={values}
                          onChange={setValue}
                          onBrowse={() => void browse(param)}
                        />
                      );
                    })}
                  </FieldGroup>
                </FieldSet>
              );
            })}
          </div>
          <div className="flex flex-col gap-2 border-t p-3">
            {error == null ? null : <FieldError>{error}</FieldError>}
            <div className="flex gap-2">
              <Button type="button" className="min-w-0 flex-1" disabled={busy} onClick={() => void generate()}>
                <Combine />
                {busy ? "Generating…" : "Generate"}
              </Button>
              <Button
                type="button"
                variant="outline"
                className="min-w-0 flex-1"
                disabled={plan == null}
                onClick={() => void savePlan()}
              >
                <Download />
                Save CSV
              </Button>
            </div>
          </div>
        </div>
      }
      main={
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {plan == null ? (
            <div className="text-muted-foreground flex flex-1 items-center justify-center px-6 text-center text-sm">
              No plan generated yet.
            </div>
          ) : (
            <>
              <p className="border-b px-3 py-2 text-sm tabular-nums">{plan.summary}</p>
              <div ref={host} className="min-h-0 flex-1">
                {theme != null && size.width > 0 && size.height > 0 ? (
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
            </>
          )}
        </div>
      }
    />
  );
}

function NumberInput({
  id,
  label,
  value,
  prefix,
  suffix,
  param,
  onChange,
}: {
  id: string;
  label: string;
  value: unknown;
  prefix?: string;
  suffix?: string;
  param: Param;
  onChange: (key: string, value: unknown) => void;
}) {
  return (
    <InputGroup>
      <InputGroupInput
        id={id}
        aria-label={prefix == null ? undefined : label}
        type="number"
        className="tabular-nums [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
        value={value == null ? "" : String(value)}
        min={param.minimum}
        max={param.maximum}
        step={param.step}
        onChange={(event) => {
          const next = event.target.value;
          onChange(param.key, next === "" ? "" : Number(next));
        }}
      />
      {prefix == null || prefix.length === 0 ? null : (
        <InputGroupAddon>
          <InputGroupText>{prefix}</InputGroupText>
        </InputGroupAddon>
      )}
      {suffix == null || suffix.length === 0 ? null : (
        <InputGroupAddon align="inline-end">
          <InputGroupText>{suffix}</InputGroupText>
        </InputGroupAddon>
      )}
    </InputGroup>
  );
}

function QuickSets({
  sets,
  onApply,
}: {
  sets: QuickSet[];
  onApply: (values: Record<string, number>) => void;
}) {
  if (sets.length === 0) {
    return null;
  }
  return (
    <div className="flex gap-1">
      {sets.map((quick) => (
        <Button
          key={quick.label}
          type="button"
          size="sm"
          variant="outline"
          className={sets.length === 1 ? undefined : "min-w-0 flex-1"}
          onClick={() => onApply(quick.values)}
        >
          {quick.label}
        </Button>
      ))}
    </div>
  );
}

function NumberRow({
  param,
  partner,
  values,
  onChange,
  onQuick,
}: {
  param: Param;
  partner?: Param;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
  onQuick: (values: Record<string, number>) => void;
}) {
  const name = labelParts(param.label).name;
  const unit = unitOf(param);
  const quick = param.quick_sets ?? [];
  return (
    <Field orientation="horizontal" className={quick.length > 0 ? "items-start" : undefined}>
      <FieldLabel htmlFor={`plan-${param.key}`} className={quick.length > 0 ? `${LABEL_CLASS} mt-1.5` : LABEL_CLASS}>
        {name}
      </FieldLabel>
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className={partner == null ? undefined : "flex gap-1.5"}>
          <NumberInput
            id={`plan-${param.key}`}
            label={param.sub_label == null || param.sub_label.length === 0 ? name : `${name} ${param.sub_label}`}
            value={values[param.key]}
            prefix={partner == null ? undefined : param.sub_label}
            suffix={partner == null ? unit : undefined}
            param={param}
            onChange={onChange}
          />
          {partner == null ? null : (
            <NumberInput
              id={`plan-${partner.key}`}
              label={`${name} ${partner.sub_label ?? ""}`.trim()}
              value={values[partner.key]}
              prefix={partner.sub_label}
              suffix={unit}
              param={partner}
              onChange={onChange}
            />
          )}
        </div>
        <QuickSets sets={quick} onApply={onQuick} />
      </div>
    </Field>
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
  if (param.kind === "energy_multiselect") {
    const selected = Array.isArray(values[param.key]) ? (values[param.key] as number[]) : [];
    const catalog = Array.isArray(param.default) ? [...(param.default as number[])].reverse() : [];
    const presets = [...(param.presets ?? [])].sort((a, b) => presetRank(a.label) - presetRank(b.label));
    const active = presets.find((preset) => sameNumbers(preset.energies, selected));
    const count =
      selected.length === 0
        ? "No layers selected"
        : selected.length === catalog.length
          ? `All ${catalog.length} layers`
          : `${selected.length} of ${catalog.length} layers`;
    return (
      <Field>
        {presets.length === 0 ? null : (
          <div className={presets.length >= 4 ? "grid grid-cols-2 gap-1" : "flex gap-1"}>
            {presets.map((preset) => {
              const view = ENERGY_PRESETS[preset.label];
              const Icon = view?.icon;
              const pressed = active?.label === preset.label;
              return (
                <Button
                  key={preset.label}
                  type="button"
                  size="sm"
                  variant={pressed ? "secondary" : "outline"}
                  className="min-w-0 flex-1"
                  title={view?.title ?? preset.label}
                  aria-pressed={pressed}
                  onClick={() => onChange(param.key, preset.energies)}
                >
                  {Icon == null ? null : <Icon />}
                  {view?.label ?? preset.label}
                </Button>
              );
            })}
          </div>
        )}
        <FieldDescription>{count}</FieldDescription>
        <div className="grid max-h-52 grid-cols-4 gap-x-2 gap-y-1 overflow-y-auto">
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
                <FieldLabel className="cursor-pointer tabular-nums" htmlFor={id}>
                  {String(energy)}
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
    return <FileField param={param} value={String(values[param.key] ?? "")} onBrowse={onBrowse} />;
  }
  if (param.kind === "button_group" || param.kind === "choice") {
    return (
      <CatalogField
        param={param}
        value={values[param.key]}
        beside
        icon={(label) => {
          const Icon = choiceIcon(label);
          return Icon == null ? null : <Icon />;
        }}
        onChange={(next) => onChange(param.key, next)}
      />
    );
  }
  return null;
}

function FileField({ param, value, onBrowse }: { param: Param; value: string; onBrowse: () => void }) {
  const id = `plan-${param.key}`;
  const dir = parentDir(value);
  return (
    <Field>
      <FieldLabel htmlFor={id}>{labelParts(param.label).name}</FieldLabel>
      <InputGroup>
        <InputGroupInput
          id={id}
          readOnly
          value={value.length === 0 ? "" : fileName(value)}
          title={value.length === 0 ? undefined : value}
          placeholder="No file selected"
        />
        <InputGroupAddon align="inline-end">
          <InputGroupButton onClick={onBrowse}>
            <FolderOpen />
            Browse
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>
      {dir.length === 0 || dir === value ? null : <FieldDescription className="truncate">{dir}</FieldDescription>}
    </Field>
  );
}
