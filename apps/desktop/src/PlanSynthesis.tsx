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
  Dices,
  Download,
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
  type LucideIcon,
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

const ENERGY_PRESETS: Record<string, { label: string; title: string; icon: LucideIcon }> = {
  "Select All": { label: "All", title: "Select every layer", icon: ListChecks },
  "Whole MeV Steps": { label: "Whole MeV", title: "Integer MeV layers", icon: Hash },
  "10 MeV Steps": { label: "10 MeV", title: "Every 10 MeV", icon: Rows3 },
  "Clear All": { label: "None", title: "Clear the selection", icon: ListX },
};

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

  const applyQuick = (next: Record<string, number>) => {
    setValues((current) => ({ ...current, ...next }));
  };
  return (
    <SidePane
      side={
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
            <FieldSet className="gap-2 rounded-lg border border-border p-3">
              <FieldLegend variant="label">Template</FieldLegend>
              <ToggleGroup
                variant="outline"
                orientation="vertical"
                spacing={2}
                className="w-full flex-col items-stretch"
                value={[template.id]}
                onValueChange={(next) => {
                  const picked = next.find((item) => item !== template.id) ?? next[0];
                  if (picked != null) {
                    selectTemplate(picked);
                  }
                }}
              >
                {catalog.templates.map((item) => (
                  <ToggleGroupItem
                    key={item.id}
                    value={item.id}
                    className="w-full min-w-0 justify-start gap-3 px-2.5 font-normal"
                  >
                    <span className="shrink-0 font-medium">{item.name}</span>
                    <span
                      className="text-muted-foreground ml-auto min-w-0 truncate text-right font-normal"
                      title={item.description}
                    >
                      {item.description}
                    </span>
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </FieldSet>
            {FIELD_SETS.map(([setId, title]) => {
              const params = template.params.filter(
                (param) =>
                  (param.field_set ?? "geometry") === setId && shown(param, values) && param.row_partner == null,
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
                        (item) => item.row_partner === param.key && shown(item, values),
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
    const presets = param.presets ?? [];
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
        <div className="grid max-h-52 grid-cols-2 gap-x-3 gap-y-1 overflow-y-auto">
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
  if ((param.kind === "button_group" || (param.kind === "choice" && shortChoices(param))) && param.choices != null) {
    return <SegmentField param={param} value={String(values[param.key] ?? "")} onChange={onChange} />;
  }
  if (param.kind === "choice" && param.choices != null) {
    const current = String(values[param.key] ?? "");
    const items = param.choices.map((choice) => ({ value: choice.value, label: choice.label }));
    const Icon = choiceIcon(param.choices.find((choice) => choice.value === current)?.label ?? "");
    return (
      <Field orientation="horizontal">
        <FieldLabel className={LABEL_CLASS}>{param.label}</FieldLabel>
        <Select
          items={items}
          value={current}
          onValueChange={(next) => {
            if (next != null) {
              onChange(param.key, next);
            }
          }}
        >
          <SelectTrigger className="w-full min-w-0 cursor-pointer">
            {Icon == null ? null : <Icon />}
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {param.choices.map((choice) => {
                const ItemIcon = choiceIcon(choice.label);
                return (
                  <SelectItem key={choice.value} value={choice.value}>
                    {ItemIcon == null ? null : <ItemIcon />}
                    {choice.label}
                  </SelectItem>
                );
              })}
            </SelectGroup>
          </SelectContent>
        </Select>
      </Field>
    );
  }
  return null;
}

function SegmentField({
  param,
  value,
  onChange,
}: {
  param: Param;
  value: string;
  onChange: (key: string, value: unknown) => void;
}) {
  const choices = param.choices ?? [];
  const current = choices.some((choice) => choice.value === value) ? value : (choices[0]?.value ?? "");
  const grid = choices.length >= 4;
  const stacked = grid || choices.some((choice) => choice.label.length > 8);
  const group = (
    <ToggleGroup
      variant="outline"
      spacing={grid ? 2 : 0}
      size="sm"
      className={grid ? "grid w-full grid-cols-2" : "w-full"}
      value={current.length === 0 ? [] : [current]}
      onValueChange={(next) => {
        const picked = next.find((item) => item !== current) ?? next[0];
        if (picked != null) {
          onChange(param.key, picked);
        }
      }}
    >
      {choices.map((choice) => {
        const Icon = choiceIcon(choice.label);
        return (
          <ToggleGroupItem key={choice.value} value={choice.value} className="min-w-0 flex-1 cursor-pointer">
            {Icon == null ? null : <Icon />}
            {choice.label}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
  if (stacked) {
    return (
      <Field>
        <FieldLabel>{param.label}</FieldLabel>
        {group}
      </Field>
    );
  }
  return (
    <Field orientation="horizontal">
      <FieldLabel className={LABEL_CLASS}>{param.label}</FieldLabel>
      <div className="min-w-0 flex-1">{group}</div>
    </Field>
  );
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
