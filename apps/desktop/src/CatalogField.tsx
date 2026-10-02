import type { ReactNode } from "react";

import { Field, FieldLabel } from "@/components/ui/field";
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

type CatalogChoice = { value: string; label: string; tooltip?: string };

export type CatalogParam = {
  label: string;
  kind: string;
  minimum?: number;
  maximum?: number;
  step?: number;
  suffix?: string;
  tooltip?: string;
  choices?: CatalogChoice[];
  visible_when?: Record<string, Array<string | number | boolean>>;
};

const BESIDE_LABEL = "w-28 shrink-0 flex-none! leading-snug";

export function catalogShown(
  param: { visible_when?: CatalogParam["visible_when"] },
  values: Record<string, unknown>,
): boolean {
  if (param.visible_when == null) {
    return true;
  }
  return Object.entries(param.visible_when).every(([key, allowed]) =>
    allowed.some((item) => item === values[key]),
  );
}

/** Two or three short labels sit in a joined button row. */
export function segmentChoices(options: readonly string[]): boolean {
  return (
    options.length >= 2 &&
    options.length < 4 &&
    options.every((option) => option.length > 0 && option.length <= 10)
  );
}

function shortChoices(choices: CatalogChoice[] | undefined): boolean {
  return segmentChoices(choices?.map((choice) => choice.label) ?? []);
}

function ChoiceControl({
  param,
  value,
  onChange,
  icon,
  beside,
  forceSegment,
}: {
  param: CatalogParam;
  value: unknown;
  onChange: (value: unknown) => void;
  icon?: (label: string) => ReactNode;
  beside: boolean;
  forceSegment: boolean;
}) {
  const choices = param.choices ?? [];
  const text = typeof value === "string" ? value : "";
  const segment = forceSegment || shortChoices(choices);
  if (segment && choices.length > 0) {
    const current = beside
      ? choices.some((choice) => choice.value === text)
        ? text
        : (choices[0]?.value ?? "")
      : text;
    const grid = choices.length >= 4;
    const stacked = !beside || grid || choices.some((choice) => choice.label.length > 8);
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
            onChange(picked);
          }
        }}
      >
        {choices.map((choice) => (
          <ToggleGroupItem
            key={choice.value}
            value={choice.value}
            title={choice.tooltip}
            className="min-w-0 flex-1 cursor-pointer"
          >
            {icon?.(choice.label)}
            {choice.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    );
    if (!stacked) {
      return (
        <Field orientation="horizontal">
          <FieldLabel className={BESIDE_LABEL}>{param.label}</FieldLabel>
          <div className="min-w-0 flex-1">{group}</div>
        </Field>
      );
    }
    return (
      <Field>
        <FieldLabel title={param.tooltip}>{param.label}</FieldLabel>
        {group}
      </Field>
    );
  }
  const items = choices.map((choice) => ({ value: choice.value, label: choice.label }));
  const selected = icon?.(choices.find((choice) => choice.value === text)?.label ?? "");
  const select = (
    <Select
      items={items}
      value={text}
      onValueChange={(next) => {
        if (next != null) {
          onChange(next);
        }
      }}
    >
      <SelectTrigger className={beside ? "w-full min-w-0 cursor-pointer" : "w-full cursor-pointer"}>
        {selected}
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          {choices.map((choice) => (
            <SelectItem key={choice.value} value={choice.value} title={choice.tooltip}>
              {icon?.(choice.label)}
              {choice.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
  if (beside) {
    return (
      <Field orientation="horizontal">
        <FieldLabel className={BESIDE_LABEL}>{param.label}</FieldLabel>
        {select}
      </Field>
    );
  }
  return (
    <Field>
      <FieldLabel title={param.tooltip}>{param.label}</FieldLabel>
      {select}
    </Field>
  );
}

export function CatalogField({
  param,
  value,
  onChange,
  icon,
  beside = false,
  forceSegment = false,
  finiteOnly = false,
}: {
  param: CatalogParam;
  value: unknown;
  onChange: (value: unknown) => void;
  icon?: (label: string) => ReactNode;
  beside?: boolean;
  forceSegment?: boolean;
  finiteOnly?: boolean;
}) {
  if (param.kind === "choice" || param.kind === "button_group") {
    return (
      <ChoiceControl
        param={param}
        value={value}
        onChange={onChange}
        icon={icon}
        beside={beside}
        forceSegment={forceSegment || param.kind === "button_group"}
      />
    );
  }
  return (
    <Field>
      <FieldLabel title={param.tooltip}>{param.label}</FieldLabel>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          inputMode={finiteOnly ? undefined : "decimal"}
          min={param.minimum}
          max={param.maximum}
          step={param.step ?? (finiteOnly ? undefined : "any")}
          title={param.tooltip}
          value={value == null || (typeof value === "number" && !Number.isFinite(value)) ? "" : String(value)}
          onChange={(event) => {
            const text = event.target.value;
            const number = Number(text);
            if (finiteOnly) {
              if (Number.isFinite(number)) {
                onChange(number);
              }
              return;
            }
            onChange(Number.isFinite(number) && text.trim() !== "" ? number : text);
          }}
        />
        {param.suffix != null && param.suffix.length > 0 ? (
          <span className="text-muted-foreground text-sm">{param.suffix}</span>
        ) : null}
      </div>
    </Field>
  );
}
