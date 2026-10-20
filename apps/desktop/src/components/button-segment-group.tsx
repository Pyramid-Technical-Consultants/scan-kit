import { cn } from "cn";

import { optionIcon } from "@/option-icons";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

type SegmentOption = string | {
  id?: string;
  label: string;
  icon?: string;
  detail?: string;
  iconOnly?: boolean;
};

function optionText(option: SegmentOption): string {
  return typeof option === "string" ? option : option.label;
}

function sameChoice(option: SegmentOption, value: string): boolean {
  if (typeof option === "string") {
    return option === value;
  }
  return (
    option.label === value || (option.id != null && option.id.length > 0 && option.id === value)
  );
}

function optionGlyph(option: SegmentOption) {
  if (typeof option === "string") {
    return optionIcon(option);
  }
  return (option.icon != null && option.icon.length > 0 ? optionIcon(option.icon) : undefined) ?? optionIcon(option.label);
}

/** Joined exclusive buttons. The first choice is pressed when none match. */
export function ButtonSegmentGroup({
  options,
  value,
  disabled,
  label,
  className,
  orientation = "horizontal",
  onChange,
}: {
  options: readonly SegmentOption[];
  value: string;
  disabled?: boolean;
  label?: string;
  className?: string;
  orientation?: "horizontal" | "vertical";
  onChange: (value: string) => void;
}) {
  const labels = options.map(optionText);
  const matched = options.find((option) => sameChoice(option, value));
  const selected = matched == null ? labels[0] : optionText(matched);
  return (
    <ToggleGroup
      variant="outline"
      spacing={0}
      size="sm"
      disabled={disabled}
      aria-label={label}
      orientation={orientation}
      className={cn("w-full", className)}
      value={selected == null ? [] : [selected]}
      onValueChange={(next) => {
        const picked = next.find((item) => item !== selected) ?? next[0];
        if (picked != null) {
          onChange(picked);
        }
      }}
    >
      {options.map((option) => {
        const label = optionText(option);
        const detail = typeof option === "string" ? "" : (option.detail ?? "");
        const iconOnly = typeof option !== "string" && option.iconOnly === true;
        const Icon = optionGlyph(option);
        return (
          <ToggleGroupItem
            key={label}
            value={label}
            aria-label={iconOnly ? label : undefined}
            title={detail.length > 0 ? detail : iconOnly ? label : undefined}
            className={cn(
              "shrink cursor-pointer text-sm group-data-horizontal/toggle-group:data-[spacing=0]:first:rounded-l-[min(var(--radius-md),10px)] group-data-horizontal/toggle-group:data-[spacing=0]:last:rounded-r-[min(var(--radius-md),10px)] [&_svg:not([class*='size-'])]:size-4",
              iconOnly
                ? "w-7 flex-none px-0"
                : orientation === "vertical"
                  ? "w-full min-w-0 flex-none justify-start overflow-hidden"
                  : "min-w-0 flex-auto overflow-hidden",
            )}
          >
            {Icon == null ? null : <Icon />}
            {iconOnly ? null : (
              <span className={cn("min-w-0 truncate", orientation === "vertical" && "max-w-full shrink-0")}>
                {label}
              </span>
            )}
            {orientation === "vertical" && !iconOnly && detail.length > 0 ? (
              <span className="text-muted-foreground ml-auto min-w-0 truncate pl-4 text-right font-normal">
                {detail}
              </span>
            ) : null}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}
