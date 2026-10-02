import { optionIcon } from "@/option-icons";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

type SegmentOption = string | { label: string; icon?: string; detail?: string };

function optionText(option: SegmentOption): string {
  return typeof option === "string" ? option : option.label;
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
  onChange,
}: {
  options: readonly SegmentOption[];
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const labels = options.map(optionText);
  const selected = labels.includes(value) ? value : labels[0];
  return (
    <ToggleGroup
      variant="outline"
      spacing={0}
      size="sm"
      disabled={disabled}
      className="w-full"
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
        const Icon = optionGlyph(option);
        return (
          <ToggleGroupItem
            key={label}
            value={label}
            title={detail.length > 0 ? detail : undefined}
            className="min-w-0 flex-1 shrink cursor-pointer text-sm group-data-horizontal/toggle-group:data-[spacing=0]:first:rounded-l-[min(var(--radius-md),10px)] group-data-horizontal/toggle-group:data-[spacing=0]:last:rounded-r-[min(var(--radius-md),10px)] [&_svg:not([class*='size-'])]:size-4"
          >
            {Icon == null ? null : <Icon />}
            {label}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}
