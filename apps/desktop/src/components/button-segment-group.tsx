import { optionIcon } from "@/option-icons";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

/** Joined exclusive buttons. The first choice is pressed when none match. */
export function ButtonSegmentGroup({
  options,
  value,
  disabled,
  onChange,
}: {
  options: readonly string[];
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const selected = options.includes(value) ? value : options[0];
  return (
    <ToggleGroup
      variant="outline"
      spacing={0}
      size="sm"
      disabled={disabled}
      value={selected == null ? [] : [selected]}
      onValueChange={(next) => {
        const picked = next.find((item) => item !== selected) ?? next[0];
        if (picked != null) {
          onChange(picked);
        }
      }}
    >
      {options.map((option) => {
        const Icon = optionIcon(option);
        return (
          <ToggleGroupItem key={option} value={option}>
            {Icon == null ? null : <Icon />}
            {option}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}
