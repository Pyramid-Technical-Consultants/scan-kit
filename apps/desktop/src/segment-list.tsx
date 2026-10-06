import { X } from "lucide-react";

import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  addSegment,
  parseSegments,
  removeSegment,
  replaceSegment,
  segmentsText,
  type SegmentItem,
} from "@/analysis-controls";

const BEAM = [
  { state: "on", label: "Beam On" },
  { state: "off", label: "Beam Off" },
  { state: "both", label: "Both" },
] as const;

const RANK = [
  { which: "all", label: "All" },
  { which: "lower_95", label: "Lower 95%" },
  { which: "upper_95", label: "Upper 5%" },
  { which: "mad", label: "MAD Outliers" },
] as const;

type Kind = { id: string; label: string };

function beamLabel(state: string | undefined): string {
  return BEAM.find((item) => item.state === state)?.label ?? "Beam On";
}

function rankLabel(which: string | undefined): string {
  return RANK.find((item) => item.which === which)?.label ?? "All";
}

function RankSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (which: string) => void;
}) {
  const selected = RANK.find((item) => item.label === value) ?? RANK[0];
  return (
    <Select
      items={RANK.map((item) => ({ label: item.label, value: item.label }))}
      value={selected.label}
      onValueChange={(next) => {
        const which = RANK.find((item) => item.label === next)?.which;
        if (which != null) {
          onChange(which);
        }
      }}
    >
      <SelectTrigger size="sm" className="w-full cursor-pointer">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          {RANK.map((item) => (
            <SelectItem key={item.which} value={item.label}>
              {item.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}

/** One Filter Data list. Each row is a segment the plot already understands. */
export function SegmentList({
  value,
  kinds,
  onChange,
}: {
  value: string;
  kinds: readonly Kind[];
  onChange: (value: string) => void;
}) {
  const items = parseSegments(value) ?? [];
  const commit = (next: SegmentItem[]) => onChange(segmentsText(next));
  const remaining = kinds.filter((kind) => !items.some((item) => item.kind === kind.id));
  const nextKind = remaining[0];
  return (
    <div className="flex flex-col gap-2">
      {items.map((item, index) => {
        const label = kinds.find((kind) => kind.id === item.kind)?.label ?? item.kind;
        const parameters =
          item.kind === "beam" ? (
            <ButtonSegmentGroup
              options={BEAM.map((choice) => choice.label)}
              value={beamLabel(item.state)}
              onChange={(next) => {
                const state = BEAM.find((choice) => choice.label === next)?.state ?? "on";
                commit(replaceSegment(items, index, { kind: "beam", state }));
              }}
            />
          ) : item.kind === "rank" ? (
            <RankSelect
              value={rankLabel(item.which)}
              onChange={(which) => commit(replaceSegment(items, index, { kind: "rank", which }))}
            />
          ) : null;
        return (
          <Field key={`${item.kind}-${index}`} orientation="horizontal">
            <FieldLabel className="flex-none! shrink-0 whitespace-nowrap">{label}</FieldLabel>
            <div className="min-w-0 flex-1">{parameters}</div>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label={`Remove ${label}`}
              title={`Remove ${label}`}
              onClick={() => commit(removeSegment(items, index))}
            >
              <X />
            </Button>
          </Field>
        );
      })}
      {nextKind != null ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-label="Add segment"
          onClick={() => commit(addSegment(items, nextKind.id))}
        >
          Add
        </Button>
      ) : null}
    </div>
  );
}
