import { X } from "lucide-react";

import { ButtonSegmentGroup } from "@/components/button-segment-group";
import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import {
  addSegment,
  parseSegments,
  removeSegment,
  replaceSegment,
  segmentsText,
  type SegmentItem,
} from "@/analysis-controls";

const BEAM = [
  { state: "both", label: "Both" },
  { state: "on", label: "On" },
  { state: "off", label: "Off" },
] as const;

const RANK = [
  { which: "all", label: "All", icon: "all", detail: "All rows" },
  { which: "lower_95", label: "95%", icon: "lower_95", detail: "Within the lower 95%" },
  { which: "upper_95", label: "5%", icon: "upper_95", detail: "Upper 5% only" },
  { which: "mad", label: "MAD", icon: "mad", detail: "MAD outliers" },
] as const;

type Kind = { id: string; label: string };

function beamLabel(state: string | undefined): string {
  return BEAM.find((item) => item.state === state)?.label ?? "Both";
}

function rankLabel(which: string | undefined): string {
  return RANK.find((item) => item.which === which)?.label ?? "All";
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
            <ButtonSegmentGroup
              options={RANK.map(({ label, icon, detail }) => ({ label, icon, detail }))}
              value={rankLabel(item.which)}
              onChange={(next) => {
                const which = RANK.find((choice) => choice.label === next)?.which ?? "all";
                commit(replaceSegment(items, index, { kind: "rank", which }));
              }}
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
