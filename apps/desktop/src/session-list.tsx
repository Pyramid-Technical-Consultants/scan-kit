import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { sessionColor } from "@/session-colors";

export type SessionListItem = { id: string; note: string };

/** Colored session checkboxes. The note stays on the id's line and ellipsizes. */
export function SessionList({
  sessions,
  isChecked,
  onCheckedChange,
}: {
  sessions: readonly SessionListItem[];
  isChecked: (id: string) => boolean;
  onCheckedChange: (id: string, checked: boolean) => void;
}) {
  return (
    <FieldSet className="min-w-0 gap-2 rounded-lg border border-border p-3">
      <FieldLegend variant="label">Sessions</FieldLegend>
      {sessions.map((session, index) => {
        const checked = isChecked(session.id);
        const color = sessionColor(index);
        const inputId = `session-list-${session.id}`;
        return (
          <Field key={session.id} orientation="horizontal" className="min-w-0">
            <Checkbox
              id={inputId}
              className="cursor-pointer"
              checked={checked}
              aria-label={`Include ${session.id}`}
              title={
                checked
                  ? `Session color in plots: ${color}`
                  : `Hidden. Check to draw ${session.id} in ${color}`
              }
              style={
                checked
                  ? { backgroundColor: color, borderColor: color, color: "#fff" }
                  : { borderColor: color }
              }
              onCheckedChange={(next) => onCheckedChange(session.id, next === true)}
            />
            <FieldLabel htmlFor={inputId} className="w-full min-w-0 flex-1 cursor-pointer overflow-hidden">
              <span className="shrink-0">{session.id}</span>
              {session.note === "" ? null : (
                <span className="text-muted-foreground min-w-0 flex-1 truncate font-normal" title={session.note}>
                  {session.note}
                </span>
              )}
            </FieldLabel>
          </Field>
        );
      })}
    </FieldSet>
  );
}
