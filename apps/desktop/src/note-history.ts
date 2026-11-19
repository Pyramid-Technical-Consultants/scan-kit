import type { NoteEdit } from "@/LibraryTable";

/** Drop one edit by identity. A newer edit can land while the write is in flight. */
export function withoutEdit(stack: NoteEdit[], edit: NoteEdit): NoteEdit[] {
  const index = stack.lastIndexOf(edit);
  if (index < 0) {
    return stack;
  }
  return stack.filter((_, i) => i !== index);
}
