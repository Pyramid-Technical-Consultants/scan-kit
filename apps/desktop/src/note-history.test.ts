import { expect, it } from "vitest";

import type { NoteEdit } from "@/LibraryTable";
import { withoutEdit } from "@/note-history";

it("drops the edit that finished, not whatever was pushed last", () => {
  const older: NoteEdit = { sessionId: "a", library: "lib", before: "", after: "one" };
  const undone: NoteEdit = { sessionId: "a", library: "lib", before: "one", after: "two" };
  const newer: NoteEdit = { sessionId: "a", library: "lib", before: "two", after: "three" };
  expect(withoutEdit([older, undone, newer], undone)).toEqual([older, newer]);
  const untouched = [older];
  expect(withoutEdit(untouched, newer)).toBe(untouched);
});
