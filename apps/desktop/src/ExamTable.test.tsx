import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { GridCell, Theme } from "@glideapps/glide-data-grid";

import { ExamTable, type ExamRow } from "./ExamTable";

type EditorProps = {
  getCellContent: (cell: readonly [number, number]) => GridCell;
  onHeaderClicked: (column: number) => void;
  columns: Array<{ title: string }>;
};

let editor: EditorProps | null = null;

vi.mock("@glideapps/glide-data-grid", () => ({
  DataEditor: (props: EditorProps) => {
    editor = props;
    return <div data-grid="" />;
  },
  emptyGridSelection: {},
  GridCellKind: { Text: "text" },
}));

const rows: ExamRow[] = [
  { exam: "b", patient: "Ann", patient_id: "2", date: "2020", description: "later", files: 2 },
  { exam: "a", patient: "Bea", patient_id: "1", date: "2021", description: "earlier", files: 10 },
];

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  editor = null;
  document.body.replaceChildren();
});

it("sorts exams from the header and reads each column", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<ExamTable rows={rows} theme={{} as Theme} width={400} height={200} />);
  });
  const text = (cell: readonly [number, number]) => {
    const content = editor?.getCellContent(cell) as { displayData?: string };
    return content.displayData ?? "";
  };
  expect(text([0, 0])).toBe("a");
  expect(text([5, 1])).toBe("2");
  expect(text([0, 8])).toBe("");
  await act(async () => {
    editor?.onHeaderClicked(5);
  });
  expect(editor?.columns[5]?.title).toContain("↑");
  expect(text([5, 0])).toBe("2");
  await act(async () => {
    editor?.onHeaderClicked(5);
  });
  expect(editor?.columns[5]?.title).toContain("↓");
  expect(text([5, 0])).toBe("10");
  await act(async () => {
    editor?.onHeaderClicked(99);
  });
  expect(text([4, 0])).toBe("earlier");
});
