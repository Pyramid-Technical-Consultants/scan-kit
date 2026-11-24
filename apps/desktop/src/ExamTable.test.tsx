import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { GridCell, Theme } from "@glideapps/glide-data-grid";

import { ExamTable, type ExamRow } from "./ExamTable";

type EditorProps = {
  getCellContent: (cell: readonly [number, number]) => GridCell;
  onHeaderClicked: (column: number) => void;
  columns: Array<{ title: string }>;
  gridSelection?: object;
  onGridSelectionChange?: (selection: object) => void;
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

it("keeps the highlight on a same-order refresh and clears it when the rows change", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  const props = { theme: {} as Theme, width: 400, height: 200 };
  await act(async () => {
    root = createRoot(host);
    root.render(<ExamTable rows={rows} {...props} />);
  });
  const selection = { rows: { has: () => true } };
  await act(async () => {
    editor?.onGridSelectionChange?.(selection);
  });
  expect(editor?.gridSelection).toBe(selection);
  // A catalog pull rebuilds rows with a new array identity every 5 s — the
  // ordered keys are identical, so the highlight must survive.
  await act(async () => {
    root?.render(<ExamTable rows={rows.map((row) => ({ ...row }))} {...props} />);
  });
  expect(editor?.gridSelection).toBe(selection);
  // A different exam set at the same positions is not the same display
  // order — the positional highlight is meaningless and must clear.
  await act(async () => {
    root?.render(
      <ExamTable
        rows={[rows[0], { ...rows[1], exam: "c" }]}
        {...props}
      />,
    );
  });
  expect(editor?.gridSelection).toEqual({});
});

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

it("sinks exams without a date last in both directions", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  const undated: ExamRow[] = [
    { exam: "z", patient: "Ann", patient_id: "2", date: "", description: "", files: 1 },
    { exam: "b", patient: "Ann", patient_id: "2", date: "2020", description: "", files: 2 },
    { exam: "a", patient: "Bea", patient_id: "1", date: "2021", description: "", files: 10 },
  ];
  await act(async () => {
    root = createRoot(host);
    root.render(<ExamTable rows={undated} theme={{} as Theme} width={400} height={200} />);
  });
  const text = (cell: readonly [number, number]) => {
    const content = editor?.getCellContent(cell) as { displayData?: string };
    return content.displayData ?? "";
  };
  await act(async () => {
    editor?.onHeaderClicked(3);
  });
  expect(text([3, 0])).toBe("2020");
  expect(text([3, 2])).toBe("");
  await act(async () => {
    editor?.onHeaderClicked(3);
  });
  expect(text([3, 0])).toBe("2021");
  expect(text([3, 2])).toBe("");
});
