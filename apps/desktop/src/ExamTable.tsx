import { useMemo, useState } from "react";
import {
  DataEditor,
  emptyGridSelection,
  GridCellKind,
  type GridCell,
  type GridColumn,
  type Theme,
} from "@glideapps/glide-data-grid";

export type ExamRow = {
  exam: string;
  patient: string;
  patient_id: string;
  date: string;
  description: string;
  files: number;
};

type ExamSortKey = "exam" | "patient" | "patient_id" | "date" | "description" | "files";

const COLUMNS: Array<{ title: string; key: ExamSortKey }> = [
  { title: "Exam", key: "exam" },
  { title: "Patient", key: "patient" },
  { title: "Patient ID", key: "patient_id" },
  { title: "Date", key: "date" },
  { title: "Description", key: "description" },
  { title: "Files", key: "files" },
];

function examValue(row: ExamRow, key: ExamSortKey): string | number {
  return row[key];
}

export function ExamTable({
  rows,
  theme,
  width,
  height,
}: {
  rows: ExamRow[];
  theme: Theme;
  width: number;
  height: number;
}) {
  const [sort, setSort] = useState<{ key: ExamSortKey; direction: "asc" | "desc" }>({
    key: "exam",
    direction: "asc",
  });
  const [gridSelection, setGridSelection] = useState(emptyGridSelection);
  const order = useMemo(() => {
    const indexes = rows.map((_, index) => index);
    indexes.sort((left, right) => {
      const a = examValue(rows[left], sort.key);
      const b = examValue(rows[right], sort.key);
      const compared = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
      return sort.direction === "asc" ? compared : -compared;
    });
    return indexes;
  }, [rows, sort]);
  const columns: GridColumn[] = COLUMNS.map((column, index) => ({
    title: sort.key === column.key ? `${column.title} ${sort.direction === "desc" ? "↓" : "↑"}` : column.title,
    id: column.key,
    width: index === 4 ? 220 : 140,
    grow: index === 4 ? 1 : 0,
  }));
  const getCellContent = ([col, row]: readonly [number, number]): GridCell => {
    const source = order[row];
    const item = source == null ? undefined : rows[source];
    const key = COLUMNS[col]?.key ?? "exam";
    const value = item == null ? "" : String(examValue(item, key));
    return {
      kind: GridCellKind.Text,
      data: value,
      displayData: value,
      allowOverlay: false,
      readonly: true,
      copyData: value,
    };
  };
  return (
    <DataEditor
      width={width}
      height={height}
      columns={columns}
      rows={rows.length}
      rowHeight={28}
      headerHeight={28}
      getCellContent={getCellContent}
      onHeaderClicked={(col) => {
        const key = COLUMNS[col]?.key;
        if (key == null) {
          return;
        }
        setSort((current) =>
          current.key === key
            ? { key, direction: current.direction === "asc" ? "desc" : "asc" }
            : { key, direction: "asc" },
        );
      }}
      theme={theme}
      gridSelection={gridSelection}
      onGridSelectionChange={setGridSelection}
      columnSelect="none"
      rowMarkers="none"
      smoothScrollX
      smoothScrollY
    />
  );
}
