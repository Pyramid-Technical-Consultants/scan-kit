import { useEffect, useMemo, useRef, useState, type WheelEvent } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  DataEditor,
  GridCellKind,
  type GridCell,
  type GridColumn,
  type Item,
} from "@glideapps/glide-data-grid";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { gridTheme } from "@/grid-theme";

export type FormField = {
  id: string;
  label: string;
  kind: string;
  value: string;
  dead?: boolean;
  tooltip?: string;
};
export type FormColumn = { name: string; label: string; dead?: boolean; tooltip?: string };
export type FormNode = {
  kind: string;
  title?: string;
  dead?: boolean;
  nodes?: FormNode[];
  fields?: FormField[];
  id?: string;
  indices?: number[];
  columns?: FormColumn[];
  rows?: string[][];
  collapsible?: boolean;
};
export type FormDoc = { root: string; nodes: FormNode[] };

function mapNodes(nodes: FormNode[], visit: (node: FormNode) => FormNode): FormNode[] {
  return nodes.map((node) => {
    const next = node.nodes == null ? node : { ...node, nodes: mapNodes(node.nodes, visit) };
    return visit(next);
  });
}

export function editField(nodes: FormNode[], id: string, value: string): FormNode[] {
  return mapNodes(nodes, (node) => {
    if (node.fields == null) {
      return node;
    }
    return {
      ...node,
      fields: node.fields.map((field) => (field.id === id ? { ...field, value } : field)),
    };
  });
}

export function fieldBoxClass(field: { kind: string; value: string }): string {
  if (field.kind === "bool") {
    return "w-fit";
  }
  if (field.kind === "int" || field.kind === "float") {
    return "w-28";
  }
  if (field.value.length > 40) {
    return "w-full max-w-lg";
  }
  if (field.value.length > 16) {
    return "w-56";
  }
  return "w-36";
}

export function gridHeight(rows: number): number {
  return Math.min(360, 36 + Math.max(rows, 1) * 28);
}

export function forwardWheel(event: WheelEvent<HTMLDivElement>) {
  if (event.deltaY === 0) {
    return;
  }
  const scroller = event.currentTarget.querySelector(".dvn-scroller");
  const pane = event.currentTarget.closest("[data-form-scroll]");
  if (!(pane instanceof HTMLElement) || !(scroller instanceof HTMLElement)) {
    return;
  }
  const max = scroller.scrollHeight - scroller.clientHeight;
  const atTop = scroller.scrollTop <= 0;
  const atBottom = scroller.scrollTop >= max - 1;
  const stuck = max <= 1 || (event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom);
  if (!stuck) {
    return;
  }
  event.preventDefault();
  pane.scrollTop += event.deltaY;
}

export function sourceColumn(
  columns: readonly { dead?: boolean }[],
  visible: number,
  hideUnused: boolean,
): number {
  if (!hideUnused) {
    return visible;
  }
  let seen = 0;
  for (let index = 0; index < columns.length; index += 1) {
    if (columns[index]?.dead) {
      continue;
    }
    if (seen === visible) {
      return index;
    }
    seen += 1;
  }
  return visible;
}

export function editCell(nodes: FormNode[], id: string, row: number, column: number, value: string): FormNode[] {
  return mapNodes(nodes, (node) => {
    if (node.kind !== "table" || node.id !== id || node.rows == null) {
      return node;
    }
    const rows = node.rows.map((cells, index) => {
      if (index !== row) {
        return cells;
      }
      const next = cells.slice();
      next[column] = value;
      return next;
    });
    return { ...node, rows };
  });
}

export function FormTree({
  nodes,
  hideUnused,
  onField,
  onCell,
}: {
  nodes: FormNode[];
  hideUnused: boolean;
  onField: (id: string, value: string) => void;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {nodes.map((node, index) => (
        <FormNodeView
          key={`${node.kind}-${node.id ?? node.title ?? index}`}
          node={node}
          hideUnused={hideUnused}
          onField={onField}
          onCell={onCell}
        />
      ))}
    </div>
  );
}

function FormNodeView({
  node,
  hideUnused,
  onField,
  onCell,
}: {
  node: FormNode;
  hideUnused: boolean;
  onField: (id: string, value: string) => void;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  if (hideUnused && node.dead) {
    return null;
  }
  if (node.kind === "section") {
    return <FormSection node={node} hideUnused={hideUnused} onField={onField} onCell={onCell} />;
  }
  if (node.kind === "fields") {
    const fields = (node.fields ?? []).filter((field) => !(hideUnused && field.dead));
    if (fields.length === 0) {
      return null;
    }
    return (
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        {fields.map((field) =>
          field.kind === "bool" ? (
            <Field key={field.id} orientation="horizontal" className={fieldBoxClass(field)}>
              <Checkbox
                id={field.id}
                checked={field.value.toLowerCase() === "true"}
                onCheckedChange={(next) => onField(field.id, next === true ? "true" : "false")}
              />
              <FieldLabel htmlFor={field.id} title={field.tooltip}>
                {field.label}
              </FieldLabel>
            </Field>
          ) : (
            <Field key={field.id} className={fieldBoxClass(field)}>
              <FieldLabel title={field.tooltip}>{field.label}</FieldLabel>
              <Input value={field.value} onChange={(event) => onField(field.id, event.target.value)} />
            </Field>
          ),
        )}
      </div>
    );
  }
  if (node.kind === "table" && node.id != null) {
    return (
      <XmlGrid
        id={node.id}
        title={node.title ?? "Table"}
        columns={node.columns ?? []}
        rows={node.rows ?? []}
        hideUnused={hideUnused}
        onCell={onCell}
      />
    );
  }
  return null;
}

function FormSection({
  node,
  hideUnused,
  onField,
  onCell,
}: {
  node: FormNode;
  hideUnused: boolean;
  onField: (id: string, value: string) => void;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  const [open, setOpen] = useState(!node.collapsible);
  const body =
    !node.collapsible || open ? (
      <CardContent>
        <FormTree nodes={node.nodes ?? []} hideUnused={hideUnused} onField={onField} onCell={onCell} />
      </CardContent>
    ) : null;
  return (
    <Card size="sm">
      <CardHeader>
        {node.collapsible ? (
          <Button
            type="button"
            variant="ghost"
            className="w-full justify-start"
            aria-expanded={open}
            onClick={() => setOpen((current) => !current)}
          >
            {open ? <ChevronDown /> : <ChevronRight />}
            {node.title}
          </Button>
        ) : (
          <CardTitle>{node.title}</CardTitle>
        )}
      </CardHeader>
      {body}
    </Card>
  );
}

function XmlGrid({
  id,
  title,
  columns,
  rows,
  hideUnused,
  onCell,
}: {
  id: string;
  title: string;
  columns: FormColumn[];
  rows: string[][];
  hideUnused: boolean;
  onCell: (id: string, cell: Item, value: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const theme = useMemo(() => gridTheme(), []);
  const shown = useMemo(
    () =>
      columns
        .map((column, index) => ({ column, index }))
        .filter((item) => !(hideUnused && item.column.dead)),
    [columns, hideUnused],
  );
  const gridColumns = useMemo<GridColumn[]>(
    () =>
      shown.map((item) => ({
        title: item.column.label,
        id: item.column.name,
        width: Math.min(180, Math.max(88, item.column.label.length * 9 + 28)),
      })),
    [shown],
  );
  useEffect(() => {
    const node = host.current;
    if (node == null) {
      return;
    }
    const measure = () => setWidth(node.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  function getCell([column, row]: Item): GridCell {
    const source = shown[column]?.index ?? column;
    const text = rows[row]?.[source] ?? "";
    return {
      kind: GridCellKind.Text,
      data: text,
      displayData: text,
      allowOverlay: true,
    };
  }
  const height = gridHeight(rows.length);
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <p className="text-muted-foreground text-sm">{title}</p>
      <div
        ref={host}
        className="w-full min-w-0 overflow-hidden"
        style={{ height, overflowAnchor: "none" }}
        onWheel={forwardWheel}
      >
        {width > 0 ? (
          <DataEditor
            width={width}
            height={height}
            columns={gridColumns}
            rows={rows.length}
            getCellContent={getCell}
            theme={theme}
            rowMarkers="number"
            scrollToActiveCell={false}
            onCellEdited={(cell, value) => {
              if (value.kind === GridCellKind.Text) {
                const source = sourceColumn(columns, cell[0], hideUnused);
                onCell(id, [source, cell[1]], value.data);
              }
            }}
          />
        ) : null}
      </div>
    </div>
  );
}
