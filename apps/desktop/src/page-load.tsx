import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

import { ProgressHairline } from "@/progress-line";
import { hairlineFraction } from "@/task-client";

type Entry = { done: number; total: number };

/// No entries hides the bar. Any unknown length sweeps. Otherwise the fullest fraction shows.
export function pageLoadFraction(entries: readonly Entry[]): Entry | null {
  if (entries.length === 0) {
    return null;
  }
  if (entries.some((entry) => hairlineFraction(entry.done, entry.total) == null)) {
    return { done: 0, total: 0 };
  }
  return entries.reduce((best, entry) => (entry.done / entry.total > best.done / best.total ? entry : best));
}

const PageLoadContext = createContext<((id: number, entry: Entry | null) => void) | null>(null);

let nextId = 1;

export function PageLoad({ children }: { children: ReactNode }) {
  const [entries, setEntries] = useState<ReadonlyMap<number, Entry>>(new Map());
  const set = useCallback((id: number, entry: Entry | null) => {
    setEntries((current) => {
      const previous = current.get(id);
      if (entry == null) {
        if (previous == null) {
          return current;
        }
        const next = new Map(current);
        next.delete(id);
        return next;
      }
      if (previous != null && previous.done === entry.done && previous.total === entry.total) {
        return current;
      }
      const next = new Map(current);
      next.set(id, entry);
      return next;
    });
  }, []);
  const shown = pageLoadFraction([...entries.values()]);
  return (
    <PageLoadContext.Provider value={set}>
      {shown != null ? <ProgressHairline done={shown.done} total={shown.total} /> : null}
      {children}
    </PageLoadContext.Provider>
  );
}

/// Register one load on the page bar. `active` false removes it. Unknown `total` sweeps.
export function usePageLoad(active: boolean, done = 0, total = 0) {
  const set = useContext(PageLoadContext);
  const id = useRef(0);
  useEffect(() => {
    if (set == null) {
      return;
    }
    if (id.current === 0) {
      id.current = nextId;
      nextId += 1;
    }
    const mine = id.current;
    set(mine, active ? { done, total } : null);
    return () => set(mine, null);
  }, [set, active, done, total]);
}
