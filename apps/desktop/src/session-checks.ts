export const MAX_SELECTED = 5;

export function headerCheck(
  rowCount: number,
  selectedCount: number,
): { checked: boolean; indeterminate: boolean } {
  if (selectedCount <= 0 || rowCount <= 0) {
    return { checked: false, indeterminate: false };
  }
  if (selectedCount >= rowCount) {
    return { checked: true, indeterminate: false };
  }
  return { checked: false, indeterminate: true };
}

/** A click on the header checkbox. Indeterminate clicks report checked, so the mixed state clears once the cap is full. */
export function headerWillFill(rowCount: number, selectedCount: number): boolean {
  if (rowCount === 0 || selectedCount === 0) {
    return true;
  }
  if (selectedCount >= rowCount) {
    return false;
  }
  if (rowCount > MAX_SELECTED && selectedCount >= MAX_SELECTED) {
    return false;
  }
  return true;
}

export function nextSessionSelection(
  idsInOrder: readonly string[],
  selected: readonly string[],
  target: string | "all",
  checked: boolean,
): { ids: string[]; capped: boolean } {
  if (target === "all") {
    if (!checked) {
      return { ids: [], capped: false };
    }
    return {
      ids: idsInOrder.slice(0, MAX_SELECTED),
      capped: idsInOrder.length > MAX_SELECTED,
    };
  }
  if (!checked) {
    return { ids: selected.filter((id) => id !== target), capped: false };
  }
  if (selected.includes(target)) {
    return { ids: [...selected], capped: false };
  }
  if (selected.length >= MAX_SELECTED) {
    return { ids: [...selected], capped: true };
  }
  return { ids: [...selected, target], capped: false };
}

/** A double-click toggles the row once. The checkbox click already counted, so its second event does not. */
export function shouldToggleRow(column: number, isDoubleClick: boolean, justToggled: boolean): boolean {
  if (isDoubleClick) {
    return !justToggled;
  }
  return column === 0;
}

/** Check the listed rows, or clear them when every one is already checked. */
export function toggleListedSessions(
  selected: readonly string[],
  targets: readonly string[],
): { ids: string[]; capped: boolean } {
  const wanted = targets.filter((id) => id !== "");
  const allOn = wanted.length > 0 && wanted.every((id) => selected.includes(id));
  if (allOn) {
    const drop = new Set(wanted);
    return { ids: selected.filter((id) => !drop.has(id)), capped: false };
  }
  const ids = [...selected];
  let capped = false;
  for (const id of wanted) {
    if (ids.includes(id)) {
      continue;
    }
    if (ids.length >= MAX_SELECTED) {
      capped = true;
      break;
    }
    ids.push(id);
  }
  return { ids, capped };
}
