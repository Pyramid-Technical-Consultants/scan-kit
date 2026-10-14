import { createRoot } from "react-dom/client";
import { act } from "react";
import { expect, it } from "vitest";

import { SessionContextMenu, sessionMenuPoint } from "./session-menu";

it("uses Glide's viewport cell bounds as the click point", () => {
  expect(sessionMenuPoint({ x: 400, y: 250 }, 12, 8)).toEqual({ x: 412, y: 258 });
});

it("opens the session menu without throwing", async () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  await act(async () => {
    root.render(
      <SessionContextMenu
        sessionId="1022244633"
        x={12}
        y={24}
        rowIds={["1022244633"]}
        rowsSelected={false}
        onClose={() => {}}
        onCopy={() => {}}
        onTune={() => {}}
        onToggleRows={() => {}}
      />,
    );
  });
  const trigger = document.querySelector("[data-slot=dropdown-menu-trigger]");
  expect(trigger).toBeInstanceOf(HTMLElement);
  expect((trigger as HTMLElement).style.position).toBe("fixed");
  expect((trigger as HTMLElement).style.left).toBe("12px");
  expect((trigger as HTMLElement).style.top).toBe("24px");
  expect(document.body.textContent).toContain("1022244633");
  expect(document.body.textContent).toContain("Select Row");
  expect(document.body.textContent).toContain("Copy Session ID");
  expect(document.body.textContent).toContain("Open in Config Tuning");
  const content = document.querySelector("[data-slot=dropdown-menu-content]");
  expect(content).toBeInstanceOf(HTMLElement);
  expect((content as HTMLElement).className).toContain("w-max");
  expect((content as HTMLElement).className).not.toContain("anchor-width");
  const item = document.querySelector("[data-slot=dropdown-menu-item]");
  expect(item).toBeInstanceOf(HTMLElement);
  expect((item as HTMLElement).className).toContain("whitespace-nowrap");
  const copied: string[] = [];
  const tuned: number[] = [];
  const toggled: number[] = [];
  let closed = 0;
  await act(async () => {
    root.render(
      <SessionContextMenu
        sessionId="1022244633"
        x={12}
        y={24}
        rowIds={["1022244633", "1022244634"]}
        rowsSelected
        onClose={() => {
          closed += 1;
        }}
        onCopy={(id) => copied.push(id)}
        onTune={() => tuned.push(1)}
        onToggleRows={() => toggled.push(1)}
      />,
    );
  });
  expect(document.body.textContent).toContain("Deselect Rows");
  const click = (selector: string) => {
    const node = document.querySelector(selector);
    expect(node).toBeInstanceOf(HTMLElement);
    (node as HTMLElement).dispatchEvent(new MouseEvent("click", { bubbles: true }));
  };
  await act(async () => {
    click("[data-slot=dropdown-menu-checkbox-item]");
    click("[data-slot=dropdown-menu-item]");
  });
  const items = [...document.querySelectorAll("[data-slot=dropdown-menu-item]")];
  const tune = items.find((node) => node.textContent?.includes("Config Tuning"));
  await act(async () => {
    tune?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(copied).toEqual(["1022244633"]);
  expect(toggled).toEqual([1]);
  expect(tuned).toEqual([1]);
  expect(closed).toBeGreaterThanOrEqual(3);
  await act(async () => {
    root.unmount();
  });
  host.remove();
});
