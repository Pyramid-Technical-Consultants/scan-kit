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
        onClose={() => {}}
        onCopy={() => {}}
        onTune={() => {}}
      />,
    );
  });
  const trigger = document.querySelector("[data-slot=dropdown-menu-trigger]");
  expect(trigger).toBeInstanceOf(HTMLElement);
  expect((trigger as HTMLElement).style.position).toBe("fixed");
  expect((trigger as HTMLElement).style.left).toBe("12px");
  expect((trigger as HTMLElement).style.top).toBe("24px");
  expect(document.body.textContent).toContain("1022244633");
  expect(document.body.textContent).toContain("Copy Session ID");
  expect(document.body.textContent).toContain("Open in Config Tuning");
  await act(async () => {
    root.unmount();
  });
  host.remove();
});
