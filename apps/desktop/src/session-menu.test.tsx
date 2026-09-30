import { createRoot } from "react-dom/client";
import { act } from "react";
import { expect, it } from "vitest";

import { SessionContextMenu } from "./session-menu";

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
  expect(document.body.textContent).toContain("1022244633");
  expect(document.body.textContent).toContain("Copy Session ID");
  expect(document.body.textContent).toContain("Open in Config Tuning");
  await act(async () => {
    root.unmount();
  });
  host.remove();
});
