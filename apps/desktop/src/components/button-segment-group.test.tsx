import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it } from "vitest";

import { ButtonSegmentGroup } from "./button-segment-group";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

function render(value: string) {
  const host = document.createElement("div");
  document.body.append(host);
  act(() => {
    root = createRoot(host);
    root.render(
      <ButtonSegmentGroup
        options={["Beam on", "Beam off", "Both"]}
        value={value}
        onChange={() => undefined}
      />,
    );
  });
  return host;
}

function pressed(host: HTMLElement): string {
  return host.querySelector("button[aria-pressed='true']")?.textContent ?? "";
}

it("presses the matching segment", () => {
  expect(pressed(render("Both"))).toContain("Both");
});

it("presses the first segment when the value is not in the group", () => {
  expect(pressed(render("missing"))).toContain("Beam on");
});
