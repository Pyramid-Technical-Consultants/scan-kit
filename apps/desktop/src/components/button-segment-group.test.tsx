import { createRoot, type Root } from "react-dom/client";
import { act, useState } from "react";
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

it("presses Monte Carlo when the value is the mc id", () => {
  const host = document.createElement("div");
  document.body.append(host);
  act(() => {
    root = createRoot(host);
    root.render(
      <ButtonSegmentGroup
        options={[
          { id: "analytic", label: "Analytic" },
          { id: "mc", label: "Monte Carlo" },
        ]}
        value="mc"
        onChange={() => undefined}
      />,
    );
  });
  expect(pressed(host)).toContain("Monte Carlo");
});

it("presses Monte Carlo and reports that label", () => {
  const host = document.createElement("div");
  document.body.append(host);
  const chosen: string[] = [];
  function Harness() {
    const [value, setValue] = useState("Analytic");
    return (
      <ButtonSegmentGroup
        options={[
          { label: "Analytic", icon: "analytic" },
          { label: "Monte Carlo", icon: "mc" },
        ]}
        value={value}
        onChange={(next) => {
          chosen.push(next);
          setValue(next);
        }}
      />
    );
  }
  act(() => {
    root = createRoot(host);
    root.render(<Harness />);
  });
  const monteCarlo = () =>
    [...host.querySelectorAll("button")].find((node) => node.textContent?.includes("Monte Carlo"));
  act(() => {
    monteCarlo()?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(chosen).toEqual(["Monte Carlo"]);
  expect(monteCarlo()?.getAttribute("aria-pressed")).toBe("true");
});

it("uses the compact select height and corner radius", () => {
  const host = render("Both");
  const group = host.querySelector("[data-slot='toggle-group']");
  const item = host.querySelector("[data-slot='toggle-group-item']");
  expect(group?.getAttribute("data-size")).toBe("sm");
  expect(item?.className).toContain("h-7");
  expect(item?.className).toContain("rounded-l-[min(var(--radius-md),10px)]");
  expect(item?.className).not.toContain("h-8");
});
