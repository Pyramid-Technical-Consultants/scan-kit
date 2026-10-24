import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { Toaster, toast } from "@/components/ui/toast";
import { clearLog, logLines } from "./debug-log";
import { notify, notifyError, notifySaved } from "./notify";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => undefined),
}));

let root: Root | null = null;

afterEach(() => {
  act(() => {
    toast.close();
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

it("logs a repeated analysis failure once and a new action every time", () => {
  clearLog();
  notifyError("plot failed", "analysis");
  notifyError("plot failed", "analysis");
  expect(logLines().filter((line) => line.includes("plot failed"))).toHaveLength(1);
  notifyError("folder failed");
  notifyError("folder failed");
  expect(logLines().filter((line) => line.includes("folder failed"))).toHaveLength(2);
});

it("can close a notice and open the folder of a saved file", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<Toaster />);
  });
  await act(async () => {
    notify("Copied session 1");
    notifySaved("C:/maps/field.csv", { title: "Input map saved" });
    notifySaved("C:/out/synthetic_slabs", { title: "Study written", folder: true });
  });
  expect(document.body.querySelector("[aria-label='Close toast']")).not.toBeNull();
  expect(document.body.textContent).toContain("field.csv");
  const show = [...document.body.querySelectorAll("button")].find((button) =>
    button.textContent?.includes("Show in folder"),
  );
  const openFolder = [...document.body.querySelectorAll("button")].find((button) =>
    button.textContent?.includes("Open folder"),
  );
  expect(show).toBeTruthy();
  expect(openFolder).toBeTruthy();
  await act(async () => {
    show?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(invoke).toHaveBeenCalledWith("scan_kit_reveal", { path: "C:/maps/field.csv" });
});
