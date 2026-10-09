import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it } from "vitest";

import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSeparator,
  FieldSet,
  FieldTitle,
} from "@/components/ui/field";
import {
  Menubar,
  MenubarCheckboxItem,
  MenubarContent,
  MenubarGroup,
  MenubarItem,
  MenubarLabel,
  MenubarMenu,
  MenubarPortal,
  MenubarRadioGroup,
  MenubarRadioItem,
  MenubarSeparator,
  MenubarShortcut,
  MenubarSub,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarTrigger,
} from "@/components/ui/menubar";
import { Select, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

it("renders the stock controls the shell does not open on its own", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(
      <div>
        <Textarea defaultValue="note" />
        <Card size="sm">
          <CardHeader>
            <CardTitle>Title</CardTitle>
            <CardDescription>Detail</CardDescription>
            <CardAction>Go</CardAction>
          </CardHeader>
          <CardContent>Body</CardContent>
          <CardFooter>Foot</CardFooter>
        </Card>
        <Dialog open>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Dialog</DialogTitle>
              <DialogDescription>Explain</DialogDescription>
            </DialogHeader>
            <DialogFooter>Done</DialogFooter>
          </DialogContent>
        </Dialog>
        <FieldSet>
          <FieldLegend>Legend</FieldLegend>
          <FieldLegend variant="label">Label</FieldLegend>
          <FieldGroup>
            <Field orientation="horizontal">
              <FieldLabel>Name</FieldLabel>
              <FieldContent>
                <FieldTitle>Title</FieldTitle>
                <FieldDescription>Help</FieldDescription>
              </FieldContent>
            </Field>
            <FieldSeparator>Or</FieldSeparator>
            <FieldError errors={[{ message: "One" }]} />
            <FieldError errors={[{ message: "One" }, { message: "Two" }]} />
            <FieldError>Child</FieldError>
            <FieldError />
          </FieldGroup>
        </FieldSet>
        <Menubar>
          <MenubarMenu open>
            <MenubarTrigger>File</MenubarTrigger>
            <MenubarPortal>
              <MenubarContent>
                <MenubarGroup>
                  <MenubarLabel inset>Label</MenubarLabel>
                  <MenubarItem inset variant="destructive">
                    Open
                    <MenubarShortcut>Ctrl+O</MenubarShortcut>
                  </MenubarItem>
                  <MenubarCheckboxItem checked inset>
                    Box
                  </MenubarCheckboxItem>
                  <MenubarSeparator />
                  <MenubarRadioGroup value="a">
                    <MenubarRadioItem value="a">A</MenubarRadioItem>
                  </MenubarRadioGroup>
                  <MenubarSub>
                    <MenubarSubTrigger inset>More</MenubarSubTrigger>
                    <MenubarSubContent>
                      <MenubarItem>Nested</MenubarItem>
                    </MenubarSubContent>
                  </MenubarSub>
                </MenubarGroup>
              </MenubarContent>
            </MenubarPortal>
          </MenubarMenu>
        </Menubar>
        <Select>
          <SelectTrigger>
            <SelectValue placeholder="Pick" />
          </SelectTrigger>
          <SelectGroup>
            <SelectLabel>Group</SelectLabel>
            <SelectItem value="one">One</SelectItem>
          </SelectGroup>
        </Select>
      </div>,
    );
  });
  expect(document.body.textContent).toContain("Title");
  expect(document.body.textContent).toContain("One");
});
