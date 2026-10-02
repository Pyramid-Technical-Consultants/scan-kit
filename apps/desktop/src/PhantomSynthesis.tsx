import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { notify, notifyError } from "@/notify";
import { SidePane } from "@/SidePane";

type Choice = { value: string; label: string };
type Params = {
  phantom: string;
  position: string;
  pixel: number;
  slice: number;
  energies: number[];
  gantry: number;
  couch: number;
  spot_pitch: number;
  range_shifter_wet: number;
  mu_per_spot: number;
  fractions: number;
};
type Catalog = {
  phantoms: Choice[];
  positions: string[];
  energies: number[];
  defaults: Params;
  save_dir: string | null;
};

const IDLE =
  "Synthetic studies carry no patient data. Open one in Dose Volume with Open Study.";

export function PhantomSynthesis() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [params, setParams] = useState<Params | null>(null);
  const [summary, setSummary] = useState("");
  const [status, setStatus] = useState(IDLE);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void invoke<Catalog>("scan_kit_phantom_catalog")
      .then((next) => {
        setCatalog(next);
        setParams(next.defaults);
      })
      .catch((reason: unknown) => notifyError(reason));
  }, []);

  useEffect(() => {
    if (params == null) {
      return;
    }
    void invoke<{ summary: string }>("scan_kit_phantom_preview", { params })
      .then((next) => setSummary(next.summary))
      .catch((reason: unknown) => {
        setSummary(reason instanceof Error ? reason.message : String(reason));
      });
  }, [params]);

  if (catalog == null || params == null) {
    return null;
  }

  const energies = [...catalog.energies].reverse();
  const selected = params.energies;

  async function writeStudy() {
    if (params == null) {
      return;
    }
    const selectedDir = await open({
      directory: true,
      title: "Folder for the synthetic study",
      defaultPath: parentDir(catalog?.save_dir ?? null),
    });
    if (typeof selectedDir !== "string") {
      return;
    }
    setBusy(true);
    try {
      const written = await invoke<{ status: string }>("scan_kit_write_phantom", {
        parent: selectedDir,
        params,
      });
      setStatus(written.status);
      notify(written.status);
    } catch (reason: unknown) {
      notifyError(reason);
    } finally {
      setBusy(false);
    }
  }

  return (
    <SidePane
      main={
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-auto p-4">
          <p className="text-sm">{summary}</p>
        </div>
      }
      side={
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <FieldGroup>
        <ChoiceField
          label="Phantom"
          value={params.phantom}
          items={catalog.phantoms}
          onChange={(phantom) => setParams({ ...params, phantom })}
        />
        <ChoiceField
          label="Patient Position"
          value={params.position}
          items={catalog.positions.map((position) => ({ value: position, label: position }))}
          onChange={(position) => setParams({ ...params, position })}
        />
        <NumberField
          label="Pixel Spacing (mm)"
          value={params.pixel}
          step={0.25}
          onChange={(pixel) => setParams({ ...params, pixel })}
        />
        <NumberField
          label="Slice Thickness (mm)"
          value={params.slice}
          step={0.5}
          onChange={(slice) => setParams({ ...params, slice })}
        />
        <Field>
          <FieldLabel>Energy Layers (MeV)</FieldLabel>
          <div className="flex flex-wrap gap-1">
            <Button type="button" size="sm" variant="outline" onClick={() => setEnergies(catalog.energies)}>
              All
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => setEnergies(catalog.energies.filter((energy) => energy % 1 === 0))}
            >
              Whole MeV
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => setEnergies(catalog.energies.filter((energy) => energy % 10 === 0))}
            >
              10 MeV
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={() => setEnergies([140, 130, 120])}>
              140–120
            </Button>
          </div>
          <p className="text-muted-foreground text-sm">
            {selected.length === 1 ? "1 layer selected" : `${selected.length} layers selected`}
          </p>
          <div className="flex max-h-48 flex-col gap-1 overflow-auto">
            {energies.map((energy) => {
              const id = `phantom-energy-${energy}`;
              const checked = selected.includes(energy);
              return (
                <Field key={energy} orientation="horizontal">
                  <Checkbox
                    id={id}
                    className="cursor-pointer"
                    checked={checked}
                    onCheckedChange={(next) => {
                      const without = selected.filter((item) => item !== energy);
                      setEnergies(next ? [...without, energy] : without);
                    }}
                  />
                  <FieldLabel className="cursor-pointer" htmlFor={id}>
                    {energy}
                  </FieldLabel>
                </Field>
              );
            })}
          </div>
        </Field>
        <NumberField
          label="Gantry Angle (°)"
          value={params.gantry}
          step={15}
          onChange={(gantry) => setParams({ ...params, gantry })}
        />
        <NumberField
          label="Couch Angle (°)"
          value={params.couch}
          step={15}
          onChange={(couch) => setParams({ ...params, couch })}
        />
        <NumberField
          label="Spot Pitch (mm)"
          value={params.spot_pitch}
          step={1}
          onChange={(spot_pitch) => setParams({ ...params, spot_pitch })}
        />
        <NumberField
          label="Range Shifter WET (mm)"
          value={params.range_shifter_wet}
          step={5}
          onChange={(range_shifter_wet) => setParams({ ...params, range_shifter_wet })}
        />
        <NumberField
          label="MU per Spot"
          value={params.mu_per_spot}
          step={0.01}
          onChange={(mu_per_spot) => setParams({ ...params, mu_per_spot })}
        />
        <NumberField
          label="Fractions"
          value={params.fractions}
          step={1}
          onChange={(fractions) => setParams({ ...params, fractions })}
        />
        <Button type="button" disabled={busy} onClick={() => void writeStudy()}>
          Write DICOM
        </Button>
      </FieldGroup>
      <p className="text-muted-foreground text-sm">{status}</p>
      </div>
      }
    />
  );

  function setEnergies(next: number[]) {
    setParams({ ...params!, energies: next });
  }
}

function ChoiceField({
  label,
  value,
  items,
  onChange,
}: {
  label: string;
  value: string;
  items: Choice[];
  onChange: (value: string) => void;
}) {
  return (
    <Field>
      <FieldLabel>{label}</FieldLabel>
      <Select items={items} value={value} onValueChange={(next) => next != null && onChange(next)}>
        <SelectTrigger className="w-full cursor-pointer">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {items.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </Field>
  );
}

function NumberField({
  label,
  value,
  step,
  onChange,
}: {
  label: string;
  value: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <Field>
      <FieldLabel>{label}</FieldLabel>
      <Input
        type="number"
        step={step}
        value={Number.isFinite(value) ? String(value) : ""}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
            onChange(next);
          }
        }}
      />
    </Field>
  );
}

function parentDir(path: string | null): string | undefined {
  if (path == null || path.length === 0) {
    return undefined;
  }
  const trimmed = path.replace(/[\\/]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return cut > 0 ? trimmed.slice(0, cut) : undefined;
}
