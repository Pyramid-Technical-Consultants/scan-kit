import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

import { CatalogField } from "@/CatalogField";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { notifyError, notifySaved } from "@/notify";
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
      const written = await invoke<{ status: string; folder: string }>("scan_kit_write_phantom", {
        parent: selectedDir,
        params,
      });
      setStatus(written.status);
      notifySaved(written.folder, { title: "Study written", folder: true });
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
        <CatalogField
          param={{ label: "Phantom", kind: "choice", choices: catalog.phantoms }}
          value={params.phantom}
          onChange={(phantom) => setParams({ ...params, phantom: String(phantom) })}
        />
        <CatalogField
          param={{
            label: "Patient Position",
            kind: "choice",
            choices: catalog.positions.map((position) => ({ value: position, label: position })),
          }}
          value={params.position}
          onChange={(position) => setParams({ ...params, position: String(position) })}
        />
        <CatalogField
          param={{ label: "Pixel Spacing (mm)", kind: "float", step: 0.25 }}
          value={params.pixel}
          finiteOnly
          onChange={(pixel) => setParams({ ...params, pixel: Number(pixel) })}
        />
        <CatalogField
          param={{ label: "Slice Thickness (mm)", kind: "float", step: 0.5 }}
          value={params.slice}
          finiteOnly
          onChange={(slice) => setParams({ ...params, slice: Number(slice) })}
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
        <CatalogField
          param={{ label: "Gantry Angle (°)", kind: "float", step: 15 }}
          value={params.gantry}
          finiteOnly
          onChange={(gantry) => setParams({ ...params, gantry: Number(gantry) })}
        />
        <CatalogField
          param={{ label: "Couch Angle (°)", kind: "float", step: 15 }}
          value={params.couch}
          finiteOnly
          onChange={(couch) => setParams({ ...params, couch: Number(couch) })}
        />
        <CatalogField
          param={{ label: "Spot Pitch (mm)", kind: "float", step: 1 }}
          value={params.spot_pitch}
          finiteOnly
          onChange={(spot_pitch) => setParams({ ...params, spot_pitch: Number(spot_pitch) })}
        />
        <CatalogField
          param={{ label: "Range Shifter WET (mm)", kind: "float", step: 5 }}
          value={params.range_shifter_wet}
          finiteOnly
          onChange={(range_shifter_wet) => setParams({ ...params, range_shifter_wet: Number(range_shifter_wet) })}
        />
        <CatalogField
          param={{ label: "MU per Spot", kind: "float", step: 0.01 }}
          value={params.mu_per_spot}
          finiteOnly
          onChange={(mu_per_spot) => setParams({ ...params, mu_per_spot: Number(mu_per_spot) })}
        />
        <CatalogField
          param={{ label: "Fractions", kind: "float", step: 1 }}
          value={params.fractions}
          finiteOnly
          onChange={(fractions) => setParams({ ...params, fractions: Number(fractions) })}
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

function parentDir(path: string | null): string | undefined {
  if (path == null || path.length === 0) {
    return undefined;
  }
  const trimmed = path.replace(/[\\/]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return cut > 0 ? trimmed.slice(0, cut) : undefined;
}
