import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

import { CatalogField } from "@/CatalogField";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { notifyError, notifySaved } from "@/notify";
import { usePageLoad } from "@/page-load";
import { SidePane } from "@/SidePane";

type Choice = { value: string; label: string };
type Params = {
  phantom: string;
  position: string;
  pixel: number | string;
  slice: number | string;
  energies: number[];
  gantry: number | string;
  couch: number | string;
  spot_pitch: number | string;
  range_shifter_wet: number | string;
  mu_per_spot: number | string;
  fractions: number | string;
};
type Catalog = {
  phantoms: Choice[];
  positions: string[];
  energies: number[];
  defaults: Params;
  save_dir: string | null;
};

const IDLE =
  "Synthetic studies carry no patient data. Open one in Volumetric with Open Study.";

export function PhantomSynthesis() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [params, setParams] = useState<Params | null>(null);
  const [summary, setSummary] = useState("");
  const [status, setStatus] = useState(IDLE);
  const [busy, setBusy] = useState(false);
  const [waiting, setWaiting] = useState(true);
  // catalog.save_dir is a mount-time value — a successful write moves the
  // remembered dir and the next dialog should follow.
  const [writtenDir, setWrittenDir] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  // A failed catalog load used to render a permanently blank tab — the pane
  // is keepMounted, so nothing ever re-asked for it.
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogRetry, setCatalogRetry] = useState(0);
  const writeBusy = useRef(false);
  const paramsRef = useRef(params);
  useLayoutEffect(() => {
    paramsRef.current = params;
  }, [params]);
  usePageLoad(waiting || busy || previewing);

  useEffect(() => {
    let live = true;
    void invoke<Catalog>("scan_kit_phantom_catalog")
      .then((next) => {
        if (!live) {
          return;
        }
        setCatalog(next);
        setParams(next.defaults);
        setWaiting(false);
      })
      .catch((reason: unknown) => {
        if (!live) {
          return;
        }
        setWaiting(false);
        setCatalogError(reason instanceof Error ? reason.message : String(reason));
        notifyError(reason);
      });
    return () => {
      live = false;
    };
  }, [catalogRetry]);

  useEffect(() => {
    if (params == null) {
      return;
    }
    let cancel = false;
    const arm = window.setTimeout(() => {
      if (cancel) {
        return;
      }
      setPreviewing(true);
      void invoke<{ summary: string }>("scan_kit_phantom_preview", { params })
        .then((next) => {
          if (!cancel) {
            setSummary(next.summary);
          }
        })
        .catch((reason: unknown) => {
          if (!cancel) {
            setSummary(reason instanceof Error ? reason.message : String(reason));
          }
        })
        .finally(() => {
          if (!cancel) {
            setPreviewing(false);
          }
        });
    }, 250);
    return () => {
      cancel = true;
      window.clearTimeout(arm);
      // A cancelled request's finally skips the clear — drop the hairline
      // during the debounce gap or it stays lit until the next preview ends.
      setPreviewing(false);
    };
  }, [params]);

  if (catalog == null || params == null) {
    return catalogError == null ? null : (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6">
        <p className="text-muted-foreground text-sm">{catalogError}</p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            setWaiting(true);
            setCatalogError(null);
            setCatalogRetry((n) => n + 1);
          }}
        >
          Retry
        </Button>
      </div>
    );
  }

  const energies = [...catalog.energies].reverse();
  const selected = params.energies;

  async function writeStudy() {
    // The dialog resolves before busy arms — without this a double-click
    // opens two dialogs and writes the study twice.
    if (writeBusy.current || params == null) {
      return;
    }
    writeBusy.current = true;
    const selectedDir = await open({
      directory: true,
      title: "Folder for the synthetic study",
      defaultPath: parentDir(writtenDir ?? catalog?.save_dir ?? null),
    }).catch((reason: unknown) => {
      notifyError(reason);
      return null;
    });
    if (typeof selectedDir !== "string") {
      writeBusy.current = false;
      return;
    }
    setBusy(true);
    try {
      // Field edits can land while the native dialog is open — write the
      // freshest params, not the pre-dialog snapshot.
      const written = await invoke<{ status: string; folder: string }>("scan_kit_write_phantom", {
        parent: selectedDir,
        params: paramsRef.current,
      });
      setStatus(written.status);
      setWrittenDir(parentDir(written.folder) ?? null);
      notifySaved(written.folder, { title: "Study written", folder: true });
    } catch (reason: unknown) {
      notifyError(reason);
    } finally {
      writeBusy.current = false;
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
          onChange={(pixel) => setParams({ ...params, pixel: pixel as Params["pixel"] })}
        />
        <CatalogField
          param={{ label: "Slice Thickness (mm)", kind: "float", step: 0.5 }}
          value={params.slice}
          finiteOnly
          onChange={(slice) => setParams({ ...params, slice: slice as Params["slice"] })}
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
            <Button type="button" size="sm" variant="outline" onClick={() => setEnergies([140, 130, 120].filter((energy) => catalog.energies.includes(energy)))}>
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
          onChange={(gantry) => setParams({ ...params, gantry: gantry as Params["gantry"] })}
        />
        <CatalogField
          param={{ label: "Couch Angle (°)", kind: "float", step: 15 }}
          value={params.couch}
          finiteOnly
          onChange={(couch) => setParams({ ...params, couch: couch as Params["couch"] })}
        />
        <CatalogField
          param={{ label: "Spot Pitch (mm)", kind: "float", step: 1 }}
          value={params.spot_pitch}
          finiteOnly
          onChange={(spot_pitch) => setParams({ ...params, spot_pitch: spot_pitch as Params["spot_pitch"] })}
        />
        <CatalogField
          param={{ label: "Range Shifter WET (mm)", kind: "float", step: 5 }}
          value={params.range_shifter_wet}
          finiteOnly
          onChange={(range_shifter_wet) => setParams({ ...params, range_shifter_wet: range_shifter_wet as Params["range_shifter_wet"] })}
        />
        <CatalogField
          param={{ label: "MU per Spot", kind: "float", step: 0.01 }}
          value={params.mu_per_spot}
          finiteOnly
          onChange={(mu_per_spot) => setParams({ ...params, mu_per_spot: mu_per_spot as Params["mu_per_spot"] })}
        />
        <CatalogField
          param={{ label: "Fractions", kind: "float", step: 1 }}
          value={params.fractions}
          finiteOnly
          onChange={(fractions) => setParams({ ...params, fractions: fractions as Params["fractions"] })}
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
  // A drive-root child has its separator at index 1 - the parent is
  // "C:\", not the drive-relative "C:".
  if (cut === 2 && trimmed[1] === ":") {
    return trimmed.slice(0, 3);
  }
  return cut > 0 ? trimmed.slice(0, cut) : undefined;
}
