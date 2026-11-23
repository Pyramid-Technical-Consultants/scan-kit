import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { LoaderCircle, Radar } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { notifyError } from "@/notify";

export interface DiscoveredDevice {
  host: string;
  version?: unknown;
  device_type?: unknown;
}

interface DevicePickerProps {
  value: string;
  onChange: (host: string) => void;
  onConnect: () => void;
  disabled?: boolean;
  busy?: boolean;
  placeholder?: string;
}

function label(part: unknown): string {
  if (typeof part === "string" && part.length > 0) {
    return part;
  }
  if (typeof part === "number") {
    return part.toString();
  }
  return "";
}

/** Subnet to scan, derived from the typed host or the IGX default subnet. */
function defaultSubnet(host: string): string {
  // URL-form hosts ("http://10.4.2.55/io/") have no octet at position 0.
  const match = /(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}/.exec(host.trim());
  return match ? `${match[1]}.0/24` : "192.168.100.0/24";
}

/**
 * Device selection row: type a host IP/URL, or scan a subnet and pick a
 * discovered device from the list.
 */
export function DevicePicker({
  value,
  onChange,
  onConnect,
  disabled = false,
  busy = false,
  placeholder = "192.168.100.184",
}: DevicePickerProps) {
  const [scanOpen, setScanOpen] = useState(false);
  const [subnet, setSubnet] = useState(() => defaultSubnet(value));
  const [scanning, setScanning] = useState(false);
  const [hits, setHits] = useState<{ subnet: string; devices: DiscoveredDevice[] } | null>(null);

  async function scan() {
    setScanning(true);
    const scanned = subnet.trim();
    try {
      const result = await invoke<{ devices: DiscoveredDevice[] }>("scan_kit_discover_subnet", {
        subnet: scanned,
      });
      // Remember which subnet produced these — a scan resolving after the
      // field was edited must not list hits under the new range.
      setHits({ subnet: scanned, devices: result.devices });
    } catch (error) {
      setHits(null);
      notifyError(error);
    } finally {
      setScanning(false);
    }
  }

  function openScan() {
    if (!scanOpen && value.trim()) {
      setSubnet(defaultSubnet(value));
    }
    setScanOpen(!scanOpen);
  }

  return (
    <div className="flex flex-col gap-2">
      <InputGroup>
        <InputGroupInput
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !busy && !event.nativeEvent.isComposing) {
              onConnect();
            }
          }}
        />
        <InputGroupAddon align="inline-end">
          <InputGroupButton
            variant="ghost"
            title="Scan the network for devices"
            aria-label="Scan the network for devices"
            aria-expanded={scanOpen}
            onClick={openScan}
            disabled={disabled}
          >
            {scanning ? <LoaderCircle className="animate-spin" /> : <Radar />}
          </InputGroupButton>
          <InputGroupButton
            variant="secondary"
            onClick={onConnect}
            disabled={busy || disabled}
          >
            Connect
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>
      {scanOpen ? (
        <div className="flex flex-col gap-2 rounded-lg border border-border p-2">
          <InputGroup>
            <InputGroupInput
              value={subnet}
              placeholder="192.168.100.0/24"
              onChange={(event) => {
                setSubnet(event.target.value);
                // Hits belong to the subnet that produced them — drop them
                // when it changes so stale results aren't listed under a
                // new (or still scanning) range.
                setHits(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !scanning && !event.nativeEvent.isComposing) {
                  void scan();
                }
              }}
            />
            <InputGroupAddon align="inline-end">
              <InputGroupButton variant="secondary" onClick={() => void scan()} disabled={scanning}>
                {scanning ? "Scanning…" : "Scan"}
              </InputGroupButton>
            </InputGroupAddon>
          </InputGroup>
          {hits !== null && hits.subnet === subnet.trim() ? (
            hits.devices.length === 0 ? (
              <p className="px-1 text-sm text-muted-foreground">No devices found on {subnet}.</p>
            ) : (
              <ul className="flex flex-col">
                {hits.devices.map((hit) => {
                  const detail = [label(hit.device_type), label(hit.version)]
                    .filter(Boolean)
                    .join(" · ");
                  return (
                    <li key={hit.host}>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start"
                        onClick={() => {
                          onChange(hit.host);
                          setScanOpen(false);
                        }}
                      >
                        <span className="font-medium">{hit.host}</span>
                        {detail ? <span className="text-muted-foreground">{detail}</span> : null}
                      </Button>
                    </li>
                  );
                })}
              </ul>
            )
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
