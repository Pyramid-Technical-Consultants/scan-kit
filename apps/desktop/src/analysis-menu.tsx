import {
  Activity,
  AudioWaveform,
  ChartColumn,
  ChartScatter,
  Cuboid,
  Ellipsis,
  Headphones,
  Layers,
  Play,
  ScrollText,
  Spline,
  TrendingDown,
  TrendingUp,
  Waypoints,
  Zap,
  type LucideIcon,
} from "lucide-react";

import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const ANALYSIS_GROUPS = [
  {
    title: "Unified Views",
    names: [
      "Binned Summary",
      "Distribution Explorer",
      "Timeslice Replay",
      "FFT Explorer",
      "Audio Explorer",
      "IC Beam Trajectory (3D)",
      "Dose Volume",
      "Session Log Compare",
    ],
  },
  {
    title: "Specialized Analysis",
    names: [
      "Beam Error Motion vs Energy",
      "Dose Accumulation",
      "Beam-Off Ramp-Down",
      "IC HV Transient Test",
      "Amplifier Command Correlations",
      "IC Peak Amplitude — Beam-Off (G3)",
    ],
  },
] as const;

export const PRIMARY_ANALYSES = [
  "Binned Summary",
  "Distribution Explorer",
  "Timeslice Replay",
] as const;

const ANALYSIS_ICONS: Record<string, LucideIcon> = {
  "Binned Summary": ChartColumn,
  "Distribution Explorer": ChartScatter,
  "Timeslice Replay": Play,
  "FFT Explorer": AudioWaveform,
  "Audio Explorer": Headphones,
  "IC Beam Trajectory (3D)": Cuboid,
  "Dose Volume": Layers,
  "Session Log Compare": ScrollText,
  "Beam Error Motion vs Energy": Spline,
  "Dose Accumulation": TrendingUp,
  "Beam-Off Ramp-Down": TrendingDown,
  "IC HV Transient Test": Zap,
  "Amplifier Command Correlations": Waypoints,
  "IC Peak Amplitude — Beam-Off (G3)": Activity,
};

const ANALYSIS_IDS: Record<string, string> = {
  "Binned Summary": "binned_summary",
  "Distribution Explorer": "distribution",
  "Timeslice Replay": "timeslice_replay",
  "FFT Explorer": "ic_fft_analysis",
  "Audio Explorer": "ic_audio_player",
  "IC Beam Trajectory (3D)": "trajectory",
  "Dose Volume": "dose_volume",
  "Session Log Compare": "session_log_compare",
  "Beam Error Motion vs Energy": "beam_motion_energy",
  "Dose Accumulation": "dose_accumulation",
  "Beam-Off Ramp-Down": "beam_off_rampdown",
  "IC HV Transient Test": "ic_hv_transient",
  "Amplifier Command Correlations": "amplifier_correlation",
  "IC Peak Amplitude — Beam-Off (G3)": "ic_peak_amplitude_beam_off",
};

export function analysisId(name: string): string | undefined {
  return ANALYSIS_IDS[name];
}

export function analysisName(id: string): string | undefined {
  return Object.keys(ANALYSIS_IDS).find((name) => ANALYSIS_IDS[name] === id);
}

export function AnalysisIcon({ name }: { name: string }) {
  const Icon = ANALYSIS_ICONS[name];
  if (Icon == null) {
    return null;
  }
  return <Icon />;
}

function hidden(omit: readonly string[] | string | undefined, name: string): boolean {
  if (omit == null) {
    return false;
  }
  return typeof omit === "string" ? omit === name : omit.includes(name);
}

/** The session-list overflow menu. `omit` hides names that already have their own button. */
export function AnalysisMenu({
  disabled,
  omit,
  onOpen,
}: {
  disabled?: boolean;
  omit?: readonly string[] | string;
  onOpen: (name: string) => void;
}) {
  const groups = ANALYSIS_GROUPS.flatMap((group) => {
    const names = group.names.filter((name) => !hidden(omit, name));
    return names.length === 0 ? [] : [{ title: group.title, names }];
  });
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        className={buttonVariants({ variant: "outline", size: "icon-sm" })}
        aria-label="More analyses"
        disabled={disabled}
      >
        <Ellipsis />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-max">
        {groups.map((group, index) => (
          <DropdownMenuGroup key={group.title}>
            {index > 0 ? <DropdownMenuSeparator /> : null}
            <DropdownMenuLabel>{group.title}</DropdownMenuLabel>
            {group.names.map((name) => (
              <DropdownMenuItem
                key={name}
                className="whitespace-nowrap"
                onClick={() => onOpen(name)}
              >
                <AnalysisIcon name={name} />
                {name}
              </DropdownMenuItem>
            ))}
          </DropdownMenuGroup>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
