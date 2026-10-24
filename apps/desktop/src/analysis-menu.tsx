import {
  ChartColumn,
  ChartScatter,
  Ellipsis,
  Layers,
  Play,
  ScrollText,
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

const CORE_ANALYSES = ["Timeline", "Bins", "Distribution", "Volumetric"] as const;

const SPECIALIZED_ANALYSES = ["Session Log Compare", "IC HV Transient Test"] as const;

const ANALYSIS_GROUPS = [
  { title: "Core Analysis", names: CORE_ANALYSES },
  { title: "Specialized Analysis", names: SPECIALIZED_ANALYSES },
] as const;

/** Session-list buttons, in the same order as Core Analysis. */
export const PRIMARY_ANALYSES = CORE_ANALYSES;

const ANALYSIS_ICONS: Record<string, LucideIcon> = {
  "Bins": ChartColumn,
  "Distribution": ChartScatter,
  "Timeline": Play,
  "Volumetric": Layers,
  "Session Log Compare": ScrollText,
  "IC HV Transient Test": Zap,
};

const ANALYSIS_IDS: Record<string, string> = {
  "Bins": "bins",
  "Distribution": "distribution",
  "Timeline": "timeline",
  "Volumetric": "volumetric",
  "Session Log Compare": "session_log_compare",
  "IC HV Transient Test": "ic_hv_transient",
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
