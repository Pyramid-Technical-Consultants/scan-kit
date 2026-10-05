import { useEffect, useRef, useState } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

type Scrub = {
  on: boolean;
  at: number;
  end: number;
  speed: 0.1 | 1 | 10;
  window: "second" | "before";
};

const SPEEDS = [
  { value: "0.1", label: "1/10" },
  { value: "1", label: "1×" },
  { value: "10", label: "10×" },
];

const WINDOWS = [
  { value: "second", label: "1 s" },
  { value: "before", label: "Before" },
];

function speedOf(value: unknown): Scrub["speed"] {
  if (typeof value !== "number") {
    return 1;
  }
  if (Math.abs(value - 0.1) < 1e-3) {
    return 0.1;
  }
  if (Math.abs(value - 10) < 1e-3) {
    return 10;
  }
  return 1;
}

function parseScrub(raw: string | undefined): Scrub {
  let value: Record<string, unknown> = {};
  if (raw != null && raw.length > 0) {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed != null && typeof parsed === "object") {
        value = parsed as Record<string, unknown>;
      }
    } catch {
      value = {};
    }
  }
  const end = typeof value.end === "number" && Number.isFinite(value.end) ? Math.max(0, value.end) : 0;
  const at = typeof value.at === "number" && Number.isFinite(value.at) ? value.at : 0;
  return {
    on: value.on === true,
    at: Math.min(Math.max(0, at), end),
    end,
    speed: speedOf(value.speed),
    window: value.window === "before" ? "before" : "second",
  };
}

function scrubText(scrub: Scrub): string {
  return JSON.stringify({
    on: scrub.on,
    at: scrub.at,
    end: scrub.end,
    speed: scrub.speed,
    window: scrub.window,
  });
}

function samePlay(left: Scrub, right: Scrub): boolean {
  return (
    left.on === right.on &&
    left.speed === right.speed &&
    left.window === right.window &&
    Math.abs(left.at - right.at) < 1e-4
  );
}

/** Keep one analysis task in flight and send the latest playhead when it settles. */
export function useScrub(
  controlValue: string | undefined,
  settled: number,
  commit: (text: string) => void,
): {
  scrub: Scrub;
  playing: boolean;
  shown: boolean;
  update: (next: Scrub) => void;
  togglePlay: () => void;
} {
  const reported = parseScrub(controlValue);
  const commitRef = useRef(commit);
  const [play, setPlay] = useState(() => reported);
  const [playing, setPlaying] = useState(false);
  const scrub: Scrub = {
    on: play.on,
    speed: play.speed,
    window: play.window,
    end: reported.end,
    at: Math.min(play.at, reported.end),
  };
  const scrubRef = useRef(scrub);
  const lastSent = useRef(scrub);
  const pending = useRef(false);

  useEffect(() => {
    commitRef.current = commit;
  }, [commit]);

  useEffect(() => {
    scrubRef.current = {
      on: scrub.on,
      at: scrub.at,
      end: scrub.end,
      speed: scrub.speed,
      window: scrub.window,
    };
  }, [scrub.on, scrub.at, scrub.end, scrub.speed, scrub.window]);

  useEffect(() => {
    pending.current = false;
  }, [settled]);

  useEffect(() => {
    const current: Scrub = {
      on: scrub.on,
      at: scrub.at,
      end: scrub.end,
      speed: scrub.speed,
      window: scrub.window,
    };
    if (controlValue == null) {
      return;
    }
    if (samePlay(lastSent.current, current)) {
      lastSent.current = { ...lastSent.current, end: current.end };
      return;
    }
    if (pending.current) {
      return;
    }
    pending.current = true;
    lastSent.current = current;
    commitRef.current(scrubText(current));
  }, [scrub.on, scrub.at, scrub.end, scrub.speed, scrub.window, settled, controlValue]);

  useEffect(() => {
    if (!playing) {
      return;
    }
    let last = performance.now();
    let frame = 0;
    const step = (now: number) => {
      const current = scrubRef.current;
      const dt = Math.max(0, (now - last) / 1000);
      last = now;
      const at = Math.min(current.end, current.at + current.speed * dt);
      if (at !== current.at) {
        const next = { ...current, at };
        scrubRef.current = next;
        setPlay(next);
      }
      if (at >= current.end) {
        setPlaying(false);
        return;
      }
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const update = (next: Scrub) => {
    if (!next.on || next.at !== scrubRef.current.at) {
      setPlaying(false);
    }
    scrubRef.current = next;
    setPlay(next);
  };

  return {
    scrub,
    playing,
    shown: controlValue != null,
    update,
    togglePlay: () => {
      if (scrubRef.current.on) {
        setPlaying((on) => !on);
      }
    },
  };
}

export function ScrubBar({
  scrub,
  playing,
  onChange,
  onTogglePlay,
}: {
  scrub: Scrub;
  playing: boolean;
  onChange: (next: Scrub) => void;
  onTogglePlay: () => void;
}) {
  const armed = scrub.on;
  const speed = scrub.speed === 0.1 ? "0.1" : scrub.speed === 10 ? "10" : "1";
  return (
    <div
      data-slot="scrub-bar"
      className="border-border flex shrink-0 flex-row items-center gap-2 border-t px-3 py-2"
    >
      <Checkbox
        checked={scrub.on}
        aria-label="Timeline"
        onCheckedChange={(checked) => onChange({ ...scrub, on: checked === true })}
      />
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled={!armed}
        aria-label="Skip to start"
        onClick={() => onChange({ ...scrub, at: 0 })}
      >
        <SkipBack />
      </Button>
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled={!armed}
        aria-label={playing ? "Pause" : "Play"}
        onClick={onTogglePlay}
      >
        {playing ? <Pause /> : <Play />}
      </Button>
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled={!armed}
        aria-label="Skip to end"
        onClick={() => onChange({ ...scrub, at: scrub.end })}
      >
        <SkipForward />
      </Button>
      <Slider
        className="min-w-0 flex-1"
        min={0}
        max={scrub.end}
        value={[scrub.at]}
        disabled={!armed}
        onValueChange={(next) => {
          const at = Array.isArray(next) ? next[0] : next;
          if (typeof at === "number") {
            onChange({ ...scrub, at });
          }
        }}
      />
      <Select
        items={SPEEDS}
        value={speed}
        disabled={!armed}
        onValueChange={(next) => {
          if (next === "0.1" || next === "1" || next === "10") {
            onChange({ ...scrub, speed: speedOf(Number(next)) });
          }
        }}
      >
        <SelectTrigger size="sm" className="w-24 cursor-pointer" aria-label="Speed">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {SPEEDS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
      <Select
        items={WINDOWS}
        value={scrub.window}
        disabled={!armed}
        onValueChange={(next) => {
          if (next === "second" || next === "before") {
            onChange({ ...scrub, window: next });
          }
        }}
      >
        <SelectTrigger size="sm" className="w-24 cursor-pointer" aria-label="Window">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {WINDOWS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
