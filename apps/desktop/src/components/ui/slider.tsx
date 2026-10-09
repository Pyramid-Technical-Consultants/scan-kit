import { Slider as SliderPrimitive } from "@base-ui/react/slider"
import { cn } from "cn"

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  marks = [],
  ...props
}: SliderPrimitive.Root.Props & {
  /** Times on the same scale as `value`. Drawn on the track, under the thumb. */
  marks?: readonly number[]
}) {
  const _values = Array.isArray(value)
    ? value
    : Array.isArray(defaultValue)
      ? defaultValue
      : [min, max]

  return (
    <SliderPrimitive.Root
      className={cn("data-horizontal:w-full data-vertical:h-full", className)}
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      thumbAlignment="edge"
      {...props}
    >
      <SliderPrimitive.Control className="relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:min-h-40 data-vertical:w-auto data-vertical:flex-col">
        <SliderPrimitive.Track
          data-slot="slider-track"
          className="relative grow overflow-hidden rounded-full bg-muted select-none data-horizontal:h-1 data-horizontal:w-full data-vertical:h-full data-vertical:w-1"
        >
          <SliderPrimitive.Indicator
            data-slot="slider-range"
            className="bg-primary select-none data-horizontal:h-full data-vertical:w-full"
          />
          {marks.map((mark) => {
            const span = max - min
            const fraction = span > 0 ? Math.min(1, Math.max(0, (mark - min) / span)) : 0
            const at = _values[0] ?? min
            return (
              <span
                key={mark}
                data-slot="slider-mark"
                data-passed={mark <= at ? "true" : "false"}
                aria-hidden
                className={cn(
                  "pointer-events-none absolute inset-y-0 w-px -translate-x-1/2",
                  mark <= at ? "bg-primary-foreground" : "bg-primary"
                )}
                style={{
                  left: `calc(0.375rem + (100% - 0.75rem) * ${fraction})`,
                }}
              />
            )
          })}
        </SliderPrimitive.Track>
        {Array.from({ length: _values.length }, (_, index) => (
          <SliderPrimitive.Thumb
            data-slot="slider-thumb"
            key={index}
            className="relative block size-3 shrink-0 rounded-full border border-ring bg-white ring-ring/50 transition-[color,box-shadow] select-none after:absolute after:-inset-2 hover:ring-3 focus-visible:ring-3 focus-visible:outline-hidden active:ring-3 disabled:pointer-events-none disabled:opacity-50"
          />
        ))}
      </SliderPrimitive.Control>
    </SliderPrimitive.Root>
  )
}

export { Slider }
