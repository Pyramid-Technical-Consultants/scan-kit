"""Session line colors (no matplotlib / pandas imports).

Used by the Qt launcher and re-exported from :mod:`scan_kit.common.plotting`.

Palette design notes
--------------------
These colors are assigned by session order (session 1 -> index 0, etc.) and are
used for overlaid line plots on a white background plus the launcher's "Use"
swatches.

The set is the seaborn "deep" qualitative palette: a muted, publication-grade
palette where every hue shares a similar saturation/lightness so no single line
visually dominates (the previous mix of ``skyblue``/``limegreen``/``purple`` did
not). Properties we rely on:

* Ordered blue -> orange -> green -> red so the most common 1-3 session overlays
  use the most distinguishable, colorblind-robust hues first (blue/orange is the
  safest pair under deuteranopia/protanopia).
* All hues hold up against a white background (no pale yellows or near-white
  tints that wash out as thin lines).
* Eight entries cover ``MAX_SESSIONS`` with margin; indexing wraps via modulo.

Keep values as hex strings — accepted directly by both matplotlib and
``QColor`` — and keep the count at >= ``MAX_SESSIONS``.
"""

DEFAULT_SESSION_COLORS: list[str] = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
]
