# scan-kit

Open-source proton pencil beam scanning session analysis toolkit with a
Qt desktop launcher and Matplotlib analysis views.

## Quick Start

### Pre-built Executables (recommended)

Download the latest release for your platform from
[**Releases**](https://github.com/Pyramid-Technical-Consultants/scan-kit/releases/latest):

| Platform | Asset |
|----------|-------|
| Windows  | `scan-kit-windows.exe` |
| Linux (x86-64) | `scan-kit-linux-amd64` |

No Python installation required — just run the executable.

### From Source

Requires Python 3.10+.

```bash
pip install .          # standard install
pip install -e .       # editable install for development
```

Run the app:

```bash
scan-kit               # via console entry point
python -m scan_kit     # as a module
```

Check version:

```bash
scan-kit --version
```

## GUI workflow

### 1) Set data source

- In **DATA SOURCE**, enter the folder that contains session data
- Leave the field (or use Enter) to refresh discovery
- Default source is the current working directory when frozen, or `test_data` in a dev install

### 2) Sort and select sessions

- Sessions can be sorted by:
  - **Date** (default, newest first)
  - **ID**
  - **MU**
- Select up to **5 sessions** by ticking **Use** in the table (no Ctrl key needed)
- Use the **✕** control to clear all **Use** checks
- The table lists **Session ID**, **Date**, **MU**, **Time (s)**, and **Note** (edit in the cell; long text also appears in the tooltip)
- Status shows how many sessions are checked and whether metadata is still loading

### 3) Add session notes (optional)

- Edit the **Note** column directly (double-click or F2 on the cell)
- Notes are saved when you finish editing the cell
- Notes are stored in `<data_source>/session_notes.json`

### 4) Run analysis views

- Click any analysis button in **RUN ANALYSIS**
- Each view runs in its own Python process and opens Matplotlib window(s)
- Close the plot window(s) when done; the launcher stays open

### 5) Quit

- Close the window, or **Esc** / **Ctrl+Q**

## Available Views

| View | What it shows |
|------|---------------|
| **Position Error vs Energy** | IC1/IC2 X and Y position error vs energy (scatter) |
| **IC Beam Trajectory** | Per-spot raw IC X/Y lines through IC2→IC1, extended upstream/downstream |
| **Position Scatter** | Planned, IC1, and IC2 spot positions overlaid by session color |
| **Dose Ratios vs Energy** | IC2/IC1, IC3/IC1, IC3/IC2 ratio differences vs energy |
| **Dose Ratios vs Position** | Dose-ratio behavior against beam position |
| **Dose Ratios vs Spot Time** | Dose-ratio behavior against spot delivery time |
| **Dose Error vs Target (%)** | Percent error versus prescribed target by energy (IC1/IC2/IC3) |
| **Spot Delivery Time** | Total, beam-on, and overhead spot timing analysis |
| **Sigma vs Energy** | IC1/IC2 X and Y sigma vs energy (violin plots) |
| **Beam-Off Ramp-Down** | Beam-off current ramp-down curves (IC1/IC2/IC3) |
| **Beam-On vs Beam-Off Current** | Beam-on and beam-off current distributions by energy |
| **IC Timeslice Replay** | Interactive media-player style viewer for raw IC1/IC2/IC3 timeslice current |
| **Dose Accumulation** | Cumulative dose delivery analysis |
| **IC Current FFT Analysis** | Frequency-domain analysis of IC current signals |
| **IC Audio Export (WAV)** | Export IC current data as audible WAV files (requires PortAudio) |

### IC Timeslice Replay

The replay view uses a media-player layout for browsing raw IC current data:

- **Top panel** (three rows): full-resolution IC1, IC2, and IC3 traces for the
  currently selected window. Layer boundaries appear as vertical lines with
  energy annotations.
- **Bottom panel** (short timeline): a compressed min/max envelope of IC1
  across the entire session. Click and drag on this timeline to select the
  window shown in the top panel.

Usage:

1. Select one or more sessions and click **IC Timeslice Replay**.
2. The bottom timeline shows the full session waveform. Drag a span to zoom
   the top panel into that region.
3. Resize or reposition the span to scrub through different parts of the
   session.

When multiple sessions are selected, traces are overlaid with distinct colors.
The detail panel auto-decimates when the selected window is very large, so
interaction stays responsive.

## Supported Session Data Layout

Session discovery supports all of the following in the selected data source:

- Unpacked session directories:
  - `<base>/<session_id>/input_map.csv`
  - `<base>/<session_id>/<session_id>/input_map.csv`
- Archive files:
  - `<session_id>.zip`
  - `<session_id>.tgz`
  - `<session_id>.tar.gz`
  - `<session_id>.tar.bz2`
  - `<session_id>.tar.xz`
  - `<session_id>.tar`

For archive-based sessions, `scan-kit` expects files under a top-level
`<session_id>/` folder inside the archive.

Timeslice-based analyses (for example beam-on/beam-off and ramp-down views)
read per-layer files from:

`<session_id>/layer-<n>/run-<m>/timeslice_data_device_units.csv`

## Building from Source

To build a standalone executable for your platform:

```bash
pip install -e ".[build]"   # install with PyInstaller
python build.py --clean     # build single executable → dist/
```

The output lands in `dist/scan-kit` (Linux) or `dist\scan-kit.exe` (Windows).
The Windows build uses a windowed executable (no console) for the main GUI; use
`python -m scan_kit --version` in development, or see the window title for the version.

## Releasing

Releases are automated via GitHub Actions. To cut a new release:

1. Update the version in `scan_kit/__init__.py`
2. Commit and push to `main`
3. Tag and push:

```bash
git tag v<version>
git push origin v<version>
```

The CI workflow builds executables for Windows and Linux, then creates a GitHub
Release with the binaries attached automatically.

## Versioning

The project follows [Semantic Versioning](https://semver.org/). The single
source of truth for the version is `scan_kit/__init__.py` (`__version__`),
which is read by `pyproject.toml` and shown in the launcher window title (`scan-kit v…`).

## License

MIT — see [LICENSE](LICENSE).
