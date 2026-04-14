# scan-kit

Open-source proton pencil beam scanning session analysis toolkit with a Textual
terminal UI and Matplotlib analysis views.

## Requirements

- Python 3.10+
- Session exports available as unpacked folders or archives

## Installation

From the repository root:

```bash
pip install .
```

Editable install for development:

```bash
pip install -e .
```

## Run the App

Start the Textual launcher:

```bash
scan-kit
```

You can also run it as a module:

```bash
python -m scan_kit.app
```

## TUI Workflow

### 1) Set data source

- In `DATA SOURCE`, enter the folder that contains session data
- Press `Enter` to refresh discovery
- Default source is `test_data` in the project root

### 2) Sort and select sessions

- Sessions can be sorted by:
  - `Date` (default, newest first)
  - `ID`
  - `MU`
- Select up to **3 sessions** at a time from the list
- Use the `X` button to clear selection
- Status line shows selected IDs and loading state while metadata is being read

### 3) Add session notes (optional)

- Highlight a session in the list to edit its note
- Notes are auto-saved while typing
- Notes are stored in `<data_source>/session_notes.json`
- A short note preview is shown inline in the session list

### 4) Run analysis views

- Click any analysis button in `RUN ANALYSIS`
- Each view runs in its own Python process and opens Matplotlib window(s)
- Close the plot window(s) to return to the launcher

### 5) Quit

- `Esc`, `Ctrl+Q`, or `Ctrl+C`

## Available Views

| View | What it shows |
|------|----------------|
| **IC1 X/Y Position Error** | IC1 X and Y position error by energy (box plots) |
| **IC1 vs IC2 Error Scatter** | IC1/IC2 position differences in X and Y (scatter) |
| **IC1 Spot Scatter (G3)** | IC1 and IC2 spot positions for G3 sessions |
| **IC1 Spot Scatter (G2)** | IC1 spot positions for G2 sessions |
| **Dose Ratios vs Energy** | IC2/IC1, IC3/IC1, IC3/IC2 ratio differences vs energy |
| **Dose Ratios vs Position** | Dose-ratio behavior against beam position |
| **Dose Ratios vs Spot Time** | Dose-ratio behavior against spot delivery time |
| **Dose Error vs Target (%)** | Percent error versus prescribed target by energy (IC1/IC2/IC3) |
| **Spot Delivery Time** | Total, beam-on, and overhead spot timing analysis |
| **Sigma X/Y Box Plots** | Sigma X and Y distributions by energy |
| **Beam-Off Ramp-Down** | Beam-off current ramp-down curves (IC1/IC2/IC3) |
| **Beam-On vs Beam-Off Current** | Beam-on and beam-off current distributions by energy |

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

## License

See [LICENSE](LICENSE).
