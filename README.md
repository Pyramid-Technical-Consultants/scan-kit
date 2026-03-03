# scan-kit

Free and open-source proton pencil beam scanning data analysis toolkit.

## Requirements

- Python 3.10 or later
- Session data as ZIP files (one per session)

## Installation

From the project root:

```bash
pip install .
```

Or install in editable mode for development:

```bash
pip install -e .
```

## Running scan-kit

Launch the TUI:

```bash
scan-kit
```

## User Guide

### 1. Set the data source

In the **DATA SOURCE** field, enter the path to the folder containing your session ZIP files. The app defaults to `test_data` in the project directory.

- Use an absolute path (e.g. `C:\Data\sessions`) or a path relative to the project root
- Press **Enter** to apply the path and refresh the session list

### 2. Select sessions

- The left panel lists all sessions found as `*.zip` files in the data directory
- Use **Space** or **Enter** to select sessions
- You can select **1 to 3 sessions** at a time
- The status bar shows how many sessions are selected and their IDs

### 3. Run an analysis

In the **RUN ANALYSIS** panel, choose one of the available views:

| View | Description |
|------|-------------|
| **IC1 X/Y Position Bars** | Position bar charts for IC1 |
| **IC1 vs IC2 Error Scatter** | Error scatter plot comparing IC1 and IC2 |
| **IC1/IC2 Spot Scatter (G3)** | Spot scatter for IC1 and IC2 (G3) |
| **IC1 Spot Scatter (G2)** | IC1 spot scatter (G2) |
| **Dose Ratios (IC2/IC1, IC3/IC1)** | Dose ratio analysis |
| **Sigma X/Y Box Plots** | Sigma X and Y box plots |

Click a button to run that analysis on the selected session(s). A notification appears when the analysis window closes.

### 4. Keyboard shortcuts

- **q** — Quit the application

## Data format

Place session ZIP files in your data directory. Each file should be named `{session_id}.zip`. The app discovers all `*.zip` files and uses the filename (without `.zip`) as the session ID.

## License

See [LICENSE](LICENSE) for details.
