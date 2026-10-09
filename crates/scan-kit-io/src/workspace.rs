//! The 2×3 dose workspace: four image cells and two plot cells.

use scan_kit_core::{dvh, Control, Panel, Series, Volume, VolumeMark};

const MARK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const GUIDE: [f32; 4] = [0.95, 0.72, 0.2, 0.9];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellView {
    Axial,
    Coronal,
    Sagittal,
    Volume,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlotKind {
    Depth,
    Lateral,
    Dvh,
    Gamma,
}

const CELL_CHOICES: &[(&str, &str)] = &[
    ("axial", "Axial"),
    ("coronal", "Coronal"),
    ("sagittal", "Sagittal"),
    ("volume", "3D"),
];

const PLOT_CHOICES: &[(&str, &str)] = &[
    ("depth", "Depth Dose"),
    ("lateral", "Lateral Profile"),
    ("dvh", "DVH"),
    ("gamma_hist", "Gamma Histogram"),
];

pub(crate) fn assign_cell(cells: &mut [CellView; 4], index: usize, next: CellView) {
    if next == CellView::Volume {
        if let Some(holder) = cells.iter().position(|cell| *cell == CellView::Volume) {
            if holder != index {
                cells[holder] = cells[index];
            }
        }
    }
    cells[index] = next;
}

pub(crate) fn cells_from(raw: &[Option<&str>]) -> [CellView; 4] {
    let mut cells = [
        CellView::Axial,
        CellView::Volume,
        CellView::Coronal,
        CellView::Sagittal,
    ];
    for (index, value) in raw.iter().enumerate().take(4) {
        let Some(value) = value else {
            continue;
        };
        let next = match *value {
            "coronal" | "Coronal" => CellView::Coronal,
            "sagittal" | "Sagittal" => CellView::Sagittal,
            "volume" | "3D" => CellView::Volume,
            "axial" | "Axial" => CellView::Axial,
            _ => continue,
        };
        assign_cell(&mut cells, index, next);
    }
    cells
}

pub(crate) fn plots_from(raw: [Option<&str>; 2], study: bool) -> [PlotKind; 2] {
    let mut plots = if study {
        [PlotKind::Dvh, PlotKind::Gamma]
    } else {
        [PlotKind::Depth, PlotKind::Lateral]
    };
    for (index, value) in raw.iter().enumerate() {
        let Some(value) = value else {
            continue;
        };
        if let Some(kind) = plot_kind(value) {
            plots[index] = kind;
        }
    }
    plots
}

fn plot_kind(value: &str) -> Option<PlotKind> {
    Some(match value {
        "depth" | "Depth dose" | "Depth Dose" => PlotKind::Depth,
        "lateral" | "Lateral profile" | "Lateral Profile" => PlotKind::Lateral,
        "dvh" | "DVH" => PlotKind::Dvh,
        "gamma_hist" | "Gamma histogram" | "Gamma Histogram" => PlotKind::Gamma,
        _ => return None,
    })
}

pub(crate) struct Workspace {
    pub dose: Volume,
    pub ct: Vec<f32>,
    pub labels: Vec<u8>,
    pub cursor: [usize; 3],
    pub cells: [CellView; 4],
    pub plots: [PlotKind; 2],
    pub ramp: u8,
    pub lo: f32,
    pub hi: f32,
    pub gain: f32,
    pub opacity: f32,
    pub mode: u8,
    pub filter: u8,
    pub y_label: String,
    pub dvh: Vec<Series>,
    pub gamma: Option<(Volume, f32)>,
}

pub(crate) fn assemble(space: &Workspace) -> (Vec<Panel>, VolumeMark) {
    let mut panels = Vec::with_capacity(6);
    for cell in space.cells {
        panels.push(image_panel(space, cell));
    }
    for kind in space.plots {
        panels.push(plot_panel(space, kind));
    }
    let n = space.dose.values.len();
    let [nx, ny, nz] = space.dose.shape;
    let mark = VolumeMark {
        values: space.dose.values.clone(),
        ct: if space.ct.len() == n {
            space.ct.clone()
        } else {
            Vec::new()
        },
        labels: if space.labels.len() == n {
            space.labels.clone()
        } else {
            Vec::new()
        },
        shape: [nx as u32, ny as u32, nz as u32],
        origin: space.dose.origin,
        voxel: space.dose.voxel,
        ramp: space.ramp,
        lo: space.lo,
        hi: space.hi,
        base_lo: 0.0,
        base_hi: 0.0,
        gain: space.gain,
        opacity: space.opacity,
        mode: space.mode,
        filter: space.filter,
        gantry: 0.0,
        unit: String::new(),
        show_phantom: false,
        field: [0.0; 6],
        sessions: Vec::new(),
        session_colors: Vec::new(),
        session_focus: 0,
        plans: Vec::new(),
        companions: Vec::new(),
    };
    (panels, mark)
}

pub(crate) fn layout() -> (u32, Vec<f32>, Vec<f32>, Vec<f32>) {
    (2, Vec::new(), vec![1.35, 1.35, 1.0], vec![1.0; 6])
}

fn image_panel(space: &Workspace, cell: CellView) -> Panel {
    let volume = &space.dose;
    let [nx, ny, nz] = volume.shape;
    // The dose plot retargets this quad at the volume atlas. A full slice here
    // is encoded and then dropped.
    let (title, values, cols, rows, xmin, xmax, ymin, ymax) = match cell {
        CellView::Coronal => (
            "Coronal",
            vec![0.0],
            1,
            1,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[2],
            volume.origin[2] + nz as f32 * volume.voxel,
        ),
        CellView::Sagittal => (
            "Sagittal",
            vec![0.0],
            1,
            1,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
            volume.origin[2],
            volume.origin[2] + nz as f32 * volume.voxel,
        ),
        CellView::Volume => (
            "3D",
            vec![0.0],
            1,
            1,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
        ),
        CellView::Axial => (
            "Axial",
            vec![0.0],
            1,
            1,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
        ),
    };
    let series = vec![Series::Heatmap {
        values,
        cols: cols as u32,
        rows: rows as u32,
        ramp: space.ramp,
        color: [1.0, 1.0, 1.0, space.opacity],
        lo: space.lo,
        hi: space.hi,
    }];
    let (x_label, y_label) = match cell {
        CellView::Axial => ("X (mm)", "Y (mm)"),
        CellView::Coronal => ("X (mm)", "Z (mm)"),
        CellView::Sagittal => ("Y (mm)", "Z (mm)"),
        CellView::Volume => ("", ""),
    };
    Panel {
        title: title.into(),
        y_label: y_label.into(),
        x_label: x_label.into(),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
}

fn plot_panel(space: &Workspace, kind: PlotKind) -> Panel {
    let volume = &space.dose;
    let [ix, iy, iz] = space.cursor;
    let (title, series, y_label) = match kind {
        PlotKind::Lateral => (
            "Lateral Profile".into(),
            vec![line(volume.lateral_profile(ix, iy, iz))],
            space.y_label.clone(),
        ),
        PlotKind::Dvh => ("DVH".into(), space.dvh.clone(), "Volume".into()),
        PlotKind::Gamma => {
            let (title, series) = gamma_panel(space);
            let y_label = match space.gamma {
                Some((_, rate)) => format!("Voxels · {rate:.0}% {}", gamma_verdict(rate)),
                None => "Voxels".into(),
            };
            (title, series, y_label)
        }
        PlotKind::Depth => (
            "Depth Dose".into(),
            vec![line(volume.depth_profile(ix, iy))],
            space.y_label.clone(),
        ),
    };
    let (xmin, xmax, ymin, ymax) = span(&series);
    let x_label = match kind {
        PlotKind::Depth => "Depth from the grid edge (mm)".into(),
        PlotKind::Lateral => "Across the beam from the crosshair (mm)".into(),
        PlotKind::Dvh | PlotKind::Gamma => String::new(),
    };
    Panel {
        title,
        y_label,
        x_label,
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
}

/// TG-218: at least 95% passes, 90% is the action level.
fn gamma_verdict(rate: f32) -> &'static str {
    if rate >= 95.0 {
        "pass"
    } else if rate >= 90.0 {
        "action"
    } else {
        "fail"
    }
}

fn gamma_heading(rate: f32) -> String {
    format!("Gamma Histogram  {rate:.0}% {}", gamma_verdict(rate))
}

fn gamma_panel(space: &Workspace) -> (String, Vec<Series>) {
    let Some((gamma, rate)) = &space.gamma else {
        return ("Gamma Histogram".into(), Vec::new());
    };
    let counts = gamma_counts(gamma);
    let edges: Vec<f32> = (0..=20).map(|i| i as f32 * 0.1).collect();
    let peak = counts.iter().copied().fold(1.0f32, f32::max);
    (
        gamma_heading(*rate),
        vec![
            Series::Bars {
                edges,
                counts: counts.to_vec(),
                color: MARK,
            },
            Series::Guide {
                xs: vec![1.0, 1.0],
                ys: vec![0.0, peak],
                color: GUIDE,
                thickness: 1.5,
            },
        ],
    )
}

pub(crate) fn dvh_line(volume: &Volume) -> Series {
    let mask = vec![true; volume.values.len()];
    let (edges, curve) = dvh(&volume.values, &mask, 32);
    let xs = edges.iter().take(curve.len()).copied().collect();
    line((xs, curve))
}

/// Axis-aligned box of the field, `[x0, x1, y0, y1, z0, z1]` in millimetres.
///
/// The reference is the volume peak. With `per_slice`, a depth counts only when
/// its own maximum reaches that fraction of the peak, and the lateral edge in
/// that slice is the same fraction of the slice maximum.
pub(crate) fn field_extent(volume: &Volume, fraction: f32, per_slice: bool) -> Option<[f32; 6]> {
    let [nx, ny, nz] = volume.shape;
    if nx == 0 || ny == 0 || nz == 0 {
        return None;
    }
    let global = volume.values.iter().copied().fold(0.0f32, f32::max);
    if !global.is_finite() || global <= 0.0 {
        return None;
    }
    let mut x0 = nx;
    let mut x1 = 0usize;
    let mut y0 = ny;
    let mut y1 = 0usize;
    let mut z0 = nz;
    let mut z1 = 0usize;
    let mut cover = |x: usize, y: usize, z: usize| {
        x0 = x0.min(x);
        x1 = x1.max(x + 1);
        y0 = y0.min(y);
        y1 = y1.max(y + 1);
        z0 = z0.min(z);
        z1 = z1.max(z + 1);
    };
    if per_slice {
        let gate = fraction * global;
        for z in 0..nz {
            let mut slice_peak = 0.0f32;
            for y in 0..ny {
                for x in 0..nx {
                    slice_peak = slice_peak.max(volume.get(x, y, z));
                }
            }
            if slice_peak < gate {
                continue;
            }
            let level = fraction * slice_peak;
            for y in 0..ny {
                for x in 0..nx {
                    if volume.get(x, y, z) >= level {
                        cover(x, y, z);
                    }
                }
            }
        }
    } else {
        let level = fraction * global;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if volume.get(x, y, z) >= level {
                        cover(x, y, z);
                    }
                }
            }
        }
    }
    if x1 <= x0 || y1 <= y0 || z1 <= z0 {
        return None;
    }
    let voxel = volume.voxel;
    let origin = volume.origin;
    Some([
        origin[0] + x0 as f32 * voxel,
        origin[0] + x1 as f32 * voxel,
        origin[1] + y0 as f32 * voxel,
        origin[1] + y1 as f32 * voxel,
        origin[2] + z0 as f32 * voxel,
        origin[2] + z1 as f32 * voxel,
    ])
}

pub(crate) fn cell_control(index: usize, cell: CellView) -> Control {
    let mut control = Control::plain(
        format!("cell{index}"),
        "View",
        CELL_CHOICES.iter().map(|(_, label)| *label),
        cell_label(cell),
    );
    for (choice, (id, _)) in control.options.iter_mut().zip(CELL_CHOICES) {
        choice.icon = (*id).to_string();
    }
    control.grouped("Cell")
}

pub(crate) fn plot_control(index: usize, kind: PlotKind) -> Control {
    let mut control = Control::plain(
        format!("plot{index}"),
        "Plot",
        PLOT_CHOICES.iter().map(|(_, label)| *label),
        plot_label(kind),
    );
    for (choice, (id, _)) in control.options.iter_mut().zip(PLOT_CHOICES) {
        choice.icon = (*id).to_string();
    }
    control.grouped("Plot")
}

fn cell_label(cell: CellView) -> &'static str {
    match cell {
        CellView::Axial => "Axial",
        CellView::Coronal => "Coronal",
        CellView::Sagittal => "Sagittal",
        CellView::Volume => "3D",
    }
}

fn plot_label(kind: PlotKind) -> &'static str {
    match kind {
        PlotKind::Depth => "Depth Dose",
        PlotKind::Lateral => "Lateral Profile",
        PlotKind::Dvh => "DVH",
        PlotKind::Gamma => "Gamma Histogram",
    }
}

fn line(pair: (Vec<f32>, Vec<f32>)) -> Series {
    line_colored(pair, MARK)
}

fn line_colored(pair: (Vec<f32>, Vec<f32>), color: [f32; 4]) -> Series {
    Series::Polyline {
        xs: pair.0,
        ys: pair.1,
        color,
        thickness: 1.5,
    }
}

fn span(series: &[Series]) -> (f32, f32, f32, f32) {
    let mut xmin = f32::MAX;
    let mut xmax = f32::MIN;
    let mut ymin = f32::MAX;
    let mut ymax = f32::MIN;
    for series in series {
        match series {
            Series::Polyline { xs, ys, .. } => {
                for (x, y) in xs.iter().zip(ys) {
                    if x.is_finite() && y.is_finite() {
                        xmin = xmin.min(*x);
                        xmax = xmax.max(*x);
                        ymin = ymin.min(*y);
                        ymax = ymax.max(*y);
                    }
                }
            }
            Series::Bars { edges, counts, .. } => {
                for edge in edges {
                    xmin = xmin.min(*edge);
                    xmax = xmax.max(*edge);
                }
                for count in counts {
                    ymin = ymin.min(0.0);
                    ymax = ymax.max(*count);
                }
            }
            _ => {}
        }
    }
    if !xmin.is_finite() {
        return (0.0, 1.0, 0.0, 1.0);
    }
    if (xmax - xmin).abs() < 1e-4 {
        xmin -= 0.5;
        xmax += 0.5;
    }
    if (ymax - ymin).abs() < 1e-4 {
        ymin -= 0.5;
        ymax += 0.5;
    }
    (xmin, xmax, ymin, ymax)
}

fn fit_panel(panel: &mut Panel) {
    let (xmin, xmax, ymin, ymax) = span(&panel.series);
    panel.xmin = xmin;
    panel.xmax = xmax;
    panel.ymin = ymin;
    panel.ymax = ymax;
}

fn profile_series<'a>(
    kind: PlotKind,
    volumes: impl IntoIterator<Item = &'a Volume>,
    at: [f32; 3],
) -> Vec<Series> {
    let mut series = Vec::new();
    for volume in volumes {
        let ix = volume.index_of(0, at[0]);
        let iy = volume.index_of(1, at[1]);
        let iz = volume.index_of(2, at[2]);
        match kind {
            PlotKind::Depth => series.push(line(volume.depth_profile(ix, iy))),
            PlotKind::Lateral => series.push(line(volume.lateral_profile(ix, iy, iz))),
            PlotKind::Dvh | PlotKind::Gamma => {}
        }
    }
    series
}

fn gamma_counts(volume: &Volume) -> [f32; 20] {
    let mut counts = [0.0f32; 20];
    for value in &volume.values {
        if !value.is_finite() || *value <= 0.0 {
            continue;
        }
        let bin = ((*value / 2.0) * 20.0).floor() as usize;
        counts[bin.min(19)] += 1.0;
    }
    counts
}

/// One curve per session, plus the γ = 1 guide. Bars would hide each other.
fn gamma_lines(volumes: &[Volume]) -> Vec<Series> {
    let mut series = Vec::new();
    let mut peak = 1.0f32;
    for volume in volumes {
        let counts = gamma_counts(volume);
        peak = peak.max(counts.iter().copied().fold(0.0, f32::max));
        let xs: Vec<f32> = (0..20).map(|bin| (bin as f32 + 0.5) * 0.1).collect();
        series.push(line((xs, counts.to_vec())));
    }
    series.push(Series::Guide {
        xs: vec![1.0, 1.0],
        ys: vec![0.0, peak],
        color: GUIDE,
        thickness: 1.5,
    });
    series
}

/// Grow a profile's fitted range so a plan or second chamber is not clipped.
pub(crate) fn widen_profiles(panel: &mut Panel, volumes: &[&Volume], at: [f32; 3]) {
    if volumes.is_empty() {
        return;
    }
    let title = panel.title.to_ascii_lowercase();
    let kind = if title.starts_with("depth") {
        PlotKind::Depth
    } else if title.starts_with("lateral") {
        PlotKind::Lateral
    } else {
        return;
    };
    let mut series = panel.series.clone();
    series.extend(profile_series(kind, volumes.iter().copied(), at));
    let (xmin, xmax, ymin, ymax) = span(&series);
    panel.xmin = xmin;
    panel.xmax = xmax;
    panel.ymin = ymin;
    panel.ymax = ymax;
}

/// Draw every loaded session on the line plots. Image cells stay on one cube.
pub(crate) fn overlay_sessions(
    panel: &mut Panel,
    volumes: &[Volume],
    at: [f32; 3],
    gamma: &[Volume],
) {
    if volumes.len() < 2 {
        return;
    }
    let title = panel.title.to_ascii_lowercase();
    if title.starts_with("depth") {
        panel.series = profile_series(PlotKind::Depth, volumes.iter(), at);
        fit_panel(panel);
    } else if title.starts_with("lateral") {
        panel.series = profile_series(PlotKind::Lateral, volumes.iter(), at);
        fit_panel(panel);
    } else if title.starts_with("dvh") {
        panel.series = volumes.iter().map(dvh_line).collect();
        fit_panel(panel);
    } else if title.starts_with("gamma") && gamma.len() > 1 {
        panel.series = gamma_lines(gamma);
        panel.xmin = 0.0;
        panel.xmax = 2.0;
        panel.ymin = 0.0;
        let peak = panel
            .series
            .iter()
            .filter_map(|series| match series {
                Series::Guide { ys, .. } => ys.iter().copied().reduce(f32::max),
                _ => None,
            })
            .fold(1.0, f32::max);
        panel.ymax = peak;
    }
}

pub(crate) fn readout(label: &str, id: &str, text: &str, group: &str) -> Control {
    Control::plain(id, label, [text], text).grouped(group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choosing_3d_swaps_with_the_cell_that_had_it() {
        let mut cells = [
            CellView::Axial,
            CellView::Volume,
            CellView::Coronal,
            CellView::Sagittal,
        ];
        assign_cell(&mut cells, 0, CellView::Volume);
        assert_eq!(
            cells,
            [
                CellView::Volume,
                CellView::Axial,
                CellView::Coronal,
                CellView::Sagittal
            ]
        );
    }

    #[test]
    fn gamma_heading_uses_the_tolerance_and_action_levels() {
        assert!(gamma_heading(95.0).ends_with("pass"));
        assert!(gamma_heading(90.0).ends_with("action"));
        assert!(gamma_heading(89.9).ends_with("fail"));
    }

    #[test]
    fn two_sessions_share_the_depth_axis() {
        let low = Volume {
            origin: [0.0; 3],
            shape: [1, 1, 2],
            voxel: 1.0,
            values: vec![1.0, 3.0],
        };
        let high = Volume {
            origin: [0.0; 3],
            shape: [1, 1, 2],
            voxel: 1.0,
            values: vec![2.0, 8.0],
        };
        let mut panel = Panel {
            title: "Depth Dose".into(),
            y_label: "Gy".into(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 1.0,
            series: Vec::new(),
            x_labels: Vec::new(),
            equal: false,
        };
        overlay_sessions(&mut panel, &[low, high], [0.5, 0.5, 0.5], &[]);
        assert_eq!(panel.series.len(), 2);
        assert!(panel.ymax >= 8.0, "{}", panel.ymax);
    }

    #[test]
    fn a_plan_profile_widens_the_depth_axis() {
        let measured = Volume {
            origin: [0.0; 3],
            shape: [1, 1, 2],
            voxel: 1.0,
            values: vec![1.0, 3.0],
        };
        let plan = Volume {
            origin: [0.0; 3],
            shape: [1, 1, 2],
            voxel: 1.0,
            values: vec![1.0, 9.0],
        };
        let mut panel = Panel {
            title: "Depth Dose".into(),
            y_label: "Gy".into(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 3.0,
            series: vec![line(measured.depth_profile(0, 0))],
            x_labels: Vec::new(),
            equal: false,
        };
        widen_profiles(&mut panel, &[&plan], [0.5, 0.5, 0.5]);
        assert!(panel.ymax >= 9.0, "{}", panel.ymax);
        assert_eq!(panel.series.len(), 1);
    }

    #[test]
    fn a_loaded_study_starts_on_dvh_and_gamma() {
        assert_eq!(
            plots_from([None, None], true),
            [PlotKind::Dvh, PlotKind::Gamma]
        );
        assert_eq!(
            plots_from([None, None], false),
            [PlotKind::Depth, PlotKind::Lateral]
        );
    }

    #[test]
    fn each_plot_kind_builds_a_panel() {
        let volume = Volume {
            origin: [0.0, 0.0, -4.0],
            shape: [2, 2, 2],
            voxel: 1.0,
            values: vec![0.0, 1.0, 0.2, 0.4, 0.1, 0.3, 0.5, 2.0],
        };
        let gamma = Volume {
            values: vec![0.2, 0.4, 1.2, 0.8, 0.1, 1.5, 0.3, 0.6],
            ..volume.clone()
        };
        for kind in [
            PlotKind::Depth,
            PlotKind::Lateral,
            PlotKind::Dvh,
            PlotKind::Gamma,
        ] {
            let space = Workspace {
                dose: volume.clone(),
                ct: Vec::new(),
                labels: Vec::new(),
                cursor: {
                    let (x, y, z) = volume.peak_index();
                    [x, y, z]
                },
                cells: [
                    CellView::Axial,
                    CellView::Volume,
                    CellView::Coronal,
                    CellView::Sagittal,
                ],
                plots: [kind, PlotKind::Depth],
                ramp: 0,
                lo: 0.0,
                hi: 2.0,
                gain: 1.0,
                opacity: 1.0,
                mode: 0,
                filter: 0,
                y_label: "Gy".into(),
                dvh: vec![dvh_line(&volume)],
                gamma: Some((gamma.clone(), 95.0)),
            };
            let (panels, mark) = assemble(&space);
            assert_eq!(panels.len(), 6, "{kind:?}");
            assert!(
                !panels[4].series.is_empty() || kind == PlotKind::Gamma,
                "{kind:?}"
            );
            assert_eq!(mark.values.len(), 8);
            assert!(panels.iter().any(|panel| panel.title.starts_with("3D")));
            assert_eq!(panels[0].x_label, "X (mm)");
            assert_eq!(panels[0].y_label, "Y (mm)");
            assert!(panels[1].x_label.is_empty());
            assert!(!panels[0].equal);
            assert!(!panels[1].equal);
            let view = cell_control(0, CellView::Axial);
            assert_eq!(
                view.options
                    .iter()
                    .map(|choice| choice.icon.as_str())
                    .collect::<Vec<_>>(),
                ["axial", "coronal", "sagittal", "volume"]
            );
            let plot = plot_control(0, PlotKind::Depth);
            assert_eq!(
                plot.labels(),
                ["Depth Dose", "Lateral Profile", "DVH", "Gamma Histogram"]
            );
            assert_eq!(
                plot.options
                    .iter()
                    .map(|choice| choice.icon.as_str())
                    .collect::<Vec<_>>(),
                ["depth", "lateral", "dvh", "gamma_hist"]
            );
        }
    }

    #[test]
    fn a_slice_edge_includes_the_wider_depth_and_a_peak_edge_does_not() {
        // z = 0 is wider but below half the peak. z = 1 is the peak, one voxel wide.
        let mut values = vec![0.0; 5];
        values[1] = 1.5;
        values[2] = 2.0;
        values[3] = 1.5;
        values.push(0.0);
        values.push(0.0);
        values.push(4.0);
        values.push(0.0);
        values.push(0.0);
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [5, 1, 2],
            voxel: 1.0,
            values,
        };
        let slice = field_extent(&volume, 0.5, true).unwrap();
        let peak = field_extent(&volume, 0.5, false).unwrap();
        assert_eq!(slice, [1.0, 4.0, 0.0, 1.0, 0.0, 2.0]);
        assert_eq!(peak, [2.0, 3.0, 0.0, 1.0, 0.0, 2.0]);
    }
}
