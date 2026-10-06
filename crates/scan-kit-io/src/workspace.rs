//! The 2×3 dose workspace: four image cells and two plot cells.

use scan_kit_core::{dvh, field_bounds, Control, Panel, Series, Volume, VolumeMark};

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
    Longitudinal,
    Both,
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
    ("depth", "Depth dose"),
    ("lateral", "Lateral profile"),
    ("longitudinal", "Longitudinal profile"),
    ("lat_long", "Lateral + longitudinal"),
    ("dvh", "DVH"),
    ("gamma_hist", "Gamma histogram"),
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
        "depth" | "Depth dose" => PlotKind::Depth,
        "lateral" | "Lateral profile" => PlotKind::Lateral,
        "longitudinal" | "Longitudinal profile" => PlotKind::Longitudinal,
        "lat_long" | "Lateral + longitudinal" => PlotKind::Both,
        "dvh" | "DVH" => PlotKind::Dvh,
        "gamma_hist" | "Gamma histogram" => PlotKind::Gamma,
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
    pub field: Option<[f32; 4]>,
    pub show_field: bool,
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
        gain: space.gain,
        opacity: space.opacity,
        mode: space.mode,
        filter: space.filter,
    };
    (panels, mark)
}

pub(crate) fn layout() -> (u32, Vec<f32>, Vec<f32>, Vec<f32>) {
    (2, Vec::new(), vec![1.35, 1.35, 1.0], vec![1.0; 6])
}

fn image_panel(space: &Workspace, cell: CellView) -> Panel {
    let volume = &space.dose;
    let [nx, ny, nz] = volume.shape;
    let [ix, iy, iz] = space.cursor;
    let (title, values, cols, rows, xmin, xmax, ymin, ymax) = match cell {
        CellView::Coronal => (
            "Coronal",
            volume.coronal(iy.min(ny - 1)),
            nx,
            nz,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[2],
            volume.origin[2] + nz as f32 * volume.voxel,
        ),
        CellView::Sagittal => (
            "Sagittal",
            volume.sagittal(ix.min(nx - 1)),
            ny,
            nz,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
            volume.origin[2],
            volume.origin[2] + nz as f32 * volume.voxel,
        ),
        CellView::Volume => {
            let image = mip(volume);
            (
                "3D",
                image,
                nx,
                ny,
                volume.origin[0],
                volume.origin[0] + nx as f32 * volume.voxel,
                volume.origin[1],
                volume.origin[1] + ny as f32 * volume.voxel,
            )
        }
        CellView::Axial => (
            "Axial",
            volume.axial(iz.min(nz - 1)),
            nx,
            ny,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
        ),
    };
    let mut series = vec![Series::Heatmap {
        values,
        cols: cols as u32,
        rows: rows as u32,
        ramp: space.ramp,
        color: [1.0, 1.0, 1.0, space.opacity],
        lo: space.lo,
        hi: space.hi,
    }];
    if cell == CellView::Axial && space.show_field {
        if let Some(bounds) = space.field {
            series.push(Series::Guide {
                xs: vec![bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]],
                ys: vec![bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]],
                color: GUIDE,
                thickness: 1.0,
            });
        }
    }
    Panel {
        title: title.into(),
        y_label: String::new(),
        x_label: String::new(),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: true,
    }
}

fn plot_panel(space: &Workspace, kind: PlotKind) -> Panel {
    let volume = &space.dose;
    let [ix, iy, iz] = space.cursor;
    let (title, series, y_label) = match kind {
        PlotKind::Lateral => (
            "Lateral profile".into(),
            vec![line(volume.lateral_profile(iy, iz))],
            space.y_label.clone(),
        ),
        PlotKind::Longitudinal => (
            "Longitudinal profile".into(),
            vec![line(volume.longitudinal_profile(ix, iz))],
            space.y_label.clone(),
        ),
        PlotKind::Both => (
            "Lateral + longitudinal".into(),
            vec![
                line(volume.lateral_profile(iy, iz)),
                line_colored(volume.longitudinal_profile(ix, iz), [0.9, 0.55, 0.2, 1.0]),
            ],
            space.y_label.clone(),
        ),
        PlotKind::Dvh => ("DVH".into(), space.dvh.clone(), "Volume".into()),
        PlotKind::Gamma => {
            let (title, series) = gamma_panel(space);
            (title, series, "Voxels".into())
        }
        PlotKind::Depth => (
            "Depth dose".into(),
            vec![line(volume.depth_profile(ix, iy))],
            space.y_label.clone(),
        ),
    };
    let (xmin, xmax, ymin, ymax) = span(&series);
    Panel {
        title,
        y_label,
        x_label: String::new(),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
}

fn gamma_panel(space: &Workspace) -> (String, Vec<Series>) {
    let Some((gamma, rate)) = &space.gamma else {
        return ("Gamma histogram".into(), Vec::new());
    };
    let mut counts = [0.0f32; 20];
    for value in &gamma.values {
        if !value.is_finite() || *value <= 0.0 {
            continue;
        }
        let bin = ((*value / 2.0) * 20.0).floor() as usize;
        counts[bin.min(19)] += 1.0;
    }
    let edges: Vec<f32> = (0..=20).map(|i| i as f32 * 0.1).collect();
    let peak = counts.iter().copied().fold(1.0f32, f32::max);
    (
        format!("Gamma histogram  {rate:.0}%"),
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

pub(crate) fn field_box(volume: &Volume, iz: usize, fraction: f32) -> Option<[f32; 4]> {
    let image = volume.axial(iz.min(volume.shape[2].saturating_sub(1)));
    let peak = image.iter().copied().fold(0.0f32, f32::max);
    field_bounds(
        &image,
        volume.shape[0],
        volume.shape[1],
        volume.origin[0],
        volume.origin[1],
        volume.voxel,
        fraction * peak,
    )
}

pub(crate) fn field_depth(volume: &Volume, fraction: f32) -> f32 {
    let peak = volume.values.iter().copied().fold(0.0f32, f32::max);
    let mut z0 = volume.shape[2];
    let mut z1 = 0usize;
    let [nx, ny, nz] = volume.shape;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if volume.get(x, y, z) >= fraction * peak {
                    z0 = z0.min(z);
                    z1 = z1.max(z + 1);
                }
            }
        }
    }
    if z1 <= z0 {
        0.0
    } else {
        (z1 - z0) as f32 * volume.voxel
    }
}

pub(crate) fn cell_control(index: usize, cell: CellView) -> Control {
    Control::plain(
        format!("cell{index}"),
        "View",
        CELL_CHOICES.iter().map(|(_, label)| *label),
        cell_label(cell),
    )
    .grouped("Cell")
}

pub(crate) fn plot_control(index: usize, kind: PlotKind) -> Control {
    Control::plain(
        format!("plot{index}"),
        "Plot",
        PLOT_CHOICES.iter().map(|(_, label)| *label),
        plot_label(kind),
    )
    .grouped("Plot")
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
        PlotKind::Depth => "Depth dose",
        PlotKind::Lateral => "Lateral profile",
        PlotKind::Longitudinal => "Longitudinal profile",
        PlotKind::Both => "Lateral + longitudinal",
        PlotKind::Dvh => "DVH",
        PlotKind::Gamma => "Gamma histogram",
    }
}

fn mip(volume: &Volume) -> Vec<f32> {
    let [nx, ny, nz] = volume.shape;
    let mut image = vec![0.0; nx * ny];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let slot = &mut image[x + nx * y];
                *slot = f32::max(*slot, volume.get(x, y, z));
            }
        }
    }
    image
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
            PlotKind::Longitudinal,
            PlotKind::Both,
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
                field: None,
                show_field: false,
            };
            let (panels, mark) = assemble(&space);
            assert_eq!(panels.len(), 6, "{kind:?}");
            assert!(
                !panels[4].series.is_empty() || kind == PlotKind::Gamma,
                "{kind:?}"
            );
            assert_eq!(mark.values.len(), 8);
            assert!(panels.iter().any(|panel| panel.title.starts_with("3D")));
        }
    }
}
