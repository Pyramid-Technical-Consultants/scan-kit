//! Spot and timeslice columns shared by the analysis scenes and the tuners.
//!
//! ponytail: sigma expected values on the spot frame come from a string scan of
//! devices.xml. Timeslice isocenter position uses the spot-file strip-to-mm fit,
//! and a duplicate `spot_no` keeps pandas' `.1` suffix.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use super::discover;
use scan_kit_core::{
    column_scale_factor, dose_error_pct, dose_ratio_pct, g2_ic2_mm, linear_fit, remap,
    remap_g2_raw, remap_g3_raw, resolve_concept_column, scale_column,
};

pub(crate) struct Sheet {
    pub(crate) num: BTreeMap<String, Vec<f32>>,
    pub(crate) wide: BTreeMap<String, Vec<f64>>,
}

pub(crate) fn spot_table(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    load_spot(root, session, false, false, false)
}

/// Spot table for configuration tuning.
///
/// Sigma prefers `spot_sigma_raw`, then scales ×2 to mm, matching
/// `session_sigma.py`. Analysis plots prefer the processed `spot_sigma` column.
pub(crate) fn tune_spot_table(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    load_spot(root, session, false, false, true)
}

pub(crate) fn slice_table(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    load_slice_metric(root, session, "position_error", false)
}

pub(crate) fn timeslice_metric(
    root: &Path,
    session: &str,
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    load_timeslice(root, session, metric)
}

/// Current, confidence, peak, field, and amplifier columns from one timeslice read.
pub(crate) fn timeslice_signals(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    cached_timeslice(root, session, "signals", false, |frames| {
        let mut out = current_from(&frames.sheets, &frames.energies, &frames.layers);
        for metric in [
            "fit_confidence",
            "peak_amplitude",
            "amplifier_error",
            "probe_field",
        ] {
            for (key, values) in
                signal_from(&frames.sheets, &frames.energies, &frames.layers, metric)
            {
                out.entry(key).or_insert(values);
            }
        }
        out
    })
}

// ponytail: keyed by spot/map length, mtime, and the first 4KB. devices.xml and
// per-layer point-time files stay stale until the spot csv changes. Past 16
// entries the map is cleared.
const SPOT_CACHE_CAP: usize = 16;

#[derive(Hash, PartialEq, Eq)]
struct SpotCacheKey {
    dir: PathBuf,
    chamber: bool,
    spot_len: u64,
    spot_ns: u128,
    spot_head: u64,
    map_len: u64,
    map_ns: u128,
    map_head: u64,
    points: bool,
    raw_sigma: bool,
}

fn spot_cache() -> &'static std::sync::Mutex<HashMap<SpotCacheKey, BTreeMap<String, Vec<f32>>>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<HashMap<SpotCacheKey, BTreeMap<String, Vec<f32>>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

pub(crate) fn file_fingerprint(path: &Path) -> Option<(u64, u128, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let ns = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 4096];
    let n = std::io::Read::read(&mut file, &mut buf).ok()?;
    Some((meta.len(), ns, fnv64(&buf[..n])))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Parsed timeslice frames shared by every metric.
///
/// ponytail: four sessions stay parsed. Past that the map is cleared. The files
/// are the expensive part; the derived tables below are the cheap repeat.
const FRAME_CACHE_CAP: usize = 4;
const TABLE_CACHE_CAP: usize = 32;

struct Frames {
    energies: Vec<f32>,
    layers: Vec<i64>,
    sheets: Vec<Sheet>,
    map: Sheet,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct FrameKey {
    dir: PathBuf,
    map_stamp: u128,
    slice_stamp: u128,
}

#[derive(Hash, PartialEq, Eq)]
struct TableKey {
    frames: FrameKey,
    spot_stamp: u128,
    kind: String,
}

fn frame_cache() -> &'static Mutex<HashMap<FrameKey, Arc<Frames>>> {
    static CACHE: OnceLock<Mutex<HashMap<FrameKey, Arc<Frames>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn table_cache() -> &'static Mutex<HashMap<TableKey, BTreeMap<String, Vec<f32>>>> {
    static CACHE: OnceLock<Mutex<HashMap<TableKey, BTreeMap<String, Vec<f32>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn frame_key(root: &Path, session: &str) -> FrameKey {
    let dir = discover::session_directory(root, session);
    FrameKey {
        map_stamp: discover::meta_stamp(&dir.join("input_map.csv")),
        slice_stamp: discover::timeslice_stamp(&dir),
        dir,
    }
}

fn cached_timeslice(
    root: &Path,
    session: &str,
    kind: &str,
    spot: bool,
    build: impl FnOnce(&Frames) -> BTreeMap<String, Vec<f32>>,
) -> BTreeMap<String, Vec<f32>> {
    let frames = frame_key(root, session);
    let key = TableKey {
        spot_stamp: if spot {
            discover::meta_stamp(&frames.dir.join("spot_data.csv"))
        } else {
            0
        },
        kind: kind.to_string(),
        frames,
    };
    if let Some(hit) = table_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
    {
        return hit.clone();
    }
    let table = open_frames(root, session)
        .map(|frames| build(&frames))
        .unwrap_or_default();
    let mut cache = table_cache().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= TABLE_CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, table.clone());
    table
}

fn open_frames(root: &Path, session: &str) -> Option<Arc<Frames>> {
    let key = frame_key(root, session);
    if let Some(hit) = frame_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
    {
        return Some(Arc::clone(hit));
    }
    let map = discover::read_session_file(root, session, "input_map.csv")
        .map(|bytes| read_sheet(&bytes))
        .unwrap_or_else(|| Sheet {
            num: BTreeMap::new(),
            wide: BTreeMap::new(),
        });
    let energies = unique_seen(&column(&map, "energy", &[]));
    let listed = discover::read_timeslice_frames(&key.dir);
    if listed.is_empty() {
        return None;
    }
    let layers = listed.iter().map(|(index, _)| *index).collect();
    let files: Vec<Vec<u8>> = listed.into_iter().map(|(_, bytes)| bytes).collect();
    let sheets = read_sheets(&files, timeslice_keep);
    let frames = Arc::new(Frames {
        energies,
        layers,
        sheets,
        map,
    });
    let mut cache = frame_cache().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= FRAME_CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, Arc::clone(&frames));
    Some(frames)
}

/// ponytail: one sample per stride past 80k rows. The plot keeps the shape;
/// raise the cap if a tail or a chamber view needs every sample.
fn sample_stride(rows: usize) -> usize {
    (rows / 80_000).max(1)
}

pub(crate) fn merged_timeslice(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    cached_timeslice(root, session, "merged", false, |frames| {
        let mut merged: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for sheet in &frames.sheets {
            for (name, values) in &sheet.num {
                merged
                    .entry(name.clone())
                    .or_default()
                    .extend(values.iter().copied());
            }
        }
        merged
    })
}

pub(crate) fn load_spot(
    root: &Path,
    session: &str,
    chamber: bool,
    points: bool,
    raw_sigma: bool,
) -> BTreeMap<String, Vec<f32>> {
    let dir = discover::session_directory(root, session);
    let key = match (
        file_fingerprint(&dir.join("spot_data.csv")),
        file_fingerprint(&dir.join("input_map.csv")),
    ) {
        (Some((spot_len, spot_ns, spot_head)), Some((map_len, map_ns, map_head))) => {
            Some(SpotCacheKey {
                dir,
                chamber,
                spot_len,
                spot_ns,
                spot_head,
                map_len,
                map_ns,
                map_head,
                points,
                raw_sigma,
            })
        }
        _ => None,
    };
    if let Some(key) = &key {
        let cache = spot_cache().lock().unwrap_or_else(|err| err.into_inner());
        if let Some(hit) = cache.get(key) {
            return hit.clone();
        }
    }
    let table = build_spot(root, session, chamber, points, raw_sigma);
    if let Some(key) = key {
        let mut cache = spot_cache().lock().unwrap_or_else(|err| err.into_inner());
        if cache.len() >= SPOT_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, table.clone());
    }
    table
}

fn build_spot(
    root: &Path,
    session: &str,
    chamber: bool,
    points: bool,
    raw_sigma: bool,
) -> BTreeMap<String, Vec<f32>> {
    let Some(spot_bytes) = discover::read_session_file(root, session, "spot_data.csv") else {
        return BTreeMap::new();
    };
    let Some(map_bytes) = discover::read_session_file(root, session, "input_map.csv") else {
        return BTreeMap::new();
    };
    let spot = read_sheet(&spot_bytes);
    let map = read_sheet(&map_bytes);
    let energy = column(&map, "energy", &[]);
    let dose = column(
        &spot,
        "ic1_total_dose",
        &["ic1_total_dose_spot", "ic1_dose"],
    );
    let n = energy.len().min(if dose.is_empty() {
        energy.len()
    } else {
        dose.len()
    });
    if n == 0 {
        return BTreeMap::new();
    }
    let target = column(&map, "charge_req", &["target_mu", "MU", "mu"]);
    let plan_x = column(&map, "x_position", &["position_x", "X_POSITION", "x_pos"]);
    let plan_y = column(&map, "y_position", &["position_y", "Y_POSITION", "y_pos"]);
    let mut doses = BTreeMap::new();
    for (concept, key) in [
        ("ic1_total_dose", "ic1_dose"),
        ("ic2_total_dose", "ic2_dose"),
        ("ic3_total_dose", "ic3_dose"),
    ] {
        let values = column(&spot, concept, &[key]);
        if values.len() >= n {
            doses.insert(key, values);
        }
    }
    let (mask_pos, mm_pos) = spot_positions(&spot, chamber, n);
    let sigma = spot_sigma(&spot, raw_sigma || chamber, n);
    let mut mask_cols: Vec<&[f32]> = vec![&energy];
    if target.len() >= n {
        mask_cols.push(&target);
    }
    if plan_x.len() >= n {
        mask_cols.push(&plan_x);
    }
    if plan_y.len() >= n {
        mask_cols.push(&plan_y);
    }
    for values in doses.values() {
        mask_cols.push(values);
    }
    for values in &mask_pos {
        mask_cols.push(values);
    }
    let keep = keep_mask(&mask_cols, n);
    if !keep.iter().any(|row| *row) {
        return BTreeMap::new();
    }
    let mut table = BTreeMap::new();
    table.insert("energy".to_string(), take_kept(&energy, &keep));
    if target.len() >= n {
        table.insert("target_mu".to_string(), take_kept(&target, &keep));
    }
    if plan_x.len() >= n && plan_y.len() >= n {
        let x = take_kept(&plan_x, &keep);
        let y = take_kept(&plan_y, &keep);
        table.insert(
            "radius".to_string(),
            x.iter()
                .zip(&y)
                .map(|(x, y)| (x * x + y * y).sqrt())
                .collect(),
        );
        table.insert("plan_x".to_string(), x);
        table.insert("plan_y".to_string(), y);
    }
    for (key, values) in &doses {
        table.insert((*key).to_string(), take_kept(values, &keep));
    }
    if let (Some(target), Some(ic1)) = (
        table.get("target_mu").cloned(),
        table.get("ic1_dose").cloned(),
    ) {
        table.insert(
            "ic1_dose_err_pct".to_string(),
            dose_error_pct(&ic1, &target),
        );
    }
    if let (Some(target), Some(ic2)) = (
        table.get("target_mu").cloned(),
        table.get("ic2_dose").cloned(),
    ) {
        table.insert(
            "ic2_dose_err_pct".to_string(),
            dose_error_pct(&ic2, &target),
        );
    }
    if let (Some(target), Some(ic3)) = (
        table.get("target_mu").cloned(),
        table.get("ic3_dose").cloned(),
    ) {
        table.insert(
            "ic3_dose_err_pct".to_string(),
            dose_error_pct(&ic3, &target),
        );
    }
    if let (Some(ic1), Some(ic2)) = (
        table.get("ic1_dose").cloned(),
        table.get("ic2_dose").cloned(),
    ) {
        table.insert("ic21_ratio".to_string(), dose_ratio_pct(&ic2, &ic1));
        if let Some(ic3) = table.get("ic3_dose").cloned() {
            table.insert("ic31_ratio".to_string(), dose_ratio_pct(&ic3, &ic1));
            table.insert("ic32_ratio".to_string(), dose_ratio_pct(&ic3, &ic2));
        }
    }
    let mm = mm_pos.map(|values| values.map(|column| take_kept(&column, &keep)));
    store_positions(&mut table, mm);
    for (key, values) in sigma {
        table.insert(key.to_string(), take_kept(&values, &keep));
    }
    let dir = discover::session_directory(root, session);
    if let Ok(xml) = std::fs::read_to_string(dir.join("config/map2map/devices.xml")) {
        add_sigma_error(&mut table, &xml);
    }
    let layer = spot.num.get("layer_id").or_else(|| map.num.get("layer_id"));
    let spot_no = spot.num.get("spot_no").filter(|values| values.len() >= n);
    let time_ms = spot_time_ms(&spot, n);
    if let (Some(time), Some(layer)) = (time_ms.as_ref(), layer.filter(|v| v.len() >= n)) {
        let ms = spot_delivery_ms(&take_kept(time, &keep), &take_kept(layer, &keep));
        table.insert("spot_time".to_string(), ms);
        let point = ["point_time", "point_time_ms", "point_time(ms)"]
            .iter()
            .find_map(|name| spot.num.get(*name))
            .filter(|values| values.len() >= n)
            .map(|values| take_kept(values, &keep))
            .or_else(|| {
                if !points {
                    return None;
                }
                let spots = spot_no?;
                let layers = layer;
                Some(lookup_point_time(
                    &dir,
                    &take_kept(spots, &keep),
                    &take_kept(layers, &keep),
                ))
            });
        if let Some(beam) = point.filter(|values| values.iter().any(|value| value.is_finite())) {
            let overhead = beam
                .iter()
                .zip(&table["spot_time"])
                .map(|(beam, total)| {
                    if beam.is_finite() && total.is_finite() {
                        (total - beam).max(0.0)
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            table.insert("beam_on_time".to_string(), beam);
            table.insert("overhead_time".to_string(), overhead);
        }
    }
    if table.contains_key("energy") && table.contains_key("target_mu") {
        if let Some(time) = wall_time(&spot, n) {
            add_dose_rate(&mut table, &take_kept(&time, &keep));
        }
    }
    if let Some(beam) = column_any(&spot, &["beam_on", "rci_in_trigger", "r_beamOk"]) {
        if beam.len() >= n {
            table.insert("beam_on".to_string(), take_kept(beam, &keep));
        }
    }
    table
}

fn spot_time_ms(spot: &Sheet, n: usize) -> Option<Vec<f64>> {
    if let Some(time) = spot.wide.get("timestamp") {
        if time.len() >= n && time.iter().take(n).any(|value| value.is_finite()) {
            return Some(time[..n].to_vec());
        }
    }
    let seconds = spot.wide.get("time_s")?;
    let nanos = spot.wide.get("time_ns")?;
    if seconds.len() < n || nanos.len() < n {
        return None;
    }
    let time: Vec<f64> = seconds
        .iter()
        .zip(nanos)
        .take(n)
        .map(|(seconds, nanos)| seconds * 1000.0 + nanos / 1e6)
        .collect();
    time.iter().any(|value| value.is_finite()).then_some(time)
}

fn row_ok(value: f32) -> bool {
    value.is_finite() && value != -1.0 && value != -10000.0
}

fn keep_mask(columns: &[&[f32]], n: usize) -> Vec<bool> {
    let mut keep = vec![true; n];
    for column in columns {
        if column.len() < n {
            continue;
        }
        for (slot, value) in keep.iter_mut().zip(*column) {
            if !row_ok(*value) {
                *slot = false;
            }
        }
    }
    keep
}

fn take_kept<T: Copy>(values: &[T], keep: &[bool]) -> Vec<T> {
    values
        .iter()
        .zip(keep)
        .filter(|(_, keep)| **keep)
        .map(|(value, _)| *value)
        .collect()
}

fn store_positions(table: &mut BTreeMap<String, Vec<f32>>, mm: [Option<Vec<f32>>; 4]) {
    let plans = [
        table.get("plan_x").cloned(),
        table.get("plan_y").cloned(),
        table.get("plan_x").cloned(),
        table.get("plan_y").cloned(),
    ];
    let keys = [
        ("ic1_x", "ic1_x_err"),
        ("ic1_y", "ic1_y_err"),
        ("ic2_x", "ic2_x_err"),
        ("ic2_y", "ic2_y_err"),
    ];
    for ((pos_key, err_key), (measured, plan)) in keys.into_iter().zip(mm.into_iter().zip(plans)) {
        let Some(measured) = measured else {
            continue;
        };
        if let Some(plan) = plan {
            if plan.len() == measured.len() {
                table.insert(
                    err_key.to_string(),
                    measured
                        .iter()
                        .zip(&plan)
                        .map(|(measured, plan)| measured - plan)
                        .collect(),
                );
            }
        }
        table.insert(pos_key.to_string(), measured);
    }
    if let (Some(a), Some(b)) = (table.get("ic1_x").cloned(), table.get("ic2_x").cloned()) {
        table.insert(
            "ic12_x_diff".to_string(),
            b.iter().zip(&a).map(|(b, a)| b - a).collect(),
        );
    }
    if let (Some(a), Some(b)) = (table.get("ic1_y").cloned(), table.get("ic2_y").cloned()) {
        table.insert(
            "ic12_y_diff".to_string(),
            b.iter().zip(&a).map(|(b, a)| b - a).collect(),
        );
    }
}

/// Columns that participate in the row mask, plus mm positions when the frame resolves.
fn spot_positions(spot: &Sheet, chamber: bool, n: usize) -> (Vec<Vec<f32>>, [Option<Vec<f32>>; 4]) {
    if chamber {
        if let Some(raw) = four_columns(spot, "spot_position_raw", n) {
            let mm = [
                Some(g3_mm(&raw[0], false)),
                Some(g3_mm(&raw[1], true)),
                Some(g3_mm(&raw[2], true)),
                Some(g3_mm(&raw[3], false)),
            ];
            return (raw.to_vec(), mm);
        }
        if let Some(raw) = four_columns(spot, "spot_raw", n) {
            let ic1_x = raw[0].iter().copied().map(remap_g2_raw).collect::<Vec<_>>();
            let ic1_y = raw[1].iter().copied().map(remap_g2_raw).collect::<Vec<_>>();
            let ic2_x = g2_ic2_mm(&ic1_x, &raw[2]);
            let ic2_y = g2_ic2_mm(&ic1_y, &raw[3]);
            return (
                raw.to_vec(),
                [Some(ic1_x), Some(ic1_y), Some(ic2_x), Some(ic2_y)],
            );
        }
    }
    let mut mask = Vec::new();
    let mut mm = [None, None, None, None];
    for (index, (ic, axis)) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")]
        .into_iter()
        .enumerate()
    {
        let Some(values) = column_suffix(spot, ic, axis, &["spot_position", "spot"]) else {
            continue;
        };
        if values.len() < n {
            continue;
        }
        let values = values[..n].to_vec();
        mask.push(values.clone());
        mm[index] = Some(values);
    }
    (mask, mm)
}

fn g3_mm(raw: &[f32], reversed: bool) -> Vec<f32> {
    raw.iter()
        .copied()
        .map(|value| {
            if reversed {
                remap(value, 1.0, 128.0, 128.0, -128.0)
            } else {
                remap(value, 1.0, 128.0, -128.0, 128.0)
            }
        })
        .collect()
}

fn spot_sigma(spot: &Sheet, prefer_raw: bool, n: usize) -> Vec<(&'static str, Vec<f32>)> {
    let order: &[&str] = if prefer_raw {
        &["spot_sigma_raw", "spot_sigma"]
    } else {
        &["spot_sigma", "spot_sigma_raw"]
    };
    let mut out = Vec::new();
    for (ic, axis, key) in [
        ("ic1", "x", "ic1_sig_x"),
        ("ic1", "y", "ic1_sig_y"),
        ("ic2", "x", "ic2_sig_x"),
        ("ic2", "y", "ic2_sig_y"),
    ] {
        let Some(values) = column_suffix(spot, ic, axis, order) else {
            continue;
        };
        if values.len() >= n {
            out.push((key, values[..n].iter().map(|value| value * 2.0).collect()));
        }
    }
    out
}

fn four_columns(spot: &Sheet, suffix: &str, n: usize) -> Option<[Vec<f32>; 4]> {
    let mut out = Vec::new();
    for (ic, axis) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")] {
        let values = column_suffix(spot, ic, axis, &[suffix])?;
        if values.len() < n {
            return None;
        }
        out.push(values[..n].to_vec());
    }
    Some([
        out[0].clone(),
        out[1].clone(),
        out[2].clone(),
        out[3].clone(),
    ])
}

fn column_suffix<'a>(
    spot: &'a Sheet,
    ic: &str,
    axis: &str,
    suffixes: &[&str],
) -> Option<&'a [f32]> {
    for suffix in suffixes {
        let with_r = format!("r_{ic}_{axis}_{suffix}");
        let plain = format!("{ic}_{axis}_{suffix}");
        if let Some(values) = spot.num.get(&with_r).or_else(|| spot.num.get(&plain)) {
            return Some(values);
        }
    }
    None
}

fn lookup_point_time(dir: &Path, spot_no: &[f32], layer: &[f32]) -> Vec<f32> {
    let found = layer_point_times(dir);
    spot_no
        .iter()
        .zip(layer)
        .map(|(spot, layer)| {
            found
                .get(&(spot.to_bits(), layer.to_bits()))
                .copied()
                .unwrap_or(f32::NAN)
        })
        .collect()
}

fn layer_point_times(dir: &Path) -> HashMap<(u32, u32), f32> {
    let mut found = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    let mut layers: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("layer-"))
        })
        .collect();
    layers.sort();
    for layer_dir in layers {
        let Ok(entries) = std::fs::read_dir(&layer_dir) else {
            continue;
        };
        let mut runs: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("run-"))
            })
            .collect();
        runs.sort();
        for run in runs {
            if let Some(path) = point_time_file(&run) {
                if let Ok(bytes) = std::fs::read(&path) {
                    insert_point_times(&mut found, &read_sheet(&bytes));
                }
                break;
            }
        }
    }
    found
}

fn point_time_file(run: &Path) -> Option<PathBuf> {
    for name in [
        "FX4_spot_data.csv",
        "IX256_1_spot_data.csv",
        "IX256_2_spot_data.csv",
        "RCI_spot_data.csv",
    ] {
        let path = run.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    let Ok(entries) = std::fs::read_dir(run) else {
        return None;
    };
    let mut others: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_spot_data.csv") && name != "TX2_spot_data.csv")
        })
        .collect();
    others.sort();
    others.into_iter().next()
}

fn insert_point_times(found: &mut HashMap<(u32, u32), f32>, sheet: &Sheet) {
    let Some(spots) = sheet.num.get("spot_no") else {
        return;
    };
    let Some(layers) = sheet.num.get("layer_id") else {
        return;
    };
    let Some(point) = ["point_time(ms)", "point_time_ms", "point_time"]
        .iter()
        .find_map(|name| sheet.num.get(*name))
    else {
        return;
    };
    for ((spot, layer), point) in spots.iter().zip(layers).zip(point) {
        if spot.is_finite() && layer.is_finite() && point.is_finite() {
            found.insert((spot.to_bits(), layer.to_bits()), *point);
        }
    }
}

fn add_sigma_error(table: &mut BTreeMap<String, Vec<f32>>, xml: &str) {
    let bands = parse_sigma_bands(xml);
    let Some(energy) = table.get("energy").cloned() else {
        return;
    };
    let mut pairs = Vec::new();
    for (key, device) in [
        ("ic1_sig_x", "IC_1_X"),
        ("ic1_sig_y", "IC_1_Y"),
        ("ic2_sig_x", "IC_2_X"),
        ("ic2_sig_y", "IC_2_Y"),
    ] {
        let Some(measured) = table.get(key).cloned() else {
            continue;
        };
        let mut error = Vec::with_capacity(measured.len());
        for (sample, energy) in measured.iter().zip(&energy) {
            match expected_mm(&bands, device, f64::from(*energy)) {
                Some(expected) if sample.is_finite() => {
                    pairs.push(*energy);
                    pairs.push(expected as f32);
                    error.push(sample - expected as f32);
                }
                _ => error.push(f32::NAN),
            }
        }
        table.insert(format!("{key}_err"), error);
    }
    if !pairs.is_empty() {
        table.insert("expected_sigma".to_string(), pairs);
    }
}

fn add_dose_rate(table: &mut BTreeMap<String, Vec<f32>>, time: &[f64]) {
    let energy = &table["energy"];
    let charge = &table["target_mu"];
    let n = energy.len().min(charge.len()).min(time.len());
    if n == 0 {
        return;
    }
    let mu_total: f64 = charge
        .iter()
        .take(n)
        .filter(|value| value.is_finite())
        .map(|value| f64::from(*value))
        .sum();
    let mut t_min = f64::MAX;
    let mut t_max = f64::MIN;
    let mut t_count = 0usize;
    for stamp in time.iter().take(n) {
        if stamp.is_finite() {
            t_min = t_min.min(*stamp);
            t_max = t_max.max(*stamp);
            t_count += 1;
        }
    }
    let span = t_max - t_min;
    if t_count < 2 || span < 0.05 || mu_total <= 0.0 {
        return;
    }
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for index in 0..n {
        if energy[index].is_finite() {
            groups
                .entry(energy[index].to_bits())
                .or_default()
                .push(index);
        }
    }
    let mut rows = Vec::new();
    for (bits, indices) in groups {
        let mut mu = 0.0f64;
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        let mut count = 0usize;
        for index in indices {
            if charge[index].is_finite() {
                mu += f64::from(charge[index]);
            }
            if time[index].is_finite() {
                lo = lo.min(time[index]);
                hi = hi.max(time[index]);
                count += 1;
            }
        }
        let duration = hi - lo;
        if count >= 2 && duration >= 0.05 && mu > 0.0 {
            rows.push((f32::from_bits(bits), (mu / duration) as f32));
        }
    }
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    table.insert(
        "rate_energy".to_string(),
        rows.iter().map(|row| row.0).collect(),
    );
    table.insert(
        "mu_rate".to_string(),
        rows.iter().map(|row| row.1).collect(),
    );
    table.insert(
        "session_avg_rate".to_string(),
        vec![(mu_total / span) as f32],
    );
}

fn wall_time(spot: &Sheet, n: usize) -> Option<Vec<f64>> {
    if let Some(stamp) = spot.wide.get("datetime") {
        if stamp.iter().take(n).any(|value| value.is_finite()) {
            return Some(stamp.iter().take(n).copied().collect());
        }
    }
    if let (Some(seconds), Some(nanos)) = (spot.wide.get("time_s"), spot.wide.get("time_ns")) {
        let n = n.min(seconds.len()).min(nanos.len());
        return Some(
            (0..n)
                .map(|index| seconds[index] + nanos[index] * 1e-9)
                .collect(),
        );
    }
    let stamp = spot.wide.get("timestamp")?;
    Some(stamp.iter().take(n).map(|value| value / 1000.0).collect())
}

pub(crate) fn load_slice_metric(
    root: &Path,
    session: &str,
    _metric: &str,
    chamber: bool,
) -> BTreeMap<String, Vec<f32>> {
    let kind = if chamber {
        "slice-chamber"
    } else {
        "slice-iso"
    };
    cached_timeslice(root, session, kind, true, |frames| {
        build_slice_metric(root, session, chamber, frames)
    })
}

fn build_slice_metric(
    root: &Path,
    session: &str,
    chamber: bool,
    frames: &Frames,
) -> BTreeMap<String, Vec<f32>> {
    let spot =
        discover::read_session_file(root, session, "spot_data.csv").map(|bytes| read_sheet(&bytes));
    let plan = plan_xy(&frames.map);
    let (axes, spot_shift) = iso_axes(spot.as_ref(), &plan, &frames.sheets);
    let mut energy = Vec::new();
    let mut beam = Vec::new();
    let mut ic1_x = Vec::new();
    let mut ic1_y = Vec::new();
    let mut ic2_x = Vec::new();
    let mut ic2_y = Vec::new();
    let mut ic1_x_mm = Vec::new();
    let mut ic1_y_mm = Vec::new();
    let mut ic2_x_mm = Vec::new();
    let mut ic2_y_mm = Vec::new();
    let mut sig = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut sig_err = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (file_index, sheet) in frames.sheets.iter().enumerate() {
        let tagged = frame_energy(
            &frames.energies,
            frames.layers.get(file_index).copied().unwrap_or(-1),
            file_index,
        );
        let Some(trigger) = sheet
            .num
            .get("rci_in_trigger")
            .or_else(|| sheet.num.get("r_beamOk"))
        else {
            continue;
        };
        let n = trigger.len();
        let layer_col = sheet.num.get("layer_id").map(Vec::as_slice);
        let spots_ic1 = first_column(sheet, &["spot_no.1", "spot_no"]);
        let spots_ic2 = first_column(sheet, &["spot_no.2", "spot_no.1", "spot_no"]);
        let bound = [
            bind_axis(sheet, "ic1", "x"),
            bind_axis(sheet, "ic1", "y"),
            bind_axis(sheet, "ic2", "x"),
            bind_axis(sheet, "ic2", "y"),
        ];
        let spot_of = [spots_ic1, spots_ic1, spots_ic2, spots_ic2];
        let plan_axis = [0usize, 1, 0, 1];
        for row in 0..n {
            let on = trigger[row] > 0.5;
            beam.push(if on { 1.0 } else { 0.0 });
            let layer = at(layer_col, row);
            energy.push(tagged);
            let mut mm = [f32::NAN; 4];
            for index in 0..4 {
                let axis = &bound[index];
                let raw = at(axis.raw, row);
                let accepted = axis_passes(axis, row);
                mm[index] = if !accepted {
                    f32::NAN
                } else if chamber {
                    if raw.is_finite() && raw >= 0.0 {
                        remap_g3_raw(raw)
                    } else {
                        f32::NAN
                    }
                } else if let Some((slope, intercept)) = axes[index] {
                    if raw.is_finite() && raw >= 0.0 {
                        slope * raw + intercept
                    } else {
                        f32::NAN
                    }
                } else {
                    f32::NAN
                };
                let spot_no = at(spot_of[index], row) + spot_shift;
                let planned = plan
                    .get(&(layer.to_bits(), spot_no.to_bits()))
                    .map(|xy| if plan_axis[index] == 0 { xy.0 } else { xy.1 })
                    .unwrap_or(f32::NAN);
                let err = if mm[index].is_finite() && planned.is_finite() {
                    mm[index] - planned
                } else {
                    f32::NAN
                };
                match index {
                    0 => {
                        ic1_x.push(err);
                        ic1_x_mm.push(mm[index]);
                    }
                    1 => {
                        ic1_y.push(err);
                        ic1_y_mm.push(mm[index]);
                    }
                    2 => {
                        ic2_x.push(err);
                        ic2_x_mm.push(mm[index]);
                    }
                    _ => {
                        ic2_y.push(err);
                        ic2_y_mm.push(mm[index]);
                    }
                }
            }
            for (index, axis) in bound.iter().enumerate() {
                let value = at(axis.sigma, row);
                let value = if !value.is_finite()
                    || value <= 0.0
                    || value > 20.0
                    || !axis_passes(axis, row)
                {
                    f32::NAN
                } else {
                    value
                };
                sig[index].push(value);
                let target = at(axis.sigma_target, row);
                sig_err[index].push(if value.is_finite() && target.is_finite() && target > 0.0 {
                    value - target
                } else {
                    f32::NAN
                });
            }
        }
    }
    if chamber {
        if ic2_closer_reversed(&ic1_x_mm, &ic2_x_mm) {
            flip_chamber_error(&mut ic2_x, &ic2_x_mm);
            negate_finite(&mut ic2_x_mm);
        }
        if ic2_closer_reversed(&ic1_y_mm, &ic2_y_mm) {
            flip_chamber_error(&mut ic2_y, &ic2_y_mm);
            negate_finite(&mut ic2_y_mm);
        }
    }
    let mut table = BTreeMap::new();
    if energy.is_empty() {
        return table;
    }
    let step = sample_stride(energy.len());
    let take = |values: Vec<f32>| values.into_iter().step_by(step).collect::<Vec<_>>();
    table.insert("energy".to_string(), take(energy));
    table.insert("beam_on".to_string(), take(beam));
    table.insert("ic1_x_err".to_string(), take(ic1_x));
    table.insert("ic1_y_err".to_string(), take(ic1_y));
    table.insert("ic2_x_err".to_string(), take(ic2_x));
    table.insert("ic2_y_err".to_string(), take(ic2_y));
    for (key, values) in [
        ("ic1_sig_x", sig[0].clone()),
        ("ic1_sig_y", sig[1].clone()),
        ("ic2_sig_x", sig[2].clone()),
        ("ic2_sig_y", sig[3].clone()),
        ("ic1_sig_x_err", sig_err[0].clone()),
        ("ic1_sig_y_err", sig_err[1].clone()),
        ("ic2_sig_x_err", sig_err[2].clone()),
        ("ic2_sig_y_err", sig_err[3].clone()),
    ] {
        table.insert(key.to_string(), take(values));
    }
    let gap = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(ic2, ic1)| ic2 - ic1)
            .collect::<Vec<_>>()
    };
    table.insert("ic1_x".to_string(), take(ic1_x_mm.clone()));
    table.insert("ic1_y".to_string(), take(ic1_y_mm.clone()));
    table.insert("ic2_x".to_string(), take(ic2_x_mm.clone()));
    table.insert("ic2_y".to_string(), take(ic2_y_mm.clone()));
    table.insert("ic12_x_diff".to_string(), take(gap(&ic2_x_mm, &ic1_x_mm)));
    table.insert("ic12_y_diff".to_string(), take(gap(&ic2_y_mm, &ic1_y_mm)));
    table
}

struct AxisCols<'a> {
    raw: Option<&'a [f32]>,
    gate: Option<&'a [f32]>,
    confidence: Option<&'a [f32]>,
    error_code: Option<&'a [f32]>,
    sigma: Option<&'a [f32]>,
    sigma_target: Option<&'a [f32]>,
}

fn bind_axis<'a>(sheet: &'a Sheet, ic: &str, axis: &str) -> AxisCols<'a> {
    let position = format!("r_{ic}_{axis}_position");
    let spot_position = format!("r_{ic}_{axis}_spot_position");
    let gates = [
        format!("{ic}_{axis}_fit_ok"),
        format!("r_{ic}_{axis}_spot_position_ok"),
        format!("r_{ic}_{axis}_position_ok"),
    ];
    AxisCols {
        raw: sheet
            .num
            .get(&position)
            .or_else(|| sheet.num.get(&spot_position))
            .map(Vec::as_slice),
        gate: gates
            .iter()
            .find_map(|name| sheet.num.get(name))
            .map(Vec::as_slice),
        confidence: sheet
            .num
            .get(&format!("r_{ic}_{axis}_confidence"))
            .map(Vec::as_slice),
        error_code: sheet
            .num
            .get(&format!("r_{ic}_{axis}_spot_error_code"))
            .map(Vec::as_slice),
        sigma: sheet
            .num
            .get(&format!("r_{ic}_{axis}_sigma"))
            .map(Vec::as_slice),
        sigma_target: sheet
            .num
            .get(&format!("{ic}_sigma_{axis}_target"))
            .map(Vec::as_slice),
    }
}

fn first_column<'a>(sheet: &'a Sheet, names: &[&str]) -> Option<&'a [f32]> {
    names
        .iter()
        .find_map(|name| sheet.num.get(*name))
        .map(Vec::as_slice)
}

fn at(column: Option<&[f32]>, row: usize) -> f32 {
    column
        .and_then(|values| values.get(row))
        .copied()
        .unwrap_or(f32::NAN)
}

fn axis_passes(axis: &AxisCols<'_>, row: usize) -> bool {
    if let Some(values) = axis.gate {
        let value = values.get(row).copied().unwrap_or(f32::NAN);
        if !value.is_finite() || value == 0.0 {
            return false;
        }
    }
    if let Some(values) = axis.confidence {
        let value = values.get(row).copied().unwrap_or(f32::NAN);
        if !value.is_finite() || value < 80.0 {
            return false;
        }
    }
    if let Some(values) = axis.error_code {
        let value = values.get(row).copied().unwrap_or(f32::NAN);
        if !value.is_finite() || value != 0.0 {
            return false;
        }
    }
    true
}

fn ic2_closer_reversed(ic1_mm: &[f32], ic2_mm: &[f32]) -> bool {
    let mut forward = Vec::new();
    let mut reversed = Vec::new();
    for (ic1, ic2) in ic1_mm.iter().zip(ic2_mm) {
        if ic1.is_finite() && ic2.is_finite() {
            forward.push((ic1 - ic2).abs());
            reversed.push((ic1 + ic2).abs());
        }
    }
    if forward.is_empty() {
        return false;
    }
    median(&reversed) < median(&forward)
}

fn negate_finite(values: &mut [f32]) {
    for value in values {
        if value.is_finite() {
            *value = -*value;
        }
    }
}

fn flip_chamber_error(err: &mut [f32], forward_mm: &[f32]) {
    for (err, mm) in err.iter_mut().zip(forward_mm) {
        if err.is_finite() && mm.is_finite() {
            let planned = *mm - *err;
            *err = -*mm - planned;
        }
    }
}

fn plan_xy(map: &Sheet) -> HashMap<(u32, u32), (f32, f32)> {
    let mut out = HashMap::new();
    let layer = column(map, "layer_id", &["layer_id"]);
    let spot = column(map, "spot_no", &["spot_no"]);
    let x = column(map, "x_position", &["position_x", "X_POSITION", "x_pos"]);
    let y = column(map, "y_position", &["position_y", "Y_POSITION", "y_pos"]);
    for (((layer, spot), x), y) in layer.iter().zip(&spot).zip(&x).zip(&y) {
        out.insert((layer.to_bits(), spot.to_bits()), (*x, *y));
    }
    out
}

fn iso_axes(
    spot: Option<&Sheet>,
    plan: &HashMap<(u32, u32), (f32, f32)>,
    sheets: &[Sheet],
) -> ([Option<(f32, f32)>; 4], f32) {
    if let Some(fitted) = plan_target_axes(plan, sheets) {
        return fitted;
    }
    (
        spot.map(strip_axes).unwrap_or([None, None, None, None]),
        0.0,
    )
}

fn plan_spans(plan: &HashMap<(u32, u32), (f32, f32)>) -> bool {
    let mut x_lo = f32::MAX;
    let mut x_hi = f32::MIN;
    let mut y_lo = f32::MAX;
    let mut y_hi = f32::MIN;
    for (x, y) in plan.values() {
        if x.is_finite() {
            x_lo = x_lo.min(*x);
            x_hi = x_hi.max(*x);
        }
        if y.is_finite() {
            y_lo = y_lo.min(*y);
            y_hi = y_hi.max(*y);
        }
    }
    x_hi - x_lo >= 0.1 || y_hi - y_lo >= 0.1
}

/// Device-target vs plan affine when the map has position span.
/// ponytail: the first finite target per spot stands in for the per-spot median.
fn plan_target_axes(
    plan: &HashMap<(u32, u32), (f32, f32)>,
    sheets: &[Sheet],
) -> Option<([Option<(f32, f32)>; 4], f32)> {
    if !plan_spans(plan) {
        return None;
    }
    let mut targets: HashMap<(u32, u32), [f32; 4]> = HashMap::new();
    for sheet in sheets {
        let names = [
            "ic1_position_x_target",
            "ic1_position_y_target",
            "ic2_position_x_target",
            "ic2_position_y_target",
        ];
        if names.iter().any(|name| !sheet.num.contains_key(*name)) {
            continue;
        }
        let n = sheet.num.get("layer_id").map(Vec::len).unwrap_or(0);
        let layer_col = sheet.num.get("layer_id").map(Vec::as_slice);
        let spots = first_column(sheet, &["spot_no.1", "spot_no"]);
        let cols = names.map(|name| sheet.num.get(name).map(Vec::as_slice));
        for row in 0..n {
            let layer = at(layer_col, row);
            let spot = at(spots, row);
            if !layer.is_finite() || !spot.is_finite() {
                continue;
            }
            let sample = [
                at(cols[0], row),
                at(cols[1], row),
                at(cols[2], row),
                at(cols[3], row),
            ];
            if sample.iter().any(|value| !value.is_finite()) {
                continue;
            }
            targets
                .entry((layer.to_bits(), spot.to_bits()))
                .or_insert(sample);
        }
    }
    if targets.len() < 10 {
        return None;
    }
    let mut best: Option<(f32, i32, i32, [(f32, f32); 4])> = None;
    for shift in -2i32..=2 {
        let mut strips = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut isos = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for ((layer, spot), sample) in &targets {
            let shifted = f32::from_bits(*spot) + shift as f32;
            let Some(planned) = plan.get(&(*layer, shifted.to_bits())) else {
                continue;
            };
            for index in 0..4 {
                strips[index].push(sample[index]);
                isos[index].push(if index % 2 == 0 { planned.0 } else { planned.1 });
            }
        }
        let matches = strips[0].len();
        if matches < 10 {
            continue;
        }
        let mut axes = [(0.0f32, 0.0f32); 4];
        let mut ok = true;
        for index in 0..4 {
            match fit_strip_iso(&strips[index], &isos[index]) {
                Some(axis) => axes[index] = axis,
                None => ok = false,
            }
        }
        if !ok {
            continue;
        }
        let mut square = 0.0f32;
        let mut count = 0.0f32;
        for ((layer, spot), sample) in &targets {
            let Some(planned) = plan.get(&(*layer, *spot)) else {
                continue;
            };
            let actual = [planned.0, planned.1, planned.0, planned.1];
            for index in 0..4 {
                let pred = axes[index].0 * sample[index] + axes[index].1;
                let err = actual[index] - pred;
                square += err * err;
                count += 1.0;
            }
        }
        if count == 0.0 {
            continue;
        }
        let verify = (square / count).sqrt();
        let key = (verify, -(matches as i32), shift.abs());
        if best
            .as_ref()
            .is_none_or(|have| key < (have.0, have.1, have.2))
        {
            best = Some((verify, -(matches as i32), shift.abs(), axes));
        }
    }
    let (_, _, _, axes) = best?;
    // The lookup shift is the one with the most plan matches, same as Python.
    let mut chosen = 0i32;
    let mut most = 0usize;
    for shift in -2i32..=2 {
        let matches = targets
            .keys()
            .filter(|(layer, spot)| {
                let shifted = f32::from_bits(*spot) + shift as f32;
                plan.contains_key(&(*layer, shifted.to_bits()))
            })
            .count();
        if matches > most {
            most = matches;
            chosen = shift;
        }
    }
    if most < 10 {
        return None;
    }
    Some((axes.map(Some), chosen as f32))
}

fn strip_axes(spot: &Sheet) -> [Option<(f32, f32)>; 4] {
    let mut axes = [None, None, None, None];
    for (index, (ic, axis)) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")]
        .into_iter()
        .enumerate()
    {
        let Some(strip) = column_suffix(spot, ic, axis, &["spot_position_raw"]) else {
            continue;
        };
        let Some(iso) = column_suffix(spot, ic, axis, &["spot_position"]) else {
            continue;
        };
        axes[index] = fit_strip_iso(strip, iso);
    }
    axes
}

fn fit_strip_iso(strip: &[f32], iso: &[f32]) -> Option<(f32, f32)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (strip, iso) in strip.iter().zip(iso) {
        if strip.is_finite() && iso.is_finite() && *strip > 1.0 && *strip < 128.0 {
            xs.push(*strip);
            ys.push(*iso);
        }
    }
    if xs.len() < 10 {
        return None;
    }
    let (slope, intercept) = linear_fit(&xs, &ys)?;
    if slope.abs() < 0.01 {
        return None;
    }
    let span =
        ys.iter().copied().fold(f32::MIN, f32::max) - ys.iter().copied().fold(f32::MAX, f32::min);
    if span < 0.1 {
        return None;
    }
    let mut resid = 0.0f32;
    for (x, y) in xs.iter().zip(&ys) {
        let err = y - (slope * x + intercept);
        resid += err * err;
    }
    if (resid / xs.len() as f32).sqrt() > 0.05 {
        return None;
    }
    Some((slope, intercept))
}

pub(crate) fn load_timeslice(
    root: &Path,
    session: &str,
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    cached_timeslice(
        root,
        session,
        &format!("metric:{metric}"),
        false,
        |frames| {
            if metric == "current_ratio" {
                return current_ratio_from(&frames.sheets, &frames.energies, &frames.layers);
            }
            if matches!(
                metric,
                "fit_confidence" | "peak_amplitude" | "amplifier_error" | "probe_field"
            ) {
                return signal_from(&frames.sheets, &frames.energies, &frames.layers, metric);
            }
            current_from(&frames.sheets, &frames.energies, &frames.layers)
        },
    )
}

pub(crate) fn timeslice_energy_only(metric: &str) -> bool {
    matches!(
        metric,
        "fit_confidence" | "peak_amplitude" | "amplifier_error" | "probe_field"
    )
}

/// Folder `layer-N` picks the Nth energy in input-map order. A path with no
/// layer folder uses the file's order instead.
fn frame_energy(energies: &[f32], layer_idx: i64, file_index: usize) -> f32 {
    let index = if layer_idx >= 0 {
        layer_idx as usize
    } else {
        file_index
    };
    energies.get(index).copied().unwrap_or(f32::NAN)
}

#[cfg(test)]
pub(crate) fn signal_table(
    files: &[Vec<u8>],
    energies: &[f32],
    layers: &[i64],
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    signal_from(&read_sheets(files, signal_column), energies, layers, metric)
}

fn signal_from(
    sheets: &[Sheet],
    energies: &[f32],
    layers: &[i64],
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    let spec: &[(&str, &str)] = match metric {
        "fit_confidence" => &[
            ("ic1_x_confidence", "r_ic1_x_confidence"),
            ("ic1_y_confidence", "r_ic1_y_confidence"),
            ("ic2_x_confidence", "r_ic2_x_confidence"),
            ("ic2_y_confidence", "r_ic2_y_confidence"),
        ],
        "peak_amplitude" => &[
            ("ic1_x_peak", "ic1_x_peak_amplitude"),
            ("ic1_y_peak", "ic1_y_peak_amplitude"),
            ("ic2_x_peak", "ic2_x_peak_amplitude"),
            ("ic2_y_peak", "ic2_y_peak_amplitude"),
        ],
        "probe_field" => &[("field_x", "mag_field_x"), ("field_y", "mag_field_y")],
        _ => &[],
    };
    let mut energy = Vec::new();
    let mut beam = Vec::new();
    if metric == "amplifier_error" {
        let mut stored = vec![Vec::new(); 6];
        for (index, sheet) in sheets.iter().enumerate() {
            let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
            let names: Vec<String> = sheet.num.keys().cloned().collect();
            let cmd_x = concept_col(sheet, &names, "amplifier_cmd_x");
            let read_x = concept_col(sheet, &names, "amplifier_readback_x");
            let cmd_y = concept_col(sheet, &names, "amplifier_cmd_y");
            let read_y = concept_col(sheet, &names, "amplifier_readback_y");
            let diff_x = column_diff(cmd_x, read_x);
            let diff_y = column_diff(cmd_y, read_y);
            let columns = [
                diff_x.as_slice(),
                diff_y.as_slice(),
                cmd_x,
                read_x,
                cmd_y,
                read_y,
            ];
            append_samples(sheet, tag, &columns, &mut energy, &mut beam, &mut stored);
        }
        return signal_rows(
            energy,
            beam,
            &[
                ("amp_x", &stored[0]),
                ("amp_y", &stored[1]),
                ("amp_cmd_x", &stored[2]),
                ("amp_read_x", &stored[3]),
                ("amp_cmd_y", &stored[4]),
                ("amp_read_y", &stored[5]),
            ],
        );
    }
    let mut stored = vec![Vec::new(); spec.len()];
    for (index, sheet) in sheets.iter().enumerate() {
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let names: Vec<String> = sheet.num.keys().cloned().collect();
        let columns: Vec<&[f32]> = spec
            .iter()
            .map(|(_, concept)| concept_col(sheet, &names, concept))
            .collect();
        append_samples(sheet, tag, &columns, &mut energy, &mut beam, &mut stored);
    }
    let pairs: Vec<(&str, &Vec<f32>)> = spec
        .iter()
        .zip(&stored)
        .map(|((key, _), values)| (*key, values))
        .collect();
    signal_rows(energy, beam, &pairs)
}

fn concept_col<'a>(sheet: &'a Sheet, names: &[String], concept: &str) -> &'a [f32] {
    resolve_concept_column(names, concept)
        .and_then(|name| sheet.num.get(name))
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// Readback minus command. An axis is absent when either column is missing.
fn column_diff(command: &[f32], readback: &[f32]) -> Vec<f32> {
    if command.is_empty() || readback.is_empty() {
        return Vec::new();
    }
    command
        .iter()
        .zip(readback)
        .map(|(command, readback)| {
            if command.is_finite() && readback.is_finite() {
                readback - command
            } else {
                f32::NAN
            }
        })
        .collect()
}

fn append_samples(
    sheet: &Sheet,
    tag: f32,
    columns: &[&[f32]],
    energy: &mut Vec<f32>,
    beam_out: &mut Vec<f32>,
    series_out: &mut [Vec<f32>],
) {
    let n = columns.iter().map(|column| column.len()).max().unwrap_or(0);
    if n == 0 {
        return;
    }
    let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
    let stride = sample_stride(n);
    for sample in (0..n).step_by(stride) {
        let any = columns
            .iter()
            .any(|column| column.get(sample).copied().unwrap_or(f32::NAN).is_finite());
        if !any {
            continue;
        }
        energy.push(tag);
        for (out, column) in series_out.iter_mut().zip(columns) {
            out.push(column.get(sample).copied().unwrap_or(f32::NAN));
        }
        beam_out.push(
            gate.and_then(|values| values.get(sample).copied())
                .unwrap_or(1.0),
        );
    }
}

fn signal_rows(
    energy: Vec<f32>,
    beam: Vec<f32>,
    pairs: &[(&str, &Vec<f32>)],
) -> BTreeMap<String, Vec<f32>> {
    let mut table = BTreeMap::new();
    if energy.is_empty() {
        return table;
    }
    table.insert("energy".to_string(), energy);
    table.insert("beam_on".to_string(), beam);
    for (key, values) in pairs {
        if values.iter().any(|value| value.is_finite()) {
            table.insert((*key).to_string(), (*values).clone());
        }
    }
    table
}

fn current_from(sheets: &[Sheet], energies: &[f32], layers: &[i64]) -> BTreeMap<String, Vec<f32>> {
    let mut out_energy = Vec::new();
    let mut ic1 = Vec::new();
    let mut ic2 = Vec::new();
    let mut ic3 = Vec::new();
    let mut beam = Vec::new();
    let mut any_ic3 = false;
    for (index, sheet) in sheets.iter().enumerate() {
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let a = scaled_current(sheet, "ic1");
        let b = scaled_current(sheet, "ic2");
        let c = ic3_current(sheet);
        let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
        let n = a.len().max(b.len()).max(c.len());
        let stride = sample_stride(n);
        for sample in (0..n).step_by(stride) {
            out_energy.push(tag);
            ic1.push(*a.get(sample).unwrap_or(&f32::NAN));
            ic2.push(*b.get(sample).unwrap_or(&f32::NAN));
            let part = *c.get(sample).unwrap_or(&f32::NAN);
            if part.is_finite() {
                any_ic3 = true;
            }
            ic3.push(part);
            beam.push(gate.and_then(|v| v.get(sample).copied()).unwrap_or(1.0));
        }
    }
    let mut table = BTreeMap::new();
    if !out_energy.is_empty() {
        table.insert("energy".to_string(), out_energy);
        table.insert("ic1_current".to_string(), ic1);
        table.insert("ic2_current".to_string(), ic2);
        if any_ic3 {
            table.insert("ic3_current".to_string(), ic3);
        }
        table.insert("beam_on".to_string(), beam);
    }
    table
}

fn current_ratio_from(
    sheets: &[Sheet],
    energies: &[f32],
    layers: &[i64],
) -> BTreeMap<String, Vec<f32>> {
    let mut out_energy = Vec::new();
    let mut ic21 = Vec::new();
    let mut ic31 = Vec::new();
    let mut ic32 = Vec::new();
    let mut any_ic3 = false;
    for (index, sheet) in sheets.iter().enumerate() {
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let a = scaled_current(sheet, "ic1");
        let b = scaled_current(sheet, "ic2");
        let c = ic3_current(sheet);
        let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
        let plateau = |values: &[f32]| plateau_mean(values, gate);
        let left = plateau(&a);
        let right = plateau(&b);
        let third = plateau(&c);
        out_energy.push(tag);
        ic21.push(sym_pct(right, left));
        if third.is_finite() {
            any_ic3 = true;
        }
        ic31.push(sym_pct(third, left));
        ic32.push(sym_pct(third, right));
    }
    let mut table = BTreeMap::new();
    if !out_energy.is_empty() {
        let rows = out_energy.len();
        table.insert("energy".to_string(), out_energy);
        table.insert("beam_on".to_string(), vec![1.0; rows]);
        table.insert("ic21_ratio".to_string(), ic21);
        if any_ic3 {
            table.insert("ic31_ratio".to_string(), ic31);
            table.insert("ic32_ratio".to_string(), ic32);
        }
    }
    table
}

fn plateau_mean(values: &[f32], gate: Option<&[f32]>) -> f32 {
    let on: Vec<f32> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let passed = gate
                .map(|gate| gate.get(index).copied().unwrap_or(0.0) > 0.5)
                .unwrap_or(true);
            (passed && value.is_finite()).then_some(*value)
        })
        .collect();
    if on.len() < 10 {
        return f32::NAN;
    }
    let mut sorted = on.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = percentile(&sorted, 0.95);
    let floor = 5.0f32.max(0.75 * p95);
    let kept: Vec<f32> = on.into_iter().filter(|value| *value >= floor).collect();
    if kept.len() < 3 {
        f32::NAN
    } else {
        kept.iter().sum::<f32>() / kept.len() as f32
    }
}

fn sym_pct(a: f32, b: f32) -> f32 {
    if !a.is_finite() || !b.is_finite() || (a + b).abs() < 1e-12 {
        f32::NAN
    } else {
        (a - b) / ((a + b) / 2.0) * 100.0
    }
}

fn scaled_current(sheet: &Sheet, ic: &str) -> Vec<f32> {
    let fallback = [
        format!("r_{ic}_current_dose"),
        format!("{ic}_current_dose"),
        format!("r_{ic}_current"),
        format!("{ic}_current"),
    ];
    let keys: Vec<String> = sheet.num.keys().cloned().collect();
    let header = resolve_concept_column(&keys, &format!("{ic}_current"))
        .map(str::to_owned)
        .or_else(|| {
            fallback
                .iter()
                .find(|name| sheet.num.contains_key(name.as_str()))
                .cloned()
        });
    let Some(header) = header else {
        return Vec::new();
    };
    let Some(values) = sheet.num.get(&header) else {
        return Vec::new();
    };
    let factor = column_scale_factor(&header).unwrap_or(1.0) as f32;
    scale_column(values, factor)
}

pub(crate) fn ic3_current(sheet: &Sheet) -> Vec<f32> {
    let keys: Vec<String> = sheet.num.keys().cloned().collect();
    let mut sum = Vec::new();
    for (concept, part) in [
        ("ic3_current_a", "a"),
        ("ic3_current_b", "b"),
        ("ic3_current_c", "c"),
        ("ic3_current_d", "d"),
    ] {
        let values = if let Some(header) = resolve_concept_column(&keys, concept) {
            let factor = column_scale_factor(header).unwrap_or(1.0) as f32;
            sheet
                .num
                .get(header)
                .map(|values| scale_column(values, factor))
                .unwrap_or_default()
        } else {
            scaled_current(sheet, &format!("ic3_{part}"))
        };
        if values.is_empty() {
            continue;
        }
        if sum.is_empty() {
            sum = values;
        } else {
            for (slot, value) in sum.iter_mut().zip(values) {
                if value.is_finite() {
                    *slot = if slot.is_finite() {
                        *slot + value
                    } else {
                        value
                    };
                }
            }
        }
    }
    if sum.is_empty() {
        scaled_current(sheet, "ic3")
    } else {
        sum
    }
}

pub(crate) fn column(sheet: &Sheet, concept: &str, extra: &[&str]) -> Vec<f32> {
    let names: Vec<String> = sheet.num.keys().cloned().collect();
    if let Some(found) = resolve_concept_column(&names, concept) {
        if let Some(values) = sheet.num.get(found) {
            return values.clone();
        }
    }
    extra
        .iter()
        .find_map(|name| sheet.num.get(*name).cloned())
        .unwrap_or_default()
}

fn column_any<'a>(sheet: &'a Sheet, names: &[&str]) -> Option<&'a [f32]> {
    names
        .iter()
        .find_map(|name| sheet.num.get(*name))
        .map(Vec::as_slice)
}

pub(crate) fn same(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs()))
}

pub(crate) fn span(values: &[f32]) -> (f32, f32) {
    let finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    let lo = finite.iter().copied().fold(f32::MAX, f32::min);
    let hi = finite.iter().copied().fold(f32::MIN, f32::max);
    if (hi - lo).abs() < 1e-6 {
        return (lo - 1.0, hi + 1.0);
    }
    let pad = (hi - lo) * 0.05;
    (lo - pad, hi + pad)
}

pub(crate) fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

pub(crate) fn percentile(sorted_or_not: &[f32], q: f64) -> f32 {
    let mut sorted = sorted_or_not.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return f32::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = (pos - lo as f64) as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

pub(crate) fn modified_z(values: &[f32]) -> Vec<f32> {
    let finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mut out = vec![0.0; values.len()];
    if finite.is_empty() {
        return out;
    }
    let med = median(&finite);
    let devs: Vec<f32> = finite.iter().map(|value| (value - med).abs()).collect();
    let mad = median(&devs);
    let mean = finite.iter().sum::<f32>() / finite.len() as f32;
    let mean_ad = devs.iter().sum::<f32>() / devs.len() as f32;
    for (slot, value) in out.iter_mut().zip(values) {
        if !value.is_finite() {
            continue;
        }
        *slot = if mad > 1e-12 {
            0.6745 * (value - med) / mad
        } else if mean_ad > 1e-12 {
            (value - mean) / (1.253314 * mean_ad)
        } else {
            0.0
        };
    }
    out
}

fn unique_seen(values: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for value in values {
        if value.is_finite() && !out.iter().any(|have: &f32| same(*have, *value)) {
            out.push(*value);
        }
    }
    out
}

pub(crate) fn spot_delivery_ms(timestamp: &[f64], layer: &[f32]) -> Vec<f32> {
    let n = timestamp.len().min(layer.len());
    let mut out = vec![f32::NAN; n];
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for index in 0..n {
        if timestamp[index].is_finite() && layer[index].is_finite() {
            groups
                .entry(layer[index].round() as i64)
                .or_default()
                .push(index);
        }
    }
    for indices in groups.values_mut() {
        indices.sort_by(|&a, &b| {
            timestamp[a]
                .partial_cmp(&timestamp[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if indices.is_empty() {
            continue;
        }
        out[indices[0]] = timestamp[indices[0]] as f32;
        for pair in indices.windows(2) {
            out[pair[1]] = (timestamp[pair[1]] - timestamp[pair[0]]) as f32;
        }
    }
    out
}

struct Band {
    device: String,
    min: f64,
    max: f64,
    k: [f64; 4],
}

fn parse_sigma_bands(xml: &str) -> Vec<Band> {
    let mut bands = Vec::new();
    let mut device = String::new();
    let mut rest = xml;
    while let Some(start) = rest.find('<') {
        rest = &rest[start + 1..];
        let tag = rest.split('>').next().unwrap_or("");
        if tag.starts_with("device") {
            if let Some(name) = xml_attr(tag, "name") {
                device = name;
            }
        } else if tag.starts_with("beam_sigma_conversions")
            && xml_attr(tag, "in_units")
                .unwrap_or_default()
                .eq_ignore_ascii_case("MEV")
            && xml_attr(tag, "out_units")
                .unwrap_or_default()
                .eq_ignore_ascii_case("mm")
        {
            bands.push(Band {
                device: device.clone(),
                min: xml_attr(tag, "min_energy")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(f64::MIN),
                max: xml_attr(tag, "max_energy")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(f64::MAX),
                k: [
                    xml_attr(tag, "K0")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    xml_attr(tag, "K1")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    xml_attr(tag, "K2")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    xml_attr(tag, "K3")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                ],
            });
        }
    }
    bands
}

pub(crate) fn interlock_sigma_mm(xml: &str, device: &str, energy: f64) -> Option<f64> {
    expected_mm(&parse_sigma_bands(xml), device, energy)
}

fn expected_mm(bands: &[Band], device: &str, energy: f64) -> Option<f64> {
    let band = bands
        .iter()
        .find(|band| band.device == device && energy >= band.min && energy <= band.max)?;
    let e = energy;
    Some(band.k[0] + band.k[1] * e + band.k[2] * e * e + band.k[3] * e * e * e)
}

fn xml_attr(tag: &str, name: &str) -> Option<String> {
    let key = format!("{name}=");
    let at = tag.find(&key)? + key.len();
    let rest = tag[at..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &rest[1..];
    Some(body.split(quote).next()?.to_string())
}

fn header_base(name: &str) -> &str {
    name.split('.').next().unwrap_or(name)
}

fn timeslice_keep(name: &str) -> bool {
    timeline_column(name) || slice_column(name)
}

fn timeline_column(name: &str) -> bool {
    current_column(name) || signal_column(name)
}

fn current_column(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    lower == "layer_id"
        || lower == "rci_in_trigger"
        || lower == "r_beamok"
        || lower == "beam_on"
        || lower.contains("current")
        || lower.contains("primary_channel")
}

fn signal_column(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    if lower == "layer_id"
        || lower == "rci_in_trigger"
        || lower == "r_beamok"
        || lower == "beam_on"
        || lower == "c_x"
        || lower == "c_y"
        || lower.contains("confidence")
        || lower.contains("peak")
    {
        return true;
    }
    const NAMES: &[&str] = &[
        "amplifier_x_target",
        "amplifier_cmd_x",
        "amplifier_y_target",
        "amplifier_cmd_y",
        "amplifier_x_readback",
        "r_xi",
        "r_xv",
        "amplifier_readback_x",
        "amplifier_y_readback",
        "r_yi",
        "r_yv",
        "amplifier_readback_y",
        "r_tx2_probe_x",
        "field_c_x",
        "r_xb",
        "mag_field_x",
        "field_x",
        "b_field_x",
        "r_tx2_probe_y",
        "field_c_y",
        "r_yb",
        "mag_field_y",
        "field_y",
        "b_field_y",
    ];
    NAMES.contains(&lower.as_str())
}

fn slice_column(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    lower == "layer_id"
        || lower == "spot_no"
        || lower == "rci_in_trigger"
        || lower == "r_beamok"
        || lower == "beam_on"
        || lower.contains("position")
        || lower.contains("sigma")
        || lower.contains("fit_ok")
        || lower.contains("confidence")
        || lower.contains("error_code")
}

// ponytail: a few chunks, not a pool. Tens of timeslice files, not thousands.
fn read_sheets(files: &[Vec<u8>], keep: fn(&str) -> bool) -> Vec<Sheet> {
    if files.len() < 2 {
        return files
            .iter()
            .map(|bytes| read_sheet_where(bytes, keep))
            .collect();
    }
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, files.len());
    let chunk = files.len().div_ceil(workers);
    let mut sheets = Vec::with_capacity(files.len());
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for piece in files.chunks(chunk) {
            joins.push(scope.spawn(move || {
                piece
                    .iter()
                    .map(|bytes| read_sheet_where(bytes, keep))
                    .collect::<Vec<_>>()
            }));
        }
        for join in joins {
            sheets.extend(join.join().unwrap());
        }
    });
    sheets
}

enum Slot {
    Skip,
    Num(Vec<f32>),
    Wide(Vec<f64>),
    Clock(Vec<f64>),
}

enum Num {
    Nan,
    Value(f64),
    Std,
}

pub(crate) fn read_sheet(bytes: &[u8]) -> Sheet {
    read_sheet_where(bytes, |_| true)
}

/// Parse a device CSV without a `String` per cell.
///
/// ponytail: the decimal scanner handles the padded fixed-point form these logs
/// use. Scientific notation and non-finite tokens fall back to `str::parse`.
/// `keep` drops columns a caller will not read. Strip columns stay dropped.
fn read_sheet_where(bytes: &[u8], keep: impl Fn(&str) -> bool) -> Sheet {
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Sheet {
            num: BTreeMap::new(),
            wide: BTreeMap::new(),
        };
    };
    let rows = bytes.iter().filter(|byte| **byte == b'\n').count();
    let mut used = HashMap::<String, ()>::new();
    let mut names = Vec::new();
    let mut slots = Vec::new();
    let mut cursor = 0usize;
    while let Some((start, end, next)) = next_cell(header, cursor) {
        let name = cell_owned(header, start, end);
        let base = header_base(&name);
        // ponytail: strip columns are not a binned metric. Stop skipping them if one is.
        if name.is_empty() || base.to_ascii_lowercase().contains("strip") || !keep(&name) {
            names.push(None);
            slots.push(Slot::Skip);
        } else {
            let key = unique_header(&name, &mut used);
            let slot = if base == "datetime" {
                Slot::Clock(Vec::with_capacity(rows))
            } else if matches!(base, "timestamp" | "time_s" | "time_ns") {
                Slot::Wide(Vec::with_capacity(rows))
            } else {
                Slot::Num(Vec::with_capacity(rows))
            };
            names.push(Some(key));
            slots.push(slot);
        }
        if next <= cursor {
            break;
        }
        cursor = next;
    }
    let mut seen = vec![0u32; slots.len()];
    let mut row = 1u32;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut cursor = 0usize;
        let mut index = 0usize;
        while index < slots.len() {
            let Some((start, end, next)) = next_cell(line, cursor) else {
                break;
            };
            push_cell(&mut slots[index], line, start, end);
            seen[index] = row;
            index += 1;
            if next <= cursor {
                break;
            }
            cursor = next;
        }
        for (index, slot) in slots.iter_mut().enumerate() {
            if seen[index] != row {
                push_nan(slot);
            }
        }
        row = row.wrapping_add(1);
        if row == 0 {
            row = 1;
            seen.fill(0);
        }
    }
    let mut num = BTreeMap::new();
    let mut wide = BTreeMap::new();
    for (name, slot) in names.into_iter().zip(slots) {
        let Some(name) = name else {
            continue;
        };
        match slot {
            Slot::Skip => {}
            Slot::Num(values) => {
                num.insert(name, values);
            }
            Slot::Wide(values) | Slot::Clock(values) => {
                wide.insert(name, values);
            }
        }
    }
    Sheet { num, wide }
}

fn unique_header(name: &str, used: &mut HashMap<String, ()>) -> String {
    if !used.contains_key(name) {
        used.insert(name.to_string(), ());
        return name.to_string();
    }
    let mut n = 1;
    loop {
        let candidate = format!("{name}.{n}");
        if !used.contains_key(&candidate) {
            used.insert(candidate.clone(), ());
            return candidate;
        }
        n += 1;
    }
}

fn next_cell(line: &str, start: usize) -> Option<(usize, usize, usize)> {
    let bytes = line.as_bytes();
    if start > bytes.len() {
        return None;
    }
    if start == bytes.len() {
        if start > 0 && bytes[start - 1] == b',' {
            return Some((start, start, start + 1));
        }
        return None;
    }
    if bytes[start] == b'"' {
        let mut i = start + 1;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                i += 1;
                break;
            }
            i += 1;
        }
        let end = i;
        let next = if i < bytes.len() && bytes[i] == b',' {
            i + 1
        } else {
            i
        };
        return Some((start, end, next));
    }
    let mut i = start;
    while i < bytes.len() && bytes[i] != b',' {
        i += 1;
    }
    let next = if i < bytes.len() { i + 1 } else { i };
    Some((start, i, next))
}

fn cell_owned(line: &str, start: usize, end: usize) -> String {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        strip_quotes(cell)
    } else {
        cell.to_string()
    }
}

fn strip_quotes(cell: &str) -> String {
    cell.replace('"', "")
}

fn push_cell(slot: &mut Slot, line: &str, start: usize, end: usize) {
    match slot {
        Slot::Skip => {}
        Slot::Num(values) => values.push(cell_f32(line, start, end)),
        Slot::Wide(values) => values.push(cell_f64(line, start, end)),
        Slot::Clock(values) => values.push(cell_datetime(line, start, end)),
    }
}

fn push_nan(slot: &mut Slot) {
    match slot {
        Slot::Skip => {}
        Slot::Num(values) => values.push(f32::NAN),
        Slot::Wide(values) | Slot::Clock(values) => values.push(f64::NAN),
    }
}

fn cell_f32(line: &str, start: usize, end: usize) -> f32 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        let stripped = strip_quotes(cell);
        return f32_bytes(stripped.as_bytes());
    }
    f32_bytes(cell.as_bytes())
}

fn f32_bytes(bytes: &[u8]) -> f32 {
    match parse_decimal(bytes) {
        Num::Nan => f32::NAN,
        Num::Value(value) => value as f32,
        Num::Std => std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(f32::NAN),
    }
}

fn cell_f64(line: &str, start: usize, end: usize) -> f64 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        return strip_quotes(cell).trim().parse().unwrap_or(f64::NAN);
    }
    cell.trim().parse().unwrap_or(f64::NAN)
}

fn cell_datetime(line: &str, start: usize, end: usize) -> f64 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        return parse_datetime(&strip_quotes(cell)).unwrap_or(f64::NAN);
    }
    parse_datetime(cell).unwrap_or(f64::NAN)
}

fn parse_decimal(bytes: &[u8]) -> Num {
    let n = bytes.len();
    let mut i = 0;
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= n {
        return Num::Nan;
    }
    if bytes[i] != b'+' && bytes[i] != b'-' && bytes[i] != b'.' && !bytes[i].is_ascii_digit() {
        return Num::Std;
    }
    let neg = if bytes[i] == b'+' || bytes[i] == b'-' {
        let minus = bytes[i] == b'-';
        i += 1;
        minus
    } else {
        false
    };
    let mut value = 0u64;
    let mut saw = false;
    while i < n && bytes[i].is_ascii_digit() {
        let digit = u64::from(bytes[i] - b'0');
        if value > (u64::MAX - digit) / 10 {
            return Num::Std;
        }
        value = value * 10 + digit;
        saw = true;
        i += 1;
    }
    let mut scale = 1u64;
    if i < n && bytes[i] == b'.' {
        i += 1;
        while i < n && bytes[i].is_ascii_digit() {
            let digit = u64::from(bytes[i] - b'0');
            if scale > u64::MAX / 10 || value > (u64::MAX - digit) / 10 {
                return Num::Std;
            }
            value = value * 10 + digit;
            scale *= 10;
            saw = true;
            i += 1;
        }
    }
    if !saw {
        return Num::Nan;
    }
    if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
        return Num::Std;
    }
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i != n {
        return Num::Std;
    }
    let mut out = value as f64 / scale as f64;
    if neg {
        out = -out;
    }
    Num::Value(out)
}

pub(crate) fn parse_datetime(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (date, time) = text.split_once(['T', ' ']).unwrap_or((text, "00:00:00"));
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let mut clock = time.split(':');
    let hour: u32 = clock.next().unwrap_or("0").parse().unwrap_or(0);
    let minute: u32 = clock.next().unwrap_or("0").parse().unwrap_or(0);
    let second: f64 = clock.next().unwrap_or("0").parse().unwrap_or(0.0);
    let days = days_from_civil(year, month, day)?;
    Some(days as f64 * 86400.0 + f64::from(hour) * 3600.0 + f64::from(minute) * 60.0 + second)
}

pub(crate) fn days_from_civil(mut year: i32, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || day == 0 || day > 31 {
        return None;
    }
    if month <= 2 {
        year -= 1;
    }
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = (year - era * 400) as u32;
    let month_prime = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * month_prime + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(i64::from(era) * 146097 + i64::from(doe) - 719468)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_1970_is_unix_zero() {
        assert_eq!(days_from_civil(1970, 1, 1), Some(0));
        assert!(parse_datetime("1970-01-01 00:00:00").unwrap().abs() < 1e-6);
    }
    #[test]
    fn first_spot_in_a_layer_keeps_its_timestamp() {
        let ms = spot_delivery_ms(&[1000.0, 1004.0, 1010.0], &[1.0, 1.0, 1.0]);
        assert_eq!(ms, vec![1000.0, 4.0, 6.0]);
    }
    #[test]
    fn duplicate_headers_keep_the_first_column() {
        let sheet = read_sheet(b"energy,energy,charge_req\n70,999,1\n");
        assert_eq!(sheet.num["energy"], vec![70.0]);
        assert_eq!(sheet.num["charge_req"], vec![1.0]);
    }
    #[test]
    fn sheet_cells_keep_alignment() {
        let sheet = read_sheet(
            b"energy,ic1_strip_sum,energy,dose,timestamp,datetime\n\"70.5\",9,1.5e-3,2,826.0233764648437500,1970-01-01 00:00:00\n3\n",
        );
        assert!((sheet.num["energy"][0] - 70.5).abs() < 1e-4);
        assert!((sheet.num["energy.1"][0] - 0.0015).abs() < 1e-6);
        assert!((sheet.num["dose"][0] - 2.0).abs() < 1e-6);
        assert!((sheet.num["energy"][1] - 3.0).abs() < 1e-6);
        assert!(sheet.num["energy.1"][1].is_nan());
        assert!(sheet.num["dose"][1].is_nan());
        assert!(!sheet.num.contains_key("ic1_strip_sum"));
        let expected: f64 = "826.0233764648437500".parse().unwrap();
        assert!((sheet.wide["timestamp"][0] - expected).abs() <= 1e-6);
        assert!(sheet.wide["timestamp"][1].is_nan());
        assert!(sheet.wide["datetime"][0].abs() < 1e-6);
        assert!(sheet.wide["datetime"][1].is_nan());
    }
    #[test]
    fn padded_log_numbers_stay_finite() {
        let sheet = read_sheet(
            b"ENERGY,CHARGE_REQ,ic1_total_dose_spot\n250.0,0.005,     0.0055517933338228\n",
        );
        assert!((sheet.num["ENERGY"][0] - 250.0).abs() < 1e-3);
        assert!((sheet.num["ic1_total_dose_spot"][0] - 0.005551793).abs() < 1e-6);
    }

    #[test]
    fn a_rewritten_timeslice_is_read_again() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-ts-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("sess");
        let file = session
            .join("layer-0")
            .join("run-0")
            .join("timeslice_data_device_units.csv");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy\n70\n").unwrap();
        std::fs::write(&file, "r_ic1_current_dose,rci_in_trigger\n1,1\n").unwrap();
        let first = merged_timeslice(&root, "sess");
        assert_eq!(first["r_ic1_current_dose"], vec![1.0]);
        assert_eq!(
            merged_timeslice(&root, "sess")["r_ic1_current_dose"],
            vec![1.0]
        );
        std::fs::write(&file, "r_ic1_current_dose,rci_in_trigger\n4,1\n5,1\n").unwrap();
        assert_eq!(
            merged_timeslice(&root, "sess")["r_ic1_current_dose"],
            vec![4.0, 5.0]
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
