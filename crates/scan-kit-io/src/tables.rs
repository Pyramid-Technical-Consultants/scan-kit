//! Spot and timeslice columns shared by the analysis scenes and the tuners.
//!
//! ponytail: sigma expected values on the spot frame come from a string scan of
//! devices.xml. Timeslice isocenter position uses that same strip geometry, and
//! the spot-file strip-to-mm fit when the device file is missing. A repeated
//! `spot_no` starts that controller's own block, and the `.1` suffix matches pandas.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use super::discover;
use scan_kit_core::{
    column_scale_factor, dose_error_pct, dose_ratio_pct, g2_ic2_mm, linear_fit, remap,
    remap_g2_raw, remap_g3_raw, scale_column,
};

#[derive(Clone)]
pub(crate) struct Sheet {
    pub(crate) num: BTreeMap<String, Vec<f32>>,
    pub(crate) wide: BTreeMap<String, Vec<f64>>,
    /// Header block of each kept column. A new block starts at each repeated
    /// `spot_no`, so IC1, IC2, IC3, and the other controllers do not share a clock.
    pub(crate) block: BTreeMap<String, u32>,
}

fn empty_sheet() -> Sheet {
    Sheet {
        num: BTreeMap::new(),
        wide: BTreeMap::new(),
        block: BTreeMap::new(),
    }
}

/// Rows published in one slice of a long file. A short file is one slice.
pub const SLICE_ROWS: usize = 4_096;

/// How a poll limits one session. Other sessions load in full.
#[derive(Clone, Copy)]
pub enum SliceTake {
    /// This many leading timeslice files, then this many rows of the next file.
    Timeslice { files: usize, tail_rows: usize },
    /// This many rows of `spot_data.csv`.
    Spot { rows: usize },
}

pub struct SessionPieces {
    pub timeslice: bool,
    pub sizes: Vec<u64>,
}

/// File sizes for one session. Timeslice sessions list each file. Anything else is the spot file.
pub fn session_pieces(root: &Path, session: &str) -> SessionPieces {
    let dir = discover::session_directory(root, session);
    if dir.is_dir() {
        let paths = discover::list_timeslice_paths(&dir);
        if !paths.is_empty() {
            return SessionPieces {
                timeslice: true,
                sizes: paths.iter().map(|(_, path)| file_len(path)).collect(),
            };
        }
    }
    SessionPieces {
        timeslice: false,
        sizes: vec![file_len(&dir.join("spot_data.csv"))],
    }
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or(1)
        .max(1)
}

struct BoundSlice {
    session: String,
    take: SliceTake,
    tail_done: bool,
}

fn bound_slice() -> &'static Mutex<Option<BoundSlice>> {
    static BOUND: OnceLock<Mutex<Option<BoundSlice>>> = OnceLock::new();
    BOUND.get_or_init(|| Mutex::new(None))
}

pub fn bind_slice(session: &str, take: SliceTake) {
    *bound_slice().lock().unwrap_or_else(|err| err.into_inner()) = Some(BoundSlice {
        session: session.to_owned(),
        take,
        tail_done: false,
    });
}

pub fn clear_slice() {
    *bound_slice().lock().unwrap_or_else(|err| err.into_inner()) = None;
}

pub fn slice_tail_done() -> bool {
    bound_slice()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .as_ref()
        .is_some_and(|bound| bound.tail_done)
}

pub(crate) fn slice_is_bound() -> bool {
    bound_slice()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .is_some()
}

fn slice_for(session: &str) -> Option<SliceTake> {
    bound_slice()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .as_ref()
        .filter(|bound| bound.session == session)
        .map(|bound| bound.take)
}

thread_local! {
    static PUBLISH_TAIL: Cell<bool> = const { Cell::new(true) };
    #[cfg(test)]
    static DISK_READS: Cell<u32> = const { Cell::new(0) };
}

fn note_disk_read(_path: &Path) {
    #[cfg(test)]
    DISK_READS.with(|reads| reads.set(reads.get().saturating_add(1)));
}

/// A finished file parsed beside the partial file must not publish end-of-file.
struct QuietTail {
    restore: bool,
}

impl QuietTail {
    fn suppress() -> Self {
        let restore = PUBLISH_TAIL.with(Cell::get);
        PUBLISH_TAIL.with(|flag| flag.set(false));
        Self { restore }
    }
}

impl Drop for QuietTail {
    fn drop(&mut self) {
        if self.restore {
            PUBLISH_TAIL.with(|flag| flag.set(true));
        }
    }
}

/// Publish only for the bound session. A parallel load of another session
/// finishes its own file and must not move this flag.
fn set_tail_done(done: bool, belongs: impl FnOnce(&str) -> bool) {
    if !PUBLISH_TAIL.with(Cell::get) {
        return;
    }
    if let Some(bound) = bound_slice()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .as_mut()
    {
        if belongs(&bound.session) {
            bound.tail_done = done;
        }
    }
}

fn path_has_session(path: &Path, session: &str) -> bool {
    path.components().any(|part| part.as_os_str() == session)
}

pub(crate) type Table = std::sync::Arc<BTreeMap<String, Vec<f32>>>;

pub(crate) fn map_sessions<T: Send>(
    session_ids: &[String],
    load: impl Fn(&str) -> T + Sync,
) -> Vec<T> {
    if session_ids.len() < 2 {
        return session_ids.iter().map(|id| load(id)).collect();
    }
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(session_ids.len());
        for session in session_ids {
            joins.push(scope.spawn(|| load(session)));
        }
        joins.into_iter().map(|join| join.join().unwrap()).collect()
    })
}

pub(crate) fn spot_table(root: &Path, session: &str) -> Table {
    session_columns(root, session, Grain::Spot, &["ic1_dose"])
}

/// Spot table for configuration tuning.
///
/// Analysis sigma stays in `ic1_sig_x`. The raw column is `ic1_sig_x_raw`.
pub(crate) fn tune_spot_table(root: &Path, session: &str) -> Table {
    session_columns(root, session, Grain::Spot, &["ic1_sig_x_raw"])
}

pub(crate) fn slice_table(root: &Path, session: &str) -> Table {
    session_columns(root, session, Grain::Sample, &["ic1_x"])
}

pub(crate) fn timeslice_metric(root: &Path, session: &str, metric: &str) -> Table {
    session_columns(
        root,
        session,
        metric_grain(metric),
        &[metric_column(metric)],
    )
}

/// Current, confidence, peak, field, and amplifier columns from one timeslice read.
pub(crate) fn timeslice_signals(root: &Path, session: &str) -> Table {
    // The signals producer also fills current, so this is one family read.
    session_columns(
        root,
        session,
        Grain::Sample,
        &["ic1_x_confidence", "ic1_x_peak", "amp_x", "field_x"],
    )
}

fn produce_signals(root: &Path, session: &str) -> Table {
    cached_timeslice(root, session, "signals", false, Family::Signals, |frames| {
        let mut out = current_from(
            &frames.sheets,
            &frames.energies,
            &frames.layers,
            frames.prior,
        );
        for metric in [
            "fit_confidence",
            "peak_amplitude",
            "amplifier_error",
            "probe_field",
        ] {
            for (key, values) in signal_from(
                &frames.sheets,
                &frames.energies,
                &frames.layers,
                metric,
                frames.prior,
            ) {
                out.entry(key).or_insert(values);
            }
        }
        out
    })
}

/// Which row clock a column belongs to.
///
/// Spot and sample rows are not the same length. Current ratios are one row
/// per timeslice file. Chamber position is a separate frame so `ic1_x` on the
/// isocenter frame stays isocenter millimetres.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Grain {
    Spot,
    SpotChamber,
    Sample,
    SampleChamber,
    Layer,
}

const FILL_CURRENT: u8 = 1;
const FILL_SIGNALS: u8 = 2;
const FILL_GEOMETRY: u8 = 4;
const FILL_RATIO: u8 = 8;
const FILL_SPOT: u8 = 16;

struct HeldColumns {
    span: (u8, usize, usize),
    parts: Vec<(u8, Table)>,
    merged: Table,
    /// Spot delivery time was included. A later ask can reuse this table.
    points: bool,
}

fn column_store() -> &'static Mutex<HashMap<(PathBuf, Grain), HeldColumns>> {
    static STORE: OnceLock<Mutex<HashMap<(PathBuf, Grain), HeldColumns>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn span_of(session: &str) -> (u8, usize, usize) {
    match slice_for(session) {
        Some(SliceTake::Timeslice { files, tail_rows }) => (1, files, tail_rows),
        Some(SliceTake::Spot { rows }) => (2, rows, 0),
        None => (0, 0, 0),
    }
}

fn metric_grain(metric: &str) -> Grain {
    if metric == "current_ratio" {
        Grain::Layer
    } else {
        Grain::Sample
    }
}

fn metric_column(metric: &str) -> &'static str {
    match metric {
        "current_ratio" => "ic21_ratio",
        "fit_confidence" => "ic1_x_confidence",
        "peak_amplitude" => "ic1_x_peak",
        "amplifier_error" => "amp_x",
        "probe_field" => "field_x",
        _ => "ic1_current",
    }
}

fn sample_fill(name: &str) -> u8 {
    if matches!(name, "ic1_current" | "ic2_current" | "ic3_current") {
        FILL_CURRENT
    } else if matches!(
        name,
        "ic1_x"
            | "ic1_y"
            | "ic2_x"
            | "ic2_y"
            | "plan_x"
            | "plan_y"
            | "ic1_x_err"
            | "ic1_y_err"
            | "ic2_x_err"
            | "ic2_y_err"
            | "ic1_sig_x"
            | "ic1_sig_y"
            | "ic2_sig_x"
            | "ic2_sig_y"
            | "ic1_sig_x_err"
            | "ic1_sig_y_err"
            | "ic2_sig_x_err"
            | "ic2_sig_y_err"
            | "ic12_x_diff"
            | "ic12_y_diff"
    ) {
        FILL_GEOMETRY
    } else if name.contains("confidence")
        || name.contains("peak")
        || name.starts_with("amp_")
        || name.starts_with("field_")
    {
        FILL_SIGNALS
    } else {
        FILL_CURRENT
    }
}

fn fills_for(grain: Grain, names: &[&str]) -> u8 {
    match grain {
        Grain::Layer => FILL_RATIO,
        Grain::Spot | Grain::SpotChamber => FILL_SPOT,
        Grain::Sample | Grain::SampleChamber => {
            let mut bits = 0;
            for name in names {
                bits |= sample_fill(name);
            }
            if bits == 0 {
                FILL_CURRENT
            } else {
                bits
            }
        }
    }
}

fn wants_point_time(names: &[&str]) -> bool {
    names
        .iter()
        .any(|name| matches!(*name, "spot_time" | "beam_on_time" | "overhead_time"))
}

fn produce_fill(root: &Path, session: &str, grain: Grain, fill: u8, points: bool) -> Table {
    match fill {
        FILL_CURRENT => load_timeslice(root, session, "ic_current"),
        FILL_SIGNALS => produce_signals(root, session),
        FILL_GEOMETRY => load_slice_metric(
            root,
            session,
            "position_error",
            grain == Grain::SampleChamber,
        ),
        FILL_RATIO => load_timeslice(root, session, "current_ratio"),
        FILL_SPOT => load_spot(root, session, grain == Grain::SpotChamber, points, false),
        _ => Arc::new(BTreeMap::new()),
    }
}

fn absorb(map: &mut BTreeMap<String, Vec<f32>>, part: &BTreeMap<String, Vec<f32>>) {
    let width = map.get("energy").map(Vec::len);
    for (key, values) in part {
        if let Some(width) = width {
            if values.len() != width {
                continue;
            }
        }
        map.entry(key.clone()).or_insert_with(|| values.clone());
    }
}

/// Columns for one session grain.
///
/// A repeat ask returns the same table. A new column runs only its producer
/// and is stored beside the columns already filled for this slice.
pub(crate) fn session_columns(root: &Path, session: &str, grain: Grain, names: &[&str]) -> Table {
    let need = fills_for(grain, names);
    let points = matches!(grain, Grain::Spot | Grain::SpotChamber) && wants_point_time(names);
    let span = span_of(session);
    let dir = discover::session_directory(root, session);
    let key = (dir, grain);
    let mut produced = Vec::new();
    {
        let store = column_store().lock().unwrap_or_else(|err| err.into_inner());
        if let Some(held) = store.get(&key) {
            let have = held.parts.iter().fold(0u8, |bits, (fill, _)| bits | fill);
            if held.span == span && (have & need) == need && (!points || held.points) {
                return Arc::clone(&held.merged);
            }
        }
    }
    for fill in [
        FILL_CURRENT,
        FILL_SIGNALS,
        FILL_GEOMETRY,
        FILL_RATIO,
        FILL_SPOT,
    ] {
        if need & fill != 0 {
            let ask_points = points && fill == FILL_SPOT;
            produced.push((fill, produce_fill(root, session, grain, fill, ask_points)));
        }
    }
    let mut store = column_store().lock().unwrap_or_else(|err| err.into_inner());
    let held = store.entry(key).or_insert_with(|| HeldColumns {
        span,
        parts: Vec::new(),
        merged: Arc::new(BTreeMap::new()),
        points: false,
    });
    if held.span != span {
        held.span = span;
        held.parts.clear();
        held.merged = Arc::new(BTreeMap::new());
        held.points = false;
    }
    if points {
        held.points = true;
    }
    let mut changed = held.merged.as_ref().is_empty();
    for (fill, part) in produced {
        if let Some(slot) = held.parts.iter_mut().find(|(have, _)| *have == fill) {
            if Arc::ptr_eq(&slot.1, &part) {
                continue;
            }
            slot.1 = part;
        } else {
            held.parts.push((fill, part));
        }
        changed = true;
    }
    if changed {
        if held.parts.len() == 1 {
            held.merged = Arc::clone(&held.parts[0].1);
        } else {
            let mut map = BTreeMap::new();
            for (_, part) in &held.parts {
                absorb(&mut map, part);
            }
            held.merged = Arc::new(map);
        }
    }
    Arc::clone(&held.merged)
}

// ponytail: keyed by spot/map length and mtime. devices.xml and per-layer
// point-time files stay stale until the spot csv changes. Past 16 entries the
// map is cleared. A same-size rewrite in the same timestamp tick is missed.
const SPOT_CACHE_CAP: usize = 16;

#[derive(Hash, PartialEq, Eq)]
struct SpotCacheKey {
    dir: PathBuf,
    chamber: bool,
    spot_stamp: u128,
    map_stamp: u128,
    points: bool,
    raw_sigma: bool,
}

fn spot_cache() -> &'static std::sync::Mutex<HashMap<SpotCacheKey, Table>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<SpotCacheKey, Table>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Parsed timeslice frames shared by every metric.
///
/// ponytail: four sessions stay parsed. Past that one entry is dropped, so a
/// session that is still loading is not thrown out with the rest. The files
/// are the expensive part; the derived tables below are the cheap repeat.
const FRAME_CACHE_CAP: usize = 4;
const TABLE_CACHE_CAP: usize = 32;

struct Frames {
    energies: Vec<f32>,
    layers: Vec<i64>,
    sheets: Vec<Arc<Sheet>>,
    map: Arc<Sheet>,
    /// `(complete files, rows of the next file)` already folded into the
    /// previous partial table. `None` builds every loaded row.
    prior: Option<(usize, usize)>,
}

fn evict_one<K, V>(map: &mut HashMap<K, V>)
where
    K: Eq + std::hash::Hash,
{
    let mut first = true;
    map.retain(|_, _| {
        let drop_this = first;
        first = false;
        !drop_this
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Family {
    /// Beam current, the gate, and `layer_id`. The default binned picture.
    Current,
    Signals,
    Geometry,
    All,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct FrameKey {
    dir: PathBuf,
    map_stamp: u128,
    slice_stamp: u128,
    family: Family,
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

fn table_cache() -> &'static Mutex<HashMap<TableKey, Table>> {
    static CACHE: OnceLock<Mutex<HashMap<TableKey, Table>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn frame_key(root: &Path, session: &str, family: Family) -> FrameKey {
    let dir = discover::session_directory(root, session);
    FrameKey {
        map_stamp: discover::meta_stamp(&dir.join("input_map.csv")),
        slice_stamp: discover::timeslice_stamp(&dir),
        family,
        dir,
    }
}

fn cached_timeslice(
    root: &Path,
    session: &str,
    kind: &str,
    spot: bool,
    family: Family,
    build: impl FnOnce(&Frames) -> BTreeMap<String, Vec<f32>>,
) -> Table {
    let frames = frame_key(root, session, family);
    let key = TableKey {
        spot_stamp: if spot {
            discover::meta_stamp(&frames.dir.join("spot_data.csv"))
        } else {
            0
        },
        kind: kind.to_string(),
        frames,
    };
    let partial = match slice_for(session) {
        Some(SliceTake::Timeslice { files, tail_rows }) => Some((files, tail_rows)),
        _ => None,
    };
    if partial.is_none() {
        if let Some(hit) = table_cache()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&key)
        {
            return Arc::clone(hit);
        }
    }
    let dir = discover::session_directory(root, session);
    if let Some((files, tail_rows)) = partial {
        match take_grown(&dir, kind, files, tail_rows) {
            GrownHit::Same(table) => return table,
            GrownHit::More(previous) => {
                let _ = take_full_table();
                let extra = open_frames(root, session, family)
                    .map(|mut frames| {
                        if let Some(inner) = Arc::get_mut(&mut frames) {
                            inner.prior = Some((previous.files, previous.tail_rows));
                        }
                        build(&frames)
                    })
                    .unwrap_or_default();
                let table = if take_full_table() {
                    Arc::new(extra)
                } else {
                    let mut table = previous.table;
                    let grown = Arc::make_mut(&mut table);
                    extend_columns(grown, extra);
                    // The tail's clock starts at 0. The combined table counts from its first row.
                    restamp_sample_time(grown);
                    table
                };
                store_growth(&dir, kind, files, tail_rows, Arc::clone(&table));
                return table;
            }
            GrownHit::Miss => {}
        }
    }
    let _ = take_full_table();
    let table = Arc::new(
        open_frames(root, session, family)
            .map(|frames| build(&frames))
            .unwrap_or_default(),
    );
    let _ = take_full_table();
    if let Some((files, tail_rows)) = partial {
        store_growth(&dir, kind, files, tail_rows, Arc::clone(&table));
        return table;
    }
    let mut cache = table_cache().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= TABLE_CACHE_CAP {
        evict_one(&mut cache);
    }
    cache.insert(key, Arc::clone(&table));
    table
}

/// Rows already published for this slice. The next poll appends the tail.
struct Grown {
    files: usize,
    tail_rows: usize,
    table: Table,
}

fn growth_cache() -> &'static Mutex<HashMap<(PathBuf, String), Grown>> {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, String), Grown>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

thread_local! {
    static FULL_TABLE: Cell<bool> = const { Cell::new(false) };
}

fn mark_full_table() {
    FULL_TABLE.with(|flag| flag.set(true));
}

fn take_full_table() -> bool {
    FULL_TABLE.with(|flag| flag.replace(false))
}

enum GrownHit {
    Same(Table),
    More(Grown),
    Miss,
}

fn take_grown(dir: &Path, kind: &str, files: usize, tail_rows: usize) -> GrownHit {
    let key = (dir.to_path_buf(), kind.to_string());
    let mut cache = growth_cache().lock().unwrap_or_else(|err| err.into_inner());
    let Some(have) = cache.get(&key) else {
        return GrownHit::Miss;
    };
    if have.files == files && have.tail_rows == tail_rows {
        return GrownHit::Same(Arc::clone(&have.table));
    }
    if files < have.files || (files == have.files && tail_rows < have.tail_rows) {
        cache.remove(&key);
        return GrownHit::Miss;
    }
    cache
        .remove(&key)
        .map(GrownHit::More)
        .unwrap_or(GrownHit::Miss)
}

fn store_growth(dir: &Path, kind: &str, files: usize, tail_rows: usize, table: Table) {
    let mut cache = growth_cache().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= TABLE_CACHE_CAP {
        evict_one(&mut cache);
    }
    cache.insert(
        (dir.to_path_buf(), kind.to_string()),
        Grown {
            files,
            tail_rows,
            table,
        },
    );
}

fn extend_columns(base: &mut BTreeMap<String, Vec<f32>>, extra: BTreeMap<String, Vec<f32>>) {
    let width = base.get("energy").map(Vec::len).unwrap_or(0);
    let extra_energy = extra.get("energy").map(Vec::len).unwrap_or(0);
    for (key, values) in extra {
        if let Some(have) = base.get_mut(&key) {
            have.extend(values);
        } else if extra_energy > 0 && values.len() == extra_energy {
            let mut padded = vec![f32::NAN; width];
            padded.extend(values);
            base.insert(key, padded);
        } else {
            base.insert(key, values);
        }
    }
}

/// First sample of `index` that is not already in the previous partial table.
fn row_origin(index: usize, prior: Option<(usize, usize)>) -> Option<usize> {
    let Some((files, tail)) = prior else {
        return Some(0);
    };
    if index < files {
        None
    } else if index == files {
        Some(tail)
    } else {
        Some(0)
    }
}

fn open_frames(root: &Path, session: &str, family: Family) -> Option<Arc<Frames>> {
    if let Some(SliceTake::Timeslice { files, tail_rows }) = slice_for(session) {
        return frames_limited(root, session, family, files, tail_rows);
    }
    let key = frame_key(root, session, family);
    if let Some(hit) = frame_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
    {
        return Some(Arc::clone(hit));
    }
    let map = map_sheet(root, session);
    let energies = unique_seen(column(&map, "energy", &[]));
    let (kind, keep) = family_parse(family);
    let (layers, sheets) = if key.dir.is_dir() {
        let listed = discover::list_timeslice_paths(&key.dir);
        if listed.is_empty() {
            return None;
        }
        let layers = listed.iter().map(|(index, _)| *index).collect();
        (layers, parse_paths(&listed, kind, keep))
    } else {
        let listed = discover::read_timeslice_frames(&key.dir);
        if listed.is_empty() {
            return None;
        }
        let layers = listed.iter().map(|(index, _)| *index).collect();
        let files: Vec<Vec<u8>> = listed.into_iter().map(|(_, bytes)| bytes).collect();
        (layers, read_sheets(&files, keep))
    };
    let frames = Arc::new(Frames {
        energies,
        layers,
        sheets,
        map,
        prior: None,
    });
    let mut cache = frame_cache().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= FRAME_CACHE_CAP {
        evict_one(&mut cache);
    }
    cache.insert(key, Arc::clone(&frames));
    Some(frames)
}

fn frames_limited(
    root: &Path,
    session: &str,
    family: Family,
    files: usize,
    tail_rows: usize,
) -> Option<Arc<Frames>> {
    let dir = discover::session_directory(root, session);
    if !dir.is_dir() {
        return None;
    }
    let listed = discover::list_timeslice_paths(&dir);
    if listed.is_empty() {
        set_tail_done(true, |name| name == session);
        return None;
    }
    let map = map_sheet(root, session);
    let energies = unique_seen(column(&map, "energy", &[]));
    let (kind, keep) = family_parse(family);
    let full = files.min(listed.len());
    let tail = (full < listed.len() && tail_rows > 0).then(|| listed[full].clone());
    let no_tail = tail.is_none();
    let (sheets, layers) = std::thread::scope(|scope| {
        let tail_job = tail.map(|(index, path)| {
            (
                index,
                scope.spawn(move || sheet_rows(&path, kind, keep, tail_rows)),
            )
        });
        let mut sheets = if full == 0 {
            Vec::new()
        } else {
            parse_paths(&listed[..full], kind, keep)
        };
        let mut layers: Vec<i64> = listed.iter().take(full).map(|(index, _)| *index).collect();
        if let Some((index, job)) = tail_job {
            sheets.push(job.join().unwrap());
            layers.push(index);
        }
        (sheets, layers)
    });
    if no_tail && full >= listed.len() {
        set_tail_done(true, |name| name == session);
    }
    Some(Arc::new(Frames {
        energies,
        layers,
        sheets,
        map,
        prior: None,
    }))
}

fn map_sheet(root: &Path, session: &str) -> Arc<Sheet> {
    let path = discover::session_directory(root, session).join("input_map.csv");
    if path.is_file() {
        return sheet_all(&path, Keep::Map, |_| true);
    }
    Arc::new(
        discover::read_session_file(root, session, "input_map.csv")
            .map(|bytes| read_sheet(&bytes))
            .unwrap_or_else(empty_sheet),
    )
}

pub(crate) fn merged_timeslice(root: &Path, session: &str) -> Table {
    cached_timeslice(root, session, "merged", false, Family::All, |frames| {
        let mut merged: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (index, sheet) in frames.sheets.iter().enumerate() {
            let Some(start) = row_origin(index, frames.prior) else {
                continue;
            };
            for (name, values) in &sheet.num {
                let slice = values.get(start..).unwrap_or(&[]);
                if slice.is_empty() {
                    continue;
                }
                merged
                    .entry(name.clone())
                    .or_default()
                    .extend_from_slice(slice);
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
) -> Table {
    let dir = discover::session_directory(root, session);
    let spot_path = dir.join("spot_data.csv");
    let map_path = dir.join("input_map.csv");
    let rows = match slice_for(session) {
        Some(SliceTake::Spot { rows }) => Some(rows),
        _ => None,
    };
    let key = (rows.is_none() && spot_path.is_file() && map_path.is_file()).then(|| SpotCacheKey {
        spot_stamp: discover::meta_stamp(&spot_path),
        map_stamp: discover::meta_stamp(&map_path),
        dir,
        chamber,
        points,
        raw_sigma,
    });
    if let Some(key) = &key {
        let cache = spot_cache().lock().unwrap_or_else(|err| err.into_inner());
        if let Some(hit) = cache.get(key) {
            return Arc::clone(hit);
        }
    }
    let table = Arc::new(build_spot(root, session, chamber, points, raw_sigma, rows));
    if let Some(key) = key {
        let mut cache = spot_cache().lock().unwrap_or_else(|err| err.into_inner());
        if cache.len() >= SPOT_CACHE_CAP {
            evict_one(&mut cache);
        }
        cache.insert(key, Arc::clone(&table));
    }
    table
}

fn build_spot(
    root: &Path,
    session: &str,
    chamber: bool,
    points: bool,
    raw_sigma: bool,
    rows: Option<usize>,
) -> BTreeMap<String, Vec<f32>> {
    let dir = discover::session_directory(root, session);
    let map_path = dir.join("input_map.csv");
    let map = if map_path.is_file() {
        sheet_all(&map_path, Keep::Map, |_| true)
    } else if let Some(bytes) = discover::read_session_file(root, session, "input_map.csv") {
        Arc::new(read_sheet(&bytes))
    } else {
        if rows.is_some() {
            set_tail_done(true, |name| name == session);
        }
        return BTreeMap::new();
    };
    let path = dir.join("spot_data.csv");
    let spot = if let Some(rows) = rows {
        sheet_rows(&path, Keep::Spot, spot_keep, rows)
    } else if path.is_file() {
        sheet_all(&path, Keep::Spot, spot_keep)
    } else {
        let Some(spot_bytes) = discover::read_session_file(root, session, "spot_data.csv") else {
            return BTreeMap::new();
        };
        Arc::new(read_sheet_where(&spot_bytes, spot_keep))
    };
    // ponytail: spot_delivery_ms sorts every timestamp in a layer, so a new row
    // can change a neighbor. The derived columns are rebuilt from this parsed
    // sheet instead of a suffix. The sheet itself is not read from disk again.
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
    let mut doses = Vec::new();
    for (concept, key) in [
        ("ic1_total_dose", "ic1_dose"),
        ("ic2_total_dose", "ic2_dose"),
        ("ic3_total_dose", "ic3_dose"),
    ] {
        let values = column(&spot, concept, &[key]);
        if values.len() >= n {
            doses.push((key, values));
        }
    }
    let (mask_pos, mm_pos) = spot_positions(&spot, chamber, n);
    let sigma = spot_sigma(&spot, raw_sigma || chamber, n);
    let mut mask_cols: Vec<&[f32]> = vec![energy];
    if target.len() >= n {
        mask_cols.push(target);
    }
    if plan_x.len() >= n {
        mask_cols.push(plan_x);
    }
    if plan_y.len() >= n {
        mask_cols.push(plan_y);
    }
    for (_, values) in &doses {
        mask_cols.push(*values);
    }
    for values in &mask_pos {
        mask_cols.push(*values);
    }
    let keep = keep_mask(&mask_cols, n);
    if !keep.iter().any(|row| *row) {
        return BTreeMap::new();
    }
    let mut table = BTreeMap::new();
    table.insert("energy".to_string(), take_kept(energy, &keep));
    if target.len() >= n {
        table.insert("target_mu".to_string(), take_kept(target, &keep));
    }
    if plan_x.len() >= n && plan_y.len() >= n {
        let x = take_kept(plan_x, &keep);
        let y = take_kept(plan_y, &keep);
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
    let mut derived = Vec::new();
    for (out, left, right, combine) in [
        (
            "ic1_dose_err_pct",
            "ic1_dose",
            "target_mu",
            dose_error_pct as fn(&[f32], &[f32]) -> Vec<f32>,
        ),
        ("ic2_dose_err_pct", "ic2_dose", "target_mu", dose_error_pct),
        ("ic3_dose_err_pct", "ic3_dose", "target_mu", dose_error_pct),
    ] {
        if let Some(values) = paired(&table, left, right, combine) {
            derived.push((out, values));
        }
    }
    if let Some(values) = paired(&table, "ic2_dose", "ic1_dose", dose_ratio_pct) {
        derived.push(("ic21_ratio", values));
        if let Some(values) = paired(&table, "ic3_dose", "ic1_dose", dose_ratio_pct) {
            derived.push(("ic31_ratio", values));
        }
        if let Some(values) = paired(&table, "ic3_dose", "ic2_dose", dose_ratio_pct) {
            derived.push(("ic32_ratio", values));
        }
    }
    for (key, values) in derived {
        table.insert(key.to_string(), values);
    }
    let mm = mm_pos.map(|column| column.map(|column| take_kept(column.as_slice(), &keep)));
    store_positions(&mut table, mm);
    for (key, values) in sigma {
        table.insert(
            key.to_string(),
            map_kept(values, &keep, |value| value * 2.0),
        );
    }
    for (ic, axis, key) in [
        ("ic1", "x", "ic1_sig_x"),
        ("ic1", "y", "ic1_sig_y"),
        ("ic2", "x", "ic2_sig_x"),
        ("ic2", "y", "ic2_sig_y"),
    ] {
        let Some(values) = column_suffix(&spot, ic, axis, &["spot_sigma_raw"]) else {
            continue;
        };
        if values.len() < n {
            continue;
        }
        table.insert(
            format!("{key}_raw"),
            map_kept(&values[..n], &keep, |value| value * 2.0),
        );
    }
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
    if let Some(layer) = spot.num.get("layer_id").filter(|values| values.len() >= n) {
        let kept = take_kept(layer, &keep);
        if kept.len() == table.get("energy").map(Vec::len).unwrap_or(0) {
            table.insert("layer_id".to_string(), kept);
        }
    }
    stamp_spot_clock(&mut table, &spot, n, &keep);
    table
}

/// Seconds from the first finite wall stamp. A missing stamp leaves the column out.
fn stamp_spot_clock(table: &mut BTreeMap<String, Vec<f32>>, spot: &Sheet, n: usize, keep: &[bool]) {
    let rows = table.get("energy").map(Vec::len).unwrap_or(0);
    if rows == 0 {
        return;
    }
    let Some(stamps) = wall_time(spot, n) else {
        return;
    };
    let kept = take_kept(&stamps, keep);
    if kept.len() != rows {
        return;
    }
    if let Some(elapsed) = elapsed_seconds(&kept) {
        table.insert("time_s".to_string(), elapsed);
    }
}

fn elapsed_seconds(stamps: &[f64]) -> Option<Vec<f32>> {
    let origin = stamps.iter().copied().find(|value| value.is_finite())?;
    Some(
        stamps
            .iter()
            .map(|value| {
                if value.is_finite() {
                    (*value - origin) as f32
                } else {
                    f32::NAN
                }
            })
            .collect(),
    )
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

fn map_kept(values: &[f32], keep: &[bool], mut map: impl FnMut(f32) -> f32) -> Vec<f32> {
    values
        .iter()
        .zip(keep)
        .filter(|(_, keep)| **keep)
        .map(|(value, _)| map(*value))
        .collect()
}

fn paired(
    table: &BTreeMap<String, Vec<f32>>,
    left: &str,
    right: &str,
    combine: fn(&[f32], &[f32]) -> Vec<f32>,
) -> Option<Vec<f32>> {
    Some(combine(table.get(left)?, table.get(right)?))
}

enum Measured<'a> {
    Same(&'a [f32]),
    Owned(Vec<f32>),
}

impl Measured<'_> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Measured::Same(values) => values,
            Measured::Owned(values) => values,
        }
    }
}

fn store_positions(table: &mut BTreeMap<String, Vec<f32>>, mm: [Option<Vec<f32>>; 4]) {
    let keys = [
        ("ic1_x", "ic1_x_err", "plan_x"),
        ("ic1_y", "ic1_y_err", "plan_y"),
        ("ic2_x", "ic2_x_err", "plan_x"),
        ("ic2_y", "ic2_y_err", "plan_y"),
    ];
    let mut errors = Vec::new();
    for ((_, err_key, plan_key), measured) in keys.into_iter().zip(mm.iter()) {
        let Some(measured) = measured else {
            continue;
        };
        let Some(plan) = table.get(plan_key) else {
            continue;
        };
        if plan.len() == measured.len() {
            errors.push((
                err_key,
                measured
                    .iter()
                    .zip(plan)
                    .map(|(measured, plan)| measured - plan)
                    .collect::<Vec<_>>(),
            ));
        }
    }
    for (key, values) in errors {
        table.insert(key.to_string(), values);
    }
    let x_diff = match (mm[0].as_deref(), mm[2].as_deref()) {
        (Some(ic1), Some(ic2)) => column_diff(ic1, ic2),
        _ => Vec::new(),
    };
    let y_diff = match (mm[1].as_deref(), mm[3].as_deref()) {
        (Some(ic1), Some(ic2)) => column_diff(ic1, ic2),
        _ => Vec::new(),
    };
    for ((pos_key, _, _), measured) in keys.into_iter().zip(mm) {
        let Some(measured) = measured else {
            continue;
        };
        table.insert(pos_key.to_string(), measured);
    }
    if !x_diff.is_empty() {
        table.insert("ic12_x_diff".to_string(), x_diff);
    }
    if !y_diff.is_empty() {
        table.insert("ic12_y_diff".to_string(), y_diff);
    }
}

/// Columns that participate in the row mask, plus mm positions when the frame resolves.
fn spot_positions<'a>(
    spot: &'a Sheet,
    chamber: bool,
    n: usize,
) -> (Vec<&'a [f32]>, [Option<Measured<'a>>; 4]) {
    if chamber {
        if let Some(raw) = four_slices(spot, "spot_position_raw", n) {
            let mm = [
                Some(Measured::Owned(g3_mm(raw[0], false))),
                Some(Measured::Owned(g3_mm(raw[1], true))),
                Some(Measured::Owned(g3_mm(raw[2], true))),
                Some(Measured::Owned(g3_mm(raw[3], false))),
            ];
            return (raw.to_vec(), mm);
        }
        if let Some(raw) = four_slices(spot, "spot_raw", n) {
            let ic1_x = raw[0].iter().copied().map(remap_g2_raw).collect::<Vec<_>>();
            let ic1_y = raw[1].iter().copied().map(remap_g2_raw).collect::<Vec<_>>();
            let ic2_x = g2_ic2_mm(&ic1_x, raw[2]);
            let ic2_y = g2_ic2_mm(&ic1_y, raw[3]);
            return (
                raw.to_vec(),
                [
                    Some(Measured::Owned(ic1_x)),
                    Some(Measured::Owned(ic1_y)),
                    Some(Measured::Owned(ic2_x)),
                    Some(Measured::Owned(ic2_y)),
                ],
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
        let values = &values[..n];
        mask.push(values);
        mm[index] = Some(Measured::Same(values));
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

fn spot_sigma(spot: &Sheet, prefer_raw: bool, n: usize) -> Vec<(&'static str, &[f32])> {
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
            out.push((key, &values[..n]));
        }
    }
    out
}

fn four_slices<'a>(spot: &'a Sheet, suffix: &str, n: usize) -> Option<[&'a [f32]; 4]> {
    let mut out = Vec::new();
    for (ic, axis) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")] {
        let values = column_suffix(spot, ic, axis, &[suffix])?;
        if values.len() < n {
            return None;
        }
        out.push(&values[..n]);
    }
    Some([out[0], out[1], out[2], out[3]])
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
    let mut made = Vec::new();
    let mut pairs = Vec::new();
    for (key, device) in [
        ("ic1_sig_x", "IC_1_X"),
        ("ic1_sig_y", "IC_1_Y"),
        ("ic2_sig_x", "IC_2_X"),
        ("ic2_sig_y", "IC_2_Y"),
    ] {
        let (Some(energy), Some(measured)) = (table.get("energy"), table.get(key)) else {
            continue;
        };
        let mut error = Vec::with_capacity(measured.len());
        for (sample, energy) in measured.iter().zip(energy) {
            match expected_mm(&bands, device, f64::from(*energy)) {
                Some(expected) if sample.is_finite() => {
                    pairs.push(*energy);
                    pairs.push(expected as f32);
                    error.push(sample - expected as f32);
                }
                _ => error.push(f32::NAN),
            }
        }
        made.push((format!("{key}_err"), error));
    }
    for (key, error) in made {
        table.insert(key, error);
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

pub(crate) fn load_slice_metric(root: &Path, session: &str, _metric: &str, chamber: bool) -> Table {
    let kind = if chamber {
        "slice-chamber"
    } else {
        "slice-iso"
    };
    cached_timeslice(root, session, kind, true, Family::Geometry, |frames| {
        build_slice_metric(root, session, chamber, frames)
    })
}

fn build_slice_metric(
    root: &Path,
    session: &str,
    chamber: bool,
    frames: &Frames,
) -> BTreeMap<String, Vec<f32>> {
    let spot_path = discover::session_directory(root, session).join("spot_data.csv");
    let spot = if spot_path.is_file() {
        Some(sheet_all(&spot_path, Keep::Spot, spot_keep))
    } else {
        discover::read_session_file(root, session, "spot_data.csv")
            .map(|bytes| Arc::new(read_sheet_where(&bytes, spot_keep)))
    };
    let plan = plan_xy(&frames.map);
    let previous = frames
        .prior
        .and_then(|span| recall_pos(root, session, chamber, span));
    let mut samples = previous
        .as_ref()
        .map(|state| state.samples.clone())
        .unwrap_or_else(|| std::array::from_fn(|_| HashMap::new()));
    let ingest_from = if previous.is_some() {
        frames.prior
    } else {
        None
    };
    if plan_spans(&plan) {
        for (index, sheet) in frames.sheets.iter().enumerate() {
            let Some(start) = row_origin(index, ingest_from) else {
                continue;
            };
            ingest_targets(sheet, &mut samples, start);
        }
    }
    let fitted = plan_spans(&plan)
        .then(|| fit_samples(&samples, &plan))
        .flatten();
    let strips = spot
        .as_deref()
        .map(strip_axes)
        .unwrap_or([None, None, None, None]);
    let (mut axes, shifts) = if let Some((fitted, shifts)) = fitted {
        let mut axes = strips;
        let mut out = [0.0; 4];
        for index in 0..4 {
            if fitted[index].is_some() {
                axes[index] = fitted[index];
                out[index] = shifts[index];
            }
        }
        (axes, out)
    } else {
        (strips, [0.0; 4])
    };
    // A parked map has no span, so the strip fit cannot run. devices.xml is the
    // conversion that wrote the spot file's millimetre columns.
    if !chamber {
        let device = device_axes(&discover::session_directory(root, session));
        for (axis, device) in axes.iter_mut().zip(device) {
            if let Some(device) = device {
                *axis = Some(device);
            }
        }
    }
    let gains = if chamber {
        [1.0; 4]
    } else {
        sigma_gain(spot.as_deref())
    };
    let append = previous
        .as_ref()
        .is_some_and(|state| !chamber && same_fit(&state.axes, &axes, &state.shifts, &shifts));
    if frames.prior.is_some() && !append {
        mark_full_table();
    }
    let row_from = if append { frames.prior } else { None };
    let rows = frames
        .sheets
        .iter()
        .map(|sheet| {
            sheet
                .num
                .get("rci_in_trigger")
                .or_else(|| sheet.num.get("r_beamOk"))
                .map(Vec::len)
                .unwrap_or(0)
        })
        .sum();
    let mut energy = Vec::with_capacity(rows);
    let mut beam = Vec::with_capacity(rows);
    let mut plan_xs = Vec::with_capacity(rows);
    let mut plan_ys = Vec::with_capacity(rows);
    let mut ic1_x = Vec::with_capacity(rows);
    let mut ic1_y = Vec::with_capacity(rows);
    let mut ic2_x = Vec::with_capacity(rows);
    let mut ic2_y = Vec::with_capacity(rows);
    let mut ic1_x_mm = Vec::with_capacity(rows);
    let mut ic1_y_mm = Vec::with_capacity(rows);
    let mut ic2_x_mm = Vec::with_capacity(rows);
    let mut ic2_y_mm = Vec::with_capacity(rows);
    let mut sig: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let mut sig_err: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let plan_ref = &plan;
    let axes_ref = &axes;
    let shifts_ref = &shifts;
    // Files do not share rows once the fit is known. Join before the table is published.
    let parts = std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for (file_index, sheet) in frames.sheets.iter().enumerate() {
            let Some(start) = row_origin(file_index, row_from) else {
                continue;
            };
            let tagged = frame_energy(
                &frames.energies,
                frames.layers.get(file_index).copied().unwrap_or(-1),
                file_index,
            );
            jobs.push(scope.spawn(move || {
                sheet_metric(
                    sheet, plan_ref, axes_ref, shifts_ref, chamber, tagged, gains, start,
                )
            }));
        }
        jobs.into_iter()
            .filter_map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });
    for part in parts {
        energy.extend(part.energy);
        beam.extend(part.beam);
        plan_xs.extend(part.plan_x);
        plan_ys.extend(part.plan_y);
        let [e0, e1, e2, e3] = part.err;
        ic1_x.extend(e0);
        ic1_y.extend(e1);
        ic2_x.extend(e2);
        ic2_y.extend(e3);
        let [m0, m1, m2, m3] = part.mm;
        ic1_x_mm.extend(m0);
        ic1_y_mm.extend(m1);
        ic2_x_mm.extend(m2);
        ic2_y_mm.extend(m3);
        for (into, values) in sig.iter_mut().zip(part.sig) {
            into.extend(values);
        }
        for (into, values) in sig_err.iter_mut().zip(part.sig_err) {
            into.extend(values);
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
        remember_pos(root, session, chamber, samples, axes, shifts);
        return table;
    }
    table.insert("energy".to_string(), energy);
    table.insert("beam_on".to_string(), beam);
    table.insert("ic1_x_err".to_string(), ic1_x);
    table.insert("ic1_y_err".to_string(), ic1_y);
    table.insert("ic2_x_err".to_string(), ic2_x);
    table.insert("ic2_y_err".to_string(), ic2_y);
    for (key, values) in [
        ("ic1_sig_x", std::mem::take(&mut sig[0])),
        ("ic1_sig_y", std::mem::take(&mut sig[1])),
        ("ic2_sig_x", std::mem::take(&mut sig[2])),
        ("ic2_sig_y", std::mem::take(&mut sig[3])),
        ("ic1_sig_x_err", std::mem::take(&mut sig_err[0])),
        ("ic1_sig_y_err", std::mem::take(&mut sig_err[1])),
        ("ic2_sig_x_err", std::mem::take(&mut sig_err[2])),
        ("ic2_sig_y_err", std::mem::take(&mut sig_err[3])),
    ] {
        table.insert(key.to_string(), values);
    }
    let gap = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(ic2, ic1)| ic2 - ic1)
            .collect::<Vec<_>>()
    };
    let x_diff = gap(&ic2_x_mm, &ic1_x_mm);
    let y_diff = gap(&ic2_y_mm, &ic1_y_mm);
    table.insert("ic1_x".to_string(), ic1_x_mm);
    table.insert("ic1_y".to_string(), ic1_y_mm);
    table.insert("ic2_x".to_string(), ic2_x_mm);
    table.insert("ic2_y".to_string(), ic2_y_mm);
    table.insert("ic12_x_diff".to_string(), x_diff);
    table.insert("ic12_y_diff".to_string(), y_diff);
    if row_from.is_some() || plan_xs.iter().any(|value| value.is_finite()) {
        table.insert("plan_x".to_string(), plan_xs);
        table.insert("plan_y".to_string(), plan_ys);
    }
    stamp_sample_time(&mut table);
    remember_pos(root, session, chamber, samples, axes, shifts);
    table
}

struct PosState {
    files: usize,
    tail_rows: usize,
    samples: [HashMap<(u32, u32), f32>; 4],
    axes: [Option<(f32, f32)>; 4],
    shifts: [f32; 4],
}

fn pos_cache() -> &'static Mutex<HashMap<(PathBuf, bool), PosState>> {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, bool), PosState>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn recall_pos(root: &Path, session: &str, chamber: bool, span: (usize, usize)) -> Option<PosState> {
    let key = (discover::session_directory(root, session), chamber);
    let mut cache = pos_cache().lock().unwrap_or_else(|err| err.into_inner());
    let state = cache.get(&key)?;
    if state.files != span.0 || state.tail_rows != span.1 {
        cache.remove(&key);
        return None;
    }
    cache.remove(&key)
}

fn remember_pos(
    root: &Path,
    session: &str,
    chamber: bool,
    samples: [HashMap<(u32, u32), f32>; 4],
    axes: [Option<(f32, f32)>; 4],
    shifts: [f32; 4],
) {
    let Some(SliceTake::Timeslice { files, tail_rows }) = slice_for(session) else {
        return;
    };
    let mut cache = pos_cache().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= TABLE_CACHE_CAP {
        evict_one(&mut cache);
    }
    cache.insert(
        (discover::session_directory(root, session), chamber),
        PosState {
            files,
            tail_rows,
            samples,
            axes,
            shifts,
        },
    );
}

fn same_fit(
    left: &[Option<(f32, f32)>; 4],
    right: &[Option<(f32, f32)>; 4],
    left_shift: &[f32; 4],
    right_shift: &[f32; 4],
) -> bool {
    left_shift == right_shift
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (Some((left_slope, left_intercept)), Some((right_slope, right_intercept))) => {
                    left_slope.to_bits() == right_slope.to_bits()
                        && left_intercept.to_bits() == right_intercept.to_bits()
                }
                (None, None) => true,
                _ => false,
            })
}

struct AxisCols<'a> {
    raw: Option<&'a [f32]>,
    spot: Option<&'a [f32]>,
    layer: Option<&'a [f32]>,
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
    let raw_name = if sheet.num.contains_key(&position) {
        Some(position)
    } else if sheet.num.contains_key(&spot_position) {
        Some(spot_position)
    } else {
        None
    };
    let (spot, layer) = clock_columns(sheet, raw_name.as_deref());
    AxisCols {
        raw: raw_name
            .as_ref()
            .and_then(|name| sheet.num.get(name))
            .map(Vec::as_slice),
        spot,
        layer,
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

/// Spot and layer written by the same controller as `column`.
///
/// Controllers log side by side and stay a spot or two apart. The block is the
/// repeated `spot_no` that owns this column. A file with one spot column still
/// shares that clock.
fn clock_columns<'a>(
    sheet: &'a Sheet,
    column: Option<&str>,
) -> (Option<&'a [f32]>, Option<&'a [f32]>) {
    let block = column.and_then(|name| sheet.block.get(name)).copied();
    let spot = block
        .and_then(|block| column_on_block(sheet, "spot_no", block))
        .or_else(|| shared_column(sheet, "spot_no"));
    let layer = block
        .and_then(|block| column_on_block(sheet, "layer_id", block))
        .or_else(|| shared_column(sheet, "layer_id"));
    (spot, layer)
}

fn column_on_block<'a>(sheet: &'a Sheet, base: &str, block: u32) -> Option<&'a [f32]> {
    let name = sheet.block.iter().find_map(|(name, clock)| {
        (*clock == block && header_base(name).eq_ignore_ascii_case(base)).then_some(name.as_str())
    })?;
    sheet.num.get(name).map(Vec::as_slice)
}

fn shared_column<'a>(sheet: &'a Sheet, base: &str) -> Option<&'a [f32]> {
    let mut found = None;
    for name in sheet.num.keys() {
        if !header_base(name).eq_ignore_ascii_case(base) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(name.as_str());
    }
    found.and_then(|name| sheet.num.get(name).map(Vec::as_slice))
}

struct SheetMetric {
    energy: Vec<f32>,
    beam: Vec<f32>,
    plan_x: Vec<f32>,
    plan_y: Vec<f32>,
    err: [Vec<f32>; 4],
    mm: [Vec<f32>; 4],
    sig: [Vec<f32>; 4],
    sig_err: [Vec<f32>; 4],
}

fn sheet_metric(
    sheet: &Sheet,
    plan: &HashMap<(u32, u32), (f32, f32)>,
    axes: &[Option<(f32, f32)>; 4],
    shifts: &[f32; 4],
    chamber: bool,
    tagged: f32,
    gains: [f32; 4],
    start: usize,
) -> Option<SheetMetric> {
    let trigger = sheet
        .num
        .get("rci_in_trigger")
        .or_else(|| sheet.num.get("r_beamOk"))?;
    let n = trigger.len();
    if start >= n {
        return None;
    }
    let count = n - start;
    let bound = [
        bind_axis(sheet, "ic1", "x"),
        bind_axis(sheet, "ic1", "y"),
        bind_axis(sheet, "ic2", "x"),
        bind_axis(sheet, "ic2", "y"),
    ];
    let plan_x = [true, false, true, false];
    let mut energy = Vec::with_capacity(count);
    let mut beam = Vec::with_capacity(count);
    let mut planned_x = Vec::with_capacity(count);
    let mut planned_y = Vec::with_capacity(count);
    let mut err: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(count));
    let mut mm: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(count));
    let mut sig: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(count));
    let mut sig_err: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(count));
    // One timeslice spot is thousands of rows. The plan key stays put.
    let mut plan_hit: [Option<(u32, u32, Option<(f32, f32)>)>; 4] = [None; 4];
    for row in start..n {
        let on = trigger[row] > 0.5;
        beam.push(if on { 1.0 } else { 0.0 });
        energy.push(tagged);
        let ic1 = plan_cached(
            plan,
            bound[0].layer,
            bound[0].spot,
            row,
            shifts[0],
            &mut plan_hit[0],
        );
        let (x, y) = ic1.unwrap_or((f32::NAN, f32::NAN));
        planned_x.push(x);
        planned_y.push(y);
        let mut position = [f32::NAN; 4];
        let mut accepted = [false; 4];
        for index in 0..4 {
            let axis = &bound[index];
            let raw = at(axis.raw, row);
            accepted[index] = axis_passes(axis, row);
            position[index] = if !accepted[index] {
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
            let planned = plan_cached(
                plan,
                axis.layer,
                axis.spot,
                row,
                shifts[index],
                &mut plan_hit[index],
            )
            .map(|xy| if plan_x[index] { xy.0 } else { xy.1 })
            .unwrap_or(f32::NAN);
            let delta = if position[index].is_finite() && planned.is_finite() {
                position[index] - planned
            } else {
                f32::NAN
            };
            err[index].push(delta);
            mm[index].push(position[index]);
        }
        for (index, axis) in bound.iter().enumerate() {
            let gain = if chamber { 1.0 } else { gains[index] };
            let value = at(axis.sigma, row);
            let value = if !value.is_finite() || value <= 0.0 || value > 20.0 || !accepted[index] {
                f32::NAN
            } else {
                value * gain
            };
            sig[index].push(value);
            let target = at(axis.sigma_target, row);
            sig_err[index].push(if value.is_finite() && target.is_finite() && target > 0.0 {
                value - target * gain
            } else {
                f32::NAN
            });
        }
    }
    Some(SheetMetric {
        energy,
        beam,
        plan_x: planned_x,
        plan_y: planned_y,
        err,
        mm,
        sig,
        sig_err,
    })
}

fn plan_cached(
    plan: &HashMap<(u32, u32), (f32, f32)>,
    layer: Option<&[f32]>,
    spot: Option<&[f32]>,
    row: usize,
    shift: f32,
    hit: &mut Option<(u32, u32, Option<(f32, f32)>)>,
) -> Option<(f32, f32)> {
    let layer = at(layer, row);
    let spot = at(spot, row) + shift;
    if !layer.is_finite() || !spot.is_finite() {
        return None;
    }
    let key = (layer.to_bits(), spot.to_bits());
    if let Some((have_layer, have_spot, xy)) = *hit {
        if have_layer == key.0 && have_spot == key.1 {
            return xy;
        }
    }
    let xy = plan.get(&key).copied();
    *hit = Some((key.0, key.1, xy));
    xy
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
    for (((layer, spot), x), y) in layer.iter().zip(spot).zip(x).zip(y) {
        out.insert((layer.to_bits(), spot.to_bits()), (*x, *y));
    }
    out
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

/// Each target is keyed by the spot and layer in that target's own controller
/// block. IC2 can be on the next spot while IC1 is still on the previous one.
/// ponytail: the first finite target per spot stands in for the per-spot median.
fn ingest_targets(sheet: &Sheet, samples: &mut [HashMap<(u32, u32), f32>; 4], start: usize) {
    let names = [
        "ic1_position_x_target",
        "ic1_position_y_target",
        "ic2_position_x_target",
        "ic2_position_y_target",
    ];
    for (index, name) in names.iter().enumerate() {
        let Some(values) = sheet.num.get(*name) else {
            continue;
        };
        let (spots, layers) = clock_columns(sheet, Some(name));
        for row in start..values.len() {
            let sample = values[row];
            let layer = at(layers, row);
            let spot = at(spots, row);
            if !sample.is_finite() || !layer.is_finite() || !spot.is_finite() {
                continue;
            }
            samples[index]
                .entry((layer.to_bits(), spot.to_bits()))
                .or_insert(sample);
        }
    }
}

fn fit_samples(
    samples: &[HashMap<(u32, u32), f32>; 4],
    plan: &HashMap<(u32, u32), (f32, f32)>,
) -> Option<([Option<(f32, f32)>; 4], [f32; 4])> {
    let mut axes = [None; 4];
    let mut shifts = [0.0; 4];
    let mut any = false;
    for index in 0..4 {
        let Some((axis, shift)) = fit_target_axis(&samples[index], plan, index % 2 == 0) else {
            continue;
        };
        axes[index] = Some(axis);
        shifts[index] = shift;
        any = true;
    }
    any.then_some((axes, shifts))
}

fn fit_target_axis(
    samples: &HashMap<(u32, u32), f32>,
    plan: &HashMap<(u32, u32), (f32, f32)>,
    x_axis: bool,
) -> Option<((f32, f32), f32)> {
    let mut best: Option<((usize, i32, i32), (f32, f32), f32)> = None;
    for shift in -2i32..=2 {
        let mut strips = Vec::new();
        let mut isos = Vec::new();
        for ((layer, spot), sample) in samples {
            let shifted = f32::from_bits(*spot) + shift as f32;
            let Some(planned) = plan.get(&(*layer, shifted.to_bits())) else {
                continue;
            };
            strips.push(*sample);
            isos.push(if x_axis { planned.0 } else { planned.1 });
        }
        if strips.len() < 10 {
            continue;
        }
        let Some(axis) = fit_strip_iso(&strips, &isos) else {
            continue;
        };
        let rank = (strips.len(), -shift.abs(), shift);
        if best.as_ref().is_none_or(|(have, _, _)| rank > *have) {
            best = Some((rank, axis, shift as f32));
        }
    }
    best.map(|(_, axis, shift)| (axis, shift))
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

/// Strip → isocenter mm from `devices.xml` and `scan_dose_system.xml`.
///
/// Magnification is virtual SAD over source-to-device distance. The sign carries
/// `reverse_strips`. Missing chambers stay unset so the strip fit can still run.
fn device_axes(dir: &Path) -> [Option<(f32, f32)>; 4] {
    let Ok(devices) = std::fs::read_to_string(dir.join("config/map2map/devices.xml")) else {
        return [None; 4];
    };
    let Ok(system) = std::fs::read_to_string(dir.join("config/map2map/scan_dose_system.xml"))
    else {
        return [None; 4];
    };
    let Some(sad) = tag_float(&system, "source_to_isocenter_distance") else {
        return [None; 4];
    };
    if !(sad > 0.0 && sad <= 1.0e5) {
        return [None; 4];
    }
    let mut axes = [None; 4];
    for (index, name) in ["IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"]
        .into_iter()
        .enumerate()
    {
        axes[index] = chamber_axis(&devices, name, sad);
    }
    axes
}

fn chamber_axis(xml: &str, name: &str, sad: f32) -> Option<(f32, f32)> {
    let needle = format!("name=\"{name}\"");
    let at = xml.find(&needle)?;
    let start = xml[..at].rfind("<ion_chamber")?;
    let rest = &xml[at..];
    let end = at + rest.find("</ion_chamber>")?;
    let body = &xml[start..end];
    let mut pitch = tag_float(body, "strip_to_mm")?;
    let count = tag_float(body, "strip_count")?;
    let sdd = tag_float(body, "source_to_device_distance_mm")?;
    if sdd.abs() <= 1.0e-4 || count <= 0.0 {
        return None;
    }
    if pitch < 0.1 {
        pitch = 1.0;
    }
    let offset = tag_float(body, "zero_offset_at_iso_mm").unwrap_or(0.0);
    let reverse = tag_text(body, "reverse_strips")
        .is_some_and(|text| text != "0" && !text.eq_ignore_ascii_case("false"));
    let sign = if reverse { -1.0 } else { 1.0 };
    let slope = sign * pitch * (sad / sdd);
    let center = count / 2.0 - 0.5;
    Some((slope, offset - slope * center))
}

fn tag_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let start = xml.find(&open)? + open.len();
    let rest = &xml[start..];
    let end = rest.find('<')?;
    Some(rest[..end].trim())
}

fn tag_float(xml: &str, tag: &str) -> Option<f32> {
    tag_text(xml, tag)?.parse().ok()
}

/// Millimetres per strip for timeslice sigma.
///
/// Modern logs store `r_*_sigma` in strips. The spot file's processed/raw pair
/// is the conversion already applied to that session. A log without the pair
/// keeps the recorded number.
fn sigma_gain(spot: Option<&Sheet>) -> [f32; 4] {
    let mut gain = [1.0; 4];
    let Some(spot) = spot else {
        return gain;
    };
    for (index, (ic, axis)) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")]
        .into_iter()
        .enumerate()
    {
        let Some(raw) = column_suffix(spot, ic, axis, &["spot_sigma_raw"]) else {
            continue;
        };
        let Some(iso) = column_suffix(spot, ic, axis, &["spot_sigma"]) else {
            continue;
        };
        let mut ratios = Vec::new();
        for (raw, iso) in raw.iter().zip(iso) {
            if *raw > 0.2 && iso.is_finite() && *iso > 0.0 {
                ratios.push(*iso / *raw);
            }
        }
        if ratios.len() < 10 {
            continue;
        }
        let ratio = median(&ratios);
        if ratio > 1.5 && ratio < 20.0 {
            gain[index] = ratio;
        }
    }
    gain
}

pub(crate) fn load_timeslice(root: &Path, session: &str, metric: &str) -> Table {
    // The default picture plots beam current. Amplifier, field, and peak
    // columns stay out of this parse.
    let family = if matches!(metric, "ic_current" | "current_ratio") {
        Family::Current
    } else {
        Family::Signals
    };
    cached_timeslice(
        root,
        session,
        &format!("metric:{metric}"),
        false,
        family,
        |frames| {
            if metric == "current_ratio" {
                if frames.prior.is_some() {
                    mark_full_table();
                }
                return current_ratio_from(&frames.sheets, &frames.energies, &frames.layers);
            }
            if matches!(
                metric,
                "fit_confidence" | "peak_amplitude" | "amplifier_error" | "probe_field"
            ) {
                return signal_from(
                    &frames.sheets,
                    &frames.energies,
                    &frames.layers,
                    metric,
                    frames.prior,
                );
            }
            current_from(
                &frames.sheets,
                &frames.energies,
                &frames.layers,
                frames.prior,
            )
        },
    )
}

/// Column names from the first line of the files for one grain.
///
/// The body stays unread. The picker uses these names and does not open a file.
///
/// ponytail: 32 header lists. A repeat plot (glyph, bins, filter) stats the same
/// files and skips the directory walk. A changed length or mtime reads the line
/// again. A timeslice file that is not there yet is not cached, so the next
/// call still looks.
const HEADER_CACHE_CAP: usize = 32;

#[derive(Hash, PartialEq, Eq)]
struct HeaderKey {
    dir: PathBuf,
    timeslice: bool,
}

struct HeaderEntry {
    watched: Vec<(PathBuf, u128)>,
    columns: Vec<String>,
}

fn header_cache() -> &'static Mutex<HashMap<HeaderKey, HeaderEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<HeaderKey, HeaderEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn grain_columns(root: &Path, session: &str, timeslice: bool) -> Vec<String> {
    let dir = discover::session_directory(root, session);
    let key = HeaderKey {
        dir: dir.clone(),
        timeslice,
    };
    if let Some(hit) = header_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
    {
        if hit
            .watched
            .iter()
            .all(|(path, stamp)| discover::meta_stamp(path) == *stamp)
        {
            return hit.columns.clone();
        }
    }
    let (columns, watched, cacheable) = read_grain_headers(&dir, timeslice);
    if cacheable {
        let mut cache = header_cache().lock().unwrap_or_else(|err| err.into_inner());
        if cache.len() >= HEADER_CACHE_CAP {
            evict_one(&mut cache);
        }
        cache.insert(
            key,
            HeaderEntry {
                watched,
                columns: columns.clone(),
            },
        );
    }
    columns
}

fn read_grain_headers(dir: &Path, timeslice: bool) -> (Vec<String>, Vec<(PathBuf, u128)>, bool) {
    let mut names = Vec::new();
    let mut watched = Vec::new();
    let mut cacheable = true;
    if timeslice {
        if let Some(path) = first_named(dir, |name| {
            name.ends_with("timeslice_data_device_units.csv")
        }) {
            names.extend(header_columns(&path));
            watched.push(watch(&path));
        } else {
            cacheable = false;
        }
    } else {
        let path = dir.join("spot_data.csv");
        names.extend(header_columns(&path));
        watched.push(watch(&path));
    }
    let map = dir.join("input_map.csv");
    names.extend(header_columns(&map));
    watched.push(watch(&map));
    (names, watched, cacheable)
}

fn watch(path: &Path) -> (PathBuf, u128) {
    let stamp = discover::meta_stamp(path);
    (path.to_path_buf(), stamp)
}

fn header_columns(path: &Path) -> Vec<String> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut line = String::new();
    let Ok(read) = std::io::BufRead::read_line(&mut std::io::BufReader::new(file), &mut line)
    else {
        return Vec::new();
    };
    if read == 0 {
        return Vec::new();
    }
    let line = line.trim_end_matches(['\r', '\n']);
    let mut names = Vec::new();
    let mut cursor = 0usize;
    while let Some((start, end, next)) = next_cell(line, cursor) {
        if next <= cursor {
            break;
        }
        let name = cell_owned(line, start, end);
        if !name.is_empty() {
            names.push(name);
        }
        cursor = next;
    }
    names
}

fn first_named(dir: &Path, pred: impl Fn(&str) -> bool) -> Option<PathBuf> {
    fn walk(dir: &Path, pred: &impl Fn(&str) -> bool, depth: u8) -> Option<PathBuf> {
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            } else if path.is_dir() && depth < 4 {
                dirs.push(path);
            }
        }
        files.sort();
        for path in files {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if pred(name) {
                return Some(path);
            }
        }
        dirs.sort();
        for child in dirs {
            if let Some(found) = walk(&child, pred, depth + 1) {
                return Some(found);
            }
        }
        None
    }
    walk(dir, &pred, 0)
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
    signal_from(
        &read_sheets(files, signal_column),
        energies,
        layers,
        metric,
        None,
    )
}

fn signal_from(
    sheets: &[Arc<Sheet>],
    energies: &[f32],
    layers: &[i64],
    metric: &str,
    prior: Option<(usize, usize)>,
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
            let Some(start) = row_origin(index, prior) else {
                continue;
            };
            let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
            let cmd_x = concept_col(sheet, "amplifier_cmd_x");
            let read_x = concept_col(sheet, "amplifier_readback_x");
            let cmd_y = concept_col(sheet, "amplifier_cmd_y");
            let read_y = concept_col(sheet, "amplifier_readback_y");
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
            append_samples(
                sheet,
                tag,
                &columns,
                start,
                &mut energy,
                &mut beam,
                &mut stored,
            );
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
        let Some(start) = row_origin(index, prior) else {
            continue;
        };
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let columns: Vec<&[f32]> = spec
            .iter()
            .map(|(_, concept)| concept_col(sheet, concept))
            .collect();
        append_samples(
            sheet,
            tag,
            &columns,
            start,
            &mut energy,
            &mut beam,
            &mut stored,
        );
    }
    let pairs: Vec<(&str, &Vec<f32>)> = spec
        .iter()
        .zip(&stored)
        .map(|((key, _), values)| (*key, values))
        .collect();
    signal_rows(energy, beam, &pairs)
}

fn concept_col<'a>(sheet: &'a Sheet, concept: &str) -> &'a [f32] {
    find_header(sheet, concept, &[])
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

/// Trigger length, or the longest requested column when the file has no trigger.
fn sample_len(sheet: &Sheet, columns: &[&[f32]]) -> usize {
    column_any(sheet, &["rci_in_trigger", "r_beamOk"])
        .map(|values| values.len())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| columns.iter().map(|column| column.len()).max().unwrap_or(0))
}

fn append_samples(
    sheet: &Sheet,
    tag: f32,
    columns: &[&[f32]],
    start: usize,
    energy: &mut Vec<f32>,
    beam_out: &mut Vec<f32>,
    series_out: &mut [Vec<f32>],
) {
    let n = sample_len(sheet, columns);
    if n == 0 || start >= n {
        return;
    }
    let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
    for sample in start..n {
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
    stamp_sample_time(&mut table);
    table
}

fn current_from(
    sheets: &[Arc<Sheet>],
    energies: &[f32],
    layers: &[i64],
    prior: Option<(usize, usize)>,
) -> BTreeMap<String, Vec<f32>> {
    let mut out_energy = Vec::new();
    let mut ic1 = Vec::new();
    let mut ic2 = Vec::new();
    let mut ic3 = Vec::new();
    let mut beam = Vec::new();
    let mut any_ic3 = false;
    for (index, sheet) in sheets.iter().enumerate() {
        let Some(start) = row_origin(index, prior) else {
            continue;
        };
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let a = scaled_current(sheet, "ic1");
        let b = scaled_current(sheet, "ic2");
        let c = ic3_current(sheet);
        let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
        let n = sample_len(sheet, &[a.as_slice(), b.as_slice(), c.as_slice()]);
        for sample in start..n {
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
        stamp_sample_time(&mut table);
    }
    table
}

/// Timeslice rows are 1 ms apart, the same clock Timeslice Replay draws.
const SAMPLE_S: f32 = 0.001;

fn stamp_sample_time(table: &mut BTreeMap<String, Vec<f32>>) {
    let Some(n) = table.get("energy").map(Vec::len).filter(|n| *n > 0) else {
        return;
    };
    table.insert(
        "time_s".to_string(),
        (0..n).map(|index| index as f32 * SAMPLE_S).collect(),
    );
}

fn restamp_sample_time(table: &mut BTreeMap<String, Vec<f32>>) {
    if table.contains_key("time_s") {
        stamp_sample_time(table);
    }
}

/// Layer-change times for the scrubber. Timeslice uses the 1 ms file clock.
/// Spot rows use the wall clock already stored on `time_s`.
pub(crate) fn timeline_layers(root: &Path, sessions: &[String], timeslice: bool) -> Vec<f32> {
    let mut marks = Vec::new();
    for session in sessions {
        if timeslice {
            marks.extend(timeslice_layer_marks(root, session));
        } else {
            marks.extend(spot_layer_marks(root, session));
        }
    }
    marks.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    marks.dedup_by(|left, right| left.to_bits() == right.to_bits());
    marks
}

fn spot_layer_marks(root: &Path, session: &str) -> Vec<f32> {
    let table = load_spot(root, session, false, false, false);
    scan_kit_core::layer_edges(
        table.get("time_s").map(Vec::as_slice).unwrap_or(&[]),
        table.get("layer_id").map(Vec::as_slice).unwrap_or(&[]),
    )
}

fn timeslice_layer_marks(root: &Path, session: &str) -> Vec<f32> {
    let Some(frames) = open_frames(root, session, Family::Current) else {
        return Vec::new();
    };
    let mut marks = Vec::new();
    let mut cursor = 0.0f32;
    let mut previous: Option<i64> = None;
    for (index, sheet) in frames.sheets.iter().enumerate() {
        let rows = clock_rows(sheet);
        let folder = frames.layers.get(index).copied().unwrap_or(-1);
        let column = sheet
            .num
            .get("layer_id")
            .filter(|values| values.iter().any(|value| value.is_finite()));
        if let Some(ids) = column {
            for row in 0..rows {
                let Some(id) = ids.get(row).copied().filter(|value| value.is_finite()) else {
                    continue;
                };
                note_layer(
                    &mut marks,
                    &mut previous,
                    id as i64,
                    cursor + row as f32 * SAMPLE_S,
                );
            }
        } else if folder >= 0 && rows > 0 {
            note_layer(&mut marks, &mut previous, folder, cursor);
        }
        cursor += rows as f32 * SAMPLE_S;
    }
    marks
}

fn clock_rows(sheet: &Sheet) -> usize {
    let triggered = sample_len(sheet, &[]);
    if triggered > 0 {
        triggered
    } else {
        sheet.num.values().map(Vec::len).max().unwrap_or(0)
    }
}

fn note_layer(marks: &mut Vec<f32>, previous: &mut Option<i64>, id: i64, at: f32) {
    if previous.is_some_and(|seen| seen != id) && at > 0.0 {
        marks.push(at);
    }
    *previous = Some(id);
}

fn current_ratio_from(
    sheets: &[Arc<Sheet>],
    energies: &[f32],
    layers: &[i64],
) -> BTreeMap<String, Vec<f32>> {
    let mut out_energy = Vec::new();
    let mut ic21 = Vec::new();
    let mut ic31 = Vec::new();
    let mut ic32 = Vec::new();
    let mut clock = Vec::new();
    let mut cursor = 0.0f32;
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
        let n = sample_len(sheet, &[a.as_slice(), b.as_slice(), c.as_slice()]);
        cursor += n as f32 * SAMPLE_S;
        clock.push(cursor);
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
        table.insert("time_s".to_string(), clock);
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
    let extra: Vec<&str> = fallback.iter().map(String::as_str).collect();
    let Some(header) = find_header(sheet, &format!("{ic}_current"), &extra) else {
        return Vec::new();
    };
    let Some(values) = sheet.num.get(header) else {
        return Vec::new();
    };
    let factor = column_scale_factor(header).unwrap_or(1.0) as f32;
    scale_column(values, factor)
}

pub(crate) fn ic3_current(sheet: &Sheet) -> Vec<f32> {
    let mut sum = Vec::new();
    for (concept, part) in [
        ("ic3_current_a", "a"),
        ("ic3_current_b", "b"),
        ("ic3_current_c", "c"),
        ("ic3_current_d", "d"),
    ] {
        let values = if let Some(header) = find_header(sheet, concept, &[]) {
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

fn find_header<'a>(sheet: &'a Sheet, concept: &str, extra: &[&str]) -> Option<&'a str> {
    if let Some(name) = header_named(sheet, concept) {
        return Some(name);
    }
    for candidate in scan_kit_core::concept_column_candidates(concept, None) {
        if let Some(name) = header_named(sheet, &candidate) {
            return Some(name);
        }
    }
    extra
        .iter()
        .find_map(|name| sheet.num.get_key_value(*name).map(|(key, _)| key.as_str()))
}

fn header_named<'a>(sheet: &'a Sheet, requested: &str) -> Option<&'a str> {
    if let Some((name, _)) = sheet.num.get_key_value(requested) {
        return Some(name);
    }
    let want = scan_kit_core::normalize_column_name(requested);
    let mut found = None;
    for name in sheet.num.keys() {
        if scan_kit_core::normalize_column_name(name) == want {
            found = Some(name.as_str());
        }
    }
    found
}

pub(crate) fn column<'a>(sheet: &'a Sheet, concept: &str, extra: &[&str]) -> &'a [f32] {
    find_header(sheet, concept, extra)
        .and_then(|name| sheet.num.get(name))
        .map(Vec::as_slice)
        .unwrap_or(&[])
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
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    let mut any = false;
    for value in values {
        if value.is_finite() {
            any = true;
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
    }
    if !any {
        return (0.0, 1.0);
    }
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

fn unique_seen(values: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for value in values {
        if !value.is_finite() || !seen.insert(value.to_bits()) {
            continue;
        }
        if out.iter().any(|have: &f32| same(*have, *value)) {
            continue;
        }
        out.push(*value);
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

fn family_parse(family: Family) -> (Keep, fn(&str) -> bool) {
    match family {
        Family::Current => (Keep::Current, current_column),
        Family::Signals => (Keep::Signals, timeline_column),
        Family::Geometry => (Keep::Geometry, slice_column),
        Family::All => (Keep::All, timeslice_keep),
    }
}

fn spot_keep(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "layer_id"
            | "spot_no"
            | "beam_on"
            | "rci_in_trigger"
            | "r_beamok"
            | "datetime"
            | "timestamp"
            | "time_s"
            | "time_ns"
    ) || lower.contains("dose")
        || lower.contains("point_time")
        || lower.contains("spot_position")
        || lower.contains("spot_raw")
        || lower.contains("spot_sigma")
        || lower.contains("_spot")
        || lower.contains("_position")
        || lower.contains("_sigma")
}

fn parse_paths(listed: &[(i64, PathBuf)], kind: Keep, keep: fn(&str) -> bool) -> Vec<Arc<Sheet>> {
    if listed.len() < 2 {
        let _quiet = QuietTail::suppress();
        return vec![sheet_all(&listed[0].1, kind, keep)];
    }
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, listed.len());
    let chunk = listed.len().div_ceil(workers);
    let mut sheets = Vec::with_capacity(listed.len());
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for piece in listed.chunks(chunk) {
            joins.push(scope.spawn(move || {
                let _quiet = QuietTail::suppress();
                piece
                    .iter()
                    .map(|(_, path)| sheet_all(path, kind, keep))
                    .collect::<Vec<_>>()
            }));
        }
        for join in joins {
            sheets.extend(join.join().unwrap());
        }
    });
    sheets
}

// ponytail: a few chunks, not a pool. Tens of timeslice files, not thousands.
fn read_sheets(files: &[Vec<u8>], keep: fn(&str) -> bool) -> Vec<Arc<Sheet>> {
    if files.len() < 2 {
        return files
            .iter()
            .map(|bytes| Arc::new(read_sheet_where(bytes, keep)))
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
                    .map(|bytes| Arc::new(read_sheet_where(bytes, keep)))
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

/// A few megabytes. Smaller files stay on one thread; the bench can move this.
const PARALLEL_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
enum Kind {
    Skip,
    Num,
    Wide,
    Clock,
}

/// Parse a device CSV without a `String` per cell.
///
/// ponytail: the decimal scanner handles the padded fixed-point form these logs
/// use. Scientific notation and non-finite tokens fall back to `str::parse`.
/// `keep` drops columns a caller will not read. Strip columns stay dropped.
/// A large file is split on newlines and parsed in row ranges.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Keep {
    Current,
    Signals,
    Geometry,
    All,
    Spot,
    Map,
}

#[derive(PartialEq, Eq, Hash)]
struct SheetKey {
    path: PathBuf,
    stamp: u128,
    keep: Keep,
}

#[derive(Clone)]
struct Layout {
    names: Vec<Option<String>>,
    kinds: Vec<Kind>,
    block: BTreeMap<String, u32>,
}

struct Tail {
    stamp: u128,
    sheet: Arc<Sheet>,
    rows: usize,
    cursor: usize,
    done: bool,
    /// The header line, including its newline, so a later window can resume.
    header: Vec<u8>,
    layout: Option<Layout>,
    file: Option<std::fs::File>,
    /// Read buffers, moved in. A later family joins them. The first family does not copy.
    pieces: Vec<Arc<Vec<u8>>>,
    joined: Option<Arc<Vec<u8>>>,
}

struct FileWindow {
    stamp: u128,
    bytes: Arc<Vec<u8>>,
    rows: usize,
    done: bool,
}

struct ParsedWindow {
    stamp: u128,
    rows: usize,
    sheet: Arc<Sheet>,
}

fn file_sheets() -> &'static Mutex<HashMap<SheetKey, Arc<Sheet>>> {
    static CACHE: OnceLock<Mutex<HashMap<SheetKey, Arc<Sheet>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tails() -> &'static Mutex<HashMap<(PathBuf, Keep), Arc<Mutex<Tail>>>> {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, Keep), Arc<Mutex<Tail>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

const FILE_SHEET_CAP: usize = 64;

fn sheet_key(path: &Path, stamp: u128, keep: Keep) -> SheetKey {
    SheetKey {
        path: path.to_path_buf(),
        stamp,
        keep,
    }
}

fn cached_sheet(key: &SheetKey) -> Option<Arc<Sheet>> {
    file_sheets()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(key)
        .map(Arc::clone)
}

fn store_sheet(key: SheetKey, sheet: &Arc<Sheet>) {
    let mut cache = file_sheets().lock().unwrap_or_else(|err| err.into_inner());
    if cache.len() >= FILE_SHEET_CAP {
        evict_one(&mut cache);
    }
    cache.insert(key, Arc::clone(sheet));
}

fn read_and_cache(path: &Path, stamp: u128, keep: Keep, pred: fn(&str) -> bool) -> Arc<Sheet> {
    let key = sheet_key(path, stamp, keep);
    if let Some(hit) = cached_sheet(&key) {
        return hit;
    }
    let (bytes, fresh) = if let Some(window) = window_of(path, stamp).filter(|window| window.done) {
        (window.bytes, false)
    } else {
        (
            {
                note_disk_read(path);
                Arc::new(std::fs::read(path).unwrap_or_default())
            },
            true,
        )
    };
    let sheet = Arc::new(read_sheet_where(&bytes, pred));
    if fresh {
        remember_window(path, stamp, Arc::clone(&bytes), finished_rows(&sheet), true);
    }
    store_sheet(key, &sheet);
    sheet
}

fn sheet_all(path: &Path, keep: Keep, pred: fn(&str) -> bool) -> Arc<Sheet> {
    let stamp = discover::meta_stamp(path);
    let key = sheet_key(path, stamp, keep);
    if let Some(hit) = cached_sheet(&key) {
        return hit;
    }
    let resume = tails()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&(path.to_path_buf(), keep))
        .is_some_and(|tail| {
            let tail = tail.lock().unwrap_or_else(|err| err.into_inner());
            tail.stamp == stamp && !tail.done && tail.rows > 0
        });
    if resume {
        return sheet_rows(path, keep, pred, usize::MAX);
    }
    read_and_cache(path, stamp, keep, pred)
}

/// `want` rows already cover this file.
///
/// ponytail: a row shorter than 64 bytes is treated as covered without a
/// sample, then clipped to `want`. Timeslice rows are ~1 KB, so a 4096-row
/// window still does not open a long file. Sample every file if a dense log
/// shows up under that floor.
fn file_fits_window(path: &Path, want: usize) -> bool {
    if want == usize::MAX {
        return true;
    }
    let Ok(len) = std::fs::metadata(path).map(|meta| meta.len()) else {
        return false;
    };
    if len == 0 {
        return true;
    }
    let floor = (want as u64).saturating_add(1).saturating_mul(64);
    if len <= floor {
        return true;
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 64 * 1024];
    let Ok(n) = file.read(&mut buf) else {
        return false;
    };
    if n == 0 {
        return true;
    }
    let lines = buf[..n]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        .max(1);
    let per = (n as u64).div_ceil(lines as u64).max(1);
    len.div_ceil(per) <= want as u64 + 1
}

fn empty_tail(stamp: u128) -> Tail {
    Tail {
        stamp,
        sheet: Arc::new(empty_sheet()),
        rows: 0,
        cursor: 0,
        done: false,
        header: Vec::new(),
        layout: None,
        file: None,
        pieces: Vec::new(),
        joined: None,
    }
}

fn tail_bytes(tail: &mut Tail) -> Arc<Vec<u8>> {
    if tail.pieces.len() == 1 {
        return Arc::clone(&tail.pieces[0]);
    }
    if let Some(joined) = &tail.joined {
        return Arc::clone(joined);
    }
    let mut all = Vec::with_capacity(tail.pieces.iter().map(|piece| piece.len()).sum());
    for piece in &tail.pieces {
        all.extend_from_slice(piece);
    }
    let joined = Arc::new(all);
    tail.pieces.clear();
    tail.pieces.push(Arc::clone(&joined));
    tail.joined = Some(Arc::clone(&joined));
    joined
}

fn windows() -> &'static Mutex<HashMap<PathBuf, FileWindow>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, FileWindow>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn parsed_windows() -> &'static Mutex<HashMap<(PathBuf, Keep), ParsedWindow>> {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, Keep), ParsedWindow>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

const WINDOW_CAP: usize = 32;

fn remember_window(path: &Path, stamp: u128, bytes: Arc<Vec<u8>>, rows: usize, done: bool) {
    if bytes.is_empty() {
        return;
    }
    let mut cache = windows().lock().unwrap_or_else(|err| err.into_inner());
    let key = path.to_path_buf();
    if cache.len() >= WINDOW_CAP && !cache.contains_key(&key) {
        evict_one(&mut cache);
    }
    cache.insert(
        key,
        FileWindow {
            stamp,
            bytes,
            rows,
            done,
        },
    );
}

/// Bytes already read for this path. The tail lock is not held across the map lock.
fn window_of(path: &Path, stamp: u128) -> Option<FileWindow> {
    {
        let cache = windows().lock().unwrap_or_else(|err| err.into_inner());
        if let Some(hit) = cache.get(path) {
            if hit.stamp == stamp && !hit.bytes.is_empty() {
                return Some(FileWindow {
                    stamp: hit.stamp,
                    bytes: Arc::clone(&hit.bytes),
                    rows: hit.rows,
                    done: hit.done,
                });
            }
        }
    }
    let slots: Vec<Arc<Mutex<Tail>>> = {
        let held = tails().lock().unwrap_or_else(|err| err.into_inner());
        held.iter()
            .filter(|((have, _), _)| have == path)
            .map(|(_, tail)| Arc::clone(tail))
            .collect()
    };
    let mut best: Option<(Arc<Mutex<Tail>>, usize)> = None;
    for slot in slots {
        let tail = slot.lock().unwrap_or_else(|err| err.into_inner());
        if tail.stamp != stamp || tail.rows == 0 || tail.pieces.is_empty() {
            continue;
        }
        let take = best.as_ref().is_none_or(|(_, rows)| tail.rows > *rows);
        if take {
            best = Some((Arc::clone(&slot), tail.rows));
        }
    }
    let (slot, _) = best?;
    let mut tail = slot.lock().unwrap_or_else(|err| err.into_inner());
    if tail.stamp != stamp || tail.pieces.is_empty() {
        return None;
    }
    Some(FileWindow {
        stamp,
        bytes: tail_bytes(&mut tail),
        rows: tail.rows,
        done: tail.done,
    })
}

fn keep_is_open(path: &Path, keep: Keep, stamp: u128) -> bool {
    tails()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&(path.to_path_buf(), keep))
        .is_some_and(|tail| {
            let tail = tail.lock().unwrap_or_else(|err| err.into_inner());
            tail.stamp == stamp && tail.rows > 0
        })
}

fn sheet_from_window(
    path: &Path,
    stamp: u128,
    keep: Keep,
    pred: fn(&str) -> bool,
    want: usize,
) -> Option<Arc<Sheet>> {
    if keep_is_open(path, keep, stamp) {
        return None;
    }
    let window = window_of(path, stamp)?;
    if window.bytes.is_empty() || (!window.done && window.rows < want) {
        return None;
    }
    let key = (path.to_path_buf(), keep);
    if let Some(hit) = parsed_windows()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
    {
        if hit.stamp == stamp && hit.rows == window.rows {
            let have = finished_rows(&hit.sheet);
            if window.done && want >= have {
                set_tail_done(true, |name| path_has_session(path, name));
            }
            return Some(shared_rows(&hit.sheet, want.min(have), have));
        }
    }
    let sheet = Arc::new(read_sheet_where(&window.bytes, pred));
    let have = finished_rows(&sheet);
    if window.done {
        store_sheet(sheet_key(path, stamp, keep), &sheet);
        if want >= have {
            set_tail_done(true, |name| path_has_session(path, name));
        }
    } else {
        let mut cache = parsed_windows()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if cache.len() >= WINDOW_CAP && !cache.contains_key(&key) {
            evict_one(&mut cache);
        }
        cache.insert(
            key,
            ParsedWindow {
                stamp,
                rows: window.rows,
                sheet: Arc::clone(&sheet),
            },
        );
    }
    Some(shared_rows(&sheet, want.min(have), have))
}

/// Index of the `n`th newline, if it is in `bytes`.
fn nth_newline(bytes: &[u8], n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let mut seen = 0usize;
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let word = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let xor = word ^ 0x0a0a_0a0a_0a0a_0a0a;
        let mut mask = xor.wrapping_sub(0x0101_0101_0101_0101) & !xor & 0x8080_8080_8080_8080;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let index = i + bit / 8;
            if bytes[index] == b'\n' {
                seen += 1;
                if seen == n {
                    return Some(index);
                }
            }
            mask &= mask - 1;
        }
        i += 8;
    }
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            seen += 1;
            if seen == n {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn trim_newlines(file: &mut std::fs::File, mut buf: Vec<u8>, newlines: usize) -> (Vec<u8>, bool) {
    let Some(at) = nth_newline(&buf, newlines) else {
        return (buf, true);
    };
    let cut = at + 1;
    let extra = buf.len() - cut;
    buf.truncate(cut);
    if extra > 0 {
        let _ = file.seek(SeekFrom::Current(-(extra as i64)));
    }
    (buf, false)
}

/// Width used before any row has been seen. A measured width replaces it.
const UNREAD_WIDTH: usize = 1024;

/// Bytes through `newlines` line breaks. The file stays where the window ends.
///
/// One read covers the guess. Bytes past the last newline are seeked back. A
/// short guess keeps reading and is not treated as the end of the file.
fn read_more(file: &mut std::fs::File, newlines: usize, bytes_per_row: usize) -> (Vec<u8>, bool) {
    if newlines == 0 {
        return (Vec::new(), false);
    }
    let pos = file.stream_position().unwrap_or(0);
    let end = file.metadata().map(|meta| meta.len()).unwrap_or(pos);
    let remain = usize::try_from(end.saturating_sub(pos)).unwrap_or(usize::MAX);
    if remain == 0 {
        return (Vec::new(), true);
    }
    let per = bytes_per_row.max(1);
    // The unread guess stays padded. A measured width is the average so far,
    // plus 64 KB for a longer last line.
    let cover = if per == UNREAD_WIDTH {
        newlines.saturating_mul(per.saturating_mul(5) / 4)
    } else {
        newlines.saturating_mul(per).saturating_add(64 * 1024)
    }
    .max(1)
    .min(remain);
    let mut buf = vec![0u8; cover];
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(_) => break,
        }
    }
    buf.truncate(filled);
    if buf.is_empty() {
        return (buf, true);
    }
    if nth_newline(&buf, newlines).is_some() || buf.len() < cover || cover == remain {
        return trim_newlines(file, buf, newlines);
    }
    let mut tmp = vec![0u8; 8 * 1024 * 1024];
    loop {
        match file.read(&mut tmp) {
            Ok(0) | Err(_) => return (buf, true),
            Ok(count) => buf.extend_from_slice(&tmp[..count]),
        }
        if nth_newline(&buf, newlines).is_some() {
            return trim_newlines(file, buf, newlines);
        }
    }
}

/// Read `newlines` and fold them. Past one step, the previous piece is parsed
/// while the next piece is read. The pieces are joined before this returns.
fn pull_window(
    file: &mut std::fs::File,
    tail: &mut Tail,
    newlines: usize,
    per: usize,
    pred: fn(&str) -> bool,
) {
    const STEP: usize = 16_384;
    if newlines <= STEP {
        let (chunk, eof) = read_more(file, newlines, per);
        fold_window(tail, chunk, pred, eof);
        return;
    }
    let (mut chunk, mut eof) = read_more(file, STEP, per);
    let mut left = newlines - STEP;
    while !eof && left > 0 {
        let take = STEP.min(left);
        let next = std::thread::scope(|scope| {
            let job = scope.spawn(|| fold_window(tail, chunk, pred, false));
            let next = read_more(file, take, per);
            job.join().unwrap();
            next
        });
        chunk = next.0;
        eof = next.1;
        left -= take;
    }
    fold_window(tail, chunk, pred, eof);
}

fn sheet_rows(path: &Path, keep: Keep, pred: fn(&str) -> bool, want: usize) -> Arc<Sheet> {
    // An earlier file in this same picture may have hit EOF. Only this file's end counts.
    set_tail_done(false, |name| path_has_session(path, name));
    let stamp = discover::meta_stamp(path);
    let key = sheet_key(path, stamp, keep);
    if let Some(hit) = cached_sheet(&key) {
        let rows = finished_rows(&hit);
        if want >= rows {
            set_tail_done(true, |name| path_has_session(path, name));
            return hit;
        }
        return Arc::new(clip_sheet(&hit, want));
    }
    if let Some(hit) = sheet_from_window(path, stamp, keep, pred, want) {
        return hit;
    }
    let resume = tails()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&(path.to_path_buf(), keep))
        .is_some_and(|tail| {
            let tail = tail.lock().unwrap_or_else(|err| err.into_inner());
            tail.stamp == stamp && !tail.done && tail.rows > 0
        });
    // A 128-window picture is wider than one timeslice file. Parse it like a
    // one-shot read instead of pulling it through the row window.
    if !resume && file_fits_window(path, want) {
        let sheet = read_and_cache(path, stamp, keep, pred);
        let have = finished_rows(&sheet);
        if want >= have {
            set_tail_done(true, |name| path_has_session(path, name));
            return sheet;
        }
        return Arc::new(clip_sheet(&sheet, want));
    }
    let map_key = (path.to_path_buf(), keep);
    // The map lock only finds the file. Parsing holds that file's lock, so
    // other sessions read at the same time.
    let slot = {
        let mut held = tails().lock().unwrap_or_else(|err| err.into_inner());
        Arc::clone(
            held.entry(map_key)
                .or_insert_with(|| Arc::new(Mutex::new(empty_tail(stamp)))),
        )
    };
    let mut tail = slot.lock().unwrap_or_else(|err| err.into_inner());
    if tail.stamp != stamp {
        *tail = empty_tail(stamp);
    }
    if !tail.done && tail.rows < want {
        if tail.file.is_none() {
            match std::fs::File::open(path) {
                Ok(mut file) => {
                    note_disk_read(path);
                    if tail.cursor > 0 && file.seek(SeekFrom::Start(tail.cursor as u64)).is_err() {
                        tail.done = true;
                    } else {
                        tail.file = Some(file);
                    }
                }
                Err(_) => tail.done = true,
            }
        }
        if !tail.done {
            let add = want - tail.rows;
            let newlines = if tail.cursor == 0 { add + 1 } else { add };
            let per = tail
                .cursor
                .checked_div(tail.rows)
                .unwrap_or(UNREAD_WIDTH)
                .clamp(64, 8 * 1024);
            if let Some(mut file) = tail.file.take() {
                pull_window(&mut file, &mut tail, newlines.max(1), per, pred);
                if !tail.done {
                    tail.file = Some(file);
                }
            }
        }
    }
    if tail.done {
        if tail.pieces.len() == 1 {
            remember_window(path, stamp, Arc::clone(&tail.pieces[0]), tail.rows, true);
        }
        set_tail_done(true, |name| path_has_session(path, name));
        tail.file = None;
        let finished = Arc::clone(&tail.sheet);
        let have = tail.rows;
        drop(tail);
        let mut cache = file_sheets().lock().unwrap_or_else(|err| err.into_inner());
        if cache.len() >= FILE_SHEET_CAP {
            evict_one(&mut cache);
        }
        cache.insert(
            SheetKey {
                path: path.to_path_buf(),
                stamp,
                keep,
            },
            Arc::clone(&finished),
        );
        return shared_rows(&finished, want.min(have), have);
    }
    let shown = tail.rows.min(want);
    shared_rows(&tail.sheet, shown, tail.rows)
}

fn shared_rows(sheet: &Arc<Sheet>, rows: usize, have: usize) -> Arc<Sheet> {
    if rows >= have {
        Arc::clone(sheet)
    } else {
        Arc::new(clip_sheet(sheet, rows))
    }
}

fn finished_rows(sheet: &Sheet) -> usize {
    sheet
        .num
        .values()
        .map(Vec::len)
        .chain(sheet.wide.values().map(Vec::len))
        .max()
        .unwrap_or(0)
}

fn clip_sheet(sheet: &Sheet, rows: usize) -> Sheet {
    Sheet {
        num: sheet
            .num
            .iter()
            .map(|(name, values)| (name.clone(), values.iter().take(rows).copied().collect()))
            .collect(),
        wide: sheet
            .wide
            .iter()
            .map(|(name, values)| (name.clone(), values.iter().take(rows).copied().collect()))
            .collect(),
        block: sheet.block.clone(),
    }
}

fn extend_sheet(dest: &mut Sheet, extra: &Sheet) {
    if dest.block.is_empty() {
        dest.block.clone_from(&extra.block);
    }
    for (name, values) in &extra.num {
        dest.num
            .entry(name.clone())
            .or_default()
            .extend_from_slice(values);
    }
    for (name, values) in &extra.wide {
        dest.wide
            .entry(name.clone())
            .or_default()
            .extend(values.iter().copied());
    }
}

fn header_end(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .map(|index| index + 1)
        .unwrap_or(bytes.len())
}

fn fold_window(tail: &mut Tail, chunk: Vec<u8>, pred: fn(&str) -> bool, file_eof: bool) {
    if chunk.is_empty() {
        tail.done = file_eof;
        return;
    }
    tail.joined = None;
    let end = if tail.cursor == 0 {
        header_end(&chunk)
    } else {
        0
    };
    if tail.cursor == 0 && tail.header.is_empty() {
        tail.header = chunk[..end].to_vec();
    }
    let len = chunk.len();
    tail.pieces.push(Arc::new(chunk));
    adopt_layout(tail, pred);
    let Some(layout) = tail.layout.clone() else {
        tail.cursor += len;
        tail.done = true;
        return;
    };
    let stored = tail.pieces.last().unwrap();
    let body = if end == 0 {
        stored.as_slice()
    } else {
        &stored[end..]
    };
    let text = String::from_utf8_lossy(body);
    let extra = sheet_from(
        layout.names,
        slots_from_body(text.as_ref(), &layout.kinds),
        layout.block,
    );
    tail.rows += finished_rows(&extra);
    extend_shared(&mut tail.sheet, &extra);
    tail.cursor += len;
    tail.done = file_eof;
}

fn adopt_layout(tail: &mut Tail, pred: fn(&str) -> bool) {
    if tail.layout.is_some() {
        return;
    }
    let header = std::str::from_utf8(&tail.header).unwrap_or("");
    let header = header.trim_end_matches(['\r', '\n']);
    if header.is_empty() {
        return;
    }
    let (names, kinds, block) = header_schema(header, &pred);
    tail.layout = Some(Layout {
        names,
        kinds,
        block,
    });
}

fn extend_shared(dest: &mut Arc<Sheet>, extra: &Sheet) {
    extend_sheet(Arc::make_mut(dest), extra);
}

fn read_sheet_where(bytes: &[u8], keep: impl Fn(&str) -> bool + Sync) -> Sheet {
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => std::borrow::Cow::Borrowed(text),
        Err(_) => String::from_utf8_lossy(bytes),
    };
    let (header, body) = text.split_once('\n').unwrap_or((text.as_ref(), ""));
    let header = header.trim_end_matches('\r');
    if header.is_empty() {
        return empty_sheet();
    }
    let (names, kinds, block) = header_schema(header, &keep);
    sheet_from(names, slots_from_body(body, &kinds), block)
}

fn slots_from_body(body: &str, kinds: &[Kind]) -> Vec<Slot> {
    let parts = body_parts(body);
    if parts.len() < 2 {
        let mut slots = fresh_slots(kinds);
        fill_rows(body, &mut slots);
        return slots;
    }
    let mut slots = Vec::new();
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for part in parts {
            joins.push(scope.spawn(move || {
                let mut slots = fresh_slots(kinds);
                fill_rows(part, &mut slots);
                slots
            }));
        }
        for join in joins {
            let part = join.join().unwrap();
            if slots.is_empty() {
                slots = part;
            } else {
                append_slots(&mut slots, part);
            }
        }
    });
    slots
}

fn header_schema(
    header: &str,
    keep: &impl Fn(&str) -> bool,
) -> (Vec<Option<String>>, Vec<Kind>, BTreeMap<String, u32>) {
    let mut used = HashMap::<String, ()>::new();
    let mut names = Vec::new();
    let mut kinds = Vec::new();
    let mut block_of = BTreeMap::new();
    let mut block = 0u32;
    let mut spots = 0u32;
    let mut cursor = 0usize;
    while let Some((start, end, next)) = next_cell(header, cursor) {
        let name = cell_owned(header, start, end);
        let base = header_base(&name);
        if base.eq_ignore_ascii_case("spot_no") {
            if spots > 0 {
                block += 1;
            }
            spots += 1;
        }
        // ponytail: strip columns are not a binned metric. Stop skipping them if one is.
        if name.is_empty() || base.to_ascii_lowercase().contains("strip") || !keep(&name) {
            names.push(None);
            kinds.push(Kind::Skip);
        } else {
            let key = unique_header(&name, &mut used);
            let kind = if base == "datetime" {
                Kind::Clock
            } else if matches!(base, "timestamp" | "time_s" | "time_ns") {
                Kind::Wide
            } else {
                Kind::Num
            };
            block_of.insert(key.clone(), block);
            names.push(Some(key));
            kinds.push(kind);
        }
        if next <= cursor {
            break;
        }
        cursor = next;
    }
    (names, kinds, block_of)
}

fn fresh_slots(kinds: &[Kind]) -> Vec<Slot> {
    kinds
        .iter()
        .map(|kind| match kind {
            Kind::Skip => Slot::Skip,
            Kind::Num => Slot::Num(Vec::new()),
            Kind::Wide => Slot::Wide(Vec::new()),
            Kind::Clock => Slot::Clock(Vec::new()),
        })
        .collect()
}

fn body_parts(body: &str) -> Vec<&str> {
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(2, 8);
    if body.len() < PARALLEL_BYTES {
        return vec![body];
    }
    let bytes = body.as_bytes();
    let mut starts = vec![0usize];
    let step = body.len() / workers;
    for index in 1..workers {
        let mut at = index * step;
        while at < bytes.len() && bytes[at] != b'\n' {
            at += 1;
        }
        if at < bytes.len() {
            at += 1;
        }
        if at > *starts.last().unwrap_or(&0) && at < bytes.len() {
            starts.push(at);
        }
    }
    let mut parts = Vec::new();
    for pair in starts.windows(2) {
        parts.push(&body[pair[0]..pair[1]]);
    }
    parts.push(&body[*starts.last().unwrap_or(&0)..]);
    parts
}

fn fill_rows(body: &str, slots: &mut [Slot]) {
    if slots
        .iter()
        .all(|slot| matches!(slot, Slot::Skip | Slot::Num(_)))
        && fill_plain(body.as_bytes(), slots)
    {
        return;
    }
    fill_rows_general(body, slots);
}

/// Unquoted numeric rows. Returns false when a quote means the general parser
/// has to see the whole body. `slots` is cleared on that failure.
fn fill_plain(bytes: &[u8], slots: &mut [Slot]) -> bool {
    let mut seen = vec![0u32; slots.len()];
    let mut row = 1u32;
    let mut i = 0usize;
    while i < bytes.len() {
        let end = bytes[i..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|at| i + at)
            .unwrap_or(bytes.len());
        let line = &bytes[i..end];
        i = if end < bytes.len() { end + 1 } else { end };
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let mut cursor = 0usize;
        let mut index = 0usize;
        while index < slots.len() {
            let Some(field) = take_plain(line, cursor) else {
                break;
            };
            let Plain::Cell(cell, next) = field else {
                clear_slots(slots);
                return false;
            };
            match &mut slots[index] {
                Slot::Skip => {}
                Slot::Num(values) => values.push(f32_bytes(cell)),
                Slot::Wide(_) | Slot::Clock(_) => {
                    clear_slots(slots);
                    return false;
                }
            }
            seen[index] = row;
            index += 1;
            if next <= cursor {
                break;
            }
            cursor = next;
        }
        close_row(slots, &mut seen, &mut row);
    }
    true
}

enum Plain<'a> {
    Cell(&'a [u8], usize),
    Quote,
}

fn take_plain(line: &[u8], start: usize) -> Option<Plain<'_>> {
    if start > line.len() {
        return None;
    }
    if start == line.len() {
        if start > 0 && line[start - 1] == b',' {
            return Some(Plain::Cell(&line[start..start], start + 1));
        }
        return None;
    }
    let mut i = start;
    while i < line.len() && line[i] != b',' {
        if line[i] == b'"' {
            return Some(Plain::Quote);
        }
        i += 1;
    }
    let next = if i < line.len() { i + 1 } else { i };
    Some(Plain::Cell(&line[start..i], next))
}

fn clear_slots(slots: &mut [Slot]) {
    for slot in slots {
        match slot {
            Slot::Skip => {}
            Slot::Num(values) => values.clear(),
            Slot::Wide(values) | Slot::Clock(values) => values.clear(),
        }
    }
}

fn fill_rows_general(body: &str, slots: &mut [Slot]) {
    let mut seen = vec![0u32; slots.len()];
    let mut row = 1u32;
    for line in body.lines() {
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
        close_row(slots, &mut seen, &mut row);
    }
}

fn close_row(slots: &mut [Slot], seen: &mut [u32], row: &mut u32) {
    for (index, slot) in slots.iter_mut().enumerate() {
        if seen[index] != *row {
            push_nan(slot);
        }
    }
    *row = row.wrapping_add(1);
    if *row == 0 {
        *row = 1;
        seen.fill(0);
    }
}

fn append_slots(base: &mut [Slot], extra: Vec<Slot>) {
    for (slot, more) in base.iter_mut().zip(extra) {
        match (slot, more) {
            (Slot::Num(values), Slot::Num(more)) => values.extend(more),
            (Slot::Wide(values), Slot::Wide(more) | Slot::Clock(more)) => values.extend(more),
            (Slot::Clock(values), Slot::Clock(more) | Slot::Wide(more)) => values.extend(more),
            _ => {}
        }
    }
}

fn sheet_from(names: Vec<Option<String>>, slots: Vec<Slot>, block: BTreeMap<String, u32>) -> Sheet {
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
    Sheet { num, wide, block }
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
        let mut buf = [0u8; 64];
        if let Some(raw) = unquote_stack(cell, &mut buf) {
            return f32_bytes(raw);
        }
        return f32_bytes(strip_quotes(cell).as_bytes());
    }
    f32_bytes(cell.as_bytes())
}

/// Copy a short quoted cell into `buf` without allocating.
/// A cell longer than the buffer returns `None`.
fn unquote_stack<'a>(cell: &str, buf: &'a mut [u8; 64]) -> Option<&'a [u8]> {
    let bytes = cell.as_bytes();
    let mut n = 0usize;
    let mut i = 1usize;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                if n == buf.len() {
                    return None;
                }
                buf[n] = b'"';
                n += 1;
                i += 2;
                continue;
            }
            break;
        }
        if n == buf.len() {
            return None;
        }
        buf[n] = bytes[i];
        n += 1;
        i += 1;
    }
    Some(&buf[..n])
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
        let mut buf = [0u8; 64];
        if let Some(raw) = unquote_stack(cell, &mut buf) {
            return f64_bytes(raw);
        }
        return f64_bytes(strip_quotes(cell).as_bytes());
    }
    f64_bytes(cell.as_bytes())
}

fn f64_bytes(bytes: &[u8]) -> f64 {
    match parse_decimal(bytes) {
        Num::Nan => f64::NAN,
        Num::Value(value) => value,
        Num::Std => std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(f64::NAN),
    }
}

fn cell_datetime(line: &str, start: usize, end: usize) -> f64 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        let mut buf = [0u8; 64];
        if let Some(raw) = unquote_stack(cell, &mut buf) {
            let text = std::str::from_utf8(raw).unwrap_or("");
            return parse_datetime(text).unwrap_or(f64::NAN);
        }
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

    /// `bind_slice` is one process-wide slot. The tests that use it take turns.
    fn slice_gate() -> &'static Mutex<()> {
        static GATE: OnceLock<Mutex<()>> = OnceLock::new();
        GATE.get_or_init(|| Mutex::new(()))
    }

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
        let quoted = read_sheet(b"energy,dose\r\n\"70.5\",2\r\n3\r\n");
        assert!((quoted.num["energy"][0] - 70.5).abs() < 1e-4);
        assert!((quoted.num["dose"][0] - 2.0).abs() < 1e-6);
        assert!((quoted.num["energy"][1] - 3.0).abs() < 1e-6);
        assert!(quoted.num["dose"][1].is_nan());
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
        assert_eq!(
            first.get("r_ic1_current_dose").map(Vec::as_slice),
            Some([1.0].as_slice())
        );
        assert_eq!(
            merged_timeslice(&root, "sess")
                .get("r_ic1_current_dose")
                .map(Vec::as_slice),
            Some([1.0].as_slice())
        );
        std::fs::write(&file, "r_ic1_current_dose,rci_in_trigger\n4,1\n5,1\n").unwrap();
        assert_eq!(
            merged_timeslice(&root, "sess")
                .get("r_ic1_current_dose")
                .map(Vec::as_slice),
            Some([4.0, 5.0].as_slice())
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_rewritten_header_is_read_again() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-header-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        let spot = session.join("spot_data.csv");
        std::fs::write(&spot, "energy,ic1_total_dose\n").unwrap();
        let first = grain_columns(&root, "sess", false);
        assert!(first.iter().any(|name| name == "energy"));
        assert!(first.iter().all(|name| name != "r_ic1_x_spot_sigma"));
        std::fs::write(&spot, "energy,ic1_total_dose,r_ic1_x_spot_sigma\n").unwrap();
        let second = grain_columns(&root, "sess", false);
        assert!(second.iter().any(|name| name == "r_ic1_x_spot_sigma"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_large_spot_file_parses_in_order() {
        let row = "1.5,2.5\n";
        let rows = PARALLEL_BYTES / row.len() + 8;
        let mut text = String::from("a,b\n");
        for _ in 0..rows {
            text.push_str(row);
        }
        let sheet = read_sheet(text.as_bytes());
        let column = sheet.num.get("a").map(Vec::as_slice).unwrap_or(&[]);
        assert_eq!(column.len(), rows);
        assert!(column.iter().all(|value| (*value - 1.5).abs() < 1.0e-5));
        assert_eq!(sheet.num.get("b").map(Vec::len), Some(rows));
    }

    #[test]
    fn timeslice_position_error_uses_each_controllers_spot() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-slice-clock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        let mut map = String::from("energy,layer_id,spot_no,position_x,position_y\n");
        for spot in 0..16 {
            map.push_str(&format!("70,1,{spot},{spot},{}\n", spot as f32 * 0.5));
        }
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        // IC3 leads, then IC1, then IC2. On one row IC2 is three spots ahead of IC1,
        // and IC3's layer is not the layer either chamber is delivering.
        let mut slice = String::from(
            "spot_no,layer_id,spot_no,layer_id,r_ic1_x_position,r_ic1_y_position,ic1_position_x_target,ic1_position_y_target,spot_no,layer_id,r_ic2_x_position,r_ic2_y_position,ic2_position_x_target,ic2_position_y_target,rci_in_trigger\n",
        );
        for spot in 0..12 {
            let ic2 = spot + 3;
            let ic1_x = spot as f32 + 64.5;
            let ic1_y = spot as f32 * 0.5 + 64.5;
            let ic2_x = ic2 as f32 + 64.5;
            let ic2_y = ic2 as f32 * 0.5 + 64.5;
            slice.push_str(&format!(
                "0,99,{spot},1,{ic1_x},{ic1_y},{ic1_x},{ic1_y},{ic2},1,{ic2_x},{ic2_y},{ic2_x},{ic2_y},1\n"
            ));
        }
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            slice,
        )
        .unwrap();
        let table = load_slice_metric(&root, "sess", "position_error", false);
        let near = |values: &[f32]| {
            values
                .iter()
                .all(|value| value.is_finite() && value.abs() < 1e-2)
        };
        assert!(near(table.get("ic1_x_err").unwrap()));
        assert!(near(table.get("ic1_y_err").unwrap()));
        assert!(near(table.get("ic2_x_err").unwrap()));
        assert!(near(table.get("ic2_y_err").unwrap()));
        let plan_x = table.get("plan_x").unwrap();
        let plan_y = table.get("plan_y").unwrap();
        assert!((plan_x[5] - 5.0).abs() < 1e-3);
        assert!((plan_y[5] - 2.5).abs() < 1e-3);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_long_timeslice_keeps_every_sample() {
        let root = std::env::temp_dir().join(format!("scan-kit-slice-all-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,layer_id\n70,1\n").unwrap();
        let rows = 80_001usize;
        let mut text = String::from(
            "layer_id,rci_in_trigger,r_ic1_x_position,r_ic1_x_confidence,ic1_x_fit_ok,r_ic1_x_spot_error_code,ic1_primary_channel\n",
        );
        for _ in 0..rows {
            text.push_str("1,1,64,100,1,0,12\n");
        }
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            text,
        )
        .unwrap();
        let position = load_slice_metric(&root, "sess", "position_error", true);
        assert_eq!(position.get("ic1_x").map(Vec::len), Some(rows));
        let current = load_timeslice(&root, "sess", "ic_current");
        assert_eq!(current.get("ic1_current").map(Vec::len), Some(rows));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sample_spot_and_layer_clocks_start_where_the_playhead_does() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-clock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::create_dir_all(session.join("layer-1/run-0")).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y,layer_id\n70,1,0,0,1\n90,2,1,0,1\n",
        )
        .unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "r_ic1_current_dose,rci_in_trigger\n1,1\n1,1\n1,1\n1,1\n",
        )
        .unwrap();
        std::fs::write(
            session.join("layer-1/run-0/timeslice_data_device_units.csv"),
            "r_ic1_current_dose,rci_in_trigger\n1,1\n1,1\n",
        )
        .unwrap();
        let samples = load_timeslice(&root, "sess", "ic_current");
        let time = samples.get("time_s").unwrap();
        assert_eq!(time.len(), 6);
        assert_eq!(time[0], 0.0);
        assert!((time[5] - 0.005).abs() < 1e-6);

        let layers = load_timeslice(&root, "sess", "current_ratio");
        let layer_time = layers.get("time_s").unwrap();
        assert_eq!(layer_time.len(), 2);
        assert!((layer_time[0] - 0.004).abs() < 1e-6);
        assert!((layer_time[1] - 0.006).abs() < 1e-6);

        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,timestamp,layer_id\n1,1000,1\n2,2500,2\n",
        )
        .unwrap();
        let spots = load_spot(&root, "sess", false, false, false);
        let spot_time = spots.get("time_s").unwrap();
        assert_eq!(spot_time.len(), 2);
        assert_eq!(spot_time[0], 0.0);
        assert!((spot_time[1] - 1.5).abs() < 1e-4);
        let slice_marks = timeslice_layer_marks(&root, "sess");
        assert_eq!(slice_marks.len(), 1);
        assert!((slice_marks[0] - 0.004).abs() < 1e-6);
        let spot_marks = spot_layer_marks(&root, "sess");
        assert_eq!(spot_marks.len(), 1);
        assert!((spot_marks[0] - 1.5).abs() < 1e-4);

        let bare = root.join("nostamp");
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::write(
            bare.join("input_map.csv"),
            "energy,charge_req,position_x,position_y,layer_id\n70,1,0,0,1\n90,2,1,0,1\n",
        )
        .unwrap();
        std::fs::write(
            bare.join("spot_data.csv"),
            "ic1_total_dose,layer_id\n1,1\n2,1\n",
        )
        .unwrap();
        let untimed = load_spot(&root, "nostamp", false, false, false);
        assert!(untimed.get("energy").is_some());
        assert!(untimed.get("time_s").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_slice_keeps_a_prefix_then_the_rest_of_the_file() {
        let _gate = slice_gate().lock().unwrap_or_else(|err| err.into_inner());
        let root = std::env::temp_dir().join(format!(
            "scan-kit-slice-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Other tests load a session named "sess" and would publish this slice's tail.
        let session = root.join("slice");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,2,4,1\n110,3,8,2\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,position_x,position_y\n1,1,0,0\n2,2,4,1\n3,3,8,2\n",
        )
        .unwrap();
        bind_slice("slice", SliceTake::Spot { rows: 2 });
        let prefix = load_spot(&root, "slice", false, false, false);
        assert_eq!(prefix.get("ic1_dose").map(Vec::len), Some(2));
        assert!(!slice_tail_done());
        clear_slice();
        bind_slice("slice", SliceTake::Spot { rows: 8 });
        let rest = load_spot(&root, "slice", false, false, false);
        assert_eq!(
            rest.get("ic1_dose").map(Vec::as_slice),
            Some([1.0, 2.0, 3.0].as_slice())
        );
        assert!(slice_tail_done());
        clear_slice();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_growing_timeslice_matches_the_full_table() {
        let _gate = slice_gate().lock().unwrap_or_else(|err| err.into_inner());
        let root = std::env::temp_dir().join(format!(
            "scan-kit-grow-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("grow");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::create_dir_all(session.join("layer-1/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy\n70\n90\n").unwrap();
        for (layer, values) in [(0, "1\n2\n3\n4\n"), (1, "5\n6\n7\n8\n")] {
            std::fs::write(
                session.join(format!(
                    "layer-{layer}/run-0/timeslice_data_device_units.csv"
                )),
                format!("r_ic1_current_dose\n{values}"),
            )
            .unwrap();
        }
        for (files, tail_rows) in [(0, 2), (0, 4), (1, 2), (2, 1)] {
            bind_slice("grow", SliceTake::Timeslice { files, tail_rows });
            let _ = load_timeslice(&root, "grow", "ic_current");
            let _ = timeslice_signals(&root, "grow");
            let _ = load_slice_metric(&root, "grow", "position_error", false);
        }
        let grown = load_timeslice(&root, "grow", "ic_current");
        let signals = timeslice_signals(&root, "grow");
        let position = load_slice_metric(&root, "grow", "position_error", false);
        clear_slice();
        let full = load_timeslice(&root, "grow", "ic_current");
        let full_signals = timeslice_signals(&root, "grow");
        let full_position = load_slice_metric(&root, "grow", "position_error", false);
        let same = |left: &BTreeMap<String, Vec<f32>>, right: &BTreeMap<String, Vec<f32>>| {
            left.len() == right.len()
                && left.iter().all(|(key, values)| {
                    right.get(key).is_some_and(|other| {
                        values.len() == other.len()
                            && values
                                .iter()
                                .zip(other)
                                .all(|(left, right)| left.to_bits() == right.to_bits())
                    })
                })
        };
        assert!(same(grown.as_ref(), full.as_ref()));
        assert!(same(signals.as_ref(), full_signals.as_ref()));
        assert!(same(position.as_ref(), full_position.as_ref()));
        assert_eq!(full.get("ic1_current").map(Vec::len), Some(8));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `SCAN_KIT_SESSION` is the session folder that contains `input_map.csv`.
    ///
    /// ```text
    /// cargo test -p scan-kit-io --release --lib -- --ignored session_load_bench --nocapture
    /// ```
    #[test]
    #[ignore]
    fn session_load_bench() {
        let dir = std::path::PathBuf::from(
            std::env::var("SCAN_KIT_SESSION").expect("set SCAN_KIT_SESSION to a session folder"),
        );
        let session = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let root = dir.parent().expect("session folder has a parent");
        let spot_path = dir.join("spot_data.csv");
        let paths = crate::discover::list_timeslice_paths(&dir);
        let started = std::time::Instant::now();
        let spot_bytes = std::fs::read(&spot_path).unwrap_or_default();
        let mut slice_files = Vec::with_capacity(paths.len());
        let mut slice_bytes = 0usize;
        for (_, path) in &paths {
            let bytes = std::fs::read(path).unwrap_or_default();
            slice_bytes += bytes.len();
            slice_files.push(bytes);
        }
        let read = started.elapsed();
        let started = std::time::Instant::now();
        let spot_sheet = read_sheet_where(&spot_bytes, spot_keep);
        let parse_spot = started.elapsed();
        let started = std::time::Instant::now();
        let _signals = read_sheets(&slice_files, timeline_column);
        let parse_signals = started.elapsed();
        let started = std::time::Instant::now();
        let _geometry = read_sheets(&slice_files, slice_column);
        let parse_geometry = started.elapsed();
        let started = std::time::Instant::now();
        let spot = load_spot(root, &session, false, false, false);
        let cold_spot = started.elapsed();
        let started = std::time::Instant::now();
        let current = load_timeslice(root, &session, "ic_current");
        let cold_current = started.elapsed();
        let started = std::time::Instant::now();
        let position = load_slice_metric(root, &session, "position_error", false);
        let cold_position = started.elapsed();
        let started = std::time::Instant::now();
        let _ = load_spot(root, &session, false, false, false);
        let hit_spot = started.elapsed();
        let started = std::time::Instant::now();
        let _ = load_timeslice(root, &session, "ic_current");
        let hit_current = started.elapsed();
        let started = std::time::Instant::now();
        let _ = load_slice_metric(root, &session, "position_error", false);
        let hit_position = started.elapsed();
        let mb = |bytes: usize, time: std::time::Duration| {
            if time.as_secs_f64() == 0.0 {
                0.0
            } else {
                bytes as f64 / time.as_secs_f64() / 1.0e6
            }
        };
        println!(
            "spot bytes {} columns {} rows {} | timeslice files {} bytes {}",
            spot_bytes.len(),
            spot_sheet.num.len(),
            spot_sheet.num.values().map(Vec::len).max().unwrap_or(0),
            paths.len(),
            slice_bytes
        );
        println!(
            "read {:.1} ms ({:.0} MB/s)",
            read.as_secs_f64() * 1.0e3,
            mb(spot_bytes.len() + slice_bytes, read)
        );
        println!(
            "parse spot {:.1} ms ({:.0} MB/s), signals {:.1} ms ({:.0} MB/s), geometry {:.1} ms ({:.0} MB/s)",
            parse_spot.as_secs_f64() * 1.0e3,
            mb(spot_bytes.len(), parse_spot),
            parse_signals.as_secs_f64() * 1.0e3,
            mb(slice_bytes, parse_signals),
            parse_geometry.as_secs_f64() * 1.0e3,
            mb(slice_bytes, parse_geometry)
        );
        println!(
            "cold spot {:.1} ms rows {} | current {:.1} ms rows {} | position {:.1} ms rows {}",
            cold_spot.as_secs_f64() * 1.0e3,
            spot.get("energy").map(Vec::len).unwrap_or(0),
            cold_current.as_secs_f64() * 1.0e3,
            current.get("energy").map(Vec::len).unwrap_or(0),
            cold_position.as_secs_f64() * 1.0e3,
            position.get("energy").map(Vec::len).unwrap_or(0)
        );
        println!(
            "cache hit spot {:.3} ms | current {:.3} ms | position {:.3} ms",
            hit_spot.as_secs_f64() * 1.0e3,
            hit_current.as_secs_f64() * 1.0e3,
            hit_position.as_secs_f64() * 1.0e3
        );
    }

    #[test]
    fn a_long_window_keeps_row_order_across_the_read_ahead_split() {
        let path = std::env::temp_dir().join(format!(
            "scan-kit-window-{}-{}.csv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pad = "x".repeat(64);
        let mut body = String::from("layer_id,ic1_current,pad\n");
        for row in 0..40_000 {
            body.push_str(&format!("0,{row},{pad}\n"));
        }
        std::fs::write(&path, body).unwrap();
        // End-of-file is process-global. This test only checks row order.
        let _quiet = QuietTail::suppress();
        let prefix = sheet_rows(&path, Keep::Current, current_column, 4_096);
        let mid = sheet_rows(&path, Keep::Current, current_column, 24_096);
        let rest = sheet_rows(&path, Keep::Current, current_column, 80_000);
        let same = |sheet: &Sheet, rows: usize, indexes: &[usize]| {
            let column = &sheet.num["ic1_current"];
            assert_eq!(column.len(), rows);
            for index in indexes {
                assert_eq!(column[*index], *index as f32);
            }
        };
        same(&prefix, 4_096, &[0, 4_095]);
        same(&mid, 24_096, &[16_383, 16_384, 20_479, 20_480]);
        same(&rest, 40_000, &[32_767, 32_768, 39_999]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_views_share_one_stored_column() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-store-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("share");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,layer_id\n70,1\n").unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,rci_in_trigger,r_ic1_current_dose,r_ic1_x_position\n1,1,4,10\n1,1,5,11\n",
        )
        .unwrap();
        let first = session_columns(&root, "share", Grain::Sample, &["ic1_current"]);
        let second = session_columns(&root, "share", Grain::Sample, &["ic1_current"]);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.get("ic1_current").map(Vec::len), Some(2));
        let position = session_columns(&root, "share", Grain::Sample, &["ic1_x"]);
        let again = session_columns(&root, "share", Grain::Sample, &["ic1_current", "ic1_x"]);
        assert!(Arc::ptr_eq(&position, &again));
        assert!(again.get("ic1_current").is_some());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_blank_signal_row_stays_on_the_trigger_clock() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-blank-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("blank");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,layer_id\n70,1\n").unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "r_ic1_x_confidence,rci_in_trigger\n1,1\n,1\n3,1\n",
        )
        .unwrap();
        let table = load_timeslice(&root, "blank", "fit_confidence");
        let column = table.get("ic1_x_confidence").expect("confidence");
        assert_eq!(column.len(), 3);
        assert!(column[1].is_nan());
        assert_eq!(column[0], 1.0);
        assert_eq!(column[2], 3.0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_later_family_parses_the_window_already_read() {
        let _quiet = QuietTail::suppress();
        let path = std::env::temp_dir().join(format!(
            "scan-kit-bytes-{}-{}.csv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut body =
            String::from("layer_id,rci_in_trigger,r_ic1_current_dose,r_ic1_x_position\n");
        for row in 0..8_000 {
            body.push_str(&format!("1,1,{row},{row}\n"));
        }
        std::fs::write(&path, &body).unwrap();
        DISK_READS.with(|reads| reads.set(0));
        let current = sheet_rows(&path, Keep::Current, current_column, 32);
        assert_eq!(current.num["r_ic1_current_dose"].len(), 32);
        let reads = DISK_READS.with(Cell::get);
        assert!(reads >= 1);
        let geometry = sheet_rows(&path, Keep::Geometry, slice_column, 32);
        assert_eq!(DISK_READS.with(Cell::get), reads);
        assert_eq!(geometry.num["r_ic1_x_position"].len(), 32);
        assert_eq!(geometry.num["r_ic1_x_position"][31], 31.0);
        let held = tails().lock().unwrap_or_else(|err| err.into_inner());
        assert!(held.get(&(path.clone(), Keep::Geometry)).is_none());
        drop(held);
        let _ = std::fs::remove_file(&path);
    }
}
