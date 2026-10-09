//! One driver for every long desktop task.
//!
//! A poll publishes one picture and may return a plot payload. The first timed
//! poll is one window. Each later timed poll doubles that window, still
//! publishing every time, and reads only the new bytes. `open_plot` drains the
//! same stage with an unlimited budget, so a one-shot call still returns the
//! scene it returns today.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, UNIX_EPOCH};

use scan_kit_core::{encode_poll, Cancel, McJob, McResult, Phase, PlotScene, Poll, Report, Volume};
use scan_kit_plot::encode_plot_reusing;
use serde_json::{json, Value};

use crate::mc::McRun;
use crate::present::apply_palette;
use crate::ComputeError;

const POLL_BUDGET: Duration = Duration::from_millis(12);
const DRAIN: Duration = Duration::from_secs(30);

trait Stage: Send {
    fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>>;
    fn report(&self) -> Report;
    fn retarget(&mut self, _options: &Value, _palette: &[[f32; 4]]) -> bool {
        false
    }
}

struct Running {
    generation: u64,
    cancel: Cancel,
    /// Taken out for the duration of a poll so cancel and a new start do not
    /// wait on the encode.
    stage: Option<Box<dyn Stage>>,
}

#[derive(Default)]
struct Driver {
    next: u64,
    plot: Option<u64>,
    library: Option<u64>,
    copy: Option<u64>,
    tasks: HashMap<u64, Running>,
}

fn driver() -> &'static Mutex<Driver> {
    static DRIVER: OnceLock<Mutex<Driver>> = OnceLock::new();
    DRIVER.get_or_init(|| {
        Mutex::new(Driver {
            next: 1,
            ..Driver::default()
        })
    })
}

fn lock() -> std::sync::MutexGuard<'static, Driver> {
    driver().lock().unwrap_or_else(|err| err.into_inner())
}

enum Slot {
    Plot,
    Library,
    Copy,
}

impl Driver {
    fn slot_id(&self, slot: &Slot) -> Option<u64> {
        match slot {
            Slot::Plot => self.plot,
            Slot::Library => self.library,
            Slot::Copy => self.copy,
        }
    }

    fn set_slot(&mut self, slot: &Slot, id: Option<u64>) {
        let cell = match slot {
            Slot::Plot => &mut self.plot,
            Slot::Library => &mut self.library,
            Slot::Copy => &mut self.copy,
        };
        *cell = id;
    }

    fn clear(&mut self, id: u64) {
        for cell in [&mut self.plot, &mut self.library, &mut self.copy] {
            if *cell == Some(id) {
                *cell = None;
            }
        }
    }

    fn take(&mut self, slot: &Slot) {
        let Some(id) = self.slot_id(slot) else {
            return;
        };
        if let Some(task) = self.tasks.get(&id) {
            task.cancel.cancel();
        }
        self.tasks.remove(&id);
        self.set_slot(slot, None);
    }
}

pub fn start_task(
    view: &str,
    path: &Path,
    sessions: &[String],
    options: &Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: &[[f32; 4]],
) -> Result<Value, String> {
    let mut driver = lock();
    let slot = slot_of(view);
    if matches!(slot, Slot::Plot) {
        if let Some(id) = driver.plot {
            if let Some(task) = driver.tasks.get_mut(&id) {
                if let Some(stage) = task.stage.as_mut() {
                    if !task.cancel.is_cancelled() && stage.retarget(options, palette) {
                        task.generation = task.generation.saturating_add(1);
                        return Ok(json!({ "task": id, "generation": task.generation }));
                    }
                }
            }
        }
    }
    driver.take(&slot);
    let stage = build_stage(
        view, path, sessions, options, background, foreground, palette,
    );
    let id = driver.next;
    driver.next = driver.next.saturating_add(1);
    driver.set_slot(&slot, Some(id));
    driver.tasks.insert(
        id,
        Running {
            generation: 1,
            cancel: Cancel::new(),
            stage: Some(stage),
        },
    );
    Ok(json!({ "task": id, "generation": 1 }))
}

pub fn poll_task(id: u64) -> Result<Vec<u8>, String> {
    let (mut stage, cancel, generation) = {
        let mut driver = lock();
        let Some(task) = driver.tasks.get_mut(&id) else {
            return Ok(encode_poll(&ended(id, 0, Phase::Cancelled, ""), None, true));
        };
        let Some(stage) = task.stage.take() else {
            return Ok(encode_poll(
                &ended(id, task.generation, Phase::Scene, ""),
                None,
                false,
            ));
        };
        (stage, task.cancel.clone(), task.generation)
    };
    let outcome = stage.poll(POLL_BUDGET, &cancel);
    let mut report = stage.report();
    report.task = id;
    report.generation = generation;
    let (finished, payload, remove) = match outcome {
        Poll::Pending { preview, .. } => (false, preview, false),
        Poll::Ready(bytes) => {
            report.phase = Phase::Done;
            (true, Some(bytes), true)
        }
        Poll::Cancelled => {
            report.phase = Phase::Cancelled;
            report.note.clear();
            (true, None, true)
        }
        Poll::Failed(message) => {
            report.phase = Phase::Failed;
            if report.note.is_empty() {
                report.note = message;
            }
            (true, None, true)
        }
    };
    let mut driver = lock();
    let Some(current) = driver.tasks.get(&id).map(|task| task.generation) else {
        return Ok(encode_poll(
            &ended(id, generation, Phase::Cancelled, ""),
            None,
            true,
        ));
    };
    if current != generation {
        if let Some(task) = driver.tasks.get_mut(&id) {
            task.stage = Some(stage);
        }
        return Ok(encode_poll(
            &ended(id, generation, Phase::Cancelled, ""),
            None,
            true,
        ));
    }
    if remove {
        driver.tasks.remove(&id);
        driver.clear(id);
    } else if let Some(task) = driver.tasks.get_mut(&id) {
        task.stage = Some(stage);
    }
    Ok(encode_poll(&report, payload.as_deref(), finished))
}

pub fn cancel_task(id: u64) {
    let mut driver = lock();
    if let Some(task) = driver.tasks.get(&id) {
        task.cancel.cancel();
    }
    driver.tasks.remove(&id);
    driver.clear(id);
}

pub(crate) fn drain_plot(
    view: &str,
    root: &Path,
    sessions: &[String],
    options: &Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: &[[f32; 4]],
) -> Result<Vec<u8>, String> {
    let mut stage = build_stage(
        view, root, sessions, options, background, foreground, palette,
    );
    match stage.poll(Duration::MAX, &Cancel::new()) {
        Poll::Ready(bytes) => Ok(bytes),
        Poll::Failed(message) => Err(message),
        Poll::Cancelled => Err("cancelled".into()),
        Poll::Pending { .. } => Err("plot did not finish".into()),
    }
}

fn slot_of(view: &str) -> Slot {
    match view {
        "library" => Slot::Library,
        "runner_copy" => Slot::Copy,
        _ => Slot::Plot,
    }
}

fn build_stage(
    view: &str,
    path: &Path,
    sessions: &[String],
    options: &Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: &[[f32; 4]],
) -> Box<dyn Stage> {
    match view {
        "library" => Box::new(LibraryStage::new(path)),
        "runner_copy" => {
            let dest = options
                .get("dest")
                .and_then(Value::as_str)
                .filter(|dest| !dest.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| path.display().to_string());
            Box::new(CopyStage::new(dest))
        }
        _ if wants_mc(view, options) => Box::new(McPlotStage::new(
            path, sessions, options, background, foreground, palette,
        )),
        _ => Box::new(SessionStage::new(
            view, path, sessions, options, background, foreground, palette,
        )),
    }
}

fn wants_mc(view: &str, options: &Value) -> bool {
    if view != "volumetric" {
        return false;
    }
    let model = options.get("model").and_then(Value::as_str).unwrap_or("");
    let study = options.get("study").and_then(Value::as_str).unwrap_or("");
    model == "mc" || !study.trim().is_empty()
}

fn ended(task: u64, generation: u64, phase: Phase, note: &str) -> Report {
    Report {
        task,
        generation,
        phase,
        done: 0,
        total: 0,
        note: note.into(),
    }
}

fn held_lines(options: &Value) -> Option<&str> {
    options
        .get("_lines")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
}

fn requested_voxel(options: &Value) -> f32 {
    scan_kit_io::volumetric::voxel_spacing(options)
}

fn report_at(phase: Phase, done: usize, total: usize) -> Report {
    Report {
        task: 0,
        generation: 0,
        phase,
        done: done as u64,
        total: total as u64,
        note: format!("{done} of {total}"),
    }
}

struct FilePlan {
    bytes: u64,
    chunks: usize,
    filled: usize,
}

struct SessionPlan {
    id: String,
    timeslice: bool,
    files: Vec<FilePlan>,
}

struct SessionStage {
    view: String,
    root: PathBuf,
    sessions: Vec<String>,
    options: Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: Vec<[f32; 4]>,
    stamps: Vec<String>,
    cached: bool,
    loaded: usize,
    plan: Vec<SessionPlan>,
    /// Windows of 4096 rows to pull into the next picture. Doubles after a timed poll.
    blocks: u32,
    /// 0 until a volumetric picture has been published. A file that is still
    /// arriving may show a coarser cube first. The finished picture uses the
    /// requested spacing.
    lattice: u8,
    report: Report,
}

fn chunks_for(bytes: u64, timeslice: bool) -> usize {
    // Timeslice lines are about a kilobyte. Counting every 48 bytes invented
    // windows, so a 128-window poll stayed on one long file and the rest were
    // read one file at a time. 512 still over-counts these rows. Spot lines
    // can be short, so they keep the 48-byte floor.
    let width = if timeslice { 512 } else { 48 };
    let rows = (bytes / width).max(1);
    rows.div_ceil(scan_kit_io::SLICE_ROWS as u64).max(1) as usize
}

impl SessionStage {
    fn new(
        view: &str,
        root: &Path,
        sessions: &[String],
        options: &Value,
        background: [f32; 4],
        foreground: [f32; 4],
        palette: &[[f32; 4]],
    ) -> Self {
        let stamps: Vec<_> = sessions
            .iter()
            .map(|session| stamp_key(view, root, session))
            .collect();
        let cached = !stamps.is_empty() && {
            let seen = seen_stamps();
            let seen = seen.lock().unwrap_or_else(|err| err.into_inner());
            stamps.iter().all(|stamp| seen.contains(stamp))
        };
        Self {
            view: view.into(),
            root: root.to_path_buf(),
            sessions: sessions.to_vec(),
            options: options.clone(),
            background,
            foreground,
            palette: palette.to_vec(),
            stamps,
            cached,
            loaded: 0,
            plan: Vec::new(),
            blocks: 1,
            lattice: 0,
            report: report_at(Phase::Chrome, 0, sessions.len()),
        }
    }

    fn ensure_plan(&mut self) {
        if !self.plan.is_empty() {
            return;
        }
        self.plan = self
            .sessions
            .iter()
            .map(|id| {
                let pieces = scan_kit_io::session_pieces(&self.root, id);
                SessionPlan {
                    id: id.clone(),
                    timeslice: pieces.timeslice,
                    files: pieces
                        .sizes
                        .iter()
                        .map(|size| FilePlan {
                            bytes: *size,
                            chunks: chunks_for(*size, pieces.timeslice),
                            filled: 0,
                        })
                        .collect(),
                }
            })
            .collect();
    }

    fn units(&self) -> usize {
        self.plan
            .iter()
            .map(|session| session.files.iter().map(|file| file.chunks).sum::<usize>())
            .sum()
    }

    fn done_units(&self) -> usize {
        self.plan
            .iter()
            .map(|session| session.files.iter().map(|file| file.filled).sum::<usize>())
            .sum()
    }

    /// Include one more row-block. Returns the session and file that moved.
    fn step(&mut self) -> Option<(usize, usize)> {
        for (session_index, session) in self.plan.iter_mut().enumerate() {
            for (file_index, file) in session.files.iter_mut().enumerate() {
                if file.filled < file.chunks {
                    file.filled += 1;
                    return Some((session_index, file_index));
                }
            }
        }
        None
    }

    fn settle(&mut self, at: (usize, usize), tail_done: bool) {
        let file = &mut self.plan[at.0].files[at.1];
        let asked = (file.filled as u64).saturating_mul(scan_kit_io::SLICE_ROWS as u64);
        // A file cannot contain more rows than it has bytes. Stop even if the reader
        // never reported the end.
        if tail_done || asked >= file.bytes {
            file.filled = file.chunks;
        } else if file.filled == file.chunks {
            // The file outlived the estimate. Double it instead of adding one
            // window per poll.
            file.chunks = file
                .chunks
                .saturating_mul(2)
                .max(file.filled.saturating_add(1));
        }
    }

    fn included(&self) -> Vec<String> {
        self.plan
            .iter()
            .take_while(|session| session.files.iter().any(|file| file.filled > 0))
            .filter(|session| session.files.iter().any(|file| file.filled > 0))
            .map(|session| session.id.clone())
            .collect()
    }

    fn load_scene(&self, ids: &[String]) -> Result<PlotScene, String> {
        self.open_scene(ids, false)
    }

    fn open_scene(&self, ids: &[String], preview: bool) -> Result<PlotScene, String> {
        let mut scene = if self.view == "volumetric" {
            if preview {
                let mut options = self.options.clone();
                options["_preview"] = Value::Bool(true);
                scan_kit_io::volumetric::volumetric(&self.root, ids, &options, None)
            } else {
                scan_kit_io::volumetric::volumetric(&self.root, ids, &self.options, None)
            }
        } else {
            scan_kit_io::analysis_scene(&self.view, &self.root, ids, &self.options)?
        };
        apply_palette(&mut scene, &self.palette);
        Ok(scene)
    }

    fn coarse_preview(&self, scene: &PlotScene) -> bool {
        scene.volume.values.len() > 1 && scene.volume.voxel > requested_voxel(&self.options) + 0.05
    }

    fn pack(&self, scene: &PlotScene, partial: bool) -> Result<Vec<u8>, String> {
        let quality = if partial { "partial" } else { "final" };
        encode_plot_reusing(
            scene,
            self.background,
            self.foreground,
            quality,
            held_lines(&self.options),
        )
    }

    fn encode(&self, ids: &[String], partial: bool) -> Result<Vec<u8>, String> {
        self.pack(&self.load_scene(ids)?, partial)
    }

    fn publish_volume_preview(&mut self) -> Poll<Vec<u8>> {
        self.lattice = 1;
        let ids = self.sessions.clone();
        let scene = match self.open_scene(&ids, false) {
            Ok(scene) => scene,
            Err(message) => {
                self.report.phase = Phase::Failed;
                self.report.note = message.clone();
                return Poll::Failed(message);
            }
        };
        // A cached session is already on disk. This picture is the requested
        // spacing. Reporting a second deposit left the hairline at 50%.
        match self.pack(&scene, false) {
            Ok(bytes) => {
                remember_stamps(&self.stamps);
                self.loaded = self.sessions.len();
                let total = self.sessions.len();
                self.report = report_at(Phase::Done, total, total);
                Poll::Ready(bytes)
            }
            Err(message) => {
                self.report.phase = Phase::Failed;
                self.report.note = message.clone();
                Poll::Failed(message)
            }
        }
    }
}

impl Stage for SessionStage {
    fn report(&self) -> Report {
        self.report.clone()
    }

    fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        if cancel.is_cancelled() {
            self.report.phase = Phase::Cancelled;
            return Poll::Cancelled;
        }
        let total = self.sessions.len();
        let unlimited = budget >= DRAIN;
        if total == 0 {
            return match self.encode(&[], false) {
                Ok(bytes) => {
                    self.report = report_at(Phase::Done, 0, 0);
                    Poll::Ready(bytes)
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
        }
        if self.cached && self.view == "volumetric" && !unlimited && self.lattice == 0 {
            return self.publish_volume_preview();
        }
        if unlimited || self.cached {
            self.loaded = total;
            let ids = self.sessions.clone();
            return match self.encode(&ids, false) {
                Ok(bytes) => {
                    remember_stamps(&self.stamps);
                    self.report = report_at(Phase::Done, total, total);
                    Poll::Ready(bytes)
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
        }
        self.ensure_plan();
        // A zero budget stays on one window, so a test can see every step.
        // A timed poll pulls several windows into one picture, then doubles
        // that count. The picture and the hairline still advance every poll.
        let quota = if budget.is_zero() {
            1
        } else {
            self.blocks.max(1)
        };
        let mut stepped = 0u32;
        let mut at = None;
        let started = std::time::Instant::now();
        while stepped < quota {
            if stepped > 0 && !budget.is_zero() && started.elapsed() >= budget {
                break;
            }
            if cancel.is_cancelled() {
                self.report.phase = Phase::Cancelled;
                return Poll::Cancelled;
            }
            match self.step() {
                Some(next) => {
                    at = Some(next);
                    stepped += 1;
                }
                None => break,
            }
        }
        let Some(at) = at else {
            let ids = self.sessions.clone();
            return match self.encode(&ids, false) {
                Ok(bytes) => {
                    remember_stamps(&self.stamps);
                    self.report = report_at(Phase::Done, self.units(), self.units());
                    Poll::Ready(bytes)
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
        };
        let ids = self.included();
        self.loaded = ids.len();
        let session = &self.plan[at.0];
        let rows = session.files[at.1].filled * scan_kit_io::SLICE_ROWS;
        let take = if session.timeslice {
            scan_kit_io::SliceTake::Timeslice {
                files: at.1,
                tail_rows: rows,
            }
        } else {
            scan_kit_io::SliceTake::Spot { rows }
        };
        scan_kit_io::bind_slice(&session.id, take);
        let preview = self.view == "volumetric" && self.lattice == 0 && !unlimited;
        if preview {
            self.lattice = 1;
        }
        let scene = if preview {
            self.open_scene(&ids, true)
        } else {
            self.load_scene(&ids)
        };
        let tail_done = scan_kit_io::slice_tail_done();
        scan_kit_io::clear_slice();
        self.settle(at, tail_done);
        let scene = match scene {
            Ok(scene) => scene,
            Err(message) => {
                self.report.phase = Phase::Failed;
                self.report.note = message.clone();
                return Poll::Failed(message);
            }
        };
        // A coarse cube while a file is still arriving. The finished picture
        // deposits again at the requested spacing.
        if preview && self.coarse_preview(&scene) && self.done_units() < self.units() {
            return match self.pack(&scene, true) {
                Ok(bytes) => {
                    let phase = if self.loaded <= 1 {
                        Phase::Parse
                    } else {
                        Phase::Scene
                    };
                    self.report = report_at(phase, self.done_units(), self.units().max(1));
                    Poll::Pending {
                        report: self.report.clone(),
                        preview: Some(bytes),
                    }
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
        }
        if self.done_units() >= self.units() && tail_done {
            let scene = if preview && self.coarse_preview(&scene) {
                match self.open_scene(&ids, false) {
                    Ok(scene) => scene,
                    Err(message) => {
                        self.report.phase = Phase::Failed;
                        self.report.note = message.clone();
                        return Poll::Failed(message);
                    }
                }
            } else {
                scene
            };
            return match self.pack(&scene, false) {
                Ok(bytes) => {
                    remember_stamps(&self.stamps);
                    self.loaded = total;
                    self.report = report_at(Phase::Done, self.units(), self.units());
                    Poll::Ready(bytes)
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
        }
        if self.done_units() >= self.units() {
            scan_kit_io::clear_slice();
            return match self.encode(&self.sessions.clone(), false) {
                Ok(bytes) => {
                    remember_stamps(&self.stamps);
                    self.loaded = total;
                    self.report = report_at(Phase::Done, self.units(), self.units());
                    Poll::Ready(bytes)
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
        }
        if !budget.is_zero() {
            // ponytail: 128 windows is 512k rows. A later poll still publishes;
            // the cap only stops one picture from swallowing the rest of the file.
            self.blocks = self.blocks.saturating_mul(2).clamp(1, 128);
        }
        match self.pack(&scene, true) {
            Ok(bytes) => {
                let phase = if self.loaded <= 1 {
                    Phase::Parse
                } else {
                    Phase::Scene
                };
                self.report = report_at(phase, self.done_units(), self.units());
                Poll::Pending {
                    report: self.report.clone(),
                    preview: Some(bytes),
                }
            }
            Err(message) => {
                self.report.phase = Phase::Failed;
                self.report.note = message.clone();
                Poll::Failed(message)
            }
        }
    }
}

/// Dose snapshots from the transport thread. The poll thread paints one while
/// the transport keeps adding histories.
struct McLive {
    done: AtomicU64,
    total: AtomicU64,
    stop: AtomicBool,
    target: AtomicU32,
    /// 0 until a cube has been copied. Release-stored after `preview` is written.
    preview_seq: AtomicU64,
    ready_flag: AtomicBool,
    note: Mutex<String>,
    preview: Mutex<Option<McResult>>,
    ready: Mutex<Option<McResult>>,
    error: Mutex<Option<String>>,
}

struct McPlotStage {
    root: PathBuf,
    sessions: Vec<String>,
    options: Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: Vec<[f32; 4]>,
    job: Option<McJob>,
    shared: Arc<McLive>,
    worker: Option<JoinHandle<()>>,
    worker_cancel: Cancel,
    /// Preview sequence already painted. 0 means the view has not seen a cube.
    seen: u64,
    report: Report,
}

fn mc_live() -> McLive {
    McLive {
        done: AtomicU64::new(0),
        total: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        target: AtomicU32::new(0),
        preview_seq: AtomicU64::new(0),
        ready_flag: AtomicBool::new(false),
        note: Mutex::new(String::new()),
        preview: Mutex::new(None),
        ready: Mutex::new(None),
        error: Mutex::new(None),
    }
}

fn job_histories(job: &McJob) -> u32 {
    match job {
        McJob::Slab(request) => request.histories,
        McJob::Patient(request) => request.histories,
    }
}

fn mc_transport(job: McJob, shared: Arc<McLive>, cancel: Cancel) {
    let mut run = match McRun::open(&job) {
        Ok(run) => run,
        Err(ComputeError::NoAdapter) => {
            *locked(&shared.error) = Some("no GPU adapter".into());
            return;
        }
        Err(err) => {
            *locked(&shared.error) = Some(err.to_string());
            return;
        }
    };
    loop {
        if shared.stop.load(Ordering::SeqCst) || cancel.is_cancelled() {
            return;
        }
        let target = shared.target.load(Ordering::Relaxed);
        if target > 0 {
            let _ = run.extend(target);
        }
        match run.poll(POLL_BUDGET, &cancel) {
            Poll::Pending { report, preview } => {
                shared.done.store(report.done, Ordering::Relaxed);
                shared.total.store(report.total, Ordering::Relaxed);
                *locked(&shared.note) = report.note;
                if let Some(result) = preview {
                    let seq = shared.preview_seq.load(Ordering::Relaxed) + 1;
                    *locked(&shared.preview) = Some(result);
                    shared.preview_seq.store(seq, Ordering::Release);
                }
            }
            Poll::Ready(result) => {
                shared.done.store(
                    shared.total.load(Ordering::Relaxed).max(1),
                    Ordering::Relaxed,
                );
                *locked(&shared.ready) = Some(result);
                shared.ready_flag.store(true, Ordering::Release);
                loop {
                    if shared.stop.load(Ordering::SeqCst) || cancel.is_cancelled() {
                        return;
                    }
                    let target = shared.target.load(Ordering::Relaxed);
                    if run.extend(target) && run.progress() < 1.0 {
                        shared.ready_flag.store(false, Ordering::Release);
                        *locked(&shared.ready) = None;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            Poll::Cancelled => return,
            Poll::Failed(message) => {
                *locked(&shared.error) = Some(message);
                return;
            }
        }
    }
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

impl McPlotStage {
    fn new(
        root: &Path,
        sessions: &[String],
        options: &Value,
        background: [f32; 4],
        foreground: [f32; 4],
        palette: &[[f32; 4]],
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            sessions: sessions.to_vec(),
            options: options.clone(),
            background,
            foreground,
            palette: palette.to_vec(),
            job: None,
            shared: Arc::new(mc_live()),
            worker: None,
            worker_cancel: Cancel::new(),
            seen: 0,
            report: report_at(Phase::Chrome, 0, 0),
        }
    }

    fn pack(&self, scene: &mut PlotScene, quality: &str) -> Result<Vec<u8>, String> {
        apply_palette(scene, &self.palette);
        encode_plot_reusing(
            scene,
            self.background,
            self.foreground,
            quality,
            held_lines(&self.options),
        )
    }

    fn paint(&self, result: &McResult, quality: &str) -> Result<Vec<u8>, String> {
        let mut scene = scan_kit_io::volumetric::volumetric(
            &self.root,
            &self.sessions,
            &self.options,
            Some(&|_| Ok(result.clone())),
        );
        self.pack(&mut scene, quality)
    }

    fn fail(&mut self, message: String) -> Poll<Vec<u8>> {
        self.report.phase = Phase::Failed;
        self.report.note = message.clone();
        Poll::Failed(message)
    }

    fn shutdown(&mut self) {
        self.worker_cancel.cancel();
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    fn sync_report(&mut self, phase: Phase) {
        self.report.phase = phase;
        self.report.done = self.shared.done.load(Ordering::Relaxed);
        self.report.total = self.shared.total.load(Ordering::Relaxed);
        let note = locked(&self.shared.note).clone();
        if !note.is_empty() {
            self.report.note = note;
        }
    }

    /// Paint `result` into the view. `finished` is the last cube.
    fn publish(&mut self, result: &McResult, seq: u64, finished: bool) -> Poll<Vec<u8>> {
        let quality = if finished { "final" } else { "partial" };
        match self.paint(result, quality) {
            Ok(bytes) => {
                self.seen = seq;
                if finished {
                    self.report.phase = Phase::Done;
                    self.report.done = self.report.total.max(self.report.done);
                    Poll::Ready(bytes)
                } else {
                    self.report.phase = Phase::Preview;
                    Poll::Pending {
                        report: self.report.clone(),
                        preview: Some(bytes),
                    }
                }
            }
            Err(message) => self.fail(message),
        }
    }

    fn paint_progress(&mut self) -> Poll<Vec<u8>> {
        let message = locked(&self.shared.error).clone();
        if let Some(message) = message {
            return self.fail(message);
        }
        self.sync_report(Phase::Compute);
        let seq = self.shared.preview_seq.load(Ordering::Acquire);
        let finished = self.shared.ready_flag.load(Ordering::Acquire);
        // The first picture is a rough cube, including when the run has already
        // reached the last history before this poll.
        if self.seen == 0 && seq > 0 {
            let result = locked(&self.shared.preview).clone();
            if let Some(result) = result {
                return self.publish(&result, seq, false);
            }
        }
        if finished {
            let result = locked(&self.shared.ready).clone();
            if let Some(result) = result {
                return self.publish(&result, seq.max(self.seen), true);
            }
        }
        if seq > self.seen {
            let result = locked(&self.shared.preview).clone();
            if let Some(result) = result {
                return self.publish(&result, seq, false);
            }
        }
        Poll::Pending {
            report: self.report.clone(),
            preview: None,
        }
    }

    fn follow(&mut self) -> Poll<Vec<u8>> {
        if self.worker.is_some() {
            return self.paint_progress();
        }
        let (mut scene, job) = capture_job(&self.root, &self.sessions, &self.options);
        let Some(job) = job else {
            return match self.pack(&mut scene, "final") {
                Ok(bytes) => {
                    self.report = report_at(Phase::Done, 1, 1);
                    Poll::Ready(bytes)
                }
                Err(message) => self.fail(message),
            };
        };
        let histories = job_histories(&job);
        self.shared
            .total
            .store(u64::from(histories.max(1)), Ordering::Relaxed);
        self.shared.target.store(histories, Ordering::Relaxed);
        *locked(&self.shared.note) = "0%".into();
        self.job = Some(job.clone());
        let shared = Arc::clone(&self.shared);
        let cancel = self.worker_cancel.clone();
        self.worker = Some(std::thread::spawn(move || {
            mc_transport(job, shared, cancel)
        }));
        // The blank capture is not a picture. The view keeps the last cube until
        // the first histories land, then each later copy replaces it.
        self.sync_report(Phase::Compute);
        Poll::Pending {
            report: self.report.clone(),
            preview: None,
        }
    }

    fn drain(&mut self, cancel: &Cancel) -> Poll<Vec<u8>> {
        let (mut scene, job) = capture_job(&self.root, &self.sessions, &self.options);
        let Some(job) = job else {
            return match self.pack(&mut scene, "final") {
                Ok(bytes) => {
                    self.report = report_at(Phase::Done, 1, 1);
                    Poll::Ready(bytes)
                }
                Err(message) => self.fail(message),
            };
        };
        self.job = Some(job.clone());
        let mut run = match McRun::open(&job) {
            Ok(run) => run,
            Err(ComputeError::NoAdapter) => return self.fail("no GPU adapter".into()),
            Err(err) => return self.fail(err.to_string()),
        };
        match run.poll(DRAIN, cancel) {
            Poll::Ready(result) => self.publish(&result, 1, true),
            Poll::Pending { .. } => self.fail("monte carlo did not finish".into()),
            Poll::Cancelled => {
                self.report.phase = Phase::Cancelled;
                Poll::Cancelled
            }
            Poll::Failed(message) => self.fail(message),
        }
    }
}

impl Drop for McPlotStage {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Stage for McPlotStage {
    fn report(&self) -> Report {
        self.report.clone()
    }

    fn retarget(&mut self, options: &Value, palette: &[[f32; 4]]) -> bool {
        if self.worker.is_none() {
            return false;
        }
        let (_scene, job) = capture_job(&self.root, &self.sessions, options);
        let Some(job) = job else {
            return false;
        };
        let same = self.job.as_ref().is_some_and(|old| same_phantom(&job, old));
        if !same {
            return false;
        }
        let histories = job_histories(&job);
        self.shared.target.store(histories, Ordering::Relaxed);
        self.shared
            .total
            .store(u64::from(histories.max(1)), Ordering::Relaxed);
        self.job = Some(job);
        self.options = options.clone();
        self.palette = palette.to_vec();
        true
    }

    fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        if cancel.is_cancelled() {
            self.shutdown();
            self.report.phase = Phase::Cancelled;
            return Poll::Cancelled;
        }
        if budget >= DRAIN {
            self.shutdown();
            return self.drain(cancel);
        }
        self.follow()
    }
}

fn capture_job(root: &Path, sessions: &[String], options: &Value) -> (PlotScene, Option<McJob>) {
    let job = std::cell::RefCell::new(None);
    let scene = scan_kit_io::volumetric::volumetric(
        root,
        sessions,
        options,
        Some(&|incoming| {
            *job.borrow_mut() = Some(incoming.clone());
            Ok(blank_result(incoming))
        }),
    );
    (scene, job.into_inner())
}

fn blank_result(job: &McJob) -> McResult {
    let (origin, shape, voxel) = match job {
        McJob::Slab(request) => (request.origin, request.shape, request.voxel_mm),
        McJob::Patient(request) => (request.origin_mm, request.shape, request.spacing_mm[0]),
    };
    let count = shape[0].saturating_mul(shape[1]).saturating_mul(shape[2]);
    McResult {
        volume: Volume {
            origin,
            shape,
            voxel,
            values: vec![0.0; count],
        },
        uncertainty: 0.0,
        ledger: [0.0; 6],
        let_d: Vec::new(),
    }
}

fn same_phantom(left: &McJob, right: &McJob) -> bool {
    match (left, right) {
        (McJob::Slab(left), McJob::Slab(right)) => {
            left.medium == right.medium
                && left.x == right.x
                && left.y == right.y
                && left.sx == right.sx
                && left.sy == right.sy
                && left.energy == right.energy
                && left.protons == right.protons
                && left.seed == right.seed
                && left.spread_pct == right.spread_pct
                && left.wet_mm == right.wet_mm
                && left.depth_mm == right.depth_mm
                && left.voxel_mm == right.voxel_mm
                && left.origin == right.origin
                && left.shape == right.shape
        }
        (McJob::Patient(left), McJob::Patient(right)) => {
            left.spots == right.spots
                && left.protons == right.protons
                && left.beams == right.beams
                && left.material == right.material
                && left.density == right.density
                && left.spacing_mm == right.spacing_mm
                && left.origin_mm == right.origin_mm
                && left.shape == right.shape
                && left.seed == right.seed
                && left.dose_to_water == right.dose_to_water
                && left.score_let == right.score_let
        }
        _ => false,
    }
}

struct Progress {
    done: AtomicU64,
    total: AtomicU64,
    stop: AtomicBool,
    preview: Mutex<Option<Vec<u8>>>,
}

struct LibraryStage {
    path: PathBuf,
    shared: Arc<Progress>,
    worker: Option<JoinHandle<Result<Vec<u8>, String>>>,
    sent: u64,
    report: Report,
}

impl LibraryStage {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            shared: Arc::new(Progress {
                done: AtomicU64::new(0),
                total: AtomicU64::new(0),
                stop: AtomicBool::new(false),
                preview: Mutex::new(None),
            }),
            worker: None,
            sent: 0,
            report: report_at(Phase::Read, 0, 0),
        }
    }

    fn note(done: u64, total: u64) -> String {
        if total == 0 {
            String::new()
        } else {
            format!("{done} of {total}")
        }
    }
}

impl Drop for LibraryStage {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Stage for LibraryStage {
    fn report(&self) -> Report {
        self.report.clone()
    }

    fn poll(&mut self, _budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        if cancel.is_cancelled() {
            self.shared.stop.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
            self.report.phase = Phase::Cancelled;
            return Poll::Cancelled;
        }
        if self.worker.is_none() {
            let path = self.path.clone();
            let shared = Arc::clone(&self.shared);
            self.worker = Some(std::thread::spawn(move || {
                let value = scan_kit_io::index_library(&path, &mut |done, total, preview| {
                    shared.done.store(done, Ordering::Relaxed);
                    shared.total.store(total, Ordering::Relaxed);
                    if let Some(preview) = preview {
                        if let Ok(bytes) = serde_json::to_vec(preview) {
                            *shared.preview.lock().unwrap_or_else(|err| err.into_inner()) =
                                Some(bytes);
                        }
                    }
                    !shared.stop.load(Ordering::SeqCst)
                })?;
                serde_json::to_vec(&value).map_err(|err| err.to_string())
            }));
        }
        let finished = self.worker.as_ref().expect("library worker").is_finished();
        if !finished {
            let done = self.shared.done.load(Ordering::Relaxed);
            let total = self.shared.total.load(Ordering::Relaxed);
            self.report = Report {
                task: 0,
                generation: 0,
                phase: Phase::Read,
                done,
                total,
                note: Self::note(done, total),
            };
            let preview = if done != self.sent {
                self.sent = done;
                self.shared
                    .preview
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .clone()
            } else {
                None
            };
            return Poll::Pending {
                report: self.report.clone(),
                preview,
            };
        }
        let worker = self.worker.take().expect("library worker");
        match worker.join() {
            Ok(Ok(bytes)) => {
                let total = self.shared.total.load(Ordering::Relaxed);
                self.report = report_at(Phase::Done, total as usize, total as usize);
                Poll::Ready(bytes)
            }
            Ok(Err(message)) if message == "cancelled" => {
                self.report.phase = Phase::Cancelled;
                Poll::Cancelled
            }
            Ok(Err(message)) => {
                self.report.phase = Phase::Failed;
                self.report.note = message.clone();
                Poll::Failed(message)
            }
            Err(_) => {
                self.report.phase = Phase::Failed;
                self.report.note = "library index panicked".into();
                Poll::Failed(self.report.note.clone())
            }
        }
    }
}

struct CopyStage {
    dest: String,
    shared: Arc<Progress>,
    worker: Option<JoinHandle<Result<Vec<u8>, String>>>,
    report: Report,
}

impl CopyStage {
    fn new(dest: String) -> Self {
        Self {
            dest,
            shared: Arc::new(Progress {
                done: AtomicU64::new(0),
                total: AtomicU64::new(0),
                stop: AtomicBool::new(false),
                preview: Mutex::new(None),
            }),
            worker: None,
            report: Report {
                task: 0,
                generation: 0,
                phase: Phase::Read,
                done: 0,
                total: 0,
                note: "Copying session".into(),
            },
        }
    }
}

impl Drop for CopyStage {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Stage for CopyStage {
    fn report(&self) -> Report {
        self.report.clone()
    }

    fn poll(&mut self, _budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        if cancel.is_cancelled() {
            self.shared.stop.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
            self.report.phase = Phase::Cancelled;
            return Poll::Cancelled;
        }
        if self.worker.is_none() {
            let dest = self.dest.clone();
            let shared = Arc::clone(&self.shared);
            self.worker = Some(std::thread::spawn(move || {
                let value = scan_kit_io::copy_runner(&dest, &mut |done, total| {
                    shared.done.store(done, Ordering::Relaxed);
                    shared.total.store(total, Ordering::Relaxed);
                    !shared.stop.load(Ordering::SeqCst)
                })?;
                serde_json::to_vec(&value).map_err(|err| err.to_string())
            }));
        }
        let finished = self.worker.as_ref().expect("copy worker").is_finished();
        if !finished {
            let done = self.shared.done.load(Ordering::Relaxed);
            let total = self.shared.total.load(Ordering::Relaxed);
            self.report = Report {
                task: 0,
                generation: 0,
                phase: Phase::Read,
                done,
                total,
                note: if total == 0 {
                    if done == 0 {
                        "Copying session".into()
                    } else {
                        format!("{done} files")
                    }
                } else {
                    format!("{done} of {total}")
                },
            };
            return Poll::Pending {
                report: self.report.clone(),
                preview: None,
            };
        }
        let worker = self.worker.take().expect("copy worker");
        match worker.join() {
            Ok(Ok(bytes)) => {
                let done = self.shared.done.load(Ordering::Relaxed).max(1);
                self.report = report_at(Phase::Done, done as usize, done as usize);
                Poll::Ready(bytes)
            }
            Ok(Err(message)) if message == "cancelled" => {
                self.report.phase = Phase::Cancelled;
                Poll::Cancelled
            }
            Ok(Err(message)) => {
                self.report.phase = Phase::Failed;
                self.report.note = message.clone();
                Poll::Failed(message)
            }
            Err(_) => {
                self.report.phase = Phase::Failed;
                self.report.note = "session copy panicked".into();
                Poll::Failed(self.report.note.clone())
            }
        }
    }
}

fn seen_stamps() -> &'static Mutex<HashSet<String>> {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn remember_stamps(stamps: &[String]) {
    let mut seen = seen_stamps().lock().unwrap_or_else(|err| err.into_inner());
    seen.extend(stamps.iter().cloned());
}

fn stamp_key(view: &str, root: &Path, session: &str) -> String {
    let dir = root.join(session);
    format!(
        "{view}\n{}\n{session}\n{}\n{}",
        root.display(),
        file_stamp(&dir.join("spot_data.csv")),
        file_stamp(&dir.join("input_map.csv")),
    )
}

fn file_stamp(path: &Path) -> String {
    let Ok(meta) = std::fs::metadata(path) else {
        return "missing".into();
    };
    let modified = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|time| time.as_nanos())
        .unwrap_or(0);
    format!("{}:{modified}", meta.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "scan-kit-task-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn write_session(root: &Path, name: &str) {
        let session = root.join(name);
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,2,4,1\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,position_x,position_y\n1,1,0,0\n2,2,4,1\n",
        )
        .unwrap();
    }

    #[test]
    fn a_small_volume_opens_as_the_final_picture() {
        let root = scratch();
        write_session(&root, "sess");
        let mut stage = SessionStage::new(
            "volumetric",
            &root,
            &["sess".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        let cancel = Cancel::new();
        match stage.poll(Duration::from_millis(12), &cancel) {
            Poll::Ready(bytes) => {
                let header = scan_kit_plot::plot_header(&bytes).unwrap();
                assert_eq!(header.quality, "final");
                assert_eq!(header.title, "Volumetric");
            }
            Poll::Pending { .. } => panic!("a two-spot field should not wait on a coarser cube"),
            Poll::Failed(message) => panic!("{message}"),
            Poll::Cancelled => panic!("cancelled"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_heavy_volume_finishes_on_the_requested_spacing() {
        let root = scratch();
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        let mut map = String::from("energy,charge_req,position_x,position_y\n");
        let mut spots = String::from("ic1_total_dose,ic2_total_dose,position_x,position_y\n");
        for i in 0..40 {
            let x = i * 5;
            map.push_str(&format!("180,0.05,{x},0\n"));
            spots.push_str(&format!("0.05,0.05,{x},0\n"));
        }
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        std::fs::write(session.join("spot_data.csv"), spots).unwrap();
        let mut stage = SessionStage::new(
            "volumetric",
            &root,
            &["sess".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        let cancel = Cancel::new();
        match stage.poll(Duration::from_millis(12), &cancel) {
            Poll::Ready(bytes) => {
                let header = scan_kit_plot::plot_header(&bytes).unwrap();
                assert_eq!(header.quality, "final");
                assert_eq!(stage.report.done, stage.report.total);
            }
            Poll::Pending { report, .. } => {
                panic!("stuck at {}/{}", report.done, report.total)
            }
            Poll::Failed(message) => panic!("{message}"),
            Poll::Cancelled => panic!("cancelled"),
        }
        let mut again = SessionStage::new(
            "volumetric",
            &root,
            &["sess".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        assert!(again.cached);
        match again.poll(Duration::from_millis(12), &cancel) {
            Poll::Ready(bytes) => {
                let header = scan_kit_plot::plot_header(&bytes).unwrap();
                assert_eq!(header.quality, "final");
            }
            Poll::Pending { report, .. } => {
                panic!("reopen stuck at {}/{}", report.done, report.total)
            }
            Poll::Failed(message) => panic!("{message}"),
            Poll::Cancelled => panic!("cancelled"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    fn session_stage(root: &Path, sessions: &[&str]) -> SessionStage {
        SessionStage::new(
            "timeline",
            root,
            &sessions
                .iter()
                .map(|id| (*id).to_owned())
                .collect::<Vec<_>>(),
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        )
    }

    #[test]
    fn three_sessions_emit_partial_scenes_and_one_final() {
        let root = scratch();
        for name in ["a", "b", "c"] {
            write_session(&root, name);
        }
        let mut stage = session_stage(&root, &["a", "b", "c"]);
        let cancel = Cancel::new();
        let mut partials = 0;
        let mut saw_final = false;
        for _ in 0..8 {
            match stage.poll(Duration::ZERO, &cancel) {
                Poll::Pending { preview, .. } => {
                    if let Some(bytes) = preview {
                        let header = scan_kit_plot::plot_header(&bytes).unwrap();
                        assert_eq!(header.quality, "partial");
                        partials += 1;
                    }
                }
                Poll::Ready(bytes) => {
                    let header = scan_kit_plot::plot_header(&bytes).unwrap();
                    assert_eq!(header.quality, "final");
                    assert_eq!(header.title, "Timeline");
                    saw_final = true;
                    break;
                }
                Poll::Failed(message) => panic!("{message}"),
                Poll::Cancelled => panic!("cancelled"),
            }
        }
        assert_eq!(partials, 2);
        assert!(saw_final);
        let paced_root = scratch();
        for name in ["d", "e", "f"] {
            write_session(&paced_root, name);
        }
        let mut paced = session_stage(&paced_root, &["d", "e", "f"]);
        let mut paced_partials = 0;
        let mut paced_ready = false;
        for _ in 0..8 {
            match paced.poll(Duration::from_secs(1), &cancel) {
                Poll::Pending {
                    preview: Some(_), ..
                } => paced_partials += 1,
                Poll::Pending { preview: None, .. } => {}
                Poll::Ready(_) => {
                    paced_ready = true;
                    break;
                }
                other => panic!("a short budget should still publish a picture: {other:?}"),
            }
        }
        assert!(paced_ready);
        assert!(paced_partials >= 1, "the first block is still a partial");
        assert!(
            paced_partials < 2,
            "a timed budget should fold the later tiny sessions into the next picture"
        );
        let streamed = scratch();
        let session = streamed.join("layers");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy\n70\n80\n90\n100\n110\n120\n130\n140\n",
        )
        .unwrap();
        for layer in 0..8 {
            let file = session
                .join(format!("layer-{layer}"))
                .join("run-0")
                .join("timeslice_data_device_units.csv");
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, format!("r_ic1_current_dose\n{layer}\n")).unwrap();
        }
        let mut stage = SessionStage::new(
            "timeline",
            &streamed,
            &["layers".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        let mut streamed_partials = 0;
        let mut streamed_ready = false;
        for _ in 0..8 {
            match stage.poll(Duration::from_secs(1), &cancel) {
                Poll::Pending {
                    preview: Some(_), ..
                } => streamed_partials += 1,
                Poll::Pending { preview: None, .. } => {}
                Poll::Ready(_) => {
                    streamed_ready = true;
                    break;
                }
                other => panic!("a long file should keep publishing: {other:?}"),
            }
        }
        assert!(streamed_ready);
        assert!(
            streamed_partials >= 2,
            "the picture should keep updating, got {streamed_partials}"
        );
        assert!(
            streamed_partials < 8,
            "doubling should finish in fewer polls than one file at a time, got {streamed_partials}"
        );
        let tight = scratch();
        let session = tight.join("layers");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy\n70\n80\n90\n100\n110\n120\n130\n140\n",
        )
        .unwrap();
        for layer in 0..8 {
            let file = session
                .join(format!("layer-{layer}"))
                .join("run-0")
                .join("timeslice_data_device_units.csv");
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, format!("r_ic1_current_dose\n{layer}\n")).unwrap();
        }
        let mut stage = SessionStage::new(
            "timeline",
            &tight,
            &["layers".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        let mut tight_partials = 0;
        let mut tight_ready = false;
        for _ in 0..16 {
            match stage.poll(Duration::from_nanos(1), &cancel) {
                Poll::Pending {
                    preview: Some(_), ..
                } => tight_partials += 1,
                Poll::Pending { preview: None, .. } => {}
                Poll::Ready(_) => {
                    tight_ready = true;
                    break;
                }
                other => panic!("a spent budget should still publish a picture: {other:?}"),
            }
        }
        assert!(tight_ready);
        assert!(
            tight_partials > streamed_partials,
            "a spent budget stops after one window, got {tight_partials} vs {streamed_partials}"
        );
        let _ = std::fs::remove_dir_all(&tight);
        let _ = std::fs::remove_dir_all(&streamed);
        let _ = std::fs::remove_dir_all(&paced_root);
        let sliced = scratch();
        let session = sliced.join("wide");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy\n70\n90\n110\n").unwrap();
        for (layer, sample) in [(0, "1"), (1, "2"), (2, "3")] {
            let file = session
                .join(format!("layer-{layer}"))
                .join("run-0")
                .join("timeslice_data_device_units.csv");
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, format!("r_ic1_current_dose\n{sample}\n")).unwrap();
        }
        let mut stage = SessionStage::new(
            "timeline",
            &sliced,
            &["wide".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        let mut sizes = Vec::new();
        for _ in 0..8 {
            match stage.poll(Duration::ZERO, &cancel) {
                Poll::Pending {
                    preview: Some(bytes),
                    ..
                } => sizes.push(bytes.len()),
                Poll::Pending { preview: None, .. } => {}
                Poll::Ready(_) => break,
                other => panic!("each timeslice file should publish: {other:?}"),
            }
        }
        assert!(
            sizes.len() >= 2,
            "expected a growing picture, got {sizes:?}"
        );
        assert!(sizes.windows(2).all(|pair| pair[1] >= pair[0]));
        let _ = std::fs::remove_dir_all(&sliced);
        let mut again = session_stage(&root, &["a", "b", "c"]);
        match again.poll(Duration::ZERO, &cancel) {
            Poll::Ready(bytes) => {
                assert_eq!(scan_kit_plot::plot_header(&bytes).unwrap().quality, "final");
            }
            other => panic!("cache hit should be one final slice: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cancel_after_the_first_session_does_not_emit_the_third() {
        let root = scratch();
        for name in ["a", "b", "c"] {
            write_session(&root, name);
        }
        let mut stage = session_stage(&root, &["a", "b", "c"]);
        let cancel = Cancel::new();
        match stage.poll(Duration::ZERO, &cancel) {
            Poll::Pending { report, preview } => {
                assert_eq!(report.done, 1);
                assert!(preview.is_some());
            }
            other => panic!("expected the first session, got {other:?}"),
        }
        cancel.cancel();
        assert!(matches!(
            stage.poll(Duration::ZERO, &cancel),
            Poll::Cancelled
        ));
        assert!(matches!(
            stage.poll(Duration::ZERO, &cancel),
            Poll::Cancelled
        ));
        assert_eq!(stage.loaded, 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn mc_view_paints_a_rough_cube_before_the_finished_one() {
        let root = scratch();
        write_session(&root, "sess");
        let options = json!({
            "model": "mc",
            "histories": "40000",
            "voxel": "4",
        });
        let mut stage = McPlotStage::new(
            &root,
            &["sess".into()],
            &options,
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        let cancel = Cancel::new();
        let started = std::time::Instant::now();
        let mut rough = false;
        loop {
            if started.elapsed() > Duration::from_secs(60) {
                panic!("monte carlo view did not finish");
            }
            match stage.poll(Duration::from_millis(12), &cancel) {
                Poll::Pending { preview, .. } => {
                    let Some(bytes) = preview else {
                        continue;
                    };
                    let header = scan_kit_plot::plot_header(&bytes).expect("header");
                    assert_eq!(header.quality, "partial");
                    assert!(
                        header
                            .panels
                            .iter()
                            .any(|panel| panel.title.starts_with("3D")),
                        "the rough cube should already show dose"
                    );
                    rough = true;
                }
                Poll::Ready(bytes) => {
                    let header = scan_kit_plot::plot_header(&bytes).expect("header");
                    assert_eq!(header.quality, "final");
                    assert!(
                        header
                            .panels
                            .iter()
                            .any(|panel| panel.title.starts_with("3D")),
                        "the finished cube should show dose"
                    );
                    break;
                }
                Poll::Failed(message) if message.contains("GPU") => {
                    let _ = std::fs::remove_dir_all(&root);
                    return;
                }
                other => panic!("monte carlo view failed: {other:?}"),
            }
        }
        assert!(
            rough,
            "the view never showed a cube before the run finished"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
