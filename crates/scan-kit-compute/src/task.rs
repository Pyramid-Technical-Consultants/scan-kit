//! One driver for every long desktop task.
//!
//! A poll does one slice and may return a plot payload. `open_plot` drains the
//! same stage with an unlimited budget, so a one-shot call still returns the
//! scene it returns today.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, UNIX_EPOCH};

use scan_kit_core::{encode_poll, Cancel, McJob, McResult, Phase, PlotScene, Poll, Report, Volume};
use scan_kit_plot::encode_plot_quality;
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
    stage: Box<dyn Stage>,
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
                if !task.cancel.is_cancelled() && task.stage.retarget(options, palette) {
                    task.generation = task.generation.saturating_add(1);
                    return Ok(json!({ "task": id, "generation": task.generation }));
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
            stage,
        },
    );
    Ok(json!({ "task": id, "generation": 1 }))
}

pub fn poll_task(id: u64) -> Result<Vec<u8>, String> {
    let mut driver = lock();
    let (bytes, remove) = {
        let Some(task) = driver.tasks.get_mut(&id) else {
            return Ok(encode_poll(&ended(id, 0, Phase::Cancelled, ""), None, true));
        };
        let cancel = task.cancel.clone();
        let generation = task.generation;
        let outcome = task.stage.poll(POLL_BUDGET, &cancel);
        let mut report = task.stage.report();
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
        (encode_poll(&report, payload.as_deref(), finished), remove)
    };
    if remove {
        driver.tasks.remove(&id);
        driver.clear(id);
    }
    Ok(bytes)
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
    if view != "dose_volume" {
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
    chrome: bool,
    loaded: usize,
    report: Report,
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
            chrome: false,
            loaded: 0,
            report: report_at(Phase::Chrome, 0, sessions.len()),
        }
    }

    fn encode(&self, ids: &[String], partial: bool) -> Result<Vec<u8>, String> {
        let mut scene = if self.view == "dose_volume" {
            scan_kit_io::dose_volume(&self.root, ids, &self.options, None)
        } else {
            scan_kit_io::analysis_scene(&self.view, &self.root, ids, &self.options)?
        };
        apply_palette(&mut scene, &self.palette);
        let quality = if partial { "partial" } else { "final" };
        encode_plot_quality(&scene, self.background, self.foreground, quality)
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
        if !self.chrome && !unlimited && !self.cached {
            self.chrome = true;
            self.report = report_at(Phase::Chrome, 0, total);
            return Poll::Pending {
                report: self.report.clone(),
                preview: None,
            };
        }
        if self.loaded < total {
            if unlimited || self.cached {
                self.loaded = total;
            } else {
                self.loaded += 1;
                let started = Instant::now();
                while self.loaded < total && started.elapsed() < budget {
                    if cancel.is_cancelled() {
                        self.report.phase = Phase::Cancelled;
                        return Poll::Cancelled;
                    }
                    self.loaded += 1;
                }
            }
        }
        let ids = self.sessions[..self.loaded].to_vec();
        let partial = self.loaded < total;
        match self.encode(&ids, partial) {
            Ok(bytes) => {
                if partial {
                    let phase = if self.loaded <= 1 {
                        Phase::Parse
                    } else {
                        Phase::Scene
                    };
                    self.report = report_at(phase, self.loaded, total);
                    Poll::Pending {
                        report: self.report.clone(),
                        preview: Some(bytes),
                    }
                } else {
                    remember_stamps(&self.stamps);
                    self.report = report_at(Phase::Done, total, total);
                    Poll::Ready(bytes)
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

struct McPlotStage {
    root: PathBuf,
    sessions: Vec<String>,
    options: Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: Vec<[f32; 4]>,
    run: Option<McRun>,
    job: Option<McJob>,
    report: Report,
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
            run: None,
            job: None,
            report: report_at(Phase::Chrome, 0, 0),
        }
    }

    fn pack(&self, scene: &mut PlotScene, quality: &str) -> Result<Vec<u8>, String> {
        apply_palette(scene, &self.palette);
        encode_plot_quality(scene, self.background, self.foreground, quality)
    }

    fn paint(&self, result: &McResult, quality: &str) -> Result<Vec<u8>, String> {
        let mut scene = scan_kit_io::dose_volume(
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

    fn advance(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        let Some(run) = self.run.as_mut() else {
            return self.fail("monte carlo is not open".into());
        };
        match run.poll(budget, cancel) {
            Poll::Pending { report, preview } => {
                self.report = report;
                match preview {
                    Some(result) => match self.paint(&result, "partial") {
                        Ok(bytes) => Poll::Pending {
                            report: self.report.clone(),
                            preview: Some(bytes),
                        },
                        Err(message) => self.fail(message),
                    },
                    None => Poll::Pending {
                        report: self.report.clone(),
                        preview: None,
                    },
                }
            }
            Poll::Ready(result) => match self.paint(&result, "final") {
                Ok(bytes) => {
                    self.report.phase = Phase::Done;
                    self.report.done = self.report.total;
                    Poll::Ready(bytes)
                }
                Err(message) => self.fail(message),
            },
            Poll::Cancelled => {
                self.report.phase = Phase::Cancelled;
                Poll::Cancelled
            }
            Poll::Failed(message) => self.fail(message),
        }
    }
}

impl Stage for McPlotStage {
    fn report(&self) -> Report {
        self.report.clone()
    }

    fn retarget(&mut self, options: &Value, palette: &[[f32; 4]]) -> bool {
        if self.run.is_none() {
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
        let histories = match &job {
            McJob::Slab(request) => request.histories,
            McJob::Patient(request) => request.histories,
        };
        if let Some(run) = self.run.as_mut() {
            if run.extend(histories) {
                let fraction = run.progress();
                self.report.total = u64::from(histories.max(1));
                self.report.done = (fraction * self.report.total as f64) as u64;
                let pct = (fraction * 100.0).round() as u32;
                self.report.note = format!("{pct}%");
            }
        }
        self.job = Some(job);
        self.options = options.clone();
        self.palette = palette.to_vec();
        true
    }

    fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        if cancel.is_cancelled() {
            self.report.phase = Phase::Cancelled;
            return Poll::Cancelled;
        }
        if self.run.is_none() {
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
            let histories = match &job {
                McJob::Slab(request) => request.histories,
                McJob::Patient(request) => request.histories,
            };
            let run = match McRun::open(&job) {
                Ok(run) => run,
                Err(ComputeError::NoAdapter) => return self.fail("no GPU adapter".into()),
                Err(err) => return self.fail(err.to_string()),
            };
            self.job = Some(job);
            self.run = Some(run);
            if budget >= DRAIN {
                return self.advance(budget, cancel);
            }
            self.report = Report {
                task: 0,
                generation: 0,
                phase: Phase::Chrome,
                done: 0,
                total: u64::from(histories),
                note: "0%".into(),
            };
            return match self.pack(&mut scene, "partial") {
                Ok(bytes) => Poll::Pending {
                    report: self.report.clone(),
                    preview: Some(bytes),
                },
                Err(message) => self.fail(message),
            };
        }
        self.advance(budget, cancel)
    }
}

fn capture_job(root: &Path, sessions: &[String], options: &Value) -> (PlotScene, Option<McJob>) {
    let job = std::cell::RefCell::new(None);
    let scene = scan_kit_io::dose_volume(
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
            self.report = Report {
                task: 0,
                generation: 0,
                phase: Phase::Read,
                done,
                total: 0,
                note: if done == 0 {
                    "Copying session".into()
                } else {
                    format!("{done} files")
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

    fn session_stage(root: &Path, sessions: &[&str]) -> SessionStage {
        SessionStage::new(
            "dose_accumulation",
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
                Poll::Pending { report, preview } => {
                    if report.phase == Phase::Chrome {
                        assert!(preview.is_none());
                        assert_eq!(report.done, 0);
                    } else if let Some(bytes) = preview {
                        let header = scan_kit_plot::plot_header(&bytes).unwrap();
                        assert_eq!(header.quality, "partial");
                        partials += 1;
                    }
                }
                Poll::Ready(bytes) => {
                    let header = scan_kit_plot::plot_header(&bytes).unwrap();
                    assert_eq!(header.quality, "final");
                    assert_eq!(header.title, "Dose Accumulation");
                    saw_final = true;
                    break;
                }
                Poll::Failed(message) => panic!("{message}"),
                Poll::Cancelled => panic!("cancelled"),
            }
        }
        assert_eq!(partials, 2);
        assert!(saw_final);
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
        assert!(matches!(
            stage.poll(Duration::ZERO, &cancel),
            Poll::Pending { preview: None, .. }
        ));
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
}
