//! One driver for every long desktop task.
//!
//! A poll publishes one picture and may return a plot payload. The first timed
//! poll is one window. Each later timed poll doubles that window and reads only
//! the new bytes, until the picture passes four windows. That poll reads the
//! rest once. `open_plot` drains the same stage with an unlimited budget, so a
//! one-shot call still returns the scene it returns today.

use std::collections::HashMap;
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
/// Estimated rows already in the picture. Past four windows the next partial
/// is most of the final trace, so the rest of the file is read once.
/// ponytail: a live tail that still wants pictures after this waits for the
/// next open. Raise the cap if a session must keep streaming past it.
const PARTIAL_ROW_CAP: usize = 4 * scan_kit_io::SLICE_ROWS;

trait Stage: Send {
    fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>>;
    fn report(&self) -> Report;
    fn retarget(
        &mut self,
        _view: &str,
        _root: &Path,
        _sessions: &[String],
        _options: &Value,
        _palette: &[[f32; 4]],
        _background: [f32; 4],
        _foreground: [f32; 4],
    ) -> bool {
        false
    }

    /// The desktop calls this with the task cancel token before the stage can
    /// start work. A later cancel reaches a running worker even while a poll
    /// is painting and the stage is not in the driver.
    fn bind_cancel(&mut self, _cancel: Cancel) {}
}

struct Running {
    generation: u64,
    cancel: Cancel,
    /// Taken out for the duration of a poll so cancel and a new start do not
    /// wait on the encode.
    stage: Option<Box<dyn Stage>>,
    /// Last report the stage produced. A poll that arrives while another
    /// holds the stage answers with this instead of a fabricated phase.
    last_report: Option<Report>,
}

#[derive(Default)]
struct Driver {
    next: u64,
    plot: Option<u64>,
    library: Option<u64>,
    copy: Option<u64>,
    export: Option<u64>,
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

pub(crate) enum Slot {
    Plot,
    Library,
    Copy,
    Export,
}

impl Driver {
    fn slot_id(&self, slot: &Slot) -> Option<u64> {
        match slot {
            Slot::Plot => self.plot,
            Slot::Library => self.library,
            Slot::Copy => self.copy,
            Slot::Export => self.export,
        }
    }

    fn set_slot(&mut self, slot: &Slot, id: Option<u64>) {
        let cell = match slot {
            Slot::Plot => &mut self.plot,
            Slot::Library => &mut self.library,
            Slot::Copy => &mut self.copy,
            Slot::Export => &mut self.export,
        };
        *cell = id;
    }

    fn clear(&mut self, id: u64) {
        for cell in [
            &mut self.plot,
            &mut self.library,
            &mut self.copy,
            &mut self.export,
        ] {
            if *cell == Some(id) {
                *cell = None;
            }
        }
    }

    fn take(&mut self, slot: &Slot) -> Option<Running> {
        let id = self.slot_id(slot)?;
        if let Some(task) = self.tasks.get(&id) {
            task.cancel.cancel();
        }
        self.set_slot(slot, None);
        self.tasks.remove(&id)
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
    let mut normalized = options.clone();
    scan_kit_io::normalize_options(&mut normalized);
    let options = &normalized;
    let slot = slot_of(view);
    // Take the stage out so the retarget's scene build runs off the driver
    // lock — it does file IO, and every other poll/cancel waits behind it.
    let checked_out = {
        let mut driver = lock();
        if matches!(slot, Slot::Plot) {
            driver.plot.and_then(|id| {
                driver.tasks.get_mut(&id).and_then(|task| {
                    (!task.cancel.is_cancelled())
                        .then(|| task.stage.take().map(|stage| (id, stage)))
                        .flatten()
                })
            })
        } else {
            None
        }
    };
    let checked_id = checked_out.as_ref().map(|(id, _)| *id);
    if let Some((id, mut stage)) = checked_out {
        // retarget runs the same fallible call tree poll wraps in
        // catch_unwind — a panic must not leave task.stage None forever.
        let retargeted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.retarget(
                view, path, sessions, options, palette, background, foreground,
            )
        }))
        .unwrap_or(false);
        if retargeted {
            let mut driver = lock();
            // The task may have been cancelled or replaced while the stage
            // was checked out — only adopt when it is still parked for us.
            if let Some(task) = driver.tasks.get_mut(&id) {
                if task.stage.is_none() && !task.cancel.is_cancelled() {
                    task.stage = Some(stage);
                    task.generation = task.generation.saturating_add(1);
                    return Ok(json!({ "task": id, "generation": task.generation }));
                }
            }
        }
    }
    let mut driver = lock();
    // This start ran its retarget off-lock; if a newer start claimed the slot
    // meanwhile it owns the slot — report it rather than evicting it.
    if checked_id.is_some() {
        if let Some(owner) = driver.plot.filter(|o| checked_id != Some(*o)) {
            let generation = driver.tasks.get(&owner).map(|t| t.generation).unwrap_or(0);
            return Ok(json!({ "task": owner, "generation": generation }));
        }
    }
    let replaced = driver.take(&slot);
    let mut stage = build_stage(
        view, path, sessions, options, background, foreground, palette,
    );
    let cancel = Cancel::new();
    stage.bind_cancel(cancel.clone());
    let id = driver.next;
    driver.next = driver.next.saturating_add(1);
    driver.set_slot(&slot, Some(id));
    driver.tasks.insert(
        id,
        Running {
            generation: 1,
            cancel,
            stage: Some(stage),
            last_report: None,
        },
    );
    // Joining the replaced worker can wait on its GPU drain — that must not
    // happen under the driver lock, or every poll/cancel stalls behind it.
    drop(driver);
    drop(replaced);
    Ok(json!({ "task": id, "generation": 1 }))
}

pub fn poll_task(id: u64) -> Result<Vec<u8>, String> {
    let (mut stage, cancel, generation) = {
        let mut driver = lock();
        let Some(task) = driver.tasks.get_mut(&id) else {
            return Ok(encode_poll(&ended(id, 0, Phase::Cancelled, ""), None, true));
        };
        let Some(stage) = task.stage.take() else {
            let report = task
                .last_report
                .clone()
                .unwrap_or_else(|| ended(id, task.generation, Phase::Scene, ""));
            return Ok(encode_poll(&report, None, false));
        };
        (stage, task.cancel.clone(), task.generation)
    };
    // A panicking poll must still reach the removal path — otherwise the task
    // is left half-registered and answers Pending forever.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stage.poll(POLL_BUDGET, &cancel)
    }))
    .unwrap_or_else(|_| {
        // The unwind skipped clear_slice — drop the stale binding or this pool
        // thread keeps truncating same-named tables under other roots.
        scan_kit_io::clear_slice();
        Poll::Failed("task panicked".into())
    });
    let report = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stage.report()));
    let (outcome, mut report) = match report {
        Ok(report) => (outcome, report),
        Err(_) => {
            // A panicking report must end the task — re-parking it would leave
            // stage=None and every later poll would fabricate Pending forever.
            scan_kit_io::clear_slice();
            (
                Poll::Failed("task panicked".into()),
                ended(id, generation, Phase::Failed, "task panicked"),
            )
        }
    };
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
            report.note = message;
            (true, None, true)
        }
    };
    let mut driver = lock();
    let Some(current) = driver.tasks.get(&id).map(|task| task.generation) else {
        // Dropping the stage joins its worker — never under the driver lock.
        drop(driver);
        drop(stage);
        return Ok(encode_poll(
            &ended(id, generation, Phase::Cancelled, ""),
            None,
            true,
        ));
    };
    if current != generation {
        drop(driver);
        // This poll's stage belongs to the superseded generation — parking it
        // would clobber the successor's stage and re-publish stale work.
        drop(stage);
        return Ok(encode_poll(
            &ended(id, generation, Phase::Cancelled, ""),
            None,
            true,
        ));
    }
    if remove {
        driver.tasks.remove(&id);
        driver.clear(id);
        drop(driver);
        // The finished stage joins its worker on drop — off the lock, or the
        // GPU drain stalls every other poll and cancel behind it.
        drop(stage);
    } else {
        if let Some(task) = driver.tasks.get_mut(&id) {
            task.last_report = Some(report.clone());
            task.stage = Some(stage);
        }
        drop(driver);
    }
    Ok(encode_poll(&report, payload.as_deref(), finished))
}

/// Cancel a task. `generation`, when given, makes a stale caller's cancel a
/// no-op — retargets reuse the task id, so an id alone cannot distinguish the
/// superseded ticket from its successor.
pub fn cancel_task(id: u64, generation: Option<u64>) {
    // Same rule as start_task: the stage drop may join a worker mid-drain.
    let removed = {
        let mut driver = lock();
        let stale = driver
            .tasks
            .get(&id)
            .is_some_and(|task| generation.is_some_and(|gen| task.generation != gen));
        if stale {
            None
        } else {
            if let Some(task) = driver.tasks.get(&id) {
                task.cancel.cancel();
            }
            driver.clear(id);
            driver.tasks.remove(&id)
        }
    };
    drop(removed);
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
    let mut options = options.clone();
    scan_kit_io::normalize_options(&mut options);
    let mut stage = build_stage(
        view, root, sessions, &options, background, foreground, palette,
    );
    // A stage may spend its first poll on setup and report Pending even with
    // an unlimited budget — keep driving until it reaches a terminal state.
    for _ in 0..64 {
        match stage.poll(Duration::MAX, &Cancel::new()) {
            Poll::Ready(bytes) => return Ok(bytes),
            Poll::Failed(message) => return Err(message),
            Poll::Cancelled => return Err("cancelled".into()),
            Poll::Pending { .. } => {}
        }
    }
    Err("plot did not finish".into())
}

/// The non-plot tasks (`write_dose`, `write_report`) drained in one call for
/// callers that have no progress surface — the MCP bridge. The stage is the
/// same one the desktop polls; the payload is the result JSON.
pub(crate) fn drain_task(
    view: &str,
    root: &Path,
    sessions: &[String],
    options: &Value,
) -> Result<Vec<u8>, String> {
    drain_plot(view, root, sessions, options, [0.0; 4], [0.0; 4], &[])
}

pub(crate) fn slot_of(view: &str) -> Slot {
    match view {
        "library" => Slot::Library,
        "runner_copy" => Slot::Copy,
        "write_dose" | "write_report" => Slot::Export,
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
        "write_dose" | "write_report" => Box::new(ExportStage::new(
            view == "write_dose",
            path,
            sessions,
            options,
        )),
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
    let study = options.get("study").and_then(Value::as_str).unwrap_or("");
    // The button writes `mc`. An older build wrote the label `Monte Carlo`.
    // Either one has to leave the analytic stage, or the fill draws that cube.
    scan_kit_io::volumetric::model_is_monte_carlo(options) || !study.trim().is_empty()
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
        // stamps/cached are computed lazily on the first poll: building them
        // here does per-session fs::metadata under the global driver lock,
        // which stalls every other task's poll and cancel.
        Self {
            view: view.into(),
            root: root.to_path_buf(),
            sessions: sessions.to_vec(),
            options: options.clone(),
            background,
            foreground,
            palette: palette.to_vec(),
            stamps: Vec::new(),
            cached: false,
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

    fn covered_rows(&self) -> usize {
        self.plan
            .iter()
            .flat_map(|session| session.files.iter())
            .map(|file| file.filled.saturating_mul(scan_kit_io::SLICE_ROWS))
            .sum()
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
                // `_preview` is written by key, so a non-object options value
                // would panic — normalize it away.
                let mut options = if self.options.is_object() {
                    self.options.clone()
                } else {
                    json!({})
                };
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
        if self.stamps.is_empty() && !self.sessions.is_empty() {
            self.stamps = self
                .sessions
                .iter()
                .map(|session| stamp_key(&self.root, session))
                .collect();
            let seen = seen_stamps();
            let seen = seen.lock().unwrap_or_else(|err| err.into_inner());
            self.cached = self
                .stamps
                .iter()
                .all(|stamp| seen.get(stamp_id(stamp)).is_some_and(|have| have == stamp));
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
        // Dose is one picture. A window per poll deposited the cube again, so
        // the bar looked like a queue that never cleared.
        if unlimited || self.cached || self.view == "volumetric" {
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
        // that count, until the picture passes four windows.
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
        // Encoding this prefix and then the rest walks the trace twice.
        if !budget.is_zero()
            && self.covered_rows() > PARTIAL_ROW_CAP
            && self.done_units() < self.units()
        {
            for session in &mut self.plan {
                for file in &mut session.files {
                    file.filled = file.chunks;
                }
            }
            let units = self.units();
            let ids = self.sessions.clone();
            return match self.encode(&ids, false) {
                Ok(bytes) => {
                    remember_stamps(&self.stamps);
                    self.loaded = total;
                    self.report = report_at(Phase::Done, units, units);
                    Poll::Ready(bytes)
                }
                Err(message) => {
                    self.report.phase = Phase::Failed;
                    self.report.note = message.clone();
                    Poll::Failed(message)
                }
            };
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
        scan_kit_io::bind_slice(&self.root, &session.id, take);
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
        if self.done_units() >= self.units() {
            // The bound scene's file window was fixed at ensure_plan — a
            // file or session that arrived mid-load would be sealed out of
            // the "final" payload forever. Encode unbound so the last read
            // picks up arrivals, like the !tail_done path always did.
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
    /// One cube per job, in the order the scene asked for them.
    preview: Mutex<Option<Vec<McResult>>>,
    /// `(histories the cubes satisfy, cubes)` — a retarget upward invalidates
    /// a published cube before the worker's next loop can retract it.
    ready: Mutex<Option<(u32, Vec<McResult>)>>,
    error: Mutex<Option<String>>,
}

struct McPlotStage {
    root: PathBuf,
    sessions: Vec<String>,
    options: Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: Vec<[f32; 4]>,
    jobs: Vec<McJob>,
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

/// Transport every job side by side. A preview or the finished picture holds
/// one cube per job, so measured and plan always carry similar histories.
fn mc_transport(jobs: Vec<McJob>, shared: Arc<McLive>, cancel: Cancel) {
    let mut runs = Vec::with_capacity(jobs.len());
    for job in &jobs {
        match McRun::open(job) {
            Ok(run) => runs.push(run),
            Err(ComputeError::NoAdapter) => {
                *locked(&shared.error) = Some("no GPU adapter".into());
                return;
            }
            Err(err) => {
                *locked(&shared.error) = Some(err.to_string());
                return;
            }
        }
    }
    // Every run sees the full budget: a step's wait already includes the
    // sibling chunks queued ahead of its own, so adapt_chunk balances the
    // combined queue to roughly the budget.
    let budget = POLL_BUDGET;
    let mut latest: Vec<Option<McResult>> = vec![None; runs.len()];
    let mut finished: Vec<Option<McResult>> = vec![None; runs.len()];
    let mut done = vec![0u64; runs.len()];
    let mut total = vec![1u64; runs.len()];
    loop {
        if shared.stop.load(Ordering::SeqCst) || cancel.is_cancelled() {
            return;
        }
        let target = shared.target.load(Ordering::Relaxed);
        let mut fresh = false;
        for (k, run) in runs.iter_mut().enumerate() {
            if target > 0 && run.extend(target) && finished[k].is_some() && run.progress() < 1.0 {
                finished[k] = None;
                shared.ready_flag.store(false, Ordering::Release);
                *locked(&shared.ready) = None;
            }
            if finished[k].is_some() {
                // extend() is false on a run with no live work — its old total
                // must not pin satisfied below a retargeted target forever.
                total[k] = total[k].max(u64::from(target));
                done[k] = done[k].max(total[k]);
                continue;
            }
            match run.poll(budget, &cancel) {
                Poll::Pending { report, preview } => {
                    done[k] = report.done;
                    total[k] = report.total.max(1);
                    if let Some(result) = preview {
                        latest[k] = Some(result);
                        fresh = true;
                    }
                }
                Poll::Ready(result) => {
                    // A job that opens already-done never reported a total —
                    // its sentinel 1 must not drag satisfied below target.
                    // Floor at the live target, not the spawn-time request —
                    // a retarget-down delivered fewer histories than it asked.
                    total[k] = total[k].max(u64::from(target));
                    done[k] = total[k];
                    latest[k] = Some(result.clone());
                    finished[k] = Some(result);
                    fresh = true;
                }
                Poll::Cancelled => return,
                Poll::Failed(message) => {
                    *locked(&shared.error) = Some(message);
                    return;
                }
            }
        }
        let (all_done, all_total) = (done.iter().sum::<u64>(), total.iter().sum::<u64>());
        shared.done.store(all_done, Ordering::Relaxed);
        shared.total.store(all_total, Ordering::Relaxed);
        *locked(&shared.note) = format!("{}%", 100 * all_done / all_total.max(1));
        if finished.iter().all(Option::is_some) {
            if !shared.ready_flag.load(Ordering::Acquire) {
                let results: Vec<McResult> = finished.iter().flatten().cloned().collect();
                let mut reached = jobs.clone();
                for (job, delivered) in reached.iter_mut().zip(&total) {
                    let delivered = (*delivered).min(u64::from(u32::MAX)) as u32;
                    match job {
                        McJob::Slab(request) => request.histories = delivered,
                        McJob::Patient(request) => request.histories = delivered,
                    }
                }
                keep_runs(&reached, &results);
                let satisfied = total.iter().copied().min().unwrap_or(0) as u32;
                *locked(&shared.ready) = Some((satisfied, results));
                shared.ready_flag.store(true, Ordering::Release);
            }
            std::thread::sleep(Duration::from_millis(20));
        } else if fresh && latest.iter().all(Option::is_some) {
            let seq = shared.preview_seq.load(Ordering::Relaxed) + 1;
            *locked(&shared.preview) = latest.iter().cloned().collect();
            shared.preview_seq.store(seq, Ordering::Release);
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
            jobs: Vec::new(),
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

    /// The scene asks for its jobs in the same order `capture_job` saw them.
    fn paint(&self, results: &[McResult], quality: &str) -> Result<Vec<u8>, String> {
        let next = std::cell::Cell::new(0usize);
        let mut scene = scan_kit_io::volumetric::volumetric(
            &self.root,
            &self.sessions,
            &self.options,
            Some(&|job| {
                let at = next.get();
                next.set(at + 1);
                results
                    .get(at)
                    .cloned()
                    .map_or_else(|| Ok(blank_result(job)), Ok)
            }),
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

    /// Paint `results` into the view. `finished` is the last set of cubes.
    fn publish(&mut self, results: &[McResult], seq: u64, finished: bool) -> Poll<Vec<u8>> {
        let quality = if finished { "final" } else { "partial" };
        match self.paint(results, quality) {
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
        // A panicked worker unwinds without touching shared.error — join it
        // and fail instead of polling Pending forever.
        if self
            .worker
            .as_ref()
            .is_some_and(|worker| worker.is_finished())
        {
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
                if !self.shared.ready_flag.load(Ordering::Acquire)
                    && locked(&self.shared.error).is_none()
                {
                    return self.fail("monte carlo worker stopped".into());
                }
            }
        }
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
            if let Some((satisfied, result)) = result {
                if satisfied >= self.shared.target.load(Ordering::Relaxed) {
                    return self.publish(&result, seq.max(self.seen), true);
                }
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
        let (mut scene, jobs) = capture_job(&self.root, &self.sessions, &self.options);
        if jobs.is_empty() {
            return match self.pack(&mut scene, "final") {
                Ok(bytes) => {
                    self.report = report_at(Phase::Done, 1, 1);
                    Poll::Ready(bytes)
                }
                Err(message) => self.fail(message),
            };
        }
        if let Some(results) = kept_results(&jobs) {
            self.jobs = jobs;
            self.report = report_at(Phase::Done, 1, 1);
            return self.publish(&results, 1, true);
        }
        let histories = job_histories(&jobs[0]);
        self.shared
            .total
            .store(all_histories(&jobs), Ordering::Relaxed);
        self.shared.target.store(histories, Ordering::Relaxed);
        *locked(&self.shared.note) = "0%".into();
        self.jobs = jobs.clone();
        let shared = Arc::clone(&self.shared);
        let cancel = self.worker_cancel.clone();
        self.worker = Some(std::thread::spawn(move || {
            mc_transport(jobs, shared, cancel)
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
        let (mut scene, jobs) = capture_job(&self.root, &self.sessions, &self.options);
        if jobs.is_empty() {
            return match self.pack(&mut scene, "final") {
                Ok(bytes) => {
                    self.report = report_at(Phase::Done, 1, 1);
                    Poll::Ready(bytes)
                }
                Err(message) => self.fail(message),
            };
        }
        self.jobs = jobs.clone();
        if let Some(results) = kept_results(&jobs) {
            return self.publish(&results, 1, true);
        }
        let mut results = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let mut run = match McRun::open(job) {
                Ok(run) => run,
                Err(ComputeError::NoAdapter) => return self.fail("no GPU adapter".into()),
                Err(err) => return self.fail(err.to_string()),
            };
            match run.poll(DRAIN, cancel) {
                Poll::Ready(result) => results.push(result),
                Poll::Pending { .. } => return self.fail("monte carlo did not finish".into()),
                Poll::Cancelled => {
                    self.report.phase = Phase::Cancelled;
                    return Poll::Cancelled;
                }
                Poll::Failed(message) => return self.fail(message),
            }
        }
        keep_runs(&jobs, &results);
        self.publish(&results, 1, true)
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

    fn bind_cancel(&mut self, cancel: Cancel) {
        self.worker_cancel = cancel;
    }

    fn retarget(
        &mut self,
        view: &str,
        root: &Path,
        sessions: &[String],
        options: &Value,
        palette: &[[f32; 4]],
        background: [f32; 4],
        foreground: [f32; 4],
    ) -> bool {
        if !wants_mc(view, options) {
            return false;
        }
        if self.worker.is_none() {
            return false;
        }
        // The reused run still belongs to its own session set — a different
        // session that produces an identical job must get a fresh task.
        if root != self.root || sessions != self.sessions {
            return false;
        }
        // A new grid cannot keep the tallies. Bail out before building that
        // grid so the replacement does not wait on the scene it is about to throw away.
        if geometry_changed(&self.options, options) {
            return false;
        }
        let (_scene, jobs) = capture_job(&self.root, &self.sessions, options);
        let same = !jobs.is_empty()
            && jobs.len() == self.jobs.len()
            && jobs
                .iter()
                .zip(&self.jobs)
                .all(|(new, old)| same_phantom(new, old));
        if !same {
            return false;
        }
        let histories = job_histories(&jobs[0]);
        self.shared.target.store(histories, Ordering::Relaxed);
        // Runs cannot shrink — a lower request still transports the old
        // count, so total never drops below what the worker reports.
        self.shared
            .total
            .fetch_max(all_histories(&jobs), Ordering::Relaxed);
        self.jobs = jobs;
        self.options = options.clone();
        self.palette = palette.to_vec();
        self.background = background;
        self.foreground = foreground;
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
            // worker_cancel is bound to `cancel`; drain with a fresh token or a
            // bound stage reports Cancelled instead of draining.
            return self.drain(&Cancel::new());
        }
        self.follow()
    }
}

fn capture_job(root: &Path, sessions: &[String], options: &Value) -> (PlotScene, Vec<McJob>) {
    let jobs = std::cell::RefCell::new(Vec::new());
    let scene = scan_kit_io::volumetric::volumetric(
        root,
        sessions,
        options,
        Some(&|incoming| {
            jobs.borrow_mut().push(incoming.clone());
            Ok(blank_result(incoming))
        }),
    );
    (scene, jobs.into_inner())
}

/// The last finished transport. A compare, plot, or cell change opens a new
/// stage on the same jobs, and that stage paints these cubes instead of
/// transporting again. A run with at least the requested histories counts.
// ponytail: one scene's cubes stay resident. Flipping between two geometries
// transports again; keep more sets if that becomes common.
fn kept_runs() -> &'static Mutex<Option<(Vec<McJob>, Vec<McResult>)>> {
    static KEPT: OnceLock<Mutex<Option<(Vec<McJob>, Vec<McResult>)>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(None))
}

fn keep_runs(jobs: &[McJob], results: &[McResult]) {
    *locked(kept_runs()) = Some((jobs.to_vec(), results.to_vec()));
}

fn kept_results(jobs: &[McJob]) -> Option<Vec<McResult>> {
    let kept = locked(kept_runs());
    let (seen, results) = kept.as_ref()?;
    let covers = seen.len() == jobs.len()
        && seen
            .iter()
            .zip(jobs)
            .all(|(old, new)| same_phantom(old, new) && job_histories(old) >= job_histories(new));
    covers.then(|| results.clone())
}

fn all_histories(jobs: &[McJob]) -> u64 {
    jobs.iter()
        .map(|job| u64::from(job_histories(job).max(1)))
        .sum()
}

fn blank_result(job: &McJob) -> McResult {
    let (origin, shape, spacing) = match job {
        McJob::Slab(request) => (request.origin, request.shape, [request.voxel_mm; 3]),
        McJob::Patient(request) => (
            request.origin_mm,
            request.shape,
            request.spacing_mm.map(|mm| mm.max(0.25)),
        ),
    };
    let count = shape[0].saturating_mul(shape[1]).saturating_mul(shape[2]);
    // The placeholder must not allocate an unbounded request shape.
    let (shape, values) = if count > 1 << 30 {
        ([0, 0, 0], Vec::new())
    } else {
        (shape, vec![0.0; count])
    };
    McResult {
        volume: Volume {
            origin,
            shape,
            spacing,
            values,
        },
        uncertainty: 0.0,
        ledger: [0.0; 6],
        let_d: Vec::new(),
    }
}

fn geometry_changed(old: &Value, new: &Value) -> bool {
    [
        "voxel", "medium", "spread", "wet", "phantom", "gantry", "margin", "gap", "model", "study",
    ]
    .into_iter()
    .any(|key| old.get(key) != new.get(key))
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
                && left.let_water == right.let_water
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
                        let mut slot = shared.preview.lock().unwrap_or_else(|err| err.into_inner());
                        // The row set grows as indexing runs — serializing it
                        // per entry is O(n²). Skip while the last preview is
                        // still unconsumed.
                        if slot.is_none() {
                            if let Ok(bytes) = serde_json::to_vec(preview) {
                                *slot = Some(bytes);
                            }
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
                    .take()
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

/// A `write_dose`/`write_report` task. The patient transport runs through the
/// same sliced `McRun` the volumetric view uses, so each poll reports history
/// progress and a cancel lands mid-run instead of after it. No worker thread:
/// `McRun::poll` already splits the run into budget-sized submits.
struct ExportStage {
    dose: bool,
    root: PathBuf,
    sessions: Vec<String>,
    options: Value,
    dest: String,
    description: String,
    report: Report,
    job: Option<McJob>,
    inputs: Option<Box<scan_kit_io::patient_view::PatientInputs>>,
    run: Option<McRun>,
}

impl ExportStage {
    fn new(dose: bool, root: &Path, sessions: &[String], options: &Value) -> Self {
        Self {
            dose,
            root: root.to_path_buf(),
            sessions: sessions.to_vec(),
            options: options.clone(),
            dest: options
                .get("dest")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            description: options
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("scan-kit MC")
                .to_owned(),
            report: Report {
                task: 0,
                generation: 0,
                phase: Phase::Read,
                done: 0,
                total: 0,
                note: "Loading study".into(),
            },
            job: None,
            inputs: None,
            run: None,
        }
    }

    fn fail(&mut self, message: String) -> Poll<Vec<u8>> {
        self.report.phase = Phase::Failed;
        self.report.note = message.clone();
        Poll::Failed(message)
    }

    /// The transport finished: weight the dose, write the file, and keep the
    /// cube so a following view or export opens instantly.
    fn finish(&mut self, result: McResult) -> Poll<Vec<u8>> {
        let Some(inputs) = self.inputs.as_ref() else {
            return self.fail("the export has no patient inputs".into());
        };
        if let Some(job) = &self.job {
            keep_runs(std::slice::from_ref(job), std::slice::from_ref(&result));
        }
        self.report.done = self.report.total;
        let run = scan_kit_io::patient_view::dose_from_result(inputs, Some(result));
        let gpu = crate::mc::adapter_info();
        let outcome = if self.dose {
            scan_kit_io::patient_view::write_patient_dose(
                inputs,
                &run,
                &self.sessions,
                &self.dest,
                &self.description,
                &gpu,
            )
        } else {
            crate::present::finish_patient_report(
                inputs,
                &run,
                &self.sessions,
                &self.options,
                &self.dest,
                &gpu,
            )
        };
        match outcome {
            Ok(value) => match serde_json::to_vec(&value) {
                Ok(bytes) => Poll::Ready(bytes),
                Err(err) => self.fail(err.to_string()),
            },
            Err(message) => self.fail(message),
        }
    }
}

impl Stage for ExportStage {
    fn report(&self) -> Report {
        self.report.clone()
    }

    fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>> {
        if cancel.is_cancelled() {
            self.report.phase = Phase::Cancelled;
            return Poll::Cancelled;
        }
        if self.run.is_none() && self.inputs.is_none() && self.dest.trim().is_empty() {
            // Fail before patient_inputs walks the study and McRun burns the
            // transport — the missing dest can only fail at write time.
            return self.fail("dest is required".into());
        }
        if self.run.is_none() {
            if self.inputs.is_none() {
                match scan_kit_io::patient_view::patient_inputs(
                    &self.root,
                    &self.sessions,
                    &self.options,
                ) {
                    Ok(inputs) => self.inputs = Some(Box::new(inputs)),
                    Err(message) => return self.fail(message),
                }
            }
            let inputs = self.inputs.as_ref().expect("patient inputs");
            let job = match scan_kit_io::patient_view::patient_request(inputs) {
                Ok(request) => McJob::Patient(request),
                Err(message) => return self.fail(message),
            };
            // The volumetric view may have already transported this request;
            // paint that cube instead of running the histories again.
            if let Some(mut results) = kept_results(std::slice::from_ref(&job)) {
                self.report.total = u64::from(inputs.histories);
                self.job = Some(job);
                return self.finish(results.swap_remove(0));
            }
            self.job = Some(job.clone());
            self.run = match McRun::open(&job) {
                Ok(run) => Some(run),
                Err(err) => return self.fail(err.to_string()),
            };
        }
        match self.run.as_mut().expect("transport").poll(budget, cancel) {
            Poll::Pending { report, .. } => {
                self.report = report;
                Poll::Pending {
                    report: self.report.clone(),
                    preview: None,
                }
            }
            Poll::Ready(result) => self.finish(result),
            Poll::Cancelled => {
                self.report.phase = Phase::Cancelled;
                Poll::Cancelled
            }
            Poll::Failed(message) => self.fail(message),
        }
    }
}

/// Latest seen stamp per session — a HashSet of raw stamps grew by one
/// permanent entry per file mtime change; keying by session identity keeps
/// it bounded by session count with the same "already read this state"
/// semantics.
fn seen_stamps() -> &'static Mutex<HashMap<String, String>> {
    static SEEN: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The `{root}\n{session}` identity prefix of a `stamp_key` — everything
/// after the second newline is the file-stamp portion.
fn stamp_id(stamp: &str) -> &str {
    stamp.rsplitn(3, '\n').nth(2).unwrap_or(stamp)
}

fn remember_stamps(stamps: &[String]) {
    let mut seen = seen_stamps().lock().unwrap_or_else(|err| err.into_inner());
    for stamp in stamps {
        seen.insert(stamp_id(stamp).to_owned(), stamp.clone());
    }
}

/// Length and mtime of the session files. The view is not part of the key:
/// timeline, bins, and the other pictures read the same files.
fn stamp_key(root: &Path, session: &str) -> String {
    // session_directory resolves nested layouts — root.join would stamp a
    // directory that never exists and freeze the "cached" flag on.
    // ponytail: timeslice-dir growth is deliberately excluded — stamping it
    // would invalidate the slice on every append and force a full reload per
    // poll; a timeslice-only session just keeps "cached" off.
    let dir = scan_kit_io::session_directory(root, session);
    format!(
        "{}\n{session}\n{}\n{}",
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
        // cached is computed on the first poll now — the constructor does no
        // file IO so it cannot run under the driver lock.
        match again.poll(Duration::from_millis(12), &cancel) {
            Poll::Ready(bytes) => {
                assert!(again.cached);
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

    #[test]
    fn a_long_spot_file_is_one_picture() {
        let root = scratch();
        let session = root.join("wide");
        std::fs::create_dir_all(&session).unwrap();
        let pad = "x".repeat(64);
        let mut map = String::from("energy,charge_req,position_x,position_y,note\n");
        let mut spots = String::from("ic1_total_dose,ic2_total_dose,position_x,position_y,note\n");
        for _ in 0..4_200 {
            map.push_str(&format!("180,0,0,0,{pad}\n"));
            spots.push_str(&format!("0,0,0,0,{pad}\n"));
        }
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        std::fs::write(session.join("spot_data.csv"), &spots).unwrap();
        // Wider than one 4096-row window, so the old per-window deposit would
        // still be pending after this poll.
        assert!(spots.len() > 300_000);
        let mut stage = SessionStage::new(
            "volumetric",
            &root,
            &["wide".into()],
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
                panic!("queued another picture at {}/{}", report.done, report.total)
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
    fn the_next_view_reuses_a_session_already_read() {
        let root = scratch();
        write_session(&root, "sess");
        let cancel = Cancel::new();
        let mut timeline = session_stage(&root, &["sess"]);
        assert!(!timeline.cached); // no stamps yet — computed on first poll
        let mut finished = false;
        for _ in 0..4 {
            if matches!(timeline.poll(Duration::ZERO, &cancel), Poll::Ready(_)) {
                finished = true;
                break;
            }
        }
        assert!(finished, "timeline did not finish");
        let mut bins = SessionStage::new(
            "bins",
            &root,
            &["sess".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        );
        match bins.poll(Duration::ZERO, &cancel) {
            Poll::Ready(bytes) => {
                assert!(bins.cached);
                assert_eq!(scan_kit_plot::plot_header(&bytes).unwrap().quality, "final");
            }
            other => panic!("a view of a session already read should be one picture: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_long_file_finishes_instead_of_repainting_the_trace() {
        let root = scratch();
        let session = root.join("wide");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n180,0,0,0\n",
        )
        .unwrap();
        let line = format!("0,0,0,0,{}\n", "x".repeat(64));
        let mut spots = String::with_capacity(line.len() * 40_000 + 80);
        spots.push_str("ic1_total_dose,ic2_total_dose,position_x,position_y,note\n");
        for _ in 0..40_000 {
            spots.push_str(&line);
        }
        // Wider than seven windows, so the third timed poll is not the end of the file.
        assert!(spots.len() > 1_400_000);
        std::fs::write(session.join("spot_data.csv"), spots).unwrap();
        let mut stage = session_stage(&root, &["wide"]);
        let cancel = Cancel::new();
        let mut partials = 0;
        let mut ready = false;
        for _ in 0..6 {
            match stage.poll(Duration::from_secs(1), &cancel) {
                Poll::Pending {
                    preview: Some(_), ..
                } => partials += 1,
                Poll::Pending { preview: None, .. } => {}
                Poll::Ready(bytes) => {
                    assert_eq!(scan_kit_plot::plot_header(&bytes).unwrap().quality, "final");
                    ready = true;
                    break;
                }
                other => panic!("a long file should publish or finish: {other:?}"),
            }
        }
        assert!(ready);
        assert_eq!(
            partials, 2,
            "two pictures, then the rest of the file in one"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_finished_transport_covers_the_same_jobs() {
        let _gpu = crate::mc::hold_gpu();
        let job = |histories: u32, energy: f32| {
            McJob::Slab(scan_kit_core::SlabRequest {
                medium: "water".into(),
                x: vec![0.0],
                y: vec![0.0],
                sx: vec![3.0],
                sy: vec![3.0],
                energy: vec![energy],
                protons: vec![1.0e6],
                histories,
                seed: 7,
                spread_pct: 1.0,
                wet_mm: 0.0,
                depth_mm: 100.0,
                voxel_mm: 4.0,
                origin: [0.0; 3],
                shape: [2, 2, 2],
            })
        };
        let jobs = [job(1_000_000, 150.0), job(1_000_000, 120.0)];
        let results: Vec<McResult> = jobs.iter().map(blank_result).collect();
        keep_runs(&jobs, &results);
        assert!(kept_results(&jobs).is_some());
        assert!(kept_results(&[job(10_000, 150.0), job(10_000, 120.0)]).is_some());
        assert!(kept_results(&[job(2_000_000, 150.0), job(2_000_000, 120.0)]).is_none());
        assert!(kept_results(&[job(1_000_000, 150.0), job(1_000_000, 121.0)]).is_none());
        assert!(kept_results(&jobs[..1]).is_none());
        *locked(kept_runs()) = None;
    }

    #[test]
    fn the_monte_carlo_button_starts_monte_carlo() {
        assert!(wants_mc("volumetric", &json!({ "model": "Monte Carlo" })));
        assert!(wants_mc("volumetric", &json!({ "model": "mc" })));
        assert!(!wants_mc("volumetric", &json!({ "model": "Analytic" })));
        assert!(!wants_mc("volumetric", &json!({})));
        assert!(!wants_mc("timeline", &json!({ "model": "Monte Carlo" })));
    }

    #[test]
    fn mc_view_paints_a_rough_cube_before_the_finished_one() {
        let _gpu = crate::mc::hold_gpu();
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
                Poll::Failed(message) if message.contains("no GPU adapter") => {
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

    #[test]
    fn a_voxel_change_replaces_the_monte_carlo_without_waiting_it_out() {
        let _gpu = crate::mc::hold_gpu();
        let root = scratch();
        write_session(&root, "sess");
        let background = [0.1, 0.1, 0.1, 1.0];
        let foreground = [0.9, 0.9, 0.9, 1.0];
        let palette = [[0.2, 0.4, 0.8, 1.0]];
        let first = start_task(
            "volumetric",
            &root,
            &["sess".into()],
            &json!({"model": "mc", "histories": "2000000", "voxel": "4"}),
            background,
            foreground,
            &palette,
        )
        .unwrap();
        let id = first["task"].as_u64().unwrap();
        let _ = poll_task(id);
        let started = std::time::Instant::now();
        let second = start_task(
            "volumetric",
            &root,
            &["sess".into()],
            &json!({"model": "mc", "histories": "2000000", "voxel": "1"}),
            background,
            foreground,
            &palette,
        )
        .unwrap();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the new voxel size waited on the old monte carlo"
        );
        let next = second["task"].as_u64().unwrap();
        assert_ne!(next, id);
        cancel_task(next, None);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn poll_report(bytes: &[u8]) -> Value {
        let length = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        serde_json::from_slice(&bytes[4..4 + length]).unwrap()
    }

    #[test]
    fn an_export_fails_on_a_missing_study_and_clears_the_task() {
        let root = scratch();
        let started = start_task(
            "write_dose",
            &root,
            &[],
            &json!({
                "study": root.join("missing").display().to_string(),
                "dest": root.join("dose.dcm").display().to_string(),
            }),
            [0.0; 4],
            [0.0; 4],
            &[],
        )
        .unwrap();
        let id = started["task"].as_u64().unwrap();
        let report = poll_report(&poll_task(id).unwrap());
        assert_eq!(report["phase"], "failed");
        assert_eq!(report["finished"], true);
        // The failed task is gone, not parked in the export slot.
        let driver = lock();
        assert!(driver.export.is_none());
        assert!(!driver.tasks.contains_key(&id));
        drop(driver);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_second_export_replaces_the_first_and_leaves_the_plot_slot() {
        let root = scratch();
        let options = json!({
            "study": root.join("missing").display().to_string(),
            "dest": root.join("out").display().to_string(),
        });
        let first =
            start_task("write_dose", &root, &[], &options, [0.0; 4], [0.0; 4], &[]).unwrap();
        let second = start_task(
            "write_report",
            &root,
            &[],
            &options,
            [0.0; 4],
            [0.0; 4],
            &[],
        )
        .unwrap();
        assert_ne!(first["task"], second["task"]);
        let driver = lock();
        assert!(!driver.tasks.contains_key(&first["task"].as_u64().unwrap()));
        assert!(driver.tasks.contains_key(&second["task"].as_u64().unwrap()));
        assert!(driver.plot.is_none());
        drop(driver);
        cancel_task(second["task"].as_u64().unwrap(), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    // Diagnostic, not a test: prints the session's uncertainty vs histories.
    #[test]
    #[ignore]
    fn session_uncertainty_curve() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/logs/1241407101/1241407101/input_map.csv");
        let text = std::fs::read_to_string(path).unwrap();
        let mut energy = Vec::new();
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut sx = Vec::new();
        let mut sy = Vec::new();
        let mut protons = Vec::new();
        for line in text.lines().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 6 {
                continue;
            }
            energy.push(cols[0].trim().parse::<f32>().unwrap());
            sx.push(cols[2].trim().parse::<f32>().unwrap());
            sy.push(cols[3].trim().parse::<f32>().unwrap());
            x.push(cols[4].trim().parse::<f32>().unwrap());
            y.push(cols[5].trim().parse::<f32>().unwrap());
            protons.push(cols[6].trim().parse::<f32>().unwrap().max(1e-6));
        }
        let job = McJob::Slab(scan_kit_core::SlabRequest {
            medium: "water".into(),
            x,
            y,
            sx,
            sy,
            energy,
            protons,
            histories: 10_000_000,
            seed: 1,
            spread_pct: 0.0,
            wet_mm: 0.0,
            depth_mm: 400.0,
            voxel_mm: 2.0,
            origin: [-130.0, -130.0, -410.0],
            shape: [130, 130, 205],
        });
        let mut run = McRun::open(&job).unwrap();
        let cancel = Cancel::new();
        let start = std::time::Instant::now();
        loop {
            match run.poll(Duration::from_millis(200), &cancel) {
                Poll::Pending { report, .. } => {
                    println!("{:>6.1}s {}", start.elapsed().as_secs_f64(), report.note);
                }
                Poll::Ready(result) => {
                    println!(
                        "READY {:.1}s ±{:.2}%",
                        start.elapsed().as_secs_f64(),
                        result.uncertainty * 100.0
                    );
                    break;
                }
                Poll::Cancelled => panic!("cancelled"),
                Poll::Failed(m) => panic!("{m}"),
            }
        }
    }

    // Diagnostic, not a test: dumps the real McJobs a session view builds.
    #[test]
    #[ignore]
    fn session_job_dump() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/logs/1241407101");
        let (_scene, jobs) = capture_job(
            &root,
            &["1241407101".into()],
            &json!({"model": "mc", "voxel": "2"}),
        );
        println!("{} jobs", jobs.len());
        for job in &jobs {
            let McJob::Slab(r) = job else { continue };
            let total: f64 = r.protons.iter().map(|&p| f64::from(p)).sum();
            let mean_e: f64 = r
                .energy
                .iter()
                .zip(&r.protons)
                .map(|(&e, &p)| f64::from(e) * f64::from(p))
                .sum::<f64>()
                / total.max(1e-30);
            let e_max = r.energy.iter().cloned().fold(0.0f32, f32::max);
            println!(
                "spots={} histories={} voxel={} shape={:?} depth={} wet={} spread={} meanE={:.0} maxE={:.0}",
                r.energy.len(), r.histories, r.voxel_mm, r.shape, r.depth_mm, r.wet_mm, r.spread_pct, mean_e, e_max
            );
            let cancel = Cancel::new();
            let mut run = McRun::open(job).unwrap();
            let start = std::time::Instant::now();
            loop {
                match run.poll(Duration::from_millis(500), &cancel) {
                    Poll::Pending { report, .. } => {
                        println!("  {:>6.1}s {}", start.elapsed().as_secs_f64(), report.note);
                    }
                    Poll::Ready(result) => {
                        println!(
                            "  READY {:.1}s ±{:.2}%",
                            start.elapsed().as_secs_f64(),
                            result.uncertainty * 100.0
                        );
                        break;
                    }
                    Poll::Cancelled => panic!("cancelled"),
                    Poll::Failed(m) => panic!("{m}"),
                }
            }
        }
    }

    // Diagnostic, not a test: times each phase of the volumetric task on a
    // real multi-spot session to find where the wall time goes.
    #[test]
    #[ignore]
    fn session_task_phase_timing() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/logs/1241407101");
        let started = start_task(
            "volumetric",
            &root,
            &["1241407101".into()],
            &json!({"model": "mc", "histories": "10000000", "voxel": "2"}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.2, 0.4, 0.8, 1.0]],
        )
        .unwrap();
        let id = started["task"].as_u64().unwrap();
        let start = std::time::Instant::now();
        let mut last = String::new();
        for _ in 0..3600 {
            let bytes = poll_task(id).unwrap();
            let length = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
            let report: Value = serde_json::from_slice(&bytes[4..4 + length]).unwrap();
            let line = format!(
                "{} {} {}/{} {}",
                report["phase"],
                report["finished"],
                report["done"],
                report["total"],
                report["note"]
            );
            if line != last {
                println!("{:>7.1}s {line}", start.elapsed().as_secs_f64());
                last = line;
            }
            if report["finished"].as_bool().unwrap_or(false) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        cancel_task(id, None);
    }

    #[test]
    fn drain_task_drives_the_stage_to_a_terminal_error() {
        let root = scratch();
        let message = drain_task(
            "write_dose",
            &root,
            &[],
            &json!({
                "study": root.join("missing").display().to_string(),
                "dest": root.join("dose.dcm").display().to_string(),
            }),
        )
        .expect_err("a missing study cannot export");
        assert_ne!(message, "plot did not finish");
    }

    #[test]
    fn an_export_without_dest_fails_before_the_transport() {
        // The Tauri surface can start an export with no dest — it used to
        // run the whole Monte Carlo and only then fail on an empty path.
        let root = scratch();
        let message = drain_task("write_dose", &root, &[], &json!({}))
            .expect_err("a missing dest cannot export");
        assert!(message.contains("dest is required"), "{message}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_plot_api_rejects_task_views_without_writing() {
        // `scan_kit_run_view` is a READ_ONLY MCP tool, so task views that write
        // files must fail here before a stage runs — the desktop reaches the
        // same stages through `start_task` instead.
        let root = scratch();
        for view in ["write_dose", "write_report", "library", "runner_copy"] {
            let message = crate::present::open_plot(
                view,
                &root,
                &[],
                &json!({
                    "study": root.join("missing").display().to_string(),
                    "dest": root.join("out.dcm").display().to_string(),
                }),
                [0.0; 4],
                [0.0; 4],
                &[],
            )
            .expect_err("task views never produce plot bytes");
            assert!(message.contains("not a plot"), "{view}: {message}");
        }
        assert!(!root.join("out.dcm").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_panicking_poll_ends_the_task_instead_of_parking_it() {
        // Without catch_unwind the checked-out stage is lost mid-unwind and the
        // task answers Pending forever with its slot still held.
        struct Panic;
        impl Stage for Panic {
            fn poll(&mut self, _budget: Duration, _cancel: &Cancel) -> Poll<Vec<u8>> {
                panic!("inside the stage");
            }
            fn report(&self) -> Report {
                report_at(Phase::Scene, 0, 1)
            }
        }
        let id = {
            let mut driver = lock();
            let id = driver.next;
            driver.next = driver.next.saturating_add(1);
            driver.tasks.insert(
                id,
                Running {
                    generation: 1,
                    cancel: Cancel::new(),
                    last_report: None,
                    stage: Some(Box::new(Panic)),
                },
            );
            id
        };
        let report = poll_report(&poll_task(id).unwrap());
        assert_eq!(report["phase"], "failed");
        assert_eq!(report["finished"], true);
        let driver = lock();
        assert!(!driver.tasks.contains_key(&id));
    }

    #[test]
    fn the_stamp_cache_keeps_one_entry_per_session() {
        // Unique id: SEEN is shared across parallel tests, so size assertions
        // on the whole map would race.
        let first = "/root\nstamp-cache-test\n1:100\n2:200";
        let second = "/root\nstamp-cache-test\n3:300\n4:400";
        assert_eq!(stamp_id(first), "/root\nstamp-cache-test");
        remember_stamps(&[first.to_owned()]);
        {
            let seen = seen_stamps().lock().unwrap_or_else(|err| err.into_inner());
            assert!(seen.get(stamp_id(first)) == Some(&first.to_owned()));
        }
        // A newer stamp replaces, not accumulates.
        remember_stamps(&[second.to_owned()]);
        let seen = seen_stamps().lock().unwrap_or_else(|err| err.into_inner());
        assert_eq!(seen.get(stamp_id(first)), Some(&second.to_owned()));
    }

    #[test]
    fn a_panicking_report_ends_the_task_instead_of_parking_it() {
        // report() runs outside the poll guard in the bugged code — a panic
        // there left stage=None and every later poll fabricated Pending.
        struct Panic;
        impl Stage for Panic {
            fn poll(&mut self, _budget: Duration, _cancel: &Cancel) -> Poll<Vec<u8>> {
                Poll::Pending {
                    report: report_at(Phase::Scene, 0, 1),
                    preview: None,
                }
            }
            fn report(&self) -> Report {
                panic!("inside the report");
            }
        }
        let id = {
            let mut driver = lock();
            let id = driver.next;
            driver.next = driver.next.saturating_add(1);
            driver.tasks.insert(
                id,
                Running {
                    generation: 1,
                    cancel: Cancel::new(),
                    stage: Some(Box::new(Panic)),
                    last_report: None,
                },
            );
            id
        };
        let report = poll_report(&poll_task(id).unwrap());
        assert_eq!(report["phase"], "failed");
        assert_eq!(report["finished"], true);
        assert!(!lock().tasks.contains_key(&id));
    }

    #[test]
    fn a_stale_generations_cancel_leaves_the_successor() {
        // Retargets reuse the task id — a superseded ticket's cancel must not
        // kill the generation it no longer owns.
        struct Idle;
        impl Stage for Idle {
            fn poll(&mut self, _budget: Duration, _cancel: &Cancel) -> Poll<Vec<u8>> {
                Poll::Pending {
                    report: report_at(Phase::Scene, 0, 1),
                    preview: None,
                }
            }
            fn report(&self) -> Report {
                report_at(Phase::Scene, 0, 1)
            }
        }
        let id = {
            let mut driver = lock();
            let id = driver.next;
            driver.next = driver.next.saturating_add(1);
            driver.tasks.insert(
                id,
                Running {
                    generation: 2,
                    cancel: Cancel::new(),
                    stage: Some(Box::new(Idle)),
                    last_report: None,
                },
            );
            id
        };
        cancel_task(id, Some(1));
        assert!(lock().tasks.contains_key(&id), "stale cancel ran");
        cancel_task(id, Some(2));
        assert!(!lock().tasks.contains_key(&id), "owner cancel missed");
    }

    #[test]
    fn a_blank_result_bounds_the_placeholder_shape() {
        // An unchecked request shape must not allocate its product — the
        // placeholder collapses to empty past the cap.
        let job = McJob::Slab(scan_kit_core::SlabRequest {
            medium: "water".into(),
            x: vec![0.0],
            y: vec![0.0],
            sx: vec![3.0],
            sy: vec![3.0],
            energy: vec![100.0],
            protons: vec![1.0e6],
            histories: 1,
            seed: 7,
            spread_pct: 1.0,
            wet_mm: 0.0,
            depth_mm: 100.0,
            voxel_mm: 4.0,
            origin: [0.0; 3],
            shape: [1 << 31, 2, 2],
        });
        let result = blank_result(&job);
        assert_eq!(result.volume.shape, [0, 0, 0]);
        assert!(result.volume.values.is_empty());
    }
}
