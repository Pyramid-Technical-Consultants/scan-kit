//! Host for the MCsquare transport shader. A run returns the finished dose.
//!
//! [`McRun`] is the sliced form of the same transport. [`run_mc`] drains it.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use scan_kit_core::{
    adapt_chunk, Cancel, McJob, McResult, PatientRequest, Phase, Poll, Report, SlabRequest, Volume,
};
use scan_kit_io::mc_tables;

use crate::ComputeError;

const TRANSPORT: &str = include_str!("mc_transport.wgsl");
const FOLD: &str = r#"
struct Fold {
    n: u32,
    mode: i32,
    scale: f32,
    gain: f32,
}
@group(0) @binding(0) var<storage, read_write> tally: array<u32>;
@group(0) @binding(1) var<storage, read_write> dose_sum: array<f32>;
@group(0) @binding(2) var<storage, read_write> dose_sq: array<f32>;
@group(0) @binding(3) var<storage, read_write> dose_out: array<f32>;
@group(0) @binding(4) var<uniform> P: Fold;
// stats: 0 dose_sum peak bits, 1-2 rel sum (u64, 1e-6 quanta), 3 rel count,
// 4 batch count written by the host. Reduces what the CPU used to read back.
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= P.n) { return; }
    let lo = tally[2u * i];
    let hi = tally[2u * i + 1u];
    var q = f32(bitcast<i32>(hi)) * 4294967296.0 + f32(lo);
    if (hi == 0xFFFFFFFFu && lo >= 0x80000000u) { q = f32(bitcast<i32>(lo)); }
    let b = q * P.scale;
    if (P.mode == 2) {
        dose_out[i] = (dose_sum[i] + b) * P.gain;
        return;
    }
    tally[2u * i] = 0u;
    tally[2u * i + 1u] = 0u;
    let s = dose_sum[i] + b;
    dose_sum[i] = s;
    dose_sq[i] += b * b;
    if (P.mode == 1) { dose_out[i] = s * P.gain; }
}

var<workgroup> red_s: array<f32, 256>;
var<workgroup> red_c: array<u32, 256>;

// stats_peak then stats_rel in the same pass: peak of dose_sum, then the relative
// spread of voxels above half of it, so the host reads 16 bytes, not 2 cubes.
@compute @workgroup_size(256)
fn stats_peak(
    @builtin(global_invocation_id) g: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = g.x + g.y * nwg.x * 256u;
    var v = 0.0;
    if (i < P.n) { v = max(dose_sum[i], 0.0); }
    red_s[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { red_s[lid] = max(red_s[lid], red_s[lid + s]); }
        workgroupBarrier();
    }
    if (lid == 0u) { atomicMax(&stats[0u], bitcast<u32>(red_s[0])); }
}

@compute @workgroup_size(256)
fn stats_rel(
    @builtin(global_invocation_id) g: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = g.x + g.y * nwg.x * 256u;
    let n = f32(atomicLoad(&stats[4u]));
    var acc = 0.0;
    var cnt = 0u;
    if (i < P.n && n > 0.0) {
        let dose = dose_sum[i];
        if (dose > 0.5 * bitcast<f32>(atomicLoad(&stats[0u]))) {
            acc = sqrt(max(n * (dose_sq[i] * n / (dose * dose) - 1.0), 0.0)) / n;
            cnt = 1u;
        }
    }
    red_s[lid] = acc;
    red_c[lid] = cnt;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            red_s[lid] += red_s[lid + s];
            red_c[lid] += red_c[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        // An out-of-range float→u32 cast is a dynamic error — clamp first so a
        // huge accumulation can't wrap into a valid-looking dose sum.
        let q = u32(min(red_s[0] * 1e6, 4294967295.0));
        let old = atomicAdd(&stats[1u], q);
        if (old + q < old) { atomicAdd(&stats[2u], 1u); }
        atomicAdd(&stats[3u], red_c[0]);
    }
}
"#;

/// How often a running transport copies the cube for the view.
const PREVIEW_GAP: Duration = Duration::from_millis(200);
/// Transport chunks a run keeps queued before its step paces on the oldest.
/// Deep enough that sibling polls and preview copies never find the GPU dry.
const MAX_INFLIGHT: usize = 8;

/// The suite shares one device. A test holds this for its whole body so two
/// polls never overlap; an overlapping read made a voxel change look stuck.
#[cfg(test)]
pub(crate) fn hold_gpu() -> std::sync::MutexGuard<'static, ()> {
    static HELD: Mutex<()> = Mutex::new(());
    HELD.lock().unwrap_or_else(|err| err.into_inner())
}

const QUANTUM_MEV: f64 = 1e-4;
const MEV_TO_J: f64 = 1.602176634e-13;

struct Launch {
    spots: Vec<f32>,
    protons: f64,
    mode: i32,
    flags: u32,
    origin_cm: [f32; 3],
    spacing_cm: [f32; 3],
    shape: [usize; 3],
    depth_cm: f32,
    wet_cm: f32,
    medium: i32,
    histories: u32,
    seed: u32,
    material: Vec<u32>,
    density: Vec<f32>,
    beams: Vec<f32>,
    origin_mm: [f32; 3],
}

/// GPU identity for dose provenance — `scan_kit.gpu.adapter_info`.
pub fn adapter_info() -> String {
    block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = tokio::time::timeout(
            Duration::from_secs(30),
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .await
        .ok()?
        .ok()?;
        let info = adapter.get_info();
        Some(format!("{} / {:?}", info.name, info.backend))
    })
    .unwrap_or_default()
}

pub fn run_mc(job: &McJob) -> Result<McResult, ComputeError> {
    let mut run = McRun::open(job)?;
    match run.poll(Duration::MAX, &Cancel::new()) {
        Poll::Ready(result) => Ok(result),
        Poll::Failed(message) => Err(ComputeError::Message(message)),
        Poll::Cancelled => Err(ComputeError::Message("cancelled".into())),
        Poll::Pending { .. } => Err(ComputeError::Message("monte carlo did not finish".into())),
    }
}

/// One Monte Carlo transport. An unlimited poll runs every history. A short poll
/// queues one chunk and returns while the GPU runs it.
pub struct McRun {
    live: Option<Live>,
    done: Option<McResult>,
}

impl McRun {
    pub fn open(job: &McJob) -> Result<Self, ComputeError> {
        let launch = match job {
            McJob::Slab(request) => slab_launch(request)?,
            McJob::Patient(request) => patient_launch(request)?,
        };
        if launch.protons <= 0.0 || launch.spots.is_empty() {
            // The empty path skips prepare's binding-cap check — bound the
            // placeholder allocation itself.
            let nvox = launch.shape[0]
                .max(1)
                .saturating_mul(launch.shape[1].max(1))
                .saturating_mul(launch.shape[2].max(1));
            if nvox > 1 << 28 {
                return Err(ComputeError::Message("voxel grid is too large".into()));
            }
            return Ok(Self {
                live: None,
                done: Some(empty_result(&launch)),
            });
        }
        Ok(Self {
            live: Some(block_on(prepare(launch))?),
            done: None,
        })
    }

    pub fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<McResult> {
        if cancel.is_cancelled() {
            return Poll::Cancelled;
        }
        if let Some(result) = &self.done {
            return Poll::Ready(result.clone());
        }
        let Some(live) = self.live.as_mut() else {
            return Poll::Failed("monte carlo has no buffers".into());
        };
        if budget >= Duration::from_secs(30) {
            return match live.drain(cancel) {
                Ok(result) => {
                    self.done = Some(result.clone());
                    Poll::Ready(result)
                }
                // A cancel landing inside the drain aborts it — that is a
                // cancellation, not a failure.
                Err(err) if cancel.is_cancelled() && err.to_string() == "cancelled" => {
                    Poll::Cancelled
                }
                Err(err) => Poll::Failed(err.to_string()),
            };
        }
        match live.step(budget, cancel) {
            Ok(Slice::Pending(preview)) => {
                let report = live.report(preview.is_some());
                Poll::Pending { report, preview }
            }
            Ok(Slice::Ready(result)) => {
                self.done = Some(result.clone());
                Poll::Ready(result)
            }
            Ok(Slice::Cancelled) => Poll::Cancelled,
            // A cancel landing inside finish()'s read aborts it — same
            // treatment the drain arm gives the identical error.
            Err(err) if cancel.is_cancelled() && err.to_string() == "cancelled" => Poll::Cancelled,
            Err(err) => Poll::Failed(err.to_string()),
        }
    }

    /// Keep the tallies and aim at `histories` when that is still ahead.
    pub fn extend(&mut self, histories: u32) -> bool {
        let Some(live) = self.live.as_mut() else {
            return false;
        };
        let target = histories
            .div_ceil(live.per_batch)
            .max(1)
            .saturating_mul(live.per_batch);
        if target < live.next {
            return false;
        }
        if target != live.histories {
            live.histories = target;
            live.finished = false;
            self.done = None;
        }
        true
    }

    pub fn progress(&self) -> f64 {
        if self.done.is_some() {
            return 1.0;
        }
        let Some(live) = &self.live else {
            return 1.0;
        };
        if live.histories == 0 {
            1.0
        } else {
            f64::from(live.next) / f64::from(live.histories)
        }
    }
}

enum Slice {
    Pending(Option<McResult>),
    Ready(McResult),
    Cancelled,
}

struct ReadTicket {
    index: wgpu::SubmissionIndex,
    /// One buffer per read: a dropped ticket can never leave a later read's
    /// staging mid-map.
    staging: wgpu::Buffer,
    receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    histories: u32,
}

struct Live {
    gpu: Arc<McGpu>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    transport_pipe: wgpu::ComputePipeline,
    transport_group: wgpu::BindGroup,
    fold_pipe: wgpu::ComputePipeline,
    stats_peak_pipe: wgpu::ComputePipeline,
    stats_rel_pipe: wgpu::ComputePipeline,
    fold_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    ledger: wgpu::Buffer,
    fold_params: wgpu::Buffer,
    out: wgpu::Buffer,
    stats: wgpu::Buffer,
    /// Buffers the bind groups borrow. Read so the field is not dead.
    kept: Vec<wgpu::Buffer>,
    base: [u8; 80],
    next: u32,
    histories: u32,
    per_batch: u32,
    chunk: u32,
    floor: u32,
    cap: u32,
    nvox: usize,
    scale: f32,
    shape: [usize; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
    /// Transport submissions queued but not yet known retired. A step only
    /// waits on the oldest once the window fills, so the GPU keeps a few
    /// chunks deep — the shared queue never drains dry between worker polls.
    inflight: VecDeque<wgpu::SubmissionIndex>,
    /// A preview copy already submitted. Its map lands whenever the copy
    /// retires — collecting it never blocks a step.
    pending_read: Option<ReadTicket>,
    finished: bool,
    previews: u32,
    last_preview: Instant,
    cached: Option<McResult>,
    /// The last readback's uncertainty for the report — cloning a whole cube
    /// per preview just for this f32 was the cost before.
    cached_uncertainty: f32,
    score_let: bool,
    let_buf: wgpu::Buffer,
}

fn slab_launch(request: &SlabRequest) -> Result<Launch, ComputeError> {
    let tables = mc_tables();
    let medium = tables
        .media
        .iter()
        .position(|name| name == &request.medium)
        .ok_or_else(|| ComputeError::Message(format!("no material {}", request.medium)))?
        as i32;
    let n = request.energy.len();
    let mut spots = Vec::new();
    let mut weights = Vec::new();
    for i in 0..n {
        let energy = request.energy.get(i).copied().unwrap_or(f32::NAN);
        let weight = request.protons.get(i).copied().unwrap_or(0.0);
        let x = request.x.get(i).copied().unwrap_or(f32::NAN);
        let y = request.y.get(i).copied().unwrap_or(f32::NAN);
        let sx = request.sx.get(i).copied().unwrap_or(0.0);
        let sy = request.sy.get(i).copied().unwrap_or(0.0);
        if energy.is_finite()
            && weight.is_finite()
            && x.is_finite()
            && y.is_finite()
            && energy > 0.0
            && weight > 0.0
        {
            spots.extend([
                x / 10.0,
                y / 10.0,
                // Same 0.2 mm floor the analytic deposit uses — a zero
                // sigma spot must not be pencil-thin here and Gaussian there.
                sx.max(0.2) / 10.0,
                sy.max(0.2) / 10.0,
                energy,
                request.spread_pct.max(0.0) / 100.0 * energy,
                0.0,
                0.0,
            ]);
            weights.push(f64::from(weight));
        }
    }
    let protons = weights.iter().sum::<f64>();
    if protons > 0.0 {
        let mut cdf = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            cdf += weight / protons;
            spots[index * 8 + 6] = cdf as f32;
        }
        let last = spots.len() - 2;
        spots[last] = 1.0;
    }
    let voxel = request.voxel_mm.max(0.25);
    let cm = voxel / 10.0;
    Ok(Launch {
        spots,
        protons,
        mode: 0,
        flags: 0,
        origin_cm: request.origin.map(|v| v / 10.0),
        spacing_cm: [cm, cm, cm],
        shape: request.shape,
        depth_cm: request.depth_mm.max(0.0) / 10.0,
        wet_cm: request.wet_mm.max(0.0) / 10.0,
        medium,
        histories: request.histories,
        seed: request.seed,
        material: vec![0],
        density: vec![0.0],
        beams: vec![0.0; 16],
        origin_mm: request.origin,
    })
}

fn patient_launch(request: &PatientRequest) -> Result<Launch, ComputeError> {
    // The shader clamps each axis to >=1 — the material/density check must
    // read the same shape or a zero axis validates 0-length buffers the GPU
    // then indexes out of bounds.
    let [nx, ny, nz] = request.shape.map(|dim| dim.max(1));
    let nvox = nx.saturating_mul(ny).saturating_mul(nz);
    if request.material.len() != nvox || request.density.len() != nvox {
        return Err(ComputeError::Message("material and density differ".into()));
    }
    let media = mc_tables().media.len();
    if request.material.iter().any(|&m| m as usize >= media) {
        return Err(ComputeError::Message("material index out of range".into()));
    }
    if request.density.iter().any(|d| !d.is_finite() || *d <= 0.0) {
        return Err(ComputeError::Message(
            "density must be finite and positive".into(),
        ));
    }
    // smx/smy are divisors in the shader — a zero is finite but produces a
    // NaN particle whose voxel index is indeterminate, depositing into a
    // real voxel with no error.
    if !request.beams.len().is_multiple_of(16)
        || !request.beams.iter().all(|value| value.is_finite())
        || request
            .beams
            .as_chunks::<16>()
            .0
            .iter()
            .any(|r| r[13] == 0.0 || r[14] == 0.0)
    {
        return Err(ComputeError::Message(
            "beams must be whole finite 16-float records".into(),
        ));
    }
    if !request.spots.len().is_multiple_of(40) {
        return Err(ComputeError::Message(
            "spots must be whole 40-float records".into(),
        ));
    }
    let mut kept_spots = Vec::new();
    let mut weights = Vec::new();
    let rows = request.spots.len() / 40;
    if request.protons.len() != rows {
        return Err(ComputeError::Message(
            "protons must carry one weight per spot record".into(),
        ));
    }
    for row in 0..rows {
        let weight = request.protons.get(row).copied().unwrap_or(0.0);
        if weight.is_finite() && weight > 0.0 {
            let record = &request.spots[row * 40..(row + 1) * 40];
            if !record.iter().all(|value| value.is_finite()) {
                return Err(ComputeError::Message(
                    "spot record holds a non-finite field".into(),
                ));
            }
            let beam = record[6];
            if beam < 0.0 || beam.fract() != 0.0 || beam as usize >= request.beams.len() / 16 {
                return Err(ComputeError::Message("spot beam index out of range".into()));
            }
            // spots[32..35] = range shifter (lo, material, rho); a shifted
            // spot's material indexes mc_tables().media like slab_launch's.
            if record[32] > 0.0 {
                if !(0.0..media as f32).contains(&record[34]) || record[34].fract() != 0.0 {
                    return Err(ComputeError::Message(
                        "spot range-shifter material out of range".into(),
                    ));
                }
                if record[35] <= 0.0 {
                    return Err(ComputeError::Message(
                        "spot range-shifter density must be positive".into(),
                    ));
                }
            }
            kept_spots.extend_from_slice(record);
            weights.push(f64::from(weight));
        }
    }
    let protons = weights.iter().sum::<f64>();
    if protons > 0.0 {
        let mut cdf = 0.0;
        let n = weights.len();
        for (index, weight) in weights.iter().enumerate() {
            cdf += weight / protons;
            kept_spots[index * 40 + 5] = if index + 1 == n { 1.0 } else { cdf as f32 };
        }
    }
    let words = nvox.div_ceil(4);
    let mut packed = vec![0u8; words * 4];
    packed[..nvox].copy_from_slice(&request.material);
    let material: Vec<u32> = packed
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect();
    let spacing = request.spacing_mm.map(|mm| mm.max(0.25) / 10.0);
    Ok(Launch {
        spots: kept_spots,
        protons,
        mode: 1,
        flags: u32::from(request.dose_to_water)
            | (u32::from(request.score_let) * 2)
            | (u32::from(request.let_water) * 4),
        origin_cm: request.origin_mm.map(|mm| mm / 10.0),
        spacing_cm: spacing,
        shape: request.shape,
        depth_cm: 0.0,
        wet_cm: 0.0,
        medium: 0,
        histories: request.histories,
        seed: request.seed,
        material,
        density: request.density.clone(),
        beams: if request.beams.is_empty() {
            vec![0.0; 16]
        } else {
            request.beams.clone()
        },
        origin_mm: request.origin_mm,
    })
}

fn empty_result(launch: &Launch) -> McResult {
    let [nx, ny, nz] = launch.shape;
    let shape = [nx.max(1), ny.max(1), nz.max(1)];
    // Size values from the clamped shape so product(shape) == values.len().
    let nvox = shape[0].saturating_mul(shape[1]).saturating_mul(shape[2]);
    McResult {
        volume: Volume {
            origin: launch.origin_mm,
            shape,
            spacing: launch.spacing_cm.map(|cm| cm * 10.0),
            values: vec![0.0; nvox],
        },
        uncertainty: 0.0,
        ledger: [0.0; 6],
        let_d: Vec::new(),
    }
}

struct McGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    transport_layout: wgpu::BindGroupLayout,
    fold_layout: wgpu::BindGroupLayout,
    transport_pipe: wgpu::ComputePipeline,
    fold_pipe: wgpu::ComputePipeline,
    stats_peak_pipe: wgpu::ComputePipeline,
    stats_rel_pipe: wgpu::ComputePipeline,
}

/// One device and one compiled transport for the process. Opening a device and
/// the transport shader is the slow part of switching into Monte Carlo.
/// ponytail: one shared queue. Two runs at once interleave submits on it.
static GPU: Mutex<Option<Arc<McGpu>>> = Mutex::new(None);

/// A wedged or lost device would otherwise poison every future run — drop the
/// cache so the next open builds a fresh device instead of cloning a dead one.
fn gpu_lost(gpu: &Arc<McGpu>) {
    let mut cached = GPU.lock().unwrap_or_else(|err| err.into_inner());
    if cached.as_ref().is_some_and(|held| Arc::ptr_eq(held, gpu)) {
        *cached = None;
    }
}

async fn shared_gpu() -> Result<Arc<McGpu>, ComputeError> {
    static OPENING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    if let Some(gpu) = GPU
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .as_ref()
        .map(Arc::clone)
    {
        return Ok(gpu);
    }
    // Two cold-start opens on different worker threads would each build a
    // device — serialize the slow path, then check once more.
    let _opening = OPENING.lock().await;
    if let Some(gpu) = GPU
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .as_ref()
        .map(Arc::clone)
    {
        return Ok(gpu);
    }
    let gpu = Arc::new(open_gpu().await?);
    *GPU.lock().unwrap_or_else(|err| err.into_inner()) = Some(Arc::clone(&gpu));
    Ok(gpu)
}

async fn open_gpu() -> Result<McGpu, ComputeError> {
    let (device, queue) = mc_device().await?;
    let transport = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc transport"),
        source: wgpu::ShaderSource::Wgsl(TRANSPORT.into()),
    });
    let fold = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc fold"),
        source: wgpu::ShaderSource::Wgsl(FOLD.into()),
    });
    let transport_layout = layout(
        &device,
        &[
            store(true),
            store(true),
            store(true),
            store(false),
            store(false),
            uniform(),
            store(true),
            store(true),
            store(false),
            store(true),
        ],
    );
    let fold_layout = layout(
        &device,
        &[
            store(false),
            store(false),
            store(false),
            store(false),
            uniform(),
            store(false),
        ],
    );
    Ok(McGpu {
        transport_pipe: pipeline(&device, &transport_layout, &transport, "main"),
        fold_pipe: pipeline(&device, &fold_layout, &fold, "main"),
        stats_peak_pipe: pipeline(&device, &fold_layout, &fold, "stats_peak"),
        stats_rel_pipe: pipeline(&device, &fold_layout, &fold, "stats_rel"),
        device,
        queue,
        transport_layout,
        fold_layout,
    })
}

async fn prepare(launch: Launch) -> Result<Live, ComputeError> {
    let [nx, ny, nz] = launch.shape;
    let shape = [nx.max(1), ny.max(1), nz.max(1)];
    let nvox = shape[0].saturating_mul(shape[1]).saturating_mul(shape[2]);
    let gpu = shared_gpu().await?;
    // The LET buffer binds nvox*16 bytes — a grid whose largest buffer beats
    // the storage binding limit fails create_buffer validation as a panic,
    // so bound it here where the failure is a clean ComputeError.
    let limits = gpu.device.limits();
    let binding_cap = limits.max_storage_buffer_binding_size as usize;
    // The LET buffer binds nvox*16 but readback stages nvox*20+80 — both
    // caps bind the run, and max_buffer_size can sit near the binding limit.
    let staging = nvox.saturating_mul(20).saturating_add(80);
    if nvox > binding_cap / 16 || staging > limits.max_buffer_size as usize {
        return Err(ComputeError::Message(format!(
            "voxel grid {nx}x{ny}x{nz} is too large for this GPU"
        )));
    }
    // spots/beams bind whole — oversized ones hit create_bind_group
    // validation as a panic, same class as the nvox guard above.
    if launch.spots.len().saturating_mul(4) > binding_cap
        || launch.beams.len().saturating_mul(4) > binding_cap
    {
        return Err(ComputeError::Message(
            "job buffers exceed the GPU binding limit".into(),
        ));
    }
    let device = gpu.device.clone();
    let queue = gpu.queue.clone();
    let transport_layout = gpu.transport_layout.clone();
    let fold_layout = gpu.fold_layout.clone();
    let transport_pipe = gpu.transport_pipe.clone();
    let fold_pipe = gpu.fold_pipe.clone();
    let stats_peak_pipe = gpu.stats_peak_pipe.clone();
    let stats_rel_pipe = gpu.stats_rel_pipe.clone();
    let tables = mc_tables();
    let histories = batch_count(launch.histories);
    let per_batch = (histories / 10).clamp(1, 100_000);
    // Round up, not down: satisfied must reach the requested target when the
    // run finishes — a floored count leaves paint_progress Pending forever.
    let histories = histories.div_ceil(per_batch).saturating_mul(per_batch);
    let volume_cm3 = f64::from(launch.spacing_cm[0])
        * f64::from(launch.spacing_cm[1])
        * f64::from(launch.spacing_cm[2]);
    let scale = (QUANTUM_MEV * launch.protons * MEV_TO_J / (volume_cm3 * 1e-3)) as f32;

    let spots = storage_init(&device, &f32_bytes(&launch.spots));
    let floats = storage_init(&device, &f32_bytes(&tables.floats));
    let ints = storage_init(&device, &i32_bytes(&tables.ints));
    let tally = storage_zero(&device, (nvox * 8) as u64);
    let ledger = storage_zero(&device, 64);
    let params = uniform_buf(&device, 80);
    let material = storage_init(&device, &u32_bytes(&launch.material));
    let density = storage_init(&device, &f32_bytes(&launch.density));
    let score_let = launch.flags & 2 != 0;
    let let_buf = storage_zero(&device, if score_let { (nvox * 16) as u64 } else { 4 });
    let beams = storage_init(&device, &f32_bytes(&launch.beams));
    let sum = storage_zero(&device, (nvox * 4) as u64);
    let sq = storage_zero(&device, (nvox * 4) as u64);
    let out = storage_zero(&device, (nvox * 4) as u64);
    let fold_params = uniform_buf(&device, 16);
    let stats = storage_zero(&device, 32);

    let transport_group = bind(
        &device,
        &transport_layout,
        &[
            &spots, &floats, &ints, &tally, &ledger, &params, &material, &density, &let_buf, &beams,
        ],
    );
    let fold_group = bind(
        &device,
        &fold_layout,
        &[&tally, &sum, &sq, &out, &fold_params, &stats],
    );

    let mut base = [0u8; 80];
    write_params(
        &mut base,
        &launch,
        launch.spots.len() / if launch.mode == 0 { 8 } else { 40 },
    );
    let floor = (per_batch / 8).max(1);
    let cap = per_batch.saturating_mul(4).max(floor);
    Ok(Live {
        device,
        queue,
        transport_pipe,
        transport_group,
        fold_pipe,
        stats_peak_pipe,
        stats_rel_pipe,
        fold_group,
        params,
        ledger,
        fold_params,
        out,
        stats,
        kept: vec![
            spots, floats, ints, tally, material, density, beams, sum, sq,
        ],
        base,
        next: 0,
        histories,
        per_batch,
        chunk: floor,
        floor,
        cap,
        nvox,
        scale,
        gpu,
        shape,
        origin: launch.origin_mm,
        spacing: launch.spacing_cm.map(|cm| cm * 10.0),
        inflight: VecDeque::new(),
        pending_read: None,
        finished: false,
        previews: 0,
        last_preview: Instant::now(),
        cached: None,
        cached_uncertainty: 0.0,
        score_let,
        let_buf,
    })
}

impl Live {
    fn report(&self, preview: bool) -> Report {
        let uncertainty = self.cached_uncertainty;
        Report {
            task: 0,
            generation: 0,
            phase: if preview {
                Phase::Preview
            } else {
                Phase::Compute
            },
            done: u64::from(self.next),
            total: u64::from(self.histories),
            note: history_note(self.next, self.histories, uncertainty),
        }
    }

    /// The original one-shot loop: every history, then the store fold.
    fn drain(&mut self, cancel: &Cancel) -> Result<McResult, ComputeError> {
        let _kept = self.kept.len();
        // Pace like step() does: queueing every history at once puts the
        // finish read behind the whole run, and collect_read's bound then
        // fails healthy drains on slower GPUs.
        while self.next < self.histories {
            while self.inflight.len() >= MAX_INFLIGHT {
                self.wait_fence(cancel)?;
                if cancel.is_cancelled() {
                    self.inflight.clear();
                    return Err(ComputeError::Message("cancelled".into()));
                }
            }
            self.submit_until(
                self.next
                    .saturating_add(self.chunk.max(1))
                    .min(self.histories),
                Some(cancel),
            );
            // submit_until early-returns on cancel without advancing next —
            // without this check the loop spins at 100% CPU forever.
            if cancel.is_cancelled() {
                self.inflight.clear();
                return Err(ComputeError::Message("cancelled".into()));
            }
        }
        self.finish(cancel)
    }

    fn step(&mut self, budget: Duration, cancel: &Cancel) -> Result<Slice, ComputeError> {
        let _kept = self.kept.len();
        if self.finished {
            return Ok(Slice::Ready(self.cached.clone().ok_or_else(|| {
                ComputeError::Message("monte carlo finished without a dose".into())
            })?));
        }
        let mut published = None;
        // A parked preview copy publishes whenever its map has landed. The
        // step never waits on it — sibling runs on the shared queue keep
        // submitting while the readback is in flight.
        if self.pending_read.is_some() {
            self.device.poll(wgpu::PollType::Poll).map_err(|err| {
                // Same eviction as wait_fence/collect_read — a dead device
                // must not hand out on the next shared_gpu open.
                gpu_lost(&self.gpu);
                ComputeError::Message(err.to_string())
            })?;
            match self
                .pending_read
                .as_ref()
                .expect("parked read")
                .receiver
                .try_recv()
            {
                Ok(Ok(())) => {
                    // Take before unpacking so an unpack error cannot leave a
                    // consumed ticket parked and previews disabled for the run.
                    let ticket = self.pending_read.take().expect("parked read");
                    let result = self.unpack_read(&ticket)?;
                    self.cached_uncertainty = result.uncertainty;
                    self.last_preview = Instant::now();
                    self.previews += 1;
                    published = Some(result);
                }
                Ok(Err(err)) => {
                    self.pending_read = None;
                    return Err(ComputeError::Message(err.to_string()));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(err) => {
                    self.pending_read = None;
                    return Err(ComputeError::Message(err.to_string()));
                }
            }
        }
        if cancel.is_cancelled() {
            self.inflight.clear();
            self.pending_read = None;
            return Ok(Slice::Cancelled);
        }
        if self.next >= self.histories {
            // finish()'s staged copy sits behind every queued chunk — its map
            // landing certifies all transport retired.
            self.inflight.clear();
            self.pending_read = None;
            let result = match self.finish(cancel) {
                // A cancel landing mid-read is still a cancel, not a failed
                // run — the task's cancel flag is the truth.
                Err(err) if cancel.is_cancelled() && err.to_string() == "cancelled" => {
                    return Ok(Slice::Cancelled);
                }
                other => other?,
            };
            self.finished = true;
            return Ok(Slice::Ready(result));
        }
        // Pace the queue: wait on the oldest submission only once the window
        // is full. Younger chunks stay in flight so a sibling run's poll
        // always finds GPU work queued.
        let mut blocked = Duration::ZERO;
        while self.inflight.len() >= MAX_INFLIGHT {
            blocked += self.wait_fence(cancel)?;
            if cancel.is_cancelled() {
                self.inflight.clear();
                self.pending_read = None;
                return Ok(Slice::Cancelled);
            }
        }
        self.chunk = adapt_chunk(self.chunk, self.floor, self.cap, blocked, budget);
        // The first chunk is the rough cube. Later copies are a few times a
        // second, so the view can sharpen while the next chunk is in flight.
        let fresh = self.next > 0
            && self.pending_read.is_none()
            && (self.previews == 0 || self.last_preview.elapsed() >= PREVIEW_GAP);
        if fresh {
            // The copies land before the next chunk's submission, so the map
            // resolves while the GPU is already back on transport. The
            // ticket parks — a later step collects it without waiting.
            self.pending_read = Some(self.begin_read(
                2,
                1.0 / (self.next.max(1) as f32),
                self.next / self.per_batch.max(1),
                self.next.max(1),
            )?);
        }
        if cancel.is_cancelled() {
            self.pending_read = None;
            return Ok(Slice::Cancelled);
        }
        let end = self.next.saturating_add(self.chunk).min(self.histories);
        self.submit_until(end, Some(cancel));
        if cancel.is_cancelled() {
            self.inflight.clear();
            self.pending_read = None;
            return Ok(Slice::Cancelled);
        }
        // A stale picture is not copied again. The view paints the last copy
        // while the next chunk keeps running.
        Ok(Slice::Pending(published))
    }

    fn submit_until(&mut self, end: u32, cancel: Option<&Cancel>) {
        while self.next < end {
            if cancel.is_some_and(Cancel::is_cancelled) {
                return;
            }
            let count = (self.per_batch - self.next % self.per_batch)
                .min(100_000)
                .min(end - self.next);
            write_u32(&mut self.base, 15, self.next);
            write_u32(&mut self.base, 16, count);
            self.queue.write_buffer(&self.params, 0, &self.base);
            self.queue
                .write_buffer(&self.ledger, 56, &0u32.to_le_bytes());
            let fold = self
                .next
                .saturating_add(count)
                .is_multiple_of(self.per_batch);
            if fold {
                self.queue.write_buffer(
                    &self.fold_params,
                    0,
                    &fold_bytes(self.nvox as u32, 0, self.scale, 1.0),
                );
            }
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.transport_pipe);
                pass.set_bind_group(0, &self.transport_group, &[]);
                pass.dispatch_workgroups(count.div_ceil(128), 1, 1);
                if fold {
                    pass.set_pipeline(&self.fold_pipe);
                    pass.set_bind_group(0, &self.fold_group, &[]);
                    let (x, y) = fold_groups(self.nvox);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
            self.inflight
                .push_back(self.queue.submit(Some(encoder.finish())));
            self.next += count;
        }
    }

    /// Wait on the oldest in-flight submission and pop it. Callers bound the
    /// window before calling, so the queue keeps the rest of the chunks.
    fn wait_fence(&mut self, cancel: &Cancel) -> Result<Duration, ComputeError> {
        let Some(index) = self.inflight.pop_front() else {
            return Ok(Duration::ZERO);
        };
        let started = Instant::now();
        loop {
            if cancel.is_cancelled() {
                return Ok(started.elapsed());
            }
            if started.elapsed() > Duration::from_secs(30) {
                // This run's submission retires only after every earlier
                // submission on the shared queue — sibling backlog is not a
                // wedged device, so fail the run without evicting it.
                return Err(ComputeError::Message(
                    "GPU submission did not retire".into(),
                ));
            }
            // A short wait so a new voxel size can abandon this chunk instead of
            // sitting behind the rest of the histories already queued. Only
            // this run's submission is awaited — a sibling run's queued chunk
            // must not stall this step.
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index.clone()),
                timeout: Some(Duration::from_millis(50)),
            }) {
                Ok(_) => return Ok(started.elapsed()),
                Err(wgpu::PollError::Timeout) => continue,
                Err(err) => {
                    if !matches!(err, wgpu::PollError::WrongSubmissionIndex(..)) {
                        gpu_lost(&self.gpu);
                    }
                    return Err(ComputeError::Message(err.to_string()));
                }
            }
        }
    }

    /// Fold + stats + one staged copy of everything a result needs, submitted
    /// as a single command buffer. The returned ticket's map completes once
    /// that submission retires — later submissions may still be running.
    fn begin_read(
        &self,
        mode: i32,
        gain: f32,
        batches: u32,
        histories: u32,
    ) -> Result<ReadTicket, ComputeError> {
        self.queue.write_buffer(
            &self.fold_params,
            0,
            &fold_bytes(self.nvox as u32, mode, self.scale, gain),
        );
        let mut stat_words = [0u8; 20];
        stat_words[16..20].copy_from_slice(&batches.to_le_bytes());
        self.queue.write_buffer(&self.stats, 0, &stat_words);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.nvox * 4 + 80 + if self.score_let { self.nvox * 16 } else { 0 }) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let (x, y) = fold_groups(self.nvox);
            pass.set_pipeline(&self.fold_pipe);
            pass.set_bind_group(0, &self.fold_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.stats_peak_pipe);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.stats_rel_pipe);
            pass.dispatch_workgroups(x, y, 1);
        }
        let nvox4 = (self.nvox * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.out, 0, &staging, 0, nvox4);
        encoder.copy_buffer_to_buffer(&self.stats, 0, &staging, nvox4, 16);
        encoder.copy_buffer_to_buffer(&self.ledger, 0, &staging, nvox4 + 16, 64);
        if self.score_let {
            encoder.copy_buffer_to_buffer(
                &self.let_buf,
                0,
                &staging,
                nvox4 + 80,
                (self.nvox * 16) as u64,
            );
        }
        let index = self.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        Ok(ReadTicket {
            index,
            staging,
            receiver,
            histories,
        })
    }

    fn collect_read(&self, ticket: &ReadTicket, cancel: &Cancel) -> Result<McResult, ComputeError> {
        // Only the copy submission is awaited: a transport chunk submitted
        // after it keeps the GPU busy while the dose is on its way back.
        let started = Instant::now();
        loop {
            if cancel.is_cancelled() {
                return Err(ComputeError::Message("cancelled".into()));
            }
            if started.elapsed() > Duration::from_secs(30) {
                // This run's submission retires only after every earlier
                // submission on the shared queue — sibling backlog is not a
                // wedged device, so fail the run without evicting it.
                return Err(ComputeError::Message(
                    "GPU submission did not retire".into(),
                ));
            }
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(ticket.index.clone()),
                timeout: Some(Duration::from_millis(50)),
            }) {
                Ok(_) | Err(wgpu::PollError::Timeout) => {}
                Err(err) => {
                    if !matches!(err, wgpu::PollError::WrongSubmissionIndex(..)) {
                        gpu_lost(&self.gpu);
                    }
                    return Err(ComputeError::Message(err.to_string()));
                }
            }
            match ticket.receiver.try_recv() {
                Ok(result) => {
                    result.map_err(|err| ComputeError::Message(err.to_string()))?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => continue,
                Err(err) => return Err(ComputeError::Message(err.to_string())),
            }
        }
        self.unpack_read(ticket)
    }

    /// The mapped staging buffer becomes the result: dose cube, uncertainty,
    /// ledger, LET — then unmapped. Only called after the map has landed.
    fn unpack_read(&self, ticket: &ReadTicket) -> Result<McResult, ComputeError> {
        let mapped = ticket
            .staging
            .slice(..)
            .get_mapped_range()
            .map_err(|err| ComputeError::Message(err.to_string()))?;
        let nvox4 = self.nvox * 4;
        let dose: Vec<f32> = mapped[..nvox4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let stat = &mapped[nvox4..nvox4 + 16];
        let rel_sum =
            (u64::from(u32_le(stat, 1)) | (u64::from(u32_le(stat, 2)) << 32)) as f64 / 1e6;
        let rel_count = u64::from(u32_le(stat, 3));
        let raw = &mapped[nvox4 + 16..nvox4 + 80];
        let mut ledger_mev = [0.0f32; 6];
        for i in 0..6 {
            let quanta = pack_i64(u32_le(raw, i * 2), u32_le(raw, i * 2 + 1));
            ledger_mev[i] = (quanta as f64 * QUANTUM_MEV / f64::from(ticket.histories)) as f32;
        }
        let let_d = if self.score_let {
            let region = &mapped[nvox4 + 80..nvox4 + 80 + self.nvox * 16];
            let mut out = vec![0.0f32; self.nvox];
            for (i, slot) in out.iter_mut().enumerate() {
                let words = &region[i * 16..i * 16 + 16];
                let den = pack_i64(u32_le(words, 2), u32_le(words, 3));
                if den > 0 {
                    let num = pack_i64(u32_le(words, 0), u32_le(words, 1));
                    *slot = (num as f64 / den as f64) as f32;
                }
            }
            out
        } else {
            Vec::new()
        };
        drop(mapped);
        ticket.staging.unmap();
        Ok(McResult {
            volume: Volume {
                origin: self.origin,
                shape: self.shape,
                spacing: self.spacing,
                values: dose,
            },
            uncertainty: if rel_count == 0 {
                0.0
            } else {
                (rel_sum / rel_count as f64) as f32
            },
            ledger: ledger_mev,
            let_d,
        })
    }

    fn finish(&mut self, cancel: &Cancel) -> Result<McResult, ComputeError> {
        let ticket = self.begin_read(
            1,
            1.0 / (self.histories.max(1) as f32),
            self.histories.max(1) / self.per_batch.max(1),
            self.histories.max(1),
        )?;
        let result = self.collect_read(&ticket, cancel)?;
        self.cached_uncertainty = result.uncertainty;
        self.cached = Some(result.clone());
        self.finished = true;
        Ok(result)
    }
}

fn history_note(done: u32, total: u32, uncertainty: f32) -> String {
    if total == 0 {
        return String::new();
    }
    let pct = (100.0 * f64::from(done) / f64::from(total)).round() as u32;
    if uncertainty > 0.0 {
        format!("{pct}% · ±{:.1}%", uncertainty * 100.0)
    } else {
        format!("{pct}%")
    }
}

fn batch_count(histories: u32) -> u32 {
    histories.max(10)
}

fn write_params(bytes: &mut [u8], launch: &Launch, nspots: usize) {
    write_i32(bytes, 0, launch.mode);
    write_u32(bytes, 1, launch.flags);
    write_f32(bytes, 2, launch.origin_cm[0]);
    write_f32(bytes, 3, launch.origin_cm[1]);
    write_f32(bytes, 4, launch.origin_cm[2]);
    write_f32(bytes, 5, launch.spacing_cm[0]);
    write_f32(bytes, 6, launch.spacing_cm[1]);
    write_f32(bytes, 7, launch.spacing_cm[2]);
    // The shader sees the same clamped dims the buffers were sized with —
    // a zero dim here makes every voxel_index() call return -1.
    write_i32(bytes, 8, launch.shape[0].clamp(1, i32::MAX as usize) as i32);
    write_i32(bytes, 9, launch.shape[1].clamp(1, i32::MAX as usize) as i32);
    write_i32(
        bytes,
        10,
        launch.shape[2].clamp(1, i32::MAX as usize) as i32,
    );
    write_f32(bytes, 11, launch.depth_cm);
    write_f32(bytes, 12, launch.wet_cm);
    write_i32(bytes, 13, launch.medium);
    write_i32(bytes, 14, nspots as i32);
    write_u32(bytes, 17, launch.seed);
}

fn write_i32(bytes: &mut [u8], index: usize, value: i32) {
    bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], index: usize, value: u32) {
    bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_f32(bytes: &mut [u8], index: usize, value: f32) {
    bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn fold_bytes(n: u32, mode: i32, scale: f32, gain: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    write_u32(&mut bytes, 0, n);
    write_i32(&mut bytes, 1, mode);
    write_f32(&mut bytes, 2, scale);
    write_f32(&mut bytes, 3, gain);
    bytes
}

fn fold_groups(nvox: usize) -> (u32, u32) {
    let groups = nvox.div_ceil(256).max(1);
    let x = groups.min(65_535);
    let y = groups.div_ceil(x);
    (x as u32, y as u32)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn store(read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn layout(device: &wgpu::Device, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(binding, entry)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            ..*entry
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn storage_init(device: &wgpu::Device, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("storage buffer mapped at creation");
        mapped.slice(..bytes.len()).copy_from_slice(bytes);
    }
    buffer.unmap();
    buffer
}

fn storage_zero(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    buffer.unmap();
    buffer
}

fn uniform_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

fn u32_le(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(
        bytes[index * 4..index * 4 + 4]
            .try_into()
            .expect("u32 word"),
    )
}

fn pack_i64(lo: u32, hi: u32) -> i64 {
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&lo.to_le_bytes());
    bytes[4..].copy_from_slice(&hi.to_le_bytes());
    i64::from_le_bytes(bytes)
}

async fn mc_device() -> Result<(wgpu::Device, wgpu::Queue), ComputeError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    // The only GPU waits with no deadline — a wedged driver would park the
    // worker thread forever, past cancel and join.
    let adapter = tokio::time::timeout(
        Duration::from_secs(30),
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .await
    .map_err(|_| ComputeError::Message("the GPU adapter request timed out".into()))?
    .map_err(|err| match err {
        wgpu::RequestAdapterError::NotFound { .. } => ComputeError::NoAdapter,
        other => ComputeError::Message(other.to_string()),
    })?;
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 9 {
        return Err(ComputeError::Message(
            "this GPU allows fewer than 9 storage buffers".into(),
        ));
    }
    tokio::time::timeout(
        Duration::from_secs(30),
        adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }),
    )
    .await
    .map_err(|_| ComputeError::Message("the GPU device request timed out".into()))?
    .map_err(|err| ComputeError::Message(err.to_string()))
}

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio")
        .block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("validate");
    }

    #[test]
    fn transport_and_fold_shaders_parse() {
        validate(TRANSPORT);
        validate(FOLD);
    }

    #[test]
    fn slab_deposits_in_the_bragg_region_and_closes_the_ledger() {
        let _gpu = super::hold_gpu();
        let request = SlabRequest {
            medium: "water".into(),
            x: vec![0.0],
            y: vec![0.0],
            sx: vec![2.0],
            sy: vec![2.0],
            energy: vec![70.0],
            protons: vec![1.0e6],
            histories: 2_000,
            seed: 1,
            spread_pct: 1.0,
            wet_mm: 0.0,
            depth_mm: 55.0,
            voxel_mm: 1.0,
            origin: [-12.0, -12.0, -55.0],
            shape: [24, 24, 55],
        };
        match run_mc(&McJob::Slab(request)) {
            Err(ComputeError::NoAdapter) => {
                eprintln!("skipped: no GPU adapter");
            }
            Err(err) => panic!("{err}"),
            Ok(result) => {
                let volume = &result.volume;
                let x = 12usize;
                let y = 12usize;
                let mut peak = 0.0f32;
                let mut peak_z = 0usize;
                for z in 0..volume.shape[2] {
                    let dose = volume.get(x, y, z);
                    if dose > peak {
                        peak = dose;
                        peak_z = z;
                    }
                }
                let depth = -volume.origin[2] - (peak_z as f32 + 0.5) * volume.axis(2);
                assert!(peak > 0.0, "no dose");
                assert!(
                    (30.0..50.0).contains(&depth),
                    "peak depth {depth:.1} mm is outside the 70 MeV Bragg region"
                );
                let entrance = volume.get(x, y, volume.shape[2] - 1);
                assert!(peak > entrance * 2.0);
                let incident = result.ledger[0];
                let accounted: f32 = result.ledger[1..].iter().sum();
                assert!(incident > 0.0);
                assert!(
                    (incident - accounted).abs() / incident < 0.15,
                    "ledger {incident} vs accounted {accounted} {:?}",
                    result.ledger
                );
                assert!(result.let_d.is_empty());
            }
        }
    }

    #[test]
    fn patient_box_deposits_downstream_and_closes_the_ledger() {
        let _gpu = super::hold_gpu();
        let spot = scan_kit_io::spot_record(100.0, 0.0, 0.0, 0.0, 0.0, f32::NAN).expect("spot");
        let beam = scan_kit_io::beam_record(0.0, 0.0, "HFS", [0.0, 0.0, 0.0]).expect("beam");
        let shape = [16usize, 50, 16];
        let nvox = shape[0] * shape[1] * shape[2];
        let request = PatientRequest {
            spots: spot.to_vec(),
            protons: vec![scan_kit_io::protons_per_mu(100.0).expect("protons")],
            beams: beam.to_vec(),
            material: vec![0; nvox],
            density: vec![1.0; nvox],
            spacing_mm: [2.0, 2.0, 2.0],
            origin_mm: [-16.0, -8.0, -16.0],
            shape,
            histories: 4_000,
            seed: 1,
            dose_to_water: true,
            score_let: true,
            let_water: false,
        };
        match run_mc(&McJob::Patient(request)) {
            Err(ComputeError::NoAdapter) => {
                eprintln!("skipped: no GPU adapter");
            }
            Err(err) => panic!("{err}"),
            Ok(result) => {
                let [nx, ny, nz] = result.volume.shape;
                let mut depth = vec![0.0f32; ny];
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            depth[y] += result.volume.get(x, y, z);
                        }
                    }
                }
                let (peak_y, peak) = depth
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                let from_iso =
                    result.volume.origin[1] + (peak_y as f32 + 0.5) * result.volume.axis(1);
                assert!(peak > 0.0, "no dose");
                assert!(
                    (40.0..110.0).contains(&from_iso),
                    "peak {from_iso:.1} mm from the isocenter is outside the 100 MeV Bragg region"
                );
                assert!(peak > depth[2] * 2.0);
                let incident = result.ledger[0];
                let accounted: f32 = result.ledger[1..].iter().sum();
                assert!(incident > 0.0);
                assert!(
                    (incident - accounted).abs() / incident < 0.15,
                    "ledger {incident} vs accounted {accounted} {:?}",
                    result.ledger
                );
                assert_eq!(result.let_d.len(), nx * ny * nz);
                let mut let_peak = 0.0f32;
                for value in &result.let_d {
                    if *value > let_peak {
                        let_peak = *value;
                    }
                }
                assert!(
                    (0.2..200.0).contains(&let_peak),
                    "peak LETd {let_peak} keV/um is outside the proton range"
                );
            }
        }
    }

    #[test]
    fn a_zero_axis_still_bounds_the_empty_path() {
        // The empty-path cap used to read the unclamped shape — a zero dim
        // hid a usize::MAX allocation behind it.
        let mut request = slab(0);
        request.protons = Vec::new();
        request.shape = [0, usize::MAX, usize::MAX];
        assert!(McRun::open(&McJob::Slab(request)).is_err());
        let mut request = slab(0);
        request.protons = Vec::new();
        request.shape = [0, 50_000, 50_000];
        assert!(McRun::open(&McJob::Slab(request)).is_err());
    }

    #[test]
    fn a_zero_source_sigma_beam_is_rejected() {
        let spot = scan_kit_io::spot_record(100.0, 0.0, 0.0, 0.0, 0.0, f32::NAN).expect("spot");
        let mut beam = scan_kit_io::beam_record(0.0, 0.0, "HFS", [0.0, 0.0, 0.0]).expect("beam");
        // smx is a divisor in the shader — a zero is finite but produces a
        // NaN particle that deposits into an indeterminate voxel.
        beam[13] = 0.0;
        let shape = [2usize, 2, 2];
        let nvox = 8;
        let request = PatientRequest {
            spots: spot.to_vec(),
            protons: vec![scan_kit_io::protons_per_mu(100.0).expect("protons")],
            beams: beam.to_vec(),
            material: vec![0; nvox],
            density: vec![1.0; nvox],
            spacing_mm: [2.0; 3],
            origin_mm: [-2.0; 3],
            shape,
            histories: 4_000,
            seed: 1,
            dose_to_water: true,
            score_let: false,
            let_water: false,
        };
        assert!(McRun::open(&McJob::Patient(request)).is_err());
    }

    fn slab(histories: u32) -> SlabRequest {
        SlabRequest {
            medium: "water".into(),
            x: vec![0.0],
            y: vec![0.0],
            sx: vec![2.0],
            sy: vec![2.0],
            energy: vec![70.0],
            protons: vec![1.0e6],
            histories,
            seed: 1,
            spread_pct: 1.0,
            wet_mm: 0.0,
            depth_mm: 40.0,
            voxel_mm: 2.0,
            origin: [-8.0, -8.0, -40.0],
            shape: [8, 8, 20],
        }
    }

    fn finish(run: &mut McRun) -> Result<McResult, String> {
        let cancel = Cancel::new();
        loop {
            match run.poll(Duration::from_micros(100), &cancel) {
                Poll::Pending { .. } => {}
                Poll::Ready(result) => return Ok(result),
                Poll::Cancelled => return Err("cancelled".into()),
                Poll::Failed(message) => return Err(message),
            }
        }
    }

    #[test]
    fn sliced_run_matches_one_shot() {
        let _gpu = super::hold_gpu();
        let job = McJob::Slab(slab(60_000));
        let one = match run_mc(&job) {
            Err(ComputeError::NoAdapter) => {
                eprintln!("skipped: no GPU adapter");
                return;
            }
            Err(err) => panic!("{err}"),
            Ok(result) => result,
        };
        let mut sliced = McRun::open(&job).expect("adapter");
        let cancel = Cancel::new();
        let mut saw_dose = false;
        let mut quiet = 0u32;
        let ready = loop {
            match sliced.poll(Duration::from_micros(100), &cancel) {
                Poll::Pending { preview, report } => {
                    assert!(report.done <= report.total);
                    if preview.is_none() {
                        quiet += 1;
                    }
                    if preview
                        .as_ref()
                        .is_some_and(|result| result.volume.values.iter().any(|value| *value > 0.0))
                    {
                        saw_dose = true;
                    }
                }
                Poll::Ready(result) => break result,
                Poll::Failed(message) => panic!("{message}"),
                Poll::Cancelled => panic!("cancelled"),
            }
        };
        assert!(saw_dose, "a sliced run never published a dose");
        assert!(quiet > 0, "every poll republished the cube");
        assert_eq!(ready.volume.values, one.volume.values);
        assert_eq!(ready.ledger, one.ledger);
        assert_eq!(ready.uncertainty, one.uncertainty);
        assert!(
            (0.0..0.5).contains(&one.uncertainty),
            "uncertainty {} out of range",
            one.uncertainty
        );
    }

    #[test]
    fn raised_target_resumes_a_finished_run() {
        let _gpu = super::hold_gpu();
        let mut run = match McRun::open(&McJob::Slab(slab(30_000))) {
            Err(ComputeError::NoAdapter) => {
                eprintln!("skipped: no GPU adapter");
                return;
            }
            Err(err) => panic!("{err}"),
            Ok(run) => run,
        };
        let cancel = Cancel::new();
        match run.poll(Duration::MAX, &cancel) {
            Poll::Ready(_) => {}
            Poll::Failed(message) => panic!("{message}"),
            other => panic!("30k run did not finish in one drain: {other:?}"),
        }
        assert!((run.progress() - 1.0).abs() < 1.0e-6);
        assert!(!run.extend(20_000));
        assert!(run.extend(60_000));
        assert!(
            (run.progress() - 0.5).abs() < 1.0e-6,
            "progress {}",
            run.progress()
        );
        let resumed = finish(&mut run).expect("resume");
        let one = run_mc(&McJob::Slab(slab(60_000))).expect("one shot");
        // The first target keeps its batch size, so the float fold is not bit-identical.
        close(&resumed.volume.values, &one.volume.values);
        close(&resumed.ledger, &one.ledger);
    }

    fn close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            let scale = left.abs().max(right.abs()).max(1.0e-6);
            assert!((left - right).abs() <= scale * 1.0e-4, "{left} vs {right}");
        }
    }

    #[test]
    fn cancel_after_a_preview_publishes_nothing_further() {
        let _gpu = super::hold_gpu();
        let mut run = match McRun::open(&McJob::Slab(slab(8_000))) {
            Err(ComputeError::NoAdapter) => {
                eprintln!("skipped: no GPU adapter");
                return;
            }
            Err(err) => panic!("{err}"),
            Ok(run) => run,
        };
        let cancel = Cancel::new();
        let mut saw = false;
        loop {
            match run.poll(Duration::from_micros(100), &cancel) {
                Poll::Pending { preview, .. } => {
                    if preview.is_some() {
                        saw = true;
                        cancel.cancel();
                    }
                }
                Poll::Ready(_) => panic!("finished before cancel"),
                Poll::Cancelled => break,
                Poll::Failed(message) => panic!("{message}"),
            }
        }
        assert!(saw, "cancel landed before any preview");
        assert!(matches!(
            run.poll(Duration::from_micros(100), &cancel),
            Poll::Cancelled
        ));
    }

    #[test]
    fn an_empty_result_matches_its_volume_shape() {
        // A zero axis is clamped per-dimension — values must follow the
        // clamped shape, not the raw product (was 16 voxels with len 1).
        let result = empty_result(&Launch {
            spots: Vec::new(),
            protons: 0.0,
            mode: 0,
            flags: 0,
            origin_cm: [0.0; 3],
            spacing_cm: [1.0; 3],
            shape: [4, 0, 4],
            depth_cm: 0.0,
            wet_cm: 0.0,
            medium: 0,
            histories: 0,
            seed: 0,
            material: Vec::new(),
            density: Vec::new(),
            beams: Vec::new(),
            origin_mm: [0.0; 3],
        });
        let [nx, ny, nz] = result.volume.shape;
        assert_eq!(result.volume.values.len(), nx * ny * nz);
    }

    #[test]
    fn patient_launch_rejects_bad_payloads() {
        fn request() -> PatientRequest {
            PatientRequest {
                spots: vec![0.0; 40],
                protons: vec![1.0],
                // A real record — an all-zero one is itself invalid (smx/smy
                // are shader divisors and a zero produces a NaN particle).
                beams: scan_kit_io::beam_record(0.0, 0.0, "HFS", [0.0, 0.0, 0.0])
                    .unwrap()
                    .to_vec(),
                material: vec![0; 8],
                density: vec![1.0; 8],
                spacing_mm: [1.0, 1.0, 1.0],
                origin_mm: [0.0, 0.0, 0.0],
                shape: [2, 2, 2],
                histories: 1,
                seed: 1,
                dose_to_water: false,
                score_let: false,
                let_water: false,
            }
        }
        let mut bad = request();
        bad.material[0] = 250;
        assert!(patient_launch(&bad).is_err());
        let mut bad = request();
        bad.density[0] = -1.0;
        assert!(patient_launch(&bad).is_err());
        let mut bad = request();
        bad.density[0] = f32::NAN;
        assert!(patient_launch(&bad).is_err());
        let mut bad = request();
        bad.spots[6] = 2.0;
        assert!(patient_launch(&bad).is_err());
        // A NaN beam record would poison the GPU-side rotation silently.
        let mut bad = request();
        bad.beams[0] = f32::NAN;
        assert!(patient_launch(&bad).is_err());
        // Empty beams is legal input — this case errors because spot 0's
        // beam index references a beam that is not there.
        let mut bad = request();
        bad.beams = Vec::new();
        assert_eq!(
            patient_launch(&bad).err().unwrap().to_string(),
            "spot beam index out of range"
        );
        let mut bad = request();
        bad.beams = vec![0.0; 8];
        assert!(patient_launch(&bad).is_err());
        // A non-finite field anywhere in the 40-float record fails, not just
        // the beam index — the shader consumes energy/phase-space fields too.
        let mut bad = request();
        bad.spots[20] = f32::INFINITY;
        assert!(patient_launch(&bad).is_err());
        // A range-shifter material index past the media table fails.
        let mut bad = request();
        bad.spots[32] = 1.0;
        bad.spots[34] = 99.0;
        assert!(patient_launch(&bad).is_err());
        assert!(patient_launch(&request()).is_ok());
    }
}
