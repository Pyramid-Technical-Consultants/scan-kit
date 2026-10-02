//! Host for the MCsquare transport shader. A run returns the finished dose.

use std::future::Future;

use scan_kit_core::{McJob, McResult, PatientRequest, SlabRequest, Volume};
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
"#;

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
    voxel_mm: f32,
}

pub fn run_mc(job: &McJob) -> Result<McResult, ComputeError> {
    let launch = match job {
        McJob::Slab(request) => slab_launch(request)?,
        McJob::Patient(request) => patient_launch(request)?,
    };
    block_on(execute(launch))
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
                sx.max(0.0) / 10.0,
                sy.max(0.0) / 10.0,
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
        voxel_mm: voxel,
    })
}

fn patient_launch(request: &PatientRequest) -> Result<Launch, ComputeError> {
    let [nx, ny, nz] = request.shape;
    let nvox = nx.saturating_mul(ny).saturating_mul(nz);
    if request.material.len() != nvox || request.density.len() != nvox {
        return Err(ComputeError::Message("material and density differ".into()));
    }
    let mut kept_spots = Vec::new();
    let mut weights = Vec::new();
    let rows = request.spots.len() / 40;
    for row in 0..rows {
        let weight = request.protons.get(row).copied().unwrap_or(0.0);
        if weight.is_finite() && weight > 0.0 {
            kept_spots.extend_from_slice(&request.spots[row * 40..(row + 1) * 40]);
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
        flags: if request.dose_to_water { 1 } else { 0 },
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
        voxel_mm: request.spacing_mm[0].max(0.25),
    })
}

async fn execute(launch: Launch) -> Result<McResult, ComputeError> {
    let [nx, ny, nz] = launch.shape;
    let nvox = nx.saturating_mul(ny).saturating_mul(nz).max(1);
    let empty = Volume {
        origin: launch.origin_mm,
        shape: [nx.max(1), ny.max(1), nz.max(1)],
        voxel: launch.voxel_mm,
        values: vec![0.0; nvox],
    };
    if launch.protons <= 0.0 || launch.spots.is_empty() {
        return Ok(McResult {
            volume: empty,
            uncertainty: 0.0,
            ledger: [0.0; 6],
        });
    }
    let (device, queue) = mc_device().await?;
    let tables = mc_tables();
    let histories = batch_count(launch.histories);
    let per_batch = (histories / 10).clamp(1, 100_000);
    let histories = histories / per_batch * per_batch;
    let volume_cm3 = f64::from(launch.spacing_cm[0])
        * f64::from(launch.spacing_cm[1])
        * f64::from(launch.spacing_cm[2]);
    let scale = (QUANTUM_MEV * launch.protons * MEV_TO_J / (volume_cm3 * 1e-3)) as f32;

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
        ],
    );
    let transport_pipe = pipeline(&device, &transport_layout, &transport);
    let fold_pipe = pipeline(&device, &fold_layout, &fold);

    let spots = storage_init(&device, &f32_bytes(&launch.spots));
    let floats = storage_init(&device, &f32_bytes(&tables.floats));
    let ints = storage_init(&device, &i32_bytes(&tables.ints));
    let tally = storage_zero(&device, (nvox * 8) as u64);
    let ledger = storage_zero(&device, 64);
    let params = uniform_buf(&device, 80);
    let material = storage_init(&device, &u32_bytes(&launch.material));
    let density = storage_init(&device, &f32_bytes(&launch.density));
    let let_tally = storage_zero(&device, 4);
    let beams = storage_init(&device, &f32_bytes(&launch.beams));
    let sum = storage_zero(&device, (nvox * 4) as u64);
    let sq = storage_zero(&device, (nvox * 4) as u64);
    let out = storage_zero(&device, (nvox * 4) as u64);
    let fold_params = uniform_buf(&device, 16);

    let transport_group = bind(
        &device,
        &transport_layout,
        &[
            &spots, &floats, &ints, &tally, &ledger, &params, &material, &density, &let_tally,
            &beams,
        ],
    );
    let fold_group = bind(
        &device,
        &fold_layout,
        &[&tally, &sum, &sq, &out, &fold_params],
    );

    let mut base = [0u8; 80];
    write_params(
        &mut base,
        &launch,
        launch.spots.len() / if launch.mode == 0 { 8 } else { 40 },
    );
    let mut next = 0u32;
    while next < histories {
        let count = (per_batch - next % per_batch)
            .min(100_000)
            .min(histories - next);
        write_u32(&mut base, 15, next);
        write_u32(&mut base, 16, count);
        queue.write_buffer(&params, 0, &base);
        queue.write_buffer(&ledger, 56, &0u32.to_le_bytes());
        dispatch(
            &device,
            &queue,
            &transport_pipe,
            &transport_group,
            count.div_ceil(64),
            1,
        );
        next += count;
        if next.is_multiple_of(per_batch) {
            queue.write_buffer(&fold_params, 0, &fold_bytes(nvox as u32, 0, scale, 1.0));
            let (x, y) = fold_groups(nvox);
            dispatch(&device, &queue, &fold_pipe, &fold_group, x, y);
        }
    }
    queue.write_buffer(
        &fold_params,
        0,
        &fold_bytes(nvox as u32, 1, scale, 1.0 / histories as f32),
    );
    let (x, y) = fold_groups(nvox);
    dispatch(&device, &queue, &fold_pipe, &fold_group, x, y);

    let dose = read_f32(&device, &queue, &out, nvox)?;
    let dose_sum = read_f32(&device, &queue, &sum, nvox)?;
    let dose_sq = read_f32(&device, &queue, &sq, nvox)?;
    let raw = read_u32(&device, &queue, &ledger, 16)?;
    let mut ledger_mev = [0.0f32; 6];
    for i in 0..6 {
        let lo = u64::from(raw[i * 2]);
        let hi = u64::from(raw[i * 2 + 1]);
        let quanta = ((hi << 32) | lo) as i64;
        ledger_mev[i] = (quanta as f64 * QUANTUM_MEV / f64::from(histories)) as f32;
    }
    Ok(McResult {
        volume: Volume {
            origin: launch.origin_mm,
            shape: [nx, ny, nz],
            voxel: launch.voxel_mm,
            values: dose,
        },
        uncertainty: uncertainty(&dose_sum, &dose_sq, histories / per_batch),
        ledger: ledger_mev,
    })
}

fn batch_count(histories: u32) -> u32 {
    histories.max(10)
}

fn uncertainty(sum: &[f32], sq: &[f32], batches: u32) -> f32 {
    let peak = sum.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 || batches == 0 {
        return 0.0;
    }
    let n = batches as f32;
    let mut total = 0.0f64;
    let mut count = 0u32;
    for (dose, square) in sum.iter().zip(sq) {
        if *dose > 0.5 * peak {
            let rel = (n * (square * n / (dose * dose) - 1.0)).max(0.0).sqrt() / n;
            total += f64::from(rel);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (total / f64::from(count)) as f32
    }
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
    write_i32(bytes, 8, launch.shape[0] as i32);
    write_i32(bytes, 9, launch.shape[1] as i32);
    write_i32(bytes, 10, launch.shape[2] as i32);
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
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some("main"),
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
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped[..bytes.len()].copy_from_slice(bytes);
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

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(x.max(1), y.max(1), 1);
    }
    queue.submit(Some(encoder.finish()));
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>, ComputeError> {
    let bytes = read_bytes(device, queue, buffer, count * 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn read_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>, ComputeError> {
    let bytes = read_bytes(device, queue, buffer, count * 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect())
}

fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<u8>, ComputeError> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
    queue.submit(Some(encoder.finish()));
    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| ComputeError::Message(err.to_string()))?;
    receiver
        .recv()
        .map_err(|err| ComputeError::Message(err.to_string()))?
        .map_err(|err| ComputeError::Message(err.to_string()))?;
    let mapped = staging.slice(..).get_mapped_range();
    let out = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

async fn mc_device() -> Result<(wgpu::Device, wgpu::Queue), ComputeError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
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
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        })
        .await
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
            Err(ComputeError::NoAdapter) => {}
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
                let depth = -volume.origin[2] - (peak_z as f32 + 0.5) * volume.voxel;
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
            }
        }
    }

    #[test]
    fn patient_box_deposits_downstream_and_closes_the_ledger() {
        let spot = scan_kit_io::spot_record(100.0, 0.0, 0.0, 0.0, 0.0, f32::NAN).expect("spot");
        let beam = scan_kit_io::beam_record(0.0, 0.0, "HFS", [0.0, 0.0, 0.0]).expect("beam");
        let shape = [16usize, 50, 16];
        let nvox = shape[0] * shape[1] * shape[2];
        let request = PatientRequest {
            spots: spot.to_vec(),
            protons: vec![scan_kit_io::protons_per_mu(100.0)],
            beams: beam.to_vec(),
            material: vec![0; nvox],
            density: vec![1.0; nvox],
            spacing_mm: [2.0, 2.0, 2.0],
            origin_mm: [-16.0, -8.0, -16.0],
            shape,
            histories: 4_000,
            seed: 1,
            dose_to_water: true,
        };
        match run_mc(&McJob::Patient(request)) {
            Err(ComputeError::NoAdapter) => {}
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
                    result.volume.origin[1] + (peak_y as f32 + 0.5) * result.volume.voxel;
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
            }
        }
    }
}
