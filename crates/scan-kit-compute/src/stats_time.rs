//! Timing trial for a min/max/sum compute reduction.
//!
//! The ignored test uploads the column, reduces it, and reads the result back.
//! Pipeline and buffer creation stay outside the timed loop. Run it with
//! `cargo test -p scan-kit-compute --release -- --ignored --nocapture stats_cpu_vs_gpu`.
//! The crossover is recorded on `scan_kit_core::stats`. The GPU is slower through
//! 1e5 and the playhead window is about 1e3, so this trial is not a shipped kernel.

use scan_kit_core::reduce_finite;

const INPUT_SHADER: &str = r#"
struct Partial {
    min_v: f32,
    max_v: f32,
    sum_v: f32,
    count: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<Partial>;

var<workgroup> smin: array<f32, 256>;
var<workgroup> smax: array<f32, 256>;
var<workgroup> ssum: array<f32, 256>;
var<workgroup> scount: array<u32, 256>;

fn fold(lid: u32) {
    var offset = 128u;
    loop {
        if (lid < offset) {
            let other = lid + offset;
            let c0 = scount[lid];
            let c1 = scount[other];
            if (c1 > 0u) {
                if (c0 == 0u) {
                    smin[lid] = smin[other];
                    smax[lid] = smax[other];
                } else {
                    smin[lid] = min(smin[lid], smin[other]);
                    smax[lid] = max(smax[lid], smax[other]);
                }
                ssum[lid] = ssum[lid] + ssum[other];
                scount[lid] = c0 + c1;
            }
        }
        workgroupBarrier();
        offset = offset / 2u;
        if (offset == 0u) {
            break;
        }
    }
}

@compute @workgroup_size(256)
fn reduce_input(
    @builtin(workgroup_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let n = arrayLength(&input);
    let stride = groups.x * 256u;
    var lo = 0.0;
    var hi = 0.0;
    var sum = 0.0;
    var count = 0u;
    for (var i = gid.x * 256u + lid.x; i < n; i = i + stride) {
        let v = input[i];
        if (count == 0u) {
            lo = v;
            hi = v;
        } else {
            lo = min(lo, v);
            hi = max(hi, v);
        }
        sum = sum + v;
        count = count + 1u;
    }
    smin[lid.x] = lo;
    smax[lid.x] = hi;
    ssum[lid.x] = sum;
    scount[lid.x] = count;
    workgroupBarrier();
    fold(lid.x);
    if (lid.x == 0u) {
        partials[gid.x] = Partial(smin[0], smax[0], ssum[0], scount[0]);
    }
}
"#;

const PARTIAL_SHADER: &str = r#"
struct Partial {
    min_v: f32,
    max_v: f32,
    sum_v: f32,
    count: u32,
}

@group(0) @binding(0) var<storage, read> parts: array<Partial>;
@group(0) @binding(1) var<storage, read_write> outp: array<Partial>;

var<workgroup> smin: array<f32, 256>;
var<workgroup> smax: array<f32, 256>;
var<workgroup> ssum: array<f32, 256>;
var<workgroup> scount: array<u32, 256>;

fn fold(lid: u32) {
    var offset = 128u;
    loop {
        if (lid < offset) {
            let other = lid + offset;
            let c0 = scount[lid];
            let c1 = scount[other];
            if (c1 > 0u) {
                if (c0 == 0u) {
                    smin[lid] = smin[other];
                    smax[lid] = smax[other];
                } else {
                    smin[lid] = min(smin[lid], smin[other]);
                    smax[lid] = max(smax[lid], smax[other]);
                }
                ssum[lid] = ssum[lid] + ssum[other];
                scount[lid] = c0 + c1;
            }
        }
        workgroupBarrier();
        offset = offset / 2u;
        if (offset == 0u) {
            break;
        }
    }
}

@compute @workgroup_size(256)
fn reduce_partials(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = arrayLength(&parts);
    var lo = 0.0;
    var hi = 0.0;
    var sum = 0.0;
    var count = 0u;
    if (lid.x < n) {
        let part = parts[lid.x];
        lo = part.min_v;
        hi = part.max_v;
        sum = part.sum_v;
        count = part.count;
    }
    smin[lid.x] = lo;
    smax[lid.x] = hi;
    ssum[lid.x] = sum;
    scount[lid.x] = count;
    workgroupBarrier();
    fold(lid.x);
    if (lid.x == 0u) {
        outp[0] = Partial(smin[0], smax[0], ssum[0], scount[0]);
    }
}
"#;

fn parse_shader(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|err| err.to_string())?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|err| err.to_string())?;
    Ok(())
}

struct Ends {
    min: f32,
    max: f32,
    sum: f32,
    count: u32,
}

fn group_count(n: usize) -> u32 {
    u32::try_from(n.div_ceil(256).clamp(1, 256)).unwrap_or(256)
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio")
        .block_on(future)
}

struct Reducer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input_pipe: wgpu::ComputePipeline,
    partial_pipe: wgpu::ComputePipeline,
    input_layout: wgpu::BindGroupLayout,
    partial_layout: wgpu::BindGroupLayout,
}

struct Job {
    input: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    input_group: wgpu::BindGroup,
    partial_group: wgpu::BindGroup,
    groups: u32,
}

impl Reducer {
    fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let input_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce-input"),
            source: wgpu::ShaderSource::Wgsl(INPUT_SHADER.into()),
        });
        let partial_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce-partial"),
            source: wgpu::ShaderSource::Wgsl(PARTIAL_SHADER.into()),
        });
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let input_layout = layout(
            &device,
            "reduce-input",
            &[bind(0, storage(true)), bind(1, storage(false))],
        );
        let partial_layout = layout(
            &device,
            "reduce-partial",
            &[bind(0, storage(true)), bind(1, storage(false))],
        );
        Self {
            input_pipe: pipeline(&device, &input_shader, "reduce_input", &input_layout),
            partial_pipe: pipeline(&device, &partial_shader, "reduce_partials", &partial_layout),
            input_layout,
            partial_layout,
            device,
            queue,
        }
    }

    fn job(&self, bytes: u64, groups: u32) -> Job {
        let input = self.buffer(
            "samples",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let partials = self.buffer(
            "partials",
            u64::from(groups) * 16,
            wgpu::BufferUsages::STORAGE,
        );
        let out = self.buffer(
            "out",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = self.buffer(
            "readback",
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let input_group = self.group(&self.input_layout, &input, &partials);
        let partial_group = self.group(&self.partial_layout, &partials, &out);
        Job {
            input,
            out,
            staging,
            input_group,
            partial_group,
            groups,
        }
    }

    fn execute(&self, job: &Job, bytes: &[u8]) -> Result<Ends, String> {
        self.queue.write_buffer(&job.input, 0, bytes);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.input_pipe);
            pass.set_bind_group(0, &job.input_group, &[]);
            pass.dispatch_workgroups(job.groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.partial_pipe);
            pass.set_bind_group(0, &job.partial_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&job.out, 0, &job.staging, 0, 16);
        self.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        job.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| err.to_string())?;
        receiver
            .recv()
            .map_err(|err| err.to_string())?
            .map_err(|err| err.to_string())?;
        let mapped = job.staging.slice(..).get_mapped_range();
        let ends = Ends {
            min: f32::from_le_bytes(mapped[0..4].try_into().unwrap()),
            max: f32::from_le_bytes(mapped[4..8].try_into().unwrap()),
            sum: f32::from_le_bytes(mapped[8..12].try_into().unwrap()),
            count: u32::from_le_bytes(mapped[12..16].try_into().unwrap()),
        };
        drop(mapped);
        job.staging.unmap();
        Ok(ends)
    }

    fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    fn group(
        &self,
        layout: &wgpu::BindGroupLayout,
        first: &wgpu::Buffer,
        second: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: first.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: second.as_entire_binding(),
                },
            ],
        })
    }
}

fn bind(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &str,
    group: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[group],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn best_of(times: usize, mut body: impl FnMut()) -> std::time::Duration {
    let mut best = std::time::Duration::MAX;
    for _ in 0..times {
        let start = std::time::Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    best
}

#[test]
fn reduce_shader_parses() {
    parse_shader(INPUT_SHADER).expect("input shader");
    parse_shader(PARTIAL_SHADER).expect("partial shader");
}

#[test]
fn a_short_column_matches_the_cpu_reduction() {
    assert_eq!(group_count(1), 1);
    assert_eq!(group_count(256), 1);
    assert_eq!(group_count(257), 2);
    assert_eq!(group_count(256 * 400), 256);
    let elapsed = best_of(1, || {});
    assert!(elapsed < std::time::Duration::from_secs(2));
    let gpu = match block_on(crate::request_device()) {
        Ok((device, queue)) => Reducer::new(device, queue),
        Err(crate::ComputeError::NoAdapter) => return,
        Err(error) => panic!("{error}"),
    };
    let data: Vec<f32> = (0..300).map(|index| index as f32 - 40.0).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    let job = gpu.job(bytes.len() as u64, group_count(data.len()));
    let check = gpu.execute(&job, &bytes).expect("reduce");
    let want = reduce_finite(&data);
    assert_eq!(check.count, want.count);
    assert_eq!(check.min, want.min);
    assert_eq!(check.max, want.max);
    let scale = want.sum.abs().max(1.0);
    assert!((check.sum - want.sum).abs() / scale < 1.0e-4);
}

#[test]
#[ignore = "release timing; cargo test -p scan-kit-compute --release -- --ignored --nocapture stats_cpu_vs_gpu"]
fn stats_cpu_vs_gpu_reduction() {
    parse_shader(INPUT_SHADER).expect("input shader");
    parse_shader(PARTIAL_SHADER).expect("partial shader");
    let sizes = [1_000usize, 100_000, 1_000_000, 10_000_000];
    let gpu = match block_on(crate::request_device()) {
        Ok((device, queue)) => Some(Reducer::new(device, queue)),
        Err(crate::ComputeError::NoAdapter) => None,
        Err(error) => panic!("{error}"),
    };
    for n in sizes {
        let data: Vec<f32> = (0..n).map(|index| (index % 997) as f32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
        let groups = group_count(n);
        let cpu = best_of(5, || {
            std::hint::black_box(reduce_finite(std::hint::black_box(&data)));
        });
        let Some(gpu) = gpu.as_ref() else {
            eprintln!("n={n} cpu_us={} gpu=skipped", cpu.as_micros());
            continue;
        };
        let job = gpu.job(bytes.len() as u64, groups);
        let check = gpu.execute(&job, &bytes).expect("reduce");
        let want = reduce_finite(&data);
        assert_eq!(check.count, want.count, "n={n}");
        assert_eq!(check.min, want.min, "n={n}");
        assert_eq!(check.max, want.max, "n={n}");
        let scale = want.sum.abs().max(1.0);
        let sum_ratio = (check.sum - want.sum).abs() / scale;
        assert!(
            sum_ratio < 1.0e-2,
            "n={n} gpu sum {} cpu {}",
            check.sum,
            want.sum
        );
        gpu.execute(&job, &bytes).expect("warmup");
        let gpu_time = best_of(5, || {
            std::hint::black_box(gpu.execute(&job, &bytes).expect("timed reduce"));
        });
        eprintln!(
            "n={n} cpu_us={} gpu_us={} gpu_over_cpu={:.2}",
            cpu.as_micros(),
            gpu_time.as_micros(),
            gpu_time.as_secs_f64() / cpu.as_secs_f64()
        );
    }
}
