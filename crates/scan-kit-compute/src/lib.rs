//! Compute kernels and the analysis view tools. Plot drawing lives in
//! `scan-kit-plot`. These two crates are the only ones that link `wgpu`.

mod kernels;
mod mc;
mod present;
#[cfg(test)]
mod stats_time;
mod task;

pub use mc::run_mc;
pub use task::{cancel_task, poll_task, start_task};

pub use kernels::compile_scientific_shaders;
pub use present::{invoke, open_plot, run_view, tool_input_schema, tools};
pub use scan_kit_plot::{
    compile_plot_shader, plot_shader_source, render_plot, request_device, GpuError as ComputeError,
};

const SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&data)) {
        return;
    }
    data[index] = data[index] + 1.0;
}
"#;

pub fn shader_source() -> &'static str {
    SHADER
}

/// Parse the round-trip shader. This does not need a GPU adapter.
pub fn compile_shader() -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(SHADER).map_err(|err| err.to_string())?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|err| err.to_string())?;
    Ok(())
}

/// Add one to each element of a storage buffer and read it back.
pub async fn round_trip_add_one(input: &[f32]) -> Result<Vec<f32>, ComputeError> {
    // One thread per element — past 65535 workgroups the dispatch fails
    // validation (a panic), and a zero-size buffer is invalid as well.
    if input.is_empty() || input.len() > 65535 * 64 {
        return Err(ComputeError::Message(format!(
            "input length {} is out of dispatch range",
            input.len()
        )));
    }
    let (device, queue) = request_device().await?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("round-trip"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bytes = input
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("data"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .map_err(|err| ComputeError::Message(err.to_string()))?;
        mapped.copy_from_slice(&bytes);
    }
    buffer.unmap();

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((input.len() as u32).div_ceil(64).max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, bytes.len() as u64);
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let started = std::time::Instant::now();
    loop {
        match device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(50)),
        }) {
            Ok(_) | Err(wgpu::PollError::Timeout) => {}
            Err(err) => return Err(ComputeError::Message(err.to_string())),
        }
        match receiver.try_recv() {
            Ok(result) => {
                result.map_err(|err| ComputeError::Message(err.to_string()))?;
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(err) => return Err(ComputeError::Message(err.to_string())),
        }
        if started.elapsed() > std::time::Duration::from_secs(30) {
            return Err(ComputeError::Message("GPU readback timed out".into()));
        }
    }
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .map_err(|err| ComputeError::Message(err.to_string()))?;
    let output = mapped
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    drop(mapped);
    staging.unmap();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio")
            .block_on(future)
    }

    #[test]
    fn sk_req_007_storage_buffer_round_trip() {
        compile_shader().expect("shader");
        match block_on(round_trip_add_one(&[1.0, 2.0, 3.0, 4.0])) {
            Ok(values) => assert_eq!(values, vec![2.0, 3.0, 4.0, 5.0]),
            Err(ComputeError::NoAdapter) => {}
            Err(error) => panic!("{error}"),
        }
    }

    #[test]
    fn an_oversized_or_empty_dispatch_errors_before_the_device() {
        // Past 65535 workgroups the dispatch fails validation — an Err must
        // come back, not a device-side panic, and it must hold with no GPU.
        let input = vec![0.0f32; 65535 * 64 + 1];
        match block_on(round_trip_add_one(&input)) {
            Err(ComputeError::Message(err)) => assert!(err.contains("dispatch"), "{err}"),
            _ => panic!("an oversized dispatch must fail fast"),
        }
        match block_on(round_trip_add_one(&[])) {
            Err(ComputeError::Message(_)) => {}
            _ => panic!("an empty dispatch must fail fast"),
        }
    }
}
