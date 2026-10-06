//! The plot renderer. Natively it draws offscreen and reads RGBA back for MCP
//! and tests. Built for `wasm32`, the desktop webview draws the same marks into
//! its canvas with WebGPU or WebGL2.

mod dose;
mod payload;
mod render;
mod text;
#[cfg(target_arch = "wasm32")]
mod web;

pub use payload::{encode_plot, encode_plot_quality, encode_plot_reusing, plot_header, PlotHeader};
#[cfg(not(target_arch = "wasm32"))]
pub use render::render_plot;
pub use render::{compile_plot_shader, plot_shader_source, Plot, PlotFrame, PlotInput};

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    Message(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no GPU adapter"),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Default adapter and device. Kernels and the offscreen plot both use it.
pub async fn request_device() -> Result<(wgpu::Device, wgpu::Queue), GpuError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|err| match err {
            wgpu::RequestAdapterError::NotFound { .. } => GpuError::NoAdapter,
            other => GpuError::Message(other.to_string()),
        })?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|err| GpuError::Message(err.to_string()))
}

/// The process-wide offscreen plot device, or `NoAdapter` for the CPU picture.
#[cfg(not(target_arch = "wasm32"))]
fn native_gpu() -> Result<&'static render::PlotGpu, GpuError> {
    static GPU: std::sync::OnceLock<Option<render::PlotGpu>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        let (device, queue) = block_on(request_device()).ok()?;
        render::PlotGpu::new(device, queue, wgpu::TextureFormat::Rgba8Unorm).ok()
    })
    .as_ref()
    .ok_or(GpuError::NoAdapter)
}

#[cfg(not(target_arch = "wasm32"))]
fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(future)
}
