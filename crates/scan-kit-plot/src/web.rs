//! The desktop plot. The webview hands over its canvas once and each packed
//! scene after that. Pointer input and hover never leave the webview.

use wasm_bindgen::prelude::*;

use crate::render::{Plot, PlotGpu, PlotInput, MAX_SIDE, MIN_SIDE};

#[wasm_bindgen]
pub struct WebPlot {
    gpu: PlotGpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    backend: String,
    plot: Option<Plot>,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Hover {
    pub x: f32,
    pub y: f32,
    /// Series index across all panels in scene order, or `undefined` over empty plot area.
    pub series: Option<u32>,
}

#[wasm_bindgen]
impl WebPlot {
    /// WebGPU when the webview has it, WebGL2 otherwise.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<WebPlot, JsValue> {
        let width = canvas.width().clamp(MIN_SIDE, MAX_SIDE);
        let height = canvas.height().clamp(MIN_SIDE, MAX_SIDE);
        let instance = wgpu::util::new_instance_with_webgpu_detection(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        })
        .await;
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(js)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(js)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(js)?;
        let caps = surface.get_capabilities(&adapter);
        let Some(first) = caps.formats.first().copied() else {
            return Err(JsValue::from_str("canvas surface has no formats"));
        };
        // Colors are already display values, so an sRGB target would lighten them twice.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .unwrap_or(first);
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);
        #[cfg(target_arch = "wasm32")]
        {
            use std::sync::Arc;
            device.on_uncaptured_error(Arc::new(|err| {
                web_sys::console::error_1(&JsValue::from_str(&format!("plot gpu: {err}")));
            }));
        }
        let backend = format!("{:?}", adapter.get_info().backend);
        let gpu = PlotGpu::new(device, queue, format).map_err(js)?;
        Ok(Self {
            gpu,
            surface,
            config,
            backend,
            plot: None,
        })
    }

    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// Replace the scene with a payload from `scan_kit_open_plot`.
    pub fn load(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let mut plot = Plot::from_payload(bytes).map_err(|err| JsValue::from_str(&err))?;
        plot.apply(self.config.width, self.config.height, &PlotInput::default());
        self.plot = Some(plot);
        Ok(())
    }

    /// Canvas backing size in device pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let limit = self
            .gpu
            .device
            .limits()
            .max_texture_dimension_2d
            .min(MAX_SIDE);
        let width = width.clamp(MIN_SIDE, limit);
        let height = height.clamp(MIN_SIDE, limit);
        if width != self.config.width || height != self.config.height {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.gpu.device, &self.config);
        }
        self.input(PlotInput::default());
    }

    pub fn pan(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
        self.input(PlotInput {
            x,
            y,
            dx,
            dy,
            drag: true,
            ..PlotInput::default()
        });
    }

    pub fn zoom(&mut self, x: f32, y: f32, wheel: f32) {
        self.input(PlotInput {
            x,
            y,
            wheel,
            ..PlotInput::default()
        });
    }

    pub fn reset(&mut self) {
        self.input(PlotInput {
            reset: true,
            ..PlotInput::default()
        });
    }

    /// Data coordinates under a canvas pixel, or `undefined` outside every plot area.
    pub fn hover(&self, x: f32, y: f32) -> Option<Hover> {
        let (hit, x, y, series) = self.plot.as_ref()?.hover(x, y);
        hit.then_some(Hover { x, y, series })
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        let Some(plot) = self.plot.as_mut() else {
            return Ok(());
        };
        if plot.is_empty() {
            return Ok(());
        }
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            // A resize leaves the first acquire outdated. Reconfigure and take
            // the next one; giving up here leaves the canvas at its default black.
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.gpu.device, &self.config);
                self.surface.get_current_texture().map_err(js)?
            }
            Err(err) => return Err(js(err)),
        };
        let encoder = {
            let view = frame.texture.create_view(&Default::default());
            plot.record(&self.gpu, &view, self.config.width, self.config.height)
                .map_err(js)?
        };
        self.gpu.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn input(&mut self, input: PlotInput) {
        let (width, height) = (self.config.width, self.config.height);
        if let Some(plot) = self.plot.as_mut() {
            plot.apply(width, height, &input);
        }
    }
}

fn js(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}
