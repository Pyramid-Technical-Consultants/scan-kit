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
    /// Panel that fills this canvas. The shell owns the grid.
    panel: Option<u32>,
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
            panel: None,
        })
    }

    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// Move time-panel cameras to the playhead window. `force` replaces a zoom.
    pub fn follow(&mut self, on: bool, lo: f32, hi: f32, force: bool) {
        if let Some(plot) = self.plot.as_mut() {
            plot.follow_time(on, lo, hi, force);
        }
    }

    /// Replace the scene with a payload from `scan_kit_open_plot`.
    pub fn load(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let mut plot = Plot::from_payload(bytes).map_err(|err| JsValue::from_str(&err))?;
        if let Some(mut previous) = self.plot.take() {
            if let Err(err) = plot.adopt_lines(&mut previous, &self.gpu.device, &self.gpu.queue) {
                self.plot = Some(previous);
                return Err(JsValue::from_str(&err));
            }
            plot.adopt_view(&previous);
            plot.keep_heatmaps(previous);
        } else if plot.needs_cached_lines() {
            return Err(JsValue::from_str(
                "plot lines are not in the previous picture",
            ));
        }
        if let Some(panel) = self.panel {
            plot.set_solo(Some(panel as usize));
        }
        plot.apply(self.config.width, self.config.height, &PlotInput::default());
        self.plot = Some(plot);
        Ok(())
    }

    /// Draw this one panel across the canvas. The shell lays the other cells out.
    pub fn solo(&mut self, panel: u32) {
        self.panel = Some(panel);
        if let Some(plot) = self.plot.as_mut() {
            plot.set_solo(Some(panel as usize));
        }
    }

    /// Crosshair voxel, or empty when the picture has no dose.
    pub fn dose_cursor(&self) -> Vec<u32> {
        self.plot
            .as_ref()
            .map(|plot| plot.dose_cursor())
            .unwrap_or_default()
    }

    /// Same voxel on every cell, so a slice drag moves the other profiles.
    pub fn set_dose_cursor(&mut self, x: u32, y: u32, z: u32) {
        if let Some(plot) = self.plot.as_mut() {
            plot.set_dose_cursor([x as usize, y as usize, z as usize]);
        }
    }

    /// Height of the dose toolbar, in framebuffer pixels. The picture starts below it.
    pub fn set_chrome(&mut self, px: f32) {
        if let Some(plot) = self.plot.as_mut() {
            plot.set_chrome(px);
        }
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

    pub fn pan(&mut self, x: f32, y: f32, dx: f32, dy: f32, buttons: u32, shift: bool) {
        self.input(PlotInput {
            x,
            y,
            dx,
            dy,
            drag: true,
            buttons: buttons as u8,
            shift,
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

    /// Panel rectangles in framebuffer pixels, for the cell and plot pickers.
    pub fn frames(&self) -> String {
        self.plot
            .as_ref()
            .map(|plot| {
                let (width, height) = (self.config.width, self.config.height);
                plot.frames_json(width, height)
            })
            .unwrap_or_else(|| "[]".into())
    }

    /// Blender numpad view for the dose turntable. `true` when the key was used.
    pub fn dose_key(&mut self, key: &str, ctrl: bool) -> bool {
        self.plot
            .as_mut()
            .is_some_and(|plot| plot.dose_key(key, ctrl))
    }

    /// Window, gain, color scale, ray mode, and sampling. Returns the level slider, or empty.
    pub fn paint(&mut self, spec: &str) -> String {
        self.plot
            .as_mut()
            .map(|plot| plot.paint(spec))
            .unwrap_or_default()
    }

    /// `rotate` turns a slice. `integral` sums through its plane or along a profile.
    pub fn dose_action(&mut self, panel: u32, action: &str) {
        if let Some(plot) = self.plot.as_mut() {
            plot.dose_action(panel as usize, action);
        }
    }

    /// Depth or lateral on the loaded cube. False when the picture has no dose yet.
    pub fn set_line(&mut self, panel: u32, label: &str) -> bool {
        self.plot
            .as_mut()
            .is_some_and(|plot| plot.set_line(panel as usize, label))
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
