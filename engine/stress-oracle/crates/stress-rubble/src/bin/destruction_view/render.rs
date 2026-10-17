//! Drawing: the chunks, impactors and ground from the stress solver's GPU buffers
//! (`destruction_render.slang`), and the text overlay (`hud_overlay.slang`).

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use glam::{Mat4, Vec3 as V3};
use stress_gpu::gpu::{Binding, Gpu};
use stress_gpu::shaders;
use winit::window::Window;

use crate::font;
use crate::sim::{HostChunk, HullVertex, Sim};

/// Uniforms of `destruction_render.slang` (`Camera`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    light: [f32; 4],
    ground: [f32; 4],
}

/// Uniforms of `hud_overlay.slang` (`Hud`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HudUniform {
    rect: [f32; 4],
    target: [f32; 4],
}

/// Orbit camera about `target` (Z up; yaw 0 looks along +y), as stress-viz's.
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub distance: f32,
    pub target: V3,
    pub fov_deg: f32,
}

impl Orbit {
    /// Framing the box `lo`-`hi` from the front left, above.
    pub fn fit(lo: V3, hi: V3) -> Orbit {
        let fov_deg = 35.0f32;
        let radius = (0.5 * (hi - lo).length()).max(0.2);
        Orbit { yaw_deg: -35.0, pitch_deg: 22.0, distance: 1.0 * radius / (0.5 * fov_deg.to_radians()).tan(), target: 0.5 * (lo + hi), fov_deg }
    }

    fn eye_dir(&self) -> V3 {
        let (y, p) = (self.yaw_deg.to_radians(), self.pitch_deg.to_radians());
        V3::new(p.cos() * y.sin(), -p.cos() * y.cos(), p.sin())
    }

    pub fn eye(&self) -> V3 {
        self.target + self.eye_dir() * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, V3::Z);
        let near = (self.distance * 0.01).max(0.01);
        Mat4::perspective_rh(self.fov_deg.to_radians(), aspect, near, self.distance * 50.0 + 100.0) * view
    }

    /// The world ray through window pixel (`x`, `y`) of a `w` x `h` window.
    pub fn ray(&self, x: f32, y: f32, w: f32, h: f32) -> (V3, V3) {
        let inv = self.view_proj(w / h).inverse();
        let (nx, ny) = (2.0 * x / w - 1.0, 1.0 - 2.0 * y / h);
        let near = inv.project_point3(V3::new(nx, ny, 0.0));
        let far = inv.project_point3(V3::new(nx, ny, 1.0));
        (self.eye(), (far - near).normalize())
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw_deg += 0.3 * dx;
        self.pitch_deg = (self.pitch_deg + 0.3 * dy).clamp(-89.0, 89.0);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        let f = -self.eye_dir();
        let right = f.cross(V3::Z).normalize_or(V3::X);
        let up = right.cross(f);
        let k = 2.0 * self.distance * (0.5 * self.fov_deg.to_radians()).tan() / height.max(1.0);
        self.target += right * (-dx * k) + up * (dy * k);
    }

    pub fn zoom(&mut self, steps: f32) {
        self.distance *= 0.9f32.powf(steps);
    }
}

/// Raw frames piped into ffmpeg (H.264 MP4).
struct Recorder {
    ffmpeg: Child,
    width: u32,
    height: u32,
}

pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    camera: wgpu::Buffer,
    group: wgpu::BindGroupLayout,
    chunk_pipeline: wgpu::RenderPipeline,
    host_chunk_pipeline: wgpu::RenderPipeline,
    hull_pipeline: wgpu::RenderPipeline,
    host_hull_pipeline: wgpu::RenderPipeline,
    /// The host chunks (`Sim::host_chunks`), their count and the signature they are of.
    host_chunks: (wgpu::Buffer, u32, Option<u64>),
    /// Hull triangles (vertex buffers and counts) of the GPU-table and host chunks, rebuilt
    /// with `host_chunks`.
    hulls: (wgpu::Buffer, u32),
    host_hulls: (wgpu::Buffer, u32),
    /// Draw chunks as their oriented boxes instead of their hulls (key B).
    pub boxes: bool,
    impactor_pipeline: wgpu::RenderPipeline,
    ground_pipeline: wgpu::RenderPipeline,
    hud_group: wgpu::BindGroupLayout,
    hud_pipeline: wgpu::RenderPipeline,
    hud_uniform: wgpu::Buffer,
    hud_pixels: Option<(wgpu::Buffer, u32, u32)>,
    depth: wgpu::TextureView,
    offscreen: wgpu::Texture,
    pub occluded: bool,
    recorder: Option<Recorder>,
}

const CLEAR: wgpu::Color = wgpu::Color { r: 0.05, g: 0.06, b: 0.08, a: 1.0 };
const SPHERE_VERTICES: u32 = 16 * 10 * 6;

impl Renderer {
    pub fn new(window: Arc<Window>, instance: &wgpu::Instance, surface: wgpu::Surface<'static>, gpu: &Gpu, record: Option<&str>) -> Result<Renderer, String> {
        let size = window.inner_size();
        let mut config = surface.get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1)).ok_or("no surface configuration")?;
        let _ = instance;
        // A linear (non-sRGB) target: the shaders' colours are display values.
        if let Some(f) = surface.get_capabilities(&gpu.adapter).formats.iter().copied().find(|f| *f == config.format.remove_srgb_suffix()) {
            config.format = f;
        }
        if surface.get_capabilities(&gpu.adapter).usages.contains(wgpu::TextureUsages::COPY_SRC) {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        surface.configure(&gpu.device, &config);

        let camera = gpu.uniform("camera", &<CameraUniform as bytemuck::Zeroable>::zeroed());
        let bindings = [Binding::Uniform, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage];
        let (group, layout) = gpu.layout("destruction render", &bindings, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT);
        let module = gpu.module(&shaders::DESTRUCTION_RENDER);
        let depth_state = |write: bool, compare: wgpu::CompareFunction| wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(write),
            depth_compare: Some(compare),
            stencil: Default::default(),
            bias: Default::default(),
        };
        let pipeline = |label: &str, vs: &str, fs: &str| {
            gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Less)),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(config.format.into())] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let chunk_pipeline = pipeline("chunks", "chunk_vs", "lit_fs");
        let host_chunk_pipeline = pipeline("host chunks", "host_chunk_vs", "lit_fs");
        let hull_pipeline = pipeline("chunk hulls", "chunk_hull_vs", "lit_fs");
        let host_hull_pipeline = pipeline("host chunk hulls", "host_hull_vs", "lit_fs");
        let impactor_pipeline = pipeline("impactors", "impactor_vs", "lit_fs");
        let ground_pipeline = pipeline("ground", "ground_vs", "ground_fs");

        let hud_uniform = gpu.uniform("hud", &HudUniform { rect: [0.0; 4], target: [1.0; 4] });
        let (hud_group, hud_layout) = gpu.layout("hud", &[Binding::Uniform, Binding::Storage], wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT);
        let hud_module = gpu.module(&shaders::HUD_OVERLAY);
        let hud_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud"),
            layout: Some(&hud_layout),
            vertex: wgpu::VertexState { module: &hud_module, entry_point: Some("hud_vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Always)),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &hud_module,
                entry_point: Some("hud_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let depth = depth_view(gpu, config.width, config.height);
        let offscreen = offscreen_texture(gpu, &config);
        let recorder = match record {
            Some(path) => Some(Recorder::start(path, config.width, config.height, config.format)?),
            None => None,
        };
        Ok(Renderer {
            window,
            surface,
            config,
            gpu: gpu.clone(),
            camera,
            group,
            chunk_pipeline,
            host_chunk_pipeline,
            host_chunks: (gpu.storage("host chunks", &[HostChunk::default()]), 0, None),
            hull_pipeline,
            host_hull_pipeline,
            hulls: (gpu.storage("chunk hulls", &[HullVertex::default()]), 0),
            host_hulls: (gpu.storage("host chunk hulls", &[HullVertex::default()]), 0),
            boxes: false,
            impactor_pipeline,
            ground_pipeline,
            hud_group,
            hud_pipeline,
            hud_uniform,
            hud_pixels: None,
            depth,
            offscreen,
            occluded: false,
            recorder,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || self.recorder.is_some() {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.depth = depth_view(&self.gpu, width, height);
        self.offscreen = offscreen_texture(&self.gpu, &self.config);
    }

    /// The HUD scale (font pixels per glyph pixel) for the window's DPI.
    fn hud_scale(&self) -> usize {
        (1.5 * self.window.scale_factor()).round().max(1.0) as usize
    }

    /// Lay out `lines` into the HUD pixel buffer (straight-alpha RGBA8, packed).
    fn upload_hud(&mut self, lines: &[(String, [u8; 3])]) -> Option<wgpu::BindGroup> {
        let scale = self.hud_scale();
        let pad = 4 * scale;
        let line_h = font::text_height(scale) + 3 * scale;
        let w = (lines.iter().map(|(l, _)| font::text_width(l, scale)).max().unwrap_or(0) + 2 * pad).min(self.config.width as usize) as u32;
        let h = (lines.len() * line_h + 2 * pad).min(self.config.height as usize) as u32;
        if w == 0 || h == 0 {
            return None;
        }
        let mut px = vec![0x9a00_0000u32 | 0x0014_100e; (w * h) as usize];
        for (i, (line, rgb)) in lines.iter().enumerate() {
            let color = 0xff00_0000 | (rgb[2] as u32) << 16 | (rgb[1] as u32) << 8 | rgb[0] as u32;
            let y0 = pad + i * line_h;
            font::for_each_pixel(line, |fx, fy| {
                for dy in 0..scale {
                    for dx in 0..scale {
                        let (x, y) = (pad + fx * scale + dx, y0 + fy * scale + dy);
                        if x < w as usize && y < h as usize {
                            px[y * w as usize + x] = color;
                        }
                    }
                }
            });
        }
        let fits = matches!(&self.hud_pixels, Some((_, cw, ch)) if *cw * *ch >= w * h);
        if !fits {
            let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hud pixels"),
                size: (w * h * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.hud_pixels = Some((buffer, w, h));
        }
        let (buffer, _, _) = self.hud_pixels.as_ref().unwrap();
        self.gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&px));
        let margin = (8 * scale) as f32;
        let u = HudUniform { rect: [margin, margin, w as f32, h as f32], target: [self.config.width as f32, self.config.height as f32, 0.0, 0.0] };
        self.gpu.queue.write_buffer(&self.hud_uniform, 0, bytemuck::bytes_of(&u));
        Some(self.gpu.bind(&self.hud_group, &[&self.hud_uniform, buffer]))
    }

    pub fn draw(&mut self, sim: &Sim, orbit: &Orbit, hud: &[(String, [u8; 3])]) -> Result<(), String> {
        let frame = if self.occluded && self.recorder.is_none() {
            None
        } else {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                _ => None,
            }
        };
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let ground = match &sim.scene.ground {
            Some(g) => [g.height as f32, (orbit.distance * 6.0).max(20.0), orbit.target.x, orbit.target.y],
            None => [0.0, 0.0, 0.0, 0.0],
        };
        let cam = CameraUniform { view_proj: orbit.view_proj(aspect).to_cols_array_2d(), light: [0.35, -0.55, 0.75, 0.0], ground };
        self.gpu.queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(&cam));
        // The solver's buffers are replaced by its rebuilds: bound afresh every frame.
        let signature = sim.host_chunk_signature();
        if self.host_chunks.2 != Some(signature) {
            let (list, host_hulls) = sim.host_chunks();
            let n = list.len() as u32;
            let data = if list.is_empty() { vec![HostChunk::default()] } else { list };
            self.host_chunks = (self.gpu.storage("host chunks", &data), n, Some(signature));
            let upload = |label: &str, v: Vec<HullVertex>| {
                let n = v.len() as u32;
                let v = if v.is_empty() { vec![HullVertex::default()] } else { v };
                (self.gpu.storage(label, &v), n)
            };
            self.hulls = upload("chunk hulls", sim.chunk_hulls());
            self.host_hulls = upload("host chunk hulls", host_hulls);
        }
        let rb = sim.world.solver.render_buffers();
        let bind = self.gpu.bind(&self.group, &[&self.camera, rb.chunks, rb.islands, rb.state, rb.index, rb.bond_dyn, rb.impactors, &self.host_chunks.0, &self.hulls.0, &self.host_hulls.0]);
        let hud_bind = self.upload_hud(hud);

        let texture = frame.as_ref().map_or(&self.offscreen, |f| &f.texture);
        let view = texture.create_view(&Default::default());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("destruction"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(CLEAR), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &bind, &[]);
            if sim.scene.ground.is_some() {
                pass.set_pipeline(&self.ground_pipeline);
                pass.draw(0..6, 0..1);
            }
            if !crate::sim::host_chunks_only() {
                if self.boxes {
                    pass.set_pipeline(&self.chunk_pipeline);
                    pass.draw(0..36, 0..rb.chunk_count);
                } else if self.hulls.1 > 0 {
                    pass.set_pipeline(&self.hull_pipeline);
                    pass.draw(0..self.hulls.1, 0..1);
                }
            }
            if self.host_chunks.1 > 0 {
                if self.boxes {
                    pass.set_pipeline(&self.host_chunk_pipeline);
                    pass.draw(0..36, 0..self.host_chunks.1);
                } else if self.host_hulls.1 > 0 {
                    pass.set_pipeline(&self.host_hull_pipeline);
                    pass.draw(0..self.host_hulls.1, 0..1);
                }
            }
            if rb.impactor_count > 0 {
                pass.set_pipeline(&self.impactor_pipeline);
                pass.draw(0..SPHERE_VERTICES, 0..rb.impactor_count);
            }
            if let Some(hb) = &hud_bind {
                pass.set_pipeline(&self.hud_pipeline);
                pass.set_bind_group(0, hb, &[]);
                pass.draw(0..6, 0..1);
            }
        }
        let readback = self.recorder.as_ref().map(|_| copy_to_buffer(&self.gpu, &mut encoder, texture));
        self.gpu.queue.submit([encoder.finish()]);
        if let (Some(rec), Some((buffer, row))) = (self.recorder.as_mut(), readback) {
            rec.write(&self.gpu, &buffer, row)?;
        }
        if let Some(frame) = frame {
            self.window.pre_present_notify();
            self.gpu.queue.present(frame);
        } else {
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        Ok(())
    }

    /// Finish the recording (closes ffmpeg's input and waits for it).
    pub fn finish(&mut self) {
        if let Some(mut rec) = self.recorder.take() {
            drop(rec.ffmpeg.stdin.take());
            let _ = rec.ffmpeg.wait();
        }
    }
}

impl Recorder {
    fn start(path: &str, width: u32, height: u32, format: wgpu::TextureFormat) -> Result<Recorder, String> {
        let pix = match format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => "bgra",
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => "rgba",
            other => return Err(format!("cannot record surface format {other:?}")),
        };
        let size = format!("{width}x{height}");
        let ffmpeg = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", &size, "-r", "60", "-i", "-"])
            .args(["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        Ok(Recorder { ffmpeg, width, height })
    }

    fn write(&mut self, gpu: &Gpu, buffer: &wgpu::Buffer, row: u32) -> Result<(), String> {
        buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map frame"));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("poll: {e:?}"))?;
        let data = buffer.get_mapped_range(..).map_err(|e| format!("mapped frame: {e:?}"))?;
        let stdin = self.ffmpeg.stdin.as_mut().ok_or("ffmpeg closed")?;
        for y in 0..self.height {
            let start = (y * row) as usize;
            stdin.write_all(&data[start..start + (self.width * 4) as usize]).map_err(|e| format!("ffmpeg: {e}"))?;
        }
        Ok(())
    }
}

fn depth_view(gpu: &Gpu, width: u32, height: u32) -> wgpu::TextureView {
    gpu.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn offscreen_texture(gpu: &Gpu, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn copy_to_buffer(gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) -> (wgpu::Buffer, u32) {
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    (buffer, row)
}
