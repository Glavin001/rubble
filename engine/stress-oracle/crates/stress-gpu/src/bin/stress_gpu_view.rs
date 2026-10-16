//! Native window demo of the end-to-end path: the stress solver's Slang compute kernels
//! (compiled to a Metal library) run explicit substeps on the GPU every frame, and Slang
//! vertex/fragment shaders draw the chunks straight from the solver's GPU state,
//! coloured by bond stress over the tensile strength, in a macOS window through wgpu.
//!
//!   stress-gpu-view [--scene slab|overhang|PATH.json] [--substeps N] [--exaggerate X]
//!   stress-gpu-view --frames 240 --screenshot f.ppm   capture frame 240, then exit
//!   STRESS_GPU_SHADERS=wgsl stress-gpu-view           the same through WGSL
//!
//! Frames are driven from the event loop, not by redraw requests, so the solver keeps
//! stepping while the window is hidden (macOS stops redrawing occluded windows); a
//! hidden window renders into an offscreen target of the same format instead.

use std::sync::Arc;
use std::time::Instant;

use bytemuck::Zeroable;
use stress_gpu::gpu::{Binding, Gpu};
use stress_gpu::shaders;
use stress_gpu::stress::GpuStress;
use stress_ref::builders::{auto_bonds, body, grid, new_scene};
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode, Support};
use stress_ref::solver::ReferenceSolver;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Uniforms of `stress_render.slang` (`Camera`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera {
    view_proj: [[f32; 4]; 4],
    light: [f32; 4],
    stress_max: f32,
    exaggerate: f32,
    pad: [f32; 2],
}

struct Options {
    scene: String,
    substeps: usize,
    exaggerate: f32,
    frames: Option<u64>,
    screenshot: Option<String>,
}

struct View {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    solver: GpuStress,
    camera: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    depth: wgpu::TextureView,
    offscreen: wgpu::Texture,
    occluded: bool,
    target: [f32; 3],
    radius: f32,
    stress_max: f32,
    exaggerate: f32,
    substeps: usize,
    frame: u64,
    sim_time: f64,
    started: Instant,
}

struct App {
    options: Options,
    view: Option<View>,
}

fn main() {
    let mut options = Options { scene: "slab".into(), substeps: 400, exaggerate: 2000.0, frames: None, screenshot: None };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--scene" => options.scene = value(),
            "--substeps" => options.substeps = value().parse().expect("--substeps N"),
            "--exaggerate" => options.exaggerate = value().parse().expect("--exaggerate X"),
            "--frames" => options.frames = Some(value().parse().expect("--frames N")),
            "--screenshot" => options.screenshot = Some(value()),
            other => panic!("unknown argument {other}"),
        }
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { options, view: None };
    event_loop.run_app(&mut app).expect("run");
}

/// A chunked slab (12 x 6 x 2 chunks of 0.2 m) cantilevered from its x = 0 edge.
fn slab() -> Scene {
    let mut s = new_scene("gpu_slab", "Chunked slab cantilever under suddenly applied gravity.");
    s.materials.insert("concrete".into(), stress_ref::showcases::static_load_concrete());
    let mut chunks = grid(Vec3::new(0.0, 0.0, 0.0), [12, 6, 2], Vec3::splat(0.2), "concrete");
    for c in chunks.iter_mut().filter(|c| c.center[0] < 0.2) {
        c.support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("slab", chunks, bonds));
    s
}

fn load_scene(name: &str) -> Scene {
    let mut scene = match name {
        "slab" => slab(),
        "overhang" => stress_ref::showcases::overhang(0.5, "overhang"),
        path => Scene::load(std::path::Path::new(path)).unwrap_or_else(|e| panic!("{path}: {e}")),
    };
    scene.sim.fracture = false;
    scene.sim.solve_mode = SolveMode::Explicit;
    scene
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.view.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("stress-gpu").with_inner_size(winit::dpi::LogicalSize::new(1100.0, 700.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.view = Some(View::new(window, &self.options));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(view) = self.view.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Named(NamedKey::Escape), state: ElementState::Pressed, .. }, .. } => event_loop.exit(),
            WindowEvent::Resized(size) => view.resize(size.width, size.height),
            WindowEvent::Occluded(hidden) => view.occluded = hidden,
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(view) = self.view.as_mut() else { return };
        let capture = self.options.frames == Some(view.frame + 1);
        let shot = if capture { self.options.screenshot.as_deref() } else { None };
        view.draw(shot);
        if capture {
            event_loop.exit();
        }
    }
}

impl View {
    fn new(window: Arc<Window>, options: &Options) -> View {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = pollster::block_on(Gpu::with_instance(&instance, Some(&surface))).expect("GPU");
        let size = window.inner_size();
        let mut config = surface.get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1)).expect("surface config");
        // Readable for screenshots of exactly what is presented.
        if surface.get_capabilities(&gpu.adapter).usages.contains(wgpu::TextureUsages::COPY_SRC) {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        surface.configure(&gpu.device, &config);

        let scene = load_scene(&options.scene);
        let reference = ReferenceSolver::new(&scene);
        let solver = GpuStress::new(&gpu, &reference).unwrap_or_else(|e| panic!("{}: {e}", options.scene));
        let stress_max = scene.materials.values().map(|m| m.tensile_strength).fold(f64::INFINITY, f64::min) as f32;
        eprintln!(
            "adapter {} ({:?}); shaders: {:?}; surface {:?} {}x{}; scene {}: {} chunks, {} bonds, dt {:.3e} s, {} substeps/frame",
            gpu.adapter_info.name,
            gpu.adapter_info.backend,
            gpu.shader_path,
            config.format,
            config.width,
            config.height,
            scene.name,
            solver.chunks.len(),
            solver.bond_count,
            solver.dt,
            options.substeps
        );
        // Frame the structure: centre and radius of the chunk centres (world).
        let centers: Vec<[f32; 3]> = solver
            .render_chunks
            .iter()
            .map(|r| std::array::from_fn(|i| r.pose_pos[i] + (0..3).map(|j| r.pose_rot[i][j] * r.center[j]).sum::<f32>()))
            .collect();
        let n = centers.len() as f32;
        let target: [f32; 3] = std::array::from_fn(|i| centers.iter().map(|c| c[i]).sum::<f32>() / n);
        let radius = centers.iter().map(|c| (0..3).map(|i| (c[i] - target[i]).powi(2)).sum::<f32>().sqrt()).fold(0.3f32, f32::max);

        let camera = gpu.uniform("camera", &Camera::zeroed());
        let bindings = [Binding::Uniform, Binding::Storage, Binding::Storage];
        let (group, layout) = gpu.layout("stress render", &bindings, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT);
        let module = gpu.module(&shaders::STRESS_RENDER);
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stress render"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("stress_vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("stress_fs"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let render_chunks = gpu.storage("render chunks", &solver.render_chunks);
        let bind = gpu.bind(&group, &[&camera, &render_chunks, &solver.state]);
        let depth = depth_view(&gpu, config.width, config.height);
        let offscreen = offscreen_texture(&gpu, &config);
        View {
            window,
            surface,
            config,
            gpu,
            solver,
            camera,
            pipeline,
            bind,
            depth,
            offscreen,
            occluded: false,
            target,
            radius,
            stress_max,
            exaggerate: options.exaggerate,
            substeps: options.substeps,
            frame: 0,
            sim_time: 0.0,
            started: Instant::now(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.depth = depth_view(&self.gpu, width, height);
        self.offscreen = offscreen_texture(&self.gpu, &self.config);
    }

    fn draw(&mut self, screenshot: Option<&str>) {
        // The window's next image, or the offscreen target while the window is hidden.
        let frame = if self.occluded {
            None
        } else {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                _ => None,
            }
        };
        let texture = frame.as_ref().map_or(&self.offscreen, |f| &f.texture);
        let aspect = self.config.width as f32 / self.config.height as f32;
        let orbit = self.started.elapsed().as_secs_f32() * 0.2;
        let cam = Camera {
            view_proj: view_projection(self.target, self.radius, orbit, aspect),
            light: [0.4, -0.6, 0.8, 0.0],
            stress_max: self.stress_max,
            exaggerate: self.exaggerate,
            pad: [0.0; 2],
        };
        self.gpu.queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(&cam));

        let view = texture.create_view(&Default::default());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.solver.record(&mut encoder, self.substeps);
        self.sim_time += self.substeps as f64 * self.solver.dt as f64;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stress"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.06, b: 0.08, a: 1.0 }), store: wgpu::StoreOp::Store },
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.draw(0..36, 0..self.solver.chunks.len() as u32);
        }
        let readback = screenshot.map(|_| copy_to_buffer(&self.gpu, &mut encoder, texture));
        self.gpu.queue.submit([encoder.finish()]);
        if let (Some(path), Some((buffer, row))) = (screenshot, readback) {
            save_ppm(&self.gpu, &buffer, row, texture.width(), texture.height(), self.config.format, path);
            let shown = if frame.is_some() { "window" } else { "offscreen (window hidden)" };
            eprintln!("frame {} (t = {:.4} s simulated, {}): wrote {path}", self.frame + 1, self.sim_time, shown);
        }
        if let Some(frame) = frame {
            self.window.pre_present_notify();
            self.gpu.queue.present(frame);
        } else {
            // Pace the hidden loop like a display would.
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        self.frame += 1;
        if self.frame % 60 == 0 {
            let secs = self.started.elapsed().as_secs_f64();
            let title = format!(
                "stress-gpu: Slang -> {:?} | {} chunks, {} bonds | {} substeps/frame, {:.0} fps | t = {:.3} s",
                self.gpu.shader_path,
                self.solver.chunks.len(),
                self.solver.bond_count,
                self.substeps,
                self.frame as f64 / secs,
                self.sim_time
            );
            self.window.set_title(&title);
            eprintln!("{title}");
        }
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

/// Column-major view-projection (depth 0..1), orbiting `target` (z up).
fn view_projection(target: [f32; 3], radius: f32, orbit: f32, aspect: f32) -> [[f32; 4]; 4] {
    let (r, h) = (radius * 2.6, radius * 1.1);
    let eye = [target[0] + r * (orbit - 1.2).cos(), target[1] + r * (orbit - 1.2).sin(), target[2] + h];
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |a: [f32; 3]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let f = norm(sub(target, eye));
    let s = norm(cross(f, [0.0, 0.0, 1.0]));
    let u = cross(s, f);
    let view = [[s[0], u[0], -f[0], 0.0], [s[1], u[1], -f[1], 0.0], [s[2], u[2], -f[2], 0.0], [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0]];
    let (near, far) = (radius * 0.05, radius * 20.0);
    let t = 1.0 / (40f32.to_radians() / 2.0).tan();
    let proj = [[t / aspect, 0.0, 0.0, 0.0], [0.0, t, 0.0, 0.0], [0.0, 0.0, far / (near - far), -1.0], [0.0, 0.0, near * far / (near - far), 0.0]];
    std::array::from_fn(|c| std::array::from_fn(|r| (0..4).map(|k| proj[k][r] * view[c][k]).sum()))
}

fn copy_to_buffer(gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) -> (wgpu::Buffer, u32) {
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot"),
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

fn save_ppm(gpu: &Gpu, buffer: &wgpu::Buffer, row: u32, w: u32, h: u32, format: wgpu::TextureFormat, path: &str) {
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map screenshot"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = buffer.get_mapped_range(..).expect("mapped screenshot");
    let bgra = matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            let p = &data[(y * row + x * 4) as usize..][..4];
            out.extend_from_slice(&if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] });
        }
    }
    std::fs::write(path, out).expect("write screenshot");
}
