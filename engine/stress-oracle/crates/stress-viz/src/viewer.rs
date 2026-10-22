//! `stress-viz view`: an interactive window on a live reference-solver run.
//!
//! The simulation runs on its own thread (its own rayon pool) as fast as it can, or
//! capped at real time, and records every frame. The window shows the live head or any
//! recorded frame (scrub, step, replay at any speed), with a free orbit camera, and
//! fires projectiles where you click. Firing while looking at a past frame branches the
//! run there: the world is restored from the nearest checkpoint, re-simulated
//! (deterministically, with the shots logged so far) to that frame, and the shot is
//! added. The solver mode can be switched (restarting the run) to trade fidelity for
//! speed.
//!
//! Projectiles are steel spheres added to `World::impactors`. The solver sizes its
//! contact substep once, at construction, from the scene's bodies; a projectile's own
//! limit is computed here with the same law (contact stiffness of the sphere against
//! the stiffest material, `2 / omega * courant_safety`) and applied through
//! `sim.max_substep`, which the world reads every frame.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use stress_ref::material::Material;
use stress_ref::math::{Mat3, Pose, Quat, Vec3};
use stress_ref::scene::{ImpactorShape, Scene};
use stress_ref::world::{Impactor, World};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera::{Aabb, Camera, CameraSpec};
use crate::canvas::Canvas;
use crate::colormap::{rgb8, Rgb};
use crate::font;
use crate::gpu::{GpuContext, GpuPanel};
use crate::plot::fmt_time;
use crate::record::{try_new_world, Engine, Frame, Recorder};
use crate::scene3d::{DrawOptions, PanelScene};
use crate::views::{fmt_number, ScaleOverrides, Scales, ViewMode};

/// What the window was asked to show.
pub struct ViewOptions {
    pub scene: Scene,
    pub views: Vec<ViewMode>,
    pub camera: CameraSpec,
    /// Drawing options; `opts.ss` is replaced by the supersampling in use.
    pub opts: DrawOptions,
    /// `--ss`; default 2, or 1 on a high-density display.
    pub ss: Option<usize>,
    pub scale_overrides: ScaleOverrides,
    /// Threads of the simulation's rayon pool.
    pub sim_threads: usize,
    /// Rigid-body engine at start (E switches).
    pub engine: Engine,
    /// Index in [`MODES`] at start (M switches).
    pub mode: usize,
}

/// The engine's label for the HUD.
fn engine_label(e: Engine) -> &'static str {
    match e {
        Engine::Standalone => "standalone (the oracle's rigid bodies and contact)",
        Engine::Rubble => "rubble AVBD GPU (stress solve owns impact islands)",
        Engine::Learned => "learned impact-outcome model (closed loop)",
        Engine::Ceiling => "oracle through the model's handoff (ceiling)",
        Engine::ALearned => "harness A: the oracle, routed islands from the learned model",
        Engine::Qs => "CPU blast_parity (quasi-static stress)",
    }
}

/// Whether solver mode `m` runs with `engine` (the rubble coupling needs an explicit
/// or adaptive solve).
fn mode_allowed(m: usize, engine: Engine) -> bool {
    engine == Engine::Standalone || !MODES[m].2.iter().any(|(p, v)| *p == "sim.solve_mode" && v.contains("implicit"))
}

/// Solver modes the window can restart in (M cycles, `--solver-mode` picks the first):
/// command-line name, HUD label and scene overrides.
pub const MODES: [(&str, &str, &[(&str, &str)]); 5] = [
    ("explicit", "true stiffness, explicit", &[]),
    ("soft", "stiffness x0.01 (fast, softer)", &[("sim.stiffness_scale", "0.01")]),
    ("implicit", "implicit stress step", &[("sim.solve_mode", "\"implicit\"")]),
    ("mass-scaling", "mass scaling to 1e-5 s", &[("sim.mass_scaling_dt", "1e-5")]),
    // Quiet clusters are solved quasi-statically and wake when their summed load
    // changes by 2% of their weight: impacts wake them, a small load at one spot of a
    // large structure may not.
    ("adaptive", "adaptive: quasi-static when quiet (small local loads may not wake)", &[("sim.solve_mode", "\"adaptive\"")]),
];

/// The index in [`MODES`] of a command-line mode name.
pub fn mode_index(name: &str) -> Option<usize> {
    MODES.iter().position(|m| m.0 == name)
}

/// Most World checkpoints kept for branching (thinned by doubling the stride).
const MAX_CHECKPOINTS: usize = 96;
/// Projectiles kept in the world at once (the oldest leaves first).
const MAX_SHOTS: usize = 32;

// ------------------------------------------------------------------ simulation thread

/// A projectile: a steel sphere fired from `position` with `velocity`.
#[derive(Clone, Copy, Debug)]
struct Shot {
    position: Vec3,
    velocity: Vec3,
    radius: f64,
    density: f64,
}

enum Command {
    Fire(Shot),
    /// Truncate the run after frame k (re-simulating to it from a checkpoint).
    Branch(usize),
    Pause(bool),
    RealTime(bool),
    Restart(usize, Engine),
}

/// State shared by the simulation thread and the window.
#[derive(Default)]
struct Timeline {
    frames: Vec<Arc<Frame>>,
    /// Bumped whenever recorded frames are dropped (restart, branch).
    generation: u64,
    /// Simulated seconds per wall second over the last second (0 when paused).
    rate: f64,
    paused: bool,
    realtime: bool,
    mode: usize,
    engine: Engine,
    shots: usize,
    substep: f64,
    bounds: Option<Aabb>,
    min_half_extent: f64,
    typical_chunk: f64,
    /// What the simulation is busy with besides advancing (re-simulating to a branch).
    busy: Option<String>,
    error: Option<String>,
}

struct Sim {
    base: Scene,
    mode: usize,
    engine: Engine,
    world: World,
    recorder: Recorder,
    /// Recorded frame index of each checkpoint, and the world after that frame.
    checkpoints: Vec<(usize, World)>,
    stride: usize,
    /// Shots fired after each recorded frame index, in order (replayed on branches).
    shots: Vec<(usize, Shot)>,
    fired: usize,
    /// Impactors of the scene itself (projectiles come after them).
    scene_impactors: usize,
    base_max_substep: Option<f64>,
    shot_material: Material,
    timeline: Arc<Mutex<Timeline>>,
}

/// The scene with mode `m`'s overrides.
fn scene_in_mode(base: &Scene, m: usize) -> Result<Scene, String> {
    let mut s = base.clone();
    for (path, value) in MODES[m].2 {
        s = s.with_override(path, value)?;
    }
    Ok(s)
}

/// Projectile material: the scene's impactor material, else its stiffest material.
fn shot_material(world: &World) -> Material {
    if let Some(i) = world.impactors.first() {
        return i.material.clone();
    }
    world.scene.materials.values().max_by(|a, b| a.youngs_modulus.total_cmp(&b.youngs_modulus)).cloned().expect("a scene has materials")
}

/// Stable contact substep of a projectile (the solver's contact law for a sphere
/// pressed against the stiffest material: half-thickness `r`, face `4 r^2`).
fn shot_dt(world: &World, imp: &Impactor) -> f64 {
    let ImpactorShape::Sphere { radius: r } = imp.shape else { return f64::INFINITY };
    let scale = world.scene.sim.stiffness_scale;
    let e = |m: &Material| m.youngs_modulus * scale / (1.0 - m.poisson_ratio * m.poisson_ratio);
    let stiffest = world.scene.materials.values().map(e).fold(e(&imp.material), f64::max);
    let k = 4.0 * r * r / (r / e(&imp.material) + r / stiffest);
    2.0 / (k / imp.mass).sqrt() * world.scene.sim.courant_safety
}

impl Sim {
    fn new(base: Scene, mode: usize, engine: Engine, timeline: Arc<Mutex<Timeline>>) -> Result<Sim, String> {
        let scene = scene_in_mode(&base, mode)?;
        let world = try_new_world(&scene, engine)?;
        let (recorder, first) = Recorder::new(&world);
        let sim = Sim {
            base,
            mode,
            engine,
            scene_impactors: world.impactors.len(),
            base_max_substep: world.scene.sim.max_substep,
            shot_material: shot_material(&world),
            // A world snapshot does not carry an external engine: with one, branches
            // re-simulate from the start instead.
            checkpoints: if engine == Engine::Standalone { vec![(0, world.clone())] } else { Vec::new() },
            stride: 1,
            shots: Vec::new(),
            fired: 0,
            world,
            recorder,
            timeline,
        };
        let mut t = sim.timeline.lock().unwrap();
        t.frames = vec![Arc::new(first)];
        t.generation += 1;
        t.mode = mode;
        t.engine = engine;
        t.shots = 0;
        t.bounds = Some(sim.recorder.bounds);
        t.min_half_extent = sim.recorder.min_half_extent;
        t.typical_chunk = sim.recorder.typical_chunk;
        t.substep = sim.world.substep_dt();
        t.error = None;
        drop(t);
        Ok(sim)
    }

    fn head(&self) -> usize {
        self.timeline.lock().unwrap().frames.len() - 1
    }

    fn spawn(&mut self, shot: &Shot) {
        // The coupled world keys engine bodies by impactor index: removing one would
        // shift the others onto stale engine bodies, so with an engine the projectiles
        // stay and their number is capped.
        if self.engine != Engine::Standalone && self.world.impactors.len() - self.scene_impactors >= MAX_SHOTS {
            self.timeline.lock().unwrap().error = Some(format!("{MAX_SHOTS} projectiles in the rubble world: R restarts"));
            return;
        }
        let r = shot.radius;
        let mass = shot.density * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        self.fired += 1;
        self.world.impactors.push(Impactor {
            name: format!("shot {}", self.fired),
            shape: ImpactorShape::Sphere { radius: r },
            mass,
            inertia: Mat3::IDENTITY * (0.4 * mass * r * r),
            pose: Pose::new(shot.position, Quat::IDENTITY),
            velocity: shot.velocity,
            angular_velocity: Vec3::ZERO,
            material: self.shot_material.clone(),
            crush: None,
            crush_used: 0.0,
            crush_depth: 0.0,
            driven: false,
        });
        if self.engine == Engine::Standalone && self.world.impactors.len() - self.scene_impactors > MAX_SHOTS {
            self.world.impactors.remove(self.scene_impactors);
        }
        self.update_substep_cap();
    }

    /// `sim.max_substep`: the scene's own cap and every projectile's contact limit.
    fn update_substep_cap(&mut self) {
        let mut cap = self.base_max_substep.unwrap_or(f64::INFINITY);
        for imp in &self.world.impactors[self.scene_impactors..] {
            cap = cap.min(shot_dt(&self.world, imp));
        }
        self.world.scene.sim.max_substep = cap.is_finite().then_some(cap);
    }

    /// Drops projectiles that left the scene (fell past the ground or flew far away).
    fn drop_lost_shots(&mut self) {
        if self.engine != Engine::Standalone {
            return; // See `spawn`.
        }
        let Some(b) = self.timeline.lock().unwrap().bounds else { return };
        let (center, reach) = (b.center(), 10.0 * b.size().norm().max(1.0));
        let floor = self.world.scene.ground.as_ref().map_or(b.lo.z, |g| g.height) - 0.5 * b.size().norm().max(1.0);
        let before = self.world.impactors.len();
        let keep = self.scene_impactors;
        let mut k = 0;
        self.world.impactors.retain(|i| {
            k += 1;
            k <= keep || (i.pose.position.z > floor && (i.pose.position - center).norm() < reach)
        });
        if self.world.impactors.len() != before {
            self.update_substep_cap();
        }
    }

    /// Advances one frame, records it and maybe a checkpoint; returns the new index.
    fn step(&mut self) -> usize {
        self.world.step_frame();
        self.drop_lost_shots();
        let frame = Arc::new(self.recorder.capture(&self.world));
        let index = {
            let mut t = self.timeline.lock().unwrap();
            t.frames.push(frame);
            t.substep = self.world.substep_dt();
            t.shots = self.world.impactors.len() - self.scene_impactors;
            t.frames.len() - 1
        };
        if self.engine == Engine::Standalone && index.is_multiple_of(self.stride) {
            self.checkpoints.push((index, self.world.clone()));
            if self.checkpoints.len() > MAX_CHECKPOINTS {
                self.stride *= 2;
                let stride = self.stride;
                self.checkpoints.retain(|(i, _)| i.is_multiple_of(stride));
            }
        }
        index
    }

    /// Fires the shots logged after frame `index`.
    fn replay_shots(&mut self, index: usize) {
        let due: Vec<Shot> = self.shots.iter().filter(|(i, _)| *i == index).map(|(_, s)| *s).collect();
        for s in due {
            self.spawn(&s);
        }
    }

    /// Makes frame `k` the head: restore the nearest checkpoint at or before it and
    /// re-simulate (with the logged shots) up to it.
    fn branch(&mut self, k: usize) {
        if k >= self.head() {
            return;
        }
        let (c, world) = match self.checkpoints.iter().rev().find(|(i, _)| *i <= k) {
            Some((i, w)) => (*i, w.clone()),
            // No snapshots with an external engine: rebuild the world from the start.
            None => match scene_in_mode(&self.base, self.mode).and_then(|s| try_new_world(&s, self.engine)) {
                Ok(w) => (0, w),
                Err(e) => {
                    self.timeline.lock().unwrap().error = Some(e);
                    return;
                }
            },
        };
        self.world = world;
        self.checkpoints.retain(|(i, _)| *i <= c);
        self.shots.retain(|(i, _)| *i < k);
        {
            let mut t = self.timeline.lock().unwrap();
            t.frames.truncate(c + 1);
            t.generation += 1;
            t.busy = (k > c).then(|| format!("re-simulating frames {c}..{k} to branch"));
        }
        self.update_substep_cap();
        let mut i = c;
        while i < k {
            self.replay_shots(i);
            i = self.step();
        }
        self.timeline.lock().unwrap().busy = None;
    }

    /// Runs until the window hangs up.
    fn run(mut self, rx: Receiver<Command>) {
        let mut paused = false;
        let mut realtime = true;
        // Real-time pacing: simulated time and wall clock at the last (re)start.
        let mut anchor = (Instant::now(), self.world.solver.time);
        let mut rate_window: std::collections::VecDeque<(Instant, f64)> = Default::default();
        loop {
            // Commands; block while paused.
            loop {
                let cmd = if paused {
                    match rx.recv() {
                        Ok(c) => c,
                        Err(_) => return,
                    }
                } else {
                    match rx.try_recv() {
                        Ok(c) => c,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                };
                match cmd {
                    Command::Fire(shot) => {
                        let head = self.head();
                        self.shots.push((head, shot));
                        self.spawn(&shot);
                    }
                    Command::Branch(k) => self.branch(k),
                    Command::Pause(p) => paused = p,
                    Command::RealTime(r) => realtime = r,
                    Command::Restart(mode, engine) => match Sim::new(self.base.clone(), mode, engine, self.timeline.clone()) {
                        Ok(s) => self = s,
                        Err(e) => self.timeline.lock().unwrap().error = Some(e),
                    },
                }
                anchor = (Instant::now(), self.world.solver.time);
                rate_window.clear();
                let mut t = self.timeline.lock().unwrap();
                t.paused = paused;
                t.realtime = realtime;
                if paused {
                    t.rate = 0.0;
                }
            }
            if realtime {
                let ahead = (self.world.solver.time - anchor.1) - anchor.0.elapsed().as_secs_f64();
                if ahead > 0.0 {
                    std::thread::sleep(Duration::from_secs_f64(ahead.min(0.02)));
                    continue;
                }
            }
            self.step();
            let now = Instant::now();
            rate_window.push_back((now, self.world.solver.time));
            while rate_window.len() > 2 && now.duration_since(rate_window[0].0).as_secs_f64() > 1.0 {
                rate_window.pop_front();
            }
            if let (Some(a), Some(b)) = (rate_window.front(), rate_window.back()) {
                let wall = b.0.duration_since(a.0).as_secs_f64();
                if wall > 0.0 {
                    self.timeline.lock().unwrap().rate = (b.1 - a.1) / wall;
                }
            }
        }
    }
}

// ------------------------------------------------------------------ window

const PRESENT_SHADER: &str = r#"
struct Present {
    size: vec2f,
    top_h: f32,
    bar_h: f32,
    srgb: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> p: Present;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var top_tex: texture_2d<f32>;
@group(0) @binding(3) var bar_tex: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) k: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((k << 1u) & 2u), f32(k & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@fragment
fn fs_present(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let i = vec2i(q.xy);
    var c = textureLoad(scene_tex, i, 0).rgb;
    if q.y < p.top_h {
        let h = textureLoad(top_tex, i, 0);
        c = mix(c, h.rgb, h.a);
    }
    let yb = q.y - (p.size.y - p.bar_h);
    if yb >= 0.0 {
        let h = textureLoad(bar_tex, vec2i(i.x, i32(yb)), 0);
        c = mix(c, h.rgb, h.a);
    }
    if p.srgb != 0u {
        c = to_linear(c);
    }
    return vec4f(c, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PresentUniform {
    size: [f32; 2],
    top_h: f32,
    bar_h: f32,
    srgb: u32,
    pad: [u32; 3],
}

/// HUD line pitch in glyph pixels.
const LINE: usize = font::GLYPH_H + 5;

/// HUD colours: the key colour is transparent, the box colour translucent.
const KEY: Rgb = [1.0, 0.0, 1.0];
const BOX: Rgb = rgb8(0x16, 0x19, 0x20);
const BOX_ALPHA: f32 = 0.78;
const INK: Rgb = rgb8(0xf2, 0xf4, 0xf8);
const MUTED: Rgb = rgb8(0xa8, 0xb0, 0xbe);
const ACCENT: Rgb = rgb8(0xff, 0xb0, 0x3a);
const LIVE: Rgb = rgb8(0x4c, 0xd9, 0x7b);
const TRACK: Rgb = rgb8(0x3a, 0x40, 0x4c);

/// RGBA8 of a HUD canvas (key colour transparent, box colour translucent).
fn hud_rgba(c: &Canvas) -> Vec<u8> {
    let key = [255u8, 0, 255];
    let bx = [BOX[0], BOX[1], BOX[2]].map(|v| (v * 255.0 + 0.5) as u8);
    let mut out = Vec::with_capacity(c.width * c.height * 4);
    for px in c.pixels.chunks_exact(3) {
        let a = if px == key {
            0
        } else if px == bx {
            (BOX_ALPHA * 255.0) as u8
        } else {
            255
        };
        out.extend_from_slice(&[px[0], px[1], px[2], a]);
    }
    out
}

struct Gfx {
    window: Arc<Window>,
    ctx: Arc<GpuContext>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    panel: GpuPanel,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    top: wgpu::Texture,
    bar: wgpu::Texture,
    group: wgpu::BindGroup,
    top_h: usize,
    bar_h: usize,
    /// HUD text scale (font pixels per glyph pixel).
    text: usize,
}

impl Gfx {
    fn new(window: Arc<Window>, ss: usize) -> Result<Gfx, String> {
        let (ctx, surface) = GpuContext::for_window(window.clone())?;
        let size = window.inner_size();
        let mut config = surface.get_default_config(&ctx.adapter, size.width.max(1), size.height.max(1)).ok_or("window surface is not supported by the adapter")?;
        // Prefer a non-sRGB format: the panel colours are already display values.
        let caps = surface.get_capabilities(&ctx.adapter);
        if let Some(f) = caps.formats.iter().copied().find(|f| !f.is_srgb()) {
            config.format = f;
        }
        let format = config.format;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&ctx.device, &config);
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_SHADER.into()),
        });
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1),
                tex(2),
                tex(3),
            ],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_present"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present"),
            size: std::mem::size_of::<PresentUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let text = (1.5 * window.scale_factor()).round().max(2.0) as usize;
        let (panel, top, bar, group, top_h, bar_h) = Gfx::targets(&ctx, &layout, &uniform, &config, ss, text);
        Ok(Gfx { window, ctx, surface, config, panel, pipeline, layout, uniform, top, bar, group, top_h, bar_h, text })
    }

    /// Size-dependent resources: the panel, the HUD textures and the bind group.
    fn targets(
        ctx: &Arc<GpuContext>,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        config: &wgpu::SurfaceConfiguration,
        ss: usize,
        text: usize,
    ) -> (GpuPanel, wgpu::Texture, wgpu::Texture, wgpu::BindGroup, usize, usize) {
        let (w, h) = (config.width as usize, config.height as usize);
        let line = LINE * text;
        let top_h = (9 * line + 6 * text).min(h);
        let bar_h = (2 * line + 2 * text).min(h.saturating_sub(top_h));
        // The panel's projection is set per frame; any is fine here.
        let proj = crate::camera::Projection { focal: 1.0, cx: 0.0, cy: 0.0 };
        let panel = GpuPanel::new(ctx, w * ss, h * ss, ss, proj);
        let hud = |label, rows: usize| {
            ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w as u32, height: rows.max(1) as u32, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (top, bar) = (hud("hud top", top_h), hud("hud bar", bar_h));
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (scene_view, top_view, bar_view) = (view(panel.output()), view(&top), view(&bar));
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&top_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bar_view) },
            ],
        });
        (panel, top, bar, group, top_h, bar_h)
    }

    fn resize(&mut self, size: PhysicalSize<u32>, ss: usize) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.ctx.device, &self.config);
        let (panel, top, bar, group, top_h, bar_h) = Gfx::targets(&self.ctx, &self.layout, &self.uniform, &self.config, ss, self.text);
        (self.panel, self.top, self.bar, self.group, self.top_h, self.bar_h) = (panel, top, bar, group, top_h, bar_h);
    }

    fn upload(&self, tex: &wgpu::Texture, c: &Canvas) {
        self.ctx.queue.write_texture(
            tex.as_image_copy(),
            &hud_rgba(c),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * c.width as u32), rows_per_image: Some(c.height as u32) },
            wgpu::Extent3d { width: c.width as u32, height: c.height as u32, depth_or_array_layers: 1 },
        );
    }

    /// Encodes the panel (already drawn) and presents it with the HUD strips.
    fn present(&mut self, top: &Canvas, bar: &Canvas) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.ctx.device, &self.config);
                return;
            }
            _ => return,
        };
        self.upload(&self.top, top);
        self.upload(&self.bar, bar);
        let u = PresentUniform {
            size: [self.config.width as f32, self.config.height as f32],
            top_h: self.top_h as f32,
            bar_h: self.bar_h as f32,
            srgb: self.config.format.is_srgb() as u32,
            pad: [0; 3],
        };
        self.ctx.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        self.panel.encode(&mut enc);
        let view = frame.texture.create_view(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ctx.queue.submit([enc.finish()]);
        self.window.pre_present_notify();
        self.ctx.queue.present(frame);
    }
}

/// Where the window looks: the live head or a recorded time.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Playhead {
    Live,
    /// Simulated time shown, and whether it advances (at `play_speed`).
    At { t: f64, playing: bool },
}

struct Drag {
    start: PhysicalPosition<f64>,
    last: PhysicalPosition<f64>,
    button: MouseButton,
    moved: bool,
    /// Started on the timeline bar.
    scrub: bool,
}

struct App {
    opts: ViewOptions,
    gfx: Option<Gfx>,
    tx: Sender<Command>,
    timeline: Arc<Mutex<Timeline>>,
    generation: u64,
    camera: Option<CameraSpec>,
    view: usize,
    scales: Scales,
    last_refit: Instant,
    playhead: Playhead,
    /// Replay speed: simulated seconds per wall second.
    play_speed: f64,
    last_redraw: Instant,
    fps: f64,
    cursor: PhysicalPosition<f64>,
    drag: Option<Drag>,
    modifiers: ModifiersState,
    shot_speed: f64,
    shot_radius: f64,
    help: bool,
    paused: bool,
    realtime: bool,
    message: Option<(String, Instant)>,
}

impl App {
    fn bounds(&self) -> Aabb {
        self.timeline.lock().unwrap().bounds.unwrap_or(Aabb { lo: Vec3::splat(-1.0), hi: Vec3::splat(1.0) })
    }

    fn send(&self, c: Command) {
        let _ = self.tx.send(c);
    }

    fn say(&mut self, m: impl Into<String>) {
        self.message = Some((m.into(), Instant::now()));
    }

    /// The camera spec with explicit distance and target, fitted to the scene.
    fn fit_camera(&mut self) {
        let Some(g) = &self.gfx else { return };
        let bounds = self.bounds();
        let mut spec = self.opts.camera.clone();
        let aspect = g.config.width as f64 / g.config.height.max(1) as f64;
        let cam = spec.build(&bounds, aspect);
        let along = (bounds.center() - cam.eye).dot(cam.forward).max(1e-3);
        spec.target = Some(cam.eye + cam.forward * along);
        spec.distance = Some(along);
        self.camera = Some(spec);
    }

    /// Pixel projection of the window at `scale` samples per pixel, its centre moved
    /// down so the scene sits between the status box and the timeline.
    fn projection(&self, cam: &Camera, scale: usize) -> Option<crate::camera::Projection> {
        let g = self.gfx.as_ref()?;
        let mut p = cam.projection(g.config.width as usize * scale, g.config.height as usize * scale);
        let status_box = 6 * LINE * g.text;
        p.cy += (0.5 * status_box.saturating_sub(g.bar_h) as f32) * scale as f32;
        Some(p)
    }

    fn current_camera(&self) -> Option<Camera> {
        let g = self.gfx.as_ref()?;
        Some(self.camera.as_ref()?.build(&self.bounds(), g.config.width as f64 / g.config.height.max(1) as f64))
    }

    /// The frame shown now (advancing the replay by `dt` wall seconds).
    fn advance(&mut self, dt: f64) -> Option<(usize, Arc<Frame>, usize)> {
        let t = self.timeline.lock().unwrap();
        let n = t.frames.len();
        let head_t = t.frames.last()?.time;
        if let Playhead::At { t: mut pt, mut playing } = self.playhead {
            if playing {
                pt += dt * self.play_speed;
            }
            self.playhead = if pt < head_t {
                Playhead::At { t: pt, playing }
            } else if playing && !t.paused {
                // Caught up with the running simulation.
                Playhead::Live
            } else {
                playing = false;
                Playhead::At { t: head_t, playing }
            };
        }
        let index = match self.playhead {
            Playhead::Live => n - 1,
            Playhead::At { t: pt, .. } => t.frames.partition_point(|f| f.time <= pt + 1e-12).saturating_sub(1),
        };
        Some((index, t.frames[index].clone(), n))
    }

    fn refit_scales(&mut self) {
        let (frames, chunk) = {
            let t = self.timeline.lock().unwrap();
            // At most ~400 frames, evenly spread.
            let step = t.frames.len().div_ceil(400).max(1);
            (t.frames.iter().step_by(step).cloned().collect::<Vec<_>>(), t.typical_chunk)
        };
        let refs: Vec<&Frame> = frames.iter().map(|f| f.as_ref()).collect();
        self.scales = Scales::fit_frames(&refs, chunk, self.opts.scale_overrides);
        self.last_refit = Instant::now();
    }

    fn mode(&self) -> ViewMode {
        self.opts.views[self.view % self.opts.views.len()]
    }

    fn redraw(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_redraw).as_secs_f64().min(0.25);
        self.last_redraw = now;
        if dt > 0.0 {
            self.fps = 0.9 * self.fps + 0.1 / dt;
        }
        let generation = self.timeline.lock().unwrap().generation;
        if generation != self.generation {
            self.generation = generation;
            if self.camera.is_none() {
                self.fit_camera();
            }
            self.refit_scales();
        }
        if self.last_refit.elapsed().as_secs_f64() > 1.0 {
            self.refit_scales();
        }
        let Some((index, frame, count)) = self.advance(dt) else { return };
        let Some(camera) = self.current_camera() else { return };
        let (bias, ground) = {
            let t = self.timeline.lock().unwrap();
            (0.4 * t.min_half_extent as f32, self.opts.scene.ground.as_ref().map(|g| g.height))
        };
        let bounds = self.bounds();
        let mode = self.mode();
        let top = self.hud_top(index, &frame, count);
        let bar = self.hud_bar(index, count);
        let ss = self.opts.opts.ss;
        let Some(proj) = self.projection(&camera, ss) else { return };
        let Some(g) = self.gfx.as_mut() else { return };
        g.panel.set_proj(proj);
        let scene = PanelScene { camera: &camera, mode, scales: &self.scales, opts: &self.opts.opts, ground, bounds, bias };
        scene.draw(&mut g.panel, &frame);
        g.present(&top, &bar);
    }

    fn hud_top(&self, index: usize, frame: &Frame, count: usize) -> Canvas {
        let g = self.gfx.as_ref().expect("window");
        let (w, s) = (g.config.width as usize, g.text);
        let mut c = Canvas::new(w, g.top_h, KEY);
        let t = self.timeline.lock().unwrap();
        let line = LINE * s;
        let pad = 3 * s;
        let mode = self.mode();
        let status = match (self.playhead, t.paused) {
            (Playhead::Live, false) => ("LIVE", LIVE),
            (Playhead::Live, true) => ("LIVE (sim paused)", ACCENT),
            (Playhead::At { playing: true, .. }, _) => ("REPLAY", ACCENT),
            (Playhead::At { playing: false, .. }, _) => ("REPLAY (paused)", ACCENT),
        };
        let end = t.frames.last().map_or(0.0, |f| f.time);
        let rate = if t.paused { "paused".to_string() } else if t.rate > 0.0 { rate_label(t.rate) } else { "-".into() };
        let mut lines: Vec<Vec<(String, Rgb)>> = vec![
            vec![
                (format!("{}  ", self.opts.scene.name), INK),
                (format!("[{}]", status.0), status.1),
                (format!("  t = {}  frame {}/{}", fmt_time(frame.time, end), index, count - 1), INK),
            ],
            vec![
                (format!("sim {rate}{}", if t.realtime { " (capped at real time, T)" } else { "" }), INK),
                (format!("  substep {}  mode: {} (M)", fmt_dt(t.substep), MODES[t.mode].1), MUTED),
            ],
            vec![(format!("engine: {} (E)", engine_label(t.engine)), INK)],
            vec![
                (format!("view: {} (V)", mode.title(&self.scales)), INK),
                (format!("  {} broken bonds, {} pieces, {} projectiles", frame.broken_bonds, pieces(frame), t.shots), MUTED),
            ],
            vec![
                (format!("shot: {} m/s, r {} m, {:.3} kg", fmt_number(self.shot_speed), fmt_number(self.shot_radius), shot_mass(self.shot_radius)), INK),
                ("  click to fire, [ ] speed, - = size".into(), MUTED),
            ],
        ];
        if self.help {
            for l in [
                "mouse: drag orbit, right/shift-drag pan, wheel zoom, click fire, drag the timeline to scrub".to_string(),
                "P pause sim, T real-time cap, M solver mode, E engine, R restart | Space play/pause, arrows step, L live".to_string(),
                format!(", . replay speed (x{}) | V view, A arrows, K cracks, C camera, H help, Q quit | {:.0} fps", fmt_number(self.play_speed), self.fps),
            ] {
                lines.push(vec![(l, MUTED)]);
            }
        } else {
            lines.push(vec![("H help".into(), MUTED)]);
        }
        if let Some(b) = &t.busy {
            lines.push(vec![(b.clone(), ACCENT)]);
        } else if let Some(e) = &t.error {
            lines.push(vec![(e.clone(), ACCENT)]);
        } else if let Some((m, at)) = &self.message {
            if at.elapsed().as_secs_f64() < 3.0 {
                lines.push(vec![(m.clone(), ACCENT)]);
            }
        }
        let rows = lines.len().min((g.top_h.saturating_sub(2 * pad)) / line);
        let width = lines.iter().take(rows).map(|l| l.iter().map(|(text, _)| font::text_width(text, s)).sum::<usize>()).max().unwrap_or(0);
        c.fill_rect(0, 0, (width + 2 * pad).min(w) as i64, (rows * line + 2 * pad) as i64, BOX);
        for (k, l) in lines.iter().take(rows).enumerate() {
            let mut x = pad as i64;
            for (text, color) in l {
                x += c.text(x, (pad + k * line) as i64, text, s, *color) as i64;
            }
        }
        c
    }

    fn hud_bar(&self, index: usize, count: usize) -> Canvas {
        let g = self.gfx.as_ref().expect("window");
        let (w, h, s) = (g.config.width as usize, g.bar_h, g.text);
        let mut c = Canvas::new(w, h, KEY);
        c.fill_rect(0, 0, w as i64, h as i64, BOX);
        let t = self.timeline.lock().unwrap();
        let end = t.frames.last().map_or(0.0, |f| f.time).max(1e-12);
        let (x0, x1) = (self.bar_x0() as i64, (w - 3 * s) as i64);
        let y = (h / 2) as i64;
        c.fill_rect(x0, y - s as i64, x1 - x0, 2 * s as i64, TRACK);
        let shown = t.frames.get(index).map_or(0.0, |f| f.time);
        let px = x0 + ((shown / end) * (x1 - x0) as f64) as i64;
        c.fill_rect(x0, y - s as i64, px - x0, 2 * s as i64, if self.playhead == Playhead::Live { LIVE } else { ACCENT });
        c.fill_rect(px - s as i64, y - 3 * s as i64, 2 * s as i64, 6 * s as i64, INK);
        let label = format!("{} / {}", fmt_time(shown, end), fmt_time(end, end));
        c.text(3 * s as i64, y - (LINE * s / 2) as i64 + s as i64, &label, s, if index + 1 == count { LIVE } else { INK });
        c
    }

    /// Left end of the timeline track (after the time label).
    fn bar_x0(&self) -> usize {
        let s = self.gfx.as_ref().map_or(2, |g| g.text);
        font::text_width("000.0 ms / 000.0 ms", s) + 8 * s
    }

    fn in_bar(&self, p: PhysicalPosition<f64>) -> bool {
        self.gfx.as_ref().is_some_and(|g| p.y >= (g.config.height as usize - g.bar_h) as f64)
    }

    /// Moves the playhead to the time under the cursor on the timeline.
    fn scrub(&mut self, x: f64) {
        let Some(g) = &self.gfx else { return };
        let (x0, x1) = (self.bar_x0() as f64, (g.config.width as usize - 3 * g.text) as f64);
        let end = self.timeline.lock().unwrap().frames.last().map_or(0.0, |f| f.time);
        let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0) * end;
        self.playhead = Playhead::At { t, playing: false };
    }

    /// Steps the playhead by `k` recorded frames.
    fn step_frames(&mut self, k: i64) {
        let (times, current) = {
            let t = self.timeline.lock().unwrap();
            let times: Vec<f64> = t.frames.iter().map(|f| f.time).collect();
            let current = match self.playhead {
                Playhead::Live => times.len() - 1,
                Playhead::At { t: pt, .. } => times.partition_point(|&x| x <= pt + 1e-12).saturating_sub(1),
            };
            (times, current)
        };
        let i = (current as i64 + k).clamp(0, times.len() as i64 - 1) as usize;
        self.playhead = Playhead::At { t: times[i], playing: false };
    }

    /// Fires a projectile along the ray under pixel `p`.
    fn fire(&mut self, p: PhysicalPosition<f64>) {
        let Some(cam) = self.current_camera() else { return };
        let Some(proj) = self.projection(&cam, 1) else { return };
        let (dx, dy) = ((p.x as f32 - proj.cx) / proj.focal, (proj.cy - p.y as f32) / proj.focal);
        let dir = (cam.forward + cam.right * dx as f64 + cam.up * dy as f64).normalized();
        // Start just outside the scene bounds along the ray, so the shot arrives at once.
        let b = self.bounds();
        let margin = 2.0 * self.shot_radius + 0.02 * b.size().norm();
        let lo = b.lo - Vec3::splat(margin);
        let hi = b.hi + Vec3::splat(margin);
        let start = match ray_box(cam.eye, dir, lo, hi) {
            Some(t_in) if t_in > 0.0 => cam.eye + dir * t_in,
            Some(_) => cam.eye + dir * (2.0 * self.shot_radius),
            None => cam.eye + dir * (0.5 * (b.center() - cam.eye).norm()),
        };
        // Firing into the past branches the run there.
        let (head, shown) = {
            let t = self.timeline.lock().unwrap();
            let head = t.frames.len() - 1;
            let shown = match self.playhead {
                Playhead::Live => head,
                Playhead::At { t: pt, .. } => t.frames.partition_point(|f| f.time <= pt + 1e-12).saturating_sub(1),
            };
            (head, shown)
        };
        if shown < head {
            self.send(Command::Branch(shown));
            self.say(format!("branched at frame {shown}: later frames dropped"));
        }
        self.send(Command::Fire(Shot { position: start, velocity: dir * self.shot_speed, radius: self.shot_radius, density: SHOT_DENSITY }));
        if self.paused {
            self.paused = false;
            self.send(Command::Pause(false));
        }
        self.playhead = Playhead::Live;
    }

    fn key(&mut self, event: &KeyEvent, el: &ActiveEventLoop) {
        if event.state != ElementState::Pressed {
            return;
        }
        let shift = self.modifiers.shift_key();
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => el.exit(),
            Key::Named(NamedKey::Space) => {
                self.playhead = match self.playhead {
                    Playhead::Live => {
                        let t = self.timeline.lock().unwrap().frames.last().map_or(0.0, |f| f.time);
                        Playhead::At { t, playing: false }
                    }
                    Playhead::At { t, playing } => {
                        // At the end, play from the start.
                        let end = self.timeline.lock().unwrap().frames.last().map_or(0.0, |f| f.time);
                        let t = if !playing && t >= end - 1e-12 { 0.0 } else { t };
                        Playhead::At { t, playing: !playing }
                    }
                }
            }
            Key::Named(NamedKey::ArrowLeft) => self.step_frames(if shift { -10 } else { -1 }),
            Key::Named(NamedKey::ArrowRight) => self.step_frames(if shift { 10 } else { 1 }),
            Key::Named(NamedKey::Home) => self.playhead = Playhead::At { t: 0.0, playing: false },
            Key::Named(NamedKey::End) => self.playhead = Playhead::Live,
            Key::Character(ch) => match ch.to_lowercase().as_str() {
                "q" => el.exit(),
                "v" => {
                    let n = self.opts.views.len();
                    self.view = if shift { (self.view + n - 1) % n } else { (self.view + 1) % n };
                }
                "p" => {
                    self.paused = !self.paused;
                    self.send(Command::Pause(self.paused));
                }
                "t" => {
                    self.realtime = !self.realtime;
                    self.send(Command::RealTime(self.realtime));
                    self.say(if self.realtime { "capped at real time" } else { "uncapped: as fast as the solver goes" });
                }
                "l" => self.playhead = Playhead::Live,
                "r" => {
                    let (mode, engine) = { let t = self.timeline.lock().unwrap(); (t.mode, t.engine) };
                    self.send(Command::Restart(mode, engine));
                    self.playhead = Playhead::Live;
                    self.say("restarted");
                }
                "m" => {
                    let (mut mode, engine) = { let t = self.timeline.lock().unwrap(); (t.mode, t.engine) };
                    loop {
                        mode = (mode + 1) % MODES.len();
                        if mode_allowed(mode, engine) {
                            break;
                        }
                    }
                    self.send(Command::Restart(mode, engine));
                    self.playhead = Playhead::Live;
                    self.say(format!("solver mode: {} (restarted)", MODES[mode].1));
                }
                "e" => {
                    let (mode, engine) = { let t = self.timeline.lock().unwrap(); (t.mode, t.engine) };
                    let other = if engine == Engine::Standalone { Engine::Rubble } else { Engine::Standalone };
                    if cfg!(feature = "rubble") {
                        let mode = if mode_allowed(mode, other) { mode } else { 0 };
                        self.send(Command::Restart(mode, other));
                        self.playhead = Playhead::Live;
                        self.say(format!("engine: {} (restarted)", engine_label(other)));
                    } else {
                        self.say("rubble: build stress-viz with --features rubble");
                    }
                }
                "c" => self.fit_camera(),
                "h" => self.help = !self.help,
                "a" => self.opts.opts.arrows = !self.opts.opts.arrows,
                "k" => self.opts.opts.cracks = !self.opts.opts.cracks,
                "[" => self.shot_speed = (self.shot_speed / 1.25).max(0.5),
                "]" => self.shot_speed = (self.shot_speed * 1.25).min(500.0),
                "-" | "_" => self.shot_radius = round_sig(self.shot_radius / 1.25).max(0.005),
                "=" | "+" => self.shot_radius = round_sig(self.shot_radius * 1.25).min(2.0),
                "," | "<" => self.play_speed = (self.play_speed / 2.0).max(1.0 / 4096.0),
                "." | ">" => self.play_speed = (self.play_speed * 2.0).min(64.0),
                _ => {}
            },
            _ => {}
        }
    }

    fn mouse_move(&mut self, p: PhysicalPosition<f64>) {
        self.cursor = p;
        let Some(d) = &mut self.drag else { return };
        let (dx, dy) = (p.x - d.last.x, p.y - d.last.y);
        d.last = p;
        if !d.moved && ((p.x - d.start.x).abs() + (p.y - d.start.y).abs()) > 4.0 {
            d.moved = true;
        }
        if !d.moved {
            return;
        }
        if d.scrub {
            self.scrub(p.x);
            return;
        }
        let pan = d.button == MouseButton::Right || self.modifiers.shift_key();
        let (Some(spec), Some(g)) = (self.camera.as_mut(), self.gfx.as_ref()) else { return };
        if pan {
            let cam = spec.build(&Aabb::EMPTY, 1.0);
            let dist = spec.distance.unwrap_or(1.0);
            let k = 2.0 * dist * (0.5 * spec.fov_deg.to_radians()).tan() / g.config.height.max(1) as f64;
            let shift = cam.right * (-dx * k) + cam.up * (dy * k);
            spec.target = Some(spec.target.unwrap_or(Vec3::ZERO) + shift);
        } else {
            spec.yaw_deg += 0.3 * dx;
            spec.pitch_deg = (spec.pitch_deg + 0.3 * dy).clamp(-89.0, 89.0);
        }
    }
}

fn round_sig(x: f64) -> f64 {
    let p = 10f64.powf(x.abs().log10().floor() - 1.0);
    (x / p).round() * p
}

const SHOT_DENSITY: f64 = 7850.0;

fn shot_mass(r: f64) -> f64 {
    SHOT_DENSITY * 4.0 / 3.0 * std::f64::consts::PI * r * r * r
}

fn pieces(f: &Frame) -> usize {
    f.chunks.iter().map(|c| c.cluster).collect::<std::collections::BTreeSet<_>>().len()
}

/// "x1.0 real time", "x0.015 real time (1/66)".
fn rate_label(r: f64) -> String {
    if r >= 0.95 {
        format!("x{} real time", fmt_number(round_sig(r)))
    } else {
        format!("x{} real time (1/{})", fmt_number(round_sig(r)), fmt_number(round_sig(1.0 / r)))
    }
}

fn fmt_dt(dt: f64) -> String {
    if dt >= 1e-3 {
        format!("{} ms", fmt_number(round_sig(dt * 1e3)))
    } else {
        format!("{} us", fmt_number(round_sig(dt * 1e6)))
    }
}

/// Entry distance of a ray into an axis-aligned box (negative: the origin is inside).
fn ray_box(o: Vec3, d: Vec3, lo: Vec3, hi: Vec3) -> Option<f64> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for (o, d, lo, hi) in [(o.x, d.x, lo.x, hi.x), (o.y, d.y, lo.y, hi.y), (o.z, d.z, lo.z, hi.z)] {
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (a, b) = ((lo - o) / d, (hi - o) / d);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t1 >= t0.max(0.0)).then_some(t0)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!("stress-viz — {}", self.opts.scene.name))
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("stress-viz: cannot open a window: {e}");
                el.exit();
                return;
            }
        };
        let ss = self.opts.ss.unwrap_or(if window.scale_factor() >= 1.5 { 1 } else { 2 });
        self.opts.opts.ss = ss;
        match Gfx::new(window, ss) {
            Ok(g) => {
                eprintln!("renderer: gpu, {}", g.ctx.description);
                self.gfx = Some(g);
            }
            Err(e) => {
                eprintln!("stress-viz: {e}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                let ss = self.opts.opts.ss;
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size, ss);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => self.key(&event, el),
            WindowEvent::CursorMoved { position, .. } => self.mouse_move(position),
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    let scrub = button == MouseButton::Left && self.in_bar(self.cursor);
                    if scrub {
                        self.scrub(self.cursor.x);
                    }
                    self.drag = Some(Drag { start: self.cursor, last: self.cursor, button, moved: false, scrub });
                }
                ElementState::Released => {
                    if let Some(d) = self.drag.take() {
                        if !d.moved && !d.scrub && d.button == MouseButton::Left && !self.modifiers.shift_key() {
                            self.fire(self.cursor);
                        }
                    }
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                if let Some(spec) = self.camera.as_mut() {
                    spec.distance = Some(spec.distance.unwrap_or(1.0) * 0.9f64.powf(steps));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }
}

/// Opens the window and runs until it is closed.
pub fn run(opts: ViewOptions) -> Result<(), String> {
    let timeline = Arc::new(Mutex::new(Timeline { realtime: true, ..Default::default() }));
    let (tx, rx) = mpsc::channel();
    let threads = opts.sim_threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?;
    // The world is built on the simulation thread (it is not `Send`: it may hold an
    // engine coupling); the first frame is recorded before the window opens.
    let (ready_tx, ready_rx) = mpsc::channel();
    if !mode_allowed(opts.mode, opts.engine) {
        return Err(format!("solver mode '{}' does not run with this engine (the rubble coupling needs an explicit or adaptive solve)", MODES[opts.mode].0));
    }
    let (scene, shared, engine, mode) = (opts.scene.clone(), timeline.clone(), opts.engine, opts.mode);
    let handle = std::thread::Builder::new()
        .name("simulation".into())
        .spawn(move || {
            pool.install(|| match Sim::new(scene, mode, engine, shared) {
                Ok(sim) => {
                    let _ = ready_tx.send(Ok(()));
                    sim.run(rx)
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
        })
        .map_err(|e| e.to_string())?;
    ready_rx.recv().map_err(|_| "simulation thread failed to start".to_string())??;
    eprintln!("simulation: {threads} threads (--sim-threads)");

    let b = timeline.lock().unwrap().bounds.unwrap_or(Aabb::EMPTY);
    let diag = b.size().norm();
    let mut app = App {
        scales: Scales { stress: 1.0, von_mises: 1.0, speed: 1.0, deformation: 1.0, exaggeration: 1.0 },
        opts,
        gfx: None,
        tx,
        timeline,
        generation: 0,
        camera: None,
        view: 0,
        last_refit: Instant::now(),
        playhead: Playhead::Live,
        play_speed: 1.0,
        last_redraw: Instant::now(),
        fps: 60.0,
        cursor: PhysicalPosition::new(0.0, 0.0),
        drag: None,
        modifiers: ModifiersState::empty(),
        shot_speed: 20.0,
        shot_radius: round_sig((0.035 * diag).clamp(0.02, 0.4)),
        help: false,
        paused: false,
        realtime: true,
        message: None,
    };
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.run_app(&mut app).map_err(|e| format!("event loop: {e}"))?;
    drop(app);
    let _ = handle.join();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stress_ref::builders;
    use stress_ref::showcases;

    fn sim(name: &str) -> Sim {
        let scene = builders::catalog().into_iter().chain(showcases::catalog()).find(|s| s.name == name).expect("catalogue scene");
        Sim::new(scene, 0, Engine::Standalone, Arc::new(Mutex::new(Timeline::default()))).unwrap()
    }

    /// A shot from outside the bounds along +y, straight at the middle chunk.
    fn shot_at_centre(s: &Sim, speed: f64) -> Shot {
        let b = s.recorder.bounds;
        let frame = s.recorder.capture(&s.world);
        let c = frame.chunks[frame.chunks.len() / 2].center.map(f64::from);
        let c = Vec3::new(c[0], c[1], c[2]);
        Shot { position: Vec3::new(c.x, b.lo.y - 0.3, c.z), velocity: Vec3::new(0.0, speed, 0.0), radius: 0.1, density: SHOT_DENSITY }
    }

    #[test]
    fn a_shot_caps_the_substep_and_hits_the_structure() {
        let mut s = sim("s_arch_keystone_removed");
        let before = s.world.substep_dt();
        let shot = shot_at_centre(&s, 30.0);
        s.spawn(&shot);
        let cap = s.world.scene.sim.max_substep.expect("a projectile caps the substep");
        assert!((cap - shot_dt(&s.world, s.world.impactors.last().unwrap())).abs() < 1e-15);
        assert!(s.world.substep_dt() <= cap.min(before) * (1.0 + 1e-12));
        let v0 = s.world.impactors.last().unwrap().velocity.norm();
        for _ in 0..40 {
            s.step();
        }
        let v1 = s.world.impactors.last().map_or(0.0, |i| i.velocity.norm());
        assert!(v1 < 0.9 * v0, "the shot is slowed by the arch: {v0} -> {v1}");
        assert!(s.world.substep_dt() > 0.0);
    }

    #[test]
    fn branching_reproduces_the_recorded_frames() {
        let mut s = sim("s_arch_keystone_removed");
        for _ in 0..10 {
            s.step();
        }
        let shot = shot_at_centre(&s, 30.0);
        s.shots.push((10, shot));
        s.spawn(&shot);
        for _ in 0..30 {
            s.step();
        }
        let before: Vec<Arc<Frame>> = s.timeline.lock().unwrap().frames.clone();
        // Thin the checkpoints so the branch must re-simulate (and replay the shot).
        s.checkpoints.retain(|(i, _)| *i == 0);
        s.branch(25);
        let after = s.timeline.lock().unwrap().frames.clone();
        assert_eq!(after.len(), 26);
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.time, b.time);
            assert_eq!(a.broken_bonds, b.broken_bonds);
            assert_eq!(a.impactors.len(), b.impactors.len(), "the logged shot is replayed at t = {}", a.time);
            for (ca, cb) in a.chunks.iter().zip(&b.chunks) {
                assert_eq!(ca.center, cb.center, "deterministic re-simulation at t = {}", a.time);
            }
        }
    }

    /// With rubble as the engine: a shot is synced into the engine and hits, and a
    /// branch (re-simulated from the start, no snapshots) reproduces the frames.
    #[cfg(feature = "rubble")]
    #[test]
    fn rubble_shots_hit_and_branches_replay_from_the_start() {
        let scene = showcases::catalog().into_iter().find(|s| s.name == "s_arch_keystone_removed").unwrap();
        let mut s = Sim::new(scene, 0, Engine::Rubble, Arc::new(Mutex::new(Timeline::default()))).unwrap();
        assert!(s.checkpoints.is_empty());
        for _ in 0..5 {
            s.step();
        }
        let shot = shot_at_centre(&s, 30.0);
        s.shots.push((5, shot));
        s.spawn(&shot);
        let v0 = shot.velocity.norm();
        for _ in 0..40 {
            s.step();
        }
        let v1 = s.world.impactors.last().unwrap().velocity.norm();
        assert!(v1 < 0.9 * v0, "the shot is slowed by the arch: {v0} -> {v1}");
        let before = s.timeline.lock().unwrap().frames.clone();
        s.branch(30);
        let after = s.timeline.lock().unwrap().frames.clone();
        assert_eq!(after.len(), 31);
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.impactors.len(), b.impactors.len(), "the logged shot is replayed at t = {}", a.time);
            for (ca, cb) in a.chunks.iter().zip(&b.chunks) {
                assert_eq!(ca.center, cb.center, "deterministic rubble re-simulation at t = {}", a.time);
            }
        }
    }

    /// Adaptive mode can be picked by name, runs (with either engine) and settles an
    /// idle structure; the rubble coupling refuses the implicit solve.
    #[test]
    fn adaptive_mode_is_selectable_and_settles_an_idle_structure() {
        use stress_ref::scene::SolveMode;
        use stress_ref::solver::Activity;
        let adaptive = mode_index("adaptive").expect("listed");
        assert!(mode_allowed(adaptive, Engine::Rubble) && mode_allowed(adaptive, Engine::Standalone));
        assert!(!mode_allowed(mode_index("implicit").unwrap(), Engine::Rubble));
        let scene = showcases::catalog().into_iter().find(|s| s.name == "s_house").unwrap();
        let mut s = Sim::new(scene, adaptive, Engine::Standalone, Arc::new(Mutex::new(Timeline::default()))).unwrap();
        assert_eq!(s.world.scene.sim.solve_mode, SolveMode::Adaptive);
        for _ in 0..600 {
            s.step();
        }
        assert!(s.world.solver.clusters.iter().all(|c| c.activity == Activity::Settled), "the idle house settles");
    }

    #[test]
    fn ray_box_entry() {
        let (lo, hi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(ray_box(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0), lo, hi), Some(4.0));
        assert!(ray_box(Vec3::new(0.0, -5.0, 3.0), Vec3::new(0.0, 1.0, 0.0), lo, hi).is_none());
        assert!(ray_box(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), lo, hi).unwrap() < 0.0, "inside");
    }
}
