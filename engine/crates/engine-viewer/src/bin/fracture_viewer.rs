//! Native macOS fracture lab: Shardwright GLB → Slang AVBD on Metal (Slang → MSL).
//!
//! ```bash
//! cargo run -p engine-viewer --release --bin fracture_viewer            # house
//! cargo run -p engine-viewer --release --bin fracture_viewer -- arch --level 1
//! cargo run -p engine-viewer --release --bin fracture_viewer -- path/to/model.fracture.glb
//! ```
//!
//! Click to shoot · drag to orbit · scroll to zoom · Space start/pause · R reset ·
//! B bonds · [ ] level · 1 house · 2 arch · drop a .glb onto the window.

// objc 0.2 `msg_send!` expands a cfg the current compiler does not know.
#![allow(unexpected_cfgs)]

use core_graphics_types::geometry::CGSize;
use engine_fracture::{build_scene, parse_fracture_glb, FractureAsset, FractureScene};
use engine_viewer::fracture::{FractureSim, Orbit, Renderer};
use metal::*;
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use std::path::{Path, PathBuf};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const TPS: f64 = 120.0;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/fracture/assets")
}

fn resolve(source: &str) -> PathBuf {
    match source {
        "house" => assets_dir().join("veneer-house.glb"),
        "arch" => assets_dir().join("masonry-arch.glb"),
        path => PathBuf::from(path),
    }
}

/// Wall-clock rates (port of `fracture/rates.mjs`): physics ticks and submitted frames.
#[derive(Default)]
struct Rates {
    start: Option<Instant>,
    frames: u32,
    steps: u32,
    step_ms: f64,
    encode_ms: f64,
    wait_ms: f64,
    summary: String,
}

impl Rates {
    fn reset(&mut self) {
        *self = Self {
            start: Some(Instant::now()),
            ..Default::default()
        };
    }
    fn step(&mut self, t: &engine_viewer::fracture::sim::Timing) {
        self.steps += 1;
        self.step_ms += t.wall_ms;
        self.encode_ms += t.encode_ms;
        self.wait_ms += t.wait_ms;
    }
    /// Returns true once per second when a new summary is published.
    fn frame(&mut self) -> bool {
        self.frames += 1;
        let start = *self.start.get_or_insert_with(Instant::now);
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed < 1.0 {
            return false;
        }
        let (tps, fps) = (self.steps as f64 / elapsed, self.frames as f64 / elapsed);
        let steps = self.steps.max(1) as f64;
        self.summary = if self.steps > 0 {
            format!(
                "{tps:.0} TPS · {fps:.0} FPS · {:.2}× · {:.1} ms/tick ({:.1} submit + {:.1} wait)",
                tps / TPS,
                self.step_ms / steps,
                self.encode_ms / steps,
                self.wait_ms / steps
            )
        } else {
            format!("{tps:.0} TPS · {fps:.0} FPS · {:.2}×", tps / TPS)
        };
        let summary = std::mem::take(&mut self.summary);
        *self = Self {
            start: Some(Instant::now()),
            summary,
            ..Default::default()
        };
        true
    }
}

struct Loaded {
    name: String,
    asset: FractureAsset,
    levels: Vec<u8>,
    level: u8,
    fragments: usize,
    hulls: usize,
    skipped: usize,
    bonds: usize,
    projectile_radius: f32,
    sim: FractureSim,
    renderer: Renderer,
    orbit: Orbit,
}

struct Drag {
    start: PhysicalPosition<f64>,
    last: PhysicalPosition<f64>,
    moved: bool,
}

struct App {
    device: Device,
    queue: CommandQueue,
    source: PathBuf,
    initial_level: Option<u8>,
    /// `--shoot`: fire one sphere at the window centre after loading (smoke tests).
    auto_shoot: bool,
    window: Option<Window>,
    layer: Option<MetalLayer>,
    loaded: Option<Loaded>,
    running: bool,
    show_bonds: bool,
    accumulator: f64,
    last_frame: Option<Instant>,
    rates: Rates,
    cursor: PhysicalPosition<f64>,
    drag: Option<Drag>,
    status: String,
}

impl App {
    fn load(&mut self, path: &Path, level: Option<u8>) {
        self.running = false;
        if let Some(l) = &mut self.loaded {
            let _ = l.sim.wait();
        }
        let started = Instant::now();
        let result = std::fs::read(path)
            .map_err(anyhow::Error::from)
            .and_then(|bytes| parse_fracture_glb(&bytes))
            .and_then(|asset| {
                let levels = asset.levels();
                let wanted = level.filter(|l| levels.contains(l)).unwrap_or(levels[0]);
                let parse_ms = started.elapsed().as_secs_f64() * 1000.0;
                let name = path
                    .file_name()
                    .map_or("model".into(), |n| n.to_string_lossy().into_owned());
                println!("parsed {name} in {parse_ms:.0} ms · levels {levels:?}");
                // Unloadable levels (e.g. degenerate baked hulls) fall back to the
                // nearest loadable one, preferring coarser levels.
                let mut order = levels.clone();
                order.sort_by_key(|&l| ((l as i32 - wanted as i32).abs(), l > wanted));
                let mut first_error = None;
                for level in order {
                    match build_scene(&asset, level) {
                        Ok(scene) => {
                            let notice = first_error.map(|e: anyhow::Error| {
                                eprintln!("level {wanted} unavailable: {e:#}");
                                format!("level {wanted} unavailable ({e}) · showing level {level}")
                            });
                            return Ok((self.build(asset, levels, scene, name)?, notice));
                        }
                        Err(e) => {
                            first_error.get_or_insert(e);
                        }
                    }
                }
                Err(first_error.unwrap())
            });
        match result {
            Ok((l, notice)) => {
                self.loaded = Some(l);
                self.status = notice.unwrap_or_else(|| "click the model to shoot".into());
            }
            Err(e) => self.fail(e),
        }
        self.rates.reset();
        self.update_title();
    }

    fn build(
        &self,
        asset: FractureAsset,
        levels: Vec<u8>,
        scene: FractureScene,
        name: String,
    ) -> anyhow::Result<Loaded> {
        let t = Instant::now();
        let level = scene.level;
        let sim = FractureSim::new(&self.device, &self.queue, &scene)?;
        let renderer = Renderer::new(&self.device, MTLPixelFormat::BGRA8Unorm, &scene)?;
        println!(
            "{name} level {level}: {} fragments · {} hulls ({} flat skipped) · {} bonds · {} bodies · built in {:.0} ms",
            scene.fragment_count,
            scene.hull_count,
            scene.skipped_hulls.len(),
            scene.bond_count,
            scene.bodies.len(),
            t.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Loaded {
            name,
            asset,
            levels,
            level,
            fragments: scene.fragment_count,
            hulls: scene.hull_count,
            skipped: scene.skipped_hulls.len(),
            bonds: scene.bond_count,
            projectile_radius: scene.projectile_radius,
            sim,
            renderer,
            orbit: Orbit::new(&scene),
        })
    }

    fn select_level(&mut self, step: i32) {
        let Some(l) = self.loaded.take() else { return };
        let i = l.levels.iter().position(|&x| x == l.level).unwrap_or(0) as i32 + step;
        let Some(&level) = l.levels.get(i.max(0) as usize).filter(|_| i >= 0) else {
            self.loaded = Some(l);
            return;
        };
        if level == l.level {
            self.loaded = Some(l);
            return;
        }
        self.running = false;
        // Build the scene first so an unloadable level keeps the current one.
        let scene = match build_scene(&l.asset, level) {
            Ok(scene) => scene,
            Err(e) => {
                eprintln!("level {level} unavailable: {e:#}");
                self.status = format!(
                    "level {level} unavailable ({e}) · staying on level {}",
                    l.level
                );
                self.loaded = Some(l);
                self.update_title();
                return;
            }
        };
        let Loaded {
            asset,
            levels,
            name,
            ..
        } = l;
        match self.build(asset, levels, scene, name) {
            Ok(next) => {
                self.loaded = Some(next);
                self.status = "click the model to shoot".into();
            }
            Err(e) => {
                self.fail(e);
                self.load(&self.source.clone(), None);
                return;
            }
        }
        self.rates.reset();
        self.update_title();
    }

    fn fail(&mut self, e: anyhow::Error) {
        eprintln!("error: {e:#}");
        self.running = false;
        self.status = format!("error: {e}");
        self.update_title();
    }

    fn update_title(&self) {
        let Some(window) = &self.window else { return };
        let title = match &self.loaded {
            Some(l) => {
                let c = l.sim.last;
                let rates = if self.rates.summary.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", self.rates.summary)
                };
                format!(
                    "Fracture lab · Metal · {} L{} · {} frags / {} hulls{} / {} bonds · {}{} · {}/16 shots · {} contacts · {} awake · {:.1} s{}",
                    l.name,
                    l.level,
                    l.fragments,
                    l.hulls,
                    if l.skipped > 0 { format!(" ({} flat skipped)", l.skipped) } else { String::new() },
                    l.bonds,
                    if self.running { "running" } else { "paused" },
                    rates,
                    l.sim.shots,
                    c.contact,
                    c.awake,
                    l.sim.steps as f64 / TPS,
                    if self.status.starts_with("error") || self.status.starts_with("level") { format!(" · {}", self.status) } else { String::new() },
                )
            }
            None => format!("Fracture lab · Metal · {}", self.status),
        };
        window.set_title(&title);
    }

    fn shoot(&mut self, at: PhysicalPosition<f64>) {
        let (Some(window), Some(l)) = (&self.window, &mut self.loaded) else {
            return;
        };
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let (u, v) = (
            at.x as f32 / size.width as f32,
            at.y as f32 / size.height as f32,
        );
        let (origin, dir) = l.orbit.aim(u, v, aspect, l.projectile_radius);
        match l.sim.launch(origin.to_array(), dir.to_array()) {
            Ok(_) => {
                if !self.running {
                    self.rates.reset();
                }
                self.running = true;
                self.accumulator = 0.0;
            }
            Err(e) => self.fail(e),
        }
    }

    /// One frame: advance physics on a 120 Hz accumulator, draw, then read counters.
    fn frame(&mut self) {
        let now = Instant::now();
        let elapsed = self
            .last_frame
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.04));
        self.last_frame = Some(now);
        let (Some(layer), Some(l)) = (&self.layer, &mut self.loaded) else {
            return;
        };
        let mut ticks = 0;
        let mut error = None;
        if self.running {
            self.accumulator = (self.accumulator + elapsed).min(4.0 / TPS);
            // Bound outstanding work: slow ticks get one tick per frame.
            let budget = ((12.0 / l.sim.timing.wall_ms.max(0.1)).floor() as u32).clamp(1, 4);
            ticks = budget.min((self.accumulator * TPS).floor() as u32);
            if ticks > 0 {
                self.accumulator -= ticks as f64 / TPS;
                if let Err(e) = l.sim.submit(ticks) {
                    error = Some(e);
                    ticks = 0;
                }
            }
        }
        // Queue the draw immediately behind physics; the same queue orders them.
        autoreleasepool(|| {
            if let Some(drawable) = layer.next_drawable() {
                let aspect =
                    drawable.texture().width() as f32 / drawable.texture().height().max(1) as f32;
                l.renderer.draw(
                    &self.device,
                    &self.queue,
                    drawable,
                    l.sim.body_buffer(),
                    l.orbit.view_proj(aspect),
                    l.sim.shots,
                    self.show_bonds,
                );
            }
        });
        if ticks > 0 {
            match l.sim.wait() {
                Ok(()) => {
                    for _ in 0..ticks {
                        self.rates.step(&l.sim.timing);
                    }
                }
                Err(e) => error = Some(e),
            }
        }
        if let Some(e) = error {
            self.fail(e);
        }
        if self.rates.frame() {
            if self.running {
                println!("{} · {}", self.rates.summary, self.counters_line());
            }
            self.update_title();
        }
    }

    fn counters_line(&self) -> String {
        let Some(l) = &self.loaded else {
            return String::new();
        };
        let c = l.sim.last;
        format!(
            "{}/16 shots · {} steps · {} contacts · {} candidates · {} awake · {:.1} s",
            l.sim.shots,
            l.sim.steps,
            c.contact,
            c.candidate,
            c.awake,
            l.sim.steps as f64 / TPS
        )
    }

    fn resize(&self) {
        if let (Some(window), Some(layer)) = (&self.window, &self.layer) {
            let size = window.inner_size();
            layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
        }
    }
}

fn autoreleasepool<T>(f: impl FnOnce() -> T) -> T {
    objc::rc::autoreleasepool(f)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Fracture lab · Metal")
            .with_inner_size(LogicalSize::new(1400.0, 900.0));
        let window = event_loop.create_window(attrs).expect("window");
        window.focus_window();
        let layer = MetalLayer::new();
        layer.set_device(&self.device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);
        layer.set_maximum_drawable_count(3);
        if let Ok(RawWindowHandle::AppKit(rw)) = window.window_handle().map(|h| h.as_raw()) {
            unsafe {
                let view = rw.ns_view.as_ptr() as *mut Object;
                let () = msg_send![view, setWantsLayer: YES];
                let () = msg_send![view, setLayer: layer.as_ref() as *const MetalLayerRef as *mut Object];
            }
        }
        layer.set_contents_scale(window.scale_factor());
        self.window = Some(window);
        self.layer = Some(layer);
        self.resize();
        self.status = format!("loading {}…", self.source.display());
        self.update_title();
        let source = self.source.clone();
        self.load(&source, self.initial_level);
        if self.auto_shoot {
            let size = self.window.as_ref().unwrap().inner_size();
            self.shoot(PhysicalPosition::new(
                size.width as f64 / 2.0,
                size.height as f64 / 2.0,
            ));
        }
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                if let (Some(w), Some(layer)) = (&self.window, &self.layer) {
                    layer.set_contents_scale(w.scale_factor());
                }
                self.resize();
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                self.source = path.clone();
                self.load(&path, None);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = position;
                let scale = self.window.as_ref().map_or(1.0, |w| w.scale_factor());
                if let Some(d) = &mut self.drag {
                    if ((position.x - d.start.x).powi(2) + (position.y - d.start.y).powi(2)).sqrt()
                        > 5.0 * scale
                    {
                        d.moved = true;
                    }
                    if d.moved {
                        if let Some(l) = &mut self.loaded {
                            l.orbit.drag(
                                ((position.x - d.last.x) / scale) as f32,
                                ((position.y - d.last.y) / scale) as f32,
                            );
                        }
                    }
                    d.last = position;
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => {
                    self.drag = Some(Drag {
                        start: self.cursor,
                        last: self.cursor,
                        moved: false,
                    })
                }
                ElementState::Released => {
                    if let Some(d) = self.drag.take() {
                        if !d.moved {
                            self.shoot(self.cursor);
                        }
                    }
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 100.0,
                    MouseScrollDelta::PixelDelta(p) => -p.y as f32,
                };
                if let Some(l) = &mut self.loaded {
                    l.orbit.zoom(dy);
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => {
                        self.running = !self.running && self.loaded.is_some();
                        self.accumulator = 0.0;
                        self.rates.reset();
                    }
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Character(c) => match c.to_ascii_lowercase().as_str() {
                        "r" => {
                            if let Some(l) = &mut self.loaded {
                                l.sim.reset();
                            }
                            self.running = false;
                            self.status = "reset · click the model to shoot".into();
                        }
                        "b" => self.show_bonds = !self.show_bonds,
                        "[" => self.select_level(-1),
                        "]" => self.select_level(1),
                        "1" | "2" => {
                            self.source = resolve(if c == "1" { "house" } else { "arch" });
                            self.load(&self.source.clone(), None);
                        }
                        _ => {}
                    },
                    _ => {}
                }
                self.update_title();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let (mut source, mut level, mut auto_shoot) = ("house".to_string(), None, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" => level = args.next().and_then(|s| s.parse().ok()),
            "--shoot" => auto_shoot = true,
            "-h" | "--help" => {
                println!("fracture_viewer [house|arch|path.glb] [--level N] [--shoot]");
                return Ok(());
            }
            other => source = other.to_string(),
        }
    }
    let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    println!("Metal device: {}", device.name());
    let queue = device.new_command_queue();
    let event_loop = EventLoop::new()?;
    let mut app = App {
        device,
        queue,
        source: resolve(&source),
        initial_level: level,
        auto_shoot,
        window: None,
        layer: None,
        loaded: None,
        running: false,
        show_bonds: false,
        accumulator: 0.0,
        last_frame: None,
        rates: Rates::default(),
        cursor: PhysicalPosition::new(0.0, 0.0),
        drag: None,
        status: String::new(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
