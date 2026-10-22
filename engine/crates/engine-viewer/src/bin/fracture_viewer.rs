//! Native macOS fracture lab: the browser lab's scenes, host plan and Slang
//! kernels (engine-fracture + Slang -> MSL) on Metal.
//!
//! ```bash
//! cargo run -p engine-viewer --release --bin fracture_viewer            # house
//! cargo run -p engine-viewer --release --bin fracture_viewer -- grid --shoot
//! cargo run -p engine-viewer --release --bin fracture_viewer -- arch --level 1
//! cargo run -p engine-viewer --release --bin fracture_viewer -- path/to/model.fracture.glb
//! ```
//!
//! 1-9 scenes (the lab's Scene menu, in its order) · [ ] level · drop a .glb onto the window ·
//! click to shoot · F fire the wrecking ball (rows, grids, suburbs) · drag to
//! orbit · scroll to zoom · Space start/pause · R reset · B bond graph ·
//! K breakable bonds (experimental) · C colours (materials, sleep state, bodies) · Esc quit.

// objc 0.2 `msg_send!` expands a cfg the current compiler does not know.
#![allow(unexpected_cfgs)]

use core_graphics_types::geometry::CGSize;
use engine_fracture::samples::{base_scene, layout_scene, Layout, Sample, SAMPLES};
use engine_fracture::scene::{Ray, Scene};
use engine_fracture::{parse_fracture_glb, FractureAsset};
use engine_viewer::fracture::{FractureSim, Orbit, Renderer};
use metal::*;
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use std::path::PathBuf;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/fracture/assets")
}

/// What is loaded: a bundled sample (the Scene menu) or a dropped GLB.
#[derive(Clone)]
enum Source {
    Sample(&'static Sample),
    File(PathBuf),
}

impl Source {
    fn parse(arg: &str) -> Self {
        match SAMPLES.iter().find(|s| s.key == arg) {
            Some(s) => Source::Sample(s),
            None => Source::File(PathBuf::from(arg)),
        }
    }
    fn file(&self) -> Option<PathBuf> {
        match self {
            Source::Sample(s) => s.file.map(|f| assets_dir().join(f)),
            Source::File(p) => Some(p.clone()),
        }
    }
    fn layout(&self) -> Option<Layout> {
        match self {
            Source::Sample(s) => s.layout,
            Source::File(_) => None,
        }
    }
    fn level(&self) -> Option<u8> {
        match self {
            Source::Sample(s) => s.level,
            Source::File(_) => None,
        }
    }
    fn max_hull_vertices(&self) -> Option<usize> {
        match self {
            Source::Sample(s) => s.max_hull_vertices,
            Source::File(_) => None,
        }
    }
    fn name(&self) -> String {
        match self {
            Source::Sample(s) => s.menu.to_string(),
            Source::File(p) => p
                .file_name()
                .map_or("model".into(), |n| n.to_string_lossy().into_owned()),
        }
    }
}

/// Wall-clock rates (as `fracture/rates.mjs`): physics ticks and frames.
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
    /// True once per second, when a new summary is published.
    fn frame(&mut self, hz: f64) -> bool {
        self.frames += 1;
        let start = *self.start.get_or_insert_with(Instant::now);
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed < 1.0 {
            return false;
        }
        let (tps, fps) = (self.steps as f64 / elapsed, self.frames as f64 / elapsed);
        let steps = self.steps.max(1) as f64;
        let summary = if self.steps > 0 {
            format!(
                "{tps:.0} TPS · {fps:.0} FPS · {:.2}× · {:.1} ms/tick ({:.1} encode + {:.1} wait)",
                tps / hz,
                self.step_ms / steps,
                self.encode_ms / steps,
                self.wait_ms / steps
            )
        } else {
            format!("{tps:.0} TPS · {fps:.0} FPS")
        };
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
    level: Option<u8>,
    scene: Scene,
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
    source: Source,
    /// The parsed GLB of the current source (kept across level switches).
    asset: Option<(PathBuf, FractureAsset)>,
    initial_level: Option<u8>,
    /// `--shoot`: fire once after loading (the row shot when there is one).
    auto_shoot: bool,
    bonds: bool,
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
    fn hz(&self) -> f64 {
        self.loaded
            .as_ref()
            .map_or(120.0, |l| 1.0 / l.scene.params.dt as f64)
    }

    /// Load `source` at `level` (default: the sample's level, else the first).
    /// An unloadable level falls back to the nearest loadable one.
    fn load(&mut self, source: Source, level: Option<u8>) {
        self.running = false;
        if let Some(l) = &mut self.loaded {
            let _ = l.sim.wait();
        }
        let result = (|| -> anyhow::Result<(Loaded, Option<String>)> {
            let Some(path) = source.file() else {
                return Ok((self.build(&source, None)?, None));
            };
            if self.asset.as_ref().is_none_or(|(p, _)| *p != path) {
                let t = Instant::now();
                let asset = parse_fracture_glb(&std::fs::read(&path)?)?;
                println!(
                    "parsed {} in {:.0} ms · levels {:?}",
                    path.display(),
                    t.elapsed().as_secs_f64() * 1000.0,
                    asset.levels()
                );
                self.asset = Some((path, asset));
            }
            let levels = self.asset.as_ref().unwrap().1.levels();
            let wanted = engine_fracture::samples::initial_level(
                &self.asset.as_ref().unwrap().1,
                level.or(source.level()),
            )
            .unwrap_or(0);
            let mut order = levels.clone();
            order.sort_by_key(|&l| ((l as i32 - wanted as i32).abs(), l > wanted));
            let mut first_error = None;
            for l in order {
                match self.build(&source, Some(l)) {
                    Ok(loaded) => {
                        let notice = first_error.map(|e: anyhow::Error| {
                            eprintln!("level {wanted} unavailable: {e:#}");
                            format!("level {wanted} unavailable ({e}) · showing level {l}")
                        });
                        return Ok((loaded, notice));
                    }
                    Err(e) => {
                        first_error.get_or_insert(e);
                    }
                }
            }
            Err(first_error.unwrap_or_else(|| anyhow::anyhow!("no levels")))
        })();
        match result {
            Ok((l, notice)) => {
                self.loaded = Some(l);
                self.source = source;
                self.status = notice.unwrap_or_default();
            }
            Err(e) => self.fail(e),
        }
        self.rates.reset();
        self.update_title();
    }

    /// One scene as the browser lab builds it: one house, preloaded bonds or
    /// none, then the sample's layout.
    fn build(&self, source: &Source, level: Option<u8>) -> anyhow::Result<Loaded> {
        let t = Instant::now();
        let asset = level.and(self.asset.as_ref().map(|(_, a)| a));
        let built = base_scene(asset, level.unwrap_or(0), source.max_hull_vertices())?;
        let one = if self.bonds {
            self.preload(&built)?
        } else {
            built.without_joints()
        };
        let scene = layout_scene(one, source.layout(), asset.is_none());
        let sim = FractureSim::new(&self.device, &self.queue, &scene)?;
        let renderer = Renderer::new(&self.device, MTLPixelFormat::BGRA8Unorm, &scene)?;
        let name = source.name();
        println!(
            "{name}{}: {} copies · {} bodies · {} colliders · {} joints · {} flat hulls skipped · built in {:.0} ms",
            level.map_or(String::new(), |l| format!(" level {l}")),
            scene.copies,
            scene.bodies.len(),
            scene.colliders.len(),
            scene.joints.len(),
            scene.skipped_hulls.len(),
            t.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Loaded {
            name,
            level,
            orbit: Orbit::new(&scene),
            scene,
            sim,
            renderer,
        })
    }

    /// Settle one copy under its own weight with breaking off, then start it
    /// asleep at the settled poses (as the browser lab's `preloadBonds`).
    fn preload(&self, scene: &Scene) -> anyhow::Result<Scene> {
        if scene.joints.is_empty() {
            return Ok(scene.clone());
        }
        let mut sim = FractureSim::new(&self.device, &self.queue, &scene.preload_variant())?;
        let mut steps = 0;
        loop {
            sim.submit(4)?;
            sim.wait()?;
            steps += 4;
            if Scene::preload_done(steps, sim.last.awake) {
                break;
            }
        }
        println!("bonds settled in {steps} steps ({} awake)", sim.last.awake);
        scene.settled(
            &sim.snapshot(scene.bodies.len())?,
            steps,
            sim.last.awake == 0,
        )
    }

    fn select_level(&mut self, step: i32) {
        let (Some(l), Some((_, asset))) = (&self.loaded, &self.asset) else {
            return;
        };
        let Some(current) = l.level else { return };
        let levels = asset.levels();
        let i = levels.iter().position(|&x| x == current).unwrap_or(0) as i32 + step;
        let Some(&level) = levels.get(i.max(0) as usize).filter(|_| i >= 0) else {
            return;
        };
        if level == current {
            return;
        }
        self.running = false;
        // Build the scene first so an unloadable level keeps the current one.
        match self.build(&self.source.clone(), Some(level)) {
            Ok(next) => {
                self.loaded = Some(next);
                self.status.clear();
            }
            Err(e) => {
                eprintln!("level {level} unavailable: {e:#}");
                self.status =
                    format!("level {level} unavailable ({e}) · staying on level {current}");
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
                let (c, s) = (l.sim.last, &l.scene);
                let copies = s.copies.max(1);
                format!(
                    "Fracture lab · Metal · {}{} · {} pieces · {} {} · {} bonds{} · {} · {}{}/16 shots · {} contacts · {} awake · {:.1} s{}",
                    l.name,
                    l.level.map_or(String::new(), |v| format!(" · L{v}")),
                    s.selected.len() as u32 * copies,
                    if s.pieces > 0 { s.pieces * copies } else { s.hull_count() as u32 },
                    if s.pieces > 0 { "boxes" } else { "hulls" },
                    s.bond_count * copies,
                    if self.bonds { " (breakable)" } else { "" },
                    if self.running { "running" } else { "paused" },
                    if self.rates.summary.is_empty() { String::new() } else { format!("{} · ", self.rates.summary) },
                    l.sim.shots(),
                    c.contact,
                    c.awake,
                    l.sim.steps() as f64 * s.params.dt as f64,
                    if self.status.is_empty() { String::new() } else { format!(" · {}", self.status) },
                )
            }
            None => format!("Fracture lab · Metal · {}", self.status),
        };
        window.set_title(&title);
    }

    fn fire(&mut self, ray: Ray) {
        let Some(l) = &mut self.loaded else { return };
        match l.sim.launch(ray.origin, ray.direction) {
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

    fn shoot(&mut self, at: PhysicalPosition<f64>) {
        let (Some(window), Some(l)) = (&self.window, &self.loaded) else {
            return;
        };
        let size = window.inner_size();
        let aspect = size.width as f64 / size.height.max(1) as f64;
        let ray = l.orbit.aim(
            at.x / size.width as f64,
            at.y / size.height as f64,
            aspect,
            l.scene.projectile_radius,
        );
        self.fire(ray);
    }

    /// One frame: advance physics on a 1/dt accumulator, draw, then read counters.
    fn frame(&mut self) {
        let now = Instant::now();
        let elapsed = self
            .last_frame
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.04));
        self.last_frame = Some(now);
        let hz = self.hz();
        let (Some(layer), Some(l)) = (&self.layer, &mut self.loaded) else {
            return;
        };
        let mut ticks = 0;
        let mut error = None;
        if self.running {
            self.accumulator = (self.accumulator + elapsed).min(4.0 / hz);
            // Bound outstanding work: slow ticks get one tick per frame.
            let budget = ((12.0 / l.sim.timing.wall_ms.max(0.1)).floor() as u32).clamp(1, 4);
            ticks = budget.min((self.accumulator * hz).floor() as u32);
            if ticks > 0 {
                self.accumulator -= ticks as f64 / hz;
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
                    drawable.texture().width() as f64 / drawable.texture().height().max(1) as f64;
                l.renderer.draw(
                    &self.device,
                    &self.queue,
                    drawable,
                    l.sim.body_buffer(),
                    l.sim.joint_buffer(),
                    l.orbit.view_proj(aspect),
                    l.sim.shots(),
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
        if self.rates.frame(hz) {
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
            "{}/16 shots · {} steps · {} contacts · {} candidates · {} awake",
            l.sim.shots(),
            l.sim.steps(),
            c.contact,
            c.candidate,
            c.awake
        )
    }

    fn resize(&self) {
        if let (Some(window), Some(layer)) = (&self.window, &self.layer) {
            let size = window.inner_size();
            layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
        }
    }

    fn key(&mut self, c: &str) {
        match c {
            "r" => {
                if let Some(l) = &mut self.loaded {
                    l.sim.reset();
                }
                self.running = false;
                self.status = "reset".into();
            }
            "b" => self.show_bonds = !self.show_bonds,
            "c" => {
                if let Some(l) = &mut self.loaded {
                    l.renderer.color_mode = (l.renderer.color_mode + 1) % 3;
                }
            }
            "k" => {
                self.bonds = !self.bonds;
                let level = self.loaded.as_ref().and_then(|l| l.level);
                self.load(self.source.clone(), level);
            }
            "f" => {
                if let Some(ray) = self.loaded.as_ref().and_then(|l| l.scene.row_shot) {
                    self.fire(ray);
                }
            }
            "[" => self.select_level(-1),
            "]" => self.select_level(1),
            digit => {
                if let Some(s) = digit
                    .parse::<usize>()
                    .ok()
                    .and_then(|d| SAMPLES.get(d.wrapping_sub(1)))
                {
                    self.load(Source::Sample(s), None);
                }
            }
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
        self.status = format!("loading {}…", self.source.name());
        self.update_title();
        self.load(self.source.clone(), self.initial_level);
        if self.auto_shoot {
            match self.loaded.as_ref().and_then(|l| l.scene.row_shot) {
                Some(ray) => self.fire(ray),
                None => {
                    let size = self.window.as_ref().unwrap().inner_size();
                    self.shoot(PhysicalPosition::new(
                        size.width as f64 / 2.0,
                        size.height as f64 / 2.0,
                    ));
                }
            }
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
            WindowEvent::DroppedFile(path) => self.load(Source::File(path), None),
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
                                (position.x - d.last.x) / scale,
                                (position.y - d.last.y) / scale,
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
                    if self.drag.take().is_some_and(|d| !d.moved) {
                        self.shoot(self.cursor);
                    }
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y as f64 * 100.0,
                    MouseScrollDelta::PixelDelta(p) => -p.y,
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
                    Key::Character(c) => self.key(&c.to_ascii_lowercase()),
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
    let (mut source, mut level, mut auto_shoot, mut bonds) =
        ("house".to_string(), None, false, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" => level = args.next().and_then(|s| s.parse().ok()),
            "--shoot" => auto_shoot = true,
            "--bonds" => bonds = true,
            "-h" | "--help" => {
                let keys: Vec<&str> = SAMPLES.iter().map(|s| s.key).collect();
                println!(
                    "fracture_viewer [{}|path.glb] [--level N] [--shoot] [--bonds]",
                    keys.join("|")
                );
                return Ok(());
            }
            other => source = other.to_string(),
        }
    }
    let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    println!("Metal device: {}", device.name());
    for (i, s) in SAMPLES.iter().enumerate() {
        println!("  {} · {}", i + 1, s.menu);
    }
    println!("  [ ] level · drop a .glb · click shoot · F wrecking ball · Space run · R reset · B bond graph · K breakable bonds · C colours: materials, sleep, bodies");
    let queue = device.new_command_queue();
    let event_loop = EventLoop::new()?;
    let mut app = App {
        device,
        queue,
        source: Source::parse(&source),
        asset: None,
        initial_level: level,
        auto_shoot,
        bonds,
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
