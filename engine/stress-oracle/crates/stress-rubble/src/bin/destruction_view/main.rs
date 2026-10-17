//! Interactive, real-time destruction on rubble's production pipeline: stress-gpu's
//! engine-coupled world (`GpuWorld::with_engine`) with rubble's AVBD engine
//! (`stress_rubble::RubbleEngine`) on one shared wgpu device, impact islands routed
//! before the step and no frame rewinds (`redo_fractured_frames = false`), exactly as
//! `stress_bench --solver rubble-gpu` runs it. Nothing is drawn from the host: chunks,
//! clusters and impactors come straight from the stress solver's GPU buffers, every chunk
//! drawn as its convex hull (its box for box chunks) and coloured by the largest
//! utilization of its unbroken bonds (blue 0, red at failure; grey: no bond left).
//!
//!   cargo run --release -p stress-rubble --bin destruction-view -- --scene wall_ram
//!
//! Scenes (keys 1-9): wall_ram, b5_v10, b5_v20, b5_v40, house_ram, house_idle, town,
//! town_idle, veneer (or a scene JSON path). A scene's scripted rams are held until the
//! first click, which launches them as the scene describes (so the outcome is the
//! benchmark's); later clicks fire steel spheres along the view ray. `--no-hold` starts
//! the rams at t = 0, as the benchmark does.
//!
//! Controls: click fire, left drag orbit, right or shift drag pan, wheel zoom, Space
//! pause, . (or N) one frame while paused, R reset, 1-9 scene, [ ] projectile speed,
//! - = projectile size, B chunk hulls / oriented boxes, Esc quit.
//!
//! Frames are paced to simulated time: one `frame_dt` of simulation per `frame_dt` of
//! wall clock when the pipeline keeps up, one frame per redraw (slow motion, shown as the
//! real-time factor) when it does not. Nothing trades accuracy for frame rate.
//!
//! Batch options: `--frames N` exits after N simulated frames; `--report` prints one
//! JSON line per frame (timing split, fragments, broken bonds) and a summary;
//! `--click F:X,Y` clicks at frame F at window fraction (X, Y) (repeatable);
//! `--record OUT.mp4` steps one simulated frame per drawn frame and pipes the window to
//! ffmpeg (60 fps of simulated time); `--size WxH` the window's logical size;
//! `--view YAW,PITCH` the initial camera; `--spin DEG` turns the camera DEG degrees per
//! simulated second; `--paused` start paused.

// stress-viz's 5x7 font (stress-viz is a binary crate: its module, included by path).
#[path = "../../../../stress-viz/src/font.rs"]
mod font;
mod render;
mod sim;

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::Vec3 as V3;
use stress_gpu::gpu::Gpu;
use stress_ref::math::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use render::{Orbit, Renderer};
use sim::{Sim, MAX_SHOTS, SCENES};

struct Options {
    scene: String,
    hold: bool,
    frames: Option<u64>,
    report: bool,
    clicks: Vec<(u64, f32, f32)>,
    record: Option<String>,
    size: (f64, f64),
    view: Option<(f32, f32)>,
    spin: f32,
    paused: bool,
}

struct Drag {
    start: PhysicalPosition<f64>,
    last: PhysicalPosition<f64>,
    button: MouseButton,
    moved: bool,
}

/// Wall-clock statistics over the last second.
#[derive(Default)]
struct Rates {
    window_start: Option<Instant>,
    draws: u32,
    sim_time: f64,
    frame_ms: Vec<f64>,
    fps: f64,
    real_time: f64,
    avg_ms: f64,
    peak_ms: f64,
}

struct App {
    opts: Options,
    gpu: Option<Gpu>,
    gfx: Option<Renderer>,
    sim: Option<Sim>,
    orbit: Option<Orbit>,
    drag: Option<Drag>,
    cursor: PhysicalPosition<f64>,
    modifiers: ModifiersState,
    paused: bool,
    step_once: bool,
    acc: f64,
    last_tick: Option<Instant>,
    message: Option<(String, Instant)>,
    rates: Rates,
    error: Option<String>,
    /// The run's last frame failed: nothing steps until R.
    failed: bool,
    report: Vec<serde_json::Value>,
}

fn main() {
    let mut opts = Options { scene: "wall_ram".into(), hold: true, frames: None, report: false, clicks: Vec::new(), record: None, size: (1280.0, 760.0), view: None, spin: 0.0, paused: false };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--scene" => opts.scene = value(),
            "--no-hold" => opts.hold = false,
            "--frames" => opts.frames = Some(value().parse().expect("--frames N")),
            "--report" => opts.report = true,
            "--click" => {
                let v = value();
                let (f, xy) = v.split_once(':').expect("--click FRAME:X,Y");
                let (x, y) = xy.split_once(',').expect("--click FRAME:X,Y");
                opts.clicks.push((f.parse().expect("frame"), x.parse().expect("x"), y.parse().expect("y")));
            }
            "--record" => opts.record = Some(value()),
            "--size" => {
                let v = value();
                let (w, h) = v.split_once('x').expect("--size WxH");
                opts.size = (w.parse().expect("width"), h.parse().expect("height"));
            }
            "--view" => {
                let v = value();
                let (y, p) = v.split_once(',').expect("--view YAW,PITCH");
                opts.view = Some((y.parse().expect("yaw"), p.parse().expect("pitch")));
            }
            "--spin" => opts.spin = value().parse().expect("--spin DEG_PER_S"),
            "--paused" => opts.paused = true,
            "--list" => {
                for (i, (name, what)) in SCENES.iter().enumerate() {
                    println!("{}  {name:<11} {what}", i + 1);
                }
                return;
            }
            other => panic!("unknown argument {other} (see the module docs)"),
        }
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let paused = opts.paused;
    let mut app = App {
        opts,
        gpu: None,
        gfx: None,
        sim: None,
        orbit: None,
        drag: None,
        cursor: PhysicalPosition::new(0.0, 0.0),
        modifiers: ModifiersState::empty(),
        paused,
        step_once: false,
        acc: 0.0,
        last_tick: None,
        message: None,
        rates: Rates::default(),
        error: None,
        failed: false,
        report: Vec::new(),
    };
    event_loop.run_app(&mut app).expect("run");
    if let Some(e) = app.error {
        eprintln!("destruction-view: {e}");
        std::process::exit(1);
    }
}

impl App {
    fn say(&mut self, m: impl Into<String>) {
        let m = m.into();
        eprintln!("{m}");
        self.message = Some((m, Instant::now()));
    }

    /// (Re)start a scene: a fresh engine and world on the shared device.
    fn load(&mut self, name: &str) {
        let gpu = self.gpu.as_ref().expect("device");
        let t = Instant::now();
        match Sim::new(gpu, name, self.opts.hold) {
            Ok(sim) => {
                let (lo, hi) = sim.bounds();
                let mut orbit = Orbit::fit(V3::new(lo.x as f32, lo.y as f32, lo.z as f32), V3::new(hi.x as f32, hi.y as f32, hi.z as f32));
                if let Some((yaw, pitch)) = self.opts.view {
                    (orbit.yaw_deg, orbit.pitch_deg) = (yaw, pitch);
                }
                // Keep the user's view across resets of the same scene.
                if self.sim.as_ref().is_none_or(|s| s.name != name) || self.orbit.is_none() {
                    self.orbit = Some(orbit);
                }
                let chunks = sim.world.solver.chunk_order().len();
                let info = format!(
                    "{name}: {chunks} chunks, {} bonds, frame {:.1} ms, loaded in {:.2} s{}",
                    sim.world.solver.mirror.bonds.iter().map(|b| b.len()).sum::<usize>(),
                    sim.frame_dt() * 1e3,
                    t.elapsed().as_secs_f64(),
                    if sim.armed.is_empty() { String::new() } else { format!("; {} ram(s) held: click launches", sim.armed.len()) }
                );
                self.sim = Some(sim);
                self.failed = false;
                self.acc = 0.0;
                self.rates = Rates::default();
                self.say(info);
            }
            Err(e) => self.say(format!("{name}: {e}")),
        }
        if let Some(g) = &self.gfx {
            g.window.set_title(&format!("destruction-view: {name} (rubble AVBD + stress-gpu)"));
        }
    }

    fn step(&mut self) {
        if self.failed {
            return;
        }
        let (Some(sim), Some(gpu)) = (self.sim.as_mut(), self.gpu.as_ref()) else { return };
        // A panic inside the pipeline (the engine's or the solver's asserts) must not
        // take the window down: the run stops there and R starts it again.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.step(gpu))).unwrap_or_else(|p| {
            Err(p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into()))
        });
        if let Err(e) = result {
            self.failed = true;
            self.paused = true;
            let m = format!("frame {} failed: {e} (R resets)", sim.frames + 1);
            self.say(m);
            return;
        }
        let t = &sim.last;
        self.rates.frame_ms.push(t.total);
        self.rates.sim_time += sim.frame_dt();
        if self.opts.report {
            let line = serde_json::json!({"scene": sim.name, "frame": sim.frames, "t": sim.time(), "frame_ms": t.total, "stress_ms": t.stress, "stress_gpu_ms": t.stress_gpu,
                "engine_ms": t.engine, "host_ms": t.host, "fragments": sim.fragments(), "broken_bonds": sim.broken_bonds(), "impactors": sim.impactor_count(), "ram": sim.ram_speed(), "max_speed": sim.max_speed()});
            println!("{line}");
            self.report.push(line);
        }
    }

    /// A click at window pixel `p`: launches the held rams, or fires a projectile.
    fn click(&mut self, p: PhysicalPosition<f64>) {
        let (Some(sim), Some(gpu), Some(g), Some(orbit)) = (self.sim.as_mut(), self.gpu.as_ref(), self.gfx.as_ref(), self.orbit.as_ref()) else { return };
        match sim.launch_armed(gpu) {
            Ok(true) => {
                let m = format!("t = {:.3} s: launched the scene's ram(s)", sim.time());
                self.say(m);
                return;
            }
            Ok(false) => {}
            Err(e) => return self.say(format!("launch failed: {e}")),
        }
        let (w, h) = (g.config.width as f32, g.config.height as f32);
        let (eye, dir) = orbit.ray(p.x as f32, p.y as f32, w, h);
        let (eye, dir) = (Vec3::new(eye.x as f64, eye.y as f64, eye.z as f64), Vec3::new(dir.x as f64, dir.y as f64, dir.z as f64));
        // Start just outside the scene's bounds along the ray, so the shot arrives at once.
        let (lo, hi) = sim.bounds();
        let margin = 2.0 * sim.shot.radius + 0.02 * (hi - lo).norm();
        let (lo, hi) = (lo - Vec3::splat(margin), hi + Vec3::splat(margin));
        let start = match sim::ray_box(eye, dir, lo, hi) {
            Some(t) if t > 0.0 => eye + dir * t,
            Some(_) => eye + dir * (2.0 * sim.shot.radius),
            None => eye + dir * (0.5 * ((lo + hi) * 0.5 - eye).norm()),
        };
        match sim.fire(gpu, start, dir) {
            Ok(()) => {
                let m = format!("t = {:.3} s: shot {} ({:.2} m steel sphere at {:.0} m/s)", sim.time(), sim.shots, 2.0 * sim.shot.radius, sim.shot.speed);
                self.say(m)
            }
            Err(e) => self.say(e),
        }
    }

    fn key(&mut self, event: &KeyEvent, el: &ActiveEventLoop) {
        if event.state != ElementState::Pressed {
            return;
        }
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => el.exit(),
            Key::Named(NamedKey::Space) => {
                self.paused = !self.paused;
                self.acc = 0.0;
            }
            Key::Character(c) => match c.to_lowercase().as_str() {
                "r" => {
                    let name = self.sim.as_ref().map_or(self.opts.scene.clone(), |s| s.name.clone());
                    self.load(&name);
                }
                "." | "n" => {
                    self.paused = true;
                    self.step_once = true;
                }
                "b" => {
                    if let Some(g) = self.gfx.as_mut() {
                        g.boxes = !g.boxes;
                    }
                }
                "[" | "]" => {
                    if let Some(sim) = self.sim.as_mut() {
                        sim.shot.speed *= if c.as_str() == "]" { 1.25 } else { 0.8 };
                        let m = format!("projectile speed {:.1} m/s", sim.shot.speed);
                        self.say(m);
                    }
                }
                "-" | "=" | "+" => {
                    if let Some(sim) = self.sim.as_mut() {
                        sim.shot.radius *= if c.as_str() == "-" { 0.8 } else { 1.25 };
                        let m = format!("projectile diameter {:.3} m", 2.0 * sim.shot.radius);
                        self.say(m);
                    }
                }
                d if d.len() == 1 && d.chars().all(|ch| ch.is_ascii_digit()) => {
                    let k = d.parse::<usize>().unwrap();
                    if (1..=SCENES.len()).contains(&k) {
                        self.load(SCENES[k - 1].0);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn mouse_move(&mut self, p: PhysicalPosition<f64>) {
        self.cursor = p;
        let Some(d) = &mut self.drag else { return };
        let (dx, dy) = ((p.x - d.last.x) as f32, (p.y - d.last.y) as f32);
        d.last = p;
        if !d.moved && ((p.x - d.start.x).abs() + (p.y - d.start.y).abs()) > 4.0 {
            d.moved = true;
        }
        if !d.moved {
            return;
        }
        let pan = d.button == MouseButton::Right || self.modifiers.shift_key();
        let height = self.gfx.as_ref().map_or(1.0, |g| g.config.height as f32);
        let Some(o) = self.orbit.as_mut() else { return };
        if pan {
            o.pan(dx, dy, height);
        } else {
            o.orbit(dx, dy);
        }
    }

    fn hud(&self) -> Vec<(String, [u8; 3])> {
        const WHITE: [u8; 3] = [235, 235, 235];
        const DIM: [u8; 3] = [160, 165, 175];
        const WARN: [u8; 3] = [255, 200, 90];
        let mut out = Vec::new();
        let Some(sim) = &self.sim else {
            out.push(("no scene loaded (1-9 picks one)".to_string(), WARN));
            if let Some((m, _)) = &self.message {
                out.push((m.clone(), WARN));
            }
            return out;
        };
        let what = SCENES.iter().find(|(n, _)| *n == sim.name).map_or("", |(_, w)| w);
        out.push((format!("{} - {what}", sim.name), WHITE));
        let r = &self.rates;
        let state = if self.paused { "  PAUSED" } else { "" };
        out.push((
            format!("t {:.3} s  frame {}  {:.2}x real time  {:.0} fps drawn{state}", sim.time(), sim.frames, r.real_time, r.fps),
            if r.real_time < 0.95 && !self.paused && sim.frames > 1 { WARN } else { WHITE },
        ));
        let t = &sim.last;
        out.push((
            format!("frame {:.1} ms = stress {:.1} (gpu {:.1}) + engine {:.1} + host {:.1}   1 s avg {:.1}, peak {:.1}", t.total, t.stress, t.stress_gpu, t.engine, t.host, r.avg_ms, r.peak_ms),
            WHITE,
        ));
        let armed = if sim.armed.is_empty() { String::new() } else { "  ram held: click launches".into() };
        out.push((
            format!(
                "fragments {}  broken bonds {}  chunks {}  impactors {}  shots {}/{MAX_SHOTS} ({:.2} m @ {:.0} m/s){armed}",
                sim.fragments(),
                sim.broken_bonds(),
                sim.world.solver.chunk_order().len(),
                sim.impactor_count(),
                sim.shots,
                2.0 * sim.shot.radius,
                sim.shot.speed
            ),
            WHITE,
        ));
        out.push(("colour: bond utilization, blue 0 -> red 1 (failure); grey: no bond left".into(), DIM));
        out.push(("click fire  drag orbit  right/shift-drag pan  wheel zoom  space pause  . step  R reset  1-9 scene  [ ] speed  - = size  B hulls/boxes".into(), DIM));
        if let Some((m, at)) = &self.message {
            if at.elapsed() < Duration::from_secs(4) {
                out.push((m.clone(), WARN));
            }
        }
        out
    }

    fn update_rates(&mut self) {
        let now = Instant::now();
        let r = &mut self.rates;
        r.draws += 1;
        let start = *r.window_start.get_or_insert(now);
        let span = now.duration_since(start).as_secs_f64();
        if span >= 1.0 {
            r.fps = r.draws as f64 / span;
            r.real_time = r.sim_time / span;
            r.avg_ms = if r.frame_ms.is_empty() { 0.0 } else { r.frame_ms.iter().sum::<f64>() / r.frame_ms.len() as f64 };
            r.peak_ms = r.frame_ms.iter().copied().fold(0.0, f64::max);
            r.draws = 0;
            r.sim_time = 0.0;
            r.frame_ms.clear();
            r.window_start = Some(now);
        }
    }

    fn summary(&self) {
        let Some(sim) = &self.sim else { return };
        let ms: Vec<f64> = self.report.iter().filter_map(|l| l["frame_ms"].as_f64()).collect();
        let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        let pct = |q: f64| sorted.get(((sorted.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
        println!(
            "{}",
            serde_json::json!({"summary": sim.name, "frames": sim.frames, "t": sim.time(), "ram": sim.ram_speed(), "fragments": sim.fragments(), "broken_bonds": sim.broken_bonds(),
                "frame_ms_mean": mean, "frame_ms_p50": pct(0.5), "frame_ms_p95": pct(0.95), "frame_ms_max": pct(1.0), "shots": sim.shots})
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("destruction-view")
            .with_inner_size(winit::dpi::LogicalSize::new(self.opts.size.0, self.opts.size.1))
            .with_resizable(self.opts.record.is_none());
        let window = Arc::new(el.create_window(attrs).expect("window"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = match pollster::block_on(Gpu::with_instance(&instance, Some(&surface))) {
            Ok(g) => g,
            Err(e) => {
                self.error = Some(e);
                return el.exit();
            }
        };
        eprintln!("adapter {} ({:?}), shaders {:?}", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.shader_path);
        match Renderer::new(window, &instance, surface, &gpu, self.opts.record.as_deref()) {
            Ok(r) => self.gfx = Some(r),
            Err(e) => {
                self.error = Some(e);
                return el.exit();
            }
        }
        self.gpu = Some(gpu);
        let scene = self.opts.scene.clone();
        self.load(&scene);
        if self.sim.is_none() && (self.opts.frames.is_some() || self.opts.record.is_some()) {
            self.error = self.message.as_ref().map(|(m, _)| m.clone());
            el.exit();
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::Occluded(hidden) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.occluded = hidden;
                }
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => self.key(&event, el),
            WindowEvent::CursorMoved { position, .. } => self.mouse_move(position),
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => self.drag = Some(Drag { start: self.cursor, last: self.cursor, button, moved: false }),
                ElementState::Released => {
                    if let Some(d) = self.drag.take() {
                        if !d.moved && d.button == MouseButton::Left && !self.modifiers.shift_key() {
                            self.click(self.cursor);
                        }
                    }
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if let Some(o) = self.orbit.as_mut() {
                    o.zoom(steps);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
            return;
        }
        // Scripted clicks due at this frame.
        let frame = self.sim.as_ref().map_or(0, |s| s.frames);
        let due: Vec<(f32, f32)> = self.opts.clicks.iter().filter(|c| c.0 == frame).map(|c| (c.1, c.2)).collect();
        self.opts.clicks.retain(|c| c.0 != frame);
        for (x, y) in due {
            let g = self.gfx.as_ref().unwrap();
            self.click(PhysicalPosition::new(x as f64 * g.config.width as f64, y as f64 * g.config.height as f64));
        }

        let now = Instant::now();
        let wall = self.last_tick.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
        self.last_tick = Some(now);
        let frame_dt = self.sim.as_ref().map_or(1.0 / 60.0, |s| s.frame_dt());
        let lockstep = self.opts.record.is_some() || self.opts.frames.is_some();
        let mut run = false;
        if self.step_once {
            self.step_once = false;
            run = true;
        } else if !self.paused {
            if lockstep {
                run = true;
            } else {
                // Paced to simulated time; at most one frame per redraw (slow motion when
                // the pipeline cannot keep up).
                self.acc = (self.acc + wall).min(0.25);
                if self.acc >= frame_dt {
                    self.acc = (self.acc - frame_dt).min(frame_dt);
                    run = true;
                }
            }
        }
        if run {
            self.step();
            // `--spin`: the camera turns with simulated time.
            if let Some(o) = self.orbit.as_mut() {
                o.yaw_deg += self.opts.spin * frame_dt as f32;
            }
        }
        self.update_rates();
        let hud = self.hud();
        let (Some(g), Some(sim), Some(orbit)) = (self.gfx.as_mut(), self.sim.as_ref(), self.orbit.as_ref()) else { return };
        if let Err(e) = g.draw(sim, orbit, &hud) {
            self.error = Some(e);
            el.exit();
        }
        if self.failed && (self.opts.frames.is_some() || self.opts.record.is_some()) {
            self.error = self.message.as_ref().map(|(m, _)| m.clone());
            el.exit();
        }
        if self.opts.frames.is_some_and(|n| sim.frames >= n) {
            if self.opts.report {
                self.summary();
            }
            el.exit();
        }
    }

    fn exiting(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = self.gfx.as_mut() {
            g.finish();
        }
    }
}
