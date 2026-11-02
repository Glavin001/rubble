//! Native macOS host for the world API's range scenes (`engine-world-wasm` `play`): the
//! timber yard (a Shardwright timber-frame house and two plank walls) and Broadside (the
//! galleon stern), stepped in Rust on wgpu exactly as the browser pages step them, drawn
//! with the fracture lab's Metal renderer over the same render scene.
//!
//! ```bash
//! cargo run -p engine-viewer --release --bin world_viewer                 # timber yard
//! cargo run -p engine-viewer --release --bin world_viewer -- timber --raze --level 1
//! cargo run -p engine-viewer --release --bin world_viewer -- broadside
//! ```
//!
//! Click to fire a cannonball where you point · drag to orbit · scroll to zoom ·
//! Space pause · R restart · T free fire / raze · 1 2 fracture level · D director's camera ·
//! Esc quit. `--quit-after S` exits S seconds in; `--volley N` fires N balls at the
//! window's centre once loaded; `--at X,Y` places the window; `--capture DIR` also renders
//! a 1600×1000 PNG into DIR every second (ffmpeg), for checking a run without the screen.

// objc 0.2 `msg_send!` expands a cfg the current compiler does not know.
#![allow(unexpected_cfgs)]

use core_graphics_types::geometry::CGSize;
use engine_fracture::scene::Scene;
use engine_viewer::fracture::{Orbit, Renderer};
use engine_world_wasm::play::{Play, POSE_FLOATS};
use engine_world_wasm::range::{BALL_RADIUS, CRUSH_BUDGET, MAX_BALLS};
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

/// A scene the viewer plays: its `Play` kind, GLB, name and free-play view (eye, target).
#[derive(Clone, Copy)]
struct Kind {
    kind: u8,
    glb: &'static str,
    name: &'static str,
    view: ([f64; 3], [f64; 3]),
    /// Whether the director's camera starts on for (free, raze).
    director: (bool, bool),
}

const TIMBER: Kind = Kind {
    kind: 1,
    glb: "timber-yard.glb",
    name: "Timber yard",
    view: ([-9.0, 4.5, -16.0], [-1.0, 1.8, 1.5]),
    director: (false, true),
};

const BROADSIDE: Kind = Kind {
    kind: 0,
    glb: "galleon-stern.glb",
    name: "Broadside",
    view: ([-7.5, 10.5, -15.0], [9.5, 3.0, 0.0]),
    director: (true, true),
};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/fracture/assets")
}

struct Loaded {
    play: Play,
    renderer: Renderer,
    orbit: Orbit,
    /// The pose buffer the renderer reads (the render scene's bodies, then the extras).
    poses: Buffer,
    joints: Buffer,
    extras_version: u32,
    splinters_uploaded: u32,
}

struct Drag {
    start: PhysicalPosition<f64>,
    last: PhysicalPosition<f64>,
    moved: bool,
}

/// Step and frame times over the last second, for the title.
#[derive(Default)]
struct Rates {
    since: Option<Instant>,
    frames: u32,
    steps: u32,
    step_ms: f64,
    shown: String,
}

struct App {
    device: Device,
    queue: CommandQueue,
    scene: Kind,
    level: u8,
    raze: bool,
    director: bool,
    glb: Vec<u8>,
    window: Option<Window>,
    layer: Option<MetalLayer>,
    loaded: Option<Loaded>,
    running: bool,
    credit: f64,
    last_frame: Option<Instant>,
    cursor: PhysicalPosition<f64>,
    drag: Option<Drag>,
    rates: Rates,
    status: String,
    quit_after: Option<f64>,
    started: Option<Instant>,
    volley: u32,
    at: Option<(f64, f64)>,
    capture: Option<PathBuf>,
    captured: u32,
}

impl App {
    /// Render the current frame offscreen and write it to `--capture`'s directory.
    fn capture(&mut self) -> anyhow::Result<()> {
        let (Some(dir), Some(l)) = (&self.capture, &mut self.loaded) else {
            return Ok(());
        };
        let (w, h) = (1600u64, 1000u64);
        let desc = TextureDescriptor::new();
        desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        desc.set_width(w);
        desc.set_height(h);
        desc.set_storage_mode(MTLStorageMode::Shared);
        desc.set_usage(MTLTextureUsage::RenderTarget);
        let target = self.device.new_texture(&desc);
        let cb = self.queue.new_command_buffer();
        l.renderer.encode(
            &self.device,
            cb,
            &target,
            &l.poses,
            &l.joints,
            l.orbit.view_proj(w as f64 / h as f64),
            0,
            false,
        );
        cb.commit();
        cb.wait_until_completed();
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        target.get_bytes(
            pixels.as_mut_ptr() as *mut _,
            w * 4,
            MTLRegion::new_2d(0, 0, w, h),
            0,
        );
        std::fs::create_dir_all(dir)?;
        let out = dir.join(format!("frame-{:03}.png", self.captured));
        self.captured += 1;
        let mut ffmpeg = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra"])
            .args(["-s", &format!("{w}x{h}"), "-i", "-"])
            .arg(&out)
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        use std::io::Write;
        ffmpeg.stdin.take().unwrap().write_all(&pixels)?;
        ffmpeg.wait()?;
        Ok(())
    }

    fn load(&mut self) {
        let t0 = Instant::now();
        match self.build() {
            Ok(l) => {
                let s = l.play.stats();
                self.status = format!(
                    "{} chunks · {} bonds (loaded in {:.1} s)",
                    s["chunks"],
                    s["bonds"],
                    t0.elapsed().as_secs_f64()
                );
                eprintln!("{}: {}", self.scene.name, self.status);
                self.loaded = Some(l);
            }
            Err(e) => {
                self.status = format!("error: {e:#}");
                eprintln!("{}", self.status);
                self.loaded = None;
            }
        }
        self.credit = 0.0;
        self.last_frame = None;
        self.rates = Rates::default();
        self.update_title();
    }

    fn build(&mut self) -> anyhow::Result<Loaded> {
        self.loaded = None;
        let parsed = engine_fracture::parse_fracture_glb(&self.glb)?;
        let scene: Scene = engine_fracture::build_scene(&parsed, self.level, None)?;
        let play = Play::load(
            rubble::GpuContext::Default,
            &self.glb,
            self.scene.kind,
            self.level,
            self.raze,
        )?;
        let renderer = Renderer::new(&self.device, MTLPixelFormat::BGRA8Unorm, &scene)?;
        let mut orbit = Orbit::new(&scene);
        let (eye, target) = self.scene.view;
        orbit.frame_shot(eye, target);
        // The render scene's bodies, the captain, the balls and the crush budget.
        let bodies = play.render_bodies + 1 + MAX_BALLS + CRUSH_BUDGET as usize;
        let poses = self.device.new_buffer(
            (bodies * POSE_FLOATS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let joints = self
            .device
            .new_buffer(64, MTLResourceOptions::StorageModeShared);
        self.director = if self.raze {
            self.scene.director.1
        } else {
            self.scene.director.0
        };
        Ok(Loaded {
            play,
            renderer,
            orbit,
            poses,
            joints,
            extras_version: u32::MAX,
            splinters_uploaded: 0,
        })
    }

    fn update_title(&self) {
        if let Some(w) = &self.window {
            let take = if self.raze { "raze" } else { "free fire" };
            let paused = if self.running { "" } else { " · paused" };
            w.set_title(&format!(
                "{} · level {} · {take}{paused} · {} · {}",
                self.scene.name, self.level, self.status, self.rates.shown
            ));
        }
    }

    fn shoot(&mut self, at: PhysicalPosition<f64>) {
        let (Some(window), Some(l)) = (&self.window, &mut self.loaded) else {
            return;
        };
        let size = window.inner_size();
        let aspect = size.width as f64 / size.height.max(1) as f64;
        let ray = l.orbit.aim(
            at.x / size.width as f64,
            at.y / size.height as f64,
            aspect,
            BALL_RADIUS as f64,
        );
        // Like the pages: a shot of your own stops the script.
        l.play.scene.set_scripted(false);
        let f = |v: [f64; 3]| v.map(|x| x as f32);
        l.play.shoot_ray(f(ray.origin), f(ray.direction));
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let wall = self
            .last_frame
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.1));
        self.last_frame = Some(now);
        if let Some(q) = self.quit_after {
            if self.started.get_or_insert(now).elapsed().as_secs_f64() > q {
                std::process::exit(0);
            }
        }
        if self.volley > 0 && self.loaded.is_some() {
            let size = self.window.as_ref().unwrap().inner_size();
            for k in 0..self.volley {
                let dx = (k as f64 - 0.5 * (self.volley - 1) as f64) * 0.04 * size.width as f64;
                self.shoot(PhysicalPosition::new(
                    0.5 * size.width as f64 + dx,
                    0.55 * size.height as f64,
                ));
            }
            self.volley = 0;
        }
        let (Some(layer), Some(l)) = (&self.layer, &mut self.loaded) else {
            return;
        };
        let stats = l.play.stats();
        let hz = stats["frameHz"].as_f64().unwrap_or(60.0);
        let mut steps = 0;
        if self.running {
            self.credit += wall * hz * l.play.scene.time_scale() as f64;
            // At most four steps a frame: when the physics is slower than real time, the
            // scene slows down rather than the frame rate collapsing.
            steps = (self.credit.floor() as u32).min(4);
            self.credit = (self.credit - steps as f64).min(1.0);
        }
        if steps > 0 {
            let t = Instant::now();
            if let Err(e) = l.play.frame(steps) {
                self.status = format!("error: {e:#}");
                self.running = false;
            }
            self.rates.step_ms += t.elapsed().as_secs_f64() * 1e3;
            self.rates.steps += steps;
        }
        // Poses between the last two steps, the extras when they changed.
        let poses = l.play.poses(self.credit as f32);
        let n = poses.len().min(l.poses.length() as usize / 4);
        unsafe { std::ptr::copy_nonoverlapping(poses.as_ptr(), l.poses.contents() as *mut f32, n) };
        let version = l.play.extras_version();
        if version != l.extras_version {
            let count = l.play.splinter_count();
            if l.extras_version == u32::MAX || count < l.splinters_uploaded {
                l.renderer
                    .write_dynamic(&self.device, 0, &l.play.extra_vertices());
            } else {
                l.renderer.write_dynamic(
                    &self.device,
                    Play::splinter_vertex_offset(l.splinters_uploaded) as u64,
                    &l.play.splinter_vertices(l.splinters_uploaded),
                );
            }
            l.extras_version = version;
            l.splinters_uploaded = count;
        }
        if self.director {
            if let Some((eye, target)) = l.play.scene.camera(wall as f32) {
                let f = |v: [f32; 3]| v.map(|x| x as f64);
                l.orbit.frame_shot(f(eye), f(target));
            }
        }
        objc::rc::autoreleasepool(|| {
            if let Some(drawable) = layer.next_drawable() {
                let aspect =
                    drawable.texture().width() as f64 / drawable.texture().height().max(1) as f64;
                l.renderer.draw(
                    &self.device,
                    &self.queue,
                    drawable,
                    &l.poses,
                    &l.joints,
                    l.orbit.view_proj(aspect),
                    0,
                    false,
                );
            }
        });
        self.rates.frames += 1;
        let since = *self.rates.since.get_or_insert(now);
        let span = (now - since).as_secs_f64();
        if span >= 1.0 {
            let r = &self.rates;
            self.rates.shown = format!(
                "{:.0} fps · {:.1} ms/step · t {:.1} s · shots {} · broken {} · crushed {} · intact {:.0}%",
                r.frames as f64 / span,
                r.step_ms / r.steps.max(1) as f64,
                stats["simSeconds"].as_f64().unwrap_or(0.0),
                stats["shots"],
                stats["brokenBonds"],
                stats["crushed"],
                100.0 * stats["intact"].as_f64().unwrap_or(1.0),
            );
            eprintln!("{}", self.rates.shown);
            if let Err(e) = self.capture() {
                eprintln!("capture: {e:#}");
            }
            self.rates = Rates {
                shown: std::mem::take(&mut self.rates.shown),
                ..Rates::default()
            };
            self.update_title();
        }
    }

    fn resize(&self) {
        if let (Some(w), Some(layer)) = (&self.window, &self.layer) {
            let size = w.inner_size();
            layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
        }
    }

    fn key(&mut self, c: &str) {
        match c {
            "r" => self.load(),
            "t" => {
                self.raze = !self.raze;
                self.load();
            }
            "1" | "2" => {
                self.level = c.parse().unwrap();
                self.load();
            }
            "d" => {
                self.director = !self.director;
                if !self.director {
                    if let Some(l) = &mut self.loaded {
                        let (eye, target) = self.scene.view;
                        l.orbit.frame_shot(eye, target);
                    }
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attrs = Window::default_attributes()
            .with_title(self.scene.name)
            .with_inner_size(LogicalSize::new(1400.0, 900.0));
        if let Some((x, y)) = self.at {
            attrs = attrs.with_position(winit::dpi::LogicalPosition::new(x, y));
        }
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
        self.status = "loading…".into();
        self.update_title();
        self.load();
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
                        // Taking the camera turns the director off.
                        self.director = false;
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
                self.director = false;
                if let Some(l) = &mut self.loaded {
                    l.orbit.zoom(dy);
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => {
                        self.running = !self.running;
                        self.last_frame = None;
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
    let mut scene = TIMBER;
    let mut level = 2;
    let mut raze = false;
    let mut quit_after = None;
    let mut volley = 0;
    let mut at = None;
    let mut capture = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "timber" => scene = TIMBER,
            "broadside" => scene = BROADSIDE,
            "--raze" => raze = true,
            "--level" => level = args.next().expect("--level N").parse()?,
            "--quit-after" => quit_after = Some(args.next().expect("--quit-after S").parse()?),
            "--volley" => volley = args.next().expect("--volley N").parse()?,
            "--capture" => capture = Some(PathBuf::from(args.next().expect("--capture DIR"))),
            "--at" => {
                let v = args.next().expect("--at X,Y");
                let (x, y) = v.split_once(',').expect("--at X,Y");
                at = Some((x.parse()?, y.parse()?));
            }
            other => anyhow::bail!("unknown argument {other} (timber | broadside, --raze, --level N, --quit-after S, --volley N, --at X,Y, --capture DIR)"),
        }
    }
    let path = assets_dir().join(scene.glb);
    let glb = std::fs::read(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    anyhow::ensure!(
        glb.starts_with(b"glTF"),
        "{} is not a GLB (a Git LFS pointer? run git lfs pull)",
        path.display()
    );
    let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    let queue = device.new_command_queue();
    let mut app = App {
        device,
        queue,
        scene,
        level,
        raze,
        director: false,
        glb,
        window: None,
        layer: None,
        loaded: None,
        running: true,
        credit: 0.0,
        last_frame: None,
        cursor: PhysicalPosition::new(0.0, 0.0),
        drag: None,
        rates: Rates::default(),
        status: String::new(),
        quit_after,
        started: None,
        volley,
        at,
        capture,
        captured: 0,
    };
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}
