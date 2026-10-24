//! The GPU stress solver coupled to rubble, in the browser: the Fracture lab's
//! "stress solver" bond mode (`engine-webgpu/fracture`).
//!
//! The same Rust as the native runs (`stress-rubble` `examples/fracture_glb.rs`): the
//! fracture GLB's level through its Shardwright stress export and stress-ref's import, the
//! `blast_parity` profile on the resident frame (rubble's step, the contact ingest and the
//! quasi-static stress cascade with its GPU splits, all on one WebGPU device), the lab's
//! orbit camera and shot rule (`engine-fracture` `view.rs`, `plan::shot_velocity`), and the
//! world drawn by `stress_rubble::render` into the page's canvas. JavaScript only loads the
//! module, forwards input and calls [`stress_frame`] once per animation frame.
//!
//! **Waiting on the GPU.** The solver reads the GPU synchronously (one status read per
//! resident frame, more while loading). In the browser those reads suspend the wasm stack
//! to the event loop (`engine-gpu-wait`, JavaScript Promise Integration), so the exports
//! that can reach a read are plain `extern "C"` functions the page calls through
//! `WebAssembly.promising` ([`stress_load`], [`stress_frame`]), or, without JSPI, through the
//! Asyncify build's driver ([`stress_asyncify_buffer`] is its save area); everything else is a
//! wasm-bindgen function that never waits. The camera, pending shots and stats live apart
//! from the world, so input arriving while a frame is suspended never meets a borrowed
//! world.

use std::cell::{Cell, RefCell};

use engine_fracture::view::{Orbit, ViewMode};
use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};
use stress_rubble::render::{ColorMode, Renderer};
use wasm_bindgen::prelude::*;

mod drive;

/// Shots per load, as the lab's other modes (`Planner::launch`).
const SHOTS: usize = 16;

struct Device {
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The canvas texture's sRGB view format (the renderer's).
    view_format: wgpu::TextureFormat,
}

struct Loaded {
    world: GpuWorld,
    renderer: Renderer,
    /// Real time not yet simulated (s).
    behind: f64,
    /// The scene's chunks, hulls and bonds (fixed for a load: counted once).
    totals: (usize, usize, usize),
}

/// The threshold profile's GPU-resident world and its renderer.
struct TLoaded {
    world: stress_rubble::threshold::ThresholdWorld,
    renderer: Renderer,
    behind: f64,
    totals: (usize, usize, usize),
}

/// What the lab's shot rule needs of the scene (asset frame, +Y up).
#[derive(Clone, Copy)]
struct Shooting {
    center: [f64; 3],
    extent: f64,
    radius: f64,
    mass: f64,
    speed: Option<f64>,
    arc: bool,
    /// The lab's translation of the asset (`engine_fracture::Scene::placement`): a point of
    /// the lab's frame is the asset's plus this.
    placement: [f64; 3],
    /// The ball rolls in on the ground when fired along the line of attack (`stress_fire`).
    rolling: bool,
}

thread_local! {
    static DEVICE: RefCell<Option<Device>> = const { RefCell::new(None) };
    static WORLD: RefCell<Option<Loaded>> = const { RefCell::new(None) };
    /// The threshold profile's GPU-resident world (`stress_rubble::threshold`), when the
    /// load chose it (`stress_set_threshold_gpu`).
    static TWORLD: RefCell<Option<TLoaded>> = const { RefCell::new(None) };
    /// A pack with driven vehicles (`defaults.drives`) in rubble's world API (`drive`).
    static DWORLD: RefCell<Option<drive::DLoaded>> = const { RefCell::new(None) };
    /// Whether the loaded scene is `DWORLD`'s (read by input that must not borrow it).
    static DRIVING: Cell<bool> = const { Cell::new(false) };
    /// The threshold profile on the GPU-resident world (default) or the resident GpuWorld.
    static THRESHOLD_GPU: Cell<bool> = const { Cell::new(true) };
    /// What the next load builds (`stress_set_glb`, `stress_set_cuboid`, `stress_set_pack`).
    static SOURCE: RefCell<Source> = const { RefCell::new(Source::Glb(Vec::new())) };
    /// What the last load could not take from its input as given (`stress_stats`).
    static WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static CAMERA: RefCell<Option<Orbit>> = const { RefCell::new(None) };
    /// The camera's view mode, kept across loads (`stress_set_view_mode`).
    static VIEW_MODE: Cell<ViewMode> = const { Cell::new(ViewMode::Orbit) };
    static SHOOTING: Cell<Option<Shooting>> = const { Cell::new(None) };
    static PENDING: RefCell<Vec<ImpactorDesc>> = const { RefCell::new(Vec::new()) };
    static FIRED: Cell<usize> = const { Cell::new(0) };
    /// The shots' mass as a multiple of the lab's default ball (`stress_set_shot_mass_scale`).
    static MASS_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The shots' speed as a multiple of the lab's (`stress_set_shot_speed_scale`).
    static SPEED_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The next load's ball radius as a multiple of the lab's (`stress_set_shot_radius_scale`).
    static RADIUS_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The ball preset of the next load (`stress_set_shot_preset`: an index into
    /// `stress_ref::wrecking::BALLS`), or the scene's own ball.
    static PRESET: Cell<Option<usize>> = const { Cell::new(None) };
    /// The loaded pack's line of attack (`defaults.wreckingYard`, the stress frame), for
    /// `stress_fire`.
    static AIM: Cell<Option<stress_ref::wrecking::Aim>> = const { Cell::new(None) };
    /// The loaded pack's demolition charges (`defaults.charges`) and the ones set off.
    static DEMOLITION: RefCell<stress_rubble::demolition::Demolition> = RefCell::new(Default::default());
    /// Charges asked for (`stress_detonate`) and not yet set off: the next frame sets them
    /// off at its world's frame (the world may be held by a suspended frame now).
    static DETONATE: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// The loaded pack's launched bonded bodies (`defaults.launches`) and the ones asked for.
    static LAUNCHER: RefCell<stress_rubble::launch::Launcher> = RefCell::new(Default::default());
    /// Launches asked for (`stress_launch`) and not yet armed: the next frame arms them at
    /// its world's frame.
    static LAUNCH: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// The bonds' strengths as a multiple of the asset's (`stress_set_bond_strength_scale`).
    static STRENGTH_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The view (`stress_set_view`): bond graph, colour mode.
    static VIEW: Cell<(bool, ColorMode)> = const { Cell::new((false, ColorMode::Material)) };
    /// The capability profile of the next `stress_load` (`stress_set_profile`).
    static PROFILE: Cell<Profile> = const { Cell::new(Profile::BlastParity) };
    /// The physics frame rate of the next `stress_load` (`stress_set_physics_hz`); 0 keeps
    /// the scene's own.
    static PHYSICS_HZ: Cell<f64> = const { Cell::new(0.0) };
    static RUNNING: Cell<bool> = const { Cell::new(false) };
    static SIZE: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
    // Dust and chips from the fragments (render only; `stress_set_fx`).
    static FX: Cell<bool> = const { Cell::new(true) };
    static STATS: RefCell<String> = RefCell::new("{}".into());
    /// The last frame's step, state copy and draw (ms).
    static PHASES: Cell<[f64; 3]> = const { Cell::new([0.0; 3]) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// The input of the next [`stress_load`].
enum Source {
    /// A fracture GLB, at the load's level.
    Glb(Vec<u8>),
    /// The lab's cuboid houses in a sample's layout.
    Cuboid(Option<engine_fracture::Layout>),
    /// A scene pack's JSON (vibe-city / blast-stress-solver).
    Pack(Vec<u8>),
}

/// Asset frame (+Y up) to the stress scene's (+Z up), as `stress_ref::shardwright` imports.
fn zup(v: [f64; 3]) -> [f64; 3] {
    [v[0], -v[2], v[1]]
}

/// The GPU device on `canvas` (WebGPU), its surface configured at the canvas size.
#[wasm_bindgen]
pub async fn stress_init(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&info.to_string().into());
        // Kept for the page (`stress_panic`): a panic traps the export ("unreachable"), and a
        // phone shows no console.
        if let Ok(mut last) = PANIC.try_lock() {
            *last = info.to_string();
        }
        // The JavaScript stack: which export reached the panic.
        web_sys::console::trace_0();
    }));
    let (width, height) = (canvas.width().max(1), canvas.height().max(1));
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(|e| JsValue::from_str(&format!("canvas surface: {e}")))?;
    let gpu = Gpu::with_instance(&instance, Some(&surface)).await.map_err(|e| JsValue::from_str(&e))?;
    let caps = surface.get_capabilities(gpu.surface_adapter());
    let format = caps.formats.first().copied().ok_or_else(|| JsValue::from_str("the canvas offers no format"))?.remove_srgb_suffix();
    let view_format = format.add_srgb_suffix();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![view_format],
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    surface.configure(&gpu.device, &config);
    SIZE.set((width, height));
    DEVICE.with_borrow_mut(|d| *d = Some(Device { gpu, surface, config, view_format }));
    Ok(())
}

/// The fracture GLB the next [`stress_load`] builds.
#[wasm_bindgen]
pub fn stress_set_glb(bytes: Vec<u8>) {
    SOURCE.set(Source::Glb(bytes));
}

/// The scene pack (its JSON bytes) the next [`stress_load`] builds: the lab's scene of
/// it (`engine_fracture::pack`) for the camera and shots, and stress-ref's import of it
/// under the load's profile (`Profile::import_options`), as the native bench imports
/// packs. The pack's supports (nodes of mass 0) are the fixed chunks, and the bonds to them
/// break like any bond.
#[wasm_bindgen]
pub fn stress_set_pack(bytes: Vec<u8>) {
    SOURCE.set(Source::Pack(bytes));
}

/// The lab's procedural cuboid houses for the next [`stress_load`], laid out as the
/// sample's `layout` (its JSON: `null`, `{columns, rows}` or `{suburb: true, ...}`).
#[wasm_bindgen]
pub fn stress_set_cuboid(layout: String) -> Result<(), JsValue> {
    let layout: Option<engine_fracture::Layout> = serde_json::from_str(&layout).map_err(|e| JsValue::from_str(&format!("layout: {e}")))?;
    SOURCE.set(Source::Cuboid(layout));
    Ok(())
}

/// The message of the last failed [`stress_load`] or [`stress_frame`].
/// The last panic's message (empty if none): what an export that trapped hit.
static PANIC: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

#[wasm_bindgen]
pub fn stress_panic() -> String {
    PANIC.try_lock().map(|p| p.clone()).unwrap_or_default()
}

#[wasm_bindgen]
pub fn stress_error() -> String {
    ERROR.with_borrow(|e| e.clone())
}

/// The last frame's numbers, as JSON: frames, sim seconds, step ms, fragments, broken
/// bonds, shots, real-time factor.
#[wasm_bindgen]
pub fn stress_stats() -> String {
    STATS.with_borrow(|s| s.clone())
}

/// The last frame's step, view (`GpuWorld::view_with`: the frame's splits followed, the
/// engine's poses read) and draw, in ms.
#[wasm_bindgen]
pub fn stress_phases() -> Vec<f64> {
    PHASES.get().to_vec()
}

/// The next threshold load on the GPU-resident world (`stress_rubble::threshold`: the
/// splits, loads and drawing on the GPU, nothing read back), or on the resident GpuWorld.
#[wasm_bindgen]
pub fn stress_set_threshold_gpu(on: bool) {
    THRESHOLD_GPU.set(on);
}

/// Whether the next threshold load runs on the GPU-resident world
/// (`stress_set_threshold_gpu`, default on).
#[wasm_bindgen]
pub fn stress_threshold_gpu() -> bool {
    THRESHOLD_GPU.get()
}

/// Running (the frames step) or paused.
#[wasm_bindgen]
pub fn stress_set_running(on: bool) {
    RUNNING.set(on);
}

#[wasm_bindgen]
pub fn stress_running() -> bool {
    RUNNING.get()
}

/// The canvas's new drawing size (applied at the next frame).
#[wasm_bindgen]
pub fn stress_resize(width: u32, height: u32) {
    SIZE.set((width.max(1), height.max(1)));
}

/// The mass of the shots fired from now on, as a multiple of the lab's default ball
/// (the radius stays, as in the lab's other modes).
#[wasm_bindgen]
pub fn stress_set_shot_mass_scale(scale: f64) {
    if scale.is_finite() && scale > 0.0 {
        MASS_SCALE.set(scale);
    }
}

/// The speed of the shots fired from now on, as a multiple of the lab's
/// (`plan::default_shot_speed`).
#[wasm_bindgen]
pub fn stress_set_shot_speed_scale(scale: f64) {
    if scale.is_finite() && scale > 0.0 {
        SPEED_SCALE.set(scale);
    }
}

/// The balls' radius from the next load on, as a multiple of the lab's (the mass stays):
/// the renderer's shot meshes are made at load.
#[wasm_bindgen]
pub fn stress_set_shot_radius_scale(scale: f64) {
    if scale.is_finite() && scale > 0.0 {
        RADIUS_SCALE.set(scale);
    }
}

/// The ball presets (`stress_ref::wrecking::BALLS`), as JSON: `[{key, menu, mass, radius,
/// speed, rolling}]` in menu order.
#[wasm_bindgen]
pub fn stress_shot_presets() -> String {
    serde_json::to_string(stress_ref::wrecking::BALLS).expect("presets serialize")
}

/// The ball from the next load on: a preset's index (`stress_shot_presets`; its mass,
/// radius and speed, the mass, speed and radius scales applied on top), or -1 (any
/// other) for the scene's own ball.
#[wasm_bindgen]
pub fn stress_set_shot_preset(index: i32) {
    PRESET.set(usize::try_from(index).ok().filter(|&k| k < stress_ref::wrecking::BALLS.len()));
}

/// Fire the ball along the loaded pack's line of attack (a Wrecking Yard structure's,
/// `stress_ref::wrecking::launch`: at the target's height, or rolling in on the ground),
/// launched at the next frame. False when the scene has no line of attack or all shots are
/// fired.
#[wasm_bindgen]
pub fn stress_fire() -> bool {
    let (Some(s), Some(aim)) = (SHOOTING.get(), AIM.get()) else { return false };
    let speed = engine_fracture::plan::default_shot_speed(s.extent, s.speed) * SPEED_SCALE.get();
    let ball = stress_ref::wrecking::Ball { key: "", menu: "", mass: s.mass * MASS_SCALE.get(), radius: s.radius, speed, rolling: s.rolling };
    let k = FIRED.get();
    queue_shot(stress_ref::wrecking::launch(&ball, &aim, &format!("shot{k}")))
}

/// The loaded pack's demolition charges (`defaults.charges`, `stress_ref::wrecking`), as
/// JSON: `[{name, delay, bonds}]` in firing order (`bonds` how many each cuts); empty for
/// a scene without any.
#[wasm_bindgen]
pub fn stress_charges() -> String {
    DEMOLITION.with_borrow(|d| serde_json::Value::from(d.charges.iter().map(|c| serde_json::json!({"name": c.name, "delay": c.delay, "bonds": c.bonds.len()})).collect::<Vec<_>>()).to_string())
}

/// Set off the loaded pack's charge `name`, or with an empty name every charge, each its
/// delay after the next frame starts: its bonds are cut between two frames, before the
/// frame steps (never by stepping one again), and the debris it throws gets its kick
/// (`ThresholdWorld::cut_bonds`, `GpuWorld::cut_bonds`). Runs the simulation. False when
/// the scene has no charge of that name.
#[wasm_bindgen]
pub fn stress_detonate(name: String) -> bool {
    let known = DEMOLITION.with_borrow(|d| d.charges.iter().any(|c| name.is_empty() || c.name == name));
    if known {
        DETONATE.with_borrow_mut(|q| q.push(name));
        RUNNING.set(true);
    }
    known
}

/// The loaded pack's launched bonded bodies (`defaults.launches`, `stress_ref::launch`), as
/// JSON: `[{name, delay, chunks}]` in the pack's order; empty for a scene without any.
#[wasm_bindgen]
pub fn stress_launches() -> String {
    LAUNCHER.with_borrow(|l| serde_json::Value::from(l.launches.iter().map(|x| serde_json::json!({"name": x.name, "delay": x.delay, "chunks": x.chunks.len()})).collect::<Vec<_>>()).to_string())
}

/// Launch the loaded pack's bonded body `name`, or with an empty name every one, each its
/// delay after the next frame starts: its island is set moving between two frames, before
/// the frame steps (`ThresholdWorld::launch_island`, `GpuWorld::launch_island`; never by
/// stepping one again). Runs the simulation. False when the scene has no launch of that
/// name.
#[wasm_bindgen]
pub fn stress_launch(name: String) -> bool {
    let known = LAUNCHER.with_borrow(|l| l.launches.iter().any(|x| name.is_empty() || x.name == name));
    if known {
        LAUNCH.with_borrow_mut(|q| q.push(name));
        RUNNING.set(true);
    }
    known
}

/// The launches asked for since the last frame, released from `frame` (each its delay
/// later), and those due before it.
fn launches_due(frame: u64, dt: f64) -> Vec<stress_ref::launch::Launch> {
    let asked = LAUNCH.with_borrow_mut(std::mem::take);
    LAUNCHER.with_borrow_mut(|l| {
        for name in asked {
            l.arm(&name, frame, dt);
        }
        l.due(frame)
    })
}

/// The charges asked for since the last frame, set off from `frame` (the world's frames
/// stepped), and those due before it.
fn charges_due(frame: u64, dt: f64) -> Vec<stress_ref::wrecking::PackCharge> {
    let asked = DETONATE.with_borrow_mut(std::mem::take);
    DEMOLITION.with_borrow_mut(|d| {
        for name in asked {
            d.arm(&name, frame, dt);
        }
        d.due(frame)
    })
}

/// The strength of every bond of the next `stress_load`, as a multiple of the asset's: the
/// tensile, compressive and shear strengths of the bonds' materials (their fracture
/// energies and stiffness stay).
#[wasm_bindgen]
pub fn stress_set_bond_strength_scale(scale: f64) {
    if scale.is_finite() && scale > 0.0 {
        STRENGTH_SCALE.set(scale);
    }
}

/// The capability profile of the next `stress_load`: `blast_parity` (the default) or
/// `threshold` (the v0 stress mode: bonds break on contact force alone, no solve).
#[wasm_bindgen]
pub fn stress_set_profile(name: String) -> Result<(), JsValue> {
    let p = Profile::parse(&name).ok_or_else(|| JsValue::from_str(&format!("no profile {name}")))?;
    PROFILE.set(p);
    Ok(())
}

/// The physics frame rate (Hz) of the next `stress_load`: one resident frame per
/// `1/hz` s of simulated time. 0 keeps the scene's own (60 Hz).
#[wasm_bindgen]
pub fn stress_set_physics_hz(hz: f64) {
    PHYSICS_HZ.set(if hz.is_finite() && hz > 0.0 { hz } else { 0.0 });
}

/// The lab's view, from the next frame drawn: the bond graph (each intact bond a line
/// between its chunks' centres) and the colour mode (the View menu, `ColorMode::from_index`):
/// 0 materials, 1 body state (moving orange, at rest blue, static grey: the coupled engine
/// keeps bodies awake, so "at rest" means slow), 2 one colour per body, kept as it breaks, 3 materials untinted (debris too).
#[wasm_bindgen]
pub fn stress_set_view(bond_graph: bool, color_mode: u32) {
    // Only noted: a frame may hold the world across a GPU wait (JSPI) while the page calls
    // this; the frame applies it before it draws.
    VIEW.set((bond_graph, ColorMode::from_index(color_mode)));
}

/// Draw dust and chips where fragments change speed sharply (impacts, breaks, landings),
/// from the next frame. Render only: the physics is the same either way. On by default.
#[wasm_bindgen]
pub fn stress_set_fx(on: bool) {
    FX.set(on);
}

/// Orbit the camera (first person: turn the look) by a pointer drag (CSS pixels), as the
/// lab's other modes.
#[wasm_bindgen]
pub fn stress_drag(dx: f64, dy: f64) {
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.drag(dx, dy);
        }
    });
}

/// Zoom by a wheel delta (the first-person modes: their travel speed).
#[wasm_bindgen]
pub fn stress_zoom(delta_y: f64) {
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.zoom(delta_y);
        }
    });
}

/// Place the orbit camera (scripted takes): yaw, pitch (radians), distance (m) and,
/// unless `target` is empty, the target `[x, y, z]` in the lab's frame
/// (`Orbit::set_orbit`). Switches to orbit mode.
#[wasm_bindgen]
pub fn stress_set_orbit(yaw: f64, pitch: f64, distance: f64, target: &[f64]) {
    let target = (target.len() == 3).then(|| [target[0], target[1], target[2]]);
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.set_orbit(yaw, pitch, distance, target);
        }
    });
}

/// `[yaw, pitch, distance, target x, y, z]`, as [`stress_set_orbit`] takes them (empty
/// before a load).
#[wasm_bindgen]
pub fn stress_orbit_state() -> Vec<f64> {
    CAMERA.with_borrow(|c| c.as_ref().map_or(Vec::new(), |c| c.orbit_state().to_vec()))
}

/// The camera's view mode, as the lab's other modes (the Camera menu: 0 orbit, 1
/// first-person walk, 2 fly); kept for the next loads.
#[wasm_bindgen]
pub fn stress_set_view_mode(mode: u32) {
    let mode = ViewMode::from_index(mode);
    VIEW_MODE.set(mode);
    let (w, h) = SIZE.get();
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.set_mode(mode, w.max(1) as f64 / h.max(1) as f64);
        }
    });
}

/// First-person mouse-look by pointer-lock movement (CSS pixels).
#[wasm_bindgen]
pub fn stress_mouse_look(dx: f64, dy: f64) {
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.mouse_look(dx, dy);
        }
    });
}

/// First-person travel for `dt` seconds (axes in -1..1: forward, right, up).
#[wasm_bindgen]
pub fn stress_travel(forward: f64, right: f64, up: f64, sprint: bool, dt: f64) {
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.travel(forward, right, up, sprint, dt);
        }
    });
}

/// Fire a projectile through viewport point (u, v in 0..1 from the top left), by the lab's
/// rule (`plan::shot_velocity`): launched into the world at the next frame. False when
/// all shots are fired.
#[wasm_bindgen]
pub fn stress_shoot(u: f64, v: f64) -> bool {
    let Some(s) = SHOOTING.get() else { return false };
    let (w, h) = SIZE.get();
    let Some(ray) = CAMERA.with_borrow(|c| c.as_ref().map(|c| c.aim(u, v, w as f64 / h as f64, s.radius))) else { return false };
    shoot(s, ray)
}

/// The shot ray through viewport point (u, v in 0..1 from the top left), as
/// [`stress_shoot`] aims: `[origin x, y, z, direction x, y, z]` in the lab's frame (empty
/// before a load). With [`stress_shoot_ray`], a click's shot can be logged and fired again.
#[wasm_bindgen]
pub fn stress_aim(u: f64, v: f64) -> Vec<f64> {
    let Some(s) = SHOOTING.get() else { return Vec::new() };
    let (w, h) = SIZE.get();
    CAMERA.with_borrow(|c| {
        c.as_ref().map_or(Vec::new(), |c| {
            let r = c.aim(u, v, w as f64 / h as f64, s.radius);
            [r.origin, r.direction].concat()
        })
    })
}

/// Fire a projectile along a ray in the lab's frame (origin, then direction), by the
/// same rule as [`stress_shoot`]: a shot that does not depend on the camera (replays and
/// scripted takes). False when all shots are fired.
#[wasm_bindgen]
pub fn stress_shoot_ray(origin: &[f64], direction: &[f64]) -> bool {
    let Some(s) = SHOOTING.get() else { return false };
    if FIRED.get() >= SHOTS && TWORLD.with_borrow(|w| w.is_none()) {
        return false;
    }
    if origin.len() != 3 || direction.len() != 3 {
        return false;
    }
    shoot(s, engine_fracture::scene::Ray { origin: [origin[0], origin[1], origin[2]], direction: [direction[0], direction[1], direction[2]] })
}

/// Queue the shot along `ray` for the next frame (`stress_shoot`, `stress_shoot_ray`).
fn shoot(s: Shooting, ray: engine_fracture::scene::Ray) -> bool {
    let speed = engine_fracture::plan::default_shot_speed(s.extent, s.speed) * SPEED_SCALE.get();
    let vel = engine_fracture::plan::shot_velocity(s.center, s.extent, s.radius, ray.origin, ray.direction, Some(speed), s.arc);
    let k = FIRED.get();
    queue_shot(ImpactorDesc {
        name: format!("shot{k}"),
        shape: ImpactorShape::Sphere { radius: s.radius },
        mass: s.mass * MASS_SCALE.get(),
        position: zup(std::array::from_fn(|k| ray.origin[k] - s.placement[k])),
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: zup(vel),
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    })
}

/// Queue a shot for the next frame and run. False when all shots are fired: the resident
/// GpuWorld draws SHOTS balls; the threshold world reuses its oldest.
fn queue_shot(desc: ImpactorDesc) -> bool {
    if FIRED.get() >= SHOTS && (DRIVING.get() || TWORLD.with_borrow(|w| w.is_none())) {
        return false;
    }
    FIRED.set(FIRED.get() + 1);
    PENDING.with_borrow_mut(|p| p.push(desc));
    RUNNING.set(true);
    true
}

fn fail(e: String) -> u32 {
    web_sys::console::error_1(&e.clone().into());
    ERROR.set(e);
    0
}

/// Build the world from the input set last (the GLB at `level`, the cuboid houses or the
/// pack). Called through `WebAssembly.promising` (it reads the GPU). Returns 1, or 0 with [`stress_error`].
#[no_mangle]
pub extern "C" fn stress_load(level: u32) -> u32 {
    match load(level as u8) {
        Ok(()) => 1,
        Err(e) => fail(e),
    }
}

fn load(level: u8) -> Result<(), String> {
    WORLD.with_borrow_mut(|w| *w = None);
    TWORLD.with_borrow_mut(|w| *w = None);
    DWORLD.with_borrow_mut(|w| *w = None);
    DRIVING.set(false);
    // The lab's own scene (its camera, centre, extent and projectile) and the stress
    // oracle's input: the GLB's level, the cuboid houses in their layout, or the pack.
    let profile = PROFILE.get();
    let mut warnings = Vec::new();
    let mut aim = None;
    let mut demolition = stress_rubble::demolition::Demolition::default();
    let mut launcher = stress_rubble::launch::Launcher::default();
    let mut drives = Vec::new();
    let mut pack = Vec::new();
    let (lab, mut scene) = SOURCE.with_borrow(|source| -> Result<_, String> {
        Ok(match source {
            Source::Cuboid(layout) => {
                let (lab, json) = engine_fracture::samples::cuboid_stress_scene(*layout).map_err(|e| format!("cuboid houses: {e}"))?;
                // Bonds to the ground break like any bond (an anchored piece comes free when they do).
                (lab, stress_ref::shardwright::import_value_with(&json, "cuboid_houses", true)?)
            }
            Source::Glb(bytes) => {
                let asset = engine_fracture::parse_fracture_glb(bytes).map_err(|e| format!("fracture GLB: {e}"))?;
                let lab = engine_fracture::build_scene(&asset, level, None).map_err(|e| format!("level {level}: {e}"))?;
                let json = engine_fracture::stress_export::shardwright_stress_json(&asset, level);
                (lab, stress_ref::shardwright::import_value_with(&json, &asset.physics.name, true)?)
            }
            Source::Pack(bytes) => {
                let (lab, info) = engine_fracture::pack::build_pack_scene(bytes).map_err(|e| format!("scene pack: {e}"))?;
                let json: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("scene pack: {e}"))?;
                let (scene, notes) = stress_ref::scene_pack::import_value_with(&json, &info.key, "structure", &info.key, profile.import_options())?;
                warnings = notes;
                aim = stress_ref::wrecking::aim_of(&json);
                demolition = stress_rubble::demolition::Demolition::of_pack(&json);
                launcher = stress_rubble::launch::Launcher::of_pack(&json).map_err(|e| format!("scene pack: {e}"))?;
                drives = stress_ref::drive::drives_of(&json).map_err(|e| format!("scene pack: {e}"))?;
                if !drives.is_empty() {
                    pack = bytes.clone();
                }
                (lab, scene)
            }
        })
    })?;
    WARNINGS.set(warnings);
    AIM.set(aim);
    DEMOLITION.set(demolition);
    DETONATE.with_borrow_mut(|q| q.clear());
    LAUNCHER.set(launcher);
    LAUNCH.with_borrow_mut(|q| q.clear());
    let mut camera = Orbit::new(&lab);
    let (w, h) = SIZE.get();
    camera.set_mode(VIEW_MODE.get(), w.max(1) as f64 / h.max(1) as f64);
    CAMERA.set(Some(camera));
    // The preset's ball, else the scene's.
    let (radius, mass, speed, rolling) = match PRESET.get().map(|k| stress_ref::wrecking::BALLS[k]) {
        Some(b) => (b.radius, b.mass, Some(b.speed), b.rolling),
        None => (lab.projectile_radius, lab.projectile_mass, lab.projectile_speed, false),
    };
    SHOOTING.set(Some(Shooting { center: lab.center, extent: lab.extent, radius: radius * RADIUS_SCALE.get(), mass, speed, arc: lab.projectile_arc, placement: lab.placement, rolling }));
    FIRED.set(0);
    RUNNING.set(false);
    STATS.set("{}".into());
    PENDING.with_borrow_mut(|p| p.clear());
    let k = STRENGTH_SCALE.get();
    if k != 1.0 {
        let used: std::collections::BTreeSet<String> = scene.bodies.iter().flat_map(|b| &b.bonds).map(|b| b.material.clone()).collect();
        for name in used {
            if let Some(m) = scene.materials.get_mut(&name) {
                m.tensile_strength *= k;
                m.compressive_strength *= k;
                m.cohesion *= k;
                m.shear_cap = m.shear_cap.map(|c| c * k);
            }
        }
    }
    let hz = PHYSICS_HZ.get();
    if hz > 0.0 {
        scene.sim.frame_dt = 1.0 / hz;
    }
    scene.validate()?;
    let mut scene = scene.with_derived();
    // The GPU's methods, as the native runs (`STRESS_METHODS=established`).
    scene.sim.methods = stress_ref::scene::Methods::established();
    profile.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    if !drives.is_empty() {
        // Driven vehicles: rubble's world API with the threshold model, as the pack is (the
        // lab's strength and rate settings are its own modes').
        if profile != Profile::Threshold {
            return Err("this scene's vehicles are driven in rubble's world API, with the threshold profile: choose Bonds · Stress solver · simple threshold".into());
        }
        if STRENGTH_SCALE.get() != 1.0 || PHYSICS_HZ.get() > 0.0 {
            WARNINGS.with_borrow_mut(|w| w.push("driven scenes run the pack's own bond strengths and rate (Bond strength and Physics are not applied)".into()));
        }
        return DEVICE.with_borrow(|d| {
            let d = d.as_ref().ok_or("stress_init first")?;
            let l = drive::load(&d.gpu, &pack, d.config.width, d.config.height, d.view_format)?;
            DWORLD.with_borrow_mut(|w| *w = Some(l));
            DRIVING.set(true);
            Ok(())
        });
    }
    if profile == Profile::Threshold && THRESHOLD_GPU.get() {
        return DEVICE.with_borrow(|d| {
            let d = d.as_ref().ok_or("stress_init first")?;
            let gpu = &d.gpu;
            let s = SHOOTING.get().expect("shooting set");
            let mut world = stress_rubble::threshold::ThresholdWorld::new(gpu, &scene, s.radius, s.mass)?;
            let shots = world.shot_impactors();
            let renderer = Renderer::for_target(gpu, &world.mirror, &shots, d.config.width, d.config.height, d.view_format);
            world.attach_view(gpu, &renderer);
            let m = &world.mirror;
            let totals = (
                m.structures.iter().map(|s| s.chunks.len()).sum::<usize>(),
                m.structures.iter().flat_map(|s| &s.chunks).map(|c| c.parts.as_ref().map_or(1, |p| p.len())).sum::<usize>(),
                world.tables.bond_count() as usize,
            );
            TWORLD.with_borrow_mut(|w| *w = Some(TLoaded { world, renderer, behind: 0.0, totals }));
            Ok(())
        });
    }
    DEVICE.with_borrow(|d| {
        let d = d.as_ref().ok_or("stress_init first")?;
        let gpu = &d.gpu;
        let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).map_err(|e| format!("rubble: {e}"))?;
        let (iterations, substeps) = profile.engine_policy();
        e.set_policy(iterations, substeps);
        let mut world = GpuWorld::with_resident_engine(gpu, &scene, e)?;
        world.redo_fractured_frames = false;
        world.set_impact_islands(true);
        world.apply_profile(profile);
        let shots = shot_meshes(&world);
        // Meshes for every chunk and for the shots to come.
        let (mut renderer, totals) = world.view_with(gpu, |mirror, _| {
            let totals = (
                mirror.structures.iter().map(|s| s.chunks.len()).sum::<usize>(),
                mirror.structures.iter().flat_map(|s| &s.chunks).map(|c| c.parts.as_ref().map_or(1, |p| p.len())).sum::<usize>(),
                mirror.structures.iter().map(|s| s.bonds.len()).sum::<usize>(),
            );
            (Renderer::for_target(gpu, mirror, &shots, d.config.width, d.config.height, d.view_format), totals)
        });
        (renderer.bond_graph, renderer.colors) = VIEW.get();
        WORLD.with_borrow_mut(|w| *w = Some(Loaded { world, renderer, behind: 0.0, totals }));
        Ok(())
    })
}

/// One animation frame `elapsed_ms` after the last: the frames of simulated time that
/// fall due while running (the backlog held to [`MAX_BEHIND_S`], so a slow frame does not
/// snowball), then the world drawn into the canvas. Called through `WebAssembly.promising`. Returns 1, or 0
/// with [`stress_error`].
/// The most simulated time one animation frame catches up (s). A fixed time, not a count
/// of physics frames: a browser frame longer than two physics frames (a 60 Hz display
/// under a 120 Hz simulation, or a hitch) otherwise drops time and the run falls behind
/// real time even when each physics frame is fast.
const MAX_BEHIND_S: f64 = 0.1;
/// Physics frames the GPU-resident threshold world may have queued ahead of the last one the
/// GPU finished (its counters read back).
const IN_FLIGHT: u64 = 3;

#[no_mangle]
pub extern "C" fn stress_frame(elapsed_ms: f64) -> u32 {
    match frame(Advance::Elapsed(elapsed_ms)) {
        Ok(()) => 1,
        Err(e) => fail(e),
    }
}

/// One animation frame that steps exactly `steps` physics frames (shots launched first),
/// running or paused, then draws: the frame-locked recording of a take (`__record`), where
/// each video frame advances a fixed amount of simulated time whatever the wall clock does.
/// Called through `WebAssembly.promising`. Returns 1, or 0 with [`stress_error`].
#[no_mangle]
pub extern "C" fn stress_frame_steps(steps: u32) -> u32 {
    match frame(Advance::Steps(steps)) {
        Ok(()) => 1,
        Err(e) => fail(e),
    }
}

/// How far a frame advances the simulation: the frames that fall due in this much wall
/// time while running (live), or exactly this many (frame-locked recording).
#[derive(Clone, Copy)]
enum Advance {
    Elapsed(f64),
    Steps(u32),
}

/// The lab's camera for the renderer: its view-projection and eye in the stress scene's
/// frame. The lab's camera looks at its own frame (the asset, +Y up, moved by
/// `placement`); the renderer draws the stress scene's (+Z up): view_proj times the map
/// from the stress frame into the lab's, column by column.
fn camera_view(width: u32, height: u32) -> ([[f32; 4]; 4], Vec3) {
    let aspect = width as f64 / height as f64;
    let (vp, eye) = CAMERA.with_borrow(|c| {
        let c = c.as_ref().expect("a camera with the world");
        (c.view_proj(aspect), c.eye(aspect))
    });
    let p = SHOOTING.get().map_or([0.0; 3], |s| s.placement);
    let col = |k: usize| -> [f32; 4] { std::array::from_fn(|r| vp[4 * k + r]) };
    let (c0, c1, c2, c3) = (col(0), col(1), col(2), col(3));
    let moved: [f32; 4] = std::array::from_fn(|r| p[0] as f32 * c0[r] + p[1] as f32 * c1[r] + p[2] as f32 * c2[r] + c3[r]);
    let view_proj = [c0, c2.map(|x| -x), c1, moved];
    let eye = Vec3::from_array(zup(std::array::from_fn(|k| eye[k] - p[k])));
    (view_proj, eye)
}

/// A frame of the threshold profile's GPU-resident world: the frames due (shots
/// launched), the view written on the GPU and drawn, the counters as the GPU last
/// finished them (never waited for). Nothing here reads the GPU.
fn frame_threshold(d: &mut Device, l: &mut TLoaded, advance: Advance) -> Result<(), String> {
    let gpu = &d.gpu;
    let (width, height) = SIZE.get();
    if (width, height) != (d.config.width, d.config.height) {
        d.config.width = width;
        d.config.height = height;
        d.surface.configure(&gpu.device, &d.config);
        let shots = l.world.shot_impactors();
        l.renderer = Renderer::for_target(gpu, &l.world.mirror, &shots, width, height, d.view_format);
        l.world.attach_view(gpu, &l.renderer);
    }
    let dt = l.world.frame_dt;
    let mut stepped = 0;
    // A launch set something moving: the effects must not read that as an impact.
    let mut launched = false;
    let t_step = web_time::Instant::now();
    let fixed = match advance {
        Advance::Steps(n) => Some(n),
        Advance::Elapsed(_) => None,
    };
    if RUNNING.get() || fixed.is_some() {
        for desc in PENDING.with_borrow_mut(std::mem::take) {
            l.world.launch(gpu, &desc);
        }
    }
    if let Some(n) = fixed {
        // Frame-locked: every frame asked for, however far the GPU is behind.
        for _ in 0..n {
            // Charges due: cut between this frame and the last, as live play does.
            for c in charges_due(l.world.frame, dt) {
                l.world.detonate(gpu, &c).map_err(|e| format!("charge {}: {e}", c.name))?;
            }
            // Launches due: set moving between this frame and the last.
            for x in launches_due(l.world.frame, dt) {
                l.world.launch_island(gpu, 0, &x).map_err(|e| format!("launch {}: {e}", x.name))?;
                launched = true;
            }
            l.world.step_frame(gpu)?;
            stepped += 1;
        }
    } else if let (true, Advance::Elapsed(elapsed_ms)) = (RUNNING.get(), advance) {
        l.behind = (l.behind + elapsed_ms * 1e-3).min(MAX_BEHIND_S.max(2.0 * dt));
        // At most `IN_FLIGHT` frames queued ahead of what the GPU has finished: when it
        // cannot keep up, the simulation runs slower than real time instead of the queue
        // (and with it the display's frames) backing up.
        while l.behind >= dt {
            l.world.poll(gpu);
            if l.world.frame >= l.world.last.frame + IN_FLIGHT {
                l.behind = 0.0;
                break;
            }
            // Charges due: cut between this frame and the last (they wait for the GPU).
            for c in charges_due(l.world.frame, dt) {
                l.world.detonate(gpu, &c).map_err(|e| format!("charge {}: {e}", c.name))?;
            }
            // Launches due: set moving between this frame and the last.
            for x in launches_due(l.world.frame, dt) {
                l.world.launch_island(gpu, 0, &x).map_err(|e| format!("launch {}: {e}", x.name))?;
                launched = true;
            }
            l.world.step_frame(gpu)?;
            l.behind -= dt;
            stepped += 1;
        }
    }
    let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
    let (view_proj, eye) = camera_view(width, height);
    let texture = match d.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => return Ok(()),
    };
    let view = texture.texture.create_view(&wgpu::TextureViewDescriptor { format: Some(d.view_format), ..Default::default() });
    let t_draw = web_time::Instant::now();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let (bonds, colors) = VIEW.get();
    let mode = colors.index();
    let lines = l.world.encode_view(gpu, &mut encoder, &l.renderer, mode, bonds);
    l.renderer.set_fx(gpu, FX.get());
    l.renderer.set_fx_dt(stepped as f64 * dt);
    if launched {
        l.renderer.set_fx_quiet();
    }
    l.renderer.draw_resident(gpu, &mut encoder, &view, view_proj, eye, lines);
    gpu.queue.submit([encoder.finish()]);
    gpu.queue.present(texture);
    let draw_ms = t_draw.elapsed().as_secs_f64() * 1e3;
    PHASES.set([step_ms, 0.0, draw_ms]);
    let st = l.world.poll(gpu);
    let s = SHOOTING.get();
    let stats = serde_json::json!({
        "gpuResident": true,
        "simSeconds": l.world.frame as f64 * dt,
        "stepMs": if stepped > 0 { step_ms / stepped as f64 } else { 0.0 },
        "chunks": l.totals.0,
        "hulls": l.totals.1,
        "bonds": l.totals.2,
        "shots": FIRED.get(),
        "warnings": WARNINGS.with_borrow(|w| w.clone()),
        "canFire": AIM.get().is_some(),
        "charges": DEMOLITION.with_borrow(|d| d.names()),
        "chargesArmed": DEMOLITION.with_borrow(|d| d.pending()) + DETONATE.with_borrow(|q| q.len()),
        "launches": LAUNCHER.with_borrow(|l| l.names()),
        // The lab's translation of the scene as built (`Scene::placement`, the asset frame):
        // takes aim at built coordinates through it.
        "placement": s.map_or([0.0; 3], |s| s.placement),
        "strikes": l.world.strike_count(),
        "launchesArmed": LAUNCHER.with_borrow(|l| l.pending()) + LAUNCH.with_borrow(|q| q.len()),
        "ballMass": s.map_or(0.0, |s| s.mass),
        "ballRadius": s.map_or(0.0, |s| s.radius),
        "ballSpeed": s.map_or(0.0, |s| engine_fracture::plan::default_shot_speed(s.extent, s.speed)),
        "frameHz": 1.0 / dt,
        "profile": PROFILE.get().name(),
        // As the GPU last finished them (a frame or two behind).
        "frames": l.world.frame,
        "statsFrame": st.frame,
        "bodies": st.bodies,
        "fragments": st.bodies,
        "brokenBonds": st.broken,
        "splitFrames": st.split_frames,
        "awake": st.awake,
        "contacts": st.contacts,
        "errors": st.error,
        "engineOverflow": st.engine_overflow,
        "compactions": 0,
    });
    STATS.set(stats.to_string());
    if st.error != 0 || st.engine_overflow != 0 {
        // Paused: one report, not one a frame.
        RUNNING.set(false);
        return Err(format!("threshold world: fracture errors {:#x}, engine overflow {:#x} (fracture: 1 no free body slot, 2 a mass sum out of range, 4 a union did not settle, 8 the impact pulse pool is full; engine: 1 contacts, 2 candidate pairs, 64 manifolds)", st.error, st.engine_overflow));
    }
    Ok(())
}

/// A frame of a scene with driven vehicles (`drive`): the frames due, each vehicle's input
/// and the world's step, then their readbacks waited for and the world drawn.
fn frame_drive(d: &mut Device, l: &mut drive::DLoaded, advance: Advance) -> Result<(), String> {
    let gpu = &d.gpu;
    let (width, height) = SIZE.get();
    if (width, height) != (d.config.width, d.config.height) {
        d.config.width = width;
        d.config.height = height;
        d.surface.configure(&gpu.device, &d.config);
        l.resize(gpu, width, height, d.view_format)?;
    }
    let dt = l.dt();
    let mut stepped = 0;
    let t_step = web_time::Instant::now();
    // Shots fired since the last frame, and charges due: between this step and the last.
    let before = |l: &mut drive::DLoaded| -> Result<(), String> {
        for desc in PENDING.with_borrow_mut(std::mem::take) {
            l.shoot(&desc)?;
        }
        for c in charges_due(l.frame(), dt) {
            l.cut(&c)?;
        }
        Ok(())
    };
    match advance {
        Advance::Steps(n) => {
            for _ in 0..n {
                before(l)?;
                l.step()?;
                stepped += 1;
            }
        }
        Advance::Elapsed(elapsed_ms) if RUNNING.get() => {
            l.behind = (l.behind + elapsed_ms * 1e-3).min(MAX_BEHIND_S.max(2.0 * dt));
            while l.behind >= dt {
                before(l)?;
                l.step()?;
                l.behind -= dt;
                stepped += 1;
            }
        }
        Advance::Elapsed(_) => {}
    }
    if stepped > 0 {
        l.poll()?;
    }
    let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
    let (view_proj, eye) = camera_view(width, height);
    let texture = match d.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => return Ok(()),
    };
    let view = texture.texture.create_view(&wgpu::TextureViewDescriptor { format: Some(d.view_format), ..Default::default() });
    let t_draw = web_time::Instant::now();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let (bonds, colors) = VIEW.get();
    let lines = l.encode_view(gpu, &mut encoder, colors.index(), bonds)?;
    l.renderer.set_fx(gpu, FX.get());
    l.renderer.set_fx_dt(stepped as f64 * dt);
    l.renderer.draw_resident(gpu, &mut encoder, &view, view_proj, eye, lines);
    gpu.queue.submit([encoder.finish()]);
    gpu.queue.present(texture);
    let draw_ms = t_draw.elapsed().as_secs_f64() * 1e3;
    PHASES.set([step_ms, 0.0, draw_ms]);
    let mut stats = l.stats(step_ms, stepped);
    stats["charges"] = serde_json::json!(DEMOLITION.with_borrow(|d| d.names()));
    stats["chargesArmed"] = serde_json::json!(DEMOLITION.with_borrow(|d| d.pending()) + DETONATE.with_borrow(|q| q.len()));
    let s = SHOOTING.get();
    stats["placement"] = serde_json::json!(s.map_or([0.0; 3], |s| s.placement));
    stats["warnings"] = serde_json::json!(WARNINGS.with_borrow(|w| w.clone()));
    STATS.set(stats.to_string());
    Ok(())
}

fn frame(advance: Advance) -> Result<(), String> {
    let driving = DEVICE.with_borrow_mut(|d| {
        DWORLD.with_borrow_mut(|w| match (d.as_mut(), w.as_mut()) {
            (Some(d), Some(l)) => Some(frame_drive(d, l, advance)),
            _ => None,
        })
    });
    if let Some(r) = driving {
        if r.is_err() {
            RUNNING.set(false);
        }
        return r;
    }
    let threshold = DEVICE.with_borrow_mut(|d| {
        TWORLD.with_borrow_mut(|w| match (d.as_mut(), w.as_mut()) {
            (Some(d), Some(l)) => Some(frame_threshold(d, l, advance)),
            _ => None,
        })
    });
    if let Some(r) = threshold {
        return r;
    }
    DEVICE.with_borrow_mut(|d| {
        let d = d.as_mut().ok_or("stress_init first")?;
        WORLD.with_borrow_mut(|w| {
            let Some(l) = w.as_mut() else { return Ok(()) };
            let gpu = &d.gpu;
            let (width, height) = SIZE.get();
            if (width, height) != (d.config.width, d.config.height) {
                d.config.width = width;
                d.config.height = height;
                d.surface.configure(&gpu.device, &d.config);
                let shots = shot_meshes(&l.world);
                l.renderer = l.world.view_with(gpu, |mirror, _| Renderer::for_target(gpu, mirror, &shots, width, height, d.view_format));
                (l.renderer.bond_graph, l.renderer.colors) = VIEW.get();
            }
            let dt = l.world.scene.sim.frame_dt;
            let mut stepped = 0;
            let mut launched = false;
            let mut step_ms = 0.0;
            let due = match advance {
                // Frame-locked: exactly the frames asked for, running or paused.
                Advance::Steps(n) => Some(n),
                Advance::Elapsed(elapsed_ms) if RUNNING.get() => {
                    l.behind = (l.behind + elapsed_ms * 1e-3).min(MAX_BEHIND_S.max(2.0 * dt));
                    let mut n = 0;
                    while l.behind >= dt {
                        l.behind -= dt;
                        n += 1;
                    }
                    Some(n)
                }
                Advance::Elapsed(_) => None,
            };
            if let Some(n) = due {
                let shots = PENDING.with_borrow_mut(std::mem::take);
                for desc in &shots {
                    let imp = stress_ref::world::Impactor::from_desc(desc, &l.world.scene);
                    l.world.add_impactor(gpu, imp)?;
                }
                for _ in 0..n {
                    let t = web_time::Instant::now();
                    // Charges due: cut between this frame and the last.
                    for c in charges_due(l.world.frame, dt) {
                        let bonds = c.bond_indices(&l.world.scene.bodies[0].bonds).map_err(|e| format!("charge {}: {e}", c.name))?;
                        l.world.cut_bonds(gpu, 0, &bonds, &c.debris, c.kick).map_err(|e| format!("charge {}: {e}", c.name))?;
                    }
                    // Launches due: set moving between this frame and the last.
                    for x in launches_due(l.world.frame, dt) {
                        l.world.launch_island(gpu, 0, &x.chunks, x.velocity, x.angular_velocity).map_err(|e| format!("launch {}: {e}", x.name))?;
                        launched = true;
                    }
                    l.world.step_frame(gpu)?;
                    step_ms += t.elapsed().as_secs_f64() * 1e3;
                    stepped += 1;
                }
            }
            let (view_proj, eye) = camera_view(width, height);
            // Every GPU read of the view (splits, bonds, the engine's motion) before the
            // canvas texture is taken: none waits while the frame holds it.
            let t_view = web_time::Instant::now();
            l.world.view_prepare(gpu);
            let texture = match d.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                _ => return Ok(()),
            };
            let view = texture.texture.create_view(&wgpu::TextureViewDescriptor { format: Some(d.view_format), ..Default::default() });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            (l.renderer.bond_graph, l.renderer.colors) = VIEW.get();
            l.renderer.set_fx(gpu, FX.get());
            l.renderer.set_fx_dt(stepped as f64 * dt);
            if launched {
                l.renderer.set_fx_quiet();
            }
    if launched {
        l.renderer.set_fx_quiet();
    }
            let want_stats = stepped > 0 || STATS.with_borrow(|s| s == "{}");
            let renderer = &l.renderer;
            let totals = l.totals;
            let (view_ms, draw_ms, stats) = l.world.view_with(gpu, |mirror, imps| {
                let view_ms = t_view.elapsed().as_secs_f64() * 1e3;
                let t_draw = web_time::Instant::now();
                renderer.draw(gpu, &mut encoder, &view, mirror, imps, view_proj, eye);
                let draw_ms = t_draw.elapsed().as_secs_f64() * 1e3;
                let stats = want_stats.then(|| {
                    let moving = mirror.clusters.iter().filter(|c| !c.anchored).count();
                    serde_json::json!({
                        "simSeconds": mirror.time,
                        "stepMs": if stepped > 0 { step_ms / stepped as f64 } else { 0.0 },
                        "moving": moving,
                        "chunks": totals.0,
                        "hulls": totals.1,
                        "bonds": totals.2,
                        "shots": FIRED.get(),
                        "warnings": WARNINGS.with_borrow(|w| w.clone()),
                        "canFire": AIM.get().is_some(),
                        "charges": DEMOLITION.with_borrow(|d| d.names()),
                        "chargesArmed": DEMOLITION.with_borrow(|d| d.pending()) + DETONATE.with_borrow(|q| q.len()),
                        "launches": LAUNCHER.with_borrow(|l| l.names()),
                        "placement": SHOOTING.get().map_or([0.0; 3], |s| s.placement),
                        "launchesArmed": LAUNCHER.with_borrow(|l| l.pending()) + LAUNCH.with_borrow(|q| q.len()),
                        "gpuWaits": engine_gpu_wait::WAITS.load(std::sync::atomic::Ordering::Relaxed),
                        "gpuYields": engine_gpu_wait::YIELDS.load(std::sync::atomic::Ordering::Relaxed),
                        "ballMass": SHOOTING.get().map_or(0.0, |s| s.mass),
                        "ballRadius": SHOOTING.get().map_or(0.0, |s| s.radius),
                        "ballSpeed": SHOOTING.get().map_or(0.0, |s| engine_fracture::plan::default_shot_speed(s.extent, s.speed)),
                        "lastShot": imps.last().map(|i| serde_json::json!({"position": i.pose.position.to_array(), "velocity": i.velocity.to_array()})),
                        "frameHz": 1.0 / dt,
                        "profile": PROFILE.get().name(),
                        // Rigid bodies the structures are now (each split's children).
                        "bodies": mirror.clusters.len(),
                    })
                });
                (view_ms, draw_ms, stats)
            });
            gpu.queue.submit([encoder.finish()]);
            gpu.queue.present(texture);
            PHASES.set([step_ms, view_ms, draw_ms]);
            if let Some(mut st) = stats {
                // After the view followed the frame's splits.
                let o = l.world.observation();
                st["frames"] = serde_json::json!(l.world.frame);
                st["fragments"] = serde_json::json!(o.values.get("fragments").copied().unwrap_or(0.0));
                st["brokenBonds"] = serde_json::json!(o.values.get("broken_bonds").copied().unwrap_or(0.0));
                // Times the engine's split-retired slots were reclaimed.
                st["compactions"] = serde_json::json!(l.world.compactions);
                STATS.set(st.to_string());
            }
            Ok(())
        })
    })
}

/// Sphere impactors for every shot to come (the renderer's impactor slots).
fn shot_meshes(world: &GpuWorld) -> Vec<stress_ref::world::Impactor> {
    let s = SHOOTING.get().expect("a loaded scene");
    (0..SHOTS)
        .map(|k| {
            let desc = ImpactorDesc { name: format!("shot{k}"), shape: ImpactorShape::Sphere { radius: s.radius }, mass: s.mass, position: [0.0; 3], orientation: [1.0, 0.0, 0.0, 0.0], velocity: [0.0; 3], angular_velocity: [0.0; 3], material: "projectile".into(), crush: None };
            stress_ref::world::Impactor::from_desc(&desc, &world.scene)
        })
        .collect()
}

/// The save area of the Asyncify build (`stress_web_bg.asyncify.wasm`, for browsers without
/// JSPI such as Safari; `engine/scripts/build-web.sh`, `fracture/gpu-wait.mjs`): a header of
/// the next free address and the end address, then room for the stack a suspended export
/// saves. Returns the area's address; the page calls it once. The JSPI build never calls it.
#[no_mangle]
pub extern "C" fn stress_asyncify_buffer() -> u32 {
    const BYTES: usize = 1 << 20;
    let area: &'static mut [u8] = Box::leak(vec![0u8; BYTES].into_boxed_slice());
    let base = area.as_mut_ptr() as usize as u32;
    area[0..4].copy_from_slice(&(base + 8).to_le_bytes());
    area[4..8].copy_from_slice(&(base + BYTES as u32).to_le_bytes());
    base
}
