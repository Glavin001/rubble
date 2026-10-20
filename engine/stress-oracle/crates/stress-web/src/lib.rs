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
//! `WebAssembly.promising` ([`stress_load`], [`stress_frame`]); everything else is a
//! wasm-bindgen function that never waits. The camera, pending shots and stats live apart
//! from the world, so input arriving while a frame is suspended never meets a borrowed
//! world.

use std::cell::{Cell, RefCell};

use engine_fracture::view::Orbit;
use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};
use stress_rubble::render::Renderer;
use wasm_bindgen::prelude::*;

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
}

thread_local! {
    static DEVICE: RefCell<Option<Device>> = const { RefCell::new(None) };
    static WORLD: RefCell<Option<Loaded>> = const { RefCell::new(None) };
    /// What the next load builds (`stress_set_glb`, `stress_set_cuboid`, `stress_set_pack`).
    static SOURCE: RefCell<Source> = const { RefCell::new(Source::Glb(Vec::new())) };
    /// What the last load could not take from its input as given (`stress_stats`).
    static WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static CAMERA: RefCell<Option<Orbit>> = const { RefCell::new(None) };
    static SHOOTING: Cell<Option<Shooting>> = const { Cell::new(None) };
    static PENDING: RefCell<Vec<ImpactorDesc>> = const { RefCell::new(Vec::new()) };
    static FIRED: Cell<usize> = const { Cell::new(0) };
    /// The shots' mass as a multiple of the lab's default ball (`stress_set_shot_mass_scale`).
    static MASS_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The bonds' strengths as a multiple of the asset's (`stress_set_bond_strength_scale`).
    static STRENGTH_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The view toggles (`stress_set_view`): bond graph, colour by body state.
    static VIEW: Cell<(bool, bool)> = const { Cell::new((false, false)) };
    /// The capability profile of the next `stress_load` (`stress_set_profile`).
    static PROFILE: Cell<Profile> = const { Cell::new(Profile::BlastParity) };
    /// The physics frame rate of the next `stress_load` (`stress_set_physics_hz`); 0 keeps
    /// the scene's own.
    static PHYSICS_HZ: Cell<f64> = const { Cell::new(0.0) };
    static RUNNING: Cell<bool> = const { Cell::new(false) };
    static SIZE: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
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
        // The JavaScript stack: which export reached the panic.
        web_sys::console::trace_0();
    }));
    let (width, height) = (canvas.width().max(1), canvas.height().max(1));
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(|e| JsValue::from_str(&format!("canvas surface: {e}")))?;
    let gpu = Gpu::with_instance(&instance, Some(&surface)).await.map_err(|e| JsValue::from_str(&e))?;
    let caps = surface.get_capabilities(&gpu.adapter);
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

/// The lab's view toggles, from the next frame drawn: the bond graph (each intact bond a
/// line between its chunks' centres) and colour by body state (moving orange, at rest
/// blue, static grey: the coupled engine keeps bodies awake, so "at rest" means slow).
#[wasm_bindgen]
pub fn stress_set_view(bond_graph: bool, sleep_colors: bool) {
    // Only noted: a frame may hold the world across a GPU wait (JSPI) while the page calls
    // this; the frame applies it before it draws.
    VIEW.set((bond_graph, sleep_colors));
}

/// Orbit the camera by a pointer drag (CSS pixels), as the lab's other modes.
#[wasm_bindgen]
pub fn stress_drag(dx: f64, dy: f64) {
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.drag(dx, dy);
        }
    });
}

/// Zoom by a wheel delta.
#[wasm_bindgen]
pub fn stress_zoom(delta_y: f64) {
    CAMERA.with_borrow_mut(|c| {
        if let Some(c) = c {
            c.zoom(delta_y);
        }
    });
}

/// Fire a projectile through viewport point (u, v in 0..1 from the top left), by the lab's
/// rule (`plan::shot_velocity`): launched into the world at the next frame. False when
/// all shots are fired.
#[wasm_bindgen]
pub fn stress_shoot(u: f64, v: f64) -> bool {
    let Some(s) = SHOOTING.get() else { return false };
    if FIRED.get() >= SHOTS {
        return false;
    }
    let (w, h) = SIZE.get();
    let Some(ray) = CAMERA.with_borrow(|c| c.as_ref().map(|c| c.aim(u, v, w as f64 / h as f64, s.radius))) else { return false };
    let vel = engine_fracture::plan::shot_velocity(s.center, s.extent, s.radius, ray.origin, ray.direction, s.speed, s.arc);
    let k = FIRED.get();
    FIRED.set(k + 1);
    PENDING.with_borrow_mut(|p| {
        p.push(ImpactorDesc {
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
    });
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
    // The lab's own scene (its camera, centre, extent and projectile) and the stress
    // oracle's input: the GLB's level, the cuboid houses in their layout, or the pack.
    let profile = PROFILE.get();
    let mut warnings = Vec::new();
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
                (lab, scene)
            }
        })
    })?;
    WARNINGS.set(warnings);
    CAMERA.set(Some(Orbit::new(&lab)));
    SHOOTING.set(Some(Shooting { center: lab.center, extent: lab.extent, radius: lab.projectile_radius, mass: lab.projectile_mass, speed: lab.projectile_speed, arc: lab.projectile_arc, placement: lab.placement }));
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
        (renderer.bond_graph, renderer.sleep_colors) = VIEW.get();
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

#[no_mangle]
pub extern "C" fn stress_frame(elapsed_ms: f64) -> u32 {
    match frame(elapsed_ms) {
        Ok(()) => 1,
        Err(e) => fail(e),
    }
}

fn frame(elapsed_ms: f64) -> Result<(), String> {
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
                (l.renderer.bond_graph, l.renderer.sleep_colors) = VIEW.get();
            }
            let dt = l.world.scene.sim.frame_dt;
            let mut stepped = 0;
            let mut step_ms = 0.0;
            if RUNNING.get() {
                l.behind = (l.behind + elapsed_ms * 1e-3).min(MAX_BEHIND_S.max(2.0 * dt));
                let shots = PENDING.with_borrow_mut(std::mem::take);
                for desc in &shots {
                    let imp = stress_ref::world::Impactor::from_desc(desc, &l.world.scene);
                    l.world.add_impactor(gpu, imp)?;
                }
                while l.behind >= dt {
                    let t = web_time::Instant::now();
                    l.world.step_frame(gpu)?;
                    step_ms += t.elapsed().as_secs_f64() * 1e3;
                    l.behind -= dt;
                    stepped += 1;
                }
            }
            let aspect = width as f64 / height as f64;
            let (vp, eye) = CAMERA.with_borrow(|c| {
                let c = c.as_ref().expect("a camera with the world");
                (c.view_proj(aspect), c.eye(aspect))
            });
            // The lab's camera looks at its own frame (the asset, +Y up, moved by
            // `placement`); the renderer draws the stress scene's (+Z up): view_proj times
            // the map from the stress frame into the lab's, column by column.
            let p = SHOOTING.get().map_or([0.0; 3], |s| s.placement);
            let col = |k: usize| -> [f32; 4] { std::array::from_fn(|r| vp[4 * k + r]) };
            let (c0, c1, c2, c3) = (col(0), col(1), col(2), col(3));
            let moved: [f32; 4] = std::array::from_fn(|r| p[0] as f32 * c0[r] + p[1] as f32 * c1[r] + p[2] as f32 * c2[r] + c3[r]);
            let view_proj = [c0, c2.map(|x| -x), c1, moved];
            let eye = Vec3::from_array(zup(std::array::from_fn(|k| eye[k] - p[k])));
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
            (l.renderer.bond_graph, l.renderer.sleep_colors) = VIEW.get();
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
                        "gpuWaits": engine_gpu_wait::WAITS.load(std::sync::atomic::Ordering::Relaxed),
                        "gpuYields": engine_gpu_wait::YIELDS.load(std::sync::atomic::Ordering::Relaxed),
                        "ballMass": SHOOTING.get().map_or(0.0, |s| s.mass),
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
