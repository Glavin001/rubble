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
    static GLB: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static CAMERA: RefCell<Option<Orbit>> = const { RefCell::new(None) };
    static SHOOTING: Cell<Option<Shooting>> = const { Cell::new(None) };
    static PENDING: RefCell<Vec<ImpactorDesc>> = const { RefCell::new(Vec::new()) };
    static FIRED: Cell<usize> = const { Cell::new(0) };
    /// The shots' mass as a multiple of the lab's default ball (`stress_set_shot_mass_scale`).
    static MASS_SCALE: Cell<f64> = const { Cell::new(1.0) };
    /// The bonds' strengths as a multiple of the asset's (`stress_set_bond_strength_scale`).
    static STRENGTH_SCALE: Cell<f64> = const { Cell::new(1.0) };
    static RUNNING: Cell<bool> = const { Cell::new(false) };
    static SIZE: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
    static STATS: RefCell<String> = RefCell::new("{}".into());
    /// The last frame's step, state copy and draw (ms).
    static PHASES: Cell<[f64; 3]> = const { Cell::new([0.0; 3]) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Asset frame (+Y up) to the stress scene's (+Z up), as `stress_ref::shardwright` imports.
fn zup(v: [f64; 3]) -> [f64; 3] {
    [v[0], -v[2], v[1]]
}

/// The GPU device on `canvas` (WebGPU), its surface configured at the canvas size.
#[wasm_bindgen]
pub async fn stress_init(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&info.to_string().into())));
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
    GLB.set(bytes);
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

/// The last frame's step, state copy (`view_state`) and draw, in ms.
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

/// Build the world from the GLB ([`stress_set_glb`]) at `level`. Called through
/// `WebAssembly.promising` (it reads the GPU). Returns 1, or 0 with [`stress_error`].
#[no_mangle]
pub extern "C" fn stress_load(level: u32) -> u32 {
    match load(level as u8) {
        Ok(()) => 1,
        Err(e) => fail(e),
    }
}

fn load(level: u8) -> Result<(), String> {
    WORLD.with_borrow_mut(|w| *w = None);
    let bytes = GLB.with_borrow(|b| b.clone());
    let asset = engine_fracture::parse_fracture_glb(&bytes).map_err(|e| format!("fracture GLB: {e}"))?;
    // The lab's own scene of the level: its camera, centre, extent and projectile.
    let lab = engine_fracture::build_scene(&asset, level, None).map_err(|e| format!("level {level}: {e}"))?;
    CAMERA.set(Some(Orbit::new(&lab)));
    SHOOTING.set(Some(Shooting { center: lab.center, extent: lab.extent, radius: lab.projectile_radius, mass: lab.projectile_mass, speed: lab.projectile_speed, arc: lab.projectile_arc, placement: lab.placement }));
    FIRED.set(0);
    RUNNING.set(false);
    PENDING.with_borrow_mut(|p| p.clear());
    let json = engine_fracture::stress_export::shardwright_stress_json(&asset, level);
    let mut scene = stress_ref::shardwright::import_value(&json, &asset.physics.name)?;
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
    scene.validate()?;
    let mut scene = scene.with_derived();
    // The GPU's methods, as the native runs (`STRESS_METHODS=established`).
    scene.sim.methods = stress_ref::scene::Methods::established();
    let profile = Profile::BlastParity;
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
        let (mirror, _) = world.view_state(gpu);
        // Meshes for every chunk and for the shots to come.
        let renderer = Renderer::for_target(gpu, &mirror, &shot_meshes(&world), d.config.width, d.config.height, d.view_format);
        WORLD.with_borrow_mut(|w| *w = Some(Loaded { world, renderer, behind: 0.0 }));
        Ok(())
    })
}

/// One animation frame `elapsed_ms` after the last: the frames of simulated time that
/// fall due while running (at most two, so a slow frame does not snowball), then the
/// world drawn into the canvas. Called through `WebAssembly.promising`. Returns 1, or 0
/// with [`stress_error`].
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
                let (mirror, _) = l.world.view_state(gpu);
                l.renderer = Renderer::for_target(gpu, &mirror, &shot_meshes(&l.world), width, height, d.view_format);
            }
            let dt = l.world.scene.sim.frame_dt;
            let mut stepped = 0;
            let mut step_ms = 0.0;
            if RUNNING.get() {
                l.behind = (l.behind + elapsed_ms * 1e-3).min(2.0 * dt);
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
            let t_view = web_time::Instant::now();
            let (mirror, imps) = l.world.view_state(gpu);
            let view_ms = t_view.elapsed().as_secs_f64() * 1e3;
            let t_draw = web_time::Instant::now();
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
            let texture = match d.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                _ => return Ok(()),
            };
            let view = texture.texture.create_view(&wgpu::TextureViewDescriptor { format: Some(d.view_format), ..Default::default() });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            l.renderer.draw(gpu, &mut encoder, &view, &mirror, &imps, view_proj, eye);
            gpu.queue.submit([encoder.finish()]);
            gpu.queue.present(texture);
            let draw_ms = t_draw.elapsed().as_secs_f64() * 1e3;
            PHASES.set([step_ms, view_ms, draw_ms]);
            if stepped > 0 || STATS.with_borrow(|s| s == "{}") {
                let o = l.world.observation();
                let moving = mirror.clusters.iter().filter(|c| !c.anchored).count();
                STATS.set(
                    serde_json::json!({
                        "frames": l.world.frame,
                        "simSeconds": mirror.time,
                        "stepMs": if stepped > 0 { step_ms / stepped as f64 } else { 0.0 },
                        "fragments": o.values.get("fragments").copied().unwrap_or(0.0),
                        "brokenBonds": o.values.get("broken_bonds").copied().unwrap_or(0.0),
                        "moving": moving,
                        "chunks": mirror.structures.iter().map(|s| s.chunks.len()).sum::<usize>(),
                        "hulls": mirror.structures.iter().flat_map(|s| &s.chunks).map(|c| c.parts.as_ref().map_or(1, |p| p.len())).sum::<usize>(),
                        "bonds": mirror.structures.iter().map(|s| s.bonds.len()).sum::<usize>(),
                        "shots": FIRED.get(),
                        "gpuWaits": engine_gpu_wait::WAITS.load(std::sync::atomic::Ordering::Relaxed),
                        "gpuYields": engine_gpu_wait::YIELDS.load(std::sync::atomic::Ordering::Relaxed),
                        "ballMass": SHOOTING.get().map_or(0.0, |s| s.mass),
                        "lastShot": imps.last().map(|i| serde_json::json!({"position": i.pose.position.to_array(), "velocity": i.velocity.to_array()})),
                        "frameHz": 1.0 / dt,
                    })
                    .to_string(),
                );
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
