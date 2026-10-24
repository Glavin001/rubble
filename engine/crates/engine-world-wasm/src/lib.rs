//! Rubble's world API in the browser for the showcase (`engine-webgpu/showcase.html`): a
//! scenario (`rubble::scenario`: a scene pack with driven vehicles as one `World`, its
//! structure a `Threshold` destructible) stepped here on this module's WebGPU device, and
//! each step's poses handed to the page, which draws them with the fracture lab's renderer
//! (`fracture/render.mjs`). The page builds the structure's render scene from the same pack
//! with engine-fracture (`buildPackScene`: its camera and floor); this module gives the
//! meshes ([`showcase_vertices`]: that scene's, coloured by material, then the vehicles'
//! parts and the balls) and every step's poses ([`showcase_poses`]).
//!
//! The render scene stands in the pack's asset frame (the lab's: the pack's frame plus its
//! placement); the world in the pack's own (the structure placed at -placement, as the lab
//! places it). So the asset frame is the world's moved by `T0^-1` (`T0` the structure's
//! transform), and a chunk at rest pose `R` (its render body's) is at `T0^-1 T R` while its
//! piece is at `T` (the render contract: the piece's origin is the instance frame).
//!
//! Exports that wait for the GPU (`showcase_load`, `showcase_frame`) suspend the wasm stack
//! (`engine-gpu-wait`): the page calls them through `WebAssembly.promising` (JSPI), as the
//! lab's stress mode does (`fracture/gpu-wait.mjs`).

use rubble::scenario::Scenario;
use rubble::{GpuContext, Transform};
use std::cell::RefCell;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};
use stress_rubble::demolition::Demolition;
use wasm_bindgen::prelude::*;

/// Floats a body takes in the draw shader's pose buffer (`fracture_draw.slang` `BodyPose`:
/// seven float4; it reads the position and the rotation).
const POSE_FLOATS: usize = 28;
/// Floats a render vertex takes (position, normal, colour, body id).
const VERTEX_FLOATS: usize = 10;
/// Colours (as the lab's stress renderer's materials, stress-rubble `material_color`).
const BODY: [f32; 3] = [0.55, 0.06, 0.05];
const TYRE: [f32; 3] = [0.06, 0.06, 0.07];
const BALL: [f32; 3] = [0.30, 0.31, 0.34];
/// Where a ball not yet fired waits (far under the ground).
const PARKED: [f32; 3] = [0.0, -1.0e4, 0.0];

struct Loaded {
    scenario: Scenario,
    demolition: Demolition,
    /// The render scene's bodies at rest (its pose buffer as built): ground, the pack's
    /// nodes in order, its projectiles.
    rest: Vec<Transform>,
    /// The asset frame from the world's: `T0^-1`.
    to_render: Transform,
    /// The pack's ball (`defaults.projectile`) and its line of attack (`wreckingYard`).
    ball: (f64, f64, f64),
    aim: Option<stress_ref::wrecking::Aim>,
    /// Shots asked for since the last frame (stress frame, Z-up).
    pending: Vec<ImpactorDesc>,
    /// The render scene's vertices, each chunk coloured by its material.
    structure: Vec<f32>,
}

thread_local! {
    static GPU: RefCell<Option<stress_gpu::gpu::Gpu>> = const { RefCell::new(None) };
    static LOADED: RefCell<Option<Loaded>> = const { RefCell::new(None) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// The scenario the next `showcase_load` builds (`showcase_select`).
    static SELECTED: RefCell<String> = const { RefCell::new(String::new()) };
}
static PANIC: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn fail(e: impl std::fmt::Display) -> bool {
    ERROR.set(e.to_string());
    false
}

fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (glam::Quat::from_array(a.rotation), glam::Vec3::from(a.position));
    let (qb, pb) = (glam::Quat::from_array(b.rotation), glam::Vec3::from(b.position));
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}

fn inverse(t: &Transform) -> Transform {
    let q = glam::Quat::from_array(t.rotation).inverse();
    Transform::new((-(q * glam::Vec3::from(t.position))).to_array(), q.to_array())
}

/// Open this module's WebGPU device (the browser's adapter, no canvas: the page draws).
#[wasm_bindgen]
pub async fn showcase_init() -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&info.to_string().into());
        if let Ok(mut last) = PANIC.try_lock() {
            *last = info.to_string();
        }
    }));
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let gpu = stress_gpu::gpu::Gpu::with_instance(&instance, None).await.map_err(|e| JsValue::from_str(&e))?;
    GPU.set(Some(gpu));
    Ok(())
}

/// The last panic's message (a panic traps the export: "unreachable").
#[wasm_bindgen]
pub fn showcase_panic() -> String {
    PANIC.lock().map(|p| p.clone()).unwrap_or_default()
}

/// Why the last call that returned false failed.
#[wasm_bindgen]
pub fn showcase_error() -> String {
    ERROR.with_borrow(|e| e.clone())
}

/// The scenarios: the bundled scene packs with drives, as JSON `[{key, group, menu}]` in
/// the lab's menu order.
#[wasm_bindgen]
pub fn showcase_scenarios() -> String {
    let list: Vec<serde_json::Value> = engine_fracture::samples::PACK_SAMPLES
        .iter()
        .filter(|s| {
            engine_fracture::samples::pack_sample_json(s.key)
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .is_some_and(|p| p["defaults"]["drives"].as_array().is_some_and(|d| !d.is_empty()))
        })
        .map(|s| serde_json::json!({"key": s.key, "group": s.group, "menu": s.menu}))
        .collect();
    serde_json::Value::from(list).to_string()
}

/// A scenario's pack (its JSON bytes): the page builds its render scene from it.
#[wasm_bindgen]
pub fn showcase_pack(key: &str) -> Result<Vec<u8>, JsValue> {
    engine_fracture::samples::pack_sample_json(key).map_err(|e| JsValue::from_str(&format!("{e:#}")))
}

/// The scenario the next [`showcase_load`] builds. (A waiting export takes no string:
/// `WebAssembly.promising` calls the raw export, past wasm-bindgen's argument glue.)
#[wasm_bindgen]
pub fn showcase_select(key: &str) {
    SELECTED.set(key.to_string());
}

/// Load the selected scenario ([`showcase_select`]): its world on this module's device.
/// Waits for the GPU. False on an error (`showcase_error`).
#[wasm_bindgen]
pub fn showcase_load() -> bool {
    LOADED.set(None);
    let key = SELECTED.with_borrow(|k| k.clone());
    match load(&key) {
        Ok(l) => {
            LOADED.set(Some(l));
            true
        }
        Err(e) => fail(format!("{key}: {e:#}")),
    }
}

fn load(key: &str) -> anyhow::Result<Loaded> {
    let pack = engine_fracture::samples::pack_sample_json(key)?;
    let context = GPU.with_borrow(|g| {
        g.as_ref().map(|g| GpuContext::Shared { info: g.adapter_info.clone(), device: g.device.clone(), queue: g.queue.clone() })
    });
    let scenario = Scenario::load(context.ok_or_else(|| anyhow::anyhow!("showcase_init first"))?, &pack)?;
    let (scene, _) = engine_fracture::pack::build_pack_scene(&pack)?;
    let f = |v: [f64; 3]| v.map(|x| x as f32);
    let rest = scene.bodies.iter().map(|b| Transform::new(f(b.p), b.q.map(|x| x as f32))).collect();
    let json: serde_json::Value = serde_json::from_slice(&pack)?;
    // Each chunk (render body 1 + node) its material's colour, as the lab's stress renderer
    // draws it, a little lighter or darker by chunk so the pieces read.
    let materials: Vec<&str> = json["scenario"]["nodeMaterials"].as_array().map_or(Vec::new(), |m| m.iter().map(|v| v.as_str().unwrap_or("")).collect());
    let mut structure = scene.vertices.clone();
    for v in structure.chunks_exact_mut(VERTEX_FLOATS) {
        let id = v[9] as usize;
        if let Some(m) = id.checked_sub(1).and_then(|k| materials.get(k)) {
            let h = (id as u32).wrapping_mul(2654435761) >> 24;
            let k = 0.94 + 0.12 * (h as f32 / 255.0);
            let c = stress_rubble::render::material_color(m);
            v[6..9].copy_from_slice(&c.map(|x| (x * k).min(1.0)));
        }
    }
    let p = &json["defaults"]["projectile"];
    let ball = (p["radius"].as_f64().unwrap_or(0.5), p["mass"].as_f64().unwrap_or(1000.0), p["speed"].as_f64().unwrap_or(20.0));
    let to_render = inverse(&scenario.instance);
    Ok(Loaded { demolition: Demolition::new(scenario.charges.clone()), scenario, rest, to_render, ball, aim: stress_ref::wrecking::aim_of(&json), pending: Vec::new(), structure })
}

/// The first body the extras take in the pose buffer: after the render scene's bodies.
#[wasm_bindgen]
pub fn showcase_body_base() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.rest.len() as u32))
}

/// Every mesh as render vertices (10 floats each, body-local): the render scene's chunks
/// (coloured by material), then the extras (body ids from [`showcase_body_base`] on): each
/// car's boxes, then its wheels (axle along x), then a ball per shot the scenario takes
/// ([`Scenario::MAX_BALLS`]).
#[wasm_bindgen]
pub fn showcase_vertices() -> Vec<f32> {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else { return Vec::new() };
        let mut out = l.structure.clone();
        let mut id = l.rest.len() as f32;
        for car in &l.scenario.cars {
            for (_, half) in &car.boxes {
                box_vertices(*half, BODY, id, &mut out);
                id += 1.0;
            }
            for &(radius, half_width, _) in &car.wheels {
                cylinder_vertices(radius, half_width, 24, TYRE, id, &mut out);
                id += 1.0;
            }
        }
        for _ in 0..Scenario::MAX_BALLS {
            sphere_vertices(l.ball.0 as f32, 16, BALL, id, &mut out);
            id += 1.0;
        }
        out
    })
}

/// Advance `steps` steps (shots asked for and charges due fired between steps), then wait
/// for their readbacks. Waits for the GPU: the scenario is taken out of its slot while
/// this runs (no borrow is held across a suspension), so a call that reaches it meanwhile
/// finds none. False on an error (`showcase_error`).
#[wasm_bindgen]
pub fn showcase_frame(steps: u32) -> bool {
    let Some(mut l) = LOADED.take() else { return fail("no scenario loaded") };
    let r = frame(&mut l, steps);
    LOADED.set(Some(l));
    match r {
        Ok(()) => true,
        Err(e) => fail(format!("{e:#}")),
    }
}

fn frame(l: &mut Loaded, steps: u32) -> anyhow::Result<()> {
    for _ in 0..steps {
        for desc in std::mem::take(&mut l.pending) {
            l.scenario.shoot(&desc)?;
        }
        for c in l.demolition.due(l.scenario.frame) {
            l.scenario.cut(&c)?;
        }
        l.scenario.step()?;
    }
    if steps > 0 {
        l.scenario.poll()?;
    }
    Ok(())
}

/// The pose buffer as of the last polled step (`POSE_FLOATS` floats a body, the draw
/// shader's layout): the render scene's bodies (the ground and the projectiles at rest,
/// each chunk where its piece is), then the extras ([`showcase_extra_vertices`]' order;
/// balls not fired parked under the ground).
#[wasm_bindgen]
pub fn showcase_poses() -> Vec<f32> {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else { return Vec::new() };
        let s = &l.scenario;
        let mut poses: Vec<Transform> = l.rest.clone();
        // The pack's nodes are the render scene's bodies 1..=n, in order.
        let to_world = s.instance;
        for (chunk, rest) in poses.iter_mut().skip(1).enumerate() {
            let Ok(motion) = s.chunk_motion(chunk as u32) else { break };
            // T0^-1 (T T0^-1) T0 R: the chunk's motion since rest, in the asset frame.
            *rest = compose(&l.to_render, &compose(&motion, &compose(&to_world, rest)));
        }
        for parts in s.car_poses() {
            poses.extend(parts.iter().map(|t| compose(&l.to_render, t)));
        }
        let balls = s.ball_poses();
        for k in 0..Scenario::MAX_BALLS {
            poses.push(balls.get(k).map_or(Transform::new(PARKED, [0.0, 0.0, 0.0, 1.0]), |(t, _)| compose(&l.to_render, t)));
        }
        let mut out = vec![0.0f32; POSE_FLOATS * poses.len()];
        for (k, t) in poses.iter().enumerate() {
            out[POSE_FLOATS * k..POSE_FLOATS * k + 3].copy_from_slice(&t.position);
            out[POSE_FLOATS * k + 4..POSE_FLOATS * k + 8].copy_from_slice(&t.rotation);
        }
        out
    })
}

/// Fire the pack's ball along a ray in the render scene's frame (the camera's aim): from
/// `origin` along `direction` at the pack's speed. False when the scenario has taken its
/// shots or none is loaded.
#[wasm_bindgen]
pub fn showcase_shoot_ray(origin: &[f64], direction: &[f64]) -> bool {
    LOADED.with_borrow_mut(|l| {
        let Some(l) = l.as_mut() else { return false };
        if origin.len() != 3 || direction.len() != 3 || l.scenario.balls.len() + l.pending.len() >= Scenario::MAX_BALLS {
            return false;
        }
        let (radius, mass, speed) = l.ball;
        // The asset frame (Y-up) to the world's, then to the stress frame (Z-up) shots use.
        let at = compose(&l.scenario.instance, &Transform::new(origin.iter().map(|&x| x as f32).collect::<Vec<_>>().try_into().unwrap(), [0.0, 0.0, 0.0, 1.0])).position;
        let d = glam::DVec3::new(direction[0], direction[1], direction[2]).normalize_or_zero() * speed;
        let zup = |v: [f64; 3]| [v[0], -v[2], v[1]];
        l.pending.push(ImpactorDesc {
            name: format!("shot{}", l.scenario.balls.len() + l.pending.len()),
            shape: ImpactorShape::Sphere { radius },
            mass,
            position: zup(at.map(|x| x as f64)),
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: zup(d.to_array()),
            angular_velocity: [0.0; 3],
            material: "projectile".into(),
            crush: None,
        });
        true
    })
}

/// Fire the pack's ball along its line of attack (`defaults.wreckingYard`), as the lab's
/// stress modes do. False without one.
#[wasm_bindgen]
pub fn showcase_fire() -> bool {
    LOADED.with_borrow_mut(|l| {
        let Some(l) = l.as_mut() else { return false };
        let Some(aim) = l.aim.clone() else { return false };
        if l.scenario.balls.len() + l.pending.len() >= Scenario::MAX_BALLS {
            return false;
        }
        let (radius, mass, speed) = l.ball;
        let ball = stress_ref::wrecking::Ball { key: "", menu: "", mass, radius, speed, rolling: false };
        let name = format!("shot{}", l.scenario.balls.len() + l.pending.len());
        l.pending.push(stress_ref::wrecking::launch(&ball, &aim, &name));
        true
    })
}

/// The scenario's charges as JSON `[{name, delay, bonds}]`, in firing order.
#[wasm_bindgen]
pub fn showcase_charges() -> String {
    LOADED.with_borrow(|l| {
        let list: Vec<serde_json::Value> = l.as_ref().map_or(Vec::new(), |l| l.demolition.charges.iter().map(|c| serde_json::json!({"name": c.name, "delay": c.delay, "bonds": c.bonds.len()})).collect());
        serde_json::Value::from(list).to_string()
    })
}

/// Set off charge `name`, or with an empty name every charge, each at its delay: cut
/// between two steps before the next. False when there is no such charge.
#[wasm_bindgen]
pub fn showcase_detonate(name: &str) -> bool {
    LOADED.with_borrow_mut(|l| {
        let Some(l) = l.as_mut() else { return false };
        l.demolition.arm(name, l.scenario.frame, l.scenario.dt)
    })
}

/// The scenario's state as JSON: steps, simulated time, the step rate, bonds broken,
/// bodies, awake bodies, contacts, shots, charges pending, and each drive's name and speed.
#[wasm_bindgen]
pub fn showcase_stats() -> String {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else { return "{}".into() };
        let s = &l.scenario;
        let d = s.world.destructible_stats(s.destructible).ok();
        serde_json::json!({
            "frames": s.frame,
            "simSeconds": s.frame as f64 * s.dt,
            "frameHz": 1.0 / s.dt,
            "brokenBonds": s.broken,
            "bonds": d.map_or(0, |d| d.bonds),
            "chunks": d.map_or(0, |d| d.chunks),
            "bodies": d.map_or(0, |d| d.actors),
            "awake": s.awake,
            "contacts": s.contacts,
            "shots": s.balls.len() + l.pending.len(),
            "canFire": l.aim.is_some() && s.balls.len() + l.pending.len() < Scenario::MAX_BALLS,
            "charges": l.demolition.names(),
            "chargesArmed": l.demolition.pending(),
            "drives": s.cars.iter().map(|c| c.drive.name.clone()).collect::<Vec<_>>(),
            "driveKmh": s.speed_kmh(),
            // The render frame less the world's (Y-up): a take's camera targets (`at`).
            "placement": l.to_render.position,
        })
        .to_string()
    })
}

/// A box's faces (`half` extents), its body `id`.
fn box_vertices(half: [f32; 3], color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let corner = |a: f32, b: f32| {
                let mut p = [0.0f32; 3];
                p[axis] = sign * half[axis];
                p[u] = a * half[u];
                p[v] = b * half[v];
                p
            };
            let mut n = [0.0f32; 3];
            n[axis] = sign;
            let quad = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
            // Counter-clockwise seen from outside (the draw does not cull).
            let tri = if sign > 0.0 { [0, 1, 2, 0, 2, 3] } else { [0, 2, 1, 0, 3, 2] };
            for k in tri {
                vertex(quad[k], n, color, id, out);
            }
        }
    }
}

/// A wheel's faces: a cylinder of `radius` and half width along x, `segs` sides.
fn cylinder_vertices(radius: f32, half_width: f32, segs: usize, color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    let at = |k: usize| {
        let a = std::f32::consts::TAU * k as f32 / segs as f32;
        (a.cos(), a.sin())
    };
    for k in 0..segs {
        let ((c0, s0), (c1, s1)) = (at(k), at(k + 1));
        let p = |x: f32, c: f32, s: f32| [x, radius * c, radius * s];
        let (a, b, c, d) = (p(-half_width, c0, s0), p(half_width, c0, s0), p(half_width, c1, s1), p(-half_width, c1, s1));
        let n0 = [0.0, c0, s0];
        let n1 = [0.0, c1, s1];
        for (q, n) in [(a, n0), (b, n0), (c, n1), (a, n0), (c, n1), (d, n1)] {
            vertex(q, n, color, id, out);
        }
        for x in [-half_width, half_width] {
            let n = [x.signum(), 0.0, 0.0];
            for q in [[x, 0.0, 0.0], p(x, c0, s0), p(x, c1, s1)] {
                vertex(q, n, color, id, out);
            }
        }
    }
}

/// A ball's faces: a sphere of `radius`, `rings` rings of twice as many segments.
fn sphere_vertices(radius: f32, rings: usize, color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    let segs = 2 * rings;
    let n = |i: usize, j: usize| {
        let (th, ph) = (std::f32::consts::PI * i as f32 / rings as f32, std::f32::consts::TAU * j as f32 / segs as f32);
        [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()]
    };
    for i in 0..rings {
        for j in 0..segs {
            let q = [n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)];
            for k in [0, 1, 2, 0, 2, 3] {
                vertex(q[k].map(|x| x * radius), q[k], color, id, out);
            }
        }
    }
}

fn vertex(p: [f32; 3], n: [f32; 3], color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    out.extend_from_slice(&p);
    out.extend_from_slice(&n);
    out.extend_from_slice(&color);
    out.push(id);
    debug_assert_eq!(out.len() % VERTEX_FLOATS, 0);
}

#[cfg(test)]
mod tests {
    /// The render frame's bookkeeping (natively): at rest every chunk's pose is its render
    /// body's, and the vehicles stand where the pack puts them in the asset frame.
    #[test]
    fn rest_poses_are_the_render_scene() {
        let pack = engine_fracture::samples::pack_sample_json("rr_drive_storefront").unwrap();
        let s = rubble::scenario::Scenario::load(rubble::GpuContext::Default, &pack).unwrap();
        let (scene, _) = engine_fracture::pack::build_pack_scene(&pack).unwrap();
        let to_render = super::inverse(&s.instance);
        for chunk in [0u32, 100, 600] {
            let motion = s.chunk_motion(chunk).unwrap();
            assert!(glam::Vec3::from(motion.position).length() < 1e-4, "{motion:?}");
        }
        // The sedan's drive point is the ground under it, in the pack's frame; drawn in the
        // asset frame it is shifted by the placement, as every chunk is.
        let json: serde_json::Value = serde_json::from_slice(&pack).unwrap();
        let drives = stress_ref::drive::drives_of(&json).unwrap();
        let car = super::compose(&to_render, &s.car_poses()[0][0]);
        let at = drives[0].at;
        // x and the street's direction: the asset frame is the world's less T0's translation.
        assert!((car.position[0] - at[0] as f32 - to_render.position[0]).abs() < 0.2, "{car:?} {at:?}");
        assert!(scene.bodies.len() > 1000);
    }
}
