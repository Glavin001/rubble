//! Rubble's world API in the browser: Broadside (`engine-webgpu/broadside.html`). The scene
//! ([`broadside`]: the Shardwright galleon stern as a threshold destructible, the captain,
//! the broadsides and the closing volley) is stepped here on this module's WebGPU device; each
//! frame's poses go to the page, which draws them with the fracture lab's renderer
//! (`fracture/render.mjs`) over the render scene it builds from the same GLB
//! (`fracture/wasm.mjs` `buildScene`, the same `engine_fracture::build_scene`).
//!
//! The pose buffer (the draw shader's `BodyPose`, [`POSE_FLOATS`] floats a body): the render
//! scene's bodies (the ground, every chunk in order, its parked projectiles), then the
//! extras the page draws as a dynamic mesh ([`broadside_extra_vertices`]): the captain,
//! [`broadside::MAX_BALLS`] balls, and the crush splinters so far.
//!
//! Exports that wait for the GPU (`broadside_load`, `broadside_frame`) suspend the wasm
//! stack (`engine-gpu-wait`): the page calls them through `WebAssembly.promising` (JSPI), as
//! the lab's stress mode does (`fracture/gpu-wait.mjs`).

pub mod broadside;

use broadside::{Broadside, Director, Take, MAX_BALLS};
use rubble::{GpuContext, Transform};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

/// Floats a body takes in the draw shader's pose buffer (`fracture_draw.slang` `BodyPose`:
/// seven float4; it reads the position and the rotation).
pub const POSE_FLOATS: usize = 28;
/// Floats a render vertex takes (position, normal, colour, body id).
pub const VERTEX_FLOATS: usize = 10;
/// Where a body not in play is drawn (far under the ground).
const PARKED: [f32; 3] = [0.0, -1.0e4, 0.0];
const COAT: [f32; 3] = [0.10, 0.12, 0.22];
const SKIN: [f32; 3] = [0.86, 0.68, 0.56];
const HAT: [f32; 3] = [0.05, 0.05, 0.06];
const BALL: [f32; 3] = [0.16, 0.16, 0.17];
/// Fresh-split oak, a crushed chunk's splinters.
const SPLINTER: [f32; 3] = [0.80, 0.64, 0.42];

struct Loaded {
    b: Broadside,
    director: Director,
    /// The render scene's body count (`build_scene`'s bodies): the extras follow.
    render_bodies: usize,
    curr: Vec<Transform>,
    prev: Option<Vec<Transform>>,
}

thread_local! {
    static GPU: RefCell<Option<stress_gpu::gpu::Gpu>> = const { RefCell::new(None) };
    static ASSET: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LEVEL: RefCell<u8> = const { RefCell::new(broadside::LEVEL) };
    static TAKE: RefCell<Take> = const { RefCell::new(Take::Walk) };
    static LOADED: RefCell<Option<Loaded>> = const { RefCell::new(None) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}
static PANIC: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
/// Why this module's device was lost, when it was (`broadside_lost`).
static LOST: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn fail(e: impl std::fmt::Display) -> bool {
    ERROR.set(e.to_string());
    false
}

/// Open this module's WebGPU device (the browser's adapter, no canvas: the page draws).
#[wasm_bindgen]
pub async fn broadside_init() -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&info.to_string().into());
        if let Ok(mut last) = PANIC.try_lock() {
            *last = info.to_string();
        }
    }));
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let gpu = stress_gpu::gpu::Gpu::with_instance(&instance, None)
        .await
        .map_err(|e| JsValue::from_str(&e))?;
    gpu.device.set_device_lost_callback(|reason, message| {
        if let Ok(mut lost) = LOST.lock() {
            *lost = format!("{reason:?}: {message}");
        }
    });
    GPU.set(Some(gpu));
    Ok(())
}

/// The last panic's message (a panic traps the export: "unreachable").
#[wasm_bindgen]
pub fn broadside_panic() -> String {
    PANIC.lock().map(|p| p.clone()).unwrap_or_default()
}

/// Why this module's WebGPU device was lost ("" while it is not): the browser's reason,
/// which a phone's missing console would otherwise hide.
#[wasm_bindgen]
pub fn broadside_lost() -> String {
    LOST.lock().map(|l| l.clone()).unwrap_or_default()
}

/// Why the last call that returned false failed.
#[wasm_bindgen]
pub fn broadside_error() -> String {
    ERROR.with_borrow(|e| e.clone())
}

/// The ship's baked GLB for the next [`broadside_load`]. (A waiting export takes no
/// arguments: `WebAssembly.promising` calls the raw export, past wasm-bindgen's glue.)
#[wasm_bindgen]
pub fn broadside_set_asset(glb: Vec<u8>) {
    ASSET.set(glb);
}

/// The fracture level the next [`broadside_load`] builds (`broadside::LEVEL`'s list).
#[wasm_bindgen]
pub fn broadside_set_level(level: u8) {
    LEVEL.set(level);
}

/// The script the next [`broadside_load`] runs: 0 the captain's walk, 1 raze the ship.
#[wasm_bindgen]
pub fn broadside_set_take(take: u8) {
    TAKE.set(if take == 1 { Take::Raze } else { Take::Walk });
}

/// Build the scene from the asset ([`broadside_set_asset`]) on this module's device.
/// Waits for the GPU. False on an error (`broadside_error`).
#[wasm_bindgen]
pub fn broadside_load() -> bool {
    LOADED.set(None);
    match load() {
        Ok(l) => {
            LOADED.set(Some(l));
            true
        }
        Err(e) => fail(format!("{e:#}")),
    }
}

fn load() -> anyhow::Result<Loaded> {
    let glb = ASSET.with_borrow(|a| a.clone());
    anyhow::ensure!(!glb.is_empty(), "broadside_set_asset first");
    let context = GPU.with_borrow(|g| {
        g.as_ref().map(|g| GpuContext::Shared {
            info: g.adapter_info.clone(),
            device: g.device.clone(),
            queue: g.queue.clone(),
        })
    });
    let level = LEVEL.with_borrow(|l| *l);
    let b = Broadside::with_take(
        context.ok_or_else(|| anyhow::anyhow!("broadside_init first"))?,
        &glb,
        level,
        TAKE.with_borrow(|t| *t),
    )?;
    let parsed = engine_fracture::parse_fracture_glb(&glb)?;
    let render_bodies = engine_fracture::build_scene(&parsed, level, None)?
        .bodies
        .len();
    let mut l = Loaded {
        b,
        director: Director::default(),
        render_bodies,
        curr: Vec::new(),
        prev: None,
    };
    l.curr = render_poses(&l);
    Ok(l)
}

/// The first extra body in the pose buffer (after the render scene's).
#[wasm_bindgen]
pub fn broadside_body_base() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.render_bodies as u32))
}

/// The extras' render vertices (body-local, [`VERTEX_FLOATS`] floats each): the captain,
/// the balls, the splinters so far. Redraw them when [`broadside_extras_version`] changes.
#[wasm_bindgen]
pub fn broadside_extra_vertices() -> Vec<f32> {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut id = l.render_bodies as f32;
        captain_vertices(id, &mut out);
        id += 1.0;
        for _ in 0..MAX_BALLS {
            sphere_vertices(broadside::BALL_RADIUS, 8, BALL, id, &mut out);
            id += 1.0;
        }
        let first = id;
        for (k, s) in l.b.splinters.iter().enumerate() {
            splinter_vertices(k as u32, s, first + k as f32, &mut out);
        }
        out
    })
}

/// Vertices each piece of debris takes ([`splinter_vertices`]): a box's 36.
const SPLINTER_VERTS: u32 = 36;

/// The vertex where debris `k`'s vertices start in [`broadside_extra_vertices`] (the
/// captain's and the balls' come first): the page appends new debris there.
#[wasm_bindgen]
pub fn broadside_splinter_vertex_offset(k: u32) -> u32 {
    let (mut captain, mut ball) = (Vec::new(), Vec::new());
    captain_vertices(0.0, &mut captain);
    sphere_vertices(broadside::BALL_RADIUS, 8, BALL, 0.0, &mut ball);
    let verts = |v: &Vec<f32>| (v.len() / VERTEX_FLOATS) as u32;
    verts(&captain) + verts(&ball) * MAX_BALLS as u32 + k * SPLINTER_VERTS
}

/// Debris so far (crush particles and fragments).
#[wasm_bindgen]
pub fn broadside_splinter_count() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.b.splinters.len() as u32))
}

/// The render vertices of debris `from` onward (append them at
/// [`broadside_splinter_vertex_offset`]`(from)`).
#[wasm_bindgen]
pub fn broadside_splinter_vertices(from: u32) -> Vec<f32> {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else {
            return Vec::new();
        };
        let first = (l.render_bodies + 1 + MAX_BALLS) as f32;
        let mut out = Vec::new();
        for (k, s) in l.b.splinters.iter().enumerate().skip(from as usize) {
            splinter_vertices(k as u32, s, first + k as f32, &mut out);
        }
        out
    })
}

/// Debris `k`'s vertices, all [`SPLINTER_VERTS`] of them: a crush particle as one thin
/// pointed sliver of fresh-split wood centred on its body (length, thickness and shade of
/// its own), a box fragment as its box (padded to the same count with degenerate triangles).
fn splinter_vertices(k: u32, s: &broadside::Splinter, id: f32, out: &mut Vec<f32>) {
    let h = |i: u32| {
        let x = k.wrapping_mul(0x9E37_79B1) ^ i.wrapping_mul(0x85EB_CA77);
        let x = (x ^ (x >> 15)).wrapping_mul(0x2C1B_3C6D);
        ((x ^ (x >> 13)) >> 8) as f32 / (1u32 << 24) as f32
    };
    let start = out.len();
    if s.particle {
        let r = s.half[0];
        let shade = 0.78 + 0.34 * h(0);
        let color = SPLINTER.map(|c| (c * shade).min(1.0));
        let (len, t) = (r * (2.2 + 2.0 * h(1)), r * (0.22 + 0.16 * h(2)));
        // A wedge: a point at one end, a small triangle at the other.
        let p = [
            glam::Vec3::new(0.5 * len, 0.0, 0.0),
            glam::Vec3::new(-0.5 * len, t, 0.0),
            glam::Vec3::new(-0.5 * len, -0.5 * t, 0.87 * t),
            glam::Vec3::new(-0.5 * len, -0.5 * t, -0.87 * t),
        ];
        let centre = (p[0] + p[1] + p[2] + p[3]) * 0.25;
        for [a, b, c] in [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]] {
            let mut n = (p[b] - p[a]).cross(p[c] - p[a]).normalize_or_zero();
            let (mut b, mut c) = (b, c);
            if n.dot(p[a] - centre) < 0.0 {
                n = -n;
                std::mem::swap(&mut b, &mut c);
            }
            for v in [a, b, c] {
                vertex((p[v] - centre).to_array(), n.to_array(), color, id, out);
            }
        }
    } else {
        let shade = 0.82 + 0.3 * h(0);
        box_vertices(s.half, SPLINTER.map(|c| (c * shade).min(1.0)), id, out);
    }
    // The same count for every piece (the page appends by index).
    let want = start + SPLINTER_VERTS as usize * VERTEX_FLOATS;
    while out.len() < want {
        vertex([0.0; 3], [0.0, 1.0, 0.0], SPLINTER, id, out);
    }
    out.truncate(want);
}

/// Changes whenever [`broadside_extra_vertices`] would.
#[wasm_bindgen]
pub fn broadside_extras_version() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.b.splinters_version))
}

/// Advance `steps` steps, then wait for their readbacks. Waits for the GPU: the scene is
/// taken out of its slot while this runs (no borrow is held across a suspension). False
/// on an error (`broadside_error`).
#[wasm_bindgen]
pub fn broadside_frame(steps: u32) -> bool {
    let Some(mut l) = LOADED.take() else {
        return fail("no scene loaded");
    };
    let r = frame(&mut l, steps);
    LOADED.set(Some(l));
    match r {
        Ok(()) => true,
        Err(e) => fail(format!("{e:#}")),
    }
}

fn frame(l: &mut Loaded, steps: u32) -> anyhow::Result<()> {
    for i in 0..steps {
        l.b.step()?;
        // The poses a step before the last, to draw between (several steps a frame).
        if i + 2 == steps {
            l.b.poll_blocking()?;
            l.curr = render_poses(l);
        }
    }
    if steps > 0 {
        l.b.poll_blocking()?;
        let now = render_poses(l);
        l.prev = Some(std::mem::replace(&mut l.curr, now));
    }
    Ok(())
}

/// The pose buffer at `alpha` (0..1) of the way from the step before the last polled one
/// to it: a frame between two steps (slow motion) draws the bodies between them. Render
/// only: the physics never sees it.
#[wasm_bindgen]
pub fn broadside_poses(alpha: f32) -> Vec<f32> {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else {
            return Vec::new();
        };
        let a = alpha.clamp(0.0, 1.0);
        let mut out = vec![0.0f32; POSE_FLOATS * l.curr.len()];
        for (k, t) in l.curr.iter().enumerate() {
            let t = match l.prev.as_ref().and_then(|p| p.get(k)) {
                Some(p) if p.position != PARKED && t.position != PARKED => between(p, t, a),
                _ => *t,
            };
            out[POSE_FLOATS * k..POSE_FLOATS * k + 3].copy_from_slice(&t.position);
            out[POSE_FLOATS * k + 4..POSE_FLOATS * k + 8].copy_from_slice(&t.rotation);
        }
        out
    })
}

fn between(p: &Transform, q: &Transform, a: f32) -> Transform {
    let x = glam::Vec3::from(p.position).lerp(glam::Vec3::from(q.position), a);
    let r = glam::Quat::from_array(p.rotation).slerp(glam::Quat::from_array(q.rotation), a);
    Transform::new(x.to_array(), r.normalize().to_array())
}

fn parked() -> Transform {
    Transform::new(PARKED, [0.0, 0.0, 0.0, 1.0])
}

/// Every drawn body's pose as of the last polled step (the module notes' order).
fn render_poses(l: &Loaded) -> Vec<Transform> {
    let b = &l.b;
    let mut poses = vec![Transform::IDENTITY];
    for (c, t) in b.chunk_poses().into_iter().enumerate() {
        poses.push(if b.crushed[c] { parked() } else { t });
    }
    poses.resize(l.render_bodies, parked());
    // The captain is a scripted point for now (no figure drawn).
    poses.push(parked());
    let balls = b.ball_poses();
    for k in 0..MAX_BALLS {
        poses.push(balls.get(k).copied().unwrap_or_else(parked));
    }
    for (k, s) in b.splinters.iter().enumerate() {
        let Some(t) = b.world.body_transform(s.body).ok().flatten() else {
            poses.push(parked());
            continue;
        };
        poses.push(Transform::new(t.position, tumble(k as u32, s, t).to_array()));
    }
    poses
}

/// A splinter's drawn turn: a crush particle never rotates, so the page tumbles its
/// splinter about an axis of its own by how far it has travelled (render only; it
/// settles when the particle does). A box fragment keeps its body's rotation.
fn tumble(k: u32, s: &broadside::Splinter, t: Transform) -> glam::Quat {
    let body = glam::Quat::from_array(t.rotation);
    if !s.particle {
        return body;
    }
    let (half, at) = (s.half, s.at);
    let h = |s: u32| {
        let x = k.wrapping_mul(0x9E37_79B1) ^ s.wrapping_mul(0x85EB_CA77);
        ((x ^ (x >> 15)).wrapping_mul(0x2C1B_3C6D) >> 8) as f32 / (1u32 << 24) as f32
    };
    let axis = glam::Vec3::new(h(1) - 0.5, h(2) - 0.5, h(3) - 0.5).normalize_or(glam::Vec3::Y);
    let start = glam::Quat::from_euler(glam::EulerRot::XYZ, 6.28 * h(4), 6.28 * h(5), 6.28 * h(6));
    let travelled = (glam::Vec3::from(t.position) - glam::Vec3::from(at)).length();
    glam::Quat::from_axis_angle(axis, travelled / half[0].max(0.005) * 0.3) * start
}

/// The take's camera for a frame `wall` seconds after the last: `[eye, target]`.
#[wasm_bindgen]
pub fn broadside_camera(wall: f32) -> Vec<f32> {
    LOADED.with_borrow_mut(|l| {
        let Some(l) = l.as_mut() else {
            return Vec::new();
        };
        let (e, t) = l.director.camera(&l.b, wall);
        [e, t].concat()
    })
}

/// Simulated seconds per wall second the take plays at now.
#[wasm_bindgen]
pub fn broadside_time_scale() -> f32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(1.0, |l| Director::time_scale(&l.b)))
}

/// Free play: stop the script (barrage, walk, finale) from here on, or start it again.
#[wasm_bindgen]
pub fn broadside_set_scripted(on: bool) {
    LOADED.with_borrow_mut(|l| {
        if let Some(l) = l.as_mut() {
            l.b.scripted = on;
        }
    })
}

/// Fire a ball from `origin` along `direction` (world, the camera's aim) at the take's
/// speed, before the next step.
#[wasm_bindgen]
pub fn broadside_shoot_ray(origin: &[f32], direction: &[f32]) -> bool {
    LOADED.with_borrow_mut(|l| {
        let Some(l) = l.as_mut() else { return false };
        if origin.len() != 3 || direction.len() != 3 {
            return false;
        }
        let d = glam::Vec3::from_slice(direction).normalize_or_zero() * broadside::BALL_SPEED;
        let from = glam::Vec3::from_slice(origin) + d.normalize_or_zero() * 1.0;
        l.b.shoot(from.to_array(), d.to_array());
        true
    })
}

/// The scene's state as JSON: steps, simulated time, the step rate, shots, bonds broken,
/// splits, chunks crushed, pieces, whether the take is over.
#[wasm_bindgen]
pub fn broadside_stats() -> String {
    LOADED.with_borrow(|l| {
        let Some(l) = l.as_ref() else {
            return "{}".into();
        };
        let b = &l.b;
        let d = b.world.destructible_stats(b.ship).ok();
        let over = b.over();
        serde_json::json!({
            "frames": b.frame,
            "simSeconds": b.seconds(),
            "frameHz": 1.0 / b.dt,
            "shots": b.total.shots,
            "brokenBonds": b.total.broken,
            "splits": b.total.splits,
            "crushed": b.total.crushed,
            "chunks": d.map_or(0, |d| d.chunks),
            "bonds": d.map_or(0, |d| d.bonds),
            "pieces": d.map_or(0, |d| d.actors),
            "splinters": b.splinters.len(),
            "intact": b.intact_share,
            "scripted": b.scripted,
            "finale": b.finale.is_some(),
            "over": over,
        })
        .to_string()
    })
}

/// A simple figure standing on its origin's... the capsule centre: coat, head, hat.
fn captain_vertices(id: f32, out: &mut Vec<f32>) {
    use broadside::{CAPTAIN_HALF, CAPTAIN_RADIUS};
    let foot = -(CAPTAIN_HALF + CAPTAIN_RADIUS);
    // Coat (a tapered box would do; two boxes read as coat and legs).
    box_vertices_at(
        [0.0, foot + 0.45, 0.0],
        [0.13, 0.45, 0.17],
        [0.07, 0.07, 0.08],
        id,
        out,
    );
    box_vertices_at([0.0, foot + 1.2, 0.0], [0.15, 0.32, 0.24], COAT, id, out);
    let mut head = Vec::new();
    sphere_vertices(0.12, 8, SKIN, id, &mut head);
    offset(&mut head, [0.0, foot + 1.66, 0.0]);
    out.extend(head);
    // A tricorn's crown and brim.
    box_vertices_at([0.0, foot + 1.82, 0.0], [0.11, 0.06, 0.11], HAT, id, out);
    box_vertices_at([0.0, foot + 1.77, 0.0], [0.22, 0.015, 0.22], HAT, id, out);
}

fn offset(v: &mut [f32], by: [f32; 3]) {
    for p in v.chunks_exact_mut(VERTEX_FLOATS) {
        for k in 0..3 {
            p[k] += by[k];
        }
    }
}

fn box_vertices_at(at: [f32; 3], half: [f32; 3], color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    let mut v = Vec::new();
    box_vertices(half, color, id, &mut v);
    offset(&mut v, at);
    out.extend(v);
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
            let quad = [
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
            ];
            let tri = if sign > 0.0 {
                [0, 1, 2, 0, 2, 3]
            } else {
                [0, 2, 1, 0, 3, 2]
            };
            for k in tri {
                vertex(quad[k], n, color, id, out);
            }
        }
    }
}

/// A sphere of `radius`, `rings` rings of twice as many segments.
fn sphere_vertices(radius: f32, rings: usize, color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    let segs = 2 * rings;
    let n = |i: usize, j: usize| {
        let (th, ph) = (
            std::f32::consts::PI * i as f32 / rings as f32,
            std::f32::consts::TAU * j as f32 / segs as f32,
        );
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
}

/// The save area of the Asyncify build (`engine_world_wasm_bg.asyncify.wasm`, for browsers
/// without JSPI; `engine/scripts/build-web.sh`, `fracture/gpu-wait.mjs`): a header of the
/// next free address and the end address, then room for the stack a suspended export saves.
/// Returns the area's address; the page calls it once. The JSPI build never calls it.
#[no_mangle]
pub extern "C" fn broadside_asyncify_buffer() -> u32 {
    const BYTES: usize = 1 << 20;
    let area: &'static mut [u8] = Box::leak(vec![0u8; BYTES].into_boxed_slice());
    let base = area.as_mut_ptr() as usize as u32;
    area[0..4].copy_from_slice(&(base + 8).to_le_bytes());
    area[4..8].copy_from_slice(&(base + BYTES as u32).to_le_bytes());
    base
}
