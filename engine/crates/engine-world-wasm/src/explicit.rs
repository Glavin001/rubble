//! The explicit-beams solver in the browser (`ExplicitSim`): soft vehicles stepped by
//! `engine_webgpu::ExplicitBeamsGpu` (fixed 0.5 ms steps, the
//! `engine_softbody::explicit_beams` rules) on the World's device, so a page renders
//! them from the solver's own buffers in the same passes as the World's scene.
//!
//! The scenes are the explicit solver's test scenes: a vehicle at rest, coasting
//! with its tyres rolling, or coasting into a static wall. A spawn places the
//! vehicle with its forward axis along the yaw given (+Z at yaw 0), its node box
//! centred on the spawn point (or the middle of its front face there, `front`) and
//! its lowest node 2 cm above the flat ground.
//! Nothing here waits on the GPU: frames are submitted, and reads resolve later
//! (`mapAsync`).

use crate::{err, gpu_buffer, WebWorld};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, NODE_IMMOVABLE, NODE_TYRE,
};
use engine_softbody::SoftVehicleDesc;
use engine_webgpu::ExplicitBeamsGpu;
use glam::{Quat, Vec3};
use wasm_bindgen::prelude::*;

/// Steps per second.
const RATE: f64 = 2000.0;
/// The most steps one frame may take (the GPU host's parameter copies).
const MAX_FRAME_STEPS: u32 = 128;
/// Frames timed between reads of the GPU times.
const TIMED_FRAMES: u32 = 256;

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// One spawned vehicle: its CPU copy at spawn (flags, tyres, beam ends) and what
/// the page reads.
struct Spawned {
    name: String,
    sim: ExplicitBeams,
    /// The node the pose reads follow (structure near the node box's centre at spawn).
    reference: u32,
    forward: Vec3,
    warnings: Vec<String>,
}

/// Explicit-beams vehicles on the World's device (`WebWorld::explicit_sim`).
#[wasm_bindgen]
pub struct ExplicitSim {
    gpu: ExplicitBeamsGpu,
    device: wgpu::Device,
    queue: wgpu::Queue,
    vehicles: Vec<Spawned>,
    boxes: Vec<(Vec3, Vec3)>,
    /// `spec.shapes`: oriented boxes, cylinders, capsules and spheres every vehicle
    /// collides with (after its boxes).
    shapes: Vec<StaticShape>,
    /// Render records: per collision triangle (nodes, vehicle) and its rest edges;
    /// per beam (ends, vehicle | tyre << 16, spawn rest bits).
    tris: wgpu::Buffer,
    beam_recs: wgpu::Buffer,
    tri_count: u32,
    steps: u64,
    timed: bool,
}

/// The spawn's vehicle: `.truck` text, JBeam files (and `.pc`), or a desc.
fn build(
    v: &serde_json::Value,
    settings: ExplicitBeamsSettings,
) -> Result<(ExplicitBeams, SoftVehicleDesc, Vec<String>), String> {
    if let Some(text) = v.get("truck").and_then(|t| t.as_str()) {
        let sim = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings)?;
        return Ok((sim, engine_softbody::truck::load_truck(text), Vec::new()));
    }
    if let Some(files) = v.get("jbeam").and_then(|t| t.as_array()) {
        let files: Vec<(String, String)> = files
            .iter()
            .map(|f| {
                let name = f.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string();
                let text = f.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string();
                (name, text)
            })
            .collect();
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pc = v.get("pc").and_then(|t| t.as_str());
        let (sim, text, warnings) =
            ExplicitBeams::from_jbeam(&refs, pc, Vec3::ZERO, Vec3::ZERO, settings)?;
        return Ok((sim, engine_softbody::truck::load_truck(&text), warnings));
    }
    if let Some(d) = v.get("desc") {
        let desc: SoftVehicleDesc =
            serde_json::from_value(d.clone()).map_err(|e| format!("desc: {e}"))?;
        let sim = ExplicitBeams::new(&desc, Vec3::ZERO, Vec3::ZERO, settings);
        return Ok((sim, desc, Vec::new()));
    }
    Err("a vehicle needs `truck`, `jbeam` or `desc`".into())
}

/// Turn every node about the vertical axis through the origin by `angle`. A whole
/// number of quarter turns permutes and negates coordinates exactly.
fn turn(sim: &mut ExplicitBeams, angle: f32) {
    let quarter = angle / std::f32::consts::FRAC_PI_2;
    let k = quarter.round();
    let exact = (quarter - k).abs() < 1e-6;
    let q = Quat::from_rotation_y(angle);
    let rot = |p: Vec3| -> Vec3 {
        if !exact {
            return q * p;
        }
        // (x, z) → (z, −x) per quarter turn (+Y rotation).
        match (k as i32).rem_euclid(4) {
            0 => p,
            1 => Vec3::new(p.z, p.y, -p.x),
            2 => Vec3::new(-p.x, p.y, -p.z),
            _ => Vec3::new(-p.z, p.y, p.x),
        }
    };
    if exact && (k as i32).rem_euclid(4) == 0 {
        return;
    }
    for n in &mut sim.nodes {
        n.pos = rot(n.pos);
        n.vel = rot(n.vel);
    }
}

/// Box (min, max) of the nodes on collision triangles; every node's when none.
fn skin_bounds(sim: &ExplicitBeams) -> (Vec3, Vec3) {
    let mut b = (Vec3::MAX, Vec3::MIN);
    for t in &sim.triangles {
        for &i in &t.nodes {
            let p = sim.nodes[i as usize].pos;
            b = (b.0.min(p), b.1.max(p));
        }
    }
    if b.0.x > b.1.x { bounds(sim) } else { b }
}

/// Node box (min, max).
fn bounds(sim: &ExplicitBeams) -> (Vec3, Vec3) {
    sim.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    })
}

/// Every node at `velocity` (immovable ones still), tyres rolling along it.
fn set_moving(sim: &mut ExplicitBeams, velocity: Vec3) {
    for n in &mut sim.nodes {
        n.vel = if n.flags & NODE_IMMOVABLE != 0 {
            Vec3::ZERO
        } else {
            velocity
        };
    }
    if velocity != Vec3::ZERO {
        sim.roll_tyres(velocity);
    }
}

/// Three numbers at `key` (`default` when absent).
fn vec3(v: &serde_json::Value, key: &str, default: Vec3) -> Result<Vec3, String> {
    let Some(a) = v.get(key) else {
        return Ok(default);
    };
    let f: Vec<f32> =
        serde_json::from_value(a.clone()).map_err(|e| format!("shapes.{key}: {e}"))?;
    match f[..] {
        [x, y, z] => Ok(Vec3::new(x, y, z)),
        _ => Err(format!("shapes.{key}: expected [x, y, z]")),
    }
}

/// `spec.shapes`: `[{ kind: 'box', centre, half, rotation: [x, y, z, w] } | { kind:
/// 'cylinder', centre, radius, halfHeight } (vertical) | { kind: 'capsule', a, b,
/// radius } | { kind: 'sphere', centre, radius }]` (world metres).
fn shapes(spec: &serde_json::Value) -> Result<Vec<StaticShape>, String> {
    let Some(list) = spec.get("shapes") else {
        return Ok(Vec::new());
    };
    let list = list.as_array().ok_or("spec.shapes: an array")?;
    list.iter()
        .map(|s| {
            let centre = vec3(s, "centre", Vec3::ZERO)?;
            let radius = num(s, "radius", 0.5) as f32;
            match s.get("kind").and_then(|k| k.as_str()).unwrap_or("box") {
                "box" => {
                    let q = match s.get("rotation") {
                        Some(r) => {
                            let q: Vec<f32> = serde_json::from_value(r.clone())
                                .map_err(|e| format!("shapes.rotation: {e}"))?;
                            match q[..] {
                                [x, y, z, w] => Quat::from_xyzw(x, y, z, w),
                                _ => return Err("shapes.rotation: expected [x, y, z, w]".into()),
                            }
                        }
                        None => Quat::IDENTITY,
                    };
                    Ok(StaticShape::oriented_box(
                        centre,
                        vec3(s, "half", Vec3::splat(0.5))?,
                        q,
                    ))
                }
                "cylinder" => Ok(StaticShape::cylinder(
                    centre,
                    radius,
                    num(s, "halfHeight", 1.0) as f32,
                )),
                "capsule" => Ok(StaticShape::capsule(
                    vec3(s, "a", Vec3::ZERO)?,
                    vec3(s, "b", Vec3::Y)?,
                    radius,
                )),
                "sphere" => Ok(StaticShape::sphere(centre, radius)),
                k => Err(format!(
                    "shapes: unknown kind {k}: box, cylinder, capsule or sphere"
                )),
            }
        })
        .collect()
}

fn num(v: &serde_json::Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

/// The `arena` scene's `ring` layout: `count` vehicles on concentric rings around the
/// origin, every one facing the centre at its own speed in [speedMin, speedMax] m/s
/// (a fixed low-discrepancy sequence in its index). The innermost ring has radius
/// `ringRadius` (default 30 m) to the front faces; rings are a vehicle length plus
/// 3 m apart, each holding as many vehicles as fit side by side with 1.5 m between
/// them, so the rings arrive at the centre in waves.
fn ring(spec: &serde_json::Value, n: usize, width: f32, length: f32) -> Vec<ArenaSpot> {
    let (vmin, vmax) = (num(spec, "speedMin", 25.0) as f32, num(spec, "speedMax", 42.0) as f32);
    let r0 = num(spec, "ringRadius", 30.0) as f32;
    let golden = |i: usize, a: f64| ((i as f64 + 1.0) * a).fract() as f32;
    let mut out = Vec::with_capacity(n);
    let mut r = r0;
    while out.len() < n {
        let m = ((std::f32::consts::TAU * r / (width + 1.5)).floor() as usize).max(1).min(n - out.len());
        // Each ring turned half a slot from the last, so waves interleave.
        let phase = 0.5 * (out.len() % 2) as f32 + golden(out.len(), 0.618_033_988_75);
        for j in 0..m {
            let k = out.len();
            let a = std::f32::consts::TAU * (j as f32 + phase) / m as f32;
            let (sa, ca) = a.sin_cos();
            // Facing the centre: forward = −(sin a, cos a); yaw 0 faces +Z.
            out.push(ArenaSpot {
                at: Vec3::new(r * sa, 0.0, r * ca),
                yaw: a + std::f32::consts::PI,
                speed: vmin + (vmax - vmin) * golden(k, 0.754_877_666_25),
                boxes: Vec::new(),
                vel: None,
                lift: 0.0,
            });
        }
        r += length + 3.0;
    }
    out
}

/// One `arena` spawn: the middle of its front face, yaw, speed, and the static
/// boxes it can hit.
struct ArenaSpot {
    at: Vec3,
    yaw: f32,
    speed: f32,
    boxes: Vec<(Vec3, Vec3)>,
    /// Spawn velocity when not along the vehicle's forward axis (a side impact).
    vel: Option<Vec3>,
    /// Height of the lowest node above 2 cm (a drop test).
    lift: f32,
}

/// The crash-test scenes, built from one vehicle (`width` across, `length` along at
/// yaw 0, facing +Z). Speeds in m/s, angles in degrees; each moving vehicle starts
/// 1 + 0.5 × its speed metres from what it hits, so the hit lands after about half a
/// second:
/// * `wall`: into a rigid wall at `speed`; `overlap` (0 to 1, default 1) of the
///   width meets it (the rest passes beside it), the vehicle turned by `angle`.
/// * `headon`: two copies nose to nose at `speed` and `speed2` (default `speed`);
///   `overlap` of the width overlaps, the second car turned by `angle` from straight
///   on.
/// * `tbone`: into the side of a second copy standing (or moving forward at
///   `carSpeed`) across its lane, turned `angle` (default 90) from its heading, hit
///   `at` metres along it from its centre.
/// * `pole`: into a 20 cm square pole `offset` metres across, turned by `angle`.
/// * `sidepole`: sliding sideways at `speed` into a 20 cm pole `at` metres along its
///   side from its centre.
/// * `limbo`: under a bar 30 cm deep whose underside is `bar` metres up (default 1).
/// * `drop`: dropped flat from `height` metres (default 8), moving forward at `speed`
///   (default 0).
/// Returns the spawns and every box once (for rendering); None for another scene.
fn crash_scene(scene: &str, spec: &serde_json::Value, width: f32, length: f32) -> Option<(Vec<ArenaSpot>, Vec<(Vec3, Vec3)>)> {
    use std::f32::consts::PI;
    let deg = |k: &str, d: f64| (num(spec, k, d) as f32).to_radians();
    let v = num(spec, "speed", 27.78) as f32;
    let run_up = |v: f32| 1.0 + 0.5 * v;
    let gap = run_up(v);
    let spot = |at: Vec3, yaw: f32, speed: f32, boxes: Vec<(Vec3, Vec3)>, vel: Option<Vec3>| ArenaSpot { at, yaw, speed, boxes, vel, lift: 0.0 };
    let pole = |x: f32, z: f32| (Vec3::new(x, 2.0, z), Vec3::new(0.1, 2.0, 0.1));
    let overlap = (num(spec, "overlap", 1.0) as f32).clamp(0.05, 1.0);
    Some(match scene {
        "wall" => {
            // Full overlap: a 4 m high wall a metre wider than the car each side; less:
            // the wall covers only that share of the car's width, on its +X side.
            // Turned by `angle`, the car arrives `gap` along its heading: the wall's face
            // and its overlap are placed there.
            let a = deg("angle", 0.0);
            let (hit_x, hit_z) = (gap * a.sin(), gap * a.cos());
            let (half_x, cx) = if overlap >= 1.0 { (0.5 * width + 1.0, 0.0) } else { (0.5 * overlap * width, 0.5 * width * (1.0 - overlap)) };
            let b = (Vec3::new(hit_x + cx, 2.0, hit_z + 0.5), Vec3::new(half_x, 2.0, 0.5));
            (vec![spot(Vec3::ZERO, a, v, vec![b], None)], vec![b])
        }
        "headon" => {
            let v2 = num(spec, "speed2", f64::from(v)) as f32;
            let dx = (1.0 - overlap) * width;
            (
                vec![
                    spot(Vec3::new(0.0, 0.0, -run_up(v) * 0.5), 0.0, v, Vec::new(), None),
                    spot(Vec3::new(dx, 0.0, run_up(v2) * 0.5), PI + deg("angle", 0.0), v2, Vec::new(), None),
                ],
                Vec::new(),
            )
        }
        "tbone" => {
            let at = num(spec, "at", 0.0) as f32;
            let struck = num(spec, "carSpeed", 0.0) as f32;
            let yaw = deg("angle", 90.0);
            let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
            // A moving struck car starts back along its path so it is at the origin when
            // the striking car arrives (gap / speed later).
            let back = fwd * (struck * gap / v.max(0.1));
            (
                vec![
                    spot(Vec3::new(at, 0.0, -(0.5 * width + gap)), 0.0, v, Vec::new(), None),
                    // Centred at the origin then: its front face's middle half a length ahead.
                    spot(fwd * (0.5 * length) - back, yaw, struck, Vec::new(), None),
                ],
                Vec::new(),
            )
        }
        "pole" => {
            // On the car's path at `gap` along its heading (turned by `angle`), `offset` across.
            let a = deg("angle", 0.0);
            let off = num(spec, "offset", 0.0) as f32;
            let b = pole((gap + 0.1) * a.sin() + off * a.cos(), (gap + 0.1) * a.cos() - off * a.sin());
            (vec![spot(Vec3::ZERO, a, v, vec![b], None)], vec![b])
        }
        "sidepole" => {
            let b = pole(num(spec, "at", 0.0) as f32, 0.5 * width + gap + 0.1);
            // Forward +X, its centre at the origin, moving along +Z (sideways).
            (vec![spot(Vec3::new(0.5 * length, 0.0, 0.0), 0.5 * PI, v, vec![b], Some(Vec3::new(0.0, 0.0, v)))], vec![b])
        }
        "limbo" => {
            let bar = num(spec, "bar", 1.0) as f32;
            let b = (Vec3::new(0.0, bar + 0.15, gap + 0.15), Vec3::new(0.5 * width + 1.0, 0.15, 0.15));
            (vec![spot(Vec3::ZERO, 0.0, v, vec![b], None)], vec![b])
        }
        "drop" => {
            let mut s = spot(Vec3::ZERO, 0.0, num(spec, "speed", 0.0) as f32, Vec::new(), None);
            s.lift = num(spec, "height", 8.0) as f32;
            (vec![s], Vec::new())
        }
        _ => return None,
    })
}

/// The `arena` scene: `count` copies of one vehicle (`width` across, `length` along)
/// in rows facing each other in pairs across a long shared wall, `cols` lanes a row.
/// Each vehicle has its own speed in [speedMin, speedMax] m/s and its own distance to
/// the wall, so its hit lands at a time spread over [hitMin, hitMax] s (a fixed
/// low-discrepancy sequence in the vehicle's index); every fourth lane has a pole a
/// quarter of the vehicle's width off centre before the wall. Lanes are `width` + 3 m
/// apart. `headOn` (0 to 1, default 0): that share of lanes has no wall, so the two
/// vehicles facing each other there collide head-on (needs vehicle–vehicle contact).
/// Returns the spawns and every box once (for rendering).
fn arena(spec: &serde_json::Value, width: f32, length: f32) -> (Vec<ArenaSpot>, Vec<(Vec3, Vec3)>) {
    let n = num(spec, "count", 256.0).max(1.0) as usize;
    if spec.get("layout").and_then(|l| l.as_str()) == Some("ring") {
        return (ring(spec, n, width, length), Vec::new());
    }
    let cols = num(spec, "cols", ((8 * n) as f64).sqrt().ceil()).max(1.0) as usize;
    let (vmin, vmax) = (num(spec, "speedMin", 25.0) as f32, num(spec, "speedMax", 42.0) as f32);
    let (tmin, tmax) = (num(spec, "hitMin", 0.3) as f32, num(spec, "hitMax", 1.2) as f32);
    let golden = |i: usize, a: f64| ((i as f64 + 1.0) * a).fract() as f32;
    let lane = width + 3.0;
    let depth = vmax * tmax + length + 6.0;
    let pair_pitch = 2.0 * depth + 1.0;
    let rows = n.div_ceil(cols);
    let head_on = num(spec, "headOn", 0.0) as f32;
    // Per lane: a wall segment, unless the lane is head-on.
    let open = |col: usize| golden(col, 0.569_840_290_998) < head_on;
    let mut all = Vec::new();
    for p in 0..rows.div_ceil(2) {
        for col in (0..cols.min(n)).filter(|&c| !open(c)) {
            let x = (col as f32 - 0.5 * (cols as f32 - 1.0)) * lane;
            all.push((Vec3::new(x, 2.0, p as f32 * pair_pitch), Vec3::new(0.5 * lane, 2.0, 0.5)));
        }
    }
    let spots = (0..n)
        .map(|k| {
            let (row, col) = (k / cols, k % cols);
            let x = (col as f32 - 0.5 * (cols as f32 - 1.0)) * lane;
            let wall = (Vec3::new(x, 2.0, (row / 2) as f32 * pair_pitch), Vec3::new(0.5 * lane, 2.0, 0.5));
            let dir = if row % 2 == 0 { 1.0 } else { -1.0 };
            let speed = vmin + (vmax - vmin) * golden(k, 0.618_033_988_75);
            let gap = speed * (tmin + (tmax - tmin) * golden(k, 0.754_877_666_25));
            let face = wall.0.z - dir * 0.5;
            let mut boxes = if open(col) { Vec::new() } else { vec![wall] };
            if col % 4 == 1 && !open(col) {
                let side = if (k / 4) % 2 == 0 { 1.0 } else { -1.0 };
                let pole = (
                    Vec3::new(x + side * 0.25 * width, 2.0, face - dir * (0.5 * gap).min(8.0)),
                    Vec3::new(0.25, 2.0, 0.25),
                );
                boxes.push(pole);
                all.push(pole);
            }
            ArenaSpot {
                at: Vec3::new(x, 0.0, face - dir * gap),
                yaw: if dir > 0.0 { 0.0 } else { std::f32::consts::PI },
                speed,
                boxes,
                vel: None,
                lift: 0.0,
            }
        })
        .collect();
    (spots, all)
}

#[wasm_bindgen]
impl WebWorld {
    /// Explicit-beams vehicles on this world's device. `spec`: `{ vehicles: [{
    /// truck | jbeam: [[name, text], …] (+ pc) | desc, at: [x, z], yaw, front }], scene:
    /// 'idle' | 'coast' | 'wall' | 'arena', speed (m/s, coast and wall), wallGap (m,
    /// default 1 + 0.5·speed); arena (see `arena`): count, cols, speedMin, speedMax,
    /// hitMin, hitMax, headOn, layout ('rows' | 'ring', see `ring`), ringRadius;
    /// shapes (static colliders for every vehicle, see `shapes`); the
    /// first vehicle copied to every spawn, drag, turbulence, fastInvSqrt (default true), timing
    /// (default true: GPU timestamps when the device has them) }`.
    #[wasm_bindgen(js_name = explicitSim)]
    pub fn explicit_sim(&self, spec: JsValue) -> Result<ExplicitSim, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        ExplicitSim::new(self.world.device(), self.world.queue(), &spec).map_err(err)
    }
}

impl ExplicitSim {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spec: &serde_json::Value,
    ) -> Result<Self, String> {
        let flag = |k: &str| spec.get(k).and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: flag("fastInvSqrt"),
            drag: flag("drag"),
            turbulence: flag("turbulence") && flag("drag"),
        };
        let scene = spec.get("scene").and_then(|s| s.as_str()).unwrap_or("idle");
        const SCENES: [&str; 10] = ["idle", "coast", "wall", "arena", "headon", "tbone", "pole", "sidepole", "limbo", "drop"];
        if !SCENES.contains(&scene) {
            return Err(format!("unknown explicit scene {scene}: {}", SCENES.join(", ")));
        }
        let speed = if scene == "idle" {
            0.0
        } else {
            num(spec, "speed", 27.78) as f32
        };
        let list = spec
            .get("vehicles")
            .and_then(|v| v.as_array())
            .ok_or("spec.vehicles: an array of vehicles")?;
        if list.is_empty() {
            return Err("spec.vehicles is empty".into());
        }
        let mut vehicles = Vec::new();
        let mut boxes = Vec::new();
        let shapes = shapes(spec)?;
        // `arena`: the first vehicle, built once, copied to every spawn.
        let mut spots = Vec::new();
        let template = if scene == "arena" || crash_scene(scene, spec, 1.0, 1.0).is_some() {
            let (mut sim, desc, warnings) = build(&list[0], settings)?;
            let f = engine_softbody::pack::forward_axis(&desc);
            turn(&mut sim, -f.x.atan2(f.z));
            // The body's extent: its collision-triangle nodes (a file's helper nodes,
            // such as camera nodes, can sit outside it), else every node.
            let (lo, hi) = skin_bounds(&sim);
            let (s, all) = crash_scene(scene, spec, hi.x - lo.x, hi.z - lo.z)
                .unwrap_or_else(|| arena(spec, hi.x - lo.x, hi.z - lo.z));
            turn(&mut sim, f.x.atan2(f.z));
            spots = s;
            boxes = all;
            Some((sim, desc, warnings))
        } else {
            None
        };
        let arena_list: Vec<serde_json::Value> = spots
            .iter()
            .map(|s| serde_json::json!({ "at": [s.at.x, s.at.z], "yaw": s.yaw, "front": true }))
            .collect();
        let list = if template.is_some() { &arena_list } else { list };
        for (vi, v) in list.iter().enumerate() {
            let (mut sim, desc, warnings) = match &template {
                Some(t) => t.clone(),
                None => build(v, settings)?,
            };
            let speed = spots.get(vi).map_or(speed, |s| s.speed);
            if sim.nodes.is_empty() {
                return Err("the vehicle has no nodes".into());
            }
            // Its forward along the spawn yaw (+Z at 0).
            let f = engine_softbody::pack::forward_axis(&desc);
            let yaw = num(v, "yaw", 0.0) as f32;
            turn(&mut sim, yaw - f.x.atan2(f.z));
            let forward = Quat::from_rotation_y(yaw) * Vec3::Z;
            let at = v.get("at").and_then(|a| a.as_array());
            let ax = at
                .and_then(|a| a.first())
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0) as f32;
            let az = at
                .and_then(|a| a.get(1))
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0) as f32;
            // `front`: the spawn point is the middle of the front face (along the
            // forward axis), else the body's centre. The body is its collision-triangle
            // nodes when it has any: a file's helper nodes (cameras and the like) can
            // sit off to one side and would move the centre off the car's centreline.
            let at3 = Vec3::new(ax, 0.0, az);
            let lateral = Vec3::Y.cross(forward);
            let skin: Vec<u32> = sim.triangles.iter().flat_map(|t| t.nodes).collect();
            let span = |axis: Vec3| {
                let d = |i: u32| sim.nodes[i as usize].pos.dot(axis);
                let all = || (0..sim.nodes.len() as u32).map(d);
                let fold = |it: &mut dyn Iterator<Item = f32>| it.fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
                if skin.is_empty() { fold(&mut all()) } else { fold(&mut skin.iter().map(|&i| d(i))) }
            };
            let (f0, f1) = span(forward);
            let (l0, l1) = span(lateral);
            let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            let front = v.get("front").and_then(|x| x.as_bool()).unwrap_or(false);
            let along = at3.dot(forward) - if front { f1 } else { 0.5 * (f0 + f1) };
            let across = at3.dot(lateral) - 0.5 * (l0 + l1);
            let lift = spots.get(vi).map_or(0.0, |s| s.lift);
            let shift = forward * along + lateral * across + Vec3::Y * (0.02 - lowest + lift);
            for n in &mut sim.nodes {
                n.pos += shift;
            }
            let (lo, hi) = bounds(&sim);
            let centre = 0.5 * (lo + hi);
            // The camera's reference: of the non-tyre nodes nearest the centre (a
            // tenth, at least 8), the one with the most beams (body structure,
            // not a panel that may come off).
            let mut near: Vec<u32> = (0..sim.nodes.len() as u32)
                .filter(|&i| sim.nodes[i as usize].flags & NODE_TYRE == 0)
                .collect();
            near.sort_by(|&a, &b| {
                let da = sim.nodes[a as usize].pos.distance_squared(centre);
                let db = sim.nodes[b as usize].pos.distance_squared(centre);
                da.total_cmp(&db)
            });
            near.truncate((near.len() / 10).max(8));
            let reference = near
                .iter()
                .copied()
                .max_by_key(|&i| (sim.incidence[i as usize].len(), std::cmp::Reverse(i)))
                .unwrap_or(0);
            set_moving(&mut sim, spots.get(vi).and_then(|s| s.vel).unwrap_or(forward * speed));
            if let Some(s) = spots.get(vi) {
                for &(c, h) in &s.boxes {
                    sim.boxes.push(StaticBox::centered(c, h));
                }
            }
            if scene == "wall" && spots.is_empty() {
                // A 4 m wall across the forward axis (snapped to x or z), `wallGap` ahead
                // of the front, a metre wider than the vehicle each side.
                let gap = num(spec, "wallGap", 1.0 + 0.5 * f64::from(speed)) as f32;
                let along_z = forward.z.abs() >= forward.x.abs();
                let dir = if along_z {
                    Vec3::Z * forward.z.signum()
                } else {
                    Vec3::X * forward.x.signum()
                };
                let front = sim
                    .nodes
                    .iter()
                    .map(|n| n.pos.dot(dir))
                    .fold(f32::MIN, f32::max);
                let half_t = 0.5;
                let side = 0.5 * if along_z { hi.x - lo.x } else { hi.z - lo.z } + 1.0;
                let c = dir * (front + gap + half_t)
                    + Vec3::Y * 2.0
                    + if along_z {
                        Vec3::X * centre.x
                    } else {
                        Vec3::Z * centre.z
                    };
                let half = if along_z {
                    Vec3::new(side, 2.0, half_t)
                } else {
                    Vec3::new(half_t, 2.0, side)
                };
                sim.boxes.push(StaticBox::centered(c, half));
                boxes.push((c, half));
            }
            sim.shapes.extend_from_slice(&shapes);
            vehicles.push(Spawned {
                name: desc.name.clone(),
                sim,
                reference,
                forward,
                warnings,
            });
        }
        let sims: Vec<ExplicitBeams> = vehicles.iter().map(|v| v.sim.clone()).collect();
        let mut gpu = ExplicitBeamsGpu::new(device, queue, &sims);
        let timed = spec.get("timing").and_then(|x| x.as_bool()).unwrap_or(true)
            && gpu.enable_frame_timing(TIMED_FRAMES);

        // Render records.
        let mut tris: Vec<[u32; 4]> = Vec::new();
        let mut recs: Vec<[u32; 4]> = Vec::new();
        for (vi, v) in vehicles.iter().enumerate() {
            let off = gpu.node_offset(vi);
            let p = |i: u32| v.sim.nodes[i as usize].pos;
            for t in &v.sim.triangles {
                let [a, b, c] = t.nodes;
                tris.push([off + a, off + b, off + c, vi as u32]);
                tris.push([
                    p(a).distance(p(b)).to_bits(),
                    p(b).distance(p(c)).to_bits(),
                    p(c).distance(p(a)).to_bits(),
                    0,
                ]);
            }
            let tyre = |i: u32| v.sim.nodes[i as usize].flags & NODE_TYRE != 0;
            for b in &v.sim.beams {
                let t = (tyre(b.a) || tyre(b.b)) as u32;
                recs.push([off + b.a, off + b.b, vi as u32 | t << 16, b.rest0.to_bits()]);
            }
        }
        use wgpu::util::DeviceExt;
        let init = |label: &str, words: &[[u32; 4]]| {
            let bytes: &[u8] = if words.is_empty() {
                &[0u8; 16]
            } else {
                bytemuck::cast_slice(words)
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let tris_buf = init("explicit-render-tris", &tris);
        let recs_buf = init("explicit-render-beams", &recs);
        Ok(Self {
            gpu,
            device: device.clone(),
            queue: queue.clone(),
            tri_count: (tris.len() / 2) as u32,
            vehicles,
            boxes,
            shapes,
            tris: tris_buf,
            beam_recs: recs_buf,
            steps: 0,
            timed,
        })
    }
}

#[wasm_bindgen]
impl ExplicitSim {
    /// Steps per second (2,000: 0.5 ms steps).
    #[wasm_bindgen(js_name = stepRate)]
    pub fn step_rate(&self) -> f64 {
        RATE
    }

    /// The most steps one [`Self::frame`] takes.
    #[wasm_bindgen(js_name = maxFrameSteps)]
    pub fn max_frame_steps(&self) -> u32 {
        MAX_FRAME_STEPS
    }

    /// Submit one frame of `steps` fixed steps for every vehicle (never waits).
    pub fn frame(&mut self, steps: u32) -> Result<(), JsValue> {
        if steps > MAX_FRAME_STEPS {
            return Err(err(format!(
                "at most {MAX_FRAME_STEPS} steps per frame, got {steps}"
            )));
        }
        if steps > 0 {
            self.gpu.frame(steps);
            self.steps += u64::from(steps);
        }
        Ok(())
    }

    /// Steps run since spawn.
    pub fn steps(&self) -> f64 {
        self.steps as f64
    }

    /// Whether GPU frame times are recorded (the device has timestamp queries).
    pub fn timed(&self) -> bool {
        self.timed
    }

    /// The node state `GPUBuffer`: 3 `vec4<f32>` per node (position relative to its
    /// vehicle's origin and mass, velocity, force), every vehicle's nodes in order.
    #[wasm_bindgen(js_name = nodesBuffer)]
    pub fn nodes_buffer(&self) -> JsValue {
        gpu_buffer(self.gpu.nodes_buffer())
    }

    /// The beam state `GPUBuffer`: 2 `vec4<f32>` per beam, (rest length, yield
    /// thresholds, gate) and (strength, force, broken 0/1, 0).
    #[wasm_bindgen(js_name = beamsBuffer)]
    pub fn beams_buffer(&self) -> JsValue {
        gpu_buffer(self.gpu.beams_buffer())
    }

    /// The per-vehicle `GPUBuffer`: `vec4<u32>` per vehicle, its origin's bits (xyz).
    #[wasm_bindgen(js_name = miscBuffer)]
    pub fn misc_buffer(&self) -> JsValue {
        gpu_buffer(self.gpu.misc_buffer())
    }

    /// Collision triangles for rendering: 2 `vec4` per triangle, (global node
    /// indices a, b, c, vehicle) as `u32` then its spawn edge lengths ab, bc, ca as
    /// `f32` bits.
    #[wasm_bindgen(js_name = trisBuffer)]
    pub fn tris_buffer(&self) -> JsValue {
        gpu_buffer(&self.tris)
    }

    /// Beams for rendering: `vec4<u32>` per beam in beam-buffer order, (global node
    /// a, b, vehicle | 1 << 16 for a tyre beam, spawn rest length bits).
    #[wasm_bindgen(js_name = beamRecordsBuffer)]
    pub fn beam_records_buffer(&self) -> JsValue {
        gpu_buffer(&self.beam_recs)
    }

    /// `{ nodes, beams, triangles, timed, vehicles: [{ name, nodes, beams,
    /// triangles, tyres, mass, nodeOffset, beamOffset, warnings }], boxes: [[cx,
    /// cy, cz, hx, hy, hz, qx, qy, qz, qw, kind]] }` (boxes: the static colliders,
    /// for rendering: the axis-aligned boxes (identity rotation, kind 0) then
    /// `spec.shapes` (`StaticShape::render_row`: kind 1 a box, 2 a vertical
    /// cylinder with (hx, hy) its radius and half height, 3 a capsule along its
    /// rotated y with radius hx and half segment hy, 4 a sphere of radius hx)).
    pub fn layout(&self) -> JsValue {
        let vehicles: Vec<serde_json::Value> = self
            .vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| {
                serde_json::json!({
                    "name": v.name, "nodes": v.sim.nodes.len(), "beams": v.sim.beams.len(),
                    "triangles": v.sim.triangles.len(), "tyres": v.sim.tyres.len(),
                    "mass": v.sim.nodes.iter().map(|n| f64::from(n.mass)).sum::<f64>(),
                    "nodeOffset": self.gpu.node_offset(i), "beamOffset": self.gpu.beam_offset(i),
                    "forward": v.forward.to_array(), "warnings": v.warnings,
                })
            })
            .collect();
        let boxes: Vec<[f32; 11]> = self
            .boxes
            .iter()
            .map(|(c, h)| [c.x, c.y, c.z, h.x, h.y, h.z, 0.0, 0.0, 0.0, 1.0, 0.0])
            .chain(self.shapes.iter().map(StaticShape::render_row))
            .collect();
        js(serde_json::json!({
            "nodes": self.vehicles.iter().map(|v| v.sim.nodes.len()).sum::<usize>(),
            "beams": self.vehicles.iter().map(|v| v.sim.beams.len()).sum::<usize>(),
            "triangles": self.tri_count,
            "timed": self.timed,
            "wide": self.gpu.is_wide(),
            "vehicles": vehicles,
            "boxes": boxes,
        }))
    }

    /// Damage of the first `limit` vehicles (all when absent) from one copy of their
    /// beams, resolved once mapped: `[{ broken, bent, beams }]` (bent: alive with its
    /// rest length off its spawn length by over 0.1 %, as `GpuBeam::bent`). A large
    /// fleet's display reads a few: every beam of thousands is tens of megabytes.
    #[wasm_bindgen(js_name = readDamage)]
    pub fn read_damage(&self, limit: Option<u32>) -> js_sys::Promise {
        let buf = self.gpu.beams_buffer().clone();
        let mut ranges = Vec::new();
        let mut rest0 = Vec::new();
        let count = limit.map_or(self.vehicles.len(), |l| (l as usize).min(self.vehicles.len()));
        for (i, v) in self.vehicles.iter().enumerate().take(count) {
            let off = self.gpu.beam_offset(i) as usize;
            let n = v.sim.beams.len();
            ranges.push((buf.clone(), off as u64 * 32, n as u64 * 32));
            rest0.push(self.gpu.rest0()[off..off + n].to_vec());
        }
        crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let mut out = Vec::new();
            let mut at = 0;
            for r0 in rest0 {
                let (mut broken, mut bent) = (0u32, 0u32);
                for (k, &rest0) in r0.iter().enumerate() {
                    let rec = &w[at + k * 8..at + k * 8 + 8];
                    let rest = f32::from_bits(rec[0]);
                    let is_broken = f32::from_bits(rec[6]) != 0.0;
                    let beam = engine_webgpu::GpuBeam {
                        broken: is_broken,
                        rest,
                        rest0,
                    };
                    broken += is_broken as u32;
                    bent += beam.bent() as u32;
                }
                at += r0.len() * 8;
                out.push(serde_json::json!({ "broken": broken, "bent": bent, "beams": r0.len() }));
            }
            Ok(js(serde_json::Value::Array(out)))
        })
    }

    /// Per vehicle its reference node's world position and velocity, resolved once
    /// mapped: 6 floats per vehicle (x, y, z, vx, vy, vz).
    #[wasm_bindgen(js_name = readPoses)]
    pub fn read_poses(&self) -> js_sys::Promise {
        let nodes = self.gpu.nodes_buffer().clone();
        let misc = self.gpu.misc_buffer().clone();
        let mut ranges = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            ranges.push((misc.clone(), i as u64 * 16, 16));
            let node = u64::from(self.gpu.node_offset(i) + v.reference);
            ranges.push((nodes.clone(), node * 48, 32));
        }
        let n = self.vehicles.len();
        crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let f = |x: u32| f32::from_bits(x);
            let mut out = Vec::with_capacity(n * 6);
            for k in 0..n {
                let r = &w[k * 12..k * 12 + 12];
                out.extend_from_slice(&[
                    f(r[0]) + f(r[4]),
                    f(r[1]) + f(r[5]),
                    f(r[2]) + f(r[6]),
                    f(r[8]),
                    f(r[9]),
                    f(r[10]),
                ]);
            }
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }

    /// GPU milliseconds of each frame since the last read (timestamps around the
    /// frame's passes), resolved once mapped; empty without timestamps or frames.
    #[wasm_bindgen(js_name = readFrameTimes)]
    pub fn read_frame_times(&mut self) -> js_sys::Promise {
        let period = self.gpu.timestamp_period_ms();
        let Some((staging, used)) = self.gpu.take_frame_times() else {
            return js_sys::Promise::resolve(&JsValue::from(
                js_sys::Float64Array::new_with_length(0),
            ));
        };
        wasm_bindgen_futures::future_to_promise(async move {
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx.await
                .map_err(|_| err("readFrameTimes: map dropped"))?
                .map_err(|e| err(format!("readFrameTimes: {e}")))?;
            let ms: Vec<f64> = {
                let data = staging.slice(..).get_mapped_range().expect("mapped");
                let ticks: &[u64] = bytemuck::cast_slice(&data[..used as usize * 16]);
                ticks
                    .chunks(2)
                    .map(|p| p[1].saturating_sub(p[0]) as f64 * period)
                    .collect()
            };
            staging.unmap();
            Ok(js_sys::Float64Array::from(ms.as_slice()).into())
        })
    }

    /// Launch every vehicle at `speed` m/s along its spawn forward axis, tyres
    /// rolling (`ExplicitBeams::roll_tyres`) about their axles where they are now:
    /// reads the node positions, then writes every node's velocity. Resolves once
    /// written (steps taken meanwhile keep their own velocities until then).
    pub fn launch(&self, speed: f32) -> js_sys::Promise {
        let nodes = self.gpu.nodes_buffer().clone();
        let mut ranges = Vec::new();
        let mut sims = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            let off = u64::from(self.gpu.node_offset(i));
            ranges.push((nodes.clone(), off * 48, v.sim.nodes.len() as u64 * 48));
            sims.push((off, v.sim.clone(), v.forward));
        }
        let queue = self.queue.clone();
        crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let mut at = 0;
            for (off, mut sim, forward) in sims {
                for n in &mut sim.nodes {
                    n.pos = Vec3::new(
                        f32::from_bits(w[at]),
                        f32::from_bits(w[at + 1]),
                        f32::from_bits(w[at + 2]),
                    );
                    at += 12;
                }
                set_moving(&mut sim, forward * speed);
                for (k, n) in sim.nodes.iter().enumerate() {
                    let offset = ((off + k as u64) * 3 + 1) * 16;
                    queue.write_buffer(&nodes, offset, bytemuck::cast_slice(&n.vel.to_array()));
                }
            }
            Ok(JsValue::UNDEFINED)
        })
    }
}

/// The high-resolution test car (`engine_softbody::hires`) as JBeam text, tuned for
/// the explicit solver's 0.5 ms step (2 kg nodes, 1.5 MN/m over 0.15 m beams, the
/// stiffness of beams shorter than 0.15 m capped), as its tests run it.
#[wasm_bindgen(js_name = hiresCarJbeam)]
pub fn hires_car_jbeam() -> String {
    use engine_softbody::hires::{hires_car_jbeam, HiresParams};
    hires_car_jbeam(&HiresParams {
        node_weight: 2.0,
        spring: 1.5e6,
        damp: 150.0,
        stiffness_length: 0.15,
        ..Default::default()
    })
    .0
}

/// The procedural sedan for the explicit solver, as `.truck` text: the acceptance
/// runs' car (`CarParams::sedan()`, 10 kg minimum node mass;
/// `docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck` is its bare
/// body), with its soft tyres when `tyres`.
#[wasm_bindgen(js_name = explicitSedanTruck)]
pub fn explicit_sedan_truck(tyres: bool) -> String {
    use engine_softbody::procedural::{car_truck, car_truck_bare, CarParams};
    let p = CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    };
    if tyres {
        car_truck(&p)
    } else {
        car_truck_bare(&p)
    }
}
