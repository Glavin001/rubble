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
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, NODE_IMMOVABLE, NODE_TYRE, STEP,
};
use engine_softbody::SoftVehicleDesc;
use engine_webgpu::ExplicitBeamsGpu;
use glam::{Quat, Vec3};
use wasm_bindgen::prelude::*;

// The crash-test lab: the node grab, per-beam strain constants, frame-state reads.
#[path = "explicit_lab.rs"]
mod crash_lab;
// Driving: raycast wheels' input and poses.
#[path = "explicit_drive.rs"]
mod drive;

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
    /// The crash lab's per-beam constants (`crash_lab::lab_beam_words`).
    lab_beams: wgpu::Buffer,
    tri_count: u32,
    steps: u64,
    timed: bool,
    /// The `doorpull` scene's rig: its grab target moves out each frame.
    pull: Option<Pull>,
    /// Driving: the vehicles' poses from the last `readDrive`, and the self-driving
    /// cars' state (`traffic`).
    drive_poses: drive::Poses,
    ai: Vec<drive::AiCar>,
}

/// What [`ExplicitSim::spawn`] spawns: the vehicles, the static boxes and shapes,
/// the door-pull rig, and the scene's name.
struct Spawn {
    vehicles: Vec<Spawned>,
    boxes: Vec<(Vec3, Vec3)>,
    shapes: Vec<StaticShape>,
    pull: Option<Pull>,
    scene: String,
}

/// A door pull ([`engine_softbody::crash_rigs::add_door_pull`]): vehicle 0's rig
/// node pulled from `start` along `dir` at `speed` m/s by a grab spring `k`.
#[derive(Clone, Copy, Debug)]
struct Pull {
    rig: u32,
    start: Vec3,
    dir: Vec3,
    speed: f32,
    k: f32,
    c: f32,
}

/// The `doorpull` rig on a spawned vehicle (its frame: `forward`, `left` the side
/// pulled): the skin nodes nearest the front door's latch edge, at 60 % of the
/// body's height and 0.2 m ahead of its wheelbase's middle (**Choice**: a door
/// handle's place on a passenger car), joined to one rig node 0.3 m outside.
fn door_pull(sim: &mut ExplicitBeams, spec: &serde_json::Value, forward: Vec3, left: Vec3) -> Pull {
    let (lo, hi) = skin_bounds(sim);
    let centre = 0.5 * (lo + hi);
    // Axles along `forward` about the centre (the vehicle is turned to its yaw).
    let (mut f, mut r) = ((0.0f32, 0u32), (0.0f32, 0u32));
    for n in sim.nodes.iter().filter(|n| n.flags & NODE_TYRE != 0) {
        let z = (n.pos - centre).dot(forward);
        if z > 0.0 { f = (f.0 + z, f.1 + 1) } else { r = (r.0 + z, r.1 + 1) }
    }
    let mid = if f.1 > 0 && r.1 > 0 { 0.5 * (f.0 / f.1 as f32 + r.0 / r.1 as f32) } else { 0.0 };
    let reach = sim.nodes.iter().map(|n| n.pos.dot(left)).fold(f32::MIN, f32::max);
    let at = centre + forward * (mid + 0.2) + left * (reach - centre.dot(left));
    let at = Vec3::new(at.x, lo.y + 0.6 * (hi.y - lo.y), at.z);
    // The body held as a door test holds it (**Choice**): the far side's nodes in the
    // lowest 40 % of its height (its sill and floor) do not move, so the pull loads
    // the door's hinges and latch instead of dragging the car on its tyres.
    let far = sim.nodes.iter().map(|n| n.pos.dot(left)).fold(f32::MAX, f32::min);
    if spec.get("holdBody").and_then(|h| h.as_bool()).unwrap_or(true) {
        for n in &mut sim.nodes {
            if n.flags & NODE_TYRE == 0 && n.pos.dot(left) < far + 0.25 && n.pos.y < lo.y + 0.4 * (hi.y - lo.y) {
                n.flags |= NODE_IMMOVABLE;
            }
        }
    }
    let count = num(spec, "doorNodes", 6.0).max(1.0) as usize;
    let door = engine_softbody::crash_rigs::door_nodes(sim, left, at, count);
    let start = at + left * 0.3;
    let rig = engine_softbody::crash_rigs::add_door_pull(sim, &door, start);
    let k = num(spec, "pullStiffness", 100_000.0) as f32;
    let m = sim.nodes[rig as usize].mass;
    Pull {
        rig,
        start,
        dir: left,
        speed: num(spec, "pullSpeed", 0.5) as f32,
        k,
        c: (2.0 * (k * m).sqrt()).min(0.5 * m / STEP),
    }
}

/// The spawn's vehicle: `.truck` text, JBeam files (and `.pc`), or a desc.
fn build(
    v: &serde_json::Value,
    settings: ExplicitBeamsSettings,
) -> Result<(ExplicitBeams, SoftVehicleDesc, Vec<String>), String> {
    // `wheels: 'raycast'`: drivable, on raycast wheels (engine_softbody::explicit_drive)
    // in place of its tyres.
    let raycast = v.get("wheels").and_then(|w| w.as_str()) == Some("raycast");
    if raycast {
        return build_raycast(v, settings);
    }
    if let Some(text) = v.get("truck").and_then(|t| t.as_str()) {
        let mut sim = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings)?;
        // `fullContact`: JBeam's contact semantics for this `.truck` vehicle (every
        // node a contacter, spawn self-collision exclusions; spec §10.1–10.2).
        if v.get("fullContact").and_then(|f| f.as_bool()).unwrap_or(false) {
            sim.opt_in_full_contact();
        }
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
        if v.get("native").and_then(|n| n.as_bool()).unwrap_or(false) {
            // JBeam mode (jbeam-semantics-spec.md): BeamNG's rules, pressure wheels.
            let desc = engine_softbody::jbeam::load_jbeam(&refs, pc)?;
            let (sim, warnings) = ExplicitBeams::from_desc_jbeam_with(
                &desc,
                engine_softbody::explicit_beams::JbeamWheels::Pressure,
                Vec3::ZERO,
                Vec3::ZERO,
                settings,
            )?;
            return Ok((sim, desc, warnings));
        }
        let (sim, text, warnings) =
            ExplicitBeams::from_jbeam(&refs, pc, Vec3::ZERO, Vec3::ZERO, settings)?;
        return Ok((sim, engine_softbody::truck::load_truck(&text), warnings));
    }
    if let Some(d) = v.get("desc") {
        let desc: SoftVehicleDesc =
            serde_json::from_value(d.clone()).map_err(|e| format!("desc: {e}"))?;
        // `native`: a desc loaded from JBeam, run in JBeam mode (pressure wheels).
        if v.get("native").and_then(|n| n.as_bool()).unwrap_or(false) {
            let (sim, warnings) = ExplicitBeams::from_desc_jbeam_with(
                &desc,
                engine_softbody::explicit_beams::JbeamWheels::Pressure,
                Vec3::ZERO,
                Vec3::ZERO,
                settings,
            )?;
            return Ok((sim, desc, warnings));
        }
        let sim = ExplicitBeams::new(&desc, Vec3::ZERO, Vec3::ZERO, settings);
        return Ok((sim, desc, Vec::new()));
    }
    Err("a vehicle needs `truck`, `jbeam` or `desc`".into())
}

/// The drive scenes' static shapes beside their boxes: the yard's five 15 cm
/// poles across z = 34 and its ramp at x = −18 (6 × 8 m, rising toward +Z at 14°).
fn drive_shapes(scene: &str) -> Vec<StaticShape> {
    match scene {
        "yard" => {
            let mut v: Vec<StaticShape> = (0..5)
                .map(|i| StaticShape::cylinder(Vec3::new(-14.0 + 3.5 * i as f32, 1.5, 34.0), 0.15, 1.5))
                .collect();
            v.push(StaticShape::oriented_box(
                Vec3::new(-18.0, 0.5, 10.0),
                Vec3::new(3.0, 0.3, 4.0),
                Quat::from_rotation_x(-0.25),
            ));
            v
        }
        "traffic" => {
            let mut v: Vec<StaticShape> = (0..10)
                .map(|i| {
                    let a = i as f32 * 2.39996;
                    let r = 18.0 + (i % 4) as f32 * 11.0;
                    StaticShape::oriented_box(
                        Vec3::new(a.cos() * r, 0.5, a.sin() * r),
                        Vec3::new(1.5, 0.5, 0.5),
                        Quat::from_rotation_y(a),
                    )
                })
                .collect();
            v.extend((0..8).map(|i| {
                let a = i as f32 * 0.785 + 0.4;
                StaticShape::cylinder(Vec3::new(a.cos() * 32.0, 1.5, a.sin() * 32.0), 0.15, 1.5)
            }));
            v
        }
        _ => Vec::new(),
    }
}

/// [`build`] on raycast wheels: `.truck` text, JBeam files (`native`: JBeam mode,
/// else through the `.truck` export) or a JBeam-mode desc.
fn build_raycast(
    v: &serde_json::Value,
    settings: ExplicitBeamsSettings,
) -> Result<(ExplicitBeams, SoftVehicleDesc, Vec<String>), String> {
    let native = v.get("native").and_then(|n| n.as_bool()).unwrap_or(false);
    let truck = |text: &str| -> Result<(ExplicitBeams, SoftVehicleDesc, Vec<String>), String> {
        let (mut sim, notes) =
            ExplicitBeams::from_truck_raycast(text, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)?;
        if v.get("fullContact").and_then(|f| f.as_bool()).unwrap_or(false) {
            sim.opt_in_full_contact();
        }
        Ok((sim, engine_softbody::truck::load_truck(text), notes))
    };
    if let Some(text) = v.get("truck").and_then(|t| t.as_str()) {
        return truck(text);
    }
    if let Some(files) = v.get("jbeam").and_then(|t| t.as_array()) {
        let files: Vec<(String, String)> = files
            .iter()
            .map(|f| {
                (
                    f.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    f.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                )
            })
            .collect();
        let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let pc = v.get("pc").and_then(|t| t.as_str());
        let desc = engine_softbody::jbeam::load_jbeam(&refs, pc)?;
        if native {
            let (sim, w) = ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings)?;
            return Ok((sim, desc, w));
        }
        let (text, mut w) = engine_softbody::truck_export::desc_to_truck(
            &desc,
            engine_softbody::truck_export::TyreDefaults::for_vehicle(&desc),
        );
        let (sim, d, mut notes) = truck(&text)?;
        w.append(&mut notes);
        return Ok((sim, d, w));
    }
    if let Some(d) = v.get("desc") {
        let desc: SoftVehicleDesc = serde_json::from_value(d.clone()).map_err(|e| format!("desc: {e}"))?;
        let (sim, w) = ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings)?;
        return Ok((sim, desc, w));
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

/// The front and rear axles' positions along +Z relative to the body's middle
/// (`lo`, `hi`: its skin box), facing +Z: the mean of the tyre nodes ahead of and
/// behind the middle; ±30 % of the length without tyres.
fn axles(sim: &ExplicitBeams, lo: Vec3, hi: Vec3) -> (f32, f32) {
    let mid = 0.5 * (lo.z + hi.z);
    let (mut f, mut r) = ((0.0f32, 0u32), (0.0f32, 0u32));
    for n in sim.nodes.iter().filter(|n| n.flags & NODE_TYRE != 0) {
        let z = n.pos.z - mid;
        if z > 0.0 { f = (f.0 + z, f.1 + 1) } else { r = (r.0 + z, r.1 + 1) }
    }
    let len = hi.z - lo.z;
    if f.1 == 0 || r.1 == 0 {
        return (0.3 * len, -0.3 * len);
    }
    (f.0 / f.1 as f32, r.0 / r.1 as f32)
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
    // Raycast wheels roll at the forward speed (their frame's forward).
    if let Some(d) = sim.drive.as_mut() {
        let [c, b, _] = d.config.refs;
        let f = (sim.nodes[c as usize].pos - sim.nodes[b as usize].pos).normalize_or_zero();
        d.roll(velocity.dot(f));
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
                mdb: false,
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
    /// The FMVSS 214 moving deformable barrier spawns here instead of the vehicle
    /// (`engine_softbody::crash_rigs`).
    mdb: bool,
}

/// The crash-test scenes, built from one vehicle (`width` across, `length` along at
/// yaw 0, facing +Z). Speeds in m/s, angles in degrees; each moving vehicle starts
/// 1 + 0.5 × its speed metres from what it hits (`runUp` metres when given), so the
/// hit lands after about half a second:
/// * `wall`: into a rigid wall at `speed`; `overlap` (0 to 1, default 1) of the
///   width meets it (the rest passes beside it), the vehicle turned by `angle`.
/// * `headon`: two copies nose to nose at `speed` and `speed2` (default `speed`);
///   `overlap` of the width overlaps, the second car turned by `angle` from straight
///   on.
/// * `tbone`: into the side of a second copy standing (or moving forward at
///   `carSpeed`) across its lane, turned `angle` (default 90) from its heading, hit
///   `at` metres along it from its centre.
/// * `pole`: into a 20 cm square pole `offset` metres across, turned by `angle`.
/// * `sidepole`: carried sideways at `speed` into a 20 cm pole `at` metres along its
///   side from its centre; `angle` (default 90) the travel's angle from its forward
///   axis (75: the oblique pole test). The car rides a carrier, as in the real test
///   (its tyres on low-friction pads on a sled, FMVSS 214 S10): its tyres' grip on
///   the ground is `crash_rigs::CARRIER_MU` for the whole scene, so it reaches the pole at
///   `speed` instead of being stopped by its own tyres' lateral grip.
/// * `limbo`: under a bar 30 cm deep whose underside is `bar` metres up (default 1).
/// * `drop`: dropped flat from `height` metres (default 8), moving forward at `speed`
///   (default 0).
/// * `mdb214`: FMVSS 214 (49 CFR 571.214 S7, S8.10–S8.11): the 1,367.6 kg moving
///   deformable barrier (`engine_softbody::crash_rigs`) into the vehicle's left
///   (driver's) side at `speed` (default 33.5 mph), crabbed `crab` degrees
///   (default 27: its wheels turned toward the vehicle's rear, so it moves rearward
///   along the struck side), its face square to the side and the face's forward edge
///   on the impact reference line: 940 mm ahead of the wheelbase's centre (508 mm
///   behind the front axle past a 2,896 mm wheelbase). The vehicle stands.
/// * `doorpull`: the vehicle at rest, its left front door's latch edge pulled
///   outward through `doorNodes` (default 6) of its skin nodes by one rig
///   ([`engine_softbody::crash_rigs::add_door_pull`]) at `pullSpeed` m/s (default
///   0.5) with a `pullStiffness` N/m (default 100,000) grab spring; its far side's
///   sill held still unless `holdBody` is false.
/// Returns the spawns and every box once (for rendering); None for another scene.
///
/// A vehicle launched off its forward axis (`vel` set: the side pole) rides a carrier:
/// `engine_softbody::crash_rigs::on_carrier`.
fn crash_scene(scene: &str, spec: &serde_json::Value, width: f32, length: f32, axles: (f32, f32)) -> Option<(Vec<ArenaSpot>, Vec<(Vec3, Vec3)>)> {
    use std::f32::consts::PI;
    let deg = |k: &str, d: f64| (num(spec, k, d) as f32).to_radians();
    let v = num(spec, "speed", 27.78) as f32;
    let run_up = |v: f32| 1.0 + 0.5 * v;
    // `runUp`: the gap (m) between the vehicle and what it hits, default 1 + v/2 s.
    let gap = num(spec, "runUp", f64::from(run_up(v))) as f32;
    let spot = |at: Vec3, yaw: f32, speed: f32, boxes: Vec<(Vec3, Vec3)>, vel: Option<Vec3>| ArenaSpot { at, yaw, speed, boxes, vel, lift: 0.0, mdb: false };
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
        "yard" => {
            // The drive yard (the soft solver's `yard`, softbody.html): the player car at
            // rest facing +Z, two cars parked side-on at x = 14 (z = 20 and 30), a
            // wall 3 m high across z = 60 and a row of poles and a ramp
            // (`drive_shapes`). Every copy drives on the spawn's wheels.
            let wall = (Vec3::new(0.0, 1.5, 60.0), Vec3::new(8.0, 1.5, 0.3));
            let parked = |z: f32, yaw: f32| {
                let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
                spot(Vec3::new(14.0, 0.0, z) + fwd * (0.5 * length), yaw, 0.0, vec![wall], None)
            };
            (
                vec![
                    spot(Vec3::new(0.0, 0.0, 0.5 * length), 0.0, 0.0, vec![wall], None),
                    parked(20.0, 0.5 * PI),
                    parked(30.0, -0.5 * PI),
                ],
                vec![wall],
            )
        }
        "traffic" => {
            // The traffic derby (the soft solver's `traffic`): `cars` copies (default 20,
            // at most 64) on a grid in a walled 140 m arena with concrete blocks and
            // poles (`drive_shapes`), each turned its own way; they drive themselves
            // (ExplicitSim::autopilot).
            let n = (num(spec, "cars", 20.0) as usize).clamp(1, 64);
            let cols = (n as f32).sqrt().ceil() as usize;
            let a = 70.0;
            let walls: Vec<(Vec3, Vec3)> = [(0.0, a, a, 0.5), (0.0, -a, a, 0.5), (a, 0.0, 0.5, a), (-a, 0.0, 0.5, a)]
                .iter()
                .map(|&(x, z, hx, hz)| (Vec3::new(x, 1.0, z), Vec3::new(hx, 1.0, hz)))
                .collect();
            let spots = (0..n)
                .map(|i| {
                    let c = (cols as f32 - 1.0) / 2.0;
                    let at = Vec3::new(((i % cols) as f32 - c) * 9.0, 0.0, ((i / cols) as f32 - c) * 11.0);
                    let yaw = (i as f32 * 1.7) % std::f32::consts::TAU;
                    let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
                    spot(at + fwd * (0.5 * length), yaw, 0.0, walls.clone(), None)
                })
                .collect();
            (spots, walls)
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
            let at = num(spec, "at", 0.0) as f32;
            // Forward +X, its centre at the origin, moving along +Z (sideways), or
            // with `angle` (degrees from its forward axis, default 90) obliquely, as an
            // oblique pole test at 75°: the pole on the path of the side's point `at`.
            let angle = num(spec, "angle", 90.0) as f32;
            let (b, vel) = if angle == 90.0 {
                (pole(at, 0.5 * width + gap + 0.1), Vec3::new(0.0, 0.0, v))
            } else {
                let d = Vec3::new(angle.to_radians().cos(), 0.0, angle.to_radians().sin());
                let p = Vec3::new(at, 0.0, 0.5 * width) + d * (gap + 0.1);
                (pole(p.x, p.z), d * v)
            };
            (vec![spot(Vec3::new(0.5 * length, 0.0, 0.0), 0.5 * PI, v, vec![b], Some(vel))], vec![b])
        }
        "limbo" => {
            let bar = num(spec, "bar", 1.0) as f32;
            let b = (Vec3::new(0.0, bar + 0.15, gap + 0.15), Vec3::new(0.5 * width + 1.0, 0.15, 0.15));
            (vec![spot(Vec3::ZERO, 0.0, v, vec![b], None)], vec![b])
        }
        "mdb214" => {
            let v = num(spec, "speed", f64::from(engine_softbody::crash_rigs::MDB_SPEED)) as f32;
            let crab = deg("crab", f64::from(engine_softbody::crash_rigs::MDB_CRAB_DEG));
            let (front_axle, rear_axle) = axles;
            let wb = front_axle - rear_axle;
            let irl = if wb <= 2.896 { 0.5 * (front_axle + rear_axle) + 0.94 } else { front_axle - 0.508 };
            let half_face = 0.5 * engine_softbody::crash_rigs::MDB_FACE[0];
            // Facing −X (yaw −90°), its face's middle at the struck side when it hits.
            let contact = Vec3::new(0.5 * width, 0.0, irl - half_face);
            let dir = Vec3::new(-crab.cos(), 0.0, -crab.sin());
            let gap = num(spec, "runUp", f64::from(run_up(v))) as f32;
            let mut barrier = spot(contact - dir * gap, -0.5 * PI, v, Vec::new(), Some(dir * v));
            barrier.mdb = true;
            (vec![spot(Vec3::new(0.0, 0.0, 0.5 * length), 0.0, 0.0, Vec::new(), None), barrier], Vec::new())
        }
        "doorpull" => (vec![spot(Vec3::new(0.0, 0.0, 0.5 * length), 0.0, 0.0, Vec::new(), None)], Vec::new()),
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
                mdb: false,
            }
        })
        .collect();
    (spots, all)
}

#[wasm_bindgen]
impl WebWorld {
    /// Explicit-beams vehicles on this world's device. `spec`: `{ vehicles: [{
    /// truck (+ fullContact: true for JBeam contact semantics) | jbeam: [[name, text], …] (+ pc, + native: true for JBeam mode with pressure
    /// wheels) | desc, at: [x, z], yaw, front }], scene:
    /// 'idle' | 'coast' | 'wall' | 'arena', speed (m/s, coast and wall), wallGap (m,
    /// default 1 + 0.5·speed); arena (see `arena`): count, cols, speedMin, speedMax,
    /// hitMin, hitMax, headOn, layout ('rows' | 'ring', see `ring`), ringRadius;
    /// shapes (static colliders for every vehicle, see `shapes`); the
    /// first vehicle copied to every spawn, drag, turbulence, fastInvSqrt (default true),
    /// centredDamping (the BeamNG profile's centred damping, spec §39; default the
    /// profile's), timing
    /// (default true: GPU timestamps when the device has them) }`.
    #[wasm_bindgen(js_name = explicitSim)]
    pub fn explicit_sim(&self, spec: JsValue) -> Result<ExplicitSim, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        ExplicitSim::new(self.world.device(), self.world.queue(), &spec).map_err(err)
    }

    /// [`Self::explicit_sim`], its GPU kernels compiled first without blocking
    /// (`createComputePipelineAsync`, see `crate::warm_pipelines`): a promise of
    /// the `ExplicitSim`. Load with this; the synchronous call is then a cache hit
    /// (a restart may use it). `progress(line)` (optional) hears each stage
    /// (`crate::Hear`: `spawn`, `kernels <json>`, then `compiled`/`failed` per
    /// kernel; a kernel the first frame does not need compiles after the promise
    /// resolves). Rejects, without creating anything more, if the browser loses the
    /// device while the kernels compile.
    #[wasm_bindgen(js_name = explicitSimAsync)]
    pub fn explicit_sim_async(
        &self,
        spec: JsValue,
        progress: Option<js_sys::Function>,
    ) -> Result<js_sys::Promise, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        let (device, queue) = (self.world.device().clone(), self.world.queue().clone());
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let say = crate::hear_fn(progress);
            say("spawn".into());
            let spawn = ExplicitSim::spawn(&spec).map_err(err)?;
            let sims: Vec<ExplicitBeams> = spawn.vehicles.iter().map(|v| v.sim.clone()).collect();
            let specs = ExplicitBeamsGpu::compute_specs(&sims, None);
            crate::warm_pipelines(&device, &specs, say).await.map_err(err)?;
            ExplicitSim::assemble(&device, &queue, &spec, spawn)
                .map(JsValue::from)
                .map_err(err)
        }))
    }
}

impl ExplicitSim {
    /// [`WebWorld::explicit_sim`] on a native device (tests and tools).
    #[doc(hidden)]
    pub fn new_native(device: &wgpu::Device, queue: &wgpu::Queue, spec: &serde_json::Value) -> Result<Self, String> {
        Self::new(device, queue, spec)
    }

    /// The GPU solver (tests and tools).
    #[doc(hidden)]
    pub fn gpu_solver(&self) -> &ExplicitBeamsGpu {
        &self.gpu
    }

    /// Vehicle `i`'s spawn state (its CPU copy at spawn).
    #[doc(hidden)]
    pub fn spawned(&self, i: usize) -> &ExplicitBeams {
        &self.vehicles[i].sim
    }

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spec: &serde_json::Value,
    ) -> Result<Self, String> {
        let spawn = Self::spawn(spec)?;
        Self::assemble(device, queue, spec, spawn)
    }

    /// The vehicles `spec` spawns and the scene around them (CPU only).
    fn spawn(spec: &serde_json::Value) -> Result<Spawn, String> {
        let flag = |k: &str| spec.get(k).and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: flag("fastInvSqrt"),
            drag: flag("drag"),
            turbulence: flag("turbulence") && flag("drag"),
            ..Default::default()
        };
        let scene = spec.get("scene").and_then(|s| s.as_str()).unwrap_or("idle");
        const SCENES: [&str; 14] = ["idle", "coast", "wall", "arena", "headon", "tbone", "pole", "sidepole", "limbo", "drop", "mdb214", "doorpull", "yard", "traffic"];
        if !SCENES.contains(&scene) {
            return Err(format!("unknown explicit scene {scene}: {}", SCENES.join(", ")));
        }
        let speed = if scene == "idle" || scene == "doorpull" || scene == "yard" || scene == "traffic" {
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
        let mut shapes = shapes(spec)?;
        shapes.extend(drive_shapes(scene));
        // `arena`: the first vehicle, built once, copied to every spawn.
        let mut spots = Vec::new();
        let mut pull: Option<Pull> = None;
        let template = if scene == "arena" || crash_scene(scene, spec, 1.0, 1.0, (1.0, -1.0)).is_some() {
            let (mut sim, desc, warnings) = build(&list[0], settings)?;
            let f = engine_softbody::pack::forward_axis(&desc);
            turn(&mut sim, -f.x.atan2(f.z));
            // The body's extent: its collision-triangle nodes (a file's helper nodes,
            // such as camera nodes, can sit outside it), else every node.
            let (lo, hi) = skin_bounds(&sim);
            let (s, all) = crash_scene(scene, spec, hi.x - lo.x, hi.z - lo.z, axles(&sim, lo, hi))
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
                _ if spots.get(vi).is_some_and(|s| s.mdb) => build(
                    &serde_json::json!({ "jbeam": [["mdb214.jbeam", engine_softbody::crash_rigs::mdb214_jbeam()]], "native": true }),
                    settings,
                )?,
                Some(t) => t.clone(),
                None => build(v, settings)?,
            };
            // `centredDamping`: the BeamNG profile's centred damping on or off
            // (jbeam-semantics-spec.md §39; its default otherwise). A Rigs of Rods
            // profile vehicle is never changed.
            if let Some(c) = spec.get("centredDamping").and_then(|c| c.as_bool()) {
                if sim.beamng_profile {
                    sim.settings.centred_damping = c;
                }
            }
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
            // Raycast wheels: the lowest wheel's bottom 2 cm up (the suspension
            // fully extended; it settles by its sag), or the body's lowest node at
            // the usual clearance, whichever is higher.
            let wheel_low = sim.drive.as_ref().map(|d| {
                d.config
                    .wheels
                    .iter()
                    .map(|w| {
                        let y = 0.5 * (sim.nodes[w.node1 as usize].pos.y + sim.nodes[w.node2 as usize].pos.y);
                        y - w.radius
                    })
                    .fold(f32::MAX, f32::min)
            });
            let rise = match wheel_low {
                Some(wl) => (sim.spawn_clearance() - lowest).max(0.02 - wl),
                None => sim.spawn_clearance() - lowest,
            };
            let shift = forward * along + lateral * across + Vec3::Y * (rise + lift);
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
            if scene == "doorpull" && vi == 0 {
                pull = Some(door_pull(&mut sim, spec, forward, lateral));
            }
            set_moving(&mut sim, spots.get(vi).and_then(|s| s.vel).unwrap_or(forward * speed));
            if spots.get(vi).is_some_and(|s| s.vel.is_some() && !s.mdb) {
                engine_softbody::crash_rigs::on_carrier(&mut sim);
            }
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
            // `surfaceContact`: the collision triangles meet the boxes and shapes by
            // their surface too (jbeam-semantics-spec.md §38; off by default).
            if spec.get("surfaceContact").and_then(|x| x.as_bool()).unwrap_or(false) {
                sim.surface_contact = true;
            }
            vehicles.push(Spawned {
                name: desc.name.clone(),
                sim,
                reference,
                forward,
                warnings,
            });
        }
        Ok(Spawn {
            vehicles,
            boxes,
            shapes,
            pull,
            scene: scene.to_string(),
        })
    }

    /// The solver on `device` for what [`Self::spawn`] spawned.
    fn assemble(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spec: &serde_json::Value,
        spawn: Spawn,
    ) -> Result<Self, String> {
        let Spawn {
            vehicles,
            boxes,
            shapes,
            pull,
            scene,
        } = spawn;
        let scene = scene.as_str();
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
        // `traffic`: every car drives itself (`ai`: 'derby' (default: two in three
        // wander, one chases), 'cruise' (wander, brake for cars ahead) or 'off').
        let ai = if scene == "traffic" {
            drive::ai_cars(vehicles.len(), spec.get("ai").and_then(|a| a.as_str()).unwrap_or("derby"))
        } else {
            Vec::new()
        };
        let tris_buf = init("explicit-render-tris", &tris);
        let recs_buf = init("explicit-render-beams", &recs);
        let lab: Vec<(&ExplicitBeams, Vec3)> = vehicles.iter().map(|v| (&v.sim, v.forward)).collect();
        let lab_beams = init("explicit-lab-beams", &crash_lab::lab_beam_words(&lab));
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
            lab_beams,
            steps: 0,
            timed,
            pull,
            drive_poses: Default::default(),
            ai,
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

    /// Submit one frame of `steps` fixed steps for every vehicle (never waits). A
    /// contact error (a vehicle past its frame bounds: the margin widens by itself)
    /// is a console warning and the sim keeps stepping; other errors are returned.
    pub fn frame(&mut self, steps: u32) -> Result<(), JsValue> {
        if crate::device_lost() {
            return Err(err(format!("the browser lost the GPU device: {}", crate::gpu_error())));
        }
        self.frame_steps(steps).map_err(err)?;
        for e in self.gpu.take_contact_errors() {
            let note = format!("explicit beams vehicle contact (the sim continues): {e}");
            #[cfg(target_arch = "wasm32")]
            web_sys::console::warn_1(&note.into());
            #[cfg(not(target_arch = "wasm32"))]
            eprintln!("{note}");
        }
        Ok(())
    }
}

impl ExplicitSim {
    /// [`Self::frame`] for native hosts and tests (errors as text, a contact
    /// error too: the sim keeps stepping).
    #[doc(hidden)]
    pub fn frame_native(&mut self, steps: u32) -> Result<(), String> {
        self.frame_steps(steps)?;
        // A contact error (a vehicle past its frame bounds: the margin widens by
        // itself) never stops the sim: the caller gets it and keeps stepping.
        let errors = self.gpu.take_contact_errors();
        if !errors.is_empty() {
            return Err(format!("vehicle contact (the sim continues): {}", errors.join("; ")));
        }
        Ok(())
    }

    /// One frame's steps; contact errors are left for the caller.
    fn frame_steps(&mut self, steps: u32) -> Result<(), String> {
        if steps > MAX_FRAME_STEPS {
            return Err(format!(
                "at most {MAX_FRAME_STEPS} steps per frame, got {steps}"
            ));
        }
        if let Some(p) = self.pull {
            // The rig's target where it is at this frame's end (vehicle frame: the
            // vehicle's origin stays where it spawned).
            let t = (self.steps + u64::from(steps)) as f32 * STEP;
            let target = self.vehicles[0].sim.origin + p.start + p.dir * (p.speed * t);
            self.gpu.set_grab(Some((0, engine_softbody::explicit_beams::Grab { node: p.rig, target, k: p.k, c: p.c })));
        }
        if steps > 0 {
            self.gpu.frame(steps);
            self.steps += u64::from(steps);
        }
        Ok(())
    }
}

#[wasm_bindgen]
impl ExplicitSim {

    /// The stability check of this sim's vehicles (`stability.rs`: read-only; the
    /// page stops stepping when it reports).
    pub fn health(&self) -> crate::stability::VehicleHealth {
        crate::stability::VehicleHealth::new(&self.device, &self.queue, &self.gpu)
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
                    // The solver profile it runs (spec §36): "beamng" or "ror".
                    "profile": if v.sim.beamng_profile { "beamng" } else { "ror" },
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
