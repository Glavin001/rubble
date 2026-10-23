//! Pack a [`VehicleDesc`] into the `gVehicleData` records plus the bodies and
//! colliders a host appends (chassis, parked anchor, parked wheel/part bodies).
//! The browser demo runs this code too (`engine-vehicle-wasm`).

use crate::desc::*;
use engine_schema::vehicle::*;
use engine_schema::{BodyFlags, BodyState, Collider, Joint, ShapeType, JOINT_FLAG_TYRE};
use glam::{Mat3, Quat, Vec3};

pub type Slot = [u32; 4];

/// Where this vehicle's records and objects land in the host's buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VehicleSlots {
    pub vehicle: u32,
    pub wheel_base: u32,
    pub curve_point_base: u32,
    pub diff_base: u32,
    pub anti_roll_base: u32,
    pub part_base: u32,
    pub body_base: u32,
    pub collider_base: u32,
    /// Absolute gAabbs index of this vehicle's 4-entry block.
    pub aabb_block: u32,
    /// First of this vehicle's tyre joints (one per wheel slot).
    pub joint_base: u32,
}

/// Pool sizes a vehicle needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Requirements {
    pub wheel_slots: u32,
    pub curve_points: u32,
    pub diffs: u32,
    pub anti_rolls: u32,
    pub parts: u32,
    pub bodies: u32,
    pub colliders: u32,
}

#[derive(Clone, Debug, Default)]
pub struct PackedVehicle {
    /// chassis, anchor, wheel parked bodies (one per wheel slot), part parked bodies.
    pub bodies: Vec<BodyState>,
    /// chassis part colliders, then rim spheres per wheel slot.
    pub colliders: Vec<Collider>,
    pub vehicle: Vec<Slot>,
    pub wheels: Vec<Vec<Slot>>,
    pub curve_points: Vec<[f32; 2]>,
    pub diffs: Vec<Vec<Slot>>,
    pub anti_rolls: Vec<Slot>,
    pub parts: Vec<Vec<Slot>>,
    /// One tyre joint (JOINT_FLAG_TYRE) per wheel slot, parked on the static anchor until a wheel
    /// stands on a dynamic body (vehicle_link).
    pub joints: Vec<Joint>,
    /// Chassis total mass and inertia (body axes).
    pub mass: f32,
    pub inertia: [f32; 3],
}

/// Far-away rest pose for parked (inactive) bodies.
pub fn parked_position(index: u32) -> [f32; 3] {
    [-5000.0 + 9.0 * (index % 1024) as f32, -5000.0, -5000.0 + 9.0 * (index / 1024) as f32]
}

pub const RIM_SPHERES_MAX: u32 = 32;
/// Default material id for detached-wheel rim spheres (renderers may hide it).
pub const WHEEL_RIM_MATERIAL: u32 = 7;

fn f(v: f32) -> u32 {
    v.to_bits()
}

fn slot3(v: Vec3, w: f32) -> Slot {
    [f(v.x), f(v.y), f(v.z), f(w)]
}

fn slot3u(v: Vec3, w: u32) -> Slot {
    [f(v.x), f(v.y), f(v.z), w]
}

fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::from_array(a)
}

/// A mount limit as packed: ≤ 0 (unbreakable) becomes 1e30, as `JOINT_BIG`.
pub fn mount_limit(v: f32) -> f32 {
    if v > 0.0 { v } else { 1e30 }
}

fn rim_count(w: &WheelDesc) -> u32 {
    w.rim_spheres.clamp(1, RIM_SPHERES_MAX)
}

pub fn requirements(desc: &VehicleDesc) -> Requirements {
    let wheel_slots = desc.wheels.len() as u32 + desc.spare_wheels;
    let spare_rims = desc.wheels.first().map(rim_count).unwrap_or(12) * desc.spare_wheels;
    let mut curve_points = 0;
    for w in &desc.wheels {
        curve_points += w.long_friction.0.len() as u32 + w.lat_friction.0.len() as u32;
    }
    curve_points += desc.steering.speed_curve.0.len() as u32;
    if let DriveDesc::Drivetrain(d) = &desc.drive {
        curve_points += d.engine.torque_curve.0.len() as u32;
    }
    // Spare wheel slots reuse the first wheel's curves.
    let diffs = match &desc.drive {
        DriveDesc::Drivetrain(d) => d.differentials.len() as u32,
        DriveDesc::Direct { .. } => 0,
    };
    Requirements {
        wheel_slots,
        curve_points,
        diffs,
        anti_rolls: desc.anti_roll.len() as u32,
        parts: desc.parts.len() as u32,
        bodies: 2 + wheel_slots + desc.parts.len() as u32,
        colliders: desc.parts.iter().map(|p| p.colliders.len() as u32).sum::<u32>()
            + desc.wheels.iter().map(rim_count).sum::<u32>()
            + spare_rims
            + wheel_slots * SHELL_SPHERES
            + track_pods(desc, 0).iter().map(|c| c.len() as u32).sum::<u32>(),
    }
}

/// Belt boxes of a track pod (see [`TrackPodDesc`]) on `body`, chassis space.
/// Each box's local +X is the axle: world_step runs the belt around it.
///
/// Side view (fwd →): an upper run over every wheel from the end wheels' axle
/// height to `top`; a lower run between the first and last road-wheel axles down
/// to the road wheels' tread line; and a ramp at each end from that tread line up
/// to the end wheel's axle height at its rim (the sloped run up to a raised
/// sprocket/idler, or a chamfer when the ends are not raised). An edge met on a
/// ramp pushes the nose up even with the track stalled.
pub fn track_pod_colliders(desc: &VehicleDesc, tr: &TrackDesc, pod: &TrackPodDesc, body: u32) -> Vec<Collider> {
    let up = v3(desc.up).normalize();
    let fwd = v3(desc.forward).normalize();
    let axle = up.cross(fwd).normalize();
    let mut ws: Vec<&WheelDesc> = tr.wheels.iter().filter_map(|&i| desc.wheels.get(i as usize)).collect();
    if ws.is_empty() {
        return Vec::new();
    }
    ws.sort_by(|a, b| v3(a.position).dot(fwd).total_cmp(&v3(b.position).dot(fwd)));
    // Bottomed-out axle height and fore/aft position (chassis up / fwd).
    let axle_h = |w: &WheelDesc| {
        let down = w.suspension_dir.map(v3).unwrap_or(-up).normalize();
        (v3(w.position) + down * w.suspension_min).dot(up)
    };
    let along = |w: &WheelDesc| v3(w.position).dot(fwd);
    let width = if pod.width > 0.0 { pod.width } else { ws.iter().map(|w| w.width).fold(0.0, f32::max) };
    let lat = ws.iter().map(|w| v3(w.position).dot(axle)).sum::<f32>() / ws.len() as f32;
    let top = ws.iter().map(|w| v3(w.position).dot(up)).fold(f32::MIN, f32::max) + pod.top;
    let low = ws.iter().map(|w| axle_h(w) - w.radius).fold(f32::MAX, f32::min) + pod.clearance;
    let road: Vec<&WheelDesc> = ws.iter().copied().filter(|w| axle_h(w) - w.radius + pod.clearance < low + 1e-3).collect();
    let (rear, front) = (ws[0], ws[ws.len() - 1]);
    let (road_rear, road_front) = (along(road[0]), along(road[road.len() - 1]));
    let run_h = ws.iter().map(|w| axle_h(w)).fold(f32::MIN, f32::max) + pod.clearance;
    let boxed = |center: glam::Vec2, half: Vec3, pitch: f32| {
        // center/half in (fwd, up); pitch rotates local fwd toward up about the axle.
        let (s, c) = pitch.sin_cos();
        let (bf, bu) = (fwd * c + up * s, up * c - fwd * s);
        Collider {
            body,
            shape: ShapeType::Box as u32,
            material: pod.material,
            generation: 0,
            local_pos: (axle * lat + fwd * center.x + up * center.y).to_array(),
            density: 0.0,
            local_rot: Quat::from_mat3(&Mat3::from_cols(axle, bu, bf)).normalize().to_array(),
            params: [half.x, half.y, half.z, half.length()],
        }
    };
    let hw = 0.5 * width;
    let mut out = Vec::new();
    // Upper run.
    let (lo, hi) = (along(rear) - rear.radius, along(front) + front.radius);
    out.push(boxed(
        glam::Vec2::new(0.5 * (lo + hi), 0.5 * (run_h + top)),
        Vec3::new(hw, 0.5 * (top - run_h), 0.5 * (hi - lo)),
        0.0,
    ));
    // Lower run between the end road-wheel axles.
    if road_front - road_rear > 1e-3 {
        out.push(boxed(
            glam::Vec2::new(0.5 * (road_rear + road_front), 0.5 * (low + top)),
            Vec3::new(hw, 0.5 * (top - low), 0.5 * (road_front - road_rear)),
            0.0,
        ));
    }
    // Ramps: bottom face from (end road axle, low) to (end wheel rim, run_h).
    let depth = run_h - low + 0.1;
    for (from, to) in [
        (glam::Vec2::new(road_front, low), glam::Vec2::new(along(front) + front.radius, run_h)),
        (glam::Vec2::new(road_rear, low), glam::Vec2::new(along(rear) - rear.radius, run_h)),
    ] {
        let d = to - from;
        let len = d.length();
        if len < 1e-3 {
            continue;
        }
        let dir = d / len * d.x.signum();
        let pitch = dir.y.atan2(dir.x);
        let inward = glam::Vec2::new(-dir.y, dir.x);
        out.push(boxed(0.5 * (from + to) + inward * (0.5 * depth), Vec3::new(hw, 0.5 * depth, 0.5 * len), pitch));
    }
    out
}

/// Pod colliders of the left and right tracks (empty where a track has none).
fn track_pods(desc: &VehicleDesc, body: u32) -> [Vec<Collider>; 2] {
    match &desc.tracks {
        Some(t) => [&t.left, &t.right].map(|tr| match &tr.pod {
            Some(pod) => track_pod_colliders(desc, tr, pod, body),
            None => Vec::new(),
        }),
        None => [Vec::new(), Vec::new()],
    }
}

struct CurveSink<'a> {
    base: u32,
    points: &'a mut Vec<[f32; 2]>,
}

impl CurveSink<'_> {
    fn push(&mut self, c: &Curve) -> u32 {
        if c.0.is_empty() {
            return NO_INDEX;
        }
        let start = self.base + self.points.len() as u32;
        assert!(start + c.0.len() as u32 <= 0xFFFF, "curve pool exceeds 16-bit point index");
        self.points.extend_from_slice(&c.0);
        start | ((c.0.len() as u32) << 16)
    }
}

/// Wheel spin-axis inertia diag for the free wheel body (axle along `axle`).
fn wheel_body_inertia(w: &WheelDesc, axle: Vec3) -> Vec3 {
    let r = w.radius;
    let ia = 0.5 * w.mass * r * r;
    let im = w.mass * (3.0 * r * r + w.width * w.width) / 12.0;
    Vec3::splat(im) + (ia - im) * axle * axle
}

struct WheelFrame {
    up: Vec3,
    fwd: Vec3,
}

/// Body record for a wheel slot that is inactive (spare) or parked.
fn parked_body(index: u32, mass: f32, inertia: Vec3) -> BodyState {
    let mut b = BodyState::dynamic(v3(parked_position(index)), Quat::IDENTITY, mass, inertia);
    b.flags = BodyFlags::None as u32;
    b.initial_pos = b.position;
    b
}

/// Rim spheres for a wheel's free body (`n` spheres around the axle).
pub fn rim_colliders_n(w: &WheelDesc, up: Vec3, fwd: Vec3, body: u32, n: u32) -> Vec<Collider> {
    let frame = WheelFrame { up, fwd };
    rim_colliders_frame(w, &frame, body, n)
}

/// Shell sphere radius and ring radius: as wide as the tyre (≤ 0.4 R), inside it.
pub fn shell_geometry(radius: f32, width: f32) -> (f32, f32) {
    let rs = (0.5 * width).min(0.4 * radius).max(1e-3);
    (rs, (radius - rs).max(0.0))
}

/// Shell spheres around a wheel centre at `center` (chassis space), in the plane of
/// `down` (suspension direction) and `fwd`, at 60°…300° from straight down.
pub fn shell_colliders(w: &WheelDesc, center: Vec3, down: Vec3, fwd: Vec3, body: u32) -> Vec<Collider> {
    let (rs, ring) = shell_geometry(w.radius, w.width);
    (1..=SHELL_SPHERES)
        .map(|k| {
            let th = std::f32::consts::TAU * k as f32 / (SHELL_SPHERES + 1) as f32;
            let p = center + (down * th.cos() + fwd * th.sin()) * ring;
            Collider {
                body,
                shape: ShapeType::Sphere as u32,
                material: w.rim_material,
                generation: 0,
                local_pos: p.to_array(),
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [rs, 0.0, 0.0, rs],
            }
        })
        .collect()
}

fn rim_colliders(w: &WheelDesc, frame: &WheelFrame, body: u32) -> Vec<Collider> {
    rim_colliders_frame(w, frame, body, rim_count(w))
}

fn rim_colliders_frame(w: &WheelDesc, frame: &WheelFrame, body: u32, n: u32) -> Vec<Collider> {
    let rr = (0.5 * w.width).min(0.5 * w.radius).max(1e-3);
    let ring = (w.radius - rr).max(0.0);
    (0..n)
        .map(|k| {
            let th = std::f32::consts::TAU * k as f32 / n as f32;
            let p = (frame.up * th.cos() + frame.fwd * th.sin()) * ring;
            Collider {
                body,
                shape: ShapeType::Sphere as u32,
                material: w.rim_material,
                generation: 0,
                local_pos: p.to_array(),
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [rr, 0.0, 0.0, rr],
            }
        })
        .collect()
}

/// Wheel static + initial dynamic record.
pub fn pack_wheel(
    desc: &VehicleDesc,
    w: &WheelDesc,
    vehicle: u32,
    local_index: u32,
    enabled: bool,
    parked_body: u32,
    parked_col_start: u32,
    curves: (u32, u32),
) -> Vec<Slot> {
    let up = v3(desc.up).normalize();
    let fwd = v3(desc.forward).normalize();
    let axle = up.cross(fwd).normalize();
    let dir = w.suspension_dir.map(v3).unwrap_or(-up).normalize();
    let steer_axis = w.steer_axis.map(v3).unwrap_or(up).normalize();
    let wfwd = w.forward.map(v3).unwrap_or(fwd).normalize();
    let wup = w.up.map(v3).unwrap_or(up).normalize();
    let pos = v3(w.position);
    let lat = pos.dot(axle);
    let side = if lat > 1e-3 { 1.0 } else if lat < -1e-3 { -1.0 } else { 0.0 };
    let mut flags = 0;
    if w.steerable {
        flags |= WF_STEER;
    }
    if w.driven {
        flags |= WF_DRIVEN;
    }
    if w.handbrake {
        flags |= WF_HANDBRAKE;
    }
    if enabled {
        flags |= WF_ENABLED;
    }
    if w.friction_model == FrictionModel::Box {
        flags |= WF_BOX_FRICTION;
    }
    let (mode, k, c) = match w.spring {
        Spring::FrequencyDamping { hz, zeta } => (0, hz, zeta),
        Spring::StiffnessDamping { k, c } => (1, k, c),
    };
    let (bump_hz, bump_zeta, bump_max) = match w.bump_stop {
        BumpStop::Spring { hz, zeta, max_force } => (hz, zeta, max_force),
        BumpStop::Rigid => {
            flags |= WF_RIGID_BUMP;
            (0.0, 0.0, 0.0)
        }
        BumpStop::None => (0.0, 0.0, 0.0),
    };
    let inertia = w.inertia.unwrap_or(0.5 * w.mass * w.radius * w.radius);
    let mut s = vec![[0u32; 4]; WHEEL_STRIDE as usize];
    s[W_CFG as usize] = [vehicle, flags, local_index, NO_INDEX];
    s[W_HP as usize] = slot3(pos, w.radius);
    s[W_DIR as usize] = slot3(dir, w.width);
    s[W_STEERAX as usize] = slot3(steer_axis, w.max_steer);
    s[W_FWD as usize] = slot3(wfwd, side);
    s[W_SUSP as usize] = [f(w.suspension_min), f(w.suspension_max), f(w.preload), mode];
    s[W_SPR as usize] = [f(k), f(c), f(bump_hz), f(bump_zeta)];
    s[W_INER as usize] = [f(inertia), f(w.angular_damping), f(w.max_brake_torque), f(w.max_handbrake_torque)];
    s[W_TIRE as usize] = [curves.0, curves.1, f(w.long_mu_scale), f(w.lat_mu_scale)];
    s[W_MISC as usize] = [f(w.mass), f(w.rolling_resistance), f(bump_max), f(w.roll_influence)];
    s[W_PARK as usize] = [parked_body, parked_col_start, rim_count(w), 0];
    s[W_UP as usize] = slot3(wup, w.drive_multiplier);
    // Dynamic state: no contact, suspension fully extended.
    s[W_C0 as usize] = [NO_INDEX, NO_INDEX, if enabled { 0 } else { WD_DISABLED }, 0];
    s[W_C1 as usize] = slot3(Vec3::ZERO, w.suspension_max);
    s[W_C2 as usize] = slot3(up, 1.0);
    s[W_O_ROT as usize] = [f(0.0), f(0.0), f(0.0), f(1.0)];
    s[W_O_POS as usize] = slot3(pos + dir * w.suspension_max, w.suspension_max);
    s[W_FP as usize] = match w.suspension_force_point {
        Some(p) => slot3(v3(p), 1.0),
        None => slot3(Vec3::ZERO, 0.0),
    };
    let (bf, bt) = if w.mount.breakable() { (mount_limit(w.mount.break_force), mount_limit(w.mount.break_torque)) } else { (1e30, 1e30) };
    s[W_MOUNT as usize] = [f(bf), f(bt), f(0.0), f(0.0)];
    s[W_MOUNT2 as usize] = slot3(w.mount.anchor.map(v3).unwrap_or(pos), 0.0);
    s
}

/// Pack a vehicle at chassis pose (`position`, `rotation`).
pub fn pack(desc: &VehicleDesc, position: Vec3, rotation: Quat, slots: VehicleSlots) -> PackedVehicle {
    let centered = desc.centered();
    let desc = &centered;
    assert!(!desc.wheels.is_empty(), "vehicle needs at least one wheel");
    assert!(
        desc.wheels.len() as u32 + desc.spare_wheels <= MAX_VEHICLE_WHEELS,
        "at most {MAX_VEHICLE_WHEELS} wheel slots per vehicle"
    );
    assert!(desc.parts.len() <= 32, "at most 32 parts per vehicle");
    let up = v3(desc.up).normalize();
    let fwd = v3(desc.forward).normalize();
    let axle = up.cross(fwd).normalize();
    let frame = WheelFrame { up, fwd };
    let mut out = PackedVehicle::default();
    let mut curves = CurveSink { base: slots.curve_point_base, points: &mut out.curve_points };

    // ---- mass properties (body origin = centre of mass) ----
    let mut mass = 0.0f32;
    let mut inertia = Vec3::ZERO;
    let mut part_props = Vec::new();
    for p in &desc.parts {
        let vol: f32 = p.colliders.iter().map(|c| c.shape.volume()).sum::<f32>().max(1e-9);
        let mut com = Vec3::ZERO;
        for c in &p.colliders {
            com += v3(c.position) * (c.shape.volume() / vol);
        }
        let mut ip = Vec3::ZERO;
        for c in &p.colliders {
            let m = p.mass * c.shape.volume() / vol;
            let d = v3(c.position) - com;
            ip += Vec3::from_array(c.shape.inertia(m))
                + m * Vec3::new(d.y * d.y + d.z * d.z, d.x * d.x + d.z * d.z, d.x * d.x + d.y * d.y);
        }
        let d = com;
        inertia += ip + p.mass * Vec3::new(d.y * d.y + d.z * d.z, d.x * d.x + d.z * d.z, d.x * d.x + d.y * d.y);
        mass += p.mass;
        part_props.push((com, ip));
    }
    for w in &desc.wheels {
        let d = v3(w.position) + w.suspension_dir.map(v3).unwrap_or(-up) * w.suspension_max;
        inertia += wheel_body_inertia(w, axle)
            + w.mass * Vec3::new(d.y * d.y + d.z * d.z, d.x * d.x + d.z * d.z, d.x * d.x + d.y * d.y);
        mass += w.mass;
    }
    let inertia = desc.inertia.map(v3).unwrap_or(inertia);
    out.mass = mass;
    out.inertia = inertia.to_array();

    // ---- bodies: chassis, anchor, wheel slots, parts ----
    let chassis = slots.body_base;
    let anchor = chassis + 1;
    let wheel_body0 = chassis + 2;
    let wheel_slots = desc.wheels.len() as u32 + desc.spare_wheels;
    let part_body0 = wheel_body0 + wheel_slots;
    let mut cb = BodyState::dynamic(position, rotation, mass, inertia);
    cb.flags |= BODY_VEHICLE;
    cb.initial_pos = cb.position;
    cb.initial_rot = cb.orientation;
    cb.initial_pad = f32::from_bits(slots.aabb_block);
    out.bodies.push(cb);
    let mut ab = BodyState::static_plane_support();
    ab.position = parked_position(anchor);
    ab.initial_pos = ab.position;
    out.bodies.push(ab);
    let spare = desc.wheels.first().cloned().unwrap_or_default();
    for i in 0..wheel_slots {
        let w = desc.wheels.get(i as usize).unwrap_or(&spare);
        out.bodies.push(parked_body(wheel_body0 + i, w.mass, wheel_body_inertia(w, axle)));
    }
    for (i, p) in desc.parts.iter().enumerate() {
        out.bodies.push(parked_body(part_body0 + i as u32, p.mass, part_props[i].1));
    }

    // ---- colliders: chassis parts, then rims ----
    let mut part_cols = Vec::new();
    for p in &desc.parts {
        let start = slots.collider_base + out.colliders.len() as u32;
        for c in &p.colliders {
            let (shape, params) = match c.shape {
                ShapeDesc::Box { half } => {
                    (ShapeType::Box, [half[0], half[1], half[2], v3(half).length()])
                }
                ShapeDesc::Sphere { radius } => (ShapeType::Sphere, [radius, 0.0, 0.0, radius]),
                ShapeDesc::Capsule { radius, half_length } => {
                    (ShapeType::Capsule, [radius, half_length, 0.0, radius + half_length])
                }
            };
            out.colliders.push(Collider {
                body: chassis,
                shape: shape as u32,
                material: c.material,
                generation: 0,
                local_pos: c.position,
                density: 0.0,
                local_rot: c.rotation,
                params,
            });
        }
        part_cols.push((start, p.colliders.len() as u32));
    }
    let mut rim_starts = Vec::new();
    for i in 0..wheel_slots {
        let w = desc.wheels.get(i as usize).unwrap_or(&spare);
        rim_starts.push(slots.collider_base + out.colliders.len() as u32);
        out.colliders.extend(rim_colliders(w, &frame, wheel_body0 + i));
    }
    // Wheel shells on the chassis (spare slots park theirs on the static anchor);
    // vehicle_post moves them with the suspension and steering every step.
    // A track pod replaces its wheels' shells: they stay parked (no W_TJ.y).
    let podded: Vec<u32> = match &desc.tracks {
        Some(t) => [&t.left, &t.right].iter().filter(|tr| tr.pod.is_some()).flat_map(|tr| tr.wheels.clone()).collect(),
        None => Vec::new(),
    };
    let mut shell_starts = Vec::new();
    for i in 0..wheel_slots {
        let real = (i as usize) < desc.wheels.len() && !podded.contains(&i);
        let w = desc.wheels.get(i as usize).unwrap_or(&spare);
        let down = w.suspension_dir.map(v3).unwrap_or(-up).normalize();
        let center = v3(w.position) + down * w.suspension_max;
        shell_starts.push(if podded.contains(&i) { NO_INDEX } else { slots.collider_base + out.colliders.len() as u32 });
        out.colliders.extend(shell_colliders(w, center, down, fwd, if real { chassis } else { anchor }));
    }
    // Track pods (vehicle_post runs their belts): V_QUERY.zw.
    let mut pod_slots = [NO_INDEX; 2];
    for (k, cols) in track_pods(desc, chassis).into_iter().enumerate() {
        if !cols.is_empty() {
            pod_slots[k] = (slots.collider_base + out.colliders.len() as u32) | ((cols.len() as u32) << POD_COUNT_SHIFT);
            out.colliders.extend(cols);
        }
    }

    // ---- wheels ----
    let mut wheel_curves = Vec::new();
    for w in &desc.wheels {
        wheel_curves.push((curves.push(&w.long_friction), curves.push(&w.lat_friction)));
    }
    for i in 0..wheel_slots {
        let real = (i as usize) < desc.wheels.len();
        let w = desc.wheels.get(i as usize).unwrap_or(&spare);
        let c = if real { wheel_curves[i as usize] } else { wheel_curves[0] };
        out.wheels.push(pack_wheel(desc, w, slots.vehicle, i, real, wheel_body0 + i, rim_starts[i as usize], c));
    }

    // ---- tyre joints: graph edges to dynamic ground, parked on the anchor ----
    for (i, rec) in out.wheels.iter_mut().enumerate() {
        rec[W_TJ as usize] = [slots.joint_base + i as u32, shell_starts[i], NO_INDEX, 0];
        // D6 record flagged JOINT_FLAG_TYRE; vehicle_link re-points it and
        // stores the wheel's ground block index in `motion`.
        out.joints.push(Joint {
            body_a: anchor,
            body_b: anchor,
            flags: JOINT_FLAG_TYRE,
            ..Joint::default()
        });
    }

    // ---- tracks: members share the driven wheel's spin, inertia and brake ----
    if let Some(t) = &desc.tracks {
        for tr in [&t.left, &t.right] {
            let leader = tr.driven_wheel as usize;
            for &m in &tr.wheels {
                out.wheels[m as usize][W_CFG as usize][3] = tr.driven_wheel;
            }
            let rec = &mut out.wheels[leader];
            rec[W_CFG as usize][1] |= WF_DRIVEN;
            rec[W_INER as usize] = [f(tr.inertia), f(tr.angular_damping), f(tr.max_brake_torque), f(0.0)];
            rec[W_UP as usize][3] = f(tr.differential_ratio);
        }
    }

    // ---- steering geometry ----
    let steered: Vec<&WheelDesc> = desc.wheels.iter().filter(|w| w.steerable).collect();
    let fixed: Vec<&WheelDesc> = desc.wheels.iter().filter(|w| !w.steerable).collect();
    let mean_fwd = |ws: &[&WheelDesc]| {
        if ws.is_empty() {
            0.0
        } else {
            ws.iter().map(|w| v3(w.position).dot(fwd)).sum::<f32>() / ws.len() as f32
        }
    };
    let wheelbase = desc
        .steering
        .wheelbase
        .unwrap_or_else(|| (mean_fwd(&steered) - mean_fwd(&fixed)).abs());
    let track = desc.steering.track.unwrap_or_else(|| {
        let lats: Vec<f32> = steered.iter().map(|w| v3(w.position).dot(axle)).collect();
        if lats.len() < 2 {
            0.0
        } else {
            lats.iter().cloned().fold(f32::MIN, f32::max) - lats.iter().cloned().fold(f32::MAX, f32::min)
        }
    });

    // ---- drive ----
    let mut flags = VF_ENABLED;
    if desc.can_sleep {
        flags |= VF_CAN_SLEEP;
    }
    if desc.reverse_brake {
        flags |= VF_REVERSE_BRAKE;
    }
    if desc.wheel_query == WheelQuery::Sphere {
        flags |= VF_SPHERE_CAST;
    }
    if desc.assists.traction_control {
        flags |= VF_TRACTION_CONTROL;
    }
    if desc.tracks.is_some() {
        flags |= VF_TRACKED;
    }
    if desc.breakable() {
        flags |= VF_BREAKABLE;
    }
    let mut v = vec![[0u32; 4]; VEH_STRIDE as usize];
    let (direct_torque, top_speed) = match &desc.drive {
        DriveDesc::Direct { max_torque, top_speed } => {
            flags |= VF_DIRECT_DRIVE;
            (*max_torque, *top_speed)
        }
        DriveDesc::Drivetrain(d) => {
            if d.transmission.mode == TransmissionMode::Manual {
                flags |= VF_MANUAL_TRANS;
            }
            let e = &d.engine;
            let t = &d.transmission;
            let torque = curves.push(&e.torque_curve);
            v[V_ENG0 as usize] = [f(e.max_torque), f(e.min_rpm), f(e.max_rpm), f(e.inertia)];
            v[V_ENG1 as usize] = [
                f(e.angular_damping),
                torque,
                f(t.clutch_strength),
                f(d.inter_diff_limited_slip.unwrap_or(1e30)),
            ];
            v[V_TR0 as usize] = [f(t.switch_time), f(t.clutch_release_time), f(t.switch_latency), f(t.shift_up_rpm)];
            assert!(t.gear_ratios.len() + t.reverse_ratios.len() <= 8, "at most 8 gear ratios");
            v[V_TR1 as usize] = [f(t.shift_down_rpm), t.gear_ratios.len() as u32, t.reverse_ratios.len() as u32, 0];
            let mut ratios = [0.0f32; 8];
            for (i, r) in t.gear_ratios.iter().chain(t.reverse_ratios.iter()).enumerate() {
                ratios[i] = *r;
            }
            v[V_GEARS0 as usize] = [f(ratios[0]), f(ratios[1]), f(ratios[2]), f(ratios[3])];
            v[V_GEARS1 as usize] = [f(ratios[4]), f(ratios[5]), f(ratios[6]), f(ratios[7])];
            for dd in &d.differentials {
                let lr = |x: Option<u32>| x.unwrap_or(NO_INDEX);
                out.diffs.push(vec![
                    [lr(dd.left), lr(dd.right), f(dd.ratio), f(dd.left_right_split)],
                    [f(dd.limited_slip_ratio.unwrap_or(1e30)), f(dd.engine_torque_ratio), 0, 0],
                ]);
            }
            (0.0, 0.0)
        }
    };
    let steer_curve = curves.push(&desc.steering.speed_curve);
    v[V_CFG as usize] = [chassis, slots.wheel_base, wheel_slots, flags];
    let pitch_roll_cos = if desc.max_pitch_roll_deg >= 179.9 { -2.0 } else { desc.max_pitch_roll_deg.to_radians().cos() };
    v[V_FWD as usize] = slot3(fwd, pitch_roll_cos);
    v[V_UP as usize] = slot3(up, desc.max_slope_deg.to_radians().cos());
    v[V_AERO0 as usize] = [f(desc.aero.drag), f(desc.aero.downforce), f(desc.gravity_scale), steer_curve];
    v[V_AERO1 as usize] = slot3(v3(desc.aero.center), direct_torque);
    v[V_STEER as usize] = [f(desc.steering.ackermann), f(wheelbase), f(track), f(desc.steering.rate)];
    v[V_RESP as usize] = [
        f(desc.response.throttle_rate),
        f(desc.response.brake_rate),
        f(desc.response.handbrake_rate),
        f(top_speed),
    ];
    v[V_POOL as usize] = [slots.diff_base, out.diffs.len() as u32, slots.anti_roll_base, desc.anti_roll.len() as u32];
    v[V_POOL2 as usize] = [slots.part_base, desc.parts.len() as u32, anchor, 0];
    v[V_MASS as usize] = [f(mass), f(inertia.x), f(inertia.y), f(inertia.z)];
    let a = desc.assists;
    v[V_ASSIST as usize] = [f(a.air_control_pitch), f(a.air_control_yaw), f(a.upright_stiffness), f(a.upright_damping)];
    if let Some(m) = desc.motorcycle {
        v[V_BIKE0 as usize] = [f(m.max_lean_deg.to_radians()), f(m.lean_spring), f(m.lean_damping), f(m.lean_integration)];
        v[V_BIKE1 as usize] = [f(m.lean_integration_decay), f(m.lean_smoothing), f(if m.lean_steering_limit { 1.0 } else { 0.0 }), 0];
    }
    v[V_BIKE_D as usize] = slot3(Vec3::Y, 0.0);
    v[V_TRACK as usize] = match &desc.tracks {
        Some(t) => [t.left.driven_wheel, t.right.driven_wheel, f(t.steer_strength), f(0.0)],
        None => [NO_INDEX, NO_INDEX, f(0.0), f(0.0)],
    };
    // V_TRACK_D.w: lateral-accel budget × track spacing (a·W, m²/s²) for the
    // speed-dependent ratio cap.
    let turn_aw = desc.tracks.as_ref().map_or(0.0, |t| {
        let lateral = |tr: &TrackDesc| {
            let ws: Vec<f32> = tr.wheels.iter().map(|&i| v3(desc.wheels[i as usize].position).dot(axle)).collect();
            ws.iter().sum::<f32>() / ws.len().max(1) as f32
        };
        t.turn_grip_g * 9.81 * (lateral(&t.left) - lateral(&t.right)).abs()
    });
    v[V_TRACK_D as usize] = [f(1.0), f(1.0), f(0.0), f(turn_aw)];
    v[V_QUERY as usize] = [desc.query_interval, 0, pod_slots[0], pod_slots[1]];
    v[V_IN0 as usize] = [f(0.0); 4];
    v[V_IN1 as usize] = [GEAR_AUTO, 0, 0, 0];
    v[V_IN2 as usize] = [!0, !0, !0, !0];
    v[V_IN3 as usize] = [0, 0, 1, 0];
    let min_rpm = match &desc.drive {
        DriveDesc::Drivetrain(d) => d.engine.min_rpm,
        _ => 0.0,
    };
    v[V_D0 as usize] = [f(min_rpm / 9.549_296_6), f(0.0), f(1.0), f(0.0)];
    v[V_REFQ as usize] = [f(0.0), f(0.0), f(0.0), f(1.0)];
    v[V_O3 as usize] = [1, 0, f(0.0), f(0.0)];
    out.vehicle = v;

    for a in &desc.anti_roll {
        out.anti_rolls.push([a.left, a.right, f(a.stiffness), 0]);
    }
    for (i, p) in desc.parts.iter().enumerate() {
        let (role, mask) = match p.role {
            PartRole::Body => (PR_BODY, 0),
            PartRole::Engine => (PR_ENGINE, 0),
            PartRole::DriveShaft { wheels } => (PR_DRIVESHAFT, wheels),
            PartRole::Steering { wheels } => (PR_STEERING, wheels),
            PartRole::Brake { wheels } => (PR_BRAKE, wheels),
        };
        let (com, ip) = part_props[i];
        // Part 0 is the root: it stays the car.
        let m = if i == 0 || !p.mount.breakable() { MountDesc::default() } else { p.mount };
        let (bf, bt) = if m.breakable() { (mount_limit(m.break_force), mount_limit(m.break_torque)) } else { (1e30, 1e30) };
        out.parts.push(vec![
            [part_cols[i].0, part_cols[i].1, role, mask],
            [f(p.mass), f(ip.x), f(ip.y), f(ip.z)],
            slot3u(com, PS_ATTACHED),
            [part_body0 + i as u32, slots.vehicle, 0, 0],
            slot3(m.anchor.map(v3).unwrap_or(com), bf),
            [f(bt), f(0.0), f(0.0), f(0.0)],
        ]);
    }
    out
}

/// Header slots for a vehicle pool layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VehicleCapacity {
    pub max_vehicles: u32,
    pub max_wheels: u32,
    pub max_curve_points: u32,
    pub max_diffs: u32,
    pub max_anti_rolls: u32,
    pub max_parts: u32,
    pub max_rays: u32,
}

impl Default for VehicleCapacity {
    fn default() -> Self {
        Self {
            max_vehicles: 64,
            max_wheels: 512,
            max_curve_points: 4096,
            max_diffs: 256,
            max_anti_rolls: 256,
            max_parts: 512,
            max_rays: 1024,
        }
    }
}

/// Region bases (float4 slots) inside gVehicleData.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VehicleLayout {
    pub vehicle_base: u32,
    pub wheel_base: u32,
    pub curve_base: u32,
    pub diff_base: u32,
    pub anti_roll_base: u32,
    pub part_base: u32,
    pub ray_base: u32,
    pub total_slots: u32,
}

impl VehicleCapacity {
    pub fn layout(&self) -> VehicleLayout {
        let vehicle_base = VH_SLOTS;
        let wheel_base = vehicle_base + self.max_vehicles * VEH_STRIDE;
        let curve_base = wheel_base + self.max_wheels * WHEEL_STRIDE;
        let diff_base = curve_base + self.max_curve_points.div_ceil(2);
        let anti_roll_base = diff_base + self.max_diffs * DIFF_STRIDE;
        let part_base = anti_roll_base + self.max_anti_rolls * ANTIROLL_STRIDE;
        let ray_base = part_base + self.max_parts * PART_STRIDE;
        let total_slots = ray_base + self.max_rays * RAY_STRIDE;
        VehicleLayout {
            vehicle_base,
            wheel_base,
            curve_base,
            diff_base,
            anti_roll_base,
            part_base,
            ray_base,
            total_slots,
        }
    }

    pub fn bytes(&self) -> u64 {
        u64::from(self.layout().total_slots) * 16
    }

    pub fn header(&self, aabb_tail: u32, vehicles: u32, wheels: u32, rays: u32, parts: u32) -> [Slot; 4] {
        let l = self.layout();
        [
            [self.max_vehicles, self.max_wheels, vehicles, wheels],
            [l.vehicle_base, l.wheel_base, l.curve_base, aabb_tail],
            [l.diff_base, l.anti_roll_base, l.part_base, l.ray_base],
            [rays, self.max_rays, parts, 0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corners(c: &Collider) -> Vec<Vec3> {
        let q = Quat::from_array(c.local_rot);
        let h = Vec3::from_slice(&c.params[..3]);
        (0..8)
            .map(|i| {
                let s = Vec3::new(if i & 1 == 0 { -1.0 } else { 1.0 }, if i & 2 == 0 { -1.0 } else { 1.0 }, if i & 4 == 0 { -1.0 } else { 1.0 });
                v3(c.local_pos) + q * (h * s)
            })
            .collect()
    }

    #[test]
    fn tank_pods_close_the_track_and_ramp_up_to_the_end_wheels() {
        let d = crate::presets::tank();
        let t = d.tracks.as_ref().unwrap();
        let cols = track_pod_colliders(&d, &t.left, t.left.pod.as_ref().unwrap(), 0);
        assert_eq!(cols.len(), 4, "upper run, lower run, two ramps");
        let ys: Vec<f32> = cols.iter().flat_map(corners).map(|p| p.y).collect();
        let zs: Vec<f32> = cols.iter().flat_map(corners).map(|p| p.z).collect();
        let lowest = ys.iter().cloned().fold(f32::MAX, f32::min);
        // Road wheels bottom out with their tread at -0.9; the pod floats 5 cm above.
        assert!((lowest - -0.85).abs() < 1e-4, "lowest {lowest}");
        // Spans the idler and sprocket rims.
        assert!(zs.iter().cloned().fold(f32::MIN, f32::max) > 3.24 && zs.iter().cloned().fold(f32::MAX, f32::min) < -3.04);
        // Ramps pitch the nose up at the front and the tail up at the rear.
        let pitch = |c: &Collider| (Quat::from_array(c.local_rot) * Vec3::Z).y;
        assert!(pitch(&cols[2]) > 0.3 && pitch(&cols[3]) < -0.3, "{} {}", pitch(&cols[2]), pitch(&cols[3]));
        // Local +X is the axle (the belt runs around it).
        for c in &cols {
            assert!((Quat::from_array(c.local_rot) * Vec3::X - Vec3::X).length() < 1e-5);
        }
        for c in &cols {
            eprintln!("pod box at {:?} half {:?} pitch {:.1}°", c.local_pos, &c.params[..3], pitch(c).asin().to_degrees());
        }
    }

    #[test]
    fn pod_counts_match_requirements() {
        let d = crate::presets::tank();
        let req = requirements(&d);
        let p = pack(&d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default());
        assert_eq!(req.colliders as usize, p.colliders.len());
        let q = p.vehicle[V_QUERY as usize];
        assert_eq!(q[2] >> POD_COUNT_SHIFT, 4);
        assert_eq!(q[3] >> POD_COUNT_SHIFT, 4);
    }
}
