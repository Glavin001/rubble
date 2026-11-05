//! Raycast wheels for explicit soft-body vehicles (CPU reference).
//!
//! The explicit node–beam vehicles crash on beam tyres. To drive, a vehicle can
//! instead carry raycast wheels: the soft solver's stage-1 raycast vehicle
//! ([`crate::pack::raycast_core`], the rigid World's `engine_vehicle` model: its
//! suspension spring and damper, bump stop, anti-roll bars, Jolt friction curves
//! bounded by a friction ellipse, brakes, handbrake, rolling resistance, steering
//! with Ackermann and a speed curve, input response rates, reverse braking and
//! traction control), evaluated as explicit forces inside the 0.5 ms step on the
//! wheels' axle (hub) nodes. The forces then flow through the beams: the body
//! bends, crumples and stays deformed as in a crash, and a crushed suspension
//! moves the hub, so the ray's origin and frame move with it.
//!
//! Per step, before the node pass ([`drive_pass`]; the kernel's `driveStep`,
//! `slang/kernels/explicit_beams_drive.slang`, one lane per vehicle, the same
//! arithmetic in the same order):
//!
//! 1. the vehicle frame (forward, up) from its reference nodes and the hubs'
//!    mass-weighted velocity give the forward speed; the driver's input is
//!    smoothed at the response rates (reverse braking first);
//! 2. per wheel a ray from the hardpoint (`rest_len` above the hub along the
//!    vehicle's up) down the suspension, against the flat ground, the boxes and
//!    the static shapes, nearest hit within the travel plus the radius; the spring,
//!    damper and bump stop give the normal load, then the anti-roll bars;
//! 3. per wheel the tyre forces with the hub's force as the node pass will see it
//!    (beams, contact sums): the lateral force that stops the hub's sideways slip
//!    this step, the longitudinal force and wheel spin solved exactly with the
//!    drive and brake torques (a monotone piecewise-linear equation in the new
//!    spin, the rigid model's spin solve in explicit form), both bounded by
//!    μ(slip)·F_n with the rigid model's ellipse and slip-direction sharing;
//! 4. the wheel force (suspension along the contact normal, tyre forces along
//!    the contact plane) goes to the hub nodes in proportion to their mass, as
//!    contact sums in the GPU's fixed point (1/32 N), so the CPU and GPU agree
//!    bit for bit.
//!
//! Simplifications (stated): the drive is the raycast model's direct drive
//! (`DriveDesc::Direct`: full torque up to a third of the top speed, then a
//! linear fade), sized from the vehicle's engine (first-gear torque at the
//! wheels, top-gear top speed), in place of the engine, clutch, gearbox and
//! differentials; the wheel force acts at the hub, not at the contact patch
//! (less roll moment, as a roll influence below 1); rays meet static geometry,
//! not the World's moving rigid bodies.

use crate::explicit_beams::{to_fixed, ExplicitBeams, StaticShape, CONTACT_FIXED, STEP};
use crate::pack::{raycast_core, SpawnOptions};
use crate::SoftVehicleDesc;
use glam::Vec3;

/// Wheels a vehicle may have (more fails to spawn).
pub const MAX_WHEELS: usize = 16;
/// Points per curve (more fails to spawn).
pub const CURVE_POINTS: usize = 8;
/// Anti-roll bars a vehicle may have.
pub const MAX_ANTI_ROLL: usize = 8;

/// One raycast wheel.
#[derive(Clone, Debug, PartialEq)]
pub struct DriveWheel {
    /// Axle (hub) nodes, vehicle-local; equal for a one-node hub.
    pub node1: u32,
    pub node2: u32,
    pub radius: f32,
    pub width: f32,
    /// Spin inertia (kg m²) and angular damping (1/s).
    pub inertia: f32,
    pub angular_damping: f32,
    /// Suspension: bump-stop length, natural (longest) length, the hardpoint's
    /// height above the hub, spring (N/m) and damper (N s/m).
    pub susp_min: f32,
    pub susp_max: f32,
    pub rest_len: f32,
    pub k: f32,
    pub c: f32,
    /// Bump stop spring and damper (0: none).
    pub bump_k: f32,
    pub bump_c: f32,
    /// Steering lock (rad), 0 for a fixed wheel.
    pub max_steer: f32,
    pub driven: bool,
    pub handbrake: bool,
    pub drive_mult: f32,
    /// Brake and handbrake torques (N m) at full input; rolling resistance (·F_n·R).
    pub brake_max: f32,
    pub handbrake_max: f32,
    pub rolling: f32,
    /// μ against slip ratio and against slip angle (degrees), their scales and
    /// peaks (slip ratio, slip angle in degrees, the tangent of the peak angle).
    pub long_curve: Vec<[f32; 2]>,
    pub lat_curve: Vec<[f32; 2]>,
    pub long_scale: f32,
    pub lat_scale: f32,
    pub peak_kappa: f32,
    pub peak_alpha: f32,
    pub tan_peak_alpha: f32,
    /// +1 left of the centre line, −1 right (Ackermann's inner wheel).
    pub side: f32,
    /// The axle's direction (node1 → node2, turned to the left) at spawn in the
    /// vehicle frame (forward, up, left components): the build's own toe and camber,
    /// taken out each step so only deformation since spawn turns the wheel.
    pub axle_rest: [f32; 3],
}

/// A vehicle's raycast wheels and drive.
#[derive(Clone, Debug, PartialEq)]
pub struct DriveConfig {
    pub wheels: Vec<DriveWheel>,
    /// Frame nodes (centre, back, left): forward = centre − back, left ≈ left − centre.
    pub refs: [u32; 3],
    /// Direct drive: torque per driven wheel at full throttle (N m) and top speed (m/s).
    pub max_torque: f32,
    pub top_speed: f32,
    /// Steering: input slew (1/s), Ackermann share, wheelbase and track (m), and
    /// the steer scale against speed (m/s).
    pub steer_rate: f32,
    pub ackermann: f32,
    pub wheelbase: f32,
    pub track: f32,
    pub speed_curve: Vec<[f32; 2]>,
    /// Input response rates (1/s).
    pub throttle_rate: f32,
    pub brake_rate: f32,
    pub handbrake_rate: f32,
    pub reverse_brake: bool,
    pub traction_control: bool,
    /// Traction control's low-pass factor per step, 1 − exp(−h/0.1 s).
    pub tc_alpha: f32,
    /// Contacts steeper than this from the suspension axis are ignored (cosine).
    pub max_slope_cos: f32,
    /// Surface friction the tyre μ is combined with (√(μ_tyre·μ_surface)).
    pub surface: f32,
    /// Anti-roll bars: (left wheel, right wheel, N/m).
    pub anti_roll: Vec<(u32, u32, f32)>,
    /// Wheel planes from each wheel's axle nodes (less their spawn toe and camber):
    /// a bent hub turns its wheel. Off (the default): every wheel's axle is the
    /// vehicle frame's left (its body, from the reference nodes), as the raycast
    /// model's chassis; the hubs' positions still carry the rays. Soft hubs on beam
    /// suspensions twist under drive and static loads (the showcase sedan's front
    /// hubs turned 25° under throttle), which the tyres' rings and rigidity beams
    /// held on beam tyres.
    pub hub_axles: bool,
}

/// The driver's input (per vehicle, set per frame by the host).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DriveInput {
    /// −1..1, negative reverses.
    pub forward: f32,
    /// −1..1, positive right.
    pub steer: f32,
    pub brake: f32,
    pub handbrake: f32,
}

/// One wheel's state and its last step's outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelState {
    /// Spin (rad/s), spin angle (rad) and steer angle (rad, + left).
    pub omega: f32,
    pub spin: f32,
    pub steer: f32,
    /// Suspension length (m), normal load (N), slip ratio, slip angle (degrees).
    pub length: f32,
    pub load: f32,
    pub kappa: f32,
    pub alpha: f32,
    pub contact: bool,
    /// The wheel's centre (world), its steered axle (unit, left) and its up.
    pub centre: Vec3,
    pub axle: Vec3,
    pub up: Vec3,
}

/// A vehicle's raycast wheels: configuration, input and state.
#[derive(Clone, Debug, PartialEq)]
pub struct Drive {
    pub config: DriveConfig,
    pub input: DriveInput,
    /// Smoothed (forward, brake, handbrake, steer).
    pub smoothed: [f32; 4],
    /// Traction control's torque cut.
    pub tc_cut: f32,
    pub wheels: Vec<WheelState>,
    /// The vehicle's forward speed at the last step (m/s).
    pub speed: f32,
}

fn sgn(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// `v / |v|` (zero stays zero), per-component quotients (the kernel's `sdiv3`).
fn nrm(v: Vec3) -> Vec3 {
    let l = sq(dot(v, v));
    if l > 0.0 {
        div3(v, l)
    } else {
        v
    }
}

/// a/b as the kernel's `sdiv`: the reciprocal, the quotient, one exact fma
/// correction (bit for bit with the GPU, which computes it the same way).
fn dv(a: f32, b: f32) -> f32 {
    let r = 1.0 / b;
    let q = a * r;
    (-q).mul_add(b, a).mul_add(r, q)
}

/// √x as the kernel's `ssqrt`: the root and one exact fma correction.
fn sq(x: f32) -> f32 {
    if x <= 0.0 {
        return x.sqrt();
    }
    let s = x.sqrt();
    (-s).mul_add(s, x).mul_add(0.5 / s, s)
}

/// atan2(y, x) as the kernel's `sAtan2` (`explicit_beams::exact_atan2`'s sequence
/// with the kernel's quotients, [`dv`]).
pub fn exact_atan2(y: f32, x: f32) -> f32 {
    const C: [u32; 12] = [
        0x3f800000, 0xbeaaaaab, 0x3e4ccccd, 0xbe124925, 0x3de38e39, 0xbdba2e8c, 0x3d9d89d9,
        0xbd888889, 0x3d70f0f1, 0xbd579436, 0x3d430c31, 0xbd321643,
    ];
    let (ax, ay) = (x.abs(), y.abs());
    let swap = ay > ax;
    let (lo, hi) = if swap { (ax, ay) } else { (ay, ax) };
    if hi == 0.0 {
        return 0.0;
    }
    let a = dv(lo, hi);
    let (t, base) = if a > f32::from_bits(0x3ed4_13cd) {
        (dv(a - 1.0, a + 1.0), f32::from_bits(0x3f490fdb))
    } else {
        (a, 0.0)
    };
    let t2 = t * t;
    let mut p = f32::from_bits(C[11]);
    for k in (0..11).rev() {
        p = p * t2 + f32::from_bits(C[k]);
    }
    let mut r = base + t * p;
    if swap {
        r = f32::from_bits(0x3fc90fdb) - r;
    }
    if x < 0.0 {
        r = f32::from_bits(0x40490fdb) - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

fn div3(v: Vec3, k: f32) -> Vec3 {
    Vec3::new(dv(v.x, k), dv(v.y, k), dv(v.z, k))
}

fn move_toward(cur: f32, target: f32, max_delta: f32) -> f32 {
    if max_delta <= 0.0 {
        return target;
    }
    let d = target - cur;
    if d.abs() <= max_delta {
        return target;
    }
    cur + sgn(d) * max_delta
}

/// A piecewise-linear curve with clamped ends (`engine_vehicle::Curve::eval`);
/// 1 when empty.
pub fn curve_eval(p: &[[f32; 2]], x: f32) -> f32 {
    if p.is_empty() {
        return 1.0;
    }
    if x <= p[0][0] {
        return p[0][1];
    }
    for i in 1..p.len() {
        if x <= p[i][0] {
            let t = dv(x - p[i - 1][0], (p[i][0] - p[i - 1][0]).max(1e-9));
            return p[i - 1][1] + (p[i][1] - p[i - 1][1]) * t;
        }
    }
    p[p.len() - 1][1]
}

/// (sin x, cos x) by their Taylor series to x¹¹ and x¹² (|x| ≤ π/2: error below
/// 6·10⁻⁸), Horner in x², IEEE single in this order (the kernel's `dSinCos`).
pub fn sin_cos(x: f32) -> (f32, f32) {
    let x2 = x * x;
    let mut s = f32::from_bits(0xb2d7322b);
    s = s * x2 + f32::from_bits(0x3638ef1d);
    s = s * x2 + f32::from_bits(0xb9500d01);
    s = s * x2 + f32::from_bits(0x3c088889);
    s = s * x2 + f32::from_bits(0xbe2aaaab);
    s = s * x2 + 1.0;
    let mut c = f32::from_bits(0x310f76c7);
    c = c * x2 + f32::from_bits(0xb493f27e);
    c = c * x2 + f32::from_bits(0x37d00d01);
    c = c * x2 + f32::from_bits(0xbab60b61);
    c = c * x2 + f32::from_bits(0x3d2aaaab);
    c = c * x2 + f32::from_bits(0xbf000000);
    c = c * x2 + 1.0;
    (x * s, c)
}

const DEGREES: u32 = 0x42652ee1; // 180/π
const TAN_145: u32 = 0x4103cf3a; // tan 1.45
const PI: u32 = 0x40490fdb;
const TAU: u32 = 0x40c90fdb;
/// Below this a direction is parallel to a slab (no division).
const PARALLEL: f32 = 1e-12;

/// A ray hit: distance and outward unit normal.
#[derive(Clone, Copy, Debug)]
struct Hit {
    t: f32,
    n: Vec3,
}

/// Ray `p + t·d` (t ≥ 0) against the slab box `lo..hi` in a frame whose axes are
/// `ax` (world): the entering face. `pl`, `dl` the ray in that frame.
fn slab(pl: Vec3, dl: Vec3, lo: Vec3, hi: Vec3, ax: [Vec3; 3]) -> Option<Hit> {
    let mut t_in = f32::MIN;
    let mut t_out = f32::MAX;
    let mut n = Vec3::ZERO;
    for a in 0..3 {
        let (p, d, l, h) = (pl[a], dl[a], lo[a], hi[a]);
        if d.abs() < PARALLEL {
            if p < l || p > h {
                return None;
            }
            continue;
        }
        let t1 = dv(l - p, d);
        let t2 = dv(h - p, d);
        let (near, far, s) = if t1 < t2 {
            (t1, t2, -1.0)
        } else {
            (t2, t1, 1.0)
        };
        if near > t_in {
            t_in = near;
            n = ax[a] * s;
        }
        if far < t_out {
            t_out = far;
        }
    }
    if t_in >= 0.0 && t_in <= t_out {
        Some(Hit { t: t_in, n })
    } else {
        None
    }
}

/// Ray against a sphere from outside: nearest t ≥ 0.
fn ray_sphere(p: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<Hit> {
    let oc = p - c;
    let b = dot(oc, d);
    let cc = dot(oc, oc) - r * r;
    if cc <= 0.0 {
        return None;
    }
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    let t = -b - sq(disc);
    if t < 0.0 {
        return None;
    }
    Some(Hit {
        t,
        n: nrm(p + d * t - c),
    })
}

/// Ray against the side of the infinite cylinder of axis `a` (unit) through `c`,
/// radius `r`, from outside: nearest t ≥ 0 and its axial coordinate.
fn ray_tube(p: Vec3, d: Vec3, c: Vec3, a: Vec3, r: f32) -> Option<(f32, f32, Vec3)> {
    let oc = p - c;
    let da = dot(d, a);
    let oa = dot(oc, a);
    let dp = d - a * da;
    let op = oc - a * oa;
    let qa = dot(dp, dp);
    if qa < PARALLEL {
        return None;
    }
    let qb = dot(op, dp);
    let qc = dot(op, op) - r * r;
    if qc <= 0.0 {
        return None;
    }
    let disc = qb * qb - qa * qc;
    if disc < 0.0 {
        return None;
    }
    let t = dv(-qb - sq(disc), qa);
    if t < 0.0 {
        return None;
    }
    let y = oa + da * t;
    Some((t, y, nrm(op + dp * t)))
}

fn closer(best: &mut Option<Hit>, h: Option<Hit>) {
    if let Some(h) = h {
        if best.is_none_or(|b| h.t < b.t) {
            *best = Some(h);
        }
    }
}

/// Ray against one static shape (the kernel's `driveShape`).
fn ray_shape(p: Vec3, d: Vec3, s: &StaticShape) -> Option<Hit> {
    match *s {
        StaticShape::OrientedBox {
            centre, half, axes, ..
        } => {
            let o = p - centre;
            let pl = Vec3::new(dot(axes[0], o), dot(axes[1], o), dot(axes[2], o));
            let dl = Vec3::new(dot(axes[0], d), dot(axes[1], d), dot(axes[2], d));
            slab(pl, dl, -half, half, axes)
        }
        StaticShape::Cylinder {
            centre,
            radius,
            half_height,
        } => {
            let mut best = None;
            // Caps: the plane facing the ray.
            if d.y.abs() >= PARALLEL {
                let (y, ny) = if d.y < 0.0 {
                    (centre.y + half_height, 1.0)
                } else {
                    (centre.y - half_height, -1.0)
                };
                let t = dv(y - p.y, d.y);
                let q = p + d * t - centre;
                if t >= 0.0 && q.x * q.x + q.z * q.z <= radius * radius {
                    best = Some(Hit {
                        t,
                        n: Vec3::new(0.0, ny, 0.0),
                    });
                }
            }
            if let Some((t, y, n)) = ray_tube(p, d, centre, Vec3::Y, radius) {
                if y >= -half_height && y <= half_height {
                    closer(&mut best, Some(Hit { t, n }));
                }
            }
            best
        }
        StaticShape::Capsule { a, b, radius } => {
            let mut best = ray_sphere(p, d, a, radius);
            closer(&mut best, ray_sphere(p, d, b, radius));
            let ab = b - a;
            let l = sq(dot(ab, ab));
            if l > 0.0 {
                let ax = div3(ab, l);
                if let Some((t, y, n)) = ray_tube(p, d, a, ax, radius) {
                    if y >= 0.0 && y <= l {
                        closer(&mut best, Some(Hit { t, n }));
                    }
                }
            }
            best
        }
        StaticShape::Sphere { centre, radius } => ray_sphere(p, d, centre, radius),
    }
}

/// The nearest hit of the ray `p + t·d`, t ≤ `t_max`: ground, boxes, shapes in order.
fn ray_cast(s: &ExplicitBeams, p: Vec3, d: Vec3, t_max: f32) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    if d.y < -PARALLEL && p.y >= s.ground {
        let t = dv(s.ground - p.y, d.y);
        best = Some(Hit { t, n: Vec3::Y });
    }
    for b in &s.boxes {
        closer(
            &mut best,
            slab(p, d, b.lo, b.hi, [Vec3::X, Vec3::Y, Vec3::Z]),
        );
    }
    for sh in &s.shapes {
        // An identity-rotation box is a box slot on the GPU (its world bounds).
        let h = match sh.as_aabb() {
            Some(b) => slab(p, d, b.lo, b.hi, [Vec3::X, Vec3::Y, Vec3::Z]),
            None => ray_shape(p, d, sh),
        };
        closer(&mut best, h);
    }
    best.filter(|h| h.t <= t_max)
}

/// The spin solve: with a hub of mass `m`, forward velocity `u` and forward force
/// `ft` (the node pass's, before the tyre), a wheel of inertia `j` and radius `r`
/// whose spin is `wd` after its drive torque and damping, a brake torque of at
/// most `b` and a tyre force of at most `a`: the new spin w and tyre force F with
/// J(w − wd)/h + R·F + B·sgn(w) = 0 (brake set-valued at w = 0) and F the force
/// that rolls the hub at wR this step, clamped to ±a. Exact on each linear piece.
pub fn spin_solve(
    m: f32,
    u: f32,
    ft: f32,
    j: f32,
    r: f32,
    wd: f32,
    b: f32,
    a: f32,
    h: f32,
) -> (f32, f32) {
    let stick = |w: f32| dv(m * (w * r - u), h) - ft;
    let f0 = clampf(stick(0.0), -a, a);
    let g0 = dv(j * (0.0 - wd), h) + r * f0;
    if g0 - b <= 0.0 && g0 + b >= 0.0 {
        return (0.0, f0);
    }
    let s = if g0 + b < 0.0 { 1.0 } else { -1.0 };
    let num = j * wd + r * m * u + h * r * ft - h * s * b;
    let den = j + m * r * r;
    let wu = dv(num, den);
    let fu = stick(wu);
    let (mut w, f) = if fu.abs() <= a {
        (wu, fu)
    } else {
        let f = sgn(fu) * a;
        (wd - dv(h * (r * f + s * b), j), f)
    };
    if s * w < 0.0 {
        w = 0.0;
    }
    (w, f)
}

/// The hub of wheel `w`: (mass, world position, velocity) mass-weighted over its nodes.
fn hub(s: &ExplicitBeams, w: &DriveWheel) -> (f32, Vec3, Vec3) {
    let n1 = &s.nodes[w.node1 as usize];
    if w.node1 == w.node2 {
        return (n1.mass, s.origin + n1.pos, n1.vel);
    }
    let n2 = &s.nodes[w.node2 as usize];
    let m = n1.mass + n2.mass;
    let x = (s.origin + n1.pos) * n1.mass + (s.origin + n2.pos) * n2.mass;
    let v = n1.vel * n1.mass + n2.vel * n2.mass;
    (m, div3(x, m), div3(v, m))
}

/// Adds wheel force `f` (newtons) to node `i` as a contact sum in the GPU's fixed
/// point: with fixed contact into the sums, else onto the force (the same value).
fn apply(s: &mut ExplicitBeams, i: u32, f: Vec3) {
    if s.fixed_contact {
        s.add_contact(i as usize, f);
        return;
    }
    let q = Vec3::new(
        to_fixed(f.x) as f32,
        to_fixed(f.y) as f32,
        to_fixed(f.z) as f32,
    ) * (1.0 / CONTACT_FIXED);
    s.nodes[i as usize].force += q;
}

/// One step's raycast wheels of `s` (before its node pass; see the module docs).
pub fn drive_pass(s: &mut ExplicitBeams) {
    let Some(mut d) = s.drive.take() else {
        return;
    };
    let h = STEP;
    let nw = d.config.wheels.len();
    let o = s.origin;
    let c = &d.config;
    // 1. Frame, speed, input.
    let x = |i: u32| o + s.nodes[i as usize].pos;
    let [rc, rb, rl] = c.refs;
    let fw = nrm(x(rc) - x(rb));
    let up = nrm(fw.cross(x(rl) - x(rc)));
    let left = up.cross(fw);
    let mut hubs = [(0.0f32, Vec3::ZERO, Vec3::ZERO); MAX_WHEELS];
    let mut msum = 0.0f32;
    let mut psum = Vec3::ZERO;
    for (k, w) in c.wheels.iter().enumerate() {
        hubs[k] = hub(s, w);
        msum += hubs[k].0;
        psum += hubs[k].2 * hubs[k].0;
    }
    let vm = div3(psum, msum);
    let speed = dot(vm, fw);
    let mut fin = clampf(d.input.forward, -1.0, 1.0);
    let sin = clampf(d.input.steer, -1.0, 1.0);
    let mut bin = clampf(d.input.brake, 0.0, 1.0);
    let hin = clampf(d.input.handbrake, 0.0, 1.0);
    if c.reverse_brake && fin * speed < 0.0 && speed.abs() > 1.0 {
        bin = bin.max(fin.abs());
        fin = 0.0;
    }
    let forward = move_toward(d.smoothed[0], fin, c.throttle_rate * h);
    let brake = move_toward(d.smoothed[1], bin, c.brake_rate * h);
    let hand = move_toward(d.smoothed[2], hin, c.handbrake_rate * h);
    let target = sin * curve_eval(&c.speed_curve, speed.abs());
    let steer = move_toward(d.smoothed[3], target, c.steer_rate * h);
    // Direct drive torque, traction control on the last step's slip.
    let sp = speed.abs();
    let mut scale = 1.0f32;
    if c.top_speed > 0.0 && forward * speed > 0.0 {
        scale = if sp < dv(c.top_speed, 3.0) {
            1.0
        } else {
            clampf(dv(c.top_speed - sp, dv(c.top_speed * 2.0, 3.0)), 0.0, 1.0)
        };
    }
    let mut cut = d.tc_cut;
    if c.traction_control {
        let v_reg = dv(sp, sp.max(3.0));
        let mut k_max = 0.0f32;
        for (k, w) in c.wheels.iter().enumerate() {
            if w.driven && d.wheels[k].contact {
                k_max = k_max.max(dv(d.wheels[k].kappa * v_reg, w.peak_kappa.max(0.03)));
            }
        }
        let cut_target = 1.0 - clampf(1.0 - dv(k_max - 0.8, 1.0), 0.2, 1.0);
        cut = lerp(cut, cut_target, c.tc_alpha);
    }
    // 2. Rays and suspension.
    let mut axle = [Vec3::ZERO; MAX_WHEELS];
    let mut hits: [Option<Hit>; MAX_WHEELS] = [None; MAX_WHEELS];
    let mut len = [0.0f32; MAX_WHEELS];
    let mut load = [0.0f32; MAX_WHEELS];
    for (k, w) in c.wheels.iter().enumerate() {
        let (_, xh, vh) = hubs[k];
        // The axle: the vehicle frame's left, or (`hub_axles`) the axle nodes' own line.
        let mut a = if w.node1 == w.node2 || !d.config.hub_axles {
            left
        } else {
            nrm(x(w.node2) - x(w.node1))
        };
        if dot(a, left) < 0.0 {
            a = -a;
        }
        // Less the spawn's toe and camber (in the vehicle frame).
        let [r0, r1, r2] = if d.config.hub_axles {
            w.axle_rest
        } else {
            [0.0, 0.0, 1.0]
        };
        let a =
            nrm(fw * (dot(a, fw) - r0) + up * (dot(a, up) - r1) + left * (dot(a, left) - r2 + 1.0));
        axle[k] = a;
        let hp = xh + up * w.rest_len;
        let hit = ray_cast(s, hp, -up, w.susp_max + w.radius);
        len[k] = w.susp_max;
        if let Some(ht) = hit {
            let cos_t = dot(ht.n, up);
            if cos_t >= c.max_slope_cos {
                let l = ht.t - w.radius;
                let rate = dv(dot(vh, ht.n), cos_t);
                let fs = w.k * (l - w.susp_max) + w.c * rate;
                let mut f = if fs < 0.0 { -fs } else { 0.0 };
                if l < w.susp_min && w.bump_k > 0.0 {
                    let fb = w.bump_k * (l - w.susp_min) + w.bump_c * rate;
                    if fb < 0.0 {
                        f -= fb;
                    }
                }
                len[k] = l;
                load[k] = f;
                hits[k] = Some(ht);
            }
        }
    }
    for &(l, r, kar) in &c.anti_roll {
        let (l, r) = (l as usize, r as usize);
        if hits[l].is_some() && hits[r].is_some() {
            let diff = len[l] - len[r];
            load[l] = (load[l] - kar * diff).max(0.0);
            load[r] = (load[r] + kar * diff).max(0.0);
        }
    }
    // 3. Tyres, spin, forces.
    for k in 0..nw {
        let w = &d.config.wheels[k];
        let (m, xh, vh) = hubs[k];
        let a = axle[k];
        let mut angle = 0.0f32;
        if w.max_steer > 0.0 {
            angle = -steer * w.max_steer;
            let wb = d.config.wheelbase;
            if d.config.ackermann > 0.0 && w.side != 0.0 && angle.abs() > 1e-4 && wb > 0.0 {
                let (sa, ca) = sin_cos(angle.abs());
                let turn_r = dv(wb, dv(sa, ca));
                let inner = (angle > 0.0) == (w.side > 0.0);
                let off = if inner { -0.5 } else { 0.5 };
                let ack = exact_atan2(wb, (turn_r + off * d.config.track).max(1e-3));
                angle = sgn(angle) * lerp(angle.abs(), ack, d.config.ackermann);
            }
        }
        let uw = nrm(up - a * dot(a, up));
        let f0 = a.cross(uw);
        let (sa, ca) = sin_cos(angle);
        let fwd = f0 * ca + uw.cross(f0) * sa;
        let j = w.inertia;
        let r = w.radius;
        let mut tau = if w.driven {
            forward * d.config.max_torque * scale * w.drive_mult * (1.0 - cut)
        } else {
            0.0
        };
        // The rev limit: no drive past the top speed at the wheel's rim (the
        // drivetrain's engine-speed clamp; a wheel spinning in the air stops there).
        if d.config.top_speed > 0.0
            && sgn(forward) * d.wheels[k].omega * w.radius >= d.config.top_speed
        {
            tau = 0.0;
        }
        let st = d.wheels[k];
        let wd = st.omega * (1.0 - w.angular_damping * h).max(0.0) + dv(h * tau, j);
        let mut b = brake * w.brake_max;
        if w.handbrake {
            b += hand * w.handbrake_max;
        }
        let mut out = st;
        let mut force = Vec3::ZERO;
        let (w_new, kappa, alpha) = if let Some(ht) = hits[k] {
            let n = ht.n;
            let fn_ = load[k];
            b += w.rolling * fn_ * r;
            let t = nrm(fwd - n * dot(n, fwd));
            let sd = n.cross(t);
            let fpre = if w.node1 == w.node2 {
                s.force_now(w.node1 as usize)
            } else {
                s.force_now(w.node1 as usize) + s.force_now(w.node2 as usize)
            };
            let vt = dot(vh, t);
            let vs = dot(vh, sd);
            let ft = dot(fpre, t);
            let fs = dot(fpre, sd);
            let kappa = dv((st.omega * r - vt).abs(), vt.abs().max(1e-3));
            let vlen = sq(vt * vt + vs * vs);
            let alpha = if vlen < 1e-3 {
                0.0
            } else {
                exact_atan2(vs.abs(), vt.abs()) * f32::from_bits(DEGREES)
            };
            let mu_l = (curve_eval(&w.long_curve, kappa.max(w.peak_kappa)) * w.long_scale).max(0.0);
            let mu_l = sq(mu_l * d.config.surface);
            let mu_t = (curve_eval(&w.lat_curve, alpha.max(w.peak_alpha)) * w.lat_scale).max(0.0);
            let mu_t = sq(mu_t * d.config.surface);
            let al = mu_l * fn_;
            let at = mu_t * fn_;
            let ft_raw = -(fs + dv(m * vs, h));
            let (_, fl_raw) = spin_solve(m, vt, ft, j, r, wd, b, f32::MAX, h);
            let (mut al_eff, mut f_t) = (al, ft_raw);
            if at <= 1e-9 {
                f_t = 0.0;
            } else {
                let e = if al > 1e-9 {
                    dv(fl_raw, al) * dv(fl_raw, al) + dv(ft_raw, at) * dv(ft_raw, at)
                } else {
                    2.0
                };
                if e > 1.0 {
                    let sc = dv(1.0, sq(e));
                    let t_prop = ft_raw.abs() * sc;
                    let l_prop = fl_raw.abs() * sc;
                    let nx = dv(kappa, w.peak_kappa.max(0.03));
                    let tan_a = if vlen < 1e-3 {
                        0.0
                    } else if vt.abs() > 0.0 {
                        dv(vs.abs(), vt.abs()).min(f32::from_bits(TAN_145))
                    } else {
                        f32::from_bits(TAN_145)
                    };
                    let ny = dv(tan_a, w.tan_peak_alpha);
                    let nn = sq(nx * nx + ny * ny);
                    let ws = if nn > 1e-6 {
                        clampf(nx.max(ny) - 1.0, 0.0, 1.0)
                    } else {
                        0.0
                    };
                    let t_cap = lerp(t_prop, if nn > 1e-6 { dv(at * ny, nn) } else { t_prop }, ws);
                    al_eff = lerp(l_prop, if nn > 1e-6 { dv(al * nx, nn) } else { l_prop }, ws);
                    f_t = clampf(ft_raw, -t_cap, t_cap);
                } else {
                    al_eff = al * sq((1.0 - dv(ft_raw, at) * dv(ft_raw, at)).max(0.0));
                }
            }
            let (w_new, f_l) = spin_solve(m, vt, ft, j, r, wd, b, al_eff, h);
            force = t * f_l + sd * f_t + n * fn_;
            out.contact = true;
            out.load = fn_;
            (w_new, kappa, alpha)
        } else {
            out.contact = false;
            out.load = 0.0;
            (spin_solve(m, 0.0, 0.0, j, r, wd, b, 0.0, h).0, 0.0, 0.0)
        };
        out.omega = w_new;
        let mut spin = st.spin + w_new * h;
        if spin > f32::from_bits(PI) {
            spin -= f32::from_bits(TAU);
        } else if spin <= -f32::from_bits(PI) {
            spin += f32::from_bits(TAU);
        }
        out.spin = spin;
        out.steer = angle;
        out.length = len[k];
        out.kappa = kappa;
        out.alpha = alpha;
        out.centre = xh + up * (w.rest_len - len[k]);
        out.axle = uw.cross(fwd);
        out.up = uw;
        d.wheels[k] = out;
        // 4. Onto the hub nodes, by mass.
        if out.contact {
            let (n1, n2) = (w.node1, w.node2);
            if n1 == n2 {
                apply(s, n1, force);
            } else {
                let r1 = dv(s.nodes[n1 as usize].mass, m);
                let r2 = dv(s.nodes[n2 as usize].mass, m);
                apply(s, n1, force * r1);
                apply(s, n2, force * r2);
            }
        }
    }
    d.smoothed = [forward, brake, hand, steer];
    d.tc_cut = cut;
    d.speed = speed;
    s.drive = Some(d);
}

impl Drive {
    /// Raycast wheels at rest for `config`.
    pub fn new(config: DriveConfig) -> Self {
        let n = config.wheels.len();
        Self {
            config,
            input: DriveInput::default(),
            smoothed: [0.0; 4],
            tc_cut: 0.0,
            wheels: vec![WheelState::default(); n],
            speed: 0.0,
        }
    }

    /// Spin every wheel to roll at `speed` (m/s, forward) without slipping.
    pub fn roll(&mut self, speed: f32) {
        for (s, w) in self.wheels.iter_mut().zip(&self.config.wheels) {
            s.omega = speed / w.radius;
        }
    }
}

fn curve_points(c: &engine_vehicle::Curve, what: &str) -> Result<Vec<[f32; 2]>, String> {
    if c.0.len() > CURVE_POINTS {
        return Err(format!(
            "raycast wheels: the {what} curve has {} points, at most {CURVE_POINTS}",
            c.0.len()
        ));
    }
    Ok(c.0.clone())
}

fn curve_peak_x(c: &[[f32; 2]]) -> f32 {
    // The x of the highest point (the first of equal ones).
    let mut best = (f32::MIN, 0.0f32);
    for p in c {
        if p[1] > best.0 {
            best = (p[1], p[0]);
        }
    }
    best.1
}

impl DriveConfig {
    /// The raycast wheels of `desc` (its `wheels`, node indices as spawned; the
    /// vehicle must be spawned without its tyres): the soft solver's raycast core
    /// ([`raycast_core`], default [`SpawnOptions`]) with its drivetrain as direct
    /// drive (see the module docs). Also returns notes on the choices made.
    pub fn from_desc(desc: &SoftVehicleDesc) -> Result<(Self, Vec<String>), String> {
        let mut notes = Vec::new();
        let plan = raycast_core(desc, &SpawnOptions::default())
            .ok_or_else(|| "raycast wheels: the vehicle has no wheels".to_string())?;
        let v = &plan.vehicle;
        if v.wheels.len() > MAX_WHEELS {
            return Err(format!(
                "raycast wheels: {} wheels, at most {MAX_WHEELS}",
                v.wheels.len()
            ));
        }
        let p = |i: u32| Vec3::from_array(desc.nodes[i as usize].position);
        let forward = Vec3::from_array(v.forward);
        let upv = Vec3::from_array(v.up);
        let left_v = upv.cross(forward);
        let mut used = std::collections::BTreeSet::new();
        let mut wheels = Vec::new();
        let mut any_driven = v.wheels.iter().any(|w| w.driven);
        if !any_driven {
            notes.push("no driven wheels in the file: the rear axle drives".into());
        }
        let rear = desc
            .wheels
            .iter()
            .map(|w| (0.5 * (p(w.node1) + p(w.node2))).dot(forward))
            .fold(f32::MAX, f32::min);
        for (i, (sw, w)) in desc.wheels.iter().zip(&v.wheels).enumerate() {
            for n in [sw.node1, sw.node2] {
                if !used.insert(n) && !(n == sw.node2 && sw.node1 == sw.node2) {
                    return Err(format!(
                        "raycast wheels: wheel {i} shares hub node {n} with another wheel"
                    ));
                }
            }
            let hub = 0.5 * (p(sw.node1) + p(sw.node2));
            let hub_rel = hub - plan.origin;
            let rest_len = (Vec3::from_array(w.position) - hub_rel).dot(upv);
            let (k, c) = match w.spring {
                engine_vehicle::Spring::StiffnessDamping { k, c } => (k, c),
                engine_vehicle::Spring::FrequencyDamping { .. } => {
                    return Err("raycast wheels: frequency springs need an effective mass".into())
                }
            };
            let m_w = k / (std::f32::consts::TAU * SpawnOptions::default().suspension_hz).powi(2);
            let (bump_k, bump_c) = match w.bump_stop {
                engine_vehicle::BumpStop::Spring { hz, zeta, .. } => {
                    let om = std::f32::consts::TAU * hz;
                    (m_w * om * om, 2.0 * m_w * zeta * om)
                }
                _ => (0.0, 0.0),
            };
            let long = curve_points(&w.long_friction, "longitudinal")?;
            let lat = curve_points(&w.lat_friction, "lateral")?;
            let peak_alpha = curve_peak_x(&lat);
            let driven = if any_driven {
                w.driven
            } else {
                (hub.dot(forward) - rear).abs() < 0.3
            };
            wheels.push(DriveWheel {
                node1: sw.node1,
                node2: sw.node2,
                radius: w.radius,
                width: w.width,
                inertia: w
                    .inertia
                    .unwrap_or(0.5 * w.mass * w.radius * w.radius)
                    .max(1e-3),
                angular_damping: w.angular_damping,
                susp_min: w.suspension_min,
                susp_max: w.suspension_max,
                rest_len,
                k,
                c,
                bump_k,
                bump_c,
                max_steer: if w.steerable { w.max_steer } else { 0.0 },
                driven,
                handbrake: w.handbrake,
                drive_mult: w.drive_multiplier,
                brake_max: w.max_brake_torque,
                handbrake_max: w.max_handbrake_torque,
                rolling: w.rolling_resistance,
                peak_kappa: curve_peak_x(&long),
                long_curve: long,
                lat_curve: lat,
                long_scale: w.long_mu_scale,
                lat_scale: w.lat_mu_scale,
                peak_alpha,
                tan_peak_alpha: peak_alpha.max(1.5).to_radians().tan(),
                side: sgn(hub.dot(left_v) - 0.0),
                axle_rest: [0.0, 0.0, 1.0],
            });
        }
        any_driven = wheels.iter().any(|w| w.driven);
        // Ackermann geometry from the wheel layout.
        let fpos: Vec<f32> = desc
            .wheels
            .iter()
            .map(|w| (0.5 * (p(w.node1) + p(w.node2))).dot(forward))
            .collect();
        let front = fpos.iter().copied().fold(f32::MIN, f32::max);
        let wheelbase = front - rear;
        let lats: Vec<f32> = desc
            .wheels
            .iter()
            .map(|w| (0.5 * (p(w.node1) + p(w.node2))).dot(left_v))
            .collect();
        let centre = 0.5
            * (lats.iter().copied().fold(f32::MIN, f32::max)
                + lats.iter().copied().fold(f32::MAX, f32::min));
        for (w, l) in wheels.iter_mut().zip(&lats) {
            w.side = sgn(l - centre);
        }
        let track = lats.iter().copied().fold(f32::MIN, f32::max)
            - lats.iter().copied().fold(f32::MAX, f32::min);
        // Drive: first-gear torque at the wheels, split over the driven wheels; top
        // speed in top gear at the engine's limit.
        let eng = desc.engine.clone().unwrap_or_default();
        let n_driven = wheels.iter().filter(|w| w.driven).count().max(1) as f32;
        let first = eng.gear_ratios.first().copied().unwrap_or(3.0);
        let top = eng.gear_ratios.last().copied().unwrap_or(1.0);
        let r_mean = wheels.iter().map(|w| w.radius).sum::<f32>() / wheels.len() as f32;
        let max_torque = if any_driven {
            eng.max_torque * first * eng.diff_ratio / n_driven
        } else {
            0.0
        };
        let top_speed =
            eng.max_rpm * std::f32::consts::TAU / 60.0 / (top * eng.diff_ratio) * r_mean;
        notes.push(format!(
            "raycast wheels: direct drive {max_torque:.0} N m per driven wheel, top speed {:.0} km/h",
            top_speed * 3.6
        ));
        // Frame nodes: the file's (centre, back, left), else the hubs (front right,
        // rear right, front left).
        let refs = match desc.refs {
            Some(r) => [r.center, r.back, r.left],
            None => {
                let pick = |front_axle: bool, left_side: bool| -> Result<u32, String> {
                    desc.wheels
                        .iter()
                        .zip(&fpos)
                        .zip(&lats)
                        .filter(|((_, &f), &l)| {
                            ((f - rear) > 0.5 * wheelbase) == front_axle
                                && (l > centre) == left_side
                        })
                        .map(|((w, _), _)| w.node1)
                        .next()
                        .ok_or_else(|| {
                            "raycast wheels: no reference nodes and no wheel at each corner"
                                .to_string()
                        })
                };
                [pick(true, false)?, pick(false, false)?, pick(true, true)?]
            }
        };
        let anti_roll: Vec<(u32, u32, f32)> = v
            .anti_roll
            .iter()
            .map(|a| (a.left, a.right, a.stiffness))
            .collect();
        if anti_roll.len() > MAX_ANTI_ROLL {
            return Err(format!(
                "raycast wheels: {} anti-roll bars, at most {MAX_ANTI_ROLL}",
                anti_roll.len()
            ));
        }
        let speed_curve = curve_points(&v.steering.speed_curve, "steering")?;
        Ok((
            Self {
                wheels,
                refs,
                max_torque,
                top_speed,
                steer_rate: v.steering.rate,
                ackermann: v.steering.ackermann,
                wheelbase,
                track,
                speed_curve,
                throttle_rate: v.response.throttle_rate,
                brake_rate: v.response.brake_rate,
                handbrake_rate: v.response.handbrake_rate,
                reverse_brake: v.reverse_brake,
                traction_control: v.assists.traction_control,
                tc_alpha: 1.0 - (-STEP / 0.1f32).exp(),
                max_slope_cos: v.max_slope_deg.to_radians().cos(),
                surface: 1.0,
                anti_roll,
                hub_axles: false,
            },
            notes,
        ))
    }
}

impl ExplicitBeams {
    /// Spawn a `.truck` vehicle without its soft tyres, on raycast wheels
    /// ([`DriveConfig::from_desc`]), yawed and moving as
    /// [`Self::from_truck_config_yaw`]; its wheels roll at the spawn's forward
    /// speed. Returns the notes on the drive's choices too.
    pub fn from_truck_raycast(
        text: &str,
        config: Option<&str>,
        position: Vec3,
        yaw_degrees: f32,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, Vec<String>), String> {
        let mut s = Self::from_truck_yawed_tyres(
            text,
            config,
            position,
            yaw_degrees,
            velocity,
            settings,
            &crate::explicit_aero::airfoil_from_env,
            false,
        )?;
        let desc = crate::truck::load_truck_config(text, config);
        let notes = s.attach_drive(&desc, velocity)?;
        Ok((s, notes))
    }

    /// A JBeam desc in JBeam mode ([`Self::from_desc_jbeam_with`]) on raycast wheels.
    pub fn from_desc_jbeam_raycast(
        desc: &SoftVehicleDesc,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, Vec<String>), String> {
        let (mut s, mut warnings) = Self::from_desc_jbeam_with(
            desc,
            crate::explicit_beams::JbeamWheels::Raycast,
            position,
            velocity,
            settings,
        )?;
        warnings.extend(s.attach_drive(desc, velocity)?);
        Ok((s, warnings))
    }

    /// Raycast wheels for this vehicle (spawned from `desc` without tyres), rolling
    /// at `velocity`'s forward part.
    pub fn attach_drive(
        &mut self,
        desc: &SoftVehicleDesc,
        velocity: Vec3,
    ) -> Result<Vec<String>, String> {
        let (config, notes) = DriveConfig::from_desc(desc)?;
        for sl in &self.slides {
            if config
                .wheels
                .iter()
                .any(|w| w.node1 == sl.node || w.node2 == sl.node)
            {
                return Err(format!(
                    "raycast wheels: hub node {} is a slide node",
                    sl.node
                ));
            }
        }
        for w in &config.wheels {
            for n in [w.node1, w.node2] {
                if n as usize >= self.nodes.len() {
                    return Err(format!(
                        "raycast wheels: hub node {n} past the vehicle's nodes"
                    ));
                }
            }
        }
        let mut config = config;
        let o = self.origin;
        // Each axle's spawn direction in the vehicle frame (drive_pass's correction).
        {
            let p = |i: u32| self.nodes[i as usize].pos + o;
            let [c, b, l] = config.refs;
            let fw = nrm(p(c) - p(b));
            let up = nrm(fw.cross(p(l) - p(c)));
            let left = up.cross(fw);
            for w in &mut config.wheels {
                let mut a = if w.node1 == w.node2 {
                    left
                } else {
                    nrm(p(w.node2) - p(w.node1))
                };
                if dot(a, left) < 0.0 {
                    a = -a;
                }
                w.axle_rest = [dot(a, fw), dot(a, up), dot(a, left)];
            }
        }
        let mut drive = Drive::new(config);
        let [c, b, _] = drive.config.refs;
        let fw = nrm(self.nodes[c as usize].pos + o - (self.nodes[b as usize].pos + o));
        drive.roll(dot(velocity, fw));
        self.drive = Some(Box::new(drive));
        Ok(notes)
    }

    /// Set the driver's input (no raycast wheels: nothing).
    pub fn set_drive_input(&mut self, input: DriveInput) {
        if let Some(d) = self.drive.as_mut() {
            d.input = input;
        }
    }
}

use crate::explicit_beams::ExplicitBeamsSettings;
