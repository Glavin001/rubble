//! Raycast wheels' records (`slang/kernels/explicit_beams_drive.slang`) from a CPU
//! [`Drive`], and back: shared by every explicit-beams host.
//!
//! gMisc at the drive header (the kernel's `driveRec()`, right after the rigid
//! coupling's two records): (on, region base, stride, 0). Vehicle v's region at
//! base + v·stride ([`DriveLayout::vehicle`]):
//!
//! ```text
//! 0  input (forward, steer, brake, handbrake)             host-written
//! 1  smoothed (forward, brake, handbrake, steer)          state
//! 2  (wheels, ref centre, ref back, ref left) u32         (0 wheels: no drive)
//! 3  (max torque, top speed, steer rate, Ackermann)
//! 4  (wheelbase, track, throttle rate, brake rate)
//! 5  (handbrake rate, flags u32: 1 reverse brake, 2 traction control, 4 hub axles, 8 rim stop, max slope cos, surface)
//! 6  (traction-control α, cut, speed, anti-roll bars u32)  cut, speed: state
//! 7  (speed-curve points u32, 0, 0, 0)
//! 8..11   speed curve, two points per record
//! 12..19  anti-roll bars (left u32, right u32, N/m, 0)
//! 20 + 21·w  wheel w:
//!    0  (node1, node2, flags u32: 1 driven 2 handbrake, long points | lat points << 16)
//!    1  (radius, width, inertia, angular damping)
//!    2  (bump length, natural length, hardpoint height, spring)
//!    3  (damper, bump spring, bump damper, max steer)
//!    4  (drive multiplier, brake, handbrake, rolling resistance)
//!    5  (long scale, lat scale, peak slip ratio, peak slip angle °)
//!    6  (tan peak angle, side, rim radius, 0)
//!    7..10   longitudinal curve; 11..14 lateral curve
//!    15 state (spin, spin angle, steer angle, length)
//!    16 state (load, slip ratio, slip angle °, contact 0/1)
//!    17..19 outputs: centre, axle, up (xyz, 0)
//!    20 the axle's spawn direction in the vehicle frame (forward, up, left, 0)
//! ```
//! Node indices are vehicle-local.

use engine_softbody::explicit_drive::{Drive, DriveInput, CURVE_POINTS, MAX_ANTI_ROLL};
use glam::Vec3;

/// Records per vehicle before its wheels, and per wheel.
pub const VEHICLE_RECS: u32 = 20;
pub const WHEEL_RECS: u32 = 23;

/// Where the drive records live in gMisc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveLayout {
    /// gMisc record of the header.
    pub header: u32,
    /// gMisc record of vehicle 0's region and the region's stride.
    pub base: u32,
    pub stride: u32,
}

impl DriveLayout {
    /// The stride for vehicles of at most `wheels` wheels.
    pub fn stride_for(wheels: usize) -> u32 {
        VEHICLE_RECS + WHEEL_RECS * wheels as u32
    }

    pub fn vehicle(&self, v: usize) -> u32 {
        self.base + v as u32 * self.stride
    }

    /// Byte offset in gMisc of vehicle v's input record.
    pub fn input_offset(&self, v: usize) -> u64 {
        u64::from(self.vehicle(v)) * 16
    }

    pub fn header_words(&self) -> [u32; 4] {
        [1, self.base, self.stride, 0]
    }
}

fn f4(x: f32, y: f32, z: f32, w: f32) -> [u32; 4] {
    [x.to_bits(), y.to_bits(), z.to_bits(), w.to_bits()]
}

fn v4(v: Vec3) -> [u32; 4] {
    f4(v.x, v.y, v.z, 0.0)
}

/// The input record.
pub fn input_words(i: DriveInput) -> [u32; 4] {
    f4(i.forward, i.steer, i.brake, i.handbrake)
}

fn curve(out: &mut [[u32; 4]], at: usize, pts: &[[f32; 2]]) {
    for (k, p) in pts.iter().take(CURVE_POINTS).enumerate() {
        let r = &mut out[at + k / 2];
        r[2 * (k % 2)] = p[0].to_bits();
        r[2 * (k % 2) + 1] = p[1].to_bits();
    }
}

/// A vehicle's region (`stride` records); `None`: no drive (zero wheels).
pub fn records(d: Option<&Drive>, stride: u32) -> Vec<[u32; 4]> {
    let mut r = vec![[0u32; 4]; stride as usize];
    let Some(d) = d else {
        return r;
    };
    let c = &d.config;
    r[0] = input_words(d.input);
    r[1] = f4(d.smoothed[0], d.smoothed[1], d.smoothed[2], d.smoothed[3]);
    r[2] = [c.wheels.len() as u32, c.refs[0], c.refs[1], c.refs[2]];
    r[3] = f4(c.max_torque, c.top_speed, c.steer_rate, c.ackermann);
    r[4] = f4(c.wheelbase, c.track, c.throttle_rate, c.brake_rate);
    let flags = c.reverse_brake as u32
        | (c.traction_control as u32) << 1
        | (c.hub_axles as u32) << 2
        | (c.rim_stop as u32) << 3;
    r[5] = [
        c.handbrake_rate.to_bits(),
        flags,
        c.max_slope_cos.to_bits(),
        c.surface.to_bits(),
    ];
    let ar = c.anti_roll.len().min(MAX_ANTI_ROLL);
    r[6] = [
        c.tc_alpha.to_bits(),
        d.tc_cut.to_bits(),
        d.speed.to_bits(),
        ar as u32,
    ];
    r[7] = [c.speed_curve.len().min(CURVE_POINTS) as u32, 0, 0, 0];
    curve(&mut r, 8, &c.speed_curve);
    for (k, &(l, rr, kk)) in c.anti_roll.iter().take(MAX_ANTI_ROLL).enumerate() {
        r[12 + k] = [l, rr, kk.to_bits(), 0];
    }
    for (k, (w, s)) in c.wheels.iter().zip(&d.wheels).enumerate() {
        let at = (VEHICLE_RECS + WHEEL_RECS * k as u32) as usize;
        let flags = w.driven as u32 | (w.handbrake as u32) << 1;
        let counts = w.long_curve.len().min(CURVE_POINTS) as u32
            | (w.lat_curve.len().min(CURVE_POINTS) as u32) << 16;
        r[at] = [w.node1, w.node2, flags, counts];
        r[at + 1] = f4(w.radius, w.width, w.inertia, w.angular_damping);
        r[at + 2] = f4(w.susp_min, w.susp_max, w.rest_len, w.k);
        r[at + 3] = f4(w.c, w.bump_k, w.bump_c, w.max_steer);
        r[at + 4] = f4(w.drive_mult, w.brake_max, w.handbrake_max, w.rolling);
        r[at + 5] = f4(w.long_scale, w.lat_scale, w.peak_kappa, w.peak_alpha);
        r[at + 6] = f4(w.tan_peak_alpha, w.side, w.rim_radius, 0.0);
        r[at + 20] = f4(w.axle_rest[0], w.axle_rest[1], w.axle_rest[2], 0.0);
        curve(&mut r, at + 7, &w.long_curve);
        curve(&mut r, at + 11, &w.lat_curve);
        r[at + 15] = f4(s.omega, s.spin, s.steer, s.length);
        r[at + 16] = f4(s.load, s.kappa, s.alpha, if s.contact { 1.0 } else { 0.0 });
        r[at + 17] = v4(s.centre);
        r[at + 18] = v4(s.axle);
        r[at + 19] = v4(s.up);
    }
    r
}

/// Read a vehicle's drive state back from its region (`words`: `stride` records as
/// u32) into `d`: smoothed input, traction cut, speed, every wheel's state.
pub fn decode(words: &[u32], d: &mut Drive) {
    let f = |rec: usize, k: usize| f32::from_bits(words[rec * 4 + k]);
    let v = |rec: usize| Vec3::new(f(rec, 0), f(rec, 1), f(rec, 2));
    d.smoothed = [f(1, 0), f(1, 1), f(1, 2), f(1, 3)];
    d.tc_cut = f(6, 1);
    d.speed = f(6, 2);
    for (k, s) in d.wheels.iter_mut().enumerate() {
        let at = (VEHICLE_RECS + WHEEL_RECS * k as u32) as usize;
        s.omega = f(at + 15, 0);
        s.spin = f(at + 15, 1);
        s.steer = f(at + 15, 2);
        s.length = f(at + 15, 3);
        s.load = f(at + 16, 0);
        s.kappa = f(at + 16, 1);
        s.alpha = f(at + 16, 2);
        s.contact = f(at + 16, 3) != 0.0;
        s.centre = v(at + 17);
        s.axle = v(at + 18);
        s.up = v(at + 19);
    }
}
