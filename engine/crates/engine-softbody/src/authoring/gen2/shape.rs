//! The generated sedan's outer surface: one analytic shape that both the physics
//! nodes and the render mesh are sampled from (`engine/docs/vehicle-gen2.md`).
//!
//! JBeam's frame throughout: +X left, +Y rear, +Z up, metres; the ground at z = 0,
//! the front axle at y = 0. Dimensions are a mid-size sedan's (the 2011 Accord
//! sedan: 4.93 m long, 1.85 m wide, 1.475 m high, a 2.80 m wheelbase).
//!
//! Three patches per side:
//! * `side(y, z)`: the lower body, z from the sill's bottom to the belt line;
//!   x is the plan outline (a superellipse, which rounds the nose and tail
//!   corners) times a gentle bulge that peaks at the shoulder height;
//! * `glass(y, g)`: the greenhouse side, g = 0 at the belt line to 1 at the roof
//!   rail (tumblehome);
//! * `top(y, u)`: hood, windshield, roof, rear window and deck lid, u = x/w_t
//!   from −1 (right edge) to 1 (left edge), crowned.

use glam::Vec3;

/// Wheel centre height (m): a 215/60 R16 tyre, 0.33 m loaded radius.
pub const WHEEL_R: f32 = 0.33;
/// Wheel-arch cut-out radius about the wheel centre (m).
pub const ARCH_R: f32 = 0.415;
/// Half track (m): the Accord's 1.59 m.
pub const TRACK_HALF: f32 = 0.795;
/// Front and rear axle y (m).
pub const AXLE_F: f32 = 0.0;
pub const AXLE_R: f32 = 2.80;
/// Body ends (m).
pub const Y_FRONT: f32 = -0.98;
pub const Y_REAR: f32 = 3.95;
/// Half width at the shoulder (m).
pub const HALF_W: f32 = 0.925;
/// Plan outline exponent: corners rounded as a real sedan's bumpers.
const PLAN_P: f32 = 6.0;

/// Monotone piecewise-cubic (smoothstep between knots) through `pts` (x sorted).
pub fn smooth(pts: &[(f32, f32)], x: f32) -> f32 {
    if x <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if x <= b.0 {
            let t = (x - a.0) / (b.0 - a.0);
            let s = t * t * (3.0 - 2.0 * t);
            return a.1 + (b.1 - a.1) * s;
        }
    }
    pts[pts.len() - 1].1
}

/// The plan outline's half width at `y` (0 at the tips).
pub fn plan_x(y: f32) -> f32 {
    let mid = 0.5 * (Y_FRONT + Y_REAR);
    let half = 0.5 * (Y_REAR - Y_FRONT);
    let t = ((y - mid) / half).abs().min(1.0);
    HALF_W * (1.0 - t.powf(PLAN_P)).max(0.0).powf(1.0 / PLAN_P)
}

/// Belt line height: hood edge, beltline, deck edge (m).
pub fn z_belt(y: f32) -> f32 {
    smooth(
        &[
            (Y_FRONT, 0.66),
            (-0.80, 0.76),
            (-0.30, 0.82),
            (0.80, 0.93),
            (1.90, 0.975),
            (2.95, 1.01),
            (3.40, 1.03),
            (3.85, 1.00),
            (Y_REAR, 0.94),
        ],
        y,
    )
}

/// Lowest point of the body side (m).
pub fn z_bottom(y: f32) -> f32 {
    smooth(
        &[
            (Y_FRONT, 0.30),
            (-0.70, 0.22),
            (-0.30, 0.18),
            (3.30, 0.18),
            (3.70, 0.26),
            (Y_REAR, 0.34),
        ],
        y,
    )
}

/// The roof's centre line height (the top patch's crown height at u = 0) inside
/// the greenhouse, `None` outside it.
pub fn roof_centre(y: f32) -> Option<f32> {
    if !(0.80..=3.32).contains(&y) {
        return None;
    }
    Some(smooth(
        &[
            (0.80, z_belt(0.80) + 0.05),
            (1.20, 1.25),
            (1.58, 1.435),
            (2.10, 1.475),
            (2.60, 1.455),
            (2.82, 1.40),
            (3.32, z_belt(3.32) + 0.03),
        ],
        y,
    ))
}

/// Greenhouse height fraction at y (0 outside, 1 at the full roof).
pub fn gh(y: f32) -> f32 {
    match roof_centre(y) {
        None => 0.0,
        Some(zc) => ((zc - 0.03 - z_belt(y)) / 0.42).clamp(0.0, 1.0),
    }
}

/// Lateral bulge of the side at height z: 1 at the shoulder, tucked at the
/// rocker and at the belt.
fn bulge(z: f32) -> f32 {
    let t = (z - 0.62) / 0.45;
    1.0 - 0.055 * t * t
}

/// A point on the lower side (left; mirror x for the right).
pub fn side(y: f32, z: f32) -> Vec3 {
    Vec3::new(plan_x(y) * bulge(z), y, z)
}

/// The side's outward normal at (y, z), numerically.
pub fn side_normal(y: f32, z: f32) -> Vec3 {
    let e = 1e-3;
    let dy = side(y + e, z) - side(y - e, z);
    let dz = side(y, z + e) - side(y, z - e);
    let n = dy.cross(dz).normalize_or(Vec3::X);
    if n.x < 0.0 {
        -n
    } else {
        n
    }
}

/// The top patch's edge half width and edge height at y.
pub fn top_edge(y: f32) -> (f32, f32) {
    let zb = z_belt(y);
    let xs = side(y, zb).x;
    let g = gh(y);
    let w_t = xs - (xs - 0.645) * g;
    let z_e = match roof_centre(y) {
        Some(zc) => (zb + (zc - 0.03 - zb).max(0.0)).max(zb),
        None => zb,
    };
    (w_t.max(0.0), z_e)
}

/// The crown of the top patch at y (m above its edge at the centre line).
fn crown(y: f32) -> f32 {
    smooth(&[(Y_FRONT, 0.03), (-0.6, 0.055), (0.6, 0.06), (1.6, 0.03), (2.8, 0.03), (3.4, 0.035), (Y_REAR, 0.03)], y)
}

/// A point on the top (u = x / w_t, −1 right edge … 1 left edge).
pub fn top(y: f32, u: f32) -> Vec3 {
    let (w_t, z_e) = top_edge(y);
    let z = z_e + crown(y) * (1.0 - u * u);
    Vec3::new(u * w_t, y, z)
}

/// A point on the greenhouse side (left), g = 0 belt, 1 roof rail.
pub fn glass(y: f32, g: f32) -> Vec3 {
    let zb = z_belt(y);
    let a = side(y, zb);
    let (w_t, z_e) = top_edge(y);
    let b = Vec3::new(w_t, y, z_e);
    // A slight outward curve (side glass is curved).
    let mid = 4.0 * g * (1.0 - g) * 0.015;
    a.lerp(b, g) + Vec3::X * mid
}

/// Whether (y, z) on the side lies in a wheel arch's cut-out.
pub fn in_arch(y: f32, z: f32) -> bool {
    [AXLE_F, AXLE_R]
        .iter()
        .any(|&ya| (y - ya).powi(2) + (z - WHEEL_R).powi(2) < ARCH_R * ARCH_R)
}

/// The arch edge height above the wheel centre line at y (None off the arch).
pub fn arch_z(y: f32) -> Option<f32> {
    for ya in [AXLE_F, AXLE_R] {
        let d = y - ya;
        if d.abs() < ARCH_R {
            return Some(WHEEL_R + (ARCH_R * ARCH_R - d * d).sqrt());
        }
    }
    None
}

/// Mirror a left point to the right.
pub fn mirror(p: Vec3) -> Vec3 {
    Vec3::new(-p.x, p.y, p.z)
}
