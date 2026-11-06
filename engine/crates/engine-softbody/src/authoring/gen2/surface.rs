//! The generated sedan's outer body surface, for its render model
//! ([`super::model`]): one smooth implicit surface f(p) = 0 built from
//! profile curves (side silhouette, plan, front section) joined by smooth
//! minimum and maximum, so every panel cut from it meets its neighbours on the
//! same surface and its normal is the field's gradient (no facets).
//!
//! Frame: the vehicle art spec's (glTF): metres, +Y up, the front facing +Z,
//! +X the car's left; the origin on the ground, centred, midway between the
//! axles. Dimensions are the spec's mid-size sedan (Accord or Camry class):
//! 4.88 m long, 1.86 m wide, 1.45 m high, a 2.83 m wheelbase, a 1.60 m track,
//! a 0.97 m front overhang, 0.14 m ground clearance, 235/45 R18 tyres.
//!
//! Panels are sampled through two families of rays from inside the body
//! ([`Param`]): vertical sections (planes z = u, rays at angle θ from straight
//! up, positive toward +X) for the middle of the car, and horizontal rays
//! about a vertical axis (plan angle φ, height y) for the nose and tail. A
//! point on the surface maps back to either ([`invert`]), so a boundary curve
//! designed in one family bounds panels sampled in the other.

use glam::Vec3;

pub const WHEELBASE: f32 = 2.83;
pub const AXLE_F: f32 = WHEELBASE / 2.0;
pub const AXLE_R: f32 = -WHEELBASE / 2.0;
pub const Z_FRONT: f32 = AXLE_F + 0.97;
pub const Z_REAR: f32 = Z_FRONT - 4.88;
pub const HALF_W: f32 = 0.93;
pub const HEIGHT: f32 = 1.45;
pub const TRACK_HALF: f32 = 0.80;
/// 235/45 R18: outer radius, section width, rim radius (18 in).
pub const TYRE_R: f32 = 0.3415;
pub const TYRE_W: f32 = 0.235;
pub const RIM_R: f32 = 0.2286;
/// Wheel-arch opening radius about the wheel centre: a 30 mm gap over the
/// tyre.
pub const ARCH_R: f32 = 0.372;
/// The vertical axes of the nose and tail families (at the arches' ends).
pub const ZC_F: f32 = AXLE_F + ARCH_R * 0.82;
pub const ZC_R: f32 = AXLE_R - ARCH_R * 0.82;
/// The section family's ray origin height.
pub const YC: f32 = 0.62;

/// Monotone piecewise cubic Hermite interpolation (Fritsch–Carlson): C¹, no
/// overshoot, through the knots.
#[derive(Clone, Debug)]
pub struct Spline {
    x: Vec<f32>,
    y: Vec<f32>,
    m: Vec<f32>,
}

impl Spline {
    pub fn new(pts: &[(f32, f32)]) -> Self {
        let n = pts.len();
        let x: Vec<f32> = pts.iter().map(|p| p.0).collect();
        let y: Vec<f32> = pts.iter().map(|p| p.1).collect();
        let d: Vec<f32> = (0..n - 1)
            .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            .collect();
        let mut m = vec![0.0; n];
        m[0] = d[0];
        m[n - 1] = d[n - 2];
        for i in 1..n - 1 {
            m[i] = if d[i - 1] * d[i] <= 0.0 {
                0.0
            } else {
                0.5 * (d[i - 1] + d[i])
            };
        }
        for i in 0..n - 1 {
            if d[i] == 0.0 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
                continue;
            }
            let (a, b) = (m[i] / d[i], m[i + 1] / d[i]);
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
        Spline { x, y, m }
    }

    pub fn at(&self, t: f32) -> f32 {
        let n = self.x.len();
        if t <= self.x[0] {
            return self.y[0] + self.m[0] * (t - self.x[0]);
        }
        if t >= self.x[n - 1] {
            return self.y[n - 1] + self.m[n - 1] * (t - self.x[n - 1]);
        }
        let i = self.x.partition_point(|&k| k <= t).min(n - 1) - 1;
        let h = self.x[i + 1] - self.x[i];
        let s = (t - self.x[i]) / h;
        let (s2, s3) = (s * s, s * s * s);
        (2.0 * s3 - 3.0 * s2 + 1.0) * self.y[i]
            + (s3 - 2.0 * s2 + s) * h * self.m[i]
            + (-2.0 * s3 + 3.0 * s2) * self.y[i + 1]
            + (s3 - s2) * h * self.m[i + 1]
    }

    pub fn slope(&self, t: f32) -> f32 {
        let e = 1e-3;
        (self.at(t + e) - self.at(t - e)) / (2.0 * e)
    }
}

/// Polynomial smooth maximum and minimum with blend radius `k`.
pub fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    b + (a - b) * h + k * h * (1.0 - h)
}
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    -smax(-a, -b, k)
}
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The profile curves (built once).
pub struct Profiles {
    /// Lower body's top along the centre line (hood, belt, deck), by z.
    pub top: Spline,
    /// Its crown (centre above the edges), by z.
    pub crown: Spline,
    /// The greenhouse's centre line (windshield, roof, backlight), by z.
    pub roof: Spline,
    /// The side glass's base (belt line), by z.
    pub belt: Spline,
    /// Body bottom, by z.
    pub bottom: Spline,
    /// Nose and tail extent by height y.
    pub nose: Spline,
    pub tail: Spline,
}

impl Profiles {
    pub fn new() -> Self {
        // The side glass's base: a gentle wedge, rising toward the tail.
        let belt = Spline::new(&[
            (Z_REAR, 0.985),
            (-1.80, 0.998),
            (-1.20, 0.995),
            (-0.30, 0.978),
            (0.40, 0.955),
            (0.80, 0.938),
            (Z_FRONT, 0.890),
        ]);
        Profiles {
            // A low bonnet falling to a slim nose; a short, near-level deck
            // ending in a crisp ducktail lip that overhangs the tail.
            top: Spline::new(&[
                (Z_REAR - 0.05, 0.996),
                (-2.445, 1.010),
                (-2.36, 1.016),
                (-2.20, 1.020),
                (-2.00, 1.022),
                (-1.80, 1.024),
                (-1.20, 1.026),
                (0.40, 0.975),
                (0.80, 0.952),
                (1.10, 0.928),
                (1.50, 0.885),
                (1.90, 0.830),
                (2.15, 0.786),
                (2.30, 0.752),
                (Z_FRONT + 0.05, 0.715),
            ]),
            crown: Spline::new(&[
                (Z_REAR, 0.025),
                (-2.0, 0.030),
                (-1.5, 0.030),
                (0.8, 0.035),
                (1.6, 0.045),
                (Z_FRONT, 0.040),
            ]),
            // A fastback roof: the crown just behind the B-pillar, a long,
            // shallow backlight down to a short deck.
            roof: Spline::new(&[
                (-2.35, 0.80),
                (-2.05, 1.030),
                (-1.68, 1.200),
                (-1.32, 1.335),
                (-1.00, 1.414),
                (-0.70, 1.446),
                (-0.40, 1.450),
                (-0.12, 1.428),
                (0.15, 1.350),
                (0.45, 1.185),
                (0.80, 0.975),
                (1.05, 0.78),
            ]),
            belt,
            bottom: Spline::new(&[
                (Z_REAR - 0.05, 0.44),
                (-2.35, 0.335),
                (-2.10, 0.235),
                (-1.85, 0.170),
                (-1.40, 0.150),
                (1.40, 0.150),
                (1.85, 0.170),
                (2.10, 0.215),
                (2.30, 0.285),
                (Z_FRONT + 0.05, 0.36),
            ]),
            // The nose: foremost at the grille's middle, leaning back above
            // it to the bonnet's edge.
            nose: Spline::new(&[
                (0.10, Z_FRONT - 0.19),
                (0.22, Z_FRONT - 0.100),
                (0.34, Z_FRONT - 0.032),
                (0.46, Z_FRONT - 0.002),
                (0.54, Z_FRONT),
                (0.62, Z_FRONT - 0.010),
                (0.68, Z_FRONT - 0.032),
                (0.74, Z_FRONT - 0.080),
                (0.85, Z_FRONT - 0.260),
            ]),
            // The tail: rearmost at the ducktail lip, the face nearly upright
            // under it, the bumper's lower half tucking under (the diffuser).
            tail: Spline::new(&[
                (0.15, Z_REAR + 0.21),
                (0.28, Z_REAR + 0.105),
                (0.42, Z_REAR + 0.035),
                (0.60, Z_REAR + 0.010),
                (0.80, Z_REAR + 0.016),
                (0.93, Z_REAR + 0.016),
                (1.00, Z_REAR),
                (1.06, Z_REAR + 0.06),
            ]),
        }
    }
}

impl Default for Profiles {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    static PROF: Profiles = Profiles::new();
}

/// The profiles (per thread, built once).
pub fn with_profiles<R>(f: impl FnOnce(&Profiles) -> R) -> R {
    PROF.with(|p| f(p))
}

/// Plan half width at (z, y): full width in the middle, rounded to the nose
/// and tail (the tips by height from [`Profiles::nose`]/`tail`).
/// Signed distance (about) to the plan outline at height y: a superellipse
/// round each end (semi-axes the half width and the end's length), smooth
/// across the centre line.
fn plan_d(p: &Profiles, x: f32, z: f32, y: f32) -> f32 {
    let w0 = 0.99;
    let (z0, tip, e) = if z > 0.0 {
        (1.78, p.nose.at(y), 2.5f32)
    } else {
        (-1.80, p.tail.at(y), 2.8f32)
    };
    let l = (tip - z0).abs().max(0.05);
    let zz = ((z - z0) * (tip - z0).signum()).max(0.0);
    let (xn, zn) = (x / w0, zz / l);
    let r = (xn.powf(e) + zn.powf(e)).powf(1.0 / e).max(1e-6);
    // |∇r| = sqrt((xn^(e−1)/w0)² + (zn^(e−1)/l)²) / r^(e−1).
    let re = r.powf(e - 1.0);
    let g = ((xn.powf(e - 1.0) / w0).powi(2) + (zn.powf(e - 1.0) / l).powi(2)).sqrt() / re;
    (r - 1.0) / g.max(1e-3)
}

/// Height of the shoulder character line (the side's crease) at z: rising
/// gently toward the tail (a wedge).
pub fn shoulder_y(z: f32) -> f32 {
    0.868 - 0.016 * (z / 2.4)
}

/// Height of the lower character line at z: the top edge of the doors'
/// scallop, sweeping up from the front door toward the rear wheel.
pub fn scallop_y(z: f32) -> f32 {
    0.50 + 0.095 * smoothstep(0.9, -1.25, z)
}

/// Section half width at height y (the body side's curvature, the shoulder
/// character line, the lower scallop and the tuck under to the rocker), at z
/// (the haunch over the rear wheel, the wheel-arch flares).
fn section_w(z: f32, y: f32) -> f32 {
    // Bodyside: fullest at 0.68 m, tucking under to the sill.
    let w0 = 0.902;
    let prof = |y: f32| {
        let t = (y - 0.68) / 0.48;
        w0 - if t < 0.0 { 0.062 } else { 0.045 } * t * t
    };
    let base = prof(y);
    // Above the shoulder line the side leans in (a crisp crease, 6 mm blend).
    let ys = shoulder_y(z);
    let above = prof(ys) - 0.30 * (y - ys);
    let mut w = smin(base, above, 0.012);
    // The haunch: the shoulder swells out over the rear wheel (and a little
    // over the front), under the crease.
    let band = smoothstep(0.42, 0.66, y) * smoothstep(ys + 0.03, ys - 0.06, y);
    let haunch = 0.017 * (-((z - (AXLE_R + 0.12)) / 0.62).powi(2)).exp()
        + 0.006 * (-((z - (AXLE_F - 0.05)) / 0.45).powi(2)).exp();
    w += haunch * band;
    // The lower scallop: below a crisp line the doors step in, deepest ahead
    // of the rear wheel, returning to the sill at the bottom; it fades out at
    // the arches.
    let yl = scallop_y(z);
    let depth = 0.006 + 0.012 * smoothstep(0.6, -0.9, z);
    // It rises out of the front door's leading edge and dies into the rear
    // door's trailing edge (under the haunch).
    let fade = smoothstep(0.76, 0.40, z) * smoothstep(-0.97, -0.62, z);
    // Crisp at the line (2 cm), a long concave return to the sill.
    let cut = smoothstep(yl + 0.006, yl - 0.02, y).powf(0.7) * smoothstep(0.34, 0.50, y);
    w -= depth * fade * cut;
    // The rear bumper's line: behind the rear wheel the side steps in 6 mm
    // below a crisp crease at 0.56 m, wrapping toward the tail.
    w -= 0.006
        * smoothstep(0.572, 0.552, y)
        * smoothstep(0.30, 0.36, y)
        * smoothstep(-1.80, -2.05, z);
    // Arch flares: a soft swell round each opening.
    for za in [AXLE_F, AXLE_R] {
        let r = ((z - za).powi(2) + (y - TYRE_R).powi(2)).sqrt();
        w += 0.018 * (-((r - 0.418) / 0.072).powi(2)).exp() * smoothstep(0.98, 0.70, y);
    }
    w
}

/// The greenhouse's half width at (z, y): the glass leans in (tumblehome) and
/// narrows toward the fastback's tail.
fn glass_w(p: &Profiles, z: f32, y: f32) -> f32 {
    let b = p.belt.at(z);
    let g = ((y - b) / (1.44 - b)).clamp(-0.5, 1.3);
    let w = 0.878 - 0.250 * g.max(0.0).powf(1.08) + 0.02 * (-g).max(0.0);
    let taper = 1.0 - 0.04 * smoothstep(-0.7, -1.95, z) - 0.02 * smoothstep(0.1, 0.8, z);
    w * taper
}

/// The body's signed field (negative inside, about the distance in metres
/// near the surface).
pub fn sdf(q: Vec3) -> f32 {
    with_profiles(|p| sdf_with(p, q))
}

pub fn sdf_with(p: &Profiles, q: Vec3) -> f32 {
    let (x, y, z) = (q.x.abs(), q.y, q.z);
    // Lower body: plan, section, top and bottom.
    let ws = section_w(z, y);
    let d_lat = smax(plan_d(p, x, z, y), x - ws, 0.05);
    let crown = p.crown.at(z);
    let top = p.top.at(z) - crown * (x / 0.93).powi(2);
    let ts = p.top.slope(z);
    let d_top = (y - top) / (1.0 + ts * ts).sqrt();
    let bs = p.bottom.slope(z);
    let d_bot = (p.bottom.at(z) - y) / (1.0 + bs * bs).sqrt();
    // The top's edge rounds over 45 mm, tightening to 20 mm at the tail (a
    // crisp lip).
    let k_top = 0.045 - 0.025 * smoothstep(-2.05, -2.40, z);
    let lower = smax(smax(d_lat, d_top, k_top), d_bot, 0.06);
    // Greenhouse.
    let rc = 0.05;
    let roof = p.roof.at(z) - rc * (x / 0.75).powi(2);
    let rs = p.roof.slope(z);
    let d_roof = (y - roof) / (1.0 + rs * rs).sqrt();
    let d_gl = x - glass_w(p, z, y);
    let d_gb = p.belt.at(z) - 0.2 - y;
    let gh = smax(smax(d_gl, d_roof, 0.055), d_gb, 0.02);
    // The greenhouse meets the body in a small fillet, a broader one round
    // the C-pillar's foot (a clean sweep into the deck).
    smin(lower, gh, 0.02 + 0.035 * smoothstep(-1.3, -2.0, z))
}

pub fn grad(q: Vec3) -> Vec3 {
    with_profiles(|p| {
        let e = 2e-4;
        let f = |d: Vec3| sdf_with(p, q + d);
        Vec3::new(
            f(Vec3::X * e) - f(-Vec3::X * e),
            f(Vec3::Y * e) - f(-Vec3::Y * e),
            f(Vec3::Z * e) - f(-Vec3::Z * e),
        ) / (2.0 * e)
    })
}

pub fn normal(q: Vec3) -> Vec3 {
    grad(q).normalize_or(Vec3::Y)
}

/// Pull a point near the surface onto it (Newton along the gradient).
pub fn project(mut q: Vec3) -> Vec3 {
    for _ in 0..6 {
        let f = sdf(q);
        let g = grad(q);
        let l2 = g.length_squared().max(1e-8);
        q -= g * (f / l2);
        if f.abs() < 1e-6 {
            break;
        }
    }
    q
}

/// A ray family over the surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// (u, θ): the section plane z = u, the ray from (0, YC, u) at angle θ
    /// from straight up, positive toward +X.
    Section,
    /// (φ, y): from (0, y, ZC_F), horizontal at plan angle φ from +Z, positive
    /// toward +X.
    Nose,
    /// (φ, y): from (0, y, ZC_R), horizontal at φ from −Z, positive toward +X.
    Tail,
}

pub fn ray(param: Param, a: f32, b: f32) -> (Vec3, Vec3) {
    match param {
        Param::Section => (Vec3::new(0.0, YC, a), Vec3::new(b.sin(), b.cos(), 0.0)),
        Param::Nose => (Vec3::new(0.0, b, ZC_F), Vec3::new(a.sin(), 0.0, a.cos())),
        Param::Tail => (Vec3::new(0.0, b, ZC_R), Vec3::new(a.sin(), 0.0, -a.cos())),
    }
}

/// The surface point of the ray (a, b): the first crossing out of the body.
pub fn hit(param: Param, a: f32, b: f32) -> Vec3 {
    with_profiles(|p| {
        let (o, d) = ray(param, a, b);
        let f = |t: f32| sdf_with(p, o + d * t);
        let mut t0 = 0.0;
        let mut f0 = f(0.0);
        if f0 >= 0.0 {
            return o;
        }
        let mut t1;
        loop {
            t1 = t0 + (-f0 * 0.8).max(0.002);
            let f1 = f(t1);
            if f1 >= 0.0 || t1 > 4.0 {
                break;
            }
            t0 = t1;
            f0 = f1;
        }
        for _ in 0..30 {
            let m = 0.5 * (t0 + t1);
            if f(m) < 0.0 {
                t0 = m;
            } else {
                t1 = m;
            }
        }
        o + d * (0.5 * (t0 + t1))
    })
}

/// The ray coordinates of a surface point.
pub fn invert(param: Param, q: Vec3) -> (f32, f32) {
    match param {
        Param::Section => (q.z, q.x.atan2(q.y - YC)),
        Param::Nose => (q.x.atan2(q.z - ZC_F), q.y),
        Param::Tail => (q.x.atan2(ZC_R - q.z), q.y),
    }
}

/// θ on section `u` where the surface (left side, θ in (0, π)) is at height
/// `y` (the side's height falls as θ grows).
pub fn theta_at_y(u: f32, y: f32) -> f32 {
    let (mut lo, mut hi) = (0.02f32, 3.12f32);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if hit(Param::Section, u, m).y > y {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// [`theta_at_y`] on the section's side only (below where its normal leans
/// past 60° from up), where the top may be as low as a side line.
pub fn theta_at_y_side(u: f32, y: f32) -> f32 {
    let (mut lo, mut hi) = (0.05f32, 2.5f32);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if normal(hit(Param::Section, u, m)).y > 0.5 {
            lo = m;
        } else {
            hi = m;
        }
    }
    let side0 = 0.5 * (lo + hi);
    let (mut lo, mut hi) = (side0, 3.12f32);
    if hit(Param::Section, u, side0).y < y {
        return side0;
    }
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if hit(Param::Section, u, m).y > y {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// θ on section `u` where the surface's top (left) is at half width `x`.
pub fn theta_at_x(u: f32, x: f32) -> f32 {
    let (mut lo, mut hi) = (0.0f32, 1.55f32);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if hit(Param::Section, u, m).x < x {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// θ on section `u` of the greenhouse's corner (roof rail or A/C-pillar):
/// where the surface normal leans 50° from straight up.
pub fn theta_corner(u: f32) -> f32 {
    let (mut lo, mut hi) = (0.05f32, 1.5f32);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        let n = normal(hit(Param::Section, u, m));
        if n.y > 50f32.to_radians().cos() {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// Height of the wheel-arch opening's edge at z (None outside it): the
/// circle above the wheel centre, straight down below it.
pub fn arch_y(z: f32) -> Option<f32> {
    for za in [AXLE_F, AXLE_R] {
        let d = z - za;
        if d.abs() <= ARCH_R {
            return Some(TYRE_R + (ARCH_R * ARCH_R - d * d).max(0.0).sqrt());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions() {
        // Length, width and height of the body (the spec's tolerances).
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for i in 0..240 {
            let u = Z_REAR + 0.01 + (Z_FRONT - Z_REAR - 0.02) * i as f32 / 239.0;
            for j in 0..120 {
                let th = -3.1 + 6.2 * j as f32 / 119.0;
                let p = hit(Param::Section, u, th);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        for j in 0..60 {
            let y = 0.2 + 0.8 * j as f32 / 59.0;
            lo = lo.min(hit(Param::Tail, 0.0, y));
            hi = hi.max(hit(Param::Nose, 0.0, y));
        }
        let l = hi.z - lo.z;
        let w = hi.x - lo.x;
        assert!((l - 4.88).abs() < 0.05, "length {l}");
        assert!((w - 1.86).abs() < 0.03, "width {w}");
        assert!((hi.y - 1.45).abs() < 0.03, "height {}", hi.y);
        assert!((lo.y - 0.14).abs() < 0.015, "clearance {}", lo.y);
    }

    #[test]
    fn invert_round_trips() {
        for &(param, a, b) in &[
            (Param::Section, 0.3, 0.9),
            (Param::Section, -1.2, 0.2),
            (Param::Nose, 0.4, 0.5),
            (Param::Tail, -0.6, 0.7),
        ] {
            let p = hit(param, a, b);
            let (a2, b2) = invert(param, p);
            assert!((a2 - a).abs() < 1e-3 && (b2 - b).abs() < 1e-3, "{param:?}");
            assert!(sdf(p).abs() < 1e-4);
        }
    }
}
