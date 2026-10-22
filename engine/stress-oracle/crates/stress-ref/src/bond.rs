//! Bond geometry, stiffness and kinematics.
//!
//! A bond is a beam-type joint between two rigid chunks (the rigid-body-spring /
//! Applied-Element family). It sits at the centroid `c` of the contact patch with a
//! local frame `(t1, t2, n)`; `n` points from chunk `a` to chunk `b`. Six springs act
//! there:
//!
//! | component        | stiffness        |
//! |------------------|------------------|
//! | axial (along n)  | `kn = E A / L`   |
//! | shear (t1, t2)   | `ks = G A / L`   |
//! | bending about t1 | `E I_t1 / L`     |
//! | bending about t2 | `E I_t2 / L`     |
//! | torsion about n  | `G J / L`        |
//!
//! `L` is the chunk spacing measured along the normal (centre of `a` to the patch plus
//! the patch to the centre of `b`). `I_t1` is the second moment of the patch for bending
//! about `t1` (stress varies along `t2`). The patch is treated as a `width[0] x width[1]`
//! rectangle (`width[0]` along `t1`).
//!
//! Generalised displacements are the relative motion of `b` with respect to `a` at the
//! patch centroid: `Delta = (u_b + theta_b x r_b) - (u_a + theta_a x r_a)`,
//! `Phi = theta_b - theta_a`, with `r_x = c - x_x`. Generalised forces follow the sign
//! convention "tension positive": the axial force `N > 0` pulls the chunks together.

use crate::material::Material;
use crate::math::{Mat3, Vec3};
use crate::scene::{BondDesc, ChunkDesc};

/// Local components: index 0 = t1, 1 = t2, 2 = n.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Local6 {
    /// Translational part (shear t1, shear t2, axial n).
    pub lin: Vec3,
    /// Rotational part (bending about t1, bending about t2, torsion about n).
    pub ang: Vec3,
}

impl Local6 {
    pub const ZERO: Local6 = Local6 { lin: Vec3::ZERO, ang: Vec3::ZERO };
    pub fn dot(&self, o: &Local6) -> f64 {
        self.lin.dot(o.lin) + self.ang.dot(o.ang)
    }
    pub fn scale(&self, s: f64) -> Local6 {
        Local6 { lin: self.lin * s, ang: self.ang * s }
    }
    pub fn add(&self, o: &Local6) -> Local6 {
        Local6 { lin: self.lin + o.lin, ang: self.ang + o.ang }
    }
    pub fn mul_elem(&self, k: &Local6) -> Local6 {
        Local6 { lin: self.lin.mul_elem(k.lin), ang: self.ang.mul_elem(k.ang) }
    }
    /// Components in the order `[lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]`.
    pub fn to_array(&self) -> [f64; 6] {
        [self.lin.x, self.lin.y, self.lin.z, self.ang.x, self.ang.y, self.ang.z]
    }
    pub fn from_array(a: [f64; 6]) -> Local6 {
        Local6 { lin: Vec3::new(a[0], a[1], a[2]), ang: Vec3::new(a[3], a[4], a[5]) }
    }
}

/// A bond's 6x6 stiffness in local generalised coordinates (component order as in
/// [`Local6::to_array`]); the Newton tangent of a joint whose components couple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat6(pub [[f64; 6]; 6]);

impl Mat6 {
    pub fn diag(d: &Local6) -> Mat6 {
        let a = d.to_array();
        let mut m = [[0.0; 6]; 6];
        for i in 0..6 {
            m[i][i] = a[i];
        }
        Mat6(m)
    }
    pub fn mul(&self, x: &Local6) -> Local6 {
        let v = x.to_array();
        Local6::from_array(std::array::from_fn(|i| (0..6).map(|j| self.0[i][j] * v[j]).sum()))
    }
    pub fn add_diag(&self, d: &Local6) -> Mat6 {
        let mut m = *self;
        for (i, x) in d.to_array().iter().enumerate() {
            m.0[i][i] += x;
        }
        m
    }
}

/// Undeformed geometry of a bond, in the body frame.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BondGeometry {
    pub a: usize,
    pub b: usize,
    pub centroid: Vec3,
    pub normal: Vec3,
    pub t1: Vec3,
    pub t2: Vec3,
    pub area: f64,
    pub width: [f64; 2],
    /// Chunk spacing along the normal.
    pub length: f64,
    /// Patch centroid relative to chunk centres.
    pub ra: Vec3,
    pub rb: Vec3,
    /// Second moment for bending about t1 (`w0 w1^3 / 12`) and about t2 (`w1 w0^3 / 12`).
    pub i_t1: f64,
    pub i_t2: f64,
    /// Saint-Venant torsion constant of the rectangle.
    pub torsion_constant: f64,
    /// Elastic section moduli: extreme-fibre bending stress = M / S.
    pub s_t1: f64,
    pub s_t2: f64,
    /// Torsional section modulus: max shear stress = T / W.
    pub torsion_modulus: f64,
    /// Effective friction radius of the patch (torsional friction lever).
    pub friction_radius: f64,
}

impl BondGeometry {
    pub fn from_desc(desc: &BondDesc, chunks: &[ChunkDesc]) -> BondGeometry {
        let xa = Vec3::from_array(chunks[desc.a].center);
        let xb = Vec3::from_array(chunks[desc.b].center);
        BondGeometry::new(
            desc.a,
            desc.b,
            xa,
            xb,
            Vec3::from_array(desc.centroid),
            Vec3::from_array(desc.normal),
            Vec3::from_array(desc.tangent),
            desc.area,
            desc.width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: usize,
        b: usize,
        xa: Vec3,
        xb: Vec3,
        centroid: Vec3,
        normal: Vec3,
        tangent: Vec3,
        area: f64,
        width: [f64; 2],
    ) -> BondGeometry {
        let n = normal.normalized();
        let t1 = (tangent - n * tangent.dot(n)).normalized();
        let t2 = n.cross(t1);
        let ra = centroid - xa;
        let rb = centroid - xb;
        let length = (ra.dot(n)).abs() + (rb.dot(n)).abs();
        assert!(length > 0.0, "bond {a}-{b}: zero chunk spacing along the normal");
        let (w0, w1) = (width[0], width[1]);
        let i_t1 = w0 * w1.powi(3) / 12.0;
        let i_t2 = w1 * w0.powi(3) / 12.0;
        let (long, short) = if w0 >= w1 { (w0, w1) } else { (w1, w0) };
        let ratio = short / long;
        let torsion_constant = long * short.powi(3) * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio.powi(4) / 12.0));
        let alpha = 1.0 / (3.0 + 1.8 * ratio);
        let torsion_modulus = alpha * long * short * short;
        BondGeometry {
            a,
            b,
            centroid,
            normal: n,
            t1,
            t2,
            area,
            width,
            length,
            ra,
            rb,
            i_t1,
            i_t2,
            torsion_constant,
            s_t1: i_t1 / (0.5 * w1),
            s_t2: i_t2 / (0.5 * w0),
            torsion_modulus,
            friction_radius: (w0 + w1) / 6.0,
        }
    }

    /// Body-frame vector -> local (t1, t2, n) components.
    pub fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.t1), v.dot(self.t2), v.dot(self.normal))
    }

    /// Local components -> body-frame vector.
    pub fn to_body(&self, l: Vec3) -> Vec3 {
        self.t1 * l.x + self.t2 * l.y + self.normal * l.z
    }

    /// Generalised relative displacement (local) from the chunk DOFs.
    pub fn kinematics(&self, ua: Vec3, tha: Vec3, ub: Vec3, thb: Vec3) -> Local6 {
        let delta = (ub + thb.cross(self.rb)) - (ua + tha.cross(self.ra));
        let phi = thb - tha;
        Local6 { lin: self.to_local(delta), ang: self.to_local(phi) }
    }

    /// Chunk loads produced by local generalised bond forces `q` (tension-positive
    /// convention). Returns `(force_a, moment_a, force_b, moment_b)` in the body
    /// frame; moments are about each chunk's centre. This is the transpose of
    /// [`kinematics`](Self::kinematics), so work and energy are consistent.
    pub fn chunk_loads(&self, q: &Local6) -> (Vec3, Vec3, Vec3, Vec3) {
        let f = self.to_body(q.lin);
        let m = self.to_body(q.ang);
        let fa = f;
        let fb = -f;
        let ma = m + self.ra.cross(fa);
        let mb = -m + self.rb.cross(fb);
        (fa, ma, fb, mb)
    }

    /// The 6x6 coupling block helpers for the static solver: `B_a`, `B_b` such that
    /// `delta = B_b [u_b; th_b] - B_a [u_a; th_a]` (translational part), as 3x3 pieces.
    pub fn skew_ra(&self) -> Mat3 {
        Mat3::skew(self.ra)
    }
    pub fn skew_rb(&self) -> Mat3 {
        Mat3::skew(self.rb)
    }

    /// Co-rotational kinematics at the chunk state (`sim.methods.corotational_bonds`).
    pub fn corotation(&self, ua: Vec3, tha: Vec3, ub: Vec3, thb: Vec3) -> CoRotation {
        CoRotation::new(self, ua, &ChunkTurn::new(tha), ub, &ChunkTurn::new(thb))
    }

    /// `corotation` with the chunks' turns computed once per substep.
    pub fn corotation_with(&self, ua: Vec3, ta: &ChunkTurn, ub: Vec3, tb: &ChunkTurn) -> CoRotation {
        CoRotation::new(self, ua, ta, ub, tb)
    }
}

/// A chunk's finite rotation (rotation vector `th`) in the forms the co-rotational bonds
/// use: its matrix, unit quaternion and Rodrigues coefficients. Computed once per chunk
/// and substep rather than once per bond.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChunkTurn {
    th: Vec3,
    rot: Mat3,
    quat: crate::math::Quat,
    /// `sin t / t`, `(1 - cos t) / t^2`.
    a: f64,
    b: f64,
}

impl ChunkTurn {
    pub fn new(th: Vec3) -> ChunkTurn {
        let t = th.norm();
        let (a, b, _) = rotation_coefficients(t);
        let k = Mat3::skew(th);
        // `cos(t / 2)` and `sin(t / 2) / t` (series below 1e-2, as the coefficients).
        let (w, s) = if t < 1e-2 {
            let t2 = t * t;
            (1.0 - t2 / 8.0 * (1.0 - t2 / 48.0), 0.5 - t2 / 48.0 * (1.0 - t2 / 80.0))
        } else {
            let (sh, ch) = (0.5 * t).sin_cos();
            (ch, sh / t)
        };
        ChunkTurn { th, rot: Mat3::IDENTITY + k * a + k * k * b, quat: crate::math::Quat { w, x: th.x * s, y: th.y * s, z: th.z * s }, a, b }
    }

    /// `(R - I) x` without forming `R x - x`.
    fn minus_identity(&self, x: Vec3) -> Vec3 {
        let vx = self.th.cross(x);
        vx * self.a + self.th.cross(vx) * self.b
    }
}

/// The rotation vector of a unit quaternion near the identity side (`w >= 0` after the
/// sign choice): `2 atan2(|v|, w) v / |v|`, by series where `|v|` is small.
fn quaternion_log(q: crate::math::Quat) -> Vec3 {
    let (w, v) = if q.w < 0.0 { (-q.w, Vec3::new(-q.x, -q.y, -q.z)) } else { (q.w, Vec3::new(q.x, q.y, q.z)) };
    let s = v.norm();
    let f = if s < 1e-3 * w {
        // atan2(s, w) / s = (1 / w)(1 - r / 3 + r^2 / 5), r = (s / w)^2.
        let r = (s / w) * (s / w);
        2.0 / w * (1.0 - r / 3.0 + r * r / 5.0)
    } else {
        2.0 * s.atan2(w) / s
    };
    v * f
}

/// `sin(t) / t`, `(1 - cos t) / t^2` and `(t - sin t) / t^3` for `t = |v|` (series
/// below 1e-2, where the closed forms cancel).
fn rotation_coefficients(t: f64) -> (f64, f64, f64) {
    if t < 1e-2 {
        let t2 = t * t;
        (1.0 - t2 / 6.0 * (1.0 - t2 / 20.0), 0.5 - t2 / 24.0 * (1.0 - t2 / 30.0), 1.0 / 6.0 - t2 / 120.0 * (1.0 - t2 / 42.0))
    } else {
        let s = (0.5 * t).sin();
        (t.sin() / t, 2.0 * s * s / (t * t), (t - t.sin()) / (t * t * t))
    }
}

/// Rotation matrix of the rotation vector `v` (Rodrigues).
#[cfg(test)]
fn rotation(v: Vec3) -> Mat3 {
    let (a, b, _) = rotation_coefficients(v.norm());
    let k = Mat3::skew(v);
    Mat3::IDENTITY + k * a + k * k * b
}

/// Right Jacobian of the exponential map: `R(v + dv) = R(v) exp(J_r(v) dv)` to first order.
#[cfg(test)]
fn right_jacobian(v: Vec3) -> Mat3 {
    let (_, b, c) = rotation_coefficients(v.norm());
    let k = Mat3::skew(v);
    Mat3::IDENTITY - k * b + k * k * c
}

/// Inverse of the right Jacobian.
fn right_jacobian_inverse(v: Vec3) -> Mat3 {
    let t = v.norm();
    let e = if t < 1e-2 {
        let t2 = t * t;
        1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    } else {
        1.0 / (t * t) - (1.0 + t.cos()) / (2.0 * t * t.sin())
    };
    let k = Mat3::skew(v);
    Mat3::IDENTITY + k * 0.5 + k * k * e
}

/// Geometrically exact (co-rotational) bond kinematics at one chunk state.
///
/// The small-strain map `kinematics` is invariant under infinitesimal rotations only:
/// its forces act at undeformed lever arms, and internal forces turn the system by
/// `sum (u_a - u_b) x f` (DECISIONS.md 36). Here each chunk turns by the finite rotation
/// `R(theta)` of its rotation vector; the bond deforms by the patch points' separation
/// `Delta = (u_b - u_a) + (R_b - I) r_b - (R_a - I) r_a` and the relative rotation
/// `phi = log(R_a^T R_b)`, both read in the mean frame `R_m = R_a exp(phi / 2)`. A rigid
/// rotation of the whole hidden field leaves both unchanged, so the bond forces (the
/// exact transpose of the map's variation, `loads`) exert no net moment about the
/// deformed configuration, and remain the gradient of the bond energy. To first order
/// in the rotations it is `kinematics`.
#[derive(Clone, Copy, Debug)]
pub struct CoRotation {
    t1: Vec3,
    t2: Vec3,
    n: Vec3,
    ra: Vec3,
    rb: Vec3,
    rot_a: Mat3,
    rot_b: Mat3,
    rot_m: Mat3,
    /// `R_a^T R_b`, `exp(phi / 2)`.
    rot_ab: Mat3,
    half: Mat3,
    /// Inverse right Jacobian of `phi`, right Jacobian of `phi / 2`.
    jac_phi_inv: Mat3,
    jac_half: Mat3,
    /// `R_m^T Delta` and `phi` (body-frame components).
    d: Vec3,
    phi: Vec3,
}

impl CoRotation {
    fn new(g: &BondGeometry, ua: Vec3, ta: &ChunkTurn, ub: Vec3, tb: &ChunkTurn) -> CoRotation {
        let (rot_a, rot_b) = (ta.rot, tb.rot);
        let phi = quaternion_log(ta.quat.conjugate() * tb.quat);
        let (a, b, c) = rotation_coefficients(0.5 * phi.norm());
        let k = Mat3::skew(phi * 0.5);
        let k2 = k * k;
        let half = Mat3::IDENTITY + k * a + k2 * b;
        let rot_m = rot_a * half;
        let delta = (ub - ua) + tb.minus_identity(g.rb) - ta.minus_identity(g.ra);
        CoRotation {
            t1: g.t1,
            t2: g.t2,
            n: g.normal,
            ra: g.ra,
            rb: g.rb,
            rot_a,
            rot_b,
            rot_m,
            rot_ab: rot_a.transpose() * rot_b,
            half,
            jac_phi_inv: right_jacobian_inverse(phi),
            jac_half: Mat3::IDENTITY - k * b + k2 * c,
            d: rot_m.transpose() * delta,
            phi,
        }
    }

    fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.t1), v.dot(self.t2), v.dot(self.n))
    }

    fn to_body(&self, l: Vec3) -> Vec3 {
        self.t1 * l.x + self.t2 * l.y + self.n * l.z
    }

    /// Generalised relative displacement (local).
    pub fn deformation(&self) -> Local6 {
        Local6 { lin: self.to_local(self.d), ang: self.to_local(self.phi) }
    }

    /// The variation of `deformation` along chunk displacement increments and spatial
    /// rotation increments (`R -> exp(w) R`), or the rate along velocities and angular
    /// velocities.
    pub fn variation(&self, dua: Vec3, wa: Vec3, dub: Vec3, wb: Vec3) -> Local6 {
        let (ea, eb) = (self.rot_a.transpose() * wa, self.rot_b.transpose() * wb);
        let ddelta = (dub - dua) - self.rot_b * self.rb.cross(eb) + self.rot_a * self.ra.cross(ea);
        let dphi = self.jac_phi_inv * (eb - self.rot_ab.transpose() * ea);
        let eta = self.half.transpose() * ea + self.jac_half * dphi * 0.5;
        let dd = self.rot_m.transpose() * ddelta + self.d.cross(eta);
        Local6 { lin: self.to_local(dd), ang: self.to_local(dphi) }
    }

    /// Chunk loads of local generalised bond forces `q` (tension-positive): the
    /// transpose of `variation`, negated, as `BondGeometry::chunk_loads`. Returns
    /// `(force_a, moment_a, force_b, moment_b)`, moments spatial (conjugate to spatial
    /// rotation increments).
    pub fn loads(&self, q: &Local6) -> (Vec3, Vec3, Vec3, Vec3) {
        let (ql, qa) = (self.to_body(q.lin), self.to_body(q.ang));
        let g = self.rot_m * ql;
        let h = ql.cross(self.d);
        let k = self.jac_phi_inv.transpose() * (qa + self.jac_half.transpose() * h * 0.5);
        let ca = -self.ra.cross(self.rot_a.transpose() * g) + self.half * h - self.rot_ab * k;
        let cb = self.rb.cross(self.rot_b.transpose() * g) + k;
        (g, -(self.rot_a * ca), -g, -(self.rot_b * cb))
    }
}

/// Elastic stiffness of the six bond springs (possibly with scaled moduli).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BondStiffness {
    pub kn: f64,
    pub ks: f64,
    pub kb_t1: f64,
    pub kb_t2: f64,
    pub kt: f64,
}

impl BondStiffness {
    pub fn new(g: &BondGeometry, m: &Material, stiffness_scale: f64) -> BondStiffness {
        let e = m.youngs_modulus * stiffness_scale;
        let gs = m.shear_modulus() * stiffness_scale;
        let l = g.length;
        BondStiffness {
            kn: e * g.area / l,
            ks: gs * g.area / l,
            kb_t1: e * g.i_t1 / l,
            kb_t2: e * g.i_t2 / l,
            kt: gs * g.torsion_constant / l,
        }
    }

    pub fn as_local(&self) -> Local6 {
        Local6 { lin: Vec3::new(self.ks, self.ks, self.kn), ang: Vec3::new(self.kb_t1, self.kb_t2, self.kt) }
    }

    pub fn scaled(&self, s: f64) -> BondStiffness {
        BondStiffness { kn: self.kn * s, ks: self.ks * s, kb_t1: self.kb_t1 * s, kb_t2: self.kb_t2 * s, kt: self.kt * s }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_bond() -> BondGeometry {
        BondGeometry::new(
            0,
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
            Vec3::Y,
            0.04,
            [0.2, 0.2],
        )
    }

    #[test]
    fn rigid_motion_produces_no_deformation() {
        let g = unit_bond();
        // A common rigid rotation theta about the origin: u = theta x x.
        let th = Vec3::new(0.01, -0.02, 0.03);
        let xa = Vec3::ZERO;
        let xb = Vec3::new(1.0, 0.0, 0.0);
        let k = g.kinematics(th.cross(xa), th, th.cross(xb), th);
        assert!(k.lin.norm() < 1e-15 && k.ang.norm() < 1e-15);
    }

    #[test]
    fn loads_are_self_equilibrated() {
        let g = BondGeometry::new(
            0,
            1,
            Vec3::new(0.1, 0.2, -0.3),
            Vec3::new(0.9, -0.1, 0.4),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.2, 0.1).normalized(),
            Vec3::new(1.0, 0.2, 0.1).normalized().any_perpendicular(),
            0.04,
            [0.3, 0.1],
        );
        let q = Local6 { lin: Vec3::new(1.0, -2.0, 3.0), ang: Vec3::new(0.5, 0.25, -0.75) };
        let (fa, ma, fb, mb) = g.chunk_loads(&q);
        let xa = Vec3::new(0.1, 0.2, -0.3);
        let xb = Vec3::new(0.9, -0.1, 0.4);
        assert!((fa + fb).norm() < 1e-14);
        let total_moment = ma + mb + xa.cross(fa) + xb.cross(fb);
        assert!(total_moment.norm() < 1e-13, "{total_moment:?}");
    }

    fn skew_bond() -> (BondGeometry, Vec3, Vec3) {
        let (xa, xb) = (Vec3::new(0.1, 0.2, -0.3), Vec3::new(0.9, -0.1, 0.4));
        let n = Vec3::new(1.0, 0.2, 0.1).normalized();
        (BondGeometry::new(0, 1, xa, xb, Vec3::new(0.5, 0.05, 0.02), n, n.any_perpendicular(), 0.04, [0.3, 0.1]), xa, xb)
    }

    /// A finite rigid rotation (and translation) of both chunks deforms the co-rotational
    /// bond by nothing (round-off); the small-strain map reads it as deformation.
    #[test]
    fn corotation_ignores_a_finite_rigid_rotation() {
        let (g, xa, xb) = skew_bond();
        let th = Vec3::new(0.3, -0.2, 0.25);
        let r = rotation(th);
        let shift = Vec3::new(0.01, 0.02, -0.03);
        let (ua, ub) = (r * xa - xa + shift, r * xb - xb + shift);
        let d = g.corotation(ua, th, ub, th).deformation();
        let lin = g.kinematics(ua, th, ub, th);
        println!("co-rotational {d:?}, small strain {lin:?}");
        assert!(d.lin.norm() < 1e-15 && d.ang.norm() < 1e-15, "{d:?}");
        assert!(lin.lin.norm() > 1e-3);
    }

    /// `variation` is the derivative of `deformation` (central differences), the loads
    /// are its transpose (virtual work), they balance forces and, about the deformed
    /// chunk centres, moments, and the map is `kinematics` to first order.
    #[test]
    fn corotation_variation_is_the_derivative_and_loads_its_transpose() {
        let (g, xa, xb) = skew_bond();
        let (ua, tha) = (Vec3::new(1e-3, -2e-3, 5e-4), Vec3::new(0.02, -0.05, 0.03));
        let (ub, thb) = (Vec3::new(-3e-3, 1e-3, 2e-3), Vec3::new(-0.04, 0.01, 0.06));
        let cr = g.corotation(ua, tha, ub, thb);
        let dirs = [Vec3::new(0.3, -0.7, 0.2), Vec3::new(-0.5, 0.1, 0.9), Vec3::new(0.2, 0.4, -0.6), Vec3::new(0.8, -0.3, -0.1)];
        let h = 1e-6;
        let at = |s: f64| g.corotation(ua + dirs[0] * s, tha + dirs[1] * s, ub + dirs[2] * s, thb + dirs[3] * s).deformation().to_array();
        let (p, m) = (at(h), at(-h));
        let fd: Vec<f64> = (0..6).map(|i| (p[i] - m[i]) / (2.0 * h)).collect();
        // A rotation-vector increment `dth` turns the chunk by `R J_r(th) dth` (spatial).
        let (wa, wb) = (rotation(tha) * (right_jacobian(tha) * dirs[1]), rotation(thb) * (right_jacobian(thb) * dirs[3]));
        let an = cr.variation(dirs[0], wa, dirs[2], wb).to_array();
        for i in 0..6 {
            assert!((fd[i] - an[i]).abs() < 1e-8 * (1.0 + an[i].abs()), "component {i}: {} vs {}", fd[i], an[i]);
        }
        let q = Local6 { lin: Vec3::new(1.0, -2.0, 3.0), ang: Vec3::new(0.5, 0.25, -0.75) };
        let (fa, ma, fb, mb) = cr.loads(&q);
        let work = -(fa.dot(dirs[0]) + ma.dot(wa) + fb.dot(dirs[2]) + mb.dot(wb));
        assert!((work - q.dot(&cr.variation(dirs[0], wa, dirs[2], wb))).abs() < 1e-12);
        assert!((fa + fb).norm() < 1e-14);
        // A rigid virtual rotation deforms nothing, so the loads balance moments about
        // the deformed chunk centres.
        let w = Vec3::new(0.4, -0.3, 0.7);
        let v = cr.variation(w.cross(xa + ua), w, w.cross(xb + ub), w);
        assert!(v.lin.norm() < 1e-14 && v.ang.norm() < 1e-14, "{v:?}");
        let moment = (xa + ua).cross(fa) + ma + (xb + ub).cross(fb) + mb;
        assert!(moment.norm() < 1e-13, "{moment:?}");
        let small = g.corotation(ua * 1e-4, tha * 1e-4, ub * 1e-4, thb * 1e-4).deformation();
        let lin = g.kinematics(ua * 1e-4, tha * 1e-4, ub * 1e-4, thb * 1e-4);
        // Second order: relative to the deformation, of the order of the rotations (6e-6).
        assert!((small.lin - lin.lin).norm() < 1e-4 * lin.lin.norm() && (small.ang - lin.ang).norm() < 1e-4 * lin.ang.norm(), "{small:?} vs {lin:?}");
    }

    #[test]
    fn square_torsion_constant_matches_handbook() {
        let g = unit_bond();
        // Square side a: J = 0.1406 a^4, tau_max = T / (0.208 a^3).
        assert!((g.torsion_constant / 0.2f64.powi(4) - 0.1406).abs() < 1e-3);
        assert!((g.torsion_modulus / 0.2f64.powi(3) - 0.208).abs() < 1e-3);
    }
}
