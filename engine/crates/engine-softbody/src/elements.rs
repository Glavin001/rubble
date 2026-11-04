//! Multi-node elements beyond two-node beams (BeamNG-parity plan, phase 3): the
//! energies, gradients and positive semi-definite 3×3 block approximations the
//! particle block solve needs, and their plasticity. `soft_step.slang` mirrors
//! them; the explicit oracle uses them directly (forces = −gradient).
//!
//! * **LBEAM** (three nodes a, m, b): resists change of the angle at m between
//!   m→a and m→b: E = ½k(θ − θ₀)². High lateral stiffness, none along the arms.
//! * **Torsion bar** (four nodes 1–2–3–4): resists twist about the 2–3 axis, the
//!   dihedral angle between planes (1, 2, 3) and (2, 3, 4): E = ½k(φ − φ₀)².
//! * **Slidenode on a rail**: a node held to a polyline of nodes, free along it:
//!   E = ½k·dist⊥²; capped rails also hold it at the ends.
//!
//! Plasticity follows the beams' rule in angle: past the deform moment the rest
//! angle moves so the element sits on it; past the strength it breaks.

use glam::{DMat3, DVec3};

/// Angle θ ∈ [0, π] at `m` between `a − m` and `b − m`, and its gradient with
/// respect to a, m, b. The gradient is scaled by (θ − θ₀) robustly: where the
/// arms are collinear and θ₀ is too, the product stays finite (it vanishes).
pub fn angle(a: DVec3, m: DVec3, b: DVec3) -> f64 {
    let (u, v) = (a - m, b - m);
    u.cross(v).length().atan2(u.dot(v))
}

/// k(θ − θ₀)·∂θ/∂(a, m, b): the LBEAM's energy gradient.
pub fn lbeam_gradient(a: DVec3, m: DVec3, b: DVec3, theta0: f64, k: f64) -> [DVec3; 3] {
    let (u, v) = (a - m, b - m);
    let (lu, lv) = (u.length(), v.length());
    if lu < 1e-9 || lv < 1e-9 {
        return [DVec3::ZERO; 3];
    }
    let cross = u.cross(v).length();
    let theta = cross.atan2(u.dot(v));
    let t = theta - theta0;
    // ∂θ/∂u = −(v − (u·v/|u|²)u)/|u×v|; (θ − θ₀)/|u×v| tends to −1/(|u||v|) as the
    // arms straighten with θ₀ = π, and the direction terms vanish with it.
    let ratio = if cross > 1e-12 * lu * lv {
        t / cross
    } else if (theta0 - std::f64::consts::PI).abs() < 1e-6 {
        -1.0 / (lu * lv)
    } else {
        return [DVec3::ZERO; 3];
    };
    let ga = -(v - u * (u.dot(v) / (lu * lu))) * (k * ratio);
    let gb = -(u - v * (u.dot(v) / (lv * lv))) * (k * ratio);
    [ga, -(ga + gb), gb]
}

/// PSD 3×3 blocks of the LBEAM for a, m, b: each arm's lateral stiffness
/// k/|arm|² across it (an upper bound of the in-plane curvature that stays
/// positive where the arms are straight), the middle node both arms'.
pub fn lbeam_blocks(a: DVec3, m: DVec3, b: DVec3, k: f64) -> [DMat3; 3] {
    let lateral = |d: DVec3| {
        let l2 = d.length_squared().max(1e-12);
        let n = d / l2.sqrt();
        (DMat3::IDENTITY - outer(n, n)) * (k / l2)
    };
    let (ha, hb) = (lateral(a - m), lateral(b - m));
    // ∂θ/∂m = −(∂θ/∂a + ∂θ/∂b): its curvature is bounded by twice the sum.
    [ha, (ha + hb) * 2.0, hb]
}

fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Dihedral angle φ ∈ (−π, π] of 1–2–3–4 about the 2→3 axis.
pub fn dihedral(x: [DVec3; 4]) -> f64 {
    let (b1, b2, b3) = (x[1] - x[0], x[2] - x[1], x[3] - x[2]);
    let (n1, n2) = (b1.cross(b2), b2.cross(b3));
    (b2.length() * b1.dot(n2)).atan2(n1.dot(n2))
}

/// Wraps an angle difference into (−π, π].
pub fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let mut a = a % t;
    if a > std::f64::consts::PI {
        a -= t;
    } else if a <= -std::f64::consts::PI {
        a += t;
    }
    a
}

/// ∂φ/∂x for the four nodes (Blondel & Karplus 1996); zero where a plane is
/// degenerate.
pub fn dihedral_jacobian(x: [DVec3; 4]) -> [DVec3; 4] {
    let (b1, b2, b3) = (x[1] - x[0], x[2] - x[1], x[3] - x[2]);
    let (n1, n2) = (b1.cross(b2), b2.cross(b3));
    let (l1, l2, lb) = (n1.length_squared(), n2.length_squared(), b2.length());
    if l1 < 1e-18 || l2 < 1e-18 || lb < 1e-9 {
        return [DVec3::ZERO; 4];
    }
    let g1 = -n1 * (lb / l1);
    let g4 = n2 * (lb / l2);
    // Projections of the outer arms on the axis, from its own ends (2→1 on 2→3,
    // 3→4 on 3→2).
    let (p, q) = (-b1.dot(b2) / (lb * lb), -b3.dot(b2) / (lb * lb));
    let g2 = g1 * (p - 1.0) - g4 * q;
    let g3 = g4 * (q - 1.0) - g1 * p;
    [g1, g2, g3, g4]
}

/// Torsion bar energy gradient k·wrap(φ − φ₀)·∂φ/∂x and Gauss–Newton blocks
/// k·∂φ/∂xᵢ ∂φ/∂xᵢᵀ (PSD).
pub fn torsion_terms(x: [DVec3; 4], phi0: f64, k: f64) -> ([DVec3; 4], [DMat3; 4]) {
    let j = dihedral_jacobian(x);
    let t = wrap(dihedral(x) - phi0);
    (j.map(|g| g * (k * t)), j.map(|g| outer(g, g) * k))
}

/// Slidenode on a polyline rail: the closest point and the unit tangent there,
/// and the segment and its parameter (for the reaction on the rail's nodes).
/// `capped`: the node is also held at the rail's ends (else it may run off them,
/// and is then free along the end segment's line).
pub fn rail_closest(p: DVec3, rail: &[DVec3], looped: bool) -> (DVec3, DVec3, usize, f64) {
    let segs = if looped { rail.len() } else { rail.len() - 1 };
    let mut best = (DVec3::ZERO, DVec3::X, 0, 0.0, f64::MAX);
    for s in 0..segs {
        let (a, b) = (rail[s], rail[(s + 1) % rail.len()]);
        let ab = b - a;
        let l2 = ab.length_squared().max(1e-18);
        let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
        let q = a + ab * t;
        let d = (p - q).length_squared();
        if d < best.4 {
            best = (q, ab / l2.sqrt(), s, t, d);
        }
    }
    (best.0, best.1, best.2, best.3)
}

/// The slidenode's gradient k·d⊥ (d = p − closest point; along the rail only
/// past a capped end) and its block k(I − t̂t̂ᵀ) (kI at a capped end).
pub fn slide_terms(p: DVec3, q: DVec3, t: DVec3, at_cap: bool, k: f64) -> (DVec3, DMat3) {
    let d = p - q;
    if at_cap {
        return (d * k, DMat3::IDENTITY * k);
    }
    let perp = d - t * d.dot(t);
    (perp * k, (DMat3::IDENTITY - outer(t, t)) * k)
}

/// Rest-angle plasticity: past `deform` (a moment, N·m) the rest angle moves so
/// the element sits on its threshold, which hardens by the angle moved; past
/// `strength` it breaks. Returns (yielded, broke).
pub fn angular_plastic(
    rest: &mut f64,
    deform: &mut f64,
    strength: f64,
    k: f64,
    error: f64,
) -> (bool, bool) {
    let moment = k * error;
    let mut out = moment.abs();
    let mut yielded = false;
    if moment.abs() > *deform {
        let excess = moment.abs() - *deform;
        *rest += error.signum() * excess / k;
        out = *deform;
        yielded = true;
    }
    (yielded, out > strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd<const N: usize>(x: [DVec3; N], e: impl Fn([DVec3; N]) -> f64) -> [DVec3; N] {
        let h = 1e-6;
        let mut g = [DVec3::ZERO; N];
        for i in 0..N {
            for c in 0..3 {
                let (mut p, mut m) = (x, x);
                p[i][c] += h;
                m[i][c] -= h;
                g[i][c] = (e(p) - e(m)) / (2.0 * h);
            }
        }
        g
    }

    #[test]
    fn lbeam_gradient_matches_finite_difference() {
        let x = [
            DVec3::new(1.0, 0.2, -0.1),
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(-0.3, 0.9, 0.4),
        ];
        let (k, th0) = (500.0, 1.2);
        let e = |x: [DVec3; 3]| 0.5 * k * (angle(x[0], x[1], x[2]) - th0).powi(2);
        let g = lbeam_gradient(x[0], x[1], x[2], th0, k);
        let f = fd(x, e);
        for i in 0..3 {
            assert!(
                (g[i] - f[i]).length() < 1e-5 * f[i].length().max(1.0),
                "{i}: {:?} vs {:?}",
                g[i],
                f[i]
            );
        }
        // Straight arms with θ₀ = π: no force, and a finite one just off it.
        let s = [DVec3::X, DVec3::ZERO, -DVec3::X];
        let g = lbeam_gradient(s[0], s[1], s[2], std::f64::consts::PI, k);
        assert!(g.iter().all(|v| v.length() < 1e-9));
        let mut bent = s;
        bent[1].y = 1e-4;
        let g = lbeam_gradient(bent[0], bent[1], bent[2], std::f64::consts::PI, k);
        let f = fd(bent, |x| {
            0.5 * k * (angle(x[0], x[1], x[2]) - std::f64::consts::PI).powi(2)
        });
        assert!(
            (g[1] - f[1]).length() < 1e-3 * f[1].length().max(1e-3),
            "{:?} vs {:?}",
            g[1],
            f[1]
        );
    }

    #[test]
    fn dihedral_jacobian_matches_finite_difference() {
        let x = [
            DVec3::new(1.0, 0.1, 0.3),
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.1, 1.0, 0.0),
            DVec3::new(-0.8, 1.2, 0.5),
        ];
        let j = dihedral_jacobian(x);
        let f = fd(x, dihedral);
        for i in 0..4 {
            assert!(
                (j[i] - f[i]).length() < 1e-5,
                "{i}: {:?} vs {:?}",
                j[i],
                f[i]
            );
        }
        // Translation and rotation leave φ alone: the Jacobian sums to zero.
        assert!(j.iter().sum::<DVec3>().length() < 1e-9);
    }

    #[test]
    fn blocks_are_psd() {
        let x = [
            DVec3::new(1.0, 0.1, 0.3),
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.1, 1.0, 0.0),
            DVec3::new(-0.8, 1.2, 0.5),
        ];
        let (_, h) = torsion_terms(x, 0.3, 200.0);
        let lb = lbeam_blocks(x[0], x[1], x[2], 300.0);
        for m in h.iter().chain(lb.iter()) {
            for v in [
                DVec3::X,
                DVec3::Y,
                DVec3::Z,
                DVec3::ONE.normalize(),
                DVec3::new(1.0, -2.0, 0.5),
            ] {
                assert!(v.dot(*m * v) >= -1e-9);
            }
        }
    }

    #[test]
    fn slidenode_is_free_along_the_rail() {
        let rail = [DVec3::ZERO, DVec3::X, DVec3::new(2.0, 1.0, 0.0)];
        let p = DVec3::new(0.4, 0.3, -0.2);
        let (q, t, s, u) = rail_closest(p, &rail, false);
        assert_eq!(s, 0);
        assert!((u - 0.4).abs() < 1e-12 && (q - DVec3::new(0.4, 0.0, 0.0)).length() < 1e-12);
        let (g, h) = slide_terms(p, q, t, false, 1000.0);
        assert!(g.dot(t).abs() < 1e-9, "no force along the rail");
        assert!((h * t).length() < 1e-9, "no stiffness along the rail");
        let e = |x: [DVec3; 1]| {
            let (q, t, _, _) = rail_closest(x[0], &rail, false);
            let d = x[0] - q;
            0.5 * 1000.0 * (d - t * d.dot(t)).length_squared()
        };
        let f = fd([p], e);
        assert!((g - f[0]).length() < 1e-4);
    }

    #[test]
    fn angular_plasticity_moves_the_rest_angle() {
        let (mut rest, mut deform) = (0.0, 10.0);
        let (y, b) = angular_plastic(&mut rest, &mut deform, 100.0, 100.0, 0.3);
        assert!(y && !b);
        // The element now carries exactly its threshold: k(0.3 − rest) = 10.
        assert!((100.0 * (0.3 - rest) - 10.0).abs() < 1e-9);
        let (_, b) = angular_plastic(&mut rest, &mut 1e9, 50.0, 100.0, 1.0);
        assert!(b);
    }
}

/// Small original JBeam fixtures for the element tests (oracle and GPU).
pub mod fixtures {
    /// A three-node arm sticking out from two fixed nodes, with (or without) an
    /// LBEAM at its middle node.
    pub fn lbeam_arm(lbeam: bool) -> String {
        format!(
            r#"{{"arm": {{"slotType": "main",
            "nodes": [["id", "posX", "posY", "posZ"], {{"nodeWeight": 2}},
                ["r1", 0, 0, 1, {{"fixed": true}}], ["r2", 0, 0, 1.2, {{"fixed": true}}],
                ["m", 0, -0.5, 1], ["t", 0, -1.0, 1]],
            "beams": [["id1:", "id2:"], {{"beamSpring": 2000000, "beamDamp": 200}},
                ["r1", "m"], ["r2", "m"], ["m", "t"]{}]
            }}}}"#,
            if lbeam {
                r#", {"beamType": "|LBEAM", "beamSpring": 20000, "beamDamp": 50}, ["r1", "m", "t"]"#
            } else {
                ""
            }
        )
    }

    /// A free node held by a torsion bar about the fixed b–c axis (1 m arm).
    pub const TORSION_BAR: &str = r#"{"tb": {"slotType": "main",
        "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1},
            ["a", 1, 0, 0, {"fixed": true}], ["b", 0, 0, 0, {"fixed": true}],
            ["c", 0, 1, 0, {"fixed": true}], ["d", 1, 1, 0]],
        "beams": [["id1:", "id2:"], {"beamSpring": 1000000, "beamDamp": 100}, ["c", "d"], ["b", "d"]],
        "torsionbars": [["id1:", "id2:", "id3:", "id4:"], {"spring": 500, "damp": 5}, ["a", "b", "c", "d"]]
    }}"#;

    /// A slidenode on a sloped rail of fixed nodes (capped or not), with damping.
    pub fn slide_rail(capped: bool, damp: f32) -> String {
        format!(
            r#"{{"rail": {{"slotType": "main",
            "nodes": [["id", "posX", "posY", "posZ"], {{"nodeWeight": 1}},
                ["r0", 0, 0, 1, {{"fixed": true}}], ["r1", 0, -1, 0.5, {{"fixed": true}}],
                ["r2", 0, -2, 0, {{"fixed": true}}], ["s", 0, -0.2, 0.9]],
            "rails": {{"slope": {{"links:": ["r0", "r1", "r2"], "looped": false, "capped": {capped}}}}},
            "slidenodes": [["id:", "railName", "attached", "fixToRail", "tolerance", "spring", "damp"],
                ["s", "slope", true, true, 0, 100000, {damp}]]
            }}}}"#
        )
    }
}
