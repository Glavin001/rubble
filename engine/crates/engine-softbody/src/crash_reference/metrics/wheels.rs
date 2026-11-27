//! Energy stored in raycast wheels' suspension springs (`explicit_drive::drive_pass`),
//! which are forces, not beams: from a sample's node positions by the pass's own rule.
//! Per wheel a ray from the hardpoint (`rest_len` above the hub, along the vehicle's up
//! from its reference nodes) down to the ground or a static box top; suspension length
//! l = hit − radius; the spring pushes k (l_max − l) while l < l_max, a bump stop
//! k_b (l_min − l) below l_min, and each anti-roll bar k_ar (l_left − l_right) between
//! two wheels on the ground. Stored: ½ k (l_max − l)² + ½ k_b (l_min − l)² + ½ k_ar Δl².
//! Static shapes other than boxes, rigid proxies and other vehicles' triangles under a
//! wheel are not cast against (the crash scenes have none).

use crate::explicit_beams::ExplicitBeams;
use glam::Vec3;

/// Ray (from `p` along unit `d`) into the static box [lo, hi]: entry distance and the
/// entry face's normal, when it enters at t ≥ 0.
fn slab(p: Vec3, d: Vec3, lo: Vec3, hi: Vec3) -> Option<(f32, Vec3)> {
    let (mut t_in, mut t_out, mut n) = (f32::MIN, f32::MAX, Vec3::ZERO);
    for a in 0..3 {
        if d[a].abs() < 1e-12 {
            if p[a] < lo[a] || p[a] > hi[a] {
                return None;
            }
            continue;
        }
        let (t1, t2) = ((lo[a] - p[a]) / d[a], (hi[a] - p[a]) / d[a]);
        let (near, far, s) = if t1 < t2 { (t1, t2, -1.0) } else { (t2, t1, 1.0) };
        if near > t_in {
            t_in = near;
            n = Vec3::ZERO;
            n[a] = s;
        }
        t_out = t_out.min(far);
    }
    (t_in >= 0.0 && t_in <= t_out).then_some((t_in, n))
}

/// Suspension energy (J) of vehicle `s`'s raycast wheels with its nodes at `pos` (world).
pub fn suspension_energy(s: &ExplicitBeams, pos: &[Vec3]) -> f64 {
    let Some(d) = s.drive.as_deref() else { return 0.0 };
    let c = &d.config;
    let p = |i: u32| pos.get(i as usize).copied().unwrap_or(Vec3::ZERO);
    let m = |i: u32| s.nodes.get(i as usize).map_or(0.0, |n| n.mass);
    let [rc, rb, rl] = c.refs;
    let fw = (p(rc) - p(rb)).normalize_or_zero();
    let up = fw.cross(p(rl) - p(rc)).normalize_or_zero();
    if up == Vec3::ZERO {
        return 0.0;
    }
    let mut len = Vec::with_capacity(c.wheels.len());
    let mut e = 0.0f64;
    for w in &c.wheels {
        let (m1, m2) = (m(w.node1), if w.node1 == w.node2 { 0.0 } else { m(w.node2) });
        let hub = (p(w.node1) * m1 + p(w.node2) * m2) / (m1 + m2).max(1e-12);
        let hp = hub + up * w.rest_len;
        let mut hit: Option<(f32, Vec3)> = None;
        if up.y > 1e-12 && hp.y >= s.ground {
            hit = Some(((hp.y - s.ground) / up.y, Vec3::Y));
        }
        for b in &s.boxes {
            if let Some(h) = slab(hp, -up, b.lo, b.hi) {
                if hit.is_none_or(|x| h.0 < x.0) {
                    hit = Some(h);
                }
            }
        }
        let l = hit.filter(|(t, n)| *t <= w.susp_max + w.radius && n.dot(up) >= c.max_slope_cos).map(|(t, _)| t - w.radius);
        if let Some(l) = l {
            if l < w.susp_max {
                e += 0.5 * f64::from(w.k) * f64::from(w.susp_max - l).powi(2);
            }
            if l < w.susp_min && w.bump_k > 0.0 {
                e += 0.5 * f64::from(w.bump_k) * f64::from(w.susp_min - l).powi(2);
            }
        }
        len.push(l);
    }
    for &(a, b, k) in &c.anti_roll {
        if let (Some(Some(la)), Some(Some(lb))) = (len.get(a as usize), len.get(b as usize)) {
            e += 0.5 * f64::from(k) * f64::from(la - lb).powi(2);
        }
    }
    e
}
