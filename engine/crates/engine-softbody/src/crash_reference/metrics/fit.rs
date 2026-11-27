//! A body's orientation: the mass-weighted best-fit rigid rotation of its nodes from the
//! first sample (Horn's quaternion method, robust to nodes that have flown off), and the
//! yaw, pitch and roll of that rotation in the body's own frame.

use glam::{DMat3, DQuat, DVec3, Vec3};

/// Eigenvector of the largest eigenvalue of a symmetric 4×4 matrix (cyclic Jacobi).
fn top_eigenvector(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, r) in v.iter_mut().enumerate() {
        r[i] = 1.0;
    }
    for _ in 0..50 {
        let off: f64 = (0..4).flat_map(|p| (0..4).filter(move |&q| q != p).map(move |q| (p, q))).map(|(p, q)| a[p][q] * a[p][q]).sum();
        let scale: f64 = (0..4).map(|p| a[p][p] * a[p][p]).sum::<f64>().max(1e-300);
        if off <= 1e-24 * scale {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for r in &mut v {
                    let (rp, rq) = (r[p], r[q]);
                    r[p] = c * rp - s * rq;
                    r[q] = s * rp + c * rq;
                }
            }
        }
    }
    let best = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap_or(0);
    [v[0][best], v[1][best], v[2][best], v[3][best]]
}

/// The rotation R and translation t minimising Σ wᵢ |R pᵢ + t − qᵢ|² over the nodes with
/// `keep` (Horn 1987: the quaternion is the top eigenvector of his 4×4 matrix).
pub fn weighted_fit(p: &[Vec3], q: &[Vec3], w: &[f32], keep: &[bool]) -> (DMat3, DVec3) {
    let (mut cp, mut cq, mut sw) = (DVec3::ZERO, DVec3::ZERO, 0.0);
    for i in 0..p.len() {
        if keep[i] {
            let wi = f64::from(w[i]);
            cp += wi * p[i].as_dvec3();
            cq += wi * q[i].as_dvec3();
            sw += wi;
        }
    }
    if sw <= 0.0 {
        return (DMat3::IDENTITY, DVec3::ZERO);
    }
    cp /= sw;
    cq /= sw;
    let mut s = [[0.0f64; 3]; 3];
    for i in 0..p.len() {
        if keep[i] {
            let wi = f64::from(w[i]);
            let (a, b) = ((p[i].as_dvec3() - cp).to_array(), (q[i].as_dvec3() - cq).to_array());
            for (r, ar) in s.iter_mut().zip(a) {
                for (x, bc) in r.iter_mut().zip(b) {
                    *x += wi * ar * bc;
                }
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let [qw, qx, qy, qz] = top_eigenvector(n);
    let r = DMat3::from_quat(DQuat::from_xyzw(qx, qy, qz, qw).normalize());
    (r, cq - r * cp)
}

/// [`weighted_fit`] made robust: refitted twice on the nodes within max(4 × the median
/// residual, 5 cm) of the previous fit, so debris and crushed metal do not turn the body.
pub fn robust_fit(p: &[Vec3], q: &[Vec3], w: &[f32]) -> (DMat3, DVec3) {
    let mut keep = vec![true; p.len()];
    let mut fit = weighted_fit(p, q, w, &keep);
    for _ in 0..2 {
        let (r, t) = fit;
        let res: Vec<f64> = p.iter().zip(q).map(|(a, b)| (r * a.as_dvec3() + t - b.as_dvec3()).length()).collect();
        let mut sorted = res.clone();
        sorted.sort_by(f64::total_cmp);
        let Some(&median) = sorted.get(sorted.len() / 2) else { break };
        let cut = (4.0 * median).max(0.05);
        keep = res.iter().map(|&x| x <= cut).collect();
        if keep.iter().filter(|&&k| k).count() < 3 {
            break;
        }
        fit = weighted_fit(p, q, w, &keep);
    }
    fit
}

/// Yaw, pitch and roll (rad) of a world rotation `r` in a body frame whose forward is
/// `fwd` (horizontal) and up +Y: R_body = Ry(yaw) Rx(pitch) Rz(roll) with the body's
/// axes left (+X), up (+Y), forward (+Z). Yaw is positive when forward turns toward left
/// (from +Z toward +X at yaw 0), pitch positive nose down, roll positive left side up.
pub fn yaw_pitch_roll(r: DMat3, fwd: Vec3) -> (f64, f64, f64) {
    let f = fwd.as_dvec3().normalize();
    let up = DVec3::Y;
    let left = up.cross(f);
    let b = DMat3::from_cols(left, up, f);
    let rb = b.transpose() * r * b;
    let (fx, fy, fz) = (rb.z_axis.x, rb.z_axis.y, rb.z_axis.z);
    (fx.atan2(fz), (-fy).clamp(-1.0, 1.0).asin(), rb.x_axis.y.atan2(rb.y_axis.y))
}
