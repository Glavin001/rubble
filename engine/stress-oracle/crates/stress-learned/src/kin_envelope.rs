//! The energy envelope of a call (planning's hard decode constraint; P2's Python decode and
//! the host compute it identically): no node of the island can move further in the frame than
//! its share of the island's mechanical energy allows. With `E` the island's energy at the
//! frame's start and `E_b = (1 + TOL) E`:
//!
//! - displacement: `|dx| <= dt (sqrt(2 E_b / m) + |g| dt)`;
//! - rotation: `|drot| <= dt sqrt(2 E_b / I_min)`, `I_min` the smallest body-frame moment.
//!
//! Energy conservation, no smoothness assumed: the oracle's own labels never leave it (0 of
//! 1.08 M chunk and 7193 impactor nodes on 9875 calls of P1's calls-v5, closest 0.97).
//!
//! `E`, in f64 from the call's arrays as stored, summed in this order:
//! 1. every chunk row (anchored ones included), then every impactor row:
//!    `0.5 m (v.v) + 0.5 w.(R diag(I_body) R^T w)`, `R` the rotation of the row's world
//!    quaternion (`*_q_world`, wxyz) divided by its norm;
//! 2. then every `fs_bond_in` row's `stored` column (the manifest's `full_state_columns.bond`),
//!    non-finite values counted as 0.
//!
//! A free node (not anchored) whose decoded `dx` or `drot` exceeds its radius is projected
//! radially onto it (`d * (r / |d|)`), counted as one admissibility clamp each; anchored chunks
//! are not constrained. Applied after the decode's physical units and before its momentum
//! projection.

use serde_json::Value;

use crate::outcome_calls::{Arr, Call};

pub const TOL: f64 = 0.05;

/// The envelope of one call: `energy` (E, before the tolerance) and per node (chunk rows, then
/// impactor rows) its displacement and rotation radii; anchored chunks have no radius (None).
#[derive(Clone, Debug, serde::Serialize)]
pub struct Envelope {
    pub energy: f64,
    pub chunk_radii: Vec<Option<(f64, f64)>>,
    pub impactor_radii: Vec<(f64, f64)>,
}

fn f64s(a: &Arr) -> Vec<f64> {
    match a {
        Arr::F32(_, d) => d.iter().map(|&x| x as f64).collect(),
        Arr::F64(_, d) => d.clone(),
        Arr::I32(_, d) => d.iter().map(|&x| x as f64).collect(),
        Arr::I64(_, d) => d.iter().map(|&x| x as f64).collect(),
    }
}

fn get(c: &Call, name: &str) -> Result<Vec<f64>, String> {
    c.array(name).map(f64s).ok_or_else(|| format!("call lacks {name}"))
}

/// `0.5 m (v.v) + 0.5 w.(R diag(I) R^T w)`, with exactly these scalar expressions and their
/// order (bit-for-bit parity needs the same in Python: no BLAS matrix products, whose
/// summation order is not fixed).
fn ke(m: f64, i: [f64; 3], q: [f64; 4], v: [f64; 3], w: [f64; 3]) -> f64 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (qw, qx, qy, qz) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    let r = [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ];
    // Iw = R diag(I) R^T
    let mut iw = [[0.0f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            iw[a][b] = r[a][0] * i[0] * r[b][0] + r[a][1] * i[1] * r[b][1] + r[a][2] * i[2] * r[b][2];
        }
    }
    let iww: [f64; 3] = std::array::from_fn(|a| iw[a][0] * w[0] + iw[a][1] * w[1] + iw[a][2] * w[2]);
    0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 0.5 * (w[0] * iww[0] + w[1] * iww[1] + w[2] * iww[2])
}

/// The envelope of `call` (a `/5` call: `fs_bond_in` carries the bonds' stored energy).
pub fn envelope(manifest: &Value, call: &Call) -> Result<Envelope, String> {
    let dt = *call.scales.get("dt").ok_or("call lacks scales.dt")?;
    let g: f64 = manifest["gravity"].as_array().map_or(0.0, |a| a.iter().map(|x| x.as_f64().unwrap_or(0.0).powi(2)).sum::<f64>().sqrt());
    let stored_col = manifest["full_state_columns"]["bond"].as_array().and_then(|c| c.iter().position(|x| x == "stored")).ok_or("no stored column (a /5 call)")?;
    let n_bond_cols = manifest["full_state_columns"]["bond"].as_array().map_or(0, |c| c.len());
    let mut e = 0.0f64;
    let mut rows = Vec::new();
    for (pre, anchored) in [("chunk", true), ("impactor", false)] {
        let m = get(call, &format!("{pre}_mass"))?;
        let i = get(call, &format!("{pre}_inertia_body"))?;
        let q = get(call, &format!("{pre}_q_world"))?;
        let v = get(call, &format!("{pre}_v"))?;
        let w = get(call, &format!("{pre}_w"))?;
        let anch = if anchored { get(call, "chunk_anchored")? } else { vec![0.0; m.len()] };
        for k in 0..m.len() {
            let i3 = [i[3 * k], i[3 * k + 1], i[3 * k + 2]];
            e += ke(m[k], i3, [q[4 * k], q[4 * k + 1], q[4 * k + 2], q[4 * k + 3]], [v[3 * k], v[3 * k + 1], v[3 * k + 2]], [w[3 * k], w[3 * k + 1], w[3 * k + 2]]);
            rows.push((pre, m[k], i3[0].min(i3[1]).min(i3[2]), anch[k] != 0.0));
        }
    }
    let fs = get(call, "fs_bond_in")?;
    if n_bond_cols > 0 {
        for r in 0..fs.len() / n_bond_cols {
            let x = fs[r * n_bond_cols + stored_col];
            if x.is_finite() {
                e += x;
            }
        }
    }
    let eb = (1.0 + TOL) * e;
    let (mut chunk_radii, mut impactor_radii) = (Vec::new(), Vec::new());
    for (pre, m, imin, anchored) in rows {
        let radii = (dt * ((2.0 * eb / m).sqrt() + g * dt), dt * (2.0 * eb / imin).sqrt());
        if pre == "chunk" {
            chunk_radii.push((!anchored).then_some(radii));
        } else {
            impactor_radii.push(radii);
        }
    }
    Ok(Envelope { energy: e, chunk_radii, impactor_radii })
}

/// Project `d` radially onto the ball of radius `r`; whether it was clamped.
pub fn clamp(d: &mut [f64; 3], r: f64) -> bool {
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n > r {
        let s = r / n;
        d.iter_mut().for_each(|x| *x *= s);
        true
    } else {
        false
    }
}
