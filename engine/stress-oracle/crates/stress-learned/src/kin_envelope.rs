//! The energy envelope of a call (planning's hard decode constraint; P2's Python decode and
//! the host compute it identically): no node of the island can move further in the frame than
//! its share of the island's mechanical energy allows. With `E` the island's energy at the
//! frame's start and `E_b = (1 + TOL) E`:
//!
//! - displacement: `|dx| <= dt (sqrt(2 E_b / m) + |g| dt)`;
//! - rotation: `|drot| <= dt sqrt(2 E_b / I_min)`, `I_min` the smallest body-frame moment.
//!
//! Agreed with P2 (`decode.energy_envelope`, 139238e3): `E` is the call's own frame-start
//! energy (`energy.KE_in + energy.E_elastic_start`, the builder's), `TOL` 2.0 (on all of
//! P1's calls-v5, 40,713 calls, +5% excluded two oracle frames: a ram struck by a ram outside
//! its island, energy the call does not see; +200% excludes none), the clamp applied after the
//! momentum projection (and before the energy bound). A call without `E_elastic_start` is
//! unbounded.
//!
//! A free node (not anchored) whose decoded `dx` or `drot` exceeds its radius is projected
//! radially onto it (`d * (r / |d|)`), counted as one admissibility clamp each; anchored chunks
//! are not constrained.

use serde_json::Value;

use crate::outcome_calls::{Arr, Call};

pub const TOL: f64 = 2.0;

/// The envelope of one call: `energy` (E, before the tolerance; None: unbounded, the call
/// carries no `E_elastic_start`) and per node (chunk rows, then impactor rows) its
/// displacement and rotation radii; anchored chunks have no radius (None).
#[derive(Clone, Debug, serde::Serialize)]
pub struct Envelope {
    pub energy: Option<f64>,
    pub chunk_radii: Vec<Option<(f64, f64)>>,
    pub impactor_radii: Vec<Option<(f64, f64)>>,
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

/// The envelope of `call` (agreed with P2's `decode.energy_envelope`):
/// `E = (energy.KE_in + energy.E_elastic_start) (1 + TOL)`, the call's own frame-start energy
/// scalars (the builder's); per free node `r_x = dt (sqrt(2 E / m) + |g| dt)` and
/// `r_r = dt sqrt(2 E / I_min)`, `I_min` the smallest of the row's `*_inertia_body`.
pub fn envelope(manifest: &Value, call: &Call) -> Result<Envelope, String> {
    let dt = *call.scales.get("dt").ok_or("call lacks scales.dt")?;
    let g: f64 = manifest["gravity"].as_array().map_or(0.0, |a| a.iter().map(|x| x.as_f64().unwrap_or(0.0).powi(2)).sum::<f64>().sqrt());
    let energy = match (call.energy.get("KE_in"), call.energy.get("E_elastic_start")) {
        (Some(k), Some(e)) => Some((k + e) * (1.0 + TOL)),
        _ => None,
    };
    let radii = |m: f64, imin: f64| energy.map(|e| (dt * ((2.0 * e / m).sqrt() + g * dt), dt * (2.0 * e / imin).sqrt()));
    let mut chunk_radii = Vec::new();
    let (m, i, anch) = (get(call, "chunk_mass")?, get(call, "chunk_inertia_body")?, get(call, "chunk_anchored")?);
    for k in 0..m.len() {
        chunk_radii.push(if anch[k] != 0.0 { None } else { radii(m[k], i[3 * k].min(i[3 * k + 1]).min(i[3 * k + 2])) });
    }
    let (m, i) = (get(call, "impactor_mass")?, get(call, "impactor_inertia_body")?);
    let impactor_radii = (0..m.len()).map(|k| radii(m[k], i[3 * k].min(i[3 * k + 1]).min(i[3 * k + 2]))).collect();
    Ok(Envelope { energy: energy.map(|e| e / (1.0 + TOL)), chunk_radii, impactor_radii })
}

/// Project `d` radially onto the ball of radius `r` when it lies outside `r (1 + 1e-9)`;
/// whether it was clamped.
pub fn clamp(d: &mut [f64; 3], r: f64) -> bool {
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n > r * (1.0 + 1e-9) {
        let s = r / n;
        d.iter_mut().for_each(|x| *x *= s);
        true
    } else {
        false
    }
}
