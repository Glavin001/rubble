//! Experimental, causal material-state network. Native motion/contact integration
//! remains unchanged. Every feature is from the current state, never a future frame.
use crate::{joint_history, phase_history};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
use stress_ref::{
    bond::Local6,
    joint::{stress_measures, JointResponse},
    math::Vec3,
    solver::{ChunkLoads, ReferenceSolver},
    stress_provider::{BondJob, StressProvider, SuppliedStress},
};

pub const FORMAT: &str = "causal-material-state/2";
pub const NY: usize = 42;
pub const NG: usize = 29; // 21 changed histories, complete damage, two finite ductilities, five modes
pub fn augment(x: &[f64]) -> Vec<f64> {
    let mut out = x.to_vec();
    let t = (x[64] / (x[49] * x[59])).max(0.);
    let cap = if x[53] > 0. {
        x[53] * x[59]
    } else {
        f64::INFINITY
    };
    let shear = x[65] / (x[51] * x[59] + x[52] * x[66]).min(cap);
    let c = if x[50] > 0. {
        (x[67] / (x[50] * x[59])).max(0.)
    } else {
        0.
    };
    let lambda = t.max(shear);
    let psi = 0.5
        * (0..6)
            .filter(|&j| j != 2 || x[2] > 0.)
            .map(|j| x[33 + j] * x[j] * x[j])
            .sum::<f64>();
    let gf = if t >= shear { x[54] } else { x[55] };
    let budget = gf * x[60];
    out.extend([
        t,
        shear,
        c,
        lambda - x[14],
        c - x[15],
        psi / budget,
        if psi > 0. {
            budget * lambda * lambda / psi
        } else {
            0.
        },
        x[0] - x[19],
        x[1] - x[20],
        x[5] - x[24],
    ]);
    out
}
fn local(a: &[f64]) -> Local6 {
    Local6 {
        lin: Vec3::new(a[0], a[1], a[2]),
        ang: Vec3::new(a[3], a[4], a[5]),
    }
}
fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.
    }
}

/// Fixed physical decoder: force is the deformation derivative of stored energy
/// with the predicted material history held fixed. This does NOT evaluate the
/// oracle's strength, damage, fatigue, or plastic return-map update.
pub fn energy_force(k: Local6, width: [f64; 2], d: Local6, h: &[f64; 21]) -> (Local6, f64) {
    let intact = Local6 {
        lin: Vec3::new(
            k.lin.x * d.lin.x,
            k.lin.y * d.lin.y,
            k.lin.z * d.lin.z * if d.lin.z < 0. { 1. - h[1] } else { 1. },
        ),
        ang: k.ang.mul_elem(d.ang),
    }
    .scale(1. - h[0]);
    let mut force = intact;
    let mut energy = 0.5 * d.dot(&intact);
    if h[0] > 0. {
        let [area, x, y, ixx, iyy, ixy] =
            stress_ref::joint::compressed_region(width, d.lin.z, d.ang.x, d.ang.y);
        let ki = k.lin.z * (1. - h[1]) / (width[0] * width[1]);
        let z = d.lin.z + d.ang.x * y - d.ang.y * x;
        let mut contact = Local6 {
            lin: Vec3::new(0., 0., ki * area * z),
            ang: Vec3::new(
                ki * (area * z * y - d.ang.y * ixy + d.ang.x * iyy),
                -ki * (area * z * x - d.ang.y * ixx + d.ang.x * ixy),
                0.,
            ),
        };
        let mut patch = 0.5
            * ki
            * (area * z * z + d.ang.y * d.ang.y * ixx + d.ang.x * d.ang.x * iyy
                - 2. * d.ang.x * d.ang.y * ixy);
        contact.lin.x = k.lin.x * (d.lin.x - h[7]);
        contact.lin.y = k.lin.y * (d.lin.y - h[8]);
        contact.ang.z = k.ang.z * (d.ang.z - h[12]);
        patch += 0.5
            * (contact.lin.x.powi(2) / k.lin.x
                + contact.lin.y.powi(2) / k.lin.y
                + contact.ang.z.powi(2) / k.ang.z);
        force = force.add(&contact.scale(h[0]));
        energy += h[0] * patch;
    }
    (force, energy)
}

/// Baseline is a cheap elastic prior, not an oracle constitutive evaluation.
/// Infinity in unset ductility is represented by explicit presence bits.
pub fn features(
    s: &ReferenceSolver,
    job: BondJob,
    dt: f64,
    fracture: bool,
) -> Result<(Vec<f64>, [f64; NY], Local6, Local6), String> {
    let b = &s.bonds[job.0][job.1];
    let g = &b.geometry;
    let a = s.chunks[job.0][g.a];
    let c = s.chunks[job.0][g.b];
    if b.rebar.is_some() {
        return Err("causal material v1 has no rebar training contract".into());
    }
    let (d, rate) = if s.config.methods.corotational_bonds {
        let cr = g.corotation(a.u, a.th, c.u, c.th);
        (cr.deformation(), cr.variation(a.v, a.w, c.v, c.w))
    } else {
        (
            g.kinematics(a.u, a.th, c.u, c.th),
            g.kinematics(a.v, a.w, c.v, c.w),
        )
    };
    let k = b.stiffness.as_local();
    let q = d.mul_elem(&k);
    let m = stress_measures(g, &q);
    let h = joint_history::pack(&b.joint);
    let st = &b.strength;
    if st.dif.is_some()
        || st.static_fatigue.is_some()
        || st.buckling_load.is_some()
        || !st.softening
        || !st.crack_contact
        || !st.exact_patch
        || !st.patch_after_damage
        || !st.exact_rate_filter
        || st.kind != stress_ref::material::BondKind::Brittle
    {
        return Err("causal material experiment requires brittle softening with exact patch contact, no DIF/fatigue/buckling/rebar".into());
    }
    let mut x = Vec::new();
    x.extend(d.to_array());
    x.extend(rate.to_array());
    x.extend(h.map(finite));
    x.extend(k.to_array());
    x.extend(b.damping.to_array());
    x.extend([
        dt,
        fracture as u8 as f64,
        h[4].is_finite() as u8 as f64,
        h[5].is_finite() as u8 as f64,
        st.tensile,
        finite(st.compressive),
        st.cohesion,
        st.friction,
        finite(st.shear_cap),
        st.g_tension,
        st.g_shear,
        st.g_compression,
        st.youngs_modulus,
        st.rate_filter_time,
        b.weibull,
        g.area,
        g.length,
        g.width[0],
        g.width[1],
        m.tension,
        m.shear,
        m.normal_compression,
        m.compression,
        m.compressive_force,
    ]);
    let mut base = [0.; NY];
    base[..21].copy_from_slice(&h.map(finite));
    let factors = b.model().secant_factors(&b.joint, &d);
    let material = q.mul_elem(&Local6 {
        lin: Vec3::new(
            1. - h[0],
            1. - h[0],
            if d.lin.z < 0. { 1. - h[1] } else { 1. - h[0] },
        ),
        ang: Vec3::splat(1. - h[0]),
    });
    let damp = rate.mul_elem(&b.damping).mul_elem(&factors);
    base[21..27].copy_from_slice(&material.add(&damp).to_array());
    base[27..33].copy_from_slice(&material.to_array());
    base[33] = 0.5 * d.dot(&material);
    base[34..39].copy_from_slice(&[
        m.tension,
        m.shear,
        m.normal_compression,
        m.compression,
        m.compressive_force,
    ]);
    base[41] = damp.dot(&rate) * dt;
    Ok((x, base, d, rate))
}
fn target(r: &SuppliedStress) -> [f64; NY] {
    let mut y = [0.; NY];
    y[..21].copy_from_slice(&joint_history::pack(&r.joint.state).map(finite));
    y[21..39].copy_from_slice(&phase_history::response_values(r));
    y[39..].copy_from_slice(&[r.joint.dissipated, r.joint.overshoot, r.damping_dissipated]);
    y
}
fn gates(base: &[f64; NY], r: &SuppliedStress) -> [f64; NG] {
    let y = target(r);
    let mut out = [0.; NG];
    for j in 0..21 {
        out[j] = (y[j] != base[j]) as u8 as f64;
    }
    out[21] = (r.joint.state.damage >= 1.) as u8 as f64;
    out[22] = r.joint.state.ductility.is_finite() as u8 as f64;
    out[23] = r.joint.state.ductility_c.is_finite() as u8 as f64;
    out[24 + joint_history::mode_code(r.joint.state.mode) as usize] = 1.;
    out
}

pub struct Teacher {
    out: Option<BufWriter<File>>,
    index: Option<BufWriter<File>>,
    pub rows: u64,
    pub input_width: usize,
    pub calls: u64,
}
impl Teacher {
    pub fn new(p: &Path) -> Result<Self, String> {
        Ok(Self {
            out: Some(BufWriter::new(File::create(p).map_err(|e| e.to_string())?)),
            index: Some(BufWriter::new(
                File::create(p.with_extension("index.u64")).map_err(|e| e.to_string())?,
            )),
            rows: 0,
            input_width: 0,
            calls: 0,
        })
    }
    pub fn reference() -> Self {
        Self {
            out: None,
            index: None,
            rows: 0,
            input_width: 69,
            calls: 0,
        }
    }
    pub fn record(
        &mut self,
        s: &ReferenceSolver,
        job: BondJob,
        dt: f64,
        fracture: bool,
    ) -> Result<SuppliedStress, String> {
        let r = s.oracle_stress_response(job, dt, fracture);
        if let Some(out) = &mut self.out {
            let (x, base, _, _) = features(s, job, dt, fracture)?;
            self.input_width = x.len();
            let y = target(&r);
            let tags = gates(&base, &r);
            for v in x.iter().chain(&base).chain(&y).chain(&tags) {
                out.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        if let Some(index) = &mut self.index {
            for value in [s.substeps, job.0 as u64, job.1 as u64] {
                index
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
        self.rows += 1;
        Ok(r)
    }
    pub fn flush(&mut self) -> Result<(), String> {
        for out in [&mut self.out, &mut self.index].into_iter().flatten() {
            out.flush().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
impl StressProvider for Teacher {
    fn responses(
        &mut self,
        s: &ReferenceSolver,
        _: &ChunkLoads,
        jobs: &[BondJob],
        dt: f64,
        fracture: bool,
    ) -> Result<Vec<SuppliedStress>, String> {
        let mut out = Vec::new();
        for &job in jobs {
            out.push(self.record(s, job, dt, fracture)?);
        }
        self.calls += 1;
        Ok(out)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
pub struct Model {
    pub format: String,
    pub trained_steps: u64,
    /// Explicit numerical experiment; legacy artifacts use the f64 harness.
    #[serde(default)]
    pub float32_inference: bool,
    pub input_scale: Vec<f64>,
    pub input_mean: Vec<f64>,
    pub input_std: Vec<f64>,
    pub output_scale: Vec<f64>,
    pub layers: Vec<Layer>,
}
impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let m: Self = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if ![
            FORMAT,
            "causal-material-state/3",
            "causal-material-state/4",
            "causal-material-state/5",
        ]
        .contains(&m.format.as_str())
            || m.trained_steps == 0
            || m.output_scale.len() != NY
            || m.layers.is_empty()
        {
            return Err("invalid or untrained causal model".into());
        }
        let n = m.input_scale.len();
        if m.input_mean.len() != n || m.input_std.len() != n {
            return Err("normalizer shape".into());
        }
        if m.input_mean.iter().any(|x| !x.is_finite())
            || m.input_scale
                .iter()
                .chain(&m.input_std)
                .chain(&m.output_scale)
                .any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err("invalid scales".into());
        }
        let mut width = n;
        for layer in &m.layers {
            if layer.weight.len() != layer.bias.len()
                || layer
                    .weight
                    .iter()
                    .any(|r| r.len() != width || r.iter().any(|x| !x.is_finite()))
                || layer.bias.iter().any(|x| !x.is_finite())
            {
                return Err("invalid dense layer".into());
            }
            width = layer.bias.len();
        }
        if width != NY + NG {
            return Err("output shape".into());
        }
        Ok(m)
    }
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        let expanded;
        if self.format != FORMAT && x.len() != 69 {
            return Err("raw feature shape".into());
        }
        let x = if self.format != FORMAT {
            expanded = {
                let mut v = augment(x);
                if self.format == "causal-material-state/5" {
                    let lambda = v[69].max(v[70]);
                    let r = v[75].max(1.);
                    v.extend([lambda - r, (lambda - r) / r]);
                }
                v
            };
            expanded.as_slice()
        } else {
            x
        };
        if x.len() != self.input_scale.len() {
            return Err("feature width mismatch".into());
        }
        let mut v: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, x)| {
                ((x / self.input_scale[i]).asinh() - self.input_mean[i]) / self.input_std[i]
            })
            .collect();
        if self.float32_inference {
            let mut narrow: Vec<f32> = v.iter().map(|x| *x as f32).collect();
            for (i, l) in self.layers.iter().enumerate() {
                narrow = l
                    .weight
                    .iter()
                    .zip(&l.bias)
                    .map(|(w, b)| {
                        let z = w
                            .iter()
                            .zip(&narrow)
                            .fold(*b as f32, |sum, (w, x)| sum + (*w as f32) * x);
                        if i + 1 < self.layers.len() {
                            z.max(0.)
                        } else {
                            z
                        }
                    })
                    .collect();
            }
            if narrow.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite float32 inference".into());
            }
            return Ok(narrow.into_iter().map(|x| x as f64).collect());
        }
        for (i, l) in self.layers.iter().enumerate() {
            v = l
                .weight
                .iter()
                .zip(&l.bias)
                .map(|(w, b)| {
                    let z = w.iter().zip(&v).fold(*b, |sum, (w, x)| sum + w * x);
                    if i + 1 < self.layers.len() {
                        z.max(0.)
                    } else {
                        z
                    }
                })
                .collect();
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite inference".into());
        }
        Ok(v)
    }
}
pub struct Provider {
    pub model: Model,
    pub calls: u64,
    pub rows: u64,
    /// Training-only observational oracle labels; never applied to this world.
    /// A run with this present is NOT an acceptance evaluation.
    pub labels: Option<Teacher>,
    pub energy_decoder: bool,
    /// Explicit ablation: freeze learned history updates, retaining mechanics.
    pub frozen_history: bool,
}
impl StressProvider for Provider {
    fn responses(
        &mut self,
        s: &ReferenceSolver,
        _: &ChunkLoads,
        jobs: &[BondJob],
        dt: f64,
        fracture: bool,
    ) -> Result<Vec<SuppliedStress>, String> {
        if self.model.format != FORMAT && !self.energy_decoder {
            return Err("physical-state checkpoints require energy or collect-energy mode; their unused raw-force heads are not trained".into());
        }
        let mut out = Vec::new();
        for &job in jobs {
            if job.2 != 1. {
                return Err("single-rate jobs required".into());
            }
            let b = &s.bonds[job.0][job.1];
            let (x, base, d, rate) = features(s, job, dt, fracture)?;
            let pred = if self.frozen_history {
                vec![0.; NY + NG]
            } else {
                self.model.predict(&x)?
            };
            if let Some(labels) = &mut self.labels {
                labels.record(s, job, dt, fracture)?;
            }
            let absolute = ["causal-material-state/4", "causal-material-state/5"]
                .contains(&self.model.format.as_str());
            let y: [f64; NY] = std::array::from_fn(|j| {
                if absolute && [0, 1, 4, 5].contains(&j) {
                    pred[j] * self.model.output_scale[j]
                } else {
                    base[j] + pred[j] * self.model.output_scale[j]
                }
            });
            let previous = joint_history::pack(&b.joint);
            let mut h = previous;
            for j in 0..21 {
                if pred[NY + j] > 0. {
                    h[j] = y[j];
                }
            }
            for j in [0, 1, 6] {
                h[j] = h[j].clamp(0., 1.);
            }
            for j in [0, 1, 2, 3, 6, 16] {
                h[j] = h[j].max(previous[j]);
            }
            h[0] = if previous[0] >= 1. || pred[NY + 21] > 0. {
                1.
            } else {
                h[0].min(1. - f64::EPSILON)
            };
            for j in 4..6 {
                h[j] = if pred[NY + 22 + j - 4] > 0. {
                    y[j].max(0.)
                } else {
                    previous[j]
                };
            }
            if self.energy_decoder {
                // Yield admissibility and history maxima are known physical
                // constraints. The network learns the inelastic state transition
                // inside that domain; it cannot damage an unloaded elastic bond.
                let effective = d.mul_elem(&b.stiffness.as_local());
                let measures = stress_measures(&b.geometry, &effective);
                let idx = stress_ref::joint::failure_indices(&b.strength, &measures, b.weibull);
                let lambda = idx.tension_shear().0;
                let old_force =
                    energy_force(b.stiffness.as_local(), b.geometry.width, d, &previous).0;
                let compression = Local6 {
                    lin: Vec3::new(0., 0., effective.lin.z.min(0.)),
                    ang: old_force.ang,
                };
                let ci = stress_ref::joint::failure_indices(
                    &b.strength,
                    &stress_measures(&b.geometry, &compression),
                    b.weibull,
                )
                .compression_family()
                .0;
                if lambda <= previous[2].max(1.) {
                    h[0] = previous[0];
                    h[4] = previous[4];
                }
                if ci <= previous[3].max(1.) || d.lin.z >= 0. {
                    h[1] = previous[1];
                    h[5] = previous[5];
                }
                if h[1] >= 1. {
                    h[0] = 1.;
                }
                h[2] = previous[2].max(lambda);
                h[3] = previous[3].max(ci);
                h[20] = idx.max();
                // Isotropic plastic sliding must advance along the elastic trial
                // traction and cannot overshoot zero elastic slip.
                let trial = [d.lin.x - previous[7], d.lin.y - previous[8]];
                let norm = trial[0] * trial[0] + trial[1] * trial[1];
                let fraction = if norm > 0. {
                    ((h[7] - previous[7]) * trial[0] + (h[8] - previous[8]) * trial[1]) / norm
                } else {
                    0.
                }
                .clamp(0., 1.);
                h[7] = previous[7] + fraction * trial[0];
                h[8] = previous[8] + fraction * trial[1];
                let trial = d.ang.z - previous[12];
                let fraction = if trial != 0. {
                    (h[12] - previous[12]) / trial
                } else {
                    0.
                }
                .clamp(0., 1.);
                h[12] = previous[12] + fraction * trial;
                h[9..12].copy_from_slice(&previous[9..12]);
            }
            if !fracture || !s.config.fracture {
                h[..7].copy_from_slice(&previous[..7]);
            }
            // Known stress-rate filter is physical state, not a learned latent vector.
            let m = stress_measures(&b.geometry, &d.mul_elem(&b.stiffness.as_local()));
            let governing = m.tension.max(m.shear).max(m.compression);
            let raw = ((governing - previous[18]) / dt).max(0.) / b.strength.youngs_modulus;
            let a = if b.strength.exact_rate_filter {
                -(-dt / b.strength.rate_filter_time).exp_m1()
            } else {
                (dt / b.strength.rate_filter_time).min(1.)
            };
            h[17] = previous[17] + (raw - previous[17]) * a;
            h[18] = governing;
            let changed = [0, 1, 7, 8, 9, 10, 11, 12]
                .iter()
                .any(|&j| h[j] != previous[j]);
            let mut dissipated = if changed { y[39].max(0.) } else { 0. };
            let mut overshoot = if changed { y[40] } else { 0. };
            let mut decoded = None;
            if self.energy_decoder {
                let k = b.stiffness.as_local();
                let now = energy_force(k, b.geometry.width, d, &h);
                let frozen = energy_force(k, b.geometry.width, d, &previous).1;
                let released = if changed { frozen - now.1 } else { 0. };
                dissipated = dissipated.min(released.max(0.));
                overshoot = released - dissipated;
                decoded = Some(now);
            }
            h[19] = previous[19] + dissipated;
            let mode = if h[0] == previous[0] && h[1] == previous[1] {
                joint_history::mode_code(b.joint.mode)
            } else {
                (0..5)
                    .max_by(|&a, &c| pred[NY + 24 + a].total_cmp(&pred[NY + 24 + c]))
                    .unwrap() as i32
            };
            let state = joint_history::unpack(h, b.joint.rebar_broken, mode)?;
            // Exact elastic rest branch; nonlinear response is learned when damaged.
            let nonlinear = h[0] > 0. || h[1] > 0.;
            let (force, stored) = decoded.unwrap_or_else(|| {
                let force = if nonlinear {
                    local(&y[27..33])
                } else {
                    d.mul_elem(&b.stiffness.as_local())
                };
                (
                    force,
                    if nonlinear {
                        y[33].max(0.)
                    } else {
                        0.5 * d.dot(&force)
                    },
                )
            });
            let damp = rate
                .mul_elem(&b.damping)
                .mul_elem(&b.model().secant_factors(&state, &d));
            out.push(SuppliedStress {
                key: (job.0, job.1),
                total_force: force.add(&damp),
                damping_dissipated: damp.dot(&rate) * dt,
                joint: JointResponse {
                    disconnected: b.connected() && !state.connected(false),
                    state,
                    force,
                    stored,
                    dissipated,
                    overshoot,
                    measures: m,
                },
            });
        }
        self.calls += 1;
        self.rows += out.len() as u64;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoded_force_is_energy_gradient_and_matches_frozen_oracle_state() {
        let w =
            stress_ref::world::World::new(&stress_ref::builders::wall_impact(12., "decoder-test"));
        let b = &w.solver.bonds[0][0];
        let k = b.stiffness.as_local();
        for case in 0..50 {
            let d = Local6::from_array(std::array::from_fn(|j| {
                ((case * 7 + j * 13) as f64).sin() * if j < 3 { 0.0001 } else { 0.003 }
            }));
            let mut old = b.joint.clone();
            old.damage = (case % 5) as f64 * 0.18;
            old.crush = (case % 3) as f64 * 0.1;
            let r = b.model().evaluate(&old, &d, 1e-5, true);
            let h = joint_history::pack(&r.state);
            let (force, energy) = energy_force(k, b.geometry.width, d, &h);
            for (a, c) in force.to_array().iter().zip(r.force.to_array()) {
                assert!((a - c).abs() < 1e-9 * (1. + c.abs()), "force {a} != {c}");
            }
            assert!((energy - r.stored).abs() < 1e-9 * (1. + r.stored));
            for j in 0..6 {
                let mut hi = d.to_array();
                let mut lo = hi;
                let eps = 1e-9;
                hi[j] += eps;
                lo[j] -= eps;
                let grad = (energy_force(k, b.geometry.width, Local6::from_array(hi), &h).1
                    - energy_force(k, b.geometry.width, Local6::from_array(lo), &h).1)
                    / (2. * eps);
                assert!(
                    (grad - force.to_array()[j]).abs() < 1e-4 * (1. + force.to_array()[j].abs())
                );
            }
        }
    }
}
