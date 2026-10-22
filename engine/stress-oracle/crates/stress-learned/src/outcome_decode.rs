//! P2's deterministic decode (`learned_impact/outcome/decode.py` and `state_io.py`): one
//! call's head outputs (`outcome_model::Outputs`) to the `learned-impact-decoded-state/1`
//! the host applies at window end. In f64, as P2 runs it:
//!
//! 1. node outputs (body frame, nondimensional) to world Δv, Δω, Δx, Δθ; anchored chunks
//!    at rest;
//! 2. a free island (from the call's inputs only: no anchored chunk, no blast load, no
//!    gravity) gets its centre-of-mass displacement set exactly and the minimum-kinetic-
//!    energy projection of its velocities onto the input momentum and angular momentum;
//! 3. the energy bound KE_out ≤ KE_in + W_gravity + tol·E_in: on a free island only the
//!    non-rigid motion is scaled, otherwise every velocity; skipped under a blast load;
//! 4. breaks where log q̂ ≥ log τ; a survivor's damage `min(d0 + Δd, 1 − 1e-6)` (f32).

use serde_json::{json, Value};
use stress_ref::world::{DecodedBody, DecodedBreak, DecodedDamage, DecodedState, DECODED_STATE_VERSION};

use crate::outcome_calls::{Arr, CallEvent, CALLS_SCHEMA};
use crate::outcome_model::{Outputs, NODE_OUT};

#[derive(Clone, Debug)]
pub struct DecodeConfig {
    pub tau: f64,
    pub project_momentum: bool,
    pub energy_bound: bool,
    /// Fraction of E_in (P0: the ledger residual reaches 4.7% under penalty contact).
    pub energy_tolerance: f64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        DecodeConfig { tau: 1.0, project_momentum: true, energy_bound: true, energy_tolerance: 0.05 }
    }
}

impl DecodeConfig {
    /// From a `learned-impact-weights/1` manifest's `decode` block (threshold break rule).
    pub fn from_manifest(m: &Value) -> Result<DecodeConfig, String> {
        let d = &m["decode"];
        if let Some(r) = d["break_rule"].as_str() {
            if !r.starts_with("log q") {
                return Err(format!("break rule {r:?}: only the failure-margin threshold is ported"));
            }
        }
        Ok(DecodeConfig { tau: d["tau"].as_f64().unwrap_or(1.0), energy_tolerance: d["energy_tolerance"].as_f64().unwrap_or(0.05), ..Default::default() })
    }
}

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn mv(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
fn mm(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}
fn tr(a: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[j][i];
        }
    }
    o
}

/// `geometry.quat_rotate` (w-first, body → world).
fn quat_rotate(q: [f64; 4], v: V3) -> V3 {
    let u = [q[1], q[2], q[3]];
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q[0])), cross(u, t))
}

fn quat_to_matrix(q: [f64; 4]) -> M3 {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// `geometry.rotvec_to_matrix`: Rodrigues, `I + [r]×` below 1e-6 rad.
fn rotvec_to_matrix(r: V3) -> M3 {
    let th = dot(r, r).sqrt();
    let sk = |k: V3| -> M3 { [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]] };
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if th < 1e-6 {
        let rx = sk(r);
        let mut o = eye;
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] += rx[i][j];
            }
        }
        return o;
    }
    let kx = sk(scale(r, 1.0 / th));
    let kk = mm(&kx, &kx);
    let (s, c) = (th.sin(), 1.0 - th.cos());
    let mut o = eye;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += s * kx[i][j] + c * kk[i][j];
        }
    }
    o
}

fn world_inertia(q: [f64; 4], principal: V3) -> M3 {
    let r = quat_to_matrix(q);
    let mut d = [[0.0; 3]; 3];
    for i in 0..3 {
        d[i][i] = principal[i];
    }
    mm(&mm(&r, &d), &tr(&r))
}

/// Gaussian elimination with partial pivoting (`torch.linalg.solve`'s LU), `n × n`.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}

fn inv3(m: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for c in 0..3 {
        let e: Vec<f64> = (0..3).map(|r| f64::from(u8::from(r == c))).collect();
        let x = solve(m.iter().map(|r| r.to_vec()).collect(), e);
        for r in 0..3 {
            o[r][c] = x[r];
        }
    }
    o
}

fn f32s(a: Option<&Arr>, what: &str) -> Result<Vec<f32>, String> {
    match a {
        Some(Arr::F32(_, d)) => Ok(d.clone()),
        _ => Err(format!("call: no f32 {what}")),
    }
}

fn i64s(a: Option<&Arr>, what: &str) -> Result<Vec<i64>, String> {
    match a {
        Some(Arr::I64(_, d)) => Ok(d.clone()),
        Some(Arr::I32(_, d)) => Ok(d.iter().map(|&x| i64::from(x)).collect()),
        _ => Err(format!("call: no integer {what}")),
    }
}

fn col(ev: &CallEvent, array: &str, name: &str) -> Result<usize, String> {
    ev.manifest["columns"][array]
        .as_array()
        .and_then(|c| c.iter().position(|x| x.as_str() == Some(name)))
        .ok_or_else(|| format!("columns.{array}: no {name}"))
}

/// Per-body state of the call (chunks, then impactors), f64.
struct Body {
    mass: f64,
    inertia: V3,
    q: [f64; 4],
    pos: V3,
    v: V3,
    w: V3,
    anchored: bool,
}

fn kinetic(bodies: &[Body], v: &[V3], w: &[V3], inertia: &[M3]) -> f64 {
    bodies.iter().enumerate().map(|(i, b)| 0.5 * (b.mass * dot(v[i], v[i]) + dot(w[i], mv(&inertia[i], w[i])))).sum()
}

fn com_rel(bodies: &[Body], pos: &[V3]) -> Vec<V3> {
    let mt: f64 = bodies.iter().map(|b| b.mass).sum();
    let mut c = [0.0; 3];
    for (b, p) in bodies.iter().zip(pos) {
        c = add(c, scale(*p, b.mass));
    }
    let c = scale(c, 1.0 / mt);
    pos.iter().map(|p| sub(*p, c)).collect()
}

/// What the decode decided besides the state (P2's `decode` flags).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecodeFlags {
    pub free: bool,
    pub external: bool,
    pub energy_bound_active: bool,
    pub energy_bound_infeasible: bool,
}

/// Decode call `ci` of `ev` from its head outputs.
pub fn decode(ev: &CallEvent, ci: usize, out: &Outputs, cfg: &DecodeConfig, meta: Value) -> Result<(DecodedState, DecodeFlags), String> {
    let call = &ev.calls[ci];
    let a = |n: &str| call.array(n);
    let nc = out.chunk_out.rows;
    let ni = out.imp_out.rows;
    let mut bodies = Vec::with_capacity(nc + ni);
    let anch = i64s(a("chunk_anchored"), "chunk_anchored")?;
    for (pre, n) in [("chunk", nc), ("impactor", ni)] {
        let m = f32s(a(&format!("{pre}_mass")), "mass")?;
        let ib = f32s(a(&format!("{pre}_inertia_body")), "inertia_body")?;
        let q = f32s(a(&format!("{pre}_q_world")), "q_world")?;
        let p = f32s(a(&format!("{pre}_pos")), "pos")?;
        let v = f32s(a(&format!("{pre}_v")), "v")?;
        let w = f32s(a(&format!("{pre}_w")), "w")?;
        let v3 = |d: &[f32], i: usize| [d[3 * i] as f64, d[3 * i + 1] as f64, d[3 * i + 2] as f64];
        for i in 0..n {
            bodies.push(Body {
                mass: m[i] as f64,
                inertia: v3(&ib, i),
                q: [q[4 * i] as f64, q[4 * i + 1] as f64, q[4 * i + 2] as f64, q[4 * i + 3] as f64],
                pos: v3(&p, i),
                v: v3(&v, i),
                w: v3(&w, i),
                anchored: pre == "chunk" && anch[i] != 0,
            });
        }
    }
    let n = bodies.len();
    let sc = |k: &str| call.scales.get(k).copied().ok_or_else(|| format!("call: no scale {k}"));
    let (v_ref, l, dt, e_in) = (sc("v")?, sc("L")?, sc("dt")?, sc("E_in")?);
    let g: V3 = ev.manifest["gravity"].as_array().map(|x| [0, 1, 2].map(|k| x.get(k).and_then(Value::as_f64).unwrap_or(0.0))).unwrap_or([0.0; 3]);

    // 1. to_physical
    let o = |i: usize, k: usize| -> f64 { if i < nc { out.chunk_out.d[i * NODE_OUT + k] as f64 } else { out.imp_out.d[(i - nc) * NODE_OUT + k] as f64 } };
    let mut v_end = Vec::with_capacity(n);
    let mut w_end = Vec::with_capacity(n);
    let mut dx = Vec::with_capacity(n);
    let mut drot = Vec::with_capacity(n);
    for (i, b) in bodies.iter().enumerate() {
        let rot = |k: usize| quat_rotate(b.q, [o(i, k), o(i, k + 1), o(i, k + 2)]);
        if b.anchored {
            for x in [&mut v_end, &mut w_end, &mut dx, &mut drot] {
                x.push([0.0; 3]);
            }
            continue;
        }
        v_end.push(add(b.v, scale(rot(0), v_ref)));
        w_end.push(add(b.w, scale(rot(3), v_ref / l)));
        dx.push(add(add(scale(b.v, dt), scale(g, 0.5 * dt * dt)), scale(rot(6), v_ref * dt)));
        drot.push(add(scale(b.w, dt), scale(rot(9), v_ref * dt / l)));
    }

    // causal freedom, from the inputs
    let gf = f32s(a("global_feat"), "global_feat")?;
    let cf = f32s(a("chunk_feat"), "chunk_feat")?;
    let ncf = cf.len() / nc.max(1);
    let any_anch = bodies.iter().any(|b| b.anchored) || call.anchored;
    let mut external = false;
    for c in ["blast_log_tnt", "blast_z", "blast_impulse"] {
        external |= gf[col(ev, "global_feat", c)?] != 0.0;
    }
    for k in 0..6 {
        let c = col(ev, "chunk_feat", &format!("blast_{k}"))?;
        external |= (0..nc).any(|i| cf[i * ncf + c] != 0.0);
    }
    let grav = gf[col(ev, "global_feat", "gravity_mag")?] != 0.0;
    let free = !any_anch && !external && !grav;

    // 2. project
    let mt: f64 = bodies.iter().map(|b| b.mass).sum();
    let inertia0: Vec<M3> = bodies.iter().map(|b| world_inertia(b.q, b.inertia)).collect();
    let pos0 = com_rel(&bodies, &bodies.iter().map(|b| b.pos).collect::<Vec<_>>());
    let (mut p0, mut l0) = ([0.0; 3], [0.0; 3]);
    for (i, b) in bodies.iter().enumerate() {
        let mvv = scale(b.v, b.mass);
        p0 = add(p0, mvv);
        l0 = add(l0, add(cross(pos0[i], mvv), mv(&inertia0[i], b.w)));
    }
    if cfg.project_momentum && free {
        let want = add(scale(p0, dt), scale(g, 0.5 * mt * dt * dt));
        let mut have = [0.0; 3];
        for (i, b) in bodies.iter().enumerate() {
            have = add(have, scale(dx[i], b.mass));
        }
        let corr = scale(sub(want, have), 1.0 / mt);
        for d in &mut dx {
            *d = add(*d, corr);
        }
    }
    let inertia1: Vec<M3> = (0..n)
        .map(|i| {
            let rd = rotvec_to_matrix(drot[i]);
            mm(&mm(&rd, &inertia0[i]), &tr(&rd))
        })
        .collect();
    let pos1 = com_rel(&bodies, &(0..n).map(|i| add(bodies[i].pos, dx[i])).collect::<Vec<_>>());
    if cfg.project_momentum && free {
        let p_t = add(p0, scale(g, mt * dt));
        let l_t = l0;
        // A_i = [[m I, 0], [m [r]×, I_w]], M_i⁻¹ = diag(I/m, I_w⁻¹)
        let mut amat = vec![vec![0.0; 6]; 6];
        let mut ax = [0.0; 6];
        let mut parts = Vec::with_capacity(n);
        for (i, b) in bodies.iter().enumerate() {
            let m = b.mass;
            let r = pos1[i];
            let rx: M3 = [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]];
            let mut ai = [[0.0; 6]; 6];
            let mut minv = [[0.0; 6]; 6];
            let iinv = inv3(&inertia1[i]);
            for p in 0..3 {
                ai[p][p] = m;
                minv[p][p] = 1.0 / m;
                for q in 0..3 {
                    ai[3 + p][q] = m * rx[p][q];
                    ai[3 + p][3 + q] = inertia1[i][p][q];
                    minv[3 + p][3 + q] = iinv[p][q];
                }
            }
            let x = [v_end[i][0], v_end[i][1], v_end[i][2], w_end[i][0], w_end[i][1], w_end[i][2]];
            for p in 0..6 {
                ax[p] += (0..6).map(|q| ai[p][q] * x[q]).sum::<f64>();
            }
            // A M⁻¹ Aᵀ
            let mut am = [[0.0; 6]; 6];
            for p in 0..6 {
                for q in 0..6 {
                    am[p][q] = (0..6).map(|k| ai[p][k] * minv[k][q]).sum();
                }
            }
            for p in 0..6 {
                for q in 0..6 {
                    amat[p][q] += (0..6).map(|k| am[p][k] * ai[q][k]).sum::<f64>();
                }
            }
            parts.push((ai, minv));
        }
        let resid: Vec<f64> = (0..6).map(|p| if p < 3 { p_t[p] } else { l_t[p - 3] } - ax[p]).collect();
        let lam = solve(amat, resid);
        for (i, (ai, minv)) in parts.iter().enumerate() {
            // M⁻¹ Aᵀ λ
            let at: Vec<f64> = (0..6).map(|k| (0..6).map(|p| ai[p][k] * lam[p]).sum()).collect();
            let d: Vec<f64> = (0..6).map(|p| (0..6).map(|k| minv[p][k] * at[k]).sum()).collect();
            v_end[i] = add(v_end[i], [d[0], d[1], d[2]]);
            w_end[i] = add(w_end[i], [d[3], d[4], d[5]]);
        }
    }

    // 3. energy bound
    let (mut active, mut infeasible) = (false, false);
    if cfg.energy_bound {
        let ke_in = kinetic(&bodies, &bodies.iter().map(|b| b.v).collect::<Vec<_>>(), &bodies.iter().map(|b| b.w).collect::<Vec<_>>(), &inertia0);
        let w_g: f64 = bodies.iter().enumerate().map(|(i, b)| b.mass * dot(g, dx[i])).sum();
        let bound = ke_in + w_g + cfg.energy_tolerance * e_in;
        let ke = kinetic(&bodies, &v_end, &w_end, &inertia1);
        active = ke > bound && !external;
        if active {
            if free {
                let mut vc = [0.0; 3];
                for (i, b) in bodies.iter().enumerate() {
                    vc = add(vc, scale(v_end[i], b.mass));
                }
                let vc = scale(vc, 1.0 / mt);
                let vr: Vec<V3> = v_end.iter().map(|v| sub(*v, vc)).collect();
                let mut i_tot = [[0.0; 3]; 3];
                let mut l_rel = [0.0; 3];
                for (i, b) in bodies.iter().enumerate() {
                    let r = pos1[i];
                    let r2 = dot(r, r);
                    for p in 0..3 {
                        for q in 0..3 {
                            i_tot[p][q] += b.mass * (if p == q { r2 } else { 0.0 } - r[p] * r[q]) + inertia1[i][p][q];
                        }
                    }
                    l_rel = add(l_rel, add(cross(r, scale(vr[i], b.mass)), mv(&inertia1[i], w_end[i])));
                }
                let om = solve(i_tot.iter().map(|r| r.to_vec()).collect(), l_rel.to_vec());
                let omega = [om[0], om[1], om[2]];
                let u: Vec<V3> = (0..n).map(|i| sub(vr[i], cross(omega, pos1[i]))).collect();
                let wres: Vec<V3> = w_end.iter().map(|w| sub(*w, omega)).collect();
                let ke_base = 0.5 * (mt * dot(vc, vc) + dot(omega, l_rel));
                let ke_res = kinetic(&bodies, &u, &wres, &inertia1);
                let s = ((bound - ke_base) / ke_res.max(1e-30)).clamp(0.0, 1.0).sqrt();
                infeasible = ke_base > bound;
                for i in 0..n {
                    v_end[i] = add(add(vc, cross(omega, pos1[i])), scale(u[i], s));
                    w_end[i] = add(omega, scale(wres[i], s));
                }
            } else {
                let s = (bound.max(0.0) / ke.max(1e-30)).min(1.0).sqrt();
                for i in 0..n {
                    v_end[i] = scale(v_end[i], s);
                    w_end[i] = scale(w_end[i], s);
                }
            }
        }
    }

    // 4. breaks and damage
    let log_tau = cfg.tau.ln();
    let bkeys = i64s(a("bond_key"), "bond_key")?;
    let bf = f32s(a("bond_feat"), "bond_feat")?;
    let nbf = bf.len() / (2 * bkeys.len()).max(1);
    let dcol = col(ev, "bond_feat", "damage")?;
    let ckeys = i64s(a("chunk_key"), "chunk_key")?;
    let ikeys = i64s(a("impactor_key"), "impactor_key")?;
    let mut bonds_broken = Vec::new();
    let mut bond_damage = Vec::new();
    for (k, &key) in bkeys.iter().enumerate() {
        if f64::from(out.bond_log_q[k]) >= log_tau {
            let t = out.bond_time_logits.row(k);
            // argmax: the first maximum, as torch/numpy
            let mut best = 0;
            for j in 1..t.len() {
                if t[j] > t[best] {
                    best = j;
                }
            }
            bonds_broken.push(DecodedBreak { key, bin: best as u32 + 1 });
        } else {
            let d0 = bf[2 * k * nbf + dcol];
            let d = (d0 + out.bond_ddamage[k]).min((1.0f64 - 1e-6) as f32);
            bond_damage.push(DecodedDamage { key, damage: f64::from(d) });
        }
    }
    let body = |i: usize, key: i64, p: Option<f64>| DecodedBody { key, anchored: bodies[i].anchored, v: v_end[i], w: w_end[i], dx: dx[i], drot: drot[i], p_perforate: p };
    let flags = DecodeFlags { free, external, energy_bound_active: active, energy_bound_infeasible: infeasible };
    let state = DecodedState {
        schema_version: DECODED_STATE_VERSION.to_string(),
        calls_schema: CALLS_SCHEMA.to_string(),
        event_id: ev.manifest["event_id"].as_str().unwrap_or("").to_string(),
        call_index: ci as i64,
        frame: call.frame as i64,
        island: 0,
        meta,
        flags: json!({"free": free, "energy_bound_active": active, "energy_bound_infeasible": infeasible}),
        chunks: (0..nc).map(|i| body(i, ckeys[i], None)).collect(),
        impactors: (0..ni).map(|j| body(nc + j, ikeys[j], Some(f64::from(1.0 / (1.0 + (-out.imp_perf_logit[j]).exp()))))).collect(),
        bonds_broken,
        bond_damage,
    };
    Ok((state, flags))
}
