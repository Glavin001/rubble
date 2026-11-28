//! Vehicle contact's residual pass and work guard (§37.1), the accumulated clamp of edge
//! contact's pairs (`ExplicitBeams::edge_contact`), the turn's hit record and the contact
//! measurement switches (docs/contact-design.md).

use super::*;

/// The push-out rate (m/s) of a pair already behind the surface (under measurement,
/// `PUSH_OUT`; 0: none, as before).
pub(crate) fn push_out_rate() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("PUSH_OUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}

/// Residual sweeps of vehicle contact per turn (under measurement, `CONTACT_SWEEPS`;
/// 1: the one residual pass, as before).
pub(crate) fn contact_sweeps() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CONTACT_SWEEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// The Jacobi sweeps on the sub-bodies' masses (`CONTACT_SPLIT=1`, the damped form).
pub(crate) fn contact_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CONTACT_SPLIT").is_ok_and(|v| v == "1"))
}

/// The accumulated sweeps all at once from the same sums (`CONTACT_JACOBI=1`).
pub(crate) fn contact_jacobi() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CONTACT_JACOBI").is_ok_and(|v| v == "1"))
}

/// Vehicle contact's residual sweeps with the accumulated clamp (under measurement,
/// `CONTACT_GS=1`; Gauss–Seidel unless `CONTACT_JACOBI=1`).
pub(crate) fn gauss_seidel() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CONTACT_GS").is_ok_and(|v| v == "1"))
}

pub(crate) fn speculative() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SPECULATIVE").is_ok_and(|v| v == "1"))
}
impl ExplicitBeamsWorld {
    /// §37.1's work guard over owner `a`'s staged hits (`on`; forces `first`,
    /// each with its residual), as `ExplicitBeams::friction_guard_pass` takes self
    /// hits: at the pair's realised mean velocity over the step (the forces now,
    /// the turn's staging sums with them, on the nodes' own masses) a hit whose
    /// force would do positive work is scaled to zero work; returns each hit's
    /// −(1 − s)·t. All found from the same sums.
    pub(super) fn guard_vehicle_pass(&self, a: usize, hits: &[Hit], on: &[bool], first: &[Vec3]) -> Vec<Option<Vec3>> {
        let mut out = vec![None; hits.len()];
        if !crate::explicit_beams::friction_guard() {
            return out;
        }
        let node = |v: usize, i: u32| &self.vehicles[v].nodes[i as usize];
        let acc = |v: usize, i: u32| {
            let n = node(v, i);
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                self.vehicles[v].force_now(i as usize) / n.mass
            }
        };
        let im = |v: usize, i: u32| {
            let n = node(v, i);
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                1.0 / n.mass
            }
        };
        for (j, x) in hits.iter().enumerate() {
            let t = first[j];
            if !on[j] || t == Vec3::ZERO {
                continue;
            }
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            let (cv, ca, ci) = if x.t > 0.0 {
                (
                    node(x.b, x.h).vel * (1.0 - x.t) + node(x.b, x.h2).vel * x.t,
                    acc(x.b, x.h) * (1.0 - x.t) + acc(x.b, x.h2) * x.t,
                    (1.0 - x.t) * (1.0 - x.t) * im(x.b, x.h) + x.t * x.t * im(x.b, x.h2),
                )
            } else {
                (node(x.b, x.h).vel, acc(x.b, x.h), im(x.b, x.h))
            };
            let v = cv
                - (node(a, i1).vel * alpha + node(a, i2).vel * beta + node(a, i0).vel * gamma);
            let ac = ca - (acc(a, i1) * alpha + acc(a, i2) * beta + acc(a, i0) * gamma);
            let inv = ci
                + two_body_inverse_mass([im(a, i1), im(a, i2), im(a, i0)], [alpha, beta, gamma]);
            let a_rest = ac - t * inv;
            let aa = t.dot(v + v + a_rest * STEP);
            let bb = STEP * t.dot(t) * inv;
            if aa + bb > 0.0 && bb > 0.0 {
                let s = if aa < 0.0 { (-aa / bb).min(1.0) } else { 0.0 };
                out[j] = Some(t * -(1.0 - s));
            }
        }
        out
    }

    /// The normal force change that brings hit `x`'s pair's normal velocity at the end of
    /// the step to zero, at the forces now (signed: negative takes back an excess).
    fn residual_signed(&self, a: usize, x: &Hit, split: Option<(&[u32], &[Vec<u32>])>) -> Option<f32> {
        let acc = |v: usize, i: u32| {
            let n = &self.vehicles[v].nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                self.vehicles[v].force_now(i as usize) / n.mass
            }
        };
        // Own masses, or the sub-bodies' (k/m) with `split` (the damped Jacobi form).
        let ka = |i: u32| split.map_or(1.0, |s| s.0.get(i as usize).copied().unwrap_or(1).max(1) as f32);
        let kb = |i: u32| split.map_or(1.0, |s| s.1[x.b].get(i as usize).copied().unwrap_or(1).max(1) as f32);
        let im = |n: &Node| if n.has(NODE_IMMOVABLE) { 0.0 } else { 1.0 / n.mass };
        let [i0, i1, i2] = x.tri;
        let [alpha, beta, gamma] = x.w;
        let va = &self.vehicles[a];
        let node = &self.vehicles[x.b].nodes[x.h as usize];
        let node2 = &self.vehicles[x.b].nodes[x.h2 as usize];
        let (n0, n1, n2) = (&va.nodes[i0 as usize], &va.nodes[i1 as usize], &va.nodes[i2 as usize]);
        let cvel = if x.t > 0.0 { node.vel * (1.0 - x.t) + node2.vel * x.t } else { node.vel };
        let v = cvel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma) + x.n * (x.spec / STEP);
        let at = acc(a, i1) * alpha + acc(a, i2) * beta + acc(a, i0) * gamma;
        let cacc = if x.t > 0.0 { acc(x.b, x.h) * (1.0 - x.t) + acc(x.b, x.h2) * x.t } else { acc(x.b, x.h) };
        let an = (cacc - at).dot(x.n);
        let vn = v.dot(x.n);
        let imc = if x.t > 0.0 {
            (1.0 - x.t) * (1.0 - x.t) * im(node) * kb(x.h) + x.t * x.t * im(node2) * kb(x.h2)
        } else {
            im(node) * kb(x.h)
        };
        let inv = imc + two_body_inverse_mass([im(n1) * ka(i1), im(n2) * ka(i2), im(n0) * ka(i0)], x.w);
        if inv <= 0.0 {
            return None;
        }
        let m = 1.0 / inv;
        Some(-(an * m) - vn * m / STEP)
    }

    pub(super) fn residual_vehicle_pass(
        &mut self,
        a: usize,
        hits: &[Hit],
        on: &[bool],
        split_a: &[u32],
        split_b: &[Vec<u32>],
        first: &[Vec3],
    ) -> Vec<Option<f32>> {
        let acc = |v: usize, i: u32| {
            let n = &self.vehicles[v].nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                self.vehicles[v].force_now(i as usize) / n.mass
            }
        };
        let im = |n: &Node, k: u32| {
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                k as f32 / n.mass
            }
        };
        if gauss_seidel() {
            // Gauss–Seidel (`CONTACT_GS=1`, under measurement): each still-approaching hit,
            // in order, takes the force that stops its pair at the forces now, every hit
            // before it included, on the nodes' own masses (no split: the others' pushes
            // are already in the sums), applied at once.
            // Each hit's normal force so far (the first pass's), which a sweep may raise or
            // lower but never take below zero (sequential impulses' accumulated clamp: an
            // excess is taken back, so a pair is not pushed apart to bounce back in).
            let mut out: Vec<Option<f32>> = vec![None; hits.len()];
            if contact_jacobi() {
                // Jacobi (`CONTACT_JACOBI=1`): every hit's change from the same sums, then
                // all applied; the same accumulated clamp. (The GPU's parallel form.)
                for _ in 0..contact_sweeps() {
                    let deltas: Vec<Option<f32>> = (0..hits.len())
                        .map(|j| {
                            if !on[j] {
                                return None;
                            }
                            let x = &hits[j];
                            let lam0 = first.get(j).map_or(0.0, |f| f.dot(x.n)).max(0.0) + out[j].unwrap_or(0.0);
                            let d = self.residual_signed(a, x, contact_split().then_some((split_a, split_b)))?;
                            let r = (lam0 + d).max(0.0) - lam0;
                            (r != 0.0).then_some(r)
                        })
                        .collect();
                    for (j, d) in deltas.into_iter().enumerate() {
                        let Some(r) = d else { continue };
                        out[j] = Some(out[j].unwrap_or(0.0) + r);
                        let x = &hits[j];
                        let f = x.n * r;
                        if x.t > 0.0 {
                            self.vehicles[x.b].add_contact(x.h as usize, f * (1.0 - x.t));
                            self.vehicles[x.b].add_contact(x.h2 as usize, f * x.t);
                        } else {
                            self.vehicles[x.b].add_contact(x.h as usize, f);
                        }
                        let va = &mut self.vehicles[a];
                        va.add_contact(x.tri[1] as usize, -(f * x.w[0]));
                        va.add_contact(x.tri[2] as usize, -(f * x.w[1]));
                        va.add_contact(x.tri[0] as usize, -(f * x.w[2]));
                    }
                }
                return out;
            }
            for j in (0..contact_sweeps() as usize).flat_map(|_| 0..hits.len()) {
                if !on[j] {
                    continue;
                }
                let x = &hits[j];
                let lam0 = first.get(j).map_or(0.0, |f| f.dot(x.n)).max(0.0) + out[j].unwrap_or(0.0);
                let Some(d) = self.residual_signed(a, x, None) else { continue };
                let lam = (lam0 + d).max(0.0);
                let r = lam - lam0;
                if r == 0.0 {
                    continue;
                }
                out[j] = Some(out[j].unwrap_or(0.0) + r);
                let f = x.n * r;
                if x.t > 0.0 {
                    self.vehicles[x.b].add_contact(x.h as usize, f * (1.0 - x.t));
                    self.vehicles[x.b].add_contact(x.h2 as usize, f * x.t);
                } else {
                    self.vehicles[x.b].add_contact(x.h as usize, f);
                }
                let va = &mut self.vehicles[a];
                va.add_contact(x.tri[1] as usize, -(f * x.w[0]));
                va.add_contact(x.tri[2] as usize, -(f * x.w[1]));
                va.add_contact(x.tri[0] as usize, -(f * x.w[2]));
            }
            return out;
        }
        let mut out = vec![None; hits.len()];
        for (j, x) in hits.iter().enumerate() {
            if !on[j] {
                continue;
            }
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            let va = &self.vehicles[a];
            let node = &self.vehicles[x.b].nodes[x.h as usize];
            let node2 = &self.vehicles[x.b].nodes[x.h2 as usize];
            let (n0, n1, n2) = (
                &va.nodes[i0 as usize],
                &va.nodes[i1 as usize],
                &va.nodes[i2 as usize],
            );
            let cvel = if x.t > 0.0 { node.vel * (1.0 - x.t) + node2.vel * x.t } else { node.vel };
            let v = cvel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma) + x.n * (x.spec / STEP);
            let at = acc(a, i1) * alpha + acc(a, i2) * beta + acc(a, i0) * gamma;
            let cacc = if x.t > 0.0 { acc(x.b, x.h) * (1.0 - x.t) + acc(x.b, x.h2) * x.t } else { acc(x.b, x.h) };
            let an = (cacc - at).dot(x.n);
            let vn = v.dot(x.n);
            // Edge contact's pairs (either vehicle's `edge_contact`): the accumulated clamp,
            // the hit's normal force (the first pass's) moved toward the stop, never below
            // zero; else only an approaching pair still approaching gets more.
            let clamp = self.vehicles[a].edge_contact || self.vehicles[x.b].edge_contact;
            let lam0 = if clamp { first.get(j).map_or(0.0, |f| f.dot(x.n)).max(0.0) } else { 0.0 };
            if !clamp && !(vn < 0.0 && an * STEP + vn < 0.0) {
                continue;
            }
            let imc = if x.t > 0.0 {
                (1.0 - x.t) * (1.0 - x.t) * im(node, split_b[x.b][x.h as usize])
                    + x.t * x.t * im(node2, split_b[x.b][x.h2 as usize])
            } else {
                im(node, split_b[x.b][x.h as usize])
            };
            let inv = imc
                + two_body_inverse_mass(
                    [
                        im(n1, split_a[i1 as usize]),
                        im(n2, split_a[i2 as usize]),
                        im(n0, split_a[i0 as usize]),
                    ],
                    x.w,
                );
            if inv <= 0.0 {
                continue;
            }
            let m = 1.0 / inv;
            let r = -(an * m) - vn * m / STEP;
            if clamp {
                let r = (lam0 + r).max(0.0) - lam0;
                if r != 0.0 {
                    out[j] = Some(r);
                }
            } else if r > 0.0 {
                out[j] = Some(r);
            }
        }
        for (j, x) in hits.iter().enumerate() {
            let Some(r) = out[j] else { continue };
            let f = x.n * r;
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            if x.t > 0.0 {
                self.vehicles[x.b].add_contact(x.h as usize, f * (1.0 - x.t));
                self.vehicles[x.b].add_contact(x.h2 as usize, f * x.t);
            } else {
                self.vehicles[x.b].add_contact(x.h as usize, f);
            }
            let va = &mut self.vehicles[a];
            va.add_contact(i1 as usize, -(f * alpha));
            va.add_contact(i2 as usize, -(f * beta));
            va.add_contact(i0 as usize, -(f * gamma));
        }
        out
    }
}

/// One hit of an owner's walk: the triangle's nodes (n₀, n₁, n₂) and weights
/// (α, β, γ), the partner and its contact point, the normal after the side vote and
/// the penetration r − δ.
pub(super) struct Hit {
    pub(super) tri: [u32; 3],
    pub(super) w: [f32; 3],
    pub(super) b: usize,
    pub(super) h: u32,
    pub(super) n: Vec3,
    pub(super) p: f32,
    /// Speculative: the gap past the band this pair may still close this step (m; 0 in
    /// the band). The response targets the normal velocity −spec/h instead of 0.
    pub(super) spec: f32,
    /// How far behind the surface the contact point is (m; 0 in front).
    pub(super) behind: f32,
    /// A second contacter node and its weight (edge against edge: the contacter's
    /// point is (1 − t)·h + t·h2); t = 0: node h alone (h2 = h), every path as before.
    pub(super) h2: u32,
    pub(super) t: f32,
    /// 0 node against face, 1 edge against edge, 2 node against edge (the owner's
    /// `tri` an edge [i, j, j], weights [s, 0, 1 − s]).
    pub(super) kind: u8,
}

impl Hit {
    /// A node-against-face hit (one contacter node).
    pub(super) fn node(tri: [u32; 3], w: [f32; 3], b: usize, h: u32, n: Vec3, p: f32, spec: f32, behind: f32) -> Self {
        Hit { tri, w, b, h, n, p, spec, behind, h2: h, t: 0.0, kind: 0 }
    }
    /// The owner's nodes with their weights, each once and only with a weight (edge
    /// kinds; a face's three as they are).
    pub(super) fn owner_nodes(&self) -> Vec<(u32, f32)> {
        let [i0, i1, i2] = self.tri;
        let [alpha, beta, gamma] = self.w;
        if self.kind == 0 {
            return vec![(i1, alpha), (i2, beta), (i0, gamma)];
        }
        [(i1, alpha), (i2, beta), (i0, gamma)].into_iter().filter(|x| x.1 > 0.0).collect()
    }
}

/// Vehicle contact's residual pass (§37.1; on, `VEHICLE_RESIDUAL=0` a measurement
/// switch: off). Measured, the 80 km/h T-bone's leak at the five offsets: off
/// 8 / 9 / 5 / 2 / 2 of 1,026 striker nodes, on 3 / 4 / 6 / 0 / 2; the contact-work
/// invariant holds either way.
pub(super) fn vehicle_residual() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("VEHICLE_RESIDUAL").is_ok_and(|v| v == "0")
            && crate::explicit_beams::residual_on()
    })
}

/// A dissipative two-body hit as the contact pass staged it: node `h` of
/// vehicle `b` takes `f`, the nodes `tri` of vehicle `a` take −f·`w`.
#[derive(Clone, Copy, Debug)]
pub(super) struct StagedHit {
    pub(super) a: usize,
    pub(super) b: usize,
    pub(super) h: u32,
    /// A second contacter node and its weight (edge hits; t = 0: h alone).
    pub(super) h2: u32,
    pub(super) t: f32,
    pub(super) tri: [u32; 3],
    pub(super) w: [f32; 3],
    pub(super) n: Vec3,
    pub(super) f: Vec3,
}
