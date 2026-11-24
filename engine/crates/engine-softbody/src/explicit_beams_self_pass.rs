//! Self contact's work guard and residual pass over the turn's hits (§37.1), and the
//! self-contact hit record (node against face, edge against edge).

use super::*;

impl ExplicitBeams {
    /// §37's residual pass over the step's two-body self hits (after the first,
    /// [`Self::apply_two_body_self_hits`], with its counts `split`): a second
    /// iteration of the mass-split pass for the approach the first left. Its
    /// sub-bodies are lighter than the nodes (k/m), so a hit whose nodes are in
    /// other hits stops only m_split·inv of its pair's approach alone (the rest a
    /// ghost speed the position steps cancel each step). Each hit's pair, at the
    /// forces now (the first pass's contact sums with them), approaching (vn < 0)
    /// and still approaching over the step, a·h + vn < 0 ([`contact_response`]'s
    /// rule, a the pair's relative normal acceleration), takes r = −(a·m +
    /// vn·m/h) > 0 along its normal with the first pass's reduced mass m:
    /// momentum-exact, and never more than the first pass's split allows (counts of
    /// the active hits only were measured to over-stop: +0.23 J of self-contact
    /// work in a tenth of the 58 km/h wall). No friction and no position step (the
    /// first pass's). All found from the sums before any is added (order free).
    /// §37.1's work guard over the step's two-body self hits, after the residual
    /// pass. A hit's force t (its first-pass force with its residual, `first`) is
    /// judged at the pair's realised mean velocity over the step: the forces now
    /// (the staging sums with the residual's) on the nodes' own masses. Its work is
    /// W(s) = s·A + s²·B, with
    /// * A = t·(2v + h·a′), a′ the pair's relative acceleration without t;
    /// * B = h·|t|²·inv, inv the nodes' own inverse masses.
    ///
    /// Where W(1) > 0, the force is scaled to the root s = −A/B (0 if A ≥ 0), and
    /// (1 − s)·t is taken back from the pair (returned per hit). The first pass's
    /// caps judge each hit at its own predicted velocity with split masses; hits
    /// sharing nodes then did +1–2 J of friction work in a tenth at rest. A force
    /// that does positive work moves the pair the way it pushes, so scaling its
    /// normal part lets nothing in. Every guard is found from the same sums before
    /// any is applied (order free, momentum-exact).
    pub(crate) fn friction_guard_pass(
        &self,
        found: &[SelfHit],
        first: &[Vec3],
    ) -> Vec<Option<Vec3>> {
        let mut out = vec![None; found.len()];
        if !friction_guard() {
            return out;
        }
        let now: Vec<Vec3> = self.forces_now();
        let acc = |i: u32| {
            let n = &self.nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                now[i as usize] / n.mass
            }
        };
        let im = |i: u32| {
            let n = &self.nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                1.0 / n.mass
            }
        };
        for (j, (x, &t)) in found.iter().zip(first).enumerate() {
            let (hn, [i0, i1, i2], [alpha, beta, gamma]) = (x.hn, x.tri, x.w);
            if t == Vec3::ZERO {
                continue;
            }
            let node = &self.nodes[hn as usize];
            let node2 = &self.nodes[x.hn2 as usize];
            let (n0, n1, n2) = (
                &self.nodes[i0 as usize],
                &self.nodes[i1 as usize],
                &self.nodes[i2 as usize],
            );
            let (cv, ca, ci) = if x.t > 0.0 {
                (
                    node.vel * (1.0 - x.t) + node2.vel * x.t,
                    acc(hn) * (1.0 - x.t) + acc(x.hn2) * x.t,
                    (1.0 - x.t) * (1.0 - x.t) * im(hn) + x.t * x.t * im(x.hn2),
                )
            } else {
                (node.vel, acc(hn), im(hn))
            };
            let v = cv - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma);
            let a = ca - (acc(i1) * alpha + acc(i2) * beta + acc(i0) * gamma);
            let inv = ci
                + crate::explicit_contact::two_body_inverse_mass(
                    [im(i1), im(i2), im(i0)],
                    [alpha, beta, gamma],
                );
            let a_rest = a - t * inv;
            let aa = t.dot(v + v + a_rest * STEP);
            let bb = STEP * t.dot(t) * inv;
            if aa + bb > 0.0 && bb > 0.0 {
                let s = if aa < 0.0 { (-aa / bb).min(1.0) } else { 0.0 };
                out[j] = Some(t * -(1.0 - s));
            }
        }
        out
    }

    pub(crate) fn residual_self_pass(
        &mut self,
        found: &[SelfHit],
        split: &[u32],
    ) -> Vec<Option<f32>> {
        let now: Vec<Vec3> = self.forces_now();
        let acc = |i: u32, n: &Node| {
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                now[i as usize] / n.mass
            }
        };
        let mut drive = Vec::with_capacity(found.len());
        for x in found {
            let (hn, [i0, i1, i2], [alpha, beta, gamma], n) = (x.hn, x.tri, x.w, x.n);
            let node = &self.nodes[hn as usize];
            let node2 = &self.nodes[x.hn2 as usize];
            let (n0, n1, n2) = (
                &self.nodes[i0 as usize],
                &self.nodes[i1 as usize],
                &self.nodes[i2 as usize],
            );
            let cv = if x.t > 0.0 { node.vel * (1.0 - x.t) + node2.vel * x.t } else { node.vel };
            let v = cv - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma);
            let at = acc(i1, n1) * alpha + acc(i2, n2) * beta + acc(i0, n0) * gamma;
            let cacc = if x.t > 0.0 { acc(hn, node) * (1.0 - x.t) + acc(x.hn2, node2) * x.t } else { acc(hn, node) };
            let a = (cacc - at).dot(n);
            let vn = v.dot(n);
            let x = vn + vn.max(0.0);
            // (Approaching pairs only: a separating one has no ghost speed, and
            // its residual over-stopped where hits share nodes, +0.4 J of
            // self-contact work in a tenth of the 48 km/h wall.)
            drive.push((a, x, vn < 0.0 && a * STEP + x < 0.0));
        }
        let im = |n: &Node, k: u32| {
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                k as f32 / n.mass
            }
        };
        let mut out = vec![None; found.len()];
        for (j, (h, &(a, x, on))) in found.iter().zip(&drive).enumerate() {
            let (hn, [i0, i1, i2], [alpha, beta, gamma]) = (h.hn, h.tri, h.w);
            if !on {
                continue;
            }
            let imc = if h.t > 0.0 {
                (1.0 - h.t) * (1.0 - h.t) * im(&self.nodes[hn as usize], split[hn as usize])
                    + h.t * h.t * im(&self.nodes[h.hn2 as usize], split[h.hn2 as usize])
            } else {
                im(&self.nodes[hn as usize], split[hn as usize])
            };
            let inv = imc
                + crate::explicit_contact::two_body_inverse_mass(
                    [
                        im(&self.nodes[i1 as usize], split[i1 as usize]),
                        im(&self.nodes[i2 as usize], split[i2 as usize]),
                        im(&self.nodes[i0 as usize], split[i0 as usize]),
                    ],
                    [alpha, beta, gamma],
                );
            if inv <= 0.0 {
                continue;
            }
            let m = 1.0 / inv;
            let r = -(a * m) - x * m / STEP;
            if r > 0.0 {
                out[j] = Some(r);
            }
        }
        // (Into the staging sums, read by the guard.)
        for (j, x) in found.iter().enumerate() {
            let ([i0, i1, i2], [alpha, beta, gamma], n) = (x.tri, x.w, x.n);
            let Some(r) = out[j] else { continue };
            let f = n * r;
            self.add_contacter(x, f);
            self.add_contact(i1 as usize, -(f * alpha));
            self.add_contact(i2 as usize, -(f * beta));
            self.add_contact(i0 as usize, -(f * gamma));
        }
        out
    }
}

/// Self edge contact with `edge_contact` (a measurement, off; `SELF_EDGE=1`: on). Off
/// because self and vehicle contact are separate passes over shared nodes: with it the
/// head-on's edges through the other car rise 99 → ~170 (`docs/contact-design.md`).
pub fn self_edge_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SELF_EDGE").is_ok_and(|v| v == "1"))
}

/// One self-contact hit: contacter node `hn` (with `hn2` at weight `t`: an edge's point
/// (1 − t)·hn + t·hn2; t = 0 the node alone) against the point Σ w·node of `tri` (an edge
/// [i, j, j] with weights [s, 0, 1 − s] for edge kinds), normal `n` (the contacter's
/// side), depth `depth`. `kind`: 0 node against face, 1 edge against edge.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SelfHit {
    pub hn: u32,
    pub tri: [u32; 3],
    pub w: [f32; 3],
    pub depth: f32,
    pub n: Vec3,
    pub hn2: u32,
    pub t: f32,
    pub kind: u8,
}

impl SelfHit {
    pub(crate) fn node(hn: u32, tri: [u32; 3], w: [f32; 3], depth: f32, n: Vec3) -> Self {
        SelfHit { hn, tri, w, depth, n, hn2: hn, t: 0.0, kind: 0 }
    }
    /// The contacter's nodes and weights ((hn2, 0) when single).
    pub(crate) fn contacter(&self) -> [(u32, f32); 2] {
        if self.t > 0.0 {
            [(self.hn, 1.0 - self.t), (self.hn2, self.t)]
        } else {
            [(self.hn, 1.0), (self.hn2, 0.0)]
        }
    }
    /// The owner's nodes with a weight, each once (edge kinds).
    pub(crate) fn owner_nodes(&self) -> Vec<(u32, f32)> {
        let [i0, i1, i2] = self.tri;
        let [a, b, c] = self.w;
        [(i1, a), (i2, b), (i0, c)].into_iter().filter(|x| x.1 > 0.0).collect()
    }
}

impl ExplicitBeams {
    /// Adds hit `x`'s force `f` on its contacter (shared by weight when an edge).
    pub(crate) fn add_contacter(&mut self, x: &SelfHit, f: Vec3) {
        if x.t > 0.0 {
            self.add_contact(x.hn as usize, f * (1.0 - x.t));
            self.add_contact(x.hn2 as usize, f * x.t);
        } else {
            self.add_contact(x.hn as usize, f);
        }
    }
}
