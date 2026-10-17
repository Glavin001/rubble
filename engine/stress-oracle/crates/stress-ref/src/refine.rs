//! Multi-level pre-fracture: replace a coarse chunk by its finer children on demand.
//!
//! The children take over the coarse chunk's rigid displacement and velocity field
//! (`u + theta x d`, `v + w x d`), which conserves its momentum and kinetic energy
//! when the children partition it. Every bond that touched the coarse chunk is
//! replaced by the finer bonds across the same interface; where the neighbour is
//! still coarse, the fine bond is re-attached to it (geometry, stiffness and
//! strength re-derived for the new chunk spacing) and inherits the damage history of
//! the coarse bond it replaces, so a refined crack keeps its state.
//!
//! The coarse chunk was rigid, so its children start without internal stress. Rather
//! than letting that mismatch ring through the structure (which would push the
//! neighbours over the refinement threshold in turn), the children are relaxed to
//! static equilibrium against their current neighbours: a small dense solve over the
//! children's degrees of freedom with every other chunk held. Velocities (and so
//! momentum and kinetic energy) are untouched.

use std::collections::HashMap;

use crate::bond::BondGeometry;
use crate::math::Vec3;
use crate::joint::JointState;
use crate::solver::{ReferenceSolver, RtBond, SolverEvent};
use crate::structure::{bond_physics, reduced_mass};

impl ReferenceSolver {
    /// Active chunk representing chunk `c` (itself or its nearest active ancestor).
    pub fn active_representative(&self, s: usize, mut c: usize) -> Option<usize> {
        loop {
            if self.chunks[s][c].active {
                return Some(c);
            }
            c = self.structures[s].chunks[c].parent?;
        }
    }

    /// Refine active chunk `p` of structure `s` into its children. Returns false if it has none.
    pub fn refine_chunk(&mut self, s: usize, p: usize) -> bool {
        let children = self.structures[s].chunks[p].children.clone();
        if children.is_empty() || !self.chunks[s][p].active {
            return false;
        }
        let ci = self.chunks[s][p].cluster;
        let ps = self.chunks[s][p];
        for &c in &children {
            let d = self.structures[s].chunks[c].center - self.structures[s].chunks[p].center;
            let st = &mut self.chunks[s][c];
            st.u = ps.u + ps.th.cross(d);
            st.th = ps.th;
            st.v = ps.v + ps.w.cross(d);
            st.w = ps.w;
            st.active = true;
            st.removed = false;
            st.cluster = ci;
        }
        self.chunks[s][p].active = false;

        // Retire the coarse bonds, remembering their history per neighbour.
        let mut inherited: HashMap<usize, JointState> = HashMap::new();
        for &bi in &self.chunk_bonds[s][p].clone() {
            let b = &mut self.bonds[s][bi];
            if !b.alive {
                continue;
            }
            let other = if b.geometry.a == p { b.geometry.b } else { b.geometry.a };
            b.alive = false;
            let mut st = b.joint.clone();
            st.plastic = Default::default();
            st.rebar_plastic = 0.0;
            st.rebar_slip = [0.0; 2];
            inherited.insert(other, st);
        }

        // Create the finer bonds that now have two distinct active representatives.
        let child_level = self.structures[s].chunks[children[0]].level;
        let child_set: std::collections::HashSet<usize> = children.iter().copied().collect();
        let existing: std::collections::HashSet<(usize, usize, usize)> = self.bonds[s]
            .iter()
            .filter(|b| b.alive)
            .map(|b| (b.source, b.geometry.a, b.geometry.b))
            .collect();
        for (si, sb) in self.structures[s].bonds.clone().iter().enumerate() {
            if sb.level != child_level {
                continue;
            }
            let (x, y) = (sb.geometry.a, sb.geometry.b);
            if !child_set.contains(&x) && !child_set.contains(&y) {
                continue;
            }
            let (Some(rx), Some(ry)) = (self.active_representative(s, x), self.active_representative(s, y)) else {
                continue;
            };
            if rx == ry || existing.contains(&(si, rx, ry)) {
                continue;
            }
            if self.chunks[s][rx].cluster != self.chunks[s][ry].cluster {
                continue;
            }
            let st = &self.structures[s];
            let g = &sb.geometry;
            let geometry = BondGeometry::new(
                rx,
                ry,
                st.chunks[rx].center,
                st.chunks[ry].center,
                g.centroid,
                g.normal,
                g.t1,
                g.area,
                g.width,
            );
            let mred = reduced_mass(&st.chunks[rx], &st.chunks[ry]);
            let (stiffness, strength, rebar, damping) =
                bond_physics(&geometry, &sb.material, sb.buckling_length, sb.rebar_spec.as_ref(), st.stiffness_scale, mred, &st.features, &st.methods);
            // Interface with the outside inherits the coarse bond's history.
            let outside = if child_set.contains(&x) { ry } else { rx };
            let outside_parent = if child_set.contains(&x) { y } else { x };
            let joint = if child_set.contains(&x) && child_set.contains(&y) {
                JointState::new()
            } else {
                inherited
                    .get(&outside)
                    .or_else(|| self.structures[s].chunks[outside_parent].parent.and_then(|q| inherited.get(&q)))
                    .cloned()
                    .unwrap_or_else(JointState::new)
            };
            self.bonds[s].push(RtBond {
                source: si,
                geometry,
                stiffness,
                strength,
                rebar,
                weibull: sb.weibull,
                damping,
                level: sb.level,
                alive: true,
                joint,
                force: Default::default(),
                elastic: Default::default(),
                measures: Default::default(),
                stored: 0.0,
            });
        }
        self.rebuild_adjacency(s);
        self.refresh_cluster_membership(ci);
        self.relax_children(s, &children);
        self.events.push(SolverEvent::Refined { time: self.time, structure: s, chunk: p });
        // Refinement can expose a crack that already disconnects the children.
        self.split_cluster(ci);
        true
    }

    /// Move the given (just activated) chunks to the static equilibrium of the bonds
    /// touching them, with all other chunks held at their current displacements.
    fn relax_children(&mut self, s: usize, children: &[usize]) {
        let n = children.len();
        let index: std::collections::HashMap<usize, usize> = children.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let dim = 6 * n;
        let mut k = vec![vec![0.0f64; dim]; dim];
        let mut rhs = vec![0.0f64; dim];
        let mut bonds: Vec<usize> = children.iter().flat_map(|&c| self.chunk_bonds[s][c].iter().copied()).collect();
        bonds.sort_unstable();
        bonds.dedup();
        for bi in bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let (sa, sb) = (&self.chunks[s][g.a], &self.chunks[s][g.b]);
            let d = g.kinematics(sa.u, sa.th, sb.u, sb.th);
            let kv = b.model().secant_factors(&b.joint, &d).mul_elem(&b.stiffness.as_local());
            let ks = [kv.lin.x, kv.lin.y, kv.lin.z, kv.ang.x, kv.ang.y, kv.ang.z];
            let axes = [g.t1, g.t2, g.normal];
            for comp in 0..6 {
                let t = axes[comp % 3];
                // Kinematic row of this component over [u_a, th_a, u_b, th_b].
                let row: [Vec3; 4] = if comp < 3 {
                    [-t, -(g.ra.cross(t)), t, g.rb.cross(t)]
                } else {
                    [Vec3::ZERO, -t, Vec3::ZERO, t]
                };
                let ends = [(g.a, 0usize), (g.b, 2usize)];
                // Contribution of held DOFs to the component's deformation.
                let mut held = 0.0;
                for &(c, blk) in &ends {
                    if !index.contains_key(&c) || self.structures[s].chunks[c].support != crate::scene::Support::None {
                        let st = &self.chunks[s][c];
                        held += row[blk].dot(st.u) + row[blk + 1].dot(st.th);
                    }
                }
                let free: Vec<(usize, f64)> = ends
                    .iter()
                    .filter(|(c, _)| index.contains_key(c) && self.structures[s].chunks[*c].support == crate::scene::Support::None)
                    .flat_map(|&(c, blk)| {
                        let base = 6 * index[&c];
                        (0..3).map(move |d| (base + d, row[blk][d])).chain((0..3).map(move |d| (base + 3 + d, row[blk + 1][d])))
                    })
                    .collect();
                for &(i, ri) in &free {
                    rhs[i] -= ks[comp] * ri * held;
                    for &(j, rj) in &free {
                        k[i][j] += ks[comp] * ri * rj;
                    }
                }
            }
        }
        // Held (supported) children keep their DOFs.
        for (i, &c) in children.iter().enumerate() {
            if self.structures[s].chunks[c].support != crate::scene::Support::None {
                for d in 0..6 {
                    k[6 * i + d][6 * i + d] = 1.0;
                    rhs[6 * i + d] = 0.0;
                }
            }
        }
        let Some(x) = solve_dense(k, rhs) else { return };
        for (i, &c) in children.iter().enumerate() {
            if self.structures[s].chunks[c].support != crate::scene::Support::None {
                continue;
            }
            let st = &mut self.chunks[s][c];
            st.u = Vec3::new(x[6 * i], x[6 * i + 1], x[6 * i + 2]);
            st.th = Vec3::new(x[6 * i + 3], x[6 * i + 4], x[6 * i + 5]);
        }
    }
}

/// Gaussian elimination with partial pivoting; `None` if singular.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            if f != 0.0 {
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}
