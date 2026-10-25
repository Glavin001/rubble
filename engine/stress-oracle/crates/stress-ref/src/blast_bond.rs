//! NVIDIA Blast's bond-impulse stress solver, the f64 reference for its Slang port.
//!
//! This is the model vibe-land destroys structures with (Blast `ExtStressSolver` from the
//! PhysX fork, solved on the GPU by `stressgpu/NvBlastExtStressGpu.cu`), written from its
//! definitions. It is NOT rubble's displacement solver ([`crate::statics`]): the unknowns
//! are one force/torque pair per bond, and the answer is the least-compliance set of bond
//! forces that balances each node's load.
//!
//! - Graph: chunk nodes (position, mass, scalar inertia; mass 0 is a support) and bonds
//!   (two nodes, centroid, normal, area, Young's modulus, material). A bond's health is its
//!   remaining area; health 0 is a broken bond.
//! - Offsets (`centerBonds`): between two dynamic nodes the bond acts at their midpoint;
//!   against a support it acts at the bond centroid. Offsets are divided by the mean dynamic
//!   offset length `L`; masses are scaled by their geometric mean `M`.
//! - Mass weighting `D` per node: [`MassWeighting::Equalized`] (Blast's `equalizeMasses`, the
//!   ExtStressSolver host path: 1 for every dynamic node, 0 for a support) or
//!   [`MassWeighting::Native`] (the resident path: `sqrt(M L^2 / I)`, `sqrt(M / m)`).
//! - Compliance column scale `s_j = sqrt((E/30 GPa) max(A,1e-4) / max(len,0.05))`,
//!   normalised by its geometric mean (`stress.cpp`, `BLAST_AREA_COMPLIANCE`).
//! - The system: with `B = D C S`, solve `min |B x - b|` with `b = -D^-1 a` (linear part also
//!   over `L`). Node-space CGLS (`StressNodeOperator.cuh`): `x = x0 + B^T mu`, `L = B B^T`,
//!   numerator `|B^T rho|^2` with each bond counted once.
//! - Islands: components of the dynamic nodes over live bonds, cut at supports
//!   (`computeIslands`). Each island has its own tolerance `tol^2 |b|^2` and stops on its own.
//! - Stress (`NvBlastExtStressFormula.h`) and damage (`NvBlastExtStressMaterialFormula.h`).
//!
//! The solver variable is kept in scaled units (`x = J / (s L M)` linear, `J / (s L^2 M)`
//! angular) across frames, as the device does; [`BlastSolver::impulse`] converts it.

use crate::math::Vec3;

/// Reference modulus the compliance weight is relative to (concrete).
pub const MODULUS_REFERENCE: f64 = 30.0e9;
/// Floors on the compliance weight's geometry (degenerate geometry, not physics).
pub const MIN_LENGTH: f64 = 0.05;
pub const MIN_AREA: f64 = 1.0e-4;

/// A chunk of the support graph.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlastNode {
    pub position: Vec3,
    /// 0 for a support (static node).
    pub mass: f64,
    /// Scalar inertia; `<= 0` uses Blast's sphere of the node's volume.
    pub inertia: f64,
    pub volume: f64,
}

impl BlastNode {
    /// Blast's fallback inertia, a solid sphere of the chunk's volume
    /// (`NvBlastExtStressSolver.cpp`, `setNodeInfo`).
    pub fn effective_inertia(&self) -> f64 {
        if self.mass <= 0.0 {
            return 0.0;
        }
        if self.inertia > 0.0 {
            return self.inertia;
        }
        let volume = if self.volume > 0.0 { self.volume } else { 1.0 };
        let radius = (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt();
        0.4 * self.mass * radius * radius
    }
}

/// A bond between two chunks.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlastBond {
    pub nodes: [usize; 2],
    pub centroid: Vec3,
    pub normal: Vec3,
    /// Authored contact area; the bond's starting health.
    pub area: f64,
    /// Young's modulus of the joint; `<= 0` uses [`MODULUS_REFERENCE`].
    pub modulus: f64,
    pub material: usize,
}

/// A material's limits (Pa), Blast's `ExtStressMaterial`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlastMaterial {
    pub compression_elastic: f64,
    pub compression_fatal: f64,
    pub tension_elastic: f64,
    pub tension_fatal: f64,
    pub shear_elastic: f64,
    pub shear_fatal: f64,
    /// Area fraction gradual damage cannot go below (0: none).
    pub residual_area_fraction: f64,
}

impl BlastMaterial {
    /// vibe-land's fallback concrete (`destruction/src/city_config.rs`).
    pub const CONCRETE: BlastMaterial = BlastMaterial {
        compression_elastic: 12e6,
        compression_fatal: 30e6,
        tension_elastic: 1.2e6,
        tension_fatal: 3e6,
        shear_elastic: 1.6e6,
        shear_fatal: 4e6,
        residual_area_fraction: 0.0,
    };
}

/// How node masses weight the system (`D`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassWeighting {
    /// Blast's `equalizeMasses`: every dynamic node weighs the same, so a load is felt as
    /// `M a` whatever the chunk's mass. The ExtStressSolver host path.
    Equalized,
    /// Each chunk's own mass and inertia. The PhysX resident (native) path.
    Native,
}

/// Solver settings. Defaults are vibe-land's production `blast` backend.
#[derive(Clone, Copy, Debug)]
pub struct BlastSettings {
    pub max_iterations: u32,
    pub tolerance: f64,
    pub warm_start: bool,
    pub mass_weighting: MassWeighting,
    /// The compliance column scale (`BLAST_AREA_COMPLIANCE`, on by default).
    pub compliance: bool,
    /// Section-modulus gain cap (`BLAST_BEND_MAX_GAIN`, 3); `<= 0` the legacy fold.
    pub bend_gain_max: f64,
    /// Bending as separate tension/compression fibres (`BLAST_BEND_FIBER`, on).
    pub fibres: bool,
    /// Gradual damage per second below the fatal limit (`BLAST_DAMAGE_RATE`, 2).
    pub damage_rate: f64,
}

impl Default for BlastSettings {
    fn default() -> Self {
        BlastSettings {
            max_iterations: 32,
            tolerance: 1e-3,
            warm_start: true,
            mass_weighting: MassWeighting::Equalized,
            compliance: true,
            bend_gain_max: 3.0,
            fibres: true,
            damage_rate: 2.0,
        }
    }
}

/// An angular/linear pair: a node's load or motion, or a bond's impulse.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AngLin {
    pub ang: Vec3,
    pub lin: Vec3,
}

impl AngLin {
    pub const ZERO: AngLin = AngLin {
        ang: Vec3::ZERO,
        lin: Vec3::ZERO,
    };
    pub fn norm2(self) -> f64 {
        self.ang.norm2() + self.lin.norm2()
    }
    fn add_scaled(self, o: AngLin, k: f64) -> AngLin {
        AngLin {
            ang: self.ang + o.ang * k,
            lin: self.lin + o.lin * k,
        }
    }
}

/// A node's weight in `D`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Weight {
    pub ang: f64,
    pub lin: f64,
}

impl Weight {
    pub fn is_static(self) -> bool {
        !(self.lin > 0.0)
    }
}

/// No island: a support, a dynamic node with no live bond, or a support-support bond.
pub const NO_ISLAND: u32 = u32::MAX;

/// The solver's fixed data, derived once per graph (`ExtStressGpuSolverImpl::prepare`).
#[derive(Clone, Debug)]
pub struct BlastGraph {
    pub nodes: Vec<BlastNode>,
    pub bonds: Vec<BlastBond>,
    pub settings: BlastSettings,
    pub length_scale: f64,
    pub mass_scale: f64,
    /// Offsets in units of `length_scale`.
    pub offset0: Vec<Vec3>,
    pub offset1: Vec<Vec3>,
    pub weight: Vec<Weight>,
    pub col_scale: Vec<f64>,
    /// Unit normal pointing from node 0 to node 1.
    pub normal: Vec<Vec3>,
    /// Distance between the two nodes (1 when degenerate).
    pub node_distance: Vec<f64>,
    /// Node -> incident bonds, `(bond, is_second)`.
    pub node_begin: Vec<usize>,
    pub node_refs: Vec<(usize, bool)>,
}

impl BlastGraph {
    pub fn new(
        nodes: Vec<BlastNode>,
        bonds: Vec<BlastBond>,
        settings: BlastSettings,
    ) -> Result<BlastGraph, String> {
        let n = nodes.len();
        for (j, b) in bonds.iter().enumerate() {
            if b.nodes[0] >= n || b.nodes[1] >= n {
                return Err(format!("bond {j} names node {:?} of {n}", b.nodes));
            }
        }
        // Geometric-mean mass.
        let (mut log_mass, mut massive) = (0.0, 0usize);
        for node in &nodes {
            if node.mass > 0.0 {
                log_mass += node.mass.ln();
                massive += 1;
            }
        }
        let mass_scale = if massive > 0 {
            (log_mass / massive as f64).exp()
        } else {
            1.0
        };

        // Offsets, centred (`centerBonds`), and the mean dynamic offset length.
        let mut offset0 = Vec::with_capacity(bonds.len());
        let mut offset1 = Vec::with_capacity(bonds.len());
        let mut normal = Vec::with_capacity(bonds.len());
        let mut node_distance = Vec::with_capacity(bonds.len());
        let (mut length_sum, mut lengths) = (0.0, 0usize);
        for b in &bonds {
            let (first, second) = (&nodes[b.nodes[0]], &nodes[b.nodes[1]]);
            let o0 = if first.mass <= 0.0 {
                -(b.centroid - second.position)
            } else if second.mass <= 0.0 {
                b.centroid - first.position
            } else {
                (second.position - first.position) * 0.5
            };
            let o1 = -o0;
            if first.mass > 0.0 {
                length_sum += o0.norm();
                lengths += 1;
            }
            if second.mass > 0.0 {
                length_sum += o1.norm();
                lengths += 1;
            }
            offset0.push(o0);
            offset1.push(o1);
            // Normal: unit, pointing from node 0 to node 1 (the upstream `p1 - p1` sign bug is
            // fixed in the fork); zero falls back to the node direction.
            let d = second.position - first.position;
            let dist = d.norm();
            let len = b.normal.norm();
            let nrm = if !(len > 0.0) {
                if dist > 0.0 {
                    d / dist
                } else {
                    Vec3::X
                }
            } else {
                let u = b.normal / len;
                if u.dot(d) < 0.0 {
                    -u
                } else {
                    u
                }
            };
            normal.push(nrm);
            node_distance.push(if dist > 1e-6 { dist } else { 1.0 });
        }
        let mut length_scale = if lengths > 0 {
            length_sum / lengths as f64
        } else {
            1.0
        };
        if !(length_scale > 0.0) {
            length_scale = 1.0;
        }
        for o in offset0.iter_mut().chain(offset1.iter_mut()) {
            *o = *o / length_scale;
        }

        let weight = nodes
            .iter()
            .map(|node| {
                let inertia = node.effective_inertia();
                match settings.mass_weighting {
                    MassWeighting::Equalized => Weight {
                        ang: if inertia > 0.0 { 1.0 } else { 0.0 },
                        lin: if node.mass > 0.0 { 1.0 } else { 0.0 },
                    },
                    MassWeighting::Native => Weight {
                        ang: if inertia > 0.0 {
                            (mass_scale * length_scale * length_scale / inertia).sqrt()
                        } else {
                            0.0
                        },
                        lin: if node.mass > 0.0 {
                            (mass_scale / node.mass).sqrt()
                        } else {
                            0.0
                        },
                    },
                }
            })
            .collect();

        let mut col_scale = vec![1.0; bonds.len()];
        if settings.compliance && !bonds.is_empty() {
            let mut log_sum = 0.0;
            for (j, b) in bonds.iter().enumerate() {
                let len = (nodes[b.nodes[0]].position - nodes[b.nodes[1]].position)
                    .norm()
                    .max(MIN_LENGTH);
                let area = b.area.max(MIN_AREA);
                let modulus = if b.modulus > 0.0 {
                    b.modulus
                } else {
                    MODULUS_REFERENCE
                };
                col_scale[j] = ((modulus / MODULUS_REFERENCE) * area / len).sqrt();
                log_sum += col_scale[j].ln();
            }
            let mean = (log_sum / bonds.len() as f64).exp();
            let recip = if mean > 0.0 { 1.0 / mean } else { 1.0 };
            for s in &mut col_scale {
                *s *= recip;
            }
        }

        // Node -> bond CSR, refs in bond order (the device's `nodeBondRef`).
        let mut node_begin = vec![0usize; n + 1];
        for b in &bonds {
            node_begin[b.nodes[0] + 1] += 1;
            node_begin[b.nodes[1] + 1] += 1;
        }
        for i in 0..n {
            node_begin[i + 1] += node_begin[i];
        }
        let mut cursor = node_begin.clone();
        let mut node_refs = vec![(0usize, false); node_begin[n]];
        for (j, b) in bonds.iter().enumerate() {
            for side in 0..2 {
                let node = b.nodes[side];
                node_refs[cursor[node]] = (j, side == 1);
                cursor[node] += 1;
            }
        }

        Ok(BlastGraph {
            nodes,
            bonds,
            settings,
            length_scale,
            mass_scale,
            offset0,
            offset1,
            weight,
            col_scale,
            normal,
            node_distance,
            node_begin,
            node_refs,
        })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// `t_j = S C^T D rho` for one bond (its bond-space value), from both endpoints.
    fn bond_term(&self, j: usize, rho: &[AngLin]) -> AngLin {
        let [n0, n1] = self.bonds[j].nodes;
        let (w0, w1) = (self.weight[n0], self.weight[n1]);
        let (a0, l0) = (rho[n0].ang * w0.ang, rho[n0].lin * w0.lin);
        let (a1, l1) = (rho[n1].ang * w1.ang, rho[n1].lin * w1.lin);
        let s = self.col_scale[j];
        AngLin {
            ang: (a0 - a1) * s,
            lin: ((l0 - l1) + (self.offset0[j].cross(a0) - self.offset1[j].cross(a1))) * s,
        }
    }

    /// `y = D C S x` over live bonds (the device's `gatherRightMultiply`).
    fn right_multiply(&self, x: &[AngLin], health: &[f64], out: &mut [AngLin]) {
        for (i, o) in out.iter_mut().enumerate() {
            let w = self.weight[i];
            let mut acc = AngLin::ZERO;
            if !(w.ang == 0.0 && w.lin == 0.0) {
                for &(j, second) in &self.node_refs[self.node_begin[i]..self.node_begin[i + 1]] {
                    if !(health[j] > 0.0) {
                        continue;
                    }
                    let s = self.col_scale[j];
                    let (ang, lin) = (x[j].ang * s, x[j].lin * s);
                    if second {
                        acc.ang += self.offset1[j].cross(lin) - ang;
                        acc.lin -= lin;
                    } else {
                        acc.ang += ang - self.offset0[j].cross(lin);
                        acc.lin += lin;
                    }
                }
            }
            *o = AngLin {
                ang: acc.ang * w.ang,
                lin: acc.lin * w.lin,
            };
        }
    }
}

/// Islands of the dynamic nodes over live bonds, cut at supports (`computeIslands`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Islands {
    pub count: usize,
    pub node: Vec<u32>,
    pub bond: Vec<u32>,
}

impl Islands {
    pub fn compute(graph: &BlastGraph, health: &[f64]) -> Islands {
        let n = graph.node_count();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let fixed = |i: usize| graph.weight[i].is_static();
        for (j, b) in graph.bonds.iter().enumerate() {
            let [n0, n1] = b.nodes;
            if !(health[j] > 0.0) || fixed(n0) || fixed(n1) {
                continue;
            }
            let (r0, r1) = (find(&mut parent, n0), find(&mut parent, n1));
            if r0 != r1 {
                parent[r0.max(r1)] = r0.min(r1);
            }
        }
        // Ids in order of first appearance over the bonds, as the device numbers them.
        let mut root_island = vec![NO_ISLAND; n];
        let mut bond = vec![NO_ISLAND; graph.bond_count()];
        let mut count = 0usize;
        for (j, b) in graph.bonds.iter().enumerate() {
            let [n0, n1] = b.nodes;
            if !(health[j] > 0.0) || (fixed(n0) && fixed(n1)) {
                continue;
            }
            let rep = find(&mut parent, if fixed(n0) { n1 } else { n0 });
            if root_island[rep] == NO_ISLAND {
                root_island[rep] = count as u32;
                count += 1;
            }
            bond[j] = root_island[rep];
        }
        let node = (0..n)
            .map(|i| {
                if fixed(i) {
                    NO_ISLAND
                } else {
                    root_island[find(&mut parent, i)]
                }
            })
            .collect();
        Islands { count, node, bond }
    }
}

/// One island's outcome of a solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IslandReport {
    /// Iterations that updated the solution.
    pub iterations: u32,
    pub converged: bool,
    /// Skipped as settled: inputs unchanged since a converged solve.
    pub skipped: bool,
    /// `|B^T r|^2` at the last test, and the island's target `tol^2 |b|^2`.
    pub residual2: f64,
    pub target2: f64,
}

/// Per-bond stresses (Pa).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BondStress {
    /// Signed: positive tension.
    pub normal: f64,
    pub shear: f64,
    /// Peak bending fibre stress, or 0 when folded into `normal` (legacy).
    pub bend: f64,
    pub compression: f64,
    pub tension: f64,
}

/// A damage verdict (`extStressBondDamage`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Damage {
    pub damage: f64,
    pub multiplier: f64,
    pub command: bool,
}

/// The solver's per-frame state: health and the warm-started solution.
#[derive(Clone, Debug)]
pub struct BlastSolver {
    pub graph: BlastGraph,
    pub health: Vec<f64>,
    pub islands: Islands,
    /// The solver variable, scaled units (see the module docs).
    pub x: Vec<AngLin>,
    has_warm_start: bool,
    /// Settled-island skip: the last converged input per island.
    last_input: Vec<AngLin>,
    last_converged: Vec<bool>,
}

impl BlastSolver {
    pub fn new(graph: BlastGraph) -> BlastSolver {
        let health: Vec<f64> = graph.bonds.iter().map(|b| b.area.max(0.0)).collect();
        let islands = Islands::compute(&graph, &health);
        let (n, m) = (graph.node_count(), graph.bond_count());
        BlastSolver {
            graph,
            health,
            islands,
            x: vec![AngLin::ZERO; m],
            has_warm_start: false,
            last_input: vec![AngLin::ZERO; n],
            last_converged: Vec::new(),
        }
    }

    /// Break or damage bonds: health changes repartition the islands and clear the warm start of
    /// the islands they touch.
    pub fn set_health(&mut self, bond: usize, health: f64) {
        self.health[bond] = health.max(0.0);
        if !(self.health[bond] > 0.0) {
            self.x[bond] = AngLin::ZERO;
        }
        self.islands = Islands::compute(&self.graph, &self.health);
        self.last_converged.clear();
    }

    /// Solve the bond impulses for node accelerations `accel` (Blast's per-node "velocity":
    /// `F/m` for a force, `g` for gravity, `w x (w x r)` centrifugal). `skip_settled` skips an
    /// island whose input is bit-identical to its last converged solve.
    pub fn solve(&mut self, accel: &[AngLin], skip_settled: bool) -> Vec<IslandReport> {
        let g = &self.graph;
        let (n, m) = (g.node_count(), g.bond_count());
        assert_eq!(accel.len(), n, "one acceleration per node");
        let islands = &self.islands;
        let warm = g.settings.warm_start && self.has_warm_start;
        let skip_mask: Vec<bool> = (0..islands.count)
            .map(|k| {
                skip_settled
                    && warm
                    && self.last_converged.get(k).copied().unwrap_or(false)
                    && (0..n)
                        .filter(|&i| islands.node[i] == k as u32)
                        .all(|i| accel[i] == self.last_input[i])
            })
            .collect();
        // `bondSettled`: with the skip on, a bond outside every island (support to support)
        // counts as settled too; it couples nothing either way.
        let skipping = skip_settled && warm;
        let settled_bond = |j: usize| {
            let k = islands.bond[j];
            if k == NO_ISLAND {
                skipping
            } else {
                skip_mask[k as usize]
            }
        };

        // b = -D^-1 a (linear over L); a support's weight is replaced by 1.
        let recip_length = 1.0 / g.length_scale;
        let b: Vec<AngLin> = (0..n)
            .map(|i| {
                let w = g.weight[i];
                let ia = if w.ang > 0.0 { 1.0 / w.ang } else { 1.0 };
                let il = if w.lin > 0.0 {
                    recip_length / w.lin
                } else {
                    recip_length
                };
                AngLin {
                    ang: accel[i].ang * -ia,
                    lin: accel[i].lin * -il,
                }
            })
            .collect();
        if !warm {
            for j in 0..m {
                if !settled_bond(j) {
                    self.x[j] = AngLin::ZERO;
                }
            }
        }
        // Live bonds of skipped islands keep their impulses; everything else enters the solve.
        let health: Vec<f64> = (0..m)
            .map(|j| if settled_bond(j) { 0.0 } else { self.health[j] })
            .collect();
        let mut rho = b.clone();
        let mut ax = vec![AngLin::ZERO; n];
        g.right_multiply(&self.x, &health, &mut ax);
        for i in 0..n {
            rho[i] = rho[i].add_scaled(ax[i], -1.0);
        }

        let mut reports = vec![IslandReport::default(); islands.count];
        for k in 0..islands.count {
            reports[k].skipped = skip_mask[k];
        }
        for i in 0..n {
            let k = islands.node[i];
            if k != NO_ISLAND {
                reports[k as usize].target2 += b[i].norm2();
            }
        }
        let tol2 = g.settings.tolerance * g.settings.tolerance;
        for r in &mut reports {
            r.target2 *= tol2;
        }

        let mut mu = vec![AngLin::ZERO; n];
        let mut pi = vec![AngLin::ZERO; n];
        let mut q = vec![AngLin::ZERO; n];
        let mut w = vec![AngLin::ZERO; n];
        let mut active: Vec<bool> = reports.iter().map(|r| !r.skipped).collect();
        let mut z_prev = vec![0.0f64; islands.count];
        let mut z2 = vec![0.0f64; islands.count];
        let mut q2 = vec![0.0f64; islands.count];
        for iteration in 0..g.settings.max_iterations {
            if !active.iter().any(|&a| a) {
                break;
            }
            // w = L rho = D C S (S C^T D rho), and z2 = |S C^T D rho|^2, each bond once.
            let t: Vec<AngLin> = (0..m)
                .map(|j| {
                    if health[j] > 0.0 {
                        g.bond_term(j, &rho)
                    } else {
                        AngLin::ZERO
                    }
                })
                .collect();
            z2.iter_mut().for_each(|z| *z = 0.0);
            for j in 0..m {
                let k = islands.bond[j];
                if health[j] > 0.0 && k != NO_ISLAND {
                    z2[k as usize] += t[j].norm2();
                }
            }
            g.right_multiply(&t, &health, &mut w);
            for k in 0..islands.count {
                if active[k] {
                    reports[k].residual2 = z2[k];
                    if z2[k] <= reports[k].target2 {
                        active[k] = false;
                        reports[k].converged = true;
                    }
                }
            }
            // pi = rho + beta pi ; q = w + beta q.
            q2.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..n {
                let k = islands.node[i];
                if k == NO_ISLAND || !active[k as usize] {
                    continue;
                }
                let k = k as usize;
                let beta = if iteration != 0 && z_prev[k] > 0.0 {
                    z2[k] / z_prev[k]
                } else {
                    0.0
                };
                pi[i] = rho[i].add_scaled(pi[i], beta);
                q[i] = w[i].add_scaled(q[i], beta);
                q2[k] += q[i].norm2();
            }
            for k in 0..islands.count {
                if active[k] {
                    if !(q2[k] > 0.0) || !q2[k].is_finite() {
                        active[k] = false;
                    }
                    z_prev[k] = z2[k];
                }
            }
            // mu += alpha pi ; rho -= alpha q.
            for i in 0..n {
                let k = islands.node[i];
                if k == NO_ISLAND || !active[k as usize] {
                    continue;
                }
                let k = k as usize;
                let alpha = z2[k] / q2[k];
                mu[i] = mu[i].add_scaled(pi[i], alpha);
                rho[i] = rho[i].add_scaled(q[i], -alpha);
            }
            for k in 0..islands.count {
                if active[k] {
                    reports[k].iterations += 1;
                }
            }
        }

        // x = x0 + S C^T D mu.
        for j in 0..m {
            if settled_bond(j) {
                continue;
            }
            if !(self.health[j] > 0.0) {
                self.x[j] = AngLin::ZERO;
                continue;
            }
            let t = g.bond_term(j, &mu);
            self.x[j] = self.x[j].add_scaled(t, 1.0);
        }

        self.has_warm_start = true;
        self.last_input.copy_from_slice(accel);
        self.last_converged = reports.iter().map(|r| r.converged || r.skipped).collect();
        reports
    }

    /// A bond's physical impulse (force and torque; Blast's "impulse" is a force because its
    /// inputs are accelerations).
    pub fn impulse(&self, bond: usize) -> AngLin {
        let g = &self.graph;
        let s = g.col_scale[bond];
        let linear = g.length_scale * g.mass_scale * s;
        let angular = g.length_scale * linear;
        AngLin {
            ang: self.x[bond].ang * angular,
            lin: self.x[bond].lin * linear,
        }
    }

    /// The bond's stresses over its remaining area (`extStressCalcBondStress`, then
    /// `extStressFibre`).
    pub fn stress(&self, bond: usize) -> BondStress {
        let area = self.health[bond];
        if !(area > 0.0) {
            return BondStress::default();
        }
        let s = &self.graph.settings;
        bond_stress(
            self.impulse(bond),
            self.graph.normal[bond],
            area,
            self.graph.node_distance[bond],
            s.bend_gain_max,
            s.fibres,
        )
    }

    /// The bond's damage this tick (`extStressBondDamage`).
    pub fn damage(&self, bond: usize, material: &BlastMaterial, dt: f64) -> Damage {
        let stress = self.stress(bond);
        bond_damage(
            stress,
            self.health[bond],
            self.graph.bonds[bond].area,
            material,
            dt,
            self.graph.settings.damage_rate,
        )
    }
}

/// `extStressCalcBondStress` and `extStressFibre`.
pub fn bond_stress(
    impulse: AngLin,
    normal: Vec3,
    area: f64,
    node_distance: f64,
    bend_gain_max: f64,
    fibres: bool,
) -> BondStress {
    let perpendicular = |v: Vec3, along: f64| (v - normal * along).norm();
    let linear_normal = impulse.lin.dot(normal);
    let mut out = BondStress {
        normal: linear_normal / area,
        shear: perpendicular(impulse.lin, linear_normal) / area,
        ..Default::default()
    };
    let angular_normal = impulse.ang.dot(normal);
    let twist = angular_normal.abs() / area;
    let bend = perpendicular(impulse.ang, angular_normal) / area;
    if bend_gain_max > 0.0 {
        let torsion = 4.81 / area.max(1e-6).sqrt();
        out.shear += twist * torsion.min(bend_gain_max);
        let section = 6.0 / area.max(1e-6).sqrt();
        out.bend = bend * section.min(bend_gain_max);
    } else {
        out.shear += twist * 2.0 / node_distance;
        out.normal += (bend * 2.0 / node_distance).copysign(out.normal);
    }
    if !fibres || bend_gain_max <= 0.0 {
        let combined = out.normal + out.bend.copysign(out.normal);
        out.compression = if combined <= 0.0 { -combined } else { 0.0 };
        out.tension = combined.max(0.0);
    } else {
        out.tension = (out.normal + out.bend).max(0.0);
        out.compression = (out.bend - out.normal).max(0.0);
    }
    out
}

/// `extStressBondDamage`: compression and tension take the larger excess, shear adds; at or
/// above 1 the bond fails outright, otherwise it loses `health min(m dt rate, 1)` (all of
/// `health m` with `dt = 0`), floored by the residual area.
pub fn bond_damage(
    stress: BondStress,
    health: f64,
    original_area: f64,
    material: &BlastMaterial,
    dt: f64,
    rate: f64,
) -> Damage {
    let mut out = Damage::default();
    if !(health > 0.0) {
        return out;
    }
    let excess = |value: f64, elastic: f64, fatal: f64| {
        if value > elastic {
            let d = fatal - elastic;
            Some((value - elastic) / if d > 0.0 { d } else { 1.0 })
        } else {
            None
        }
    };
    let mut axial = excess(
        stress.compression,
        material.compression_elastic,
        material.compression_fatal,
    )
    .unwrap_or(0.0);
    if let Some(t) = excess(
        stress.tension,
        material.tension_elastic,
        material.tension_fatal,
    ) {
        axial = axial.max(t);
    }
    out.multiplier =
        axial + excess(stress.shear, material.shear_elastic, material.shear_fatal).unwrap_or(0.0);
    out.command = out.multiplier > 0.0;
    if !out.command {
        return out;
    }
    let mut damage = if out.multiplier >= 1.0 {
        health
    } else if dt > 0.0 {
        health * (out.multiplier * dt * rate).min(1.0)
    } else {
        health * out.multiplier
    };
    if out.multiplier < 1.0 && material.residual_area_fraction > 0.0 {
        let floor = original_area * material.residual_area_fraction;
        if health - damage < floor {
            damage = if health > floor { health - floor } else { 0.0 };
        }
    }
    out.damage = damage;
    out
}
