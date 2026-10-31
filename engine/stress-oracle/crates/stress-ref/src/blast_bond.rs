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

mod native;
use native::native_pcg;
use serde::{Deserialize, Serialize};

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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlastMaterial {
    pub compression_elastic: f64,
    pub compression_fatal: f64,
    pub tension_elastic: f64,
    pub tension_fatal: f64,
    pub shear_elastic: f64,
    pub shear_fatal: f64,
    /// Area fraction gradual damage cannot go below (0: none).
    pub residual_area_fraction: f64,
    /// Young's modulus the bonds of this material are weighted by (0: [`MODULUS_REFERENCE`]).
    #[serde(default)]
    pub modulus: f64,
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
        modulus: 30e9,
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
    pub solver: SolverKind,
    /// [`SolverKind::Native`]: also stop when a preconditioned step moves the forces by at most
    /// this fraction of them (`ExtStressGpuSolveParams::forceTolerance`; 0 off).
    pub force_tolerance: f64,
}

/// Which solve runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    /// The CUDA host path (ExtStressSolver's GPU solve, vibe-land's `blast` backend):
    /// unpreconditioned node-space CGLS restarted every solve.
    HostCgls,
    /// The resident path (PhysX's native destruction stage, vibe-land's `native` backend):
    /// projected PCG with the two-step Chebyshev block-Jacobi polynomial
    /// (`componentStressSolve`), rigid modes projected out, reliable residual, settled reuse.
    Native,
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
            solver: SolverKind::HostCgls,
            force_tolerance: 0.0,
        }
    }
}

impl BlastSettings {
    /// PhysX's native destruction stage as vibe-land configures it on the Mac
    /// (`destruction/src/native_runtime.rs`): native weighting, the resident PCG, 16 iterations
    /// at tolerance 1e-3, warm-started.
    pub fn native() -> BlastSettings {
        BlastSettings {
            max_iterations: 16,
            mass_weighting: MassWeighting::Native,
            solver: SolverKind::Native,
            ..Default::default()
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

/// What Blast itself reads from a scene pack, kept beside the scene so a Blast-model run takes
/// Blast's inputs rather than rubble's hull-derived ones: the pack's node centroids, masses and
/// volumes, its bond centroids, normals and areas, and both limit sets of each material
/// (`scene_pack::ImportOptions::blast_authoring`). Node `i` is the scene's chunk `i` and bond
/// `k` the scene's bond `k` of its one body; positions are in the scene's frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlastAuthoring {
    pub nodes: Vec<BlastNodeDesc>,
    pub bonds: Vec<BlastBondDesc>,
    pub materials: Vec<BlastMaterial>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlastNodeDesc {
    pub position: [f64; 3],
    pub mass: f64,
    pub volume: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlastBondDesc {
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub area: f64,
    pub material: usize,
}

impl BlastAuthoring {
    /// The solver graph of `scene`'s one body from its authoring, and the materials by index.
    pub fn graph(
        scene: &crate::scene::Scene,
        settings: BlastSettings,
    ) -> Result<(BlastGraph, Vec<BlastMaterial>), String> {
        let a = scene
            .blast
            .as_ref()
            .ok_or("the scene has no Blast authoring (import the pack with blast_authoring)")?;
        let body = match scene.bodies.as_slice() {
            [body] => body,
            _ => {
                return Err(format!(
                    "Blast authoring needs exactly one body, the scene has {}",
                    scene.bodies.len()
                ))
            }
        };
        if a.nodes.len() != body.chunks.len() || a.bonds.len() != body.bonds.len() {
            return Err(format!(
                "Blast authoring has {} nodes and {} bonds, the body {} chunks and {} bonds",
                a.nodes.len(),
                a.bonds.len(),
                body.chunks.len(),
                body.bonds.len()
            ));
        }
        let nodes = a
            .nodes
            .iter()
            .map(|n| BlastNode {
                position: Vec3::from_array(n.position),
                mass: n.mass,
                inertia: 0.0,
                volume: n.volume,
            })
            .collect();
        let mut bonds = Vec::with_capacity(a.bonds.len());
        for (k, (d, b)) in a.bonds.iter().zip(&body.bonds).enumerate() {
            let material = a.materials.get(d.material).ok_or(format!(
                "bond {k}: material {} of {}",
                d.material,
                a.materials.len()
            ))?;
            bonds.push(BlastBond {
                nodes: [b.a, b.b],
                centroid: Vec3::from_array(d.centroid),
                normal: Vec3::from_array(d.normal),
                area: d.area,
                modulus: material.modulus,
                material: d.material,
            });
        }
        Ok((
            BlastGraph::new(nodes, bonds, settings)?,
            a.materials.clone(),
        ))
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
        // A component's root is its lowest node; the components with a coupling bond are the
        // islands, numbered in order of their roots (as blast_topology.slang numbers them; the
        // device numbers them by first appearance over the bonds, which changes nothing but the
        // ids).
        let mut has_bond = vec![false; n];
        for (j, b) in graph.bonds.iter().enumerate() {
            let [n0, n1] = b.nodes;
            if health[j] > 0.0 && !(fixed(n0) && fixed(n1)) {
                let rep = find(&mut parent, if fixed(n0) { n1 } else { n0 });
                has_bond[rep] = true;
            }
        }
        let mut root_island = vec![NO_ISLAND; n];
        let mut count = 0usize;
        for i in 0..n {
            if !fixed(i) && parent[i] == i && has_bond[i] {
                root_island[i] = count as u32;
                count += 1;
            }
        }
        let mut bond = vec![NO_ISLAND; graph.bond_count()];
        for (j, b) in graph.bonds.iter().enumerate() {
            let [n0, n1] = b.nodes;
            if health[j] > 0.0 && !(fixed(n0) && fixed(n1)) {
                bond[j] = root_island[find(&mut parent, if fixed(n0) { n1 } else { n0 })];
            }
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
    /// Solve in f32, the device's precision (the reference is f64): the same recurrence in the
    /// arithmetic the GPU runs, for checking a capped iterate, which f32 rounding moves far more
    /// than it moves a converged one.
    pub device_precision: bool,
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
            device_precision: false,
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
        // The native path skips only on its certificate (`beginNativeSettledReuse`), so with or
        // without `skip_settled`.
        let native = g.settings.solver == SolverKind::Native;
        let skip_settled = skip_settled || native;
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

        if !warm {
            for j in 0..m {
                if !settled_bond(j) {
                    self.x[j] = AngLin::ZERO;
                }
            }
        }
        // Live bonds of skipped islands keep their impulses; everything else enters the solve.
        let live: Vec<bool> = (0..m)
            .map(|j| !settled_bond(j) && self.health[j] > 0.0)
            .collect();
        let mut reports = vec![IslandReport::default(); islands.count];
        for k in 0..islands.count {
            reports[k].skipped = skip_mask[k];
            // The native certificate stands for a converged solve (beginNativeSettledReuse).
            reports[k].converged = skip_mask[k] && native;
        }
        let enters: Vec<bool> = (0..m).map(|j| !settled_bond(j)).collect();
        match (native, self.device_precision) {
            (false, true) => {
                cgls::<f32>(g, islands, accel, &live, &enters, &mut self.x, &mut reports)
            }
            (false, false) => {
                cgls::<f64>(g, islands, accel, &live, &enters, &mut self.x, &mut reports)
            }
            (true, true) => {
                native_pcg::<f32>(g, islands, accel, &live, warm, &mut self.x, &mut reports)
            }
            (true, false) => {
                native_pcg::<f64>(g, islands, accel, &live, warm, &mut self.x, &mut reports)
            }
        }

        self.has_warm_start = true;
        self.last_input.copy_from_slice(accel);
        // The native certificate (`commitNativeSettledReuse`): a warm solve that converged with
        // no update stores an output the next identical input may keep.
        self.last_converged = reports
            .iter()
            .map(|r| {
                if native {
                    r.skipped || (warm && r.converged && r.iterations == 0)
                } else {
                    r.converged || r.skipped
                }
            })
            .collect();
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

/// The arithmetic the solve runs in: f64 (the reference) or f32 (the device's).
trait Real:
    Copy
    + Default
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    fn of(x: f64) -> Self;
    fn to(self) -> f64;
    fn finite(self) -> bool;
}

impl Real for f64 {
    fn of(x: f64) -> Self {
        x
    }
    fn to(self) -> f64 {
        self
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
}

impl Real for f32 {
    fn of(x: f64) -> Self {
        x as f32
    }
    fn to(self) -> f64 {
        self as f64
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
}

/// An angular/linear pair in `T`.
#[derive(Clone, Copy, Default)]
struct Pair<T> {
    a: [T; 3],
    l: [T; 3],
}

fn add<T: Real>(u: [T; 3], v: [T; 3]) -> [T; 3] {
    [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
}
fn sub<T: Real>(u: [T; 3], v: [T; 3]) -> [T; 3] {
    [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
}
fn scale<T: Real>(u: [T; 3], k: T) -> [T; 3] {
    [u[0] * k, u[1] * k, u[2] * k]
}
fn cross<T: Real>(u: [T; 3], v: [T; 3]) -> [T; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}
fn dot<T: Real>(u: [T; 3], v: [T; 3]) -> T {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}
fn of3<T: Real>(v: Vec3) -> [T; 3] {
    [T::of(v.x), T::of(v.y), T::of(v.z)]
}
fn to3<T: Real>(v: [T; 3]) -> Vec3 {
    Vec3::new(v[0].to(), v[1].to(), v[2].to())
}
impl<T: Real> Pair<T> {
    fn norm2(self) -> T {
        dot(self.a, self.a) + dot(self.l, self.l)
    }
    fn plus(self, o: Pair<T>, k: T) -> Pair<T> {
        Pair {
            a: add(self.a, scale(o.a, k)),
            l: add(self.l, scale(o.l, k)),
        }
    }
}

/// The node-space CGLS of [`BlastSolver::solve`] in `T`, every island at once: `b`, the
/// residual of the warm start, the iterations, and `x += B^T mu` over the bonds that `enter`
/// (live bonds `live` couple). Fills each report's iterations, convergence and norms.
fn cgls<T: Real>(
    g: &BlastGraph,
    islands: &Islands,
    accel: &[AngLin],
    live: &[bool],
    enters: &[bool],
    x_out: &mut [AngLin],
    reports: &mut [IslandReport],
) {
    let (n, m) = (g.node_count(), g.bond_count());
    let zero = T::default();
    let w: Vec<(T, T)> = g
        .weight
        .iter()
        .map(|w| (T::of(w.ang), T::of(w.lin)))
        .collect();
    let s: Vec<T> = g.col_scale.iter().map(|&s| T::of(s)).collect();
    let o0: Vec<[T; 3]> = g.offset0.iter().map(|&o| of3(o)).collect();
    let o1: Vec<[T; 3]> = g.offset1.iter().map(|&o| of3(o)).collect();
    let mut x: Vec<Pair<T>> = x_out
        .iter()
        .map(|v| Pair {
            a: of3(v.ang),
            l: of3(v.lin),
        })
        .collect();
    // t_j = S C^T D v.
    let bond_term = |j: usize, v: &[Pair<T>]| -> Pair<T> {
        let [n0, n1] = g.bonds[j].nodes;
        let (a0, l0) = (scale(v[n0].a, w[n0].0), scale(v[n0].l, w[n0].1));
        let (a1, l1) = (scale(v[n1].a, w[n1].0), scale(v[n1].l, w[n1].1));
        Pair {
            a: scale(sub(a0, a1), s[j]),
            l: scale(
                add(sub(l0, l1), sub(cross(o0[j], a0), cross(o1[j], a1))),
                s[j],
            ),
        }
    };
    // (D C S y)_i over live bonds (gatherRightMultiply).
    let gather = |i: usize, y: &[Pair<T>]| -> Pair<T> {
        let mut acc = Pair::<T>::default();
        if w[i].0 == zero && w[i].1 == zero {
            return acc;
        }
        for &(j, second) in &g.node_refs[g.node_begin[i]..g.node_begin[i + 1]] {
            if !live[j] {
                continue;
            }
            let (ya, yl) = (scale(y[j].a, s[j]), scale(y[j].l, s[j]));
            if second {
                acc.a = add(acc.a, sub(cross(o1[j], yl), ya));
                acc.l = sub(acc.l, yl);
            } else {
                acc.a = add(acc.a, sub(ya, cross(o0[j], yl)));
                acc.l = add(acc.l, yl);
            }
        }
        Pair {
            a: scale(acc.a, w[i].0),
            l: scale(acc.l, w[i].1),
        }
    };

    // b = -D^-1 a (linear over L); a support's weight is replaced by 1.
    let recip_length = T::of(1.0 / g.length_scale);
    let one = T::of(1.0);
    let b: Vec<Pair<T>> = (0..n)
        .map(|i| {
            let ia = if w[i].0 > zero { one / w[i].0 } else { one };
            let il = if w[i].1 > zero {
                recip_length / w[i].1
            } else {
                recip_length
            };
            Pair {
                a: scale(of3(accel[i].ang), -ia),
                l: scale(of3(accel[i].lin), -il),
            }
        })
        .collect();
    let mut rho: Vec<Pair<T>> = (0..n).map(|i| b[i].plus(gather(i, &x), -one)).collect();
    let mut target2 = vec![zero; islands.count];
    for i in 0..n {
        let k = islands.node[i];
        if k != NO_ISLAND {
            target2[k as usize] = target2[k as usize] + b[i].norm2();
        }
    }
    let tol = T::of(g.settings.tolerance);
    for t in &mut target2 {
        *t = *t * tol * tol;
    }

    let mut mu = vec![Pair::<T>::default(); n];
    let mut pi = vec![Pair::<T>::default(); n];
    let mut q = vec![Pair::<T>::default(); n];
    let mut active: Vec<bool> = reports.iter().map(|r| !r.skipped).collect();
    let mut z_prev = vec![zero; islands.count];
    let mut z2 = vec![zero; islands.count];
    let mut q2 = vec![zero; islands.count];
    for iteration in 0..g.settings.max_iterations {
        if !active.iter().any(|&a| a) {
            break;
        }
        // t = B^T rho, and z2 = |t|^2, each bond once.
        let t: Vec<Pair<T>> = (0..m)
            .map(|j| {
                if live[j] {
                    bond_term(j, &rho)
                } else {
                    Pair::default()
                }
            })
            .collect();
        z2.iter_mut().for_each(|z| *z = zero);
        for j in 0..m {
            let k = islands.bond[j];
            if live[j] && k != NO_ISLAND {
                z2[k as usize] = z2[k as usize] + t[j].norm2();
            }
        }
        for k in 0..islands.count {
            if active[k] {
                reports[k].residual2 = z2[k].to();
                if z2[k] <= target2[k] {
                    active[k] = false;
                    reports[k].converged = true;
                }
            }
        }
        // pi = rho + beta pi ; q = B t + beta q.
        q2.iter_mut().for_each(|v| *v = zero);
        for i in 0..n {
            let k = islands.node[i];
            if k == NO_ISLAND || !active[k as usize] {
                continue;
            }
            let k = k as usize;
            let beta = if iteration != 0 && z_prev[k] > zero {
                z2[k] / z_prev[k]
            } else {
                zero
            };
            pi[i] = rho[i].plus(pi[i], beta);
            q[i] = gather(i, &t).plus(q[i], beta);
            q2[k] = q2[k] + q[i].norm2();
        }
        for k in 0..islands.count {
            if active[k] {
                if !(q2[k] > zero) || !q2[k].finite() {
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
            mu[i] = mu[i].plus(pi[i], alpha);
            rho[i] = rho[i].plus(q[i], -alpha);
        }
        for k in 0..islands.count {
            if active[k] {
                reports[k].iterations += 1;
            }
        }
    }
    for (k, r) in reports.iter_mut().enumerate() {
        r.target2 = target2[k].to();
    }

    // x = x0 + S C^T D mu.
    for j in 0..m {
        if !enters[j] {
            continue;
        }
        x[j] = if live[j] {
            x[j].plus(bond_term(j, &mu), one)
        } else {
            Pair::default()
        };
        x_out[j] = AngLin {
            ang: to3(x[j].a),
            lin: to3(x[j].l),
        };
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
