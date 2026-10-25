//! Blast's bond-impulse stress solver on the GPU: the Slang port of the PhysX fork's GPU
//! backend (`stressgpu/NvBlastExtStressGpu.cu`), host path. The kernels are
//! `shaders/blast_cgls.slang` (the per-island node-space CGLS) and
//! `shaders/blast_material.slang` (stresses and damage); the f64 reference they are diffed
//! against is `stress_ref::blast_bond`, which also derives the graph's fixed data (scales,
//! offsets, weights, column scales) that this uploads.
//!
//! Layout: Blast's islands each own a contiguous range of node slots (their dynamic nodes,
//! in node order) and of bond slots (their live bonds, in bond order); bonds in no island
//! (broken, or support to support) follow. Supports have no slot.

use bytemuck::{Pod, Zeroable};
use stress_ref::blast_bond::{
    AngLin, BlastGraph, BlastMaterial, BondStress, Damage, IslandReport, Islands, NO_ISLAND,
};
use stress_ref::math::Vec3;

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

const NO_NODE: u32 = u32::MAX;
const REF_SECOND: u32 = 0x8000_0000;
const NODE_VECTORS: usize = 10;
const REPORT_CONVERGED: u32 = 1;
const REPORT_SKIPPED: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Params {
    island_count: u32,
    max_iterations: u32,
    warm: u32,
    skip_settled: u32,
    tolerance: f32,
    recip_length: f32,
    node_count: u32,
    bond_count: u32,
    linear_scale: f32,
    angular_scale: f32,
    bend_gain_max: f32,
    fibres: u32,
    dt: f32,
    damage_rate: f32,
    apply_damage: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct NodeGpu {
    w_ang: f32,
    w_lin: f32,
    ref_begin: u32,
    ref_end: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct BondGpu {
    n0: u32,
    n1: u32,
    s: f32,
    health: f32,
    o0: [f32; 4],
    o1: [f32; 4],
}

const CGLS_BINDINGS: [Binding; 9] = [
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
    Binding::Storage,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::StorageRw,
];
const MATERIAL_BINDINGS: [Binding; 6] = [
    Binding::Uniform,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
];

fn f4(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

fn v3(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}

/// The slots of one topology: which node and bond each slot holds, and the island ranges.
#[derive(Clone, Debug, Default)]
struct Slots {
    node_slot: Vec<u32>,
    bond_slot: Vec<u32>,
    slot_bond: Vec<usize>,
    slot_node: Vec<usize>,
    ranges: Vec<[u32; 4]>,
}

impl Slots {
    fn new(graph: &BlastGraph, islands: &Islands) -> Slots {
        let mut node_slot = vec![NO_NODE; graph.node_count()];
        let mut bond_slot = vec![u32::MAX; graph.bond_count()];
        let (mut slot_node, mut slot_bond) = (Vec::new(), Vec::new());
        let mut by_island: Vec<(Vec<usize>, Vec<usize>)> =
            vec![(Vec::new(), Vec::new()); islands.count];
        for (i, &k) in islands.node.iter().enumerate() {
            if k != NO_ISLAND {
                by_island[k as usize].0.push(i);
            }
        }
        let mut loose = Vec::new();
        for (j, &k) in islands.bond.iter().enumerate() {
            if k != NO_ISLAND {
                by_island[k as usize].1.push(j);
            } else {
                loose.push(j);
            }
        }
        let mut ranges = Vec::with_capacity(islands.count);
        for (nodes, bonds) in &by_island {
            let (n0, b0) = (slot_node.len() as u32, slot_bond.len() as u32);
            for &i in nodes {
                node_slot[i] = slot_node.len() as u32;
                slot_node.push(i);
            }
            for &j in bonds {
                bond_slot[j] = slot_bond.len() as u32;
                slot_bond.push(j);
            }
            ranges.push([n0, slot_node.len() as u32, b0, slot_bond.len() as u32]);
        }
        for j in loose {
            bond_slot[j] = slot_bond.len() as u32;
            slot_bond.push(j);
        }
        Slots {
            node_slot,
            bond_slot,
            slot_bond,
            slot_node,
            ranges,
        }
    }
}

/// A bond's results after a solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BondResult {
    pub impulse: AngLin,
    pub stress: BondStress,
    pub damage: Damage,
}

/// Blast's stress solver on a device, for one support graph.
pub struct GpuBlastSolver {
    gpu: Gpu,
    pub graph: BlastGraph,
    pub health: Vec<f64>,
    pub islands: Islands,
    materials: Vec<BlastMaterial>,
    slots: Slots,
    solve: Kernel,
    stress: Kernel,
    params: wgpu::Buffer,
    islands_buf: wgpu::Buffer,
    nodes: wgpu::Buffer,
    refs: wgpu::Buffer,
    bonds: wgpu::Buffer,
    accel: wgpu::Buffer,
    x: wgpu::Buffer,
    scratch: wgpu::Buffer,
    report: wgpu::Buffer,
    geometry: wgpu::Buffer,
    materials_buf: wgpu::Buffer,
    out: wgpu::Buffer,
    has_warm_start: bool,
}

impl GpuBlastSolver {
    pub fn new(gpu: &Gpu, graph: BlastGraph, materials: Vec<BlastMaterial>) -> GpuBlastSolver {
        let health: Vec<f64> = graph.bonds.iter().map(|b| b.area.max(0.0)).collect();
        let islands = Islands::compute(&graph, &health);
        let solve = gpu.compute(&shaders::BLAST_CGLS, "blast_solve", &CGLS_BINDINGS);
        let stress = gpu.compute(
            &shaders::BLAST_MATERIAL,
            "blast_bond_stress",
            &MATERIAL_BINDINGS,
        );
        let empty = |label: &str| gpu.storage::<u32>(label, &[0; 4]);
        let mut s = GpuBlastSolver {
            gpu: gpu.clone(),
            params: gpu.uniform("blast params", &Params::default()),
            islands_buf: empty("blast islands"),
            nodes: empty("blast nodes"),
            refs: empty("blast refs"),
            bonds: empty("blast bonds"),
            accel: empty("blast accel"),
            x: empty("blast x"),
            scratch: empty("blast scratch"),
            report: empty("blast report"),
            geometry: empty("blast geometry"),
            materials_buf: empty("blast materials"),
            out: empty("blast out"),
            graph,
            health,
            islands,
            materials,
            slots: Slots::default(),
            solve,
            stress,
            has_warm_start: false,
        };
        s.upload_topology(&vec![AngLin::ZERO; s.graph.bond_count()]);
        s
    }

    /// Lay out the current islands and upload everything that depends on them; `x` (by bond)
    /// is the solution carried over.
    fn upload_topology(&mut self, x: &[AngLin]) {
        let g = &self.graph;
        let slots = Slots::new(g, &self.islands);
        let mut nodes = vec![NodeGpu::default(); slots.slot_node.len().max(1)];
        let mut refs = Vec::new();
        for (slot, &i) in slots.slot_node.iter().enumerate() {
            let begin = refs.len() as u32;
            for &(j, second) in &g.node_refs[g.node_begin[i]..g.node_begin[i + 1]] {
                refs.push(slots.bond_slot[j] | if second { REF_SECOND } else { 0 });
            }
            nodes[slot] = NodeGpu {
                w_ang: g.weight[i].ang as f32,
                w_lin: g.weight[i].lin as f32,
                ref_begin: begin,
                ref_end: refs.len() as u32,
            };
        }
        let m = slots.slot_bond.len();
        let mut bonds = vec![BondGpu::default(); m.max(1)];
        let mut geometry = vec![[0.0f32; 4]; 2 * m.max(1)];
        let mut xs = vec![[0.0f32; 4]; 2 * m.max(1)];
        for (slot, &j) in slots.slot_bond.iter().enumerate() {
            let b = g.bonds[j];
            bonds[slot] = BondGpu {
                n0: slots.node_slot[b.nodes[0]],
                n1: slots.node_slot[b.nodes[1]],
                s: g.col_scale[j] as f32,
                health: self.health[j] as f32,
                o0: f4(g.offset0[j]),
                o1: f4(g.offset1[j]),
            };
            let n = g.normal[j];
            geometry[2 * slot] = [
                n.x as f32,
                n.y as f32,
                n.z as f32,
                g.node_distance[j] as f32,
            ];
            geometry[2 * slot + 1] = [b.area as f32, f32::from_bits(b.material as u32), 0.0, 0.0];
            xs[2 * slot] = f4(x[j].ang);
            xs[2 * slot + 1] = f4(x[j].lin);
        }
        let materials: Vec<[f32; 4]> = self
            .materials
            .iter()
            .flat_map(|m| {
                [
                    [
                        m.compression_elastic as f32,
                        m.compression_fatal as f32,
                        m.tension_elastic as f32,
                        m.tension_fatal as f32,
                    ],
                    [
                        m.shear_elastic as f32,
                        m.shear_fatal as f32,
                        m.residual_area_fraction as f32,
                        0.0,
                    ],
                ]
            })
            .collect();
        let gpu = &self.gpu;
        let ranges: Vec<[u32; 4]> = if slots.ranges.is_empty() {
            vec![[0; 4]]
        } else {
            slots.ranges.clone()
        };
        self.islands_buf = gpu.storage("blast islands", &ranges);
        self.nodes = gpu.storage("blast nodes", &nodes);
        self.refs = gpu.storage(
            "blast refs",
            &if refs.is_empty() { vec![0u32] } else { refs },
        );
        self.bonds = gpu.storage("blast bonds", &bonds);
        self.accel = gpu.storage("blast accel", &vec![[0.0f32; 4]; 2 * nodes.len()]);
        self.x = gpu.storage("blast x", &xs);
        self.scratch = gpu.storage(
            "blast scratch",
            &vec![[0.0f32; 4]; NODE_VECTORS * nodes.len() + 2 * bonds.len()],
        );
        self.report = gpu.storage("blast report", &vec![[0u32; 4]; ranges.len()]);
        self.geometry = gpu.storage("blast geometry", &geometry);
        self.materials_buf = gpu.storage(
            "blast materials",
            &if materials.is_empty() {
                vec![[0.0f32; 4]; 2]
            } else {
                materials
            },
        );
        self.out = gpu.storage("blast out", &vec![[0.0f32; 4]; 4 * bonds.len()]);
        self.slots = slots;
    }

    fn write_params(&self, warm: bool, skip_settled: bool, dt: f64, apply_damage: bool) {
        let g = &self.graph;
        let s = &g.settings;
        let p = Params {
            island_count: self.islands.count as u32,
            max_iterations: s.max_iterations,
            warm: warm as u32,
            skip_settled: skip_settled as u32,
            tolerance: s.tolerance as f32,
            recip_length: (1.0 / g.length_scale) as f32,
            node_count: self.slots.slot_node.len() as u32,
            bond_count: self.slots.slot_bond.len() as u32,
            linear_scale: (g.length_scale * g.mass_scale) as f32,
            angular_scale: (g.length_scale * g.length_scale * g.mass_scale) as f32,
            bend_gain_max: s.bend_gain_max as f32,
            fibres: s.fibres as u32,
            dt: dt as f32,
            damage_rate: s.damage_rate as f32,
            apply_damage: apply_damage as u32,
            pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Solve the bond impulses for node accelerations `accel` (as `BlastSolver::solve`).
    pub fn solve(&mut self, accel: &[AngLin], skip_settled: bool) -> Vec<IslandReport> {
        assert_eq!(
            accel.len(),
            self.graph.node_count(),
            "one acceleration per node"
        );
        let mut a = vec![[0.0f32; 4]; 2 * self.slots.slot_node.len().max(1)];
        for (slot, &i) in self.slots.slot_node.iter().enumerate() {
            a[2 * slot] = f4(accel[i].ang);
            a[2 * slot + 1] = f4(accel[i].lin);
        }
        self.gpu
            .queue
            .write_buffer(&self.accel, 0, bytemuck::cast_slice(&a));
        let warm = self.graph.settings.warm_start && self.has_warm_start;
        self.write_params(warm, skip_settled, 0.0, false);
        if self.islands.count > 0 {
            let bind = self.gpu.bind(
                &self.solve.group,
                &[
                    &self.params,
                    &self.islands_buf,
                    &self.nodes,
                    &self.refs,
                    &self.bonds,
                    &self.accel,
                    &self.x,
                    &self.scratch,
                    &self.report,
                ],
            );
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.solve.dispatch(
                    &mut pass,
                    &bind,
                    self.islands.count * self.solve.workgroup as usize,
                );
            }
            self.gpu.queue.submit([encoder.finish()]);
        }
        self.has_warm_start = true;
        let report: Vec<[u32; 4]> = self.gpu.read_range(&self.report, 0, self.islands.count);
        report
            .iter()
            .map(|r| IslandReport {
                iterations: r[0],
                converged: r[1] & REPORT_CONVERGED != 0,
                skipped: r[1] & REPORT_SKIPPED != 0,
                residual2: f32::from_bits(r[2]) as f64,
                target2: f32::from_bits(r[3]) as f64,
            })
            .collect()
    }

    /// Every bond's impulse, stresses and damage verdict; with `apply_damage`, the damage is
    /// taken from the bonds' health (a broken bond repartitions the islands).
    pub fn bond_results(&mut self, dt: f64, apply_damage: bool) -> Vec<BondResult> {
        let warm = self.has_warm_start;
        self.write_params(warm, false, dt, apply_damage);
        let m = self.slots.slot_bond.len();
        if m > 0 {
            let bind = self.gpu.bind(
                &self.stress.group,
                &[
                    &self.params,
                    &self.bonds,
                    &self.x,
                    &self.geometry,
                    &self.materials_buf,
                    &self.out,
                ],
            );
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.stress.dispatch(&mut pass, &bind, m);
            }
            self.gpu.queue.submit([encoder.finish()]);
        }
        let out: Vec<[f32; 4]> = self.gpu.read_range(&self.out, 0, 4 * m);
        let mut results = vec![BondResult::default(); self.graph.bond_count()];
        for (slot, &j) in self.slots.slot_bond.iter().enumerate() {
            let o = &out[4 * slot..4 * slot + 4];
            results[j] = BondResult {
                impulse: AngLin {
                    ang: v3(o[0]),
                    lin: v3(o[1]),
                },
                stress: BondStress {
                    normal: o[2][0] as f64,
                    shear: o[2][1] as f64,
                    bend: o[2][2] as f64,
                    compression: o[2][3] as f64,
                    tension: o[3][0] as f64,
                },
                damage: Damage {
                    damage: o[3][1] as f64,
                    multiplier: o[3][2] as f64,
                    command: o[3][3] != 0.0,
                },
            };
        }
        if apply_damage {
            let bonds: Vec<BondGpu> = self.gpu.read_range(&self.bonds, 0, m);
            let broke = self
                .slots
                .slot_bond
                .iter()
                .enumerate()
                .any(|(slot, &j)| self.health[j] > 0.0 && !(bonds[slot].health > 0.0));
            for (slot, &j) in self.slots.slot_bond.iter().enumerate() {
                self.health[j] = bonds[slot].health as f64;
            }
            if broke {
                self.repartition();
            }
        }
        results
    }

    /// Every bond's physical impulse.
    pub fn impulses(&mut self) -> Vec<AngLin> {
        let x = self.solution();
        let g = &self.graph;
        (0..g.bond_count())
            .map(|j| {
                let linear = g.length_scale * g.mass_scale * g.col_scale[j];
                let angular = g.length_scale * linear;
                AngLin {
                    ang: x[j].ang * angular,
                    lin: x[j].lin * linear,
                }
            })
            .collect()
    }

    /// The solver variable by bond (scaled units, as `BlastSolver::x`).
    pub fn solution(&self) -> Vec<AngLin> {
        let m = self.slots.slot_bond.len();
        let xs: Vec<[f32; 4]> = self.gpu.read_range(&self.x, 0, 2 * m);
        let mut x = vec![AngLin::ZERO; self.graph.bond_count()];
        for (slot, &j) in self.slots.slot_bond.iter().enumerate() {
            x[j] = AngLin {
                ang: v3(xs[2 * slot]),
                lin: v3(xs[2 * slot + 1]),
            };
        }
        x
    }

    /// Set a bond's health (0 breaks it): the islands are repartitioned.
    pub fn set_health(&mut self, bond: usize, health: f64) {
        self.health[bond] = health.max(0.0);
        self.repartition();
    }

    fn repartition(&mut self) {
        let mut x = self.solution();
        for (j, h) in self.health.iter().enumerate() {
            if !(*h > 0.0) {
                x[j] = AngLin::ZERO;
            }
        }
        self.islands = Islands::compute(&self.graph, &self.health);
        self.upload_topology(&x);
    }
}
