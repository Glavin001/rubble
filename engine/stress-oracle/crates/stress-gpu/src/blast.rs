//! Blast's bond-impulse stress solver on the GPU: the Slang port of the PhysX fork's GPU
//! backend (`stressgpu/NvBlastExtStressGpu.cu`), host path. The kernels are
//! `shaders/blast_cgls.slang` (the per-island node-space CGLS), `shaders/blast_material.slang`
//! (stresses and damage) and `shaders/blast_topology.slang` (the islands, from the bonds'
//! health); the f64 reference they are diffed against is `stress_ref::blast_bond`, which also
//! derives the graph's fixed data (scales, offsets, weights, column scales) uploaded here.
//!
//! Nodes and bonds keep their indices (the structure's chunks and bonds): an island is a list
//! of each, which the GPU rebuilds when a bond breaks, so a frame reads nothing back to run.

use bytemuck::{Pod, Zeroable};
use stress_ref::blast_bond::{
    AngLin, BlastGraph, BlastMaterial, BondStress, Damage, IslandReport, Islands, NO_ISLAND,
};
use stress_ref::math::Vec3;

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::shaders;

const REF_SECOND: u32 = 0x8000_0000;
const NODE_VECTORS: usize = 10;
const SCAN_BLOCK: usize = 1024;
const REPORT_CONVERGED: u32 = 1;
const REPORT_SKIPPED: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Params {
    pub island_count: u32,
    pub max_iterations: u32,
    pub warm: u32,
    pub skip_settled: u32,
    pub tolerance: f32,
    pub recip_length: f32,
    pub node_count: u32,
    pub bond_count: u32,
    pub linear_scale: f32,
    pub angular_scale: f32,
    pub bend_gain_max: f32,
    pub fibres: u32,
    pub dt: f32,
    pub damage_rate: f32,
    pub apply_damage: u32,
    pub pad: u32,
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

const CGLS_BINDINGS: [Binding; 10] = [
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
    Binding::Storage,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::Storage,
];
const MATERIAL_BINDINGS: [Binding; 6] = [
    Binding::Uniform,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
];
const TOPOLOGY_BINDINGS: [Binding; 8] = [
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::StorageRw,
];

/// The topology kernels by what they cover.
#[derive(Clone, Copy)]
enum Over {
    Nodes,
    Bonds,
    ScanBlocks,
    One,
    Islands,
}
const TOPOLOGY: [(&str, Over); 12] = [
    ("bt_reset", Over::Nodes),
    ("bt_union", Over::Bonds),
    ("bt_flatten", Over::Nodes),
    ("bt_count_nodes", Over::Nodes),
    ("bt_count_bonds", Over::Bonds),
    ("bt_scan_blocks", Over::ScanBlocks),
    ("bt_scan_top", Over::One),
    ("bt_scan_add", Over::ScanBlocks),
    ("bt_islands", Over::Nodes),
    ("bt_scatter_nodes", Over::Nodes),
    ("bt_scatter_bonds", Over::Bonds),
    ("bt_sort", Over::Islands),
];

fn f4(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

fn v3(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}

/// Words of `blast_topology.slang`'s `topo`.
fn topo_words(nodes: usize) -> usize {
    10 * nodes + 3 * nodes.div_ceil(SCAN_BLOCK) + 4
}

/// A bond's results after a solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BondResult {
    pub impulse: AngLin,
    pub stress: BondStress,
    pub damage: Damage,
}

/// The device buffers of one support graph, for a frame that records the solve into its own
/// encoder ([`GpuBlastSolver::record_topology`], [`GpuBlastSolver::record_solve`],
/// [`GpuBlastSolver::record_stress`]).
pub struct BlastBuffers {
    pub params: wgpu::Buffer,
    pub islands: wgpu::Buffer,
    pub nodes: wgpu::Buffer,
    pub refs: wgpu::Buffer,
    pub bonds: wgpu::Buffer,
    /// Per node 2 float4 (angular, linear acceleration, structure frame): the frame's input.
    pub accel: wgpu::Buffer,
    /// Per bond 2 float4: the solver variable (scaled units).
    pub x: wgpu::Buffer,
    pub scratch: wgpu::Buffer,
    pub report: wgpu::Buffer,
    pub lists: wgpu::Buffer,
    pub topo: wgpu::Buffer,
    /// Written by the topology: the solve's and sort's size (islands, 1, 1), then the island
    /// count.
    pub args: wgpu::Buffer,
    /// `args` copied for the indirect dispatches (a buffer a dispatch binds for writing cannot
    /// size it).
    pub dispatch: wgpu::Buffer,
    pub geometry: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    /// Per bond 4 float4: impulse (angular, linear), (normal, shear, bend, compression),
    /// (tension, damage, multiplier, command).
    pub out: wgpu::Buffer,
}

/// Blast's stress solver on a device, for one support graph.
pub struct GpuBlastSolver {
    gpu: Gpu,
    pub graph: BlastGraph,
    pub buffers: BlastBuffers,
    solve: Kernel,
    stress: Kernel,
    topology: Vec<(Kernel, Over)>,
    solve_bind: wgpu::BindGroup,
    stress_bind: wgpu::BindGroup,
    topology_bind: wgpu::BindGroup,
    has_warm_start: bool,
    /// A bond's health changed since the islands were built.
    topology_dirty: bool,
}

impl GpuBlastSolver {
    pub fn new(gpu: &Gpu, graph: BlastGraph, materials: Vec<BlastMaterial>) -> GpuBlastSolver {
        let (n, m) = (graph.node_count(), graph.bond_count());
        let mut nodes = vec![NodeGpu::default(); n.max(1)];
        let mut refs = Vec::new();
        for i in 0..n {
            let begin = refs.len() as u32;
            for &(j, second) in &graph.node_refs[graph.node_begin[i]..graph.node_begin[i + 1]] {
                refs.push(j as u32 | if second { REF_SECOND } else { 0 });
            }
            nodes[i] = NodeGpu {
                w_ang: graph.weight[i].ang as f32,
                w_lin: graph.weight[i].lin as f32,
                ref_begin: begin,
                ref_end: refs.len() as u32,
            };
        }
        let mut bonds = vec![BondGpu::default(); m.max(1)];
        let mut geometry = vec![[0.0f32; 4]; 2 * m.max(1)];
        for (j, b) in graph.bonds.iter().enumerate() {
            bonds[j] = BondGpu {
                n0: b.nodes[0] as u32,
                n1: b.nodes[1] as u32,
                s: graph.col_scale[j] as f32,
                health: b.area.max(0.0) as f32,
                o0: f4(graph.offset0[j]),
                o1: f4(graph.offset1[j]),
            };
            let nrm = graph.normal[j];
            geometry[2 * j] = [
                nrm.x as f32,
                nrm.y as f32,
                nrm.z as f32,
                graph.node_distance[j] as f32,
            ];
            geometry[2 * j + 1] = [b.area as f32, f32::from_bits(b.material as u32), 0.0, 0.0];
        }
        let mut material_rows: Vec<[f32; 4]> = materials
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
        if material_rows.is_empty() {
            material_rows = vec![[0.0; 4]; 2];
        }
        let args = gpu.storage("blast args", &[0u32; 4]);
        let dispatch = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blast dispatch"),
            size: 16,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffers = BlastBuffers {
            params: gpu.uniform("blast params", &Params::default()),
            islands: gpu.storage("blast islands", &vec![[0u32; 4]; n.max(1)]),
            nodes: gpu.storage("blast nodes", &nodes),
            refs: gpu.storage(
                "blast refs",
                &if refs.is_empty() { vec![0u32] } else { refs },
            ),
            bonds: gpu.storage("blast bonds", &bonds),
            accel: gpu.storage("blast accel", &vec![[0.0f32; 4]; 2 * n.max(1)]),
            x: gpu.storage("blast x", &vec![[0.0f32; 4]; 2 * m.max(1)]),
            scratch: gpu.storage(
                "blast scratch",
                &vec![[0.0f32; 4]; NODE_VECTORS * n.max(1) + 2 * m.max(1)],
            ),
            report: gpu.storage("blast report", &vec![[0u32; 4]; n.max(1)]),
            lists: gpu.storage("blast lists", &vec![0u32; (n + m).max(1)]),
            topo: gpu.storage("blast topology", &vec![0u32; topo_words(n.max(1))]),
            args,
            dispatch,
            geometry: gpu.storage("blast geometry", &geometry),
            materials: gpu.storage("blast materials", &material_rows),
            out: gpu.storage("blast out", &vec![[0.0f32; 4]; 4 * m.max(1)]),
        };
        let solve = gpu.compute(&shaders::BLAST_CGLS, "blast_solve", &CGLS_BINDINGS);
        let stress = gpu.compute(
            &shaders::BLAST_MATERIAL,
            "blast_bond_stress",
            &MATERIAL_BINDINGS,
        );
        let topology: Vec<(Kernel, Over)> = TOPOLOGY
            .iter()
            .map(|&(e, over)| {
                (
                    gpu.compute(&shaders::BLAST_TOPOLOGY, e, &TOPOLOGY_BINDINGS),
                    over,
                )
            })
            .collect();
        let b = &buffers;
        let solve_bind = gpu.bind(
            &solve.group,
            &[
                &b.params, &b.islands, &b.nodes, &b.refs, &b.bonds, &b.accel, &b.x, &b.scratch,
                &b.report, &b.lists,
            ],
        );
        let stress_bind = gpu.bind(
            &stress.group,
            &[&b.params, &b.bonds, &b.x, &b.geometry, &b.materials, &b.out],
        );
        let topology_bind = gpu.bind(
            &topology[0].0.group,
            &[
                &b.params, &b.nodes, &b.bonds, &b.topo, &b.islands, &b.lists, &b.report, &b.args,
            ],
        );
        let s = GpuBlastSolver {
            gpu: gpu.clone(),
            graph,
            buffers,
            solve,
            stress,
            topology,
            solve_bind,
            stress_bind,
            topology_bind,
            has_warm_start: false,
            topology_dirty: true,
        };
        s.write_params(false, false, 0.0, false);
        s
    }

    /// The frame's parameters (`blast_common.slang` `BlastParams`).
    pub fn params(&self, warm: bool, skip_settled: bool, dt: f64, apply_damage: bool) -> Params {
        let g = &self.graph;
        let s = &g.settings;
        Params {
            // The dispatch is sized on the GPU (`args`); this only bounds it.
            island_count: g.node_count() as u32,
            max_iterations: s.max_iterations,
            warm: warm as u32,
            skip_settled: skip_settled as u32,
            tolerance: s.tolerance as f32,
            recip_length: (1.0 / g.length_scale) as f32,
            node_count: g.node_count() as u32,
            bond_count: g.bond_count() as u32,
            linear_scale: (g.length_scale * g.mass_scale) as f32,
            angular_scale: (g.length_scale * g.length_scale * g.mass_scale) as f32,
            bend_gain_max: s.bend_gain_max as f32,
            fibres: s.fibres as u32,
            dt: dt as f32,
            damage_rate: s.damage_rate as f32,
            apply_damage: apply_damage as u32,
            pad: 0,
        }
    }

    fn write_params(&self, warm: bool, skip_settled: bool, dt: f64, apply_damage: bool) {
        self.gpu.queue.write_buffer(
            &self.buffers.params,
            0,
            bytemuck::bytes_of(&self.params(warm, skip_settled, dt, apply_damage)),
        );
    }

    /// Record the islands' rebuild from the bonds' health.
    pub fn record_topology(&self, encoder: &mut wgpu::CommandEncoder) {
        let (n, m) = (self.graph.node_count(), self.graph.bond_count());
        for (k, over) in &self.topology {
            if let Over::Islands = over {
                encoder.copy_buffer_to_buffer(&self.buffers.args, 0, &self.buffers.dispatch, 0, 16);
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &self.topology_bind, &[]);
            match over {
                Over::Nodes => pass.dispatch_workgroups(groups(n, k.workgroup), 1, 1),
                Over::Bonds => pass.dispatch_workgroups(groups(m, k.workgroup), 1, 1),
                Over::ScanBlocks => pass.dispatch_workgroups(groups(n, SCAN_BLOCK as u32), 1, 1),
                Over::One => pass.dispatch_workgroups(1, 1, 1),
                Over::Islands => pass.dispatch_workgroups_indirect(&self.buffers.dispatch, 0),
            }
        }
    }

    /// Record the solve of every island (one group each, sized on the GPU).
    pub fn record_solve(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.solve.pipeline);
        pass.set_bind_group(0, &self.solve_bind, &[]);
        pass.dispatch_workgroups_indirect(&self.buffers.dispatch, 0);
    }

    /// Record every bond's stresses and damage (and, with the params' `apply_damage`, the
    /// damage taken from its health).
    pub fn record_stress(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        self.stress
            .dispatch(&mut pass, &self.stress_bind, self.graph.bond_count().max(1));
    }

    /// Solve the bond impulses for node accelerations `accel` (as `BlastSolver::solve`).
    pub fn solve(&mut self, accel: &[AngLin], skip_settled: bool) -> Vec<IslandReport> {
        assert_eq!(
            accel.len(),
            self.graph.node_count(),
            "one acceleration per node"
        );
        let a: Vec<[f32; 4]> = accel.iter().flat_map(|v| [f4(v.ang), f4(v.lin)]).collect();
        if !a.is_empty() {
            self.gpu
                .queue
                .write_buffer(&self.buffers.accel, 0, bytemuck::cast_slice(&a));
        }
        let warm = self.graph.settings.warm_start && self.has_warm_start;
        self.write_params(warm, skip_settled, 0.0, false);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        if self.topology_dirty {
            self.record_topology(&mut encoder);
            self.topology_dirty = false;
        }
        self.record_solve(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.has_warm_start = true;
        self.reports()
    }

    /// The islands' reports of the last solve, by island.
    pub fn reports(&self) -> Vec<IslandReport> {
        let count = self.island_count();
        let report: Vec<[u32; 4]> = self.gpu.read_range(&self.buffers.report, 0, count);
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

    fn island_count(&self) -> usize {
        let args: Vec<u32> = self.gpu.read_range(&self.buffers.args, 3, 1);
        args[0] as usize
    }

    /// The islands as the GPU built them (each node's and bond's island; and it checks that the
    /// lists are sorted and cover every member once).
    pub fn islands(&self) -> Islands {
        let (n, m) = (self.graph.node_count(), self.graph.bond_count());
        let count = self.island_count();
        let ranges: Vec<[u32; 4]> = self.gpu.read_range(&self.buffers.islands, 0, count);
        let lists: Vec<u32> = self.gpu.read_range(&self.buffers.lists, 0, n + m);
        let mut out = Islands {
            count,
            node: vec![NO_ISLAND; n],
            bond: vec![NO_ISLAND; m],
        };
        for (k, r) in ranges.iter().enumerate() {
            for (members, (lo, hi)) in
                [(&mut out.node, (r[0], r[1])), (&mut out.bond, (r[2], r[3]))]
            {
                let list = &lists[lo as usize..hi as usize];
                assert!(
                    list.windows(2).all(|w| w[0] < w[1]),
                    "island {k}: list not ascending: {list:?}"
                );
                for &i in list {
                    assert_eq!(
                        members[i as usize], NO_ISLAND,
                        "island {k}: member {i} listed twice"
                    );
                    members[i as usize] = k as u32;
                }
            }
        }
        out
    }

    /// Every bond's impulse, stresses and damage verdict; with `apply_damage`, the damage is
    /// taken from the bonds' health (a broken bond's islands are rebuilt before the next solve).
    pub fn bond_results(&mut self, dt: f64, apply_damage: bool) -> Vec<BondResult> {
        self.write_params(self.has_warm_start, false, dt, apply_damage);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.record_stress(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        if apply_damage {
            self.topology_dirty = true;
        }
        let m = self.graph.bond_count();
        let out: Vec<[f32; 4]> = self.gpu.read_range(&self.buffers.out, 0, 4 * m);
        (0..m)
            .map(|j| {
                let o = &out[4 * j..4 * j + 4];
                BondResult {
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
                }
            })
            .collect()
    }

    /// Every bond's health on the device.
    pub fn health(&self) -> Vec<f64> {
        let bonds: Vec<BondGpu> =
            self.gpu
                .read_range(&self.buffers.bonds, 0, self.graph.bond_count());
        bonds.iter().map(|b| b.health as f64).collect()
    }

    /// Every bond's physical impulse.
    pub fn impulses(&self) -> Vec<AngLin> {
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
        let xs: Vec<[f32; 4]> =
            self.gpu
                .read_range(&self.buffers.x, 0, 2 * self.graph.bond_count());
        xs.chunks(2)
            .map(|p| AngLin {
                ang: v3(p[0]),
                lin: v3(p[1]),
            })
            .collect()
    }

    /// Set a bond's health (0 breaks it, and clears its impulse): its islands are rebuilt before
    /// the next solve.
    pub fn set_health(&mut self, bond: usize, health: f64) {
        let h = health.max(0.0) as f32;
        let at = (bond * std::mem::size_of::<BondGpu>() + 12) as u64;
        self.gpu
            .queue
            .write_buffer(&self.buffers.bonds, at, bytemuck::bytes_of(&h));
        if !(h > 0.0) {
            self.gpu
                .queue
                .write_buffer(&self.buffers.x, (bond * 32) as u64, &[0u8; 32]);
        }
        self.topology_dirty = true;
    }
}
