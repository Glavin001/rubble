//! Blast's bond-impulse stress solver on the GPU: the Slang port of the PhysX fork's GPU
//! backend (`stressgpu/NvBlastExtStressGpu.cu`), host path. The kernels are
//! `shaders/blast_cgls.slang` (the per-island node-space CGLS), `shaders/blast_material.slang`
//! (stresses and damage), `shaders/blast_topology.slang` (the islands, from the bonds' health)
//! and `shaders/blast_fracture.slang` (its inputs and breaks in [`crate::fracture`]'s frame); the
//! f64 reference they are diffed against is `stress_ref::blast_bond`, which also derives the
//! graph's fixed data (scales, offsets, weights, column scales) uploaded here.
//!
//! Nodes and bonds keep their indices (the structure's chunks and bonds): an island is a list
//! of each, which the GPU rebuilds when a bond breaks, so a frame reads nothing back to run.

use bytemuck::{Pod, Zeroable};
use stress_ref::blast_bond::{SolverKind, 
    AngLin, BlastGraph, BlastMaterial, BondStress, Damage, IslandReport, Islands, NO_ISLAND,
};
use stress_ref::math::Vec3;

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

const REF_SECOND: u32 = 0x8000_0000;
const NODE_VECTORS: usize = 10;
/// `blast_native.slang` `NV`: its scratch float4s per node.
const NATIVE_NODE_VECTORS: usize = 24;
const SCAN_BLOCK: usize = 1024;
const REPORT_CONVERGED: u32 = 1;
const REPORT_SKIPPED: u32 = 2;

/// `blast_common.slang` `BlastParams`.
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
    pub force_tolerance: f32,
    pub gravity: [f32; 4],
    pub w_alive: u32,
    pub a_dirty: u32,
    pub a_counters: u32,
    pub wide_min: u32,
    pub wide_rows: u32,
    pub wide_blocks: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct NodeGpu {
    w_ang: f32,
    w_lin: f32,
    ref_begin: u32,
    ref_end: u32,
    /// Over the length scale (the native solve's rigid modes).
    position: [f32; 4],
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

use Binding::{Storage as R, StorageRw as W, Uniform as U};
const CGLS_BINDINGS: [Binding; 10] = [U, R, R, R, W, R, W, W, W, R];
const NATIVE_BINDINGS: [Binding; 10] = [U, R, R, R, W, R, W, W, W, R];
const MATERIAL_BINDINGS: [Binding; 7] = [U, W, W, R, R, W, W];
const TOPOLOGY_BINDINGS: [Binding; 8] = [U, R, R, W, W, W, W, W];
const FRACTURE_BINDINGS: [Binding; 9] = [U, W, W, R, R, R, R, W, R];

/// The topology kernels and their indirect sizes (`blast_topology.slang` `A_*`, words).
const TOPOLOGY: [(&str, u64); 12] = [
    ("bt_reset", A_NODES),
    ("bt_union", A_BONDS),
    ("bt_flatten", A_NODES),
    ("bt_count_nodes", A_NODES),
    ("bt_count_bonds", A_BONDS),
    ("bt_scan_blocks", A_SCAN),
    ("bt_scan_top", A_ONE),
    ("bt_scan_add", A_SCAN),
    ("bt_islands", A_NODES),
    ("bt_scatter_nodes", A_NODES),
    ("bt_scatter_bonds", A_BONDS),
    ("bt_sort", A_SORT),
];
const A_SOLVE: u64 = 0;
const A_COUNT: usize = 3;
const A_NODES: u64 = 4;
const A_BONDS: u64 = 7;
const A_SCAN: u64 = 10;
const A_ONE: u64 = 13;
const A_SORT: u64 = 16;
const ARGS_BYTES: u64 = 80;

fn f4(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

fn v3(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}

/// Where `blast_topology.slang`'s `topo` keeps its totals (`blast_common.slang` `topo_total`):
/// islands, nodes, bonds, a union that did not settle, a pending rebuild.
fn topo_total(nodes: usize) -> usize {
    10 * nodes + 3 * nodes.div_ceil(SCAN_BLOCK)
}

/// A bond's results after a solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BondResult {
    pub impulse: AngLin,
    pub stress: BondStress,
    pub damage: Damage,
}

/// The device buffers of one support graph.
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
    /// Indirect sizes the topology writes (`A_*`).
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
    gate: Kernel,
    topology: Vec<(Kernel, u64)>,
    solve_bind: wgpu::BindGroup,
    stress_bind: wgpu::BindGroup,
    topology_bind: wgpu::BindGroup,
    has_warm_start: bool,
    /// The native solve of islands over `wide_min` nodes (`blast_native_wide.slang`).
    wide: Option<Wide>,
}

/// `blast_native_wide.slang`: its kernels, its state and partial sums, its row table and the
/// indirect sizes copied from it.
struct Wide {
    /// Each kernel with its bind group (on WGSL past the device's storage limit, each entry has
    /// its own layout).
    kernels: Vec<(Kernel, Grid, wgpu::BindGroup)>,
    meta: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    rows: u32,
    blocks: u32,
    min: u32,
}

/// What a wide kernel runs over.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Grid {
    One,
    Islands,
    Nodes,
    Bonds,
    Rows,
}

/// The wide solve's kernels for one frame: set-up, the iteration (recorded `max_iterations`
/// times), the end.
const WIDE_SETUP: &[(&str, Grid)] = &[
    ("w_reset", Grid::One),
    ("w_classify", Grid::Islands),
    ("w_args", Grid::One),
    ("w_init_nodes", Grid::Nodes),
    ("w_init_bonds", Grid::Bonds),
    ("w_setup", Grid::Rows),
    ("w_gram_nodes", Grid::Nodes),
    ("w_gram", Grid::Rows),
    ("w_warm", Grid::Nodes),
];
const WIDE_ITERATION: &[(&str, Grid)] = &[
    ("w_proj_part_r", Grid::Nodes),
    ("w_proj_solve", Grid::Rows),
    ("w_proj_apply_r", Grid::Nodes),
    ("w_bond_r", Grid::Bonds),
    ("w_node_rn", Grid::Nodes),
    ("w_test", Grid::Rows),
    ("w_rebuild", Grid::Nodes),
    ("w_proj_part_rb", Grid::Nodes),
    ("w_proj_solve_rb", Grid::Rows),
    ("w_proj_apply_rb", Grid::Nodes),
    ("w_bond_rb", Grid::Bonds),
    ("w_node_rnb", Grid::Nodes),
    ("w_decide", Grid::Rows),
    ("w_pre1", Grid::Nodes),
    ("w_bond_loc", Grid::Bonds),
    ("w_pre2", Grid::Nodes),
    ("w_proj_part_g", Grid::Nodes),
    ("w_proj_solve", Grid::Rows),
    ("w_proj_apply_g", Grid::Nodes),
    ("w_max", Grid::Nodes),
    ("w_mag", Grid::Rows),
    ("w_scale", Grid::Nodes),
    ("w_gamma", Grid::Rows),
    ("w_dir", Grid::Nodes),
    ("w_bond_p", Grid::Bonds),
    ("w_node_q", Grid::Nodes),
    ("w_den", Grid::Rows),
    ("w_update", Grid::Nodes),
];
const WIDE_END: &[(&str, Grid)] = &[("w_final", Grid::Bonds), ("w_report", Grid::Rows)];
const WIDE_BINDINGS: [Binding; 13] = [U, R, R, R, W, R, W, W, W, R, W, W, R];
/// `blast_native_wide.slang` `ROW_WORDS`, `PARTIAL`, `M_ROWS`.
const WIDE_ROW_WORDS: u32 = 16;
const WIDE_PARTIAL: u32 = 6;
const WIDE_META_ROWS: u32 = 16;
const WIDE_THREADS: usize = 1024;

/// Islands over this many nodes take the wide solve (`STRESS_BLAST_WIDE_MIN`; `0` sends every
/// island there, a large number none).
fn wide_min() -> u32 {
    std::env::var("STRESS_BLAST_WIDE_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)
}

impl GpuBlastSolver {
    pub fn new(gpu: &Gpu, graph: BlastGraph, materials: Vec<BlastMaterial>) -> GpuBlastSolver {
        Self::with_wide_min(gpu, graph, materials, wide_min())
    }

    /// `new`, the native solve taking islands over `wide_min` nodes to its many-group path.
    pub fn with_wide_min(gpu: &Gpu, graph: BlastGraph, materials: Vec<BlastMaterial>, wide_min: u32) -> GpuBlastSolver {
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
                position: f4(graph.nodes[i].position * (1.0 / graph.length_scale)),
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
        // The islands are built before the first solve (a pending rebuild).
        let mut topo = vec![0u32; topo_total(n.max(1)) + 5];
        topo[topo_total(n.max(1)) + 4] = 1;
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
                &vec![[0.0f32; 4]; NODE_VECTORS.max(NATIVE_NODE_VECTORS) * n.max(1) + 2 * m.max(1)],
            ),
            report: gpu.storage("blast report", &vec![[0u32; 4]; n.max(1)]),
            lists: gpu.storage("blast lists", &vec![0u32; (n + m).max(1)]),
            topo: gpu.storage("blast topology", &topo),
            args: gpu.storage("blast args", &[0u32; ARGS_BYTES as usize / 4]),
            dispatch: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("blast dispatch"),
                size: ARGS_BYTES,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            geometry: gpu.storage("blast geometry", &geometry),
            materials: gpu.storage("blast materials", &material_rows),
            out: gpu.storage("blast out", &vec![[0.0f32; 4]; 4 * m.max(1)]),
        };
        let native = graph.settings.solver == SolverKind::Native;
        let solve = if native {
            gpu.compute(&shaders::BLAST_NATIVE, "blast_native", &NATIVE_BINDINGS)
        } else {
            gpu.compute(&shaders::BLAST_CGLS, "blast_solve", &CGLS_BINDINGS)
        };
        let stress = gpu.compute(
            &shaders::BLAST_MATERIAL,
            "blast_bond_stress",
            &MATERIAL_BINDINGS,
        );
        let gate = gpu.compute(&shaders::BLAST_TOPOLOGY, "bt_gate", &TOPOLOGY_BINDINGS);
        let topology = TOPOLOGY
            .iter()
            .map(|&(e, at)| {
                (
                    gpu.compute(&shaders::BLAST_TOPOLOGY, e, &TOPOLOGY_BINDINGS),
                    at,
                )
            })
            .collect();
        let b = &buffers;
        let solve_buffers = [&b.params, &b.islands, &b.nodes, &b.refs, &b.bonds, &b.accel, &b.x, &b.scratch, &b.report, &b.lists];
        let solve_bind = gpu.bind(&solve.group, &solve_buffers);
        let stress_bind = gpu.bind(
            &stress.group,
            &[
                &b.params,
                &b.bonds,
                &b.x,
                &b.geometry,
                &b.materials,
                &b.out,
                &b.topo,
            ],
        );
        let topology_bind = gpu.bind(
            &gate.group,
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
            gate,
            topology,
            solve_bind,
            stress_bind,
            topology_bind,
            has_warm_start: false,
            wide: None,
        };
        let mut s = s;
        if native {
            s.wide = Some(s.build_wide(gpu, wide_min));
        }
        s.write_params(&s.params(false, false, 0.0, false));
        s
    }

    fn build_wide(&self, gpu: &Gpu, min: u32) -> Wide {
        let (n, m) = (self.graph.node_count().max(1), self.graph.bond_count().max(1));

        let rows = (n as u32 / min.max(1)) + 1;
        let blocks = (n.max(m).div_ceil(WIDE_THREADS) + 1) as u32;
        let words = (rows * WIDE_ROW_WORDS + rows * blocks * WIDE_PARTIAL) as usize;
        let wide = gpu.storage("blast wide", &vec![[0.0f32; 4]; words]);
        let meta = gpu.storage("blast wide meta", &vec![0u32; (WIDE_META_ROWS + rows) as usize]);
        let dispatch = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blast wide dispatch"),
            size: 48,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b = &self.buffers;
        let buffers = [&b.params, &b.islands, &b.nodes, &b.refs, &b.bonds, &b.accel, &b.x, &b.scratch, &b.report, &b.lists, &wide, &meta, &b.args];
        let kernels = WIDE_SETUP
            .iter()
            .chain(WIDE_ITERATION)
            .chain(WIDE_END)
            .map(|&(e, g)| {
                let k = gpu.compute(&shaders::BLAST_NATIVE_WIDE, e, &WIDE_BINDINGS);
                let bind = gpu.bind(&k.group, &buffers);
                (k, g, bind)
            })
            .collect();
        Wide { kernels, meta, dispatch, rows, blocks, min }
    }

    /// The parameters of a solve (`blast_common.slang` `BlastParams`); `warm` as the settings
    /// allow it, once a solve has run.
    pub fn params(&self, warm: bool, skip_settled: bool, dt: f64, apply_damage: bool) -> Params {
        let g = &self.graph;
        let s = &g.settings;
        Params {
            // The dispatch is sized on the GPU (`args`); this only bounds it.
            island_count: g.node_count() as u32,
            max_iterations: s.max_iterations,
            warm: (warm && s.warm_start && self.has_warm_start) as u32,
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
            force_tolerance: s.force_tolerance as f32,
            wide_min: self.wide.as_ref().map_or(u32::MAX, |w| w.min),
            wide_rows: self.wide.as_ref().map_or(0, |w| w.rows),
            wide_blocks: self.wide.as_ref().map_or(0, |w| w.blocks),
            ..Default::default()
        }
    }

    pub fn write_params(&self, p: &Params) {
        self.gpu
            .queue
            .write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(p));
    }

    /// Record the islands' rebuild from the bonds' health. It runs only when a bond broke since
    /// the last one (gated on the GPU: nothing is read back).
    pub fn record_topology(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gate.pipeline);
            pass.set_bind_group(0, &self.topology_bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for (i, (k, at)) in self.topology.iter().enumerate() {
            // The gate sized the rebuild; bt_scan_top then sizes the sort (and the solve).
            if i == 0 || *at == A_SORT {
                encoder.copy_buffer_to_buffer(
                    &self.buffers.args,
                    0,
                    &self.buffers.dispatch,
                    0,
                    ARGS_BYTES,
                );
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &self.topology_bind, &[]);
            pass.dispatch_workgroups_indirect(&self.buffers.dispatch, 4 * at);
        }
    }

    /// Record the solve of every island (one group each, sized on the GPU).
    pub fn record_solve(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.solve.pipeline);
            pass.set_bind_group(0, &self.solve_bind, &[]);
            pass.dispatch_workgroups_indirect(&self.buffers.dispatch, 4 * A_SOLVE);
        }
        if let Some(w) = &self.wide {
            self.record_wide(encoder, w);
        }
    }

    /// The large islands' solve: set-up (their rows listed, then the sizes copied for the
    /// indirect dispatches), `max_iterations` iterations, the end.
    fn record_wide(&self, encoder: &mut wgpu::CommandEncoder, w: &Wide) {
        let setup = WIDE_SETUP.len();
        let iteration = WIDE_ITERATION.len();
        let run = |pass: &mut wgpu::ComputePass<'_>, (k, grid, bind): &(Kernel, Grid, wgpu::BindGroup)| {
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, bind, &[]);
            match grid {
                Grid::One => pass.dispatch_workgroups(1, 1, 1),
                Grid::Islands => pass.dispatch_workgroups(crate::gpu::groups(self.graph.node_count().max(1), k.workgroup), 1, 1),
                Grid::Nodes => pass.dispatch_workgroups_indirect(&w.dispatch, 0),
                Grid::Bonds => pass.dispatch_workgroups_indirect(&w.dispatch, 16),
                Grid::Rows => pass.dispatch_workgroups_indirect(&w.dispatch, 32),
            }
        };
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for k in &w.kernels[..3] {
                run(&mut pass, k);
            }
        }
        // meta words 4..16: the node, bond and row grids.
        encoder.copy_buffer_to_buffer(&w.meta, 16, &w.dispatch, 0, 48);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for k in &w.kernels[3..setup] {
            run(&mut pass, k);
        }
        for _ in 0..self.graph.settings.max_iterations {
            for k in &w.kernels[setup..setup + iteration] {
                run(&mut pass, k);
            }
        }
        for k in &w.kernels[setup + iteration..] {
            run(&mut pass, k);
        }
    }

    /// Record every bond's stresses and damage (and, with the params' `apply_damage`, the
    /// damage taken from its health).
    pub fn record_stress(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        self.stress
            .dispatch(&mut pass, &self.stress_bind, self.graph.bond_count().max(1));
    }

    /// A solve has been recorded (the next one warm-starts from it).
    pub fn mark_solved(&mut self) {
        self.has_warm_start = true;
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
        self.write_params(&self.params(true, skip_settled, 0.0, false));
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.record_topology(&mut encoder);
        self.record_solve(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.mark_solved();
        self.reports()
    }

    /// The islands' reports of the last solve, by island.
    pub fn reports(&self) -> Vec<IslandReport> {
        let report: Vec<[u32; 4]> =
            self.gpu
                .read_range(&self.buffers.report, 0, self.island_count());
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
        self.gpu.read_range::<u32>(&self.buffers.args, A_COUNT, 1)[0] as usize
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
        self.write_params(&self.params(true, false, dt, apply_damage));
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.record_stress(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
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
        let dirty = 4 * (topo_total(self.graph.node_count().max(1)) + 4) as u64;
        self.gpu
            .queue
            .write_buffer(&self.buffers.topo, dirty, bytemuck::bytes_of(&1u32));
    }
}

/// What [`BlastFrame`] reads and writes of the GPU fracture (`fracture.slang`), and where its
/// `work` and `atom` keep the bonds' alive flags, dirty marks and counters.
pub struct FractureBuffers<'a> {
    pub work: &'a wgpu::Buffer,
    pub atom: &'a wgpu::Buffer,
    pub frames: &'a wgpu::Buffer,
    pub bodies: &'a wgpu::Buffer,
    pub loads: &'a wgpu::Buffer,
    pub w_alive: u32,
    pub a_dirty: u32,
    pub a_counters: u32,
}

/// Blast's solver in the GPU fracture's frame (`Profile::BlastGpu`): each frame, after the
/// fracture's loads, Blast's inputs, its islands (when a bond broke), the solve, the stresses
/// and damage, and the bonds that broke marked in the fracture for its split. The fracture's
/// chunks and bonds are the graph's nodes and bonds, index for index.
pub struct BlastFrame {
    pub solver: GpuBlastSolver,
    node_frame: wgpu::Buffer,
    loads: Kernel,
    breaks: Kernel,
    bind: Option<wgpu::BindGroup>,
    gravity: Vec3,
    w_alive: u32,
    a_dirty: u32,
    a_counters: u32,
}

impl BlastFrame {
    pub fn new(
        gpu: &Gpu,
        graph: BlastGraph,
        materials: Vec<BlastMaterial>,
        gravity: Vec3,
    ) -> BlastFrame {
        let node_frame: Vec<[f32; 4]> = graph
            .nodes
            .iter()
            .map(|n| {
                [
                    n.position.x as f32,
                    n.position.y as f32,
                    n.position.z as f32,
                    n.mass as f32,
                ]
            })
            .collect();
        let node_frame = gpu.storage(
            "blast node frame",
            &if node_frame.is_empty() {
                vec![[0.0f32; 4]]
            } else {
                node_frame
            },
        );
        let loads = gpu.compute(&shaders::BLAST_FRACTURE, "blast_loads", &FRACTURE_BINDINGS);
        let breaks = gpu.compute(&shaders::BLAST_FRACTURE, "blast_breaks", &FRACTURE_BINDINGS);
        let solver = GpuBlastSolver::new(gpu, graph, materials);
        BlastFrame {
            solver,
            node_frame,
            loads,
            breaks,
            bind: None,
            gravity,
            w_alive: 0,
            a_dirty: 0,
            a_counters: 0,
        }
    }

    /// Bind the fracture's buffers (again whenever they change).
    pub fn bind(&mut self, gpu: &Gpu, f: &FractureBuffers) {
        let b = &self.solver.buffers;
        self.bind = Some(gpu.bind(
            &self.loads.group,
            &[
                &b.params,
                f.work,
                f.atom,
                f.frames,
                f.bodies,
                f.loads,
                &self.node_frame,
                &b.accel,
                &b.bonds,
            ],
        ));
        (self.w_alive, self.a_dirty, self.a_counters) = (f.w_alive, f.a_dirty, f.a_counters);
    }

    /// Record the frame's Blast work after `fr_loads` (`dt`: the frame's duration, for the
    /// gradual damage), before the fracture's gate and split.
    pub fn record(&mut self, encoder: &mut wgpu::CommandEncoder, dt: f64) {
        let bind = self.bind.as_ref().expect("BlastFrame::bind before record");
        let mut p = self.solver.params(true, true, dt, true);
        let g = self.gravity;
        p.gravity = [g.x as f32, g.y as f32, g.z as f32, 0.0];
        (p.w_alive, p.a_dirty, p.a_counters) = (self.w_alive, self.a_dirty, self.a_counters);
        self.solver.write_params(&p);
        let (n, m) = (
            self.solver.graph.node_count(),
            self.solver.graph.bond_count(),
        );
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.loads.dispatch(&mut pass, bind, n.max(1));
        }
        self.solver.record_topology(encoder);
        self.solver.record_solve(encoder);
        self.solver.record_stress(encoder);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.breaks.dispatch(&mut pass, bind, m.max(1));
        }
        self.solver.mark_solved();
    }
}
