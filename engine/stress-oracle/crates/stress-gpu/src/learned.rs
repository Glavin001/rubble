//! The learned impact-outcome model's engine side (`STRESS_IMPACT_PUNCH=learned`, off by
//! default; docs/learned-stress-solver/learned-stress-solver-plan.md, P3): the buffers a
//! call batch lives in and the kernels of `shaders/learned.slang` that fill them, decode
//! the network's outputs and apply them, in the resident frame's one submission.
//!
//! Layouts are `learned.slang`'s (its header comment); this module mirrors them.

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

/// float4 words per call, node and bond (`learned.slang`).
pub const CALL_F4: usize = 8;
pub const NODE_F4: usize = 12;
pub const BOND_F4: usize = 2;

/// `calls[2].y` flags.
pub const CALL_FREE: u32 = 1;
pub const CALL_EXTERNAL: u32 = 2;
/// `calls[4].z` decode flags.
pub const DEC_BOUND_ACTIVE: u32 = 1;
pub const DEC_BOUND_INFEASIBLE: u32 = 2;
pub const DEC_NONFINITE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DecodeParams {
    call_count: u32,
    pad0: u32,
    log_tau: f32,
    tolerance: f32,
}

const DECODE_BINDINGS: [Binding; 4] = [Binding::Uniform, Binding::StorageRw, Binding::StorageRw, Binding::StorageRw];

/// The decode's configuration (P2's `DecodeConfig`; from the weights' manifest).
#[derive(Clone, Copy, Debug)]
pub struct DecodeConfig {
    /// τ on the failure margin: a bond breaks when q̂ ≥ τ.
    pub tau: f64,
    /// The KE bound's tolerance, a fraction of E_in (P0: the ledger residual reaches 4.7%).
    pub energy_tolerance: f64,
}

impl Default for DecodeConfig {
    fn default() -> DecodeConfig {
        DecodeConfig { tau: 1.0, energy_tolerance: env_f64("STRESS_LEARNED_ENERGY_TOL").unwrap_or(0.05) }
    }
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// One node's inputs for the decode (a chunk or an impactor), world axes, Z-up.
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeIn {
    /// Relative to the call's centre of mass.
    pub pos: [f32; 3],
    pub mass: f32,
    /// x, y, z, w.
    pub q: [f32; 4],
    pub inertia: [f32; 3],
    pub anchored: bool,
    pub v: [f32; 3],
    pub w: [f32; 3],
    /// The network's 12 body-frame outputs.
    pub out: [f32; 12],
}

/// One call's inputs for the decode.
#[derive(Clone, Debug, Default)]
pub struct CallIn {
    /// L, m, v, c, f_t, E, GA, dt, E_in (`scales`, the calls schema's order).
    pub scales: [f32; 9],
    pub gravity: [f32; 3],
    pub flags: u32,
    pub chunks: usize,
    pub nodes: Vec<NodeIn>,
    /// Per live bond: damage at frame start, log q̂, Δdamage, time bin (1..8).
    pub bonds: Vec<(f32, f32, f32, u32)>,
}

/// Packs calls in `learned.slang`'s layouts: (calls, nodes, bonds) float4 words.
pub fn pack(calls: &[CallIn]) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let (mut cw, mut nw, mut bw) = (Vec::new(), Vec::new(), Vec::new());
    let (mut first, mut bfirst) = (0u32, 0u32);
    let b = f32::from_bits;
    for (g, c) in calls.iter().enumerate() {
        let s = &c.scales;
        cw.push([s[0], s[1], s[2], s[3]]);
        cw.push([s[4], s[5], s[6], s[7]]);
        cw.push([s[8], b(c.flags), b(first), b(c.nodes.len() as u32)]);
        cw.push([c.gravity[0], c.gravity[1], c.gravity[2], b(c.chunks as u32)]);
        cw.push([b(bfirst), b(c.bonds.len() as u32), 0.0, 0.0]);
        cw.extend([[0.0; 4]; CALL_F4 - 5]);
        for (j, n) in c.nodes.iter().enumerate() {
            let kind = u32::from(j >= c.chunks);
            nw.push([n.pos[0], n.pos[1], n.pos[2], n.mass]);
            nw.push(n.q);
            nw.push([n.inertia[0], n.inertia[1], n.inertia[2], f32::from(u8::from(n.anchored))]);
            nw.push([n.v[0], n.v[1], n.v[2], b(g as u32)]);
            nw.push([n.w[0], n.w[1], n.w[2], b(kind)]);
            let o = &n.out;
            nw.push([o[0], o[1], o[2], o[3]]);
            nw.push([o[4], o[5], o[6], o[7]]);
            nw.push([o[8], o[9], o[10], o[11]]);
            nw.extend([[0.0; 4]; 4]);
        }
        for &(d0, lq, dd, bin) in &c.bonds {
            bw.push([d0, lq, dd, b(bin)]);
            bw.push([0.0, 0.0, b(u32::MAX), 0.0]);
        }
        first += c.nodes.len() as u32;
        bfirst += c.bonds.len() as u32;
    }
    (cw, nw, bw)
}

/// One node's decoded frame-end state (world, Z-up).
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeOut {
    pub v: [f32; 3],
    pub w: [f32; 3],
    pub dx: [f32; 3],
    pub drot: [f32; 3],
}

/// The decode kernel (`learned_decode`).
pub struct Decode {
    kernel: Kernel,
}

impl Decode {
    pub fn new(gpu: &Gpu) -> Decode {
        Decode { kernel: gpu.compute(&shaders::LEARNED, "learned_decode", &DECODE_BINDINGS) }
    }

    /// One group per call, into `encoder`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, cfg: &DecodeConfig, call_count: u32, calls: &wgpu::Buffer, nodes: &wgpu::Buffer, bonds: &wgpu::Buffer) {
        let params = gpu.uniform("learned decode params", &DecodeParams { call_count, pad0: 0, log_tau: cfg.tau.ln() as f32, tolerance: cfg.energy_tolerance as f32 });
        let group = gpu.bind(&self.kernel.group, &[&params, calls, nodes, bonds]);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(call_count.max(1), 1, 1);
    }

    /// The decode of `calls` on their own (tests): per call, its nodes' decoded state, its
    /// bonds' (broke, damage) and its decode flags.
    pub fn run(&self, gpu: &Gpu, cfg: &DecodeConfig, calls: &[CallIn]) -> Vec<(Vec<NodeOut>, Vec<(bool, f32)>, u32)> {
        let (cw, nw, bw) = pack(calls);
        let one = [[0f32; 4]];
        let cb = gpu.storage("learned calls", &cw);
        let nb = gpu.storage("learned nodes", if nw.is_empty() { &one[..] } else { &nw });
        let bb = gpu.storage("learned bonds", if bw.is_empty() { &one[..] } else { &bw });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.encode(gpu, &mut enc, cfg, calls.len() as u32, &cb, &nb, &bb);
        gpu.queue.submit([enc.finish()]);
        let cr: Vec<[f32; 4]> = gpu.read(&cb);
        let nr: Vec<[f32; 4]> = gpu.read(&nb);
        let br: Vec<[f32; 4]> = gpu.read(&bb);
        let (mut first, mut bfirst) = (0usize, 0usize);
        let mut out = Vec::new();
        for (g, c) in calls.iter().enumerate() {
            let nodes = (0..c.nodes.len())
                .map(|j| {
                    let k = NODE_F4 * (first + j);
                    let t = |o: usize| [nr[k + o][0], nr[k + o][1], nr[k + o][2]];
                    NodeOut { v: t(8), w: t(9), dx: t(10), drot: t(11) }
                })
                .collect();
            let bonds = (0..c.bonds.len()).map(|j| {
                let w = br[BOND_F4 * (bfirst + j) + 1];
                (w[1].to_bits() != 0, w[0])
            }).collect();
            out.push((nodes, bonds, cr[CALL_F4 * g + 4][2].to_bits()));
            first += c.nodes.len();
            bfirst += c.bonds.len();
        }
        out
    }
}
