//! The learned impact-outcome model's engine side (`STRESS_IMPACT_PUNCH=learned`, off by
//! default; docs/learned-stress-solver/learned-stress-solver-plan.md, P3): the buffers a
//! call batch lives in and the kernels of `shaders/learned.slang` that fill them, decode
//! the network's outputs and apply them, in the resident frame's one submission. The
//! batch's control words come back with the frame's one read (`Learned::range`).
//!
//! Layouts are `learned.slang`'s (its header comment); this module mirrors them.
//!
//! Configuration (explicit; a batch over a capacity fails the frame with a clear error,
//! never a silent drop):
//! - `STRESS_LEARNED_STUB=zero|punch`: the network's place until its weights are wired
//!   (`zero`: every output 0; `punch`: the punch's plug as an inelastic outcome);
//! - `STRESS_LEARNED_ENERGY_TOL` (0.05): the KE bound's tolerance, a fraction of E_in;
//! - `STRESS_LEARNED_MAX_CALLS` (64), `_MAX_NODES` (65536), `_MAX_BONDS` (262144),
//!   `_MAX_CALL_ISLANDS` (2048), `_MAX_CALL_CHUNKS` (4096: the scales' group sort);
//! - `STRESS_DEBRIS_ROUTING=1`: islands the launched debris reaches join the call
//!   (as the host's routing, `World::debris_reach_routing`).

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

/// float4 words per call, node and bond; uint4 words per call island; float4 per impactor.
pub const CALL_F4: usize = 8;
pub const NODE_F4: usize = 14;
pub const BOND_F4: usize = 2;
pub const CISL_U4: usize = 2;
pub const IMP_F4: usize = 3;

/// `calls[2].y` flags.
pub const CALL_FREE: u32 = 1;
pub const CALL_EXTERNAL: u32 = 2;
pub const CALL_ANCHORED: u32 = 4;
/// `calls[4].z` decode flags.
pub const DEC_BOUND_ACTIVE: u32 = 1;
pub const DEC_BOUND_INFEASIBLE: u32 = 2;
pub const DEC_NONFINITE: u32 = 4;

/// Control words (`learned.slang` `CTL_*`).
pub mod ctl {
    pub const PENDING: usize = 0;
    pub const CALLS: usize = 1;
    pub const NODES: usize = 2;
    pub const BONDS: usize = 3;
    pub const FALLBACKS: usize = 5;
    pub const BOUND: usize = 6;
    pub const INFEASIBLE: usize = 7;
    pub const OVERFLOW: usize = 8;
    pub const CISL: usize = 9;
    pub const STALE: usize = 10;
    pub const APPLIED: usize = 11;
    pub const BROKEN: usize = 12;
    pub const FB_NONFINITE: usize = 14;
    pub const FB_INFEASIBLE: usize = 15;
    pub const FB_RANGE: usize = 16;
    pub const ROUTED: usize = 17;
    pub const WORDS: usize = 32;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    impactor_count: u32,
    engine_base: u32,
    island_cap: u32,
    stub: u32,
    dt: f32,
    speed: f32,
    log_tau: f32,
    tolerance: f32,
    gravity: [f32; 4],
    max_calls: u32,
    max_nodes: u32,
    max_bonds: u32,
    max_call_chunks: u32,
    debris: u32,
    max_cisl: u32,
    chunk_count: u32,
    pad0: u32,
}

const BINDINGS: [Binding; 28] = [
    Binding::Uniform,   // 0 params
    Binding::StorageRw, // 1 ctl
    Binding::StorageRw, // 2 calls
    Binding::StorageRw, // 3 nodes
    Binding::StorageRw, // 4 bonds
    Binding::StorageRw, // 5 isl
    Binding::StorageRw, // 6 islf
    Binding::StorageRw, // 7 cisl
    Binding::StorageRw, // 8 chunk_node
    Binding::StorageRw, // 9 frozen
    Binding::Storage,   // 10 imps
    Binding::Storage,   // 11 islands
    Binding::Storage,   // 12 chunk statics
    Binding::Storage,   // 13 chunk state
    Binding::Storage,   // 14 ctab
    Binding::StorageRw, // 15 engine bodies
    Binding::Storage,   // 16 bond statics
    Binding::StorageRw, // 17 bond records
    Binding::StorageRw, // 18 cascade status
    Binding::Storage,   // 19 split counters
    Binding::Storage,   // 20 split parents
    Binding::Storage,   // 21 split work
    Binding::Storage,   // 22 split slots
    Binding::Storage,   // 23 split control
    Binding::Storage,   // 24 split firsts
    Binding::Storage,   // 25 routing flag
    Binding::Storage,   // 26 split child records
    Binding::StorageRw, // 27 indirect arguments
];

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

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// What stands in for the network (`STRESS_LEARNED_STUB`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stub {
    /// Every output 0: ballistic motion, no break.
    Zero = 1,
    /// The punch's plug as an outcome: its bonds broken, the plug and the impactor at their
    /// common (inelastic) velocity.
    Punch = 2,
    /// NaN outputs: every call falls back to the punch (a test of the gate).
    NonFinite = 3,
}

impl Stub {
    /// `STRESS_LEARNED_STUB`, if set (an unknown value is an error).
    pub fn from_env() -> Result<Option<Stub>, String> {
        match std::env::var("STRESS_LEARNED_STUB").as_deref() {
            Err(_) => Ok(None),
            Ok("zero") => Ok(Some(Stub::Zero)),
            Ok("punch") => Ok(Some(Stub::Punch)),
            Ok("nonfinite") => Ok(Some(Stub::NonFinite)),
            Ok(other) => Err(format!("STRESS_LEARNED_STUB={other}: expected zero, punch or nonfinite")),
        }
    }
}

/// One impactor as the batch sees it (uploaded when the list changes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactorInfo {
    pub body: u32,
    /// Bounding radius about the body's centre (the routing check's).
    pub bound: f32,
    /// Face radius (`World::impactor_radius`).
    pub face: f32,
    pub mass: f32,
    /// Principal inertia (body frame).
    pub inertia: [f32; 3],
    /// Plane-strain modulus as the solver scales it.
    pub modulus: f32,
    /// |half extents| (the rim speed's lever arm).
    pub half_norm: f32,
}

/// The solver's and engine's tables a batch reads and writes, for one frame.
pub struct Tables<'a> {
    pub islands: &'a wgpu::Buffer,
    pub chunks: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub ctab: &'a wgpu::Buffer,
    pub bodies: &'a wgpu::Buffer,
    pub bond_static: &'a wgpu::Buffer,
    pub bond_dyn: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    /// The split pass's counters (`[0]` live islands).
    pub counters: &'a wgpu::Buffer,
    /// The routing check's flag.
    pub route: &'a wgpu::Buffer,
    /// Indirect arguments over the live islands (one group each), if any.
    pub island_args: Option<&'a wgpu::Buffer>,
    pub s: Scalars,
}

/// The frame's scalars a batch's kernels take.
#[derive(Clone, Copy, Debug)]
pub struct Scalars {
    pub island_cap: u32,
    pub engine_base: u32,
    pub dt: f64,
    pub speed: f64,
    pub gravity: [f32; 3],
}

struct Kernels {
    clear: Kernel,
    select: Kernel,
    debris: Kernel,
    count_bonds: Kernel,
    compact: Kernel,
    gather_chunks: Kernel,
    gather_bonds: Kernel,
    gather_impactors: Kernel,
    finish: Kernel,
    stub: Kernel,
    decode: Kernel,
    gate: Kernel,
    tally: Kernel,
    check: Kernel,
    apply_bonds: Kernel,
    apply_bodies: Kernel,
    apply_impactors: Kernel,
    tally_apply: Kernel,
    place: Kernel,
}

/// Running totals over the run (logged; tests read them).
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub batches: u64,
    pub calls: u64,
    pub fallbacks: u64,
    pub fallback_nonfinite: u64,
    pub fallback_infeasible: u64,
    pub fallback_range: u64,
    pub bound_active: u64,
    pub applied: u64,
    pub stale: u64,
    pub broken: u64,
    pub routed_islands: u64,
}

pub struct Learned {
    k: Kernels,
    pub cfg: DecodeConfig,
    pub stub: Stub,
    debris: bool,
    max_calls: u32,
    max_nodes: u32,
    max_bonds: u32,
    max_cisl: u32,
    max_call_chunks: u32,
    ctl: wgpu::Buffer,
    calls: wgpu::Buffer,
    nodes: wgpu::Buffer,
    bonds: wgpu::Buffer,
    isl: wgpu::Buffer,
    islf: wgpu::Buffer,
    cisl: wgpu::Buffer,
    chunk_node: wgpu::Buffer,
    frozen: wgpu::Buffer,
    imps: wgpu::Buffer,
    largs: wgpu::Buffer,
    dummy: wgpu::Buffer,
    /// Distinct stand-ins for read-write slots a kernel does not use (the Metal path binds
    /// the module's whole group 0; one buffer may not be read-write twice in a dispatch).
    dummy_rw: [wgpu::Buffer; 3],
    imp_list: Vec<ImpactorInfo>,
    island_cap: u32,
    chunk_count: u32,
    /// A batch stood at the last read: its window end is due this frame.
    pub pending: bool,
    pub totals: Totals,
}

fn rw(gpu: &Gpu, label: &str, bytes: u64, indirect: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    if indirect {
        usage |= wgpu::BufferUsages::INDIRECT;
    }
    let b = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage, mapped_at_creation: false });
    // Zeroed (wgpu zero-initialises new buffers; explicit for clarity of intent).
    b
}

impl Learned {
    pub fn new(gpu: &Gpu, stub: Stub) -> Learned {
        let k = |e: &str| gpu.compute(&shaders::LEARNED, e, &BINDINGS);
        let kernels = Kernels {
            clear: k("learned_clear"),
            select: k("learned_select"),
            debris: k("learned_debris"),
            count_bonds: k("learned_count_bonds"),
            compact: k("learned_compact"),
            gather_chunks: k("learned_gather_chunks"),
            gather_bonds: k("learned_gather_bonds"),
            gather_impactors: k("learned_gather_impactors"),
            finish: k("learned_finish"),
            stub: k("learned_stub"),
            decode: k("learned_decode"),
            gate: k("learned_gate"),
            tally: k("learned_tally"),
            check: k("learned_check"),
            apply_bonds: k("learned_apply_bonds"),
            apply_bodies: k("learned_apply_bodies"),
            apply_impactors: k("learned_apply_impactors"),
            tally_apply: k("learned_tally_apply"),
            place: k("learned_place"),
        };
        let max_calls = env_u32("STRESS_LEARNED_MAX_CALLS", 64).min(64);
        let max_nodes = env_u32("STRESS_LEARNED_MAX_NODES", 65536);
        let max_bonds = env_u32("STRESS_LEARNED_MAX_BONDS", 262144);
        let max_cisl = env_u32("STRESS_LEARNED_MAX_CALL_ISLANDS", 2048);
        let max_call_chunks = env_u32("STRESS_LEARNED_MAX_CALL_CHUNKS", 4096).min(4096);
        Learned {
            k: kernels,
            cfg: DecodeConfig::default(),
            stub,
            debris: std::env::var("STRESS_DEBRIS_ROUTING").is_ok_and(|v| v == "1"),
            max_calls,
            max_nodes,
            max_bonds,
            max_cisl,
            max_call_chunks,
            ctl: rw(gpu, "learned ctl", 4 * ctl::WORDS as u64, false),
            calls: rw(gpu, "learned calls", 16 * (CALL_F4 * max_calls as usize) as u64, false),
            nodes: rw(gpu, "learned nodes", 16 * (NODE_F4 * max_nodes as usize) as u64, false),
            bonds: rw(gpu, "learned bonds", 16 * (BOND_F4 * max_bonds as usize) as u64, false),
            isl: rw(gpu, "learned islands", 16, false),
            islf: rw(gpu, "learned island bounds", 16, false),
            cisl: rw(gpu, "learned call islands", 16 * (CISL_U4 * max_cisl as usize) as u64, false),
            chunk_node: rw(gpu, "learned chunk map", 16, false),
            frozen: rw(gpu, "learned held bodies", 16 * 4 * (max_cisl as u64 + 32), false),
            imps: rw(gpu, "learned impactors", 16 * IMP_F4 as u64, false),
            largs: rw(gpu, "learned indirect arguments", 16 * 4, true),
            dummy: rw(gpu, "learned unbound", 16, false),
            dummy_rw: [rw(gpu, "learned unbound rw 0", 16, false), rw(gpu, "learned unbound rw 1", 16, false), rw(gpu, "learned unbound rw 2", 16, false)],
            imp_list: Vec::new(),
            island_cap: 0,
            chunk_count: 0,
            pending: false,
            totals: Totals::default(),
        }
    }

    /// The impactors (uploaded when they change).
    pub fn set_impactors(&mut self, gpu: &Gpu, list: &[ImpactorInfo]) {
        if list == self.imp_list.as_slice() {
            return;
        }
        let words: Vec<[f32; 4]> = list
            .iter()
            .flat_map(|i| [[f32::from_bits(i.body), i.bound, i.face, i.mass], [i.inertia[0], i.inertia[1], i.inertia[2], i.modulus], [i.half_norm, 0.0, 0.0, 0.0]])
            .collect();
        self.imps = if words.is_empty() { gpu.storage("learned impactors", &[[0f32; 4]; IMP_F4]) } else { gpu.storage("learned impactors", &words) };
        self.imp_list = list.to_vec();
    }

    /// Room for the tables' islands and chunks (the per-island and per-chunk buffers).
    /// Growing the chunk map drops a pending batch's map: only between batches.
    pub fn ensure(&mut self, gpu: &Gpu, island_cap: u32, chunk_count: u32) {
        if island_cap > self.island_cap {
            self.isl = rw(gpu, "learned islands", 16 * island_cap as u64, false);
            self.islf = rw(gpu, "learned island bounds", 16 * 3 * island_cap as u64, false);
            self.island_cap = island_cap;
        }
        if chunk_count > self.chunk_count {
            assert!(!self.pending, "the chunk map grew under a pending learned batch");
            self.chunk_node = rw(gpu, "learned chunk map", 4 * chunk_count as u64, false);
            self.chunk_count = chunk_count;
        }
    }

    fn params(&self, gpu: &Gpu, t: &Scalars) -> wgpu::Buffer {
        let g = t.gravity;
        gpu.uniform(
            "learned params",
            &Params {
                impactor_count: self.imp_list.len() as u32,
                engine_base: t.engine_base,
                island_cap: t.island_cap,
                stub: self.stub as u32,
                dt: t.dt as f32,
                speed: (t.speed * (1.0 - 1e-3)) as f32,
                log_tau: self.cfg.tau.ln() as f32,
                tolerance: self.cfg.energy_tolerance as f32,
                gravity: [g[0], g[1], g[2], (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()],
                max_calls: self.max_calls,
                max_nodes: self.max_nodes,
                max_bonds: self.max_bonds,
                max_call_chunks: self.max_call_chunks,
                debris: self.debris as u32,
                max_cisl: self.max_cisl,
                chunk_count: self.chunk_count,
                pad0: 0,
            },
        )
    }

    fn buffers<'a>(&'a self, params: &'a wgpu::Buffer, t: &'a Tables, split: [&'a wgpu::Buffer; 6]) -> [&'a wgpu::Buffer; 28] {
        let [parents, work, slots, control, firsts, out] = split;
        [
            // The indirect arguments only where `learned_compact` writes them (a dispatch that
            // reads them as its arguments may not also bind them read-write).
            params, &self.ctl, &self.calls, &self.nodes, &self.bonds, &self.isl, &self.islf, &self.cisl, &self.chunk_node, &self.frozen, &self.imps, t.islands, t.chunks, t.state, t.ctab, t.bodies, t.bond_static, t.bond_dyn, t.status, t.counters, parents, work, slots, control, firsts, t.route, out, &self.dummy_rw[2],
        ]
    }

    fn no_split(&self) -> [&wgpu::Buffer; 6] {
        [&self.dummy, &self.dummy, &self.dummy, &self.dummy, &self.dummy, &self.dummy]
    }

    /// The window end of the batch that stood at the last read (`pending`), in the `post`
    /// encoder before the split pass: its breaks, damage and bodies (`learned_place` places
    /// the children in the split pass: `place_bind`).
    pub fn encode_apply(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, t: &Tables) {
        if !self.pending {
            return;
        }
        let params = self.params(gpu, &t.s);
        let bufs = self.buffers(&params, t, self.no_split());
        let mut pass = enc.begin_compute_pass(&Default::default());
        for (k, indirect) in [(&self.k.check, Some(0u64)), (&self.k.apply_bonds, Some(0)), (&self.k.apply_bodies, Some(32)), (&self.k.apply_impactors, Some(0)), (&self.k.tally_apply, None)] {
            let group = gpu.bind(&k.group, &bufs);
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &group, &[]);
            match indirect {
                Some(at) => pass.dispatch_workgroups_indirect(&self.largs, at),
                None => pass.dispatch_workgroups(1, 1, 1),
            }
        }
    }

    /// The split pass's hook (`learned_place`): its bind group from the pass's buffers.
    pub fn place_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.k.place.pipeline
    }

    pub fn place_bind(&self, gpu: &Gpu, sc: &Scalars, bodies: &wgpu::Buffer, pass: &crate::gpu_pass::GpuPass) -> wgpu::BindGroup {
        let params = self.params(gpu, sc);
        let d = &self.dummy;
        let t = Tables { islands: d, chunks: d, state: d, ctab: d, bodies, bond_static: d, bond_dyn: &self.dummy_rw[0], status: &self.dummy_rw[1], counters: &pass.counters, route: d, island_args: None, s: *sc };
        let bufs = self.buffers(&params, &t, [&pass.parents, &pass.work, &pass.slots, &pass.control, &pass.firsts, &pass.out]);
        gpu.bind(&self.k.place.group, &bufs)
    }

    /// After the split pass, in the same submission: the last batch's map cleared (if one
    /// was pending), then with `batch` a new batch: select, compact, gather, the network
    /// (`network`, or the stub), decode, gate and tally. The routing check runs before it.
    pub fn encode_batch(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, t: &Tables, batch: bool, network: Option<&dyn Fn(&mut wgpu::CommandEncoder)>) {
        let params = self.params(gpu, &t.s);
        let bufs = self.buffers(&params, t, self.no_split());
        let mut bufs_compact = bufs;
        bufs_compact[27] = &self.largs;
        enum D {
            One,
            Islands,
            Threads(u32),
            Indirect(u64),
        }
        let run = |enc: &mut wgpu::CommandEncoder, k: &Kernel, d: D| {
            let group = gpu.bind(&k.group, if std::ptr::eq(k, &self.k.compact) { &bufs_compact } else { &bufs });
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &group, &[]);
            match d {
                D::One => pass.dispatch_workgroups(1, 1, 1),
                D::Islands => match t.island_args {
                    Some(a) => pass.dispatch_workgroups_indirect(a, 0),
                    None => pass.dispatch_workgroups(t.s.island_cap.max(1), 1, 1),
                },
                D::Threads(n) => pass.dispatch_workgroups(n.div_ceil(64).max(1), 1, 1),
                D::Indirect(at) => pass.dispatch_workgroups_indirect(&self.largs, at),
            }
        };
        if self.pending {
            run(enc, &self.k.clear, D::One);
        }
        if !batch {
            return;
        }
        run(enc, &self.k.select, D::Islands);
        if self.debris {
            run(enc, &self.k.debris, D::Threads(t.s.island_cap));
        }
        run(enc, &self.k.count_bonds, D::Islands);
        run(enc, &self.k.compact, D::One);
        run(enc, &self.k.gather_chunks, D::Indirect(16));
        run(enc, &self.k.gather_bonds, D::Indirect(16));
        run(enc, &self.k.gather_impactors, D::Indirect(0));
        run(enc, &self.k.finish, D::Indirect(0));
        match network {
            Some(net) => net(enc),
            None => run(enc, &self.k.stub, D::Indirect(0)),
        }
        run(enc, &self.k.decode, D::Indirect(0));
        run(enc, &self.k.gate, D::Indirect(0));
        run(enc, &self.k.tally, D::One);
    }

    /// The control words' range (for the frame's one read).
    pub fn range(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.ctl, 0, 4 * ctl::WORDS as u64)
    }

    /// The frame's read of the control words: `applied` / `batched` say what this frame
    /// recorded. Every fallback is logged; a batch over a capacity is an error.
    pub fn follow(&mut self, bytes: &[u8], frame: u64, applied: bool, batched: bool) -> Result<(), String> {
        let w: &[u32] = bytemuck::cast_slice(&bytes[..4 * ctl::WORDS]);
        if applied {
            self.totals.applied += w[ctl::APPLIED] as u64;
            self.totals.stale += w[ctl::STALE] as u64;
            self.totals.broken += w[ctl::BROKEN] as u64;
            if w[ctl::STALE] > 0 {
                eprintln!("learned impact frame {frame}: {} call(s) not applied at their window end: their islands changed under them (bodies released with their frame-start motion)", w[ctl::STALE]);
            }
        }
        if batched {
            if w[ctl::OVERFLOW] != 0 {
                let o = w[ctl::OVERFLOW];
                let what: Vec<&str> = [(1, "calls (STRESS_LEARNED_MAX_CALLS)"), (2, "nodes (STRESS_LEARNED_MAX_NODES)"), (4, "bonds (STRESS_LEARNED_MAX_BONDS)"), (8, "call islands (STRESS_LEARNED_MAX_CALL_ISLANDS)"), (16, "chunks in one call (STRESS_LEARNED_MAX_CALL_CHUNKS)")]
                    .iter()
                    .filter(|(b, _)| o & b != 0)
                    .map(|(_, s)| *s)
                    .collect();
                return Err(format!("learned impact frame {frame}: the call batch exceeds its capacity in {}", what.join(", ")));
            }
            self.totals.batches += 1;
            self.totals.calls += w[ctl::CALLS] as u64;
            self.totals.routed_islands += w[ctl::ROUTED] as u64;
            self.totals.bound_active += w[ctl::BOUND] as u64;
            if w[ctl::FALLBACKS] > 0 {
                self.totals.fallbacks += w[ctl::FALLBACKS] as u64;
                self.totals.fallback_nonfinite += w[ctl::FB_NONFINITE] as u64;
                self.totals.fallback_infeasible += w[ctl::FB_INFEASIBLE] as u64;
                self.totals.fallback_range += w[ctl::FB_RANGE] as u64;
                eprintln!(
                    "learned impact frame {frame}: {} of {} call(s) fell back to the GPU punch (non-finite {}, KE bound infeasible {}, input range {})",
                    w[ctl::FALLBACKS],
                    w[ctl::CALLS],
                    w[ctl::FB_NONFINITE],
                    w[ctl::FB_INFEASIBLE],
                    w[ctl::FB_RANGE]
                );
            }
            self.pending = w[ctl::PENDING] != 0;
        } else if applied {
            self.pending = false;
        }
        if std::env::var("STRESS_LEARNED_TRACE").is_ok() {
            eprintln!("learned impact frame {frame}: applied {applied} batched {batched}: ctl {:?}", &w[..18]);
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ the decode alone (tests)

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
            nw.extend([[0.0; 4]; NODE_F4 - 8]);
        }
        for &(d0, lq, dd, bin) in &c.bonds {
            bw.push([d0, lq, dd, b(bin)]);
            bw.push([0.0, 0.0, b(u32::MAX), b(g as u32)]);
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

/// The decode kernel on its own (`learned_decode`, tests).
pub struct Decode {
    kernel: Kernel,
}

impl Decode {
    pub fn new(gpu: &Gpu) -> Decode {
        Decode { kernel: gpu.compute(&shaders::LEARNED, "learned_decode", &BINDINGS) }
    }

    /// The decode of `calls`: per call, its nodes' decoded state, its bonds' (broke,
    /// damage) and its decode flags.
    pub fn run(&self, gpu: &Gpu, cfg: &DecodeConfig, calls: &[CallIn]) -> Vec<(Vec<NodeOut>, Vec<(bool, f32)>, u32)> {
        let (cw, nw, bw) = pack(calls);
        let one = [[0f32; 4]];
        let cb = gpu.storage("learned calls", &cw);
        let nb = gpu.storage("learned nodes", if nw.is_empty() { &one[..] } else { &nw });
        let bb = gpu.storage("learned bonds", if bw.is_empty() { &one[..] } else { &bw });
        let mut words = [0u32; ctl::WORDS];
        words[ctl::CALLS] = calls.len() as u32;
        let ctl_buf = gpu.storage("learned ctl", &words);
        let params = gpu.uniform("learned params", &Params { log_tau: cfg.tau.ln() as f32, tolerance: cfg.energy_tolerance as f32, ..Default::default() });
        let dummies: Vec<wgpu::Buffer> = (0..28).map(|_| gpu.storage("learned unbound", &[0u32; 4])).collect();
        let mut bufs: [&wgpu::Buffer; 28] = std::array::from_fn(|i| &dummies[i]);
        bufs[0] = &params;
        bufs[1] = &ctl_buf;
        bufs[2] = &cb;
        bufs[3] = &nb;
        bufs[4] = &bb;
        let group = gpu.bind(&self.kernel.group, &bufs);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernel.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(calls.len().max(1) as u32, 1, 1);
        }
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
            let bonds = (0..c.bonds.len())
                .map(|j| {
                    let w = br[BOND_F4 * (bfirst + j) + 1];
                    (w[1].to_bits() != 0, w[0])
                })
                .collect();
            out.push((nodes, bonds, cr[CALL_F4 * g + 4][2].to_bits()));
            first += c.nodes.len();
            bfirst += c.bonds.len();
        }
        out
    }
}
