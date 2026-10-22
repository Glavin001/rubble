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
//! - routing (`learned_route.rs`): the CPU's `select_islands` on the GPU, every routed body
//!   in one call (P1's call builder); `STRESS_LEARNED_ROUTE=world|recorder` whose routing
//!   (the coupled world's, tiered, the default; or a standalone world's), the other switches the world's
//!   (`STRESS_DEBRIS_ROUTING=1` among them);
//!   `STRESS_LEARNED_ROUTING=sweep` the earlier rule instead (the punch's sweep, calls per
//!   impactor group, islands approaching a struck one joined: no CPU counterpart).

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

/// float4 words per call, node and bond; uint4 words per call island; float4 per impactor.
pub const CALL_F4: usize = 10;
pub const NODE_F4: usize = 18;
pub const BOND_F4: usize = 2;
pub const CISL_U4: usize = 2;
pub const IMP_F4: usize = 4;

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
    pub const JOINED: usize = 19;
    pub const ROUTE: usize = 20;
    pub const ROUTE_EDGES: usize = 21;
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
    speed_ref: f32,
    out_bases: [u32; 4],
    out_bases2: [u32; 4],
    modes: [u32; 4],
}

/// Where the network's outputs sit in its buffer (float offsets; P2's outputs.npz shapes in
/// the global node order: every call's chunks, then every call's impactors; bonds in call
/// order): `chunk_out` [Nc, 12], `imp_out` [Ni, 12], `bond_log_q` [E], `bond_ddamage` [E],
/// `bond_time_logits` [E, 8]. `learned_adopt` copies them into the decode's slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutBases {
    pub chunk_out: u32,
    pub imp_out: u32,
    pub bond_log_q: u32,
    pub bond_ddamage: u32,
    pub bond_time_logits: u32,
}

const BINDINGS: [Binding; 29] = [
    Binding::Uniform,   // 0 params
    Binding::StorageRw, // 1 ctl
    Binding::StorageRw, // 2 calls
    Binding::StorageRw, // 3 nodes
    Binding::StorageRw, // 4 bonds
    Binding::StorageRw, // 5 isl
    Binding::StorageRw, // 6 call islands' frames (learned_feat)
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
    Binding::Storage,   // 28 the network's outputs
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
    /// The punch's outcome plus a fixed per-chunk pattern: pieces that do not move rigidly
    /// (a test of the kept-deformation boundary).
    PunchWobble = 4,
    /// The punch's outcome with the impactor placed 5 mm into the plug (a test of the window
    /// end's handover of overlaps to the engine).
    PunchOverlap = 5,
}

impl Stub {
    /// `STRESS_LEARNED_STUB`, if set (an unknown value is an error).
    pub fn from_env() -> Result<Option<Stub>, String> {
        match std::env::var("STRESS_LEARNED_STUB").as_deref() {
            Err(_) => Ok(None),
            Ok("zero") => Ok(Some(Stub::Zero)),
            Ok("punch") => Ok(Some(Stub::Punch)),
            Ok("nonfinite") => Ok(Some(Stub::NonFinite)),
            Ok("punch-wobble") => Ok(Some(Stub::PunchWobble)),
            Ok("punch-overlap") => Ok(Some(Stub::PunchOverlap)),
            Ok(other) => Err(format!("STRESS_LEARNED_STUB={other}: expected zero, punch, punch-wobble or nonfinite")),
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
    /// Half extents (a sphere's radius on every axis: its oriented box, as `Impactor::obox`).
    pub half: [f32; 3],
    /// Young's modulus as the solver scales it (not plane-strain: the features' modulus).
    pub modulus_scaled: f32,
    pub friction: f32,
    pub crushable: bool,
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
    /// The joint materials (the routing's bond capacities).
    pub materials: Option<&'a wgpu::Buffer>,
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
    adopt: Kernel,
    check: Kernel,
    apply_bonds: Kernel,
    apply_bodies: Kernel,
    apply_impactors: Kernel,
    tally_apply: Kernel,
    place: Kernel,
    unsplit: Kernel,
}

/// Running totals over the run (logged; tests read them).
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Resident frames the path followed (a batch could be made).
    pub resident_frames: u64,
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
    /// Islands joined to a call because they approach a struck island within the frame.
    pub joined: u64,
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
    cisl: wgpu::Buffer,
    cislf: wgpu::Buffer,
    chunk_node: wgpu::Buffer,
    frozen: wgpu::Buffer,
    imps: wgpu::Buffer,
    largs: wgpu::Buffer,
    dummy: wgpu::Buffer,
    /// Distinct stand-ins for read-write slots a kernel does not use (the Metal path binds
    /// the module's whole group 0; one buffer may not be read-write twice in a dispatch).
    dummy_rw: [wgpu::Buffer; 3],
    /// The network's output buffer and where its arrays sit (`set_network_outputs`).
    out: (Option<wgpu::Buffer>, OutBases),
    imp_list: Vec<ImpactorInfo>,
    island_cap: u32,
    chunk_count: u32,
    /// The input features (`learned_feat.rs`), once the static tables are set.
    pub feat: Option<crate::learned_feat::Features>,
    /// `STRESS_LEARNED_DRY=1`: batches are built (features included) but no call stands:
    /// each goes to the punch, unlogged (a feature export's mode: the engine's motion stays
    /// the plain resident frame's).
    pub dry: bool,
    /// The frame the batch is made in (the features' broken-bond stamps).
    pub frame: u32,
    /// The frame the last batch was made in (its key map's stamp).
    pub last_batch_frame: u32,
    /// `STRESS_DECODED_DEFORMATION=1`: a call's chunks keep their decoded departure from
    /// their body's fit as hidden state at the window end (P0's boundary choice).
    pub keep_deformation: bool,
    /// A batch stood at the last read: its window end is due this frame.
    pub pending: bool,
    pub totals: Totals,
    /// `STRESS_LEARNED_TIMING=1`: GPU time per stage (`learned_timing.rs`).
    pub timer: Option<std::rc::Rc<crate::learned_timing::StageTimer>>,
    /// The CPU's routing on the GPU (`set_routing`); None: the earlier sweep.
    route: Option<crate::learned_route::Route>,
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
            adopt: k("learned_adopt"),
            check: k("learned_check"),
            apply_bonds: k("learned_apply_bonds"),
            apply_bodies: k("learned_apply_bodies"),
            apply_impactors: k("learned_apply_impactors"),
            tally_apply: k("learned_tally_apply"),
            place: k("learned_place"),
            unsplit: k("learned_unsplit"),
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
            cisl: rw(gpu, "learned call islands", 16 * (CISL_U4 * max_cisl as usize) as u64, false),
            cislf: rw(gpu, "learned call island frames", 16 * 2 * max_cisl as u64, false),
            chunk_node: rw(gpu, "learned chunk map", 16, false),
            frozen: rw(gpu, "learned held bodies", 16 * 4 * (max_cisl as u64 + 32), false),
            imps: rw(gpu, "learned impactors", 16 * IMP_F4 as u64, false),
            largs: rw(gpu, "learned indirect arguments", 16 * 4, true),
            dummy: rw(gpu, "learned unbound", 16, false),
            dummy_rw: [rw(gpu, "learned unbound rw 0", 16, false), rw(gpu, "learned unbound rw 1", 16, false), rw(gpu, "learned unbound rw 2", 16, false)],
            out: (None, OutBases::default()),
            imp_list: Vec::new(),
            island_cap: 0,
            chunk_count: 0,
            feat: None,
            dry: std::env::var("STRESS_LEARNED_DRY").is_ok_and(|v| v == "1"),
            frame: 0,
            last_batch_frame: u32::MAX,
            keep_deformation: std::env::var("STRESS_DECODED_DEFORMATION").is_ok_and(|v| v == "1"),
            pending: false,
            totals: Totals::default(),
            timer: crate::learned_timing::StageTimer::new(gpu).map(std::rc::Rc::new),
            route: None,
        }
    }

    /// The input features: static tables from the world's pristine structures, columns
    /// placed by `map` (the schema the network was trained on). `layout_cap`: bond records
    /// the solver's tables hold.
    pub fn set_features(&mut self, gpu: &Gpu, world: &stress_ref::world::World, map: crate::learned_feat::ColumnMap, layout_cap: u32) -> Result<(), String> {
        let statics = crate::learned_feat::statics(world)?;
        let scene = crate::learned_feat::FeatScene { restitution: world.scene.sim.contact_restitution as f32, contact_friction: world.scene.sim.contact_friction.map(|f| f as f32) };
        self.feat = Some(crate::learned_feat::Features::new(gpu, statics, scene, map, self.max_nodes, self.max_bonds, self.max_cisl, self.max_calls, layout_cap));
        Ok(())
    }

    /// The CPU's routing for `world`'s chunks with `cfg`'s switches (the coupled world's:
    /// `RouteConfig::of`), unless `STRESS_LEARNED_ROUTING=sweep`.
    pub fn set_routing(&mut self, gpu: &Gpu, world: &stress_ref::world::World, cfg: crate::learned_route::RouteConfig) -> Result<(), String> {
        match std::env::var("STRESS_LEARNED_ROUTING").ok().as_deref() {
            Some("sweep") => self.route = None,
            None | Some("cpu") => {
                if world.impactors.len() > 31 {
                    return Err(format!("the learned routing takes at most 31 impactors ({} given)", world.impactors.len()));
                }
                let mut r = crate::learned_route::Route::new(gpu, world);
                r.cfg = cfg;
                self.route = Some(r);
            }
            Some(o) => return Err(format!("STRESS_LEARNED_ROUTING={o}: cpu (default) or sweep")),
        }
        Ok(())
    }

    /// The CPU's routing (`set_routing`): a batch is encoded every resident frame, its
    /// kernels gated by this frame's routing check (untiered: not gated).
    pub fn routes_exactly(&self) -> bool {
        self.route.is_some()
    }

    /// The network's output buffer (P0's kernel writes it; `learned_adopt` reads it).
    pub fn set_network_outputs(&mut self, buffer: wgpu::Buffer, bases: OutBases) {
        self.out = (Some(buffer), bases);
    }

    /// The impactors (uploaded when they change).
    pub fn set_impactors(&mut self, gpu: &Gpu, list: &[ImpactorInfo]) {
        if list == self.imp_list.as_slice() {
            return;
        }
        let words: Vec<[f32; 4]> = list
            .iter()
            .flat_map(|i| {
                let hn = (i.half[0] * i.half[0] + i.half[1] * i.half[1] + i.half[2] * i.half[2]).sqrt();
                [[f32::from_bits(i.body), i.bound, i.face, i.mass], [i.inertia[0], i.inertia[1], i.inertia[2], i.modulus], [hn, i.half[0], i.half[1], i.half[2]], [i.modulus_scaled, i.friction, i.face, f32::from(u8::from(i.crushable))]]
            })
            .collect();
        self.imps = if words.is_empty() { gpu.storage("learned impactors", &[[0f32; 4]; IMP_F4]) } else { gpu.storage("learned impactors", &words) };
        self.imp_list = list.to_vec();
    }

    /// Room for the tables' islands and chunks (the per-island and per-chunk buffers).
    /// Growing the chunk map drops a pending batch's map: only between batches.
    pub fn ensure(&mut self, gpu: &Gpu, island_cap: u32, chunk_count: u32) {
        if let Some(r) = self.route.as_mut() {
            r.ensure(gpu, island_cap);
        }
        if island_cap > self.island_cap {
            self.isl = rw(gpu, "learned islands", 16 * 5 * island_cap as u64, false);
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
                stub: self.stub as u32 | if self.dry { 0x100 } else { 0 },
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
                speed_ref: t.speed as f32,
                out_bases: [self.out.1.chunk_out, self.out.1.imp_out, self.out.1.bond_log_q, self.out.1.bond_ddamage],
                out_bases2: [self.out.1.bond_time_logits, 0, 0, 0],
                modes: [self.route.is_some() as u32, self.route.as_ref().is_some_and(|r| r.always()) as u32, 0, 0],
            },
        )
    }

    fn buffers<'a>(&'a self, params: &'a wgpu::Buffer, t: &'a Tables, split: [&'a wgpu::Buffer; 6]) -> [&'a wgpu::Buffer; 29] {
        let [parents, work, slots, control, firsts, out] = split;
        [
            // The indirect arguments only where `learned_compact` writes them (a dispatch that
            // reads them as its arguments may not also bind them read-write).
            params, &self.ctl, &self.calls, &self.nodes, &self.bonds, &self.isl, &self.cislf, &self.cisl, &self.chunk_node, &self.frozen, &self.imps, t.islands, t.chunks, t.state, t.ctab, t.bodies, t.bond_static, t.bond_dyn, t.status, t.counters, parents, work, slots, control, firsts, t.route, out, &self.dummy_rw[2], self.out.0.as_ref().unwrap_or(&self.dummy),
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
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: self.timer.as_ref().and_then(|t| t.writes("apply")) });
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
        let t = Tables { islands: d, chunks: d, state: d, ctab: d, bodies, bond_static: d, bond_dyn: &self.dummy_rw[0], status: &self.dummy_rw[1], counters: &pass.counters, route: d, island_args: None, materials: None, s: *sc };
        let bufs = self.buffers(&params, &t, [&pass.parents, &pass.work, &pass.slots, &pass.control, &pass.firsts, &pass.out]);
        gpu.bind(&self.k.place.group, &bufs)
    }

    /// The stage a batch kernel's time is booked to.
    fn stage_of(&self, k: &Kernel) -> &'static str {
        let k = &k.pipeline;
        if [&self.k.select, &self.k.debris, &self.k.count_bonds, &self.k.compact, &self.k.clear].iter().any(|x| &x.pipeline == k) {
            "select"
        } else if [&self.k.gather_chunks, &self.k.gather_bonds, &self.k.gather_impactors].iter().any(|x| &x.pipeline == k) {
            "gather"
        } else if &self.k.finish.pipeline == k {
            "features"
        } else if [&self.k.stub, &self.k.adopt].iter().any(|x| &x.pipeline == k) {
            "network"
        } else {
            "decode"
        }
    }

    /// After the frame's read: the frame's stage times collected (`STRESS_LEARNED_TIMING=1`).
    pub fn collect_timing(&self, gpu: &Gpu) {
        if let Some(t) = &self.timer {
            t.collect(gpu);
        }
    }

    fn batch_buffers<'a>(&'a self, t: &'a Tables) -> crate::learned_feat::BatchBuffers<'a> {
    crate::learned_feat::BatchBuffers {
        ctl: &self.ctl,
        calls: &self.calls,
        nodes: &self.nodes,
        bonds: &self.bonds,
        cisl: &self.cisl,
        cislf: &self.cislf,
        imps: &self.imps,
        chunks: t.chunks,
        bond_static: t.bond_static,
        bond_dyn: t.bond_dyn,
        islands: t.islands,
        ctab: t.ctab,
        bodies: t.bodies,
        state: t.state,
        counters: t.counters,
        island_cap: t.s.island_cap,
        engine_base: t.s.engine_base,
        chunk_count: self.chunk_count,
        prev_frame: self.last_batch_frame,
        impactor_count: self.imp_list.len() as u32,
        frame: self.frame,
        dt: t.s.dt,
        speed: t.s.speed,
        gravity: t.s.gravity,
    }
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
            let stage = self.stage_of(k);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: self.timer.as_ref().and_then(|t| t.writes(stage)) });
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
            // (before the deformation is kept: it is measured from the bodies' placed frames)
            run(enc, &self.k.unsplit, D::Indirect(32));
            if self.keep_deformation {
                if let Some(f) = &self.feat {
                    f.encode_deform(gpu, enc, &self.batch_buffers(t), self.timer.as_deref());
                }
            }
            run(enc, &self.k.clear, D::One);
        }
        if !batch {
            return;
        }
        match &self.route {
            Some(r) => {
                let rt = crate::learned_route::RouteTables {
                    islands: t.islands,
                    chunks: t.chunks,
                    state: t.state,
                    ctab: t.ctab,
                    bodies: t.bodies,
                    bond_static: t.bond_static,
                    materials: t.materials.expect("the joint materials (learned routing)"),
                    counters: t.counters,
                    route: t.route,
                    imps: &self.imps,
                    isl: &self.isl,
                    ctl: &self.ctl,
                };
                r.encode(gpu, enc, &rt, t.s.island_cap, t.s.engine_base, self.imp_list.len() as u32, t.s.dt, t.s.speed, self.timer.as_deref());
            }
            None => {
                run(enc, &self.k.select, D::Islands);
                // (debris reach, and the islands approaching a struck one: always)
                run(enc, &self.k.debris, D::Threads(t.s.island_cap));
            }
        }
        run(enc, &self.k.count_bonds, D::Islands);
        run(enc, &self.k.compact, D::One);
        run(enc, &self.k.gather_chunks, D::Indirect(16));
        run(enc, &self.k.gather_bonds, D::Indirect(16));
        run(enc, &self.k.gather_impactors, D::Indirect(0));
        let fb = self.batch_buffers(t);
        if let Some(f) = &self.feat {
            f.encode_pre(gpu, enc, &fb, (&self.largs, 0), self.max_bonds, self.timer.as_deref());
        }
        run(enc, &self.k.finish, D::Indirect(0));
        if let Some(f) = &self.feat {
            f.encode_post(gpu, enc, &fb, (&self.largs, 0), self.max_bonds, self.timer.as_deref());
        }
        match network {
            Some(net) => {
                net(enc);
                run(enc, &self.k.adopt, D::Indirect(0));
            }
            None => run(enc, &self.k.stub, D::Indirect(0)),
        }
        run(enc, &self.k.decode, D::Indirect(0));
        run(enc, &self.k.gate, D::Indirect(0));
        run(enc, &self.k.tally, D::One);
    }

    /// The control words' ranges (for the frame's one read): the batch's, then the
    /// features' (edge totals and overflow bits) when they run.
    pub fn ranges(&self) -> Vec<(&wgpu::Buffer, u64, u64)> {
        let mut r = vec![(&self.ctl, 0, 4 * ctl::WORDS as u64)];
        if let Some(f) = &self.feat {
            r.push(f.range());
        }
        r
    }

    /// The frame's read of the control words: `applied` / `batched` say what this frame
    /// recorded. Every fallback is logged; a batch over a capacity is an error.
    pub fn follow(&mut self, parts: &[Vec<u8>], frame: u64, applied: bool, batched: bool) -> Result<(), String> {
        self.totals.resident_frames += 1;
        let bytes = &parts[0];
        let w: &[u32] = bytemuck::cast_slice(&bytes[..4 * ctl::WORDS]);
        if batched && w[ctl::CALLS] > 0 {
            if let Some(f) = parts.get(1) {
                let fw: &[u32] = bytemuck::cast_slice(&f[..32]);
                if fw[5] != 0 {
                    return Err(format!("learned impact frame {frame}: a call holds more broken bonds than its cracked-face candidates ({} per chunk)", crate::learned_feat::CRACKED_PER_NODE));
                }
                if fw[3] != 0 {
                    let what: Vec<&str> = [(1, "cracked faces"), (2, "impactor contacts"), (4, "structure contacts")].iter().filter(|(b, _)| fw[3] & b != 0).map(|(_, s)| *s).collect();
                    return Err(format!("learned impact frame {frame}: the call batch's {} exceed their capacity (STRESS_LEARNED_MAX_BONDS / _MAX_CALL_ISLANDS / _MAX_NODES)", what.join(", ")));
                }
            }
        }
        if applied {
            self.totals.applied += w[ctl::APPLIED] as u64;
            self.totals.stale += w[ctl::STALE] as u64;
            self.totals.broken += w[ctl::BROKEN] as u64;
            if w[ctl::STALE] > 0 {
                eprintln!("learned impact frame {frame}: {} call(s) not applied at their window end: their islands changed under them (bodies released with their frame-start motion)", w[ctl::STALE]);
            }
        }
        if batched {
            if w[ctl::OVERFLOW] & 128 != 0 {
                let r = w[ctl::ROUTE];
                return Err(format!(
                    "learned impact frame {frame}: the routing {} ({} joining pairs; STRESS_LEARNED_MAX_ROUTE_PAIRS per island slot)",
                    if r & 1 != 0 { "found more joining pairs than its capacity" } else { "did not converge its components" },
                    w[ctl::ROUTE_EDGES]
                ));
            }
            if w[ctl::OVERFLOW] != 0 {
                let o = w[ctl::OVERFLOW];
                let what: Vec<&str> = [(1, "calls (STRESS_LEARNED_MAX_CALLS)"), (2, "nodes (STRESS_LEARNED_MAX_NODES)"), (4, "bonds (STRESS_LEARNED_MAX_BONDS)"), (8, "call islands (STRESS_LEARNED_MAX_CALL_ISLANDS)"), (16, "chunks in one call (STRESS_LEARNED_MAX_CALL_CHUNKS)"), (64, "bonds in one call (131072)")]
                    .iter()
                    .filter(|(b, _)| o & b != 0)
                    .map(|(_, s)| *s)
                    .collect();
                return Err(format!("learned impact frame {frame}: the call batch exceeds its capacity in {}", what.join(", ")));
            }
            self.totals.batches += 1;
            self.totals.calls += w[ctl::CALLS] as u64;
            self.totals.routed_islands += w[ctl::ROUTED] as u64;
            self.totals.joined += w[ctl::JOINED] as u64;
            self.totals.bound_active += w[ctl::BOUND] as u64;
            if w[ctl::FALLBACKS] > 0 && !self.dry {
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
            self.last_batch_frame = frame as u32;
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
    let (mut chunk_base, mut imp_base) = (0u32, 0u32);
    let b = f32::from_bits;
    for (g, c) in calls.iter().enumerate() {
        let s = &c.scales;
        cw.push([s[0], s[1], s[2], s[3]]);
        cw.push([s[4], s[5], s[6], s[7]]);
        cw.push([s[8], b(c.flags), b(first), b(c.nodes.len() as u32)]);
        cw.push([c.gravity[0], c.gravity[1], c.gravity[2], b(c.chunks as u32)]);
        cw.push([b(bfirst), b(c.bonds.len() as u32), 0.0, 0.0]);
        cw.push([0.0; 4]);
        cw.push([0.0, 0.0, b(chunk_base), b(imp_base)]);
        cw.extend([[0.0; 4]; CALL_F4 - 7]);
        chunk_base += c.chunks as u32;
        imp_base += (c.nodes.len() - c.chunks) as u32;
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
    adopt: Kernel,
}

impl Decode {
    pub fn new(gpu: &Gpu) -> Decode {
        Decode { kernel: gpu.compute(&shaders::LEARNED, "learned_decode", &BINDINGS), adopt: gpu.compute(&shaders::LEARNED, "learned_adopt", &BINDINGS) }
    }

    /// The decode of `calls`: per call, its nodes' decoded state, its bonds' (broke,
    /// damage) and its decode flags.
    /// With `adopted`, the outputs come from a network buffer (`learned_adopt` first) in
    /// place of `NodeIn::out` and the bonds' log q̂ / Δdamage / bin.
    pub fn run(&self, gpu: &Gpu, cfg: &DecodeConfig, calls: &[CallIn], adopted: Option<(&[f32], OutBases)>) -> Vec<(Vec<NodeOut>, Vec<(bool, f32)>, u32)> {
        let (cw, nw, bw) = pack(calls);
        let one = [[0f32; 4]];
        let cb = gpu.storage("learned calls", &cw);
        let nb = gpu.storage("learned nodes", if nw.is_empty() { &one[..] } else { &nw });
        let bb = gpu.storage("learned bonds", if bw.is_empty() { &one[..] } else { &bw });
        let mut words = [0u32; ctl::WORDS];
        words[ctl::CALLS] = calls.len() as u32;
        let ctl_buf = gpu.storage("learned ctl", &words);
        let bases = adopted.map(|a| a.1).unwrap_or_default();
        let params = gpu.uniform(
            "learned params",
            &Params { log_tau: cfg.tau.ln() as f32, tolerance: cfg.energy_tolerance as f32, out_bases: [bases.chunk_out, bases.imp_out, bases.bond_log_q, bases.bond_ddamage], out_bases2: [bases.bond_time_logits, 0, 0, 0], ..Default::default() },
        );
        let dummies: Vec<wgpu::Buffer> = (0..29).map(|_| gpu.storage("learned unbound", &[0u32; 4])).collect();
        let lout = adopted.map(|a| gpu.storage("learned network outputs", a.0));
        let mut bufs: [&wgpu::Buffer; 29] = std::array::from_fn(|i| &dummies[i]);
        if let Some(l) = &lout {
            bufs[28] = l;
        }
        bufs[0] = &params;
        bufs[1] = &ctl_buf;
        bufs[2] = &cb;
        bufs[3] = &nb;
        bufs[4] = &bb;
        let group = gpu.bind(&self.kernel.group, &bufs);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        if lout.is_some() {
            let group = gpu.bind(&self.adopt.group, &bufs);
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.adopt.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(calls.len().max(1) as u32, 1, 1);
        }
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

// ------------------------------------------------------------------ feature export (tests, diagnostics)

/// One call's feature arrays as the GPU made them (`Learned::export`, extra reads: for
/// tests and diagnostics only), named and shaped as the calls schema's, with each edge set's
/// identity per undirected edge (row 2k is its first twin).
#[derive(Clone, Debug, Default)]
pub struct ExportCall {
    pub chunk_keys: Vec<i64>,
    pub impactor_keys: Vec<i64>,
    /// L, m, v, c, f_t, E, GA, dt, E_in.
    pub scales: [f32; 9],
    /// name → (shape, values); edge arrays `[2E, F]` in the GPU's edge order.
    pub arrays: std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// Per edge type ("bond", "cracked", "impactor_contact", "structure_contact"): per
    /// undirected edge, (src key, dst key) of its first twin (chunk keys; an impactor's key
    /// is its index with the top bit set) and the edge's own key (a bond's), else 0.
    pub edges: std::collections::BTreeMap<String, Vec<(i64, i64, i64)>>,
    /// The call's predicted contact rows (`learned_feat.slang` CON_F4 float4 each).
    pub contacts: Vec<[[f32; 4]; 6]>,
    /// Per chunk node: its world position at frame start.
    pub chunk_pos: Vec<[f32; 3]>,
    /// Per chunk node: its decoded frame-end displacement (after the decode; zero before).
    pub chunk_dx: Vec<[f32; 3]>,
    /// Per chunk node: the feature kernels' scratch words (`learned_feat.slang` W_*).
    pub work: Vec<[f32; crate::learned_feat::WORK_F]>,
}

/// An impactor's key in `ExportCall::edges` (top bit set, apart from chunk keys).
pub const IMPACTOR_KEY_BIT: i64 = 1 << 62;

impl Learned {
    /// The last batch's features, read back (several reads: not for the frame's loop).
    /// The last routing's parts (`learned_route.rs`; diagnostics).
    pub fn route_parts(&self, gpu: &Gpu, n: usize) -> Vec<[f32; 4]> {
        self.route.as_ref().map(|r| r.parts(gpu, n)).unwrap_or_default()
    }

    /// The last routing's seeds, labels and control words (`learned_route.rs`; diagnostics).
    pub fn route_seeds(&self, gpu: &Gpu, n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        self.route.as_ref().map(|r| r.seeds(gpu, n)).unwrap_or_default()
    }

    /// The motion the last batch's hold took from its bodies (a test's view of the state the
    /// batch routed from): per held call island (island, velocity, angular velocity) and per
    /// held impactor (index, velocity, angular velocity); the engine's Y-up frame.
    pub fn held_motion(&self, gpu: &Gpu) -> (Vec<(u32, [f32; 3], [f32; 3])>, Vec<(u32, [f32; 3], [f32; 3])>) {
        let ctl: Vec<u32> = gpu.read_range(&self.ctl, 0, ctl::WORDS);
        let (ncalls, ncisl) = (ctl[ctl::CALLS] as usize, ctl[ctl::CISL] as usize);
        if ncalls == 0 {
            return (Vec::new(), Vec::new());
        }
        let calls: Vec<[f32; 4]> = gpu.read_range(&self.calls, 0, CALL_F4 * ncalls);
        let cisl: Vec<[u32; 4]> = gpu.read_range(&self.cisl, 0, CISL_U4 * ncisl.max(1));
        let frozen: Vec<[f32; 4]> = gpu.read_range(&self.frozen, 0, 4 * (self.max_cisl as usize + 32));
        let v3 = |w: [f32; 4]| [w[0], w[1], w[2]];
        let mut islands = Vec::new();
        for e in 0..ncisl {
            let g = cisl[CISL_U4 * e][3] as usize;
            if calls[CALL_F4 * g + 4][3].to_bits() != 0 || frozen[4 * e][0].to_bits() == u32::MAX {
                continue;
            }
            islands.push((cisl[CISL_U4 * e][0], v3(frozen[4 * e + 2]), v3(frozen[4 * e + 3])));
        }
        let mut imps = Vec::new();
        let mask = (0..ncalls).filter(|&g| calls[CALL_F4 * g + 4][3].to_bits() == 0).fold(0u32, |m, g| m | calls[CALL_F4 * g + 5][3].to_bits());
        for k in 0..self.imp_list.len().min(32) {
            if mask & (1 << k) != 0 {
                let e = self.max_cisl as usize + k;
                imps.push((k as u32, v3(frozen[4 * e + 2]), v3(frozen[4 * e + 3])));
            }
        }
        (islands, imps)
    }

    pub fn export(&self, gpu: &Gpu) -> Vec<ExportCall> {
        let Some(f) = &self.feat else { return Vec::new() };
        let ctlw: Vec<u32> = gpu.read(&self.ctl);
        let n_calls = ctlw[ctl::CALLS] as usize;
        if n_calls == 0 {
            return Vec::new();
        }
        let calls: Vec<[f32; 4]> = gpu.read_range(&self.calls, 0, CALL_F4 * n_calls);
        let n_nodes = ctlw[ctl::NODES] as usize;
        let n_bonds = ctlw[ctl::BONDS] as usize;
        let nodes: Vec<[f32; 4]> = gpu.read_range(&self.nodes, 0, NODE_F4 * n_nodes.max(1));
        let bedge: Vec<[u32; 4]> = gpu.read_range(&f.bedge, 0, n_bonds.max(1));
        let fctl: Vec<u32> = gpu.read(&f.fctl);
        let feat: Vec<f32> = gpu.read(&f.feat);
        let edges: Vec<u32> = gpu.read(&f.edges);
        let l = &f.layout;
        let st = &f.map.strides;
        let u = |x: f32| x.to_bits();
        let chunk_key = |n: usize| -> i64 {
            let w = nodes[NODE_F4 * n + 12];
            ((u(w[2]) as i64) << 32) | u(w[3]) as i64
        };
        let total_chunks = fctl[4] as usize;
        let mut out = Vec::new();
        for g in 0..n_calls {
            let c = |k: usize| calls[CALL_F4 * g + k];
            let first = u(c(2)[2]) as usize;
            let n = u(c(2)[3]) as usize;
            let nc = u(c(3)[3]) as usize;
            let bfirst = u(c(4)[0]) as usize;
            let nb = u(c(4)[1]) as usize;
            let (cbase, ibase) = (u(c(6)[2]) as usize, u(c(6)[3]) as usize);
            let mut e = ExportCall { scales: [c(0)[0], c(0)[1], c(0)[2], c(0)[3], c(1)[0], c(1)[1], c(1)[2], c(1)[3], c(2)[0]], ..Default::default() };
            e.chunk_keys = (0..nc).map(|j| chunk_key(first + j)).collect();
            e.impactor_keys = (nc..n).map(|j| u(nodes[NODE_F4 * (first + j) + 12][0]) as i64).collect();
            let node_key = |global: usize| -> i64 {
                if global < total_chunks {
                    e.chunk_keys[global - cbase]
                } else {
                    IMPACTOR_KEY_BIT | e.impactor_keys[global - total_chunks - ibase]
                }
            };
            // a section's rows [r0, r0 + rows), split into its arrays
            let mut take = |arrays: &mut std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>, s: usize, base: u32, r0: usize, rows: &[usize]| {
                let stride = st[s] as usize;
                let mut col = 0usize;
                for (name, w) in &f.map.arrays[s] {
                    let mut vals = Vec::with_capacity(rows.len() * w);
                    for &r in rows {
                        let at = base as usize + (r0 + r) * stride + col;
                        vals.extend_from_slice(&feat[at..at + w]);
                    }
                    arrays.insert(name.clone(), (vec![rows.len(), *w], vals));
                    col += w;
                }
            };
            let mut arrays = std::collections::BTreeMap::new();
            take(&mut arrays, 0, l.chunk, cbase, &(0..nc).collect::<Vec<_>>());
            take(&mut arrays, 1, l.imp, ibase, &(0..n - nc).collect::<Vec<_>>());
            take(&mut arrays, 2, l.glob, g, &[0]);
            // bonds: the live ones (both chunks in the call)
            let live: Vec<usize> = (bfirst..bfirst + nb).filter(|&b| bedge[b][0] != u32::MAX && bedge[b][1] != u32::MAX).collect();
            let rows: Vec<usize> = live.iter().flat_map(|&b| [2 * b, 2 * b + 1]).collect();
            take(&mut arrays, 3, l.bond, 0, &rows);
            let bk = &f.statics.bond_key;
            e.edges.insert(
                "bond".into(),
                live.iter()
                    .map(|&b| {
                        let k = bk[bedge[b][2] as usize];
                        (chunk_key(bedge[b][0] as usize), chunk_key(bedge[b][1] as usize), ((k.0 as i64) << 32) | k.1 as i64)
                    })
                    .collect(),
            );
            // the three edge sets: the call's rows, from fctl
            let caps = [l.cap_cracked as usize, l.cap_ic as usize, l.cap_sc as usize];
            let ebase = [0usize, 2 * caps[0], 2 * (caps[0] + caps[1])];
            let kbase = 4 * (caps[0] + caps[1] + caps[2]);
            let kb = [kbase, kbase + caps[0], kbase + caps[0] + caps[1]];
            for (s, (name, sect, base)) in [("cracked", 4usize, l.cracked), ("impactor_contact", 5, l.ic), ("structure_contact", 6, l.sc)].into_iter().enumerate() {
                let count = fctl[8 + 6 * g + s] as usize;
                let at = fctl[8 + 6 * g + 3 + s] as usize;
                take(&mut arrays, sect, base, at, &(0..count).collect::<Vec<_>>());
                let mut list = Vec::new();
                for k in 0..count / 2 {
                    let row = at + 2 * k;
                    let (src, dst) = (edges[ebase[s] + 2 * row] as usize, edges[ebase[s] + 2 * row + 1] as usize);
                    let key = if s == 0 {
                        let (ks, kc) = (edges[kb[s] + 2 * ((at / 2) + k)], edges[kb[s] + 2 * ((at / 2) + k) + 1]);
                        ((ks as i64) << 32) | kc as i64
                    } else {
                        0
                    };
                    list.push((node_key(src), node_key(dst), key));
                }
                e.edges.insert(name.into(), list);
            }
            e.arrays = arrays;
            let (row0, rows) = (u(c(8)[0]) as usize, u(c(8)[1]) as usize);
            if rows > 0 {
                let con: Vec<[f32; 4]> = gpu.read_range(&f.contacts, crate::learned_feat::CON_F4 * row0, crate::learned_feat::CON_F4 * rows);
                e.contacts = con.chunks(6).map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]]).collect();
            }
            let wk: Vec<f32> = gpu.read_range(&f.work, crate::learned_feat::WORK_F * first, crate::learned_feat::WORK_F * nc.max(1));
            e.work = (0..nc).map(|j| std::array::from_fn(|k| wk[crate::learned_feat::WORK_F * j + k])).collect();
            e.chunk_dx = (0..nc).map(|j| { let w = nodes[NODE_F4 * (first + j) + 10]; [w[0], w[1], w[2]] }).collect();
            e.chunk_pos = (0..nc).map(|j| { let w = nodes[NODE_F4 * (first + j) + 13]; [w[0], w[1], w[2]] }).collect();
            out.push(e);
        }
        out
    }
}

impl Drop for Learned {
    fn drop(&mut self) {
        if let Some(t) = &self.timer {
            eprintln!("{}", t.report());
        }
    }
}
