//! GPU-resident bond-graph fracture over a rigid engine with fixed slots
//! (`shaders/fracture.slang`, engine/docs/stress-threshold-gpu.md): the threshold profile's
//! frame without the host. Every chunk's colliders keep their engine slots for good and the
//! engine holds a body slot for as many bodies as there can be, so a split allocates
//! nothing: each broken body's largest piece keeps the body (the minimum edit), every other
//! piece takes a free slot, and the chunks' colliders are re-parented in place.
//!
//! A frame is recorded into the caller's encoder: the loads and breaks, then a one-thread
//! gate that sizes every split kernel's indirect dispatch (zero when no bond broke). Nothing
//! is read back to run it; [`GpuFracture::counters`] is a diagnostic read.
//!
//! [`reference_split`] is the same split on the host (f64), for the tests: the topology and
//! the slots must match exactly, the mass properties to f32.

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::ingest::{EngineBuffers, EngineIngest};
use crate::shaders;
use stress_ref::capability::{LoadModel, StrikeModel};

/// Per chunk (`fracture.slang` `chunks`): (centre (structure frame), mass); inertia about
/// the centre (xx, yy, zz, xy); (xz, yz, first part as bits, parts as bits); (Young's
/// modulus, Poisson's ratio, cbrt(volume), smallest half extent).
pub type ChunkRecord = [[f32; 4]; 4];
/// Per part: (origin (structure frame), engine collider as bits); rotation (x, y, z, w).
pub type PartRecord = [[f32; 4]; 2];
/// Per engine body: (centre of mass (structure frame), structure as bits); principal
/// rotation (structure frame, x, y, z, w).
pub type FrameRecord = [[f32; 4]; 2];

/// Chunk flag (`info`): a support (its piece is anchored, static).
pub const INFO_SUPPORT: u32 = 1;

/// What the fracture is made from, once, at load.
#[derive(Clone, Debug, Default)]
pub struct FractureTables {
    pub chunks: Vec<ChunkRecord>,
    pub info: Vec<u32>,
    /// Per bond: chunk a, chunk b (in the same body), tensile capacity (N) as bits, area
    /// (m^2) as bits (the strike's share, `fr_strike_breaks`).
    pub bonds: Vec<[u32; 4]>,
    pub parts: Vec<PartRecord>,
    /// Each chunk's engine body at the start.
    pub chunk_body: Vec<u32>,
    /// Per engine body slot (`body_count` of them): its frame (zero for slots no chunk
    /// uses), and whether the fracture may give it to a new piece.
    pub frames: Vec<FrameRecord>,
    pub free: Vec<u32>,
    /// The engine's body capacity (its previous-pose region starts there).
    pub max_bodies: u32,
    /// Carried impact pulses each half of the pool holds (impact.slang; `fracture.slang`
    /// `l_pool`): from the engine's contact records, so it holds every pulse a frame can
    /// keep; more is an error, reported.
    pub pool_cap: u32,
    /// Impacts at their Hertz peak force (the threshold profile's), else their mean.
    pub impact_peak: bool,
    /// |g| dt: closing speeds within it are not impacts.
    pub band: f32,
    pub dt: f32,
    /// Per chunk 3 float4, for strike prediction (`fracture.slang` `geometry`): (box centre
    /// (structure frame), bounding radius); (half extents, plane-strain modulus E'); box
    /// rotation (structure frame, x, y, z, w). Empty: no strikes.
    pub geometry: Vec<[[f32; 4]; 3]>,
    /// A slower closing is no strike (world_coupling.rs `tiered_impact_speed`).
    pub strike_speed: f32,
    /// How a frame's loads and a strike break bonds (stress-ref `capability`).
    pub load_model: LoadModel,
    pub strike_model: StrikeModel,
    /// Gravity (the stress frame), for the load path.
    pub gravity: [f32; 3],
    /// The load path's and the analytic strike's tables (empty for neither).
    pub ext: ExtTables,
    /// The load path's warm start: each chunk's displacement (translation, rotation;
    /// structure frame). Empty: from rest.
    pub lp_init: Vec<[[f32; 4]; 2]>,
    /// The in-step contact fracture: on when above 0, and the load path's sweeps (even) of a
    /// strike's load (`capability::Profile::contact_fracture`). The load path's and the
    /// analytic strike's need `ext`; the threshold rule's `geometry`.
    pub contact_sweeps: u32,
    /// The engine's contact rows a step holds at most (its `max_contacts`): the in-step
    /// contact fracture reads that many at most.
    pub max_contacts: u32,
}

/// What the load path and the analytic strike read (`fracture.slang` `ext`).
#[derive(Clone, Debug, Default)]
pub struct ExtTables {
    /// Per bond (as `bonds`): centroid, normal, t1 (structure frame), then its JointBond
    /// (stress-layout `GpuJointBond`, its material an index into `materials`).
    pub bonds: Vec<[[f32; 4]; 10]>,
    /// Per chunk: ((first, count) of its bonds in `adjacency`, as bits); the analytic
    /// strike's material: (density, tensile, compressive, shear); (DIF reference rate,
    /// exponent, transition rate, high exponent); (DIF max (0: none), Poisson's ratio,
    /// Young's modulus, -).
    pub chunks: Vec<[[f32; 4]; 4]>,
    /// Each chunk's bonds (ascending), in chunk order.
    pub adjacency: Vec<u32>,
    /// stress-layout `GpuJointMaterial`s.
    pub materials: Vec<[[f32; 4]; 5]>,
}

/// Per entry of `adjacency` (`fracture.slang` `lp_sides`): what a load-path sweep takes of
/// that bond for that chunk. Its arms are f32 differences of the f32 centroid and centres,
/// as the kernels make them.
fn sides(ext: &ExtTables, chunks: &[ChunkRecord], bonds: &[[u32; 4]]) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(5 * ext.adjacency.len());
    for (k, c) in ext.chunks.iter().enumerate() {
        let (first, count) = (c[0][0].to_bits() as usize, c[0][1].to_bits() as usize);
        for &i in &ext.adjacency[first..first + count] {
            let (b, bond) = (&ext.bonds[i as usize], bonds[i as usize]);
            let arm = |chunk: u32| {
                let c = chunks[chunk as usize][0];
                [b[0][0] - c[0], b[0][1] - c[1], b[0][2] - c[2]]
            };
            let (ra, rb) = (arm(bond[0]), arm(bond[1]));
            let first_side = bond[0] as usize == k;
            let other = if first_side { bond[1] } else { bond[0] };
            let (stiff0, stiff1) = (b[5], b[6]);
            out.push([ra[0], ra[1], ra[2], f32::from_bits(i)]);
            out.push([rb[0], rb[1], rb[2], f32::from_bits(other)]);
            out.push([b[2][0], b[2][1], b[2][2], stiff0[1]]);
            out.push([b[1][0], b[1][1], b[1][2], stiff0[0]]);
            out.push([stiff0[2], stiff0[3], stiff1[0], if first_side { 1.0 } else { 0.0 }]);
        }
    }
    out
}

impl ExtTables {
    fn words(&self) -> (Vec<[f32; 4]>, u32, u32, u32) {
        let mut w: Vec<[f32; 4]> = self.bonds.iter().flatten().copied().collect();
        let chunks = w.len() as u32;
        w.extend(self.chunks.iter().flatten().copied());
        let adj = w.len() as u32;
        for q in self.adjacency.chunks(4) {
            let mut v = [0.0f32; 4];
            for (i, &b) in q.iter().enumerate() {
                v[i] = f32::from_bits(b);
            }
            w.push(v);
        }
        let mat = w.len() as u32;
        w.extend(self.materials.iter().flatten().copied());
        (w, chunks, adj, mat)
    }
}

/// Fixed-point scale of the load path's per-body sums (N and N m): 1/256 resolution.
const LP_SCALE: f32 = 256.0;

impl FractureTables {
    pub fn chunk_count(&self) -> u32 {
        self.chunks.len() as u32
    }
    pub fn bond_count(&self) -> u32 {
        self.bonds.len() as u32
    }
    pub fn body_count(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Fixed-point scales of a piece's mass sums (`fracture.slang` `add64`): powers of two
    /// that keep every chunk's term within 2^30 (a piece's offsets from its root are at
    /// most its structure's extent), so a 64-bit sum of up to 2^32 of them cannot overflow.
    pub fn scales(&self) -> [f32; 3] {
        let n = self.chunks.len();
        // Each structure's extent: the largest distance between two of its chunks' centres
        // is at most twice the largest distance from any one of them.
        let mut first: std::collections::HashMap<u32, [f32; 3]> = std::collections::HashMap::new();
        let mut reach: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for k in 0..n {
            let s = self.frames.get(self.chunk_body[k] as usize).map_or(0, |f| f[0][3].to_bits());
            let c = self.chunks[k][0];
            let o = *first.entry(s).or_insert([c[0], c[1], c[2]]);
            let d = ((c[0] - o[0]).powi(2) + (c[1] - o[1]).powi(2) + (c[2] - o[2]).powi(2)).sqrt();
            let r = reach.entry(s).or_insert(0.0);
            *r = r.max(d);
        }
        let extent = 2.0 * reach.values().copied().fold(0.0f32, f32::max) + 1e-3;
        let mut bm = 0.0f32;
        let mut bs = 0.0f32;
        let mut bj = 0.0f32;
        for c in &self.chunks {
            let m = c[0][3];
            bm = bm.max(m);
            bs = bs.max(m * extent);
            let i = c[1][0].abs().max(c[1][1].abs()).max(c[1][2].abs()).max(c[1][3].abs()).max(c[2][0].abs()).max(c[2][1].abs());
            bj = bj.max(i + 2.0 * m * extent * extent);
        }
        let scale = |b: f32| 2f32.powi((30.0 - (b.max(1e-30) as f64).log2().ceil()) as i32 - 1);
        [scale(bm), scale(bs), scale(bj)]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FractureParams {
    chunk_count: u32,
    bond_count: u32,
    body_count: u32,
    max_bodies: u32,
    step: u32,
    pool_cap: u32,
    impact_peak: u32,
    parity: u32,
    dt: f32,
    band: f32,
    scale_m: f32,
    scale_s: f32,
    scale_j: f32,
    strike_speed: f32,
    shot_count: u32,
    pad3: f32,
    load_model: u32,
    strike_model: u32,
    fracture_work: u32,
    sweeps: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    lp_c1: f32,
    lp_c2: f32,
    ext_chunks: u32,
    ext_adj: u32,
    ext_mat: u32,
    lp_scale: f32,
    cs_sweeps: u32,
    cs_row_cap: u32,
    cs_c1: f32,
    cs_c2: f32,
    cs_row_offset: u32,
    pad7: f32,
    pad8: f32,
}

// WebGPU binds a uniform at its size rounded up to 16 bytes; Metal does not check it.
const _: () = assert!(std::mem::size_of::<FractureParams>() % 16 == 0);

/// A shot, for strike prediction (`fracture.slang` `shots`).
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub body: u32,
    /// Its sphere's bounding radius (the punch list's: 1.01 r + 1e-3).
    pub bound: f32,
    /// Its contact face radius (Hertz).
    pub face: f32,
    pub mass: f32,
    /// Plane-strain modulus E'.
    pub modulus: f32,
    /// Which launch it is (1, 2, ...; 0 parked): a later analytic strike of the same launch
    /// on what it broke out is not a new strike (`fr_il_mark`).
    pub launch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ScanParams {
    n: u32,
    in_base: u32,
    out_base: u32,
    block_base: u32,
}

/// Where `fracture.slang`'s `work` and `atom` arrays start (its `w_*` and `a_*`).
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub n: u32,
    pub e: u32,
    pub b: u32,
}

impl Layout {
    pub fn w_body(&self) -> u32 {
        0
    }
    pub fn w_alive(&self) -> u32 {
        self.n
    }
    pub fn w_label(&self) -> u32 {
        self.w_alive() + self.e
    }
    pub fn w_slot(&self) -> u32 {
        self.w_label() + self.n
    }
    pub fn w_new(&self) -> u32 {
        self.w_slot() + self.n
    }
    pub fn w_rank(&self) -> u32 {
        self.w_new() + self.n
    }
    pub fn w_free(&self) -> u32 {
        self.w_rank() + self.n
    }
    pub fn w_free_rank(&self) -> u32 {
        self.w_free() + self.b
    }
    pub fn w_free_list(&self) -> u32 {
        self.w_free_rank() + self.b
    }
    pub fn scan_new_blocks(&self) -> u32 {
        self.w_free_list() + self.b
    }
    pub fn scan_free_blocks(&self) -> u32 {
        self.scan_new_blocks() + 1025
    }
    pub fn w_pulses(&self) -> u32 {
        self.scan_free_blocks() + 1025
    }
    /// Per chunk: the launch whose analytic strike broke it out.
    pub fn w_marks(&self) -> u32 {
        self.w_pulses() + 4 * self.n
    }
    /// Per chunk: the struck body slot + 1 whose plug it came out in this frame.
    pub fn w_carry(&self) -> u32 {
        self.w_marks() + self.n
    }
    /// The struck structures' chunks (per chunk), then the struck structures (per body
    /// slot): the in-step contact fracture's lists.
    pub fn w_cs_list(&self) -> u32 {
        self.w_carry() + self.n
    }
    /// The struck bodies' tubes (2 words per chunk: its interval along the path), listed
    /// body by body: the analytic strike's run (`fracture.slang` `fr_il_tube_*`).
    pub fn w_il_tube(&self) -> u32 {
        self.w_cs_list() + self.n + self.b
    }
    pub fn work_words(&self) -> u32 {
        self.w_il_tube() + 2 * self.n
    }
    pub fn a_dirty(&self) -> u32 {
        3 * self.n + SUMS * self.n
    }
    pub fn a_counters(&self) -> u32 {
        self.a_dirty() + 3 * self.b
    }
    /// Per body slot 9 words: the strike's bounds, entry time, and entry chunk.
    pub fn a_strike(&self) -> u32 {
        self.a_counters() + 16
    }
    /// Per body slot 14 words of the load path's sums.
    pub fn a_lp(&self) -> u32 {
        self.a_strike() + 9 * self.b
    }
    /// Per body slot 6 words of the analytic strike: its run's end, its tube's count, its
    /// plug's mass, its tube's start and fill.
    pub fn a_il(&self) -> u32 {
        self.a_lp() + 14 * self.b
    }
    /// Per shot ([`MAX_SHOTS`]) the earliest strike time of the frame.
    pub fn a_first(&self) -> u32 {
        self.a_il() + 6 * self.b
    }
    /// After the shots' words, the analytic strike's tubes listed so far; then per body
    /// slot 12 words of the in-step contact strike.
    pub fn a_cs(&self) -> u32 {
        self.a_first() + MAX_SHOTS as u32 + 1
    }
    pub fn atom_words(&self) -> u32 {
        self.a_cs() + 12 * self.b
    }
    /// `lp` (float4s): per chunk [`LP_CHUNK`], then per body slot [`LP_BODY`].
    pub fn lp_words(&self) -> u32 {
        LP_CHUNK * self.n + LP_BODY * self.b
    }
}

const SUMS: u32 = 20;
/// float4s per chunk in `lp` (`fracture.slang` `LP_CHUNK`): the load path's state (0..8),
/// its block's factor (8..14), the contact fracture's increment (14..22).
const LP_CHUNK: u32 = 22;
const CS_STATE: usize = 14;
/// float4s per body slot in `lp` (`fracture.slang` `lp_body`): the analytic strike's record
/// (2), the in-step contact strike's (5).
pub const LP_BODY: u32 = 7;
/// The shots strike prediction checks, at most.
pub const MAX_SHOTS: usize = 64;
const BODY_WORDS: u32 = 7;
const ARGS: u64 = 16;
const ARG_CHUNKS: u64 = 0;
const ARG_BONDS: u64 = 1;
const ARG_BODIES: u64 = 2;
const ARG_SCAN_CHUNKS: u64 = 3;
const ARG_SCAN_BODIES: u64 = 4;
const ARG_ONE: u64 = 5;
const ARG_CS_BODIES: u64 = 6;
const ARG_CS_ROWS: u64 = 7;
const ARG_CS_PICK: u64 = 8;
const ARG_CS_LIST: u64 = 9;
const ARG_CS_CHUNKS: u64 = 10;
const ARG_CS_BONDS: u64 = 11;
const ARG_CS_SLOTS: u64 = 12;
const ARG_CS_SAVE: u64 = 13;

/// The counters (`fracture.slang` `C_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counters {
    pub broken: u32,
    pub pieces: u32,
    /// Bits: 1 no free slot, 2 a mass sum out of range, 4 a union did not settle, 8 the
    /// impact pulse pool overflowed (`pool_cap`), 32 a load-path block was not positive
    /// definite (16, an analytic strike's thickness past a cap of passes, is no longer set:
    /// its run has no cap).
    pub error: u32,
    pub split_frames: u32,
    /// Analytic strikes by outcome (`fr_il_strike`), in all.
    pub bounces: u32,
    pub craters: u32,
    pub spalls: u32,
    pub perforations: u32,
    /// The load path's largest failure index (1: a bond's strength), in all.
    pub peak_utilization: f32,
    /// The in-step contact fracture's strikes solved and the bonds they broke, in all.
    pub contact_strikes: u32,
    pub contact_breaks: u32,
}

impl Counters {
    /// From the 16 counter words.
    pub fn from_words(c: &[u32]) -> Counters {
        Counters { broken: c[1], pieces: c[2], error: c[3], split_frames: c[4], bounces: c[7], craters: c[8], spalls: c[9], perforations: c[10], peak_utilization: f32::from_bits(c[11]), contact_strikes: c[12], contact_breaks: c[13] }
    }
}

/// An in-step contact strike's record (`fr_cs_record`), per body slot struck in the last
/// step that had one: the contact point and the direction the striker pushes in (the
/// stress frame), its closing speed, the strike's effective mass and impulse `J = m_eff v`,
/// the load path's peak force `pi J / (2 tau)` and the threshold rule's Hertz force `J /
/// tau`, the striker's face radius, modulus, mass and body, the struck chunk, and when within
/// the step the two meet (s).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContactStrike {
    pub contact: [f32; 3],
    pub peak: f32,
    pub direction: [f32; 3],
    pub speed: f32,
    pub m_eff: f32,
    pub face: f32,
    pub modulus: f32,
    pub body: u32,
    pub chunk: u32,
    pub hertz: f32,
    pub mass: f32,
    pub t_in: f32,
}

impl ContactStrike {
    pub fn impulse(&self) -> f32 {
        self.m_eff * self.speed
    }
}

/// An analytic strike's record (`fr_il_strike`), per body slot struck this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StrikeRecord {
    pub thickness: f32,
    pub ballistic_speed: f32,
    pub residual_speed: f32,
    /// Bits: 1 damage, 2 crater, 4 spall, 8 perforate.
    pub outcome: u32,
    pub crater_depth: f32,
    pub spall_depth: f32,
    pub contact_radius: f32,
    pub ballistic_energy: f32,
}

const BINDINGS: [Binding; 29] = [
    Binding::Uniform,   // params
    Binding::Storage,   // chunks
    Binding::Storage,   // bonds
    Binding::Storage,   // parts
    Binding::StorageRw, // work
    Binding::StorageRw, // atom
    Binding::StorageRw, // frames (+ staging)
    Binding::StorageRw, // engine bodies
    Binding::StorageRw, // engine colliders
    Binding::StorageRw, // ingest collider table
    Binding::Storage,   // ingest manifolds
    Binding::Storage,   // ingest sort
    Binding::StorageRw, // loads and pulses
    Binding::StorageRw, // indirect sizes
    Binding::Uniform,   // scan params
    Binding::Storage,   // chunk info
    Binding::Uniform,   // view params
    Binding::Storage,   // view chunks
    Binding::StorageRw, // the renderer's objects
    Binding::StorageRw, // the renderer's bond lines
    Binding::Storage,   // the shots' bodies
    Binding::Storage,   // strike geometry
    Binding::Storage,   // strike shots
    Binding::Storage,   // ext (load path, analytic strike)
    Binding::StorageRw, // lp
    Binding::StorageRw, // the engine's contact rows (contact fracture)
    Binding::Storage,   // the engine's counters
    Binding::StorageRw, // the contact rows' saved anchors (contact fracture)
    Binding::Storage,   // the load path's bond sides
];

const ENTRIES: [&str; 73] = [
    "fr_loads",
    "fr_breaks",
    "fr_gate",
    "fr_uf_init",
    "fr_body_init",
    "fr_uf_union",
    "fr_uf_flatten",
    "fr_best_count",
    "fr_best_root",
    "fr_mark",
    "fr_scan_blocks",
    "fr_scan_top",
    "fr_scan_add",
    "fr_freelist",
    "fr_assign",
    "fr_piece",
    "fr_commit",
    "fr_reparent",
    "fr_body_clear",
    "fr_view_chunks",
    "fr_view_shots",
    "fr_view_bonds",
    "fr_strike_reset",
    "fr_strike_bounds",
    "fr_strike_find",
    "fr_strike_entry",
    "fr_strike_nearest",
    "fr_strike_force",
    "fr_strike_breaks",
    "fr_lp_clear",
    "fr_lp_sum",
    "fr_lp_relief",
    "fr_lp_block",
    "fr_lp_sweep_even",
    "fr_lp_sweep_odd",
    "fr_lp_breaks",
    "fr_il_first_reset",
    "fr_il_first",
    "fr_il_begin",
    "fr_il_tube_count",
    "fr_il_tube_base",
    "fr_il_tube_collect",
    "fr_il_run",
    "fr_il_strike",
    "fr_il_mark",
    "fr_il_breaks",
    "fr_il_carry",
    "fr_cs_begin",
    "fr_cs_contacts",
    "fr_cs_pick_a",
    "fr_cs_pick_b",
    "fr_cs_pick_c",
    "fr_cs_pick_d",
    "fr_cs_record",
    "fr_cs_hertz",
    "fr_cs_count",
    "fr_cs_base",
    "fr_cs_collect",
    "fr_cs_gate",
    "fr_cs_solve",
    "fr_cs_gate2",
    "fr_cs_save",
    "fr_cs_fix",
    "fr_cs_clean",
    "fr_csl_begin",
    "fr_csl_tube_count",
    "fr_csl_tube_base",
    "fr_csl_tube_collect",
    "fr_csl_run",
    "fr_csl_strike",
    "fr_csl_mark",
    "fr_csl_breaks",
    "fr_csl_carry",
];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewParams {
    mode: u32,
    shot_count: u32,
    impactor_base: u32,
    pad0: u32,
    rest_speed: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

/// What the view kernels draw from (`fracture.slang` `fr_view_*`), set once per renderer.
pub struct ViewTables {
    /// Per chunk 3 float4: (renderer object as bits, render centre (structure frame));
    /// rotation (structure frame, x, y, z, w); (material colour, -).
    pub chunks: Vec<[[f32; 4]; 3]>,
    /// The shots' engine bodies.
    pub shot_bodies: Vec<u32>,
    /// The renderer's first shot object.
    pub impactor_base: u32,
}

struct View {
    params: wgpu::Buffer,
    chunks: wgpu::Buffer,
    shots: wgpu::Buffer,
    shot_count: u32,
    impactor_base: u32,
    /// The renderer's buffers the groups bind, and the groups (chunks, shots, bonds).
    bound: Option<(wgpu::Buffer, wgpu::Buffer, [wgpu::BindGroup; 3])>,
    mode: u32,
}

fn cs_pass<'e>(encoder: &'e mut wgpu::CommandEncoder, at: Option<wgpu::ComputePassTimestampWrites<'_>>) -> wgpu::ComputePass<'e> {
    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("contact fracture"), timestamp_writes: at })
}

/// The in-step contact fracture's timestamps (`STRESS_CS_TIMING=1`).
struct CsTiming {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
}

const CS_STAMPS: u32 = 6;
/// The load path's frame stages' timestamps, after the in-step pass's.
const LP_STAMPS: u32 = 6;

impl CsTiming {
    fn new(gpu: &Gpu) -> Option<CsTiming> {
        if !std::env::var("STRESS_CS_TIMING").is_ok_and(|v| v == "1") || !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let n = CS_STAMPS + LP_STAMPS;
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("contact fracture timing"), ty: wgpu::QueryType::Timestamp, count: n });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("contact fracture timing"), size: 256 + 8 * LP_STAMPS as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = storage_bytes(gpu, "contact fracture timing", 8 * n as u64, wgpu::BufferUsages::empty());
        Some(CsTiming { queries, resolve, read })
    }
}

pub struct GpuFracture {
    pub layout: Layout,
    kernels: Vec<Kernel>,
    params: FractureParams,
    params_buf: wgpu::Buffer,
    scan_new: wgpu::Buffer,
    scan_free: wgpu::Buffer,
    pub chunks: wgpu::Buffer,
    pub info: wgpu::Buffer,
    pub bonds: wgpu::Buffer,
    pub parts: wgpu::Buffer,
    pub work: wgpu::Buffer,
    pub atom: wgpu::Buffer,
    pub frames: wgpu::Buffer,
    pub loads: wgpu::Buffer,
    /// The gate's sizes (`fr_gate` writes them), and their copy the indirect dispatches read
    /// (a buffer bound for writing cannot drive a dispatch in the same usage scope).
    pub sizes: wgpu::Buffer,
    pub args: wgpu::Buffer,
    pub geometry: wgpu::Buffer,
    pub shots: wgpu::Buffer,
    pub ext: wgpu::Buffer,
    pub lp: wgpu::Buffer,
    lp_sides: wgpu::Buffer,
    /// Per engine contact row the in-step contact fracture's saved anchors (2 float4).
    cs_anchors: wgpu::Buffer,
    /// `STRESS_CS_TIMING=1`: its stages' timestamps.
    cs_timing: Option<CsTiming>,
    engine_counters: Option<wgpu::Buffer>,
    /// Bind groups per kernel (the scans: the new-piece scan's, then the free slots').
    binds: Vec<wgpu::BindGroup>,
    scan_binds: Vec<wgpu::BindGroup>,
    /// What the core kernels bind in the view's slots, and the view once set.
    /// One per view slot (a buffer bound read-only and read-write in one dispatch is
    /// invalid): view chunks, objects, lines, shots.
    placeholder: [wgpu::Buffer; 4],
    placeholder_uniform: wgpu::Buffer,
    view: Option<View>,
    engine_bodies: Option<wgpu::Buffer>,
    engine_colliders: Option<wgpu::Buffer>,
    engine_contacts: Option<wgpu::Buffer>,
    ingest_buffers: Option<[wgpu::Buffer; 3]>,
}

fn storage_bytes(gpu: &Gpu, label: &str, bytes: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

impl GpuFracture {
    /// The fracture of `t`, bound to the engine's buffers and the ingest's (both must keep
    /// their buffers while it lives; [`Self::rebind`] after either replaces one).
    pub fn new(gpu: &Gpu, t: &FractureTables, engine: &EngineBuffers, ingest: &EngineIngest) -> Result<GpuFracture, String> {
        let (n, e, b) = (t.chunk_count(), t.bond_count(), t.body_count());
        // The scans' two levels hold 2^20 words (`fr_scan_top`: one group of 1024 blocks).
        if n > 1 << 20 || b > 1 << 20 {
            return Err(format!("fracture: {n} chunks and {b} body slots; the scans hold at most 2^20 of each"));
        }
        if t.info.len() != n as usize || t.chunk_body.len() != n as usize || t.free.len() != b as usize {
            return Err("fracture: tables of mismatched lengths".into());
        }
        let layout = Layout { n, e, b };
        let [scale_m, scale_s, scale_j] = t.scales();
        let (ext_words, ext_chunks, ext_adj, ext_mat) = t.ext.words();
        let needs_ext = t.load_model != LoadModel::Threshold || t.strike_model != StrikeModel::Hertz;
        if needs_ext && (t.ext.bonds.len() != e as usize || t.ext.chunks.len() != n as usize) {
            return Err("fracture: the load path's and analytic strike's tables do not match the bonds and chunks".into());
        }
        if !t.lp_init.is_empty() && t.lp_init.len() != n as usize {
            return Err("fracture: the load path's warm start does not match the chunks".into());
        }
        if t.contact_sweeps % 2 != 0 {
            return Err(format!("fracture: {} contact-fracture sweeps; an even count", t.contact_sweeps));
        }
        if t.contact_sweeps > 0 && t.geometry.len() != n as usize {
            return Err("fracture: the in-step contact fracture needs the strike geometry".into());
        }
        let (cs_c1, cs_c2) = stress_ref::load_path::momentum_coefficients(stress_ref::load_path::IMPACT_LOWER);
        let (sweeps, lp_c1, lp_c2) = match t.load_model {
            LoadModel::LoadPath { sweeps } => {
                if sweeps == 0 || sweeps % 2 != 0 {
                    return Err(format!("fracture: {sweeps} load-path sweeps; an even count of at least 2"));
                }
                let (c1, c2) = stress_ref::load_path::momentum_coefficients(stress_ref::load_path::LOWER);
                (sweeps, c1 as f32, c2 as f32)
            }
            LoadModel::Threshold => (0, 0.0, 0.0),
        };
        let params = FractureParams {
            chunk_count: n,
            bond_count: e,
            body_count: b,
            max_bodies: t.max_bodies,
            step: 0,
            pool_cap: t.pool_cap,
            impact_peak: t.impact_peak as u32,
            dt: t.dt,
            band: t.band,
            scale_m,
            scale_s,
            scale_j,
            strike_speed: t.strike_speed,
            load_model: matches!(t.load_model, LoadModel::LoadPath { .. }) as u32,
            strike_model: matches!(t.strike_model, StrikeModel::Analytic { .. }) as u32,
            fracture_work: matches!(t.strike_model, StrikeModel::Analytic { fracture_work: true }) as u32,
            sweeps,
            gx: t.gravity[0],
            gy: t.gravity[1],
            gz: t.gravity[2],
            lp_c1,
            lp_c2,
            ext_chunks,
            ext_adj,
            ext_mat,
            lp_scale: LP_SCALE,
            cs_sweeps: t.contact_sweeps,
            cs_row_cap: t.max_contacts,
            cs_c1: cs_c1 as f32,
            cs_c2: cs_c2 as f32,
            ..Default::default()
        };
        if !t.geometry.is_empty() && t.geometry.len() != n as usize {
            return Err("fracture: strike geometry of mismatched length".into());
        }
        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fracture params"),
            size: std::mem::size_of::<FractureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        let scan_new = gpu.uniform("fracture scan new", &ScanParams { n, in_base: layout.w_new(), out_base: layout.w_rank(), block_base: layout.scan_new_blocks() });
        let scan_free = gpu.uniform("fracture scan free", &ScanParams { n: b, in_base: layout.w_free(), out_base: layout.w_free_rank(), block_base: layout.scan_free_blocks() });
        let flat = |v: &[[[f32; 4]; 4]]| -> Vec<[f32; 4]> { v.iter().flatten().copied().collect() };
        let chunks = gpu.storage("fracture chunks", &flat(&t.chunks));
        let info = gpu.storage("fracture chunk info", &t.info);
        let bonds = gpu.storage("fracture bonds", &t.bonds);
        let parts: Vec<[f32; 4]> = t.parts.iter().flatten().copied().collect();
        let parts = gpu.storage("fracture parts", &parts);
        // work: each chunk's body, every bond intact, the free slots; the rest the kernels write.
        let mut words = vec![0u32; layout.work_words() as usize];
        words[..n as usize].copy_from_slice(&t.chunk_body);
        words[layout.w_alive() as usize..(layout.w_alive() + e) as usize].fill(1);
        words[layout.w_free() as usize..(layout.w_free() + b) as usize].copy_from_slice(&t.free);
        let work = gpu.storage("fracture work", &words);
        let atom = storage_bytes(gpu, "fracture atom", 4 * layout.atom_words() as u64, wgpu::BufferUsages::empty());
        let mut frames: Vec<[f32; 4]> = t.frames.iter().flatten().copied().collect();
        // + staging (9 per chunk) + the strikes (3 per body slot).
        frames.resize((2 * b + 9 * n + 3 * b) as usize, [0.0; 4]);
        let frames = gpu.storage("fracture frames", &frames);
        // Per chunk its load, the pulse pool's two halves, per chunk its force and moment.
        let loads = storage_bytes(gpu, "fracture loads", 16 * (n as u64 + 2 * 3 * t.pool_cap as u64 + 2 * n as u64), wgpu::BufferUsages::empty());
        let sizes = storage_bytes(gpu, "fracture sizes", 12 * ARGS, wgpu::BufferUsages::empty());
        let args = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fracture args"),
            size: 12 * ARGS,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g: Vec<[f32; 4]> = t.geometry.iter().flatten().copied().collect();
        let geometry = storage_bytes(gpu, "fracture strike geometry", 16 * g.len() as u64, wgpu::BufferUsages::empty());
        if !g.is_empty() {
            gpu.queue.write_buffer(&geometry, 0, bytemuck::cast_slice(&g));
        }
        let shots = storage_bytes(gpu, "fracture strike shots", 32 * MAX_SHOTS as u64, wgpu::BufferUsages::empty());
        let ext = storage_bytes(gpu, "fracture ext", 16 * ext_words.len() as u64, wgpu::BufferUsages::empty());
        if !ext_words.is_empty() {
            gpu.queue.write_buffer(&ext, 0, bytemuck::cast_slice(&ext_words));
        }
        let mut lp_words = vec![[0.0f32; 4]; layout.lp_words() as usize];
        for (k, x) in t.lp_init.iter().enumerate() {
            for copy in 0..2 {
                lp_words[LP_CHUNK as usize * k + 2 * copy] = x[0];
                lp_words[LP_CHUNK as usize * k + 2 * copy + 1] = x[1];
            }
        }
        let lp = gpu.storage("fracture lp", &lp_words);
        let side_words = if needs_ext || t.contact_sweeps > 0 && !t.ext.bonds.is_empty() { sides(&t.ext, &t.chunks, &t.bonds) } else { Vec::new() };
        let lp_sides = storage_bytes(gpu, "fracture lp sides", 16 * side_words.len() as u64, wgpu::BufferUsages::empty());
        if !side_words.is_empty() {
            gpu.queue.write_buffer(&lp_sides, 0, bytemuck::cast_slice(&side_words));
        }
        let kernels = ENTRIES.iter().map(|entry| gpu.compute(&shaders::FRACTURE, entry, &BINDINGS)).collect();
        let mut f = GpuFracture {
            layout,
            kernels,
            params,
            params_buf,
            scan_new,
            scan_free,
            chunks,
            info,
            bonds,
            parts,
            work,
            atom,
            frames,
            loads,
            sizes,
            args,
            geometry,
            shots,
            ext,
            lp,
            lp_sides,
            cs_anchors: storage_bytes(gpu, "fracture contact anchors", 32 * t.max_contacts as u64, wgpu::BufferUsages::empty()),
            cs_timing: CsTiming::new(gpu),
            engine_counters: None,
            binds: Vec::new(),
            scan_binds: Vec::new(),
            placeholder: std::array::from_fn(|_| storage_bytes(gpu, "fracture view placeholder", 16, wgpu::BufferUsages::empty())),
            placeholder_uniform: gpu.uniform("fracture view placeholder", &ViewParams::default()),
            view: None,
            engine_bodies: None,
            engine_colliders: None,
            engine_contacts: None,
            ingest_buffers: None,
        };
        f.rebind(gpu, engine, ingest);
        Ok(f)
    }

    /// The bind groups, against the engine's and the ingest's buffers as they are now.
    pub fn rebind(&mut self, gpu: &Gpu, engine: &EngineBuffers, ingest: &EngineIngest) {
        let group = |scan: &wgpu::Buffer, k: &Kernel| {
            let buffers: [&wgpu::Buffer; 29] = [
                &self.params_buf,
                &self.chunks,
                &self.bonds,
                &self.parts,
                &self.work,
                &self.atom,
                &self.frames,
                engine.bodies,
                engine.colliders,
                &ingest.collider_info,
                &ingest.manifolds,
                &ingest.sort,
                &self.loads,
                &self.sizes,
                scan,
                &self.info,
                &self.placeholder_uniform,
                &self.placeholder[0],
                &self.placeholder[1],
                &self.placeholder[2],
                &self.placeholder[3],
                &self.geometry,
                &self.shots,
                &self.ext,
                &self.lp,
                engine.contacts,
                engine.counters,
                &self.cs_anchors,
                &self.lp_sides,
            ];
            gpu.bind(&k.group, &buffers)
        };
        let binds = self.kernels.iter().map(|k| group(&self.scan_new, k)).collect();
        let scan_binds = [10, 11, 12].iter().map(|&i| group(&self.scan_free, &self.kernels[i])).collect();
        self.binds = binds;
        self.scan_binds = scan_binds;
        if let Some(v) = self.view.as_mut() {
            v.bound = None;
        }
        self.engine_bodies = Some(engine.bodies.clone());
        self.engine_counters = Some(engine.counters.clone());
        self.engine_contacts = Some(engine.contacts.clone());
        self.engine_colliders = Some(engine.colliders.clone());
        self.ingest_buffers = Some([ingest.collider_info.clone(), ingest.manifolds.clone(), ingest.sort.clone()]);
    }

    /// The view's tables (once per renderer layout).
    pub fn set_view(&mut self, gpu: &Gpu, t: &ViewTables) {
        let chunks: Vec<[f32; 4]> = t.chunks.iter().flatten().copied().collect();
        self.view = Some(View {
            params: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fracture view params"),
                size: std::mem::size_of::<ViewParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            chunks: gpu.storage("fracture view chunks", &chunks),
            shots: gpu.storage("fracture view shots", &t.shot_bodies),
            shot_count: t.shot_bodies.len() as u32,
            impactor_base: t.impactor_base,
            bound: None,
            mode: u32::MAX,
        });
    }

    /// The renderer's objects (`objects`: chunks and shots) and, with `bonds`, its bond
    /// lines (`lines`, 2 vertices per bond) from the engine's bodies as they are: recorded
    /// into `encoder` after the frame's work. `mode`: 0 materials, 1 body state, 2 bodies.
    pub fn encode_view(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, objects: &wgpu::Buffer, lines: &wgpu::Buffer, mode: u32, bonds: bool) {
        let Some(v) = self.view.as_mut() else { return };
        if v.mode != mode {
            v.mode = mode;
            let p = ViewParams { mode, shot_count: v.shot_count, impactor_base: v.impactor_base, rest_speed: 0.05, ..Default::default() };
            gpu.queue.write_buffer(&v.params, 0, bytemuck::bytes_of(&p));
        }
        if v.bound.as_ref().is_none_or(|(o, l, _)| o != objects || l != lines) {
            let bodies = self.engine_bodies.as_ref().expect("engine bound");
            let colliders = self.engine_colliders.as_ref().expect("engine bound");
            let ing = self.ingest_buffers.as_ref().expect("ingest bound");
            let group = |i: usize| {
                let counters = self.engine_counters.as_ref().expect("engine bound");
                let contacts = self.engine_contacts.as_ref().expect("engine bound");
                let buffers: [&wgpu::Buffer; 29] = [
                    &self.params_buf, &self.chunks, &self.bonds, &self.parts, &self.work, &self.atom, &self.frames, bodies, colliders, &ing[0], &ing[1], &ing[2], &self.loads, &self.sizes, &self.scan_new, &self.info, &v.params, &v.chunks, objects, lines, &v.shots, &self.geometry, &self.shots, &self.ext, &self.lp, contacts, counters, &self.cs_anchors, &self.lp_sides,
                ];
                gpu.bind(&self.kernels[i].group, &buffers)
            };
            let at = |name: &str| ENTRIES.iter().position(|e| *e == name).expect("view entry");
            let groups = [group(at("fr_view_chunks")), group(at("fr_view_shots")), group(at("fr_view_bonds"))];
            v.bound = Some((objects.clone(), lines.clone(), groups));
        }
        let (_, _, groups) = v.bound.as_ref().expect("bound");
        let mut pass = encoder.begin_compute_pass(&Default::default());
        let run = |pass: &mut wgpu::ComputePass<'_>, name: &str, group: &wgpu::BindGroup, n: u32| {
            let k = &self.kernels[ENTRIES.iter().position(|e| *e == name).expect("view entry")];
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups_of(n), 1, 1);
        };
        run(&mut pass, "fr_view_chunks", &groups[0], self.layout.n);
        run(&mut pass, "fr_view_shots", &groups[1], v.shot_count.max(1));
        if bonds {
            run(&mut pass, "fr_view_bonds", &groups[2], self.layout.e);
        }
    }

    fn kernel(&self, name: &str) -> (usize, &Kernel) {
        let i = ENTRIES.iter().position(|e| *e == name).expect("fracture entry");
        (i, &self.kernels[i])
    }

    /// The frame's loads and breaks, then its splits (gated on the GPU). `step` is the
    /// engine's next step (a re-parented body's generation).
    pub fn encode_frame(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, step: u32) {
        // The pool's halves swap every frame.
        self.params.parity ^= 1;
        self.params.step = step;
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        let (n, e, b) = (self.layout.n as usize, self.layout.e as usize, self.layout.b as usize);
        if self.params.load_model == 1 && self.cs_timing.is_some() {
            // `STRESS_CS_TIMING=1`: the load path's stages, a pass each.
            let at = |i: u32| {
                let t = self.cs_timing.as_ref().expect("timing");
                wgpu::ComputePassTimestampWrites { query_set: &t.queries, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(CS_STAMPS + i) }
            };
            drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(at(0)) }));
            let stages: [&[&str]; 5] = [&["fr_loads"], &["fr_lp_clear", "fr_lp_sum", "fr_lp_relief"], &["fr_lp_block"], &[], &["fr_lp_breaks"]];
            for (i, names) in stages.iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(at(i as u32 + 1)) });
                for name in names.iter() {
                    let size = match *name {
                        "fr_lp_clear" => b,
                        "fr_lp_breaks" => e,
                        _ => n,
                    };
                    self.direct(&mut pass, name, size);
                }
                if i == 3 {
                    for s in 0..self.params.sweeps {
                        self.direct(&mut pass, if s % 2 == 0 { "fr_lp_sweep_even" } else { "fr_lp_sweep_odd" }, n);
                    }
                }
            }
            let t = self.cs_timing.as_ref().expect("timing");
            encoder.resolve_query_set(&t.queries, CS_STAMPS..CS_STAMPS + LP_STAMPS, &t.resolve, 256);
            encoder.copy_buffer_to_buffer(&t.resolve, 256, &t.read, 8 * CS_STAMPS as u64, 8 * LP_STAMPS as u64);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            if self.params.load_model == 1 && self.cs_timing.is_some() {
                // Done above, timed.
            } else {
            self.direct(&mut pass, "fr_loads", n);
            if self.params.load_model == 0 {
                self.direct(&mut pass, "fr_breaks", e);
            } else {
                self.direct(&mut pass, "fr_lp_clear", b);
                self.direct(&mut pass, "fr_lp_sum", n);
                self.direct(&mut pass, "fr_lp_relief", n);
                self.direct(&mut pass, "fr_lp_block", n);
                for s in 0..self.params.sweeps {
                    self.direct(&mut pass, if s % 2 == 0 { "fr_lp_sweep_even" } else { "fr_lp_sweep_odd" }, n);
                }
                self.direct(&mut pass, "fr_lp_breaks", e);
            }
            }
            if self.params.shot_count > 0 {
                self.direct(&mut pass, "fr_strike_reset", b);
                self.direct(&mut pass, "fr_strike_bounds", n);
                self.direct(&mut pass, "fr_strike_find", b);
                self.direct(&mut pass, "fr_strike_entry", n);
                self.direct(&mut pass, "fr_strike_nearest", n);
                self.direct(&mut pass, "fr_strike_force", b);
                if self.params.strike_model == 0 {
                    self.direct(&mut pass, "fr_strike_breaks", e);
                } else {
                    self.direct(&mut pass, "fr_il_first_reset", MAX_SHOTS);
                    self.direct(&mut pass, "fr_il_first", b);
                    self.direct(&mut pass, "fr_il_begin", b);
                    self.direct(&mut pass, "fr_il_tube_count", n);
                    self.direct(&mut pass, "fr_il_tube_base", b);
                    self.direct(&mut pass, "fr_il_tube_collect", n);
                    // A group per body slot.
                    self.direct(&mut pass, "fr_il_run", 64 * b);
                    self.direct(&mut pass, "fr_il_strike", b);
                    self.direct(&mut pass, "fr_il_mark", n);
                    self.direct(&mut pass, "fr_il_breaks", e);
                }
            }
            self.direct(&mut pass, "fr_gate", 1);
        }
        self.record_split(encoder);
        if self.params.strike_model != 0 && self.params.shot_count > 0 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.direct(&mut pass, "fr_il_carry", n);
        }
    }

    /// The in-step contact fracture (`fracture.slang` `fr_cs_*`), encoded within the
    /// engine's step where its contacts are made and kept (`WgpuSim::step_encode_with` at
    /// `"warm"`, its rows from [`Self::set_contact_rows`]): this step's contacts closing
    /// faster than `strike_speed` on a structure are its strikes, each struck structure's
    /// largest judged by the profile's stress solver (the threshold rule, the load path or
    /// the analytic strike), what breaks split, and the rows of the moved colliders
    /// re-anchored to their pieces, before the step solves. Nothing is read back; without a
    /// strike the solve and the split dispatch nothing.
    pub fn encode_contact_fracture(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.params.cs_sweeps == 0 {
            return;
        }
        let load_path = self.params.load_model == 1;
        let analytic = !load_path && self.params.strike_model == 1;
        let stamp = |i: u32, end: bool| -> Option<wgpu::ComputePassTimestampWrites<'_>> {
            self.cs_timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites {
                query_set: &t.queries,
                beginning_of_pass_write_index: (!end).then_some(i),
                end_of_pass_write_index: end.then_some(i),
            })
        };
        // The detection, sized by the step's contact rows; then everything after it sized by
        // what it found (`fr_cs_gate`): a step without a strike dispatches no group past here.
        {
            let mut p = cs_pass(encoder, stamp(0, false));
            self.direct(&mut p, "fr_cs_begin", 1);
        }
        encoder.copy_buffer_to_buffer(&self.sizes, 0, &self.args, 0, 12 * ARGS);
        {
            let mut p = cs_pass(encoder, stamp(1, true));
            self.indirect(&mut p, "fr_cs_contacts", ARG_CS_ROWS, None);
            self.direct(&mut p, "fr_cs_gate", 1);
        }
        encoder.copy_buffer_to_buffer(&self.sizes, 0, &self.args, 0, 12 * ARGS);
        {
            // The profile's verdict.
            let mut p = cs_pass(encoder, stamp(2, true));
            for stage in ["fr_cs_pick_a", "fr_cs_pick_b", "fr_cs_pick_c", "fr_cs_pick_d"] {
                self.indirect(&mut p, stage, ARG_CS_PICK, None);
            }
            self.indirect(&mut p, "fr_cs_record", ARG_CS_LIST, None);
            if load_path {
                self.indirect(&mut p, "fr_cs_count", ARG_CS_CHUNKS, None);
                self.indirect(&mut p, "fr_cs_base", ARG_CS_LIST, None);
                self.indirect(&mut p, "fr_cs_collect", ARG_CS_CHUNKS, None);
                // A group per struck structure, all its sweeps in the one dispatch.
                self.indirect(&mut p, "fr_cs_solve", ARG_CS_BODIES, None);
            } else if analytic {
                self.indirect(&mut p, "fr_csl_begin", ARG_CS_SLOTS, None);
                self.indirect(&mut p, "fr_csl_tube_count", ARG_CS_CHUNKS, None);
                self.indirect(&mut p, "fr_csl_tube_base", ARG_CS_SLOTS, None);
                self.indirect(&mut p, "fr_csl_tube_collect", ARG_CS_CHUNKS, None);
                // A group per struck structure.
                self.indirect(&mut p, "fr_csl_run", ARG_CS_BODIES, None);
                self.indirect(&mut p, "fr_csl_strike", ARG_CS_SLOTS, None);
                self.indirect(&mut p, "fr_csl_mark", ARG_CS_CHUNKS, None);
                self.indirect(&mut p, "fr_csl_breaks", ARG_CS_BONDS, None);
            } else {
                self.indirect(&mut p, "fr_cs_hertz", ARG_CS_BONDS, None);
            }
            self.direct(&mut p, "fr_cs_gate2", 1);
        }
        encoder.copy_buffer_to_buffer(&self.sizes, 0, &self.args, 0, 12 * ARGS);
        {
            let mut p = cs_pass(encoder, stamp(3, true));
            self.indirect(&mut p, "fr_cs_save", ARG_CS_SAVE, None);
            self.direct(&mut p, "fr_gate", 1);
        }
        self.record_split(encoder);
        {
            let mut p = cs_pass(encoder, stamp(4, false));
            self.indirect(&mut p, "fr_cs_fix", ARG_CS_SAVE, None);
            if analytic {
                self.indirect(&mut p, "fr_csl_carry", ARG_CS_CHUNKS, None);
            }
            self.indirect(&mut p, "fr_cs_clean", ARG_CS_LIST, None);
        }
        drop(cs_pass(encoder, stamp(5, true)));
        if let Some(t) = &self.cs_timing {
            encoder.resolve_query_set(&t.queries, 0..CS_STAMPS, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.read, 0, 8 * CS_STAMPS as u64);
        }
    }

    /// `STRESS_CS_TIMING=1` (a diagnostic, where the device has timestamp queries): the
    /// in-step contact fracture's GPU time by stage in its last step (µs): the detection,
    /// the verdict, the save and gate, the split, the fix and clean. Waits for the GPU.
    pub fn contact_timing(&self, gpu: &Gpu) -> Option<[f64; 5]> {
        let t = self.cs_timing.as_ref()?;
        let w: Vec<u64> = gpu.read_range(&t.read, 0, CS_STAMPS as usize);
        let period = gpu.queue.get_timestamp_period() as f64 * 1e-3;
        let d = |a: usize, b: usize| w[b].saturating_sub(w[a]) as f64 * period;
        Some([d(0, 1), d(1, 2), d(2, 3), d(3, 4), d(4, 5)])
    }

    /// `STRESS_CS_TIMING=1`: the load path's frame stages in the last frame (µs): the
    /// loads, the sums and relief, the blocks, the sweeps, the breaks.
    pub fn load_path_timing(&self, gpu: &Gpu) -> Option<[f64; 5]> {
        let t = self.cs_timing.as_ref()?;
        let w: Vec<u64> = gpu.read_range(&t.read, CS_STAMPS as usize, LP_STAMPS as usize);
        let period = gpu.queue.get_timestamp_period() as f64 * 1e-3;
        let d = |a: usize, b: usize| w[b].saturating_sub(w[a]) as f64 * period;
        Some([d(0, 1), d(1, 2), d(2, 3), d(3, 4), d(4, 5)])
    }

    /// Where the coming step's contact rows start in the engine's contact buffer
    /// (`WgpuSim::next_contact_offset`), for the in-step contact fracture.
    pub fn set_contact_rows(&mut self, gpu: &Gpu, offset: u32) {
        if self.params.cs_row_offset != offset {
            self.params.cs_row_offset = offset;
            gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        }
    }

    /// Diagnostic read: each body slot's in-step contact strike record (meaningful for a
    /// body struck in the last step that had a strike).
    pub fn contact_strikes(&self, gpu: &Gpu) -> Vec<ContactStrike> {
        let w: Vec<[f32; 4]> = gpu.read_range(&self.lp, (LP_CHUNK * self.layout.n) as usize, (LP_BODY * self.layout.b) as usize);
        w.chunks_exact(LP_BODY as usize)
            .map(|c| ContactStrike {
                contact: [c[2][0], c[2][1], c[2][2]],
                peak: c[2][3],
                direction: [c[3][0], c[3][1], c[3][2]],
                speed: c[3][3],
                m_eff: c[4][0],
                face: c[4][1],
                modulus: c[4][2],
                body: c[4][3].to_bits(),
                chunk: c[5][0].to_bits(),
                hertz: c[5][1],
                mass: c[5][2],
                t_in: c[5][3],
            })
            .collect()
    }

    /// Diagnostic read: each chunk's in-step contact increment (translation, rotation;
    /// structure frame) and its load (force with its fixed flag, moment), as the last
    /// strike left them.
    pub fn contact_increments(&self, gpu: &Gpu) -> Vec<[[f32; 4]; 4]> {
        let w: Vec<[f32; 4]> = gpu.read_range(&self.lp, 0, (LP_CHUNK * self.layout.n) as usize);
        w.chunks_exact(LP_CHUNK as usize).map(|c| [c[CS_STATE], c[CS_STATE + 1], c[CS_STATE + 6], c[CS_STATE + 7]]).collect()
    }

    /// The shots strike prediction checks (in order: the first that reaches a body is its
    /// strike), at most [`MAX_SHOTS`]. None until set. Needs the tables' `geometry`.
    pub fn set_shots(&mut self, gpu: &Gpu, shots: &[Shot]) -> Result<(), String> {
        if shots.len() > MAX_SHOTS {
            return Err(format!("fracture: {} shots; strike prediction holds at most {MAX_SHOTS}", shots.len()));
        }
        if !shots.is_empty() && self.layout.n > 0 && self.geometry.size() < 48 * self.layout.n as u64 {
            return Err("fracture: shots set without strike geometry".into());
        }
        let words: Vec<[f32; 4]> = shots.iter().flat_map(|s| [[f32::from_bits(s.body), s.bound, s.face, s.mass], [s.modulus, f32::from_bits(s.launch), 0.0, 0.0]]).collect();
        if !words.is_empty() {
            gpu.queue.write_buffer(&self.shots, 0, bytemuck::cast_slice(&words));
        }
        self.params.shot_count = shots.len() as u32;
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        Ok(())
    }

    /// Only the split (the gate, then every split kernel): the bonds already marked broken
    /// and their bodies dirty (the tests' entry, and a frame's second half).
    pub fn encode_split(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, step: u32) {
        self.set_step(gpu, step);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.direct(&mut pass, "fr_gate", 1);
        }
        self.record_split(encoder);
    }

    fn set_step(&mut self, gpu: &Gpu, step: u32) {
        if self.params.step != step {
            self.params.step = step;
            gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        }
    }

    fn direct(&self, pass: &mut wgpu::ComputePass<'_>, name: &str, n: usize) {
        let (i, k) = self.kernel(name);
        k.dispatch(pass, &self.binds[i], n);
    }

    fn indirect(&self, pass: &mut wgpu::ComputePass<'_>, name: &str, arg: u64, scan: Option<usize>) {
        let (i, k) = self.kernel(name);
        pass.set_pipeline(&k.pipeline);
        let bind = match scan {
            Some(s) => &self.scan_binds[s],
            None => &self.binds[i],
        };
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups_indirect(&self.args, 12 * arg);
    }

    /// The split kernels, sized by the gate (`fr_gate`, in the pass before).
    fn record_split(&self, encoder: &mut wgpu::CommandEncoder) {
        // `STRESS_FRACTURE_SPLIT=0` (a diagnostic): bonds break, bodies never split.
        if std::env::var("STRESS_FRACTURE_SPLIT").is_ok_and(|v| v == "0") {
            return;
        }
        encoder.copy_buffer_to_buffer(&self.sizes, 0, &self.args, 0, 12 * ARGS);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        let pass = &mut pass;
        self.indirect(pass, "fr_uf_init", ARG_CHUNKS, None);
        self.indirect(pass, "fr_body_init", ARG_BODIES, None);
        self.indirect(pass, "fr_uf_union", ARG_BONDS, None);
        self.indirect(pass, "fr_uf_flatten", ARG_CHUNKS, None);
        self.indirect(pass, "fr_best_count", ARG_CHUNKS, None);
        self.indirect(pass, "fr_best_root", ARG_CHUNKS, None);
        self.indirect(pass, "fr_mark", ARG_CHUNKS, None);
        // The new pieces ranked, then the free slots listed (ascending).
        self.indirect(pass, "fr_scan_blocks", ARG_SCAN_CHUNKS, None);
        self.indirect(pass, "fr_scan_top", ARG_ONE, None);
        self.indirect(pass, "fr_scan_add", ARG_CHUNKS, None);
        self.indirect(pass, "fr_scan_blocks", ARG_SCAN_BODIES, Some(0));
        self.indirect(pass, "fr_scan_top", ARG_ONE, Some(1));
        self.indirect(pass, "fr_scan_add", ARG_BODIES, Some(2));
        self.indirect(pass, "fr_freelist", ARG_BODIES, None);
        self.indirect(pass, "fr_assign", ARG_CHUNKS, None);
        self.indirect(pass, "fr_piece", ARG_CHUNKS, None);
        self.indirect(pass, "fr_commit", ARG_CHUNKS, None);
        self.indirect(pass, "fr_reparent", ARG_CHUNKS, None);
        self.indirect(pass, "fr_body_clear", ARG_BODIES, None);
    }

    /// Diagnostic read (waits for the GPU).
    pub fn counters(&self, gpu: &Gpu) -> Counters {
        let c: Vec<u32> = gpu.read_range(&self.atom, self.layout.a_counters() as usize, 16);
        Counters::from_words(&c)
    }

    /// Diagnostic read: each chunk's load-path displacement (translation, rotation;
    /// structure frame), as the last frame's sweeps left it.
    pub fn lp_displacements(&self, gpu: &Gpu) -> Vec<[[f32; 4]; 2]> {
        let w: Vec<[f32; 4]> = gpu.read_range(&self.lp, 0, (LP_CHUNK * self.layout.n) as usize);
        w.chunks_exact(LP_CHUNK as usize).map(|c| [c[0], c[1]]).collect()
    }

    /// Diagnostic read: each chunk's relieved load-path load (force with its fixed flag,
    /// moment; structure frame), as the last frame left it.
    pub fn lp_loads(&self, gpu: &Gpu) -> Vec<[[f32; 4]; 2]> {
        let w: Vec<[f32; 4]> = gpu.read_range(&self.lp, 0, (LP_CHUNK * self.layout.n) as usize);
        w.chunks_exact(LP_CHUNK as usize).map(|c| [c[6], c[7]]).collect()
    }

    /// Diagnostic read: each body slot's analytic strike record (meaningful for a body
    /// struck in the last frame).
    pub fn strike_records(&self, gpu: &Gpu) -> Vec<StrikeRecord> {
        let w: Vec<[f32; 4]> = gpu.read_range(&self.lp, (LP_CHUNK * self.layout.n) as usize, (LP_BODY * self.layout.b) as usize);
        w.chunks_exact(LP_BODY as usize)
            .map(|c| StrikeRecord {
                thickness: c[0][0],
                ballistic_speed: c[0][1],
                residual_speed: c[0][2],
                outcome: c[0][3].to_bits(),
                crater_depth: c[1][0],
                spall_depth: c[1][1],
                contact_radius: c[1][2],
                ballistic_energy: c[1][3],
            })
            .collect()
    }

    /// The load and strike models it runs.
    pub fn models(&self) -> (u32, u32) {
        (self.params.load_model, self.params.strike_model)
    }

    /// Diagnostic read: each chunk's engine body.
    pub fn chunk_bodies(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.read_range(&self.work, 0, self.layout.n as usize)
    }

    /// Diagnostic read: whether each bond is intact.
    pub fn bonds_intact(&self, gpu: &Gpu) -> Vec<bool> {
        let w: Vec<u32> = gpu.read_range(&self.work, self.layout.w_alive() as usize, self.layout.e as usize);
        w.into_iter().map(|x| x != 0).collect()
    }

    /// Diagnostic read: each engine body slot's frame (centre of mass and principal
    /// rotation in its structure's frame).
    pub fn body_frames(&self, gpu: &Gpu) -> Vec<FrameRecord> {
        let f: Vec<[f32; 4]> = gpu.read_range(&self.frames, 0, 2 * self.layout.b as usize);
        f.chunks_exact(2).map(|p| [p[0], p[1]]).collect()
    }

    /// Break `bonds` and mark their bodies dirty, as `fr_breaks` would (the tests' and the
    /// predictors' host entry): `intact` is every bond's state before, `chunk_body` every
    /// chunk's body. Two uploads, written before the next submission.
    pub fn break_bonds(&self, gpu: &Gpu, bonds: &[u32], intact: &[bool], chunk_body: &[u32], bond_table: &[[u32; 4]]) {
        let mut alive: Vec<u32> = intact.iter().map(|&b| b as u32).collect();
        let mut dirty = vec![0u32; self.layout.b as usize];
        let mut count = 0u32;
        for &i in bonds {
            if alive[i as usize] != 0 {
                alive[i as usize] = 0;
                dirty[chunk_body[bond_table[i as usize][0] as usize] as usize] = 1;
                count += 1;
            }
        }
        gpu.queue.write_buffer(&self.work, 4 * self.layout.w_alive() as u64, bytemuck::cast_slice(&alive));
        gpu.queue.write_buffer(&self.atom, 4 * self.layout.a_dirty() as u64, bytemuck::cast_slice(&dirty));
        // The frame's broken count (`C_FRAME_BROKEN`): the gate runs the split.
        gpu.queue.write_buffer(&self.atom, 4 * self.layout.a_counters() as u64, bytemuck::bytes_of(&count));
    }

    pub fn body_words() -> u32 {
        BODY_WORDS
    }

    pub fn groups(n: u32) -> u32 {
        groups(n as usize, 64)
    }
}

/// The split on the host (f64): which engine body each chunk ends on, every new piece's
/// slot, and each body's mass, centre of mass and inertia about it (structure frame),
/// after `broken` bonds break. The same rules as `fracture.slang`: pieces are the connected
/// components of each body over its intact bonds, rooted at their lowest chunk; a body's
/// largest piece (ties: the lowest root) keeps it; the k-th other piece by root takes the
/// k-th free slot.
fn groups_of(n: u32) -> u32 {
    n.div_ceil(64).max(1)
}

pub fn reference_split(t: &FractureTables, intact: &mut [bool], broken: &[u32], chunk_body: &mut [u32], free: &mut [u32]) -> Vec<(u32, f64, [f64; 3], [[f64; 3]; 3])> {
    let n = t.chunks.len();
    let mut dirty = vec![false; t.frames.len()];
    for &i in broken {
        if intact[i as usize] {
            intact[i as usize] = false;
            dirty[chunk_body[t.bonds[i as usize][0] as usize] as usize] = true;
        }
    }
    // Union-find over the dirty bodies' intact bonds, roots the lowest chunk.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (i, b) in t.bonds.iter().enumerate() {
        if !intact[i] || !dirty[chunk_body[b[0] as usize] as usize] {
            continue;
        }
        let (ra, rb) = (find(&mut parent, b[0] as usize), find(&mut parent, b[1] as usize));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let label: Vec<usize> = (0..n).map(|c| find(&mut parent, c)).collect();
    let mut count = vec![0u32; n];
    for c in 0..n {
        if dirty[chunk_body[c] as usize] {
            count[label[c]] += 1;
        }
    }
    let roots: Vec<usize> = (0..n).filter(|&c| dirty[chunk_body[c] as usize] && label[c] == c).collect();
    let mut best: std::collections::HashMap<u32, (u32, usize)> = std::collections::HashMap::new();
    for &r in &roots {
        let e = best.entry(chunk_body[r]).or_insert((0, usize::MAX));
        if count[r] > e.0 || (count[r] == e.0 && r < e.1) {
            *e = (count[r], r);
        }
    }
    let free_list: Vec<u32> = (0..free.len() as u32).filter(|&s| free[s as usize] != 0).collect();
    let mut slot = vec![u32::MAX; n];
    let mut k = 0;
    for &r in &roots {
        let s = chunk_body[r];
        slot[r] = if best[&s].1 == r {
            s
        } else {
            let f = free_list[k];
            k += 1;
            free[f as usize] = 0;
            f
        };
    }
    for c in 0..n {
        if dirty[chunk_body[c] as usize] {
            chunk_body[c] = slot[label[c]];
        }
    }
    // Every piece's mass properties (structure frame), by root.
    roots
        .iter()
        .map(|&r| {
            let mut m = 0.0;
            let mut s = [0.0f64; 3];
            for c in (0..n).filter(|&c| label[c] == r && chunk_body[c] == slot[r]) {
                let k = &t.chunks[c];
                m += k[0][3] as f64;
                for a in 0..3 {
                    s[a] += k[0][3] as f64 * k[0][a] as f64;
                }
            }
            let com = s.map(|x| x / m);
            let mut i = [[0.0f64; 3]; 3];
            for c in (0..n).filter(|&c| label[c] == r && chunk_body[c] == slot[r]) {
                let k = &t.chunks[c];
                let mc = k[0][3] as f64;
                let d = [k[0][0] as f64 - com[0], k[0][1] as f64 - com[1], k[0][2] as f64 - com[2]];
                let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let own = [[k[1][0], k[1][3], k[2][0]], [k[1][3], k[1][1], k[2][1]], [k[2][0], k[2][1], k[1][2]]];
                for a in 0..3 {
                    for b in 0..3 {
                        i[a][b] += own[a][b] as f64 + mc * (if a == b { dd } else { 0.0 } - d[a] * d[b]);
                    }
                }
            }
            (slot[r], m, com, i)
        })
        .collect()
}
