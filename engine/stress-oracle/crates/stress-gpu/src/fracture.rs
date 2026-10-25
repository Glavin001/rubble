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
    /// Chunk crushing (`fracture.slang` `fr_crush_*`), opt-in: `None` runs none of it.
    pub crush: Option<CrushTables>,
}

/// What chunk crushing is made from (engine/docs/destruction-reactions.md section 1).
#[derive(Clone, Debug, Default)]
pub struct CrushTables {
    /// Per chunk: (capPressure, cohesion, frictionSlope, crushEnergy), (crushViscosity,
    /// volume, crushable 0/1, -). An uncrushable chunk's material row is ignored.
    pub chunks: Vec<[[f32; 4]; 2]>,
    /// Per bond: (centroid (structure frame), area).
    pub bonds: Vec<[f32; 4]>,
    /// A crushed chunk's mass kept as `fragments` boxes (`pool`), else removed.
    pub conserve: bool,
    /// Fragments per crushed chunk (conserve): a power of two, at most 64.
    pub fragments: u32,
    /// The reserved fragment bodies (engine body, its one box collider): the budget; a
    /// crush past it is an error (`CRUSH_ERROR_FRAGMENTS`), never a silent drop.
    pub pool: Vec<[u32; 2]>,
    /// Crushes one frame reports at most (the events' room); more is an error.
    pub max_events: u32,
    /// Where a crushed chunk's body is parked (rubble frame): x = (slot % 1024) step,
    /// y, z = (slot / 1024) step.
    pub park_y: f32,
    pub park_step: f32,
}

/// `Counters::error` bit: a crush found the fragment pool spent.
pub const CRUSH_ERROR_FRAGMENTS: u32 = 16;
/// `Counters::error` bit: more crushes in a frame than `CrushTables::max_events`.
pub const CRUSH_ERROR_EVENTS: u32 = 32;
/// Words per crush event (`fracture.slang` `fr_crush_apply`): 6 float4.
pub const CRUSH_EVENT_WORDS: u32 = 24;
/// The counters word holding this frame's crush count (`C_CRUSH_EVENTS`).
pub const C_CRUSH_EVENTS: usize = 7;

/// A crush, as `fr_crush_apply` records it (rubble's frame, Y up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrushEvent {
    pub chunk: u32,
    /// Its first fragment's index in the pool (`CrushTables::pool`); `None`: removed.
    pub first_fragment: Option<u32>,
    /// 1 contact loads, 2 a predicted strike.
    pub cause: u32,
    /// Its body as the split left it (alone): solver position, orientation, velocities.
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: f32,
    pub volume: f32,
    /// The highest (p, q) it met (Pa).
    pub peak_pressure: f32,
    pub peak_deviator: f32,
    /// Each fragment's half extents (its box frame); zero when removed.
    pub fragment_half: [f32; 3],
}

impl CrushEvent {
    /// `count` events from their words (`GpuFracture::crush_events_range`).
    pub fn decode(words: &[[f32; 4]], count: u32) -> Vec<CrushEvent> {
        words
            .chunks_exact(6)
            .take(count as usize)
            .map(|r| {
                let first = r[0][1].to_bits();
                CrushEvent {
                    chunk: r[0][0].to_bits(),
                    first_fragment: (first != u32::MAX).then_some(first),
                    cause: r[0][2].to_bits() & 0xf,
                    position: [r[1][0], r[1][1], r[1][2]],
                    mass: r[1][3],
                    orientation: r[2],
                    linear_velocity: [r[3][0], r[3][1], r[3][2]],
                    peak_pressure: r[3][3],
                    angular_velocity: [r[4][0], r[4][1], r[4][2]],
                    peak_deviator: r[4][3],
                    fragment_half: [r[5][0], r[5][1], r[5][2]],
                    volume: r[5][3],
                }
            })
            .collect()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CrushParams {
    enabled: u32,
    mode: u32,
    fragments: u32,
    pool_size: u32,
    max_events: u32,
    t_bonds: u32,
    t_adj: u32,
    t_list: u32,
    t_pool: u32,
    r_bill: u32,
    r_shot: u32,
    r_events: u32,
    r_dist: u32,
    park_y: f32,
    park_step: f32,
    pad0: f32,
}

/// The crush tables and state on the GPU (`fracture.slang` `crush_tab`, `crush_rw`).
fn crush_buffers(gpu: &Gpu, t: &FractureTables) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, CrushParams), String> {
    let (n, e, b) = (t.chunk_count() as usize, t.bond_count() as usize, t.body_count() as usize);
    let Some(c) = &t.crush else {
        let none = CrushParams::default();
        return Ok((storage_bytes(gpu, "fracture crush tables (none)", 16, wgpu::BufferUsages::empty()), storage_bytes(gpu, "fracture crush state (none)", 16, wgpu::BufferUsages::empty()), gpu.uniform("fracture crush params", &none), none));
    };
    if c.chunks.len() != n || c.bonds.len() != e {
        return Err(format!("fracture crush: {} chunk and {} bond records for {n} chunks and {e} bonds", c.chunks.len(), c.bonds.len()));
    }
    if t.geometry.len() != n {
        return Err("fracture crush: needs the chunks' box geometry (FractureTables::geometry)".into());
    }
    if c.conserve && !(c.fragments.is_power_of_two() && c.fragments <= 64) {
        return Err(format!("fracture crush: {} fragments per chunk; a power of two up to 64", c.fragments));
    }
    if c.max_events == 0 {
        return Err("fracture crush: max_events must be at least 1".into());
    }
    // Each chunk's bonds (adjacency), 4 indices a float4.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, bond) in t.bonds.iter().enumerate() {
        adj[bond[0] as usize].push(i as u32);
        adj[bond[1] as usize].push(i as u32);
    }
    let mut tab: Vec<[f32; 4]> = c.chunks.iter().flatten().copied().collect();
    let t_bonds = tab.len() as u32;
    tab.extend_from_slice(&c.bonds);
    let t_adj = tab.len() as u32;
    let mut list = Vec::new();
    for a in &adj {
        tab.push([f32::from_bits(list.len() as u32), f32::from_bits(a.len() as u32), 0.0, 0.0]);
        list.extend_from_slice(a);
    }
    let t_list = tab.len() as u32;
    for q in list.chunks(4) {
        let mut w = [0.0f32; 4];
        for (i, x) in q.iter().enumerate() {
            w[i] = f32::from_bits(*x);
        }
        tab.push(w);
    }
    let t_pool = tab.len() as u32;
    let pool: &[[u32; 2]] = if c.conserve { &c.pool } else { &[] };
    for p in pool {
        tab.push([f32::from_bits(p[0]), f32::from_bits(p[1]), 0.0, 0.0]);
    }
    let r_bill = n as u32;
    let r_shot = r_bill + 2 * b as u32;
    let r_dist = r_shot + MAX_SHOTS as u32;
    let r_events = r_dist + n as u32;
    let rw_words = r_events as u64 + 6 * c.max_events as u64;
    let params = CrushParams {
        enabled: 1,
        mode: c.conserve as u32,
        fragments: if c.conserve { c.fragments } else { 0 },
        pool_size: pool.len() as u32,
        max_events: c.max_events,
        t_bonds,
        t_adj,
        t_list,
        t_pool,
        r_bill,
        r_shot,
        r_events,
        r_dist,
        park_y: c.park_y,
        park_step: c.park_step,
        ..Default::default()
    };
    Ok((gpu.storage("fracture crush tables", &tab), storage_bytes(gpu, "fracture crush state", 16 * rw_words, wgpu::BufferUsages::empty()), gpu.uniform("fracture crush params", &params), params))
}

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
}

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
    pub fn work_words(&self) -> u32 {
        self.w_pulses() + 4 * self.n
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
    /// Per body slot 2 words: crush's strike entry (`fracture.slang` `a_crush`).
    pub fn a_crush(&self) -> u32 {
        self.a_strike() + 9 * self.b
    }
    pub fn atom_words(&self) -> u32 {
        self.a_crush() + 2 * self.b
    }
}

const SUMS: u32 = 20;
/// The shots strike prediction checks, at most.
pub const MAX_SHOTS: usize = 64;
const BODY_WORDS: u32 = 7;
const ARGS: u64 = 9;
const ARG_CHUNKS: u64 = 0;
const ARG_BONDS: u64 = 1;
const ARG_BODIES: u64 = 2;
const ARG_SCAN_CHUNKS: u64 = 3;
const ARG_SCAN_BODIES: u64 = 4;
const ARG_ONE: u64 = 5;
const ARG_CRUSH_CHUNKS: u64 = 6;
const ARG_CRUSH_SCAN: u64 = 7;
const ARG_CRUSH_ONE: u64 = 8;

/// The counters (`fracture.slang` `C_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub broken: u32,
    pub pieces: u32,
    /// Bits: 1 no free slot, 2 a mass sum out of range, 4 a union did not settle, 8 the
    /// impact pulse pool overflowed (`pool_cap`).
    pub error: u32,
    pub split_frames: u32,
}

const BINDINGS: [Binding; 26] = [
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
    Binding::Storage,   // crush tables
    Binding::StorageRw, // crush state, bills, events
    Binding::Uniform,   // crush params
];

const ENTRIES: [&str; 40] = [
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
    "fr_crush_loads",
    "fr_crush_entry",
    "fr_crush_nearest",
    "fr_strike_crush",
    "fr_crush_bill",
    "fr_crush_push",
    "fr_crush_bonds",
    "fr_crush_mark",
    "fr_crush_gate",
    "fr_crush_apply",
    "fr_crush_end",
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
    /// Chunk crushing's tables, state and params (placeholders without it).
    pub crush_tab: wgpu::Buffer,
    pub crush_rw: wgpu::Buffer,
    crush_buf: wgpu::Buffer,
    crush: CrushParams,
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
        let loads = storage_bytes(gpu, "fracture loads", 16 * (n as u64 + 2 * 3 * t.pool_cap as u64), wgpu::BufferUsages::empty());
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
        let (crush_tab, crush_rw, crush_buf, crush) = crush_buffers(gpu, t)?;
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
            crush_tab,
            crush_rw,
            crush_buf,
            crush,
            binds: Vec::new(),
            scan_binds: Vec::new(),
            placeholder: std::array::from_fn(|_| storage_bytes(gpu, "fracture view placeholder", 16, wgpu::BufferUsages::empty())),
            placeholder_uniform: gpu.uniform("fracture view placeholder", &ViewParams::default()),
            view: None,
            engine_bodies: None,
            engine_colliders: None,
            ingest_buffers: None,
        };
        f.rebind(gpu, engine, ingest);
        Ok(f)
    }

    /// The bind groups, against the engine's and the ingest's buffers as they are now.
    pub fn rebind(&mut self, gpu: &Gpu, engine: &EngineBuffers, ingest: &EngineIngest) {
        let group = |scan: &wgpu::Buffer, k: &Kernel| {
            let buffers: [&wgpu::Buffer; 26] = [
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
                &self.crush_tab,
                &self.crush_rw,
                &self.crush_buf,
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
                let buffers: [&wgpu::Buffer; 26] = [
                    &self.params_buf, &self.chunks, &self.bonds, &self.parts, &self.work, &self.atom, &self.frames, bodies, colliders, &ing[0], &ing[1], &ing[2], &self.loads, &self.sizes, &self.scan_new, &self.info, &v.params, &v.chunks, objects, lines, &v.shots, &self.geometry, &self.shots, &self.crush_tab, &self.crush_rw, &self.crush_buf,
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
        let crush = self.crush.enabled != 0;
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.direct(&mut pass, "fr_loads", self.layout.n as usize);
            if crush {
                // Before the breaks: the chunk's bonds as the frame's loads met them.
                self.direct(&mut pass, "fr_crush_loads", self.layout.n as usize);
            }
            self.direct(&mut pass, "fr_breaks", self.layout.e as usize);
            if self.params.shot_count > 0 {
                self.direct(&mut pass, "fr_strike_reset", self.layout.b as usize);
                self.direct(&mut pass, "fr_strike_bounds", self.layout.n as usize);
                self.direct(&mut pass, "fr_strike_find", self.layout.b as usize);
                self.direct(&mut pass, "fr_strike_entry", self.layout.n as usize);
                self.direct(&mut pass, "fr_strike_nearest", self.layout.n as usize);
                if crush {
                    // Before fr_strike_force, which drops a strike it finds no entry for.
                    self.direct(&mut pass, "fr_crush_entry", self.layout.n as usize);
                    self.direct(&mut pass, "fr_crush_nearest", self.layout.n as usize);
                    self.direct(&mut pass, "fr_strike_crush", self.layout.b as usize);
                }
                self.direct(&mut pass, "fr_strike_force", self.layout.b as usize);
                self.direct(&mut pass, "fr_strike_breaks", self.layout.e as usize);
                if crush {
                    self.direct(&mut pass, "fr_crush_bill", self.params.shot_count as usize);
                    self.direct(&mut pass, "fr_crush_push", self.layout.b as usize);
                }
            }
            if crush {
                self.direct(&mut pass, "fr_crush_bonds", self.layout.e as usize);
            }
            self.direct(&mut pass, "fr_gate", 1);
        }
        self.record_split(encoder);
        if crush {
            self.record_crush(encoder);
        }
    }

    /// After the split: each chunk crushed this frame ranked (chunk order), its event and
    /// fragments written and its body parked (gated: nothing on a frame with no crush).
    fn record_crush(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.direct(&mut pass, "fr_crush_mark", self.layout.n as usize);
            self.direct(&mut pass, "fr_crush_gate", 1);
        }
        encoder.copy_buffer_to_buffer(&self.sizes, 0, &self.args, 0, 12 * ARGS);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        let pass = &mut pass;
        self.indirect(pass, "fr_scan_blocks", ARG_CRUSH_SCAN, None);
        self.indirect(pass, "fr_scan_top", ARG_CRUSH_ONE, None);
        self.indirect(pass, "fr_scan_add", ARG_CRUSH_CHUNKS, None);
        self.indirect(pass, "fr_crush_apply", ARG_CRUSH_CHUNKS, None);
        self.indirect(pass, "fr_crush_end", ARG_CRUSH_ONE, None);
    }

    /// Whether the fracture crushes chunks (`FractureTables::crush`).
    pub fn crushes(&self) -> bool {
        self.crush.enabled != 0
    }

    /// Where this frame's crush events are in `crush_rw` (bytes: offset, size), for a
    /// readback; their count is the counters' word [`C_CRUSH_EVENTS`].
    pub fn crush_events_range(&self) -> (u64, u64) {
        (16 * self.crush.r_events as u64, 16 * 6 * self.crush.max_events as u64)
    }

    /// Diagnostic read: each chunk's crush state (damage, peak p, peak q, flags as bits:
    /// 1 crushed, cause << 4).
    pub fn crush_state(&self, gpu: &Gpu) -> Vec<[f32; 4]> {
        if !self.crushes() {
            return Vec::new();
        }
        gpu.read_range(&self.crush_rw, 0, self.layout.n as usize)
    }

    /// Diagnostic read: this frame's crush events.
    pub fn crush_events(&self, gpu: &Gpu) -> Vec<CrushEvent> {
        if !self.crushes() {
            return Vec::new();
        }
        let c: Vec<u32> = gpu.read_range(&self.atom, self.layout.a_counters() as usize, 16);
        let words: Vec<[f32; 4]> = gpu.read_range(&self.crush_rw, self.crush.r_events as usize, 6 * self.crush.max_events as usize);
        CrushEvent::decode(&words, c[C_CRUSH_EVENTS].min(self.crush.max_events))
    }

    /// Diagnostic read: chunks crushed and fragment pool entries taken, in all.
    pub fn crush_counts(&self, gpu: &Gpu) -> (u32, u32) {
        let c: Vec<u32> = gpu.read_range(&self.atom, self.layout.a_counters() as usize, 16);
        (c[9], c[8])
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
        let words: Vec<[f32; 4]> = shots.iter().flat_map(|s| [[f32::from_bits(s.body), s.bound, s.face, s.mass], [s.modulus, 0.0, 0.0, 0.0]]).collect();
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
        Counters { broken: c[1], pieces: c[2], error: c[3], split_frames: c[4] }
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

    /// Break `bonds` (and mark their chunks' bodies dirty) touching only their own words, so
    /// a step still in flight keeps every other bond's state: a host pass's breaks between
    /// frames, split by the next frame's gate. `chunk_body` is each chunk's body as last read.
    pub fn break_bonds_sparse(&self, gpu: &Gpu, bonds: &[u32], chunk_body: &[u32], bond_table: &[[u32; 4]]) {
        if bonds.is_empty() {
            return;
        }
        let (zero, one) = (0u32, 1u32);
        for &i in bonds {
            gpu.queue.write_buffer(&self.work, 4 * (self.layout.w_alive() + i) as u64, bytemuck::bytes_of(&zero));
            for end in [bond_table[i as usize][0], bond_table[i as usize][1]] {
                let body = chunk_body[end as usize];
                gpu.queue.write_buffer(&self.atom, 4 * (self.layout.a_dirty() + body) as u64, bytemuck::bytes_of(&one));
            }
        }
        // The frame's broken count (`C_FRAME_BROKEN`, cleared by every frame's gate).
        gpu.queue.write_buffer(&self.atom, 4 * self.layout.a_counters() as u64, bytemuck::bytes_of(&(bonds.len() as u32)));
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
