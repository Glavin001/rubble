//! The World side of the explicit vehicles' coupling (`slang/kernels/vehicle_coupling.slang`)
//! as backend-neutral data: its parameter block, the layout of its work buffer, its
//! bindings, the window clock and the ordered commands of a World step. Each host
//! (engine-world on wgpu, engine-cuda's `CudaWorld`) allocates the buffers, binds them
//! and replays the commands; none holds the order itself.
//!
//! Per World step:
//! 1. [`emit`], before the rigid step: the last window's impulses become impulse records
//!    (the host then tells the rigid step to apply them: `rigid_host::GpuImpulses`);
//! 2. the rigid step;
//! 3. [`frame`], after it: the clears, the vehicles' bounds, the candidates from the
//!    World's resting table when the step kept it, the proxy plan, the proxied list's
//!    count copied to the indirect arguments, the share counts, [`Cmd::Frame`] (the
//!    explicit frame over the window), the chunk loads, the asleep bits.

use super::super::CouplingLayout;
use engine_softbody::explicit_beams::ExplicitBeams;

/// The explicit solver's step (s).
pub const STEP: f64 = engine_softbody::explicit_beams::STEP as f64;
/// A vehicle's broadphase proxy: its swept box grown by this (m) when it leaves the
/// last one (so a vehicle at rest or cruising refattens rarely).
pub const PROXY_SLACK: f32 = 0.5;
/// Proxy flags (the vehicle table's 4th word): out of the World broadphase.
pub const PROXY_OUT: u32 = 2;
/// Impulse record keys (after the host's, the terrain's 0x300000 + collider).
pub const KEY_BASE: u32 = 0x50_0000;
/// Windows between status reads of the hosts that read the status word on their own.
pub const STATUS_EVERY: u64 = 8;
/// The status word's bits (work[1]), in order.
pub const ST_NAMES: [&str; 7] = [
    "more colliders than the shapes room",
    "more bodies than the bodies room",
    "more hull planes than the planes room",
    "more impulse records than IMPULSE_CAP",
    "more chunk records than the threshold ingest's rows",
    "a coupling loop past its cap or pass budget (a corrupt count; the frame is invalid)",
    "more broadphase candidates than their room (three times the shapes room: a collider can be listed by its resting cell and as moving or moved)",
];

/// The status bits as an error (none: Ok): a room exceeded, or a loop held to its cap or
/// past its pass budget.
pub fn status_error(bits: u32, cap: super::super::CouplingCapacity) -> Result<(), String> {
    if bits == 0 {
        return Ok(());
    }
    let names: Vec<&str> = (0..ST_NAMES.len())
        .filter(|k| bits & (1 << k) != 0)
        .map(|k| ST_NAMES[k])
        .collect();
    Err(format!(
        "explicit vehicles: {} (capacity {:?}; raise ExplicitVehiclesDef::capacity)",
        names.join(", "),
        cap
    ))
}

/// The farthest any raycast wheel's ray reaches below its hub (m): its travel plus radius
/// less the hardpoint's height. The proxy plan's margin is at least this while wheels hit
/// bodies, so a body under a wheel is a proxy whenever the ray can reach it.
pub fn wheel_reach(vehicles: &[ExplicitBeams]) -> f32 {
    vehicles
        .iter()
        .filter_map(|v| v.drive.as_deref())
        .flat_map(|d| d.config.wheels.iter())
        .map(|w| (w.susp_max + w.radius - w.rest_len).max(0.0))
        .fold(0.0, f32::max)
}

/// The module's entries, in the order every host keeps their kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    Bounds,
    Plan,
    Share,
    Loads,
    LoadsEnd,
    Emit,
    EmitEnd,
    SleepMask,
    Query,
}

impl Entry {
    pub const ALL: [Entry; 9] = [
        Entry::Bounds,
        Entry::Plan,
        Entry::Share,
        Entry::Loads,
        Entry::LoadsEnd,
        Entry::Emit,
        Entry::EmitEnd,
        Entry::SleepMask,
        Entry::Query,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Entry::Bounds => "cpl_bounds",
            Entry::Plan => "cpl_plan",
            Entry::Share => "cpl_share",
            Entry::Loads => "cpl_loads",
            Entry::LoadsEnd => "cpl_loads_end",
            Entry::Emit => "cpl_emit",
            Entry::EmitEnd => "cpl_emit_end",
            Entry::SleepMask => "cpl_sleep_mask",
            Entry::Query => "cpl_query",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    /// The bindings its dispatch binds: `cpl_query` has its own set (with the World
    /// broadphase's buffers every pass would bind 14 storage buffers, past WebGPU's
    /// per-stage limit).
    pub fn set(self) -> &'static [(u32, Access)] {
        if self == Entry::Query {
            QUERY_SET
        } else {
            MAIN_SET
        }
    }
}

/// How a pass binds a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Uniform,
    Read,
    Write,
}

/// What a binding of the module holds (its `[[vk::binding(n, 0)]]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// [`Params`] (uniform).
    Params,
    /// The rigid plan's bodies (binding 0), colliders (2), hull arena (18), scratch
    /// atomics (21: the impulse records), constraint ids (14: the resting table) and
    /// collider bounds (9).
    RigidBodies,
    RigidColliders,
    RigidHulls,
    RigidScratch,
    RestIds,
    RestAabbs,
    /// `pose_pack`'s header and moved list.
    MovedHeader,
    Moved,
    /// The explicit plan's nodes and gMisc.
    ExplicitNodes,
    ExplicitMisc,
    /// The coupling's own work buffer ([`WorkLayout`]).
    Work,
    /// The threshold ingest's collider table, records and row counts (stand-ins of
    /// [`INGEST_DUMMY_BYTES`] zero bytes without a fracture).
    IngestInfo,
    IngestRecords,
    IngestRows,
}

/// The buffer behind binding `binding`.
pub fn source(binding: u32) -> Source {
    match binding {
        0 => Source::Params,
        1 => Source::RigidBodies,
        2 => Source::RigidColliders,
        3 => Source::RigidHulls,
        4 => Source::ExplicitNodes,
        5 => Source::ExplicitMisc,
        8 => Source::RigidScratch,
        9 => Source::Work,
        10 => Source::IngestInfo,
        11 => Source::IngestRecords,
        12 => Source::IngestRows,
        13 => Source::RestIds,
        14 => Source::RestAabbs,
        15 => Source::MovedHeader,
        16 => Source::Moved,
        b => panic!("vehicle_coupling.slang has no binding {b}"),
    }
}

/// Every entry's bindings but `cpl_query`'s.
pub const MAIN_SET: &[(u32, Access)] = &[
    (0, Access::Uniform),
    (1, Access::Read),
    (2, Access::Read),
    (3, Access::Read),
    (4, Access::Read),
    (5, Access::Write),
    (8, Access::Write),
    (9, Access::Write),
    (10, Access::Read),
    (11, Access::Write),
    (12, Access::Write),
];

/// `cpl_query`'s bindings (gMisc: the vehicles' group boxes, pieceAccept).
pub const QUERY_SET: &[(u32, Access)] = &[
    (0, Access::Uniform),
    (1, Access::Read),
    (2, Access::Read),
    (5, Access::Write),
    (8, Access::Write),
    (9, Access::Write),
    (13, Access::Read),
    (14, Access::Read),
    (15, Access::Read),
    (16, Access::Read),
];

/// Bytes of each stand-in for the threshold ingest's buffers.
pub const INGEST_DUMMY_BYTES: u64 = 160;
/// The proxied list's indirect arguments as created: (count, 1, 1, 0).
pub const LIST_ARGS_INIT: [u32; 4] = [0, 1, 1, 0];

/// The module's `Params` (uniform, binding 0), field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub vehicles: u32,
    pub colliders: u32,
    pub max_bodies: u32,
    pub couple_rec: u32,
    pub b: u32,
    pub s: u32,
    pub planes: u32,
    pub recs: u32,
    pub misc_base: u32,
    pub window: f32,
    pub margin: f32,
    pub key_base: u32,
    pub impulse_base: u32,
    pub impulse_count: u32,
    pub impulse_first: u32,
    pub impulse_cap: u32,
    pub ingest_rows: u32,
    pub ingest_slot: u32,
    pub step: u32,
    pub filter: u32,
    pub row_cap: u32,
    /// Tests: cpl_plan's serial walk (1) instead of its lane-parallel records.
    pub serial_plan: u32,
    /// Vehicle sleep: gMisc record of vehicle 0's sleep record (0: off).
    pub sleep_base: u32,
    /// work's word of the proxied list (0: every vehicle).
    pub list_base: u32,
    /// Hull proxies carry their vertices and edges (surface contact).
    pub hull_geom: u32,
    /// 0: cpl_plan scans every collider; 1: cpl_query's candidates from the World
    /// broadphase's resting table.
    pub mode: u32,
    pub rest_base: u32,
    pub hash_size: u32,
    pub aabb_base: u32,
    pub levels: u32,
    pub cell: f32,
    pub large_static_base: u32,
    pub slack: f32,
    pub cand_cap: u32,
    /// work's word of the broadphase region (candidate counts, proxies, candidates).
    pub extra_base: u32,
    /// Passes one invocation of the plan passes may take (explicit_loop_cap.slang; see
    /// [`loop_budget`]).
    pub loop_budget: u32,
    /// gMisc record of the explicit nodes' retired bits (0: none retired).
    pub retire_base: u32,
    /// The coupling's window per piece: gMisc records of the nodes' labels and of the
    /// pieces' boxes (0: one window per vehicle).
    pub label_base: u32,
    pub piece_box_base: u32,
    /// To the uniform's 16-byte size.
    pub pad: u32,
}

/// The coupling passes' pass budget: four times a bound on one invocation's passes
/// (every collider; every resting-table head on each of 33 levels; shapes against
/// shapes, planes and bodies; candidates against candidates; the vehicles), held to
/// 2^22..2^31. A sound state never reaches it; a corrupt count stops there and sets
/// ST_LOOP. `fixed`: a test's budget instead (`explicit_plan::loop_budget_override`).
pub fn loop_budget(p: &Params, fixed: Option<u32>) -> u32 {
    let [c, h, s, pl, b, k, v] = [
        p.colliders,
        p.hash_size,
        p.s,
        p.planes,
        p.b,
        p.cand_cap,
        p.vehicles,
    ]
    .map(u64::from);
    let unit = c + 33 * h + s * (s + pl + b) + k * k + 64 * v + 4096;
    fixed.unwrap_or_else(|| (4 * unit).clamp(1 << 22, 1 << 31) as u32)
}

/// Where the last rigid step left the awake-indexed broadphase's resting table (the
/// rigid host's rest view), for `cpl_query`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RestTable {
    pub rest_base: u32,
    pub hash_size: u32,
    pub aabb_base: u32,
    pub levels: u32,
    pub cell: f32,
}

/// What a window's [`Params`] are made from.
#[derive(Clone, Copy, Debug)]
pub struct ParamsInput<'a> {
    pub layout: &'a CouplingLayout,
    pub work: &'a WorkLayout,
    /// The World's colliders now.
    pub colliders: u32,
    /// The window's explicit steps.
    pub steps: u32,
    /// The proxy plan's extra reach (m).
    pub margin: f32,
    /// Windows run before this one.
    pub windows: u64,
    /// The vehicles' collision filter.
    pub categories: u16,
    pub mask: u16,
    /// The threshold ingest's rows per slot (0: none).
    pub ingest_rows: u32,
    pub serial_plan: bool,
    /// gMisc record of vehicle 0's sleep record while vehicle sleep is on (else 0).
    pub sleep_base: u32,
    /// Some vehicle has surface contact.
    pub hull_geom: bool,
    /// The resting table the rigid step kept (the plan by `cpl_query`), or none (the scan).
    pub rest: Option<RestTable>,
    pub retire_base: u32,
    /// The piece windows' label and box records ((0, 0): one window per vehicle).
    pub piece_records: (u32, u32),
    pub loop_budget: Option<u32>,
}

/// A window's parameter block.
pub fn params(i: &ParamsInput) -> Params {
    let cap = i.layout.cap;
    let rest = i.rest;
    let p = Params {
        vehicles: i.work.vehicles,
        colliders: i.colliders,
        max_bodies: i.work.max_bodies,
        couple_rec: 0,
        b: cap.bodies,
        s: cap.shapes,
        planes: cap.planes,
        recs: i.layout.recs(),
        misc_base: i.layout.misc_base,
        window: (f64::from(i.steps) * STEP) as f32,
        margin: i.margin,
        key_base: KEY_BASE,
        impulse_base: engine_schema::IMPULSE_BASE,
        impulse_count: engine_schema::IMPULSE_COUNT,
        impulse_first: engine_schema::IMPULSE_FIRST,
        impulse_cap: engine_schema::IMPULSE_CAP,
        ingest_rows: i.ingest_rows,
        ingest_slot: 1,
        step: i.windows as u32,
        filter: u32::from(i.categories) | u32::from(i.mask) << 16,
        row_cap: i.work.ingest_rows(cap.shapes).min(i.ingest_rows),
        serial_plan: u32::from(i.serial_plan),
        sleep_base: i.sleep_base,
        list_base: i.work.list_word as u32,
        hull_geom: u32::from(i.hull_geom),
        mode: u32::from(rest.is_some()),
        rest_base: rest.map_or(0, |r| r.rest_base),
        hash_size: rest.map_or(1, |r| r.hash_size),
        aabb_base: rest.map_or(0, |r| r.aabb_base),
        levels: rest.map_or(0, |r| r.levels),
        cell: rest.map_or(1.0, |r| r.cell),
        large_static_base: engine_schema::LARGE_STATIC_BASE,
        slack: PROXY_SLACK,
        cand_cap: i.work.cand_cap,
        extra_base: i.work.extra_word as u32,
        loop_budget: 0,
        retire_base: i.retire_base,
        label_base: i.piece_records.0,
        piece_box_base: i.piece_records.1,
        pad: 0,
    };
    Params {
        loop_budget: loop_budget(&p, i.loop_budget),
        ..p
    }
}

/// The work buffer (binding 9), in words: [0] chunk rows used, [1] status,
/// [2..2+max_bodies) share counts (all cleared each window); per vehicle 8 words of
/// bounds (cpl_bounds rewrites an awake vehicle's, a sleeping one's stand); per vehicle 4
/// words written once by the host (node offset, nodes, mass bits, proxy flags); the asleep
/// bits, a word per 32 vehicles; the proxied list (count, 1, 1, 0, the vehicles; its count
/// cleared each window); then the broadphase region: per vehicle its candidate count, its
/// proxy (8 words), its candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkLayout {
    pub vehicles: u32,
    pub max_bodies: u32,
    /// Bytes cleared each window, from 0.
    pub cleared: u64,
    /// Byte offset of the vehicle table.
    pub table: u64,
    pub sleep_mask_word: u64,
    pub mask_words: u64,
    pub list_word: u64,
    pub extra_word: u64,
    /// Candidates per vehicle: three times the shapes room.
    pub cand_cap: u32,
}

impl WorkLayout {
    pub fn new(vehicles: u32, max_bodies: u32, shapes: u32) -> Self {
        let nv = u64::from(vehicles);
        let cleared = 4 * (2 + u64::from(max_bodies));
        let table = cleared + 32 * nv;
        let mask_words = nv.div_ceil(32);
        let sleep_mask_word = table / 4 + 4 * nv;
        let list_word = (sleep_mask_word + mask_words).next_multiple_of(4);
        let extra_word = list_word + 4 + nv;
        Self {
            vehicles,
            max_bodies,
            cleared,
            table,
            sleep_mask_word,
            mask_words,
            list_word,
            extra_word,
            cand_cap: 3 * shapes,
        }
    }

    /// The buffer's bytes.
    pub fn bytes(&self) -> u64 {
        let nv = u64::from(self.vehicles);
        4 * (self.extra_word + 9 * nv + u64::from(self.cand_cap) * nv)
    }

    /// Byte offset of the status word.
    pub const STATUS_OFFSET: u64 = 4;

    /// Byte offset of vehicle `v`'s table row (its 4th word: the proxy flags).
    pub fn vehicle(&self, v: u32) -> u64 {
        self.table + 16 * u64::from(v)
    }

    /// Byte offset of vehicle `v`'s broadphase proxy (8 words; the last: refattens).
    pub fn proxy(&self, v: u32) -> u64 {
        4 * (self.extra_word + u64::from(self.vehicles) + 8 * u64::from(v))
    }

    /// Byte offset of vehicle 0's candidate count (a word per vehicle).
    pub fn cand_counts(&self) -> u64 {
        4 * self.extra_word
    }

    /// Chunk records a window can make at most: a record per proxy shape.
    pub fn ingest_rows(&self, shapes: u32) -> u32 {
        self.vehicles * shapes
    }

    /// `cpl_sleep_mask`'s workgroups: a thread per 32 vehicles, 64 a group.
    pub fn sleep_mask_groups(&self) -> u32 {
        self.vehicles.div_ceil(32).div_ceil(64)
    }
}

/// A vehicle's row of the table at [`WorkLayout::table`]: its first node in the explicit
/// buffers, its nodes, its mass, its proxy flags (none).
pub fn vehicle_row(node_offset: u32, vehicle: &ExplicitBeams) -> [u32; 4] {
    let mass: f32 = vehicle.nodes.iter().map(|n| n.mass).sum();
    [node_offset, vehicle.nodes.len() as u32, mass.to_bits(), 0]
}

/// A buffer of the coupling's own a command names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buf {
    Work,
    /// The proxied list's indirect arguments (16 bytes, [`LIST_ARGS_INIT`]): a buffer
    /// bound for storage cannot also be a dispatch's indirect arguments on WebGPU.
    ListArgs,
    /// A host's own status readback (16 bytes).
    Status,
}

/// One command of the coupling's part of a World step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    /// Zero `size` bytes of the work buffer from `offset`.
    Clear { offset: u64, size: u64 },
    Copy {
        src: Buf,
        src_offset: u64,
        dst: Buf,
        dst_offset: u64,
        size: u64,
    },
    /// The dispatches up to the next command that is not one run together (a compute pass
    /// on WebGPU), timed as `segment` when the host times the step (`before_step`: among
    /// the segments ahead of the rigid step's).
    Pass {
        label: &'static str,
        segment: Option<&'static str>,
        before_step: bool,
    },
    /// `entry` over `groups` workgroups.
    Dispatch { entry: Entry, groups: u32 },
    /// `entry` over the workgroups in the proxied list's indirect arguments
    /// ([`Buf::ListArgs`], offset 0).
    Indirect { entry: Entry },
    /// A timing mark between passes.
    Mark(&'static str),
    /// The explicit frame over the window goes here (the host first asks its planner
    /// for a contact replan: the rigid bodies push the vehicles).
    Frame,
}

/// Step 1, before the rigid step: the last window's impulses as impulse records.
pub fn emit() -> Vec<Cmd> {
    vec![
        Cmd::Pass {
            label: "vehicle coupling emit",
            segment: Some("cpl_emit"),
            before_step: true,
        },
        Cmd::Indirect { entry: Entry::Emit },
        Cmd::Dispatch {
            entry: Entry::EmitEnd,
            groups: 1,
        },
    ]
}

/// The clears that open a window's plan.
fn clears(work: &WorkLayout) -> [Cmd; 2] {
    [
        Cmd::Clear {
            offset: 0,
            size: work.cleared,
        },
        Cmd::Clear {
            offset: 4 * work.list_word,
            size: 4,
        },
    ]
}

/// What a window's commands depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOpts {
    /// The rigid step kept its resting table: the candidates by `cpl_query`.
    pub query: bool,
    /// Vehicle sleep is on: the asleep bits after the frame.
    pub sleep: bool,
}

/// Step 3, after the rigid step: the next window's plan, the explicit frame and the
/// chunk loads.
pub fn frame(work: &WorkLayout, o: FrameOpts) -> Vec<Cmd> {
    let nv = work.vehicles;
    let mut c = clears(work).to_vec();
    c.push(Cmd::Pass {
        label: "vehicle coupling plan",
        segment: Some("cpl_plan"),
        before_step: false,
    });
    c.push(Cmd::Dispatch {
        entry: Entry::Bounds,
        groups: nv,
    });
    if o.query {
        c.push(Cmd::Dispatch {
            entry: Entry::Query,
            groups: nv,
        });
    }
    c.push(Cmd::Dispatch {
        entry: Entry::Plan,
        groups: nv,
    });
    c.push(Cmd::Copy {
        src: Buf::Work,
        src_offset: 4 * work.list_word,
        dst: Buf::ListArgs,
        dst_offset: 0,
        size: 4,
    });
    // Only the vehicles with proxies (cpl_plan's list).
    c.push(Cmd::Pass {
        label: "vehicle coupling share",
        segment: Some("cpl_share"),
        before_step: false,
    });
    c.push(Cmd::Indirect {
        entry: Entry::Share,
    });
    c.push(Cmd::Mark("vehicle coupling plan"));
    c.push(Cmd::Frame);
    c.push(Cmd::Mark("explicit frame"));
    c.push(Cmd::Pass {
        label: "vehicle coupling loads",
        segment: Some("cpl_loads"),
        before_step: false,
    });
    c.push(Cmd::Indirect {
        entry: Entry::Loads,
    });
    c.push(Cmd::Dispatch {
        entry: Entry::LoadsEnd,
        groups: 1,
    });
    if o.sleep {
        c.push(Cmd::Dispatch {
            entry: Entry::SleepMask,
            groups: work.sleep_mask_groups(),
        });
    }
    c
}

/// After a window, for a host that reads the status on its own every [`STATUS_EVERY`]
/// windows ([`Clock::window_done`]): the first words of work to [`Buf::Status`].
pub fn status_read() -> Cmd {
    Cmd::Copy {
        src: Buf::Work,
        src_offset: 0,
        dst: Buf::Status,
        dst_offset: 0,
        size: 16,
    }
}

/// Tests and measurement: the plan alone from the current state (the clears, the bounds,
/// the candidates when `query`, the plan, the share counts of every vehicle), untimed.
pub fn plan_alone(work: &WorkLayout, query: bool, label: &'static str) -> Vec<Cmd> {
    let nv = work.vehicles;
    let mut c = clears(work).to_vec();
    c.push(Cmd::Pass {
        label,
        segment: None,
        before_step: false,
    });
    c.push(Cmd::Dispatch {
        entry: Entry::Bounds,
        groups: nv,
    });
    if query {
        c.push(Cmd::Dispatch {
            entry: Entry::Query,
            groups: nv,
        });
    }
    c.push(Cmd::Dispatch {
        entry: Entry::Plan,
        groups: nv,
    });
    c.push(Cmd::Dispatch {
        entry: Entry::Share,
        groups: nv,
    });
    c
}

/// Tests: `cpl_emit` alone over the proxied list (no `cpl_emit_end`: the caller puts the
/// impulse list back).
pub fn emit_alone() -> Vec<Cmd> {
    vec![
        Cmd::Pass {
            label: "vehicle coupling emit",
            segment: None,
            before_step: false,
        },
        Cmd::Indirect { entry: Entry::Emit },
    ]
}

/// The coupling's clock and what one window owes the next: the explicit steps owed to
/// World time, and whether the last frame's impulses wait to be emitted.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Clock {
    /// Explicit steps owed to World time (the remainder carried).
    pub carry: f64,
    /// A frame ran: its impulses are emitted at the next step's start.
    pub emit_pending: bool,
    /// That frame was encoded while some vehicle could be awake (vehicle sleep): its emit
    /// wakes the rigid step. Otherwise every vehicle was asleep with nothing written to
    /// one, so it stepped only if the plan woke it beside a moving body (then the rigid
    /// step runs anyway): the emit schedules its records without a wake.
    pub emit_wakes: bool,
    /// Windows (World steps) and explicit steps run.
    pub windows: u64,
    pub steps: u64,
}

impl Default for Clock {
    fn default() -> Self {
        Self {
            carry: 0.0,
            emit_pending: false,
            emit_wakes: true,
            windows: 0,
            steps: 0,
        }
    }
}

impl Clock {
    /// The last window's emit, once: `Some(wakes)` when a frame ran since the last call.
    pub fn take_emit(&mut self) -> Option<bool> {
        std::mem::take(&mut self.emit_pending).then_some(self.emit_wakes)
    }

    /// The explicit steps of a window of `dt` seconds (the remainder carried).
    pub fn window_steps(&mut self, dt: f32) -> u32 {
        self.carry += f64::from(dt) / STEP;
        let steps = (self.carry + 1e-6).floor().max(0.0);
        self.carry -= steps;
        steps as u32
    }

    /// A window of `steps` was encoded; `needs_step`: some vehicle may be awake. Returns
    /// whether a host that reads the status on its own reads it with this window.
    pub fn window_done(&mut self, steps: u32, needs_step: bool) -> bool {
        self.windows += 1;
        self.steps += u64::from(steps);
        self.emit_pending = steps > 0;
        self.emit_wakes = needs_step;
        self.windows % STATUS_EVERY == 1
    }

    /// As after the vehicles are put back to their spawn state (a restart).
    pub fn reset(&mut self) {
        self.carry = 0.0;
        self.emit_pending = false;
        self.emit_wakes = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The work buffer's regions as the kernel reads them (vehicle_coupling.slang
    /// wBounds, vehicleAt, wCandCount, wProxy, wCands) for a fleet that needs two words
    /// of asleep bits.
    #[test]
    fn work_layout_is_the_kernels() {
        let (nv, max_bodies, shapes) = (40u32, 1000u32, 64u32);
        let w = WorkLayout::new(nv, max_bodies, shapes);
        let (nv64, mb) = (u64::from(nv), u64::from(max_bodies));
        assert_eq!(w.cleared, 4 * (2 + mb));
        // wBounds(0), then vehicleAt(0).
        assert_eq!(w.table, 4 * (2 + mb + 8 * nv64));
        assert_eq!(w.vehicle(3), 4 * (2 + mb + 8 * nv64 + 4 * 3));
        assert_eq!(w.sleep_mask_word, 2 + mb + 12 * nv64);
        assert_eq!(w.mask_words, 2);
        assert_eq!(w.list_word % 4, 0);
        assert!(w.list_word >= w.sleep_mask_word + w.mask_words);
        assert_eq!(w.extra_word, w.list_word + 4 + nv64);
        assert_eq!(w.cand_counts(), 4 * w.extra_word);
        assert_eq!(w.proxy(2), 4 * (w.extra_word + nv64 + 16));
        assert_eq!(w.bytes(), 4 * (w.extra_word + 9 * nv64 + 3 * 64 * nv64));
        assert_eq!(w.sleep_mask_groups(), 1);
    }

    /// The parameter block is the uniform's size, and the clock gives the World's 33,
    /// 33, 34 steps at 60 Hz.
    #[test]
    fn params_size_and_clock() {
        assert_eq!(std::mem::size_of::<Params>(), 160);
        let mut c = Clock::default();
        let steps: Vec<u32> = (0..6).map(|_| c.window_steps(1.0 / 60.0)).collect();
        assert_eq!(steps.iter().sum::<u32>(), 200);
        assert!(steps.iter().all(|&s| s == 33 || s == 34), "{steps:?}");
        assert_eq!(c.take_emit(), None);
        assert!(c.window_done(33, true));
        assert_eq!(c.take_emit(), Some(true));
        assert_eq!(c.take_emit(), None);
        for _ in 0..7 {
            assert!(!c.window_done(0, false));
        }
        assert!(c.window_done(34, false));
        assert_eq!(c.take_emit(), Some(false));
    }

    /// A window's commands: the order the wgpu host recorded before this plan existed.
    #[test]
    fn frame_order() {
        let w = WorkLayout::new(3, 100, 64);
        let names = |cmds: &[Cmd]| -> Vec<String> {
            cmds.iter()
                .map(|c| match c {
                    Cmd::Clear { .. } => "clear".into(),
                    Cmd::Copy { dst, .. } => format!("copy>{dst:?}"),
                    Cmd::Pass { .. } => "|".into(),
                    Cmd::Dispatch { entry, groups } => format!("{}x{groups}", entry.name()),
                    Cmd::Indirect { entry } => format!("{}*", entry.name()),
                    Cmd::Mark(_) => "mark".into(),
                    Cmd::Frame => "FRAME".into(),
                })
                .collect()
        };
        let mut all = frame(
            &w,
            FrameOpts {
                query: true,
                sleep: true,
            },
        );
        all.push(status_read());
        assert_eq!(
            names(&all).join(" "),
            "clear clear | cpl_boundsx3 cpl_queryx3 cpl_planx3 copy>ListArgs | cpl_share* mark FRAME mark | cpl_loads* \
             cpl_loads_endx1 cpl_sleep_maskx1 copy>Status"
        );
        let least = frame(
            &w,
            FrameOpts {
                query: false,
                sleep: false,
            },
        );
        assert_eq!(
            names(&least).join(" "),
            "clear clear | cpl_boundsx3 cpl_planx3 copy>ListArgs | cpl_share* mark FRAME mark | cpl_loads* cpl_loads_endx1"
        );
        assert_eq!(names(&emit()).join(" "), "| cpl_emit* cpl_emit_endx1");
        assert_eq!(
            names(&plan_alone(&w, true, "x")).join(" "),
            "clear clear | cpl_boundsx3 cpl_queryx3 cpl_planx3 cpl_sharex3"
        );
    }
}
