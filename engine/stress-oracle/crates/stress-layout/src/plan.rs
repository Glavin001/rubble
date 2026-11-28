//! The fracture world's frame as a command list, without a GPU: what stress-gpu's
//! `GpuFracture` dispatches each step, in order, as data. A host (wgpu today; CUDA and
//! native Metal by the same list) allocates the buffers, binds them and replays the
//! commands; none of them decides an order of its own (engine/docs/world-on-cuda.md, C6).
//!
//! The list for a frame does not depend on what breaks in it: every dispatch whose size
//! the frame decides is indirect, sized on the GPU by a gate kernel (`fr_gate`,
//! `fr_quiet`, `fr_cs_gate`, `fr_crush_gate`) that writes [`Buf::Sizes`]; a copy to
//! [`Buf::Args`] then precedes the dispatches that read them (WebGPU does not let one
//! buffer be written as storage and read as indirect arguments in one pass).

/// The shader an entry is in (its file in stress-gpu `shaders/`, and the module name of
/// each backend's build of it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Module {
    Fracture,
    FractureLog,
    /// The contact ingest (`coupling.slang`).
    Coupling,
    /// The stress solver's frame: the fracture's breaks in, its loads, its breaks out.
    BlastFracture,
    BlastTopology,
    BlastMotion,
    /// The solve: an island a group (`blast_native`), or the host-path CGLS (`blast_cgls`).
    BlastNative,
    BlastCgls,
    /// The solve of the large islands, over many groups.
    BlastNativeWide,
    BlastWideGate,
    BlastVirial,
    BlastMaterial,
    BlastView,
}

impl Module {
    pub fn name(self) -> &'static str {
        match self {
            Module::Fracture => "fracture",
            Module::FractureLog => "fracture_log",
            Module::Coupling => "coupling",
            Module::BlastFracture => "blast_fracture",
            Module::BlastTopology => "blast_topology",
            Module::BlastMotion => "blast_motion",
            Module::BlastNative => "blast_native",
            Module::BlastCgls => "blast_cgls",
            Module::BlastNativeWide => "blast_native_wide",
            Module::BlastWideGate => "blast_wide_gate",
            Module::BlastVirial => "blast_virial",
            Module::BlastMaterial => "blast_material",
            Module::BlastView => "blast_view",
        }
    }
}

/// The triples of [`Buf::Sizes`] / [`Buf::Args`] (12 bytes each: an indirect dispatch's
/// workgroup counts), by index.
pub mod arg {
    /// Triples in all.
    pub const COUNT: u64 = 25;
    pub const CHUNKS: u64 = 0;
    pub const BONDS: u64 = 1;
    pub const BODIES: u64 = 2;
    pub const SCAN_CHUNKS: u64 = 3;
    pub const SCAN_BODIES: u64 = 4;
    pub const ONE: u64 = 5;
    pub const CS_BODIES: u64 = 6;
    pub const CS_ROWS: u64 = 7;
    pub const CS_PICK: u64 = 8;
    pub const CS_LIST: u64 = 9;
    pub const CS_CHUNKS: u64 = 10;
    pub const CS_BONDS: u64 = 11;
    pub const CS_SLOTS: u64 = 12;
    pub const CS_SAVE: u64 = 13;
    pub const CRUSH_CHUNKS: u64 = 14;
    pub const CRUSH_SCAN: u64 = 15;
    pub const CRUSH_ONE: u64 = 16;
    /// The quiet gate's: the chunks and bonds of a frame that is not quiet (0 groups when
    /// it is), then the ingest's five from `Q_CHUNKS + 2`.
    pub const Q_CHUNKS: u64 = 17;
    pub const Q_BONDS: u64 = 18;
    /// The quiet gate's state (its streak in the second word).
    pub const Q_STATE: u64 = 24;
    /// Bytes of the triples.
    pub const BYTES: u64 = 12 * COUNT;
}

/// The shots strike prediction checks, at most.
pub const MAX_SHOTS: u32 = 64;

/// A row of `fr_il_run` / `fr_gi_solve` groups (WebGPU's largest dispatch dimension).
pub const RUN_ROW: u32 = 65_535;

/// The in-step contact fracture's timestamps (a diagnostic), and the load path's frame
/// stages' after them.
pub const CS_STAMPS: u32 = 6;
pub const LP_STAMPS: u32 = 6;

/// A buffer a command copies between (the dispatches' own buffers are their module's
/// bindings).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Buf {
    /// The gates' workgroup counts (storage).
    Sizes,
    /// The same, as indirect arguments.
    Args,
    /// The fracture's `work` words.
    Work,
    /// The rigid engine's step counters (engine-schema `StepCounters`).
    EngineCounters,
    /// The rigid engine's bodies (engine-schema `BodyState`).
    EngineBodies,
    /// The ingest's copy of them at the frame's start.
    StartBodies,
    /// The stress solver's workgroup counts (storage; [`solver::slot`]) and their copy as
    /// indirect arguments.
    SolverArgs,
    SolverDispatch,
    /// The wide solve's row table (its node, bond and row grids from byte 16) and its
    /// indirect arguments: the iterations' (bytes 0..48, which the gate zeroes once no row
    /// is active), then the set-up's and the end's (48..96).
    WideMeta,
    WideDispatch,
}

/// A timestamp a pass writes, where the host times the stage (a diagnostic; hosts that
/// do not time ignore it): query `index`, at the pass's end or its beginning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub index: u32,
    pub end: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    /// The dispatches up to the next command that is not one run together (a compute pass
    /// on WebGPU). A pass may hold none (a timing mark).
    Pass { label: Option<&'static str>, stamp: Option<Stamp> },
    Copy { src: Buf, src_offset: u64, dst: Buf, dst_offset: u64, size: u64 },
    /// `entry` over `items` items: `ceil(items / workgroup size)` workgroups, one at least.
    Dispatch { module: Module, entry: &'static str, items: u64 },
    /// `entry` over `groups` workgroups (one at least) in rows of `row` along x, as many
    /// rows as they take along y: the kernel numbers its group `y * row + x`.
    DispatchRows { module: Module, entry: &'static str, groups: u32, row: u32 },
    /// `entry` over the workgroups of triple `arg` of [`Buf::Args`]. `scan`: the free-slot
    /// scan's bind group of that index in place of the entry's own (the scan kernels run
    /// over the chunks and again over the body slots).
    Indirect { module: Module, entry: &'static str, arg: u64, scan: Option<u8> },
    /// `entry` over the workgroups at byte `offset` of `args`.
    IndirectAt { module: Module, entry: &'static str, args: Buf, offset: u64 },
    /// `entry` over `groups` workgroups.
    Groups { module: Module, entry: &'static str, groups: u32 },
    /// The parameter block (`coupling.slang` `IngestParams`, as words) of the module's
    /// dispatches from here to its next one.
    Params { module: Module, words: [u32; 8] },
    /// The contact ingest's gated merge goes here ([`ingest::merge`]).
    IngestMerge,
    /// The stress solver's frame goes here (stress-gpu `BlastFrame::record`).
    Solver,
}

/// How a predicted strike's breaks are decided (`FractureParams::strike_model`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strike {
    Hertz,
    Analytic,
    Radial,
}

/// What a frame's list depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOpts {
    /// Behind the quiet gate, with the ingest's merge in the frame.
    pub gated: bool,
    /// The chunks and bonds in use, and the body slots.
    pub chunks: u64,
    pub bonds: u64,
    pub bodies: u64,
    /// The chunks the tables hold (the crush mark runs over them all).
    pub table_chunks: u64,
    pub crush: bool,
    /// The stress solver decides the bonds (in place of the contact-force threshold).
    pub solver: bool,
    /// The load path decides them: its sweeps a frame.
    pub load_path: Option<u32>,
    pub strike: Strike,
    /// A strike's structural response (`gi_mode`).
    pub response: bool,
    pub shots: u32,
    pub stress_view: bool,
    /// Open the frame's event lists (not already opened by the in-step contact fracture).
    pub events: bool,
    /// Bonds the host breaks this frame.
    pub host_breaks: u64,
    /// The load path's stages timed, a pass each (`STRESS_CS_TIMING=1`).
    pub timed: bool,
    /// Bodies split (off: bonds break and nothing splits, a diagnostic).
    pub split: bool,
}

const F: Module = Module::Fracture;

fn pass(out: &mut Vec<Cmd>) {
    out.push(Cmd::Pass { label: None, stamp: None });
}

fn labelled(out: &mut Vec<Cmd>, label: &'static str) {
    out.push(Cmd::Pass { label: Some(label), stamp: None });
}

fn direct(out: &mut Vec<Cmd>, entry: &'static str, items: u64) {
    out.push(Cmd::Dispatch { module: F, entry, items });
}

fn indirect(out: &mut Vec<Cmd>, entry: &'static str, arg: u64) {
    out.push(Cmd::Indirect { module: F, entry, arg, scan: None });
}

fn copy_args(out: &mut Vec<Cmd>) {
    out.push(Cmd::Copy { src: Buf::Sizes, src_offset: 0, dst: Buf::Args, dst_offset: 0, size: arg::BYTES });
}

fn sweep(s: u32) -> &'static str {
    if s % 2 == 0 {
        "fr_lp_sweep_even"
    } else {
        "fr_lp_sweep_odd"
    }
}

/// A frame: its loads and breaks, its predicted strikes, then its splits, crushes and
/// log (each gated on the GPU).
pub fn frame(o: &FrameOpts) -> Vec<Cmd> {
    let mut out = Vec::new();
    let (n, e, b) = (o.chunks, o.bonds, o.bodies);
    let out = &mut out;
    pass(out);
    direct(out, "fr_frame_begin", 1);
    if o.gated {
        direct(out, "fr_quiet", 1);
        out.push(Cmd::IngestMerge);
    }
    if o.events || o.host_breaks > 0 {
        pass(out);
        if o.events {
            // Before the host's breaks, which it records.
            direct(out, "fr_events_reset", 1);
        }
        if o.host_breaks > 0 {
            direct(out, "fr_host_breaks", o.host_breaks);
        }
    }
    let timed = o.timed && o.load_path.is_some();
    if let (true, Some(sweeps)) = (timed, o.load_path) {
        // The load path's stages, a pass each.
        let at = |i: u32| Some(Stamp { index: CS_STAMPS + i, end: true });
        out.push(Cmd::Pass { label: None, stamp: at(0) });
        let loads: &[&'static str] = if o.crush { &["fr_loads", "fr_crush_loads"] } else { &["fr_loads"] };
        let stages: [&[&'static str]; 5] = [loads, &["fr_lp_clear", "fr_lp_sum", "fr_lp_relief"], &["fr_lp_block"], &[], &["fr_lp_breaks"]];
        for (i, names) in stages.iter().enumerate() {
            out.push(Cmd::Pass { label: None, stamp: at(i as u32 + 1) });
            for &name in names.iter() {
                let size = match name {
                    "fr_lp_clear" => b,
                    "fr_lp_breaks" => e,
                    _ => n,
                };
                direct(out, name, size);
            }
            if i == 3 {
                for s in 0..sweeps {
                    direct(out, sweep(s), n);
                }
            }
        }
    }
    labelled(out, "fracture loads");
    if !timed {
        if o.gated {
            indirect(out, "fr_loads", arg::Q_CHUNKS);
        } else {
            direct(out, "fr_loads", n);
        }
        if o.crush && !o.solver {
            // Before the breaks: the chunk's bonds as the frame's loads met them.
            direct(out, "fr_crush_loads", n);
        }
        if o.solver {
            // The stress solver decides the bonds, between the passes.
        } else if let Some(sweeps) = o.load_path {
            direct(out, "fr_lp_clear", b);
            direct(out, "fr_lp_sum", n);
            direct(out, "fr_lp_relief", n);
            direct(out, "fr_lp_block", n);
            for s in 0..sweeps {
                direct(out, sweep(s), n);
            }
            if o.stress_view {
                direct(out, "fr_lp_view", n);
            }
            direct(out, "fr_lp_breaks", e);
        } else {
            if o.stress_view {
                direct(out, "fr_stress_clear", n);
                direct(out, "fr_stress_breaks", e);
            }
            if o.gated {
                indirect(out, "fr_breaks", arg::Q_BONDS);
            } else {
                direct(out, "fr_breaks", e);
            }
        }
    }
    if o.solver {
        out.push(Cmd::Solver);
    }
    pass(out);
    if o.crush && o.solver {
        // After the solve: its bond forces as the solve met them.
        direct(out, "fr_crush_loads", n);
    }
    if o.shots > 0 {
        let shots = o.shots as u64;
        direct(out, "fr_strike_reset", b);
        direct(out, "fr_strike_bounds", n);
        direct(out, "fr_strike_find", b);
        direct(out, "fr_strike_entry", n);
        direct(out, "fr_strike_nearest", n);
        if o.crush {
            // Before fr_strike_force, which drops a strike it finds no entry for: the
            // contact patch, the chunks on it, their crush from each shot's kinetic energy.
            direct(out, "fr_strike_patch", b);
            direct(out, "fr_strike_cover", n);
            direct(out, "fr_strike_claim", n);
            direct(out, "fr_strike_crush", shots);
        }
        direct(out, "fr_strike_force", b);
        match o.strike {
            Strike::Hertz => direct(out, "fr_strike_breaks", e),
            Strike::Radial => direct(out, "fr_strike_radial", e),
            Strike::Analytic => {
                if o.response {
                    direct(out, "fr_gi_begin", 1);
                }
                direct(out, "fr_il_first_reset", MAX_SHOTS as u64);
                direct(out, "fr_il_first", b);
                direct(out, "fr_il_begin", b);
                direct(out, "fr_il_tube_count", n);
                direct(out, "fr_il_tube_base", b);
                direct(out, "fr_il_tube_collect", n);
                // A group per body slot.
                out.push(Cmd::DispatchRows { module: F, entry: "fr_il_run", groups: b as u32, row: RUN_ROW });
                direct(out, "fr_il_strike", b);
                if o.response {
                    // The structural response of what the strikes left in their
                    // structures (a shot strikes one body a frame: a group per shot, those
                    // past the struck ones leave at once), before the local outcome's
                    // breaks.
                    direct(out, "fr_cs_count", n);
                    direct(out, "fr_cs_base", MAX_SHOTS as u64);
                    direct(out, "fr_cs_collect", n);
                    out.push(Cmd::DispatchRows { module: F, entry: "fr_gi_solve", groups: o.shots.min(MAX_SHOTS), row: RUN_ROW });
                    direct(out, "fr_cs_clean", MAX_SHOTS as u64);
                }
                direct(out, "fr_il_mark", n);
                direct(out, "fr_il_breaks", e);
            }
        }
        if o.crush {
            direct(out, "fr_crush_bill", shots);
            direct(out, "fr_crush_push", b);
        }
    }
    if o.crush {
        direct(out, "fr_crush_bonds", e);
    }
    direct(out, "fr_gate", 1);
    split(out, o.split);
    if o.strike == Strike::Analytic && o.shots > 0 {
        pass(out);
        direct(out, "fr_il_carry", n);
    }
    if o.crush {
        crush(out, o.table_chunks);
    }
    log(out);
    std::mem::take(out)
}

/// The split kernels, sized by the gate (`fr_gate`, in the pass before).
pub fn split(out: &mut Vec<Cmd>, on: bool) {
    if !on {
        return;
    }
    copy_args(out);
    labelled(out, "fracture split");
    indirect(out, "fr_uf_init", arg::CHUNKS);
    indirect(out, "fr_body_init", arg::BODIES);
    indirect(out, "fr_uf_union", arg::BONDS);
    indirect(out, "fr_uf_flatten", arg::CHUNKS);
    indirect(out, "fr_best_count", arg::CHUNKS);
    indirect(out, "fr_best_root", arg::CHUNKS);
    indirect(out, "fr_mark", arg::CHUNKS);
    // The new pieces ranked, then the free slots listed (ascending).
    indirect(out, "fr_scan_blocks", arg::SCAN_CHUNKS);
    indirect(out, "fr_scan_top", arg::ONE);
    indirect(out, "fr_scan_add", arg::CHUNKS);
    out.push(Cmd::Indirect { module: F, entry: "fr_scan_blocks", arg: arg::SCAN_BODIES, scan: Some(0) });
    out.push(Cmd::Indirect { module: F, entry: "fr_scan_top", arg: arg::ONE, scan: Some(1) });
    out.push(Cmd::Indirect { module: F, entry: "fr_scan_add", arg: arg::BODIES, scan: Some(2) });
    indirect(out, "fr_freelist", arg::BODIES);
    indirect(out, "fr_assign", arg::CHUNKS);
    indirect(out, "fr_piece", arg::CHUNKS);
    indirect(out, "fr_commit", arg::CHUNKS);
    indirect(out, "fr_reparent", arg::CHUNKS);
    indirect(out, "fr_body_clear", arg::BODIES);
}

/// After the split: each chunk crushed this frame ranked (chunk order), its event and
/// fragments written and its body parked (gated: nothing on a frame with no crush).
pub fn crush(out: &mut Vec<Cmd>, table_chunks: u64) {
    pass(out);
    direct(out, "fr_crush_mark", table_chunks);
    direct(out, "fr_crush_gate", 1);
    copy_args(out);
    labelled(out, "fracture crush");
    indirect(out, "fr_scan_blocks", arg::CRUSH_SCAN);
    indirect(out, "fr_scan_top", arg::CRUSH_ONE);
    indirect(out, "fr_scan_add", arg::CRUSH_CHUNKS);
    indirect(out, "fr_crush_apply", arg::CRUSH_CHUNKS);
    indirect(out, "fr_crush_end", arg::CRUSH_ONE);
}

/// The frame's change log after its split (`fracture_log.slang`), sized by the same gate.
pub fn log(out: &mut Vec<Cmd>) {
    copy_args(out);
    pass(out);
    for (entry, arg) in [("log_bonds", arg::BONDS), ("log_chunks", arg::CHUNKS), ("log_frames", arg::BODIES)] {
        out.push(Cmd::Indirect { module: Module::FractureLog, entry, arg, scan: None });
    }
}

/// Only the split (the gate, then every split kernel) and the log: the bonds already
/// marked broken and their bodies dirty, with the host's own breaks first. `crush`: with
/// crushing, the crushed chunks' events and fragments after the split (as a frame's);
/// `Some(true)`: a blast window's crush damage recorded first, and the crushed chunks'
/// bonds broken (`table_chunks`, `table_bonds`: the tables' sizes).
pub fn split_only(host_breaks: u64, crush_blast: Option<bool>, table_chunks: u64, table_bonds: u64, on: bool) -> Vec<Cmd> {
    let mut out = Vec::new();
    let out = &mut out;
    pass(out);
    if host_breaks > 0 {
        direct(out, "fr_host_breaks", host_breaks);
    }
    if crush_blast == Some(true) {
        direct(out, "fr_crush_blast", table_chunks);
        direct(out, "fr_crush_bonds", table_bonds);
    }
    direct(out, "fr_gate", 1);
    split(out, on);
    if crush_blast.is_some() {
        crush(out, table_chunks);
    }
    log(out);
    std::mem::take(out)
}

/// What the in-step contact fracture's list depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContactOpts {
    /// The step's event lists start here.
    pub events: bool,
    /// Where `work` holds the step's contact rows' count (a word index).
    pub rows_word: u64,
    /// The profile's verdict: the load path, the analytic strike, or the threshold rule.
    pub load_path: bool,
    pub analytic: bool,
    /// The load path's verdict with the strike's structural response (`gi_mode`).
    pub response: bool,
    /// Its stages timed (`STRESS_CS_TIMING=1`).
    pub timed: bool,
    pub split: bool,
}

const CONTACT: &str = "contact fracture";

/// The in-step contact fracture, replayed within the engine's step where its contacts are
/// made and kept: this step's contacts closing fast on a structure are its strikes, each
/// struck structure's largest judged by the profile's stress solver, what breaks split,
/// and the rows of the moved colliders re-anchored to their pieces, before the step
/// solves. Nothing is read back; without a strike the solve and the split dispatch
/// nothing.
pub fn contact_fracture(o: &ContactOpts) -> Vec<Cmd> {
    let mut out = Vec::new();
    let out = &mut out;
    if o.events {
        // The contact fracture's breaks and splits are the step's first.
        pass(out);
        direct(out, "fr_events_reset", 1);
    }
    // The step's contact rows' count (`StepCounters` word 2) where the kernels read it.
    out.push(Cmd::Copy { src: Buf::EngineCounters, src_offset: 8, dst: Buf::Work, dst_offset: 4 * o.rows_word, size: 4 });
    let stage = |out: &mut Vec<Cmd>, index: u32, end: bool| {
        out.push(Cmd::Pass { label: Some(CONTACT), stamp: o.timed.then_some(Stamp { index, end }) });
    };
    // The detection, sized by the step's contact rows; then everything after it sized by
    // what it found (`fr_cs_gate`): a step without a strike dispatches no group past here.
    stage(out, 0, false);
    direct(out, "fr_cs_begin", 1);
    copy_args(out);
    stage(out, 1, true);
    indirect(out, "fr_cs_contacts", arg::CS_ROWS);
    direct(out, "fr_cs_gate", 1);
    copy_args(out);
    // The profile's verdict.
    stage(out, 2, true);
    for pick in ["fr_cs_pick_a", "fr_cs_pick_b", "fr_cs_pick_c", "fr_cs_pick_d"] {
        indirect(out, pick, arg::CS_PICK);
    }
    indirect(out, "fr_cs_record", arg::CS_LIST);
    if o.load_path {
        indirect(out, "fr_cs_count", arg::CS_CHUNKS);
        indirect(out, "fr_cs_base", arg::CS_LIST);
        indirect(out, "fr_cs_collect", arg::CS_CHUNKS);
        // A group per struck structure, all its sweeps in the one dispatch.
        indirect(out, if o.response { "fr_gi_solve" } else { "fr_cs_solve" }, arg::CS_BODIES);
    } else if o.analytic {
        indirect(out, "fr_csl_begin", arg::CS_SLOTS);
        indirect(out, "fr_csl_tube_count", arg::CS_CHUNKS);
        indirect(out, "fr_csl_tube_base", arg::CS_SLOTS);
        indirect(out, "fr_csl_tube_collect", arg::CS_CHUNKS);
        // A group per struck structure.
        indirect(out, "fr_csl_run", arg::CS_BODIES);
        indirect(out, "fr_csl_strike", arg::CS_SLOTS);
        indirect(out, "fr_csl_mark", arg::CS_CHUNKS);
        indirect(out, "fr_csl_breaks", arg::CS_BONDS);
    } else {
        indirect(out, "fr_cs_hertz", arg::CS_BONDS);
    }
    direct(out, "fr_cs_gate2", 1);
    copy_args(out);
    stage(out, 3, true);
    indirect(out, "fr_cs_save", arg::CS_SAVE);
    direct(out, "fr_gate", 1);
    split(out, o.split);
    stage(out, 4, false);
    indirect(out, "fr_cs_fix", arg::CS_SAVE);
    if o.analytic {
        indirect(out, "fr_csl_carry", arg::CS_CHUNKS);
    }
    indirect(out, "fr_cs_clean", arg::CS_LIST);
    stage(out, 5, true);
    std::mem::take(out)
}

/// The contact ingest (`coupling.slang`): the rigid engine's contact rows as each
/// chunk's manifolds, which the fracture's loads read.
pub mod ingest {
    use super::{arg, Buf, Cmd, Module};

    const M: Module = Module::Coupling;

    /// Bytes of an engine body (engine-schema `BodyState`).
    pub const BODY_BYTES: u64 = 112;

    /// The gate's triples of each merge pass, after the gate's first (`coupling.slang`
    /// `GATE_IG_*`): clear, count, scan, scan blocks, scan add, scatter, merge.
    pub const GATE_SLOTS: [u64; 7] = [2, 3, 4, 5, 6, 3, 6];

    /// What an ingest was made for.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Dims {
        pub max_rows: u32,
        pub max_substeps: u32,
        pub max_bodies: u32,
        pub chunk_count: u32,
    }

    fn groups(n: u64, size: u64) -> u32 {
        n.div_ceil(size).max(1) as u32
    }

    /// `coupling.slang` `IngestParams`: row_offset, max_rows, substep, chunk_count,
    /// record_count, max_bodies, h, gate.
    fn params(d: &Dims, row_offset: u32, substep: u32, record_count: u32, h: f32, gate: u32) -> Cmd {
        Cmd::Params { module: M, words: [row_offset, d.max_rows, substep, d.chunk_count, record_count, d.max_bodies, h.to_bits(), gate] }
    }

    /// The frame's start: the engine's bodies as they are now (closing speeds use them).
    pub fn begin_frame(d: &Dims) -> Vec<Cmd> {
        vec![Cmd::Copy { src: Buf::EngineBodies, src_offset: 0, dst: Buf::StartBodies, dst_offset: 0, size: BODY_BYTES * d.max_bodies as u64 }]
    }

    /// One engine substep's rows (the step just before this in the list), from
    /// `row_offset` of the engine's contacts: its records, at slot `substep`.
    pub fn substep(d: &Dims, row_offset: u32, substep: u32, h: f32) -> Vec<Cmd> {
        assert!(substep < d.max_substeps, "ingest: {substep} substeps, room for {}", d.max_substeps);
        vec![
            params(d, row_offset, substep, 2 * d.max_rows * d.max_substeps, h, 0),
            Cmd::Pass { label: Some("ingest rows"), stamp: None },
            Cmd::Groups { module: M, entry: "ingest_rows", groups: groups(d.max_rows as u64, 64) },
        ]
    }

    /// The records a merge of `substeps` substeps reads; with `rows`, substep 0 and the
    /// first `rows` rows of substep 1 (a host's own records there).
    pub fn record_count(d: &Dims, substeps: u32, rows: Option<u32>) -> u32 {
        match rows {
            Some(rows) => {
                assert!(d.max_substeps >= 2, "ingest: no second substep slot");
                2 * d.max_rows + 2 * rows.min(d.max_rows)
            }
            None if substeps == 0 => 0,
            None => 2 * d.max_rows * substeps,
        }
    }

    /// The frame's manifolds from its first `record_count` records. `gate`: behind the
    /// quiet gate (`ingest_gate`), whose triples start at that triple of the fracture's
    /// sizes; on a quiet frame the merge then dispatches no group (what it would write is
    /// already there).
    pub fn merge(d: &Dims, record_count: u32, gate: Option<u32>) -> Vec<Cmd> {
        let records = (record_count as u64).max(1);
        let n = d.chunk_count as u64;
        let mut out = vec![params(d, 0, 0, record_count, 0.0, gate.unwrap_or(0))];
        if gate.is_some() {
            out.push(Cmd::Pass { label: Some("ingest gate"), stamp: None });
            out.push(Cmd::Groups { module: M, entry: "ingest_any", groups: groups(records, 64) });
            out.push(Cmd::Groups { module: M, entry: "ingest_gate", groups: 1 });
            out.push(Cmd::Copy { src: Buf::Sizes, src_offset: 0, dst: Buf::Args, dst_offset: 0, size: arg::BYTES });
        }
        let passes: [(&'static str, &'static str, u32); 7] = [
            ("ingest clear", "ingest_clear", groups(2 * n, 64)),
            ("ingest count", "ingest_count", groups(records, 64)),
            ("ingest scan", "ingest_scan", groups(n, 1024)),
            ("ingest scan blocks", "ingest_scan_blocks", 1),
            ("ingest scan add", "ingest_scan_add", groups(n, 64)),
            ("ingest scatter", "ingest_scatter", groups(records, 64)),
            ("ingest merge", "ingest_merge", groups(n, 64)),
        ];
        for ((label, entry, groups), slot) in passes.into_iter().zip(GATE_SLOTS) {
            out.push(Cmd::Pass { label: Some(label), stamp: None });
            out.push(match gate {
                Some(first) => Cmd::Indirect { module: M, entry, arg: first as u64 + slot, scan: None },
                None => Cmd::Groups { module: M, entry, groups },
            });
        }
        out
    }
}

/// The stress solver's frame (vibe-land's GPU stress solver, `blast_*.slang`): the bond
/// impulses that carry the frame's loads, solved per island, and the bonds they break,
/// in place of the fracture's contact-force threshold.
pub mod solver {
    use super::{Buf, Cmd, Module};

    /// Word indices of [`Buf::SolverArgs`] (three words a grid).
    pub mod slot {
        pub const SOLVE: u64 = 0;
        pub const NODES: u64 = 4;
        pub const BONDS: u64 = 7;
        pub const SCAN: u64 = 10;
        pub const ONE: u64 = 13;
        pub const SORT: u64 = 16;
        /// Bytes of the counts.
        pub const BYTES: u64 = 80;
    }

    /// What a wide kernel runs over.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Grid {
        One,
        Islands,
        Nodes,
        Bonds,
        Rows,
    }

    /// The islands' rebuild from the bonds' health, after its gate (`bt_gate`): each
    /// entry and the grid it runs over.
    pub const TOPOLOGY: [(&str, u64); 12] = [
        ("bt_reset", slot::NODES),
        ("bt_union", slot::BONDS),
        ("bt_flatten", slot::NODES),
        ("bt_count_nodes", slot::NODES),
        ("bt_count_bonds", slot::BONDS),
        ("bt_scan_blocks", slot::SCAN),
        ("bt_scan_top", slot::ONE),
        ("bt_scan_add", slot::SCAN),
        ("bt_islands", slot::NODES),
        ("bt_scatter_nodes", slot::NODES),
        ("bt_scatter_bonds", slot::BONDS),
        ("bt_sort", slot::SORT),
    ];

    /// The wide solve: its set-up, one iteration, its end.
    pub const WIDE_SETUP: &[(&str, Grid)] = &[
    ("w_reset", Grid::One),
    ("w_classify", Grid::Islands),
    ("w_args", Grid::One),
    ("w_init_nodes", Grid::Nodes),
    ("w_init_bonds", Grid::Bonds),
    ("w_setup", Grid::Rows),
    ("w_warm", Grid::Nodes),
    ];
    pub const WIDE_ITERATION: &[(&str, Grid)] = &[
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
    pub const WIDE_END: &[(&str, Grid)] = &[("w_final", Grid::Bonds), ("w_report", Grid::Rows)];

    /// What the solver's list depends on.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Opts {
        /// The graph's nodes (the fracture's chunks) and bonds.
        pub nodes: u64,
        pub bonds: u64,
        /// The native solve (projected PCG; with the islands' rigid-mode frames and the
        /// wide path), or the host-path CGLS.
        pub native: bool,
        /// The solve's iterations at most (each wide iteration is in the list).
        pub iterations: u32,
        /// The bonds' virial, for crushing.
        pub virial: bool,
        pub stress_view: bool,
    }

    fn pass(out: &mut Vec<Cmd>) {
        out.push(Cmd::Pass { label: None, stamp: None });
    }

    /// The islands' rebuild. It runs only when a bond broke since the last one (gated on
    /// the GPU: nothing is read back).
    pub fn topology(out: &mut Vec<Cmd>, native: bool) {
        pass(out);
        out.push(Cmd::Groups { module: Module::BlastTopology, entry: "bt_gate", groups: 1 });
        for (i, &(entry, at)) in TOPOLOGY.iter().enumerate() {
            // The gate sized the rebuild; bt_scan_top then sizes the sort (and the solve).
            if i == 0 || at == slot::SORT {
                out.push(Cmd::Copy { src: Buf::SolverArgs, src_offset: 0, dst: Buf::SolverDispatch, dst_offset: 0, size: slot::BYTES });
            }
            pass(out);
            out.push(Cmd::IndirectAt { module: Module::BlastTopology, entry, args: Buf::SolverDispatch, offset: 4 * at });
        }
        if native {
            // The rebuilt islands' rigid-mode frames, one group each (the sort's size: zero
            // on a frame without a break).
            pass(out);
            out.push(Cmd::IndirectAt { module: Module::BlastMotion, entry: "blast_motion", args: Buf::SolverDispatch, offset: 4 * slot::SORT });
        }
    }

    /// The solve of every island (one group each, sized on the GPU), then the large
    /// islands' over many groups.
    pub fn solve(out: &mut Vec<Cmd>, o: &Opts) {
        pass(out);
        let (module, entry) = if o.native { (Module::BlastNative, "blast_native") } else { (Module::BlastCgls, "blast_solve") };
        out.push(Cmd::IndirectAt { module, entry, args: Buf::SolverDispatch, offset: 4 * slot::SOLVE });
        if o.native {
            wide(out, o);
        }
    }

    /// The large islands' solve: set-up (their rows listed, then the sizes copied for the
    /// indirect dispatches), every iteration behind its gate, the end.
    pub fn wide(out: &mut Vec<Cmd>, o: &Opts) {
        const W: Module = Module::BlastNativeWide;
        // `sizes`: the byte offset of the copy of the sizes a kernel takes (0: the
        // iterations').
        let run = |out: &mut Vec<Cmd>, &(entry, grid): &(&'static str, Grid), sizes: u64| {
            out.push(match grid {
                Grid::One => Cmd::Groups { module: W, entry, groups: 1 },
                Grid::Islands => Cmd::Dispatch { module: W, entry, items: o.nodes.max(1) },
                Grid::Nodes => Cmd::IndirectAt { module: W, entry, args: Buf::WideDispatch, offset: sizes },
                Grid::Bonds => Cmd::IndirectAt { module: W, entry, args: Buf::WideDispatch, offset: sizes + 16 },
                Grid::Rows => Cmd::IndirectAt { module: W, entry, args: Buf::WideDispatch, offset: sizes + 32 },
            });
        };
        pass(out);
        for k in &WIDE_SETUP[..3] {
            run(out, k, 48);
        }
        // The node, bond and row grids, once for the iterations and once for the set-up
        // and the end.
        out.push(Cmd::Copy { src: Buf::WideMeta, src_offset: 16, dst: Buf::WideDispatch, dst_offset: 0, size: 48 });
        out.push(Cmd::Copy { src: Buf::WideMeta, src_offset: 16, dst: Buf::WideDispatch, dst_offset: 48, size: 48 });
        pass(out);
        for k in &WIDE_SETUP[3..] {
            run(out, k, 48);
        }
        for _ in 0..o.iterations {
            out.push(Cmd::Groups { module: Module::BlastWideGate, entry: "w_gate", groups: 1 });
            for k in WIDE_ITERATION {
                run(out, k, 0);
            }
        }
        for k in WIDE_END {
            run(out, k, 48);
        }
    }

    /// The frame's solver work after the fracture's loads, before its gate and split.
    pub fn frame(o: &Opts) -> Vec<Cmd> {
        let mut out = Vec::new();
        let out = &mut out;
        let (n, m) = (o.nodes.max(1), o.bonds.max(1));
        const F: Module = Module::BlastFracture;
        pass(out);
        // The fracture's own breaks (strikes, crushes) into the solver first: one bond state.
        out.push(Cmd::Dispatch { module: F, entry: "blast_sync", items: m });
        out.push(Cmd::Dispatch { module: F, entry: "blast_loads", items: n });
        topology(out, o.native);
        solve(out, o);
        if o.virial {
            // The bonds' virial for crushing, before the frame's damage.
            pass(out);
            out.push(Cmd::Dispatch { module: Module::BlastVirial, entry: "blast_virial", items: n });
        }
        // Every bond's stresses and damage.
        pass(out);
        out.push(Cmd::Dispatch { module: Module::BlastMaterial, entry: "blast_bond_stress", items: m });
        if o.stress_view {
            pass(out);
            out.push(Cmd::Dispatch { module: Module::BlastView, entry: "blast_view_clear", items: n });
            out.push(Cmd::Dispatch { module: Module::BlastView, entry: "blast_view", items: m });
        }
        pass(out);
        out.push(Cmd::Dispatch { module: F, entry: "blast_breaks", items: m });
        std::mem::take(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatches(cmds: &[Cmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, Cmd::Dispatch { .. } | Cmd::DispatchRows { .. } | Cmd::Indirect { .. })).count()
    }

    fn copies(cmds: &[Cmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, Cmd::Copy { .. })).count()
    }

    fn threshold() -> FrameOpts {
        FrameOpts {
            gated: true,
            chunks: 1000,
            bonds: 3000,
            bodies: 1000,
            table_chunks: 1000,
            crush: false,
            solver: false,
            load_path: None,
            strike: Strike::Hertz,
            response: false,
            shots: 0,
            stress_view: false,
            events: true,
            host_breaks: 0,
            timed: false,
            split: true,
        }
    }

    /// The counts of engine/docs/world-on-cuda.md's inventory (the fracture's own
    /// dispatches: without the ingest's ten and the solver's).
    #[test]
    fn a_frame_s_dispatch_counts() {
        // begin, quiet, events reset, loads, breaks, gate, 19 split, 3 log.
        let f = frame(&threshold());
        assert_eq!(dispatches(&f), 28);
        assert_eq!(copies(&f), 2);
        // Predicted shots under the Hertz rule: reset, bounds, find, entry, nearest,
        // force, breaks.
        let shots = FrameOpts { shots: 4, ..threshold() };
        assert_eq!(dispatches(&frame(&shots)), 28 + 7);
        // The analytic strike: 6 shared, 8 of its own, its breaks' mark and breaks, the carry.
        let analytic = FrameOpts { strike: Strike::Analytic, ..shots };
        assert_eq!(dispatches(&frame(&analytic)), 28 + 6 + 10 + 1);
        // The load path with 16 sweeps: loads, 4, the sweeps, its breaks, in place of
        // loads and breaks.
        let load_path = FrameOpts { load_path: Some(16), ..threshold() };
        assert_eq!(dispatches(&frame(&load_path)), 28 - 2 + 1 + 4 + 16 + 1);
        // Crushing: its loads, its bonds, then mark, gate and 5 gated.
        let crush = FrameOpts { crush: true, ..threshold() };
        let f = frame(&crush);
        assert_eq!(dispatches(&f), 28 + 2 + 7);
        assert_eq!(copies(&f), 3);
    }

    /// A list never changes with what a frame breaks: every size the frame decides is an
    /// indirect dispatch's. Only the host's own inputs change it.
    #[test]
    fn the_split_is_all_indirect() {
        let mut out = Vec::new();
        split(&mut out, true);
        assert_eq!(out.len(), 2 + 19);
        assert!(matches!(out[0], Cmd::Copy { src: Buf::Sizes, dst: Buf::Args, size: arg::BYTES, .. }));
        assert!(out[2..].iter().all(|c| matches!(c, Cmd::Indirect { module: Module::Fracture, .. })));
        let mut off = Vec::new();
        split(&mut off, false);
        assert!(off.is_empty());
    }

    /// No pass reads indirect arguments the same pass's gate wrote: after a gate
    /// (`fr_gate`, `fr_cs_begin`, `fr_cs_gate`, `fr_cs_gate2`, `fr_crush_gate`, `fr_quiet`)
    /// a copy to the arguments comes before the next indirect dispatch.
    #[test]
    fn a_copy_follows_every_gate() {
        let lists = [
            frame(&FrameOpts { crush: true, shots: 2, strike: Strike::Analytic, response: true, ..threshold() }),
            frame(&FrameOpts { gated: false, ..threshold() }),
            contact_fracture(&ContactOpts { events: true, rows_word: 7, load_path: true, analytic: false, response: true, timed: false, split: true }),
            contact_fracture(&ContactOpts { events: false, rows_word: 7, load_path: false, analytic: true, response: false, timed: true, split: true }),
            split_only(3, Some(true), 100, 200, true),
        ];
        const GATES: [&str; 6] = ["fr_gate", "fr_cs_begin", "fr_cs_gate", "fr_cs_gate2", "fr_crush_gate", "fr_quiet"];
        for list in &lists {
            let mut stale = false;
            for c in list {
                match c {
                    Cmd::Dispatch { entry, .. } if GATES.contains(entry) => stale = true,
                    // The ingest's merge copies the sizes itself.
                    Cmd::Copy { src: Buf::Sizes, dst: Buf::Args, .. } | Cmd::IngestMerge => stale = false,
                    Cmd::Indirect { entry, .. } => assert!(!stale, "{entry} reads arguments a gate has rewritten"),
                    _ => {}
                }
            }
        }
    }

    /// engine/docs/world-on-cuda.md's count of the solver's frame at 16 iterations: the
    /// sync and loads, the topology's gate and 12, the islands' frames, the solve, the wide
    /// path's 3 + 4 + 16 x (its gate + 28) + 2, the stress, the breaks.
    #[test]
    fn the_solver_s_frame() {
        let o = solver::Opts { nodes: 3711, bonds: 9000, native: true, iterations: 16, virial: false, stress_view: false };
        let f = solver::frame(&o);
        let launches = f.iter().filter(|c| matches!(c, Cmd::Dispatch { .. } | Cmd::Groups { .. } | Cmd::IndirectAt { .. })).count();
        assert_eq!(solver::WIDE_ITERATION.len(), 28);
        assert_eq!(launches, 2 + 13 + 1 + 1 + (3 + 4 + 16 * 29 + 2) + 1 + 1);
        assert_eq!(launches, 492);
        // The arguments are copied twice for the topology, twice for the wide path.
        assert_eq!(copies(&f), 4);
        // The host path: no frames, no wide path.
        let host = solver::frame(&solver::Opts { native: false, ..o });
        assert_eq!(host.iter().filter(|c| matches!(c, Cmd::Dispatch { .. } | Cmd::Groups { .. } | Cmd::IndirectAt { .. })).count(), 2 + 13 + 1 + 1 + 1);
        assert!(host.iter().any(|c| matches!(c, Cmd::IndirectAt { module: Module::BlastCgls, entry: "blast_solve", .. })));
    }

    #[test]
    fn the_ingest_s_merge() {
        let d = ingest::Dims { max_rows: 4096, max_substeps: 2, max_bodies: 1000, chunk_count: 5000 };
        // The gate's two, then seven passes of one dispatch each, all indirect behind it.
        let gated = ingest::merge(&d, ingest::record_count(&d, 1, None), Some(arg::Q_CHUNKS as u32));
        assert_eq!(dispatches(&gated) + gated.iter().filter(|c| matches!(c, Cmd::Groups { .. })).count(), 9);
        assert_eq!(gated.iter().filter(|c| matches!(c, Cmd::Indirect { .. })).count(), 7);
        assert!(gated.iter().all(|c| !matches!(c, Cmd::Indirect { arg, .. } if *arg < arg::Q_CHUNKS + 2 || *arg >= arg::Q_STATE)));
        let plain = ingest::merge(&d, ingest::record_count(&d, 1, None), None);
        assert_eq!(plain.iter().filter(|c| matches!(c, Cmd::Groups { .. })).count(), 7);
        assert_eq!(copies(&plain), 0);
        // 2 * 5000 chunk words over 64, 8192 records over 64, 5000 chunks over 1024.
        assert!(plain.contains(&Cmd::Groups { module: Module::Coupling, entry: "ingest_clear", groups: 157 }));
        assert!(plain.contains(&Cmd::Groups { module: Module::Coupling, entry: "ingest_count", groups: 128 }));
        assert!(plain.contains(&Cmd::Groups { module: Module::Coupling, entry: "ingest_scan", groups: 5 }));
        assert_eq!(ingest::record_count(&d, 0, None), 0);
        assert_eq!(ingest::record_count(&d, 1, Some(10)), 2 * 4096 + 20);
    }

    #[test]
    fn the_contact_fracture_s_counts() {
        let o = ContactOpts { events: true, rows_word: 0, load_path: false, analytic: false, response: false, timed: false, split: true };
        // events reset, begin, contacts, gate, 4 picks, record, hertz, gate2, save, gate,
        // 19 split, fix, clean.
        assert_eq!(dispatches(&contact_fracture(&o)), 34);
        assert_eq!(dispatches(&contact_fracture(&ContactOpts { load_path: true, ..o })), 37);
        assert_eq!(dispatches(&contact_fracture(&ContactOpts { analytic: true, ..o })), 42);
        // Four copies of the sizes and the rows' count.
        assert_eq!(copies(&contact_fracture(&o)), 5);
    }
}
