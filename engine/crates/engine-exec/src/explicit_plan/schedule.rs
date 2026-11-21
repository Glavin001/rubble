//! One explicit-beams frame for every host: which path it runs, its parameter copies
//! and its launches in order.
//!
//! [`FramePlanner::commands`] lists a frame's plan commands. Around them the split
//! paths add launches of their own: self-collision split (a frame a step a dispatch,
//! with the walk and the pairs between), the split group path (the groups' substeps as
//! dispatches, the fleet-wide front and the levels across the GPU), and the wide passes'
//! reused pair list. [`frame`] decides which of them this frame runs, writes the
//! parameter copies they read, and orders every launch. A host replays the list
//! (`engine-webgpu`, `engine-cuda`, `engine-metal`): it maps each [`Op`] to its
//! pipeline and buffers and owns nothing of the order.

use super::{
    self_split_module_names, split_module_name, wide_list_module_name, Cmd, FramePlanner, Kernel, PARAM_COPIES,
    PARAM_LEVEL_COPIES, PARAM_STRIDE, PERF_AHEAD_APART, PERF_FLEET_ANY, PERF_FLEET_FRONT, PERF_FLEET_FUSE,
    PERF_FLEET_HITLAT, PERF_FLEET_KEEP, PERF_FLEET_LEVELS, PERF_FLEET_POST, PERF_FLEET_STATE, PERF_KEEP_TRACK_REST, PERF_MEMBER_LIST,
    PERF_WHEELS_APART, PERF_WIDE_FUSED, PERF_WIDE_LIST, PERF_DRIVE_AHEAD, PERF_SPLIT_WHEELS_AHEAD, PERF_TYRE_AHEAD,
    PERF_TYRE_EARLY, W_PERF_FLAGS, W_SLEEP, W_VV_FLEET, W_VV_FLEET_NEXT,
    W_VV_TRI_PAIRS, WIDE_AUX, WIDE_BEAMS_PASS, WIDE_END, WIDE_TRIS, FLEET_LEVEL_MAX, FLEET_LEVEL_SLOTS_SHIFT,
};

/// Byte offsets in the indirect arguments ([`super::BIND_ARGS`]) of the split group
/// path's dispatches (the kernel's vvArgs): the groups, and the vehicles when there is a
/// group.
pub const SPLIT_GROUP_ARGS: u64 = 64;
pub const SPLIT_MEMBER_ARGS: u64 = 80;
pub const SPLIT_TESTS_ARGS: u64 = 96;
/// Self-collision split with vehicle contact: every vehicle's step in every frame
/// (the kernel's vvArgs words 28..30).
pub const SPLIT_ALL_ARGS: u64 = 112;
/// Self-collision split without vehicle contact, vehicle sleep listed: the awake
/// vehicles' steps (the awake list's words 40..42, `explicit_beams_sleep.slang`).
pub const SELF_AWAKE_ARGS: u64 = 160;
/// The kernel's step-word phases (`explicit_beams_split.slang`): the triangle tests in
/// their own dispatches, the levels (and the tests' pairs) after them.
pub const SPLIT_TESTS: u32 = 0x2000_0000;
pub const SPLIT_LEVELS: u32 = 0x4000_0000;
/// The kernel's SPLIT_BEGIN: the parameter copy past the frame's end marks the groups'
/// frame start.
pub const SPLIT_BEGIN: u32 = u32::MAX;
/// Bytes of each of the fleet-wide front's two argument buffers (bindings 10 and 11 of
/// its plan module; [`Args::Fleet`]).
pub const FLEET_ARGS_BYTES: u64 = 512;
/// Byte offsets in the fleet-wide front's arguments: in the contact plan's (the scans,
/// the first plan, the larger groups), the first buffer's (the elements, the grids, the
/// sorted copy, one workgroup, the levels) and the second's (the walk, the pairs).
pub const FLEET_ELEMENT_ARGS: u64 = 0;
pub const FLEET_GRID_ARGS: u64 = 16;
pub const FLEET_SORT_ARGS: u64 = 32;
pub const FLEET_ONE_ARGS: u64 = 48;
pub const FLEET_WALK_ARGS: u64 = 0;
pub const FLEET_PAIRS_ARGS: u64 = 16;
/// The levels across the GPU: a level slot's at this plus 16 bytes a slot.
pub const FLEET_LEVEL_ARGS: u64 = 64;
pub const FLEET_SCAN_ARGS: u64 = 128;
pub const FLEET_FIRST_ARGS: u64 = 144;
pub const FLEET_LARGER_ARGS: u64 = 176;
/// The levels across the GPU's parameter copies' step words (the kernel's SPLIT_LV,
/// LV_*): SPLIT_LV | phase << 8 | level.
pub const SPLIT_LV: u32 = 0x1000_0000;
pub const LV_SNAP: u32 = 4;
pub const LV_PLAN: u32 = 5;
pub const LV_END: u32 = 6;
/// Bytes of the self-collision split's indirect arguments (two kinds, three slots of
/// three words: explicit_beams_selfsplit.slang selfArgsAt, selfWalkArgsAt).
pub const SELF_ARGS_BYTES: u64 = 128;

/// A dispatch of the split group path (contact-groups-design.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupOp {
    /// A groups' dispatch (`tests`: the tests across workgroups) of parameter copy `copy`,
    /// its size at byte `args` of the plan's arguments.
    /// `part` names it in timings: 0 the frame's start, 1 a step's front (candidates,
    /// grids, lists), 2 the tests' walk, 3 the listed pairs' tests, 4 the levels (the
    /// hits applied; the last with the frame's end), 5 a whole pass in the group's workgroup,
    /// 6 the levels across the GPU's end.
    Group { copy: u32, tests: bool, args: u64, part: u8 },
    /// A dispatch of the contact pass's front over the fleet ([`PERF_FLEET_FRONT`]):
    /// `kernel` 0 the thread kernels' module, 1 the plan module, 2 the tests' module, 3 the
    /// throttles', 4 the levels across the GPU's;
    /// parameter copy `copy`; its size at byte `args` of the contact plan's arguments
    /// (`from` 0) or of the fleet's first or second (1, 2). `part` names it: 0 the pairs'
    /// scans, 1 the owners' parts, 2 the triangles and contact points, 3 the grids'
    /// offsets, 4 the sorted copy, 5 the listed triangles' prefix, 6 the walk, 7 the
    /// pairs' prefix, 8 the listed pairs, 9 the throttles, 10 the levels' plan, 11 their
    /// snapshot, 12 a level's hits, 13 its residual phases.
    Fleet { copy: u32, kernel: u8, args: u64, from: u8, part: u8 },
    /// The vehicles' step s (a group's members; with no group, none).
    Member(u32),
    /// Self-collision split: every vehicle's step s.
    SelfMember(u32),
    /// Self-collision split: step `copy`'s walk and pairs over the solo vehicles.
    SelfWalk { copy: u32, offset: u64 },
    SelfPairs { copy: u32, offset: u64 },
    /// Self-collision split with the members' kept pair lists ([`PERF_MEMBER_LIST`]): the
    /// members' rest of a step (their self-collision hits applied) after its walk and
    /// pairs, before the groups' pass; parameter copy `copy` (the step with the kernel's
    /// SPLIT_LEVELS). No workgroups without a group.
    SelfRest { copy: u32 },
    /// The members' raycast wheels on each other before step `copy` (copy steps + 1: the
    /// frame's end, the last step's reactions only), a workgroup a group
    /// ([`Switches::wheels_bodies`]).
    Wheels { copy: u32 },
}

impl GroupOp {
    pub fn label(self) -> &'static str {
        match self {
            GroupOp::Group { part: 0, .. } => "explicit split begin",
            GroupOp::Group { part: 1, .. } => "explicit split front",
            GroupOp::Group { part: 2, .. } => "explicit split tests walk",
            GroupOp::Group { part: 3, .. } => "explicit split tests pairs",
            GroupOp::Group { part: 4, .. } => "explicit split levels",
            GroupOp::Group { part: 6, .. } => "explicit levels' end",
            GroupOp::Group { .. } => "explicit split pass",
            GroupOp::Fleet { part: 0, .. } => "explicit fleet scans",
            GroupOp::Fleet { part: 1, .. } => "explicit fleet owners' parts",
            GroupOp::Fleet { part: 2, .. } => "explicit fleet triangles, points",
            GroupOp::Fleet { part: 3, .. } => "explicit fleet grids",
            GroupOp::Fleet { part: 4, .. } => "explicit fleet sorted copy",
            GroupOp::Fleet { part: 5, .. } => "explicit fleet listed prefix",
            GroupOp::Fleet { part: 6, .. } => "explicit fleet walk",
            GroupOp::Fleet { part: 7, .. } => "explicit fleet pairs' prefix",
            GroupOp::Fleet { part: 8, .. } => "explicit fleet pairs",
            GroupOp::Fleet { part: 10, .. } => "explicit levels' plan",
            GroupOp::Fleet { part: 11, .. } => "explicit levels' snapshot",
            GroupOp::Fleet { part: 12, .. } => "explicit levels' hits",
            GroupOp::Fleet { part: 13, .. } => "explicit levels' residuals",
            GroupOp::Fleet { .. } => "explicit fleet throttles",
            GroupOp::Member(_) => "explicit split member",
            GroupOp::SelfMember(_) => "explicit self step",
            GroupOp::SelfWalk { .. } => "explicit self walk",
            GroupOp::SelfPairs { .. } => "explicit self pairs",
            GroupOp::SelfRest { .. } => "explicit self members' hits",
            GroupOp::Wheels { .. } => "explicit split wheels on cars",
        }
    }
}

/// A dispatch of self-collision split without vehicle contact (selfcol-design.md) or
/// of the wide passes' reused pair list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfDispatch {
    /// The step dispatch of parameter copy `copy` over `groups` workgroups (vehicle sleep
    /// listed: the awake vehicles', from [`SELF_AWAKE_ARGS`]; [`FrameKind::self_awake`]).
    Step { copy: u32, groups: u32 },
    /// The wide passes' list dispatch (explicit_beams_wide_list) of copy `copy`.
    List { copy: u32, groups: u32 },
    /// The walk dispatch of copy `copy`, its size at byte `offset` of the arguments.
    Walk { copy: u32, offset: u64 },
    /// The pairs dispatch of copy `copy`, its size at byte `offset` of the arguments.
    Pairs { copy: u32, offset: u64 },
}

impl SelfDispatch {
    pub fn label(self) -> &'static str {
        match self {
            SelfDispatch::Step { .. } => "explicit self step",
            SelfDispatch::List { .. } => "explicit wide list",
            SelfDispatch::Walk { .. } => "explicit self walk",
            SelfDispatch::Pairs { .. } => "explicit self pairs",
        }
    }

    /// Whether the dispatch binds the arguments buffer as storage (the step dispatch
    /// alone: a buffer cannot be storage and indirect in one dispatch).
    pub fn binds_args(self) -> bool {
        matches!(self, SelfDispatch::Step { .. } | SelfDispatch::List { .. })
    }
}

/// A solo frame of `steps` steps with self-collision split: per step its step dispatch
/// (a workgroup a vehicle), its walk dispatch (indirect: the vehicles that build their
/// pair list) and its pairs dispatch (indirect: the listed pairs of the vehicles that
/// reuse it), then the last step's rest and the frame's end (step dispatch `steps`).
pub fn self_split_schedule(nv: u32, steps: u32) -> Vec<SelfDispatch> {
    let mut out = Vec::new();
    for s in 0..=steps {
        out.push(SelfDispatch::Step { copy: s, groups: nv });
        if s == steps {
            break;
        }
        // Step 0's slot 2, then odd and even steps' (selfArgsAt); the walk's after the
        // pairs' three.
        let slot = u64::from(if s == 0 { 2 } else { s & 1 });
        out.push(SelfDispatch::Walk { copy: s, offset: 12 * (3 + slot) });
        out.push(SelfDispatch::Pairs { copy: s, offset: 12 * slot });
    }
    out
}

/// What a host chose for its frames and what its modules allow.
pub struct Switches<'a> {
    /// The split group path ([`super`]: contact-groups-design.md).
    pub group_split: bool,
    /// Its small groups' triangle tests across workgroups.
    pub group_split_tests: bool,
    /// Self-collision's listed pairs across workgroups (selfcol-design.md).
    pub self_split: bool,
    /// The wide passes with the reused pair list, and with it the aux pass's work fused
    /// onto the list and hits kernels.
    pub wide_list: bool,
    pub wide_fused: bool,
    /// Raycast wheels see the rigid proxies and other cars (the drive header's bit 2,
    /// `engine_softbody::explicit_drive::DriveConfig::see_bodies`): the split group path
    /// runs the members' wheels on each other ([`GroupOp::Wheels`]) and the wide passes'
    /// fused aux work is left out ([`Switches::wide_fused_on`]).
    pub wheels_bodies: bool,
    /// Vehicle contact's candidates in a grid (the owners' state by vehicle needs it).
    pub vv_grid: bool,
    /// The kernels are the current ones (not a test variant).
    pub current: bool,
    /// Measurement: the current kernel's ablation build (self split runs on it).
    pub ablate: bool,
    /// A diagnostic pipeline replaces a contact pass (the frames keep the groups'
    /// workgroup).
    pub debug: bool,
    /// The planner's wide passes may run (false: a test variant's per-vehicle frame).
    pub wide_allowed: bool,
    /// The fleet's frame variant.
    pub aero: bool,
    pub lanes: u32,
    /// Vehicles in the upload.
    pub vehicles: usize,
    /// Level slots the groups' reports ask for ([`PERF_FLEET_LEVELS`]).
    pub level_want: u32,
    /// Measurement only (time; results meaningless): self split's walks (bit 0) and/or
    /// pairs (bit 1) left out.
    pub self_split_skip: u32,
    /// Whether a module is compiled (a browser host compiles them asynchronously; until
    /// then frames keep the paths that do not need it, the same results).
    pub ready: &'a dyn Fn(&str) -> bool,
}

impl Switches<'_> {
    /// The wide passes' fused aux work as frames take it: not while wheels see bodies
    /// (it runs step s + 1's wheels beside step s's proxies).
    pub fn wide_fused_on(&self) -> bool {
        self.wide_fused && !self.wheels_bodies
    }
}

/// The schedules left out while raycast wheels see bodies: they run the wheels' rays or
/// tyres before or beside the step's proxy integration (or, the wide passes', on another
/// workgroup at once), where a ray would meet a body's last pose. A host clears them from
/// its perf flags while [`Switches::wheels_bodies`] (the kernel fails a frame that takes
/// them with the bit set).
pub const WHEELS_AHEAD_FLAGS: u32 =
    PERF_DRIVE_AHEAD | PERF_SPLIT_WHEELS_AHEAD | PERF_AHEAD_APART | PERF_TYRE_AHEAD | PERF_TYRE_EARLY | PERF_WHEELS_APART;

/// The paths a frame runs ([`frame`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameKind {
    /// Self-collision split without contact: the frame's steps (0: not this frame).
    pub self_split: u32,
    /// Self-collision split with vehicle contact (on the split group path).
    pub self_split_vv: bool,
    /// The split group path (with or without self split).
    pub group_split: bool,
    /// The wide passes with the reused pair list: the frame's steps (0: not).
    pub wide_list: u32,
    /// The level slots the parameters carry ([`PERF_FLEET_LEVELS`]).
    pub level_slots: u32,
    /// Self split's step dispatches run over the awake vehicles ([`SELF_AWAKE_ARGS`]).
    pub self_awake: bool,
}

impl FrameKind {
    /// The frame ran a split path (self split, the split group path).
    pub fn split(&self) -> bool {
        self.group_split || self.self_split > 0
    }
}

/// Which launches a host may time as one ([`Launch`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fine {
    /// One launch: its own unit.
    One,
    /// Part of a split path's unit (self split's frame, the split group path): timed
    /// apart only when the host times every dispatch of the split paths.
    Split,
    /// One of a wide beams command's list, walk and pairs: timed apart only by the
    /// wide passes' profile.
    Tail,
}

/// A launch of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// A plan command ([`FramePlanner::commands`]); a copy ends the compute pass before it.
    Cmd(Cmd),
    /// Self-collision split's (no contact) or the wide list's dispatch.
    SelfOne(SelfDispatch),
    /// The split group path's dispatch (self split with contact among them).
    Group(GroupOp),
}

/// One launch and the unit a host times it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub op: Op,
    /// Launches of one unit came from one plan command, or are the split group path's.
    pub unit: u32,
    pub fine: Fine,
    /// The unit's label when it is not its plan command's.
    pub unit_label: Option<&'static str>,
}

/// A frame: its parameter copies (the bytes from binding 0's start a host writes before
/// the launches), its launches in order and the paths it runs.
#[derive(Clone, Debug)]
pub struct Frame {
    pub params: Vec<u8>,
    pub launches: Vec<Launch>,
    pub kind: FrameKind,
}

impl Frame {
    /// The launches as runs between copies: a host's compute passes.
    pub fn ops(&self) -> impl Iterator<Item = &Op> {
        self.launches.iter().map(|l| &l.op)
    }
}

/// An indirect-arguments buffer of the split paths. Each is storage at binding 10 or 11
/// of some launches and the indirect arguments of others; a host never binds one as
/// both in one launch (WebGPU forbids it), hence the spares the fleet's plan module
/// writes its unread half into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArgBuf {
    /// The plan's arguments ([`super::BIND_ARGS`]).
    Plan,
    /// Self-collision split's ([`SELF_ARGS_BYTES`]).
    SelfSplit,
    /// The fleet-wide front's first and second ([`FLEET_ARGS_BYTES`] each).
    Fleet1,
    Fleet2,
    /// Spares for the fleet plan module's other binding.
    Spare1,
    Spare2,
}

/// A launch's workgroups: a count, or three words of an argument buffer at a byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grid {
    Groups(u32),
    Indirect(ArgBuf, u64),
}

/// A split path launch for any host: its module and entry ([`split_modules`]), its
/// parameter copy (binding 0 at `copy * PARAM_STRIDE`), its workgroups, and the
/// argument buffers at bindings 10 and 11 (bindings 1..9 are the plan's).
///
/// [`split_modules`]: super::split_modules
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitLaunch {
    pub module: String,
    pub entry: &'static str,
    pub copy: u32,
    pub grid: Grid,
    pub binding10: Option<ArgBuf>,
    pub binding11: Option<ArgBuf>,
}

/// What a split path [`Op`] launches (`None` for a plan command: [`Kernel::module_for`]).
/// `aero` and `lanes`: the fleet's frame variant; `kind`: the frame's paths.
pub fn split_launch(op: &Op, aero: bool, lanes: u32, kind: &FrameKind) -> Option<SplitLaunch> {
    use super::{
        split_module_name as split, SPLIT_GROUP_ENTRY, SPLIT_MEMBER_ENTRY, SPLIT_TESTS_ENTRY, SPLIT_WHEELS_ENTRY,
        WIDE_LIST_ENTRY,
    };
    let selfs = |vv: bool, i: usize| {
        let (name, entry, _) = self_split_module_names(aero, lanes, vv).swap_remove(i);
        (name, entry)
    };
    let l = |(module, entry): (String, &'static str), copy, grid, b10, b11| SplitLaunch {
        module,
        entry,
        copy,
        grid,
        binding10: b10,
        binding11: b11,
    };
    Some(match *op {
        Op::Cmd(_) => return None,
        Op::SelfOne(SelfDispatch::Step { copy, groups }) => {
            // (Vehicle sleep listed: the awake vehicles only, the awake list's arguments.)
            let grid = if kind.self_awake { Grid::Indirect(ArgBuf::Plan, SELF_AWAKE_ARGS) } else { Grid::Groups(groups) };
            l(selfs(false, 0), copy, grid, Some(ArgBuf::SelfSplit), None)
        }
        Op::SelfOne(SelfDispatch::List { copy, groups }) => l(
            (wide_list_module_name(lanes).to_string(), WIDE_LIST_ENTRY),
            copy,
            Grid::Groups(groups),
            Some(ArgBuf::SelfSplit),
            None,
        ),
        Op::SelfOne(SelfDispatch::Walk { copy, offset }) => l(selfs(false, 1), copy, Grid::Indirect(ArgBuf::SelfSplit, offset), None, None),
        Op::SelfOne(SelfDispatch::Pairs { copy, offset }) => l(selfs(false, 2), copy, Grid::Indirect(ArgBuf::SelfSplit, offset), None, None),
        Op::Group(GroupOp::Group { copy, tests: true, args, .. }) => {
            l((split("tests", aero, lanes), SPLIT_TESTS_ENTRY), copy, Grid::Indirect(ArgBuf::Plan, args), None, None)
        }
        Op::Group(GroupOp::Group { copy, tests: false, args, .. }) => l(
            (split("group", aero, lanes), SPLIT_GROUP_ENTRY),
            copy,
            Grid::Indirect(ArgBuf::Plan, args),
            Some(ArgBuf::Fleet1),
            Some(ArgBuf::Fleet2),
        ),
        Op::Group(GroupOp::Fleet { copy, kernel, args, from, .. }) => {
            let (which, entry) = match kernel {
                0 => ("fleetfront", "explicit_beams_fleet_front"),
                1 => ("fleetplan", "explicit_beams_fleet_plan"),
                3 => ("fleetpost", "explicit_beams_fleet_post"),
                4 => ("fleetlevels", "explicit_beams_fleet_levels"),
                _ => ("tests", SPLIT_TESTS_ENTRY),
            };
            // (The plan kernel writes the arguments after the ones it runs from: the
            // first buffer when run from the contact plan's, the second from the first.)
            let (b10, b11) = match (kernel, from) {
                (1, 0) => (Some(ArgBuf::Fleet1), Some(ArgBuf::Spare1)),
                (1, _) => (Some(ArgBuf::Spare2), Some(ArgBuf::Fleet2)),
                _ => (None, None),
            };
            let src = match from {
                0 => ArgBuf::Plan,
                1 => ArgBuf::Fleet1,
                _ => ArgBuf::Fleet2,
            };
            l((split(which, aero, lanes), entry), copy, Grid::Indirect(src, args), b10, b11)
        }
        // (The groups' module's defines and the tests' bindings, a workgroup a group.)
        Op::Group(GroupOp::Wheels { copy }) => l(
            (split("wheels", aero, lanes), SPLIT_WHEELS_ENTRY),
            copy,
            Grid::Indirect(ArgBuf::Plan, SPLIT_GROUP_ARGS),
            None,
            None,
        ),
        Op::Group(GroupOp::Member(s)) => l(
            (split("member", aero, lanes), SPLIT_MEMBER_ENTRY),
            s,
            Grid::Indirect(ArgBuf::Plan, SPLIT_MEMBER_ARGS),
            None,
            None,
        ),
        Op::Group(GroupOp::SelfMember(s)) => l(selfs(true, 0), s, Grid::Indirect(ArgBuf::Plan, SPLIT_ALL_ARGS), Some(ArgBuf::SelfSplit), None),
        Op::Group(GroupOp::SelfWalk { copy, offset }) => l(selfs(true, 1), copy, Grid::Indirect(ArgBuf::SelfSplit, offset), None, None),
        Op::Group(GroupOp::SelfPairs { copy, offset }) => l(selfs(true, 2), copy, Grid::Indirect(ArgBuf::SelfSplit, offset), None, None),
        // (The vehicles' module again, a workgroup a vehicle when the plan formed a
        // group (none otherwise): a solo vehicle's returns at once.)
        Op::Group(GroupOp::SelfRest { copy }) => {
            l(selfs(true, 0), copy, Grid::Indirect(ArgBuf::Plan, SPLIT_MEMBER_ARGS), Some(ArgBuf::SelfSplit), None)
        }
    })
}

/// The perf flags of a frame on the split group path: vehicle contact's state by owner
/// vehicle ([`PERF_FLEET_STATE`], contact-groups-design.md §8) only with the tests in
/// their own dispatches, the candidates' grid, the pair list and the state's regions (a
/// fleet of two or more); otherwise the groups' slots, the same results. `vv`: the frame
/// runs self-collision split with vehicle contact ([`FrameKind::self_split_vv`]; it
/// decides [`PERF_KEEP_TRACK_REST`]).
pub fn fleet_flags(planner: &FramePlanner, sw: &Switches, vv: bool) -> u32 {
    let w = &planner.words;
    let ok = sw.group_split_tests && sw.vv_grid && w[W_VV_TRI_PAIRS] != 0 && w[W_VV_FLEET] != 0 && w[W_VV_FLEET_NEXT] != 0;
    // (The front as fleet-wide dispatches: with the state by vehicle, once its modules
    // are compiled; a browser host's async compiles.)
    let front = ok
        && planner.perf_flags() & PERF_FLEET_STATE != 0
        && ["fleetfront", "fleetplan", "fleetpost", "fleetlevels"]
            .iter()
            .all(|w| (sw.ready)(&split_module_name(w, sw.aero, sw.lanes)));
    let mut flags = planner.perf_flags();
    if !ok {
        flags &= !PERF_FLEET_STATE;
    }
    if !front {
        // (Groups of any size on the fleet-wide passes: only with the front there.)
        flags &= !(PERF_FLEET_FRONT | PERF_FLEET_ANY | PERF_FLEET_POST | PERF_FLEET_FUSE | PERF_FLEET_HITLAT | PERF_FLEET_KEEP);
    }
    // (The kept candidates, PERF_FLEET_KEEP: the listed triangles' throttle updates
    // across the fleet, PERF_FLEET_POST, take the elements' dispatch, a thread a
    // triangle (a pass that reuses the kept pairs computes no walk, so no prefix of the
    // listed triangles: the kernel's fkPost). The levels across the GPU are not taken
    // with them.)
    if flags & PERF_FLEET_KEEP != 0 {
        flags &= !PERF_FLEET_LEVELS;
    }
    // (The kept candidates' tracking in the members' rest, PERF_KEEP_TRACK_REST: the
    // host's alone, when the frame has the rest dispatches and the fused scans, and no
    // wheels in the rest's dispatch beside the members; else the scans' dispatch.)
    flags &= !PERF_KEEP_TRACK_REST;
    let rest = PERF_FLEET_FRONT | PERF_FLEET_KEEP | PERF_FLEET_FUSE | PERF_MEMBER_LIST;
    if front
        && sw.group_split_tests
        && flags & rest == rest
        && flags & PERF_SPLIT_WHEELS_AHEAD == 0
        && member_rest_on(planner, vv)
    {
        flags |= PERF_KEEP_TRACK_REST;
    }
    // (The levels across the GPU: with groups of any size and the throttles across the
    // fleet, once their module is compiled.)
    let levels = PERF_FLEET_STATE | PERF_FLEET_FRONT | PERF_FLEET_ANY | PERF_FLEET_POST;
    if flags & levels != levels || !(sw.ready)(&split_module_name("fleetlevels", sw.aero, sw.lanes)) {
        flags &= !PERF_FLEET_LEVELS;
    }
    flags
}

/// The level slots of a frame of `steps` steps with the levels across the GPU
/// ([`PERF_FLEET_LEVELS`]; 0: not this frame): as many as the groups' reports ask for,
/// at most [`FLEET_LEVEL_MAX`] and the parameter copies' room (four a slot: a level's
/// application and residual phases; three more: the snapshot, the plan, the end; after
/// the steps'). `RUBBLE_FLEET_LEVEL_SLOTS=n`: n every frame (a measurement, the tests).
/// Levels past them run in their group's workgroup in the end dispatch (the same
/// results).
pub fn fleet_level_slots(planner: &FramePlanner, sw: &Switches, steps: u32) -> u32 {
    if fleet_flags(planner, sw, false) & PERF_FLEET_LEVELS == 0 {
        return 0;
    }
    let room = (PARAM_COPIES + PARAM_LEVEL_COPIES).saturating_sub(3 * steps + 2 + 3) / 4;
    let want = match std::env::var("RUBBLE_FLEET_LEVEL_SLOTS").ok().and_then(|v| v.parse().ok()) {
        Some(n) => n,
        None => sw.level_want,
    };
    room.min(want).min(FLEET_LEVEL_MAX as u32)
}

/// [`fleet_flags`] for a frame of `slots` level slots past the switches
/// ([`FLEET_LEVEL_SLOTS_SHIFT`]; with the levels across the GPU).
pub fn fleet_flags_for(planner: &FramePlanner, sw: &Switches, vv: bool, slots: u32) -> u32 {
    let flags = fleet_flags(planner, sw, vv);
    if flags & PERF_FLEET_LEVELS == 0 {
        return flags;
    }
    flags | slots << FLEET_LEVEL_SLOTS_SHIFT
}

/// Whether the self split's modules (with vehicle contact or not) are compiled.
fn self_split_ready(sw: &Switches, vv: bool) -> bool {
    self_split_module_names(sw.aero, sw.lanes, vv).iter().all(|(m, _, _)| (sw.ready)(m))
}

/// Whether this frame runs the split group path.
pub fn group_split_on(planner: &FramePlanner, sw: &Switches, steps: u32) -> bool {
    sw.group_split
        && planner.contact_active()
        && sw.current
        && !sw.debug
        && steps > 0
        && 3 * steps + 2 <= PARAM_COPIES
        // The tests' dispatch is 64 workgroups a group (at most one group per two
        // vehicles), the vehicles' one a workgroup each: within WebGPU's 65,535 a
        // dimension. A larger fleet keeps the groups' workgroup (the same results).
        && sw.vehicles * 64 <= 65_535
        // (A browser host's async compiles: the groups' workgroup until its modules
        // are compiled, the same results.)
        && ["member", "group", "tests"].iter().all(|w| (sw.ready)(&split_module_name(w, sw.aero, sw.lanes)))
}

/// Whether a frame with self-collision split and vehicle contact runs the members' rest
/// dispatches: with [`PERF_MEMBER_LIST`] a member's step ends at its self-collision
/// tests (the kernel's contactMemberStep), its hits applied in a dispatch of their own
/// after the step's walk and pairs (memberRest).
pub fn member_rest_on(planner: &FramePlanner, vv: bool) -> bool {
    vv && planner.perf_flags() & PERF_MEMBER_LIST != 0
}

/// Whether this frame (its commands) runs self-collision split with vehicle contact.
fn self_split_vv_on(planner: &FramePlanner, sw: &Switches, steps: u32, cmds: &[Cmd]) -> bool {
    sw.self_split
        && planner.contact_active()
        && !planner.has_surface()
        && sw.current
        && !sw.debug
        && steps > 0
        && 3 * steps + 2 <= PARAM_COPIES
        && sw.vehicles * 64 <= 65_535
        && self_split_ready(sw, true)
        && cmds.iter().all(|c| match c {
            Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => {
                matches!(kernel, Kernel::Vv(_) | Kernel::Sleep | Kernel::Solo | Kernel::Contact | Kernel::ContactBig)
            }
            Cmd::Copy { .. } => true,
        })
}

/// Whether this frame (its commands) runs self-collision split.
fn self_split_on(planner: &FramePlanner, sw: &Switches, steps: u32, cmds: &[Cmd]) -> bool {
    sw.self_split
        && !planner.contact_active()
        && !planner.has_surface()
        // (Ablate: measurement only, the step module's ablation build.)
        && (sw.current || sw.ablate)
        && !sw.debug
        && steps > 0
        && steps < PARAM_COPIES
        && self_split_ready(sw, false)
        && cmds.iter().any(|c| matches!(c, Cmd::Dispatch { kernel: Kernel::Frame, .. } | Cmd::Indirect { kernel: Kernel::Frame, .. }))
        && cmds.iter().all(|c| match c {
            // (Vehicle sleep's awake list before the frame, when it is listed.)
            Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => matches!(kernel, Kernel::Frame | Kernel::Sleep),
            Cmd::Copy { .. } => true,
        })
}

/// The list kernel's workgroups for a step of `nv` vehicles: with [`PERF_AHEAD_APART`]
/// (and the fused coupling and wheels) a second workgroup a vehicle for step s + 1's
/// coupling and wheels (the kernel's `wideAheadApart`).
pub fn list_groups(planner: &FramePlanner, sw: &Switches, nv: u32) -> u32 {
    let f = planner.perf_flags();
    if sw.wide_fused_on() && f & PERF_AHEAD_APART != 0 && f & PERF_WHEELS_APART != 0 {
        3 * nv
    } else if sw.wide_fused_on() && f & PERF_AHEAD_APART != 0 {
        2 * nv
    } else {
        nv
    }
}

/// The split group path's dispatches of a frame of `steps` steps, in order (with
/// self-collision split and contact, every vehicle's step, the walk and the pairs in
/// each substep).
pub fn group_split_ops(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, steps: u32) -> Vec<GroupOp> {
    let mut out = vec![GroupOp::Group { copy: steps + 1, tests: false, args: SPLIT_GROUP_ARGS, part: 0 }];
    // With self-collision split: every vehicle's step (copy s, s = 0 ..= steps: the
    // last the solo vehicles' last rest and frame end), each step's walk and pairs
    // over the solo vehicles (explicit_beams_selfsplit.slang), then the groups.
    let vv = kind.self_split_vv;
    let slots = kind.level_slots;
    let fleet = fleet_flags(planner, sw, vv);
    // Raycast wheels on each other (Switches::wheels_bodies): before each step's members,
    // the last step's reactions and this step's rays; after the last step the reactions.
    let wheels = sw.wheels_bodies;
    for s in 0..steps + u32::from(vv) {
        if wheels && s < steps {
            out.push(GroupOp::Wheels { copy: s });
        }
        if vv {
            out.push(GroupOp::SelfMember(s));
            if s == steps {
                break;
            }
            let slot = u64::from(if s == 0 { 2 } else { s & 1 });
            out.push(GroupOp::SelfWalk { copy: s, offset: 12 * (3 + slot) });
            out.push(GroupOp::SelfPairs { copy: s, offset: 12 * slot });
            if member_rest_on(planner, vv) {
                out.push(GroupOp::SelfRest { copy: steps + 2 + s });
            }
        } else {
            out.push(GroupOp::Member(s));
        }
        if sw.group_split_tests && fleet & PERF_FLEET_FRONT != 0 {
            // The front and the tests as fleet-wide dispatches (explicit_beams_fleet.slang),
            // then the groups' levels. The step's three copies are the phases.
            let (c0, ct, cl) = (s, 2 * steps + 2 + s, steps + 2 + s);
            // (A group past VV_PART members: its whole pass in its own workgroup,
            // no workgroups without one; with PERF_FLEET_ANY it is on the fleet-wide
            // passes as every other.)
            if fleet & PERF_FLEET_ANY == 0 {
                out.push(GroupOp::Group { copy: ct, tests: false, args: FLEET_LARGER_ARGS, part: 5 });
            }
            // (PERF_FLEET_FUSE: the scans in the plan's workgroup, the sorted copy in the
            // grids' workgroups.)
            // (With the kept candidates the scans' dispatch stays: it also tracks every
            // node, fkTrack; the kernel's fuseScansOn.)
            // (PERF_KEEP_TRACK_REST: the tracking in the members' rest, the scans fused.)
            let fuse = fleet & PERF_FLEET_FUSE != 0;
            let keep = fleet & (PERF_FLEET_KEEP | PERF_KEEP_TRACK_REST) == PERF_FLEET_KEEP;
            if !fuse || keep {
                out.push(GroupOp::Fleet { copy: c0, kernel: 0, args: FLEET_SCAN_ARGS, from: 0, part: 0 });
            }
            out.push(GroupOp::Fleet { copy: c0, kernel: 1, args: FLEET_FIRST_ARGS, from: 0, part: 1 });
            out.push(GroupOp::Fleet { copy: ct, kernel: 0, args: FLEET_ELEMENT_ARGS, from: 1, part: 2 });
            out.push(GroupOp::Fleet { copy: c0, kernel: 2, args: FLEET_GRID_ARGS, from: 1, part: 3 });
            if !fuse {
                out.push(GroupOp::Fleet { copy: cl, kernel: 0, args: FLEET_SORT_ARGS, from: 1, part: 4 });
            }
            out.push(GroupOp::Fleet { copy: ct, kernel: 1, args: FLEET_ONE_ARGS, from: 1, part: 5 });
            out.push(GroupOp::Fleet { copy: ct, kernel: 2, args: FLEET_WALK_ARGS, from: 2, part: 6 });
            out.push(GroupOp::Fleet { copy: cl, kernel: 1, args: FLEET_ONE_ARGS, from: 1, part: 7 });
            out.push(GroupOp::Fleet { copy: cl, kernel: 2, args: FLEET_PAIRS_ARGS, from: 2, part: 8 });
            // (The listed triangles' throttle updates, a thread each: the walk's size.)
            // (With the kept candidates: a thread a triangle, the elements' size.)
            if fleet & PERF_FLEET_POST != 0 {
                if fleet & PERF_FLEET_KEEP != 0 {
                    out.push(GroupOp::Fleet { copy: cl, kernel: 3, args: FLEET_ELEMENT_ARGS, from: 1, part: 9 });
                } else {
                    out.push(GroupOp::Fleet { copy: cl, kernel: 3, args: FLEET_WALK_ARGS, from: 2, part: 9 });
                }
            }
            out.push(GroupOp::Group { copy: cl, tests: false, args: SPLIT_GROUP_ARGS, part: 4 });
            // The levels across the GPU (explicit_beams_fleet_levels.slang): the plan (one
            // workgroup when a group exists), the snapshot, each slot's application and
            // residual phases (a thread a hit of that level), then the end (a workgroup a
            // group). Their copies after the steps' ([`frame`]).
            if slots > 0 {
                let base = 3 * steps + 2;
                out.push(GroupOp::Fleet { copy: base + 4 * slots + 1, kernel: 1, args: FLEET_FIRST_ARGS, from: 0, part: 10 });
                out.push(GroupOp::Fleet { copy: base + 4 * slots, kernel: 4, args: FLEET_LEVEL_ARGS, from: 1, part: 11 });
                for l in 1..=slots {
                    for phase in 0..4 {
                        let args = FLEET_LEVEL_ARGS + 16 * u64::from(l);
                        out.push(GroupOp::Fleet { copy: base + 4 * (l - 1) + phase, kernel: 4, args, from: 1, part: if phase == 0 { 12 } else { 13 } });
                    }
                }
                out.push(GroupOp::Group { copy: base + 4 * slots + 2, tests: false, args: SPLIT_GROUP_ARGS, part: 6 });
            }
        } else if sw.group_split_tests {
            // The front, the tests across workgroups (the walk, then the listed
            // pairs), then the levels.
            out.push(GroupOp::Group { copy: 2 * steps + 2 + s, tests: false, args: SPLIT_GROUP_ARGS, part: 1 });
            out.push(GroupOp::Group { copy: 2 * steps + 2 + s, tests: true, args: SPLIT_TESTS_ARGS, part: 2 });
            out.push(GroupOp::Group { copy: steps + 2 + s, tests: true, args: SPLIT_TESTS_ARGS, part: 3 });
            out.push(GroupOp::Group { copy: steps + 2 + s, tests: false, args: SPLIT_GROUP_ARGS, part: 4 });
        } else {
            out.push(GroupOp::Group { copy: s, tests: false, args: SPLIT_GROUP_ARGS, part: 5 });
        }
    }
    if wheels {
        out.push(GroupOp::Wheels { copy: steps + 1 });
    }
    out
}

/// A frame of `steps` steps: the planner's parameters and commands
/// ([`FramePlanner::write_params`], [`FramePlanner::commands`]), the split paths this
/// frame runs, the parameter copies they read, and every launch in order.
pub fn frame(planner: &mut FramePlanner, sw: &Switches, steps: u32) -> Frame {
    let mut bytes = planner.write_params(steps);
    let mut cmds = planner.commands(steps, sw.wide_allowed);
    let mut kind = FrameKind { self_awake: planner.sleep_listed(), ..Default::default() };
    let selfs = self_split_on(planner, sw, steps, &cmds);
    kind.self_split = if selfs { steps } else { 0 };
    let vv = !selfs && self_split_vv_on(planner, sw, steps, &cmds);
    kind.self_split_vv = vv;
    let group = vv || group_split_on(planner, sw, steps);
    // Vehicle sleep's awake-indexed dispatch on a split path (a step a dispatch): the
    // kernel's sleepPad0 = 2 in every copy. The awake list's kernel then also wakes
    // the sleeping members of a group with an awake member and writes the split
    // dispatches' arguments (explicit_beams_sleep.slang).
    if planner.sleep_listed() && (selfs || group) {
        planner.words[W_SLEEP + 1] = 2;
        let at = 4 * (W_SLEEP + 1);
        for k in 0..=steps as usize {
            let at = PARAM_STRIDE as usize * k + at;
            bytes[at..at + 4].copy_from_slice(&2u32.to_le_bytes());
        }
    }
    // A wide frame with the reused pair list: every copy (step k: the wide passes, the
    // list kernel, the walk and the pairs) carries PERF_WIDE_LIST.
    let wide = cmds.iter().any(|c| matches!(c, Cmd::Dispatch { kernel: Kernel::Wide(_), .. }));
    let wl = wide
        && sw.wide_list
        && steps > 0
        && !planner.has_surface()
        && sw.current
        && self_split_ready(sw, false)
        && (sw.ready)(wide_list_module_name(sw.lanes));
    kind.wide_list = if wl { steps } else { 0 };
    let params = if wl {
        let stride = PARAM_STRIDE as usize;
        let at = 4 * W_PERF_FLAGS;
        let flags = planner.words[W_PERF_FLAGS];
        for k in 0..=steps as usize {
            let fused = if sw.wide_fused_on() { PERF_WIDE_FUSED } else { 0 };
            bytes[stride * k + at..stride * k + at + 4].copy_from_slice(&(flags | PERF_WIDE_LIST | fused).to_le_bytes());
        }
        bytes
    } else if selfs {
        // A step dispatch reads copy s (s = 0 ..= steps), the pairs dispatch too.
        bytes
    } else if group {
        kind.group_split = true;
        // Copy steps + 1: the groups' frame start (the kernel's SPLIT_BEGIN); with the
        // tests split, copy steps + 2 + s step s's second phase (SPLIT_LEVELS) and copy
        // 2 steps + 2 + s its first group dispatch and walk (SPLIT_TESTS). The
        // vehicles' steps read copy s (the step alone).
        // (The members' rest dispatches read the SPLIT_LEVELS copies too.)
        let rest = member_rest_on(planner, vv);
        let extra = if sw.group_split_tests { 2 * steps + 1 } else if rest { steps + 1 } else { 1 };
        // (The levels across the GPU: after those, four copies a level slot, then the
        // snapshot's, the plan's and the end's, whatever the step.)
        let slots = if sw.group_split_tests { fleet_level_slots(planner, sw, steps) } else { 0 };
        kind.level_slots = slots;
        let lv = if slots > 0 { 4 * slots + 3 } else { 0 };
        bytes.resize((PARAM_STRIDE * u64::from(steps + 1 + extra + lv)) as usize, 0);
        // The owners' state by vehicle only where its dispatches and regions are (the
        // groups' dispatches' copies carry the flags that hold this frame).
        let flags = fleet_flags_for(planner, sw, vv, slots);
        // (The kept candidates' tracking in the members' rest: the steps' copies, which
        // the scans and the first plan read, say so too.)
        if flags & PERF_KEEP_TRACK_REST != 0 {
            let stride = PARAM_STRIDE as usize;
            let at = 4 * W_PERF_FLAGS;
            for k in 0..=steps as usize {
                let w = u32::from_le_bytes(bytes[stride * k + at..stride * k + at + 4].try_into().unwrap());
                bytes[stride * k + at..stride * k + at + 4].copy_from_slice(&(w | PERF_KEEP_TRACK_REST).to_le_bytes());
            }
        }
        let mut put = |k: u32, word: u32| {
            let mut w = planner.words;
            w[W_PERF_FLAGS] = flags;
            w[11] = word;
            let at = (PARAM_STRIDE * u64::from(k)) as usize;
            bytes[at..at + 4 * w.len()].copy_from_slice(bytemuck::cast_slice(&w));
        };
        put(steps + 1, SPLIT_BEGIN);
        for s in 0..steps {
            if sw.group_split_tests || rest {
                put(steps + 2 + s, s | SPLIT_LEVELS);
            }
            if sw.group_split_tests {
                put(2 * steps + 2 + s, s | SPLIT_TESTS);
            }
        }
        if slots > 0 {
            let base = 3 * steps + 2;
            for l in 1..=slots {
                for phase in 0..4 {
                    put(base + 4 * (l - 1) + phase, SPLIT_LV | phase << 8 | l);
                }
            }
            put(base + 4 * slots, SPLIT_LV | LV_SNAP << 8);
            put(base + 4 * slots + 1, SPLIT_LV | LV_PLAN << 8 | slots);
            put(base + 4 * slots + 2, SPLIT_LV | LV_END << 8 | slots);
        }
        bytes
    } else {
        let used = cmds
            .iter()
            .map(|c| match *c {
                Cmd::Dispatch { param_copy, .. } => param_copy,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let n = ((u64::from(used) + 1) * PARAM_STRIDE).min(bytes.len() as u64) as usize;
        bytes.truncate(n);
        bytes
    };
    if group {
        // The solo frame as planned (every vehicle when no group formed, else none),
        // then the split path's dispatches (below) in place of the groups' frames:
        // with a group, every vehicle a step a dispatch. With self-collision split,
        // every vehicle a step a dispatch in every frame (no solo frame).
        cmds.retain(|c| match c {
            Cmd::Indirect { kernel: Kernel::Contact | Kernel::ContactBig, .. } => false,
            Cmd::Indirect { kernel: Kernel::Solo, .. } => !vv,
            _ => true,
        });
    }
    let nv = planner.vehicles();
    let mut launches = Vec::new();
    for (unit, cmd) in cmds.iter().enumerate() {
        let unit = unit as u32;
        let one = |op| Launch { op, unit, fine: Fine::One, unit_label: None };
        let is_frame = matches!(cmd, Cmd::Dispatch { kernel: Kernel::Frame, .. } | Cmd::Indirect { kernel: Kernel::Frame, .. });
        if is_frame && kind.self_split > 0 {
            for d in self_split_schedule(nv, kind.self_split) {
                match d {
                    SelfDispatch::Walk { .. } if sw.self_split_skip & 1 != 0 => continue,
                    SelfDispatch::Pairs { .. } if sw.self_split_skip & 2 != 0 => continue,
                    _ => launches.push(Launch {
                        op: Op::SelfOne(d),
                        unit,
                        fine: Fine::Split,
                        unit_label: Some("explicit self split frame"),
                    }),
                }
            }
            continue;
        }
        // A wide frame with the reused pair list: no triangles' or hits' pass of its own;
        // after a step's beams the list kernel's part, the walk, the pairs and the hits'
        // part; before the frame's end the list's end.
        if let (true, Cmd::Dispatch { kernel: Kernel::Wide(k), param_copy, .. }) = (wl, *cmd) {
            if k == WIDE_TRIS {
                continue;
            }
            // (PERF_WIDE_FUSED: steps 1..'s coupling and wheels ran on the step
            // before's list and hits kernels.)
            if k == WIDE_AUX && sw.wide_fused_on() && param_copy >= 1 && param_copy < steps {
                continue;
            }
            if k == WIDE_END {
                launches.push(one(Op::SelfOne(SelfDispatch::List { copy: steps, groups: nv })));
            }
            launches.push(one(Op::Cmd(*cmd)));
            if k == WIDE_BEAMS_PASS {
                let slot = u64::from(if param_copy == 0 { 2 } else { param_copy & 1 });
                for d in [
                    SelfDispatch::List { copy: param_copy, groups: list_groups(planner, sw, nv) },
                    SelfDispatch::Walk { copy: param_copy, offset: 12 * (3 + slot) },
                    SelfDispatch::Pairs { copy: param_copy, offset: 12 * slot },
                ] {
                    launches.push(Launch { op: Op::SelfOne(d), unit, fine: Fine::Tail, unit_label: None });
                }
            }
            continue;
        }
        launches.push(one(Op::Cmd(*cmd)));
    }
    if group {
        let unit = cmds.len() as u32;
        let label = if vv { "explicit self split groups" } else { "explicit split groups" };
        for op in group_split_ops(planner, sw, &kind, steps) {
            match op {
                GroupOp::SelfWalk { .. } if sw.self_split_skip & 1 != 0 => continue,
                GroupOp::SelfPairs { .. } if sw.self_split_skip & 2 != 0 => continue,
                _ => launches.push(Launch { op: Op::Group(op), unit, fine: Fine::Split, unit_label: Some(label) }),
            }
        }
    }
    Frame { params, launches, kind }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split frame's dispatches for 1, 4 and 64 vehicles, 1 to 34 steps (and the
    /// most a parameter buffer holds): a step dispatch for each parameter copy 0 ..= steps
    /// over every vehicle; between them the walk and pairs dispatches of that step, each
    /// sized from three words inside the arguments buffer, a slot its own for each of the
    /// two dispatches of consecutive steps; no indirect dispatch binds the arguments (wgpu
    /// rejects a buffer used as storage and indirect in one dispatch).
    #[test]
    fn self_split_schedule_is_bounded_and_never_binds_its_arguments_indirectly() {
        for nv in [1u32, 4, 64] {
            for steps in [1u32, 2, 3, 34, PARAM_COPIES - 1] {
                let d = self_split_schedule(nv, steps);
                assert_eq!(d.len() as u32, 3 * steps + 1, "nv {nv}, steps {steps}");
                let mut last: Option<(u64, u64)> = None;
                for (k, x) in d.iter().enumerate() {
                    match *x {
                        SelfDispatch::Step { copy, groups } => {
                            assert_eq!(copy as usize, k / 3);
                            assert_eq!(groups, nv);
                            assert!(copy <= steps && copy < PARAM_COPIES);
                            assert!(x.binds_args());
                        }
                        // (The wide passes' list dispatch: not in this schedule.)
                        SelfDispatch::List { .. } => panic!("a list dispatch in the self split schedule"),
                        SelfDispatch::Walk { copy, offset } | SelfDispatch::Pairs { copy, offset } => {
                            assert!(!x.binds_args(), "an indirect dispatch binds its arguments");
                            assert!(copy < steps);
                            assert_eq!(offset % 4, 0);
                            assert!(offset + 12 <= SELF_ARGS_BYTES, "arguments past the buffer at {offset}");
                        }
                    }
                }
                // The two kinds never share a slot, and consecutive steps never share one
                // (a step dispatch clears the next step's while the last is still read).
                for s in 0..steps as usize {
                    let (SelfDispatch::Walk { offset: w, .. }, SelfDispatch::Pairs { offset: p, .. }) =
                        (d[3 * s + 1], d[3 * s + 2])
                    else {
                        panic!("step {s}: not a walk then a pairs dispatch");
                    };
                    assert_ne!(w, p);
                    if let Some((lw, lp)) = last {
                        assert!(lw != w && lp != p, "steps {} and {s} share a slot", s - 1);
                    }
                    last = Some((w, p));
                }
                assert!(matches!(d.last(), Some(SelfDispatch::Step { copy, .. }) if *copy == steps));
            }
        }
    }
}
