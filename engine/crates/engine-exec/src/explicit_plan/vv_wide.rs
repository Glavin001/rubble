//! A fleet with vehicle contact on the wide passes while the contact plan holds no group
//! (a part of `explicit_plan.rs`; vehicle-perf-first-principles.md §8.57, the kernels in
//! `explicit_beams_vvwide.slang`).
//!
//! With vehicle contact on, a frame whose plan holds no group steps every vehicle alone
//! in a workgroup of its own (the `Solo` frame, or a step a dispatch). The wide passes
//! run the same steps a thread an element across the GPU, as a fleet with contact off
//! does. The plan decides on the GPU, so the frame holds both schedules and a gate
//! ([`VV_GATE`]) sizes them there: the wide launches from its buffer ([`ArgBuf::Wide`]),
//! each a zero count in a frame with a group; today's per-vehicle launches that run
//! every frame from its [`WideSlot::Old`], zero in a frame without one. The same results
//! bit for bit as the path it replaces (`--ab-switch vvwide`).
//!
//! "No group" is no group of two or more members (§8.63): with vehicle contact the plan
//! makes a vehicle without partners a group of its own, whose member step is the solo
//! frame's bit for bit. The plan counts its groups of two or more apart
//! ([`plan_groups_of_two`]), the gate and the host's read go by that count, and in a wide
//! frame the split group path's dispatches that the plan's arguments size launch from
//! the gate's buffer ([`gated_slot`]): the plan's counts with such a group, none without.

use super::schedule::{group_split_ops, surf_args_offset, ArgBuf, FrameKind, Grid, GroupOp, Op, SelfDispatch, SplitLaunch, Switches};
use super::{
    self_split_module_names, solo_rest_module_name, wide_list_module_name, Cmd, FramePlanner, BIND_MISC, BIND_TABLE,
    PERF_AHEAD_APART, PERF_HIT_SLOT_MAP, PERF_WHEELS_APART, PERF_WIDE_FUSED, PERF_WIDE_LIST, SOLO_REST_ENTRY, SURF_FAST_PASSES, VV_GATE,
    VV_WIDE_CLOSE, VV_WIDE_NODES, WIDE_AUX, WIDE_BEAMS_PASS, WIDE_BEGIN, WIDE_CENTRED_BEAMS, WIDE_CENTRED_NODES,
    WIDE_CLOSE, WIDE_END, WIDE_GRID_BUCKETS, WIDE_HITS, WIDE_LIST_ENTRY, WIDE_MAX_VEHICLES, WIDE_NODES, WIDE_PASSES,
    WIDE_REBASE, WIDE_TRIS,
};

/// Bytes of the gate's buffer ([`ArgBuf::Wide`]): a slot of four words (x, 1, 1, unused)
/// for each [`WideSlot`]. (Not state: the gate writes it every wide frame, and no
/// snapshot or fixture holds it.)
pub const WIDE_ARGS_BYTES: u64 = 256;

/// The plan's arguments' word ([`super::BIND_ARGS`], the kernel's vvArgs) holding its
/// count of groups of two or more members plus one; 0 in a plan written before it.
pub const PLAN_GROUPS_OF_TWO_WORD: usize = 15;
/// The plan's arguments' word holding its group count (groups of one included).
pub const PLAN_GROUPS_WORD: usize = 3;

/// The plan's groups of two or more members from its arguments' words (`args`, at least
/// [`PLAN_GROUPS_OF_TWO_WORD`] + 1 of them; the kernel's explicit_vv_gate decides the
/// same way): word 15 less one, or for a plan written before that word (0 there), its
/// group count (a plan then made no group of one).
pub fn plan_groups_of_two(args: &[u32]) -> Option<u32> {
    let two = *args.get(PLAN_GROUPS_OF_TWO_WORD)?;
    if two != 0 {
        Some(two - 1)
    } else {
        args.get(PLAN_GROUPS_WORD).copied()
    }
}

/// A slot of the gate's buffer (the kernel's VW_*): the workgroups of the wide launches it
/// sizes when the plan holds no group (0 with one); [`WideSlot::Old`] the vehicles when
/// the plan holds a group (0 without).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WideSlot {
    /// A workgroup a vehicle: the aux and hits passes, the list kernel, the surface prep.
    Nv = 0,
    /// The list kernel with a second workgroup a vehicle, and with a third.
    Nv2 = 1,
    Nv3 = 2,
    /// A thread a vehicle: begin, close.
    Veh = 3,
    /// A thread a node: the rebase, the centred nodes.
    Nodes = 4,
    /// The node pass: nodes, group slots, grid buckets.
    NodesAll = 5,
    /// A thread a beam: the centred beams.
    Beams = 6,
    /// The beam pass: beams, candidates, triangles.
    BeamsAll = 7,
    /// The end pass: triangles, beams.
    End = 8,
    Old = 9,
    /// The split group path's dispatches the plan's arguments size ([`gated_slot`]): the
    /// plan's counts at its byte offset 64 (the groups'), 80 (the members'), 96 (the
    /// tests'), 128 (the fleet's scans), 144 (its first plan) and 176 (the larger groups')
    /// with a group of two or more, 0 without.
    Groups = 10,
    Members = 11,
    Tests = 12,
    FleetScan = 13,
    FleetFirst = 14,
    FleetLarger = 15,
}

impl WideSlot {
    /// Its byte offset in the gate's buffer.
    pub fn offset(self) -> u64 {
        16 * self as u64
    }
}

/// In a wide frame, the gate's slot that sizes a split group path dispatch the plan's
/// arguments size at byte `offset` (the kernel's VW_GROUPS ..): the plan's count with a
/// group of two or more members, 0 without, when the groups of one are the wide passes'.
/// Panics for an offset no such dispatch reads.
pub fn gated_slot(offset: u64) -> WideSlot {
    use super::schedule::{FLEET_FIRST_ARGS, FLEET_LARGER_ARGS, FLEET_SCAN_ARGS, SPLIT_GROUP_ARGS, SPLIT_MEMBER_ARGS, SPLIT_TESTS_ARGS};
    match offset {
        SPLIT_GROUP_ARGS => WideSlot::Groups,
        SPLIT_MEMBER_ARGS => WideSlot::Members,
        SPLIT_TESTS_ARGS => WideSlot::Tests,
        FLEET_SCAN_ARGS => WideSlot::FleetScan,
        FLEET_FIRST_ARGS => WideSlot::FleetFirst,
        FLEET_LARGER_ARGS => WideSlot::FleetLarger,
        o => panic!("explicit plan: a wide frame's dispatch from the plan's arguments at byte {o} has no gate slot"),
    }
}

/// A launch this schedule adds to the split group path's ([`GroupOp::Wide`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WideOp {
    /// The gate, one thread, on parameter copy `mode`: 0 the slots from the contact
    /// plan's group count; 1 the self split's arguments zeroed, 2 the surface pass's (a
    /// frame's start, where the wide frame's walk, pairs and surface launches are not
    /// today's path's own: in a frame with a group nothing else clears them).
    Gate { mode: u32 },
    /// Wide pass `kind` ([`WIDE_PASSES`] index; the node and close passes: the modules
    /// with the frame-bounds check) on parameter copy `copy`, sized from `slot`.
    Pass { kind: u8, copy: u32, slot: WideSlot },
    /// The list kernel of copy `copy` (the self split's arguments at binding 10).
    List { copy: u32, slot: WideSlot },
    /// The surface pass's prep or plan ([`SURF_FAST_PASSES`] 0 or 2) of step `copy`, a
    /// workgroup a vehicle when the plan holds no group.
    SurfBind { copy: u32, pass: u8 },
    /// [`GroupOp::SelfMember`] and [`GroupOp::SoloRest`] sized from [`WideSlot::Old`]:
    /// every vehicle when the plan holds a group, none otherwise.
    SelfMember(u32),
    SoloRest { copy: u32 },
}

impl WideOp {
    pub fn label(self) -> &'static str {
        match self {
            WideOp::Gate { .. } => "explicit vv gate",
            WideOp::Pass { kind, .. } => WIDE_PASSES[kind as usize],
            WideOp::List { .. } => "explicit wide list",
            WideOp::SurfBind { pass, .. } => SURF_FAST_PASSES[pass as usize].0,
            WideOp::SelfMember(_) => "explicit self step",
            WideOp::SoloRest { .. } => "explicit self solo hits",
        }
    }
}

impl FramePlanner {
    /// A fleet with vehicle contact runs the frames whose plan holds no group as the wide
    /// passes (on), or a workgroup a vehicle as before (off, the default here; a host
    /// turns it on, [`vv_wide_default`]). The same results bit for bit. It acts only
    /// where the wide passes could run the fleet but for its contact
    /// ([`Self::vv_wide`]).
    pub fn set_vv_wide(&mut self, on: bool) {
        self.vv_wide = on;
    }

    /// [`Self::set_vv_wide`] is on and can act: vehicle contact active, the wide passes
    /// possible and chosen ([`Self::set_wide`]; a fleet with surface contact: its pass as
    /// pipelines), no vehicle sleep (a sleeping vehicle's step count stands, and the wide
    /// passes need one step parity for the fleet).
    pub fn vv_wide(&self) -> bool {
        self.vv_wide && self.contact_active() && self.wide_ok() && self.wide_on && !self.sleep_on()
    }

    /// The contact plan's groups of two or more members as the host last read them back
    /// ([`plan_groups_of_two`]; `None`: not yet), a hint only: [`Self::vv_wide`]'s
    /// frames hold both schedules, a launch each, and a launch costs a host and a driver microseconds whether or not the GPU gives it a
    /// workgroup, so a frame takes them only after a plan without a group was seen.
    /// Until then, and after a plan with a group, it is today's frame, which serves a
    /// plan without a group as before. Either frame is right for either plan, so a
    /// stale or missing hint costs time and never changes a result; a host reads the
    /// count without waiting (the plan's arguments, words 3 and 15, after a frame).
    pub fn set_vv_wide_seen(&mut self, groups: Option<u32>) {
        self.vv_seen = groups;
    }

    /// The gate's slots when the plan holds no group (the kernel's explicit_vv_gate):
    /// [`Self::wide_plan`]'s workgroup counts by [`WideSlot`]; [`WideSlot::Old`] 0.
    pub fn vv_wide_counts(&self) -> [u32; 10] {
        let groups = |n: u32| n.div_ceil(256).max(1);
        let nv = self.nv;
        let [nodes, beams, tris, cands, slots] = self.total;
        [
            nv,
            2 * nv,
            3 * nv,
            groups(nv),
            groups(nodes),
            groups(nodes + slots + nv * WIDE_GRID_BUCKETS as u32),
            groups(beams),
            groups(beams + cands + tris),
            groups(tris + beams),
            0,
        ]
    }
}

/// [`FramePlanner::set_vv_wide`]'s default: `RUBBLE_VV_WIDE=1` on, `0` off; otherwise the
/// host's own (`host`).
pub fn vv_wide_default(host: bool) -> bool {
    match std::env::var("RUBBLE_VV_WIDE").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => host,
    }
}

/// The modules this schedule launches beside the split group path's own.
fn modules(sw: &Switches, vv: bool, surface: bool) -> Vec<String> {
    let mut out: Vec<String> = [VV_GATE.0, VV_WIDE_NODES.0, VV_WIDE_CLOSE.0, wide_list_module_name(sw.lanes)].map(String::from).to_vec();
    out.extend(WIDE_PASSES.iter().map(|m| m.to_string()));
    if !vv {
        // (The wide frame's own walk and pairs, and its surface pipelines.)
        out.extend(self_split_module_names(sw.aero, sw.lanes, false).into_iter().skip(1).map(|(m, _, _)| m));
        if surface {
            out.extend(SURF_FAST_PASSES.iter().map(|(m, _)| m.to_string()));
        }
    }
    out
}

/// Whether a frame on the split group path (`vv`: with self-collision split) runs this
/// schedule: [`FramePlanner::vv_wide`], the last plan the host saw held no group
/// ([`FramePlanner::set_vv_wide_seen`]), the host's wide passes with the reused pair list
/// and the tests split, no wheels on each other (their dispatch is a group's), at least
/// the gate's three parameter copies, and every module compiled.
pub(super) fn on(planner: &FramePlanner, sw: &Switches, steps: u32, vv: bool) -> bool {
    planner.vv_wide()
        && planner.vv_seen == Some(0)
        && sw.wide_allowed
        && sw.wide_list
        && sw.group_split_tests
        && sw.current
        && !sw.debug
        && !sw.wheels_bodies
        && steps >= 2
        && modules(sw, vv, planner.has_surface()).iter().all(|m| (sw.ready)(m))
}

/// The perf flags `flags` of a step's parameter copy on the split group path. Today's
/// frame: without [`PERF_HIT_SLOT_MAP`] (on that path bit 27 is the level slots', which
/// the groups' pass reads from these copies when its tests are not split). With the
/// wide frame for a plan without a group (`kind.vv_wide`): its passes read these copies
/// too, so with the reused pair list's flag, the aux work fused where a wide frame takes
/// it, and the map (the tests are split there, and no kernel of the split group path
/// reads the two wide flags).
pub(super) fn step_flags(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, flags: u32) -> u32 {
    if !kind.vv_wide {
        return flags & !PERF_HIT_SLOT_MAP;
    }
    flags | PERF_WIDE_LIST | if sw.wide_fused_on(planner) { PERF_WIDE_FUSED } else { 0 }
}

/// The slot that sizes wide pass `kind`.
fn pass_slot(kind: usize) -> WideSlot {
    match kind {
        WIDE_BEGIN | WIDE_CLOSE => WideSlot::Veh,
        WIDE_REBASE | WIDE_CENTRED_NODES => WideSlot::Nodes,
        WIDE_NODES => WideSlot::NodesAll,
        WIDE_CENTRED_BEAMS => WideSlot::Beams,
        WIDE_BEAMS_PASS => WideSlot::BeamsAll,
        WIDE_END => WideSlot::End,
        WIDE_HITS | WIDE_AUX => WideSlot::Nv,
        _ => unreachable!("wide pass {kind} has no slot"),
    }
}

/// The split group path's launches of a frame, after the plan's commands: today's
/// ([`group_split_ops`]), or with `kind.vv_wide` the gate, then those with the wide
/// frame's between them, in the order of a wide frame with the reused pair list
/// ([`super::schedule::frame`]).
///
/// With self-collision split (`kind.self_split_vv`) the walk, the pairs and the surface
/// pipelines are that path's own dispatches and one launch of each serves both
/// schedules: a step is the wide passes up to the list kernel, the vehicles' step (sized
/// from [`WideSlot::Old`]), the walk, the pairs, the wide hits, then the rests, the
/// surface pipelines and the groups' passes. Without it the wide frame brings its own
/// walk, pairs and surface pipelines ahead of each step's members' dispatch.
pub(super) fn ops(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, steps: u32) -> Vec<Op> {
    if !kind.vv_wide {
        return group_split_ops(planner, sw, kind, steps).into_iter().map(Op::Group).collect();
    }
    let nv = planner.vehicles();
    let vv = kind.self_split_vv;
    let fused = sw.wide_fused_on(planner);
    let surface = planner.has_surface();
    let w = |o: WideOp| Op::Group(GroupOp::Wide(o));
    let list_slot = match planner.perf_flags() & (PERF_AHEAD_APART | PERF_WHEELS_APART) {
        f if fused && f == PERF_AHEAD_APART | PERF_WHEELS_APART => WideSlot::Nv3,
        f if fused && f & PERF_AHEAD_APART != 0 => WideSlot::Nv2,
        _ => WideSlot::Nv,
    };
    let mut start = vec![w(WideOp::Gate { mode: 0 })];
    if !vv {
        start.push(w(WideOp::Gate { mode: 1 }));
        if surface {
            start.push(w(WideOp::Gate { mode: 2 }));
        }
    }
    let mut pre: Vec<Vec<Op>> = vec![Vec::new(); steps as usize];
    let mut hits: Vec<Vec<Op>> = vec![Vec::new(); steps as usize];
    let mut end = Vec::new();
    for (k, pass, _) in planner.wide_plan(steps) {
        if pass == WIDE_TRIS {
            continue;
        }
        let op = w(WideOp::Pass { kind: pass as u8, copy: k, slot: pass_slot(pass) });
        match pass {
            WIDE_BEGIN | WIDE_REBASE => start.push(op),
            WIDE_AUX if k == steps => end.push(op),
            // (The aux work fused: steps 1..'s ran on the step before's list and hits kernels.)
            WIDE_AUX if fused && k >= 1 => {}
            WIDE_END => {
                end.push(w(WideOp::List { copy: steps, slot: WideSlot::Nv }));
                end.push(op);
            }
            WIDE_CLOSE => end.push(op),
            WIDE_BEAMS_PASS => {
                let at = &mut pre[k as usize];
                at.push(op);
                at.push(w(WideOp::List { copy: k, slot: list_slot }));
                if !vv {
                    let slot = u64::from(if k == 0 { 2 } else { k & 1 });
                    at.push(Op::SelfOne(SelfDispatch::Walk { copy: k, offset: 12 * (3 + slot) }));
                    at.push(Op::SelfOne(SelfDispatch::Pairs { copy: k, offset: 12 * slot }));
                }
            }
            WIDE_HITS => {
                let at = &mut hits[k as usize];
                at.push(op);
                if !vv && surface {
                    at.extend((0..SURF_FAST_PASSES.len() as u8).map(|pass| match surf_args_offset(k, pass) {
                        None => w(WideOp::SurfBind { copy: k, pass }),
                        Some(_) => Op::Group(GroupOp::Surf { copy: k, pass, groups: nv }),
                    }));
                }
            }
            _ => pre[k as usize].push(op),
        }
    }
    let mut out = start;
    // (Begin and rebase, then the vehicles' origins and steps into the table: a wide
    // frame's copy, [`FramePlanner::commands`].)
    out.push(Op::Cmd(Cmd::Copy {
        src: BIND_MISC,
        src_offset: 0,
        dst: BIND_TABLE,
        dst_offset: (WIDE_MAX_VEHICLES * 4 * 16) as u64,
        size: u64::from(nv) * 16,
    }));
    for g in group_split_ops(planner, sw, kind, steps) {
        match g {
            GroupOp::SelfMember(s) => {
                out.append(if s < steps { &mut pre[s as usize] } else { &mut end });
                out.push(w(WideOp::SelfMember(s)));
            }
            GroupOp::SelfPairs { copy, .. } => {
                out.push(Op::Group(g));
                out.append(&mut hits[copy as usize]);
            }
            GroupOp::SoloRest { copy } => out.push(w(WideOp::SoloRest { copy })),
            GroupOp::Member(s) => {
                out.append(&mut pre[s as usize]);
                out.append(&mut hits[s as usize]);
                out.push(Op::Group(g));
            }
            other => out.push(Op::Group(other)),
        }
    }
    out.append(&mut end);
    debug_assert!(pre.iter().chain(&hits).all(Vec::is_empty), "a wide step without its place in the schedule");
    out
}

/// The (module, entry) of [`WideOp::Pass`] `kind`: the wide pass's own, or for the node
/// and close passes the ones with the frame-bounds check.
pub fn pass_module(kind: u8) -> (&'static str, &'static str) {
    match kind as usize {
        WIDE_NODES => VV_WIDE_NODES,
        WIDE_CLOSE => VV_WIDE_CLOSE,
        k => (WIDE_PASSES[k], WIDE_PASSES[k]),
    }
}

/// What a [`WideOp`] launches ([`super::schedule::split_launch`]).
pub(super) fn launch(op: &WideOp, aero: bool, lanes: u32) -> SplitLaunch {
    let l = |(module, entry): (&str, &'static str), copy, slot: WideSlot, b10| SplitLaunch {
        module: module.to_string(),
        entry,
        copy,
        grid: Grid::Indirect(ArgBuf::Wide, slot.offset()),
        binding10: b10,
        binding11: None,
    };
    match *op {
        WideOp::Gate { mode } => SplitLaunch {
            module: VV_GATE.0.to_string(),
            entry: VV_GATE.1,
            copy: mode,
            grid: Grid::Groups(1),
            binding10: Some([ArgBuf::Plan, ArgBuf::SelfSplit, ArgBuf::Surf][mode as usize]),
            binding11: Some(ArgBuf::Wide),
        },
        WideOp::Pass { kind, copy, slot } => l(pass_module(kind), copy, slot, None),
        WideOp::List { copy, slot } => l((wide_list_module_name(lanes), WIDE_LIST_ENTRY), copy, slot, Some(ArgBuf::SelfSplit)),
        WideOp::SurfBind { copy, pass } => l(SURF_FAST_PASSES[pass as usize], copy, WideSlot::Nv, Some(ArgBuf::Surf)),
        WideOp::SelfMember(s) => {
            let (name, entry, _) = self_split_module_names(aero, lanes, true).swap_remove(0);
            l((&name, entry), s, WideSlot::Old, Some(ArgBuf::SelfSplit))
        }
        WideOp::SoloRest { copy } => l((&solo_rest_module_name(aero, lanes, true), SOLO_REST_ENTRY), copy, WideSlot::Old, None),
    }
}

#[cfg(test)]
mod tests {
    use super::super::schedule::{frame, split_launch};
    use super::super::{ExplicitUpload, Kernel, W_PERF_FLAGS};
    use super::*;

    fn cars(n: usize, surface: bool) -> Vec<engine_softbody::explicit_beams::ExplicitBeams> {
        use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
        let text = "nodes\n0, 0.0, 0.0, 0.0\n1, 0.3, 0.0, 0.0\n2, 0.0, 0.3, 0.0\nbeams\n0, 1\n1, 2\n2, 0\nend\n";
        (0..n)
            .map(|k| {
                let mut v = ExplicitBeams::from_truck(text, glam::Vec3::new(5.0 * k as f32, 1.0, 0.0), glam::Vec3::ZERO, ExplicitBeamsSettings::default())
                    .expect("the truck parses");
                v.surface_contact = surface;
                v
            })
            .collect()
    }

    fn switches(ready: &dyn Fn(&str) -> bool, vehicles: usize, lanes: u32) -> Switches<'_> {
        Switches {
            group_split: true,
            group_split_tests: true,
            self_split: true,
            wide_list: true,
            wide_fused: true,
            wheels_bodies: false,
            vv_grid: true,
            group_apart: true,
            current: true,
            ablate: false,
            debug: false,
            wide_allowed: true,
            aero: false,
            lanes,
            vehicles,
            level_want: 0,
            self_split_skip: 0,
            ready,
        }
    }

    /// Every wide launch's slot holds the workgroups the wide plan gives that pass, and
    /// sits inside the gate's buffer.
    #[test]
    fn the_gates_slots_are_the_wide_plans_counts() {
        for surface in [false, true] {
            let up = ExplicitUpload::new(&cars(3, surface), None);
            let mut planner = FramePlanner::new(&up.layout);
            planner.set_surface_fast(true);
            planner.set_wide(true);
            let counts = planner.vv_wide_counts();
            for (_, pass, groups) in planner.wide_plan(6) {
                if pass == WIDE_TRIS {
                    continue;
                }
                let slot = pass_slot(pass);
                assert_eq!(counts[slot as usize], groups, "pass {pass}");
                assert!(slot.offset() + 12 <= WIDE_ARGS_BYTES);
            }
            assert!(WideSlot::FleetLarger.offset() + 16 == WIDE_ARGS_BYTES);
        }
    }

    /// The host's read of the plan's groups of two or more: word 15 less one; a plan
    /// before that word (0 there) by its group count, word 3.
    #[test]
    fn the_hosts_read_counts_groups_of_two_or_more() {
        let mut args = [0u32; 48];
        args[3] = 4;
        assert_eq!(plan_groups_of_two(&args), Some(4));
        args[15] = 1;
        assert_eq!(plan_groups_of_two(&args), Some(0));
        args[15] = 3;
        assert_eq!(plan_groups_of_two(&args), Some(2));
        assert_eq!(plan_groups_of_two(&args[..15]), None);
    }

    /// Cars with vehicle contact and self-collision split: off, today's launches; on, the
    /// gate first, each step's wide passes ahead of the vehicles' step (now sized by the
    /// gate), the wide hits after the step's pairs, the wide frame's end before the last
    /// vehicles' dispatch; every launch of the split group path still there, in order;
    /// no Solo frame; the steps' copies carry the reused pair list's flag.
    #[test]
    fn contact_cars_with_self_split_get_the_wide_frame_between_todays_launches() {
        let up = ExplicitUpload::new(&cars(3, false), None);
        let ready = |_: &str| true;
        let sw = switches(&ready, 3, up.layout.lanes);
        let steps = 4;
        let mut planner = FramePlanner::new(&up.layout);
        planner.set_wide(true);
        let off = frame(&mut planner, &sw, steps);
        assert!(off.kind.self_split_vv && !off.kind.vv_wide);
        planner.set_vv_wide(true);
        // (Not before a plan without a group was seen, nor after one with a group.)
        assert!(!frame(&mut planner, &sw, steps).kind.vv_wide);
        planner.set_vv_wide_seen(Some(2));
        assert!(!frame(&mut planner, &sw, steps).kind.vv_wide);
        planner.set_vv_wide_seen(Some(0));
        let on = frame(&mut planner, &sw, steps);
        assert!(on.kind.self_split_vv && on.kind.vv_wide);
        let ops: Vec<Op> = on.ops().copied().collect();
        let wide = |o: &Op| if let Op::Group(GroupOp::Wide(x)) = o { Some(*x) } else { None };
        let at = |want: WideOp| ops.iter().position(|o| wide(o) == Some(want)).unwrap_or_else(|| panic!("no {want:?}"));
        let gate = at(WideOp::Gate { mode: 0 });
        assert!(ops[..gate].iter().all(|o| matches!(o, Op::Cmd(Cmd::Dispatch { kernel: Kernel::Vv(_), .. }))), "the gate after the plan's passes");
        assert!(ops.iter().all(|o| wide(o).is_none_or(|x| !matches!(x, WideOp::Gate { mode: 1 | 2 }))), "self split's own dispatches clear their slots");
        assert!(ops.iter().all(|o| !matches!(o, Op::Cmd(Cmd::Indirect { kernel: Kernel::Solo, .. }))));
        let copy = ops.iter().position(|o| matches!(o, Op::Cmd(Cmd::Copy { dst: BIND_TABLE, .. }))).expect("the table's copy");
        assert!(at(WideOp::Pass { kind: WIDE_REBASE as u8, copy: 0, slot: WideSlot::Nodes }) < copy);
        for s in 0..steps {
            let nodes = at(WideOp::Pass { kind: WIDE_NODES as u8, copy: s, slot: WideSlot::NodesAll });
            let beams = at(WideOp::Pass { kind: WIDE_BEAMS_PASS as u8, copy: s, slot: WideSlot::BeamsAll });
            let list = ops.iter().position(|o| matches!(wide(o), Some(WideOp::List { copy, .. }) if copy == s)).expect("the list");
            let member = at(WideOp::SelfMember(s));
            let pairs = ops.iter().position(|o| matches!(o, Op::Group(GroupOp::SelfPairs { copy, .. }) if *copy == s)).expect("the pairs");
            let hit = at(WideOp::Pass { kind: WIDE_HITS as u8, copy: s, slot: WideSlot::Nv });
            assert!(copy < nodes && nodes < beams && beams < list && list < member && member < pairs && pairs + 1 == hit, "step {s}");
        }
        let last = at(WideOp::SelfMember(steps));
        assert_eq!(last + 1, ops.len());
        assert!(at(WideOp::Pass { kind: WIDE_CLOSE as u8, copy: steps, slot: WideSlot::Veh }) + 1 == last);
        // Today's launches, in order, with the two that run every frame resized.
        let old: Vec<GroupOp> = off.ops().filter_map(|o| if let Op::Group(g) = o { Some(*g) } else { None }).collect();
        let kept: Vec<GroupOp> = ops
            .iter()
            .filter_map(|o| match o {
                Op::Group(GroupOp::Wide(WideOp::SelfMember(s))) => Some(GroupOp::SelfMember(*s)),
                Op::Group(GroupOp::Wide(WideOp::SoloRest { copy })) => Some(GroupOp::SoloRest { copy: *copy }),
                Op::Group(GroupOp::Wide(_)) => None,
                Op::Group(g) => Some(*g),
                _ => None,
            })
            .collect();
        assert_eq!(kept, old);
        let l = split_launch(&ops[last], false, up.layout.lanes, &on.kind).unwrap();
        assert_eq!(l.grid, Grid::Indirect(ArgBuf::Wide, WideSlot::Old.offset()));
        // No launch of the wide frame is sized from the plan's arguments: the gate's slots
        // size those (none when the plan's groups are all of one); today's frame keeps them.
        let sized = |f: &super::super::schedule::Frame, buf: ArgBuf| {
            f.ops().filter_map(|o| split_launch(o, false, up.layout.lanes, &f.kind)).filter(|l| matches!(l.grid, Grid::Indirect(b, _) if b == buf)).count()
        };
        assert_eq!(sized(&on, ArgBuf::Plan), 0);
        assert!(sized(&off, ArgBuf::Plan) > 0);
        for o in on.ops() {
            if let (Op::Group(GroupOp::Group { args, .. } | GroupOp::Fleet { from: 0, args, .. }), Some(l)) = (o, split_launch(o, false, up.layout.lanes, &on.kind)) {
                assert_eq!(l.grid, Grid::Indirect(ArgBuf::Wide, gated_slot(*args).offset()), "{o:?}");
            }
        }
        for k in 0..=steps as usize {
            let at = 512 * k + 4 * W_PERF_FLAGS;
            let flags = u32::from_le_bytes(on.params[at..at + 4].try_into().unwrap());
            assert_ne!(flags & PERF_WIDE_LIST, 0, "copy {k}");
        }
        // Until its modules are compiled: today's frame.
        let late = |m: &str| m != VV_GATE.0;
        assert!(!frame(&mut planner, &switches(&late, 3, up.layout.lanes), steps).kind.vv_wide);
    }

    /// Cars with surface contact, its pass in the kernels with vehicle contact (the wgpu
    /// host's default): on, the Solo frame's place is the wide frame's, with its own
    /// walk, pairs and surface pipelines (the prep and the plan sized by the gate) ahead
    /// of each step's members' dispatch, and the self split's and the surface pass's
    /// arguments zeroed at the frame's start; no launch binds the buffer it is sized from.
    #[test]
    fn surface_cars_get_the_wide_frame_with_its_own_walk_pairs_and_pipelines() {
        let up = ExplicitUpload::new(&cars(3, true), None);
        let ready = |_: &str| true;
        let sw = switches(&ready, 3, up.layout.lanes);
        let steps = 4;
        let mut planner = FramePlanner::new(&up.layout);
        planner.set_surface_fast(true);
        planner.set_surface_fast_contact(false);
        planner.set_wide(true);
        planner.set_vv_wide(true);
        planner.set_vv_wide_seen(Some(0));
        let on = frame(&mut planner, &sw, steps);
        assert!(!on.kind.self_split_vv && on.kind.group_split && on.kind.vv_wide);
        let ops: Vec<Op> = on.ops().copied().collect();
        let wide = |o: &Op| if let Op::Group(GroupOp::Wide(x)) = o { Some(*x) } else { None };
        for mode in 0..3 {
            assert!(ops.iter().any(|o| wide(o) == Some(WideOp::Gate { mode })), "gate {mode}");
        }
        assert!(ops.iter().all(|o| !matches!(o, Op::Cmd(Cmd::Indirect { kernel: Kernel::Solo, .. }))));
        for s in 0..steps {
            let member = ops.iter().position(|o| *o == Op::Group(GroupOp::Member(s))).expect("the members' step");
            let hit = ops.iter().position(|o| wide(o) == Some(WideOp::Pass { kind: WIDE_HITS as u8, copy: s, slot: WideSlot::Nv })).expect("the hits");
            let walk = ops.iter().position(|o| matches!(o, Op::SelfOne(SelfDispatch::Walk { copy, .. }) if *copy == s)).expect("the walk");
            assert!(walk < hit && hit + SURF_FAST_PASSES.len() < member, "step {s}");
            for j in 0..SURF_FAST_PASSES.len() {
                let l = split_launch(&ops[hit + 1 + j], false, up.layout.lanes, &on.kind).unwrap();
                assert_eq!((l.module.as_str(), l.copy), (SURF_FAST_PASSES[j].0, s));
                match surf_args_offset(s, j as u8) {
                    None => assert_eq!((l.grid, l.binding10), (Grid::Indirect(ArgBuf::Wide, 0), Some(ArgBuf::Surf))),
                    Some(at) => assert_eq!((l.grid, l.binding10), (Grid::Indirect(ArgBuf::Surf, at), None)),
                }
            }
        }
        for o in &ops {
            let Some(l) = split_launch(o, false, up.layout.lanes, &on.kind) else { continue };
            if let Grid::Indirect(buf, _) = l.grid {
                assert!(l.binding10 != Some(buf) && l.binding11 != Some(buf), "{o:?} binds the buffer it is sized from");
            }
            assert!(l.binding10.is_none() || l.binding10 != l.binding11, "{o:?} binds one buffer twice");
        }
    }
}
