//! Continuous contact's place in the split group path (`docs/continuous-contact-design.md`,
//! "The GPU port"): each step's members' dispatch cut in two at the node update (the kernel's
//! `SPLIT_CC_NODES` and `SPLIT_CC_REST`, `explicit_beams_split.slang`), the continuous passes
//! between. The two halves read parameter copies of their own after every other copy of the
//! frame: step s's copy, its step word with the half's bit.

use super::schedule::{member_rest_on, FrameKind, GroupOp, Switches};
use super::{FramePlanner, PARAM_STRIDE};

/// The step word's bit for the node update's half, and the rest's; both: the continuous
/// pass between them (in the edge pass's module and entry, a workgroup a group).
pub const SPLIT_CC_NODES: u32 = 0x0800_0000;
pub const SPLIT_CC_REST: u32 = 0x0400_0000;
/// Parameter copies a step: the node update's half, the continuous pass, the rest's half.
const PER_STEP: u32 = 3;
/// The parameter buffer's room for them, past its other copies ([`super::params_bytes`]).
pub const PARAM_CC_COPIES: u32 = PER_STEP * (super::PARAM_COPIES - 1);

/// Whether this frame's members run in two halves.
pub fn on(planner: &FramePlanner, kind: &FrameKind) -> bool {
    planner.continuous && kind.group_split && !kind.self_split_vv
}

/// The first of the halves' copies ([`PER_STEP`] a step): after the frame's others
/// ([`super::schedule::frame`]'s copies: steps + 1 + its extra and level copies).
pub fn first_copy(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, steps: u32) -> u32 {
    let extra = if sw.group_split_tests {
        2 * steps + 1
    } else if member_rest_on(planner, kind.self_split_vv) {
        steps + 1
    } else {
        1
    };
    let slots = kind.level_slots;
    steps + 1 + extra + if slots > 0 { 4 * slots + 3 } else { 0 }
}

/// Step s's members: the whole step, or with continuous contact its two halves.
pub fn member_ops(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, steps: u32, s: u32, out: &mut Vec<GroupOp>) {
    if !on(planner, kind) {
        out.push(GroupOp::Member(s));
        return;
    }
    let c = first_copy(planner, sw, kind, steps) + PER_STEP * s;
    out.push(GroupOp::Member(c));
    out.push(GroupOp::Edge { copy: c + 1 });
    out.push(GroupOp::Member(c + 2));
}

/// The halves' parameter copies, appended: step s's copy with its step word's half bit.
pub fn copies(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, steps: u32, bytes: &mut Vec<u8>) {
    if !on(planner, kind) {
        return;
    }
    let stride = PARAM_STRIDE as usize;
    let first = first_copy(planner, sw, kind, steps) as usize;
    let per = PER_STEP as usize;
    bytes.resize(stride * (first + per * steps as usize), 0);
    for s in 0..steps as usize {
        for (h, bit) in [SPLIT_CC_NODES, SPLIT_CC_NODES | SPLIT_CC_REST, SPLIT_CC_REST].into_iter().enumerate() {
            let (from, to) = (stride * s, stride * (first + per * s + h));
            bytes.copy_within(from..from + stride, to);
            let at = to + 4 * 11;
            let w = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) | bit;
            bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
}

/// Continuous contact's tables (the kernels' `gP.cc`; 0: none; explicit_beams_split_cc.slang).
/// gStatic at the base: (the per-node scratch's first float4 in gBeams, [`CC_BEAMS_NODE`] a
/// node: the step's start position (the member's snapshot, relative), then world x1, its
/// value before the pass, the velocity, its value before; the per-node words' first in gAcc;
/// the counters' first in gAcc; vehicles); (the surface edges' first record in gStatic,
/// their count, r_q's bits, the own clearance's bits); per vehicle three records: (its first
/// edge, its edges, its pair list's first word in gNext, its capacity in pairs), (its grid's
/// entries' first word in gNext, their capacity, its spawn exclusions' first entry in gLists,
/// their count) and (its grid's bucket counts and cursors' first word in gAcc, 0, 0, 0); per edge (global i, global j, its triangles' first entry in gLists, their
/// count), the entries the global triangles; a spawn exclusion four global nodes (a point and
/// a triangle, or two edges). gAcc per node [`CC_ACC_NODE`] words (velocity sums xyz, position
/// sums xyz, active pairs, zone label), then per vehicle its pairs' count, then
/// [`CC_COUNTERS`]. A grid: in gAcc [`CC_GRID_BUCKETS`] starts and as many cursors; in gNext
/// two words an entry (primitive | kind << 31, packed cell).
/// (Word W_CC + 1: the per-vehicle pair counts' first word in gAcc, then the counters.)
pub const W_CC: usize = 109;
pub const CC_BEAMS_NODE: usize = 5;
pub const CC_ACC_NODE: usize = 8;
pub const CC_COUNTERS: usize = 16;
pub const CC_PAIR_WORDS: usize = 6;
pub const CC_GRID_BUCKETS: usize = 1024;
/// Pairs a surface node or edge may list on average (the list's capacity; past it the frame
/// fails with an error, never drops a pair).
pub const CC_PAIRS_PER_ITEM: usize = 32;
/// The own clearance (m): `engine_softbody::explicit_contact::Dat::self_gap`'s default.
pub const CC_SELF_GAP: f32 = 0.002;

pub(super) fn cc_tables(
    vehicles: &[engine_softbody::explicit_beams::ExplicitBeams],
    statics: &mut Vec<[u32; 4]>,
    lists: &mut Vec<u32>,
    beams: &mut Vec<[f32; 4]>,
    next_init: &mut Vec<u32>,
    acc_init: &mut Vec<i32>,
) -> (u32, u32) {
    let nv = vehicles.len();
    let total: usize = vehicles.iter().map(|v| v.nodes.len()).sum();
    // Each vehicle's collision triangles' edges, each once with its triangles (global).
    let (mut node_at, mut tri_at) = (0u32, 0u32);
    let mut per: Vec<Vec<([u32; 2], Vec<u32>)>> = Vec::new();
    let mut firsts = Vec::new();
    for v in vehicles {
        let mut m: std::collections::BTreeMap<[u32; 2], Vec<u32>> = Default::default();
        for (t, tri) in v.triangles.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (tri.nodes[k], tri.nodes[(k + 1) % 3]);
                m.entry([node_at + a.min(b), node_at + a.max(b)]).or_default().push(tri_at + t as u32);
            }
        }
        per.push(m.into_iter().collect());
        firsts.push((node_at, tri_at));
        node_at += v.nodes.len() as u32;
        tri_at += v.triangles.len() as u32;
    }
    let edges: usize = per.iter().map(Vec::len).sum();
    let base = statics.len() as u32;
    let scratch_at = beams.len() as u32;
    beams.extend(std::iter::repeat_n([0.0f32; 4], CC_BEAMS_NODE * total));
    let acc_at = acc_init.len() as u32;
    acc_init.extend(std::iter::repeat_n(0, CC_ACC_NODE * total + nv + CC_COUNTERS));
    let edges_at = base + 2 + 3 * nv as u32;
    statics.push([scratch_at, acc_at, acc_at + (CC_ACC_NODE * total + nv) as u32, nv as u32]);
    statics.push([edges_at, edges as u32, 0.08f32.to_bits(), CC_SELF_GAP.to_bits()]);
    let mut first = 0u32;
    for ((v, p), &(n0, _)) in vehicles.iter().zip(&per).zip(&firsts) {
        let cap = CC_PAIRS_PER_ITEM * (v.nodes.len() + p.len());
        let list_at = next_init.len() as u32;
        next_init.extend(std::iter::repeat_n(0, CC_PAIR_WORDS * cap));
        let grid_cap = 64 * (v.triangles.len() + p.len());
        let grid_at = next_init.len() as u32;
        next_init.extend(std::iter::repeat_n(0, 2 * grid_cap));
        let buckets_at = acc_init.len() as u32;
        acc_init.extend(std::iter::repeat_n(0, 2 * CC_GRID_BUCKETS));
        // The own pairs under the clearance at spawn (seam neighbours: never paired).
        let x: Vec<glam::Vec3> = v.nodes.iter().map(|n| v.origin + n.pos).collect();
        let excl = engine_softbody::explicit_contact::close_pairs_at(&x, v, CC_SELF_GAP);
        let excl_at = lists.len() as u32;
        lists.extend(excl.iter().flat_map(|(k, _)| k.map(|i| n0 + i)));
        statics.push([first, p.len() as u32, list_at, cap as u32]);
        statics.push([grid_at, grid_cap as u32, excl_at, excl.len() as u32]);
        statics.push([buckets_at, 0, 0, 0]);
        first += p.len() as u32;
    }
    for p in &per {
        for (e, tris) in p {
            statics.push([e[0], e[1], lists.len() as u32, tris.len() as u32]);
            lists.extend(tris);
        }
    }
    (base, acc_at + (CC_ACC_NODE * total) as u32)
}
