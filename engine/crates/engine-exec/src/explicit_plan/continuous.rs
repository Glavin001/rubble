//! Continuous contact's place in the split group path (`docs/continuous-contact-design.md`,
//! "The GPU port"): each step's members' dispatch cut in two at the node update (the kernel's
//! `SPLIT_CC_NODES` and `SPLIT_CC_REST`, `explicit_beams_split.slang`), the continuous passes
//! between. The two halves read parameter copies of their own after every other copy of the
//! frame: step s's copy, its step word with the half's bit.

use super::schedule::{member_rest_on, FrameKind, GroupOp, Switches};
use super::{FramePlanner, PARAM_STRIDE};

/// The step word's bit for the node update's half, and the rest's.
pub const SPLIT_CC_NODES: u32 = 0x0800_0000;
pub const SPLIT_CC_REST: u32 = 0x0400_0000;

/// Whether this frame's members run in two halves.
pub fn on(planner: &FramePlanner, kind: &FrameKind) -> bool {
    planner.continuous && kind.group_split && !kind.self_split_vv
}

/// The first of the halves' copies (two a step): after the frame's others
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
    let c = first_copy(planner, sw, kind, steps) + 2 * s;
    out.push(GroupOp::Member(c));
    out.push(GroupOp::Member(c + 1));
}

/// The halves' parameter copies, appended: step s's copy with its step word's half bit.
pub fn copies(planner: &FramePlanner, sw: &Switches, kind: &FrameKind, steps: u32, bytes: &mut Vec<u8>) {
    if !on(planner, kind) {
        return;
    }
    let stride = PARAM_STRIDE as usize;
    let first = first_copy(planner, sw, kind, steps) as usize;
    bytes.resize(stride * (first + 2 * steps as usize), 0);
    for s in 0..steps as usize {
        for (h, bit) in [SPLIT_CC_NODES, SPLIT_CC_REST].into_iter().enumerate() {
            let (from, to) = (stride * s, stride * (first + 2 * s + h));
            bytes.copy_within(from..from + stride, to);
            let at = to + 4 * 11;
            let w = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) | bit;
            bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
}
