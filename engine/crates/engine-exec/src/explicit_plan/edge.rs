//! Edge vehicle contact's tables in the upload (`ExplicitBeams::edge_contact`,
//! explicit_beams_vv_edges.slang; [`super::W_EDGE`]).

use super::schedule::Switches;
use super::*;

/// The owners' state by vehicle and the schedules built on it: lossless switches that edge
/// vehicle contact does not run on yet (its hits run in the small groups' own levels,
/// explicit_beams_vv_edges.slang), so an upload with it leaves them off.
pub const PERF_FLEET_FAMILY: u32 = PERF_FLEET_STATE
    | PERF_FLEET_FRONT
    | PERF_FLEET_ANY
    | PERF_FLEET_POST
    | PERF_FLEET_LEVELS
    | PERF_FLEET_HITLAT
    | PERF_LEVELS_ONCE
    | PERF_FLEET_FUSE
    | PERF_FLEET_KEEP
    | PERF_HIT_SLOT_MAP
    | PERF_FLEET_WALK_BOX;

/// The upload has edge contact; each vehicle with it has two-body and dissipative contact.
pub(super) fn any_edge(vehicles: &[ExplicitBeams]) -> bool {
    // Edge vehicle contact (`ExplicitBeams::edge_contact`): the kernels run its hits as
    // two-body dissipative hits only (they read the turn's snapshot, never another hit's
    // force: explicit_beams_vv_edges.slang), so a vehicle with it needs both.
    assert!(
        vehicles.iter().all(|v| !v.edge_contact || (v.two_body_contact && v.dissipative_contact)),
        "explicit beams: edge vehicle contact needs two-body and dissipative contact on the same vehicle"
    );
    vehicles.iter().any(|v| v.edge_contact)
}

/// Edge vehicle contact's tables ([`W_EDGE`], explicit_beams_vv_edges.slang; none
/// without it): per vehicle two gStatic records, (first edge record, edges, first
/// scratch float4 in gBeams, first hit word in gNext) and (its EDGE_ACC words in gAcc:
/// hits, listed edges, listed points, its edges' box as six keys; its hit room; its
/// first triangle; its listed edges in gNext, then its listed points); per edge
/// (global i, global j, first triangle entry in gLists, triangles); per triangle
/// entry the vehicle's own triangle index, bit 31 on the outer skin. Scratch per edge
/// EDGE_SCRATCH float4 (and two a vehicle, unused); per vehicle a hit list of two
/// words a hit, room for its edges and contact points.
pub(super) fn edge_tables(
    vehicles: &[ExplicitBeams],
    statics: &mut Vec<[u32; 4]>,
    lists: &mut Vec<u32>,
    beams: &mut Vec<[f32; 4]>,
    next_init: &mut Vec<u32>,
    acc_init: &mut Vec<i32>,
) -> u32 {
    let nv = vehicles.len();
    let topo: Vec<Vec<engine_softbody::explicit_contact::EdgeTopology>> =
        vehicles.iter().map(engine_softbody::explicit_contact::skin_edge_topology).collect();
    // (A measurement switch first, `RUBBLE_EDGE_ABLATE` bits, time only: 1 the pair tests,
    // 2 the grids, 4 the edges' step data skipped; then the vehicles' records.)
    let ablate = std::env::var("RUBBLE_EDGE_ABLATE").ok().and_then(|v| v.parse().ok()).unwrap_or(0u32);
    // (`RUBBLE_EDGE_SKIN`, m: the candidate list's margin, a measurement; 0: the kernel's.)
    let skin = std::env::var("RUBBLE_EDGE_SKIN").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
    statics.push([ablate, skin.to_bits(), 0, 0]);
    let base = statics.len() as u32;
    let recs_at = base + 2 * nv as u32;
    let mut edge_recs: Vec<[u32; 4]> = Vec::new();
    let (mut node_at, mut tri_at) = (0u32, 0u32);
    let mut scratch_at = beams.len() as u32;
    for (v, edges) in vehicles.iter().zip(&topo) {
        let room = (edges.len() + v.contact_points.len()) as u32;
        let hits_at = next_init.len() as u32;
        let list_at = hits_at + 2 * room;
        // (The listed edges and points, then the grid's entries: EDGE_CELLS_MAX an edge, 8 a
        // point; then the long edges' list, the owner work list and its scan's block sums, the
        // frame-coherent candidates.)
        next_init.extend(std::iter::repeat_n(
            0,
            2 * room as usize
                + (3 + EDGE_CELLS_MAX) * edges.len()
                + edges.len() / 32
                + 1
                + 9 * v.contact_points.len()
                + 2 * EDGE_CAND * (edges.len() + v.contact_points.len()),
        ));
        let counter = acc_init.len() as u32;
        acc_init.extend(std::iter::repeat_n(0, EDGE_ACC));
        statics.push([recs_at + edge_recs.len() as u32, edges.len() as u32, scratch_at, hits_at]);
        statics.push([counter, room, tri_at, list_at]);
        for e in edges {
            edge_recs.push([node_at + e.i, node_at + e.j, lists.len() as u32, e.tris.len() as u32]);
            lists.extend(e.tris.iter().map(|&(t, outer)| t | (outer as u32) << 31));
        }
        // (Then two spare float4 and the nodes at the candidates' last build.)
        scratch_at += (EDGE_SCRATCH * edges.len() + 2 + v.nodes.len()) as u32;
        node_at += v.nodes.len() as u32;
        tri_at += v.triangles.len() as u32;
    }
    statics.extend(edge_recs);
    beams.extend(std::iter::repeat_n([0.0f32; 4], scratch_at as usize - beams.len()));
    base
}

/// Contact offsets' tables ([`super::W_OFFSET`], explicit_beams_members.slang contactOffsets):
/// at base - 1 the vectors' first float4 in gBeams; per global node (its outer triangles'
/// first entry in gLists, their count, its offset's bits, 0), the triangles (live outer,
/// not anti-clip-only) in their order; then per vehicle (its largest offset's bits, 0, 0, 0).
pub(super) fn offset_tables(
    vehicles: &[ExplicitBeams],
    statics: &mut Vec<[u32; 4]>,
    lists: &mut Vec<u32>,
    beams: &mut Vec<[f32; 4]>,
) -> u32 {
    let total: usize = vehicles.iter().map(|v| v.nodes.len()).sum();
    statics.push([beams.len() as u32, 0, 0, 0]);
    beams.extend(std::iter::repeat_n([0.0f32; 4], total));
    let base = statics.len() as u32;
    let mut tri_at = 0u32;
    for v in vehicles {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); v.nodes.len()];
        for (t, tri) in v.triangles.iter().enumerate() {
            if v.tri_anti_only(t) || !v.tri_outer.get(t).copied().unwrap_or(true) {
                continue;
            }
            for &i in &tri.nodes {
                adj[i as usize].push(tri_at + t as u32);
            }
        }
        for (i, a) in adj.iter().enumerate() {
            let o = v.contact_offset.get(i).copied().unwrap_or(0.0);
            statics.push([lists.len() as u32, a.len() as u32, o.to_bits(), 0]);
            lists.extend(a);
        }
        tri_at += v.triangles.len() as u32;
    }
    for v in vehicles {
        let m = v.contact_offset.iter().copied().fold(0.0f32, f32::max);
        statics.push([m.to_bits(), 0, 0, 0]);
    }
    base
}

/// Edge vehicle contact runs only as the split group path's pass: a frame with contact
/// off that path would drop its edge hits.
pub(super) fn edge_frame_check(planner: &FramePlanner, sw: &Switches, group: bool, steps: u32) {
    // (A browser host whose split path's modules still compile runs the groups' workgroup,
    // without edge hits, until they are.)
    let waiting = ["member", "tests", "group", "groupbegin", "grouplevels", "grouplv"]
        .iter()
        .any(|w| !(sw.ready)(&split_module_name(w, sw.aero, sw.lanes)));
    assert!(
        planner.words[W_EDGE] == 0 || group || waiting || !planner.contact_active() || steps == 0,
        "explicit beams: edge vehicle contact needs the split group path (group_split on, not debug)"
    );
}

pub(super) fn edge_module_ready(sw: &Switches) -> bool {
    (sw.ready)(&split_module_name("edges", sw.aero, sw.lanes))
}

/// Whether the split group path runs edge vehicle contact's pass: the frame's parameters
/// have edge contact ([`super::schedule::frame`] clears it until the module is compiled).
pub fn edge_pass_on(planner: &FramePlanner) -> bool {
    planner.words[W_EDGE] != 0
}

/// The perf flags an upload with edge contact or contact offsets runs: all but the
/// levels-once module, which has neither the offsets' code (EB_NO_OFFSETS) nor the edge
/// pairs' residual clamp (EB_EDGE_CLAMP: the groups' and the fleet levels' modules).
pub(super) fn contact_flags(flags: u32, edge: bool, offset: bool) -> u32 {
    if edge || offset {
        flags & !PERF_LEVELS_ONCE
    } else {
        flags
    }
}
