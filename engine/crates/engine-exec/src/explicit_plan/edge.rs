//! Edge vehicle contact's tables in the upload (`ExplicitBeams::edge_contact`,
//! explicit_beams_vv_edges.slang; [`super::W_EDGE`]).

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
    let base = statics.len() as u32;
    let recs_at = base + 2 * nv as u32;
    let mut edge_recs: Vec<[u32; 4]> = Vec::new();
    let (mut node_at, mut tri_at) = (0u32, 0u32);
    let mut scratch_at = beams.len() as u32;
    for (v, edges) in vehicles.iter().zip(&topo) {
        let room = (edges.len() + v.contact_points.len()) as u32;
        let hits_at = next_init.len() as u32;
        let list_at = hits_at + 2 * room;
        // (The listed edges and points, then their grid-sorted copies, then each entry's
        // bucket and rank.)
        next_init.extend(std::iter::repeat_n(0, 2 * room as usize + 3 * (edges.len() + v.contact_points.len())));
        let counter = acc_init.len() as u32;
        acc_init.extend(std::iter::repeat_n(0, EDGE_ACC));
        statics.push([recs_at + edge_recs.len() as u32, edges.len() as u32, scratch_at, hits_at]);
        statics.push([counter, room, tri_at, list_at]);
        for e in edges {
            edge_recs.push([node_at + e.i, node_at + e.j, lists.len() as u32, e.tris.len() as u32]);
            lists.extend(e.tris.iter().map(|&(t, outer)| t | (outer as u32) << 31));
        }
        scratch_at += (EDGE_SCRATCH * edges.len() + 2) as u32;
        node_at += v.nodes.len() as u32;
        tri_at += v.triangles.len() as u32;
    }
    statics.extend(edge_recs);
    beams.extend(std::iter::repeat_n([0.0f32; 4], scratch_at as usize - beams.len()));
    base
}
