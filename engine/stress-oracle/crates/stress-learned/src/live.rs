//! Live MGN-T feature packing from a stress-ref mirror (inference / fracture-lab gate).
//!
//! Packs the same `NODE_IN` / `EDGE_IN` layout training uses (`learned_impact.model.encode_inputs`).
//! Honest gaps vs a full oracle frame are listed in [`FEATURE_GAPS`] — do not invent
//! `chunk_load_bins` or true QS probes here.

use stress_ref::bond::Local6;
use stress_ref::solver::{ReferenceSolver, RtBond};

use crate::constants::{CHUNK_STATE_DIM, EDGE_IN, JOINT_STATE_DIM, K, LOCAL6_DIM, NODE_IN};
use crate::joint_pack::pack_joint_state;

/// Fail-bin classes: 0 = none, 1..=K = time bin.
pub const FAIL_BINS: usize = K + 1;
/// Decoder output floats per edge: fail logits + peak util.
pub const OUT_PER_EDGE: usize = FAIL_BINS + 1;

/// Documented zeros / stand-ins vs SCHEMA / oracle recording.
pub const FEATURE_GAPS: &str = "\
chunk_load_bins: zero (no contact/scripted binning on the live mirror; need World::learned_bins). \
bond_qs_force: current bond.force (not a cascade-off QS probe). \
normalize: caller must apply train-time RMS scales (or accept distribution shift). \
chunk_contact: unused by MGN encode. \
";

/// One graph snapshot ready for MGN-T encode / Slang host.
#[derive(Clone, Debug)]
pub struct LiveFeatures {
    pub n_nodes: usize,
    pub n_edges: usize,
    /// Row-major `[N * NODE_IN]`.
    pub node_in: Vec<f32>,
    /// Row-major `[E * EDGE_IN]`.
    pub edge_in: Vec<f32>,
    /// Undirected endpoints `[E * 2]` as dense node ids.
    pub endpoints: Vec<i32>,
    /// Parallel to edges: `(structure, bond)` in the mirror.
    pub bond_keys: Vec<(usize, usize)>,
    /// Parallel to nodes: `(structure, chunk)`.
    pub chunk_keys: Vec<(usize, usize)>,
}

impl LiveFeatures {
    pub fn gaps() -> &'static str {
        FEATURE_GAPS
    }
}

/// Pack active chunks + **connected** bonds from `solver`.
///
/// Disconnected / dead bonds are omitted so edge indices match a live break list.
pub fn pack_live_features(solver: &ReferenceSolver) -> LiveFeatures {
    let mut chunk_keys = Vec::new();
    let mut chunk_of: Vec<Vec<Option<usize>>> = solver
        .structures
        .iter()
        .map(|s| vec![None; s.chunks.len()])
        .collect();
    for (s, st) in solver.structures.iter().enumerate() {
        for (c, _) in st.chunks.iter().enumerate() {
            if !solver.chunks[s][c].active {
                continue;
            }
            let id = chunk_keys.len();
            chunk_of[s][c] = Some(id);
            chunk_keys.push((s, c));
        }
    }
    let n = chunk_keys.len();
    let mut bond_keys = Vec::new();
    for (s, bonds) in solver.bonds.iter().enumerate() {
        for (b, bond) in bonds.iter().enumerate() {
            if !bond.connected() {
                continue;
            }
            let ga = bond.geometry.a;
            let gb = bond.geometry.b;
            if chunk_of[s][ga].is_none() || chunk_of[s][gb].is_none() {
                continue;
            }
            bond_keys.push((s, b));
        }
    }
    let e = bond_keys.len();
    let mut node_in = vec![0.0f32; n * NODE_IN];
    let mut edge_in = vec![0.0f32; e * EDGE_IN];
    let mut endpoints = vec![0i32; e * 2];
    let g = solver.config.gravity;

    for (ni, &(s, c)) in chunk_keys.iter().enumerate() {
        let ch = &solver.structures[s].chunks[c];
        let st = &solver.chunks[s][c];
        let base = ni * NODE_IN;
        node_in[base] = ch.mass as f32;
        node_in[base + 1] = ch.inertia.m[0][0] as f32;
        node_in[base + 2] = ch.inertia.m[1][1] as f32;
        node_in[base + 3] = ch.inertia.m[2][2] as f32;
        node_in[base + 4] = ch.half_extents.x as f32;
        node_in[base + 5] = ch.half_extents.y as f32;
        node_in[base + 6] = ch.half_extents.z as f32;
        write_chunk_state(&mut node_in[base + 7..base + 7 + CHUNK_STATE_DIM], st);
        // loads: base+19 .. base+19+24 stay zero (FEATURE_GAPS).
        let cl = &solver.clusters[st.cluster];
        let gl = cl.pose.inverse_transform_vector(g);
        let go = base + 7 + CHUNK_STATE_DIM + K * LOCAL6_DIM;
        node_in[go] = gl.x as f32;
        node_in[go + 1] = gl.y as f32;
        node_in[go + 2] = gl.z as f32;
    }

    for (ei, &(s, b)) in bond_keys.iter().enumerate() {
        let bond: &RtBond = &solver.bonds[s][b];
        let ia = chunk_of[s][bond.geometry.a].unwrap() as i32;
        let ib = chunk_of[s][bond.geometry.b].unwrap() as i32;
        endpoints[ei * 2] = ia;
        endpoints[ei * 2 + 1] = ib;
        let base = ei * EDGE_IN;
        let joint = pack_joint_state(&bond.joint);
        edge_in[base..base + JOINT_STATE_DIM].copy_from_slice(&joint);
        write_local6(&mut edge_in[base + JOINT_STATE_DIM..base + EDGE_IN], &bond.force);
    }

    LiveFeatures {
        n_nodes: n,
        n_edges: e,
        node_in,
        edge_in,
        endpoints,
        bond_keys,
        chunk_keys,
    }
}

/// Argmax fail-bin per edge; returns dense edge indices whose best class is ≠ 0.
pub fn fail_edge_indices(pred: &[f32], n_edges: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for e in 0..n_edges {
        let base = e * OUT_PER_EDGE;
        if base + FAIL_BINS > pred.len() {
            break;
        }
        let mut best = 0usize;
        let mut best_v = pred[base];
        for c in 1..FAIL_BINS {
            let v = pred[base + c];
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        if best > 0 {
            out.push(e);
        }
    }
    out
}

/// Map dense edge indices → `(structure, bond)` keys (skips OOB).
pub fn bond_keys_for_edges(live: &LiveFeatures, edges: &[usize]) -> Vec<(usize, usize)> {
    edges
        .iter()
        .filter_map(|&e| live.bond_keys.get(e).copied())
        .collect()
}

fn write_chunk_state(dst: &mut [f32], st: &stress_ref::solver::ChunkState) {
    let v = [
        st.u.x, st.u.y, st.u.z, st.th.x, st.th.y, st.th.z, st.v.x, st.v.y, st.v.z, st.w.x, st.w.y, st.w.z,
    ];
    for (i, x) in v.into_iter().enumerate() {
        dst[i] = x as f32;
    }
}

fn write_local6(dst: &mut [f32], f: &Local6) {
    for (i, x) in f.to_array().into_iter().enumerate() {
        dst[i] = x as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stress_ref::builders;

    #[test]
    fn pack_dims_and_connected_only() {
        let scene = builders::cantilever(6, "live_pack_cant");
        let world = stress_ref::world::World::new(&scene);
        let live = pack_live_features(&world.solver);
        assert_eq!(live.node_in.len(), live.n_nodes * NODE_IN);
        assert_eq!(live.edge_in.len(), live.n_edges * EDGE_IN);
        assert_eq!(live.endpoints.len(), live.n_edges * 2);
        assert_eq!(live.bond_keys.len(), live.n_edges);
        assert!(live.n_nodes > 0);
        assert!(live.n_edges > 0);
        // Loads slice is zero (honest gap).
        let load0 = 7 + CHUNK_STATE_DIM;
        assert!(live.node_in[load0..load0 + K * LOCAL6_DIM].iter().all(|&x| x == 0.0));
        assert!(FEATURE_GAPS.contains("chunk_load_bins"));
    }

    #[test]
    fn fail_argmax_skips_none() {
        let n = 3;
        let mut pred = vec![0.0f32; n * OUT_PER_EDGE];
        // edge 0: class 0 wins
        pred[0] = 2.0;
        pred[1] = 0.1;
        // edge 1: class 2 wins
        let b = OUT_PER_EDGE;
        pred[b] = 0.1;
        pred[b + 2] = 3.0;
        // edge 2: class 1 wins
        let b2 = 2 * OUT_PER_EDGE;
        pred[b2 + 1] = 1.5;
        assert_eq!(fail_edge_indices(&pred, n), vec![1, 2]);
    }
}
