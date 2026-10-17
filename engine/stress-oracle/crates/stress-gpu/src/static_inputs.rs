//! The inputs of one cluster's static solve, bit for bit, so a solve whose inputs repeat
//! one that left its start state unchanged (converged, or unconverged and kept) can be skipped (the engine-coupled
//! quasi-static frame, `GpuWorld::solve_static_all`): the GPU solve is deterministic, so
//! the same start state, bond histories and loads give the same equilibrium, and that
//! equilibrium is already the state. Lossless: a skipped solve is one that would return
//! its start bit for bit.

use stress_ref::math::Vec3;
use stress_ref::solver::{ChunkLoads, ReferenceSolver};

#[derive(Clone, Debug, PartialEq)]
pub struct StaticInputs {
    chunks: Vec<usize>,
    bonds: Vec<usize>,
    /// Bits of every chunk's start state (u, th) and its static load (force, moment).
    bits: Vec<u64>,
    joints: Vec<stress_ref::joint::JointState>,
}

fn push(bits: &mut Vec<u64>, v: Vec3) {
    bits.extend([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]);
}

impl StaticInputs {
    pub fn of(m: &ReferenceSolver, ci: usize, loads: &ChunkLoads) -> StaticInputs {
        let cl = &m.clusters[ci];
        let s = cl.structure;
        let mut bits = Vec::with_capacity(12 * cl.chunks.len());
        for (&c, (f, t)) in cl.chunks.iter().zip(crate::statics::static_loads(m, ci, loads)) {
            let st = &m.chunks[s][c];
            push(&mut bits, st.u);
            push(&mut bits, st.th);
            push(&mut bits, f);
            push(&mut bits, t);
        }
        StaticInputs { chunks: cl.chunks.clone(), bonds: cl.bonds.clone(), bits, joints: cl.bonds.iter().map(|&bi| m.bonds[s][bi].joint.clone()).collect() }
    }

    /// Whether the cluster's start state is still `self`'s (the solve left it unchanged).
    pub fn same_state(&self, m: &ReferenceSolver, ci: usize) -> bool {
        let cl = &m.clusters[ci];
        cl.chunks == self.chunks
            && cl.chunks.iter().enumerate().all(|(k, &c)| {
                let st = &m.chunks[cl.structure][c];
                let b = &self.bits[12 * k..12 * k + 6];
                [st.u.x, st.u.y, st.u.z, st.th.x, st.th.y, st.th.z].iter().zip(b).all(|(v, &w)| v.to_bits() == w)
            })
    }
}
