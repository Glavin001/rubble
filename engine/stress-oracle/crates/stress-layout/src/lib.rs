//! The GPU stress solver's data layout, without a GPU: the per-chunk, per-bond and
//! per-island records of `stress-gpu`'s kernels (`shaders/world.slang`), the joint law's
//! data (`joint`), and the packing of a stress-ref solver's structures into them.
//! `stress-gpu` builds its buffers from these; rubble's cooker writes them for its own
//! GPU stress kernels, so both read one layout.

pub mod joint;


use stress_ref::math::{Mat3, Quat, Vec3};
use stress_ref::scene::{SolveMode, Support};
use stress_ref::solver::{Activity, ReferenceSolver};

use crate::joint::{bond_law, to_gpu_state, GpuJointBond, GpuJointState, MaterialTable};

pub const ISLAND_ANCHORED: u32 = 1;
pub const ISLAND_DRIVEN: u32 = 2;
pub const ISLAND_SETTLED: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BondStatic {
    pub t1: [f32; 4],
    pub t2: [f32; 4],
    pub normal: [f32; 4],
    pub ra: [f32; 4],
    pub rb: [f32; 4],
    pub centroid: [f32; 4],
    pub c_lin: [f32; 4],
    pub c_ang: [f32; 4],
    pub law: GpuJointBond,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BondDyn {
    pub js: GpuJointState,
    pub force_lin: [f32; 4],
    pub force_ang: [f32; 4],
    pub sums: [f32; 4],
    pub comps: [f32; 4],
    pub events: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkStatic {
    pub center: [f32; 4],
    pub inertia: [[f32; 4]; 3],
    pub inv: [[f32; 4]; 3],
    pub scale: [f32; 4],
    pub info: [u32; 4],
    pub load_range: [u32; 4],
    pub half: [f32; 4],
    pub crot: [[f32; 4]; 3],
    pub cmat: [f32; 4],
    pub cinfo: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Island {
    pub range: [u32; 4],
    pub info: [u32; 4],
    pub com: [f32; 4],
    pub inertia: [[f32; 4]; 3],
    pub inv: [[f32; 4]; 3],
    pub wcom: [f32; 4],
    pub winv: [[f32; 4]; 3],
    pub rotation: [f32; 4],
    pub position: [f32; 4],
    pub position_err: [f32; 4],
    pub velocity: [f32; 4],
    pub velocity_err: [f32; 4],
    pub angular_velocity: [f32; 4],
    /// Substeps run, path coverage (`coverage`, device bits), wide chunk groups, -.
    pub done: [u32; 4],
    pub probes: [u32; 4],
    pub energy: [f32; 4],
    /// The rounding error of `rotation` (compensated quaternion, world.slang `turn_left`).
    pub rotation_err: [f32; 4],
    /// World angular momentum about the centre of mass (compensated with
    /// `momentum_err`); the GPU derives `angular_velocity` from it.
    pub momentum: [f32; 4],
    pub momentum_err: [f32; 4],
}

pub fn v4(v: Vec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

pub fn rows(m: &Mat3) -> [[f32; 4]; 3] {
    std::array::from_fn(|r| [m.m[r][0] as f32, m.m[r][1] as f32, m.m[r][2] as f32, 0.0])
}

/// A quaternion as f32 components plus their rounding errors (`Island::rotation_err`).
pub fn split_quat(q: Quat) -> ([f32; 4], [f32; 4]) {
    let (x, y, z, w) = (split1(q.x), split1(q.y), split1(q.z), split1(q.w));
    ([x.0, y.0, z.0, w.0], [x.1, y.1, z.1, w.1])
}

/// A value as an f32 plus the f32 rounding error of that value.
pub fn split1(x: f64) -> (f32, f32) {
    let hi = x as f32;
    (hi, (x - hi as f64) as f32)
}

/// A vector as an f32 value plus the f32 rounding error of that value.
pub fn split(v: Vec3) -> ([f32; 4], [f32; 4]) {
    let hi = v4(v, 0.0);
    let lo = v4(v - Vec3::new(hi[0] as f64, hi[1] as f64, hi[2] as f64), 0.0);
    (hi, lo)
}

pub fn joined(hi: [f32; 4], lo: [f32; 4]) -> Vec3 {
    Vec3::new(hi[0] as f64 + lo[0] as f64, hi[1] as f64 + lo[1] as f64, hi[2] as f64 + lo[2] as f64)
}

pub fn vec3(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}


/// A solver's structures as the GPU records: chunks by cluster (`chunks[k] = (structure,
/// chunk)`, `chunk_index[s][c] = k`), each chunk's hidden state (u, th | reaction x, v |
/// reaction y, w | reaction z), one island per cluster, the bonds by cluster with their
/// law data and history, and the chunk-to-bond incidence (CSR offsets into this buffer,
/// then entries `2 bond + side`).
pub struct StructureTables {
    pub chunk_index: Vec<Vec<u32>>,
    pub chunks: Vec<(usize, usize)>,
    pub chunk_static: Vec<ChunkStatic>,
    pub state: Vec<[f32; 4]>,
    pub islands: Vec<Island>,
    pub table: MaterialTable,
    pub bonds: Vec<(usize, usize)>,
    /// `bond_index[s][b]`: the GPU bond of structure `s`'s bond `b` (`u32::MAX`: none).
    pub bond_index: Vec<Vec<u32>>,
    pub bond_static: Vec<BondStatic>,
    pub bond_dyn: Vec<BondDyn>,
    pub csr: Vec<u32>,
}

/// The GPU joint material table's capacity.
pub const MAX_MATERIALS: usize = 64;

/// Pack the solver's current structures (static data, history and state).
pub fn pack(m: &ReferenceSolver) -> Result<StructureTables, String> {
    pack_with(m, &[])
}

/// `pack`, with the clusters flagged in `exclude` (by cluster index; missing: included)
/// left out of the chunk and bond tables: each keeps its island record, with empty
/// chunk and bond ranges. For quiet clusters the GPU would not touch (stress-gpu
/// `GpuSolver::quiet_clusters`).
pub fn pack_with(m: &ReferenceSolver, exclude: &[bool]) -> Result<StructureTables, String> {
    let excluded = |ci: usize| exclude.get(ci).copied().unwrap_or(false);
        let mut chunk_index: Vec<Vec<u32>> = m.structures.iter().map(|st| vec![u32::MAX; st.chunks.len()]).collect();
        let chunk_total: usize = m.clusters.iter().enumerate().filter(|(ci, _)| !excluded(*ci)).map(|(_, cl)| cl.chunks.len()).sum();
        let bond_total: usize = m.clusters.iter().enumerate().filter(|(ci, _)| !excluded(*ci)).map(|(_, cl)| cl.bonds.len()).sum();
        let mut chunks = Vec::with_capacity(chunk_total);
        let mut chunk_static = Vec::with_capacity(chunk_total);
        let mut state = Vec::with_capacity(4 * chunk_total);
        let mut islands = Vec::with_capacity(m.clusters.len());
        for (ci, cl) in m.clusters.iter().enumerate() {
            let s = cl.structure;
            let st = &m.structures[s];
            let begin = chunks.len() as u32;
            let packed_chunks: &[usize] = if excluded(ci) { &[] } else { &cl.chunks };
            for &c in packed_chunks {
                let data = &st.chunks[c];
                let cs = &m.chunks[s][c];
                chunk_index[s][c] = chunks.len() as u32;
                chunks.push((s, c));
                let mu = cs.inertia_scale;
                chunk_static.push(ChunkStatic {
                    center: v4(data.center, data.mass),
                    inertia: rows(&data.inertia),
                    inv: rows(&data.inv_inertia),
                    scale: [mu as f32, (1.0 / (data.mass * mu)) as f32, (1.0 / mu) as f32, 0.0],
                    info: [
                        match data.support {
                            Support::None => 0,
                            Support::Fixed => 1,
                            Support::Pinned => 2,
                        },
                        ci as u32,
                        s as u32,
                        c as u32,
                    ],
                    load_range: [0; 4],
                    ..Default::default()
                });
                let r = cs.reaction.0;
                state.extend([v4(cs.u, 0.0), v4(cs.th, r.x), v4(cs.v, r.y), v4(cs.w, r.z)]);
            }
            // Drift-removal weights (solver.rs `remove_rigid_drift`).
            let weight = |c: usize| m.chunks[s][c].inertia_scale;
            let scaled = cl.chunks.iter().any(|&c| weight(c) != 1.0);
            let (wmass, wcom, winertia) = if scaled {
                let mass: f64 = cl.chunks.iter().map(|&c| st.chunks[c].mass * weight(c)).sum();
                let com = cl.chunks.iter().map(|&c| st.chunks[c].center * (st.chunks[c].mass * weight(c))).fold(Vec3::ZERO, |a, b| a + b) / mass;
                let mut inertia = Mat3::ZERO;
                for &c in &cl.chunks {
                    let ch = &st.chunks[c];
                    let r = ch.center - com;
                    inertia += (ch.inertia + (Mat3::IDENTITY * r.norm2() - Mat3::outer(r, r)) * ch.mass) * weight(c);
                }
                (mass, com, inertia)
            } else {
                (cl.mass, cl.com, cl.inertia)
            };
            // Under an engine (`integrate_rigid` off) every free cluster's rigid motion is
            // the engine's: it is driven (solver.rs `integrate_rigid`, `remove_rigid_drift`).
            let driven = cl.driven || !m.config.integrate_rigid;
            let mut flags = if cl.anchored { ISLAND_ANCHORED } else { 0 } | if driven { ISLAND_DRIVEN } else { 0 };
            // Quasi-static mode: every cluster moves rigidly between the frame's static
            // solves; adaptive: the settled ones, until their loads change (world.rs
            // `wake_loaded_clusters`: by more than `wake_threshold` of weight + norm).
            let settled = match m.config.mode {
                SolveMode::QuasiStatic => true,
                SolveMode::Adaptive => cl.activity == Activity::Settled,
                _ => false,
            };
            let mut settle = [0.0f32; 2];
            if settled {
                flags |= ISLAND_SETTLED;
                let weight = cl.mass * m.config.gravity.norm();
                let allowed = if m.config.mode == SolveMode::Adaptive { m.config.wake_threshold * (weight + cl.settled_load_norm) } else { f64::INFINITY };
                settle = [cl.settled_load_norm as f32, allowed as f32];
            }
            let (position, position_err) = split(cl.pose.position);
            let (velocity, velocity_err) = split(cl.velocity);
            let (rotation, rotation_err) = split_quat(cl.pose.rotation);
            let (momentum, momentum_err) = split(cl.world_inertia() * cl.angular_velocity);
            let inv = cl.inertia.inverse().unwrap_or(Mat3::ZERO);
            let winv = winertia.inverse().unwrap_or(Mat3::ZERO);
            islands.push(Island {
                range: [begin, chunks.len() as u32, 0, 0],
                info: [flags, 0, 0, m.substeps as u32],
                com: v4(cl.com, cl.mass),
                inertia: rows(&cl.inertia),
                inv: rows(&inv),
                wcom: v4(wcom, wmass),
                winv: rows(&winv),
                rotation,
                rotation_err,
                momentum,
                momentum_err,
                position,
                position_err,
                velocity,
                velocity_err,
                angular_velocity: v4(cl.angular_velocity, 0.0),
                energy: [0.0, 0.0, settle[0], settle[1]],
                ..Default::default()
            });
        }
        let mut table = MaterialTable::default();
        let mut bonds = Vec::with_capacity(bond_total);
        let mut bond_index: Vec<Vec<u32>> = m.structures.iter().map(|st| vec![u32::MAX; st.bonds.len()]).collect();
        let mut bond_static = Vec::with_capacity(bond_total);
        let mut bond_dyn = Vec::with_capacity(bond_total);
        // Chunk-to-bond incidence, `2 bond + side` in bond order (CSR built below).
        let mut incident: Vec<(u32, u32)> = Vec::with_capacity(2 * bond_total);
        for (ci, cl) in m.clusters.iter().enumerate() {
            let s = cl.structure;
            islands[ci].range[2] = bonds.len() as u32;
            let packed_bonds: &[usize] = if excluded(ci) { &[] } else { &cl.bonds };
            for &bi in packed_bonds {
                let b = &m.bonds[s][bi];
                let g = &b.geometry;
                let material = table.index(&b.strength);
                let mut law = bond_law(g, &b.stiffness, &b.strength, b.rebar.as_ref(), b.weibull, material);
                let (a, bb) = (chunk_index[s][g.a], chunk_index[s][g.b]);
                law.ids = [material, a, bb, ci as u32];
                let i = bonds.len() as u32;
                incident.push((a, 2 * i));
                incident.push((bb, 2 * i + 1));
                bond_index[s][bi] = i;
                bonds.push((s, bi));
                bond_static.push(BondStatic {
                    t1: v4(g.t1, 0.0),
                    t2: v4(g.t2, 0.0),
                    normal: v4(g.normal, 0.0),
                    ra: v4(g.ra, 0.0),
                    rb: v4(g.rb, 0.0),
                    centroid: v4(g.centroid, 0.0),
                    c_lin: v4(b.damping.lin, 0.0),
                    c_ang: v4(b.damping.ang, 0.0),
                    law,
                });
                bond_dyn.push(BondDyn {
                    js: to_gpu_state(&b.joint),
                    force_lin: v4(b.force.lin, b.stored),
                    force_ang: v4(b.force.ang, b.joint.utilization),
                    ..Default::default()
                });
            }
            islands[ci].range[3] = bonds.len() as u32;
        }
        if table.materials.len() > MAX_MATERIALS {
            return Err(format!("{} joint materials (the GPU table holds {MAX_MATERIALS})", table.materials.len()));
        }
        // CSR incidence: offsets (into this buffer) then entries, in bond order.
        let n = chunks.len();
        let mut degree = vec![0u32; n + 1];
        for &(c, _) in &incident {
            degree[c as usize + 1] += 1;
        }
        let mut csr = vec![0u32; n + 1 + incident.len()];
        let mut at = n + 1;
        for c in 0..n {
            csr[c] = at as u32;
            at += degree[c + 1] as usize;
        }
        csr[n] = at as u32;
        // Entries per chunk in bond order (a stable fill).
        let mut fill: Vec<usize> = (0..n).map(|c| csr[c] as usize).collect();
        for &(c, e) in &incident {
            csr[fill[c as usize]] = e;
            fill[c as usize] += 1;
        }
        Ok(StructureTables { chunk_index, chunks, chunk_static, state, islands, table, bonds, bond_index, bond_static, bond_dyn, csr })
}
