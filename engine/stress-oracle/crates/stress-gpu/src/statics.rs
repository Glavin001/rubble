//! Static equilibrium of clusters on the GPU (`island_statics` in `world.slang`):
//! statics.rs `equilibrate` with the loads the host gives, for the frame tails of the
//! adaptive and quasi-static modes (settle, cascade). The external loads are computed
//! here exactly as statics.rs `static_loads` does (rigid acceleration and frame loads
//! of the cluster, body frame).

use stress_ref::math::Vec3;
use stress_ref::solver::{ChunkLoads, ReferenceSolver};

/// Outcome of one cluster's solve (statics.rs `StaticReport`, the parts that matter).
#[derive(Clone, Copy, Debug, Default)]
pub struct Equilibrium {
    pub converged: bool,
    pub residual: f64,
    pub newton_iterations: u32,
    pub cg_iterations: u32,
}

/// solver.rs `net_load`: net external force and torque about the com, world frame.
fn net_load(m: &ReferenceSolver, ci: usize, loads: &ChunkLoads) -> (Vec3, Vec3) {
    let cl = &m.clusters[ci];
    let s = cl.structure;
    let com = cl.com_world();
    let mut f = Vec3::ZERO;
    let mut t = Vec3::ZERO;
    for &c in &cl.chunks {
        let ch = &m.structures[s].chunks[c];
        let fc = loads.force[s][c] + m.config.gravity * ch.mass;
        let x = cl.pose.transform_point(ch.center + m.chunks[s][c].u);
        f += fc;
        t += (x - com).cross(fc) + loads.torque[s][c];
    }
    for rl in &m.replacements {
        if rl.structure == s && m.chunks[s][rl.chunk].active && m.chunks[s][rl.chunk].cluster == ci {
            let k = rl.factor(m.time);
            let x = cl.pose.transform_point(m.structures[s].chunks[rl.chunk].center);
            let fw = cl.pose.transform_vector(rl.force * k);
            f += fw;
            t += (x - com).cross(fw) + cl.pose.transform_vector(rl.moment * k);
        }
    }
    (f, t)
}

/// solver.rs `rigid_acceleration`.
fn rigid_acceleration(m: &ReferenceSolver, ci: usize, loads: &ChunkLoads) -> (Vec3, Vec3) {
    let cl = &m.clusters[ci];
    if cl.anchored {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let (f, t) = net_load(m, ci, loads);
    let iw = cl.world_inertia();
    let w = cl.angular_velocity;
    let alpha = iw.inverse().expect("cluster inertia invertible") * (t - w.cross(iw * w));
    (f / cl.mass, alpha)
}

/// statics.rs `static_loads`: each chunk's external load in the cluster's body frame
/// (gravity, the given loads, replacement loads, minus the rigid inertial loads), in the
/// cluster's chunk order.
pub fn static_loads(m: &ReferenceSolver, ci: usize, loads: &ChunkLoads) -> Vec<(Vec3, Vec3)> {
    let (a, alpha) = rigid_acceleration(m, ci, loads);
    let cl = &m.clusters[ci];
    let s = cl.structure;
    let w = cl.angular_velocity;
    let rot = cl.rotation();
    let rml = m.config.features.rigid_motion_loads;
    cl.chunks
        .iter()
        .map(|&c| {
            // solver.rs `frame_loads` at the cluster's current time.
            let ch = &m.structures[s].chunks[c];
            let st = &m.chunks[s][c];
            let r_world = cl.pose.transform_vector(ch.center + st.u - cl.com);
            let iw = rot * ch.inertia * rot.transpose();
            let mut f_world = loads.force[s][c] + m.config.gravity * ch.mass;
            let mut t_world = loads.torque[s][c];
            if rml {
                f_world -= (a + alpha.cross(r_world) + w.cross(w.cross(r_world))) * ch.mass;
                t_world -= iw * alpha + w.cross(iw * w);
            }
            let mut f = cl.pose.inverse_transform_vector(f_world);
            let mut mo = cl.pose.inverse_transform_vector(t_world);
            if rml {
                let wb = cl.pose.inverse_transform_vector(w);
                f -= wb.cross(st.v) * (2.0 * ch.mass);
                mo -= wb.cross(ch.inertia * st.w) + st.w.cross(ch.inertia * wb) + st.w.cross(ch.inertia * st.w);
            }
            for rl in &m.replacements {
                if rl.structure == s && rl.chunk == c {
                    let k = rl.factor(m.time);
                    f += rl.force * k;
                    mo += rl.moment * k;
                }
            }
            (f, mo)
        })
        .collect()
}
