//! Debug snapshots: the complete visual and diagnostic state of a [`World`] at one
//! instant, in world coordinates, for renderers, debuggers and recordings.
//!
//! A snapshot is a read-only view; taking one never changes the simulation. Per chunk
//! it reports the box (including its hidden displacement and rotation), the cluster it
//! belongs to, its velocity, its hidden deformation, and the **average stress tensor**
//! of the chunk from its bond forces (Love-Weber formula, the standard particle stress
//! of discrete-element methods):
//!
//! `sigma = 1/V * sum_bonds sym(r_c (x) f_c)`
//!
//! where `r_c` runs from the chunk centre to the bond's patch centroid and `f_c` is the
//! force the bond applies to the chunk (tension positive). Contact forces are not
//! included, so the field shows the stress carried through the bond network.
//!
//! Per bond it reports the patch (centroid, normal, extents), its damage, crushing,
//! strength lost to static fatigue, utilization (failure index) and mode, and whether it is
//! intact, cracked or broken. Broken bonds keep their last world placement on the
//! chunk that held them (`a`), which is where a renderer draws the crack.

use serde::Serialize;

use crate::joint::FailureMode;
use crate::math::{Mat3, Quat, Vec3};
use crate::scene::ImpactorShape;
use crate::world::World;

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub scene: String,
    pub time: f64,
    pub frame: u64,
    pub chunks: Vec<ChunkView>,
    pub bonds: Vec<BondView>,
    pub impactors: Vec<ImpactorView>,
    /// Ground plane height, if the scene has one.
    pub ground: Option<f64>,
    pub energy: EnergyView,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkView {
    pub body: usize,
    pub chunk: usize,
    /// Box centre (m) and orientation (rotation matrix rows) in the world frame.
    pub center: [f64; 3],
    pub rotation: [[f64; 3]; 3],
    pub half_extents: [f64; 3],
    pub material: String,
    /// Refinement level (0 = coarsest).
    pub level: u32,
    /// Unique id of the rigid cluster carrying the chunk (fragments differ).
    pub cluster: u64,
    /// The cluster holds supported chunks.
    pub anchored: bool,
    /// Supported (fixed) chunk.
    pub supported: bool,
    pub velocity: [f64; 3],
    /// Hidden (elastic) displacement in the world frame (m).
    pub deformation: [f64; 3],
    /// Love-Weber average stress (Pa, world frame, tension positive, symmetric).
    pub stress: [[f64; 3]; 3],
    /// Von Mises equivalent of `stress` (Pa).
    pub von_mises: f64,
    /// Largest principal stress (Pa, tension positive).
    pub max_principal: f64,
    /// Largest failure index of the chunk's live bonds (1 = at strength).
    pub utilization: f64,
    /// Largest damage `max(D, Dc)` of the chunk's bonds, broken bonds count as 1.
    pub damage: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BondStatus {
    Intact,
    /// Damaged but still connecting its chunks.
    Cracked,
    /// No longer connecting (fully damaged, or the chunks are in different clusters).
    Broken,
}

#[derive(Clone, Debug, Serialize)]
pub struct BondView {
    pub body: usize,
    pub a: usize,
    pub b: usize,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub tangent: [f64; 3],
    /// Patch extents along `tangent` and `normal x tangent` (m).
    pub width: [f64; 2],
    pub status: BondStatus,
    pub damage: f64,
    pub crush: f64,
    /// Strength lost to static fatigue (0..1).
    pub fatigue_loss: f64,
    pub utilization: f64,
    pub mode: Option<FailureMode>,
    /// Axial stress of the patch (Pa, tension positive) and its max shear stress.
    pub axial_stress: f64,
    pub shear_stress: f64,
    /// Energy dissipated by this bond so far (J).
    pub dissipated: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImpactorView {
    pub name: String,
    pub shape: ImpactorShape,
    pub center: [f64; 3],
    pub rotation: [[f64; 3]; 3],
    pub velocity: [f64; 3],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct EnergyView {
    pub mechanical: f64,
    pub dissipated: f64,
    pub bond_dissipation: f64,
    pub contact_dissipation: f64,
    pub kinetic: f64,
    pub elastic: f64,
}

fn rows(m: Mat3) -> [[f64; 3]; 3] {
    m.m
}

/// Eigenvalues of a symmetric 3x3 matrix (closed form, Smith 1961), descending.
pub fn symmetric_eigenvalues(s: &Mat3) -> [f64; 3] {
    let m = &s.m;
    let p1 = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    let q = s.trace() / 3.0;
    if p1 <= 1e-30 * (q * q).max(1e-300) {
        let mut d = [m[0][0], m[1][1], m[2][2]];
        d.sort_by(|a, b| b.total_cmp(a));
        return d;
    }
    let p2 = (m[0][0] - q).powi(2) + (m[1][1] - q).powi(2) + (m[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let b = (*s - Mat3::IDENTITY * q) * (1.0 / p);
    let r = (b.det() / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [e1, 3.0 * q - e1 - e3, e3]
}

pub fn von_mises(s: &Mat3) -> f64 {
    let m = &s.m;
    let d = (m[0][0] - m[1][1]).powi(2) + (m[1][1] - m[2][2]).powi(2) + (m[2][2] - m[0][0]).powi(2);
    let o = m[0][1].powi(2) + m[1][2].powi(2) + m[0][2].powi(2);
    (0.5 * d + 3.0 * o).sqrt()
}

impl World {
    /// The current state as a [`Snapshot`].
    pub fn snapshot(&self) -> Snapshot {
        let solver = &self.solver;
        let mut chunks = Vec::new();
        let mut bonds = Vec::new();
        for (s, st) in solver.structures.iter().enumerate() {
            // Per-chunk stress sums (body frame) and bond maxima.
            let n = st.chunks.len();
            let mut stress = vec![Mat3::ZERO; n];
            let mut util = vec![0.0f64; n];
            let mut damage = vec![0.0f64; n];
            for b in &solver.bonds[s] {
                let g = &b.geometry;
                let (ca, cb) = (&solver.chunks[s][g.a], &solver.chunks[s][g.b]);
                let connected = b.alive && ca.active && cb.active && ca.cluster == cb.cluster;
                let d = if connected { b.joint.damage.max(b.joint.crush) } else { 1.0 };
                for c in [g.a, g.b] {
                    damage[c] = damage[c].max(d);
                }
                if connected {
                    let (fa, _, fb, _) = g.chunk_loads(&b.force);
                    stress[g.a] += sym_outer(g.ra, fa);
                    stress[g.b] += sym_outer(g.rb, fb);
                    for c in [g.a, g.b] {
                        util[c] = util[c].max(b.joint.utilization);
                    }
                }
                let holder = if ca.active { g.a } else { g.b };
                let cl = &solver.clusters[solver.chunks[s][holder].cluster];
                let status = if !connected {
                    BondStatus::Broken
                } else if b.joint.damage > 0.0 || b.joint.crush > 0.0 {
                    BondStatus::Cracked
                } else {
                    BondStatus::Intact
                };
                if !(solver.chunks[s][g.a].active || solver.chunks[s][g.b].active) {
                    continue;
                }
                bonds.push(BondView {
                    body: s,
                    a: g.a,
                    b: g.b,
                    centroid: cl.pose.transform_point(g.centroid + solver.chunks[s][holder].u).to_array(),
                    normal: cl.pose.transform_vector(g.normal).to_array(),
                    tangent: cl.pose.transform_vector(g.t1).to_array(),
                    width: g.width,
                    status,
                    damage: b.joint.damage,
                    crush: b.joint.crush,
                    fatigue_loss: 1.0 - crate::joint::fatigue_factor(&b.strength, &b.joint),
                    utilization: b.joint.utilization,
                    mode: b.joint.mode,
                    axial_stress: b.force.lin.z / g.area,
                    shear_stress: b.measures.shear,
                    dissipated: b.joint.dissipated,
                });
            }
            for (c, ch) in st.chunks.iter().enumerate() {
                let state = &solver.chunks[s][c];
                if !state.active {
                    continue;
                }
                let cl = &solver.clusters[state.cluster];
                let r = cl.rotation();
                let hidden = Quat::from_axis_angle(state.th, state.th.norm()).to_mat3();
                let volume = ch.volume();
                let sigma = r * stress[c] * r.transpose() * (1.0 / volume);
                let (v, _) = solver.chunk_velocity(s, c);
                chunks.push(ChunkView {
                    body: s,
                    chunk: c,
                    center: solver.chunk_position(s, c).to_array(),
                    rotation: rows(r * hidden * ch.rotation),
                    half_extents: ch.half_extents.to_array(),
                    material: ch.material.clone(),
                    level: ch.level,
                    cluster: cl.id,
                    anchored: cl.anchored,
                    supported: ch.support != crate::scene::Support::None,
                    velocity: v.to_array(),
                    deformation: cl.pose.transform_vector(state.u).to_array(),
                    stress: rows(sigma),
                    von_mises: von_mises(&sigma),
                    max_principal: symmetric_eigenvalues(&sigma)[0],
                    utilization: util[c],
                    damage: damage[c],
                });
            }
        }
        let impactors = self
            .impactors
            .iter()
            .map(|i| ImpactorView {
                name: i.name.clone(),
                shape: i.shape.clone(),
                center: i.pose.position.to_array(),
                rotation: rows(i.pose.rotation.to_mat3()),
                velocity: i.velocity.to_array(),
            })
            .collect();
        let e = &solver.energy;
        Snapshot {
            scene: self.scene.name.clone(),
            time: solver.time,
            frame: self.frame,
            chunks,
            bonds,
            impactors,
            ground: self.scene.ground.as_ref().map(|g| g.height),
            energy: EnergyView {
                mechanical: self.mechanical_energy(),
                dissipated: self.dissipated_energy(),
                bond_dissipation: e.bond_dissipation,
                contact_dissipation: self.contact.dissipated,
                kinetic: solver.kinetic_energy(),
                elastic: solver.elastic_energy(),
            },
        }
    }
}

/// `sym(r (x) f)`.
fn sym_outer(r: Vec3, f: Vec3) -> Mat3 {
    let o = Mat3::outer(r, f);
    (o + o.transpose()) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenvalues_of_a_rotated_diagonal() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalized(), 0.7).to_mat3();
        let s = q * Mat3::diag(Vec3::new(3.0, -1.0, 2.0)) * q.transpose();
        let e = symmetric_eigenvalues(&s);
        for (a, b) in e.iter().zip([3.0, 2.0, -1.0]) {
            assert!((a - b).abs() < 1e-12, "{e:?}");
        }
        // Uniaxial stress: von Mises equals the stress.
        assert!((von_mises(&Mat3::diag(Vec3::new(5.0, 0.0, 0.0))) - 5.0).abs() < 1e-12);
    }
}
