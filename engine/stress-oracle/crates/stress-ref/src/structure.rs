//! Static (per-scene) data of a destructible structure: chunks and bonds with their
//! mass, geometry, stiffness and strength resolved from the scene description.

use crate::bond::{BondGeometry, BondStiffness, Local6};
use crate::joint::{JointStrength, RebarParams};
use crate::material::{bond_strength_factor, Material};
use crate::contact::Part;
use crate::hull::ConvexHull;
use crate::math::{Mat3, Quat, Vec3};
use crate::scene::{box_mass_inertia, BodyDesc, Features, Scene, Support};

/// One face of a box chunk (body frame).
#[derive(Clone, Copy, Debug)]
pub struct Face {
    /// Outward unit normal.
    pub normal: Vec3,
    /// Face centre relative to the chunk centre.
    pub offset: Vec3,
    pub area: f64,
}

#[derive(Clone, Debug)]
pub struct ChunkData {
    pub center: Vec3,
    pub half_extents: Vec3,
    /// Orientation of the box in the body frame.
    pub rotation: Mat3,
    pub mass: f64,
    /// Inertia about the chunk centre, body frame.
    pub inertia: Mat3,
    pub inv_inertia: Mat3,
    pub material: String,
    /// Physical Young's modulus and Poisson ratio of the chunk material (contact).
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub support: Support,
    pub groups: Vec<String>,
    pub level: u32,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Box: 6 faces (-x, +x, -y, +y, -z, +z); hull: its faces.
    pub faces: Vec<Face>,
    /// The convex hull of a hull chunk (chunk frame, centred), `None` for a box.
    pub hull: Option<std::sync::Arc<ConvexHull>>,
    /// The convex parts of a compound chunk (chunk frame); `faces` are then the bounding
    /// box's.
    pub parts: Option<std::sync::Arc<Vec<Part>>>,
    /// Solid volume (m^3).
    pub volume: f64,
    /// Bond indices touching this chunk.
    pub bonds: Vec<usize>,
}

impl ChunkData {
    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// The eight corners relative to the chunk centre (body frame).
    pub fn corners(&self) -> [Vec3; 8] {
        let h = self.half_extents;
        let mut out = [Vec3::ZERO; 8];
        for (i, c) in out.iter_mut().enumerate() {
            let s = Vec3::new(
                if i & 1 == 0 { -h.x } else { h.x },
                if i & 2 == 0 { -h.y } else { h.y },
                if i & 4 == 0 { -h.z } else { h.z },
            );
            *c = self.rotation * s;
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct BondData {
    pub geometry: BondGeometry,
    /// Stiffness with the scene's stiffness scale applied.
    pub stiffness: BondStiffness,
    pub strength: JointStrength,
    pub rebar: Option<RebarParams>,
    /// Weibull strength multiplier (1 for deterministic materials).
    pub weibull: f64,
    /// Dashpot coefficients per component (stiffness-proportional, `beta k`).
    pub damping: Local6,
    pub level: u32,
    pub groups: Vec<String>,
    /// Face of chunk `a` (and `b`) the bond covers.
    pub face_a: usize,
    pub face_b: usize,
    /// Joint material, buckling length and rebar, kept to re-derive the bond when it
    /// is re-attached to coarser chunks (multi-level refinement).
    pub material: Material,
    pub buckling_length: Option<f64>,
    pub rebar_spec: Option<(f64, Material)>,
}

/// Stiffness, strength, rebar and dashpots of a bond with the given geometry.
pub fn bond_physics(
    geometry: &BondGeometry,
    material: &Material,
    buckling_length: Option<f64>,
    rebar: Option<&(f64, Material)>,
    stiffness_scale: f64,
    reduced_mass: f64,
    features: &Features,
    methods: &crate::scene::Methods,
) -> (BondStiffness, JointStrength, Option<RebarParams>, Local6) {
    let stiffness = BondStiffness::new(geometry, material, stiffness_scale);
    let mut strength = JointStrength::new(material, geometry, buckling_length, stiffness_scale, features);
    strength.exact_patch = methods.exact_joint_patch;
    strength.patch_after_damage = methods.patch_after_damage;
    strength.exact_rate_filter = methods.exact_rate_filter;
    let rebar = rebar
        .filter(|_| features.rebar)
        .map(|(area, steel)| RebarParams::new(*area, steel, geometry.length, stiffness_scale));
    // Stiffness-proportional dashpots: damping ratio `zeta` at the bond's own axial
    // frequency, proportionally less for slower (global) modes.
    let omega = (stiffness.kn / reduced_mass).sqrt();
    let zeta = if features.damping { material.damping_ratio } else { 0.0 };
    let beta = 2.0 * zeta / omega;
    (stiffness, strength, rebar, stiffness.as_local().scale(beta))
}

/// Reduced mass of a bond's two chunks (a supported chunk counts as immovable).
pub fn reduced_mass(a: &ChunkData, b: &ChunkData) -> f64 {
    match (a.support, b.support) {
        (Support::None, Support::None) => a.mass * b.mass / (a.mass + b.mass),
        (Support::None, _) => a.mass,
        (_, Support::None) => b.mass,
        _ => a.mass.min(b.mass),
    }
}

#[derive(Clone, Debug)]
pub struct Structure {
    pub name: String,
    pub chunks: Vec<ChunkData>,
    pub bonds: Vec<BondData>,
    pub initial_position: Vec3,
    pub initial_rotation: Quat,
    pub initial_velocity: Vec3,
    pub initial_angular_velocity: Vec3,
    pub stiffness_scale: f64,
    /// Model switches the bonds were derived with (kept for re-derivation on refinement).
    pub features: Features,
    pub methods: crate::scene::Methods,
}

/// The face of a box whose outward normal is closest to `dir` (body frame).
fn face_index(faces: &[Face], dir: Vec3) -> usize {
    (0..faces.len()).max_by(|&i, &j| faces[i].normal.dot(dir).total_cmp(&faces[j].normal.dot(dir))).unwrap()
}

fn faces_of(half: Vec3, rot: Mat3) -> Vec<Face> {
    let mut faces = [Face { normal: Vec3::ZERO, offset: Vec3::ZERO, area: 0.0 }; 6];
    for axis in 0..3 {
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        let area = 4.0 * half[a] * half[b];
        for (s, sign) in [(0usize, -1.0f64), (1usize, 1.0f64)] {
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            let n = rot * n;
            faces[2 * axis + s] = Face { normal: n, offset: n * half[axis], area };
        }
    }
    faces.to_vec()
}

impl Structure {
    /// Build structure `index` of `scene` with its stiffness scale and Weibull seed.
    pub fn from_scene(scene: &Scene, index: usize) -> Structure {
        let body: &BodyDesc = &scene.bodies[index];
        let scale = scene.sim.stiffness_scale;
        let features = scene.sim.features;
        let methods = scene.sim.methods;
        let mut chunks: Vec<ChunkData> = body
            .chunks
            .iter()
            .map(|c| {
                let m = scene.material(&c.material);
                let rotation = Quat::from_wxyz(c.orientation).to_mat3();
                let half = Vec3::from_array(c.half_extents);
                let hull = c.hull.as_ref().map(|points| {
                    let pts: Vec<Vec3> = points.iter().map(|p| Vec3::from_array(*p)).collect();
                    std::sync::Arc::new(ConvexHull::new(&pts).expect("validated hull"))
                });
                let parts = c.compound.as_ref().map(|parts| {
                    std::sync::Arc::new(
                        parts
                            .iter()
                            .map(|part| {
                                // Centred on its centroid by translation (the hull is not rebuilt).
                                let hull = part.hull().expect("validated part");
                                let centroid = hull.unit_mass().centroid;
                                Part { offset: centroid, hull: std::sync::Arc::new(hull.translated(centroid)) }
                            })
                            .collect::<Vec<_>>(),
                    )
                });
                let (mass, inertia, volume, faces) = match (&hull, &parts) {
                    (None, Some(parts)) => {
                        // Each part's inertia about its centroid, moved to the chunk
                        // centre (the parts' centroid) by the parallel-axis theorem.
                        let (mut volume, mut local) = (0.0, Mat3::ZERO);
                        for p in parts.iter() {
                            let um = p.hull.unit_mass();
                            let r = p.offset;
                            volume += um.volume;
                            local += um.inertia + (Mat3::IDENTITY * r.dot(r) - Mat3::outer(r, r)) * um.volume;
                        }
                        let inertia = rotation * (local * m.density) * rotation.transpose();
                        (m.density * volume, inertia, volume, faces_of(half, rotation))
                    }
                    (None, None) => {
                        let (mass, inertia) = box_mass_inertia(c.half_extents, c.orientation, m.density);
                        (mass, inertia, 8.0 * half.x * half.y * half.z, faces_of(half, rotation))
                    }
                    (Some(h), _) => {
                        let um = h.unit_mass();
                        let faces = h
                            .faces
                            .iter()
                            .map(|f| Face { normal: rotation * f.normal, offset: rotation * f.centroid, area: f.area })
                            .collect();
                        let inertia = rotation * (um.inertia * m.density) * rotation.transpose();
                        (m.density * um.volume, inertia, um.volume, faces)
                    }
                };
                ChunkData {
                    center: Vec3::from_array(c.center),
                    half_extents: half,
                    rotation,
                    mass,
                    inertia,
                    inv_inertia: inertia.inverse().expect("chunk inertia is invertible"),
                    material: c.material.clone(),
                    youngs_modulus: m.youngs_modulus,
                    poisson_ratio: m.poisson_ratio,
                    support: c.support,
                    groups: c.groups.clone(),
                    level: c.level,
                    parent: c.parent,
                    children: Vec::new(),
                    faces,
                    hull,
                    parts,
                    volume,
                    bonds: Vec::new(),
                }
            })
            .collect();
        for i in 0..chunks.len() {
            if let Some(p) = chunks[i].parent {
                chunks[p].children.push(i);
            }
        }

        let mut bonds = Vec::with_capacity(body.bonds.len());
        for (bi, desc) in body.bonds.iter().enumerate() {
            let m: &Material = scene.material(&desc.material);
            let geometry = BondGeometry::from_desc(desc, &body.chunks);
            let rebar_spec = desc.rebar.as_ref().map(|r| (r.area, scene.material(&r.material).clone()));
            let mred = reduced_mass(&chunks[desc.a], &chunks[desc.b]);
            let (stiffness, strength, rebar, damping) =
                bond_physics(&geometry, m, desc.buckling_length, rebar_spec.as_ref(), scale, mred, &features, &methods);
            let weibull = if features.weibull { bond_strength_factor(m.weibull_modulus, scene.sim.seed, index, bi) } else { 1.0 };
            let face_a = face_index(&chunks[desc.a].faces, geometry.normal);
            let face_b = face_index(&chunks[desc.b].faces, -geometry.normal);
            chunks[desc.a].bonds.push(bi);
            chunks[desc.b].bonds.push(bi);
            bonds.push(BondData {
                geometry,
                stiffness,
                strength,
                rebar,
                weibull,
                damping,
                level: desc.level,
                groups: desc.groups.clone(),
                face_a,
                face_b,
                material: m.clone(),
                buckling_length: desc.buckling_length,
                rebar_spec,
            });
        }

        Structure {
            name: body.name.clone(),
            chunks,
            bonds,
            initial_position: Vec3::from_array(body.position),
            initial_rotation: Quat::from_wxyz(body.orientation),
            initial_velocity: Vec3::from_array(body.linear_velocity),
            initial_angular_velocity: Vec3::from_array(body.angular_velocity),
            stiffness_scale: scale,
            features,
            methods,
        }
    }

    pub fn bond_other(&self, bond: usize, chunk: usize) -> usize {
        let g = &self.bonds[bond].geometry;
        if g.a == chunk {
            g.b
        } else {
            g.a
        }
    }
}
