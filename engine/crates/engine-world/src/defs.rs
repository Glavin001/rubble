//! Creation descriptors with defaults (`..Default::default()`); copied on
//! create, never referenced afterwards.

use crate::{BodyId, MaterialId};
use engine_schema::convex::{CollisionFilter, ConvexHull};
use engine_schema::mass::MassProperties;
use engine_schema::{JointDesc, Material};
use glam::{Mat3, Quat, Vec3};

/// How a body moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BodyKind {
    /// Never moves; infinite mass.
    Static,
    /// Moved by the solver.
    #[default]
    Dynamic,
}

/// Where a dynamic body's mass properties come from.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MassSpec {
    /// Sum of the shapes' masses from their densities, about the body origin
    /// (which must then be the centre of mass, up to `MASS_ORIGIN_TOLERANCE`).
    #[default]
    FromShapes,
    /// Shapes' mass distribution scaled to this total mass (kg).
    Mass(f32),
    /// Explicit mass (kg) and principal inertia (kg m²) in the body frame.
    Explicit { mass: f32, inertia: [f32; 3] },
}

/// A rigid body (`World::create_body`).
#[derive(Clone, Debug)]
pub struct BodyDef {
    pub kind: BodyKind,
    pub position: [f32; 3],
    /// Quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: MassSpec,
    /// Start asleep (woken by contact, a joint or a command).
    pub asleep: bool,
    /// Echoed back with this body's poses and events.
    pub user_data: u64,
}

impl Default for BodyDef {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: MassSpec::FromShapes,
            asleep: false,
            user_data: 0,
        }
    }
}

/// Collision geometry, in the shape's local frame.
#[derive(Clone, Debug)]
pub enum Shape {
    Sphere {
        radius: f32,
    },
    /// Half extents.
    Box {
        half: [f32; 3],
    },
    /// Along local +Y: a segment of half length `half_len` swept by `radius`.
    Capsule {
        radius: f32,
        half_len: f32,
    },
    /// A convex hull, in the shape frame (`local_position`/`local_rotation`).
    Convex(ConvexHull),
}

/// One collider of a body (`World::create_body`).
#[derive(Clone, Debug)]
pub struct ShapeDef {
    pub shape: Shape,
    pub local_position: [f32; 3],
    pub local_rotation: [f32; 4],
    /// kg/m³, for `MassSpec::FromShapes` / `MassSpec::Mass`.
    pub density: f32,
    pub material: MaterialId,
    pub filter: CollisionFilter,
}

impl Default for ShapeDef {
    fn default() -> Self {
        Self {
            shape: Shape::Box { half: [0.5; 3] },
            local_position: [0.0; 3],
            local_rotation: [0.0, 0.0, 0.0, 1.0],
            density: 1000.0,
            material: MaterialId::DEFAULT,
            filter: CollisionFilter::default(),
        }
    }
}

impl ShapeDef {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            ..Self::default()
        }
    }
}

/// Surface response (`World::create_material`). Friction combines per contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialDef {
    pub friction: f32,
    pub restitution: f32,
    /// Speculative contact margin (m).
    pub contact_offset: f32,
}

impl Default for MaterialDef {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
        }
    }
}

impl From<MaterialDef> for Material {
    fn from(m: MaterialDef) -> Self {
        Material {
            friction: m.friction,
            restitution: m.restitution,
            contact_offset: m.contact_offset,
            _pad: 0.0,
        }
    }
}

/// A joint between two bodies (`World::create_joint`): any `JointDesc` preset
/// (`JointDesc::ball`, `hinge`, `fixed`, `d6`, ... with limits, drives and
/// break thresholds). Its `body_a` / `body_b` indices are replaced by these.
#[derive(Clone, Debug)]
pub struct JointDef {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub desc: JointDesc,
}

/// Off-axis inertia smaller than this fraction of the largest moment is
/// treated as zero when shapes are combined; anything larger needs
/// `MassSpec::Explicit` (the solver takes principal moments in the body frame).
pub const PRINCIPAL_TOLERANCE: f32 = 1e-4;

/// The combined centre of mass may sit this far (m) from the body origin.
pub const MASS_ORIGIN_TOLERANCE: f32 = 1e-4;

/// Mass and principal inertia of `shapes` about the body origin.
pub(crate) fn shapes_mass(shapes: &[ShapeDef]) -> anyhow::Result<(f32, Vec3)> {
    let mut mass = 0.0f32;
    let mut moment = Vec3::ZERO;
    let mut tensor = Mat3::ZERO;
    for s in shapes {
        // Each shape's mass, centre (shape frame) and tensor about that centre.
        let (m, centre, inertia) = match &s.shape {
            Shape::Sphere { radius } => principal(MassProperties::sphere(*radius, s.density)),
            Shape::Box { half } => principal(MassProperties::cuboid(Vec3::from(*half), s.density)),
            Shape::Capsule { radius, half_len } => {
                principal(MassProperties::capsule(*radius, *half_len, s.density))
            }
            Shape::Convex(hull) => {
                let h = hull.mass_properties(s.density);
                (h.mass, h.center_of_mass, h.inertia)
            }
        };
        let r = Mat3::from_quat(Quat::from_array(s.local_rotation));
        let p = Vec3::from(s.local_position) + r * centre;
        // Rotate the tensor into the body frame, then shift it to the origin.
        let rotated = r * inertia * r.transpose();
        let shift = Mat3::from_diagonal(Vec3::splat(p.length_squared())) - outer(p, p);
        tensor += rotated + shift * m;
        mass += m;
        moment += p * m;
    }
    anyhow::ensure!(
        mass > 0.0,
        "a dynamic body needs shapes with positive density or MassSpec::Explicit"
    );
    anyhow::ensure!(
        (moment / mass).length() <= MASS_ORIGIN_TOLERANCE,
        "the shapes' centre of mass is {:?} from the body origin: place the shapes about \
         their centre of mass or use MassSpec::Explicit",
        moment / mass
    );
    let diag = Vec3::new(tensor.x_axis.x, tensor.y_axis.y, tensor.z_axis.z);
    let off = tensor
        .y_axis
        .x
        .abs()
        .max(tensor.z_axis.x.abs())
        .max(tensor.z_axis.y.abs());
    anyhow::ensure!(
        off <= PRINCIPAL_TOLERANCE * diag.max_element(),
        "the shapes' inertia is not diagonal in the body frame (off-axis {off}): \
         align the body frame with the principal axes or use MassSpec::Explicit"
    );
    Ok((mass, diag))
}

fn principal(m: MassProperties) -> (f32, Vec3, Mat3) {
    (m.mass, Vec3::ZERO, Mat3::from_diagonal(m.inertia))
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_box_matches_the_formula() {
        let (m, i) = shapes_mass(&[ShapeDef {
            density: 2.0,
            ..ShapeDef::new(Shape::Box {
                half: [0.5, 1.0, 1.5],
            })
        }])
        .unwrap();
        let expect = MassProperties::cuboid(Vec3::new(0.5, 1.0, 1.5), 2.0);
        assert!((m - expect.mass).abs() < 1e-5);
        assert!((i - expect.inertia).abs().max_element() < 1e-4);
    }

    #[test]
    fn symmetric_dumbbell_uses_the_parallel_axis_shift() {
        let ball = |x: f32| ShapeDef {
            local_position: [x, 0.0, 0.0],
            ..ShapeDef::new(Shape::Sphere { radius: 0.5 })
        };
        let (m, i) = shapes_mass(&[ball(-2.0), ball(2.0)]).unwrap();
        let one = MassProperties::sphere(0.5, 1000.0);
        assert!((m - 2.0 * one.mass).abs() < 1e-2);
        assert!((i.x - 2.0 * one.inertia.x).abs() < 1e-2);
        assert!((i.y - 2.0 * (one.inertia.y + one.mass * 4.0)).abs() / i.y < 1e-5);
    }

    #[test]
    fn off_centre_shapes_need_explicit_mass() {
        let s = ShapeDef {
            local_position: [1.0, 0.0, 0.0],
            ..ShapeDef::default()
        };
        assert!(shapes_mass(&[s]).is_err());
    }
}
