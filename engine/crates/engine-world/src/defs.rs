//! Creation definitions, after Box3D's `b3*Def`: defaults via `Default`, copied
//! on create, never referenced afterwards. A field the solver does not support
//! yet is accepted at its default and refused otherwise (`Unsupported`); it is
//! never silently ignored.

use crate::error::unsupported;
use engine_schema::convex::{CollisionFilter, ConvexHull};
use engine_schema::heightfield::HeightField;
use engine_schema::mass::MassProperties;
use engine_schema::Material;
use glam::{Mat3, Vec3};

/// How a body moves (Box3D `b3BodyType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BodyType {
    /// Never moves; infinite mass. The default, as in Box3D.
    #[default]
    Static,
    /// Moved by the user's velocity or target transform (not yet supported).
    Kinematic,
    /// Moved by the solver.
    Dynamic,
}

/// Which of a body's moves a frame reports ([`World::set_body_move_reports`]). A host that
/// draws the body from the exported GPU poses ([`World::export_poses`]) needs only its sleep
/// and wake changes, so the moves stay O(events) on the CPU. The simulation is the same either
/// way: this filters what [`Frame`] carries, nothing else.
///
/// [`World::set_body_move_reports`]: crate::World::set_body_move_reports
/// [`World::export_poses`]: crate::World::export_poses
/// [`Frame`]: crate::Frame
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoveReports {
    /// Every step it moves (the default).
    #[default]
    All,
    /// Only the steps it falls asleep, wakes, or is first reported.
    SleepWake,
    /// Never.
    None,
}

/// Axes a body may not translate along or rotate about (not yet supported).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionLocks {
    pub linear_x: bool,
    pub linear_y: bool,
    pub linear_z: bool,
    pub angular_x: bool,
    pub angular_y: bool,
    pub angular_z: bool,
}

impl MotionLocks {
    /// The kernels' six lock bits (linear x, y, z, angular x, y, z).
    pub fn bits(&self) -> u32 {
        [
            self.linear_x,
            self.linear_y,
            self.linear_z,
            self.angular_x,
            self.angular_y,
            self.angular_z,
        ]
        .iter()
        .enumerate()
        .map(|(k, &on)| (on as u32) << k)
        .sum()
    }

    pub fn from_bits(bits: u32) -> Self {
        let on = |k: u32| bits & (1 << k) != 0;
        Self {
            linear_x: on(0),
            linear_y: on(1),
            linear_z: on(2),
            angular_x: on(3),
            angular_y: on(4),
            angular_z: on(5),
        }
    }
}

/// A rigid body (`World::create_body`), Box3D `b3BodyDef`.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyDef {
    pub body_type: BodyType,
    /// Body origin, world.
    pub position: [f32; 3],
    /// Quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// Of the centre of mass.
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    /// Box3D damping (1/s): `v /= 1 + h c` each step.
    pub linear_damping: f32,
    pub angular_damping: f32,
    /// Multiplies gravity for this body.
    pub gravity_scale: f32,
    /// m/s below which the body may sleep. Not yet per body (the world's
    /// `AvbdParams::sleep_lin` applies); any value is refused.
    pub sleep_threshold: Option<f32>,
    /// Axes (world frame) the body may not move along or turn about.
    pub motion_locks: MotionLocks,
    /// Continuous collision (the kernels' `BODY_CCD`): uncapped contact
    /// reach, the fast broadphase list and a time-of-impact contact, so a
    /// fast body never tunnels (projectiles, debris, vehicles at speed).
    pub is_bullet: bool,
    /// Not yet per body (sleep is per world); `false` is refused.
    pub enable_sleep: bool,
    pub is_awake: bool,
    /// Not yet supported (true).
    pub is_enabled: bool,
    /// Not yet supported (false).
    pub allow_fast_rotation: bool,
    pub name: Option<String>,
    /// Echoed back with this body's poses and events.
    pub user_data: u64,
}

impl Default for BodyDef {
    fn default() -> Self {
        Self {
            body_type: BodyType::Static,
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            sleep_threshold: None,
            motion_locks: MotionLocks::default(),
            is_bullet: false,
            enable_sleep: true,
            is_awake: true,
            is_enabled: true,
            allow_fast_rotation: false,
            name: None,
            user_data: 0,
        }
    }
}

impl BodyDef {
    /// A dynamic body at `position` (the common case, spelled short).
    pub fn dynamic(position: [f32; 3]) -> Self {
        Self {
            body_type: BodyType::Dynamic,
            position,
            ..Self::default()
        }
    }

    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.linear_damping >= 0.0 && self.angular_damping >= 0.0,
            "negative damping"
        );
        anyhow::ensure!(
            self.gravity_scale.is_finite(),
            "gravity_scale must be finite"
        );
        if self.sleep_threshold.is_some() || !self.enable_sleep {
            return unsupported("per-body sleep settings (planned P3; sleep is per world)");
        }
        if self.allow_fast_rotation {
            return unsupported("allow_fast_rotation (planned P3)");
        }
        if !self.is_enabled {
            return unsupported("disabled bodies (is_enabled = false)");
        }
        Ok(())
    }
}

/// Surface response (Box3D `b3SurfaceMaterial`), set per shape. Rubble keeps a
/// 64-entry GPU table and interns distinct `(friction, restitution,
/// contact_offset)` triples into it; a 65th distinct one is an error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMaterial {
    pub friction: f32,
    pub restitution: f32,
    /// Not yet supported (0).
    pub rolling_resistance: f32,
    /// Conveyor surface speed; not yet supported (0).
    pub tangent_velocity: [f32; 3],
    /// The application's material, reported in hits and queries.
    pub user_material_id: u64,
    /// Debug draw colour (0: default).
    pub custom_color: u32,
    /// Rubble: speculative contact margin (m).
    pub contact_offset: f32,
}

impl Default for SurfaceMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.0,
            rolling_resistance: 0.0,
            tangent_velocity: [0.0; 3],
            user_material_id: 0,
            custom_color: 0,
            contact_offset: 0.01,
        }
    }
}

impl SurfaceMaterial {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.rolling_resistance != 0.0 {
            return unsupported("rolling_resistance (planned)");
        }
        if self.tangent_velocity != [0.0; 3] {
            return unsupported("tangent_velocity (planned)");
        }
        Ok(())
    }

    pub(crate) fn gpu(&self) -> Material {
        Material {
            friction: self.friction,
            restitution: self.restitution,
            contact_offset: self.contact_offset,
            _pad: 0.0,
        }
    }

    pub(crate) fn key(&self) -> [u32; 3] {
        [
            self.friction.to_bits(),
            self.restitution.to_bits(),
            self.contact_offset.to_bits(),
        ]
    }
}

/// Collision geometry for `World::create_*_shape`, in the body frame.
#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    /// Along local +Y between `center ± half_len`.
    Capsule {
        center: [f32; 3],
        radius: f32,
        half_len: f32,
    },
    /// Half extents, placed by `transform`.
    Box {
        half: [f32; 3],
        transform: crate::Transform,
    },
    /// A convex hull placed by `transform`.
    Hull {
        hull: ConvexHull,
        transform: crate::Transform,
    },
    /// The half-space below `normal · x = offset` (body frame), on static
    /// bodies only: an infinite ground. At most 8 per world.
    Plane {
        normal: [f32; 3],
        offset: f32,
    },
    /// Terrain: a grid of heights centred on `transform` (its +Y is up), on
    /// static bodies only. Heights can be edited in place
    /// (`World::set_height_field_region`).
    HeightField {
        field: HeightField,
        transform: crate::Transform,
    },
}

/// One shape of a body (Box3D `b3ShapeDef`).
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeDef {
    pub material: SurfaceMaterial,
    /// kg/m³.
    pub density: f32,
    pub filter: CollisionFilter,
    /// Overlap events without collision response (planned P4).
    pub is_sensor: bool,
    pub enable_sensor_events: bool,
    pub enable_contact_events: bool,
    pub enable_hit_events: bool,
    /// Recompute the body's mass properties from its shapes (Box3D default).
    pub update_body_mass: bool,
    pub name: Option<String>,
    pub user_data: u64,
}

impl Default for ShapeDef {
    fn default() -> Self {
        Self {
            material: SurfaceMaterial::default(),
            density: 1.0,
            filter: CollisionFilter::default(),
            is_sensor: false,
            enable_sensor_events: false,
            enable_contact_events: false,
            enable_hit_events: false,
            update_body_mass: true,
            name: None,
            user_data: 0,
        }
    }
}

impl ShapeDef {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        self.material.validate()?;
        anyhow::ensure!(self.density >= 0.0, "negative density");
        Ok(())
    }
}

/// Mass, centre of mass (body frame) and rotational inertia about the centre
/// (body frame), Box3D `b3MassData`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassData {
    pub mass: f32,
    pub center: [f32; 3],
    /// Inertia tensor about `center`, row by row (symmetric).
    pub rotational_inertia: [[f32; 3]; 3],
}

impl MassData {
    pub(crate) fn tensor(&self) -> Mat3 {
        Mat3::from_cols_array_2d(&self.rotational_inertia)
    }

    /// A body with principal inertia `inertia` about `center`, axes aligned.
    pub fn principal(mass: f32, center: [f32; 3], inertia: [f32; 3]) -> Self {
        Self {
            mass,
            center,
            rotational_inertia: Mat3::from_diagonal(Vec3::from(inertia)).to_cols_array_2d(),
        }
    }
}

/// Mass properties of one shape: mass, centre (body frame) and tensor about it.
pub(crate) fn geometry_mass(g: &Geometry, density: f32) -> (f32, Vec3, Mat3) {
    match g {
        Geometry::Sphere { center, radius } => {
            let m = MassProperties::sphere(*radius, density);
            (m.mass, Vec3::from(*center), Mat3::from_diagonal(m.inertia))
        }
        Geometry::Capsule {
            center,
            radius,
            half_len,
        } => {
            let m = MassProperties::capsule(*radius, *half_len, density);
            (m.mass, Vec3::from(*center), Mat3::from_diagonal(m.inertia))
        }
        Geometry::Box { half, transform } => {
            let m = MassProperties::cuboid(Vec3::from(*half), density);
            let r = Mat3::from_quat(transform.q());
            (
                m.mass,
                transform.p(),
                rotate_tensor(r, Mat3::from_diagonal(m.inertia)),
            )
        }
        Geometry::Plane { .. } | Geometry::HeightField { .. } => (0.0, Vec3::ZERO, Mat3::ZERO),
        Geometry::Hull { hull, transform } => {
            let h = hull.mass_properties(density);
            let r = Mat3::from_quat(transform.q());
            (
                h.mass,
                transform.p() + r * h.center_of_mass,
                rotate_tensor(r, h.inertia),
            )
        }
    }
}

/// `R I R^T`, skipped for the identity so axis-aligned shapes stay exact.
fn rotate_tensor(r: Mat3, i: Mat3) -> Mat3 {
    if r == Mat3::IDENTITY {
        i
    } else {
        r * i * r.transpose()
    }
}

/// Combine shape masses: total mass, centre, tensor about the centre. A single
/// shape keeps its own values unchanged.
pub(crate) fn combine(parts: &[(f32, Vec3, Mat3)]) -> Option<MassData> {
    let mass: f32 = parts.iter().map(|p| p.0).sum();
    if mass <= 0.0 {
        return None;
    }
    if let [(m, c, i)] = parts {
        return Some(MassData {
            mass: *m,
            center: c.to_array(),
            rotational_inertia: i.to_cols_array_2d(),
        });
    }
    let center = parts.iter().map(|p| p.1 * p.0).sum::<Vec3>() / mass;
    let mut tensor = Mat3::ZERO;
    for &(m, c, i) in parts {
        let d = c - center;
        let shift = Mat3::from_diagonal(Vec3::splat(d.length_squared())) - outer(d, d);
        tensor += i + shift * m;
    }
    Some(MassData {
        mass,
        center: center.to_array(),
        rotational_inertia: tensor.to_cols_array_2d(),
    })
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transform;

    #[test]
    fn box_mass_matches_the_formula() {
        let g = Geometry::Box {
            half: [0.5, 1.0, 1.5],
            transform: Transform::IDENTITY,
        };
        let m = combine(&[geometry_mass(&g, 2.0)]).unwrap();
        let e = MassProperties::cuboid(Vec3::new(0.5, 1.0, 1.5), 2.0);
        assert_eq!(m.mass, e.mass);
        assert_eq!(m.center, [0.0; 3]);
        let t = m.tensor();
        assert_eq!(Vec3::new(t.x_axis.x, t.y_axis.y, t.z_axis.z), e.inertia);
    }

    #[test]
    fn off_centre_shapes_give_a_shifted_centre() {
        let a = geometry_mass(
            &Geometry::Sphere {
                center: [1.0, 0.0, 0.0],
                radius: 0.5,
            },
            1.0,
        );
        let b = geometry_mass(
            &Geometry::Sphere {
                center: [3.0, 0.0, 0.0],
                radius: 0.5,
            },
            1.0,
        );
        let m = combine(&[a, b]).unwrap();
        assert!((m.center[0] - 2.0).abs() < 1e-6);
        let one = MassProperties::sphere(0.5, 1.0);
        let t = m.tensor();
        assert!((t.y_axis.y - 2.0 * (one.inertia.y + one.mass)).abs() / t.y_axis.y < 1e-5);
    }
}
