//! Scenes stored as raw GPU records (`engine_schema::WorldSnapshot`, the
//! RUBSNAP1 files `engine-scenes`' `scene_export` writes), rebuilt through the
//! world's own calls: bodies, shapes, mass and joints become handles, so the
//! scene behaves like one built with `create_body` / `create_shape` /
//! `create_joint`.

use crate::{
    BodyDef, BodyId, BodyType, Geometry, JointDef, JointId, MassData, MotionLocks, ShapeDef,
    ShapeId, SurfaceMaterial, Transform, World,
};
use engine_schema::{
    convex::CollisionFilter, BodyFlags, JointDesc, ShapeType, BODY_CCD, BODY_LOCK_SHIFT,
    COLLIDER_CONTACT_EVENTS, COLLIDER_HIT_EVENTS, COLLIDER_SENSOR,
};

/// Handles of an imported scene, by record index.
#[derive(Clone, Debug, Default)]
pub struct SnapshotImport {
    /// One per body record.
    pub bodies: Vec<BodyId>,
    /// One per collider record.
    pub shapes: Vec<ShapeId>,
    /// One per joint record; `None` for a broken record (a slot the scene
    /// keeps for later, such as a released drag joint): it constrains nothing.
    pub joints: Vec<Option<JointId>>,
}

/// The six lock bits above `BODY_LOCK_SHIFT`.
const LOCK_BITS: u32 = 0x3F;

impl World {
    /// Build `snap`'s bodies, shapes and joints in this world, in record
    /// order, and apply its solver settings when it carries them.
    ///
    /// Each body record becomes a body at its record pose (the solver frame:
    /// the centre of mass, principal axes) with its velocities, sleep state
    /// and continuous-collision flag. A dynamic record's inverse mass and
    /// inverse principal inertia become its mass data, centred on that pose.
    /// Each collider becomes a shape with its record's geometry, density,
    /// filter, events and material (`snap.materials[material slot]`); joints
    /// come back through [`JointDesc::from_gpu`]. Records a description cannot
    /// express (hull or heightfield colliders, rotated capsules, joints
    /// carrying solver state) are refused with the reason.
    pub fn import_snapshot(
        &mut self,
        snap: &engine_schema::WorldSnapshot,
    ) -> anyhow::Result<SnapshotImport> {
        if let Some(p) = snap.params {
            self.set_solver(p);
        }
        let mut defs = Vec::with_capacity(snap.bodies.len());
        let mut masses = Vec::with_capacity(snap.bodies.len());
        for (i, b) in snap.bodies.iter().enumerate() {
            let dynamic = b.is_dynamic();
            let kinematic = !dynamic && b.flags & BodyFlags::Kinematic as u32 != 0;
            let body_type = if dynamic {
                BodyType::Dynamic
            } else if kinematic {
                BodyType::Kinematic
            } else {
                BodyType::Static
            };
            if dynamic {
                let inertia = b
                    .inv_inertia
                    .map(|v| if v > 0.0 { 1.0 / v } else { f32::INFINITY });
                anyhow::ensure!(
                    inertia.iter().all(|v| v.is_finite()),
                    "body record {i}: zero inverse inertia (an infinite moment) is not a mass \
                     World takes"
                );
                masses.push(Some(MassData::principal(
                    1.0 / b.inv_mass,
                    [0.0; 3],
                    inertia,
                )));
            } else {
                masses.push(None);
            }
            defs.push(BodyDef {
                body_type,
                position: b.position,
                rotation: b.orientation,
                linear_velocity: b.linear_vel,
                angular_velocity: b.angular_vel,
                motion_locks: MotionLocks::from_bits((b.flags >> BODY_LOCK_SHIFT) & LOCK_BITS),
                is_bullet: b.flags & BODY_CCD != 0,
                is_awake: b.flags & BodyFlags::Sleeping as u32 == 0,
                ..BodyDef::default()
            });
        }
        let bodies = self.create_bodies(&defs)?;
        for (id, mass) in bodies.iter().zip(masses) {
            if let Some(m) = mass {
                self.body_set_mass_data(*id, m)?;
            }
        }
        let mut shapes = Vec::with_capacity(snap.colliders.len());
        for (i, c) in snap.colliders.iter().enumerate() {
            let body = *bodies.get(c.body as usize).ok_or_else(|| {
                anyhow::anyhow!("collider record {i}: body {} out of range", c.body)
            })?;
            let slot = (c.material & 0x0FFF) as usize;
            let m = snap.materials.get(slot).ok_or_else(|| {
                anyhow::anyhow!("collider record {i}: material slot {slot} out of the table")
            })?;
            let def = ShapeDef {
                material: SurfaceMaterial {
                    friction: m.friction,
                    restitution: m.restitution,
                    contact_offset: m.contact_offset,
                    ..SurfaceMaterial::default()
                },
                density: c.density,
                filter: CollisionFilter::of(c),
                is_sensor: c.material & COLLIDER_SENSOR != 0,
                enable_contact_events: c.material & COLLIDER_CONTACT_EVENTS != 0,
                enable_hit_events: c.material & COLLIDER_HIT_EVENTS != 0,
                update_body_mass: false,
                ..ShapeDef::default()
            };
            let identity = c.local_rot == [0.0, 0.0, 0.0, 1.0];
            let geometry = match c.shape {
                s if s == ShapeType::Sphere as u32 => Geometry::Sphere {
                    center: c.local_pos,
                    radius: c.params[0],
                },
                s if s == ShapeType::Capsule as u32 => {
                    anyhow::ensure!(
                        identity,
                        "collider record {i}: a rotated capsule (Geometry::Capsule runs along Y)"
                    );
                    Geometry::Capsule {
                        center: c.local_pos,
                        radius: c.params[0],
                        half_len: c.params[1],
                    }
                }
                s if s == ShapeType::Box as u32 => Geometry::Box {
                    half: [c.params[0], c.params[1], c.params[2]],
                    transform: Transform::new(c.local_pos, c.local_rot),
                },
                s if s == ShapeType::Plane as u32 => {
                    // World-space plane on a static body: back into the body's frame.
                    let b = &snap.bodies[c.body as usize];
                    let q = glam::Quat::from_array(b.orientation);
                    let n = glam::Vec3::new(c.params[0], c.params[1], c.params[2]);
                    let p = glam::Vec3::from(b.position);
                    Geometry::Plane {
                        normal: (q.conjugate() * n).to_array(),
                        offset: c.params[3] - n.dot(p),
                    }
                }
                s => anyhow::bail!(
                    "collider record {i}: shape type {s} (hulls, planes and heightfields need \
                     their arena data, which a snapshot does not carry)"
                ),
            };
            shapes.push(self.create_shape(body, &def, geometry)?);
        }
        let mut joints = Vec::with_capacity(snap.joints.len());
        for (i, j) in snap.joints.iter().enumerate() {
            if j.is_broken() {
                joints.push(None);
                continue;
            }
            let desc =
                JointDesc::from_gpu(j).map_err(|e| anyhow::anyhow!("joint record {i}: {e}"))?;
            let end = |b: u32| {
                bodies
                    .get(b as usize)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("joint record {i}: body {b} out of range"))
            };
            let def = JointDef::Custom {
                body_a: end(j.body_a)?,
                body_b: end(j.body_b)?,
                desc,
                user_data: 0,
            };
            joints.push(Some(self.create_joint(&def)?));
        }
        Ok(SnapshotImport {
            bodies,
            shapes,
            joints,
        })
    }
}
