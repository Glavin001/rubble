//! Scene building shared by every host: [`SceneBuilder`] writes bodies, colliders,
//! hulls and joints into any [`SceneSink`]. The Metal world implements it (engine-metal),
//! and so does [`HostWorld`], a plain host-side world that records a scene as a
//! [`WorldSnapshot`] without a GPU, which is how `scene_export` writes the browser's
//! scenes on any platform.

use crate::convex::{ConvexHull, HullArena};
use crate::snapshot::WorldSnapshot;
use crate::*;
use glam::{Quat, Vec3};

/// What scene builders write into.
pub trait SceneSink {
    fn add_body(&mut self, state: BodyState) -> u32;
    fn add_collider(&mut self, col: Collider) -> u32;
    /// Base record of `hull` in the world's hull arena.
    fn insert_hull(&mut self, hull: &ConvexHull) -> u32;
    fn body(&self, id: u32) -> BodyState;
    fn set_body(&mut self, id: u32, state: BodyState);
    fn add_joint_desc(&mut self, desc: &JointDesc) -> u32;
    fn add_material(&mut self, m: Material) -> u32;
    fn set_default_friction(&mut self, friction: f32);
    fn force_sleep(&mut self, id: u32);
}

/// The material table a world starts with: entry 0 is the default material.
pub fn default_materials() -> Vec<Material> {
    let mut m = vec![
        Material {
            friction: 0.0,
            restitution: 0.0,
            contact_offset: 0.0,
            _pad: 0.0
        };
        64
    ];
    m[0] = Material {
        friction: 0.5,
        restitution: 0.0,
        contact_offset: 0.01,
        _pad: 0.0,
    };
    m
}

/// A scene recorded on the host, with the Metal world's bookkeeping (default material,
/// torque arms from collider reach, sleeping), for export as a [`WorldSnapshot`].
pub struct HostWorld {
    pub bodies: Vec<BodyState>,
    pub colliders: Vec<Collider>,
    pub materials: Vec<Material>,
    material_count: u32,
    pub joints: Vec<Joint>,
    pub hulls: HullArena,
    pub params: AvbdParams,
    body_radius: Vec<f32>,
}

impl HostWorld {
    pub fn new(params: AvbdParams) -> Self {
        Self {
            bodies: Vec::new(),
            colliders: Vec::new(),
            materials: default_materials(),
            material_count: 1,
            joints: Vec::new(),
            hulls: HullArena::default(),
            params,
            body_radius: Vec::new(),
        }
    }

    /// The scene as the Metal world's `snapshot()` reports it after building.
    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            bodies: self.bodies.clone(),
            colliders: self.colliders.clone(),
            materials: self.materials.clone(),
            joints: self.joints.clone(),
            params: Some(self.params),
        }
    }
}

impl SceneSink for HostWorld {
    fn add_body(&mut self, state: BodyState) -> u32 {
        self.bodies.push(state);
        self.bodies.len() as u32 - 1
    }
    fn add_collider(&mut self, col: Collider) -> u32 {
        if let Some(r) = collider_reach(&col) {
            let b = col.body as usize;
            if self.body_radius.len() <= b {
                self.body_radius.resize(b + 1, 0.0);
            }
            self.body_radius[b] = self.body_radius[b].max(r);
        }
        self.colliders.push(col);
        self.colliders.len() as u32 - 1
    }
    fn insert_hull(&mut self, hull: &ConvexHull) -> u32 {
        self.hulls.insert(hull).expect("hull arena range")
    }
    fn body(&self, id: u32) -> BodyState {
        self.bodies[id as usize]
    }
    fn set_body(&mut self, id: u32, state: BodyState) {
        self.bodies[id as usize] = state;
    }
    fn add_joint_desc(&mut self, desc: &JointDesc) -> u32 {
        let mut j = desc.to_gpu();
        if desc.torque_arm <= 0.0 {
            let r = |id: u32| {
                if self.bodies[id as usize].inv_mass == 0.0 {
                    0.0
                } else {
                    self.body_radius.get(id as usize).copied().unwrap_or(0.0)
                }
            };
            j.torque_arm = default_torque_arm(r(desc.body_a), r(desc.body_b));
        }
        self.joints.push(j);
        self.joints.len() as u32 - 1
    }
    fn add_material(&mut self, m: Material) -> u32 {
        let id = self.material_count;
        self.materials[id as usize] = m;
        self.material_count += 1;
        id
    }
    fn set_default_friction(&mut self, friction: f32) {
        self.materials[0].friction = friction;
    }
    fn force_sleep(&mut self, id: u32) {
        let mut b = self.body(id);
        b.flags = (b.flags & !(BodyFlags::Awake as u32)) | BodyFlags::Sleeping as u32;
        b.linear_vel = [0.0; 3];
        b.angular_vel = [0.0; 3];
        b.sleep_timer = self.params.sleep_time;
        self.set_body(id, b);
    }
}

/// Host-side scene helpers for demos and benches.
pub struct SceneBuilder<'a> {
    world: &'a mut dyn SceneSink,
}

impl<'a> SceneBuilder<'a> {
    pub fn new(world: &'a mut dyn SceneSink) -> Self {
        Self { world }
    }

    pub fn world_mut(&mut self) -> &mut dyn SceneSink {
        self.world
    }

    pub fn add_ground_plane(&mut self, y: f32) -> u32 {
        let body = self.world.add_body(BodyState {
            position: [0.0, y, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, y],
        });
        body
    }

    /// Static box ground (three-avbd `ground(extent)`). Top face at `top_y`.
    /// `half_xz` is the horizontal half-extent (three-avbd: max(100, 2·extent)).
    /// Thickness is deep so Jacobi soft contacts can't tunnel through a thin slab.
    pub fn add_ground_box(&mut self, top_y: f32) -> u32 {
        self.add_ground_box_extent(top_y, 200.0)
    }

    pub fn add_ground_box_extent(&mut self, top_y: f32, half_xz: f32) -> u32 {
        // Top face matches three-avbd (y=0.5). Thickness is deeper than their size.z=1
        // slab: our Jacobi soft contacts otherwise tunnel through a thin floor at scale.
        let half_h = 25.0;
        let hx = half_xz.max(50.0);
        let body = self.world.add_body(BodyState {
            position: [0.0, top_y - half_h, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [hx, half_h, hx, hx.max(half_h)],
        });
        body
    }

    pub fn add_sphere(&mut self, pos: Vec3, radius: f32, mass: f32) -> u32 {
        let inertia = 0.4 * mass * radius * radius;
        let body = self.world.add_body(BodyState::dynamic(
            pos,
            Quat::IDENTITY,
            mass,
            Vec3::splat(inertia),
        ));
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Sphere as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: mass / ((4.0 / 3.0) * std::f32::consts::PI * radius.powi(3)),
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [radius, 0.0, 0.0, radius],
        });
        body
    }

    pub fn add_box(&mut self, pos: Vec3, half: Vec3, mass: f32) -> u32 {
        let h = half * 2.0;
        let inertia = Vec3::new(
            (1.0 / 12.0) * mass * (h.y * h.y + h.z * h.z),
            (1.0 / 12.0) * mass * (h.x * h.x + h.z * h.z),
            (1.0 / 12.0) * mass * (h.x * h.x + h.y * h.y),
        );
        let body = self
            .world
            .add_body(BodyState::dynamic(pos, Quat::IDENTITY, mass, inertia));
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: mass / (8.0 * half.x * half.y * half.z),
            local_rot: [0.0, 0.0, 0.0, 1.0],
            // params.w = bounding-sphere radius (three-avbd length(half)), used by BP sphere cull.
            params: [half.x, half.y, half.z, half.length()],
        });
        body
    }

    pub fn add_capsule(&mut self, pos: Vec3, radius: f32, half_len: f32, mass: f32) -> u32 {
        let cyl = std::f32::consts::PI * radius * radius * (2.0 * half_len);
        let sph = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
        let vol = (cyl + sph).max(1e-8);
        let body = self.world.add_body(BodyState::dynamic(
            pos,
            Quat::IDENTITY,
            mass,
            Vec3::splat(0.2 * mass * (half_len + radius).powi(2)),
        ));
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Capsule as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: mass / vol,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [radius, half_len, 0.0, radius + half_len],
        });
        body
    }

    /// Dynamic body made of convex hulls (one hull, or a compound). Hulls are in
    /// one input frame; `pos`/`rot` place that frame. Mass and inertia are the
    /// compound's, from uniform `density`; children keep their own colliders.
    pub fn add_convex_hull(
        &mut self,
        hulls: &[crate::convex::ConvexHull],
        pos: Vec3,
        rot: Quat,
        density: f32,
    ) -> u32 {
        let cooked = crate::convex::cook_body(hulls, density);
        let (p, q) = cooked.body_pose(pos, rot);
        let body = self
            .world
            .add_body(BodyState::dynamic(p, q, cooked.mass, cooked.inertia));
        for child in &cooked.children {
            let base = self.world.insert_hull(&child.hull);
            self.world.add_collider(Collider {
                body,
                shape: ShapeType::ConvexHull as u32,
                material: 0,
                generation: 0,
                local_pos: child.local_pos.to_array(),
                density,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: crate::convex::convex_collider_params(
                    base,
                    1.0,
                    child.hull.bounding_radius(),
                ),
            });
        }
        body
    }

    pub fn add_compound_box_sphere(
        &mut self,
        pos: Vec3,
        box_half: Vec3,
        sphere_offset: Vec3,
        sphere_radius: f32,
        mass: f32,
    ) -> u32 {
        let body = self.world.add_body(BodyState::dynamic(
            pos,
            Quat::IDENTITY,
            mass,
            Vec3::splat(0.15 * mass),
        ));
        let box_vol = 8.0 * box_half.x * box_half.y * box_half.z;
        let sph_vol = (4.0 / 3.0) * std::f32::consts::PI * sphere_radius.powi(3);
        let dens = mass / (box_vol + sph_vol).max(1e-8);
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: dens,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [box_half.x, box_half.y, box_half.z, box_half.length()],
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Sphere as u32,
            material: 0,
            generation: 0,
            local_pos: sphere_offset.to_array(),
            density: dens,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [sphere_radius, 0.0, 0.0, sphere_radius],
        });
        body
    }

    /// Add a joint at rest in the bodies' current orientations (frame B is
    /// derived from frame A, three-avbd `rest: 'current'`).
    pub fn add_joint(&mut self, desc: JointDesc) -> u32 {
        let qa = Quat::from_array(self.world.body(desc.body_a).orientation);
        let qb = Quat::from_array(self.world.body(desc.body_b).orientation);
        self.world.add_joint_desc(&desc.rest_from_bodies(qa, qb))
    }

    /// Body-local frames for a joint placed at a world anchor with a world
    /// axis (hinge / slider / twist axis = frame X).
    pub fn world_frames(
        &self,
        a: u32,
        b: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> (JointFrame, JointFrame) {
        let ba = self.world.body(a);
        let bb = self.world.body(b);
        let qa = Quat::from_array(ba.orientation);
        let qb = Quat::from_array(bb.orientation);
        let la = qa.conjugate() * (anchor - Vec3::from_array(ba.position));
        let lb = qb.conjugate() * (anchor - Vec3::from_array(bb.position));
        let axis_a = qa.conjugate() * axis;
        (JointFrame::with_axis(la, axis_a), JointFrame::at(lb))
    }

    pub fn add_ball_joint(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> u32 {
        self.add_joint(JointDesc::ball(a, b, anchor_a, anchor_b))
    }

    pub fn add_fixed_joint(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> u32 {
        self.add_joint(JointDesc::fixed(
            a,
            b,
            JointFrame::at(anchor_a),
            JointFrame::at(anchor_b),
        ))
    }

    /// Prismatic joint; `axis` is in body A's local frame.
    pub fn add_slider(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> u32 {
        self.add_joint(JointDesc::slider(
            a,
            b,
            JointFrame::with_axis(anchor_a, axis),
            JointFrame::at(anchor_b),
        ))
    }

    /// D6 with every DOF locked except twist about `axis` (body A local);
    /// adjust motions with `add_joint(JointDesc::d6(..).with_motion(..))`.
    pub fn add_d6(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, axis: Vec3) -> u32 {
        self.add_joint(
            JointDesc::d6(
                a,
                b,
                JointFrame::with_axis(anchor_a, axis),
                JointFrame::at(anchor_b),
            )
            .with_motion(Dof::Twist, Motion::Free),
        )
    }

    /// Revolute joint about `axis` (body A local).
    pub fn add_hinge(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, axis: Vec3) -> u32 {
        self.add_joint(JointDesc::hinge(
            a,
            b,
            JointFrame::with_axis(anchor_a, axis),
            JointFrame::at(anchor_b),
        ))
    }

    pub fn add_distance(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        length: f32,
    ) -> u32 {
        self.add_joint(JointDesc::distance(a, b, anchor_a, anchor_b, length))
    }

    pub fn add_spring(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest: f32,
        stiffness: f32,
    ) -> u32 {
        self.add_joint(JointDesc::spring(a, b, anchor_a, anchor_b, rest, stiffness))
    }

    pub fn pile_spheres(&mut self, n: u32, spacing: f32) {
        let side = (n as f32).cbrt().ceil() as u32;
        let mut placed = 0u32;
        for y in 0..side {
            for z in 0..side {
                for x in 0..side {
                    if placed >= n {
                        return;
                    }
                    let pos = Vec3::new(
                        (x as f32 - side as f32 * 0.5) * spacing,
                        1.0 + y as f32 * spacing,
                        (z as f32 - side as f32 * 0.5) * spacing,
                    );
                    self.add_sphere(pos, spacing * 0.45, 1.0);
                    placed += 1;
                }
            }
        }
    }

    pub fn rope_capsules(&mut self, segments: u32, start: Vec3, seg_len: f32) {
        let half = seg_len * 0.35;
        let radius = (seg_len * 0.12).min(0.05);
        let mut prev = None;
        for i in 0..segments {
            let pos = start + Vec3::new(0.0, -(i as f32) * seg_len, 0.0);
            let id = self.add_capsule(pos, radius, half, 0.2);
            if let Some(p) = prev {
                self.add_ball_joint(p, id, Vec3::new(0.0, -half, 0.0), Vec3::new(0.0, half, 0.0));
            }
            prev = Some(id);
        }
    }

    pub fn sleeping_field(&mut self, n: u32, origin: Vec3, spacing: f32) {
        for i in 0..n {
            let x = (i % 600) as f32 * spacing;
            let z = (i / 600) as f32 * spacing;
            let id = self.add_sphere(origin + Vec3::new(x, 0.5, z), spacing * 0.4, 1.0);
            self.world.force_sleep(id);
        }
    }
}
