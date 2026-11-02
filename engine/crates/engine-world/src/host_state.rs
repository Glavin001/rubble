//! Scene save and restore (a demo's reset key), and the small controls the web
//! pages need on top of Box3D's API: body type changes in place, fixed material
//! slots for colliders that name one directly, and a joint's starting stiffness.

use crate::{
    BodyId, BodySlot, BodyType, JointId, JointSlot, ShapeSlot, SurfaceMaterial, Transform, World,
};
use engine_schema::JointFrame;
use glam::Vec3;

/// A whole world at one instant ([`World::save_state`]): every GPU buffer and
/// the host's bookkeeping of bodies, shapes, joints and vehicles.
pub struct WorldSnapshot {
    sim: engine_webgpu::SimSnapshot,
    bodies: Vec<BodySlot>,
    free_bodies: Vec<u32>,
    shapes: Vec<ShapeSlot>,
    free_shapes: Vec<u32>,
    free_colliders: Vec<(u32, u64)>,
    collider_shape: Vec<u32>,
    joints: Vec<JointSlot>,
    materials: Vec<[u32; 3]>,
    vehicles: Vec<engine_vehicle::VehicleHandle>,
    soft_outputs: Vec<engine_softbody::SoftHandle>,
    skins: usize,
    terrain: Option<crate::terrain::TerrainSaved>,
    /// Bonds-model destructibles: their chunk bodies, bond joints and which bonds
    /// are intact (the joints' GPU records are in `sim`).
    destructibles: Vec<crate::destruction::DestructibleSlot>,
    bond_owner: std::collections::HashMap<u32, (u32, u32)>,
}

impl World {
    /// Copy the world (every GPU buffer, on the GPU, and the host's records)
    /// for [`World::restore_state`]: a scene reset. Bonds-model destructibles are
    /// covered (their joints and bookkeeping); not supported with threshold
    /// destructibles, assemblies, characters or projectiles, whose host state it
    /// does not cover. Terrain is saved with no area open (its settings,
    /// patches and watched shapes; the pool's buffers are not copied).
    pub fn save_state(&self) -> anyhow::Result<WorldSnapshot> {
        self.check_restorable()?;
        self.snapshot_with(self.sim.save_state())
    }

    /// [`World::save_state`] in a fraction of the GPU memory when the world's buffers
    /// are mostly unused (a scene before it runs): each buffer is copied up to its
    /// last non-zero word and restored with zeros past it, the same bytes. Waits for
    /// the GPU's measure (native hosts; the web uses [`World::measure_state`]).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_state_compact(&self) -> anyhow::Result<WorldSnapshot> {
        self.check_restorable()?;
        self.snapshot_with(self.sim.save_state_compact()?)
    }

    /// First half of a compact [`World::save_state`] without waiting: map
    /// [`StateMeasure::buffer`](engine_webgpu::StateMeasure::buffer), then
    /// [`World::save_state_measured`] with its words before stepping or changing the
    /// world (that refuses a measure the world has moved on from).
    pub fn measure_state(&self) -> anyhow::Result<engine_webgpu::StateMeasure> {
        self.check_restorable()?;
        Ok(self.sim.measure_state())
    }

    /// Second half of a compact snapshot ([`World::measure_state`]).
    pub fn save_state_measured(
        &self,
        measure: &engine_webgpu::StateMeasure,
        extents: &[u32],
    ) -> anyhow::Result<WorldSnapshot> {
        self.check_restorable()?;
        self.snapshot_with(self.sim.save_state_measured(measure, extents)?)
    }

    fn snapshot_with(&self, sim: engine_webgpu::SimSnapshot) -> anyhow::Result<WorldSnapshot> {
        let terrain = self.terrain_save()?;
        Ok(WorldSnapshot {
            sim,
            bodies: self.bodies.clone(),
            free_bodies: self.free_bodies.clone(),
            shapes: self.shapes.clone(),
            free_shapes: self.free_shapes.clone(),
            free_colliders: self.free_colliders.clone(),
            collider_shape: self.collider_shape.clone(),
            joints: self.joints.clone(),
            materials: self.materials.clone(),
            vehicles: self.vehicles.clone(),
            soft_outputs: self.soft_outputs.clone(),
            skins: self.skins.len(),
            terrain,
            destructibles: self.destructibles.clone(),
            bond_owner: self.bond_owner.clone(),
        })
    }

    /// Return the world to a [`World::save_state`] snapshot: bodies, shapes,
    /// joints, vehicles and soft bodies as they were, joints unbroken, parts
    /// back on. The step count keeps counting (frames stay ordered); the next
    /// frame reports every body whose pose differs from the last one delivered.
    /// Handles of objects created after the save are invalid afterwards. Open
    /// terrain areas are dropped, not baked, and the heightfields go back to
    /// their saved samples.
    pub fn restore_state(&mut self, s: &WorldSnapshot) -> anyhow::Result<()> {
        self.check_restorable()?;
        anyhow::ensure!(
            s.skins == self.skins.len(),
            "soft skins were bound after the save"
        );
        anyhow::ensure!(
            s.terrain.is_some() || self.terrain.is_none(),
            "terrain was enabled after the save"
        );
        let later_patches = self.terrain_patches_since(s.terrain.as_ref());
        self.sim.restore_state(&s.sim);
        // What was delivered stays what the next frame diffs against.
        let delivered: Vec<_> = self
            .bodies
            .iter()
            .map(|b| (b.last_transform, b.last_asleep, b.last_solver))
            .collect();
        self.bodies.clone_from(&s.bodies);
        for (b, d) in self.bodies.iter_mut().zip(delivered) {
            (b.last_transform, b.last_asleep, b.last_solver) = d;
        }
        self.free_bodies.clone_from(&s.free_bodies);
        self.shapes.clone_from(&s.shapes);
        self.free_shapes.clone_from(&s.free_shapes);
        self.free_colliders.clone_from(&s.free_colliders);
        self.collider_shape.clone_from(&s.collider_shape);
        self.joints.clone_from(&s.joints);
        self.materials.clone_from(&s.materials);
        self.vehicles.clone_from(&s.vehicles);
        self.soft_outputs.clone_from(&s.soft_outputs);
        self.destructibles.clone_from(&s.destructibles);
        self.bond_owner.clone_from(&s.bond_owner);
        self.terrain_restore(s.terrain.as_ref(), later_patches)?;
        self.resync = true;
        self.dirty = true;
        self.restored_at = self.step;
        Ok(())
    }

    fn check_restorable(&self) -> anyhow::Result<()> {
        // Bonds-model destructibles are bodies, shapes and joints plus host bookkeeping,
        // all saved; a threshold destructible's fracture state lives outside the sim.
        let bonds_only = self
            .destructibles
            .iter()
            .all(|d| !d.alive || d.model == crate::StrengthModel::Bonds);
        anyhow::ensure!(
            bonds_only && self.assemblies.is_empty() && self.motion.is_empty(),
            "save/restore covers bodies, shapes, joints, vehicles, soft bodies, terrain \
             and Bonds-model destructibles only (not threshold destructibles, assemblies, \
             characters or projectiles)"
        );
        Ok(())
    }

    /// The live body in slot `index` (frames, GPU buffers and guidance zones
    /// name bodies by slot).
    pub fn body_at(&self, index: u32) -> Option<BodyId> {
        let b = self.bodies.get(index as usize)?;
        b.alive.then(|| BodyId::new(index, b.generation))
    }

    /// Change a body's type and rewrite it whole at `transform` with these
    /// velocities (Box3D `b3Body_SetType` with the pose in one upload): a parked
    /// static body launched, or a spent one put back. Its shapes and mass data
    /// stay; it is awake. Commands queued for it this step are dropped.
    pub fn body_set_type(
        &mut self,
        id: BodyId,
        body_type: BodyType,
        transform: Transform,
        linear_velocity: [f32; 3],
        angular_velocity: [f32; 3],
    ) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        anyhow::ensure!(
            self.bodies[i].owner == crate::Owner::User,
            "body {} belongs to a vehicle, soft body or destructible",
            id.index()
        );
        self.dirty = true;
        self.sim.cancel_body_commands(id.index());
        let step = self.step;
        let slot = &mut self.bodies[i];
        slot.body_type = body_type;
        slot.origin = transform;
        slot.velocity = (linear_velocity, angular_velocity);
        slot.asleep = false;
        slot.written_step = step;
        let state = crate::body_state(slot);
        self.sim.write_bodies(id.index(), &[state]);
        Ok(())
    }

    /// Pin GPU material slot `slot` to `material`, for colliders that name a
    /// slot directly rather than a [`SurfaceMaterial`]: vehicle part colliders
    /// (`ColliderDesc::material`), wheel rims (`rim_material`) and soft-body
    /// nodes. Shapes with this material then share the slot. Refused when a
    /// shape already uses the slot with another material.
    pub fn set_material_slot(
        &mut self,
        slot: u32,
        material: SurfaceMaterial,
    ) -> anyhow::Result<()> {
        material.validate()?;
        anyhow::ensure!(slot < 64, "material slot {slot} out of range (64)");
        let key = material.key();
        let s = slot as usize;
        if let Some(&old) = self.materials.get(s) {
            let used = self
                .shapes
                .iter()
                .any(|sh| sh.alive && sh.def.material.key() == old);
            anyhow::ensure!(
                old == key || !used,
                "material slot {slot} is in use by a shape"
            );
        }
        let default = SurfaceMaterial::default();
        while self.materials.len() <= s {
            let i = self.materials.len() as u32;
            self.materials.push(default.key());
            self.sim.set_material(i, default.gpu());
        }
        self.materials[s] = key;
        self.sim.set_material(slot, material.gpu());
        Ok(())
    }

    /// Size the broadphase grid as if a collider of bounding radius `radius`
    /// (m) moved, before any does: bodies parked static and launched later keep
    /// the grid they will need (the same cell as when they move).
    pub fn reserve_moving_radius(&mut self, radius: f32) {
        self.sim.reserve_movable_radius(radius);
    }

    /// Start a joint's linear rows at stiffness `k` (N/m) instead of the
    /// solver's `k_start` (AVBD ramps the penalty up from there): a stiff bolt
    /// that should carry its load from the first step. Only before the joint
    /// has been stepped.
    pub fn joint_set_linear_penalty(&mut self, id: JointId, k: f32) -> anyhow::Result<()> {
        let i = self.live_joint(id)?;
        anyhow::ensure!(k > 0.0 && k.is_finite(), "penalty must be positive");
        self.dirty = true;
        self.sim.set_joint_linear_penalty(i as u32, k);
        Ok(())
    }

    /// Register joints a soft body wrote itself (`before..` the sim's count),
    /// so their breaks report with handles. They belong to the soft body: no
    /// body lists them and their frames are never re-expressed.
    pub(crate) fn adopt_joints(&mut self, before: u32) {
        for index in before..self.sim.joint_count() {
            let i = index as usize;
            let empty = JointSlot {
                generation: 1,
                alive: false,
                bodies: [0, 0],
                user_data: 0,
                origin_frames: [JointFrame::at(Vec3::ZERO); 2],
            };
            if self.joints.len() <= i {
                self.joints.resize(i + 1, empty);
            }
            self.joints[i].alive = true;
        }
    }
}
