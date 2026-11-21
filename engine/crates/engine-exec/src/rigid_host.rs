//! Host-side state of a rigid world that every host replaying [`crate::rigid_plan`]
//! keeps the same way (WgpuSim, CudaSim, MetalSim): what the scene's bodies and
//! colliders turn on, joint slots and their default torque arm, and the broadphase
//! grid the step uses. One definition, so a host gains a feature by calling these
//! rather than copying another host's code (docs/world-on-cuda.md, C3).

use crate::rigid_plan;
use engine_schema::convex::CollisionFilter;
use engine_schema::{BodyFlags, BodyState, Collider, Joint, ShapeType, BODY_CCD, BODY_CCD_FAST};

/// What uploaded bodies turn on: `(ccd, kinematic)` — a CCD body (either CCD flag), a
/// kinematic body (kinematic and not dynamic).
pub fn body_motion_flags(bodies: &[BodyState]) -> (bool, bool) {
    let ccd = bodies
        .iter()
        .any(|b| b.flags & (BODY_CCD | BODY_CCD_FAST) != 0);
    let kin = bodies.iter().any(|b| {
        b.flags & (BodyFlags::Kinematic as u32 | BodyFlags::Dynamic as u32)
            == BodyFlags::Kinematic as u32
    });
    (ccd, kin)
}

/// What colliders turn on: convex hulls, height fields, collision filters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColliderFlags {
    pub convex: bool,
    pub height_field: bool,
    pub filters: bool,
}

impl ColliderFlags {
    pub fn of(colliders: &[Collider]) -> Self {
        Self {
            convex: colliders
                .iter()
                .any(|c| c.shape == ShapeType::ConvexHull as u32),
            height_field: colliders
                .iter()
                .any(|c| c.shape == ShapeType::HeightField as u32),
            filters: colliders.iter().any(CollisionFilter::is_set),
        }
    }

    /// Flags of more colliders added to a scene: set ones stay set.
    pub fn add(&mut self, colliders: &[Collider]) {
        let more = Self::of(colliders);
        self.convex |= more.convex;
        self.height_field |= more.height_field;
        self.filters |= more.filters;
    }
}

// One definition for every host and the scene export (`HostWorld`): in engine-schema,
// which they all depend on.
pub use engine_schema::joint::{joint_from_desc, note_collider_reach};

/// A joint slot: a released one, else the next after `count` (which grows).
pub fn alloc_joint_slot(count: &mut u32, free: &mut Vec<u32>, max_joints: u32) -> u32 {
    match free.pop() {
        Some(id) => id,
        None => {
            let id = *count;
            assert!(id < max_joints, "joint capacity exceeded");
            *count += 1;
            id
        }
    }
}

/// Byte offset of joint `id` in binding 5.
pub fn joint_offset(id: u32) -> u64 {
    u64::from(id) * std::mem::size_of::<Joint>() as u64
}

/// The broadphase grid a scene steps with when nothing is cached: `(cell, levels, hier)`.
/// `hier` is the hierarchical grid (`Some` forced, `None` automatic: on for convex
/// scenes, and for a box scene the uniform grid's large list cannot hold).
pub fn choose_grid(
    colliders: &[Collider],
    bodies: &[BodyState],
    forced: Option<bool>,
    convex: bool,
    movable_radius: f32,
) -> (f32, u32, bool) {
    let mut hier = forced.unwrap_or(convex);
    let mut r = rigid_plan::grid_cell_with(colliders, bodies, hier, movable_radius);
    // Automatic grid: more dynamics wider than a uniform cell than its large list holds
    // take the hierarchical grid, which finds every pair at any size, instead of failing
    // the step.
    if !hier
        && forced.is_none()
        && rigid_plan::uniform_large_dynamics_overflow(colliders, bodies, r.0)
    {
        hier = true;
        r = rigid_plan::grid_cell_with(colliders, bodies, true, movable_radius);
    }
    (r.0, r.1, hier)
}
