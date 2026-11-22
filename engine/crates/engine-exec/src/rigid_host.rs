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
    choose_grid_with(colliders, bodies, forced, convex, movable_radius, &mut || {
        rigid_plan::grid_cell_with(colliders, bodies, true, movable_radius)
    })
}

/// [`choose_grid`] with the hierarchical grid's `(cell, levels)` from `hier_grid` when it is
/// the one chosen: a host that keeps the grid's geometry between steps
/// (`engine_schema::convex::HierGridGeometry`, the same values) passes it here instead of
/// sizing every collider again.
pub fn choose_grid_with(
    colliders: &[Collider],
    bodies: &[BodyState],
    forced: Option<bool>,
    convex: bool,
    movable_radius: f32,
    hier_grid: &mut dyn FnMut() -> (f32, u32),
) -> (f32, u32, bool) {
    let mut hier = forced.unwrap_or(convex);
    let mut r = if hier {
        hier_grid()
    } else {
        rigid_plan::grid_cell_with(colliders, bodies, false, movable_radius)
    };
    // Automatic grid: more dynamics wider than a uniform cell than its large list holds
    // take the hierarchical grid, which finds every pair at any size, instead of failing
    // the step.
    if !hier
        && forced.is_none()
        && rigid_plan::uniform_large_dynamics_overflow(colliders, bodies, r.0)
    {
        hier = true;
        r = hier_grid();
    }
    (r.0, r.1, hier)
}

/// Hull-arena records that hold a height field with a soil (its header: magic, a side
/// count in range on both axes, the soil bit): the soil passes are then planned.
pub fn records_have_soil(records: &[[u32; 4]]) -> bool {
    use engine_schema::heightfield::*;
    let sides = HEIGHT_FIELD_MIN_SAMPLES..=HEIGHT_FIELD_MAX_SAMPLES;
    records.iter().any(|h| {
        h[0] == HEIGHT_FIELD_MAGIC
            && h[3] & HEIGHT_FIELD_SOIL != 0
            && sides.contains(&h[1])
            && sides.contains(&h[2])
    })
}

/// A height field's static support body at `pos`, `rot`.
pub fn height_field_support(pos: glam::Vec3, rot: glam::Quat) -> BodyState {
    let mut support = BodyState::static_plane_support();
    support.position = pos.to_array();
    support.orientation = rot.to_array();
    support.initial_pos = support.position;
    support.initial_rot = support.orientation;
    support
}

/// The collider of a height field inserted at arena record `base`, on `body`.
pub fn height_field_collider(
    body: u32,
    base: u32,
    field: &engine_schema::heightfield::HeightField,
    material: u32,
) -> Collider {
    Collider {
        body,
        shape: ShapeType::HeightField as u32,
        material,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: engine_schema::heightfield::height_field_collider_params(base, field),
    }
}

/// A dynamic body made of convex hulls (cooked: mass, inertia, centre) at `pos`, `rot`:
/// its state and its colliders on body `body`, each hull inserted into `arena`.
pub fn convex_body(
    arena: &mut engine_schema::convex::HullArena,
    hulls: &[engine_schema::convex::ConvexHull],
    pos: glam::Vec3,
    rot: glam::Quat,
    density: f32,
    body: u32,
) -> (BodyState, Vec<Collider>) {
    let cooked = engine_schema::convex::cook_body(hulls, density);
    let (p, q) = cooked.body_pose(pos, rot);
    let state = BodyState::dynamic(p, q, cooked.mass, cooked.inertia);
    let colliders = cooked
        .children
        .iter()
        .map(|child| Collider {
            body,
            shape: ShapeType::ConvexHull as u32,
            material: 0,
            generation: 0,
            local_pos: child.local_pos.to_array(),
            density,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: engine_schema::convex::convex_collider_params(
                arena.insert(&child.hull).expect("hull arena capacity"),
                1.0,
                child.hull.bounding_radius(),
            ),
        })
        .collect();
    (state, colliders)
}

/// Impulse records a host's GPU pass appended for the next rigid step, after the host's
/// own (the terrain coupling's reactions, the explicit vehicles' window impulses:
/// `IMPULSE_COUNT` atomics, then `IMPULSE_FIRST` set to the count). The step that runs
/// next schedules `apply_impulses` for them ([`rigid_plan::MotionInputs::impulses`]);
/// whether that step must run although nothing is awake is the host's wake, set beside
/// [`Self::mark`] by a pass whose records must wake the bodies.
///
/// [`rigid_plan::MotionInputs::impulses`]: crate::rigid_plan::MotionInputs::impulses
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuImpulses {
    pending: bool,
}

impl GpuImpulses {
    /// A GPU pass appended records for the next step.
    pub fn mark(&mut self) {
        self.pending = true;
    }

    /// Records wait for the next step that runs.
    pub fn pending(&self) -> bool {
        self.pending
    }

    /// `MotionInputs::impulses` of the step being encoded: the host queued `queued`
    /// records of its own, or a GPU pass appended some.
    pub fn this_step(&self, queued: usize) -> bool {
        queued > 0 || self.pending
    }

    /// The step that applies them was encoded.
    pub fn applied(&mut self) {
        self.pending = false;
    }
}
