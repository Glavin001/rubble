//! What a step reports back: the bodies that moved and the events it raised.

use crate::{BodyId, JointId};
use engine_schema::{MountBreak, StepCounters};
use engine_softbody::{BeamBreak, PieceDetached};

/// A body pose: position and quaternion (x, y, z, w).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

/// A body whose pose or sleep state changed in a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovedBody {
    pub body: BodyId,
    pub pose: Pose,
    /// The pose last reported for this body (`None` the first time: nothing to
    /// interpolate from). Hosts interpolate `previous → pose` between fixed steps.
    pub previous: Option<Pose>,
    pub user_data: u64,
}

/// Something that happened in a step.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A breakable joint exceeded its force or torque threshold. The joint stays
    /// allocated (broken) until `World::remove_joint`.
    JointBroke {
        joint: JointId,
        force: f32,
        torque: f32,
    },
    /// A vehicle part or wheel mount broke.
    MountBroke(MountBreak),
    /// A soft-body beam broke.
    BeamBroke(BeamBreak),
    /// A soft-body piece (a door, a bumper) came loose.
    PieceDetached(PieceDetached),
    /// The body fell asleep: its pose is final until it wakes.
    BodySlept { body: BodyId, user_data: u64 },
    /// The body woke.
    BodyWoke { body: BodyId, user_data: u64 },
}

/// The results of one step, delivered by `World::poll` in step order. Valid
/// until the host drops it; nothing in it refers back into the world.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// The step these results belong to (0-based, counting every `World::step`).
    pub step: u64,
    /// The GPU ran this step. A step is skipped when nothing is awake and
    /// nothing was written: nothing moved and no event was raised.
    pub ran: bool,
    /// Bodies whose pose or sleep state changed, sorted by body index.
    pub moved: Vec<MovedBody>,
    pub events: Vec<Event>,
    /// Events overwritten in the GPU ring before they were read (more than the
    /// ring holds between two frames). Never silently zero when events were lost.
    pub lost_events: u32,
    /// This frame's readback was dropped (`Backpressure::Drop`): `moved` is empty
    /// though bodies may have moved. Call `World::read_poses` to resynchronise.
    pub dropped: bool,
    /// The step's counters (bodies awake, contacts, colours, overflow flags).
    pub counters: Option<StepCounters>,
}
