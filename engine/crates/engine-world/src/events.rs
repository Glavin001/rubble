//! What a step reports back, grouped as Box3D groups its events (body moves,
//! joint events) plus Rubble's destruction stream.

use crate::{BodyId, JointId, Transform};
use engine_schema::{MountBreak, StepCounters};
use engine_softbody::{BeamBreak, PieceDetached};

/// A body whose pose or sleep state changed in a step (Box3D
/// `b3BodyMoveEvent`, plus the previous pose for interpolation; moves the host
/// made, such as teleports, are reported too).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyMoveEvent {
    pub body: BodyId,
    /// The body origin's pose.
    pub transform: Transform,
    /// The pose last reported (`None` the first time: nothing to interpolate
    /// from). Hosts interpolate `previous → transform` between fixed steps.
    pub previous: Option<Transform>,
    pub fell_asleep: bool,
    pub woke: bool,
    pub user_data: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyEvents {
    pub moves: Vec<BodyMoveEvent>,
}

/// Something broke.
#[derive(Clone, Debug, PartialEq)]
pub enum DestructionEvent {
    /// A breakable joint or bond exceeded its limit. It stays allocated
    /// (broken) until `World::destroy_joint`.
    JointBroke {
        joint: JointId,
        force: f32,
        torque: f32,
        user_data: u64,
    },
    /// A vehicle part or wheel mount broke.
    MountBroke(MountBreak),
    /// A soft-body beam broke.
    BeamBroke(BeamBreak),
    /// A soft-body piece (a door, a bumper) came loose.
    PieceDetached(PieceDetached),
}

/// The results of one step, delivered by `World::poll` in step order. Valid
/// until dropped; nothing in it refers back into the world.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// The step these results belong to (0-based, counting every `step`).
    pub step: u64,
    /// The GPU ran this step. A step is skipped when nothing is awake and
    /// nothing was written: nothing moved and nothing happened.
    pub ran: bool,
    pub body_events: BodyEvents,
    pub destruction_events: Vec<DestructionEvent>,
    /// Records overwritten in the GPU event ring before they were read (more
    /// than it holds between two frames). Never silently zero.
    pub lost_events: u32,
    /// This frame's readback was dropped (`Backpressure::Drop`): `body_events`
    /// is empty though bodies may have moved; `World::read_transforms` resyncs.
    pub dropped: bool,
    /// The step's counters (awake bodies, contacts, colours, overflow flags).
    pub counters: Option<StepCounters>,
}
