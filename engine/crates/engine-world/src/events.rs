//! What a step reports back, grouped as Box3D groups its events (body moves,
//! joint events) plus Rubble's destruction stream and per-frame outputs for
//! vehicles, soft bodies and skinned meshes.

use crate::{ActorInfo, AssemblyId, BodyId, DestructibleId, JointId, Transform};
use engine_schema::{MountBreak, StepCounters};
use engine_softbody::{BeamBreak, PieceDetached, SoftHandle};
use engine_vehicle::{VehicleHandle, VehicleState};
use std::collections::BTreeMap;

/// A body whose pose or sleep state changed in a step (Box3D
/// `b3BodyMoveEvent`, plus the previous pose for interpolation; moves the host
/// made, such as teleports, are reported too). Soft-body particles are not
/// reported here: see [`Frame::soft`] and [`Frame::skins`].
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
    /// A connection of an assembly (or a composed vehicle) broke: a mount
    /// sheared off, a joint past its break limit. The joint stays allocated
    /// (broken) until `World::destroy_joint`.
    ConnectionBroke {
        assembly: AssemblyId,
        connection: String,
        joint: JointId,
        force: f32,
        torque: f32,
    },
    /// Bonds of a destructible broke this step (bond indices into its asset).
    BondsBroken {
        destructible: DestructibleId,
        bonds: Vec<u32>,
    },
    /// An actor (connected chunks) of a destructible came apart. The parent
    /// is described by value: its chunks have already moved to the children.
    Split {
        destructible: DestructibleId,
        parent: ActorInfo,
        children: Vec<ActorInfo>,
    },
}

/// A vehicle's state after the step: speeds, engine, gear, and per wheel the
/// pose (spin and steer included), suspension length, spin rate and contact,
/// which is what drawing raycast wheels needs.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleOutput {
    pub vehicle: VehicleHandle,
    pub state: VehicleState,
}

/// A soft body's node positions after the step (`World::soft_enable_output`).
#[derive(Clone, Debug, PartialEq)]
pub struct SoftOutput {
    pub soft: SoftHandle,
    pub nodes: Vec<[f32; 3]>,
}

/// Two shapes started touching (either has `enable_contact_events`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactBeginEvent {
    pub shape_a: crate::ShapeId,
    pub shape_b: crate::ShapeId,
    /// The deepest contact point and the normal from A to B (world).
    pub point: [f32; 3],
    pub normal: [f32; 3],
    /// Closing speed along the normal at that point, before the solve (m/s).
    pub approach_speed: f32,
}

/// Two shapes stopped touching (or one was destroyed while touching).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContactEndEvent {
    pub shape_a: crate::ShapeId,
    pub shape_b: crate::ShapeId,
}

/// Contact begin and end events, and hits: begins at or above
/// `WorldDef::hit_event_threshold` on shapes with `enable_hit_events`
/// (impact sounds, effects, damage).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContactEvents {
    pub begin: Vec<ContactBeginEvent>,
    pub end: Vec<ContactEndEvent>,
    pub hits: Vec<ContactBeginEvent>,
}

/// A shape entered or left a sensor (`ShapeDef::is_sensor` with
/// `enable_sensor_events`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SensorEvent {
    pub sensor_shape: crate::ShapeId,
    pub visitor_shape: crate::ShapeId,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SensorEvents {
    pub begin: Vec<SensorEvent>,
    pub end: Vec<SensorEvent>,
}

/// A batch of queued ray casts ([`crate::World::cast_rays`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RayBatchId(pub u64);

/// A ray batch's results, traced against the step's final poses.
#[derive(Clone, Debug, PartialEq)]
pub struct RayBatchResult {
    pub batch: RayBatchId,
    /// One per ray, in the order queued.
    pub hits: Vec<Option<crate::RayResult>>,
}

/// A skinned soft-body mesh ([`crate::World::soft_bind_skin`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkinId(pub u32);

/// A skinned mesh's deformed vertices and normals after the step.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinOutput {
    pub skin: SkinId,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
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
    /// Every vehicle (rigid, or a soft vehicle's core).
    pub vehicles: Vec<VehicleOutput>,
    /// Soft bodies with output enabled.
    pub soft: Vec<SoftOutput>,
    /// Bound skins.
    pub skins: Vec<SkinOutput>,
    /// Ray batches queued before this step.
    pub ray_results: Vec<RayBatchResult>,
    /// Shapes that started or stopped touching this step, and hits.
    pub contact_events: ContactEvents,
    /// Shapes that entered or left sensors this step.
    pub sensor_events: SensorEvents,
    /// Overlap queries queued before this step.
    pub overlap_results: Vec<crate::OverlapResult>,
    /// Records overwritten in the GPU event ring before they were read (more
    /// than it holds between two frames). Never silently zero.
    pub lost_events: u32,
    /// This frame's readback was dropped (`Backpressure::Drop`): it carries
    /// nothing. The next delivered frame has `resynced` set and reports every
    /// body that changed since the last delivered frame.
    pub dropped: bool,
    /// This frame reports every change since the last delivered frame (it
    /// follows dropped ones).
    pub resynced: bool,
    /// The step's counters (awake bodies, contacts, colours, overflow flags).
    pub counters: Option<StepCounters>,
}

/// Several steps' frames folded into one, for hosts that step more often than
/// they draw (`World::poll_coalesced`).
#[derive(Clone, Debug, Default)]
pub struct CoalescedFrame {
    /// The first and last step folded in (`None` when there were none).
    pub steps: Option<(u64, u64)>,
    /// Per body: the latest transform, the earliest previous transform, and
    /// the net sleep change across the steps.
    pub moves: Vec<BodyMoveEvent>,
    /// Every destruction event, in step order, with its step.
    pub destruction_events: Vec<(u64, DestructionEvent)>,
    /// The latest output per vehicle, soft body and skin.
    pub vehicles: Vec<VehicleOutput>,
    pub soft: Vec<SoftOutput>,
    pub skins: Vec<SkinOutput>,
    /// Every ray batch's results.
    pub ray_results: Vec<RayBatchResult>,
    /// Every step's contact events, in step order.
    pub contact_events: ContactEvents,
    pub sensor_events: SensorEvents,
    /// Every overlap query's results.
    pub overlap_results: Vec<crate::OverlapResult>,
    pub lost_events: u32,
    /// Any folded step's readback was dropped (a later one resynced it).
    pub dropped: bool,
    /// The last step that ran.
    pub counters: Option<StepCounters>,
}

/// Fold frames (in step order) into one.
pub fn coalesce(frames: Vec<Frame>) -> CoalescedFrame {
    let mut out = CoalescedFrame::default();
    let (Some(first), Some(last)) = (frames.first(), frames.last()) else {
        return out;
    };
    out.steps = Some((first.step, last.step));
    let mut moves: BTreeMap<BodyId, BodyMoveEvent> = BTreeMap::new();
    let mut vehicles = BTreeMap::new();
    let mut soft = BTreeMap::new();
    let mut skins = BTreeMap::new();
    for f in frames {
        for m in f.body_events.moves {
            moves
                .entry(m.body)
                .and_modify(|e| {
                    e.transform = m.transform;
                    if m.previous.is_none() {
                        e.previous = None;
                    }
                    if m.fell_asleep {
                        e.fell_asleep = !e.woke;
                        e.woke = false;
                    } else if m.woke {
                        e.woke = !e.fell_asleep;
                        e.fell_asleep = false;
                    }
                })
                .or_insert(m);
        }
        out.destruction_events
            .extend(f.destruction_events.into_iter().map(|e| (f.step, e)));
        for v in f.vehicles {
            vehicles.insert(v.vehicle.0, v);
        }
        for s in f.soft {
            soft.insert(s.soft.0, s);
        }
        for s in f.skins {
            skins.insert(s.skin, s);
        }
        out.ray_results.extend(f.ray_results);
        out.contact_events.begin.extend(f.contact_events.begin);
        out.contact_events.end.extend(f.contact_events.end);
        out.contact_events.hits.extend(f.contact_events.hits);
        out.sensor_events.begin.extend(f.sensor_events.begin);
        out.sensor_events.end.extend(f.sensor_events.end);
        out.overlap_results.extend(f.overlap_results);
        out.lost_events += f.lost_events;
        out.dropped |= f.dropped;
        if f.counters.is_some() {
            out.counters = f.counters;
        }
    }
    out.moves = moves.into_values().collect();
    out.vehicles = vehicles.into_values().collect();
    out.soft = soft.into_values().collect();
    out.skins = skins.into_values().collect();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(body: u64, x: f32, prev: Option<f32>, slept: bool, woke: bool) -> BodyMoveEvent {
        let t = |x: f32| Transform::new([x, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        BodyMoveEvent {
            body: BodyId::from_bits(body),
            transform: t(x),
            previous: prev.map(t),
            fell_asleep: slept,
            woke,
            user_data: 0,
        }
    }

    fn frame(step: u64, moves: Vec<BodyMoveEvent>) -> Frame {
        Frame {
            step,
            ran: true,
            body_events: BodyEvents { moves },
            ..Frame::default()
        }
    }

    #[test]
    fn coalescing_keeps_latest_pose_earliest_previous_and_net_sleep() {
        let a = 1 << 32;
        let frames = vec![
            frame(4, vec![mv(a, 1.0, Some(0.0), false, false)]),
            frame(5, vec![mv(a, 2.0, Some(1.0), true, false)]),
            frame(6, vec![]),
            frame(7, vec![mv(a, 3.0, Some(2.0), false, true)]),
        ];
        let c = coalesce(frames);
        assert_eq!(c.steps, Some((4, 7)));
        assert_eq!(c.moves.len(), 1);
        let m = &c.moves[0];
        assert_eq!(m.transform.position[0], 3.0);
        assert_eq!(m.previous.unwrap().position[0], 0.0);
        assert!(
            !m.fell_asleep && !m.woke,
            "slept then woke nets to no change"
        );
    }
}
