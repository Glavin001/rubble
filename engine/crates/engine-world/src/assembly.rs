//! Assemblies: parts (rigid bodies, destructibles) and the connections
//! between them (joints, breakable mounts), placed as one. A vehicle is an
//! assembly with a controller on top: [`VehicleDef`] builds a rigid or soft
//! vehicle core and mounts extra parts onto its chassis, its parts or its soft
//! nodes, so one call composes rigid, soft, bending and fracturing pieces.
//!
//! Connections are given once, as a frame in the assembly's (or vehicle's
//! design) space; each body's joint frame is derived from where that body
//! starts, so authoring never deals with per-body frames.

use crate::defs::{BodyDef, Geometry, ShapeDef};
use crate::destruction::{DestructibleDef, DestructibleId};
use crate::events::DestructionEvent;
use crate::joints::{JointBase, JointDef, WeldJointDef};
use crate::{BodyId, JointId, Transform, World};
use engine_softbody::{SoftBackend, SoftHandle, SoftVehicleDesc, SpawnOptions};
use engine_vehicle::{VehicleDesc, VehicleHandle};
use glam::Vec3;

/// What a part is made of.
#[derive(Clone, Debug)]
pub enum PartModel {
    /// A rigid body: its pose (in the assembly frame) and its shapes.
    Rigid {
        body: BodyDef,
        shapes: Vec<(ShapeDef, Geometry)>,
    },
    /// A destructible; its `transform` is in the assembly frame.
    Destructible(DestructibleDef),
}

#[derive(Clone, Debug)]
pub struct PartDef {
    pub name: String,
    pub model: PartModel,
}

/// One end of a connection.
#[derive(Clone, Debug, PartialEq)]
pub enum PartRef {
    /// A rigid part by name (for a destructible part: its chunk 0).
    Part(String),
    /// A chunk of a destructible part.
    Chunk(String, u32),
    /// The vehicle's chassis (the core of a soft vehicle).
    Chassis,
    /// One of the vehicle's own parts (`VehicleDesc::parts` index).
    VehiclePart(u32),
    /// A node of the soft vehicle's lattice.
    SoftNode(u32),
    /// Any existing body.
    Body(BodyId),
}

/// How two parts are held together.
#[derive(Clone, Debug)]
pub enum ConnectionKind {
    /// A rigid weld that breaks past `break_force` (N) or `break_torque`
    /// (N m), like a bolted mount.
    Mount { break_force: f32, break_torque: f32 },
    /// Any typed joint; its bodies and frames are filled in from the
    /// connection (its base's other fields, such as break limits, are kept).
    Joint(JointDef),
}

#[derive(Clone, Debug)]
pub struct ConnectionDef {
    pub name: String,
    pub a: PartRef,
    pub b: PartRef,
    /// Where the connection is, in the assembly frame (joint z: hinge axis).
    pub frame: Transform,
    pub kind: ConnectionKind,
}

/// Parts and connections placed together (`World::create_assembly`).
#[derive(Clone, Debug, Default)]
pub struct AssemblyDef {
    pub transform: Transform,
    pub parts: Vec<PartDef>,
    pub connections: Vec<ConnectionDef>,
    pub user_data: u64,
}

/// A vehicle's core: the controller's rigid chassis with wheels and
/// drivetrain, or a soft node-beam vehicle with its rigid core.
#[derive(Clone, Debug)]
pub enum VehicleModel {
    Rigid(VehicleDesc),
    Soft(SoftVehicleDesc, SpawnOptions),
}

/// A vehicle composed of a core and extra parts (`World::create_vehicle`).
/// Parts and connection frames are in the vehicle's design space (the frame
/// its descriptor is authored in); `transform` places the vehicle.
#[derive(Clone, Debug)]
pub struct VehicleDef {
    pub model: VehicleModel,
    pub transform: Transform,
    pub parts: Vec<PartDef>,
    pub connections: Vec<ConnectionDef>,
    pub user_data: u64,
}

crate::handles::handle!(
    /// A placed assembly (or a composed vehicle's extra parts).
    AssemblyId
);

/// A built part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartHandle {
    Body(BodyId),
    Destructible(DestructibleId),
}

/// What `World::create_vehicle` built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VehicleBuild {
    /// The controller: drive it with `set_vehicle_input` (soft vehicles:
    /// `set_soft_vehicle_input`). `None` for a soft body without wheels.
    pub vehicle: Option<VehicleHandle>,
    pub soft: Option<SoftHandle>,
    pub chassis: BodyId,
    pub assembly: AssemblyId,
}

pub(crate) struct AssemblySlot {
    pub generation: u32,
    pub alive: bool,
    pub parts: Vec<(String, PartHandle)>,
    pub connections: Vec<(String, Option<JointId>)>,
    pub user_data: u64,
}

/// Where assembly-relative things resolve against a vehicle.
struct VehicleContext {
    vehicle: Option<VehicleHandle>,
    soft: Option<SoftHandle>,
    chassis: BodyId,
    /// Vehicle design space -> world.
    frame: Transform,
}

impl World {
    /// Place parts and connect them.
    pub fn create_assembly(&mut self, def: &AssemblyDef) -> anyhow::Result<AssemblyId> {
        self.build_assembly(
            def.transform,
            &def.parts,
            &def.connections,
            def.user_data,
            None,
        )
    }

    /// Compose a vehicle: its core (rigid controller or soft lattice with a
    /// rigid core), then extra parts mounted onto its chassis, its own parts
    /// or its soft nodes.
    pub fn create_vehicle(&mut self, def: &VehicleDef) -> anyhow::Result<VehicleBuild> {
        let t = def.transform;
        let (vehicle, soft, chassis, design) = match &def.model {
            VehicleModel::Rigid(desc) => {
                let h = self.add_vehicle(desc, t.position, t.rotation)?;
                let body = self.sim.vehicle_host().chassis_body(h);
                // The chassis body sits at the design-space centre of mass.
                let origin = Vec3::from(self.sim.vehicle_host().origin(h));
                let chassis = self
                    .owned_body(body)
                    .ok_or_else(|| anyhow::anyhow!("chassis body"))?;
                let design = design_frame(self.sim.host_body(body), origin);
                (Some(h), None, chassis, design)
            }
            VehicleModel::Soft(desc, options) => {
                let s = self.add_soft_vehicle(desc, t.position, t.rotation, options)?;
                let entry = &self.sim.soft_host().entries[s.0 as usize];
                let (vehicle, core) = match entry.core {
                    Some((v, body)) => (Some(v), body),
                    None => (None, entry.first_body),
                };
                let chassis = self
                    .owned_body(core)
                    .or_else(|| self.soft_node_body(s, 0))
                    .ok_or_else(|| anyhow::anyhow!("soft vehicle core"))?;
                // Soft nodes are spawned from design space by `transform`.
                (vehicle, Some(s), chassis, t)
            }
        };
        let ctx = VehicleContext {
            vehicle,
            soft,
            chassis,
            frame: design,
        };
        let assembly = self.build_assembly(
            design,
            &def.parts,
            &def.connections,
            def.user_data,
            Some(&ctx),
        )?;
        Ok(VehicleBuild {
            vehicle,
            soft,
            chassis,
            assembly,
        })
    }

    /// A built part by name.
    pub fn assembly_part(&self, id: AssemblyId, name: &str) -> anyhow::Result<PartHandle> {
        let a = &self.assemblies[self.live_assembly(id)?];
        a.parts
            .iter()
            .find(|(n, _)| n == name)
            .map(|p| p.1)
            .ok_or_else(|| anyhow::anyhow!("no part named {name:?}"))
    }

    /// A connection's joint, if it has not broken and been removed.
    pub fn assembly_connection(
        &self,
        id: AssemblyId,
        name: &str,
    ) -> anyhow::Result<Option<JointId>> {
        let a = &self.assemblies[self.live_assembly(id)?];
        a.connections
            .iter()
            .find(|(n, _)| n == name)
            .map(|c| c.1)
            .ok_or_else(|| anyhow::anyhow!("no connection named {name:?}"))
    }

    fn live_assembly(&self, id: AssemblyId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.assemblies.get(i) {
            Some(a) if a.alive && a.generation == id.generation() => Ok(i),
            _ => Err(crate::StaleHandle("assembly", id.to_bits()).into()),
        }
    }

    fn soft_node_body(&self, soft: SoftHandle, node: u32) -> Option<BodyId> {
        let e = self.sim.soft_host().entries.get(soft.0 as usize)?;
        if node >= e.particles {
            return None;
        }
        let i = e.first_body + node;
        let b = self.bodies.get(i as usize)?;
        b.alive.then(|| BodyId::new(i, b.generation))
    }

    fn build_assembly(
        &mut self,
        frame: Transform,
        parts: &[PartDef],
        connections: &[ConnectionDef],
        user_data: u64,
        vehicle: Option<&VehicleContext>,
    ) -> anyhow::Result<AssemblyId> {
        let mut built: Vec<(String, PartHandle)> = Vec::with_capacity(parts.len());
        for p in parts {
            anyhow::ensure!(
                built.iter().all(|(n, _)| *n != p.name),
                "duplicate part name {:?}",
                p.name
            );
            let handle = match &p.model {
                PartModel::Rigid { body, shapes } => {
                    let pose = compose(frame, Transform::new(body.position, body.rotation));
                    let def = BodyDef {
                        position: pose.position,
                        rotation: pose.rotation,
                        user_data: if body.user_data == 0 {
                            user_data
                        } else {
                            body.user_data
                        },
                        ..body.clone()
                    };
                    let b = self.create_body(&def)?;
                    for (s, g) in shapes {
                        self.create_shape(b, s, g.clone())?;
                    }
                    PartHandle::Body(b)
                }
                PartModel::Destructible(d) => {
                    let placed = DestructibleDef {
                        transform: compose(frame, d.transform),
                        ..d.clone()
                    };
                    PartHandle::Destructible(self.create_destructible(&placed)?)
                }
            };
            built.push((p.name.clone(), handle));
        }
        let index = match self.assemblies.iter().position(|a| !a.alive) {
            Some(i) => i,
            None => {
                self.assemblies.push(AssemblySlot {
                    generation: 0,
                    alive: false,
                    parts: Vec::new(),
                    connections: Vec::new(),
                    user_data: 0,
                });
                self.assemblies.len() - 1
            }
        };
        let mut joints = Vec::with_capacity(connections.len());
        for (k, c) in connections.iter().enumerate() {
            let a = self.resolve(&c.a, &built, vehicle)?;
            let b = self.resolve(&c.b, &built, vehicle)?;
            let world = compose(frame, c.frame);
            let local_a = relative(self.origin_pose(a)?, world);
            let local_b = relative(self.origin_pose(b)?, world);
            let place = |base: &JointBase| JointBase {
                body_a: a,
                body_b: b,
                local_frame_a: local_a,
                local_frame_b: local_b,
                user_data: if base.user_data == 0 {
                    user_data
                } else {
                    base.user_data
                },
                ..base.clone()
            };
            let def = match &c.kind {
                ConnectionKind::Mount {
                    break_force,
                    break_torque,
                } => {
                    let mut base = place(&JointBase::new(a, b));
                    base.break_force = *break_force;
                    base.break_torque = *break_torque;
                    JointDef::Weld(WeldJointDef {
                        base,
                        linear_stiffness: None,
                        angular_stiffness: None,
                    })
                }
                ConnectionKind::Joint(j) => {
                    let mut j = j.clone();
                    match &mut j {
                        JointDef::Revolute(d) => d.base = place(&d.base),
                        JointDef::Prismatic(d) => d.base = place(&d.base),
                        JointDef::Spherical(d) => d.base = place(&d.base),
                        JointDef::Weld(d) => d.base = place(&d.base),
                        JointDef::Distance(d) => d.base = place(&d.base),
                        JointDef::Filter(d) => d.base = place(&d.base),
                        JointDef::Custom { .. } => {
                            return crate::error::unsupported(
                                "custom joints as assembly connections",
                            );
                        }
                    }
                    j
                }
            };
            let joint = self.create_joint(&def)?;
            self.connection_owner
                .insert(joint.index(), (index as u32, k as u32));
            joints.push((c.name.clone(), Some(joint)));
        }
        let slot = &mut self.assemblies[index];
        slot.generation = slot.generation.wrapping_add(1).max(1);
        slot.alive = true;
        slot.parts = built;
        slot.connections = joints;
        slot.user_data = user_data;
        Ok(AssemblyId::new(index as u32, slot.generation))
    }

    fn resolve(
        &self,
        r: &PartRef,
        built: &[(String, PartHandle)],
        vehicle: Option<&VehicleContext>,
    ) -> anyhow::Result<BodyId> {
        let part = |name: &str| {
            built
                .iter()
                .find(|(n, _)| n == name)
                .map(|p| p.1)
                .ok_or_else(|| anyhow::anyhow!("no part named {name:?}"))
        };
        let need =
            || vehicle.ok_or_else(|| anyhow::anyhow!("{r:?} needs a vehicle (create_vehicle)"));
        Ok(match r {
            PartRef::Part(name) => match part(name)? {
                PartHandle::Body(b) => b,
                PartHandle::Destructible(d) => self.destructible_chunk_body(d, 0)?,
            },
            PartRef::Chunk(name, chunk) => match part(name)? {
                PartHandle::Destructible(d) => self.destructible_chunk_body(d, *chunk)?,
                PartHandle::Body(_) => anyhow::bail!("{name:?} is not a destructible part"),
            },
            PartRef::Chassis => need()?.chassis,
            PartRef::VehiclePart(p) => {
                let v = need()?
                    .vehicle
                    .ok_or_else(|| anyhow::anyhow!("a soft vehicle has no rigid parts"))?;
                let body = self.sim.vehicle_host().part_body(v, *p);
                self.owned_body(body)
                    .ok_or_else(|| anyhow::anyhow!("vehicle part {p} has no body"))?
            }
            PartRef::SoftNode(n) => {
                let s = need()?
                    .soft
                    .ok_or_else(|| anyhow::anyhow!("a rigid vehicle has no soft nodes"))?;
                self.soft_node_body(s, *n)
                    .ok_or_else(|| anyhow::anyhow!("soft node {n} out of range"))?
            }
            PartRef::Body(b) => {
                self.live_body(*b)?;
                *b
            }
        })
    }

    /// A body's origin pose as uploaded (before it has simulated): the host
    /// mirror's solver pose, through the body's solver frame.
    fn origin_pose(&self, b: BodyId) -> anyhow::Result<Transform> {
        let i = self.live_body(b)?;
        let s = self.sim.host_body(b.index());
        let solver = Transform::new(s.position, s.orientation);
        Ok(self.bodies[i].frame().to_origin(solver))
    }

    /// Joint breaks that belong to assembly connections, as
    /// `ConnectionBroke`; everything else passes through.
    pub(crate) fn connection_events(
        &mut self,
        events: Vec<DestructionEvent>,
    ) -> Vec<DestructionEvent> {
        events
            .into_iter()
            .map(|e| {
                if let DestructionEvent::JointBroke {
                    joint,
                    force,
                    torque,
                    ..
                } = &e
                {
                    if let Some(&(a, k)) = self.connection_owner.get(&joint.index()) {
                        let slot = &self.assemblies[a as usize];
                        if slot.alive {
                            return DestructionEvent::ConnectionBroke {
                                assembly: AssemblyId::new(a, slot.generation),
                                connection: slot.connections[k as usize].0.clone(),
                                joint: *joint,
                                force: *force,
                                torque: *torque,
                            };
                        }
                    }
                }
                e
            })
            .collect()
    }
}

/// The world pose of the vehicle's design space, from the chassis body (at
/// the design-space centre of mass `origin`).
fn design_frame(chassis: engine_schema::BodyState, origin: Vec3) -> Transform {
    let q = glam::Quat::from_array(chassis.orientation);
    let p = Vec3::from(chassis.position) - q * origin;
    Transform::new(p.to_array(), chassis.orientation)
}

pub(crate) fn compose(a: Transform, b: Transform) -> Transform {
    let q = a.q();
    Transform::new(
        (a.p() + q * b.p()).to_array(),
        (q * b.q()).normalize().to_array(),
    )
}

/// `world` expressed in the frame of `body` (both world poses).
fn relative(body: Transform, world: Transform) -> Transform {
    let inv = body.q().conjugate();
    Transform::new(
        (inv * (world.p() - body.p())).to_array(),
        (inv * world.q()).normalize().to_array(),
    )
}
