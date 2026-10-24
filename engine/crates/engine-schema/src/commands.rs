//! GPU command stream applied at the start of each step.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandType {
    Nop = 0,
    SpawnBody = 1,
    DestroyBody = 2,
    WakeBody = 3,
    TeleportBody = 4,
    SetVelocity = 5,
    SpawnCollider = 6,
    DestroyCollider = 7,
    ReparentCollider = 8,
    SpawnJoint = 9,
    DestroyJoint = 10,
    /// Force and torque for one step: `f0` = force (N), `f1` = torque (N m).
    AttachForce = 11,
    /// Impulse `f0` (N s) at world point `f1`.
    ApplyImpulse = 12,
    /// Force `f0` (N) at world point `f1`, for one step.
    AddForceAt = 13,
    /// Angular impulse `f0` (N m s).
    AngularImpulse = 14,
    /// Move the solver frame (centre of mass, principal axes) and set the mass.
    SetMassFrame = 15,
    /// Kinematic bodies: `f0` = target position, `f1` = target rotation
    /// (solver pose); the velocity that reaches them over the next step.
    SetKinematicTarget = 16,
    /// One step of damping and gravity scale: `v = f0.w v + f0.xyz`,
    /// `w = f1.x w`.
    Damp = 17,
    /// Motion locks: `f0.x` = the six lock bits.
    SetLocks = 18,
}

/// `GpuCommand::b` bit: a sleeping body ignores the command (Box3D `wake = false`).
pub const CMD_NO_WAKE: u32 = 1;
/// `GpuCommand::b` bit on `ApplyImpulse`: through the centre of mass.
pub const CMD_AT_CENTER: u32 = 2;
/// `GpuCommand::b` bit on `SetVelocity`: set the linear part only.
pub const CMD_LINEAR_ONLY: u32 = 4;
/// `GpuCommand::b` bit on `SetVelocity`: set the angular part only.
pub const CMD_ANGULAR_ONLY: u32 = 8;

/// `GpuCommand::c` of a command applied by the head of its body's run (the
/// head carries the run length; 0 is a lone command, one thread each).
pub const CMD_RUN_MEMBER: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCommand {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub f0: [f32; 4],
    pub f1: [f32; 4],
    pub f2: [f32; 4],
}

impl GpuCommand {
    pub fn nop() -> Self {
        Self::zeroed()
    }

    pub fn wake(body: u32) -> Self {
        Self {
            op: CommandType::WakeBody as u32,
            a: body,
            ..Self::zeroed()
        }
    }

    pub fn teleport(body: u32, pos: [f32; 3], quat: [f32; 4]) -> Self {
        Self {
            op: CommandType::TeleportBody as u32,
            a: body,
            f0: [pos[0], pos[1], pos[2], 0.0],
            f1: quat,
            ..Self::zeroed()
        }
    }

    pub fn set_velocity(body: u32, lin: [f32; 3], ang: [f32; 3]) -> Self {
        Self {
            op: CommandType::SetVelocity as u32,
            a: body,
            f0: [lin[0], lin[1], lin[2], 0.0],
            f1: [ang[0], ang[1], ang[2], 0.0],
            ..Self::zeroed()
        }
    }

    /// Impulse `j` (N s) at world `point`: dv = j/m, dw = I_world^-1 (r x j).
    pub fn apply_impulse(body: u32, j: [f32; 3], point: [f32; 3]) -> Self {
        Self {
            op: CommandType::ApplyImpulse as u32,
            a: body,
            f0: [j[0], j[1], j[2], 0.0],
            f1: [point[0], point[1], point[2], 0.0],
            ..Self::zeroed()
        }
    }

    /// Force (N) through the centre of mass and torque (N m) for the next step
    /// only: dv = F dt / m, dw = I_world^-1 tau dt. Submit again each step.
    pub fn add_force(body: u32, force: [f32; 3], torque: [f32; 3]) -> Self {
        Self {
            op: CommandType::AttachForce as u32,
            a: body,
            f0: [force[0], force[1], force[2], 0.0],
            f1: [torque[0], torque[1], torque[2], 0.0],
            ..Self::zeroed()
        }
    }

    /// Set the linear velocity, keeping the angular.
    pub fn set_linear_velocity(body: u32, v: [f32; 3]) -> Self {
        Self {
            b: CMD_LINEAR_ONLY,
            ..Self::set_velocity(body, v, [0.0; 3])
        }
    }

    /// Set the angular velocity, keeping the linear.
    pub fn set_angular_velocity(body: u32, w: [f32; 3]) -> Self {
        Self {
            b: CMD_ANGULAR_ONLY,
            ..Self::set_velocity(body, [0.0; 3], w)
        }
    }

    /// Force (N) at world `point` for the next step only.
    pub fn add_force_at(body: u32, force: [f32; 3], point: [f32; 3]) -> Self {
        Self {
            op: CommandType::AddForceAt as u32,
            a: body,
            f0: [force[0], force[1], force[2], 0.0],
            f1: [point[0], point[1], point[2], 0.0],
            ..Self::zeroed()
        }
    }

    /// Impulse `j` (N s) through the centre of mass.
    pub fn apply_impulse_at_center(body: u32, j: [f32; 3]) -> Self {
        Self {
            op: CommandType::ApplyImpulse as u32,
            a: body,
            b: CMD_AT_CENTER,
            f0: [j[0], j[1], j[2], 0.0],
            ..Self::zeroed()
        }
    }

    /// Angular impulse (N m s): dw = I_world^-1 L.
    pub fn angular_impulse(body: u32, l: [f32; 3]) -> Self {
        Self {
            op: CommandType::AngularImpulse as u32,
            a: body,
            f0: [l[0], l[1], l[2], 0.0],
            ..Self::zeroed()
        }
    }

    /// Re-centre a body on new mass properties without moving its origin:
    /// `offset` is the new centre in the old body frame, `rotation` the new
    /// principal axes relative to the old, then the inverse principal inertia
    /// and inverse mass.
    pub fn damp(body: u32, add: [f32; 3], linear: f32, angular: f32) -> Self {
        Self {
            op: CommandType::Damp as u32,
            a: body,
            f0: [add[0], add[1], add[2], linear],
            f1: [angular, 0.0, 0.0, 0.0],
            ..Self::zeroed()
        }
    }

    pub fn set_locks(body: u32, locks: u32) -> Self {
        Self {
            op: CommandType::SetLocks as u32,
            a: body,
            f0: [f32::from_bits(locks), 0.0, 0.0, 0.0],
            ..Self::zeroed()
        }
    }

    pub fn set_kinematic_target(body: u32, position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            op: CommandType::SetKinematicTarget as u32,
            a: body,
            f0: [position[0], position[1], position[2], 0.0],
            f1: rotation,
            ..Self::zeroed()
        }
    }

    pub fn set_mass_frame(
        body: u32,
        offset: [f32; 3],
        rotation: [f32; 4],
        inv_inertia: [f32; 3],
        inv_mass: f32,
    ) -> Self {
        Self {
            op: CommandType::SetMassFrame as u32,
            a: body,
            f0: [offset[0], offset[1], offset[2], 0.0],
            f1: rotation,
            f2: [inv_inertia[0], inv_inertia[1], inv_inertia[2], inv_mass],
            ..Self::zeroed()
        }
    }

    /// This command, ignored by a sleeping body instead of waking it.
    pub fn without_wake(mut self) -> Self {
        self.b |= CMD_NO_WAKE;
        self
    }

    pub fn reparent(
        collider: u32,
        new_body: u32,
        local_pos: [f32; 3],
        local_rot: [f32; 4],
    ) -> Self {
        Self {
            op: CommandType::ReparentCollider as u32,
            a: collider,
            b: new_body,
            f0: [local_pos[0], local_pos[1], local_pos[2], 0.0],
            f1: local_rot,
            ..Self::zeroed()
        }
    }
}
