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
}

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
