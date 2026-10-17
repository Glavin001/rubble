//! GPU joint record (288 B) and host-side presets.
//!
//! Every joint is a D6 descriptor: six degrees of freedom (X, Y, Z in joint
//! frame A, then twist about frame X and the two swings about frame Y / Z),
//! each LOCKED, LIMITED or FREE. Ball, fixed, hinge, slider, cone, distance and
//! spring are presets of that descriptor (PhysX `PxD6JointCreate`). Solver rows
//! and their AVBD state live in `slang/physics/joints.slang`.
//!
//! Row slots: 0..2 linear (slot 0 is the radial row in distance mode), 3 twist,
//! 4 swing1 (the cone row in cone mode), 5 swing2, 6 drive.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Stiffness at or above this is a hard (augmented-Lagrangian) row; below it the
/// row is a plain penalty spring with exactly that stiffness (three-avbd `HARD`).
pub const JOINT_HARD: f32 = 1.0e30;
/// Stands in for "unbounded" (break thresholds, limits); never infinity on GPU.
pub const JOINT_BIG: f32 = 3.0e38;

pub const JOINT_FLAG_BROKEN: u32 = 1 << 0;
/// Let the two bodies collide (PhysX `eCOLLISION_ENABLED`). Off by default.
pub const JOINT_FLAG_COLLIDE: u32 = 1 << 1;
/// Linear DOFs are replaced by one radial row on |pA - pB| (distance / spring).
pub const JOINT_FLAG_RADIAL: u32 = 1 << 2;
/// Limited swings form one elliptical cone row instead of two angle rows.
pub const JOINT_FLAG_CONE: u32 = 1 << 3;
pub const JOINT_FLAG_DRIVE: u32 = 1 << 4;
/// Fracture bond: a weld whose six rows run in the bond frame with fixed
/// per-row penalties, breaking by Shardwright's utilization against
/// `JointDesc::bond` capacities (stored in `lin_lo` and `twist_lo`).
pub const JOINT_FLAG_BOND: u32 = 1 << 5;
pub const JOINT_DRIVE_DOF_SHIFT: u32 = 8;
pub const JOINT_PRESET_SHIFT: u32 = 24;

pub const MOTION_LOCKED: u32 = 0;
pub const MOTION_LIMITED: u32 = 1;
pub const MOTION_FREE: u32 = 2;

pub const JOINT_ROWS: usize = 7;
pub const ROW_DRIVE: usize = 6;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointType {
    Ball = 0,
    Fixed = 1,
    Hinge = 2,
    Slider = 3,
    D6 = 4,
    Distance = 5,
    MaxDistance = 6,
    Spring = 7,
    Cone = 8,
    /// Breakable fracture bond (`JointDesc::bond`).
    Bond = 9,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dof {
    X = 0,
    Y = 1,
    Z = 2,
    Twist = 3,
    Swing1 = 4,
    Swing2 = 5,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Motion {
    Locked = MOTION_LOCKED,
    Limited = MOTION_LIMITED,
    Free = MOTION_FREE,
}

/// GPU joint record. Must match `struct Joint` in `slang/physics/common.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Joint {
    pub body_a: u32,
    pub body_b: u32,
    pub flags: u32,
    /// 2 bits per DOF (X, Y, Z, twist, swing1, swing2): 0 locked, 1 limited, 2 free.
    pub motion: u32,
    pub anchor_a: [f32; 3],
    /// Angular rows are scaled by this length so their penalties live in the
    /// linear rows' range (three-avbd `torqueArm`).
    pub torque_arm: f32,
    /// Joint frame A orientation in body A (PhysX localPose0.q).
    pub frame_a: [f32; 4],
    pub anchor_b: [f32; 3],
    pub lin_stiffness: f32,
    /// Joint frame B orientation in body B. Plastic yield rewrites it.
    pub frame_b: [f32; 4],
    /// `frame_b` at creation; bend for `break_bend` is measured from here.
    pub rest_start: [f32; 4],
    pub lin_lo: [f32; 3],
    pub twist_lo: f32,
    pub lin_hi: [f32; 3],
    pub twist_hi: f32,
    /// Swing limits (cone semi-axes, or symmetric pyramid half-angles).
    pub swing_y: f32,
    pub swing_z: f32,
    pub radial_min: f32,
    pub radial_max: f32,
    pub ang_stiffness: f32,
    /// 0 = hard limits; > 0 = soft limit spring (PhysX `PxJointLimitParameters`).
    pub limit_stiffness: f32,
    pub limit_damping: f32,
    pub drive_force_limit: f32,
    pub break_force: f32,
    pub break_torque: f32,
    pub yield_torque: f32,
    pub break_bend: f32,
    pub drive_target: f32,
    pub drive_velocity: f32,
    /// 0 = hard velocity motor; > 0 = position servo spring.
    pub drive_stiffness: f32,
    pub drive_damping: f32,
    pub lambda: [f32; 8],
    pub penalty: [f32; 8],
    /// Row coordinate at the start of the step (C0 / x⁻ for damping and motors).
    pub c0: [f32; 8],
}

impl Joint {
    pub fn is_broken(&self) -> bool {
        self.flags & JOINT_FLAG_BROKEN != 0
    }

    pub fn motion_of(&self, dof: Dof) -> Motion {
        match (self.motion >> (2 * dof as u32)) & 3 {
            MOTION_LOCKED => Motion::Locked,
            MOTION_LIMITED => Motion::Limited,
            _ => Motion::Free,
        }
    }

    /// Bond break capacities: tension (N), compression (N), shear (N), bending
    /// moment (N·m); <= 0 means that mode never breaks.
    pub fn bond_strength(&self) -> [f32; 4] {
        [
            self.lin_lo[0],
            self.lin_lo[1],
            self.lin_lo[2],
            self.twist_lo,
        ]
    }

    pub fn preset(&self) -> u32 {
        self.flags >> JOINT_PRESET_SHIFT
    }

    /// Constraint force magnitude carried by the linear rows (N).
    pub fn linear_force(&self) -> f32 {
        Vec3::from_slice(&self.lambda[0..3]).length()
    }

    /// Constraint torque magnitude carried by the angular rows (N·m).
    pub fn angular_torque(&self) -> f32 {
        Vec3::from_slice(&self.lambda[3..6]).length() * self.torque_arm
    }

    /// Plastic bend accumulated since creation (rad), as in three-avbd `jointBend`.
    pub fn bend(&self) -> f32 {
        let d = Quat::from_array(self.rest_start).conjugate() * Quat::from_array(self.frame_b);
        2.0 * d.xyz().length().atan2(d.w.abs())
    }
}

/// A rigid frame on one body: anchor point and orientation, both body-local.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointFrame {
    pub anchor: Vec3,
    pub rot: Quat,
}

impl JointFrame {
    pub fn at(anchor: Vec3) -> Self {
        Self {
            anchor,
            rot: Quat::IDENTITY,
        }
    }

    /// Frame whose X axis is `axis` (hinge / slider / twist axis).
    pub fn with_axis(anchor: Vec3, axis: Vec3) -> Self {
        Self {
            anchor,
            rot: Quat::from_rotation_arc(Vec3::X, axis.normalize_or(Vec3::X)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDrive {
    pub dof: Dof,
    /// Position target (m or rad); only used when `stiffness > 0`.
    pub target: f32,
    /// Velocity target (m/s or rad/s).
    pub velocity: f32,
    /// 0 = hard velocity motor (three-avbd motor row); > 0 = position servo.
    pub stiffness: f32,
    pub damping: f32,
    /// Maximum force (N) or torque (N·m) the drive may apply.
    pub force_limit: f32,
}

impl JointDrive {
    pub fn motor(dof: Dof, velocity: f32, force_limit: f32) -> Self {
        Self {
            dof,
            target: 0.0,
            velocity,
            stiffness: 0.0,
            damping: 0.0,
            force_limit,
        }
    }

    pub fn servo(dof: Dof, target: f32, stiffness: f32, damping: f32, force_limit: f32) -> Self {
        Self {
            dof,
            target,
            velocity: 0.0,
            stiffness,
            damping,
            force_limit,
        }
    }
}

/// Host description of a joint. Build one with a preset, adjust, then `to_gpu`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDesc {
    pub preset: JointType,
    pub body_a: u32,
    pub body_b: u32,
    pub frame_a: JointFrame,
    pub frame_b: JointFrame,
    pub motion: [Motion; 6],
    pub lin_lo: Vec3,
    pub lin_hi: Vec3,
    pub twist: (f32, f32),
    pub swing: (f32, f32),
    pub cone: bool,
    pub radial: Option<(f32, f32)>,
    pub lin_stiffness: f32,
    pub ang_stiffness: f32,
    pub limit_stiffness: f32,
    pub limit_damping: f32,
    pub break_force: f32,
    pub break_torque: f32,
    pub yield_torque: f32,
    pub break_bend: f32,
    pub drive: Option<JointDrive>,
    pub collide: bool,
    /// 0 lets the host pick from the bodies' collider sizes.
    pub torque_arm: f32,
    pub bond: Option<BondSpec>,
}

/// Fracture bond rows: fixed penalties in the bond frame and break capacities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondSpec {
    /// Penalties (normal, u, v, about normal, about u, about v), in N/m and
    /// N·m/rad.
    pub k: [f32; 6],
    /// Tension (N), compression (N), shear (N), bending moment (N·m); <= 0
    /// never breaks in that mode.
    pub strength: [f32; 4],
}

impl JointDesc {
    fn base(preset: JointType, a: u32, b: u32, fa: JointFrame, fb: JointFrame) -> Self {
        Self {
            preset,
            body_a: a,
            body_b: b,
            frame_a: fa,
            frame_b: fb,
            motion: [Motion::Locked; 6],
            lin_lo: Vec3::ZERO,
            lin_hi: Vec3::ZERO,
            twist: (0.0, 0.0),
            swing: (0.0, 0.0),
            cone: false,
            radial: None,
            lin_stiffness: JOINT_BIG,
            ang_stiffness: JOINT_BIG,
            limit_stiffness: 0.0,
            limit_damping: 0.0,
            break_force: JOINT_BIG,
            break_torque: JOINT_BIG,
            yield_torque: JOINT_BIG,
            break_bend: JOINT_BIG,
            drive: None,
            collide: false,
            torque_arm: 0.0,
            bond: None,
        }
    }

    /// Spherical joint: anchors coincide, rotation free.
    pub fn ball(a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        let mut d = Self::base(
            JointType::Ball,
            a,
            b,
            JointFrame::at(anchor_a),
            JointFrame::at(anchor_b),
        );
        d.motion[3..].fill(Motion::Free);
        d
    }

    /// Weld: all six DOFs locked.
    pub fn fixed(a: u32, b: u32, frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self::base(JointType::Fixed, a, b, frame_a, frame_b)
    }

    /// Revolute about frame X; twist free until `with_twist_limit`.
    pub fn hinge(a: u32, b: u32, frame_a: JointFrame, frame_b: JointFrame) -> Self {
        let mut d = Self::base(JointType::Hinge, a, b, frame_a, frame_b);
        d.motion[Dof::Twist as usize] = Motion::Free;
        d
    }

    /// Prismatic along frame X; free until `with_linear_limit`.
    pub fn slider(a: u32, b: u32, frame_a: JointFrame, frame_b: JointFrame) -> Self {
        let mut d = Self::base(JointType::Slider, a, b, frame_a, frame_b);
        d.motion[Dof::X as usize] = Motion::Free;
        d
    }

    /// Ball joint whose frame-B X axis stays inside an elliptical cone about
    /// frame-A X (semi-angles about Y and Z), twist optionally limited.
    pub fn cone(
        a: u32,
        b: u32,
        frame_a: JointFrame,
        frame_b: JointFrame,
        swing_y: f32,
        swing_z: f32,
    ) -> Self {
        let mut d = Self::base(JointType::Cone, a, b, frame_a, frame_b);
        d.motion[Dof::Twist as usize] = Motion::Free;
        d.motion[Dof::Swing1 as usize] = Motion::Limited;
        d.motion[Dof::Swing2 as usize] = Motion::Limited;
        d.swing = (swing_y, swing_z);
        d.cone = true;
        d
    }

    /// Rigid rod: |pA - pB| = length, rotation free.
    pub fn distance(a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, length: f32) -> Self {
        let mut d = Self::ball(a, b, anchor_a, anchor_b);
        d.preset = JointType::Distance;
        d.radial = Some((length, length));
        d
    }

    /// Tether: |pA - pB| <= max_length.
    pub fn max_distance(a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, max_length: f32) -> Self {
        let mut d = Self::distance(a, b, anchor_a, anchor_b, max_length);
        d.preset = JointType::MaxDistance;
        d.radial = Some((0.0, max_length));
        d
    }

    /// Spring with rest length (three-avbd `T_SPRING`): soft radial row.
    pub fn spring(
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest: f32,
        stiffness: f32,
    ) -> Self {
        let mut d = Self::distance(a, b, anchor_a, anchor_b, rest);
        d.preset = JointType::Spring;
        d.lin_stiffness = stiffness;
        d
    }

    /// Fracture bond at a bond centroid (Shardwright): a weld whose rows run in
    /// the bond frame (normal = A→B bond normal, u, v), all in body A's frame,
    /// at rest when B's orientation is A's times `rest` (qB = qA · rest).
    pub fn bond(
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        normal: Vec3,
        frame_u: Vec3,
        rest: Quat,
        spec: BondSpec,
    ) -> Self {
        let n = normal.normalize_or(Vec3::X);
        let u = (frame_u - n * frame_u.dot(n)).normalize_or(n.any_orthonormal_vector());
        let frame = Quat::from_mat3(&glam::Mat3::from_cols(n, u, n.cross(u))).normalize();
        let fa = JointFrame {
            anchor: anchor_a,
            rot: frame,
        };
        let fb = JointFrame {
            anchor: anchor_b,
            rot: (rest.conjugate() * frame).normalize(),
        };
        let mut d = Self::base(JointType::Bond, a, b, fa, fb);
        d.torque_arm = 1.0;
        d.bond = Some(spec);
        d
    }

    /// Generic D6; every DOF starts locked.
    pub fn d6(a: u32, b: u32, frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self::base(JointType::D6, a, b, frame_a, frame_b)
    }

    pub fn with_motion(mut self, dof: Dof, motion: Motion) -> Self {
        self.motion[dof as usize] = motion;
        self
    }

    /// Limit translation along frame-A X (slider travel).
    pub fn with_linear_limit(mut self, dof: Dof, lo: f32, hi: f32) -> Self {
        let i = dof as usize;
        assert!(i < 3, "linear limit on an angular DOF");
        self.motion[i] = Motion::Limited;
        self.lin_lo[i] = lo;
        self.lin_hi[i] = hi;
        self
    }

    pub fn with_twist_limit(mut self, lo: f32, hi: f32) -> Self {
        self.motion[Dof::Twist as usize] = Motion::Limited;
        self.twist = (lo, hi);
        self
    }

    pub fn with_soft_limits(mut self, stiffness: f32, damping: f32) -> Self {
        self.limit_stiffness = stiffness;
        self.limit_damping = damping;
        self
    }

    pub fn with_drive(mut self, drive: JointDrive) -> Self {
        self.drive = Some(drive);
        self
    }

    /// Breaks once the constraint force or torque exceeds these (PhysX
    /// `setBreakForce`). Use `JOINT_BIG` for "never".
    pub fn with_break(mut self, force: f32, torque: f32) -> Self {
        self.break_force = force;
        self.break_torque = torque;
        self
    }

    /// Plastic weld: past `torque` the rest pose follows the bodies; past
    /// `bend` radians of accumulated bend it breaks (three-avbd yield/breakBend).
    pub fn with_yield(mut self, torque: f32, bend: f32) -> Self {
        self.yield_torque = torque;
        self.break_bend = bend;
        self
    }

    /// Finite stiffness makes the rows penalty springs (soft joint, mouse drag).
    pub fn with_stiffness(mut self, linear: f32, angular: f32) -> Self {
        self.lin_stiffness = linear;
        self.ang_stiffness = angular;
        self
    }

    pub fn with_collision(mut self, collide: bool) -> Self {
        self.collide = collide;
        self
    }

    pub fn with_torque_arm(mut self, arm: f32) -> Self {
        self.torque_arm = arm;
        self
    }

    /// Set frame B's orientation so the joint is at rest in the bodies' current
    /// orientations (three-avbd `rest: 'current'`).
    pub fn rest_from_bodies(mut self, rot_a: Quat, rot_b: Quat) -> Self {
        self.frame_b.rot = (rot_b.conjugate() * rot_a * self.frame_a.rot).normalize();
        self
    }

    pub fn to_gpu(&self) -> Joint {
        let mut motion = 0u32;
        for (i, m) in self.motion.iter().enumerate() {
            motion |= (*m as u32) << (2 * i);
        }
        let mut flags = (self.preset as u32) << JOINT_PRESET_SHIFT;
        if self.collide {
            flags |= JOINT_FLAG_COLLIDE;
        }
        if self.cone {
            flags |= JOINT_FLAG_CONE;
        }
        let (radial_min, radial_max) = match self.radial {
            Some(r) => {
                flags |= JOINT_FLAG_RADIAL;
                r
            }
            None => (0.0, 0.0),
        };
        let mut j = Joint::zeroed();
        if let Some(d) = self.drive {
            flags |= JOINT_FLAG_DRIVE | ((d.dof as u32) << JOINT_DRIVE_DOF_SHIFT);
            j.drive_target = d.target;
            j.drive_velocity = d.velocity;
            j.drive_stiffness = d.stiffness;
            j.drive_damping = d.damping;
            j.drive_force_limit = if d.force_limit > 0.0 {
                d.force_limit
            } else {
                JOINT_BIG
            };
        }
        if let Some(bond) = self.bond {
            flags |= JOINT_FLAG_BOND;
            // Slots 0..2 normal/u/v, 3..5 about normal/u/v.
            j.penalty[..6].copy_from_slice(&bond.k);
            j.lin_lo = [bond.strength[0], bond.strength[1], bond.strength[2]];
            j.twist_lo = bond.strength[3];
        }
        let frame_b = self.frame_b.rot.normalize().to_array();
        j.body_a = self.body_a;
        j.body_b = self.body_b;
        j.flags = flags;
        j.motion = motion;
        j.anchor_a = self.frame_a.anchor.to_array();
        j.torque_arm = if self.torque_arm > 0.0 {
            self.torque_arm
        } else {
            1.0
        };
        j.frame_a = self.frame_a.rot.normalize().to_array();
        j.anchor_b = self.frame_b.anchor.to_array();
        j.lin_stiffness = self.lin_stiffness;
        j.frame_b = frame_b;
        j.rest_start = frame_b;
        if self.bond.is_none() {
            j.lin_lo = self.lin_lo.to_array();
            j.twist_lo = self.twist.0;
        }
        j.lin_hi = self.lin_hi.to_array();
        j.twist_hi = self.twist.1;
        j.swing_y = self.swing.0;
        j.swing_z = self.swing.1;
        j.radial_min = radial_min;
        j.radial_max = radial_max;
        j.ang_stiffness = self.ang_stiffness;
        j.limit_stiffness = self.limit_stiffness;
        j.limit_damping = self.limit_damping;
        j.break_force = self.break_force;
        j.break_torque = self.break_torque;
        j.yield_torque = self.yield_torque;
        j.break_bend = self.break_bend;
        j
    }
}

/// Bounding reach of a collider from its body's center (None for planes).
pub fn collider_reach(col: &crate::Collider) -> Option<f32> {
    if col.shape == crate::ShapeType::Plane as u32 {
        return None;
    }
    Some(Vec3::from_array(col.local_pos).length() + col.params[3])
}

/// three-avbd torque arm `|size_a + size_b|²` from each dynamic body's reach
/// (0 for static ends). Keeps angular penalties in the linear rows' range.
pub fn default_torque_arm(reach_a: f32, reach_b: f32) -> f32 {
    let size = 2.0 * (reach_a + reach_b);
    (size * size).max(1.0e-3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_bits_round_trip() {
        let j = JointDesc::hinge(1, 2, JointFrame::at(Vec3::ZERO), JointFrame::at(Vec3::ZERO))
            .with_twist_limit(-1.0, 1.0)
            .to_gpu();
        assert_eq!(j.motion_of(Dof::X), Motion::Locked);
        assert_eq!(j.motion_of(Dof::Twist), Motion::Limited);
        assert_eq!(j.motion_of(Dof::Swing2), Motion::Locked);
        assert_eq!(j.preset(), JointType::Hinge as u32);
        assert!(!j.is_broken());
    }

    #[test]
    fn bond_frame_and_rest() {
        let rest = Quat::from_rotation_z(0.3);
        let spec = BondSpec {
            k: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            strength: [10.0, 20.0, 30.0, 40.0],
        };
        let d = JointDesc::bond(1, 2, Vec3::X, -Vec3::X, Vec3::Y, Vec3::Z, rest, spec);
        let j = d.to_gpu();
        assert!(j.flags & JOINT_FLAG_BOND != 0);
        assert_eq!(j.bond_strength(), [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(&j.penalty[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let fa = Quat::from_array(j.frame_a);
        assert!(
            (fa * Vec3::X - Vec3::Y).length() < 1e-5,
            "frame X is the normal"
        );
        assert!((fa * Vec3::Y - Vec3::Z).length() < 1e-5, "frame Y is u");
        // At rest (qB = qA · rest) both world frames coincide.
        let qa = Quat::from_rotation_y(0.4);
        let qb = qa * rest;
        assert!((qa * fa).dot(qb * Quat::from_array(j.frame_b)).abs() > 0.99999);
    }

    #[test]
    fn rest_from_bodies_zeroes_relative_rotation() {
        let qa = Quat::from_rotation_y(0.7);
        let qb = Quat::from_rotation_z(-0.3);
        let d = JointDesc::fixed(
            0,
            1,
            JointFrame::with_axis(Vec3::ZERO, Vec3::Y),
            JointFrame::at(Vec3::ZERO),
        )
        .rest_from_bodies(qa, qb);
        let ca = qa * d.frame_a.rot;
        let cb = qb * d.frame_b.rot;
        assert!(ca.dot(cb).abs() > 0.99999);
    }
}
