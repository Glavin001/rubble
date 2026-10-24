//! Joint definitions after Box3D (`b3JointDef` base plus per-type defs), built
//! on Rubble's AVBD joint presets (`JointDesc`).
//!
//! Conventions follow Box3D: frames are measured from each body's origin;
//! revolute joints turn about frame z; prismatic joints slide along frame-A x;
//! spherical cones open about frame-A z and twist about frame-B z; angles are
//! radians. Rubble's presets use frame x as the hinge and twist axis, so the
//! frames are turned by -90° about y on the way in.
//!
//! One deliberate difference: AVBD springs are stiffnesses, not frequencies.
//! Box3D's `hertz` / `damping_ratio` need the effective mass, which the GPU
//! owns, so Rubble's springs take `stiffness` (N/m or N·m/rad) and `damping`.

use crate::error::unsupported;
use crate::{BodyId, Transform};
use engine_schema::{Dof, JointDesc, JointDrive, JointFrame, Motion, JOINT_BIG};
use glam::{Quat, Vec3};

/// Fields every joint has (Box3D `b3JointDef`).
#[derive(Clone, Debug, PartialEq)]
pub struct JointBase {
    pub body_a: BodyId,
    pub body_b: BodyId,
    /// Joint frame on body A, from its origin.
    pub local_frame_a: Transform,
    pub local_frame_b: Transform,
    /// Joint events above these (planned; non-default values are refused).
    pub force_threshold: f32,
    pub torque_threshold: f32,
    pub collide_connected: bool,
    /// Rubble: the joint breaks past this constraint force (N) or torque (N·m),
    /// raising `DestructionEvent::JointBroke`.
    pub break_force: f32,
    pub break_torque: f32,
    pub user_data: u64,
}

impl JointBase {
    pub fn new(body_a: BodyId, body_b: BodyId) -> Self {
        Self {
            body_a,
            body_b,
            local_frame_a: Transform::IDENTITY,
            local_frame_b: Transform::IDENTITY,
            force_threshold: f32::MAX,
            torque_threshold: f32::MAX,
            collide_connected: false,
            break_force: f32::INFINITY,
            break_torque: f32::INFINITY,
            user_data: 0,
        }
    }

    /// Both frames at `anchor_a` on A and `anchor_b` on B, unrotated.
    pub fn at(body_a: BodyId, anchor_a: [f32; 3], body_b: BodyId, anchor_b: [f32; 3]) -> Self {
        Self {
            local_frame_a: Transform::new(anchor_a, [0.0, 0.0, 0.0, 1.0]),
            local_frame_b: Transform::new(anchor_b, [0.0, 0.0, 0.0, 1.0]),
            ..Self::new(body_a, body_b)
        }
    }
}

/// A spring in physical units (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    pub stiffness: f32,
    pub damping: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RevoluteJointDef {
    pub base: JointBase,
    /// Joint angle that counts as zero (B relative to A), radians.
    pub target_angle: f32,
    /// Pulls the angle towards `target_angle`.
    pub spring: Option<Spring>,
    pub limit: Option<(f32, f32)>,
    /// `(speed rad/s, max torque N·m)`.
    pub motor: Option<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticJointDef {
    pub base: JointBase,
    pub target_translation: f32,
    pub spring: Option<Spring>,
    pub limit: Option<(f32, f32)>,
    /// `(speed m/s, max force N)`.
    pub motor: Option<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SphericalJointDef {
    pub base: JointBase,
    /// Cone half-angle about frame-A z, radians.
    pub cone_angle: Option<f32>,
    pub twist_limit: Option<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeldJointDef {
    pub base: JointBase,
    /// `None`: rigid. Otherwise penalty stiffnesses (N/m, N·m/rad).
    pub linear_stiffness: Option<f32>,
    pub angular_stiffness: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DistanceJointDef {
    pub base: JointBase,
    pub length: f32,
    /// A spring around `length` instead of a rigid rod.
    pub spring: Option<f32>,
    /// `(min, max)` length: a rope when `min` is 0.
    pub limit: Option<(f32, f32)>,
}

/// Disables collision between two bodies and nothing else.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterJointDef {
    pub base: JointBase,
}

/// Any joint `World::create_joint` accepts.
#[derive(Clone, Debug, PartialEq)]
pub enum JointDef {
    Revolute(RevoluteJointDef),
    Prismatic(PrismaticJointDef),
    Spherical(SphericalJointDef),
    Weld(WeldJointDef),
    Distance(DistanceJointDef),
    Filter(FilterJointDef),
    /// Rubble's full joint model (D6, bonds, cones, yield, drives): `desc`'s
    /// frames are in each body's frame from its origin; its body indices are
    /// replaced by `body_a` / `body_b`.
    Custom {
        body_a: BodyId,
        body_b: BodyId,
        desc: JointDesc,
        user_data: u64,
    },
}

impl JointDef {
    pub fn bodies(&self) -> (BodyId, BodyId) {
        match self {
            JointDef::Custom { body_a, body_b, .. } => (*body_a, *body_b),
            _ => {
                let b = self.base().expect("typed joint");
                (b.body_a, b.body_b)
            }
        }
    }

    pub fn user_data(&self) -> u64 {
        match self {
            JointDef::Custom { user_data, .. } => *user_data,
            _ => self.base().expect("typed joint").user_data,
        }
    }

    fn base(&self) -> Option<&JointBase> {
        Some(match self {
            JointDef::Revolute(d) => &d.base,
            JointDef::Prismatic(d) => &d.base,
            JointDef::Spherical(d) => &d.base,
            JointDef::Weld(d) => &d.base,
            JointDef::Distance(d) => &d.base,
            JointDef::Filter(d) => &d.base,
            JointDef::Custom { .. } => return None,
        })
    }

    /// The Rubble preset, with frames still in the body-origin frames.
    pub(crate) fn desc(&self) -> anyhow::Result<JointDesc> {
        let Some(base) = self.base() else {
            let JointDef::Custom { desc, .. } = self else {
                unreachable!()
            };
            return Ok(*desc);
        };
        if base.force_threshold != f32::MAX || base.torque_threshold != f32::MAX {
            return unsupported("joint event thresholds (planned P1.5b: GPU threshold events)");
        }
        let fa = frame(base.local_frame_a);
        let fb = frame(base.local_frame_b);
        // Box3D's z axis is Rubble's x axis.
        let z_to_x = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        let za = JointFrame {
            anchor: fa.anchor,
            rot: fa.rot * z_to_x,
        };
        let zb = JointFrame {
            anchor: fb.anchor,
            rot: fb.rot * z_to_x,
        };
        let mut d = match self {
            JointDef::Revolute(r) => {
                let turn = Quat::from_rotation_x(-r.target_angle);
                let mut d = JointDesc::hinge(
                    0,
                    0,
                    za,
                    JointFrame {
                        rot: zb.rot * turn,
                        ..zb
                    },
                );
                if let Some((lo, hi)) = r.limit {
                    d = d.with_twist_limit(lo, hi);
                }
                d.drive = drive(Dof::Twist, r.spring, r.motor)?;
                d
            }
            JointDef::Prismatic(p) => {
                let mut d = JointDesc::slider(0, 0, fa, fb);
                if let Some((lo, hi)) = p.limit {
                    d = d.with_linear_limit(Dof::X, lo, hi);
                }
                d.drive = drive(Dof::X, p.spring, p.motor)?;
                if let (Some(_), Some(dr)) = (p.spring, d.drive.as_mut()) {
                    dr.target = p.target_translation;
                }
                d
            }
            JointDef::Spherical(s) => {
                let mut d = match s.cone_angle {
                    Some(a) => JointDesc::cone(0, 0, za, zb, a, a),
                    None => {
                        let mut d = JointDesc::ball(0, 0, za.anchor, zb.anchor);
                        d.frame_a = za;
                        d.frame_b = zb;
                        d
                    }
                };
                if let Some((lo, hi)) = s.twist_limit {
                    d = d.with_twist_limit(lo, hi);
                }
                d
            }
            JointDef::Weld(w) => {
                let mut d = JointDesc::fixed(0, 0, fa, fb);
                if w.linear_stiffness.is_some() || w.angular_stiffness.is_some() {
                    d = d.with_stiffness(
                        w.linear_stiffness.unwrap_or(JOINT_BIG),
                        w.angular_stiffness.unwrap_or(JOINT_BIG),
                    );
                }
                d
            }
            JointDef::Distance(s) => {
                let (pa, pb) = (fa.anchor, fb.anchor);
                match (s.spring, s.limit) {
                    (Some(k), None) => JointDesc::spring(0, 0, pa, pb, s.length, k),
                    (None, Some((lo, hi))) => {
                        let mut d = JointDesc::distance(0, 0, pa, pb, s.length);
                        d.radial = Some((lo, hi));
                        d
                    }
                    (None, None) => JointDesc::distance(0, 0, pa, pb, s.length),
                    (Some(_), Some(_)) => {
                        return unsupported("a distance joint with both a spring and limits");
                    }
                }
            }
            JointDef::Filter(_) => {
                let mut d = JointDesc::d6(0, 0, fa, fb);
                d.motion = [Motion::Free; 6];
                d
            }
            JointDef::Custom { .. } => unreachable!(),
        };
        d.collide = base.collide_connected;
        if base.break_force.is_finite() || base.break_torque.is_finite() {
            d = d.with_break(finite(base.break_force), finite(base.break_torque));
        }
        Ok(d)
    }
}

fn frame(t: Transform) -> JointFrame {
    JointFrame {
        anchor: Vec3::from(t.position),
        rot: t.q(),
    }
}

fn finite(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        JOINT_BIG
    }
}

fn drive(
    dof: Dof,
    spring: Option<Spring>,
    motor: Option<(f32, f32)>,
) -> anyhow::Result<Option<JointDrive>> {
    Ok(match (spring, motor) {
        (Some(s), None) => Some(JointDrive::servo(
            dof,
            0.0,
            s.stiffness,
            s.damping,
            JOINT_BIG,
        )),
        (None, Some((speed, max))) => Some(JointDrive::motor(dof, speed, max)),
        (None, None) => None,
        (Some(_), Some(_)) => return unsupported("a spring and a motor on one joint axis"),
    })
}
