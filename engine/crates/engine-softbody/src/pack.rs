//! Build the GPU records of one soft vehicle (design step 4): particles and node
//! colliders for `gBodies` / `gColliders`, beam records and per-particle
//! adjacency and colours for `gSoftData`, and the stage-1 rigid core: a
//! `VehicleDesc` carrying the raycast wheels and drivetrain, attached to the
//! frame nodes by ball joints.

use crate::color::color_graph;
use crate::desc::*;
use crate::layout::*;
use crate::model::*;
use engine_schema::convex::CollisionFilter;
use engine_schema::{BodyState, Collider, ShapeType};
use engine_vehicle::{
    AntiRollDesc, AssistDesc, Curve, DifferentialDesc, DriveDesc, DrivetrainDesc, EngineDesc,
    PartColliderDesc, PartDesc, PartRole, ShapeDesc, Spring, SteeringDesc, TransmissionDesc,
    VehicleDesc, WheelDesc,
};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Spawn options for a soft vehicle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpawnOptions {
    /// Collision sphere radius of every colliding node (m).
    pub node_radius: f32,
    /// `gMaterials` index of node colliders (its friction is the node friction).
    pub node_material: u32,
    /// Fraction of every node's mass moved into the rigid core (engine, gearbox, axles).
    pub core_mass_fraction: f32,
    /// Build the rigid core with raycast wheels and drivetrain (stage 1).
    pub rigid_core: bool,
    /// Natural frequency (Hz) and damping ratio of the raycast suspension.
    pub suspension_hz: f32,
    pub suspension_zeta: f32,
    pub suspension_travel: f32,
    /// Initial linear velocity of every particle (and the core).
    pub velocity: [f32; 3],
}

impl Default for SpawnOptions {
    fn default() -> Self {
        Self {
            node_radius: 0.06,
            node_material: 9,
            core_mass_fraction: 0.25,
            rigid_core: true,
            suspension_hz: 1.6,
            suspension_zeta: 0.45,
            suspension_travel: 0.3,
            velocity: [0.0; 3],
        }
    }
}

/// The rigid core of a vehicle (stage 1): the frame proxy carrying the wheels.
#[derive(Clone, Debug)]
pub struct CorePlan {
    pub vehicle: VehicleDesc,
    /// Desc-space origin of the core body.
    pub origin: Vec3,
    /// Frame nodes (desc node indices) the core attaches to.
    pub frame_nodes: Vec<u32>,
}

/// One soft vehicle's records, with particle indices relative to its first particle.
#[derive(Clone, Debug)]
pub struct PackedSoft {
    pub bodies: Vec<BodyState>,
    pub colliders: Vec<Collider>,
    /// Collider index (relative) per node, `u32::MAX` when the node has none.
    pub node_collider: Vec<u32>,
    pub beams: Vec<Beam>,
    /// Per node: beam indices (relative) touching it.
    pub adjacency: Vec<Vec<u32>>,
    pub colors: Vec<u32>,
    pub color_count: u32,
    pub core: Option<CorePlan>,
    pub masses: Vec<f32>,
}

/// World transform applied to desc-space points.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Placement {
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.position + self.rotation * p
    }
}

/// Unit forward of the vehicle in desc space (from the reference nodes, else the
/// steering axle, else −X like most Rigs of Rods trucks).
pub fn forward_axis(desc: &SoftVehicleDesc) -> Vec3 {
    let p = |i: u32| Vec3::from_array(desc.nodes[i as usize].position);
    if let Some(r) = desc.refs {
        let f = p(r.center) - p(r.back);
        let f = Vec3::new(f.x, 0.0, f.z);
        if f.length() > 1e-4 {
            return f.normalize();
        }
    }
    let mid = |w: &SoftWheel| 0.5 * (p(w.node1) + p(w.node2));
    let steer: Vec<Vec3> = desc
        .wheels
        .iter()
        .filter(|w| w.steerable)
        .map(mid)
        .collect();
    let other: Vec<Vec3> = desc
        .wheels
        .iter()
        .filter(|w| !w.steerable)
        .map(mid)
        .collect();
    if !steer.is_empty() && !other.is_empty() {
        let a = steer.iter().copied().sum::<Vec3>() / steer.len() as f32;
        let b = other.iter().copied().sum::<Vec3>() / other.len() as f32;
        let f = Vec3::new(a.x - b.x, 0.0, a.z - b.z);
        if f.length() > 1e-4 {
            return f.normalize();
        }
    }
    -Vec3::X
}

/// Frame nodes the rigid core attaches to: each wheel's axle node pair (the hub
/// structure). Only these are held rigid, so the bodywork between and around
/// the axles stays free to crumple.
pub fn frame_nodes(desc: &SoftVehicleDesc) -> Vec<u32> {
    let mut set = std::collections::BTreeSet::new();
    for w in &desc.wheels {
        set.insert(w.node1);
        set.insert(w.node2);
    }
    set.into_iter()
        .filter(|&i| !desc.nodes[i as usize].fixed)
        .collect()
}

pub fn pack_soft(
    desc: &SoftVehicleDesc,
    place: Placement,
    opts: &SpawnOptions,
    body_index: u32,
    group: i16,
) -> PackedSoft {
    let n = desc.nodes.len();
    let mut masses: Vec<f32> = desc.nodes.iter().map(|nd| nd.mass.max(1.0e-3)).collect();
    let core = match &desc.core {
        // An authored core: a whole vehicle, jointed to its nodes as given.
        Some(c) if opts.rigid_core => Some(CorePlan {
            vehicle: c.vehicle.clone(),
            origin: Vec3::from_array(c.vehicle.center_of_mass()),
            frame_nodes: c.nodes.clone(),
        }),
        _ if opts.rigid_core && !desc.wheels.is_empty() => Some(core_plan(desc, &mut masses, opts)),
        _ => None,
    };
    let vel = Vec3::from_array(opts.velocity);
    let mut bodies = Vec::with_capacity(n);
    for (i, nd) in desc.nodes.iter().enumerate() {
        let p = place.apply(Vec3::from_array(nd.position));
        let mut b = BodyState::dynamic(p, Quat::IDENTITY, masses[i], Vec3::ZERO);
        b.inv_inertia = [0.0; 3];
        b.flags |= BODY_PARTICLE;
        if nd.fixed {
            b.inv_mass = 0.0;
            b.flags &= !1;
        } else {
            b.linear_vel = vel.to_array();
        }
        b.initial_pos = p.to_array();
        bodies.push(b);
    }
    let mut colliders = Vec::new();
    let mut node_collider = vec![u32::MAX; n];
    for (i, nd) in desc.nodes.iter().enumerate() {
        if !nd.collision {
            continue;
        }
        let mut c = Collider {
            body: i as u32,
            shape: ShapeType::Sphere as u32,
            material: opts.node_material,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [opts.node_radius, 0.0, 0.0, 0.0],
        };
        CollisionFilter {
            group,
            ..Default::default()
        }
        .apply(&mut c);
        node_collider[i] = colliders.len() as u32;
        colliders.push(c);
    }
    let core_nodes: Vec<u32> = core.as_ref().map_or(Vec::new(), |c| c.frame_nodes.clone());
    let mut beams = Vec::with_capacity(desc.beams.len());
    let mut adjacency = vec![Vec::new(); n];
    for sb in &desc.beams {
        let l0 = desc.rest_length(sb);
        let rest = l0 * sb.precompression.max(1e-3);
        let kind = sb.kind.code();
        let (deform_c, deform_t) = match sb.kind {
            // Bounded beams never deform: the slot holds the stiff limit spring
            // (JBeam beamLimitSpring via `deform`; RoR's default beam spring otherwise).
            BeamKind::Bounded => (
                if sb.deform.is_finite() && sb.deform < f32::MAX {
                    sb.deform
                } else {
                    crate::truck::DEFAULT_SPRING
                },
                f32::INFINITY,
            ),
            BeamKind::Normal => (sb.deform, sb.deform),
            _ => (f32::INFINITY, f32::INFINITY),
        };
        let (short_bound, long_bound) = match sb.kind {
            BeamKind::Bounded | BeamKind::Support => (sb.short_bound, sb.long_bound),
            _ => (0.0, 0.0),
        };
        let beam = Beam {
            a: body_index + sb.a,
            b: body_index + sb.b,
            rest,
            rest0: rest,
            spring: sb.spring,
            damp: sb.damp,
            deform_c,
            strength: sb.strength,
            // Stage 1: the raycast wheels steer, and the core holds the axle nodes,
            // so a hydro on an axle node keeps its length instead of fighting the core.
            coef: if sb.kind == BeamKind::Hydro {
                if core_nodes.contains(&sb.a) || core_nodes.contains(&sb.b) {
                    0.0
                } else {
                    sb.hydro_factor
                }
            } else {
                sb.plastic_coef
            },
            short_bound,
            long_bound,
            kind,
            flags: if sb.visible { 0 } else { BEAM_INVISIBLE },
            body: 0,
            group: sb.break_group,
            penalty: sb.spring,
            lambda: 0.0,
            deform_t,
            force: 0.0,
        };
        let id = beams.len() as u32;
        adjacency[sb.a as usize].push(id);
        adjacency[sb.b as usize].push(id);
        beams.push(beam);
    }
    let edges: Vec<(u32, u32)> = desc.beams.iter().map(|b| (b.a, b.b)).collect();
    let (colors, color_count) = color_graph(n, &edges);
    PackedSoft {
        bodies,
        colliders,
        node_collider,
        beams,
        adjacency,
        colors,
        color_count,
        core,
        masses,
    }
}

fn core_plan(desc: &SoftVehicleDesc, masses: &mut [f32], opts: &SpawnOptions) -> CorePlan {
    let p = |i: u32| Vec3::from_array(desc.nodes[i as usize].position);
    let frame = frame_nodes(desc);
    // The core (engine, gearbox, axles) takes a share of every node's mass, so
    // its inertia follows the vehicle's mass distribution.
    let mut core_mass = 0.0f32;
    let mut centroid = Vec3::ZERO;
    let mut taken = vec![0.0f32; masses.len()];
    for (i, m) in masses.iter_mut().enumerate() {
        if desc.nodes[i].fixed {
            continue;
        }
        let t = *m * opts.core_mass_fraction;
        *m -= t;
        taken[i] = t;
        core_mass += t;
        centroid += p(i as u32) * t;
    }
    let origin = if core_mass > 0.0 {
        centroid / core_mass
    } else {
        Vec3::ZERO
    };
    let mut inertia = Vec3::splat(1.0);
    for (i, &t) in taken.iter().enumerate() {
        let r = p(i as u32) - origin;
        inertia += t * Vec3::new(
            r.y * r.y + r.z * r.z,
            r.x * r.x + r.z * r.z,
            r.x * r.x + r.y * r.y,
        );
    }
    let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for &i in &frame {
        let r = p(i) - origin;
        mn = mn.min(r);
        mx = mx.max(r);
    }
    let total_mass: f32 =
        masses.iter().sum::<f32>() + core_mass + desc.wheels.iter().map(|w| w.mass).sum::<f32>();
    let up = Vec3::Y;
    let forward = forward_axis(desc);
    let nw = desc.wheels.len().max(1) as f32;
    // Raycast suspension sized for the whole vehicle's weight per wheel.
    let m_w = total_mass / nw;
    let omega = std::f32::consts::TAU * opts.suspension_hz;
    let k = m_w * omega * omega;
    let c = 2.0 * opts.suspension_zeta * m_w * omega;
    let sag = 9.81 / (omega * omega);
    let travel = opts.suspension_travel.max(sag + 0.05);
    // Hardpoint above the axle so the loaded wheel centre sits on the axle nodes.
    let max_len = travel;
    let rest_len = max_len - sag;
    let any_steer = desc.wheels.iter().any(|w| w.steerable);
    let front = desc
        .wheels
        .iter()
        .map(|w| (0.5 * (p(w.node1) + p(w.node2))).dot(forward))
        .fold(f32::MIN, f32::max);
    let brake = 0.9 * 9.81 * total_mass / nw;
    let mut wheels = Vec::new();
    for w in &desc.wheels {
        let hub = 0.5 * (p(w.node1) + p(w.node2)) - origin;
        let is_front = (hub + origin).dot(forward) > front - 0.3;
        let steerable = if any_steer { w.steerable } else { is_front };
        wheels.push(WheelDesc {
            position: (hub + up * rest_len).to_array(),
            radius: w.radius,
            width: w.width.max(0.05),
            mass: w.mass,
            inertia: Some(0.5 * w.mass * w.radius * w.radius),
            suspension_min: 0.02,
            suspension_max: max_len,
            spring: Spring::StiffnessDamping { k, c },
            max_steer: if steerable {
                if w.max_steer > 0.0 {
                    w.max_steer
                } else {
                    35f32.to_radians()
                }
            } else {
                0.0
            },
            steerable,
            driven: w.propulsion != 0,
            max_brake_torque: if w.braked { brake * w.radius } else { 0.0 },
            max_handbrake_torque: if w.braked && !steerable {
                2.0 * brake * w.radius
            } else {
                0.0
            },
            handbrake: w.braked && !steerable,
            rolling_resistance: 0.012,
            ..Default::default()
        });
    }
    // One differential per driven axle (pairs left/right by lateral side).
    let right = forward.cross(up).normalize();
    let mut differentials = Vec::new();
    let mut driven: Vec<usize> = (0..wheels.len()).filter(|&i| wheels[i].driven).collect();
    driven.sort_by(|&a, &b| {
        let fa = Vec3::from_array(wheels[a].position).dot(forward);
        let fb = Vec3::from_array(wheels[b].position).dot(forward);
        fa.partial_cmp(&fb).unwrap()
    });
    for pair in driven.chunks(2) {
        let (l, r) = if pair.len() == 2 {
            let sa = Vec3::from_array(wheels[pair[0]].position).dot(right);
            if sa < 0.0 {
                (pair[0], pair[1])
            } else {
                (pair[1], pair[0])
            }
        } else {
            (pair[0], pair[0])
        };
        differentials.push(DifferentialDesc {
            left: Some(l as u32),
            right: if pair.len() == 2 {
                Some(r as u32)
            } else {
                None
            },
            ratio: desc.engine.as_ref().map_or(3.5, |e| e.diff_ratio),
            ..Default::default()
        });
    }
    let n_diff = differentials.len().max(1) as f32;
    for d in &mut differentials {
        d.engine_torque_ratio = 1.0 / n_diff;
    }
    let eng = desc.engine.clone().unwrap_or_default();
    let drive = if differentials.is_empty() {
        DriveDesc::Direct {
            max_torque: 0.0,
            top_speed: 1.0,
        }
    } else {
        DriveDesc::Drivetrain(DrivetrainDesc {
            engine: EngineDesc {
                max_torque: eng.max_torque,
                max_rpm: eng.max_rpm,
                ..Default::default()
            },
            transmission: TransmissionDesc {
                gear_ratios: eng.gear_ratios.clone(),
                reverse_ratios: vec![-eng.reverse_ratio.abs()],
                shift_up_rpm: eng.shift_up_rpm,
                shift_down_rpm: eng.shift_down_rpm,
                switch_time: 0.3,
                clutch_release_time: 0.2,
                ..Default::default()
            },
            differentials,
            ..Default::default()
        })
    };
    // A compact block between the axles (engine and drivetrain), well inside the bodywork.
    // Anti-roll bars on every left/right wheel pair of an axle (half the wheel rate).
    let mut anti_roll = Vec::new();
    for i in 0..wheels.len() {
        for j in i + 1..wheels.len() {
            let (a, b) = (
                Vec3::from_array(wheels[i].position),
                Vec3::from_array(wheels[j].position),
            );
            if (a.dot(forward) - b.dot(forward)).abs() < 0.3 && (a.dot(right) * b.dot(right)) < 0.0
            {
                let (l, r) = if a.dot(right) < 0.0 { (i, j) } else { (j, i) };
                anti_roll.push(AntiRollDesc {
                    left: l as u32,
                    right: r as u32,
                    stiffness: 0.5 * k,
                });
            }
        }
    }
    let half = Vec3::new((mx.x - mn.x) * 0.2, 0.1, (mx.z - mn.z) * 0.15).max(Vec3::splat(0.08));
    let center = (mn + mx) * 0.5;
    let vehicle = VehicleDesc {
        name: format!("{} core", desc.name),
        forward: forward.to_array(),
        up: up.to_array(),
        parts: vec![PartDesc {
            name: "core".into(),
            role: PartRole::Engine,
            mass: core_mass.max(1.0),
            colliders: vec![PartColliderDesc {
                shape: ShapeDesc::Box {
                    half: half.to_array(),
                },
                position: center.to_array(),
                rotation: [0.0, 0.0, 0.0, 1.0],
                material: 0,
            }],
            ..Default::default()
        }],
        inertia: Some(inertia.to_array()),
        auto_center_of_mass: false,
        center_of_mass_offset: [0.0; 3],
        wheels,
        drive,
        anti_roll,
        // Less lock at speed, as the rigid presets: full lock at 40 km/h would roll
        // a tall node–beam body.
        steering: SteeringDesc {
            ackermann: 1.0,
            speed_curve: Curve::road_steering(),
            ..Default::default()
        },
        assists: AssistDesc {
            traction_control: true,
            ..Default::default()
        },
        can_sleep: true,
        ..Default::default()
    };
    CorePlan {
        vehicle,
        origin,
        frame_nodes: frame,
    }
}
