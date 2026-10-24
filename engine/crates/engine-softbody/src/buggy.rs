//! vibe-land's dune buggy as a hybrid soft/rigid vehicle: its tube frame bends
//! (node–beam), its bolted-on parts snap off (rigid bodies on breakable ball
//! joints), and the drivetrain parts ride the rigid core with their mounts.
//!
//! Built from the same baked asset as the rigid preset
//! (`engine-vehicle/assets/dune_buggy.json`, `hybrid` section; see
//! `engine/scenes/dune-buggy`): the frame graph's nodes and beams, the nodes
//! each bolted part attaches to, and the core's attachment nodes (the lower
//! wishbone pivots). The core is `engine_vehicle::presets::dune_buggy` with only
//! its drivetrain parts (engine, gearbox, steering column) and a root carrying
//! the moving suspension's mass, so its wheels, springs and drive are the rigid
//! buggy's.

use crate::desc::*;
use engine_vehicle::{presets, MountDesc, PartColliderDesc, PartDesc, PartRole, ShapeDesc};
use serde::Deserialize;

/// Beam properties per structural class (calibrated against the crash tests in
/// `engine-webgpu/tests/softbody.rs`, 240 Hz × 10 iterations).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuggyBeams {
    pub spring: f32,
    pub damp: f32,
    /// Yield and break thresholds (N) of the safety cell.
    pub cell_deform: f32,
    pub cell_strength: f32,
    /// Yield and break thresholds (N) of the crumple zones (horns, bumpers, nose).
    pub crumple_deform: f32,
    pub crumple_strength: f32,
    /// Bolted parts' joint limits and detachable pieces' mount beams, as a
    /// multiple of their rated mounts (the rigid preset's).
    pub mount_scale: f32,
}

impl Default for BuggyBeams {
    fn default() -> Self {
        Self {
            // The frame's nodes are light (median 1.3 kg): stiffer beams ring at
            // 240 Hz and spike loads (3e6 N/m tore the rear bumper off in a
            // frontal crash, hybrid_buggy_crash_probe).
            spring: 1.0e6,
            damp: 2000.0,
            cell_deform: 30_000.0,
            cell_strength: 400_000.0,
            crumple_deform: 6_000.0,
            crumple_strength: 100_000.0,
            mount_scale: 1.0,
        }
    }
}

#[derive(Deserialize)]
struct Node {
    position: [f32; 3],
    mass: f32,
    group: String,
}

#[derive(Deserialize)]
struct Beam {
    a: u32,
    b: u32,
    kind: String,
    cls: Option<String>,
    group: Option<String>,
    strength: Option<f32>,
}

#[derive(Deserialize)]
struct Rigid {
    part: usize,
    nodes: Vec<u32>,
    break_force: f32,
}

#[derive(Deserialize)]
struct Hybrid {
    nodes: Vec<Node>,
    beams: Vec<Beam>,
    core_nodes: Vec<u32>,
    core_parts: Vec<String>,
    core_root_mass: f32,
    soft_groups: Vec<String>,
    rigid_parts: Vec<Rigid>,
}

#[derive(Deserialize)]
struct Asset {
    hybrid: Hybrid,
}

/// The hybrid dune buggy with the default beam calibration.
pub fn dune_buggy_soft() -> SoftVehicleDesc {
    dune_buggy_soft_with(&BuggyBeams::default())
}

pub fn dune_buggy_soft_with(k: &BuggyBeams) -> SoftVehicleDesc {
    let asset: Asset =
        serde_json::from_str(include_str!("../../engine-vehicle/assets/dune_buggy.json"))
            .expect("dune_buggy.json");
    let h = asset.hybrid;
    let rigid = presets::dune_buggy();
    let group_id = |g: &str| {
        h.soft_groups
            .iter()
            .position(|x| x == g)
            .map_or(0, |i| i as u32 + 1)
    };

    let nodes = h
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| SoftNode {
            // "<group>:<index>": frame, front_bumper, rear_bumper or nose.
            name: format!("{}:{i}", n.group),
            position: n.position,
            mass: n.mass.max(0.5),
            collision: true,
            contacter: true,
            ..Default::default()
        })
        .collect();

    let beams = h
        .beams
        .iter()
        .map(|b| {
            let crumple = b.cls.as_deref() == Some("crumple") || b.group.is_some();
            let (deform, strength) = if crumple {
                (k.crumple_deform, k.crumple_strength)
            } else {
                (k.cell_deform, k.cell_strength)
            };
            match b.kind.as_str() {
                // A detachable piece's mounts: elastic up to the rated mount
                // (shared among its beams), then the whole group lets go.
                "mount" => SoftBeam {
                    a: b.a,
                    b: b.b,
                    spring: k.spring,
                    damp: k.damp,
                    deform: f32::INFINITY,
                    strength: b.strength.unwrap_or(f32::INFINITY) * k.mount_scale,
                    break_group: b.group.as_deref().map_or(0, group_id),
                    visible: false,
                    ..Default::default()
                },
                kind => SoftBeam {
                    a: b.a,
                    b: b.b,
                    spring: k.spring,
                    damp: k.damp,
                    deform,
                    strength,
                    visible: kind == "tube",
                    ..Default::default()
                },
            }
        })
        .collect();

    // The core: the rigid buggy's wheels and drive, its drivetrain parts, and a
    // root carrying the moving suspension (a small box low in the floor).
    // Its centre of mass is the whole car's (core and frame, SoftCore), with the
    // rigid preset's handling offset (vibe-land's 0.2 m below the computed centre).
    let mut core = rigid.clone();
    core.name = "dune_buggy_core".into();
    let mut parts = vec![PartDesc {
        name: "core".into(),
        role: PartRole::Body,
        mass: h.core_root_mass,
        colliders: vec![PartColliderDesc {
            shape: ShapeDesc::Box {
                half: [0.12, 0.04, 0.5],
            },
            position: [0.0, 0.5, 0.0],
            ..Default::default()
        }],
        mount: MountDesc::default(),
    }];
    parts.extend(
        rigid
            .parts
            .iter()
            .filter(|p| h.core_parts.contains(&p.name))
            .cloned(),
    );
    core.parts = parts;

    let rigid_parts = h
        .rigid_parts
        .iter()
        .map(|r| {
            let p = &rigid.parts[r.part];
            SoftRigidPart {
                name: p.name.clone(),
                mass: p.mass,
                colliders: p.colliders.clone(),
                attach: r
                    .nodes
                    .iter()
                    .map(|&node| SoftAttach {
                        node,
                        break_force: r.break_force * k.mount_scale,
                        break_torque: 0.0,
                    })
                    .collect(),
            }
        })
        .collect();

    // Named node sets (SoftVehicleDesc::parts): the frame, then the pieces that
    // tear off whole, so PieceDetached events and damage readouts name them.
    let parts = ["frame", "front_bumper", "rear_bumper", "nose"]
        .iter()
        .map(|&g| SoftPart {
            name: g.into(),
            nodes: h
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.group == g)
                .map(|(i, _)| i as u32)
                .collect(),
        })
        .collect();

    SoftVehicleDesc {
        name: "dune_buggy_hybrid".into(),
        parts,
        nodes,
        beams,
        core: Some(SoftCore {
            vehicle: core,
            nodes: h.core_nodes,
        }),
        rigid_parts,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_buggy_is_whole() {
        let d = dune_buggy_soft();
        let core = d.core.as_ref().unwrap();
        assert_eq!(
            core.vehicle.parts.len(),
            4,
            "root + engine, gearbox, steering column"
        );
        assert_eq!(core.nodes.len(), 8, "the lower wishbone pivots");
        assert_eq!(d.rigid_parts.len(), 10);
        assert!(d
            .rigid_parts
            .iter()
            .all(|p| p.attach.len() == 3 && p.mass > 0.0 && !p.colliders.is_empty()));
        // Every beam references real nodes; the graph is one connected piece.
        let n = d.nodes.len();
        assert!(d
            .beams
            .iter()
            .all(|b| (b.a as usize) < n && (b.b as usize) < n && b.a != b.b));
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(p: &mut Vec<usize>, mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for b in &d.beams {
            let (x, y) = (
                root(&mut parent, b.a as usize),
                root(&mut parent, b.b as usize),
            );
            parent[x] = y;
        }
        let r0 = root(&mut parent, 0);
        assert!(
            (0..n).all(|i| root(&mut parent, i) == r0),
            "frame graph is connected"
        );
        // The three detachable pieces hang on breakable mount beams.
        for g in 1..=3 {
            let mounts: Vec<&SoftBeam> = d.beams.iter().filter(|b| b.break_group == g).collect();
            assert!(
                !mounts.is_empty()
                    && mounts
                        .iter()
                        .all(|b| b.strength.is_finite() && b.strength > 0.0),
                "group {g}"
            );
        }
        // Named parts cover every node exactly once.
        let names: Vec<&str> = d.parts.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["frame", "front_bumper", "rear_bumper", "nose"]);
        assert!(d.parts.iter().all(|p| !p.nodes.is_empty()));
        assert_eq!(d.parts.iter().map(|p| p.nodes.len()).sum::<usize>(), n);
        // The whole buggy weighs what the rigid one does.
        let rigid = presets::dune_buggy();
        let m_rigid: f32 = rigid.parts.iter().map(|p| p.mass).sum::<f32>()
            + rigid.wheels.iter().map(|w| w.mass).sum::<f32>();
        let m = d.total_mass()
            + d.rigid_parts.iter().map(|p| p.mass).sum::<f32>()
            + core.vehicle.parts.iter().map(|p| p.mass).sum::<f32>()
            + core.vehicle.wheels.iter().map(|w| w.mass).sum::<f32>();
        assert!(
            (m - m_rigid).abs() < 0.03 * m_rigid,
            "hybrid {m} kg vs rigid {m_rigid} kg"
        );
    }
}
