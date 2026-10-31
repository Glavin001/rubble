//! Assemblies and composed vehicles: parts of different physical models
//! connected by joints and breakable mounts, placed with one call.

use engine_softbody::procedural::{car_truck, CarParams};
use engine_softbody::truck::load_truck;
use engine_vehicle::presets;
use glam::{Quat, Vec3};
use rubble::*;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 8192,
        max_colliders: 16_384,
        max_contacts: 131_072,
        max_joints: 4096,
        max_candidates: 262_144,
        max_colors: 32,
        max_commands: 1024,
        max_hull_vertices: 4096,
    }
}

fn world(soft: bool) -> World {
    let mut def = WorldDef {
        capacity: capacity(),
        ..WorldDef::default()
    };
    if soft {
        def.contacts = ContactStorage::Canonical;
        def.soft = Some(SoftCapacity::default());
        def.solver = engine_softbody::soft_scene_params(120.0, 10);
    }
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -9.81, 0.0]).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -5.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    let rough = ShapeDef {
        material: SurfaceMaterial {
            friction: 1.0,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    w.create_box_shape(g, &rough, [400.0, 5.0, 400.0]).unwrap();
    w
}

fn boxed(half: [f32; 3], density: f32) -> Vec<(ShapeDef, Geometry)> {
    let s = ShapeDef {
        density,
        ..ShapeDef::default()
    };
    vec![(
        s,
        Geometry::Box {
            half,
            transform: Transform::IDENTITY,
        },
    )]
}

fn part(name: &str, body: BodyDef, shapes: Vec<(ShapeDef, Geometry)>) -> PartDef {
    PartDef {
        name: name.into(),
        model: PartModel::Rigid { body, shapes },
    }
}

fn at(p: [f32; 3]) -> Transform {
    Transform::new(p, [0.0, 0.0, 0.0, 1.0])
}

fn run(w: &mut World, steps: u32) -> Vec<Frame> {
    let mut frames = Vec::new();
    for _ in 0..steps {
        w.step().unwrap();
        frames.extend(w.poll().unwrap());
    }
    frames.extend(w.poll_blocking().unwrap());
    frames
}

fn body(w: &World, a: AssemblyId, name: &str) -> BodyId {
    match w.assembly_part(a, name).unwrap() {
        PartHandle::Body(b) => b,
        PartHandle::Destructible(_) => panic!("{name} is destructible"),
    }
}

fn pos(w: &World, b: BodyId) -> Vec3 {
    Vec3::from(w.read_body_state(b).unwrap().transform.position)
}

/// A sign hung from a beam by a breakable mount; a weight pulls it off.
#[test]
fn a_breakable_mount_reports_its_name() {
    let mut w = world(false);
    let def = AssemblyDef {
        transform: at([0.0, 6.0, 0.0]),
        parts: vec![
            part("beam", BodyDef::default(), boxed([2.0, 0.1, 0.1], 1.0)),
            part(
                "sign",
                BodyDef::dynamic([0.0, -1.0, 0.0]),
                boxed([0.5, 0.5, 0.05], 2000.0),
            ),
        ],
        connections: vec![ConnectionDef {
            name: "bolt".into(),
            a: PartRef::Part("beam".into()),
            b: PartRef::Part("sign".into()),
            frame: at([0.0, -0.4, 0.0]),
            kind: ConnectionKind::Mount {
                break_force: 500.0,
                break_torque: f32::INFINITY,
            },
        }],
        user_data: 9,
    };
    let a = w.create_assembly(&def).unwrap();
    let sign = body(&w, a, "sign");
    let frames = run(&mut w, 60);
    let broke = frames
        .iter()
        .flat_map(|f| f.destruction_events.iter())
        .find_map(|e| match e {
            DestructionEvent::ConnectionBroke {
                assembly,
                connection,
                force,
                ..
            } => Some((*assembly, connection.clone(), *force)),
            _ => None,
        });
    let (assembly, name, force) = broke.expect("the 100 kg sign tears the 500 N mount");
    assert_eq!(assembly, a);
    assert_eq!(name, "bolt");
    assert!(force > 500.0);
    assert!(pos(&w, sign).y < 4.5, "the sign fell");
}

/// A door on a hinge given once in assembly space swings about the hinge.
#[test]
fn a_hinge_connection_swings_about_its_axis() {
    let mut w = world(false);
    let hinge = RevoluteJointDef {
        base: JointBase::new(BodyId::NULL, BodyId::NULL),
        target_angle: 0.0,
        spring: None,
        limit: None,
        motor: None,
    };
    // Hinge axis (joint z) along world Y: a vertical door hinge.
    let up = Quat::from_rotation_arc(Vec3::Z, Vec3::Y).to_array();
    let def = AssemblyDef {
        transform: at([3.0, 1.2, 0.0]),
        parts: vec![
            part("post", BodyDef::default(), boxed([0.05, 1.0, 0.05], 1.0)),
            part(
                "door",
                BodyDef::dynamic([0.5, 0.0, 0.0]),
                boxed([0.45, 0.9, 0.03], 500.0),
            ),
        ],
        connections: vec![ConnectionDef {
            name: "hinge".into(),
            a: PartRef::Part("post".into()),
            b: PartRef::Part("door".into()),
            frame: Transform::new([0.05, 0.0, 0.0], up),
            kind: ConnectionKind::Joint(JointDef::Revolute(hinge)),
        }],
        user_data: 0,
    };
    let a = w.create_assembly(&def).unwrap();
    let door = body(&w, a, "door");
    w.body_apply_linear_impulse(door, [0.0, 0.0, 50.0], [3.9, 1.2, 0.0], true)
        .unwrap();
    run(&mut w, 30);
    let p = pos(&w, door);
    assert!(
        (p.y - 1.2).abs() < 0.05,
        "a hinged door does not sag: {p:?}"
    );
    assert!(p.z > 0.05, "it swung about the vertical hinge: {p:?}");
    let r = (p - Vec3::new(3.05, 1.2, 0.0)).length();
    assert!((r - 0.45).abs() < 0.08, "it stays at the hinge radius: {r}");
}

/// A sedan with a roof rack mounted on its chassis and a breakable crate
/// (destructible) bonded on: one call, and the parts ride along.
#[test]
fn a_rigid_vehicle_carries_mounted_and_destructible_parts() {
    let mut w = world(false);
    let crate_asset = {
        // Two boxes bonded together: a minimal destructible.
        let chunk = |x: f32| ChunkDesc {
            pose: at([x, 0.0, 0.0]),
            mass: 5.0,
            inertia: [0.05; 3],
            shapes: vec![Geometry::Box {
                half: [0.15; 3],
                transform: Transform::IDENTITY,
            }],
            fragment: None,
            node: None,
            node_frame: None,
        };
        let desc = DestructibleAssetDesc {
            name: Some("crate".into()),
            chunks: vec![chunk(-0.15), chunk(0.15)],
            bonds: vec![BondDesc {
                a: 0,
                b: Some(1),
                anchor_a: [0.15, 0.0, 0.0],
                anchor_b: [-0.15, 0.0, 0.0],
                rest: [0.0, 0.0, 0.0, 1.0],
                normal: [1.0, 0.0, 0.0],
                frame_u: [0.0, 1.0, 0.0],
                strength: [1e5, 1e6, 1e5, 1e4],
                stiffness: [1e7; 6],
            }],
            preload: None,
            solver: None,
            stress: None,
        };
        w.create_destructible_asset(&desc).unwrap()
    };
    let def = VehicleDef {
        model: VehicleModel::Rigid(presets::sedan()),
        transform: at([0.0, 1.5, 0.0]),
        parts: vec![
            part(
                "rack",
                BodyDef::dynamic([0.0, 1.0, 0.0]),
                boxed([0.6, 0.05, 1.0], 200.0),
            ),
            PartDef {
                name: "crate".into(),
                model: PartModel::Destructible(DestructibleDef {
                    transform: at([0.0, 1.25, 0.0]),
                    anchors: Anchors::None,
                    asleep: false,
                    ..DestructibleDef::new(crate_asset)
                }),
            },
        ],
        connections: vec![
            ConnectionDef {
                name: "rack mount".into(),
                a: PartRef::Chassis,
                b: PartRef::Part("rack".into()),
                frame: at([0.0, 0.95, 0.0]),
                kind: ConnectionKind::Mount {
                    break_force: f32::INFINITY,
                    break_torque: f32::INFINITY,
                },
            },
            ConnectionDef {
                name: "strap".into(),
                a: PartRef::Part("rack".into()),
                b: PartRef::Chunk("crate".into(), 0),
                frame: at([-0.15, 1.1, 0.0]),
                kind: ConnectionKind::Mount {
                    break_force: f32::INFINITY,
                    break_torque: f32::INFINITY,
                },
            },
        ],
        user_data: 0,
    };
    let built = w.create_vehicle(&def).unwrap();
    let rack = body(&w, built.assembly, "rack");
    let gap0 = pos(&w, rack) - pos(&w, built.chassis);
    w.set_vehicle_input(
        built.vehicle.unwrap(),
        VehicleInput {
            forward: 1.0,
            ..VehicleInput::default()
        },
    );
    let frames = run(&mut w, 120);
    let gap = pos(&w, rack) - pos(&w, built.chassis);
    let moved = pos(&w, built.chassis);
    assert!(moved.length() > 1.0, "the car drove: {moved:?}");
    assert!(
        (gap.length() - gap0.length()).abs() < 0.05,
        "the rack rides on the chassis"
    );
    assert!(frames
        .iter()
        .flat_map(|f| f.destruction_events.iter())
        .all(|e| !matches!(e, DestructionEvent::ConnectionBroke { .. })));
    match w.assembly_part(built.assembly, "crate").unwrap() {
        PartHandle::Destructible(d) => assert_eq!(w.destructible_stats(d).unwrap().actors, 1),
        PartHandle::Body(_) => panic!("crate is destructible"),
    }
}

/// A soft node-beam car with a rigid part bolted onto one of its nodes (the
/// hybrid dune-buggy pattern), composed in one call.
#[test]
fn a_soft_vehicle_carries_a_part_bolted_to_a_node() {
    let mut w = world(true);
    let car = load_truck(&car_truck(&CarParams::default()));
    let node = car.nodes[0].position;
    let def = VehicleDef {
        model: VehicleModel::Soft(car, SpawnOptions::default()),
        transform: at([0.0, 1.0, 0.0]),
        parts: vec![part(
            "light",
            BodyDef::dynamic([node[0], node[1] + 0.15, node[2]]),
            boxed([0.1, 0.1, 0.1], 300.0),
        )],
        connections: vec![ConnectionDef {
            name: "bracket".into(),
            a: PartRef::SoftNode(0),
            b: PartRef::Part("light".into()),
            frame: at([node[0], node[1] + 0.05, node[2]]),
            kind: ConnectionKind::Mount {
                break_force: f32::INFINITY,
                break_torque: f32::INFINITY,
            },
        }],
        user_data: 0,
    };
    let built = w.create_vehicle(&def).unwrap();
    assert!(built.soft.is_some());
    let light = body(&w, built.assembly, "light");
    run(&mut w, 60);
    let p = pos(&w, light);
    assert!(p.is_finite());
    assert!(p.y > 0.0 && p.y < 3.0, "the light stays on the car: {p:?}");
}
