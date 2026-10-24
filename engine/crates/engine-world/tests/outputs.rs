//! Per-frame outputs beyond body moves: vehicles (wheels), soft-body nodes and
//! skins, resync after dropped frames, coalesced polling, shape meshes.

use engine_softbody::procedural::{car_truck, CarParams};
use engine_softbody::truck::load_truck;
use engine_vehicle::presets;
use rubble::*;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 256,
        max_hull_vertices: 256,
    }
}

fn ground(w: &mut World) -> BodyId {
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
    g
}

fn soft_world() -> World {
    let def = WorldDef {
        capacity: capacity(),
        contacts: ContactStorage::Canonical,
        soft: Some(SoftCapacity::default()),
        solver: engine_softbody::soft_scene_params(120.0, 10),
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -9.81, 0.0]).unwrap();
    ground(&mut w);
    w
}

fn rigid_world() -> World {
    let def = WorldDef {
        capacity: capacity(),
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    ground(&mut w);
    w
}

fn run(w: &mut World, steps: u32) -> Vec<Frame> {
    for _ in 0..steps {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap()
}

#[test]
fn soft_nodes_are_not_moves_but_their_own_output() {
    let mut w = soft_world();
    let car = load_truck(&car_truck(&CarParams::default()));
    let nodes = car.nodes.len();
    let h = w
        .add_soft_vehicle(
            &car,
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions::default(),
        )
        .unwrap();
    w.soft_enable_output(h, true).unwrap();
    let frames = run(&mut w, 20);
    let moved_most = frames
        .iter()
        .map(|f| f.body_events.moves.len())
        .max()
        .unwrap();
    assert!(
        moved_most < nodes / 4,
        "{moved_most} moves per step for a {nodes}-node car: soft nodes leak into moves"
    );
    let last = frames.last().unwrap();
    let out = last.soft.iter().find(|s| s.soft == h).expect("soft output");
    assert_eq!(out.nodes.len(), nodes);
    assert!(out.nodes.iter().all(|p| p.iter().all(|v| v.is_finite())));
    assert_eq!(
        last.vehicles.len(),
        1,
        "the soft car's rigid core reports as a vehicle"
    );
    // Driving goes through one call.
    w.set_soft_vehicle_input(
        h,
        VehicleInput {
            forward: 1.0,
            ..VehicleInput::default()
        },
    )
    .unwrap();
    run(&mut w, 5);
}

#[test]
fn skins_deform_with_the_soft_body_and_arrive_each_frame() {
    let mut w = soft_world();
    let car = load_truck(&car_truck(&CarParams::default()));
    let h = w
        .add_soft_vehicle(
            &car,
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions::default(),
        )
        .unwrap();
    // A quad hovering over the car's first nodes, in the spawn frame.
    let p = |i: usize| {
        let n = car.nodes[i].position;
        [n[0], n[1] + 1.0, n[2]]
    };
    let vertices = [p(0), p(1), p(2), p(3)];
    let skin = w
        .soft_bind_skin(h, &vertices, &[[0, 1, 2], [0, 2, 3]], None)
        .unwrap();
    let frames = run(&mut w, 30);
    let out = frames
        .last()
        .unwrap()
        .skins
        .iter()
        .find(|s| s.skin == skin)
        .expect("skin output");
    assert_eq!(out.positions.len(), 4);
    assert_eq!(out.normals.len(), 4);
    assert!(out
        .positions
        .iter()
        .all(|v| v.iter().all(|x| x.is_finite())));
}

#[test]
fn every_frame_carries_each_vehicles_wheels() {
    let mut w = rigid_world();
    let v = w
        .add_vehicle(&presets::sedan(), [0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    w.set_vehicle_input(
        v,
        VehicleInput {
            forward: 1.0,
            ..VehicleInput::default()
        },
    );
    let frames = run(&mut w, 120);
    let out = &frames.last().unwrap().vehicles;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].vehicle, v);
    let wheels = &out[0].state.wheels;
    assert_eq!(wheels.len(), presets::sedan().wheels.len());
    assert!(
        wheels.iter().any(|wh| wh.omega.abs() > 0.1),
        "driven wheels spin"
    );
    // The frame's copy decodes exactly what the blocking read returns.
    let blocking = w.vehicle_state(v);
    assert_eq!(blocking.wheels.len(), wheels.len());
    assert_eq!(blocking, out[0].state);
}

/// A body that settles inside dropped frames is still reported: the next
/// delivered frame diffs every body against what was last delivered.
#[test]
fn bodies_settling_inside_dropped_frames_are_reported_later() {
    let def = WorldDef {
        capacity: capacity(),
        readback_frames: 1,
        backpressure: Backpressure::Drop,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    ground(&mut w);
    let b = w.create_body(&BodyDef::dynamic([0.0, 0.5, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    // Step without polling: the single slot stays busy, later frames drop,
    // and the box falls asleep somewhere in between.
    for _ in 0..200 {
        w.step().unwrap();
    }
    let dropped = w.poll_blocking().unwrap();
    assert!(dropped.iter().any(|f| f.dropped));
    // The next delivered frame resyncs.
    let f = w.step_sync().unwrap();
    assert!(f.resynced, "the frame after drops is a resync");
    let truth = w.read_body_state(b).unwrap();
    let reported = w.body_transform(b).unwrap().unwrap();
    assert_eq!(reported.position, truth.transform.position, "no stale pose");
    assert_eq!(
        w.body_is_awake(b).unwrap(),
        truth.awake,
        "sleep state caught up"
    );
    assert!(!truth.awake, "the box settled while frames were dropped");
}

#[test]
fn coalesced_polling_folds_steps() {
    let mut w = rigid_world();
    let b = w.create_body(&BodyDef::dynamic([0.0, 10.0, 0.0])).unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.5)
        .unwrap();
    w.step_sync().unwrap();
    let before = w.body_transform(b).unwrap().unwrap();
    for _ in 0..4 {
        w.step().unwrap();
    }
    w.poll_blocking().map(|f| assert!(f.len() <= 4)).unwrap();
    // poll_blocking drained them; step four more and fold them.
    for _ in 0..4 {
        w.step().unwrap();
    }
    let _ = w.sim().download_bodies(); // wait for the GPU
    let mut c = w.poll_coalesced().unwrap();
    while c.steps.map_or(0, |(a, z)| z - a + 1) < 4 {
        let more = w.poll_coalesced().unwrap();
        if more.steps.is_none() {
            continue;
        }
        c = more;
    }
    let m = c.moves.iter().find(|m| m.body == b).unwrap();
    assert!(m.transform.position[1] < before.position[1]);
    assert!(m.previous.is_some());
}

#[test]
fn box_and_hull_shapes_give_meshes() {
    let mut w = rigid_world();
    let b = w.create_body(&BodyDef::dynamic([0.0, 3.0, 0.0])).unwrap();
    let s = w
        .create_box_shape(b, &ShapeDef::default(), [0.5, 1.0, 1.5])
        .unwrap();
    let m = w.shape_mesh(s).unwrap().unwrap();
    assert_eq!(m.vertices.len(), 8);
    assert_eq!(m.triangles.len(), 12);
    let ball = w
        .create_sphere_shape(b, &ShapeDef::default(), [0.0, 2.0, 0.0], 0.3)
        .unwrap();
    assert!(w.shape_mesh(ball).unwrap().is_none());
    assert!(matches!(
        w.shape_geometry(ball).unwrap(),
        Geometry::Sphere { .. }
    ));
}
