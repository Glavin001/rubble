//! Anti-clip (BeamNG-parity plan, phases 1 and 2): nodes stay on the side of the
//! triangles they came from, through crashes. Penetrations are counted with
//! `engine_softbody::metrics::PenetrationTracker` (a node behind a triangle it
//! was in front of, inside its prism).

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::metrics::{PenetrationTracker, TrackedBody};
use engine_softbody::procedural::{car_desc, CarParams};
use engine_softbody::{
    soft_scene_params, SoftBackend, SoftCapacity, SoftHandle, SoftSettings, SoftVehicleDesc,
    SpawnOptions,
};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::Vec3;

fn world(hz: f32, iterations: u32, settings: SoftSettings) -> WgpuSim {
    let cap = WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    let mut p = soft_scene_params(hz, iterations);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    sim.set_soft_settings(settings);
    for (i, mu) in [(0, 1.0), (9, 0.8)] {
        sim.set_material(
            i,
            Material {
                friction: mu,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
    }
    static_box(
        &mut sim,
        Vec3::new(0.0, -5.0, 0.0),
        Vec3::new(400.0, 5.0, 400.0),
    );
    sim
}

fn static_box(sim: &mut WgpuSim, center: Vec3, half: Vec3) {
    let b = sim.add_body(BodyState::static_plane_support());
    sim.add_collider(Collider {
        body: b,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: center.to_array(),
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half.x, half.y, half.z, half.length()],
    });
}

fn spawn(sim: &mut WgpuSim, desc: &SoftVehicleDesc, at: Vec3, v: Vec3) -> SoftHandle {
    sim.add_soft_vehicle(
        desc,
        at.to_array(),
        [0.0, 0.0, 0.0, 1.0],
        &SpawnOptions {
            rigid_core: false,
            velocity: v.to_array(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Steps `frames` × `per` steps, tracking penetrations; returns the tracker.
fn track(
    sim: &mut WgpuSim,
    cars: &[(SoftHandle, &SoftVehicleDesc)],
    frames: u32,
    per: u32,
    mut each: impl FnMut(&WgpuSim, &PenetrationTracker, u32),
) -> PenetrationTracker {
    let mut first = 0;
    let bodies = cars
        .iter()
        .map(|(_, d)| {
            let t = TrackedBody::from_desc(d, first);
            first += d.nodes.len() as u32;
            t
        })
        .collect();
    let mut tr = PenetrationTracker::new(bodies);
    for f in 0..frames {
        for _ in 0..per {
            sim.step().unwrap();
        }
        let x: Vec<Vec3> = cars
            .iter()
            .flat_map(|(h, _)| {
                sim.read_soft_particles(*h)
                    .into_iter()
                    .map(|b| Vec3::from_array(b.position))
            })
            .collect();
        let broken: Vec<Vec<bool>> = cars
            .iter()
            .map(|(h, _)| sim.read_soft_beams(*h).iter().map(|b| b.broken()).collect())
            .collect();
        tr.update(&x, Some(&broken));
        each(sim, &tr, f);
    }
    tr
}

/// What the GPU held for each pair the tracker first sees penetrating: the node's
/// contact triangle this step and its side cache.
#[test]
#[ignore]
fn wall_penetration_probe() {
    let (desc, _) = car_desc(&CarParams::sedan());
    let push = std::env::var("PUSH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let mut sim = world(
        240.0,
        10,
        SoftSettings {
            contact_push_out: push,
            displacement_bound: std::env::var("GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SoftSettings::default().displacement_bound),
            contact_stiffness: std::env::var("KC")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SoftSettings::default().contact_stiffness),
            ..Default::default()
        },
    );
    static_box(&mut sim, Vec3::new(0.0, 1.5, 3.0), Vec3::new(3.0, 1.5, 0.2));
    let at = Vec3::new(0.0, 0.06 - desc.bounds().0[1], -1.0);
    let h = spawn(&mut sim, &desc, at, Vec3::new(0.0, 0.0, 56.0 / 3.6));
    let l = SoftCapacity::default().layout();
    let ord = sim.soft_host().entry(h).ordinal_base;
    let mut seen = std::collections::HashSet::new();
    let tr = track(&mut sim, &[(h, &desc)], 360, 1, |sim, tr, f| {
        for &(_, t, n, depth) in &tr.last {
            if !seen.insert((t, n)) {
                continue;
            }
            let o = ord + n;
            let slots = sim.read_soft_slots(
                l.node_slot_base + engine_softbody::NODE_SLOT_STRIDE * o,
                engine_softbody::NODE_SLOT_STRIDE,
            );
            let contacts: Vec<(i64, f32)> = (0..engine_softbody::NODE_CONTACTS as usize)
                .filter(|&k| slots[3 * k][3] != engine_softbody::SOFT_NONE)
                .map(|k| (slots[3 * k][3] as i64, f32::from_bits(slots[3 * k + 2][0])))
                .collect();
            let cache = sim.read_soft_slots(l.pair_cache_base + o, 1)[0];
            let cached: Vec<(u32, i32)> = cache
                .iter()
                .filter(|&&e| e != 0)
                .map(|&e| ((e >> 1) - 1, if e & 1 == 1 { -1 } else { 1 }))
                .collect();
            let tri = desc.triangles[t as usize].nodes;
            eprintln!(
                "frame {f}: node {n} behind tri {t} {tri:?} by {:.1} cm; contacts (tri, gap0) {contacts:?}, cache {cached:?}",
                depth * 100.0,
            );
        }
    });
    eprintln!("max {:?}", tr.max);
}

/// Settings from the environment: `GAMMA` (displacement bound), `PUSH`, `KC`.
fn env_settings() -> SoftSettings {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let d = SoftSettings::default();
    SoftSettings {
        displacement_bound: env("GAMMA").unwrap_or(d.displacement_bound),
        contact_push_out: env("PUSH").unwrap_or(d.contact_push_out),
        contact_stiffness: env("KC").unwrap_or(d.contact_stiffness),
        ..d
    }
}

/// Penetrations (max over the run) in the phase 2 scenes: a 100 km/h wall, a
/// 2 × 80 km/h head-on, and a 60 km/h slide into a 25 cm side pole.
fn phase2_scene(scene: &str, hz: f32, iters: u32) -> engine_softbody::metrics::PenetrationCount {
    let (desc, _) = car_desc(&CarParams::sedan());
    let lift = 0.06 - desc.bounds().0[1];
    let mut sim = world(hz, iters, env_settings());
    let per = (hz / 240.0).round().max(1.0) as u32;
    let cars: Vec<SoftHandle> = match scene {
        "wall100" => {
            static_box(&mut sim, Vec3::new(0.0, 1.5, 3.0), Vec3::new(3.0, 1.5, 0.2));
            vec![spawn(
                &mut sim,
                &desc,
                Vec3::new(0.0, lift, -1.0),
                Vec3::new(0.0, 0.0, 100.0 / 3.6),
            )]
        }
        "headon" => {
            let v = 80.0 / 3.6;
            let a = spawn(
                &mut sim,
                &desc,
                Vec3::new(0.0, lift, -2.6),
                Vec3::new(0.0, 0.0, v),
            );
            let b = sim
                .add_soft_vehicle(
                    &desc,
                    [0.0, lift, 2.6],
                    [0.0, 1.0, 0.0, 0.0],
                    &SpawnOptions {
                        rigid_core: false,
                        velocity: [0.0, 0.0, -v],
                        ..Default::default()
                    },
                )
                .unwrap();
            vec![a, b]
        }
        "sidepole" => {
            // A 25 cm pole (registered as a soft obstacle) 1.6 m to the car's left.
            let pole = sim.add_body(BodyState::static_plane_support());
            let c = sim.add_collider(Collider {
                body: pole,
                shape: ShapeType::Capsule as u32,
                material: 0,
                generation: 0,
                local_pos: [1.6, 2.0, 0.3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [0.125, 2.0, 0.0, 2.2],
            });
            sim.add_soft_obstacle(c).unwrap();
            vec![spawn(
                &mut sim,
                &desc,
                Vec3::new(0.0, lift, 0.0),
                Vec3::new(60.0 / 3.6, 0.0, 0.0),
            )]
        }
        _ => panic!("scene {scene}"),
    };
    let pairs: Vec<(SoftHandle, &SoftVehicleDesc)> = cars.iter().map(|&h| (h, &desc)).collect();
    track(&mut sim, &pairs, 300, per, |_, _, _| {}).max
}

#[test]
#[ignore]
fn phase2_scenes_probe() {
    let hz = std::env::var("HZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(240.0);
    let it = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    for scene in ["wall100", "headon", "sidepole"] {
        eprintln!("{scene}: {:?}", phase2_scene(scene, hz, it));
    }
}

/// The sedan into a wall at 56 km/h, 240 Hz × 10, no rigid core: before the side
/// cache and multiple contacts per node, 44 nodes ended up behind a triangle they
/// were in front of; now about 5 (the explicit oracle: 2).
#[test]
fn the_56_wall_keeps_nodes_on_their_side() {
    let (desc, _) = car_desc(&CarParams::sedan());
    let mut sim = world(240.0, 10, SoftSettings::default());
    static_box(&mut sim, Vec3::new(0.0, 1.5, 3.0), Vec3::new(3.0, 1.5, 0.2));
    let at = Vec3::new(0.0, 0.06 - desc.bounds().0[1], -1.0);
    let h = spawn(&mut sim, &desc, at, Vec3::new(0.0, 0.0, 56.0 / 3.6));
    let tr = track(&mut sim, &[(h, &desc)], 360, 1, |_, _, _| {});
    eprintln!("penetrations {:?}", tr.max);
    assert!(tr.max.total() <= 10, "{:?}", tr.max);
}
