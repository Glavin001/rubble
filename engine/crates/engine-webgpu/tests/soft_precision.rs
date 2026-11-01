//! Float precision far from the origin (BeamNG-parity plan, phase 4): a soft
//! body's particles are solved in its own local frame, so the same crash 10 km
//! from the world origin crushes as it does at the origin. With world f32
//! positions (about 1 mm apart at 10 km) the sub-millimetre corrections of a
//! 240 Hz step were rounded away and the 56 km/h wall crush grew by 17 cm.

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::calibrate::{residual_displacement, rms_max};
use engine_softbody::procedural::{car_desc, CarParams};
use engine_softbody::{soft_scene_params, SoftCapacity, SoftSettings, SpawnOptions};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::Vec3;

/// The sedan without a rigid core into a rigid wall at 56 km/h, the whole scene
/// moved by `offset`: (max residual crush, RMS) of its body nodes, and its sag
/// after standing for a second before the run.
fn wall_crash(offset: Vec3) -> (f32, f32) {
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
    let mut p = soft_scene_params(240.0, 10);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    sim.set_soft_settings(SoftSettings::default());
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
    for (center, half) in [
        (Vec3::new(0.0, -5.0, 0.0), Vec3::new(400.0, 5.0, 400.0)),
        (Vec3::new(0.0, 1.5, 3.0), Vec3::new(3.0, 1.5, 0.2)),
    ] {
        let b = sim.add_body(BodyState::static_plane_support());
        sim.add_collider(Collider {
            body: b,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: (center + offset).to_array(),
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [half.x, half.y, half.z, half.length()],
        });
    }
    let (desc, _) = car_desc(&CarParams::sedan());
    let at = Vec3::new(0.0, 0.06 - desc.bounds().0[1], -1.0);
    let v = 56.0 / 3.6;
    let h = sim
        .add_soft_vehicle(
            &desc,
            (at + offset).to_array(),
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions {
                rigid_core: false,
                velocity: [0.0, 0.0, v],
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..360 {
        sim.step().unwrap();
    }
    let body = desc.part("body").unwrap().nodes.clone();
    let rest: Vec<Vec3> = body
        .iter()
        .map(|&i| Vec3::from_array(desc.nodes[i as usize].position) + at)
        .collect();
    let now: Vec<Vec3> = {
        let all = sim.read_soft_particles(h);
        body.iter()
            .map(|&i| Vec3::from_array(all[i as usize].position) - offset)
            .collect()
    };
    let (rms, max) = rms_max(&residual_displacement(&rest, &now));
    (max, rms)
}

#[test]
#[ignore]
fn crush_spread_probe() {
    for o in [0.0f32, 0.0, 1.0, 3.0, 17.0, 10_000.0, 10_003.0] {
        let (m, r) = wall_crash(Vec3::new(o, 0.0, o));
        eprintln!(
            "offset {o}: max {:.2} cm, RMS {:.3} cm",
            m * 100.0,
            r * 100.0
        );
    }
}

/// The RMS crush matches to 0.1 mm. The deepest node (a single node of a chaotic
/// crash) differs by about 1.3 cm: the node-to-ground and node-to-wall contacts are
/// the rigid pipeline's, in world f32, and already move it 0.2 cm for a 1 m shift
/// and 0.4 cm at 10,003 m (`crush_spread_probe`). Before local frames the 10 km
/// crash crushed 58 cm against 41.
#[test]
fn a_crash_ten_km_out_crushes_as_at_the_origin() {
    let (max0, rms0) = wall_crash(Vec3::ZERO);
    let (max1, rms1) = wall_crash(Vec3::new(10_000.0, 0.0, 10_000.0));
    eprintln!(
        "origin: crush max {:.2} cm, RMS {:.3} cm; 10 km: max {:.2} cm, RMS {:.3} cm",
        max0 * 100.0,
        rms0 * 100.0,
        max1 * 100.0,
        rms1 * 100.0
    );
    assert!(
        (rms1 - rms0).abs() < 0.001 && (max1 - max0).abs() < 0.02,
        "crush differs 10 km out: max {max0} vs {max1}, RMS {rms0} vs {rms1}"
    );
}
