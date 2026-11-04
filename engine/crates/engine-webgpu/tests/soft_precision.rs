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

/// A crash is chaotic: shifting the scene a few metres (no loss of precision)
/// already moves the RMS crush by about 0.2 cm and the deepest node by 1.3 cm,
/// through the order of contacts and whether a panel tears. 10 km out the RMS
/// crush lands within 1 cm of that spread and the deepest node within 2 cm (the
/// node contacts with the ground and wall are the rigid pipeline's, in world
/// f32). Before local frames the 10 km crash crushed 17 cm deeper.
#[test]
fn a_crash_ten_km_out_crushes_as_at_the_origin() {
    let near: Vec<(f32, f32)> = [0.0f32, 1.0, 3.0, 17.0, 50.0]
        .iter()
        .map(|&o| wall_crash(Vec3::new(o, 0.0, o)))
        .collect();
    let (max1, rms1) = wall_crash(Vec3::new(10_000.0, 0.0, 10_000.0));
    let band = |f: fn(&(f32, f32)) -> f32, margin: f32| {
        let v: Vec<f32> = near.iter().map(f).collect();
        (
            v.iter().copied().fold(f32::MAX, f32::min) - margin,
            v.iter().copied().fold(f32::MIN, f32::max) + margin,
        )
    };
    let (lo_max, hi_max) = band(|c| c.0, 0.02);
    let (lo_rms, hi_rms) = band(|c| c.1, 0.01);
    eprintln!("near the origin: {near:?}; 10 km: max {max1:.4} m, RMS {rms1:.4} m");
    assert!(
        (lo_max..=hi_max).contains(&max1) && (lo_rms..=hi_rms).contains(&rms1),
        "10 km out: max {max1} not in {lo_max}..{hi_max} or RMS {rms1} not in {lo_rms}..{hi_rms}"
    );
}
