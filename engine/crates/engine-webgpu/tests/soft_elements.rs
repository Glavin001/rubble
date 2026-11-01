//! LBEAMs, torsion bars and slidenodes on the GPU against the explicit oracle
//! (BeamNG-parity plan, phase 3): each element alone, compared at static
//! equilibrium, which does not depend on the integrator.

use engine_schema::WorldCapacity;
use engine_softbody::elements::fixtures;
use engine_softbody::explicit_ref::{ExplicitSettings, ExplicitSim};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::{
    soft_scene_params, SoftCapacity, SoftSettings, SoftVehicleDesc, SpawnOptions,
};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

fn gpu_settle(desc: &SoftVehicleDesc, seconds: f32) -> Vec<Vec3> {
    let cap = WorldCapacity {
        max_bodies: 1024,
        max_colliders: 2048,
        max_contacts: 8192,
        max_joints: 64,
        max_candidates: 16384,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    let mut p = soft_scene_params(240.0, 20);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    sim.set_soft_settings(SoftSettings::default());
    let h = sim
        .add_soft_vehicle(
            desc,
            [0.0; 3],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions {
                rigid_core: false,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..(240.0 * seconds) as u32 {
        sim.step().unwrap();
    }
    sim.read_soft_particles(h)
        .iter()
        .map(|b| Vec3::from_array(b.position))
        .collect()
}

fn oracle_settle(desc: &SoftVehicleDesc, seconds: f64) -> Vec<Vec3> {
    let mut s = ExplicitSim::new(ExplicitSettings::default());
    s.add_vehicle(desc, Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO);
    s.run(seconds);
    s.positions()
}

fn node(d: &SoftVehicleDesc, name: &str) -> usize {
    d.nodes.iter().position(|n| n.name == name).unwrap()
}

#[test]
fn torsion_bar_sag_matches_the_oracle() {
    let d = load_jbeam(&[("tb.jbeam", fixtures::TORSION_BAR)], None).unwrap();
    let i = node(&d, "d");
    let (g, o) = (gpu_settle(&d, 3.0)[i], oracle_settle(&d, 3.0)[i]);
    eprintln!("torsion sag: GPU {:.4} m, oracle {:.4} m", -g.y, -o.y);
    assert!(-g.y > 0.01, "the bar sags under the node's weight");
    assert!((g - o).length() < 0.002, "GPU {g:?} vs oracle {o:?}");
}

#[test]
fn lbeam_arm_matches_the_oracle() {
    let d = load_jbeam(&[("arm.jbeam", &fixtures::lbeam_arm(true))], None).unwrap();
    let i = node(&d, "t");
    let (g, o) = (gpu_settle(&d, 3.0)[i], oracle_settle(&d, 3.0)[i]);
    eprintln!("LBEAM tip: GPU {g:?}, oracle {o:?}");
    assert!((g - o).length() < 0.003, "GPU {g:?} vs oracle {o:?}");
    assert!((g.y - 1.0).abs() < 0.05, "held up");
}

#[test]
fn slidenode_rests_at_the_cap_like_the_oracle() {
    let d = load_jbeam(&[("r.jbeam", &fixtures::slide_rail(true, 300.0))], None).unwrap();
    let i = node(&d, "s");
    let (g, o) = (gpu_settle(&d, 4.0)[i], oracle_settle(&d, 4.0)[i]);
    eprintln!("slidenode: GPU {g:?}, oracle {o:?}");
    assert!((g - o).length() < 0.01, "GPU {g:?} vs oracle {o:?}");
    assert!(
        (g - Vec3::new(0.0, 0.0, 2.0)).length() < 0.03,
        "at the low cap"
    );
}

/// breakGroupType 1 on the GPU: the passive beam breaks without breaking its group.
#[test]
fn a_passive_group_member_breaks_alone() {
    let jb = r#"{"opt": {"slotType": "main",
        "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1},
            ["a", 0, 0, 0, {"fixed": true}], ["b", 1, 0, 0], ["c", 2, 0, 0]],
        "beams": [["id1:", "id2:"],
            {"beamSpring": 100000, "beamDamp": 0, "beamDeform": 5000, "beamStrength": 2000,
             "breakGroup": "g", "breakGroupType": 1, "precompressionRange": -0.1},
            ["a", "b"],
            {"breakGroupType": 0, "precompressionRange": "", "beamStrength": "FLT_MAX"},
            ["b", "c"]]
    }}"#;
    let d = load_jbeam(&[("opt.jbeam", jb)], None).unwrap();
    let cap = WorldCapacity {
        max_bodies: 256,
        max_colliders: 512,
        max_contacts: 4096,
        max_joints: 64,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    sim.set_params(soft_scene_params(240.0, 10));
    let h = sim
        .add_soft_vehicle(
            &d,
            [0.0; 3],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions {
                rigid_core: false,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..24 {
        sim.step().unwrap();
    }
    let beams = sim.read_soft_beams(h);
    assert!(beams[0].broken(), "the precompressed beam breaks");
    assert!(!beams[1].broken(), "it does not set off its group");
    assert!(beams[0].group_passive);
}
