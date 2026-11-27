//! The generated 4x4 at its own mass (`engine-vehiclegen` `designs/suv4x4/mass.rs`,
//! `engine/docs/generated-vehicles-lessons.md` §21), parked on its raycast wheels: it
//! settles, no beam yields or breaks, its ride height holds still and its kinetic energy
//! does not grow once it has settled. Also prints its mass, the axle split and the ride
//! height (the numbers the lessons doc records).
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test suv4x4_rest -- --nocapture`

use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody_authoring::authoring::gen2::vg_input;
use engine_softbody_authoring::vehicle_compare::{lowest_point, settings};
use glam::Vec3;
use std::path::PathBuf;

const DESIGN: &str = "suv_4x4";
const DT: f32 = 0.0005;

fn rig() -> (engine_softbody::desc::SoftVehicleDesc, ExplicitBeams) {
    let inp = vg_input::VgInput::find(DESIGN, None, None).expect("design");
    let b = vg_input::build(inp).expect("rig");
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("suv4x4_rest");
    for (p, bytes) in vg_input::files(&b) {
        let path = dir.join(&p);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
    }
    let files = engine_softbody::jbeam::read_vehicle_dir(&dir, DESIGN).unwrap();
    let pc = std::fs::read_to_string(dir.join(DESIGN).join(format!("{DESIGN}.pc"))).ok();
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap();
    let (mut sim, _) = ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    sim.surface_contact = true;
    (desc, sim)
}

fn kinetic(s: &ExplicitBeams) -> f64 {
    s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum()
}

/// Mass-weighted mean height of the body's nodes (wheels, hubs and rims left out).
fn ride_height(s: &ExplicitBeams, body: &[usize]) -> f64 {
    let (mut m, mut y) = (0.0f64, 0.0f64);
    for &i in body {
        let n = &s.nodes[i];
        m += n.mass as f64;
        y += n.mass as f64 * (s.origin.y + n.pos.y) as f64;
    }
    y / m
}

#[test]
fn suv_4x4_rests_at_its_own_mass() {
    let (desc, mut sim) = rig();
    let susp = |n: &str| n.starts_with("hub") || n.starts_with("rim") || n.starts_with("whe");
    let body: Vec<usize> = (0..desc.nodes.len()).filter(|&i| !susp(&desc.nodes[i].name)).collect();
    // Mass and the static axle split (the centre of mass between the axles, as rigged).
    let mass: f64 = desc.nodes.iter().map(|n| n.mass as f64).sum();
    let wheel_kg: f64 = desc.wheels.iter().map(|w| w.mass as f64).sum();
    let hub = |w: &engine_softbody::desc::SoftWheel| 0.5 * (desc.nodes[w.node1 as usize].position[2] + desc.nodes[w.node2 as usize].position[2]);
    let zs: Vec<f32> = desc.wheels.iter().map(hub).collect();
    let (zf, zr) = (zs.iter().cloned().fold(f32::MIN, f32::max) as f64, zs.iter().cloned().fold(f32::MAX, f32::min) as f64);
    let mut mz: f64 = desc.nodes.iter().map(|n| n.mass as f64 * n.position[2] as f64).sum();
    mz += desc.wheels.iter().map(|w| w.mass as f64 * hub(w) as f64).sum::<f64>();
    let com_z = mz / (mass + wheel_kg);
    let front = (com_z - zr) / (zf - zr);
    println!("{DESIGN}: nodes {mass:.1} kg + raycast wheels {wheel_kg:.1} kg = {:.1} kg; axle split {:.1} % front / {:.1} % rear", mass + wheel_kg, 100.0 * front, 100.0 * (1.0 - front));

    let low = lowest_point(&sim);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - low;
    }
    let steps = |s: f32| (s / DT).round() as usize;
    let mut ke = Vec::new();
    let mut heights = Vec::new();
    for k in 0..steps(4.0) {
        sim.step();
        if (k + 1) % steps(0.1) == 0 {
            ke.push(kinetic(&sim));
            heights.push(ride_height(&sim, &body));
        }
    }
    let finite = sim.nodes.iter().all(|n| n.pos.is_finite() && n.vel.is_finite());
    let broken = sim.beams.iter().filter(|b| b.broken).count();
    let bent = sim.beams.iter().filter(|b| b.bent()).count();
    // Settled after 1.5 s: from there the ride height holds within 1 mm, and the kinetic
    // energy's envelope does not grow (a parked car only loses energy: its largest value in
    // the last second is at most its largest over 1.5-2.5 s).
    let settled = 15 - 1;
    let h0 = heights[settled];
    let h_spread = heights[settled..].iter().map(|h| (h - h0).abs()).fold(0.0, f64::max);
    let ke_settled = ke[settled..25].iter().cloned().fold(0.0, f64::max);
    let ke_max_after = ke[30..].iter().cloned().fold(0.0, f64::max);
    println!(
        "{DESIGN}: parked 4 s: ride height {:.1} mm (body COM), moved {:.2} mm after 1.5 s; KE at 0.5 s {:.3} J, largest over 1.5-2.5 s {:.5} J, over 3-4 s {:.5} J, at 4 s {:.5} J; broken {broken}, bent {bent}",
        h0 * 1e3,
        h_spread * 1e3,
        ke[4],
        ke_settled,
        ke_max_after,
        ke.last().unwrap()
    );
    assert!(finite, "state not finite");
    assert_eq!((broken, bent), (0, 0), "beams broken, bent at rest");
    assert!(h_spread <= 0.001, "ride height moved {:.2} mm after settling", h_spread * 1e3);
    assert!(ke_max_after <= ke_settled.max(1e-3), "kinetic energy grew at rest: {ke_settled} J over 1.5-2.5 s, {ke_max_after} J over 3-4 s");
}
