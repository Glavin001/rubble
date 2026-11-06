//! How still a parked explicit vehicle gets (CPU reference): the repo sedan (or the
//! procedural sedan on raycast wheels) dropped onto its ground at rest (`spawn_clearance`), every 0.25 s the largest node speed component |v|∞ over the
//! last 0.25 s of steps, its RMS node speed and the kinetic energy per kilogram. The
//! vehicle sleep threshold (`explicit_sleep::VehicleSleep`) must sit above the resting
//! floor this reaches and far below any motion that matters.
//!
//! cargo run --release -p engine-softbody --example sleep_probe -- [file.truck] [secs=8] [raycast=1] [fixture=1]
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, STEP};
use glam::Vec3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = args
        .iter()
        .find(|a| !a.contains('='))
        .cloned()
        .unwrap_or_else(|| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
            )
            .into()
        });
    let secs: f32 = args
        .iter()
        .find_map(|a| a.strip_prefix("secs="))
        .map_or(8.0, |s| s.parse().unwrap());
    let raycast = args.iter().any(|a| a == "raycast=1");
    let fixture = args.iter().any(|a| a == "fixture=1");
    let mut sim = if fixture {
        // The JBeam fixture car on pressure wheels (the GPU sleep tests' car).
        let d = engine_softbody::jbeam::load_jbeam(
            &[("t.jbeam", engine_softbody::jbeam_fixture::PART2_CAR)],
            None,
        )
        .unwrap();
        let settings = ExplicitBeamsSettings {
            turbulence: false,
            drag: false,
            ..Default::default()
        };
        ExplicitBeams::from_desc_jbeam_with(
            &d,
            engine_softbody::explicit_beams::JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings,
        )
        .unwrap()
        .0
    } else if raycast {
        // The procedural sedan on raycast wheels (explicit_drive), as the World benches.
        let text =
            engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
                minimass: 10.0,
                ..engine_softbody::procedural::CarParams::sedan()
            });
        ExplicitBeams::from_truck_raycast(
            &text,
            None,
            Vec3::ZERO,
            0.0,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .unwrap()
        .0
    } else {
        let text = std::fs::read_to_string(&file).unwrap();
        ExplicitBeams::from_truck(
            &text,
            Vec3::ZERO,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .unwrap()
    };
    let lift = sim.spawn_clearance() - sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    // x=<m>: the vehicle moved along x (the rounding of every position changes).
    let dx: f32 = args
        .iter()
        .find_map(|a| a.strip_prefix("x="))
        .map_or(0.0, |s| s.parse().unwrap());
    for n in &mut sim.nodes {
        n.pos.y += lift;
        n.pos.x += dx;
    }
    let mass: f32 = sim.nodes.iter().map(|n| n.mass).sum();
    println!("{file}: {} nodes, {mass:.0} kg", sim.nodes.len());
    println!("   t(s)   max|v|inf (m/s) node (y mm)   rms|v - vcm| (m/s)   |vcm| (m/s)   KE/kg (J/kg)   max disp over window (um)   max |mean v| over window (m/s)");
    let per = (0.25 / STEP).round() as u32;
    let total = (secs / STEP).round() as u32;
    let mut vmax = 0.0f32;
    let mut arg = 0usize;
    let mut start: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos).collect();
    let mut lo = start.clone();
    let mut hi = start.clone();
    for k in 1..=total {
        sim.step();
        for (i, n) in sim.nodes.iter().enumerate() {
            let s = n.vel.abs().max_element();
            if s > vmax {
                vmax = s;
                arg = i;
            }
            lo[i] = lo[i].min(n.pos);
            hi[i] = hi[i].max(n.pos);
        }
        if k % per == 0 {
            let ke: f32 = sim
                .nodes
                .iter()
                .map(|n| 0.5 * n.mass * n.vel.length_squared())
                .sum();
            let vcm = sim.nodes.iter().map(|n| n.mass * n.vel).sum::<Vec3>() / mass;
            let rms = (sim
                .nodes
                .iter()
                .map(|n| (n.vel - vcm).length_squared())
                .sum::<f32>()
                / sim.nodes.len() as f32)
                .sqrt();
            let disp = (0..sim.nodes.len())
                .map(|i| (hi[i] - lo[i]).max_element())
                .fold(0.0f32, f32::max);
            let mean = (0..sim.nodes.len())
                .map(|i| (sim.nodes[i].pos - start[i]).abs().max_element())
                .fold(0.0f32, f32::max)
                / (per as f32 * STEP);
            println!(
                "{:7.2}   {vmax:10.3e} {arg:4} ({:6.1})   {rms:12.3e}   {:10.3e}   {:12.3e}   {:10.1}   {mean:10.3e}",
                k as f32 * STEP,
                sim.nodes[arg].pos.y * 1e3,
                vcm.length(),
                ke / mass,
                disp * 1e6
            );
            vmax = 0.0;
            start = sim.nodes.iter().map(|n| n.pos).collect();
            lo = start.clone();
            hi = start.clone();
        }
    }
}
