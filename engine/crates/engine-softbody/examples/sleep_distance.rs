//! Creep and vehicle sleep against distance from the origin (CPU reference, the World's
//! clock: 120 Hz frames of 17 and 16 steps, each frame rebasing the vehicle origin as the
//! GPU does). A vehicle spawned at rest `d` m out along x (and z), dropped 2 cm onto its
//! ground, as `place_vehicle` spawns them (world positions, origin 0, so the first frame
//! rebases beyond `REBASE_RADIUS`). Reports when it falls asleep (default rule), and its
//! creep: the centre of mass's travel and the largest node travel over the last 2 s of a
//! run with sleep off.
//!
//! cargo run --release -p engine-softbody --example sleep_distance -- [car=sedan|fixture]
//!     [secs=12] [d=0,30,60,100,500,1000,2000,4000]
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_sleep::VehicleSleep;
use glam::Vec3;

fn car(kind: &str) -> ExplicitBeams {
    let mut s = if kind == "fixture" {
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
            JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings,
        )
        .unwrap()
        .0
    } else {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
        ))
        .unwrap();
        ExplicitBeams::from_truck(
            &text,
            Vec3::ZERO,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .unwrap()
    };
    let lift = s.spawn_clearance() - s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += lift;
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(&format!("{k}=")).map(str::to_string))
    };
    let kind = get("car").unwrap_or_else(|| "sedan".into());
    let secs: f32 = get("secs").map_or(12.0, |s| s.parse().unwrap());
    let ds: Vec<f32> = get("d").map_or(
        vec![0.0, 30.0, 60.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0],
        |s| s.split(',').map(|x| x.parse().unwrap()).collect(),
    );
    let frames = (secs * 120.0).round() as u32;
    let tail = 240;
    println!("{kind}, {secs} s: d (m) | asleep at (s) | creep over the last 2 s, sleep off: CoM (mm), largest node (mm), node box (um)");
    for d in ds {
        let base = car(&kind);
        let mass: f32 = base.nodes.iter().map(|n| n.mass).sum();
        let at = Vec3::new(d, 0.0, d * 0.5);
        let place = |mut s: ExplicitBeams| {
            for n in &mut s.nodes {
                n.pos += at;
            }
            s
        };
        let mut on = ExplicitBeamsWorld::new(vec![place(base.clone())]);
        on.sleep = Some(VehicleSleep::default());
        let mut off = ExplicitBeamsWorld::new(vec![place(base)]);
        let mut slept = None;
        // Vehicle-local positions (relative to its origin, which a resting vehicle never
        // rebases again): world f32 positions at 4 km have a 0.24 mm spacing of their own.
        let world = |w: &ExplicitBeamsWorld| -> Vec<Vec3> {
            w.vehicles[0].nodes.iter().map(|n| n.pos).collect()
        };
        let mut start: Vec<Vec3> = Vec::new();
        let mut lo: Vec<Vec3> = Vec::new();
        let mut hi: Vec<Vec3> = Vec::new();
        for f in 0..frames {
            let k = if f % 2 == 0 { 17 } else { 16 };
            on.frame(k);
            off.frame(k);
            if slept.is_none() && on.vehicles[0].sleep.asleep {
                slept = Some(f as f32 / 120.0);
            }
            if f == frames - tail {
                start = world(&off);
                lo = start.clone();
                hi = start.clone();
            }
            if f > frames - tail {
                // Relative to the start so large coordinates do not hide the travel.
                for (i, p) in world(&off).iter().enumerate() {
                    lo[i] = lo[i].min(*p);
                    hi[i] = hi[i].max(*p);
                }
            }
        }
        let end = world(&off);
        let w = &off.vehicles[0];
        let com = |p: &[Vec3]| {
            p.iter()
                .zip(&w.nodes)
                .map(|(x, n)| *x * n.mass)
                .sum::<Vec3>()
                / mass
        };
        let creep = (com(&end) - com(&start)).length();
        let node = end
            .iter()
            .zip(&start)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0f32, f32::max);
        let bx = lo
            .iter()
            .zip(&hi)
            .map(|(a, b)| (*b - *a).max_element())
            .fold(0.0f32, f32::max);
        println!(
            "{d:7.0} | {:>8} | {:8.3} {:8.3} {:10.1}",
            slept.map_or("never".to_string(), |s| format!("{s:.2}")),
            creep * 1e3,
            node * 1e3,
            bx * 1e6
        );
    }
}
