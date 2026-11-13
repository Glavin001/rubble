//! Creep and vehicle sleep against distance from the origin (CPU reference, the World's
//! clock: 120 Hz frames of 17 and 16 steps, each frame rebasing the vehicle origin as the
//! GPU does). A vehicle spawned at rest `d` m out along x (and z), dropped 2 cm onto its
//! ground, as `place_vehicle` spawns them (world positions, origin 0, so the first frame
//! rebases beyond `REBASE_RADIUS`). Reports when it falls asleep (default rule), and its
//! creep: the centre of mass's travel and the largest node travel over the last 2 s of a
//! run with sleep off.
//!
//! cargo run --release -p engine-softbody --example sleep_distance --
//!     [car=sedan|fixture|fixture_raycast|etkk] [secs=12] [d=0,30,60,100,500,1000,2000,4000]
//!     [spawn=local]
//!
//! sedan: the repo sedan (RoR profile, beam tyres); fixture: the JBeam fixture car on
//! pressure wheels; fixture_raycast: the same car on raycast wheels; etkk: the ETK
//! K-series kr6 on raycast wheels (BeamNG profile; RUBBLE_ETK, its `vehicles` directory).
//! spawn=local: the origin at the placement, the nodes as built. SD_TRACE=1 prints the
//! motion every 0.5 s and the nodes that moved most; SD_SUB="from|to;..." edits the
//! fixture's JBeam text (diagnostics).
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_sleep::VehicleSleep;
use glam::Vec3;

fn car(kind: &str) -> ExplicitBeams {
    let mut s = if kind == "etkk" {
        // The ETK K-series kr6 (community JBeam, BeamNG profile) on raycast wheels, as the
        // vehicle-compare harness spawns it; RUBBLE_ETK: the `vehicles` directory.
        let dir = std::env::var("RUBBLE_ETK").unwrap_or_else(|_| {
            format!(
                "{}/Downloads/jbeams/etkk_lucasbe/vehicles",
                std::env::var("HOME").unwrap()
            )
        });
        let dir = std::path::Path::new(&dir);
        let files = engine_softbody::jbeam::read_vehicle_dir(dir, "etkk").expect("etkk");
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pc = std::fs::read_to_string(dir.join("etkk/kr6.pc")).expect("kr6.pc");
        let desc = engine_softbody::jbeam::load_jbeam(&refs, Some(&pc)).unwrap();
        ExplicitBeams::from_desc_jbeam_raycast(
            &desc,
            Vec3::ZERO,
            Vec3::ZERO,
            engine_softbody::vehicle_compare::settings(),
        )
        .unwrap()
        .0
    } else if kind == "fixture_raycast" {
        // The fixture car on raycast wheels (no tyre nodes): the rocking's source check.
        let d = engine_softbody::jbeam::load_jbeam(
            &[("t.jbeam", engine_softbody::jbeam_fixture::PART2_CAR)],
            None,
        )
        .unwrap();
        ExplicitBeams::from_desc_jbeam_raycast(
            &d,
            Vec3::ZERO,
            Vec3::ZERO,
            engine_softbody::vehicle_compare::settings(),
        )
        .unwrap()
        .0
    } else if kind == "fixture" {
        // SD_SUB="from|to;from|to": text substitutions in the fixture's JBeam (a diagnostic
        // knob for tracing its settling: tyre damping, friction, pressure).
        let mut text = engine_softbody::jbeam_fixture::PART2_CAR.to_string();
        if let Ok(sub) = std::env::var("SD_SUB") {
            for pair in sub.split(';').filter(|p| !p.is_empty()) {
                let (a, b) = pair.split_once('|').expect("from|to");
                assert!(text.contains(a), "SD_SUB: {a} not in the fixture");
                text = text.replace(a, b);
            }
        }
        let d = engine_softbody::jbeam::load_jbeam(&[("t.jbeam", text.as_str())], None).unwrap();
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
    let trace = std::env::var("SD_TRACE").is_ok();
    println!("{kind}, {secs} s: d (m) | asleep at (s) | creep over the last 2 s, sleep off: CoM (mm), largest node (mm), node box (um)");
    for d in ds {
        let base = car(&kind);
        let mass: f32 = base.nodes.iter().map(|n| n.mass).sum();
        let at = Vec3::new(d, 0.0, d * 0.5);
        // spawn=local: the origin at the placement and the nodes as built (exact);
        // default: the offset added to every node (rounded to the float spacing there),
        // as `place_vehicle` places them.
        let local = args.iter().any(|a| a == "spawn=local");
        let place = |mut s: ExplicitBeams| {
            if local {
                s.origin = at;
            } else {
                for n in &mut s.nodes {
                    n.pos += at;
                }
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
            if trace && f % 60 == 59 {
                // SD_TRACE=1: every 0.5 s (sleep off): the CoM's vertical velocity, the
                // fastest node (index, flags, |v|), the fastest node in ground contact
                // (within 5 mm of it: horizontal |v|, vertical |v|).
                let v = &off.vehicles[0];
                let m: f32 = v.nodes.iter().map(|n| n.mass).sum();
                let cv = v.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
                let (fi, fv) = v
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (i, n.vel.abs().max_element()))
                    .fold((0, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
                let g = v.ground - v.origin.y;
                let (gi, gh, gy) = v
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.pos.y - g < 0.005)
                    .map(|(i, n)| (i, Vec3::new(n.vel.x, 0.0, n.vel.z).length(), n.vel.y.abs()))
                    .fold((0, 0.0f32, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
                let contacts = v.nodes.iter().filter(|n| n.pos.y - g < 0.005).count();
                // The chassis (the fixture's first 8 nodes): its fastest node.
                let chassis = v
                    .nodes
                    .iter()
                    .take(8)
                    .map(|n| n.vel.length())
                    .fold(0.0f32, f32::max);
                println!("  t {:5.1} s | CoM v ({:+.5} {:+.5} {:+.5}) chassis {:.5} | fastest node {fi} ({:#x}) {:.5} | {contacts} at the ground, fastest {gi} ({:#x}) horiz {:.5} vert {:.5}",
                    (f + 1) as f32 / 120.0, cv.x, cv.y, cv.z, chassis, v.nodes[fi].flags, fv, v.nodes[gi].flags, gh, gy);
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
        if trace {
            // The nodes that moved most over the tail (their box), with their flags.
            let mut v: Vec<(f32, usize)> = lo
                .iter()
                .zip(&hi)
                .enumerate()
                .map(|(i, (a, b))| ((*b - *a).max_element(), i))
                .collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (bx, i) in v.iter().take(8) {
                let n = &w.nodes[*i];
                eprintln!(
                    "  node {i}: box {:.1} um flags {:#x} mass {:.3} pos {:?} vel {:?}",
                    bx * 1e6,
                    n.flags,
                    n.mass,
                    n.pos,
                    n.vel
                );
            }
        }
        println!(
            "{d:7.0} | {:>8} | {:8.3} {:8.3} {:10.1}",
            slept.map_or("never".to_string(), |s| format!("{s:.2}")),
            creep * 1e3,
            node * 1e3,
            bx * 1e6
        );
    }
}
