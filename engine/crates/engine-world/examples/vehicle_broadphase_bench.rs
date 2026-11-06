//! The explicit vehicles' coupling plan and step against map size
//! (`engine/docs/explicit-vehicles-in-world.md` §10): `cars` repo sedans (surface contact
//! on) in lanes on a square map of side `side` m with `n` rigid bodies: half static posts
//! (boxes and hulls), half boxes resting on the ground (they fall asleep), spread
//! uniformly, plus a fixed cluster of 40 posts and 20 boxes around each car (the same
//! neighbourhood at every n). No GPU fracture: the World keeps its resting table.
//!
//! Per n, after the scene settles (asleep), interleaved A/B (the GPU is shared):
//! * the coupling plan alone (bounds, query or scan, plan, share), `reps` plans in one
//!   submission, mean wall ms per plan: from the World broadphase against the scan of
//!   every collider;
//! * a World step's span (submit to completion, its wait), both modes.
//!
//! ```text
//! perl -e 'alarm 900; exec @ARGV' cargo run --release -p engine-world --example \
//!     vehicle_broadphase_bench -- --cars 1 --n 10000,30000,100000 [--side 1000]
//! ```

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::*;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn sedan(at: Vec3, speed: f32) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let b = |s: &ExplicitBeams| {
        s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        })
    };
    let (lo, hi) = b(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let (lo, hi) = b(&sim);
    let shift = Vec3::new(at.x - 0.5 * (lo.x + hi.x), at.y + 0.02 - lo.y, at.z - hi.z);
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, speed);
    }
    sim.ground = at.y;
    sim.surface_contact = !std::env::args().any(|a| a == "--no-surface");
    sim.dissipative_contact = true;
    sim
}

/// A small deterministic generator (xorshift).
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn world(n: usize, cars: usize, side: f32) -> (World, Vec<ExplicitBeams>) {
    let cap = WorldCapacity {
        max_bodies: (n as u32 + 64 * cars as u32 + 64)
            .next_power_of_two()
            .max(4096),
        max_colliders: (n as u32 + 64 * cars as u32 + 64)
            .next_power_of_two()
            .max(8192),
        ..WorldCapacity::default()
    };
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: cap,
            ..WorldDef::default()
        },
    )
    .unwrap();
    // The ground slab 30 cm under the cars' own ground.
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.8, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(
        g,
        &ShapeDef::default(),
        [0.5 * side + 10.0, 0.5, 0.5 * side + 10.0],
    )
    .unwrap();
    let pts: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 != 0 { 0.15 } else { -0.15 },
                if i & 2 != 0 { 0.45 } else { -0.45 },
                if i & 4 != 0 { 0.15 } else { -0.15 },
            )
        })
        .collect();
    let hull = engine_schema::convex::ConvexHull::from_points(&pts).unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut statics = Vec::new();
    let mut dynamics = Vec::new();
    // Cars on a grid of lanes 12 m apart about the origin.
    let per_row = (cars as f32).sqrt().ceil() as usize;
    let car_at: Vec<Vec3> = (0..cars)
        .map(|i| {
            Vec3::new(
                12.0 * (i % per_row) as f32 - 6.0 * per_row as f32,
                0.0,
                12.0 * (i / per_row) as f32 - 6.0 * per_row as f32,
            )
        })
        .collect();
    let near_car = |p: Vec3| {
        car_at
            .iter()
            .any(|c| (p.x - c.x).abs() < 4.0 && (p.z - (c.z - 2.4)).abs() < 6.0)
    };
    for c in &car_at {
        for k in 0..40 {
            let side = if k % 2 == 0 { -1.0 } else { 1.0 };
            statics.push(Vec3::new(
                c.x + side * (1.05 + 0.5 * ((k / 2) % 3) as f32),
                0.45,
                c.z - 6.0 + 0.6 * (k / 6) as f32,
            ));
        }
        for k in 0..20 {
            dynamics.push(Vec3::new(
                c.x + if k % 2 == 0 { -1.7 } else { 1.7 },
                -0.1,
                c.z - 6.0 + 0.6 * (k / 2) as f32,
            ));
        }
    }
    while statics.len() + dynamics.len() < n + 60 * cars {
        let p = Vec3::new((rng.f() - 0.5) * side, 0.0, (rng.f() - 0.5) * side);
        if near_car(p) {
            continue;
        }
        if (statics.len() + dynamics.len()) % 2 == 0 {
            statics.push(Vec3::new(p.x, 0.45, p.z));
        } else {
            dynamics.push(Vec3::new(p.x, -0.1, p.z));
        }
    }
    let defs: Vec<BodyDef> = statics
        .iter()
        .map(|p| BodyDef {
            position: p.to_array(),
            ..BodyDef::default()
        })
        .collect();
    let ids = w.create_bodies(&defs).unwrap();
    for (k, b) in ids.into_iter().enumerate() {
        if k % 3 == 0 {
            w.create_hull_shape(b, &ShapeDef::default(), &hull).unwrap();
        } else {
            w.create_box_shape(b, &ShapeDef::default(), [0.15, 0.45, 0.15])
                .unwrap();
        }
    }
    let defs: Vec<BodyDef> = dynamics
        .iter()
        .map(|p| BodyDef::dynamic(p.to_array()))
        .collect();
    let ids = w.create_bodies(&defs).unwrap();
    for b in ids {
        w.create_box_shape(b, &ShapeDef::default(), [0.2, 0.2, 0.2])
            .unwrap();
    }
    let fleet: Vec<ExplicitBeams> = car_at.iter().map(|&c| sedan(c, 0.5)).collect();
    (w, fleet)
}

fn main() {
    let cars: usize = arg("--cars").map_or(1, |s| s.parse().unwrap());
    let side: f32 = arg("--side").map_or(1000.0, |s| s.parse().unwrap());
    let reps: u32 = arg("--reps").map_or(20, |s| s.parse().unwrap());
    let ns: Vec<usize> = arg("--n")
        .unwrap_or_else(|| "10000,30000,100000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    println!("| bodies | cars | plan, broadphase (ms) | plan, scan (ms) | step span, broadphase (ms) | step span, scan (ms) | proxy shapes (car 0) | settled awake |");
    println!("|---|---|---|---|---|---|---|---|");
    for &n in &ns {
        let (mut w, fleet) = world(n, cars, side);
        let def = ExplicitVehiclesDef {
            capacity: engine_webgpu::CouplingCapacity {
                bodies: 96,
                shapes: 128,
                planes: 4096,
            },
            ..ExplicitVehiclesDef::default()
        };
        // `--no-sleep`: the cars stay awake (their frame and surface pass run every step).
        let def = if std::env::args().any(|a| a == "--no-sleep") {
            ExplicitVehiclesDef { sleep: None, ..def }
        } else {
            def
        };
        w.add_explicit_vehicles(&fleet, &def).unwrap();
        // Settle: the boxes land and fall asleep.
        for _ in 0..240 {
            w.step().unwrap();
            w.poll_blocking().unwrap();
        }
        let awake = w.counters().map_or(0, |c| c.awake_count);
        let mut span = [Vec::new(), Vec::new()];
        let mut plan = [Vec::new(), Vec::new()];
        for round in 0..8 {
            for (m, scan) in [(0usize, false), (1, true)] {
                let (m, scan) = if round % 2 == 0 {
                    (m, scan)
                } else {
                    (1 - m, !scan)
                };
                w.set_explicit_plan_scan(scan);
                for _ in 0..6 {
                    let t = std::time::Instant::now();
                    w.step().unwrap();
                    w.poll_blocking().unwrap();
                    span[m].push(t.elapsed().as_secs_f64() * 1e3);
                }
                if let Ok(ms) = w.debug_time_explicit_plan(scan, reps) {
                    plan[m].push(ms);
                }
                // The plan's records rebuilt for the next window by a real step.
                w.step().unwrap();
                w.poll_blocking().unwrap();
            }
        }
        let shapes = w.explicit_window_impulses().first().map_or(0, |v| v.len());
        let used = w.explicit_plan_used_broadphase();
        let med = |v: &Vec<f64>| {
            if v.is_empty() {
                f64::NAN
            } else {
                median(v.clone())
            }
        };
        println!(
            "| {n} | {cars} | {:.3} | {:.3} | {:.2} | {:.2} | {shapes} | {awake} (broadphase kept: {used:?}) |",
            med(&plan[0]),
            med(&plan[1]),
            med(&span[0]),
            med(&span[1])
        );
    }
}
