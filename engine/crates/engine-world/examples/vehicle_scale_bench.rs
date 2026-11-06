//! Scaling benchmark of explicit soft-body vehicles in the World (large maps,
//! `engine/docs/large-maps.md`: a step's cost should follow awake bodies and active
//! destruction, never totals). Repo sedans on their beam tyres, World steps of 1/120 s
//! (17 or 16 explicit steps each), every step's span from its submit to its completion
//! (the bench waits for each step, so the GPU is idle at each submit: the GPU time plus
//! the submit latency and the readback, an upper bound).
//!
//! Scenarios (one per run; each prints one CSV line per point to stdout):
//!   fleet [N..]          N cars driving at 10 m/s in lanes 6 m apart, 15 m apart along a
//!                        lane (fixed density): 64 128 256 512 1024 by default
//!   density [N]          N cars (128) spread (40 m by 20 m), in traffic (nose to tail 1 m
//!                        apart, same speed: contact groups, no contact), in a pile-up
//!                        (pairs head on at 10 m/s each, 8 m apart)
//!   parked [P] [D]       P parked cars (1000) in a lot, settled at rest, plus D (10)
//!                        driving away from them
//!   bodies [N..]         N rigid boxes (10k 30k 100k), 1 % awake (falling and bouncing,
//!                        the rest asleep on the ground), plus 10 driving cars away
//!   map [L..]            nothing active: an L m square map (500 1000 2000 4000) with
//!                        buildings (400 per km², sleeping bodies) and parked cars (64 per km²)
//!
//! Environment: VS_WARM (steps before the measured ones, default 120), VS_STEPS (measured,
//! 120), VS_SETTLE (steps parked cars settle before the measured ones, 480), VS_HZ (120),
//! VS_SLEEP=0 (vehicle sleep off; the World's default is on), VS_LABEL (a CSV column).
//!
//! cargo run --release -p engine-world --example vehicle_scale_bench -- fleet 64 128
//! (one run at a time; bounded: perl -e 'alarm 900; exec @ARGV' ...).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::explicit_scene::place_vehicle;
use rubble::*;

fn sedan() -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    ExplicitBeams::from_truck(
        &text,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap()
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// A world with a ground plane and room for `bodies` rigid bodies.
fn world(bodies: u32) -> anyhow::Result<World> {
    let hz: f32 = env("VS_HZ", 120.0);
    let mut cap = WorldCapacity::small();
    let b = (bodies + 1024).next_power_of_two().max(cap.max_bodies);
    cap.max_bodies = b;
    cap.max_colliders = (2 * b).max(cap.max_colliders);
    cap.max_contacts = cap.max_contacts.max(8 * b);
    cap.max_candidates = cap.max_candidates.max(16 * b);
    let mut solver = AvbdParams::default();
    solver.dt = 1.0 / hz;
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: cap,
            solver,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )?;
    let g = w.create_body(&BodyDef::default())?;
    w.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )?;
    Ok(w)
}

fn add_vehicles(w: &mut World, cars: &[ExplicitBeams]) -> anyhow::Result<()> {
    let mut def = ExplicitVehiclesDef::default();
    // VS_SLEEP=0: vehicle sleep off (the World's default is on).
    if std::env::var("VS_SLEEP").as_deref() == Ok("0") {
        def.sleep = None;
    }
    w.add_explicit_vehicles(cars, &def)
}

/// Steps `warm` steps, then `steps` measured: (p50, p95, p99) of the step span in ms.
fn measure(w: &mut World, warm: u32, steps: u32) -> anyhow::Result<(f64, f64, f64)> {
    for _ in 0..warm {
        w.step()?;
        w.poll_blocking()?;
    }
    let mut ms = Vec::with_capacity(steps as usize);
    let mut ran = 0;
    for _ in 0..steps {
        w.step()?;
        ran += w.poll_blocking()?.iter().filter(|f| f.ran).count();
        if let Some(t) = w.last_submit_instant() {
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let s = w.explicit_sleep_states();
    eprintln!(
        "  {ran} of {steps} measured steps ran; {} of {} vehicles asleep",
        s.iter().filter(|x| x.1).count(),
        s.len()
    );
    // VS_AWAKE=1: where the awake vehicles are (their first node, world), up to 12.
    if std::env::var("VS_AWAKE").is_ok() {
        if let Some(x) = w.explicit_vehicles() {
            let p = x.positions();
            let o = x.origins();
            for (v, _) in s.iter().enumerate().filter(|(_, x)| !x.1).take(12) {
                eprintln!("    awake {v}: node 0 at {:?} (origin {:?})", p[v][0], o[v]);
            }
        }
    }
    Ok((pct(&mut ms, 0.5), pct(&mut ms, 0.95), pct(&mut ms, 0.99)))
}

fn report(scenario: &str, x: &str, cars: usize, bodies: usize, r: (f64, f64, f64), w: &World) {
    let label = std::env::var("VS_LABEL").unwrap_or_default();
    let steps = w.explicit_vehicle_steps().map_or(0, |s| s.0);
    println!(
        "{label},{scenario},{x},{cars},{bodies},{:.3},{:.3},{:.3},{steps}",
        r.0, r.1, r.2
    );
    eprintln!(
        "{scenario} {x}: {cars} cars, {bodies} bodies: p50 {:.2} p95 {:.2} p99 {:.2} ms",
        r.0, r.1, r.2
    );
}

/// Cars in lanes along +Z: `lanes` lanes `dx` apart, `n` cars, `dz` apart along a lane
/// (front to front), at `speed`, starting at `x0`, `z0`.
fn lanes(
    t: &ExplicitBeams,
    n: usize,
    lanes: usize,
    dx: f32,
    dz: f32,
    speed: f32,
    x0: f32,
    z0: f32,
) -> Vec<ExplicitBeams> {
    (0..n)
        .map(|i| {
            let at = Vec3::new(
                x0 + (i % lanes) as f32 * dx,
                0.0,
                z0 - (i / lanes) as f32 * dz,
            );
            place_vehicle(t, at, 0, speed, 0.0)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "fleet".into());
    let nums: Vec<usize> = args.iter().skip(1).filter_map(|a| a.parse().ok()).collect();
    let warm: u32 = env("VS_WARM", 120);
    let steps: u32 = env("VS_STEPS", 120);
    let settle: u32 = env("VS_SETTLE", 480);
    let t = sedan();
    let len = {
        let (lo, hi) = t.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        });
        (hi - lo).max_element()
    };
    match mode.as_str() {
        "fleet" => {
            let ns = if nums.is_empty() {
                vec![64, 128, 256, 512, 1024]
            } else {
                nums
            };
            for n in ns {
                let mut w = world(0)?;
                let lanes_n = (n as f32).sqrt().ceil() as usize;
                add_vehicles(&mut w, &lanes(&t, n, lanes_n, 6.0, 15.0, 10.0, 0.0, 0.0))?;
                let r = measure(&mut w, warm, steps)?;
                report("fleet", &n.to_string(), n, 0, r, &w);
            }
        }
        "density" => {
            let n = nums.first().copied().unwrap_or(128);
            for case in ["spread", "traffic", "pileup"] {
                let mut w = world(0)?;
                let lanes_n = (n as f32).sqrt().ceil() as usize;
                let cars = match case {
                    "spread" => lanes(&t, n, lanes_n, 20.0, 40.0, 10.0, 0.0, 0.0),
                    "traffic" => lanes(&t, n, lanes_n, 3.5, len + 1.0, 10.0, 0.0, 0.0),
                    _ => (0..n)
                        .map(|i| {
                            let pair = i / 2;
                            let at = Vec3::new(
                                (pair % lanes_n) as f32 * 6.0,
                                0.0,
                                (pair / lanes_n) as f32 * 40.0,
                            );
                            if i % 2 == 0 {
                                place_vehicle(&t, at - Vec3::Z * 4.0, 0, 10.0, 0.0)
                            } else {
                                place_vehicle(&t, at + Vec3::Z * 4.0, 2, 10.0, 0.0)
                            }
                        })
                        .collect(),
                };
                add_vehicles(&mut w, &cars)?;
                // The pile-up meets within the warm-up's first steps: measure from the start.
                let r = measure(&mut w, if case == "pileup" { 0 } else { warm }, steps)?;
                report("density", case, n, 0, r, &w);
            }
        }
        "parked" => {
            let p = nums.first().copied().unwrap_or(1000);
            let d = nums.get(1).copied().unwrap_or(10);
            let mut w = world(0)?;
            let cols = (p as f32).sqrt().ceil() as usize;
            let mut cars: Vec<ExplicitBeams> = (0..p)
                .map(|i| {
                    place_vehicle(
                        &t,
                        Vec3::new(
                            (i % cols) as f32 * 3.0,
                            0.0,
                            (i / cols) as f32 * (len + 1.5),
                        ),
                        0,
                        0.0,
                        0.0,
                    )
                })
                .collect();
            cars.extend(lanes(&t, d, d, 6.0, 15.0, 10.0, -200.0, 0.0));
            add_vehicles(&mut w, &cars)?;
            let r = measure(&mut w, settle, steps)?;
            report("parked", &format!("{p}+{d}"), p + d, 0, r, &w);
        }
        "bodies" => {
            let ns = if nums.is_empty() {
                vec![10_000, 30_000, 100_000]
            } else {
                nums
            };
            for n in ns {
                let mut w = world(n as u32)?;
                let cols = (n as f32).sqrt().ceil() as usize;
                let defs: Vec<BodyDef> = (0..n)
                    .map(|i| {
                        let awake = i % 100 == 0;
                        let p = [
                            (i % cols) as f32 * 1.5,
                            if awake { 4.0 + (i % 7) as f32 } else { 0.5 },
                            100.0 + (i / cols) as f32 * 1.5,
                        ];
                        BodyDef {
                            is_awake: awake,
                            ..BodyDef::dynamic(p)
                        }
                    })
                    .collect();
                let ids = w.create_bodies(&defs)?;
                for b in ids {
                    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])?;
                }
                add_vehicles(&mut w, &lanes(&t, 10, 10, 6.0, 15.0, 10.0, -200.0, 0.0))?;
                let r = measure(&mut w, warm, steps)?;
                report("bodies", &n.to_string(), 10, n, r, &w);
            }
        }
        "map" => {
            let ls = if nums.is_empty() {
                vec![500, 1000, 2000, 4000]
            } else {
                nums
            };
            for l in ls {
                let l = l as f32;
                let km2 = l * l / 1e6;
                let mut w = world((400.0 * km2) as u32)?;
                let nb = (400.0 * km2).round() as usize;
                let cols = (nb as f32).sqrt().ceil() as usize;
                let pitch = l / cols as f32;
                for i in 0..nb {
                    // Buildings as sleeping dynamic bodies at rest (a static box this
                    // large past LARGE_STATIC_CAP of them overflows the grid's list of
                    // oversized statics: a rigid-side limit, not this benchmark's).
                    let p = [
                        (i % cols) as f32 * pitch - 0.5 * l,
                        4.0,
                        (i / cols) as f32 * pitch - 0.5 * l,
                    ];
                    let b = w.create_body(&BodyDef {
                        is_awake: false,
                        ..BodyDef::dynamic(p)
                    })?;
                    w.create_box_shape(b, &ShapeDef::default(), [2.0, 4.0, 2.0])?;
                }
                let nc = (64.0 * km2).round().max(1.0) as usize;
                let cc = (nc as f32).sqrt().ceil() as usize;
                let cp = l / cc as f32;
                let cars: Vec<ExplicitBeams> = (0..nc)
                    .map(|i| {
                        // Between the buildings.
                        let at = Vec3::new(
                            (i % cc) as f32 * cp - 0.5 * l + 0.5 * pitch,
                            0.0,
                            (i / cc) as f32 * cp - 0.5 * l + 0.5 * pitch,
                        );
                        place_vehicle(&t, at, 0, 0.0, 0.0)
                    })
                    .collect();
                add_vehicles(&mut w, &cars)?;
                let r = measure(&mut w, settle, steps)?;
                report("map", &format!("{l}"), nc, nb, r, &w);
            }
        }
        m => anyhow::bail!("unknown scenario {m}"),
    }
    Ok(())
}
