//! The brick ring drive (`rubble::brick_ring_drive`) natively: the ring alone (the rigid
//! baseline) or with cars on raycast wheels driven at the wall; each quarter second the
//! first car's centroid, speed and broken beams and the awake bodies; then the GPU ms per
//! World step (with `BR_PHASES=1` per segment: the rigid step against the vehicles').
//!
//! cargo run --release -p engine-world --example brick_ring_drive_probe
//! (one at a time on a quiet GPU; bounded: run it under `perl -e 'alarm 600; exec @ARGV'`).
//! Environment:
//! - BR_CAR: none (default, the ring alone), etk (a converted vehicle folder, BR_MOD and
//!   BR_PC: default engine-webgpu/__mods/etkk_tex, etkk/kc6.pc) or sedan (the procedural sedan);
//! - BR_N cars (1 or 2), BR_SPEED (m/s at spawn), BR_RUNUP (m from the wall), BR_THROTTLE
//!   (0..1, held), BR_SECS, BR_SHAPES (coupling room per car);
//! - the ring: BR_SCALE, BR_RADIUS, BR_COURSES, BR_TIER, BR_ROWS, BR_DENSITY, BR_BALL
//!   (default: the demo's ring110k with its ball when there is no car); BR_SLEEP=1 sleep on.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::explicit_drive::DriveInput;
use rubble::brick_ring_drive::{brick_ring_drive, car_starts, ring_coupling, RingSpec};
use rubble::showcase_scene::jbeam_vehicle_on;
use rubble::*;

fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> anyhow::Result<()> {
    let car = std::env::var("BR_CAR").unwrap_or_else(|_| "none".into());
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: true,
        turbulence: true,
        ..Default::default()
    };
    let template: Option<ExplicitBeams> = match car.as_str() {
        "none" => None,
        "sedan" => {
            // The explicit solver's procedural sedan (explicitSedanTruck).
            let text =
                engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
                    minimass: 10.0,
                    ..engine_softbody::procedural::CarParams::sedan()
                });
            let (s, notes) = ExplicitBeams::from_truck_raycast(
                &text,
                None,
                glam::Vec3::ZERO,
                0.0,
                glam::Vec3::ZERO,
                settings,
            )
            .map_err(anyhow::Error::msg)?;
            println!(
                "sedan: {} nodes, {} beams; {notes:?}",
                s.nodes.len(),
                s.beams.len()
            );
            Some(s)
        }
        _ => {
            let dir = std::env::var("BR_MOD").unwrap_or_else(|_| {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../engine-webgpu/__mods/etkk_tex"
                )
                .into()
            });
            let dir = std::path::Path::new(&dir);
            let list: Vec<String> =
                serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
            let mut files = Vec::new();
            for f in list.iter().filter(|f| f.ends_with(".jbeam")) {
                files.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
            }
            let pc = std::fs::read_to_string(
                dir.join(std::env::var("BR_PC").unwrap_or_else(|_| "etkk/kc6.pc".into())),
            )?;
            let refs: Vec<(&str, &str)> = files
                .iter()
                .map(|(n, t)| (n.as_str(), t.as_str()))
                .collect();
            let (s, w) = jbeam_vehicle_on(&refs, Some(&pc), settings, JbeamWheels::Raycast)?;
            println!(
                "{car}: {} nodes, {} beams, {} triangles, drive {}; {} warnings",
                s.nodes.len(),
                s.beams.len(),
                s.triangles.len(),
                s.drive.is_some(),
                w.len()
            );
            Some(s)
        }
    };
    // BR_CPU=1: the car alone on the CPU reference (no World, no GPU): from z = 60 at
    // BR_SPEED on full throttle across the origin rebase radius (100 m), each frame
    // checked for non-finite nodes.
    if std::env::var("BR_CPU").as_deref() == Ok("1") {
        let t = template
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("BR_CPU needs BR_CAR"))?;
        let speed = env_f("BR_SPEED", 20.0);
        let mut sim = rubble::explicit_scene::place_vehicle(
            t,
            glam::Vec3::new(0.0, 0.5, 60.0),
            0,
            speed,
            0.5,
        );
        if let Some(d) = sim.drive.as_mut() {
            d.roll(speed);
        }
        sim.set_drive_input(DriveInput {
            forward: 1.0,
            ..Default::default()
        });
        for f in 1..=(env_f("BR_SECS", 4.0) * 60.0) as u32 {
            let o0 = sim.origin;
            sim.frame();
            let bad = sim
                .nodes
                .iter()
                .filter(|n| !n.pos.is_finite() || !n.vel.is_finite())
                .count();
            let c = sim.nodes.iter().map(|n| n.pos).sum::<glam::Vec3>() / sim.nodes.len() as f32
                + sim.origin;
            if f % 15 == 0 || bad > 0 || sim.origin != o0 {
                println!(
                    "frame {f}: centroid ({:.2}, {:.2}, {:.2}), origin {:?}{}{}",
                    c.x,
                    c.y,
                    c.z,
                    sim.origin,
                    if sim.origin != o0 { " REBASED" } else { "" },
                    if bad > 0 {
                        format!(" {bad} NON-FINITE")
                    } else {
                        String::new()
                    }
                );
            }
            if bad > 0 {
                break;
            }
        }
        return Ok(());
    }
    let demo = RingSpec::demo();
    let ring = RingSpec {
        scale: env_f("BR_SCALE", demo.scale),
        radius: env_f("BR_RADIUS", demo.radius),
        courses: env_f("BR_COURSES", demo.courses as f32) as u32,
        tier: env_f("BR_TIER", demo.tier as f32) as u32,
        rows: env_f("BR_ROWS", demo.rows as f32) as u32,
        density: env_f(
            "BR_DENSITY",
            if template.is_some() {
                rubble::brick_ring_drive::drive_ring().density
            } else {
                demo.density
            },
        ),
        ball: env_f("BR_BALL", if template.is_some() { 0.0 } else { 1.0 }) > 0.0,
    };
    let n = env_f("BR_N", 1.0) as usize;
    let cars = car_starts(&ring, n, env_f("BR_RUNUP", 30.0), env_f("BR_SPEED", 0.0));
    let phases = std::env::var("BR_PHASES").as_deref() == Ok("1");
    let t0 = std::time::Instant::now();
    let mut s = brick_ring_drive(
        GpuContext::Default,
        ring,
        |d| {
            d.gpu_timing = if phases {
                GpuTiming::Phases
            } else {
                GpuTiming::Step
            };
            // BR_SLEEP=1: demo.html's ?sleep=on (bodies sleep after 0.5 s still).
            if std::env::var("BR_SLEEP").as_deref() == Ok("1") {
                d.solver.sleep_time = 0.5;
            }
        },
        template.as_slice(),
        &cars,
        &ring_coupling(env_f("BR_SHAPES", 1024.0) as u32),
    )?;
    println!(
        "{} bricks, wall at {:.1} m, {:.1} m high; built in {:.1} s",
        s.bricks,
        ring.inner(),
        ring.height(),
        t0.elapsed().as_secs_f32()
    );
    let w = &mut s.world;
    let throttle = env_f("BR_THROTTLE", 1.0);
    if let Some(x) = w.explicit_vehicles_mut() {
        for v in 0..x.vehicle_count() {
            x.set_drive_input(
                v,
                DriveInput {
                    forward: throttle,
                    ..Default::default()
                },
            );
        }
    }
    let dt = w.solver().dt;
    let steps = (env_f("BR_SECS", 4.0) / dt).round() as u32;
    let mut ms = Vec::new();
    let mut seg: std::collections::BTreeMap<&'static str, (f64, u32)> = Default::default();
    let mut awake = 0u32;
    let mut wall = std::time::Instant::now();
    for k in 1..=steps {
        let ts = std::time::Instant::now();
        w.step()?;
        for f in w.poll_blocking()? {
            for (n, t) in f.gpu_time.iter().flat_map(|t| t.segments.iter()) {
                let e = seg.entry(n).or_default();
                e.0 += t;
                e.1 += 1;
            }
            if let (Some(t), true) = (&f.gpu_time, k > 30) {
                ms.push(t.total_ms);
            }
            if let Some(c) = f.counters {
                awake = c.awake_count;
            }
        }
        // Guards (a GPU run must never hang the machine): a slow step is printed at once;
        // every step the cars' nodes are checked, and the run stops at the first
        // non-finite one, or when car 0's node 0 nears the origin rebase radius (100 m)
        // unless BR_FAR=1.
        let step_ms = ts.elapsed().as_secs_f64() * 1e3;
        if step_ms > 250.0 {
            println!("step {k}: {step_ms:.0} ms wall");
        }
        if let Some(x) = w.explicit_vehicles() {
            let pos = x.positions();
            let bad = pos.iter().flatten().filter(|p| !p.is_finite()).count();
            // The explicit frame's loop-cap flag (a loop that reached its cap or the
            // frame's loop budget): the frame's results are not to be trusted.
            if let Err(e) = x.kernel_status() {
                println!("step {k}: explicit kernel status: {e}; stopping");
                break;
            }
            if bad > 0 {
                println!("step {k}: {bad} non-finite nodes; stopping");
                break;
            }
            if std::env::var("BR_FAR").as_deref() != Ok("1") && pos[0][0].length() > 95.0 {
                println!("step {k}: car 0 node 0 at {:.1} m from the origin; stopping (BR_FAR=1 to go on)", pos[0][0].length());
                break;
            }
        }
        if k % 15 == 0 {
            // Each car's proxy shapes in its last window (the coupling's room per car).
            let shapes: Vec<usize> = w
                .explicit_window_impulses()
                .iter()
                .map(|v| v.len())
                .collect();
            let line = match w.explicit_vehicles() {
                Some(x) => {
                    let (pos, vel, beams) = (x.positions(), x.velocities(), x.beams());

                    let broken: Vec<usize> = beams
                        .iter()
                        .map(|b| b.iter().filter(|b| b.broken).count())
                        .collect();
                    let c = pos[0].iter().copied().sum::<glam::Vec3>() / pos[0].len() as f32;
                    let v = vel[0].iter().copied().sum::<glam::Vec3>() / vel[0].len() as f32;
                    let bad = pos.iter().flatten().filter(|p| !p.is_finite()).count();
                    format!(
                        "car 0 at ({:.2}, {:.2}, {:.2}) {:.1} m/s; beams broken {:?}; proxy shapes {:?}{}",
                        c.x,
                        c.y,
                        c.z,
                        v.length(),
                        broken,
                        shapes,
                        if bad > 0 {
                            format!(" NON-FINITE {bad}")
                        } else {
                            String::new()
                        }
                    )
                }
                None => String::new(),
            };
            println!(
                "t {:.2} s ({:.0} ms/step wall): awake {awake}; {line}",
                k as f32 * dt,
                wall.elapsed().as_secs_f64() * 1e3 / 15.0
            );
            wall = std::time::Instant::now();
            for e in w
                .explicit_vehicles_mut()
                .map(|x| x.take_contact_errors())
                .unwrap_or_default()
            {
                println!("  error: {e}");
            }
        }
    }
    if phases {
        let mut v: Vec<(&str, f64)> = seg
            .iter()
            .map(|(n, (t, _))| (*n, t / f64::from(steps)))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let list: Vec<String> = v
            .iter()
            .take(14)
            .map(|(n, t)| format!("{n} {t:.2}"))
            .collect();
        println!("segments (mean ms per step): {}", list.join(", "));
    }
    ms.sort_by(f64::total_cmp);
    if !ms.is_empty() {
        println!(
            "GPU per World step (after the first 30): median {:.2} ms, p95 {:.2} ({} steps)",
            ms[ms.len() / 2],
            ms[ms.len() * 95 / 100],
            ms.len()
        );
    }
    Ok(())
}
