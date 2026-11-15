//! The ETK (BeamNG community content, local only) into one obstacle in a small GPU World,
//! through the explicit vehicles' rigid coupling: a diagnosis of what the coupling
//! transmits (`engine/docs/explicit-vehicles-in-world.md`). Compare with the CPU
//! reference `engine-softbody` example `coupling_probe`.
//!
//! Environment:
//! - CW_WALL: `anchored` (one static 10 × 3 × 2 m box body), `free` (one dynamic box,
//!   CW_MASS kg, default 1500, 2.4 × 1.4 × 1 m), `bricks` (CW_COLS × CW_ROWS × CW_DEEP
//!   free 1 × 0.5 × 0.5 m bricks of CW_DENSITY kg/m³, default 8 × 6 × 4 at 100: 25 kg
//!   each, running bond), `none`;
//! - CW_HZ: World steps per second (60, 120, 240; default 60), CW_KMH (50), CW_SECS (1.0),
//!   CW_RUNUP (m from the nose to the wall at spawn, default 1.0).
//!
//! Run as the one GPU job: `engine/scripts/with-gpu-lock.py --limit 300 -- env
//! RUBBLE_GPU_LOCK=0 target/release/examples/coupling_wall_gpu`.
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use glam::{Quat, Vec3};
use rubble::brick_ring_drive::{ring_coupling, ring_world_def};
use rubble::explicit_scene::place_vehicle;
use rubble::showcase_scene::jbeam_vehicle_on;
use rubble::*;

fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

const GROUND: f32 = 0.5;

fn main() -> anyhow::Result<()> {
    let wall = std::env::var("CW_WALL").unwrap_or_else(|_| "anchored".into());
    let hz = env_f("CW_HZ", 60.0);
    let kmh = env_f("CW_KMH", 50.0);
    let secs = env_f("CW_SECS", 1.0);
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    // CW_MOD: another mod folder (a generated car's, `vehicle_gen_rig`),
    // CW_PC its `.pc` under the folder (default the ETK's).
    let dir = std::env::var("CW_MOD").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../engine-webgpu/__mods/etkk_tex"
        ))
    });
    let list: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
    let mut files = Vec::new();
    for f in list.iter().filter(|f| f.ends_with(".jbeam")) {
        files.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
    }
    let pc = std::fs::read_to_string(dir.join(std::env::var("CW_PC").unwrap_or_else(|_| "etkk/kc6.pc".into())))?;
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let (template, _): (ExplicitBeams, _) =
        jbeam_vehicle_on(&refs, Some(&pc), settings, JbeamWheels::Raycast)?;
    let masses: Vec<f32> = template.nodes.iter().map(|n| n.mass).collect();
    let m: f32 = masses.iter().sum();

    let (cols, rows, deep) = (
        env_f("CW_COLS", 8.0) as i32,
        env_f("CW_ROWS", 6.0) as i32,
        env_f("CW_DEEP", 4.0) as i32,
    );
    // CW_FLY=1: no ground and no World gravity (the bricks float as stacked, the car
    // flies on its own gravity, vertical only): horizontal momentum must close.
    let fly = std::env::var("CW_FLY").as_deref() == Ok("1");
    let mut def = ring_world_def(4096);
    def.solver.dt = 1.0 / hz;
    if fly {
        def.gravity = [0.0, 0.0, 0.0];
    }
    let mut w = World::new(GpuContext::Default, &def)?;
    let g = w.create_body(&BodyDef::default())?;
    if !fly {
        w.create_box_shape(
            g,
            &ShapeDef {
                material: SurfaceMaterial {
                    friction: 0.5,
                    ..Default::default()
                },
                density: 0.0,
                filter: engine_schema::convex::CollisionFilter {
                    categories: 0x8000,
                    ..Default::default()
                },
                ..ShapeDef::default()
            },
            [100.0, 0.5, 100.0],
        )?;
    }
    let face = 0.0f32;
    let mut free_box = None;
    let mut bricks = Vec::new();
    match wall.as_str() {
        "anchored" => {
            let b = w.create_body(&BodyDef {
                position: [0.0, GROUND + 1.5, face + 1.0],
                ..BodyDef::default()
            })?;
            w.create_box_shape(
                b,
                &ShapeDef {
                    density: 0.0,
                    ..ShapeDef::default()
                },
                [5.0, 1.5, 1.0],
            )?;
        }
        "free" => {
            let half = [1.2f32, 0.7, 0.5];
            let mass = env_f("CW_MASS", 1500.0);
            let b = w.create_body(&BodyDef {
                linear_damping: 0.0,
                angular_damping: 0.0,
                ..BodyDef::dynamic([0.0, GROUND + 0.75, face + half[2]])
            })?;
            w.create_box_shape(
                b,
                &ShapeDef {
                    density: mass / (8.0 * half[0] * half[1] * half[2]),
                    ..ShapeDef::default()
                },
                half,
            )?;
            free_box = Some((b, mass));
        }
        "bricks" => {
            let density = env_f("CW_DENSITY", 100.0);
            let mut defs = Vec::new();
            for d in 0..deep {
                for r in 0..rows {
                    for c in 0..cols {
                        let x = (c as f32 - 0.5 * (cols - 1) as f32) * 1.02
                            + if r % 2 == 1 { 0.51 } else { 0.0 };
                        defs.push(BodyDef {
                            rotation: Quat::IDENTITY.to_array(),
                            ..BodyDef::dynamic([
                                x,
                                GROUND + 0.25 + 0.52 * r as f32,
                                face + 0.25 + 0.52 * d as f32,
                            ])
                        });
                    }
                }
            }
            bricks = w.create_bodies(&defs)?;
            let shape = ShapeDef {
                material: SurfaceMaterial {
                    friction: 0.5,
                    ..Default::default()
                },
                density,
                ..ShapeDef::default()
            };
            for id in &bricks {
                w.create_box_shape(*id, &shape, [0.5, 0.25, 0.25])?;
            }
        }
        _ => {}
    }
    if let Ok(v) = std::env::var("CW_SHOOT") {
        // No car: a rigid-only check of the World's momentum among the bricks. One more
        // brick (CW_SHOOT_MASS kg, default 1453) at v m/s into the wall's middle.
        let v: f32 = v.parse()?;
        let mass = env_f("CW_SHOOT_MASS", 1453.0);
        let b = w.create_body(&BodyDef {
            linear_velocity: [0.0, 0.0, v],
            linear_damping: 0.0,
            angular_damping: 0.0,
            ..BodyDef::dynamic([0.0, GROUND + 0.75, face - 1.0])
        })?;
        w.create_box_shape(
            b,
            &ShapeDef {
                density: mass / (8.0 * 0.9 * 0.5 * 0.25),
                ..ShapeDef::default()
            },
            [0.9, 0.5, 0.25],
        )?;
        let bm = 25.0f32 * env_f("CW_DENSITY", 100.0) / 100.0;
        let total = |w: &World| -> Vec3 {
            bricks
                .iter()
                .map(|q| Vec3::from(w.read_body_state(*q).unwrap().linear_velocity) * bm)
                .sum::<Vec3>()
                + Vec3::from(w.read_body_state(b).unwrap().linear_velocity) * mass
        };
        let p0 = total(&w);
        for k in 1..=(secs * hz) as u32 {
            w.step()?;
            w.poll_blocking()?;
            if k % (hz as u32 / 10).max(1) == 0 {
                let p = total(&w);
                println!(
                    "t {:.2}: total momentum {p:.1?} (start {p0:.1?}); shooter v {:.2?}",
                    k as f32 / hz,
                    Vec3::from(w.read_body_state(b)?.linear_velocity)
                );
            }
        }
        return Ok(());
    }
    let runup = env_f("CW_RUNUP", 1.0);
    let car = place_vehicle(
        &template,
        Vec3::new(0.0, GROUND, face - runup),
        0,
        kmh / 3.6,
        GROUND,
    );
    let mut car = car;
    if fly {
        car.ground = -1.0e4;
    }
    if let Some(d) = car.drive.as_mut() {
        d.roll(kmh / 3.6);
    }
    let zlo = car.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
    let zhi = car.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let rear: Vec<usize> = (0..car.nodes.len())
        .filter(|&i| car.nodes[i].pos.z < zlo + 0.33 * (zhi - zlo))
        .collect();
    let len0 = zhi - zlo;
    w.add_explicit_vehicles(std::slice::from_ref(&car), &ring_coupling(1024))?;
    let p_car = |w: &World| -> Vec3 {
        w.explicit_vehicles().unwrap().velocities()[0]
            .iter()
            .zip(&masses)
            .map(|(v, m)| *v * *m)
            .sum()
    };
    let p0 = p_car(&w);
    let steps = (secs * hz).round() as u32;
    let mut peak_win = 0.0f32;
    let mut j_total = Vec3::ZERO;
    println!(
        "wall {wall}, {hz} Hz World, {kmh} km/h, car {m:.0} kg, {} bricks",
        bricks.len()
    );
    for k in 1..=steps {
        w.step()?;
        w.poll_blocking()?;
        let x = w.explicit_vehicles().unwrap();
        if let Err(e) = x.kernel_status() {
            println!("step {k}: kernel status {e}; stopping");
            break;
        }
        let pos = x.positions();
        if pos[0].iter().any(|p| !p.is_finite()) {
            println!("step {k}: non-finite; stopping");
            break;
        }
        let imp = w.explicit_window_impulses();
        let j: Vec3 = imp[0].iter().map(|s| Vec3::from(s.1)).sum();
        j_total += j;
        let p = p_car(&w);
        let fwin = j.z * hz;
        peak_win = peak_win.max(fwin);
        if k % (hz as u32 / 30).max(1) == 0 || k == steps {
            let rz: f32 = rear.iter().map(|&i| pos[0][i].z * masses[i]).sum::<f32>()
                / rear.iter().map(|&i| masses[i]).sum::<f32>();
            let broken = x.beams()[0].iter().filter(|b| b.broken).count();
            let moving = bricks
                .iter()
                .filter(|b| {
                    w.read_body_state(**b)
                        .map(|s| Vec3::from(s.linear_velocity).length() > 0.5)
                        .unwrap_or(false)
                })
                .count();
            println!(
                "t {:.3}: car v_z {:.2} m/s; window impulse on shapes {:.0} N s (mean {:.0} kN), {} shapes; rear-third z {rz:.3}; broken {broken}; bricks moving {moving}",
                k as f32 / hz,
                p.z / m,
                j.z,
                fwin / 1e3,
                imp[0].len()
            );
        }
    }
    let x = w.explicit_vehicles().unwrap();
    let pos = x.positions();
    let (lo, hi) = pos[0]
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), q| (l.min(q.z), h.max(q.z)));
    let p1 = p_car(&w);
    let broken = x.beams()[0].iter().filter(|b| b.broken).count();
    let bent = x.beams()[0]
        .iter()
        .filter(|b| (b.rest - b.rest0).abs() > 1e-3 * b.rest0.abs())
        .count();
    println!(
        "end: car dp_z {:.0} N s (from {:.0}); sum of window impulses on shapes {:.1?} N s; peak window-mean force {:.0} kN; length {:.3} (was {len0:.3}); broken {broken}, bent {bent}",
        p1.z - p0.z,
        p0.z,
        j_total,
        peak_win / 1e3,
        hi - lo
    );
    if !bricks.is_empty() {
        let bm = 25.0f32 * env_f("CW_DENSITY", 100.0) / 100.0;
        let pb: Vec3 = bricks
            .iter()
            .map(|b| Vec3::from(w.read_body_state(*b).unwrap().linear_velocity) * bm)
            .sum();
        println!(
            "bricks' momentum {pb:.1?} N s; car dp + bricks p (horizontal) x {:.1} z {:.1} N s",
            p1.x - p0.x + pb.x,
            p1.z - p0.z + pb.z
        );
    }
    if let Some((b, mass)) = free_box {
        let v = Vec3::from(w.read_body_state(b)?.linear_velocity);
        println!(
            "free box {mass} kg: v {v:.3?}; box momentum z {:.0} N s; car + box dp_z {:.1} N s",
            v.z * mass,
            p1.z - p0.z + v.z * mass
        );
    }
    Ok(())
}
