//! Vehicle–vehicle contact of the explicit solver against Rigs of Rods' own
//! (the reference harness `ror_ref`, `scale` mode with `vv=1`), as a black box,
//! on two copies of one real `.truck` / `.car` file.
//!
//! ```bash
//! RUBBLE_AIRFOIL_DIR=… cargo run -p engine-webgpu --release --bin explicit_vv_compare -- \
//!     <file.truck> ror=<path>/ror_ref scene=rear speed=10 frames=90 steps=17 nudges=6 \
//!     out=<dir> [gpu=1] [config=<name>]
//! ```
//!
//! Scenes (both vehicles at x, z ≥ 100, lowest node 2 cm above the ground, tyres
//! rolling, no per-node drag; the vehicle's forward axis from its camera nodes):
//!   rear    vehicle 0 still, vehicle 1 behind it in the same lane at `speed=` m/s,
//!           `gap=` m (default 1) between them
//!   headon  vehicle 1 ahead of vehicle 0, turned by `yaw=` degrees (default 180),
//!           both at `speed=` towards each other
//!   tbone   vehicle 1 standing across vehicle 0's lane (turned `yaw=`, default 90),
//!           vehicle 0 into its side at `speed=`
//!   pass    side by side lanes `lane=` m apart (default 6), vehicle 1 overtaking at
//!           `speed=` (no contact: checks a two-vehicle world against single runs)
//!
//! Runs `nudges=` K runs per engine, run k spawning the moving vehicle 0.1 mm/s × k
//! faster (the accuracy rule, `engine/docs/explicit-beams-acceptance/ACCURACY-RULE.md`):
//! RoR's own spread (median node RMS over its run pairs), Rubble's distance to RoR
//! (median over every Rubble–RoR pair), and broken / bent counts, per vehicle. Also
//! a run of each engine stopped `pre=` frames in (default: before the first contact
//! Rubble sees) for the before-contact agreement, and RoR without vehicle contact
//! (`vv=0`) for scale. `gpu=1` adds Rubble's GPU (`ExplicitBeamsGpu`).

use engine_softbody::explicit_beams::{yaw_offset, ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::truck::parse_truck;
use glam::Vec3;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct End {
    pos: [Vec<Vec3>; 2],
    broken: [usize; 2],
    bent: [usize; 2],
}

struct Spawn {
    at: Vec3,
    vel: Vec3,
    yaw: f32,
}

fn spawn(text: &str, config: Option<&str>, s: &Spawn) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let mut sim = ExplicitBeams::from_truck_config_yaw(
        text,
        config,
        Vec3::new(s.at.x, 0.0, s.at.z),
        s.yaw,
        s.vel,
        settings,
    )
    .expect("spawn");
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    sim.roll_tyres(s.vel);
    sim
}

fn rms(a: &[Vec3], b: &[Vec3]) -> f32 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x.distance_squared(*y)))
        .sum();
    (s / a.len().max(1) as f64).sqrt() as f32
}

fn median(mut x: Vec<f32>) -> f32 {
    if x.is_empty() {
        return f32::NAN;
    }
    x.sort_by(f32::total_cmp);
    let m = x.len() / 2;
    if x.len() % 2 == 1 {
        x[m]
    } else {
        0.5 * (x[m - 1] + x[m])
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ror(
    bin: &Path,
    truck: &Path,
    dir: &Path,
    tag: &str,
    spawns: &[Spawn],
    frames: u32,
    steps: u32,
    vv: bool,
    config: Option<&str>,
) -> End {
    let scene = dir.join(format!("{tag}.scene"));
    let mut t = String::new();
    for s in spawns {
        writeln!(
            t,
            "vehicle {} {} {} {} {} {}",
            s.at.x, s.at.z, s.vel.x, s.vel.y, s.vel.z, s.yaw
        )
        .unwrap();
    }
    std::fs::write(&scene, t).unwrap();
    let out = dir.join(format!("{tag}.json"));
    let status = std::process::Command::new(bin)
        .arg(truck)
        .arg("scale")
        .arg(format!("scene={}", scene.display()))
        .arg("warm=0")
        .arg(format!("frames={frames}"))
        .arg(format!("steps={steps}"))
        .arg("roll=1")
        .arg("drag=0")
        .arg("sample=0,1")
        .arg("threads=1")
        .arg(format!("vv={}", vv as u8))
        .args(config.map(|c| format!("config={c}")))
        .arg(format!("out={}", out.display()))
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()
        .expect("ror_ref");
    assert!(status.success(), "ror_ref failed ({tag})");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let mut end = End {
        pos: [Vec::new(), Vec::new()],
        broken: [0; 2],
        bent: [0; 2],
    };
    for e in v["vehicles"].as_array().unwrap() {
        let i = e["index"].as_u64().unwrap() as usize;
        let p: Vec<f32> = e["pos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        end.pos[i] = p.chunks(3).map(Vec3::from_slice).collect();
        end.broken[i] = e["broken"].as_u64().unwrap() as usize;
        end.bent[i] = e["bent"].as_u64().unwrap() as usize;
    }
    end
}

fn cpu_end(world: &ExplicitBeamsWorld) -> End {
    let v = &world.vehicles;
    End {
        pos: [v[0].positions(), v[1].positions()],
        broken: [v[0].damage().0, v[1].damage().0],
        bent: [v[0].damage().1, v[1].damage().1],
    }
}

fn run_cpu(
    text: &str,
    config: Option<&str>,
    spawns: &[Spawn],
    frames: u32,
    steps: u32,
    contact: bool,
) -> (End, Option<u64>) {
    let fleet: Vec<ExplicitBeams> = spawns.iter().map(|s| spawn(text, config, s)).collect();
    let mut w = ExplicitBeamsWorld::new(fleet);
    w.contact = contact;
    for _ in 0..frames {
        w.frame(steps);
    }
    (cpu_end(&w), w.first_hit_step)
}

fn run_gpu(text: &str, config: Option<&str>, spawns: &[Spawn], frames: u32, steps: u32) -> End {
    use engine_webgpu::{default_device, ExplicitBeamsGpu};
    let (_, device, queue) = default_device().expect("device");
    let fleet: Vec<ExplicitBeams> = spawns.iter().map(|s| spawn(text, config, s)).collect();
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    for _ in 0..frames {
        gpu.frame(steps);
    }
    gpu.wait();
    let pos = gpu.positions();
    let beams = gpu.beams();
    let count = |k: usize, f: &dyn Fn(&engine_webgpu::GpuBeam) -> bool| {
        beams[k].iter().filter(|b| f(b)).count()
    };
    End {
        pos: [pos[0].clone(), pos[1].clone()],
        broken: [count(0, &|b| b.broken), count(1, &|b| b.broken)],
        bent: [count(0, &|b| b.bent()), count(1, &|b| b.bent())],
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let truck = PathBuf::from(
        args.first()
            .expect("usage: explicit_vv_compare <file.truck> key=value…"),
    );
    let kv: HashMap<&str, &str> = args
        .iter()
        .skip(1)
        .filter_map(|a| a.split_once('='))
        .collect();
    let get = |k: &str, d: f32| kv.get(k).map_or(d, |v| v.parse().unwrap());
    let bin = PathBuf::from(kv.get("ror").copied().unwrap_or(
        "/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref",
    ));
    let scene = kv.get("scene").copied().unwrap_or("rear");
    let speed = get("speed", 10.0);
    let frames = get("frames", 90.0) as u32;
    let steps = get("steps", 17.0) as u32;
    let nudges = get("nudges", 6.0) as usize;
    let config = kv.get("config").copied();
    let gpu = kv.get("gpu") == Some(&"1");
    let dir = PathBuf::from(kv.get("out").copied().unwrap_or("/tmp/explicit_vv_compare"));
    std::fs::create_dir_all(&dir)?;
    let text = std::fs::read_to_string(&truck)?;

    // The vehicle's forward axis and extent along it (spawned at the origin).
    let probe = spawn(
        &text,
        config,
        &Spawn {
            at: Vec3::ZERO,
            vel: Vec3::ZERO,
            yaw: 0.0,
        },
    );
    let file = parse_truck(&text);
    let p = |i: u32| probe.nodes[i as usize].pos;
    let forward = file
        .cameras
        .as_ref()
        .map(|c| {
            let d = p(c.center) - p(c.back);
            Vec3::new(d.x, 0.0, d.z).normalize()
        })
        .unwrap_or(-Vec3::X);
    let lateral = Vec3::Y.cross(forward).normalize();
    let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
        (a.min(n.pos.dot(forward)), b.max(n.pos.dot(forward)))
    });
    let length = hi - lo;
    let gap = get("gap", 1.0);
    let base = Vec3::new(130.0, 0.0, 130.0);
    let spawns_for = |k: usize| -> Vec<Spawn> {
        let nudge = 1e-4 * k as f32;
        match scene {
            "rear" => vec![
                Spawn {
                    at: base,
                    vel: Vec3::ZERO,
                    yaw: 0.0,
                },
                Spawn {
                    at: base - forward * (length + gap),
                    vel: forward * (speed + nudge),
                    yaw: 0.0,
                },
            ],
            "headon" | "tbone" => {
                // Vehicle 1 turned by `yaw=` (head-on 180°, T-bone 90°) about its
                // spawn point, its near face `gap` ahead of vehicle 0's front,
                // centred on vehicle 0's centre line; head-on both drive at
                // `speed`, T-bone vehicle 1 stands.
                let yaw = get("yaw", if scene == "headon" { 180.0 } else { 90.0 });
                let turned: Vec<Vec3> =
                    probe.nodes.iter().map(|n| yaw_offset(n.pos, yaw)).collect();
                let span = |v: &[Vec3], axis: Vec3| {
                    v.iter().fold((f32::MAX, f32::MIN), |(a, b), p| {
                        (a.min(p.dot(axis)), b.max(p.dot(axis)))
                    })
                };
                let own: Vec<Vec3> = probe.nodes.iter().map(|n| n.pos).collect();
                let (alo, ahi) = span(&own, lateral);
                let (rlo, _) = span(&turned, forward);
                let (clo, chi) = span(&turned, lateral);
                let v1 = if scene == "headon" {
                    -forward * speed
                } else {
                    Vec3::ZERO
                };
                vec![
                    Spawn {
                        at: base,
                        vel: forward * (speed + nudge),
                        yaw: 0.0,
                    },
                    Spawn {
                        at: base
                            + forward * (hi + gap - rlo)
                            + lateral * (0.5 * (alo + ahi) - 0.5 * (clo + chi)),
                        vel: v1,
                        yaw,
                    },
                ]
            }
            "pass" => vec![
                Spawn {
                    at: base,
                    vel: forward * speed * 0.5,
                    yaw: 0.0,
                },
                Spawn {
                    at: base + lateral * get("lane", 6.0) - forward * (length + gap),
                    vel: forward * (speed + nudge),
                    yaw: 0.0,
                },
            ],
            s => panic!("scene {s}: rear, headon or pass"),
        }
    };
    if let Some(yaws) = kv.get("spawncheck") {
        // Spawned node positions against ror_ref's (frames=0) for each yaw.
        for yaw in yaws.split(',').map(|y| y.parse::<f32>().unwrap()) {
            let sp = [
                Spawn {
                    at: Vec3::new(150.0, 0.0, 150.0),
                    vel: Vec3::ZERO,
                    yaw,
                },
                Spawn {
                    at: Vec3::new(300.0, 0.0, 300.0),
                    vel: Vec3::ZERO,
                    yaw: 0.0,
                },
            ];
            let r = run_ror(&bin, &truck, &dir, "spawn", &sp, 0, steps, true, config);
            let c = spawn(&text, config, &sp[0]).positions();
            let max = c
                .iter()
                .zip(&r.pos[0])
                .map(|(a, b)| a.distance(*b))
                .fold(0.0f32, f32::max);
            let exact = c.iter().zip(&r.pos[0]).filter(|(a, b)| a == b).count();
            println!(
                "yaw {yaw}: rms {:.2e} m, max {max:.2e} m, bit-equal {exact}/{}",
                rms(&c, &r.pos[0]),
                c.len()
            );
            if kv.contains_key("verbose") {
                let off: Vec<String> = c
                    .iter()
                    .zip(&r.pos[0])
                    .enumerate()
                    .filter(|(_, (a, b))| a.distance(**b) > 1e-3)
                    .map(|(i, (a, b))| {
                        format!(
                            "{i}:{:.3},{:.3},{:.3}/{:.3},{:.3},{:.3}",
                            a.x, a.y, a.z, b.x, b.y, b.z
                        )
                    })
                    .collect();
                println!("  off: {}", off.join(" "));
            }
        }
        return Ok(());
    }
    if let Some(counts) = kv.get("bench") {
        // GPU ms per frame (median of the timed frames), contact on and off, for
        // fleets of `counts` vehicles driving in a grid `spacing=` m apart (no
        // contact): the cost of the contact plan when nothing meets.
        use engine_webgpu::{default_device, ExplicitBeamsGpu};
        let (_, device, queue) = default_device()?;
        let spacing = get("spacing", 20.0);
        for n in counts.split(',').map(|c| c.parse::<usize>().unwrap()) {
            let cols = (n as f32).sqrt().ceil() as usize;
            let fleet: Vec<ExplicitBeams> = (0..n)
                .map(|k| {
                    let at = base + lateral * (spacing * (k % cols) as f32)
                        - forward * (spacing * 3.0 * (k / cols) as f32);
                    spawn(
                        &text,
                        config,
                        &Spawn {
                            at: at + Vec3::new(500.0, 0.0, 500.0),
                            vel: forward * speed,
                            yaw: 0.0,
                        },
                    )
                })
                .collect();
            let mut out = Vec::new();
            for trial in 0..4 {
                let on = trial % 2 == 0;
                let mut g = ExplicitBeamsGpu::new(&device, &queue, &fleet);
                g.set_vehicle_contact(on);
                for _ in 0..10 {
                    g.frame(steps);
                }
                g.wait();
                let timed = g.enable_frame_timing(frames);
                let t0 = std::time::Instant::now();
                for _ in 0..frames {
                    g.frame(steps);
                }
                g.wait();
                let wall = t0.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
                let mut t = if timed {
                    g.frame_times_ms()
                } else {
                    Vec::new()
                };
                t.sort_by(f64::total_cmp);
                out.push((on, t.get(t.len() / 2).copied().unwrap_or(wall)));
                if on {
                    g.contact_status().map_err(|e| anyhow::anyhow!(e))?;
                }
            }
            let med = |on: bool| {
                let v: Vec<f64> = out.iter().filter(|o| o.0 == on).map(|o| o.1).collect();
                v.iter().sum::<f64>() / v.len() as f64
            };
            println!(
                "N={n}: GPU ms/frame contact on {:.4}, off {:.4} (ratio {:.3})",
                med(true),
                med(false),
                med(true) / med(false)
            );
        }
        return Ok(());
    }
    if let Some(n) = kv.get("stepwise") {
        // GPU against CPU step by step (one-step frames): node RMS per vehicle and
        // the triangles whose contact throttles differ.
        use engine_webgpu::{default_device, ExplicitBeamsGpu};
        let n: u32 = n.parse()?;
        let from: u32 = kv.get("from").map_or(0, |v| v.parse().unwrap());
        let fleet: Vec<ExplicitBeams> = spawns_for(0)
            .iter()
            .map(|s| spawn(&text, config, s))
            .collect();
        let (_, device, queue) = default_device()?;
        let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
        let mut w = ExplicitBeamsWorld::new(fleet);
        for k in 0..n {
            w.frame(1);
            gpu.frame(1);
            if k + 1 < from {
                continue;
            }
            gpu.wait();
            let gp = gpu.positions();
            let ct = gpu.contact_throttles();
            let st = gpu.self_collision_throttles();
            let mut diff = Vec::new();
            for v in 0..2 {
                for (t, c) in ct[v].iter().enumerate() {
                    if *c != w.vehicles[v].contact_throttle[t] {
                        diff.push(format!(
                            "c{v}.{t}:{c:?}/{:?}",
                            w.vehicles[v].contact_throttle[t]
                        ));
                    }
                }
                for (t, c) in st[v].iter().enumerate() {
                    let tr = &w.vehicles[v].triangles[t];
                    if c[0] != tr.wait && !(c[0] <= 0 && tr.wait <= 0) {
                        diff.push(format!("s{v}.{t}:{c:?}/{:?}", (tr.wait, tr.idle)));
                    }
                }
            }
            println!(
                "step {} hits {} self {} {} rms {:.2e} {:.2e} diffs {}",
                k + 1,
                w.last.hits,
                w.vehicles[0].last_contacts,
                w.vehicles[1].last_contacts,
                rms(&gp[0], &w.vehicles[0].positions()),
                rms(&gp[1], &w.vehicles[1].positions()),
                diff.iter().take(6).cloned().collect::<Vec<_>>().join(" ")
            );
        }
        return Ok(());
    }
    let mut report = String::new();
    writeln!(
        report,
        "# {} — scene {scene}, {speed} m/s, {frames} frames × {steps} steps",
        truck.display()
    )?;
    writeln!(
        report,
        "nodes {}, beams {}, contact points {}, triangles {}",
        probe.nodes.len(),
        probe.beams.len(),
        probe.contact_points.len(),
        probe.triangles.len()
    )?;

    // Before contact.
    let (_, first) = run_cpu(&text, config, &spawns_for(0), frames, steps, true);
    let pre = kv
        .get("pre")
        .map(|v| v.parse::<u32>().unwrap())
        .or(first.map(|s| ((s.saturating_sub(1)) / u64::from(steps)) as u32))
        .unwrap_or(frames);
    writeln!(
        report,
        "Rubble CPU first contact: step {first:?} (frame {:?})",
        first.map(|s| (s - 1) / u64::from(steps))
    )?;
    if pre > 0 {
        let r = run_ror(
            &bin,
            &truck,
            &dir,
            "pre",
            &spawns_for(0),
            pre,
            steps,
            true,
            config,
        );
        let (c, _) = run_cpu(&text, config, &spawns_for(0), pre, steps, true);
        writeln!(
            report,
            "before contact ({pre} frames): node RMS CPU–RoR vehicle 0 {:.2e} m, vehicle 1 {:.2e} m",
            rms(&c.pos[0], &r.pos[0]),
            rms(&c.pos[1], &r.pos[1])
        )?;
        if gpu {
            let g = run_gpu(&text, config, &spawns_for(0), pre, steps);
            writeln!(
                report,
                "before contact ({pre} frames): node RMS GPU–CPU vehicle 0 {:.2e} m, vehicle 1 {:.2e} m",
                rms(&g.pos[0], &c.pos[0]),
                rms(&g.pos[1], &c.pos[1])
            )?;
        }
    }
    // Ensembles.
    let ror: Vec<End> = (0..nudges)
        .map(|k| {
            run_ror(
                &bin,
                &truck,
                &dir,
                &format!("ror{k}"),
                &spawns_for(k),
                frames,
                steps,
                true,
                config,
            )
        })
        .collect();
    let cpu: Vec<End> = (0..nudges)
        .map(|k| run_cpu(&text, config, &spawns_for(k), frames, steps, true).0)
        .collect();
    let gpu_runs: Vec<End> = if gpu {
        (0..nudges)
            .map(|k| run_gpu(&text, config, &spawns_for(k), frames, steps))
            .collect()
    } else {
        Vec::new()
    };
    let ror_off = run_ror(
        &bin,
        &truck,
        &dir,
        "ror_novv",
        &spawns_for(0),
        frames,
        steps,
        false,
        config,
    );
    let (cpu_off, _) = run_cpu(&text, config, &spawns_for(0), frames, steps, false);
    writeln!(report)?;
    writeln!(report, "| vehicle | RoR spread M_RR (m) | CPU–RoR M_XR (m) | ratio | GPU–RoR (m) | GPU–CPU (m) | RoR vv=0 vs vv=1 (m) | CPU vv=0 – RoR vv=0 (m) |")?;
    writeln!(report, "|---|---|---|---|---|---|---|---|")?;
    for i in 0..2 {
        let mut rr = Vec::new();
        for a in 0..ror.len() {
            for b in a + 1..ror.len() {
                rr.push(rms(&ror[a].pos[i], &ror[b].pos[i]));
            }
        }
        let xr: Vec<f32> = cpu
            .iter()
            .flat_map(|c| ror.iter().map(move |r| rms(&c.pos[i], &r.pos[i])))
            .collect();
        let gr: Vec<f32> = gpu_runs
            .iter()
            .flat_map(|c| ror.iter().map(move |r| rms(&c.pos[i], &r.pos[i])))
            .collect();
        let gc: Vec<f32> = gpu_runs
            .iter()
            .flat_map(|c| cpu.iter().map(move |r| rms(&c.pos[i], &r.pos[i])))
            .collect();
        let (mrr, mxr) = (median(rr), median(xr));
        writeln!(
            report,
            "| {i} | {mrr:.4} | {mxr:.4} | {:.2} | {:.4} | {:.4} | {:.4} | {:.2e} |",
            mxr / mrr,
            median(gr),
            median(gc),
            rms(&ror_off.pos[i], &ror[0].pos[i]),
            rms(&cpu_off.pos[i], &ror_off.pos[i]),
        )?;
    }
    writeln!(report)?;
    writeln!(report, "| vehicle | RoR broken (runs) | CPU broken | GPU broken | RoR bent | CPU bent | GPU bent |")?;
    writeln!(report, "|---|---|---|---|---|---|---|")?;
    for i in 0..2 {
        let list = |e: &[End], f: &dyn Fn(&End) -> usize| {
            e.iter()
                .map(f)
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        writeln!(
            report,
            "| {i} | {} | {} | {} | {} | {} | {} |",
            list(&ror, &|e| e.broken[i]),
            list(&cpu, &|e| e.broken[i]),
            list(&gpu_runs, &|e| e.broken[i]),
            list(&ror, &|e| e.bent[i]),
            list(&cpu, &|e| e.bent[i]),
            list(&gpu_runs, &|e| e.bent[i]),
        )?;
    }
    print!("{report}");
    std::fs::write(dir.join("report.md"), report)?;
    Ok(())
}
