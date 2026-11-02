//! Explicit beams, GPU against the CPU reference on one `.truck` file, and a
//! many-vehicle benchmark.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin explicit_beams_check -- car.truck \
//!     pos=0,0.5,0 vel=0,0,15.6 box=0,1,4,3,1,0.5 secs=1
//! cargo run -p engine-webgpu --release --bin explicit_beams_check -- car.truck bench \
//!     cars=1,10,40,100,400,1000
//! ```
//! The path `sedan` is the procedural crash-test sedan (`car_truck_bare`, minimass
//! 10). Options: `pos=x,y,z` spawn position, `vel=x,y,z` spawn velocity,
//! `box=cx,cy,cz,hx,hy,hz` a static box (centre, half extents), `secs=` simulated
//! time, `bare=1` drops the `wheels`, `engine` and `brakes` sections, `turb=1` turns
//! turbulence on (off by default: CPU and GPU draw different noise), `roll=1` spins
//! the tyres to roll at the spawn velocity, `drag=0` turns the per-node drag off,
//! `trace=<file>` with `trace_every=<frames>` (or `trace_steps=<steps>`, GPU only)
//! writes the GPU run: u32 nodes, u32
//! beams, then per sample every node's world position (f32 xyz) and every beam's
//! broken flag (u8).
//!
//! Comparison: every 0.1 s, GPU-vs-CPU node distance (RMS, max) and broken/bent
//! beam counts on each. Bench: GPU ms per 60 fps frame (33 steps; submit to
//! completion, median and best of 120) for each vehicle count, and the CPU
//! reference's single-thread time for one vehicle.

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, StaticBox,
};
use engine_softbody::procedural::{car_truck_bare, strip_truck_sections, CarParams};
use engine_webgpu::{default_device, render_scene_json, ExplicitBeamsGpu};
use glam::Vec3;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

fn vec3(s: &str) -> Vec3 {
    let v: Vec<f32> = s.split(',').map(|x| x.parse().expect("number")).collect();
    Vec3::new(v[0], v[1], v[2])
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        anyhow::bail!("usage: explicit_beams_check <file.truck | sedan> [bench] key=value…");
    };
    let bench = args.iter().any(|a| a == "bench");
    let kv: HashMap<&str, &str> = args
        .iter()
        .skip(1)
        .filter_map(|a| a.split_once('='))
        .collect();
    let mut text = if path == "sedan" {
        car_truck_bare(&CarParams {
            minimass: 10.0,
            ..CarParams::sedan()
        })
    } else {
        std::fs::read_to_string(path)?
    };
    if kv.get("bare") == Some(&"1") {
        text = strip_truck_sections(&text, &["wheels", "engine", "brakes"]);
    }
    let settings = ExplicitBeamsSettings {
        drag: kv.get("drag") != Some(&"0"),
        turbulence: kv.get("turb") == Some(&"1"),
        ..Default::default()
    };
    let pos = kv.get("pos").map(|s| vec3(s)).unwrap_or(Vec3::ZERO);
    let vel = kv.get("vel").map(|s| vec3(s)).unwrap_or(Vec3::ZERO);
    let mut cpu =
        ExplicitBeams::from_truck(&text, pos, vel, settings).map_err(|e| anyhow::anyhow!(e))?;
    if let Some(b) = kv.get("box") {
        let v: Vec<f32> = b.split(',').map(|x| x.parse().expect("number")).collect();
        anyhow::ensure!(v.len() == 6, "box=cx,cy,cz,hx,hy,hz");
        cpu.boxes.push(StaticBox::centered(
            Vec3::new(v[0], v[1], v[2]),
            Vec3::new(v[3], v[4], v[5]),
        ));
    }
    // `ground=1`: stand the vehicle on its lowest node, 2 cm above the ground.
    if kv.get("ground") == Some(&"1") {
        let lowest = cpu.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        for n in &mut cpu.nodes {
            n.pos.y += 0.02 - lowest;
        }
    }
    // `roll=1`: tyres rolling at the spawn velocity (not skidding).
    if kv.get("roll") == Some(&"1") {
        cpu.roll_tyres(vel);
    }
    // `scene=<file.json>`: what a renderer needs besides the trace: the skin
    // (every `cab` triangle, collision or not), each tyre's node rings, the boxes.
    if let Some(out) = kv.get("scene") {
        let boxes: Vec<(Vec3, Vec3)> = cpu
            .boxes
            .iter()
            .map(|b| ((b.lo + b.hi) * 0.5, (b.hi - b.lo) * 0.5))
            .collect();
        // `visual=sedan`: the procedural sedan's render mesh (its nodes are the truck's
        // first nodes): per vertex the frame nodes and coordinates, triangles, materials.
        let visual = if kv.get("visual") == Some(&"sedan") {
            let (mesh, _) = engine_softbody::visual::procedural_car_visual(&CarParams::sedan());
            let verts: Vec<String> = mesh
                .bindings
                .iter()
                .map(|b| {
                    format!(
                        "[{},{},{},{},{},{}]",
                        b.nodes[0], b.nodes[1], b.nodes[2], b.coords[0], b.coords[1], b.coords[2]
                    )
                })
                .collect();
            let vt: Vec<String> = mesh
                .tris
                .iter()
                .map(|t| format!("[{},{},{}]", t[0], t[1], t[2]))
                .collect();
            let mats: Vec<String> = mesh.tri_material.iter().map(|m| m.to_string()).collect();
            format!(
                ",\"visual\":{{\"bindings\":[{}],\"triangles\":[{}],\"materials\":[{}]}}",
                verts.join(","),
                vt.join(","),
                mats.join(",")
            )
        } else {
            String::new()
        };
        std::fs::write(
            out,
            render_scene_json(&text, cpu.nodes.len(), 1, &boxes, &visual),
        )?;
    }
    let (info, device, queue) = default_device()?;
    println!(
        "{}: {} nodes, {} beams, {} collision triangles, {} contacters ({})",
        path,
        cpu.nodes.len(),
        cpu.beams.len(),
        cpu.triangles.len(),
        cpu.candidates.len(),
        info.name
    );
    if bench {
        return run_bench(&device, &queue, &cpu, &kv);
    }

    let secs: f32 = kv.get("secs").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let trace_every: u32 = kv
        .get("trace_every")
        .map(|s| s.parse().unwrap())
        .unwrap_or(8);
    let mut trace = match kv.get("trace") {
        Some(f) => {
            let mut w = std::io::BufWriter::new(std::fs::File::create(f)?);
            w.write_all(&(cpu.nodes.len() as u32).to_le_bytes())?;
            w.write_all(&(cpu.beams.len() as u32).to_le_bytes())?;
            Some(w)
        }
        None => None,
    };
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    let mut clock = FrameClock::default();
    let frames = (secs * 60.0).round() as u32;
    let mut write_sample = |gpu: &ExplicitBeamsGpu| -> anyhow::Result<()> {
        if let Some(w) = trace.as_mut() {
            for p in &gpu.positions()[0] {
                for c in p.to_array() {
                    w.write_all(&c.to_le_bytes())?;
                }
            }
            let flags: Vec<u8> = gpu.beams()[0].iter().map(|b| b.broken as u8).collect();
            w.write_all(&flags)?;
        }
        Ok(())
    };
    write_sample(&gpu)?;
    // `trace_steps=N`: the GPU run alone, sampled every N steps (to line up with a
    // reference trace sampled the same way).
    if let Some(n) = kv.get("trace_steps").map(|s| s.parse::<u32>().unwrap()) {
        let total = (secs / engine_softbody::explicit_beams::STEP).round() as u32;
        let mut done = 0;
        while done < total {
            gpu.frame(n);
            done += n;
            write_sample(&gpu)?;
        }
        if let Some(mut w) = trace {
            w.flush()?;
        }
        println!("trace: {} samples every {n} steps", total / n + 1);
        return Ok(());
    }
    println!("   t     rms(mm)   max(mm)   cpu broken/bent   gpu broken/bent");
    for f in 1..=frames {
        let steps = cpu.frame();
        let s2 = clock.next();
        debug_assert_eq!(steps, s2);
        gpu.frame(s2);
        if f % trace_every == 0 {
            write_sample(&gpu)?;
        }
        if f % 6 == 0 || f == frames {
            let g = &gpu.positions()[0];
            let c = cpu.positions();
            let (rms, max) = distance(&c, g);
            let gb = gpu.beams();
            let g_broken = gb[0].iter().filter(|b| b.broken).count();
            let g_bent = gb[0].iter().filter(|b| b.bent()).count();
            let (c_broken, c_bent) = cpu.damage();
            println!(
                "{:5.2} {:9.3} {:9.3}   {:7}/{:<7}   {:7}/{:<7}",
                f as f32 / 60.0,
                rms * 1e3,
                max * 1e3,
                c_broken,
                c_bent,
                g_broken,
                g_bent
            );
        }
    }
    if let Some(mut w) = trace {
        w.flush()?;
    }
    Ok(())
}

/// RMS and maximum distance between matching nodes.
fn distance(a: &[Vec3], b: &[Vec3]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (p, q) in a.iter().zip(b) {
        let d = p.distance(*q);
        sum += (d as f64) * (d as f64);
        max = max.max(d);
    }
    (((sum / a.len().max(1) as f64).sqrt()) as f32, max)
}

fn run_bench(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    one: &ExplicitBeams,
    kv: &HashMap<&str, &str>,
) -> anyhow::Result<()> {
    let counts: Vec<usize> = kv
        .get("cars")
        .unwrap_or(&"1,10,40,100,400,1000")
        .split(',')
        .map(|s| s.parse().expect("count"))
        .collect();
    // CPU reference: one vehicle, single thread.
    let mut cpu = one.clone();
    for _ in 0..5 {
        cpu.rebase_origin();
        for _ in 0..33 {
            cpu.step();
        }
    }
    let frames = 30;
    let t = Instant::now();
    for _ in 0..frames {
        cpu.rebase_origin();
        for _ in 0..33 {
            cpu.step();
        }
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
    println!("CPU reference, 1 vehicle, single thread: {cpu_ms:.3} ms per frame (33 steps)");
    println!("vehicles   ms/frame (median)   best   per vehicle (µs)   pipelined ms/frame   1-step frame ms");
    for &n in &counts {
        // Copies side by side, 10 m apart (every vehicle is independent anyway).
        let fleet: Vec<ExplicitBeams> = (0..n)
            .map(|i| {
                let mut v = one.clone();
                for node in &mut v.nodes {
                    node.pos.x += 10.0 * (i % 40) as f32;
                    node.pos.z += 10.0 * (i / 40) as f32;
                }
                v
            })
            .collect();
        let mut gpu = ExplicitBeamsGpu::new(device, queue, &fleet);
        // Warm up (pipeline creation, GPU clocks), then time each frame + wait;
        // the median resists other work on the machine.
        for _ in 0..60 {
            gpu.frame(33);
            gpu.wait();
        }
        let mut times: Vec<f64> = (0..120)
            .map(|_| {
                let t = Instant::now();
                gpu.frame(33);
                gpu.wait();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(|a, b| a.total_cmp(b));
        let ms = times[times.len() / 2];
        let best = times[0];
        // Pipelined, as a game runs: 120 frames submitted back to back, one wait.
        let t = Instant::now();
        for _ in 0..120 {
            gpu.frame(33);
        }
        gpu.wait();
        let piped = t.elapsed().as_secs_f64() * 1e3 / 120.0;
        // Pipelined cost per dispatch for 1, 33 and 132 steps (`slope=1`): the slope
        // is the GPU's cost per step, the intercept the cost per dispatch.
        if kv.get("slope") == Some(&"1") {
            for steps in [1u32, 33, 132] {
                let t = Instant::now();
                for _ in 0..120 {
                    gpu.frame(steps);
                }
                gpu.wait();
                println!(
                    "  {n} vehicles, {steps:3} steps per dispatch: {:.3} ms per dispatch",
                    t.elapsed().as_secs_f64() * 1e3 / 120.0
                );
            }
        }
        // `split=1`: 33 steps as 33 one-step dispatches in one pass, against one
        // 33-step dispatch: the difference is 32 dispatches' overhead.
        if kv.get("split") == Some(&"1") {
            for (steps, dispatches) in [(33u32, 1u32), (1, 33), (1, 99)] {
                let t = Instant::now();
                for _ in 0..60 {
                    gpu.frame_split(steps, dispatches);
                }
                gpu.wait();
                println!(
                    "  {n} vehicles, {dispatches:3} dispatches of {steps:2} steps in one pass: {:.3} ms per frame",
                    t.elapsed().as_secs_f64() * 1e3 / 60.0
                );
            }
        }
        // The fixed cost: a frame of a single step, submit to completion.
        let mut one: Vec<f64> = (0..60)
            .map(|_| {
                let t = Instant::now();
                gpu.frame(1);
                gpu.wait();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        one.sort_by(|a, b| a.total_cmp(b));
        println!(
            "{n:8} {ms:14.3} {best:10.3} {:16.2} {piped:18.3} {:16.3}",
            ms * 1e3 / n as f64,
            one[one.len() / 2]
        );
    }
    Ok(())
}
