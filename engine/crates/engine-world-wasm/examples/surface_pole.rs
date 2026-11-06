//! Pole impacts with and without surface contact (jbeam-semantics-spec.md §38): the
//! crash lab's own scenes (`ExplicitSim`'s `sidepole` / `pole`, the page's spawn), the
//! vehicle stepped on the CPU reference or the GPU. Reports, per run, how deep the pole
//! got into the vehicle: the deepest node, beam point and triangle point inside the
//! pole box, its p99 over every inside element and frame; the peak force (the
//! vehicle's momentum change along the approach per frame); the crush at the pole
//! (struck side against far side in the pole's lane, at its deepest and at the end);
//! and the wall-clock cost per step.
//!
//! ```text
//! cargo run --release -p engine-world-wasm --example surface_pole -- \
//!     --car sedan1|etkk [--scene side|oblique|front] [--kmh 32] [--at 0] \
//!     [--surface 0|1] [--gpu] [--secs 0.6] [--etk <vehicles dir>]
//! ```
//! The ETK (an imported BeamNG mod, never committed) is read from `--etk` or
//! `RUBBLE_ETK` (the `vehicles` directory holding `etkk/`), as the page's
//! `mod=/__mods/etkk&pc=etkk/kr6.pc` loads it.

use engine_softbody::explicit_beams::{ExplicitBeams, FrameClock, StaticBox, STEP};
use engine_world_wasm::ExplicitSim;
use glam::Vec3;
use std::time::Instant;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn flag(k: &str) -> bool {
    std::env::args().any(|x| x == k)
}

fn car(name: &str) -> serde_json::Value {
    match name {
        "sedan1" | "sedan" => {
            use engine_softbody::authoring::{build, style, BuildParams, Resolution};
            let b = build(
                &style(name).expect("style"),
                &BuildParams {
                    resolution: Resolution::parse("medium").unwrap(),
                    strength: 1.0,
                    paint: None,
                },
            )
            .expect("build");
            serde_json::json!({ "truck": b.truck })
        }
        "etkk" => {
            let dir = arg("--etk")
                .or_else(|| std::env::var("RUBBLE_ETK").ok())
                .expect("--etk <vehicles dir> or RUBBLE_ETK");
            let dir = std::path::Path::new(&dir);
            let files = engine_softbody::jbeam::read_vehicle_dir(dir, "etkk").expect("etkk");
            let pc = std::fs::read_to_string(dir.join("etkk/kr6.pc")).expect("kr6.pc");
            serde_json::json!({ "jbeam": files, "pc": pc })
        }
        _ => panic!("--car sedan1|etkk"),
    }
}

/// Depth of p inside the box (0 outside): its distance to the nearest face.
fn depth(b: &StaticBox, p: Vec3) -> f32 {
    let d = (p - b.lo).min(b.hi - p);
    d.min_element().max(0.0)
}

/// Deepest point of segment a–b in the box: its run inside (slabs), sampled at 33 points.
fn seg_depth(bx: &StaticBox, a: Vec3, b: Vec3) -> f32 {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    let d = b - a;
    for i in 0..3 {
        if d[i].abs() < 1e-12 {
            if a[i] <= bx.lo[i] || a[i] >= bx.hi[i] {
                return 0.0;
            }
            continue;
        }
        let (ta, tb) = ((bx.lo[i] - a[i]) / d[i], (bx.hi[i] - a[i]) / d[i]);
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
    }
    if !(t0 < t1) {
        return 0.0;
    }
    (0..=32)
        .map(|k| depth(bx, a + d * (t0 + (t1 - t0) * k as f32 / 32.0)))
        .fold(0.0, f32::max)
}

/// Deepest point of triangle a b c in the box, sampled on a 16-step barycentric grid.
fn tri_depth(bx: &StaticBox, x: [Vec3; 3]) -> f32 {
    let lo = x[0].min(x[1]).min(x[2]);
    let hi = x[0].max(x[1]).max(x[2]);
    if lo.cmpge(bx.hi).any() || hi.cmple(bx.lo).any() {
        return 0.0;
    }
    let n = 16;
    let mut m = 0.0f32;
    for i in 0..=n {
        for j in 0..=n - i {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            m = m.max(depth(bx, x[0] * (1.0 - u - v) + x[1] * u + x[2] * v));
        }
    }
    m
}

struct Depths {
    node: f32,
    beam: f32,
    tri: f32,
    samples: Vec<f32>,
}

fn depths(
    s: &ExplicitBeams,
    p: &[Vec3],
    bx: &StaticBox,
    live_beam: &dyn Fn(usize) -> bool,
) -> Depths {
    let mut out = Depths {
        node: 0.0,
        beam: 0.0,
        tri: 0.0,
        samples: Vec::new(),
    };
    for (i, &x) in p.iter().enumerate() {
        let _ = i;
        let d = depth(bx, x);
        if d > 0.0 {
            out.node = out.node.max(d);
            out.samples.push(d);
        }
    }
    for (j, b) in s.beams.iter().enumerate() {
        if !live_beam(j) {
            continue;
        }
        let d = seg_depth(bx, p[b.a as usize], p[b.b as usize]);
        if d > 0.0 {
            out.beam = out.beam.max(d);
            out.samples.push(d);
        }
    }
    for (t, tri) in s.triangles.iter().enumerate() {
        if s.tri_dead.get(t).copied().unwrap_or(false) || s.tri_anti_only(t) {
            continue;
        }
        let d = tri_depth(bx, tri.nodes.map(|i| p[i as usize]));
        if d > 0.0 {
            out.tri = out.tri.max(d);
            out.samples.push(d);
        }
    }
    out
}

fn main() {
    let car_name = arg("--car").unwrap_or_else(|| "sedan1".into());
    let scene = arg("--scene").unwrap_or_else(|| "side".into());
    let kmh: f32 = arg("--kmh").map(|x| x.parse().unwrap()).unwrap_or(32.0);
    let at: f32 = arg("--at").map(|x| x.parse().unwrap()).unwrap_or(0.0);
    let surface = arg("--surface").map(|x| x == "1").unwrap_or(false);
    let gpu = flag("--gpu");
    let secs: f32 = arg("--secs").map(|x| x.parse().unwrap()).unwrap_or(0.6);
    let mut spec = serde_json::json!({
        "vehicles": [car(&car_name)],
        "speed": kmh / 3.6,
        "timing": false,
        "surfaceContact": surface,
        "runUp": 0.3,
    });
    match scene.as_str() {
        "side" => {
            spec["scene"] = "sidepole".into();
            spec["at"] = at.into();
        }
        "oblique" => {
            spec["scene"] = "sidepole".into();
            spec["at"] = at.into();
            spec["angle"] = 75.0.into();
        }
        "front" => {
            spec["scene"] = "pole".into();
            spec["offset"] = at.into();
        }
        _ => panic!("--scene side|oblique|front"),
    }
    let (_, device, queue) = engine_webgpu::default_device().expect("a GPU device");
    // --arena N: N copies into walls and poles (the arena scene), GPU wall clock per
    // frame only (the cost at scale).
    if let Some(n) = arg("--arena") {
        spec["scene"] = "arena".into();
        spec["count"] = n.parse::<f64>().unwrap().into();
        let mut sim = ExplicitSim::new_native(&device, &queue, &spec).expect("spawn");
        let mut clock = FrameClock::default();
        let (mut cost, mut steps) = (0.0f64, 0u64);
        for f in 0..(secs * 60.0) as usize {
            let k = clock.next();
            let t = Instant::now();
            sim.frame_native(k).unwrap();
            sim.gpu_solver().wait();
            if f >= 10 {
                cost += t.elapsed().as_secs_f64();
                steps += u64::from(k);
            }
        }
        println!(
            "{car_name} arena x{n} surface {}: {:.3} ms per 60 fps frame, {:.1} µs/step for the fleet",
            surface as u8,
            cost / steps as f64 * 1e3 * 100.0 / 3.0,
            cost / steps as f64 * 1e6
        );
        return;
    }
    let mut sim = ExplicitSim::new_native(&device, &queue, &spec).expect("spawn");
    let mut cpu = sim.spawned(0).clone();
    let bx = *cpu.boxes.last().expect("the pole box");
    let p0 = cpu.positions();
    let mass: Vec<f32> = cpu.nodes.iter().map(|n| n.mass).collect();
    let v0 = cpu.nodes[0].vel;
    let dir = v0.normalize();
    let lateral = Vec3::Y.cross(dir);
    let pc = 0.5 * (bx.lo + bx.hi);
    // The pole's lane: skin nodes within 15 cm of its axis across the approach; the
    // struck side (nearer the pole than the lane's middle) and the far side.
    let skin: std::collections::BTreeSet<u32> =
        cpu.triangles.iter().flat_map(|t| t.nodes).collect();
    let lane: Vec<usize> = skin
        .iter()
        .map(|&i| i as usize)
        .filter(|&i| ((p0[i] - pc).dot(lateral)).abs() < 0.15)
        .collect();
    let mid = lane.iter().map(|&i| p0[i].dot(dir)).sum::<f32>() / lane.len().max(1) as f32;
    let (struck, far): (Vec<usize>, Vec<usize>) = lane.iter().partition(|&&i| p0[i].dot(dir) > mid);
    let far_c =
        |p: &[Vec3]| far.iter().map(|&i| p[i].dot(dir)).sum::<f32>() / far.len().max(1) as f32;
    let f0 = far_c(&p0);
    let crush = |p: &[Vec3]| {
        struck
            .iter()
            .map(|&i| (p0[i].dot(dir) - f0) - (p[i].dot(dir) - far_c(p)))
            .fold(0.0f32, f32::max)
    };
    let momentum = |vel: &[Vec3]| {
        vel.iter()
            .zip(&mass)
            .map(|(v, m)| v.dot(dir) * m)
            .sum::<f32>()
    };
    let total_mass: f32 = mass.iter().sum();
    let frames = (secs * 60.0).round() as usize;
    let mut clock = FrameClock::default();
    let mut steps = 0u64;
    let mut cost = 0.0f64;
    let (mut dn, mut db, mut dt, mut all) = (0.0f32, 0.0f32, 0.0f32, Vec::new());
    let mut peak = 0.0f32;
    let mut crush_max = 0.0f32;
    let mut p_prev = momentum(&cpu.nodes.iter().map(|n| n.vel).collect::<Vec<_>>());
    let mut surf_hits = 0u64;
    let mut last_p = p0.clone();
    for _ in 0..frames {
        let (p, vel, live): (Vec<Vec3>, Vec<Vec3>, Vec<bool>) = if gpu {
            let n = clock.next();
            let t = Instant::now();
            sim.frame_native(n).unwrap();
            sim.gpu_solver().wait();
            cost += t.elapsed().as_secs_f64();
            steps += u64::from(n);
            let p = sim.gpu_solver().positions().remove(0);
            let v = sim.gpu_solver().velocities().remove(0);
            (p, v, vec![true; cpu.beams.len()])
        } else {
            let t = Instant::now();
            cpu.check_wings();
            cpu.rebase_origin();
            let n = cpu.clock.next();
            for _ in 0..n {
                cpu.step();
                surf_hits += u64::from(cpu.surface_last);
            }
            cost += t.elapsed().as_secs_f64();
            steps += u64::from(n);
            (
                cpu.positions(),
                cpu.nodes.iter().map(|n| n.vel).collect(),
                cpu.beams.iter().map(|b| !b.broken).collect(),
            )
        };
        let d = depths(&cpu, &p, &bx, &|j| live[j]);
        dn = dn.max(d.node);
        db = db.max(d.beam);
        dt = dt.max(d.tri);
        all.extend(d.samples);
        let pm = momentum(&vel);
        peak = peak.max(-(pm - p_prev) * 60.0);
        p_prev = pm;
        crush_max = crush_max.max(crush(&p));
        last_p = p;
    }
    all.sort_by(f32::total_cmp);
    let p99 = all
        .get(((all.len() as f32 * 0.99) as usize).min(all.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0);
    let ke_end: f32 = 0.5 * total_mass * (p_prev / total_mass).powi(2);
    println!(
        "{car_name} {scene} {kmh} km/h at {at} surface {} {}: depth max node {:.1} beam {:.1} tri {:.1} mm, p99 {:.1} mm ({} samples); peak {:.0} kN; crush max {:.0} end {:.0} mm; {:.1} µs/step ({} steps, {} surface hits); approach KE left {:.1} kJ",
        surface as u8,
        if gpu { "gpu" } else { "cpu" },
        dn * 1e3,
        db * 1e3,
        dt * 1e3,
        p99 * 1e3,
        all.len(),
        peak / 1e3,
        crush_max * 1e3,
        crush(&last_p) * 1e3,
        cost / steps as f64 * 1e6,
        steps,
        surf_hits,
        ke_end / 1e3,
    );
    let _ = STEP;
}
