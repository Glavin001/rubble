//! The solo-car core kernel's go/no-go (perf doc §8.43): the lab's sedan with contact
//! off (no self-collision, no World coupling, no wheels, no ground in reach), from the
//! lab's own states, stepped by the CPU reference, today's per-vehicle frame kernel,
//! today's wide passes and the solo core kernel; every node bit for bit with the CPU
//! after every frame and every state word of the GPU paths against each other; then
//! each path's GPU time a frame.
//!
//! `cargo run --profile fast -p engine-world --example solo_core_check -- [--frames N]
//!   [--time N] [--states spawn,m2.5,m6] [--paths frame,wide,solo] [--dump DIR]`
//! (under `scripts/with-gpu-lock.py`, `RUBBLE_GPU_LOCK=0`). `--dump DIR`: each state's
//! buffers and the two kernels' WGSL for the Chrome replay (`tools/solo-car/replay.mjs`).
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_webgpu::ExplicitBeamsGpu;
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::{lab_car, LabOptions};
use serde_json::json;

fn steps() -> u32 {
    arg("--steps").and_then(|x| x.parse().ok()).unwrap_or(34)
}

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn template() -> anyhow::Result<ExplicitBeams> {
    let dir = arg("--car").unwrap_or_else(|| {
        format!(
            "{}/Development/rubble-artifacts/sedan-live/mods/sedan_new",
            std::env::var("HOME").unwrap()
        )
    });
    let dir = std::path::Path::new(&dir);
    let list: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
    let mut texts = Vec::new();
    let mut pc = None;
    for f in &list {
        if f.ends_with(".jbeam") {
            texts.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
        } else if f.ends_with(".pc") {
            pc = Some(std::fs::read_to_string(dir.join(f))?);
        }
    }
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    // The lab's settings (vehicle_lab.rs `car`): drag and turbulence on.
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: true,
        turbulence: true,
        ..Default::default()
    };
    Ok(rubble::showcase_scene::jbeam_vehicle_on(
        &files,
        pc.as_deref(),
        settings,
        JbeamWheels::Raycast,
    )?
    .0)
}

/// Contact off: no self-collision candidates, no World proxies or static shapes, no
/// wheels or drivetrain, no surface contact, the contact-only rules off; the ground
/// far below (the mountain's own -1e5).
fn strip(mut s: ExplicitBeams) -> ExplicitBeams {
    s.candidates.clear();
    s.rigid = None;
    s.shapes.clear();
    s.boxes.clear();
    s.drive = None;
    s.surface_contact = false;
    s.anti_clip = false;
    s.two_body_contact = false;
    s.own_travel_reach = false;
    s.dissipative_contact = false;
    // (No contact pending from the state's last step either.)
    s.contact_shift.clear();
    s.contact_acc.clear();
    s.slide_hold.clear();
    // (Debris retirement is the host's, frame by frame: none here.)
    s.retired.clear();
    if arg("--centred").as_deref() == Some("0") {
        s.settings.centred_damping = false;
    }
    s.ground = -1.0e5;
    s
}

fn states(
    template: &ExplicitBeams,
    want: &[String],
) -> anyhow::Result<Vec<(String, ExplicitBeams)>> {
    let script = json!({ "scene": "mountain", "cars": 1, "events": [] });
    let opts = LabOptions::from_json(&script);
    let mut out = Vec::new();
    for w in want {
        if w == "spawn" {
            out.push((w.clone(), strip(lab_car(template, &opts, None))));
            continue;
        }
        let t: f64 = w.trim_start_matches('m').parse()?;
        let mut lab = CpuLab::new(template, opts.clone())?;
        lab.bench.set_script(&script)?;
        for _ in 0..(t * 60.0).round() as u64 {
            lab.step()?;
        }
        out.push((w.clone(), strip(lab.sim.clone())));
    }
    Ok(out)
}

fn same_nodes(cpu: &ExplicitBeams, gpu: &ExplicitBeamsGpu) -> Option<String> {
    let (gp, gv) = (&gpu.positions()[0], &gpu.velocities()[0]);
    let cp = cpu.positions();
    for i in 0..cpu.nodes.len() {
        let a = cp[i].to_array().map(f32::to_bits) == gp[i].to_array().map(f32::to_bits);
        let b = cpu.nodes[i].vel.to_array().map(f32::to_bits) == gv[i].to_array().map(f32::to_bits);
        if !(a && b) {
            return Some(format!(
                "node {i}: CPU {:?} {:?} GPU {:?} {:?}",
                cp[i], cpu.nodes[i].vel, gp[i], gv[i]
            ));
        }
    }
    let gb = &gpu.beams()[0];
    for (j, (c, g)) in cpu.beams.iter().zip(gb).enumerate() {
        if c.broken != g.broken || c.rest.to_bits() != g.rest.to_bits() {
            return Some(format!(
                "beam {j}: CPU broken {} rest {} GPU broken {} rest {}",
                c.broken, c.rest, g.broken, g.rest
            ));
        }
    }
    None
}

fn make(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    s: &ExplicitBeams,
    path: &str,
) -> anyhow::Result<ExplicitBeamsGpu> {
    let mut g = ExplicitBeamsGpu::new(device, queue, std::slice::from_ref(s));
    g.set_vehicle_contact(false);
    match path {
        "frame" => {
            // Measurement: another build of the frame module (`--frame-wgsl FILE`).
            if let Some(f) = arg("--frame-wgsl") {
                g.debug_set_frame_wgsl(&engine_webgpu::uniformize_workgroup_lanes(
                    &std::fs::read_to_string(f)?,
                ));
            }
        }
        "wide" => anyhow::ensure!(g.set_wide(true), "no wide passes for this car"),
        "solo" => g.set_solo_core(true).map_err(anyhow::Error::msg)?,
        // The solo core kernel on even frames, today's frame kernel on odd ones.
        "switch" => {}
        p => anyhow::bail!("unknown path {p}"),
    }
    Ok(g)
}

/// The Chrome replay's bundle: each binding's initial bytes (the upload's), the frame's
/// parameters (copy 0, as the frame kernels read them), the solo table, both kernels'
/// WGSL (today's frame module and the car's solo core module) and a manifest.
fn dump(dir: &std::path::Path, s: &ExplicitBeams, frame: &ExplicitBeamsGpu) -> anyhow::Result<()> {
    use engine_exec::explicit_plan as plan;
    std::fs::create_dir_all(dir)?;
    let up = plan::ExplicitUpload::new(std::slice::from_ref(s), None);
    let table = up.solo_core_table().map_err(anyhow::Error::msg)?;
    let mut planner = plan::FramePlanner::new(&up.layout);
    let params = planner.write_params(steps());
    std::fs::write(dir.join("b0.bin"), &params[..plan::PARAM_STRIDE as usize])?;
    for b in 1..=8u32 {
        std::fs::write(dir.join(format!("b{b}.bin")), up.bytes(b))?;
    }
    std::fs::write(
        dir.join("table.bin"),
        bytemuck::cast_slice::<u32, u8>(&table.words),
    )?;
    std::fs::write(dir.join("solo.wgsl"), engine_webgpu::solo_core_wgsl(&table))?;
    std::fs::write(dir.join("frame.wgsl"), frame.debug_frame_wgsl())?;
    let manifest = json!({ "steps": steps(), "nodes": table.nodes, "hot": table.hot });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

fn median(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
    (q(0.5), q(0.95), *v.last().unwrap())
}

fn main() -> anyhow::Result<()> {
    let frames: u32 = arg("--frames").and_then(|x| x.parse().ok()).unwrap_or(20);
    let timed: u32 = arg("--time").and_then(|x| x.parse().ok()).unwrap_or(60);
    let want: Vec<String> = arg("--states")
        .unwrap_or_else(|| "spawn,m2.5,m6".into())
        .split(',')
        .map(String::from)
        .collect();
    let paths: Vec<String> = arg("--paths")
        .unwrap_or_else(|| "frame,wide,solo".into())
        .split(',')
        .map(String::from)
        .collect();
    let t = template()?;
    println!("settings {:?}", t.settings);
    let list = states(&t, &want)?;
    let (_, device, queue) = engine_webgpu::default_device_with(true)?;
    let mut bad = 0;
    for (name, s) in &list {
        // Bit for bit: every path against the CPU each frame, every state word of each
        // path against the first's.
        let mut cpu = s.clone();
        let mut gpus = Vec::new();
        for p in &paths {
            gpus.push((p.as_str(), make(&device, &queue, s, p)?));
        }
        let mut first_bad: Vec<Option<String>> = vec![None; gpus.len()];
        for f in 0..frames {
            for _ in 0..steps() {
                cpu.step();
            }
            for (p, g) in gpus.iter_mut() {
                if *p == "switch" {
                    g.set_solo_core(f % 2 == 0).map_err(anyhow::Error::msg)?;
                }
                g.frame(steps());
            }
            for (k, (p, g)) in gpus.iter().enumerate() {
                if first_bad[k].is_some() {
                    continue;
                }
                if let Some(d) = same_nodes(&cpu, g) {
                    first_bad[k] = Some(format!("{p} vs CPU at frame {f}: {d}"));
                    continue;
                }
                if k > 0 {
                    let diff = engine_webgpu::state_differences(
                        &gpus[0].1.state_words(),
                        &g.state_words(),
                        6,
                    );
                    if !diff.is_empty() {
                        first_bad[k] = Some(format!(
                            "{p} vs {} state words at frame {f}: {}",
                            gpus[0].0,
                            diff.join("; ")
                        ));
                    }
                }
            }
        }
        for (k, (p, g)) in gpus.iter().enumerate() {
            g.kernel_status().map_err(anyhow::Error::msg)?;
            match &first_bad[k] {
                None => println!(
                    "{name}: {p}: {frames} frames bit for bit with the CPU{}",
                    if k > 0 { " and every state word" } else { "" }
                ),
                Some(d) => {
                    bad += 1;
                    println!("{name}: {p}: DIFFERS {d}");
                }
            }
        }
        // Time: each path from the state, `timed` frames, GPU time a frame.
        for p in &paths {
            let mut g = make(&device, &queue, s, p)?;
            g.frame(steps());
            g.wait();
            g.enable_frame_timing(timed);
            for _ in 0..timed {
                g.frame(steps());
            }
            g.wait();
            let ms = g.frame_times_ms();
            if ms.is_empty() {
                println!("{name}: {p}: no timestamps");
                continue;
            }
            let (m, p95, w) = median(ms);
            println!(
                "{name}: {p}: GPU {m:.3} ms a frame median, p95 {p95:.3}, worst {w:.3} ({timed} frames of {} steps)",
                steps()
            );
        }
        if let Some(dir) = arg("--dump") {
            let dir = std::path::Path::new(&dir).join(name);
            dump(&dir, s, &make(&device, &queue, s, "frame")?)?;
            println!("{name}: replay bundle in {}", dir.display());
        }
    }
    anyhow::ensure!(bad == 0, "{bad} paths differ");
    Ok(())
}
