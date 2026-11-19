//! Runs the floor kernel on the lab's sedan against the CPU reference.
//!
//!   floor [--frames F] [--lift M] [--speed V] [--core] [--fast] [--cars N] [--time K] [--every E]
//! --core: no self-collision (the reference with its candidates cleared).
//! Checks every frame (bit for bit unless --fast), then times K one-frame dispatches.
use engine_softbody::explicit_beams::JbeamWheels;
use gpu_floor::{car, floor::*, gpu::Gpu};

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}
fn flag(k: &str) -> bool {
    std::env::args().any(|a| a == k)
}

fn main() -> anyhow::Result<()> {
    let frames: usize = arg("--frames").map_or(10, |x| x.parse().unwrap());
    let lift: f32 = arg("--lift").map_or(0.05, |x| x.parse().unwrap());
    let speed: f32 = arg("--speed").map_or(0.0, |x| x.parse().unwrap());
    let cars: usize = arg("--cars").map_or(1, |x| x.parse().unwrap());
    let time: usize = arg("--time").map_or(0, |x| x.parse().unwrap());
    let every: usize = arg("--every").map_or(1, |x| x.parse().unwrap());
    let fast = flag("--fast");
    let t = car::load_template(&car::default_car_dir(), JbeamWheels::Raycast)?;
    let mut cpu = car::place(&t, lift, speed);
    cpu.drive = None;
    if flag("--core") {
        cpu.candidates.clear();
    }
    if let Some(gap) = arg("--wall").map(|x| x.parse::<f32>().unwrap()) {
        // A rigid wall `gap` m ahead of the car's front (+z), 10 m wide, 3 m high, 2 m thick.
        let front = cpu.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        cpu.boxes.push(engine_softbody::explicit_beams::StaticBox { lo: glam::Vec3::new(-5.0, -1.0, front + gap), hi: glam::Vec3::new(5.0, 3.0, front + gap + 2.0) });
    }
    let handoff: usize = arg("--handoff").map_or(0, |x| x.parse().unwrap());
    for _ in 0..handoff * STEPS {
        cpu.step();
    }
    let (l, st) = layout_and_statics(&cpu)?;
    println!("layout: nodes {} records {} lanes {} slots {} add entries {} (max {}) centred entries {} (max {}) groups {}", l.nn, l.nr, LANES, l.rs, l.nadd, l.max_add, l.ncen, l.max_cen, l.ng);
    if flag("--emu") {
        // The kernel's source on the CPU, against the reference (no GPU).
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/gpu-floor/emu");
        let mut extra = String::new();
        if flag("--sums") {
            extra += "#define FAST_SUMS 1\n";
        }
        // --by-step: one substep a run, compared after each (finds the first differing step).
        let by_step = flag("--by-step");
        extra += if flag("--runtime-sizes") || by_step { "#define RUNTIME_SIZES 1\n" } else { "#define RUNTIME_SIZES 0\n" };
        let mut e = Emu::new(l, &st, &extra, &dir)?;
        e.reverse = flag("--reverse");
        e.upload(&state_of(&cpu, &e.l, &st));
        if by_step {
            e.steps = 1;
            for step in 1..=frames * STEPS {
                e.run(1)?;
                cpu.step();
                let g = e.download();
                let (bad, notes) = compare(&cpu, &g);
                if bad != 0 {
                    println!("emu step {step}: differing words {bad}; contacts {} pairs {} builds {}", cpu.last_contacts, g.misc[2], g.misc[5]);
                    for n in notes {
                        println!("  {n}");
                    }
                    anyhow::bail!("emu: first difference at step {step}");
                }
            }
            println!("emu: bit for bit over {} steps", frames * STEPS);
            return Ok(());
        }
        e.upload(&state_of(&cpu, &e.l, &st));
        for fr in 1..=frames {
            e.run(1)?;
            for _ in 0..STEPS {
                cpu.step();
            }
            let g = e.download();
            let (bad, notes) = compare(&cpu, &g);
            let broken = cpu.beams.iter().filter(|b| b.broken).count();
            if bad != 0 || fr % every == 0 || fr == frames {
                println!("emu frame {fr}: differing words {bad}; broken {broken}, pairs {} builds {} most hits {} contacts {}", g.misc[2], g.misc[5], g.misc[6], cpu.last_contacts);
            }
            if bad != 0 {
                for n in notes {
                    println!("  {n}");
                }
                anyhow::bail!("emu: first difference at frame {fr}");
            }
        }
        println!("emu: bit for bit over {frames} frames");
        return Ok(());
    }
    let gpu = Gpu::new()?;
    // --fast: Metal's fast math; --sums: unordered (atomic) beam and centred sums.
    let sums = flag("--sums");
    let mut extra = String::new();
    if sums {
        extra += "#define FAST_SUMS 1\n";
    }
    // --runtime-sizes: the counts as uniforms (a ledger measurement).
    extra += if flag("--runtime-sizes") { "#define RUNTIME_SIZES 1\n" } else { "#define RUNTIME_SIZES 0\n" };
    // --per-step: one dispatch a substep (34 a frame), the state reloaded each time.
    if flag("--per-step") {
        extra += "#define PER_STEP 1\n";
    }
    let extra = extra.as_str();
    let f = Floor::new(gpu, l, &st, cars, fast, extra)?;
    let (mt, w, tg) = f.threads_ok();
    println!("pipeline: max threads {mt}, simd {w}, threadgroup memory {tg} B");
    if flag("--compile-only") {
        return Ok(());
    }
    anyhow::ensure!(mt as usize >= LANES, "the kernel cannot run {LANES} lanes a threadgroup ({mt})");
    let s0 = state_of(&cpu, &f.l, &st);
    for c in 0..cars {
        f.upload(c, &s0);
    }
    let mut first_bad = None;
    let t0 = std::time::Instant::now();
    let mut cpu_s = 0.0;
    for fr in 1..=frames {
        let ms = f.run(1);
        let c0 = std::time::Instant::now();
        for _ in 0..STEPS {
            cpu.step();
        }
        cpu_s += c0.elapsed().as_secs_f64();
        let g = f.download(0);
        let (bad, notes) = compare(&cpu, &g);
        if bad != 0 || fr % every == 0 || fr == frames {
            let (lo, hi) = cpu.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.y), b.max(n.pos.y)));
            let broken = cpu.beams.iter().filter(|b| b.broken).count();
            println!("frame {fr}: gpu {ms:.3} ms, differing words {bad}; y {lo:.3}..{hi:.3}, broken {broken}");
        }
        if bad != 0 && first_bad.is_none() {
            first_bad = Some(fr);
            for n in notes {
                println!("  {n}");
            }
            if !fast && !sums {
                break;
            }
        }
    }
    println!("check: {} ({:.1} s; CPU reference {:.2} ms a frame)", match first_bad { None => "bit for bit".to_string(), Some(f) => format!("first difference at frame {f}") }, t0.elapsed().as_secs_f64(), cpu_s * 1000.0 / frames as f64);
    if time > 0 {
        let mut ms: Vec<f64> = (0..time).map(|_| f.run(1)).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| ms[((ms.len() - 1) as f64 * q) as usize];
        println!("timing ({} cars, one frame a dispatch, {time} dispatches): median {:.4} p95 {:.4} worst {:.4} best {:.4} ms", cars, p(0.5), p(0.95), ms[ms.len() - 1], ms[0]);
    }
    if flag("--phases") {
        // Each phase's cost: a frame with it left out against the whole frame, from the
        // same state each time (saved after the check), medians of `--time` dispatches.
        let save = f.download(0);
        let names = [(0u32, "S0 node pass"), (9, "S0 triangle breaks"), (11, "S0 groups bookkeeping"), (1, "S1 centred slots"), (2, "S2 centred per node"), (3, "S3 beams"), (4, "S4 node beam sums"), (10, "S4 records' groups"), (5, "S5a triangle frames"), (7, "S5c pair tests"), (8, "S6-S8 hits")];
        let k = time.max(20);
        let mut med = |skip: u32| {
            let mut ms: Vec<f64> = (0..k).map(|_| { for c in 0..cars { f.upload(c, &save); } f.run_skip(1, skip) }).collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ms[ms.len() / 2]
        };
        let all = med(0);
        let none = med(u32::MAX);
        println!("phases ({cars} cars; ms a frame, medians of {k}): whole {all:.4}, every phase left out {none:.4}");
        for (bit, name) in names {
            let t = med(1 << bit);
            println!("  {name:24} {:.4} ms (frame without it {t:.4})", all - t);
        }
    }
    Ok(())
}
