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
    let mut f = Floor::new(gpu, l, &st, cars.max(2), fast, extra)?;
    let (mt, w, tg) = f.threads_ok();
    println!("pipeline: max threads {mt}, simd {w}, threadgroup memory {tg} B");
    if flag("--compile-only") {
        return Ok(());
    }
    anyhow::ensure!(mt as usize >= LANES, "the kernel cannot run {LANES} lanes a threadgroup ({mt})");
    // The reference's run, and its own sensitivity: the same run from starts moved by up
    // to a micron a node (three of them). A deviation of the GPU's free run inside that
    // spread is what the reference itself does under a change no one could measure.
    let t0 = std::time::Instant::now();
    let mut refs = vec![cpu.clone()];
    let mut pert: Vec<Vec<Metrics>> = Vec::new();
    for k in 0..3u32 {
        let mut c = cpu.clone();
        let mut sd = 0x9E37_79B9u32.wrapping_mul(k + 1);
        for n in &mut c.nodes {
            let mut r = || {
                sd = sd.wrapping_mul(1664525).wrapping_add(1013904223);
                ((sd >> 8) as f32 / 8388608.0 - 1.0) * 1.0e-6
            };
            n.pos += glam::Vec3::new(r(), r(), r());
        }
        let mut m = Vec::new();
        for _ in 0..frames {
            for _ in 0..STEPS {
                c.step();
            }
            m.push(Metrics::of_cpu(&c));
        }
        pert.push(m);
    }
    let c0 = std::time::Instant::now();
    for _ in 0..frames {
        let mut c = refs.last().unwrap().clone();
        for _ in 0..STEPS {
            c.step();
        }
        refs.push(c);
    }
    let cpu_ms = c0.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!("reference: {frames} frames and three perturbed runs in {:.1} s ({cpu_ms:.2} ms a frame, with clones)", t0.elapsed().as_secs_f64());
    // Car 0 runs free from the start; car 1 starts every frame from the reference's state
    // (the one-frame defect: what a frame of the GPU differs by, before chaos amplifies it).
    f.cars = 2;
    f.upload(0, &state_of(&refs[0], &f.l, &st));
    let first = f.snapshot(0);
    let mut first_bad = None;
    let mut worst = Defect::default();
    let mut verdict_ok = true;
    println!("frame | gpu ms (2 cars) | one-frame defect: words, max pos mm, max vel mm/s | free run vs reference: node max mm, rms mm, com mm, KE %, broken, bent | reference's own spread: node max mm, com mm, KE %, broken, bent");
    for fr in 1..=frames {
        f.upload(1, &state_of(&refs[fr - 1], &f.l, &st));
        let ms = f.run(1);
        let (g0, g1) = (f.download(0), f.download(1));
        let (bad1, notes) = compare(&refs[fr], &g1);
        let d1 = Defect::of(&refs[fr], &g1);
        worst = worst.max(d1);
        let (bad0, _) = compare(&refs[fr], &g0);
        let mr = Metrics::of_cpu(&refs[fr]);
        let mg = Metrics::of_gpu(&refs[fr], &g0);
        let dev = mg.dev(&mr);
        let spread = pert.iter().map(|p| p[fr - 1].dev(&mr)).fold(Dev::default(), Dev::max);
        // Inside three times the reference's own spread (with a floor of 0.01 mm, 0.01 %).
        let ok = !mg.nan && dev.node_max <= 3.0 * spread.node_max + 1e-5 && dev.com <= 3.0 * spread.com + 1e-5 && dev.ke <= 3.0 * spread.ke + 1e-4
            && dev.broken <= 3.0 * spread.broken + 2.0 && dev.bent <= 3.0 * spread.bent + 5.0;
        verdict_ok &= ok;
        if fr % every == 0 || fr == frames || (!ok && fr % every.max(1) == 0) {
            println!(
                "{fr:5} | {ms:7.3} | {bad1:6} {:9.5} {:9.4} | {:9.4} {:8.4} {:8.4} {:7.3} {:3} {:4}{} | {:9.4} {:8.4} {:7.3} {:3} {:4}",
                d1.pos * 1e3, d1.vel * 1e3, dev.node_max * 1e3, dev.node_rms * 1e3, dev.com * 1e3, dev.ke * 100.0, dev.broken, dev.bent,
                if mg.nan { " NaN" } else if bad0 == 0 { " =" } else { "" },
                spread.node_max * 1e3, spread.com * 1e3, spread.ke * 100.0, spread.broken, spread.bent
            );
        }
        if bad1 != 0 && first_bad.is_none() {
            first_bad = Some(fr);
            for n in notes.iter().take(6) {
                println!("  {n}");
            }
        }
    }
    println!(
        "check: one-frame defect {} (worst over the run: pos {:.6} mm, vel {:.5} mm/s); free run {} the reference's own micron spread (x3); reference {cpu_ms:.2} ms a frame on the CPU",
        match first_bad { None => "none: bit for bit every frame".to_string(), Some(f) => format!("first at frame {f}") },
        worst.pos * 1e3, worst.vel * 1e3, if verdict_ok { "INSIDE" } else { "OUTSIDE" }
    );
    // Timing: `cars` cars, all from the free run's last state.
    let last = f.snapshot(0);
    f.cars = cars;
    let reset = |f: &Floor| {
        for c in 0..cars {
            f.restore(c, &last);
        }
    };
    reset(&f);
    println!("state timed: pairs listed {}, list builds so far {}, most hits in a step {}", f.download(0).misc[2], f.download(0).misc[5], f.download(0).misc[6]);
    if time > 0 {
        // The whole run again, timed: every frame of `cars` cars from the start.
        for c in 0..cars {
            f.restore(c, &first);
        }
        let mut ms: Vec<f64> = (0..frames).map(|_| f.run(1)).collect();
        let warm = ms.split_off(ms.len().min(3));
        let mut ms = if warm.is_empty() { ms } else { warm };
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| ms[((ms.len() - 1) as f64 * q) as usize];
        println!("timing, the run ({cars} cars, a frame a dispatch, frames 4..{frames}): median {:.4} p95 {:.4} worst {:.4} best {:.4} ms a frame", p(0.5), p(0.95), ms[ms.len() - 1], ms[0]);
        reset(&f);
        // (Each dispatch from the same state: what a frame costs there, not a drifting run.)
        let mut ms: Vec<f64> = (0..time).map(|_| { reset(&f); f.run(1) }).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| ms[((ms.len() - 1) as f64 * q) as usize];
        println!("timing, the last state ({} cars, {time} dispatches from it): median {:.4} p95 {:.4} worst {:.4} best {:.4} ms", cars, p(0.5), p(0.95), ms[ms.len() - 1], ms[0]);
    }
    if flag("--phases") {
        // Each phase's cost: a frame with it left out against the whole frame, from the
        // same state each time (saved after the check), medians of `--time` dispatches.
        let names = [(0u32, "S0 node pass"), (9, "S0 triangle breaks"), (11, "S0 groups bookkeeping"), (1, "S1 centred slots"), (2, "S2 centred per node"), (3, "S3 beams"), (4, "S4 node beam sums"), (10, "S4 records' groups"), (5, "S5a triangle frames"), (7, "S5c pair tests"), (8, "S6-S8 hits")];
        let k = time.max(20);
        let med = |skip: u32| {
            let mut ms: Vec<f64> = (0..k).map(|_| { reset(&f); f.run_skip(1, skip) }).collect();
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

/// What a run is judged by.
#[derive(Clone, Default)]
struct Metrics {
    pos: Vec<glam::Vec3>,
    com: glam::DVec3,
    ke: f64,
    broken: f64,
    bent: f64,
    nan: bool,
}
#[derive(Clone, Copy, Default)]
struct Dev {
    node_max: f64,
    node_rms: f64,
    com: f64,
    ke: f64,
    broken: f64,
    bent: f64,
}
impl Dev {
    fn max(self, o: Dev) -> Dev {
        Dev { node_max: self.node_max.max(o.node_max), node_rms: self.node_rms.max(o.node_rms), com: self.com.max(o.com), ke: self.ke.max(o.ke), broken: self.broken.max(o.broken), bent: self.bent.max(o.bent) }
    }
}
impl Metrics {
    fn build(pos: Vec<glam::Vec3>, vel: &[glam::Vec3], mass: &[f32], broken: usize, bent: usize) -> Self {
        let (mut m, mut c, mut ke) = (0.0f64, glam::DVec3::ZERO, 0.0f64);
        let mut nan = false;
        for i in 0..pos.len() {
            let w = mass[i] as f64;
            m += w;
            c += pos[i].as_dvec3() * w;
            ke += 0.5 * w * vel[i].as_dvec3().length_squared();
            nan |= !pos[i].is_finite() || !vel[i].is_finite();
        }
        Self { pos, com: c / m, ke, broken: broken as f64, bent: bent as f64, nan }
    }
    fn of_cpu(c: &engine_softbody::explicit_beams::ExplicitBeams) -> Self {
        let pos: Vec<_> = c.nodes.iter().map(|n| n.pos + c.origin).collect();
        let vel: Vec<_> = c.nodes.iter().map(|n| n.vel).collect();
        let mass: Vec<_> = c.nodes.iter().map(|n| n.mass).collect();
        Self::build(pos, &vel, &mass, c.beams.iter().filter(|b| b.broken).count(), c.beams.iter().filter(|b| b.bent()).count())
    }
    fn of_gpu(r: &engine_softbody::explicit_beams::ExplicitBeams, g: &State) -> Self {
        let n = r.nodes.len();
        let pos: Vec<_> = (0..n).map(|i| glam::Vec3::new(g.node[i * 3][0], g.node[i * 3][1], g.node[i * 3][2])).collect();
        let vel: Vec<_> = (0..n).map(|i| glam::Vec3::new(g.node[i * 3 + 1][0], g.node[i * 3 + 1][1], g.node[i * 3 + 1][2])).collect();
        let mass: Vec<_> = r.nodes.iter().map(|n| n.mass).collect();
        let broken = (0..r.beams.len()).filter(|&j| g.rec[j * 3 + 1][3] != 0.0).count();
        let bent = (0..r.beams.len()).filter(|&j| g.rec[j * 3 + 1][3] == 0.0 && (g.rec[j * 3][0] - r.beams[j].rest0).abs() > 1e-3 * r.beams[j].rest0.abs()).count();
        Self::build(pos, &vel, &mass, broken, bent)
    }
    /// This run against the reference `r`: the farthest node and the rms over nodes (m),
    /// the centre of mass (m), the kinetic energy (relative, of at least 1 J), counts.
    fn dev(&self, r: &Metrics) -> Dev {
        let (mut mx, mut sq) = (0.0f64, 0.0f64);
        for (a, b) in self.pos.iter().zip(&r.pos) {
            let d = (*a - *b).length() as f64;
            let d = if d.is_nan() { f64::INFINITY } else { d };
            mx = mx.max(d);
            sq += d * d;
        }
        Dev { node_max: mx, node_rms: (sq / self.pos.len().max(1) as f64).sqrt(), com: (self.com - r.com).length(), ke: (self.ke - r.ke).abs() / r.ke.max(1.0), broken: (self.broken - r.broken).abs(), bent: (self.bent - r.bent).abs() }
    }
}
/// The one-frame defect in physical units.
#[derive(Clone, Copy, Default)]
struct Defect {
    pos: f64,
    vel: f64,
}
impl Defect {
    fn of(r: &engine_softbody::explicit_beams::ExplicitBeams, g: &State) -> Self {
        let mut d = Self::default();
        for (i, n) in r.nodes.iter().enumerate() {
            let p = glam::Vec3::new(g.node[i * 3][0], g.node[i * 3][1], g.node[i * 3][2]);
            let v = glam::Vec3::new(g.node[i * 3 + 1][0], g.node[i * 3 + 1][1], g.node[i * 3 + 1][2]);
            let (dp, dv) = ((p - n.pos).length() as f64, (v - n.vel).length() as f64);
            d.pos = if dp.is_nan() { f64::INFINITY } else { d.pos.max(dp) };
            d.vel = if dv.is_nan() { f64::INFINITY } else { d.vel.max(dv) };
        }
        d
    }
    fn max(self, o: Self) -> Self {
        Self { pos: self.pos.max(o.pos), vel: self.vel.max(o.vel) }
    }
}
