//! Two bodies in contact (mechanism 2): the CPU reference's run of a pair scene, and
//! (later options) the pair kernel against it.
//!   pair --scene headon|side|rear|apart [--speed V] [--gap G] [--frames F]
use engine_softbody::explicit_beams::JbeamWheels;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use gpu_floor::{car, floor::*, gpu::Gpu};

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let scene = arg("--scene").unwrap_or("headon".into());
    let speed: f32 = arg("--speed").map_or(14.0, |x| x.parse().unwrap());
    let gap: f32 = arg("--gap").map_or(0.3, |x| x.parse().unwrap());
    let frames: usize = arg("--frames").map_or(30, |x| x.parse().unwrap());
    let t = car::load_template(&car::default_car_dir(), JbeamWheels::Raycast)?;
    let cars = car::pair_scene(&t, &scene, speed, gap)?;
    let mut w = ExplicitBeamsWorld::new(cars);
    println!("final guard {}, throttle {}", w.final_guard, w.throttle);
    if std::env::args().any(|a| a == "--emu") {
        // The pair kernel's source on the CPU (two bodies' workgroups and their contact
        // group in lockstep) against the reference, every state word of both bodies.
        let (mut l, mut st) = layout_and_statics(&w.vehicles[0])?;
        list_args(&mut st);

        make_v2(&w.vehicles[0], &mut l, &mut st)?;
        st.pair = 2;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/gpu-floor/emu-pair");
        let by_step = std::env::args().any(|a| a == "--by-step");
        let xa = arg("--xatom").map_or(String::new(), |m| format!("#define XATOM {m}\n"));
        let mut e = Emu::new(l, &st, &format!("{}{xa}", if by_step { "#define RUNTIME_SIZES 1\n" } else { "#define RUNTIME_SIZES 0\n" }), &dir)?;
        e.reverse = std::env::args().any(|a| a == "--reverse");
        for c in 0..2 {
            let s = state_of(&w.vehicles[c], &e.l, &st);
            e.upload_at(c, &s);
        }
        let per = if by_step { 1 } else { STEPS };
        e.steps = per;
        for k in 1..=frames * STEPS / per {
            e.run(1)?;
            let mut hits = 0;
            for _ in 0..per {
                w.step();
                hits += w.last.hits;
            }
            let mut bad_all = 0;
            for c in 0..2 {
                let g = e.download_at(c);
                let (bad, notes) = compare(&w.vehicles[c], &g);
                if bad != 0 {
                    println!("{} {k}: body {c}: differing words {bad}", if by_step { "step" } else { "frame" });
                    for n in notes {
                        println!("  {n}");
                    }
                }
                bad_all += bad;
            }
            if bad_all != 0 {
                anyhow::bail!("emu: first difference at {} {k} (body-body hits in it {hits})", if by_step { "step" } else { "frame" });
            }
            if !by_step && (k % 5 == 0 || k == frames) {
                let broken: Vec<usize> = w.vehicles.iter().map(|v| v.beams.iter().filter(|b| b.broken).count()).collect();
                println!("emu frame {k}: both bodies bit for bit; body-body hits in the frame {hits}, broken {broken:?}");
            }
        }
        println!("emu: bit for bit over {frames} frames");
        return Ok(());
    }
    if std::env::args().any(|a| a == "--gpu") {
        // The pair kernel on the GPU: `--pairs N` pairs in one dispatch (three workgroups a
        // pair), pair 0 checked against the reference every frame, then timed.
        let flag = |k: &str| std::env::args().any(|a| a == k);
        let pairs: usize = arg("--pairs").map_or(1, |x| x.parse().unwrap());
        let time: usize = arg("--time").map_or(30, |x| x.parse().unwrap());
        let fast = flag("--fast");
        let (mut l, mut st) = layout_and_statics(&w.vehicles[0])?;
        list_args(&mut st);

        make_v2(&w.vehicles[0], &mut l, &mut st)?;
        st.pair = 2;
        let mut extra = String::from("#define RUNTIME_SIZES 0\n");
        if flag("--native") {
            extra += "#define NATIVE 1\n";
        }
        if let Some(m) = arg("--regmode") {
            extra += &format!("#define REGMODE {m}\n");
        }
        if let Some(m) = arg("--xatom") {
            extra += &format!("#define XATOM {m}\n");
        }
        let by_step = flag("--by-step");
        let prof: Option<usize> = arg("--profile").map(|x| x.parse().unwrap());
        if by_step || prof.is_some() {
            extra += "#define PER_STEP 1\n";
        }
        let mut f = Floor::new(Gpu::new()?, l, &st, 2 * pairs, fast, &extra)?;
        f.single = by_step || prof.is_some();
        let (mt, _, tg) = f.threads_ok();
        println!("pipeline: max threads {mt}, threadgroup memory {tg} B; {pairs} pair(s), {} workgroups", 2 * pairs);
        if flag("--compile-only") {
            return Ok(());
        }
        anyhow::ensure!(mt as usize >= LANES);
        for p in 0..pairs {
            for c in 0..2 {
                f.upload(2 * p + c, &state_of(&w.vehicles[c], &f.l, &st));
            }
        }
        if let Some(fr) = prof {
            // --profile F: where frame F + 1's time goes (floor.rs profile).
            let mut groups = GROUPS.to_vec();
            groups.push(("pair: boxes, publish, partner's nodes, triangle cull", 1 << 12));
            groups.push(("pair: triangles x clusters", 1 << 17));
            groups.push(("pair: cluster points' box test", 1 << 18));
            groups.push(("pair: candidates' rule", 1 << 19));
            groups.push(("pair: hit counts, owners' turns", 1 << 20));
            profile(&f, 2 * pairs, fr, time.max(9), &groups);
            return Ok(());
        }
        if by_step {
            // One substep a dispatch, compared after each: the first differing step, its
            // words, and how the kernel's contact pass counted against the reference's.
            for step in 1..=frames * STEPS {
                f.run(1);
                w.step();
                let g: Vec<State> = (0..2).map(|c| f.download(c)).collect();
                let bad: Vec<usize> = (0..2).map(|c| compare(&w.vehicles[c], &g[c]).0).collect();
                if bad[0] + bad[1] != 0 || step % 34 == 0 {
                    println!("check step {step}: differing words {bad:?}; reference hits {} candidates {}; status {:#x}", w.last.hits, w.last.candidates, g[0].misc[1] | g[1].misc[1]);
                }
                if bad[0] + bad[1] != 0 {
                    for c in 0..2 {
                        for n in compare(&w.vehicles[c], &g[c]).1.iter().take(8) {
                            println!("check   body {c}: {n}");
                        }
                    }
                    println!("check: first differing word at step {step}");
                    return Ok(());
                }
            }
            println!("check: both bodies bit for bit every step ({} steps)", frames * STEPS);
            return Ok(());
        }
        let first: Vec<Vec<Vec<u8>>> = (0..2 * pairs).map(|g| f.snapshot(g)).collect();
        // The reference's own sensitivity: the same run from a start moved by a micron.
        let mut wp = {
            let mut cars: Vec<_> = w.vehicles.clone();
            let mut sd = 0x9E37_79B9u32;
            for c in &mut cars {
                for n in &mut c.nodes {
                    let mut r = || {
                        sd = sd.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((sd >> 8) as f32 / 8388608.0 - 1.0) * 1.0e-6
                    };
                    n.pos += glam::Vec3::new(r(), r(), r());
                }
            }
            ExplicitBeamsWorld::new(cars)
        };
        let dev = |a: &ExplicitBeamsWorld, pos: &dyn Fn(usize, usize) -> glam::Vec3| -> f64 {
            let mut m = 0.0f64;
            for c in 0..2 {
                for (i, n) in a.vehicles[c].nodes.iter().enumerate() {
                    let d = (pos(c, i) - n.pos).length() as f64;
                    m = m.max(if d.is_nan() { f64::INFINITY } else { d });
                }
            }
            m
        };
        let mut first_bad = None;
        let mut status = 0u32;
        let mut ms_run = Vec::new();
        println!("frame | gpu ms | differing words (both bodies) | body-body hits in the frame | farthest node from the reference, mm | the reference's own from a micron, mm | broken (reference)");
        for fr in 1..=frames {
            ms_run.push(f.run(1));
            let mut hits = 0;
            for _ in 0..STEPS {
                w.step();
                wp.step();
                hits += w.last.hits;
            }
            let g: Vec<State> = (0..2).map(|c| f.download(c)).collect();
            let bad: usize = (0..2).map(|c| compare(&w.vehicles[c], &g[c]).0).sum();
            status |= g[0].misc[1] | g[1].misc[1];
            let d = dev(&w, &|c, i| glam::Vec3::new(g[c].node[i * 3][0], g[c].node[i * 3][1], g[c].node[i * 3][2]));
            let dr = dev(&w, &|c, i| wp.vehicles[c].nodes[i].pos);
            if bad != 0 && first_bad.is_none() {
                first_bad = Some(fr);
                for c in 0..2 {
                    for n in compare(&w.vehicles[c], &g[c]).1.iter().take(4) {
                        println!("check   body {c}: {n}");
                    }
                }
            }
            if fr % 5 == 0 || fr == frames {
                let broken: Vec<usize> = w.vehicles.iter().map(|v| v.beams.iter().filter(|b| b.broken).count()).collect();
                println!("{fr:5} | {:7.3} | {bad:6} | {hits:5} | {:9.4} | {:9.4} | {broken:?}", ms_run[fr - 1], d * 1e3, dr * 1e3);
            }
        }
        println!("check: {}; kernel status {status:#x} (0: no list overflowed, no cross-group barrier missed)", match first_bad { None => "both bodies bit for bit every frame".to_string(), Some(fr) => format!("first differing word at frame {fr}") });
        let q = |ms: &mut Vec<f64>| {
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |x: f64| ms[((ms.len() - 1) as f64 * x) as usize];
            format!("median {:.4} p95 {:.4} worst {:.4} best {:.4}", p(0.5), p(0.95), ms[ms.len() - 1], ms[0])
        };
        f.flush = false;
        let last: Vec<Vec<Vec<u8>>> = (0..2).map(|g| f.snapshot(g)).collect();
        // The run again from the start, every pair the same, timed.
        for g in 0..2 * pairs {
            f.restore(g, &first[g]);
        }
        let mut ms: Vec<f64> = (0..frames).map(|_| f.run(1)).collect();
        let mut ms = ms.split_off(ms.len().min(3));
        println!("timing, the run ({pairs} pair(s), a frame a dispatch, frames 4..{frames}): {} ms a frame", q(&mut ms));
        let mut ms: Vec<f64> = (0..time)
            .map(|_| {
                for g in 0..2 * pairs {
                    f.restore(g, &last[g % 2]);
                }
                f.run(1)
            })
            .collect();
        println!("timing, the last state ({pairs} pair(s), {time} dispatches from it): {} ms", q(&mut ms));
        // Each part's cost at that state: the frame with it left out against the whole.
        let med = |skip: u32| {
            let mut ms: Vec<f64> = (0..time.max(12))
                .map(|_| {
                    for g in 0..2 * pairs {
                        f.restore(g, &last[g % 2]);
                    }
                    f.run_skip(1, skip)
                })
                .collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ms[ms.len() / 2]
        };
        let all = med(0);
        println!("phases ({pairs} pair(s); ms a frame at the last state): whole {all:.4}");
        for (bit, name) in [(12u32, "contact pass (boxes meet)"), (0, "S0 node pass"), (1, "S1 centred scalars"), (2, "C1+C2 centred gather"), (3, "S3 beams"), (4, "S4 force gather"), (5, "S4 pair rows"), (8, "hit passes")] {
            println!("  {name:26} {:.4} ms", all - med(1 << bit));
        }
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    for fr in 1..=frames {
        let (mut hits, mut cands, mut tris, mut most) = (0u64, 0u64, 0u64, 0u32);
        for _ in 0..34 {
            w.step();
            hits += w.last.hits as u64;
            cands += w.last.candidates as u64;
            tris += w.last.triangles_tested as u64;
            most = most.max(w.last.hits as u32);
        }
        if fr % 5 == 0 || fr == frames {
            let broken: Vec<usize> = w.vehicles.iter().map(|v| v.beams.iter().filter(|b| b.broken).count()).collect();
            let selfc: Vec<u32> = w.vehicles.iter().map(|v| v.last_contacts).collect();
            println!("frame {fr}: body-body hits a step {:.1} (most {most}), candidates a step {:.0}, triangles tested a step {:.0}; self contacts {selfc:?}, broken {broken:?}, partner pairs {}", hits as f64 / 34.0, cands as f64 / 34.0, tris as f64 / 34.0, w.last.partner_pairs);
        }
    }
    println!("{frames} frames in {:.1} s ({:.1} ms a frame); first hit at step {:?}", t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() * 1000.0 / frames as f64, w.first_hit_step);
    Ok(())
}
