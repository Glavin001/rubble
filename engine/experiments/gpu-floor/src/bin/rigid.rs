//! A body against rigid bodies (mechanism 3): the CPU reference's run of a scene, and
//! (later options) the kernel against it.
//!   rigid --scene bricks|rocks|post [--speed V] [--frames F] [--gap M]
//!   --reach (with --emu or --gpu): a frame by the kernel its plan gives it (the contact-free
//!   kernel while everything is out of reach, the whole one after; `reach_plan`);
//!   --reach-eager: the contact-free kernel whenever the body starts a frame clear by the
//!   margin (frames then change kernel part way); --reach-cont: a frame with nothing for
//!   the contact-free kernel by the kernel that takes a frame up (else by the one kernel as
//!   it was); --list N: the run's frames timed one by one; --reach-mix (--cars 2): two
//!   bodies with different kernels in one dispatch against each alone.
use engine_softbody::explicit_beams::JbeamWheels;
use gpu_floor::{car, floor::*, gpu::Gpu};

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let scene = arg("--scene").unwrap_or("bricks".into());
    let speed: f32 = arg("--speed").map_or(11.0, |x| x.parse().unwrap());
    let frames: usize = arg("--frames").map_or(30, |x| x.parse().unwrap());
    let t = car::load_template(&car::default_car_dir(), JbeamWheels::Raycast)?;
    let mut c = car::rigid_scene_at(&t, &scene, speed, arg("--gap").map(|x| x.parse().unwrap()))?;
    let reach = std::env::args().any(|a| a == "--reach");
    let eager = std::env::args().any(|a| a == "--reach-eager");
    // --surface: the reference's surface contact on (the car's triangles, beams and nodes in
    // the band against the shapes, not only the nodes inside them).
    c.surface_contact = std::env::args().any(|a| a == "--surface");
    if c.surface_contact {
        println!("surface contact on: {} segments beside the {} triangles and {} nodes", engine_softbody::explicit_surface::surface_segments(&c).len(), c.triangles.len(), c.nodes.len());
    }
    if std::env::args().any(|a| a == "--emu") {
        // The kernel's source on the CPU against the reference: the body's every state
        // word, and the proxies' poses, velocities, counts and window impulses.
        let (mut l, mut st) = layout_and_statics(&c)?;
        list_args(&mut st);

        make_v2(&c, &mut l, &mut st)?;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/gpu-floor/emu-rigid");
        let by_step = std::env::args().any(|a| a == "--by-step");
        let mut e = Emu::new(l, &st, if by_step { "#define RUNTIME_SIZES 1\n" } else { "#define RUNTIME_SIZES 0\n" }, &dir)?;
        e.reverse = std::env::args().any(|a| a == "--reverse");
        e.upload(&state_of(&c, &e.l, &st));
        let per = if by_step { 1 } else { STEPS };
        e.steps = per;
        anyhow::ensure!(!(reach && by_step), "--reach: a frame a dispatch");
        let mut rk = if reach { Some(e.reach_kernels(&st, "#define RUNTIME_SIZES 0\n")?) } else { None };
        // (Frames by kernel: the contact-free one alone, both, the whole one alone.)
        let (mut by, mut handed) = ([0usize; 3], Vec::new());
        for k in 1..=frames * STEPS / per {
            if let Some((lean, cont)) = rk.as_mut() {
                let (is_lean, done) = e.run_reach(lean, cont, &st, eager)?;
                by[if !is_lean { 2 } else if done as usize == STEPS { 0 } else { 1 }] += 1;
                if is_lean && (done as usize) < STEPS {
                    handed.push(format!("{k}: {done}"));
                }
            } else {
                e.run(1)?;
            }
            c.rigid.as_mut().unwrap().begin_window();
            let mut touching = 0;
            for _ in 0..per {
                c.step();
                touching += c.rigid.as_ref().unwrap().bodies.iter().map(|b| b.count).sum::<u32>();
            }
            let g = e.download();
            let (bad, notes) = compare(&c, &g);
            let (bad_r, notes_r) = compare_rigid(c.rigid.as_ref().unwrap(), &g.rig);
            if bad + bad_r != 0 {
                println!("{} {k}: differing words: body {bad}, proxies {bad_r} (node-in-shape contacts {touching})", if by_step { "step" } else { "frame" });
                for n in notes.iter().chain(&notes_r).take(14) {
                    println!("  {n}");
                }
                anyhow::bail!("emu: first difference at {} {k}", if by_step { "step" } else { "frame" });
            }
            if !by_step && (k % 5 == 0 || k == frames) {
                let r = c.rigid.as_ref().unwrap();
                println!("emu frame {k}: body and proxies bit for bit; contacts in the frame {touching}, bodies moving {}, broken beams {}", r.bodies.iter().filter(|b| b.vel.length() > 0.01).count(), c.beams.iter().filter(|b| b.broken).count());
            }
        }
        if reach {
            println!("frames by kernel: contact-free alone {}, contact-free then whole {}, whole alone {}; handed over at (frame: substeps done) {}", by[0], by[1], by[2], if handed.is_empty() { "none".to_string() } else { handed.join(", ") });
        }
        println!("emu: bit for bit over {frames} frames");
        return Ok(());
    }
    if std::env::args().any(|a| a == "--gpu") {
        // The kernel on the GPU: car 0 checked against the reference every frame, then
        // `--cars N` copies of the scene timed.
        let flag = |k: &str| std::env::args().any(|a| a == k);
        let cars: usize = arg("--cars").map_or(1, |x| x.parse().unwrap());
        let time: usize = arg("--time").map_or(30, |x| x.parse().unwrap());
        let (mut l, mut st) = layout_and_statics(&c)?;
        list_args(&mut st);

        make_v2(&c, &mut l, &mut st)?;
        let mut extra = String::from("#define RUNTIME_SIZES 0\n");
        if flag("--native") {
            extra += "#define NATIVE 1\n";
        }
        // --direct: surface contact's box tests a lane a pair (the whole query), not staged.
        if flag("--direct") {
            extra += "#define SURF_STAGED 0\n";
        }
        // --no-fallback: a staged pair with more than 16 candidates at its least gap sets the
        // status instead of taking the whole query (a smaller kernel).
        if flag("--surf-empty") {
            extra += "#define SURF_EMPTY 1\n";
        }
        if flag("--no-fallback") {
            extra += "#define SURF_FALLBACK 0\n";
        }
        let prof: Option<usize> = arg("--profile").map(|x| x.parse().unwrap());
        if prof.is_some() {
            extra += "#define PER_STEP 1\n";
        }
        let mut f = Floor::new(Gpu::new()?, l, &st, cars, flag("--fast"), &extra)?;
        f.single = prof.is_some();
        let (mt, _, tg) = f.threads_ok();
        println!("pipeline: max threads {mt}, threadgroup memory {tg} B");
        // (--reach-direct: the whole kernel's dispatch after the contact-free one always over
        // every group; --reach-nocont, a measurement: no such dispatch.)
        let opt = ReachOpt { eager, cont_always: flag("--reach-cont"), direct: flag("--reach-direct"), no_cont: flag("--reach-nocont") };
        if reach {
            anyhow::ensure!(prof.is_none(), "--reach: no --profile");
            f.enable_reach(&st, flag("--fast"), &extra)?;
        }
        if flag("--compile-only") {
            return Ok(());
        }
        anyhow::ensure!(mt as usize >= LANES);
        f.upload(0, &state_of(&c, &f.l, &st));
        if let Some(fr) = prof {
            // --profile F: where frame F + 1's time goes (the shapes' tests are inside the
            // node pass: its cost with and without them).
            f.cars = 1;
            let mut groups = vec![("S0 node pass, no shapes", 1u32 << 0), ("rigid: nodes in the union listed", 1 << 13), ("rigid: listed nodes' shapes, rule", 1 << 28), ("rigid: proxies integrate", 1 << 27)];
            groups.extend_from_slice(&GROUPS[1..]);
            if c.surface_contact {
                groups.extend_from_slice(&[("surface: near slots, search", 1 << 21), ("surface: cull, tasks, direct tests", 1 << 22), ("surface: A, an edge's least gap", 1 << 24), ("surface: B, kept candidates", 1 << 25), ("surface: R, a pair's contact", 1 << 26), ("surface: slots, hits applied", 1 << 23)]);
            }
            profile(&f, 1, fr, time.max(9), &groups);
            return Ok(());
        }
        let first = f.snapshot(0);
        let mut first_bad = None;
        let mut status = 0;
        println!("frame | gpu ms | differing words: body, proxies | node-in-shape contacts in the frame | farthest node from the reference mm | fastest proxy m/s{}", if reach { " | kernel (contact-free: its substeps)" } else { "" });
        f.cars = 1;
        let host_only = flag("--reach-host");
        // A frame by the plan or by the one kernel; what ran, for the frame's line.
        // (`plans`: the cars' plans worked out before, for a timed frame; else from the
        // cars' blocks now.)
        type Plans = Vec<([u32; 8], bool)>;
        let frame_with = |f: &Floor, by: &mut [usize; 3], plans: Option<&Plans>| -> (f64, String) {
            if !reach {
                // (--reach-host, a measurement: the plan's host work before the one kernel.)
                if host_only {
                    f.plan_only(&st);
                }
                return (f.run(1), String::new());
            }
            let (ms, lean, done) = match plans {
                Some(p) => f.run_planned(p, &opt),
                None => f.run_reach(&st, &opt),
            };
            by[if lean == 0 { 2 } else if done as usize == STEPS { 0 } else { 1 }] += 1;
            (ms, if lean == 0 { " | whole".to_string() } else if done as usize == STEPS { " | contact-free".to_string() } else { format!(" | contact-free {done}, then whole") })
        };
        let frame = |f: &Floor, by: &mut [usize; 3]| frame_with(f, by, None);
        let mut by = [0usize; 3];
        for fr in 1..=frames {
            let (ms, what) = frame(&f, &mut by);
            c.rigid.as_mut().unwrap().begin_window();
            let mut touching = 0;
            for _ in 0..STEPS {
                c.step();
                touching += c.rigid.as_ref().unwrap().bodies.iter().map(|b| b.count).sum::<u32>();
            }
            let g = f.download(0);
            let (bad, notes) = compare(&c, &g);
            let (bad_r, notes_r) = compare_rigid(c.rigid.as_ref().unwrap(), &g.rig);
            status |= g.misc[1];
            let far = c.nodes.iter().enumerate().map(|(i, n)| (glam::Vec3::new(g.node[i * 3][0], g.node[i * 3][1], g.node[i * 3][2]) - n.pos).length()).fold(0.0f32, |a, b| if b.is_nan() { f32::INFINITY } else { a.max(b) });
            if bad + bad_r != 0 && first_bad.is_none() {
                first_bad = Some(fr);
                for n in notes.iter().chain(&notes_r).take(8) {
                    println!("  {n}");
                }
            }
            if fr % 5 == 0 || fr == frames {
                let fastest = c.rigid.as_ref().unwrap().bodies.iter().map(|b| b.vel.length()).fold(0.0f32, f32::max);
                println!("{fr:5} | {ms:7.3} | {bad:6} {bad_r:5} | {touching:5} | {:9.4} | {fastest:.2}{what}", far * 1e3);
            }
        }
        println!("check: {}; kernel status {status:#x}", match first_bad { None => "body and proxies bit for bit every frame".to_string(), Some(fr) => format!("first differing word at frame {fr}") });
        if reach {
            println!("frames by kernel: contact-free alone {}, contact-free then whole {}, whole alone {}", by[0], by[1], by[2]);
        }
        let q = |ms: &mut Vec<f64>| {
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |x: f64| ms[((ms.len() - 1) as f64 * x) as usize];
            format!("median {:.4} p95 {:.4} worst {:.4} best {:.4}", p(0.5), p(0.95), ms[ms.len() - 1], ms[0])
        };
        f.flush = false;
        f.cars = cars;
        let last = f.snapshot(0);
        for g in 0..cars {
            f.restore(g, &first);
        }
        let mut by = [0usize; 3];
        // (The run's plans frame by frame, kept for the timed repeats.)
        let mut kept: Vec<Plans> = Vec::new();
        let run: Vec<(f64, String)> = (0..frames)
            .map(|_| {
                if reach {
                    kept.push(f.plans(&st, eager));
                }
                frame_with(&f, &mut by, kept.last())
            })
            .collect();
        let mut ms: Vec<f64> = run.iter().map(|x| x.0).collect();
        if reach {
            // The run's frames by what ran (the median of each kind).
            for (kind, name) in [(" | contact-free", "contact-free alone"), (" | contact-free ", "contact-free then whole"), (" | whole", "whole alone")] {
                let mut k: Vec<f64> = run.iter().filter(|x| if kind.ends_with(' ') { x.1.starts_with(kind) } else { x.1 == kind }).map(|x| x.0).collect();
                if !k.is_empty() {
                    k.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    println!("timing, the run's frames by kernel ({cars} cars): {name}: {} frames, median {:.4} ms, worst {:.4}", k.len(), k[k.len() / 2], k[k.len() - 1]);
                }
            }
        }
        let mut ms = ms.split_off(ms.len().min(3));
        println!("timing, the run ({cars} cars, a frame a dispatch, frames 4..{frames}): {} ms a frame", q(&mut ms));
        if let Some(reps) = arg("--list").map(|x| x.parse::<usize>().unwrap()) {
            // --list N: the run N times over (five frames first each time, for the GPU's
            // clock), each frame's median, and what ran (c the contact-free kernel alone,
            // h it and then the whole one, w the whole one alone).
            let mut all: Vec<Vec<f64>> = vec![Vec::new(); frames];
            let mut kinds = String::new();
            for rep in 0..reps {
                for _ in 0..5 {
                    frame(&f, &mut by);
                }
                for g in 0..cars {
                    f.restore(g, &first);
                }
                for (k, slot) in all.iter_mut().enumerate() {
                    let (ms, what) = frame_with(&f, &mut by, kept.get(k));
                    slot.push(ms);
                    if rep == 0 {
                        kinds.push(if !reach || what == " | whole" { 'w' } else if what == " | contact-free" { 'c' } else { 'h' });
                    }
                    let _ = k;
                }
            }
            let med: Vec<f64> = all.iter_mut().map(|v| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }).collect();
            println!("run by frame ({cars} cars, ms, median of {reps}): {}", med.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" "));
            println!("kernel by frame: {kinds}");
            println!("run's mean: {:.4} ms a frame", med.iter().sum::<f64>() / frames as f64);
        }
        if reach && flag("--reach-mix") {
            // Two bodies in one dispatch with different kernels (the first state for one,
            // the last for the other), each against its own frame run alone.
            let words = |f: &Floor, car: usize| -> Vec<u32> {
                let g = f.download(car);
                let mut w: Vec<u32> = Vec::new();
                w.extend(bytemuck::cast_slice::<[f32; 4], u32>(&g.node));
                w.extend(bytemuck::cast_slice::<[f32; 4], u32>(&g.rec));
                w.extend(&g.rig);
                w.extend(&g.misc[..12]);
                w.extend(bytemuck::cast_slice::<_, u32>(&g.acc));
                w.extend(bytemuck::cast_slice::<_, u32>(&g.shift));
                w
            };
            anyhow::ensure!(cars >= 2, "--reach-mix: --cars 2");
            f.flush = true;
            f.cars = 1;
            let mut alone = Vec::new();
            for snap in [&first, &last] {
                f.restore(0, snap);
                let (_, lean, done) = f.run_reach(&st, &opt);
                alone.push((words(&f, 0), lean, done));
            }
            f.cars = 2;
            f.restore(0, &first);
            f.restore(1, &last);
            let (_, lean, _) = f.run_reach(&st, &opt);
            let bad: Vec<usize> = (0..2).map(|c| words(&f, c).iter().zip(&alone[c].0).filter(|(a, b)| a != b).count()).collect();
            println!("two bodies, one dispatch ({lean} of them given the contact-free kernel; alone: the first state's {} with {} substeps there, the last state's {}): differing words {} and {}", if alone[0].1 == 1 { "contact-free" } else { "whole" }, alone[0].2, if alone[1].1 == 1 { "contact-free" } else { "whole" }, bad[0], bad[1]);
            f.flush = false;
            f.cars = cars;
        }
        let mut by = [0usize; 3];
        for g in 0..cars {
            f.restore(g, &last);
        }
        let at_last: Option<Plans> = if reach { Some(f.plans(&st, eager)) } else { None };
        let mut ms: Vec<f64> = (0..time)
            .map(|_| {
                for g in 0..cars {
                    f.restore(g, &last);
                }
                frame_with(&f, &mut by, at_last.as_ref()).0
            })
            .collect();
        println!("timing, the last state ({cars} cars, {time} dispatches from it): {} ms{}", q(&mut ms), if reach { format!(" (contact-free alone {}, then whole {}, whole alone {})", by[0], by[1], by[2]) } else { String::new() });
        if reach {
            return Ok(());
        }
        // Each part's cost at that state: the frame with it left out against the whole.
        let med = |skip: u32| {
            let mut ms: Vec<f64> = (0..time.max(12))
                .map(|_| {
                    for g in 0..cars {
                        f.restore(g, &last);
                    }
                    f.run_skip(1, skip)
                })
                .collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ms[ms.len() / 2]
        };
        let all = med(0);
        println!("phases ({cars} cars; ms a frame at the last state): whole {all:.4}");
        for (bit, name) in [(13u32, "rigid shapes and proxies"), (0, "S0 node pass"), (1, "S1 centred scalars"), (2, "C1+C2 centred gather"), (3, "S3 beams"), (4, "S4 force gather"), (5, "S4 pair rows"), (8, "hit passes"), (21, "surface pass (and its box)")] {
            println!("  {name:26} {:.4} ms", all - med(1 << bit));
        }
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    for fr in 1..=frames {
        let (mut touching, mut surf, mut quer) = (0u32, 0u32, 0u32);
        for _ in 0..34 {
            c.step();
            touching += c.rigid.as_ref().unwrap().bodies.iter().map(|b| b.count).sum::<u32>();
            surf += c.surface_last;
            quer += c.surface_queries;
        }
        c.rigid.as_mut().unwrap().begin_window();
        if fr % 5 == 0 || fr == frames {
            let r = c.rigid.as_ref().unwrap();
            let moved = r.bodies.iter().filter(|b| b.vel.length() > 0.01).count();
            let fastest = r.bodies.iter().map(|b| b.vel.length()).fold(0.0f32, f32::max);
            let m: f32 = c.nodes.iter().map(|n| n.mass).sum();
            let vz: f32 = c.nodes.iter().map(|n| n.mass * n.vel.z).sum::<f32>() / m;
            println!("frame {fr}: node-in-shape contacts a step {:.1}; surface hits a step {:.1}, narrow tests a step {:.0}; bodies moving {moved} of {}, fastest {fastest:.2} m/s (body 0 at z {:.3}, vz {:.2}); car vz {vz:.2}; broken beams {}, self contacts {}", touching as f32 / 34.0, surf as f32 / 34.0, quer as f32 / 34.0, r.bodies.len(), r.bodies[0].pos.z, r.bodies[0].vel.z, c.beams.iter().filter(|b| b.broken).count(), c.last_contacts);
        }
    }
    println!("{frames} frames in {:.1} s ({:.1} ms a frame)", t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() * 1000.0 / frames as f64);
    Ok(())
}
