//! A body against rigid bodies (mechanism 3): the CPU reference's run of a scene, and
//! (later options) the kernel against it.
//!   rigid --scene bricks|rocks [--speed V] [--frames F]
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
    let mut c = car::rigid_scene(&t, &scene, speed)?;
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
        for k in 1..=frames * STEPS / per {
            e.run(1)?;
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
        let prof: Option<usize> = arg("--profile").map(|x| x.parse().unwrap());
        if prof.is_some() {
            extra += "#define PER_STEP 1\n";
        }
        let mut f = Floor::new(Gpu::new()?, l, &st, cars, flag("--fast"), &extra)?;
        f.single = prof.is_some();
        let (mt, _, tg) = f.threads_ok();
        println!("pipeline: max threads {mt}, threadgroup memory {tg} B");
        if flag("--compile-only") {
            return Ok(());
        }
        anyhow::ensure!(mt as usize >= LANES);
        f.upload(0, &state_of(&c, &f.l, &st));
        if let Some(fr) = prof {
            // --profile F: where frame F + 1's time goes (the shapes' tests are inside the
            // node pass: its cost with and without them).
            f.cars = 1;
            let mut groups = vec![("S0 node pass, no shapes", 1u32 << 0), ("rigid shapes and proxies", 1 << 13)];
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
        println!("frame | gpu ms | differing words: body, proxies | node-in-shape contacts in the frame | farthest node from the reference mm | fastest proxy m/s");
        f.cars = 1;
        for fr in 1..=frames {
            let ms = f.run(1);
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
                println!("{fr:5} | {ms:7.3} | {bad:6} {bad_r:5} | {touching:5} | {:9.4} | {fastest:.2}", far * 1e3);
            }
        }
        println!("check: {}; kernel status {status:#x}", match first_bad { None => "body and proxies bit for bit every frame".to_string(), Some(fr) => format!("first differing word at frame {fr}") });
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
        let mut ms: Vec<f64> = (0..frames).map(|_| f.run(1)).collect();
        let mut ms = ms.split_off(ms.len().min(3));
        println!("timing, the run ({cars} cars, a frame a dispatch, frames 4..{frames}): {} ms a frame", q(&mut ms));
        let mut ms: Vec<f64> = (0..time)
            .map(|_| {
                for g in 0..cars {
                    f.restore(g, &last);
                }
                f.run(1)
            })
            .collect();
        println!("timing, the last state ({cars} cars, {time} dispatches from it): {} ms", q(&mut ms));
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
        for (bit, name) in [(13u32, "rigid shapes and proxies"), (0, "S0 node pass"), (1, "S1 centred scalars"), (2, "C1+C2 centred gather"), (3, "S3 beams"), (4, "S4 force gather"), (5, "S4 pair rows"), (8, "hit passes")] {
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
