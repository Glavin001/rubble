//! A body against rigid bodies (mechanism 3): the CPU reference's run of a scene, and
//! (later options) the kernel against it.
//!   rigid --scene bricks|rocks [--speed V] [--frames F]
use engine_softbody::explicit_beams::JbeamWheels;
use gpu_floor::{car, floor::*};

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
    if std::env::args().any(|a| a == "--emu") {
        // The kernel's source on the CPU against the reference: the body's every state
        // word, and the proxies' poses, velocities, counts and window impulses.
        let (mut l, mut st) = layout_and_statics(&c)?;
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
    let t0 = std::time::Instant::now();
    for fr in 1..=frames {
        let mut touching = 0u32;
        for _ in 0..34 {
            c.step();
            touching += c.rigid.as_ref().unwrap().bodies.iter().map(|b| b.count).sum::<u32>();
        }
        c.rigid.as_mut().unwrap().begin_window();
        if fr % 5 == 0 || fr == frames {
            let r = c.rigid.as_ref().unwrap();
            let moved = r.bodies.iter().filter(|b| b.vel.length() > 0.01).count();
            let fastest = r.bodies.iter().map(|b| b.vel.length()).fold(0.0f32, f32::max);
            println!("frame {fr}: node-in-shape contacts a step {:.1}; bodies moving {moved} of {}, fastest {fastest:.2} m/s; broken beams {}, self contacts {}", touching as f32 / 34.0, r.bodies.len(), c.beams.iter().filter(|b| b.broken).count(), c.last_contacts);
        }
    }
    println!("{frames} frames in {:.1} s ({:.1} ms a frame)", t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() * 1000.0 / frames as f64);
    Ok(())
}
