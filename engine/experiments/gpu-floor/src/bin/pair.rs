//! Two bodies in contact (mechanism 2): the CPU reference's run of a pair scene, and
//! (later options) the pair kernel against it.
//!   pair --scene headon|side|rear|apart [--speed V] [--gap G] [--frames F]
use engine_softbody::explicit_beams::JbeamWheels;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use gpu_floor::{car, floor::*};

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
        make_v2(&w.vehicles[0], &mut l, &mut st)?;
        st.pair = 2;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/gpu-floor/emu-pair");
        let by_step = std::env::args().any(|a| a == "--by-step");
        let mut e = Emu::new(l, &st, if by_step { "#define RUNTIME_SIZES 1\n" } else { "#define RUNTIME_SIZES 0\n" }, &dir)?;
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
