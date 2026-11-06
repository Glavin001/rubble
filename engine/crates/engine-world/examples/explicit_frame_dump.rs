//! Test harness: the explicit vehicles' frames from one crash-town state, compared across
//! builds (one kernel against another, bit for bit). `save` steps the town and writes the
//! vehicles' GPU state to a file; `run` builds the same town, puts that state back, runs
//! frames and writes every state word (`ExplicitBeamsGpu::state_words`) to a file. Run
//! `run` with each build on the same saved state and compare the outputs (`cmp`).
//!
//!   explicit_frame_dump save <cars> <steps before> <state file>
//!   explicit_frame_dump run <cars> <state file> <out file> [frames]
//!   explicit_frame_dump time <cars> <state file> [repeats]   (needs RUBBLE_GPU_TIMING=1)
//!
//! (EV_HZ and EV_RAYCAST as explicit_vehicles_bench; bounded: perl -e 'alarm 600; exec @ARGV'.)

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use rubble::explicit_scene::town_coupling;
use rubble::*;

fn sedan() -> ExplicitBeams {
    if std::env::var("EV_RAYCAST").as_deref() == Ok("1") {
        let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
            minimass: 10.0,
            ..engine_softbody::procedural::CarParams::sedan()
        });
        return ExplicitBeams::from_truck_raycast(&text, None, glam::Vec3::ZERO, 0.0, glam::Vec3::ZERO, ExplicitBeamsSettings::default())
            .unwrap()
            .0;
    }
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .unwrap();
    ExplicitBeams::from_truck(&text, glam::Vec3::ZERO, glam::Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("");
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(16);
    let hz: f32 = std::env::var("EV_HZ").ok().and_then(|v| v.parse().ok()).unwrap_or(120.0);
    let mut town = rubble::explicit_scene::crash_town_at(
        GpuContext::Default,
        &sedan(),
        n.div_ceil(4).min(32),
        n,
        15.0,
        8.0,
        30,
        &town_coupling(),
        Some(1.0 / hz),
        Some(32),
    )?;
    let w = &mut town.world;
    let steps = (1.0 / hz / engine_softbody::explicit_beams::STEP).round() as u32;
    match mode {
        "save" => {
            let before: usize = args[2].parse()?;
            for _ in 0..before {
                w.step()?;
                w.poll_blocking()?;
            }
            let shapes: Vec<usize> = w.explicit_window_impulses().iter().map(|v| v.len()).collect();
            let x = w.explicit_vehicles_mut().unwrap();
            let snap = x.snapshot();
            std::fs::write(&args[3], x.snapshot_to_bytes(&snap))?;
            println!("{n} cars after {before} steps: proxy shapes per car {shapes:?}; saved {}", args[3]);
        }
        "run" => {
            let frames: usize = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(3);
            let x = w.explicit_vehicles_mut().unwrap();
            let snap = x.snapshot_from_bytes(&std::fs::read(&args[2])?);
            x.restore(&snap);
            for _ in 0..frames {
                x.frame(steps);
            }
            x.wait();
            let mut out = String::new();
            for (name, words) in x.state_words() {
                out.push_str(&format!("{name} {}\n", words.len()));
                for c in words.chunks(16) {
                    out.push_str(&c.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" "));
                    out.push('\n');
                }
            }
            std::fs::write(&args[3], out)?;
            println!("{frames} frames of {steps} steps from {}: {}", args[2], args[3]);
        }
        "time" => {
            // GPU ms per frame (its own pass timestamps), `reps` x 3 frames from the state.
            let reps: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(9);
            let x = w.explicit_vehicles_mut().unwrap();
            let snap = x.snapshot_from_bytes(&std::fs::read(&args[2])?);
            assert!(x.enable_frame_timing(64));
            let mut t = Vec::new();
            for _ in 0..reps {
                x.restore(&snap);
                x.request_contact_replan();
                x.frame(steps);
                x.wait();
                t.extend(x.frame_times_ms());
            }
            t.sort_by(f64::total_cmp);
            println!("frame ms median {:.3} min {:.3} max {:.3} (n {})", t[t.len() / 2], t[0], t[t.len() - 1], t.len());
        }
        _ => anyhow::bail!("usage: explicit_frame_dump save <cars> <steps> <file> | run <cars> <file> <out> [frames]"),
    }
    Ok(())
}
