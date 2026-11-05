//! Frame times of explicit soft-body vehicles in the World (`explicit_scene::crash_town`):
//! N repo sedans at 15-18 m/s into prefractured cuboid houses past crates, one World step
//! per 1/60 s (the houses' threshold solver), every step's GPU time from the step's own
//! timestamps (RUBBLE_GPU_TIMING marks around the whole submission: emit, rigid step,
//! plan, explicit frame, loads, fracture). Also the explicit frame alone (no World), with
//! the wide passes on and off and with the coupling built and on (no proxies), for the
//! coupling's own cost.
//!
//! cargo run --release -p engine-world --example explicit_vehicles_bench -- 1 16 64
//! (one at a time on a quiet GPU; bounded: run it under `perl -e 'alarm 900; exec @ARGV'`).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{CouplingCapacity, ExplicitBeamsGpu};
use rubble::explicit_scene::{crash_town, place_vehicle, town_coupling};
use rubble::*;

fn sedan() -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    ExplicitBeams::from_truck(&text, glam::Vec3::ZERO, glam::Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap()
}

fn stats(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "no samples".into();
    }
    v.sort_by(f64::total_cmp);
    let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!("median {:.2} ms, p95 {:.2}, p99 {:.2}, max {:.2} ({} steps)", p(0.5), p(0.95), p(0.99), p(1.0), v.len())
}

/// The explicit frame alone: `n` sedans side by side, 34-step frames (1/60 s), GPU ms.
fn alone(n: usize, wide: bool, coupled: bool) -> String {
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    let t = sedan();
    let fleet: Vec<ExplicitBeams> = (0..n)
        .map(|i| place_vehicle(&t, glam::Vec3::new(i as f32 * 4.0, 0.0, 0.0), 0, 15.0, 0.0))
        .collect();
    let cap = CouplingCapacity { bodies: 64, shapes: 255, planes: 4096 };
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, &fleet, coupled.then_some(cap));
    if coupled {
        gpu.set_coupling_on(true);
    } else {
        gpu.set_wide(wide);
    }
    let timed = gpu.enable_frame_timing(256);
    for _ in 0..20 {
        gpu.frame(34);
    }
    gpu.wait();
    let _ = gpu.frame_times_ms();
    for _ in 0..60 {
        gpu.frame(34);
    }
    gpu.wait();
    if !timed {
        return "no timestamps".into();
    }
    format!("{} (wide {})", stats(gpu.frame_times_ms()), gpu.is_wide())
}

fn main() -> anyhow::Result<()> {
    let counts: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let counts = if counts.is_empty() { vec![1, 16, 64] } else { counts };
    // SAFETY: set before any thread reads it.
    unsafe { std::env::set_var("RUBBLE_GPU_TIMING", "1") };
    for &n in &counts {
      for vehicles in [n, 0] {
        let houses = n.div_ceil(4);
        // EV_NO_CONTACT=1: the vehicles couple with nothing (filter mask 0): the plan and
        // frame without the strikes' destruction.
        let mut def = town_coupling();
        if std::env::var("EV_NO_CONTACT").is_ok() {
            def.mask = 0;
        }
        let mut town = crash_town(GpuContext::Default, &sedan(), houses, vehicles, 15.0, 8.0, 30, &def)?;
        let w = &mut town.world;
        let dt = w.solver().dt;
        let mut ms = Vec::new();
        let steps = (2.0 / dt).round() as u32;
        let mut broken = 0usize;
        for _ in 0..steps {
            w.step()?;
            for f in w.poll_blocking()? {
                for e in &f.destruction_events {
                    if let DestructionEvent::BondsBroken { bonds, .. } = e {
                        broken += bonds.len();
                    }
                }
            }
            if let Some(t) = w.last_step_gpu_ms() {
                ms.push(t);
            }
        }
        println!(
            "{vehicles} vehicles, {} houses, World step {:.2} ms: {}; {broken} bonds broken; explicit steps {:?}",
            town.houses.len(),
            dt * 1e3,
            stats(ms),
            w.explicit_vehicle_steps()
        );
        drop(town);
      }
        println!("  explicit frame alone, wide passes on:  {}", alone(n, true, false));
        println!("  explicit frame alone, wide passes off: {}", alone(n, false, false));
        println!("  explicit frame alone, coupling on:     {}", alone(n, false, true));
    }
    Ok(())
}
