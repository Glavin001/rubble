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

/// The repo sedan on its beam tyres, or (EV_RAYCAST=1) on raycast wheels (the drive), or
/// (EV_PAGE=1) the crash town page's car: the procedural sedan with its soft tyres
/// (`explicitSedanTruck(true)`) and the page's settings (fast inverse square root, drag
/// and turbulence), as `createVehicleTown` builds it.
fn sedan() -> ExplicitBeams {
    if std::env::var("EV_PAGE").as_deref() == Ok("1") {
        let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
            minimass: 10.0,
            ..engine_softbody::procedural::CarParams::sedan()
        });
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag: true,
            turbulence: true,
            ..Default::default()
        };
        return ExplicitBeams::from_truck(&text, glam::Vec3::ZERO, glam::Vec3::ZERO, settings).unwrap();
    }
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    if std::env::var("EV_RAYCAST").as_deref() == Ok("1") {
        // (The repo sedan has no wheels section: the procedural sedan, as the drive's
        // tests.)
        let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
            minimass: 10.0,
            ..engine_softbody::procedural::CarParams::sedan()
        });
        return ExplicitBeams::from_truck_raycast(
            &text,
            None,
            glam::Vec3::ZERO,
            0.0,
            glam::Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .unwrap()
        .0;
    }
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
    // EV_SPAN=1: no timestamp marks; the GPU time is each step's span from its submit to
    // its completion (the bench waits for every step, so the GPU is idle at each submit).
    // Marks (the default) are empty passes whose timestamps Metal may take before the
    // passes encoded ahead of them finish.
    let span = std::env::var("EV_SPAN").as_deref() == Ok("1");
    if !span {
        // SAFETY: set before any thread reads it.
        unsafe { std::env::set_var("RUBBLE_GPU_TIMING", "1") };
    }
    for &n in &counts {
      for vehicles in [n, 0] {
        // EV_MAX_HOUSES: the most houses (default 32, the baseline's; 64 fit the fracture's
        // 2^20 body slots once the town's capacity is sized per house).
        let max_houses: usize = std::env::var("EV_MAX_HOUSES").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let houses = n.div_ceil(4).min(max_houses);
        // EV_NO_CONTACT=1: the vehicles couple with nothing (filter mask 0): the plan and
        // frame without the strikes' destruction.
        let mut def = town_coupling();
        if std::env::var("EV_NO_CONTACT").is_ok() {
            def.mask = 0;
        }
        // EV_NO_VV=1 (measurement): no vehicle–vehicle contact.
        if std::env::var("EV_NO_VV").is_ok() {
            def.vehicle_contact = false;
        }
        // EV_HZ: World steps per second (default the houses' solver's, 60).
        let hz: Option<f32> = std::env::var("EV_HZ").ok().and_then(|v| v.parse().ok());
        let mut town = rubble::explicit_scene::crash_town_at(
            GpuContext::Default,
            &sedan(),
            houses,
            vehicles,
            15.0,
            8.0,
            // The page builds the town without settling the houses first.
            if std::env::var("EV_PAGE").as_deref() == Ok("1") { 0 } else { 30 },
            &def,
            hz.map(|h| 1.0 / h),
            // At most 32 houses (the fracture's chunk capacity); the rest of the
            // fleet meets head on in open lanes.
            Some(max_houses),
        )?;
        let w = &mut town.world;
        // EV_MARGIN=m (measurement): the vehicle contact's frame-bounds margin.
        if let Some(m) = std::env::var("EV_MARGIN").ok().and_then(|v| v.parse::<f32>().ok()) {
            if let Some(x) = w.explicit_vehicles_mut() {
                x.set_contact_margin(m);
            }
        }
        // EV_VMARGIN=v:m (measurement): vehicle v's own frame-bounds margin.
        if let Some((v, m)) = std::env::var("EV_VMARGIN").ok().and_then(|s| {
            let (a, b) = s.split_once(':')?;
            Some((a.parse::<usize>().ok()?, b.parse::<f32>().ok()?))
        }) {
            if let Some(x) = w.explicit_vehicles_mut() {
                x.set_vehicle_contact_margin(v, m);
            }
        }
        // EV_CULL=0 (measurement): the coupling's shape cull off.
        if std::env::var("EV_CULL").as_deref() == Ok("0") {
            if let Some(x) = w.explicit_vehicles_mut() {
                x.set_coupling_cull(false);
            }
        }
        // EV_PHASES=1: the World's own per-segment timestamps (`GpuTiming::Phases`: the
        // rigid step's phases, the coupling's passes, each explicit dispatch by kernel),
        // their mean ms over the run printed below.
        let phases = std::env::var("EV_PHASES").as_deref() == Ok("1");
        if phases {
            w.set_gpu_timing(GpuTiming::Phases)?;
        }
        let mut seg_ms: std::collections::BTreeMap<&'static str, (f64, u32)> = Default::default();
        let dt = w.solver().dt;
        let mut ms = Vec::new();
        let mut wall = Vec::new();
        // EV_SECS: simulated seconds (default 2).
        let secs: f32 = std::env::var("EV_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
        let steps = (secs / dt).round() as u32;
        let mut broken = 0usize;
        for _ in 0..steps {
            let t0 = std::time::Instant::now();
            w.step()?;
            let frames = w.poll_blocking()?;
            let span_ms = w.last_submit_instant().map(|t| t.elapsed().as_secs_f64() * 1e3);
            for f in frames {
                if let Some(t) = &f.gpu_time {
                    let mut per: std::collections::BTreeMap<&'static str, f64> = Default::default();
                    for (n, ms) in &t.segments {
                        *per.entry(n).or_default() += ms;
                    }
                    for (n, ms) in per {
                        let e = seg_ms.entry(n).or_default();
                        e.0 += ms;
                        e.1 += 1;
                    }
                }
                for e in &f.destruction_events {
                    if let DestructionEvent::BondsBroken { bonds, .. } = e {
                        broken += bonds.len();
                    }
                }
            }
            wall.push(t0.elapsed().as_secs_f64() * 1e3);
            if std::env::var("EV_GROUPS").is_ok() && w.step_count() % 20 == 0 {
                let sc = w.step_count();
                if let Some(x) = w.explicit_vehicles_mut() {
                    let mut g = x.contact_group_sizes();
                    g.sort_unstable();
                    eprintln!("step {}: {} groups, sizes {:?}, contact margin {} m, widest own {} m", sc, g.len(), g, x.contact_margin(), (0..x.vehicle_count()).map(|v| x.vehicle_contact_margin(v)).fold(0.0f32, f32::max));
                    for e in x.take_contact_errors() {
                        eprintln!("  error: {e}");
                    }
                    let mut shapes: Vec<usize> = w.explicit_window_impulses().iter().map(|v| v.len()).collect();
                    shapes.sort_unstable();
                    eprintln!("  proxy shapes per vehicle: {shapes:?}");
                }
            }
            if span {
                ms.extend(span_ms);
            } else if let Some(t) = w.last_step_gpu_ms() {
                ms.push(t);
            }
        }
        println!(
            "{vehicles} vehicles, {} houses, World step {:.2} ms: GPU {}; wall (step + poll) {}; {broken} bonds broken; explicit steps {:?}",
            town.houses.len(),
            dt * 1e3,
            stats(ms),
            stats(wall),
            w.explicit_vehicle_steps()
        );
        if phases {
            let mut v: Vec<(&str, f64)> = seg_ms.iter().map(|(n, (ms, k))| (*n, ms / f64::from(*k))).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            let list: Vec<String> = v.iter().take(10).map(|(n, ms)| format!("{n} {ms:.2}")).collect();
            println!("  segments (mean ms per step): {}", list.join(", "));
        }
        drop(town);
      }
        if std::env::var("EV_ALONE").as_deref() == Ok("0") {
            continue;
        }
        println!("  explicit frame alone, wide passes on:  {}", alone(n, true, false));
        println!("  explicit frame alone, wide passes off: {}", alone(n, false, false));
        println!("  explicit frame alone, coupling on:     {}", alone(n, false, true));
    }
    Ok(())
}
