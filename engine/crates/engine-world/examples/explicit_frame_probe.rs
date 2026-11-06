//! Measurement: the explicit vehicles' frame on one crash-town state, timed with its own
//! pass timestamps, under switches (each from the same snapshot): the coupling's shape
//! cull on and off, vehicle contact off, the coupling off. Diagnostic only.
//!
//! cargo run --release -p engine-world --example explicit_frame_probe -- <cars> <steps before>
//! (EV_HZ, EV_RAYCAST as explicit_vehicles_bench; bounded: perl -e 'alarm 600; exec @ARGV').

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    // SAFETY: before any thread reads it (the device asks for timestamp queries).
    unsafe { std::env::set_var("RUBBLE_GPU_TIMING", "1") };
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n = args.first().copied().unwrap_or(64);
    let before = args.get(1).copied().unwrap_or(150);
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
    for _ in 0..before {
        w.step()?;
        w.poll_blocking()?;
    }
    let shapes: Vec<usize> = w.explicit_window_impulses().iter().map(|v| v.len()).collect();
    let steps = (1.0 / hz / engine_softbody::explicit_beams::STEP).round() as u32;
    let x = w.explicit_vehicles_mut().unwrap();
    let mut s = shapes.clone();
    s.sort_unstable();
    println!("{n} cars, after {before} steps at {hz} Hz; frame of {steps} steps; proxy shapes per car {s:?}");
    assert!(x.enable_frame_timing(64));
    let snap = x.snapshot();
    let mut time = |x: &mut engine_webgpu::ExplicitBeamsGpu, label: &str, set: &dyn Fn(&mut engine_webgpu::ExplicitBeamsGpu)| {
        let mut t = Vec::new();
        for _ in 0..9 {
            x.restore(&snap);
            set(x);
            x.request_contact_replan();
            x.frame(steps);
            x.wait();
            t.extend(x.frame_times_ms());
        }
        x.restore(&snap);
        x.set_coupling_on(true);
        x.set_coupling_cull(true);
        x.set_coupling_workgroup_cache(true);
        x.set_vehicle_contact(true);
        x.set_drive_pass_for_timing(true);
        println!("  {label:40} median {:.3} ms (min {:.3})", median(t.clone()), t.iter().copied().fold(f64::MAX, f64::min));
        if std::env::var("EV_PROFILE").is_ok() {
            x.restore(&snap);
            set(x);
            x.request_contact_replan();
            let p = x.profile_frame(steps);
            x.wait();
            println!("    per dispatch: {p:?}");
            x.restore(&snap);
            x.set_coupling_on(true);
            x.set_coupling_cull(true);
            x.set_coupling_workgroup_cache(true);
            x.set_vehicle_contact(true);
        }
    };
    time(x, "as shipped", &|_| {});
    // EV_ABLATE=m1,m2,...: the same frame with phases skipped (EB_ABLATE masks,
    // ExplicitBeamsGpu::set_ablation; time only), with vehicle contact on and off.
    if let Ok(list) = std::env::var("EV_ABLATE") {
        for m in list.split(',').filter_map(|m| m.trim().parse::<u32>().ok()) {
            x.set_ablation(m);
            time(x, &format!("ablate {m}"), &|_| {});
            time(x, &format!("ablate {m}, vehicle contact off"), &|x| x.set_vehicle_contact(false));
            time(x, &format!("ablate {m}, coupling off"), &|x| x.set_coupling_on(false));
        }
        x.set_ablation(0);
        if std::env::var("EV_ABLATE_ONLY").is_ok() {
            return Ok(());
        }
    }
    time(x, "workgroup copies off", &|x| x.set_coupling_workgroup_cache(false));
    time(x, "workgroup copies off, vehicle contact off", &|x| {
        x.set_coupling_workgroup_cache(false);
        x.set_vehicle_contact(false)
    });
    time(x, "shape cull off", &|x| x.set_coupling_cull(false));
    if std::env::var("EV_RAYCAST").as_deref() == Ok("1") {
        time(x, "raycast wheels' pass skipped (time only)", &|x| x.set_drive_pass_for_timing(false));
    }
    time(x, "vehicle contact off", &|x| x.set_vehicle_contact(false));
    time(x, "vehicle contact and shape cull off", &|x| {
        x.set_vehicle_contact(false);
        x.set_coupling_cull(false)
    });
    time(x, "coupling off", &|x| x.set_coupling_on(false));
    time(x, "coupling and vehicle contact off", &|x| {
        x.set_coupling_on(false);
        x.set_vehicle_contact(false)
    });
    Ok(())
}
