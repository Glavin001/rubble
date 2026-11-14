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
    // EV_ETK=<folder>[:<pc>] (measurement): a BeamNG JBeam vehicle (the ETK K-series:
    // __mods/etkk_tex, etkk/kc6.pc) in JBeam mode with its pressure wheels, turned to
    // face +Z (showcase_scene::jbeam_vehicle's steps, on demo/one-world-etk).
    if let Ok(spec) = std::env::var("EV_ETK") {
        let (dir, pc) = spec
            .split_once(':')
            .unwrap_or((spec.as_str(), "etkk/kc6.pc"));
        let dir = std::path::Path::new(dir);
        let list = std::fs::read_to_string(dir.join("files.json")).unwrap();
        let names: Vec<&str> = list.split('"').filter(|s| s.ends_with(".jbeam")).collect();
        let files: Vec<(String, String)> = names
            .iter()
            .map(|f| (f.to_string(), std::fs::read_to_string(dir.join(f)).unwrap()))
            .collect();
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(n, t)| (n.as_str(), t.as_str()))
            .collect();
        let pc = std::fs::read_to_string(dir.join(pc)).ok();
        let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap();
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag: true,
            turbulence: true,
            ..Default::default()
        };
        let (mut sim, _) = ExplicitBeams::from_desc_jbeam_with(
            &desc,
            engine_softbody::explicit_beams::JbeamWheels::Pressure,
            glam::Vec3::ZERO,
            glam::Vec3::ZERO,
            settings,
        )
        .unwrap();
        let f = engine_softbody::pack::forward_axis(&desc);
        let angle = -f.x.atan2(f.z);
        let quarter = angle / std::f32::consts::FRAC_PI_2;
        let k = quarter.round();
        let rot = |p: glam::Vec3| -> glam::Vec3 {
            if (quarter - k).abs() > 1e-6 {
                return glam::Quat::from_rotation_y(angle) * p;
            }
            match (k as i32).rem_euclid(4) {
                0 => p,
                1 => glam::Vec3::new(p.z, p.y, -p.x),
                2 => glam::Vec3::new(-p.x, p.y, -p.z),
                _ => glam::Vec3::new(-p.z, p.y, p.x),
            }
        };
        for n in &mut sim.nodes {
            n.pos = rot(n.pos);
            n.vel = rot(n.vel);
        }
        eprintln!(
            "EV_ETK: {} nodes, {} beams, {} triangles",
            sim.nodes.len(),
            sim.beams.len(),
            sim.triangles.len()
        );
        return sim;
    }
    if std::env::var("EV_PAGE").as_deref() == Ok("1") {
        // The crash town page's car (explicit_vehicles_bench's EV_PAGE).
        let text =
            engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
                minimass: 10.0,
                ..engine_softbody::procedural::CarParams::sedan()
            });
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag: true,
            turbulence: true,
            ..Default::default()
        };
        return ExplicitBeams::from_truck(&text, glam::Vec3::ZERO, glam::Vec3::ZERO, settings)
            .unwrap();
    }
    if std::env::var("EV_RAYCAST").as_deref() == Ok("1") {
        let text =
            engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
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
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .unwrap();
    let mut s = ExplicitBeams::from_truck(
        &text,
        glam::Vec3::ZERO,
        glam::Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    // EV_SURFACE=1: surface contact on (jbeam-semantics-spec.md §38), with its §37 law.
    if std::env::var("EV_SURFACE").as_deref() == Ok("1") {
        s.surface_contact = true;
        s.dissipative_contact = true;
    }
    s
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    // SAFETY: before any thread reads it (the device asks for timestamp queries).
    unsafe { std::env::set_var("RUBBLE_GPU_TIMING", "1") };
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    // EV_TRI_STATS=reach (m, the anti-clip band past r; measurement, no GPU): the
    // self-collision grid walk's work at the rest pose, per triangle its box's cells
    // (more than 64: every candidate) and the candidates in its box.
    if let Some(extra) = std::env::var("EV_TRI_STATS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        let s = sedan();
        let r = s.collision_range + extra;
        let cell = s.grid_cell;
        let (mut every, mut cells, mut inbox) = (0usize, Vec::new(), Vec::new());
        for t in &s.triangles {
            let p = t.nodes.map(|i| s.nodes[i as usize].pos);
            let lo = p[0].min(p[1]).min(p[2]) - glam::Vec3::splat(r);
            let hi = p[0].max(p[1]).max(p[2]) + glam::Vec3::splat(r);
            let span = (hi / cell).floor() - (lo / cell).floor() + glam::Vec3::ONE;
            let c = span.x * span.y * span.z;
            if span.max_element() > 64.0 || c > 64.0 {
                every += 1;
            }
            cells.push(c as usize);
            inbox.push(
                s.candidates
                    .iter()
                    .filter(|&&h| {
                        let q = s.nodes[h as usize].pos;
                        q.cmpge(lo).all() && q.cmple(hi).all()
                    })
                    .count(),
            );
        }
        cells.sort_unstable();
        inbox.sort_unstable();
        let q = |v: &Vec<usize>, f: f32| v[((v.len() - 1) as f32 * f) as usize];
        println!(
            "{} triangles, {} candidates, r {r}, cell {cell}: every-candidate {every}; cells p50 {} p90 {} max {}; in box p50 {} p90 {} max {} sum {}",
            s.triangles.len(), s.candidates.len(), q(&cells, 0.5), q(&cells, 0.9), q(&cells, 1.0),
            q(&inbox, 0.5), q(&inbox, 0.9), q(&inbox, 1.0), inbox.iter().sum::<usize>()
        );
        return Ok(());
    }
    let n = args.first().copied().unwrap_or(64);
    let before = args.get(1).copied().unwrap_or(150);
    let hz: f32 = std::env::var("EV_HZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120.0);
    let mut town = rubble::explicit_scene::crash_town_at(
        GpuContext::Default,
        &sedan(),
        n.div_ceil(4).min(32),
        n,
        15.0,
        8.0,
        30,
        &{
            // EV_NO_CONTACT=1 (measurement): the vehicles couple with nothing.
            let mut def = town_coupling();
            if std::env::var("EV_NO_CONTACT").is_ok() {
                def.mask = 0;
            }
            def
        },
        Some(1.0 / hz),
        Some(32),
    )?;
    let w = &mut town.world;
    let trace = std::env::var("EV_TRACE").is_ok();
    for k in 0..before {
        let t = std::time::Instant::now();
        w.step()?;
        w.poll_blocking()?;
        if trace {
            eprintln!("step {k}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let shapes: Vec<usize> = w
        .explicit_window_impulses()
        .iter()
        .map(|v| v.len())
        .collect();
    let steps = (1.0 / hz / engine_softbody::explicit_beams::STEP).round() as u32;
    let x = w.explicit_vehicles_mut().unwrap();
    let mut s = shapes.clone();
    s.sort_unstable();
    println!("{n} cars, after {before} steps at {hz} Hz; frame of {steps} steps; proxy shapes per car {s:?}");
    let rebased = x.origins().iter().filter(|o| **o != glam::Vec3::ZERO).count();
    println!("  {rebased} of {n} vehicles re-based (origin moved off 0)");
    assert!(x.enable_frame_timing(64));
    let snap = x.snapshot();
    let mut time = |x: &mut engine_webgpu::ExplicitBeamsGpu,
                    label: &str,
                    set: &dyn Fn(&mut engine_webgpu::ExplicitBeamsGpu)| {
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
        x.set_surface_pairs(true);
        x.set_surface_culls(true);
        x.set_coupling_on(true);
        x.set_coupling_cull(true);
        x.set_coupling_workgroup_cache(true);
        x.set_vehicle_contact(true);
        x.set_drive_pass_for_timing(true);
        println!(
            "  {label:40} median {:.3} ms (min {:.3})",
            median(t.clone()),
            t.iter().copied().fold(f64::MAX, f64::min)
        );
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
    if std::env::var("EV_SURFACE").as_deref() == Ok("1") {
        x.frame(steps);
        x.wait();
        let mut st = x.surface_pair_stats();
        st.sort_unstable();
        println!("  surface pairs (listed, hits) per car after a frame: {st:?}");
        x.restore(&snap);
        time(x, "surface pair path off", &|x| x.set_surface_pairs(false));
        time(x, "surface edge culls off", &|x| x.set_surface_culls(false));
        time(x, "surface pairs and culls off", &|x| {
            x.set_surface_pairs(false);
            x.set_surface_culls(false)
        });
        if std::env::var("EV_SURFACE_ONLY").is_ok() {
            return Ok(());
        }
    }
    // EV_ABLATE=m1,m2,...: the same frame with phases skipped (EB_ABLATE masks,
    // ExplicitBeamsGpu::set_ablation; time only), with vehicle contact on and off.
    if let Ok(list) = std::env::var("EV_ABLATE") {
        for m in list.split(',').filter_map(|m| m.trim().parse::<u32>().ok()) {
            x.set_ablation(m);
            time(x, &format!("ablate {m}"), &|_| {});
            time(x, &format!("ablate {m}, vehicle contact off"), &|x| {
                x.set_vehicle_contact(false)
            });
            time(x, &format!("ablate {m}, coupling off"), &|x| {
                x.set_coupling_on(false)
            });
        }
        x.set_ablation(0);
        if std::env::var("EV_ABLATE_ONLY").is_ok() {
            return Ok(());
        }
    }
    time(x, "workgroup copies off", &|x| {
        x.set_coupling_workgroup_cache(false)
    });
    time(x, "workgroup copies off, vehicle contact off", &|x| {
        x.set_coupling_workgroup_cache(false);
        x.set_vehicle_contact(false)
    });
    time(x, "shape cull off", &|x| x.set_coupling_cull(false));
    if std::env::var("EV_RAYCAST").as_deref() == Ok("1") {
        time(x, "raycast wheels' pass skipped (time only)", &|x| {
            x.set_drive_pass_for_timing(false)
        });
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
