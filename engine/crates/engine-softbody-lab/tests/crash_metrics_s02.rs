//! S02 on the CPU reference, as `crash_ref` runs it: every metric of the crash measuring
//! kit (`engine_softbody::crash_reference::metrics`) is there and finite. Run with
//! `--release` (rigging the generated 4x4 and 0.6 s of 2 kHz steps; `--nocapture`
//! prints the values).

use engine_softbody::crash_reference::{self as cr, metrics::Recorder, Contact, CpuRunner, Jitter, Runner, Sample};
use std::collections::BTreeMap;

#[test]
fn crash_metrics_s02_emits_every_metric() {
    let s = cr::get("S02").expect("S02");
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
    let built = s.build(&templates, Jitter::run(0), Contact::Edge).expect("build");
    let mut last = Sample::default();
    let report = cr::run(&s, &templates, &built, 0, &mut CpuRunner::new(&built), &mut |x| last = x.clone()).expect("run");
    let m = &report.metrics;
    for (k, v) in m {
        println!("{k} = {v:.4}");
    }
    let mut want: Vec<String> = Vec::new();
    for n in [
        "mass", "speed_before", "v_along_end", "rebound_ratio", "length0", "crush_peak", "crush_residual", "bent_frac", "broken_frac", "plastic_work_j", "speed_end", "vx_end", "vz_end", "dv_along", "yaw_deg", "yaw_peak_deg", "yaw_abs_peak_deg", "speed_peak", "com_end_past_box0", "pitch_peak_deg", "pitch_end_deg", "roll_peak_deg", "lateral_drift", "decel_peak_g", "decel_mean_g", "pulse_ms", "max_node_speed", "detached_parts", "detached_mass",
    ] {
        want.push(format!("car.{n}"));
    }
    for s in ["front_lower", "front_upper", "cabin", "door", "rear", "wheel", "other"] {
        for side in ["left", "right"] {
            for q in ["work_j", "bent_frac", "broken_frac"] {
                want.push(format!("car.zone.{s}_{side}.{q}"));
            }
        }
    }
    for z in ["front", "front_left", "front_right", "front_upper", "front_lower", "cabin", "rear", "left", "right"] {
        want.push(format!("car.zone.{z}.work_j"));
    }
    for z in ["front", "cabin", "rear"] {
        want.push(format!("car.zone.{z}.bent_frac"));
    }
    for z in ["front", "cabin", "rear", "front_upper"] {
        want.push(format!("car.zone.{z}.work_frac"));
    }
    want.push("car.zone.front_lr_ratio".into());
    want.push("car.zone.lr_ratio".into());
    let gauges: Vec<String> = templates[s.designs[0]].zones.gauges.iter().map(|g| g.name.clone()).chain(["length".into(), "length_left".into(), "length_right".into(), "height".into()]).collect();
    for g in &gauges {
        for q in ["rest", "change_peak", "change_residual", "shrink_frac"] {
            want.push(format!("car.gauge.{g}.{q}"));
        }
    }
    for n in ["finite", "ke_start_j", "ke_end_j", "momentum_x_start", "momentum_z_start", "momentum_x_end", "momentum_z_end", "energy_gain_j", "box_depth_m", "plastic_work_j", "ke_lost_j", "com_speed_end"] {
        want.push(format!("all.{n}"));
    }
    let missing: Vec<&String> = want.iter().filter(|k| !m.get(*k).is_some_and(|v| v.is_finite())).collect();
    assert!(missing.is_empty(), "missing or not finite: {missing:?}");
    // Diagnostics for the report: the beam kinds, the plastic work by the simple rule
    // (spawn threshold × set) against the hardened one, the wheels' spin energy at spawn.
    let f = &built.fleet[0];
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for b in &f.beams {
        *kinds.entry(format!("{:?}/{:?}", b.kind, b.ty)).or_default() += 1;
    }
    println!("beam kinds: {kinds:?}");
    let simple: f64 = f
        .beams
        .iter()
        .zip(&last.beams[0])
        .filter(|(b, _)| b.ty == engine_softbody::explicit_beams::ExplicitType::Deformable && b.k != 0.0)
        .map(|(b, s)| {
            let d = f64::from(s.rest - b.rest0);
            let t = if d < 0.0 { b.tc } else { b.tt };
            if t.is_finite() {
                f64::from(t) * d.abs()
            } else {
                0.0
            }
        })
        .sum();
    println!("plastic work: hardened {:.1} J, threshold x set {simple:.1} J", m["car.plastic_work_j"]);
    if let Some(d) = f.drive.as_deref() {
        let spin: f64 = d.config.wheels.iter().zip(&d.wheels).map(|(w, s)| 0.5 * f64::from(w.inertia) * f64::from(s.omega).powi(2)).sum();
        println!("wheel spin energy at spawn: {spin:.1} J ({} wheels)", d.wheels.len());
    }
    assert_eq!(m["all.finite"], 1.0);
    // An intact car measured end to end at its first sample: the robust length is the
    // world extent along its heading.
    let rest = f.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max) - f.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min);
    assert!((m["car.length0"] - f64::from(rest)).abs() < 1e-5, "{} vs {rest}", m["car.length0"]);
}

/// The ledger counts the raycast wheels' spin: the sedan as spawned at 50 km/h, sampled
/// first with its wheels stopped and then spinning as spawned, rises by exactly Σ ½ I ω².
#[test]
fn crash_metrics_wheel_spin_in_the_ledger() {
    let s = cr::get("S02").expect("S02");
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
    let built = s.build(&templates, Jitter::run(0), Contact::Edge).expect("build");
    let mut x = CpuRunner::new(&built).sample().expect("sample");
    let d = built.fleet[0].drive.as_deref().expect("raycast wheels");
    let want: f64 = d.config.wheels.iter().zip(&x.wheel_spin[0]).map(|(w, o)| 0.5 * f64::from(w.inertia) * f64::from(*o).powi(2)).sum();
    println!("spin energy at spawn {want:.1} J, ω {:?}", x.wheel_spin[0]);
    assert!(want > 100.0, "the wheels spin at spawn");
    let mut stopped = x.clone();
    stopped.wheel_spin[0].iter_mut().for_each(|w| *w = 0.0);
    x.t = 0.004;
    let mut r = Recorder::new(&templates, &built);
    r.push(&stopped);
    r.push(&x);
    let gain = r.finish()["all.energy_gain_j"];
    println!("energy gain {gain:.3} J");
    assert!((gain - want).abs() < 1e-6 * want, "{gain} vs {want}");
}
