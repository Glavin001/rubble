//! The step change on the reference itself: each catalogue scene run by stress-ref at its
//! own substep and at the true stable step (its Courant safety scaled so its stress step
//! equals the power-iteration bound at safety 0.9), compared probe by probe and by
//! outcome; and at half its own substep, for the reference's refinement spread.
//!
//!   stress-gpu-stepcheck [name-prefix ...] [--duration S]

use stress_gpu::stability::true_stable_dt;
use stress_ref::observation::Observation;
use stress_ref::scene::Scene;
use stress_ref::solver::ReferenceSolver;
use stress_ref::world::World;

fn probe_error(a: &Observation, b: &Observation) -> f64 {
    let mut worst = 0.0f64;
    for (name, r) in &a.probes {
        let Some(g) = b.probes.get(name) else { continue };
        let range = r.v.iter().filter(|x| x.is_finite()).fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
        // Compare at the reference's sample times (interpolating the other series).
        for (t, v) in r.t.iter().zip(&r.v) {
            if let Some(w) = g.at(*t) {
                if v.is_finite() && w.is_finite() {
                    worst = worst.max((v - w).abs() / range);
                } else if v.is_finite() != w.is_finite() {
                    worst = f64::INFINITY;
                }
            }
        }
    }
    worst
}

fn outcome(o: &Observation) -> String {
    let v = |k: &str| o.values.get(k).copied().unwrap_or(f64::NAN);
    format!("broken {:.0} frags {:.0} maxidx {:.3} diss {:.4e}", v("broken_bonds"), v("fragments"), v("max_failure_index"), v("bond_dissipation"))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut duration = 0.1;
    if let Some(i) = args.iter().position(|a| a == "--duration") {
        duration = args[i + 1].parse().expect("--duration S");
        args.drain(i..i + 2);
    }
    for mut scene in stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()) {
        if !args.is_empty() && !args.iter().any(|p| scene.name.starts_with(p.as_str())) {
            continue;
        }
        scene.sim.duration = scene.sim.duration.min(duration);
        let m = ReferenceSolver::new(&scene);
        let (reference_dt, truth) = (m.stable_dt(), true_stable_dt(&m, 1000, 0.9));
        let ratio = truth / reference_dt;
        let run = |s: &Scene| -> (Observation, f64) {
            let t = std::time::Instant::now();
            let o = World::new(s).run();
            (o, t.elapsed().as_secs_f64())
        };
        let (base, t_base) = run(&scene);
        let mut fine = scene.clone();
        fine.sim.scale_safety(0.5);
        let (half, _) = run(&fine);
        let mut coarse: Scene = scene.clone();
        coarse.sim.courant_safety *= ratio;
        let (big, t_big) = run(&coarse);
        let stable = big.probes.values().all(|p| p.v.iter().all(|x| x.is_finite() && x.abs() < 1e12));
        println!(
            "{:<26} step x{ratio:5.1}: probes {:.1e} (refinement spread {:.1e}); {}; time {t_base:.1} s -> {t_big:.1} s",
            scene.name,
            probe_error(&base, &big),
            probe_error(&base, &half),
            if stable { "stable" } else { "UNSTABLE" }
        );
        println!("{:<26}   own step:  {}", "", outcome(&base));
        println!("{:<26}   half step: {}", "", outcome(&half));
        println!("{:<26}   true step: {}", "", outcome(&big));
    }
}
