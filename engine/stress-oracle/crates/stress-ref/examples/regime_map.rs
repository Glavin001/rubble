//! Regime map: per scene, the three times that decide which solver tier can carry it
//! (the real-time analysis, section 8.2), from the oracle's own model.
//! - `t_c`, the loading time: an impactor's Hertz contact time
//!   `2.87 (m^2 / (R E*^2 v))^(1/5)` (a box impactor: `R` its smallest half extent),
//!   a load's rise or pulse time, a removal's duration (0: sudden);
//! - `T_w`, the wave transit across the structure's largest extent at `sqrt(E / rho)`, and
//!   (impacts) `T_w,member` across its extent along the impact (a wall's thickness);
//! - `T_1`, the structure's fundamental period (inverse iteration, intact, its largest
//!   anchored cluster).
//!
//! The suggested tier follows the analysis's rules with `>>` read as a factor of 10:
//! `t_c >= 10 T_1` quasi-static (1), `t_c >= T_w` structural dynamics (2), else an
//! explicit impact window (3). These are hypotheses for the oracle to calibrate, not
//! verdicts.
//!
//! ```text
//! cargo run --release -p stress-ref --example regime_map [pack.json ...]
//! ```

use stress_ref::scene::{EventDesc, ImpactorShape, LoadDesc, Scene, TimeFunction};
use stress_ref::solver::ReferenceSolver;

fn rise(f: &TimeFunction) -> Option<f64> {
    match f {
        TimeFunction::Constant { .. } => None,
        TimeFunction::Ramp { t0, t1, .. } => Some(t1 - t0),
        TimeFunction::HalfSine { duration, .. } => Some(*duration),
        TimeFunction::Friedlander { duration, .. } => Some(*duration),
        TimeFunction::Table { points } => points.windows(2).filter(|w| w[0][1] != w[1][1]).map(|w| w[1][0] - w[0][0]).reduce(f64::min),
    }
}

fn loading_time(s: &Scene) -> Option<(f64, &'static str)> {
    let mut best: Option<(f64, &'static str)> = None;
    let mut take = |t: f64, what: &'static str| {
        if best.is_none_or(|(b, _)| t < b) {
            best = Some((t, what));
        }
    };
    for imp in &s.impactors {
        let v = imp.velocity.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v <= 0.0 {
            continue;
        }
        let r = match imp.shape {
            ImpactorShape::Sphere { radius } => radius,
            ImpactorShape::Box { half_extents } => half_extents.iter().copied().fold(f64::INFINITY, f64::min),
        };
        let mi = s.material(&imp.material);
        let target = s.bodies.first().and_then(|b| b.chunks.first()).map(|c| s.material(&c.material)).unwrap_or(mi);
        let es = |m: &stress_ref::material::Material| (1.0 - m.poisson_ratio * m.poisson_ratio) / (m.youngs_modulus * s.sim.stiffness_scale);
        let e_star = 1.0 / (es(mi) + es(target));
        take(2.87 * (imp.mass * imp.mass / (r * e_star * e_star * v)).powf(0.2), "impact");
    }
    for l in &s.loads {
        match l {
            LoadDesc::PointForce { magnitude, .. } => {
                if let Some(t) = rise(magnitude) {
                    take(t, "load");
                }
            }
            LoadDesc::Pressure { pressure, .. } => {
                if let Some(t) = rise(pressure) {
                    take(t, "pressure");
                }
            }
            _ => {}
        }
    }
    for e in &s.events {
        match e {
            EventDesc::RemoveSupports { duration, .. } | EventDesc::RemoveChunks { duration, .. } => take(*duration, "removal"),
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
    best
}

/// Peak Hertz contact pressure of the strongest impact over the target's compressive
/// strength: `delta = (15 m v^2 / (16 E* sqrt R))^(2/5)`, `F = 4/3 E* sqrt R delta^(3/2)`,
/// `p0 = (6 F E*^2 / (pi^3 R^2))^(1/3)`.
fn contact_pressure_ratio(s: &Scene) -> Option<f64> {
    s.impactors
        .iter()
        .filter_map(|imp| {
            let v = imp.velocity.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v <= 0.0 {
                return None;
            }
            let r = match imp.shape {
                ImpactorShape::Sphere { radius } => radius,
                ImpactorShape::Box { half_extents } => half_extents.iter().copied().fold(f64::INFINITY, f64::min),
            };
            let mi = s.material(&imp.material);
            let target = s.bodies.first().and_then(|b| b.chunks.first()).map(|c| s.material(&c.material))?;
            let es = |m: &stress_ref::material::Material| (1.0 - m.poisson_ratio * m.poisson_ratio) / (m.youngs_modulus * s.sim.stiffness_scale);
            let e_star = 1.0 / (es(mi) + es(target));
            let delta = (15.0 * imp.mass * v * v / (16.0 * e_star * r.sqrt())).powf(0.4);
            let force = 4.0 / 3.0 * e_star * r.sqrt() * delta.powf(1.5);
            let p0 = (6.0 * force * e_star * e_star / (std::f64::consts::PI.powi(3) * r * r)).cbrt();
            Some(p0 / target.compressive_strength)
        })
        .reduce(f64::max)
}

fn report(s: &Scene) {
    let sv = ReferenceSolver::new(s);
    // The largest anchored cluster's structure.
    let Some(ci) = (0..sv.clusters.len()).filter(|&c| sv.clusters[c].anchored).max_by_key(|&c| sv.clusters[c].chunks.len()) else {
        println!("{:28} no anchored structure", s.name);
        return;
    };
    let st = &sv.structures[sv.clusters[ci].structure];
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for &c in &sv.clusters[ci].chunks {
        let ch = &st.chunks[c];
        for d in 0..3 {
            let h = ch.half_extents.to_array()[d];
            lo[d] = lo[d].min(ch.center.to_array()[d] - h);
            hi[d] = hi[d].max(ch.center.to_array()[d] + h);
        }
        *counts.entry(ch.material.as_str()).or_default() += 1;
    }
    let extent = (0..3).map(|d| hi[d] - lo[d]).fold(0.0, f64::max);
    // The struck member's thickness: the structure's extent along the first moving
    // impactor's velocity (the wall's thickness for a ram).
    let member = s.impactors.iter().find(|i| i.velocity.iter().any(|&v| v != 0.0)).map(|imp| {
        let v = stress_ref::math::Vec3::from_array(imp.velocity).normalized();
        let (mut a, mut b) = (f64::INFINITY, f64::NEG_INFINITY);
        for &c in &sv.clusters[ci].chunks {
            let ch = &st.chunks[c];
            let reach: f64 = (0..3).map(|k| (ch.rotation.col(k).dot(v)).abs() * ch.half_extents.to_array()[k]).sum();
            let x = ch.center.dot(v);
            a = a.min(x - reach);
            b = b.max(x + reach);
        }
        b - a
    });
    let main = counts.iter().max_by_key(|e| *e.1).map(|e| *e.0).unwrap_or("");
    let m = s.material(main);
    let wave_speed = (m.youngs_modulus * s.sim.stiffness_scale / m.density).sqrt();
    let t_w = extent / wave_speed;
    let t_1 = sv.fundamental_period(ci);
    let t_m = member.map(|thickness| thickness / wave_speed);
    let tc = loading_time(s);
    let tier = match (tc, t_1) {
        (Some((t, _)), Some(p)) if t >= 10.0 * p => "1 quasi-static",
        (Some((t, _)), _) if t >= t_w => "2 structural dynamics",
        (Some(_), _) => "3 impact window",
        (None, _) => "-",
    };
    let f = |x: Option<f64>| x.map_or("-".to_string(), |v| format!("{v:.3e}"));
    println!(
        "{:28} {:5} chunks  t_c {:>10} ({:8})  T_w {:.3e}  T_w,member {:>10}  T_1 {:>10}  t_c/T_w {:>9}  t_c/T_w,member {:>9}  t_c/T_1 {:>9}  p0/f_c {:>7}  tier {}",
        s.name,
        sv.clusters[ci].chunks.len(),
        f(tc.map(|x| x.0)),
        tc.map_or("-", |x| x.1),
        t_w,
        f(t_m),
        f(t_1),
        tc.map_or("-".into(), |x| format!("{:.2}", x.0 / t_w)),
        match (tc, t_m) {
            (Some(x), Some(tm)) => format!("{:.2}", x.0 / tm),
            _ => "-".into(),
        },
        match (tc, t_1) {
            (Some(x), Some(p)) => format!("{:.3}", x.0 / p),
            _ => "-".into(),
        },
        contact_pressure_ratio(s).map_or("-".into(), |x| format!("{x:.1}")),
        tier
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        for s in stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()) {
            report(&s);
        }
    } else {
        for p in &args {
            let path = std::path::Path::new(p);
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            match stress_ref::scene_pack::import(path, &name, "structure") {
                Ok(s) => report(&s),
                Err(e) => println!("{name}: {e}"),
            }
        }
    }
}
