//! What the crash suite's 4 ms time histories keep of a pulse (`engine_softbody::crash_reference
//! ::metrics::series`), measured on the CPU reference's own 2 kHz trace (`series::Fine`).
//!
//! 1. The trace changes nothing: a run traced every step and the same run untraced (a runner
//!    that declines `start_trace`) report the same metrics to the bit (but the trace's own).
//! 2. From the trace: the J211 CFC 60 peak at 2 kHz (what a test's accelerometer reports,
//!    `ref.cfc60_peak_decel_g`), the suite's 4 ms estimator on the raw samples
//!    (`ref.peak_decel_g`) and the same estimator on the CFC 60 velocity (how the NHTSA
//!    references' "as Rubble samples" peak is made): the box-car's loss and the aliasing.
//!    `--nocapture` prints them.
//!
//! `cargo test --release -p engine-softbody-lab --test crash_series_rate -- --nocapture`
//! (S11 and S12, a few seconds each).

use engine_softbody::crash_reference::{self as cr, metrics::series, metrics::signal, Contact, CpuRunner, Jitter, Runner, Sample};

/// The CPU reference without its trace.
struct Untraced(CpuRunner);

impl Runner for Untraced {
    fn frame(&mut self, steps: u32) -> Result<(), String> {
        self.0.frame(steps)
    }
    fn sample(&mut self) -> Result<Sample, String> {
        self.0.sample()
    }
    fn contact_removed(&mut self) -> Option<f64> {
        self.0.contact_removed()
    }
}

fn probe(id: &str) {
    let s = cr::get(id).unwrap();
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).unwrap();
    let built = s.build(&templates, Jitter::run(0), Contact::Edge).unwrap();
    let a = cr::run(&s, &templates, &built, 0, &mut CpuRunner::new(&built), &mut |_| {}).unwrap();
    let b = cr::run(&s, &templates, &built, 0, &mut Untraced(CpuRunner::new(&built)), &mut |_| {}).unwrap();
    for (k, v) in &b.metrics {
        assert_eq!(v.to_bits(), a.metrics[k].to_bits(), "{k}: {v} vs {}", a.metrics[k]);
    }
    assert!(b.series.as_ref().unwrap().fine.is_none());
    let fine = a.series.as_ref().unwrap().fine.as_ref().expect("the CPU reference traces every step");
    assert_eq!(fine.rate_hz.round(), 2000.0);
    let m = &a.metrics;
    let g = series::G0;
    // The suite's estimator on the CFC 60 velocity: integrate the 1 kHz CFC 60 curve from the
    // contact's velocity, sample it every 4 ms, central differences.
    let f = &fine.bodies[0].reference;
    let dt = 1.0 / fine.curve_hz;
    let v = signal::integrate(&f.ax_g.iter().map(|x| x * g).collect::<Vec<_>>(), dt, f.vx[0]);
    let every = (0.004 / dt).round() as usize;
    let vs: Vec<f64> = v.iter().step_by(every).copied().collect();
    let ts: Vec<f64> = fine.t_ms.iter().step_by(every).copied().collect();
    let a4 = signal::differentiate(&vs, 0.004);
    let (tc4, pc4) = ts.iter().zip(&a4).filter(|(t, _)| **t >= 0.0).map(|(t, a)| (*t, -a / g)).fold((0.0, f64::MIN), |b, x| if x.1 > b.1 { x } else { b });
    // The CFC 60 velocity against the raw trace.
    let dv = v.iter().zip(&f.vx).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("{id}: reference location {} nodes", m["car.ref.nodes"]);
    println!("  CFC 60 at 2 kHz: {:.2} g at {:.1} ms", m["car.ref.cfc60_peak_decel_g"], m["car.ref.cfc60_peak_ms"]);
    println!("  4 ms estimator, raw samples (the report): {:.2} g at {:.1} ms", m["car.ref.peak_decel_g"], m["car.ref.peak_ms"]);
    println!("  4 ms estimator on the CFC 60 velocity: {pc4:.2} g at {tc4:.1} ms");
    println!("  largest |integrated CFC 60 velocity − raw| {dv:.3} m/s");
    // A finite pulse in the band the NHTSA references show (tens of g), and the 4 ms estimate
    // below the CFC 60 peak (a box-car of 8 ms keeps less of a peak than CFC 60).
    assert!(m["car.ref.cfc60_peak_decel_g"] > 10.0 && m["car.ref.cfc60_peak_decel_g"] < 200.0);
    assert!(m["car.ref.peak_decel_g"] <= m["car.ref.cfc60_peak_decel_g"] * 1.05);
}

#[test]
fn crash_series_rate_s11() {
    probe("S11");
}

#[test]
fn crash_series_rate_s12() {
    probe("S12");
}
