//! Layer 1 of the vehicle test strategy on the GPU kernel: the same small analytic
//! cases as `engine-softbody/tests/invariants_small.rs` (shared bodies in
//! `invariants_common`), each against its closed form with the same tolerances.
//! Skipped when no adapter is found.
//! Run: `cargo test --release -p engine-webgpu --test invariants_small_gpu -- --nocapture`

#[path = "../../engine-softbody/tests/invariants_common/mod.rs"]
mod invariants_common;
use invariants_common::*;

use engine_softbody::explicit_beams::ExplicitBeams;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

struct Gpu {
    g: ExplicitBeamsGpu,
    _dev: (wgpu::Device, wgpu::Queue),
}

impl Sim for Gpu {
    fn steps(&mut self, n: u32) {
        self.g.frame(n);
    }
    fn pos(&self) -> Vec<Vec3> {
        self.g.positions().remove(0)
    }
    fn vel(&self) -> Vec<Vec3> {
        self.g.velocities().remove(0)
    }
    fn beams(&self) -> Vec<(bool, f32)> {
        self.g.beams()[0]
            .iter()
            .map(|b| (b.broken, b.rest))
            .collect()
    }
    fn set_vel(&mut self, node: u32, v: Vec3) {
        self.g.set_velocity(0, node, v);
    }
    fn label(&self) -> &'static str {
        "GPU"
    }
}

fn gpu(s: ExplicitBeams) -> Option<Box<dyn Sim>> {
    let (_, device, queue) = default_device().ok()?;
    let g = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&s));
    Some(Box::new(Gpu {
        g,
        _dev: (device, queue),
    }))
}

fn skip() -> bool {
    if default_device().is_err() {
        eprintln!("no GPU adapter: skipped");
        return true;
    }
    false
}

#[test]
fn beam_frequency_and_damping_gpu() {
    if skip() {
        return;
    }
    for zeta in [0.0, BX_ZETA_BODY, BX_ZETA_LID] {
        let o = oscillator(&gpu, SKIN_K, zeta, SKIN_M).unwrap();
        let shift = (o.omega_discrete / o.omega_continuous - 1.0).abs();
        let mut pass = rel(o.omega, o.omega_discrete) < 1e-3
            && rel(o.omega, o.omega_continuous) < shift + 1e-3;
        if zeta > 0.0 {
            pass &= rel(o.sigma, o.sigma_discrete) < 0.01;
        }
        row(
            &format!("beam ω, σ (ζ {zeta})"),
            "GPU",
            format!("{:.2} rad/s, σ {:.2}/s", o.omega, o.sigma),
            format!("ω_d {:.2}, σ_d {:.2}", o.omega_discrete, o.sigma_discrete),
            pass,
        );
        assert!(pass);
    }
}

#[test]
fn beam_yield_and_break_gpu() {
    if skip() {
        return;
    }
    for jbeam in [false, true] {
        let y = yield_unload(&gpu, jbeam).unwrap();
        let pass = rel(y.set, y.set_closed) < 0.01
            && (y.yield_force - 1000.0).abs() < 2.6
            && rel(y.unload_slope, 1.0e5) < 1e-3;
        row(
            &format!("yield/set/unload {}", if jbeam { "JBeam" } else { "RoR" }),
            "GPU",
            format!(
                "Fy {:.1} N, set {:.3} mm, slope {:.0}",
                y.yield_force,
                y.set * 1e3,
                y.unload_slope
            ),
            format!("1000 N, {:.3} mm, 1e5", y.set_closed * 1e3),
            pass,
        );
        assert!(pass);
    }
    for ductile in [false, true] {
        let (x, f, xc, fc) = break_force(&gpu, ductile).unwrap();
        let pass = rel(x, xc) < 0.01 && rel(f, fc) < 0.01;
        row(
            &format!(
                "break {}",
                if ductile {
                    "ductile RoR"
                } else {
                    "brittle JBeam"
                }
            ),
            "GPU",
            format!("x {:.3} mm, F {:.0} N", x * 1e3, f),
            format!("x {:.3} mm, F {:.0} N", xc * 1e3, fc),
            pass,
        );
        assert!(pass);
    }
}

#[test]
fn anisotropic_period_gpu() {
    if skip() {
        return;
    }
    let (t, _tc, td) = anisotropic_period(&gpu).unwrap();
    let pass = rel(t, td) < 2e-3;
    row(
        "ANISOTROPIC period",
        "GPU",
        format!("{:.4} ms", t * 1e3),
        format!("step {:.4} ms", td * 1e3),
        pass,
    );
    assert!(pass);
}

#[test]
fn bar_wave_and_ripple_gpu() {
    if skip() {
        return;
    }
    let (c, c0) = wave_speed(&gpu, SKIN_K, SKIN_M, 0.21).unwrap();
    let pass = rel(c, c0) < 0.02;
    row(
        "bar wave speed",
        "GPU",
        format!("{c:.1} m/s"),
        format!("{c0:.1} m/s"),
        pass,
    );
    assert!(pass);
    let [w, wd, s, sd, _, _] = chain_mode(&gpu, 4, SKIN_K, BX_ZETA_BODY, SKIN_M, 0.21).unwrap();
    let pass = rel(w, wd) < 5e-3 && rel(s, sd) < 0.03;
    row(
        "bar mode 1 decay",
        "GPU",
        format!("ω {w:.1}, σ {s:.2}"),
        format!("ω {wd:.1}, σ {sd:.2}"),
        pass,
    );
    assert!(pass);
}

#[test]
fn ground_contact_gpu() {
    if skip() {
        return;
    }
    let (mut wd, mut we) = (0.0f64, 0.0f64);
    for v in [1.0f32, 10.0, 30.0, 100.0] {
        for phase in 0..3 {
            let h0 = 0.05 + v * 0.0005 * (phase as f32 / 3.0);
            let (d, r, vi) = ground_drop(&gpu, v, h0).unwrap();
            wd = wd.max(d / (vi * 0.0005 + 10.0 * 0.0005f64.powi(2)));
            we = we.max(r / vi);
        }
    }
    let pass = wd <= 1.0 && we <= 0.2 + 1e-4;
    row(
        "ground: depth / v·h, rebound / v",
        "GPU",
        format!("{wd:.3}, {we:.4}"),
        "≤ 1, ≤ 0.2".into(),
        pass,
    );
    assert!(pass);
}

#[test]
fn latch_release_gpu() {
    if skip() {
        return;
    }
    let (f, fc, latch, hinges) = latch_release(&gpu).unwrap();
    let pass = rel(f, fc) < 0.02 && latch && hinges == 0;
    row(
        "latch release",
        "GPU",
        format!("{f:.0} N, group {latch}, hinges broken {hinges}"),
        format!("{fc:.0} N"),
        pass,
    );
    assert!(pass);
}
