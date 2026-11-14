//! "Not worse than main" tests for the explicit solver's JBeam mode (the BeamNG
//! profile) and the loader changes of PR #105, item by item from the regression
//! audit's table "Soft-body solver that existed on main" (S3, S4, S5, S7, S8, S10).
//! Main is `30ab0e1d`. Where main's arithmetic is needed it is copied here as a
//! local function whose name starts with `main_` (from `git show
//! 30ab0e1d:engine/crates/engine-softbody/src/explicit_beams.rs`); the branch's is
//! called through its public API.
//!
//! S1 and S2 are covered elsewhere: `main_parity_golden.rs` lists every scenario
//! the BeamNG profile (S1) and `exact_length` (S2) change in `CHANGED`, each bounded
//! and with no more motion than main; `main_parity_cases.rs::
//! the_exact_length_is_within_two_ulp_of_mains` pins `exact_length` to 2 ulp of
//! main's `q * (1 / sqrt(q))` and no further from √q.

use engine_softbody::explicit_beams::*;
use engine_softbody::jbeam::{load_jbeam, read_vehicle_dir};
use glam::Vec3;

/// A small deterministic generator (64-bit LCG, top bits), for sweeps.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f32 {
        (lo + (hi - lo) * self.next()) as f32
    }
}

// ---------------------------------------------------------------------------------
// S3: `dampCutoffHz`. Main: a prewarped bilinear low-pass, cutoff clamped to 0.45 of
// the step rate (900 Hz). Branch: a one-pole RC, a = 1 − exp(−2π·f·Δt), no clamp.
// ---------------------------------------------------------------------------------

/// Main's `DampFilter::new` and `step` (30ab0e1d explicit_beams.rs:760-785).
#[derive(Clone, Copy)]
struct MainDampFilter {
    a: f32,
    b: f32,
    u: f32,
    y: f32,
}

impl MainDampFilter {
    const DAMP_CUTOFF_MAX: f32 = 0.45;
    fn new(hz: f32) -> Self {
        let fc = (hz * STEP).min(Self::DAMP_CUTOFF_MAX);
        let k = (std::f32::consts::PI * fc).tan();
        Self { a: k / (1.0 + k), b: (k - 1.0) / (k + 1.0), u: 0.0, y: 0.0 }
    }
    fn step(&mut self, u: f32) -> f32 {
        let y = self.a * (u + self.u) - self.b * self.y;
        self.u = u;
        self.y = y;
        y
    }
}

/// The largest |y − y_ref| over `steps` for main's and the branch's filter at
/// cutoff `hz`, driven by `input(n)`. The reference is the ideal first-order
/// low-pass dy/dt = 2π·hz·(u − y) in f64, driven the way the solver drives the
/// filter: the beam's rate is sampled once per step and held for that step, so over
/// step n the input is u[n] and the exact solution at the step's end is
/// y_ref[n] = e^{−ωΔt}·y_ref[n−1] + (1 − e^{−ωΔt})·u[n].
fn filter_errors(hz: f32, steps: u32, input: impl Fn(u32) -> f32) -> (f64, f64) {
    let decay = (-2.0 * std::f64::consts::PI * f64::from(hz) * f64::from(STEP)).exp();
    let (mut main, mut branch) = (MainDampFilter::new(hz), DampFilter::new(hz));
    let (mut y_ref, mut em, mut eb) = (0.0f64, 0.0f64, 0.0f64);
    for n in 0..steps {
        let u = input(n);
        y_ref = decay * y_ref + (1.0 - decay) * f64::from(u);
        em = em.max((f64::from(main.step(u)) - y_ref).abs());
        eb = eb.max((f64::from(branch.step(u)) - y_ref).abs());
    }
    (em, eb)
}

/// S3: for a step and for sines below, at and above the cutoff, at cutoffs from
/// 5 Hz to main's 900 Hz clamp and past it, the branch's one-pole is at least as
/// close to the ideal first-order low-pass (input held per step, as the solver
/// samples it) as main's bilinear filter. Measured (max |error| over 1 s, unit
/// input): the branch within 1.9e-6 everywhere (f32 rounding); main off by 7.7e-4
/// (5 Hz, sine at 0.1·fc) up to 0.66 (900 Hz, sine at fc), and past its clamp (it
/// filters at 900 Hz whatever is asked) 0.44 at 1500 Hz and 0.24 at 5 kHz. Past
/// main's clamp the branch stays stable (bounded by its input).
///
/// Caveat, measured but not asserted (a different reference): against the
/// continuous low-pass's frequency response 1/(1 + j·f/fc) at the sample instants,
/// main's prewarped bilinear is closer: it is exact at f = fc (|H| = 1/√2) where the
/// branch's gain is off by 7.4e-6 (5 Hz) to 0.18 (900 Hz), and the branch leads by
/// half a step in phase. Its gain error at 0.1·fc and 2·fc is lower than main's.
#[test]
fn s3_damp_cutoff_is_no_further_from_the_ideal_low_pass_than_mains() {
    let pi2 = 2.0 * std::f64::consts::PI;
    let steps = 2000; // 1 s
    let mut table = String::new();
    let mut worse = Vec::new();
    for hz in [5.0f32, 50.0, 200.0, 500.0, 900.0, 1500.0, 5000.0, 20000.0] {
        let mut cases: Vec<(String, Box<dyn Fn(u32) -> f32>)> = vec![("step".into(), Box::new(|_| 1.0))];
        for r in [0.1f64, 0.5, 1.0, 2.0, 5.0] {
            let f = f64::from(hz) * r;
            if f >= 0.5 / f64::from(STEP) {
                continue; // at or past Nyquist the held input is not that sine
            }
            cases.push((
                format!("sine {r}·fc"),
                Box::new(move |n| (pi2 * f * f64::from(n) * f64::from(STEP)).sin() as f32),
            ));
        }
        for (name, input) in cases {
            let (em, eb) = filter_errors(hz, steps, input);
            table.push_str(&format!("{hz} Hz {name}: main {em:.3e}, branch {eb:.3e}\n"));
            if eb > em {
                worse.push(format!("{hz} Hz {name}: branch {eb:e} > main {em:e}"));
            }
        }
    }
    eprintln!("dampCutoffHz, max |y - ideal| over 1 s:\n{table}");
    assert!(worse.is_empty(), "the branch's filter is further from the ideal low-pass than main's:\n{}", worse.join("\n"));

    // Stable where main clamped the cutoff (and at any cutoff): driven by ±1 at the
    // Nyquist rate (the worst case for a one-pole), the output never leaves [−1, 1].
    for hz in [900.0f32, 1000.0, 1500.0, 5000.0, 1.0e5, 1.0e9] {
        let mut f = DampFilter::new(hz);
        assert!(f.a > 0.0 && f.a <= 1.0 && f.b == f.a - 1.0, "{hz} Hz: a {} b {}", f.a, f.b);
        for n in 0..4000 {
            let y = f.step(if n % 2 == 0 { 1.0 } else { -1.0 });
            assert!(y.is_finite() && y.abs() <= 1.0 + 1.0e-6, "{hz} Hz: y {y} at step {n}");
        }
    }

    // For the report: the frequency-response comparison of the caveat above.
    let mut fr = String::new();
    for hz in [5.0f64, 50.0, 200.0, 500.0, 900.0] {
        let (m, b) = (MainDampFilter::new(hz as f32), DampFilter::new(hz as f32));
        for r in [0.1f64, 1.0, 2.0] {
            let w = pi2 * hz * r * f64::from(STEP);
            if hz * r >= 1000.0 {
                continue;
            }
            // |H(e^{jw})| of y = a(u + u⁻) − b·y⁻ and y = a·u − b·y⁻.
            let mag = |num: (f64, f64), b: f64| {
                let den = (1.0 + b * w.cos(), -b * w.sin());
                ((num.0 * num.0 + num.1 * num.1) / (den.0 * den.0 + den.1 * den.1)).sqrt()
            };
            let hm = mag((f64::from(m.a) * (1.0 + w.cos()), -f64::from(m.a) * w.sin()), f64::from(m.b));
            let hb = mag((f64::from(b.a), 0.0), f64::from(b.b));
            let hc = 1.0 / (1.0 + r * r).sqrt();
            fr.push_str(&format!("{hz} Hz at {r}·fc: |H| error main {:.2e}, branch {:.2e}\n", (hm - hc).abs(), (hb - hc).abs()));
        }
    }
    eprintln!("dampCutoffHz gain vs the continuous low-pass (not asserted):\n{fr}");
}

// ---------------------------------------------------------------------------------
// S4: torsion angle. Main: `f32::atan2`. Branch: `exact_atan2` (a fixed series the
// CPU and the kernels run identically).
// ---------------------------------------------------------------------------------

/// Distance in f32 ulps between `got` and the f64 angle `want` rounded to f32, with
/// angles taken modulo 2π (atan2(−0, −1) is −π in IEEE and +π from `exact_atan2`:
/// the same torsion angle after `wrap_angle`).
fn angle_ulps(got: f32, want: f64) -> u32 {
    let mut d = f64::from(got) - want;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    let r = (want + d) as f32; // got, unwrapped next to `want`
    let w = want as f32;
    // Bits as a monotonic integer line (through ±0).
    let key = |x: f32| {
        let b = x.to_bits() as i64;
        if b & 0x8000_0000 != 0 {
            -(b & 0x7fff_ffff)
        } else {
            b
        }
    };
    (key(r) - key(w)).unsigned_abs() as u32
}

/// S4: over a dense sweep of directions at radii from 1e-38 to 1e38 (subnormals
/// included), the axes, mixed tiny/huge pairs, the origin and 2 million random
/// points, `exact_atan2` against the f64 atan2, next to `f32::atan2` (main's).
///
/// **Less accurate than main, pinned at what was measured** (2.48 M inputs, this
/// host's libm): `exact_atan2` at most 3 ulp (2.77e-7 rad), `f32::atan2` at most
/// 1 ulp (1.51e-7 rad); per input `exact_atan2` is at most 3 ulp further than
/// `f32::atan2` (worst near 0.42-0.50 rad, just past the series' range reduction at
/// tan(π/8), normal and subnormal inputs alike). The branch trades this for CPU and
/// GPU kernels computing the same bits (no backend's atan2 is the same function).
/// The assertions hold it to those numbers so it cannot get worse.
#[test]
fn s4_exact_atan2_error_is_pinned_at_three_ulp_over_mains() {
    let mut inputs: Vec<(f32, f32)> = Vec::new();
    for r in [1.0e-38f64, 1.0e-30, 1.0e-12, 1.0e-6, 1.0e-3, 1.0, 7.3, 1.0e3, 1.0e6, 1.0e12, 1.0e30, 1.0e38] {
        let n = 40_000;
        for i in 0..n {
            let t = std::f64::consts::TAU * (f64::from(i) + 0.5) / f64::from(n) - std::f64::consts::PI;
            inputs.push(((r * t.sin()) as f32, (r * t.cos()) as f32));
        }
    }
    for v in [1.0e-38f32, 1.0e-20, 1.0, 1.0e20, 3.0e38] {
        for (y, x) in [(0.0, v), (0.0, -v), (v, 0.0), (-v, 0.0), (v, v), (-v, v), (v, -v), (-v, -v)] {
            inputs.push((y, x));
        }
    }
    for (a, b) in [(1.0e-30f32, 1.0e30f32), (1.0e-38, 3.0e38), (1.0, 1.0e-30)] {
        for (y, x) in [(a, b), (b, a), (-a, b), (a, -b), (-b, -a)] {
            inputs.push((y, x));
        }
    }
    inputs.push((0.0, 0.0));
    let mut rng = Lcg(4);
    for _ in 0..2_000_000 {
        inputs.push((rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)));
    }
    let (mut worst_exact, mut worst_main, mut worst_excess) = (0u32, 0u32, 0i64);
    let mut worst_at = String::new();
    let (mut rad_exact, mut rad_main) = (0.0f64, 0.0f64);
    for &(y, x) in &inputs {
        let want = f64::from(y).atan2(f64::from(x));
        let (e, m) = (exact_atan2(y, x), y.atan2(x));
        assert!(e.is_finite(), "exact_atan2({y}, {x}) = {e}");
        let (ue, um) = (angle_ulps(e, want), angle_ulps(m, want));
        worst_exact = worst_exact.max(ue);
        worst_main = worst_main.max(um);
        if i64::from(ue) - i64::from(um) > worst_excess {
            worst_excess = i64::from(ue) - i64::from(um);
            worst_at = format!("atan2({y:e}, {x:e}) = {want:e}: exact_atan2 {e:e} ({ue} ulp), f32::atan2 {m:e} ({um} ulp)");
        }
        let rad = |a: f32| {
            let d = (f64::from(a) - want).abs();
            d.min((d - std::f64::consts::TAU).abs())
        };
        rad_exact = rad_exact.max(rad(e));
        rad_main = rad_main.max(rad(m));
    }
    eprintln!(
        "atan2 over {} inputs: exact_atan2 at most {worst_exact} ulp ({rad_exact:.2e} rad), f32::atan2 at most {worst_main} ulp ({rad_main:.2e} rad); per input exact - main at most {worst_excess} ulp, at {worst_at}",
        inputs.len()
    );
    assert!(worst_excess <= 3, "exact_atan2 is {worst_excess} ulp further from atan2 than f32::atan2 at {worst_at} (measured: 3)");
    assert!(worst_exact <= 3, "exact_atan2 at most {worst_exact} ulp (measured: 3)");
    assert!(rad_exact <= 2.8e-7, "exact_atan2 at most {rad_exact:e} rad (measured: 2.77e-7)");
}

// ---------------------------------------------------------------------------------
// S5: JBeam yield sign guards.
// ---------------------------------------------------------------------------------

/// Main's `jbeam_force_filtered` without a filter on the fast path (30ab0e1d
/// explicit_beams.rs:2889-3013) for JBeam plain, support and bounded beams; also
/// whether the yield ran with the wrong sign (the case the branch guards).
fn main_jbeam_force(beam: &mut Beam, na: &Node, nb: &Node) -> (f32, bool, bool) {
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let inv = fast_inv_sqrt(q);
    let len = q * inv;
    let e = len - beam.rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let mut k = beam.k;
    let c = beam.c;
    let mut damp = c * u;
    match beam.kind {
        ExplicitKind::JbeamBounded => {
            let split = beam.tc;
            let au = u.abs();
            if au > split {
                damp = (c * split + beam.tt * (au - split)).copysign(u);
            }
            let long = beam.long_bound * beam.rest;
            let short = beam.short_bound * beam.rest;
            let o = if e > long {
                e - long
            } else if e < -short {
                -e - short
            } else {
                0.0
            };
            if o > 0.0 {
                let zone = beam.plastic;
                let s = if zone > 0.0 { (o / zone).min(1.0) } else { 1.0 };
                k += (beam.limit_k - k) * s;
                damp += (beam.limit_c * u - damp) * s;
            }
        }
        ExplicitKind::JbeamSupport => {
            if e > 0.0 {
                if len > beam.long_bound {
                    beam.broken = true;
                    beam.force = 0.0;
                    return (0.0, false, false);
                }
                k = 0.0;
                damp = 0.0;
            }
        }
        _ => {}
    }
    let mut f = -k * e - damp;
    beam.force = f;
    let mut tau = f.abs();
    let mut wrong_sign = false;
    if tau > beam.gate && beam.ty == ExplicitType::Deformable && k != 0.0 {
        if f > beam.tc && e < 0.0 {
            let delta = e + (beam.tc / k) * (1.0 - beam.plastic);
            wrong_sign = !(delta < 0.0);
            let old = beam.rest;
            let floor = beam.short_bound.max(MIN_REST);
            let free = beam.rest + delta;
            if free < floor {
                beam.rest = floor.min(old);
                f = -k * (len - beam.rest) - damp;
            } else {
                beam.rest = free;
                f -= (f - beam.tc) / 2.0;
            }
            tau = f.abs();
            if beam.rest < old {
                beam.tc = (beam.tc * old / beam.rest).min(beam.limit_k.max(beam.tc));
                beam.gate = beam.tc.min(beam.tt).min(beam.strength);
            }
        } else if f < -beam.tt && e > 0.0 {
            let delta = e - (beam.tt / k) * (1.0 - beam.plastic);
            wrong_sign = !(delta > 0.0);
            let old = beam.rest;
            let ceiling = beam.long_bound;
            let free = beam.rest + delta;
            if free > ceiling {
                beam.rest = ceiling.max(old);
                f = -k * (len - beam.rest) - damp;
            } else {
                beam.rest = free;
                f -= (f + beam.tt) / 2.0;
            }
            tau = f.abs();
            if beam.rest > old {
                beam.tt = (beam.tt * beam.rest / old).min(beam.limit_k.max(beam.tt));
                beam.gate = beam.tc.min(beam.tt).min(beam.strength);
                beam.strength -= (beam.rest - old) * k;
            }
        }
    }
    (f, tau > beam.strength, wrong_sign)
}

fn node(pos: Vec3, vel: Vec3) -> Node {
    Node { pos, vel, force: Vec3::ZERO, mass: 1.0, friction: 1.0, flags: 0 }
}

/// A deformable JBeam plain beam along x: rest `rest`, spring `k`, damping `c`,
/// yield `deform` both ways; bounds in metres as JBeam mode sets them.
fn plain(rest: f32, k: f32, c: f32, deform: f32) -> Beam {
    let mut b = Beam::new(0, 1, ExplicitKind::JbeamPlain, ExplicitType::Deformable, k, c, rest, deform, 1.0e9, 0.0);
    b.short_bound = 0.5 * rest;
    b.long_bound = 2.0 * rest;
    b.limit_k = 1.0e9;
    b
}

/// S5: where main's yield ran with the wrong sign (a beam pressed only by its
/// damping: compressed by less than its yield strain while closing fast, or the
/// tension mirror), main moved the rest length the wrong way: longer while
/// compressed, shorter while stretched, and with a yield strain past the rest length
/// to a negative rest length. The branch's guards skip that yield: the rest length
/// stays put and everything stays finite. Measured: main 1.0 → 1.078 m (compression),
/// 1.0 → 0.919 m (tension), 0.15 → −0.130 m (tension, negative); the branch keeps
/// 1.0, 1.0 and 0.15 m. In a 2 m/s two-node impact (only ever compressed) main's
/// rest length ends at 1.123 m, the branch's at 0.9996 m.
#[test]
fn s5_yield_sign_guards_keep_the_rest_length_physical() {
    // (rest, k, c, deform, extension, closing rate): the force is the damping's.
    let cases = [
        // (|extension| 2 cm: well past the fast inverse square root's 0.2 % error.)
        ("compression, damping-driven", 1.0f32, 1.0e4f32, 1.0e5f32, 1.0e3f32, -0.02f32, -1.0f32),
        ("tension, damping-driven", 1.0, 1.0e4, 1.0e5, 1.0e3, 0.02, 1.0),
        ("tension past the rest length", 0.15, 1.0e4, 1.0e5, 3.0e3, 0.02, 1.0),
    ];
    for (name, rest, k, c, deform, e, rate) in cases {
        let na = node(Vec3::new(rest + e, 0.0, 0.0), Vec3::new(rate, 0.0, 0.0));
        let nb = node(Vec3::ZERO, Vec3::ZERO);
        let mut m = plain(rest, k, c, deform);
        let (fm, _, wrong) = main_jbeam_force(&mut m, &na, &nb);
        let mut b = plain(rest, k, c, deform);
        let out = jbeam_force_filtered(&mut b, &na, &nb, true, None, false);
        eprintln!("S5 {name}: main rest {rest} -> {} (force {fm}), branch rest -> {} (force {})", m.rest, b.rest, out.force);
        assert!(wrong, "{name}: the case must be one main yields with the wrong sign");
        // Main's wrong-sign yield, as recorded in the doc comment.
        if e < 0.0 {
            assert!(m.rest > rest, "{name}: main lengthened the compressed beam");
        } else {
            assert!(m.rest < rest, "{name}: main shortened the stretched beam");
        }
        // The branch: no yield against its sign, a finite force of the right sign.
        assert_eq!(b.rest, rest, "{name}: the branch keeps the rest length");
        assert!(b.rest > 0.0 && out.force.is_finite() && b.tc.is_finite() && b.tt.is_finite());
        assert_eq!(out.force.signum(), -rate.signum(), "{name}: the force resists the motion");
    }
    // Main reached a negative rest length in the third case.
    let mut m = plain(0.15, 1.0e4, 1.0e5, 3.0e3);
    main_jbeam_force(&mut m, &node(Vec3::new(0.17, 0.0, 0.0), Vec3::X), &node(Vec3::ZERO, Vec3::ZERO));
    assert!(m.rest < 0.0, "main's rest length {}", m.rest);

    // Dynamics: two 1 kg nodes closing at 2 m/s on a stiff, heavily damped beam
    // (a crash-like impulse). After the hit, main's beam is longer than at spawn
    // although only ever pressed; the branch's is not.
    let run = |use_main: bool| {
        let mut beam = plain(1.0, 2.0e5, 2.0e3, 2.0e3);
        let mut n = [node(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)), node(Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0))];
        for _ in 0..400 {
            let (fa, fb) = (n[1], n[0]);
            let f = if use_main {
                main_jbeam_force(&mut beam, &fa, &fb).0
            } else {
                jbeam_force_filtered(&mut beam, &fa, &fb, true, None, false).force
            };
            // Positive = compression: pushes the nodes apart along a − b.
            let dir = (n[1].pos - n[0].pos).normalize();
            n[1].vel += dir * f * STEP;
            n[0].vel -= dir * f * STEP;
            for x in &mut n {
                x.pos += x.vel * STEP;
            }
        }
        (beam.rest, n[1].pos.x - n[0].pos.x)
    };
    let (rest_main, len_main) = run(true);
    let (rest_branch, len_branch) = run(false);
    eprintln!("S5 impact: rest main {rest_main} (length {len_main}), branch {rest_branch} (length {len_branch})");
    assert!(rest_main > 1.0, "main's beam grew under compression: {rest_main}");
    assert!(rest_branch <= 1.0 && rest_branch.is_finite() && len_branch.is_finite(), "branch rest {rest_branch}");
}

/// S5, the ordinary case: wherever main's yield had the right sign (or did not
/// run), the branch's `jbeam_force_filtered` (fast path, no filter, main's strength
/// drop) is bit-identical to main's: force, break flag and every beam field, for
/// random JBeam plain, support and bounded beams. (`main_parity_golden`'s JBeam
/// scenarios cannot show this: they run the whole BeamNG profile, S1.)
#[test]
fn s5_without_a_guard_the_jbeam_force_is_mains_bit_for_bit() {
    let mut rng = Lcg(5);
    let (mut ordinary, mut yielded, mut guarded) = (0u32, 0u32, 0u32);
    for i in 0..200_000 {
        let kind = [ExplicitKind::JbeamPlain, ExplicitKind::JbeamSupport, ExplicitKind::JbeamBounded][i % 3];
        let rest = rng.range(0.12, 2.0);
        let k = rng.range(1.0e3, 5.0e6);
        let c = rng.range(0.0, 2.0e3);
        let deform = rng.range(50.0, 5.0e4);
        let ty = if kind == ExplicitKind::JbeamBounded { ExplicitType::Actuator } else { ExplicitType::Deformable };
        let mut beam = Beam::new(0, 1, kind, ty, k, c, rest, deform, rng.range(1.0e3, 1.0e6), 0.0);
        if kind == ExplicitKind::JbeamBounded {
            beam.limit_k = rng.range(0.0, 1.0e6);
            beam.limit_c = rng.range(0.0, 1.0e3);
            beam.short_bound = rng.range(0.0, 0.5);
            beam.long_bound = rng.range(0.0, 0.5);
            beam.plastic = rng.range(0.0, 0.05);
            beam.tc = rng.range(0.0, 1.0);
            beam.tt = rng.range(0.0, 5.0e3);
        } else {
            beam.gate = deform.min(beam.strength);
            beam.limit_k = rng.range(0.0, 1.0e6);
            beam.short_bound = rest * rng.range(0.3, 1.0);
            beam.long_bound = rest * rng.range(1.0, 1.5);
        }
        let len = rest * rng.range(0.6, 1.4);
        let dir = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).try_normalize().unwrap_or(Vec3::X);
        let va = Vec3::new(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), rng.range(-5.0, 5.0));
        let na = node(dir * len, va);
        let nb = node(Vec3::ZERO, Vec3::ZERO);
        let mut m = beam;
        let (fm, bm, wrong) = main_jbeam_force(&mut m, &na, &nb);
        if wrong {
            guarded += 1;
            continue;
        }
        let mut b = beam;
        let out = jbeam_force_filtered(&mut b, &na, &nb, true, None, false);
        assert_eq!(
            (out.force.to_bits(), out.check_break, b),
            (fm.to_bits(), bm, m),
            "case {i}: {beam:?} at length {len}, rate {va}"
        );
        ordinary += 1;
        yielded += u32::from(m.rest != beam.rest);
    }
    eprintln!("S5: {ordinary} ordinary cases bit-identical to main ({yielded} of them yielded), {guarded} guarded cases skipped");
    assert!(ordinary > 150_000 && yielded > 1_000, "the sweep must exercise the yield ({ordinary}, {yielded})");
}

// ---------------------------------------------------------------------------------
// S7: slide nodes in the vehicle frame, per-node sums.
// ---------------------------------------------------------------------------------

fn golden(name: &str) -> Vec<u32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/main_golden").join(name);
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The final state's node positions in a golden scenario's words (the layout of
/// `main_parity_golden.rs::state`: count, per node position and velocity bits, a
/// hash; spawn first, final last).
fn final_positions(w: &[u32]) -> Vec<Vec3> {
    let n = w[0] as usize;
    let at = n * 6 + 3;
    let m = w[at] as usize;
    (0..m)
        .map(|i| {
            let p = &w[at + 1 + i * 6..at + 4 + i * 6];
            Vec3::new(f32::from_bits(p[0]), f32::from_bits(p[1]), f32::from_bits(p[2]))
        })
        .collect()
}

/// The largest distance between two runs' nodes, each taken relative to its node 0
/// (fixed: its World position is the spawn point, so this removes the origin), in f64.
fn shape_error(a: &[Vec3], b: &[Vec3]) -> f64 {
    let rel = |p: &[Vec3], i: usize| [0, 1, 2].map(|k| f64::from(p[i].to_array()[k]) - f64::from(p[0].to_array()[k]));
    (0..a.len())
        .map(|i| {
            let (x, y) = (rel(a, i), rel(b, i));
            ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// S7: the slide-node rigs of the goldens (one node on a rail; two nodes on one
/// rail), 400 steps at 2 kHz spawned 360 m out, against the same run at the origin.
/// Main's error is re-derived from its own goldens (its far run against its origin
/// run, which the branch reproduces bit for bit); the branch's far run must be no
/// further from the origin run than main's. Measured (largest node distance, shape
/// relative to the fixed node 0): one node main 7.59e-6 m, branch 4.6e-7 m; two
/// nodes on one rail main 6.23e-6 m, branch 8.2e-7 m.
#[test]
fn s7_slide_nodes_far_out_are_no_further_from_the_origin_run_than_mains() {
    let mut s = ExplicitBeamsSettings::default();
    s.fast_inv_sqrt = true;
    for rig in ["rig__slide_free", "rig__slide_two_on_one_rail"] {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/main_golden/trucks/{rig}.truck")),
        )
        .unwrap();
        let run = |at: Vec3| {
            let vel = Vec3::new(2.0, 0.0, 1.0);
            let mut v = ExplicitBeams::from_truck(&text, at, vel, s).unwrap();
            v.roll_tyres(vel);
            for i in 0..400 {
                if i % 33 == 0 {
                    v.rebase_origin();
                }
                v.step();
            }
            v.nodes.iter().map(|n| n.pos).collect::<Vec<_>>()
        };
        let origin = Vec3::new(0.0, 0.5, 0.0);
        let far = Vec3::new(300.0, 0.5, -200.0);
        let main_origin = final_positions(&golden(&format!("truck__{rig}__origin_fast.bin")));
        let main_far = final_positions(&golden(&format!("truck__{rig}__far_fast.bin")));
        let branch_origin = run(origin);
        assert_eq!(branch_origin, main_origin, "{rig}: the origin run is main's (golden)");
        let branch_far = run(far);
        let (em, eb) = (shape_error(&main_far, &main_origin), shape_error(&branch_far, &main_origin));
        eprintln!("S7 {rig}: 360 m out vs the origin run: main {em:.3e} m, branch {eb:.3e} m (branch vs main far {:.3e} m)", shape_error(&branch_far, &main_far));
        assert!(eb <= em, "{rig}: the branch is {eb} m from the origin run, main {em} m");
    }
}

// ---------------------------------------------------------------------------------
// S8: precompression on plain beams.
// ---------------------------------------------------------------------------------

const S8_TRUCK: &str = "precompression test
globals
100, 0, mat
nodes
0, 0, 1, 0
1, 1, 1, 0
2, 0, 2, 0
3, 1, 2, 0
beams
0, 1, n, 0, 0.9
0, 2
1, 2, n, 0, 0
2, 3, n, 0, -1
0, 3, n
end
";

/// S8: a `.truck` `beams` line's fifth field (Rubble's extension) scales a plain
/// beam's rest length; a line without it, or with 0 or a negative value there, keeps
/// main's rest length (the spawn distance) bit for bit. And no `.truck` in the repo
/// carries such a field, so main-era content spawns as on main (the goldens'
/// `truck_spawn__*` digests and `.truck` scenarios pin those bit for bit).
///
/// Also measured (S8's other reach): `ExplicitBeams::new` of a JBeam desc now applies
/// `beamPrecompression` to plain and support beams (main applied it to bounded beams
/// only); the latched-box fixture has 5 such beams (`ExplicitBeams::new` is used by the
/// World's `desc` spawn and `flexmesh`), asserted below as the intended rule.
#[test]
fn s8_precompression_scales_plain_beams_only_when_asked() {
    let v = ExplicitBeams::from_truck(S8_TRUCK, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap();
    let spawn = |a: u32, b: u32| v.nodes[a as usize].pos.distance(v.nodes[b as usize].pos);
    let rest = |a: u32, b: u32| v.beams.iter().find(|x| (x.a, x.b) == (a, b) || (x.a, x.b) == (b, a)).unwrap().rest;
    assert_eq!(rest(0, 1), spawn(0, 1) * 0.9, "the fifth field scales the rest length");
    for (a, b) in [(0, 2), (1, 2), (2, 3), (0, 3)] {
        assert_eq!(rest(a, b).to_bits(), spawn(a, b).to_bits(), "{a}-{b}: main's rest length (spawn distance)");
    }

    // Every committed .truck: no plain beam carries a precompression.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut files = Vec::new();
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                // Not the gitignored reference tree, builds, dependencies or other worktrees.
                if !name.starts_with('.') && !["inspiration", "target", "node_modules", "out", "dist"].contains(&name.as_str()) {
                    walk(&p, out);
                }
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("truck")) {
                out.push(p);
            }
        }
    }
    walk(&root.join("engine"), &mut files);
    walk(&root.join("web"), &mut files);
    assert!(files.len() >= 3, "found {} .truck files", files.len());
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap();
        let file = engine_softbody::truck::parse_truck(&text);
        for (i, b) in file.beams.iter().enumerate() {
            if b.kind != engine_softbody::desc::BeamKind::Bounded {
                assert_eq!(b.precompression, 1.0, "{}: beam {i} carries a precompression", path.display());
            }
        }
    }
    eprintln!("S8: {} .truck files, none with a plain-beam precompression", files.len());

    // JBeam descs through `ExplicitBeams::new`: rest = spawn × beamPrecompression.
    let text = engine_softbody::jbeam_fixture::latched_box_jbeam();
    let desc = load_jbeam(&[("v.jbeam", text.as_str())], None).unwrap();
    let sim = ExplicitBeams::new(&desc, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default());
    let mut scaled = 0;
    for (b, d) in sim.beams.iter().zip(&desc.beams) {
        let spawn = sim.nodes[b.a as usize].pos.distance(sim.nodes[b.b as usize].pos);
        if d.kind != engine_softbody::desc::BeamKind::Bounded && d.precompression != 1.0 {
            assert_eq!(b.rest, spawn * d.precompression);
            scaled += 1;
        }
    }
    eprintln!("S8: latched box through ExplicitBeams::new: {scaled} non-bounded beams take their beamPrecompression (main: spawn length)");
}

// ---------------------------------------------------------------------------------
// S10: loader additions. Each feature on a tiny synthetic fixture; the same fixture
// without the feature loads by main's rules. The committed fixtures (latched box,
// hires car) are pinned field by field against main by `main_parity_golden.rs::
// the_loaders_and_the_packer_reproduce_main`.
// ---------------------------------------------------------------------------------

/// A main part with one pressure wheel row and slot `tire` (part `tire_a`), whose
/// `pressureWheels` section is `tire_rows`; `own` is a modifier before the row.
fn wheel_vehicle(tire_rows: &str, own: &str) -> String {
    format!(
        r#"{{
"car": {{
    "slotType": "main",
    "slots": [["type", "default", "description"], ["tire", "tire_a", "Tyre"]],
    "nodes": [["id", "posX", "posY", "posZ"], {{"nodeWeight": 5}},
        ["a1", -0.6, 0, 0.3], ["a2", 0.6, 0, 0.3], ["c", 0, 1, 0.3]],
    "beams": [["id1:", "id2:"], ["a1", "a2"], ["a1", "c"], ["a2", "c"]],
    "pressureWheels": [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"],
        {own}
        ["FL", "wheel_FL", "tire_FL", "a1", "a2", 9999, "c", 1]]
}},
"tire_a": {{
    "slotType": "tire",
    "pressureWheels": [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"]{tire_rows}]
}}
}}"#
    )
}

/// S10: a pressure wheel row takes the `pressureWheels` modifiers of the other parts
/// of its slot tree (BeamNG's tyre parts carry only modifiers); its own part's
/// modifiers still win. Without such modifiers the row loads by main's rule (its own
/// modifiers, else the 0.3 m / 0.2 m defaults).
#[test]
fn s10_pressure_wheels_inherit_modifiers_across_parts() {
    let load = |t: &str| load_jbeam(&[("v.jbeam", t)], None).unwrap();
    let inherited = load(&wheel_vehicle(r#", {"radius": 0.42, "tireWidth": 0.25, "brakeTorque": 800}"#, ""));
    assert_eq!(inherited.wheels.len(), 1);
    assert_eq!((inherited.wheels[0].radius, inherited.wheels[0].width), (0.42, 0.25), "from the tyre part");
    assert!(inherited.wheels[0].braked);
    let own = load(&wheel_vehicle(r#", {"radius": 0.42, "tireWidth": 0.25}"#, r#"{"radius": 0.35},"#));
    assert_eq!((own.wheels[0].radius, own.wheels[0].width), (0.35, 0.25), "own part's modifier wins");
    // Not using the feature: main's rule.
    let plain = load(&wheel_vehicle("", ""));
    assert_eq!((plain.wheels[0].radius, plain.wheels[0].width), (0.3, 0.2), "main's defaults");
    assert!(!plain.wheels[0].braked);
    let plain_own = load(&wheel_vehicle("", r#"{"radius": 0.35, "tireWidth": 0.22},"#));
    assert_eq!((plain_own.wheels[0].radius, plain_own.wheels[0].width), (0.35, 0.22));
}

/// A main part with slots `wheel_F` (default `wheel_F_std`) and `body` (default
/// `body_a`), each local part adding one node; a `.pc` choosing `wheel` and `body`.
fn slot_vehicle(wheel: &str, body: &str) -> (String, String) {
    let jbeam = r#"{
"car": {
    "slotType": "main",
    "slots": [["type", "default", "description"], ["wheel_F", "wheel_F_std", "Wheel"], ["body", "body_a", "Body"]],
    "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 5}, ["m1", 0, 0, 0], ["m2", 1, 0, 0]],
    "beams": [["id1:", "id2:"], ["m1", "m2"]]
},
"wheel_F_std": {"slotType": "wheel_F", "nodes": [["id", "posX", "posY", "posZ"], ["w1", 0, 0, 1]]},
"body_a": {"slotType": "body", "nodes": [["id", "posX", "posY", "posZ"], ["b1", 0, 1, 0]]},
"body_b": {"slotType": "body", "nodes": [["id", "posX", "posY", "posZ"], ["b2", 0, 2, 0]]}
}"#;
    let pc = format!(r#"{{"format": 2, "mainPartName": "car", "parts": {{"wheel_F": "{wheel}", "body": "{body}"}}}}"#);
    (jbeam.to_string(), pc)
}

/// S10: a slot whose configured part is missing (stock or another mod's content)
/// takes its local default instead of dropping the slot. A configuration whose parts
/// all exist assembles exactly the parts main assembled, with no substitution note.
#[test]
fn s10_missing_slot_parts_are_substituted() {
    let names = |jbeam: &str, pc: &str| {
        let d = load_jbeam(&[("v.jbeam", jbeam)], Some(pc)).unwrap();
        (d.nodes.iter().map(|n| n.name.clone()).collect::<Vec<_>>(), d.warnings)
    };
    let (j, pc) = slot_vehicle("wheel_F_stock", "body_stock");
    let (nodes, warnings) = names(&j, &pc);
    // Main: both slots dropped (`m1, m2` only) with "not found" warnings.
    assert_eq!(nodes, ["m1", "m2", "w1", "b1"], "{warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("`wheel_F_stock` not found: the local `wheel_F_std` stands in")), "{warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("`body_stock` not found: the local `body_a` stands in")), "{warnings:?}");
    // Not using the feature: the configured parts, main's assembly.
    let (j, pc) = slot_vehicle("wheel_F_std", "body_b");
    let (nodes, warnings) = names(&j, &pc);
    assert_eq!(nodes, ["m1", "m2", "w1", "b2"]);
    assert!(!warnings.iter().any(|w| w.contains("stands in") || w.contains("not found")), "{warnings:?}");
}

/// S10: a file without the outer braces (as some community files are) loads as if
/// they were there; the same text with braces loads to the identical desc.
#[test]
fn s10_a_brace_less_file_loads_as_the_braced_one() {
    let body = r#""box": {"slotType": "main",
        "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 2}, ["a", 0, 0, 0], ["b", 1, 0, 0], ["c", 0, 1, 0]],
        "beams": [["id1:", "id2:"], {"beamSpring": 100000, "beamDamp": 10}, ["a", "b"], ["b", "c"], ["c", "a"]]}"#;
    let braced = load_jbeam(&[("v.jbeam", &format!("{{{body}}}"))], None).unwrap();
    let bare = load_jbeam(&[("v.jbeam", body)], None).unwrap();
    assert_eq!(serde_json::to_value(&bare).unwrap(), serde_json::to_value(&braced).unwrap());
    assert_eq!((braced.nodes.len(), braced.beams.len()), (3, 3));
}

/// S10: `read_vehicle_dir` takes `.JBEAM` (any case) as well as `.jbeam`; a vehicle
/// whose files are all lower-case `.jbeam` reads exactly the files main read.
#[test]
fn s10_jbeam_extension_is_case_insensitive() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("main_parity_profile_ext");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("veh/sub")).unwrap();
    std::fs::write(dir.join("veh/a.jbeam"), "{}").unwrap();
    std::fs::write(dir.join("veh/sub/b.jbeam"), "{}").unwrap();
    std::fs::write(dir.join("veh/notes.txt"), "x").unwrap();
    let files = |d: &std::path::Path| read_vehicle_dir(d, "veh").unwrap().into_iter().map(|f| f.0).collect::<Vec<_>>();
    let sep = std::path::MAIN_SEPARATOR;
    // Main's rule (extension == "jbeam"): these two.
    assert_eq!(files(&dir), [format!("veh{sep}a.jbeam"), format!("veh{sep}sub{sep}b.jbeam")]);
    std::fs::write(dir.join("veh/C.JBEAM"), "{}").unwrap();
    std::fs::write(dir.join("veh/d.JBeam"), "{}").unwrap();
    assert_eq!(
        files(&dir),
        [format!("veh{sep}C.JBEAM"), format!("veh{sep}a.jbeam"), format!("veh{sep}d.JBeam"), format!("veh{sep}sub{sep}b.jbeam")]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// S10: a beam's second and third `breakGroup` names are break groups too
/// (`extra_break_groups`), numbered in first-seen order with the others, which
/// shifts the numbers of groups named after them. A vehicle naming one group per
/// beam gets main's numbering (1-based, first-seen order).
#[test]
fn s10_second_and_third_break_group_names() {
    let vehicle = |groups: &str| {
        format!(
            r#"{{"v": {{"slotType": "main",
            "nodes": [["id", "posX", "posY", "posZ"], {{"nodeWeight": 2}}, ["a", 0, 0, 0], ["b", 1, 0, 0], ["c", 0, 1, 0], ["d", 0, 0, 1]],
            "beams": [["id1:", "id2:"], {{"beamSpring": 100000, "beamDamp": 10}},
                {{"breakGroup": {groups}}}, ["a", "b"],
                {{"breakGroup": "later"}}, ["b", "c"],
                {{"breakGroup": ""}}, ["c", "a"], ["a", "d"]]}}}}"#
        )
    };
    let d = load_jbeam(&[("v.jbeam", &vehicle(r#"["first", "second", "third"]"#))], None).unwrap();
    let jb = |i: usize| d.beams[i].jbeam.unwrap();
    assert_eq!(d.beams[0].break_group, 1);
    assert_eq!(jb(0).extra_break_groups, [2, 3], "the second and third names");
    assert_eq!(d.beams[1].break_group, 4, "`later` numbered after them (main: 2)");
    assert_eq!(d.break_groups, ["first", "second", "third", "later"]);
    // Not using the feature: one name per beam, main's numbering.
    let d = load_jbeam(&[("v.jbeam", &vehicle(r#""first""#))], None).unwrap();
    assert_eq!(d.beams.iter().map(|b| b.break_group).collect::<Vec<_>>(), [1, 2, 0, 0]);
    assert!(d.beams.iter().all(|b| b.jbeam.map_or(true, |j| j.extra_break_groups == [0, 0])));
}
