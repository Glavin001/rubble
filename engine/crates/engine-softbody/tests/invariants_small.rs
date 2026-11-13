//! Layer 1 of the vehicle test strategy (`engine/docs/vehicle-test-strategy.md`):
//! small cases, one feature each, compared with a closed form on the CPU
//! reference (the GPU runs the same bodies: `engine-webgpu/tests/invariants_small_gpu.rs`).
//! Every tolerance is derived in its comment; anything else is labelled a Choice.
//! Run: `cargo test --release -p engine-softbody --test invariants_small -- --nocapture`
//! (each test prints its report-table row).

mod invariants_common;
use invariants_common::*;

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitKind, StaticBox, Triangle, GRAVITY, NODE_IMMOVABLE, NODE_NO_GROUND, STEP,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{DVec3, Vec3};

// --- 1. One beam -----------------------------------------------------------------

/// Natural frequency of one node on one beam: the step's own frequency
/// ω_d = acos(1 − ω²h²/2)/h (the symplectic map's eigenvalue, exact) to 0.1 %
/// (zero crossings interpolated over 30 periods), and √(k/m) to the step's
/// known shift (ω_d/ω − 1 ≈ (ωh)²/24, 0.6 % at the skin's ωh = 0.38).
#[test]
fn beam_natural_frequency() {
    for (k, m) in [(SKIN_K, SKIN_M), (1.0e5, 1.0)] {
        let o = oscillator(&cpu, k, 0.0, m).unwrap();
        let shift = (o.omega_discrete / o.omega_continuous - 1.0).abs();
        let pass = rel(o.omega, o.omega_discrete) < 1e-3
            && rel(o.omega, o.omega_continuous) < shift + 1e-3;
        row(
            &format!("beam ω, k {k:.1e} m {m}"),
            "CPU",
            format!("{:.2} rad/s", o.omega),
            format!(
                "ω_d {:.2}, √(k/m) {:.2}",
                o.omega_discrete, o.omega_continuous
            ),
            pass,
        );
        assert!(pass);
    }
}

/// Damping: the decay rate σ of the peaks (least squares over 30 periods) against
/// the step's −ln√(1 − hc/m)/h (exact, 1 %: the peak refinement's error) and
/// the continuous ζω (to the step's shift, hc/(2m) relative); the logarithmic
/// decrement δ = σ·T against 2πζ/√(1 − ζ²). The BX's body and boot lid ratios.
#[test]
fn beam_damping_and_log_decrement() {
    for zeta in [BX_ZETA_BODY, BX_ZETA_LID] {
        let o = oscillator(&cpu, SKIN_K, zeta, SKIN_M).unwrap();
        let delta = o.sigma * 2.0 * std::f64::consts::PI / o.omega;
        let delta_c = 2.0 * std::f64::consts::PI * zeta / (1.0 - zeta * zeta).sqrt();
        let pass = rel(o.sigma, o.sigma_discrete) < 0.01
            && rel(o.sigma, o.sigma_continuous) < 0.03
            && rel(delta, delta_c) < 0.03;
        row(
            &format!("beam damping ζ {zeta}"),
            "CPU",
            format!("σ {:.2}/s, δ {:.4}", o.sigma, delta),
            format!(
                "σ_d {:.2}, ζω {:.2}, δ {:.4}",
                o.sigma_discrete, o.sigma_continuous, delta_c
            ),
            pass,
        );
        assert!(pass);
    }
}

/// Yield, plastic set (F − F_y)/k (with the hardening correction, see
/// [`yield_unload`]) to 1 % of the set, the first yield at F_y (to one step's
/// load increment, k·v·h = 2.5 N), unloading along slope k (0.1 %) and zero force
/// at the set (0.1 mm). RoR plain and JBeam NORMAL beams.
#[test]
fn beam_yield_plastic_set_and_unloading() {
    for jbeam in [false, true] {
        let y = yield_unload(&cpu, jbeam).unwrap();
        let pass = rel(y.set, y.set_closed) < 0.01
            && (y.yield_force - 1000.0).abs() < 2.6
            && rel(y.unload_slope, 1.0e5) < 1e-3
            && (y.zero_force_at - y.set).abs() < 1e-4;
        row(
            &format!("yield/set/unload {}", if jbeam { "JBeam" } else { "RoR" }),
            "CPU",
            format!(
                "Fy {:.1} N, set {:.3} mm, slope {:.0}, F=0 at {:.3} mm",
                y.yield_force,
                y.set * 1e3,
                y.unload_slope,
                y.zero_force_at * 1e3
            ),
            format!("1000 N, {:.3} mm, 1e5, set", y.set_closed * 1e3),
            pass,
        );
        assert!(pass);
    }
}

/// Break force: a brittle JBeam beam breaks at S (extension S/k, one increment),
/// a ductile RoR beam where its hardened yield meets its weakened strength (1 %).
/// And the RoR rule's documented exception: a beam with an infinite deform
/// threshold never breaks (its tests start at the deform threshold).
#[test]
fn beam_break_force() {
    for ductile in [false, true] {
        let (x, f, xc, fc) = break_force(&cpu, ductile).unwrap();
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
            "CPU",
            format!("x {:.3} mm, F {:.0} N", x * 1e3, f),
            format!("x {:.3} mm, F {:.0} N", xc * 1e3, fc),
            pass,
        );
        assert!(pass);
    }
    // RoR: deform ∞ never breaks.
    let k = 1.0e5;
    let s = driven(plain(0, 1, k, 0.0, 1.0, f32::INFINITY, 2000.0), 1.0, 0.05);
    let mut sim = Cpu(s);
    let rows = pull(&mut sim, k, 1.0, 1.0, 0.05, 0.05, 1);
    assert!(
        !rows.last().unwrap().3,
        "RoR rule: infinite deform never breaks"
    );
}

/// The elastic energy of a pulled beam equals F²/2k (the CPU's ledger measure,
/// 10⁻⁶), and the work done on it to 3 cm (through a yield) equals the stored
/// energy plus the plastic dissipation ∫Tt dL (1 %).
#[test]
fn beam_energy_stored_and_dissipated() {
    let (u, f2, w, budget) = energy_stored(&cpu).unwrap();
    let pass = rel(u, f2) < 1e-6 && rel(w, budget) < 0.01;
    row(
        "beam energy",
        "CPU",
        format!("U {:.4} J, W {:.3} J", u, w),
        format!("F²/2k {:.4} J, U + ∫Tt dL {:.3} J", f2, budget),
        pass,
    );
    assert!(pass);
}

// --- 2. Beam types -----------------------------------------------------------------

/// SUPPORT (spec §4): −k·e in compression, no force stretched, broken past
/// (1 + b)·ℓ₀ (b = beamLongBound). Swept at 0.05 m/s; force to 10⁻³ relative.
#[test]
fn support_beam_pushes_only_and_breaks_by_length() {
    let k = 1.0e5;
    let s = make_driven(jbeam_pair(
        r#""beamType": "|SUPPORT", "beamSpring": 100000, "beamDamp": 0, "beamDeform": "FLT_MAX",
           "beamStrength": "FLT_MAX", "beamLongBound": 0.5"#,
        1.0,
    ));
    let rows = sweep(s, -0.02, 0.6, 0.05);
    let mut worst = 0.0f64;
    for &(x, f, broken) in &rows {
        if broken {
            break;
        }
        let want = if x < 0.0 { -k * x } else { 0.0 };
        worst = worst.max((f + want).abs() - 0.0);
    }
    // (Tension positive in `sweep`: compression gives f = k·x < 0.)
    let brk = rows.iter().find(|r| r.2).map(|r| r.0);
    let pass = worst < 1e-3 * k * 0.02 && brk.is_some_and(|x| (x - 0.5).abs() < 1e-4);
    row(
        "SUPPORT",
        "CPU",
        format!(
            "force error {worst:.3} N, break at +{:.4} m",
            brk.unwrap_or(f64::NAN)
        ),
        "0, +0.5 m (b·ℓ₀)".into(),
        pass,
    );
    assert!(pass);
}

/// BOUNDED (spec §5): k inside the bounds; past a bound by o the stiffness blends
/// to the limit spring, k + (k_lim − k)·min(o/z, 1), on the whole extension.
#[test]
fn bounded_beam_follows_its_blended_law() {
    let (k, kl, l, s_, z) = (1.0e4f64, 1.0e6f64, 0.1f64, 0.05f64, 0.01f64);
    let s = make_driven(jbeam_pair(
        &format!(
            r#""beamType": "|BOUNDED", "beamSpring": {k}, "beamDamp": 0, "beamLimitSpring": {kl},
               "beamLimitDamp": 0, "beamLongBound": {l}, "beamShortBound": {s_}, "boundZone": {z},
               "beamStrength": "FLT_MAX", "beamDeform": "FLT_MAX""#
        ),
        1.0,
    ));
    let rows = sweep(s, -0.08, 0.13, 0.05);
    let law = |x: f64| {
        let o = if x > l {
            x - l
        } else if x < -s_ {
            -x - s_
        } else {
            0.0
        };
        let kk = k + (kl - k) * (o / z).min(1.0);
        kk * x
    };
    let worst = rows
        .iter()
        .map(|&(x, f, _)| (f - law(x)).abs() / law(x).abs().max(1.0))
        .fold(0.0, f64::max);
    let pass = worst < 2e-3;
    row(
        "BOUNDED",
        "CPU",
        format!("worst relative force error {worst:.2e}"),
        "≤ 2e-3 (f32 lengths)".into(),
        pass,
    );
    assert!(pass);
}

/// ANISOTROPIC (spec §20): the bilinear oscillator's period, half on each side,
/// T = π√(μ/k) + π√(μ/kₑ); against the step's own (0.2 %) and the continuous
/// (the step's shift).
#[test]
fn anisotropic_beam_is_a_bilinear_oscillator() {
    let (t, tc, td) = anisotropic_period(&cpu).unwrap();
    let pass = rel(t, td) < 2e-3 && rel(t, tc) < rel(td, tc) + 2e-3;
    row(
        "ANISOTROPIC period",
        "CPU",
        format!("{:.4} ms", t * 1e3),
        format!("step {:.4} ms, continuous {:.4} ms", td * 1e3, tc * 1e3),
        pass,
    );
    assert!(pass);
}

/// PRESSURED (spec §21): F = (p_abs,0·ℓ₀/(ℓ₀ − c·(ℓ₀ − ℓ)) − p_atm)·A (pressure ×
/// area of the isothermal cylinder), compressed and stretched; 10⁻³ relative.
#[test]
fn pressured_beam_follows_the_gas_law() {
    let (p_gauge, area, coef) = (2.0e5f64, 0.002f64, 0.8f64);
    let s = make_driven(jbeam_pair(
        &format!(
            r#""beamType": "|PRESSURED", "beamSpring": 0, "beamDamp": 0, "pressure": {p_gauge},
               "surface": {area}, "volumeCoef": {coef}, "beamStrength": "FLT_MAX", "beamDeform": "FLT_MAX""#
        ),
        1.0,
    ));
    let rows = sweep(s, -0.5, 0.3, 0.2);
    let atm = 101_325.0;
    let law = |x: f64| {
        let l = 1.0 + x;
        ((atm + p_gauge) * 1.0 / (1.0 - coef * (1.0 - l)) - atm) * area
    };
    let worst = rows
        .iter()
        .map(|&(x, f, _)| (-f - law(x)).abs() / law(x).abs())
        .fold(0.0, f64::max);
    let pass = worst < 1e-3;
    row(
        "PRESSURED gas law",
        "CPU",
        format!("worst relative error {worst:.2e} over ℓ 0.5–1.3 ℓ₀"),
        "≤ 1e-3".into(),
        pass,
    );
    assert!(pass);
}

/// An LBEAM triangle in free flight, spun and shaken (no damping): momentum
/// changes by M·g·t to f32 rounding; angular momentum about the centre of mass
/// is kept (10⁻³: rounding of 4000 steps); the energy (kinetic + gravity + the
/// LBEAM's and arms' ½k(ℓ − L)²) does not drift (the last tenth's mean within
/// 0.1 % of the first's) and oscillates within the symplectic bound ω_max·h/2 of
/// it (0.5 % here: k chosen for ω_max·h ≈ 0.01). A follower force (the pre-
/// 2026-11-05 LBEAM) drifts.
#[test]
fn lbeam_conserves_momentum_angular_momentum_and_energy() {
    let text = r#"{ "t": { "information": {"name": "l"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false, "selfCollision": false},
            ["a", 1, 0, 0], ["b", 0, 0, 1], ["c", 0, 0, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 200, "beamDamp": 0, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"},
            ["a", "c"], ["b", "c"],
            {"beamType": "|LBEAM", "beamSpring": 200, "beamDamp": 0}, ["a", "b", "c"]] } }"#;
    let mut s = spawn_jbeam(text);
    assert!(
        s.beams.iter().any(|b| b.kind == ExplicitKind::JbeamLbeam),
        "the LBEAM loaded"
    );
    for n in &mut s.nodes {
        n.flags |= NODE_NO_GROUND;
    }
    s.nodes[0].vel = Vec3::new(0.3, 0.1, 0.5);
    s.nodes[1].vel = Vec3::new(-0.2, 0.4, 0.0);
    s.nodes[2].vel = Vec3::new(0.0, -0.1, -0.3);
    // (One step first: a spawn's first step integrates no force.)
    s.step();
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    let mass: f64 = w.vehicles[0].nodes.iter().map(|n| n.mass as f64).sum();
    let ang = |v: &ExplicitBeams| {
        let p = v.positions();
        let c: DVec3 = v
            .nodes
            .iter()
            .zip(&p)
            .map(|(n, x)| x.as_dvec3() * n.mass as f64)
            .sum::<DVec3>()
            / mass;
        let vc: DVec3 = momentum(v) / mass;
        v.nodes
            .iter()
            .zip(&p)
            .map(|(n, x)| (x.as_dvec3() - c).cross((n.vel.as_dvec3() - vc) * n.mass as f64))
            .sum::<DVec3>()
    };
    let p0 = momentum(&w.vehicles[0]);
    let l0 = ang(&w.vehicles[0]);
    let steps = 4000;
    let mut e = Vec::with_capacity(steps);
    for _ in 0..steps {
        w.step();
        e.push(internal_energy(&w.vehicles[0]));
    }
    let v = &w.vehicles[0];
    let dp =
        momentum(v) - p0 - DVec3::new(0.0, mass * GRAVITY as f64 * STEP as f64 * steps as f64, 0.0);
    let dl = (ang(v) - l0).length() / l0.length();
    let tenth = steps / 10;
    let first = e[..tenth].iter().sum::<f64>() / tenth as f64;
    let last = e[steps - tenth..].iter().sum::<f64>() / tenth as f64;
    let wh = omega_max(v) * STEP as f64;
    let swing = e.iter().map(|x| (x - first).abs()).fold(0.0, f64::max);
    let pass = dp.length() < 1e-5 * mass
        && dl < 1e-3
        && (last - first).abs() < 1e-3 * first
        && swing <= wh / 2.0 * first;
    row(
        "LBEAM conservation",
        "CPU",
        format!(
            "Δp {:.1e} kg·m/s, ΔL/L {:.1e}, drift {:.2e} J, swing {:.2e} J",
            dp.length(),
            dl,
            last - first,
            swing
        ),
        format!("M·g·t, 0, 0, ≤ {:.2e} J (ω_max·h/2·E)", wh / 2.0 * first),
        pass,
    );
    assert!(pass);
}

/// Winter's effective width (`authoring::vehicle::sheet_ultimate_stress`) against
/// an independent f64 evaluation of its closed form, σ_cr = kπ²E/(12(1 − ν²))·(t/b)²,
/// σ_u = σ_y·√(σ_cr/σ_y)·(1 − 0.22·√(σ_cr/σ_y)), capped at σ_y; and the doc's
/// worked value, 0.8 mm on 0.2 m bays: σ_cr 12 MPa, σ_u 48 MPa.
#[test]
fn winter_effective_width_closed_form() {
    use engine_softbody::authoring::vehicle::sheet_ultimate_stress;
    let (e, sy, nu, kb) = (210e9f64, 210e6f64, 0.3f64, 4.0f64);
    let mut worst = 0.0f64;
    for t in [0.6e-3f64, 0.8e-3, 1.0e-3, 1.6e-3] {
        for b in [0.05f64, 0.1, 0.2, 0.3] {
            let cr =
                kb * std::f64::consts::PI.powi(2) * e / (12.0 * (1.0 - nu * nu)) * (t / b).powi(2);
            let lam = (cr / sy).sqrt();
            let want = sy * (lam * (1.0 - 0.22 * lam)).min(1.0);
            let got = sheet_ultimate_stress(e as f32, sy as f32, t as f32, b as f32) as f64;
            worst = worst.max(rel(got, want));
        }
    }
    let worked = sheet_ultimate_stress(210e9, 210e6, 0.8e-3, 0.2) as f64;
    let pass = worst < 1e-5 && (worked / 1e6 - 48.0).abs() < 1.0;
    row(
        "Winter σ_u closed form",
        "CPU",
        format!("worst {worst:.1e}, 0.8 mm/0.2 m {:.1} MPa", worked / 1e6),
        "1e-5, 48 MPa".into(),
        pass,
    );
    assert!(pass);
}

// --- 3. Contact ----------------------------------------------------------------------

/// A node dropped onto the ground over a grid of speeds (0.5–100 m/s) and
/// crossing phases: never deeper than v·h (+ g·h², the recurrence in
/// [`ground_drop`]), and the rebound at most 0.2·v (the documented 80 % removal
/// of the approach velocity).
#[test]
fn node_on_ground_never_tunnels_and_rebounds_at_most_a_fifth() {
    let mut worst_depth = 0.0f64;
    let mut worst_e = 0.0f64;
    for v in [0.5f32, 1.0, 2.0, 5.0, 10.0, 15.6, 22.2, 30.0, 50.0, 100.0] {
        for phase in 0..7 {
            let h0 = 0.05 + v * STEP * (phase as f32 / 7.0);
            let (d, r, vi) = ground_drop(&cpu, v, h0).unwrap();
            let bound = vi * STEP as f64 + 10.0 * (STEP as f64).powi(2);
            worst_depth = worst_depth.max(d / bound);
            worst_e = worst_e.max(r / vi);
        }
    }
    let pass = worst_depth <= 1.0 && worst_e <= 0.2 + 1e-4;
    row(
        "ground: depth / v·h, rebound / v",
        "CPU",
        format!("{worst_depth:.3}, {worst_e:.4}"),
        "≤ 1, ≤ 0.2".into(),
        pass,
    );
    assert!(pass);
}

/// A two-node bar dropped flat and tilted onto the ground: neither node below
/// −v·h, the bar's centre of mass rebounds at most 0.2·v, and its mechanical
/// energy (kinetic, gravity, ½k·e²) never rises (the contact only removes energy;
/// tolerance: the symplectic oscillation of its beam, ω·h/2 of the beam's energy).
#[test]
fn two_node_bar_on_ground_never_tunnels_or_gains_energy() {
    let g = GRAVITY;
    let mut worst = (0.0f64, 0.0f64, f64::MIN);
    for v in [2.0f32, 10.0, 30.0] {
        for tilt in [0.0f32, 0.05, 0.3] {
            let mk = |x: f32, y: f32| {
                let mut q = node(Vec3::new(x, y, 0.0), 2.0, 0);
                q.flags = 0;
                q.vel = Vec3::new(0.0, -v, 0.0);
                q.force = Vec3::new(0.0, 2.0 * g, 0.0);
                q
            };
            let (y0, y1) = (0.05, 0.05 + tilt);
            let len = (1.0f32 + tilt * tilt).sqrt();
            let mut s = system(
                vec![mk(0.0, y0), mk(1.0, y1)],
                vec![plain(0, 1, 2.0e5, 20.0, len, f32::INFINITY, f32::INFINITY)],
            );
            s.ground = 0.0;
            let mut w = ExplicitBeamsWorld::new(vec![s]);
            w.enable_ledger();
            let e0 = w.ledger.as_ref().unwrap().start;
            let mut most = f64::MIN;
            let mut deepest = 0.0f64;
            let mut cm_up = 0.0f64;
            for _ in 0..1200 {
                w.step();
                let v_ = &w.vehicles[0];
                for n in &v_.nodes {
                    deepest = deepest.max(-(v_.origin.y + n.pos.y) as f64);
                }
                cm_up = cm_up.max(momentum(v_).y / 4.0);
                most = most.max(w.ledger.as_ref().unwrap().mechanical() - e0);
            }
            let omega_h = (2.0e5f64 / 1.0).sqrt() * STEP as f64;
            let osc = 0.5 * 4.0 * (v as f64).powi(2);
            // Its impact speed after falling from 5 cm (+ the tilt).
            let vi = ((v as f64).powi(2) + 2.0 * -g as f64 * (0.05 + tilt as f64)).sqrt();
            worst.0 = worst.0.max(deepest / (vi * STEP as f64));
            worst.1 = worst.1.max(cm_up / vi);
            worst.2 = worst.2.max(most / (omega_h / 2.0 * osc));
        }
    }
    let pass = worst.0 <= 1.0 && worst.1 <= 0.2 + 1e-3 && worst.2 <= 1.0;
    row(
        "two-node bar on ground",
        "CPU",
        format!(
            "depth/vh {:.3}, rebound/v {:.4}, energy rise/bound {:.3}",
            worst.0, worst.1, worst.2
        ),
        "≤ 1, ≤ 0.2, ≤ 1".into(),
        pass,
    );
    assert!(pass);
}

/// A static box: the node's approach velocity loses 80 % each step and no depth
/// push is applied (p = 0), so it stops inside at depth ≤ 1.25·v·h (geometric
/// sum 1 + 0.2 + 0.04 + …) and never bounces; it cannot leave by the far face
/// while 1.25·v·h < H, the half-thickness. Speeds up to 0.9 of that bound, 4
/// phases: no tunnelling and no rebound. Past it (1.5×) reported.
#[test]
fn node_into_box_stops_inside_its_tunnelling_bound() {
    let half = 0.05f32;
    let vmax = half / (1.25 * STEP);
    let mut fails = Vec::new();
    let mut past = Vec::new();
    for frac in [0.1f32, 0.3, 0.5, 0.7, 0.9, 1.5] {
        let v = frac * vmax;
        for phase in 0..4 {
            let mut q = node(
                Vec3::new(0.0, 1.05 + v * STEP * phase as f32 / 4.0, 0.0),
                1.0,
                0,
            );
            q.flags = 0;
            q.vel = Vec3::new(0.0, -v, 0.0);
            let mut s = system(vec![q], vec![]);
            s.ground = -1.0e3;
            s.boxes.push(StaticBox::centered(
                Vec3::new(0.0, 1.0 - half, 0.0),
                Vec3::new(2.0, half, 2.0),
            ));
            for _ in 0..400 {
                s.step();
            }
            let y = s.nodes[0].pos.y;
            let through = y < 1.0 - 2.0 * half + 1e-4;
            if frac <= 1.0 && (through || s.nodes[0].vel.y > 1e-3) {
                fails.push((v, phase, y));
            }
            if frac > 1.0 && through {
                past.push(phase);
            }
        }
    }
    let pass = fails.is_empty();
    row(
        "box tunnelling bound 0.8·H/h",
        "CPU",
        format!(
            "no tunnel ≤ 0.9× ({} fails); at 1.5× {} of 4 phases tunnel",
            fails.len(),
            past.len()
        ),
        format!("v < {vmax:.0} m/s for H = 5 cm"),
        pass,
    );
    assert!(pass, "{fails:?}");
}

/// A node against another vehicle's triangle (vehicle contact), over a grid of
/// approach speeds and angles and crossing phases. The contact acts in a band
/// |δ| ≤ r (r = 2 cm) and pushes to the side the node is on, so it is never
/// pushed through while its normal travel per step v·cosθ·h ≤ r (it must land
/// in the near half of the band): v_n ≤ r/h = 40 m/s. Both contact rules (RoR's
/// and the two-body rule). Normal speeds up to 0.95·r/h must not pass; above,
/// reported.
#[test]
fn node_never_passes_a_triangle_below_the_band_speed() {
    let r = 0.02f32;
    let vlim = r / STEP;
    let mut fails = Vec::new();
    let mut above = (0usize, 0usize);
    for two_body in [false, true] {
        for vn in [1.0f32, 5.0, 10.0, 15.6, 22.2, 30.0, 38.0, 50.0, 70.0] {
            for deg in [0.0f32, 30.0, 60.0] {
                for phase in 0..5 {
                    let th = deg.to_radians();
                    let vel = Vec3::new(vn * th.tan(), -vn, 0.0);
                    let heavy = |p: Vec3| {
                        let mut q = node(p, 1.0e6, 0);
                        q.flags = 0;
                        q
                    };
                    let mut tri = ExplicitBeams::empty(exact());
                    tri.ground = -1.0e3;
                    tri.nodes = vec![
                        heavy(Vec3::new(-3.0, 3.0, -3.0)),
                        heavy(Vec3::new(3.0, 3.0, -3.0)),
                        heavy(Vec3::new(0.0, 3.0, 6.0)),
                    ];
                    tri.triangles.push(Triangle {
                        nodes: [0, 1, 2],
                        wait: 0,
                        idle: 0,
                    });
                    tri.two_body_contact = two_body;
                    tri.finish();
                    let start = 3.0 + r + 0.001 + vn * STEP * phase as f32 / 5.0;
                    let mut q = node(Vec3::new(-0.5 * vel.x * 0.01, start, 0.0), 1.0, 0);
                    q.flags = 0;
                    q.vel = vel;
                    let mut p = ExplicitBeams::empty(exact());
                    p.ground = -1.0e3;
                    p.nodes = vec![q];
                    p.two_body_contact = two_body;
                    p.finish();
                    let mut w = ExplicitBeamsWorld::new(vec![tri, p]);
                    w.throttle = false;
                    for _ in 0..200 {
                        w.step();
                    }
                    let plane = w.vehicles[0].positions()[0].y;
                    let y = w.vehicles[1].positions()[0].y;
                    let passed = y < plane;
                    if passed && vn <= 0.95 * vlim {
                        fails.push((two_body, vn, deg, phase));
                    }
                    if vn > vlim {
                        above.1 += 1;
                        above.0 += passed as usize;
                    }
                }
            }
        }
    }
    let pass = fails.is_empty();
    row(
        "node vs triangle, band speed r/h",
        "CPU",
        format!(
            "{} passes ≤ 0.95·r/h; above r/h {} of {} pass",
            fails.len(),
            above.0,
            above.1
        ),
        format!("0 below {vlim:.0} m/s normal"),
        pass,
    );
    assert!(pass, "{fails:?}");
}

// --- 4. Waves and ripple ---------------------------------------------------------------

/// A struck bar's stress wave travels at c₀ = l·√(k/m) = √(E/ρ) of the discrete
/// bar (2 %: the Airy front's centre over 30–90 nodes). At the sedan's skin (k
/// 8.7e5 N/m, 1.53 kg, 0.21 m) and at the step's stability bound for a bar
/// (ω_max·h = 2√(k/m)·h < 2: k < m/h²; at 0.95 of it). Steel's √(E/ρ) is 5,170
/// m/s: reported against both (what "rippling" means: waves an order of magnitude
/// slower than steel's cross the car in milliseconds, not microseconds).
#[test]
fn struck_bar_wave_speed() {
    let steel = (210e9f64 / 7850.0).sqrt();
    let (l, m) = (0.21f32, SKIN_M);
    let cap = 0.95 * m / (STEP * STEP);
    for (label, k) in [("skin k", SKIN_K), ("0.95 bound", cap)] {
        let (c, c0) = wave_speed(&cpu, k, m, l).unwrap();
        let pass = rel(c, c0) < 0.02;
        row(
            &format!("bar wave speed, {label}"),
            "CPU",
            format!("{c:.1} m/s ({:.1} % of steel's)", 100.0 * c / steel),
            format!("l√(k/m) {c0:.1} m/s"),
            pass,
        );
        assert!(pass);
    }
}

/// A bar's lowest mode (the ripple of a 1 m span of 0.21 m bays at the skin's k
/// and m) with the BX body's beam ratio ζ_b = 0.0275: frequency against the
/// step's (0.5 %), decay rate against the step's (3 %), and the continuous modal
/// ratio ζ₁ = ζ_b·ω₁/ω_p (stiffness-proportional dashpots damp a low mode far less
/// than their pair ratio).
#[test]
fn bar_lowest_mode_decays_at_the_stiffness_proportional_rate() {
    let o = chain_mode(&cpu, 4, SKIN_K, BX_ZETA_BODY, SKIN_M, 0.21).unwrap();
    let [w, wd, s, sd, z1, zb] = o;
    let pass = rel(w, wd) < 5e-3 && rel(s, sd) < 0.03;
    row(
        "bar mode 1 decay (ripple)",
        "CPU",
        format!(
            "f {:.1} Hz, σ {:.2}/s (ζ₁ {:.4}, decay to 1 % in {:.0} ms)",
            w / (2.0 * std::f64::consts::PI),
            s,
            s / w,
            1e3 * (100f64).ln() / s
        ),
        format!(
            "f {:.1} Hz, σ {sd:.2}/s; ζ₁ = ζ_b·ω₁/ω_p = {z1:.4} (ζ_b {zb})",
            wd / (2.0 * std::f64::consts::PI)
        ),
        pass,
    );
    assert!(pass);
}

// --- 5. Hardware -------------------------------------------------------------------

/// A plate on two hinges and a latch, pulled at a point: the latch lets go at
/// F = S·a/p (statics of a plate on three supports, plus its own weight's share),
/// 2 %; its whole break group goes with it, and the hinges hold.
#[test]
fn latch_releases_at_its_statically_shared_strength() {
    let (f, fc, latch, hinges) = latch_release(&cpu).unwrap();
    let pass = rel(f, fc) < 0.02 && latch && hinges == 0;
    row(
        "latch release",
        "CPU",
        format!("{f:.0} N, group broken {latch}, hinge beams broken {hinges}"),
        format!("{fc:.0} N, true, 0"),
        pass,
    );
    assert!(pass);
}

// --- 6. Conservation per feature -----------------------------------------------------

/// Each feature's small system in free flight (no ground, no air): momentum
/// changes by exactly M·g·t (f32 rounding, 10⁻⁵ of M·g·t) and the ledger's
/// mechanical energy never rises (0.5 %: the symplectic oscillation, ω_max·h ≤
/// 0.01 here; damped systems strictly fall). Plain with damping, support,
/// bounded, anisotropic, a break group breaking.
#[test]
fn free_flight_conserves_momentum_and_never_gains_energy() {
    let cases: Vec<(&str, ExplicitBeams)> = vec![
        ("plain + damping", {
            let mut s = jbeam_pair(
                r#""beamSpring": 200, "beamDamp": 0.5, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX""#,
                1.0,
            );
            s.nodes[1].vel = Vec3::new(0.3, 0.2, -0.1);
            s
        }),
        ("support", {
            let mut s = jbeam_pair(
                r#""beamType": "|SUPPORT", "beamSpring": 200, "beamDamp": 0, "beamLongBound": 10, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX""#,
                1.0,
            );
            s.nodes[1].vel = Vec3::new(-0.3, 0.0, 0.0);
            s
        }),
        ("bounded", {
            let mut s = jbeam_pair(
                r#""beamType": "|BOUNDED", "beamSpring": 100, "beamDamp": 0, "beamLimitSpring": 200, "beamLimitDamp": 0, "beamLongBound": 0.05, "beamShortBound": 0.05, "boundZone": 0.02, "beamStrength": "FLT_MAX""#,
                1.0,
            );
            s.nodes[1].vel = Vec3::new(0.4, 0.0, 0.0);
            s
        }),
        ("anisotropic", {
            let mut s = jbeam_pair(
                r#""beamType": "|ANISOTROPIC", "beamSpring": 100, "beamDamp": 0, "springExpansion": 200, "dampExpansion": 0, "transitionZone": 0.05, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX""#,
                1.0,
            );
            s.nodes[1].vel = Vec3::new(0.4, 0.0, 0.0);
            s
        }),
        ("break group", {
            let mut s = spawn_jbeam(engine_softbody::jbeam_fixture::PART2_GROUPS);
            for n in &mut s.nodes {
                n.flags |= NODE_NO_GROUND;
            }
            s.nodes[1].vel = Vec3::new(40.0, 0.0, 0.0);
            s
        }),
    ];
    for (label, mut s) in cases {
        for n in &mut s.nodes {
            n.flags |= NODE_NO_GROUND;
            n.flags &= !NODE_IMMOVABLE;
        }
        // (One step first: a spawn's first step integrates no force.)
        s.step();
        let mut w = ExplicitBeamsWorld::new(vec![s]);
        let mass: f64 = w.vehicles[0].nodes.iter().map(|n| n.mass as f64).sum();
        let p0 = momentum(&w.vehicles[0]);
        let e0 = internal_energy(&w.vehicles[0]);
        let wh = omega_max(&w.vehicles[0]) * STEP as f64;
        let mut most = f64::MIN;
        let steps = 2000;
        for _ in 0..steps {
            w.step();
            most = most.max(internal_energy(&w.vehicles[0]) - e0);
        }
        let ke0 = e0;
        let dp = momentum(&w.vehicles[0])
            - p0
            - DVec3::new(0.0, mass * GRAVITY as f64 * STEP as f64 * steps as f64, 0.0);
        let mgt = mass * -GRAVITY as f64 * STEP as f64 * steps as f64;
        // (The bound is reached by a single mode: 1 % for rounding.)
        let pass = dp.length() < 1e-5 * mgt && most <= 1.01 * wh / 2.0 * ke0;
        row(
            &format!("free flight: {label}"),
            "CPU",
            format!(
                "Δp {:.1e} kg·m/s, internal energy rise {:.2e} J",
                dp.length(),
                most
            ),
            format!(
                "≤ {:.1e}, ≤ {:.1e} (ω_max·h/2·E)",
                1e-5 * mgt,
                wh / 2.0 * ke0
            ),
            pass,
        );
        assert!(pass, "{label}");
    }
}
