//! `meshwheels`, `meshwheels2` and `shocks2` in the explicit-beams solver, each
//! checked from its definition (`engine/docs/explicit-beams-wheels2-shocks2-spec.md`,
//! its §8 plan where it needs no reference run): the mesh wheels' spawn (rim rates,
//! spoke bounds, thresholds, section order, legacy numbers, masses), and shocks2's
//! rates (the spec's worked checks), bounds options, force at prescribed rates,
//! break and line validation.

use engine_softbody::explicit_beams::*;
use engine_softbody::truck::parse_truck;
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
    centred_damping: false,
};

fn spawn(src: &str) -> ExplicitBeams {
    ExplicitBeams::from_truck(src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap()
}

/// A chassis tetrahedron, axle nodes 0 and 1, then `extra` (directives and the wheel).
fn wheel_truck(extra: &str) -> String {
    format!(
        "wheel test
globals
1000, 0, tracks/semi
nodes
0, 0, 0.5, 0
1, 0, 0.5, 0.3
2, 0.5, 0.5, 0.25
3, 0, 1.0, 0.15
beams
0, 1
0, 2
1, 2
0, 3
1, 3
2, 3
{extra}
end
"
    )
}

const WHEELS: &str = "wheels\n0.45, 0.2, 4, 0, 1, 2, 1, 0, 3, 30, 50000, 300\n";
const MESH: &str =
    "meshwheels\n0.45, 0.3, 0.2, 4, 0, 1, 2, 1, 0, 3, 30, 50000, 300, l, rim.mesh, mat\n";
const MESH2: &str =
    "meshwheels2\n0.45, 0.3, 0.2, 4, 0, 1, 2, 1, 0, 3, 30, 50000, 300, l, rim.mesh, mat\n";

#[test]
fn meshwheels_is_wheels_with_another_layout() {
    let mut a = spawn(&wheel_truck(WHEELS));
    let mut b = spawn(&wheel_truck(MESH));
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.beams, b.beams);
    assert_eq!(a.nodes.len(), 4 + 8);
    assert_eq!(a.beams.len(), 6 + 9 * 4);
    // The same trajectory (dropped onto the ground, 1 s).
    a.run(1.0);
    b.run(1.0);
    assert_eq!(a.positions(), b.positions());
}

#[test]
fn meshwheels2_rim_takes_the_unscaled_beam_defaults_and_spokes_long_bound() {
    let n = 4;
    // No beam defaults: rim 9e6 / 12000.
    let s = spawn(&wheel_truck(MESH2));
    let per_ray = &s.beams[..9];
    for (r, b) in per_ray.iter().enumerate() {
        let (k, c) = if (4..8).contains(&r) {
            (9_000_000.0, 12_000.0)
        } else {
            (50_000.0, 300.0)
        };
        assert_eq!((b.k, b.c), (k, c), "row {}", r + 1);
    }
    assert_eq!(per_ray[0].kind, ExplicitKind::LimitedSpring);
    assert_eq!(per_ray[1].kind, ExplicitKind::LimitedSpring);
    assert_eq!(per_ray[0].long_bound, 0.15f64 as f32);
    assert_eq!(per_ray[1].long_bound, 0.15f64 as f32);
    assert_eq!(per_ray[0].short_bound, 0.66);
    assert_eq!(
        (per_ray[0].limit_k, per_ray[0].limit_c),
        (9_000_000.0, 12_000.0)
    );
    assert_eq!(per_ray[8].ty, ExplicitType::Virtual);
    // Node 2 is nearer A2 (node 1): the rigidity beam braces the inner ring.
    assert_eq!((per_ray[8].a, per_ray[8].b), (2, 5));
    assert_eq!(s.beams.len(), 6 + 9 * n);

    // The beam defaults at the line, and their scale (not applied to the rim
    // rates or the break threshold, applied to the deform threshold).
    let src = wheel_truck(&format!(
        "set_beam_defaults_scale 2, 2, 2, 2\nset_beam_defaults 700000, 350, 60000000, 80000000\n{MESH2}"
    ));
    let s = spawn(&src);
    for (r, b) in s.beams[..8].iter().enumerate() {
        let (k, c) = if r >= 4 {
            (700_000.0, 350.0)
        } else {
            (50_000.0, 300.0)
        };
        assert_eq!((b.k, b.c), (k, c), "row {}", r + 1);
        assert_eq!(b.strength, 80_000_000.0);
        assert_eq!(
            (b.tc, b.tt, b.gate),
            (120_000_000.0, 120_000_000.0, 120_000_000.0)
        );
    }
    // meshwheels2 with beam defaults equal to its tyre rates differs from
    // meshwheels only in the spokes' long bound.
    let same = "set_beam_defaults 50000, 300\n";
    let m1 = spawn(&wheel_truck(&format!("{same}{MESH}")));
    let m2 = spawn(&wheel_truck(&format!("{same}{MESH2}")));
    assert_eq!(m1.nodes, m2.nodes);
    for (i, (x, y)) in m1.beams.iter().zip(&m2.beams).enumerate() {
        let spoke = i < 36 && (i % 9 == 0 || i % 9 == 1);
        let mut y = *y;
        if spoke {
            assert_eq!(y.long_bound, 0.15f64 as f32);
            y.long_bound = 0.0;
        }
        assert_eq!(*x, y, "beam {i}");
    }
}

#[test]
fn mesh_wheel_lines_need_sixteen_arguments_and_two_axle_nodes() {
    let short = "meshwheels2\n0.45, 0.3, 0.2, 4, 0, 1, 2, 1, 0, 3, 30, 50000, 300, l, rim.mesh\n";
    let s = spawn(&wheel_truck(short));
    assert_eq!(s.nodes.len(), 4);
    let same =
        "meshwheels2\n0.45, 0.3, 0.2, 4, 0, 0, 2, 1, 0, 3, 30, 50000, 300, l, rim.mesh, mat\n";
    let s = spawn(&wheel_truck(same));
    assert_eq!(s.nodes.len(), 4);
    // wheels2 stays unsupported.
    let w2 = "wheels2\n0.3, 0.45, 0.2, 4, 0, 1, 9999, 1, 0, 3, 30, 50000, 300, 400000, 500\n";
    let e = ExplicitBeams::from_truck(&wheel_truck(w2), Vec3::ZERO, Vec3::ZERO, EXACT).unwrap_err();
    assert!(e.contains("wheels2"), "{e}");
}

#[test]
fn wheel_sections_spawn_in_section_order_and_legacy_numbers_follow_the_file() {
    // File order: meshwheels2 (R 0.3), wheels (R 0.4), meshwheels (R 0.5); 4 rays.
    let src = wheel_truck(
        "meshwheels2\n0.3, 0.2, 0.2, 4, 0, 1, 9999, 1, 0, 3, 30, 50000, 300, l, m, m\n\
         wheels\n0.4, 0.2, 4, 0, 1, 9999, 1, 0, 3, 30, 50000, 300\n\
         meshwheels\n0.5, 0.2, 0.2, 4, 0, 1, 9999, 1, 0, 3, 30, 50000, 300, l, m, m\n\
         beams\n3, 4\n3, 13\n3, 21\n",
    );
    let s = spawn(&src);
    assert_eq!(s.nodes.len(), 4 + 3 * 8);
    let radius = |i: usize| (s.nodes[i].pos - s.nodes[0].pos).length();
    assert!((radius(4) - 0.4).abs() < 1e-5, "wheels first");
    assert!((radius(12) - 0.5).abs() < 1e-5, "then meshwheels");
    assert!((radius(20) - 0.3).abs() < 1e-5, "then meshwheels2");
    // Beams: wheels, meshwheels, meshwheels2 (8 per ray each), then the file's.
    assert!(s.beams[..32].iter().all(|b| b.b >= 4 && b.b < 12));
    assert!(s.beams[32..64].iter().all(|b| b.b >= 12 && b.b < 20));
    assert!(s.beams[64..96].iter().all(|b| b.b >= 20 && b.b < 28));
    // Legacy 4 is meshwheels2's first node (file order), real index 20; legacy 13
    // is wheels' second, real 5; legacy 21 is meshwheels' second, real 13.
    let ends: Vec<_> = s.beams[96 + 6..].iter().map(|b| (b.a, b.b)).collect();
    assert_eq!(ends, vec![(3, 20), (3, 5), (3, 13)]);
}

/// Node masses by the spec's rule (§2.5, §5), in f64: every non-virtual beam adds
/// L/2 per non-tyre end; shocks count with their precompressed length.
#[test]
fn masses_count_spokes_on_the_axle_side_and_shocks_precompressed() {
    let src = wheel_truck(&format!(
        "{MESH2}shocks2\n2, 3, 50000, 3000, 1, 0.5, 40000, 2000, 0.5, 2, 0.2, 0.3, 0.9, n\n"
    ));
    let s = spawn(&src);
    let p = |i: u32| s.nodes[i as usize].pos.as_dvec3();
    let mut halves = vec![0.0f64; 4];
    for b in &s.beams {
        if b.ty == ExplicitType::Virtual {
            continue;
        }
        let mut l = (p(b.a) - p(b.b)).length();
        if b.kind == ExplicitKind::Shock2 {
            l *= 0.9;
        }
        for e in [b.a, b.b] {
            if e < 4 {
                halves[e as usize] += 0.5 * l;
            }
        }
    }
    let total: f64 = halves.iter().sum();
    for i in 0..4 {
        let expect = (1000.0 * halves[i] / total).max(50.0);
        let got = s.nodes[i].mass as f64;
        assert!(
            (got - expect).abs() < 1e-3 * expect,
            "node {i}: {got} vs {expect}"
        );
    }
    // Tyre nodes: M / (2n).
    assert!(s.nodes[4..].iter().all(|n| n.mass == 30.0 / 8.0));
}

// --- shocks2 ---

fn table() -> Shock2 {
    Shock2 {
        k_in: 50_000.0,
        c_in: 3_000.0,
        k_out: 40_000.0,
        c_out: 2_000.0,
        pk_in: 1.0,
        pc_in: 0.5,
        pk_out: 0.5,
        pc_out: 2.0,
        k_bump: 9e6,
        c_bump: 12_000.0,
        soft: false,
    }
}

fn close(a: (f32, f32), b: (f32, f32)) -> bool {
    let rel = |x: f32, y: f32| (x - y).abs() <= 1e-5 * y.abs().max(1.0);
    rel(a.0, b.0) && rel(a.1, b.1)
}

#[test]
fn shocks2_rates_match_the_worked_checks() {
    let sh = table();
    let r = |e: f32, u: f32| sh.rates(e, u, 0.5, 0.2, 0.3);
    assert_eq!(r(0.0, 0.0), (50_000.0, 3_000.0));
    assert_eq!(r(-0.05, -1.0), (62_500.0, 3_375.0));
    assert_eq!(r(0.15, 0.5), (60_000.0, 6_000.0));
    assert_eq!(r(-0.11, -1.0), (9e6, 12_000.0));
    let lambda = (-0.05f32 / 0.15) * (-0.05f32 / 0.15);
    assert!(close(
        r(-0.05, 0.2),
        (40_000.0 + 20_000.0 * lambda, 2_000.0 + 4_000.0 * lambda)
    ));
    // Past the long bound the bump rates replace, whatever the velocity.
    assert_eq!(r(0.16, -1.0), (9e6, 12_000.0));
    // Zero bounds: full progression, and any nonzero e is past them.
    assert_eq!(sh.rates(0.0, 0.0, 0.5, 0.0, 0.0), (100_000.0, 4_500.0));
    assert_eq!(sh.rates(1e-4, 0.0, 0.5, 0.0, 0.0), (9e6, 12_000.0));
    // Zero progression factors: the bare rates inside the bounds.
    let flat = Shock2 {
        pk_in: 0.0,
        pc_in: 0.0,
        pk_out: 0.0,
        pc_out: 0.0,
        ..sh
    };
    assert_eq!(flat.rates(-0.09, -1.0, 0.5, 0.2, 0.3), (50_000.0, 3_000.0));
    assert_eq!(flat.rates(0.14, 1.0, 0.5, 0.2, 0.3), (40_000.0, 2_000.0));
}

#[test]
fn shocks2_soft_bump() {
    let sh = Shock2 {
        soft: true,
        ..table()
    };
    let r = |e: f32, u: f32| sh.rates(e, u, 0.5, 0.2, 0.3);
    // Inside both soft bounds (e_l = 0.12, e_s = -0.08): the base rates.
    assert_eq!(r(0.1, 0.5), table().rates(0.1, 0.5, 0.5, 0.2, 0.3));
    // Near the long bound, extending: out rates, then the bump on them.
    let lam = (0.13f32 / 0.15) * (0.13f32 / 0.15);
    let (k0, c0) = (40_000.0 + 20_000.0 * lam, 2_000.0 + 4_000.0 * lam);
    let q = ((0.13f32 - 0.12) * 5.0) / 0.15;
    let mu = q * q;
    assert!(close(
        r(0.13, 0.5),
        (k0 + (k0 + 100.0) * 0.5 * mu, c0 + (c0 + 100.0) * 2.0 * mu)
    ));
    // ... compressing there: the bare in rates.
    assert_eq!(r(0.13, -0.5), (50_000.0, 3_000.0));
    // Near the short bound, compressing: in rates and the out factors (quirk).
    let (k0, c0) = (50_000.0 + 50_000.0 * 0.81f32, 3_000.0 + 1_500.0 * 0.81f32);
    assert!(close(
        r(-0.09, -0.5),
        (
            k0 + (k0 + 100.0) * 0.5 * 0.25,
            c0 + (c0 + 100.0) * 2.0 * 0.25
        )
    ));
    // ... extending there: the bare out rates.
    assert_eq!(r(-0.09, 0.5), (40_000.0, 2_000.0));
    // Past the short bound: at least the bump rates.
    assert!(close(r(-0.11, -1.0), (9e6, 13_700.0)));
}

/// Two 1000 kg nodes 0.5 m apart along z at y = 10, node 1 fixed, and `line` in
/// a `shocks2` section (after `pre`, directives).
fn shock_truck(pre: &str, line: &str) -> String {
    format!(
        "shock test
globals
0, 0, tracks/semi
nodes
0, 0, 10, 0, l, 1000
1, 0, 10, 0.5, l, 1000
fixes
1
{pre}
shocks2
{line}
end
"
    )
}

/// Move node 0 to extension `e` from `rest` along the axis with speed `u`
/// (positive separating) and step once; returns the last beam's force.
fn force_at(s: &mut ExplicitBeams, rest: f32, e: f32, u: f32) -> f32 {
    let b = s.nodes[1].pos;
    let len = rest + e;
    // Pre-step position so that after the node pass (v·h) it is at `len`.
    s.nodes[0].vel = Vec3::new(0.0, 0.0, -u);
    s.nodes[0].force = Vec3::ZERO;
    s.nodes[0].pos = b - Vec3::new(0.0, 0.0, len) - s.nodes[0].vel * STEP;
    s.step();
    s.beams.last().unwrap().force
}

#[test]
fn shocks2_force_at_set_rates() {
    for (opts, pre) in [("n", 1.0f32), ("s", 1.0), ("n", 0.8), ("s", 1.2)] {
        let line =
            format!("0, 1, 50000, 3000, 1, 0.5, 40000, 2000, 0.5, 2, 0.2, 0.3, {pre}, {opts}");
        let mut s = spawn(&shock_truck("", &line));
        let b = s.beams[0];
        assert_eq!(b.kind, ExplicitKind::Shock2);
        assert_eq!(b.ty, ExplicitType::Actuator);
        assert_eq!(b.rest, 0.5 * pre);
        assert_eq!(b.rest0, b.rest);
        let sh = s.shocks2[0];
        assert_eq!(sh.soft, opts == "s");
        for u in [-2.0f32, -0.5, 0.0, 0.5, 2.0] {
            for k in 0..=24 {
                let e = b.rest * (-1.2 * 0.2 + (1.2 * 0.3 + 1.2 * 0.2) * k as f32 / 24.0);
                let f = force_at(&mut s, b.rest, e, u);
                // The solver's own e and u (from the integrated positions).
                let d = s.nodes[0].pos - s.nodes[1].pos;
                let q = d.dot(d);
                // (The exact path's length, `rubble_exact_lengths`, spec §33.)
                let (len, inv) = engine_softbody::explicit_beams::exact_length(q);
                let ee = len - b.rest;
                let uu = (s.nodes[0].vel - s.nodes[1].vel).dot(d) * inv;
                let (kk, cc) = sh.rates(ee, uu, b.rest, 0.2, 0.3);
                assert_eq!(f, -kk * ee - cc * uu, "{opts} pre {pre} u {u} e {e}");
                assert!((ee - e).abs() < 1e-5 && (uu - u).abs() < 1e-4);
                assert_eq!(s.beams[0].rest, b.rest, "never yields");
            }
        }
    }
}

#[test]
fn shocks2_metric_bounds() {
    let bounds = |sb: f32, lb: f32, opts: &str| {
        let line = format!("0, 1, 50000, 3000, 0, 0, 40000, 2000, 0, 0, {sb}, {lb}, 1, {opts}");
        let s = spawn(&shock_truck("", &line));
        (s.beams[0].short_bound, s.beams[0].long_bound)
    };
    let l = 0.5f32;
    assert_eq!(bounds(0.1, 0.2, "n"), (0.1, 0.2));
    assert_eq!(bounds(0.1, 0.2, "m"), (0.1 / l, 0.2 / l));
    assert_eq!(bounds(0.4, 0.6, "M"), ((l - 0.4) / l, (0.6 - l) / l));
    // A long bound shorter than the spawn length clamps to 0; sb above 1 to 1.
    assert_eq!(bounds(-0.5, 0.3, "M"), (1.0, 0.0));
    // `m` then `M` (on the divided values).
    let (sb, lb) = (0.1 / l, 0.7 / l);
    assert_eq!(bounds(0.1, 0.7, "mM"), ((l - sb) / l, (lb - l) / l));
    assert_eq!(bounds(0.1, 0.7, "Mm"), ((l - sb) / l, (lb - l) / l));
}

#[test]
fn shocks2_spawn_parameters_and_break() {
    // Break 1000: S = 4000; G = the deform threshold (400 kN floor without
    // advanced deformation); a detacher group master takes its group along.
    let pre = "set_beam_defaults 600000, 330, 400000, 1000\ndetacher_group 3\nbeams\n0, 1\n";
    let line = "0, 1, 50000, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 1, n";
    let mut s = spawn(&shock_truck(pre, line));
    assert_eq!(s.beams.len(), 2);
    let shock = 1usize;
    let b = s.beams[shock];
    assert_eq!(b.kind, ExplicitKind::Shock2);
    assert_eq!((b.strength, b.gate), (4000.0, 400_000.0));
    assert_eq!(b.group, 3);
    assert_eq!(
        (s.shocks2[0].k_bump, s.shocks2[0].c_bump),
        (600_000.0, 330.0)
    );
    // Pull: the beam breaks on the first step with |F| > max(G, S).
    s.beams[0].broken = true; // the plain beam would take the load; drop it
    s.beams[0].group = 3;
    let mut broke = None;
    for i in 0..2000 {
        let u = 0.5 * i as f32;
        let f = force_at(&mut s, b.rest, 0.0, u);
        if s.beams[shock].broken {
            assert!(f.abs() > 400_000.0, "broke at {f}");
            broke = Some(i);
            break;
        }
        assert!(f.abs() <= 400_000.0);
    }
    assert!(broke.is_some());

    // With a live group member: it goes with the master.
    let mut s = spawn(&shock_truck(pre, line));
    s.beams[0].k = 0.0;
    s.beams[0].c = 0.0;
    for i in 0..2000 {
        force_at(&mut s, b.rest, 0.0, 0.5 * i as f32);
        if s.beams[shock].broken {
            break;
        }
    }
    assert!(s.beams[shock].broken && s.beams[0].broken);
}

#[test]
fn shocks2_validation_keeps_the_line_after_a_discarded_one() {
    let lines = "0, 1, -1, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 1, n\n\
                 0, 1, -2, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 1, n\n\
                 0, 1, -3, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 1, n\n\
                 0, 1, 50000, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 1\n\
                 0, 1, 50000, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3";
    let file = parse_truck(&shock_truck("", lines));
    // First dropped, second kept unchecked, third dropped, fourth kept (no
    // options), fifth too short.
    let springs: Vec<f32> = file.shocks2.iter().map(|s| s.spring_in).collect();
    assert_eq!(springs, vec![-2.0, 50_000.0]);
    // Precompression 0 is accepted.
    let file = parse_truck(&shock_truck(
        "",
        "0, 1, 50000, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 0",
    ));
    assert_eq!(file.shocks2[0].precompression, 0.0);
}

#[test]
fn shocks2_spawn_after_shocks_whatever_the_file_order() {
    let src = "order test
globals
0, 0, tracks/semi
nodes
0, 0, 10, 0, l, 1000
1, 0, 10, 0.5, l, 1000
2, 0.5, 10, 0.5, l, 1000
hydros
0, 2, 0.1
shocks2
0, 1, 50000, 3000, 0, 0, 40000, 2000, 0, 0, 0.2, 0.3, 1
shocks
1, 2, 10000, 500, 0.2, 0.3, 1
beams
0, 1
end
";
    let s = spawn(src);
    let kinds: Vec<_> = s.beams.iter().map(|b| b.kind).collect();
    assert_eq!(
        kinds,
        vec![
            ExplicitKind::Plain,
            ExplicitKind::LimitedSpring,
            ExplicitKind::Shock2,
            ExplicitKind::Plain
        ]
    );
    assert_eq!(s.beams[3].ty, ExplicitType::Actuator);
}
