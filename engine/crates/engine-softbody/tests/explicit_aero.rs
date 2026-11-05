//! Wings, the fuselage drag, airfoil tables and command beams with no input
//! (`engine/docs/explicit-beams-aero-commands-spec.md`), each checked from its
//! definition: table building and lookup bins, one wing panel's forces against an
//! independent evaluation, chains and induced drag, the per-frame break check, the
//! fuselage-drag variants and its one-step lag, command beams' gate and rope rule,
//! self-centering in both key orders, and the spawn order.
//!
//! Tests that need Rigs of Rods' NACA0009 table read it from `RUBBLE_AIRFOIL_DIR`
//! and are skipped when it is not set (Rubble ships no airfoil files).

use engine_softbody::explicit_aero::*;
use engine_softbody::explicit_beams::*;
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
};

/// A synthetic airfoil file: integer degrees outside ±20, 0.1° steps inside, with
/// cl = angle/100, cd = 0.01 + |angle|/1000, cm = −angle/1000.
fn synthetic_afl() -> String {
    let mut s = String::from("A\n700 version\n1234 device\n1 2 3\nalpha cl cd cm:\n");
    let mut row = |a10: i32| {
        let a = a10 as f32 / 10.0;
        let ip = a10 / 10;
        let fp = (a10 % 10).abs();
        let sign = if a10 < 0 && ip == 0 { "-" } else { "" };
        s.push_str(&format!(
            "{sign}{ip}.{fp}  {}  {}  {}\n",
            a / 100.0,
            0.01 + a.abs() / 1000.0,
            -a / 1000.0
        ));
    };
    for d in -180..-20 {
        row(d * 10);
    }
    for a10 in -200..=200 {
        row(a10);
    }
    for d in 21..=180 {
        row(d * 10);
    }
    s
}

fn naca() -> Option<Airfoil> {
    airfoil_from_env("NACA0009.afl").map(|t| Airfoil::parse("NACA0009.afl", &t))
}

#[test]
fn airfoil_rows_land_on_their_bins_and_gaps_fill_linearly() {
    let a = Airfoil::parse("syn", &synthetic_afl());
    assert_eq!(a.cl.len(), AIRFOIL_ENTRIES);
    // −0.5 before the 0.0 row is −0.5° (the fraction negated), index 1795.
    assert_eq!(a.cl[1795], -0.5f32 / 100.0);
    assert_eq!(a.cl[1614], -18.6f32 / 100.0);
    assert_eq!(a.cl[1800], 0.0);
    assert_eq!(a.cl[1805], 0.5f32 / 100.0);
    assert_eq!(a.cl[0], -1.8);
    assert_eq!(a.cl[3600], 1.8);
    // Between −30° and −29° (indices 1500, 1510) the gap is filled linearly in
    // single precision in the specified order.
    let (vj, vi) = (a.cd[1500], a.cd[1510]);
    for k in 0..9 {
        assert_eq!(a.cd[1501 + k], vj + ((k + 1) as f32 * (vi - vj)) / 10.0);
    }
    // Lines after index 3600 are not read; a missing file is all zero.
    let more = synthetic_afl() + "180.5 9 9 9\n";
    assert_eq!(Airfoil::parse("syn", &more), a);
    let z = Airfoil::zero("none");
    assert!(z.cl.iter().chain(&z.cd).chain(&z.cm).all(|&v| v == 0.0));
}

#[test]
fn lookups_wrap_and_take_the_bin_at_or_below() {
    let a = Airfoil::parse("syn", &synthetic_afl());
    let at = |i: usize| (a.cl[i], a.cd[i], a.cm[i]);
    for (angle, index) in [
        (-540.0f32, 0usize),
        (-180.0, 0),
        (-179.95, 0),
        (-20.05, 1599),
        (-0.05, 1799),
        (0.0, 1800),
        (0.04999, 1800),
        (19.95, 1999),
        (179.99, 3599),
        (180.0, 3600),
        (360.1, 1801),
    ] {
        assert_eq!(a.lookup(angle, 0.3, 0.0), at(index), "angle {angle}");
        // With no deflection the chord ratio does not matter.
        assert_eq!(a.lookup(angle, 0.7, 0.0), at(index), "angle {angle}");
    }
    // Deflection ±10° at chord ratio 0.7: lift and moment shifted, drag from the
    // bin 1.15 × 0.3 × d away plus a quadratic term.
    for d in [10.0f32, -10.0] {
        let (cz, cx, cm) = a.lookup(5.0, 0.7, d);
        let sg = d.signum() as f64;
        let root = d.abs().sqrt() as f64;
        let ad = (5.0f64 + 1.15 * (1.0 - 0.7f32 as f64) * d as f64) as f32;
        let id = ((ad + 180.0) * 10.0) as usize;
        assert_eq!(
            cz,
            (a.cl[1850] as f64 - 0.66 * sg * (1.0 - 0.7f32 as f64) * root) as f32
        );
        assert_eq!(
            cm,
            (a.cm[1850] as f64 + 0.20 * sg * (1.0 - 0.7f32 as f64) * root) as f32
        );
        assert_eq!(
            cx,
            (a.cd[id] as f64 + 0.00015 * (1.0 - 0.7f32 as f64) * d as f64 * d as f64) as f32
        );
    }
}

#[test]
fn naca0009_table_when_available() {
    let Some(a) = naca() else {
        eprintln!("RUBBLE_AIRFOIL_DIR not set: skipped");
        return;
    };
    assert!(a.loaded);
    // The α = 0 drag coefficient and identical table ends.
    assert_eq!(a.cd[1800], 0.0052);
    assert_eq!(
        (a.cl[0], a.cd[0], a.cm[0]),
        (a.cl[3600], a.cd[3600], a.cm[3600])
    );
    // A symmetric section: no lift at zero, opposite lift at ±5°.
    assert_eq!(a.cl[1800], 0.0);
    assert_eq!(a.cl[1850], -a.cl[1750]);
}

#[test]
fn densities_at_sea_level_and_aloft() {
    assert_eq!(wing_density(0.0), (101_325.0f64 * 0.0000120896) as f32);
    assert_eq!(fuse_density(0.0), 101_325.0f32 * 0.0000120896f32);
    assert!((wing_density(0.0) - 1.2249787).abs() < 1e-6);
    // Thinner with height (the approximate power is coarse: about 0.64 at 3 km).
    let r = wing_density(3000.0) / wing_density(0.0);
    assert!((0.6..0.8).contains(&r), "{r}");
    assert!(fuse_density(3000.0) < fuse_density(1000.0));
}

/// The angle (degrees) between chord and flow, from f64 trigonometry.
fn reference_angle(c: Vec3, w: Vec3) -> f64 {
    let (c, w) = (c.as_dvec3(), w.as_dvec3());
    (c.dot(w) / (c.length() * w.length()))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

#[test]
fn angle_of_attack_is_the_signed_chord_to_flow_angle() {
    // Chord along +x, span along −z, normal c × b along +y.
    let c = Vec3::X;
    let b = Vec3::NEG_Z;
    let n = normalise(c.cross(b));
    assert_eq!(n, Vec3::Y);
    let mut seen = Vec::new();
    for deg in [-30.0f32, -12.5, -2.5, 0.0, 2.5, 12.5, 30.0, 89.0, 91.0] {
        let r = deg.to_radians();
        let w = Vec3::new(r.cos(), r.sin(), 0.0) * 30.0;
        let a = angle_of_attack(c, n, b, w);
        assert!(
            (a.abs() as f64 - reference_angle(c, w)).abs() < 1e-3,
            "{deg}: {a}"
        );
        seen.push((deg, a));
    }
    // The sign follows the flow's side of the chord.
    for &(d, a) in &seen {
        if let Some(&(_, b)) = seen.iter().find(|x| x.0 == -d && d != 0.0) {
            assert_eq!(a, -b, "{d}");
        }
    }
    // Yaw (flow out of the plane): only the in-plane part counts.
    let w = Vec3::new(10.0f32.to_radians().cos(), 10.0f32.to_radians().sin(), 0.3) * 20.0;
    let in_plane = Vec3::new(w.x, w.y, 0.0);
    assert!((angle_of_attack(c, n, b, w).abs() as f64 - reference_angle(c, in_plane)).abs() < 1e-3);
    // Opposite flow: 180°; no flow: 0.
    assert!((angle_of_attack(c, n, b, -Vec3::X).abs() - 180.0).abs() < 1e-4);
    assert_eq!(angle_of_attack(c, n, b, Vec3::ZERO), 0.0);
}

/// A 1 m × 1 m flat panel (8 distinct slots, 5 cm thick) of unit-mass nodes.
fn panel_positions() -> Vec<Vec3> {
    // FLD, FRD, FLU, FRU, BLD, BRD, BLU, BRU: front at x = 0, back at x = 1, left
    // at z = 0, right at z = −1.
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.05, 0.0),
        Vec3::new(0.0, 0.05, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.05, 0.0),
        Vec3::new(1.0, 0.05, -1.0),
    ]
}

#[test]
fn one_wing_panel_matches_an_independent_evaluation() {
    let table = Airfoil::parse("syn", &synthetic_afl());
    let x = panel_positions();
    for efficacy in [1.0f32, 0.76] {
        let mut w = Wing::new([0, 1, 2, 3, 4, 5, 6, 7], efficacy, 1.0, 'n', 0, &x);
        chain_wings(std::slice::from_mut(&mut w), &x);
        let ind = w.induced.unwrap();
        assert_eq!(ind.side, WingSide::Left);
        assert_eq!((ind.span, ind.area), (1.0, 1.0));
        for speed in [0.0f32, 5.0, 30.0, 80.0] {
            for deg in [-30.0f32, -7.5, 0.0, 2.5, 15.0] {
                for y in [0.0f32, 1000.0, 5000.0] {
                    // Every node moving at −flow.
                    let r = deg.to_radians();
                    let flow = Vec3::new(r.cos(), r.sin(), 0.0) * speed;
                    let f = w.forces(|i| x[i as usize], |_| -flow, y, &table);
                    // Independent: lift ⟂ span and flow, drag along the flow.
                    let rho = wing_density(y) as f64;
                    let wv = flow.as_dvec3();
                    let v = wv.length();
                    let alpha = angle_of_attack(Vec3::X, Vec3::Y, Vec3::NEG_Z, flow);
                    let (cz, cx, cm) = table.lookup(alpha, 1.0, 0.0);
                    let (cz, cx, cm) = (cz as f64, cx as f64, cm as f64);
                    let b = glam::DVec3::NEG_Z;
                    let wf = wv * (cx * 0.5 * rho * v) + b.cross(-wv) * (cz * 0.5 * rho * v);
                    let mu = -cm * 0.5 * rho * v * v;
                    let e = efficacy as f64;
                    let n = glam::DVec3::Y;
                    let front = wf * (e * 0.75 / 4.0) + n * (e * mu);
                    let back = wf * (e * 0.25 / 4.0) - n * (e * mu / 3.0);
                    let induced = wv * (cx * cx * 0.25 * rho * v / 3.14159);
                    let tol = |r: glam::DVec3| 1e-5 * (1.0 + r.length());
                    assert!(
                        (f.front.as_dvec3() - front).length() < tol(front),
                        "front {speed} {deg} {y}"
                    );
                    assert!(
                        (f.back.as_dvec3() - back).length() < tol(back),
                        "back {speed} {deg} {y}"
                    );
                    let (i, side) = f.induced.unwrap();
                    assert_eq!(side, WingSide::Left);
                    assert!((i.as_dvec3() - induced).length() < tol(induced));
                    if speed == 0.0 {
                        assert_eq!(f.front.length(), 0.0);
                    }
                }
            }
        }
    }
}

#[test]
fn wing_forces_reach_the_slots_in_order_and_repeat_on_repeated_slots() {
    // The Viper layout: four distinct nodes, each slot pair repeated.
    let table = Airfoil::parse("syn", &synthetic_afl());
    let pts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
    ];
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1e3;
    for p in pts {
        s.nodes.push(Node {
            pos: p,
            vel: Vec3::new(-30.0, -3.0, 0.0),
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags: NODE_NO_GROUND | NODE_IMMOVABLE,
        });
    }
    let x: Vec<Vec3> = pts.to_vec();
    s.airfoils.push(std::sync::Arc::new(table.clone()));
    let mut w = Wing::new([0, 1, 0, 1, 3, 2, 3, 2], 0.76, 1.0, 'n', 0, &x);
    chain_wings(std::slice::from_mut(&mut w), &x);
    s.wings.push(w.clone());
    s.finish();
    s.step();
    let f = w.forces(
        |i| x[i as usize],
        |_| Vec3::new(-30.0, -3.0, 0.0),
        0.0,
        &table,
    );
    let (i, _) = f.induced.unwrap();
    let g = Vec3::new(0.0, GRAVITY, 0.0);
    // FLD = FLU = node 0, FRD = FRU = node 1; BLD = BLU = node 3 (with the left
    // induced drag twice), BRD = BRU = node 2.
    assert_eq!(s.nodes[0].force, g + f.front + f.front);
    assert_eq!(s.nodes[1].force, g + f.front + f.front);
    assert_eq!(s.nodes[3].force, g + i + i + f.back + f.back);
    assert_eq!(s.nodes[2].force, g + f.back + f.back);
}

#[test]
fn chains_put_induced_drag_on_their_ends() {
    // Three wings chained right to left (each FRD is the previous FLD), then a lone
    // wing.
    let x: Vec<Vec3> = (0..16)
        .map(|i| Vec3::new((i % 2) as f32, 0.0, -((i / 2) as f32)))
        .collect();
    // Wing k spans z from −k−1 (right) to −k... nodes: front at x = 0 even, back odd.
    let wing = |fl: u32, fr: u32| {
        Wing::new(
            [fl, fr, fl, fr, fl + 1, fr + 1, fl + 1, fr + 1],
            1.0,
            1.0,
            'n',
            0,
            &x,
        )
    };
    let mut wings = vec![wing(2, 0), wing(4, 2), wing(6, 4), wing(12, 10)];
    chain_wings(&mut wings, &x);
    let first = wings[0].induced.unwrap();
    assert_eq!(first.side, WingSide::Right);
    assert_eq!(first.span, 3.0);
    assert_eq!(first.area, 3.0);
    assert_eq!(wings[1].induced, None);
    let last = wings[2].induced.unwrap();
    assert_eq!(
        (last.side, last.span, last.area),
        (WingSide::Left, 3.0, 3.0)
    );
    let lone = wings[3].induced.unwrap();
    assert_eq!(
        (lone.side, lone.span, lone.area),
        (WingSide::Left, 1.0, 1.0)
    );
}

#[test]
fn a_wing_breaks_at_the_frame_check_once_its_area_doubles() {
    let x = panel_positions();
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1e3;
    for &p in &x {
        s.nodes.push(Node {
            pos: p,
            vel: Vec3::new(-20.0, 0.0, 0.0),
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags: NODE_NO_GROUND,
        });
    }
    s.airfoils
        .push(std::sync::Arc::new(Airfoil::parse("syn", &synthetic_afl())));
    s.wings
        .push(Wing::new([0, 1, 2, 3, 4, 5, 6, 7], 1.0, 1.0, 'n', 0, &x));
    s.finish();
    // The reference: (2 × the larger doubled half area)².
    assert_eq!(s.wings[0].break_ref, 4.0);
    s.check_wings();
    assert!(!s.wings[0].broken);
    // Stretch the back edge by 1.9×: not broken; by 2.1× (prescribed, mid-frame):
    // not broken until the next frame's check.
    for n in &mut s.nodes[4..] {
        n.pos.x = 1.9;
    }
    s.check_wings();
    assert!(!s.wings[0].broken);
    for n in &mut s.nodes[4..] {
        n.pos.x = 2.1;
    }
    s.step();
    assert!(!s.wings[0].broken, "the check is per frame, not per step");
    s.frame();
    assert!(s.wings[0].broken);
    // Broken for good, and no force: back to the original size, a step adds only
    // gravity.
    for (n, p) in s.nodes.iter_mut().zip(&x) {
        n.pos = *p;
    }
    s.frame();
    assert!(s.wings[0].broken);
    s.step();
    assert!(s
        .nodes
        .iter()
        .all(|n| n.force == Vec3::new(0.0, GRAVITY, 0.0)));
}

const BLOCK: &str = "fuse block
nodes
0, 0, 0, 0
1, 1, 0, 0
2, 0, 1, 0
3, 0, 0, 1
4, 1, 1, 1
beams
0, 1
0, 2
0, 3
1, 4
2, 4
3, 4
1, 2
2, 3
1, 3
0, 4
FUSE
end
";

fn block(fuse: &str, settings: ExplicitBeamsSettings) -> ExplicitBeams {
    let text = BLOCK.replace("FUSE", fuse);
    let mut s = ExplicitBeams::from_truck_with_airfoils(
        &text,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(30.0, 0.0, 0.0),
        settings,
        &|_| None,
    )
    .unwrap();
    s.ground = -1e4;
    s
}

#[test]
fn fusedrag_width_zero_removes_all_air_drag_and_draws_no_noise() {
    let on = ExplicitBeamsSettings::default();
    let mut zero = block("fusedrag\n0, 4, 0.0, NACA0009.afl", on);
    let mut plain = block("", on);
    assert_eq!(zero.fusedrag.as_ref().unwrap().width, 0.0);
    for _ in 0..200 {
        zero.step();
        plain.step();
    }
    // No drag: every node keeps 30 m/s along x (beams are at rest, gravity is
    // along y). The noise sequence never advanced.
    assert!(
        zero.nodes.iter().all(|n| (n.vel.x - 30.0).abs() < 1e-3),
        "{:?}",
        zero.nodes[0].vel
    );
    assert_eq!(zero.noise, 1);
    assert_eq!(zero.fuse_force, Vec3::ZERO);
    // Without `fusedrag` the 0.05 drag slows it.
    assert!(plain.nodes[0].vel.x < 29.99);
    assert_ne!(plain.noise, 1);
}

#[test]
fn fusedrag_force_lags_one_step_and_follows_its_formula() {
    for y in [0.0f32, 500.0, 3000.0] {
        let mut s = block("fusedrag\n0, 4, 1.5", EXACT);
        for n in &mut s.nodes {
            n.pos.y += y;
        }
        let mut twin = block("", EXACT);
        for n in &mut twin.nodes {
            n.pos.y += y;
        }
        assert_eq!(s.fuse_force, Vec3::ZERO);
        s.step();
        twin.step();
        // D from front node 0's velocity after step 1's integration.
        let d1 = s.fuse_force;
        let v = s.nodes[0].vel;
        let rho = fuse_density(s.origin.y + s.nodes[0].pos.y) as f64;
        let k = (1.5f32 * 1.5f32) as f64 * 0.5 * 0.5 * rho * v.length() as f64 / 5.0;
        assert_eq!(d1, -v * (k as f32));
        // Step 2 integrates step 1's restart (D = 0): the same as with no drag at
        // all; step 3 integrates D.
        s.step();
        twin.step();
        for (a, b) in s.nodes.iter().zip(&twin.nodes) {
            assert_eq!(a.vel, b.vel, "D first integrated in step n + 2");
        }
        s.step();
        twin.step();
        assert!(s.nodes[0].vel.x < twin.nodes[0].vel.x - 1e-4);
        // Total force N·D = (W²/4)·ρ·|v|·(−v).
        let total = (s.fuse_force * 5.0).length() as f64;
        let v = s.nodes[0].vel.length() as f64;
        let expect =
            1.5f64.powi(2) / 4.0 * fuse_density(s.origin.y + s.nodes[0].pos.y) as f64 * v * v;
        assert!(
            (total - expect).abs() < 1e-4 * expect,
            "{total} vs {expect}"
        );
    }
}

#[test]
fn fusedrag_autocalc_width_from_file_extents() {
    let s = block("fusedrag\n0, 4, autocalc, 0.5", EXACT);
    // z extent 1 × y extent 1 × 0.5.
    assert_eq!(s.fusedrag.as_ref().unwrap().width, 0.5);
    let s = block("fusedrag\n0, 4, autocalc", EXACT);
    assert_eq!(s.fusedrag.as_ref().unwrap().width, 1.0);
}

const COMMANDS: &str = "commands
globals
0, 0, x
set_beam_defaults 9000000, 12000, 400000, 5000
nodes
0, 0, 0, 0
1, 1, 0, 0
2, 0, 1, 0
3, 0, 0, 1
4, 1, 1, 1
beams
0, 1
hydros
0, 2, 0.1
shocks
0, 3, 10000, 100, 0.1, 0.1, 1.0
commands2
1, 4, 0.5, 0.25, 0.8, 1.5, 3, 4, c
commands
2, 3, 0.2, 0.9, 1.2, 1, 2, r
end
";

#[test]
fn commands_spawn_after_shocks_before_hydros_with_the_deform_gate() {
    let s =
        ExplicitBeams::from_truck_with_airfoils(COMMANDS, Vec3::ZERO, Vec3::ZERO, EXACT, &|_| None)
            .unwrap();
    let ends: Vec<_> = s.beams.iter().map(|b| (b.a, b.b)).collect();
    assert_eq!(ends, vec![(0, 1), (0, 3), (1, 4), (2, 3), (0, 2)]);
    let (cmd, rope, hydro) = (&s.beams[2], &s.beams[3], &s.beams[4]);
    assert_eq!(
        (cmd.kind, cmd.ty),
        (ExplicitKind::Plain, ExplicitType::Actuator)
    );
    assert_eq!(
        (rope.kind, rope.ty),
        (ExplicitKind::Rope, ExplicitType::Actuator)
    );
    // Gate = deform threshold, strength = break threshold: |F| > max(G, S) breaks.
    for b in [cmd, rope, hydro] {
        assert_eq!((b.gate, b.strength), (400_000.0, 5000.0));
        assert_eq!(b.k, 9_000_000.0);
        assert_eq!(b.rest, b.rest0);
    }
    assert_eq!(s.centering.len(), 1);
    assert_eq!(s.centering[0].beam, 2);
    assert_eq!(s.centering[0].centre, 1.15);
}

#[test]
fn a_command_pulled_between_strength_and_gate_does_not_break() {
    let mut s =
        ExplicitBeams::from_truck_with_airfoils(COMMANDS, Vec3::ZERO, Vec3::ZERO, EXACT, &|_| None)
            .unwrap();
    s.ground = -1e3;
    let j = 3; // the rope command 2–3, pulled: a normal spring–damper
    let b = s.beams[j];
    let mut probe = b;
    let na = s.nodes[b.a as usize];
    let mut nb = s.nodes[b.b as usize];
    // 20 kN: above S = 5 kN, below G = 400 kN.
    let dir = (nb.pos - na.pos).normalize();
    nb.pos += dir * (20_000.0 / b.k);
    let out = beam_force(&mut probe, &na, &nb, false);
    assert!(out.force.abs() > 15_000.0 && out.force.abs() < 400_000.0);
    assert!(!out.check_break && !probe.broken);
    // Past the gate it breaks.
    let mut nb2 = s.nodes[b.b as usize];
    nb2.pos += dir * (500_000.0 / b.k);
    let mut probe = b;
    assert!(beam_force(&mut probe, &na, &nb2, false).check_break);
    // Never yields.
    assert_eq!(probe.rest, b.rest);
    let _ = &mut s;
}

#[test]
fn a_rope_command_under_compression_keeps_a_tenth_of_its_damping() {
    let s =
        ExplicitBeams::from_truck_with_airfoils(COMMANDS, Vec3::ZERO, Vec3::ZERO, EXACT, &|_| None)
            .unwrap();
    let mut b = s.beams[3];
    let na = s.nodes[b.a as usize];
    let mut nb = s.nodes[b.b as usize];
    let d = na.pos - nb.pos;
    nb.pos += d * 0.1; // shorter
    nb.vel = d.normalize() * 2.0; // closing at 2 m/s
    let out = beam_force(&mut b, &na, &nb, false);
    let u = (na.vel - nb.vel).dot(na.pos - nb.pos) / (na.pos - nb.pos).length();
    assert!(
        (out.force - (-(0.1 * 12_000.0) * u)).abs() < 1e-2,
        "{} {u}",
        out.force
    );
}

/// Rest length per step of a self-centering beam with limits (0.8, 1.5), rates
/// 0.5 (shorten) and 0.25 (lengthen): it lengthens at 0.25·h a step toward
/// 1.15 × spawn length and stops there.
#[test]
fn self_centering_moves_to_its_centre_and_holds_in_both_key_orders() {
    for (ck, ek) in [(3, 4), (4, 3)] {
        let mut c = CenteringCommand::new(0, 0.5, 0.25, 0.8, 1.5, ck, ek);
        let l0 = 2.0f32;
        let mut l = l0;
        let mut trace = Vec::new();
        for _ in 0..20_000 {
            let before = l;
            l = c.update(l, l0, STEP);
            trace.push(l);
            if l == before {
                break;
            }
            // Each moving step adds rate·h (to rounding).
            assert!(((l - before) - 0.25 * STEP).abs() < 2e-6, "{}", l - before);
        }
        let r = l / l0;
        assert!(
            ((r - 1.15).abs() as f64) < 1e-4 || l == 1.15 * l0,
            "ends at the centre: {r}"
        );
        // ≈ 0.3 m at 0.125 mm a step.
        assert!((2350..2450).contains(&trace.len()), "{} steps", trace.len());
        // Holds.
        for _ in 0..100 {
            assert_eq!(c.update(l, l0, STEP), l);
        }
    }
    // Overshoot snaps to the centre: start just past it, moving the other way.
    let mut c = CenteringCommand::new(0, 0.5, 0.25, 0.8, 1.5, 1, 2);
    c.mode = 1;
    let l = c.update(1.16 * 2.0, 2.0, STEP);
    assert_eq!(l, 1.15f32 * 2.0);
    assert_eq!(c.mode, 0);
    // Symmetric limits: the centre is 1; nothing moves.
    let mut c = CenteringCommand::new(0, 0.5, 0.5, 0.5, 1.5, 1, 2);
    assert_eq!(c.update(2.0, 2.0, STEP), 2.0);
}

#[test]
fn a_self_centering_command_moves_in_the_solver_and_plain_ones_do_not() {
    let mut s =
        ExplicitBeams::from_truck_with_airfoils(COMMANDS, Vec3::ZERO, Vec3::ZERO, EXACT, &|_| None)
            .unwrap();
    s.ground = -1e3;
    let rest0 = s.beams[2].rest;
    for _ in 0..100 {
        s.step();
    }
    assert!((s.beams[2].rest - (rest0 + 100.0 * 0.25 * STEP)).abs() < 1e-5);
    assert_eq!(s.beams[3].rest, s.beams[3].rest0);
}

#[test]
fn parses_wings_fusedrag_and_command_options() {
    let text = "w
nodes
0, 0, 0, 0
1, 0, 0, -1
2, 1, 0, -1
3, 1, 0, 0
beams
0, 1
commands2
0, 2, 0.3, 0.3, 1.0, 1.95, 1, 2, ip Left 0.0, 0.0, smooth smooth 0 0
1, 3, 0.3, 0.3, 1.0, 1.95, 1, 2, rc
1, 3, 0.3, 0.3, 1.0, 1.95, 1, 2, pc
wings
0, 1, 0, 1, 3, 2, 3, 2, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, n, 1.0, 0.0, 0.0, NACA0009.afl 0.76
0, 1, 0, 1, 3, 2, 3, 2, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, S
fusedrag
0, 3, 0.0, NACA0009.afl
end
";
    let f = engine_softbody::truck::parse_truck(text);
    assert_eq!(f.commands.len(), 3);
    assert!(!f.commands[0].centering && !f.commands[0].rope);
    assert!(f.commands[1].centering && f.commands[1].rope);
    assert!(!f.commands[2].centering, "p before c wins");
    assert_eq!(f.wings.len(), 1, "a wing with no airfoil spawns nothing");
    assert_eq!(f.wings[0].nodes, [0, 1, 0, 1, 3, 2, 3, 2]);
    assert_eq!(f.wings[0].efficacy, 0.76);
    assert_eq!(f.wings[0].airfoil, "NACA0009.afl");
    let fd = f.fusedrag.unwrap();
    assert_eq!((fd.front, fd.rear, fd.width), (0, 3, 0.0));
}
