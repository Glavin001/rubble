//! What the solver's JBeam beam law does with a yield force over travel
//! (`engine/docs/section-matching.md`): one beam and one four-chord cell driven
//! quasi-statically through the solver's own force function (`jbeam_force`, the beam pass's
//! path for a plain JBeam beam), printing force against shortening.
//!
//! The fast tests assert the law's statements on synthetic beams set up exactly as the JBeam
//! spawn sets them (`ExplicitBeams::push_jbeam_beam`); the `#[ignore]`d ones drive the
//! generated sedan's own crash-box and rail chords (run with `--release -- --ignored
//! --nocapture`).

use engine_softbody::explicit_beams::{jbeam_force, Beam, ExplicitBeams, ExplicitKind, ExplicitType, Node, MIN_REST};
use glam::Vec3;

fn node(p: Vec3) -> Node {
    Node { pos: p, vel: Vec3::ZERO, force: Vec3::ZERO, mass: 1.0, friction: 0.0, flags: 0 }
}

/// A NORMAL beam as the JBeam spawn makes it: spring `k`, deform `tc` (both ways), rest
/// `len`, `deformLimit` d (∞: none), `deformLimitExpansion` de, strength `s`, no
/// `deformLimitStress` (hardening uncapped), no damping.
fn jbeam_normal(k: f32, tc: f32, len: f32, d: f32, de: f32, s: f32) -> Beam {
    let mut b = Beam::new(0, 1, ExplicitKind::JbeamPlain, ExplicitType::Deformable, k, 0.0, len, tc, s, 0.0);
    b.gate = tc.min(s);
    b.limit_k = f32::INFINITY;
    b.short_bound = if d.is_finite() { (1.0 - d) * len } else { 0.0 };
    b.long_bound = (1.0 + de) * len;
    b
}

/// One beam from z = 0 to z = L₀ − s, s stepped from 0 to `s_max` in `n` steps and then back
/// until the force is zero: (s, force +compression, rest) at every step.
fn compress(beam: &mut Beam, s_max: f32, n: usize) -> Vec<(f32, f32, f32)> {
    let l0 = beam.rest;
    let mut out = Vec::new();
    for i in 0..=n {
        let s = s_max * i as f32 / n as f32;
        let o = jbeam_force(beam, &node(Vec3::ZERO), &node(Vec3::new(0.0, 0.0, l0 - s)), false);
        out.push((s, o.force, beam.rest));
    }
    // unload
    let mut s = s_max;
    while s > 0.0 {
        s -= s_max / n as f32;
        let o = jbeam_force(beam, &node(Vec3::ZERO), &node(Vec3::new(0.0, 0.0, l0 - s)), false);
        out.push((s, o.force, beam.rest));
        if o.force <= 0.0 {
            break;
        }
    }
    out
}

/// Trapezoid ∫F ds over the loading branch, and the work given back on unloading.
fn works(curve: &[(f32, f32, f32)]) -> (f64, f64) {
    let (mut wl, mut wu) = (0.0f64, 0.0f64);
    for w in curve.windows(2) {
        let ds = f64::from(w[1].0 - w[0].0);
        let f = 0.5 * f64::from(w[0].1 + w[1].1);
        if ds > 0.0 {
            wl += f * ds;
        } else {
            wu -= f * ds;
        }
    }
    (wl, wu)
}

fn print_curve(name: &str, c: &[(f32, f32, f32)], l0: f32, every: usize) {
    println!("{name}: shortening (m), share of L0, force (kN), rest (m)");
    for (i, &(s, f, r)) in c.iter().enumerate() {
        if i % every == 0 || i + 1 == c.len() {
            println!("  {s:.4} {:.3} {:9.2} {r:.4}", s / l0, f / 1e3);
        }
    }
}

/// A 0.5 m beam with deformLimit 0.7: elastic to tc₀, then yield with the threshold
/// hardening as tc₀·L₀/L, locked at 0.3 L₀ (3.33 tc₀), elastic again past it.
#[test]
fn one_beam_yields_hardens_and_locks() {
    let (k, tc0, l0) = (5.0e6f32, 50e3f32, 0.5f32);
    let mut b = jbeam_normal(k, tc0, l0, 0.7, 1.2, 1e9);
    let c = compress(&mut b, 0.40, 40_000);
    print_curve("0.5 m beam, deformLimit 0.7", &c, l0, 2000);
    // elastic until tc₀ at s = tc₀ / k
    let first = c.iter().position(|&(_, _, r)| r < l0).unwrap();
    assert!((c[first].0 - tc0 / k).abs() < 2e-5, "yield at {}", c[first].0);
    // yielding: the force is the hardened threshold tc₀ L₀ / rest
    for &(s, f, r) in &c[first + 10..] {
        if r > 0.151 && s < 0.40 {
            let want = tc0 * l0 / r;
            assert!((f - want).abs() < 0.002 * want, "s {s}: {f} against {want}");
        }
    }
    // locked at the deformLimit: rest 0.15 = 0.3 L₀; force there 3.33 tc₀, then the spring
    let lock = c.iter().position(|&(_, _, r)| (r - 0.15).abs() < 1e-6).expect("locks");
    let (s_lock, f_lock, _) = c[lock];
    println!("locks at s = {s_lock:.4} m ({:.3} L0), force {:.1} kN = {:.2} tc0", s_lock / l0, f_lock / 1e3, f_lock / tc0);
    assert!((f_lock / tc0 - 1.0 / 0.3).abs() < 0.01);
    let at_end = c.iter().filter(|x| x.0 >= 0.3999).map(|x| x.1).next().unwrap();
    assert!((at_end - k * (0.40 - (l0 - 0.15))).abs() < 0.005 * at_end, "elastic past the lock: {at_end}");
    // plastic work: tc₀ L₀ ln(L₀ / 0.3 L₀) = 1.204 tc₀ L₀ (against 0.7 tc₀ L₀ at a constant tc₀)
    let (wl, wu) = works(&c);
    let wp = f64::from(tc0 * l0) * (1.0f64 / 0.3).ln();
    let peak = f64::from(at_end);
    println!("work in {:.0} J, given back {:.0} J (½F²/k at the end {:.0} J), plastic {:.0} J (closed form {:.0})", wl, wu, 0.5 * peak * peak / f64::from(k), wl - wu, wp);
    assert!(((wl - wu) - wp).abs() < 0.01 * wp);
    assert!((wu - 0.5 * peak * peak / f64::from(k)).abs() < 0.02 * wu);
}

/// A beam shorter than MIN_REST / 0.3 = 0.333 m stops yielding at MIN_REST (0.1 m), not at
/// its deformLimit: a 0.18 m crash-box chord can crush 44 % of its length (1.8 tc₀ there).
/// One under 0.1 m never yields in compression at all.
#[test]
fn a_short_beam_locks_at_min_rest() {
    let (k, tc0) = (4.0e6f32, 40e3f32);
    let mut b = jbeam_normal(k, tc0, 0.18, f32::INFINITY, 1.2, 1e9);
    let c = compress(&mut b, 0.12, 24_000);
    print_curve("0.18 m beam, no deformLimit", &c, 0.18, 2000);
    let min_rest = c.iter().map(|x| x.2).fold(f32::MAX, f32::min);
    assert_eq!(min_rest, MIN_REST);
    let lock = c.iter().position(|&(_, _, r)| r == MIN_REST).unwrap();
    println!("locks at s = {:.4} m ({:.3} of 0.18), force {:.2} tc0", c[lock].0, c[lock].0 / 0.18, c[lock].1 / tc0);
    assert!((c[lock].1 / tc0 - 1.8).abs() < 0.01);
    let mut short = jbeam_normal(k, tc0, 0.09, 0.7, 1.2, 1e9);
    let c = compress(&mut short, 0.03, 3000);
    assert!(c.iter().all(|x| x.2 == 0.09), "a 0.09 m beam never yields shorter");
}

/// In tension the threshold hardens as tt₀·L/L₀ and the strength falls by k·(set):
/// a beam pulled steadily breaks when tt = strength − k·set.
#[test]
fn tension_hardens_and_eats_the_strength() {
    let (k, t0, l0, s) = (5.0e6f32, 50e3f32, 0.5f32, 150e3f32);
    let mut b = jbeam_normal(k, t0, l0, 0.7, 1.2, s);
    let mut broke_at = None;
    for i in 0..=20_000 {
        let e = 0.2 * i as f32 / 20_000.0;
        let o = jbeam_force(&mut b, &node(Vec3::ZERO), &node(Vec3::new(0.0, 0.0, l0 + e)), false);
        if o.check_break && broke_at.is_none() {
            broke_at = Some((e, -o.force, b.rest, b.strength));
        }
    }
    let (e, f, r, st) = broke_at.expect("breaks");
    println!("pulled: breaks at extension {e:.4} m, force {:.1} kN, rest {r:.4}, strength left {:.1} kN", f / 1e3, st / 1e3);
    // force = tt₀ r / L₀ and strength = S − k (r − L₀) meet
    let want_r = (s + k * l0) / (t0 / l0 + k);
    assert!((r - want_r).abs() < 2e-4, "{r} against {want_r}");
}

/// Four corner chords and four face diagonals between two end faces held rigid by ties
/// shorter than MIN_REST (the truck rail's lattice: chords 430 kN, 0.266 × 0.09 m apart,
/// diagonals 215 kN, 0.95 m bays), driven by its end face: axially, then in bending.
/// Returns (s or rotation, force or moment) curves and prints them.
fn cell(h: f32, w: f32, len: f32, chord: f32, diag: f32, k: f32) -> (Vec<Vec3>, Vec<Beam>) {
    let mut p = Vec::new();
    for z in [0.0, len] {
        for (x, y) in [(-0.5 * w, -0.5 * h), (0.5 * w, -0.5 * h), (0.5 * w, 0.5 * h), (-0.5 * w, 0.5 * h)] {
            p.push(Vec3::new(x, y, z));
        }
    }
    let mut beams = Vec::new();
    let mut add = |a: u32, b: u32, tc: f32, kk: f32| {
        let l = p[a as usize].distance(p[b as usize]);
        let mut x = jbeam_normal(kk, tc, l, f32::INFINITY, 1.2, 1e12);
        x.a = a;
        x.b = b;
        beams.push(x);
    };
    for i in 0..4u32 {
        add(i, i + 4, chord, k); // chords
        add(i, (i + 1) % 4 + 4, diag, 0.5 * k); // one diagonal a face
        add(i, (i + 1) % 4, 1e12, k); // section ties, both ends
        add(i + 4, (i + 1) % 4 + 4, 1e12, k);
    }
    add(0, 2, 1e12, k);
    add(4, 6, 1e12, k);
    (p, beams)
}

/// Every beam's force on the end face's nodes (4..8): (Σ F_z, Σ y F_z about the centroid).
fn end_load(p: &[Vec3], beams: &mut [Beam]) -> (f32, f32) {
    let (mut fz, mut m) = (0.0, 0.0);
    for b in beams.iter_mut() {
        let (pa, pb) = (p[b.a as usize], p[b.b as usize]);
        let o = jbeam_force(b, &node(pa), &node(pb), false);
        let d = (pa - pb).normalize();
        for (i, sgn) in [(b.a as usize, 1.0), (b.b as usize, -1.0)] {
            if i >= 4 {
                let f = d * o.force * sgn;
                fz += f.z;
                m += p[i].y * f.z;
            }
        }
    }
    // the beams push the end face back: the driver's push and moment
    (fz, m)
}

#[test]
fn a_four_chord_cell_axially_and_in_bending() {
    let (h, w, len, chord, diag, k) = (0.266f32, 0.09f32, 0.95f32, 430e3f32, 215e3f32, 2.0e7f32);
    // axial
    let (p0, mut beams) = cell(h, w, len, chord, diag, k);
    let mut peak: f32 = 0.0;
    println!("four-chord cell, axial: shortening (m), force (kN)");
    for i in 0..=4000 {
        let s = 0.2 * i as f32 / 4000.0;
        let p: Vec<Vec3> = p0.iter().enumerate().map(|(j, x)| if j >= 4 { *x - Vec3::Z * s } else { *x }).collect();
        let (f, _) = end_load(&p, &mut beams);
        peak = peak.max(f);
        if i % 200 == 0 {
            println!("  {s:.3} {:8.1}", f / 1e3);
        }
    }
    // first yield of the four chords: 4 × 430 = 1,720 kN, plus the diagonals' share
    // two diagonals across the w faces, two across the h faces
    let cos_sum = 2.0 * len / (len * len + w * w).sqrt() + 2.0 * len / (len * len + h * h).sqrt();
    println!("chords alone {:.0} kN; the diagonals at their yield add {:.0} kN axially", 4.0 * chord / 1e3, diag * cos_sum / 1e3);
    // bending about x through the centroid
    let (p0, mut beams) = cell(h, w, len, chord, diag, k);
    println!("four-chord cell, bending: end rotation (rad), moment (kN·m)");
    let mut mpk: f32 = 0.0;
    for i in 0..=4000 {
        let phi = 0.2 * i as f32 / 4000.0;
        let (s, c) = phi.sin_cos();
        let p: Vec<Vec3> = p0
            .iter()
            .enumerate()
            .map(|(j, x)| if j >= 4 { Vec3::new(x.x, c * x.y + s * (x.z - len), len - s * x.y + c * (x.z - len)) } else { *x })
            .collect();
        let (_, m) = end_load(&p, &mut beams);
        mpk = mpk.max(m);
        if i % 200 == 0 {
            println!("  {phi:.3} {:8.1}", m / 1e3);
        }
    }
    let chords_only = 2.0 * chord * h;
    println!("peak axial {:.0} kN; peak moment {:.1} kN·m (chords alone 2 F h = {:.1}; the C section's M_p 155)", peak / 1e3, mpk / 1e3, chords_only / 1e3);
    assert!(peak >= 4.0 * chord * 0.999);
    assert!(mpk >= chords_only * 0.999);
}

/// The generated sedan's own crash-box and rail chords, driven alone.
#[test]
#[ignore = "diagnostic: rigs the sedan and drives its front chords one at a time"]
fn the_sedans_front_chords_alone() {
    let t = engine_softbody_authoring::crash_templates::template("sedan").expect("sedan");
    let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(&t.desc, Vec3::ZERO, Vec3::ZERO, t.settings).expect("spawn");
    let names: Vec<&str> = t.desc.nodes.iter().map(|n| n.name.as_str()).collect();
    for (a, b) in [("sh_a1_ul", "sh_a2_ul"), ("sh_a2_dl", "sh_a4_l"), ("sh_a4_l", "sh_a5_l"), ("sh_a1_ul", "sh_a4_l")] {
        let id = |n: &str| names.iter().position(|x| *x == n).unwrap() as u32;
        let j = s.beams.iter().position(|x| (x.a, x.b) == (id(a), id(b)) || (x.a, x.b) == (id(b), id(a))).expect("beam");
        let mut beam = s.beams[j];
        beam.c = 0.0;
        let (l0, k, tc0, floor) = (beam.rest, beam.k, beam.tc, beam.short_bound.max(MIN_REST));
        let c = compress(&mut beam, (l0 - floor) + 0.02, 20_000);
        let (wl, wu) = works(&c);
        let lock = c.iter().position(|x| x.2 <= floor + 1e-6).map(|i| c[i]);
        println!(
            "{a}–{b}: L0 {l0:.3} m, k {:.2e} N/m, tc0 {:.1} kN, yield at {:.1} mm, stops yielding at rest {floor:.3} ({:.0} % of L0 crushed){}; plastic work {:.0} J ({:.2} kJ/m of crush), elastic at tc0 {:.0} J",
            k,
            tc0 / 1e3,
            tc0 / k * 1e3,
            100.0 * (l0 - floor) / l0,
            lock.map_or(String::new(), |(_, f, _)| format!(", force there {:.1} kN = {:.2} tc0", f / 1e3, f / tc0)),
            wl - wu,
            (wl - wu) / f64::from(l0 - floor) / 1e3,
            0.5 * f64::from(tc0) * f64::from(tc0) / f64::from(k),
        );
    }
}
