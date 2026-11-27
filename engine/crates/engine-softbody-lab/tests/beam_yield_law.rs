//! Rubble's crush law and crush cells (`engine/docs/beam-yield-law.md`,
//! `engine_softbody::explicit_crush`), CPU reference: one beam and one cell driven through the
//! solver's own beam pass (positions prescribed, `step_rest`) and dynamically (`step`), against the
//! closed forms; and what a cell of independent chords does under the same off-centre load.
//!
//! The fast tests assert the law's statements (beam-yield-law.md §3, §5). The `#[ignore]`d ones
//! print telemetry and write the curves to `out/beam-law/*.csv` for `tools/beam-law/plot.py`
//! (`--release -- --ignored --nocapture`).

use engine_softbody::crash_reference::metrics::beams::{CellPlastic, Plastic};
use engine_softbody::desc::{BeamKind, JbeamBeam, SoftBeam, SoftNode, SoftVehicleDesc};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, MIN_REST};
use engine_softbody::explicit_crush::{assert_cpu_only, cell_id, CrushLaw, Curve, MIN_REST_FLOOR};
use engine_vehiclegen::crush_law::member_crush;
use engine_vehiclegen::structure::{Grade, Member, Section};
use glam::Vec3;
use std::io::Write;

/// A NORMAL beam a–b as the JBeam loader makes it: spring `k`, damping `c`, `beamDeform` `peak`,
/// `deformLimit` `dl`, optional crush law.
fn beam(a: u32, b: u32, k: f32, c: f32, peak: f32, dl: f32, law: Option<CrushLaw>) -> SoftBeam {
    SoftBeam {
        a,
        b,
        kind: BeamKind::Normal,
        spring: k,
        damp: c,
        deform: peak,
        strength: f32::INFINITY,
        jbeam: Some(JbeamBeam { deform: peak, deform_limit: dl, crush: law, ..Default::default() }),
        ..Default::default()
    }
}

fn law(plateau: f32, drop: f32, min_rest: f32, tension: f32) -> CrushLaw {
    CrushLaw { plateau, drop, min_rest, tension, cell: 0, moment: [0.0; 2] }
}

fn spawn(nodes: &[(Vec3, f32)], beams: Vec<SoftBeam>) -> ExplicitBeams {
    let desc = SoftVehicleDesc {
        name: "beam law".into(),
        nodes: nodes.iter().enumerate().map(|(i, &(p, m))| SoftNode { name: format!("n{i}"), position: p.to_array(), mass: m, collision: false, ..Default::default() }).collect(),
        beams,
        ..Default::default()
    };
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Raycast, Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, settings).expect("spawn").0
}

/// The nodes at `pos` (at rest) and one beam pass; returns each beam's force (+ compression).
fn pass(s: &mut ExplicitBeams, pos: &[Vec3]) -> Vec<f32> {
    for (n, p) in s.nodes.iter_mut().zip(pos) {
        n.pos = *p;
        n.vel = Vec3::ZERO;
        n.force = Vec3::ZERO;
    }
    s.step_rest();
    s.beams.iter().map(|b| b.force).collect()
}

/// One beam along z from the origin, shortened by each of `path` in turn (m; negative stretches):
/// (shortening, force, rest) per pass, and the ledger's plastic work at the end.
fn drive(s: &mut ExplicitBeams, path: &[f32]) -> (Vec<(f32, f32, f32)>, f64) {
    let l0 = s.nodes[1].pos.z;
    let mut p = Plastic::of(s, 0);
    let out = path
        .iter()
        .map(|&d| {
            let f = pass(s, &[Vec3::ZERO, Vec3::new(0.0, 0.0, l0 - d)])[0];
            p.advance(s.beams[0].rest);
            (d, f, s.beams[0].rest)
        })
        .collect();
    (out, p.work)
}

/// Linear steps from a to b (b included).
fn ramp(a: f32, b: f32, n: usize) -> Vec<f32> {
    (1..=n).map(|i| a + (b - a) * i as f32 / n as f32).collect()
}

/// External work ∫F d(shortening) along a driven curve (trapezoid), starting from rest.
fn work_in(c: &[(f32, f32, f32)]) -> f64 {
    let mut w = 0.0f64;
    let (mut d0, mut f0) = (0.0f32, 0.0f32);
    for &(d, f, _) in c {
        w += 0.5 * f64::from(f0 + f) * f64::from(d - d0);
        (d0, f0) = (d, f);
    }
    w
}

#[test]
fn the_curve_maps_back_exactly_and_integrates_in_closed_form() {
    let c = Curve { peak: 50e3, plateau: 20e3, drop: 0.02 };
    let k = 5e6;
    assert!(c.stable(k) && !Curve { drop: 0.01, ..c }.stable(k));
    // on the drop: g − kΔ = C(s + Δ)
    let g = 52e3;
    let d = c.flow(g, k, 0.005);
    assert!((g - k * d - c.at(0.005 + d)).abs() < 1.0, "{d}");
    // past the drop: onto the plateau
    let d = c.flow(60e3, k, 0.019);
    assert!((60e3 - k * d - 20e3).abs() < 1e-2 && 0.019 + d > c.drop);
    assert_eq!(c.flow(19e3, k, 0.03), 0.0);
    // ∫C ds: the drop's trapezoid, then the plateau
    let w = c.work(0.0, 0.1);
    let d = f64::from(c.drop);
    assert!((w - (0.5 * (50e3 + 20e3) * d + 20e3 * (0.1 - d))).abs() < 1e-9 * w);
    assert!((c.work(0.0, 0.01) + c.work(0.01, 0.1) - w).abs() < 1e-9 * w);
}

/// The statements of beam-yield-law.md §3.3 on one beam through the solver: elastic to the peak,
/// down the drop to the plateau with the force continuous, densified at deformLimit (here under the
/// old 0.1 m: the least rest is the law's), a spring past it; the ledger's ∫C ds is the work taken.
#[test]
fn one_beam_peaks_drops_to_its_plateau_and_densifies() {
    let (k, p, q, d, l0) = (5.0e6f32, 50e3f32, 20e3f32, 0.02f32, 0.18f32);
    let mut s = spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, l0), 1.0)], vec![beam(0, 1, k, 0.0, p, 0.73, Some(law(q, d, MIN_REST_FLOOR, 0.0)))]);
    let (c, plastic) = drive(&mut s, &ramp(0.0, 0.15, 30_000));
    let floor = (1.0 - 0.73) * l0;
    let first = c.iter().position(|x| x.2 < l0).unwrap();
    assert!((c[first].0 - p / k).abs() < 2e-5, "first yield at {}", c[first].0);
    let curve = Curve { peak: p, plateau: q, drop: d };
    let mut last = 0.0f32;
    for &(sh, f, r) in &c[first..] {
        if r > floor + 1e-6 {
            assert!((f - curve.at(l0 - r)).abs() < 0.002 * p, "s {sh}: {f} against C {}", curve.at(l0 - r));
        }
        // continuous: no step larger than the elastic increment of one pass
        assert!((f - last).abs() <= k * 0.15 / 30_000.0 * 1.01 + 1.0 || last == 0.0, "jump at {sh}: {last} → {f}");
        last = f;
    }
    let lock = c.iter().position(|x| (x.2 - floor).abs() < 1e-6).expect("densifies at deformLimit");
    assert!(floor < MIN_REST, "the old least rest would have stopped it at 0.1 m");
    let end = c.last().unwrap();
    assert!((end.1 - k * (end.0 - (l0 - floor))).abs() < 0.005 * end.1, "a spring past the floor");
    let want = curve.work(0.0, f64::from(l0 - floor));
    let stored = 0.5 * f64::from(end.1).powi(2) / f64::from(k);
    println!("lock at s = {:.4} m; plastic work {plastic:.0} J (closed form {want:.0}); work in {:.0} J = plastic + stored {:.0} J", c[lock].0, work_in(&c), stored);
    assert!((plastic - want).abs() < 1e-3 * want);
    assert!((work_in(&c) - plastic - stored).abs() < 2e-3 * want);
}

/// Closed load cycles (load, unload, reload, reverse into tension, compress again, back to the
/// start length): the work put in is never less than what the beam stores at the end (nothing is
/// created), and equals the ledger's plastic work plus the stored energy.
#[test]
fn closed_cycles_create_no_energy() {
    let (k, p, q, d, l0) = (5.0e6f32, 50e3f32, 20e3f32, 0.02f32, 0.5f32);
    let mut s = spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, l0), 1.0)], vec![beam(0, 1, k, 0.0, p, 0.73, Some(law(q, d, MIN_REST_FLOOR, 60e3)))]);
    let mut path = ramp(0.0, 0.015, 3000); // onto the drop
    path.extend(ramp(0.015, 0.0, 3000)); // unload, into tension
    path.extend(ramp(0.0, 0.08, 16000)); // reload: no second peak
    path.extend(ramp(0.08, 0.0, 16000));
    path.extend(ramp(0.0, -0.06, 12000)); // stretch past the tensile yield
    path.extend(ramp(-0.06, 0.1, 32000)); // crush again
    path.extend(ramp(0.1, 0.0, 20000)); // back to the start length
    let (c, plastic) = drive(&mut s, &path);
    let stored = 0.5 * f64::from(k) * f64::from(l0 - s.beams[0].rest).powi(2);
    let w = work_in(&c);
    println!("work in {w:.1} J, stored {stored:.1} J, plastic (ledger) {plastic:.1} J, residual {:.2} J", w - plastic - stored);
    assert!(w >= stored - 1e-6, "energy created: {w} put in, {stored} stored");
    assert!((w - plastic - stored).abs() < 2e-3 * plastic, "ledger {plastic} against {}", w - stored);
    // the reload stays on the plateau (the set only grows): no force above the plateau after the drop
    let after: f32 = c[6000..22000].iter().map(|x| x.1).fold(0.0, f32::max);
    assert!(after < q * 1.0 + 0.002 * p + 1.0 || after <= c[..3000].iter().map(|x| x.1).fold(0.0, f32::max), "reload peaked at {after}");
}

#[test]
#[should_panic(expected = "steeper than k/2")]
fn a_drop_steeper_than_half_k_is_refused() {
    spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, 0.3), 1.0)], vec![beam(0, 1, 5e6, 0.0, 50e3, 0.73, Some(law(20e3, 0.01, MIN_REST_FLOOR, 0.0)))]);
}

#[test]
#[should_panic(expected = "under the least the 0.5 ms step allows")]
fn a_least_rest_under_the_floor_is_refused() {
    spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, 0.3), 1.0)], vec![beam(0, 1, 5e6, 0.0, 50e3, 0.73, Some(law(50e3, 0.0, 0.02, 0.0)))]);
}

#[test]
#[should_panic(expected = "runs on the CPU reference only")]
fn a_gpu_upload_of_the_crush_law_is_refused() {
    let s = spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, 0.3), 1.0)], vec![beam(0, 1, 5e6, 0.0, 50e3, 0.73, Some(law(50e3, 0.0, MIN_REST_FLOOR, 0.0)))]);
    assert_cpu_only(&[s]);
}

#[test]
fn a_record_without_the_law_prints_as_derive_printed_it() {
    let x = engine_softbody::explicit_jbeam_ext::BeamExt { beam: 3, mask: 2, deform: [1, 0], trigger: f32::INFINITY, ..Default::default() };
    assert_eq!(
        format!("{x:?}"),
        "BeamExt { beam: 3, mask: 2, deform: [1, 0], trigger: inf, e1: [0.0, 0.0, 0.0, 0.0], e2: [0.0, 0.0, 0.0, 0.0], ramp_delta: 0.0, ramp_steps: 0, extra_groups: [0, 0], ramp_left: 0, pending: false, pressure: [0.0, 0.0] }"
    );
    // and a fleet without the keys uploads as before
    let s = spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, 0.3), 1.0)], vec![beam(0, 1, 5e6, 0.0, 50e3, 0.73, None)]);
    assert!(s.beam_ext.is_empty());
    assert_cpu_only(&[s]);
}

/// The drop at crash rate (beam-yield-law.md §3.3): a 0.18 m beam anchored at one end, a 20 kg
/// mass at 15.6 m/s on the other, 0.5 ms steps (at k 5 MN/m one step closes 7.8 mm: 39 kN). The
/// softening at H = k/2 and k/8 neither chatters nor creates energy: the force never passes the
/// peak and stays compressive while the mass closes, and the step's own mechanical energy never
/// rises (each yield only takes from it). (The step's energy: ½m·v² + ½k·e·e₋, e₋ the extension
/// one step before about today's rest: what the solver's step, symplectic Euler, keeps for a
/// spring; the elastic control run shows it constant where ½m·v² + ½k·e² swings by ωh.) The
/// ledger's ∫C ds, sampled each step, counts a little more than the step takes (printed).
#[test]
fn the_drop_is_stable_at_crash_rate() {
    let (k, p, q, l0, m, v) = (5.0e6f32, 50e3f32, 20e3f32, 0.18f32, 20.0f32, 15.6f32);
    let e0 = 0.5 * f64::from(m * v * v);
    let run = |b: SoftBeam| {
        let mut s = spawn(&[(Vec3::ZERO, 1e9), (Vec3::new(0.0, 0.0, l0), m)], vec![b]);
        s.nodes[1].vel = Vec3::new(0.0, 0.0, -v);
        let mut plastic = Plastic::of(&s, 0);
        let (mut fmax, mut emax, mut emin, mut flips, mut last) = (0.0f32, 0.0f64, f64::MAX, 0, 0.0f32);
        let (mut len_prev, mut mech_prev, mut rise) = (l0, e0, 0.0f64);
        for _ in 0..400 {
            s.step();
            let b = s.beams[0];
            plastic.advance(b.rest);
            let rel = s.nodes[1].vel - s.nodes[0].vel;
            let len = (s.nodes[1].pos - s.nodes[0].pos).length();
            let mech = 0.5 * f64::from(m) * f64::from(rel.length_squared()) + 0.5 * f64::from(k) * f64::from(len - b.rest) * f64::from(len_prev - b.rest);
            rise = rise.max(mech - mech_prev);
            mech_prev = mech;
            let en = mech + plastic.work;
            len_prev = len;
            (emax, emin) = (emax.max(en), emin.min(en));
            fmax = fmax.max(b.force);
            if rel.z < 0.0 && b.force < 0.0 && last >= 0.0 {
                flips += 1;
            }
            last = b.force;
        }
        (fmax, emax, emin, flips, plastic.work, rise)
    };
    let el = run(beam(0, 1, k, 0.0, 1e9, 0.73, None));
    println!("elastic control: step energy {:.2} … {:.2} J of {e0:.2} J", el.2, el.1);
    assert!(el.1 <= e0 * 1.0001 && el.2 >= e0 * 0.9999, "the step energy is the integrator's invariant");
    let flat = run(beam(0, 1, k, 0.0, q, 0.73, None));
    println!("JBeam rule at the plateau: peak force {:.1} kN, step energy + ledger max {:.1} J of {e0:.1} J, plastic {:.1} J, largest one-step rise {:.3} J", flat.0 / 1e3, flat.1, flat.4, flat.5);
    for (name, d) in [("H = k/2", 2.0 * 30e3 / 5e6), ("H = k/8", 8.0 * 30e3 / 5e6)] {
        let (fmax, emax, _, flips, w, rise) = run(beam(0, 1, k, 0.0, p, 0.73, Some(law(q, d, MIN_REST_FLOOR, 0.0))));
        println!("{name}: peak force {:.1} kN (P {:.1}), step energy + ledger max {:.1} J of {e0:.1} J, plastic {w:.1} J, largest one-step rise {rise:.3} J, sign flips {flips}", fmax / 1e3, p / 1e3, emax);
        assert!(fmax <= p * 1.01, "{name}: force {fmax} past the peak");
        assert!(rise <= 1e-6 * e0, "{name}: the step energy rose {rise} J in one step");
        assert_eq!(flips, 0, "{name}: chatter");
    }
}

/// What JBeam's own parameters already do (beam-yield-law.md §2.1): a peak then a plateau from two
/// NORMAL beams in parallel, a plateau beam flat at Q (`deformLimitStress` = `beamDeform`) and a fuse
/// that breaks at S = P − Q (`beamStrength`, `beamDeform` FLT_MAX). The force peaks at P and drops to
/// Q in one step when the fuse breaks (its stored ½S²/k is gone with it: no energy created), then
/// stays at Q. What it cannot give: the drop is a jump (no softening length), the fuse breaks in
/// tension too, and neither beam knows the other's state (a cell of such pairs still turns, §3.5).
#[test]
fn existing_parameters_peak_then_plateau_by_a_fuse() {
    let (k, p, q, l0) = (5.0e6f32, 50e3f32, 20e3f32, 0.5f32);
    let mut plateau = beam(0, 1, k, 0.0, q, 0.73, None);
    plateau.deform_limit_stress = q;
    let mut fuse = beam(0, 1, k, 0.0, f32::INFINITY, f32::INFINITY, None);
    fuse.strength = p - q;
    let mut s = spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, l0), 1.0)], vec![plateau, fuse]);
    let mut out = Vec::new();
    for d in ramp(0.0, 0.1, 10_000) {
        let f = pass(&mut s, &[Vec3::ZERO, Vec3::new(0.0, 0.0, l0 - d)]);
        out.push((d, s.nodes[0].force.z.abs(), s.beams[1].broken, f[0]));
    }
    let peak = out.iter().map(|x| x.1).fold(0.0, f32::max);
    let at = out.iter().position(|x| x.2).expect("the fuse breaks");
    println!("fuse pair: peak {:.1} kN at {:.4} m, then {:.1} kN; the fuse broke at {:.4} m", peak / 1e3, out[at - 1].0, out[at + 10].1 / 1e3, out[at].0);
    assert!((peak - p).abs() < 0.01 * p, "peak {peak}");
    assert!(out[at + 10..].iter().all(|x| (x.1 - q).abs() < 0.01 * q), "plateau after the break");
}

// --- crush cells -------------------------------------------------------------------------------

/// The sedan's crash box (70 × 110 × 1.5 mm HSLA, 0.18 m) or a rail bay (70 × 110 × 1.8, 0.38 m).
fn member(kind: &'static str, t: f64, len: f64) -> Member {
    Member { name: kind.into(), kind, part: "body_shell", section: Section::Box { w: 0.07, h: 0.11 }, gauge: t, grade: Grade::Hsla, length: len, sheet_area: len * 2.0 * (0.07 + 0.11 - 2.0 * t), mesh: Default::default(), on_shell: true }
}

/// A cell's chord values from its section (beam-yield-law.md §5.1; quasi-static, as the calculator
/// gives them): (peak, plateau, drop, tension, moments) per chord (forces a quarter of the section's).
struct Cut {
    peak: f32,
    plateau: f32,
    drop: f32,
    tension: f32,
    moment: [f32; 2],
    /// With X diagonals: the side and top walls' plastic shear (N), held by the diagonals in tension.
    shear: Option<[f32; 2]>,
}

fn cut(m: &Member, k: f32) -> Cut {
    let c = member_crush(m).unwrap();
    let (peak, plateau) = ((c.peak / 4.0) as f32, (c.plateau / 4.0) as f32);
    Cut { peak, plateau, drop: (c.drop as f32).max(2.05 * (peak - plateau) / k), tension: (c.tension / 4.0) as f32, moment: c.moment.map(|x| x as f32), shear: None }
}

/// [`cut`] with the walls' shear held by X diagonals (beam-yield-law.md §5.3).
fn cut_x(m: &Member, k: f32) -> Cut {
    Cut { shear: Some(member_crush(m).unwrap().shear.map(|v| v as f32)), ..cut(m, k) }
}

const CK: f32 = 4.0e6;
const B: f32 = 0.07;
const H: f32 = 0.11;

/// Corner `i` (0 top-out, 1 bottom-out, 2 top-in, 3 bottom-in) of a section at z.
fn corner(i: usize, z: f32) -> Vec3 {
    let x = if i < 2 { 0.5 * B } else { -0.5 * B };
    let y = if i % 2 == 0 { 0.5 * H } else { -0.5 * H };
    Vec3::new(x, y, z)
}

/// The cell's beams: four chords (nodes 0–3 rear at z 0, 4–7 front at z L), as a crush cell when
/// `cell`, else independent crush-law chords (or the JBeam rule when `law` is false).
fn chords(cut: &Cut, cell: bool, crush: bool) -> Vec<SoftBeam> {
    (0..4)
        .map(|i| {
            let mut l = law(cut.plateau, cut.drop, MIN_REST_FLOOR, cut.tension);
            if cell {
                l.cell = cell_id("test cell");
                l.moment = cut.moment;
            }
            let mut b = beam(i as u32, i as u32 + 4, CK, 0.0, if crush { cut.peak } else { cut.plateau }, 0.73, crush.then_some(l));
            if !crush {
                b.deform_limit_stress = cut.plateau;
            }
            b
        })
        .collect()
}

#[test]
fn a_cell_folds_at_its_plateau_and_keeps_its_moment_while_folding() {
    let cb = member("crash_box", 0.0015, 0.18);
    let ct = cut(&cb, CK);
    let l = 0.18f32;
    let nodes: Vec<(Vec3, f32)> = (0..8).map(|i| (corner(i % 4, if i < 4 { 0.0 } else { l }), 1.0)).collect();
    let at = |dz: f32, th: f32| -> Vec<Vec3> { (0..8).map(|i| if i < 4 { corner(i, 0.0) } else { let c = corner(i - 4, l - dz); c + Vec3::Z * (th * c.y) }).collect() };
    let mut s = spawn(&nodes, chords(&ct, true, true));
    let mut cp = CellPlastic::of(&s, 0).expect("a cell");
    // uniform crush past the drop: N = 4 × the chord's curve
    let curve = Curve { peak: 4.0 * ct.peak, plateau: 4.0 * ct.plateau, drop: ct.drop };
    let mut wl = 0.0f64;
    let mut last = (0.0f32, 0.0f32);
    for dz in ramp(0.0, 0.06, 6000) {
        let f = pass(&mut s, &at(dz, 0.0));
        let n: f32 = f.iter().sum();
        wl += 0.5 * f64::from(n + last.1) * f64::from(dz - last.0);
        last = (dz, n);
        cp.advance(s.beams.iter().map(|b| b.rest).collect::<Vec<_>>().try_into().unwrap(), [false; 4]);
        let set = 0.18 - s.beams[0].rest;
        if set > 1e-4 {
            assert!((n - curve.at(set)).abs() < 0.003 * curve.peak, "dz {dz}: N {n} against {}", curve.at(set));
        }
    }
    let n_pl = last.1;
    // then rotate the front face about the strong axis at that crush: the moment rises to the
    // strong collapse moment and holds, the axial force stays at the plateau
    let mut m_hold = 0.0f32;
    for th in ramp(0.0, 0.15, 3000) {
        let f = pass(&mut s, &at(0.06, th));
        let m: f32 = (0..4).map(|i| -f[i] * corner(i, 0.0).y).sum();
        let n: f32 = f.iter().sum();
        assert!((n - n_pl).abs() < 0.003 * n_pl, "θ {th}: N {n} left the plateau {n_pl}");
        m_hold = m;
        cp.advance(s.beams.iter().map(|b| b.rest).collect::<Vec<_>>().try_into().unwrap(), [false; 4]);
    }
    println!("cell: plateau {:.1} kN, moment held {:.2} kN·m (collapse moment {:.2}); ledger {:.0} J", n_pl / 1e3, m_hold.abs() / 1e3, ct.moment[0] / 1e3, cp.work);
    assert!((m_hold.abs() - ct.moment[0]).abs() < 0.01 * ct.moment[0]);
    // the same rotation on independent chords crushed as far: the moment is the chords' yields' couple
    let mut s2 = spawn(&nodes, chords(&ct, false, true));
    for dz in ramp(0.0, 0.06, 6000) {
        pass(&mut s2, &at(dz, 0.0));
    }
    let f = (0..3000).map(|i| pass(&mut s2, &at(0.06, 0.15 * (i + 1) as f32 / 3000.0))).last().unwrap();
    let m2: f32 = (0..4).map(|i| -f[i] * corner(i, 0.0).y).sum();
    println!("independent chords: moment {:.2} kN·m at θ 0.15 (their couple at fixed crush: H(Q + Tt))", m2.abs() / 1e3);
    let _ = wl;
}

/// The off-centre load that turns today's cell (beam-yield-law.md §5.2): a 1000 kg ram carrying 0.6 of the cell's plateau work over 73 % of its length
/// pushes the cell's front plate through a pin at height `e` above its centre (one beam to a load
/// node tied to the four front corners); rear face anchored, front corners 2 kg. Returns (top pair's
/// shortening, bottom pair's, the plate's pitch (rad), ram force mean (N), stroke (m)).
fn off_centre(cell: bool, crush: bool, e: f32, len: f32, cut: &Cut, out: Option<&str>) -> (f32, f32, f32, f32, f32) {
    let mut nodes: Vec<(Vec3, f32)> = (0..8).map(|i| (corner(i % 4, if i < 4 { 0.0 } else { len }), if i < 4 { 1e9 } else { 2.0 })).collect();
    nodes.push((Vec3::new(0.0, e, len + 0.02), 2.0)); // 8: the load node
    nodes.push((Vec3::new(0.0, e, len + 0.12), 1000.0)); // 9: the ram
    let mut beams = chords(cut, cell, crush);
    // the long faces' diagonals, as the load path builds them: together a quarter of the chords'
    // plateau along the member, flat (front-load-path.md §3.1 rule 3); with `shear` (`cut.shear`
    // set) both diagonals of each face, each holding its wall's plastic shear in tension
    let mut faces = vec![(0u32, 1u32), (2, 3), (0, 2), (1, 3)];
    if cut.shear.is_some() {
        faces.extend([(1u32, 0u32), (3, 2), (2, 0), (3, 1)]);
    }
    let cos = |a: u32, b: u32| len / (corner(a as usize, len) - corner(b as usize, 0.0)).length();
    let share = 0.25 * 4.0 * cut.plateau / faces.iter().map(|&(a, b)| cos(a, b)).sum::<f32>();
    for &(a, b) in &faces {
        // tension: the wall's plastic shear V over the diagonal's sine (V·d/w)
        let tt = cut.shear.map_or(0.0, |[vh, vb]| {
            let (v, w) = if (a % 2) != (b % 2) { (vh, H) } else { (vb, B) };
            v / (w / (corner(a as usize, len) - corner(b as usize, 0.0)).length())
        });
        let mut d = beam(4 + a, b, CK, 0.0, share, 0.73, crush.then(|| law(share, 0.0, MIN_REST_FLOOR, tt)));
        d.deform_limit_stress = share;
        beams.push(d);
    }
    let tie = |a: u32, b: u32| SoftBeam { a, b, kind: BeamKind::Normal, spring: 5e6, damp: 200.0, deform: f32::INFINITY, strength: f32::INFINITY, jbeam: Some(JbeamBeam { deform: f32::INFINITY, ..Default::default() }), ..Default::default() };
    for (a, b) in [(4, 5), (6, 7), (4, 6), (5, 7), (4, 7), (5, 6), (0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)] {
        beams.push(tie(a, b));
    }
    for c in 4..8 {
        beams.push(tie(c, 8));
    }
    beams.push(tie(8, 9));
    let mut s = spawn(&nodes, beams);
    let v = (2.0 * 0.6 * 4.0 * cut.plateau * 0.73 * len / 1000.0).sqrt();
    s.nodes[9].vel = Vec3::new(0.0, 0.0, -v);
    let mut csv = out.map(|p| std::fs::File::create(p).unwrap());
    if let Some(f) = csv.as_mut() {
        writeln!(f, "t,stroke,ram_force,top,bottom,pitch").unwrap();
    }
    let z0 = s.nodes[9].pos.z;
    let (mut fsum, mut n) = (0.0f64, 0);
    let mut state = (0.0, 0.0, 0.0, 0.0);
    for i in 0..400 {
        s.step();
        let p = |k: usize| s.nodes[k].pos;
        let top = len - 0.5 * ((p(4) - p(0)).length() + (p(6) - p(2)).length());
        let bot = len - 0.5 * ((p(5) - p(1)).length() + (p(7) - p(3)).length());
        let pitch = ((p(4).z + p(6).z) - (p(5).z + p(7).z)).atan2(2.0 * H);
        let fr = s.beams.last().unwrap().force;
        fsum += f64::from(fr);
        n += 1;
        let stroke = z0 - s.nodes[9].pos.z;
        if let Some(f) = csv.as_mut() {
            writeln!(f, "{:.4},{stroke:.5},{fr:.1},{top:.5},{bot:.5},{pitch:.5}", i as f32 * 5e-4).unwrap();
        }
        state = (top, bot, pitch, stroke);
    }
    (state.0, state.1, state.2, (fsum / f64::from(n)) as f32, state.3)
}

/// With the load inside the cell's reserve (e = 0.3 M_c / N), the crush cell with its walls' shear
/// (X diagonals in tension) folds along its length (both pairs shorten alike, the plate barely
/// turns); the same chords independent turn (the near pair crushes, the far one does not).
#[test]
fn an_off_centre_load_folds_a_cell_and_turns_independent_chords() {
    let cb = member("crash_box", 0.0015, 0.18);
    let ct = cut_x(&cb, CK);
    let e = 0.3 * ct.moment[0] / (4.0 * ct.plateau);
    let (t1, b1, p1, _, _) = off_centre(true, true, e, 0.18, &ct, None);
    let (t2, b2, p2, _, _) = off_centre(false, true, e, 0.18, &ct, None);
    println!("e {:.1} mm: cell top {:.1} / bottom {:.1} mm, pitch {:.3}; independent top {:.1} / bottom {:.1} mm, pitch {:.3}", e * 1e3, t1 * 1e3, b1 * 1e3, p1, t2 * 1e3, b2 * 1e3, p2);
    assert!(b1 > 0.8 * t1 && t1 > 0.02 && p1.abs() < 0.05, "the cell folds: top {t1}, bottom {b1}, pitch {p1}");
    assert!(b2 < 0.5 * t2, "independent chords turn: top {t2}, bottom {b2}");
}

/// Telemetry: one beam's force against shortening for the JBeam rule (hardening, MIN_REST),
/// `deformLimitStress` flat, and the crush law, beside the section's target (peak, plateau, 73 %
/// stroke), quasi-static; written to out/beam-law/one_beam.csv.
#[test]
#[ignore = "telemetry: writes out/beam-law/one_beam.csv"]
fn one_beam_curves_against_the_section() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/beam-law");
    std::fs::create_dir_all(&dir).unwrap();
    let mut f = std::fs::File::create(dir.join("one_beam.csv")).unwrap();
    writeln!(f, "member,law,shortening,force").unwrap();
    for (name, m, len) in [("crash_box", member("crash_box", 0.0015, 0.18), 0.18f32), ("rail_bay", member("front_rail", 0.0018, 0.38), 0.38)] {
        let ct = cut(&m, CK);
        let lp = |law: Option<CrushLaw>, peak: f32, cap: f32| {
            let mut b = beam(0, 1, CK, 0.0, peak, 0.73, law);
            b.deform_limit_stress = cap;
            b
        };
        let laws = [
            ("jbeam_hardening", lp(None, ct.plateau, f32::INFINITY)),
            ("jbeam_flat", lp(None, ct.plateau, ct.plateau)),
            ("crush_law", lp(Some(law(ct.plateau, ct.drop, MIN_REST_FLOOR, ct.tension)), ct.peak, f32::INFINITY)),
        ];
        for (lname, b) in laws {
            let mut s = spawn(&[(Vec3::ZERO, 1.0), (Vec3::new(0.0, 0.0, len), 1.0)], vec![b]);
            let (c, plastic) = drive(&mut s, &ramp(0.0, 0.85 * len, 20_000));
            for x in c.iter().step_by(20) {
                writeln!(f, "{name},{lname},{:.5},{:.2}", x.0, x.1).unwrap();
            }
            let stroke = c.iter().rev().find(|x| x.2 > s.beams[0].rest + 1e-7).map_or(0.0, |x| x.0);
            let w = c.iter().filter(|x| x.0 <= 0.73 * len).collect::<Vec<_>>();
            let e73 = work_in(&w.iter().map(|x| **x).collect::<Vec<_>>());
            println!("{name} {lname}: densifies at {:.3} m ({:.0} %), energy to 73 % {:.2} kJ ({:.1} kJ/m), plastic {:.2} kJ", stroke, 100.0 * stroke / len, e73 / 1e3, e73 / 1e3 / f64::from(0.73 * len), plastic / 1e3);
        }
        let target = 4.0 * ct.plateau * 0.73 * len;
        println!("{name}: section peak {:.1} kN, plateau {:.1} kN, drop {:.1} mm (a chord: a quarter); target to 73 %: {:.2} kJ ({:.1} kJ/m), per chord {:.2} kJ", 4.0 * ct.peak / 1e3, 4.0 * ct.plateau / 1e3, ct.drop * 1e3, target / 1e3, target / 1e3 / (0.73 * len), target / 4e3);
        writeln!(f, "{name},target_peak,0,{:.2}", ct.peak).unwrap();
        writeln!(f, "{name},target_plateau,{:.5},{:.2}", 0.73 * len, ct.plateau).unwrap();
    }
}

/// Telemetry: the off-centre ram on the crash box and a rail bay, cell against independent chords
/// against the JBeam rule, at eccentricities 0, 0.3, 1 and 2 × M_c/N; out/beam-law/off_centre_*.csv.
#[test]
#[ignore = "telemetry: writes out/beam-law/off_centre_*.csv"]
fn off_centre_cells() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/beam-law");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, m, len) in [("crash_box", member("crash_box", 0.0015, 0.18), 0.18f32), ("rail_bay", member("front_rail", 0.0018, 0.38), 0.38)] {
        let ct = cut(&m, CK);
        let ecc = ct.moment[0] / (4.0 * ct.plateau);
        println!("{name}: plateau {:.1} kN, strong collapse moment {:.2} kN·m, reserve eccentricity M/N {:.1} mm", 4.0 * ct.plateau / 1e3, ct.moment[0] / 1e3, ecc * 1e3);
        println!("  law              e/ecc  top (mm) bottom (mm) pitch (rad) mean ram force (kN) stroke (m)");
        let cx = cut_x(&m, CK);
        for (lname, cell, crush, x) in [("crush_cell_x", true, true, true), ("crush_chords_x", false, true, true), ("crush_cell", true, true, false), ("crush_chords", false, true, false), ("jbeam_flat", false, false, false)] {
            for r in [0.0f32, 0.3, 1.0, 2.0] {
                let path = dir.join(format!("off_centre_{name}_{lname}_{r}.csv"));
                let (t, b, p, f, st) = off_centre(cell, crush, r * ecc, len, if x { &cx } else { &ct }, Some(path.to_str().unwrap()));
                println!("  {lname:16} {r:5.1} {:9.1} {:11.1} {p:11.3} {:19.1} {st:10.3}", t * 1e3, b * 1e3, f / 1e3);
            }
        }
    }
}
