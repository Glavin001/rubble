//! Continuous contact against a known physical solution, not against our own earlier output
//! (owner, 2026-11-26): two identical elastic bars meet end to end. Elastic theory (the 1-D
//! wave equation, Saint-Venant's impact of rods) gives the exact answer: the contact lasts
//! 2L/c (c = √(E/ρ), the bar's wave speed), the bars part with the speeds they arrived with
//! reversed, and the mechanical energy (kinetic + elastic) is conserved: with undamped,
//! unbreakable beams nothing may take it away. The bars here are square prisms of node-beams
//! with a collision skin, so the contact is the cars' continuous contact.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test contact_known_solutions -- --nocapture`

use engine_softbody::desc::{SoftBeam, SoftNode, SoftTriangle, SoftVehicleDesc};
use engine_softbody::explicit_beams::{ExplicitBeams, STEP};
use engine_softbody::explicit_contact::{Dat, ExplicitBeamsWorld};
use glam::Vec3;

/// Slices along the bar, the slice spacing (m), the square's side (m), a node's mass (kg) and
/// the beams' spring (N/m).
const SLICES: usize = 11;
const DX: f32 = 0.05;
const SIDE: f32 = 0.1;
const MASS: f32 = 0.5;
const K: f32 = 2.0e5;

/// A square prism bar along x from 0 to (SLICES − 1)·DX: corners per slice, every edge and
/// face diagonal a beam (no damping, no yield, no break), its surface the collision skin
/// (outward).
fn bar() -> SoftVehicleDesc {
    let mut d = SoftVehicleDesc::default();
    let id = |s: usize, c: usize| (4 * s + c) as u32;
    for s in 0..SLICES {
        for c in 0..4 {
            let (y, z) = [(0.0, 0.0), (SIDE, 0.0), (SIDE, SIDE), (0.0, SIDE)][c];
            d.nodes.push(SoftNode { name: format!("b{s}_{c}"), position: [s as f32 * DX, y, z], mass: MASS, collision: true, contacter: true, ..Default::default() });
        }
    }
    let mut beam = |a: u32, b: u32| {
        d.beams.push(SoftBeam { a, b, spring: K, damp: 0.0, deform: f32::MAX, strength: f32::MAX, ..Default::default() });
    };
    for s in 0..SLICES {
        for c in 0..4 {
            beam(id(s, c), id(s, (c + 1) % 4));
        }
        beam(id(s, 0), id(s, 2));
        beam(id(s, 1), id(s, 3));
        if s + 1 < SLICES {
            for c in 0..4 {
                beam(id(s, c), id(s + 1, c));
                beam(id(s, c), id(s + 1, (c + 1) % 4));
                beam(id(s, (c + 1) % 4), id(s + 1, c));
            }
        }
    }
    // The skin: the four long faces and the two ends, wound outward.
    let centre = Vec3::new((SLICES - 1) as f32 * DX * 0.5, SIDE * 0.5, SIDE * 0.5);
    let mut tri = |t: [u32; 3], d: &mut SoftVehicleDesc| {
        let p = t.map(|i| Vec3::from_array(d.nodes[i as usize].position));
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        let t = if n.dot((p[0] + p[1] + p[2]) / 3.0 - centre) < 0.0 { [t[0], t[2], t[1]] } else { t };
        d.triangles.push(SoftTriangle { nodes: t, collision: true, ..Default::default() });
    };
    for s in 0..SLICES - 1 {
        for c in 0..4 {
            let (a, b, e, f) = (id(s, c), id(s, (c + 1) % 4), id(s + 1, c), id(s + 1, (c + 1) % 4));
            tri([a, b, f], &mut d);
            tri([a, f, e], &mut d);
        }
    }
    for s in [0, SLICES - 1] {
        tri([id(s, 0), id(s, 1), id(s, 2)], &mut d);
        tri([id(s, 0), id(s, 2), id(s, 3)], &mut d);
    }
    d
}

fn spawn(d: &SoftVehicleDesc, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let mut s = ExplicitBeams::from_desc_jbeam_raycast(d, at, vel, engine_softbody_authoring::vehicle_compare::settings()).expect("spawn").0;
    // (BARS_BARE=1: no surface or edge contact, a diagnosis.)
    let bare = std::env::var("BARS_BARE").is_ok_and(|x| x == "1");
    s.surface_contact = !bare;
    s.edge_contact = !bare;
    if bare {
        s.candidates.clear();
    }
    s.finish();
    s
}

/// Kinetic energy and centre-of-mass velocity along x of a body.
fn motion(s: &ExplicitBeams) -> (f64, f64) {
    let m: f64 = s.nodes.iter().map(|n| n.mass as f64).sum();
    let ke = s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum();
    (ke, s.nodes.iter().map(|n| n.mass as f64 * n.vel.x as f64).sum::<f64>() / m)
}

/// Kinetic + each beam's ½k(ℓ − L)² + gravity's potential, by hand.
fn mechanical(w: &ExplicitBeamsWorld) -> f64 {
    w.vehicles.iter().map(|s| {
        let ke: f64 = s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum();
        let pe: f64 = s.beams.iter().map(|b| { let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length(); 0.5 * b.k as f64 * ((l - b.rest) as f64).powi(2) }).sum();
        let g: f64 = s.nodes.iter().map(|n| n.mass as f64 * 9.81 * (s.origin.y + n.pos.y) as f64).sum();
        ke + pe + g
    }).sum()
}

/// The integrator's own conserved energy (symplectic Euler, kick then drift: the state is
/// x_{n+1} with v_{n+1/2}): E = KE + V + (h/2)·Σ v·F(x), F the beams' and gravity's force at x.
/// Exact for a quadratic potential; the raw KE + V swings by O(ωh) with the bars' ringing
/// (ωh ≈ 1 here), so it is the measure an impact's energy is judged by.
fn shadow(w: &ExplicitBeamsWorld) -> f64 {
    mechanical(w) + w.vehicles.iter().map(|s| {
        let mut f: Vec<glam::DVec3> = s.nodes.iter().map(|n| glam::DVec3::new(0.0, -9.81 * n.mass as f64, 0.0)).collect();
        for b in &s.beams {
            let d = (s.nodes[b.b as usize].pos - s.nodes[b.a as usize].pos).as_dvec3();
            let l = d.length();
            if l <= 0.0 { continue; }
            let fb = d / l * (b.k as f64 * (l - b.rest as f64));
            f[b.a as usize] += fb;
            f[b.b as usize] -= fb;
        }
        0.5 * STEP as f64 * s.nodes.iter().zip(&f).map(|(n, f)| n.vel.as_dvec3().dot(*f)).sum::<f64>()
    }).sum::<f64>()
}

/// Two bars head on at `v` m/s each, 1 cm apart, high above the ground: the run's
/// (energy kept, rebound speed over arrival speed, contact time over 2L/c).
fn bars_head_on(v: f32) -> (f64, f64, f64) {
    let d = bar();
    let len = (SLICES - 1) as f32 * DX;
    let a = spawn(&d, Vec3::new(-len - 0.005, 5.0, 0.0), Vec3::X * v);
    let b = spawn(&d, Vec3::new(0.005, 5.0, 0.0), -Vec3::X * v);
    let span = |s: &ExplicitBeams| s.nodes.iter().map(|n| s.origin.x + n.pos.x).fold((f32::MAX, f32::MIN), |(l, h), x| (l.min(x), h.max(x)));
    println!("| start | bar A x {:?}, bar B x {:?} |", span(&a), span(&b));
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    w.continuous = Some(Dat::new(0.08, 0.85));
    // (BARS_CC=0: no continuous contact at all, a diagnosis.)
    if std::env::var("BARS_CC").is_ok_and(|x| x == "0") {
        w.continuous = None;
    }
    // (BARS_SELF=0: no self contact within a bar, a diagnosis.)
    if let (true, Some(d)) = (std::env::var("BARS_SELF").is_ok_and(|x| x == "0"), w.continuous.as_mut()) {
        d.self_contact = false;
    }
    // (BARS_VC=0: no discrete vehicle contact pass, a diagnosis.)
    if std::env::var("BARS_VC").is_ok_and(|x| x == "0") {
        w.contact = false;
    }
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().start;
    let (own0, sh0) = (mechanical(&w), shadow(&w));
    let ke0: f64 = w.vehicles.iter().map(|s| motion(s).0).sum();
    // c = DX·√(k_axial/m): four longitudinal and eight diagonal beams carry the axial load;
    // the measured contact time is compared with 2L/c of the bar's own long-wave speed.
    let (mut first, mut last) = (None, None);
    let steps = (0.2 / STEP) as usize;
    for s in 0..steps {
        w.step();
        if std::env::var("BARS_TRACE").is_ok() && (s % 20 == 0 || (60..90).contains(&s)) {
            let l = w.ledger.as_ref().unwrap();
            println!("trace {s} shadow {:+.4} raw {:+.4} | air {:+.4} beams {:+.4} self {:+.4} slides {:+.4} cn {:+.4} cf {:+.4} gv {:+.4} gh {:+.4} cshift {:+.4} gshift {:+.4}", shadow(&w) - sh0, mechanical(&w) - own0, l.w_air, l.w_beams, l.w_self_contact, l.w_slides, l.w_contact_normal, l.w_contact_friction, l.w_ground_vertical, l.w_ground_horizontal, l.w_contact_shift, l.w_ground_shift);
        }
        let (_, va) = motion(&w.vehicles[0]);
        if va < 0.9 * v as f64 && first.is_none() {
            first = Some(s);
        }
        if first.is_some() && last.is_none() && va < -0.5 * v as f64 {
            last = Some(s);
        }
    }
    let l = w.ledger.as_ref().unwrap();
    let kept = l.mechanical() / e0;
    let c = w.continuous.as_ref().map(|d| d.total).unwrap_or_default();
    println!("| contact | CCD impacts {}, rounds {}, row iterations {}, zones {}, unresolved {}, speculative KE {:.1} J, mechanical energy changed in the contact pass {:.1} J (DAT_ENERGY=1) |", c.ccd.hits, c.ccd.rounds, c.ccd.row_iters, c.ccd.zones, c.ccd.unresolved, c.spec_ke, c.contact_de);
    println!("| by hand | mechanical now {:.1} J, start {:.1} J (ledger {:.1} -> {:.1}) |", mechanical(&w), own0, e0, l.mechanical());
    // Energy kept, by the integrator's conserved energy, over the bars' kinetic energy at the start.
    let kept_shadow = 1.0 + (shadow(&w) - sh0) / ke0;
    println!("| shadow | conserved energy changed {:+.2} J of {:.1} J kinetic: kept {:.4} |", shadow(&w) - sh0, ke0, kept_shadow);
    let (_, va) = motion(&w.vehicles[0]);
    let (_, vb) = motion(&w.vehicles[1]);
    let rebound = 0.5 * (vb - va) / v as f64;
    // The bar's long-wave speed from its axial stiffness per slice (4 edges + 8 diagonals'
    // axial share, cos² of the diagonal's angle) and its mass per slice.
    let diag = (DX * DX + SIDE * SIDE).sqrt();
    let k_axial = 4.0 * K + 8.0 * K * (DX / diag).powi(2);
    let c = DX as f64 * (k_axial as f64 / (4.0 * MASS) as f64).sqrt();
    let t_contact = (last.unwrap_or(steps) as f64 - first.unwrap_or(0) as f64) * STEP as f64;
    let ratio = t_contact / (2.0 * len as f64 / c);
    println!("| bars {v} m/s | energy kept {:.4} | rebound {:.3} of arrival | contact {:.1} ms, {:.2} of 2L/c ({:.1} ms) |", kept, rebound, t_contact * 1e3, ratio, 2e3 * len as f64 / c);
    let _ = kept;
    (kept_shadow, rebound, ratio)
}

/// The elastic bars' mechanical energy is conserved through their impact (elastic theory:
/// exactly; here at most 1 % lost: the discrete bar's own rounding and its contact).
#[test]
fn bars_head_on_conserve_energy() {
    // (BARS_V: another speed, a diagnosis.)
    let v = std::env::var("BARS_V").ok().and_then(|x| x.parse().ok()).unwrap_or(5.0);
    let (kept, _, _) = bars_head_on(v);
    assert!(kept > 0.99 && kept < 1.0 + 1e-3, "kept {kept}");
}

/// The bars part with their arrival speed reversed (elastic theory: exactly; the discrete
/// bar's own ringing and the rounding of the measure allow 10 %). FAILS today (2026-11-26):
/// 0.40 of the arrival speed at e = 0, the rest kept as the bars' ringing (energy is kept).
#[test]
fn bars_head_on_rebound() {
    let (_, rebound, _) = bars_head_on(5.0);
    assert!((rebound - 1.0).abs() < 0.1, "rebound {rebound}");
}
