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
    s.surface_contact = true;
    s.edge_contact = true;
    s.finish();
    s
}

/// Kinetic energy and centre-of-mass velocity along x of a body.
fn motion(s: &ExplicitBeams) -> (f64, f64) {
    let m: f64 = s.nodes.iter().map(|n| n.mass as f64).sum();
    let ke = s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum();
    (ke, s.nodes.iter().map(|n| n.mass as f64 * n.vel.x as f64).sum::<f64>() / m)
}

/// Two bars head on at `v` m/s each, 1 cm apart, high above the ground: the run's
/// (energy kept, rebound speed over arrival speed, contact time over 2L/c).
fn bars_head_on(v: f32) -> (f64, f64, f64) {
    let d = bar();
    let len = (SLICES - 1) as f32 * DX;
    let a = spawn(&d, Vec3::new(-len - 0.005, 5.0, 0.0), Vec3::X * v);
    let b = spawn(&d, Vec3::new(0.005, 5.0, 0.0), -Vec3::X * v);
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    w.continuous = Some(Dat::new(0.08, 0.85));
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().start;
    let own0: f64 = w.vehicles.iter().map(|s| {
        let ke: f64 = s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum();
        let pe: f64 = s.beams.iter().map(|b| { let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length(); 0.5 * b.k as f64 * ((l - b.rest) as f64).powi(2) }).sum();
        let g: f64 = s.nodes.iter().map(|n| n.mass as f64 * 9.81 * (s.origin.y + n.pos.y) as f64).sum();
        ke + pe + g
    }).sum();
    // c = DX·√(k_axial/m): four longitudinal and eight diagonal beams carry the axial load;
    // the measured contact time is compared with 2L/c of the bar's own long-wave speed.
    let (mut first, mut last) = (None, None);
    let steps = (0.2 / STEP) as usize;
    for s in 0..steps {
        w.step();
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
    // (The same by hand: kinetic + each beam's ½k(ℓ − L)², gravity's work apart.)
    let own = |w: &ExplicitBeamsWorld| -> f64 {
        w.vehicles.iter().map(|s| {
            let ke: f64 = s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum();
            let pe: f64 = s.beams.iter().map(|b| { let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length(); 0.5 * b.k as f64 * ((l - b.rest) as f64).powi(2) }).sum();
            let g: f64 = s.nodes.iter().map(|n| n.mass as f64 * 9.81 * (s.origin.y + n.pos.y) as f64).sum();
            ke + pe + g
        }).sum()
    };
    let c = w.continuous.as_ref().unwrap().total;
    println!("| contact | CCD impacts {}, zones {}, unresolved {}, speculative KE {:.1} J |", c.ccd.hits, c.ccd.zones, c.ccd.unresolved, c.spec_ke);
    println!("| by hand | mechanical now {:.1} J, start {:.1} J (ledger {:.1} -> {:.1}) |", own(&w), own0, e0, l.mechanical());
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
    (kept, rebound, ratio)
}

/// The elastic bars' mechanical energy is conserved through their impact (elastic theory:
/// exactly; here at most 1 % lost: the discrete bar's own rounding and its contact).
#[test]
fn bars_head_on_conserve_energy() {
    let (kept, _, _) = bars_head_on(5.0);
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
