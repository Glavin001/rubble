//! The engine handover adds no energy. Chunks rested by the world's own contact model are
//! handed to the rigid-body engine for a frame (`World::attach_engine` on the same
//! world, so the world's contact state is there to hand over). Nothing does work on them
//! but gravity, so their kinetic plus potential energy may grow only by lifting them out
//! of their contact compression (2 x the contacts' stored energy at rest: m g d for a
//! face's equal compressions), plus what the engine does to bodies resting exactly on
//! its own geometry, plus the f32 rounding of their poses into and out of the engine.
//!
//! Ported from `stress-physx/tests/handover_energy.rs` (where PhysX, handed resting
//! debris, pushed out of the ground and out of its neighbours the overlaps the world
//! holds at rest) with rubble as the engine.
//!
//! Cases: a cube rocked onto its face (`cube`); two cubes, the second born 2 cm into the
//! first (`pair`); two stacked cubes bonded by a compliant joint, broken under the top's
//! weight (`split`: layer contact hands the joint's compression over as an
//! indentation, `hand_over_split_joints`). `BLOWUP_METHODS=proven` runs them with
//! `Methods::proven()` (layer contact), else the established methods.

use stress_rubble::RubbleEngine;
use stress_ref::engine::RigidEngine;
use stress_ref::math::{Quat, Vec3};
use stress_ref::scene::{BodyDesc, GroundDesc, Scene, SolveMode};
use stress_ref::world::World;

fn methods() -> stress_ref::scene::Methods {
    if std::env::var("BLOWUP_METHODS").is_ok_and(|v| v == "proven") {
        stress_ref::scene::Methods::proven()
    } else {
        stress_ref::scene::Methods::established()
    }
}

fn scene(bodies: Vec<BodyDesc>) -> Scene {
    use stress_ref::builders::*;
    let mut s = new_scene("handover", "chunks resting on the ground, handed to the engine");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    // The split case's joint: concrete a thousand times as compliant, so the weight it
    // carries compresses it measurably (24 kN over 1 m: 2.4 mm).
    let mut soft = oracle_concrete(None);
    soft.youngs_modulus *= 1e-3;
    s.materials.insert("soft".into(), soft);
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    s.bodies = bodies;
    s.sim.methods = methods();
    s.sim.solve_mode = SolveMode::Adaptive;
    s
}

fn cube(name: &str, at: Vec3, q: Quat) -> BodyDesc {
    use stress_ref::builders::*;
    let mut b = body(name, vec![chunk(Vec3::ZERO, Vec3::splat(0.5), "concrete")], Vec::new());
    b.position = at.to_array();
    b.orientation = q.to_wxyz();
    b
}

/// Kinetic plus potential energy of every active chunk.
fn mechanical(w: &World) -> f64 {
    let m = &w.solver;
    let g = m.config.gravity.norm();
    let mut e = 0.0;
    for s in 0..m.structures.len() {
        for c in 0..m.structures[s].chunks.len() {
            if !m.chunks[s][c].active {
                continue;
            }
            let ch = &m.structures[s].chunks[c];
            let (v, om) = m.chunk_velocity(s, c);
            let cl = &m.clusters[m.chunks[s][c].cluster];
            let r = cl.rotation() * ch.rotation;
            let iw = r * ch.inertia * r.transpose();
            e += 0.5 * ch.mass * v.norm2() + 0.5 * om.dot(iw * om) + ch.mass * g * m.chunk_position(s, c).z;
        }
    }
    e
}

/// The total weight and the farthest chunk centre (for the f32 rounding of the poses).
fn weight_and_reach(w: &World) -> (f64, f64) {
    let m = &w.solver;
    let g = m.config.gravity.norm();
    let mut weight = 0.0;
    let mut far: f64 = 0.0;
    for s in 0..m.structures.len() {
        for c in 0..m.structures[s].chunks.len() {
            weight += m.structures[s].chunks[c].mass * g;
            far = far.max(m.chunk_position(s, c).norm());
        }
    }
    (weight, far)
}

/// Rest `w` for `frames`, then hand it to `engine` for one frame: the change of kinetic
/// plus potential energy, the contacts' stored energy at rest, and the world.
fn rest_then_hand_over(mut w: World, frames: usize, engine: Box<dyn RigidEngine>, label: &str) -> (f64, f64, World) {
    for _ in 0..frames {
        w.step_frame();
    }
    let stored = w.contact.stored;
    let e0 = mechanical(&w);
    let chunks: Vec<(usize, usize)> = (0..w.solver.structures.len()).flat_map(|s| (0..w.solver.structures[s].chunks.len()).map(move |c| (s, c))).collect();
    let x0: Vec<Vec3> = chunks.iter().map(|&(s, c)| w.solver.chunk_position(s, c)).collect();
    eprintln!("{label}: at the handover the chunk centres are {:?}", x0.iter().map(|x| x.to_array().map(|v| (v * 1e6).round() / 1e6)).collect::<Vec<_>>());
    w.attach_engine(engine);
    w.step_frame();
    let e1 = mechanical(&w);
    let moved = chunks.iter().zip(&x0).map(|(&(s, c), x)| (w.solver.chunk_position(s, c) - *x).norm()).fold(0.0, f64::max);
    let driven = w.solver.clusters.iter().all(|c| c.driven || c.anchored);
    eprintln!("{label}: rested with contacts storing {stored:.3e} J; the engine (driving every body: {driven}) moved a chunk up to {moved:.3e} m, kinetic + potential {:+.3e} J", e1 - e0);
    (e1 - e0, stored, w)
}

/// A case's handover against its allowance: what the engine adds to the same bodies
/// resting exactly on its geometry (`touching`, handed over at once), the lift out of the
/// compression, the rounding.
fn check(label: &str, make: &dyn Fn() -> Box<dyn RigidEngine>, rested: World, frames: usize, touching: Scene) {
    let (gain, stored, w) = rest_then_hand_over(rested, frames, make(), label);
    let (own, _, _) = rest_then_hand_over(World::new(&touching), 0, make(), &format!("{label}, just touching"));
    let (weight, far) = weight_and_reach(&w);
    let rounding = weight * 4.0 * (f32::EPSILON as f64 / 2.0) * far.max(0.5);
    // The engine's own losses at rest (its solver lets a stack sink) are not credited.
    let allowance = 2.0 * stored + own.max(0.0) + rounding;
    eprintln!("{label}: gain {gain:+.3e} J against {allowance:.3e} J (compression {:.3e}, the engine's own {own:+.3e}, rounding {rounding:.3e})", 2.0 * stored);
    assert!(gain <= allowance, "{label}: the handover added {gain:.3e} J, more than the engine adds at its own rest plus lifting the chunks out of their contact compression ({allowance:.3e} J)");
}

fn cube_case(label: &str, make: &dyn Fn() -> Box<dyn RigidEngine>) {
    // Dropped on a face 4 degrees off flat from 1 cm: the world rests it where its
    // contact holds it (established: rocked onto a pulled-in corner, its true corner
    // below the ground).
    let q = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.07);
    let lowest = 0.5 * (0.07f64.cos() + 0.07f64.sin());
    let rested = World::new(&scene(vec![cube("cube", Vec3::new(0.0, 0.0, lowest + 0.01), q)]));
    check(label, make, rested, 120, scene(vec![cube("cube", Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY)]));
}

fn pair_case(label: &str, make: &dyn Fn() -> Box<dyn RigidEngine>) {
    // Two cubes on the ground, the second born 2 cm into the first (as chunks of a split
    // can be): the world keeps that overlap at rest.
    let rested = World::new(&scene(vec![cube("a", Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY), cube("b", Vec3::new(0.98, 0.0, 0.5), Quat::IDENTITY)]));
    let touching = scene(vec![cube("a", Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY), cube("b", Vec3::new(1.0, 0.0, 0.5), Quat::IDENTITY)]);
    check(label, make, rested, 120, touching);
}

fn split_case(label: &str, make: &dyn Fn() -> Box<dyn RigidEngine>) {
    use stress_ref::builders::*;
    // Two stacked cubes, one body, bonded by a compliant joint carrying the top's weight
    // (2.4 mm of compression). Rested 1 s, the joint broken under that load (its damage
    // complete: the cluster splits; layer contact hands the compression over as an
    // indentation), rested 1 s more, then handed over.
    let stack = || grid(Vec3::new(-0.5, -0.5, 0.0), [1, 1, 2], Vec3::splat(1.0), "concrete");
    let chunks = stack();
    let bonds = auto_bonds(&chunks, |_, _| "soft".into());
    // Explicit: the joint's force is evaluated every substep (a settled cluster's would
    // be its static solve's), so the split hands over the compression it carries.
    let mut s = scene(vec![body("stack", chunks, bonds)]);
    s.sim.solve_mode = SolveMode::Explicit;
    let mut w = World::new(&s);
    for _ in 0..60 {
        w.step_frame();
    }
    eprintln!("{label}: before the split the centres are {:?}, the joint's force {:?}", (0..2).map(|c| w.solver.chunk_position(0, c).z).collect::<Vec<_>>(), w.solver.bonds[0][0].elastic.lin.to_array());
    let (e_before, joint) = (mechanical(&w), w.solver.bonds[0][0].stored);
    w.break_bond(0, 0);
    assert_eq!(w.solver.clusters.len(), 2, "the stack split");
    for f in 0..60 {
        w.step_frame();
        if f == 59 {
            eprintln!("{label}: {} frames after the split: centres {:?}, kinetic + potential {:+.4e} J from before it (the joint stored {joint:.4e} J), contacts storing {:.4e} J", f + 1, (0..2).map(|c| w.solver.chunk_position(0, c).z).collect::<Vec<_>>(), mechanical(&w) - e_before, w.contact.stored);
        }
    }
    let parts = stack();
    let mut touching = scene(vec![body("bottom", vec![parts[0].clone()], Vec::new()), body("top", vec![parts[1].clone()], Vec::new())]);
    touching.sim.solve_mode = SolveMode::Explicit;
    check(label, make, w, 0, touching);
}

fn rubble() -> Option<Box<dyn RigidEngine>> {
    RubbleEngine::boxed(Vec3::new(0.0, 0.0, -9.81))
}

#[test]
fn cube_handover_to_rubble_adds_no_energy() {
    if rubble().is_none() {
        return eprintln!("no wgpu adapter; skipping");
    }
    cube_case("cube, rubble", &|| rubble().expect("rubble"));
}

#[test]
fn pair_handover_to_rubble_adds_no_energy() {
    if rubble().is_none() {
        return eprintln!("no wgpu adapter; skipping");
    }
    pair_case("pair, rubble", &|| rubble().expect("rubble"));
}

/// Known gap (findings/rubble-engine.md): rubble resolves `1 - alpha` (1 %) of the
/// 0.7 mm overlap the world holds between the split halves per step and lifts the
/// top by 17 um in the frame: +0.51 J against an allowance of 0.46 J (established).
/// With `BLOWUP_METHODS=proven` the world hands the overlap over as an indentation
/// rubble cannot hold (`set_rest_overlaps` is not implemented): +0.51 J against 0.11 J.
#[test]
#[ignore = "known gap: rubble does not hold the world's rest overlaps (findings/rubble-engine.md)"]
fn split_handover_to_rubble_adds_no_energy() {
    if rubble().is_none() {
        return eprintln!("no wgpu adapter; skipping");
    }
    split_case("split, rubble", &|| rubble().expect("rubble"));
}
