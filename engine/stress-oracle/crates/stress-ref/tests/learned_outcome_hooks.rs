//! The read-only hooks the learned impact-outcome recorder uses must not change the
//! simulation: a watched frame is a frame, a routing prediction leaves the world as it was,
//! and a world that adopts another profile's model steps like a world built under it.

use std::collections::HashSet;

use stress_ref::builders;
use stress_ref::capability::Profile;
use stress_ref::scene::{Methods, Scene, SolveMode};
use stress_ref::world::World;

/// A free wall (no gravity) struck by a ram, under the A12 real-time configuration.
fn scene() -> Scene {
    let mut s = builders::wall_impact(30.0, "hooks");
    s.gravity = [0.0; 3];
    for b in &mut s.bodies {
        for c in &mut b.chunks {
            c.support = stress_ref::scene::Support::None;
        }
    }
    s.sim.methods = Methods::established();
    s.sim.methods.true_step = true;
    s.sim.courant_safety = 0.35;
    s.sim.stress_courant_safety = Some(0.5);
    s.sim.solve_mode = SolveMode::Adaptive;
    // Start the ram at the face: its strike falls in the first frame.
    s.impactors[0].position[1] = -0.2505;
    s.sim.duration = 1.0;
    s
}

fn state(w: &World) -> Vec<u64> {
    let mut out = Vec::new();
    for s in 0..w.solver.structures.len() {
        for c in 0..w.solver.structures[s].chunks.len() {
            let (v, om) = w.solver.chunk_velocity(s, c);
            for x in [w.solver.chunk_position(s, c), v, om] {
                out.extend(x.to_array().map(f64::to_bits));
            }
        }
    }
    for b in w.solver.bonds.iter().flatten() {
        out.push(b.joint.damage.to_bits());
    }
    for i in &w.impactors {
        out.extend(i.velocity.to_array().map(f64::to_bits));
        out.extend(i.pose.position.to_array().map(f64::to_bits));
    }
    out
}

#[test]
fn watched_frames_and_routing_predictions_change_nothing() {
    let s = scene();
    let (mut a, mut b) = (World::new(&s), World::new(&s));
    let mut substeps = 0usize;
    for _ in 0..2 {
        a.step_frame();
        let (clusters, impactors) = b.predict_island_members(s.sim.frame_dt, &HashSet::new());
        assert!(!clusters.is_empty() && impactors == vec![0], "the strike frame routes the wall and the ram");
        let _ = b.predict_contacts(s.sim.frame_dt, 0.0);
        b.step_frame_watched(|_| substeps += 1).unwrap();
        assert_eq!(state(&a), state(&b));
    }
    assert!(substeps > 100);
    assert!(a.solver.broken_bond_count() > 0, "the test strike breaks bonds");
}

#[test]
fn predicted_punch_is_the_breaking_plugs_of_the_predicted_contacts() {
    let w = World::new(&scene());
    let contacts = w.predict_contacts(1.0 / 60.0, stress_ref::world::tiered_impact_speed());
    assert_eq!(contacts.len(), 1);
    let plug = contacts[0].plug.as_ref().expect("a plug");
    assert!(plug.breaks(), "30 m/s through a 0.2 m wall punches out");
    let mut want: Vec<(usize, usize)> = plug.interface.iter().map(|&b| (contacts[0].structure, b)).collect();
    want.sort_unstable();
    assert_eq!(w.predict_punch(1.0 / 60.0), want);
    assert!(contacts[0].peak_force > 0.0 && contacts[0].hertz_duration > 0.0 && contacts[0].time < 1.0 / 60.0);
}

#[test]
fn adopting_blast_parity_steps_like_a_blast_parity_world() {
    let s = scene();
    let mut qs = s.clone();
    Profile::BlastParity.apply_scene(&mut qs);
    let mut template = World::new(&qs);
    Profile::BlastParity.apply_config(&mut template.solver.config);
    let mut branch = World::new(&s);
    branch.reduce_to_rigid();
    branch.adopt_model(&template).unwrap();
    for _ in 0..2 {
        template.step_frame();
        branch.step_frame();
        assert_eq!(state(&template), state(&branch));
    }
}


/// Every bond's failure margin (`RtBond::fail_margin`) reaches 1 exactly when the joint
/// law disconnects it: brittle and ductile softening, no softening, impact and crushing.
#[test]
fn the_failure_margin_reaches_one_exactly_when_a_bond_disconnects() {
    let mut cases: Vec<(&str, Scene)> = vec![("softening", scene())];
    let mut brittle = scene();
    brittle.sim.features.set("softening", false).unwrap();
    cases.push(("no softening", brittle));
    let mut ductile = scene();
    for m in ductile.materials.values_mut() {
        m.kind = stress_ref::material::BondKind::Ductile;
    }
    cases.push(("ductile", ductile));
    let mut slow_heavy = scene();
    slow_heavy.impactors[0].velocity = [0.0, 8.0, 0.0];
    slow_heavy.impactors[0].mass = 8000.0;
    cases.push(("slow heavy", slow_heavy));
    for (name, s) in cases {
        let mut w = World::new(&s);
        let (mut broke, mut live_near) = (0usize, 0.0f64);
        for _ in 0..3 {
            w.step_frame();
            for b in w.solver.bonds.iter().flatten() {
                let q = b.fail_margin();
                assert_eq!(q >= 1.0, !b.connected(), "{name}: margin {q} with connected {}", b.connected());
                if !b.connected() {
                    broke += 1;
                } else {
                    live_near = live_near.max(q);
                }
            }
        }
        assert!(broke > 0, "{name}: the strike breaks bonds");
        assert!(live_near > 0.0, "{name}: live bonds carry a margin");
    }
}
