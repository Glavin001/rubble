//! Harness A of the learned-impact plan (2026-10-22): the standalone oracle with each routed
//! island's frame supplied through the provider interface (`World::step_frame_island_provider`,
//! the island's full state across the boundary). No rubble.
//!
//! 1. With the oracle's island step as provider it is the standalone oracle bit for bit,
//!    every frame of whole events (impact, debris flight, settling).
//! 2. A provider whose island differs (the oracle's island step from a perturbed start) is
//!    applied to the island alone: every chunk outside the island is the native frame's,
//!    bit for bit, and the world stays consistent (each chunk's cluster holds it).

use stress_ref::scene::Scene;
use stress_ref::world::{oracle_island_step, IslandMembers, IslandState, World};
use stress_rubble::scenes::realtime;

#[path = "rewind_scenes/mod.rs"]
mod rewind_scenes;

fn scenes() -> Vec<Scene> {
    let row = realtime(rewind_scenes::wall_row(2, 40.0, 0.1, 1.0), 1.0);
    let b5 = realtime(stress_ref::builders::wall_impact(20.0, "b5_wall_impact_v20"), 1.0);
    vec![row, b5]
}

fn frames(s: &Scene) -> usize {
    (s.sim.duration / s.sim.frame_dt).round() as usize
}

#[test]
fn the_oracle_island_step_through_the_provider_is_the_standalone_oracle() {
    for scene in scenes() {
        let (mut native, mut harness) = (World::new(&scene), World::new(&scene));
        let mut routed = 0;
        let mut chunks_routed = 0;
        for frame in 0..frames(&scene) {
            native.solver.reset_peak_utilizations();
            native.step_frame();
            let m = harness.step_frame_island_provider(&mut |w: &World, m: &IslandMembers| oracle_island_step(w, m)).unwrap();
            routed += usize::from(!m.chunks.is_empty() || !m.impactors.is_empty());
            chunks_routed += m.chunks.len();
            assert_eq!(stress_learned::outcome::fingerprint(&native), stress_learned::outcome::fingerprint(&harness), "{}: frame {frame} differs", scene.name);
            assert_eq!(native.solver.clusters.len(), harness.solver.clusters.len(), "{}: frame {frame}: clusters", scene.name);
            // each frame's peak utilisation as the recorder's (the next call's input)
            let peaks = |w: &World| w.solver.bonds.iter().flatten().map(|b| b.peak_utilization.to_bits()).collect::<Vec<_>>();
            assert_eq!(peaks(&native), peaks(&harness), "{}: frame {frame}: peak utilisation", scene.name);
        }
        eprintln!("{}: {} frames identical; {routed} routed, {chunks_routed} chunk-frames in islands", scene.name, frames(&scene));
    }
}

#[test]
fn a_different_island_changes_the_island_alone() {
    for scene in scenes() {
        let mut w = World::new(&scene);
        let mut differed = 0;
        for frame in 0..frames(&scene).min(20) {
            let mut native = w.clone();
            native.solver.reset_peak_utilizations();
            native.step_frame();
            // the provider's island: the oracle's step from a start with the impactors 1% faster
            let mut provider = |start: &World, m: &IslandMembers| -> Result<IslandState, String> {
                let mut p = start.clone();
                for imp in &mut p.impactors {
                    imp.velocity = imp.velocity * 1.01;
                }
                oracle_island_step(&p, m)
            };
            let m = w.step_frame_island_provider(&mut provider).unwrap();
            for s in 0..w.solver.structures.len() {
                for c in 0..w.solver.structures[s].chunks.len() {
                    let ci = w.solver.chunks[s][c].cluster;
                    if w.solver.chunks[s][c].active {
                        assert!(w.solver.clusters[ci].chunks.contains(&c) && w.solver.clusters[ci].structure == s, "{}: frame {frame}: chunk ({s},{c}) not in its cluster", scene.name);
                    }
                    if !m.chunks.contains(&(s, c)) {
                        assert_eq!(w.solver.chunk_position(s, c), native.solver.chunk_position(s, c), "{}: frame {frame}: rest chunk ({s},{c}) moved", scene.name);
                    } else if w.solver.chunk_position(s, c) != native.solver.chunk_position(s, c) {
                        differed += 1;
                    }
                }
            }
            // continue from the native frame, so every frame starts as the oracle's
            w = native;
        }
        assert!(differed > 0, "{}: the provider's island never showed", scene.name);
        eprintln!("{}: island chunk-frames that took the provider's outcome: {differed}", scene.name);
    }
}
