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

/// The full-state contract's ceiling (decoded state /2, `decoded_full::island_frame`): the
/// oracle's own outcome through it, every column and the decode's own body numbering (its
/// breaks in the oracle's time order, a substep's breaks together), is the standalone
/// oracle bit for bit: the world's fingerprint and the island's whole state every frame. The fields the
/// columns do not carry (the contact history, the per-chunk world arrays) are the native
/// step's here (`Base::Native`).
#[test]
fn the_oracle_through_decoded_state_2_is_the_oracle() {
    use stress_learned::decoded_full::{island_frame, oracle_full2, Base};
    for scene in scenes() {
        let (mut native, mut harness) = (World::new(&scene), World::new(&scene));
        let mut routed = 0;
        for frame in 0..frames(&scene) {
            native.solver.reset_peak_utilizations();
            native.step_frame();
            let m = harness
                .step_frame_island_provider(&mut |w: &World, m: &IslandMembers| {
                    let (d, full) = oracle_full2(w, m, frame as i64, None)?;
                    let st = island_frame(w, m, &d, &full, Base::Native, false)?;
                    Ok(st)
                })
                .unwrap();
            routed += usize::from(!m.chunks.is_empty());
            assert_eq!(stress_learned::outcome::fingerprint(&native), stress_learned::outcome::fingerprint(&harness), "{}: frame {frame}: fingerprint", scene.name);
            // the island's whole state, bit for bit (the pure caches aside)
            if !m.chunks.is_empty() && stress_learned::relabel::state_bytes(&native.extract_island(&m).without_caches()).unwrap() != stress_learned::relabel::state_bytes(&harness.extract_island(&m).without_caches()).unwrap() {
                // which parts of the island's state differ
                let (a, b) = (stress_learned::decoded_full::to_value(&harness.extract_island(&m)).unwrap(), stress_learned::decoded_full::to_value(&native.extract_island(&m)).unwrap());
                let parts = |v: &rmpv::Value| -> Vec<(String, rmpv::Value)> {
                    let mut out = Vec::new();
                    for (k, x) in v.as_map().unwrap() {
                        let k = k.as_str().unwrap().to_string();
                        match x.as_map() {
                            Some(inner) if k == "solver" => out.extend(inner.iter().map(|(k2, y)| (format!("solver.{}", k2.as_str().unwrap()), y.clone()))),
                            _ => out.push((k, x.clone())),
                        }
                    }
                    out
                };
                let diff: Vec<String> = parts(&a).into_iter().zip(parts(&b)).filter(|(x, y)| x.1 != y.1).map(|(x, _)| x.0).collect();
                let (ha, na) = (harness.extract_island(&m), native.extract_island(&m));
                let (x, y) = (stress_learned::decoded_full::columns_of_state(&ha, None).unwrap(), stress_learned::decoded_full::columns_of_state(&na, None).unwrap());
                for (p, q) in x.iter().zip(&y) {
                    let at: std::collections::HashMap<i64, usize> = q.keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
                    let mut worst = std::collections::BTreeMap::<String, f64>::new();
                    for (i, k) in p.keys.iter().enumerate() {
                        let Some(&j) = at.get(k) else { worst.insert(format!("missing {k}"), 1.0); continue };
                        for (ci, c) in p.columns.iter().enumerate() {
                            let Some(cj) = q.columns.iter().position(|d| d == c) else { continue };
                            let (u, v) = (p.rows[i * p.columns.len() + ci], q.rows[j * q.columns.len() + cj]);
                            if u.to_bits() != v.to_bits() && !(u.is_nan() && v.is_nan()) {
                                let e = worst.entry(c.clone()).or_default();
                                *e = e.max((u - v).abs());
                            }
                        }
                    }
                    eprintln!("  {}: {worst:?}", p.kind);
                }
                eprintln!("  next_cluster_id harness {} native {}; clusters {} / {}", ha.solver.next_cluster_id, na.solver.next_cluster_id, ha.solver.clusters.len(), na.solver.clusters.len());
                panic!("{}: frame {frame} differs; island parts {diff:?}", scene.name);
            }
        }
        assert!(routed > 0);
        eprintln!("{}: {} frames through /2, identical", scene.name, frames(&scene));
    }
}

/// The contact bounds of the decode (`stress_learned::contact_bound`, planning 2026-10-22)
/// never act on the oracle's own island frames: no contact ends deeper than
/// `max(depth_start, sqrt(2 E_avail / k))` (so `project` returns the state as it is, bit for
/// bit), and the event's energy budget with the contacts' energy counted is never exceeded.
#[test]
fn the_contact_bounds_never_act_on_the_oracle() {
    for scene in scenes() {
        let mut w = World::new(&scene);
        let mut ledger = stress_learned::energy_budget::Ledger::default();
        let (mut calls, mut contacts) = (0, 0);
        for frame in 0..frames(&scene) {
            w.step_frame_island_provider(&mut |start: &World, m: &IslandMembers| {
                let out = oracle_island_step(start, m)?;
                let end = World::merge_island(start, &out, m);
                let ((k0, e0, p0), (k1, e1, p1)) = (start.island_energy(m), end.island_energy(m));
                let ec0 = start.island_contact_energy(m);
                let remaining = ledger.open(k0 + e0 + ec0);
                let e_avail = stress_learned::contact_bound::available_energy(start, m, remaining);
                let (cs, _, ec1) = stress_learned::contact_bound::check(start, &end, m, e_avail);
                for c in &cs {
                    assert!(!c.over(), "{}: frame {frame}: contact {c:?} over the bound", scene.name);
                }
                let (projected, p) = stress_learned::contact_bound::project(start, m, &out, e_avail);
                assert_eq!(p.clamps, 0, "{}: frame {frame}", scene.name);
                assert_eq!(stress_learned::relabel::state_bytes(&projected).unwrap(), stress_learned::relabel::state_bytes(&out).unwrap(), "{}: frame {frame}: projection changed the oracle's state", scene.name);
                let delta = (k1 + e1 + p1 + ec1) - (k0 + e0 + p0 + ec0);
                assert!(delta <= remaining * (1.0 + 1e-9) + 1e-12, "{}: frame {frame}: delta {delta} over the remaining budget {remaining}", scene.name);
                ledger.close(delta);
                calls += 1;
                contacts += cs.len();
                Ok(out)
            })
            .unwrap();
        }
        eprintln!("{}: {calls} routed calls, {contacts} frame-end contacts, ledger {ledger:?}", scene.name);
        assert!(calls > 0 && contacts > 0, "{}: nothing checked", scene.name);
    }
}
