//! The learned path's routing (`learned_route.slang`) against the CPU's (`World::
//! select_islands` through `predict_island_members`, P1's recorder's call): after every
//! frame, the bodies the GPU batch routed (its one call's chunk and impactor keys) and the
//! bodies the CPU routing puts in islands from the same state (the coupled world's CPU copy,
//! synced), for the next frame. On the M5 scenes and P1's evaluation recipes; with the coupled
//! world's routing (tiered, the default) and a standalone world's (untiered:
//! `STRESS_LEARNED_ROUTE=recorder`), debris reach off and on, the walls left whole (dry batches) and fractured (the
//! punch-like stub).
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_routing -- --nocapture
#![cfg(feature = "gpu")]

use std::collections::{BTreeSet, HashSet};

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::keys;
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode};
use stress_rubble::scenes::{later, realtime, wall_ram_late, wall_row_late};
use stress_rubble::RubbleEngine;

#[derive(Default)]
struct Tally {
    frames: usize,
    routed: usize,
    agree: usize,
    bad: Vec<String>,
}

fn perturbed(mut scene: Scene, dx: f64, dz: f64) -> Scene {
    for imp in &mut scene.impactors {
        imp.position[0] += dx;
        imp.position[2] += dz;
    }
    scene
}

/// The CPU routing's members from `w`'s state, as chunk keys and impactor keys.
fn cpu_members(w: &mut stress_ref::world::World, tiered: bool) -> (BTreeSet<i64>, BTreeSet<i64>) {
    if !tiered {
        w.qs_islands = false;
    } else {
        w.qs_islands = true;
        w.scene.sim.solve_mode = SolveMode::QuasiStatic;
    }
    let dt = w.scene.sim.frame_dt;
    let (clusters, impactors) = w.predict_island_members(dt, &HashSet::new());
    let ids: HashSet<u64> = clusters.into_iter().collect();
    let mut chunks = BTreeSet::new();
    for cl in &w.solver.clusters {
        if ids.contains(&cl.id) {
            for &c in &cl.chunks {
                chunks.insert(keys::chunk_key(cl.structure, c));
            }
        }
    }
    (chunks, impactors.into_iter().map(keys::impactor_key).collect())
}

fn run(gpu: &Gpu, scene: Scene, tiered: bool, debris: bool, dry: bool, t: &mut Tally) {
    std::env::set_var("STRESS_LEARNED_ROUTE", if tiered { "world" } else { "recorder" });
    std::env::set_var("STRESS_LEARNED_DRY", if dry { "1" } else { "0" });
    let p = Profile::BlastParityRouted;
    let mut scene = scene;
    p.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.set_tiered(true).expect("tiered coupling");
    w.set_debris_reach_routing(debris);
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut batches = 0;
    let mut resident = 0;
    for f in 0..frames {
        w.step_frame(gpu).expect("frame");
        let Some(l) = w.learned.as_ref() else { continue };
        // a frame the learned path did not follow (a coupled one) is not compared
        if l.totals.resident_frames == resident {
            continue;
        }
        resident = l.totals.resident_frames;
        // a frame without a batch routed nothing on the GPU (tiered: the routing check
        // found no impactor near; its flag is a superset of the CPU's seeds)
        let batched = l.totals.batches > batches;
        batches = l.totals.batches;
        let export = if batched { l.export(gpu) } else { Vec::new() };
        // the batch's hold zeroed its bodies' motion after routing: given back to the CPU's copy
        let held = if batched { l.held_motion(gpu) } else { (Vec::new(), Vec::new()) };
        assert!(export.len() <= 1, "one call per frame");
        let (g_chunks, g_imps): (BTreeSet<i64>, BTreeSet<i64>) = match export.first() {
            Some(c) => (c.chunk_keys.iter().copied().collect(), c.impactor_keys.iter().copied().collect()),
            None => (BTreeSet::new(), BTreeSet::new()),
        };
        w.sync(gpu);
        let all: Vec<usize> = (0..w.solver.mirror.clusters.len()).collect();
        w.solver.sync_chunk_state(gpu, &all);
        let mut world = w.with_shell(|s| s.standalone_copy());
        let zup = |v: [f32; 3]| Vec3::new(v[0] as f64, -v[2] as f64, v[1] as f64);
        for &(i, v, om) in &held.0 {
            let cl = &mut world.solver.clusters[i as usize];
            (cl.velocity, cl.angular_velocity) = (zup(v), zup(om));
        }
        for &(k, v, om) in &held.1 {
            let imp = &mut world.impactors[k as usize];
            (imp.velocity, imp.angular_velocity) = (zup(v), zup(om));
        }
        let (c_chunks, c_imps) = cpu_members(&mut world, tiered);
        if std::env::var("ROUTE_TRACE").is_ok() {
            eprintln!("  trace {} f{f}: batched {batched} batches {} GPU {} CPU {} time {:.4}", scene.name, batches, g_chunks.len(), c_chunks.len(), world.solver.time);
        }
        t.frames += 1;
        if !c_chunks.is_empty() || !g_chunks.is_empty() {
            t.routed += 1;
        }
        if std::env::var("ROUTE_DEBUG").is_ok() && g_chunks != c_chunks {
            let n = world.solver.clusters.len();
            let parts = w.learned.as_ref().unwrap().route_parts(gpu, n + world.impactors.len());
            for (i, cl) in world.solver.clusters.iter().enumerate() {
                if c_chunks.contains(&keys::chunk_key(cl.structure, cl.chunks[0])) || cl.chunks.len() < 6 {
                    let p = &parts[8 * i..8 * i + 8];
                    eprintln!("   cl {i} id {} chunks {} bonds {} anchored {} m {:.2} v {:?} w {:?}\n     gpu flags {} lo {:?} hi {:?} m {} v {:?} rt {} cap {} mod {} rad {} reach {}", cl.id, cl.chunks.len(), cl.bonds.len(), cl.anchored, cl.mass, cl.velocity, cl.angular_velocity,
                        p[0][3].to_bits(), &p[0][..3], &p[1][..3], p[1][3], &p[5][..3], p[5][3], p[6][3], p[3][3], p[4][3], p[2][3]);
                }
            }
            eprintln!("   totals {:?} export {}", w.learned.as_ref().unwrap().totals, export.len());
            eprintln!("   seeds/labels/actl {:?}", w.learned.as_ref().unwrap().route_seeds(gpu, n + world.impactors.len()));
            eprintln!("   ground {:?}", world.scene.ground.as_ref().map(|g| (g.height, g.material.clone())));
            for (k, imp) in world.impactors.iter().enumerate() {
                let p = &parts[8 * (n + k)..8 * (n + k) + 8];
                eprintln!("   imp {k} v {:?} pos {:?}\n     gpu flags {} lo {:?} hi {:?} v {:?} m {} rad {} mod {}", imp.velocity, imp.pose.position, p[0][3].to_bits(), &p[0][..3], &p[1][..3], &p[5][..3], p[1][3], p[4][3], p[3][3]);
            }
        }
        if g_chunks == c_chunks && (g_imps == c_imps || c_chunks.is_empty()) {
            t.agree += 1;
        } else {
            t.bad.push(format!(
                "{} tiered {tiered} debris {debris} frame {f}: GPU {} chunks {:?} imp, CPU {} chunks {:?} imp (GPU only {}, CPU only {})",
                scene.name,
                g_chunks.len(),
                g_imps,
                c_chunks.len(),
                c_imps,
                g_chunks.difference(&c_chunks).count(),
                c_chunks.difference(&g_chunks).count()
            ));
        }
    }
}

#[test]
fn gpu_routing_matches_the_cpu_routing() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "punch");
    let gpu = Gpu::new().expect("GPU");
    let mut scenes = vec![
        perturbed(realtime(later(wall_ram_late(), 2), 0.25), 7.3e-4, -4.1e-4),
        perturbed(realtime(later(wall_row_late(2), 2), 0.25), -5.2e-4, 8.9e-4),
    ];
    use stress_learned::procedural::{wall_recipes, with_splits_and_ensembles, EnsemblePlan, Ranges};
    for family in ["free_wall", "anchored_wall"] {
        let recipes = with_splits_and_ensembles(wall_recipes(family, 1, 24, 101, &Ranges::pilot()).expect("recipes"), &EnsemblePlan::default());
        let fast = |r: &&stress_learned::outcome::OutcomeRecipe| r.scene.impactors.iter().any(|i| Vec3::from_array(i.velocity).norm() > 15.0);
        for r in recipes.iter().filter(|r| r.parameters["split"] != "train").filter(fast).take(2) {
            // (its strike moved into the resident frames: the first are coupled ones)
            scenes.push(realtime(later(r.scene.clone(), 4), 0.3));
        }
    }
    let mut t = Tally::default();
    let skip = std::env::var("ROUTE_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for scene in scenes.iter().skip(skip).take(std::env::var("ROUTE_SCENES").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX)) {
        // dry: the walls as the engine leaves them; not dry: the punch-like stub fractures them
        // (fragments meeting fragments: the untiered routing's own seeds)
        let configs: Vec<(bool, bool, bool)> = match std::env::var("ROUTE_CONFIG").ok().as_deref() {
            Some("untiered-fractured") => vec![(false, false, false)],
            _ => vec![(true, false, true), (true, true, true), (false, false, true), (true, false, false), (false, false, false), (false, true, false)],
        };
        for (tiered, debris, dry) in configs {
            let before = (t.frames, t.routed, t.agree);
            run(&gpu, scene.clone(), tiered, debris, dry, &mut t);
            eprintln!("{} tiered {tiered} debris {debris} dry {dry}: {} frames, {} routed, {} agree", scene.name, t.frames - before.0, t.routed - before.1, t.agree - before.2);
        }
    }
    for b in &t.bad {
        eprintln!("  {b}");
    }
    eprintln!("routing parity: {} frames ({} routed), {} agree, {} differ", t.frames, t.routed, t.agree, t.bad.len());
    assert!(t.routed > 0, "nothing routed");
    assert!(t.bad.is_empty(), "{} frames route differently from the CPU", t.bad.len());
}
