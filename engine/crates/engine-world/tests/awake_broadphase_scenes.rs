//! The awake-indexed broadphase finds exactly the full pass's candidate pairs, as a set,
//! on the scenes the equivalence harness runs: before every compared step, on the same GPU
//! state, both ways (`WgpuSim::debug_broadphase_pairs`), with bit-identical collider
//! bounds. The demo's 100k Grid (uniform grid, sleep on, a CCD ball at step 300) and Brick
//! Ring 28k (its rolling ball, sleep on), and the city at war (threshold houses: a GPU
//! fracture that re-parents colliders and re-frames bodies between steps).

use rubble::*;

fn sorted(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v
}

fn bounds(sim: &WgpuSim) -> Vec<u32> {
    let n = sim.collider_count() as usize;
    let at = u64::from(sim.capacity().max_bodies) * 8;
    let mut w = sim.read_words(9, at, n * 8);
    // The spare word holds the grids' chain links.
    for box_ in w.chunks_mut(8) {
        box_[7] = 0;
    }
    w
}

/// Steps `steps`, comparing every `every`-th; returns the comparisons made with the awake
/// pass planned.
fn compare_steps(
    w: &mut World,
    steps: u32,
    every: u32,
    label: &str,
    mut before: impl FnMut(&mut World, u32),
) -> usize {
    let mut compared = 0;
    for s in 0..steps {
        before(w, s);
        if s % every == 0 {
            w.poll_blocking().unwrap();
            let sim = w.sim_mut();
            if sim.awake_broadphase_planned() {
                // Both runs start from the same scratch (binding 21: the sleep archive lists
                // the sleepers touched this step there, and a second run would spread from
                // the first run's), and so does the step taken after them.
                let scratch = sim.debug_copy_binding(21);
                let awake = sorted(sim.debug_broadphase_pairs());
                let ab = bounds(sim);
                sim.debug_restore_binding(21, &scratch);
                sim.set_awake_broadphase(false);
                let full = sorted(sim.debug_broadphase_pairs());
                let fb = bounds(sim);
                sim.debug_restore_binding(21, &scratch);
                sim.set_awake_broadphase(true);
                let only_a: Vec<_> = awake
                    .iter()
                    .filter(|p| full.binary_search(p).is_err())
                    .collect();
                let only_f: Vec<_> = full
                    .iter()
                    .filter(|p| awake.binary_search(p).is_err())
                    .collect();
                if !(only_a.is_empty() && only_f.is_empty()) {
                    let cols = sim.download_colliders();
                    let bodies = sim.download_bodies();
                    let n_awake = bodies
                        .iter()
                        .filter(|b| b.inv_mass > 0.0 && b.flags & 4 == 0)
                        .count();
                    let differ: Vec<usize> = (0..ab.len() / 8)
                        .filter(|&c| ab[c * 8..c * 8 + 8] != fb[c * 8..c * 8 + 8])
                        .collect();
                    eprintln!(
                        "{label} step {s}: {} bodies, {n_awake} awake dynamic; {} bounds differ, first {:?}",
                        bodies.len(),
                        differ.len(),
                        &differ[..differ.len().min(8)]
                    );
                    for &&(i, j) in only_a.iter().chain(only_f.iter()) {
                        for c in [i, j] {
                            let k = &cols[c as usize];
                            let b = &bodies[k.body as usize];
                            let bx = &ab[c as usize * 8..c as usize * 8 + 8];
                            let f = |w: &[u32]| {
                                w.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>()
                            };
                            let (rest_base, grid) = sim.debug_rest_layout();
                            let nc = u64::from(sim.collider_count());
                            let links = rest_base + 32 + 2 * u64::from(grid);
                            let at = sim.read_words(14, links + 3 * nc + u64::from(c), 1)[0];
                            eprintln!("  c{c}: resting-table bucket {at:#x}");
                            eprintln!(
                                "  c{c}: body {} shape {} params {:?} | flags {:#x} sleep timer {} inv_mass {} pos {:?} vel {:?} | aabb {:?}",
                                k.body, k.shape, k.params, b.flags, b.sleep_timer, b.inv_mass, b.position, b.linear_vel, f(bx)
                            );
                        }
                    }
                }
                assert!(
                    only_a.is_empty() && only_f.is_empty(),
                    "{label} step {s}: awake-only {only_a:?}, full-only {only_f:?}"
                );
                let stale = (0..ab.len() / 8)
                    .filter(|&c| ab[c * 8..c * 8 + 8] != fb[c * 8..c * 8 + 8])
                    .count();
                if stale > 0 {
                    let cols = sim.download_colliders();
                    let bodies = sim.download_bodies();
                    let f = |w: &[u32]| w.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>();
                    let differ: Vec<usize> = (0..ab.len() / 8)
                        .filter(|&c| ab[c * 8..c * 8 + 8] != fb[c * 8..c * 8 + 8])
                        .collect();
                    let mut by_body = std::collections::BTreeMap::new();
                    for &c in &differ {
                        *by_body.entry(cols[c].body).or_insert(0usize) += 1;
                    }
                    eprintln!("{label} step {s}: {} bodies hold the {stale} differing colliders: {:?}", by_body.len(), by_body.iter().take(12).collect::<Vec<_>>());
                    for &c in differ.iter().take(4) {
                        let b = &bodies[cols[c].body as usize];
                        eprintln!(
                            "  c{c}: body {} flags {:#x} inv_mass {} sleep timer {} pos {:?} vel {:?}\n    awake pass {:?}\n    full pass  {:?}",
                            cols[c].body, b.flags, b.inv_mass, b.sleep_timer, b.position, b.linear_vel,
                            f(&ab[c * 8..c * 8 + 8]), f(&fb[c * 8..c * 8 + 8])
                        );
                    }
                }
                assert_eq!(
                    stale, 0,
                    "{label} step {s}: {stale} colliders' bounds differ"
                );
                compared += 1;
            }
        }
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    compared
}

fn scene(name: &str) -> (World, Vec<BodyId>) {
    let (host, _) = engine_scenes::build_host(name).unwrap();
    let snap = host.snapshot();
    let n = snap.bodies.len() as u32 + 4;
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: n.next_power_of_two(),
            max_colliders: n.next_power_of_two(),
            max_contacts: (16 * n).next_power_of_two(),
            max_candidates: (16 * n).next_power_of_two(),
            max_colors: 32,
            ..WorldCapacity::small()
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let imported = w.import_snapshot(&snap).unwrap();
    let mut solver = w.solver();
    solver.sleep_time = 0.5;
    w.set_solver(solver);
    (w, imported.bodies)
}

#[test]
fn grid_awake_broadphase_pairs_equal_the_full_pass() {
    let (mut w, _) = scene("100k Grid");
    let n = compare_steps(&mut w, 600, 3, "grid", |w, s| {
        if s == 300 {
            // A CCD ball into the settled grid (grid_strike's).
            let b = w
                .create_body(&BodyDef {
                    position: [44.0, 28.6, 55.4],
                    ..BodyDef::default()
                })
                .unwrap();
            w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.35)
                .unwrap();
            w.body_set_bullet(b, true).unwrap();
            w.body_set_linear_velocity(b, [-17.6, -8.0, -22.0]).unwrap();
        }
    });
    eprintln!("grid: {n} steps compared");
    assert!(n > 100, "compared {n}");
}

#[test]
fn ring_awake_broadphase_pairs_equal_the_full_pass() {
    // The demo's planning (the uniform awake pass only while few bodies are awake), then the
    // awake pass at every awake fraction (`set_uniform_rest_always`), every step compared.
    for always in [false, true] {
        let (mut w, _) = scene("Brick Ring (28k)");
        w.sim_mut().set_uniform_rest_always(always);
        let n = compare_steps(&mut w, 400, 1, "ring", |_, _| {});
        eprintln!("ring (always {always}): {n} steps compared");
        if always {
            assert!(n > 390, "compared {n}");
        }
    }
}

#[test]
fn grid_awake_broadphase_pairs_equal_the_full_pass_at_every_awake_fraction() {
    let (mut w, _) = scene("100k Grid");
    w.sim_mut().set_uniform_rest_always(true);
    let n = compare_steps(&mut w, 450, 3, "grid always", |w, s| {
        if s == 300 {
            let b = w
                .create_body(&BodyDef {
                    position: [44.0, 28.6, 55.4],
                    ..BodyDef::default()
                })
                .unwrap();
            w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.35)
                .unwrap();
            w.body_set_bullet(b, true).unwrap();
            w.body_set_linear_velocity(b, [-17.6, -8.0, -22.0]).unwrap();
        }
    });
    eprintln!("grid (always): {n} steps compared");
    assert!(n >= 149, "compared {n}");
}

/// The city (threshold houses: a GPU fracture that re-parents colliders and re-frames
/// bodies between steps, with no host write): every step of a war compared with the full
/// pass, pairs and bounds. A struck house's static remainder is re-framed by its split; the
/// step after it bounds every collider and rebuilds the resting table
/// (`StepModes::gpu_reframes`).
#[test]
fn city_awake_broadphase_pairs_equal_the_full_pass() {
    for sleep in [true, false] {
        let house = rubble::city::city_house().unwrap();
        let def = rubble::city::CityDef {
            houses: 16,
            sleep,
            ..Default::default()
        };
        let mut w = World::new(GpuContext::Default, &def.world_def(&house).unwrap()).unwrap();
        let mut c = rubble::city::City::begin(&mut w, &house, def).unwrap();
        c.place(&mut w, 16).unwrap();
        for _ in 0..rubble::city::CITY_SETTLE {
            w.step().unwrap();
            w.poll_blocking().unwrap();
        }
        c.start(&mut w, rubble::city::CityScenario::from_name("war").unwrap())
            .unwrap();
        let label = format!("city (sleep {sleep})");
        assert!(w.sim().awake_broadphase_planned(), "{label}: the awake pass is planned");
        let n = compare_steps(&mut w, 150, 1, &label, |_, _| {});
        eprintln!("{label}: {n} steps compared");
        assert!(n > 140, "{label}: compared {n}");
    }
}
