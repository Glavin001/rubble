//! The fracture lab's `threshold` mode in the world API (`StrengthModel::Threshold`):
//! the masonry arch and a cuboid house, each shot as the lab shoots, break bonds and
//! split into new bodies, as the lab's own `ThresholdWorld` does on the same scene.
#![cfg(feature = "stress")]

use rubble::*;

/// The capacity the lab's `ThresholdWorld::new` gives the scene (its plan hints follow
/// the contact capacity).
fn capacity(asset: &DestructibleAssetDesc) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

/// A world with the asset's threshold solver and the stress scene's ground, `offset`
/// placing the asset.
fn world_for(asset: &DestructibleAssetDesc, offset: [f32; 3]) -> World {
    let mut solver = asset
        .solver_for(StrengthModel::Threshold)
        .expect("threshold solver");
    // THRESHOLD_SLEEP=1 (probe): the engine's default sleep thresholds instead of never.
    let sleep = std::env::var("THRESHOLD_SLEEP").is_ok();
    if sleep {
        let d = AvbdParams::default();
        solver.sleep_lin = d.sleep_lin;
        solver.sleep_ang = d.sleep_ang;
        solver.sleep_time = d.sleep_time;
    }
    let def = WorldDef {
        capacity: capacity(asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: sleep,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let stress = asset.stress.as_ref().expect("stress source");
    if let Some(h) = stress.ground_height() {
        let g = w.create_body(&BodyDef::default()).unwrap();
        w.create_shape(
            g,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: h + offset[1],
            },
        )
        .unwrap();
    }
    w
}

/// The lab's shot at its scene's centre from straight in front (`stress_shoot` through
/// the middle of the view): origin, velocity, radius and mass, in the asset frame.
fn lab_shot(
    lab: &engine_fracture::Scene,
    target: Option<[f64; 3]>,
) -> ([f64; 3], [f64; 3], f64, f64) {
    let c = target.unwrap_or(lab.center);
    let origin = [c[0], c[1], c[2] + 3.0 * lab.extent.max(1.0)];
    let dir = [0.0, 0.0, -1.0];
    let speed = engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed);
    let v = engine_fracture::plan::shot_velocity(
        c,
        lab.extent,
        lab.projectile_radius,
        origin,
        dir,
        Some(speed),
        lab.projectile_arc,
    );
    (origin, v, lab.projectile_radius, lab.projectile_mass)
}

struct Outcome {
    broken: u32,
    bodies: u32,
    splits: usize,
    /// Bonds broken by each frame's end.
    timeline: Vec<u32>,
}

/// Place `asset` as a threshold destructible, fire the lab's shot, step `frames`.
fn run_world(asset: &DestructibleAssetDesc, lab: &engine_fracture::Scene, frames: u32) -> Outcome {
    run_world_among(asset, lab, frames, false)
}

/// `run_world`, with `neighbours`: copies of the asset placed 80 m and 160 m along x in
/// the same world (one more placed and destroyed before the shot's target, whose freed
/// tables the target then takes), and the first neighbour destroyed at the end. The
/// neighbours never break, and the target breaks as it does alone.
fn run_world_among(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
    frames: u32,
    neighbours: bool,
) -> Outcome {
    let placement = asset.stress.as_ref().unwrap().placement();
    // The lab's stress world is the asset frame less its placement.
    let offset = placement.map(|x| -x as f32);
    let mut w = if neighbours {
        let mut def = asset.solver_for(StrengthModel::Threshold).unwrap();
        def.sleep_lin = -1.0;
        let mut c = capacity(asset);
        c.max_bodies *= 4;
        c.max_colliders = c.max_colliders.max(1 << 14) * 4;
        c.max_hull_vertices *= 4;
        let mut w = World::new(
            GpuContext::Default,
            &WorldDef {
                capacity: c,
                gravity: [0.0, def.gravity_y, 0.0],
                solver: def,
                enable_sleep: false,
                contacts: ContactStorage::Canonical,
                ..WorldDef::default()
            },
        )
        .unwrap();
        let stress = asset.stress.as_ref().unwrap();
        let h = stress.ground_height().unwrap();
        let g = w.create_body(&BodyDef::default()).unwrap();
        w.create_shape(
            g,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: h + offset[1],
            },
        )
        .unwrap();
        w
    } else {
        world_for(asset, offset)
    };
    let a = w.create_destructible_asset(asset).unwrap();
    let place = |w: &mut World, dx: f32| {
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new([offset[0] + dx, offset[1], offset[2]], [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap()
    };
    let mut others = Vec::new();
    if neighbours {
        others.push(place(&mut w, 80.0));
        let gone = place(&mut w, 240.0);
        w.destroy_destructible(gone).unwrap();
    }
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: std::env::var("THRESHOLD_START_ASLEEP").is_ok(),
            ..DestructibleDef::new(a)
        })
        .unwrap();
    if neighbours {
        others.push(place(&mut w, 160.0));
    }
    let stats = w.destructible_stats(id).unwrap();
    assert!(stats.chunks > 10 && stats.bonds > 10, "{stats:?}");
    let (origin, v, r, m) = lab_shot(lab, target(asset));
    let at = [0, 1, 2].map(|k| (origin[k] + offset[k] as f64) as f32);
    let ball = w
        .create_body(&BodyDef {
            linear_velocity: v.map(|x| x as f32),
            ..BodyDef::dynamic(at)
        })
        .unwrap();
    w.create_sphere_shape(
        ball,
        &ShapeDef {
            update_body_mass: false,
            ..ShapeDef::default()
        },
        [0.0; 3],
        r as f32,
    )
    .unwrap();
    let i = (0.4 * m * r * r) as f32;
    w.body_set_mass_data(
        ball,
        MassData {
            mass: m as f32,
            center: [0.0; 3],
            rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
        },
    )
    .unwrap();
    w.body_set_projectile(ball, Some(Projectile::default()))
        .unwrap();
    let mut broken = 0;
    let mut timeline = Vec::new();
    let mut splits = 0;
    let mut new_bodies = Vec::new();
    // Splits to check a frame later: (frame, parent body, its pose then, children).
    let mut fresh: Vec<(u32, BodyId, Transform, Vec<BodyId>)> = Vec::new();
    let dt = asset.solver_for(StrengthModel::Threshold).unwrap().dt;
    for n in 0..frames {
        w.step().unwrap();
        // A piece starts with its parent's rigid motion: right after the split its
        // origin (the instance frame carried along) is where its parent's was.
        for (at, parent, pose, children) in fresh.iter().filter(|s| s.0 + 1 == n) {
            for c in children.iter().filter(|c| *c != parent) {
                // Compared at the piece's centre of mass, within one step of its motion.
                let st = w.read_body_state(*c).unwrap();
                let centre = glam::Vec3::from(w.body_local_center(*c).unwrap());
                let at_now = glam::Quat::from_array(st.transform.rotation) * centre
                    + glam::Vec3::from(st.transform.position);
                let at_split = glam::Quat::from_array(pose.rotation) * centre
                    + glam::Vec3::from(pose.position);
                let d = (at_now - at_split).length();
                let a = glam::Quat::from_array(st.transform.rotation)
                    .angle_between(glam::Quat::from_array(pose.rotation));
                let v = glam::Vec3::from(st.linear_velocity).length();
                let spin = glam::Vec3::from(st.angular_velocity).length();
                assert!(d < 2.0 * v * dt + 0.02 && a < 2.0 * spin * dt + 0.01, "frame {at}: piece {c:?} starts {d} m and {a} rad from where its parent was (v {v} m/s, w {spin} rad/s)");
            }
        }
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                match e {
                    DestructionEvent::BondsBroken {
                        destructible,
                        bonds,
                    } => {
                        assert_eq!(*destructible, id, "only the shot's target breaks");
                        broken += bonds.len() as u32;
                    }
                    DestructionEvent::Split {
                        parent, children, ..
                    } => {
                        splits += 1;
                        let pose = w.read_body_state(parent.bodies[0]).unwrap().transform;
                        fresh.push((
                            n,
                            parent.bodies[0],
                            pose,
                            children.iter().map(|c| c.bodies[0]).collect(),
                        ));
                        // The children partition the parent.
                        let mut chunks: Vec<u32> =
                            children.iter().flat_map(|c| c.chunks.clone()).collect();
                        chunks.sort_unstable();
                        assert_eq!(chunks, parent.chunks);
                        let mass: f32 = children.iter().map(|c| c.mass).sum();
                        assert!(
                            (mass - parent.mass).abs() <= 1e-3 * parent.mass.max(1.0),
                            "{mass} vs {}",
                            parent.mass
                        );
                        for c in children {
                            assert_eq!(c.bodies.len(), 1);
                            assert!(
                                w.body_is_valid(c.bodies[0]),
                                "a split's piece is a live body"
                            );
                            new_bodies.push(c.bodies[0]);
                        }
                    }
                    _ => {}
                }
            }
            timeline.push(broken);
        }
    }
    for f in w.poll_blocking().unwrap() {
        for e in &f.destruction_events {
            if let DestructionEvent::BondsBroken { bonds, .. } = e {
                broken += bonds.len() as u32;
            }
        }
    }
    let stats = w.destructible_stats(id).unwrap();
    assert_eq!(
        stats.bonds - stats.bonds_intact,
        broken,
        "events report every broken bond"
    );
    // Every chunk's body is the one its actor names, and pieces move as bodies.
    for c in 0..stats.chunks {
        let body = w.destructible_chunk_body(id, c).unwrap();
        assert!(w.body_is_valid(body));
        assert_eq!(w.destructible_actor_of(id, c).unwrap().bodies, vec![body]);
    }
    for b in new_bodies.iter().filter(|b| w.body_is_valid(**b)) {
        assert!(
            w.body_transform(*b).unwrap().is_some(),
            "piece {b:?} has a reported pose"
        );
    }
    if neighbours {
        for o in &others {
            let s = w.destructible_stats(*o).unwrap();
            assert_eq!(s.bonds_intact, s.bonds, "a neighbour stands intact");
        }
        // The first neighbour goes; the target's pieces (some on spare slots it reserved)
        // stay valid and the world keeps stepping.
        w.destroy_destructible(others[0]).unwrap();
        for _ in 0..10 {
            w.step().unwrap();
            w.poll_blocking().unwrap();
        }
        for c in 0..stats.chunks {
            assert!(w.body_is_valid(w.destructible_chunk_body(id, c).unwrap()));
        }
    }
    w.destroy_destructible(id).unwrap();
    Outcome {
        broken,
        bodies: stats.actors,
        splits,
        timeline,
    }
}

/// The lab's `ThresholdWorld` on the same scene and shot (stress frame: the asset frame
/// less its placement, turned Z up).
fn run_lab(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
    frames: u32,
) -> (u32, u32, Vec<u32>) {
    let (info, device, queue) = engine_webgpu::default_device().unwrap();
    let gpu = stress_gpu::gpu::Gpu::from_device(info, device, queue);
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let (origin, v, r, m) = lab_shot(lab, target(asset));
    let p = stress.placement();
    let zup = |v: [f64; 3]| [v[0], -v[2], v[1]];
    let mut world = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, r, m).unwrap();
    world.launch(
        &gpu,
        &stress_ref::scene::ImpactorDesc {
            name: "shot0".into(),
            shape: stress_ref::scene::ImpactorShape::Sphere { radius: r },
            mass: m,
            position: zup([0, 1, 2].map(|k| origin[k] - p[k])),
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: zup(v),
            angular_velocity: [0.0; 3],
            material: "projectile".into(),
            crush: None,
        },
    );
    let mut timeline = Vec::new();
    for _ in 0..frames {
        world.step_frame(&gpu).unwrap();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        timeline.push(world.fracture.counters(&gpu).broken);
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let c = world.fracture.counters(&gpu);
    assert_eq!(c.error, 0);
    let bodies = {
        let mut owners = world.fracture.chunk_bodies(&gpu);
        owners.sort_unstable();
        owners.dedup();
        owners.len() as u32
    };
    (c.broken, bodies, timeline)
}

/// Where the shot aims: the scene's centre, or for an arch (open at its centre) its
/// highest chunk, the keystone.
fn target(asset: &DestructibleAssetDesc) -> Option<[f64; 3]> {
    if asset.chunks.iter().any(|c| c.node.is_some()) {
        let top = asset
            .chunks
            .iter()
            .max_by(|a, b| a.pose.position[1].total_cmp(&b.pose.position[1]))?;
        return Some(top.pose.position.map(|x| x as f64));
    }
    None
}

/// The first frame that broke bonds, and how many it broke.
fn first_break(timeline: &[u32]) -> Option<(usize, u32)> {
    timeline
        .iter()
        .position(|&b| b > 0)
        .map(|f| (f, timeline[f]))
}

fn arch() -> (DestructibleAssetDesc, engine_fracture::Scene) {
    let glb = include_bytes!("../../engine-webgpu/fracture/assets/masonry-arch.glb");
    let asset = DestructibleAssetDesc::from_glb(glb, 1).unwrap();
    let parsed = engine_fracture::parse_fracture_glb(glb).unwrap();
    let lab = engine_fracture::build_scene(&parsed, 1, None).unwrap();
    (asset, lab)
}

fn house() -> (DestructibleAssetDesc, engine_fracture::Scene) {
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None).unwrap();
    (asset, lab)
}

fn check(name: &str, (asset, lab): (DestructibleAssetDesc, engine_fracture::Scene)) {
    let frames = 120;
    let world = run_world(&asset, &lab, frames);
    let (lab_broken, lab_bodies, lab_timeline) = run_lab(&asset, &lab, frames);
    eprintln!("{name}: world broke {} bonds into {} bodies ({} splits); the lab {lab_broken} into {lab_bodies}", world.broken, world.bodies, world.splits);
    assert!(
        world.broken > 0 && world.splits > 0,
        "{name}: the shot breaks bonds and splits the structure"
    );
    // The same pipeline on the same scene: the strike breaks the same bonds in the same
    // frame. After the first split the cascade follows which body slots the pieces take
    // (the lab parks its 16 shots between the clusters and the spare slots), so totals
    // are compared loosely.
    assert_eq!(
        first_break(&world.timeline),
        first_break(&lab_timeline),
        "{name}: the first break"
    );
    let ratio = world.bodies as f64 / lab_bodies as f64;
    assert!(
        (0.75..=1.33).contains(&ratio),
        "{name}: {} bodies against the lab's {lab_bodies}",
        world.bodies
    );
}

/// Destructibles share one fracture per world: the cuboid house breaks among neighbours as
/// it does alone (in tables placed after others, into a destroyed one's freed range).
#[test]
fn a_house_among_neighbours_breaks_as_alone() {
    let (asset, lab) = house();
    let frames = 120;
    let alone = run_world(&asset, &lab, frames);
    let among = run_world_among(&asset, &lab, frames, true);
    eprintln!(
        "alone {} bonds into {} bodies; among neighbours {} into {}",
        alone.broken, alone.bodies, among.broken, among.bodies
    );
    assert_eq!(
        first_break(&among.timeline),
        first_break(&alone.timeline),
        "the first break"
    );
    let ratio = among.bodies as f64 / alone.bodies as f64;
    assert!(
        (0.75..=1.33).contains(&ratio),
        "{} bodies against {} alone",
        among.bodies,
        alone.bodies
    );
}

#[test]
fn masonry_arch_breaks_and_splits_as_in_the_lab() {
    check("arch", arch());
}

#[test]
fn cuboid_house_breaks_and_splits_as_in_the_lab() {
    check("house", house());
}

#[test]
fn limits_fail_loudly() {
    let (asset, _) = house();
    let place = |w: &mut World| {
        let a = w.create_destructible_asset(&asset).unwrap();
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            ..DestructibleDef::new(a)
        })
    };
    // The ingest reads canonical contact rows.
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: capacity(&asset),
            ..WorldDef::default()
        },
    )
    .unwrap();
    let e = place(&mut w).unwrap_err().to_string();
    assert!(e.contains("ContactStorage::Canonical"), "{e}");
    // Every piece a split can make needs a body slot, reserved up front.
    let small = WorldCapacity {
        max_bodies: 64,
        ..capacity(&asset)
    };
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: small,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )
    .unwrap();
    let e = place(&mut w).unwrap_err().to_string();
    assert!(e.contains("max_bodies"), "{e}");
    // Strike prediction's shot table holds 64 projectiles.
    let mut w = world_for(&asset, [0.0; 3]);
    let id = place(&mut w).unwrap();
    // Pieces belong to the destructible.
    let piece = w.destructible_chunk_body(id, 0).unwrap();
    assert!(w
        .destroy_body(piece)
        .unwrap_err()
        .to_string()
        .contains("destroy the destructible"));
    for k in 0..65 {
        let b = w
            .create_body(&BodyDef::dynamic([k as f32, 50.0, 0.0]))
            .unwrap();
        w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.1)
            .unwrap();
        let r = w.body_set_projectile(b, Some(Projectile::default()));
        assert_eq!(r.is_ok(), k < 64, "projectile {k}: {r:?}");
    }
}
