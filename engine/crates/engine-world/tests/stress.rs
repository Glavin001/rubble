//! The fracture lab's `stress` mode in the world API (`StrengthModel::Stress`, the
//! blast_parity profile): the masonry arch and a cuboid house, each shot as the lab
//! shoots, break bonds and split into new bodies, as the lab's own stress world
//! (`GpuWorld` over `RubbleEngine`) does on the same scene.
#![cfg(feature = "stress")]

use rubble::*;

/// The stress scene both runs use: the asset placed at `offset`, which for
/// `-placement` is the lab's own stress frame (the asset frame less its placement).
fn scene_for(
    asset: &DestructibleAssetDesc,
    offset: [f32; 3],
    strength: f32,
) -> stress_ref::scene::Scene {
    let solver = asset.solver_for(StrengthModel::Stress).unwrap();
    asset
        .stress
        .as_ref()
        .unwrap()
        .stress_scene(
            strength,
            solver.dt,
            solver.gravity_y,
            Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        )
        .unwrap()
}

/// A world with the asset's stress solver and the stress scene's ground.
fn world_for(asset: &DestructibleAssetDesc, offset: [f32; 3]) -> World {
    let solver = asset
        .solver_for(StrengthModel::Stress)
        .expect("stress solver");
    let def = WorldDef {
        capacity: stress_rubble::capacity_for(&scene_for(asset, offset, 1.0)),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        // The solver's own sleep settings (never asleep), as the lab's engine has them.
        enable_sleep: true,
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

struct Outcome {
    broken: u32,
    bodies: u32,
    splits: usize,
    /// Bonds broken by each frame's end.
    timeline: Vec<u32>,
}

/// Place `asset` as a stress destructible, launch the lab's shot, step `frames`.
fn run_world(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
    frames: u32,
    strength: f32,
) -> Outcome {
    let placement = asset.stress.as_ref().unwrap().placement();
    let offset = placement.map(|x| -x as f32);
    let mut w = world_for(asset, offset);
    let a = w.create_destructible_asset(asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Stress,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            strength_scale: strength,
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    let stats = w.destructible_stats(id).unwrap();
    assert!(stats.chunks > 10 && stats.bonds > 10, "{stats:?}");
    for c in 0..stats.chunks {
        assert!(w.body_is_valid(w.destructible_chunk_body(id, c).unwrap()));
    }
    let (origin, v, r, m) = lab_shot(lab, target(asset));
    let shot = w
        .destructible_launch_impactor(
            id,
            &ImpactorDef {
                shape: ImpactorShape::Sphere { radius: r as f32 },
                mass: m as f32,
                position: [0, 1, 2].map(|k| (origin[k] + offset[k] as f64) as f32),
                linear_velocity: v.map(|x| x as f32),
                ..ImpactorDef::default()
            },
        )
        .unwrap();
    let mut broken = 0;
    let mut timeline = Vec::new();
    let mut splits = 0;
    let mut new_bodies = Vec::new();
    for n in 0..frames {
        w.step().unwrap();
        if n == 0 {
            let ball = w
                .destructible_impactor_body(id, shot)
                .unwrap()
                .expect("the impactor's body, from its first step");
            assert!(w.body_is_valid(ball));
        }
        if std::env::var("STRESS_TEST_TRACE").is_ok() && n % 10 == 0 {
            let b = w.destructible_impactor_body(id, shot).unwrap().unwrap();
            let st = w.read_body_state(b).unwrap();
            eprintln!(
                "frame {n}: shot at {:?} v {:?} (target {:?}, offset {offset:?})",
                st.transform.position,
                st.linear_velocity,
                target(asset)
            );
        }
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                match e {
                    DestructionEvent::BondsBroken {
                        destructible,
                        bonds,
                    } => {
                        assert_eq!(*destructible, id);
                        broken += bonds.len() as u32;
                    }
                    DestructionEvent::Split {
                        parent, children, ..
                    } => {
                        splits += 1;
                        // The parent's body ended; the children partition its chunks.
                        assert!(!w.body_is_valid(parent.bodies[0]));
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
                            new_bodies.push(c.bodies[0]);
                        }
                    }
                    _ => {}
                }
            }
        }
        timeline.push(broken);
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
        assert!(w.body_is_valid(body), "chunk {c}'s body {body:?}");
        assert_eq!(w.destructible_actor_of(id, c).unwrap().bodies, vec![body]);
    }
    for b in new_bodies.iter().filter(|b| w.body_is_valid(**b)) {
        assert!(
            w.body_transform(*b).unwrap().is_some(),
            "piece {b:?} has a reported pose"
        );
    }
    w.destroy_destructible(id).unwrap();
    // The world goes on without it.
    w.step_sync().unwrap();
    Outcome {
        broken,
        bodies: stats.actors,
        splits,
        timeline,
    }
}

/// The lab's stress mode on the same scene and shot (stress-web `load` and its shot),
/// with or without its slot compaction (a hosted engine reuses slots instead).
fn run_lab(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
    frames: u32,
    strength: f32,
    compaction: bool,
) -> (u32, u32, Vec<u32>) {
    use stress_ref::capability::Profile;
    let (info, device, queue) = engine_webgpu::default_device().unwrap();
    let gpu = stress_gpu::gpu::Gpu::from_device(info.clone(), device.clone(), queue.clone());
    let stress = asset.stress.as_ref().unwrap();
    let placement = stress.placement();
    let scene = scene_for(asset, placement.map(|x| -x as f32), strength);
    let mut e = stress_rubble::RubbleEngine::with_device_capacity(
        stress_ref::math::Vec3::from_array(scene.gravity),
        info,
        device,
        queue,
        stress_rubble::capacity_for(&scene),
    )
    .unwrap();
    let (iterations, substeps) = Profile::BlastParity.engine_policy();
    e.set_policy(iterations, substeps);
    e.compaction = compaction;
    let mut world = stress_gpu::world::GpuWorld::with_resident_engine(&gpu, &scene, e).unwrap();
    world.redo_fractured_frames = false;
    world.set_impact_islands(true);
    world.apply_profile(Profile::BlastParity);
    let (origin, v, r, m) = lab_shot(lab, target(asset));
    let zup = |v: [f64; 3]| [v[0], -v[2], v[1]];
    // The world's unrotated shot in the stress frame (Z up).
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let q = (glam::DQuat::from_xyzw(-h, 0.0, 0.0, h).inverse() * glam::DQuat::IDENTITY).normalize();
    let desc = stress_ref::scene::ImpactorDesc {
        name: "shot0".into(),
        shape: stress_ref::scene::ImpactorShape::Sphere {
            radius: r as f32 as f64,
        },
        mass: m as f32 as f64,
        // The shot as the world's API takes it (f32, world frame = the asset frame
        // less its placement).
        position: zup([0, 1, 2].map(|k| (origin[k] - placement[k] as f32 as f64) as f32 as f64)),
        orientation: [q.w, q.x, q.y, q.z],
        velocity: zup(v.map(|x| x as f32 as f64)),
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    };
    let imp = stress_ref::world::Impactor::from_desc(&desc, &world.scene);
    world.add_impactor(&gpu, imp).unwrap();
    // Bonds broken so far: each counted once, the first frame it is not connected
    // (as the world's `BondsBroken` events report them).
    let connected = |m: &stress_ref::solver::ReferenceSolver| -> Vec<bool> {
        m.bonds.iter().flatten().map(|b| b.connected()).collect()
    };
    let mut intact = world.view_with(&gpu, |m, _| connected(m));
    let mut broken = 0;
    let mut timeline = Vec::new();
    for _ in 0..frames {
        world.step_frame(&gpu).unwrap();
        let now = world.view_with(&gpu, |m, _| connected(m));
        for (was, now) in intact.iter_mut().zip(now) {
            if *was && !now {
                *was = false;
                broken += 1;
            }
        }
        timeline.push(broken);
    }
    let bodies = world.view_with(&gpu, |m, _| m.clusters.len() as u32);
    (*timeline.last().unwrap(), bodies, timeline)
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

fn check(
    name: &str,
    (asset, lab): (DestructibleAssetDesc, engine_fracture::Scene),
    frames: u32,
    strength: f32,
) {
    let world = run_world(&asset, &lab, frames, strength);
    let (lab_broken, lab_bodies, lab_timeline) = run_lab(&asset, &lab, frames, strength, false);
    let (_, compacted_bodies, compacted) = run_lab(&asset, &lab, frames, strength, true);
    eprintln!(
        "{name}: world broke {} bonds into {} bodies ({} splits), first {:?}; the lab {lab_broken} into {lab_bodies} ({} and {compacted_bodies} with its slot compaction)",
        world.broken,
        world.bodies,
        world.splits,
        first_break(&world.timeline),
        compacted.last().unwrap(),
    );
    assert!(
        world.broken > 0 && world.splits > 0,
        "{name}: the shot breaks bonds and splits the structure"
    );
    // The same stress world, engine and scene on the same body slots: frame for frame
    // the same bonds break, into the same bodies.
    assert_eq!(
        world.timeline, lab_timeline,
        "{name}: bonds broken by frame"
    );
    assert_eq!(world.bodies, lab_bodies, "{name}: bodies");
    // The lab's own run reclaims retired slots, which renumbers bodies and so the
    // solve's order: the same until the first compaction, then a cascade of its own.
    assert_eq!(
        first_break(&world.timeline),
        first_break(&compacted),
        "{name}: the first break"
    );
}

#[test]
fn masonry_arch_breaks_and_splits_as_in_the_lab() {
    // The arch holds the lab's shot at its masonry's full strength (in the lab too):
    // a tenth of it, as the lab's strength slider sets it.
    check("arch", arch(), 120, 0.1);
}

#[test]
fn cuboid_house_breaks_and_splits_as_in_the_lab() {
    check("house", house(), 120, 1.0);
}

#[test]
fn limits_fail_loudly() {
    let (asset, _) = house();
    let place = |w: &mut World, model: StrengthModel| {
        let a = w.create_destructible_asset(&asset).unwrap();
        w.create_destructible(&DestructibleDef {
            model,
            ..DestructibleDef::new(a)
        })
    };
    // The ingest reads canonical contact rows.
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let e = place(&mut w, StrengthModel::Stress)
        .unwrap_err()
        .to_string();
    assert!(e.contains("ContactStorage::Canonical"), "{e}");
    // The lab's headroom for splits.
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: WorldCapacity {
                max_bodies: 64,
                ..WorldCapacity::default()
            },
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )
    .unwrap();
    let e = place(&mut w, StrengthModel::Stress)
        .unwrap_err()
        .to_string();
    assert!(e.contains("max_bodies"), "{e}");
    // One stress destructible per world, and none beside a threshold one.
    let mut w = world_for(&asset, [0.0; 3]);
    let id = place(&mut w, StrengthModel::Stress).unwrap();
    let e = place(&mut w, StrengthModel::Stress)
        .unwrap_err()
        .to_string();
    assert!(e.contains("one StrengthModel::Stress"), "{e}");
    let e = place(&mut w, StrengthModel::Threshold)
        .unwrap_err()
        .to_string();
    assert!(e.contains("cannot share a world"), "{e}");
    // Pieces belong to the destructible.
    let piece = w.destructible_chunk_body(id, 0).unwrap();
    assert!(w
        .destroy_body(piece)
        .unwrap_err()
        .to_string()
        .contains("destroy the destructible"));
    // Impactors need a mass and a size.
    let bad = ImpactorDef {
        mass: 0.0,
        ..ImpactorDef::default()
    };
    assert!(w.destructible_launch_impactor(id, &bad).is_err());
    // A world body beside the structure steps with it.
    let b = w.create_body(&BodyDef::dynamic([30.0, 5.0, 0.0])).unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.2)
        .unwrap();
    for _ in 0..10 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    let y = w.read_body_state(b).unwrap().transform.position[1];
    assert!(y < 5.0, "the world's own body falls ({y})");
    w.destroy_destructible(id).unwrap();
    // The other model fits once it is gone.
    place(&mut w, StrengthModel::Threshold)
        .map(|_| ())
        .unwrap_or_else(|e| {
            // Threshold's own capacity needs (its reserved slots) may not fit this world.
            assert!(!e.to_string().contains("cannot share a world"), "{e}")
        });
}
