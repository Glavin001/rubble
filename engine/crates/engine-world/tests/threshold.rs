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
    world_with(asset, offset, Backpressure::Block, 3)
}

fn world_with(asset: &DestructibleAssetDesc, offset: [f32; 3], backpressure: Backpressure, readback_frames: u32) -> World {
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .expect("threshold solver");
    let def = WorldDef {
        capacity: capacity(asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        backpressure,
        readback_frames,
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
    /// Frames whose readback was dropped.
    dropped: usize,
}

/// Place `asset` as a threshold destructible, fire the lab's shot, step `frames`.
fn run_world(asset: &DestructibleAssetDesc, lab: &engine_fracture::Scene, frames: u32) -> Outcome {
    run_world_with(asset, lab, frames, Backpressure::Block, 1)
}

/// `run_world` with the world's backpressure and the host polling every `poll_every` steps
/// (1: waiting for each frame; else without waiting, one readback slot, so frames drop).
fn run_world_with(asset: &DestructibleAssetDesc, lab: &engine_fracture::Scene, frames: u32, backpressure: Backpressure, poll_every: u32) -> Outcome {
    let readback_frames = if poll_every == 1 { 3 } else { 1 };
    let (mut w, id, _) = shoot_with(asset, lab, backpressure, readback_frames);
    run_shot_with(&mut w, id, asset, frames, poll_every)
}

/// `asset` placed as a threshold destructible (at the lab's stress world) with the lab's
/// shot fired at it: the world, the destructible and its instance transform.
fn shoot(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
) -> (World, DestructibleId, Transform) {
    shoot_with(asset, lab, Backpressure::Block, 3)
}

/// `shoot` with the world's backpressure and readback slots.
fn shoot_with(asset: &DestructibleAssetDesc, lab: &engine_fracture::Scene, backpressure: Backpressure, readback_frames: u32) -> (World, DestructibleId, Transform) {
    let placement = asset.stress.as_ref().unwrap().placement();
    // The lab's stress world is the asset frame less its placement.
    let offset = placement.map(|x| -x as f32);
    let mut w = world_with(asset, offset, backpressure, readback_frames);
    let a = w.create_destructible_asset(asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
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
    (w, id, Transform::new(offset, [0.0, 0.0, 0.0, 1.0]))
}

/// The shot's steps, the host polling every `poll_every` steps (1: waiting for each frame).
fn run_shot_with(w: &mut World, id: DestructibleId, asset: &DestructibleAssetDesc, frames: u32, poll_every: u32) -> Outcome {
    let mut broken = 0;
    let mut timeline = Vec::new();
    let mut splits = 0;
    let mut new_bodies = Vec::new();
    // Splits to check a frame later: (frame, parent body, its pose then, children).
    let mut fresh: Vec<(u32, BodyId, Transform, Vec<BodyId>)> = Vec::new();
    let mut dropped = 0;
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
        let delivered = if poll_every == 1 {
            w.poll_blocking().unwrap()
        } else if n % poll_every == 0 {
            w.poll().unwrap()
        } else {
            Vec::new()
        };
        for f in delivered {
            dropped += f.dropped as usize;
            for e in &f.destruction_events {
                match e {
                    DestructionEvent::BondsBroken {
                        destructible,
                        bonds,
                        ..
                    } => {
                        assert_eq!(*destructible, id);
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
    // A host that dropped frames catches up at the next frame delivered: a few more steps,
    // their frames waited for.
    let mut tail = Vec::new();
    for _ in 0..if poll_every == 1 { 0 } else { 3 } {
        w.step().unwrap();
        tail.extend(w.poll_blocking().unwrap());
    }
    tail.extend(w.poll_blocking().unwrap());
    for f in tail {
        dropped += f.dropped as usize;
        for e in &f.destruction_events {
            if let DestructionEvent::Split { .. } = e {
                splits += 1;
            }
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
    w.destroy_destructible(id).unwrap();
    Outcome {
        broken,
        bodies: stats.actors,
        splits,
        timeline,
        dropped,
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

#[test]
fn masonry_arch_breaks_and_splits_as_in_the_lab() {
    check("arch", arch());
}

#[test]
fn cuboid_house_breaks_and_splits_as_in_the_lab() {
    check("house", house());
}

/// The arch's keystone shot splits its anchored body in consecutive frames (the piece
/// one split leaves splits again the next step). Every static piece stays exactly at the
/// instance frame, whether read from the GPU straight after a step (its split not yet
/// polled) or from the frames; a split's pieces are reported in the frame that reports
/// the split, at their parent's origin; and every chunk's world pose is continuous.
#[test]
fn consecutive_splits_keep_pieces_in_place() {
    let (asset, lab) = arch();
    let (mut w, id, instance) = shoot(&asset, &lab);
    let dt = asset.solver_for(StrengthModel::Threshold).unwrap().dt;
    let chunks = w.destructible_stats(id).unwrap().chunks;
    let same = |a: &Transform, b: &Transform, tol: f32| {
        let d = (glam::Vec3::from(a.position) - glam::Vec3::from(b.position)).length();
        // Half the angle between them, near enough (acos loses it in f32).
        let (qa, qb) = (glam::Vec4::from(a.rotation), glam::Vec4::from(b.rotation));
        let r = (qa - qb).length().min((qa + qb).length());
        d <= tol && r <= tol
    };
    // Per chunk: its body's origin when last polled, and the body's centre of mass then
    // (a point fixed to the chunk: chunk-local is the identity).
    let mut last: Vec<Option<(Transform, glam::Vec3)>> = vec![None; chunks as usize];
    // Frames in which an anchored parent split.
    let mut anchored_splits = Vec::new();
    for n in 0..120u32 {
        w.step().unwrap();
        // Straight from the GPU, before this step is polled (it may have split them).
        for c in 0..chunks {
            let b = w.destructible_chunk_body(id, c).unwrap();
            if w.body_type(b).unwrap() == BodyType::Static {
                let pose = w.read_body_state(b).unwrap().transform;
                assert!(
                    same(&pose, &instance, 1e-4),
                    "frame {n}: static {b:?} (chunk {c}) read at {pose:?}, not the instance frame"
                );
            }
        }
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                let DestructionEvent::Split {
                    parent, children, ..
                } = e
                else {
                    continue;
                };
                if parent.anchored {
                    anchored_splits.push(n);
                }
                let at = w
                    .body_transform(parent.bodies[0])
                    .unwrap()
                    .expect("the parent has a pose");
                for c in children {
                    let b = c.bodies[0];
                    let pose = w
                        .body_transform(b)
                        .unwrap()
                        .expect("a split's piece is reported with the split");
                    assert!(
                        same(&pose, &at, 1e-4),
                        "frame {n}: piece {b:?} reported at {pose:?}, its parent at {at:?}"
                    );
                    assert!(
                        b == parent.bodies[0]
                            || f.body_events
                                .moves
                                .iter()
                                .any(|m| m.body == b && m.transform == pose),
                        "frame {n}: piece {b:?} has no move with the split"
                    );
                }
            }
        }
        for c in 0..chunks {
            let b = w.destructible_chunk_body(id, c).unwrap();
            let pose = w
                .body_transform(b)
                .unwrap()
                .expect("every piece has a pose");
            let st = w.read_body_state(b).unwrap();
            assert!(
                same(&st.transform, &pose, 1e-4),
                "frame {n}: {b:?} polled at {pose:?}, read at {:?}",
                st.transform
            );
            if w.body_type(b).unwrap() == BodyType::Static {
                assert!(
                    same(&pose, &instance, 1e-4),
                    "frame {n}: static {b:?} (chunk {c}) at {pose:?}"
                );
            }
            if let Some((before, centre)) = last[c as usize] {
                let at = |t: &Transform| {
                    glam::Quat::from_array(t.rotation) * centre + glam::Vec3::from(t.position)
                };
                let v = glam::Vec3::from(st.linear_velocity).length()
                    + glam::Vec3::from(st.angular_velocity).length()
                        * (at(&pose) - glam::Vec3::from(pose.position)).length();
                let d = (at(&pose) - at(&before)).length();
                assert!(
                    d <= 2.0 * v * dt + 0.02,
                    "frame {n}: chunk {c} jumped {d} m (v {v} m/s)"
                );
            }
            last[c as usize] = Some((pose, glam::Vec3::from(w.body_local_center(b).unwrap())));
        }
    }
    assert!(
        anchored_splits.windows(2).any(|p| p[1] == p[0] + 1),
        "the shot splits an anchored body in consecutive frames: {anchored_splits:?}"
    );
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

#[test]
fn breaks_and_splits_survive_dropped_readbacks() {
    // A host that never waits (`Backpressure::Drop`, one readback slot, polling every few
    // steps): frames drop, yet the GPU's change log keeps every break and split for the next
    // frame delivered, so the events still report every broken bond and every chunk's body is
    // a live piece (`run_world_with`'s checks).
    let (asset, lab) = house();
    let o = run_world_with(&asset, &lab, 120, Backpressure::Drop, 5);
    let every = run_world(&asset, &lab, 120);
    println!("dropped {} frames: {} bonds broken, {} splits, {} bodies; read every frame: {} bonds, {} bodies", o.dropped, o.broken, o.splits, o.bodies, every.broken, every.bodies);
    assert!(o.dropped > 0, "frames were dropped");
    assert!(o.broken > 0 && o.splits > 0, "the shot broke and split the house");
    // The same strike: the first break is the same, totals near (the cascade after the
    // first split is not bit-reproducible between native worlds).
    let ratio = o.broken as f64 / every.broken as f64;
    assert!((0.7..=1.3).contains(&ratio), "{} vs {} bonds", o.broken, every.broken);
}
