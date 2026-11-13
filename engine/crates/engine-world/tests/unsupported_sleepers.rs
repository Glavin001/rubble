//! Nothing is left in the air: a sleeping body whose support is taken away wakes and
//! falls, though no awake body touches it.
//!
//! A sleeping body is as good as fixed in space, so whatever rests on it stays where it is
//! for as long as it sleeps. Removing a support is not a contact: a wake rule that only
//! fires on a touch from an awake body misses it, and the piece above hangs. The cases here
//! are that failure's simplest forms, each stated as what must be true afterwards.
//!
//! `World::destroy_body` wakes the sleeping pile on the body it removes, and so does a chunk
//! crushed to dust on the GPU (`World::sleeping_pile`). Kill bounds destroy through
//! `destroy_body`. A split's pieces are awake and wake what they touch.
//!
//!   cargo test --release -p engine-world --test unsupported_sleepers -- --test-threads=1 --nocapture

use rubble::*;

const HALF: f32 = 0.5;

fn world() -> World {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let mut solver = w.solver();
    solver.sleep_time = 0.5;
    w.set_solver(solver);
    let ground = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(
        ground,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )
    .unwrap();
    w
}

/// A unit cube with its centre at height `y` over the origin.
fn cube(w: &mut World, y: f32, body_type: BodyType) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            body_type,
            ..BodyDef::dynamic([0.0, y, 0.0])
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [HALF; 3]).unwrap();
    b
}

fn run(w: &mut World, steps: u32) {
    for _ in 0..steps {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
}

fn height(w: &World, b: BodyId) -> f32 {
    w.body_transform(b).unwrap().expect("a polled pose").position[1]
}

/// Steps until every one of `bodies` sleeps (at most 600), then 30 more.
fn settle(w: &mut World, bodies: &[BodyId]) {
    for _ in 0..600 {
        run(w, 1);
        if bodies.iter().all(|&b| !w.body_is_awake(b).unwrap()) {
            run(w, 30);
            return;
        }
    }
    panic!("the bodies never fell asleep");
}

/// A cube asleep on a static pedestal; the pedestal is destroyed. The cube comes down to the
/// ground.
#[test]
fn a_sleeper_falls_when_its_static_support_is_destroyed() {
    let mut w = world();
    let pedestal = cube(&mut w, HALF, BodyType::Static);
    let top = cube(&mut w, 3.0 * HALF + 0.01, BodyType::Dynamic);
    settle(&mut w, &[top]);
    let rested = height(&w, top);
    assert!((rested - 3.0 * HALF).abs() < 0.05, "it rests on the pedestal: {rested}");
    w.destroy_body(pedestal).unwrap();
    run(&mut w, 240);
    let after = height(&w, top);
    eprintln!("static support destroyed: {rested:.3} m -> {after:.3} m");
    assert!((after - HALF).abs() < 0.05, "the cube is left in the air at {after} m (it rested at {rested} m)");
}

/// Three cubes stacked and asleep; the bottom one (dynamic, asleep) is destroyed. Both above
/// come down by its height: the wake-up reaches the one that never touched it.
#[test]
fn a_sleeping_stack_falls_when_its_bottom_sleeper_is_destroyed() {
    let mut w = world();
    let bottom = cube(&mut w, HALF + 0.01, BodyType::Dynamic);
    let middle = cube(&mut w, 3.0 * HALF + 0.02, BodyType::Dynamic);
    let top = cube(&mut w, 5.0 * HALF + 0.03, BodyType::Dynamic);
    settle(&mut w, &[bottom, middle, top]);
    let rested = [height(&w, middle), height(&w, top)];
    w.destroy_body(bottom).unwrap();
    run(&mut w, 300);
    let after = [height(&w, middle), height(&w, top)];
    eprintln!("bottom sleeper destroyed: middle {:.3} -> {:.3} m, top {:.3} -> {:.3} m", rested[0], after[0], rested[1], after[1]);
    assert!((after[0] - HALF).abs() < 0.05, "the middle cube is left at {} m", after[0]);
    assert!((after[1] - 3.0 * HALF).abs() < 0.08, "the top cube is left at {} m", after[1]);
}

/// A plank asleep across two static posts; one post is destroyed. The plank cannot stay
/// level on one post: its free end comes down.
#[test]
fn a_sleeping_plank_tips_when_one_of_its_two_posts_is_destroyed() {
    let mut w = world();
    let post = |w: &mut World, x: f32| {
        let b = w
            .create_body(&BodyDef {
                body_type: BodyType::Static,
                ..BodyDef::dynamic([x, 1.0, 0.0])
            })
            .unwrap();
        w.create_box_shape(b, &ShapeDef::default(), [0.2, 1.0, 0.2]).unwrap();
        b
    };
    let (left, _right) = (post(&mut w, -1.5), post(&mut w, 1.5));
    let plank = w.create_body(&BodyDef::dynamic([0.0, 2.11, 0.0])).unwrap();
    w.create_box_shape(plank, &ShapeDef::default(), [1.9, 0.1, 0.3]).unwrap();
    settle(&mut w, &[plank]);
    let rested = height(&w, plank);
    w.destroy_body(left).unwrap();
    run(&mut w, 300);
    let after = height(&w, plank);
    eprintln!("one post destroyed: plank centre {rested:.3} m -> {after:.3} m");
    assert!(after < rested - 0.3, "the plank still lies level at {after} m on one post");
}

/// A cube asleep on the roof of a five-storey bearing-wall building (a threshold
/// destructible with support loss and blast loading); a large charge at its foot brings the
/// building down. The roof's pieces start to fall from rest, which no contact rule notices:
/// the cube must come down with them.
#[cfg(feature = "stress")]
#[test]
fn a_sleeper_on_a_roof_comes_down_with_the_building() {
    use engine_fracture::district::DistrictScene;
    use rubble::lab::*;
    let def = LabDef {
        source: LabSource::district(DistrictScene::Block).unwrap(),
        arrangement: LabArrangement::One,
        settings: LabSettings {
            bonds: LabBonds::Model(StrengthModel::Threshold),
            support_loss: true,
            blast_loading: true,
            ..LabSettings::default()
        },
    };
    let plan = LabPlan::new(&def).unwrap();
    // The lab's stress worlds never sleep; this one sleeps as a large map's does
    // (`DestructibleAssetDesc::sleeping_solver_for`: the default sleep thresholds, and any
    // touch from an awake body wakes).
    let mut wd = plan.world_def().unwrap();
    let d = engine_schema::AvbdParams::default();
    wd.enable_sleep = true;
    wd.solver.sleep_lin = d.sleep_lin;
    wd.solver.sleep_ang = d.sleep_ang;
    wd.solver.sleep_time = d.sleep_time;
    wd.solver.flags |= engine_schema::FLAG_WAKE_ON_TOUCH;
    let mut w = World::new(GpuContext::Default, &wd).unwrap();
    let mut lab = Lab::place(&mut w, &plan).unwrap();
    // The roof's top: the footings (0.4 m) and five 3 m storeys.
    let roof = 0.4 + 5.0 * 3.0;
    let cube = w.create_body(&BodyDef::dynamic([0.3, roof + HALF + 0.01, 2.0])).unwrap();
    w.create_box_shape(cube, &ShapeDef::default(), [HALF; 3]).unwrap();
    let step = |w: &mut World, lab: &mut Lab, n: u32| {
        for _ in 0..n {
            lab.step(w).unwrap();
            w.poll_blocking().unwrap();
        }
    };
    for _ in 0..600 {
        step(&mut w, &mut lab, 1);
        if !w.body_is_awake(cube).unwrap() {
            break;
        }
    }
    step(&mut w, &mut lab, 30);
    assert!(!w.body_is_awake(cube).unwrap(), "the cube sleeps on the roof");
    let rested = height(&w, cube);
    assert!((rested - (roof + HALF)).abs() < 0.1, "it rests on the roof: {rested}");
    let whole = lab.stats(&w).unwrap();
    assert_eq!(whole.bonds_intact, whole.bonds, "the building stands whole under it");
    // Charges along the ground floor, inside.
    for x in [-5.0, 0.0, 5.0] {
        for z in [-3.0, 3.0] {
            lab.queue(&w, &LabAction::Blast { position: [x, 1.5, z], tnt_kg: 100.0 }, 0).unwrap();
        }
    }
    step(&mut w, &mut lab, 600);
    let st = lab.stats(&w).unwrap();
    let after = height(&w, cube);
    eprintln!(
        "building blown: {} of {} bonds broken, {} bodies; cube {rested:.2} m -> {after:.2} m",
        st.bonds - st.bonds_intact,
        st.bonds,
        st.bodies
    );
    assert!(st.bonds - st.bonds_intact > st.bonds / 4, "the charges bring the building down");
    assert!(after < rested - 3.0, "the cube is left in the air at {after:.2} m (it rested at {rested:.2} m)");
}

/// A small cube asleep on one top block of a wall; a shot crushes that block to dust
/// (`CrushMass::Remove`) without touching the cube. The block is gone, not moved: no awake
/// body is left where it was. The cube must drop into the gap.
#[cfg(feature = "stress")]
#[test]
fn a_sleeper_drops_when_the_chunk_under_it_is_crushed_to_dust() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../engine-webgpu/fracture/packs/crush-wall.json");
    let asset = DestructibleAssetDesc::from_pack(&std::fs::read(path).unwrap()).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.sleeping_solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut capacity = stress_rubble::capacity_for(&scene);
    capacity.max_bodies = capacity.max_bodies.max((chunks as u32 + 64).next_power_of_two());
    capacity.max_colliders = capacity.max_colliders.max(capacity.max_bodies + chunks as u32 + 64);
    capacity.max_contacts = stress_rubble::threshold::contact_capacity(chunks, capacity.max_contacts, 1 << 30);
    capacity.max_candidates = capacity.max_candidates.max(capacity.max_contacts);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: true,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )
    .unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let a = w.create_destructible_asset(&asset).unwrap();
    let wall = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            crush: Some(Crush {
                mass: CrushMass::Remove,
                ..Crush::default()
            }),
            ..DestructibleDef::new(a)
        })
        .unwrap();
    // The top block on the struck face (+x) nearest z = 0.2.
    let at = |c: u32| {
        let p = w.destructible_chunk_pose(wall, c).unwrap().position;
        [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]
    };
    let top = (0..asset.chunks.len() as u32).map(|c| at(c)[1]).fold(f32::MIN, f32::max);
    let block = (0..asset.chunks.len() as u32)
        .filter(|&c| (at(c)[1] - top).abs() < 0.05 && at(c)[0] > 0.0)
        .min_by(|&a, &b| (at(a)[2] - 0.2).abs().total_cmp(&(at(b)[2] - 0.2).abs()))
        .unwrap();
    let b = at(block);
    let small = 0.12f32;
    let cube = w.create_body(&BodyDef::dynamic([b[0], b[1] + 0.2 + small + 0.005, b[2]])).unwrap();
    w.create_box_shape(cube, &ShapeDef::default(), [small; 3]).unwrap();
    for _ in 0..600 {
        w.step_sync().unwrap();
        if !w.body_is_awake(cube).unwrap() {
            break;
        }
    }
    for _ in 0..30 {
        w.step_sync().unwrap();
    }
    assert!(!w.body_is_awake(cube).unwrap(), "the cube sleeps on the wall");
    let rested = height(&w, cube);
    // A dense ball smaller than the block, through the block's middle: 8 cm under the cube.
    let r = 0.1f32;
    let ball = w
        .create_body(&BodyDef {
            linear_velocity: [-60.0, 0.0, 0.0],
            ..BodyDef::dynamic([b[0] + 4.0, b[1] - 0.02, b[2]])
        })
        .unwrap();
    w.create_sphere_shape(ball, &ShapeDef { update_body_mass: false, ..ShapeDef::default() }, [0.0; 3], r).unwrap();
    let (mass, i) = (2000.0f32, 0.4 * 2000.0 * r * r);
    w.body_set_mass_data(ball, MassData { mass, center: [0.0; 3], rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]] }).unwrap();
    w.body_set_projectile(ball, Some(Projectile::default())).unwrap();
    let mut crushed = Vec::new();
    for _ in 0..240 {
        let f = w.step_sync().unwrap();
        for e in &f.destruction_events {
            if let DestructionEvent::ChunkCrushed { chunk, .. } = e {
                crushed.push(*chunk);
            }
        }
    }
    let after = height(&w, cube);
    eprintln!("crushed {crushed:?} (the cube's block is {block}); cube {rested:.3} m -> {after:.3} m");
    assert!(crushed.contains(&block), "the shot crushes the block under the cube");
    assert!(after < rested - 0.2, "the cube is left in the air at {after:.3} m over the crushed block (it rested at {rested:.3} m)");
}
