//! Every stress model in the world API (`StrengthModel::{Threshold, LoadPath, ImpactLocal,
//! BlastGpu}`) against the lab's own fracture world (stress-rubble `ThresholdWorld`) on the
//! same scene and shot: both set their fracture up through stress-rubble's `apply_profile`,
//! so they must agree on the bonds the shot breaks (as a set), the bodies the structure
//! splits into, and, for `blast_gpu`, Blast's bond forces against the f32 reference solve.
//!
//! Native wgpu worlds are not bit-reproducible run to run, and the two worlds number their
//! body slots differently (the lab parks its 16 shots between the clusters and the spare
//! slots), so after the first split the cascade may differ: the first break is compared
//! exactly, the totals within a documented tolerance.
//!
//!   cargo test --release -p engine-world --test stress_models -- --test-threads=1
#![cfg(feature = "stress")]

use rubble::*;
use stress_ref::capability::Profile;

/// The capacity the lab's `ThresholdWorld` gives the scene.
fn capacity(asset: &DestructibleAssetDesc, model: StrengthModel) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(model).unwrap();
    let scene = stress.stress_scene(model, 1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

fn profile(model: StrengthModel) -> Profile {
    match model {
        StrengthModel::Threshold => Profile::Threshold,
        StrengthModel::LoadPath => Profile::LoadPath,
        StrengthModel::ImpactLocal => Profile::ImpactLocal,
        StrengthModel::BlastGpu => Profile::BlastGpu,
        StrengthModel::Bonds => unreachable!(),
    }
}

const STRESS: [StrengthModel; 4] = [
    StrengthModel::Threshold,
    StrengthModel::LoadPath,
    StrengthModel::ImpactLocal,
    StrengthModel::BlastGpu,
];

/// The models to check (`STRESS_MODELS=load_path,blast_gpu`: those alone).
fn models() -> Vec<StrengthModel> {
    match std::env::var("STRESS_MODELS") {
        Ok(v) => v.split(',').map(|n| StrengthModel::parse(n).expect("a model name")).collect(),
        Err(_) => STRESS.to_vec(),
    }
}

/// The lab's shot at the scene's centre from straight in front: origin, velocity, radius
/// and mass, in the asset frame.
fn lab_shot(lab: &engine_fracture::Scene, model: StrengthModel) -> ([f64; 3], [f64; 3], f64, f64) {
    let c = lab.center;
    let origin = [c[0], c[1], c[2] + 3.0 * lab.extent.max(1.0)];
    // The analytic strike holes a wall only above its ballistic limit: the lab's default shot
    // rebounds from both houses under it, in both worlds, so it is fired at 6 times the
    // speed (`STRESS_SHOT_SPEED` overrides, a probe).
    let own = if model == StrengthModel::ImpactLocal { 6.0 } else { 1.0 };
    let k: f64 = std::env::var("STRESS_SHOT_SPEED").map_or(own, |v| v.parse().unwrap());
    let speed = k * engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed);
    let v = engine_fracture::plan::shot_velocity(
        c,
        lab.extent,
        lab.projectile_radius,
        origin,
        [0.0, 0.0, -1.0],
        Some(speed),
        lab.projectile_arc,
    );
    (origin, v, lab.projectile_radius, lab.projectile_mass)
}

/// Frames before any lab shot here arrives (the first strike lands at frame 13). The two
/// rigid engines settle a structure slightly differently (rubble's `World` and the lab's
/// `RubbleEngine` already differ in contact count at step 1): the cuboid house settles with 2
/// bonds broken at frame 2 in the lab and none in the world, on main before this test too.
/// Settling breaks are reported, not compared; the strike is.
const SETTLE: usize = 10;

struct Outcome {
    /// Each bond's state at the end, in the tables' order.
    intact: Vec<bool>,
    /// The structure's bodies at the end.
    bodies: u32,
    /// Bonds broken by each frame's end.
    timeline: Vec<u32>,
}

impl Outcome {
    fn broken(&self) -> u32 {
        self.intact.iter().filter(|&&i| !i).count() as u32
    }

    /// Bonds broken before the shot can arrive (frame [`SETTLE`]): the structure settling.
    fn settle_breaks(&self) -> u32 {
        self.timeline.get(SETTLE - 1).copied().unwrap_or(0)
    }

    /// The strike's first break: the first frame from [`SETTLE`] on that breaks more bonds
    /// than the settling did, and how many it adds.
    fn first_break(&self) -> Option<(usize, u32)> {
        let before = self.settle_breaks();
        let f = (SETTLE..self.timeline.len()).find(|&f| self.timeline[f] > before)?;
        Some((f, self.timeline[f] - before))
    }
}

/// The world: `asset` placed under `model` at the lab's stress world, the ground plane,
/// the lab's shot (a continuous-collision ball with the in-step contact fracture, as the
/// lab's shots are then), `frames` steps.
fn run_world(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
    model: StrengthModel,
    frames: u32,
    shot: bool,
) -> (World, DestructibleId, Outcome) {
    let stress = asset.stress.as_ref().unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let solver = asset.solver_for(model).unwrap();
    let def = WorldDef {
        capacity: capacity(asset, model),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
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
    let a = w.create_destructible_asset(asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    if shot {
        let (origin, v, r, m) = lab_shot(lab, model);
        let at = [0, 1, 2].map(|k| (origin[k] + offset[k] as f64) as f32);
        // The lab's shots are continuous-collision bodies when the contact fracture is on.
        let contact = profile(model).contact_fracture().unwrap() > 0;
        let ball = w
            .create_body(&BodyDef {
                linear_velocity: v.map(|x| x as f32),
                is_bullet: contact,
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
    }
    let stats = w.destructible_stats(id).unwrap();
    let mut timeline = Vec::new();
    for _ in 0..frames {
        let r = w.step();
        w.poll_blocking().unwrap();
        let s = w.destructible_stats(id).unwrap();
        timeline.push(s.bonds - s.bonds_intact);
        // STRESS_TRACE=1 (a probe): each step's counters, in both worlds.
        if std::env::var("STRESS_TRACE").is_ok() {
            let c = w.counters().unwrap();
            eprintln!("world {} step {}: contacts {} awake {} broken {}", model.name(), timeline.len(), c.contact_count, c.awake_count, s.bonds - s.bonds_intact);
        }
        r.unwrap();
    }
    let s = w.destructible_stats(id).unwrap();
    let intact = (0..stats.bonds)
        .map(|b| w.destructible_bond_intact(id, b).unwrap())
        .collect();
    let mut bodies: Vec<u32> = (0..s.chunks)
        .map(|c| w.destructible_chunk_body(id, c).unwrap().index())
        .collect();
    bodies.sort_unstable();
    bodies.dedup();
    let out = Outcome {
        intact,
        bodies: bodies.len() as u32,
        timeline,
    };
    (w, id, out)
}

/// The lab's fracture world on the same scene and shot (stress frame: the asset frame less
/// its placement, turned Z up).
fn run_lab(
    asset: &DestructibleAssetDesc,
    lab: &engine_fracture::Scene,
    model: StrengthModel,
    frames: u32,
) -> Outcome {
    let (info, device, queue) = engine_webgpu::default_device().unwrap();
    let gpu = stress_gpu::gpu::Gpu::from_device(info, device, queue);
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(model).unwrap();
    let scene = stress.stress_scene(model, 1.0, solver.dt).unwrap();
    let (origin, v, r, m) = lab_shot(lab, model);
    let p = stress.placement();
    let zup = |v: [f64; 3]| [v[0], -v[2], v[1]];
    let mut world =
        stress_rubble::threshold::ThresholdWorld::with_profile(&gpu, &scene, r, m, profile(model))
            .unwrap();
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
        if std::env::var("STRESS_TRACE").is_ok() {
            let st = world.poll(&gpu);
            eprintln!("lab {} step {}: contacts {} awake {} broken {} error {}", model.name(), timeline.len(), st.contacts, st.awake, st.broken, st.error);
        }
    }
    let c = world.fracture.counters(&gpu);
    assert_eq!(c.error, 0, "the lab's fracture world errs: {c:?}");
    let mut owners = world.fracture.chunk_bodies(&gpu);
    owners.sort_unstable();
    owners.dedup();
    Outcome {
        intact: world.fracture.bonds_intact(&gpu),
        bodies: owners.len() as u32,
        timeline,
    }
}

fn house() -> (DestructibleAssetDesc, engine_fracture::Scene) {
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None).unwrap();
    (asset, lab)
}

fn house_1story() -> (DestructibleAssetDesc, engine_fracture::Scene) {
    let pack = include_bytes!("../../../stress-oracle/packs/house-1story.json");
    let asset = DestructibleAssetDesc::from_pack(pack).unwrap();
    let (lab, _) = engine_fracture::pack::build_pack_scene(pack).unwrap();
    (asset, lab)
}

/// The world and the lab on one scene under `model`: the same first break, the broken
/// bonds as sets within `tolerance` of each other (Jaccard distance), the bodies within
/// the same.
fn check(name: &str, (asset, lab): &(DestructibleAssetDesc, engine_fracture::Scene), model: StrengthModel) {
    let frames = 120;
    let (_, _, world) = run_world(asset, lab, model, frames, true);
    let lab_run = run_lab(asset, lab, model, frames);
    let both = world
        .intact
        .iter()
        .zip(&lab_run.intact)
        .filter(|(a, b)| !**a && !**b)
        .count() as f64;
    let either = world
        .intact
        .iter()
        .zip(&lab_run.intact)
        .filter(|(a, b)| !**a || !**b)
        .count() as f64;
    let jaccard = if either == 0.0 { 1.0 } else { both / either };
    eprintln!(
        "{name} {}: world broke {} bonds into {} bodies, the lab {} into {}; settling broke {} / {}; first breaks {:?} / {:?}; broken sets' Jaccard {jaccard:.3}",
        model.name(),
        world.broken(),
        world.bodies,
        lab_run.broken(),
        lab_run.bodies,
        world.settle_breaks(),
        lab_run.settle_breaks(),
        world.first_break(),
        lab_run.first_break()
    );
    assert_eq!(world.intact.len(), lab_run.intact.len(), "{name}: the same bonds");
    assert!(
        lab_run.broken() > 0,
        "{name} {}: the lab's shot breaks something",
        model.name()
    );
    assert_eq!(
        world.first_break(),
        lab_run.first_break(),
        "{name} {}: the first break",
        model.name()
    );
    // The cascade after the first split follows the body slots the pieces take (see the
    // module comment): within a third, as tests/threshold.rs holds the threshold model.
    assert!(
        jaccard >= 0.67,
        "{name} {}: broken sets' Jaccard {jaccard:.3}",
        model.name()
    );
    let ratio = world.bodies as f64 / lab_run.bodies as f64;
    assert!(
        (0.75..=1.33).contains(&ratio),
        "{name} {}: {} bodies against the lab's {}",
        model.name(),
        world.bodies,
        lab_run.bodies
    );
}

#[test]
fn every_stress_model_breaks_the_cuboid_house_as_the_lab() {
    let scene = house();
    for model in models() {
        check("cuboid house", &scene, model);
    }
}

#[test]
fn every_stress_model_breaks_house_1story_as_the_lab() {
    let scene = house_1story();
    for model in models() {
        check("house-1story", &scene, model);
    }
}

/// `blast_gpu` at rest: the world's Blast solve of the anchored pack is the f32 reference
/// solve (stress-ref `BlastSolver`, device precision) frame for frame, as the lab's is
/// (stress-rubble tests/blast_gpu_world.rs). The world steps its fracture at rest for this
/// model (the warm-started solve converges over frames), though nothing is awake.
#[test]
fn blast_gpu_forces_are_the_reference_solve() {
    use stress_ref::blast_bond::{AngLin, BlastSolver};
    use stress_ref::math::Vec3;
    let (asset, lab) = house_1story();
    let (mut w, _, _) = run_world(&asset, &lab, StrengthModel::BlastGpu, 0, false);
    let graph = w.stress_blast_graph().expect("Blast's graph");
    let mut reference = BlastSolver::new(graph.clone());
    reference.device_precision = true;
    let solver = asset.solver_for(StrengthModel::BlastGpu).unwrap();
    let scene = asset
        .stress
        .as_ref()
        .unwrap()
        .stress_scene(StrengthModel::BlastGpu, 1.0, solver.dt)
        .unwrap();
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::from_array(scene.gravity),
        };
        graph.node_count()
    ];
    for frame in 0..20 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        reference.solve(&accel, true);
        let device = w.stress_blast_impulses().unwrap();
        let r: Vec<AngLin> = (0..graph.bond_count()).map(|j| reference.impulse(j)).collect();
        let scale = r
            .iter()
            .map(|j| j.lin.norm().max(j.ang.norm()))
            .fold(0.0, f64::max)
            .max(1e-30);
        let e = device
            .iter()
            .zip(&r)
            .map(|(x, y)| (x.lin - y.lin).norm().max((x.ang - y.ang).norm()))
            .fold(0.0, f64::max)
            / scale;
        eprintln!("frame {frame}: {e:.3e} from the f32 reference");
        // BLAST_DIAG=1 (a probe): the bonds furthest from the reference.
        if std::env::var("BLAST_DIAG").is_ok() {
            let mut worst: Vec<(f64, usize)> = device
                .iter()
                .zip(&r)
                .enumerate()
                .map(|(j, (x, y))| ((x.lin - y.lin).norm().max((x.ang - y.ang).norm()) / scale, j))
                .collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(d, j) in worst.iter().take(6) {
                let b = &graph.bonds[j];
                eprintln!(
                    "  bond {j} nodes {:?} masses {:.1}/{:.1} diff {d:.3e} device lin {:?} ref lin {:?}",
                    b.nodes,
                    graph.nodes[b.nodes[0]].mass,
                    graph.nodes[b.nodes[1]].mass,
                    device[j].lin,
                    r[j].lin
                );
            }
        }
        // The lab's bound (blast_gpu_world.rs): f32 rounding's spread.
        assert!(e < 3e-2, "frame {frame}: {e:.3e}");
    }
}
