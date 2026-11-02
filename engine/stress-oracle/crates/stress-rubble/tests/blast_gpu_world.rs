//! vibe-land's stress model in the GPU-only world (`Profile::BlastGpu`: stress-gpu's Slang port
//! of Blast's solver in `ThresholdWorld`'s fracture frame). Under gravity alone a house is a
//! static structure on its supports: the world's Blast solve must be the f32 reference's,
//! frame for frame. Struck, it breaks and splits on the GPU with nothing read back, and the
//! run repeats bit for bit.
//!
//!   cargo test --release -p stress-rubble --features gpu --test blast_gpu_world
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::{AngLin, BlastSolver};
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Scene};
use stress_rubble::threshold::ThresholdWorld;

fn house() -> Scene {
    let p = Profile::BlastGpu;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let mut scene =
        stress_ref::scene_pack::import_with(&path, "house-1story", "house", p.import_options())
            .expect("house-1story")
            .with_derived();
    p.apply_scene(&mut scene);
    scene
        .materials
        .entry("projectile".into())
        .or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

fn world(gpu: &Gpu, scene: &Scene) -> ThresholdWorld {
    ThresholdWorld::with_profile(gpu, scene, 0.35, 1400.0, Profile::BlastGpu)
        .expect("blast_gpu world")
}

fn difference(a: &[AngLin], b: &[AngLin]) -> f64 {
    let scale = b
        .iter()
        .map(|j| j.lin.norm().max(j.ang.norm()))
        .fold(0.0, f64::max)
        .max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.lin - y.lin).norm().max((x.ang - y.ang).norm()))
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn a_house_under_gravity_is_the_reference_solve_frame_for_frame() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let scene = house();
    let mut w = world(&gpu, &scene);
    let graph = w.fracture.blast().expect("Blast set").solver.graph.clone();
    let mut reference = BlastSolver::new(graph.clone());
    reference.device_precision = true;
    // The house is anchored: its structure frame is the world's, and gravity is its only load.
    let g = Vec3::from_array(scene.gravity);
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: g
        };
        graph.node_count()
    ];
    for frame in 0..20 {
        w.step_frame(&gpu).expect("frame");
        reference.solve(&accel, true);
        let device = w.fracture.blast().unwrap().solver.impulses();
        let r: Vec<AngLin> = (0..graph.bond_count())
            .map(|j| reference.impulse(j))
            .collect();
        let e = difference(&device, &r);
        println!("frame {frame}: {e:.3e} from the f32 reference");
        // Bounded by f32 rounding's spread on this house (blast_host_vs_reference: 3e-3 over
        // the first frames), so the frame's inputs are the reference's.
        assert!(e < 3e-2, "frame {frame}: {e:.3e}");
    }
    let stats = w.poll(&gpu);
    assert_eq!(stats.error, 0, "{stats:?}");
}

/// A ball through the house: the bonds Blast breaks split the house on the GPU.
fn strike(gpu: &Gpu) -> (Vec<bool>, u32, u32, Vec<f64>) {
    let scene = house();
    let mut w = world(gpu, &scene);
    let all: Vec<_> = scene.bodies.iter().flat_map(|b| &b.chunks).collect();
    let c = all[all.len() / 2].center;
    let shot = ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.35 },
        mass: 1400.0,
        position: [c[0] - 6.0, c[1], c[2]],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [60.0, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    };
    for frame in 0..60 {
        if frame == 2 {
            assert!(w.launch(gpu, &shot));
        }
        w.step_frame(gpu).expect("frame");
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let stats = w.poll(gpu);
    assert_eq!(stats.error, 0, "{stats:?}");
    let blast = w.fracture.blast().unwrap().solver.health();
    (w.fracture.bonds_intact(gpu), stats.broken, stats.bodies, blast)
}

#[test]
fn a_struck_house_breaks_splits_and_repeats() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let (intact, broken, bodies, blast) = strike(&gpu);
    let intact_count = intact.iter().filter(|b| **b).count();
    println!(
        "{broken} bonds broken ({} of {} intact), {bodies} bodies",
        intact_count,
        intact.len()
    );
    assert!(
        broken > 0 && bodies > 1,
        "the strike breaks bonds and splits the house"
    );
    // The counters are read a frame or more late (`poll` never waits); the bonds were read after
    // the GPU finished.
    assert!(broken as usize <= intact.len() - intact_count);
    // One bond state: every bond the fracture broke (Blast's breaks, and the strike's own
    // predicted ones) is broken in Blast too, and the other way round.
    let differ: Vec<usize> = (0..intact.len()).filter(|&j| intact[j] != (blast[j] > 0.0)).collect();
    assert!(differ.is_empty(), "bonds whose state the fracture and Blast disagree on: {differ:?}");
    let again = strike(&gpu);
    assert_eq!(again.0, intact, "the run repeats bit for bit");
}

#[test]
fn a_frame_at_rest_sends_the_gpu_nothing() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    // The destruction stage keeps its state on the GPU: once warm, a frame without stimuli
    // writes no buffer and creates none (stress-gpu's `Gpu::uploads` counts every byte it
    // sends: the fracture's frame state, the Blast parameters, the ingest's passes).
    let scene = house();
    let mut w = world(&gpu, &scene);
    for _ in 0..5 {
        w.step_frame(&gpu).expect("frame");
    }
    let before = gpu.upload_bytes();
    for _ in 0..30 {
        w.step_frame(&gpu).expect("frame");
    }
    let sent = gpu.upload_bytes() - before;
    println!("30 frames at rest: {sent} bytes uploaded");
    assert_eq!(sent, 0, "{sent} bytes over 30 frames at rest");
}

#[test]
fn a_chunk_load_loads_the_solve_for_one_frame() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    // PhysX's chunk loads: a force and a torque on two chunks for one frame enter the solve as
    // F/m and -T/I (prepareLoads); the frame after has none. The anchored house's structure
    // frame is the world's, so the reference takes them as they are.
    use stress_gpu::blast::ChunkLoad;
    let scene = house();
    let mut w = world(&gpu, &scene);
    let graph = w.fracture.blast().expect("Blast set").solver.graph.clone();
    let mut reference = BlastSolver::new(graph.clone());
    reference.device_precision = true;
    let g = Vec3::from_array(scene.gravity);
    let gravity = vec![AngLin { ang: Vec3::ZERO, lin: g }; graph.node_count()];
    let dynamic: Vec<usize> = (0..graph.node_count()).filter(|&k| graph.nodes[k].mass > 0.0).collect();
    let loads = [
        ChunkLoad { chunk: dynamic[dynamic.len() / 3] as u32, force: Vec3::new(3e4, 0.0, -8e4), torque: Vec3::new(2e3, -4e3, 1e3), ..Default::default() },
        ChunkLoad { chunk: dynamic[2 * dynamic.len() / 3] as u32, force: Vec3::new(0.0, -5e4, 0.0), torque: Vec3::new(0.0, 0.0, 6e3), ..Default::default() },
    ];
    let mut loaded = gravity.clone();
    for l in &loads {
        let n = graph.nodes[l.chunk as usize];
        loaded[l.chunk as usize].lin = g + l.force * (1.0 / n.mass);
        loaded[l.chunk as usize].ang = l.torque * (-1.0 / n.effective_inertia());
    }
    let mut unloaded = BlastSolver::new(graph.clone());
    unloaded.device_precision = true;
    for frame in 0..6 {
        let accel = if frame == 3 { &loaded } else { &gravity };
        if frame == 3 {
            w.fracture.blast_mut().unwrap().set_chunk_loads(&gpu, &loads).unwrap();
        }
        w.step_frame(&gpu).expect("frame");
        reference.solve(accel, true);
        unloaded.solve(&gravity, true);
        let device = w.fracture.blast().unwrap().solver.impulses();
        let r: Vec<AngLin> = (0..graph.bond_count()).map(|j| reference.impulse(j)).collect();
        let u: Vec<AngLin> = (0..graph.bond_count()).map(|j| unloaded.impulse(j)).collect();
        let (e, moved) = (difference(&device, &r), difference(&r, &u));
        println!("frame {frame}: {e:.3e} from the reference; the load moved it {moved:.3e}");
        assert!(e < 3e-2, "frame {frame}: {e:.3e}");
        if frame == 3 {
            assert!(moved > 0.1, "the chunk loads change the bond forces: {moved:.3e}");
        }
    }
}

#[test]
fn bonds_the_fracture_breaks_itself_break_in_blast_too() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    // One bond state, as the target has: a bond the fracture breaks itself (here the host's
    // breaks; a predicted strike's or a crushed chunk's alike) leaves Blast's islands too,
    // or Blast would go on loading a bond the split has cut.
    let scene = house();
    let mut w = world(&gpu, &scene);
    for _ in 0..3 {
        w.step_frame(&gpu).expect("frame");
    }
    let graph = w.fracture.blast().unwrap().solver.graph.clone();
    let chosen: Vec<u32> = (0..graph.bond_count() as u32).step_by(97).collect();
    w.fracture.break_bonds_sparse(&chosen);
    for _ in 0..3 {
        w.step_frame(&gpu).expect("frame");
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let intact = w.fracture.bonds_intact(&gpu);
    let blast = w.fracture.blast().unwrap().solver.health();
    for &j in &chosen {
        assert!(!intact[j as usize] && !(blast[j as usize] > 0.0), "bond {j}: fracture intact {}, Blast health {}", intact[j as usize], blast[j as usize]);
    }
    let differ: Vec<usize> = (0..intact.len()).filter(|&j| intact[j] != (blast[j] > 0.0)).collect();
    assert!(differ.is_empty(), "bonds whose state the fracture and Blast disagree on: {differ:?}");
}

#[test]
fn crushing_in_blasts_frame_takes_the_bond_forces_virial() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    // PhysX's evaluateChunkMaterials: a chunk's stress is the virial of its bonds' solved forces
    // at their centroids (with its contacts', none on the standing house): its pressure and
    // deviator after the first frame are those of Blast's own forces, as the host computes them.
    use stress_rubble::threshold::FractureOptions;
    let mut scene = house();
    for (name, m) in scene.materials.iter_mut() {
        if name != "projectile" {
            m.crush = Some(stress_ref::crush::presets::concrete());
        }
    }
    let mut w = ThresholdWorld::with_options(&gpu, &scene, 0.35, 1400.0, Profile::BlastGpu, FractureOptions { crush: true, ..FractureOptions::default() })
        .expect("blast_gpu world with crushing");
    w.step_frame(&gpu).expect("frame");
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let graph = w.fracture.blast().unwrap().solver.graph.clone();
    let impulses = w.fracture.blast().unwrap().solver.impulses();
    let health = w.fracture.blast().unwrap().solver.health();
    let state = w.fracture.crush_state(&gpu);
    let rows = &w.tables.crush.as_ref().unwrap().chunks;
    let mut compared = 0;
    for k in 0..graph.node_count() {
        if rows[k][1][2].to_bits() & 1 == 0 {
            continue;
        }
        let mut v = [0.0f64; 6];
        for (j, b) in graph.bonds.iter().enumerate() {
            if !(health[j] > 0.0) || !b.nodes.contains(&k) {
                continue;
            }
            let f = if b.nodes[0] == k { impulses[j].lin } else { impulses[j].lin * -1.0 };
            let r = b.centroid - graph.nodes[k].position;
            let s = stress_ref::crush::sym_outer([r.x, r.y, r.z], [f.x, f.y, f.z]);
            for c in 0..6 {
                v[c] += s[c];
            }
        }
        let (p, q) = stress_ref::crush::invariants(v, rows[k][1][1] as f64);
        let (dp, dq) = (state[k][1] as f64, state[k][2] as f64);
        let scale = p.abs().max(q).max(1.0);
        assert!((dp - p.max(0.0)).abs() <= 1e-3 * scale, "chunk {k}: pressure {dp} vs {p}");
        assert!((dq - q).abs() <= 1e-3 * scale, "chunk {k}: deviator {dq} vs {q}");
        compared += 1;
    }
    println!("{compared} crushable chunks' stresses from Blast's bond forces");
    assert!(compared > 100);
}

/// The lab's cuboid house as stress-web loads it for `blast_gpu`: a scene without Blast
/// authoring, so Blast's inputs come from the structures the threshold world resolved.
fn cuboid_house() -> Scene {
    let (_lab, json) = engine_fracture::samples::cuboid_stress_scene(None).expect("cuboid house");
    let mut scene = stress_ref::shardwright::import_value_with(&json, "cuboid_houses", true).expect("import").with_derived();
    scene.sim.methods = stress_ref::scene::Methods::established();
    Profile::BlastGpu.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

#[test]
fn a_scene_without_blast_authoring_stands_and_breaks_under_blast() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let scene = cuboid_house();
    assert!(scene.blast.is_none());
    // Under gravity alone the house stands: Blast breaks nothing.
    let mut w = world(&gpu, &scene);
    let graph = w.fracture.blast().expect("Blast set").solver.graph.clone();
    assert_eq!(graph.node_count(), scene.bodies.iter().map(|b| b.chunks.len()).sum::<usize>());
    for _ in 0..120 {
        w.step_frame(&gpu).expect("frame");
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let stats = w.poll(&gpu);
    assert_eq!(stats.error, 0, "{stats:?}");
    assert_eq!(stats.broken, 0, "the house stands under gravity: {stats:?}");
    // A ball through it breaks bonds and splits it.
    let mut w = world(&gpu, &scene);
    let all: Vec<_> = scene.bodies.iter().flat_map(|b| &b.chunks).collect();
    let c = all[all.len() / 2].center;
    let shot = ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.35 },
        mass: 1400.0,
        position: [c[0] - 6.0, c[1], c[2]],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [60.0, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    };
    for frame in 0..60 {
        if frame == 2 {
            assert!(w.launch(&gpu, &shot));
        }
        w.step_frame(&gpu).expect("frame");
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let stats = w.poll(&gpu);
    println!("struck: {stats:?}");
    assert_eq!(stats.error, 0, "{stats:?}");
    assert!(stats.broken > 0 && stats.bodies > 1, "the strike breaks bonds and splits the house: {stats:?}");
}
