//! The GPU fracture's change log (stress-gpu `fracture_log`): a struck house breaks and splits
//! (`Profile::BlastGpu`), and the state a host rebuilds from the log alone (broken bonds,
//! chunks' bodies, body frames) is the GPU's own, bit for bit, whether it reads every frame or
//! only now and then (entries wait until acknowledged).
//!
//!   cargo test --release -p stress-rubble --features gpu --test fracture_log
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::capability::Profile;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};
use stress_rubble::threshold::ThresholdWorld;

/// The host's view, from the tables and then the log only.
struct HostState {
    intact: Vec<bool>,
    chunk_body: Vec<u32>,
    frames: Vec<stress_gpu::fracture::FrameRecord>,
    cursor: stress_gpu::fracture_log::LogCursor,
}

fn run(gpu: &Gpu, read_every: usize) -> (usize, usize) {
    let p = Profile::BlastGpu;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let mut scene =
        stress_ref::scene_pack::import_with(&path, "house", "house", p.import_options())
            .unwrap()
            .with_derived();
    p.apply_scene(&mut scene);
    scene
        .materials
        .entry("projectile".into())
        .or_insert_with(stress_ref::builders::elastic_steel);
    let mut w = ThresholdWorld::with_profile(gpu, &scene, 0.35, 1400.0, p).expect("world");
    let mut host = HostState {
        intact: vec![true; w.tables.bond_count() as usize],
        chunk_body: w.tables.chunk_body.clone(),
        frames: w.tables.frames.clone(),
        cursor: [0; 3],
    };
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
    let (mut entries, mut reads) = (0usize, 0usize);
    for frame in 0..48 {
        if frame == 2 {
            assert!(w.launch(gpu, &shot));
        }
        w.step_frame(gpu).expect("frame");
        if frame % read_every != read_every - 1 && frame != 47 {
            continue;
        }
        reads += 1;
        let bytes: Vec<u32> = gpu.read(&w.fracture.log.buffer);
        let log = w
            .fracture
            .log
            .decode(bytemuck::cast_slice(&bytes), host.cursor);
        assert!(!log.overflow, "frame {frame}: the log overflowed");
        let mut last_step = 0;
        for &(j, step) in &log.bonds {
            assert!(step >= last_step, "entries in step order");
            last_step = step;
            assert!(host.intact[j as usize], "bond {j} broken twice");
            host.intact[j as usize] = false;
        }
        for &(c, body, _) in &log.chunks {
            host.chunk_body[c as usize] = body;
        }
        for &(b, _, f) in &log.frames {
            host.frames[b as usize] = f;
        }
        entries += log.bonds.len() + log.chunks.len() + log.frames.len();
        host.cursor = log.cursor;
        w.fracture.log.acknowledge(gpu, host.cursor, false);
        // The GPU's own state, read in full.
        assert_eq!(
            host.intact,
            w.fracture.bonds_intact(gpu),
            "frame {frame}: bonds"
        );
        assert_eq!(
            host.chunk_body,
            w.fracture.chunk_bodies(gpu),
            "frame {frame}: chunk bodies"
        );
        let frames = w.fracture.body_frames(gpu);
        let bits = |f: &[stress_gpu::fracture::FrameRecord]| {
            f.iter()
                .flat_map(|r| r.iter().flatten().map(|x| x.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&host.frames),
            bits(&frames),
            "frame {frame}: body frames"
        );
    }
    let broken = host.intact.iter().filter(|b| !**b).count();
    println!("read every {read_every}: {reads} reads, {entries} entries, {broken} bonds broken");
    assert!(
        broken > 0
            && host
                .chunk_body
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1,
        "the strike broke and split the house"
    );
    (broken, entries)
}

#[test]
fn a_host_rebuilds_the_gpu_state_from_the_log_alone() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let every = run(&gpu, 1);
    let sparse = run(&gpu, 7);
    assert_eq!(
        every.0, sparse.0,
        "the same breaks however often the host reads"
    );
}
