//! The Topple tower's charges (`stress_ref::topple`, the pack's `defaults.charges`) set
//! off on the GPU, between frames, as stress-web sets them off (`stress_detonate`).
//!
//! * threshold (the GPU-resident world): the charges cut their bonds (the thrown columns
//!   out over the street), the tower leans out over the street as one body, then falls
//!   across it and breaks up where it lands.
//! * blast_parity (the resident GpuWorld): the charges cut their bonds between frames and
//!   the tower comes free of its footings (the cut itself; what it does next is the
//!   profile's: README, Topple).
//!
//!   cargo test --release -p stress-rubble --features gpu --test topple
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::topple::{BACK_ROW, STREET_SIDE};
use stress_rubble::scenes::topple;

/// The charges are set off before this frame (half a second of standing first).
const AT: u64 = 30;

/// The storey of each chunk (`storey-N`), None for the supports.
fn storeys(scene: &stress_ref::scene::Scene) -> Vec<Option<usize>> {
    scene.bodies[0].chunks.iter().map(|c| c.groups.get(1).and_then(|g| g.strip_prefix("storey-")).and_then(|n| n.parse().ok())).collect()
}

/// The mean of `pos` over the chunks of storey `s`, the thrown columns left out.
fn centre(pos: &[Vec3], storey: &[Option<usize>], debris: &[usize], s: usize) -> Vec3 {
    let ks: Vec<usize> = (0..pos.len()).filter(|&k| storey[k] == Some(s) && !debris.contains(&k)).collect();
    ks.iter().fold(Vec3::ZERO, |a, &k| a + pos[k]) * (1.0 / ks.len() as f64)
}

#[test]
fn charges_topple_the_tower_across_the_street_threshold() {
    let gpu = Gpu::new().expect("GPU");
    let frames = 420;
    let (scene, mut demolition) = topple(Profile::Threshold, frames as f64 / 60.0);
    let storey = storeys(&scene);
    let top = storey.iter().flatten().copied().max().unwrap();
    let debris: Vec<usize> = demolition.charges.iter().flat_map(|c| c.debris.clone()).collect();
    let tower: Vec<usize> = (0..storey.len()).filter(|&k| storey[k].is_some() && !debris.contains(&k)).collect();
    let ball = stress_ref::wrecking::BALLS[0];
    let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
    let start: Vec<Vec3> = w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect();
    let charge_bonds: usize = demolition.charges.iter().map(|c| c.bonds.len()).sum();
    let mut cut = Vec::new();
    // The frame the last charge went off, and the lean a second after it.
    let mut released = None;
    let mut leaning = None;
    for f in 0..frames {
        if w.frame == AT {
            assert!(demolition.arm("", w.frame, w.frame_dt));
        }
        for c in demolition.due(w.frame) {
            cut.push((c.name.clone(), w.frame, w.detonate(&gpu, &c).expect("detonate")));
            if c.name == BACK_ROW {
                released = Some(w.frame);
            }
        }
        w.step_frame(&gpu).expect("frame");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let st = w.poll(&gpu);
        assert_eq!((st.error, st.engine_overflow), (0, 0), "frame {f}");
        if f as u64 + 1 == AT {
            // Standing until the charges: nothing broken, nothing moved.
            assert_eq!(st.broken, 0, "bonds broke before the charges");
        }
        if released.is_some_and(|r| w.frame == r + 60) {
            // A second after the tower came free: one body leaning out over the street.
            let pos: Vec<Vec3> = w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect();
            let body = w.fracture.chunk_bodies(&gpu);
            let mut on = std::collections::HashMap::new();
            for &k in &tower {
                *on.entry(body[k]).or_insert(0usize) += 1;
            }
            let largest = *on.values().max().unwrap() as f64 / tower.len() as f64;
            let axis = centre(&pos, &storey, &debris, top) - centre(&pos, &storey, &debris, 2);
            let lean = (-axis.y).atan2(axis.z).to_degrees();
            println!("a second after the back row: lean {lean:.1} deg, largest body {:.0}% of the storeys, broken {}", 100.0 * largest, st.broken);
            assert!(lean > 1.0, "leaning out over the street: {lean} deg");
            assert!(largest > 0.85, "one mass: its largest body {largest}");
            leaning = Some(lean);
        }
    }
    let names: Vec<&str> = cut.iter().map(|c| c.0.as_str()).collect();
    assert_eq!(names, [STREET_SIDE, BACK_ROW], "in firing order");
    assert_eq!(cut[0].1, AT);
    assert_eq!(cut.iter().map(|c| c.2).sum::<usize>(), charge_bonds, "every charge bond cut");
    assert!(leaning.is_some(), "the tower came free");
    let st = w.poll(&gpu);
    let pos: Vec<Vec3> = w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect();
    // The thrown columns out over the street; the top storey across it, on the ground.
    let thrown = debris.iter().map(|&k| start[k].y - pos[k].y).fold(f64::INFINITY, f64::min);
    let (top0, top1) = (centre(&start, &storey, &debris, top), centre(&pos, &storey, &debris, top));
    println!("{frames} frames: broken {} of {}, bodies {}, columns thrown at least {thrown:.1} m, top storey {:.1} m toward the street and at z {:.1} m", st.broken, scene.bodies[0].bonds.len(), st.bodies, top0.y - top1.y, top1.z);
    assert!(thrown > 3.0, "the street-side columns thrown out: {thrown} m");
    assert!(top0.y - top1.y > 15.0, "the top storey across the street: {} m", top0.y - top1.y);
    assert!(top1.z < 6.0, "the top storey on the ground: z {}", top1.z);
    assert!(st.broken as usize > charge_bonds + 100, "broken up where it landed: {} bonds", st.broken);
}

#[test]
fn charges_cut_free_between_frames_blast_parity() {
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParity;
    let frames = 60;
    let (scene, mut demolition) = topple(p, frames as f64 / 60.0);
    let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    let debris: Vec<usize> = demolition.charges.iter().flat_map(|c| c.debris.clone()).collect();
    for _ in 0..frames {
        if w.frame == AT {
            assert!(demolition.arm("", w.frame, scene.sim.frame_dt));
        }
        for c in demolition.due(w.frame) {
            let bonds = c.bond_indices(&scene.bodies[0].bonds).expect("charge bonds");
            if c.name == STREET_SIDE {
                w.sync(&gpu);
                assert_eq!(w.observation().values["broken_bonds"], 0.0, "nothing broke before the charges");
            }
            let n = w.cut_bonds(&gpu, 0, &bonds, &c.debris, c.kick).expect("cut");
            assert!(n > 0, "{}: bonds cut", c.name);
            // Every bond of the charge broken now, before the next frame steps.
            assert!(bonds.iter().all(|&b| !w.solver.mirror.bonds[0][b].connected()), "{}", c.name);
        }
        w.step_frame(&gpu).expect("frame");
    }
    w.sync(&gpu);
    // The thrown columns move out over the street; no chunk of the tower is anchored now.
    let o = w.observation();
    let moved: Vec<f64> = debris.iter().map(|&k| -o.bodies.values().next().unwrap()[k].displacement[1]).collect();
    println!("blast_parity: broken {}, fragments {}, thrown columns {moved:.1?} m", o.values["broken_bonds"], o.values["fragments"]);
    assert!(moved.iter().all(|&d| d > 1.0), "the street-side columns thrown out: {moved:?}");
    let storey = storeys(&scene);
    let m = &w.solver.mirror;
    assert!((0..storey.len()).filter(|&k| storey[k].is_some()).all(|k| !m.clusters[m.chunks[0][k].cluster].anchored), "the tower is free of its footings");
}
