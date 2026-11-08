//! Continuous collision in the stress coupling, as production runs it: each capability
//! profile's one engine step a frame, the GPU-resident frame, with the bodies a scene marks
//! as fast movers (`ImpactorDesc::ccd`, on by default; `BodyDesc::ccd`) given the engine's
//! time-of-impact contacts (`Profile::engine_ccd`, rubble's `BODY_CCD`).
//!
//! - A marked ball never passes a 5 cm steel slab at 60-1200 m/s, in every profile and in
//!   the threshold world; unmarked, the same ball passes it (the control).
//! - A marked structure never passes it either, and every free piece it breaks into is a
//!   CCD body, on the GPU's split path; an unmarked structure's pieces are not, and an
//!   anchored part of a marked one is not (it is static).
//! - The frame reads nothing back for CCD: the resident frame's reads are the same with
//!   and without marked bodies (no whole-world substeps, no readback of their speeds).
//!
//!   cargo test --release -p stress-rubble --features gpu --test ccd_coupled
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::builders::*;
use stress_ref::capability::Profile;
use stress_ref::engine::BodyKey;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_rubble::scenes::realtime;
use stress_rubble::RubbleEngine;

/// The slab's faces (m, along +y): the ball comes from below.
const SLAB_FRONT: f64 = 1.0;
const SLAB_BACK: f64 = 1.05;

/// A fixed steel slab 5 cm thick, 1 m square, too strong to break: anything beyond it
/// passed through it.
fn slab_body() -> BodyDesc {
    let mut chunks = grid(Vec3::new(-0.5, SLAB_FRONT, -0.5), [5, 1, 5], Vec3::new(0.2, SLAB_BACK - SLAB_FRONT, 0.2), "steel");
    for c in &mut chunks {
        c.support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "steel".into());
    body("slab", chunks, bonds)
}

fn base(name: &str) -> Scene {
    let mut s = new_scene(name, "continuous collision through the stress coupling");
    s.gravity = [0.0; 3];
    s.materials.insert("steel".into(), elastic_steel());
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.bodies.push(slab_body());
    s
}

/// A 2 kg steel ball of 5 cm radius at `speed` m/s along +y, half a frame (60 Hz) before
/// the slab: it reaches it in the first frame.
fn ball(speed: f64, ccd: bool) -> ImpactorDesc {
    ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.05 },
        mass: 2.0,
        position: [0.03, SLAB_FRONT - 0.05 - speed / 60.0 * 0.5, 0.02],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
        ccd,
    }
}

/// The resident world of `scene` under `profile`, as the fracture lab makes it
/// (stress-web `load`): the profile's engine settings, impact islands, the GPU's frame.
fn resident(gpu: &Gpu, scene: &Scene, profile: Profile) -> GpuWorld {
    let mut scene = scene.clone();
    profile.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    e.apply_profile(profile);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(profile != Profile::Threshold);
    w.apply_profile(profile);
    if std::env::var("CCD_NO_ISLANDS").is_ok() {
        w.set_impact_islands(false);
    }
    if profile == Profile::BlastParityRouted {
        w.set_tiered(true).expect("tiered coupling");
    }
    w
}

fn run(gpu: &Gpu, w: &mut GpuWorld, frames: usize) {
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
}

const PROFILES: [Profile; 3] = [Profile::BlastParity, Profile::BlastParityRouted, Profile::Threshold];

#[test]
fn a_ccd_ball_never_passes_a_thin_slab() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    for profile in PROFILES {
        for speed in [60.0, 250.0, 600.0, 1200.0] {
            let mut s = realtime(base("ccd_slab"), 0.25);
            s.impactors.push(ball(speed, true));
            let mut w = resident(&gpu, &s, profile);
            run(&gpu, &mut w, 15);
            let y = w.solver.impactors[0].pose.position.y;
            eprintln!("{} {speed} m/s: ball at y {y:.3} ({} m/s)", profile.name(), w.solver.impactors[0].velocity.norm());
            assert!(y < SLAB_FRONT, "{} {speed} m/s: the CCD ball passed the slab (y {y:.3})", profile.name());
        }
    }
}

/// The control: without `ccd` the same ball passes the slab in the profiles' one step a
/// frame (the impact island of the routed profile catches it in the stress solve, so only
/// the unrouted profiles are checked).
#[test]
fn an_unmarked_ball_passes_the_slab() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    for profile in [Profile::BlastParity, Profile::Threshold] {
        let mut s = realtime(base("plain_slab"), 0.25);
        s.impactors.push(ball(250.0, false));
        let mut w = resident(&gpu, &s, profile);
        run(&gpu, &mut w, 15);
        let y = w.solver.impactors[0].pose.position.y;
        eprintln!("{} plain 250 m/s: ball at y {y:.3}", profile.name());
        assert!(y > SLAB_BACK, "{}: the control did not pass (y {y:.3}): the test cannot see tunnelling", profile.name());
    }
}

/// A free concrete wall (`ccd` as given) hit by a 2 t steel ram at 100 m/s, 0.4 m in front
/// of the slab: the wall (or its pieces) is driven into the slab at speed.
fn wall_into_slab(ccd: bool) -> Scene {
    let mut s = base("wall_into_slab");
    let chunks = grid(Vec3::new(-0.3, SLAB_FRONT - 0.6, -0.3), [6, 2, 6], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    let mut wall = body("wall", chunks, bonds);
    wall.ccd = ccd;
    s.bodies.push(wall);
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 2000.0,
        position: [0.0, SLAB_FRONT - 0.6 - 0.1 - 100.0 / 60.0 * 0.5, 0.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 100.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
        ccd: true,
    });
    realtime(s, 0.25)
}

/// Chunks of the wall (structure 1) beyond the slab's back face.
fn chunks_beyond(w: &GpuWorld) -> usize {
    let m = &w.solver.mirror;
    (0..m.structures[1].chunks.len()).filter(|&c| m.chunk_position(1, c).y > SLAB_BACK).count()
}

#[test]
fn a_ccd_structure_never_passes_a_thin_slab() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    for profile in [Profile::BlastParity, Profile::Threshold] {
        let mut marked = resident(&gpu, &wall_into_slab(true), profile);
        if std::env::var("CCD_DEBUG").is_ok() {
            for f in 0..15 {
                marked.step_frame(&gpu).expect("frame");
                marked.sync(&gpu);
                let m = &marked.solver.mirror;
                let ys: Vec<f64> = (0..m.structures[1].chunks.len()).map(|c| m.chunk_position(1, c).y).collect();
                let beyond = ys.iter().filter(|&&y| y > SLAB_BACK).count();
                let inside = ys.iter().filter(|&&y| y > SLAB_FRONT - 0.05 && y <= SLAB_BACK).count();
                let ram = &marked.solver.impactors[0];
                eprintln!("  frame {f}: clusters {} beyond {beyond} inside {inside} max y {:.3}; ram y {:.3} v {:.1}; resident {}", m.clusters.len(), ys.iter().cloned().fold(f64::MIN, f64::max), ram.pose.position.y, ram.velocity.norm(), marked.resident_frames);
            }
        } else {
            run(&gpu, &mut marked, 15);
        }
        let mut plain = resident(&gpu, &wall_into_slab(false), profile);
        run(&gpu, &mut plain, 15);
        if std::env::var("CCD_DEBUG").is_ok() {
            let m = &marked.solver.mirror;
            for cl in &m.clusters {
                if cl.structure != 1 {
                    continue;
                }
                let flags = marked.engine_body_flags(BodyKey::Cluster(cl.id)).unwrap_or(0);
                let ys: Vec<String> = cl.chunks.iter().map(|&c| format!("{:.2}", m.chunk_position(1, c).y)).collect();
                eprintln!("  cluster {} ccd {} v {:.1} anchored {} y [{}]", cl.id, flags & engine_schema::BODY_CCD != 0, cl.velocity.norm(), cl.anchored, ys.join(" "));
            }
        }
        let (on, off) = (chunks_beyond(&marked), chunks_beyond(&plain));
        eprintln!("{}: wall chunks beyond the slab: CCD {on}, plain {off} of 72; clusters {} / {}", profile.name(), marked.solver.mirror.clusters.len(), plain.solver.mirror.clusters.len());
        assert_eq!(on, 0, "{}: CCD wall chunks passed the slab", profile.name());
        assert!(off > 0, "{}: the plain wall did not pass: the test cannot see tunnelling", profile.name());
    }
}

/// A weak concrete beam (8 chunks of 0.1 m, `ccd` as given) fixed at its first chunk (an
/// anchored part), struck mid-span by a 50 kg steel ball at 10 m/s: it breaks, on the
/// GPU's split path, into free pieces and the anchored stub.
fn struck_beam(ccd: bool) -> Scene {
    let mut s = new_scene("struck_beam", "a weak beam broken by a ball");
    s.gravity = [0.0; 3];
    let mut m = oracle_concrete(None);
    m.tensile_strength = 2e4;
    m.cohesion = 2e4;
    m.compressive_strength = 2e5;
    s.materials.insert("weak".into(), m);
    s.materials.insert("steel".into(), elastic_steel());
    let mut chunks = grid(Vec3::new(-0.05, -0.05, 0.0), [8, 1, 1], Vec3::splat(0.1), "weak");
    chunks[0].support = Support::Fixed;
    let bonds = auto_bonds(&chunks, |_, _| "weak".into());
    let mut beam = body("beam", chunks, bonds);
    beam.ccd = ccd;
    s.bodies.push(beam);
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.1 },
        mass: 50.0,
        position: [0.5, -0.3, 0.05],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 10.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
        ccd: false,
    });
    realtime(s, 0.25)
}

#[test]
fn every_free_piece_of_a_ccd_structure_is_a_ccd_body() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    for profile in [Profile::BlastParity, Profile::Threshold] {
        for ccd in [true, false] {
            let mut w = resident(&gpu, &struck_beam(ccd), profile);
            run(&gpu, &mut w, 15);
            let m = &w.solver.mirror;
            assert!(m.clusters.len() > 1, "{}: nothing split: the test needs a split", profile.name());
            assert!(w.resident_frames > 0, "{}: no resident frame ran", profile.name());
            for cl in &m.clusters {
                let flags = w.engine_body_flags(BodyKey::Cluster(cl.id)).expect("the cluster's engine body");
                let is_ccd = flags & engine_schema::BODY_CCD != 0;
                eprintln!("{} ccd {ccd}: cluster {} ({} chunks, anchored {}): BODY_CCD {is_ccd}", profile.name(), cl.id, cl.chunks.len(), cl.anchored);
                assert_eq!(is_ccd, ccd && !cl.anchored, "{} ccd {ccd}: cluster {} (anchored {})", profile.name(), cl.id, cl.anchored);
            }
        }
    }
}

/// No readback for CCD: the resident frame's reads with a fast marked ball in the scene
/// are those of the same scene without it (the whole-world substeps read the fast bodies'
/// speeds back each frame).
#[test]
fn ccd_bodies_add_no_readback() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let reads = |ccd: bool| {
        let mut s = realtime(base("ccd_reads"), 0.25);
        let mut b = ball(250.0, ccd);
        // Far from the slab: the frames are the same either way.
        b.position[0] += 20.0;
        s.impactors.push(b);
        let mut w = resident(&gpu, &s, Profile::BlastParity);
        run(&gpu, &mut w, 15);
        (w.resident_frames, w.resident_reads())
    };
    let (marked, plain) = (reads(true), reads(false));
    eprintln!("resident frames and reads: CCD {marked:?}, plain {plain:?}");
    assert!(marked.0 > 0, "no resident frame ran");
    assert_eq!(marked, plain);
}
